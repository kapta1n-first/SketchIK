"""Keep a live task at an API boundary; resend only on an explicit local command."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import tempfile
import time


def process_identity(pid):
    """Protect the control command against a terminated process and reused PID."""
    value = Path(f'/proc/{pid}/stat').read_text()
    fields = value[value.rfind(')')+2:].split()
    if fields[0] == 'Z':
        raise ProcessLookupError('Task process has exited')
    return {'pid': pid, 'start_ticks': fields[19]}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_state(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


class RequestRecovery:
    def __init__(self, run, client, robot_state, emit, *, poll_s=.2):
        self.run, self.client, self.robot_state, self.emit = Path(run), client, robot_state, emit
        self.folder = self.run/'request_recovery'
        self.folder.mkdir(exist_ok=True)
        self.state_path = self.folder/'state.json'
        self.poll_s = poll_s
        self.pauses = self.resumes = 0
        self.pause_wall_s = 0.

    def measured_state(self):
        state = self.robot_state()
        return {key: state[key] for key in ('step', 'q', 'position', 'quaternion_xyzw',
            'finger_joint_positions', 'gripper_command') if key in state}

    def wait_for_resume(self, error, messages, tools):
        from .planning import save_planning_history
        self.pauses += 1
        pause = self.folder/f'pause_{self.pauses:04d}'
        pause.mkdir(exist_ok=False)
        before = self.measured_state()
        request_digest = digest({'messages': messages, 'tools': tools})
        save_planning_history(self.run, messages)
        state = {'status': 'waiting_for_operator', 'run': str(self.run.resolve()),
            **process_identity(os.getpid()), 'pause_id': pause.name,
            'nonce': secrets.token_hex(16), 'paused_at_unix': time.time(),
            'robot_state': before, 'source_step': before.get('step'),
            'pending_input_sha256': request_digest,
            'failed_request_sha256': self.client.last_request_metadata.get('body_sha256'),
            'message_count': len(messages), 'tool_count': len(tools or []),
            'error_type': type(error).__name__, 'http_status': error.response_metadata.get('http_status'),
            'automatic_retries': 0, 'operator_resumed_requests': self.resumes,
            'scope': 'Live process only. No environment reconstruction, camera motion or action replay.'}
        write_state(pause/'checkpoint.json', state)
        write_state(self.state_path, state)
        self.emit({'event': 'api_waiting_for_operator', 'source_step': before.get('step'),
            'pause_id': pause.name, 'state_file': str(self.state_path),
            'http_status': state['http_status'], 'automatic_retries': 0,
            'resume_command': f'python -m sketchik.request_recovery {self.run} resume'})
        command_file = pause/'command.json'
        heartbeat = time.monotonic()
        wait_started = heartbeat
        try:
            while True:
                if command_file.exists():
                    command = json.loads(command_file.read_text())
                    if command.get('nonce') != state['nonce'] or command.get('action') not in ('resume', 'abort'):
                        raise ValueError('Invalid task continuation command; no request sent')
                    command_file.rename(pause/'consumed_command.json')
                    if command['action'] == 'abort':
                        raise InterruptedError('Operator aborted the paused API request')
                    if self.measured_state() != before:
                        raise RuntimeError('Robot state changed while API was paused; no request sent')
                    if digest({'messages': messages, 'tools': tools}) != request_digest:
                        raise RuntimeError('Pending conversation changed while API was paused; no request sent')
                    self.resumes += 1
                    state.update(status='resumed', resumed_at_unix=time.time(),
                        operator_resumed_requests=self.resumes)
                    write_state(self.state_path, state)
                    write_state(pause/'outcome.json', state)
                    self.emit({'event': 'api_resumed_by_operator', 'pause_id': pause.name,
                        'source_step': before.get('step'), 'automatic_retries': 0})
                    return
                if time.monotonic()-heartbeat >= 30:
                    self.emit({'event': 'api_still_waiting_for_operator', 'pause_id': pause.name,
                        'source_step': before.get('step'), 'automatic_retries': 0})
                    heartbeat = time.monotonic()
                time.sleep(self.poll_s)
        except BaseException:
            state.update(status='aborted', ended_at_unix=time.time())
            write_state(self.state_path, state)
            write_state(pause/'outcome.json', state)
            raise
        finally:
            self.pause_wall_s += time.monotonic()-wait_started


def submit_command(run, action):
    run = Path(run).resolve()
    if action not in ('resume', 'abort'):
        raise ValueError('Expected resume or abort')
    state = json.loads((run/'request_recovery/state.json').read_text())
    if state.get('status') != 'waiting_for_operator' or state.get('run') != str(run):
        raise ValueError('This task is not waiting for an API continuation command')
    if process_identity(state['pid']) != {key: state[key] for key in ('pid', 'start_ticks')}:
        raise ProcessLookupError('Original live task process no longer exists; cannot restore its simulator')
    pause_id = state['pause_id']
    if '/' in pause_id or not pause_id.startswith('pause_'):
        raise ValueError('Invalid pause identifier')
    folder = run/'request_recovery'/pause_id
    checkpoint = json.loads((folder/'checkpoint.json').read_text())
    if checkpoint.get('nonce') != state.get('nonce'):
        raise ValueError('Task pause identity mismatch')
    with tempfile.NamedTemporaryFile(mode='w', dir=folder, prefix='.command_', delete=False) as stream:
        temporary = Path(stream.name)
        json.dump({'action': action, 'nonce': state['nonce'], 'requested_at_unix': time.time()}, stream)
    try:
        os.link(temporary, folder/'command.json')  # Atomic and exclusive; no half-written command is visible.
    finally:
        temporary.unlink()
    return {'command': action, 'pause_id': pause_id, 'source_step': state.get('source_step'),
        'status': 'submitted', 'automatic_retries': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('action', choices=('resume', 'abort'))
    args = parser.parse_args()
    print(json.dumps(submit_command(args.run, args.action)))


if __name__ == '__main__':
    main()
