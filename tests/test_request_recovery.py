"""Local HTTP and live-process tests; no model service or simulator GT."""
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
import threading
import time

import pytest

from sketchik.agent import ChatClient, ProviderError, RejectedResponse
from sketchik.request_recovery import RequestRecovery, submit_command


def wait_until(predicate):
    deadline = time.monotonic()+5
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.01)
    raise AssertionError('Timed out waiting for the local test checkpoint')


def setup_recovery(tmp_path, client):
    state = {'step': 93, 'q': [0.]*7, 'position': [0., 0., 1.],
        'quaternion_xyzw': [0., 0., 0., 1.], 'gripper_command': 1.}
    events = []
    recovery = RequestRecovery(tmp_path, client, lambda: deepcopy(state), events.append, poll_s=.01)
    client.request_pause_handler = recovery.wait_for_resume
    return recovery, state, events


def worker_for(client, messages, tools):
    outcomes = []
    def work():
        try:
            outcomes.append(client.complete(messages, tools))
        except BaseException as exc:
            outcomes.append(exc)
    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    return worker, outcomes


def test_http_failure_waits_without_retry_then_resends_same_request_on_explicit_resume(tmp_path):
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_POST(self):
            requests.append(self.rfile.read(int(self.headers['Content-Length'])))
            if len(requests) == 1:
                self.send_response(503); self.end_headers()
                self.wfile.write(b'{"error":{"code":"rate_limit_exceeded"}}')
            else:
                self.send_response(200); self.end_headers()
                self.wfile.write(json.dumps({'choices': [{'finish_reason': 'stop',
                    'message': {'role': 'assistant', 'content': 'ready'}}]}).encode())
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    client = ChatClient(f'http://127.0.0.1:{server.server_port}/v1', 'fixture', 'private-test-key',
        transport_dir=tmp_path/'a_transport')
    recovery, state, events = setup_recovery(tmp_path, client)
    original = deepcopy(state)
    messages = [{'role': 'user', 'content': 'Original task; already completed prefix remains in history.'}]
    tools = [{'type': 'function', 'function': {'name': 'probe', 'parameters': {'type': 'object'}}}]
    worker, outcomes = worker_for(client, messages, tools)
    try:
        wait_until(recovery.state_path.exists)
        time.sleep(.08)
        assert len(requests) == 1 and not outcomes and state == original and worker.is_alive()
        checkpoint = json.loads(recovery.state_path.read_text())
        assert checkpoint['status'] == 'waiting_for_operator' and checkpoint['http_status'] == 503
        assert 'private-test-key' not in recovery.state_path.read_text()
        assert json.loads((tmp_path/'planning_transcript.json').read_text()) == messages
        command = submit_command(tmp_path, 'resume')
        assert command['automatic_retries'] == 0
        worker.join(5)
        assert not worker.is_alive() and outcomes[0]['message']['content'] == 'ready'
        assert len(requests) == 2 and requests[0] == requests[1]
        assert state == original and client.operator_resumed_requests == 1
        assert json.loads(recovery.state_path.read_text())['status'] == 'resumed'
        with pytest.raises(ValueError, match='not waiting'):
            submit_command(tmp_path, 'resume')
    finally:
        if worker.is_alive():
            submit_command(tmp_path, 'abort'); worker.join(5)
        server.shutdown(); server.server_close(); thread.join(5)


@pytest.mark.parametrize('action,changed', [('abort', False), ('resume', True)])
def test_abort_or_changed_state_cannot_resend_request(tmp_path, action, changed):
    client = ChatClient('http://127.0.0.1:1/v1', 'fixture', 'private-test-key')
    class Opener:
        calls = 0
        def open(self, request, timeout):
            self.calls += 1
            raise OSError('fixture connection lost')
    opener = client.opener = Opener()
    recovery, state, events = setup_recovery(tmp_path, client)
    worker, outcomes = worker_for(client, [{'role': 'user', 'content': 'same task'}], [])
    wait_until(recovery.state_path.exists)
    if changed: state['step'] += 1
    submit_command(tmp_path, action); worker.join(5)
    assert not worker.is_alive() and opener.calls == 1
    assert isinstance(outcomes[0], RuntimeError if changed else InterruptedError)
    assert json.loads(recovery.state_path.read_text())['status'] == 'aborted'


def test_stale_process_or_changed_input_cannot_resume(tmp_path):
    client = ChatClient('http://127.0.0.1:1/v1', 'fixture', 'private-test-key')
    recovery, state, events = setup_recovery(tmp_path, client)
    client._complete_once = lambda *a: (_ for _ in ()).throw(ProviderError('fixture'))
    messages = [{'role': 'user', 'content': 'fixed'}]
    worker, outcomes = worker_for(client, messages, [])
    wait_until(recovery.state_path.exists)
    p = recovery.state_path; checkpoint = json.loads(p.read_text())
    wrong = {**checkpoint, 'start_ticks': 'wrong'}; p.write_text(json.dumps(wrong))
    with pytest.raises(ProcessLookupError): submit_command(tmp_path, 'resume')
    p.write_text(json.dumps(checkpoint)); messages[0]['content'] = 'changed'
    submit_command(tmp_path, 'resume'); worker.join(5)
    assert not worker.is_alive() and isinstance(outcomes[0], RuntimeError)
    assert 'conversation changed' in str(outcomes[0])


def test_completed_length_refusal_does_not_enter_connection_recovery(tmp_path):
    client = ChatClient('http://127.0.0.1:1/v1', 'fixture', 'private-test-key')
    recovery, state, events = setup_recovery(tmp_path, client)
    client._complete_once = lambda *a: (_ for _ in ()).throw(RejectedResponse('length'))
    with pytest.raises(RejectedResponse): client.complete([], [])
    assert not recovery.state_path.exists() and not events


def test_partial_stream_tool_call_is_not_exposed_before_manual_resume(tmp_path):
    client = ChatClient('http://127.0.0.1:1/v1', 'fixture', 'private-test-key', stream=True)
    recovery, state, events = setup_recovery(tmp_path, client)
    chunk = {'choices': [{'index': 0, 'delta': {'tool_calls': [{'index': 0, 'id': 'incomplete',
        'function': {'name': 'execute_actions', 'arguments': '{"actions":['}}]}, 'finish_reason': None}]}
    class Opener:
        calls = 0
        def open(self, request, timeout):
            self.calls += 1
            return BytesIO(b'data: '+json.dumps(chunk).encode()+b'\n\ndata: [DONE]\n\n')
    opener = client.opener = Opener()
    worker, outcomes = worker_for(client, [], [])
    wait_until(recovery.state_path.exists)
    assert not outcomes and opener.calls == 1
    submit_command(tmp_path, 'abort'); worker.join(5)
    assert isinstance(outcomes[0], InterruptedError) and opener.calls == 1
