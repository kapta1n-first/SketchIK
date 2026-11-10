"""Agent A grasp trial + read-only Agent B placement anticipation during execution."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image

from .agent import ChatClient, load_settings, transcript_copy
from .grasp_trial import run_selection, execute_selected, planning_dialogue, restore_planning_images, image_part, save_planning_history
from .perception import SavedObservationRobot
from .libero import back_project
from .tools import ToolHub, PERCEPTION_TOOLS, obj


TEXT = {'type': 'string', 'minLength': 1}
TEXTS = {'type': 'array', 'items': TEXT}
CANDIDATE = obj({'candidate_id': TEXT,
    'role': {'enum': ['placement_reference', 'support_reference', 'entry_reference', 'approach_reference']},
    'observation_id': TEXT, 'camera': {'enum': ['agentview', 'robot0_eye_in_hand']},
    'pixel_xy': {'type': 'array', 'items': {'type': 'number'}, 'minItems': 2, 'maxItems': 2},
    'description': TEXT}, ['candidate_id', 'role', 'observation_id', 'camera', 'pixel_xy', 'description'])
ANALYSIS_SCHEMA = obj({
    'source_step': {'type': 'integer', 'minimum': 0},
    'grasp_object': TEXT, 'destination': TEXT,
    'observed_evidence': TEXTS, 'placement_outline': TEXTS,
    'missing_information': TEXTS, 'post_grasp_checks': TEXTS,
    'ready_for_grounding': {'type': 'boolean'},
    'placement_candidates': {'type': 'array', 'items': CANDIDATE},
}, ['source_step', 'grasp_object', 'destination', 'observed_evidence',
    'placement_outline', 'missing_information', 'post_grasp_checks', 'ready_for_grounding', 'placement_candidates'])


def stamp():
    return {'utc': datetime.now(timezone.utc).isoformat(), 'monotonic_s': time.monotonic()}


def historical(value):
    """A frozen snapshot must never be advertised as the live robot state."""
    if isinstance(value, dict):
        return {k: False if k == 'current' else historical(v) for k, v in value.items()}
    if isinstance(value, list):
        return [historical(v) for v in value]
    return value


class HistoryHub(ToolHub):
    def state(self):
        return {**super().state(), 'historical_only': True, 'live_state_available': False}

    def call(self, name, arguments):
        result, images = super().call(name, arguments)
        result = {**historical(result), 'historical_only': True, 'live_state_available': False}
        for image in images:
            if image.get('type') == 'text':
                try:
                    value = json.loads(image['text'])
                except (ValueError, TypeError):
                    continue
                image['text'] = json.dumps(historical(value))
        return result, images


def restore_handoff(trial, run, emit, *, sam_request=None):
    """Restore the actual A history and IDs; no execution results are read."""
    cfg = json.loads((trial / 'config.json').read_text())
    source = Path(cfg['observation'])
    hub = HistoryHub(SavedObservationRobot(source), None, run,
                     lambda event: emit(historical(event)), allowed_tools=PERCEPTION_TOOLS,
                     sam_request=sam_request)
    shutil.copytree(trial / 'observations', run / 'observations')
    for folder in sorted((run / 'observations').iterdir()):
        meta = json.loads((folder / 'observation.json').read_text())
        step = meta['source_step']
        if not 0 <= step <= hub.robot.steps:
            raise ValueError('Historical observation is from after the final handoff state')
        frames = {}
        for camera in hub.robot.cameras:
            with np.load(folder / (camera + '.npz')) as data:
                frames[camera] = {'rgb': np.asarray(Image.open(folder / (camera + '.png')).convert('RGB')),
                    **{k: data[k].copy() for k in ('depth_m', 'K', 'world_from_camera')},
                    'camera': camera, 'step': step, 'convention': 'saved calibrated pregrasp frame'}
        hub.observations[folder.name] = frames
    for line in (trial / 'planning_events.jsonl').read_text().splitlines():
        event = json.loads(line)
        if event.get('event') != 'tool_result' or event.get('name') != 'segment' or not event.get('ok'):
            continue
        record = event['result']
        for candidate in record['candidates']:
            identity = candidate['candidate_id']
            mask_path = run / 'observations' / record['observation_id'] / 'masks' / (identity + '.png')
            hub.vision.candidates[identity] = {**candidate, 'number': int(identity.removeprefix('mask_')),
                'mask': np.asarray(Image.open(mask_path)) > 0, 'observation_id': record['observation_id'],
                'prompt': record['prompt']}
        hub.vision.segment_cache[(record['observation_id'], record['prompt'])] = {
            k: record[k] for k in ('candidate_ids', 'sam_client_wall_s', 'sam_inference_wall_s', 'sam_model_pid')}
    messages = restore_planning_images(json.loads((trial / 'planning_transcript.json').read_text()),
                                      [source, trial / 'observations', trial / 'planning_images'])
    manifest_file=trial/'history_manifest.json'
    if manifest_file.exists():
        manifest=json.loads(manifest_file.read_text())
        parts=[{'type':'text','text':'Additional historical frames automatically saved during earlier A observation motions. Each source_step is the actual capture time, not the current live state. All periodic frames are included.'}]
        for identity in manifest['periodic_observation_ids']:
            frames=hub.vision.frames(identity)
            for camera in hub.robot.cameras:
                frame=frames[camera]
                parts.extend([{'type':'text','text':json.dumps({'observation_id':identity,'source_step':frame['step'],
                    'camera':camera,'historical_only':True,'capture_reason':'automatic_periodic_history'})},
                    image_part(run/'observations'/identity/(camera+'.png'))])
        if manifest['periodic_observation_ids']:messages.append({'role':'user','content':parts})
    return hub, messages


class TimedClient:
    def __init__(self, client, emit):
        self.client, self.emit = client, emit
        self.round = 0

    def complete(self, messages, tools):
        self.round += 1
        self.emit({'event': 'b_api_started', 'round': self.round})
        try:
            result = self.client.complete(messages, tools)
        except Exception as exc:
            self.emit({'event': 'b_api_finished', 'round': self.round, 'error_type': type(exc).__name__})
            raise
        self.emit({'event': 'b_api_finished', 'round': self.round, 'usage': result.get('usage')})
        return result


def prepare_geometry(hub, analysis):
    """Bind B's historical pixel references to real depth; not a placement pose."""
    results, ids = [], set()
    for candidate in analysis['placement_candidates']:
        identity = candidate['candidate_id']
        if identity in ids:
            raise ValueError('Duplicate placement reference ID')
        ids.add(identity)
        record = {**candidate, 'historical_only': True, 'semantic_role_verified': False,
                  'release_authorized': False, 'eef_pose_computed': False}
        try:
            frames = hub.vision.frames(candidate['observation_id'])
            frame = frames[candidate['camera']]
            record.update(world_xyz=back_project(frame, candidate['pixel_xy']).tolist(),
                          source_step=frame['step'], status='observed_surface_reference',
                          calculation='programmatic back-projection using the same saved RGB-D/calibration')
        except (ValueError, KeyError) as exc:
            record.update(status='rejected', error=str(exc))
        results.append(record)
    return {'references': results, 'placement_pose_bound': False,
            'requires_current_grasp_state': True,
            'note': 'Reference points may later fill A sketch parameters; a point is not a support/free-space or release certificate.'}


def analyse(trial, run, env_file, max_turns=12, max_tool_calls=60):
    root = Path(__file__).resolve().parents[2]
    run.mkdir(exist_ok=False)
    def emit(event):
        with (run / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps({**event, **stamp()}, allow_nan=False) + '\n')
    messages = []
    emit({'event': 'b_started', 'source_trial': str(trial), 'robot_actions': 0})
    try:
        hub, messages = restore_handoff(trial, run, emit)
        system = (root / 'prompts/placement_anticipation.md').read_text()
        messages[0] = {'role': 'system', 'content': system}
        sketch = json.loads((trial / 'sketch.json').read_text())
        messages.append({'role': 'user', 'content': json.dumps({
            'phase': 'Agent B starts: all previous assistant messages came from Agent A.',
            'history_source_step': hub.robot.steps, 'historical_only': True,
            'live_grasp_result_available': False, 'planned_grasp': sketch,
            'request': 'Prepare possible placement/support/entry reference pixels from the historical images while the grasp chain proceeds. Perception tools are yours to use. Save references for later binding of A placement sketch, plus missing information; do not replace A task planning or assume grasp success.'})})
        settings = load_settings(env_file)
        client = ChatClient(settings.get('SKETCHIK_BASE_URL'), settings.get('SKETCHIK_MODEL'),
                            settings.get('SKETCHIK_API_KEY'), settings.get('SKETCHIK_REASONING'), timeout=120)
        tool = {'type': 'function', 'function': {'name': 'submit_placement_analysis',
                'description': 'Save provisional placement analysis from pregrasp history; no action or success certification.',
                'parameters': ANALYSIS_SCHEMA}}
        (run / 'system_prompt.md').write_text(system)
        (run / 'tools.json').write_text(json.dumps(hub.schemas + [tool], indent=2))
        (run / 'input.json').write_text(json.dumps(transcript_copy(messages), indent=2))
        (run / 'config.json').write_text(json.dumps({'provider': client.metadata,
            'source_trial': str(trial), 'history_source_step': hub.robot.steps,
            'max_turns': max_turns, 'max_tool_calls': max_tool_calls, 'robot_actions': 0}, indent=2))
        analysis, _, timing = planning_dialogue(TimedClient(client, emit), hub, messages, tool,
                                              max_turns=max_turns, max_tool_calls=max_tool_calls)
        geometry = prepare_geometry(hub, analysis)
        (run / 'prepared_geometry.json').write_text(json.dumps(geometry, indent=2))
        result = {'status': 'analysis_saved', 'analysis': analysis, 'prepared_geometry': geometry, 'timing': timing,
                  'robot_actions': 0, 'live_grasp_result_seen': False, 'executable_placement': False}
        (run / 'result.json').write_text(json.dumps(result, indent=2))
        emit({'event': 'b_completed', 'status': result['status'], 'robot_actions': 0})
    except Exception as exc:
        (run / 'error.json').write_text(json.dumps({'error_type': type(exc).__name__,
            'error': str(exc), 'robot_actions': 0, 'automatic_retry': False}, indent=2))
        emit({'event': 'b_failed', 'error_type': type(exc).__name__})
        raise
    finally:
        save_planning_history(run,messages)
        (run/'transcript.json').write_text(json.dumps(transcript_copy(messages),indent=2))


def overlap_report(execution, b_run):
    """Measure actual API/physical-stepping interval overlap, never hypothetical speedup."""
    events = [json.loads(v) for v in (execution / 'events.jsonl').read_text().splitlines()]
    physics = [datetime.fromisoformat(v['utc']).timestamp() for v in events if v['event'] == 'physics_step']
    b_events = [json.loads(v) for v in (b_run / 'events.jsonl').read_text().splitlines()]
    starts, overlap = {}, 0.
    for event in b_events:
        instant = datetime.fromisoformat(event['utc']).timestamp()
        if event['event'] == 'b_api_started':
            starts[event['round']] = instant
        elif event['event'] == 'b_api_finished' and event['round'] in starts and physics:
            overlap += max(0., min(instant, physics[-1]) - max(starts[event['round']], physics[0]))
    return {'physical_actions_recorded': len(physics),
            'physical_stepping_window_s': physics[-1] - physics[0] if physics else 0.,
            'b_api_overlap_with_physical_stepping_window_s': overlap,
            'b_api_physically_overlapped': overlap > 0,
            'note': 'Window includes rendering/control between steps. Not saved-time or placement-success evidence.'}


def run_trial(args):
    root = Path(__file__).resolve().parents[2]
    run = root / 'runs' / (datetime.now().strftime('%Y%m%dT%H%M%S') + '_two_agent_trial')
    run.mkdir(exist_ok=False)
    def emit(event):
        record = {**event, **stamp()}
        with (run / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')
        print(json.dumps(record), flush=True)
    trial, execution, child, output = None, None, None, None
    try:
        emit({'event': 'a_started'})
        trial=run/'agent_a'
        command=[sys.executable,'-m','sketchik.grasp_trial','observe-plan',
            '--source-observation',str(args.observation.resolve()),'--output-run',str(trial),
            '--env-file',str(args.env_file.resolve()),'--gpu-uuid',args.gpu_uuid,
            '--history-interval-s',str(args.history_interval_s),'--max-turns',str(args.max_turns),
            '--max-tool-calls',str(args.max_tool_calls)]
        for pid in args.peer_pid:command+=['--peer-pid',str(pid)]
        with (run/'agent_a_console.log').open('x') as stream:
            result=subprocess.run(command,cwd=root,env={**os.environ,'PYTHONPATH':str(root/'src')},
                                  stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode or not (trial/'sketch.json').exists():
            raise RuntimeError('A observation/planning failed; see agent_a_console.log and saved history')
        (run / 'agent_a.json').write_text(json.dumps({'trial': str(trial)}, indent=2))
        emit({'event': 'a_sketch_saved', 'trial': str(trial)})
        output = (run / 'agent_b_console.log').open('w')
        child = subprocess.Popen([sys.executable, '-m', 'sketchik.anticipate_trial', 'analyse',
            '--trial', str(trial), '--output', str(run / 'agent_b'), '--env-file', str(args.env_file.resolve()),
            '--max-turns', str(args.max_turns), '--max-tool-calls', str(args.max_tool_calls)],
            cwd=root, env={**os.environ, 'PYTHONPATH': str(root / 'src')}, stdout=output, stderr=subprocess.STDOUT)
        emit({'event': 'b_launched_during_grasp_pipeline', 'pid': child.pid})
        run_selection(trial, args.gpu_uuid, args.peer_pid)
        selected = json.loads((trial / 'selection_result.json').read_text())
        if selected['candidate_id'] is None:
            (run / 'result.json').write_text(json.dumps({'status': 'no_grasp_selected', 'agent_a': str(trial),
                'agent_b_started': True, 'robot_actions': 0}, indent=2))
            return run
        emit({'event': 'grasp_selected', 'candidate_id': selected['candidate_id']})
        def execution_ready(execution_run):
            nonlocal execution
            execution = execution_run
            (run / 'execution.json').write_text(json.dumps({'run': str(execution)}, indent=2))
            emit({'event': 'execution_ready_after_reset_match', 'execution': str(execution)})
        execute_selected(trial, args.gpu_uuid, args.peer_pid, args.lift_test_m, on_execution_ready=execution_ready)
        emit({'event': 'grasp_execution_finished', 'execution': str(execution)})
    except Exception as exc:
        (run / 'error.json').write_text(json.dumps({'error_type': type(exc).__name__, 'error': str(exc),
            'agent_a': str(trial) if trial else None, 'execution': str(execution) if execution else None}, indent=2))
        emit({'event': 'trial_failed', 'error_type': type(exc).__name__})
        raise
    finally:
        if child is not None:
            code = child.wait()
            output.close()
            emit({'event': 'b_process_finished', 'exit_code': code})
            previous = json.loads((run / 'result.json').read_text()) if (run / 'result.json').exists() else {}
            report = {**previous, 'agent_a': str(trial), 'execution': str(execution) if execution else None, 'agent_b': str(run / 'agent_b'),
                'agent_b_exit_code': code, 'placement_executed': False}
            if execution and (execution / 'execution_summary.json').exists():
                summary = json.loads((execution / 'execution_summary.json').read_text())
                report.update(grasp_commands_completed=summary['all_commands_completed'],
                              execution_error=summary['error'], grasp_actions=summary['grasp_actions'])
            if (run / 'error.json').exists():
                report['trial_error_file'] = str(run / 'error.json')
            if execution and (execution / 'events.jsonl').exists() and (run / 'agent_b/events.jsonl').exists():
                report.update(overlap_report(execution, run / 'agent_b'))
            (run / 'result.json').write_text(json.dumps(report, indent=2))
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('run')
    p.add_argument('--observation', type=Path, required=True)
    p.add_argument('--gpu-uuid', required=True)
    p.add_argument('--peer-pid', type=int, action='append', default=[])
    p.add_argument('--lift-test-m', type=float, default=0.)
    p.add_argument('--history-interval-s', type=float, default=1.)
    b = sub.add_parser('analyse')
    b.add_argument('--trial', type=Path, required=True)
    b.add_argument('--output', type=Path, required=True)
    for item in (p, b):
        item.add_argument('--env-file', type=Path, required=True)
        item.add_argument('--max-turns', type=int, default=12)
        item.add_argument('--max-tool-calls', type=int, default=60)
    args = parser.parse_args()
    if args.command == 'analyse':
        analyse(args.trial.resolve(), args.output.resolve(), args.env_file, args.max_turns, args.max_tool_calls)
    else:
        print(json.dumps({'run': str(run_trial(args))}), flush=True)


if __name__ == '__main__':
    main()
