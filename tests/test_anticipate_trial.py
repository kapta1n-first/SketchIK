import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
from jsonschema import Draft202012Validator

from sketchik.agent import ProviderError, transcript_copy
from sketchik.anticipate_trial import ANALYSIS_SCHEMA, restore_handoff, overlap_report, prepare_geometry
from sketchik.grasp_trial import SKETCH_SCHEMA, planning_dialogue, planning_instructions
from sketchik.tools import ToolHub, PERCEPTION_TOOLS
from test_agent import FakeRobot, call
from test_grasp_trial import sketch
from test_perception_tools import candidate


def handoff(tmp_path):
    robot = FakeRobot()
    source = tmp_path / 'source'; source.mkdir()
    (source / 'initial_state.json').write_text(json.dumps(robot.state()))
    (source / 'metadata.json').write_text(json.dumps(robot.metadata))
    for c in robot.cameras:
        f = robot.camera(c)
        Image.fromarray(f['rgb']).save(source / (c + '.png'))
        np.savez(source / (c + '.npz'), **{k: f[k] for k in ('depth_m', 'K', 'world_from_camera')})
    trial = tmp_path / 'a'; trial.mkdir()
    (trial / 'config.json').write_text(json.dumps({'observation': str(source)}))
    def emit(event):
        with (trial / 'planning_events.jsonl').open('a') as stream:
            stream.write(json.dumps(event) + '\n')
    def sam(*args):
        return {'images': [{'camera': c, 'candidates': [candidate(0, .8)]} for c in robot.cameras]}
    hub = ToolHub(robot, None, trial, emit, sam_request=sam, allowed_tools=PERCEPTION_TOOLS)
    initial, pictures = hub.observe()
    tool = {'type': 'function', 'function': {'name': 'submit_grasp_sketch', 'parameters': SKETCH_SCHEMA}}
    class A:
        turn = 0
        def complete(self, messages, tools):
            self.turn += 1
            item = (call('segment', {'observation_id': initial['observation_id'], 'prompt': 'target'}, 'a1')
                    if self.turn == 1 else call('submit_grasp_sketch', sketch(), 'a2'))
            return {'message': {'role': 'assistant', 'tool_calls': [item]}}
    messages = [{'role': 'system', 'content': 'A'}, {'role': 'user', 'content': pictures}]
    planning_dialogue(A(), hub, messages, tool)
    (trial / 'planning_transcript.json').write_text(json.dumps(transcript_copy(messages)))
    return trial, messages


def test_full_history_images_and_mask_ids_are_preserved_without_live_state(tmp_path):
    trial, original = handoff(tmp_path)
    run = tmp_path / 'b'; run.mkdir()
    hub, messages = restore_handoff(trial, run, lambda event: None,
        sam_request=lambda *args: pytest.fail('handed-off exact SAM query should use cache'))
    assert messages == original
    assert 'official_success' not in json.dumps(messages)
    result, _ = hub.call('segment', {'observation_id': 'obs_00000', 'prompt': 'target'})
    assert result['ok'] and result['result']['cached']
    assert result['historical_only'] and result['result']['current'] is False
    assert set(hub.vision.candidates) == {'mask_00001', 'mask_00002'}
    viewed, pictures = hub.call('look', {'observation_id': 'obs_00000', 'candidate_ids': ['mask_00001']})
    assert viewed['ok'] and viewed['result']['current'] is False
    for part in pictures:
        if part['type'] == 'text':
            assert json.loads(part['text'])['current'] is False
    state, _ = hub.call('get_state', {})
    assert state['result']['live_state_available'] is False
    point, _ = hub.call('back_project', {'observation_id': 'obs_00000', 'camera': 'agentview', 'pixel_xy': [4, 5]})
    assert point['result']['world_xyz'] == [4, 5, 1] and point['result']['current'] is False
    denied, _ = hub.call('move_to', {'source_step': 0, 'params': {'position': [0, 0, 1]}})
    assert not denied['ok'] and hub.robot.steps == 0


def test_b_can_inspect_then_submit_analysis_and_cannot_dispatch_motion(tmp_path):
    trial, _ = handoff(tmp_path)
    run = tmp_path / 'b'; run.mkdir()
    hub, messages = restore_handoff(trial, run, lambda event: None)
    tool = {'type': 'function', 'function': {'name': 'submit_placement_analysis', 'parameters': ANALYSIS_SCHEMA}}
    analysis = {'source_step': 0, 'grasp_object': 'target', 'destination': 'container',
        'observed_evidence': ['visible rim'], 'placement_outline': ['obtain interior observation'],
        'missing_information': ['support surface'], 'post_grasp_checks': ['actual held orientation'],
        'ready_for_grounding': False, 'placement_candidates': []}
    Draft202012Validator.check_schema(ANALYSIS_SCHEMA)
    class B:
        turn = 0
        def complete(self, messages, tools):
            self.turn += 1
            names = {v['function']['name'] for v in tools}
            assert not {'move_to', 'execute_plan', 'measure', 'depth', 'submit_grasp_sketch'} & names
            item = (call('get_state', {}, 'b1') if self.turn == 1
                    else call('submit_placement_analysis', analysis, 'b2'))
            return {'message': {'role': 'assistant', 'tool_calls': [item]}}
    actual, _, timing = planning_dialogue(B(), hub, messages, tool)
    assert actual == analysis and timing['rounds'] == 2
    class BadB:
        def complete(self, *args):
            return {'message': {'role': 'assistant', 'tool_calls': [
                call('move_to', {'source_step': 0, 'params': {'position': [0, 0, 1]}}, 'bad')]}}
    with pytest.raises(ProviderError, match='motion is not exposed'):
        planning_dialogue(BadB(), hub, messages, tool)
    assert hub.robot.steps == 0


def test_overlap_requires_actual_physics_and_api_intervals(tmp_path):
    execution = tmp_path / 'execute'; execution.mkdir()
    b = tmp_path / 'b'; b.mkdir()
    def event(name, seconds, **kwargs):
        return {'event': name, 'utc': f'2026-11-10T00:00:{seconds:02d}+00:00', **kwargs}
    physics = [event('physics_step', i) for i in (2, 5, 8)]
    api = [event('b_api_started', 0, round=1), event('b_api_finished', 3, round=1),
           event('b_api_started', 6, round=2), event('b_api_finished', 10, round=2)]
    (execution / 'events.jsonl').write_text('\n'.join(map(json.dumps, physics)))
    (b / 'events.jsonl').write_text('\n'.join(map(json.dumps, api)))
    result = overlap_report(execution, b)
    assert result['physical_actions_recorded'] == 3
    assert result['b_api_overlap_with_physical_stepping_window_s'] == 3
    (execution / 'events.jsonl').write_text('')
    assert overlap_report(execution, b)['b_api_physically_overlapped'] is False


def test_symbolic_only_a_has_no_perception_tools_or_conflicting_instructions(tmp_path):
    guide = (Path(__file__).resolve().parents[1] / 'prompts/grasp_trial.md').read_text()
    system = planning_instructions(guide, allow_perception=False)
    assert '本轮开放submit_grasp_sketch及六个只读感知工具' not in system
    assert '## 可用感知工具' not in system
    assert '本轮只开放submit_grasp_sketch' in system
    assert '常驻Molmo选择ID或NONE' in system
    assert planning_instructions(guide) == guide
    trial, _ = handoff(tmp_path)
    source = Path(json.loads((trial/'config.json').read_text())['observation'])
    from sketchik.perception import SavedObservationRobot
    run = tmp_path / 'a_only'; run.mkdir()
    hub = ToolHub(SavedObservationRobot(source), None, run, lambda event: None, allowed_tools=frozenset())
    description, pictures = hub.observe()  # Program records the actual images, no model tool call.
    assert not hub.schemas and len(pictures) > 1
    class A:
        def complete(self, messages, tools):
            assert [t['function']['name'] for t in tools] == ['submit_grasp_sketch']
            return {'message': {'role': 'assistant', 'tool_calls': [call('submit_grasp_sketch', sketch())]}}
    result, _, _ = planning_dialogue(A(), hub, [{'role':'system','content':system},
        {'role':'user','content':pictures}], {'type':'function','function':{'name':'submit_grasp_sketch','parameters':SKETCH_SCHEMA}})
    assert result == sketch() and description['static_observation']


def test_b_reference_xyz_comes_from_real_saved_depth_and_is_not_a_release_pose(tmp_path):
    trial, _ = handoff(tmp_path)
    run = tmp_path / 'b'; run.mkdir()
    hub, _ = restore_handoff(trial, run, lambda event: None)
    point = {'candidate_id':'support1','role':'support_reference','observation_id':'obs_00000',
        'camera':'agentview','pixel_xy':[4,5],'description':'possible visible support, semantics unverified'}
    result = prepare_geometry(hub, {'placement_candidates':[point,{**point,'candidate_id':'outside','pixel_xy':[99,99]}]})
    good, bad = result['references']
    assert good['world_xyz'] == [4,5,1] and good['source_step'] == 0
    assert not good['semantic_role_verified'] and not good['release_authorized'] and not good['eef_pose_computed']
    assert bad['status'] == 'rejected' and 'world_xyz' not in bad
    assert result['requires_current_grasp_state'] and not result['placement_pose_bound']
    assert prepare_geometry(hub, {'placement_candidates':[]})['references'] == []
    with pytest.raises(ValueError, match='Duplicate'):
        prepare_geometry(hub, {'placement_candidates':[point, point]})
