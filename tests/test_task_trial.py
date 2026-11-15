import json
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
from sketchik.geometry_journal import GeometryJournal
from sketchik.task_trial import write_dimensions

def test_acquisition_only_request_is_historical_not_a_persistent_system_rule(tmp_path, monkeypatch):
    from sketchik.task_trial import table_observation
    from sketchik.stage_context import StageContext, EPISODE_INSTRUCTIONS

    catalog = {'components': {'G1': {'sources': []}}, 'views': [{'image_file': 'mock.png'}]}
    handoff = {'objects': [{'object_id': 'object', 'component_ids': ['G1'], 'required_surfaces': []}]}
    batch = {'status': 'completed', 'robot_actions': 0, 'observation_ids': [],
        'wall_s': 0., 'captured_views': 0}
    (tmp_path/'scene_catalog').mkdir()
    hub = SimpleNamespace(run=tmp_path, robot=SimpleNamespace(steps=30))
    messages = [{'role': 'system', 'content': EPISODE_INSTRUCTIONS+'\nPersistent execution instructions'}]
    monkeypatch.setattr('sketchik.table_scan.run_table_scan', lambda _: ({'wall_s': 0., 'captured_views': 4}, []))
    monkeypatch.setattr('sketchik.task_trial.reset_observation_arm', lambda _: ({'wall_s': 0.}, []))
    monkeypatch.setattr('sketchik.grounding_packet.prepare_table_context', lambda _: {'measured': True})
    monkeypatch.setattr('sketchik.scene_catalog.build_scene_catalog', lambda _: catalog)
    monkeypatch.setattr('sketchik.scene_catalog.public_catalog', lambda _: {})
    monkeypatch.setattr('sketchik.scene_catalog.binding_schema', lambda _: {})
    monkeypatch.setattr('sketchik.scene_catalog.bind_requirements', lambda *_: deepcopy(handoff))
    monkeypatch.setattr('sketchik.guided_observation.run_guided_observation', lambda *_: (batch, []))
    monkeypatch.setattr('sketchik.task_trial.image_part', lambda _: {'type': 'text', 'text': 'mock image'})
    monkeypatch.setattr('sketchik.task_trial.compact_view_feedback', lambda _: {})
    def plan(client, hub, source, tool, **kwargs):
        assert tool['function']['name'] == 'submit_scene_requirements'
        request = json.loads(source[1]['content'])
        assert source[1]['role'] == 'user' and request['phase'] == 'initial_acquisition_request'
        assert 'submit_scene_requirements' in request['instruction']
        source.extend([{'role': 'assistant', 'content': None, 'tool_calls': [
            {'id': 'acquire', 'type': 'function', 'function': {'name': 'submit_scene_requirements', 'arguments': '{}'}}]},
            {'role': 'tool', 'tool_call_id': 'acquire', 'content': '{"saved":true}'}])
        return {'source_step': 30}, {}, {}
    monkeypatch.setattr('sketchik.task_trial.planning_dialogue', plan)
    table_observation(None, hub, messages, max_turns=1, max_tool_calls=1)
    original = deepcopy(messages)
    active, _ = StageContext(retain_episode_references=True).apply(messages)
    assert messages == original and active == messages
    assert [m['content'] for m in active if m['role'] == 'system'] == [messages[0]['content']]
    assert 'initial acquisition-only request is now historical' in active[-1]['content'][0]['text']
    assert 'current execution phase' in active[-1]['content'][0]['text']


def test_dimensions_keeps_unknown_cavity_separate(tmp_path):
    journal = GeometryJournal(tmp_path)
    journal.update_object('microwave', {'geometry_status': 'observed_partial', 'source_step': 30,
        'observed_geometry': {'observed_span_xyz_m': [.4,.3,.2], 'observed_principal_spans_m': [.4,.3,.2],
            'observed_principal_axes_world':[[1,0,0],[0,1,0],[0,0,1]],'principal_axis_gap_ratios':[.4,.4,.6]}})
    data = write_dimensions(SimpleNamespace(run=tmp_path, journal=journal), [{'object_id': 'mug'}])
    assert data['objects']['microwave']['observed_span_xyz_m'] == [.4,.3,.2]
    assert data['objects']['microwave']['observed_principal_axes_world']==[[1,0,0],[0,1,0],[0,0,1]]
    assert data['objects']['microwave']['principal_axis_gap_ratios']==[.4,.4,.6]
    assert data['objects']['microwave']['usable_cavity_dimensions_m'] is None
    assert data['objects']['mug']['observed_span_xyz_m'] is None
    assert json.loads((tmp_path / 'dimensions.json').read_text()) == data


def test_video_rgb_render_preserves_native_pixels_without_depth_request():
    from sketchik.libero import LiberoRobot
    native=np.arange(18,dtype=np.uint8).reshape(2,3,3)
    calls=[]
    def render(**kwargs):
        calls.append(kwargs)
        assert kwargs['depth'] is False
        return native
    robot=LiberoRobot.__new__(LiberoRobot)
    robot.cameras=('agentview','robot0_eye_in_hand');robot.size=1024
    robot.env=SimpleNamespace(sim=SimpleNamespace(render=render))
    np.testing.assert_array_equal(robot.camera_rgb('agentview'),native[::-1])
    assert calls==[{'camera_name':'agentview','width':1024,'height':1024,'depth':False}]
    with pytest.raises(ValueError,match='Unknown calibrated camera'):
        robot.camera_rgb('unknown')
    assert len(calls)==1



@pytest.mark.parametrize('error_type', [OSError, TypeError, RuntimeError])
def test_unexpected_scan_error_prevents_reset_and_first_model_call(tmp_path, monkeypatch, error_type):
    from sketchik.task_trial import table_observation
    error = error_type('injected unexpected scan fault')
    events = []
    hub = SimpleNamespace(run=tmp_path, robot=SimpleNamespace(steps=30))
    def scan(_):
        events.append('scan')
        raise error
    def forbidden(*args, **kwargs):
        pytest.fail('No reset, model request or further operation after a program fault')
    monkeypatch.setattr('sketchik.table_scan.run_table_scan', scan)
    monkeypatch.setattr('sketchik.task_trial.reset_observation_arm', forbidden)
    monkeypatch.setattr('sketchik.task_trial.planning_dialogue', forbidden)
    with pytest.raises(error_type) as caught:
        table_observation(None, hub, [], max_turns=1, max_tool_calls=1)
    assert caught.value is error and hub.robot.steps == 30 and events == ['scan']
