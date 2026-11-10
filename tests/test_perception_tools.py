import base64
from io import BytesIO
import json

import numpy as np
from PIL import Image
import pytest

from sketchik.agent import transcript_copy
from sketchik.grasp_trial import SKETCH_SCHEMA, planning_dialogue, restore_planning_images
from sketchik.perception import SavedObservationRobot, image_part
from sketchik.tools import ToolHub, PERCEPTION_TOOLS, tool_schemas
from test_agent import FakeRobot, FakeExecutor, call
from test_grasp_trial import sketch


def candidate(number, score, shape=(16, 16)):
    mask = np.zeros(shape, dtype=bool)
    mask[3:10, 2:9] = True
    return {'id': number, 'score': score, 'box_xyxy': [2, 3, 9, 10],
            'mask_shape': list(shape),
            'mask_packbits_base64': base64.b64encode(np.packbits(mask.ravel())).decode()}


@pytest.fixture
def hub(tmp_path):
    robot = FakeRobot()
    requests = []
    def sam(url, payload):
        requests.append((url, payload))
        return {'images': [{'camera': c, 'candidates': [candidate(0, .99), candidate(1, .5)]}
                           for c in reversed(robot.cameras)], 'inference_wall_s': .1, 'model_pid': 123}
    hub = ToolHub(robot, FakeExecutor(robot), tmp_path, lambda x: None, sam_request=sam)
    hub.requests = requests
    hub.observe()
    return hub


def test_segment_keeps_every_candidate_caches_exact_frame_and_preserves_prompt(hub):
    args = {'observation_id': 'obs_00000', 'prompt': 'visible handle of the yellow mug'}
    one, pictures = hub.call('segment', args)
    assert one['ok'] and len(one['result']['candidates']) == 4
    assert {c['score'] for c in one['result']['candidates']} == {.99, .5}
    assert sum(p['type'] == 'image_url' for p in pictures) == 2
    assert hub.requests[0][1]['prompt'] == args['prompt']
    assert [i['camera'] for i in hub.requests[0][1]['images']] == list(hub.robot.cameras)
    two, _ = hub.call('segment', args)
    assert two['result']['cached'] and len(hub.requests) == 1
    hub.robot.steps = 12
    old, _ = hub.call('segment', args)
    assert old['result']['source_step'] == 0 and old['result']['current'] is False
    hub.observe()
    fresh, _ = hub.call('segment', {**args, 'observation_id': 'obs_00001'})
    assert fresh['ok'] and not fresh['result']['cached'] and len(hub.requests) == 2
    assert hub.executor.calls == 0 and hub.robot.steps == 12


def test_segment_empty_is_normal_and_bad_camera_or_resolution_is_rejected(hub):
    hub.vision.request = lambda *a: {'images': [{'camera': c, 'candidates': []} for c in hub.robot.cameras]}
    result, _ = hub.call('segment', {'observation_id': 'obs_00000', 'prompt': 'unobserved target'})
    assert result['ok'] and not result['result']['found']
    for views in [
        [{'camera': 'agentview', 'candidates': []}] * 2,
        [{'camera': c, 'candidates': [candidate(0, .8, (8, 8))]} for c in hub.robot.cameras],
    ]:
        hub.vision.request = lambda *a, views=views: {'images': views}
        result, _ = hub.call('segment', {'observation_id': 'obs_00000', 'prompt': 'invalid reply'})
        assert not result['ok'] and not hub.vision.candidates


def test_look_crop_zoom_mapping_and_history_never_recaptures(hub, monkeypatch):
    def no_capture(camera):
        raise AssertionError('saved look must not render a new image')
    monkeypatch.setattr(hub.robot, 'camera', no_capture)
    hub.robot.steps = 9
    result, pictures = hub.call('look', {'observation_id': 'obs_00000', 'camera': 'agentview',
        'grid': True, 'crop_xyxy': [2, 3, 10, 12], 'zoom': 2})
    assert result['ok'] and result['result']['source_step'] == 0 and not result['result']['current']
    view = result['result']['views'][0]
    assert view['display_size'] == [16, 18] and view['crop_xyxy'] == [2, 3, 10, 12]
    raw = next(p['image_url']['url'] for p in pictures if p['type'] == 'image_url')
    assert Image.open(BytesIO(base64.b64decode(raw.split(',')[1]))).size == (16, 18)
    point, _ = hub.call('back_project', {'observation_id': 'obs_00000', 'camera': 'agentview', 'pixel_xy': [4, 5]})
    assert point['result']['world_xyz'] == [4, 5, 1]
    bad, _ = hub.call('look', {'observation_id': 'obs_00000', 'crop_xyxy': [0, 0, 30, 30]})
    assert not bad['ok'] and hub.executor.calls == 0


def test_no_depth_or_geometry_measurement_tool_and_foreign_mask_rejected(hub):
    names = {s['function']['name'] for s in tool_schemas()}
    assert {'segment', 'look', 'back_project', 'get_state'} <= names
    assert not {'depth', 'measure'} & names
    _, _ = hub.call('segment', {'observation_id': 'obs_00000', 'prompt': 'object'})
    hub.observe()
    bad, _ = hub.call('look', {'observation_id': 'obs_00001', 'candidate_ids': ['mask_00001']})
    assert not bad['ok']
    bad, _ = hub.call('measure', {})
    assert not bad['ok'] and hub.executor.calls == 0


def test_static_observation_has_no_fake_remaining_motion_budget(tmp_path):
    source = tmp_path / 'source'; source.mkdir()
    robot = FakeRobot()
    (source/'initial_state.json').write_text(json.dumps(robot.state()))
    (source/'metadata.json').write_text(json.dumps(robot.metadata))
    for c in robot.cameras:
        f = robot.camera(c)
        Image.fromarray(f['rgb']).save(source/(c+'.png'))
        np.savez(source/(c+'.npz'), **{k:f[k] for k in ('depth_m', 'K', 'world_from_camera')})
    run = tmp_path/'run'; run.mkdir()
    hub = ToolHub(SavedObservationRobot(source), None, run, lambda x: None, allowed_tools=PERCEPTION_TOOLS)
    description, _ = hub.observe()
    assert description['static_observation'] and description['state']['remaining_actions'] is None
    assert 'official_success' not in json.dumps(description)
    assert not any(s['function']['name'].startswith('move_') for s in hub.schemas)


def test_grasp_planning_can_inspect_then_submit_with_full_tool_history(hub):
    hub.schemas = [s for s in hub.schemas if s['function']['name'] in PERCEPTION_TOOLS]
    hub.validators = {s['function']['name']: hub.validators[s['function']['name']] for s in hub.schemas}
    submit = {'type':'function', 'function': {'name':'submit_grasp_sketch', 'parameters':SKETCH_SCHEMA}}
    class Client:
        turn = 0
        def complete(self, messages, tools):
            self.turn += 1
            names = {t['function']['name'] for t in tools}
            assert {'segment', 'look', 'back_project', 'get_state', 'submit_grasp_sketch'} <= names
            assert not {'measure', 'depth', 'move_to'} & names
            if self.turn == 1:
                calls = [call('segment', {'observation_id':'obs_00000', 'prompt':'visible handle'}, 'one')]
            else:
                assert any(m['role']=='tool' for m in messages)
                assert messages[-1]['role']=='user' and any(p['type']=='image_url' for p in messages[-1]['content'])
                calls = [call('submit_grasp_sketch', sketch(), 'two')]
            return {'message':{'role':'assistant','tool_calls':calls}, 'finish_reason':'tool_calls'}
    messages = [{'role':'system', 'content':'fixture'}]
    actual, _, timing = planning_dialogue(Client(), hub, messages, submit)
    assert actual == sketch() and timing['rounds'] == 2 and hub.executor.calls == 0
    assert messages[-1]['tool_call_id'] == 'two'
    copied = transcript_copy(messages)
    restored = restore_planning_images(copied, [hub.run/'observations'])
    assert restored == messages


def test_grasp_planner_rejects_action_batch_before_dispatch(hub):
    hub.schemas = [s for s in hub.schemas if s['function']['name'] in PERCEPTION_TOOLS]
    submit = {'type':'function', 'function': {'name':'submit_grasp_sketch', 'parameters':SKETCH_SCHEMA}}
    class Client:
        def complete(self, messages, tools):
            return {'message':{'role':'assistant','tool_calls':[
                call('segment', {'observation_id':'obs_00000','prompt':'object'}, 'one'),
                call('move_relative', {'source_step':0,'params':{'delta_m':[0,0,.1]}}, 'two')]}}
    from sketchik.agent import ProviderError
    with pytest.raises(ProviderError, match='motion is not exposed'):
        planning_dialogue(Client(), hub, [], submit)
    assert not hub.requests and hub.executor.calls == 0
