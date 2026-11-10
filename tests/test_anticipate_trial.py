import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
from jsonschema import Draft202012Validator

from sketchik.agent import ProviderError, transcript_copy
from sketchik.anticipate_trial import ANALYSIS_SCHEMA, restore_handoff, overlap_report, prepare_geometry
from sketchik.grasp_trial import SKETCH_SCHEMA, planning_dialogue, planning_instructions, replay_observation_prelude
from sketchik.perception import ObservationHistory
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


@pytest.mark.parametrize('timeout', [120, 300])
@pytest.mark.parametrize('direct', [False, True])
def test_b_timeout_is_explicit_and_complete_handoff_stays_read_only(tmp_path, monkeypatch, timeout, direct):
    from sketchik import anticipate_trial
    trial, original = handoff(tmp_path)
    (trial/'sketch.json').write_text(json.dumps(sketch()))
    run = tmp_path/'b_timeout'
    captured = {}
    analysis = {'source_step': 0, 'grasp_object': 'target', 'destination': 'container',
        'observed_evidence': [], 'placement_outline': [], 'missing_information': ['geometry'],
        'post_grasp_checks': [], 'ready_for_grounding': False, 'placement_candidates': []}
    class Client:
        def __init__(self, *args, **kwargs):
            captured['timeout'] = kwargs['timeout']
            captured['use_env_proxy'] = kwargs['use_env_proxy']
            self.metadata = {'timeout_s': kwargs['timeout'], 'automatic_retries': 0}
        def complete(self, messages, tools):
            captured['messages'] = messages.copy()
            return {'message': {'role': 'assistant', 'tool_calls': [
                call('submit_placement_analysis', analysis, 'b_final')]}}
    monkeypatch.setattr(anticipate_trial, 'ChatClient', Client)
    monkeypatch.setattr(anticipate_trial, 'load_settings', lambda _: {})
    anticipate_trial.analyse(trial, run, None, api_timeout=timeout, direct=direct)
    assert captured['timeout'] == timeout
    assert captured['use_env_proxy'] is not direct
    assert captured['messages'][1:len(original)] == original[1:]
    result = json.loads((run/'result.json').read_text())
    assert result['robot_actions'] == 0 and not result['executable_placement']


class TickRobot(FakeRobot):
    on_step = None

    def step(self, joints, gripper=-1.):
        self.steps += 1
        state = {**self.state(), 'joint_target': list(joints), 'gripper_command': gripper}
        if self.on_step:self.on_step(state)

    def camera(self, name):
        frame = super().camera(name)
        frame['world_from_camera'][0,3] = self.steps / 100
        return frame


def test_periodic_history_uses_physics_time_and_keeps_keyframes(tmp_path):
    robot=TickRobot();events=[]
    hub=ToolHub(robot,None,tmp_path,events.append,allowed_tools=PERCEPTION_TOOLS)
    initial,pictures=hub.observe()
    recorder=ObservationHistory(robot,tmp_path,events.append,interval_s=1.)
    recorder.attach(hub);robot.on_step=recorder.on_step
    for _ in range(45):robot.step([0.]*7)
    final,final_pictures=hub.observe()
    recorder.close();robot.on_step=None
    manifest=json.loads((tmp_path/'history_manifest.json').read_text())
    assert manifest['periodic_observation_ids']==['obs_00001','obs_00002']
    assert [hub.observations[i]['agentview']['step'] for i in manifest['periodic_observation_ids']]==[20,40]
    assert initial['source_step']==0 and final['source_step']==45
    assert manifest['recorded_control_steps']==45 and manifest['interval_simulation_s']==1.
    assert len(list((tmp_path/'observations').iterdir()))==4
    for _ in range(5):robot.state()  # No physical stepping during an API wait.
    assert len(list((tmp_path/'observations').iterdir()))==4
    with pytest.raises(ValueError):ObservationHistory(robot,tmp_path,events.append,interval_s=0.)


def test_temporal_handoff_preserves_every_periodic_frame_and_each_calibration(tmp_path):
    trial=tmp_path/'a';trial.mkdir();robot=TickRobot()
    def emit(event):
        with (trial/'planning_events.jsonl').open('a') as stream:stream.write(json.dumps(event)+'\n')
    hub=ToolHub(robot,None,trial,emit,allowed_tools=PERCEPTION_TOOLS)
    _,first=hub.observe()
    recorder=ObservationHistory(robot,trial,emit);recorder.attach(hub);robot.on_step=recorder.on_step
    for _ in range(45):robot.step([0.]*7)
    _,last=hub.observe();recorder.close();robot.on_step=None
    source=trial/'final_observation';source.mkdir()
    (source/'initial_state.json').write_text(json.dumps(robot.state()))
    (source/'metadata.json').write_text(json.dumps(robot.metadata))
    for camera in robot.cameras:
        f=robot.camera(camera);Image.fromarray(f['rgb']).save(source/(camera+'.png'))
        np.savez(source/(camera+'.npz'),**{k:f[k] for k in ('depth_m','K','world_from_camera')})
    (trial/'config.json').write_text(json.dumps({'observation':str(source)}))
    (trial/'planning_transcript.json').write_text(json.dumps(transcript_copy([
        {'role':'system','content':'A initial observation'}, {'role':'user','content':first},
        {'role':'user','content':last}])))
    run=tmp_path/'b';run.mkdir()
    b,messages=restore_handoff(trial,run,lambda event:None)
    for identity,expected in [('obs_00000',4.),('obs_00001',4.2),('obs_00002',4.4),('obs_00003',4.45)]:
        result,_=b.call('back_project',{'observation_id':identity,'camera':'agentview','pixel_xy':[4,5]})
        assert result['ok'] and result['result']['world_xyz'][0]==pytest.approx(expected)
        assert result['result']['current'] is False
    extra=messages[-1]['content']
    assert sum(v['type']=='image_url' for v in extra)==4
    descriptions=[json.loads(v['text']) for v in extra[1:] if v['type']=='text']
    assert {v['source_step'] for v in descriptions}=={20,40}
    assert all(v['historical_only'] for v in descriptions)


def test_recorded_observation_replay_validates_all_controls_before_dispatch(tmp_path):
    class Robot:
        max_steps=100
        calls=[]
        def step(self,q,g):self.calls.append((q,g))
    controls=[{'source_step':i,'joint_target':[0.]*7,'gripper_command':-1.} for i in (1,2)]
    name='controls.jsonl'
    metadata={'observation_prelude':{'kind':'recorded_observation_controls','steps':2,'controls_file':name}}
    (tmp_path/name).write_text('\n'.join(map(json.dumps,controls)))
    robot=Robot();robot.calls=[]
    replay_observation_prelude(robot,metadata,tmp_path)
    assert len(robot.calls)==2
    controls[-1]['gripper_command']=1.
    (tmp_path/name).write_text('\n'.join(map(json.dumps,controls)))
    robot.calls=[]
    with pytest.raises(ValueError,match='no replay dispatched'):replay_observation_prelude(robot,metadata,tmp_path)
    assert robot.calls==[]


def test_observation_a_can_move_b_remains_readonly_and_prompt_describes_parameters(tmp_path):
    from test_agent import FakeExecutor
    robot=FakeRobot();allowed=PERCEPTION_TOOLS|{'move_relative'}
    hub=ToolHub(robot,FakeExecutor(robot),tmp_path,lambda event:None,allowed_tools=allowed)
    hub.observe()
    submit={'type':'function','function':{'name':'submit_grasp_sketch','parameters':SKETCH_SCHEMA}}
    class A:
        turn=0
        def complete(self,messages,tools):
            self.turn+=1
            item=(call('move_relative',{'source_step':0,'params':{'delta_m':[0,0,.02]}},'move') if self.turn==1
                  else call('submit_grasp_sketch',{**sketch(),'source_step':3},'submit'))
            return {'message':{'role':'assistant','tool_calls':[item]}}
    result,_,_=planning_dialogue(A(),hub,[],submit,allow_actions=True)
    assert result['source_step']==3 and hub.executor.calls==1
    with pytest.raises(ProviderError,match='Read-only phase'):planning_dialogue(A(),hub,[],submit)
    guide=(Path(__file__).resolve().parents[1]/'prompts/grasp_trial.md').read_text()
    system=planning_instructions(guide,active_observation=True)
    assert '允许适量感知查询和空爪观察移动' in system
    assert '观察阶段直接调用move_to/move_pose等工具' in system
    assert '默认1秒仿真时间' in system and '相同画面反复分割/反投影不会增加可见信息' in system


def test_move_and_stale_submit_in_one_reply_does_not_replay_observation_motion(tmp_path):
    from test_agent import FakeExecutor
    robot=FakeRobot()
    hub=ToolHub(robot,FakeExecutor(robot),tmp_path,lambda event:None,
                allowed_tools=PERCEPTION_TOOLS|{'move_relative'})
    submit={'type':'function','function':{'name':'submit_grasp_sketch','parameters':SKETCH_SCHEMA}}
    class A:
        turn=0
        def complete(self,messages,tools):
            self.turn+=1
            if self.turn==1:
                calls=[call('move_relative',{'source_step':0,'params':{'delta_m':[0,0,.02]}},'move'),
                       call('submit_grasp_sketch',sketch(),'stale')]
            else:
                receipt=next(m for m in messages if m.get('tool_call_id')=='stale')
                assert json.loads(receipt['content'])['saved'] is False
                calls=[call('submit_grasp_sketch',{**sketch(),'source_step':3},'fresh')]
            return {'message':{'role':'assistant','tool_calls':calls}}
    result,_,timing=planning_dialogue(A(),hub,[],submit,allow_actions=True)
    assert result['source_step']==3 and timing['rounds']==2 and hub.executor.calls==1
