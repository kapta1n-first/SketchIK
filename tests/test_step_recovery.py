from contextlib import contextmanager
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest
from jsonschema import Draft202012Validator, ValidationError
from scipy.spatial.transform import Rotation

from sketchik.agent import ProviderError,RejectedResponse
from sketchik.kinematics import IKError
from sketchik.primitives import MotionExecutor, simple_rotation_target_pose
from sketchik.step_recovery import run_step,repeats_stalled_goal
from sketchik.task_execution import BindingUnavailable, execute_sketch


STEP={'primitive':'move','target_id':'object','reference_target_id':'object',
      'camera':'agentview','intent':'approach the selected visible surface',
      'contact_relation':'Maintain observed pad contact'}
POSE={'position':[.1,0.,1.],'quaternion_xyzw':[0.,0.,0.,1.]}


def blocked(stage='ik_endpoint'):
    exc=BindingUnavailable('specific full-pose failure')
    exc.failure={'failure_stage':stage,'error':'specific full-pose failure','executed':False}
    return exc


def fixture(tmp_path, *, outputs=(), compile_failure=False, repaired=True):
    robot=SimpleNamespace(steps=30,max_steps=1000,done=False)
    state={'step':30,'q':[0.]*7,'position':[0.,0.,1.],
           'quaternion_xyzw':[0.,0.,0.,1.],'gripper_command':1.}
    robot.state=lambda:{**deepcopy(state),'step':robot.steps}
    records=[];calls=[];queue=iter(outputs)
    def observe():
        calls.append(('observe',robot.steps))
        return {'observation_id':f'actual_{robot.steps}','source_step':robot.steps,'state':robot.state()},[
            {'type':'text','text':f'CURRENT evidence {robot.steps}'}]
    def move(**params):
        calls.append(('move',deepcopy(params)));robot.steps+=1
        item=next(queue)
        if isinstance(item,Exception):raise item
        return {**deepcopy(item),'actions':1,'state':robot.state()}
    @contextmanager
    def search(**kwargs):yield
    hub=SimpleNamespace(run=tmp_path,robot=robot,emit=records.append,observe=observe,
        executor=SimpleNamespace(move=move,ik=SimpleNamespace(recovery_search=search)),
        journal=SimpleNamespace(sync_observations=lambda _:None))
    stage={'plan':{'source_step':30,'subtask':'one stage','steps':[deepcopy(STEP)]},
           'source_step':30,'selections':[{'anchor':{'kind':'current_eef'},'offset_m':[.1,0,0]}],
           'execution_receipts':[]}
    compiled={'parameters':deepcopy(POSE),'audit':{'compiled_pose':deepcopy(POSE)}}
    class Grounder:
        requests=0
        def compile_step(self,stage,index,description,folder):
            calls.append(('compile',index))
            if compile_failure:raise blocked()
            return deepcopy(compiled)
        def recover_compile(self,failure,folder):
            calls.append(('T1_compile',failure.failure['failure_stage']))
            raise blocked()
        def record_execution(self,receipt):calls.append(('receipt',receipt['primitive']))
    return hub,Grounder(),stage,calls,records,compiled




def test_successful_t1_ik_does_not_call_b_or_change_absolute_goal(tmp_path):
    hub,b,stage,calls,_,compiled=fixture(tmp_path,compile_failure=True,outputs=[{'converged':True}])
    b.recover_compile=lambda failure,folder:{**deepcopy(compiled),'ik_recovery_search':True}
    desc,pics=hub.observe();result=run_step(hub,b,stage,0,desc,pics,tmp_path/'step')
    assert result['compiled']['parameters']==POSE and b.requests==0
    assert [c for c in calls if c[0]=='move'][0][1]==POSE


def test_dispatch_uses_compiled_joint_record_without_replanning_primitive(tmp_path):
    hub, compiler, stage, calls, _, compiled = fixture(tmp_path)
    record = {'artifact_id': 'same-compiled-branch', 'trajectory': [[.01]*7]}
    compiled['compiled_motion'] = record
    hub.executor.move = lambda **kwargs: pytest.fail('Compiled move must not enter a second IK planning chain')
    dispatched = []
    def execute_prepared(value):
        dispatched.append(deepcopy(value))
        hub.robot.steps += 1
        return {'converged': True, 'actions': 1, 'state': hub.robot.state(),
            'compiled_motion_id': value['artifact_id']}
    hub.executor.execute_prepared_motion = execute_prepared
    desc, pictures = hub.observe()
    result = run_step(hub, compiler, stage, 0, desc, pictures, tmp_path/'step')
    assert dispatched == [record] and result['output']['compiled_motion_id'] == record['artifact_id']
    assert result['output']['converged'] and hub.robot.steps == 31


def test_progressing_motion_continues_absolute_goal_once_without_reapplying_delta(tmp_path):
    hub,b,stage,calls,_,_=fixture(tmp_path,outputs=[{'converged':False,
        'motion_progress':{'recent_progress':True},'continuation_goal':deepcopy(POSE)}])
    def continuation(output,**kwargs):
        calls.append(('continue',deepcopy(output['continuation_goal'])))
        hub.robot.steps+=1
        return {'converged':True,'actions':1,'state':hub.robot.state()}
    hub.executor.continue_motion=continuation
    desc,pics=hub.observe();result=run_step(hub,b,stage,0,desc,pics,tmp_path/'step')
    assert result['output']['converged'] and result['output']['actions']==2
    assert [c[0] for c in calls].count('move')==1 and b.requests==0
    assert [c for c in calls if c[0]=='continue'][0][1]==POSE






@pytest.mark.parametrize('changed',['position','orientation'])
def test_stall_check_allows_new_geometric_target(changed):
    failure={'failure_stage':'motion','executed':True,'original_absolute_goal':deepcopy(POSE),
        'actual_motion':{'motion_progress':{'recent_progress':False}}}
    candidate=deepcopy(POSE)
    if changed=='position':candidate['position'][0]+=.01
    else:candidate['quaternion_xyzw']=Rotation.from_euler('z',.1).as_quat().tolist()
    assert not repeats_stalled_goal(failure,{'audit':{'compiled_pose':candidate}},POSE)












def test_unknown_dispatch_outcome_cannot_enter_recovery(tmp_path):
    hub,b,stage,_,_,_=fixture(tmp_path,outputs=[IKError('raised after physical movement')])
    desc,pics=hub.observe()
    with pytest.raises(RuntimeError,match='state advanced'):run_step(hub,b,stage,0,desc,pics,tmp_path/'step')
    assert b.requests==0




def test_controller_continuation_keeps_absolute_pose_gripper_and_joint_target():
    class Robot:
        steps=10;max_steps=100;done=False;value=.04
        def state(self):
            return {'step':self.steps,'position':[self.value,0.,1.],
                'quaternion_xyzw':[0.,0.,0.,1.],'q':[self.value]*7,'gripper_command':1.}
        def step(self,target,*args):
            assert not args  # no hidden gripper operation
            np.testing.assert_array_equal(target,np.zeros(7))
            self.value*=.7;self.steps+=1
    e=MotionExecutor.__new__(MotionExecutor);e.robot=Robot();e.emit=lambda _:None
    e.ik=SimpleNamespace(lower=np.full(7,-1.),upper=np.ones(7))
    failed={'converged':False,'motion_progress':{'recent_progress':True},'continuation_goal':{
        'source_step':10,'position':[0.,0.,1.],'quaternion_xyzw':[0.,0.,0.,1.],
        'joint_target':[0.]*7,'gripper_command':1.}}
    out=e.continue_motion(failed)
    assert out['converged'] and out['absolute_goal_preserved'] and out['gripper_command_preserved']
    assert out['actions']<40
    with pytest.raises(ValueError,match='current progressing'):e.continue_motion(failed)


def test_controller_continuation_stops_when_progress_stalls():
    assert not MotionExecutor._recent_progress([(1.,.2,.5)]*10)['recent_progress']
    e=MotionExecutor.__new__(MotionExecutor)
    robot=SimpleNamespace(steps=0,max_steps=100,done=False)
    robot.state=lambda:{'step':robot.steps,'position':[.1,0.,1.],
        'quaternion_xyzw':[0,0,0,1],'q':[.1]*7,'gripper_command':1.}
    robot.step=lambda *args:setattr(robot,'steps',robot.steps+1)
    e.robot=robot;e.emit=lambda _:None;e.ik=SimpleNamespace(lower=np.full(7,-1.),upper=np.ones(7))
    failed={'converged':False,'motion_progress':{'recent_progress':True},'continuation_goal':{
        'source_step':0,'position':[0,0,1],'quaternion_xyzw':[0,0,0,1],
        'joint_target':[0.]*7,'gripper_command':1.}}
    out=e.continue_motion(failed)
    assert not out['converged'] and out['actions']<=10


@pytest.mark.parametrize('maximum', [1, 40])
def test_continuation_checks_contact_after_last_step_without_advancing_again(maximum):
    class Robot:
        steps = 10
        max_steps = 100
        done = False
        value = .04
        def state(self):
            return {'step': self.steps, 'position': [self.value, 0., 1.],
                'quaternion_xyzw': [0., 0., 0., 1.], 'q': [self.value]*7,
                'gripper_command': 1.}
        def step(self, target):
            np.testing.assert_array_equal(target, np.zeros(7))
            self.steps += 1
            self.value *= .7
    executor = MotionExecutor.__new__(MotionExecutor)
    executor.robot = Robot()
    executor.emit = lambda event: None
    executor.ik = SimpleNamespace(lower=np.full(7, -1.), upper=np.ones(7))
    failed = {'converged': False, 'motion_progress': {'recent_progress': True},
        'continuation_goal': {'source_step': 10, 'position': [0., 0., 1.],
            'quaternion_xyzw': [0., 0., 0., 1.], 'joint_target': [0.]*7,
            'gripper_command': 1.}}
    checked = []
    def guard(state):
        checked.append(state['step'])
        if state['step'] == 11:
            return {'failure_stage': 'carried_contact', 'error': 'contact requires review'}
    result = executor.continue_motion(failed, max_steps=maximum, stop_condition=guard)
    assert executor.robot.steps == 11 and result['actions'] == 1
    assert not result['converged']
    assert result['continuation_interruption']['failure_stage'] == 'carried_contact'
    assert result['absolute_goal_preserved'] and result['gripper_command_preserved']
    assert checked == [10, 11]


def test_t1_robot_jaw_guard_uses_frozen_contact_baseline_without_new_rgbd(tmp_path, monkeypatch):
    from sketchik.episode_state import EpisodeState
    from sketchik.task_execution import NumericalCompiler
    episode = EpisodeState(tmp_path)
    baseline = {'source_step': 20,
        'robot_gripper_geometry': {'current_inner_jaw_gap_m': .020},
        'carried_observed_geometry': {'object_id': 'object', 'reference_available': True}}
    episode.data.update(contact_baseline=deepcopy(baseline), contact_boundary_step=19)
    robot = SimpleNamespace(steps=30)
    events = []
    compiler = NumericalCompiler.__new__(NumericalCompiler)
    compiler.hub = SimpleNamespace(episode_state=episode, robot=robot,
        executor=SimpleNamespace(ik=SimpleNamespace(model=object())),
        carried_target_id='object', carried_surface_reference=None, emit=events.append)
    monkeypatch.setattr('sketchik.robot_geometry.measured_inner_jaw_gap',
        lambda ik, state: state['measured_gap'])
    guard = compiler.continuation_guard()
    assert guard({'step': 30, 'gripper_command': 1., 'measured_gap': .020}) is None
    robot.steps = 31
    interrupted = guard({'step': 31, 'gripper_command': 1., 'measured_gap': .001})
    assert interrupted['failure_stage'] == 'carried_contact'
    assert episode.data['carry_premise'] == 'invalidated_by_contact_change'
    assert episode.data['contact_baseline'] == baseline
    assert episode.data['contact_warning']['source_step'] == 31
    assert len(events) == 1 and events[0]['check_phase'] == 'T1'
    # Later wider jaws or a readable predicted cloud cannot clear the warning.
    assert guard({'step': 31, 'gripper_command': 1., 'measured_gap': .020})
    assert len(events) == 1


def test_t1_interruption_returns_to_same_agent_instead_of_reporting_convergence(tmp_path):
    hub, compiler, stage, calls, events, _ = fixture(tmp_path, outputs=[{
        'converged': False, 'motion_progress': {'recent_progress': True},
        'continuation_goal': deepcopy(POSE)}])
    stop = lambda state: {'failure_stage': 'carried_contact', 'error': 'jaw changed'}
    compiler.continuation_guard = lambda: stop
    def continuation(output, *, max_steps, stop_condition):
        assert max_steps == 40 and stop_condition is stop
        hub.robot.steps += 1
        return {'converged': False, 'actions': 1, 'state': hub.robot.state(),
            'continuation_interruption': stop_condition(hub.robot.state())}
    hub.executor.continue_motion = continuation
    description, pictures = hub.observe()
    outcome = run_step(hub, compiler, stage, 0, description, pictures, tmp_path/'step')
    assert outcome['failure']['failure_stage'] == 'carried_contact'
    assert outcome['output']['actions'] == 2
    assert [call[0] for call in calls].count('move') == 1
    assert outcome['recovery'][-1]['tier'] == 'T3'



def test_failed_same_target_ik_returns_to_agent_without_motion_or_model_repair(tmp_path):
    hub, executor, stage, calls, events, _ = fixture(tmp_path, compile_failure=True)
    description, pictures = hub.observe()
    with pytest.raises(BindingUnavailable) as caught:
        run_step(hub, executor, stage, 0, description, pictures, tmp_path/'step')
    assert hub.robot.steps == 30
    assert not any(c[0] == 'move' for c in calls)
    assert [e['tier'] for e in caught.value.recovery] == ['T1', 'T1', 'T3']
    assert not hasattr(executor, 'repair_step')


def test_stalled_motion_returns_actual_failure_without_blind_continuation(tmp_path):
    hub, executor, stage, calls, events, _ = fixture(tmp_path, outputs=[{
        'converged':False, 'motion_progress':{'recent_progress':False}}])
    hub.executor.continue_motion=lambda *a,**k:pytest.fail('No blind continuation')
    description, pictures=hub.observe()
    result=run_step(hub,executor,stage,0,description,pictures,tmp_path/'step')
    assert hub.robot.steps==31
    assert [c[0] for c in calls].count('move')==1
    assert result['failure']['failure_stage']=='motion'
    assert result['output']['converged'] is False
    assert result['recovery'][-1]['tier']=='T3'


def test_continuation_preserves_compiled_motion_and_whole_target_hold(tmp_path):
    first={'source_step_start':10,'source_step_end':91,'samples':81,
        'last_sent_joint_target_rad':[1.,2.], 'last_command_consecutive_steps':81,
        'last_command_held_since_step':11,'peak_current_setpoint_joint_error_rad':.3,
        'first_joint_tolerance_exceedance_step':11,'recent_measured_window':{'samples':10}}
    last={**first,'source_step_start':91,'source_step_end':101,'samples':10,
        'last_command_consecutive_steps':10,'last_command_held_since_step':92,
        'peak_current_setpoint_joint_error_rad':.2,'first_joint_tolerance_exceedance_step':92}
    hub,b,stage,calls,_,_=fixture(tmp_path,outputs=[{'converged':False,
        'motion_progress':{'recent_progress':True},'continuation_goal':deepcopy(POSE),
        'compiled_motion_id':'original_hash','tracking':first}])
    def continuation(output,**kw):
        hub.robot.steps+=1
        return {'converged':True,'actions':1,'state':hub.robot.state(),'tracking':last}
    hub.executor.continue_motion=continuation
    desc,pics=hub.observe();result=run_step(hub,b,stage,0,desc,pics,tmp_path/'step')
    output=result['output']
    assert output['compiled_motion_id']=='original_hash'
    assert output['tracking']['last_command_consecutive_steps']==91
    assert output['tracking']['last_command_held_since_step']==11
    assert output['tracking']['samples']==91
    assert output['tracking']['peak_current_setpoint_joint_error_rad']==.3
    assert output['tracking']['recent_measured_window']['samples']==10
    assert output['motion_attempts'][0]['tracking']==first
    assert output['motion_attempts'][1]['tracking']==last
