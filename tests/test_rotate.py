from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from jsonschema import Draft202012Validator, ValidationError
from scipy.spatial.transform import Rotation

from sketchik.kinematics import IKError
from sketchik.primitives import MotionExecutor, simple_rotation_target_pose
from sketchik.tools import PARAMS, ToolHub, tool_schemas


ROTATIONS=[('rotate_roll','x'),('rotate_pitch','y'),('rotate_yaw','z')]
INVALID_ANGLES=[True,np.bool_(False),'0.2',None,float('nan'),float('inf'),-float('inf')]


def entry_state(rotation=None):
    return {'position':[.2,.3,1.],'quaternion_xyzw':(
        Rotation.identity() if rotation is None else rotation).as_quat().tolist()}


@pytest.mark.parametrize('primitive,axis',ROTATIONS)
@pytest.mark.parametrize('initial',[
    Rotation.from_euler('xyz',[.3,-.2,.4]),
    Rotation.from_quat([.2,-.4,.1,.3]),
    Rotation.from_euler('xyz',[2.7,1.2,-2.8]),
])
def test_relative_world_rotation_preserves_xyz_and_full_existing_orientation(primitive,axis,initial):
    state=entry_state(initial);before=deepcopy(state)
    target=simple_rotation_target_pose(state,primitive,angle_rad=-.7)
    expected=Rotation.from_euler(axis,-.7)*initial
    assert target['position']==state['position']
    assert Rotation.from_quat(target['quaternion_xyzw']).as_matrix()==pytest.approx(expected.as_matrix())
    assert np.linalg.norm(target['quaternion_xyzw'])==pytest.approx(1.)
    assert state==before
    # Resolving against a tool axis would right-multiply and give another pose.
    assert not np.allclose(expected.as_matrix(),(initial*Rotation.from_euler(axis,-.7)).as_matrix())


@pytest.mark.parametrize('primitive,axis',ROTATIONS)
def test_zero_relative_angle_is_valid_and_preserves_entry_pose(primitive,axis):
    initial=Rotation.from_quat([.2,-.4,.1,.3]);state=entry_state(initial)
    target=simple_rotation_target_pose(state,primitive,angle_rad=0.)
    assert target['position']==state['position']
    assert Rotation.from_quat(target['quaternion_xyzw']).as_matrix()==pytest.approx(initial.as_matrix())


@pytest.mark.parametrize('primitive,axis',ROTATIONS)
def test_only_relative_angle_rad_is_accepted(primitive,axis):
    invalid=[{}, {'target_yaw':.2}, {'delta_yaw':.1}, {'target_pitch':.2}, {'delta_pitch':.1},
             {'angle_rad':.2,'axis':[1,0,0]}, {'angle_rad':.2,'frame':'tool'},
             {'angle_rad':.2,'pivot_world_m':[0,0,0]}, {'angle_rad':.2,'settle_steps':80}]
    for angles in invalid:
        with pytest.raises(ValueError):simple_rotation_target_pose(entry_state(),primitive,**angles)


@pytest.mark.parametrize('primitive,axis',ROTATIONS)
@pytest.mark.parametrize('angle',INVALID_ANGLES)
def test_rotation_rejects_nonfinite_or_nonnumeric_angles(primitive,axis,angle):
    with pytest.raises(ValueError):
        simple_rotation_target_pose(entry_state(),primitive,angle_rad=angle)


@pytest.mark.parametrize('primitive',['rotate','rotate_wrist','move',''])
def test_other_rotation_names_are_not_public_primitives(primitive):
    if primitive in ('rotate','rotate_wrist'):assert not hasattr(MotionExecutor,primitive)
    with pytest.raises(ValueError):simple_rotation_target_pose(entry_state(),primitive,angle_rad=.2)


class Robot:
    def __init__(self):
        self.steps=0;self.max_steps=1000;self.calls=[]
        self.current={**entry_state(Rotation.from_euler('xyz',[.3,-.2,.4])),
                      'q':[0.]*7,'step':0,'gripper_command':1.}
    def state(self):return deepcopy(self.current)
    def step(self,joint_positions,gripper=None):
        self.calls.append((np.asarray(joint_positions).copy(),gripper))
        self.steps+=1
        self.current.update(q=np.asarray(joint_positions).tolist(),step=self.steps,
            position=self.target['position'],quaternion_xyzw=self.target['quaternion_xyzw'])
        if gripper is not None:self.current['gripper_command']=gripper


def executor_fixture():
    robot=Robot();paths=[]
    def path(start,position,quaternion):
        robot.target={'position':np.asarray(position).tolist(),'quaternion_xyzw':np.asarray(quaternion).tolist()}
        paths.append(deepcopy(robot.target))
        return [SimpleNamespace(q=np.full(7,.03),elapsed_s=0.,position_error_m=0.,rotation_error_rad=0.)]
    executor=MotionExecutor.__new__(MotionExecutor)
    executor.robot=robot;executor.ik=SimpleNamespace(cartesian_path=path);executor.emit=lambda event:None
    return executor,robot,paths


@pytest.mark.parametrize('primitive,params',[
    ('move',{'position':[.3,.2,.9]}),
    ('move_relative',{'delta_m':[.1,-.1,-.1]}),
])
def test_xyz_movement_preserves_full_entry_orientation_and_gripper(primitive,params):
    executor,robot,paths=executor_fixture();original=robot.state()
    result=getattr(executor,primitive)(**params)
    assert result['converged'] and len(paths)==1
    assert paths[0]['position']==pytest.approx([.3,.2,.9])
    assert Rotation.from_quat(paths[0]['quaternion_xyzw']).as_matrix()==pytest.approx(
        Rotation.from_quat(original['quaternion_xyzw']).as_matrix())
    assert all(gripper is None for _,gripper in robot.calls)
    assert robot.state()['gripper_command']==original['gripper_command']


def test_move_schema_accepts_preserve_quaternion_or_both_directions_only():
    validator=Draft202012Validator(PARAMS['move']);position={'position':[.3,.2,.9]}
    for orientation in [{},{'quaternion_xyzw':[0,0,0,1]},
                        {'finger_direction':[0,0,-1],'closing_axis':[1,0,0]}]:
        validator.validate({**position,**orientation})
    for orientation in [{'finger_direction':[0,0,-1]},{'closing_axis':[1,0,0]},
                        {'quaternion_xyzw':[0,0,0,1],'finger_direction':[0,0,-1]},
                        {'quaternion_xyzw':[0,0,0,1],'finger_direction':[0,0,-1],'closing_axis':[1,0,0]}]:
        with pytest.raises(ValidationError):validator.validate({**position,**orientation})


@pytest.mark.parametrize('primitive,axis',ROTATIONS)
def test_executor_uses_ik_and_absolute_joints_preserving_xyz_and_gripper(primitive,axis):
    executor,robot,paths=executor_fixture()
    original=robot.state();initial=Rotation.from_quat(original['quaternion_xyzw'])
    result=getattr(executor,primitive)(angle_rad=.4,settle_steps=2)
    assert result['converged'] and len(paths)==1
    assert paths[0]['position']==original['position']
    assert Rotation.from_quat(paths[0]['quaternion_xyzw']).as_matrix()==pytest.approx(
        (Rotation.from_euler(axis,.4)*initial).as_matrix())
    assert robot.calls and all(target.shape==(7,) and gripper is None for target,gripper in robot.calls)
    assert robot.state()['gripper_command']==1.


@pytest.mark.parametrize('primitive,axis',ROTATIONS)
def test_invalid_requests_send_zero_joint_commands(primitive,axis):
    executor,robot,paths=executor_fixture();method=getattr(executor,primitive)
    for angle in INVALID_ANGLES:
        with pytest.raises(ValueError):method(angle_rad=angle)
    for angles in [{},{'target_yaw':.2},{'angle_rad':.2,'axis':[1,0,0]}]:
        with pytest.raises(TypeError):method(**angles)
    with pytest.raises(TypeError):method(.3)
    for settle in [-1,201,1.5]:
        with pytest.raises(ValueError):method(angle_rad=.4,settle_steps=settle)
    assert robot.steps==0 and robot.calls==[] and paths==[]
    assert robot.state()['gripper_command']==1.


@pytest.mark.parametrize('primitive,axis',ROTATIONS)
def test_ik_refusal_sends_zero_joint_commands(primitive,axis):
    executor,robot,_=executor_fixture()
    def refusal(*args):raise IKError('No solution for requested full orientation')
    executor.ik.cartesian_path=refusal
    with pytest.raises(IKError):getattr(executor,primitive)(angle_rad=.4)
    assert robot.steps==0 and robot.calls==[]
    assert robot.state()['gripper_command']==1.


@pytest.mark.parametrize('primitive,axis',ROTATIONS)
def test_exactly_three_rotation_schemas_are_published(primitive,axis):
    definitions={tool['function']['name']:tool['function'] for tool in tool_schemas()}
    expected={name for name,_ in ROTATIONS}
    assert {name for name in PARAMS if name.startswith('rotate')}==expected
    assert {name for name in definitions if name.startswith('rotate')}==expected
    assert set(PARAMS[primitive]['properties'])=={'angle_rad','settle_steps'}
    validator=Draft202012Validator(PARAMS[primitive])
    for params in [{'angle_rad':-.4},{'angle_rad':.4,'settle_steps':80}]:
        validator.validate(params)
        Draft202012Validator(definitions[primitive]['parameters']).validate({'source_step':0,'params':params})
        Draft202012Validator(definitions['execute_plan']['parameters']).validate({
            'source_step':0,'steps':[{'primitive':primitive,'params':params}]})
    invalid=[{}, {'angle_rad':None}, {'angle_rad':True},
             {'angle_rad':.2,'target_yaw':.2}, {'angle_rad':.2,'delta_pitch':.2},
             {'angle_rad':.2,'axis':[0,0,1]}, {'angle_rad':.2,'frame':'world'},
             {'angle_rad':.2,'pivot_world_m':[0,0,0]}, {'angle_rad':.2,'settle_steps':201}]
    for params in invalid:
        with pytest.raises(ValidationError):validator.validate(params)


def test_three_tools_and_numeric_plan_dispatch_through_ik_execution_chain():
    executor,robot,paths=executor_fixture();hub=ToolHub.__new__(ToolHub)
    hub.robot=robot;hub.executor=executor;hub.finished=None;hub.fatal=False;hub.emit=lambda event:None
    hub.validators={tool['function']['name']:Draft202012Validator(tool['function']['parameters'])
                    for tool in tool_schemas()}
    hub.observe=lambda:({'source_step':robot.steps},[])
    assert all(not hasattr(executor,name) and name not in hub.validators for name in ('move_pose','move_to'))
    result,_=hub.call('move',{'source_step':robot.steps,'params':{'position':entry_state()['position']}})
    assert result['ok'] and result['result']['completed_steps']==1
    for primitive,_ in ROTATIONS:
        result,_=hub.call(primitive,{'source_step':robot.steps,'params':{'angle_rad':.2}})
        assert result['ok'] and result['result']['completed_steps']==1
    result,_=hub.call('execute_plan',{'source_step':robot.steps,'steps':[
        {'primitive':primitive,'params':{'angle_rad':-.1}} for primitive,_ in ROTATIONS]})
    assert result['ok'] and result['result']['completed_steps']==3
    assert len(paths)==7 and robot.state()['gripper_command']==1.
    before=robot.steps
    result,_=hub.call('rotate_yaw',{'source_step':before,'params':{'angle_rad':.2,'axis':[0,0,1]}})
    assert not result['ok'] and result['dispatch']=='no_physics_steps' and robot.steps==before
