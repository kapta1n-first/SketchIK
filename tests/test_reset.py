from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from sketchik.kinematics import IKError, RobotIK
from sketchik.primitives import MotionExecutor


@pytest.fixture(scope='module')
def robot_asset():
    # CPU-only robot model, no environment, rendering, object state or API.
    from robosuite.models.robots import Panda
    from robosuite.models.grippers import gripper_factory
    arm=Panda(idn=0);gripper=gripper_factory('PandaGripper',idn=0);arm.add_gripper(gripper)
    return arm.get_xml(),arm.joints,gripper.important_sites['grip_site']


class Robot:
    def __init__(self,asset):
        xml,self.joint_names,self.site=asset
        self.robot=SimpleNamespace(robot_model=SimpleNamespace(get_xml=lambda:xml))
        self.fk=RobotIK(xml,self.joint_names,self.site)
        self.q=np.array([0.,-.5,0.,-2.,0.,1.5,.7]);self.steps=0;self.max_steps=5000
        self.gripper_command=-1.;self.calls=[];self.track=True
    def state(self):
        p,r=self.fk.fk(self.q)
        return {'q':self.q.tolist(),'position':p.tolist(),
            'quaternion_xyzw':Rotation.from_matrix(r).as_quat().tolist(),
            'step':self.steps,'gripper_command':self.gripper_command}
    def step(self,target,gripper=None):
        self.calls.append((np.asarray(target).copy(),gripper));self.steps+=1
        if self.track:self.q=np.asarray(target).copy()
        if gripper is not None:self.gripper_command=gripper


@pytest.fixture
def executor(robot_asset):
    robot=Robot(robot_asset);events=[];executor=MotionExecutor(robot,events.append)
    home=deepcopy(executor.initial_state)
    robot.q+=np.array([.04,-.03,.02,.04,-.04,.03,-.04])
    return executor,robot,events,home


def test_reset_uses_saved_real_fk_pose_and_initial_ik_seed_without_env_reset(executor,monkeypatch):
    motion,robot,events,home=executor;robot.gripper_command=1.
    assert motion.initial_state==home and not np.array_equal(robot.q,home['q'])
    original=motion.ik.solve;solves=[]
    def solve(position,quaternion,seed):
        solves.append((np.asarray(position).copy(),np.asarray(quaternion).copy(),np.asarray(seed).copy()))
        return original(position,quaternion,seed)
    monkeypatch.setattr(motion.ik,'solve',solve)
    result=motion.reset(settle_steps=2)
    assert result['converged'] and len(solves)>1 and robot.calls
    np.testing.assert_array_equal(solves[0][2],home['q'])
    np.testing.assert_array_equal(solves[0][0],home['position'])
    actual=robot.state()
    assert np.linalg.norm(np.asarray(actual['position'])-home['position'])<.0005
    assert (Rotation.from_quat(actual['quaternion_xyzw'])*Rotation.from_quat(home['quaternion_xyzw']).inv()).magnitude()<np.deg2rad(.5)
    error=np.asarray(actual['q'])-home['q']
    np.testing.assert_array_equal(result['reset']['joint_error_from_initial_rad'],error)
    assert result['reset']['max_joint_error_from_initial_rad']==pytest.approx(max(abs(error)))
    assert result['reset']['initial_joints_match']==(max(abs(error))<=.02)
    assert robot.gripper_command==1. and all(gripper is None for _,gripper in robot.calls)
    assert result['reset']['gripper_command_preserved']
    assert not result['reset']['collision_checked_by_reset'] and motion.initial_state==home
    assert events[-1]['event']=='reset_finished'


def test_reset_executes_the_supplied_route_and_preserves_its_complete_endpoint(executor,monkeypatch):
    motion,robot,events,home=executor
    middle=(np.asarray(robot.state()['position'])+home['position'])/2+[0,0,.02]
    route=[{'position':middle.tolist(),'quaternion_xyzw':home['quaternion_xyzw']},
           {'position':home['position'],'quaternion_xyzw':(-np.asarray(home['quaternion_xyzw'])).tolist()}]
    before=deepcopy(route);targets=[];original=motion.ik.cartesian_path
    def path(seed,position,quaternion):
        targets.append((np.asarray(position).copy(),np.asarray(quaternion).copy()))
        return original(seed,position,quaternion)
    monkeypatch.setattr(motion.ik,'cartesian_path',path)
    result=motion.reset(checked_waypoints=route,settle_steps=2)
    assert result['converged'] and result['reset']['checked_waypoints_supplied']
    assert len(targets)==2 and route==before
    for target,waypoint in zip(targets,route):
        np.testing.assert_array_equal(target[0],waypoint['position'])
        np.testing.assert_allclose(Rotation.from_quat(target[1]).as_matrix(),
                                   Rotation.from_quat(waypoint['quaternion_xyzw']).as_matrix(),atol=1e-12)
    planned=next(event for event in events if event['event']=='motion_planned')
    assert planned['waypoints']==route and not planned['collision_checked']


@pytest.mark.parametrize('invalid',['empty','wrong_position','wrong_orientation','missing_orientation',
                                   'relative_endpoint','invalid_intermediate','unknown_field'])
def test_invalid_reset_route_sends_zero_actions(executor,invalid):
    motion,robot,events,home=executor
    route=[{key:home[key] for key in ('position','quaternion_xyzw')}]
    if invalid=='empty':route=[]
    elif invalid=='wrong_position':route[0]['position']=(np.asarray(home['position'])+[.001,0,0]).tolist()
    elif invalid=='wrong_orientation':route[0]['quaternion_xyzw']=(Rotation.from_euler('z',.01)*Rotation.from_quat(home['quaternion_xyzw'])).as_quat().tolist()
    elif invalid=='missing_orientation':del route[0]['quaternion_xyzw']
    elif invalid=='relative_endpoint':route[0]['relative_m']=route[0].pop('position')
    elif invalid=='invalid_intermediate':route.insert(0,{'position':[np.nan,0,0]})
    else:route[0]['unexpected']=True
    with pytest.raises(ValueError):motion.reset(checked_waypoints=route)
    assert robot.calls==[] and robot.steps==0 and motion.initial_state==home


@pytest.mark.parametrize('settle',[-1,201,True,1.2])
def test_invalid_settling_or_home_ik_failure_sends_zero_actions(executor,monkeypatch,settle):
    motion,robot,*_=executor
    with pytest.raises(ValueError):motion.reset(settle_steps=settle)
    calls=[]
    def refusal(*args):calls.append(True);raise IKError('Home full pose unavailable')
    monkeypatch.setattr(motion.ik,'solve',refusal)
    with pytest.raises(IKError):motion.reset()
    assert len(calls)==1 and robot.calls==[] and robot.steps==0


def test_reset_reports_actual_nonconvergence_without_claiming_original_joints(executor):
    motion,robot,events,home=executor;robot.track=False
    result=motion.reset(settle_steps=0)
    assert not result['converged'] and robot.calls
    assert result['reset']['max_joint_error_from_initial_rad']==pytest.approx(.04)
    assert not result['reset']['initial_joints_match']
    assert motion.initial_state==home and result['reset']['gripper_command_preserved']


def test_recorded_observation_return_solves_full_route_from_actual_entry_without_actions(executor,tmp_path):
    import json
    from sketchik.motion_checks import observation_return_route
    motion,robot,events,home=executor
    current=robot.state();current['step']=10
    intermediate=deepcopy(current);intermediate['step']=5
    original_q=robot.q.copy()
    robot.q=(robot.q+np.asarray(home['q']))/2
    intermediate.update(robot.state());intermediate['step']=5
    robot.q=original_q
    sources={'old':{**deepcopy(home),'gripper_command':-1.},'middle':intermediate,'current':current}
    for name,state in sources.items():
        folder=tmp_path/'observations'/name;folder.mkdir(parents=True)
        (folder/'observation.json').write_text(json.dumps({'state':state}))
    hub=SimpleNamespace(run=tmp_path,ik_only=True,executor=motion,
        journal=SimpleNamespace(data={'scene_clouds':{key:{'source_step':value['step']} for key,value in sources.items()}}))
    route=observation_return_route(hub,None,current,home,home['step'])
    assert robot.calls==[] and robot.q.tolist()==current['q']
    assert [item['source_step'] for item in route['recorded_anchors']]==[5,home['step'],home['step']]
    assert route['waypoints'][-1]=={key:home[key] for key in ('position','quaternion_xyzw')}
    assert route['ik_pass'] and route['path_checked'] and not route['collision_checked']
    assert all(check['path_checked'] for check in route['segment_checks'])
    assert route['robot_actions_during_planning']==0 and not route['environment_reset']


def test_failed_return_segment_and_closed_gripper_never_send_actions(executor,tmp_path,monkeypatch):
    import json
    from sketchik.motion_checks import observation_return_route
    motion,robot,events,home=executor
    source={**robot.state(),'step':5}
    folder=tmp_path/'observations'/'old';folder.mkdir(parents=True)
    (folder/'observation.json').write_text(json.dumps({'state':source}))
    hub=SimpleNamespace(run=tmp_path,ik_only=True,executor=motion,
        journal=SimpleNamespace(data={'scene_clouds':{'old':{'source_step':5}}}))
    actual={**robot.state(),'step':10}
    def no_path(*args,**kwargs):raise IKError('Still discontinuous')
    monkeypatch.setattr(motion.ik,'cartesian_path',no_path)
    with pytest.raises(IKError):observation_return_route(hub,None,actual,home,0)
    with pytest.raises(ValueError,match='open-gripper'):
        observation_return_route(hub,None,{**actual,'gripper_command':1.},home,0)
    assert robot.calls==[]
