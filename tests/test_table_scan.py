from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from sketchik.program_scan import look_at
from sketchik.table_scan import DIRECTIONS,_promote_oblique_candidate,run_table_scan,scan_candidates,table_geometry


def surfaces():
    x,y=np.meshgrid(np.linspace(-.3,.5,80),np.linspace(-.2,.6,80))
    table=np.column_stack([x.ravel(),y.ravel(),np.full(x.size,.95)])
    x,y=np.meshgrid(np.linspace(.02,.12,16),np.linspace(.1,.2,16))
    raised=np.column_stack([x.ravel(),y.ravel(),np.full(x.size,1.08)])
    return np.r_[table,raised]


def test_table_scan_import_does_not_preempt_renderer_configuration():
    root=Path(__file__).resolve().parents[1]
    result=subprocess.run([sys.executable,'-c','import sys; import sketchik.table_scan; assert "mujoco" not in sys.modules'],
        env={**os.environ,'PYTHONPATH':str(root/'src')},capture_output=True,text=True)
    assert result.returncode==0,result.stderr


@pytest.fixture
def scan_hub(tmp_path,monkeypatch):
    state={'step':30,'q':np.zeros(7),'position':[0.,.1,1.3],'quaternion_xyzw':[1.,0.,0.,0.],
           'finger_joint_names':[],'finger_joint_positions':[],'gripper_command':-1.}
    calls={'frames':[],'builds':[],'plans':[],'moves':[],'events':[],'captures':0}
    robot=SimpleNamespace(steps=30,state=lambda:deepcopy(state))
    mount=np.eye(4);mount[2,3]=.1
    fixed=look_at(np.array([1.,.2,1.8]),np.array([.1,.2,.95]),np.array([0.,1.,0.]))
    def camera(name):
        eef=np.eye(4);eef[:3,:3]=Rotation.from_quat(state['quaternion_xyzw']).as_matrix();eef[:3,3]=state['position']
        transform=fixed.copy() if name=='agentview' else eef@np.linalg.inv(mount)
        return {'step':robot.steps,'camera':name,'world_from_camera':transform,
            'K':np.array([[90.,0.,64.],[0.,90.,64.],[0.,0.,1.]]),
            'depth_m':np.ones((128,128)),'rgb':np.zeros((128,128,3),np.uint8)}
    robot.camera=camera
    def move(waypoints):
        calls['moves'].append(deepcopy(waypoints));robot.steps+=3
        state.update(step=robot.steps,position=waypoints[-1]['position'],quaternion_xyzw=waypoints[-1]['quaternion_xyzw'])
        return {'converged':True,'official_success':False}
    hub=SimpleNamespace(run=tmp_path,robot=robot,executor=SimpleNamespace(ik=None,move_waypoints=move),
        observations={'obs_00000':{name:camera(name) for name in ('agentview','robot0_eye_in_hand')}},emit=calls['events'].append)
    def read(identity,camera):return {},[{'type':'image_url','image_url':{'url':identity+'/'+camera}}]
    def observe_scene():
        calls['captures']+=1;identity=f'obs_{len(hub.observations):05d}'
        hub.observations[identity]={name:camera(name) for name in ('agentview','robot0_eye_in_hand')}
        return {'observation_id':identity,'source_step':robot.steps},[
            read(identity,name)[1][0] for name in ('agentview','robot0_eye_in_hand')]
    hub.read_image=read;hub.observe_scene=observe_scene
    class Clearance:
        def __init__(self,*args):self.points=surfaces();self.revision=0
        def add_frame(self,frame,state):
            assert frame['step']==state['step'];calls['frames'].append((frame,deepcopy(state)))
        def build(self):
            self.revision+=1;calls['builds'].append(len(calls['frames']))
            return {'obstacle_points':len(self.points),'margin_m':.012}
        def set_state(self,state):pass
        def visibility(self,*args,**kwargs):return .9
        def plan(self,state,candidate):
            calls['plans'].append((deepcopy(state),deepcopy(candidate),self.revision))
            return {'waypoints':[{key:candidate[key] for key in ('position','quaternion_xyzw')}],
                'checked_configurations':10,'observed_clearance_passed':True,'route':'direct'}
    monkeypatch.setattr('sketchik.observed_clearance.ObservedClearance',Clearance)
    return hub,state,calls,Clearance


def test_table_scan_five_directions_preserve_every_native_frame_and_refresh_same_step_obstacles(scan_hub):
    hub,state,calls,_=scan_hub
    result,pictures=run_table_scan(hub)
    assert result['status']=='completed' and result['captured_views']==5
    assert [item['target']['target_id'] for item in result['attempts']]==['table_'+name for name in DIRECTIONS]
    assert len(pictures)==12 and len({item['image_url']['url'] for item in pictures})==12
    assert calls['captures']==len(calls['moves'])==5 and calls['builds']==[2,4,6,8,10,12]
    assert [item[0]['step'] for item in calls['plans']]==[30,33,36,39,42]
    assert [item[2] for item in calls['plans']]==[1,2,3,4,5]
    for index,(frame,source) in enumerate(calls['frames']):
        assert frame['step']==source['step']==30+3*(index//2)
    for attempt,waypoints in zip(result['attempts'],calls['moves']):
        assert attempt['checked_route']['waypoints']==waypoints
        assert attempt['selected']['quaternion_xyzw']==waypoints[-1]['quaternion_xyzw']
        assert 'official_success' not in attempt['execution']
        bearing=np.asarray(attempt['selected']['camera_position_m'])-attempt['geometry']['view_aim_world_m']
        name=attempt['target']['target_id'].removeprefix('table_')
        if name=='top':np.testing.assert_allclose(bearing[:2],0,atol=1e-12)
        else:
            front=np.asarray(result['front_direction_world_xy']);left=np.array([-front[1],front[0]])
            expected={'front':front,'back':-front,'left':left,'right':-left}[name]
            np.testing.assert_allclose(bearing[:2]/np.linalg.norm(bearing[:2]),expected,atol=1e-12)
        assert bearing[2]>0
    assert result['source_step']==45 and result['robot_actions']==15 and result['agent_requests']==0
    assert not result['complete_backside_coverage_certified'] and result['full_scene_native_history_preserved']
    assert len(result['geometry_updates'])==6
    for geometry in result['geometry_updates']:assert Path(geometry['geometry_point_file']).exists()
    assert json.loads(Path(result['diagnostic_file']).read_text())==result


def test_initial_targets_and_camera_roll_stay_fixed_while_observed_window_and_entry_state_change(scan_hub,monkeypatch):
    hub,state,calls,base=scan_hub
    class ExpandingClearance(base):
        def build(self):
            if self.revision:
                self.points=np.r_[surfaces(),surfaces()+[.12*self.revision,0.,0.]]
            return super().build()
    monkeypatch.setattr('sketchik.observed_clearance.ObservedClearance',ExpandingClearance)
    generated={}
    def record_candidates(source,wrist,geometry,points,above,direction,clearance):
        # All five camera/EEF proposal sets must precede any physical command.
        assert not calls['moves'] and source['step']==wrist['step']==30
        name=DIRECTIONS[len(generated)]
        generated[name]=deepcopy(scan_candidates(source,wrist,geometry,points,above,direction,clearance))
        return deepcopy(generated[name])
    monkeypatch.setattr('sketchik.table_scan.scan_candidates',record_candidates)
    move=hub.executor.move_waypoints
    def measured_roll_error(waypoints):
        result=move(waypoints)
        measured=Rotation.from_euler('z',.001*len(calls['moves']))*Rotation.from_quat(state['quaternion_xyzw'])
        state['quaternion_xyzw']=measured.as_quat().tolist()
        return result
    hub.executor.move_waypoints=measured_roll_error
    result,_=run_table_scan(hub)
    assert result['captured_views']==5
    reference=result['planning_reference']
    assert reference['source_step']==30 and reference['targets_frozen']
    initial_geometry=reference['geometry']
    assert result['geometry_updates'][-1]['display_xy_max_m'][0]>initial_geometry['display_xy_max_m'][0]+.3
    for index,(name,attempt) in enumerate(zip(DIRECTIONS,result['attempts'])):
        assert attempt['candidates']==reference['candidates'][name]==generated[name]
        assert attempt['selected']==generated[name][0]
        assert attempt['geometry']==initial_geometry and attempt['planning_reference_source_step']==30
        assert attempt['observed_geometry']==result['geometry_updates'][index]
        assert calls['plans'][index][0]['step']==30+3*index
        assert calls['plans'][index][2]==index+1
        if index:
            previous=Rotation.from_quat(result['attempts'][index-1]['selected']['quaternion_xyzw'])
            actual=Rotation.from_quat(calls['plans'][index][0]['quaternion_xyzw'])
            assert (actual*previous.inv()).magnitude()==pytest.approx(.001*index)
    assert calls['builds']==[2,4,6,8,10,12]


def test_scan_geometry_and_top_candidates_are_observed_and_preserve_calibrated_mount(scan_hub,tmp_path):
    hub,state,_,clearance_type=scan_hub
    geometry,points,above=table_geometry(surfaces()+[.6,-.4,.3],tmp_path/'geometry.npz')
    centre=np.asarray(geometry['table_center_world_m'])
    np.testing.assert_allclose(centre[:2],(.7,-.2),atol=.03)
    assert centre[2]==pytest.approx(1.25,abs=.01)
    assert geometry['centre_is_observed_rectangle_not_gt']
    wrist=hub.robot.camera('robot0_eye_in_hand')
    candidates=scan_candidates(state,wrist,geometry,points,above,None,clearance_type())
    assert len(candidates)==3
    eef=np.eye(4);eef[:3,:3]=Rotation.from_quat(state['quaternion_xyzw']).as_matrix();eef[:3,3]=state['position']
    mount=np.linalg.inv(wrist['world_from_camera'])@eef
    for candidate in candidates:
        camera=np.asarray(candidate['world_from_camera']);desired=camera@mount
        np.testing.assert_allclose(camera[:3,2],[0.,0.,-1.],atol=1e-12)
        np.testing.assert_allclose(desired[:3,3],candidate['position'],atol=1e-12)
        np.testing.assert_allclose(desired[:3,:3],Rotation.from_quat(candidate['quaternion_xyzw']).as_matrix(),atol=1e-12)
        assert 0<=candidate['table_frustum_fraction']<=1 and 0<=candidate['above_table_frustum_fraction']<=1


def test_unreachable_directions_respect_candidate_budget_and_keep_initial_known_images(scan_hub,monkeypatch):
    hub,_,calls,base=scan_hub
    class Reject(base):
        def plan(self,state,candidate):
            calls['plans'].append(candidate);raise ValueError('No observed-clearance route')
    monkeypatch.setattr('sketchik.observed_clearance.ObservedClearance',Reject)
    result,pictures=run_table_scan(hub,max_ik_candidates=2)
    assert result['status']=='completed_with_skipped_views' and result['captured_views']==0
    assert len(calls['plans'])==10 and calls['moves']==[] and len(pictures)==2
    assert all(item['status']=='skipped_no_checked_view' and len(item['checks'])==2 for item in result['attempts'])


def test_oblique_candidate_budget_covers_near_middle_and_far_distance_families(scan_hub,tmp_path):
    hub,state,_,clearance_type=scan_hub
    geometry,points,above=table_geometry(surfaces(),tmp_path/'geometry.npz')
    wrist=hub.robot.camera('robot0_eye_in_hand')
    candidates=scan_candidates(state,wrist,geometry,points,above,np.array([1.,0.]),clearance_type())
    assert len(candidates)==9
    assert len({item['standoff_m'] for item in candidates[:3]})==3
    assert min(item['standoff_m'] for item in candidates[:3])==min(item['standoff_m'] for item in candidates)
    for candidate in candidates:
        expected=np.linalg.norm(np.asarray(candidate['position'])-state['position'])
        assert candidate['estimated_direct_eef_travel_m']==pytest.approx(expected)


def test_oblique_priority_promotes_one_reliable_view_and_preserves_distance_fallbacks():
    def candidate(name,distance,elevation,score,**coverage):
        return {'name':name,'standoff_m':distance,'elevation_rad':np.deg2rad(elevation),'score':score,
            'table_frustum_fraction':1.,'above_table_frustum_fraction':1.,
            'known_surface_visible_fraction':.9,**coverage}
    # A near high-angle view can be the only reachable fallback. It must remain
    # in the first round even when a later middle-distance slanted view is tried.
    proposals=[candidate('middle_high',.7,80,1.4),candidate('far_high',.9,80,1.38),
        candidate('near_high',.5,80,1.3),candidate('middle_medium',.7,65,1.35),
        candidate('far_medium',.9,65,1.33),candidate('near_medium',.5,65,1.2),
        candidate('middle_low',.7,50,1.32),candidate('far_low',.9,50,1.28),
        candidate('near_low',.5,50,1.25,above_table_frustum_fraction=.6)]
    original=list(proposals)
    ordered=_promote_oblique_candidate(proposals)
    assert ordered[0]['name']=='middle_low'
    assert [item['standoff_m'] for item in ordered[:3]]==[.7,.9,.5]
    assert ordered[2] is original[2] and ordered[6] is original[0]
    assert {id(item) for item in ordered}=={id(item) for item in original}
    assert sum(current is not previous for current,previous in zip(ordered,original))==2


@pytest.mark.parametrize('metric,value',[
    ('table_frustum_fraction',.949),('above_table_frustum_fraction',.899),
    ('known_surface_visible_fraction',.799)])
def test_oblique_priority_does_not_promote_poorly_observed_views(metric,value):
    high={'standoff_m':.7,'elevation_rad':np.deg2rad(80),'score':1.4,
        'table_frustum_fraction':1.,'above_table_frustum_fraction':1.,
        'known_surface_visible_fraction':.9}
    low={**high,'elevation_rad':np.deg2rad(50),'score':1.3,metric:value}
    ordered=_promote_oblique_candidate([high,low])
    assert ordered[0] is high and ordered[1] is low


def test_insufficient_table_geometry_returns_initial_images_and_no_fake_pose(scan_hub,monkeypatch):
    from sketchik.grounding_packet import ObservedSupportUnavailable
    hub,_,calls,_=scan_hub
    def unavailable(*args):raise ObservedSupportUnavailable('No broad observed support')
    monkeypatch.setattr('sketchik.table_scan.table_window',unavailable)
    result,pictures=run_table_scan(hub)
    assert result['status']=='initial_geometry_unavailable' and len(pictures)==2
    assert len(result['attempts'])==5 and calls['moves']==calls['plans']==[]
    assert all('selected' not in item and item['status']=='skipped_geometry_unavailable' for item in result['attempts'])


@pytest.mark.parametrize('exception', [OSError('disk failure'), TypeError('internal type error'),
                                      RuntimeError('internal controller failure')])
@pytest.mark.parametrize('phase', ['initial_geometry', 'candidate_generation', 'motion', 'capture_after_move'])
def test_unexpected_scan_program_fault_propagates_and_stops_future_physics(
        scan_hub, monkeypatch, exception, phase):
    hub, state, calls, _ = scan_hub
    def fail(*args, **kwargs):
        raise exception
    if phase == 'initial_geometry':
        monkeypatch.setattr('sketchik.table_scan.table_geometry', fail)
    elif phase == 'candidate_generation':
        monkeypatch.setattr('sketchik.table_scan.scan_candidates', fail)
    elif phase == 'motion':
        original = hub.executor.move_waypoints
        def moved_then_failed(*args, **kwargs):
            original(*args, **kwargs)
            raise exception
        hub.executor.move_waypoints = moved_then_failed
    else:
        original = hub.observe_scene
        def captured_then_failed(*args, **kwargs):
            original(*args, **kwargs)
            raise exception
        hub.observe_scene = captured_then_failed
    with pytest.raises(type(exception)) as caught:
        run_table_scan(hub)
    assert caught.value is exception
    expected_moves = 1 if phase in ('motion', 'capture_after_move') else 0
    assert len(calls['moves']) == expected_moves
    assert hub.robot.steps == 30 + expected_moves*3
    assert not any(event.get('event') == 'table_scan_finished' for event in calls['events'])


def test_numerical_scan_ik_failure_stops_with_actual_picture(scan_hub):
    from sketchik.kinematics import IKError
    hub, _, calls, _ = scan_hub
    def fail(*args, **kwargs):
        raise IKError('expected checked-route numerical rejection')
    hub.executor.move_waypoints = fail
    result, pictures = run_table_scan(hub)
    assert result['status'] == 'stopped_after_view_failure'
    assert result['attempts'][0]['status'] == 'motion_failed'
    assert calls['captures'] == 1 and calls['moves'] == []
    assert len(pictures) == 4


def test_nonconverged_view_keeps_actual_picture_and_stops_suffix(scan_hub):
    hub,_,calls,_=scan_hub;move=hub.executor.move_waypoints
    def fail(waypoints):return {**move(waypoints),'converged':False}
    hub.executor.move_waypoints=fail
    result,pictures=run_table_scan(hub)
    assert result['status']=='stopped_after_view_failure' and result['captured_views']==1
    assert result['attempts'][0]['status']=='motion_not_converged'
    assert result['unattempted_directions']==['front','back','left','right']
    assert calls['captures']==len(calls['moves'])==1 and len(pictures)==4


def test_table_scan_captures_missing_initial_observation_before_generating_views(scan_hub):
    hub,_,calls,_=scan_hub;hub.observations={}
    result,pictures=run_table_scan(hub)
    assert result['initial_observation_id']=='obs_00000' and result['captured_views']==5
    assert calls['captures']==6 and len(pictures)==12
    assert result['sources'][0]['observation_id']=='obs_00000'


@pytest.mark.parametrize('budget',[0,True,1.5])
def test_invalid_candidate_budget_cannot_send_motion(scan_hub,budget):
    hub,_,calls,_=scan_hub
    with pytest.raises(ValueError,match='positive integer'):run_table_scan(hub,max_ik_candidates=budget)
    assert calls['moves']==[]
