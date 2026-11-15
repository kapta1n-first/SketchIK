"""Complementary camera evidence, not semantic or collision certification."""
from copy import deepcopy
from pathlib import Path
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from sketchik.guided_observation import (projection_quality,proposal_views,
    measured_patch_quality,run_guided_observation)
from sketchik.scene_catalog import binding_schema,bind_requirements,public_catalog
from jsonschema import Draft202012Validator,ValidationError

HAND_BOUNDS=[[-.104,-.032,-.123],[.104,.032,.01]]


def component(tmp_path):
    x,y=np.meshgrid(np.linspace(-.1,.1,30),np.linspace(-.1,.1,30))
    points=np.column_stack([x.ravel(),y.ravel(),np.full(x.size,1.)])
    points=np.vstack([points,points+[0,0,.2]])
    path=tmp_path/'points.npz';np.savez(path,world_xyz_m=points)
    patch={'candidate_id':'floor','observation_id':'old','camera':'agentview',
        'bounds_xyxy':[100,100,300,300],'world_height_m':1.,'look_at_world_m':[0,0,1.],
        'observed_span_xyz_m':[.2,.2,0.],'visible_pixel_sample_count':500}
    return {'component_id':'G1','point_cloud_file':str(path),'point_count':len(points),
        'observed_bounds_world_m':[[-.1,-.1,1.],[.1,.1,1.2]],'observed_span_xyz_m':[.2,.2,.2],
        'sources':[{'observation_id':'old','camera':'agentview','source_step':10,
            'bounds_xyxy':[1,1,50,50],'point_count':100}], 'patches':[patch],
        'source_steps':[10],'last_observed_step':10,
        'semantic_identity_verified':False,'complete_dimensions_known':False}


def test_visibility_scoring_accounts_for_unrelated_foreground_obstacle():
    x,y=np.meshgrid(np.linspace(-.1,.1,30),np.linspace(-.1,.1,30))
    points=np.column_stack([x.ravel(),y.ravel(),np.ones(x.size)])
    K=np.array([[500.,0,512],[0,500.,512],[0,0,1.]])
    free=projection_quality(points,points,np.eye(4),K)
    occluded=projection_quality(points,np.vstack([points,points*.5]),np.eye(4),K)
    assert free['visible_cells']>100 and free['quality']>0
    assert occluded['visible_cells']==0 and occluded['quality']==0
    assert not free['unknown_occluders_certified']


def test_proposals_preserve_actual_camera_mount_and_spread_bearings(tmp_path):
    c=component(tmp_path)
    camera=np.eye(4);camera[:3,:3]=np.diag([1.,-1.,-1.]);camera[:3,3]=[0,0,1.5]
    wrist={'world_from_camera':camera,'K':np.array([[500.,0,512],[0,500.,512],[0,0,1.]]),
        'depth_m':np.ones((1024,1024)),'step':10}
    state={'position':[.02,0,1.55],'quaternion_xyzw':Rotation.from_euler('xyz',[.3,.2,.1]).as_quat(),'step':10}
    eef=np.eye(4);eef[:3,:3]=Rotation.from_quat(state['quaternion_xyzw']).as_matrix();eef[:3,3]=state['position']
    mount=np.linalg.inv(camera)@eef
    with np.load(c['point_cloud_file']) as data:points=data['world_xyz_m']
    proposals=proposal_views(state,wrist,[c],'interior_support',points,hand_bounds=HAND_BOUNDS)
    assert len({v['bearing_family'] for v in proposals[:8]})>=4
    for v in proposals:
        actual=np.asarray(v['world_from_camera'])@mount
        np.testing.assert_allclose(actual[:3,3],v['position'],atol=1e-9)
        np.testing.assert_allclose(actual[:3,:3],Rotation.from_quat(v['quaternion_xyzw']).as_matrix(),atol=1e-9)
        assert abs(v['world_from_camera'][2][0])<1e-9
        assert not v['collision_checked'] and v['semantic_surface_status']=='unconfirmed'
        assert v['look_at_world_m']==[0,0,1.]
    with pytest.raises(ValueError,match='physical step'):proposal_views(
        {**state,'step':11},wrist,[c],'interior_support',points,hand_bounds=HAND_BOUNDS)


def test_small_floor_patch_does_not_frame_hand_inside_tall_container(tmp_path):
    """The cavity floor can fit in the image while the hand hits a nearby rim."""
    c=component(tmp_path)
    camera=np.eye(4);camera[:3,:3]=np.diag([1.,-1.,-1.]);camera[:3,3]=[0,0,1.6]
    wrist={'world_from_camera':camera,'K':np.array([[667.,0,512],[0,667.,512],[0,0,1.]]),
        'depth_m':np.ones((1024,1024)),'step':10}
    state={'position':[0,0,1.5],'quaternion_xyzw':[1,0,0,0],'step':10}
    with np.load(c['point_cloud_file']) as data:points=data['world_xyz_m']
    proposals=proposal_views(state,wrist,[c],'interior_support',points,hand_bounds=HAND_BOUNDS)
    assert proposals
    from itertools import product
    corners=np.asarray(list(product(*zip(*HAND_BOUNDS))))
    for view in proposals:
        aim=np.asarray(view['look_at_world_m']);eye=np.asarray(view['camera_position_m'])
        bearing=(eye-aim)/np.linalg.norm(eye-aim)
        hand=corners@Rotation.from_quat(view['quaternion_xyzw']).as_matrix().T+view['position']
        gap=((hand-aim)@bearing).min()-((points-aim)@bearing).max()
        assert gap>=.03-1e-9
        assert view['hand_target_projection_gap_m']==pytest.approx(gap)
        assert not view['collision_checked']
    # FOV-only nominal distance is .25 m for the small lower patch. The
    # full-container/hand extent must move at least one view farther away.
    assert any(v['framing_distance_m']>.35 for v in proposals)


def test_roof_only_measurement_does_not_count_as_interior_evidence(tmp_path):
    c=component(tmp_path);c['patches'][0]['world_height_m']=1.2
    assert measured_patch_quality([c])==0
    c['patches'][0]['world_height_m']=1.
    assert measured_patch_quality([c])==1


def catalog(tmp_path):
    c=component(tmp_path)
    return {'source_step':10,'source_steps':[10],'scope':'all observed tabletop fragments',
        'table_sources':[{'observation_id':'old','camera':'agentview','source_step':10}],
        'components':{'G1':c,'G2':{**deepcopy(c),'component_id':'G2'}},
        'candidates':{},'point_cloud_file':c['point_cloud_file'],'table_height_m':.8}


def test_many_to_many_binding_retains_non_task_groups_and_stale_bounds(tmp_path):
    cat=catalog(tmp_path)
    requirements={'source_step':10,'objects':[{'object_id':'part_'+str(i),'component_ids':['G1'],
        'description':'distinct part','interactions':['touch'],'required_surfaces':['contact_surface'],
        'missing_surfaces':[]} for i in range(2)]}
    Draft202012Validator(binding_schema(cat)).validate(requirements)
    handoff=bind_requirements(requirements,cat)
    assert all(v['evidence'][0]['bounds_xyxy']==[1,1,50,50] for v in handoff['objects'])
    public=public_catalog(cat,objects=handoff['objects'],journal={'objects':{'part_0':{'invalidated_at_step':20}}})
    assert len(public['components'])==2 and public['components'][1]['semantic_aliases']==[]
    assert public['components'][0]['pose_validity']=='uncertain_after_contact'
    assert not public['collision_free_certified']
    assert public['components'][0]['observed_span_xyz_m']==[.2,.2,.2]
    assert not handoff['coverage_complete']
    with pytest.raises(ValidationError):
        Draft202012Validator(binding_schema(cat)).validate({**requirements,'coverage_complete':True})


@pytest.fixture
def guided_hub(tmp_path,monkeypatch):
    cat=catalog(tmp_path);(tmp_path/'scene_catalog').mkdir()
    cat['components']['G1']['patches']=[]
    calls=[];state={'step':10,'position':[0,0,1.5],'quaternion_xyzw':[0,0,0,1],'q':[0]*7}
    robot=SimpleNamespace(steps=10,state=lambda:deepcopy(state),camera=lambda name:{'step':state['step'],'camera':name})
    def move(**kw):
        assert kw['position_tolerance_m']==.012
        calls.append('move');robot.steps+=5;state['step']=robot.steps;return {'converged':True}
    hub=SimpleNamespace(run=tmp_path,robot=robot,ik_only=True,emit=lambda event:None,
        executor=SimpleNamespace(ik=SimpleNamespace(fk=lambda q:(np.zeros(3),np.eye(3))),move=move),
        observe_scene=lambda:({'observation_id':'new'},['actual image']))
    class Clearance:
        def __init__(self,*args):pass
        def add_frame(self,*args):pass
        def build(self):pass
    monkeypatch.setattr('sketchik.observed_clearance.ObservedClearance',Clearance)
    monkeypatch.setattr('sketchik.guided_observation.hand_bounds_in_eef',lambda *args:HAND_BOUNDS)
    candidates=[{'position':[0,0,1.2],'quaternion_xyzw':[0,0,0,1],
        'camera_position_m':[0,.3,1.2],'predicted_quality':.5} for i in range(9)]
    monkeypatch.setattr('sketchik.guided_observation.proposal_views',lambda *args,**kwargs:candidates)
    handoff=bind_requirements({'source_step':10,'objects':[{'object_id':'inside','component_ids':['G1'],
        'description':'container inside','interactions':['place'],'required_surfaces':['interior_support']} ]},cat)
    return hub,cat,handoff,calls


def test_ik_rejection_budget_never_dispatches_an_unchecked_view(guided_hub,monkeypatch):
    hub,cat,handoff,calls=guided_hub
    def reject(*args):calls.append('IK');raise ValueError('not reachable')
    monkeypatch.setattr('sketchik.guided_observation.camera_route',reject)
    result,images=run_guided_observation(hub,cat,handoff,max_ik_candidates=2)
    assert calls==['IK','IK'] and result['captured_views']==0 and images==[]
    assert result['attempts'][0]['acquisition_status']=='unreachable'
    assert not result['collision_checked'] and not result['semantic_coverage_certified']


def test_no_gain_stops_extra_cameras_but_still_requires_a_review(guided_hub,monkeypatch):
    hub,cat,handoff,calls=guided_hub
    def route(*args):
        assert args[0].ik_only
        return {'waypoints':[{'position':[0,0,1.2],'quaternion_xyzw':[0,0,0,1]}]}
    monkeypatch.setattr('sketchik.guided_observation.camera_route',route)
    monkeypatch.setattr('sketchik.guided_observation.refresh_component_evidence',lambda *args:(np.zeros((1,3)),{}))
    result,images=run_guided_observation(hub,cat,handoff)
    assert calls==['move'] and result['captured_views']==1 and images==['actual image']
    assert result['attempts'][0]['acquisition_status']=='no_gain_ready_for_review'
    assert not result['semantic_coverage_certified']
    assert (hub.run/'scene_catalog/catalog_after_acquisition.json').exists()


def test_program_exception_propagates_without_motion_or_api_retry(guided_hub,monkeypatch):
    hub,cat,handoff,calls=guided_hub
    def bug(*args):raise RuntimeError('broken code')
    monkeypatch.setattr('sketchik.guided_observation.camera_route',bug)
    with pytest.raises(RuntimeError,match='broken code'):run_guided_observation(hub,cat,handoff)
    assert calls==[]


def test_catalog_omission_fallback_uses_saved_source_points_without_deleting_other_groups(tmp_path):
    cat=catalog(tmp_path)
    with np.load(cat['point_cloud_file']) as data:points=data['world_xyz_m']
    np.savez(cat['point_cloud_file'],world_xyz_m=points,pixel_xy=np.tile([20,20],(len(points),1)),
        source_index=np.zeros(len(points),int))
    requirements={'source_step':10,'objects':[{'object_id':'part','component_ids':[],
        'description':'unlisted part','interactions':['touch'],'required_surfaces':['contact_surface'],
        'source_regions':[{'observation_id':'old','camera':'agentview','bounds_xyxy':[10,10,30,30]}]}]}
    Draft202012Validator(binding_schema(cat)).validate(requirements)
    result=bind_requirements(requirements,cat)
    assert len(cat['components'])==3 and result['objects'][0]['component_ids']==['F3']
    assert cat['components']['F3']['source_steps']==[10]
    assert not cat['components']['F3']['semantic_identity_verified']


def test_raised_camera_transfer_checks_all_segments_with_successive_joint_seeds(monkeypatch):
    from sketchik.guided_observation import camera_route
    calls=[]
    def checked(hub,clearance,state,target):
        calls.append((deepcopy(state),deepcopy(target)))
        return {'mode':'ik_only','waypoints':[target],'final_joint_positions':[len(calls)]*7,
            'path_checked':True,'ik_samples':2,'collision_checked':False}
    monkeypatch.setattr('sketchik.guided_observation.motion_route',checked)
    hub=SimpleNamespace(ik_only=True)
    state={'position':[0,0,1.],'quaternion_xyzw':[0,0,0,1],'q':[0]*7}
    pose={'position':[.2,.1,1.1],'quaternion_xyzw':[1,0,0,0]}
    result=camera_route(hub,None,state,pose,1.4)
    assert [v['position'][2] for v in result['waypoints']]==[1.4,1.4,1.1]
    assert [v[0]['q'][0] for v in calls]==[0,1,2]
    assert not result['collision_checked'] and not result['continuous_collision_certified']


def test_catalog_baseline_points_and_group_labels_survive_final_table_rebuild(tmp_path,monkeypatch):
    from sketchik.scene_catalog import build_scene_catalog
    folder=tmp_path/'table_context';folder.mkdir()
    x,y=np.meshgrid(np.linspace(0,.1,15),np.linspace(0,.1,15))
    points=np.column_stack([x.ravel(),y.ravel(),np.full(x.size,1.)])
    cloud=folder/'observed_table.npz'
    np.savez(cloud,world_xyz_m=points,rgb=np.zeros((len(points),3),np.uint8),
        pixel_xy=np.tile([20,20],(len(points),1)),source_index=np.zeros(len(points),int))
    table={'point_cloud_file':str(cloud),'sources':[{'observation_id':'old','camera':'agentview','source_step':10}],
        'source_steps':[10],'observed_table_height_m':.8}
    (folder/'context.json').write_text(json.dumps(table))
    monkeypatch.setattr('sketchik.perception.saved_frame',lambda *args:{})
    monkeypatch.setattr('sketchik.surface_evidence.summarize_horizontal_surfaces',lambda *args:{'horizontal_patches':[]})
    hub=SimpleNamespace(run=tmp_path,robot=SimpleNamespace(steps=10),emit=lambda *args:None)
    result=build_scene_catalog(hub)
    np.savez(cloud,world_xyz_m=points[:1])
    np.savez(folder/'display_components.npz',point_group_ids=[0])
    with np.load(result['point_cloud_file']) as saved:np.testing.assert_array_equal(saved['world_xyz_m'],points)
    with np.load(result['component_labels_file']) as labels:
        assert len(labels['point_group_ids'])==len(points)
    assert Path(result['component_labels_file']).parent==tmp_path/'scene_catalog'
