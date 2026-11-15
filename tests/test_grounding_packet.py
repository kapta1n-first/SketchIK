import numpy as np
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from sketchik.grounding_packet import table_window, focus_bounds, in_window


def test_table_display_uses_observed_support_instead_of_largest_remote_floor():
    # The distant floor is much broader but has less observed support. Never
    # hardcode a particular table height or task workspace rectangle.
    x,y=np.meshgrid(np.linspace(-.4,.2,80),np.linspace(.1,.7,80))
    table=np.column_stack([x.ravel(),y.ravel(),np.full(x.size,1.2)])
    x,y=np.meshgrid(np.linspace(2,4,60),np.linspace(2,4,60))
    floor=np.column_stack([x.ravel(),y.ravel(),np.zeros(x.size)])
    height,low,high=table_window(np.concatenate([table,floor]))
    assert abs(height-1.2)<1e-6
    assert np.all(low<[-.39,.11]) and np.all(high>[.19,.69])
    assert np.all(high<1.)


def test_local_display_crop_does_not_add_or_move_observed_points():
    rng=np.random.default_rng(31)
    points=rng.uniform([-.1,.2,.8],[.1,.4,1.],(1000,3))
    bounds=focus_bounds(np.r_[points,[[20,20,20]]])
    chosen=points[in_window(points,bounds)]
    assert np.array_equal(chosen,points)
    assert not in_window(np.array([[20,20,20]]),bounds)[0]


@pytest.fixture
def fresh_table(tmp_path,monkeypatch):
    from PIL import Image
    records=[];frames={};states={};original={}
    for index,(identity,step) in enumerate([('obs_initial',10),('obs_periodic',20),('obs_action_end',30)]):
        folder=tmp_path/'observations'/identity;folder.mkdir(parents=True)
        state={'step':step,'q':[index/10.]*7,'finger_joint_positions':[step/1000.,-step/1000.]}
        cameras=[];states[identity]=state
        for camera_index,camera in enumerate(('agentview','robot0_eye_in_hand')):
            k=np.array([[200.+10*index,0,64.],[0,200.+10*index,64.],[0,0,1.]])
            transform=np.eye(4);transform[:3,3]=[index*.04,camera_index*.02,0.]
            rgb=np.full((128,128,3),30+index*30+camera_index,np.uint8)
            frame={'rgb':rgb,'depth_m':np.ones((128,128),np.float32),
                   'K':k,'world_from_camera':transform,'step':step}
            frames[identity,camera]=frame
            Image.fromarray(rgb).save(folder/(camera+'.png'))
            np.savez(folder/(camera+'.npz'),**{key:frame[key] for key in ['depth_m','K','world_from_camera']})
            cameras.append({'camera':camera,'K':k.tolist(),'world_from_camera':transform.tolist()})
        (folder/'observation.json').write_text(json.dumps({'source_step':step,'state':state,'cameras':cameras}))
        original.update({path:path.read_bytes() for path in folder.iterdir()})
    class Filter:
        def __init__(self,*args):pass
        def non_robot_mask(self,points,state):
            records.append(('filter',state,len(points)))
            return np.ones(len(points),bool)
        def set_state(self,state):records.append(('restore',state))
    monkeypatch.setattr('sketchik.observed_clearance.ObservedClearance',Filter)
    keyframes={identity:{'observation_id':identity,'source_step':states[identity]['step']}
               for identity in ('obs_initial','obs_action_end')}
    hub=SimpleNamespace(run=tmp_path,journal=SimpleNamespace(data={'scene_clouds':keyframes}),
        executor=SimpleNamespace(ik=None),robot=SimpleNamespace(state=lambda:states['obs_action_end']),
        observations={'unrelated_live_frame':{'world_from_camera':np.full((4,4),999.)}})
    return hub,frames,states,records,original


def test_fresh_table_uses_only_real_scene_keyframes_and_each_saved_calibration(fresh_table):
    from sketchik.grounding_packet import prepare_table_context
    hub,frames,states,records,original=fresh_table
    result=prepare_table_context(hub)
    manifest=json.loads((hub.run/'scan_evidence.json').read_text())
    expected=[(identity,camera) for identity in ('obs_initial','obs_action_end')
              for camera in ('agentview','robot0_eye_in_hand')]
    assert [(item['observation_id'],item['camera']) for item in manifest['sources']]==expected
    assert manifest['derived_selection'] and manifest['periodic_rgbd_history_preserved']
    assert manifest['full_observation_history_preserved'] and manifest['original_scene_clouds_preserved']
    assert 'periodic-only frames excluded' in manifest['selection']
    assert result['source_steps']==[10,30] and result['source_state_and_calibration_matched']
    assert [item[1] for item in records[:-1]]==[states[identity] for identity,_ in expected]
    assert all(item[2]==64*64 for item in records[:-1])
    assert records[-1]==('restore',states['obs_action_end'])
    with np.load(result['point_cloud_file']) as cloud:
        for index,(identity,camera) in enumerate(expected):
            selected=cloud['source_index']==index
            assert selected.any()
            uv=cloud['pixel_xy'][selected]
            assert np.all(uv%2==0)
            frame=frames[identity,camera]
            rays=np.linalg.solve(frame['K'],np.c_[uv,np.ones(len(uv))].T).T
            expected_xyz=rays@frame['world_from_camera'][:3,:3].T+frame['world_from_camera'][:3,3]
            np.testing.assert_allclose(cloud['world_xyz_m'][selected],expected_xyz,atol=1e-7)
            np.testing.assert_array_equal(cloud['rgb'][selected],frame['rgb'][uv[:,1],uv[:,0]])
    assert len(result['views'])==3 and all(Path(view['image_file']).exists() for view in result['views'])
    assert all(view['mode']=='rgb' and not view['selectable'] for view in result['views'])
    for path,raw in original.items():assert path.read_bytes()==raw


def test_existing_table_manifest_keeps_its_explicit_sources(fresh_table):
    from sketchik.grounding_packet import prepare_table_context
    hub,_,_,records,_=fresh_table
    manifest=hub.run/'scan_evidence.json'
    raw=json.dumps({'sources':[{'observation_id':'obs_initial','camera':'agentview'}]})
    manifest.write_text(raw)
    result=prepare_table_context(hub)
    assert [(item['observation_id'],item['camera']) for item in result['sources']]==[('obs_initial','agentview')]
    assert manifest.read_text()==raw and len(records)==2


@pytest.mark.parametrize('mismatch',['state_step','calibration','scene_step'])
def test_fresh_table_rejects_mismatched_source_state_or_calibration(fresh_table,mismatch):
    from sketchik.grounding_packet import prepare_table_context
    hub,_,_,_,_=fresh_table
    path=hub.run/'observations/obs_action_end/observation.json'
    metadata=json.loads(path.read_text())
    if mismatch=='state_step':metadata['state']['step']-=1
    elif mismatch=='calibration':metadata['cameras'][0]['world_from_camera'][0][3]+=.1
    else:hub.journal.data['scene_clouds']['obs_action_end']['source_step']-=1
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError,match='source step|calibration'):prepare_table_context(hub)
    assert not (hub.run/'scan_evidence.json').exists()
    assert not (hub.run/'table_context/context.json').exists()


def test_fresh_table_without_scene_keyframes_does_not_select_periodic_history(tmp_path):
    from sketchik.grounding_packet import prepare_table_context
    hub=SimpleNamespace(run=tmp_path,journal=SimpleNamespace(data={'observations':{'obs_periodic':{'source_step':20}}}))
    assert prepare_table_context(hub) is None
    assert not (tmp_path/'scan_evidence.json').exists() and not (tmp_path/'table_context').exists()


@pytest.mark.parametrize('moved',[False,True])
def test_optional_coarse_region_leaves_local_cloud_unavailable_without_guessing(tmp_path,monkeypatch,moved):
    from sketchik.grounding_packet import build_focused_context
    def no_projection(*args,**kwargs):
        pytest.fail('No region was supplied; local geometry must not be projected or guessed')
    monkeypatch.setattr('sketchik.grounding_packet.frame_points',no_projection)
    record={'invalidated_at_step':15} if moved else {}
    hub=SimpleNamespace(journal=SimpleNamespace(data={'objects':{'target':record}}))
    handoff={'objects':[{'object_id':'target','evidence':[
        {'observation_id':'old','camera':'agentview','visible_evidence':'visible target without a coarse box'}]}]}
    folder=tmp_path/'local_geometry'
    result=build_focused_context(hub,handoff,'target',{'observation_id':'current'},folder,active_views_only=True)
    assert not result['local_context_available']
    assert result['unavailable_reason']=='no_coarse_observation_region'
    assert result['selectable_views']==[] and result['cloud_image'] is None
    assert result['world_display_min_m'] is None and result['world_display_max_m'] is None
    assert result['historical_points']==result['current_points']==0
    assert result['historical_geometry_may_be_stale'] is moved
    assert not result['historical_and_current_fused'] and not result['hidden_surface_completion']
    assert not result['known_robot_removed'] and not result['object_gt_used']
    assert json.loads((folder/'context.json').read_text())==result
    assert {path.name for path in folder.iterdir()}=={'context.json'}




@pytest.mark.parametrize('moved,group',[(False,'historical'),(True,'current')])
def test_active_views_keep_both_measured_point_files(tmp_path,monkeypatch,moved,group):
    from sketchik.grounding_packet import build_focused_context
    frame={'rgb':np.full((32,32,3),100,np.uint8),'depth_m':np.ones((32,32)),
        'K':np.array([[200,0,16],[0,200,16],[0,0,1]]),'world_from_camera':np.eye(4),'step':10}
    current={**frame,'step':20}
    monkeypatch.setattr('sketchik.perception.saved_frame',lambda *args:frame)
    class Filter:
        def __init__(self,*args):pass
        def set_state(self,*args):pass
        def non_robot_mask(self,points,state):return np.ones(len(points),bool)
    monkeypatch.setattr('sketchik.observed_clearance.ObservedClearance',Filter)
    folder=tmp_path/'observations/old';folder.mkdir(parents=True)
    (folder/'observation.json').write_text(json.dumps({'state':{}}))
    record={'invalidated_at_step':15} if moved else {}
    state={'position':[0,0,1]}
    hub=SimpleNamespace(run=tmp_path,journal=SimpleNamespace(data={'objects':{'target':record}}),
        executor=SimpleNamespace(ik=None),robot=SimpleNamespace(state=lambda:state),
        observations={'now':{'agentview':current,'robot0_eye_in_hand':current}})
    handoff={'objects':[{'object_id':'target','evidence':[{'observation_id':'old','camera':'agentview','bounds_xyxy':[0,0,32,32]}]}]}
    result=build_focused_context(hub,handoff,'target',{'observation_id':'now','state':state},tmp_path/'pack',active_views_only=True)
    assert len(result['selectable_views'])==3
    assert all(v['mode']=='rgb' for v in result['selectable_views'])
    assert all('_'+group+'_' in v['view_id'] and v['selectable'] for v in result['selectable_views'])
    for name in ['historical_surfaces','current_surfaces']:
        with np.load(tmp_path/'pack'/(name+'.npz')) as data:assert len(data['world_xyz_m'])>32
    assert result['historical_and_current_fused'] is False
    measured=json.loads((tmp_path/'dimensions.json').read_text())['objects']['target']
    key='current_observed_shape' if moved else 'initial_observed_shape'
    assert measured[key]['geometry_group']==group
    assert not measured[key]['semantic_identity_verified']
    assert measured[key]['components']
    with np.load(tmp_path/'pack/extent_groups.npz') as data:
        assert len(data['point_group_ids'])==result['current_points' if moved else 'historical_points']
    if not moved:
        from sketchik.observed_dimensions import carried_dimensions,write_measurement
        before=measured['initial_observed_shape']
        write_measurement(tmp_path,'target',carried_dimensions(np.ones((43,3)),40))
        after=json.loads((tmp_path/'dimensions.json').read_text())['objects']['target']
        assert after['initial_observed_shape']==before
        assert after['dimension_source_step']==40
