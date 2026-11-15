import json
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from jsonschema import Draft202012Validator,ValidationError









@pytest.mark.parametrize('with_opening',[False,True])
@pytest.mark.parametrize('focus_role',['interior_support','contact_surface','overall_shape'])
def test_required_focus_points_survive_old_exterior_crop_and_table_display_sampling(tmp_path,monkeypatch,with_opening,focus_role):
    from sketchik.scene_cloud import frame_points
    from sketchik.grounding_packet import build_focused_context
    frame={'step':10,'rgb':np.full((32,32,3),100,np.uint8),
        'depth_m':np.ones((32,32)), 'K':np.array([[200.,0,16],[0,200.,16],[0,0,1.]]),
        'world_from_camera':np.eye(4)}
    points,colors,pixels=frame_points(frame)
    table=tmp_path/'table_context';table.mkdir()
    selected=(pixels[:,0]%2==0)&(pixels[:,1]%2==0)
    cloud=table/'points.npz'
    np.savez(cloud,world_xyz_m=points[selected],rgb=colors[selected],pixel_xy=pixels[selected],
        source_index=np.zeros(selected.sum(),np.uint16))
    (table/'context.json').write_text(json.dumps({'point_cloud_file':str(cloud),
        'sources':[{'observation_id':'support','camera':'agentview','source_step':10}],
        'observed_table_height_m':.8,'display_xy_min_m':[-1,-1],'display_xy_max_m':[1,1]}))
    obs=tmp_path/'observations/support';obs.mkdir(parents=True)
    (obs/'observation.json').write_text(json.dumps({'state':{}}))
    old_obs=tmp_path/'observations/old';old_obs.mkdir(parents=True)
    (old_obs/'observation.json').write_text(json.dumps({'state':{}}))
    monkeypatch.setattr('sketchik.perception.saved_frame',lambda *args:frame)
    class Filter:
        def __init__(self,*args):pass
        def non_robot_mask(self,xyz,state):return np.ones(len(xyz),bool)
        def set_state(self,*args):pass
    monkeypatch.setattr('sketchik.observed_clearance.ObservedClearance',Filter)
    hub=SimpleNamespace(run=tmp_path,journal=SimpleNamespace(data={'objects':{'target':{}}}),
        executor=SimpleNamespace(ik=None),robot=SimpleNamespace(state=lambda:{'position':[0,0,0]}),
        observations={'now':{'agentview':{**frame,'step':20}}})
    support={'observation_id':'support','camera':'agentview','surface_role':focus_role,
        'bounds_xyxy':[0,0,32,32]}
    old={'observation_id':'old','camera':'agentview','surface_role':'overall_shape','bounds_xyxy':[8,8,24,24]}
    evidence=[old,support]
    opening_calls=[]
    if with_opening:
        evidence.append({**old,'surface_role':'opening','rim_pixels_xy':[[8,8],[23,8],[23,23]]})
        def opening(*args):
            opening_calls.append(True)
            # The deliberately opposite normal must not govern an exterior contact.
            return {'proposed_surface_normal_world':[0,0,1 if focus_role=='contact_surface' else -1]},None
        monkeypatch.setattr('sketchik.program_scan.describe_opening',opening)
    handoff={'objects':[{'object_id':'target','evidence':evidence}]}
    out=tmp_path/'focus'
    result=build_focused_context(hub,handoff,'target',{'observation_id':'now','state':{}},out,
        active_views_only=True,selected_view=support)
    assert result['focus_reference']==support
    assert len(result['selectable_views'])==3
    with np.load(out/'historical_surfaces.npz') as data:
        assert len(data['world_xyz_m'])>=32*32
        assert [31,31] in data['pixel_xy'].tolist()
        np.testing.assert_allclose(data['world_xyz_m'].max(axis=0),points.max(axis=0))
    assert result['focus_visibility'][0]['labelled_source_visible_pixels']>=32
    assert len(opening_calls)==int(with_opening and focus_role=='interior_support')


@pytest.mark.parametrize('front,centre,camera_right',[
    ([0,-1,0],[0,-1,.5],[1,0,0]),
    ([0,0,1],[0,0,1],[1,0,0]),
    (None,[1,-1,.4],[1,0,0])])
def test_widely_separated_focus_views_stay_on_the_observed_opening_side(front,centre,camera_right):
    from sketchik.observation_reference import focus_directions
    views=focus_directions(centre,camera_right,front)
    outward=np.asarray(centre if front is None else front,float);outward/=np.linalg.norm(outward)
    assert len(views)==3
    assert all(np.dot(direction,outward)>0 for _,direction in views)
    assert np.degrees(np.arccos(np.clip(np.dot(views[1][1],views[2][1]),-1,1)))==pytest.approx(120.)
    assert np.degrees(np.arccos(np.clip(np.dot(views[0][1],views[1][1]),-1,1)))==pytest.approx(60.)
