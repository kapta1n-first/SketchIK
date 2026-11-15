import json

import numpy as np
import pytest

from sketchik.cloud_extents import summarize_components,principal_extents
from sketchik.cloud_extents import seeded_display_bounds,fragment_filter


def grid(origin, shape, step=.004):
    return np.indices(shape).reshape(3, -1).T * step + origin


def test_two_objects_and_table_are_separate():
    table = grid(np.array([0., 0., .5]), (10, 10, 1))
    first = grid(np.array([.1, 0., .52]), (5, 5, 3))
    second = grid(np.array([.3, 0., .52]), (5, 5, 3))
    points = np.vstack((table, first, second))
    result = summarize_components(points, table_height=.5)
    assert len(result['components']) == 2
    assert np.all(result['point_group_ids'][:len(table)] == 0)
    assert len(set(result['point_group_ids'][len(table):])) == 2
    assert result['table_threshold_world_z_m'] == .506
    assert result['complete_shape_known'] is False
    json.dumps({k: v for k, v in result.items() if k != 'point_group_ids'})


def test_connected_thin_handle_extreme_is_preserved():
    body = grid(np.array([.1, 0., .6]), (7, 7, 3))
    handle = np.array([[.124 + .004 * i, .012, .604] for i in range(12)])
    points = np.vstack((body, handle))
    result = summarize_components(points, min_voxels=32)
    component = result['components'][0]
    assert len(result['components']) == 1
    assert component['point_count'] == len(points)
    assert np.isclose(component['bounds_world_m'][1][0], handle[-1, 0])
    assert component['span_world_m'][0] >= component['robust_span_world_m'][0]


def test_nonfinite_and_tiny_fragments():
    main = grid(np.array([.1, 0., .6]), (4, 4, 2))
    points = np.vstack((main, [[np.nan, 0, 0], [np.inf, 0, 0], [.7, 0, .6]]))
    result = summarize_components(points)
    assert len(result['components']) == 1
    assert result['components'][0]['point_count'] == 32
    np.testing.assert_array_equal(result['point_group_ids'][-3:], [0, 0, 0])


def test_rotated_body_reports_world_span_and_limited_scope():
    local = grid(np.array([0., 0., 0.]), (12, 5, 3))
    theta = np.pi / 4
    rotation = np.array([[np.cos(theta), -np.sin(theta), 0],
                         [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    points = local @ rotation.T + [.2, .3, .7]
    result = summarize_components(points, min_voxels=16)
    assert len(result['components']) == 1
    span = result['components'][0]['span_world_m']
    assert span[0] > .04 and span[1] > .04
    assert result['table_threshold_world_z_m'] is None
    assert 'possible table/background' in result['scope']
    assert result['complete_shape_known'] is False


def test_rotated_thin_body_pairs_each_length_with_its_world_direction():
    from scipy.spatial.transform import Rotation
    local=grid(np.zeros(3),(20,9,3))
    rotation=Rotation.from_euler('xyz',[.21,-.32,.67]).as_matrix()
    points=local@rotation.T+[.2,-.4,.9]
    component=summarize_components(points,min_voxels=16)['components'][0]
    axes=np.asarray(component['observed_principal_axes_world'])
    np.testing.assert_allclose(abs(axes@rotation),np.eye(3),atol=1e-10)
    np.testing.assert_allclose(axes@axes.T,np.eye(3),atol=1e-10)
    assert np.linalg.det(axes)==pytest.approx(1.)
    assert component['observed_principal_spans_m']==pytest.approx([.076,.032,.008])
    np.testing.assert_allclose(np.ptp(points@axes.T,axis=0),component['observed_principal_spans_m'])
    assert min(component['principal_axis_gap_ratios'])>.5
    # A world-XY bounding box cannot identify which direction is thin.
    assert component['span_world_m'][0]>.05 and component['span_world_m'][1]>.04


def test_equal_variances_do_not_present_arbitrary_axes_as_well_separated():
    result=principal_extents(grid(np.zeros(3),(7,7,7)))
    assert max(abs(np.asarray(result['principal_axis_gap_ratios'])))<1e-12
    assert set(result)=={'observed_principal_spans_m','observed_principal_axes_world','principal_axis_gap_ratios'}
    # A planar patch has one distinct normal but no unique in-plane axes.
    result=principal_extents(grid(np.zeros(3),(7,7,1)))
    assert result['principal_axis_gap_ratios'][0]==pytest.approx(0,abs=1e-12)
    assert result['principal_axis_gap_ratios'][2]==pytest.approx(1.)


@pytest.mark.parametrize('points',[[],[[0,0,0],[1,1,1]],[[0,0,0],[1,0,0],[0,np.nan,0]]])
def test_principal_extents_reject_missing_or_nonfinite_samples(points):
    with pytest.raises(ValueError,match='finite XYZ'):principal_extents(points)


def test_seeded_crop_grows_beyond_box_to_thin_handle_without_table_or_other_object(tmp_path):
    body=grid(np.array([0.,0.,.52]),(10,8,3),step=.006)
    handle=np.array([[.054+.006*i,.018,.526] for i in range(17)])
    other=grid(np.array([.35,.3,.52]),(8,8,3),step=.006)
    table=grid(np.array([0.,0.,.5]),(20,20,1),step=.006)
    points=np.vstack([body,handle,other,table])
    pixels=np.full((len(points),2),100,np.int32)
    pixels[:len(body)]=[20,20]
    # A little unrelated background in the same coarse image rectangle must
    # not extend the selected object's display to a second remote component.
    pixels[len(body)+len(handle):len(body)+len(handle)+2]=[21,21]
    cloud=tmp_path/'cloud.npz'
    np.savez(cloud,world_xyz_m=points,pixel_xy=pixels,source_index=np.zeros(len(points),np.uint16))
    meta={'point_cloud_file':str(cloud),'observed_table_height_m':.5,
        'sources':[{'observation_id':'source','camera':'agentview'}]}
    refs=[{'observation_id':'source','camera':'agentview','bounds_xyxy':[10,10,40,40]}]
    bounds,audit=seeded_display_bounds(meta,refs,tmp_path/'derived')
    assert bounds[1][0]>=handle[-1,0] and bounds[1][0]<.3
    assert bounds[0][2]<.5 and audit['semantic_identity_verified'] is False
    assert audit['complete_shape_known'] is False and len(audit['component_ids'])==1
    assert audit['seeds'][0]['other_fragment_seed_points']==2
    keep=fragment_filter(audit)(points)
    assert np.all(keep[:len(body)+len(handle)])
    assert not np.any(keep[len(body)+len(handle):len(body)+len(handle)+len(other)])
    assert np.all(keep[-len(table):])
    first=(tmp_path/'derived/display_components.json').read_bytes()
    reused,_=seeded_display_bounds(meta,refs,tmp_path/'derived')
    np.testing.assert_array_equal(bounds,reused)
    assert (tmp_path/'derived/display_components.json').read_bytes()==first


def test_component_crop_rebuilds_for_changed_observation_file(tmp_path):
    body=grid(np.array([0.,0.,.6]),(8,8,3),step=.006)
    cloud=tmp_path/'cloud.npz'
    def save(points):np.savez(cloud,world_xyz_m=points,pixel_xy=np.full((len(points),2),20),
        source_index=np.zeros(len(points),np.uint16))
    save(body)
    meta={'point_cloud_file':str(cloud),'observed_table_height_m':.5,
        'sources':[{'observation_id':'source','camera':'agentview'}]}
    refs=[{'observation_id':'source','camera':'agentview','bounds_xyxy':[10,10,40,40]}]
    before,_=seeded_display_bounds(meta,refs,tmp_path/'derived')
    save(body+[.2,0,0])
    after,_=seeded_display_bounds(meta,refs,tmp_path/'derived')
    assert after[0][0]-before[0][0]==pytest.approx(.2)
