from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from sketchik.persistent_cloud import build_context, depth_contradictions
from sketchik.live_cloud import render_window, save_snapshot
from sketchik.cloud_selection import public_view, resolve_cloud_point


def frame(depth=1.):
    return {'camera': 'agentview', 'step': 40, 'depth_m': np.full((32, 32), depth),
        'rgb': np.full((32, 32, 3), [100, 30, 50], np.uint8),
        'K': np.array([[80., 0, 16], [0, 80., 16], [0, 0, 1]]),
        'world_from_camera': np.eye(4)}


def cloud(tmp_path, z=1.):
    x, y = np.meshgrid(np.linspace(-.15, .15, 20), np.linspace(-.15, .15, 20))
    xyz = np.column_stack([x.ravel(), y.ravel(), np.full(x.size, z)]).astype(np.float32)
    rgb = np.full((len(xyz), 3), [20, 60, 180], np.uint8)
    path = tmp_path/'original.npz'
    np.savez(path, world_xyz_m=xyz, rgb=rgb)
    return path, xyz, rgb


def test_occlusion_is_not_free_space_and_original_points_are_never_erased(tmp_path):
    path, xyz, _ = cloud(tmp_path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    assert not depth_contradictions(xyz, [frame(.5)]).any()
    assert depth_contradictions(xyz, [frame(1.5)]).all()
    # A second consistent view prevents a grazing contradictory ray retiring it.
    assert not depth_contradictions(xyz, [frame(1.5), frame(1.)]).any()
    layers = [{'kind': 'initial_object_shape', 'source_cloud_file': str(path)}]
    result = build_context(layers, [frame(1.5)], tmp_path/'now')
    report = result['report']
    assert report['original_retained_samples'] == len(xyz) and report['source_points_deleted'] == 0
    assert report['layers'][0]['historical_unresolved_points'] == len(xyz)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    with np.load(report['layers'][0]['row_identity_file']) as rows:
        np.testing.assert_array_equal(rows['original_point_ids'], np.arange(len(xyz)))


def test_supported_pose_keeps_all_exact_original_members_including_occluded_members(tmp_path):
    path, xyz, _ = cloud(tmp_path)
    ids = np.arange(0, len(xyz), 2)
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    translation = [.1, .2, .3]
    layers = [{'kind': 'supported_registered_pose', 'source_cloud_file': str(path),
        'point_ids': ids, 'transform': {'rotation': rotation.tolist(), 'translation_m': translation}}]
    result = build_context(layers, [frame(.1)], tmp_path/'now', max_display_points=30)
    record = result['report']['layers'][0]
    assert record['original_point_count'] == len(ids) and record['display_point_count'] == 30
    assert record['historical_unresolved_points'] == 0 and not record['current_selection_allowed']
    with np.load(record['row_identity_file']) as rows:
        np.testing.assert_array_equal(rows['original_point_ids'], ids)
        np.testing.assert_allclose(rows['current_world_xyz_m'], xyz[ids]@rotation.T+translation)


def test_articulated_historical_surfaces_are_visible_context_but_cannot_be_pointed(tmp_path):
    path, xyz, rgb = cloud(tmp_path, z=.8)
    context = build_context([{'kind': 'initial_object_shape', 'source_cloud_file': str(path),
        'historical_unresolved': True}], [frame(1.)], tmp_path/'retained')
    source = frame()
    # Only the left half is currently observed; the right half survives in history.
    source['depth_m'][:, 16:] = np.nan
    snapshot = save_snapshot({'source_step': 40, 'observation_id': 'current',
        'state': {'step': 40}}, [source], tmp_path/'snapshot')
    snapshot['retained_context'] = context
    view = render_window(snapshot, [[-.2, -.2, .7], [.2, .2, 1.1]],
        [('top', [0, 0, 1])], tmp_path/'view', 'local')[0]
    from PIL import Image
    with np.load(view['index_map_file']) as data:
        mapping = data['point_index']
    image = np.asarray(Image.open(view['image_file']))
    pale = (image[:, :, 0] >= 195) & (image[:, :, 0] <= 230) & (image[:, :, 0] == image[:, :, 1])
    # Exclude the ordinary 225 background: the historical surface is tinted 207.
    historical_pixels = np.argwhere(pale & (image[:, :, 0] != 225) & (mapping < 0))
    y, x = historical_pixels[len(historical_pixels)//2]
    with pytest.raises(ValueError, match='unknown/background'):
        resolve_cloud_point({'view_id': view['view_id'], 'pixel_xy': [int(x), int(y)]},
            [public_view(view)], [view])
    # Real current points remain selectable even when the old geometry is nearer.
    y, x = np.argwhere(mapping >= 0)[10]
    point, audit = resolve_cloud_point({'view_id': view['view_id'], 'pixel_xy': [int(x), int(y)]},
        [public_view(view)], [view])
    assert point[2] == pytest.approx(1.) and audit['carry_reference_eligible'] is False
    assert view['coverage']['retained_original_samples'] == len(xyz)


def test_bad_original_membership_never_moves_neighbouring_points(tmp_path):
    path, _, _ = cloud(tmp_path)
    with pytest.raises(ValueError, match='unique original rows'):
        build_context([{'kind': 'supported_registered_pose', 'source_cloud_file': str(path),
            'point_ids': np.array([0, 0])}], [frame()], tmp_path/'bad')


def test_surface_motion_transfers_only_identical_native_samples_across_layers(tmp_path, monkeypatch):
    from sketchik import surface_motion
    path, xyz, rgb = cloud(tmp_path)
    uv = np.c_[np.arange(len(xyz)) % 20, np.arange(len(xyz))//20]
    np.savez(path, world_xyz_m=xyz, rgb=rgb, pixel_xy=uv, source_index=np.zeros(len(xyz), int))
    sources = [{'observation_id': 'original', 'camera': 'agentview'}]
    neighbour = tmp_path/'neighbour.npz'
    # Same XYZ and pixel numbers, but a different original frame: no identity transfer.
    np.savez(neighbour, world_xyz_m=xyz, rgb=rgb, pixel_xy=uv, source_index=np.zeros(len(xyz), int))
    def estimate(old, *_):
        points = old['world_xyz_m'].copy()
        supported = np.arange(len(points)) % 2 == 0
        points[supported, 0] += .05
        return points, supported, {'status': 'supported_partial'}
    monkeypatch.setattr(surface_motion, 'track_saved_shape', estimate)
    layers = [
        {'kind': 'initial_table', 'source_cloud_file': str(path), 'sources': sources,
            'dirty_bounds': [[[-1, -1, 0], [1, 1, 2]]]},
        {'kind': 'initial_object_shape', 'source_cloud_file': str(path), 'sources': sources,
            'historical_unresolved': True, 'motion_tracking_run': str(tmp_path)},
        {'kind': 'initial_object_shape', 'source_cloud_file': str(neighbour),
            'sources': [{'observation_id': 'other', 'camera': 'agentview'}], 'historical_unresolved': True}]
    result = build_context(layers, [frame()], tmp_path/'context')
    first, second, third = result['report']['layers']
    assert first['surface_motion_supported_points'] == second['surface_motion_supported_points'] == 200
    assert third['surface_motion_supported_points'] == 0
    assert result['report']['original_retained_samples'] == 1200
    for record in (first, second):
        assert not record['current_selection_allowed']
        with np.load(record['row_identity_file']) as d:
            np.testing.assert_array_equal(d['original_point_ids'], np.arange(400))
            expected = xyz.copy(); expected[::2, 0] += .05
            np.testing.assert_allclose(d['current_world_xyz_m'], expected)
