from copy import deepcopy
import json

import numpy as np
import pytest

from sketchik.cloud_selection import public_view, resolve_cloud_point
from sketchik.live_cloud import save_snapshot, render_window
from sketchik.scene_cloud import frame_points


def frames(step=30):
    result = []
    for index, name in enumerate(('agentview', 'robot0_eye_in_hand')):
        transform = np.eye(4); transform[0, 3] = index*.3
        result.append({'camera': name, 'step': step,
            'depth_m': np.full((32, 32), 1., np.float32),
            'rgb': np.full((32, 32, 3), [50+index*100, 80, 100], np.uint8),
            'K': np.array([[80., 0., 16.], [0., 80., 16.], [0., 0., 1.]]),
            'world_from_camera': transform})
    return result


def description(step=30):
    return {'source_step': step, 'observation_id': f'obs_{step:05d}',
        'state': {'q': [0.]*7, 'position': [0., 0., 1.], 'step': step}}


@pytest.mark.parametrize('footprint', [1, 3])
def test_current_subset_pixels_resolve_full_native_table_and_original_camera(tmp_path, footprint):
    source = frames()
    snapshot = save_snapshot(description(), source, tmp_path/'snapshots')
    # This window excludes the first camera: local row 0 is not full row 0.
    views = render_window(snapshot, [[.20, -.3, .9], [.55, .3, 1.1]],
        [('top', [0., 0., 1.])], tmp_path/'views', 'live', footprint_pixels=footprint)
    view = views[0]
    with np.load(view['index_map_file']) as data:
        mapping = data['point_index']
    y, x = np.argwhere(mapping >= 1024)[70]
    point, audit = resolve_cloud_point({'kind': 'cloud_point', 'view_id': view['view_id'],
        'pixel_xy': [int(x), int(y)]}, [public_view(view)], views)
    assert audit['source_observation']['camera'] == 'robot0_eye_in_hand'
    u, v = audit['source_pixel_xy']
    ray = np.linalg.solve(source[1]['K'], [u, v, 1.])
    expected = ray*source[1]['depth_m'][v, u]+source[1]['world_from_camera'][:3, 3]
    np.testing.assert_allclose(point, expected, atol=3e-8)
    assert audit['binding_scope'] == 'single_sample_context'
    assert audit['carry_reference_eligible'] is False
    assert view['point_footprint_pixels'] == footprint
    assert view['point_count'] == 2048 and view['coverage']['current_region_points'] < 2048
    for pixel in ([0, 0], [18, 30], [1023, 1023]):
        with pytest.raises(ValueError, match='unknown/background'):
            resolve_cloud_point({'view_id': view['view_id'], 'pixel_xy': pixel}, [public_view(view)], views)


def test_current_snapshot_is_not_a_fusion_or_whole_object_identity(tmp_path):
    first = save_snapshot(description(), frames(), tmp_path/'snapshots')
    newer = frames(31)
    for frame in newer:
        frame['depth_m'][:] = 1.5
    second = save_snapshot(description(31), newer, tmp_path/'snapshots')
    with np.load(first['cloud_file']) as old, np.load(second['cloud_file']) as current:
        assert len(current['world_xyz_m']) == 2048
        assert np.all(current['world_xyz_m'][:, 2] == 1.5)
        assert np.all(old['world_xyz_m'][:, 2] == 1.)
    view = render_window(second, [[-1., -1., .8], [1., 1., 1.8]],
        [('top', [0, 0, 1])], tmp_path/'view', 'live', target_hint='cup')[0]
    assert view['slot_target_hint'] == 'cup' and 'object_id' not in view
    assert not view['historical_points_fused'] and not view['carry_reference_eligible']
    assert view['coverage']['complete_object_observed'] is False
    assert public_view(view)['source_observations'] == second['sources']


def test_missing_current_object_is_unknown_not_old_shape(tmp_path):
    snapshot = save_snapshot(description(), frames(), tmp_path/'snapshots')
    view = render_window(snapshot, [[2., 2., 2.], [3., 3., 3.]],
        [('top', [0, 0, 1])], tmp_path/'view', 'missing')[0]
    assert view['coverage']['status'] == 'current_unobserved' and not view['selectable']
    with np.load(view['index_map_file']) as data:
        assert np.all(data['point_index'] == -1)
    assert view['sources'] == snapshot['sources']


def test_fixed_viewport_keeps_metric_scale_when_visible_samples_change(tmp_path):
    bounds = [[-1., -1., .5], [1., 1., 2.]]
    a = save_snapshot(description(), frames(), tmp_path/'snapshots')
    source = frames(31); source[0]['depth_m'][:16] = np.nan
    b = save_snapshot(description(31), source, tmp_path/'snapshots')
    va = render_window(a, bounds, [('top', [0, 0, 1])], tmp_path/'a', 'a')[0]
    vb = render_window(b, bounds, [('top', [0, 0, 1])], tmp_path/'b', 'b', previous_bounds=bounds)[0]
    assert va['pixels_per_metre'] == vb['pixels_per_metre']
    assert vb['framing_changed'] is False


def test_snapshot_refuses_mixed_steps_and_never_overwrites_old_point_table(tmp_path):
    source = frames(); source[1]['step'] = 29
    with pytest.raises(ValueError, match='physical steps'):
        save_snapshot(description(), source, tmp_path)
    snapshot = save_snapshot(description(), frames(), tmp_path)
    before = open(snapshot['cloud_file'], 'rb').read()
    with pytest.raises(FileExistsError):
        save_snapshot(description(), frames(), tmp_path)
    assert open(snapshot['cloud_file'], 'rb').read() == before


def test_backprojection_cache_invalidates_depth_rgb_calibration_and_step():
    frame = frames()[0]
    first = frame_points(frame)
    assert frame_points(deepcopy(frame)) is first
    frame['depth_m'][0, 0] = 2.
    second = frame_points(frame)
    assert second is not first and second[0][0, 2] == 2.
    frame['rgb'][0, 0] = [1, 2, 3]
    third = frame_points(frame)
    assert third is not second and third[1][0].tolist() == [1, 2, 3]
    frame['world_from_camera'][0, 3] = .7
    fourth = frame_points(frame)
    assert fourth is not third
    assert fourth[0][0, 0] == pytest.approx(third[0][0, 0]+.7)
    frame['step'] += 1
    assert frame_points(frame) is not fourth
    with pytest.raises(ValueError):
        first[0][0] = 0.
