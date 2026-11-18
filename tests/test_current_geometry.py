from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from sketchik.current_geometry import inspect_snapshot, selected_geometry
from sketchik.scene_cloud import frame_points
from sketchik.cloud_selection import public_view, resolve_cloud_point


def frame(camera='agentview', step=40):
    depth = np.full((64, 96), np.nan)
    depth[10:26, 10:26] = 1.
    depth[36:52, 70:86] = 1.
    rgb = np.zeros((64, 96, 3), np.uint8)
    rgb[10:26, 10:26] = [150, 40, 50]
    rgb[36:52, 70:86] = [40, 160, 50]
    return {'depth_m': depth, 'rgb': rgb, 'K': np.array([[500., 0., 48.], [0., 500., 32.], [0., 0., 1.]]),
        'world_from_camera': np.eye(4), 'camera': camera, 'step': step}


def source(camera='agentview', step=40, bounds=None):
    return {'frame': frame(camera, step), 'observation_id': 'obs_current',
        'bounds_xyxy': bounds or [0, 0, 96, 64]}


def test_current_candidates_keep_all_fragments_and_exact_original_samples(tmp_path):
    inputs = [source()]
    original = deepcopy(inputs)
    inspection = inspect_snapshot(inputs, 40, tmp_path/'inspect')
    assert len(inspection['components']) == 2
    assert inspection['semantic_identity_verified'] is False and not inspection['old_world_points_merged']
    assert inspection['sam_requests'] == 0
    xyz, rgb, pixels = frame_points(inputs[0]['frame'])
    with np.load(inspection['cloud_file']) as data:
        np.testing.assert_array_equal(data['world_xyz_m'], xyz)
        np.testing.assert_array_equal(data['rgb'], rgb)
        np.testing.assert_array_equal(data['pixel_xy'], pixels)
        assert set(data['source_index']) == {0}
    np.testing.assert_array_equal(inputs[0]['frame']['rgb'], original[0]['frame']['rgb'])
    np.testing.assert_array_equal(inputs[0]['frame']['depth_m'], original[0]['frame']['depth_m'])
    assert len(inspection['candidate_image_files']) == 1


@pytest.mark.parametrize('error', ['mixed_step', 'duplicate_camera', 'outside', 'non_integer'])
def test_invalid_snapshot_or_region_fails_before_creating_an_evidence_folder(tmp_path, error):
    inputs = [source()]
    if error == 'mixed_step': inputs.append(source('robot0_eye_in_hand', step=41))
    elif error == 'duplicate_camera': inputs.append(source())
    elif error == 'outside': inputs[0]['bounds_xyxy'] = [0, 0, 100, 64]
    else: inputs[0]['bounds_xyxy'][2] = 96.
    with pytest.raises(ValueError): inspect_snapshot(inputs, 40, tmp_path/'inspect')
    assert not (tmp_path/'inspect').exists()


def test_robot_exclusion_does_not_erase_original_points_or_select_the_remaining_object(tmp_path):
    inspection = inspect_snapshot([source()], 40, tmp_path/'inspect',
        non_robot_mask=lambda xyz: xyz[:, 0] > 0)
    assert len(inspection['components']) == 1
    assert inspection['robot_filter']['excluded_points'] == 256
    with np.load(inspection['cloud_file']) as data:
        assert len(data['world_xyz_m']) == 512
    with np.load(inspection['point_group_ids_file']) as data:
        assert (data['point_group_ids'] == 0).sum() == 256
    assert not inspection['semantic_identity_verified']
    with pytest.raises(ValueError, match='boolean'):
        inspect_snapshot([source()], 40, tmp_path/'bad_filter',
            non_robot_mask=lambda xyz: np.ones(len(xyz), int))


def test_selecting_one_or_multiple_groups_never_snaps_and_preserves_native_provenance(tmp_path):
    inputs = [source(), source('robot0_eye_in_hand')]
    inspection = inspect_snapshot(inputs, 40, tmp_path/'inspect')
    chosen = selected_geometry(inspection, ['G1', 'G2'], 40, tmp_path/'selected', 'object_current_001')
    with np.load(chosen['cloud_file']) as data:
        xyz, pixels, sources = data['world_xyz_m'], data['pixel_xy'], data['source_index']
    assert len(xyz) == 1024 and set(sources) == {0, 1}
    assert len(chosen['selectable_views']) == 3
    for view in chosen['selectable_views']:
        with np.load(view['index_map_file']) as data:
            mapping = data['point_index']
        positions = np.argwhere(mapping >= 0)
        y, x = positions[len(positions)//2]
        index = int(mapping[y, x])
        with np.load(view['point_group_ids_file']) as data:
            group_id = int(data['point_group_ids'][index])
        point, provenance = resolve_cloud_point({'kind': 'cloud_point', 'view_id': view['view_id'],
            'pixel_xy': [int(x), int(y)], 'group_id': group_id}, [public_view(view)], [view])
        np.testing.assert_array_equal(point, xyz[index])
        assert provenance['source_pixel_xy'] == pixels[index].tolist()
        assert provenance['source_observation']['camera'] == inputs[int(sources[index])]['frame']['camera']
        assert provenance['source_observation']['source_step'] == 40
        with pytest.raises(ValueError, match='unknown/background'):
            resolve_cloud_point({'kind': 'cloud_point', 'view_id': view['view_id'],
                'pixel_xy': [0, 0], 'group_id': group_id}, [public_view(view)], [view])
    assert chosen['observed_shape']['complete_shape_known'] is False


@pytest.mark.parametrize('candidates,step', [([], 40), (['G0'], 40), (['G99'], 40), (['G1', 'G1'], 40), (['G1'], 41)])
def test_empty_unknown_duplicate_and_stale_candidates_cannot_publish_points(tmp_path, candidates, step):
    inspection = inspect_snapshot([source()], 40, tmp_path/'inspect')
    with pytest.raises(ValueError):
        selected_geometry(inspection, candidates, step, tmp_path/'selected', 'object_current_001')
    assert not (tmp_path/'selected').exists()


def test_native_rectangle_keeps_original_uv_and_is_not_a_semantic_mask(tmp_path):
    inspection = inspect_snapshot([source(bounds=[5, 5, 30, 30])], 40, tmp_path/'inspect')
    assert inspection['point_count'] == 256 and len(inspection['components']) == 1
    with np.load(inspection['cloud_file']) as data:
        assert data['pixel_xy'][:, 0].min() == 10
        assert data['pixel_xy'][:, 1].min() == 10
    assert 'search windows' in inspection['scope']
    assert 'not object identities' in inspection['scope']


def test_tabletop_separation_retains_visible_fragments_below_and_above_the_table(tmp_path):
    value = source()
    value['frame']['depth_m'][:] = 1.
    value['frame']['depth_m'][10:26, 10:26] = .8
    value['frame']['depth_m'][36:52, 70:86] = 1.2
    inspection = inspect_snapshot([value], 40, tmp_path/'inspect', table_height=1.)
    assert len(inspection['components']) == 2
    heights = sorted(component['bounds_world_m'][0][2] for component in inspection['components'])
    np.testing.assert_allclose(heights, [.8, 1.2])
    assert inspection['table_reference']['below_table_surfaces_retained'] is True
    assert inspection['point_count'] == 64*96
    assert inspection['table_reference']['excluded_from_grouping_points'] == 64*96-512
