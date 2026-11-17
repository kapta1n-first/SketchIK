from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from sketchik.rigid_binding import template_members, save_bindings


def original(tmp_path):
    xyz = np.arange(30, dtype=float).reshape(10, 3)/100
    path = tmp_path/'original.npz'
    np.savez(path, world_xyz_m=xyz, rgb=np.zeros((10, 3), np.uint8),
        pixel_xy=np.column_stack([np.arange(10), np.zeros(10)]).astype(np.uint16),
        source_index=np.zeros(10, int))
    sources = [{'observation_id': 'old', 'camera': 'agentview', 'source_step': 10}]
    view = {'cloud_file': str(path), 'view_id': 'original', 'sources': sources}
    with np.load(path) as data:
        ids = np.array([8, 2, 5])  # Filtered order is deliberately not original row order.
        old = {key: data[key][ids] for key in data.files}
    return view, old, sources, ids


def test_membership_uses_original_row_ids_after_filter_and_reorder(tmp_path):
    view, old, sources, ids = original(tmp_path)
    old.update(template_source_cloud_file=np.asarray(view['cloud_file']), template_point_ids=ids)
    selected, rows = template_members(view, old, sources)
    np.testing.assert_array_equal(selected, ids)
    np.testing.assert_array_equal(rows, [0, 1, 2])


def test_cross_table_membership_joins_exact_native_identity_not_nearest_geometry(tmp_path):
    view, old, sources, ids = original(tmp_path)
    selected, rows = template_members(view, old, sources)
    np.testing.assert_array_equal(selected, sorted(ids))
    np.testing.assert_array_equal(np.asarray(old['world_xyz_m'])[rows],
        np.asarray([[.06, .07, .08], [.15, .16, .17], [.24, .25, .26]]))
    missing = deepcopy(sources); missing[0]['observation_id'] = 'another_capture'
    assert not len(template_members(view, old, missing)[0])
    old['pixel_xy'] = np.zeros((3, 2), np.uint16)
    with pytest.raises(ValueError, match='unique native'):
        template_members(view, old, sources)


def test_matching_identity_with_changed_coordinate_frame_is_rejected(tmp_path):
    view, old, sources, ids = original(tmp_path)
    old['world_xyz_m'] = old['world_xyz_m']+.3
    with pytest.raises(ValueError, match='coordinate frames differ'):
        template_members(view, old, sources)


def test_world_pose_requires_complete_unique_registration_and_writes_no_image(tmp_path):
    view, old, sources, ids = original(tmp_path)
    registration = {'status': 'accepted', 'search_complete': True, 'source_step': 40,
        'transform': {'rotation': np.eye(3).tolist(), 'translation_m': [.1, .2, .3]}}
    result = save_bindings(old, sources, registration, [view], tmp_path/'bound', 'object')
    with np.load(result['original']['members_file']) as data:
        np.testing.assert_array_equal(data['original_point_ids'], sorted(ids))
    assert result['original']['bitmap_reused']
    assert not list((tmp_path/'bound').glob('*.png'))
    for unavailable in ({**registration, 'status': 'unavailable'}, {**registration, 'search_complete': False}):
        with pytest.raises(ValueError, match='complete accepted unique'):
            save_bindings(old, sources, unavailable, [view], tmp_path/'invalid', 'object')
