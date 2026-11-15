import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from sketchik.vertical_envelope import current_vertical_envelope


def test_rotation_projects_all_hand_corners_and_observed_points():
    state = {'position': [1, 2, .7],
             'quaternion_xyzw': Rotation.from_euler('y', 90, degrees=True).as_quat().tolist()}
    result = current_vertical_envelope(state, [[-.2, -.1, -.3], [.4, .1, .5]],
                                       [[-.6, 0, 0], [.25, 0, 0]])
    assert result['current_eef_world_z_m'] == .7
    assert result['hand_z_relative_eef_m'] == pytest.approx([-.4, .2])
    assert result['carried_observed_z_relative_eef_m'] == pytest.approx([-.25, .6])
    assert result['hand_and_observed_z_relative_eef_m'] == pytest.approx([-.4, .6])
    assert result['height_above_observed_carried_min_m'] == pytest.approx(.85)
    assert result['carried_observed_point_count'] == 2
    assert result['carried_geometry_status'] == 'observed_points'
    assert result['holding_status_from_geometry'] == 'not_assessed'
    assert result['orientation_scope'] == 'current_orientation_only'
    assert result['complete_held_shape_known'] is False
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('carried', [None, []])
def test_missing_carried_surface_has_no_invented_height(carried):
    points = None if carried is None else np.empty((0, 3))
    result = current_vertical_envelope(
        {'position': [0, 0, 1], 'quaternion_xyzw': [0, 0, 0, 1]},
        [[-.1, -.1, -.2], [.1, .1, .3]], points)
    assert result['hand_and_observed_z_relative_eef_m'] == pytest.approx([-.2, .3])
    assert result['carried_observed_point_count'] == (None if carried is None else 0)
    assert result['carried_geometry_status'] == ('not_supplied' if carried is None else 'no_supplied_points')
    assert result['holding_status_from_geometry'] == 'not_assessed'
    assert 'carried_observed_z_relative_eef_m' not in result
    assert 'height_above_observed_carried_min_m' not in result
    assert result['complete_held_shape_known'] is False


@pytest.mark.parametrize('state,bounds,carried', [
    ({'position': [0, 0, float('nan')], 'quaternion_xyzw': [0, 0, 0, 1]}, [[0]*3, [1]*3], None),
    ({'position': [0]*3, 'quaternion_xyzw': [0, 0, 0, 2]}, [[0]*3, [1]*3], None),
    ({'position': [0]*3, 'quaternion_xyzw': [0, 0, 0, 1]}, [[0]*3, [1, 1, float('inf')]], None),
    ({'position': [0]*3, 'quaternion_xyzw': [0, 0, 0, 1]}, [[1]*3, [0]*3], None),
    ({'position': [0]*3, 'quaternion_xyzw': [0, 0, 0, 1]}, [[0]*3, [1]*3], [[0, float('nan'), 0]]),
])
def test_rejects_invalid_geometry(state, bounds, carried):
    with pytest.raises(ValueError):
        current_vertical_envelope(state, bounds, carried)
