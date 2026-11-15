"""Observed vertical extent of the hand and carried points at the current EEF pose."""

from itertools import product

import numpy as np
from scipy.spatial.transform import Rotation

from .kinematics import quaternion, vector


def current_vertical_envelope(state, hand_bounds_in_eef_tool_m, carried_local=None):
    """Project local hand bounds and observed carried points onto world vertical.

    This is a current-orientation measurement. Carried points describe only the
    supplied visible surface; their lowest point is not a certified object base.
    """
    position = vector(state['position'], 3, 'position')
    rotation = Rotation.from_quat(quaternion(state['quaternion_xyzw']))
    bounds = np.asarray(hand_bounds_in_eef_tool_m, dtype=float)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
        raise ValueError('hand_bounds_in_eef_tool_m must be two finite 3D corners')
    if np.any(bounds[0] > bounds[1]):
        raise ValueError('hand bounds minimum must not exceed maximum')
    corners = np.asarray(list(product(*zip(bounds[0], bounds[1]))), dtype=float)
    hand_z = rotation.apply(corners)[:, 2]
    hand_range = [float(hand_z.min()), float(hand_z.max())]

    result = {
        'current_eef_world_z_m': float(position[2]),
        'hand_z_relative_eef_m': hand_range,
        'hand_and_observed_z_relative_eef_m': hand_range.copy(),
        'carried_observed_point_count': None,
        'carried_geometry_status': 'not_supplied',
        'holding_status_from_geometry': 'not_assessed',
        'orientation_scope': 'current_orientation_only',
        'complete_held_shape_known': False,
    }
    if carried_local is None:
        return result
    carried = np.asarray(carried_local, dtype=float)
    if carried.ndim != 2 or carried.shape[1] != 3 or not np.isfinite(carried).all():
        raise ValueError('carried_local must be an Nx3 array of finite points')
    result['carried_observed_point_count'] = int(len(carried))
    result['carried_geometry_status'] = 'observed_points' if len(carried) else 'no_supplied_points'
    if not len(carried):
        return result
    carried_z = rotation.apply(carried)[:, 2]
    carried_range = [float(carried_z.min()), float(carried_z.max())]
    result['carried_observed_z_relative_eef_m'] = carried_range
    result['hand_and_observed_z_relative_eef_m'] = [
        min(hand_range[0], carried_range[0]), max(hand_range[1], carried_range[1])]
    result['height_above_observed_carried_min_m'] = (
        result['hand_and_observed_z_relative_eef_m'][1] - carried_range[0])
    return result
