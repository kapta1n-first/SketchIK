"""Initial pad-contact evidence from native RGB-D and known robot geometry.

This checks an intended close without requiring a whole-object carry template.
It neither identifies the material between fingers nor certifies attachment.
Occlusion and missing depth remain unknown, rather than becoming empty space.
"""
from __future__ import annotations

import numpy as np


def assess_initial_contact(before, after):
    """Reject only a near-closed grasp with positively observed empty pad space."""
    result = {'status': 'unknown', 'holding_certified': False,
        'target_identity_certified': False, 'whole_object_reference_required': False,
        'empty_gap_threshold_m': .003, 'contact_margin_m': .004,
        'minimum_visible_free_fraction': .8}
    if not before.get('available') or not after.get('available'):
        return {**result, 'reason': 'Current calibrated pad/depth evidence is unavailable.'}
    if after['near_pad_surface_samples']:
        return {**result, 'status': 'contact_geometry_supported',
            'reason': 'Current non-robot surfaces reach the measured pad contact zone; '
                'their identity, friction and attachment remain unverified.'}
    if (before['jaw_gap_m'] > .010 and after['jaw_gap_m'] <= .003
            and before['near_pad_surface_samples'] == 0
            and before['visible_free_fraction'] >= .8
            and after.get('close_translation_m', np.inf) <= .005
            and after.get('close_rotation_rad', np.inf) <= np.deg2rad(3.)):
        return {**result, 'status': 'contradicted',
            'reason': 'The intended close nearly reached the empty-jaw configuration, '
                'while native depth positively shows free space in the pre-close pad zone '
                'and no non-robot surface supports the post-close pad zone. '
                'Review the actual stopped grasp before executing its suffix. '
                'This is a contact-evidence contradiction, not certified empty fingers.'}
    return {**result, 'reason': 'No supported pad contact is visible, but closure, visibility '
        'or pre-close evidence does not establish a contradiction. Contact remains unknown.'}


def _pad_zone(geometry):
    pads = [pad['bounds_in_eef_tool_m'] for finger in geometry.get('fingers', [])
        for pad in finger.get('pads', []) if pad is not None]
    if len(pads) != 2:
        return None
    bounds = np.asarray(pads, dtype=float)
    if bounds.shape != (2, 2, 3) or not np.isfinite(bounds).all():
        raise ValueError('Pad bounds must be finite paired 3D bounds')
    bounds = bounds[np.argsort(bounds[:, 0, 0])]
    low = np.r_[bounds[0, 1, 0], np.maximum(bounds[0, 0, 1:], bounds[1, 0, 1:])]
    high = np.r_[bounds[1, 0, 0], np.minimum(bounds[0, 1, 1:], bounds[1, 1, 1:])]
    if np.any(high < low):
        return None
    return low, high


def _visible_free(probes, frames):
    free = np.zeros(len(probes), dtype=bool)
    cameras = []
    for frame in frames:
        depth = np.asarray(frame['depth_m'])
        transform = np.asarray(frame['world_from_camera'])
        camera = (probes-transform[:3, 3]) @ transform[:3, :3]
        positive = np.flatnonzero(np.isfinite(camera).all(axis=1) & (camera[:, 2] > .04))
        projected = camera[positive] @ np.asarray(frame['K']).T
        uv = np.rint(projected[:, :2]/projected[:, 2, None]).astype(int)
        inside = np.all((uv >= 0) & (uv < np.asarray(depth.shape[::-1])), axis=1)
        ids, uv = positive[inside], uv[inside]
        measured = depth[uv[:, 1], uv[:, 0]]
        valid = np.isfinite(measured) & (measured > 0)
        clear = valid & (measured-camera[ids, 2] > .004)
        free[ids[clear]] = True
        cameras.append({'camera': frame.get('camera'), 'source_step': frame['step'],
            'valid_probes': int(valid.sum()), 'free_probes': int(clear.sum())})
    return float(free.mean()), cameras


def _observe_pads(ik, clearance, state, frames):
    from .robot_geometry import gripper_geometry_in_eef
    from .scene_cloud import frame_points
    if not frames:
        return {'available': False}
    if any(frame['step'] != state['step'] for frame in frames):
        raise ValueError('Initial grasp depth and robot state must share a physical step')
    clearance.set_state(state)
    origin, rotation = ik.fk(state['q'])
    geometry = gripper_geometry_in_eef(ik, origin, rotation, state)
    zone = _pad_zone(geometry or {})
    if zone is None:
        return {'available': False}
    low, high = zone
    count = 0
    camera_counts = []
    for frame in frames:
        xyz, _, _ = frame_points(frame)
        local = (xyz-origin) @ rotation
        nearby = np.all((local >= low-.004) & (local <= high+.004), axis=1)
        points = xyz[nearby]
        # Remove known robot surfaces only. A cabinet ROI is never promoted to
        # an object template, and surface presence is not semantic identity.
        keep = clearance.non_robot_mask(points, state, margin=.001) if len(points) else []
        samples = int(np.count_nonzero(keep))
        count += samples
        camera_counts.append({'camera': frame.get('camera'), 'source_step': frame['step'],
            'near_pad_non_robot_samples': samples})
    # Inset from the known pads to avoid treating their own measured surfaces
    # as free space. Every probe retains its native camera visibility test.
    inset = np.minimum((high-low)/4, .002)
    axes = [np.linspace(a, b, n) for a, b, n in zip(low+inset, high-inset, (7, 5, 5))]
    local_probes = np.asarray(np.meshgrid(*axes, indexing='ij')).reshape(3, -1).T
    fraction, visibility = _visible_free(local_probes @ rotation.T + origin, frames)
    return {'available': True, 'source_step': state['step'],
        'jaw_gap_m': geometry['current_inner_jaw_gap_m'],
        'pad_zone_eef_tool_m': [low.tolist(), high.tolist()],
        'near_pad_surface_samples': count, 'surface_cameras': camera_counts,
        'visible_free_fraction': fraction, 'visibility_cameras': visibility,
        'probe_count': len(local_probes)}


def inspect_initial_contact(ik, before_state, before_frames, after_state, after_frames):
    """Pure observation/FK; restore IK state and never take a robot/camera step."""
    from scipy.spatial.transform import Rotation
    from .observed_clearance import ObservedClearance
    saved = np.asarray(ik.data.qpos).copy()
    try:
        clearance = ObservedClearance(ik)
        before = _observe_pads(ik, clearance, before_state, before_frames)
        after = _observe_pads(ik, clearance, after_state, after_frames)
        after['close_translation_m'] = float(np.linalg.norm(
            np.asarray(after_state['position'])-before_state['position']))
        after['close_rotation_rad'] = float((
            Rotation.from_quat(after_state['quaternion_xyzw']) *
            Rotation.from_quat(before_state['quaternion_xyzw']).inv()).magnitude())
        return {**assess_initial_contact(before, after), 'before': before, 'after': after,
            'source_step': after_state['step'], 'robot_actions': 0, 'model_requests': 0,
            'scope': 'Local observed pad-contact geometry only; no task-object truth, '
                'whole-object registration, holding certification or collision planning.'}
    finally:
        ik.data.qpos[:] = saved
        ik.fk(saved[ik.q_index])
