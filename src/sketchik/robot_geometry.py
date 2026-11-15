"""Compact bounds of known robot hand geometry, without task-object state.

Importing this module does not import MuJoCo or robosuite, so it is safe before
``runtime.configure``. The caller supplies a robot-only IK model after FK.
"""
from __future__ import annotations

from itertools import product

import numpy as np


# MuJoCo mjtGeom values; keeping these data constants avoids an early engine import.
_SPHERE, _CAPSULE, _ELLIPSOID, _CYLINDER, _BOX, _MESH = range(2, 8)
_CORNERS = np.asarray(list(product((-1., 1.), repeat=3)))


def recorded_robot_geometry(ik, state, recorded_goal=None):
    """Moving robot body boxes at actual q and an already recorded joint goal.

    No IK search, physics, task model or collision test. The goal is omitted if
    its exact joint branch was not recorded; an EEF pose is never re-solved here.
    All robot-only joint positions and FK caches are restored on return/error.
    """
    model, data = ik.model, ik.data
    before = np.asarray(data.qpos).copy()
    groups = {}
    for geom in np.flatnonzero(np.asarray(model.geom_contype) | np.asarray(model.geom_conaffinity)):
        body = int(model.geom_bodyid[geom])
        ancestor = body
        while ancestor and not model.body_jntnum[ancestor]:
            ancestor = int(model.body_parentid[ancestor])
        if ancestor:
            groups.setdefault(body, []).append(int(geom))

    def boxes(q):
        for name, value in zip(state.get('finger_joint_names', []), state.get('finger_joint_positions', [])):
            data.qpos[model.joint(name).qposadr[0]] = value
        ik.fk(q)
        result = []
        for body, geoms in groups.items():
            origin = np.asarray(data.xpos[body])
            rotation = np.asarray(data.xmat[body]).reshape(3, 3)
            bounds = _geometry_bounds_in_eef(ik, origin, rotation, geoms)
            if bounds is None:
                raise ValueError('Cannot bound a moving robot collision geometry')
            local = np.asarray(list(product(*zip(*bounds))))
            result.append({'body': model.body(body).name, 'corners_world_m':
                (local @ rotation.T + origin).tolist()})
        return result

    try:
        actual = boxes(state['q'])
        goal_q = (recorded_goal or {}).get('pose', {}).get('joint_target')
        goal = boxes(goal_q) if goal_q is not None else []
        return {'actual_source_step': state['step'], 'actual_joint_positions_rad': list(state['q']),
            'actual_parts': actual, 'recorded_goal_parts': goal,
            'recorded_goal_joint_positions_rad': goal_q,
            'goal_failure_step': (recorded_goal or {}).get('failure_observed_step'),
            'goal_status': 'exact_recorded_joint_branch' if goal else 'no_recorded_joint_branch',
            'finger_configuration_source_step': state['step'],
            'scope': 'Known moving robot body-oriented collision-shape envelopes. '
                'Goal uses the recorded joint branch and current measured fingers. '
                'Boxes are not solid occupancy, actual contact, a new route or a collision certificate.'}
    finally:
        data.qpos[:] = before
        ik.fk(before[ik.q_index])


def pad_reference_geometry(geometry):
    """Present existing robot bounds about the actual pad midpoint, in tool axes.

    Pure translation of measured mechanical bounds. No task objects, physics,
    grasp selection, centre estimation, or collision certification is involved.
    Missing calibration remains missing, never a guessed zero offset.
    """
    if not geometry or len(geometry.get('pad_centres_in_eef_tool_m', [])) != 2:
        return None
    pads = np.asarray(geometry['pad_centres_in_eef_tool_m'], dtype=float)
    if pads.shape != (2, 3) or not np.isfinite(pads).all():
        return None
    midpoint = pads.mean(axis=0)
    def shifted(bounds):
        if bounds is None:
            return None
        value = np.asarray(bounds, dtype=float)
        if value.shape != (2, 3) or not np.isfinite(value).all():
            raise ValueError('Robot envelope must contain finite min/max XYZ')
        return (value-midpoint).tolist()
    palm = shifted((geometry.get('palm_and_fixed_hand') or {}).get('bounds_in_eef_tool_m'))
    return {'source_step': geometry['source_step'], 'unit': 'm',
        'origin': 'pad_midpoint', 'axes': geometry['axis_meaning'],
        'current_inner_jaw_gap_m': geometry.get('current_inner_jaw_gap_m'),
        'hand_bounds_relative_pad_midpoint_tool_m': shifted(geometry.get('hand_bounds_in_eef_tool_m')),
        'palm_bounds_relative_pad_midpoint_tool_m': palm,
        'palm_extent_along_negative_tool_z_from_pad_m': None if palm is None else {
            'nearest_signed_distance_m': -palm[1][2],
            'farthest_signed_distance_m': -palm[0][2],
            'meaning': 'Axis projection of the full palm envelope behind the pads, not the distance to its far end alone. Negative means the envelope extends forward of the pad plane; not free space or a collision check.'},
        'fingers': [{'length_along_tool_z_m': finger['length_along_tool_z_m'],
            'bounds_relative_pad_midpoint_tool_m': shifted(finger['bounds_in_eef_tool_m']),
            'pad_bounds_relative_pad_midpoint_tool_m': [shifted(pad['bounds_in_eef_tool_m'])
                for pad in finger.get('pads', []) if pad is not None]}
            for finger in geometry.get('fingers', [])],
        'scope': 'Current robot envelope about pad_midpoint, not solid occupancy or a collision check. '
            'Positive tool Z points from palm toward fingertips. Program converts the chosen reference '
            'to EEF at the requested orientation; do not add that compensation to offset_m.'}


def hand_bounds_in_eef(ik, origin, rotation) -> list[list[float]] | None:
    """Return hand collision-geometry bounds ``[min_xyz, max_xyz]`` in EEF meters.

    The hand is the EEF site's body and all its descendants, including articulated
    finger bodies in their current FK configuration. An empty fixed marker body
    is resolved up its fixed parent chain to the first subtree with collisions;
    this search never crosses a parent with a joint or reaches the world body.
    Arm ancestors, unrelated bodies, and geoms with both collision masks zero are
    excluded. Mesh vertices and box corners use the current ``geom_xpos/xmat``;
    curved primitives use a
    conservative local box. This describes mechanical volume, not collision-free
    motion or task-object geometry. It neither advances nor refreshes simulation.

    Return None when the site is on the world body, there is no hand collision
    geometry, or a collision shape cannot be bounded by the supported primitives.
    The caller must set finger joints and run FK before supplying the EEF frame.
    """
    origin = np.asarray(origin, dtype=float)
    rotation = np.asarray(rotation, dtype=float)
    if (origin.shape != (3,) or rotation.shape != (3, 3)
            or not np.isfinite(origin).all() or not np.isfinite(rotation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-6)):
        raise ValueError("EEF origin and rotation must be a finite rigid frame")

    model, data = ik.model, ik.data
    root = int(model.site_bodyid[ik.site_id])
    if root == 0:
        return None

    children = [[] for _ in range(model.nbody)]
    for body in range(1, model.nbody):
        children[int(model.body_parentid[body])].append(body)
    collision_geoms = np.flatnonzero(np.asarray(model.geom_contype)
                                    | np.asarray(model.geom_conaffinity))
    while True:
        hand_bodies = set()
        pending = [root]
        while pending:
            body = pending.pop()
            hand_bodies.add(body)
            pending.extend(children[body])
        if any(int(model.geom_bodyid[geom]) in hand_bodies for geom in collision_geoms):
            break
        if model.body_jntnum[root]:
            return None
        parent = int(model.body_parentid[root])
        if parent == 0 or model.body_jntnum[parent]:
            return None
        root = parent

    selected = [geom for geom in collision_geoms
                if int(model.geom_bodyid[geom]) in hand_bodies]
    return _geometry_bounds_in_eef(ik, origin, rotation, selected)


def _geometry_bounds_in_eef(ik, origin, rotation, geoms):
    model, data = ik.model, ik.data
    lower = np.full(3, np.inf)
    upper = np.full(3, -np.inf)
    for geom in geoms:
        kind = int(model.geom_type[geom])
        if kind == _MESH:
            mesh = int(model.geom_dataid[geom])
            if mesh < 0:
                return None
            first = int(model.mesh_vertadr[mesh])
            count = int(model.mesh_vertnum[mesh])
            vertices = np.asarray(model.mesh_vert[first:first + count], dtype=float)
            if not count:
                return None
        else:
            size = np.asarray(model.geom_size[geom], dtype=float)
            if kind == _SPHERE:
                half_size = np.repeat(size[0], 3)
            elif kind == _CAPSULE:
                half_size = np.array([size[0], size[0], size[1] + size[0]])
            elif kind == _CYLINDER:
                half_size = np.array([size[0], size[0], size[1]])
            elif kind in (_ELLIPSOID, _BOX):
                half_size = size
            else:
                # Do not silently omit an unsupported part of the mechanical hand.
                return None
            vertices = _CORNERS * half_size

        geom_rotation = np.asarray(data.geom_xmat[geom]).reshape(3, 3)
        geom_origin = np.asarray(data.geom_xpos[geom])
        vertices_eef = (vertices @ geom_rotation.T + geom_origin - origin) @ rotation
        if not np.isfinite(vertices_eef).all():
            return None
        lower = np.minimum(lower, vertices_eef.min(axis=0))
        upper = np.maximum(upper, vertices_eef.max(axis=0))

    if not np.isfinite(lower).all():
        return None
    return [lower.tolist(), upper.tolist()]


def measured_inner_jaw_gap(ik, state):
    """Known pad geometry at this actual joint state; no image/contact inference."""
    model = ik.model
    for name, value in zip(state['finger_joint_names'], state['finger_joint_positions']):
        ik.data.qpos[model.joint(name).qposadr[0]] = value
    origin, rotation = ik.fk(state['q'])
    pads = [i for i in range(model.ngeom)
            if (model.geom_contype[i] or model.geom_conaffinity[i])
            and 'pad_collision' in (model.geom(i).name or '')]
    return _inner_pad_gap(ik, origin, rotation, pads)


def _inner_pad_gap(ik, origin, rotation, pads):
    if len(pads) != 2:
        return None
    bounds = [_geometry_bounds_in_eef(ik, origin, rotation, [index]) for index in pads]
    if any(value is None for value in bounds):
        return None
    ordered = sorted(bounds, key=lambda value: value[0][0])
    return max(0., float(ordered[1][0][0] - ordered[0][1][0]))


def gripper_geometry_in_eef(ik, origin, rotation, state):
    """Read full hand, individual fingers and pads from current robot-only FK.

    Lengths are collision-envelope extents along the named EEF axes, not visual
    guesses. This function never writes qpos, advances simulation or reads task
    objects. The caller sets actual finger joints and refreshes FK first.
    """
    bounds = hand_bounds_in_eef(ik, origin, rotation)
    if bounds is None:
        return None
    model, data = ik.model, ik.data
    origin, rotation = np.asarray(origin), np.asarray(rotation)
    collisions = [i for i in range(model.ngeom)
                  if model.geom_contype[i] or model.geom_conaffinity[i]]

    def descendants(body, root):
        while body and body != root:
            body = int(model.body_parentid[body])
        return body == root

    def metrics(value):
        if value is None:
            return None
        return {'bounds_in_eef_tool_m': value,
                'span_xyz_m': (np.asarray(value[1]) - value[0]).tolist()}

    pads = [i for i in collisions if 'pad_collision' in (model.geom(i).name or '')]
    fingers, finger_geoms = [], set()
    for name, position in zip(state['finger_joint_names'], state['finger_joint_positions']):
        joint = model.joint(name)
        root = int(model.jnt_bodyid[joint.id])
        geoms = [i for i in collisions if descendants(int(model.geom_bodyid[i]), root)]
        finger_bounds = _geometry_bounds_in_eef(ik, origin, rotation, geoms)
        if finger_bounds is None:
            continue
        finger_geoms.update(geoms)
        fingers.append({'joint_name': name, 'joint_position': position,
            **metrics(finger_bounds),
            'length_along_tool_z_m': float(finger_bounds[1][2] - finger_bounds[0][2]),
            'tip_z_relative_eef_m': float(finger_bounds[1][2]),
            'pads': [metrics(_geometry_bounds_in_eef(ik, origin, rotation, [i]))
                     for i in pads if i in geoms]})

    # Resolve the same fixed-marker parent chain as hand_bounds_in_eef. The
    # remaining hand geoms include the palm/crossbar, never the arm ancestors.
    root = int(model.site_bodyid[ik.site_id])
    while not any(descendants(int(model.geom_bodyid[i]), root) for i in collisions):
        root = int(model.body_parentid[root])
    fixed_geoms = [i for i in collisions if i not in finger_geoms
                   and descendants(int(model.geom_bodyid[i]), root)]
    result = {'source': 'known robot collision model and actual finger joint state',
        'source_step': state['step'], 'coordinate_frame': 'eef_tool', 'unit': 'm',
        'axis_meaning': {'x': 'finger closing/separation', 'y': 'transverse thickness',
                         'z': 'finger forward direction'},
        'hand_bounds_in_eef_tool_m': bounds,
        'hand_span_xyz_m': metrics(bounds)['span_xyz_m'],
        'palm_and_fixed_hand': metrics(_geometry_bounds_in_eef(ik, origin, rotation, fixed_geoms)),
        'fingers': fingers,
        'finger_joint_positions_rad_or_m': state['finger_joint_positions'],
        'meaning': 'Robot mechanical envelope at this opening; not object dimensions, a grasp proposal or collision-free certification. EEF is not the fingertip or object centre.'}
    if len(pads) == 2:
        centres = np.asarray([data.geom_xpos[i] for i in pads])
        local = (centres - origin) @ rotation
        gap = _inner_pad_gap(ik, origin, rotation, pads)
        if gap is not None:
            result['current_inner_jaw_gap_m'] = gap
        result['pad_centres_in_eef_tool_m'] = local.tolist()
        result['pad_midpoint_in_eef_tool_m'] = local.mean(axis=0).tolist()
    return result
