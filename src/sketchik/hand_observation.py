"""CPU-only card of current observed world surfaces and known robot hand geometry."""
from __future__ import annotations

from itertools import product
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation


def failed_goal_context(failure=None, outcome=None):
    """Select a recorded goal, never infer a new destination or failure cause."""
    if failure:
        audit=failure.get('compiled_from_selection') or {}
        pose=failure.get('original_absolute_goal') or audit.get('compiled_pose')
        if pose:
            return {'pose':pose,'failure_stage':failure.get('failure_stage'),
                'failure_observed_step':(failure.get('actual_motion',{}).get('state') or {}).get('step',
                    audit.get('execution_source_step',audit.get('source_step'))),
                'source':'current_recovery_failure','executed':failure.get('executed',False)}
    candidates=[]
    for motion in (outcome or {}).get('motions',[]):
        if motion.get('converged') is False and motion.get('goal'):
            candidates.append({'pose':motion['goal'],'failure_stage':'motion',
                'failure_observed_step':motion.get('actual_stop',{}).get('step'),
                'source':'last_failed_stage_outcome','executed':True})
    for rejected in (outcome or {}).get('compile_rejections',[]):
        if rejected.get('target_pose'):
            candidates.append({'pose':rejected['target_pose'],'failure_stage':rejected.get('failure_stage'),
                'failure_observed_step':rejected.get('source_step'),
                'source':'last_failed_stage_outcome','executed':False})
    return max(candidates,key=lambda item:item['failure_observed_step'] or -1) if candidates else None


def render_hand_observation(points_world, rgb, state, folder, hand_bounds, pad_centres=None,
                            *, predicted_points_world=None, failed_goal=None, hand_parts=None,
                            carried_local=None, robot_geometry=None, scene_points_world=None, scene_rgb=None):
    """Render supplied, unassigned near-hand surfaces in the current world frame.

    The caller owns selection of observed non-robot points. No attachment or object
    identity is inferred here. Bounds and pads are expressed in the EEF tool frame.
    """
    points = np.asarray(points_world)
    colors = np.asarray(rgb)
    bounds = np.asarray(hand_bounds, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('points_world must be finite Nx3 world metres')
    if colors.shape != points.shape or not np.issubdtype(colors.dtype, np.number) or not np.isfinite(colors).all():
        raise ValueError('rgb must be finite Nx3 aligned with points_world')
    if np.any(colors < 0) or np.any(colors > 255):
        raise ValueError('rgb must lie in 0..255')
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[0] > bounds[1]):
        raise ValueError('hand_bounds must be ordered finite EEF bounds, shape 2x3')
    if state.get('frame', 'world') != 'world' or state.get('coordinate_frame', 'world') != 'world':
        raise ValueError('EEF state must be in world frame')
    origin = np.asarray(state['position'], dtype=float)
    quat = np.asarray(state['quaternion_xyzw'], dtype=float)
    if origin.shape != (3,) or quat.shape != (4,) or not np.isfinite(origin).all() or not np.isfinite(quat).all() or not np.isclose(np.linalg.norm(quat), 1, atol=1e-5):
        raise ValueError('EEF state requires finite world position and unit xyzw quaternion')
    step = state.get('source_step', state.get('step'))
    if isinstance(step, bool) or not isinstance(step, (int, np.integer)) or step < 0:
        raise ValueError('EEF state requires nonnegative source_step or step')
    pads = np.empty((0, 3), dtype=float) if pad_centres is None else np.asarray(pad_centres, dtype=float)
    if pads.ndim != 2 or pads.shape[1] != 3 or not np.isfinite(pads).all():
        raise ValueError('pad_centres must be finite Nx3 EEF points')

    rotation = Rotation.from_quat(quat).as_matrix()
    corners_local = np.asarray(list(product(*zip(bounds[0], bounds[1]))), dtype=float)
    corners = corners_local @ rotation.T + origin
    pads_world = pads @ rotation.T + origin
    local_parts=np.asarray(hand_parts if hand_parts is not None else [bounds],float)
    if (local_parts.ndim!=3 or local_parts.shape[1:]!=(2,3) or not len(local_parts)
            or not np.isfinite(local_parts).all() or np.any(local_parts[:,0]>local_parts[:,1])):
        raise ValueError('Hand parts must be ordered finite EEF bounds')
    part_corners=np.asarray([list(product(*zip(part[0],part[1]))) for part in local_parts],float)
    actual_boxes=part_corners@rotation.T+origin
    predicted=np.empty((0,3)) if predicted_points_world is None else np.asarray(predicted_points_world)
    if predicted.ndim!=2 or predicted.shape[1]!=3 or not np.isfinite(predicted).all():
        raise ValueError('Predicted scan must be finite Nx3 world points')
    goal_boxes=np.empty((0,8,3));goal_pads=np.empty((0,3));goal_carried=np.empty((0,3))
    if failed_goal is not None:
        pose=failed_goal['pose'];goal_origin=np.asarray(pose['position'],float)
        goal_quat=np.asarray(pose['quaternion_xyzw'],float)
        if (goal_origin.shape!=(3,) or goal_quat.shape!=(4,) or not np.isfinite(goal_origin).all()
                or not np.isfinite(goal_quat).all() or not np.isclose(np.linalg.norm(goal_quat),1.,atol=1e-5)):
            raise ValueError('Failed goal requires a finite world pose and unit xyzw quaternion')
        goal_rotation=Rotation.from_quat(goal_quat).as_matrix()
        goal_boxes=part_corners@goal_rotation.T+goal_origin
        goal_pads=pads@goal_rotation.T+goal_origin
        if carried_local is not None:
            local=np.asarray(carried_local,float)
            if local.ndim!=2 or local.shape[1:]!=(3,) or not np.isfinite(local).all():
                raise ValueError('Goal carried hypothesis must be finite associated EEF points')
            goal_carried=local@goal_rotation.T+goal_origin
    extent_points = np.concatenate((points, predicted, corners, pads_world, origin[None, :],
        goal_boxes.reshape(-1,3),goal_pads,goal_carried))
    lower, upper = extent_points.min(axis=0), extent_points.max(axis=0)
    middle = (lower + upper) / 2
    span = max(float(np.max(upper - lower)), .01) * 1.12
    robot_actual = np.asarray([part['corners_world_m'] for part in
        (robot_geometry or {}).get('actual_parts', [])], float).reshape(-1, 8, 3)
    robot_goal = np.asarray([part['corners_world_m'] for part in
        (robot_geometry or {}).get('recorded_goal_parts', [])], float).reshape(-1, 8, 3)
    scene = points if scene_points_world is None else np.asarray(scene_points_world)
    scene_colors = colors if scene_rgb is None else np.asarray(scene_rgb)
    if (scene.ndim != 2 or scene.shape[1:] != (3,) or scene_colors.shape != scene.shape
            or not np.isfinite(scene).all() or not np.isfinite(scene_colors).all()
            or np.any(scene_colors < 0) or np.any(scene_colors > 255)):
        raise ValueError('Robot context scene must be finite aligned XYZ/RGB')
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / 'hand_observation.npz', points_world_m=points,
                        rgb=colors, eef_position_world_m=origin,
                        eef_quaternion_xyzw=quat, hand_corners_world_m=corners,
                        actual_hand_parts_world_m=actual_boxes,
                        hand_bounds_eef_tool_m=bounds, pad_centres_world_m=pads_world,
                        predicted_reference_world_m=predicted,
                        failed_goal_hand_parts_world_m=goal_boxes,failed_goal_pad_centres_world_m=goal_pads,
                        failed_goal_carried_hypothesis_world_m=goal_carried,
                        actual_robot_parts_world_m=robot_actual,
                        recorded_goal_robot_parts_world_m=robot_goal,
                        robot_context_points_world_m=scene, robot_context_rgb=scene_colors,
                        source_step=np.asarray(step))
    metadata = {'source_step': int(step), 'point_count': int(len(points)),
                'hand_world_bounds_m': [corners.min(axis=0).tolist(), corners.max(axis=0).tolist()],
                'frame': 'world', 'scope': 'current observed non-robot near-hand surfaces; identities unassigned',
                'selectable': False, 'complete_object_shape_known': False,
                'carried_attachment_certified': False,
                'predicted_reference_point_count':len(predicted), 'predicted_reference_is_new_observation':False,
                'rendered_point_count': min(len(points), 30000),
                'evidence_layers': {
                    'rgb_surfaces': {'source_step': int(step), 'kind': 'current_measured_surfaces',
                        'object_identity_certified': False},
                    'known_robot_hand_and_pads': {'source_step': int(step), 'kind': 'current_known_robot_geometry', 'hand_color': 'royalblue', 'pad_color': 'red'},
                    'magenta_shape': {'evaluated_at_step': int(step), 'kind': 'eef_propagated_observed_shape_hypothesis',
                        'present': bool(len(predicted)), 'new_observation': False, 'attachment_certified': False},
                    'orange_hand_cyan_shape': {'kind': 'unreached_recorded_goal_comparison',
                        'present': failed_goal is not None, 'is_current_pose': False, 'new_observation': False}},
                'panel_layout': [*(('current_and_recorded_goal_world_'+plane if failed_goal is not None else 'current_world_'+plane) for plane in ('XY', 'XZ', 'YZ')),
                    'unreached_goal_world_3D' if failed_goal is not None else 'current_world_3D']}
    if len(robot_actual):
        metadata['known_robot_geometry'] = robot_geometry
        metadata['panel_layout'][-1] = 'actual_and_recorded_joint_goal_full_robot_world_3D'
        metadata['evidence_layers']['known_full_robot'] = {'source_step': int(step),
            'actual_color': 'royalblue', 'recorded_goal_color': 'darkorange',
            'goal_is_current_pose': False, 'collision_checked': False}
    if failed_goal is not None:
        metadata['failed_goal_projection']={**failed_goal,
            'hand_configuration':'current actual finger opening projected at the recorded goal, not historical execution',
            'hand_part_count':len(goal_boxes),'carried_hypothesis_point_count':len(goal_carried),
            'hand_configuration_source_step': int(step),
            'failure_at_current_physical_step': failed_goal.get('failure_observed_step') == int(step),
            'hand_world_bounds_m': [goal_boxes.reshape(-1,3).min(axis=0).tolist(), goal_boxes.reshape(-1,3).max(axis=0).tolist()],
            'part_world_bounds_m': [[box.min(axis=0).tolist(), box.max(axis=0).tolist()] for box in goal_boxes],
            'pad_centres_world_m': goal_pads.tolist(),
            'envelope_scope': 'Known EEF-frame part bounds transformed to the recorded pose using the current measured finger opening; excludes arm ancestors. Not a mesh-contact test, current hand pose or executed command.',
            'observed_scene_moved':False,'new_observation':False,'selectable':False,
            'collision_checked':False,'contact_cause_certified':False,
            'meaning':'Orange: hypothetical known robot parts. Cyan: supplied associated held-point hypothesis only. Projection/envelope overlap does not certify collision or attachment.'}
    (folder / 'hand_observation.json').write_text(json.dumps(metadata, indent=2))

    # Three orthographic comparisons preserve distinct actual and recorded-goal layers.
    # Observed scene points stay fixed; the goal is never a current observation.
    indices = np.linspace(0, len(points)-1, min(len(points), 30000), dtype=int) if len(points) else np.empty(0, dtype=int)
    plotted_colors = colors[indices].astype(float)/255.
    drawn = predicted[np.linspace(0, len(predicted)-1, min(3000, len(predicted)), dtype=int)] if len(predicted) else predicted
    goal_drawn = goal_carried[np.linspace(0, len(goal_carried)-1, min(3000, len(goal_carried)), dtype=int)] if len(goal_carried) else goal_carried
    fig = plt.figure(figsize=(10.24, 10.24), dpi=100)
    for slot, (first, second) in enumerate(((0, 1), (0, 2), (1, 2)), 1):
        axis = fig.add_subplot(2, 2, slot)
        if len(indices):
            axis.scatter(points[indices, first], points[indices, second], c=plotted_colors, s=.5)
        if len(drawn):
            axis.scatter(drawn[:, first], drawn[:, second], color='magenta', s=.6, alpha=.35)
        for box in actual_boxes:
            for i in range(8):
                for bit in (1, 2, 4):
                    j = i^bit
                    if i < j:
                        axis.plot(box[[i, j], first], box[[i, j], second], color='royalblue', linewidth=1.2)
        if failed_goal is not None:
            for box in goal_boxes:
                for i in range(8):
                    for bit in (1, 2, 4):
                        j = i^bit
                        if i < j:
                            axis.plot(box[[i, j], first], box[[i, j], second],
                                color='darkorange', linestyle='--', linewidth=1.1)
            if len(goal_pads):
                axis.scatter(goal_pads[:, first], goal_pads[:, second], color='darkorange', marker='+', s=30)
        if len(pads_world):
            axis.scatter(pads_world[:, first], pads_world[:, second], color='red', marker='x', s=36)
        axis.scatter(origin[first], origin[second], color='blue', marker='+', s=32)
        axis.set_xlim(middle[first]-span/2, middle[first]+span/2)
        axis.set_ylim(middle[second]-span/2, middle[second]+span/2)
        axis.set_aspect('equal', adjustable='box')
        axis.set_xlabel('World '+'XYZ'[first]+' (m)')
        axis.set_ylabel('World '+'XYZ'[second]+' (m)')
        label = 'ACTUAL blue / TARGET orange ' if failed_goal is not None else 'CURRENT '
        axis.set_title(label+'XYZ'[first]+'XYZ'[second]+' | step '+str(step), fontsize=9)
        axis.grid(alpha=.25)
    ax = fig.add_subplot(224, projection='3d')
    if len(robot_actual):
        displayed = np.linspace(0, len(scene)-1, min(len(scene), 30000), dtype=int) if len(scene) else []
        if len(displayed):
            ax.scatter(*scene[displayed].T, c=scene_colors[displayed].astype(float)/255., s=.4, depthshade=False)
    elif len(indices):
        ax.scatter(*points[indices].T, c=plotted_colors, s=.5, depthshade=False)
    goal_panel = failed_goal is not None
    boxes, pads3, shape3 = ((goal_boxes, goal_pads, goal_drawn) if goal_panel else
                           (actual_boxes, pads_world, drawn))
    hand_color, shape_color = ('darkorange', 'cyan') if goal_panel else ('royalblue', 'magenta')
    if len(robot_actual):
        boxes = np.empty((0, 8, 3))
        for group, color, style in ((robot_actual, 'royalblue', '-'), (robot_goal, 'darkorange', '--')):
            for box in group:
                for i in range(8):
                    for bit in (1, 2, 4):
                        j = i ^ bit
                        if i < j:
                            ax.plot(*box[[i, j]].T, color=color, linestyle=style, linewidth=.8)
        context = np.concatenate((scene, robot_actual.reshape(-1, 3), robot_goal.reshape(-1, 3)))
        lower3, upper3 = context.min(0), context.max(0)
        middle3 = (lower3+upper3)/2
        span3 = max(float(np.max(upper3-lower3)), .01)*1.08
    else:
        middle3, span3 = middle, span
    for box in boxes:
        for i in range(8):
            for bit in (1, 2, 4):
                j = i^bit
                if i < j:
                    ax.plot(*box[[i, j]].T, color=hand_color, linewidth=1.2)
    if len(pads3):
        ax.scatter(*pads3.T, color='darkorange' if goal_panel else 'red', marker='x', s=36)
    if len(shape3):
        ax.scatter(*shape3.T, color=shape_color, s=.6, alpha=.4, depthshade=False)
    for coordinate, centre in enumerate(middle3):
        getattr(ax, ('set_xlim', 'set_ylim', 'set_zlim')[coordinate])(centre-span3/2, centre+span3/2)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel('World X (m)'); ax.set_ylabel('World Y (m)'); ax.set_zlabel('World Z (m)')
    ax.set_title('Full robot: ACTUAL blue / recorded goal orange' if len(robot_actual) else
        'UNREACHED goal comparison' if goal_panel else 'CURRENT world 3D', fontsize=9)
    fig.suptitle('RGB: measured surfaces now | BLUE outline: ROBOT HAND | red: current pads\n'
        'Magenta: old observed shape propagated by current EEF, NOT a new measurement\n'+
        ('Dashed orange hand: recorded FAILED goal, NOT the current pose | cyan hypothesis only in 3D' if goal_panel else
         'Equal metres; no object identity, attachment or collision certification'), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .91))
    picture = folder/'hand_observation.png'
    fig.savefig(picture, dpi=100)
    plt.close(fig)
    return picture


def prepare_hand_observation(hub, description, folder, *, cloud_file=None, gripper_geometry=None,
                             failure=None, carried_local=None):
    """Prepare a current, robot-aware visual card from saved observed surfaces.

    Returns None when the hand model cannot supply a valid envelope. A supplied
    focused cloud must already be the current observation's world-frame cloud.
    """
    geometry = gripper_geometry or {}
    bounds = geometry.get('hand_bounds_in_eef_tool_m')
    pads = geometry.get('pad_centres_in_eef_tool_m')
    ik = getattr(getattr(hub, 'executor', None), 'ik', None)
    model = getattr(ik, 'model', None)
    data = getattr(ik, 'data', None)
    capable = (model is not None and data is not None and hasattr(model, 'ngeom')
               and hasattr(model, 'joint') and hasattr(ik, 'fk'))
    if bounds is None and not capable:
        return None
    source_step = description['source_step']
    observed_id = description['observation_id']
    live = hub.robot.state()
    if live['step'] != source_step or description['state']['step'] != source_step:
        raise ValueError('Hand card requires the current physical source_step')
    state = {**description['state'], **live}
    if bounds is None:
        from .robot_geometry import gripper_geometry_in_eef
        qpos_before = np.asarray(data.qpos).copy()
        try:
            for name, value in zip(state['finger_joint_names'], state['finger_joint_positions']):
                data.qpos[model.joint(name).qposadr[0]] = value
            origin, rotation = ik.fk(state['q'])
            geometry=gripper_geometry_in_eef(ik,origin,rotation,state) or {}
            bounds=geometry.get('hand_bounds_in_eef_tool_m');pads=geometry.get('pad_centres_in_eef_tool_m')
        finally:
            data.qpos[:] = qpos_before
            # FK is a cache update only; restore the actual robot pose as well.
            ik.fk(live['q'])
        if bounds is None:
            return None
    if cloud_file is not None:
        with np.load(cloud_file) as cloud:
            points = np.asarray(cloud['world_xyz_m']).copy()
            colors = np.asarray(cloud['rgb']).copy()
        source_kind = 'current focused cloud'
    else:
        if not capable:
            return None
        from .observed_clearance import ObservedClearance
        clearance = ObservedClearance(ik)
        try:
            frames = hub.observations[observed_id]
            if len(frames) != 2 or any(frame['step'] != source_step for frame in frames.values()):
                raise ValueError('Hand card requires current calibrated dual RGB-D frames')
            for frame in frames.values():
                clearance.add_frame(frame, state)
            clearance.build()
            points, colors = clearance.points.copy(), clearance.colors.copy()
        finally:
            ik.fk(live['q'])
        source_kind = 'current calibrated dual RGB-D'
    origin = np.asarray(state['position'], dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('Current cloud requires finite world XYZ metres')
    near = (np.all(np.abs(points[:, :2] - origin[:2]) <= .22, axis=1)
            & (np.abs(points[:, 2] - origin[2]) <= .25))
    episode=getattr(getattr(hub,'episode_state',None),'data',{})
    reference=getattr(hub,'carried_surface_reference',None)
    predicted=None
    associated=(state.get('gripper_command')==1. and getattr(hub,'carried_target_id',None)
        and episode.get('carry_premise')!='invalidated_by_contact_change' and not episode.get('contact_warning'))
    if (associated and reference
            and reference.get('object_id')==hub.carried_target_id):
        predicted=reference['local_xyz_m']@Rotation.from_quat(state['quaternion_xyzw']).as_matrix().T+origin
        if carried_local is None:carried_local=reference['local_xyz_m']
    outcome=episode.get('last_failed_stage_outcome')
    goal=failed_goal_context(failure,outcome)
    robot_geometry = None
    scene_points, scene_colors = None, None
    if capable and hasattr(ik, 'q_index'):
        from .robot_geometry import recorded_robot_geometry
        robot_geometry = recorded_robot_geometry(ik, state, goal)
        all_boxes = np.asarray([part['corners_world_m'] for part in
            robot_geometry['actual_parts']+robot_geometry['recorded_goal_parts']], float)
        if all_boxes.size:
            lower = all_boxes.reshape(-1, 3).min(0)-.08
            upper = all_boxes.reshape(-1, 3).max(0)+.08
            context = np.all((points >= lower) & (points <= upper), axis=1)
            scene_points, scene_colors = points[context], colors[context]
    parts=[item['bounds_in_eef_tool_m'] for item in
        [geometry.get('palm_and_fixed_hand') or {},*geometry.get('fingers',[])] if item.get('bounds_in_eef_tool_m')]
    if not associated:carried_local=None
    picture = render_hand_observation(points[near], colors[near], state, folder, bounds, pads,
        predicted_points_world=predicted,failed_goal=goal,hand_parts=parts or None,carried_local=carried_local,
        robot_geometry=robot_geometry,scene_points_world=scene_points,scene_rgb=scene_colors)
    metadata_file = Path(folder) / 'hand_observation.json'
    metadata = json.loads(metadata_file.read_text())
    metadata.update(observation_id=observed_id, source_kind=source_kind,
                    crop='EEF world-centred XY +/-0.22 m, Z +/-0.25 m')
    if predicted is not None:
        metadata['predicted_reference_source']={
            'kind':reference.get('source_kind','observed_component'),
            'source_step':reference.get('source_step'),
            'meaning':'Retained observed shape under the current grasp hypothesis, transformed by EEF. Actual RGB slip or separation overrides this prediction.'}
    if predicted is not None:
        metadata['evidence_layers']['magenta_shape']['shape_source_step'] = reference.get('source_step')
    if goal is not None:
        metadata['evidence_layers']['orange_hand_cyan_shape']['failure_observed_step'] = goal.get('failure_observed_step')
    metadata_file.write_text(json.dumps(metadata, indent=2))
    return {'image_file': str(picture), 'metadata': metadata}
