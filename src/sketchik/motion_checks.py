"""Explicit motion checks and read-only robot-model feasibility evidence."""
from __future__ import annotations

import json


def motion_route(hub, clearance, state, pose):
    if not getattr(hub, 'ik_only', False):
        return clearance.plan(state, pose)
    # This ablation keeps endpoint and full-pose path IK. Geometry can still
    # support perception, but contributes no motion rejection or detour.
    ik = hub.executor.ik
    direct = {key: pose[key] for key in ('position', 'quaternion_xyzw')}
    ik.solve(direct['position'], direct['quaternion_xyzw'], state['q'])
    path = ik.cartesian_path(state['q'], direct['position'], direct['quaternion_xyzw'])
    return {'waypoints': [direct], 'mode': 'ik_only', 'ik_pass': True,
            'path_checked': True, 'ik_samples': len(path),
            'final_joint_positions':path[-1].q.tolist(),
            'checked_configurations': 0, 'collision_checked': False,
            'observed_surface_checked': False, 'observed_clearance_passed': None,
            'continuous_collision_certified': False, 'carried_volume_proxy_checked': False}


def observation_return_route(hub, clearance, state, home, since_step):
    """Recheck recorded open-gripper observation poses; never replay joint commands.

    This is only an initial-observation reset fallback. All segments are solved
    from the actual entry and preceding planned joints before any action, with
    the caller's existing collision mode and the exact original full endpoint.
    """
    if state.get('gripper_command') != -1.:
        raise ValueError('Recorded observation return requires the initial open-gripper phase')
    anchors=[]
    for identity,scene in hub.journal.data.get('scene_clouds',{}).items():
        source=scene['source_step']
        if not since_step<=source<state['step']:continue
        metadata=json.loads((hub.run/'observations'/identity/'observation.json').read_text())
        saved=metadata['state']
        if saved['step']!=source or saved.get('gripper_command')!=-1.:
            raise ValueError('Return anchor must be a same-step open-gripper observation')
        anchors.append({'observation_id':identity,'source_step':source,
            'position':saved['position'],'quaternion_xyzw':saved['quaternion_xyzw']})
    anchors.sort(key=lambda item:item['source_step'],reverse=True)
    if not anchors:raise ValueError('No recorded observation stops available for return')
    # Preserve the actual initial endpoint; the last measured home stop may
    # have a small tracking error and is not a replacement for this target.
    anchors.append({'source_step':home['step'],'source':'executor.initial_state',
        **{key:home[key] for key in ('position','quaternion_xyzw')}})
    if len(anchors)>30:raise ValueError('Recorded observation return exceeds the waypoint bound')
    planned=dict(state);waypoints=[];checks=[]
    for anchor in anchors:
        pose={key:anchor[key] for key in ('position','quaternion_xyzw')}
        checked=motion_route(hub,clearance,planned,pose)
        if checked.get('mode')=='ik_only':
            # The full path was just solved; use its terminal joints to check
            # the next segment rather than solving this identical path twice.
            planned.update(q=checked['final_joint_positions'],**pose)
            waypoints.extend(checked['waypoints'])
            checks.append({key:checked.get(key) for key in
                ('mode','ik_pass','path_checked','collision_checked','observed_clearance_passed')})
            continue
        for point in checked['waypoints']:
            path=hub.executor.ik.cartesian_path(planned['q'],point['position'],point['quaternion_xyzw'])
            planned.update(q=path[-1].q,position=point['position'],quaternion_xyzw=point['quaternion_xyzw'])
            waypoints.append(point)
        checks.append({key:checked.get(key) for key in
            ('mode','ik_pass','path_checked','collision_checked','observed_clearance_passed')})
        if len(waypoints)>30:raise ValueError('Checked observation return exceeds the waypoint bound')
    return {'waypoints':waypoints,'strategy':'reverse_recorded_observation_stops',
        'source_step':state['step'],'recorded_anchors':anchors,'segment_checks':checks,
        'ik_pass':True,'path_checked':True,'collision_checked':not getattr(hub,'ik_only',False),
        'continuous_collision_certified':False,'robot_actions_during_planning':0,
        'gripper_command_preserved':True,'environment_reset':False}
