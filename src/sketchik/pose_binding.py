"""Deterministic conversion of the agent's visual anchor/offset into an EEF pose."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation
from jsonschema import Draft202012Validator

from .tools import obj, V3, Q4, PARAMS


BINDING = obj({
    'anchor': obj({'kind': {'enum':['current_eef','cloud_point','image_point','opening']},
        'view_id': {'type':'string','minLength':1},
        'group_id': {'type':'integer','minimum':0},
        'observation_id': {'type':'string'}, 'camera': {'enum':['agentview','robot0_eye_in_hand']},
        'pixel_xy': {'type':'array','items':{'type':'integer','minimum':0,'maximum':1023},'minItems':2,'maxItems':2},
        'opening_index': {'type':'integer','minimum':0}}, ['kind']),
    'reference': {'enum':['eef','pad_0','pad_1','pad_midpoint','held_bottom','held_center'],
        'description':'Point to align with anchor+offset. held_center is the world-axis AABB centre of '
            'the associated observed carried points after applying the TARGET orientation; held_bottom '
            'uses that same XY centre and min Z, not a single lowest point with unknown XY. '
            'The program compensates the EEF offset. These are observed-shape references, not complete object geometry.'},
    'offset_m': V3, 'offset_frame': {'enum':['world','tool','opening']},
    'orientation': {'anyOf':[{'const':'preserve'},
        obj({'rpy_world_rad':V3},['rpy_world_rad']),
        obj({'rpy_delta_tool_rad':V3},['rpy_delta_tool_rad']),
        obj({'finger_direction':V3,'closing_axis':V3},['finger_direction','closing_axis'])]}},
    ['anchor','reference','offset_m','offset_frame','orientation'])

def compile_rotation(binding, packet, primitive):
    """Compile a single wrist/tilt angle from the actual entry orientation."""
    from .primitives import simple_rotation_target_pose
    if primitive not in ('rotate_roll','rotate_pitch','rotate_yaw'):raise ValueError('Unknown rotation tool')
    Draft202012Validator(PARAMS[primitive]).validate(binding)
    angles={key:value for key,value in binding.items() if key!='settle_steps'}
    pose=simple_rotation_target_pose(packet['eef'],primitive,**angles)
    return dict(binding),pose,{'binding':binding,'rotation_tool':primitive,
        'compiled_pose':pose,'source_step':packet['source_step'],'executed':False,
        'reference_is_complete_object_geometry':False}


def opening_basis(opening):
    from .kinematics import vector
    normal=vector(opening['proposed_surface_normal_world'],3,'opening normal')
    if np.linalg.norm(normal)<1e-8:raise ValueError('Opening normal must be nonzero')
    normal=normal/np.linalg.norm(normal)
    up=np.array([0.,0.,1.])-normal*normal[2]
    if np.linalg.norm(up)<.1:raise ValueError('Opening has no reliable gravity-aligned tangent')
    up/=np.linalg.norm(up)
    return np.column_stack([np.cross(up,normal),up,normal])


def check_opening_owner(opening, packet):
    """Do not bind an observed opening to a different A-selected object."""
    if 'operation_reference_target_id' in packet and opening.get('target_id') != packet['operation_reference_target_id']:
        raise ValueError('Observed opening target_id is missing or does not match operation_reference_target_id')


def compile_binding(binding, packet, resolve_frame, carried_local=None, cloud_views=None):
    """No task-specific coordinates; all references come from public evidence."""
    from .libero import back_project
    from .kinematics import quaternion, vector, rotation_from_directions
    # Historical numeric control probes can still be read; the live B schema
    # no longer asks a language model to produce a unit quaternion.
    legacy=isinstance(binding.get('orientation'),list)
    Draft202012Validator(BINDING).validate({**binding,'orientation':'preserve'} if legacy else binding)
    state=packet['eef']
    orientation=binding['orientation']
    if isinstance(orientation,dict) and 'rpy_world_rad' in orientation:
        quat=Rotation.from_euler('xyz',vector(orientation['rpy_world_rad'],3,'rpy_world_rad')).as_quat()
    elif isinstance(orientation,dict) and 'rpy_delta_tool_rad' in orientation:
        current=Rotation.from_quat(quaternion(state['quaternion_xyzw']))
        delta=Rotation.from_euler('xyz',vector(orientation['rpy_delta_tool_rad'],3,'rpy_delta_tool_rad'))
        quat=(current*delta).as_quat()
    elif isinstance(orientation,dict):
        quat=rotation_from_directions(orientation['finger_direction'],orientation['closing_axis'])
    else:
        quat=quaternion(state['quaternion_xyzw'] if orientation=='preserve' else orientation)
    rotation=Rotation.from_quat(quat).as_matrix()
    anchor=binding['anchor']
    opening=None
    provenance={}
    if anchor['kind']=='current_eef':
        point=vector(state['position'],3,'current EEF')
    elif anchor['kind']=='cloud_point':
        from .cloud_selection import resolve_cloud_point
        point,provenance=resolve_cloud_point(anchor,packet.get('cloud_views',[]),cloud_views or [])
        if provenance.get('binding_scope') == 'single_sample_context':
            provenance['semantic_owner_assertion'] = packet.get('operation_reference_target_id')
        retained = [value for value in packet.get('retained_cloud_point_bindings', [])
            if value['view_id'] == anchor['view_id'] and value['pixel_xy'] == anchor['pixel_xy']]
        if retained:
            value = retained[0]
            wrong_time = (value['source_step'] != packet['source_step']
                if value['kind'] == 'retained_grasp_point_current_eef' else value['source_step'] > packet['source_step'])
            if (any(item != value for item in retained) or wrong_time
                    or value['cloud_point_index'] != provenance['cloud_point_index']
                    or not np.array_equal(value['original_world_xyz_m'], point)):
                raise ValueError('Retained point binding differs from this frozen sample or physical step')
            point = vector(value['world_xyz_m'], 3, 'current retained sample')
            provenance.update(original_anchor_world_xyz_m=value['original_world_xyz_m'],
                coordinate_estimate=value, carry_reference_eligible=False,
                binding_scope='one_retained_grasp_sample')
    elif anchor['kind']=='image_point':
        key=(anchor.get('observation_id'),anchor.get('camera'))
        allowed={(v['observation_id'],v['camera']) for v in packet.get('image_references',[])}
        if key not in allowed:raise ValueError('B may select a pixel only on a supplied native RGB image')
        if 'pixel_xy' not in anchor:raise ValueError('image_point requires pixel_xy')
        point=back_project(resolve_frame(*key),anchor['pixel_xy'])
    else:
        index=anchor.get('opening_index')
        if index is None or index>=len(packet.get('observed_opening_frames',[])):
            raise ValueError('Missing observed opening reference')
        opening=packet['observed_opening_frames'][index]
        check_opening_owner(opening,packet)
        point=vector(opening['anchor_world_m'],3,'opening anchor')
    offset=vector(binding['offset_m'],3,'offset_m')
    if binding['offset_frame']=='tool':offset=rotation@offset
    elif binding['offset_frame']=='opening':
        if opening is None:
            openings=packet.get('observed_opening_frames',[])
            if len(openings)!=1:raise ValueError('Opening offset frame needs one unambiguous observed opening')
            opening=openings[0]
        check_opening_owner(opening,packet)
        offset=opening_basis(opening)@offset
    reference=binding['reference']
    if reference=='eef':reference_world=np.zeros(3)
    elif reference in ('pad_0','pad_1','pad_midpoint'):
        pads=packet.get('robot_gripper_geometry',{}).get('pad_centres_in_eef_tool_m',[])
        if not isinstance(pads,(list,tuple,np.ndarray)) or len(pads)!=2:
            raise ValueError('Known finger-pad geometry unavailable')
        centres=[vector(pad,3,'pad centre') for pad in pads]
        local=(centres[0]+centres[1])/2 if reference=='pad_midpoint' else centres[int(reference[-1])]
        reference_world=rotation@local
    else:
        if carried_local is None or len(carried_local)<25:
            raise ValueError('Observed carried surfaces unavailable; cannot bind held-object reference')
        oriented=np.asarray(carried_local)@rotation.T
        low,high=oriented.min(axis=0),oriented.max(axis=0)
        reference_world=(low+high)/2
        if reference=='held_bottom':reference_world[2]=low[2]
    pose={'position':(point+offset-reference_world).tolist(),'quaternion_xyzw':quat.tolist()}
    return pose,{**provenance,'binding':binding,'anchor_world_xyz_m':np.asarray(point).tolist(),
        'offset_world_m':offset.tolist(),'reference_offset_world_m':reference_world.tolist(),
        'compiled_pose':pose,'source_step':packet['source_step'],'executed':False,
        'reference_is_complete_object_geometry':False}
