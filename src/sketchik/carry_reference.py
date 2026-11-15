"""Identity and validity of one observed point set within a real grasp.

A failed association never licenses the nearest scene fragment to replace it.
All records are episode-local hypotheses, not attachment certificates.
"""
from __future__ import annotations

import hashlib
import numpy as np


def initialize_reference(record, local_xyz_m, rgb, grasp_epoch, point_ids, *, world_xyz_m=None):
    points = np.asarray(local_xyz_m, dtype=float).copy()
    ids = np.asarray(point_ids, dtype=np.int64).copy()
    digest = hashlib.sha256()
    digest.update(str(record['object_id']).encode())
    digest.update(str(grasp_epoch).encode())
    digest.update(str(record.get('source_cloud_file', '')).encode())
    digest.update(ids.tobytes())
    digest.update(points.tobytes())
    frozen_world = None if world_xyz_m is None else np.asarray(world_xyz_m, dtype=float).copy()
    if frozen_world is not None and (frozen_world.shape != points.shape or not np.isfinite(frozen_world).all()):
        raise ValueError('Original world samples must align with the carried reference')
    return {**record, 'original_world_xyz_m': frozen_world,
        'original_world_sha256': None if frozen_world is None else hashlib.sha256(frozen_world.tobytes()).hexdigest(),
        'reference_id': digest.hexdigest(), 'grasp_epoch': grasp_epoch,
        'relation_revision': 0, 'validity': 'active', 'point_ids': ids,
        'local_xyz_m': points, 'original_local_xyz_m': points.copy(), 'rgb': None if rgb is None else np.asarray(rgb).copy()}


def suspend(reference, source_step, reason, *, kind='depth_contradiction'):
    """Keep the original samples, transform and first contradiction evidence."""
    if reference.get('validity') != 'suspended' or kind == 'relative_pose_changed':
        reference.update(validity='suspended', suspended_at_step=int(source_step),
            suspension_kind=kind, suspension_reason=str(reason))
        reference.pop('reviewed_at_step', None)
        reference.pop('review_pose_relation', None)


def assess(reference, source_step, pose_relation):
    if pose_relation == 'changed':
        if reference.get('validity')=='active' and reference.get('relocalized_at_step')==int(source_step):
            return  # Same physical event already has a supported revised pose.
        suspend(reference, source_step, 'A observed a changed object-to-hand relation; '
            'the old transform requires explicit relocalization or a new grasp.', kind='relative_pose_changed')
    elif reference.get('validity') == 'suspended':
        reference.update(reviewed_at_step=int(source_step), review_pose_relation=pose_relation)


def can_test(reference):
    """Only A review plus new support of these same points may lift suspension."""
    return reference.get('validity') != 'suspended' or (
        reference.get('suspension_kind') == 'depth_contradiction'
        and reference.get('review_pose_relation') == 'unchanged'
        and reference.get('reviewed_at_step', -1) >= reference['suspended_at_step'])


def report(reference):
    return {key: reference[key] for key in ('reference_id', 'grasp_epoch', 'relation_revision',
        'validity', 'suspended_at_step', 'suspension_kind', 'suspension_reason',
        'reviewed_at_step', 'review_pose_relation', 'restored_at_step', 'reference_scope',
        'relocalized_at_step', 'relation_basis') if key in reference}


def pose_rebind_identity_error(reference, object_id, grasp_epoch, current_state):
    """Check the retained grasp identity, independently of matching readiness."""
    if not reference:
        return 'no_original_carried_reference'
    if reference.get('object_id') != object_id:
        return 'different_object'
    if reference.get('grasp_epoch') != grasp_epoch:
        return 'grasp_epoch_changed'
    if current_state.get('gripper_command') != 1.:
        return 'current_gripper_not_commanded_closed'
    if reference.get('validity') not in ('active', 'suspended'):
        return 'original_reference_retired'
    return None


def relocalization_readiness(reference, *, object_id, grasp_epoch, explicit_pose_rebind,
                             point_set_scope, current_state):
    reason = pose_rebind_identity_error(reference, object_id, grasp_epoch, current_state)
    if reason:
        return reason
    if explicit_pose_rebind is not True:
        return 'explicit_current_pose_rebind_required'
    if point_set_scope != 'one_rigid_object':
        return 'selected_patch_is_mixed_or_unknown'
    if reference.get('validity') != 'suspended':
        return 'no_suspended_relation_to_rebuild'
    if reference.get('original_world_xyz_m') is None:
        return 'original_world_capture_not_retained'
    return None


def relocalize_reference(reference, *, object_id, grasp_epoch, explicit_pose_rebind,
                         point_set_scope, registration, frozen_world_xyz_m, current_state):
    """Opt-in pose revision of the same frozen samples within one grasp.

    Internal numerical evidence only. This does not identify an object, certify
    attachment, acknowledge a contact warning, clear old world anchors, or seed
    a new point set. The caller must obtain the original world samples by the
    stored source/IDs and supply registration evaluated on the current patch.
    """
    unavailable = lambda reason: {'status': 'unavailable', 'reason': reason,
        'robot_actions': 0, 'agent_requests': 0, 'holding_certified': False}
    reason = relocalization_readiness(reference, object_id=object_id, grasp_epoch=grasp_epoch,
        explicit_pose_rebind=explicit_pose_rebind, point_set_scope=point_set_scope,
        current_state=current_state)
    if reason:
        return unavailable(reason)
    if (registration.get('status') != 'accepted' or
            registration.get('search_complete') is not True):
        return unavailable('no_supported_unique_registration')
    if registration.get('source_step') != current_state['step']:
        return unavailable('registration_predates_current_snapshot')
    captured = reference.get('original_world_xyz_m')
    if captured is None:
        return unavailable('original_world_capture_not_retained')
    original = reference.get('original_local_xyz_m')
    if original is None:
        return unavailable('original_reference_samples_not_retained')
    world = np.asarray(frozen_world_xyz_m, dtype=float)
    original = np.asarray(original, dtype=float)
    if (world.shape != original.shape or world.ndim != 2 or world.shape[1] != 3
            or len(world) != len(reference['point_ids']) or not np.isfinite(world).all()):
        return unavailable('original_sample_identity_or_geometry_mismatch')
    if hashlib.sha256(world.tobytes()).hexdigest() != reference.get('original_world_sha256'):
        return unavailable('original_world_capture_or_sample_order_mismatch')
    # A new patch never becomes the carried shape. The supplied old world
    # samples must be rigidly identical, in their original order, to the frozen
    # original reference, including samples currently occluded.
    from .object_registration import rigid_fit, DegenerateRigidFit
    try:
        check_r, check_t = rigid_fit(world, original)
    except DegenerateRigidFit:
        return unavailable('original_reference_is_geometrically_degenerate')
    if np.max(np.linalg.norm(world@check_r.T+check_t-original, axis=1)) > 1e-6:
        return unavailable('original_sample_identity_or_geometry_mismatch')
    transform = registration['transform']
    rotation = np.asarray(transform['rotation'], dtype=float)
    translation = np.asarray(transform['translation_m'], dtype=float)
    if (rotation.shape != (3, 3) or translation.shape != (3,)
            or not np.isfinite(rotation).all() or not np.isfinite(translation).all()
            or not np.allclose(rotation.T@rotation, np.eye(3), atol=1e-7, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-7, rtol=0)):
        raise ValueError('Accepted registration must provide a finite proper rigid transform')
    from scipy.spatial.transform import Rotation
    origin = np.asarray(current_state['position'], dtype=float)
    quat = np.asarray(current_state['quaternion_xyzw'], dtype=float)
    if (origin.shape != (3,) or quat.shape != (4,) or not np.isfinite(origin).all()
            or not np.isfinite(quat).all() or not np.isclose(np.linalg.norm(quat), 1., atol=1e-7)):
        raise ValueError('Current robot pose must be finite with a unit xyzw quaternion')
    updated = (world@rotation.T+translation-origin)@Rotation.from_quat(quat).as_matrix()
    # All validation precedes the single relation update. Original samples, IDs,
    # colors, source capture step and reference identity remain unchanged.
    previous = {key: reference[key] for key in ('relation_revision', 'validity',
        'suspended_at_step', 'suspension_kind', 'suspension_reason') if key in reference}
    previous.update(retired_at_step=int(current_state['step']),
        local_points_sha256=hashlib.sha256(reference['local_xyz_m'].tobytes()).hexdigest())
    reference.setdefault('relation_history', []).append(previous)
    reference.update(local_xyz_m=updated, relation_revision=reference['relation_revision']+1,
        validity='active', relocalized_at_step=int(current_state['step']),
        relation_basis='Explicit current clean-patch RGB-D registration of original samples',
        rigid_attachment_verified=False)
    for key in ('suspended_at_step', 'suspension_kind', 'suspension_reason',
            'reviewed_at_step', 'review_pose_relation', 'restored_at_step'):
        reference.pop(key, None)
    return {'status': 'relocalized', 'reference_id': reference['reference_id'],
        'grasp_epoch': reference['grasp_epoch'], 'relation_revision': reference['relation_revision'],
        'source_step': int(current_state['step']), 'points': len(updated),
        'original_points_preserved': True, 'holding_certified': False,
        'robot_actions': 0, 'agent_requests': 0,
        'scope': 'Same original samples at a newly estimated object-to-hand relation. '
            'Old world-point invalidations and contact warnings remain independent.'}
