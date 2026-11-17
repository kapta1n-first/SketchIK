from copy import deepcopy

import numpy as np
import pytest

from sketchik.grasp_contact import assess_initial_contact, _visible_free, _pad_zone


def evidence():
    before = {'available': True, 'jaw_gap_m': .08, 'near_pad_surface_samples': 0,
        'visible_free_fraction': .95}
    after = {'available': True, 'jaw_gap_m': .002, 'near_pad_surface_samples': 0,
        'close_translation_m': .0002, 'close_rotation_rad': .0006}
    return before, after


def test_first_empty_close_detected_without_a_carry_template_or_contact_baseline():
    before, after = evidence()
    report = assess_initial_contact(before, after)
    assert report['status'] == 'contradicted'
    assert report['holding_certified'] is report['target_identity_certified'] is False
    assert report['whole_object_reference_required'] is False


@pytest.mark.parametrize('field,value', [
    ('visible_free_fraction', .2), ('near_pad_surface_samples', 1), ('jaw_gap_m', .002)])
def test_initial_occlusion_or_thin_preclose_surface_never_certifies_empty(field, value):
    before, after = evidence(); before[field] = value
    assert assess_initial_contact(before, after)['status'] == 'unknown'


@pytest.mark.parametrize('field,value', [
    ('jaw_gap_m', .012), ('close_translation_m', .03), ('close_rotation_rad', .2)])
def test_incomplete_closure_or_changed_pose_remains_unknown(field, value):
    before, after = evidence(); after[field] = value
    assert assess_initial_contact(before, after)['status'] == 'unknown'


def test_visible_thin_contact_is_supported_even_with_small_closed_gap():
    before, after = evidence(); after['near_pad_surface_samples'] = 1
    report = assess_initial_contact(before, after)
    assert report['status'] == 'contact_geometry_supported'
    assert not report['holding_certified'] and not report['target_identity_certified']


def test_missing_calibration_is_unknown_and_inputs_are_immutable():
    before, after = evidence(); saved = deepcopy((before, after))
    assert assess_initial_contact({'available': False}, after)['status'] == 'unknown'
    assess_initial_contact(before, after)
    assert (before, after) == saved


def frame(depth):
    return {'depth_m': np.asarray(depth, float), 'K': np.eye(3),
        'world_from_camera': np.eye(4), 'camera': 'test', 'step': 7}


def test_free_space_requires_valid_depth_behind_probe_not_occlusion_or_missing_depth():
    probes = np.array([[0., 0., 1.]])
    assert _visible_free(probes, [frame([[1.2]])])[0] == 1.
    for depth in [.8, 1., 0., np.nan, np.inf]:
        assert _visible_free(probes, [frame([[depth]])])[0] == 0.
    assert _visible_free(np.array([[2., 0., 1.]]), [frame([[1.2]])])[0] == 0.


def test_second_camera_can_reveal_free_space_hidden_from_first():
    probes = np.array([[0., 0., 1.]])
    assert _visible_free(probes, [frame([[.8]]), frame([[1.2]])])[0] == 1.


def test_contact_zone_uses_actual_inner_faces_and_common_pad_patch():
    geometry = {'fingers': [{'pads': [{'bounds_in_eef_tool_m': box}]} for box in [
        [[-.05, -.008, -.012], [-.04, .008, .004]],
        [[.04, -.006, -.01], [.05, .006, .005]]]]}
    low, high = _pad_zone(geometry)
    np.testing.assert_allclose(low, [-.04, -.006, -.01])
    np.testing.assert_allclose(high, [.04, .006, .004])
    assert _pad_zone({}) is None


def test_native_contact_inspection_restores_all_robot_dofs_even_on_stale_frame():
    from robosuite.models.robots import Panda
    from robosuite.models.grippers import PandaGripper
    from scipy.spatial.transform import Rotation
    from sketchik.kinematics import RobotIK
    from sketchik.grasp_contact import inspect_initial_contact
    arm, gripper = Panda(), PandaGripper()
    arm.add_gripper(gripper)
    ik = RobotIK(arm.get_xml(), arm.joints, gripper.important_sites['grip_site'])
    q = np.array([.2, -.13, 0., -2.44, 0., 2.2, .78])
    position, rotation = ik.fk(q)
    state = {'step': 7, 'q': q.tolist(), 'position': position.tolist(),
        'quaternion_xyzw': Rotation.from_matrix(rotation).as_quat().tolist(),
        'finger_joint_names': gripper.joints, 'finger_joint_positions': [.04, -.04]}
    saved = ik.data.qpos.copy()
    native = {**frame(np.full((12, 12), 2.)), 'rgb': np.zeros((12, 12, 3), np.uint8)}
    report = inspect_initial_contact(ik, state, [native], state, [native])
    assert report['robot_actions'] == report['model_requests'] == 0
    assert not report['holding_certified'] and not report['target_identity_certified']
    np.testing.assert_array_equal(ik.data.qpos, saved)
    with pytest.raises(ValueError, match='physical step'):
        inspect_initial_contact(ik, state, [native], {**state, 'step': 8}, [native])
    np.testing.assert_array_equal(ik.data.qpos, saved)
