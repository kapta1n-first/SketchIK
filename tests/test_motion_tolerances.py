"""Measured tracking thresholds, independently of numerical IK accuracy."""
from copy import deepcopy
from types import SimpleNamespace
import time

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from sketchik.primitives import MotionExecutor


def executor(position_error_m, rotation_error_deg=.89, joint_error_rad=.01166):
    class Robot:
        steps = 0
        max_steps = 100
        done = False

        def __init__(self):
            self.calls = []
            self.measured = {'position': [position_error_m, 0., 0.],
                'quaternion_xyzw': Rotation.from_euler('z', rotation_error_deg, degrees=True).as_quat().tolist(),
                'q': [joint_error_rad]+[0.]*6, 'gripper_command': -1.}

        def state(self):
            return {**deepcopy(self.measured), 'step': self.steps}

        def step(self, target):
            self.calls.append(np.asarray(target).copy())
            self.steps += 1

    motion = MotionExecutor.__new__(MotionExecutor)
    motion.robot = Robot(); motion.emit = lambda _: None
    return motion


@pytest.mark.parametrize('error,tolerance,accepted', [
    (.005457840017584937, None, False),  # Ordinary tracking restored to5mm; camera allowance remains explicit.
    (.005, None, True), (.00501, None, False),
    (.01199, .012, True), (.01201, .012, False),
    (.006, .005, False),
])
def test_measured_tracking_boundary_does_not_rewrite_actual_pose(error, tolerance, accepted):
    motion = executor(error); before = motion.robot.state()
    options = {} if tolerance is None else {'position_tolerance_m': tolerance}
    result = motion._track_joint_targets(np.zeros((1, 7)), np.zeros(3), [0, 0, 0, 1],
        np.zeros(7), 4, time.perf_counter(), -1., **options)
    assert result['converged'] is accepted
    assert result['state']['position'] == before['position']
    assert result['state']['quaternion_xyzw'] == before['quaternion_xyzw']
    assert result['position_error_m'] == pytest.approx(error)
    assert result['tracking_tolerances'] == {'position_m': .005 if tolerance is None else tolerance,
        'orientation_deg': 3., 'max_joint_rad': .02}
    if not accepted:
        assert result['continuation_goal']['position_tolerance_m'] == result['tracking_tolerances']['position_m']
    assert all(np.array_equal(target, np.zeros(7)) for target in motion.robot.calls)


@pytest.mark.parametrize('rotation,joint', [(3.01, .01), (.89, .02001)])
def test_camera_position_allowance_keeps_rotation_and_joint_checks(rotation, joint):
    motion = executor(.00545784, rotation, joint)
    result = motion._track_joint_targets(np.zeros((1, 7)), np.zeros(3), [0, 0, 0, 1],
        np.zeros(7), 1, time.perf_counter(), -1., position_tolerance_m=.012)
    assert not result['converged']


@pytest.mark.parametrize('tolerance', [0, -.01, float('inf'), float('nan'), True, .021])
def test_invalid_tracking_allowance_is_rejected_before_physics(tolerance):
    motion = executor(.005)
    with pytest.raises(ValueError, match='position_tolerance_m'):
        motion.move_waypoints([{'position': [0, 0, 0]}], position_tolerance_m=tolerance)
    assert motion.robot.steps == 0


def test_continuation_retains_the_failed_motion_allowance():
    motion = executor(.0115, 0., .03)
    motion.ik = SimpleNamespace(lower=np.full(7, -2.), upper=np.full(7, 2.))
    original_step = motion.robot.step
    def settle_joint(target):
        original_step(target)
        motion.robot.measured['q'] = list(target)
    motion.robot.step = settle_joint
    failure = {'converged': False, 'motion_progress': {'recent_progress': True},
        'continuation_goal': {'position': [0, 0, 0], 'quaternion_xyzw': [0, 0, 0, 1],
            'joint_target': [0]*7, 'gripper_command': -1., 'source_step': 0,
            'position_tolerance_m': .012}}
    result = motion.continue_motion(failure, max_steps=4)
    assert result['converged'] and result['actions'] == 1
    assert result['position_error_m'] == pytest.approx(.0115)
    assert result['tracking_tolerances']['position_m'] == .012
