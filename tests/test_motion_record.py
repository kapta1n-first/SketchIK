"""One recorded joint branch, exact entrance, and measured tracking evidence."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from sketchik.kinematics import Solution
from sketchik.motion_record import MotionTrace
from sketchik.primitives import MotionExecutor
from sketchik.task_execution import previous_stage_outcome


def motion_fixture():
    class Robot:
        steps = 30
        max_steps = 1000
        done = False
        joint_names = [f'joint_{i}' for i in range(7)]

        def __init__(self):
            self.measured = {'q': [0.]*7, 'position': [0., 0., 1.],
                'quaternion_xyzw': [0., 0., 0., 1.], 'gripper_command': -1.,
                'finger_joint_names': ['left', 'right'], 'finger_joint_positions': [.04, -.04]}
            self.commands = []

        def state(self):
            return {**deepcopy(self.measured), 'step': self.steps}

        def step(self, target):
            self.commands.append(np.asarray(target).copy())
            self.steps += 1
            self.measured['q'] = list(target)

    robot = Robot()
    plans, cache_updates, events = [], [], []

    def path(q, position, quaternion):
        plans.append((deepcopy(q), deepcopy(position), deepcopy(quaternion)))
        return [Solution(np.full(7, .025), 1e-7, 1e-7, .01, 1)]

    motion = MotionExecutor.__new__(MotionExecutor)
    motion.robot = robot
    motion.ik = SimpleNamespace(cartesian_path=path, fk=lambda q: cache_updates.append(list(q)))
    motion.emit = events.append
    return motion, plans, cache_updates, events


def test_prepared_motion_dispatches_the_recorded_branch_once_without_replanning():
    motion, plans, cache_updates, events = motion_fixture()
    source = motion.robot.state()
    prepared = motion.prepare_motion([{'position': source['position'],
        'quaternion_xyzw': source['quaternion_xyzw']}], settle_steps=0)
    assert motion.robot.state() == source and not motion.robot.commands
    assert cache_updates[-1] == source['q'] and len(plans) == 1
    original = deepcopy(prepared)
    result = motion.execute_prepared_motion(prepared)
    np.testing.assert_array_equal(motion.robot.commands, prepared['trajectory'])
    assert len(plans) == 1 and prepared == original
    assert result['converged'] and result['compiled_motion_id'] == prepared['artifact_id']
    assert events[0]['compiled_motion_id'] == events[-1]['compiled_motion_id'] == prepared['artifact_id']
    assert result['tracking']['goal_joint_positions_rad'] == prepared['planned']['joint_target_final']
    # An old compiled branch cannot be reused at a later real physical state.
    commands = len(motion.robot.commands)
    with pytest.raises(ValueError, match='entrance state'):
        motion.execute_prepared_motion(prepared)
    assert len(motion.robot.commands) == commands


def test_internal_numpy_waypoints_are_recorded_without_changing_the_numeric_goal():
    motion, plans, _, _ = motion_fixture()
    point = np.array([0., 0., 1.])
    quaternion = np.array([0., 0., 0., 1.])
    prepared = motion.prepare_motion([{'position': point, 'quaternion_xyzw': quaternion}], settle_steps=0)
    assert prepared['planned']['waypoints'][0] == {'position': point.tolist(), 'quaternion_xyzw': quaternion.tolist()}
    assert motion.execute_prepared_motion(prepared)['converged']
    np.testing.assert_array_equal(point, [0., 0., 1.])
    assert len(plans) == 1


@pytest.mark.parametrize('changed', ['trajectory', 'q', 'step', 'fingers', 'gripper', 'executor'])
def test_changed_record_or_entrance_rejects_before_any_physics(changed):
    motion, plans, _, _ = motion_fixture()
    prepared = motion.prepare_motion([{'position': [0, 0, 1]}], settle_steps=0)
    if changed == 'trajectory':
        prepared['trajectory'][0][0] += .001
    elif changed == 'q':
        motion.robot.measured['q'][0] += 1e-10
    elif changed == 'step':
        motion.robot.steps += 1
    elif changed == 'fingers':
        motion.robot.measured['finger_joint_positions'][0] -= .001
    elif changed == 'gripper':
        motion.robot.measured['gripper_command'] = 1.
    else:
        motion = deepcopy(motion)
    with pytest.raises(ValueError, match='record changed|entrance state|another executor'):
        motion.execute_prepared_motion(prepared)
    assert motion.robot.commands == [] and len(plans) == 1


def test_tracking_distinguishes_current_setpoint_lag_from_final_goal_and_plateau():
    motion, _, _, _ = motion_fixture()
    trace = MotionTrace(motion.robot.state(), np.full(7, .5), motion.robot.joint_names)
    # The remote final goal does not count as lag against an intermediate setpoint.
    motion.robot.steps += 1
    trace.sample(motion.robot.state(), np.zeros(7), 'trajectory')
    assert trace.peak_setpoint_error == 0
    for _ in range(81):
        motion.robot.steps += 1
        trace.sample(motion.robot.state(), np.full(7, .5), 'settling')
    result = trace.summary(motion.robot.state())
    assert result['samples'] == 82 and result['final_goal_commanded']
    assert result['last_command_consecutive_steps'] == 81
    assert result['max_goal_joint_error_rad'] == .5
    assert result['actual_joint_positions_rad'] == [0.]*7
    assert result['recent_measured_window']['position_axis_ranges_m'] == [0., 0., 0.]
    assert result['recent_measured_window']['joint_ranges_rad'] == [0.]*7
    assert 'converged' not in result and 'collision' not in result


def test_compact_outcome_preserves_recorded_joint_branch_and_tracking_without_mutation():
    motion, _, _, _ = motion_fixture()
    state = motion.robot.state()
    tracking = MotionTrace(state, np.full(7, .5), motion.robot.joint_names).summary(state)
    receipt = {'steps': [{'primitive': 'move', 'parameters': {'position': [0, 0, 1],
        'quaternion_xyzw': [0, 0, 0, 1]}, 'result': {'converged': False,
        'tracking': tracking, 'state': state, 'compiled_motion_id': 'recorded'}}]}
    before = deepcopy(receipt)
    result = previous_stage_outcome(receipt)['motions'][0]
    assert result['goal']['joint_target'] == [.5]*7
    assert result['actual_stop']['q'] == [0.]*7
    assert result['tracking'] == tracking and result['compiled_motion_id'] == 'recorded'
    result['tracking']['actual_joint_positions_rad'][0] = 99
    assert receipt == before


def test_failed_relative_rotation_outcome_keeps_its_recorded_absolute_goal():
    goal = {'position': [.1, .2, 1.], 'quaternion_xyzw': [0, 0, 0, 1], 'joint_target': [.1]*7}
    receipt = {'steps': [{'primitive': 'rotate_yaw', 'parameters': {'angle_rad': .1},
        'result': {'converged': False, 'continuation_goal': goal, 'state': {'step': 4}}}]}
    assert previous_stage_outcome(receipt)['motions'][0]['goal'] == goal
