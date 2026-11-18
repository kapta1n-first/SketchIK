import numpy as np
import pytest

from sketchik.primitives import MotionExecutor


def executor(settling):
    class Robot:
        steps = 0
        max_steps = 200
        finger = .04
        command = -1
        def state(self):
            return {'q': [0.]*7, 'finger_joint_positions': [self.finger, -self.finger], 'step': self.steps}
        def step(self, q, command):
            self.steps += 1
            self.command = command
            target = 0. if command == 1 else .04
            self.finger += .5*(target-self.finger)
    value = MotionExecutor.__new__(MotionExecutor)
    value.robot = Robot(); value.emit = lambda event: None; value.settle_gripper = settling
    return value


def test_online_one_step_close_waits_for_finger_state_but_does_not_claim_held():
    value = executor(True)
    state = value.set_gripper('close', steps=1)
    assert 12 <= value.robot.steps <= 60
    assert max(abs(np.asarray(state['finger_joint_positions']))) < .0001
    assert 'holding' not in state


@pytest.mark.parametrize('steps',[1,10])
def test_short_open_is_not_a_target_partial_gap(steps):
    value=executor(True);value.robot.finger=0.
    facts=value.gripper_control_context()
    state=value.set_gripper('open',steps=steps)
    assert facts['target_gap_control_available'] is False
    assert facts['last_command_continues_during_motion'] is True
    assert facts['steps_role']=='minimum_dwell_then_finger_settling'
    assert state['finger_joint_positions'][0]>.0399
    assert value.robot.command==-1
    assert value.robot.steps>=max(12,steps)


def test_legacy_duration_and_insufficient_settling_budget_are_preserved():
    value = executor(False)
    value.set_gripper('close', steps=1)
    assert value.robot.steps == 1
    value = executor(True); value.robot.max_steps = 5
    with pytest.raises(ValueError, match='settling budget'):
        value.set_gripper('close', steps=1)
    assert value.robot.steps == 0
