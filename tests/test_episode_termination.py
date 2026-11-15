from types import SimpleNamespace

import numpy as np
import pytest

from sketchik.primitives import MotionExecutor
from sketchik.task_execution import execute_sketch


def terminating_executor():
    class Robot:
        done=False
        steps=0
        max_steps=10000
        def state(self):
            return {'step':self.steps,'q':[0.]*7,'position':[0.,0.,1.],
                'quaternion_xyzw':[0.,0.,0.,1.],'gripper_command':1.,
                'finger_joint_positions':[0.,0.]}
        def step(self,*args):
            assert not self.done, 'Must never send another command after termination'
            self.steps+=1
            self.done=self.steps==2
    value=MotionExecutor.__new__(MotionExecutor)
    value.robot=Robot();value.settle_gripper=False;value.emit=lambda _:None
    value.ik=SimpleNamespace(cartesian_path=lambda *args:[
        SimpleNamespace(q=np.full(7,.3),elapsed_s=0.,position_error_m=0.,rotation_error_rad=0.)])
    return value


@pytest.mark.parametrize('primitive',['move','wait','set_gripper'])
def test_mid_primitive_termination_stops_without_success_inference(primitive):
    executor=terminating_executor()
    if primitive=='move':result=executor.move([.2,.3,1.1])
    elif primitive=='wait':result=executor.wait(10)
    else:result=executor.set_gripper('close',steps=10)
    assert executor.robot.steps==2 and result['episode_ended'] is True
    assert 'official_success' not in result and 'task_complete' not in result
    if primitive=='move':assert result['converged'] is False and result['actions']==2


def test_already_ended_motion_does_not_plan_or_send_another_command():
    executor=terminating_executor();executor.robot.done=True
    executor.ik.cartesian_path=lambda *_:pytest.fail('No new planning after termination')
    result=executor.move([.2,.3,1.1])
    assert result['episode_ended'] and result['actions']==0
    assert executor.robot.steps==0 and result['converged'] is False


def test_sketch_stops_suffix_when_episode_ends_during_move(tmp_path):
    robot=SimpleNamespace(done=False,steps=1,max_steps=100)
    def move(**kwargs):
        robot.done=True;robot.steps=2
        return {'episode_ended':True,'converged':False}
    receipts=[]
    hub=SimpleNamespace(robot=robot,run=tmp_path,emit=lambda _:None,
        executor=SimpleNamespace(move=move,wait=lambda **_:pytest.fail('Suffix dispatched')),
        observe=lambda:({'observation_id':'actual','source_step':robot.steps},[]))
    grounder=SimpleNamespace(compile_step=lambda *args:{'parameters':{'position':[.2,.3,1.1]}},
        record_execution=receipts.append)
    plan={'source_step':1,'subtask':'move then wait','steps':[
        {'primitive':'move'},{'primitive':'wait'}]}
    hub.episode_state=SimpleNamespace(begin_stage=lambda _:None)
    bound={'semantic_plan':plan,'plan':plan,'source_step':1,'needs_observation':False}
    result=execute_sketch(hub,grounder,plan,0,frozen_binding=bound)
    assert result['episode_ended'] and not result['commands_completed']
    assert result['remaining_suffix_executed'] is False and len(receipts)==1


def evaluation_adapter(diagnostic, outcomes, *, max_steps=3, backend_stop=None):
    """Exercise the actual joint adapter without simulator object state."""
    from sketchik.libero import LiberoRobot
    robot = LiberoRobot.__new__(LiberoRobot)
    robot.steps = 0
    robot.max_steps = max_steps
    robot.done = False
    robot.completion_diagnostic = diagnostic
    robot.first_official_success_step = None
    robot.gripper_command = -1.
    robot.on_step = None
    controller = SimpleNamespace(goal_qpos=np.zeros(7))
    physical = SimpleNamespace(steps=0, refreshed=-1)
    model = SimpleNamespace(jnt_range=np.tile([-2., 2.], (7, 1)))
    def advance(action):
        physical.steps += 1
        controller.goal_qpos = action[:7].copy()
        robot.env.done = physical.steps == backend_stop
        return None, 0., outcomes[min(physical.steps-1, len(outcomes)-1)], {}
    def forward():
        physical.refreshed = physical.steps
    robot.env = SimpleNamespace(done=False, step=advance,
        sim=SimpleNamespace(model=model, forward=forward))
    robot.robot = SimpleNamespace(arm_joint_indexes=np.arange(7),
        part_controllers={'right': controller})
    def state():
        assert physical.refreshed == physical.steps, 'FK must refresh before state/evaluation'
        return {'step': robot.steps, 'q': controller.goal_qpos.tolist(),
            'official_success': bool(outcomes[min(physical.steps-1, len(outcomes)-1)]),
            'done': robot.done}
    robot.state = state
    return robot, physical


def test_standard_trial_still_stops_on_first_official_done():
    robot, physical = evaluation_adapter(False, [True, False])
    robot.step(np.zeros(7))
    assert robot.done and physical.steps == 1
    with pytest.raises(RuntimeError, match='Episode is over'):
        robot.step(np.zeros(7))
    assert physical.steps == 1


def test_completion_diagnostic_continues_without_latching_final_success_or_leaking_metric():
    from sketchik.tools import public
    robot, physical = evaluation_adapter(True, [True, False, False])
    first = robot.step(np.zeros(7))
    assert first['official_success'] is True and not robot.done
    second = robot.step(np.zeros(7))
    assert second['official_success'] is False and not robot.done
    assert robot.first_official_success_step == 1
    third = robot.step(np.zeros(7))
    assert third['official_success'] is False and robot.done
    assert physical.steps == 3
    assert 'official_success' not in public(third)
    assert 'first_official_success_step' not in third
    with pytest.raises(RuntimeError, match='Episode is over'):
        robot.step(np.zeros(7))
    assert physical.steps == 3


def test_completion_diagnostic_retains_earlier_backend_termination():
    robot, physical = evaluation_adapter(True, [True, False], max_steps=10, backend_stop=2)
    robot.step(np.zeros(7))
    robot.step(np.zeros(7))
    assert robot.done and physical.steps == 2
    with pytest.raises(RuntimeError, match='Episode is over'):
        robot.step(np.zeros(7))


def test_completion_diagnostic_flag_is_explicit_boolean():
    from sketchik.libero import LiberoRobot
    with pytest.raises(ValueError, match='must be boolean'):
        LiberoRobot(completion_diagnostic='true')


@pytest.mark.parametrize('option',['--planner-layout','--b-api-max-tokens','--execution-source','--frozen-stage-source','--observe-only'])
def test_retired_branch_options_are_not_exposed(option,monkeypatch,capsys):
    from sketchik.task_trial import main
    monkeypatch.setattr('sys.argv',['task_trial','--task','0','--env-file','/private.env',
        '--gpu-uuid','unused','--output','/unused',option,'old'])
    with pytest.raises(SystemExit) as exc:main()
    assert exc.value.code==2
    assert 'unrecognized arguments' in capsys.readouterr().err


def test_completion_diagnostic_cli_passes_explicit_mode_to_fresh_single_runner(monkeypatch):
    from sketchik.task_trial import main
    received = []
    monkeypatch.setattr('sketchik.task_trial.run', lambda args: received.append(args))
    monkeypatch.setattr('sys.argv', ['task_trial', '--completion-diagnostic', '--task', '0',
        '--env-file', '/private.env', '--gpu-uuid', 'unused', '--output', '/unused'])
    main()
    assert len(received) == 1 and received[0].completion_diagnostic is True
