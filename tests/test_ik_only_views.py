"""View execution keeps observed-clearance filtering optional and explicit."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from test_table_scan import scan_hub

from sketchik.table_scan import run_table_scan
from sketchik.task_trial import reset_observation_arm


def _route_ik(calls):
    def solve(position, quaternion, seed):
        calls.append(('solve', list(position), list(quaternion), np.asarray(seed).tolist()))
        return SimpleNamespace(q=np.asarray(seed).copy())

    def path(seed, position, quaternion):
        calls.append(('path', list(position), list(quaternion), np.asarray(seed).tolist()))
        return [SimpleNamespace(q=np.asarray(seed).copy())]

    return SimpleNamespace(solve=solve, cartesian_path=path)


def test_table_view_ik_only_uses_ik_without_observed_clearance(scan_hub, monkeypatch):
    hub, _, records, _ = scan_hub
    calls = []
    hub.ik_only = True
    hub.executor.ik = _route_ik(calls)

    def reject(*_args):
        raise AssertionError('Observed clearance must not be called')

    monkeypatch.setattr('sketchik.observed_clearance.ObservedClearance.plan', reject, raising=False)
    result, _ = run_table_scan(hub)
    assert result['captured_views'] == 5
    assert len(calls) == 10
    assert [call[0] for call in calls] == ['solve', 'path'] * 5
    assert records['plans'] == []
    assert result['observed_surface_collision_filter'] is False
    for attempt in result['attempts']:
        assert attempt['checks'][0]['observed_clearance_passed'] is None
        assert attempt['checks'][0]['collision_checked'] is False
        assert attempt['checked_route']['mode'] == 'ik_only'


def test_default_table_view_still_rejects_observed_collision(scan_hub, monkeypatch):
    hub, _, records, clearance = scan_hub

    def reject(self, state, candidate):
        records['plans'].append((state, candidate))
        raise ValueError('observed collision')

    monkeypatch.setattr(clearance, 'plan', reject)
    result, _ = run_table_scan(hub, max_ik_candidates=1)
    assert result['captured_views'] == 0
    assert len(records['plans']) == 5
    assert records['moves'] == []
    assert result['observed_surface_collision_filter'] is True


def _reset_hub(tmp_path, *, ik_only):
    state = {'step': 4, 'q': [0.] * 7, 'position': [0., 0., 0.],
             'quaternion_xyzw': [0., 0., 0., 1.]}
    home = {**state, 'step': 0, 'position': [0.1, 0., 0.]}
    calls = []
    robot = SimpleNamespace(steps=4, state=lambda: state,
                            camera=lambda camera: calls.append(('camera', camera)))
    executor = SimpleNamespace(ik=_route_ik(calls), initial_state=home,
                               reset=lambda **kwargs: calls.append(('reset', kwargs)) or {'converged': True})
    hub = SimpleNamespace(run=tmp_path, ik_only=ik_only, robot=robot, executor=executor,
                          journal=SimpleNamespace(data={'scene_clouds': {'old': {}}}),
                          observations={}, fatal=False,
                          observe_scene=lambda: ({'observation_id': 'current'}, ['image']),
                          emit=lambda event: calls.append(('event', event['event'])))
    return hub, calls


def test_reset_ik_only_skips_history_and_clearance_but_executes_checked_route(tmp_path, monkeypatch):
    hub, calls = _reset_hub(tmp_path, ik_only=True)

    def forbidden(*args, **kwargs):
        raise AssertionError('historical frames and clearance must not be loaded')

    monkeypatch.setattr('sketchik.observed_clearance.ObservedClearance', forbidden)
    monkeypatch.setattr('sketchik.perception.saved_frame', forbidden)
    result, pictures = reset_observation_arm(hub)
    assert [call[0] for call in calls] == ['solve', 'path', 'reset', 'event']
    assert calls[2][1]['checked_waypoints'] == result['checked_route']['waypoints']
    assert result['checked_route']['mode'] == 'ik_only'
    assert result['observation_id'] == 'current' and pictures == ['image']
    assert not hub.fatal


@pytest.mark.parametrize('failed_call', ['solve', 'path'])
def test_reset_ik_only_ik_failure_sends_no_reset(tmp_path, monkeypatch, failed_call):
    hub, calls = _reset_hub(tmp_path, ik_only=True)

    def failure(*args):
        calls.append((failed_call,))
        raise ValueError('IK unavailable')

    setattr(hub.executor.ik, 'solve' if failed_call == 'solve' else 'cartesian_path', failure)
    with pytest.raises(ValueError, match='IK unavailable'):
        reset_observation_arm(hub)
    assert not any(call[0] == 'reset' for call in calls)
    assert hub.fatal
    assert calls[-1] == ('event', 'observation_arm_reset_failed')


def test_reset_default_still_builds_saved_observation_surface(tmp_path, monkeypatch):
    hub, calls = _reset_hub(tmp_path, ik_only=False)
    observation = tmp_path / 'observations' / 'old'
    observation.mkdir(parents=True)
    (observation / 'observation.json').write_text(json.dumps({'state': hub.robot.state()}))
    frames = []

    class Clearance:
        def __init__(self, ik):
            assert ik is hub.executor.ik
            calls.append(('clearance',))

        def add_frame(self, frame, state):
            frames.append((frame, state))

        def build(self):
            calls.append(('build',))

        def plan(self, state, pose):
            calls.append(('plan',))
            return {'waypoints': [pose], 'mode': 'observed_surfaces'}

    monkeypatch.setattr('sketchik.observed_clearance.ObservedClearance', Clearance)
    monkeypatch.setattr('sketchik.perception.saved_frame',
                        lambda run, identity, camera: f'{identity}/{camera}')
    result, _ = reset_observation_arm(hub)
    assert [frame for frame, _ in frames] == ['old/agentview', 'old/robot0_eye_in_hand']
    assert all(state == hub.robot.state() for _, state in frames)
    assert [call[0] for call in calls] == ['clearance', 'build', 'plan', 'reset', 'event']
    assert result['checked_route']['mode'] == 'observed_surfaces'


def test_observation_reset_uses_one_checked_return_route_after_direct_ik_rejection(tmp_path,monkeypatch):
    from sketchik.kinematics import IKError
    hub,calls=_reset_hub(tmp_path,ik_only=True)
    hub.robot.state()['gripper_command']=-1.
    folder=tmp_path/'observation_resets';folder.mkdir()
    (folder/'reset_000.json').write_text(json.dumps({'source_step':2,'execution':{'converged':True}}))
    def reject(*args):raise IKError('Discontinuous IK branch; no trajectory dispatched')
    monkeypatch.setattr(hub.executor.ik,'cartesian_path',reject)
    home=hub.executor.initial_state
    route={'waypoints':[{key:home[key] for key in ('position','quaternion_xyzw')}],
        'strategy':'reverse_recorded_observation_stops','collision_checked':False,'path_checked':True}
    checked=[]
    def fallback(actual_hub,clearance,state,target,since):
        assert actual_hub is hub and clearance is None
        assert state['step']==4 and target is home and since==2
        checked.append(True);return route
    monkeypatch.setattr('sketchik.motion_checks.observation_return_route',fallback)
    result,_=reset_observation_arm(hub)
    assert checked==[True]
    assert [call[0] for call in calls].count('reset')==1
    assert next(call[1]['checked_waypoints'] for call in calls if call[0]=='reset')==route['waypoints']
    assert result['checked_route']==route and 'Discontinuous' in result['direct_ik_rejection']
    assert not hub.fatal


def test_failed_observation_return_is_terminal_without_any_reset_dispatch(tmp_path,monkeypatch):
    from sketchik.kinematics import IKError
    hub,calls=_reset_hub(tmp_path,ik_only=True)
    def reject(*args):raise IKError('Same strict continuity bound rejected the return')
    monkeypatch.setattr(hub.executor.ik,'cartesian_path',reject)
    monkeypatch.setattr('sketchik.motion_checks.observation_return_route',reject)
    with pytest.raises(IKError,match='strict continuity'):
        reset_observation_arm(hub)
    assert not any(call[0]=='reset' for call in calls)
    assert hub.fatal
