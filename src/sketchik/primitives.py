"""A few bounded analytic primitives; no high-level agent or task recipes."""
from __future__ import annotations

from copy import deepcopy
from collections import deque
import math
from numbers import Real
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .kinematics import RobotIK, joint_trajectory, quaternion, rotation_from_directions, vector


POSITION_TOLERANCE_M = .005
ROTATION_TOLERANCE_RAD = math.radians(3)
JOINT_TOLERANCE_RAD = .02


def _position_tolerance(value):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not math.isfinite(value) or not 0 < value <= .02):
        raise ValueError('position_tolerance_m must be finite and within (0, .02]')
    return float(value)


def _rotation_vector(value, name):
    try:
        values=list(value)
    except TypeError as exc:
        raise ValueError(f"{name} must have 3 finite numbers") from exc
    if len(values)!=3 or any(isinstance(v,(bool,np.bool_)) or not isinstance(v,Real) for v in values):
        raise ValueError(f"{name} must have 3 finite numbers")
    try:
        return vector(values,3,name)
    except (TypeError,ValueError,OverflowError) as exc:
        raise ValueError(f"{name} must have 3 finite numbers") from exc


def _rotation_angle(value, name):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,Real):
        raise ValueError(f"{name} must be a finite number in radians")
    try:
        angle=float(value)
    except (ValueError,OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number in radians") from exc
    if not math.isfinite(angle):
        raise ValueError(f"{name} must be a finite number in radians")
    return angle


def rotation_target_pose(state, axis, angle_rad, frame='world', pivot_world_m=None):
    """Full EEF target from one real entry pose; no motion or object estimate.

    Tool axes are resolved using the entry orientation. An omitted pivot keeps
    the entry EEF position; an explicit pivot rotates its position as well.
    """
    if frame not in ('world','tool'):
        raise ValueError("frame must be world or tool")
    angle=_rotation_angle(angle_rad,'angle_rad')
    direction=_rotation_vector(axis,'axis')
    magnitude=float(np.max(np.abs(direction)))
    if magnitude==0:
        raise ValueError("axis must be nonzero")
    # Scale first so finite very large or small non-unit axes normalize safely.
    direction=direction/magnitude
    direction=direction/np.linalg.norm(direction)
    position=vector(state['position'],3,'position')
    rotation=Rotation.from_quat(quaternion(state['quaternion_xyzw'])).as_matrix()
    world_axis=rotation@direction if frame=='tool' else direction
    delta=Rotation.from_rotvec(world_axis*math.remainder(angle,2*math.pi)).as_matrix()
    target_position=position.copy()
    if pivot_world_m is not None:
        pivot=_rotation_vector(pivot_world_m,'pivot_world_m')
        target_position=pivot+delta@(position-pivot)
    return {'position':target_position.tolist(),
        'quaternion_xyzw':Rotation.from_matrix(delta@rotation).as_quat().tolist()}


def simple_rotation_target_pose(state, primitive, **angles):
    """Left-multiply the full entry orientation by a relative world X/Y/Z turn.

    Roll/pitch/yaw name the fixed world X/Y/Z axes, respectively. The sole
    angle_rad is a signed relative angle; entry XYZ stays unchanged. No Euler
    decomposition, axis projection, or OSC yaw compensation is applied.
    """
    axes={'rotate_roll':[1.,0.,0.], 'rotate_pitch':[0.,1.,0.], 'rotate_yaw':[0.,0.,1.]}
    if primitive not in axes:
        raise ValueError('primitive must be rotate_roll, rotate_pitch or rotate_yaw')
    if set(angles)!={'angle_rad'}:
        raise ValueError(f'{primitive} requires only angle_rad')
    return rotation_target_pose(state,axes[primitive],angles['angle_rad'])


class MotionExecutor:
    def __init__(self, robot, emit=None, *, settle_gripper=False):
        self.robot = robot
        self.settle_gripper = settle_gripper
        self.emit = emit or (lambda record: None)
        self.ik = RobotIK(robot.robot.robot_model.get_xml(), robot.joint_names, robot.site)
        state = robot.state()
        p, r = self.ik.fk(state["q"])
        re = (Rotation.from_matrix(r) * Rotation.from_quat(state["quaternion_xyzw"]).inv()).magnitude()
        if np.linalg.norm(p - state["position"]) > 1e-6 or re > 1e-6:
            raise RuntimeError("Robot-only FK does not match the real simulator tool frame")
        self.initial_state = {'step':state['step'],'q':np.asarray(state['q'],float).tolist(),
            'position':np.asarray(p,float).tolist(),
            'quaternion_xyzw':Rotation.from_matrix(r).as_quat().tolist()}
        self.emit({"event": "robot_fk_verified", "position_error_m": float(np.linalg.norm(p-state["position"])),
                   "rotation_error_rad": float(re), "robot_only_nq": self.ik.model.nq})

    def gripper_control_context(self):
        """Describe the actual command policy, without a new control interface."""
        return {'commands':{'open':-1,'close':1},'target_gap_control_available':False,
            'last_command_continues_during_motion':True,
            'steps_role':'minimum_dwell_then_finger_settling' if self.settle_gripper else 'command_duration',
            'finger_settling':({'minimum_steps':'max(requested_steps,12)',
                'maximum_steps':'max(requested_steps,60), limited by remaining episode budget',
                'stable_consecutive_steps':3,'max_finger_joint_position_change':.0001}
                if self.settle_gripper else None),
            'meaning':'Binary open/close, not a target jaw width. Fewer steps do not request a narrower gap. '
                'Settling or a nonzero gap does not certify holding.'}

    def reset(self, *, settle_steps=80, checked_waypoints=None):
        """Return to the saved initial full EEF pose, retaining the gripper.

        This does not reset the environment or require the original IK branch.
        A caller's checked route is executed unchanged after its full endpoint
        is verified; this method itself performs no collision certification.
        """
        if isinstance(settle_steps,bool) or not isinstance(settle_steps,int) or not 0<=settle_steps<=200:
            raise ValueError('settle_steps must be 0..200')
        started=time.perf_counter()
        home=deepcopy(self.initial_state)
        pose={key:home[key] for key in ('position','quaternion_xyzw')}
        route=None
        if checked_waypoints is not None:
            if not isinstance(checked_waypoints,list) or not 1<=len(checked_waypoints)<=30:
                raise ValueError('checked_waypoints requires 1..30 waypoints ending at the initial full EEF pose')
            route=deepcopy(checked_waypoints)
            last=route[-1]
            if not isinstance(last,dict) or 'position' not in last or 'quaternion_xyzw' not in last:
                raise ValueError('Reset route must end at the absolute initial position and full quaternion')
            end_position=vector(last['position'],3,'reset route endpoint')
            end_rotation=Rotation.from_quat(quaternion(last['quaternion_xyzw']))
            home_rotation=Rotation.from_quat(home['quaternion_xyzw'])
            if (np.linalg.norm(end_position-home['position'])>1e-6 or
                    (end_rotation*home_rotation.inv()).magnitude()>1e-6):
                raise ValueError('Reset route endpoint differs from the initial full EEF pose')
        entry=self.robot.state()
        # Confirm the saved full pose with real IK using its original joint seed;
        # continuous execution still starts from the current real joint state.
        reference=self.ik.solve(home['position'],home['quaternion_xyzw'],home['q'])
        self.emit({'event':'reset_started','source_step':entry['step'],
            'initial_source_step':home['step'],'target_pose':pose,
            'home_ik_seed':'initial_q','checked_waypoints_supplied':route is not None,
            'collision_checked_by_reset':False})
        result=(self.move(**pose,settle_steps=settle_steps) if route is None else
                self.move_waypoints(route,settle_steps=settle_steps))
        actual=self.robot.state()
        joint_error=np.asarray(actual['q'])-np.asarray(home['q'])
        maximum=float(np.max(np.abs(joint_error)))
        details={'scope':'initial full EEF pose; original joint branch is not required',
            'initial_source_step':home['step'],'target_pose':pose,'initial_q':home['q'],
            'home_ik_seed':'initial_q','home_ik_reference_q':np.asarray(reference.q).tolist(),
            'home_ik_wall_s':float(reference.elapsed_s),
            'joint_error_from_initial_rad':joint_error.tolist(),'max_joint_error_from_initial_rad':maximum,
            'joint_match_tolerance_rad':.02,'initial_joints_match':maximum<=.02,
            'gripper_command_preserved':actual['gripper_command']==entry['gripper_command'],
            'checked_waypoints_supplied':route is not None,'collision_checked_by_reset':False}
        result={**result,'reset':details,'reset_wall_s':time.perf_counter()-started}
        self.emit({'event':'reset_finished','converged':result['converged'],
            'source_step':actual['step'],'reset_wall_s':result['reset_wall_s'],**details})
        return result

    def move_relative(self, delta_m, quaternion_xyzw=None, *, settle_steps=80):
        # Resolve once at entry; it is never re-added during control or settling.
        position = np.asarray(self.robot.state()["position"]) + vector(delta_m, 3, "delta_m")
        return self.move(position, quaternion_xyzw=quaternion_xyzw, settle_steps=settle_steps)

    def move(self, position, *, quaternion_xyzw=None, finger_direction=None, closing_axis=None,
             settle_steps=80, position_tolerance_m=POSITION_TOLERANCE_M):
        """Absolute world XYZ; preserve entry rotation unless a full one is given."""
        if quaternion_xyzw is not None and (finger_direction is not None or closing_axis is not None):
            raise ValueError("Choose quaternion OR two directions")
        if finger_direction is not None or closing_axis is not None:
            if finger_direction is None or closing_axis is None:
                raise ValueError("Both finger_direction and closing_axis are required")
            quaternion_xyzw = rotation_from_directions(finger_direction, closing_axis)
        return self.move_waypoints([{"position": position, **(
            {"quaternion_xyzw": quaternion_xyzw} if quaternion_xyzw is not None else {})}],
            settle_steps=settle_steps, position_tolerance_m=position_tolerance_m)

    def rotate_roll(self, *, angle_rad, settle_steps=80):
        """Turn full EEF orientation about world X, keeping entry XYZ/gripper."""
        pose=simple_rotation_target_pose(self.robot.state(),'rotate_roll',angle_rad=angle_rad)
        return self.move(**pose,settle_steps=settle_steps)

    def rotate_pitch(self, *, angle_rad, settle_steps=80):
        """Turn full EEF orientation about world Y, keeping entry XYZ/gripper."""
        pose=simple_rotation_target_pose(self.robot.state(),'rotate_pitch',angle_rad=angle_rad)
        return self.move(**pose,settle_steps=settle_steps)

    def rotate_yaw(self, *, angle_rad, settle_steps=80):
        """Turn full EEF orientation about world Z, keeping entry XYZ/gripper."""
        pose=simple_rotation_target_pose(self.robot.state(),'rotate_yaw',angle_rad=angle_rad)
        return self.move(**pose,settle_steps=settle_steps)

    def move_waypoints(self, waypoints, *, settle_steps=80,
                       position_tolerance_m=POSITION_TOLERANCE_M):
        if getattr(self.robot, 'done', False):
            return {'event':'motion_finished','episode_ended':True,'converged':False,
                'actions':0,'state':self.robot.state(),'termination':'Target not attempted: episode already ended'}
        prepared = self.prepare_motion(waypoints, settle_steps=settle_steps,
            position_tolerance_m=position_tolerance_m)
        return self.execute_prepared_motion(prepared)

    def prepare_motion(self, waypoints, *, settle_steps=80,
                       position_tolerance_m=POSITION_TOLERANCE_M):
        position_tolerance_m = _position_tolerance(position_tolerance_m)
        source = self.robot.state()
        try:
            return self._prepare_motion(waypoints, settle_steps=settle_steps,
                position_tolerance_m=position_tolerance_m)
        finally:
            if hasattr(getattr(self, 'ik', None), 'fk'):
                self.ik.fk(source['q'])

    def _prepare_motion(self, waypoints, *, settle_steps=80,
                        position_tolerance_m=POSITION_TOLERANCE_M):
        """Plan all waypoints first, then track joint targets with physics enabled.

        A waypoint uses absolute `position` or `relative_m` from the previous
        planned waypoint, and optionally a world quaternion. No implicit lift.
        """
        if not isinstance(waypoints, list) or not 1 <= len(waypoints) <= 30:
            raise ValueError("Expected 1..30 waypoints")
        if not isinstance(settle_steps, int) or not 0 <= settle_steps <= 200:
            raise ValueError("settle_steps must be 0..200")
        position_tolerance_m = _position_tolerance(position_tolerance_m)
        start = time.perf_counter()
        state = self.robot.state()
        if getattr(self.robot,'done',False):
            raise ValueError('Episode ended before numerical compilation; no action dispatched')
        q = np.array(state["q"])
        pos = np.array(state["position"])
        quat = np.array(state["quaternion_xyzw"])
        solutions, ranges = [], []
        # No action is sent if any later waypoint has invalid geometry or IK.
        for waypoint in waypoints:
            if not isinstance(waypoint, dict) or set(waypoint) - {"position", "relative_m", "quaternion_xyzw"}:
                raise ValueError("Unknown waypoint fields")
            if ("position" in waypoint) == ("relative_m" in waypoint):
                raise ValueError("Waypoint requires exactly one position / relative_m")
            pos = (vector(waypoint["position"], 3, "position") if "position" in waypoint
                   else pos + vector(waypoint["relative_m"], 3, "relative_m"))
            quat = quaternion(waypoint.get("quaternion_xyzw", quat))
            path = self.ik.cartesian_path(q, pos, quat)
            samples = joint_trajectory(q, [solution.q for solution in path])
            ranges.append(samples)
            solutions.extend(path)
            q = path[-1].q
        trajectory = np.concatenate(ranges)
        required = len(trajectory) + settle_steps
        if required > self.robot.max_steps - self.robot.steps:
            raise RuntimeError(f"Insufficient episode budget: need <= {required} actions")
        planned = {"event": "motion_planned", "source_step": state["step"],
                   "target_position": pos.tolist(), "target_quaternion_xyzw": quat.tolist(),
                   "waypoints": waypoints, "trajectory_steps": len(trajectory),
                   "planning_wall_s": time.perf_counter()-start,
                   "ik_wall_s": sum(s.elapsed_s for s in solutions),
                   "ik_solves": len(solutions), "joint_target_final": q.tolist(),
                   "ik_endpoint_position_error_m": solutions[-1].position_error_m,
                   "ik_endpoint_rotation_error_rad": solutions[-1].rotation_error_rad,
                   "gripper_command": state["gripper_command"], "collision_checked": False,
                   "tracking_tolerances": self._tracking_tolerances(position_tolerance_m)}
        from .motion_record import seal_motion, source_state
        return seal_motion({'version': 1, 'executor_identity': id(self),
            'source_state': source_state(state), 'planned': planned,
            'trajectory': trajectory.tolist(), 'settle_steps': settle_steps,
            'position_tolerance_m': position_tolerance_m})

    def execute_prepared_motion(self, prepared):
        from .motion_record import validate_motion
        validate_motion(prepared, self.robot.state(), id(self))
        if getattr(self.robot, 'done', False):
            raise ValueError('Episode ended before prepared motion; no action dispatched')
        planned = prepared['planned']
        trajectory = np.asarray(prepared['trajectory'], float)
        if len(trajectory) + prepared['settle_steps'] > self.robot.max_steps - self.robot.steps:
            raise RuntimeError('Prepared motion no longer fits physical budget; no action dispatched')
        self.emit({**planned, 'compiled_motion_id': prepared['artifact_id']})
        result = self._track_joint_targets(trajectory, planned['target_position'],
            planned['target_quaternion_xyzw'], planned['joint_target_final'], prepared['settle_steps'],
            time.perf_counter()-planned['planning_wall_s'], planned['gripper_command'],
            prepared['position_tolerance_m'], compiled_motion_id=prepared['artifact_id'])
        return result

    @staticmethod
    def _tracking_tolerances(position_tolerance_m):
        return {'position_m': position_tolerance_m, 'orientation_deg': 3.,
                'max_joint_rad': JOINT_TOLERANCE_RAD}

    def _track_joint_targets(self, trajectory, pos, quat, q, settle_steps, start, gripper_command,
                             position_tolerance_m=POSITION_TOLERANCE_M, *, compiled_motion_id=None):
        position_tolerance_m = _position_tolerance(position_tolerance_m)
        execution_start = time.perf_counter()
        executed = 0
        recent = deque(maxlen=10)
        from .motion_record import MotionTrace
        tracking = MotionTrace(self.robot.state(), q, getattr(self.robot, 'joint_names', ()))
        def sample(target, phase):
            state = self.robot.state()
            tracking.sample(state, target, phase)
            pe = float(np.linalg.norm(np.asarray(state['position']) - pos))
            re = float((Rotation.from_quat(state['quaternion_xyzw']) * Rotation.from_quat(quat).inv()).magnitude())
            qe = float(np.max(np.abs(np.asarray(state['q']) - q)))
            recent.append((pe, re, qe))
            return state, pe, re, qe
        for target in trajectory:
            if getattr(self.robot,'done',False):break
            self.robot.step(target)
            executed += 1
            sample(target, 'trajectory')
        for _ in range(settle_steps):
            if getattr(self.robot,'done',False):break
            current = self.robot.state()
            pe = np.linalg.norm(np.array(current["position"]) - pos)
            re = (Rotation.from_quat(current["quaternion_xyzw"]) * Rotation.from_quat(quat).inv()).magnitude()
            if (pe <= position_tolerance_m and re <= ROTATION_TOLERANCE_RAD
                    and np.max(np.abs(np.array(current["q"])-q)) <= JOINT_TOLERANCE_RAD):
                break
            self.robot.step(q)
            executed += 1
            sample(q, 'settling')
        current = self.robot.state()
        pe = float(np.linalg.norm(np.array(current["position"]) - pos))
        re = float((Rotation.from_quat(current["quaternion_xyzw"]) * Rotation.from_quat(quat).inv()).magnitude())
        result = {"event": "motion_finished", "converged": bool(pe <= position_tolerance_m and re <= ROTATION_TOLERANCE_RAD
                  and np.max(np.abs(np.array(current["q"])-q)) <= JOINT_TOLERANCE_RAD),
                  "position_error_m": pe, "rotation_error_deg": math.degrees(re),
                  "actions": executed, "execution_wall_s": time.perf_counter()-execution_start,
                  "total_wall_s": time.perf_counter()-start, "state": current,
                  "tracking": tracking.summary(current),
                  "tracking_tolerances": self._tracking_tolerances(position_tolerance_m)}
        if compiled_motion_id is not None:
            result['compiled_motion_id'] = compiled_motion_id
        if not result['converged'] and not getattr(self.robot, 'done', False):
            progress = self._recent_progress(recent)
            result['motion_progress'] = progress
            result['continuation_goal'] = {'position':np.asarray(pos).tolist(),
                'quaternion_xyzw':np.asarray(quat).tolist(), 'joint_target':np.asarray(q).tolist(),
                'gripper_command':gripper_command, 'source_step':current['step'],
                'position_tolerance_m':position_tolerance_m}
            result['physical_cause'] = 'not_determined'
        if getattr(self.robot,'done',False):result['episode_ended']=True
        self.emit(result)
        return result

    @staticmethod
    def _recent_progress(samples):
        if len(samples) < 6:
            return {'recent_progress':False,'samples':len(samples),'reason':'Insufficient recent measured motion'}
        first = np.mean(list(samples)[:3], axis=0)
        last = np.mean(list(samples)[-3:], axis=0)
        delta = first - last
        scales = np.array([.005, math.radians(3), .02])
        progressing = bool(np.linalg.norm(first / scales) - np.linalg.norm(last / scales) > .02
            and (delta[0] > .0005 or delta[1] > math.radians(.1) or delta[2] > .001))
        return {'recent_progress':progressing,'samples':len(samples),
            'recent_position_progress_m':float(delta[0]), 'recent_rotation_progress_rad':float(delta[1]),
            'recent_joint_error_progress_rad':float(delta[2]),
            'meaning':'Measured error reduction only; contact cause and attachment remain unknown'}

    def continue_motion(self, failed_result, *, max_steps=40, stop_condition=None):
        """Continue one absolute goal/setpoint, never reapply a relative delta/angle."""
        if type(max_steps) is not int or not 1 <= max_steps <= 40:
            raise ValueError('Continuation max_steps must be 1..40')
        if stop_condition is not None and not callable(stop_condition):
            raise ValueError('Continuation stop_condition must be callable')
        goal = failed_result.get('continuation_goal') or {}
        state = self.robot.state()
        if (failed_result.get('converged') is not False
                or not failed_result.get('motion_progress', {}).get('recent_progress')
                or goal.get('source_step') != state['step']
                or goal.get('gripper_command') != state['gripper_command']
                or getattr(self.robot, 'done', False)):
            raise ValueError('Continuation needs current progressing motion with the preserved gripper')
        q = vector(goal['joint_target'], 7, 'joint_target')
        if np.any(q < self.ik.lower) or np.any(q > self.ik.upper):
            raise ValueError('Continuation target exceeds joint limits')
        pos = vector(goal['position'], 3, 'position')
        quat = quaternion(goal['quaternion_xyzw'])
        position_tolerance_m = _position_tolerance(goal.get('position_tolerance_m', POSITION_TOLERANCE_M))
        started = time.perf_counter(); recent = deque(maxlen=10); executed = 0
        from .motion_record import MotionTrace
        tracking = MotionTrace(state, q, getattr(self.robot, 'joint_names', ()))
        interruption = None
        limit = min(max_steps, self.robot.max_steps - self.robot.steps)
        while executed < limit and not getattr(self.robot, 'done', False):
            state = self.robot.state()
            if stop_condition is not None:
                interruption = stop_condition(state)
                if interruption is not None:
                    break
            pe = float(np.linalg.norm(np.asarray(state['position']) - pos))
            re = float((Rotation.from_quat(state['quaternion_xyzw']) * Rotation.from_quat(quat).inv()).magnitude())
            qe = float(np.max(np.abs(np.asarray(state['q']) - q)))
            if pe <= position_tolerance_m and re <= ROTATION_TOLERANCE_RAD and qe <= JOINT_TOLERANCE_RAD:break
            recent.append((pe, re, qe))
            if len(recent) == 10 and not self._recent_progress(recent)['recent_progress']:break
            self.robot.step(q); executed += 1
            tracking.sample(self.robot.state(), q, 'T1_continuation')
        state = self.robot.state()
        # The last allowed step must also be checked, even without another loop.
        if interruption is None and stop_condition is not None:
            interruption = stop_condition(state)
        pe = float(np.linalg.norm(np.asarray(state['position']) - pos))
        re = float((Rotation.from_quat(state['quaternion_xyzw']) * Rotation.from_quat(quat).inv()).magnitude())
        qe = float(np.max(np.abs(np.asarray(state['q']) - q)))
        result = {'event':'motion_continuation_finished','converged':bool(interruption is None and pe <= position_tolerance_m and re <= ROTATION_TOLERANCE_RAD and qe <= JOINT_TOLERANCE_RAD),
            'position_error_m':pe,'rotation_error_deg':math.degrees(re),'actions':executed,
            'execution_wall_s':time.perf_counter()-started,'state':state,'absolute_goal_preserved':True,
            'gripper_command_preserved':state['gripper_command']==goal['gripper_command'],
            'motion_progress':self._recent_progress(recent),'physical_cause':'not_determined',
            'tracking': tracking.summary(state),
            'tracking_tolerances':self._tracking_tolerances(position_tolerance_m)}
        if interruption is not None:
            result['continuation_interruption'] = deepcopy(interruption)
        if getattr(self.robot,'done',False):result['episode_ended']=True
        self.emit(result)
        return result

    def wait(self, steps=10):
        if not isinstance(steps, int) or steps < 1 or steps > 200:
            raise ValueError("steps must be 1..200")
        if steps > self.robot.max_steps - self.robot.steps:
            raise RuntimeError("Insufficient episode budget")
        q = self.robot.state()["q"]
        for _ in range(steps):
            if getattr(self.robot,'done',False):break
            self.robot.step(q)
        state=self.robot.state()
        if getattr(self.robot,'done',False):state={**state,'episode_ended':True}
        return state

    def set_gripper(self, command, steps=15):
        if command not in ("open", "close"):
            raise ValueError("command must be open or close")
        if not isinstance(steps, int) or not 1 <= steps <= 200 or steps > self.robot.max_steps-self.robot.steps:
            raise ValueError("Invalid steps or insufficient budget")
        q = self.robot.state()["q"]
        minimum = max(12, steps) if self.settle_gripper else steps
        remaining = self.robot.max_steps-self.robot.steps
        if minimum > remaining: raise ValueError('Insufficient gripper settling budget')
        maximum = min(remaining, max(minimum, 60)) if self.settle_gripper else steps
        previous = np.asarray(self.robot.state()['finger_joint_positions'])
        steady, executed = 0, 0
        for _ in range(maximum):
            if getattr(self.robot,'done',False):break
            self.robot.step(q, -1 if command == "open" else 1)
            executed += 1
            current = np.asarray(self.robot.state()['finger_joint_positions'])
            steady = steady+1 if np.max(abs(current-previous)) < .0001 else 0
            previous = current
            if executed >= minimum and (not self.settle_gripper or steady >= 3): break
        state = self.robot.state()
        if getattr(self.robot,'done',False):state={**state,'episode_ended':True}
        self.emit({"event": "gripper_finished", "command": command, "actions": executed,
                   'requested_steps': steps, 'finger_state_settled': steady >= 3 if self.settle_gripper else None, "state": state,
                   "note": "Command sent; no object-holding or placement success inferred"})
        return state

    def release(self, steps=15):
        """Open at current pose only; does not choose or certify a placement."""
        return self.set_gripper("open", steps)
