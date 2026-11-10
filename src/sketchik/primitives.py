"""A few bounded analytic primitives; no high-level agent or task recipes."""
from __future__ import annotations

import math
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .kinematics import RobotIK, joint_trajectory, quaternion, rotation_from_directions, vector


class MotionExecutor:
    def __init__(self, robot, emit=None):
        self.robot = robot
        self.emit = emit or (lambda record: None)
        self.ik = RobotIK(robot.robot.robot_model.get_xml(), robot.joint_names, robot.site)
        state = robot.state()
        p, r = self.ik.fk(state["q"])
        re = (Rotation.from_matrix(r) * Rotation.from_quat(state["quaternion_xyzw"]).inv()).magnitude()
        if np.linalg.norm(p - state["position"]) > 1e-6 or re > 1e-6:
            raise RuntimeError("Robot-only FK does not match the real simulator tool frame")
        self.emit({"event": "robot_fk_verified", "position_error_m": float(np.linalg.norm(p-state["position"])),
                   "rotation_error_rad": float(re), "robot_only_nq": self.ik.model.nq})

    def move_to(self, position, quaternion_xyzw=None, *, settle_steps=80):
        """Absolute world EEF target. Missing rotation preserves current rotation."""
        return self.move_waypoints([{"position": position, **(
            {"quaternion_xyzw": quaternion_xyzw} if quaternion_xyzw is not None else {})}],
            settle_steps=settle_steps)

    def move_relative(self, delta_m, quaternion_xyzw=None, *, settle_steps=80):
        # Resolve once at entry; it is never re-added during control or settling.
        position = np.asarray(self.robot.state()["position"]) + vector(delta_m, 3, "delta_m")
        return self.move_to(position, quaternion_xyzw, settle_steps=settle_steps)

    def move_pose(self, position, *, quaternion_xyzw=None, finger_direction=None, closing_axis=None,
                  settle_steps=80):
        if quaternion_xyzw is not None and (finger_direction is not None or closing_axis is not None):
            raise ValueError("Choose quaternion OR two directions")
        if finger_direction is not None or closing_axis is not None:
            if finger_direction is None or closing_axis is None:
                raise ValueError("Both finger_direction and closing_axis are required")
            quaternion_xyzw = rotation_from_directions(finger_direction, closing_axis)
        if quaternion_xyzw is None:
            raise ValueError("move_pose requires a full orientation")
        return self.move_to(position, quaternion_xyzw, settle_steps=settle_steps)

    def move_waypoints(self, waypoints, *, settle_steps=80):
        """Plan all waypoints first, then track joint targets with physics enabled.

        A waypoint uses absolute `position` or `relative_m` from the previous
        planned waypoint, and optionally a world quaternion. No implicit lift.
        """
        if not isinstance(waypoints, list) or not 1 <= len(waypoints) <= 30:
            raise ValueError("Expected 1..30 waypoints")
        if not isinstance(settle_steps, int) or not 0 <= settle_steps <= 200:
            raise ValueError("settle_steps must be 0..200")
        start = time.perf_counter()
        state = self.robot.state()
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
                   "gripper_command": state["gripper_command"], "collision_checked": False}
        self.emit(planned)
        execution_start = time.perf_counter()
        executed = 0
        for target in trajectory:
            self.robot.step(target)
            executed += 1
        for _ in range(settle_steps):
            current = self.robot.state()
            pe = np.linalg.norm(np.array(current["position"]) - pos)
            re = (Rotation.from_quat(current["quaternion_xyzw"]) * Rotation.from_quat(quat).inv()).magnitude()
            if pe <= .005 and re <= math.radians(3) and np.max(np.abs(np.array(current["q"])-q)) <= .02:
                break
            self.robot.step(q)
            executed += 1
        current = self.robot.state()
        pe = float(np.linalg.norm(np.array(current["position"]) - pos))
        re = float((Rotation.from_quat(current["quaternion_xyzw"]) * Rotation.from_quat(quat).inv()).magnitude())
        result = {"event": "motion_finished", "converged": bool(pe <= .005 and re <= math.radians(3)
                  and np.max(np.abs(np.array(current["q"])-q)) <= .02),
                  "position_error_m": pe, "rotation_error_deg": math.degrees(re),
                  "actions": executed, "execution_wall_s": time.perf_counter()-execution_start,
                  "total_wall_s": time.perf_counter()-start, "state": current}
        self.emit(result)
        return result

    def wait(self, steps=10):
        if not isinstance(steps, int) or steps < 1 or steps > 200:
            raise ValueError("steps must be 1..200")
        if steps > self.robot.max_steps - self.robot.steps:
            raise RuntimeError("Insufficient episode budget")
        q = self.robot.state()["q"]
        for _ in range(steps):
            self.robot.step(q)
        return self.robot.state()

    def set_gripper(self, command, steps=15):
        if command not in ("open", "close"):
            raise ValueError("command must be open or close")
        if not isinstance(steps, int) or not 1 <= steps <= 200 or steps > self.robot.max_steps-self.robot.steps:
            raise ValueError("Invalid steps or insufficient budget")
        q = self.robot.state()["q"]
        for _ in range(steps):
            self.robot.step(q, -1 if command == "open" else 1)
        state = self.robot.state()
        self.emit({"event": "gripper_finished", "command": command, "actions": steps, "state": state,
                   "note": "Command sent; no object-holding or placement success inferred"})
        return state

    def release(self, steps=15):
        """Open at current pose only; does not choose or certify a placement."""
        return self.set_gripper("open", steps)
