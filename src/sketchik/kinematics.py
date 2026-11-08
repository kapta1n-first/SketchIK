"""Robot-only numerical IK. The model contains no table, task objects or fixtures."""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp


def vector(value, size, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must have {size} finite numbers")
    return result


def quaternion(value):
    q = vector(value, 4, "quaternion_xyzw")
    if not np.isclose(np.linalg.norm(q), 1, atol=1e-3):
        raise ValueError("quaternion_xyzw must be unit length")
    return Rotation.from_quat(q).as_quat()


def rotation_from_directions(finger_direction, closing_axis):
    """Panda grip_site +Z points towards fingertips, +X along finger separation.

    Axis sign does not change symmetric grasp geometry, but does change full pose.
    Directions are in the world frame; parallel/near-parallel input is rejected.
    """
    z = vector(finger_direction, 3, "finger_direction")
    x = vector(closing_axis, 3, "closing_axis")
    if np.linalg.norm(z) < 1e-8 or np.linalg.norm(x) < 1e-8:
        raise ValueError("Directions must be nonzero")
    z = z / np.linalg.norm(z)
    x = x / np.linalg.norm(x)
    x = x - z * np.dot(z, x)
    if np.linalg.norm(x) < 0.1:
        raise ValueError("Closing axis must not be parallel to finger direction")
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return Rotation.from_matrix(np.column_stack([x, y, z])).as_quat()


@dataclass
class Solution:
    q: np.ndarray
    position_error_m: float
    rotation_error_rad: float
    elapsed_s: float
    evaluations: int


class IKError(RuntimeError):
    pass


class RobotIK:
    def __init__(self, robot_xml, joint_names, site_name):
        self.model = mujoco.MjModel.from_xml_string(robot_xml)
        self.data = mujoco.MjData(self.model)
        joints = [self.model.joint(name).id for name in joint_names]
        if len(joints) != 7 or any(self.model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE for j in joints):
            raise ValueError("This version supports the seven-hinge Panda arm")
        self.q_index = self.model.jnt_qposadr[joints]
        self.limits = self.model.jnt_range[joints].copy()
        self.site_id = self.model.site(site_name).id
        self.lower = self.limits[:, 0] + 1e-5
        self.upper = self.limits[:, 1] - 1e-5

    def fk(self, q):
        q = vector(q, 7, "q")
        self.data.qpos[self.q_index] = q
        mujoco.mj_fwdPosition(self.model, self.data)
        return (self.data.site_xpos[self.site_id].copy(),
                self.data.site_xmat[self.site_id].reshape(3, 3).copy())

    def solve(self, position, quat_xyzw, seed):
        target = vector(position, 3, "position")
        target_r = Rotation.from_quat(quaternion(quat_xyzw))
        seed = vector(seed, 7, "seed")
        if np.any(seed < self.limits[:, 0]) or np.any(seed > self.limits[:, 1]):
            raise IKError("Measured seed is outside joint limits")
        start = time.perf_counter()
        # Weak regularization resolves the redundant seventh joint near the seed.
        # Full SO(3) error is used; no Euler component is silently discarded.
        def residual(q):
            p, r = self.fk(q)
            return np.r_[p - target, 0.15 * (target_r * Rotation.from_matrix(r).inv()).as_rotvec(),
                         1e-5 * (q - seed)]
        candidates = [seed, seed + np.array([0, .12, 0, -.12, 0, .12, 0]),
                      seed + np.array([.12, -.12, .12, 0, -.12, 0, 0])]
        total = 0
        best_error = None
        for candidate in candidates:
            result = least_squares(residual, np.clip(candidate, self.lower, self.upper),
                                   bounds=(self.lower, self.upper), max_nfev=120,
                                   ftol=1e-9, xtol=1e-9, gtol=1e-9)
            total += result.nfev
            p, r = self.fk(result.x)
            pe = float(np.linalg.norm(p - target))
            re = float((target_r * Rotation.from_matrix(r).inv()).magnitude())
            best_error = (pe, re) if best_error is None or pe < best_error[0] else best_error
            if pe < 0.0005 and re < math.radians(0.5):
                return Solution(result.x.copy(), pe, re, time.perf_counter() - start, total)
        raise IKError(f"No IK solution within 0.5mm / 0.5deg: best errors={best_error}")

    def cartesian_path(self, start_q, position, quat_xyzw, spacing_m=.02, angle_step_rad=.14):
        """Straight Cartesian segments, solved before execution; no obstacle search."""
        q = vector(start_q, 7, "start_q").copy()
        p0, r0 = self.fk(q)
        p1 = vector(position, 3, "position")
        q1 = quaternion(quat_xyzw)
        rotations = Rotation.from_quat([Rotation.from_matrix(r0).as_quat(), q1])
        angle = (rotations[1] * rotations[0].inv()).magnitude()
        count = max(1, math.ceil(np.linalg.norm(p1 - p0) / spacing_m), math.ceil(angle / angle_step_rad))
        if count > 150:
            raise IKError("Requested segment exceeds first-version planning bound")
        slerp = Slerp([0., 1.], rotations)
        solutions = []
        for i in range(1, count + 1):
            fraction = i / count
            sol = self.solve(p0 + fraction * (p1 - p0), slerp(fraction).as_quat(), q)
            if np.max(np.abs(sol.q - q)) > .4:
                raise IKError("Discontinuous IK branch; no trajectory dispatched")
            solutions.append(sol)
            q = sol.q
        return solutions


def joint_trajectory(start_q, targets, dt=.05, max_velocity=.6, max_acceleration=1.2):
    """Rest-to-rest quintic joint segments. Bounds concern commands, not physics."""
    if dt <= 0 or max_velocity <= 0 or max_acceleration <= 0:
        raise ValueError("Trajectory bounds must be positive")
    q = vector(start_q, 7, "start_q")
    output = []
    for target in targets:
        target = vector(target, 7, "target")
        distance = float(np.max(np.abs(target - q)))
        duration = max(2 * dt, 1.875 * distance / max_velocity,
                       math.sqrt(5.774 * distance / max_acceleration))
        count = math.ceil(duration / dt)
        for i in range(1, count + 1):
            t = i / count
            blend = 10*t**3 - 15*t**4 + 6*t**5
            output.append(q + blend * (target - q))
        q = target
    return np.array(output)
