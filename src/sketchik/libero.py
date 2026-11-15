"""LIBERO adapter. Object GT is neither returned nor used by motion planning."""
from __future__ import annotations

import importlib
import random
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


class LiberoRobot:
    cameras = ("agentview", "robot0_eye_in_hand")

    def __init__(self, variant="pro", suite="libero_10", task=0, seed=0, init_state=0,
                 size=1024, max_steps=2000, *, completion_diagnostic=False):
        if type(completion_diagnostic) is not bool:
            raise ValueError('completion_diagnostic must be boolean')
        import robosuite
        from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
        from robosuite.controllers.parts.controller_factory import load_part_controller_config
        prefix = "liberopro.liberopro" if variant == "pro" else "libero.libero"
        module = importlib.import_module(prefix)
        assets = Path(module.__file__).parent / "assets"
        if not assets.is_dir():
            raise FileNotFoundError(f"{variant} simulator assets missing: {assets}; "
                                    "no cross-benchmark assets substituted. Use --variant pro on this server.")
        benchmark = importlib.import_module(prefix + ".benchmark")
        envs = importlib.import_module(prefix + ".envs")
        bddl = importlib.import_module(prefix + ".envs.bddl_utils")
        random.seed(seed)
        np.random.seed(seed)
        tasks = benchmark.get_benchmark(suite)(task_order_index=0)
        if not 0 <= task < tasks.n_tasks:
            raise ValueError("task outside suite")
        states = tasks.get_task_init_states(task)
        if not 0 <= init_state < len(states):
            raise ValueError("init_state is a separate explicit index; outside available states")
        task_info = tasks.get_task(task)
        bddl_path = tasks.get_task_bddl_file_path(task)
        config = robosuite.load_composite_controller_config(controller=None, robot="Panda")
        arm = load_part_controller_config(default_controller="JOINT_POSITION")
        arm.update(input_type="absolute", kp=150, gripper={"type": "GRIP"})
        config["body_parts"] = {"right": arm}
        task_class = envs.TASK_MAPPING[bddl.get_problem_info(bddl_path)["problem_name"]]
        # Installed LIBERO-Pro 0.2 has an uninitialised local `sensors` when
        # use_object_obs=False. Bypass only its object-observable builder, locally;
        # never enable hidden object sensors or patch the installed dependency.
        class RobotObservationTask(task_class):
            def _setup_observables(self):
                return ManipulationEnv._setup_observables(self)
        # LIBERO validates the problem against the concrete class name.
        RobotObservationTask.__name__ = task_class.__name__
        self.env = RobotObservationTask(
            bddl_path, robots=["Panda"], controller_configs=config,
            initialization_noise=None, use_object_obs=False, use_camera_obs=False,
            has_renderer=False, has_offscreen_renderer=True, render_gpu_device_id=-1,
            camera_names=list(self.cameras), camera_heights=size, camera_widths=size,
            camera_depths=True, control_freq=20, horizon=max_steps,
            ignore_done=False, hard_reset=True, seed=seed)
        self.size, self.max_steps, self.steps = size, max_steps, 0
        self.done = False
        self.completion_diagnostic = completion_diagnostic
        self.first_official_success_step = None
        self.gripper_command = -1.0
        self.on_step = None
        self.env.reset()
        # This simulator state is used exclusively for the benchmark reset.
        self.env.sim.set_state_from_flattened(states[init_state])
        self.env.sim.forward()
        self.env._post_process()
        self.env._update_observables(force=True)
        self.robot = self.env.robots[0]
        self.joint_names = list(self.robot.robot_model.joints)
        self.q_index = [self.env.sim.model.get_joint_qpos_addr(j) for j in self.joint_names]
        self.site = self.robot.gripper["right"].important_sites["grip_site"]
        self.site_id = self.env.sim.model.site_name2id(self.site)
        controller = self.robot.part_controllers["right"]
        if controller.name != "JOINT_POSITION" or controller.input_type != "absolute" or self.env.action_dim != 8:
            raise RuntimeError("Expected absolute 7-joint position controller plus gripper; refusing OSC fallback")
        self.metadata = {"variant": variant, "suite": suite, "task": task,
                         "task_language": task_info.language, "seed": seed, "init_state_id": init_state,
                         "bddl_file": bddl_path, "controller": controller.name,
                         "input_type": controller.input_type, "kp": arm["kp"],
                         "control_hz": 20, "action_dim": self.env.action_dim,
                         "object_observations": False, "joint_names": self.joint_names,
                         "eef_site": self.site, "max_steps": max_steps, "image_size": size,
                         "completion_diagnostic": completion_diagnostic,
                         "termination_policy": ("physical_budget_or_backend_termination; official success is evaluation-only"
                             if completion_diagnostic else "official benchmark done or physical budget")}
        from importlib.metadata import distribution
        packages = {}
        for package in ("mujoco", "robosuite", "numpy", "scipy", "torch",
                        "rpent-liberopro" if variant == "pro" else "rpent-libero"):
            dist = distribution(package)
            packages[package] = {"version": dist.version, "origin": dist.read_text("direct_url.json")}
        self.metadata["dependencies"] = packages

    def state(self) -> dict:
        sim = self.env.sim
        fingers = self.robot.gripper["right"].joints
        finger_q = [float(sim.data.qpos[sim.model.get_joint_qpos_addr(j)]) for j in fingers]
        return {"step": self.steps, "q": sim.data.qpos[self.q_index].copy().tolist(),
                "position": sim.data.site_xpos[self.site_id].copy().tolist(),
                "quaternion_xyzw": Rotation.from_matrix(sim.data.site_xmat[self.site_id].reshape(3, 3)).as_quat().tolist(),
                "finger_joint_positions": finger_q, "finger_joint_names": list(fingers),
                "gripper_command": self.gripper_command,
                "official_success": bool(self.env._check_success()), "done": self.done}

    def step(self, joints, gripper=None):
        q = np.asarray(joints, dtype=float)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError("Expected seven finite absolute joint angles")
        ids = self.robot.arm_joint_indexes
        limits = self.env.sim.model.jnt_range[ids]
        if np.any(q < limits[:, 0]) or np.any(q > limits[:, 1]):
            raise ValueError("Joint limit violation; no action sent")
        if self.done or self.steps >= self.max_steps:
            raise RuntimeError("Episode is over; no further action dispatched")
        command = self.gripper_command if gripper is None else float(gripper)
        if command not in (-1.0, 1.0):
            raise ValueError("Gripper command must be -1=open or +1=close")
        start = time.perf_counter()
        _, _, done, _ = self.env.step(np.r_[q, command])
        self.steps += 1
        # LIBERO replaces the step return's done with _check_success(); the
        # underlying env.done still tracks the physical horizon. In this
        # explicit diagnostic, never use task GT to select the next action.
        self.done = (bool(self.steps >= self.max_steps or getattr(self.env, 'done', False))
                     if getattr(self, 'completion_diagnostic', False) else bool(done))
        self.gripper_command = command
        # Integration advances qpos beyond the last cached site/camera transforms.
        # Refresh without advancing time before recording state or RGB-D.
        self.env.sim.forward()
        # Verify the real controller received absolute angles without scaling/clipping.
        received = self.robot.part_controllers["right"].goal_qpos
        if not np.allclose(received, q, atol=1e-12, rtol=0):
            self.done = True
            raise RuntimeError("Backend changed absolute joint target; execution stopped")
        state = self.state()
        if state.get('official_success') and getattr(self, 'first_official_success_step', None) is None:
            self.first_official_success_step = self.steps
        state.update(joint_target=q.tolist(), step_wall_s=time.perf_counter() - start)
        if self.on_step is not None:
            self.on_step(state)
        return state

    def camera_rgb(self, name):
        """Native video frame without unused depth conversion or calibration."""
        if name not in self.cameras:
            raise ValueError(f"Unknown calibrated camera {name}")
        rgb = self.env.sim.render(camera_name=name, width=self.size, height=self.size, depth=False)
        return np.ascontiguousarray(rgb[::-1])

    def camera(self, name):
        if name not in self.cameras:
            raise ValueError(f"Unknown calibrated camera {name}")
        from robosuite.utils.camera_utils import (
            get_camera_intrinsic_matrix, get_camera_extrinsic_matrix, get_real_depth_map)
        sim = self.env.sim
        rgb, raw_depth = sim.render(camera_name=name, width=self.size, height=self.size, depth=True)
        return {"rgb": np.ascontiguousarray(rgb[::-1]),
                "depth_m": np.ascontiguousarray(get_real_depth_map(sim, raw_depth)[::-1]),
                "K": get_camera_intrinsic_matrix(sim, name, self.size, self.size),
                "world_from_camera": get_camera_extrinsic_matrix(sim, name),
                "camera": name, "step": self.steps,
                "convention": "top-left pixel origin; camera x right, y down, z forward; depth metres"}

    def close(self):
        self.env.close()


def back_project(frame: dict, pixel_xy) -> np.ndarray:
    """One observed pixel -> world XYZ. No extrapolation or hidden geometry."""
    xy = np.asarray(pixel_xy, dtype=float)
    if xy.shape != (2,) or not np.isfinite(xy).all():
        raise ValueError("Expected finite pixel [x,y]")
    x, y = np.rint(xy).astype(int)
    depth = frame["depth_m"]
    if not (0 <= x < depth.shape[1] and 0 <= y < depth.shape[0]):
        raise ValueError("Pixel outside image")
    z = depth[y, x]
    if not np.isfinite(z) or z <= 0:
        raise ValueError("Invalid depth")
    camera_point = z * np.linalg.solve(frame["K"], [x, y, 1])
    return (frame["world_from_camera"] @ np.r_[camera_point, 1])[:3]
