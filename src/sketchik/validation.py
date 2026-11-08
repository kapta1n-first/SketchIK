"""Real simulator smoke and minimal frozen JSON plan runner."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import traceback

import imageio.v2 as imageio
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from .kinematics import IKError
from .primitives import MotionExecutor


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def run_validation(robot, run: Path, command, plan_file):
    start = time.perf_counter()
    writer = imageio.get_writer(str(run / "video.mp4"), fps=20, codec="libx264", quality=7)
    log = (run / "events.jsonl").open("w", buffering=1)
    records = []
    render_time = 0.0

    def emit(record):
        record = dict(record, wall_since_start_s=time.perf_counter()-start,
                      utc=datetime.now(timezone.utc).isoformat())
        log.write(json.dumps(record, default=json_default) + "\n")
        if record["event"] != "physics_step":
            records.append(record)

    def record_step(state):
        nonlocal render_time
        before = time.perf_counter()
        writer.append_data(robot.camera("agentview")["rgb"])
        elapsed = time.perf_counter()-before
        render_time += elapsed
        emit({"event": "physics_step", "render_encode_wall_s": elapsed, **state})

    def snapshot(name):
        folder = run / name
        folder.mkdir()
        (folder / "state.json").write_text(json.dumps(robot.state(), indent=2))
        for camera in robot.cameras:
            f = robot.camera(camera)
            Image.fromarray(f["rgb"]).save(folder / (camera+".png"))
            np.savez_compressed(folder / (camera+".npz"), depth_m=f["depth_m"], K=f["K"],
                                world_from_camera=f["world_from_camera"])

    robot.on_step = record_step
    ok = True
    error = None
    try:
        executor = MotionExecutor(robot, emit)
        xml = robot.robot.robot_model.get_xml()
        (run / "robot_only.xml").write_text(xml)
        emit({"event": "robot_model", "sha256": hashlib.sha256(xml.encode()).hexdigest(),
              "joint_limits_rad": executor.ik.limits, "no_environment_collision_planner": True})
        writer.append_data(robot.camera("agentview")["rgb"])
        if command == "smoke":
            # Only relative free-space control probes. These are not manipulation recipes.
            executor.wait(steps=10)
            snapshot("00_settled")
            home = robot.state()
            results = []
            results.append(executor.move_relative([0., 0., .04]))
            snapshot("01_up")
            results.append(executor.move_relative([.03, 0., 0.]))
            snapshot("02_side")
            current = robot.state()
            r = Rotation.from_rotvec([0., 0., np.deg2rad(15)]) * Rotation.from_quat(current["quaternion_xyzw"])
            results.append(executor.move_pose(current["position"], quaternion_xyzw=r.as_quat()))
            snapshot("03_rotate")
            results.append(executor.move_waypoints([
                {"relative_m": [-.03, .025, 0.], "quaternion_xyzw": home["quaternion_xyzw"]},
                {"position": home["position"]}]))
            snapshot("04_waypoints_return")
            executor.set_gripper("close", steps=15)
            closed = robot.state()
            executor.release(steps=15)
            opened = robot.state()
            snapshot("05_released")
            closed_width = float(abs(np.diff(closed["finger_joint_positions"])[0]))
            open_width = float(abs(np.diff(opened["finger_joint_positions"])[0]))
            emit({"event": "gripper_measured", "closed_joint_separation_m": closed_width,
                  "open_joint_separation_m": open_width,
                  "opening_changed": open_width > closed_width + .02})
            before = robot.steps
            try:
                executor.move_to([1.4, 0., .8])
                raise AssertionError("Impossible IK target was accepted")
            except IKError as exc:
                emit({"event": "unreachable_rejected", "error": str(exc),
                      "zero_actions": robot.steps == before})
            ok = all(x["converged"] for x in results) and open_width > closed_width+.02 and robot.steps == before
        else:
            if plan_file is None:
                raise ValueError("execute requires --plan")
            raw = plan_file.read_text()
            (run / "plan.json").write_text(raw)
            plan = json.loads(raw)
            if set(plan) != {"version", "steps"} or plan["version"] != 1 or not isinstance(plan["steps"], list):
                raise ValueError("Expected {version:1, steps:[{primitive, params}, ...]}")
            allowed = {"move_to", "move_relative", "move_pose", "move_waypoints", "set_gripper", "release", "wait"}
            # Validate method names and Python signatures for the whole plan before dispatch.
            import inspect
            for item in plan["steps"]:
                if set(item) != {"primitive", "params"} or item["primitive"] not in allowed:
                    raise ValueError("Unknown primitive / fields")
                inspect.signature(getattr(executor, item["primitive"])).bind(**item["params"])
            for i, item in enumerate(plan["steps"]):
                emit({"event": "primitive_started", "index": i, **item})
                result = getattr(executor, item["primitive"])(**item["params"])
                snapshot(f"step_{i:03d}")
                if result.get("converged") is False:
                    ok = False
                    emit({"event": "plan_stopped", "reason": "motion did not converge", "index": i})
                    break
    except Exception as exc:
        ok = False
        error = f"{type(exc).__name__}: {exc}"
        emit({"event": "exception", "error": error, "traceback": traceback.format_exc()})
    finally:
        robot.on_step = None
        writer.close()
        log.close()
    result = {"control_validation_passed": bool(ok), "error": error, "actions": robot.steps,
              "simulated_seconds": robot.steps/20, "wall_seconds": time.perf_counter()-start,
              "render_encode_seconds": render_time, "official_task_success": robot.state()["official_success"],
              "note": "Basic control validation only; not a grasp/place task success test",
              "records": records}
    (run / "result.json").write_text(json.dumps(result, default=json_default, indent=2))
    return result
