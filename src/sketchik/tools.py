"""Model-facing tools. No arbitrary Python, shell, object GT or memory access."""
from __future__ import annotations

import base64
from io import BytesIO
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
from jsonschema import Draft202012Validator

from .libero import back_project


def obj(properties, required=()):
    return {"type": "object", "properties": properties, "required": list(required), "additionalProperties": False}


V3 = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}
Q4 = {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4}
STEPS = {"type": "integer", "minimum": 1, "maximum": 200}
WAYPOINT = obj({"position": V3, "relative_m": V3, "quaternion_xyzw": Q4})
WAYPOINT["oneOf"] = [{"required": ["position"], "not": {"required": ["relative_m"]}},
                     {"required": ["relative_m"], "not": {"required": ["position"]}}]
PARAMS = {
    "move_to": obj({"position": V3, "quaternion_xyzw": Q4}, ["position"]),
    "move_relative": obj({"delta_m": V3, "quaternion_xyzw": Q4}, ["delta_m"]),
    "move_pose": obj({"position": V3, "quaternion_xyzw": Q4,
                      "finger_direction": V3, "closing_axis": V3}, ["position"]),
    "move_waypoints": obj({"waypoints": {"type": "array", "items": WAYPOINT, "minItems": 1, "maxItems": 30}}, ["waypoints"]),
    "set_gripper": obj({"command": {"enum": ["open", "close"]}, "steps": STEPS}, ["command"]),
    "release": obj({"steps": STEPS}),
    "wait": obj({"steps": STEPS}),
}
PARAMS["move_pose"]["oneOf"] = [
    {"required": ["quaternion_xyzw"], "not": {"anyOf": [{"required": ["finger_direction"]}, {"required": ["closing_axis"]}]}},
    {"required": ["finger_direction", "closing_axis"], "not": {"required": ["quaternion_xyzw"]}},
]
for key in ("move_to", "move_relative", "move_pose", "move_waypoints"):
    PARAMS[key]["properties"]["settle_steps"] = {"type": "integer", "minimum": 0, "maximum": 200}
DESCRIPTIONS = {
    "move_to": "Move EEF to world position (metres), optional xyzw orientation. Missing orientation preserves current orientation. IK + joint control, no collision planner.",
    "move_relative": "Move by signed world XYZ displacement from actual entry pose; preserve orientation unless specified. Displacement is resolved once.",
    "move_pose": "Move with full orientation: unit xyzw quaternion OR world finger_direction (+tool Z) and closing_axis (+tool X). No automatic grasp pose generation.",
    "move_waypoints": "Move through supplied EEF waypoints. position is absolute; relative_m is from preceding planned waypoint (first from actual entry). Can express caller-planned detours; does not search for obstacle-free paths.",
    "set_gripper": "Command open/close while holding current joints. Does not establish object identity or holding success.",
    "release": "Open gripper at current pose only. Does not compute a placement point or certify placement.",
    "wait": "Hold measured joint positions and previous gripper command while advancing 1..200 physics steps (20Hz). Not a wall-clock sleep or success detector.",
}


def tool_schemas():
    definitions = {
        "observe": ("Get current labelled agentview and wrist RGB, robot state, tool axes, remaining action budget and calibration; depth stays local for back_project.", obj({})),
        "get_state": ("Read current robot state without rendering images or advancing physics.", obj({})),
        "back_project": ("Convert a pixel from a saved observation to observed world XYZ using that exact camera's metric depth. Historical results are labelled stale, not silently refreshed.", obj({
            "observation_id": {"type": "string"}, "camera": {"enum": ["agentview", "robot0_eye_in_hand"]},
            "pixel_xy": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}},
            ["observation_id", "camera", "pixel_xy"])),
    }
    for name, params in PARAMS.items():
        definitions[name] = (DESCRIPTIONS[name], obj({"source_step": {"type": "integer", "minimum": 0},
                                                    "params": params}, ["source_step", "params"]))
    alternatives = [obj({"primitive": {"const": n}, "params": p}, ["primitive", "params"]) for n,p in PARAMS.items()]
    definitions["execute_plan"] = ("Execute one numeric analytic action plan. Multiple steps are allowed; stop at the first error/nonconvergence, retain completed prefix and return fresh observations. This is not object-level Grounding.", obj({
        "source_step": {"type": "integer", "minimum": 0},
        "steps": {"type": "array", "minItems": 1, "maxItems": 30, "items": {"oneOf": alternatives}}}, ["source_step", "steps"]))
    definitions["finish"] = ("End session with your visible-evidence assessment, independently of official benchmark success. Uncertain/unfinished tasks should use blocked.", obj({
        "source_step": {"type": "integer", "minimum": 0}, "assessment": {"enum": ["completed", "blocked"]},
        "reason": {"type": "string", "minLength": 1}}, ["source_step", "assessment", "reason"]))
    return [{"type": "function", "function": {"name": n, "description": d, "parameters": s}}
            for n,(d,s) in definitions.items()]


def public(value):
    """Official success is evaluation-only, not an oracle in the planner context."""
    if isinstance(value, dict):
        return {k: public(v) for k,v in value.items() if k != "official_success"}
    if isinstance(value, (list, tuple)):
        return [public(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


class ToolHub:
    def __init__(self, robot, executor, run: Path, emit):
        self.robot, self.executor, self.run, self.emit = robot, executor, run, emit
        self.schemas = tool_schemas()
        self.validators = {t["function"]["name"]: Draft202012Validator(t["function"]["parameters"]) for t in self.schemas}
        self.observations = {}
        self.finished = None
        self.fatal = False

    def state(self):
        from scipy.spatial.transform import Rotation
        state = public(self.robot.state())
        r = Rotation.from_quat(state["quaternion_xyzw"]).as_matrix()
        state["tool_axes_world"] = {"finger_direction": r[:,2].tolist(), "closing_axis": r[:,0].tolist()}
        state["remaining_actions"] = self.robot.max_steps - self.robot.steps
        return state

    def observe(self):
        token = f"obs_{len(self.observations):05d}"
        frames = {c: self.robot.camera(c) for c in self.robot.cameras}
        state = self.state()
        folder = self.run / "observations" / token
        folder.mkdir(parents=True)
        self.observations[token] = frames
        description = {"observation_id": token, "source_step": state["step"], "state": state,
                       "task_language": self.robot.metadata["task_language"], "cameras": []}
        parts = []
        for camera, frame in frames.items():
            Image.fromarray(frame["rgb"]).save(folder / (camera+".png"))
            np.savez_compressed(folder / (camera+".npz"), depth_m=frame["depth_m"], K=frame["K"],
                                world_from_camera=frame["world_from_camera"])
            description["cameras"].append({"camera": camera, "width": frame["rgb"].shape[1],
                                          "height": frame["rgb"].shape[0], "K": public(frame["K"]),
                                          "world_from_camera": public(frame["world_from_camera"]),
                                          "convention": frame["convention"]})
            buffer = BytesIO()
            Image.fromarray(frame["rgb"]).save(buffer, format="PNG")
            parts += [{"type": "text", "text": f"{token}: camera={camera}, source_step={state['step']}"},
                      {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()}}]
        (folder / "observation.json").write_text(json.dumps(description, indent=2))
        self.emit({"event": "observation_saved", **description})
        return description, [{"type": "text", "text": json.dumps(description)}] + parts

    def call(self, name, arguments):
        before = self.robot.steps
        start = time.perf_counter()
        images = []
        result = {"ok": False}
        action = name in PARAMS or name == "execute_plan"
        entered_executor = False
        try:
            if self.finished or self.fatal:
                raise ValueError("Session has ended; no further execution")
            if name not in self.validators:
                raise ValueError("Unknown tool")
            errors = list(self.validators[name].iter_errors(arguments))
            if errors:
                raise ValueError("Invalid parameters: " + errors[0].message)
            if "source_step" in arguments and arguments["source_step"] != self.robot.steps:
                raise ValueError(f"Stale source_step; current step is {self.robot.steps}; no action sent")
            if name == "observe":
                value, images = self.observe()
            elif name == "get_state":
                value = self.state()
            elif name == "back_project":
                frames = self.observations.get(arguments["observation_id"])
                if frames is None:
                    raise ValueError("Unknown observation_id; use an actual saved observation")
                frame = frames[arguments["camera"]]
                xyz = back_project(frame, arguments["pixel_xy"])
                value = {"world_xyz": xyz.tolist(), "source_step": frame["step"],
                         "camera": frame["camera"], "observation_id": arguments["observation_id"],
                         "current": frame["step"] == self.robot.steps,
                         "meaning": "Observed visible surface point, not a grasp pose or free-space certificate"}
            elif name == "finish":
                self.finished = dict(arguments)
                value = self.finished
            else:
                steps = arguments["steps"] if name == "execute_plan" else [{"primitive": name, "params": arguments["params"]}]
                value = {"steps": [], "completed_steps": 0, "all_commands_completed": True}
                for index, step in enumerate(steps):
                    entered_executor = True
                    output = getattr(self.executor, step["primitive"])(**step["params"])
                    value["steps"].append({"index": index, "primitive": step["primitive"], "result": public(output)})
                    if output.get("converged") is False:
                        value["all_commands_completed"] = False
                        break
                    value["completed_steps"] += 1
            result = {"ok": True, "result": public(value)}
        except Exception as exc:
            # Known parameter / IK planning failures can be repaired by the Agent.
            # Unexpected errors after entering execution stop the session; never replay automatically.
            recoverable = isinstance(exc, ValueError) or type(exc).__name__ == "IKError"
            self.fatal = self.fatal or (entered_executor and not recoverable)
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                      "fatal": self.fatal, "action_count": self.robot.steps-before,
                      "dispatch": ("execution_failed_or_uncertain" if self.fatal else
                                   "no_physics_steps" if self.robot.steps == before else "partial_execution")}
            if action and "value" in locals():
                if isinstance(value, dict) and "all_commands_completed" in value:
                    value["all_commands_completed"] = False
                result["partial_plan"] = public(value)
        if action:
            try:
                description, images = self.observe()
                result["observation"] = description
            except Exception as exc:
                self.fatal = True
                result["observation_error"] = type(exc).__name__
                result["fatal"] = True
        result["source_step"] = self.robot.steps
        result["tool_wall_s"] = time.perf_counter()-start
        self.emit({"event": "tool_result", "name": name, "arguments": arguments, **result})
        return result, images
