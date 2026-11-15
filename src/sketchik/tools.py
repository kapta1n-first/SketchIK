"""Model-facing tools, public artifacts and per-run files. No shell or object GT."""
from __future__ import annotations

import base64
from io import BytesIO
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
from jsonschema import Draft202012Validator

from .libero import back_project
from .timing import timings_for


def obj(properties, required=()):
    return {"type": "object", "properties": properties, "required": list(required), "additionalProperties": False}


V3 = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}
Q4 = {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4}
STEPS = {"type": "integer", "minimum": 1, "maximum": 200}
WAYPOINT = obj({"position": V3, "relative_m": V3, "quaternion_xyzw": Q4})
WAYPOINT["oneOf"] = [{"required": ["position"], "not": {"required": ["relative_m"]}},
                     {"required": ["relative_m"], "not": {"required": ["position"]}}]
PARAMS = {
    "move_relative": obj({"delta_m": V3, "quaternion_xyzw": Q4}, ["delta_m"]),
    "move": obj({"position": V3, "quaternion_xyzw": Q4,
                 "finger_direction": V3, "closing_axis": V3}, ["position"]),
    "rotate_roll": obj({"angle_rad":{"type":"number"}},["angle_rad"]),
    "rotate_pitch": obj({"angle_rad":{"type":"number"}},["angle_rad"]),
    "rotate_yaw": obj({"angle_rad":{"type":"number"}},["angle_rad"]),
    "move_waypoints": obj({"waypoints": {"type": "array", "items": WAYPOINT, "minItems": 1, "maxItems": 30}}, ["waypoints"]),
    "reset": obj({"settle_steps":{"type":"integer","minimum":0,"maximum":200}}),
    "set_gripper": obj({"command": {"enum": ["open", "close"]}, "steps": STEPS}, ["command"]),
    "release": obj({"steps": STEPS}),
    "wait": obj({"steps": STEPS}),
}
PARAMS["move"]["oneOf"] = [
    {"required": ["quaternion_xyzw"], "not": {"anyOf": [{"required": ["finger_direction"]}, {"required": ["closing_axis"]}]}},
    {"required": ["finger_direction", "closing_axis"], "not": {"required": ["quaternion_xyzw"]}},
    {"not": {"anyOf": [{"required": ["quaternion_xyzw"]}, {"required": ["finger_direction"]}, {"required": ["closing_axis"]}]}},
]
for key in ("move_relative", "move", "rotate_roll", "rotate_pitch", "rotate_yaw", "move_waypoints"):
    PARAMS[key]["properties"]["settle_steps"] = {"type": "integer", "minimum": 0, "maximum": 200}
DESCRIPTIONS = {
    "move_relative": "Move by signed world XYZ displacement from actual entry pose; preserve orientation unless specified. Displacement is resolved once.",
    "move": "Move EEF to absolute world XYZ (metres) using IK and joint control. Preserve the full entry orientation when omitted; optionally supply a unit xyzw quaternion OR both world finger_direction (+tool Z) and closing_axis (+tool X). Keep the gripper command. No automatic grasp pose generation or collision planner.",
    "rotate_roll": "Rotate the full EEF orientation by signed relative angle_rad radians about fixed world X using real IK. Keep actual entry XYZ (metres) and gripper command; R1=Rx(angle_rad)@R0. settle_steps defaults to 80. No automatic collision planning.",
    "rotate_pitch": "Rotate the full EEF orientation by signed relative angle_rad radians about fixed world Y using real IK. Keep actual entry XYZ (metres) and gripper command; R1=Ry(angle_rad)@R0. settle_steps defaults to 80. No automatic collision planning.",
    "rotate_yaw": "Rotate the full EEF orientation by signed relative angle_rad radians about fixed world Z using real IK. Keep actual entry XYZ (metres) and gripper command; R1=Rz(angle_rad)@R0. settle_steps defaults to 80. No automatic collision planning.",
    "move_waypoints": "Move through supplied EEF waypoints. position is absolute; relative_m is from preceding planned waypoint (first from actual entry). Can express caller-planned detours; does not search for obstacle-free paths.",
    "reset": "Physically return the arm to this executor's recorded initial full EEF pose using IK and joint control. Keep the gripper command, world objects and observation history. Does not reset the environment or guarantee identical arm joint angles; no automatic collision planning in the ordinary interface.",
    "set_gripper": "Command open/close while holding current joints. Does not establish object identity or holding success.",
    "release": "Open gripper at current pose only. Does not compute a placement point or certify placement.",
    "wait": "Hold measured joint positions and previous gripper command while advancing 1..200 physics steps (20Hz). Not a wall-clock sleep or success detector.",
}


def tool_schemas():
    definitions = {
        "observe": ("Get current labelled agentview and wrist RGB, robot state, tool axes, remaining action budget and calibration; depth stays local for back_project.", obj({})),
        "get_state": ("Read current robot state without rendering images or advancing physics.", obj({})),
        "look": ("View current or saved RGB with optional pixel grid, crop, zoom and candidate masks. Grid labels retain original pixels; use those original pixels with back_project. Does not move the camera.", obj({
            "observation_id": {"type": "string"},
            "camera": {"enum": ["all", "agentview", "robot0_eye_in_hand"]},
            "grid": {"type": "boolean"},
            "crop_xyxy": {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4},
            "zoom": {"type": "number", "minimum": 1, "maximum": 4},
            "candidate_ids": {"type": "array", "items": {"type": "string"}, "uniqueItems": True}})),
        "segment": ("Text-prompt SAM3 segmentation of both saved calibrated views, including whole objects or visible parts. Returns ALL candidate IDs/scores/boxes and numbered mask images; no highest-score selection. Same observation/prompt is cached. No robot movement.", obj({
            "observation_id": {"type": "string"}, "prompt": {"type": "string", "minLength": 1, "maxLength": 2048}},
            ["observation_id", "prompt"])),
        "read_text_file": ("Read project instructions/examples, your current-run workspace files, or saved public observation JSON. Paths are relative to list_dir roots; no raw simulator/credential files.", obj({
            "path": {"type": "string", "minLength": 1},
            "max_chars": {"type": "integer", "minimum": 1, "maximum": 40000}}, ["path"])),
        "write_text_file": ("Save a temporary plan/note in workspace/ for this run. Project instructions and original observation artifacts are read-only. Existing files require explicit overwrite=true; no long-term memory.", obj({
            "path": {"type": "string", "minLength": 1}, "content": {"type": "string", "maxLength": 100000},
            "overwrite": {"type": "boolean"}}, ["path", "content"])),
        "list_dir": ("List readable project instructions/examples, current-run workspace or saved public observations. Does not advance the robot.", obj({"path": {"type": "string"}})),
        "read_image": ("View a saved RGB from a known observation_id and camera. Returns the actual historical image, its original source_step and whether it is still current. Does not capture a new frame or move the robot.", obj({
            "observation_id": {"type": "string"}, "camera": {"enum": ["agentview", "robot0_eye_in_hand"]}},
            ["observation_id", "camera"])),
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


PERCEPTION_TOOLS = frozenset(("observe", "get_state", "read_image", "back_project", "look", "segment"))


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
    def __init__(self, robot, executor, run: Path, emit, project_root=None,
                 sam_url=None, sam_request=None, allowed_tools=None):
        self.robot, self.executor, self.run, self.emit = robot, executor, run, emit
        self.project_root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
        self.workspace = self.run / "workspace"
        self.workspace.mkdir(exist_ok=True)
        self.schemas = [s for s in tool_schemas() if allowed_tools is None or s["function"]["name"] in allowed_tools]
        self.validators = {t["function"]["name"]: Draft202012Validator(t["function"]["parameters"]) for t in self.schemas}
        self.observations = {}
        self.finished = None
        self.fatal = False
        from .perception import PerceptionTools
        self.vision = PerceptionTools(self, sam_url or os.environ.get("SKETCHIK_SAM3_URL", "http://127.0.0.1:18761"), sam_request)

    def file_path(self, name, write=False):
        """Only public docs/examples, observation artifacts, and this run's scratch files.

        Raw run logs/results may contain evaluation GT; do not expose them via IO.
        """
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError("Use a relative path under the listed roots")
        key = relative.parts[0]
        if key == "workspace":
            base, tail = self.workspace, relative.parts[1:]
        elif key == "observations" and not write:
            base, tail = self.run / "observations", relative.parts[1:]
        elif not write and key in ("prompts", "examples"):
            base, tail = self.project_root / key, relative.parts[1:]
        elif not write and name in ("README.md", "changes.md"):
            path = self.project_root / name
            if path.is_symlink():
                raise ValueError("Symlink paths are not supported")
            return path
        else:
            raise ValueError("Readable roots: README.md, changes.md, prompts/, examples/, observations/, workspace/; only workspace/ is writable")
        path = base.joinpath(*tail)
        # Reject symlink components instead of following links into raw run/private files.
        current = base
        if current.is_symlink():
            raise ValueError("Symlink paths are not supported")
        for part in tail:
            current = current / part
            if current.is_symlink():
                raise ValueError("Symlink paths are not supported")
        if not path.resolve().is_relative_to(base.resolve()):
            raise ValueError("Path escapes listed root")
        return path

    def display_image(self, observation_id, camera, frame):
        """Presentation hook; saved native RGB-D and pixel geometry stay local."""
        buffer = BytesIO()
        Image.fromarray(frame['rgb']).save(buffer, format='PNG')
        return {}, {'type':'image_url','image_url':{'url':
            'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()}}

    def read_image(self, observation_id, camera):
        frames = self.observations.get(observation_id)
        if frames is None:
            raise ValueError("Unknown observation_id; use list_dir or observe")
        frame = frames[camera]
        description = {"observation_id": observation_id, "camera": camera,
                       "source_step": frame["step"], "current": frame["step"] == self.robot.steps,
                       "width": frame["rgb"].shape[1], "height": frame["rgb"].shape[0],
                       "meaning": "Historical RGB, not a new observation"}
        display, picture = self.display_image(observation_id,camera,frame)
        if display:description['display_annotation']=display
        return description, [{"type": "text", "text": json.dumps(description)},picture]

    def state(self):
        from scipy.spatial.transform import Rotation
        state = public(self.robot.state())
        r = Rotation.from_quat(state["quaternion_xyzw"]).as_matrix()
        state["tool_axes_world"] = {"finger_direction": r[:,2].tolist(), "closing_axis": r[:,0].tolist()}
        state["remaining_actions"] = (None if getattr(self.robot, "static_observation", False)
                                      else self.robot.max_steps - self.robot.steps)
        return state

    def observe(self):
        token = f"obs_{len(self.observations):05d}"
        with timings_for(self).scope('rgbd_camera_render'):
            frames = {c: self.robot.camera(c) for c in self.robot.cameras}
        state = self.state()
        folder = self.run / "observations" / token
        folder.mkdir(parents=True)
        self.observations[token] = frames
        description = {"observation_id": token, "source_step": state["step"], "state": state,
                       "task_language": self.robot.metadata["task_language"], "cameras": [],
                       "static_observation": bool(getattr(self.robot, "static_observation", False))}
        parts = []
        for camera, frame in frames.items():
            with timings_for(self).scope('rgbd_file_write'):
                Image.fromarray(frame["rgb"]).save(folder / (camera+".png"))
                np.savez_compressed(folder / (camera+".npz"), depth_m=frame["depth_m"], K=frame["K"],
                                    world_from_camera=frame["world_from_camera"])
            description["cameras"].append({"camera": camera, "width": frame["rgb"].shape[1],
                                          "height": frame["rgb"].shape[0], "K": public(frame["K"]),
                                          "world_from_camera": public(frame["world_from_camera"]),
                                          "convention": frame["convention"]})
            with timings_for(self).scope('rgb_display_annotation'):
                display, picture = self.display_image(token,camera,frame)
            label = f"{token}: camera={camera}, source_step={state['step']}"
            if display:label += '; display_annotation='+json.dumps(display)
            parts += [{"type": "text", "text": label},picture]
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
            elif name in ("look", "segment"):
                value, images = getattr(self.vision, name)(**arguments)
            elif name == "read_text_file":
                path = self.file_path(arguments["path"])
                if path.suffix.lower() not in (".md", ".txt", ".json", ".jsonl", ".yaml", ".yml", ".csv"):
                    raise ValueError("Use read_image for RGB; only text files can be read here")
                limit = arguments.get("max_chars", 20000)
                with path.open(encoding="utf-8") as stream:
                    content = stream.read(limit+1)
                value = {"path": arguments["path"], "content": content[:limit], "truncated": len(content)>limit}
            elif name == "write_text_file":
                path = self.file_path(arguments["path"], write=True)
                if not path.suffix.lower() in (".md", ".txt", ".json", ".jsonl", ".yaml", ".yml", ".csv"):
                    raise ValueError("Only temporary text/plan files may be written")
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("w" if arguments.get("overwrite", False) else "x", encoding="utf-8") as stream:
                    stream.write(arguments["content"])
                value = {"path": arguments["path"], "characters_written": len(arguments["content"]),
                         "scope": "current run workspace, not long-term memory"}
            elif name == "list_dir":
                name = arguments.get("path", "")
                if name in ("", "."):
                    value = {"path": "", "entries": ["README.md", "changes.md", "prompts/", "examples/", "observations/", "workspace/"]}
                else:
                    path = self.file_path(name)
                    value = {"path": name, "entries": [p.name + ("/" if p.is_dir() else "")
                              for p in sorted(path.iterdir()) if not p.is_symlink()]}
            elif name == "read_image":
                value, images = self.read_image(**arguments)
            elif name == "back_project":
                legend = self.run/'observations'/arguments['observation_id']/'a_display'/(arguments['camera']+'.json')
                if legend.exists():
                    x0,y0,x1,y1 = json.loads(legend.read_text())['panel_bounds_xyxy']
                    x,y = arguments['pixel_xy']
                    if x0 <= x < x1 and y0 <= y < y1:
                        raise ValueError('World-axis legend pixel is not an observed surface')
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
