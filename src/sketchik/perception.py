"""Read-only RGB-D perception for high-level tools; geometry stays in Python."""
from __future__ import annotations

import base64
from copy import deepcopy
from io import BytesIO
import json
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .grasp_preview import CAMERAS, request


def image_part(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return {"type": "image_url", "image_url": {
        "url": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()}}


class SavedObservationRobot:
    """Read-only camera/state adapter for the existing static grasp trial."""
    cameras = CAMERAS
    static_observation = True

    def __init__(self, folder):
        self.folder = folder
        self._state = json.loads((folder / "initial_state.json").read_text())
        self.steps = self._state["step"]
        self.max_steps = self.steps  # No motion is exposed in this planning phase.
        self.metadata = {"task_language": json.loads((folder / "metadata.json").read_text())["task_language"]}

    def state(self):
        return deepcopy(self._state)

    def camera(self, name):
        if name not in self.cameras:
            raise ValueError("Unknown calibrated camera")
        with np.load(self.folder / (name + ".npz")) as data:
            return {"rgb": np.asarray(Image.open(self.folder / (name + ".png")).convert("RGB")),
                    "depth_m": data["depth_m"].copy(), "K": data["K"].copy(),
                    "world_from_camera": data["world_from_camera"].copy(),
                    "camera": name, "step": self.steps,
                    "convention": "top-left pixels; camera x right, y down, z forward; metric depth"}


class ObservationHistory:
    """Record controls/video; legacy diagnostics may explicitly sample RGB-D."""
    def __init__(self, robot, run, emit, interval_s=1., video=None):
        if interval_s is not None and (not np.isfinite(interval_s) or interval_s < .05):
            raise ValueError('History interval must be at least one 20Hz simulation step')
        self.robot, self.run, self.emit, self.video = robot, run, emit, video
        self.interval_steps = None if interval_s is None else max(1, round(interval_s * 20))
        self.interval_s = None if self.interval_steps is None else self.interval_steps / 20
        self.hub = None
        self.next_step = None
        self.periodic_ids = []
        self.control_count = 0
        self.video_frames = 0
        self.video_wall_s = 0.
        self.periodic_wall_s = 0.
        self.stream = (run / 'observation_controls.jsonl').open('x', buffering=1)

    def attach(self, hub):
        self.hub = hub
        self.next_step = None if self.interval_steps is None else self.robot.steps + self.interval_steps

    def on_step(self, state):
        self.control_count += 1
        record = {'source_step': state['step'], 'joint_target': state['joint_target'],
                  'gripper_command': state['gripper_command']}
        self.stream.write(json.dumps(record, allow_nan=False) + '\n')
        if self.video is not None:
            started = time.perf_counter()
            self.video.append_data(self.robot.camera_rgb('agentview'))
            self.video_frames += 1
            self.video_wall_s += time.perf_counter() - started
        if self.hub is not None and self.next_step is not None and state['step'] >= self.next_step:
            started = time.perf_counter()
            observation, _ = self.hub.observe()
            self.periodic_wall_s += time.perf_counter() - started
            self.periodic_ids.append(observation['observation_id'])
            self.emit({'event': 'periodic_history_saved', 'observation_id': observation['observation_id'],
                       'source_step': state['step'], 'interval_simulation_s': self.interval_s})
            self.next_step = state['step'] + self.interval_steps

    def close(self):
        self.stream.close()
        manifest = {'interval_simulation_s': self.interval_s, 'periodic_observation_ids': self.periodic_ids,
                    'periodic_rgbd_enabled': self.interval_steps is not None,
                    'video_recorded': self.video is not None,
                    'video_frames': self.video_frames, 'video_wall_s': self.video_wall_s,
                    'periodic_capture_wall_s': self.periodic_wall_s,
                    'recorded_control_steps': self.control_count,
                    'keyframes': 'Initial image and each completed/failed action are also saved by ToolHub.',
                    'sampling_clock': '20Hz physics steps; no captures during API waiting without physics'}
        (self.run / 'history_manifest.json').write_text(json.dumps(manifest, indent=2))


class PerceptionTools:
    def __init__(self, hub, sam_url, sam_request=None):
        self.hub, self.sam_url = hub, sam_url.rstrip("/")
        self.request = sam_request or request
        self.candidates = {}
        self.segment_cache = {}

    def frames(self, observation_id):
        if observation_id not in self.hub.observations:
            raise ValueError("Unknown observation_id; use observe or a supplied observation")
        return self.hub.observations[observation_id]

    def metadata(self, observation_id):
        frames = self.frames(observation_id)
        step = next(iter(frames.values()))["step"]
        return {"observation_id": observation_id, "source_step": step,
                "current": step == self.hub.robot.steps,
                "static_observation": bool(getattr(self.hub.robot, "static_observation", False))}

    def overlay(self, observation_id, camera, candidate_ids):
        frame = self.frames(observation_id)[camera]
        rgb = frame["rgb"].copy()
        labels = []
        for identity in candidate_ids:
            candidate = self.candidates.get(identity)
            if candidate is None or candidate["observation_id"] != observation_id:
                raise ValueError("Candidate ID does not belong to this observation")
            if candidate["camera"] != camera:
                continue
            color = np.array([235, 90 + (candidate["number"] * 47) % 130, 60])
            mask = candidate["mask"]
            rgb[mask] = (.65 * rgb[mask] + .35 * color).astype(np.uint8)
            labels.append((candidate["box_xyxy"], identity))
        image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default(size=max(12, image.width // 55))
        for box, identity in labels:
            # Boxes may extend outside the image; that does not invalidate an observed mask.
            draw.rectangle(box, outline="red", width=2)
            draw.text((max(0, box[0]), max(0, box[1])), identity, font=font,
                      fill="red", stroke_width=1, stroke_fill="white")
        return image

    def look(self, observation_id=None, camera="all", grid=False, crop_xyxy=None, zoom=1., candidate_ids=None):
        if observation_id is None:
            observation_id = self.hub.observe()[0]["observation_id"]
        frames = self.frames(observation_id)
        cameras = list(frames) if camera == "all" else [camera]
        views, parts = [], []
        for cam in cameras:
            image = self.overlay(observation_id, cam, candidate_ids or [])
            w, h = image.size
            crop = list(crop_xyxy or [0, 0, w, h])
            x0, y0, x1, y1 = crop
            if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
                raise ValueError("Crop must be inside the original image")
            if grid:
                draw = ImageDraw.Draw(image)
                font = ImageFont.load_default(size=max(9, min(18, min(x1-x0, y1-y0) // 25)))
                gap = max(16, int(max(x1 - x0, y1 - y0) / 6))
                for x in range(x0, x1, gap):
                    draw.line([(x, y0), (x, y1)], fill="yellow")
                    draw.text((x + 2, y0 + 2), str(x), font=font,
                              fill="yellow", stroke_width=1, stroke_fill="black")
                for y in range(y0, y1, gap):
                    draw.line([(x0, y), (x1, y)], fill="yellow")
                    if y != y0:
                        draw.text((x0 + 2, y + 2), str(y), font=font,
                                  fill="yellow", stroke_width=1, stroke_fill="black")
            image = image.crop(crop)
            image = image.resize((round(image.width * zoom), round(image.height * zoom)))
            view = {"camera": cam, "original_size": [w, h], "crop_xyxy": crop,
                    "zoom": zoom, "display_size": list(image.size),
                    "pixel_mapping": "original_pixel = displayed_pixel / zoom + crop_xyxy[:2]",
                    "grid_labels": "original full-image coordinates"}
            views.append(view)
            parts += [{"type": "text", "text": json.dumps({**self.metadata(observation_id), **view})}, image_part(image)]
            folder = self.hub.run / "observations" / observation_id / "views"
            folder.mkdir(exist_ok=True)
            image.save(folder / f"view_{len(list(folder.glob('*.png'))):05d}_{cam}.png")
        return {**self.metadata(observation_id), "views": views}, parts

    def segment(self, observation_id, prompt):
        frames = self.frames(observation_id)
        key = (observation_id, prompt)
        cached = key in self.segment_cache
        if not cached:
            if set(frames) != set(CAMERAS):
                raise ValueError("SAM service requires both calibrated camera views")
            payload = {"prompt": prompt, "images": []}
            for camera in CAMERAS:
                part = image_part(Image.fromarray(frames[camera]["rgb"]))
                payload["images"].append({"camera": camera, "image_base64": part["image_url"]["url"].split(",", 1)[1]})
            begin = time.perf_counter()
            response = self.request(self.sam_url + "/infer", payload)
            elapsed = time.perf_counter() - begin
            views = response.get("images", [])
            if len(views) != 2 or {v["camera"] for v in views} != set(CAMERAS):
                raise ValueError("SAM returned mismatched camera labels")
            pending = []
            for view in views:
                frame = frames[view["camera"]]
                for c in view["candidates"]:
                    h, w = c["mask_shape"]
                    if (h, w) != frame["rgb"].shape[:2]:
                        raise ValueError("SAM Mask and calibrated RGB-D resolutions differ")
                    bits = np.unpackbits(np.frombuffer(base64.b64decode(c["mask_packbits_base64"], validate=True), dtype=np.uint8))
                    if len(bits) < h * w:
                        raise ValueError("SAM Mask data is incomplete")
                    box = np.asarray(c["box_xyxy"], dtype=float)
                    if box.shape != (4,) or not np.isfinite(box).all() or not np.isfinite(c["score"]):
                        raise ValueError("SAM returned invalid box or score")
                    if box[0] > box[2] or box[1] > box[3] or not 0 <= c["score"] <= 1:
                        raise ValueError("SAM returned invalid box bounds or score range")
                    mask = bits[:h*w].reshape(h, w).astype(bool)
                    pending.append({"camera": view["camera"], "mask": mask,
                                    "score": float(c["score"]), "box_xyxy": box.tolist()})
            ids = []
            for c in pending:
                number = len(self.candidates) + 1
                identity = f"mask_{number:05d}"
                c.update(number=number, candidate_id=identity, observation_id=observation_id, prompt=prompt)
                self.candidates[identity] = c
                ids.append(identity)
                folder = self.hub.run / "observations" / observation_id / "masks"
                folder.mkdir(exist_ok=True)
                Image.fromarray(c["mask"].astype(np.uint8) * 255).save(folder / (identity + ".png"))
            self.segment_cache[key] = {"candidate_ids": ids, "sam_client_wall_s": elapsed,
                "sam_inference_wall_s": response.get("inference_wall_s"), "sam_model_pid": response.get("model_pid")}
        record = self.segment_cache[key]
        candidates = [{k: c[k] for k in ("candidate_id", "camera", "score", "box_xyxy")}
                      | {"mask_pixels": int(c["mask"].sum())}
                      for identity in record["candidate_ids"] for c in [self.candidates[identity]]]
        value = {**self.metadata(observation_id), **record, "prompt": prompt,
                 "cached": cached, "candidates": candidates, "found": bool(candidates),
                 "selection": "all candidates; no automatic highest-score or positional selection"}
        _, pictures = self.look(observation_id, candidate_ids=record["candidate_ids"])
        return value, pictures


def saved_frame(source, identity, camera):
    folder = source / 'observations' / identity
    description = json.loads((folder / 'observation.json').read_text())
    with np.load(folder / (camera+'.npz')) as arrays:
        frame = {name: arrays[name].copy() for name in arrays.files}
    frame.update(rgb=np.asarray(Image.open(folder / (camera+'.png')).convert('RGB')),
                 step=description['source_step'], camera=camera,
                 convention='top-left pixels; camera x right, y down, z forward; metric depth')
    return frame


def verify_replay_observation(robot, source, identity, *, rgb_roundoff_pixels=0, diagnostic_dir=None, snapshot=False):
    folder = source / identity if snapshot else source / 'observations' / identity
    if snapshot:
        saved_state=json.loads((folder/'initial_state.json').read_text())
        description={'source_step':saved_state['step'],'state':saved_state}
    else:description = json.loads((folder / 'observation.json').read_text())
    state = robot.state()
    if state['step'] != description['source_step']:
        raise ValueError('Replay observation step mismatch')
    for field in ['q', 'position', 'quaternion_xyzw', 'finger_joint_positions']:
        if not np.array_equal(state[field], description['state'][field]):
            raise ValueError(f'Replay robot state mismatch: {identity}/{field}')
    differences = {}
    for camera in robot.cameras:
        actual=robot.camera(camera)
        if snapshot:
            with np.load(folder/(camera+'.npz')) as arrays:expected={key:arrays[key] for key in arrays.files}
            expected['rgb']=np.asarray(Image.open(folder/(camera+'.png')).convert('RGB'))
        else:expected=saved_frame(source,identity,camera)
        for field in ['depth_m', 'K', 'world_from_camera']:
            if not np.array_equal(actual[field], expected[field]):
                raise ValueError(f'Replay native RGB-D/calibration mismatch: {identity}/{camera}/{field}')
        diff = np.abs(actual['rgb'].astype(np.int16)-expected['rgb'].astype(np.int16))
        pixels = int(np.count_nonzero(np.any(diff, axis=2)))
        maximum = int(diff.max())
        differences[camera] = {'different_rgb_pixels': pixels, 'max_channel_difference': maximum}
        if pixels > rgb_roundoff_pixels or maximum > 1:
            # A repeated render at unchanged physics sometimes removes a raster
            # difference. Preserve the first draw; never advance or restore GT.
            first_difference = dict(differences[camera])
            if diagnostic_dir is not None:
                diagnostic_dir.mkdir(parents=True, exist_ok=True)
                Image.fromarray(actual['rgb']).save(diagnostic_dir / (identity+'_'+camera+'_first.png'))
            redraws=[]
            for redraw in range(1,4):
                again = robot.camera(camera)
                now = robot.state()
                if now['step']!=state['step'] or any(not np.array_equal(now[field],state[field])
                        for field in ['q','position','quaternion_xyzw','finger_joint_positions']):
                    raise ValueError('Replay state changed during repeated render')
                for field in ['depth_m', 'K', 'world_from_camera']:
                    if not np.array_equal(again[field], expected[field]):
                        raise ValueError(f'Replay repeated render geometry mismatch: {identity}/{camera}/{field}')
                diff = np.abs(again['rgb'].astype(np.int16)-expected['rgb'].astype(np.int16))
                pixels, maximum = int(np.count_nonzero(np.any(diff, axis=2))), int(diff.max())
                accepted=pixels<=rgb_roundoff_pixels and maximum<=1
                redraws.append({'redraw':redraw,'different_rgb_pixels':pixels,
                    'max_channel_difference':maximum,'accepted':accepted})
                if diagnostic_dir is not None:
                    Image.fromarray(again['rgb']).save(diagnostic_dir/f'{identity}_{camera}_redraw_{redraw:02d}.png')
                    (diagnostic_dir/f'{identity}_{camera}_redraws.json').write_text(json.dumps({
                        'source_step':state['step'],'first_draw_difference':first_difference,
                        'rgb_roundoff_limit_pixels':rgb_roundoff_pixels,'redraws':redraws},indent=2))
                if accepted:break
            else:
                raise ValueError(f'Replay native RGB-D/calibration mismatch: {identity}/{camera}/rgb')
            differences[camera] = {'different_rgb_pixels': pixels, 'max_channel_difference': maximum,
                                   'same_physical_step_rerender': True, 'first_draw_difference': first_difference,
                                   'same_physical_step_redraws':redraws}
    exact = all(item['different_rgb_pixels'] == 0 for item in differences.values())
    return {'observation_id': identity, 'source_step': state['step'],
            'robot_state_exact': True, 'native_geometry_and_calibration_exact': True,
            'native_rgbd_and_calibration_exact': exact, 'rgb_roundoff_limit_pixels': rgb_roundoff_pixels,
            'rgb_differences': differences}


def snapshot(robot, folder, metadata, episode_id):
    from .tools import public
    folder.mkdir(exist_ok=False)
    state = public(robot.state()) | {'episode_id': episode_id}
    (folder/'initial_state.json').write_text(json.dumps(state, indent=2))
    (folder/'metadata.json').write_text(json.dumps(metadata | {'episode_id': episode_id}, indent=2))
    (folder/'robot_only.xml').write_text(robot.robot.robot_model.get_xml())
    for camera in robot.cameras:
        frame = robot.camera(camera)
        Image.fromarray(frame['rgb']).save(folder/(camera+'.png'))
        np.savez_compressed(folder/(camera+'.npz'), **{k: frame[k] for k in ('depth_m', 'K', 'world_from_camera')})
    return state
