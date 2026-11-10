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
