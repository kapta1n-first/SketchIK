"""A calibrated world-direction compass on a copy of a native RGB image.

The compass is orthographic: it projects *directions*, not world points, onto
the camera's right/down plane. It is never a pixel-to-surface correspondence.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def annotate_world_axes(
    rgb: np.ndarray, world_from_camera: np.ndarray, *, source_identity=None
) -> tuple[np.ndarray, dict]:
    """Return an RGB copy and calibration metadata; do not resize the image.

    ``world_from_camera`` must be a rigid 4x4 transform with camera axes
    x=right, y=down and z=forward. Positive camera z points away from the
    viewer. Translation has no effect on this direction-only compass.
    """
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("rgb must be an HxWx3 uint8 RGB image")
    height, width = rgb.shape[:2]
    if min(height, width) < 64:
        raise ValueError("rgb must be at least 64 pixels on each side")
    transform = np.asarray(world_from_camera, dtype=float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("world_from_camera must be a finite 4x4 transform")
    rotation = transform[:3, :3]
    if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-7, rtol=0):
        raise ValueError("world_from_camera must have a homogeneous bottom row")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0):
        raise ValueError("world_from_camera rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6, rtol=0):
        raise ValueError("world_from_camera rotation must be proper (det=+1)")

    margin = min(8, min(height, width) // 16)
    panel_width = min(280, width - 2 * margin)
    panel_height = min(230, height - 2 * margin)
    panel = Image.new("RGB", (panel_width, panel_height), (24, 29, 37))
    draw = ImageDraw.Draw(panel)
    font_size = max(8, min(13, panel_width // 20))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    origin = np.array([panel_width / 2, panel_height * 0.43])
    arm_length = min(panel_width * 0.27, panel_height * 0.28)
    draw.text((6, 5), "WORLD +XYZ", fill="white", font=font)
    if source_identity is not None:
        identity=source_identity['observation_id'];camera=source_identity['camera']
        if camera not in ('agentview','robot0_eye_in_hand') or not isinstance(identity,str):
            raise ValueError('Invalid source image identity')
        label=identity+' | '+('front' if camera=='agentview' else 'wrist')
        draw.text((6, 5+font_size+2),label,fill="white",font=font)
    colors = {"+X": (255, 115, 115), "+Y": (115, 225, 145), "+Z": (125, 180, 255)}
    world_axes_camera = {}
    axes = {}
    near_optical_threshold = 0.15
    near_image_plane_threshold = 0.05
    for index, (label, color) in enumerate(colors.items()):
        # e_world -> R.T @ e_world. Screen y is already camera-down: do not flip.
        direction = rotation.T[:, index]
        projection = direction[:2]
        projected_length = float(np.linalg.norm(projection))
        delta = projection * arm_length
        endpoint = origin + delta
        near_optical = projected_length < near_optical_threshold
        # Qualitative text tolerates tiny tilt/calibration residuals; the exact
        # vector and foreshortened arrow below are never snapped to the plane.
        if abs(direction[2]) < near_image_plane_threshold:
            relation = "in image plane"
        elif direction[2] > 0:
            relation = "away from viewer"
        else:
            relation = "toward viewer"
        if near_optical:
            # A cross goes into the display; a dot comes out toward the viewer.
            x, y = origin
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=color, width=1)
            if direction[2] > 0:
                draw.line((x - 3, y - 3, x + 3, y + 3), fill=color, width=1)
                draw.line((x - 3, y + 3, x + 3, y - 3), fill=color, width=1)
            else:
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
            draw.text((x + 8, y + 4), label, fill=color, font=font)
        else:
            draw.line((*origin, *endpoint), fill=color, width=2)
            unit = projection / projected_length
            normal = np.array([-unit[1], unit[0]])
            head = min(8.0, arm_length * 0.2)
            base = endpoint - unit * head
            draw.polygon(
                [tuple(endpoint), tuple(base + normal * head / 2),
                 tuple(base - normal * head / 2)], fill=color,
            )
            text_x = float(np.clip(endpoint[0] + 4, 0, panel_width - 23))
            text_y = float(np.clip(endpoint[1] - font_size / 2, 22, panel_height - 67))
            draw.text((text_x, text_y), label, fill=color, font=font)
        world_axes_camera[label] = direction.tolist()
        axes[label] = {
            "screen_delta_px": delta.tolist(),
            "near_optical_axis": near_optical,
            "viewer_relation": relation,
        }

    # Fixed legend carries depth direction even if a projected arrow is tiny.
    row_height = font_size + 2
    legend_y = panel_height - 4 * row_height - 5
    for index, (label, color) in enumerate(colors.items()):
        relation = axes[label]["viewer_relation"]
        draw.text((6, legend_y + index * row_height), f"{label}: {relation}",
                  fill=color, font=font)
    draw.text((6, panel_height - row_height - 3), "Direction only; no points/scale",
              fill=(220, 220, 220), font=font)

    result = np.array(rgb, copy=True)
    result[margin:margin + panel_height, margin:margin + panel_width] = np.asarray(panel)
    return result, {
        'source_image':source_identity,
        "kind": "world_direction_compass",
        "projection": "orthographic camera right/down; direction only",
        "world_axes_camera": world_axes_camera,
        "axes": axes,
        "near_optical_axis_threshold": near_optical_threshold,
        "approximate_image_plane_abs_camera_z_threshold": near_image_plane_threshold,
        "compass_arm_length_px": arm_length,
        "panel_bounds_xyxy": [margin, margin, margin + panel_width, margin + panel_height],
        "annotation_pixels_not_surface": True,
        "metric_scale": False,
        "image_size_wh": [width, height],
        "source_calibration": {
            "world_from_camera": transform.tolist(),
            "camera_axes": "x right, y down, z forward (away from viewer)",
        },
    }
