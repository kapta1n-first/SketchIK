"""Compact evidence for visible horizontal patches in a calibrated RGB-D frame.

This measures only observed pixels. A patch is neither an object identity nor a
claim about hidden extent, empty space, or whether an object can rest on it.
"""

import numpy as np
from scipy import ndimage


def summarize_horizontal_surfaces(frame: dict, *, max_patches: int = 8) -> dict:
    """Return JSON-serializable visible surface evidence from depth/K/camera pose.

    Bounds use native image pixels, with an exclusive upper right corner. They
    are image envelopes and may contain holes or unrelated surfaces. The height
    is an approximate elevation of sampled visible pixels.
    """
    if not isinstance(max_patches, int) or max_patches < 1:
        raise ValueError("max_patches must be a positive integer")
    depth = np.asarray(frame["depth_m"])
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError("depth_m must be a two-dimensional image")
    k = np.asarray(frame["K"], dtype=float)
    pose = np.asarray(frame["world_from_camera"], dtype=float)
    if k.shape != (3, 3) or pose.shape != (4, 4) or not np.isfinite(k).all() or not np.isfinite(pose).all():
        raise ValueError("invalid camera calibration")
    if abs(np.linalg.det(k)) < 1e-12:
        raise ValueError("K must be invertible")
    height, width = depth.shape
    empty = {"scope": "approximate_visible_near_horizontal_pixels_only", "horizontal_patches": []}
    if height < 3 or width < 3:
        return empty
    stride = max(1, int(np.ceil(max(height, width) / 256)))
    ys = np.arange(stride // 2, height, stride)
    xs = np.arange(stride // 2, width, stride)
    if len(ys) < 3 or len(xs) < 3:
        return empty
    sampled = np.asarray(depth[np.ix_(ys, xs)], dtype=float)
    valid = np.isfinite(sampled) & (sampled > 0)
    xx, yy = np.meshgrid(xs, ys)
    rays = np.stack((xx, yy, np.ones_like(xx)), axis=-1) @ np.linalg.inv(k).T
    xyz = (rays * sampled[..., None]) @ pose[:3, :3].T + pose[:3, 3]
    xyz[~valid] = np.nan
    # Central differences span two sampled pixels. Reject discontinuities before
    # calculating normals so a depth silhouette cannot become a false plane.
    delta_x = xyz[1:-1, 2:] - xyz[1:-1, :-2]
    delta_y = xyz[2:, 1:-1] - xyz[:-2, 1:-1]
    local_depth = sampled[1:-1, 1:-1]
    depth_jump = np.maximum.reduce((
        abs(sampled[1:-1, 2:] - local_depth),
        abs(sampled[1:-1, :-2] - local_depth),
        abs(sampled[2:, 1:-1] - local_depth),
        abs(sampled[:-2, 1:-1] - local_depth),
    ))
    normal = np.cross(delta_x, delta_y)
    length = np.linalg.norm(normal, axis=-1)
    normal_z = np.divide(abs(normal[..., 2]), length, out=np.zeros_like(length), where=length > 1e-10)
    candidates = np.zeros(sampled.shape, dtype=bool)
    candidates[1:-1, 1:-1] = (
        valid[1:-1, 1:-1] & np.isfinite(length) &
        (depth_jump < np.maximum(0.025, 0.025 * local_depth)) &
        (normal_z >= 0.97)
    )
    # Height bins prevent separated levels joined by thin or noisy bridges from
    # being reported as one plane. The two offsets avoid arbitrary bin edges.
    labels_seen = []
    z = xyz[..., 2]
    for offset in (0.0, 0.015):
        bins = np.zeros(z.shape, dtype=np.int32)
        bins[candidates] = np.floor((z[candidates] + offset) / 0.03).astype(np.int32)
        for value in np.unique(bins[candidates]):
            labels, count = ndimage.label(candidates & (bins == value))
            for index in range(1, count + 1):
                rows, cols = np.nonzero(labels == index)
                if rows.size < 12:
                    continue
                levels = z[rows, cols]
                center = float(np.median(levels))
                residual = float(np.sqrt(np.mean((levels - center) ** 2)))
                if residual > 0.012:
                    continue
                bounds = [int(xs[cols.min()]), int(ys[rows.min()]),
                          int(min(width, xs[cols.max()] + stride)),
                          int(min(height, ys[rows.max()] + stride))]
                labels_seen.append({
                    "bounds_xyxy": bounds,
                    "visible_pixel_sample_count": int(rows.size),
                    "visible_pixel_sample_fraction": round(float(rows.size / sampled.size), 5),
                    "world_height_m": round(center, 6),
                    "normal_angle_to_z_deg": round(float(np.degrees(np.arccos(np.clip(np.median(normal_z[rows - 1, cols - 1]), 0, 1)))), 2),
                    "height_residual_rmse_m": round(residual, 6),
                })
    # Offset passes can describe the same patch. Keep the larger observation.
    labels_seen.sort(key=lambda item: item["visible_pixel_sample_count"], reverse=True)
    distinct = []
    for patch in labels_seen:
        if any(abs(patch["world_height_m"] - prior["world_height_m"]) < 0.015 and
               _overlap(patch["bounds_xyxy"], prior["bounds_xyxy"]) > 0.7
               for prior in distinct):
            continue
        distinct.append(patch)
        if len(distinct) >= max_patches:
            break
    return {"scope": "approximate_visible_near_horizontal_pixels_only", "horizontal_patches": distinct}


def _overlap(a, b):
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    area = (a[2] - a[0]) * (a[3] - a[1])
    return intersection / area if area else 0.0
