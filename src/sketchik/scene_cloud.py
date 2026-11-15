"""Full calibrated RGB-D surfaces and segmentation-free observation movement.

No model requests, object masks, simulator object state or completion of hidden
surfaces. All valid source pixels are retained; only plotting is sampled.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np


def frame_points(frame):
    depth = np.asarray(frame['depth_m']).squeeze()
    rgb = np.asarray(frame['rgb'])
    if depth.ndim != 2 or rgb.shape != (*depth.shape, 3):
        raise ValueError('Native RGB and depth must have matching dimensions')
    valid = np.isfinite(depth) & (depth > 0)
    y, x = np.nonzero(valid)
    rays = np.linalg.solve(frame['K'], np.vstack([x, y, np.ones(len(x))])).T
    camera = rays * depth[y, x, None]
    transform = np.asarray(frame['world_from_camera'])
    xyz = camera @ transform[:3, :3].T + transform[:3, 3]
    return xyz.astype(np.float32), rgb[y, x].astype(np.uint8), np.column_stack([x, y]).astype(np.uint16)


def save_scene(frames, folder, observation_id):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    begin = time.perf_counter()
    steps = {frame['step'] for frame in frames.values()}
    if len(steps) != 1:
        raise ValueError('Scene views must share one refreshed physical state')
    arrays, sources, timings = [], [], []
    for index, (camera, frame) in enumerate(frames.items()):
        start = time.perf_counter()
        xyz, rgb, pixels = frame_points(frame)
        arrays.append((xyz, rgb, pixels))
        sources.append(np.full(len(xyz), index, np.uint8))
        timings.append({'camera': camera, 'source_id': index, 'source_step': frame['step'],
                        'points': len(xyz), 'projection_wall_s': time.perf_counter()-start,
                        'K': np.asarray(frame['K']).tolist(),
                        'world_from_camera': np.asarray(frame['world_from_camera']).tolist()})
    xyz = np.concatenate([v[0] for v in arrays])
    rgb = np.concatenate([v[1] for v in arrays])
    pixels = np.concatenate([v[2] for v in arrays])
    if not len(xyz):
        raise ValueError('No finite positive observed depth')
    start = time.perf_counter()
    # Uncompressed NPZ avoids spending seconds compressing complete point sets.
    # Native PNG/RGB-D files are unchanged; no transmission encoding is changed.
    cloud = folder / 'observed_scene.npz'
    np.savez(cloud, world_xyz_m=xyz, rgb=rgb, pixel_xy=pixels, source_id=np.concatenate(sources))
    save_s = time.perf_counter()-start
    start = time.perf_counter()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sample = np.linspace(0, len(xyz)-1, min(60000, len(xyz)), dtype=int)
    figure = plt.figure(figsize=(10, 5))
    ax = figure.add_subplot(121, projection='3d')
    ax.scatter(*xyz[sample].T, c=rgb[sample]/255., s=.7, depthshade=False)
    ax.set_xlabel('World X (m)'); ax.set_ylabel('World Y (m)'); ax.set_zlabel('World Z (m)')
    ax.set_box_aspect(np.maximum(np.ptp(xyz, axis=0), .01))
    ax.view_init(32, 35)
    ax.set_title('Entire observed scene / RGB')
    top = figure.add_subplot(122)
    drawing_order = sample[np.argsort(xyz[sample, 2])]
    top.scatter(xyz[drawing_order, 0], xyz[drawing_order, 1], c=rgb[drawing_order]/255., s=.6)
    top.set_aspect('equal'); top.set_xlabel('World X (m)'); top.set_ylabel('World Y (m)')
    top.set_title('Entire scene / top projection')
    figure.suptitle('Observed surfaces only; display sampled, all XYZ saved')
    figure.tight_layout()
    picture = folder / 'scene_3d.png'
    figure.savefig(picture, dpi=120)
    plt.close(figure)
    report = {'observation_id': observation_id, 'source_step': next(iter(steps)),
        'scope': 'all valid pixels in both complete native RGB-D views',
        'point_count': len(xyz), 'point_cloud_file': str(cloud), 'cloud_image': str(picture),
        'world_min_xyz_m': xyz.min(axis=0).tolist(), 'world_max_xyz_m': xyz.max(axis=0).tolist(),
        'scene_span_xyz_m': np.ptp(xyz, axis=0).tolist(), 'scene_span_is_object_dimensions': False,
        'frames': timings, 'point_save_wall_s': save_s, 'render_wall_s': time.perf_counter()-start,
        'wall_s': time.perf_counter()-begin, 'display_points': len(sample),
        'sam_requests': 0, 'molmo_requests': 0, 'object_gt_used': False,
        'hidden_surface_completion': False, 'robot_points_filtered': False}
    (folder / 'scene.json').write_text(json.dumps(report, indent=2))
    return report


def region_points(frame, bounds):
    """A coarse image rectangle is a context window, never a semantic mask."""
    x0, y0, x1, y1 = bounds
    h, w = np.asarray(frame['depth_m']).squeeze().shape
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError('Invalid native image rectangle')
    xyz, _, pixels = frame_points(frame)
    selected = (pixels[:, 0] >= x0) & (pixels[:, 0] < x1) & (pixels[:, 1] >= y0) & (pixels[:, 1] < y1)
    return xyz[selected]
