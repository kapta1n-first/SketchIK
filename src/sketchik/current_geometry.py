"""Current local RGB-D fragments, with explicit model-selected identities.

Only one physical snapshot is used. No historical world points, simulator
objects, hidden-surface completion, automatic object selection or motion.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .cloud_extents import summarize_components, principal_extents
from .scene_cloud import frame_points


def validate_rectangle(bounds, shape):
    if (not isinstance(bounds, (list, tuple)) or len(bounds) != 4 or
            any(type(value) is not int for value in bounds)):
        raise ValueError('bounds_xyxy requires four native-image integers')
    x0, y0, x1, y1 = bounds
    height, width = shape
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError('bounds_xyxy is outside the native image or empty')
    return x0, y0, x1, y1


def inspect_snapshot(inputs, source_step, folder, *, non_robot_mask=None,
                     table_height=None, render_images=True):
    """Generate every local geometric candidate, without binding an object.

    inputs contain saved frames and A's native-image rectangles. Non-robot
    filtering labels excluded samples G0 but does not remove the original
    point/source/UV table. Voxelization groups samples; it never replaces them.
    """
    started = time.perf_counter()
    if not inputs or len(inputs) > 2:
        raise ValueError('Supply one or two current native views')
    if len({value['frame']['camera'] for value in inputs}) != len(inputs):
        raise ValueError('Use at most one rectangle per current camera')
    sources, chunks = [], []
    for source_id, value in enumerate(inputs):
        frame = value['frame']
        if frame['step'] != source_step:
            raise ValueError('Local geometry cannot combine different physical snapshots')
        bounds = validate_rectangle(value['bounds_xyxy'], np.asarray(frame['depth_m']).squeeze().shape)
        xyz, rgb, uv = frame_points(frame)
        x0, y0, x1, y1 = bounds
        inside = ((uv[:, 0] >= x0) & (uv[:, 0] < x1) &
            (uv[:, 1] >= y0) & (uv[:, 1] < y1))
        chunks.append((xyz[inside], rgb[inside], uv[inside],
            np.full(int(inside.sum()), source_id, np.uint16)))
        sources.append({'observation_id': value['observation_id'], 'camera': frame['camera'],
            'source_step': int(source_step), 'bounds_xyxy': list(bounds)})
    xyz, rgb, uv, source_ids = [np.concatenate([chunk[index] for chunk in chunks]) for index in range(4)]
    keep = np.ones(len(xyz), bool)
    if non_robot_mask is not None and len(xyz):
        keep = np.asarray(non_robot_mask(xyz))
        if keep.shape != (len(xyz),) or keep.dtype != np.bool_:
            raise ValueError('Robot filter must return one boolean for each original point')
    # Separate the observed tabletop from touching objects without declaring
    # every surface below it irrelevant (for example a drawer or low handle).
    # All original samples, including this narrow band, remain in the NPZ.
    plane = np.zeros(len(xyz), bool)
    if table_height is not None:
        if not np.isfinite(table_height):
            raise ValueError('Observed table height must be finite')
        plane = np.abs(xyz[:, 2]-float(table_height)) <= .006
    grouped_mask = keep & ~plane
    grouped = summarize_components(xyz[grouped_mask], None, min_voxels=8)
    labels = np.zeros(len(xyz), np.int32)
    labels[grouped_mask] = grouped.pop('point_group_ids')
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=False)
    cloud = folder/'current_roi_points.npz'
    np.savez(cloud, world_xyz_m=xyz, rgb=rgb, pixel_xy=uv, source_index=source_ids)
    group_file = folder/'candidate_groups.npz'
    np.savez_compressed(group_file, point_group_ids=labels)
    images = []
    try:
        label_font = ImageFont.truetype('DejaVuSans.ttf', 22)
    except OSError:
        label_font = ImageFont.load_default()
    for source_id, value in enumerate(inputs):
        if not render_images:
            break
        image = np.asarray(value['frame']['rgb']).copy()
        visible = (source_ids == source_id) & (labels > 0)
        for group_id in np.unique(labels[visible]):
            selected = visible & (labels == group_id)
            pixels = uv[selected]
            color = np.asarray([(83*int(group_id)) % 200+30,
                (131*int(group_id)) % 200+30, (47*int(group_id)) % 200+30])
            image[pixels[:, 1], pixels[:, 0]] = np.rint(
                .7*image[pixels[:, 1], pixels[:, 0]]+.3*color).astype(np.uint8)
        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)
        draw.rectangle(value['bounds_xyxy'], outline='yellow', width=2)
        for group_id in np.unique(labels[visible]):
            pixels = uv[visible & (labels == group_id)]
            # A screen annotation position, never an object/grasp centre.
            position = tuple(np.median(pixels, axis=0).astype(int))
            text = f'G{group_id}'
            draw.rectangle(draw.textbbox(position, text, font=label_font), fill='white')
            draw.text(position, text, fill='black', font=label_font)
        path = folder/f'candidates_{source_id}.png'; image.save(path)
        images.append(str(path))
    result = {'source_step': int(source_step), 'sources': sources,
        'cloud_file': str(cloud), 'point_group_ids_file': str(group_file),
        'candidate_image_files': images, 'point_count': len(xyz), **grouped,
        'robot_filter': {'available': non_robot_mask is not None,
            'excluded_points': int((~keep).sum()),
            'scope': 'Known source-state robot geometry only. Near-robot object samples can also be excluded; inspect raw RGB. Not a collision check.'},
        'table_reference': {'observed_world_z_m': table_height, 'excluded_band_half_width_m': .006,
            'excluded_from_grouping_points': int((plane & keep).sum()),
            'below_table_surfaces_retained': True,
            'scope': 'Narrow measured tabletop band only; raw points stay saved. Not a semantic floor or cavity detector.'},
        'scope': 'Current native-image rectangles are search windows; G groups are visible geometric fragments, not object identities.',
        'semantic_identity_verified': False, 'complete_shape_known': False,
        'old_world_points_merged': False, 'sam_requests': 0, 'molmo_requests': 0,
        'view_eye_direction_world': (-np.asarray(inputs[0]['frame']['world_from_camera'])[:3, 2]).tolist(),
        'wall_s': time.perf_counter()-started}
    (folder/'inspection.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def validate_selection(inspection, candidate_ids, source_step):
    if inspection['source_step'] != source_step:
        raise ValueError('Candidate snapshot predates the current state; inspect current native views')
    known = {'G'+str(value['group_id']): value['group_id'] for value in inspection['components']}
    if (not candidate_ids or len(candidate_ids) != len(set(candidate_ids)) or
            any(value not in known for value in candidate_ids)):
        raise ValueError('Select explicitly displayed positive candidate IDs; G0, unknown IDs and empty selections cannot bind')
    return known


def selected_geometry(inspection, candidate_ids, source_step, folder, prefix):
    """Publish only A-selected visible samples; preserve exact native provenance."""
    from .cloud_selection import write_views
    known = validate_selection(inspection, candidate_ids, source_step)
    if not isinstance(prefix, str) or Path(prefix).name != prefix or prefix in ('', '.', '..') or '\\' in prefix:
        raise ValueError('Geometry prefix must be one safe filename component')
    with np.load(inspection['cloud_file']) as source:
        arrays = {key: source[key].copy() for key in ('world_xyz_m', 'rgb', 'pixel_xy', 'source_index')}
    with np.load(inspection['point_group_ids_file']) as source:
        labels = source['point_group_ids'].copy()
    if labels.shape != (len(arrays['world_xyz_m']),):
        raise ValueError('Candidate IDs and current point table do not align')
    selected = np.isin(labels, [known[value] for value in candidate_ids])
    if int(selected.sum()) < 30:
        raise ValueError('Too few selected visible points for an exact point image')
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=False)
    cloud = (folder/'selected_observed_points.npz').resolve()
    np.savez(cloud, **{key: value[selected] for key, value in arrays.items()})
    ids = labels[selected]
    # The existing carry association can read this observed group table after
    # an actual close; it remains a partial-shape hypothesis, not attachment.
    np.savez_compressed(folder/'extent_groups.npz', point_group_ids=ids, cloud_file=str(cloud))
    # Caller supplies a renderer direction from the calibrated camera, never
    # estimated object axes. A fixed world oblique is the explicit fallback.
    eye = inspection['view_eye_direction_world']
    views = write_views(cloud, deepcopy(inspection['sources']), folder, prefix, eye,
        role='Current selected observed object/part; identity is an A judgment',
        selectable=True, group_ids=ids)
    xyz = arrays['world_xyz_m'][selected].astype(float)
    low, high = xyz.min(axis=0), xyz.max(axis=0)
    robust = np.percentile(xyz, [1, 99], axis=0)
    selected_ids = {known[value] for value in candidate_ids}
    shape = {'source_steps': [int(source_step)], 'point_count': len(xyz),
        'bounds_world_m': [low.tolist(), high.tolist()], 'span_world_m': (high-low).tolist(),
        'robust_span_world_m': (robust[1]-robust[0]).tolist(), **principal_extents(xyz),
        'components': [deepcopy(value) for value in inspection['components'] if value['group_id'] in selected_ids],
        'candidate_ids': list(candidate_ids), 'complete_shape_known': False,
        'scope': 'Only selected currently visible samples; historical whole-object shape is stored separately. No completed centre, hidden shape or usable cavity certification.'}
    result = {'source_step': int(source_step), 'sources': deepcopy(inspection['sources']),
        'cloud_file': str(cloud), 'selectable_views': views, 'observed_shape': shape}
    (folder/'selected_geometry.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result
