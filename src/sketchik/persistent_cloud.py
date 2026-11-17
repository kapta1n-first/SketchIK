"""Retain observed shapes separately from their current coordinate hypotheses.

This is display context, not segmentation, hidden-surface completion or a new
grounding tool. Unlocalized old geometry remains archived and visibly historical;
it never becomes a selectable current sample merely by appearing in a picture.
"""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np


def capture_layers(artifacts):
    """Freeze only public geometry and already supported poses on the owner thread."""
    from .carry_reference import active_reference
    from .rigid_binding import valid_binding
    hub = artifacts.hub
    catalog = getattr(hub, 'scene_catalog', None) or {}
    dirty = set(artifacts.live_cloud_files.dynamic)
    layers = []
    path = catalog.get('point_cloud_file')
    if path:
        dirty_groups = sorted({int(key[1:]) for item in artifacts.handoff['objects']
            if item['object_id'] in dirty for key in item.get('component_ids', [])
            if key.startswith('G') and key[1:].isdigit()})
        layers.append({'kind': 'initial_table', 'source_cloud_file': path,
            'sources': deepcopy(catalog.get('table_sources', [])),
            'group_file': catalog.get('component_labels_file'), 'dirty_groups': dirty_groups,
            'dirty_bounds': [artifacts.live_cloud_files.original_bounds[target].tolist()
                for target in dirty if target in artifacts.live_cloud_files.original_bounds]})
    seen = {str(Path(path).resolve())} if path else set()
    for target, keys in artifacts.original_shape_views.items():
        for key in keys:
            view = artifacts.views[key]
            source = str(Path(view['cloud_file']).resolve())
            if source in seen:
                continue
            seen.add(source)
            layers.append({'kind': 'initial_object_shape', 'object_id': target,
                'source_cloud_file': source, 'historical_unresolved': target in dirty,
                'sources': deepcopy(view.get('sources', [])),
                **({'motion_tracking_run': str(hub.run)} if target in dirty and
                    artifacts.live_cloud_files.shape_motion(target) != 'rigid' else {})})
    reference = active_reference(hub)
    if reference is not None and reference.get('source_cloud_file'):
        from scipy.spatial.transform import Rotation
        state = hub.robot.state()
        layers.append({'kind': 'supported_grasp_prediction', 'object_id': reference['object_id'],
            'source_cloud_file': reference['source_cloud_file'],
            'point_ids': np.asarray(reference['point_ids']).copy(),
            'current_world_xyz_m': Rotation.from_quat(state['quaternion_xyzw']).apply(
                reference['local_xyz_m'])+state['position'],
            'reference_id': reference['reference_id'],
            'relation_revision': reference['relation_revision'], 'holding_certified': False})
    registered = set()
    for binding in artifacts.rigid_view_bindings.values():
        target = binding['owner']
        if (reference is not None and reference['object_id'] == target) or not valid_binding(hub, binding, target):
            continue
        key = (binding['original_cloud_file'], binding['pose_id'])
        if key in registered:
            continue
        registered.add(key)
        # Union exact memberships across projections; never transform the full
        # spatial window just because some of its points belong to this object.
        same = [b for b in artifacts.rigid_view_bindings.values()
            if (b['original_cloud_file'], b['pose_id']) == key and valid_binding(hub, b, target)]
        ids = []
        for b in same:
            with np.load(b['members_file']) as data:
                ids.append(data['original_point_ids'])
        layers.append({'kind': 'supported_registered_pose', 'object_id': target,
            'source_cloud_file': binding['original_cloud_file'],
            'point_ids': np.unique(np.concatenate(ids)), 'transform': deepcopy(binding['transform']),
            'pose_id': binding['pose_id'], 'holding_certified': False})
    return layers


def depth_contradictions(points, frames, tolerance=.012):
    """Old surfaces now in measured free space; occlusion is not disappearance."""
    contradicted = np.zeros(len(points), bool)
    supported = np.zeros(len(points), bool)
    for frame in frames:
        transform = np.asarray(frame['world_from_camera'])
        camera = (points-transform[:3, 3])@transform[:3, :3]
        z = camera[:, 2]
        projected = camera@np.asarray(frame['K']).T
        valid = z > 1e-6
        pixels = np.zeros((len(points), 2), np.int64)
        pixels[valid] = np.rint(projected[valid, :2]/projected[valid, 2, None]).astype(np.int64)
        depth = np.asarray(frame['depth_m']).squeeze()
        valid &= ((pixels[:, 0] >= 0) & (pixels[:, 0] < depth.shape[1]) &
                  (pixels[:, 1] >= 0) & (pixels[:, 1] < depth.shape[0]))
        ids = np.flatnonzero(valid)
        measured = depth[pixels[ids, 1], pixels[ids, 0]]
        finite = np.isfinite(measured) & (measured > 0)
        supported[ids] |= finite & (np.abs(measured-z[ids]) <= tolerance)
        contradicted[ids] |= finite & (measured-z[ids] > tolerance)
    # A consistent second view prevents one grazing/noisy ray from retiring it.
    return contradicted & ~supported


def build_context(layers, frames, folder, *, max_display_points=300_000):
    """Keep every source row; only this display derivative is sampled.

    The small per-layer manifest points to the complete immutable source. A
    supported pose preserves exact original members and all their transformed
    coordinates, including currently occluded members. Dynamic nonrigid scopes
    may estimate independent local surface motions; never one whole-object pose.
    """
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    xyz_parts, rgb_parts, records = [], [], []
    total = 0
    tracking, transfers = prepare_surface_motion(layers, frames)
    for number, layer in enumerate(layers):
        with np.load(layer['source_cloud_file']) as data:
            ids = np.asarray(layer.get('point_ids', np.arange(len(data['world_xyz_m']))), np.int64)
            if len(np.unique(ids)) != len(ids) or np.any(ids < 0) or np.any(ids >= len(data['world_xyz_m'])):
                raise ValueError('Retained shape membership must reference unique original rows')
            original = data['world_xyz_m'][ids]
            colors = data['rgb'][ids].copy()
            codes = sample_codes(data, layer.get('sources', []), transfers['source_keys'], ids)
        points = np.asarray(layer.get('current_world_xyz_m', original)).copy()
        if layer.get('transform'):
            pose = layer['transform']
            points = original@np.asarray(pose['rotation']).T+pose['translation_m']
        if points.shape != original.shape or not np.isfinite(points).all():
            raise ValueError('Retained world samples must align with original source members')
        historical = np.full(len(points), bool(layer.get('historical_unresolved')), bool)
        if layer['kind'] == 'initial_table':
            if layer.get('group_file'):
                with np.load(layer['group_file']) as data:
                    labels = data['point_group_ids']
                if len(labels) != len(points):
                    raise ValueError('Initial table labels must align with the immutable baseline')
                historical |= np.isin(labels, layer['dirty_groups'])
            else:
                for low, high in layer.get('dirty_bounds', []):
                    historical |= np.all((points >= low) & (points <= high), axis=1)
        tracked = np.zeros(len(points), bool)
        if codes is not None and len(transfers['codes']):
            rows = np.searchsorted(transfers['codes'], codes)
            valid = rows < len(transfers['codes'])
            valid[valid] &= transfers['codes'][rows[valid]] == codes[valid]
            # Only identical original RGB-D samples may share an update; spatial
            # neighbours, other objects and different source times never do.
            points[valid] = transfers['positions'][rows[valid]]
            historical[valid] = False
            tracked |= valid
        predicted = layer['kind'].startswith('supported_')
        if not predicted:
            historical |= depth_contradictions(points, frames)
        # Pale, neutral history is not a statement that those old XYZ are current.
        colors[historical] = 195+np.rint(colors[historical].mean(axis=1, keepdims=True)*.14).astype(np.uint8)
        sample = np.linspace(0, len(points)-1, min(len(points), max_display_points), dtype=np.int64)
        xyz_parts.append(points[sample]); rgb_parts.append(colors[sample])
        record = {k: deepcopy(v) for k, v in layer.items()
            if k not in ('point_ids', 'current_world_xyz_m', 'motion_tracking_run')}
        record.update(source_cloud_file=str(Path(layer['source_cloud_file']).resolve()),
            original_point_count=len(ids), display_point_count=len(sample),
            historical_unresolved_points=int(historical.sum()),
            surface_motion_supported_points=int(tracked.sum()), current_selection_allowed=False)
        if number in tracking:
            record['surface_motion'] = tracking[number]
        samples = folder/f'layer_{number:02d}_rows.npz'
        arrays = {'original_point_ids': ids, 'display_sample_rows': sample,
            'historical_unresolved': historical}
        if predicted or tracked.any():
            arrays['current_world_xyz_m'] = points
        if tracked.any():
            arrays['surface_motion_supported'] = tracked
        np.savez(samples, **arrays)
        record['row_identity_file'] = str(samples.resolve())
        records.append(record); total += len(ids)
    report = {'layers': records, 'original_retained_samples': total,
        'display_samples': sum(len(p) for p in xyz_parts), 'source_points_deleted': 0,
        'selection': 'Retained layers are context only. Select only actual current RGB-D samples '
            'in this projection, or an explicitly supported original rigid template.',
        'scope': 'RGB includes unchanged-pose assumptions and supported rigid predictions. '
            'Pale neutral geometry is unresolved historical shape, not a current surface. '
            'Articulated scopes use supported local surface motions only, never a whole-object rigid transform. '
            'Unmatched regions and hidden joint relationships remain unresolved.'}
    path = folder/'retained_layers.json'; path.write_text(json.dumps(report, indent=2))
    return {'world_xyz_m': np.concatenate(xyz_parts) if xyz_parts else np.empty((0, 3)),
        'rgb': np.concatenate(rgb_parts) if rgb_parts else np.empty((0, 3), np.uint8),
        'report': report, 'manifest_file': str(path.resolve())}


def sample_codes(data, sources, source_keys, ids):
    """Exact native-frame/pixel identity, shared across immutable saved tables."""
    source_field = 'source_index' if 'source_index' in data else 'source_id'
    if not sources or source_field not in data or 'pixel_xy' not in data:
        return None
    source_ids = np.asarray(data[source_field][ids], int)
    if len(source_ids) and (source_ids.min() < 0 or source_ids.max() >= len(sources)):
        raise ValueError('Retained source indices do not align with calibrated observations')
    names = [(s['observation_id'], s['camera']) for s in sources]
    indices = np.asarray([source_keys[name] for name in names], np.uint64)
    pixels = np.asarray(data['pixel_xy'][ids], np.uint64)
    if np.any(pixels >= 1024):
        raise ValueError('Native sample identity requires the original 1024-pixel camera bounds')
    return (indices[source_ids] << np.uint64(20)) + pixels[:, 1]*np.uint64(1024)+pixels[:, 0]


def prepare_surface_motion(layers, frames):
    """Background-only estimates and exact-provenance reuse for duplicate layers."""
    from .surface_motion import track_saved_shape
    source_keys = {name: index for index, name in enumerate(sorted({
        (s['observation_id'], s['camera']) for layer in layers for s in layer.get('sources', [])}))}
    reports, codes, positions = {}, [], []
    for number, layer in enumerate(layers):
        if not layer.get('motion_tracking_run') or not layer.get('sources'):
            continue
        with np.load(layer['source_cloud_file']) as data:
            fields = ('world_xyz_m', 'rgb', 'pixel_xy', 'source_index')
            if not all(field in data for field in fields):
                continue
            old = {field: data[field].copy() for field in fields}
        updated, supported, report = track_saved_shape(old, layer['sources'], frames,
            Path(layer['motion_tracking_run']))
        reports[number] = report
        if supported.any():
            ids = np.flatnonzero(supported)
            codes.append(sample_codes(old, layer['sources'], source_keys, ids))
            positions.append(updated[ids])
    values = {'source_keys': source_keys, 'codes': np.empty(0, np.uint64), 'positions': np.empty((0, 3))}
    if codes:
        keys, xyz = np.concatenate(codes), np.concatenate(positions)
        unique, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
        discrepancy = np.zeros(len(unique))
        np.maximum.at(discrepancy, inverse, np.linalg.norm(xyz-xyz[first[inverse]], axis=1))
        unambiguous = discrepancy <= .003
        values.update(codes=unique[unambiguous], positions=xyz[first[unambiguous]])
    return reports, values
