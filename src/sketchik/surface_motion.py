"""Local rigid motion of retained surface samples, never an articulated model.

RGB-D feature consensus supplies several motions. Only small spatial cells with
local, distinct, consistent correspondences and current native-depth support
receive a transform. No whole-object/part identity or hidden joint is inferred.
The result is display context; it does not enable historical point selection.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.spatial import cKDTree

from .object_registration import correspondence_seeds, depth_checks, rigid_fit


def motion_models(source, destination, *, max_models=8):
    """Peel independent feature consensuses, retaining distinct metric pairs."""
    source, destination = np.asarray(source, float), np.asarray(destination, float)
    if source.shape != destination.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError('Surface motion requires aligned Nx3 feature pairs')
    if not np.isfinite(source).all() or not np.isfinite(destination).all():
        raise ValueError('Surface motion feature pairs must be finite')
    # Repeated observations of the same feature do not add independent support.
    _, unique = np.unique(np.rint(np.c_[source, destination]/.002).astype(np.int64),
        axis=0, return_index=True)
    source, destination = source[unique], destination[unique]
    remaining = np.arange(len(source))
    models = []
    for _ in range(max_models):
        if len(remaining) < 8:
            break
        seeds = correspondence_seeds(source[remaining], destination[remaining])
        if not seeds:
            break
        rotation, translation = max(seeds, key=lambda pose: int((np.linalg.norm(
            source[remaining]@pose[0].T+pose[1]-destination[remaining], axis=1) < .006).sum()))
        inside = np.linalg.norm(source@rotation.T+translation-destination, axis=1) < .006
        selected = np.intersect1d(np.flatnonzero(inside), remaining)
        if len(selected) < 8:
            break
        rotation, translation = rigid_fit(source[selected], destination[selected])
        if not any(np.percentile(np.linalg.norm(source@rotation.T+translation-
                (source@r.T+t), axis=1), 90) < .006 for r, t in models):
            models.append((rotation, translation))
        remaining = np.setdiff1d(remaining, selected)
    return source, destination, models


def track_samples(points, source, destination, frames, *, cell_size=.025,
                  neighborhood_radius=.055, budget_s=3.):
    """Preserve every row, and move only independently supported local cells.

    Cells crossing observed motion boundaries are rejected. Unseen or ambiguous
    regions keep their original coordinates as unresolved history. Even accepted
    cells are a local motion estimate, not a complete physical rigid part.
    """
    started = time.perf_counter()
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('Retained surface points must be finite Nx3')
    if not frames or any(frame['step'] != frames[0]['step'] for frame in frames):
        raise ValueError('Surface motion requires one current physical snapshot')
    current = points.astype(float, copy=True)
    supported = np.zeros(len(points), bool)
    report = {'status': 'unavailable', 'source_step': int(frames[0]['step']),
        'original_samples': len(points), 'supported_samples': 0, 'moved_samples': 0,
        'original_points_deleted': 0, 'cell_size_m': cell_size,
        'neighborhood_radius_m': neighborhood_radius, 'cells': [],
        'semantic_identity_verified': False, 'selectable': False,
        'scope': 'Local observed feature/depth motion only. No whole articulated object, '
            'hidden joint, attachment or collision certification; unresolved rows remain historical.'}
    source, destination, models = motion_models(source, destination)
    report.update(distinct_feature_pairs=len(source), motion_models=len(models))
    if not models:
        report.update(reason='insufficient_distinct_motion_features', wall_s=time.perf_counter()-started)
        return current, supported, report
    feature_tree = cKDTree(source)
    keys, inverse = np.unique(np.floor(points/cell_size).astype(np.int64), axis=0, return_inverse=True)
    order = np.argsort(inverse, kind='stable')
    offsets = np.r_[0, np.cumsum(np.bincount(inverse, minlength=len(keys)))]
    centers = (keys+.5)*cell_size
    neighbours = feature_tree.query_ball_point(centers, neighborhood_radius, workers=1)
    complete = True
    for cell, nearby in enumerate(neighbours):
        if time.perf_counter()-started > budget_s:
            complete = False
            break
        if len(nearby) < 8:
            continue
        ids = order[offsets[cell]:offsets[cell+1]]
        if len(ids) < 12:
            continue
        a, b = source[nearby], destination[nearby]
        candidates = []
        for rotation, translation in models:
            inside = np.linalg.norm(a@rotation.T+translation-b, axis=1) < .006
            if int(inside.sum()) < 8 or float(inside.mean()) < .8:
                continue
            local = a[inside]
            _, spread, axes = np.linalg.svd(local-local.mean(axis=0), full_matrices=False)
            if spread[1]/np.sqrt(len(local)) < .003:
                continue
            # Interpolate in the feature-supported surface footprint; never
            # propagate a front-panel motion to an unseen back wall.
            basis = axes[:2].T
            projected = (points[ids]-local.mean(axis=0))@basis
            feature_projected = (local-local.mean(axis=0))@basis
            if not np.all((projected >= feature_projected.min(axis=0)-.008) &
                          (projected <= feature_projected.max(axis=0)+.008)):
                continue
            candidates.append((rotation, translation, int(inside.sum())))
        if len(candidates) != 1:
            continue
        rotation, translation, count = candidates[0]
        predicted = points[ids]@rotation.T+translation
        # Check all cell surfaces; sparse features must not drag an adjacent
        # surface or an independently moving region through measured free space.
        evidence = depth_checks(predicted, frames)
        if not any(e['supported_pixels'] >= 8 and e['support_fraction_of_testable'] >= .8
                for e in evidence):
            continue
        if any(e['known_depth_pixels'] >= 8 and e['contradiction_fraction'] > .1 for e in evidence):
            continue
        current[ids] = predicted
        supported[ids] = True
        moved = bool(np.max(np.linalg.norm(predicted-points[ids], axis=1)) > .006)
        report['cells'].append({'cell': keys[cell].tolist(), 'sample_count': len(ids),
            'feature_support': count, 'moved': moved,
            'transform': {'rotation': rotation.tolist(), 'translation_m': translation.tolist()},
            'native_depth': evidence})
    report.update(status='supported_partial' if supported.any() else 'unavailable',
        reason='supported_local_cells' if supported.any() else 'no_supported_local_cell',
        supported_samples=int(supported.sum()),
        moved_samples=int((supported & (np.linalg.norm(current-points, axis=1) > .006)).sum()),
        unresolved_samples=int((~supported).sum()), search_complete=complete,
        wall_s=time.perf_counter()-started)
    return current, supported, report


def track_saved_shape(old, sources, frames, run, *, budget_s=10.):
    """Read archived calibrated RGB-D only; no robot/model/GT access."""
    from .object_registration import feature_correspondences
    from .perception import saved_frame
    started = time.perf_counter()
    sample = np.asarray(old['world_xyz_m'])[::max(1, len(old['world_xyz_m'])//2000)]
    if (len(sample) < 8 or np.linalg.svd(sample-sample.mean(axis=0), compute_uv=False)[1]/
            np.sqrt(len(sample)) < .003):
        # A degenerate old template cannot constrain any local rotation. Avoid
        # image decoding/feature work, rather than inventing geometric support.
        return track_samples(old['world_xyz_m'], np.empty((0, 3)), np.empty((0, 3)), frames)
    source, destination = feature_correspondences(old, sources,
        lambda observation, camera: saved_frame(run, observation, camera), frames)
    flow_source, flow_destination = static_camera_pairs(old, sources, frames,
        lambda observation, camera: saved_frame(run, observation, camera))
    source = np.concatenate([source, flow_source])
    destination = np.concatenate([destination, flow_destination])
    feature_s = time.perf_counter()-started
    points, supported, report = track_samples(old['world_xyz_m'], source, destination, frames,
        budget_s=max(0., budget_s-feature_s)/2)
    report.update(feature_wall_s=feature_s, static_camera_flow_pairs=len(flow_source),
        wall_s=time.perf_counter()-started,
        agent_requests=0, robot_actions=0, object_gt_used=False)
    return points, supported, report



def static_camera_pairs(old, sources, frames, resolve_frame):
    """Forward/backward KLT for a physically unchanged calibrated camera only.

    The wrist is not assumed fixed. Metric depth is checked at both ends and
    the original sample identity is recovered through exact source membership.
    Tracks initialize hypotheses only; local consensus/depth checks still apply.
    """
    import cv2
    from .libero import back_project
    a, b = [], []
    for now in frames:
        available = []
        for source_id, source in enumerate(sources):
            if source['camera'] != now['camera']:
                continue
            ids = np.flatnonzero(old['source_index'] == source_id)
            if len(ids) >= 30:
                available.append((len(ids), source_id, ids))
        # Same calibrated camera, maximum retained pixel coverage. This is a
        # numerical initialization choice, not selection of an object candidate.
        for _, source_id, ids in sorted(available, key=lambda item: -item[0]):
            source = sources[source_id]
            before = resolve_frame(source['observation_id'], source['camera'])
            if (np.asarray(before['rgb']).shape != np.asarray(now['rgb']).shape or
                    not np.allclose(before['K'], now['K'], atol=1e-8, rtol=0) or
                    not np.allclose(before['world_from_camera'], now['world_from_camera'], atol=1e-8, rtol=0)):
                continue
            image_a = cv2.cvtColor(np.asarray(before['rgb']), cv2.COLOR_RGB2GRAY)
            image_b = cv2.cvtColor(np.asarray(now['rgb']), cv2.COLOR_RGB2GRAY)
            mask = np.zeros(image_a.shape, np.uint8)
            pixels = old['pixel_xy'][ids].astype(int)
            mask[pixels[:, 1], pixels[:, 0]] = 255
            features = cv2.goodFeaturesToTrack(image_a, maxCorners=1800, qualityLevel=.005,
                minDistance=5, mask=mask, blockSize=5)
            if features is None:
                break
            options = {'winSize': (25, 25), 'maxLevel': 4,
                'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01)}
            forward, good, error = cv2.calcOpticalFlowPyrLK(image_a, image_b, features, None, **options)
            backward, reverse, _ = cv2.calcOpticalFlowPyrLK(image_b, image_a, forward, None, **options)
            keep = ((good.ravel() > 0) & (reverse.ravel() > 0) & (error.ravel() < 20.) &
                (np.linalg.norm(backward-features, axis=2).ravel() <= 1.))
            tree = cKDTree(old['world_xyz_m'][ids])
            for p, q in zip(features[keep, 0], forward[keep, 0]):
                try:
                    origin = back_project(before, [int(round(v)) for v in p])
                    target = back_project(now, [int(round(v)) for v in q])
                except (ValueError, IndexError):
                    continue
                distance, row = tree.query(origin)
                if distance <= .004:
                    a.append(old['world_xyz_m'][ids[row]])
                    b.append(target)
            break
    return np.asarray(a, float).reshape(-1, 3), np.asarray(b, float).reshape(-1, 3)
