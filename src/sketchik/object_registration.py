"""Per-episode RGB-D rigid relocalization of an already named observed object.

SIFT depth correspondences/RANSAC and multi-start trimmed ICP are numerical
hypotheses, not semantic or grasp certification. Original samples never change.
Ambiguous, occluded or inconsistent fits remain unavailable for point selection.
"""
from __future__ import annotations

from itertools import permutations, product
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree


class DegenerateRigidFit(ValueError):
    """Observed correspondences cannot constrain a proper rigid rotation."""


def catalog_template(catalog, component_ids):
    """Use every group explicitly bound by A, never choose the largest group."""
    if not component_ids or any(key not in catalog['components'] for key in component_ids):
        return None
    groups = [catalog['components'][key].get('group_id') for key in component_ids]
    if any(value is None for value in groups):
        return None  # A coarse fallback rectangle has no isolated native point identity.
    with np.load(catalog['point_cloud_file']) as data:
        arrays = {key: data[key].copy() for key in ('world_xyz_m', 'rgb', 'pixel_xy')}
        arrays['source_index'] = data['source_index' if 'source_index' in data else 'source_id'].copy()
    with np.load(catalog['component_labels_file']) as data: labels = data['point_group_ids'].copy()
    if labels.shape != (len(arrays['world_xyz_m']),):
        raise ValueError('Initial identity groups do not align with original RGB-D points')
    chosen = np.isin(labels, groups)
    if chosen.sum() < 60: return None
    return {key: value[chosen] for key, value in arrays.items()}, catalog['table_sources']


def expand_named_template(template, view):
    """Keep whole measured fragments supported by the previously named groups.

    A group is linked using common measured world surfaces before manipulation;
    every point of that group is retained, including additional observed sides.
    This does not choose the largest/first group or fill unobserved surfaces.
    """
    if template is None or not view.get('point_group_ids_file'):
        return template
    baseline, _ = template
    with np.load(view['cloud_file']) as data:
        arrays = {key: data[key].copy() for key in ('world_xyz_m', 'rgb', 'pixel_xy', 'source_index')}
    with np.load(view['point_group_ids_file']) as data: labels = data['point_group_ids'].copy()
    if labels.shape != (len(arrays['world_xyz_m']),): raise ValueError('Template groups do not align')
    tree = cKDTree(baseline['world_xyz_m'])
    keep = np.zeros(len(labels), bool)
    for group in np.unique(labels[labels > 0]):
        mask = labels == group
        sample = arrays['world_xyz_m'][mask][::max(1, int(mask.sum())//1000)]
        distance, _ = tree.query(sample, workers=1)
        if (distance <= .005).mean() >= .15:
            keep |= mask
    if keep.sum() < 60: return template
    return {key: value[keep] for key, value in arrays.items()}, view['sources']


def rigid_fit(source, target):
    """Proper SE(3), without scale/reflection; inputs must constrain rotation."""
    source, target = np.asarray(source, float), np.asarray(target, float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or len(source) < 3:
        raise ValueError('Rigid correspondences require aligned Nx3 points')
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError('Nonfinite rigid correspondences')
    a, b = source.mean(axis=0), target.mean(axis=0)
    u, s, vt = np.linalg.svd((source-a).T@(target-b))
    if s[1] < 1e-10:
        raise DegenerateRigidFit('Collinear correspondences do not determine rotation')
    rotation = vt.T@np.diag([1., 1., np.linalg.det(vt.T@u.T)])@u.T
    return rotation, b-rotation@a


def voxel_samples(xyz, rgb, size=.003, limit=2400):
    """Solver-only sampling. Keep an original point/color, never average RGB."""
    _, indices = np.unique(np.floor(np.asarray(xyz)/size).astype(np.int64), axis=0, return_index=True)
    indices.sort()
    if len(indices) > limit:
        indices = indices[np.linspace(0, len(indices)-1, limit, dtype=int)]
    return np.asarray(xyz, float)[indices], np.asarray(rgb)[indices], indices


def _pca(points):
    _, axes = np.linalg.eigh(np.cov(points.T))
    axes = axes[:, ::-1]
    axes[:, -1] *= np.linalg.det(axes)
    return axes


def pca_seeds(old, current):
    a, b = _pca(old), _pca(current)
    for order in permutations(range(3)):
        for signs in product((-1., 1.), repeat=3):
            middle = np.eye(3)[:, order]@np.diag(signs)
            if np.linalg.det(middle) < .5:
                continue
            rotation = b@middle@a.T
            yield rotation, np.median(current, axis=0)-rotation@np.median(old, axis=0)


def refine(old, current, rotation, translation, iterations=22):
    """Fit current visible samples to the complete old sample set (occlusion)."""
    tree = cKDTree(old)
    for index in range(iterations):
        back = (current-translation)@rotation
        distances, nearest = tree.query(back, workers=1)
        cutoff = min(.025 if index < 5 else .012, max(.004, float(np.quantile(distances, .8))))
        keep = distances <= cutoff
        if keep.sum() < 12:
            break
        try:
            updated_r, updated_t = rigid_fit(old[nearest[keep]], current[keep])
        except ValueError:
            break
        delta = np.linalg.norm(updated_t-translation)+np.linalg.norm(updated_r-rotation)*.05
        rotation, translation = updated_r, updated_t
        if delta < 2e-6:
            break
    return rotation, translation


def depth_checks(points, frames):
    """Native-depth support/free-space evidence; robot occlusion is not failure.

    An observed surface nearer than a predicted surface can occlude it. A
    predicted surface significantly in front of actual depth contradicts a fit.
    Missing depth/out-of-frame pixels give no support, never positive evidence.
    """
    reports = []
    for frame in frames:
        transform = np.asarray(frame['world_from_camera'])
        camera = (points-transform[:3, 3])@transform[:3, :3]
        front = camera[:, 2] > 1e-5
        pixels = camera@np.asarray(frame['K']).T
        uv = np.rint(pixels[:, :2]/np.maximum(pixels[:, 2:3], 1e-9)).astype(int)
        depth = np.asarray(frame['depth_m']).squeeze()
        h, w = depth.shape
        valid = front & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        ids = np.flatnonzero(valid)
        if len(ids):
            # Nearest predicted point per native pixel, not hidden back points.
            order = np.lexsort((camera[ids, 2], uv[ids, 1]*w+uv[ids, 0]))
            ids = ids[order]
            key = uv[ids, 1]*w+uv[ids, 0]
            ids = ids[np.r_[True, key[1:] != key[:-1]]]
        measured = depth[uv[ids, 1], uv[ids, 0]]
        known = np.isfinite(measured) & (measured > 0)
        residual = camera[ids[known], 2]-measured[known]
        support = int((np.abs(residual) <= .008).sum())
        contradiction = int((residual < -.012).sum())
        testable = int((residual <= .008).sum())  # exclude surfaces hidden by nearer geometry
        reports.append({'camera': frame['camera'], 'source_step': int(frame['step']),
            'known_depth_pixels': len(residual), 'supported_pixels': support,
            'free_space_contradictions': contradiction,
            'support_fraction_of_testable': support/max(testable, 1),
            'contradiction_fraction': contradiction/max(testable, 1)})
    return reports


def fit_evidence(old, rgb, current, current_rgb, rotation, translation, frames=()):
    predicted = old@rotation.T+translation
    d, nearest = cKDTree(predicted).query(current, workers=1)
    matches = d <= .006
    color = np.linalg.norm(np.asarray(current_rgb, float)-np.asarray(rgb, float)[nearest], axis=1)
    old_dist, _ = cKDTree(current).query(predicted, workers=1)
    good = matches & (color < 85.)
    coverage = float(matches.mean())
    matched_color = float(np.median(color[matches])) if matches.any() else None
    rmse = float(np.sqrt(np.mean(np.minimum(d, .02)**2)))
    reports = depth_checks(predicted, frames)
    reasons = []
    if coverage < .75: reasons.append('insufficient_current_surface_overlap')
    if float((old_dist <= .008).mean()) < .25: reasons.append('too_little_old_shape_observed')
    if rmse > .005: reasons.append('surface_residual_too_large')
    if matched_color is None or matched_color > 65. or float(good.mean()) < .55:
        reasons.append('appearance_not_supported')
    if reports:
        if not any(item['supported_pixels'] >= 30 and item['support_fraction_of_testable'] >= .65 for item in reports):
            reasons.append('no_native_depth_support')
        if any(item['known_depth_pixels'] >= 30 and item['contradiction_fraction'] > .12 for item in reports):
            reasons.append('native_free_space_contradiction')
    return {'rotation': rotation.tolist(), 'translation_m': translation.tolist(),
        'current_overlap_fraction': coverage, 'old_surface_overlap_fraction': float((old_dist <= .008).mean()),
        'clipped_rmse_m': rmse, 'median_rgb_distance': matched_color,
        'color_supported_fraction': float(good.mean()), 'native_depth': reports,
        'score': coverage-rmse*20.-min(255. if matched_color is None else matched_color, 255.)/1000.,
        'rejection_reasons': reasons}


def feature_correspondences(old, sources, resolve_frame, current_frames):
    """Match only original template pixels to current dual RGB, with metric depth."""
    if not sources:
        return np.empty((0, 3)), np.empty((0, 3))
    import cv2
    from .libero import back_project
    detector = cv2.SIFT_create(nfeatures=1800)
    current = []
    for frame in current_frames:
        keypoints, descriptors = detector.detectAndCompute(np.asarray(frame['rgb']), None)
        current.append((frame, keypoints, descriptors))
    pairs_old, pairs_new = [], []
    old_xyz = np.asarray(old['world_xyz_m'], float)
    captured_xyz = np.asarray(old.get('original_world_xyz_m', old_xyz), float)
    if captured_xyz.shape != old_xyz.shape or not np.isfinite(captured_xyz).all():
        raise ValueError('Native capture coordinates must align with the retained point samples')
    for index in np.unique(old['source_index']):
        source = sources[int(index)]
        frame = resolve_frame(source['observation_id'], source['camera'])
        mask = np.zeros(np.asarray(frame['depth_m']).squeeze().shape, np.uint8)
        uv = old['pixel_xy'][old['source_index'] == index].astype(int)
        mask[uv[:, 1], uv[:, 0]] = 255
        # Do not expand masks onto adjacent background. Dense source pixels
        # constrain feature centres; their world point must also fit template.
        keys, descriptors = detector.detectAndCompute(np.asarray(frame['rgb']), mask)
        if descriptors is None or len(descriptors) < 2:
            continue
        source_ids = np.flatnonzero(old['source_index'] == index)
        old_tree = cKDTree(captured_xyz[source_ids])
        for now, now_keys, now_descriptors in current:
            if now_descriptors is None or len(now_descriptors) < 2:
                continue
            matches = cv2.BFMatcher().knnMatch(descriptors, now_descriptors, k=2)
            for values in matches:
                if len(values) != 2 or values[0].distance >= .7*values[1].distance:
                    continue
                match = values[0]
                try:
                    a = back_project(frame, [int(round(v)) for v in keys[match.queryIdx].pt])
                    b = back_project(now, [int(round(v)) for v in now_keys[match.trainIdx].pt])
                except (ValueError, IndexError):
                    continue
                distance, sample = old_tree.query(a)
                if distance <= .004:
                    # The native pixel belongs to its original capture. Map
                    # its sample ID to the template's current coordinate frame.
                    pairs_old.append(old_xyz[source_ids[sample]]); pairs_new.append(b)
    return np.asarray(pairs_old, float).reshape(-1, 3), np.asarray(pairs_new, float).reshape(-1, 3)


def correspondence_seeds(old, new):
    if len(old) < 8:
        return []
    rng = np.random.default_rng(0)
    scored = []
    for _ in range(300):
        ids = rng.choice(len(old), 3, replace=False)
        try: rotation, translation = rigid_fit(old[ids], new[ids])
        except ValueError: continue
        residual = np.linalg.norm(old@rotation.T+translation-new, axis=1)
        inside = residual < .006
        if inside.sum() < 8:
            continue
        # Distinct metric support; repeated matches at one rim do not constrain
        # the whole object's orientation.
        if np.linalg.svd(old[inside]-old[inside].mean(axis=0), compute_uv=False)[1]/np.sqrt(inside.sum()) < .003:
            continue
        try:
            rotation, translation = rigid_fit(old[inside], new[inside])
        except DegenerateRigidFit:
            # A valid minimal sample can still collect repeated/collinear
            # inliers. Skip that hypothesis, not the caller's recovery loop.
            continue
        if any(np.median(np.linalg.norm(old@rotation.T+translation-(old@r.T+t), axis=1)) < .005
                for _, r, t in scored):
            continue
        scored.append((int(inside.sum()), rotation, translation))
    return [(r, t) for _, r, t in sorted(scored, key=lambda item: -item[0])[:4]]


def color_correspondences(old_xyz, old_rgb, current_xyz, current_rgb):
    """Conservative mutual unique RGB matches for numerical initialization only.

    Uniform/repeated colors provide no correspondence. These seeds never skip
    geometric, appearance, native-depth, ambiguity or search-budget checks.
    """
    old_rgb, current_rgb = np.asarray(old_rgb,float), np.asarray(current_rgb,float)
    if len(old_rgb)<8 or len(current_rgb)<8:
        return np.empty((0,3)), np.empty((0,3))
    forward, nearest = cKDTree(current_rgb).query(old_rgb,k=2,workers=1)
    backward, reverse = cKDTree(old_rgb).query(current_rgb,k=2,workers=1)
    indices=np.arange(len(old_rgb))
    match=nearest[:,0]
    keep=(forward[:,0]<=25.)&(forward[:,0]<.8*forward[:,1])
    keep &= (reverse[match,0]==indices)&(backward[match,0]<=25.)
    keep &= backward[match,0]<.8*backward[match,1]
    return np.asarray(old_xyz)[keep],np.asarray(current_xyz)[match[keep]]


def register_object(old, sources, inspection, current_frames, resolve_frame, *, budget_s=8., search_xy_bounds=None,
                    other_observed_points=None, other_component_bounds=()):
    """No model/tool/robot calls. Failure/ambiguity returns no usable transform."""
    started = time.perf_counter()
    xyz, rgb, _ = voxel_samples(old['world_xyz_m'], old['rgb'])
    if len(xyz) < 60:
        return {'status': 'unavailable', 'reason': 'insufficient_old_shape', 'wall_s': time.perf_counter()-started}
    with np.load(inspection['cloud_file']) as data:
        observed, colors = data['world_xyz_m'].copy(), data['rgb'].copy()
    with np.load(inspection['point_group_ids_file']) as data: labels = data['point_group_ids'].copy()
    if labels.shape != (len(observed),): raise ValueError('Current registration group table does not align')
    groups = []
    identity_conflicts = []
    other_tree = cKDTree(other_observed_points) if other_observed_points is not None and len(other_observed_points) else None
    old_diameter = np.linalg.norm(np.ptp(xyz, axis=0))
    if search_xy_bounds is not None:
        search_xy_bounds = np.asarray(search_xy_bounds, float)
        if search_xy_bounds.shape != (2, 2) or not np.isfinite(search_xy_bounds).all() or np.any(search_xy_bounds[0] >= search_xy_bounds[1]):
            raise ValueError('Registration XY search bounds must be finite ordered 2x2')
    for group in np.unique(labels[labels > 0]):
        keep = labels == group
        if search_xy_bounds is not None:
            xy = observed[keep, :2]
            if not np.any(np.all((xy >= search_xy_bounds[0]) & (xy <= search_xy_bounds[1]), axis=1)):
                continue
        points, group_rgb, _ = voxel_samples(observed[keep], colors[keep], limit=1400)
        diameter = np.linalg.norm(np.ptp(points, axis=0))
        if len(points) >= 60 and .25*old_diameter <= diameter <= 1.65*old_diameter:
            # Identical objects need temporal identity evidence. A fitting
            # shape near another initial instance is not uniquely identified
            # merely because that instance shifted a few centimetres.
            centre = np.median(points, axis=0)
            target_distance = np.linalg.norm(centre-np.median(xyz, axis=0))
            conflicts = []
            for low, high in other_component_bounds:
                low, high = np.asarray(low), np.asarray(high)
                extent = np.linalg.norm(high-low)
                distance = np.linalg.norm(centre-(low+high)/2)
                if .5*old_diameter <= extent <= 2.*old_diameter and distance < .5*extent and distance < .5*target_distance:
                    conflicts.append(distance)
            if conflicts:
                identity_conflicts.append({'candidate_id': 'G'+str(group),
                    'reason': 'nearer_other_initial_instance_identity_is_ambiguous',
                    'other_component_distance_m': min(conflicts)})
                continue
            if other_tree is not None:
                overlap = float((other_tree.query(points, workers=1)[0] <= .006).mean())
                if overlap >= .65:
                    identity_conflicts.append({'candidate_id': 'G'+str(group),
                        'other_initial_component_overlap_fraction': overlap})
                    continue  # A different known surface still occupies this pose.
            groups.append((int(group), points, group_rgb))
    prepared_at = time.perf_counter()
    pairs_old, pairs_new = feature_correspondences(old, sources, resolve_frame, current_frames)
    feature_seeds = correspondence_seeds(pairs_old, pairs_new)
    features_at = time.perf_counter()
    hypotheses = []
    complete = True
    coarse_old, coarse_rgb, _ = voxel_samples(xyz, rgb, size=.006, limit=420)
    queue = []
    color_initializations = []
    for group, points, group_rgb in groups:
        color_seeds=[]
        if not feature_seeds:
            a,b=color_correspondences(xyz,rgb,points,group_rgb)
            color_seeds=correspondence_seeds(a,b)
            color_initializations.append({'candidate_id':'G'+str(group),
                'unique_mutual_rgb_pairs':len(a),'seeds':len(color_seeds)})
        coarse_current, coarse_colors, _ = voxel_samples(points, group_rgb, size=.006, limit=300)
        coarse = []
        for rotation, translation in [*feature_seeds, *color_seeds, *pca_seeds(coarse_old, coarse_current)]:
            if time.perf_counter()-started > budget_s:
                complete = False; break
            rotation, translation = refine(coarse_old, coarse_current, rotation, translation, iterations=10)
            evidence = fit_evidence(coarse_old, coarse_rgb, coarse_current, coarse_colors, rotation, translation)
            coarse.append((evidence['score'], rotation, translation))
        if not complete: break
        coarse.sort(key=lambda item: -item[0])
        for score, rotation, translation in coarse[:6]:
            # Retain at least two poses, plus near-ties for symmetric objects.
            if score < coarse[min(1, len(coarse)-1)][0]-.06: continue
            queue.append((score, group, points, group_rgb, rotation, translation))
    coarse_at = time.perf_counter()
    for _, group, points, group_rgb, rotation, translation in sorted(queue, key=lambda item: -item[0]):
        if time.perf_counter()-started > budget_s:
            complete = False; break
        rotation, translation = refine(xyz, points, rotation, translation, iterations=18)
        evidence = fit_evidence(xyz, rgb, points, group_rgb, rotation, translation)
        evidence['candidate_id'] = 'G'+str(group)
        hypotheses.append(evidence)
    for item in hypotheses:
        if item['rejection_reasons']: continue
        predicted = xyz@np.asarray(item['rotation']).T+item['translation_m']
        native = depth_checks(predicted, current_frames)
        item['native_depth'] = native
        if not any(v['supported_pixels'] >= 30 and v['support_fraction_of_testable'] >= .65 for v in native):
            item['rejection_reasons'].append('no_native_depth_support')
        if any(v['known_depth_pixels'] >= 30 and v['contradiction_fraction'] > .12 for v in native):
            item['rejection_reasons'].append('native_free_space_contradiction')
    viable = sorted([item for item in hypotheses if not item['rejection_reasons']], key=lambda value: -value['score'])
    best = viable[0] if viable else None
    distinct = []
    if best:
        reference = xyz@np.asarray(best['rotation']).T+best['translation_m']
        for item in viable[1:]:
            moved = xyz@np.asarray(item['rotation']).T+item['translation_m']
            if np.percentile(np.linalg.norm(moved-reference, axis=1), 90) > .008 and item['score'] >= best['score']-.06:
                distinct.append(item)
    accepted = bool(best and complete and not distinct)
    return {'status': 'accepted' if accepted else 'unavailable',
        'reason': ('supported_unique_fit' if accepted else 'search_budget_exhausted' if not complete else
            'ambiguous_pose_or_identity' if distinct else 'native_depth_inconsistent' if best else 'no_supported_fit'),
        'transform': {'rotation': best['rotation'], 'translation_m': best['translation_m']} if accepted else None,
        'best_hypothesis': best, 'competing_supported_poses': distinct[:3],
        'hypotheses': sorted(hypotheses, key=lambda item: -item['score'])[:6],
        'feature_correspondences': len(pairs_old), 'feature_initializations': len(feature_seeds),
        'current_groups_considered': len(groups), 'search_complete': complete,
        'color_initializations_without_image_features':color_initializations,
        'excluded_identity_conflicts': identity_conflicts,
        'search_xy_bounds_m': None if search_xy_bounds is None else search_xy_bounds.tolist(),
        'search_scope': 'Observed workspace XY plus hand/template extent when provided; objects outside this search remain unknown.',
        'timing_s': {'prepare': prepared_at-started, 'features': features_at-prepared_at,
            'coarse': coarse_at-features_at, 'refinement_and_validation': time.perf_counter()-coarse_at},
        'source_step': inspection['source_step'], 'wall_s': time.perf_counter()-started,
        'agent_requests': 0, 'robot_actions': 0, 'object_gt_used': False,
        'scope': 'Thresholded rigid pose estimate of earlier observed samples, not new observation, holding or semantic certification. Hidden surfaces remain unknown.'}


def save_registered_cloud(old, sources, registration, folder):
    """Preserve every old point/color/native source; store new XYZ separately."""
    if registration.get('status') != 'accepted' or not registration.get('transform'):
        raise ValueError('Unavailable registration cannot publish usable coordinates')
    transform = registration['transform']
    r, t = np.asarray(transform['rotation'], float), np.asarray(transform['translation_m'], float)
    if r.shape != (3, 3) or t.shape != (3,) or not np.isfinite(r).all() or not np.isfinite(t).all():
        raise ValueError('Invalid registration transform')
    if not np.allclose(r.T@r, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(r), 1., atol=1e-6):
        raise ValueError('Registration must be a proper rigid transform')
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=False)
    path = folder/'registered_old_points.npz'
    np.savez(path, **{**old, 'original_world_xyz_m': old.get('original_world_xyz_m', old['world_xyz_m']),
        'world_xyz_m': np.asarray(old['world_xyz_m'], float)@r.T+t})
    (folder/'registration.json').write_text(json.dumps({**registration, 'original_sources': sources}, indent=2))
    return path
