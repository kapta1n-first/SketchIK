"""Camera geometry shared by full-table acquisition and focused observations."""
import numpy as np

from .scene_cloud import region_points


def unit(value):
    value = np.asarray(value, float)
    length = np.linalg.norm(value)
    if length < 1e-8 or not np.isfinite(value).all():
        raise ValueError('Finite nonzero direction required')
    return value / length


def describe_region(frame, bounds):
    points = region_points(frame, bounds)
    if len(points) < 64:
        raise ValueError('Insufficient positive observed depth in target region')
    # Trim extreme background/depth outliers, without claiming segmentation.
    camera = np.asarray(frame['world_from_camera'])
    depth = (points - camera[:3, 3]) @ camera[:3, 2]
    lo, hi = np.quantile(depth, [.05, .95])
    points = points[(depth >= lo) & (depth <= hi)]
    sample = points[np.linspace(0, len(points)-1, min(12000, len(points)), dtype=int)]
    anchor = np.median(sample, axis=0)
    _, values, axes = np.linalg.svd(sample-sample.mean(axis=0), full_matrices=False)
    normal = axes[-1]
    if np.dot(normal, camera[:3, 3]-anchor) < 0:
        normal = -normal
    return {'anchor_world_m': anchor.tolist(), 'principal_axes_world': axes.tolist(),
            'proposed_surface_normal_world': normal.tolist(), 'singular_values': values.tolist(),
            'observed_span_xyz_m': np.ptp(points, axis=0).tolist(),
            'observed_principal_spans_m': np.ptp((points-anchor) @ axes.T, axis=0).tolist(),
            'point_count': len(points), 'source_step': frame['step'],
            'source_camera_position_m': camera[:3, 3].tolist(),
            'context_min_world_m': points.min(axis=0).tolist(),
            'context_max_world_m': points.max(axis=0).tolist(),
            'scope': 'coarse RGB region with depth-tail trimming; may contain background',
            'anchor_is_object_center': False, 'complete_dimensions_known': False,
            'usable_cavity_dimensions_m': None}, sample


def look_at(eye, anchor, previous_x):
    z = unit(np.asarray(anchor)-eye)
    x = np.asarray(previous_x)-z*np.dot(previous_x, z)
    if np.linalg.norm(x) < 1e-6:
        basis = np.eye(3)[np.argmin(abs(z))]
        x = basis-z*np.dot(basis, z)
    x = unit(x)
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack([x, np.cross(z, x), z])
    transform[:3, 3] = eye
    return transform


def describe_opening(frame, bounds, rim_pixels):
    """A labels visible rim pixels; geometry computes their metric plane."""
    from .libero import back_project
    if not 3 <= len(rim_pixels) <= 6:
        raise ValueError('Opening needs 3..6 non-collinear visible rim pixels')
    x0,y0,x1,y1=bounds
    if any(not (x0<=x<x1 and y0<=y<y1) for x,y in rim_pixels):
        raise ValueError('Opening rim labels must belong to the selected source region')
    points = np.asarray([back_project(frame, pixel) for pixel in rim_pixels])
    anchor = points.mean(axis=0)
    _, values, axes = np.linalg.svd(points-anchor, full_matrices=False)
    if values[1] < .015:
        raise ValueError('Opening rim points are nearly collinear')
    normal = axes[-1]
    if np.dot(normal, frame['world_from_camera'][:3, 3]-anchor) < 0:
        normal = -normal
    residual = abs((points-anchor) @ normal)
    if residual.max() > .018:
        raise ValueError('Selected opening rim points do not support one plane')
    geometry, sample = describe_region(frame, bounds)
    geometry.update(anchor_world_m=anchor.tolist(), proposed_surface_normal_world=normal.tolist(),
        opening_frame=True, rim_pixels_xy=rim_pixels, rim_world_xyz_m=points.tolist(),
        rim_plane_max_residual_m=float(residual.max()), rim_semantics='A-labelled visible opening rim; no simulator object GT')
    deeper = (sample-anchor) @ normal < -.008
    geometry['observed_points_behind_rim'] = int(deeper.sum())
    geometry['camera_framing_scope']='A-labelled visible rim; internal surface identity still needs A review'
    return geometry, points


def diverse_view_order(candidates):
    """Try different camera bearings before spending IK on similar standoffs."""
    families={}
    for candidate in sorted(candidates,key=lambda item:item['score'],reverse=True):
        families.setdefault(candidate.get('bearing_family',0),[]).append(candidate)
    ordered=[]
    while families:
        for family in sorted(families,key=lambda key:families[key][0]['score'],reverse=True):
            ordered.append(families[family].pop(0))
            if not families[family]:del families[family]
    return ordered
