"""World-axis extents of connected observed point-cloud fragments.

Groups are geometric fragments, not object identities or complete shapes.
"""

import numpy as np
import json
from pathlib import Path
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def principal_extents(points):
    """Pair visible-sample PCA spans with world directions, never a grasp axis.

    Rows are ordered by sample variance. Signs are arbitrary; near-equal
    variances make the corresponding directions unstable. No object centre,
    semantic pose, completed shape or contact direction is returned.
    """
    xyz=np.asarray(points,dtype=np.float64)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or len(xyz)<3 or not np.isfinite(xyz).all():
        raise ValueError('At least three finite XYZ samples required for principal extents')
    centred=xyz-xyz.mean(axis=0)
    values,vectors=np.linalg.eigh(centred.T@centred/len(xyz))
    values=np.maximum(values[::-1],0.)
    axes=vectors[:,::-1].T.copy()
    for axis in axes:
        if axis[np.argmax(abs(axis))]<0:axis*=-1
    if np.linalg.det(axes)<0:axes[-1]*=-1
    gaps=(values[:-1]-values[1:])/np.maximum(values[:-1],np.finfo(float).tiny)
    return {'observed_principal_spans_m':np.ptp(centred@axes.T,axis=0).tolist(),
        'observed_principal_axes_world':axes.tolist(),
        'principal_axis_gap_ratios':[float(gaps[0]),float(min(gaps)),float(gaps[1])]}


def summarize_components(points, table_height=None, voxel_m=.004, link_m=.009,
                         min_voxels=32):
    """Group observed world points and measure every sufficiently large fragment.

    ``point_group_ids`` aligns with the input rows; zero marks excluded points and
    fragments below ``min_voxels``. The caller should store that array separately
    from the JSON-safe summary.
    """
    xyz = np.asarray(points, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if not (np.isfinite(voxel_m) and voxel_m > 0 and np.isfinite(link_m) and link_m > 0):
        raise ValueError("voxel_m and link_m must be positive finite distances")
    if not isinstance(min_voxels, (int, np.integer)) or min_voxels < 1:
        raise ValueError("min_voxels must be a positive integer")
    if table_height is not None and not np.isfinite(table_height):
        raise ValueError("table_height must be finite")

    ids = np.zeros(len(xyz), dtype=np.int32)
    valid = np.isfinite(xyz).all(axis=1)
    threshold = None if table_height is None else float(table_height) + .006
    if threshold is not None:
        valid &= xyz[:, 2] > threshold
    selected = xyz[valid]
    scope = ("observed points above table threshold" if threshold is not None
             else "observed points including possible table/background")
    summary = {"components": [], "scope": scope, "complete_shape_known": False,
               "table_threshold_world_z_m": threshold,
               "voxel_m": float(voxel_m), "link_m": float(link_m),
               "min_voxels": int(min_voxels)}
    if not len(selected):
        return {**summary, "point_group_ids": ids}

    cells = np.floor(selected / voxel_m).astype(np.int64)
    unique, inverse = np.unique(cells, axis=0, return_inverse=True)
    counts = np.bincount(inverse, minlength=len(unique))
    centers = np.column_stack([
        np.bincount(inverse, weights=selected[:, axis], minlength=len(unique)) / counts
        for axis in range(3)])
    pairs = cKDTree(centers).query_pairs(link_m, output_type="ndarray")
    rows = np.concatenate((pairs[:, 0], pairs[:, 1]))
    cols = np.concatenate((pairs[:, 1], pairs[:, 0]))
    graph = coo_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cols)),
                       shape=(len(unique), len(unique))).tocsr()
    _, labels = connected_components(graph, directed=False)
    voxel_counts = np.bincount(labels)
    # Component IDs follow the lexicographic order of their first voxel.
    kept = [label for label in np.unique(labels)
            if voxel_counts[label] >= min_voxels]
    kept.sort(key=lambda label: tuple(unique[np.flatnonzero(labels == label)[0]]))
    point_labels = labels[inverse]
    selected_ids = np.zeros(len(selected), dtype=np.int32)
    for group_id, label in enumerate(kept, 1):
        mask = point_labels == label
        original = selected[mask]
        low = original.min(axis=0)
        high = original.max(axis=0)
        robust = np.percentile(original, [1, 99], axis=0)
        selected_ids[mask] = group_id
        summary["components"].append({
            "group_id": group_id,
            "point_count": int(len(original)),
            "voxel_count": int(voxel_counts[label]),
            "bounds_world_m": [low.tolist(), high.tolist()],
            "span_world_m": (high - low).tolist(),
            "robust_span_world_m": (robust[1] - robust[0]).tolist(),
            **(principal_extents(original) if len(original)>=3 else {}),
        })
    ids[valid] = selected_ids
    return {**summary, "point_group_ids": ids}


def table_components(table, folder):
    """Label all observed tabletop fragments, including non-task objects."""
    cloud=Path(table['point_cloud_file'])
    stat=cloud.stat()
    identity={'schema_version':3,'cloud_file':str(cloud.resolve()),'size':stat.st_size,'mtime_ns':stat.st_mtime_ns,
        'table_height_m':table['observed_table_height_m'],
        'voxel_m':.006,'link_m':.015,'min_voxels':16}
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    metadata=folder/'display_components.json';labels_file=folder/'display_components.npz'
    with np.load(cloud) as data:
        points=data['world_xyz_m'];pixels=data['pixel_xy'];sources=data['source_index']
    summary=json.loads(metadata.read_text()) if metadata.exists() and labels_file.exists() else None
    if summary is None or summary.get('input')!=identity:
        grouped=summarize_components(points,identity['table_height_m'],voxel_m=identity['voxel_m'],
            link_m=identity['link_m'],min_voxels=identity['min_voxels'])
        labels=grouped.pop('point_group_ids')
        valid=labels>0
        cells,first=np.unique(np.floor(points[valid]/identity['voxel_m']).astype(np.int32),
            axis=0,return_index=True)
        np.savez_compressed(labels_file,point_group_ids=labels,
            voxel_cells=cells,voxel_group_ids=labels[valid][first])
        summary={'input':identity,**grouped}
        metadata.write_text(json.dumps(summary,indent=2))
    else:
        with np.load(labels_file) as data:labels=data['point_group_ids']
        if labels.shape!=(len(points),):
            raise ValueError('Table display component labels do not match the saved point table')
    return summary,labels,points,pixels,sources


def seeded_display_bounds(table, references, folder):
    """Grow a display window from actual source regions to connected fragments."""
    summary,labels,points,pixels,sources=table_components(table,folder)
    identity=summary['input'];folder=Path(folder)
    metadata=folder/'display_components.json';labels_file=folder/'display_components.npz'
    selected=set();seeds=[]
    for reference in references:
        index=next((i for i,source in enumerate(table['sources']) if
            (source['observation_id'],source['camera'])==
            (reference['observation_id'],reference['camera'])),None)
        if index is None:continue
        x0,y0,x1,y1=reference['bounds_xyxy']
        mask=(sources==index)&(pixels[:,0]>=x0)&(pixels[:,0]<x1)&(
            pixels[:,1]>=y0)&(pixels[:,1]<y1)&(labels>0)
        ids,counts=np.unique(labels[mask],return_counts=True)
        if not len(ids) or counts.max()<8:continue
        chosen=int(ids[np.argmax(counts)])
        selected.add(chosen)
        seeds.append({'observation_id':reference['observation_id'],'camera':reference['camera'],
            'bounds_xyxy':reference['bounds_xyxy'],'component_id':chosen,
            'seed_points':int(counts.max()),'other_fragment_seed_points':int(counts.sum()-counts.max())})
    components=[item for item in summary['components'] if item['group_id'] in selected]
    if not components:return None,{'method':'observed_connected_fragment_growth','available':False,'seeds':seeds}
    bounds=np.asarray([item['bounds_world_m'] for item in components])
    low=bounds[:,0].min(axis=0)-.015;high=bounds[:,1].max(axis=0)+.015
    low[2]=min(low[2],identity['table_height_m']-.025)
    return (low,high),{'method':'observed_connected_fragment_growth','available':True,
        'seeds':seeds,'component_ids':sorted(selected),'voxel_m':identity['voxel_m'],
        'link_m':identity['link_m'],'component_metadata':str(metadata),
        'component_labels':str(labels_file),'table_threshold_world_z_m':summary['table_threshold_world_z_m'],
        'tabletop_retained_in_local_window':True,
        'semantic_identity_verified':False,'complete_shape_known':False,
        'scope':'Full extents of seeded observed geometric fragments; nearby touching objects may be grouped.'}


def fragment_filter(crop):
    """Retain selected observed fragments and their local support, not neighbours."""
    if not crop.get('available'):return None
    with np.load(crop['component_labels']) as data:
        selected=np.isin(data['voxel_group_ids'],crop['component_ids'])
        centres=(data['voxel_cells'][selected]+.5)*crop['voxel_m']
    tree=cKDTree(centres)
    def keep(points):
        points=np.asarray(points)
        support=points[:,2]<=crop['table_threshold_world_z_m']
        ids=np.flatnonzero(~support)
        distances=tree.query(points[ids],distance_upper_bound=crop['link_m'])[0]
        support[ids]=np.isfinite(distances)
        return support
    return keep
