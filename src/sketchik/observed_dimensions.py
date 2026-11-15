"""Program-written partial measurements, without complete-shape claims or GT."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np


_CARRIED_KEYS=('object_dimensions_m','dimensions_frame','dimensions_kind',
    'dimension_source_step','dimension_checked_at_step','dimension_scope','dimensions_valid')
_INVALIDATION_KEYS=('dimension_invalidated_at_step','dimension_invalidation_reason')


def _save_dimensions(path, data):
    temporary=path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False))
    temporary.replace(path)


def write_measurement(run, identity, measurement):
    path=Path(run)/'dimensions.json'
    data=json.loads(path.read_text()) if path.exists() else {'unit':'m','objects':{}}
    record=data.setdefault('objects',{}).setdefault(identity,{})
    record.update(measurement)
    if measurement.get('dimensions_valid') is True:
        for key in _INVALIDATION_KEYS:record.pop(key,None)
    record['complete_dimensions_known']=False
    _save_dimensions(path,data)


def invalidate_carried_dimensions(run, identity, source_step, reason):
    """Retire this run's tool-relative visible extent after its grasp changes.

    Historical measurements remain auditable, but have no current numeric
    value. Repeated invalidation without a new measurement is a no-op.
    """
    path=Path(run)/'dimensions.json'
    if not path.exists():return False
    data=json.loads(path.read_text())
    record=data.get('objects',{}).get(identity)
    if not record or record.get('dimensions_frame')!='eef_tool' or \
            record.get('dimensions_kind')!='visible_surface_extent_estimate' or \
            record.get('dimensions_valid') is False or record.get('object_dimensions_m') is None:
        return False
    invalidation={'dimension_invalidated_at_step':int(source_step),
        'dimension_invalidation_reason':str(reason)}
    previous={key:record[key] for key in (*_CARRIED_KEYS,'complete_dimensions_known') if key in record}
    previous.update(invalidation)
    record.setdefault('carried_dimension_history',[]).append(previous)
    record['object_dimensions_m']=None
    record['dimensions_valid']=False
    record.update(invalidation)
    # These describe the retired measurement, not the current grasp.
    record.pop('dimension_source_step',None)
    record.pop('dimension_checked_at_step',None)
    record.pop('dimension_scope',None)
    _save_dimensions(path,data)
    return True


def carried_dimensions(points, source_step):
    points=np.asarray(points)
    if len(points)<25 or not np.isfinite(points).all():
        raise ValueError('Insufficient finite carried surface for a dimension estimate')
    return {'object_dimensions_m':np.ptp(points,axis=0).tolist(),
        'dimensions_frame':'eef_tool','dimensions_kind':'visible_surface_extent_estimate',
        'dimension_source_step':int(source_step),
        'dimension_checked_at_step':int(source_step),
        'dimension_scope':'Current observed carried component, including any visible handle; hidden shape unknown',
        'dimensions_valid':True,
        'complete_dimensions_known':False}


def cavity_dimensions(points, table_height, opening):
    """Measure visible horizontal support in A's opening region.

    Width/depth describe the observed support footprint. An upper plane gives
    a rough height when visible. These are not certified usable cavity sizes.
    """
    points=np.asarray(points)
    if len(points)<64:return None
    outward=np.asarray(opening['proposed_surface_normal_world'],float).copy()
    outward[2]=0
    if np.linalg.norm(outward)<.1:return None
    outward/=np.linalg.norm(outward)
    right=np.cross([0.,0.,1.],outward)
    basis=np.column_stack([right,-outward,[0.,0.,1.]])
    anchor=np.asarray(opening['anchor_world_m'])
    # Exclude the foreground beyond A's measured opening and the observed
    # table. Neither threshold encodes an object/task location or dimension.
    points=points[((points-anchor)@outward<.015)&(points[:,2]>table_height+.015)]
    if len(points)<64:return None
    _,indices=np.unique(np.floor(points/.004).astype(np.int32),axis=0,return_index=True)
    points=points[indices]
    bins,counts=np.unique(np.floor(points[:,2]/.002).astype(int),return_counts=True)
    planes=[]
    for index in np.argsort(counts)[::-1][:16]:
        z=(bins[index]+.5)*.002
        layer=points[abs(points[:,2]-z)<.002]
        if len(layer)<64:continue
        centred=layer-layer.mean(axis=0)
        _,_,axes=np.linalg.svd(centred,full_matrices=False)
        if abs(axes[-1,2])<.98:continue
        local=layer@basis
        low,high=np.quantile(local,[.01,.99],axis=0)
        span=high-low
        if min(span[:2])<.025:continue
        cells=np.unique(np.floor(local[:,:2]/.006).astype(np.int32),axis=0)
        occupancy=len(cells)/np.prod(np.maximum(1,np.ceil(span[:2]/.006)))
        # A thin height slice through vertical walls is algebraically flat,
        # but only outlines a ring. It cannot stand in for a support plane.
        if occupancy<.25:continue
        planes.append((float(span[0]*span[1]),float(np.median(layer[:,2])),low,high,len(layer)))
    floors=[p for p in planes if p[1]<anchor[2]]
    if not floors:return None
    floor=max(floors,key=lambda p:p[0])
    uppers=[p for p in planes if p[1]>anchor[2] and p[1]>floor[1]+.02 and
        np.all(np.minimum(p[3][:2],floor[3][:2])-np.maximum(p[2][:2],floor[2][:2])>0)]
    upper=max(uppers,key=lambda p:p[0]) if uppers else None
    return {'approximate_cavity_dimensions_m':[
            float(floor[3][0]-floor[2][0]),float(floor[3][1]-floor[2][1]),
            upper[1]-floor[1] if upper else None],
        'cavity_dimension_order':['visible_support_width','visible_support_depth','height_to_observed_upper_plane'],
        'cavity_dimension_axes_world':basis.T.tolist(),
        'observed_support_height_world_m':floor[1],
        'observed_upper_height_world_m':upper[1] if upper else None,
        'cavity_dimension_kind':'visible_surface_estimate; not certified usable clearance',
        'cavity_dimension_scope':'A opening region, gravity-horizontal observed planes; hidden extents unknown',
        'opening_reference_source_step':opening['source_step'],
        'complete_dimensions_known':False,'cavity_clearance_verified':False}


def horizontal_layer_summary(points, max_layers=6):
    """Summarize observed horizontal surfaces in a robot-filtered world ROI.

    Classify normals in the full 3D cloud before height slicing, so a thin
    slice through a vertical wall cannot masquerade as a horizontal plane.
    """
    points=np.asarray(points,dtype=float)
    if points.ndim!=2 or points.shape[1]!=3 or max_layers<1:
        return []
    points=points[np.isfinite(points).all(axis=1)]
    if len(points)<64:return []
    _,indices=np.unique(np.floor(points/.004).astype(np.int64),axis=0,return_index=True)
    points=points[indices]
    if len(points)<64:return []
    from scipy.spatial import cKDTree
    distances,neighbors=cKDTree(points).query(points,k=24,distance_upper_bound=.016)
    horizontal=np.zeros(len(points),dtype=bool)
    for index in range(len(points)):
        adjacent=points[neighbors[index,np.isfinite(distances[index])]]
        if len(adjacent)<12:continue
        centred=adjacent-adjacent.mean(axis=0)
        eigenvalues,eigenvectors=np.linalg.eigh(centred.T@centred)
        if eigenvalues[1]<=0 or eigenvalues[0]/eigenvalues.sum()>=.1:continue
        horizontal[index]=abs(eigenvectors[2,0])>=.98
    points=points[horizontal]
    if len(points)<64:return []
    bins,counts=np.unique(np.floor(points[:,2]/.003).astype(np.int64),return_counts=True)
    candidates=[]
    for index in np.argsort(counts)[::-1]:
        z=(bins[index]+.5)*.003
        layer=points[np.abs(points[:,2]-z)<.0045]
        if len(layer)<64:continue
        low,high=np.quantile(layer[:,:2],[.01,.99],axis=0)
        span=high-low
        if np.min(span)<.025:continue
        cells=np.unique(np.floor(layer[:,:2]/.006).astype(np.int64),axis=0)
        occupancy=len(cells)/np.prod(np.maximum(1,np.ceil(span/.006)))
        if occupancy<.25:continue
        centred=layer-layer.mean(axis=0)
        _,_,axes=np.linalg.svd(centred,full_matrices=False)
        if abs(axes[-1,2])<.98:continue
        residual=float(np.quantile(np.abs(layer[:,2]-np.median(layer[:,2])),.95))
        if residual>.005:continue
        candidates.append((len(cells),float(np.median(layer[:,2])),{
            'world_height_m':float(np.median(layer[:,2])),
            'observed_span_xy_m':span.tolist(),
            'observed_bounds_xy_world_m':[low.tolist(),high.tolist()],
            'observed_point_count':int(len(layer)),
            'height_residual_p95_m':residual,
            'scope':'Observed horizontal surface within supplied ROI; support, aperture and clearance unverified'}))
    selected=[]
    for _,height,record in sorted(candidates,key=lambda item:item[0],reverse=True):
        if all(abs(height-other_height)>=.012 for other_height,_ in selected):
            selected.append((height,record))
            if len(selected)>=max_layers:break
    return [record for _,record in sorted(selected,key=lambda item:item[0])]


def dimension_context(run, identities, *, current_source_step=None):
    """Publish measurements without presenting an old world pose as current.

    The source file retains every recorded extent. A previously captured
    ``current`` ROI is not propagated with the held object; only the separate
    carried reference follows the EEF. Suppress that ROI's old world positions
    in the live context once its capture step differs from the request step.
    """
    path=Path(run)/'dimensions.json'
    data=json.loads(path.read_text()) if path.exists() else {'objects':{}}
    keys=['object_dimensions_m','dimensions_frame','dimensions_kind','dimension_source_step','dimensions_valid',
        'approximate_cavity_dimensions_m','cavity_dimension_order','cavity_dimension_axes_world',
        'observed_support_height_world_m','observed_upper_height_world_m',
        'cavity_clearance_verified','cavity_dimension_kind','complete_dimensions_known',
        'source_steps','scope','status','horizontal_layers','initial_observed_shape','current_observed_shape',
        'horizontal_layer_source_steps','horizontal_layer_scope','horizontal_layer_geometry_group']
    objects={}
    for identity,value in data['objects'].items():
        if identity not in identities:continue
        current={key:value[key] for key in keys if key in value}
        if value.get('dimensions_valid') is False:
            for key in _CARRIED_KEYS:current.pop(key,None)
            current.update({'object_dimensions_m':None,'dimensions_valid':False})
            current.update({key:value[key] for key in _INVALIDATION_KEYS if key in value})
        if current_source_step is not None:
            shape=value.get('current_observed_shape')
            if shape is not None and shape.get('source_steps') != [current_source_step]:
                current['current_observed_shape']=None
                current['current_shape_status']={
                    'status':'earlier_observation_not_updated',
                    'source_steps':shape.get('source_steps',[]),
                    'current_source_step':int(current_source_step),
                    'reason':'Old world-coordinate ROI extents are not a current object pose. '
                        'Use current carried geometry or a current observation; original extents remain in dimensions.json.'}
            if (value.get('horizontal_layer_geometry_group')=='current' and
                    value.get('horizontal_layer_source_steps') != [current_source_step]):
                current['horizontal_layers']=[]
                current['horizontal_layer_geometry_group']='earlier_observation'
                current['horizontal_layer_status']='not_updated_for_current_source_step'
        objects[identity]=current
    result={'unit':'m','objects':objects}
    if current_source_step is not None:result['current_source_step']=int(current_source_step)
    return result
