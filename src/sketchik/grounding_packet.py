"""Operation-focused views of recorded RGB-D, without semantic completion.

A's observation boxes define a display window only. Historical surfaces and
current surfaces are stored separately; neither is a certified object mask.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from .scene_cloud import frame_points,region_points
from .observation_reference import focus_reference,observed_region_mask,focus_directions


class FocusReferenceError(ValueError):
    """A-labelled source evidence cannot support the requested projection."""


class ObservedSupportUnavailable(ValueError):
    """Valid observed data does not contain a broad usable support surface."""


def table_window(points):
    """Observed dominant horizontal support; no task coordinates or GT.

    A height histogram followed by connected XY cells suppresses remote walls
    at the same height. The result is only a display crop, not a scene model.
    """
    from scipy.ndimage import label, binary_closing
    points = np.asarray(points)
    if len(points) < 100: raise ObservedSupportUnavailable('Insufficient surfaces for table context')
    heights = np.floor(points[:, 2]/.005).astype(int)
    bins, counts = np.unique(heights, return_counts=True)
    candidates = np.argsort(counts)[::-1][:12]
    best = None
    for candidate in candidates:
        z = (bins[candidate]+.5)*.005
        plane = points[abs(points[:, 2]-z)<.008]
        if len(plane)<100: continue
        cells = np.unique(np.floor(plane[:, :2]/.015).astype(int), axis=0)
        origin, extent = cells.min(axis=0), np.ptp(cells,axis=0)+1
        if np.prod(extent)>2_000_000: continue
        grid=np.zeros(tuple(extent),bool);grid[tuple((cells-origin).T)]=True
        labels,_=label(binary_closing(grid,iterations=1))
        plane_cells=np.floor(plane[:,:2]/.015).astype(int)-origin
        memberships=labels[tuple(plane_cells.T)]
        sizes=np.bincount(memberships,minlength=int(labels.max())+1);sizes[0]=0
        if not sizes.max():continue
        component=int(sizes.argmax())
        occupied=np.column_stack(np.nonzero(labels==component))+origin
        if len(occupied)<100: continue
        low=occupied.min(axis=0)*.015;high=(occupied.max(axis=0)+1)*.015
        score=int(sizes[component])
        if best is None or score>best[0]:
            # The bin centre only selected a slab. Report the dominant
            # connected surface's actual measured elevation, not that centre.
            best=(score,float(np.median(plane[memberships==component,2])),low,high)
    if best is None: raise ObservedSupportUnavailable('No broad observed horizontal support found')
    _,height,low,high=best
    return height,low-.03,high+.03


def prepare_table_context(hub):
    """Write one explicit whole-table handoff artifact for this episode."""
    from .perception import saved_frame
    from .observed_clearance import ObservedClearance
    manifest=hub.run/'scan_evidence.json'
    start=time.perf_counter()
    derived=None
    if manifest.exists():
        sources=json.loads(manifest.read_text())['sources']
    else:
        keyframes=hub.journal.data.get('scene_clouds',{})
        if not keyframes:return None
        sources=[]
        for identity,scene in keyframes.items():
            metadata=json.loads((hub.run/'observations'/identity/'observation.json').read_text())
            if metadata['source_step']!=scene['source_step']:
                raise ValueError('Table keyframe source step does not match its saved observation')
            cameras={camera['camera'] for camera in metadata['cameras']}
            for camera in ('agentview','robot0_eye_in_hand'):
                if camera not in cameras:raise ValueError('Table keyframe requires both saved RGB-D cameras')
                sources.append({'observation_id':identity,'camera':camera,'source_step':metadata['source_step']})
        derived={'source_run':str(hub.run),'sources':sources,'derived_selection':True,
            'selection':'journal scene_clouds keyframes only; initial and completed observation captures; periodic-only frames excluded',
            'source_observation_ids':list(keyframes),'full_observation_history_preserved':True,
            'periodic_rgbd_history_preserved':True,'original_scene_clouds_preserved':True}
    if not sources:return None
    folder=hub.run/'table_context';folder.mkdir(exist_ok=True)
    filter_=ObservedClearance(hub.executor.ik)
    actual=hub.robot.state()
    chunks=[];reports=[]
    try:
        for source in sources:
            identity,camera=source['observation_id'],source['camera']
            frame=saved_frame(hub.run,identity,camera)
            metadata=json.loads((hub.run/'observations'/identity/'observation.json').read_text())
            state=metadata['state']
            if state['step']!=frame['step'] or source.get('source_step',frame['step'])!=frame['step']:
                raise ValueError('Table RGB-D and robot state must share the saved source step')
            calibration=next(item for item in metadata['cameras'] if item['camera']==camera)
            if any(not np.array_equal(frame[key],calibration[key]) for key in ('K','world_from_camera')):
                raise ValueError('Table RGB-D calibration does not match its saved source observation')
            xyz,rgb,uv=frame_points(frame)
            # The native images and original complete fused scan remain intact.
            # This display artifact uses a 2-pixel derived sampling grid.
            selected=(uv[:,0]%2==0)&(uv[:,1]%2==0)
            xyz,rgb,uv=xyz[selected],rgb[selected],uv[selected]
            selected=filter_.non_robot_mask(xyz,state)
            xyz,rgb,uv=xyz[selected],rgb[selected],uv[selected]
            chunks.append((xyz,rgb,uv,np.full(len(xyz),len(reports),np.uint16)))
            reports.append({'observation_id':identity,'camera':camera,'source_step':frame['step']})
    finally:filter_.set_state(actual)
    if derived is not None:manifest.write_text(json.dumps(derived,indent=2))
    xyz,rgb,uv,sids=map(np.concatenate,zip(*chunks))
    height,low,high=table_window(xyz)
    selected=np.all((xyz[:,:2]>=low)&(xyz[:,:2]<=high),axis=1)&(xyz[:,2]>=height-.025)
    xyz,rgb,uv,sids=xyz[selected],rgb[selected],uv[selected],sids[selected]
    np.savez(folder/'observed_table.npz',world_xyz_m=xyz,rgb=rgb,pixel_xy=uv,source_index=sids)
    from .cloud_selection import write_views
    views=write_views(folder/'observed_table.npz',reports,folder,'table',(1,-1,.8),
        role='historical whole table; moved object poses are stale',selectable=False)
    report={'cloud_image':views[0]['image_file'],'views':views,'point_cloud_file':str(folder/'observed_table.npz'),
        'sources':reports,'source_steps':sorted({s['source_step'] for s in reports}),
        'points':len(xyz),'observed_table_height_m':height,
        'display_xy_min_m':low.tolist(),'display_xy_max_m':high.tolist(),
        'tabletop_retained':True,'all_observed_contents_within_display_retained':True,
        'scope':'historical multiview table context; not current object poses after contact',
        'derived_pixel_stride':2,'full_native_and_scan_preserved':True,
        'source_manifest_file':str(manifest),'source_state_and_calibration_matched':True,
        'robot_removed_using_source_state':True,'hidden_surface_completion':False,
        'world_crop_is_gt':False,'wall_s':time.perf_counter()-start}
    (folder/'context.json').write_text(json.dumps(report,indent=2))
    return report


def focus_bounds(points):
    """Trim distant background for display, never infer object dimensions."""
    points = np.asarray(points)
    if len(points) < 32:
        raise ValueError('Insufficient observed region for a focused display')
    low, high = np.quantile(points, [.02, .98], axis=0)
    return low-.015, high+.015


def in_window(points, bounds):
    low, high = bounds
    return np.all((points >= low) & (points <= high), axis=1)


def prepare_cavity_dimensions(hub, handoff):
    """Write measurements before planning; A is never asked to fill them."""
    from .perception import saved_frame
    from .observed_dimensions import cavity_dimensions, write_measurement
    table_file=hub.run/'table_context/context.json'
    opening_file=hub.run/'opening_evidence.json'
    if not table_file.exists() or not opening_file.exists():return {}
    table=json.loads(table_file.read_text())
    with np.load(table['point_cloud_file']) as data:points=data['world_xyz_m']
    results={}
    for opening in json.loads(opening_file.read_text()):
        identity=opening.get('target',{}).get('target_id')
        target=next((v for v in handoff['objects'] if v['object_id']==identity),None)
        if target is None:continue
        evidence=next((v for v in target['evidence'] if v.get('bounds_xyxy')),None)
        if evidence is None:continue
        frame=saved_frame(hub.run,evidence['observation_id'],evidence['camera'])
        xyz,_,uv=frame_points(frame)
        x0,y0,x1,y1=evidence['bounds_xyxy']
        mask=(uv[:,0]>=x0)&(uv[:,0]<x1)&(uv[:,1]>=y0)&(uv[:,1]<y1)
        mask &= np.all((xyz[:,:2]>=table['display_xy_min_m'])&(xyz[:,:2]<=table['display_xy_max_m']),axis=1)
        mask &= xyz[:,2]>=table['observed_table_height_m']-.025
        if np.count_nonzero(mask)<32:continue
        region=points[in_window(points,focus_bounds(xyz[mask]))]
        measured=cavity_dimensions(region,table['observed_table_height_m'],opening)
        if measured:
            measured['cavity_dimension_source_steps']=table['source_steps']
            write_measurement(hub.run,identity,measured)
            results[identity]=measured
    return results


def _save_cloud(folder, name, chunks):
    if not chunks:
        np.savez(folder/(name+'.npz'), world_xyz_m=np.empty((0, 3), np.float32),
                 rgb=np.empty((0, 3), np.uint8), pixel_xy=np.empty((0, 2), np.uint16),
                 source_index=np.empty(0, np.uint16))
        return np.empty((0, 3)), np.empty((0, 3))
    xyz, rgb, pixels, sources = map(np.concatenate, zip(*chunks))
    np.savez(folder/(name+'.npz'), world_xyz_m=xyz, rgb=rgb,
             pixel_xy=pixels, source_index=sources)
    return xyz, rgb


def render_context_angles(points, bounds, source_camera_position, picture):
    """Three virtual views of the same historical measured surfaces, no motion."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    centre=(bounds[0]+bounds[1])/2
    eye=np.asarray(source_camera_position)-centre
    azimuth=float(np.degrees(np.arctan2(eye[1],eye[0])))
    elevation=float(np.degrees(np.arctan2(eye[2],np.linalg.norm(eye[:2]))))
    indices=np.linspace(0,len(points)-1,min(100000,len(points)),dtype=int)
    normalizer=Normalize(float(points[:,2].min()),float(points[:,2].max()))
    fig=plt.figure(figsize=(18,6))
    for index,(offset,title) in enumerate([(0,'Toward the observed face'),(-30,'Oblique left'),(30,'Oblique right')],1):
        ax=fig.add_subplot(1,3,index,projection='3d')
        ax.scatter(*points[indices].T,c=points[indices,2],cmap='viridis',norm=normalizer,s=1,depthshade=False)
        ax.view_init(elevation,azimuth+offset)
        for setter,low,high in zip([ax.set_xlim,ax.set_ylim,ax.set_zlim],*bounds):setter(low,high)
        ax.set_box_aspect(np.maximum(bounds[1]-bounds[0],.01))
        ax.set_title(title);ax.set_xlabel('World X (m)');ax.set_ylabel('World Y (m)');ax.set_zlabel('World Z (m)')
    fig.suptitle('Same HISTORICAL multiview surfaces from three virtual directions / color = world height')
    fig.text(.5,.03,'No new robot movement or hidden-surface completion. Historical shape is not a current object pose after contact.',ha='center')
    fig.tight_layout(rect=[0,.05,1,.95]);fig.savefig(picture,dpi=125);plt.close(fig)


def render_carried_surfaces(local, state, folder, rgb=None, hand_bounds=None, pad_centres=None):
    """Single carried-component detail; no semantic mask or full-shape claim."""
    from scipy.spatial.transform import Rotation
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=Path(folder);folder.mkdir(exist_ok=True)
    rotation=Rotation.from_quat(state['quaternion_xyzw']).as_matrix()
    origin=np.asarray(state['position'])
    points=np.asarray(local)@rotation.T+origin
    plot_points=[points,origin[None,:]]
    colors=np.full((len(points),3),.45) if rgb is None else np.asarray(rgb)/255.
    np.savez(folder/'carried_surfaces.npz',world_xyz_m=points,eef_xyz_m=local,
        rgb=np.round(colors*255).astype(np.uint8),original_rgb_available=rgb is not None)
    fig=plt.figure(figsize=(9,4))
    ax=fig.add_subplot(121,projection='3d')
    ax.scatter(*points.T,c=colors,s=3,depthshade=False)
    ax.set_box_aspect(np.maximum(np.ptp(points,axis=0),.01))
    ax.set_xlabel('World X (m)');ax.set_ylabel('World Y (m)');ax.set_zlabel('World Z (m)')
    ax.set_title('Current carried component')
    side=fig.add_subplot(122)
    side.scatter(points[:,1],points[:,2],c=colors,s=3)
    for axis,color,name in zip(rotation.T,['r','g','b'],['tool X / jaws','tool Y','tool Z / fingers']):
        ax.quiver(*origin,*axis,length=.04,color=color)
        ax.text(*(origin+.045*axis),name,color=color,fontsize=7)
    if hand_bounds is not None:
        from itertools import product
        low,high=np.asarray(hand_bounds)
        corners=np.array(list(product(*zip(low,high))))@rotation.T+origin
        plot_points.append(corners)
        for i in range(8):
            for bit in (1,2,4):
                j=i^bit
                if i<j:
                    ax.plot(*corners[[i,j]].T,color='grey',linewidth=.7)
                    side.plot(corners[[i,j],1],corners[[i,j],2],color='grey',linewidth=.7)
    if pad_centres is not None:
        pads=np.asarray(pad_centres)@rotation.T+origin
        ax.scatter(*pads.T,c='red',marker='x',s=22)
        side.scatter(pads[:,1],pads[:,2],c='red',marker='x',s=22)
    ax.set_box_aspect(np.maximum(np.ptp(np.concatenate(plot_points),axis=0),.01))
    side.set_aspect('equal');side.set_xlabel('World Y (m)');side.set_ylabel('World Z (m)');side.grid(alpha=.3)
    fig.suptitle('Current carried surfaces / RGB; grey: known hand envelope; red: pads',fontsize=10)
    fig.tight_layout();picture=folder/'carried_detail.png';fig.savefig(picture,dpi=125);plt.close(fig)
    return picture


def build_focused_context(hub, handoff, target_id, description, folder, *, active_views_only=False, selected_view=None, whole_object=False, view_prefix=None):
    start = time.perf_counter()
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    target = next(item for item in handoff['objects'] if item['object_id'] == target_id)
    recorded_evidence = [v for v in target['evidence'] if v.get('bounds_xyxy')]
    primary=(next((v for v in recorded_evidence if v.get('surface_role')=='overall_shape'),
        recorded_evidence[0] if recorded_evidence else None) if whole_object else
        focus_reference(recorded_evidence,selected_view))
    evidence = [primary] if primary is not None else []
    record = hub.journal.data['objects'].get(target_id, {})
    invalidated = record.get('invalidated_at_step')
    if not evidence:
        # A coarse region is optional. Keep the native image-point path usable
        # without inventing an object crop, mask, centre, or display bounds.
        result={'target_id':target_id,'local_context_available':False,
            'unavailable_reason':'no_coarse_observation_region',
            'scope':'A supplied no coarse image region; no local point cloud was constructed',
            'cloud_image':None,'selectable_views':[],'sources':[],
            'historical_points':0,'current_points':0,
            'world_display_min_m':None,'world_display_max_m':None,
            'viewport_is_object_dimensions':False,
            'historical_geometry_may_be_stale':invalidated is not None,
            'historical_and_current_fused':False,'known_robot_removed':False,
            'view_selection':'unavailable: no A coarse region',
            'fallback':'current native RGB image_point anchors remain available',
            'hidden_surface_completion':False,'object_gt_used':False,
            'wall_s':time.perf_counter()-start}
        (folder/'context.json').write_text(json.dumps(result,indent=2))
        return result
    from .perception import saved_frame
    from .observed_clearance import ObservedClearance

    primary = evidence[0]
    surface_role=primary.get('surface_role','overall_shape')
    focused=surface_role!='overall_shape' and not whole_object
    # Union the focal patch with the object's other task-contact/context regions.
    # Neither an old exterior box may erase the floor, nor a tight floor crop
    # erase the open door needed by the following stage.
    evidence += [v for v in recorded_evidence if v!=primary]
    frame = saved_frame(hub.run, primary['observation_id'], primary['camera'])
    xyz, _, pixels = frame_points(frame)
    x0, y0, x1, y1 = primary['bounds_xyxy']
    selected = (pixels[:, 0] >= x0) & (pixels[:, 0] < x1) & (pixels[:, 1] >= y0) & (pixels[:, 1] < y1)
    table_meta=hub.run/'table_context/context.json'
    if table_meta.exists():
        table=json.loads(table_meta.read_text())
        on_table=np.all((xyz[:,:2]>=table['display_xy_min_m'])&(xyz[:,:2]<=table['display_xy_max_m']),axis=1)&(xyz[:,2]>=table['observed_table_height_m']-.025)
        if focused:
            selected &= xyz[:,2]>=table['observed_table_height_m']-.025
        elif np.count_nonzero(selected&on_table)>=32:selected &= on_table
    primary_region=xyz[selected]
    if len(primary_region)<32:raise FocusReferenceError('Labelled focus source has insufficient positive observed depth')
    focus_aim=np.median(primary_region,axis=0)
    bounds = (primary_region.min(axis=0)-.015,primary_region.max(axis=0)+.015) if focused else focus_bounds(primary_region)
    protected_bounds=[bounds] if focused else []
    for reference in evidence[1:]:
        extra=saved_frame(hub.run,reference['observation_id'],reference['camera'])
        if invalidated is not None and extra['step']<invalidated:continue
        points=region_points(extra,reference['bounds_xyxy'])
        if table_meta.exists():
            relevant=points[:,2]>=table['observed_table_height_m']-.025
            if reference.get('surface_role','overall_shape')=='overall_shape':
                relevant &= np.all((points[:,:2]>=table['display_xy_min_m'])&
                    (points[:,:2]<=table['display_xy_max_m']),axis=1)
            points=points[relevant]
        if len(points)<32:raise FocusReferenceError('Related source has insufficient positive observed depth')
        extra_bounds=(points.min(axis=0)-.015,points.max(axis=0)+.015) if (
            reference.get('surface_role','overall_shape')!='overall_shape') else focus_bounds(points)
        if reference.get('surface_role','overall_shape')!='overall_shape':protected_bounds.append(extra_bounds)
        bounds=(np.minimum(bounds[0],extra_bounds[0]),np.maximum(bounds[1],extra_bounds[1]))
    crop={'method':'source_region_window','available':False}
    if table_meta.exists() and invalidated is None:
        from .cloud_extents import seeded_display_bounds
        grown,crop=seeded_display_bounds(table,evidence,hub.run/'table_context')
        if grown is not None:
            # Preserve exact extrema of labelled contact/support patches too;
            # geometric grouping cannot overrule an explicit visible reference.
            bounds=grown
            for low,high in protected_bounds:
                bounds=(np.minimum(bounds[0],low),np.maximum(bounds[1],high))
    if whole_object:focus_aim=(np.asarray(bounds[0])+np.asarray(bounds[1]))/2
    historical_bounds = bounds
    from .cloud_extents import fragment_filter
    keep_fragment=fragment_filter(crop)
    if invalidated is not None:
        # Current display also covers the actual EEF neighbourhood after a
        # possible grasp/move. This is a window, not an object pose estimate.
        eef = np.asarray(description['state']['position'])
        bounds = (np.minimum(bounds[0], eef-.12), np.maximum(bounds[1], eef+.12))
    manifest = hub.run/'scan_evidence.json'
    scan_sources = json.loads(manifest.read_text()).get('sources', []) if manifest.exists() else []
    sources, seen = [], set()
    for source in [*scan_sources, *recorded_evidence, *evidence]:
        key = (source['observation_id'], source['camera'])
        if key not in seen:
            sources.append({'observation_id': key[0], 'camera': key[1]})
            seen.add(key)
    current_id = description['observation_id']
    current_frames = hub.observations[current_id]
    ik, actual_state = hub.executor.ik, hub.robot.state()
    robot_filter = ObservedClearance(ik)
    historical, current, source_reports = [], [], []
    protected={(v['observation_id'],v['camera']) for v in evidence}
    try:
        # Read the explicit episode handoff, retaining source frame/pixel IDs.
        # Avoid re-projecting the same recorded native views for every move.
        if table_meta.exists():
            table=json.loads(table_meta.read_text())
            with np.load(table['point_cloud_file']) as data:
                points,colors,uv,sids=(data[k] for k in ['world_xyz_m','rgb','pixel_xy','source_index'])
                mask=in_window(points,historical_bounds)
                if keep_fragment is not None:
                    selected=np.flatnonzero(mask)
                    mask[selected]&=keep_fragment(points[selected])
                # Re-read explicit surface sources at native density below;
                # the broad table display's stride must not erase their pixels.
                protected_source=np.array([(v['observation_id'],v['camera']) in protected for v in table['sources']])
                mask &= ~protected_source[sids]
                if invalidated is not None:
                    valid_source=np.array([v['source_step']<invalidated for v in table['sources']])
                    mask &= valid_source[sids]
                historical.append((points[mask],colors[mask],uv[mask],sids[mask]))
            source_reports=[{**v,'role':'historical_shape'} for v in table['sources']]
            recorded={(v['observation_id'],v['camera']) for v in table['sources']}
            sources=[v for v in sources if (v['observation_id'],v['camera']) not in recorded
                or (v['observation_id'],v['camera']) in protected]
        for source in sources:
            frame = saved_frame(hub.run, source['observation_id'], source['camera'])
            # Do not fuse surfaces recorded after contact into an old static
            # shape. Their current evidence comes from the current dual frame.
            if invalidated is not None and frame['step'] >= invalidated:
                continue
            state = json.loads((hub.run/'observations'/source['observation_id']/'observation.json').read_text())['state']
            points, colors, uv = frame_points(frame)
            mask = in_window(points, historical_bounds)
            if keep_fragment is not None:
                selected=np.flatnonzero(mask)
                mask[selected]&=keep_fragment(points[selected])
            points, colors, uv = points[mask], colors[mask], uv[mask]
            mask = robot_filter.non_robot_mask(points, state)
            points, colors, uv = points[mask], colors[mask], uv[mask]
            index = len(source_reports)
            historical.append((points, colors, uv, np.full(len(points), index, np.uint16)))
            source_reports.append({**source, 'source_step': frame['step'], 'points': len(points), 'role': 'historical_shape'})
        for camera, frame in current_frames.items():
            points, colors, uv = frame_points(frame)
            mask = in_window(points, bounds)
            if keep_fragment is not None:
                selected=np.flatnonzero(mask)
                mask[selected]&=keep_fragment(points[selected])
            points, colors, uv = points[mask], colors[mask], uv[mask]
            mask = robot_filter.non_robot_mask(points, actual_state)
            points, colors, uv = points[mask], colors[mask], uv[mask]
            index = len(source_reports)
            current.append((points, colors, uv, np.full(len(points), index, np.uint16)))
            source_reports.append({'observation_id': current_id, 'camera': camera,
                'source_step': frame['step'], 'points': len(points), 'role': 'current_surfaces'})
    finally:
        robot_filter.set_state(actual_state)
    old_xyz, old_rgb = _save_cloud(folder, 'historical_surfaces', historical)
    now_xyz, now_rgb = _save_cloud(folder, 'current_surfaces', current)
    if len(now_xyz) < 32:
        raise FocusReferenceError('Current focused window has insufficient observed surfaces')
    from .observed_dimensions import horizontal_layer_summary, write_measurement
    use_historical = invalidated is None and len(old_xyz) >= 32
    layers = horizontal_layer_summary(old_xyz if use_historical else now_xyz)
    layer_sources = [v for v in source_reports if v['role'] ==
        ('historical_shape' if use_historical else 'current_surfaces')]
    layer_measurement = {'horizontal_layers': layers,
        'horizontal_layer_source_steps': sorted({int(v['source_step']) for v in layer_sources}),
        'horizontal_layer_scope': 'Measured visible horizontal surfaces in the operation ROI; no semantic floor/roof identity or usable cavity clearance certified',
        'horizontal_layer_geometry_group': 'historical' if use_historical else 'current'}
    write_measurement(hub.run, target_id, layer_measurement)
    from .cloud_extents import summarize_components
    table_height = table.get('observed_table_height_m') if table_meta.exists() else None
    extent_summary = summarize_components(old_xyz if use_historical else now_xyz,
        table_height=table_height)
    group_ids = extent_summary.pop('point_group_ids')
    extent_summary.update(source_steps=layer_measurement['horizontal_layer_source_steps'],
        geometry_group=layer_measurement['horizontal_layer_geometry_group'],
        axes='world_XYZ',semantic_identity_verified=False,
        meaning='Connected observed fragments in the operation ROI. A group may contain multiple touching surfaces; one object may occupy multiple groups. Bounds are visible extents, not usable cavity dimensions.')
    np.savez_compressed(folder/'extent_groups.npz',point_group_ids=group_ids,
        cloud_file=str(folder/('historical_surfaces.npz' if use_historical else 'current_surfaces.npz')))
    from .cloud_selection import write_views
    # An actual labelled support view or labelled rim defines the viewing side.
    # A first exterior object photograph must not silently choose the cavity view.
    primary_frame=saved_frame(hub.run,primary['observation_id'],primary['camera'])
    view_source=(next((value for role in ('interior_support','opening') for value in evidence
        if value.get('surface_role')==role),primary) if whole_object else primary)
    view_frame=saved_frame(hub.run,view_source['observation_id'],view_source['camera'])
    eye=np.asarray(view_frame['world_from_camera'])[:3,3]-focus_aim
    # Opening-side projections apply to an entry/support operation. An exterior
    # contact may legitimately be observed from another side of that object.
    opening=next((v for v in evidence if v.get('rim_pixels_xy') and
        v.get('surface_role')=='opening'),None) if surface_role in ('opening','interior_support') else None
    direction_source='real labelled source bearing'
    outward_normal=None
    if opening is not None:
        from .program_scan import describe_opening,unit
        opening_frame=saved_frame(hub.run,opening['observation_id'],opening['camera'])
        try:
            measured,_=describe_opening(opening_frame,opening['bounds_xyxy'],opening['rim_pixels_xy'])
        except ValueError as exc:
            raise FocusReferenceError('Invalid observed opening reference: '+str(exc)) from exc
        normal=np.asarray(measured['proposed_surface_normal_world'])
        outward_normal=normal
        bearing=unit(eye)
        if np.dot(normal,bearing)<=0:
            raise FocusReferenceError('Opening and focus sources view opposite sides; review the actual surface references')
        vertical=np.array([0.,0.,1.])-normal*normal[2]
        if np.linalg.norm(vertical)>1e-6:
            vertical=unit(vertical)
            eye=unit(normal+max(0.,float(bearing@vertical))*vertical)
        else:eye=normal
        direction_source='A-labelled depth-fitted opening outward normal with source-supported elevation'
    directions=focus_directions(eye,np.asarray(view_frame['world_from_camera'])[:3,0],outward_normal)
    views=[]
    if not active_views_only or (invalidated is None and len(old_xyz)>=32):
        views+=write_views(folder/'historical_surfaces.npz',source_reports,folder,
            (view_prefix or target_id)+'_historical',eye,role='historical multiview surfaces',selectable=invalidated is None,
            group_ids=group_ids if use_historical else None,directions=directions)
    if not active_views_only or invalidated is not None or not views:
        views+=write_views(folder/'current_surfaces.npz',source_reports,folder,
            (view_prefix or target_id)+'_current',eye,role='current dual-view surfaces',selectable=True,
            group_ids=group_ids if not use_historical else None,directions=directions)
    focus_visibility=[]
    for view in views:
        with np.load(view['cloud_file']) as data:
            labelled=observed_region_mask(data['world_xyz_m'],primary_frame,primary['bounds_xyxy'])
        with np.load(view['index_map_file']) as data:
            indices=data['point_index'];valid=indices>=0
            visible=int(np.count_nonzero(labelled[indices[valid]]))
        audit={'view_id':view['view_id'],'labelled_source_points':int(labelled.sum()),
            'labelled_source_visible_pixels':visible,'semantic_identity':'A assertion; not geometrically certified'}
        focus_visibility.append(audit)
        view['focus_reference']={key:value for key,value in primary.items() if key!='visible_evidence'}
        view['projection_direction_source']=direction_source
    if not whole_object and surface_role=='interior_support' and not any(v['labelled_source_visible_pixels']>=32 for v in focus_visibility):
        raise FocusReferenceError('Labelled internal support is not exposed in the focused projection; review its real source before manipulation')
    picture=Path(views[0]['image_file'])
    dimensions_file=hub.run/'dimensions.json'
    known=json.loads(dimensions_file.read_text()).get('objects',{}).get(target_id,{})
    if use_historical and 'initial_observed_shape' not in known:
        write_measurement(hub.run,target_id,{'initial_observed_shape':extent_summary,
            'initial_observed_shape_view':{'image_file':str(picture),
                'source_steps':extent_summary['source_steps'],
                'current_object_pose':False,'selectable':False}})
    elif not use_historical:
        write_measurement(hub.run,target_id,{'current_observed_shape':extent_summary})
    result = {'target_id': target_id, 'cloud_image': str(picture), 'local_context_available':True,
        **layer_measurement,
        'observed_extent_groups':extent_summary,
        'selectable_views':views,
        'sources': source_reports, 'historical_points': len(old_xyz), 'current_points': len(now_xyz),
        'world_display_min_m': bounds[0].tolist(), 'world_display_max_m': bounds[1].tolist(),
        'viewport_is_object_dimensions': False, 'historical_geometry_may_be_stale': invalidated is not None,
        'historical_and_current_fused': False, 'known_robot_removed': True,
        'view_selection':'three separated source-backed RGB projections in the opening/source front hemisphere',
        'focus_reference':{key:value for key,value in primary.items() if key!='visible_evidence'},
        'reference_scope':'whole observed object; view bearing/coverage regions do not select the execution part' if whole_object else 'observation display',
        'projection_direction_source':direction_source,'focus_visibility':focus_visibility,
        'display_crop':crop,
        'hidden_surface_completion': False, 'object_gt_used': False, 'wall_s': time.perf_counter()-start}
    (folder/'context.json').write_text(json.dumps(result, indent=2))
    return result
