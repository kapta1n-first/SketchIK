"""Five calibrated table views before any agent request, without object GT.

The table centre is the centre of an observed display rectangle. View coverage
concerns measured surfaces only; no unseen backside or complete scene is certified.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .grounding_packet import table_window, ObservedSupportUnavailable
from .motion_checks import motion_route
from .program_scan import look_at, unit


DIRECTIONS=('top','front','back','left','right')


def _promote_oblique_candidate(candidates):
    """Give one well-framed lower view a turn without losing distance fallbacks."""
    reliable=[candidate for candidate in candidates
        if candidate['table_frustum_fraction']>=.95
        and candidate['above_table_frustum_fraction']>=.9
        and candidate['known_surface_visible_fraction']>=.8]
    if reliable:
        preferred=min(reliable,key=lambda candidate:(candidate['elevation_rad'],-candidate['score']))
        first=next(index for index,candidate in enumerate(candidates)
            if candidate['standoff_m']==preferred['standoff_m'])
        selected=next(index for index,candidate in enumerate(candidates) if candidate is preferred)
        # Swap within its family: the first round still includes every distance,
        # and other families keep their existing first-round fallbacks.
        candidates[first],candidates[selected]=candidates[selected],candidates[first]
    return candidates


def table_geometry(points, point_file):
    """Measured support window and its visible contents, not object dimensions."""
    points=np.asarray(points)
    height,low,high=table_window(points)
    mask=np.all((points[:,:2]>=low)&(points[:,:2]<=high),axis=1)&(points[:,2]>=height-.025)
    selected=points[mask]
    if not len(selected):raise ObservedSupportUnavailable('Observed table window has no measured surfaces')
    np.savez(point_file,world_xyz_m=selected)
    sample=selected[np.linspace(0,len(selected)-1,min(5000,len(selected)),dtype=int)]
    above=sample[:,2]>height+.025
    centre=np.r_[(low+high)/2,height]
    aim=centre.copy()
    aim[2]+=.5*max(0.,float(np.quantile(selected[:,2],.95)-height))
    return {'observed_table_height_m':float(height),'display_xy_min_m':low.tolist(),
        'display_xy_max_m':high.tolist(),'table_center_world_m':centre.tolist(),
        'view_aim_world_m':aim.tolist(),'geometry_point_file':str(point_file),
        'observed_points':len(selected),'centre_is_observed_rectangle_not_gt':True,
        'hidden_surface_completion':False},sample,above


def scan_candidates(state,wrist,geometry,points,above,direction,clearance):
    """Calibrated multi-distance camera proposals and their full EEF poses."""
    if wrist['step']!=state['step']:raise ValueError('Camera and robot state must share one physical step')
    k=np.asarray(wrist['K']);height,width=np.asarray(wrist['depth_m']).shape
    camera0=np.asarray(wrist['world_from_camera'])
    eef=np.eye(4);eef[:3,:3]=Rotation.from_quat(state['quaternion_xyzw']).as_matrix();eef[:3,3]=state['position']
    mount=np.linalg.inv(camera0)@eef
    aim=np.asarray(geometry['view_aim_world_m'])
    span=np.asarray(geometry['display_xy_max_m'])-geometry['display_xy_min_m']
    radius=max(.1,float(np.linalg.norm(span)/2))
    half_fov=min(np.arctan(width/(2*k[0,0])),np.arctan(height/(2*k[1,1])))
    nominal=max(.25,radius/np.tan(half_fov))
    elevations=[np.pi/2] if direction is None else np.deg2rad([50.,65.,80.])
    result=[]
    for elevation in elevations:
        bearing=np.array([0.,0.,1.]) if direction is None else np.r_[direction*np.cos(elevation),np.sin(elevation)]
        for factor in (.7,.9,1.1):
            distance=nominal*factor;eye=aim+distance*bearing
            camera=look_at(eye,aim,camera0[:3,0]);desired=camera@mount
            local=(points-eye)@camera[:3,:3];screen=local@k.T
            uv=screen[:,:2]/np.maximum(screen[:,2:3],1e-8)
            inside=(local[:,2]>.04)&(uv[:,0]>.04*width)&(uv[:,0]<.96*width)&(uv[:,1]>.04*height)&(uv[:,1]<.96*height)
            table_fraction=float(np.mean(inside[~above])) if np.any(~above) else float(np.mean(inside))
            above_fraction=float(np.mean(inside[above])) if np.any(above) else table_fraction
            if not inside.any():continue
            visible=clearance.visibility(camera,k,points,width=width,height=height)
            travel=float(np.linalg.norm(desired[:3,3]-np.asarray(state['position'])))
            score=.4*table_fraction+.6*above_fraction+.5*visible+.1*np.sin(elevation)
            score-=.1*np.linalg.norm(eye-camera0[:3,3])/nominal
            score-=.1*travel/nominal
            result.append({'position':desired[:3,3].tolist(),
                'quaternion_xyzw':Rotation.from_matrix(desired[:3,:3]).as_quat().tolist(),
                'world_from_camera':camera.tolist(),'camera_position_m':eye.tolist(),
                'standoff_m':float(distance),'elevation_rad':float(elevation),'score':float(score),
                'estimated_direct_eef_travel_m':travel,
                'table_frustum_fraction':table_fraction,'above_table_frustum_fraction':above_fraction,
                'known_surface_visible_fraction':float(visible),'hidden_surfaces_checked':False})
    # Try a useful view from each distance family before spending the bounded
    # IK budget on several high-coverage but similarly unreachable far views.
    groups={}
    for candidate in sorted(result,key=lambda item:item['score'],reverse=True):
        groups.setdefault(candidate['standoff_m'],[]).append(candidate)
    ordered=[]
    while groups:
        for distance in sorted(groups,key=lambda value:groups[value][0]['score'],reverse=True):
            ordered.append(groups[distance].pop(0))
            if not groups[distance]:del groups[distance]
    return _promote_oblique_candidate(ordered) if direction is not None else ordered


def _native_pictures(hub,identity):
    pictures=[]
    for camera in ('agentview','robot0_eye_in_hand'):
        _,parts=hub.read_image(identity,camera);pictures.extend(parts)
    return pictures


def _capture(hub):
    before=set(hub.observations)
    try:
        description,pictures=hub.observe_scene()
        return description,pictures,None
    except ObservedSupportUnavailable as exc:
        captured=[identity for identity in hub.observations if identity not in before]
        if not captured:raise
        identity=captured[-1]
        frame=hub.observations[identity]['agentview']
        return {'observation_id':identity,'source_step':frame['step']},_native_pictures(hub,identity),str(exc)


def run_table_scan(hub,*,max_ik_candidates=6):
    """Check fixed initial camera proposals against each actual state and new surfaces."""
    from .observed_clearance import ObservedClearance
    from .kinematics import IKError
    from .tools import public
    if isinstance(max_ik_candidates,bool) or not isinstance(max_ik_candidates,(int,np.integer)) or max_ik_candidates<1:
        raise ValueError('max_ik_candidates must be a positive integer')
    begin=time.perf_counter();initial_step=hub.robot.steps
    folder=Path(hub.run)/'table_scan';folder.mkdir(parents=True,exist_ok=True)
    index=len(list(folder.glob('scan_*.json')));diagnostic=folder/f'scan_{index:03d}.json'
    summary={'status':'completed','source_step':initial_step,'request_source_step':initial_step,
        'attempts':[],'captured_views':0,'observation_ids':[],'sources':[],'geometry_updates':[],
        'max_ik_candidates':int(max_ik_candidates),'diagnostic_file':str(diagnostic),
        'agent_requests':0,'object_gt_used':False,'full_scene_native_history_preserved':True,
        'observed_surface_collision_filter':not getattr(hub,'ik_only',False),
        'bounded_observed_detour_search':not getattr(hub,'ik_only',False),
        'complete_backside_coverage_certified':False,'continuous_collision_certified':False}
    pictures=[]
    def save():
        summary.update(source_step=hub.robot.steps,robot_actions=hub.robot.steps-initial_step,wall_s=time.perf_counter()-begin)
        diagnostic.write_text(json.dumps(public(summary),ensure_ascii=False,indent=2,allow_nan=False))
    def add_source(identity,state):
        frames=hub.observations[identity]
        for camera in ('agentview','robot0_eye_in_hand'):
            frame=frames[camera]
            if frame['step']!=state['step']:raise ValueError('Scan RGB-D and source robot state differ')
            clearance.add_frame({**frame,'camera':camera},state)
            summary['sources'].append({'observation_id':identity,'camera':camera,'source_step':state['step']})
        report=clearance.build();clearance.set_state(state)
        summary['observed_obstacles']=report
    hub.emit({'event':'table_scan_started','source_step':initial_step});save()
    try:
        identity=next(reversed(hub.observations),None)
        frames=hub.observations.get(identity,{})
        if not all(camera in frames and frames[camera].get('step')==initial_step for camera in ('agentview','robot0_eye_in_hand')):
            description,parts,error=_capture(hub);identity=description['observation_id'];pictures.extend(parts)
            if error:summary['initial_scene_error']=error
        else:pictures.extend(_native_pictures(hub,identity))
        summary['initial_observation_id']=identity
        initial=hub.robot.state();clearance=ObservedClearance(hub.executor.ik)
        add_source(identity,initial)
        geometry,points,above=table_geometry(clearance.points,folder/f'geometry_{index:03d}_00.npz')
        summary['geometry_updates'].append(geometry)
        front=unit(np.asarray(hub.observations[identity]['agentview']['world_from_camera'])[:2,3]-np.asarray(geometry['table_center_world_m'])[:2])
        left=np.array([-front[1],front[0]])
        directions={'top':None,'front':front,'back':-front,'left':left,'right':-left}
        summary['front_direction_world_xy']=front.tolist()
        planning_geometry=geometry
        wrist=hub.observations[identity]['robot0_eye_in_hand']
        # Freeze both the measured window and calibrated camera roll before the
        # first move. Newly visible table edges must not push later goals away.
        candidates_by_direction={name:scan_candidates(initial,wrist,planning_geometry,
            points,above,directions[name],clearance) for name in DIRECTIONS}
        summary['planning_reference']={'observation_id':identity,'source_step':initial['step'],
            'geometry':planning_geometry,'world_from_camera':np.asarray(wrist['world_from_camera']).tolist(),
            'state':initial,'candidates':candidates_by_direction,'targets_frozen':True,
            'candidate_scores_source_step':initial['step']}
    except ObservedSupportUnavailable as exc:
        summary.update(status='initial_geometry_unavailable',error=f'{type(exc).__name__}: {exc}')
        summary['attempts']=[{'target':{'target_id':'table_'+name},'status':'skipped_geometry_unavailable'} for name in DIRECTIONS]
        save();hub.emit({'event':'table_scan_finished',**public(summary)});return public(summary),pictures
    for slot,name in enumerate(DIRECTIONS):
        attempt={'target':{'target_id':'table_'+name},'source_step':hub.robot.steps,'checks':[],'move_called':False}
        summary['attempts'].append(attempt);stop=False
        try:
            state=hub.robot.state();candidates=candidates_by_direction[name]
            attempt.update(geometry=planning_geometry,observed_geometry=geometry,candidates=candidates,
                planning_reference_source_step=initial['step']);chosen=None
            for candidate_index,candidate in enumerate(candidates[:max_ik_candidates]):
                checked=time.perf_counter()
                try:
                    route=motion_route(hub,clearance,state,candidate)
                    check={'candidate_index':candidate_index,'accepted':True,
                        'checked_configurations':route['checked_configurations'],
                        'wall_s':time.perf_counter()-checked}
                    if getattr(hub,'ik_only',False):
                        check.update(observed_surface_checked=False,observed_clearance_passed=None,
                            collision_checked=False,mode='ik_only')
                    else:check['observed_clearance_passed']=True
                    attempt['checks'].append(check);chosen=candidate;break
                except (ValueError, IKError) as exc:
                    attempt['checks'].append({'candidate_index':candidate_index,'accepted':False,
                        'error':str(exc),'wall_s':time.perf_counter()-checked})
            if chosen is None:attempt['status']='skipped_no_checked_view'
            else:
                attempt.update(selected=chosen,checked_route=route,move_called=True)
                hub.emit({'event':'table_scan_view_selected','target_id':'table_'+name,'candidate':chosen})
                try:
                    motion=hub.executor.move_waypoints(route['waypoints'])
                    attempt.update(execution=public(motion),status='view_captured' if motion['converged'] else 'motion_not_converged')
                    stop=not motion['converged']
                except IKError as exc:attempt.update(status='motion_failed',error=str(exc));stop=True
                observation,parts,error=_capture(hub);pictures.extend(parts)
                identity=observation['observation_id'];summary['observation_ids'].append(identity);summary['captured_views']+=1
                attempt.update(captured_observation_id=identity,execution_source_step=hub.robot.steps)
                if error:attempt['scene_error']=error;stop=True
                add_source(identity,hub.robot.state())
                geometry,points,above=table_geometry(clearance.points,folder/f'geometry_{index:03d}_{slot+1:02d}.npz')
                summary['geometry_updates'].append(geometry)
        except ObservedSupportUnavailable as exc:
            attempt.update(status='view_failed',error=f'{type(exc).__name__}: {exc}');stop=attempt['move_called']
        hub.emit({'event':'table_scan_view_finished',**public(attempt)})
        if stop:
            summary['status']='stopped_after_view_failure'
            summary['unattempted_directions']=list(DIRECTIONS[slot+1:])
        save()
        if stop:break
    if summary['status']=='completed' and summary['captured_views']<len(DIRECTIONS):summary['status']='completed_with_skipped_views'
    save();hub.emit({'event':'table_scan_finished',**public(summary)})
    return public(summary),pictures
