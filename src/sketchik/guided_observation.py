"""Program-controlled complementary views of bound scene components.

Geometry scores concern known visibility and acquisition gain. Only A can
confirm an actual task surface; no candidate camera or horizontal patch does so.
"""
import json
import time
from itertools import product

import numpy as np
from scipy.spatial.transform import Rotation

from .program_scan import look_at,unit,diverse_view_order
from .motion_checks import motion_route
from .scene_catalog import refresh_component_evidence
from .tools import public
from .robot_geometry import hand_bounds_in_eef


def voxel_points(points,size=.006):
    points=np.asarray(points)
    if not len(points):return points
    _,indices=np.unique(np.floor(points/size),axis=0,return_index=True)
    return points[indices]


def projection_quality(points,scene,camera,K,width=1024,height=1024):
    """Known-surface z-buffer quality, without claiming unknown occluders free."""
    def project(xyz):
        local=(xyz-camera[:3,3])@camera[:3,:3]
        screen=local@K.T
        uv=np.floor(screen[:,:2]/np.maximum(screen[:,2:3],1e-9)/4).astype(int)
        valid=(local[:,2]>.04)&(uv[:,0]>=0)&(uv[:,0]<width//4)&(uv[:,1]>=0)&(uv[:,1]<height//4)
        return local[:,2],uv,valid
    depth,uv,valid=project(scene)
    zbuffer=np.full((height//4,width//4),np.inf)
    np.minimum.at(zbuffer,(uv[valid,1],uv[valid,0]),depth[valid])
    depth,uv,valid=project(points);visible=np.zeros(len(points),bool)
    visible[valid]=depth[valid]<=zbuffer[uv[valid,1],uv[valid,0]]+.012
    selected=uv[visible]
    cells=len(np.unique(selected,axis=0)) if len(selected) else 0
    short=float(np.ptp(selected,axis=0).min()*4) if len(selected)>1 else 0.
    quality=min(1.,cells/200)*min(1.,short/80)
    return {'quality':float(quality),'visible_cells':cells,'projected_short_edge_px':short,
        'visible_fraction':float(visible.mean()) if len(visible) else 0.,
        'unknown_occluders_certified':False}


def interior_patches(components):
    candidates=[]
    for component in components:
        low,high=np.asarray(component['observed_bounds_world_m'])
        for patch in component['patches']:
            # This geometric prior avoids repeatedly aiming at the uppermost
            # shell. Remaining candidates are still NOT identified as floors.
            fraction=(patch['world_height_m']-low[2])/max(high[2]-low[2],.001)
            x0,y0,x1,y1=patch['bounds_xyxy'];short=min(x1-x0,y1-y0)
            if fraction>.85 or short<12:continue
            candidate={**patch,'geometric_lower_level_fraction':float(fraction)}
            candidates.append(candidate)
    candidates.sort(key=lambda patch:(-patch['visible_pixel_sample_count'],patch['geometric_lower_level_fraction']))
    distinct=[]
    for patch in candidates:
        if any(np.linalg.norm(np.asarray(patch['look_at_world_m'])-v['look_at_world_m'])<.04 for v in distinct):continue
        distinct.append(patch)
    return distinct[:3]


def measured_patch_quality(components,observation_id=None):
    if observation_id is not None:
        components=[{**v,'patches':[p for p in v['patches'] if p['observation_id']==observation_id]} for v in components]
    patches=interior_patches(components)
    def score(patch):
        x0,y0,x1,y1=patch['bounds_xyxy']
        return min(1.,patch['visible_pixel_sample_count']/400)*min(1.,min(x1-x0,y1-y0)/80)
    return max((score(patch) for patch in patches),default=0.)


def proposal_views(state,wrist,components,need,scene,previous=(),*,hand_bounds):
    if wrist['step']!=state['step']:raise ValueError('Camera and state must share a refreshed physical step')
    chunks=[]
    for component in components:
        with np.load(component['point_cloud_file']) as data:chunks.append(data['world_xyz_m'])
    points=voxel_points(np.concatenate(chunks))
    low,high=points.min(axis=0),points.max(axis=0)
    patches=interior_patches(components) if need=='interior_support' else []
    aims=[]
    for patch in patches:
        aim=np.asarray(patch['look_at_world_m'])
        selected=(abs(points[:,2]-patch['world_height_m'])<.012)&(
            np.linalg.norm(points[:,:2]-aim[:2],axis=1)<max(.04,np.linalg.norm(patch['observed_span_xyz_m'][:2])))
        if selected.sum()<12:continue
        aims.append((aim,points[selected],patch['candidate_id']))
    if not aims:
        for fraction in ([.3,.55] if need=='interior_support' else [.5]):
            aim=(low+high)/2;aim[2]=low[2]+fraction*(high[2]-low[2])
            aims.append((aim,points,None))
    old=np.asarray(wrist['world_from_camera']);eef=np.eye(4)
    eef[:3,:3]=Rotation.from_quat(state['quaternion_xyzw']).as_matrix();eef[:3,3]=state['position']
    mount=np.linalg.inv(old)@eef
    bounds=np.asarray(hand_bounds,dtype=float)
    if bounds.shape!=(2,3) or not np.isfinite(bounds).all() or np.any(bounds[1]<bounds[0]):
        raise ValueError('Finite complete hand bounds required for camera framing')
    corners=np.asarray(list(product(*zip(bounds[0],bounds[1]))))
    hand_in_camera=corners@mount[:3,:3].T+mount[:3,3]
    forward_reach=float(hand_in_camera[:,2].max())
    K=np.asarray(wrist['K']);height,width=np.asarray(wrist['depth_m']).squeeze().shape
    half_fov=min(np.arctan(width/(2*K[0,0])),np.arctan(height/(2*K[1,1])))
    result=[]
    for aim_index,(aim,reference,patch_id) in enumerate(aims):
        radius=max(.02,float(np.linalg.norm(np.ptp(reference,axis=0))/2))
        nominal=float(np.clip(radius/(.8*np.tan(half_fov)),.25,.6))
        for direction in range(8):
            azimuth=direction*np.pi/4
            for elevation_deg in (15,30,45):
                elevation=np.deg2rad(elevation_deg)
                bearing=np.array([np.cos(azimuth)*np.cos(elevation),np.sin(azimuth)*np.cos(elevation),np.sin(elevation)])
                # A small floor patch sets the image scale, not how close the
                # hand may sit to the rest of its container. Construct the eye
                # outside the complete target's view-facing extent, including
                # the mechanical hand in front of the wrist camera. This is a
                # framing separation plane, not obstacle/path collision search.
                target_front=float(((points-aim)@bearing).max())
                standoff=max(nominal,target_front+forward_reach+.03)
                for distance in (standoff,standoff+.1):
                    eye=aim+distance*bearing
                    if np.linalg.norm(eye-old[:3,3])>.75:continue
                    if any(np.linalg.norm(eye-v['camera_position_m'])<.025 for v in previous):continue
                    # Gravity fixes image roll; copying the previous wrist X
                    # made each new view inherit arbitrary hand roll.
                    image_right=np.cross(unit(aim-eye),[0.,0.,1.])
                    camera=look_at(eye,aim,image_right);desired=camera@mount
                    measurement=projection_quality(reference,scene,camera,K,width,height)
                    if measurement['visible_cells']<12:continue
                    novelty=min((np.arccos(np.clip(np.dot(bearing,unit(np.asarray(v['camera_position_m'])-aim)),-1,1))
                        for v in previous),default=np.pi/2)
                    if previous and novelty<np.deg2rad(20) and measurement['quality']<1.5*max(
                            v.get('predicted_quality',0.) for v in previous):continue
                    travel=np.linalg.norm(desired[:3,3]-np.asarray(state['position']))
                    score=2*measurement['quality']+.4*min(novelty,np.pi/2)-.8*travel+np.sin(elevation)
                    result.append({'position':desired[:3,3].tolist(),
                        'quaternion_xyzw':Rotation.from_matrix(desired[:3,:3]).as_quat().tolist(),
                        'camera_position_m':eye.tolist(),'world_from_camera':camera.tolist(),
                        'look_at_world_m':aim.tolist(),'patch_candidate_id':patch_id,'score':float(score),
                        'bearing_family':direction,'predicted_quality':measurement['quality'],
                        'predicted_evidence':measurement,'bearing_novelty_rad':float(novelty),
                        'framing_distance_m':float(distance),
                        'hand_forward_from_camera_m':forward_reach,
                        'target_front_from_aim_m':target_front,
                        'hand_target_projection_gap_m':float(distance-forward_reach-target_front),
                        'standoff_scope':'Complete observed target and current hand envelope along viewing axis; other obstacles and paths not certified',
                        'semantic_surface_status':'unconfirmed','collision_checked':False})
    return diverse_view_order(result)


def camera_route(hub,clearance,state,pose,transit_height):
    """Define a raised observation transfer, then check every full-pose segment.

    This is a generic camera motion profile, not collision search/certification.
    In IK-only mode no obstacle configuration rejects a trajectory.
    """
    height=max(transit_height,state['position'][2],pose['position'][2])
    lift={'position':[*state['position'][:2],height],
        'quaternion_xyzw':state['quaternion_xyzw']}
    above={'position':[*pose['position'][:2],height],
        'quaternion_xyzw':pose['quaternion_xyzw']}
    planned=dict(state);waypoints=[];checks=[]
    for target in (lift,above,{key:pose[key] for key in ('position','quaternion_xyzw')}):
        if np.linalg.norm(np.asarray(target['position'])-planned['position'])<.002 and (
                abs(np.dot(target['quaternion_xyzw'],planned['quaternion_xyzw']))>1-1e-6):continue
        checked=motion_route(hub,clearance,planned,target)
        for point in checked['waypoints']:
            if checked.get('mode')=='ik_only':q=checked['final_joint_positions']
            else:q=hub.executor.ik.cartesian_path(planned['q'],point['position'],point['quaternion_xyzw'])[-1].q
            planned.update(q=q,**point);waypoints.append(point)
        checks.append({key:checked.get(key) for key in
            ('mode','path_checked','ik_samples','collision_checked')})
    if not waypoints:raise ValueError('Camera transfer has no new pose')
    return {'waypoints':waypoints,'profile':'raise_transfer_lower','transit_height_m':height,
        'segments':checks,'collision_checked':not getattr(hub,'ik_only',False),
        'continuous_collision_certified':False}


def run_guided_observation(hub,catalog,handoff,*,max_views=4,max_views_per_need=2,max_ik_candidates=8):
    from .observed_clearance import ObservedClearance
    if min(max_views,max_views_per_need,max_ik_candidates)<1:raise ValueError('Positive acquisition budgets required')
    started=time.perf_counter();initial_step=hub.robot.steps
    folder=hub.run/'guided_observation';folder.mkdir(exist_ok=True)
    summary={'source_step':initial_step,'status':'completed','attempts':[],'observation_ids':[],
        'captured_views':0,'max_views':max_views,'max_views_per_need':max_views_per_need,
        'max_ik_candidates':max_ik_candidates,'object_gt_used':False,'semantic_coverage_certified':False,
        'all_scene_obstacles_used_for_visibility':True,'collision_checked':not getattr(hub,'ik_only',False)}
    pictures=[];previous=[];scene=[]
    with np.load(catalog['point_cloud_file']) as data:scene=voxel_points(data['world_xyz_m'])
    grouped={}
    for item in handoff['objects']:
        key=tuple(sorted(item['component_ids']))
        goal=grouped.setdefault(key,{'component_ids':key,'object_ids':[],'need':'external_contact','requested':False})
        goal['object_ids'].append(item['object_id'])
        missing=item.get('acquisition_needs',[])
        goal['requested']|=bool(missing)
        if any(v in missing for v in ('interior_support','opening')):goal['need']='interior_support'
    goals=sorted(grouped.values(),key=lambda goal:goal['need']!='interior_support')
    def save():
        summary.update(source_step=hub.robot.steps,robot_actions=hub.robot.steps-initial_step,
            wall_s=time.perf_counter()-started)
        (folder/'batch.json').write_text(json.dumps(public(summary),indent=2))
        (hub.run/'scene_catalog/catalog_after_acquisition.json').write_text(json.dumps(public(catalog),indent=2))
    hub.emit({'event':'guided_observation_started','source_step':initial_step,'needs':len(goals)})
    for goal in goals:
        components=[catalog['components'][key] for key in goal['component_ids']]
        record={**goal,'acquisition_status':'reused_ready_for_review','views':[],'semantic_status':'unconfirmed'}
        summary['attempts'].append(record)
        # Whole-table baseline already provides complementary outside views.
        # Internal needs get a focused batch; a horizontal candidate alone is
        # never declared semantically fulfilled.
        if not goal['requested']:continue
        prior_views=[]
        for shot in range(max_views_per_need):
            if summary['captured_views']>=max_views:
                record['acquisition_status']='budget_exhausted';break
            state=hub.robot.state();wrist=hub.robot.camera('robot0_eye_in_hand')
            clearance=ObservedClearance(hub.executor.ik)
            for camera in ('agentview','robot0_eye_in_hand'):clearance.add_frame(hub.robot.camera(camera),state)
            clearance.build()
            origin,rotation=hub.executor.ik.fk(state['q'])
            bounds=hand_bounds_in_eef(hub.executor.ik,origin,rotation)
            if bounds is None:raise ValueError('Complete robot hand geometry unavailable for complementary observation')
            before_quality=measured_patch_quality(components)
            candidates=proposal_views(state,wrist,components,goal['need'],scene,prior_views,hand_bounds=bounds)
            view={'source_step':state['step'],'candidate_count':len(candidates),'checks':[]}
            record['views'].append(view);selected=None;route=None
            for index,candidate in enumerate(candidates[:max_ik_candidates]):
                begin=time.perf_counter()
                try:
                    # Frame from above the complete observed inventory, not
                    # only the target. This does not certify hidden clearance.
                    height=max(c['observed_bounds_world_m'][1][2] for c in catalog['components'].values())+.15
                    route=camera_route(hub,clearance,state,candidate,height)
                    view['checks'].append({'candidate_index':index,'path_ik_passed':True,
                        'wall_s':time.perf_counter()-begin});selected=candidate;break
                except (ValueError,RuntimeError) as exc:
                    if not isinstance(exc,ValueError) and type(exc).__name__!='IKError':raise
                    view['checks'].append({'candidate_index':index,'path_ik_passed':False,
                        'error':str(exc),'wall_s':time.perf_counter()-begin})
            if selected is None:
                record['acquisition_status']='unreachable';save();break
            view['selected']=selected;view['checked_route']=route
            hub.emit({'event':'guided_observation_view_selected','object_ids':goal['object_ids'],
                'need':goal['need'],'candidate':selected})
            # Camera placement need not reach manipulation precision. RGB-D
            # calibration below always comes from the actual synchronized pose.
            if len(route['waypoints'])==1:
                motion=hub.executor.move(**route['waypoints'][0],position_tolerance_m=.012)
            else:
                motion=hub.executor.move_waypoints(route['waypoints'],position_tolerance_m=.012)
            view['motion']=public(motion)
            actual,parts=hub.observe_scene();pictures.extend(parts)
            identity=actual['observation_id'];summary['observation_ids'].append(identity);summary['captured_views']+=1
            view.update(observation_id=identity,motion_converged=bool(motion['converged']))
            if not motion['converged']:
                record['acquisition_status']='motion_failed';summary['status']='stopped_after_motion_failure';save()
                return summary,pictures
            added,_=refresh_component_evidence(hub,catalog,goal['component_ids'],identity,clearance)
            scene=voxel_points(np.concatenate([scene,added]))
            previous.append(selected);prior_views.append(selected)
            fresh_patches=[p for c in components for p in c['patches'] if p['observation_id']==identity]
            after_quality=measured_patch_quality(components,identity)
            new_voxels=sum(c.get('last_new_voxels',0) for c in components)
            old_voxels=sum(c.get('last_old_voxels',0) for c in components)
            view.update(new_observed_voxels=new_voxels,old_observed_voxels=old_voxels,
                actual_horizontal_candidates=len(fresh_patches),
                before_measured_patch_quality=before_quality,actual_measured_patch_quality=after_quality,
                semantic_surface_status='A review required')
            gain=new_voxels>=max(24,.05*old_voxels) or after_quality>max(.1,1.2*before_quality)
            record['acquisition_status']='improved_ready_for_review' if gain else 'no_gain_ready_for_review'
            save();hub.emit({'event':'guided_observation_view_finished','object_ids':goal['object_ids'],**view})
            # Acquisition thresholds only stop redundant camera motions. They
            # do not certify any semantic surface or trigger manipulation.
            if after_quality>=.5 or not gain:break
    save();hub.emit({'event':'guided_observation_finished',**public(summary)})
    return summary,pictures
