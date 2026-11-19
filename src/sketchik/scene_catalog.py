"""Whole-table measured components and source regions for task-conditioned views.

All components remain in the scene inventory. A binds semantic objects to one
or more groups; grouping itself does not identify objects or certify free space.
"""
from copy import deepcopy
from itertools import product
import json
from pathlib import Path
import time
import shutil

import numpy as np

from .cloud_extents import table_components
from .tools import obj


def build_scene_catalog(hub):
    from .perception import saved_frame
    from .surface_evidence import summarize_horizontal_surfaces
    from .cloud_selection import write_views
    started=time.perf_counter()
    table=json.loads((hub.run/'table_context/context.json').read_text())
    summary,labels,points,pixels,sids=table_components(table,hub.run/'table_context')
    folder=hub.run/'scene_catalog';folder.mkdir(exist_ok=True)
    components={f"G{item['group_id']}":{'component_id':f"G{item['group_id']}",
        'group_id':item['group_id'],'observed_bounds_world_m':item['bounds_world_m'],
        'observed_span_xyz_m':item['span_world_m'],'point_count':item['point_count'],
        'sources':[],'patches':[],'semantic_identity_verified':False,'complete_dimensions_known':False}
        for item in summary['components']}
    candidates={}
    # Catalog IDs and row labels refer to this immutable initial baseline.
    # The final whole-table artifact can subsequently incorporate new shots.
    baseline=folder/'baseline_table.npz'
    shutil.copyfile(table['point_cloud_file'],baseline)
    for component in components.values():
        mask=labels==component['group_id']
        path=folder/(component['component_id']+'_points.npz')
        np.savez_compressed(path,world_xyz_m=points[mask])
        component['point_cloud_file']=str(path)
    for source_index,source in enumerate(table['sources']):
        selected=sids==source_index
        uv=pixels[selected];xyz=points[selected];groups=labels[selected]
        camera_short='front' if source['camera']=='agentview' else 'wrist'
        for group in np.unique(groups[groups>0]):
            mask=groups==group
            if mask.sum()<8:continue
            low=uv[mask].min(axis=0);high=uv[mask].max(axis=0)+2
            candidate_id=f"G{group}_{source['observation_id']}_{camera_short}"
            ref={'candidate_id':candidate_id,**source,'bounds_xyxy':[*low.tolist(),*high.tolist()],
                'kind':'component_source','point_count':int(mask.sum()),'semantic_identity':'unconfirmed'}
            candidates[candidate_id]=ref;components[f'G{group}']['sources'].append(ref)
        frame=saved_frame(hub.run,source['observation_id'],source['camera'])
        patches=summarize_horizontal_surfaces(frame)['horizontal_patches']
        for index,patch in enumerate(patches):
            x0,y0,x1,y1=patch['bounds_xyxy']
            inside=(uv[:,0]>=x0)&(uv[:,0]<x1)&(uv[:,1]>=y0)&(uv[:,1]<y1)&(
                abs(xyz[:,2]-patch['world_height_m'])<.012)&(groups>0)
            ids,counts=np.unique(groups[inside],return_counts=True)
            if not len(ids) or counts.max()<8:continue
            group=int(ids[np.argmax(counts)])
            if counts.max()<.6*counts.sum():continue
            actual=xyz[inside&(groups==group)]
            key=f"P_{source['observation_id']}_{camera_short}_{index}"
            ref={'candidate_id':key,**source,**patch,'component_id':f'G{group}',
                'kind':'horizontal_patch','look_at_world_m':np.median(actual,axis=0).tolist(),
                'observed_span_xyz_m':np.ptp(actual,axis=0).tolist(),
                'semantic_identity':'unconfirmed; horizontal does not mean internal support'}
            candidates[key]=ref;components[f'G{group}']['patches'].append(ref)
    for component in components.values():
        component['sources'].sort(key=lambda source:source['point_count'],reverse=True)
        component['source_steps']=sorted({source['source_step'] for source in component['sources']})
        component['last_observed_step']=max(component['source_steps'])
        # Keep a few distinct measured levels/regions, rather than all repeats
        # of the same plane. Selection is geometric, with no floor assignment.
        kept=[]
        for patch in sorted(component['patches'],key=lambda value:value['visible_pixel_sample_count'],reverse=True):
            if any(abs(patch['world_height_m']-v['world_height_m'])<.015 and
                np.linalg.norm(np.asarray(patch['look_at_world_m'])-v['look_at_world_m'])<.06 for v in kept):continue
            kept.append(patch)
        component['patches']=kept[:6]
    table_height = table['observed_table_height_m']
    # Broad, low fragments can be parts of the tabletop. Keep them in the
    # full view and use the raised groups to frame the two detail views.
    component_bounds = [item['observed_bounds_world_m'] for item in components.values()]
    raised_bounds = [bounds for bounds in component_bounds if bounds[1][2] > table_height + .04]
    detail_viewports = None
    if raised_bounds:
        low = np.min([bounds[0] for bounds in raised_bounds], axis=0) - [.06,.06,.02]
        high = np.max([bounds[1] for bounds in raised_bounds], axis=0) + [.06,.06,.02]
        viewport = np.array(list(product(*zip(low, high))))
        detail_viewports = {'opposite': viewport, 'top': viewport}
    views=write_views(table['point_cloud_file'],table['sources'],folder,'whole_scene',(1,-1,.8),
        role='all observed tabletop components, including non-task objects',selectable=False,
        group_ids=labels,directions=[
            ('overview',(1,-1,.8)),
            ('opposite',(-1,1,.8)),
            ('top',(0,0,1)),
        ],viewports=detail_viewports)
    report={'source_step':hub.robot.steps,'source_steps':table['source_steps'],
        'table_height_m':table['observed_table_height_m'],'components':components,
        'candidates':candidates,'views':views,'all_components_retained':True,
        'component_labels_file':views[0]['point_group_ids_file'],
        'point_cloud_file':str(baseline),'table_sources':table['sources'],
        'catalog_version':1,'wall_s':time.perf_counter()-started,
        'scope':'All observed tabletop geometry, including unbound objects and ungrouped points. Touching objects may merge, occlusion may split them; no hidden completion or collision certification.'}
    (folder/'catalog.json').write_text(json.dumps(report,indent=2))
    hub.emit({'event':'whole_scene_catalog_prepared','components':len(components),
        'source_step':hub.robot.steps,'wall_s':report['wall_s'],'all_components_retained':True})
    return report


def refresh_component_evidence(hub,catalog,component_ids,observation_id,robot_filter):
    """Add actual new surfaces inside known component context windows.

    An old-surface correspondence mask would erase newly revealed interiors;
    these are spatial context windows, explicitly not semantic object masks.
    """
    from .scene_cloud import frame_points
    from .surface_evidence import summarize_horizontal_surfaces
    from .perception import saved_frame
    updated={};scene=[]
    state=hub.robot.state()
    for key in component_ids:
        catalog['components'][key].update(last_new_voxels=0,last_old_voxels=0)
    for camera in ('agentview','robot0_eye_in_hand'):
        frame=saved_frame(hub.run,observation_id,camera)
        xyz,rgb,uv=frame_points(frame)
        mask=robot_filter.non_robot_mask(xyz,state);xyz=xyz[mask];uv=uv[mask]
        scene.append(xyz)
        patches=summarize_horizontal_surfaces(frame)['horizontal_patches']
        for key in component_ids:
            component=catalog['components'][key]
            low,high=np.asarray(component['observed_bounds_world_m'])
            selected=np.all((xyz>=low-.03)&(xyz<=high+.03),axis=1)&(
                xyz[:,2]>catalog['table_height_m']+.006)
            if selected.sum()<32:continue
            local=xyz[selected];pixels=uv[selected]
            updated.setdefault(key,[]).append(local)
            lo=pixels.min(axis=0);hi=pixels.max(axis=0)+1
            short='front' if camera=='agentview' else 'wrist'
            source={'candidate_id':f'{key}_{observation_id}_{short}',
                'observation_id':observation_id,'camera':camera,'source_step':state['step'],
                'bounds_xyxy':[*lo.tolist(),*hi.tolist()],'kind':'component_source',
                'point_count':int(len(local)),'semantic_identity':'unconfirmed spatial context'}
            component['sources'].append(source);catalog['candidates'][source['candidate_id']]=source
            for index,patch in enumerate(patches):
                x0,y0,x1,y1=patch['bounds_xyxy']
                match=(pixels[:,0]>=x0)&(pixels[:,0]<x1)&(pixels[:,1]>=y0)&(
                    pixels[:,1]<y1)&(abs(local[:,2]-patch['world_height_m'])<.012)
                if match.sum()<32:continue
                measured=local[match]
                ref={'candidate_id':f'P_{key}_{observation_id}_{short}_{index}',
                    'component_id':key,'observation_id':observation_id,'camera':camera,
                    'source_step':state['step'],**patch,'kind':'horizontal_patch',
                    'look_at_world_m':np.median(measured,axis=0).tolist(),
                    'observed_span_xyz_m':np.ptp(measured,axis=0).tolist(),
                    'semantic_identity':'unconfirmed; horizontal does not mean internal support'}
                component['patches'].append(ref);catalog['candidates'][ref['candidate_id']]=ref
    for key,parts in updated.items():
        component=catalog['components'][key]
        with np.load(component['point_cloud_file']) as data:old=data['world_xyz_m']
        combined=np.concatenate([old,*parts]);_,indices=np.unique(np.floor(combined/.006),axis=0,return_index=True)
        points=combined[indices]
        component['last_new_voxels']=max(0,len(points)-len(np.unique(np.floor(old/.006),axis=0)))
        component['last_old_voxels']=len(np.unique(np.floor(old/.006),axis=0))
        path=hub.run/'scene_catalog'/f'{key}_{observation_id}_points.npz'
        np.savez_compressed(path,world_xyz_m=points)
        component.update(point_cloud_file=str(path),point_count=len(points),
            observed_bounds_world_m=[points.min(axis=0).tolist(),points.max(axis=0).tolist()],
            observed_span_xyz_m=np.ptp(points,axis=0).tolist(),
            source_steps=sorted(set(component['source_steps'])|{state['step']}),last_observed_step=state['step'])
    catalog['source_steps']=sorted(set(catalog['source_steps'])|{state['step']})
    catalog['source_step']=state['step']
    robot_filter.set_state(state)
    return np.concatenate(scene),updated


def public_catalog(catalog,*,objects=(),journal=None):
    groups=[]
    for item in catalog['components'].values():
        entry={key:item[key] for key in ('component_id','observed_bounds_world_m',
            'observed_span_xyz_m','point_count','source_steps','last_observed_step',
            'semantic_identity_verified','complete_dimensions_known')}
        owners=[obj for obj in objects if item['component_id'] in obj.get('component_ids',[])]
        entry['semantic_aliases']=[obj['object_id'] for obj in owners]
        records=(journal or {}).get('objects',{})
        invalidated=[records[obj['object_id']]['invalidated_at_step'] for obj in owners
            if records.get(obj['object_id'],{}).get('invalidated_at_step') is not None]
        entry['pose_validity']='uncertain_after_contact' if invalidated else 'last_observed; unchanged pose assumed, not verified'
        if invalidated:entry['invalidated_at_step']=min(invalidated)
        groups.append(entry)
    return {'source_step':catalog['source_step'],'source_steps':catalog['source_steps'],
        'catalog_version':catalog.get('catalog_version',1),
        'scope':catalog['scope'],'all_components_retained':True,'components':groups,
        'collision_free_certified':False,
        'staleness_policy':'Historical bounds are not updated poses. A shared group is uncertain if any bound part moved. Unbound objects may also have been bumped; use current RGB.'}


def binding_schema(catalog):
    text={'type':'string','minLength':1}
    source=obj({'observation_id':{'enum':sorted({v['observation_id'] for v in catalog['table_sources']})},
        'camera':{'enum':['agentview','robot0_eye_in_hand']},
        'bounds_xyxy':{'type':'array','items':{'type':'integer','minimum':0,'maximum':1024},
            'minItems':4,'maxItems':4}},['observation_id','camera','bounds_xyxy'])
    item=obj({'object_id':text,'component_ids':{'type':'array','items':{'enum':sorted(catalog['components'])},
        'uniqueItems':True},'source_regions':{'type':'array','items':source,'minItems':1,
            'description':'Optional coarse native-image regions only when G groups omit an essential part. Not contact points.'},'description':text,
        'interactions':{'type':'array','items':text,'minItems':1},
        'shape_motion':{'enum':['rigid','articulated','unknown'],
            'description':'Optional claim about this referenced point-set scope, not its name. rigid keeps shape and changes pose; articulated has relative part motion; omission is unknown.'},
        'required_surfaces':{'type':'array','items':{'enum':['overall_shape','contact_surface','opening','interior_support']},
            'minItems':1,'uniqueItems':True},
        'missing_surfaces':{'type':'array','items':{'enum':['overall_shape','contact_surface','opening','interior_support']},
            'uniqueItems':True,'description':'Surfaces that still need a complementary real view. Empty means reuse baseline for A review.'}},
        ['object_id','component_ids','description','interactions','required_surfaces'])
    item['anyOf']=[{'properties':{'component_ids':{'minItems':1}}},
        {'required':['source_regions']}]
    return obj({'source_step':{'const':catalog['source_step']},
        'objects':{'type':'array','items':item,'minItems':1}},['source_step','objects'])


def bind_requirements(requirements,catalog):
    """Write source regions mechanically; only A names identities and needs."""
    objects=[];ids=set()
    for item in requirements['objects']:
        if item['object_id'] in ids:raise ValueError('Duplicate task object identity')
        ids.add(item['object_id']);sources=[];component_ids=list(item['component_ids'])
        for region in item.get('source_regions',[]):
            x0,y0,x1,y1=region['bounds_xyxy']
            if x1<=x0 or y1<=y0:raise ValueError('Fallback source region is empty')
            source_index=next((index for index,source in enumerate(catalog['table_sources']) if
                (source['observation_id'],source['camera'])==(region['observation_id'],region['camera'])),None)
            if source_index is None:raise ValueError('Fallback region must use an actual baseline source')
            with np.load(catalog['point_cloud_file']) as data:
                uv=data['pixel_xy'];xyz=data['world_xyz_m']
                mask=(data['source_index']==source_index)&(uv[:,0]>=x0)&(uv[:,0]<x1)&(
                    uv[:,1]>=y0)&(uv[:,1]<y1)&(xyz[:,2]>catalog['table_height_m']+.006)
                points=xyz[mask]
            if len(points)<32:raise ValueError('Fallback region has insufficient observed above-table points')
            key=f"F{len(catalog['components'])+1}"
            path=Path(catalog['point_cloud_file']).parent/(key+'_points.npz')
            np.savez_compressed(path,world_xyz_m=points)
            step=catalog['table_sources'][source_index]['source_step']
            ref={**region,'candidate_id':key+'_source','source_step':step,'kind':'component_source',
                'point_count':len(points),'semantic_identity':'A coarse region; no object mask'}
            catalog['components'][key]={'component_id':key,'observed_bounds_world_m':[
                points.min(axis=0).tolist(),points.max(axis=0).tolist()],
                'observed_span_xyz_m':np.ptp(points,axis=0).tolist(),'point_count':len(points),
                'point_cloud_file':str(path),'sources':[ref],'patches':[],
                'source_steps':[step],'last_observed_step':step,
                'semantic_identity_verified':False,'complete_dimensions_known':False}
            catalog['candidates'][ref['candidate_id']]=ref;component_ids.append(key)
        for key in component_ids:
            component=catalog['components'][key]
            for ref in component['sources'][:3]:
                source={k:deepcopy(ref[k]) for k in ('observation_id','camera','bounds_xyxy')}
                source.update(surface_role='overall_shape',visible_evidence=
                    'Program-projected source of '+key+'; semantic operation surface awaits A confirmation.')
                if source not in sources:sources.append(source)
        if not sources:raise ValueError('Selected component has no recorded RGB-D source')
        objects.append({'object_id':item['object_id'],'description':item['description'],
            'shape_motion':item.get('shape_motion','unknown'),
            'component_ids':component_ids,'interactions':item['interactions'],
            'required_surfaces':item['required_surfaces'],'needed_geometry':', '.join(item['required_surfaces']),
            'acquisition_needs':item.get('missing_surfaces',[
                role for role in item['required_surfaces'] if role=='interior_support']),
            'evidence':sources,'missing':['A confirmation of '+v for v in item['required_surfaces'] if v!='overall_shape']})
    return {'source_step':requirements['source_step'],'objects':objects,'coverage_complete':False,
        'whole_task_coverage':'Components bound; program acquires evidence, A confirms semantic surfaces and first sketch next.'}


def catalog_candidates(catalog,component_ids=None):
    chosen=set(catalog['components'] if component_ids is None else component_ids)
    result={}
    for key in chosen:
        component=catalog['components'][key]
        for ref in [*component['sources'][:3],*component['sources'][-2:],*component['patches']]:
            result[ref['candidate_id']]={k:deepcopy(v) for k,v in ref.items()
                if k not in ('look_at_world_m','point_count')}
    return result
