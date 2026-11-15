"""Per-run observed geometry files, not memory or completed object models.

User-authorized dimension summaries describe observed samples only. Container
exterior bounds never stand in for cavity dimensions or certified free space.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .grasp_preview import mask_clouds, points_from_mask
from .geometry_journal import GeometryJournal
from .cloud_extents import principal_extents


def observed_summary(points):
    points=np.asarray(points,float)
    if len(points)<64: raise ValueError('Insufficient observed geometry')
    lower,upper=points.min(axis=0),points.max(axis=0)
    centre=np.median(points,axis=0)
    refs={}
    for name,quantile in [('low',.1),('middle',.5),('high',.9)]:
        threshold=np.quantile(points[:,2],quantile)
        ids=np.argsort(np.abs(points[:,2]-threshold))[:max(64,len(points)//30)]
        mean=points[ids].mean(axis=0)
        selected=ids[np.argmin(np.linalg.norm(points[ids]-mean,axis=1))]
        refs[name]=points[selected].tolist()
    return {'observed_min_world':lower.tolist(),'observed_max_world':upper.tolist(),
        'observed_span_xyz_m':(upper-lower).tolist(),'observed_median_world':centre.tolist(),
        **principal_extents(points),
        'observed_reference_points_world':refs,'point_count':len(points),
        'complete_object_dimensions_verified':False,'hidden_geometry_completed':False}


def surface_components(frame,mask):
    """Unlabelled upward observed patches. Selection of an interior remains semantic."""
    depth=frame['depth_m'].squeeze();h,w=depth.shape
    valid=np.isfinite(depth)&(depth>.02)&(depth<3)
    xyz,_=points_from_mask(frame,valid,Image.new('RGB',(w,h)))
    dense=np.full((h,w,3),np.nan);dense[valid]=xyz
    dx=dense[1:-1,2:]-dense[1:-1,:-2]
    dy=dense[2:,1:-1]-dense[:-2,1:-1]
    normal=np.cross(dx,dy);norm=np.linalg.norm(normal,axis=2)
    up=np.zeros((h,w),bool)
    up[1:-1,1:-1]=(np.abs(normal[:,:,2])/np.maximum(norm,1e-12)>.94)&(norm>1e-12)
    up &= ndimage.binary_erosion(mask)&valid
    labels,count=ndimage.label(up)
    candidates=[]
    eligible=np.flatnonzero(np.bincount(labels.ravel())>=150)
    for identity in eligible[eligible>0]:
        component=labels==identity
        if component.sum()<150:continue
        points=dense[component]
        centre=points.mean(axis=0)
        _,_,vt=np.linalg.svd(points-centre,full_matrices=False)
        n=vt[-1];n=n if n[2]>=0 else -n
        rms=float(np.sqrt(np.mean(((points-centre)@n)**2)))
        if rms>.005:continue
        y,x=np.nonzero(component)
        index=np.argmin(np.linalg.norm(points-centre,axis=1))
        candidates.append({'candidate_id':len(candidates)+1,'pixel_xy':[int(x[index]),int(y[index])],
            'world_xyz':points[index].tolist(),'normal_world':n.tolist(),
            'observed_span_xyz_m':np.ptp(points,axis=0).tolist(),'rms_m':rms,'points':len(points),
            'semantic_role_verified':False,'cavity_free_space_verified':False})
    return candidates


def build(source,output):
    output.mkdir(exist_ok=False)
    start=time.perf_counter()
    collection=json.loads((source/'collection.json').read_text())
    journal=GeometryJournal(source);journal.collection(collection)
    report={'source':str(source),'scope':'current-run observed geometry, no long-term memory',
            'robot_actions':0,'graspgen_requests':0,'object_gt_used':False,'targets':[]}
    for target in collection['targets']:
        folder=output/target['target_id'];folder.mkdir()
        record={'target':target,'sources':[],'status':'preparing','interior_surface_candidates':[]}
        points,colors=[],[]
        try:
            grouped={}
            for item in target['views']:grouped.setdefault(item['observation_id'],[]).append(item)
            for identity,items in grouped.items():
                obs=source/'observations'/identity
                directory=folder/identity;directory.mkdir()
                primary=items[0]
                region={'camera':primary['camera'],'bounds_xyxy':primary['bounds_xyxy'],
                        'description':primary['view_description'],'visible_evidence':primary['view_description']}
                _,_,selection=mask_clouds(obs,target['target_description'],{},'http://127.0.0.1:18761',directory,
                    sam_prompt=target['sam_prompt'],molmo_url='http://127.0.0.1:18762',selection_context=region)
                for item in items:
                    camera=item['camera'];cloud=directory/camera/'observed_object.npz'
                    if not cloud.exists():
                        record['sources'].append({'observation_id':identity,'camera':camera,'status':'no_selected_mask'})
                        continue
                    with np.load(cloud) as f:
                        xyz,rgb,mask=f['points'],f['colors'],f['mask']
                    points.append(xyz);colors.append(rgb)
                    record['sources'].append({'observation_id':identity,'camera':camera,'status':'selected_mask',
                                              'point_count':len(xyz),'source':str(cloud)})
                    if target['role']=='placement_target' and not record['interior_surface_candidates']:
                        with np.load(obs/(camera+'.npz')) as f:frame={key:f[key] for key in ['depth_m','K','world_from_camera']}
                        candidates=surface_components(frame,mask)
                        for candidate in candidates:candidate.update(observation_id=identity,camera=camera)
                        record['interior_surface_candidates']=candidates
                        image=Image.open(obs/(camera+'.png')).convert('RGB');draw=ImageDraw.Draw(image)
                        for candidate in candidates:
                            x,y=candidate['pixel_xy'];draw.ellipse([x-7,y-7,x+7,y+7],outline='red',width=3)
                            draw.text((x+9,y-10),str(candidate['candidate_id']),fill='red',stroke_width=1,stroke_fill='white')
                        image.save(folder/'unlabelled_surface_candidates.png')
            if not points:raise ValueError('No semantically selected observed object Mask in A-selected views')
            xyz,rgb=np.concatenate(points),np.concatenate(colors)
            np.savez_compressed(folder/'observed_points.npz',world_xyz_m=xyz,rgb=rgb)
            summary=observed_summary(xyz)
            record.update(status='observed_geometry_saved',summary=summary,
                container_bounds_are_not_cavity_dimensions=(target['role']=='placement_target'))
            # A compact 3D visualization; coordinates remain in the data file.
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            sample=np.linspace(0,len(xyz)-1,min(40000,len(xyz)),dtype=int)
            fig=plt.figure(figsize=(7,6));ax=fig.add_subplot(111,projection='3d')
            ax.scatter(*xyz[sample].T,c=rgb[sample],s=.8,depthshade=False)
            ax.set_xlabel('World X (m)');ax.set_ylabel('World Y (m)');ax.set_zlabel('World Z (m)')
            ax.set_box_aspect(np.maximum(np.ptp(xyz,axis=0),.005));ax.view_init(35,35)
            ax.set_title(target['target_id']+' / observed points only')
            fig.tight_layout();fig.savefig(folder/'cloud_3d.png',dpi=140);plt.close(fig)
        except Exception as exc:record.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        (folder/'geometry.json').write_text(json.dumps(record,indent=2))
        journal.update_object(target['target_id'],{
            'geometry_status':record['status'],'geometry_file':str(folder/'geometry.json'),
            'point_cloud_file':str(folder/'observed_points.npz') if points else None,
            'observed_geometry':record.get('summary'),
            'interior_surface_candidates':record['interior_surface_candidates']})
        report['targets'].append(record)
        (output/'result.json').write_text(json.dumps(report,indent=2))
    report.update(wall_s=time.perf_counter()-start)
    (output/'result.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'output':str(output),'wall_s':report['wall_s'],
                      'targets':[(t['target']['target_id'],t['status']) for t in report['targets']]}),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();build(args.source.resolve(),args.output.resolve())


if __name__=='__main__':main()
