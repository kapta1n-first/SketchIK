"""Selectable orthographic projections of real observed points, with exact provenance."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _draw_metric_legend(image, mapping, right, down, forward, scale):
    """Screen directions are computed from the projection, never guessed by B."""
    size=image.width
    if size<512:
        return {'world_positive_axes_screen_xy':{name:[float(axis@right),float(axis@down)]
            for name,axis in zip('XYZ',np.eye(3))},'pixels_per_metre':float(scale),
            'legend_pixel_bounds':None,'annotation_pixels_selectable':False}
    width,height=220,155
    panels=[(10,50),(size-width-10,50),(10,size-height-30),(size-width-10,size-height-30)]
    x,y=min(panels,key=lambda p:np.count_nonzero(mapping[p[1]:p[1]+height,p[0]:p[0]+width]>=0))
    mapping[y:y+height,x:x+width]=-1
    draw=ImageDraw.Draw(image)
    draw.rectangle((x,y,x+width-1,y+height-1),fill=(245,245,245),outline=(160,160,160))
    draw.text((x+8,y+5),'WORLD directions / metres',fill='black')
    origin=np.array([x+102,y+72],float)
    directions={}
    for axis,color,name in zip(np.eye(3),[(195,35,35),(20,125,45),(35,65,205)],['X','Y','Z']):
        projected=np.array([axis@right,axis@down])
        directions[name]=projected.tolist()
        end=origin+43*projected
        if np.linalg.norm(projected)<.05:
            draw.text((x+8,y+99),f'+{name}: '+('toward viewer' if axis@forward<0 else 'away from viewer'),fill=color)
            continue
        draw.line([tuple(origin),tuple(end)],fill=color,width=3)
        unit=projected/np.linalg.norm(projected);perp=np.array([-unit[1],unit[0]])
        draw.polygon([tuple(end),tuple(end-8*unit+4*perp),tuple(end-8*unit-4*perp)],fill=color)
        label=end+7*unit
        draw.text(tuple(label),f'+{name}',fill=color,anchor='mm')
    # A 1/2/5 metric scale in the image plane; length is exact for this
    # orthographic projection, independent of scene depth.
    ideal=130/scale
    power=10.**np.floor(np.log10(ideal))
    length=max(v*power for v in (1,2,5) if v*power<=ideal)
    start=(x+12,y+126);end=(start[0]+length*scale,start[1])
    draw.line([start,end],fill='black',width=3)
    for px in (start[0],end[0]):draw.line([(px,start[1]-4),(px,start[1]+4)],fill='black',width=2)
    draw.text((x+12,y+137),f'{length:g} m (image-plane scale)',fill='black')
    return {'world_positive_axes_screen_xy':directions,'pixels_per_metre':float(scale),
        'scale_bar_m':float(length),'legend_pixel_bounds':[x,y,x+width,y+height],
        'annotation_pixels_selectable':False}


def rasterize(points, rgb, mode='height', eye=(0, 0, 1), size=1024, viewport_points=None,
              footprint_pixels=3):
    """Orthographic z-buffer: each painted pixel indexes one actual input point."""
    if type(footprint_pixels) is not int or footprint_pixels < 1 or footprint_pixels % 2 != 1:
        raise ValueError('Point footprint must be a positive odd integer')
    radius = footprint_pixels // 2
    points = np.asarray(points)
    if len(points) < 30 or not np.isfinite(points).all():
        raise ValueError('Insufficient finite observed points')
    forward = -np.asarray(eye, float)
    if not np.isfinite(forward).all() or np.linalg.norm(forward)<1e-8:
        raise ValueError('Virtual viewing direction must be finite and nonzero')
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.])
    if np.linalg.norm(right) < 1e-6: right = np.array([1.,0.,0.])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    horizontal, vertical, depth = points @ right, points @ down, points @ forward
    viewport = points if viewport_points is None else np.asarray(viewport_points)
    viewport_x,viewport_y=viewport@right,viewport@down
    x0,x1,y0,y1 = viewport_x.min(),viewport_x.max(),viewport_y.min(),viewport_y.max()
    scale = (size-120) / max(x1-x0,y1-y0,1e-6)
    x = np.rint((horizontal-(x0+x1)/2)*scale+size/2).astype(int)
    y = np.rint((vertical-(y0+y1)/2)*scale+size/2+20).astype(int)
    # First retain the nearest point at each projected centre, then compare
    # the shifted centre maps. A farther point at the same centre has
    # exactly the same footprint and can never win any output pixel. This is
    # the same z-buffer without sorting copies of a large point table.
    valid=(x>=-radius)&(x<size+radius)&(y>=45-radius)&(y<size-25+radius)
    source=np.flatnonzero(valid).astype(np.int32)
    padded_size=size+2*radius
    pixel=(y[source]+radius)*padded_size+x[source]+radius
    order=np.lexsort((source,depth[source],pixel))
    centre=np.full(padded_size*padded_size,-1,np.int32)
    if len(order):
        ordered=pixel[order]
        first=np.r_[True,ordered[1:]!=ordered[:-1]]
        chosen=order[first]
        centre[pixel[chosen]]=source[chosen]
    centre=centre.reshape(padded_size,padded_size)
    index_map=np.full((size,size),-1,np.int32)
    best_depth=np.full((size,size),np.inf)
    for dx in range(-radius,radius+1):
        for dy in range(-radius,radius+1):
            candidate=centre[radius-dy:radius-dy+size,radius-dx:radius-dx+size]
            visible=candidate>=0
            candidate_depth=np.full((size,size),np.inf)
            candidate_depth[visible]=depth[candidate[visible]]
            chosen=visible & ((candidate_depth<best_depth) |
                ((candidate_depth==best_depth)&((index_map<0)|(candidate<index_map))))
            index_map[chosen]=candidate[chosen]
            best_depth[chosen]=candidate_depth[chosen]
    index_map[:45]=-1
    index_map[size-25:]=-1
    picture=np.full((size,size,3),225,np.uint8)
    mapped=index_map>=0
    if mode=='rgb':
        picture[mapped]=rgb[index_map[mapped]]
    elif mode=='height':
        from matplotlib import colormaps
        low,high=points[:,2].min(),points[:,2].max()
        value=(points[:,2]-low)/max(high-low,1e-6)
        colors=np.round(colormaps['viridis'](value)[:,:3]*255).astype(np.uint8)
        picture[mapped]=colors[index_map[mapped]]
    else: raise ValueError('Unknown render mode')
    image=Image.fromarray(picture)
    draw=ImageDraw.Draw(image)
    label='OBSERVED point cloud / '+mode+' / '+('top view' if np.array_equal(eye,(0,0,1)) else 'oblique view')
    draw.text((18,12),label,fill='black')
    legend='RGB from observed surfaces.' if mode=='rgb' else 'Height: purple low, yellow high.'
    draw.text((18,size-20),'Grey = unknown. '+legend+' Each surface pixel maps to an observed point.',fill='black')
    metric=_draw_metric_legend(image,index_map,right,down,forward,scale)
    return image,index_map,{'mode':mode,'virtual_camera_eye_direction':list(eye),
        'projection':'orthographic','point_footprint_pixels':footprint_pixels,'painted_pixel_count':int(mapped.sum()),
        'point_footprint_half_width_m':float(radius/scale),
        'viewport':'all input points' if viewport_points is None else 'explicit reference points; display crop only',
        'hidden_surface_completion':False,'per_pixel_point_index_saved':True,**metric}


def annotate_groups(image, mapping, group_ids):
    """Label visible geometric groups and make every label pixel unselectable."""
    group_ids=np.asarray(group_ids)
    if group_ids.ndim!=1 or np.any(group_ids<0) or not np.issubdtype(group_ids.dtype,np.integer):
        raise ValueError('group_ids must be a one-dimensional nonnegative integer array')
    if np.any(mapping>=len(group_ids)):
        raise ValueError('group_ids must align with the point table')
    visible=mapping>=0
    visible_groups=np.zeros(mapping.shape,np.int64)
    visible_groups[visible]=group_ids[mapping[visible]]
    draw=ImageDraw.Draw(image)
    labels=[]
    for group_id in sorted(int(v) for v in np.unique(visible_groups) if v>0):
        pixels=np.argwhere(visible_groups==group_id)
        # Keep the centre surface available for pointing. Place the small tag
        # near the visible group's upper-left display extent instead.
        corner=np.min(pixels,axis=0)
        py,px=pixels[np.argmin(np.sum((pixels-corner)**2,axis=1))]
        label=f'G{group_id}'
        text_box=draw.textbbox((0,0),label)
        width=text_box[2]-text_box[0]+8
        height=text_box[3]-text_box[1]+8
        x0=int(np.clip(px-width//2,0,image.width-width))
        y0=int(np.clip(py-height//2,0,image.height-height))
        x1,y1=x0+width,y0+height
        draw.rectangle((x0,y0,x1-1,y1-1),fill=(255,255,255),outline=(0,0,0))
        draw.text((x0+4-text_box[0],y0+4-text_box[1]),label,fill=(0,0,0))
        mapping[y0:y1,x0:x1]=-1
        labels.append({'group_id':group_id,'bounds_xyxy':[x0,y0,x1,y1]})
    return labels


def write_views(cloud_file, sources, folder, prefix, eye, *, role, selectable=True,
                group_ids=None, directions=None):
    """Selected projections share an unchanged point table and exact local maps."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    cloud_file=Path(cloud_file).resolve()
    with np.load(cloud_file) as data:
        points=data['world_xyz_m'];rgb=data['rgb']
    if group_ids is not None:
        group_ids=np.asarray(group_ids)
        if group_ids.shape!=(len(points),):
            raise ValueError('group_ids must align with the point table')
        group_file=(folder/(prefix+'_groups.npz')).resolve()
        np.savez_compressed(group_file,point_group_ids=group_ids)
    if len(points)<30:return []
    eye=np.asarray(eye,float)
    if np.linalg.norm(eye)<1e-8:raise ValueError('Missing source view direction')
    angle=np.radians(35)
    turn=np.array([[np.cos(angle),-np.sin(angle),0],
                   [np.sin(angle),np.cos(angle),0],[0,0,1]])
    directions=[('face',eye),('oblique',turn@eye),('top',(0,0,1))] if directions is None else directions
    views=[]
    for name,direction in directions:
        view_id=prefix+'_'+name
        picture,mapping,meta=rasterize(points,rgb,'rgb',tuple(direction),footprint_pixels=9)
        if group_ids is not None:
            meta['component_label_bounds_xyxy']=annotate_groups(picture,mapping,group_ids)
            meta['point_group_ids_file']=str(group_file)
            meta['group_ids']=[int(value) for value in np.unique(group_ids)]
        # Labels occupy only the unmapped header, never an indexed cloud pixel.
        ImageDraw.Draw(picture).text((18,30),view_id+' / '+role,fill='black')
        image_file=(folder/(view_id+'.png')).resolve();picture.save(image_file)
        map_file=(folder/(view_id+'.npz')).resolve()
        np.savez_compressed(map_file,point_index=mapping)
        view={'view_id':view_id,'image_file':str(image_file),
            'index_map_file':str(map_file),'cloud_file':str(cloud_file),
            'sources':sources,'role':role,'selectable':bool(selectable),
            'width':1024,'height':1024,'point_count':len(points),**meta}
        views.append(view)
    (folder/(prefix+'_views.json')).write_text(json.dumps(views,indent=2))
    return views


def public_view(view):
    keys=['view_id','role','selectable','width','height','point_count','mode']
    keys.extend(key for key in ('point_footprint_pixels','point_footprint_half_width_m','pixels_per_metre')
        if key in view)
    if 'component_label_bounds_xyxy' in view:keys.append('component_label_bounds_xyxy')
    if 'group_ids' in view:keys.append('group_ids')
    for key in ('height_range_world_z_m','same_pixel_map_as','coordinate_estimate'):
        if key in view:keys.append(key)
    result={key:view[key] for key in keys}
    result['source_observations']=[{key:source[key] for key in ('observation_id','source_step','camera') if key in source}
        for source in view.get('sources',[])]
    result['group_scope']='Frozen point table of this view only; G numbers are not persistent object identities across repackaging or source steps.'
    return result


def height_auxiliary(view, folder):
    """Recolour the exact indexed RGB view; no reprojection or new geometry."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    with np.load(view['cloud_file']) as data:
        heights=data['world_xyz_m'][:,2].copy()
    with np.load(view['index_map_file']) as data:mapping=data['point_index'].copy()
    low,high=float(heights.min()),float(heights.max())
    from matplotlib import colormaps
    colors=np.round(colormaps['viridis']((heights-low)/max(high-low,1e-6))[:,:3]*255).astype(np.uint8)
    picture=np.asarray(Image.open(view['image_file']).convert('RGB')).copy()
    mapped=mapping>=0
    picture[mapped]=colors[mapping[mapped]]
    picture=Image.fromarray(picture)
    draw=ImageDraw.Draw(picture)
    # Only the already unselectable header/footer are rewritten. Group labels,
    # compass, viewport and all surface-to-point indices remain exactly aligned.
    draw.rectangle((0,0,picture.width-1,44),fill=(225,225,225))
    title = ('REGISTERED POSE ESTIMATE / world height / earlier observed samples' if view.get('coordinate_estimate')
        else 'OBSERVED point cloud / world height / same indexed RGB projection')
    draw.text((18,12),title,fill='black')
    identity=view['view_id']+'_height'
    draw.text((18,30),identity+f' / World Z: purple {low:.3f} m to yellow {high:.3f} m',fill='black')
    draw.rectangle((0,picture.height-25,picture.width-1,picture.height-1),fill=(225,225,225))
    footer = ('Grey = unknown. Z is estimated after registration; original capture times remain historical.' if view.get('coordinate_estimate')
        else 'Grey = unknown. Height is observed world Z, not semantic surface identity.')
    draw.text((18,picture.height-20),footer,fill='black')
    image_file=(folder/(identity+'.png')).resolve()
    map_file=(folder/(identity+'.npz')).resolve()
    picture.save(image_file);np.savez_compressed(map_file,point_index=mapping)
    result={**view,'mode':'height','height_range_world_z_m':[low,high],
        'same_pixel_map_as':view['view_id'],'view_id':identity,
        'image_file':str(image_file),'index_map_file':str(map_file)}
    (folder/(identity+'.json')).write_text(json.dumps(result,indent=2))
    return result


def resolve_cloud_point(anchor, allowed_views, local_views):
    """Exact displayed pixel -> unchanged point ID -> XYZ. No nearest snapping."""
    identity=anchor.get('view_id')
    allowed={v['view_id']:v for v in allowed_views}
    local={v['view_id']:v for v in local_views}
    if identity not in allowed or identity not in local:
        raise ValueError('Cloud view was not supplied to B')
    view=local[identity]
    if not allowed[identity]['selectable'] or not view['selectable']:
        raise ValueError('Historical shape is stale; select a current cloud point')
    pixel=anchor.get('pixel_xy')
    if not isinstance(pixel,list) or len(pixel)!=2 or any(type(v) is not int for v in pixel):
        raise ValueError('cloud_point requires integer pixel_xy')
    x,y=pixel
    with np.load(view['index_map_file']) as data:mapping=data['point_index']
    if not (0<=x<mapping.shape[1] and 0<=y<mapping.shape[0]):
        raise ValueError('Cloud pixel outside supplied image')
    index=int(mapping[y,x])
    if index<0:raise ValueError('Cloud pixel is unknown/background; no observed point (no snapping)')
    with np.load(view['cloud_file']) as data:
        if index>=len(data['world_xyz_m']):raise ValueError('Invalid cloud point index')
        point=data['world_xyz_m'][index].astype(float)
        provenance={'cloud_view_id':identity,'cloud_pixel_xy':pixel,'cloud_point_index':index,
            'cloud_file':view['cloud_file'],'cloud_role':view['role']}
        if view.get('coordinate_estimate'):
            provenance['coordinate_estimate'] = view['coordinate_estimate']
        if 'source_index' in data and 'pixel_xy' in data:
            source=int(data['source_index'][index])
            if source>=len(view['sources']):raise ValueError('Invalid cloud source index')
            provenance.update(source_observation=view['sources'][source],
                source_pixel_xy=data['pixel_xy'][index].tolist())
        if view.get('point_group_ids_file'):
            with np.load(view['point_group_ids_file']) as groups:
                ids=groups['point_group_ids']
            if ids.shape!=(len(data['world_xyz_m']),):
                raise ValueError('Observed group IDs do not align with the point table')
            actual=int(ids[index])
            requested=anchor.get('group_id')
            if type(requested) is not int or requested!=actual:
                raise ValueError('Cloud point group mismatch: requested group_id='+str(requested)+
                    ', actual group_id='+str(actual)+', observed world_xyz_m='+json.dumps(point.tolist())+
                    '. G0 is unlabelled/excluded geometry, not a semantic object. '
                    'Select a visible point in the intended group; no snapping or motion was performed.')
            provenance['cloud_group_id']=actual
    if not np.isfinite(point).all():raise ValueError('Nonfinite observed cloud point')
    return point,provenance
