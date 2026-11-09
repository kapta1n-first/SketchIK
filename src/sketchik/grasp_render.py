"""Render saved grasp poses as native gripper meshes; no robot actuation."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def camera_pose(eye, target):
    z=np.asarray(eye)-target;z=z/np.linalg.norm(z)
    x=np.cross([0,0,1],z)
    if np.linalg.norm(x)<1e-6:x=np.cross([0,1,0],z)
    x/=np.linalg.norm(x);y=np.cross(z,x)
    pose=np.eye(4);pose[:3,:3]=np.column_stack([x,y,z]);pose[:3,3]=eye
    return pose


def caption(image, lines):
    picture=Image.fromarray(image[:,:,:3]).convert('RGB');draw=ImageDraw.Draw(picture)
    font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font=ImageFont.truetype(font_path,20)
    draw.rectangle((0,0,picture.width,32*len(lines)+8),fill=(245,247,250))
    for i,text in enumerate(lines):draw.text((10,6+32*i),text,fill=(25,30,40),font=font)
    return picture


def native_visual_parts(ik, seed):
    import mujoco
    import trimesh
    from .grasp_screen import transform_at
    ik.fk(seed);m,d=ik.model,ik.data
    root=m.body_parentid[m.site_bodyid[ik.site_id]]
    base=transform_at(d.xpos[root],d.xmat[root].reshape(3,3))
    meshes=[];pads=[]
    for geom in range(m.ngeom):
        parent=m.geom_bodyid[geom]
        while parent and parent!=root:parent=m.body_parentid[parent]
        name=m.geom(geom).name or ''
        if parent!=root or not (m.geom_group[geom]==1 or 'pad_collision' in name):continue
        kind=m.geom_type[geom]
        if kind==mujoco.mjtGeom.mjGEOM_MESH:
            j=m.geom_dataid[geom];a,n=m.mesh_vertadr[j],m.mesh_vertnum[j];f,nf=m.mesh_faceadr[j],m.mesh_facenum[j]
            mesh=trimesh.Trimesh(m.mesh_vert[a:a+n].copy(),m.mesh_face[f:f+nf].copy(),process=False)
        elif kind==mujoco.mjtGeom.mjGEOM_BOX:mesh=trimesh.creation.box(2*m.geom_size[geom])
        else:continue
        mesh.apply_transform(np.linalg.inv(base)@transform_at(d.geom_xpos[geom],d.geom_xmat[geom].reshape(3,3)))
        if 'pad' in name:color=[.08,.09,.11,1.]
        elif 'finger1' in name:color=[.03,.66,.95,1.]
        elif 'finger2' in name:color=[1.,.47,.08,1.]
        else:color=[.63,.67,.73,1.]
        meshes.append((mesh,color))
        if 'pad_collision' in name:
            pads.append((np.linalg.inv(base)@np.r_[d.geom_xpos[geom],1])[:3])
    return meshes,np.array(pads)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--screen',type=Path,required=True)
    parser.add_argument('--gpu-uuid',required=True)
    parser.add_argument('--peer-pid',type=int,action='append',default=[])
    parser.add_argument('--output',type=Path)
    parser.add_argument('--stills-only',action='store_true')
    args=parser.parse_args();root=Path(__file__).resolve().parents[2]
    from .runtime import configure
    gpu=configure(root,args.gpu_uuid,'pro',args.peer_pid)
    os.environ['EGL_DEVICE_ID']=str(gpu['egl_device'])
    import pyrender
    import imageio.v2 as imageio
    from .kinematics import RobotIK
    from .grasp_screen import robot_gripper
    start=time.perf_counter();output=args.output or root/'runs'/(datetime.now().strftime('%Y%m%dT%H%M%S')+'_grasp_render');output.mkdir()
    result=json.loads((args.screen/'result.json').read_text());preview=Path(result['source_preview']);observation=Path(result['source_observation'])
    candidates=np.load(preview/'candidates.npz');grasps,scores=candidates['grasps'],candidates['scores']
    state=json.loads((observation/'initial_state.json').read_text());meta=json.loads((observation/'metadata.json').read_text())
    ik=RobotIK((observation/'robot_only.xml').read_text(),meta['joint_names'],meta['eef_site'])
    _,_,calibration=robot_gripper(ik,state['q']);parts,pads=native_visual_parts(ik,state['q'])
    selected=[rank-1 for rank in result['retained_original_ranks']]
    if not selected:raise RuntimeError('No retained poses to show')
    def create_scene(cloud=False):
        scene=pyrender.Scene(bg_color=[.96,.97,.99,0],ambient_light=[.45,.45,.45])
        nodes=[]
        for mesh,color in parts:
            material=pyrender.MetallicRoughnessMaterial(baseColorFactor=color,metallicFactor=.2,roughnessFactor=.6,doubleSided=True)
            nodes.append(scene.add(pyrender.Mesh.from_trimesh(mesh,material=material,smooth=False)))
        if cloud:
            # Preserve only observed points; do not substitute a GT object mesh.
            obj=candidates['points'];rgb=candidates['colors']
            scene.add(pyrender.Mesh.from_points(obj,colors=rgb))
            full=np.load(args.screen/'full_observed_scene.npz');xyz=full['points'];colors=full['colors']
            centre=np.median(obj,axis=0)
            use=(np.linalg.norm(xyz[:,:2]-centre[:2],axis=1)<.16)&(xyz[:,2]<np.quantile(obj[:,2],.02)+.005)&(~full['is_target'])
            ids=np.flatnonzero(use)
            if len(ids)>120000:ids=np.random.default_rng(0).choice(ids,120000,replace=False)
            scene.add(pyrender.Mesh.from_points(xyz[ids],colors=colors[ids]))
        return scene,nodes
    render_timings={'photo':[],'candidate':[],'animation':[]}
    def timed_render(renderer, scene, bucket, **kwargs):
        begin=time.perf_counter();result=renderer.render(scene,**kwargs)
        render_timings[bucket].append(time.perf_counter()-begin)
        return result
    def pose_nodes(scene,nodes,g):
        for node in nodes:scene.set_pose(node,g)
    # Calibrated projection onto original photos, with real depth occlusion.
    overlay_scene,nodes=create_scene()
    renderer=pyrender.OffscreenRenderer(1024,1024,point_size=2.)
    photo_panels={index:[] for index in selected}
    for camera_name in ['agentview','robot0_eye_in_hand']:
        rgb=np.asarray(Image.open(observation/(camera_name+'.png')).convert('RGB'))
        frame=np.load(observation/(camera_name+'.npz'));K=frame['K'];T=frame['world_from_camera']@np.diag([1,-1,-1,1])
        cam=overlay_scene.add(pyrender.IntrinsicsCamera(K[0,0],K[1,1],K[0,2],K[1,2],znear=.01,zfar=5.),pose=T)
        light=overlay_scene.add(pyrender.DirectionalLight(color=np.ones(3),intensity=2.),pose=T)
        mask_path=preview/camera_name/'selected_mask.png'
        target=np.asarray(Image.open(mask_path))>0 if mask_path.exists() else np.zeros(rgb.shape[:2],dtype=bool)
        for index in selected:
            pose_nodes(overlay_scene,nodes,grasps[index])
            image,depth=timed_render(renderer,overlay_scene,'photo',flags=pyrender.RenderFlags.RGBA)
            visible=(depth>0)&((~np.isfinite(frame['depth_m']))|(depth<=frame['depth_m']+.002))
            composed=rgb.copy();composed[visible]=image[visible,:3]
            ys,xs=np.nonzero(visible|target)
            if len(xs):
                box=(max(0,int(xs.min())-65),max(0,int(ys.min())-65),min(1024,int(xs.max())+65),min(1024,int(ys.max())+65))
            else:box=(0,0,1024,1024)
            crop=Image.fromarray(composed).crop(box)
            scale=min(760/crop.width,650/crop.height)
            crop=crop.resize((int(crop.width*scale),int(crop.height*scale)),Image.Resampling.LANCZOS)
            panel=Image.new('RGB',(800,740),'white');panel.paste(crop,((800-crop.width)//2,80+(650-crop.height)//2))
            panel=caption(np.asarray(panel),[f'Candidate ID {index+1} | {camera_name}', 'Solid native gripper on original RGB; depth-aware occlusion'])
            panel.save(output/f'rank_{index+1:03d}_{camera_name}.png');photo_panels[index].append(panel)
            if index==selected[0]:panel.save(output/(camera_name+'_photo_overlay.png'))
        overlay_scene.remove_node(cam);overlay_scene.remove_node(light)
    renderer.delete()
    for index,panels in photo_panels.items():
        photos=Image.new('RGB',(1600,740),'white')
        for i,panel in enumerate(panels):photos.paste(panel,(800*i,0))
        photos.save(output/f'rank_{index+1:03d}_photos.png')
        if index==selected[0]:photos.save(output/'grasp_on_original_photos.png')
    # Close-up render with native solid gripper and observed target points.
    scene,nodes=create_scene(cloud=True)
    centre=np.median(candidates['points'],axis=0);g=grasps[selected[0]]
    all_vertices=np.concatenate([mesh.vertices for mesh,_ in parts])
    def fitting_camera(pose, include_approach=False, reverse=False):
        vertices=all_vertices@pose[:3,:3].T+pose[:3,3]
        bounds=[vertices,np.quantile(candidates['points'],[.01,.99],axis=0)]
        if include_approach:bounds.append(vertices-pose[:3,2]*result['approach_m'])
        vertices=np.concatenate(bounds);low,high=vertices.min(axis=0),vertices.max(axis=0)
        focus=(low+high)/2;radius=np.linalg.norm(high-low)/2
        look=pose[:3,1]*(-1 if reverse else 1);look[2]=abs(look[2])+.30;look/=np.linalg.norm(look)
        distance=radius/np.tan(np.deg2rad(21))*1.14
        return camera_pose(focus+look*distance,focus),focus
    T,focus=fitting_camera(g,include_approach=True)
    camera=scene.add(pyrender.PerspectiveCamera(yfov=np.deg2rad(42),znear=.01,zfar=3.),pose=T)
    for eye in [centre+np.array([.3,-.2,.5]),centre+np.array([-.3,.3,.3])]:
        scene.add(pyrender.DirectionalLight(color=np.ones(3),intensity=2.),pose=camera_pose(eye,centre))
    def closing_arrows(picture, pose, view):
        world=pads@pose[:3,:3].T+pose[:3,3];mid=world.mean(axis=0)
        fy=760/(2*np.tan(np.deg2rad(21)))
        def project(point):
            point=view[:3,:3].T@(point-view[:3,3]);depth=-point[2]
            return np.array([450+fy*point[0]/depth,380-fy*point[1]/depth])
        draw=ImageDraw.Draw(picture)
        for point,color in zip(world,[(0,130,225),(225,105,0)]):
            end=point+(mid-point)/np.linalg.norm(mid-point)*.018
            p0,p1=project(point),project(end);direction=p1-p0;direction/=np.linalg.norm(direction)
            side=np.array([-direction[1],direction[0]])
            draw.line([tuple(p0),tuple(p1)],fill=color,width=5)
            draw.polygon([tuple(p1),tuple(p1-direction*13+side*6),tuple(p1-direction*13-side*6)],fill=color)
        return picture
    renderer=pyrender.OffscreenRenderer(900,760,point_size=3.)
    panels=[]
    for index in selected:
        pose_nodes(scene,nodes,grasps[index]);view,_=fitting_camera(grasps[index]);scene.set_pose(camera,view);image,_=timed_render(renderer,scene,'candidate')
        pic=caption(image,[f'Original rank {index+1} | model score {scores[index]:.3f}', 'Blue/orange = two fingers; grey = palm; black = pads', 'Observed target only; pose preview, no grasp physics'])
        pic=closing_arrows(pic,grasps[index],view)
        pic.save(output/f'rank_{index+1:03d}_solid.png');panels.append(pic)
    gallery=Image.new('RGB',(900*min(4,len(panels)),760*((len(panels)+3)//4)),(245,247,250))
    for i,pic in enumerate(panels):gallery.paste(pic,((i%4)*900,(i//4)*760))
    gallery.save(output/'all_retained_grippers.png')
    # Keep exact orientation; translate from 10cm back to the saved pose, then hold.
    if not args.stills_only:
        movie=imageio.get_writer(str(output/f'rank{selected[0]+1:03d}_approach.mp4'),fps=12,codec='libx264',quality=8,macro_block_size=2)
        view,_=fitting_camera(g,include_approach=True);scene.set_pose(camera,view)
        gif_frames=[]
        for t in np.r_[np.linspace(0,1,30),np.ones(12)]:
            pose=g.copy();pose[:3,3]-=g[:3,2]*result['approach_m']*(1-t)
            pose_nodes(scene,nodes,pose);image,_=timed_render(renderer,scene,'animation')
            pic=caption(image,[f'Rank {selected[0]+1}: approach pose demonstration',f'Remaining approach: {100*result["approach_m"]*(1-t):.1f} cm', 'Fingers stay open. No closure, contact or robot execution.'])
            movie.append_data(np.asarray(pic));small=pic.resize((600,507));gif_frames.append(small)
        movie.close();gif_frames[0].save(output/'approach.gif',save_all=True,append_images=gif_frames[1:],duration=83,loop=0)
    renderer.delete()
    (output/'render_metadata.json').write_text(json.dumps({'screen_source':str(args.screen.resolve()),'gpu':gpu,
        'native_gripper_calibration':calibration,'displayed_original_ranks':[i+1 for i in selected],
        'object_geometry':'observed colored RGB-D points only, no hidden mesh','robot_actions':0,
        'render_type':'native visual gripper meshes and original RGB with depth occlusion','animation_created':not args.stills_only,
        'fingers_close':False,'contact_simulated':False,'render_frame_wall_s':render_timings,
        'frame_timing_scope':'OffscreenRenderer.render including GPU rendering and CPU readback; excludes pose/camera setup, caption, file encoding and saving',
        'render_wall_s':time.perf_counter()-start},indent=2))
    print(json.dumps({'output':str(output),'wall_s':time.perf_counter()-start,'robot_actions':0},indent=2))


if __name__=='__main__':main()
