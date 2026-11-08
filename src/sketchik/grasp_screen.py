"""Read-only full observed scene / gripper-approach / robot-only IK screening."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .grasp_preview import CAMERAS, points_from_mask
from .kinematics import RobotIK, IKError


def observed_scene(observation, preview):
    points, colors, origins, target = [], [], [], []
    for index, camera in enumerate(CAMERAS):
        image = Image.open(observation/(camera+'.png')).convert('RGB')
        with np.load(observation/(camera+'.npz')) as frame:
            depth=frame['depth_m'].squeeze()
            mask=np.isfinite(depth)&(depth>.02)&(depth<3.)
            xyz,rgb=points_from_mask(frame,mask,image)
            selected=preview/camera/'selected_mask.png'
            object_mask=np.asarray(Image.open(selected))>0 if selected.exists() else np.zeros_like(mask)
            if object_mask.shape!=mask.shape:raise ValueError('Mask/depth shape mismatch')
            points.append(xyz.astype(np.float32));colors.append((rgb*255).astype(np.uint8))
            origins.append(np.full(len(xyz),index,dtype=np.uint8));target.append(object_mask[mask])
    return np.concatenate(points),np.concatenate(colors),np.concatenate(origins),np.concatenate(target)


def transform_at(position, rotation):
    t=np.eye(4);t[:3,:3]=rotation;t[:3,3]=position
    return t


def robot_gripper(ik, seed):
    """Use only robot MJCF geometry; map GraspGen +Z/+X to grip_site axes."""
    import trimesh
    model,data=ik.model,ik.data
    for j in range(model.njnt):
        name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,j)
        if name and 'finger_joint' in name:
            data.qpos[model.jnt_qposadr[j]]=model.jnt_range[j,1] if model.jnt_range[j,1]>0 else model.jnt_range[j,0]
    p,r=ik.fk(seed)
    eef_body=model.site_bodyid[ik.site_id];root=model.body_parentid[eef_body]
    base=transform_at(data.xpos[root],data.xmat[root].reshape(3,3))
    site=transform_at(p,r);base_to_site=np.linalg.inv(base)@site
    descendants=[]
    for body in range(model.nbody):
        parent=body
        while parent and parent!=root:parent=model.body_parentid[parent]
        if parent==root:descendants.append(body)
    meshes=[]
    for geom in range(model.ngeom):
        if model.geom_bodyid[geom] not in descendants or model.geom_group[geom]!=0:continue
        kind=model.geom_type[geom]
        if kind==mujoco.mjtGeom.mjGEOM_MESH:
            mesh_id=model.geom_dataid[geom]
            va,vn=model.mesh_vertadr[mesh_id],model.mesh_vertnum[mesh_id]
            fa,fn=model.mesh_faceadr[mesh_id],model.mesh_facenum[mesh_id]
            mesh=trimesh.Trimesh(model.mesh_vert[va:va+vn].copy(),model.mesh_face[fa:fa+fn].copy(),process=False)
        elif kind==mujoco.mjtGeom.mjGEOM_BOX:
            mesh=trimesh.creation.box(2*model.geom_size[geom])
        else:raise ValueError(f'Unsupported gripper collision geometry {kind}')
        mesh.apply_transform(np.linalg.inv(base)@transform_at(data.geom_xpos[geom],data.geom_xmat[geom].reshape(3,3)))
        meshes.append(mesh)
    # Both conventions use +Z toward fingers and X as the finger separation axis.
    # Validate the model's measured pad separation rather than assuming Euler angles.
    pads=[i for i in range(model.ngeom) if 'pad_collision' in (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i) or '')]
    if len(pads)!=2:raise ValueError('Expected two native finger pads')
    axis=base[:3,:3].T@(data.geom_xpos[pads[0]]-data.geom_xpos[pads[1]])
    axis/=np.linalg.norm(axis)
    if abs(axis[0])<.999 or not np.allclose(base_to_site[:3,:3],np.eye(3),atol=1e-6):
        raise ValueError('Native gripper axes require an explicit convention alignment')
    return trimesh.util.concatenate(meshes),base_to_site,{'base_body':model.body(root).name,
        'base_to_grip_site':base_to_site.tolist(),'native_pad_separation_axis_in_base':axis.tolist(),
        'gripper_geometry':'native robot-only MJCF collision geoms at fully open fingers',
        'alignment':'GraspGen +Z approach/+X separation aligned to native base; symmetric X sign irrelevant to physical grasp'}


def approach_poses(grasp, length, spacing=.01):
    n=max(1,int(np.ceil(length/spacing)))
    poses=np.repeat(grasp[None],n+1,axis=0)
    distances=np.linspace(length,0,n+1)
    poses[:,:3,3]-=distances[:,None]*grasp[:3,2]
    return poses,distances


def proximity_check(tree, samples, poses, clearance):
    distances=[]
    for pose in poses:
        world=samples@pose[:3,:3].T+pose[:3,3]
        nearest=tree.query(world,k=1,workers=1,distance_upper_bound=clearance)[0]
        distances.append(float(nearest.min()))
        if distances[-1]<clearance:break
    return np.asarray(distances),bool(min(distances)<clearance)


def render(scene, colors, grasps, scores, records, object_points, mesh, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import plotly.graph_objects as go
    rng=np.random.default_rng(0)
    ids=rng.choice(len(scene),min(len(scene),60000),replace=False)
    centre=np.median(object_points,axis=0)
    pts=scene[ids]-centre;rgb=colors[ids]/255.
    wires=np.array([[.04,0,.052],[.04,0,.097],[0,0,.097],[0,0,0],[0,0,.097],[-.04,0,.097],[-.04,0,.052]])
    palette={'retained':'#16a34a','scene_collision':'#dc2626','ik_rejected':'#7c3aed'}
    fig=plt.figure(figsize=(16,7))
    for k,(elev,azim) in enumerate([(25,-65),(65,-100)]):
        ax=fig.add_subplot(1,2,k+1,projection='3d');ax.scatter(*pts.T,c=rgb,s=.4,alpha=.5,depthshade=False)
        for g,record in zip(grasps,records):
            p=wires@g[:3,:3].T+g[:3,3]-centre
            ax.plot(*p.T,color=palette[record['status']],lw=1.6 if record['status']=='retained' else .6,
                    alpha=.9 if record['status']=='retained' else .2)
        ax.set(xlim=(-.26,.26),ylim=(-.26,.26),zlim=(-.23,.25),xlabel='world X offset (m)',ylabel='world Y offset (m)',zlabel='world Z offset (m)')
        ax.set_box_aspect([1,1,1]);ax.view_init(elev,azim)
    from matplotlib.lines import Line2D
    fig.legend([Line2D([0],[0],color=palette[key],lw=3) for key in palette],['Retained by local screen','Observed-scene proximity rejected','IK rejected'],loc='lower center',ncol=3)
    fig.suptitle('All 128 original candidates on observed scene | sampled gripper/10cm approach + IK | not executed')
    fig.tight_layout(rect=[0,.04,1,.95]);fig.savefig(output/'screened_candidates.png',dpi=150);plt.close(fig)
    keep=[i for i,r in enumerate(records) if r['status']=='retained'][:3]
    if keep:
        fig=plt.figure(figsize=(5*len(keep),5))
        for panel,i in enumerate(keep):
            g=grasps[i];ax=fig.add_subplot(1,len(keep),panel+1,projection='3d')
            ax.scatter(*pts.T,c=rgb,s=.8,alpha=.6,depthshade=False)
            verts=np.asarray(mesh.vertices)@g[:3,:3].T+g[:3,3]-centre
            ax.add_collection3d(Poly3DCollection(verts[mesh.faces],facecolor=(.1,.7,.2,.35),edgecolor='none'))
            ax.set(xlim=(-.2,.2),ylim=(-.2,.2),zlim=(-.18,.22));ax.view_init(25,-65);ax.set_box_aspect([1,1,1])
            ax.set_title(f"Original rank {i+1}, score {scores[i]:.3f}")
        fig.suptitle('Retained native gripper poses | full arm path / grasp not verified')
        fig.tight_layout();fig.savefig(output/'retained_grippers.png',dpi=160);plt.close(fig)
    rgb_text=[f'rgb({r},{g},{b})' for r,g,b in colors[ids]]
    traces=[go.Scatter3d(x=pts[:,0],y=pts[:,1],z=pts[:,2],mode='markers',marker={'size':1.5,'color':rgb_text},name='Observed scene (display sample)')]
    for i,(g,record) in enumerate(zip(grasps,records)):
        p=wires@g[:3,:3].T+g[:3,3]-centre
        traces.append(go.Scatter3d(x=p[:,0],y=p[:,1],z=p[:,2],mode='lines',line={'width':5,'color':palette[record['status']]},
            name=f"Rank {i+1}: {record['status']}, score={scores[i]:.3f}",legendgroup=record['status'],visible=True))
    go.Figure(traces).update_layout(title='All original candidates; green retained / red scene proximity / purple IK; not executed',scene={'aspectmode':'data'}).write_html(output/'screened_candidates_3d.html',include_plotlyjs=True)


def main():
    import trimesh
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview',type=Path,required=True)
    parser.add_argument('--clearance-m',type=float,default=.005)
    parser.add_argument('--approach-m',type=float,default=.10)
    args=parser.parse_args()
    if not 0<args.clearance_m<=.03 or not 0<args.approach_m<=.3:parser.error('Invalid diagnostic bounds')
    root=Path(__file__).resolve().parents[2]
    output=root/'runs'/(datetime.now().strftime('%Y%m%dT%H%M%S')+'_grasp_screen');output.mkdir()
    start=time.perf_counter()
    metadata=json.loads((args.preview/'metadata.json').read_text())
    observation=Path(metadata['source']['observation_directory'])
    d=np.load(args.preview/'candidates.npz');grasps,scores=d['grasps'],d['scores']
    scene,colors,origins,target=observed_scene(observation,args.preview)
    np.savez_compressed(output/'full_observed_scene.npz',points=scene,colors=colors,camera_index=origins,is_target=target)
    tree=cKDTree(scene[~target]);cloud_wall=time.perf_counter()-start
    state=json.loads((observation/'initial_state.json').read_text());meta=json.loads((observation/'metadata.json').read_text())
    ik=RobotIK((observation/'robot_only.xml').read_text(),meta['joint_names'],meta['eef_site'])
    seed=np.array(state['q']);p,r=ik.fk(seed)
    if np.linalg.norm(p-state['position'])>1e-6 or (Rotation.from_matrix(r)*Rotation.from_quat(state['quaternion_xyzw']).inv()).magnitude()>1e-6:
        raise RuntimeError('Robot-only FK does not match saved observed robot state')
    mesh,base_to_site,calibration=robot_gripper(ik,seed)
    mesh.export(output/'native_gripper_collision.ply')
    samples,_=trimesh.sample.sample_surface(mesh,6000,seed=0)
    samples=np.vstack([samples,mesh.vertices])
    records=[];collision_wall=0.;ik_wall=0.
    for i,g in enumerate(grasps):
        begin=time.perf_counter();poses,distances=approach_poses(g,args.approach_m)
        near,colliding=proximity_check(tree,samples,poses,args.clearance_m)
        collision_wall+=time.perf_counter()-begin
        row={'original_rank':i+1,'score':float(scores[i]),'grasp_base_world':g.tolist(),
             'approach_sample_distances_m':distances[:len(near)].tolist(),
             'scene_near_distances_m':[float(v) if np.isfinite(v) else None for v in near],
             'scene_collision':colliding,'ik_checked':False,'status':'scene_collision' if colliding else 'pending'}
        if not colliding:
            begin=time.perf_counter()
            try:
                pre=poses[0]@base_to_site;goal=g@base_to_site
                sol=ik.solve(pre[:3,3],Rotation.from_matrix(pre[:3,:3]).as_quat(),seed)
                path=ik.cartesian_path(sol.q,goal[:3,3],Rotation.from_matrix(goal[:3,:3]).as_quat(),spacing_m=.01)
                row.update(status='retained',pregrasp_q=sol.q.tolist(),grasp_q=path[-1].q.tolist(),
                    position_error_m=path[-1].position_error_m,rotation_error_rad=path[-1].rotation_error_rad)
            except IKError as e:row.update(status='ik_rejected',ik_error=str(e))
            ik_wall+=time.perf_counter()-begin;row['ik_checked']=True
        records.append(row)
        with (output/'candidate_checks.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if (i+1)%16==0:print(json.dumps({'checked':i+1,'elapsed_s':time.perf_counter()-start}),flush=True)
    kept=np.array([r['status']=='retained' for r in records]);np.savez_compressed(output/'screened_grasps.npz',grasps=grasps,scores=scores,retained=kept)
    report={'source_preview':str(args.preview.resolve()),'source_observation':str(observation),'robot_actions':0,'object_gt_used':False,
        'scene_points':len(scene),'obstacle_points':int((~target).sum()),'target_points_excluded':int(target.sum()),'candidates':len(grasps),
        'scene_rejected':sum(r['status']=='scene_collision' for r in records),'ik_rejected':sum(r['status']=='ik_rejected' for r in records),'retained':int(kept.sum()),
        'retained_original_ranks':[i+1 for i,v in enumerate(kept) if v],'clearance_m':args.clearance_m,'approach_m':args.approach_m,
        'approach_spacing_m':.01,'gripper_surface_samples':len(samples),'native_gripper_calibration':calibration,
        'scene_depth_valid_range_m':[.02,3.],
        'distance_records':'null means no sampled pair closer than clearance; stop checking that candidate at first proximity hit',
        'scene_extraction_save_tree_wall_s':cloud_wall,'collision_wall_s':collision_wall,'ik_wall_s':ik_wall,'screen_wall_s':time.perf_counter()-start,
        'limitations':['Observed surfaces only; unobserved space is not certified free','Sampled gripper proximity, not continuous volumetric collision certification',
            'No full-arm scene/self-collision or home-to-pregrasp path planning','IK uses saved robot-only model and limits; not actual grasp success'],
        'source_candidates_sha256':hashlib.sha256((args.preview/'candidates.npz').read_bytes()).hexdigest()}
    (output/'result.json').write_text(json.dumps(report,indent=2))
    render(scene,colors,grasps,scores,records,d['points'],mesh,output)
    print(json.dumps({'output':str(output),**report},indent=2))


if __name__=='__main__':main()
