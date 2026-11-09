"""Render real GraspGen candidates; never execute motion or use object GT."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime
import json
from pathlib import Path
import sys
import time
import urllib.request

import numpy as np
from PIL import Image, ImageDraw


def request(url, data=None, timeout=180):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    encoded = None if data is None else json.dumps(data, allow_nan=False).encode()
    req = urllib.request.Request(url, data=encoded, headers={"Content-Type": "application/json"})
    with opener.open(req, timeout=timeout) as response:
        return json.load(response)


CAMERAS = ("agentview", "robot0_eye_in_hand")


def points_from_mask(frame, mask, image):
    depth = frame["depth_m"].squeeze()
    if depth.shape != mask.shape or np.asarray(image).shape[:2] != mask.shape:
        raise ValueError("RGB, depth and Mask resolutions must match")
    valid = mask & np.isfinite(depth) & (depth > .02) & (depth < 3.)
    y,x = np.nonzero(valid)
    rays = np.linalg.solve(frame["K"], np.vstack([x,y,np.ones(len(x))])).T
    camera_points = rays * depth[y,x,None]
    transform = frame["world_from_camera"]
    return camera_points @ transform[:3,:3].T + transform[:3,3], np.asarray(image)[y,x]/255.


def validate_mask_choice(answer, mapping):
    parsed=json.loads(answer.strip())
    if not isinstance(parsed,dict) or set(parsed)!={'selected_ids','reason'} or not isinstance(parsed['reason'],str):
        raise ValueError('Expected selected_ids and reason JSON')
    ids=parsed['selected_ids']
    if not isinstance(ids,list) or any(type(i) is not int or i not in mapping for i in ids):
        raise ValueError('Unprovided Mask ID')
    cameras=[mapping[i]['camera'] for i in ids]
    if len(cameras)!=len(set(cameras)):raise ValueError('More than one Mask selected per camera')
    return {mapping[i]['camera']:mapping[i]['local_id'] for i in ids}


def select_masks(folder, views, target, molmo_url, output):
    mapping={};items=[];next_id=1
    for camera in CAMERAS:
        image=Image.open(folder/(camera+'.png')).convert('RGB');overlay=np.array(image);labels=[]
        for c in views[camera]['candidates']:
            h,w=c['mask_shape'];mask=np.unpackbits(np.frombuffer(base64.b64decode(c['mask_packbits_base64']),dtype=np.uint8))[:h*w].reshape(h,w).astype(bool)
            if mask.shape!=overlay.shape[:2]:raise ValueError('Mask/RGB size mismatch')
            mapping[next_id]={'camera':camera,'local_id':c['id'],'score':c['score']}
            color=np.array([(next_id*83)%180+60,(next_id*53)%180+60,(next_id*113)%180+60])
            overlay[mask]=(.72*overlay[mask]+.28*color).astype(np.uint8)
            labels.append((c['box_xyxy'],next_id));next_id+=1
        picture=Image.fromarray(overlay);draw=ImageDraw.Draw(picture)
        from PIL import ImageFont
        font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',25)
        for box,identity in labels:
            box=[max(0,min(1023,float(v))) for v in box]
            draw.rectangle(box,outline='red',width=3);draw.text((box[0]+3,box[1]+3),str(identity),font=font,fill='red',stroke_width=2,stroke_fill='white')
        path=output/(camera+'_mask_choices.png');picture.save(path)
        items.append({'label':camera+' numbered SAM3 candidates','image_base64':base64.b64encode(path.read_bytes()).decode()})
    if not mapping:
        (output/'molmo_mask_selection.json').write_text(json.dumps({'selected_ids':[],'reason':'SAM returned no candidates; Molmo not called'}))
        return {},set(CAMERAS)
    prompt=(Path(__file__).resolve().parents[2]/'prompts/mask_selector.md').read_text()+'\n'+json.dumps({'target_description':target,'candidate_mapping':mapping})
    (output/'molmo_mask_input.json').write_text(json.dumps({'prompt':prompt,'image_files':[c+'_mask_choices.png' for c in CAMERAS]},indent=2))
    start=time.perf_counter();response=request(molmo_url+'/infer',{'prompt':prompt,'images':items})
    response['client_wall_s']=time.perf_counter()-start
    (output/'molmo_mask_response.json').write_text(json.dumps(response,indent=2))
    selected=validate_mask_choice(response['answer'],mapping)
    (output/'molmo_mask_selection.json').write_text(json.dumps({'selected_camera_ids':selected,'mapping':mapping},indent=2))
    return selected,set(CAMERAS)-set(selected)


def mask_clouds(folder, target, candidate_ids, sam_url, output, sam_prompt=None, molmo_url=None):
    images = {c: Image.open(folder/(c+".png")).convert("RGB") for c in CAMERAS}
    if any(image.size != (1024,1024) for image in images.values()):
        raise ValueError("Dual-view preview requires native 1024x1024 RGB-D; do not upscale saved 256 images")
    query = sam_prompt or target
    payload = {"prompt": query, "images": [{"camera": c, "image_base64": base64.b64encode(
        (folder/(c+".png")).read_bytes()).decode()} for c in CAMERAS]}
    begin = time.perf_counter()
    result = request(sam_url+"/infer", payload)
    client_wall = time.perf_counter()-begin
    (output/"sam_response.json").write_text(json.dumps(result))
    views = {v['camera']: v for v in result['images']}
    if len(result['images']) != 2 or set(views) != set(CAMERAS):
        raise ValueError("SAM response camera labels do not match the requested pair")
    rejected_views=set()
    if molmo_url:
        if candidate_ids:raise ValueError('Do not combine forced Mask IDs with semantic Molmo selection')
        candidate_ids,rejected_views=select_masks(folder,views,target,molmo_url,output)
    all_points,all_colors,origins,records = [],[],[],[]
    palette = [(255,80,40), (40,210,100), (100,100,255), (220,80,220)]
    for index,camera in enumerate(CAMERAS):
        directory=output/camera;directory.mkdir()
        image=images[camera];image.save(directory/"input_rgb.png")
        masks={};overlay=np.asarray(image).copy()
        for candidate in views[camera]['candidates']:
            h,w=candidate['mask_shape']
            if (h,w)!=(image.height,image.width):raise ValueError("Returned Mask size mismatch")
            mask=np.unpackbits(np.frombuffer(base64.b64decode(candidate['mask_packbits_base64']),dtype=np.uint8))[:h*w].reshape(h,w).astype(bool)
            masks[candidate['id']]=mask
            color=np.array(palette[candidate['id']%len(palette)])
            overlay[mask]=(.5*overlay[mask]+.5*color).astype(np.uint8)
        picture=Image.fromarray(overlay);draw=ImageDraw.Draw(picture)
        for candidate in views[camera]['candidates']:
            draw.rectangle(candidate['box_xyxy'],outline=palette[candidate['id']%len(palette)],width=3)
            draw.text(candidate['box_xyxy'][:2],f"ID {candidate['id']} score={candidate['score']:.2f}",fill='white')
        picture.save(directory/"sam_masks.png")
        selected=candidate_ids.get(camera)
        if camera in rejected_views:
            records.append({'camera':camera,'sam_candidate_count':len(masks),'status':'molmo_none','point_count':0});continue
        record={"camera":camera,"width":image.width,"height":image.height,"sam_candidate_count":len(masks)}
        if selected is None and not masks:
            record.update(status="no_detection",point_count=0);records.append(record);continue
        selection = ("molmo_mask_id" if molmo_url else "explicit_candidate_id") if selected is not None else "highest_sam_score_per_view"
        if selected is None:
            selected=max(views[camera]['candidates'],key=lambda candidate:candidate['score'])['id']
        if selected not in masks:raise ValueError(f"{camera}: selected candidate does not exist")
        mask=masks[selected]
        with np.load(folder/(camera+".npz")) as frame:
            points,colors=points_from_mask(frame,mask,image)
            np.savez_compressed(directory/"observed_object.npz",points=points,colors=colors,mask=mask,
                                K=frame['K'],world_from_camera=frame['world_from_camera'])
        Image.fromarray((mask*255).astype(np.uint8)).save(directory/"selected_mask.png")
        all_points.append(points);all_colors.append(colors);origins.append(np.full(len(points),index,dtype=np.uint8))
        selected_score=next(c['score'] for c in views[camera]['candidates'] if c['id']==selected)
        record.update(status="selected_for_preview",selected_candidate_id=selected,selection=selection,
                      selected_score=selected_score,point_count=len(points));records.append(record)
    source={"input":"native dual-view LIBERO RGB-D plus batched SAM3 masks","observation_directory":str(folder.resolve()),
        "target_description":target,"actual_sam_prompt":query,"views":records,"sam_inference_wall_s":result['inference_wall_s'],
        "sam_client_wall_s":client_wall,"sam_model_pid":result['model_pid'],"sam_request_index":result['request_index'],
        "geometry_source":"all valid visible depth pixels, separately calibrated then concatenated in world frame; no object GT",
        "cross_view_identity_certified":False,"duplicate_points_removed":False}
    (output/"perception.json").write_text(json.dumps(source,indent=2))
    if not all_points:raise ValueError("Neither view returned an observed target Mask")
    points,colors=np.concatenate(all_points),np.concatenate(all_colors)
    np.savez_compressed(output/"observed_object.npz",points=points,colors=colors,camera_index=np.concatenate(origins),camera_names=np.array(CAMERAS))
    source['point_count']=len(points)
    (output/"perception.json").write_text(json.dumps(source,indent=2))
    return points,colors,source


def render(points, colors, grasps, scores, output, graspgen_root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    sys.path.insert(0, str(graspgen_root.resolve()))
    from grasp_gen.robot import get_gripper_info
    info = get_gripper_info("franka_panda")
    controls = np.asarray(info.control_points_visualization[0])
    mesh = info.visual_mesh
    centre = np.median(points,axis=0)
    views = [(25,-65),(15,35),(75,-90)]
    geometry = [points-centre]
    geometry.extend(np.asarray(mesh.vertices)@g[:3,:3].T+g[:3,3]-centre for g in grasps[:8])
    radius = max(float(np.abs(np.concatenate(geometry)).max())*1.08, .09)
    def decorate(ax):
        for setter in (ax.set_xlim, ax.set_ylim, ax.set_zlim):setter(-radius,radius)
        from matplotlib.ticker import MaxNLocator
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):axis.set_major_locator(MaxNLocator(5))
        ax.set_box_aspect([1,1,1])
        ax.set_xlabel("X (m)");ax.set_ylabel("Y (m)");ax.set_zlabel("Z (m)")
    def cloud(ax):
        p = points-centre
        ax.scatter(*p.T,c=colors,s=2,alpha=.85,rasterized=True,depthshade=False)
    def wire(ax,g,color,lw=2):
        vertices=controls@g[:3,:3].T+g[:3,3]-centre
        ax.plot(*vertices.T,color=color,lw=lw)
    top = min(8,len(grasps))
    fig=plt.figure(figsize=(16,5))
    for i,(elev,azim) in enumerate(views):
        ax=fig.add_subplot(1,3,i+1,projection="3d")
        cloud(ax)
        for j,g in enumerate(grasps[:top]):wire(ax,g,plt.cm.viridis(j/max(top-1,1)),1.8)
        ax.view_init(elev,azim);decorate(ax)
        ax.set_title(f"Top {top} predicted grasps, view {i+1}")
    fig.suptitle("GraspGen Panda candidates | observed point cloud | not executed / no collision or IK validation")
    fig.tight_layout();fig.savefig(output/"candidates_overview.png",dpi=160);plt.close(fig)
    fig=plt.figure(figsize=(15,5))
    for j,g in enumerate(grasps[:3]):
        ax=fig.add_subplot(1,3,j+1,projection="3d")
        vertices=np.asarray(mesh.vertices)@g[:3,:3].T+g[:3,3]-centre
        polygons=Poly3DCollection(vertices[np.asarray(mesh.faces)],facecolor=(.25,.55,.85,.18),edgecolor="none")
        ax.add_collection3d(polygons);cloud(ax);wire(ax,g,"tab:blue",2.4)
        ax.view_init(22,-65);decorate(ax)
        ax.set_title(f"Rank {j+1}, model score {scores[j]:.3f}")
    fig.suptitle("Official Panda gripper geometry at generated poses | display centred on observed object")
    fig.tight_layout();fig.savefig(output/"top3_grippers.png",dpi=160);plt.close(fig)
    import plotly.graph_objects as go
    p=points-centre
    rgb=[f"rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})" for c in colors]
    traces=[go.Scatter3d(x=p[:,0],y=p[:,1],z=p[:,2],mode="markers",marker={"size":2,"color":rgb},name="Observed object")]
    for j,g in enumerate(grasps[:20]):
        vertices=controls@g[:3,:3].T+g[:3,3]-centre
        traces.append(go.Scatter3d(x=vertices[:,0],y=vertices[:,1],z=vertices[:,2],mode="lines",
            line={"width":6},name=f"Rank {j+1} score={scores[j]:.3f}",visible=True if j<5 else "legendonly"))
    fig=go.Figure(traces)
    fig.update_layout(title="Actual GraspGen predictions — toggle candidates in legend; not executed",
                      scene={"aspectmode":"data","xaxis_title":"X (m)","yaxis_title":"Y (m)","zaxis_title":"Z (m)"})
    fig.write_html(output/"candidates_3d.html",include_plotlyjs=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-json",type=Path)
    parser.add_argument("--observation-dir",type=Path)
    parser.add_argument("--target")
    parser.add_argument("--sam-prompt")
    parser.add_argument("--molmo-url")
    parser.add_argument("--candidate-id",action="append",default=[],metavar="CAMERA=ID")
    parser.add_argument("--sam-url",default="http://127.0.0.1:18761")
    parser.add_argument("--grasp-url",default="http://127.0.0.1:18760")
    parser.add_argument("--num-grasps",type=int,default=128)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    output=args.output or root/"runs"/(datetime.now().strftime("%Y%m%dT%H%M%S")+"_grasp_preview")
    output.mkdir(parents=True,exist_ok=False)
    if bool(args.sample_json)==bool(args.observation_dir):parser.error("Choose exactly one input source")
    if args.sample_json:
        data=json.loads(args.sample_json.read_text())
        points=np.asarray(data["pc"],dtype=np.float32)
        colors=np.asarray(data.get("pc_color",np.full_like(points,.5)),dtype=float)
        if colors.max()>1:colors/=255.
        source={"input":"official GraspGen real-object point-cloud sample", "file":str(args.sample_json.resolve()),
                "stored_sample_grasps_used":False}
    else:
        if not args.target:parser.error("An observed target description is required")
        candidate_ids={}
        for entry in args.candidate_id:
            camera,value=entry.split('=',1)
            if camera not in CAMERAS or camera in candidate_ids:parser.error("Expected unique CAMERA=ID entries")
            candidate_ids[camera]=int(value)
        points,colors,source=mask_clouds(args.observation_dir,args.target,candidate_ids,args.sam_url,output,args.sam_prompt,args.molmo_url)
    if len(points)<64:raise ValueError("Too few observed points for meaningful grasp preview")
    metadata={"source":source,"service":request(args.grasp_url+"/health"),"robot_actions":0}
    (output/"metadata.json").write_text(json.dumps(metadata,indent=2))
    begin=time.perf_counter()
    result=request(args.grasp_url+"/infer",{"points":points.tolist(),"num_grasps":args.num_grasps,"seed":0})
    grasps=np.asarray(result.pop("grasps"));scores=np.asarray(result.pop("scores"))
    if len(grasps)==0:raise RuntimeError("Model returned no grasp candidates; no fabricated fallback")
    order=np.argsort(-scores);grasps,scores=grasps[order],scores[order]
    np.savez_compressed(output/"candidates.npz",points=points,colors=colors,grasps=grasps,scores=scores)
    report={**result,"client_wall_s":time.perf_counter()-begin,"candidates":len(scores),
            "score_min":float(scores.min()),"score_max":float(scores.max()),"robot_actions":0,
            "input_is_simulator_gt":False,"raw_pose_frame":"input cloud frame, GraspGen base-link convention"}
    (output/"result.json").write_text(json.dumps(report,indent=2))
    render(points,colors,grasps,scores,output,root/".runtime/GraspGen")
    print(json.dumps({"output":str(output),**report},indent=2))


if __name__=="__main__":main()
