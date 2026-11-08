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


def mask_cloud(folder, camera, target, candidate_id, sam_url, output):
    image_path = folder / (camera+".png")
    frame = np.load(folder / (camera+".npz"))
    raw = image_path.read_bytes()
    image = Image.open(image_path).convert("RGB")
    result = request(sam_url+"/infer", {"image_base64": base64.b64encode(raw).decode(), "prompt": target})
    (output/"sam_response.json").write_text(json.dumps(result))
    image.save(output/"input_rgb.png")
    masks = []
    overlay = np.asarray(image).copy()
    palette = [(255,80,40), (40,210,100), (100,100,255), (220,80,220)]
    for candidate in result["candidates"]:
        h,w = candidate["mask_shape"]
        mask = np.unpackbits(np.frombuffer(base64.b64decode(candidate["mask_packbits_base64"]),dtype=np.uint8))[:h*w].reshape(h,w).astype(bool)
        masks.append(mask)
        color = np.array(palette[candidate["id"]%len(palette)])
        overlay[mask] = (.5*overlay[mask]+.5*color).astype(np.uint8)
    picture = Image.fromarray(overlay)
    draw = ImageDraw.Draw(picture)
    for c in result["candidates"]:
        draw.rectangle(c["box_xyxy"], outline=palette[c["id"]%len(palette)], width=2)
        draw.text(c["box_xyxy"][:2], f"ID {c['id']} score={c['score']:.2f}", fill="white")
    picture.save(output/"sam_masks.png")
    if candidate_id is None:
        if len(masks) != 1:
            raise ValueError(f"SAM returned {len(masks)} candidates. Inspect sam_masks.png and specify --candidate-id; no top-score selection.")
        candidate_id = 0
    if not 0 <= candidate_id < len(masks):
        raise ValueError("Selected candidate does not exist")
    mask = masks[candidate_id]
    depth = frame["depth_m"].squeeze()
    valid = mask & np.isfinite(depth) & (depth > .02) & (depth < 3.)
    y,x = np.nonzero(valid)
    rays = np.linalg.solve(frame["K"], np.vstack([x,y,np.ones(len(x))])).T
    points_camera = rays * depth[y,x,None]
    transform = frame["world_from_camera"]
    points = points_camera @ transform[:3,:3].T + transform[:3,3]
    colors = np.asarray(image)[y,x]/255.
    Image.fromarray((mask*255).astype(np.uint8)).save(output/"selected_mask.png")
    np.savez_compressed(output/"observed_object.npz", points=points, colors=colors, mask=mask,
                        K=frame["K"], world_from_camera=transform)
    return points,colors,{"input": "saved LIBERO RGB-D plus SAM3 mask", "observation_directory": str(folder.resolve()),
        "camera": camera, "target_description": target, "selected_candidate_id": candidate_id,
        "sam_candidate_count": len(masks), "sam_inference_wall_s": result["inference_wall_s"],
        "point_count": len(points), "geometry_source": "visible depth pixels only; no object GT"}


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
    parser.add_argument("--camera",default="agentview")
    parser.add_argument("--target")
    parser.add_argument("--candidate-id",type=int)
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
        points,colors,source=mask_cloud(args.observation_dir,args.camera,args.target,args.candidate_id,args.sam_url,output)
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
