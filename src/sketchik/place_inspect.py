"""Inspect saved AnyPlace poses: robot-only IK, observed clearance and render.

This is a diagnostic, not a full collision/motion planner or release authority.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', type=Path, required=True)
    parser.add_argument('--robot-observation', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--peer-pid', type=int, action='append', default=[])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    from .runtime import configure
    gpu = configure(root, args.gpu_uuid, 'pro', args.peer_pid)
    os.environ['EGL_DEVICE_ID'] = str(gpu['egl_device'])
    import mujoco
    import pyrender
    import trimesh
    from PIL import Image, ImageDraw
    from scipy.spatial.transform import Rotation
    from .kinematics import RobotIK, IKError
    from .grasp_render import native_visual_parts
    from .grasp_screen import robot_gripper
    from .place_preview import transform_points

    start = time.perf_counter()
    preview = args.preview
    result = json.loads((preview/'result.json').read_text())
    state = json.loads(Path(result['inputs']['held_state']).read_text())
    meta = json.loads((args.robot_observation/'metadata.json').read_text())
    ik = RobotIK((args.robot_observation/'robot_only.xml').read_text(), meta['joint_names'], meta['eef_site'])
    _, base_to_site, _ = robot_gripper(ik, state['q'])
    for name, value in zip(state['finger_joint_names'], state['finger_joint_positions']):
        ik.data.qpos[ik.model.jnt_qposadr[ik.model.joint(name).id]] = value
    parts, _ = native_visual_parts(ik, state['q'])
    full = np.load(preview/'inputs.npz')
    candidates = np.load(preview/'candidates.npz')
    scene = pyrender.Scene(bg_color=[0,0,0,0], ambient_light=[.65,.65,.65])
    nodes = []
    for mesh, color in parts:
        material = pyrender.MetallicRoughnessMaterial(baseColorFactor=color, doubleSided=True)
        nodes.append(scene.add(pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)))
    renderer = pyrender.OffscreenRenderer(1024,1024,point_size=2.5)
    directory = preview/'inspection'
    directory.mkdir(exist_ok=False)
    destination = Path(result['inputs']['destination_observation'])
    anchor = full['anchor']
    normal = np.asarray(result['anchor_provenance']['local_plane_normal_world'])
    records, thumbnails = [], []
    for index, (delta, eef) in enumerate(zip(candidates['world_deltas'], candidates['eef_poses']), 1):
        child = transform_points(full['child'], delta)
        base = eef @ np.linalg.inv(base_to_site)
        world_meshes = []
        for node, (mesh, _) in zip(nodes, parts):
            scene.set_pose(node, base)
            moved = mesh.copy(); moved.apply_transform(base); world_meshes.append(moved)
        gripper_vertices = np.concatenate([m.vertices for m in world_meshes])
        signed = (child-anchor) @ normal
        grip_signed = (gripper_vertices-anchor) @ normal
        record = {'id':index, 'eef_position':eef[:3,3].tolist(),
            'object_visible_bottom_to_local_floor_m':float(signed.min()),
            'gripper_min_to_local_floor_m':float(grip_signed.min()),
            'floor_test_scope':'observed local plane extrapolation; not whole cavity certification',
            'object_geometry_complete':False, 'arm_collision_checked':False}
        try:
            solution = ik.solve(eef[:3,3], Rotation.from_matrix(eef[:3,:3]).as_quat(), state['q'])
            record.update(final_ik=True, q=solution.q.tolist(), ik_s=solution.elapsed_s)
        except IKError as error:
            record.update(final_ik=False, ik_error=str(error))
        # Explicitly test a direct segment, not pretend it is an obstacle-free route.
        try:
            route = ik.cartesian_path(state['q'], eef[:3,3], Rotation.from_matrix(eef[:3,:3]).as_quat())
            record.update(direct_segment_ik=True, direct_segment_samples=len(route))
        except IKError as error:
            record.update(direct_segment_ik=False, direct_segment_error=str(error))
        # Merely coarse evidence rejection; do not convert to release permission.
        record['visible_floor_gap_rejected'] = bool(signed.min() > .02 or signed.min() < -.01)
        record['release_approved'] = False
        record['not_approved_reason'] = 'No verified same-episode cavity association, collision-free entry/withdrawal or stable support'
        records.append(record)
        point_node = scene.add(pyrender.Mesh.from_points(child, colors=np.tile([1.,.1,.55],(len(child),1))))
        panels = []
        for camera in ('agentview','robot0_eye_in_hand'):
            frame = np.load(destination/(camera+'.npz'))
            rgb = np.asarray(Image.open(destination/(camera+'.png')).convert('RGB')).copy()
            K = frame['K']; camera_pose = frame['world_from_camera'] @ np.diag([1,-1,-1,1])
            cam = scene.add(pyrender.IntrinsicsCamera(K[0,0],K[1,1],K[0,2],K[1,2],znear=.01,zfar=4), pose=camera_pose)
            light = scene.add(pyrender.DirectionalLight(color=np.ones(3),intensity=1.5),pose=camera_pose)
            color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
            observed = frame['depth_m'].squeeze()
            visible = (depth>0) & ((~np.isfinite(observed)) | (depth<=observed+.002))
            rgb[visible] = color[visible,:3]
            panel = Image.fromarray(rgb)
            draw = ImageDraw.Draw(panel)
            draw.rectangle((0,0,1024,65),fill='white')
            draw.text((10,6), f'AnyPlace {index:02d} | {camera} | partial observed held object: pink',fill='black')
            draw.text((10,27), f'Local floor gap {signed.min()*1000:.1f} mm | final IK: {record["final_ik"]}',fill='black')
            draw.text((10,47), 'DEPTH OCCLUDED OVERLAY - unknown surfaces are not reconstructed',fill='black')
            panel.save(directory/f'candidate_{index:02d}_{camera}.png')
            panels.append(panel)
            scene.remove_node(cam); scene.remove_node(light)
        pair = Image.new('RGB',(2048,1024),'white')
        pair.paste(panels[0],(0,0)); pair.paste(panels[1],(1024,0))
        pair.save(directory/f'candidate_{index:02d}.jpg',quality=92)
        pair.thumbnail((640,320)); thumbnails.append(pair)
        scene.remove_node(point_node)
    renderer.delete()
    sheet = Image.new('RGB',(1280,320*((len(thumbnails)+1)//2)),'white')
    for i, thumbnail in enumerate(thumbnails):sheet.paste(thumbnail,((i%2)*640,(i//2)*320))
    sheet.save(directory/'all_candidates.jpg',quality=94)
    report = {'candidates':records,'robot_actions':0,'gpu':gpu,
              'elapsed_s':time.perf_counter()-start,
              'summary':{'total':len(records),'final_ik_pass':sum(r['final_ik'] for r in records),
                         'direct_segment_ik_pass':sum(r['direct_segment_ik'] for r in records),
                         'visible_floor_gap_rejected':sum(r['visible_floor_gap_rejected'] for r in records),
                         'release_approved':0}}
    (directory/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report['summary']),flush=True)


if __name__ == '__main__':
    main()
