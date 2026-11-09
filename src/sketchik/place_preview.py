"""Offline AnyPlace candidate generation from saved, observed RGB-D only.

Does not dispatch robot actions. Destination observations may come from a separate
diagnostic episode: those candidates must not be replayed without scene checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
import types

import numpy as np


def transform_points(points, transform):
    return points @ transform[:3, :3].T + transform[:3, 3]


def target_eef_pose(world_delta, measured_position, measured_quat):
    from scipy.spatial.transform import Rotation
    current = np.eye(4)
    current[:3, :3] = Rotation.from_quat(measured_quat).as_matrix()
    current[:3, 3] = measured_position
    return world_delta @ current


class NoDisplay:
    """Suppress upstream Meshcat display only; preserve all tensor computation."""
    def __getitem__(self, key):
        return self

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def load_upstream(source, checkpoint):
    import torch
    # The selected transformer uses torch_cluster FPS, not the legacy Group/KNN
    # class. Fail explicitly if that unused branch is ever invoked; no fake KNN.
    try:
        import knn_cuda  # noqa: F401
    except ImportError:
        module = types.ModuleType('knn_cuda')
        class UnsupportedKNN:
            def __init__(self, *args, **kwargs):
                raise RuntimeError('Legacy knn_cuda Group branch is not supported')
        module.KNN = UnsupportedKNN
        sys.modules['knn_cuda'] = module
    logger = types.ModuleType('airobot')
    logger.log_warn = logging.warning
    logger.log_debug = logging.debug
    sys.modules.setdefault('airobot', logger)
    sys.path.insert(0, str(source.resolve()))
    from anyplace.model.transformer.policy import NSMTransformerSingleTransformationRegression
    from anyplace.utils.anyplace.multistep_pose_regression_anyplace import multistep_regression_scene
    data = torch.load(checkpoint, map_location='cpu', weights_only=False)
    config = data['args']['model']
    refine = config['refine_pose']
    if refine['type'] != 'nsm_transformer':
        raise ValueError('Only official NSM transformer checkpoint supported')
    kwargs = dict(config['nsm_transformer'])
    kwargs.update(refine.get('model_kwargs', {}).get('nsm_transformer', {}))
    model = NSMTransformerSingleTransformationRegression(feat_dim=refine['feat_dim'],
                                                       mc_vis=NoDisplay(), **kwargs).cuda().eval()
    model.load_state_dict(data['refine_pose_model_state_dict'], strict=True)
    return model, multistep_regression_scene, kwargs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('.runtime/AnyPlace'))
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--held-cloud', type=Path, required=True)
    parser.add_argument('--held-state', type=Path, required=True)
    parser.add_argument('--destination-observation', type=Path, required=True)
    parser.add_argument('--destination-point', type=Path, required=True,
                        help='Saved observed-world anchor JSON; provenance retained')
    parser.add_argument('--crop-radius', type=float, default=.18)
    parser.add_argument('--plane-range', type=float, nargs=2,
                        help='Optional observed local-plane signed crop range in metres; diagnostic ROI only')
    parser.add_argument('--candidates', type=int, default=20)
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--peer-pid', action='append', type=int, default=[])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    begin = time.perf_counter()
    try:
        run(args, begin)
    except Exception as error:
        (args.output/'error.json').write_text(json.dumps({'error': repr(error),
            'elapsed_s': time.perf_counter()-begin, 'robot_actions': 0}, indent=2))
        raise


def run(args, begin):
    from PIL import Image
    from .grasp_preview import CAMERAS, points_from_mask
    from .runtime import validate_model_peers
    root = Path(__file__).resolve().parents[2]
    peers = validate_model_peers(root, args.peer_pid)
    preflight = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                         '--format=csv,noheader,nounits'], text=True)
    occupied = {int(row.split(',')[1]) for row in preflight.splitlines()
                if row.split(',')[0].strip() == args.gpu_uuid}
    if occupied-peers:
        raise RuntimeError('GPU contains unapproved processes')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_uuid
    import torch
    np.random.seed(0)
    torch.manual_seed(0)
    child = np.load(args.held_cloud)['points']
    state = json.loads(args.held_state.read_text())
    point = json.loads(args.destination_point.read_text())
    anchor = np.asarray(point['world_xyz'], dtype=float)
    if anchor.shape != (3,) or not np.isfinite(anchor).all() or not 0 < args.crop_radius < 1:
        raise ValueError('Invalid observed anchor/crop')
    clouds, colors = [], []
    for camera in CAMERAS:
        image = Image.open(args.destination_observation/(camera+'.png')).convert('RGB')
        with np.load(args.destination_observation/(camera+'.npz')) as frame:
            xyz, rgb = points_from_mask(frame, np.ones(np.asarray(image).shape[:2], bool), image)
        keep = np.all(np.abs(xyz-anchor) <= args.crop_radius, axis=1)
        if args.plane_range is not None:
            low, high = args.plane_range
            normal = np.asarray(point['local_plane_normal_world'], dtype=float)
            if normal.shape != (3,) or not np.isclose(np.linalg.norm(normal), 1.) or not low < high:
                raise ValueError('Invalid observed plane / crop range')
            distance = (xyz-anchor) @ normal
            keep &= (distance >= low) & (distance <= high)
        clouds.append(xyz[keep]); colors.append(rgb[keep])
    parent, rgb = np.concatenate(clouds), np.concatenate(colors)
    if len(parent) < 8192 or len(child) < 1024:
        raise ValueError('Insufficient real points for upstream FPS; do not fabricate points')
    # Bound CPU/GPU preprocessing cost, retain deterministic real samples only.
    rng = np.random.default_rng(0)
    if len(parent) > 40000:
        ids = rng.choice(len(parent), 40000, replace=False)
        parent, rgb = parent[ids], rgb[ids]
    if len(child) > 20000:
        child = child[rng.choice(len(child), 20000, replace=False)]
    np.savez_compressed(args.output/'inputs.npz', parent=parent, parent_colors=rgb,
                        child=child, anchor=anchor)
    setup = time.perf_counter()
    model, infer, kwargs = load_upstream(args.source, args.checkpoint)
    torch.cuda.synchronize()
    loaded = time.perf_counter()
    with torch.inference_mode():
        transforms = infer(NoDisplay(), parent, child, None, model, None,
            scene_scale=1., scene_mean=np.zeros(3), grid_pts=np.empty((0, 3)),
            n_iters=args.iterations, init_k_val=args.candidates, run_affordance=False,
            init_orig_ori=False, no_sc_score=True, with_coll=False,
            add_per_iter_noise=True, per_iter_noise_kwargs={
                'rot': {'angle_deg':20, 'rate':6.5},
                'trans': {'trans_dist':.03, 'rate':5.5}},
            variable_size_crop=False, timestep_emb_decay_factor=20)
    torch.cuda.synchronize()
    finished = time.perf_counter()
    transforms = np.asarray(transforms)
    if transforms.ndim != 3 or transforms.shape[1:] != (4,4) or not np.isfinite(transforms).all():
        raise ValueError('Invalid model transforms')
    eef = np.stack([target_eef_pose(t, state['position'], state['quaternion_xyzw']) for t in transforms])
    np.savez_compressed(args.output/'candidates.npz', world_deltas=transforms, eef_poses=eef)
    result = {'status':'generated_not_execution_approved', 'robot_actions':0,
        'source_commit':subprocess.check_output(['git','-C',str(args.source),'rev-parse','HEAD'],text=True).strip(),
        'checkpoint':str(args.checkpoint.resolve()),
        'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        'inputs':{key:str(getattr(args,key).resolve()) for key in
                  ['held_cloud','held_state','destination_observation','destination_point']},
        'anchor_provenance':point, 'crop_radius_m':args.crop_radius,
        'destination_fields_used':['world_xyz', 'local_plane_normal_world only when plane-range is supplied'],
        'manual_floor_polygon_used':False,
        'local_observed_plane_crop_range_m':args.plane_range,
        'parent_points':len(parent), 'child_points':len(child), 'model_kwargs':kwargs,
        'iterations':args.iterations, 'candidate_count':len(transforms),
        'gpu_uuid':args.gpu_uuid, 'preflight':preflight,
        'timing_s':{'prepare':setup-begin,'load':loaded-setup,'inference':finished-loaded,
                    'total':finished-begin},
        'peak_torch_memory_bytes':torch.cuda.max_memory_allocated(),
        'compatibility':['Meshcat display suppressed','airobot logging only mapped to stdlib',
                         'unused legacy KNN branch fails explicitly if called'],
        'learned_success_scores_available':False,'hidden_object_gt_used':False,
        'held_rigid_transform_assumed':True,'scene_and_path_collision_certified':False,
        'destination_observation_same_episode_as_grasp':False}
    (args.output/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result['timing_s']),flush=True)


if __name__ == '__main__':
    main()
