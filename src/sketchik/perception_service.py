"""Local resident SAM3 or GraspGen worker; no robot movement or hidden simulator GT."""
from __future__ import annotations

import argparse
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


def reserve_device(gpu_uuid, peer_pids=()):
    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    selected = next((r.split(",") for r in rows.splitlines() if r.split(",")[0] == gpu_uuid), None)
    if selected is None:
        raise RuntimeError("Requested GPU UUID not found")
    valid_peers = set()
    project = Path(__file__).resolve().parents[2]
    for pid in peer_pids:
        proc = Path(f"/proc/{pid}")
        command = (proc/"cmdline").read_bytes().replace(b"\0", b" ").decode()
        if proc.stat().st_uid != os.getuid() or "sketchik.perception_service" not in command or (proc/"cwd").resolve() != project:
            raise RuntimeError("Peer PID is not this project's model service")
        valid_peers.add(pid)
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True)
    occupied = [int(row.split(",")[1]) for row in processes.splitlines() if row.split(",")[0] == gpu_uuid]
    if set(occupied) - valid_peers or (not valid_peers and int(selected[2]) > 5) or int(selected[1]) < 4000:
        raise RuntimeError("GPU busy or insufficient free memory; other processes are not stopped")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_uuid
    return {"uuid": gpu_uuid, "free_mib_before_load": int(selected[1]), "peer_pids": sorted(valid_peers)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["graspgen", "sam3"])
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--peer-pid", type=int, action="append", default=[])
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--graspgen-root", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    gpu = reserve_device(args.gpu_uuid, args.peer_pid)
    os.environ.update(USE_FLAX="0", USE_TF="0", PYTHONDONTWRITEBYTECODE="1")
    start = time.perf_counter()
    import numpy as np
    import torch
    from PIL import Image
    torch.set_num_threads(4)
    if args.kind == "graspgen":
        if args.graspgen_root is None:
            parser.error("GraspGen root is required")
        sys.path.insert(0, str(args.graspgen_root.resolve()))
        # Only trusted downloaded official checkpoints are used. PyTorch >=2.6
        # changed the default; this preserves upstream checkpoint loading semantics.
        os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
        from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
        cfg = load_grasp_cfg(str(args.checkpoint.resolve()))
        sampler = GraspGenSampler(cfg)
        def infer(request):
            points = np.asarray(request["points"], dtype=np.float32)
            if points.ndim != 2 or points.shape[1] != 3 or not 64 <= len(points) <= 100000 or not np.isfinite(points).all():
                raise ValueError("Expected 64..100000 finite metric XYZ points")
            count = int(request.get("num_grasps", 128))
            if not 1 <= count <= 512:
                raise ValueError("num_grasps must be 1..512")
            seed = int(request.get("seed", 0))
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            grasps, scores = GraspGenSampler.run_inference(points, sampler, num_grasps=count,
                topk_num_grasps=count, min_grasps=1, max_tries=1, remove_outliers=True)
            return {"grasps": grasps.detach().cpu().numpy().tolist(), "scores": scores.detach().cpu().numpy().tolist(),
                    "frame": "same metric coordinate frame as input points; GraspGen gripper base frame",
                    "backend": "official diffusion-discriminator", "collision_checked": False, "ik_checked": False}
    else:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model import box_ops
        from sam3.model.data_misc import FindStage, interpolate
        model = build_sam3_image_model(checkpoint_path=str(args.checkpoint.resolve()), device="cuda", load_from_HF=False)
        processor = Sam3Processor(model, confidence_threshold=.3, device="cuda")
        def infer(request):
            items = request.get("images")
            if not isinstance(items, list) or len(items) != 2:
                raise ValueError("Provide exactly two labelled RGB images in images")
            if {v.get('camera') for v in items} != {'agentview', 'robot0_eye_in_hand'}:
                raise ValueError("Expected unique agentview and robot0_eye_in_hand labels")
            images = [Image.open(BytesIO(base64.b64decode(v["image_base64"], validate=True))).convert("RGB") for v in items]
            prompt = request["prompt"]
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("SAM3 requires a nonempty target description")
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                state = processor.set_image_batch(images)
                state['backbone_out'].update(model.backbone.forward_text([prompt], device='cuda'))
                find = FindStage(img_ids=torch.arange(2, device='cuda'), text_ids=torch.zeros(2, dtype=torch.long, device='cuda'),
                                 input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
                                 input_points=None, input_points_mask=None)
                output = model.forward_grounding(backbone_out=state['backbone_out'], find_input=find,
                    geometric_prompt=model._get_dummy_prompt(num_prompts=2), find_target=None)
                probabilities = (output['pred_logits'].sigmoid() * output['presence_logit_dec'].sigmoid().unsqueeze(1)).squeeze(-1)
                views = []
                for index, (item, image) in enumerate(zip(items, images)):
                    keep = probabilities[index] > processor.confidence_threshold
                    scores = probabilities[index][keep].float().cpu().numpy().reshape(-1)
                    boxes = box_ops.box_cxcywh_to_xyxy(output['pred_boxes'][index][keep])
                    boxes = (boxes * torch.tensor([image.width,image.height,image.width,image.height],device='cuda')).float().cpu().numpy()
                    logits = output['pred_masks'][index][keep]
                    masks = (interpolate(logits.unsqueeze(1), (image.height, image.width), mode='bilinear', align_corners=False).sigmoid() > .5).cpu().numpy()
                    masks = masks.reshape(len(scores), image.height, image.width)
                    h,w = image.height,image.width
                    views.append({"camera": item['camera'], "width": w, "height": h,
                        "candidates": [{"id": i, "score": float(score), "box_xyxy": box.tolist(),
                        "mask_shape": [h, w],
                        "mask_packbits_base64": base64.b64encode(np.packbits(mask.astype(bool).ravel()).tobytes()).decode()}
                        for i,(mask,score,box) in enumerate(zip(masks,scores,boxes))]})
            return {"prompt": prompt, "images": views, "batch_size": 2,
                    "processing": "one batched image encoder and grounding call; per-camera masks, no cross-view identity certification",
                    "selection": "all returned candidates, no top-score selection", "threshold": .3}
    torch.cuda.synchronize()
    metadata = {"kind": args.kind, "pid": os.getpid(), "gpu": gpu, "port": args.port,
                "checkpoint": str(args.checkpoint.resolve()), "model_load_wall_s": time.perf_counter()-start,
                "resident": True, "requests": 0, "allocated_mib": torch.cuda.memory_allocated()/2**20,
                "torch": torch.__version__}
    lock = threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *values):
            print(fmt % values, flush=True)
        def send(self, status, data):
            raw = json.dumps(data, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        def do_GET(self):
            self.send(200 if self.path == "/health" else 404, metadata)
        def do_POST(self):
            if self.path != "/infer":
                self.send(404, {"error": "Unknown route"})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 24*1024**2:
                    raise ValueError("Invalid request length")
                request = json.loads(self.rfile.read(length))
                with lock:
                    torch.cuda.synchronize()
                    begin = time.perf_counter()
                    output = infer(request)
                    torch.cuda.synchronize()
                    metadata["requests"] += 1
                    output.update(model_pid=os.getpid(), request_index=metadata["requests"],
                                  inference_wall_s=time.perf_counter()-begin,
                                  allocated_mib=torch.cuda.memory_allocated()/2**20,
                                  peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
                print(json.dumps({"event": "inference", "kind": args.kind, "request": output["request_index"],
                                  "wall_s": output["inference_wall_s"]}), flush=True)
                self.send(200, output)
            except Exception as exc:
                import traceback
                traceback.print_exc()
                self.send(500, {"error": f"{type(exc).__name__}: {exc}"})
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    directory = root / ".runtime" / "perception"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / (args.kind+".json")).write_text(json.dumps(metadata, indent=2))
    print(json.dumps({"event": "ready", **metadata}), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
