from __future__ import annotations

import argparse
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description="Independent LIBERO IK primitives")
    parser.add_argument("command", choices=["probe", "smoke", "execute"])
    parser.add_argument("--gpu-uuid", required=True, help="Exact idle GPU UUID; renderer only")
    parser.add_argument("--variant", choices=["pro", "standard"], default="pro")
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init-state", type=int, default=0)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--plan", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    run = root / "runs" / (time.strftime("%Y%m%dT%H%M%S") + "_" + args.command)
    run.mkdir(parents=True, exist_ok=False)
    from .runtime import configure
    gpu = configure(root, args.gpu_uuid, args.variant)
    from .libero import LiberoRobot
    import numpy as np
    from PIL import Image
    start = time.perf_counter()
    try:
        robot = LiberoRobot(args.variant, args.suite, args.task, args.seed, args.init_state, args.size)
    except Exception as exc:
        import traceback
        (run / "startup_error.json").write_text(json.dumps({
            "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
            "gpu": gpu, "variant": args.variant, "task": args.task,
            "seed": args.seed, "init_state_id": args.init_state,
        }, indent=2))
        raise
    try:
        metadata = robot.metadata | {"gpu": gpu, "initialization_wall_s": time.perf_counter() - start}
        (run / "metadata.json").write_text(json.dumps(metadata, indent=2))
        state = robot.state()
        (run / "initial_state.json").write_text(json.dumps(state, indent=2))
        for name in robot.cameras:
            frame = robot.camera(name)
            Image.fromarray(frame["rgb"]).save(run / (name + ".png"))
            np.savez_compressed(run / (name + ".npz"), depth_m=frame["depth_m"],
                                K=frame["K"], world_from_camera=frame["world_from_camera"])
        if args.command == "probe":
            xml = robot.robot.robot_model.get_xml()
            (run / "robot_only.xml").write_text(xml)
            print(json.dumps({"run": str(run), "metadata": metadata, "state": state}, indent=2))
        else:
            from .validation import run_validation, json_default
            result = run_validation(robot, run, args.command, args.plan)
            print(json.dumps({"run": str(run), "result": result}, default=json_default, indent=2))
            if not result["control_validation_passed"]:
                raise SystemExit(1)
    finally:
        robot.close()


if __name__ == "__main__":
    main()
