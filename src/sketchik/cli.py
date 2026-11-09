from __future__ import annotations

import argparse
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description="Independent LIBERO IK primitives")
    parser.add_argument("command", choices=["probe", "smoke", "execute", "agent", "agent-smoke"])
    parser.add_argument("--gpu-uuid", required=True, help="Exact GPU UUID; renderer only")
    parser.add_argument("--peer-pid", type=int, action="append", default=[], help="Allow this project's resident model PID on the renderer GPU")
    parser.add_argument("--variant", choices=["pro", "standard"], default="pro")
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init-state", type=int, default=0)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--settle-steps", type=int, default=0, help="Initial physics settling before capturing observations; recorded and replayable")
    parser.add_argument("--env-file", type=Path, help="Literal SKETCHIK_* settings; not shell-sourced")
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--reasoning-effort", help="Optional provider setting; no silent model-specific fallback")
    parser.add_argument("--api-timeout", type=float, default=90)
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--max-tool-calls", type=int, default=100)
    parser.add_argument("--max-episode-steps", type=int, default=2000)
    args = parser.parse_args()
    if not 0 <= args.settle_steps <= 200 or args.settle_steps >= args.max_episode_steps:
        parser.error("settle-steps must be 0..200 and below episode budget")
    if min(args.api_timeout, args.max_turns, args.max_tool_calls, args.max_episode_steps) <= 0:
        parser.error("API timeout and execution budgets must be positive")
    client = None
    if args.command == "agent":
        # Validate settings before acquiring a renderer or constructing the environment.
        from .agent import ChatClient, load_settings
        settings = load_settings(args.env_file)
        client = ChatClient(args.base_url or settings.get("SKETCHIK_BASE_URL"),
                            args.model or settings.get("SKETCHIK_MODEL"), settings.get("SKETCHIK_API_KEY"),
                            args.reasoning_effort or settings.get("SKETCHIK_REASONING"), args.api_timeout)
    root = Path(__file__).resolve().parents[2]
    run = root / "runs" / (time.strftime("%Y%m%dT%H%M%S") + "_" + args.command)
    run.mkdir(parents=True, exist_ok=False)
    from .runtime import configure
    gpu = configure(root, args.gpu_uuid, args.variant, args.peer_pid)
    from .libero import LiberoRobot
    import numpy as np
    from PIL import Image
    start = time.perf_counter()
    try:
        robot = LiberoRobot(args.variant, args.suite, args.task, args.seed, args.init_state,
                            max_steps=args.max_episode_steps)
    except Exception as exc:
        import traceback
        (run / "startup_error.json").write_text(json.dumps({
            "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
            "gpu": gpu, "variant": args.variant, "task": args.task,
            "seed": args.seed, "init_state_id": args.init_state,
        }, indent=2))
        raise
    try:
        prelude=None
        if args.settle_steps:
            q=robot.state()['q']
            prelude={'steps':args.settle_steps,'joint_target':q,'gripper':-1.0}
            for _ in range(args.settle_steps):robot.step(q,-1.0)
        metadata = robot.metadata | {"gpu": gpu, "initialization_wall_s": time.perf_counter() - start,
                                     "observation_prelude":prelude}
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
            result = run_validation(robot, run, args.command, args.plan, client=client,
                                    max_turns=args.max_turns, max_tool_calls=args.max_tool_calls)
            print(json.dumps({"run": str(run), "result": result}, default=json_default, indent=2))
            if not result["control_validation_passed"]:
                raise SystemExit(1)
    finally:
        robot.close()


if __name__ == "__main__":
    main()
