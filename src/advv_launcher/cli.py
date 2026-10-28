from __future__ import annotations

import argparse
import logging
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from advv import cli as coordinator
from advv.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, gpu_groups, load_config, parse_gpus
from advv.config import validate_config
from advv.environment import require_advv
from advv.errors import ADVVError
from advv.storage import read_json, run_lock, safe_id

from .background import launch


def run_parser():
    parser = argparse.ArgumentParser(
        prog="advv run", description="Run augmentation in the background with automatic nohup and logs"
    )
    parser.add_argument("count", type=coordinator.positive_count, nargs="?", metavar="N")
    parser.add_argument("--target-count", type=coordinator.positive_count, metavar="N")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--gpus", required=True, help="Physical GPU pairs, e.g. 4,5,6,7")
    parser.add_argument("--image", metavar="FILE", help="Single image relative to assets/inputs")
    parser.add_argument("--prompt", help="Optional augmentation instruction; CLIP criteria stay fixed")
    parser.add_argument(
        "--repair-mode", choices=["endpoints", "swept"],
        help="Background repair: endpoints (new default) or swept (legacy full drag path)",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--foreground", action="store_true",
        help="Explicit diagnostic mode: stay in this terminal instead of automatic nohup",
    )
    return parser


def validate_arguments(parser, args):
    if args.count is not None and args.target_count is not None:
        parser.error("Specify N or --target-count N, not both")
    if args.count is not None:
        args.target_count = args.count
    if args.resume and args.target_count is not None:
        parser.error("Resume uses the saved target count; omit N and --target-count")
    if args.prompt is not None and not args.prompt.strip():
        parser.error("--prompt must be nonempty text")
    if args.resume and args.prompt is not None:
        parser.error("Resume uses the saved prompt; omit --prompt or start a new run")
    if args.resume and args.image is not None:
        parser.error("Resume uses the saved image selection; omit --image or start a new run")
    if args.resume and args.repair_mode is not None:
        parser.error("Resume uses the saved repair mode; omit --repair-mode or start a new run")


def resolve_run(args):
    """Validate cheap configuration/input checks without cleanup, model loading or CUDA."""
    require_advv()
    gpu_groups(args.gpus)
    if args.resume:
        if not args.run_dir or args.run_id:
            raise ADVVError("Resume requires --run-dir and --gpus; input/N changes require a new run")
        root = args.run_dir.resolve()
        cfg = read_json(root / "config.json")
        validate_config(cfg)
        # Reject an already-running coordinator before creating another background process.
        with run_lock(root):
            pass
        command = ["run", "--resume", "--run-dir", str(root), "--gpus", args.gpus]
    else:
        if args.run_dir:
            raise ADVVError("Use --run-dir only with --resume")
        cfg = load_config(
            args.config, target_count=args.target_count, gpus=args.gpus,
            prompt=args.prompt, image=args.image,
            repair_mode=args.repair_mode,
        )
        if not cfg["run"]["target_count"]:
            raise ADVVError("Specify a positive total N")
        run_id = args.run_id or (
            datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ_") + uuid4().hex[:8]
        )
        safe_id(run_id)
        root = Path(cfg["run"]["output_root"]) / run_id
        if root.exists():
            raise ADVVError("Run ID already exists; choose a new ID or --resume")
        command = [
            "run", str(cfg["run"]["target_count"]), "--config", str(args.config.resolve()),
            "--gpus", args.gpus, "--run-id", run_id,
            "--repair-mode", cfg["generation"]["repair_mode"],
        ]
        if args.image is not None:
            command += ["--image", args.image]
        if args.prompt is not None:
            # Equals form preserves a literal value even when it begins with '--'. No shell parsing.
            command += ["--prompt=" + args.prompt]
    return command, root, {
        "target_count": cfg["run"]["target_count"],
        "image": cfg["dataset"].get("image"),
        "gpu_ids": parse_gpus(args.gpus),
        "resume": args.resume,
        "repair_mode": cfg["generation"].get("repair_mode", "swept"),
    }


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] != "run":
        return coordinator.main(argv)
    parser = run_parser()
    args = parser.parse_args(argv[1:])
    validate_arguments(parser, args)
    try:
        command, root, details = resolve_run(args)
        if args.foreground:
            return coordinator.main(command)
        result = launch(command, root, PROJECT_ROOT / "logs", details=details)
        print("ADVV background process started (initializing; check the log for readiness).", flush=True)
        print(f"PID: {result['pid']}", flush=True)
        print(f"Run: {root}", flush=True)
        print(f"Log: {result['log_path']}", flush=True)
        print(f"PID file: {result['pid_path']}", flush=True)
        print(f"Status: {result['status_path']}", flush=True)
        print(f"Watch: tail -n 100 -f {shlex.quote(result['log_path'])}", flush=True)
        print(f"Stop: kill -TERM {result['pid']}", flush=True)
        return 0
    except (ADVVError, OSError, ValueError, KeyError) as exc:
        logging.error("%s: %s", type(exc).__name__, exc)
        return 1
