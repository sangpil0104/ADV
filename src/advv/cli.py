from __future__ import annotations

import argparse
import copy
import json
import logging
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .backends.process import LocalBackend
from .config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, gpu_groups, load_config, parse_gpus
from .config import validate_config
from .cleanup import cleanup_existing_runs
from .environment import require_advv
from .errors import ADVVError
from .pipeline import Pipeline, create_run
from .preflight import preflight
from .prepare import prepare
from .reporting import export_run, report_run
from .storage import read_json


def positive_count(value: str) -> int:
    try:
        count = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("N must be a positive integer") from exc
    if count <= 0:
        raise argparse.ArgumentTypeError("N must be a positive integer")
    return count


def main(argv=None):
    def interrupted(signum, frame):
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(
        prog="advv", description="Dashcam accident augmentation with DragFlow and paired CLIP"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "preflight"):
        p = sub.add_parser(name)
        p.add_argument(
            "--config",
            type=Path,
            default=DEFAULT_CONFIG_PATH,
            help="Optional config override; defaults to the installed project config",
        )
        p.add_argument(
            "--target-count",
            type=positive_count,
            metavar="N",
            help="Total new, unique images passing CLIP 3-of-4 AND mandatory accident criterion",
        )
        p.add_argument(
            "--gpus", required=True, help="Physical GPU pairs, e.g. 4,5,6,7 runs two candidates in parallel"
        )
        p.add_argument(
            "--image",
            metavar="FILE",
            help="Augment only this file relative to assets/inputs (e.g. image1.png); default: all inputs",
        )
        p.add_argument(
            "--repair-mode", choices=["endpoints", "swept"],
            help="Background repair: endpoints (new default) or swept (legacy full drag path)",
        )
        if name == "run":
            p.add_argument("--prompt", help="Optional augmentation instruction; CLIP criteria stay fixed")
            p.add_argument(
                "count",
                type=positive_count,
                nargs="?",
                metavar="N",
                help="Total accepted images to collect across all inputs (e.g. advv run 10)",
            )
            p.add_argument("--run-id", help="Optional run name; generated automatically if omitted")
            p.add_argument("--run-dir", type=Path)
            p.add_argument("--resume", action="store_true")
    p = sub.add_parser("prepare", help="Download the pinned models; inference remains offline")
    p.add_argument("--project", type=Path, default=Path.cwd())
    p.add_argument(
        "--only", choices=["all", "clip", "segmenter", "ground", "dragflow", "support"], default="all"
    )
    for name in ("export", "report"):
        p = sub.add_parser(name)
        p.add_argument("--run-dir", type=Path, required=True)
    p = sub.add_parser(
        "visualize", help="Group previews by kind and build an image overview without inference"
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-dir", type=Path)
    group.add_argument("--all", action="store_true", help="Include assets/outputs and legacy runs")
    args = parser.parse_args(argv)
    if args.command == "run":
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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        require_advv()
        started = time.monotonic()
        if args.command == "prepare":
            prepare(args.project, only=args.only)
            return 0
        if args.command == "visualize":
            from .storage import run_lock
            from .visualization_collection import build_all_collections, build_collection

            if args.all:
                print(json.dumps(build_all_collections(PROJECT_ROOT), ensure_ascii=False, indent=2))
            else:
                root = args.run_dir.resolve()
                with run_lock(root):
                    result = build_collection(root)
                print(
                    json.dumps(
                        {"run_dir": str(root), "count": result["count"], "artifacts": result["artifacts"]},
                        ensure_ascii=False,
                        indent=2,
                    )
                )
            return 0
        if args.command in ("report", "export"):
            from .storage import run_lock

            root = args.run_dir.resolve()
            with run_lock(root):
                print((report_run if args.command == "report" else export_run)(root))
            return 0
        if args.command == "run" and args.resume:
            if not args.run_dir or args.run_id or args.target_count is not None:
                raise ADVVError("Resume requires --run-dir and --gpus; input/N changes require a new run")
            root = args.run_dir.resolve()
            cfg = read_json(root / "config.json")
            validate_config(cfg)
            cfg["execution"]["selected_gpu_ids"] = parse_gpus(args.gpus)
            gpu_groups(args.gpus)
            cleanup_existing_runs(Path(cfg["run"]["output_root"]), exclude=root)
            preflight(cfg, check_inputs=False)
            initial_seconds = time.monotonic() - started
        else:
            cfg = load_config(
                args.config,
                target_count=args.target_count,
                gpus=args.gpus,
                prompt=getattr(args, "prompt", None),
                image=args.image,
                repair_mode=args.repair_mode,
            )
            if args.command == "run":
                if args.run_dir:
                    raise ADVVError("Use --run-dir only with --resume")
                if not args.run_id:
                    args.run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ_") + uuid4().hex[:8]
                from .storage import safe_id

                safe_id(args.run_id)
                if (Path(cfg["run"]["output_root"]) / args.run_id).exists():
                    raise ADVVError("Run ID already exists; choose a new ID or --resume")
                if not cfg["run"]["target_count"]:
                    raise ADVVError("Specify a positive total N")
                cleanup_existing_runs(Path(cfg["run"]["output_root"]))
            checks = preflight(cfg)
            if args.command == "preflight":
                print(
                    json.dumps(
                        {"status": checks["status"], "gpus": checks["gpus"], "inputs": checks["inputs"]},
                        indent=2,
                    )
                )
                return 0
            root = create_run(
                cfg,
                args.run_id,
                provenance=LocalBackend.provenance,
                initial_seconds=time.monotonic() - started,
            )
            initial_seconds = 0.0
        groups = gpu_groups(cfg["execution"]["selected_gpu_ids"])
        backends = []
        for group in groups:
            lane_cfg = copy.deepcopy(cfg)
            lane_cfg["execution"]["selected_gpu_ids"] = group
            backends.append(LocalBackend(lane_cfg, root))
        logging.info(
            "Run: %s | GPU groups: %s | Shared target: %s | Repair mode: %s",
            root, groups, cfg["run"]["target_count"], cfg["generation"].get("repair_mode", "swept"),
        )
        state = Pipeline(
            root, backends, gpu_ids=cfg["execution"]["selected_gpu_ids"], initial_seconds=initial_seconds
        ).run()
        print(
            json.dumps(
                {
                    "run_dir": str(root),
                    "status": state["status"],
                    "accepted_count": state["accepted_count"],
                    "last_error": state.get("last_error"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if state["status"] == "completed" else 130 if state["status"] == "interrupted" else 1
    except (ADVVError, OSError, ValueError, KeyError) as exc:
        logging.error("%s: %s", type(exc).__name__, exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
