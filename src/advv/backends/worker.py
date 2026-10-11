from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

from ..environment import require_advv
from ..errors import BackendError, FatalBackendError
from ..storage import atomic_json, read_json, within


def main():
    require_advv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["generator", "verifier", "segmenter"], required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    cfg = read_json(args.run_dir / "config.json")
    from ..pipeline import implementation_hash

    if read_json(args.run_dir / "manifest.json")["implementation_sha256"] != implementation_hash():
        raise FatalBackendError("worker_implementation_changed")
    backend = None
    for line in sys.stdin:
        request = json.loads(line)
        started = time.monotonic()
        try:
            with contextlib.redirect_stdout(sys.stderr):
                if backend is None:
                    if args.kind == "verifier":
                        from .clip import CLIPVerifier

                        backend = CLIPVerifier(cfg)
                    elif args.kind == "segmenter":
                        from .segmentation import VehicleSegmenter

                        backend = VehicleSegmenter(cfg, args.run_dir)
                    else:
                        from .dragflow import DragFlow

                        backend = DragFlow(cfg, args.run_dir)
                result = backend(request)
            result.setdefault("info", {})["elapsed_seconds"] = time.monotonic() - started
            result["info"]["python_prefix"] = sys.prefix
            result["info"]["visible_gpu_ids"] = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            response = {"ok": True, "result": result, "identity": request["identity"]}
        except Exception as exc:
            # No prompt, rejected score, image path, traceback or raw model output in public logs.
            response = {
                "ok": False,
                "identity": request["identity"],
                "error": {
                    "code": type(exc).__name__,
                    "fatal": isinstance(exc, FatalBackendError) or not isinstance(exc, BackendError),
                    "elapsed_seconds": time.monotonic() - started,
                },
            }
        if request.get("receipt"):
            atomic_json(within(args.run_dir, request["receipt"]), response)
        print(json.dumps(response, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
