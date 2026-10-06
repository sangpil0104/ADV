"""Per-sentence CLIP deltas for (source, candidate) pairs, for calibrating verifier.max_delta (T026).

Reads images only and writes one JSON under a new output directory. It never changes runs.
Run with the CLIP environment, e.g.
    .venv-qwen/bin/python scripts/clip_calibration.py --pairs pairs.json --device cpu --out _workspace/<new_dir>
    CUDA_VISIBLE_DEVICES=<A> .venv-qwen/bin/python scripts/clip_calibration.py ... --device cuda:0
pairs.json: [{"name": "...", "source": "path", "candidate": "path", "label": "normal|violation|..."}]
--from-runs adds every exported (eligible) candidate of the given run directories with label "accepted".
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from advv.backends.clip import Clip
from advv.storage import atomic_json, read_json
from advv.verifiers.clip_filter import load_sentences, normalized, similarities

PROJECT = Path(__file__).resolve().parents[1]


def run_pairs(run_dir: Path) -> list[dict]:
    sources = {s["source_id"]: s for s in read_json(run_dir / "sources.json")}
    pairs = []
    for path in sorted((run_dir / "records").glob("*.json")):
        record = read_json(path)
        if record.get("export_status") == "eligible" and record.get("image_path"):
            pairs.append(
                {
                    "name": f"{run_dir.name}/{record['candidate_id']}",
                    "source": str(run_dir / sources[record["source_id"]]["image_path"]),
                    "candidate": str(run_dir / record["image_path"]),
                    "label": "accepted",
                }
            )
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", type=Path)
    parser.add_argument("--from-runs", type=Path, nargs="*", default=[])
    parser.add_argument("--sentences", type=Path, default=PROJECT / "prompts/implausible_physics_v1.txt")
    parser.add_argument("--device", required=True, help="cpu, or cuda:0 with CUDA_VISIBLE_DEVICES set")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--out", type=Path, required=True, help="New directory for calibration.json")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not os.environ.get("CUDA_VISIBLE_DEVICES"):
        print("Set CUDA_VISIBLE_DEVICES to the checked GPU before using --device cuda:0", file=sys.stderr)
        return 1
    pairs = (json.loads(args.pairs.read_text()) if args.pairs else []) + [
        p for run in args.from_runs for p in run_pairs(run.resolve())
    ]
    if not pairs:
        print("No pairs given", file=sys.stderr)
        return 1
    out = args.out.resolve()
    if out.exists() or out.is_relative_to(PROJECT / "runs"):
        print("--out must be a new directory outside runs/", file=sys.stderr)
        return 1
    lock = read_json(PROJECT / "models/weights.lock.json")["models"]["openai/clip-vit-large-patch14"]
    cfg = {
        "verifier": {
            "model_path": lock["path"],
            "model_id": "openai/clip-vit-large-patch14",
            "model_revision": lock["revision"],
            "dtype": args.dtype,
        }
    }
    clip = Clip(cfg, device=args.device)
    sentences = load_sentences(args.sentences)
    texts = normalized(clip({"texts": sentences, "images": []})["text_embeddings"], count=len(sentences))
    cache: dict[str, list[float]] = {}

    def sims(path: str) -> np.ndarray:
        if path not in cache:
            image = clip({"texts": [], "images": [path]})["image_embeddings"]
            cache[path] = similarities(normalized(image, count=1, dim=texts.shape[1])[0], texts)
        return np.array(cache[path])

    rows = []
    for pair in pairs:
        delta = sims(pair["candidate"]) - sims(pair["source"])
        top = int(delta.argmax())
        rows.append({**pair, "deltas": delta.tolist(), "max_delta": float(delta[top]), "max_sentence": sentences[top]})
        print(f"{pair['name']} [{pair.get('label')}]: {delta[top]:+.4f} {sentences[top]}", flush=True)
    out.mkdir(parents=True)
    atomic_json(
        out / "calibration.json",
        {
            "model_revision": lock["revision"],
            "device": args.device,
            "dtype": args.dtype,
            "sentences": sentences,
            "similarity": "cosine",
            "pairs": rows,
        },
    )
    print(out / "calibration.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
