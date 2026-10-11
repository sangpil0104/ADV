"""Rebuild accepted drag previews only; no generation, segmentation or CLIP inference."""

from __future__ import annotations

import argparse
from pathlib import Path

from advv.environment import require_advv
from advv.errors import DataError
from advv.reporting import accepted_records, report_run
from advv.storage import atomic_json, read_json, run_lock
from advv.visualization import render
from advv.visualization_collection import build_collection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    require_advv()
    root = args.run_dir.resolve()
    with run_lock(root):
        cfg = read_json(root / "config.json")
        if cfg.get("schema_version") != "2.0":
            raise DataError("Use the original renderer for legacy VQA runs")
        sources = {s["source_id"]: s for s in read_json(root / "sources.json")}
        build_collection(root)
        for record in accepted_records(root):
            record["visualization"] = render(root, sources[record["source_id"]], record, cfg["visualization"])
            atomic_json(root / f"records/{record['candidate_id']}.json", record)
            print(root / record["visualization"]["artifacts"]["comparison"])
        report_run(root)


if __name__ == "__main__":
    main()
