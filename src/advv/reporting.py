"""Accepted-only reports and exports; failures appear only as counts and minimal timing."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .errors import DataError
from .storage import atomic_bytes, atomic_json, file_hash, pixel_hash, read_json, within
from .timing import attempts_for
from .verifiers.decision import decide


def records_for(run_dir):
    return sorted([read_json(p) for p in (run_dir / "records").glob("*.json")], key=lambda r: r["sequence"])


def accepted_records(run_dir):
    return [
        r
        for r in records_for(run_dir)
        if r.get("export_status") == "eligible"
        and (r.get("schema_version") != "2.0" or r.get("presentation_complete"))
    ]


def export_run(run_dir: Path) -> Path:
    manifest = read_json(run_dir / "manifest.json")
    if manifest["backend"] not in ("dragflow+clip_local", "dragflow+qwen_local"):
        raise DataError("Production export refuses fake or unrecognized backends")
    legacy = manifest["schema_version"] == "1.1"
    sources = {s["source_id"]: s for s in read_json(run_dir / "sources.json")}
    seen = {s["pixel_sha256"] for s in sources.values()}
    cfg = read_json(run_dir / "config.json")
    rows = []
    for r in accepted_records(run_dir):
        source = sources[r["source_id"]]
        if r["backend"] != manifest["backend"] or source["split"] not in ("source", "train"):
            raise DataError("Ineligible record in export")
        if not legacy and (
            decide(r["clip"]) != "accepted"
            or r["clip"]["criteria_sha256"] != cfg["_assets"]["criteria_sha256"]
        ):
            raise DataError("Invalid CLIP decision in export")
        path = within(run_dir, r["image_path"], must_exist=True)
        pixels = pixel_hash(path)
        if pixels != r["pixel_sha256"] or file_hash(path) != r["file_sha256"] or pixels in seen:
            raise DataError("Export artifact failed hash/dedup verification")
        seen.add(pixels)
        rows.append(
            {
                "schema_version": manifest["schema_version"],
                "candidate_id": r["candidate_id"],
                "source_id": r["source_id"],
                "image_path": r["image_path"],
                "source_image_path": source["image_path"],
                "split": source["split"],
                "group_id": source["group_id"],
                "domain": source["domain"],
                "label": None,
                "task": "image_only",
                "edit_task": r.get("plan", {}).get("sampler_version"),
                "accident_vehicle_ids": (r.get("plan", {}).get("accident_group") or {}).get("member_ids"),
                "motion_diagnostics": r.get("generation_info", {}).get("motion_diagnostics"),
                "pixel_sha256": pixels,
                "file_sha256": r["file_sha256"],
                "backend": r["backend"],
                "verification_status": "legacy_unverified" if legacy else "clip_verified",
                "record_path": f"records/{r['candidate_id']}.json",
                "visualization_artifacts": r.get("visualization", {}).get("artifacts", {}),
                "timing": r.get("timing"),
            }
        )
    destination = run_dir / "exports/images.jsonl"
    payload = "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows).encode()
    if legacy:
        atomic_bytes(run_dir / "exports/legacy_images.jsonl", payload)
        atomic_bytes(destination, b"")  # legacy images do not enter the CLIP dataset
    else:
        atomic_bytes(destination, payload)
    return destination


def report_run(run_dir: Path, *, final_status=None) -> Path:
    from .visualization_collection import build_collection

    state = read_json(run_dir / "state.json")
    cfg = read_json(run_dir / "config.json")
    manifest = read_json(run_dir / "manifest.json")
    legacy = manifest["schema_version"] == "1.1"
    collection = build_collection(run_dir)
    accepted = accepted_records(run_dir)
    entries = attempts_for(run_dir)
    counts = dict(Counter(r["outcome"] for r in entries))
    sources = read_json(run_dir / "sources.json")
    timer = read_json(run_dir / "timings/state.json") if (run_dir / "timings/state.json").exists() else {}
    total = timer.get("run_active_seconds")
    count = 0 if legacy else len(accepted)
    result = {
        "run_id": run_dir.name,
        "backend": manifest["backend"],
        "schema_version": manifest["schema_version"],
        "edit_task": cfg.get("sampler", {}).get("version"),
        "status": final_status or state["status"],
        "accepted": count,
        "legacy_unverified": len(accepted) if legacy else 0,
        "target": cfg["run"]["target_count"],
        "attempts": len(entries),
        "status_counts": counts,
        "run_active_seconds": total,
        "timing_basis": "shared_run_active_wall_seconds" if not legacy else "legacy_partial",
        "execution_segments": state.get("segments", []),
        "pending_cost_seconds": timer.get("pending_cost_seconds"),
        "timing_complete": timer.get("timing_complete", False),
        "mean_active_seconds_per_accepted": total / count if count and total is not None else None,
        "per_source": {
            s["source_id"]: {
                "attempts": sum(e["source_id"] == s["source_id"] for e in entries),
                "accepted": sum(r["source_id"] == s["source_id"] for r in accepted) if not legacy else 0,
                "held": state.get("held_sources", {}).get(s["source_id"]),
            }
            for s in sources
        },
        "held_sources": state.get("held_sources", {}),
        "last_error": {
            "code": state["last_error"].get("code", state["last_error"].get("type", "legacy_error"))
        }
        if state.get("last_error")
        else None,
        "cleanup_pending_count": sum(
            read_json(p).get("state") == "pending" for p in (run_dir / "cleanup").glob("*.json")
        ),
        "accepted_images": [
            {
                "candidate_id": r["candidate_id"],
                "image_path": r["image_path"],
                "timing": r.get("timing"),
                "verification_status": "legacy_unverified" if legacy else "clip_verified",
                "motion_diagnostics": r.get("generation_info", {}).get("motion_diagnostics"),
            }
            for r in accepted
        ],
    }
    result["visualization_collection"] = {
        "count": collection["count"],
        "artifacts": collection["artifacts"],
        "verification_status": collection["verification_status"],
    }
    atomic_bytes(
        run_dir / "timings/attempts.jsonl",
        "".join(json.dumps(e, ensure_ascii=False, allow_nan=False) + "\n" for e in entries).encode(),
    )
    atomic_json(run_dir / "report.json", result)
    lines = [
        f"# ADVV {run_dir.name}",
        "",
        f"Status: **{result['status']}** | Backend: {manifest['backend']}",
        "",
        f"CLIP accepted: **{count}/{result['target']}** | Attempts: {len(entries)}",
        f"Outcome counts: {counts}",
        f"Held sources (not attempted): {result['held_sources'] or 'none'}",
        "",
        f"Active seconds: {total} | Pending seconds: {result['pending_cost_seconds']}",
        f"Active seconds / accepted: {result['mean_active_seconds_per_accepted']}",
        "",
        "CLIP is a visual proxy, not proof of physical laws or exact accident-type preservation.",
        "Whole-group masks/translation are input contracts. Output vehicle motion is diagnosed separately, not certified by CLIP.",
        " / ".join(f"[{kind}]({path})" for kind, path in collection["artifacts"].items()),
        "",
        "| Accepted candidate | Verification | Attempt seconds | Including failures | Preview |",
        "| --- | --- | --- | --- | --- |",
    ]
    for record in accepted:
        links = [
            f"[{name}]({relative})"
            for name, relative in record.get("visualization", {}).get("artifacts", {}).items()
            if within(run_dir, relative).is_file()
        ]
        t = record.get("timing", {})
        label = "legacy_unverified" if legacy else "clip_verified"
        lines.append(
            f"| {record['candidate_id']} | {label} | {t.get('attempt_elapsed_seconds')} | "
            f"{t.get('accepted_cost_seconds')} | {' / '.join(links)} |"
        )
    path = run_dir / "report.md"
    atomic_bytes(path, ("\n".join(lines) + "\n").encode())
    return path
