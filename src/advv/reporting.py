from __future__ import annotations

import html
import json
from collections import Counter
from pathlib import Path

from .backends.dragflow_speedups import parse_speedups, speedup_record
from .backends.process import LocalBackend
from .config import require_clip_run
from .errors import DataError
from .human_review import PASS
from .human_review import enabled as human_review_enabled
from .human_review import summary as human_summary
from .storage import atomic_bytes, atomic_json, file_hash, pixel_hash, read_json, within
from .verifiers.clip_filter import check_frozen, decide, frozen_sentences, frozen_source


def records_for(run_dir):
    return sorted([read_json(p) for p in (run_dir / "records").glob("*.json")], key=lambda r: r["sequence"])


def export_speedups(record: dict, configured: dict[str, bool]) -> dict:
    """Per-row execution-path summary; the candidate's own generation_info wins over the run config."""
    info = (record.get("generation_info") or {}).get("speedups") or speedup_record(configured)
    return {
        "official_execution_path": info["official_execution_path"],
        "enabled": info["enabled"],
        "tf32": "tf32" in info["enabled"],
    }


def export_filter(record: dict) -> dict:
    """Per-row CLIP filter record: every sentence's source/candidate score and delta, plus identity."""
    f = record["clip_filter"]
    keys = ("verdict", "limit", "top", "similarity", "criterion", "model_id", "model_revision")
    return {
        **{k: f[k] for k in keys},
        "sentences_sha256": f["sentences_sha256"],
        "sentences_file_sha256": f["sentences_file_sha256"],
        "sentences": [{k: r[k] for k in ("text", "source", "candidate", "delta")} for r in f["sentences"]],
    }


def export_run(run_dir: Path) -> Path:
    manifest = read_json(run_dir / "manifest.json")
    cfg = read_json(run_dir / "config.json")
    require_clip_run(cfg)
    if manifest["backend"] != LocalBackend.provenance:
        raise DataError("Production export refuses fake or unrecognized backends")
    sources = {s["source_id"]: s for s in read_json(run_dir / "sources.json")}
    review = human_review_enabled(cfg)
    configured = parse_speedups(cfg["generator"].get("speedups"))
    sentences = cfg["_assets"]["implausible_sentences"]
    frozen_sentences(run_dir, sentences, cfg["_assets"]["implausible_sentences_sha256"])
    seen = {s["pixel_sha256"] for s in sources.values()}
    rows = []
    for r in records_for(run_dir):
        if r.get("export_status") != "eligible":
            continue
        # With the optional human stage, only images a person passed enter the dataset.
        if review and r.get("human_review", {}).get("decision") != PASS:
            continue
        source = sources[r["source_id"]]
        if (
            r["backend"] != manifest["backend"]
            or decide(r) != "accepted"
            or r["clip_filter"]["limit"] != cfg["verifier"]["max_delta"]
            or r["clip_filter"]["sentences_sha256"] != cfg["_assets"]["implausible_sentences_sha256"]
            or source["split"] not in ("source", "train")
        ):
            raise DataError("Ineligible record in export")
        check_frozen(r, sentences, frozen_source(run_dir, r["source_id"]))
        path = within(run_dir, r["image_path"], must_exist=True)
        pixels = pixel_hash(path)
        if (
            pixels != r["pixel_sha256"]
            or file_hash(path) != r["file_sha256"]
            or (review and r["human_review"]["file_sha256"] != r["file_sha256"])
            or pixels in seen
        ):
            raise DataError("Export artifact failed hash/dedup verification")
        seen.add(pixels)
        rows.append(
            {
                "schema_version": "1.1",
                "candidate_id": r["candidate_id"],
                "source_id": r["source_id"],
                "image_path": r["image_path"],
                "source_image_path": source["image_path"],
                "split": source["split"],
                "group_id": source["group_id"],
                "domain": source["domain"],
                "label": None,
                "task": "image_only",
                "pixel_sha256": pixels,
                "file_sha256": r["file_sha256"],
                "backend": r["backend"],
                "generator_speedups": export_speedups(r, configured),
                "clip_filter": export_filter(r),
                "record_path": f"records/{r['candidate_id']}.json",
                "human_review": r.get("human_review"),
                "visualization_artifacts": r.get("visualization", {}).get("artifacts", {}),
            }
        )
    destination = run_dir / "exports/images.jsonl"
    atomic_bytes(destination, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode())
    return destination


def filter_statistics(records: list[dict], cfg: dict) -> dict:
    """Which sentences reject candidates, and how far each sentence rises over its source."""
    judged = [r["clip_filter"] for r in records if r.get("clip_filter")]
    rejected = [f for f in judged if f["verdict"] == "rejected"]
    deltas: dict[str, list[float]] = {t: [] for t in cfg["_assets"]["implausible_sentences"]}
    for f in judged:
        for row in f["sentences"]:
            deltas.setdefault(row["text"], []).append(row["delta"])
    v = cfg["verifier"]
    return {
        "model_id": v["model_id"],
        "model_revision": v["model_revision"],
        "similarity": v["similarity"],
        "criterion": v["criterion"],
        "limit": v["max_delta"],
        "sentences_sha256": cfg["_assets"]["implausible_sentences_sha256"],
        "sentences_file_sha256": cfg["_assets"]["implausible_sentences_file_sha256"],
        "judged": len(judged),
        "rejected": len(rejected),
        # A candidate rejected by several sentences counts once for each of them.
        "rejected_by_sentence": dict(
            Counter(f["sentences"][i]["text"] for f in rejected for i in f["exceeded"]).most_common()
        ),
        "top_sentence_of_rejected": dict(Counter(f["top"]["text"] for f in rejected).most_common()),
        "delta_by_sentence": {
            text: {"n": len(values), "max": max(values), "mean": sum(values) / len(values)} if values else {"n": 0}
            for text, values in deltas.items()
        },
    }


def report_run(run_dir: Path, *, final_status=None) -> Path:
    state = read_json(run_dir / "state.json")
    if final_status is not None:
        state["status"] = final_status
    records = records_for(run_dir)
    counts = dict(Counter(r["status"] for r in records))
    sources = read_json(run_dir / "sources.json")
    per_source = {}
    for source in sources:
        subset = [r for r in records if r["source_id"] == source["source_id"]]
        per_source[source["source_id"]] = {
            "attempts": len(subset),
            "accepted": sum(r.get("export_status") == "eligible" for r in subset),
        }
    accepted = sum(r.get("export_status") == "eligible" for r in records)
    cfg = read_json(run_dir / "config.json")
    require_clip_run(cfg)
    review = human_review_enabled(cfg)
    result = {
        "run_id": run_dir.name,
        "backend": read_json(run_dir / "manifest.json")["backend"],
        "status": state["status"],
        "accepted": accepted,
        "target": cfg["run"]["target_count"],
        "attempts": len(records),
        "status_counts": counts,
        "clip_filter": filter_statistics(records, cfg),
        "duplicates": sum(r.get("export_status") == "duplicate" for r in records),
        "human_review": {"enabled": review, **(human_summary(records) if review else {})},
        "generator_speedups": speedup_record(parse_speedups(cfg["generator"].get("speedups"))),
        "acceptance_rate": accepted / len(records) if records else None,
        "per_source": per_source,
        "region_proposals": state.get("region_proposals"),
        # Experimental object_region_v2 restriction; null means every usable proposal was selectable.
        "region_phrase_filter": (cfg["sampler"].get("object_region") or {}).get("region_phrase_filter"),
        "sampling_skipped": state.get("sampling_skipped"),
        "last_error": state.get("last_error"),
        "cleanup_pending": [r["candidate_id"] for r in records if r.get("cleanup_pending")],
        "visualization_errors": [
            r["candidate_id"] for r in records if r.get("visualization", {}).get("status") == "error"
        ],
    }
    profiles = [read_json(p) for p in (run_dir / "profiles").glob("*.json")]
    references = [read_json(p) for p in sorted((run_dir / "clip").rglob("*.json")) if "_receipts" not in str(p.relative_to(run_dir))]
    stage_seconds = {
        "generation": sum(
            a.get("elapsed_seconds", 0) for r in records for a in r.get("generation_attempts", [])
        )
    }
    for stage, owners in [("profile", profiles), ("clip_reference", references), ("clip", records)]:
        stage_seconds[stage] = sum(
            a.get("elapsed_seconds", 0)
            for owner in owners
            for check in owner.get("checks", {}).values()
            for a in check.get("attempts", [])
        )
    result["stage_elapsed_seconds"] = stage_seconds
    result["peak_vram_bytes"] = {
        "generator_max_single_logical_device": max(
            [v for r in records for v in r.get("generation_info", {}).get("peak_vram_bytes", [])] or [0]
        ),
        "qwen_max_single_logical_device": peak(profiles),
        "clip_max_single_logical_device": peak(records + references),
    }
    atomic_json(run_dir / "report.json", result)
    lines = [
        f"# ADVV {run_dir.name}",
        "",
        f"Status: **{state['status']}** | Backend: `{result['backend']}`",
        "",
        speedup_line(result["generator_speedups"]),
        "",
    ]
    if result["region_phrase_filter"] is not None:
        lines += [
            "Region phrase filter (experimental): only proposals matching "
            f"`{html.escape(json.dumps(result['region_phrase_filter']))}` were selectable.",
            "",
        ]
    lines += [
        f"Accepted: **{accepted}/{result['target']}** | Attempts: {len(records)} | Duplicates: {result['duplicates']}",
        "",
        (
            "Human review: pass {pass} / fail {fail} / pending {pending}. "
            "Export contains human-passed images only.".format(**result["human_review"])
            if review
            else "Human review: disabled. Export contains CLIP-filter-accepted images."
        ),
        "",
        filter_lines(result["clip_filter"]),
        "",
        "CLIP similarities are visual-text estimates, not physical proof or supervised ground truth. "
        "Fine local artifacts are outside this filter.",
        "",
        "| Candidate | Final status | Top delta | Export | Human | Preview |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for record in records:
        links = []
        for name, relative in record.get("visualization", {}).get("artifacts", {}).items():
            if within(run_dir, relative).is_file():
                links.append(f"[{name}]({relative})")
        lines.append(
            f"| {html.escape(record['candidate_id'])} | {record['status']} | {top_delta(record)} "
            f"| {record.get('export_status', 'pending')} "
            f"| {record.get('human_review', {}).get('decision', '-')} | {' / '.join(links)} |"
        )
    if state.get("last_error"):
        lines.extend(["", "Last error: " + html.escape(str(state["last_error"]))])
    path = run_dir / "report.md"
    atomic_bytes(path, ("\n".join(lines) + "\n").encode())
    return path


def peak(owners: list[dict]) -> int:
    return max(
        [
            (a.get("info") or {}).get("peak_vram_bytes", 0)
            for owner in owners
            for c in owner.get("checks", {}).values()
            for a in c.get("attempts", [])
        ]
        or [0]
    )


def top_delta(record: dict) -> str:
    f = record.get("clip_filter")
    return f"{f['top']['delta']:+.3f}" if f else "-"


def filter_lines(stats: dict) -> str:
    lines = [
        f"CLIP filter: `{stats['model_id']}@{stats['model_revision']}`, {stats['similarity']} similarity; "
        f"rejected when any sentence rises over the source by more than {stats['limit']} "
        f"(sentences `{stats['sentences_sha256'][:12]}`). Judged {stats['judged']}, rejected {stats['rejected']}.",
    ]
    if stats["rejected_by_sentence"]:
        lines += ["", "| Rejecting sentence | Candidates |", "| --- | --- |"]
        lines += [f"| {html.escape(t)} | {n} |" for t, n in stats["rejected_by_sentence"].items()]
    return "\n".join(lines)


def speedup_line(record: dict) -> str:
    if record["official_execution_path"]:
        return "DragFlow execution: official path (no speedup patches)."
    line = "DragFlow execution: **modified** by speedups " + ", ".join(f"`{k}`" for k in record["enabled"]) + "."
    if record["precision_variants"]:
        line += " Precision variant (not the official fp32 recipe): " + ", ".join(record["precision_variants"]) + "."
    return line
