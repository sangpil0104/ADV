"""Scoped, journaled deletion of terminal nonaccepted artifacts, including legacy runs."""

from __future__ import annotations

import os
import stat
from pathlib import Path

from .errors import DataError
from .storage import atomic_json, read_json, run_lock, safe_id, within
from .timing import save_attempt


def _validate_tree(root: Path, relative: str) -> Path:
    path = within(root, relative)
    current = path
    while current != root.resolve():
        if current.is_symlink():
            raise DataError("cleanup_blocked: symlink")
        current = current.parent
    if not path.exists():
        return path
    nodes = [path]
    if path.is_dir():
        for base, dirs, files in os.walk(path, followlinks=False):
            nodes.extend(Path(base) / name for name in dirs + files)
    for node in nodes:
        mode = node.lstat().st_mode
        if stat.S_ISLNK(mode) or not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise DataError("cleanup_blocked: unsupported file")
        if stat.S_ISREG(mode) and node.stat().st_nlink != 1:
            raise DataError("cleanup_blocked: hardlink")
    return path


def _remove_checked(path: Path):
    if not path.exists():
        return
    if path.is_dir():
        for child in path.iterdir():
            _remove_checked(child)
        path.rmdir()
    else:
        path.unlink()


def _owned_paths(cid, *, staging_only=False):
    safe_id(cid)
    if staging_only:
        return [f".staging/{cid}"]
    return [
        f".staging/{cid}",
        f"candidates/{cid}",
        f"quarantine/{cid}",
        f"visualizations/{cid}",
        f"visualizations/comparison/{cid}.png",
        f"visualizations/drag_plan/{cid}.png",
        f"visualizations/metadata/{cid}.json",
        f"edit_regions/{cid}.png",
        f"records/{cid}.json",
    ]


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _strings(v)


def cleanup_candidate(root: Path, cid: str, *, staging_only=False) -> None:
    """Caller holds the run lock and has persisted minimal timing before calling."""
    root = root.resolve()
    safe_id(cid)
    ledger = root / f"timings/attempts/{cid}.json"
    accepted_path = root / f"records/{cid}.json"
    if staging_only:
        if not accepted_path.is_file() or read_json(accepted_path).get("export_status") != "eligible":
            raise DataError("Cannot remove committed staging without an accepted record")
    else:
        if not ledger.is_file():
            raise DataError("Persist minimal timing before deleting candidate details")
        if read_json(ledger).get("outcome") == "accepted":
            raise DataError("Cannot clean accepted artifacts as a failure")
    paths = _owned_paths(cid, staging_only=staging_only)
    if not staging_only:
        for kind, suffix in (("comparison", "png"), ("drag_plan", "png"), ("metadata", "json")):
            folder = within(root, f"visualizations/{kind}")
            paths.extend(str(p.relative_to(root)) for p in folder.glob(f".{cid}.{suffix}.*"))
    protected = []
    for p in (root / "records").glob("*.json"):
        record = read_json(p)
        if record.get("export_status") == "eligible":
            protected.extend(_strings(record))
    for relative in paths:
        if any(v == relative or v.startswith(relative + "/") for v in protected):
            raise DataError("cleanup_blocked: accepted reference")
    checked = [_validate_tree(root, rel) for rel in paths]
    journal = within(root, f"cleanup/{cid}.json")
    atomic_json(
        journal, {"candidate_id": cid, "state": "pending", "staging_only": staging_only, "paths": paths}
    )
    for path in checked:
        _remove_checked(path)
    atomic_json(journal, {"candidate_id": cid, "state": "cleaned", "staging_only": staging_only})
    if not staging_only:
        entry = read_json(ledger)
        entry["cleanup_state"] = "cleaned"
        save_attempt(root, entry)


def recover_cleanup(root: Path):
    for path in sorted((root / "cleanup").glob("*.json")):
        journal = read_json(path)
        if journal["state"] == "pending":
            cleanup_candidate(root, journal["candidate_id"], staging_only=journal["staging_only"])


def _legacy_entry(record):
    stages = {}
    values = [a.get("elapsed_seconds") for a in record.get("generation_attempts", [])]
    stages["generation"] = sum(x for x in values if isinstance(x, (int, float))) if values else None
    for name, check in record.get("checks", {}).items():
        values = [a.get("elapsed_seconds") for a in check.get("attempts", [])]
        stages[name] = sum(x for x in values if isinstance(x, (int, float))) if values else None
    outcome = (
        "duplicate"
        if record.get("export_status") == "duplicate"
        else "rejected"
        if record["status"] in ("rejected", "uncertain")
        else "error"
    )
    return {
        "candidate_id": record["candidate_id"],
        "source_id": record["source_id"],
        "attempt_index": record.get("plan", {}).get("attempt_index"),
        "sequence": record["sequence"],
        "outcome": outcome,
        "attempt_elapsed_seconds": None,
        "stage_seconds": stages,
        "timing_complete": False,
        "cleanup_state": "pending",
    }


def cleanup_existing_runs(output_root: Path, *, exclude: Path | None = None) -> dict:
    result = {"cleaned_candidates": 0, "active_runs_skipped": 0}
    if not output_root.exists():
        return result
    for root in sorted(output_root.iterdir()):
        if root.is_symlink() or not root.is_dir() or exclude and root.resolve() == exclude.resolve():
            continue
        if not all(
            (root / name).is_file() for name in ("manifest.json", "config.json", "state.json", "sources.json")
        ):
            continue
        manifest, state = read_json(root / "manifest.json"), read_json(root / "state.json")
        if (
            manifest.get("schema_version") not in ("1.1", "2.0")
            or manifest.get("backend") not in ("dragflow+qwen_local", "dragflow+clip_local", "fake")
            or manifest.get("run_id") != root.name
        ):
            continue
        if state.get("status") not in ("completed", "failed", "interrupted"):
            result["active_runs_skipped"] += 1
            continue
        try:
            with run_lock(root):
                recover_cleanup(root)
                records = [read_json(p) for p in (root / "records").glob("*.json")]
                for record in records:
                    if record.get("export_status") == "eligible":
                        continue
                    if record.get("status") not in (
                        "rejected",
                        "uncertain",
                        "verification_error",
                        "generation_error",
                        "accepted",
                        "error",
                        "duplicate",
                    ):
                        raise DataError("cleanup_blocked: unclassified legacy record")
                    if record["status"] == "accepted" and record.get("export_status") != "duplicate":
                        raise DataError("cleanup_blocked: uncommitted acceptance")
                    cid = safe_id(record["candidate_id"])
                    if not (root / f"timings/attempts/{cid}.json").exists():
                        save_attempt(root, _legacy_entry(record))
                    cleanup_candidate(root, cid)
                    result["cleaned_candidates"] += 1
                # v2 failure details live in staging; use the terminal timing ledger as authority.
                from .timing import attempts_for

                for entry in attempts_for(root):
                    if (
                        entry["outcome"] in ("rejected", "duplicate", "error")
                        and entry.get("cleanup_state") != "cleaned"
                    ):
                        cleanup_candidate(root, entry["candidate_id"])
                # Known old worker logs are debug streams; rebuild them from accepted IDs only.
                accepted_ids = [r["candidate_id"] for r in records if r.get("export_status") == "eligible"]
                for log in (root / "logs").glob("*"):
                    if log.name not in ("generator.log", "verifier.log", "segmenter.log"):
                        raise DataError("cleanup_blocked: unclassified legacy log")
                    _validate_tree(root, str(log.relative_to(root)))
                    from .storage import atomic_bytes

                    atomic_bytes(
                        log,
                        (
                            "Legacy debug stream removed; accepted candidates: "
                            + ", ".join(accepted_ids)
                            + "\n"
                        ).encode(),
                    )
                from .reporting import export_run, report_run

                report_run(root)
                if manifest["backend"] != "fake":
                    export_run(root)
        except DataError as exc:
            if str(exc).startswith("Another writer holds this run"):
                result["active_runs_skipped"] += 1
                continue
            raise
    return result
