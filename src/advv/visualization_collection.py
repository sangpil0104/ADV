"""Group accepted previews by kind and migrate saved runs without inference."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from .errors import DataError
from .gallery import gallery_records, render_gallery
from .storage import atomic_bytes, atomic_json, digest, file_hash, read_json, within
from .visualization import LAYOUT_VERSION, artifact_paths

COLLECTION_VERSION = "by_kind_collection_v1"
KINDS = ("comparison", "drag_plan", "metadata")


def _replace_paths(value, mapping):
    if isinstance(value, str):
        return mapping.get(value, value)
    if isinstance(value, list):
        return [_replace_paths(v, mapping) for v in value]
    if isinstance(value, dict):
        return {k: _replace_paths(v, mapping) for k, v in value.items()}
    return value


def _checked(root, relative):
    from .cleanup import _validate_tree

    return _validate_tree(root, relative)


def _migrate_record(root, record):
    cid = record["candidate_id"]
    previous = record.get("visualization", {}).get("artifacts", {})
    destinations = previous | artifact_paths(cid)
    mapping = {}
    for kind in KINDS:
        old = previous.get(kind)
        if not old:
            raise DataError(f"Missing accepted {kind}: {cid}")
        source = _checked(root, old)
        if not source.is_file():
            raise DataError(f"Missing accepted visualization: {old}")
        destination = _checked(root, destinations[kind])
        mapping[old] = destinations[kind]
        # Also keep the canonical old mapping after an interrupted migration updated its record.
        ext = "json" if kind == "metadata" else "png"
        mapping[f"visualizations/{cid}/{kind}.{ext}"] = destinations[kind]
        if kind != "metadata" and source != destination:
            if destination.exists() and file_hash(destination) != file_hash(source):
                raise DataError(f"Conflicting visualization destination: {destinations[kind]}")
            if not destination.exists():
                atomic_bytes(destination, source.read_bytes())
    original_metadata = read_json(within(root, previous["metadata"]))
    if original_metadata.get("candidate_id") != cid:
        raise DataError("Visualization metadata belongs to a different candidate")
    metadata = copy.deepcopy(original_metadata)
    metadata["artifacts"] = destinations
    metadata["layout_version"] = LAYOUT_VERSION
    target_metadata = within(root, destinations["metadata"])
    if target_metadata.exists() and previous["metadata"] != destinations["metadata"]:
        if read_json(target_metadata) != metadata:
            raise DataError(f"Conflicting visualization metadata destination: {cid}")
    elif not target_metadata.exists() or read_json(target_metadata) != metadata:
        atomic_json(target_metadata, metadata)
    updated = copy.deepcopy(record)
    updated["visualization"].update(artifacts=destinations, layout_version=LAYOUT_VERSION)
    if updated != record:
        # All new files are durable before this record publishes their paths.
        atomic_json(within(root, f"records/{cid}.json"), updated)
    return updated, mapping


def _rewrite_references(root, mapping):
    for name in ("report.json",):
        path = _checked(root, name)
        if path.is_file():
            original = read_json(path)
            updated = _replace_paths(original, mapping)
            if original != updated:
                atomic_json(path, updated)
    path = _checked(root, "report.md")
    if path.is_file():
        original = path.read_text(encoding="utf-8")
        updated = original
        for old, new in mapping.items():
            updated = updated.replace(f"]({old})", f"]({new})")
        if updated != original:
            atomic_bytes(path, updated.encode())
    for name in ("exports/images.jsonl", "exports/legacy_images.jsonl"):
        path = _checked(root, name)
        if not path.is_file():
            continue
        original = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        updated = _replace_paths(original, mapping)
        if updated != original:
            atomic_bytes(
                path,
                "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in updated).encode(),
            )


def _remove_old_previews(root, records):
    # Delete only the three migrated files, after every reference and new file is durable.
    # Unknown user files in an old candidate folder are left alone.
    for record in records:
        cid = record["candidate_id"]
        paths = artifact_paths(cid)
        for kind in KINDS:
            ext = "json" if kind == "metadata" else "png"
            old = _checked(root, f"visualizations/{cid}/{kind}.{ext}")
            if not old.exists():
                continue
            new = within(root, paths[kind], must_exist=True)
            if old == new:
                continue
            if kind == "metadata":
                expected = read_json(old)
                expected["artifacts"] = record["visualization"]["artifacts"]
                expected["layout_version"] = LAYOUT_VERSION
                if expected != read_json(new):
                    raise DataError("Migrated metadata changed before old-path cleanup")
            elif file_hash(old) != file_hash(new):
                raise DataError("Migrated preview changed before old-path cleanup")
            old.unlink()
        folder = _checked(root, f"visualizations/{cid}")
        if folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()


def build_collection(root: Path):
    """Caller holds run_lock. Record→links→index are updated before old paths are removed."""
    root = root.resolve()
    records = gallery_records(root)
    legacy = read_json(root / "manifest.json")["schema_version"] == "1.1"
    mapping, migrated = {}, []
    for record in records:
        updated, replacements = _migrate_record(root, record)
        mapping.update(replacements)
        migrated.append(updated)
    _rewrite_references(root, mapping)
    for kind in KINDS:
        _checked(root, f"visualizations/{kind}").mkdir(parents=True, exist_ok=True)
    gallery = render_gallery(root, migrated)
    plan_gallery = render_gallery(root, migrated, kind="drag_plan")
    items = [
        {
            "index": i,
            "candidate_id": r["candidate_id"],
            "source_id": r["source_id"],
            "image_path": r["image_path"],
            "artifacts": r["visualization"]["artifacts"],
            "verification_status": "legacy_unverified" if legacy else "clip_verified",
        }
        for i, r in enumerate(migrated, 1)
    ]
    collection = {
        "version": COLLECTION_VERSION,
        "run_id": root.name,
        "count": len(items),
        "verification_status": "legacy_unverified" if legacy else "clip_verified",
        "artifacts": {kind: f"visualizations/{kind}/" for kind in KINDS}
        | {
            "generated_grid": gallery["image_path"],
            "drag_plan_grid": plan_gallery["image_path"],
            "index": "visualizations/index.json",
        },
        "identity": digest([COLLECTION_VERSION, items, gallery["identity"], plan_gallery["identity"]]),
        "items": items,
    }
    path = _checked(root, "visualizations/index.json")
    if not path.exists() or read_json(path) != collection:
        atomic_json(path, collection)
    _remove_old_previews(root, migrated)
    return collection


def build_all_collections(project: Path):
    from .storage import run_lock

    results = []
    for base in (project / "assets/outputs", project / "runs"):
        for path in sorted(base.glob("*/manifest.json")):
            root = path.parent
            if root.is_symlink():
                continue
            manifest = read_json(path)
            if manifest.get("schema_version") not in ("1.1", "2.0") or manifest.get("run_id") != root.name:
                continue
            try:
                with run_lock(root):
                    collection = build_collection(root)
                results.append({"run_dir": str(root), "status": "ready", "count": collection["count"]})
            except DataError as exc:
                if str(exc).startswith("Another writer holds this run"):
                    results.append({"run_dir": str(root), "status": "skipped_active"})
                    continue
                raise
    return results
