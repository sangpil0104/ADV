import copy
import json
import shutil

import pytest
from PIL import Image, ImageChops, ImageOps

from advv.cli import main
from advv.errors import DataError
from advv.pipeline import Pipeline, create_run
from advv.storage import atomic_bytes, atomic_json, file_hash, load_rgb, read_json, run_lock
from advv.visualization import artifact_paths, render_plan_panel
from advv.visualization_collection import build_all_collections, build_collection
from conftest import FakeBackend


def test_all_accepted_grouped_in_order_and_rejections_absent(cfg):
    cfg["run"]["target_count"] = 3
    root = create_run(cfg, "collection", provenance="fake")
    backend = FakeBackend(cfg, outcomes=[True, False, True, True])
    assert Pipeline(root, backend).run()["status"] == "completed"
    index = read_json(root / "visualizations/index.json")
    assert index["count"] == 3
    accepted = sorted([read_json(p) for p in (root / "records").glob("*.json")], key=lambda r: r["sequence"])
    ids = [r["candidate_id"] for r in accepted]
    assert [r["candidate_id"] for r in index["items"]] == ids
    for kind in ("comparison", "drag_plan", "metadata"):
        assert len(list((root / "visualizations" / kind).iterdir())) == 3
    grid = read_json(root / "visualizations/generated_grid.json")
    assert [r["candidate_id"] for r in grid["images"]] == ids
    plans = read_json(root / "visualizations/drag_plan_grid.json")
    assert [r["candidate_id"] for r in plans["images"]] == ids
    assert plans["count"] == grid["count"] == 3
    assert plans["columns"] == grid["columns"]
    assert [r["image_path"] for r in plans["images"]] == [artifact_paths(cid)["drag_plan"] for cid in ids]
    assert index["artifacts"]["drag_plan_grid"] == "visualizations/drag_plan_grid.png"
    failed = backend.generations[1]["edit_id"]
    assert failed not in (root / "visualizations/index.json").read_text()
    assert not any(failed in str(p) for p in (root / "visualizations").rglob("*"))
    for record in accepted:
        assert record["visualization"]["artifacts"] == artifact_paths(record["candidate_id"])
        assert not (root / "visualizations" / record["candidate_id"]).exists()
    with Image.open(root / "visualizations/generated_grid.png") as image:
        assert image.getpixel((12 + 240, 90 + 135)) == (1, 15, 37)
    record = accepted[0]
    source = read_json(root / "sources.json")[0]
    metadata = read_json(root / record["visualization"]["artifacts"]["metadata"])
    preview, _ = render_plan_panel(
        root,
        load_rgb(root / source["image_path"]),
        record["plan"],
        metadata["display_geometry"],
        cfg["visualization"],
        show_labels=False,
    )
    thumb = ImageOps.contain(preview, (720, 405), Image.Resampling.LANCZOS)
    with Image.open(root / plans["image_path"]) as overview:
        assert overview.size == (1476, 846)
        x, y = 12 + (720 - thumb.width) // 2, 12 + (405 - thumb.height) // 2
        actual = overview.crop((x, y, x + thumb.width, y + thumb.height))
        assert ImageChops.difference(actual, thumb).getbbox() is None
    assert read_json(root / "report.json")["visualization_collection"]["count"] == 3
    assert (
        read_json(root / "report.json")["visualization_collection"]["artifacts"]["drag_plan_grid"]
        == plans["image_path"]
    )


def test_rebuild_idempotent_and_repairs_gallery_without_inference(cfg, monkeypatch):
    root = create_run(cfg, "rebuild", provenance="fake")
    backend = FakeBackend(cfg)
    assert Pipeline(root, backend).run()["status"] == "completed"
    protected = [
        p
        for folder in ("records", "accepted", "inputs", "artifacts", "timings")
        for p in (root / folder).rglob("*")
        if p.is_file()
    ]
    before = {p: file_hash(p) for p in protected}
    outputs = [p for p in (root / "visualizations").rglob("*") if p.is_file()]
    unchanged = {p: (file_hash(p), p.stat().st_mtime_ns) for p in outputs}
    with run_lock(root):
        build_collection(root)
    assert {p: (file_hash(p), p.stat().st_mtime_ns) for p in outputs} == unchanged
    (root / "visualizations/generated_grid.png").unlink()
    with run_lock(root):
        build_collection(root)
    (root / "visualizations/drag_plan_grid.png").unlink()

    def no_text(*args, **kwargs):
        raise AssertionError("Photo-only drag plan gallery must not draw text")

    with monkeypatch.context() as patch:
        patch.setattr("PIL.ImageDraw.ImageDraw.text", no_text)
        with run_lock(root):
            build_collection(root)
    assert (root / "visualizations/generated_grid.png").is_file()
    assert (root / "visualizations/drag_plan_grid.png").is_file()
    for name in ("generated_grid.png", "drag_plan_grid.png"):
        path = root / "visualizations" / name
        assert file_hash(path) == unchanged[path][0]
    assert {p: file_hash(p) for p in protected} == before
    assert len(backend.generations) == 1 and len(backend.calls) == 1


def make_old_layout(root, *, legacy=False):
    p = next((root / "records").glob("*.json"))
    record = read_json(p)
    cid = record["candidate_id"]
    old = {
        kind: f"visualizations/{cid}/{kind}.{'json' if kind == 'metadata' else 'png'}"
        for kind in ("comparison", "drag_plan", "metadata")
    }
    current = record["visualization"]["artifacts"]
    metadata = read_json(root / current["metadata"])
    hashes = {kind: file_hash(root / current[kind]) for kind in ("comparison", "drag_plan")}
    for kind in ("comparison", "drag_plan"):
        atomic_bytes(root / old[kind], (root / current[kind]).read_bytes())
        (root / current[kind]).unlink()
    metadata["artifacts"] = old
    metadata.pop("layout_version", None)
    atomic_json(root / old["metadata"], metadata)
    (root / current["metadata"]).unlink()
    record["visualization"]["artifacts"] = old
    record["visualization"].pop("layout_version", None)
    if legacy:
        record["schema_version"] = "1.1"
        record.pop("clip")
        manifest = read_json(root / "manifest.json")
        manifest["schema_version"] = "1.1"
        atomic_json(root / "manifest.json", manifest)
    atomic_json(p, record)
    export = {
        "candidate_id": cid,
        "image_path": record["image_path"],
        "verification_status": "unchanged",
        "visualization_artifacts": old,
    }
    atomic_bytes(root / "exports/images.jsonl", (json.dumps(export) + "\n").encode())
    atomic_bytes(root / "report.md", f"[comparison]({old['comparison']})\n".encode())
    return record, hashes, metadata


@pytest.mark.parametrize("legacy", [False, True])
def test_existing_results_move_losslessly_and_update_links(cfg, legacy):
    root = create_run(cfg, "migration", provenance="fake")
    Pipeline(root, FakeBackend(cfg)).run()
    original, hashes, old_metadata = make_old_layout(root, legacy=legacy)
    cid = original["candidate_id"]
    originals = {
        p: file_hash(p)
        for folder in ("accepted", "inputs", "artifacts", "timings")
        for p in (root / folder).rglob("*")
        if p.is_file()
    }
    with run_lock(root):
        result = build_collection(root)
    record = read_json(root / f"records/{cid}.json")
    updated = copy.deepcopy(record)
    updated["visualization"] = original["visualization"]
    assert updated == original
    assert not (root / "visualizations" / cid).exists()
    for kind, checksum in hashes.items():
        assert file_hash(root / artifact_paths(cid)[kind]) == checksum
    metadata = read_json(root / artifact_paths(cid)["metadata"])
    metadata["artifacts"] = old_metadata["artifacts"]
    metadata.pop("layout_version")
    assert metadata == old_metadata
    assert {p: file_hash(p) for p in originals} == originals
    assert artifact_paths(cid)["comparison"] in (root / "report.md").read_text()
    export = json.loads((root / "exports/images.jsonl").read_text())
    assert export["visualization_artifacts"] == artifact_paths(cid)
    assert export["verification_status"] == "unchanged"
    if legacy:
        assert result["verification_status"] == "legacy_unverified"
        for name in ("generated_grid", "drag_plan_grid"):
            assert (
                read_json(root / f"visualizations/{name}.json")["images"][0]["verification_status"]
                == "legacy_unverified"
            )


def test_migration_crash_after_record_update_can_resume(cfg, monkeypatch):
    import advv.visualization_collection as module

    root = create_run(cfg, "migration_crash", provenance="fake")
    Pipeline(root, FakeBackend(cfg)).run()
    record, hashes, _ = make_old_layout(root)
    original = module._rewrite_references

    def fail(*args, **kwargs):
        raise OSError("simulated interrupted link refresh")

    monkeypatch.setattr(module, "_rewrite_references", fail)
    with pytest.raises(OSError), run_lock(root):
        build_collection(root)
    assert (root / "visualizations" / record["candidate_id"]).is_dir()
    monkeypatch.setattr(module, "_rewrite_references", original)
    with run_lock(root):
        build_collection(root)
    assert not (root / "visualizations" / record["candidate_id"]).exists()
    assert artifact_paths(record["candidate_id"])["comparison"] in (root / "report.md").read_text()
    for kind, checksum in hashes.items():
        assert file_hash(root / artifact_paths(record["candidate_id"])[kind]) == checksum


def test_empty_run_and_active_batch_lock(cfg, tmp_path):
    root = create_run(cfg, "empty_collection", provenance="fake")
    project = tmp_path / "project"
    base = project / "assets/outputs"
    base.mkdir(parents=True)
    copied = base / root.name
    shutil.copytree(root, copied)
    with run_lock(copied):
        result = build_all_collections(project)
    assert result[0]["status"] == "skipped_active"
    assert not (copied / "visualizations/index.json").exists()
    assert build_all_collections(project)[0]["count"] == 0
    assert not list((copied / "visualizations/comparison").iterdir())
    assert read_json(copied / "visualizations/drag_plan_grid.json")["count"] == 0
    assert (copied / "visualizations/drag_plan_grid.png").is_file()


def test_changed_image_or_mismatched_metadata_refused(cfg):
    root = create_run(cfg, "invalid_collection", provenance="fake")
    Pipeline(root, FakeBackend(cfg)).run()
    record = read_json(next((root / "records").glob("*.json")))
    metadata_path = root / record["visualization"]["artifacts"]["metadata"]
    metadata = read_json(metadata_path)
    metadata["candidate_id"] = "another"
    atomic_json(metadata_path, metadata)
    with pytest.raises(DataError, match="different candidate"), run_lock(root):
        build_collection(root)
    metadata["candidate_id"] = record["candidate_id"]
    atomic_json(metadata_path, metadata)
    Image.new("RGB", (120, 80), "pink").save(root / record["image_path"])
    with pytest.raises(DataError, match="Accepted image changed"), run_lock(root):
        build_collection(root)


def test_cli_rebuild_needs_no_models_or_resume(cfg, monkeypatch):
    root = create_run(cfg, "cli_collection", provenance="fake")
    Pipeline(root, FakeBackend(cfg)).run()

    def forbidden(*args, **kwargs):
        raise AssertionError("No inference in visualization rebuild")

    monkeypatch.setattr("advv.cli.preflight", forbidden)
    monkeypatch.setattr("advv.cli.LocalBackend", forbidden)
    assert main(["visualize", "--run-dir", str(root)]) == 0
