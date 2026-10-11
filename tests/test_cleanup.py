import os

import pytest

from advv.cleanup import cleanup_candidate, cleanup_existing_runs
from advv.errors import DataError
from advv.pipeline import Pipeline, create_run
from advv.storage import atomic_bytes, atomic_json, file_hash, read_json, run_lock
from advv.timing import attempts_for, save_attempt
from conftest import FakeBackend


def minimal(cid="bad"):
    return {
        "candidate_id": cid,
        "source_id": "source",
        "attempt_index": 0,
        "sequence": 0,
        "outcome": "rejected",
        "attempt_elapsed_seconds": 12.0,
        "stage_seconds": {},
        "timing_complete": True,
        "cleanup_state": "pending",
    }


def test_requires_durable_minimal_timing_before_deletion(tmp_path):
    artifact = tmp_path / ".staging/bad/generated.png"
    atomic_bytes(artifact, b"image")
    with pytest.raises(DataError, match="timing"):
        cleanup_candidate(tmp_path, "bad")
    assert artifact.exists()
    save_attempt(tmp_path, minimal())
    cleanup_candidate(tmp_path, "bad")
    assert not artifact.exists()
    assert read_json(tmp_path / "cleanup/bad.json") == {
        "candidate_id": "bad",
        "state": "cleaned",
        "staging_only": False,
    }
    cleanup_candidate(tmp_path, "bad")  # idempotent
    assert attempts_for(tmp_path)[0]["attempt_elapsed_seconds"] == 12.0


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "accepted_reference"])
def test_cleanup_protects_external_and_accepted_files(tmp_path, kind):
    original = tmp_path / "inputs/original.png"
    atomic_bytes(original, b"protected")
    artifact = tmp_path / "candidates/bad/generated.png"
    artifact.parent.mkdir(parents=True)
    if kind == "symlink":
        artifact.symlink_to(original)
    elif kind == "hardlink":
        os.link(original, artifact)
    else:
        atomic_bytes(artifact, b"shared")
        atomic_json(
            tmp_path / "records/good.json",
            {"export_status": "eligible", "image_path": "candidates/bad/generated.png"},
        )
    save_attempt(tmp_path, minimal())
    with pytest.raises(DataError, match="cleanup_blocked"):
        cleanup_candidate(tmp_path, "bad")
    assert original.read_bytes() == b"protected"
    assert artifact.exists()


def make_legacy(root):
    root.mkdir()
    atomic_json(
        root / "manifest.json",
        {"schema_version": "1.1", "run_id": root.name, "backend": "dragflow+qwen_local"},
    )
    atomic_json(root / "config.json", {"run": {"target_count": 1}})
    atomic_json(root / "state.json", {"status": "interrupted", "last_error": None})
    atomic_json(root / "sources.json", [])
    atomic_json(
        root / "records/bad.json",
        {
            "candidate_id": "bad",
            "source_id": "source",
            "sequence": 0,
            "plan": {"attempt_index": 0},
            "status": "rejected",
            "export_status": "excluded",
            "generation_attempts": [{"elapsed_seconds": 40}],
            "checks": {
                "physical": {"attempts": [{"elapsed_seconds": 2, "raw_completion": "private details"}]}
            },
        },
    )
    for rel in (
        "candidates/bad/generated.png",
        "candidates/bad/backend_input/operation.png",
        "quarantine/bad/generated.png",
        "visualizations/bad/drag_plan.png",
        "visualizations/bad/comparison.png",
        "visualizations/bad/metadata.json",
        "visualizations/comparison/bad.png",
        "visualizations/drag_plan/bad.png",
        "visualizations/metadata/bad.json",
        "visualizations/comparison/.bad.png.atomic_tmp",
        "edit_regions/bad.png",
    ):
        atomic_bytes(root / rel, b"private candidate")
    atomic_bytes(root / "logs/generator.log", b"bad private prompt/score")
    atomic_bytes(root / "report.md", b"bad private candidate link")


def test_legacy_cleanup_removes_details_and_preserves_partial_times(tmp_path):
    output = tmp_path / "runs"
    output.mkdir()
    root = output / "legacy"
    make_legacy(root)
    original = root / "inputs/original.png"
    atomic_bytes(original, b"original")
    result = cleanup_existing_runs(output)
    assert result["cleaned_candidates"] == 1
    assert not (root / "records/bad.json").exists()
    assert not (root / "visualizations/bad").exists()
    for kind in ("comparison", "drag_plan", "metadata"):
        assert not list((root / "visualizations" / kind).iterdir())
    assert original.read_bytes() == b"original"
    entry = attempts_for(root)[0]
    assert entry["stage_seconds"]["generation"] == 40
    assert entry["attempt_elapsed_seconds"] is None and entry["timing_complete"] is False
    assert "private" not in (root / "logs/generator.log").read_text()
    assert "bad" not in (root / "report.md").read_text()
    assert cleanup_existing_runs(output)["cleaned_candidates"] == 0


def test_active_and_locked_runs_are_not_cleaned(tmp_path):
    output = tmp_path / "runs"
    output.mkdir()
    root = output / "legacy"
    make_legacy(root)
    atomic_json(root / "state.json", {"status": "running"})
    assert cleanup_existing_runs(output)["active_runs_skipped"] == 1
    atomic_json(root / "state.json", {"status": "interrupted"})
    with run_lock(root):
        assert cleanup_existing_runs(output)["active_runs_skipped"] == 1
    assert (root / "records/bad.json").exists()


def test_legacy_accepted_remains_but_is_not_clip_export(cfg):
    root = create_run(cfg, "accepted_old", provenance="fake")
    assert Pipeline(root, FakeBackend(cfg)).run()["status"] == "completed"
    record_path = next((root / "records").glob("*.json"))
    record = read_json(record_path)
    checksum = file_hash(root / record["image_path"])
    record.update(schema_version="1.1", backend="dragflow+qwen_local")
    atomic_json(record_path, record)
    manifest = read_json(root / "manifest.json")
    manifest.update(schema_version="1.1", backend="dragflow+qwen_local")
    atomic_json(root / "manifest.json", manifest)
    cleanup_existing_runs(root.parent)
    assert file_hash(root / record["image_path"]) == checksum
    assert (root / "exports/images.jsonl").read_text() == ""
    assert "legacy_unverified" in (root / "exports/legacy_images.jsonl").read_text()
    assert read_json(root / "report.json")["accepted"] == 0


def test_timing_ledger_rejects_detailed_failure_data(tmp_path):
    entry = minimal()
    entry["clip"] = {"scores": [1, 2]}
    with pytest.raises(DataError):
        save_attempt(tmp_path, entry)
