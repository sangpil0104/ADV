import pytest

from advv.errors import DataError, FatalBackendError
from advv.pipeline import Pipeline, create_run
from advv.reporting import export_run, records_for
from advv.storage import file_hash, read_json
from conftest import REJECT, FakeBackend


def run(cfg, backend, run_id="test"):
    root = create_run(cfg, run_id, provenance="fake")
    state = Pipeline(root, backend).run()
    return root, state


def test_quota_rejection_deletion_and_clip_inputs(cfg):
    backend = FakeBackend([REJECT], [(20, 30, 40), (50, 60, 70)])
    root, state = run(cfg, backend)
    assert state["status"] == "completed"
    assert state["accepted_count"] == 1 and state["attempts"] == 2
    rejected, accepted = records_for(root)
    cid = rejected["candidate_id"]
    assert rejected["status"] == "rejected" and rejected["clip_filter"]["exceeded"] == [0]
    assert accepted["clip_filter"]["verdict"] == "accepted" and accepted["clip_filter"]["exceeded"] == []
    assert rejected["image_path"] is None
    assert not (root / f"candidates/{cid}/generated.png").exists()
    assert not (root / f"visualizations/{cid}/comparison.png").exists()
    assert (root / f"visualizations/{cid}/drag_plan.png").is_file()
    assert (root / accepted["image_path"]).is_file()
    assert all("visualizations" not in str(p) for call in backend.calls for p in call["images"])
    # Sentences once, the source once, then one candidate image per CLIP call.
    texts, source, *candidates = backend.embeds
    assert texts["images"] == [] and len(texts["texts"]) == len(cfg["_assets"]["implausible_sentences"])
    assert len(source["images"]) == 1 and source["texts"] == []
    assert [len(c["images"]) for c in candidates] == [1, 1] and all(c["texts"] == [] for c in candidates)
    with pytest.raises(DataError, match="fake"):
        export_run(root)


def test_source_and_accepted_duplicates_excluded(cfg):
    cfg["run"]["target_count"] = 2
    backend = FakeBackend(None, ["source", (5, 6, 7), (5, 6, 7), (8, 9, 10)])
    root, state = run(cfg, backend)
    assert state["status"] == "completed" and state["accepted_count"] == 2
    assert [r["export_status"] for r in records_for(root)] == [
        "duplicate",
        "eligible",
        "duplicate",
        "eligible",
    ]


def test_resume_reuses_references_and_does_not_regenerate(cfg):
    first = FakeBackend([KeyboardInterrupt()])
    root, state = run(cfg, first)
    assert state["status"] == "interrupted"
    second = FakeBackend(None, [])
    state = Pipeline(root, second, gpu_ids=["2", "3"]).run()
    assert state["status"] == "completed" and state["accepted_count"] == 1
    assert second.generations == []
    # Frozen sentence and source embeddings are reused; only the unfinished candidate is embedded.
    assert len(second.embeds) == 1 and "candidates" in str(second.embeds[0]["images"][0])
    assert state["segments"][-1]["selected_gpu_ids"] == ["2", "3"]


def test_resume_rejected_does_not_regenerate_deleted_image(cfg):
    first = FakeBackend([REJECT], [(1, 5, 9), KeyboardInterrupt()])
    root, state = run(cfg, first)
    assert state["status"] == "interrupted"
    second = FakeBackend(None, [(60, 70, 80)])
    state = Pipeline(root, second).run()
    assert state["status"] == "completed"
    assert len(second.generations) == 1 and second.generations[0][1] == 1
    assert records_for(root)[0]["image_path"] is None


def test_technical_retry_is_bounded_and_raw_preserved(cfg):
    backend = FakeBackend(["invalid", "invalid"], [(10, 20, 30), (30, 40, 50)])
    root, state = run(cfg, backend)
    assert state["status"] == "completed"
    record = records_for(root)[0]
    entry = record["checks"]["clip"]
    assert record["status"] == "verification_error" and entry["status"] == "error"
    assert len(entry["attempts"]) == 2 and all(a["error_type"] == "ResponseError" for a in entry["attempts"])
    assert all((root / a["receipt"]).is_file() for a in entry["attempts"])


def test_fatal_oom_stops_without_retry(cfg):
    backend = FakeBackend(colors=[FatalBackendError("CUDA out of memory")])
    root, state = run(cfg, backend)
    assert state["status"] == "failed" and state["attempts"] == 1
    assert len(backend.generations) == 1
    assert records_for(root)[0]["status"] == "generation_error"


def test_no_eligible_source(cfg):
    backend = FakeBackend(uncertain_source=True)
    _, state = run(cfg, backend)
    assert state["status"] == "failed" and state["attempts"] == 0
    assert "no_eligible_sources" in state["last_error"]["message"]
    assert backend.generations == []


def test_visualization_recovery_does_not_generate_extra(cfg, monkeypatch):
    import advv.pipeline as module

    original = module.render

    def fail(*args, **kwargs):
        raise OSError("simulated renderer failure")

    monkeypatch.setattr(module, "render", fail)
    root, state = run(cfg, FakeBackend())
    assert state["status"] == "failed" and state["accepted_count"] == 1
    monkeypatch.setattr(module, "render", original)
    second = FakeBackend([], [])
    state = Pipeline(root, second).run()
    assert state["status"] == "completed"
    assert second.calls == [] and second.generations == []


def test_total_quota_round_robin(cfg, tmp_path):
    from PIL import Image

    Image.new("RGB", (130, 100), (60, 80, 100)).save(tmp_path / "input/b.png")
    cfg["run"]["target_count"] = 2
    backend = FakeBackend(None, [(1, 2, 3), (4, 5, 6)])
    root, state = run(cfg, backend)
    assert state["accepted_count"] == 2
    assert len({r["source_id"] for r in records_for(root)}) == 2
    assert all(x[1] == 0 for x in backend.generations)


def test_receipt_survives_interruption_before_record_commit(cfg):
    from advv.storage import atomic_json

    class ReceiptThenInterrupt(FakeBackend):
        def embed(self, images, texts, *, receipt=None):
            result = super().embed(images, texts, receipt=receipt)
            if images and "candidates" in images[0].parts:
                # Model returned successfully; coordinator hasn't committed the check yet.
                atomic_json(images[0].parents[2] / receipt, {"ok": True, "result": result})
                raise KeyboardInterrupt()
            return result

    root, state = run(cfg, ReceiptThenInterrupt())
    assert state["status"] == "interrupted"
    second = FakeBackend([], [])
    assert Pipeline(root, second).run()["status"] == "completed"
    assert second.calls == []


def test_snapshot_and_preview_leave_original_unchanged(cfg, tmp_path):
    path = tmp_path / "input/a.png"
    before = file_hash(path)
    root, state = run(cfg, FakeBackend())
    assert state["status"] == "completed" and before == file_hash(path)
    record = records_for(root)[0]
    metadata = read_json(root / record["visualization"]["artifacts"]["metadata"])
    assert metadata["display_mode"] == "EFFECTIVE INPUT"
    assert metadata["original_size"] == [120, 80]
    assert 0 < metadata["area_fraction"] < 1
