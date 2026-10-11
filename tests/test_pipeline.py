import pytest
from PIL import Image

from advv.errors import BackendError, DataError, FatalBackendError
from advv.pipeline import Pipeline, create_run
from advv.reporting import export_run, records_for
from advv.storage import file_hash, read_json
from advv.timing import attempts_for
from conftest import FakeBackend, ManualClock


def test_quota_deletes_all_rejected_details(cfg):
    backend = FakeBackend(cfg, outcomes=[False, True])
    root = create_run(cfg, "quota", provenance="fake")
    state = Pipeline(root, backend).run()
    assert state["status"] == "completed" and state["accepted_count"] == 1
    assert state["attempts"] == 2
    entries = attempts_for(root)
    assert [e["outcome"] for e in entries] == ["rejected", "accepted"]
    failed_id = entries[0]["candidate_id"]
    for subdir in (".staging", "records", "artifacts", "visualizations", "candidates", "quarantine"):
        assert not any(failed_id in str(p) for p in (root / subdir).rglob("*"))
    assert "clip" not in entries[0] and "plan" not in entries[0]
    assert failed_id not in (root / "report.md").read_text()
    assert len(records_for(root)) == 1
    accepted = records_for(root)[0]
    assert (root / accepted["visualization"]["artifacts"]["comparison"]).is_file()
    with pytest.raises(DataError, match="fake"):
        export_run(root)
    assert backend.closed


def test_source_and_accepted_duplicates_do_not_consume_quota(cfg):
    cfg["run"]["target_count"] = 2
    backend = FakeBackend(cfg, colors=["source", (1, 2, 3), (1, 2, 3), (4, 5, 6)])
    root = create_run(cfg, "duplicates", provenance="fake")
    state = Pipeline(root, backend).run()
    assert state["status"] == "completed" and state["accepted_count"] == 2
    assert [r["outcome"] for r in attempts_for(root)] == ["duplicate", "accepted", "duplicate", "accepted"]
    assert len(records_for(root)) == 2


def test_resume_does_not_repeat_completed_generation(cfg):
    root = create_run(cfg, "resume", provenance="fake")
    first = FakeBackend(cfg, outcomes=[KeyboardInterrupt()])
    assert Pipeline(root, first).run()["status"] == "interrupted"
    second = FakeBackend(cfg)
    state = Pipeline(root, second, gpu_ids=["2", "3"]).run()
    assert state["status"] == "completed"
    assert second.generations == [] and second.segment_calls == []
    assert len(second.calls) == 1
    third = FakeBackend(cfg)
    assert Pipeline(root, third).run()["status"] == "completed"
    assert third.calls == [] and third.generations == []


def test_retry_errors_removed_and_fatal_stops(cfg):
    root = create_run(cfg, "retry", provenance="fake")
    backend = FakeBackend(cfg, outcomes=[BackendError("first"), True])
    assert Pipeline(root, backend).run()["status"] == "completed"
    assert len(backend.generations) == 1 and len(backend.calls) == 2
    root2 = create_run(cfg, "fatal", provenance="fake")
    fatal = FakeBackend(cfg, colors=[FatalBackendError("OOM")])
    state = Pipeline(root2, fatal).run()
    assert state["status"] == "failed" and state["accepted_count"] == 0
    assert len(fatal.generations) == 1
    assert attempts_for(root2)[0]["outcome"] == "error"
    assert not list((root2 / ".staging").glob("*"))


def test_no_vehicle_segments(cfg):
    root = create_run(cfg, "no_cars", provenance="fake")
    backend = FakeBackend(cfg, empty_segments=True)
    state = Pipeline(root, backend).run()
    assert state["status"] == "failed" and state["last_error"]["code"] == "no_eligible_sources"
    assert not backend.generations


def test_visualization_recovery_without_extra_inference(cfg, monkeypatch):
    import advv.pipeline as module

    root = create_run(cfg, "visualization", provenance="fake")
    real_render = module.render
    monkeypatch.setattr(module, "render", lambda *a, **kw: (_ for _ in ()).throw(OSError("disk")))
    assert Pipeline(root, FakeBackend(cfg)).run()["status"] == "failed"
    monkeypatch.setattr(module, "render", real_render)
    backend = FakeBackend(cfg)
    state = Pipeline(root, backend).run()
    assert state["status"] == "completed" and state["accepted_count"] == 1
    assert backend.generations == [] and backend.calls == []


def test_total_quota_round_robin_and_originals_unchanged(cfg, tmp_path):
    Image.new("RGB", (120, 80), (80, 20, 90)).save(tmp_path / "input/b.png")
    before = {p.name: file_hash(p) for p in (tmp_path / "input").glob("*")}
    cfg["run"]["target_count"] = 3
    backend = FakeBackend(cfg)
    root = create_run(cfg, "round_robin", provenance="fake")
    assert Pipeline(root, backend).run()["accepted_count"] == 3
    ids = [p["source_id"] for p in backend.generations]
    assert ids[0] == ids[2] and ids[0] != ids[1]
    assert before == {p.name: file_hash(p) for p in (tmp_path / "input").glob("*")}


def test_failure_inclusive_cost_is_conserved(cfg):
    timer = ManualClock()
    root = create_run(cfg, "timing", provenance="fake")
    backend = FakeBackend(cfg, outcomes=[False, False, True], timer=timer, durations=[40, 50, 60])
    assert Pipeline(root, backend, clock=timer).run()["status"] == "completed"
    record = records_for(root)[0]
    assert record["timing"]["attempt_elapsed_seconds"] == pytest.approx(60)
    assert record["timing"]["accepted_cost_seconds"] >= 150
    state = read_json(root / "timings/state.json")
    assert sum(r["timing"]["accepted_cost_seconds"] for r in records_for(root)) + state[
        "pending_cost_seconds"
    ] == pytest.approx(state["run_active_seconds"])
    assert read_json(root / "report.json")["mean_active_seconds_per_accepted"] >= 150


def test_missing_accepted_visualization_is_rebuilt_without_inference(cfg):
    root = create_run(cfg, "missing_preview", provenance="fake")
    assert Pipeline(root, FakeBackend(cfg)).run()["status"] == "completed"
    record = records_for(root)[0]
    comparison = root / record["visualization"]["artifacts"]["comparison"]
    comparison.unlink()
    backend = FakeBackend(cfg)
    assert Pipeline(root, backend).run()["status"] == "completed"
    assert comparison.is_file() and not backend.generations and not backend.calls
