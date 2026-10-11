import copy

import pytest
from PIL import Image

from advv.errors import BackendError, DataError
from advv.ingest import scan_sources
from advv.pipeline import Pipeline, create_run
from advv.storage import atomic_json, read_json, run_lock
from advv.timing import attempts_for
from conftest import FakeBackend


def test_normal_rejections_have_no_global_attempt_cap(cfg):
    backend = FakeBackend(
        cfg, outcomes=[False] * 15, colors=[(i, 2, 3) for i in range(15)] + [KeyboardInterrupt()]
    )
    root = create_run(cfg, "many_no", provenance="fake")
    state = Pipeline(root, backend).run()
    assert state["status"] == "interrupted" and state["attempts"] == 16
    assert state["accepted_count"] == 0
    assert attempts_for(root)[-1]["outcome"] == "interrupted"
    report = read_json(root / "report.json")
    assert report["mean_active_seconds_per_accepted"] is None
    assert report["pending_cost_seconds"] > 0


def test_consecutive_technical_failure_limit(cfg):
    backend = FakeBackend(cfg, outcomes=[BackendError("bad") for _ in range(6)])
    root = create_run(cfg, "errors", provenance="fake")
    state = Pipeline(root, backend).run()
    assert state["status"] == "failed" and state["attempts"] == 3
    assert state["last_error"]["code"] == "consecutive_technical_failures"
    assert all(e["outcome"] == "error" for e in attempts_for(root))
    assert not list((root / ".staging").glob("*"))


def test_changed_recipe_blocks_resume(cfg):
    root = create_run(cfg, "changed", provenance="fake")
    changed = copy.deepcopy(cfg)
    changed["run"]["seed"] += 1
    atomic_json(root / "config.json", changed)
    with pytest.raises(DataError, match="Recipe changed"):
        Pipeline(root, FakeBackend(cfg))


def test_resume_exhausted_stage_does_not_repeat_technical_retries(cfg):
    root = create_run(cfg, "exhausted", provenance="fake")
    assert Pipeline(root, FakeBackend(cfg, colors=[KeyboardInterrupt()])).run()["status"] == "interrupted"
    path = next((root / ".staging").glob("*/record.json"))
    pending = read_json(path)
    pending["retries"]["generation"] = cfg["execution"]["max_technical_retries"] + 1
    atomic_json(path, pending)
    backend = FakeBackend(cfg)
    assert Pipeline(root, backend).run()["status"] == "completed"
    assert len(backend.generations) == 1 and backend.generations[0]["attempt_index"] == 1
    assert [e["outcome"] for e in attempts_for(root)] == ["error", "accepted"]


def test_two_writers_blocked(cfg):
    root = create_run(cfg, "locking", provenance="fake")
    with run_lock(root):
        with pytest.raises(DataError, match="Another writer"):
            with run_lock(root):
                pytest.fail("Second writer acquired")


def test_exif_normalization(cfg, tmp_path):
    image = Image.new("RGB", (80, 120), (20, 80, 130))
    exif = Image.Exif()
    exif[274] = 6
    image.save(tmp_path / "input/rotated.jpg", exif=exif)
    source = [s for s in scan_sources(cfg)[0] if s.original_path.endswith("rotated.jpg")][0]
    assert (source.width, source.height) == (120, 80)


def test_effective_geometry_mapping():
    from advv.backends.dragflow import to_original, upstream_instruction

    assert to_original([8, 4], [16, 8], [128, 64]) == [64, 32]
    instruction = upstream_instruction(
        {
            "operation": "relocation",
            "source_point": [10, 12],
            "target_point": [20, 25],
            "anchor_point": None,
            "source_prompt": "source",
            "target_prompt": "edit",
        }
    )
    assert instruction["region_operations"]["0"]["task"] == "transformation"


@pytest.mark.parametrize("after_commit", [False, True])
def test_interrupted_acceptance_commit_never_repeats_inference(cfg, monkeypatch, after_commit):
    import advv.pipeline as module

    real_write = module.atomic_json
    interrupted = False

    def interrupt(path, value):
        nonlocal interrupted
        if path.parent.name == "records" and not interrupted:
            interrupted = True
            if after_commit:
                real_write(path, value)
            raise KeyboardInterrupt()
        real_write(path, value)

    root = create_run(cfg, "crash", provenance="fake")
    monkeypatch.setattr(module, "atomic_json", interrupt)
    assert Pipeline(root, FakeBackend(cfg)).run()["status"] == "interrupted"
    monkeypatch.setattr(module, "atomic_json", real_write)
    backend = FakeBackend(cfg)
    result = Pipeline(root, backend).run()
    assert result["status"] == "completed" and result["accepted_count"] == 1
    assert backend.generations == [] and backend.calls == []
    assert len([e for e in attempts_for(root) if e["outcome"] == "accepted"]) == 1


def test_cleanup_crash_recovery_does_not_repeat_rejected_inference(cfg, monkeypatch):
    import advv.cleanup as cleanup

    original = cleanup._remove_checked
    failed = False

    def fail_once(path):
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("simulated disk failure")
        return original(path)

    root = create_run(cfg, "cleanup_crash", provenance="fake")
    monkeypatch.setattr(cleanup, "_remove_checked", fail_once)
    first = FakeBackend(cfg, outcomes=[False])
    assert Pipeline(root, first).run()["status"] == "failed"
    assert attempts_for(root)[0]["cleanup_state"] == "pending"
    monkeypatch.setattr(cleanup, "_remove_checked", original)
    second = FakeBackend(cfg, colors=[(100, 50, 20)])
    state = Pipeline(root, second).run()
    assert state["status"] == "completed"
    assert len(second.generations) == 1 and second.generations[0]["attempt_index"] == 1
    entries = attempts_for(root)
    assert [e["outcome"] for e in entries] == ["rejected", "accepted"]
    assert entries[0]["cleanup_state"] == "cleaned"
    assert entries[0]["timing_complete"] is False


def test_single_conda_guard_and_worker_interpreter(cfg, tmp_path, monkeypatch):
    import io
    import sys
    from advv.environment import require_advv
    from advv.errors import ConfigError
    from advv.backends.process import Worker

    require_advv()
    observed = []

    class Process:
        def __init__(self, args, **kwargs):
            observed.append((args, kwargs))
            self.stdin = io.StringIO()
            self.stdout = io.StringIO()

        def poll(self):
            return 0

    monkeypatch.setattr("advv.backends.process.subprocess.Popen", Process)
    Worker("verifier", cfg, tmp_path).close()
    assert observed[0][0][0] == sys.executable
    assert observed[0][1]["env"]["CUDA_VISIBLE_DEVICES"] == "0"
    monkeypatch.setenv("CONDA_PREFIX", str(tmp_path / "other"))
    with pytest.raises(ConfigError):
        require_advv()
    monkeypatch.setenv("CONDA_PREFIX", sys.prefix)
    monkeypatch.setattr(sys, "base_prefix", str(tmp_path / "base"))
    with pytest.raises(ConfigError):
        require_advv()
