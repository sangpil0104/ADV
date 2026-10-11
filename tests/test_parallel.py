"""Exercise real concurrent scheduling with CPU-only, explicitly fake backends."""

import copy
import threading
from concurrent.futures import CancelledError

import pytest

from advv.config import gpu_groups, recipe_hash, validate_config
from advv.errors import ConfigError, FatalBackendError
from advv.pipeline import Pipeline, create_run
from advv.reporting import records_for
from advv.storage import read_json
from advv.timing import attempts_for
from conftest import FakeBackend


class Lane(FakeBackend):
    def __init__(self, cfg, *, barrier=None, first_color=None, first_outcome=True):
        super().__init__(cfg)
        self.barrier = barrier
        self.first_color = first_color
        self.first_outcome = first_outcome

    def generate(self, source, plan, root):
        if not self.generations and self.barrier:
            self.barrier.wait(timeout=10)
        self.colors = [
            self.first_color
            if not self.generations and self.first_color
            else (plan.attempt_index + 1, 22, 43)
        ]
        return super().generate(source, plan, root)

    def verify(self, image, *, receipt):
        self.outcomes = [self.first_outcome if not self.calls else True]
        return super().verify(image, receipt=receipt)


def parallel_config(cfg, count):
    cfg["execution"]["selected_gpu_ids"] = ["4", "5", "6", "7"]
    cfg["run"]["target_count"] = count
    return cfg


def test_two_lanes_share_exact_ten_and_commit_only_on_coordinator(cfg, monkeypatch):
    cfg = parallel_config(cfg, 10)
    barrier = threading.Barrier(2)
    lanes = [Lane(cfg, barrier=barrier), Lane(cfg, barrier=barrier)]
    original = Pipeline._commit

    def commit(self, record):
        assert threading.current_thread() is threading.main_thread()
        return original(self, record)

    monkeypatch.setattr(Pipeline, "_commit", commit)
    root = create_run(cfg, "parallel_ten", provenance="fake")
    state = Pipeline(root, lanes).run()
    assert state["status"] == "completed" and state["accepted_count"] == 10
    assert sum(len(b.generations) for b in lanes) == 10
    assert all(b.generations and b.closed for b in lanes)
    assert len(list((root / "accepted").glob("*.png"))) == 10
    assert not list((root / ".staging").glob("*"))
    records = records_for(root)
    assert {tuple(r["gpu_group"]) for r in records} == {("4", "5"), ("6", "7")}
    assert len({r["sequence"] for r in records}) == 10
    assert len({r["plan"]["attempt_index"] for r in records}) == 10
    assert state["segments"][-1]["gpu_groups"] == [["4", "5"], ["6", "7"]]
    timer = read_json(root / "timings/state.json")
    assert sum(r["timing"]["accepted_cost_seconds"] for r in records) + timer[
        "pending_cost_seconds"
    ] == pytest.approx(timer["run_active_seconds"])


@pytest.mark.parametrize("duplicate", [False, True])
def test_rejected_or_simultaneously_duplicate_candidates_do_not_fill_quota(cfg, duplicate):
    cfg = parallel_config(cfg, 2)
    barrier = threading.Barrier(2)
    color = (100, 120, 130) if duplicate else None
    lanes = [
        Lane(cfg, barrier=barrier, first_color=color, first_outcome=duplicate),
        Lane(cfg, barrier=barrier, first_color=color),
    ]
    root = create_run(cfg, "duplicate" if duplicate else "rejected", provenance="fake")
    state = Pipeline(root, lanes).run()
    assert state["status"] == "completed" and state["accepted_count"] == 2
    assert sum(len(b.generations) for b in lanes) == 3
    entries = attempts_for(root)
    assert sum(e["outcome"] == "accepted" for e in entries) == 2
    rejected = [e for e in entries if e["outcome"] != "accepted"]
    assert len(rejected) == 1
    assert rejected[0]["outcome"] == ("duplicate" if duplicate else "rejected")
    cid = rejected[0]["candidate_id"]
    assert not (root / f".staging/{cid}").exists()
    assert not (root / f"artifacts/{cid}").exists()
    assert not (root / f"records/{cid}.json").exists()


def test_one_remaining_slot_runs_only_one_lane(cfg):
    cfg = parallel_config(cfg, 1)
    lanes = [Lane(cfg), Lane(cfg)]
    root = create_run(cfg, "one", provenance="fake")
    assert Pipeline(root, lanes).run()["accepted_count"] == 1
    assert [len(b.generations) for b in lanes] == [1, 0]


def test_fast_lane_can_refill_before_slow_lane_finishes(cfg):
    cfg = parallel_config(cfg, 3)
    release = threading.Event()

    class Slow(Lane):
        def generate(self, source, plan, root):
            assert release.wait(10), "Scheduler blocked refill behind slow lane"
            return super().generate(source, plan, root)

    class Fast(Lane):
        def generate(self, source, plan, root):
            if self.generations:
                release.set()
            return super().generate(source, plan, root)

    lanes = [Slow(cfg), Fast(cfg)]
    root = create_run(cfg, "refill", provenance="fake")
    state = Pipeline(root, lanes).run()
    assert state["status"] == "completed" and release.is_set()
    assert [len(b.generations) for b in lanes] == [1, 2]


@pytest.mark.parametrize("fatal", [False, True])
def test_interrupt_or_fatal_failure_cancels_both_lanes_before_unlock(cfg, fatal):
    cfg = parallel_config(cfg, 2)
    barrier, stopped = threading.Barrier(2), threading.Event()

    class Fails(Lane):
        def generate(self, source, plan, root):
            barrier.wait(10)
            raise FatalBackendError("fatal") if fatal else KeyboardInterrupt()

    class Waits(Lane):
        def generate(self, source, plan, root):
            barrier.wait(10)
            assert stopped.wait(10), "Other lane was not stopped"
            raise CancelledError()

        def cancel(self):
            stopped.set()
            self.close()

    lanes = [Fails(cfg), Waits(cfg)]
    root = create_run(cfg, "cancel", provenance="fake")
    state = Pipeline(root, lanes).run()
    assert state["status"] == ("failed" if fatal else "interrupted")
    assert stopped.is_set() and all(b.closed for b in lanes)
    assert state["accepted_count"] == 0
    assert not read_json(root / "timings/state.json")["active"]
    if not fatal:
        assert len(attempts_for(root)) == 2
        assert all(e["outcome"] == "interrupted" for e in attempts_for(root))
        resumed = [Lane(cfg), Lane(cfg)]
        assert Pipeline(root, resumed).run()["accepted_count"] == 2


def test_resume_committed_and_scored_parallel_candidates_without_new_inference(cfg, monkeypatch):
    cfg = parallel_config(cfg, 2)
    barrier = threading.Barrier(2)

    class Scored(Lane):
        def verify(self, image, *, receipt):
            result = super().verify(image, receipt=receipt)
            barrier.wait(10)
            return result

    original = Pipeline._finish_candidate

    def interrupt(self, record, future):
        original(self, record, future)
        raise KeyboardInterrupt()

    monkeypatch.setattr(Pipeline, "_finish_candidate", interrupt)
    root = create_run(cfg, "resume", provenance="fake")
    state = Pipeline(root, [Scored(cfg), Scored(cfg)]).run()
    assert state["status"] == "interrupted" and state["accepted_count"] == 1
    monkeypatch.setattr(Pipeline, "_finish_candidate", original)
    # One GPU pair can recover work originally scheduled on two pairs.
    backend = Lane(cfg)
    state = Pipeline(root, backend, gpu_ids=["6", "7"]).run()
    assert state["status"] == "completed" and state["accepted_count"] == 2
    assert not backend.generations and not backend.calls


def test_scored_rejections_are_cleaned_even_if_another_lane_interrupts(cfg, monkeypatch):
    cfg = parallel_config(cfg, 2)
    barrier = threading.Barrier(2)

    class Rejects(Lane):
        def verify(self, image, *, receipt):
            result = super().verify(image, receipt=receipt)
            barrier.wait(10)
            return result

    def interrupt(self, record, future):
        raise KeyboardInterrupt()

    monkeypatch.setattr(Pipeline, "_finish_candidate", interrupt)
    root = create_run(cfg, "rejected_interrupt", provenance="fake")
    backends = [Rejects(cfg, first_outcome=False), Rejects(cfg, first_outcome=False)]
    assert Pipeline(root, backends).run()["status"] == "interrupted"
    assert not list((root / ".staging").glob("*"))
    assert len(attempts_for(root)) == 2
    assert all(e["outcome"] == "rejected" for e in attempts_for(root))


@pytest.mark.parametrize("ids", ["4", "4,5,6", "4,5,4,7", "4,x"])
def test_invalid_gpu_groups(cfg, ids):
    with pytest.raises(ConfigError):
        gpu_groups(ids)
    cfg["execution"]["selected_gpu_ids"] = ids
    with pytest.raises(ConfigError):
        validate_config(cfg)


def test_gpu_groups_and_count_may_change_on_resume_without_changing_recipe(cfg):
    original = recipe_hash(cfg)
    cfg["execution"]["selected_gpu_ids"] = ["4", "5", "6", "7"]
    assert gpu_groups(cfg["execution"]["selected_gpu_ids"]) == [["4", "5"], ["6", "7"]]
    assert recipe_hash(cfg) == original


def test_worker_gpu_isolation_between_pairs(cfg, tmp_path, monkeypatch):
    import io
    from advv.backends.process import Worker

    observed = []

    class Process:
        def __init__(self, args, **kwargs):
            observed.append(kwargs["env"]["CUDA_VISIBLE_DEVICES"])
            self.stdin, self.stdout = io.StringIO(), io.StringIO()

        def poll(self):
            return 0

    monkeypatch.setattr("advv.backends.process.subprocess.Popen", Process)
    for ids in (["4", "5"], ["6", "7"]):
        lane_cfg = copy.deepcopy(cfg)
        lane_cfg["execution"]["selected_gpu_ids"] = ids
        for kind in ("generator", "verifier", "segmenter"):
            Worker(kind, lane_cfg, tmp_path).close()
    assert observed == ["4,5", "4", "4", "6,7", "6", "6"]


def test_cancelled_backend_cannot_spawn_another_stage(cfg, tmp_path, monkeypatch):
    from advv.backends.process import LocalBackend

    def forbidden(*args, **kwargs):
        pytest.fail("Cancelled backend spawned a new GPU worker")

    monkeypatch.setattr("advv.backends.process.Worker", forbidden)
    backend = LocalBackend(cfg, tmp_path)
    backend.cancel()
    with pytest.raises(CancelledError):
        backend.verify(tmp_path / "never.png", receipt="never.json")
