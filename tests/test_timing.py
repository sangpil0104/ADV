import pytest

from advv.storage import read_json
from advv.timing import ActiveClock
from conftest import ManualClock


def test_pause_is_excluded_but_pending_and_restart_cost_are_preserved(tmp_path):
    manual = ManualClock()
    clock = ActiveClock(tmp_path, clock=manual, initial_seconds=10)
    clock.start()
    manual.advance(40)
    clock.finish()
    assert clock.state["pending_cost_seconds"] == 50
    manual.advance(3600)  # Time while the process was stopped must not be billed.
    resumed = ActiveClock(tmp_path, clock=manual)
    resumed.start(overhead=5)
    manual.advance(50)
    cost, watermark = resumed.allocation()
    assert cost == 105
    resumed.reconcile([{"timing": {"accepted_active_watermark": watermark}}])
    manual.advance(7)
    resumed.finish()
    assert resumed.state["pending_cost_seconds"] == 7
    assert resumed.state["run_active_seconds"] == 112


def test_unfinished_clock_marks_time_incomplete(tmp_path):
    manual = ManualClock()
    first = ActiveClock(tmp_path, clock=manual)
    first.start()
    manual.advance(5)
    first.checkpoint()
    manual.advance(30)  # Process dies without a durable end timestamp.
    resumed = ActiveClock(tmp_path, clock=manual)
    resumed.start()
    assert resumed.state["run_active_seconds"] == pytest.approx(5)
    assert resumed.state["timing_complete"] is False
    resumed.finish()
    assert read_json(tmp_path / "timings/state.json")["active"] is False
