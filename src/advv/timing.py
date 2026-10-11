"""Active wall-clock accounting across process lifetimes and unsuccessful candidates."""

from __future__ import annotations

import time
from pathlib import Path

from .errors import DataError
from .storage import atomic_json, now, read_json, safe_id, within

ATTEMPT_FIELDS = {
    "candidate_id",
    "source_id",
    "attempt_index",
    "sequence",
    "outcome",
    "error_code",
    "attempt_elapsed_seconds",
    "stage_seconds",
    "timing_complete",
    "cleanup_state",
    "accepted_cost_seconds",
    "accepted_active_watermark",
}


def save_attempt(root: Path, entry: dict) -> None:
    if set(entry) - ATTEMPT_FIELDS:
        raise DataError("Detailed candidate information cannot enter the timing ledger")
    cid = safe_id(entry["candidate_id"])
    atomic_json(within(root, f"timings/attempts/{cid}.json"), entry)


def attempts_for(root: Path) -> list[dict]:
    return sorted(
        [read_json(p) for p in (root / "timings/attempts").glob("*.json")], key=lambda r: r["sequence"]
    )


class ActiveClock:
    def __init__(self, root: Path, *, initial_seconds=0.0, clock=None):
        self.root, self.clock = root, clock or time.monotonic
        self.path = root / "timings/state.json"
        self.state = (
            read_json(self.path)
            if self.path.exists()
            else {
                "run_active_seconds": float(initial_seconds),
                "allocated_seconds": 0.0,
                "pending_cost_seconds": float(initial_seconds),
                "timing_complete": True,
                "active": False,
                "segments": [],
            }
        )
        self.last = None

    def start(self, *, overhead=0.0):
        if self.state["active"]:
            self.state["timing_complete"] = False
        self.state["active"] = True
        self.state["run_active_seconds"] += overhead
        self.state["segments"].append({"started_at": now(), "initial_seconds": overhead})
        self.last = self.clock()
        self.checkpoint()

    def checkpoint(self):
        if self.last is not None:
            current = self.clock()
            self.state["run_active_seconds"] += max(0.0, current - self.last)
            self.last = current
        self.state["pending_cost_seconds"] = max(
            0.0, self.state["run_active_seconds"] - self.state["allocated_seconds"]
        )
        atomic_json(self.path, self.state)
        return self.state["run_active_seconds"]

    def allocation(self):
        total = self.checkpoint()
        return total - self.state["allocated_seconds"], total

    def reconcile(self, accepted_records):
        watermark = max([r["timing"]["accepted_active_watermark"] for r in accepted_records] or [0.0])
        self.state["allocated_seconds"] = watermark
        self.state["run_active_seconds"] = max(self.state["run_active_seconds"], watermark)
        self.checkpoint()

    def finish(self):
        self.checkpoint()
        self.state["active"] = False
        self.state["segments"][-1]["ended_at"] = now()
        self.last = None
        atomic_json(self.path, self.state)
