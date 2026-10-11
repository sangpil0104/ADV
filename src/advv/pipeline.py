from __future__ import annotations

import copy
import logging
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from tqdm import tqdm

from .cleanup import cleanup_candidate, recover_cleanup
from .config import gpu_groups, recipe_hash, validate_config
from .contracts import EditRequest, Source
from .errors import BackendError, DataError, FatalBackendError, ResponseError
from .ingest import scan_sources, snapshot_sources
from .sampler import candidate_id, sample_edit, source_motion_feasible, validate_plan
from .segmentation import validate_segments
from .storage import (
    atomic_bytes,
    atomic_image,
    atomic_json,
    digest,
    file_hash,
    load_rgb,
    now,
    pixel_hash,
    read_json,
    run_lock,
    safe_id,
    within,
)
from .timing import ActiveClock, attempts_for, save_attempt
from .verifiers.decision import decide
from .visualization import render

LOG = logging.getLogger(__name__)


def implementation_hash() -> str:
    base = Path(__file__).parent
    return digest({str(p.relative_to(base)): file_hash(p) for p in sorted(base.rglob("*.py"))})


def create_run(cfg: dict, run_id: str, *, provenance: str, initial_seconds=0.0) -> Path:
    started = time.monotonic()
    validate_config(cfg)
    safe_id(run_id)
    sources, stats = scan_sources(cfg)
    root = Path(cfg["run"]["output_root"]).resolve()
    data_root = Path(cfg["dataset"]["root"]).resolve()
    if root.is_relative_to(data_root):
        raise DataError("output_root must be outside dataset root")
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / run_id
    run_dir.mkdir(exist_ok=False)
    atomic_json(run_dir / "config.json", copy.deepcopy(cfg))
    sources = snapshot_sources(sources, run_dir)
    atomic_json(run_dir / "sources.json", [s.to_dict() for s in sources])
    atomic_json(
        run_dir / "manifest.json",
        {
            "schema_version": "2.0",
            "project": "ADVV",
            "run_id": run_id,
            "created_at": now(),
            "backend": provenance,
            "recipe_sha256": recipe_hash(cfg),
            "implementation_sha256": implementation_hash(),
            "ingest": stats,
            "input_manifest_sha256": digest([s.to_dict() for s in sources]),
        },
    )
    atomic_json(
        run_dir / "state.json",
        {
            "status": "ready",
            "accepted_count": 0,
            "attempts": 0,
            "cursor": {},
            "next_source": 0,
            "segments": [],
            "consecutive_technical_failures": 0,
        },
    )
    clock = ActiveClock(run_dir, initial_seconds=initial_seconds + time.monotonic() - started)
    clock.checkpoint()
    return run_dir


def _relocate(value, old, new):
    if isinstance(value, str):
        return new + value[len(old) :] if value.startswith(old + "/") else value
    if isinstance(value, dict):
        return {k: _relocate(v, old, new) for k, v in value.items()}
    if isinstance(value, list):
        return [_relocate(v, old, new) for v in value]
    return value


class Pipeline:
    def __init__(self, run_dir: Path, backend, *, gpu_ids=None, initial_seconds=0.0, clock=None):
        self.root = run_dir.resolve()
        self.cfg = read_json(self.root / "config.json")
        validate_config(self.cfg)
        self.manifest = read_json(self.root / "manifest.json")
        if (
            self.manifest["schema_version"] != "2.0"
            or recipe_hash(self.cfg) != self.manifest["recipe_sha256"]
        ):
            raise DataError("Recipe changed or legacy run; start a new CLIP run")
        if implementation_hash() != self.manifest["implementation_sha256"]:
            raise DataError("Implementation changed; use original code or start a new run")
        self.sources = [Source(**s) for s in read_json(self.root / "sources.json")]
        if digest([s.to_dict() for s in self.sources]) != self.manifest["input_manifest_sha256"]:
            raise DataError("Source manifest changed")
        for source in self.sources:
            if pixel_hash(within(self.root, source.image_path, must_exist=True)) != source.pixel_sha256:
                raise DataError("Source snapshot changed")
        self.backends = list(backend) if isinstance(backend, (list, tuple)) else [backend]
        if not self.backends or any(b.provenance != self.manifest["backend"] for b in self.backends):
            raise DataError("Backend provenance changed")
        self.by_id = {s.source_id: s for s in self.sources}
        self.backend = self.backends[0]
        self.gpu_ids = gpu_ids or self.cfg["execution"]["selected_gpu_ids"]
        self.gpu_groups = gpu_groups(self.gpu_ids)
        if len(self.backends) != len(self.gpu_groups) or len({id(b) for b in self.backends}) != len(
            self.backends
        ):
            raise DataError("Each GPU pair requires its own independent backend")
        self.initial_seconds, self.clock_function = initial_seconds, clock
        self.accepted, self.segment_cache = [], {}
        self.seen = {s.pixel_sha256 for s in self.sources}

    def save_state(self):
        attempt_ids = {r["candidate_id"] for r in attempts_for(self.root) + self.pending_records()}
        self.state.update(accepted_count=len(self.accepted), updated_at=now(), attempts=len(attempt_ids))
        atomic_json(self.root / "state.json", self.state)

    def pending_records(self):
        return sorted(
            [read_json(p) for p in (self.root / ".staging").glob("*/record.json")],
            key=lambda r: r["sequence"],
        )

    def save_private(self, record):
        atomic_json(within(self.root, f".staging/{record['candidate_id']}/record.json"), record)

    def _retry(self, record, stage, operation):
        errors = record.setdefault("retries", {}).get(stage, 0)
        if errors > self.cfg["execution"]["max_technical_retries"]:
            raise BackendError("stage_retry_exhausted")
        while errors <= self.cfg["execution"]["max_technical_retries"]:
            self.save_private(record)
            # Only the coordinator mutates the shared clock; lane intervals may overlap.
            clock = self.clock_function or time.monotonic
            start = clock()
            try:
                result = operation()
                return result
            except (BackendError, ResponseError) as exc:
                if isinstance(exc, FatalBackendError):
                    raise
                errors += 1
                record["retries"][stage] = errors
                if errors > self.cfg["execution"]["max_technical_retries"]:
                    raise
            finally:
                end = clock()
                times = record.setdefault("stage_seconds", {})
                times[stage] = times.get(stage, 0.0) + end - start
                self.save_private(record)

    def _segments(self):
        for source in self.sources:
            if source.split not in self.cfg["dataset"]["eligible_splits"]:
                continue
            path = within(self.root, f"segments/{source.source_id}/result.json")
            if path.exists():
                result = read_json(path)
            else:
                for attempt in range(1 + self.cfg["execution"]["max_technical_retries"]):
                    try:
                        result = self.backend.segment(source, self.root)
                        break
                    except FatalBackendError:
                        raise
                    except BackendError:
                        if attempt == self.cfg["execution"]["max_technical_retries"]:
                            result = {
                                "segments": [],
                                "source_sha256": source.pixel_sha256,
                                "settings_sha256": digest(self.cfg["segmentation"]),
                                "info": {"error_code": "segmenter_error"},
                            }
                atomic_json(path, result)
            if result["source_sha256"] != source.pixel_sha256 or result["settings_sha256"] != digest(
                self.cfg["segmentation"]
            ):
                raise DataError("Segment cache identity changed")
            validate_segments(source, result, self.root)
            if result.get("selection"):
                from .accident_group import selection_identity

                if result["selection"]["identity"] != selection_identity(self.cfg):
                    raise DataError("Accident selection cache identity changed")
            self.segment_cache[source.source_id] = result
            self.clock.checkpoint()
        self.eligible = [
            s for s in self.sources if self.segment_cache.get(s.source_id, {}).get("accident_group")
        ]
        if not self.eligible:
            raise DataError("no_eligible_sources")
        # Deterministic pre-check; a held source stays held for the whole run.
        held = dict(self.state.get("held_sources", {}))
        for source in self.eligible:
            sid = source.source_id
            if sid not in held and not source_motion_feasible(
                source, self.segment_cache[sid], self.cfg, self.root
            ):
                held[sid] = "no_valid_motion"
                LOG.warning("Holding source %s: no capped in-frame group translation exists", sid)
        self.state["held_sources"] = held
        self.save_state()
        self.eligible = [s for s in self.eligible if s.source_id not in held]
        if not self.eligible:
            raise DataError("no_movable_sources")

    def _recover(self):
        recover_cleanup(self.root)
        self.seen = {s.pixel_sha256 for s in self.sources}
        committed = sorted(
            [read_json(p) for p in (self.root / "records").glob("*.json")], key=lambda r: r["sequence"]
        )
        self.accepted = []
        for record in committed:
            if record.get("status") != "accepted" or record.get("export_status") != "eligible":
                raise DataError("Nonaccepted record in public records")
            if decide(record["clip"]) != "accepted" or record["backend"] != self.backend.provenance:
                raise DataError("Invalid accepted decision/provenance")
            if record["clip"]["criteria_sha256"] != self.cfg["_assets"]["criteria_sha256"]:
                raise DataError("CLIP criteria changed")
            path = within(self.root, record["image_path"], must_exist=True)
            if (
                pixel_hash(path) != record["pixel_sha256"]
                or file_hash(path) != record["file_sha256"]
                or record["pixel_sha256"] in self.seen
            ):
                raise DataError("Accepted image changed/duplicated")
            if digest(record["plan"]) != record["plan_sha256"]:
                raise DataError("Accepted plan changed")
            if digest(record["effective"]) != record["effective_sha256"]:
                raise DataError("Accepted effective geometry changed")
            validate_plan(
                EditRequest(**record["plan"]), self.by_id[record["source_id"]], self.root, self.cfg["sampler"]
            )
            self.seen.add(record["pixel_sha256"])
            # Repairs only artifacts, never invokes the generator or CLIP.
            self._finish_presentation(record)
            save_attempt(self.root, record["timing"])
            self.accepted.append(record)
            cleanup_candidate(self.root, record["candidate_id"], staging_only=True)
        self.clock.reconcile(self.accepted)
        all_entries = attempts_for(self.root) + self.pending_records()
        for entry in all_entries:
            index = entry.get("attempt_index", entry.get("plan", {}).get("attempt_index"))
            if index is not None:
                sid = entry["source_id"]
                self.state["cursor"][sid] = max(self.state["cursor"].get(sid, 0), index + 1)
        for entry in attempts_for(self.root):
            if entry["outcome"] in ("rejected", "duplicate", "error"):
                # Idempotent even when a cleanup journal was written just before a crash.
                if (
                    entry.get("cleanup_state") != "cleaned"
                    or (self.root / f".staging/{entry['candidate_id']}").exists()
                ):
                    cleanup_candidate(self.root, entry["candidate_id"])
        if len(self.accepted) > self.cfg["run"]["target_count"]:
            raise DataError("Accepted count exceeds quota")

    def _timing(self, record, outcome, *, cleanup_state="pending"):
        current = self.clock.checkpoint()
        return {
            "candidate_id": record["candidate_id"],
            "source_id": record["source_id"],
            "attempt_index": record["attempt_index"],
            "sequence": record["sequence"],
            "outcome": outcome,
            "attempt_elapsed_seconds": current - record["started_active_seconds"],
            "stage_seconds": dict(record.get("stage_seconds", {})),
            "timing_complete": self.clock.state["timing_complete"],
            "cleanup_state": cleanup_state,
        }

    def _discard(self, record, outcome, error_code=None):
        entry = self._timing(record, outcome)
        entry["timing_complete"] = False  # Cleanup has not finished; preserve this on a crash.
        if error_code:
            entry["error_code"] = error_code
        before_save = self.clock.checkpoint()
        save_attempt(self.root, entry)
        start = self.clock.checkpoint()
        entry["stage_seconds"]["persistence"] = (
            entry["stage_seconds"].get("persistence", 0.0) + start - before_save
        )
        cleanup_candidate(self.root, record["candidate_id"])
        end = self.clock.checkpoint()
        entry["cleanup_state"] = "cleaned"
        entry["stage_seconds"]["cleanup"] = end - start
        entry["attempt_elapsed_seconds"] = end - record["started_active_seconds"]
        entry["timing_complete"] = self.clock.state["timing_complete"]
        save_attempt(self.root, entry)

    def _finish_presentation(self, record):
        artifacts = record.get("visualization", {}).get("artifacts", {})
        complete = record.get("presentation_complete") and all(
            key in artifacts and within(self.root, artifacts[key]).is_file()
            for key in ("drag_plan", "comparison", "metadata")
        )
        if not complete:
            record["visualization"] = render(
                self.root, self.by_id[record["source_id"]].to_dict(), record, self.cfg["visualization"]
            )
            record["presentation_complete"] = True
            atomic_json(self.root / f"records/{record['candidate_id']}.json", record)

    def _commit(self, record):
        persistence_start = self.clock.checkpoint()
        cid = record["candidate_id"]
        old, new = f".staging/{cid}", f"artifacts/{cid}"
        source_dir = within(self.root, old)
        # Only verified, unique candidates are copied to public artifacts.
        for file in sorted(source_dir.rglob("*")):
            if file.is_symlink():
                raise DataError("Unexpected symlink in private candidate")
            if file.is_file() and file != source_dir / "record.json":
                if file.stat().st_nlink > 1:
                    raise DataError("Unexpected hardlink in private candidate")
                relative = (Path(new) / file.relative_to(source_dir)).as_posix()
                atomic_bytes(within(self.root, relative), file.read_bytes())
        accepted = _relocate(copy.deepcopy(record), old, new)
        image_path = f"accepted/{cid}.png"
        atomic_image(within(self.root, image_path), load_rgb(within(self.root, record["image_path"])))
        accepted.update(
            image_path=image_path,
            file_sha256=file_hash(within(self.root, image_path)),
            status="accepted",
            export_status="eligible",
            presentation_complete=False,
            requested_plan_sha256=record["plan_sha256"],
        )
        accepted["plan_sha256"] = digest(accepted["plan"])
        accepted["effective_sha256"] = digest(accepted["effective"])
        accepted["timing"] = self._timing(record, "accepted", cleanup_state="cleaned")
        cost, watermark = self.clock.allocation()
        accepted["timing"].update(accepted_cost_seconds=cost, accepted_active_watermark=watermark)
        # Prepare required graphics, then freeze the accounting boundary immediately before commit.
        accepted["visualization"] = render(
            self.root, self.by_id[record["source_id"]].to_dict(), accepted, self.cfg["visualization"]
        )
        record["stage_seconds"]["persistence"] += self.clock.checkpoint() - persistence_start
        accepted["stage_seconds"] = dict(record["stage_seconds"])
        accepted["timing"] = self._timing(record, "accepted", cleanup_state="cleaned")
        cost, watermark = self.clock.allocation()
        accepted["timing"].update(accepted_cost_seconds=cost, accepted_active_watermark=watermark)
        atomic_json(self.root / f"records/{cid}.json", accepted)  # acceptance commit authority
        self.clock.reconcile(self.accepted + [accepted])
        self._finish_presentation(accepted)  # final footer uses the committed cost; overhead is pending
        save_attempt(self.root, accepted["timing"])
        self.accepted.append(accepted)
        self.seen.add(accepted["pixel_sha256"])
        cleanup_candidate(self.root, cid, staging_only=True)
        from .visualization_collection import build_collection

        build_collection(self.root)

    def _infer(self, record, backend):
        """A lane owns only its candidate's private files, never the shared quota."""
        source = self.by_id[record["source_id"]]
        if record["status"] == "reserved":
            plan = self._retry(
                record,
                "planning",
                lambda: sample_edit(
                    source,
                    self.segment_cache[source.source_id],
                    self.cfg,
                    record["attempt_index"],
                    self.root,
                ),
            )
            record.update(plan=plan.to_dict(), plan_sha256=digest(plan.to_dict()), status="planned")
            self.save_private(record)
        plan = EditRequest(**record["plan"])
        if digest(record["plan"]) != record["plan_sha256"]:
            raise DataError("Private plan changed")
        validate_plan(plan, source, self.root, self.cfg["sampler"])
        if record["status"] == "planned":
            result = self._retry(record, "generation", lambda: backend.generate(source, plan, self.root))
            expected = within(self.root, f".staging/{plan.edit_id}/backend/generated.png")
            if result.path.resolve() != expected.resolve():
                raise FatalBackendError("unexpected_generator_output")
            record.update(
                image_path=str(expected.relative_to(self.root)),
                pixel_sha256=pixel_hash(expected),
                file_sha256=file_hash(expected),
                effective=result.effective,
                effective_sha256=digest(result.effective),
                generation_info=result.info,
                status="generated",
            )
            self.save_private(record)
        candidate = within(self.root, record["image_path"], must_exist=True)
        if (
            pixel_hash(candidate) != record["pixel_sha256"]
            or digest(record["effective"]) != record["effective_sha256"]
        ):
            raise DataError("Candidate/effective geometry changed")
        if record["status"] == "generated":

            def verify():
                result = backend.verify(candidate, receipt=f".staging/{record['candidate_id']}/clip.json")
                decide(result)
                if result["criteria_sha256"] != self.cfg["_assets"]["criteria_sha256"]:
                    raise ResponseError("CLIP criterion identity mismatch")
                return result

            record["clip"] = self._retry(record, "verification", verify)
            record["status"] = "scored"
            self.save_private(record)

    def _finish_candidate(self, record, future):
        # Coordinator-only: score decision, global dedup, durable commit and quota.
        try:
            future.result()
        except (BackendError, ResponseError) as exc:
            self._discard(record, "error", type(exc).__name__)
            if isinstance(exc, FatalBackendError):
                raise
            return "error"
        except DataError:
            if record["status"] != "reserved":
                raise
            self._discard(record, "error", "sampling_error")
            return "error"
        if decide(record["clip"]) == "rejected":
            self._discard(record, "rejected")
            return "rejected"
        if record["pixel_sha256"] in self.seen:
            self._discard(record, "duplicate")
            return "duplicate"
        if len(self.accepted) >= self.cfg["run"]["target_count"]:
            raise DataError("Shared accepted quota is already full")
        self._commit(record)
        return "accepted"

    def _reserve(self):
        index = self.state["next_source"] % len(self.eligible)
        source = self.eligible[index]
        attempt = self.state["cursor"].get(source.source_id, 0)
        entries = attempts_for(self.root) + self.pending_records()
        record = {
            "schema_version": "2.0",
            "candidate_id": candidate_id(source, self.cfg, attempt),
            "source_id": source.source_id,
            "attempt_index": attempt,
            "sequence": max([r["sequence"] for r in entries] or [-1]) + 1,
            "backend": self.backend.provenance,
            "status": "reserved",
            "created_at": now(),
            "started_active_seconds": self.clock.checkpoint(),
            "stage_seconds": dict.fromkeys(
                ("planning", "generation", "verification", "persistence", "cleanup"), 0.0
            ),
        }
        self.save_private(record)
        self.state["cursor"][source.source_id] = attempt + 1
        self.state["next_source"] = (index + 1) % len(self.eligible)
        self.save_state()
        return record

    def _run_candidates(self, target):
        pending = self.pending_records()
        if len(self.accepted) + len(pending) > target:
            raise DataError("Pending reservations exceed shared quota")
        entries = attempts_for(self.root) + pending
        if len(self.accepted) < target and entries:
            ids = [s.source_id for s in self.eligible]
            last = max(entries, key=lambda r: r["sequence"])["source_id"]
            self.state["next_source"] = (ids.index(last) + 1) % len(ids) if last in ids else 0
        pool = ThreadPoolExecutor(max_workers=len(self.backends), thread_name_prefix="advv-candidate")
        running, idle = {}, list(range(len(self.backends)))
        try:
            with tqdm(
                total=target,
                initial=len(self.accepted),
                desc="accepted",
                disable=not self.cfg["report"]["terminal_progress"],
            ) as bar:
                while len(self.accepted) < target:
                    # Every in-flight candidate reserves one remaining output slot.
                    # Near the target this prevents overshoot AND unnecessary GPU work.
                    while idle and len(self.accepted) + len(running) < target:
                        record = pending.pop(0) if pending else self._reserve()
                        lane = idle.pop(0)
                        record["gpu_group"] = self.gpu_groups[lane]
                        self.save_private(record)
                        future = pool.submit(self._infer, record, self.backends[lane])
                        running[future] = (lane, record)
                    done, _ = wait(running, timeout=0.2, return_when=FIRST_COMPLETED)
                    self.clock.checkpoint()
                    # A fast lane can continue while another lane is still generating.
                    for future in sorted(done, key=lambda f: running[f][1]["sequence"]):
                        lane, record = running.pop(future)
                        outcome = self._finish_candidate(record, future)
                        idle.append(lane)
                        self.state["consecutive_technical_failures"] = (
                            self.state["consecutive_technical_failures"] + 1 if outcome == "error" else 0
                        )
                        self.save_state()
                        if (
                            self.state["consecutive_technical_failures"]
                            >= self.cfg["execution"]["max_consecutive_technical_failures"]
                        ):
                            raise FatalBackendError("consecutive_technical_failures")
                        bar.update(len(self.accepted) - bar.n)
                        bar.set_postfix(
                            attempts=len(attempts_for(self.root)),
                            in_flight=len(running),
                            active_seconds=round(self.clock.state["run_active_seconds"], 1),
                        )
        finally:
            # Stop child processes before joining threads or releasing the run lock.
            # cancel() also prevents a generation->verification race from starting a new GPU process.
            for backend in self.backends:
                getattr(backend, "cancel", backend.close)()
            pool.shutdown(wait=True, cancel_futures=True)
            # Another lane may have finished a rejected score while this lane stopped.
            # Keep unfinished work for resume, but never retain a known CLIP rejection.
            for record in self.pending_records():
                if (
                    record["status"] != "scored"
                    or (self.root / f"records/{record['candidate_id']}.json").exists()
                ):
                    continue
                ledger = self.root / f"timings/attempts/{record['candidate_id']}.json"
                if ledger.exists() and read_json(ledger)["outcome"] in ("rejected", "duplicate", "error"):
                    # A failed cleanup already has a durable journal. Preserve its partial
                    # accounting for recovery, rather than retrying/masking the error here.
                    continue
                if decide(record["clip"]) == "rejected":
                    self._discard(record, "rejected")
                elif record["pixel_sha256"] in self.seen:
                    self._discard(record, "duplicate")

    def run(self):
        with run_lock(self.root):
            self.state = read_json(self.root / "state.json")
            self.clock = ActiveClock(self.root, clock=self.clock_function)
            self.clock.start(overhead=self.initial_seconds)
            self.state.update(status="running", last_error=None)
            self.state["segments"].append(
                {
                    "started_at": now(),
                    "selected_gpu_ids": self.gpu_ids,
                    "gpu_groups": self.gpu_groups,
                    "candidate_concurrency": len(self.backends),
                }
            )
            self.save_state()
            final_status = "failed"
            try:
                self._recover()
                target = self.cfg["run"]["target_count"]
                if len(self.accepted) < target:
                    self._segments()
                self._run_candidates(target)
                final_status = "completed"
            except KeyboardInterrupt:
                final_status = "interrupted"
                for pending in self.pending_records():
                    if not (self.root / f"records/{pending['candidate_id']}.json").exists():
                        save_attempt(self.root, self._timing(pending, "interrupted", cleanup_state="private"))
            except Exception as exc:
                code = (
                    str(exc)
                    if str(exc)
                    in ("no_eligible_sources", "no_movable_sources", "consecutive_technical_failures")
                    else type(exc).__name__
                )
                self.state["last_error"] = {"code": code}
                LOG.error("Run stopped: %s", code)
            finally:
                for backend in self.backends:
                    backend.close()
                self.state["segments"][-1]["ended_at"] = now()
                self.state["status"] = final_status
                self.save_state()
                try:
                    from .reporting import export_run, report_run

                    if self.backend.provenance != "fake":
                        export_run(self.root)
                    report_run(self.root)
                    self.clock.finish()
                    # This final metadata refresh is outside the measured active segment.
                    report_run(self.root)
                except Exception as exc:
                    self.state.update(status="failed", last_error={"code": type(exc).__name__})
                    if self.clock.state["active"]:
                        self.clock.finish()
                    self.save_state()
                    raise
        return self.state
