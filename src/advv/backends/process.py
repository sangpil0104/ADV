from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import CancelledError
from pathlib import Path

from ..contracts import Generated
from ..environment import require_advv
from ..errors import BackendError, FatalBackendError
from ..storage import digest, read_json, within


class Worker:
    def __init__(self, kind: str, cfg: dict, run_dir: Path):
        require_advv()
        selected = cfg["execution"]["selected_gpu_ids"]
        gpus = selected[:2] if kind == "generator" else selected[:1]
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=",".join(gpus),
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            TOKENIZERS_PARALLELISM="false",
            PYTHONUNBUFFERED="1",
            MPLBACKEND="Agg",
        )
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "advv.backends.worker", "--kind", kind, "--run-dir", str(run_dir)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            # Third-party progress/debug can contain rejected prompts and image paths.
            # Durable failures expose error codes only, not these raw streams.
            stderr=subprocess.DEVNULL,
            cwd=str(run_dir),
            env=env,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        self.responses = queue.Queue()
        self._close_lock = threading.Lock()
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()
        self.timeout = cfg["execution"]["worker_timeout_seconds"]

    def _read(self):
        try:
            for line in self.proc.stdout:
                self.responses.put(line)
        finally:
            self.responses.put(None)

    def request(self, request: dict) -> dict:
        try:
            self.proc.stdin.write(json.dumps(request) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise FatalBackendError("worker_pipe_closed") from exc
        end = time.monotonic() + self.timeout
        while True:
            try:
                line = self.responses.get(timeout=min(0.2, max(0.001, end - time.monotonic())))
                break
            except queue.Empty:
                if time.monotonic() >= end:
                    self.close()
                    raise BackendError("worker_timeout")
        if line is None:
            raise FatalBackendError("worker_exited")
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FatalBackendError("worker_protocol_error") from exc
        if response.get("identity") != request["identity"]:
            raise FatalBackendError("worker_identity_mismatch")
        if not response["ok"]:
            cls = FatalBackendError if response["error"]["fatal"] else BackendError
            raise cls(response["error"]["code"])
        return response["result"]

    def close(self):
        with self._close_lock:
            self._close()

    def _close(self):
        if self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.proc.wait()
        self.thread.join(timeout=2)
        for pipe in (self.proc.stdin, self.proc.stdout):
            if not pipe.closed:
                pipe.close()


class LocalBackend:
    provenance = "dragflow+clip_local"

    def __init__(self, cfg: dict, run_dir: Path):
        self.cfg, self.run_dir = cfg, run_dir
        self.worker = None
        self._lifecycle_lock = threading.Lock()
        self._cancelled = threading.Event()

    def _call(self, kind, request, receipt):
        if self._cancelled.is_set():
            raise CancelledError("Candidate lane stopped")
        request = dict(request, receipt=receipt)
        request["identity"] = digest({"request": request, "assets": self.cfg["_assets"], "kind": kind})
        saved = within(self.run_dir, receipt)
        if saved.exists():
            response = read_json(saved)
            if response.get("identity") != request["identity"]:
                raise FatalBackendError("receipt_identity_changed")
            if response["ok"]:
                return response["result"]
        self.close()
        with self._lifecycle_lock:
            if self._cancelled.is_set():
                raise CancelledError("Candidate lane stopped")
            worker = self.worker = Worker(kind, self.cfg, self.run_dir)
        try:
            return worker.request(request)
        except Exception:
            if self._cancelled.is_set():
                raise CancelledError("Candidate lane stopped") from None
            raise
        finally:
            self.close()

    def segment(self, source, run_dir):
        return self._call(
            "segmenter", {"source": source.to_dict()}, f"segments/{source.source_id}/receipt.json"
        )

    def verify(self, image, *, receipt):
        return self._call("verifier", {"image": str(image)}, receipt)

    def generate(self, source, plan, run_dir):
        result = self._call(
            "generator",
            {"source": source.to_dict(), "plan": plan.to_dict()},
            f".staging/{plan.edit_id}/generation_receipt.json",
        )
        return Generated(within(run_dir, result["path"]), result["effective"], result["info"])

    def close(self):
        with self._lifecycle_lock:
            worker, self.worker = self.worker, None
        if worker is not None:
            worker.close()

    def cancel(self):
        """Prevent new subprocesses before terminating this lane's current GPU process."""
        self._cancelled.set()
        self.close()
