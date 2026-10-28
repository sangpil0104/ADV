"""Detached coordinator startup and logs. Never changes an existing run's core code."""
from __future__ import annotations

import argparse
import fcntl
import os
import shutil
import signal
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from advv.environment import require_advv
from advv.errors import ADVVError
from advv.storage import atomic_bytes, atomic_json, digest, file_hash, now, read_json


def spawn_detached(command: list[str], stream):
    """Use real nohup without a shell; disconnect all terminal handles and process group."""
    executable = shutil.which("nohup")
    if executable is None:
        raise ADVVError("nohup executable not found; install coreutils or use explicit --foreground")
    return subprocess.Popen(
        [executable, *command],
        stdin=subprocess.DEVNULL,
        stdout=stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
        env=dict(os.environ, PYTHONUNBUFFERED="1"),
    )


def _latest_link(root: Path, name: str, target: Path):
    temporary = root / f".{name}.{uuid4().hex}"
    try:
        temporary.symlink_to(target.name)
        os.replace(temporary, root / name)
    finally:
        temporary.unlink(missing_ok=True)


def launch(argv: list[str], run_dir: Path, log_root: Path, *, details: dict):
    require_advv()
    log_root = log_root.resolve()
    log_root.mkdir(parents=True, exist_ok=True)
    # Serialize aliases across simultaneous launches. Each launch keeps unique durable paths.
    with (log_root / ".launcher.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        for name in ("latest.log", "latest.pid", "latest.json"):
            path = log_root / name
            if path.exists() and not path.is_symlink():
                raise ADVVError(f"Reserved log alias is an existing regular file: {path}")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        base = log_root / f"{stamp}_{uuid4().hex[:8]}"
        log, pid_file, status = (base.with_suffix(ext) for ext in (".log", ".pid", ".json"))
        info = {
            "version": "background_launch_v1", "status": "launching", "created_at": now(),
            "run_dir": str(run_dir.resolve()), "argv": argv, "cwd": str(Path.cwd()),
            "log_path": str(log), "pid_path": str(pid_file), "status_path": str(status),
            "python": sys.executable, **details,
            "launcher_sha256": digest({
                p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))
            }),
        }
        atomic_json(status, info)
        proc = None
        try:
            with log.open("xb") as stream:
                proc = spawn_detached(
                    [sys.executable, "-u", "-m", "advv_launcher.background", str(status)], stream
                )
            atomic_bytes(pid_file, f"{proc.pid}\n".encode())
            _latest_link(log_root, "latest.log", log)
            _latest_link(log_root, "latest.pid", pid_file)
            _latest_link(log_root, "latest.json", status)
        except BaseException:
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            atomic_json(status, dict(info, status="launch_failed", ended_at=now()))
            raise
        return dict(info, pid=proc.pid)


def _progress(run_dir: Path, stop: threading.Event, interval=60):
    while not stop.wait(interval):
        try:
            state = read_json(run_dir / "state.json")
            message = (
                f"status={state['status']} accepted={state['accepted_count']} "
                f"attempts_started={state['attempts']}"
            )
        except (OSError, ValueError, KeyError):
            message = "initializing (configuration/model checks; no run state yet)"
        # Only aggregate progress, never candidate scores, prompts or rejected output paths.
        print(f"{now()} ADVV heartbeat: {message}", flush=True)


def run_tracked(status_path: Path, execute, *, heartbeat_seconds=60):
    """Track coordinator lifetime, including errors before a run directory exists."""
    signal.signal(signal.SIGHUP, signal.SIG_IGN)

    def interrupted(signum, frame):
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, interrupted)
    info = read_json(status_path)
    info.update(pid=os.getpid(), status="running", started_at=now())
    atomic_json(status_path, info)
    print(f"{now()} ADVV coordinator PID {os.getpid()}: initializing", flush=True)
    stop = threading.Event()
    monitor = threading.Thread(
        target=_progress, args=(Path(info["run_dir"]), stop, heartbeat_seconds), daemon=True
    )
    code = 1
    monitor.start()
    try:
        code = execute(info["argv"])
        code = 0 if code is None else int(code)
    except KeyboardInterrupt:
        code = 130
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1 if exc.code else 0
    except Exception as exc:
        # Do not dump model traceback locals or generation details into persistent logs.
        print(f"{now()} ADVV coordinator failed: {type(exc).__name__}", flush=True)
        code = 1
    finally:
        stop.set()
        monitor.join(timeout=2)
        outcome = "completed" if code == 0 else "interrupted" if code == 130 else "failed"
        atomic_json(status_path, dict(info, status=outcome, exit_code=code, ended_at=now()))
        print(f"{now()} ADVV coordinator finished: {outcome} (exit_code={code})", flush=True)
    return code


def main():
    parser = argparse.ArgumentParser(description="Internal detached ADVV coordinator")
    parser.add_argument("status_path", type=Path)
    args = parser.parse_args()
    require_advv()
    from advv.cli import main as coordinator_main

    # Call the foreground coordinator directly; no recursive console launch.
    return run_tracked(args.status_path, coordinator_main)


if __name__ == "__main__":
    raise SystemExit(main())
