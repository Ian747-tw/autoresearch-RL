"""
WorkerManager — optional parallel background experiment execution.

Rules:
- One orchestrator controls scheduling (this class is called by the orchestrator only)
- Each worker is a subprocess with dedicated GPU assignment
- Resource budgets are enforced before submission
- Workers are health-checked periodically
- Kill/pause/resume are supported
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CONFIG_DIR = ".drl_autoresearch"
WORKERS_JSON = "workers.json"
RESOURCE_LOG = "resource_log.jsonl"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_worker_id() -> str:
    return f"worker-{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# WorkerStatus dataclass
# ---------------------------------------------------------------------------


@dataclass
class WorkerStatus:
    worker_id: str
    status: str                  # queued|running|paused|done|failed|killed
    experiment: dict
    gpu_index: Optional[int]
    cpu_count: int
    pid: Optional[int]
    started_at: Optional[str]
    finished_at: Optional[str]
    exit_code: Optional[int]
    log_file: str
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "worker_id": self.worker_id,
            "status": self.status,
            "experiment": self.experiment,
            "gpu_index": self.gpu_index,
            "cpu_count": self.cpu_count,
            "pid": self.pid,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "exit_code": self.exit_code,
            "log_file": self.log_file,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WorkerStatus":
        return cls(
            worker_id=d["worker_id"],
            status=d.get("status", "queued"),
            experiment=d.get("experiment", {}),
            gpu_index=d.get("gpu_index"),
            cpu_count=d.get("cpu_count", 1),
            pid=d.get("pid"),
            started_at=d.get("started_at"),
            finished_at=d.get("finished_at"),
            exit_code=d.get("exit_code"),
            log_file=d.get("log_file", ""),
            notes=d.get("notes", ""),
        )


# ---------------------------------------------------------------------------
# WorkerManager
# ---------------------------------------------------------------------------


class WorkerManager:
    """
    Manages a pool of background subprocess workers for parallel experiment
    execution.  All mutations to internal state are guarded by a threading.Lock.
    """

    def __init__(
        self,
        project_dir: Path,
        max_workers: int = 1,
        hardware_config: Optional[dict] = None,
    ) -> None:
        self.project_dir = Path(project_dir)
        self.max_workers = max_workers
        self.hardware_config: dict = hardware_config or {}

        self._lock = threading.Lock()
        # worker_id -> WorkerStatus
        self._workers: Dict[str, WorkerStatus] = {}
        # worker_id -> subprocess.Popen (in-memory only)
        self._procs: Dict[str, subprocess.Popen] = {}
        # GPU index -> worker_id (or None if free)
        self._gpu_assignments: Dict[int, Optional[str]] = {}
        # log file size snapshots for stall detection: worker_id -> (size, timestamp)
        self._log_size_snapshots: Dict[str, tuple] = {}

        self._config_dir = self.project_dir / CONFIG_DIR
        self._config_dir.mkdir(parents=True, exist_ok=True)

        # Initialise GPU tracking from hardware_config
        self._init_gpu_pool()

        # Load persisted state if it exists
        self.load_state()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _init_gpu_pool(self) -> None:
        """Populate _gpu_assignments from hardware_config."""
        gpu_count = self.hardware_config.get("gpu_count", 0)
        gpus = self.hardware_config.get("gpus", [])
        if gpus:
            for gpu in gpus:
                idx = int(gpu.get("index", 0))
                self._gpu_assignments[idx] = None
        elif gpu_count > 0:
            for i in range(gpu_count):
                self._gpu_assignments[i] = None

    # ------------------------------------------------------------------
    # Resource assignment
    # ------------------------------------------------------------------

    def assign_resources(self, experiment: dict) -> dict:
        """
        Determine GPU index, CPU count, and RAM for an experiment.

        Returns {"gpu_index": int|None, "cpu_count": int, "ram_gb": float}.
        Raises RuntimeError if resources are unavailable.
        Logs every decision to resource_log.jsonl.
        """
        with self._lock:
            return self._assign_resources_locked(experiment)

    def _assign_resources_locked(self, experiment: dict) -> dict:
        """Inner resource assignment; caller must hold self._lock."""
        budget = experiment.get("resource_budget", {})
        requested_vram = float(budget.get("gpu_memory_gb", 0.0))
        cpu_count = int(budget.get("cpu_cores", self.hardware_config.get("cpu_cores", 4)))

        # Total VRAM available
        total_vram = self._total_vram_gb()

        # Check VRAM budget: existing assignments + new request <= 90%
        assigned_vram = self._assigned_vram_gb()
        if total_vram > 0 and (assigned_vram + requested_vram) > total_vram * 0.9:
            reason = (
                f"VRAM budget exceeded: assigned={assigned_vram:.1f}GB + "
                f"requested={requested_vram:.1f}GB > 90% of total={total_vram:.1f}GB"
            )
            self._log_resource_decision(experiment, None, cpu_count, False, reason)
            raise RuntimeError(reason)

        # Find a free GPU
        gpu_index: Optional[int] = None
        if self._gpu_assignments:
            for idx, owner in self._gpu_assignments.items():
                if owner is None:
                    gpu_index = idx
                    break

            if gpu_index is None:
                # All GPUs are busy; refuse if a GPU is strictly needed
                if requested_vram > 0:
                    reason = "All GPUs are currently assigned and experiment needs GPU VRAM."
                    self._log_resource_decision(experiment, None, cpu_count, False, reason)
                    raise RuntimeError(reason)
                # Otherwise fall back to CPU-only
                reason = "No free GPU; proceeding CPU-only."
                self._log_resource_decision(experiment, None, cpu_count, True, reason)
                return {"gpu_index": None, "cpu_count": cpu_count, "ram_gb": float(budget.get("ram_gb", 0.0))}

        result = {"gpu_index": gpu_index, "cpu_count": cpu_count, "ram_gb": float(budget.get("ram_gb", 0.0))}
        self._log_resource_decision(experiment, gpu_index, cpu_count, True, "assigned")
        return result

    def _total_vram_gb(self) -> float:
        gpus = self.hardware_config.get("gpus", [])
        if gpus:
            return sum(float(g.get("vram_gb", 0.0)) for g in gpus)
        return float(self.hardware_config.get("gpu_memory_limit_gb", 0.0))

    def _assigned_vram_gb(self) -> float:
        """Sum VRAM of all currently running workers that have a GPU assigned."""
        total = 0.0
        for ws in self._workers.values():
            if ws.status in ("running", "queued", "paused") and ws.gpu_index is not None:
                vram = float(ws.experiment.get("resource_budget", {}).get("gpu_memory_gb", 0.0))
                total += vram
        return total

    def _log_resource_decision(
        self,
        experiment: dict,
        gpu_index: Optional[int],
        cpu_count: int,
        success: bool,
        reason: str,
    ) -> None:
        """Append a resource allocation decision to resource_log.jsonl."""
        log_path = self._config_dir / RESOURCE_LOG
        entry = {
            "timestamp": _now_iso(),
            "run_id": experiment.get("run_id", ""),
            "gpu_index": gpu_index,
            "cpu_count": cpu_count,
            "success": success,
            "reason": reason,
        }
        try:
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        except OSError:
            pass  # non-fatal

    # ------------------------------------------------------------------
    # Submit
    # ------------------------------------------------------------------

    def submit(self, experiment: dict, command: List[str]) -> str:
        """
        Submit an experiment as a new background subprocess.

        Returns the worker_id.
        Raises RuntimeError if max_workers is reached or resources unavailable.
        """
        with self._lock:
            running_count = sum(
                1 for ws in self._workers.values()
                if ws.status in ("running", "queued", "paused")
            )
            if running_count >= self.max_workers:
                raise RuntimeError(
                    f"max_workers={self.max_workers} already active; cannot submit more."
                )

            resources = self._assign_resources_locked(experiment)
            gpu_index = resources["gpu_index"]
            cpu_count = resources["cpu_count"]

            worker_id = _make_worker_id()
            logs_dir = self.project_dir / "logs" / "runs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            log_file = str(logs_dir / f"{worker_id}.log")

            ws = WorkerStatus(
                worker_id=worker_id,
                status="queued",
                experiment=experiment,
                gpu_index=gpu_index,
                cpu_count=cpu_count,
                pid=None,
                started_at=None,
                finished_at=None,
                exit_code=None,
                log_file=log_file,
                notes="",
            )
            self._workers[worker_id] = ws

            # Build environment
            env = dict(os.environ)
            if gpu_index is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
            else:
                env["CUDA_VISIBLE_DEVICES"] = ""

            # Mark GPU as taken
            if gpu_index is not None:
                self._gpu_assignments[gpu_index] = worker_id

            # Launch subprocess
            try:
                log_fh = open(log_file, "w", encoding="utf-8")  # noqa: WPS515
                proc = subprocess.Popen(
                    command,
                    env=env,
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    cwd=str(self.project_dir),
                    close_fds=True,
                )
                log_fh.close()  # Popen inherits the fd; we can close our handle
            except Exception as exc:
                ws.status = "failed"
                ws.notes = f"Launch error: {exc}"
                ws.finished_at = _now_iso()
                if gpu_index is not None:
                    self._gpu_assignments[gpu_index] = None
                self._save_state_locked()
                raise

            ws.status = "running"
            ws.pid = proc.pid
            ws.started_at = _now_iso()
            self._procs[worker_id] = proc
            self._log_size_snapshots[worker_id] = (0, time.monotonic())
            self._save_state_locked()

        return worker_id

    # ------------------------------------------------------------------
    # Status queries
    # ------------------------------------------------------------------

    def get_status(self, worker_id: str) -> WorkerStatus:
        with self._lock:
            if worker_id not in self._workers:
                raise KeyError(f"Unknown worker_id: {worker_id!r}")
            ws = self._workers[worker_id]
            self._poll_worker_locked(ws)
            return ws

    def get_all_status(self) -> List[WorkerStatus]:
        with self._lock:
            for ws in list(self._workers.values()):
                self._poll_worker_locked(ws)
            return list(self._workers.values())

    def _poll_worker_locked(self, ws: WorkerStatus) -> None:
        """Check if the process has finished and update ws accordingly."""
        if ws.status not in ("running", "paused"):
            return
        proc = self._procs.get(ws.worker_id)
        if proc is None:
            # Process reference lost (e.g. after reload from disk)
            return
        retcode = proc.poll()
        if retcode is not None:
            ws.exit_code = retcode
            ws.finished_at = _now_iso()
            ws.status = "done" if retcode == 0 else "failed"
            if ws.gpu_index is not None:
                self._gpu_assignments[ws.gpu_index] = None
            self._save_state_locked()

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def kill(self, worker_id: str, reason: str = "") -> None:
        """Send SIGKILL to the worker process."""
        with self._lock:
            ws = self._get_worker_locked(worker_id)
            proc = self._procs.get(worker_id)
            if proc is not None:
                try:
                    proc.kill()
                    proc.wait(timeout=5)
                except (ProcessLookupError, subprocess.TimeoutExpired, OSError):
                    pass
            ws.status = "killed"
            ws.finished_at = _now_iso()
            ws.notes = reason or "killed by WorkerManager"
            if ws.gpu_index is not None:
                self._gpu_assignments[ws.gpu_index] = None
            self._save_state_locked()

    def pause(self, worker_id: str) -> None:
        """Send SIGSTOP to suspend the worker process."""
        with self._lock:
            ws = self._get_worker_locked(worker_id)
            if ws.status != "running":
                raise RuntimeError(f"Worker {worker_id} is not running (status={ws.status})")
            proc = self._procs.get(worker_id)
            if proc is not None and proc.pid is not None:
                try:
                    os.kill(proc.pid, signal.SIGSTOP)
                except (ProcessLookupError, PermissionError, OSError) as exc:
                    raise RuntimeError(f"SIGSTOP failed: {exc}") from exc
            ws.status = "paused"
            self._save_state_locked()

    def resume(self, worker_id: str) -> None:
        """Send SIGCONT to resume a paused worker process."""
        with self._lock:
            ws = self._get_worker_locked(worker_id)
            if ws.status != "paused":
                raise RuntimeError(f"Worker {worker_id} is not paused (status={ws.status})")
            proc = self._procs.get(worker_id)
            if proc is not None and proc.pid is not None:
                try:
                    os.kill(proc.pid, signal.SIGCONT)
                except (ProcessLookupError, PermissionError, OSError) as exc:
                    raise RuntimeError(f"SIGCONT failed: {exc}") from exc
            ws.status = "running"
            self._save_state_locked()

    def _get_worker_locked(self, worker_id: str) -> WorkerStatus:
        if worker_id not in self._workers:
            raise KeyError(f"Unknown worker_id: {worker_id!r}")
        return self._workers[worker_id]

    # ------------------------------------------------------------------
    # Health checks
    # ------------------------------------------------------------------

    def health_check_all(self) -> List[str]:
        """
        Run health checks across all active workers.

        Returns a list of worker_ids that are unhealthy (stalled, OOM, etc.).
        Also polls each process and updates status.
        """
        unhealthy: List[str] = []
        stall_threshold_secs = 600  # 10 minutes

        with self._lock:
            for worker_id, ws in list(self._workers.items()):
                if ws.status not in ("running", "paused"):
                    continue

                proc = self._procs.get(worker_id)

                # 1. Process liveness
                if proc is None or proc.poll() is not None:
                    # Already finished — update status
                    self._poll_worker_locked(ws)
                    continue

                # 2. Log file growing check
                log_path = Path(ws.log_file)
                if log_path.exists():
                    current_size = log_path.stat().st_size
                    prev_size, prev_time = self._log_size_snapshots.get(worker_id, (current_size, time.monotonic()))
                    elapsed = time.monotonic() - prev_time

                    if current_size == prev_size and elapsed >= stall_threshold_secs:
                        note = (
                            f"Stalled: log has not grown for {elapsed/60:.0f} minutes "
                            f"(size={current_size} bytes)."
                        )
                        if note not in ws.notes:
                            ws.notes = (ws.notes + " | " + note).strip(" | ")
                        unhealthy.append(worker_id)
                    elif current_size != prev_size:
                        # Reset snapshot when file grows
                        self._log_size_snapshots[worker_id] = (current_size, time.monotonic())
                    # First snapshot
                    if worker_id not in self._log_size_snapshots:
                        self._log_size_snapshots[worker_id] = (current_size, time.monotonic())
                else:
                    # Log file missing for a running worker — suspicious after > 30 s
                    if ws.started_at:
                        try:
                            started = datetime.fromisoformat(ws.started_at)
                            if started.tzinfo is None:
                                started = started.replace(tzinfo=timezone.utc)
                            age = (datetime.now(timezone.utc) - started).total_seconds()
                            if age > 30:
                                unhealthy.append(worker_id)
                                ws.notes = (ws.notes + " | log file missing").strip(" | ")
                        except ValueError:
                            pass

                # 3. Best-effort OOM detection from kernel log
                self._check_oom_locked(ws)

            if unhealthy:
                self._save_state_locked()

        return unhealthy

    def _check_oom_locked(self, ws: WorkerStatus) -> None:
        """Best-effort check for OOM killer activity in /var/log/kern.log or dmesg."""
        if ws.pid is None:
            return
        try:
            kern_log = Path("/var/log/kern.log")
            if kern_log.exists():
                text = kern_log.read_text(encoding="utf-8", errors="ignore")
                if f"Killed process {ws.pid}" in text or f"pid {ws.pid}" in text:
                    if "oom" not in ws.notes.lower():
                        ws.notes = (ws.notes + " | OOM suspected from kern.log").strip(" | ")
        except OSError:
            pass  # non-fatal

    # ------------------------------------------------------------------
    # Blocking wait
    # ------------------------------------------------------------------

    def wait_for_any(self, timeout: float = 60.0) -> Optional[WorkerStatus]:
        """
        Block until any one running worker finishes, or timeout expires.

        Returns the finished WorkerStatus, or None if timeout reached.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                for worker_id, ws in list(self._workers.items()):
                    if ws.status in ("running", "paused"):
                        proc = self._procs.get(worker_id)
                        if proc is not None and proc.poll() is not None:
                            self._poll_worker_locked(ws)
                            return ws
            time.sleep(0.5)
        return None

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup_done(self) -> None:
        """Remove finished/failed/killed workers from the in-memory registry."""
        with self._lock:
            terminal = ("done", "failed", "killed")
            to_remove = [
                wid for wid, ws in self._workers.items()
                if ws.status in terminal
            ]
            for wid in to_remove:
                self._workers.pop(wid, None)
                self._procs.pop(wid, None)
                self._log_size_snapshots.pop(wid, None)
            if to_remove:
                self._save_state_locked()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self) -> None:
        """Persist worker registry to .drl_autoresearch/workers.json."""
        with self._lock:
            self._save_state_locked()

    def _save_state_locked(self) -> None:
        """Inner save — caller must hold self._lock."""
        path = self._config_dir / WORKERS_JSON
        data = [ws.to_dict() for ws in self._workers.values()]
        tmp = path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)
        except OSError:
            pass  # non-fatal

    def load_state(self) -> None:
        """Load persisted worker registry from .drl_autoresearch/workers.json."""
        path = self._config_dir / WORKERS_JSON
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                return
            with self._lock:
                for entry in data:
                    try:
                        ws = WorkerStatus.from_dict(entry)
                        self._workers[ws.worker_id] = ws
                        # We cannot restore Popen objects; mark running as unknown
                        if ws.status in ("running", "queued", "paused"):
                            ws.notes = (ws.notes + " | state reloaded; proc ref lost").strip(" | ")
                        # Restore GPU assignments for still-active workers
                        if ws.status in ("running", "queued", "paused") and ws.gpu_index is not None:
                            self._gpu_assignments[ws.gpu_index] = ws.worker_id
                    except (KeyError, TypeError, ValueError):
                        continue
        except (OSError, json.JSONDecodeError):
            pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        counts = {"running": 0, "queued": 0, "done": 0, "failed": 0}
        with self._lock:
            for ws in self._workers.values():
                counts[ws.status] = counts.get(ws.status, 0) + 1
        return (
            f"WorkerManager(max={self.max_workers}, "
            f"running={counts.get('running',0)}, "
            f"queued={counts.get('queued',0)}, "
            f"done={counts.get('done',0)}, "
            f"failed={counts.get('failed',0)})"
        )
