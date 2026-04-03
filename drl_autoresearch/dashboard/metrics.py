"""
MetricsCollector — reads experiment data from logs and aggregates for the dashboard.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Optional psutil import
# ---------------------------------------------------------------------------
try:
    import psutil as _psutil  # type: ignore

    _PSUTIL_AVAILABLE = True
except ImportError:
    _psutil = None  # type: ignore
    _PSUTIL_AVAILABLE = False


CONFIG_DIR = ".drl_autoresearch"
STATE_FILENAME = "state.json"
INCIDENTS_FILENAME = "incidents.json"
DECISIONS_FILENAME = "decisions.json"
WORKERS_FILENAME = "workers.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# DashboardData
# ---------------------------------------------------------------------------


@dataclass
class DashboardData:
    timestamp: str
    project_name: str
    current_phase: str
    active_run_id: Optional[str]
    total_runs: int
    kept_runs: int
    discarded_runs: int
    crashed_runs: int
    best_run_id: Optional[str]
    best_metric_value: Optional[float]
    best_metric_name: str
    experiment_timeline: list  # [{run_id, timestamp, metric, status, hypothesis}]
    training_curves: dict  # run_id -> {steps: [], rewards: [], losses: []}
    eval_curves: dict  # run_id -> {steps: [], eval_rewards: []}
    resource_usage: dict  # {cpu_percent, ram_gb, gpu_memory_gb, gpu_util}
    workers: list  # [{worker_id, status, run_id, started_at}]
    incidents: list  # [{id, type, severity, description, timestamp}]
    top_runs: list  # [{run_id, metric, hypothesis, changes}]
    recent_decisions: list  # [{timestamp, decision, reason}]
    next_experiment: Optional[dict]
    morning_summary: Optional[dict]
    workflow: dict  # {project_mode, bootstrap_state, refresh_cooldown_remaining_runs, last_refresh_reason}

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "project_name": self.project_name,
            "current_phase": self.current_phase,
            "active_run_id": self.active_run_id,
            "total_runs": self.total_runs,
            "kept_runs": self.kept_runs,
            "discarded_runs": self.discarded_runs,
            "crashed_runs": self.crashed_runs,
            "best_run_id": self.best_run_id,
            "best_metric_value": self.best_metric_value,
            "best_metric_name": self.best_metric_name,
            "experiment_timeline": self.experiment_timeline,
            "training_curves": self.training_curves,
            "eval_curves": self.eval_curves,
            "resource_usage": self.resource_usage,
            "workers": self.workers,
            "incidents": self.incidents,
            "top_runs": self.top_runs,
            "recent_decisions": self.recent_decisions,
            "next_experiment": self.next_experiment,
            "morning_summary": self.morning_summary,
            "workflow": self.workflow,
        }


# ---------------------------------------------------------------------------
# MetricsCollector
# ---------------------------------------------------------------------------


class MetricsCollector:
    """Reads experiment data from logs and aggregates for the dashboard."""

    def __init__(self, project_dir: Path) -> None:
        self.project_dir = Path(project_dir)
        self._config_dir = self.project_dir / CONFIG_DIR

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def collect(self) -> DashboardData:
        """Read all logs and aggregate into DashboardData."""
        state = self._load_state()
        timeline = self.collect_experiment_timeline()
        training_curves, eval_curves = self._split_curves(timeline)

        # Active run: most recent "running" entry from timeline
        active_run_id: Optional[str] = None
        for entry in reversed(timeline):
            if entry.get("status") == "running":
                active_run_id = entry.get("run_id")
                break
        if active_run_id is None:
            flags = state.get("flags", {})
            if isinstance(flags, dict):
                flagged_run_id = flags.get("active_run_id")
                if isinstance(flagged_run_id, str) and flagged_run_id.strip():
                    active_run_id = flagged_run_id.strip()

        top_runs = self.collect_best_models(n=5, timeline=timeline)

        morning = self.get_overnight_summary(
            state=state, timeline=timeline
        )

        return DashboardData(
            timestamp=_now_iso(),
            project_name=state.get("project_name", self.project_dir.name),
            current_phase=state.get("current_phase", "research"),
            active_run_id=active_run_id,
            total_runs=state.get("total_runs", len(timeline)),
            kept_runs=state.get("kept_runs", 0),
            discarded_runs=state.get("discarded_runs", 0),
            crashed_runs=state.get("crashed_runs", 0),
            best_run_id=state.get("best_run_id"),
            best_metric_value=state.get("best_metric_value"),
            best_metric_name=state.get("best_metric_name", "reward"),
            experiment_timeline=timeline,
            training_curves=training_curves,
            eval_curves=eval_curves,
            resource_usage=self.collect_resource_usage(),
            workers=self.collect_worker_status(state=state),
            incidents=self.collect_incidents(),
            top_runs=top_runs,
            recent_decisions=self.collect_recent_decisions(),
            next_experiment=self._next_experiment(state),
            morning_summary=morning,
            workflow=self._collect_workflow_state(state),
        )

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _load_state(self) -> dict:
        state_path = self._config_dir / STATE_FILENAME
        if state_path.exists():
            try:
                return json.loads(state_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def _next_experiment(self, state: dict) -> Optional[dict]:
        queue = state.get("queue", [])
        return queue[0] if queue else None

    def _collect_workflow_state(self, state: dict) -> dict:
        flags = state.get("flags", {})
        if not isinstance(flags, dict):
            flags = {}

        total_runs = state.get("total_runs", 0)
        try:
            total_runs_i = int(total_runs)
        except (TypeError, ValueError):
            total_runs_i = 0

        cooldown_window = 3
        last_refresh_total_runs = flags.get("last_refresh_total_runs")
        cooldown_remaining = 0
        if isinstance(last_refresh_total_runs, int):
            delta = total_runs_i - last_refresh_total_runs
            cooldown_remaining = max(0, cooldown_window - delta)

        return {
            "project_mode": flags.get("project_mode", "improve"),
            "loop_running": bool(flags.get("loop_running", False)),
            "current_activity": flags.get("current_activity"),
            "agent_backend": flags.get("agent_backend"),
            "active_run_id": flags.get("active_run_id"),
            "last_agent_exit_code": flags.get("last_agent_exit_code"),
            "build_bootstrap_started": bool(flags.get("build_bootstrap_started", False)),
            "build_bootstrap_complete": bool(flags.get("build_bootstrap_complete", True)),
            "build_bootstrap_research_applied": bool(
                flags.get("build_bootstrap_research_applied", False)
            ),
            "refresh_cooldown_remaining_runs": cooldown_remaining,
            "last_refresh_reason": flags.get("last_refresh_reason"),
        }

    # ------------------------------------------------------------------
    # Training curves
    # ------------------------------------------------------------------

    def collect_training_curves(self) -> list[dict]:
        """
        Collect training curves from run artifact JSON files under
        logs/artifacts/<run_id>/metrics.json.

        Each artifact file is expected to contain:
            {"steps": [...], "rewards": [...], "losses": [...]}
        """
        artifacts_dir = self.project_dir / "logs" / "artifacts"
        results = []
        if not artifacts_dir.is_dir():
            return results
        for run_dir in sorted(artifacts_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            metrics_file = run_dir / "metrics.json"
            if metrics_file.exists():
                try:
                    data = json.loads(metrics_file.read_text(encoding="utf-8"))
                    data["run_id"] = run_dir.name
                    results.append(data)
                except Exception:
                    pass
        return results

    def _split_curves(
        self, timeline: list[dict]
    ) -> tuple[dict, dict]:
        """
        Build per-run curve dicts from the artifact files and/or from the
        scalar metrics embedded in the timeline rows.

        Returns (training_curves, eval_curves) where each is:
            {run_id: {"steps": [...], "rewards": [...], "losses": []}}
        """
        training_curves: dict = {}
        eval_curves: dict = {}

        # Prefer full artifact curves when they exist
        for artifact in self.collect_training_curves():
            run_id = artifact.get("run_id", "")
            if not run_id:
                continue
            training_curves[run_id] = {
                "steps": artifact.get("steps", []),
                "rewards": artifact.get("rewards", []),
                "losses": artifact.get("losses", []),
            }
            if "eval_steps" in artifact or "eval_rewards" in artifact:
                eval_curves[run_id] = {
                    "steps": artifact.get("eval_steps", artifact.get("steps", [])),
                    "eval_rewards": artifact.get("eval_rewards", []),
                }

        # Fallback: single-point curves from registry scalars
        for i, entry in enumerate(timeline):
            run_id = entry.get("run_id", "")
            if not run_id or run_id in training_curves:
                continue
            train_val = entry.get("train_reward_mean")
            eval_val = entry.get("eval_reward_mean")
            step = i  # use run index as x-axis position
            if train_val is not None:
                training_curves[run_id] = {
                    "steps": [step],
                    "rewards": [train_val],
                    "losses": [],
                }
            if eval_val is not None:
                eval_curves[run_id] = {
                    "steps": [step],
                    "eval_rewards": [eval_val],
                }

        return training_curves, eval_curves

    # ------------------------------------------------------------------
    # Experiment timeline
    # ------------------------------------------------------------------

    def collect_experiment_timeline(self) -> list[dict]:
        """Read logs/experiment_registry.tsv and parse into list of experiment dicts."""
        tsv_path = self.project_dir / "logs" / "experiment_registry.tsv"
        if not tsv_path.exists():
            return []

        try:
            text = tsv_path.read_text(encoding="utf-8")
        except Exception:
            return []

        lines = text.splitlines()
        if len(lines) < 2:
            return []

        header = [c.strip() for c in lines[0].split("\t")]
        results = []

        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            try:
                raw = line.split("\t")
                raw += [""] * (len(header) - len(raw))
                row: dict[str, Any] = {}
                for i, col in enumerate(header):
                    val = raw[i] if i < len(raw) else ""
                    # unescape
                    val = val.replace("\\n", "\n").replace("\\t", "\t").replace("\\\\", "\\")
                    # coerce known numerics
                    if col in {
                        "train_reward_mean", "train_reward_std",
                        "eval_reward_mean", "eval_reward_std",
                        "custom_metric_value", "success_rate",
                        "constraint_violations", "seed_count",
                        "wall_clock_seconds", "gpu_memory_gb", "ram_gb",
                    }:
                        row[col] = float(val) if val.strip() != "" else None
                    else:
                        row[col] = val
                # Normalise key names for dashboard consumption
                results.append(
                    {
                        "run_id": row.get("run_id", ""),
                        "timestamp": row.get("timestamp", ""),
                        "status": row.get("status", ""),
                        "keep_decision": row.get("keep_decision", ""),
                        "hypothesis": row.get("hypothesis", ""),
                        "change_summary": row.get("change_summary", ""),
                        "algorithm": row.get("algorithm", ""),
                        "train_reward_mean": row.get("train_reward_mean"),
                        "train_reward_std": row.get("train_reward_std"),
                        "eval_reward_mean": row.get("eval_reward_mean"),
                        "eval_reward_std": row.get("eval_reward_std"),
                        "wall_clock_seconds": row.get("wall_clock_seconds"),
                        "notes": row.get("notes", ""),
                        "agent": row.get("agent", ""),
                        "branch": row.get("branch", ""),
                        "commit": row.get("commit", ""),
                    }
                )
            except Exception:
                continue  # skip malformed rows

        return results

    # ------------------------------------------------------------------
    # Resource usage
    # ------------------------------------------------------------------

    def collect_resource_usage(self) -> dict:
        """
        Return current system resource usage.
        Uses psutil if available, falls back to /proc on Linux,
        and nvidia-smi for GPU. Returns {"available": false} on complete failure.
        """
        result: dict[str, Any] = {"available": False}

        # ---- CPU / RAM via psutil ----
        if _PSUTIL_AVAILABLE:
            try:
                result["cpu_percent"] = _psutil.cpu_percent(interval=0.1)
                vm = _psutil.virtual_memory()
                result["ram_gb"] = round(vm.used / 1024 ** 3, 2)
                result["ram_total_gb"] = round(vm.total / 1024 ** 3, 2)
                result["ram_percent"] = vm.percent
                result["available"] = True
            except Exception:
                pass
        else:
            # /proc fallback (Linux only)
            try:
                cpu_pct = self._read_proc_cpu()
                if cpu_pct is not None:
                    result["cpu_percent"] = cpu_pct
                    result["available"] = True
            except Exception:
                pass
            try:
                ram_info = self._read_proc_meminfo()
                if ram_info:
                    result.update(ram_info)
                    result["available"] = True
            except Exception:
                pass

        # ---- GPU via nvidia-smi ----
        try:
            gpu_info = self._read_nvidia_smi()
            if gpu_info:
                result.update(gpu_info)
                result["available"] = True
        except Exception:
            pass

        return result

    def _read_proc_cpu(self) -> Optional[float]:
        """Compute a single CPU% sample from /proc/stat."""
        if not Path("/proc/stat").exists():
            return None
        # First sample
        line1 = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0]
        vals1 = list(map(int, line1.split()[1:]))
        import time
        time.sleep(0.1)
        line2 = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0]
        vals2 = list(map(int, line2.split()[1:]))
        idle1 = vals1[3]
        idle2 = vals2[3]
        total1 = sum(vals1)
        total2 = sum(vals2)
        d_total = total2 - total1
        d_idle = idle2 - idle1
        if d_total == 0:
            return 0.0
        return round(100.0 * (1.0 - d_idle / d_total), 1)

    def _read_proc_meminfo(self) -> Optional[dict]:
        meminfo = Path("/proc/meminfo")
        if not meminfo.exists():
            return None
        data: dict[str, int] = {}
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 2:
                key = parts[0].rstrip(":")
                try:
                    data[key] = int(parts[1])
                except ValueError:
                    pass
        total_kb = data.get("MemTotal", 0)
        available_kb = data.get("MemAvailable", 0)
        if total_kb == 0:
            return None
        used_kb = total_kb - available_kb
        return {
            "ram_gb": round(used_kb / 1024 ** 2, 2),
            "ram_total_gb": round(total_kb / 1024 ** 2, 2),
            "ram_percent": round(used_kb / total_kb * 100, 1),
        }

    def _read_nvidia_smi(self) -> Optional[dict]:
        """Query nvidia-smi for GPU memory and utilisation."""
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                stderr=subprocess.DEVNULL,
                timeout=5,
            ).decode("utf-8").strip()
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            return None

        gpus = []
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                try:
                    gpus.append(
                        {
                            "gpu_util": float(parts[0]),
                            "gpu_memory_used_mb": float(parts[1]),
                            "gpu_memory_total_mb": float(parts[2]),
                        }
                    )
                except ValueError:
                    pass

        if not gpus:
            return None

        # Aggregate across all GPUs (sum memory, mean util)
        total_used = sum(g["gpu_memory_used_mb"] for g in gpus)
        total_cap = sum(g["gpu_memory_total_mb"] for g in gpus)
        avg_util = sum(g["gpu_util"] for g in gpus) / len(gpus)

        return {
            "gpu_util": round(avg_util, 1),
            "gpu_memory_gb": round(total_used / 1024, 2),
            "gpu_memory_total_gb": round(total_cap / 1024, 2),
            "gpu_count": len(gpus),
            "gpus": gpus,
        }

    # ------------------------------------------------------------------
    # Worker status
    # ------------------------------------------------------------------

    def collect_worker_status(self, state: Optional[dict] = None) -> list[dict]:
        """
        Return active workers. Source priority:
        1. .drl_autoresearch/workers.json  (most detailed)
        2. active_workers list from state.json
        """
        workers_path = self._config_dir / WORKERS_FILENAME
        if workers_path.exists():
            try:
                data = json.loads(workers_path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    return data
            except Exception:
                pass

        # Fallback to state.json active_workers list
        if state is None:
            state = self._load_state()
        active: list = state.get("active_workers", [])
        return [
            {
                "worker_id": w if isinstance(w, str) else w.get("worker_id", str(w)),
                "status": "active",
                "run_id": w.get("run_id") if isinstance(w, dict) else None,
                "started_at": w.get("started_at") if isinstance(w, dict) else None,
            }
            for w in active
        ]

    # ------------------------------------------------------------------
    # Incidents
    # ------------------------------------------------------------------

    def collect_incidents(self) -> list[dict]:
        """
        Read open incidents from .drl_autoresearch/incidents.json.
        Returns empty list if the file does not exist.
        """
        incidents_path = self._config_dir / INCIDENTS_FILENAME
        if not incidents_path.exists():
            return []
        try:
            data = json.loads(incidents_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [i for i in data if not i.get("resolved", False)]
        except Exception:
            pass
        return []

    # ------------------------------------------------------------------
    # Best models
    # ------------------------------------------------------------------

    def collect_best_models(
        self,
        n: int = 5,
        metric: str = "eval_reward_mean",
        timeline: Optional[list[dict]] = None,
    ) -> list[dict]:
        """Return the top N runs ranked by metric, descending."""
        if timeline is None:
            timeline = self.collect_experiment_timeline()

        candidates = [
            r for r in timeline if r.get(metric) is not None
        ]
        ranked = sorted(candidates, key=lambda r: r[metric], reverse=True)[:n]

        return [
            {
                "run_id": r.get("run_id", ""),
                "metric": r.get(metric),
                "metric_name": metric,
                "hypothesis": r.get("hypothesis", ""),
                "change_summary": r.get("change_summary", ""),
                "status": r.get("status", ""),
                "keep_decision": r.get("keep_decision", ""),
                "algorithm": r.get("algorithm", ""),
            }
            for r in ranked
        ]

    # ------------------------------------------------------------------
    # Recent decisions
    # ------------------------------------------------------------------

    def collect_recent_decisions(self, n: int = 20) -> list[dict]:
        """
        Read recent keep/discard decisions.
        Source: .drl_autoresearch/decisions.json if present, else derived
        from the last N timeline entries.
        """
        decisions_path = self._config_dir / DECISIONS_FILENAME
        if decisions_path.exists():
            try:
                data = json.loads(decisions_path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    return data[-n:]
            except Exception:
                pass

        # Derive from timeline
        timeline = self.collect_experiment_timeline()
        decisions = []
        for entry in timeline:
            kd = entry.get("keep_decision", "")
            if kd in ("keep", "discard"):
                decisions.append(
                    {
                        "timestamp": entry.get("timestamp", ""),
                        "run_id": entry.get("run_id", ""),
                        "decision": kd,
                        "reason": entry.get("notes", ""),
                        "next_step": "",
                    }
                )
        return decisions[-n:]

    # ------------------------------------------------------------------
    # Overnight summary
    # ------------------------------------------------------------------

    def get_overnight_summary(
        self,
        state: Optional[dict] = None,
        timeline: Optional[list[dict]] = None,
    ) -> Optional[dict]:
        """
        Return a morning-review summary dict when the project has been
        running for more than 4 hours without interaction, or None otherwise.
        """
        if state is None:
            state = self._load_state()
        if timeline is None:
            timeline = self.collect_experiment_timeline()

        initialized_at_str = state.get("initialized_at", "")
        if not initialized_at_str:
            return None

        try:
            initialized_at = datetime.fromisoformat(initialized_at_str)
        except ValueError:
            return None

        now = datetime.now(timezone.utc)
        # Make initialized_at timezone-aware if naive
        if initialized_at.tzinfo is None:
            initialized_at = initialized_at.replace(tzinfo=timezone.utc)

        elapsed_hours = (now - initialized_at).total_seconds() / 3600.0
        if elapsed_hours < 4.0:
            return None

        kept = [r for r in timeline if r.get("keep_decision") == "keep"]
        crashed = [r for r in timeline if r.get("status") == "crashed"]
        discarded = [r for r in timeline if r.get("keep_decision") == "discard"]

        best_run: Optional[dict] = None
        best_val: Optional[float] = None
        for r in kept:
            v = r.get("eval_reward_mean")
            if v is not None and (best_val is None or v > best_val):
                best_val = v
                best_run = r

        return {
            "elapsed_hours": round(elapsed_hours, 1),
            "total_runs": len(timeline),
            "kept_runs": len(kept),
            "discarded_runs": len(discarded),
            "crashed_runs": len(crashed),
            "best_run_id": best_run.get("run_id") if best_run else None,
            "best_eval_reward": best_val,
            "current_phase": state.get("current_phase", "research"),
            "next_in_queue": len(state.get("queue", [])),
            "generated_at": _now_iso(),
        }
