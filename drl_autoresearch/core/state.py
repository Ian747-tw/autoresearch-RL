"""
ProjectState — manages .drl_autoresearch/state.json for the target project.
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


STATE_VERSION = "1.0.0"
STATE_FILENAME = "state.json"
CONFIG_DIR = ".drl_autoresearch"

VALID_PHASES = [
    "research",
    "baseline",
    "experimenting",
    "focused_tuning",
    "ablation",
    "converged",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProjectState:
    """Manages persistent state for a DRL AutoResearch project."""

    def __init__(
        self,
        project_dir: Path,
        version: str = STATE_VERSION,
        project_name: str = "",
        initialized_at: str = "",
        current_phase: str = "research",
        current_branch: Optional[str] = None,
        best_run_id: Optional[str] = None,
        best_metric_value: Optional[float] = None,
        best_metric_name: str = "reward",
        total_runs: int = 0,
        kept_runs: int = 0,
        discarded_runs: int = 0,
        crashed_runs: int = 0,
        last_updated: str = "",
        active_workers: Optional[List[str]] = None,
        queue: Optional[List[Dict[str, Any]]] = None,
        flags: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.project_dir = Path(project_dir)
        self.version = version
        self.project_name = project_name or self.project_dir.name
        self.initialized_at = initialized_at or _now_iso()
        self.current_phase = current_phase
        self.current_branch = current_branch
        self.best_run_id = best_run_id
        self.best_metric_value = best_metric_value
        self.best_metric_name = best_metric_name
        self.total_runs = total_runs
        self.kept_runs = kept_runs
        self.discarded_runs = discarded_runs
        self.crashed_runs = crashed_runs
        self.last_updated = last_updated or _now_iso()
        self.active_workers: List[str] = active_workers if active_workers is not None else []
        self.queue: List[Dict[str, Any]] = queue if queue is not None else []
        self.flags: Dict[str, Any] = flags if flags is not None else {}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @property
    def _config_dir(self) -> Path:
        return self.project_dir / CONFIG_DIR

    @property
    def _state_path(self) -> Path:
        return self._config_dir / STATE_FILENAME

    @classmethod
    def load(cls, project_dir: Path) -> "ProjectState":
        """Load state from disk, or return a fresh default state if not found."""
        project_dir = Path(project_dir)
        state_path = project_dir / CONFIG_DIR / STATE_FILENAME

        if not state_path.exists():
            return cls(project_dir=project_dir)

        with state_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        return cls(
            project_dir=project_dir,
            version=data.get("version", STATE_VERSION),
            project_name=data.get("project_name", project_dir.name),
            initialized_at=data.get("initialized_at", _now_iso()),
            current_phase=data.get("current_phase", "research"),
            current_branch=data.get("current_branch"),
            best_run_id=data.get("best_run_id"),
            best_metric_value=data.get("best_metric_value"),
            best_metric_name=data.get("best_metric_name", "reward"),
            total_runs=data.get("total_runs", 0),
            kept_runs=data.get("kept_runs", 0),
            discarded_runs=data.get("discarded_runs", 0),
            crashed_runs=data.get("crashed_runs", 0),
            last_updated=data.get("last_updated", _now_iso()),
            active_workers=data.get("active_workers", []),
            queue=data.get("queue", []),
            flags=data.get("flags", {}),
        )

    def save(self) -> None:
        """Write state to disk atomically using a temp file + rename."""
        self._config_dir.mkdir(parents=True, exist_ok=True)
        self.last_updated = _now_iso()

        payload = json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

        fd, tmp_path = tempfile.mkstemp(
            dir=self._config_dir, prefix=".state_tmp_", suffix=".json"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
            os.replace(tmp_path, self._state_path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def update_best(
        self,
        run_id: str,
        metric_value: float,
        metric_name: str,
    ) -> bool:
        """
        Update best model if metric_value improves on the current best.

        Returns True if the best was updated, False otherwise.
        Higher is assumed to be better (reward maximisation).
        """
        if self.best_metric_value is None or metric_value > self.best_metric_value:
            self.best_run_id = run_id
            self.best_metric_value = metric_value
            self.best_metric_name = metric_name
            return True
        return False

    def add_to_queue(self, experiment: dict) -> None:
        """Append an experiment dict to the pending queue."""
        if not isinstance(experiment, dict):
            raise TypeError("experiment must be a dict")
        self.queue.append(experiment)

    def pop_queue(self) -> Optional[dict]:
        """Remove and return the next experiment from the queue, or None if empty."""
        if not self.queue:
            return None
        return self.queue.pop(0)

    def set_phase(self, phase: str) -> None:
        """Update the current research phase."""
        if phase not in VALID_PHASES:
            raise ValueError(
                f"Invalid phase {phase!r}. Must be one of: {VALID_PHASES}"
            )
        self.current_phase = phase

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "project_name": self.project_name,
            "initialized_at": self.initialized_at,
            "current_phase": self.current_phase,
            "current_branch": self.current_branch,
            "best_run_id": self.best_run_id,
            "best_metric_value": self.best_metric_value,
            "best_metric_name": self.best_metric_name,
            "total_runs": self.total_runs,
            "kept_runs": self.kept_runs,
            "discarded_runs": self.discarded_runs,
            "crashed_runs": self.crashed_runs,
            "last_updated": self.last_updated,
            "active_workers": self.active_workers,
            "queue": self.queue,
            "flags": self.flags,
        }

    def __repr__(self) -> str:
        return (
            f"ProjectState(project={self.project_name!r}, "
            f"phase={self.current_phase!r}, "
            f"total_runs={self.total_runs}, "
            f"best={self.best_metric_value})"
        )
