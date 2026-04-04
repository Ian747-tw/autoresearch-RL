"""
Experiment registry — central truth of what experiments ran and what happened.
All writes are atomic (tmp-then-rename or locked-append).
"""

from __future__ import annotations

import os
import random
import string
import time
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# File-locking helpers (Linux fcntl with fallback)
# ---------------------------------------------------------------------------

try:
    import fcntl

    def _lock_file(fh):
        deadline = time.monotonic() + 10.0
        while True:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return
            except BlockingIOError:
                if time.monotonic() > deadline:
                    raise TimeoutError("Could not acquire file lock within 10 s")
                time.sleep(0.05)

    def _unlock_file(fh):
        fcntl.flock(fh, fcntl.LOCK_UN)

except ImportError:  # Windows or other platforms
    def _lock_file(fh):  # noqa: F811
        pass

    def _unlock_file(fh):  # noqa: F811
        pass


# ---------------------------------------------------------------------------
# RunRecord dataclass
# ---------------------------------------------------------------------------

COLUMNS = [
    "run_id", "parent_run_id", "timestamp", "agent", "branch",
    "commit", "environment", "algorithm", "config_summary",
    "change_summary", "hypothesis", "rules_checked",
    "train_reward_mean", "train_reward_std", "eval_reward_mean",
    "eval_reward_std", "custom_metric_name", "custom_metric_value",
    "success_rate", "constraint_violations", "seed_count",
    "wall_clock_seconds", "gpu_memory_gb", "ram_gb",
    "status", "keep_decision", "notes",
]

_NUMERIC_FIELDS = {
    "train_reward_mean", "train_reward_std", "eval_reward_mean",
    "eval_reward_std", "custom_metric_value", "success_rate",
    "constraint_violations", "seed_count", "wall_clock_seconds",
    "gpu_memory_gb", "ram_gb",
}


def _rand4() -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=4))


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _make_run_id() -> str:
    return f"run_{_now_iso()}_{_rand4()}"


def _classify_run_outcome(status: str, keep_decision: str = "") -> str:
    normalized = str(status or "").strip().lower()
    if normalized == "completed":
        return "keep" if str(keep_decision or "").strip().lower() == "keep" else "discard"
    if normalized == "crashed":
        return "crash"
    return "discard"


def _escape_tsv(value: str) -> str:
    """Escape tabs and newlines so a field never breaks the TSV structure."""
    return str(value).replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n")


def _unescape_tsv(value: str) -> str:
    return value.replace("\\n", "\n").replace("\\t", "\t").replace("\\\\", "\\")


@dataclass
class RunRecord:
    # identifiers
    run_id: str = field(default_factory=_make_run_id)
    parent_run_id: str = ""
    timestamp: str = field(default_factory=_now_iso)
    agent: str = ""
    branch: str = ""
    commit: str = ""
    environment: str = ""
    algorithm: str = ""
    # human-readable summaries
    config_summary: str = ""
    change_summary: str = ""
    hypothesis: str = ""
    rules_checked: str = ""
    # numeric metrics
    train_reward_mean: Optional[float] = None
    train_reward_std: Optional[float] = None
    eval_reward_mean: Optional[float] = None
    eval_reward_std: Optional[float] = None
    custom_metric_name: str = ""
    custom_metric_value: Optional[float] = None
    success_rate: Optional[float] = None
    constraint_violations: Optional[float] = None
    seed_count: Optional[float] = None
    wall_clock_seconds: Optional[float] = None
    gpu_memory_gb: Optional[float] = None
    ram_gb: Optional[float] = None
    # outcome
    status: str = ""          # "completed" | "crashed" | "running" | …
    keep_decision: str = ""   # "keep" | "discard"
    notes: str = ""

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_tsv_row(self) -> str:
        parts = []
        for col in COLUMNS:
            val = getattr(self, col)
            if val is None:
                parts.append("")
            else:
                parts.append(_escape_tsv(str(val)))
        return "\t".join(parts)

    @classmethod
    def from_tsv_row(cls, row: str) -> "RunRecord":
        raw = row.rstrip("\n").split("\t")
        # Pad if the file has fewer columns than the current schema
        raw += [""] * (len(COLUMNS) - len(raw))
        kwargs: dict = {}
        for i, col in enumerate(COLUMNS):
            raw_val = _unescape_tsv(raw[i]) if i < len(raw) else ""
            if col in _NUMERIC_FIELDS:
                kwargs[col] = float(raw_val) if raw_val.strip() != "" else None
            else:
                kwargs[col] = raw_val
        return cls(**kwargs)

    def get_metric(self, metric: str) -> Optional[float]:
        val = getattr(self, metric, None)
        return float(val) if val is not None and str(val).strip() != "" else None


# ---------------------------------------------------------------------------
# ExperimentRegistry
# ---------------------------------------------------------------------------

class ExperimentRegistry:
    TSV_PATH = "logs/experiment_registry.tsv"
    COLUMNS = COLUMNS

    def __init__(self, project_dir: Path) -> None:
        self.project_dir = Path(project_dir)
        self.tsv_path = self.project_dir / self.TSV_PATH

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Create the TSV with a header row if it does not already exist."""
        self.tsv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.tsv_path.exists():
            self._atomic_write_header()

    def _atomic_write_header(self) -> None:
        tmp = self.tsv_path.with_suffix(".tsv.tmp")
        tmp.write_text("\t".join(COLUMNS) + "\n", encoding="utf-8")
        tmp.replace(self.tsv_path)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_run(self, run: RunRecord) -> None:
        """Append a run record atomically using file locking."""
        self.tsv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.tsv_path.exists():
            self._atomic_write_header()

        row_line = run.to_tsv_row() + "\n"
        retries = 5
        for attempt in range(retries):
            try:
                with open(self.tsv_path, "a", encoding="utf-8") as fh:
                    _lock_file(fh)
                    try:
                        fh.write(row_line)
                    finally:
                        _unlock_file(fh)
                try:
                    from drl_autoresearch.core.agent_contract import audit_event

                    audit_event(
                        "registry_add",
                        {
                            "run_id": run.run_id,
                            "status": run.status,
                            "keep_decision": run.keep_decision,
                        },
                    )
                except Exception:
                    pass
                return
            except (OSError, TimeoutError):
                if attempt == retries - 1:
                    raise
                time.sleep(0.1 * (attempt + 1))

    def update_run(self, run_id: str, updates: dict) -> None:
        """
        Rewrite the TSV with updated fields for the given run_id.
        Uses atomic write: write to .tmp then rename.
        """
        runs = self.get_history()
        found = False
        for run in runs:
            if run.run_id == run_id:
                for key, value in updates.items():
                    if hasattr(run, key):
                        setattr(run, key, value)
                found = True
                break
        if not found:
            raise KeyError(f"run_id {run_id!r} not found in registry")

        tmp = self.tsv_path.with_suffix(".tsv.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write("\t".join(COLUMNS) + "\n")
            for run in runs:
                fh.write(run.to_tsv_row() + "\n")
        tmp.replace(self.tsv_path)
        try:
            from drl_autoresearch.core.agent_contract import audit_event

            audit_event(
                "registry_update",
                {
                    "run_id": run_id,
                    "updated_fields": sorted(list(updates.keys())),
                },
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    def _read_rows(self) -> list[RunRecord]:
        if not self.tsv_path.exists():
            return []
        with open(self.tsv_path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
        records = []
        for line in lines[1:]:  # skip header
            line = line.strip()
            if line:
                try:
                    records.append(RunRecord.from_tsv_row(line))
                except Exception:
                    pass  # skip malformed rows
        return records

    def get_history(self) -> list[RunRecord]:
        """All runs in insertion order."""
        return self._read_rows()

    def get_run(self, run_id: str) -> Optional[RunRecord]:
        for run in self._read_rows():
            if run.run_id == run_id:
                return run
        return None

    def get_best(
        self,
        metric: str = "eval_reward_mean",
        higher_is_better: bool = True,
    ) -> Optional[RunRecord]:
        candidates = [r for r in self._read_rows() if r.get_metric(metric) is not None]
        if not candidates:
            return None
        return max(candidates, key=lambda r: r.get_metric(metric)) if higher_is_better \
            else min(candidates, key=lambda r: r.get_metric(metric))

    def get_recent(self, n: int = 10) -> list[RunRecord]:
        runs = self._read_rows()
        return runs[-n:] if len(runs) >= n else runs

    def get_kept(self) -> list[RunRecord]:
        return [
            r
            for r in self._read_rows()
            if _classify_run_outcome(r.status, r.keep_decision) == "keep"
        ]

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def get_plateau_detection(
        self, window: int = 5
    ) -> tuple[bool, str]:
        """
        Returns (is_plateau, reason).
        A plateau is: in the last `window` kept runs, no improvement > 0.5%
        in eval_reward_mean.
        """
        kept = self.get_kept()
        if len(kept) < window:
            return False, f"Insufficient kept runs ({len(kept)} < {window})"

        recent_kept = kept[-window:]
        values = [r.get_metric("eval_reward_mean") for r in recent_kept]
        values = [v for v in values if v is not None]

        if len(values) < 2:
            return False, "Insufficient metric data in recent kept runs"

        best_in_window = max(values)
        first_in_window = values[0]
        if first_in_window == 0.0:
            return False, "Cannot compute percentage improvement from zero baseline"

        improvement_pct = (best_in_window - first_in_window) / abs(first_in_window) * 100.0
        if improvement_pct <= 0.5:
            return True, (
                f"No improvement > 0.5% over last {window} kept runs "
                f"(best={best_in_window:.4f}, first={first_in_window:.4f}, "
                f"delta={improvement_pct:.2f}%)"
            )
        return False, (
            f"Still improving: {improvement_pct:.2f}% over last {window} kept runs"
        )

    def get_failure_patterns(self) -> dict:
        """Return a dict mapping failure pattern name to count."""
        runs = self._read_rows()
        patterns: dict[str, int] = {}
        for run in runs:
            outcome = _classify_run_outcome(run.status, run.keep_decision)
            if outcome == "crash":
                key = "crash"
                patterns[key] = patterns.get(key, 0) + 1
            if outcome == "discard":
                key = "discarded"
                patterns[key] = patterns.get(key, 0) + 1
            if run.constraint_violations is not None and run.constraint_violations > 0:
                key = "constraint_violation"
                patterns[key] = patterns.get(key, 0) + 1
        return patterns

    def summary_stats(self) -> dict:
        runs = self._read_rows()
        kept = [r for r in runs if _classify_run_outcome(r.status, r.keep_decision) == "keep"]
        crashed = [r for r in runs if _classify_run_outcome(r.status, r.keep_decision) == "crash"]
        discarded = [r for r in runs if _classify_run_outcome(r.status, r.keep_decision) == "discard"]

        eval_values = [
            r.get_metric("eval_reward_mean")
            for r in runs
            if r.get_metric("eval_reward_mean") is not None
        ]

        return {
            "total_runs": len(runs),
            "kept": len(kept),
            "discarded": len(discarded),
            "crashed": len(crashed),
            "best_eval_reward_mean": max(eval_values) if eval_values else None,
            "latest_eval_reward_mean": eval_values[-1] if eval_values else None,
        }
