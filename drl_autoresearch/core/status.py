"""
drl_autoresearch.core.status
-----------------------------
Print a human-readable project status summary.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List, Optional

from drl_autoresearch.cli import console
from drl_autoresearch.core.state import ProjectState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_last_n_registry_rows(
    project_dir: Path, n: int = 5
) -> List[dict]:
    """Return the last *n* rows from the experiment registry as dicts."""
    path = project_dir / "logs" / "experiment_registry.tsv"
    if not path.exists():
        return []
    rows: List[dict] = []
    try:
        with path.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                rows.append(dict(row))
    except OSError:
        return []
    return rows[-n:]


def _phase_bar(phase: str) -> str:
    """Return a tiny ASCII progress indicator for the current phase."""
    phases = [
        "research",
        "baseline",
        "experimenting",
        "focused_tuning",
        "ablation",
        "converged",
    ]
    try:
        idx = phases.index(phase)
    except ValueError:
        idx = 0
    filled = "█" * (idx + 1)
    empty  = "░" * (len(phases) - idx - 1)
    return f"[{filled}{empty}] {phase}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(project_dir: Path) -> int:
    """Print the project status summary.

    Returns 0 always (status is informational).
    """
    project_dir = Path(project_dir).resolve()

    config_dir = project_dir / ".drl_autoresearch"
    if not config_dir.is_dir():
        console(
            "Project not initialised. Run `drl-autoresearch init` first.", "error"
        )
        return 1

    state = ProjectState.load(project_dir)

    # Header.
    print()
    print("=" * 60)
    print(f"  DRL AutoResearch — Project Status")
    print(f"  {project_dir}")
    print("=" * 60)

    # Phase.
    print(f"  Phase       : {_phase_bar(state.current_phase)}")

    # Run counts.
    print(f"  Total runs  : {state.total_runs}")
    print(f"  Kept        : {state.kept_runs}")
    print(f"  Discarded   : {state.discarded_runs}")
    print(f"  Crashed     : {state.crashed_runs}")

    # Best model.
    if state.best_run_id:
        print(
            f"  Best run    : {state.best_run_id[:8]}  "
            f"{state.best_metric_name}={state.best_metric_value:.4f}"
        )
    else:
        print("  Best run    : (none yet)")

    # Queue depth.
    print(f"  Queue depth : {len(state.queue)}")

    # Active workers.
    if state.active_workers:
        print(f"  Workers     : {', '.join(state.active_workers)}")
    else:
        print("  Workers     : (none active)")

    # Last updated.
    print(f"  Last updated: {state.last_updated}")

    # Flags.
    if state.flags:
        print(f"  Flags       : {json.dumps(state.flags)}")

    print()

    # Recent experiments.
    recent = _read_last_n_registry_rows(project_dir, n=5)
    if recent:
        print("  Recent experiments:")
        for row in recent:
            run_id      = row.get("run_id", "?")[:8]
            hypothesis  = row.get("hypothesis", "")[:40]
            metric      = row.get("metric_value", "")
            status      = row.get("status", "")
            ts          = row.get("timestamp", "")[:19]
            metric_str  = f"{float(metric):.4f}" if metric else "N/A"
            print(f"    [{run_id}] {ts}  {status:<14} {metric_str}  {hypothesis}")
    else:
        print("  No experiments recorded yet.")

    # Next action hint.
    print()
    _print_next_hint(state)
    print()

    return 0


def _print_next_hint(state: ProjectState) -> None:
    """Print a contextual 'what to do next' suggestion."""
    if state.current_phase == "research":
        console("Tip: run `drl-autoresearch run` to start the baseline loop.", "info")
    elif state.current_phase == "baseline":
        console("Tip: run `drl-autoresearch plan` to review the current plan.", "info")
    elif state.current_phase in ("experimenting", "focused_tuning", "ablation"):
        console("Tip: run `drl-autoresearch run` to continue experimenting.", "info")
    elif state.current_phase == "converged":
        console(
            "Research has converged. Review results in logs/experiment_registry.tsv.",
            "success",
        )
    else:
        console("Run `drl-autoresearch doctor` to check for issues.", "info")
