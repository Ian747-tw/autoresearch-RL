"""
drl_autoresearch.core.run
--------------------------
Start the autonomous experiment loop.

Responsibilities
----------------
1. Load project state and validate config.
2. Pop experiments from the queue (or generate a baseline if queue is empty).
3. Dispatch experiments (sequentially or in parallel via multiprocessing).
4. Record results to experiment_registry.tsv and update state.json.
5. Advance the research phase when phase-transition criteria are met.
"""

from __future__ import annotations

import concurrent.futures
import csv
import json
import os
import signal
import sys
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from drl_autoresearch.cli import console
from drl_autoresearch.core.state import ProjectState, VALID_PHASES


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------

_REGISTRY_COLUMNS = [
    "run_id",
    "timestamp",
    "hypothesis",
    "params_json",
    "metric_name",
    "metric_value",
    "status",
    "notes",
]


def _registry_path(project_dir: Path) -> Path:
    return project_dir / "logs" / "experiment_registry.tsv"


def _append_registry(
    project_dir: Path,
    run_id: str,
    hypothesis: str,
    params: dict,
    metric_name: str,
    metric_value: Optional[float],
    status: str,
    notes: str = "",
) -> None:
    """Append one row to the experiment registry TSV."""
    path = _registry_path(project_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, delimiter="\t")
        if write_header:
            writer.writerow(_REGISTRY_COLUMNS)
        writer.writerow([
            run_id,
            datetime.now(timezone.utc).isoformat(),
            hypothesis,
            json.dumps(params),
            metric_name,
            "" if metric_value is None else metric_value,
            status,
            notes,
        ])


# ---------------------------------------------------------------------------
# Experiment execution stub
# ---------------------------------------------------------------------------

def _execute_experiment(experiment: Dict[str, Any], project_dir: Path) -> Dict[str, Any]:
    """
    Execute a single experiment dict and return a result dict.

    In a real deployment this would launch a training subprocess or call a
    skill script found in ``skills/``.  Here we provide a robust skeleton
    that:

    * Looks for a ``skill`` key in the experiment dict and tries to run
      ``<project_dir>/skills/<skill>.py`` via a subprocess.
    * Falls back to a no-op if no skill is found, logging a warning.

    The returned dict always contains:
        run_id, metric_name, metric_value, status, notes
    """
    import subprocess  # stdlib, deferred import

    run_id      = experiment.get("run_id", str(uuid.uuid4()))
    hypothesis  = experiment.get("hypothesis", "unnamed")
    params      = experiment.get("params", {})
    metric_name = experiment.get("metric_name", "reward")
    skill_name  = experiment.get("skill", None)

    skill_path = (project_dir / "skills" / f"{skill_name}.py") if skill_name else None

    if skill_path and skill_path.exists():
        try:
            env = {**os.environ, "DRL_EXPERIMENT_PARAMS": json.dumps(params)}
            completed = subprocess.run(
                [sys.executable, str(skill_path)],
                capture_output=True,
                text=True,
                timeout=experiment.get("timeout_seconds", 3600),
                env=env,
            )
            if completed.returncode == 0:
                # Expect skill to print JSON result on last line of stdout.
                lines = [l for l in completed.stdout.strip().splitlines() if l.strip()]
                result_json: dict = {}
                if lines:
                    try:
                        result_json = json.loads(lines[-1])
                    except json.JSONDecodeError:
                        pass
                metric_value = float(result_json.get(metric_name, 0.0))
                return {
                    "run_id": run_id,
                    "hypothesis": hypothesis,
                    "params": params,
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                    "status": "completed",
                    "notes": result_json.get("notes", ""),
                }
            else:
                return {
                    "run_id": run_id,
                    "hypothesis": hypothesis,
                    "params": params,
                    "metric_name": metric_name,
                    "metric_value": None,
                    "status": "crashed",
                    "notes": completed.stderr[:500],
                }
        except subprocess.TimeoutExpired:
            return {
                "run_id": run_id,
                "hypothesis": hypothesis,
                "params": params,
                "metric_name": metric_name,
                "metric_value": None,
                "status": "crashed",
                "notes": "Timed out.",
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "run_id": run_id,
                "hypothesis": hypothesis,
                "params": params,
                "metric_name": metric_name,
                "metric_value": None,
                "status": "crashed",
                "notes": str(exc),
            }
    else:
        # No skill — produce a placeholder result so the loop progresses.
        console(
            f"  No skill script found for experiment {run_id!r}. "
            "Recording as 'pending_skill'.",
            "warning",
        )
        return {
            "run_id": run_id,
            "hypothesis": hypothesis,
            "params": params,
            "metric_name": metric_name,
            "metric_value": None,
            "status": "pending_skill",
            "notes": f"Skill '{skill_name}' not found in skills/.",
        }


# ---------------------------------------------------------------------------
# Phase advancement logic
# ---------------------------------------------------------------------------

def _maybe_advance_phase(state: ProjectState) -> bool:
    """Check phase-transition criteria and advance if met.  Returns True if advanced."""
    current_idx = VALID_PHASES.index(state.current_phase) if state.current_phase in VALID_PHASES else 0

    if state.current_phase == "research" and state.total_runs >= 1:
        state.set_phase("baseline")
        return True
    if state.current_phase == "baseline" and state.kept_runs >= 3:
        state.set_phase("experimenting")
        return True
    if state.current_phase == "experimenting" and state.kept_runs >= 10:
        state.set_phase("focused_tuning")
        return True
    if state.current_phase == "focused_tuning" and state.kept_runs >= 20:
        state.set_phase("ablation")
        return True
    if state.current_phase == "ablation" and state.kept_runs >= 30:
        state.set_phase("converged")
        return True
    return False


# ---------------------------------------------------------------------------
# Baseline experiment generator
# ---------------------------------------------------------------------------

def _generate_baseline_experiments() -> List[Dict[str, Any]]:
    """Return a minimal set of baseline experiments when the queue is empty."""
    return [
        {
            "run_id": str(uuid.uuid4()),
            "hypothesis": "baseline_default_hyperparams",
            "params": {
                "learning_rate": 3e-4,
                "batch_size": 64,
                "gamma": 0.99,
                "n_steps": 2048,
            },
            "metric_name": "reward",
            "skill": None,
        }
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(
    project_dir: Path,
    parallel: int = 1,
    dry_run: bool = False,
) -> int:
    """Execute the autonomous experiment loop.

    Parameters
    ----------
    project_dir:
        Root of the target project.
    parallel:
        Maximum number of concurrent experiments.
    dry_run:
        If True, validate and print what would run without executing.
    """
    project_dir = Path(project_dir).resolve()

    # Validate project is initialised.
    config_dir = project_dir / ".drl_autoresearch"
    if not config_dir.is_dir():
        console(
            "Project not initialised. Run `drl-autoresearch init` first.", "error"
        )
        return 1

    state = ProjectState.load(project_dir)
    console(f"Project: {state.project_name}  |  Phase: {state.current_phase}", "info")

    if state.current_phase == "converged":
        console("Research loop has converged. No further experiments to run.", "info")
        return 0

    # Populate queue from state or generate baseline.
    experiments = list(state.queue)
    if not experiments:
        console("Queue is empty — generating baseline experiments.", "info")
        experiments = _generate_baseline_experiments()

    console(
        f"Experiments queued: {len(experiments)}  |  Parallel workers: {parallel}",
        "info",
    )

    if dry_run:
        console("Dry-run mode — no experiments will be executed.", "warning")
        for exp in experiments:
            print(
                f"  [{exp.get('run_id', '?')[:8]}] "
                f"{exp.get('hypothesis', 'unnamed')} "
                f"params={json.dumps(exp.get('params', {}))}"
            )
        return 0

    # Clear queue before running (will be repopulated on next plan refresh).
    state.queue.clear()

    # -----------------------------------------------------------------------
    # Execute experiments (parallel or sequential).
    # -----------------------------------------------------------------------
    _shutdown_requested = False

    def _handle_sigint(sig, frame):  # noqa: ANN001
        nonlocal _shutdown_requested
        _shutdown_requested = True
        console("Shutdown requested — finishing current experiments...", "warning")

    old_handler = signal.signal(signal.SIGINT, _handle_sigint)

    try:
        if parallel > 1:
            results = _run_parallel(experiments, project_dir, parallel)
        else:
            results = _run_sequential(experiments, project_dir)
    finally:
        signal.signal(signal.SIGINT, old_handler)

    # -----------------------------------------------------------------------
    # Record results and update state.
    # -----------------------------------------------------------------------
    for result in results:
        run_id       = result["run_id"]
        metric_name  = result["metric_name"]
        metric_value = result.get("metric_value")
        status       = result["status"]
        hypothesis   = result.get("hypothesis", "")
        params       = result.get("params", {})
        notes        = result.get("notes", "")

        state.total_runs += 1
        if status == "completed":
            state.kept_runs += 1
            if metric_value is not None:
                improved = state.update_best(run_id, metric_value, metric_name)
                if improved:
                    console(
                        f"New best: {metric_name}={metric_value:.4f}  (run {run_id[:8]})",
                        "success",
                    )
        elif status == "crashed":
            state.crashed_runs += 1
        else:
            state.discarded_runs += 1

        _append_registry(
            project_dir,
            run_id=run_id,
            hypothesis=hypothesis,
            params=params,
            metric_name=metric_name,
            metric_value=metric_value,
            status=status,
            notes=notes,
        )

        level = "success" if status == "completed" else "warning" if status != "crashed" else "error"
        console(
            f"Run {run_id[:8]}: status={status}  "
            f"metric={metric_value if metric_value is not None else 'N/A'}",
            level,
        )

    # Phase advancement.
    advanced = _maybe_advance_phase(state)
    if advanced:
        console(f"Phase advanced to: {state.current_phase}", "info")

    state.save()
    console(
        f"Loop complete. Total runs: {state.total_runs}  "
        f"Best {state.best_metric_name}: "
        f"{state.best_metric_value if state.best_metric_value is not None else 'N/A'}",
        "success",
    )
    return 0


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------

def _run_sequential(
    experiments: List[Dict[str, Any]], project_dir: Path
) -> List[Dict[str, Any]]:
    results = []
    for exp in experiments:
        console(f"  Running {exp.get('run_id', '?')[:8]} — {exp.get('hypothesis', '')}", "info")
        try:
            result = _execute_experiment(exp, project_dir)
        except Exception as exc:  # noqa: BLE001
            result = {
                "run_id": exp.get("run_id", str(uuid.uuid4())),
                "hypothesis": exp.get("hypothesis", ""),
                "params": exp.get("params", {}),
                "metric_name": exp.get("metric_name", "reward"),
                "metric_value": None,
                "status": "crashed",
                "notes": traceback.format_exc()[:500],
            }
        results.append(result)
    return results


def _run_parallel(
    experiments: List[Dict[str, Any]], project_dir: Path, n_workers: int
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = [None] * len(experiments)  # type: ignore[list-item]
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_idx = {
            executor.submit(_execute_experiment, exp, project_dir): i
            for i, exp in enumerate(experiments)
        }
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as exc:  # noqa: BLE001
                exp = experiments[idx]
                results[idx] = {
                    "run_id": exp.get("run_id", str(uuid.uuid4())),
                    "hypothesis": exp.get("hypothesis", ""),
                    "params": exp.get("params", {}),
                    "metric_name": exp.get("metric_name", "reward"),
                    "metric_value": None,
                    "status": "crashed",
                    "notes": str(exc),
                }
    return results
