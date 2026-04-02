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
import shutil
import signal
import sys
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from drl_autoresearch.cli import console
from drl_autoresearch.core.orchestrator import Orchestrator
from drl_autoresearch.core.state import ProjectState, VALID_PHASES
from drl_autoresearch.logging.journal import ProjectJournal


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
_WORKFLOW_METADATA = ".drl_autoresearch/skill_pack.json"
_BUILD_PLAN_DIR = "implementation_plan"
_PLAN_FILE = ".drl_autoresearch/plan.json"
_REFRESH_COOLDOWN_RUNS = 3


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

    timestamp = datetime.now(timezone.utc).isoformat()

    if not path.exists():
        with path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh, delimiter="\t")
            writer.writerow(_REGISTRY_COLUMNS)
            writer.writerow([
                run_id,
                timestamp,
                hypothesis,
                json.dumps(params),
                metric_name,
                "" if metric_value is None else metric_value,
                status,
                notes,
            ])
        return

    header_line = path.read_text(encoding="utf-8").splitlines()[0]
    header = header_line.split("\t")

    row = {col: "" for col in header}
    row["run_id"] = run_id
    row["timestamp"] = timestamp
    row["hypothesis"] = hypothesis
    row["params_json"] = json.dumps(params)
    row["metric_name"] = metric_name
    row["metric_value"] = "" if metric_value is None else str(metric_value)
    row["status"] = status
    row["notes"] = notes

    # Compatible mapping for the original 27-column DRL registry schema.
    if "custom_metric_name" in row:
        row["custom_metric_name"] = metric_name
    if "custom_metric_value" in row:
        row["custom_metric_value"] = "" if metric_value is None else str(metric_value)
    if "eval_reward_mean" in row and metric_name == "reward":
        row["eval_reward_mean"] = "" if metric_value is None else str(metric_value)
    if "seed_count" in row:
        row["seed_count"] = row["seed_count"] or "1"

    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=header, delimiter="\t", extrasaction="ignore"
        )
        writer.writerow(row)


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


def _normalize_orchestrator_experiment(
    exp: Dict[str, Any],
    state: ProjectState,
) -> Dict[str, Any]:
    """Adapt orchestrator experiment schema to the run executor schema."""
    return {
        "run_id": exp.get("run_id", str(uuid.uuid4())),
        "hypothesis": exp.get("hypothesis", "orchestrator_generated"),
        "params": exp.get("params", {}),
        "metric_name": exp.get("metric_name", state.best_metric_name or "reward"),
        "skill": exp.get("skill"),
    }


def _load_project_mode(project_dir: Path, state: ProjectState) -> str:
    mode = state.flags.get("project_mode")
    if mode in ("build", "improve"):
        return mode

    meta_path = project_dir / _WORKFLOW_METADATA
    if meta_path.exists():
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            mode = data.get("project_mode")
            if mode in ("build", "improve"):
                return mode
        except (OSError, json.JSONDecodeError):
            pass

    return "improve"


def _load_onboarding_context(project_dir: Path) -> Dict[str, Any]:
    config_dir = project_dir / ".drl_autoresearch"
    yaml_path = config_dir / "onboarding.yaml"
    json_path = config_dir / "onboarding.json"

    if yaml_path.exists():
        try:
            import yaml  # type: ignore

            data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:  # noqa: BLE001
            pass

    if json_path.exists():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except (OSError, json.JSONDecodeError):
            pass

    return {}


def _write_compact_build_plan_folder(project_dir: Path, state: ProjectState) -> Path:
    plan_dir = project_dir / _BUILD_PLAN_DIR
    plan_dir.mkdir(parents=True, exist_ok=True)

    ob = _load_onboarding_context(project_dir)
    project = ob.get("project", {}) if isinstance(ob.get("project", {}), dict) else {}
    hard_rules = ob.get("hard_rules", [])
    if not isinstance(hard_rules, list):
        hard_rules = []
    rules = [str(r) for r in hard_rules if str(r).strip() and str(r).strip().lower() != "none"]
    rules_block = "\n".join(f"- {r}" for r in rules[:6]) or "- Follow NON_NEGOTIABLE_RULES.md strictly."

    env = project.get("env") or "unspecified"
    objective = project.get("objective") or "maximize task success metric"
    success_metric = project.get("success_metric") or state.best_metric_name or "reward"

    files = {
        "01_research.md": (
            "# Step 1 — Research Framing\n\n"
            f"- Environment/domain: {env}\n"
            f"- Objective: {objective}\n"
            f"- Success metric: {success_metric}\n"
            "- Run deep research to identify 3-5 promising design directions.\n"
            "- Keep only high-signal findings linked to expected training impact.\n"
            "- Reject ideas that violate hard rules or budget constraints.\n"
        ),
        "02_system_design.md": (
            "# Step 2 — System Design\n\n"
            "- Choose a compact baseline architecture and algorithm family.\n"
            "- Define reward shaping policy with anti-hacking checks.\n"
            "- Define feature/observation design with minimal complexity.\n"
            "- Keep changes incremental and testable.\n"
        ),
        "03_training_plan.md": (
            "# Step 3 — Build and Training Plan\n\n"
            "- Build minimal runnable training pipeline first.\n"
            "- Validate one clean baseline run before wider sweeps.\n"
            "- Use short probe runs for risky hypotheses.\n"
            "- Promote only improvements measured by eval metric.\n"
        ),
        "04_rules_and_risks.md": (
            "# Step 4 — Rules and Risk Controls\n\n"
            "Hard rules to enforce:\n"
            f"{rules_block}\n\n"
            "- Any redesign must pass policy checks before execution.\n"
            "- If stuck for long, trigger research refresh and adjust plan.\n"
            "- Keep documentation compact to avoid token waste.\n"
        ),
    }

    for filename, content in files.items():
        path = plan_dir / filename
        if not path.exists():
            path.write_text(content, encoding="utf-8")

    return plan_dir


def _trigger_research_and_plan_refresh(project_dir: Path, reason: str, dry_run: bool) -> bool:
    if dry_run:
        console(
            f"Dry-run: would trigger deep research + plan refresh ({reason}).",
            "info",
        )
        return False

    from drl_autoresearch.core import plan as plan_mod
    from drl_autoresearch.core import research as research_mod

    plan_path = project_dir / _PLAN_FILE
    if not plan_path.exists():
        plan_mod.run(project_dir=project_dir, refresh=True)
    rc = research_mod.run(project_dir=project_dir)
    return rc == 0


def _recent_stuck_signal(project_dir: Path, window: int = 6) -> tuple[bool, str]:
    path = _registry_path(project_dir)
    if not path.exists():
        return False, ""

    rows: List[Dict[str, str]] = []
    try:
        with path.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            rows = [dict(r) for r in reader]
    except OSError:
        return False, ""

    if not rows:
        return False, ""

    recent = rows[-window:]
    failure_count = sum(1 for r in recent if r.get("status") in {"crashed", "discarded", "pending_skill"})
    if failure_count >= max(3, window // 2):
        return True, "repeated_failures"

    completed = [r for r in rows if r.get("status") == "completed" and r.get("metric_value")]
    if len(completed) < window + 1:
        return False, ""

    prev = completed[:-window]
    curr = completed[-window:]
    try:
        prev_best = max(float(r["metric_value"]) for r in prev)
        curr_best = max(float(r["metric_value"]) for r in curr)
    except (ValueError, KeyError):
        return False, ""

    if curr_best <= prev_best:
        return True, "no_progress"

    return False, ""


def _prepare_build_mode(
    project_dir: Path,
    state: ProjectState,
    dry_run: bool,
) -> None:
    if state.flags.get("build_bootstrap_complete", False):
        return
    if state.flags.get("build_bootstrap_started", False):
        return

    console(
        "Build mode active: running compact research/bootstrap workflow before training loop.",
        "info",
    )

    plan_dir = _write_compact_build_plan_folder(project_dir, state)
    refreshed = _trigger_research_and_plan_refresh(
        project_dir=project_dir,
        reason="build_mode_bootstrap",
        dry_run=dry_run,
    )

    state.flags["build_bootstrap_started"] = True
    state.flags["build_bootstrap_research_applied"] = bool(refreshed)
    if dry_run:
        console(
            f"Dry-run: compact build plan folder ready at `{plan_dir}`.",
            "info",
        )
        return

    journal = ProjectJournal(project_dir)
    journal.initialize(project_name=state.project_name, spec={})
    journal.log_event(
        "build_mode_bootstrap",
        (
            f"Build mode bootstrap prepared in `{_BUILD_PLAN_DIR}/`.\n"
            "Deep research and plan refresh completed before training loop."
        ),
        metadata={"research_refreshed": refreshed, "plan_folder": _BUILD_PLAN_DIR},
    )


def _maybe_refresh_when_stuck(
    project_dir: Path,
    state: ProjectState,
    orchestrator: Optional[Orchestrator],
    dry_run: bool,
) -> None:
    stuck, reason = (False, "")
    if orchestrator is not None:
        try:
            stuck, reason = orchestrator.should_trigger_research_refresh()
        except Exception:  # noqa: BLE001
            stuck, reason = (False, "")
    if not stuck:
        stuck, reason = _recent_stuck_signal(project_dir)
    if not stuck:
        return

    last_refresh_runs = int(state.flags.get("last_refresh_total_runs", -10_000))
    if (state.total_runs - last_refresh_runs) < _REFRESH_COOLDOWN_RUNS:
        console(
            "Stuck signal detected but refresh cooldown is active; skipping refresh this run.",
            "info",
        )
        return

    console(
        f"Stuck signal detected ({reason}); triggering research + redesign refresh.",
        "warning",
    )
    refreshed = _trigger_research_and_plan_refresh(
        project_dir=project_dir,
        reason=reason,
        dry_run=dry_run,
    )
    if dry_run:
        return

    state.flags["last_refresh_reason"] = reason
    state.flags["last_refresh_total_runs"] = state.total_runs
    journal = ProjectJournal(project_dir)
    journal.initialize(project_name=state.project_name, spec={})
    journal.log_event(
        "stuck_refresh",
        "No meaningful progress detected. Triggered compact research/plan refresh.",
        metadata={"reason": reason, "refresh_applied": refreshed},
    )


def _finalize_build_mode_if_complete(
    project_dir: Path,
    state: ProjectState,
    results: List[Dict[str, Any]],
) -> None:
    if state.flags.get("project_mode") != "build":
        return
    if state.flags.get("build_bootstrap_complete", False):
        return

    has_completed = any(r.get("status") == "completed" for r in results)
    if not has_completed:
        return

    plan_dir = project_dir / _BUILD_PLAN_DIR
    if plan_dir.exists():
        shutil.rmtree(plan_dir)
        console(f"Build plan folder removed: {_BUILD_PLAN_DIR}/", "info")

    overview = (
        f"Build bootstrap completed. Project transitioned to normal training loop.\n"
        f"Phase: {state.current_phase}\n"
        f"Best metric: {state.best_metric_name}={state.best_metric_value}\n"
        "Implementation details are now tracked through logs and registry."
    )

    journal = ProjectJournal(project_dir)
    journal.initialize(project_name=state.project_name, spec={})
    journal.log_event(
        "build_mode_completed",
        overview,
        metadata={"plan_folder_deleted": True},
    )

    state.flags["build_bootstrap_complete"] = True


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
    orchestrator: Optional[Orchestrator] = None
    try:
        orchestrator = Orchestrator(project_dir)
        orchestrator.load()
    except Exception as exc:  # noqa: BLE001
        console(f"Orchestrator unavailable, using local fallback: {exc}", "warning")

    project_mode = _load_project_mode(project_dir, state)
    state.flags["project_mode"] = project_mode
    console(
        f"Project: {state.project_name}  |  Phase: {state.current_phase}  |  Mode: {project_mode}",
        "info",
    )

    if project_mode == "build":
        _prepare_build_mode(project_dir=project_dir, state=state, dry_run=dry_run)

    _maybe_refresh_when_stuck(
        project_dir=project_dir,
        state=state,
        orchestrator=orchestrator,
        dry_run=dry_run,
    )

    if state.current_phase == "converged":
        console("Research loop has converged. No further experiments to run.", "info")
        return 0

    # Populate queue from state or generate baseline.
    experiments = list(state.queue)
    if not experiments:
        if orchestrator is not None:
            next_exp = orchestrator.decide_next_experiment()
            if next_exp is not None:
                experiments = [_normalize_orchestrator_experiment(next_exp, state)]
                console("Queue is empty — using orchestrator-selected experiment.", "info")
        if not experiments:
            console("Queue is empty — generating baseline experiments (fallback).", "info")
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

    _finalize_build_mode_if_complete(project_dir=project_dir, state=state, results=results)

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
