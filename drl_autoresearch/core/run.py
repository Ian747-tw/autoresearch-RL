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
5. Preserve keep/discard state and best-run tracking across cycles.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import shutil
import signal
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from drl_autoresearch.cli import console
from drl_autoresearch.core.agent_contract import initialize_contract, load_contract
from drl_autoresearch.core.agent_runner import (
    autonomous_policy_allowed,
    build_agent_prompt,
    detect_backend,
    run_agent_cycle,
)
from drl_autoresearch.core.orchestrator import Orchestrator
from drl_autoresearch.core.state import ProjectState, VALID_PHASES
from drl_autoresearch.logging.incidents import IncidentLog
from drl_autoresearch.logging.journal import ProjectJournal
from drl_autoresearch.logging.registry import ExperimentRegistry, RunRecord


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
_AGENT_LOOP_SLEEP_SECONDS = 2
_RUNTIME_CONTRACTS_DIR = ".drl_autoresearch/runtime/contracts"


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
    """Append one row to the experiment registry using the registry helper."""
    registry = ExperimentRegistry(project_dir=project_dir)
    registry.initialize()
    config_summary = json.dumps(params, ensure_ascii=False)
    record = RunRecord(
        run_id=run_id,
        hypothesis=hypothesis,
        agent="controller",
        config_summary=config_summary,
        change_summary=config_summary,
        rules_checked="controller_fallback",
        eval_reward_mean=metric_value if metric_name == "reward" else None,
        custom_metric_name="" if metric_name == "reward" else metric_name,
        custom_metric_value=None if metric_name == "reward" else metric_value,
        seed_count=1,
        status=status,
        keep_decision="keep" if status == "completed" else "discard",
        notes=notes,
    )
    registry.add_run(record)


def _read_registry_rows(project_dir: Path) -> list[dict[str, Any]]:
    registry = ExperimentRegistry(project_dir=project_dir)
    rows: list[dict[str, Any]] = []
    for record in registry.get_history():
        metric_name = record.custom_metric_name or "reward"
        metric_value: Optional[float]
        if record.custom_metric_name:
            metric_value = record.custom_metric_value
        else:
            metric_value = record.eval_reward_mean
        rows.append(
            {
                "run_id": record.run_id,
                "timestamp": record.timestamp,
                "hypothesis": record.hypothesis,
                "params_json": record.config_summary,
                "metric_name": metric_name,
                "metric_value": metric_value,
                "status": record.status,
                "keep_decision": record.keep_decision,
                "notes": record.notes,
            }
        )
    return rows


def _classify_run_outcome(status: str, keep_decision: str = "") -> str:
    """Map registry status/decision to one keep/discard/crash bucket."""
    normalized = str(status or "").strip().lower()
    if normalized == "completed":
        return "keep" if str(keep_decision or "").strip().lower() == "keep" else "discard"
    if normalized == "crashed":
        return "crash"
    return "discard"


def _normalize_registry_keep_decisions(project_dir: Path) -> None:
    """Rewrite keep/discard so only improving completed runs remain kept."""
    registry = ExperimentRegistry(project_dir=project_dir)
    runs = registry.get_history()
    best_metric_name: Optional[str] = None
    best_metric_value: Optional[float] = None

    for run in runs:
        metric_name = (run.custom_metric_name or "reward").strip() or "reward"
        metric_value = (
            run.custom_metric_value
            if run.custom_metric_name
            else run.eval_reward_mean
        )

        desired_keep = "discard"
        if run.status == "completed" and metric_value is not None:
            if best_metric_value is None:
                desired_keep = "keep"
                best_metric_name = metric_name
                best_metric_value = metric_value
            elif metric_name == best_metric_name and metric_value > best_metric_value:
                desired_keep = "keep"
                best_metric_value = metric_value

        if run.keep_decision != desired_keep:
            registry.update_run(run.run_id, {"keep_decision": desired_keep})


def _sync_state_from_registry(state: ProjectState, project_dir: Path) -> list[dict[str, Any]]:
    _normalize_registry_keep_decisions(project_dir)
    rows = _read_registry_rows(project_dir)
    state.total_runs = len(rows)
    state.kept_runs = 0
    state.crashed_runs = 0
    state.discarded_runs = 0

    best_run_id: Optional[str] = None
    best_metric_name = state.best_metric_name or "reward"
    best_metric_value: Optional[float] = None

    for row in rows:
        status = str(row.get("status", "")).strip()
        outcome = _classify_run_outcome(status, str(row.get("keep_decision", "")))
        if outcome == "keep":
            state.kept_runs += 1
        elif outcome == "crash":
            state.crashed_runs += 1
        else:
            state.discarded_runs += 1

        metric_name = str(row.get("metric_name", "") or best_metric_name).strip() or best_metric_name
        raw_value = row.get("metric_value")
        try:
            metric_value = float(raw_value) if raw_value not in ("", None) else None
        except (TypeError, ValueError):
            metric_value = None
        if outcome == "keep" and metric_value is not None:
            if best_metric_value is None or metric_value > best_metric_value:
                best_metric_value = metric_value
                best_metric_name = metric_name
                best_run_id = str(row.get("run_id") or "")

    state.best_run_id = best_run_id
    state.best_metric_name = best_metric_name
    state.best_metric_value = best_metric_value
    return rows


def _set_loop_flags(
    state: ProjectState,
    *,
    running: bool,
    backend: Optional[str],
    activity: str,
    active_run_id: Optional[str] = None,
    last_exit_code: Optional[int] = None,
) -> None:
    state.flags["loop_running"] = running
    state.flags["current_activity"] = activity
    state.flags["agent_backend"] = backend
    state.flags["active_run_id"] = active_run_id
    state.flags["last_agent_exit_code"] = last_exit_code


def _contract_path(project_dir: Path, run_id: str) -> Path:
    return project_dir / _RUNTIME_CONTRACTS_DIR / f"{run_id}.json"


def _file_fingerprint(path: Path) -> tuple[bool, int, int]:
    if not path.exists():
        return (False, 0, 0)
    stat = path.stat()
    return (True, int(stat.st_mtime_ns), int(stat.st_size))


def _snapshot_project_tree(project_dir: Path) -> dict[str, tuple[int, int]]:
    excluded_prefixes = (
        ".git/",
        ".venv/",
        "__pycache__/",
        "logs/agent_sessions/",
        "logs/artifacts/",
    )
    excluded_exact = {
        ".drl_autoresearch/state.json",
        "logs/experiment_registry.tsv",
        "logs/project_journal.md",
        "logs/incidents.md",
        "logs/handoffs.md",
    }
    snapshot: dict[str, tuple[int, int]] = {}
    for path in project_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(project_dir).as_posix()
        if rel in excluded_exact or any(rel.startswith(prefix) for prefix in excluded_prefixes):
            continue
        stat = path.stat()
        snapshot[rel] = (int(stat.st_mtime_ns), int(stat.st_size))
    return snapshot


def _detect_project_tree_changes(
    before: dict[str, tuple[int, int]],
    after: dict[str, tuple[int, int]],
) -> list[str]:
    changed: list[str] = []
    all_paths = sorted(set(before) | set(after))
    for rel in all_paths:
        if before.get(rel) != after.get(rel):
            changed.append(rel)
    return changed


def _collect_skill_inventory(project_dir: Path) -> list[str]:
    skills_dir = project_dir / "skills"
    if not skills_dir.is_dir():
        return []
    return sorted(
        path.relative_to(project_dir).as_posix()
        for path in skills_dir.rglob("*")
        if path.is_file()
    )


def _validate_agent_contract(
    project_dir: Path,
    *,
    run_id: str,
    contract_path: Path,
    before_tree: dict[str, tuple[int, int]],
    before_registry: tuple[bool, int, int],
    before_journal: tuple[bool, int, int],
    before_incidents: tuple[bool, int, int],
    before_handoffs: tuple[bool, int, int],
) -> tuple[list[str], dict[str, Any]]:
    contract = load_contract(contract_path)
    if not contract:
        return (["missing contract audit file"], {})

    events = contract.get("events", [])
    if not isinstance(events, list):
        events = []

    after_tree = _snapshot_project_tree(project_dir)
    changed_project_files = _detect_project_tree_changes(before_tree, after_tree)

    registry_changed = _file_fingerprint(_registry_path(project_dir)) != before_registry
    journal_changed = _file_fingerprint(project_dir / "logs" / "project_journal.md") != before_journal
    incidents_changed = _file_fingerprint(project_dir / "logs" / "incidents.md") != before_incidents
    handoffs_changed = _file_fingerprint(project_dir / "logs" / "handoffs.md") != before_handoffs

    allowed_checks = [
        e for e in events if e.get("type") == "check" and bool(e.get("allowed"))
    ]
    registry_events = [
        e for e in events if e.get("type") in {"registry_add", "registry_update"} and e.get("run_id") == run_id
    ]
    journal_events = [e for e in events if str(e.get("type", "")).startswith("journal_")]
    incident_events = [e for e in events if e.get("type") == "incident_report"]
    handoff_events = [e for e in events if e.get("type") == "handoff_record"]
    skill_events = [e for e in events if e.get("type") == "skill_consulted"]

    violations: list[str] = []
    if changed_project_files and not allowed_checks:
        violations.append(
            "project files changed without an allowed `drl-autoresearch check`"
        )
    if registry_changed and not registry_events:
        violations.append("experiment registry changed without ExperimentRegistry helper usage")
    if journal_changed and not journal_events:
        violations.append("project journal changed without ProjectJournal helper usage")
    if incidents_changed and not incident_events:
        violations.append("incidents log changed without IncidentLog helper usage")
    if handoffs_changed and not handoff_events:
        violations.append("handoffs log changed without HandoffLog helper usage")

    skills_available = _collect_skill_inventory(project_dir)
    if skills_available and not skill_events:
        violations.append("skills/ was not consulted and recorded before acting")

    details = {
        "changed_project_files": changed_project_files,
        "skills_available": skills_available,
        "allowed_check_count": len(allowed_checks),
        "registry_helper_events": len(registry_events),
        "journal_helper_events": len(journal_events),
        "incident_helper_events": len(incident_events),
        "handoff_helper_events": len(handoff_events),
        "skill_consultations": [str(e.get("skill_path", "")) for e in skill_events],
    }
    return violations, details


def _mark_agent_cycle_failed(
    project_dir: Path,
    *,
    run_id: str,
    backend: str,
    hypothesis: str,
    params: dict[str, Any],
    metric_name: str,
    notes: str,
    violations: list[str],
) -> None:
    incident_log = IncidentLog(project_dir)
    incident_log.initialize()
    incident_id = incident_log.report(
        incident_type="rule_violation",
        run_id=run_id,
        description="Autonomous agent cycle violated the runtime contract.",
        evidence={"backend": backend, "violations": "; ".join(violations), "notes": notes},
        severity="critical",
    )

    journal = ProjectJournal(project_dir)
    journal.initialize(project_name=project_dir.name, spec={})
    journal.log_event(
        "agent_contract_violation",
        (
            f"Cycle `{run_id}` was marked failed because the autonomous agent bypassed required runtime hooks.\n\n"
            f"Violations:\n- " + "\n- ".join(violations)
        ),
        metadata={"backend": backend, "incident_id": incident_id},
    )

    registry = ExperimentRegistry(project_dir=project_dir)
    existing = registry.get_run(run_id)
    violation_notes = f"{notes}\nContract violations: {'; '.join(violations)}".strip()
    if existing is not None:
        registry.update_run(
            run_id,
            {
                "notes": violation_notes,
                "rules_checked": "contract_violation",
            },
        )
        return

    config_summary = json.dumps(params, ensure_ascii=False)
    registry.add_run(
        RunRecord(
            run_id=run_id,
            hypothesis=hypothesis,
            agent=backend,
            config_summary=config_summary,
            change_summary=config_summary,
            rules_checked="contract_violation",
            custom_metric_name="" if metric_name == "reward" else metric_name,
            status="pending_agent",
            keep_decision="discard",
            notes=violation_notes,
        )
    )


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
    """Apply an explicit phase request without auto-promotion heuristics."""
    requested_phase = state.flags.get("requested_phase")
    if requested_phase not in VALID_PHASES:
        return False
    if requested_phase == state.current_phase:
        state.flags.pop("requested_phase", None)
        return False
    state.set_phase(str(requested_phase))
    state.flags.pop("requested_phase", None)
    return True


# ---------------------------------------------------------------------------
# Baseline experiment generator
# ---------------------------------------------------------------------------

def _generate_baseline_experiments() -> List[Dict[str, Any]]:
    """Return a minimal agent-driven fallback when the queue is empty."""
    return [
        {
            "run_id": str(uuid.uuid4()),
            "hypothesis": (
                "Agent-driven fallback cycle. Infer the next meaningful baseline or build step "
                "from the project spec, codebase, and current logs."
            ),
            "params": {"mode": "agent_driven_fallback"},
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
    other_information = project.get("other_information") or "none provided"

    files = {
        "BOOTSTRAP_CONTEXT.md": (
            "# Build Bootstrap Context\n\n"
            f"- Environment/domain: {env}\n"
            f"- Objective: {objective}\n"
            f"- Success metric: {success_metric}\n"
            f"- Other information: {other_information}\n\n"
            "Hard rules to enforce:\n"
            f"{rules_block}\n\n"
            "Instructions:\n"
            "- Treat this folder as lightweight context only.\n"
            "- Decide the actual build, research, and training plan from the project spec, codebase, and logs.\n"
            "- Do not treat this file as a fixed template or canned algorithm recommendation.\n"
            "- Keep outputs compact and update the normal project backbone as you go.\n"
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


def _load_policy_config(project_dir: Path) -> dict[str, Any]:
    config_dir = project_dir / ".drl_autoresearch"
    yaml_path = config_dir / "policy.yaml"
    if yaml_path.exists():
        try:
            import yaml  # type: ignore

            data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            pass

    json_path = config_dir / "policy.json"
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            pass

    return {}


def _refresh_cooldown_enabled(project_dir: Path) -> bool:
    config = _load_policy_config(project_dir)
    raw_value = config.get("refresh_cooldown_enabled", True)
    if isinstance(raw_value, bool):
        return raw_value
    lowered = str(raw_value).strip().lower()
    if lowered in {"true", "yes", "on", "1"}:
        return True
    if lowered in {"false", "no", "off", "0"}:
        return False
    return True


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
        requested = state.flags.get("research_refresh_requested")
        if isinstance(requested, str) and requested.strip():
            stuck, reason = True, requested.strip()
        elif requested:
            stuck, reason = True, str(
                state.flags.get("research_refresh_reason") or "requested"
            )
    if not stuck:
        return

    if not _refresh_cooldown_enabled(project_dir):
        console(
            "Research refresh requested and cooldown is disabled; refreshing immediately.",
            "warning",
        )
    else:
        last_refresh_runs = int(state.flags.get("last_refresh_total_runs", -10_000))
        if (state.total_runs - last_refresh_runs) < _REFRESH_COOLDOWN_RUNS:
            console(
                "Research refresh requested but cooldown is active; skipping refresh this run.",
                "info",
            )
            return

    console(
        f"Research refresh requested ({reason}); triggering research + redesign refresh.",
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
    state.flags["research_refresh_requested"] = False
    state.flags.pop("research_refresh_reason", None)
    journal = ProjectJournal(project_dir)
    journal.initialize(project_name=state.project_name, spec={})
    journal.log_event(
        "stuck_refresh",
        "Triggered compact research/plan refresh from an explicit refresh request.",
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


def _select_next_experiments(
    state: ProjectState,
    orchestrator: Optional[Orchestrator],
) -> list[dict[str, Any]]:
    experiments = list(state.queue)
    if experiments:
        return experiments
    if orchestrator is not None:
        next_exp = orchestrator.decide_next_experiment()
        if next_exp is not None:
            console("Queue is empty — using orchestrator-selected experiment.", "info")
            return [_normalize_orchestrator_experiment(next_exp, state)]
    console("Queue is empty — generating baseline experiments (fallback).", "info")
    return _generate_baseline_experiments()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(
    project_dir: Path,
    parallel: int = 1,
    dry_run: bool = False,
    once: bool = False,
    agent_backend: str = "auto",
) -> int:
    """Execute the autonomous agent-driven experiment loop.

    Parameters
    ----------
    project_dir:
        Root of the target project.
    parallel:
        Maximum number of concurrent experiments.
    dry_run:
        If True, validate and print what would run without executing.
    once:
        If True, run a single autonomous cycle and exit.
    agent_backend:
        ``auto`` | ``codex`` | ``claude``.
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

    backend = detect_backend(project_dir, preferred=agent_backend)
    if backend is None:
        console(
            "No supported coding-agent CLI detected for this project. Install Codex or Claude Code.",
            "error",
        )
        return 1
    if not autonomous_policy_allowed(project_dir):
        console(
            "Autonomous run requires onboarding permission policy `open`, `project-only`, or `bootstrap-only`.",
            "error",
        )
        return 1
    if parallel > 1:
        console(
            "Agent-driven runtime currently executes one autonomous cycle at a time; ignoring --parallel > 1.",
            "warning",
        )
        parallel = 1

    _shutdown_requested = False

    def _handle_sigint(sig, frame):  # noqa: ANN001
        nonlocal _shutdown_requested
        _shutdown_requested = True
        console("Shutdown requested — stopping after current autonomous cycle.", "warning")

    old_handler = signal.signal(signal.SIGINT, _handle_sigint)
    journal = ProjectJournal(project_dir)
    journal.initialize(project_name=state.project_name, spec={})

    try:
        cycle_count = 0
        while not _shutdown_requested:
            state = ProjectState.load(project_dir)
            _sync_state_from_registry(state, project_dir)
            state.flags["project_mode"] = _load_project_mode(project_dir, state)
            project_mode = state.flags["project_mode"]

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
                break

            experiments = _select_next_experiments(state, orchestrator)
            if not experiments:
                console("No experiments available for the next autonomous cycle.", "warning")
                break

            experiment = experiments[0]
            console(
                f"Autonomous cycle {cycle_count + 1}  |  backend={backend}  |  run={str(experiment.get('run_id', '?'))[:8]}",
                "info",
            )
            console(
                f"Experiments queued: {len(experiments)}  |  Parallel workers: {parallel}",
                "info",
            )

            if dry_run:
                console("Dry-run mode — autonomous agent will not be launched.", "warning")
                print(
                    f"  [{experiment.get('run_id', '?')[:8]}] "
                    f"{experiment.get('hypothesis', 'unnamed')} "
                    f"params={json.dumps(experiment.get('params', {}))}"
                )
                break

            run_id = str(experiment.get("run_id", str(uuid.uuid4())))
            contract_path = _contract_path(project_dir, run_id)
            initialize_contract(
                contract_path,
                run_id=run_id,
                backend=backend,
                project_mode=str(project_mode),
                phase=str(state.current_phase),
                hypothesis=str(experiment.get("hypothesis", "")),
            )
            pre_rows = _read_registry_rows(project_dir)
            before_tree = _snapshot_project_tree(project_dir)
            before_registry = _file_fingerprint(_registry_path(project_dir))
            before_journal = _file_fingerprint(project_dir / "logs" / "project_journal.md")
            before_incidents = _file_fingerprint(project_dir / "logs" / "incidents.md")
            before_handoffs = _file_fingerprint(project_dir / "logs" / "handoffs.md")
            state.queue = experiments[1:]
            _set_loop_flags(
                state,
                running=True,
                backend=backend,
                activity="agent_cycle_running",
                active_run_id=run_id or None,
            )
            state.save()

            prompt = build_agent_prompt(
                project_dir=project_dir,
                state=state.to_dict(),
                experiment=experiment,
                project_mode=project_mode,
            )
            agent_result = run_agent_cycle(
                project_dir=project_dir,
                backend=backend,
                prompt=prompt,
                env={
                    "DRL_AUTORESEARCH_RUN_ID": run_id,
                    "DRL_AUTORESEARCH_CONTRACT_PATH": str(contract_path),
                    "DRL_AUTORESEARCH_AGENT_BACKEND": backend,
                    "DRL_AUTORESEARCH_PROJECT_DIR": str(project_dir),
                },
                dangerous=True,
            )

            violations, contract_details = _validate_agent_contract(
                project_dir=project_dir,
                run_id=run_id,
                contract_path=contract_path,
                before_tree=before_tree,
                before_registry=before_registry,
                before_journal=before_journal,
                before_incidents=before_incidents,
                before_handoffs=before_handoffs,
            )

            post_rows = _read_registry_rows(project_dir)
            if len(post_rows) == len(pre_rows):
                _append_registry(
                    project_dir,
                    run_id=run_id,
                    hypothesis=str(experiment.get("hypothesis", "")),
                    params=experiment.get("params", {}),
                    metric_name=str(experiment.get("metric_name", "reward")),
                    metric_value=None,
                    status="crashed" if not agent_result.ok else "pending_agent",
                    notes=(
                        f"Autonomous backend={backend} exit_code={agent_result.exit_code}. "
                        f"stdout={agent_result.stdout_log.name} stderr={agent_result.stderr_log.name}"
                    ),
                )
                post_rows = _read_registry_rows(project_dir)

            if violations:
                _mark_agent_cycle_failed(
                    project_dir=project_dir,
                    run_id=run_id,
                    backend=backend,
                    hypothesis=str(experiment.get("hypothesis", "")),
                    params=experiment.get("params", {}),
                    metric_name=str(experiment.get("metric_name", "reward")),
                    notes=(
                        f"Autonomous backend={backend} exit_code={agent_result.exit_code}. "
                        f"stdout={agent_result.stdout_log.name} stderr={agent_result.stderr_log.name}"
                    ),
                    violations=violations,
                )
                post_rows = _read_registry_rows(project_dir)
                console(
                    f"Cycle {run_id[:8]} violated the runtime contract and was marked failed.",
                    "error",
                )
                for violation in violations:
                    console(f"  {violation}", "error")
                if contract_details.get("changed_project_files"):
                    console(
                        "  changed files: " + ", ".join(contract_details["changed_project_files"][:6]),
                        "error",
                    )

            state = ProjectState.load(project_dir)
            _sync_state_from_registry(state, project_dir)

            new_rows = post_rows[len(pre_rows):]
            for row in new_rows:
                status = str(row.get("status", "unknown"))
                keep_decision = str(row.get("keep_decision", ""))
                outcome = _classify_run_outcome(status, keep_decision)
                metric_name = str(row.get("metric_name", state.best_metric_name or "reward"))
                raw_metric = row.get("metric_value")
                try:
                    metric_value = float(raw_metric) if raw_metric not in ("", None) else None
                except (TypeError, ValueError):
                    metric_value = None
                level = "success" if outcome == "keep" else "warning" if outcome == "discard" else "error"
                console(
                    f"Run {str(row.get('run_id', '?'))[:8]}: status={status}  keep={keep_decision or 'discard'}  metric={metric_value if metric_value is not None else 'N/A'}",
                    level,
                )

            advanced = _maybe_advance_phase(state)
            if advanced:
                console(f"Phase advanced to: {state.current_phase}", "info")

            _finalize_build_mode_if_complete(project_dir=project_dir, state=state, results=new_rows)
            _set_loop_flags(
                state,
                running=True,
                backend=backend,
                activity="idle_between_cycles",
                active_run_id=None,
                last_exit_code=agent_result.exit_code,
            )
            state.save()

            journal.update_current_state_section(
                {
                    "Phase": state.current_phase,
                    "Best run": state.best_run_id or "none",
                    f"Best {state.best_metric_name}": (
                        state.best_metric_value if state.best_metric_value is not None else "n/a"
                    ),
                    "Total runs": state.total_runs,
                    "Kept / Discarded / Crashed": (
                        f"{state.kept_runs} / {state.discarded_runs} / {state.crashed_runs}"
                    ),
                    "Current bottleneck": f"agent_backend={backend}",
                    "Known dead ends": state.flags.get("last_refresh_reason") or "none",
                }
            )

            console(
                f"Cycle complete. Total runs: {state.total_runs}  Best {state.best_metric_name}: "
                f"{state.best_metric_value if state.best_metric_value is not None else 'N/A'}",
                "success",
            )

            cycle_count += 1
            if once:
                break
            time.sleep(_AGENT_LOOP_SLEEP_SECONDS)
    finally:
        signal.signal(signal.SIGINT, old_handler)
        state = ProjectState.load(project_dir)
        _sync_state_from_registry(state, project_dir)
        _set_loop_flags(
            state,
            running=False,
            backend=backend,
            activity="stopped",
            active_run_id=None,
            last_exit_code=state.flags.get("last_agent_exit_code"),
        )
        state.save()

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
