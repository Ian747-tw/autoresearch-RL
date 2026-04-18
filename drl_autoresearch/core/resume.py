"""
drl_autoresearch.core.resume
-----------------------------
Compact recovery command for interrupted or new agent sessions.

Behavior:
1. Validate project is initialized.
2. Print current status summary.
3. Perform token-saving context sync from tail windows of key logs.
4. Print a compact session checkpoint.
5. Continue the training loop by default (unless --no-run).
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from drl_autoresearch.cli import console


def _tail_lines(path: Path, n: int) -> list[str]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return lines[-n:]


def _print_tail(project_dir: Path, rel_path: str, n: int) -> None:
    path = project_dir / rel_path
    lines = _tail_lines(path, n)
    print()
    print(f"--- {rel_path} (last {n} lines) ---")
    if not lines:
        print("(missing or empty)")
        return
    for line in lines:
        print(line)


def _load_state(project_dir: Path) -> dict:
    path = project_dir / ".drl_autoresearch" / "state.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def _recent_registry_rows(project_dir: Path, limit: int = 3) -> list[dict]:
    path = project_dir / "logs" / "experiment_registry.tsv"
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            rows = [r for r in reader if isinstance(r, dict)]
    except OSError:
        return []
    if not rows:
        return []
    return rows[-limit:]


def _print_session_checkpoint(project_dir: Path) -> None:
    state = _load_state(project_dir)
    flags = state.get("flags", {}) if isinstance(state.get("flags", {}), dict) else {}
    mode = flags.get("project_mode", "improve")
    phase = state.get("current_phase", "research")
    best_run = state.get("best_run_id", "none")
    best_metric_name = state.get("best_metric_name", "reward")
    best_metric_val = state.get("best_metric_value")

    recent = _recent_registry_rows(project_dir, limit=3)
    outcomes = []
    for r in recent:
        rid = (r.get("run_id") or "?")[:8]
        status = r.get("status", "unknown")
        keep = r.get("keep_decision", "") or "discard"
        outcomes.append(f"{rid}:{status}/{keep}")
    outcomes_str = ", ".join(outcomes) if outcomes else "no recent runs"

    refresh_reason = flags.get("last_refresh_reason") or "none"
    intent = "continue orchestrator loop with compact context"
    current_activity = flags.get("current_activity")
    current_activity_note = flags.get("current_activity_note")
    stop_brief_pending = bool(flags.get("stop_brief_pending", False))

    print()
    print("=== Session Sync Checkpoint (compact) ===")
    print(f"phase/mode: {phase} / {mode}")
    print(f"best run + metric: {best_run} ({best_metric_name}={best_metric_val})")
    print(f"latest 3 outcomes: {outcomes_str}")
    if current_activity and stop_brief_pending:
        suffix = f" ({current_activity_note})" if current_activity_note else ""
        print(f"interrupted activity: {current_activity}{suffix}")
    print(f"handoff constraints: last_refresh_reason={refresh_reason}")
    print(f"next experiment intent: {intent}")


def _consume_stop_brief(project_dir: Path) -> None:
    state = _load_state(project_dir)
    flags = state.get("flags", {})
    if not isinstance(flags, dict) or not flags.get("stop_brief_pending"):
        return
    flags.pop("stop_brief_pending", None)
    flags.pop("stop_requested_at", None)
    flags.pop("stop_signal_sent", None)
    flags.pop("current_activity", None)
    flags.pop("current_activity_note", None)
    flags.pop("active_run_id", None)
    flags.pop("loop_pid", None)
    flags["loop_running"] = False
    state["last_updated"] = datetime.now(timezone.utc).isoformat()
    path = project_dir / ".drl_autoresearch" / "state.json"
    try:
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(path)
    except OSError:
        return


def _store_resume_override(project_dir: Path, message: str) -> None:
    message = str(message or "").strip()
    if not message:
        return
    path = project_dir / ".drl_autoresearch" / "state.json"
    state = _load_state(project_dir)
    flags = state.get("flags", {})
    if not isinstance(flags, dict):
        flags = {}
        state["flags"] = flags
    flags["resume_override_message"] = message
    state["last_updated"] = datetime.now(timezone.utc).isoformat()
    try:
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(path)
    except OSError:
        return


def run(
    project_dir: Path,
    parallel: int = 1,
    dry_run: bool = False,
    no_run: bool = False,
    agent_backend: str = "auto",
    message: str = "",
) -> int:
    project_dir = Path(project_dir).resolve()
    config_dir = project_dir / ".drl_autoresearch"
    if not config_dir.is_dir():
        console("Project not initialised. Run `drl-autoresearch init` first.", "error")
        return 1

    resume_message = str(message or "").strip()
    console("Resuming session with compact context sync.", "info")
    if resume_message:
        _store_resume_override(project_dir, resume_message)
        console("Stored a one-shot resume override for the next agent cycle.", "info")

    try:
        from drl_autoresearch.dashboard.metrics import MetricsCollector
        from drl_autoresearch.core.run import _sync_state_from_registry
        from drl_autoresearch.core.state import ProjectState

        state = ProjectState.load(project_dir)
        _sync_state_from_registry(state, project_dir)
        state.save()
        MetricsCollector(project_dir).reconcile_dashboard_backends()
    except Exception:
        pass

    # Reuse existing status output for continuity.
    from drl_autoresearch.core import status as status_mod

    status_mod.run(project_dir=project_dir)

    # Token-saving tails (fixed windows).
    _print_tail(project_dir, "logs/experiment_registry.tsv", 25)
    _print_tail(project_dir, "logs/project_journal.md", 100)
    _print_tail(project_dir, "logs/handoffs.md", 80)

    _print_session_checkpoint(project_dir)
    _consume_stop_brief(project_dir)
    try:
        from drl_autoresearch.dashboard.metrics import MetricsCollector

        MetricsCollector(project_dir).reconcile_dashboard_backends()
    except Exception:
        pass

    if no_run:
        console("Resume sync completed. Not starting run loop (--no-run).", "success")
        return 0

    console("Starting autonomous loop after resume sync.", "info")
    from drl_autoresearch.core import run as run_mod

    return run_mod.run(
        project_dir=project_dir,
        parallel=parallel,
        dry_run=dry_run,
        agent_backend=agent_backend,
    )
