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
from pathlib import Path
from typing import Optional

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


def _detect_last_incident_signal(project_dir: Path) -> Optional[str]:
    lines = _tail_lines(project_dir / "logs" / "incidents.md", 120)
    if not lines:
        return None
    blob = "\n".join(lines).lower()
    if "critical" in blob:
        return "critical_incident_present"
    if "open incident" in blob or "open incidents" in blob:
        return "open_incidents_present"
    return None


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

    incident_signal = _detect_last_incident_signal(project_dir) or "none"
    refresh_reason = flags.get("last_refresh_reason") or "none"
    intent = "continue orchestrator loop with compact context"

    print()
    print("=== Session Sync Checkpoint (compact) ===")
    print(f"phase/mode: {phase} / {mode}")
    print(f"best run + metric: {best_run} ({best_metric_name}={best_metric_val})")
    print(f"latest 3 outcomes: {outcomes_str}")
    print(f"open incidents/handoff constraints: {incident_signal}; last_refresh_reason={refresh_reason}")
    print(f"next experiment intent: {intent}")


def run(
    project_dir: Path,
    parallel: int = 1,
    dry_run: bool = False,
    no_run: bool = False,
    agent_backend: str = "auto",
) -> int:
    project_dir = Path(project_dir).resolve()
    config_dir = project_dir / ".drl_autoresearch"
    if not config_dir.is_dir():
        console("Project not initialised. Run `drl-autoresearch init` first.", "error")
        return 1

    console("Resuming session with compact context sync.", "info")

    try:
        from drl_autoresearch.core.run import _sync_state_from_registry
        from drl_autoresearch.core.state import ProjectState

        state = ProjectState.load(project_dir)
        _sync_state_from_registry(state, project_dir)
        state.save()
    except Exception:
        pass

    # Reuse existing status output for continuity.
    from drl_autoresearch.core import status as status_mod

    status_mod.run(project_dir=project_dir)

    # Token-saving tails (fixed windows).
    _print_tail(project_dir, "logs/experiment_registry.tsv", 25)
    _print_tail(project_dir, "logs/project_journal.md", 120)
    _print_tail(project_dir, "logs/handoffs.md", 80)
    _print_tail(project_dir, "logs/incidents.md", 80)

    _print_session_checkpoint(project_dir)

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
