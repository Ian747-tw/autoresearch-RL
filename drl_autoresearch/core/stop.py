"""
drl_autoresearch.core.stop
---------------------------
Request the active autonomous run loop to stop and record a compact
resume brief describing what it was doing and what should happen next.
"""

from __future__ import annotations

import json
import os
import signal
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from drl_autoresearch.cli import console


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_state(project_dir: Path) -> dict[str, Any]:
    path = project_dir / ".drl_autoresearch" / "state.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_state(project_dir: Path, payload: dict[str, Any]) -> None:
    config_dir = project_dir / ".drl_autoresearch"
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / "state.json"
    payload["last_updated"] = _now_iso()
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _next_step_hint(state: dict[str, Any]) -> str:
    queue = state.get("queue", [])
    if isinstance(queue, list) and queue:
        head = queue[0]
        if isinstance(head, dict):
            hypothesis = str(head.get("hypothesis", "") or "").strip()
            if hypothesis:
                return hypothesis[:120]
    phase = str(state.get("current_phase", "research") or "research")
    return f"continue the highest-signal {phase} step from the current code/log state"


def _compose_stop_brief(state: dict[str, Any], brief: str = "") -> tuple[str, str]:
    flags = state.get("flags", {})
    if not isinstance(flags, dict):
        flags = {}
        state["flags"] = flags
    prior_activity = str(flags.get("current_activity", "") or "").strip() or "autonomous_cycle"
    active_run_id = str(flags.get("active_run_id", "") or "").strip()
    next_step = str(brief or "").strip() or _next_step_hint(state)
    run_suffix = f" (run {active_run_id[:8]})" if active_run_id else ""
    note = f"Stopped by user. Was {prior_activity}{run_suffix}. Next: {next_step}."
    return "stopping", note[:320]


def run(project_dir: Path, brief: str = "") -> int:
    project_dir = Path(project_dir).resolve()
    config_dir = project_dir / ".drl_autoresearch"
    if not config_dir.is_dir():
        console("Project not initialised. Run `drl-autoresearch init` first.", "error")
        return 1

    state = _load_state(project_dir)
    flags = state.get("flags", {})
    if not isinstance(flags, dict):
        flags = {}
        state["flags"] = flags

    activity, note = _compose_stop_brief(state, brief=brief)
    flags["current_activity"] = activity
    flags["current_activity_note"] = note
    flags["stop_brief_pending"] = True
    flags["stop_requested_at"] = _now_iso()

    pid_raw = flags.get("loop_pid")
    pid: int | None = None
    try:
        pid = int(pid_raw) if pid_raw not in (None, "") else None
    except (TypeError, ValueError):
        pid = None

    sent_signal = False
    if pid is not None and _process_exists(pid):
        try:
            os.kill(pid, signal.SIGINT)
            sent_signal = True
            flags["stop_signal_sent"] = True
        except OSError:
            flags["stop_signal_sent"] = False
    else:
        flags["stop_signal_sent"] = False

    _save_state(project_dir, state)

    if sent_signal:
        console("Stop requested — the gateway will stop after the current autonomous cycle.", "warning")
    else:
        console("No active gateway PID found. Saved a compact stop brief for resume anyway.", "warning")
    console(note, "info")
    return 0
