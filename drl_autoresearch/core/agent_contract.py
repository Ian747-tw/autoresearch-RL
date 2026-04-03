"""
Audit trail for one autonomous agent cycle.

The controller creates a contract file for each cycle. Helper APIs and
`drl-autoresearch check` append audit events to that file via environment
variables so the controller can verify that the agent used the expected
backbone integrations.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


ENV_CONTRACT_PATH = "DRL_AUTORESEARCH_CONTRACT_PATH"
ENV_RUN_ID = "DRL_AUTORESEARCH_RUN_ID"
ENV_BACKEND = "DRL_AUTORESEARCH_AGENT_BACKEND"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def initialize_contract(
    path: Path,
    *,
    run_id: str,
    backend: str,
    project_mode: str,
    phase: str,
    hypothesis: str,
) -> None:
    payload = {
        "run_id": run_id,
        "backend": backend,
        "project_mode": project_mode,
        "phase": phase,
        "hypothesis": hypothesis,
        "created_at": _now_iso(),
        "events": [],
    }
    _atomic_write_json(Path(path), payload)


def load_contract(path: Path) -> dict[str, Any]:
    contract_path = Path(path)
    if not contract_path.exists():
        return {}
    try:
        data = json.loads(contract_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def append_event(path: Path, event_type: str, payload: dict[str, Any]) -> None:
    contract_path = Path(path)
    data = load_contract(contract_path)
    if not data:
        data = {"events": []}
    events = data.get("events")
    if not isinstance(events, list):
        events = []
        data["events"] = events
    events.append(
        {
            "type": event_type,
            "timestamp": _now_iso(),
            **payload,
        }
    )
    _atomic_write_json(contract_path, data)


def env_contract_path() -> Optional[Path]:
    raw = os.environ.get(ENV_CONTRACT_PATH, "").strip()
    return Path(raw) if raw else None


def env_run_id() -> str:
    return os.environ.get(ENV_RUN_ID, "").strip()


def audit_event(event_type: str, payload: dict[str, Any]) -> None:
    path = env_contract_path()
    if path is None:
        return
    try:
        append_event(path, event_type, payload)
    except Exception:
        return


def record_skill_consultation(skill_path: str, note: str = "") -> None:
    audit_event(
        "skill_consulted",
        {
            "run_id": env_run_id(),
            "skill_path": skill_path,
            "note": note,
        },
    )
