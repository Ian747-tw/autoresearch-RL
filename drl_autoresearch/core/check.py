"""
check — validate an agent action against hard rules and permissions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from drl_autoresearch.core.agent_contract import audit_event, env_run_id

from .policy import PolicyEngine


def run(project_dir: Path, action: str, details: Dict[str, Any] | None = None) -> int:
    """Check whether *action* is permitted under current policy.

    Prints a human-readable result and returns 0 (allowed) or 1 (blocked).
    """
    project_dir = Path(project_dir)
    details = details or {}

    engine = PolicyEngine(project_dir=project_dir)
    engine.load()

    decision = engine.check(action_type=action, details=details)

    status = "[✓] ALLOWED" if decision.allowed else "[✗] BLOCKED"
    print(f"{status}  action={action!r}  mode={decision.mode!r}")
    print(f"  Reason: {decision.reason}")

    if decision.violated_rules:
        print("  Violated rules:")
        for rule in decision.violated_rules:
            print(f"    • {rule}")

    if decision.requires_confirmation:
        print("  Note: confirmation required before execution.")

    audit_event(
        "check",
        {
            "run_id": env_run_id(),
            "action": action,
            "allowed": bool(decision.allowed),
            "requires_confirmation": bool(decision.requires_confirmation),
            "reason": decision.reason,
            "violated_rules": list(decision.violated_rules or []),
            "details": details,
        },
    )

    return 0 if decision.allowed else 1
