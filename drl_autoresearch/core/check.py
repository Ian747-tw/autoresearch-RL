"""
check — validate an agent action against hard rules and permissions.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

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

    return 0 if decision.allowed else 1
