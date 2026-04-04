"""
drl_autoresearch.core.plan
---------------------------
Show or refresh the implementation plan stored in
``.drl_autoresearch/plan.json``.

Plan schema
-----------
{
  "version": "1.0",
  "generated_at": "<iso-timestamp>",
  "phase": "<current_phase>",
  "best_run_id": "<id or null>",
  "best_metric_value": <float or null>,
  "hypotheses": [
    {
      "id": "<uuid>",
      "title": "<short description>",
      "rationale": "<why try this>",
      "params": {<hyperparameter overrides>},
      "priority": <1–5>,
      "status": "pending|running|completed|discard"
    },
    ...
  ],
  "notes": "<free-text strategic notes>"
}
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from drl_autoresearch.cli import console
from drl_autoresearch.core.state import ProjectState


_PLAN_FILENAME = "plan.json"


# ---------------------------------------------------------------------------
# Default plan generation
# ---------------------------------------------------------------------------

def _default_hypotheses(state: ProjectState) -> List[Dict[str, Any]]:
    """Return a compact agent-driven starter set without canned templates."""
    base = [
        {
            "id": str(uuid.uuid4()),
            "title": "Agent-driven next step",
            "rationale": (
                "Inspect the current spec, codebase, logs, and registry, then choose the "
                "single highest-signal next action."
            ),
            "params": {"phase": state.current_phase, "mode": "agent_driven_plan"},
            "priority": 5,
            "status": "pending",
        },
        {
            "id": str(uuid.uuid4()),
            "title": "Evidence-led follow-up",
            "rationale": (
                "Use the strongest completed run and the most recent failures to decide the "
                "next targeted experiment or build task."
            ),
            "params": {"phase": state.current_phase, "uses_history": True},
            "priority": 4,
            "status": "pending",
        },
    ]

    if state.current_phase == "research":
        base.append({
            "id": str(uuid.uuid4()),
            "title": "Research/build bootstrap",
            "rationale": (
                "If the project is not yet runnable, prioritize the smallest research or "
                "implementation step that makes the training loop real."
            ),
            "params": {"project_readiness": "bootstrap"},
            "priority": 3,
            "status": "pending",
        })
    else:
        base.append({
            "id": str(uuid.uuid4()),
            "title": "Keep/discard pressure test",
            "rationale": (
                "Propose one measurable change that can beat the current best result or get "
                "discarded cleanly with useful evidence."
            ),
            "params": {"project_readiness": "iterative_loop"},
            "priority": 3,
            "status": "pending",
        })

    return base


def _build_plan(state: ProjectState) -> Dict[str, Any]:
    return {
        "version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "phase": state.current_phase,
        "best_run_id": state.best_run_id,
        "best_metric_value": state.best_metric_value,
        "hypotheses": _default_hypotheses(state),
        "notes": (
            "Auto-generated plan. Edit hypotheses freely. "
            "Treat it as lightweight context, not a fixed experiment template. "
            "Run `drl-autoresearch plan --refresh` to regenerate."
        ),
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _display_plan(plan: Dict[str, Any]) -> None:
    """Pretty-print the plan to stdout."""
    print()
    print("=" * 60)
    print(f"  DRL AutoResearch — Research Plan")
    print(f"  Generated : {plan.get('generated_at', 'unknown')[:19]}")
    print(f"  Phase     : {plan.get('phase', 'unknown')}")
    best_val = plan.get("best_metric_value")
    best_id  = plan.get("best_run_id")
    if best_id:
        print(f"  Best run  : {best_id[:8]}  value={best_val:.4f}" if best_val is not None else f"  Best run  : {best_id[:8]}")
    print("=" * 60)
    print()

    hypotheses: List[Dict[str, Any]] = plan.get("hypotheses", [])
    if not hypotheses:
        print("  No hypotheses in plan.")
        return

    # Sort by priority descending, then by status (pending first).
    _status_order = {"pending": 0, "running": 1, "completed": 2, "discard": 3, "discarded": 3}
    hypotheses_sorted = sorted(
        hypotheses,
        key=lambda h: (-h.get("priority", 0), _status_order.get(h.get("status", "pending"), 99)),
    )

    for h in hypotheses_sorted:
        status   = h.get("status", "pending")
        if status == "discarded":
            status = "discard"
        priority = h.get("priority", 0)
        hid      = h.get("id", "?")[:8]
        title    = h.get("title", "")
        rationale= h.get("rationale", "")
        params   = h.get("params", {})

        status_icon = {
            "pending":   "[~]",
            "running":   "[>]",
            "completed": "[✓]",
            "discard":   "[-]",
        }.get(status, "[?]")

        print(f"  {status_icon} (P{priority}) [{hid}] {title}")
        if rationale:
            print(f"          {rationale}")
        if params:
            print(f"          params: {json.dumps(params)}")
        print()

    notes = plan.get("notes", "")
    if notes:
        print(f"  Notes: {notes}")
    print()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(project_dir: Path, refresh: bool = False) -> int:
    """Show or regenerate the research plan.

    Returns 0 on success, 1 on error.
    """
    project_dir = Path(project_dir).resolve()
    config_dir  = project_dir / ".drl_autoresearch"

    if not config_dir.is_dir():
        console(
            "Project not initialised. Run `drl-autoresearch init` first.", "error"
        )
        return 1

    plan_path = config_dir / _PLAN_FILENAME
    state     = ProjectState.load(project_dir)

    if refresh or not plan_path.exists():
        console("Generating plan from current state...", "info")
        plan = _build_plan(state)
        try:
            plan_path.write_text(
                json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            console(f"Plan written to {plan_path.relative_to(project_dir)}", "success")
        except OSError as exc:
            console(f"Could not write plan: {exc}", "error")
            return 1
    else:
        try:
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            console(f"Could not read plan file: {exc}", "error")
            return 1

    _display_plan(plan)
    return 0
