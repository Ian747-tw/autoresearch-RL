"""
drl_autoresearch.core.research
--------------------------------
Trigger a mid-training literature / research refresh.

What this module does
---------------------
1. Reads the current plan and experiment history.
2. Searches the local ``skills/`` directory for any ``research_*.py`` skill
   scripts and executes them (each script should print JSON to stdout with
   a list of new hypotheses or notes).
3. If no research skills are found, applies a built-in heuristic refresh:
   - Identifies under-explored hyperparameter regions based on registry data.
   - Appends new hypothesis entries to the plan.
4. Saves the updated plan.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from drl_autoresearch.cli import console
from drl_autoresearch.core.state import ProjectState


_PLAN_FILENAME = "plan.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_plan(project_dir: Path) -> Optional[Dict[str, Any]]:
    path = project_dir / ".drl_autoresearch" / _PLAN_FILENAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _save_plan(project_dir: Path, plan: Dict[str, Any]) -> None:
    path = project_dir / ".drl_autoresearch" / _PLAN_FILENAME
    path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_registry(project_dir: Path) -> List[Dict[str, str]]:
    path = project_dir / "logs" / "experiment_registry.tsv"
    if not path.exists():
        return []
    rows: List[Dict[str, str]] = []
    try:
        with path.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                rows.append(dict(row))
    except OSError:
        pass
    return rows


def _run_research_skills(project_dir: Path) -> List[Dict[str, Any]]:
    """Run research skill scripts or fall back to bundled markdown playbooks."""
    skills_dir = project_dir / "skills"
    new_hypotheses: List[Dict[str, Any]] = []

    if not skills_dir.is_dir():
        return new_hypotheses

    skill_scripts = sorted(skills_dir.glob("research_*.py"))
    skill_playbooks = [
        path for path in sorted(skills_dir.glob("*.md"))
        if "research" in path.stem.lower()
    ]
    if not skill_scripts and skill_playbooks:
        for playbook in skill_playbooks:
            console(f"  Using research playbook: {playbook.name}", "info")
            title = f"Research playbook review — {playbook.stem}"
            new_hypotheses.append(
                {
                    "id": str(uuid.uuid4()),
                    "title": title,
                    "rationale": (
                        f"Consult `{playbook.name}` and derive the next project-specific "
                        "research direction from current results, code, and constraints."
                    ),
                    "params": {"skill_playbook": playbook.name, "mode": "agent_driven_refresh"},
                    "priority": 3,
                    "status": "pending",
                }
            )
        return new_hypotheses
    if not skill_scripts:
        return new_hypotheses

    for script in skill_scripts:
        console(f"  Running research skill: {script.name}", "info")
        try:
            completed = subprocess.run(
                [sys.executable, str(script)],
                capture_output=True,
                text=True,
                timeout=300,
                env={**os.environ},
            )
            if completed.returncode == 0:
                lines = [l for l in completed.stdout.strip().splitlines() if l.strip()]
                for line in lines:
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, list):
                            new_hypotheses.extend(obj)
                        elif isinstance(obj, dict):
                            new_hypotheses.append(obj)
                    except json.JSONDecodeError:
                        pass
            else:
                console(
                    f"  Skill {script.name} exited with code {completed.returncode}: "
                    f"{completed.stderr[:200]}",
                    "warning",
                )
        except subprocess.TimeoutExpired:
            console(f"  Skill {script.name} timed out.", "warning")
        except Exception as exc:  # noqa: BLE001
            console(f"  Skill {script.name} error: {exc}", "warning")

    return new_hypotheses


def _heuristic_refresh(
    plan: Dict[str, Any], registry_rows: List[Dict[str, str]]
) -> List[Dict[str, Any]]:
    """
    Return minimal agent-facing refresh hypotheses without algorithm templates.
    """
    new_hypotheses: List[Dict[str, Any]] = []
    existing_titles = {h.get("title", "") for h in plan.get("hypotheses", [])}

    completed_rows = [
        r for r in registry_rows if r.get("status") == "completed" and r.get("metric_value")
    ]

    if len(completed_rows) < 2:
        title = "Agent-driven sparse-data research refresh"
        if title not in existing_titles:
            new_hypotheses.append({
                "id": str(uuid.uuid4()),
                "title": title,
                "rationale": (
                    "Completed-run history is sparse. Inspect the project spec, current code, "
                    "and failure modes to propose the next build or training direction instead "
                    "of relying on canned algorithm comparisons."
                ),
                "params": {"mode": "agent_driven_refresh", "sparse_data": True},
                "priority": 3,
                "status": "pending",
            })
        return new_hypotheses

    title = "Agent-driven targeted research refresh"
    if title not in existing_titles:
        new_hypotheses.append({
            "id": str(uuid.uuid4()),
            "title": title,
            "rationale": (
                "Review the strongest and weakest recent runs, then derive the next "
                "highest-signal change from project-specific evidence rather than fixed "
                "hyperparameter or algorithm templates."
            ),
            "params": {"mode": "agent_driven_refresh", "sparse_data": False},
            "priority": 4,
            "status": "pending",
        })

    return new_hypotheses


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(project_dir: Path) -> int:
    """Trigger a mid-training research refresh.

    Returns 0 on success, 1 on error.
    """
    project_dir = Path(project_dir).resolve()
    config_dir  = project_dir / ".drl_autoresearch"

    if not config_dir.is_dir():
        console(
            "Project not initialised. Run `drl-autoresearch init` first.", "error"
        )
        return 1

    console("Starting research refresh...", "info")

    state         = ProjectState.load(project_dir)
    plan          = _load_plan(project_dir)
    registry_rows = _read_registry(project_dir)

    if plan is None:
        console(
            "No plan found. Run `drl-autoresearch plan --refresh` to create one.", "warning"
        )
        return 1

    # 1. Try research skill scripts first.
    new_hypotheses = _run_research_skills(project_dir)

    # 2. Fall back to heuristic refresh.
    if not new_hypotheses:
        console("No research skills found — applying heuristic refresh.", "info")
        new_hypotheses = _heuristic_refresh(plan, registry_rows)

    if not new_hypotheses:
        console("No new hypotheses generated — plan is already comprehensive.", "info")
        return 0

    # 3. Merge new hypotheses into the plan (avoid duplicates by title).
    existing_titles = {h.get("title", "") for h in plan.get("hypotheses", [])}
    added = 0
    for h in new_hypotheses:
        h.setdefault("id", str(uuid.uuid4()))
        h.setdefault("status", "pending")
        h.setdefault("priority", 3)
        if h.get("title", "") not in existing_titles:
            plan["hypotheses"].append(h)
            existing_titles.add(h["title"])
            added += 1
            console(f"  + {h['title']}", "success")

    # 4. Update plan metadata.
    plan["generated_at"] = datetime.now(timezone.utc).isoformat()
    plan["phase"] = state.current_phase

    _save_plan(project_dir, plan)
    console(f"Research refresh complete — {added} new hypothesis/es added.", "success")
    return 0


class ResearchPlanner:
    """Class wrapper around the module-level run() for callers that expect a class interface."""

    def __init__(self, project_dir: Path):
        self.project_dir = project_dir

    def refresh(self) -> int:
        return run(self.project_dir)

    def get_plan(self) -> Optional[Dict[str, Any]]:
        return _load_plan(self.project_dir)
