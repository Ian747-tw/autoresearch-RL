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
    """Run any research_*.py scripts in skills/ and collect new hypotheses."""
    skills_dir = project_dir / "skills"
    new_hypotheses: List[Dict[str, Any]] = []

    if not skills_dir.is_dir():
        return new_hypotheses

    skill_scripts = sorted(skills_dir.glob("research_*.py"))
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
    Identify hyperparameter regions not yet explored and return new hypotheses.

    Strategy:
    - Extract learning rates already tried.
    - Suggest a log-midpoint between the best and worst performing runs if
      there are enough data points.
    - Fall back to suggesting a standard PPO vs SAC vs TD3 comparison if the
      registry is sparse.
    """
    new_hypotheses: List[Dict[str, Any]] = []

    completed_rows = [
        r for r in registry_rows if r.get("status") == "completed" and r.get("metric_value")
    ]

    if len(completed_rows) < 2:
        # Sparse data — suggest common algorithm comparison.
        algorithms = ["PPO", "SAC", "TD3"]
        existing_titles = {h.get("title", "") for h in plan.get("hypotheses", [])}
        for algo in algorithms:
            title = f"Algorithm comparison — {algo}"
            if title not in existing_titles:
                new_hypotheses.append({
                    "id": str(uuid.uuid4()),
                    "title": title,
                    "rationale": f"Compare {algo} against current baseline.",
                    "params": {"algorithm": algo},
                    "priority": 3,
                    "status": "pending",
                })
        return new_hypotheses

    # Extract tried learning rates from plan params.
    tried_lrs = set()
    for h in plan.get("hypotheses", []):
        lr = h.get("params", {}).get("learning_rate")
        if lr is not None:
            tried_lrs.add(float(lr))

    # Find the best-performing run's learning rate (if parseable).
    try:
        best_row = max(completed_rows, key=lambda r: float(r.get("metric_value", 0)))
        best_params = json.loads(best_row.get("params_json", "{}"))
        best_lr = best_params.get("learning_rate")
        worst_row = min(completed_rows, key=lambda r: float(r.get("metric_value", 0)))
        worst_params = json.loads(worst_row.get("params_json", "{}"))
        worst_lr = worst_params.get("learning_rate")

        if best_lr and worst_lr and best_lr != worst_lr:
            import math
            mid_lr = math.exp((math.log(best_lr) + math.log(worst_lr)) / 2)
            rounded_lr = float(f"{mid_lr:.2e}")
            if rounded_lr not in tried_lrs:
                new_hypotheses.append({
                    "id": str(uuid.uuid4()),
                    "title": f"Learning rate mid-point (lr={rounded_lr:.2e})",
                    "rationale": (
                        f"Log-midpoint between best (lr={best_lr}) and worst (lr={worst_lr}) runs."
                    ),
                    "params": {"learning_rate": rounded_lr},
                    "priority": 4,
                    "status": "pending",
                })
    except (ValueError, KeyError, json.JSONDecodeError):
        pass

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
