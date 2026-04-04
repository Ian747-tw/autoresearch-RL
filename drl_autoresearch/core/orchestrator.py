"""
Orchestrator — the single decision authority for DRL AutoResearch.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Optional yaml import with json fallback
try:
    import yaml  # type: ignore

    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

from drl_autoresearch.core.state import ProjectState, VALID_PHASES
from drl_autoresearch.core.policy import PolicyEngine, PolicyDecision

CONFIG_DIR = ".drl_autoresearch"
POLICY_FILE = "policy.yaml"
HARDWARE_FILE = "hardware.yaml"
PYTHON_ENV_FILE = "python_env.yaml"

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_yaml_or_json(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if _YAML_AVAILABLE:
        return yaml.safe_load(text)
    import json
    return json.loads(text)


def _make_run_id() -> str:
    return f"run-{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class Orchestrator:
    """
    Single decision authority that manages the full AutoResearch loop.

    Responsibilities:
    - Load and persist project state
    - Enforce policy via PolicyEngine
    - Decide the next experiment based on phase and constraints
    - Record results and update state
    - Surface explicitly requested research refreshes
    - Manage worker assignments
    """

    def __init__(self, project_dir: Path) -> None:
        self.project_dir = Path(project_dir)
        self._state: Optional[ProjectState] = None
        self._policy: Optional[PolicyEngine] = None
        self._hardware_config: Dict[str, Any] = {}
        self._policy_config: Dict[str, Any] = {}
        self._python_env_config: Dict[str, Any] = {}
        # In-memory run history: list of result dicts keyed by run_id
        self._run_history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load state, policy engine, and all config files."""
        self._state = ProjectState.load(self.project_dir)

        self._policy = PolicyEngine(self.project_dir)
        self._policy.load()

        self._hardware_config = self._load_config(HARDWARE_FILE)
        self._policy_config = self._load_config(POLICY_FILE)
        self._python_env_config = self._load_config(PYTHON_ENV_FILE)

        # Rebuild run history from state flags if available
        self._run_history = self._state.flags.get("run_history", [])

    def _load_config(self, filename: str) -> Dict[str, Any]:
        path = self.project_dir / CONFIG_DIR / filename
        if not path.exists():
            json_path = path.with_suffix(".json")
            if json_path.exists():
                import json
                return json.loads(json_path.read_text(encoding="utf-8")) or {}
            return {}
        result = _load_yaml_or_json(path)
        return result if isinstance(result, dict) else {}

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize_project(self, spec: dict) -> None:
        """
        First-time project setup.

        Creates .drl_autoresearch/ directory, writes initial state, and
        scaffolds default config files if they don't exist.

        spec keys (all optional):
          - project_name: str
          - initial_phase: str  (default: "research")
          - metric_name: str    (default: "reward")
        """
        if self._state is None:
            self._state = ProjectState.load(self.project_dir)

        config_dir = self.project_dir / CONFIG_DIR
        config_dir.mkdir(parents=True, exist_ok=True)

        # Apply spec overrides
        if "project_name" in spec:
            self._state.project_name = spec["project_name"]
        if "initial_phase" in spec:
            self._state.set_phase(spec.get("initial_phase", "research"))
        if "metric_name" in spec:
            self._state.best_metric_name = spec["metric_name"]

        # Scaffold default config files (never overwrite existing)
        self._scaffold_config_files(config_dir)

        self._state.save()

        # Re-load policy now that files exist
        if self._policy is None:
            self._policy = PolicyEngine(self.project_dir)
        self._policy.load()

    def _scaffold_config_files(self, config_dir: Path) -> None:
        """Write placeholder config files if they do not already exist."""
        defaults: Dict[str, str] = {
            "permissions.yaml": (
                "# DRL AutoResearch — permissions\n"
                "mode: open  # locked | prompted | bootstrap-only | open | project-only\n"
                "action_overrides: {}\n"
            ),
            "policy.yaml": (
                "# DRL AutoResearch — policy\n"
                "refresh_cooldown_enabled: true\n"
            ),
            "hardware.yaml": (
                "# DRL AutoResearch — hardware constraints\n"
                "gpu_memory_limit_gb: 24\n"
                "max_parallel_workers: 2\n"
                "compute_budget_hours: 48\n"
            ),
            "python_env.yaml": (
                "# DRL AutoResearch — python environment\n"
                "venv_path: .venv\n"
                "python_version: '3.10'\n"
            ),
        }
        for fname, content in defaults.items():
            target = config_dir / fname
            if not target.exists():
                target.write_text(content, encoding="utf-8")

        # NON_NEGOTIABLE_RULES.md at project root
        rules_path = self.project_dir / "NON_NEGOTIABLE_RULES.md"
        if not rules_path.exists():
            rules_path.write_text(
                "# NON_NEGOTIABLE_RULES\n\n"
                "These rules are enforced unconditionally by the policy engine.\n\n"
                "- Do not modify evaluation code without explicit user approval\n"
                "- Do not perform global package installs\n"
                "- Do not use privileged or confidential information\n"
                "- Do not exceed the configured compute budget\n"
                "- Do not silently fall back to CPU without notifying the user\n",
                encoding="utf-8",
            )

    # ------------------------------------------------------------------
    # Core decision loop
    # ------------------------------------------------------------------

    def decide_next_experiment(self) -> Optional[Dict[str, Any]]:
        """
        Decide what experiment to run next.

        Priority:
          1. User-queued experiments (from state.queue)
          2. Phase-appropriate auto-generated experiment
          3. None if converged or no valid experiment can be constructed
        """
        self._ensure_loaded()
        state = self._state  # type: ignore[assignment]

        if state.current_phase == "converged":
            return None

        # 1. Check user queue first
        queued = state.pop_queue()
        if queued is not None:
            queued.setdefault("run_id", _make_run_id())
            queued.setdefault("parent_run_id", state.best_run_id)
            queued.setdefault("hypothesis", "User-queued experiment")
            queued.setdefault("changes", [])
            queued.setdefault("expected_effect", "Unknown — user-specified")
            queued.setdefault("risk_level", "medium")
            queued.setdefault("skills_needed", [])
            queued.setdefault(
                "resource_budget", self._default_resource_budget()
            )
            state.save()
            return queued

        # 2. Auto-generate based on current phase
        experiment = self._generate_experiment_for_phase(state)
        return experiment

    def _generate_experiment_for_phase(
        self, state: ProjectState
    ) -> Optional[Dict[str, Any]]:
        """Generate a minimal experiment envelope and leave direction to the agent."""
        phase = state.current_phase
        run_id = _make_run_id()
        budget = self._default_resource_budget()
        if phase not in VALID_PHASES:
            return None

        best_metric = (
            f"{state.best_metric_name}={state.best_metric_value}"
            if state.best_metric_value is not None
            else "no established best yet"
        )
        return {
            "run_id": run_id,
            "hypothesis": (
                f"Agent-driven {phase} cycle. Infer the highest-signal next step from the "
                "project spec, current code, logs, and prior results."
            ),
            "changes": [],
            "expected_effect": (
                f"Make measurable progress appropriate for phase `{phase}` while preserving "
                f"the current best result ({best_metric})."
            ),
            "risk_level": "medium",
            "parent_run_id": state.best_run_id,
            "skills_needed": [],
            "resource_budget": budget,
            "params": {
                "phase": phase,
                "project_mode": state.flags.get("project_mode", "improve"),
            },
        }

    def _default_resource_budget(self) -> Dict[str, Any]:
        """Build a resource budget dict from hardware config."""
        gpu_limit = self._hardware_config.get("gpu_memory_limit_gb", 24)
        compute_hours = self._hardware_config.get("compute_budget_hours", 48)
        return {
            "gpu_memory_gb": gpu_limit,
            "max_wall_time_hours": min(8, compute_hours),
            "cpu_cores": self._hardware_config.get("cpu_cores", 8),
        }

    # ------------------------------------------------------------------
    # Recording results
    # ------------------------------------------------------------------

    def record_result(self, run_id: str, result: dict) -> None:
        """
        Record the outcome of a completed run and update state accordingly.

        result dict expected keys:
          - metric_value: float
          - metric_name: str         (optional, defaults to state.best_metric_name)
          - status: str              "success" | "crashed" | "discard"
          - notes: str               (optional)
        """
        self._ensure_loaded()
        state = self._state  # type: ignore[assignment]

        status = str(result.get("status", "success") or "success").strip().lower()
        if status == "discarded":
            status = "discard"
        metric_value = result.get("metric_value")
        metric_name = result.get("metric_name", state.best_metric_name)

        state.total_runs += 1

        if status == "crashed":
            state.crashed_runs += 1
        elif status == "discard":
            state.discarded_runs += 1
        else:
            state.kept_runs += 1
            if metric_value is not None:
                state.update_best(run_id, float(metric_value), metric_name)

        # Store result in flags-backed run history
        record = {
            "run_id": run_id,
            "metric_value": metric_value,
            "metric_name": metric_name,
            "status": status,
            "notes": result.get("notes", ""),
            "timestamp": _now_iso(),
        }
        self._run_history.append(record)
        state.flags["run_history"] = self._run_history

        self._maybe_advance_phase(state)

        state.save()

    def _maybe_advance_phase(self, state: ProjectState) -> None:
        """Respect an explicit phase request without inventing heuristics."""
        requested_phase = state.flags.get("requested_phase")
        if requested_phase not in VALID_PHASES:
            return
        if requested_phase == state.current_phase:
            state.flags.pop("requested_phase", None)
            return
        state.set_phase(str(requested_phase))
        state.flags.pop("requested_phase", None)

    # ------------------------------------------------------------------
    # Research refresh detection
    # ------------------------------------------------------------------

    def should_trigger_research_refresh(self) -> Tuple[bool, str]:
        """Return an explicitly requested research refresh, if any."""
        self._ensure_loaded()
        state = self._state  # type: ignore[assignment]
        requested = state.flags.get("research_refresh_requested")
        if isinstance(requested, str) and requested.strip():
            return True, requested.strip()
        if requested:
            reason = str(state.flags.get("research_refresh_reason") or "requested")
            return True, reason
        return False, ""

    # ------------------------------------------------------------------
    # Worker management
    # ------------------------------------------------------------------

    def assign_worker(self, worker_id: str, experiment: dict) -> None:
        """Register a worker as active and associate it with an experiment."""
        self._ensure_loaded()
        state = self._state  # type: ignore[assignment]

        if worker_id not in state.active_workers:
            state.active_workers.append(worker_id)

        # Track assignment in flags
        assignments: Dict[str, Any] = state.flags.get("worker_assignments", {})
        assignments[worker_id] = {
            "experiment_run_id": experiment.get("run_id"),
            "assigned_at": _now_iso(),
        }
        state.flags["worker_assignments"] = assignments
        state.save()

    def release_worker(self, worker_id: str) -> None:
        """Mark a worker as no longer active."""
        self._ensure_loaded()
        state = self._state  # type: ignore[assignment]

        if worker_id in state.active_workers:
            state.active_workers.remove(worker_id)

        assignments: Dict[str, Any] = state.flags.get("worker_assignments", {})
        assignments.pop(worker_id, None)
        state.flags["worker_assignments"] = assignments
        state.save()

    # ------------------------------------------------------------------
    # Status summary
    # ------------------------------------------------------------------

    def get_status_summary(self) -> dict:
        """Return a dict suitable for the CLI 'status' command."""
        self._ensure_loaded()
        state = self._state  # type: ignore[assignment]

        trigger, refresh_reason = self.should_trigger_research_refresh()

        return {
            "project_name": state.project_name,
            "project_dir": str(self.project_dir),
            "current_phase": state.current_phase,
            "current_branch": state.current_branch,
            "best_run_id": state.best_run_id,
            "best_metric_value": state.best_metric_value,
            "best_metric_name": state.best_metric_name,
            "total_runs": state.total_runs,
            "kept_runs": state.kept_runs,
            "discarded_runs": state.discarded_runs,
            "crashed_runs": state.crashed_runs,
            "active_workers": list(state.active_workers),
            "queue_depth": len(state.queue),
            "last_updated": state.last_updated,
            "initialized_at": state.initialized_at,
            "research_refresh_needed": trigger,
            "refresh_reason": refresh_reason,
            "policy_mode": self._policy.mode if self._policy else "unknown",
            "phase_order": list(VALID_PHASES),
            "flags": {
                k: v
                for k, v in state.flags.items()
                if k not in ("run_history", "worker_assignments")
            },
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._state is None or self._policy is None:
            self.load()

    def __repr__(self) -> str:
        phase = self._state.current_phase if self._state else "unloaded"
        return (
            f"Orchestrator(project={self.project_dir.name!r}, phase={phase!r})"
        )
