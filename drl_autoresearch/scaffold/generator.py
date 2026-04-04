"""
drl_autoresearch.scaffold.generator
------------------------------------
ScaffoldGenerator: creates all project files in a TARGET DRL project when
``drl-autoresearch init`` is run.

Design notes
~~~~~~~~~~~~
- Uses Python str.format() templating only — no Jinja2 dependency.
- Never overwrites existing files unless overwrite=True is passed explicitly.
- Created files live inside the TARGET project, not the plugin install.
- Directories: .drl_autoresearch/, logs/, skills/, dashboard/
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# OnboardingResult — lightweight dataclass (no external deps)
# ---------------------------------------------------------------------------

class OnboardingResult:
    """Holds all answers collected during the interactive onboarding wizard.

    All fields have sensible defaults so the generator can run even when
    onboarding is skipped (``drl-autoresearch init --auto``).
    """

    def __init__(
        self,
        project_name: str = "",
        environment: str = "CartPole-v1",
        obs_type: str = "vector",
        action_space: str = "discrete",
        objective: str = "maximise cumulative episodic reward",
        success_metric: str = "mean_eval_reward",
        target_value: str = "195.0 over 100 consecutive episodes",
        wall_clock_budget: str = "24 hours",
        compute_budget: str = "10",
        other_information: str = "",
        max_vram: str = "auto",
        reward_modification_allowed: bool = True,
        env_modification_allowed: bool = False,
        eval_modification_allowed: bool = False,
        offline_data_allowed: bool = False,
        imitation_allowed: bool = False,
        modification_policy: str = "all",
        offline_data_policy: str = "no",
        imitation_policy: str = "no",
        algorithm_recommendations: str = "PPO, DQN, A2C",
        user_rules: Optional[list[str]] = None,
        hardware_summary: str = "",
        python_env_summary: str = "",
    ) -> None:
        self.project_name = project_name
        self.environment = environment
        self.obs_type = obs_type
        self.action_space = action_space
        self.objective = objective
        self.success_metric = success_metric
        self.target_value = target_value
        self.wall_clock_budget = wall_clock_budget
        self.compute_budget = compute_budget
        self.other_information = other_information
        self.max_vram = max_vram
        self.reward_modification_allowed = reward_modification_allowed
        self.env_modification_allowed = env_modification_allowed
        self.eval_modification_allowed = eval_modification_allowed
        self.offline_data_allowed = offline_data_allowed
        self.imitation_allowed = imitation_allowed
        self.modification_policy = modification_policy
        self.offline_data_policy = offline_data_policy
        self.imitation_policy = imitation_policy
        self.algorithm_recommendations = algorithm_recommendations
        self.user_rules = user_rules or []
        self.hardware_summary = hardware_summary
        self.python_env_summary = python_env_summary


# ---------------------------------------------------------------------------
# ScaffoldGenerator
# ---------------------------------------------------------------------------

class ScaffoldGenerator:
    """Generate all scaffold files inside a target DRL project directory.

    Parameters
    ----------
    project_dir:
        Absolute path to the root of the TARGET project.
    onboarding_result:
        Populated OnboardingResult from the onboarding wizard, or a default
        instance when running with ``--auto`` / ``--skip-onboarding``.
    """

    def __init__(
        self,
        project_dir: Path,
        onboarding_result: "OnboardingResult",
    ) -> None:
        self.project_dir = Path(project_dir).resolve()
        self.ob = onboarding_result
        self._created: list[Path] = []

        # Lazily computed timestamp — same for all files in one run.
        self._timestamp: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_all(self) -> list[Path]:
        """Run all generators in order.  Returns list of created file paths."""
        self._created = []
        self.generate_config_dir()
        self.generate_markdown_docs()
        self.generate_skills()
        self.generate_logs_structure()
        self.generate_dashboard_config()
        return list(self._created)

    # ------------------------------------------------------------------
    # Sub-generators
    # ------------------------------------------------------------------

    def generate_config_dir(self) -> None:
        """Create .drl_autoresearch/ and write all YAML / JSON config files."""
        config_dir = self.project_dir / ".drl_autoresearch"
        config_dir.mkdir(parents=True, exist_ok=True)

        self._write_file(config_dir / "policy.yaml",      self._render_policy_yaml())
        self._write_file(config_dir / "hardware.yaml",    self._render_hardware_yaml())
        self._write_file(config_dir / "python_env.yaml",  self._render_python_env_yaml())
        self._write_file(config_dir / "permissions.yaml", self._render_permissions_yaml())
        self._write_file(config_dir / "state.json",       self._render_state_json())

    def generate_markdown_docs(self) -> None:
        """Create top-level Markdown guidance and specification files."""
        self._write_file(self.project_dir / "CLAUDE.md",               self._render_claude_md())
        self._write_file(self.project_dir / "AGENT.md",                self._render_agent_md())
        self._write_file(self.project_dir / "ORCHESTRATOR.md",         self._render_orchestrator_md())
        self._write_file(self.project_dir / "USER_SPEC.md",            self._render_user_spec_md())
        self._write_file(self.project_dir / "NON_NEGOTIABLE_RULES.md", self._render_non_negotiable_md())
        self._write_file(self.project_dir / "IMPLEMENTATION_PLAN.md",  self._render_implementation_plan_md())

    def generate_skills(self) -> None:
        """Create the skills/ directory and all playbook files."""
        skills_dir = self.project_dir / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)

        skill_files = {
            "investigate.md":          _SKILL_INVESTIGATE,
            "reward_shaping.md":       _SKILL_REWARD_SHAPING,
            "exploration.md":          _SKILL_EXPLORATION,
            "env_diagnostics.md":      _SKILL_ENV_DIAGNOSTICS,
            "ablation.md":             _SKILL_ABLATION,
            "checkpoint_selection.md": _SKILL_CHECKPOINT_SELECTION,
            "compute_budgeting.md":    _SKILL_COMPUTE_BUDGETING,
            "mid_training_research.md": _SKILL_MID_TRAINING_RESEARCH,
        }
        for filename, content in skill_files.items():
            self._write_file(skills_dir / filename, content)

        # Ensure a .gitkeep so the directory is tracked in empty repos
        gitkeep = skills_dir / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()

    def generate_logs_structure(self) -> None:
        """Create logs/ directory structure and placeholder files."""
        logs_dir = self.project_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        registry_header = (
            "run_id\ttimestamp\thypothesis\tparams_json\t"
            "metric_name\tmetric_value\tstatus\tnotes\n"
        )
        self._write_file(logs_dir / "experiment_registry.tsv", registry_header)
        self._write_file(logs_dir / "incidents.md",            _INCIDENTS_TEMPLATE)
        self._write_file(logs_dir / "handoffs.md",             _HANDOFFS_TEMPLATE)

        # Sub-directory placeholders
        for subdir in ("checkpoints", "tensorboard", "videos"):
            d = logs_dir / subdir
            d.mkdir(parents=True, exist_ok=True)
            gitkeep = d / ".gitkeep"
            if not gitkeep.exists():
                gitkeep.touch()

    def generate_dashboard_config(self) -> None:
        """Create dashboard/ config stub."""
        dash_dir = self.project_dir / "dashboard"
        dash_dir.mkdir(parents=True, exist_ok=True)
        self._write_file(dash_dir / "config.json", self._render_dashboard_config_json())

    # ------------------------------------------------------------------
    # Internal write helper
    # ------------------------------------------------------------------

    def _write_file(self, path: Path, content: str, overwrite: bool = False) -> bool:
        """Write *content* to *path*.

        Parameters
        ----------
        path:
            Destination file path (will be created including parents).
        content:
            Text content to write (UTF-8).
        overwrite:
            If False (default) skip the file if it already exists and print
            a warning.  If True always write.

        Returns
        -------
        bool
            True if the file was written, False if it was skipped.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists() and not overwrite:
            _warn(f"Skipped (already exists): {path.relative_to(self.project_dir)}")
            return False

        path.write_text(content, encoding="utf-8")
        _info(f"Created {path.relative_to(self.project_dir)}")
        self._created.append(path)
        return True

    # ------------------------------------------------------------------
    # Timestamp helper
    # ------------------------------------------------------------------

    @property
    def _ts(self) -> str:
        if self._timestamp is None:
            self._timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        return self._timestamp

    # ------------------------------------------------------------------
    # Config file renderers
    # ------------------------------------------------------------------

    def _render_policy_yaml(self) -> str:
        ob = self.ob
        return _POLICY_YAML_TEMPLATE.format(
            timestamp=self._ts,
            project_name=ob.project_name or self.project_dir.name,
            compute_budget=ob.compute_budget,
            max_vram=ob.max_vram,
        )

    def _render_hardware_yaml(self) -> str:
        ob = self.ob
        # Use whatever hardware summary onboarding collected, or fall back
        # to a sensible placeholder that the doctor command can fill later.
        if ob.hardware_summary:
            return (
                "# DRL AutoResearch — hardware configuration\n"
                "# Auto-detected during `drl-autoresearch init`.\n"
                "# Run `drl-autoresearch doctor` to refresh.\n\n"
                + ob.hardware_summary
            )
        return _HARDWARE_YAML_DEFAULT

    def _render_python_env_yaml(self) -> str:
        ob = self.ob
        if ob.python_env_summary:
            return (
                "# DRL AutoResearch — Python environment\n"
                "# Auto-detected during `drl-autoresearch init`.\n\n"
                + ob.python_env_summary
            )
        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        return _PYTHON_ENV_YAML_TEMPLATE.format(python_version=py_ver)

    def _render_permissions_yaml(self) -> str:
        ob = self.ob
        reward_flag  = str(ob.reward_modification_allowed).lower()
        env_flag     = str(ob.env_modification_allowed).lower()
        eval_flag    = str(ob.eval_modification_allowed).lower()
        offline_flag = str(ob.offline_data_allowed).lower()
        imitation_flag = str(ob.imitation_allowed).lower()
        return _PERMISSIONS_YAML_TEMPLATE.format(
            reward_modification_allowed=reward_flag,
            env_modification_allowed=env_flag,
            eval_modification_allowed=eval_flag,
            offline_data_allowed=offline_flag,
            imitation_allowed=imitation_flag,
        )

    def _render_state_json(self) -> str:
        state = {
            "version": "1.0.0",
            "project_name": self.ob.project_name or self.project_dir.name,
            "initialized_at": self._ts,
            "current_phase": "research",
            "current_branch": None,
            "best_run_id": None,
            "best_metric_value": None,
            "best_metric_name": self.ob.success_metric or "reward",
            "total_runs": 0,
            "kept_runs": 0,
            "discarded_runs": 0,
            "crashed_runs": 0,
            "last_updated": self._ts,
            "active_workers": [],
            "queue": [],
            "flags": {
                "research_refresh_due": False,
                "human_review_required": False,
                "incident_count": 0,
            },
        }
        return json.dumps(state, indent=2, ensure_ascii=False) + "\n"

    def _render_dashboard_config_json(self) -> str:
        cfg = {
            "project_name": self.ob.project_name or self.project_dir.name,
            "metric": self.ob.success_metric or "mean_eval_reward",
            "target_value": self.ob.target_value,
            "refresh_interval_seconds": 5,
            "max_chart_points": 500,
        }
        return json.dumps(cfg, indent=2, ensure_ascii=False) + "\n"

    # ------------------------------------------------------------------
    # Markdown renderers
    # ------------------------------------------------------------------

    def _render_claude_md(self) -> str:
        ob = self.ob
        algo_text = ob.algorithm_recommendations or "PPO, DQN, A2C"
        return _CLAUDE_MD_TEMPLATE.format(
            project_name=ob.project_name or self.project_dir.name,
            environment=ob.environment,
            obs_type=ob.obs_type,
            action_space=ob.action_space,
            objective=ob.objective,
            success_metric=ob.success_metric,
            target_value=ob.target_value,
            wall_clock_budget=ob.wall_clock_budget,
            compute_budget=ob.compute_budget,
            other_information=ob.other_information or "None provided.",
            algorithm_recommendations=algo_text,
            timestamp=self._ts,
        )

    def _render_agent_md(self) -> str:
        ob = self.ob
        return _AGENT_MD_TEMPLATE.format(
            project_name=ob.project_name or self.project_dir.name,
            success_metric=ob.success_metric,
            environment=ob.environment,
            timestamp=self._ts,
        )

    def _render_orchestrator_md(self) -> str:
        ob = self.ob
        return _ORCHESTRATOR_MD_TEMPLATE.format(
            project_name=ob.project_name or self.project_dir.name,
            timestamp=self._ts,
        )

    def _render_user_spec_md(self) -> str:
        ob = self.ob
        algo_block = ob.algorithm_recommendations or "- PPO (good default for continuous and discrete)\n- DQN (discrete action spaces)\n- A2C (lighter compute footprint)"
        return _USER_SPEC_MD_TEMPLATE.format(
            project_name=ob.project_name or self.project_dir.name,
            environment=ob.environment,
            obs_type=ob.obs_type,
            action_space=ob.action_space,
            objective=ob.objective,
            success_metric=ob.success_metric,
            target_value=ob.target_value,
            wall_clock_budget=ob.wall_clock_budget,
            compute_budget=ob.compute_budget,
            other_information=ob.other_information or "None provided.",
            modification_policy=ob.modification_policy,
            offline_data_policy=ob.offline_data_policy,
            imitation_policy=ob.imitation_policy,
            algorithm_recommendations=algo_block,
        )

    def _render_non_negotiable_md(self) -> str:
        ob = self.ob
        if ob.user_rules:
            rules_block = "\n".join(f"- {rule}" for rule in ob.user_rules)
        else:
            rules_block = "_(No additional user-defined rules were specified during onboarding.)_"
        return _NON_NEGOTIABLE_MD_TEMPLATE.format(
            user_rules_as_bullet_list=rules_block,
            timestamp=self._ts,
        )

    def _render_implementation_plan_md(self) -> str:
        ob = self.ob
        return _IMPLEMENTATION_PLAN_MD_TEMPLATE.format(
            project_name=ob.project_name or self.project_dir.name,
            environment=ob.environment,
            success_metric=ob.success_metric,
            target_value=ob.target_value,
            algorithm_recommendations=ob.algorithm_recommendations or "PPO, DQN, A2C",
            timestamp=self._ts,
        )


# ---------------------------------------------------------------------------
# Console helpers (avoid importing from cli to keep this module standalone)
# ---------------------------------------------------------------------------

def _info(msg: str) -> None:
    print(f"  [~] {msg}")


def _warn(msg: str) -> None:
    print(f"  [!] {msg}", file=sys.stderr)


# ===========================================================================
# Template strings — all use str.format() placeholders
# ===========================================================================

# ---------------------------------------------------------------------------
# .drl_autoresearch/policy.yaml
# ---------------------------------------------------------------------------

_POLICY_YAML_TEMPLATE = """\
# DRL AutoResearch Policy Configuration
# Generated: {timestamp}
# Project: {project_name}

permission_mode: open  # locked | prompted | bootstrap-only | open | project-only

# Actions that ALWAYS require confirmation regardless of mode
always_confirm:
  - edit_eval
  - edit_reward
  - global_install
  - exceed_compute

# Actions that are ALWAYS blocked
always_block: []  # Populated from NON_NEGOTIABLE_RULES.md

# Compute budget
compute_budget:
  max_gpu_hours: {compute_budget}
  max_vram_gb: {max_vram}
  max_parallel_workers: 1

# Eval integrity
eval_integrity:
  protect_eval_code: true
  require_fixed_eval_seed: true
  min_eval_episodes: 10

# Maximum number of experiments to run before requiring human review.
max_experiments: 100

# Primary optimisation metric (higher is better unless flip_sign: true).
metric:
  name: mean_eval_reward
  flip_sign: false

# Phase transition thresholds.
phases:
  baseline_min_runs: 3
  focused_tuning_improvement_threshold: 0.05  # 5% relative improvement
  convergence_patience: 10                     # runs without improvement

# Hypothesis generation.
hypothesis:
  max_concurrent: 4
  sources:
    - internal_history
    - research_knowledge_base

# Checkpoint retention policy.
checkpoints:
  keep_top_n: 5
  delete_on_discard: false

# Enable cooldown between stuck-triggered refreshes.
refresh_cooldown_enabled: true
"""

# ---------------------------------------------------------------------------
# .drl_autoresearch/hardware.yaml (default when no detection data available)
# ---------------------------------------------------------------------------

_HARDWARE_YAML_DEFAULT = """\
# DRL AutoResearch — hardware configuration
# Auto-populated by `drl-autoresearch doctor`.
# Run `drl-autoresearch doctor` to detect and fill real values.

gpu:
  backend: auto          # auto | cuda | rocm | mps | cpu
  devices: []            # leave empty to use all available
  mixed_precision: true

cpu:
  workers: auto          # auto -> os.cpu_count()

memory:
  max_ram_gb: auto       # auto -> total system RAM
  vram_limit_gb: null    # null -> no limit

detected:
  cpu_model: unknown
  cpu_cores: 0
  cpu_threads: 0
  ram_gb: 0.0
  has_gpu: false
  gpu_count: 0
  gpus: []
  cuda_available: false
  cuda_version: null
  multi_gpu_allowed: false
"""

# ---------------------------------------------------------------------------
# .drl_autoresearch/python_env.yaml
# ---------------------------------------------------------------------------

_PYTHON_ENV_YAML_TEMPLATE = """\
# DRL AutoResearch — Python environment configuration
# Detected Python: {python_version}

python_min_version: "3.10"
detected_python_version: "{python_version}"

required_packages:
  - torch
  - numpy
  - pandas

optional_packages:
  - aiohttp      # required for dashboard
  - textual     # required for interactive TUI
  - psutil       # improved hardware detection
  - pyyaml       # YAML config support
  - tensorboard  # training visualisation
  - gymnasium    # RL environments
"""

# ---------------------------------------------------------------------------
# .drl_autoresearch/permissions.yaml
# ---------------------------------------------------------------------------

_PERMISSIONS_YAML_TEMPLATE = """\
# DRL AutoResearch — agent permissions
# Controls what autonomous agents are allowed to do.
# These values are enforced by the PolicyEngine (drl-autoresearch check).

allow:
  modify_hyperparameters: true
  add_new_experiments: true
  delete_checkpoints: false     # set true to allow pruning old checkpoints
  modify_policy_yaml: false     # set true to allow self-modifying policy
  internet_access: false
  run_shell_commands: false     # for custom skill shell commands
  modify_reward_function: {reward_modification_allowed}
  modify_environment: {env_modification_allowed}
  modify_eval_protocol: {eval_modification_allowed}
  use_offline_data: {offline_data_allowed}
  use_imitation_learning: {imitation_allowed}

require_human_approval:
  - delete_checkpoints
  - modify_policy_yaml
  - run_shell_commands
  - exceed_compute_budget
  - modify_eval_protocol
  - global_package_install
"""

# ---------------------------------------------------------------------------
# logs/incidents.md placeholder
# ---------------------------------------------------------------------------

_INCIDENTS_TEMPLATE = """\
# Incident Log

This file records policy violations, unexpected behaviours, and agent
incidents that require human review.

Format: one entry per incident, newest at top.

---

<!-- Incidents are appended here by the PolicyEngine. -->
"""

# ---------------------------------------------------------------------------
# logs/handoffs.md placeholder
# ---------------------------------------------------------------------------

_HANDOFFS_TEMPLATE = """\
# Agent Handoff Log

Each time an agent session ends, it must write a handoff entry here
BEFORE stopping.  The next agent reads this file first.

## Handoff Entry Format

```
## Handoff — {timestamp} — Agent: {agent_id}

### What I did this session
- <bullet>

### Current state
- Phase: <phase>
- Best run: <run_id> with <metric>=<value>
- Queue length: <N>

### What to do next
- <bullet>

### Open questions / blockers
- <bullet>

### Files changed this session
- <path>: <why>
```

---

<!-- Handoff entries are appended here. -->
"""

# ===========================================================================
# CLAUDE.md
# ===========================================================================

_CLAUDE_MD_TEMPLATE = """\
# CLAUDE.md — Agent Operating Guide

**Project**: {project_name}
**Generated**: {timestamp}
**Read this file first.  Every time.  Before touching any code.**

---

## 1. Project Overview

| Field | Value |
|---|---|
| Environment | {environment} |
| Observation Type | {obs_type} |
| Action Space | {action_space} |
| Objective | {objective} |
| Success Metric | {success_metric} |
| Target Value | {target_value} |
| Wall-Clock Budget | {wall_clock_budget} |
| Compute Budget | {compute_budget} GPU-hours |
| Other Information | {other_information} |
| Recommended Algorithms | {algorithm_recommendations} |

You are a **research engineer** running an autonomous DRL research loop.
Your job is to improve `{success_metric}` on the held-out eval set.
Nothing else matters.  Train reward is a proxy, not the goal.

---

## 2. Hard Rules (Non-Negotiable)

The full rules are in `NON_NEGOTIABLE_RULES.md`.  Key ones:

1. **Never silently change the eval protocol.**  Any change to evaluation
   logic must be logged, justified, and flagged for human review.
2. **Never claim improvement without running a proper eval.**  Train reward
   does not count.  Min 10 episodes on the held-out eval set.
3. **Never exceed the compute budget** without explicit human approval.
   Check `policy.yaml` before scheduling a long run.
4. **Never mix multiple major changes into one experiment** unless
   explicitly designed as a bundle.  One change per run.
5. **Never install packages globally.**  Use project-local environment only.
6. **Never reuse random seeds across experimental conditions.**
7. **Always log what changed and why** before running an experiment.
8. **Always revert changes that do not improve eval metric** (unless
   flagged as exploratory).
9. **Never silently modify reward logic.**  Reward changes require the
   `edit_reward` approval gate.
10. **Log every incident** in `logs/incidents.md`, even minor ones.

Violations are enforced by `drl-autoresearch check`.  Critical violations
pause the loop and require human intervention.

---

## 3. Experiment Loop Procedure

Follow this procedure for every experiment.  Do not skip steps.

### Step 0: Read context
- Read `logs/handoffs.md` — understand what the previous agent did.
- Read `IMPLEMENTATION_PLAN.md` — understand the current research plan.
- Run `drl-autoresearch status` — check current phase and queue.
- Read `logs/experiment_registry.tsv` (last 10 rows) for recent results.

### Step 1: Formulate hypothesis
- State the hypothesis in one sentence: "Changing X from A to B will
  improve `{success_metric}` because Y."
- Hypothesis must be falsifiable and testable with one experiment.
- Log hypothesis in `logs/experiment_registry.tsv` *before* running.

### Step 2: Design the experiment
- Change exactly ONE thing from the current best configuration.
- Use a new random seed different from all previous runs.
- Keep eval protocol, eval seed, and eval episode count identical to
  all other runs (see Eval Integrity section).
- If the change touches reward logic: run `drl-autoresearch check
  --action edit_reward` first.
- If the change touches eval logic: run `drl-autoresearch check
  --action edit_eval` first.

### Step 3: Run the experiment
- Record start time and hardware in the registry.
- Set a wallclock timeout consistent with the compute budget.
- Monitor for crashes — a crashed run must be logged with
  `status=crashed`, never silently retried.

### Step 4: Evaluate
- Evaluate the final policy on the **held-out eval set** only.
- Minimum 10 eval episodes; use the fixed eval seed from `policy.yaml`.
- Record `{success_metric}` in `logs/experiment_registry.tsv`.
- Record variance/std if running multiple eval seeds.

### Step 5: Keep or discard
- If eval metric improved vs. current best: mark `status=kept`,
  update `best_run_id` in `state.json`.
- If eval metric did not improve: mark `status=discarded`, revert
  all code changes to the best-known-good config.
- If exploratory run: mark `status=exploratory` and document findings.

### Step 6: Update plan
- Update `IMPLEMENTATION_PLAN.md` with the outcome.
- If result suggests a new hypothesis, add it to the queue.
- If stuck (N consecutive discards without improvement), trigger
  `drl-autoresearch research` to do a literature refresh.

### Step 7: Handoff
- Write a handoff entry in `logs/handoffs.md` before stopping.
- Include: what you ran, the result, next recommended action.

---

## 4. File Modification Policy

### Files you MAY modify freely:
- `train.py` — training script (hyperparams, architecture, optimiser)
- `IMPLEMENTATION_PLAN.md` — update plan with findings
- `logs/experiment_registry.tsv` — append new rows
- `logs/handoffs.md` — append handoff entries
- `logs/incidents.md` — append incident entries
- `.drl_autoresearch/state.json` — update via `drl-autoresearch` CLI
- `skills/*.md` — add new skills or improve existing ones

### Files you MAY modify with `drl-autoresearch check` approval:
- Reward function code — requires `edit_reward` gate
- Evaluation code — requires `edit_eval` gate
- `.drl_autoresearch/policy.yaml` — requires `modify_policy_yaml` gate
- `.drl_autoresearch/permissions.yaml` — requires human approval

### Files you must NEVER modify:
- `NON_NEGOTIABLE_RULES.md`
- `CLAUDE.md` (this file)
- `AGENT.md`
- `.drl_autoresearch/policy.yaml` without `check` approval
- Any file outside the project directory

---

## 5. Recording Results

Every experiment must have exactly one row in
`logs/experiment_registry.tsv` with these columns:

| Column | Description |
|---|---|
| `run_id` | UUID or sequential ID |
| `timestamp` | ISO-8601 UTC |
| `hypothesis` | One-sentence description |
| `params_json` | JSON dict of changed parameters |
| `metric_name` | `{success_metric}` |
| `metric_value` | Numeric result |
| `status` | `kept` / `discarded` / `crashed` / `exploratory` |
| `notes` | Free text: anomalies, observations |

Do NOT round metric values.  Record the raw number.

---

## 6. Logging Conventions

- **experiment_registry.tsv**: One row per run, tab-separated.
- **logs/handoffs.md**: One entry per agent session.  Append only.
- **logs/incidents.md**: One entry per incident.  Append only.
- **state.json**: Managed by the `drl-autoresearch` CLI.  Edit via CLI.
- **TensorBoard logs**: Store in `logs/tensorboard/<run_id>/`.
- **Checkpoints**: Store in `logs/checkpoints/<run_id>/`.
- **Videos**: Store in `logs/videos/<run_id>/`.

Logging is mandatory.  An unlogged experiment never happened.

---

## 7. When to Stop vs. Keep Going

**Keep going if:**
- You have budget remaining and a plausible hypothesis.
- The last run showed positive signal (even if below threshold).
- You are in the baseline or early experimenting phase.

**Stop and write a handoff if:**
- Compute budget is near or at limit.
- Wall-clock budget is near limit.
- You have hit `convergence_patience` consecutive non-improvements.
- You need human input (approval gate, incident, ambiguous spec).
- You have made a significant discovery worth flagging.
- Something feels wrong that you cannot diagnose alone.

**Never stop without writing a handoff.**

---

## 8. How to Use Skills Files

Skills files in `skills/` are step-by-step playbooks for specific
research tasks.  Use them when you encounter the corresponding situation.

| Skill File | When to Use |
|---|---|
| `investigate.md` | Regression, unexpected failure, suspicious result |
| `reward_shaping.md` | Considering adding potential-based shaping |
| `exploration.md` | Entropy collapse, poor coverage, agent stuck |
| `env_diagnostics.md` | Suspected environment bug or misconfiguration |
| `ablation.md` | Need to isolate which change drove improvement |
| `checkpoint_selection.md` | Selecting final policy for evaluation |
| `compute_budgeting.md` | Deciding run length, batch size, parallelism |
| `mid_training_research.md` | Current direction is stuck; need new ideas |

Each skill file defines its output format.  Follow it exactly.

---

## 9. Escalation Protocol

Escalate to human (stop loop, write incident, await response) when:

1. **Critical policy violation detected** — by you or by PolicyEngine.
2. **Compute budget exceeded** — even by a small amount.
3. **Eval protocol compromised** — any change to eval code or seed.
4. **Reward hacking suspected** — train reward diverges from eval.
5. **Environment bug found** — results are non-reproducible or
   physically implausible.
6. **Conflicting results** — two well-controlled runs give opposite
   conclusions.
7. **Hardware failure / OOM** — repeated crashes suggest infra issue.

To escalate:
1. Append to `logs/incidents.md` with full context.
2. Set `flags.human_review_required = true` in `state.json`
   (via `drl-autoresearch check`).
3. Write a handoff entry in `logs/handoffs.md`.
4. Stop the loop.

---

## 10. Evaluation Integrity Requirements

These requirements protect the validity of all results.

- **Fixed eval seed**: Use `eval_integrity.require_fixed_eval_seed` from
  `policy.yaml`.  Never change it mid-project.
- **Fixed eval code**: The evaluation function must be identical across
  all runs.  Any change invalidates comparisons.
- **Min eval episodes**: At least `eval_integrity.min_eval_episodes`
  (default: 10).  More is better for high-variance environments.
- **Held-out eval set**: Training and evaluation must use separate
  seeds / episode initializations.  No overlap.
- **No cherry-picking**: Report mean eval reward, not max.  Report
  standard deviation or confidence interval if N > 1 seed.
- **No post-hoc metric selection**: The target metric is
  `{success_metric}`.  Do not switch to a different metric
  mid-project without updating `USER_SPEC.md` and logging the reason.

---

## 11. Resource Limits

| Resource | Limit |
|---|---|
| GPU hours (total) | {compute_budget} |
| Max parallel workers | 1 (see policy.yaml) |
| Checkpoint storage | Top 5 kept (see policy.yaml) |
| Wall-clock budget | {wall_clock_budget} |

If a single run will consume more than 20% of the remaining GPU budget,
require `exceed_compute` approval before launching it.

Monitor `state.json` for current spend (update after each run).

---

## 12. Handoff Protocol

Before stopping, write an entry to `logs/handoffs.md` with:

1. **Session summary**: What you attempted and the outcomes.
2. **Current best**: Run ID, metric value, config path.
3. **Current phase**: From `state.json`.
4. **Queue**: Next experiments in queue (from `state.json`).
5. **Open hypotheses**: Ideas you did not have time to test.
6. **Blockers**: Anything that requires human attention.
7. **Files changed**: List every file you modified and why.
8. **Recommended next action**: Concrete, specific.

The next agent reads this before doing anything else.  Make it useful.
"""

# ===========================================================================
# AGENT.md
# ===========================================================================

_AGENT_MD_TEMPLATE = """\
# AGENT.md — Agent Behaviour Standard

**Project**: {project_name}
**Generated**: {timestamp}

This document defines the expected behaviour for ALL autonomous agents
operating in this project, regardless of model (Claude, Codex, Gemini, etc.).

---

## 1. Role Definition

You are a **research engineer**, not a random code mutator.

Your role has three responsibilities:
1. **Propose** scientifically valid hypotheses about what will improve
   `{success_metric}` on `{environment}`.
2. **Test** those hypotheses with rigorous, controlled experiments.
3. **Learn** from results and update the research plan accordingly.

You are NOT here to:
- Make the code "cleaner" for its own sake.
- Try techniques because they are novel or fashionable.
- Make many changes at once to ship results faster.
- Optimise train reward rather than eval metric.

---

## 2. Scientific Discipline Requirements

Every experiment must meet these standards before its result is used to
update the plan:

**Validity**
- One independent variable changed per experiment (unless explicitly
  designing a bundle ablation).
- All other variables held constant vs. the current best configuration.
- New random seed, different from all previous runs.

**Reproducibility**
- Full config (params_json) logged before the run starts.
- Code state (git hash or diff) recorded if applicable.
- Hardware and software versions recorded.

**Measurement**
- Eval metric measured on the held-out set only.
- Minimum 10 eval episodes.
- Report mean and std dev — never report a single episode.

**Interpretation**
- Use effect size and variance, not just point estimate.
- A 1% improvement within 1 std dev of noise is not evidence of
  improvement.
- If two seeds give contradictory results, run more seeds before
  concluding anything.

---

## 3. Valid Evidence of Improvement

Improvement is **only** claimed when ALL of the following are true:

1. `{success_metric}` increased on the held-out eval set.
2. The improvement is larger than the standard deviation of baseline
   measurements (i.e., signal > noise).
3. The eval protocol (seed, episode count, eval function) is identical
   to all comparison runs.
4. The result is logged in `experiment_registry.tsv` with status=kept.

Single-episode results, train reward increases, and proxy metrics alone
do NOT constitute evidence of improvement.

---

## 4. Handoff Protocol

Before ending a session, write an entry to `logs/handoffs.md`:

```
## Handoff — <ISO timestamp> — Agent: <agent_id>

### What I did
- <bullet point per experiment>

### Results
- Run <id>: <metric>=<value>, status=<kept/discarded/crashed>

### Current best
- Run ID: <id>
- {success_metric}: <value>
- Config: <path or params_json>

### Recommended next action
<One specific, concrete action>

### Open hypotheses (untested)
- <hypothesis>

### Blockers / human attention needed
- <item or "none">

### Files changed
- <path>: <why>
```

If you stop without writing a handoff, the loop is broken for the next
agent.  This is a protocol violation.

---

## 5. Incident Reporting Protocol

Append to `logs/incidents.md` immediately when:

- A policy check is triggered (even if approved).
- An experiment crashes unexpectedly.
- You observe a result that contradicts a previous conclusion.
- You suspect reward hacking.
- You suspect an environment bug.
- Any action requires human approval but you cannot wait.
- You observe a resource budget overrun.

Incident entry format:
```
## Incident — <ISO timestamp>

**Severity**: low | medium | high | critical
**Type**: policy_violation | crash | anomaly | reward_hacking |
          env_bug | resource_overrun | other
**Description**: <what happened>
**Evidence**: <what data supports this>
**Action taken**: <what you did>
**Human action required**: yes | no
**Status**: open | resolved
```

---

## 6. Never-Do List

These actions are unconditionally prohibited.  No exception.

- **Never trust train reward alone.**  It is a proxy.  Always evaluate
  on the held-out eval set before drawing conclusions.

- **Never claim improvement from noisy evidence.**  If the result is
  within one std dev of baseline noise, it is not an improvement.

- **Never silently change eval logic.**  Any change to the eval
  function, eval seed, or eval episode count must be logged,
  approved via `drl-autoresearch check --action edit_eval`, and
  flagged in the handoff.

- **Never silently change reward logic.**  Any change to reward
  shaping, reward scaling, or reward function must be logged and
  approved via `drl-autoresearch check --action edit_reward`.

- **Never silently exceed resource budgets.**  Check remaining GPU
  hours in `policy.yaml` and `state.json` before launching any run.
  If a run will exceed budget, get `exceed_compute` approval first.

- **Never install packages outside policy.**  Run
  `drl-autoresearch check --action global_install` before any
  `pip install` or `conda install`.  Never install globally.

- **Never mix multiple major changes into one experiment** unless
  explicitly designed and documented as a bundle ablation.  Mixed
  changes make results uninterpretable.

- **Never delete or modify `NON_NEGOTIABLE_RULES.md`.**

- **Never modify `CLAUDE.md` or `AGENT.md`** without human approval.

- **Never stop without writing a handoff entry** in `logs/handoffs.md`.

---

## 7. Communication Standards

When reporting results or status:

- Be precise about numbers.  "Improved slightly" is not acceptable.
  Write "mean eval reward increased from 187.3 ± 4.1 to 193.8 ± 3.2
  over 20 eval episodes."
- Distinguish between train metric and eval metric explicitly.
- State uncertainty.  If variance is high, say so.
- Separate facts from hypotheses.  "The result was X" is a fact.
  "This probably happened because Y" is a hypothesis.
- Report negative results with the same detail as positive ones.
  Knowing what does not work is valuable.
- Flag surprises immediately — do not bury unexpected results in notes.
"""

# ===========================================================================
# ORCHESTRATOR.md
# ===========================================================================

_ORCHESTRATOR_MD_TEMPLATE = """\
# ORCHESTRATOR.md — Orchestrator Decision Logic

**Project**: {project_name}
**Generated**: {timestamp}

This document describes how the DRL AutoResearch orchestrator makes
decisions.  Agents and humans can read this to understand automated
behaviour.

---

## 1. Phase Definitions and Transitions

The orchestrator tracks a `current_phase` in `state.json`.

### Phase: `research`
**Entry**: Project initialised, no baseline established.
**Activity**: Literature review, hypothesis generation from prior knowledge,
environment diagnostics.
**Exit condition**: At least one baseline run completed and logged.

### Phase: `baseline`
**Entry**: After `research` phase confirms direction.
**Activity**: Run the recommended starting algorithm with default
hyperparameters across `baseline_min_runs` seeds to establish variance
and baseline `{success_metric}`.
**Exit condition**: `baseline_min_runs` (default 3) seeds completed with
consistent results.

### Phase: `experimenting`
**Entry**: After baseline is established.
**Activity**: Systematic hypothesis testing.  One change per run.
**Exit condition**: Either focused improvement direction identified OR
`convergence_patience` consecutive non-improvements.

### Phase: `focused_tuning`
**Entry**: When a technique shows >5% relative improvement over baseline.
**Activity**: Fine-grained hyperparameter search around the winning
configuration.  May run more seeds per hypothesis.
**Exit condition**: Marginal improvements <1% for `convergence_patience`
consecutive runs.

### Phase: `ablation`
**Entry**: When approaching target metric or preparing final report.
**Activity**: Ablation study to confirm which changes contributed.
**Exit condition**: Ablation complete and results documented.

### Phase: `converged`
**Entry**: When target value is reached or budget is exhausted.
**Activity**: Final checkpoint selection, documentation.
**Exit condition**: Human sign-off.

---

## 2. Experiment Scheduling Logic

The orchestrator maintains a `queue` in `state.json`.

Queue entry schema:
```json
{{
  "experiment_id": "uuid",
  "hypothesis": "string",
  "params_json": {{}},
  "priority": 0,
  "depends_on": null,
  "estimated_gpu_hours": 1.0,
  "created_at": "ISO timestamp"
}}
```

**Scheduling rules (in priority order):**

1. Always run pending experiments before generating new ones.
2. Sort queue by `priority` descending, then `created_at` ascending.
3. Never run an experiment whose `depends_on` run is still pending.
4. Never schedule an experiment that would exceed `compute_budget.max_gpu_hours`.
5. Never run more than `max_parallel_workers` experiments simultaneously.

---

## 3. Research Refresh Triggers

The orchestrator triggers `drl-autoresearch research` (literature refresh)
when ANY of the following conditions are met:

- `convergence_patience` consecutive experiments with no improvement.
- Current phase is `research` and no experiments have been queued yet.
- `flags.research_refresh_due` is `true` in `state.json`.
- Human runs `drl-autoresearch research` manually.

During a research refresh, the orchestrator:
1. Reads `IMPLEMENTATION_PLAN.md` for current direction.
2. Queries the research knowledge base with the current best config and
   results.
3. Generates new hypotheses based on returned literature.
4. Adds highest-priority hypotheses to the queue.
5. Updates `IMPLEMENTATION_PLAN.md`.
6. Sets `flags.research_refresh_due = false`.

---

## 4. Rule Enforcement Process

Before executing any action, the orchestrator calls:
```
drl-autoresearch check --action <action_name> --details <json>
```

Exit codes:
- `0` — approved, proceed.
- `1` — blocked, do not proceed.
- `2` — requires human confirmation (pause loop).

When code `2` is returned:
1. Append to `logs/incidents.md`.
2. Set `flags.human_review_required = true` in `state.json`.
3. Stop the experiment loop.
4. Write a handoff entry in `logs/handoffs.md`.
5. Await human confirmation before resuming.

---

## 5. Worker Management

Workers are tracked in `state.json` under `active_workers` (list of
worker IDs or process identifiers).

Rules:
- Maximum `compute_budget.max_parallel_workers` workers (default: 1).
- Each worker owns one experiment at a time.
- If a worker crashes, it must be removed from `active_workers` and
  the experiment logged with `status=crashed`.
- Zombie workers (registered but not responding) are logged as incidents.

---

## 6. How to Interpret state.json

```
state.json field          Meaning
────────────────────────  ─────────────────────────────────────────────────
version                   Schema version; do not edit manually.
project_name              Human-readable project name.
initialized_at            When `drl-autoresearch init` was run.
current_phase             Active research phase (see Phase Definitions).
current_branch            Git branch (if applicable).
best_run_id               run_id of the current best checkpoint.
best_metric_value         Numeric eval metric for best_run_id.
best_metric_name          Which metric (e.g. mean_eval_reward).
total_runs                All runs ever attempted (incl. crashed).
kept_runs                 Runs that improved eval metric and were kept.
discarded_runs            Runs that did not improve and were reverted.
crashed_runs              Runs that failed to complete.
last_updated              Timestamp of last state.json modification.
active_workers            Workers currently running experiments.
queue                     Pending experiments (ordered list).
flags                     Boolean/counter flags for orchestrator logic.
  research_refresh_due    True if a literature refresh should run next.
  human_review_required   True if the loop is paused awaiting human.
  incident_count          Total incidents logged this session.
```

When reading state.json, the most important fields are:
1. `current_phase` — tells you where you are.
2. `best_run_id` + `best_metric_value` — tells you the current standard.
3. `queue` — tells you what to do next.
4. `flags.human_review_required` — if true, do not run experiments.
"""

# ===========================================================================
# USER_SPEC.md
# ===========================================================================

_USER_SPEC_MD_TEMPLATE = """\
# DRL Research Specification

## Project: {project_name}

## Environment
- **Simulator**: {environment}
- **Observation Type**: {obs_type}
- **Action Space**: {action_space}
- **Objective**: {objective}

## Success Criteria
- **Real Metric**: {success_metric}
- **Target Value**: {target_value}
- **Wall-Clock Budget**: {wall_clock_budget}
- **Compute Budget**: {compute_budget} GPU-hours
- **Other Information**: {other_information}

## Constraints
- Allowed modification scope: {modification_policy}
- Offline data policy: {offline_data_policy}
- Imitation learning policy: {imitation_policy}

## Token Philosophy
- Keep outputs compact and token-efficient.
- Prefer targeted reads and line pointers over loading whole files.
- Treat compact artifacts as navigators; open original sources only when needed.

## Algorithm Starting Point
Based on your environment, the recommended starting algorithms are:

{algorithm_recommendations}

## Notes

_Add any additional context about the project, environment quirks,
known failure modes, or domain knowledge here._
"""

# ===========================================================================
# NON_NEGOTIABLE_RULES.md
# ===========================================================================

_NON_NEGOTIABLE_MD_TEMPLATE = """\
# Non-Negotiable Rules

These rules MUST NOT be broken under any circumstances.
The PolicyEngine enforces these automatically.
This file must not be modified by agents.

---

## User-Defined Rules

{user_rules_as_bullet_list}

---

## System-Level Rules (Always Active)

- NEVER silently change the evaluation protocol (eval function, eval seed,
  eval episode count).  Any change requires `edit_eval` approval and must
  be logged.
- NEVER claim improvement without evaluating on the held-out eval set with
  a minimum of 10 episodes.
- NEVER exceed the compute budget without explicit user approval via the
  `exceed_compute` gate.
- NEVER install packages in the global Python environment.  All installs
  must target the project-local environment.
- NEVER mix multiple major changes into one experiment (unless explicitly
  designed as a bundle ablation).
- NEVER reuse random seeds across different experimental conditions.
- NEVER silently change reward logic.  All reward modifications require
  `edit_reward` approval and must be logged.
- ALWAYS log what changed and why, in `experiment_registry.tsv`, before
  running an experiment.
- ALWAYS revert changes that do not improve the eval metric, unless the
  run was explicitly flagged as exploratory.
- ALWAYS write a handoff entry in `logs/handoffs.md` before ending a
  session.
- ALWAYS log incidents in `logs/incidents.md` immediately when they occur.
- NEVER delete or modify this file (`NON_NEGOTIABLE_RULES.md`).

---

## Enforcement

These rules are enforced by the PolicyEngine.
Call `drl-autoresearch check --action <action>` before any gated action.
Violations are logged to `logs/incidents.md`.
Critical violations set `flags.human_review_required = true` in
`state.json` and pause the loop until human confirmation is received.

## Last Updated
{timestamp}
"""

# ===========================================================================
# IMPLEMENTATION_PLAN.md
# ===========================================================================

_IMPLEMENTATION_PLAN_MD_TEMPLATE = """\
# Implementation Plan

**Project**: {project_name}
**Environment**: {environment}
**Target**: {success_metric} >= {target_value}
**Generated**: {timestamp}

_This file is managed by the orchestrator and research agents.
Update it after each experiment with findings and revised direction._

---

## Current Phase

`research`

---

## Research Direction

_To be filled by the research agent after the initial literature review._

Recommended starting algorithms: {algorithm_recommendations}

---

## Phase 1: Baseline

**Goal**: Establish a reproducible baseline to measure all future
improvements against.

**Tasks**:
- [ ] Run default algorithm with default hyperparameters, 3 different
      random seeds
- [ ] Record mean ± std of `{success_metric}` across 3 seeds
- [ ] Identify high-variance hyperparameters from the baseline runs
- [ ] Log all results in `experiment_registry.tsv`

**Success criterion**: 3 baseline runs completed with consistent variance.

---

## Phase 2: Experimenting

**Goal**: Systematically explore the hypothesis space to find techniques
that improve beyond baseline.

**Active Hypotheses**:

_Hypotheses are added here by the orchestrator and research agents.
Each hypothesis must be falsifiable and testable in one experiment._

| ID | Hypothesis | Expected Effect | Priority | Status |
|----|-----------|----------------|----------|--------|
| H1 | _(To be filled after baseline)_ | - | - | pending |

**Completed Experiments**:

_Append here after each experiment._

---

## Phase 3: Focused Tuning

**Goal**: Fine-tune the best configuration found in Phase 2.

_Activated when a technique shows >5% relative improvement over baseline._

---

## Phase 4: Ablation

**Goal**: Confirm which changes are actually responsible for the
improvement, and document for future reference.

_Activated when approaching target metric or compute budget._

---

## Key Findings

_Append findings after each significant experiment._

---

## Revision History

| Date | Author | Change |
|------|--------|--------|
| {timestamp} | scaffold | Initial plan generated |
"""

# ===========================================================================
# Skill files
# ===========================================================================

_SKILL_INVESTIGATE = """\
# Skill: Investigate — Diagnosing Regressions and Unexpected Failures

## Purpose
Use this skill when a previously working configuration stops performing as
expected, when a new experiment produces results that contradict a prior
conclusion, or when you observe a result that is physically implausible or
statistically impossible.  This is the systematic debugging playbook for DRL
research.

## Prerequisites
- The `run_id` of the failing/regressed run.
- The `run_id` of the last known-good run.
- Access to both runs' `params_json` from `experiment_registry.tsv`.
- Checkpoint files for both runs (if kept).

## Step-by-Step Procedure

### Step 1: Confirm the regression is real
Before investigating, verify the result is not within noise:
- Check the std dev of baseline eval runs.
- If the "regression" is within 1 std dev, it may be noise.
- Run one more evaluation of both configs to confirm.

### Step 2: Diff the configurations
- Extract `params_json` for both runs from `experiment_registry.tsv`.
- List every parameter that differs.
- If more than one parameter differs, the cause is ambiguous — you may
  need to run bisection experiments.

### Step 3: Check for data loading and environment changes
- Verify the environment version is the same.
- Check if any dependencies were updated between the two runs.
- Verify the eval seed and eval protocol are unchanged.
- Check if observation normalization statistics were reset.

### Step 4: Evaluate across seeds
- Evaluate both configs with 3 different random seeds.
- If one config is consistently better across all seeds, the signal is real.
- If results are mixed, you have a high-variance problem, not a regression.

### Step 5: Bisection (for multi-change regressions)
If multiple parameters changed:
- Create a run that restores parameters one at a time from the failing config
  back to the good config.
- The parameter whose restoration fixes the regression is the culprit.

### Step 6: Check training stability
- Plot the training reward curve: did it diverge, collapse, or plateau early?
- Check gradient norms if available.
- Check entropy: did exploration collapse before the policy was trained?
- Check loss: are policy loss, value loss, entropy loss in expected ranges?

### Step 7: Check the environment itself
- Run `skills/env_diagnostics.md` if you suspect an environment bug.
- Verify episode lengths, reward scaling, observation ranges.

## Key Diagnostic Questions
1. Is this a real regression or noise?
2. What exactly changed between the good and bad run?
3. Is the eval protocol identical for both runs?
4. Is the environment deterministic given the same seed?
5. Did training converge normally or collapse early?

## Common Failure Modes
- **Seed sensitivity**: The good result was an outlier, not robust.
- **Dependency drift**: A package updated silently changed behaviour.
- **Eval protocol change**: The eval function or seed was inadvertently
  changed.
- **Observation normalizer reset**: Running stats reset between runs
  caused distribution shift.
- **Learning rate too high**: Instability in fine-tuned runs.
- **Batch size change**: Changed effective learning rate and update
  frequency simultaneously.

## Output Format
After completing this investigation, log in `experiment_registry.tsv`
and append to the handoff:

```
Investigation Result — Run: <run_id>
Root cause: <one sentence>
Evidence: <data that supports the conclusion>
Action taken: <what was done to resolve or document>
Confidence: low | medium | high
```
"""

_SKILL_REWARD_SHAPING = """\
# Skill: Reward Shaping — When and How to Modify the Reward Signal

## Purpose
Use this skill before and during any modification to the reward function.
Reward shaping is a high-risk, potentially high-reward intervention.  Done
incorrectly it leads to reward hacking, where the agent achieves high shaped
reward while failing on the true objective.  This playbook ensures shaping is
justified, implemented correctly, and validated rigorously.

## Prerequisites
- Evidence that the current unshaped reward is causing a specific learning
  problem (sparse reward, misleading reward, reward too noisy).
- Approval via `drl-autoresearch check --action edit_reward` is required
  before any code change.
- Baseline `{success_metric}` values from multiple seeds.

## Step-by-Step Procedure

### Step 1: Diagnose the problem reward is causing
Do NOT shape reward unless you can clearly articulate the problem.  Valid
reasons:
- **Sparse reward**: Agent rarely receives non-zero reward; exploration fails.
- **Delayed reward**: Credit assignment horizon is too long for current
  algorithm.
- **Noisy reward**: High-variance reward slows learning and increases
  required sample count.
- **Misleading proxy**: Current reward leads agent to a local optimum that
  does not reflect the true objective.

Document the problem in your hypothesis entry.

### Step 2: Choose a shaping approach
For potential-based reward shaping (provably policy-invariant):
- Define a potential function Phi(s) that captures desired intermediate
  progress.
- Shaped reward: r'(s,a,s') = r(s,a,s') + gamma * Phi(s') - Phi(s)
- This guarantees the optimal policy under r' is the same as under r.

For other approaches (not policy-invariant, higher risk):
- Auxiliary reward terms (subgoal achievement, distance-to-goal)
- Normalising or clipping reward
- Intrinsic motivation (see exploration.md instead)

### Step 3: Implement in isolation
- Create the shaped reward as an additive wrapper, NOT by modifying the
  base environment reward.
- The unmodified base reward must remain accessible for eval.
- Use a shaping_weight parameter so you can ablate it to zero.

### Step 4: Verify policy-invariance (if applicable)
- Run the agent trained on r' and evaluate it on the original r metric.
- If eval metric improves, the shaping genuinely helps learning.
- If eval metric stays the same or drops, shaping may not help or may hurt.

### Step 5: Test shaped vs. unshaped reward
- Run identical configs except shaping_weight=1.0 vs shaping_weight=0.0.
- Evaluate both on the original eval metric.
- The shaped version must outperform unshoped on the REAL metric,
  not just on the shaped reward.

### Step 6: Monitor for reward hacking
- Watch for train reward diverging from eval metric (a key hacking signal).
- If shaped reward is high but true eval metric is low, the agent is
  hacking the shape.
- Check qualitative behaviour (video or episode replay if possible).

## Key Diagnostic Questions
1. What specific learning failure justifies adding shaping?
2. Is the shaping potential-based (provably safe) or ad hoc (risky)?
3. Does the agent's shaped-reward improvement transfer to the true metric?
4. Is there any sign of reward hacking (divergence between proxy and eval)?

## Common Failure Modes
- **Reward hacking**: Agent exploits the shape, ignores true objective.
- **Shaped reward dominates**: Weight too high; agent ignores the true
  reward signal entirely.
- **Goal misspecification**: The potential function encodes the wrong
  notion of progress.
- **Overfitting to shaping**: Policy performs well with shape but poorly
  without it — does not transfer.

## Output Format
```
Reward Shaping Experiment — Run: <run_id>
Shaping type: <potential-based | auxiliary | normalisation | other>
Shape definition: <one sentence or formula>
Shaping weight: <value>
Train reward improvement: <delta>
Eval metric improvement: <delta>
Reward hacking observed: yes | no | unclear
Decision: keep | discard | investigate further
```
"""

_SKILL_EXPLORATION = """\
# Skill: Exploration — Diagnosing and Fixing Exploration Failures

## Purpose
Use this skill when the agent fails to explore effectively: it converges
prematurely to a suboptimal policy, fails to visit important regions of the
state space, shows entropy collapse in the policy distribution, or makes no
progress despite many training steps.

## Prerequisites
- Training logs and/or TensorBoard data for the failing run.
- Access to policy entropy logs (or ability to add entropy logging).
- Baseline entropy values from a healthy training run (if available).

## Step-by-Step Procedure

### Step 1: Measure entropy
Policy entropy is the primary diagnostic for exploration failures.
- For discrete action spaces: H(pi) = -sum(pi(a|s) * log(pi(a|s)))
  Should be close to log(|A|) early in training, declining gradually.
- For continuous policies (Gaussian): H = 0.5 * log(2*pi*e*sigma^2)
  Should remain non-trivially positive until late training.
- **Entropy collapse**: entropy drops to near-zero in the first
  10-20% of training.  This is pathological.

### Step 2: Check the action distribution
- Plot the marginal distribution of actions taken during training.
- Are some actions never taken?  Are action probabilities degenerate?
- For continuous spaces: is the policy's std dev collapsing to zero?

### Step 3: Check reward signal coverage
- Plot cumulative reward distribution over episodes.  Is it bimodal
  (all-or-nothing)?  Is non-zero reward extremely rare (<1% of steps)?
- Sparse reward environments require exploration bonuses or curriculum.

### Step 4: Diagnose root cause
Common causes:
- **Entropy coefficient too low**: Increase entropy bonus (PPO: ent_coef).
- **Learning rate too high**: Premature convergence.
- **KL constraint too tight** (PPO): Max KL or clip_ratio too conservative.
- **Value function overfit**: Critic too confident early, preventing
  exploration in actor-critic methods.
- **Reward scale too large**: Agent becomes overly conservative.
- **Poor initialisation**: Network weights cause deterministic outputs
  from initialisation.

### Step 5: Choose an intervention
Ordered by safety and simplicity:
1. **Increase entropy bonus**: `ent_coef` 0.01 → 0.05 (safe, one param).
2. **Reduce learning rate**: Allow slower but more stable exploration.
3. **Add epsilon-greedy noise** (discrete): Simple coverage mechanism.
4. **Add Gaussian noise to actions** (continuous): Simple.
5. **Add intrinsic motivation** (ICM, RND, count-based): More complex,
   use only after simpler options fail.
6. **Redesign episode initialisation**: Randomise start states more
   aggressively (if environment allows).

### Step 6: Epsilon / noise schedule
If using epsilon-greedy or action noise:
- Log `epsilon` or `noise_std` in experiment params.
- Use a decay schedule (linear, cosine, or exponential).
- Never set exploration to zero before target metric is reached.

### Step 7: Validate fix
- Re-run with intervention, compare entropy curves.
- Does exploration stay higher longer?
- Does `{success_metric}` improve on eval set?

## Key Diagnostic Questions
1. What is the policy entropy at 10%, 50%, 90% of training?
2. Are there actions in the action space that are never taken?
3. What fraction of episodes receive non-zero reward?
4. Is the value function overfit early in training?
5. Is the reward scale appropriate (roughly unit-scale is ideal)?

## Common Failure Modes
- **Premature entropy collapse**: Entropy drops before any meaningful
  reward is obtained.  Fix: increase entropy coefficient.
- **Action distribution degenerate at init**: Bad weight initialisation.
  Fix: orthogonal init, smaller init scale.
- **Exploration inconsistent across seeds**: High seed variance suggests
  exploration is fragile.  Fix: more robust exploration strategy.
- **Intrinsic reward dominates extrinsic**: Bonus too large.  Fix: scale
  down intrinsic coefficient.

## Output Format
```
Exploration Diagnostic — Run: <run_id>
Entropy at 10%: <value>
Entropy at 50%: <value>
Entropy at 90%: <value>
Root cause: <one sentence>
Intervention: <what was changed>
Result: <eval metric before vs after>
```
"""

_SKILL_ENV_DIAGNOSTICS = """\
# Skill: Environment Diagnostics — Debugging the Environment Itself

## Purpose
Use this skill when you suspect the environment is misconfigured, buggy, or
behaving in ways that make learning impossible or misleading.  Environment
bugs are among the most insidious issues in DRL because they silently corrupt
all experiments that use the environment — producing confident but wrong
results.

## Prerequisites
- Access to the environment code or configuration.
- At least one example episode (observations, actions, rewards, dones).
- Expected ranges for observations and rewards from the task specification
  (`USER_SPEC.md`).

## Step-by-Step Procedure

### Step 1: Verify observation range and distribution
- Sample 1000 observations from random-policy rollouts.
- Check min, max, mean, std for each observation dimension.
- Expected: most observation dimensions should have mean ≈ 0, std ≈ 1
  after normalisation.  Before normalisation, check against spec.
- **Red flags**: NaN/Inf values; observations outside physically plausible
  range; all-zero or all-constant observations.

### Step 2: Verify reward scale
- Collect 100 random-policy episodes and compute reward statistics.
- Expected scale: rewards between -10 and +10 per step (rough guideline).
  Very large rewards (|r| > 100) or very small rewards (|r| < 0.001) can
  destabilise training.
- Check for NaN/Inf rewards.

### Step 3: Verify action handling
- Confirm the environment clips or normalises actions correctly.
- For continuous spaces: check min/max action values are handled gracefully.
- For discrete spaces: verify all actions are valid and none silently fail.
- Apply extreme actions (min, max, random) and verify physically plausible
  state transitions.

### Step 4: Verify episode length and termination conditions
- Record episode lengths from 100 random-policy rollouts.
- Are episodes terminating too early (< 10 steps)?  Too late (never done)?
- Verify that terminal states are marked correctly (`done=True`).
- Check if truncation (timeout) vs. true termination is handled correctly.

### Step 5: Verify reset behaviour
- Call `env.reset()` 100 times and examine initial observations.
- Are initial states sufficiently diverse (stochastic reset)?
- Does the environment reset completely, or do some state variables persist
  across episodes?

### Step 6: Verify determinism (if required)
- Seed the environment with the same seed twice.
- Run identical actions for 100 steps.
- Observations and rewards should be bit-for-bit identical.
- If they differ, the environment has hidden stochasticity or a seeding bug.

### Step 7: Check observation normalisation pipeline
- If using a running normaliser (e.g. VecNormalize): check that statistics
  are not being updated during eval.
- Verify that eval uses the same normaliser statistics as training.
- Check for normaliser drift over very long training runs.

### Step 8: Stress test edge cases
- Zero-velocity / zero-input episode.
- Max-input episode.
- Episode that terminates immediately.
- Verify none cause crashes or degenerate rewards.

## Key Diagnostic Questions
1. Are observation values within expected physical range?
2. Is reward scale appropriate for the learning algorithm?
3. Are episode terminations correct and consistent?
4. Is the environment truly deterministic given a fixed seed?
5. Does the normalisation pipeline apply identically during train and eval?

## Common Failure Modes
- **VecNormalize updating during eval**: Eval results are affected by the
  eval episodes themselves — not reproducible.
- **Reward clipping asymmetry**: Negative rewards clipped more than positive,
  creating a biased signal.
- **Observation range mismatch**: Normalisation trained on one range but
  applied to another (e.g. after env version change).
- **Wrong `done` flag**: Episode does not terminate when it should
  (or terminates too early).
- **Stale state on reset**: Hidden variables not reset, causing
  non-Markovian observations.

## Output Format
```
Environment Diagnostic — Date: <ISO timestamp>
Observation range check: PASS | FAIL | WARN
  Issues: <list or "none">
Reward scale check: PASS | FAIL | WARN
  mean ± std: <value>
Action handling check: PASS | FAIL
Episode length check: PASS | FAIL | WARN
  mean / min / max: <values>
Reset behaviour: PASS | FAIL
Determinism check: PASS | FAIL | SKIPPED
Normaliser check: PASS | FAIL | N/A
Root cause (if any): <one sentence>
Recommendation: <action to take>
```
"""

_SKILL_ABLATION = """\
# Skill: Ablation — Scientific Ablation Protocol

## Purpose
Use this skill when you need to determine which of several changes is
responsible for an improvement, when preparing a final report, or when an
experiment with bundled changes succeeded and you need to understand which
components contributed.  Ablation studies are the scientific method for DRL
improvements.

## Prerequisites
- A "full" configuration that achieves a clear improvement over baseline.
- Clear enumeration of all changes from baseline to full config.
- Sufficient compute budget for `N_changes * N_seeds` experiments.
- Baseline eval metric with variance from `experiment_registry.tsv`.

## Step-by-Step Procedure

### Step 1: Enumerate the changes
List every parameter or code change between baseline and the full config.
Be exhaustive — even changes you believe are irrelevant must be included.

Example:
```
Change A: learning rate 3e-4 → 1e-4
Change B: added entropy bonus ent_coef=0.01
Change C: network hidden size 64 → 256
Change D: added reward normalisation
```

### Step 2: Design the ablation structure
For N changes, you have two main designs:

**Additive ablation** (add one change at a time from baseline):
- Run baseline
- Run baseline + A
- Run baseline + A + B
- Run baseline + A + B + C
- ...
Advantage: Shows contribution order.  Cost: N * seeds experiments.

**Leave-one-out ablation** (remove one change at a time from full):
- Run full config
- Run full - A
- Run full - B
- Run full - C
- ...
Advantage: Shows net contribution.  Cost: (N+1) * seeds experiments.

Choose based on budget.  Leave-one-out is preferred when the full config
is known to work and you need to isolate contributions.

### Step 3: Determine required number of seeds
Minimum seeds depends on baseline variance:
- Variance of eval metric from baseline (sigma_baseline)
- Expected effect size (delta)
- Minimum detectable effect: delta > 2 * sigma_baseline / sqrt(N_seeds)
- Rule of thumb: use at least 3 seeds; use 5 seeds for final claims.

### Step 4: Run all ablation experiments
- Each ablation run must use a different random seed.
- All eval protocol settings must be identical (seed, episode count, function).
- Log every run in `experiment_registry.tsv`.

### Step 5: Analyse results
For each ablation variant, report:
- Mean eval metric
- Standard deviation across seeds
- Delta vs. full config
- Is delta > 1 std dev of noise?

If removing change X drops performance by > 1 std dev, X is important.
If removing X has no effect, X is not contributing.
If results are noisy, run more seeds before concluding.

### Step 6: Interpret carefully
- **Interaction effects**: Two changes that each do nothing alone may be
  important together.  Additive ablation will miss this.
- **Cancellation**: A harmful change may be masked by a beneficial one.
- **Stochasticity**: With few seeds, apparent differences may be noise.

### Step 7: Document findings
Update `IMPLEMENTATION_PLAN.md` with the ablation conclusions.

## Key Diagnostic Questions
1. Is the full configuration robustly better than baseline (multiple seeds)?
2. Which changes are necessary vs. incidental?
3. Are there interaction effects between changes?
4. Are there any changes that are actually harmful?

## Common Failure Modes
- **Too few seeds**: Noise masquerades as effect.  Use min 3 seeds, ideally 5.
- **Confounded ablation**: Two things changed at once (e.g. batch size and
  learning rate), making attribution impossible.
- **Eval protocol drift**: Different ablations use slightly different eval
  settings, making direct comparison invalid.
- **Missing a change**: Forgetting a small code change means the "baseline"
  config is not actually the baseline.

## Output Format
```
Ablation Summary — Date: <ISO timestamp>

Full config: Run <run_id>, metric = <value> ± <std>
Baseline:    Run <run_id>, metric = <value> ± <std>

| Change Removed | Mean Metric | Std | Delta vs Full | Significant? |
|---------------|-------------|-----|--------------|-------------|
| Change A      | <value>     | <s> | <delta>      | yes/no/maybe|
| Change B      | ...         | ... | ...          | ...         |

Conclusions:
- Critical changes: <list>
- Non-contributing changes: <list>
- Potentially harmful changes: <list>
- Interaction effects suspected: <list>
```
"""

_SKILL_CHECKPOINT_SELECTION = """\
# Skill: Checkpoint Selection — Selecting, Validating, and Maintaining Checkpoints

## Purpose
Use this skill when selecting the final policy checkpoint for reporting or
deployment, when implementing checkpoint pruning to manage disk space, or
when auditing whether the current "best checkpoint" was selected validly.
Incorrect checkpoint selection is a form of overfitting on eval data and
invalidates reported results.

## Prerequisites
- A set of candidate checkpoints from one or more runs.
- Access to the eval set (distinct from any data used during training).
- `experiment_registry.tsv` to trace the history of checkpoint evaluations.

## Step-by-Step Procedure

### Step 1: Classify checkpoints by source
Checkpoints fall into categories based on how they were selected:
- **Last checkpoint**: Simply the final model state.
- **Best-during-training checkpoint**: Saved based on training reward.
  These are biased — avoid using for final eval.
- **Best-on-eval checkpoint** (correct): Evaluated on the held-out eval set
  without modifying the eval set between evaluations.

Only best-on-eval checkpoints are valid for final reporting.

### Step 2: Avoid checkpoint selection data leakage
If you evaluate K checkpoints on the same eval set and report the best, you
are implicitly overfitting to the eval set.  To avoid this:
- Use a fixed held-out eval set for all comparisons (same seed, same episodes).
- If evaluating many checkpoints, use a separate validation set for selection
  and keep the eval set for final measurement only.
- Report: "selected based on validation set performance at <N> eval episodes."

### Step 3: Evaluate candidates
For each candidate checkpoint:
- Run eval with the fixed eval seed from `policy.yaml`.
- Minimum `eval_integrity.min_eval_episodes` episodes.
- Record: mean reward, std dev, min, max, success rate (if applicable).
- Use identical eval function and normaliser state for all candidates.

### Step 4: Apply selection criteria
Standard selection rule: best mean eval reward, provided variance is not
pathologically high (e.g. std > mean suggests brittle policy).

Additional criteria to consider:
- **Consistency**: Does the policy perform well across all eval episodes,
  or does it have high variance?  A lower-mean but more consistent policy
  may be preferable.
- **Wall-clock performance**: Later checkpoints are not always better due
  to potential overfitting.
- **Robustness**: If you have multiple eval seeds, prefer checkpoints that
  perform consistently across all of them.

### Step 5: Final validation
After selecting the best checkpoint:
- Re-evaluate it on the held-out eval set with at least 10 episodes.
- This is the number you report.  It must come from a clean, unseen eval.
- If you have been touching the eval set throughout training, you cannot
  claim this number is unbiased — flag this as a limitation.

### Step 6: Checkpoint pruning
When disk quota is a concern:
- Always keep: the current best checkpoint, the baseline checkpoint.
- Keep top N per `policy.yaml` (default: 5).
- Require `delete_checkpoints` approval before deleting.
- Log deletions in `logs/incidents.md` (low severity).

## Key Diagnostic Questions
1. Was this checkpoint selected based on the same eval data used for
   final reporting? (If yes: data leakage risk.)
2. How many eval episodes was this checkpoint evaluated on?
3. Is the std dev of eval performance within acceptable range?
4. Was the eval protocol (seed, function, normaliser) identical for all
   candidates?

## Common Failure Modes
- **Eval set contamination**: Selecting checkpoint by running many evals
  on the same eval set effectively optimises for that set.
- **Using training curve best**: The highest-training-reward checkpoint is
  not the best eval policy.
- **Different normaliser states**: Evaluating checkpoints with different
  observation normaliser statistics, making comparisons invalid.
- **Reporting best single episode**: High-variance policy appears excellent
  in one episode but fails in general.

## Output Format
```
Checkpoint Selection — Date: <ISO timestamp>

Candidates evaluated: <N>
Selection method: best mean eval reward on held-out set

Selected checkpoint:
  run_id: <run_id>
  epoch/step: <checkpoint step>
  eval metric (mean ± std, N episodes): <value>
  eval seed: <seed>
  eval episodes: <N>
  data leakage risk: none | low | high

Runner-up checkpoint:
  run_id: <run_id>
  eval metric: <value>

Decision rationale: <one sentence>
```
"""

_SKILL_COMPUTE_BUDGETING = """\
# Skill: Compute Budgeting — Reasoning About Compute Trade-Offs

## Purpose
Use this skill before scheduling a long experiment, when deciding batch size
and learning rate, when considering whether to run more seeds vs. more
configurations, or when the compute budget is running low and prioritisation
is needed.  Wasting compute on long runs of bad hypotheses is one of the
most common and avoidable failures in DRL research.

## Prerequisites
- The total compute budget from `policy.yaml` (max_gpu_hours).
- Current spent budget from `state.json` or experiment logs.
- Expected training time for one run (from previous runs or estimate).

## Step-by-Step Procedure

### Step 1: Always run a small probe first
Before committing to a full-length run, run a short probe:
- Use 10-20% of the full training steps.
- If the probe shows zero signal (flat reward, entropy collapse, NaN),
  abort and iterate on the hypothesis.
- A short probe costs 10-20% of budget; an uninformative full run costs 100%.

Rule: Never run a full-length experiment without a probe unless you have
strong prior evidence the configuration will work.

### Step 2: Estimate compute cost before scheduling
For each experiment in the queue:
- Estimate GPU-hours = (steps * step_time_seconds) / 3600
- Check remaining budget = max_gpu_hours - sum(completed_run_gpu_hours)
- If experiment cost > 20% of remaining budget: require `exceed_compute`
  approval before scheduling.
- Log estimated cost in `params_json` of the registry entry.

### Step 3: Batch size vs. learning rate trade-off
A common compute trap: large batch sizes reduce gradient noise but require
proportionally adjusted learning rates and more compute per update.
- Doubling batch size requires roughly doubling learning rate (linear
  scaling rule) to maintain similar convergence speed.
- More compute per update + same wall clock = fewer updates = slower
  learning if learning rate is not adjusted.
- Rule: when exploring batch size, always explore learning rate jointly.

### Step 4: Know when to stop early
Implement and respect early stopping:
- If reward has not improved in `convergence_patience` evaluations AND
  is clearly below baseline, the run is likely going to fail.
- Stop early rather than running to completion.
- Log as `status=discarded (early stop)` with reason.

Signs that justify early stopping:
- Entropy collapsed to near-zero in first 20% of training.
- Training reward is below random policy after 30% of steps.
- Loss values are NaN or exploding.
- Eval metric after 25% of steps is worse than baseline.

### Step 5: Seed count vs. configuration count trade-off
Given a fixed compute budget B and per-run cost C:
- You can run B/C configurations with 1 seed each.
- Or B/(2C) configurations with 2 seeds each.
- Or B/(3C) configurations with 3 seeds each.

Early exploration phases: 1-2 seeds per configuration (maximise coverage).
Focused tuning phase: 3-5 seeds (maximise confidence per decision).

### Step 6: GPU memory optimisation
If OOM errors are limiting batch size or model size:
- Enable mixed precision (policy.yaml: `mixed_precision: true`).
- Use gradient checkpointing for large networks.
- Reduce replay buffer size (for off-policy methods).
- Reduce minibatch size while keeping total batch size the same.
- Clear GPU cache between runs.

### Step 7: Prioritise the queue
When budget is low, sort the queue:
1. High-confidence hypotheses (supported by prior evidence).
2. Cheap experiments (low GPU-hours).
3. Experiments that unblock other experiments.
4. Ablation experiments (can be deferred if budget is critical).
5. Exploratory experiments (lowest priority when budget is low).

## Key Diagnostic Questions
1. What fraction of total budget has been spent?
2. Does this experiment require a probe first?
3. Is the estimated cost proportional to the expected information gain?
4. Should I run more seeds or more configurations?
5. Are there early stopping criteria in place?

## Common Failure Modes
- **No probe before full run**: Wastes full budget on uninformative run.
- **Not tracking cumulative cost**: Budget exhausted silently.
- **Running all seeds before filtering**: Spending 5x budget on a bad idea.
- **No early stopping**: Long runs of clearly failing configurations.
- **Forgetting GPU memory constraints**: Scheduling runs that OOM.

## Output Format
```
Compute Budget Check — Date: <ISO timestamp>

Total budget: <N> GPU-hours
Spent to date: <N> GPU-hours
Remaining: <N> GPU-hours

Experiment being planned: <hypothesis>
Estimated cost: <N> GPU-hours (<X>% of remaining)
Probe first: yes | no | already done
Approval required: yes | no

Queue (next 3):
  1. <hypothesis> — est. <N> GPU-hours
  2. <hypothesis> — est. <N> GPU-hours
  3. <hypothesis> — est. <N> GPU-hours

Total queue cost: <N> GPU-hours
Budget will last: <N> more experiments
```
"""

_SKILL_MID_TRAINING_RESEARCH = """\
# Skill: Mid-Training Research — Triggering and Conducting a Research Refresh

## Purpose
Use this skill when the current research direction is exhausted (consecutive
non-improvements equal `convergence_patience`), when a surprising result
suggests an unexplored approach, or when the current algorithms are hitting
fundamental limitations that literature may have solutions for.  This is the
process for updating the research direction mid-project based on new
information.

## Prerequisites
- Evidence that current direction is stuck (quantitative, not subjective).
- The current best configuration and its eval metric.
- Summary of all hypotheses already tested (from `experiment_registry.tsv`).
- Sufficient remaining budget to justify a direction change.

## Step-by-Step Procedure

### Step 1: Confirm the direction is genuinely stuck
Before triggering a research refresh, verify the quantitative evidence:
- Count consecutive non-improvements against `convergence_patience`.
- Verify the non-improvements are not due to noise (check variance).
- Check whether the failure is fundamental (algorithm limitation) or
  tactical (bad hyperparameter search).
- If the problem is tactical, iterate on hyperparameters before refreshing.

### Step 2: Write a "current state" summary
Before searching for new directions, document exactly where you are:
```
Current best: run_id=<id>, metric=<value>
Techniques tried: <list>
Techniques that helped: <list>
Techniques that did not help: <list>
Suspected bottleneck: <hypothesis about root cause>
```
This summary guides the research search and prevents re-discovering
already-tried approaches.

### Step 3: Formulate research queries
Generate specific queries based on the suspected bottleneck:
- "PPO with sparse reward in <environment type>"
- "Sample efficiency improvement for <observation type>"
- "Credit assignment in <episode length> horizon environments"
- "Exploration in <action space type> action spaces"
- "<algorithm> known limitations and alternatives"

Avoid broad queries like "improve DRL".  Specific queries return
actionable results.

### Step 4: Evaluate candidate new directions
For each candidate technique found:
- Is it applicable to the current environment type?
- Does it address the specific bottleneck identified in Step 2?
- What is the implementation complexity?  (Prefer simple changes.)
- Is there prior empirical evidence of it working on similar environments?
- Is it compatible with the current constraints (reward mod allowed? etc.)?

Score candidates on: relevance, expected gain, implementation cost.

### Step 5: Design cheap probe experiments for top candidates
Before committing to a full direction change:
- For each top candidate, design a minimal probe (10-20% of full steps).
- The probe must be falsifiable: what signal in the probe justifies
  proceeding to a full run?
- Example: "If the probe shows >10% improvement in eval metric at 20%
  of training, proceed with full run."

### Step 6: Update IMPLEMENTATION_PLAN.md
- Document the new direction and the evidence that motivated it.
- Keep the history of previous directions — do not delete.
- Assign priorities to new hypotheses.
- Update `state.json` queue with new experiments.

### Step 7: Set `research_refresh_due = false`
After the refresh is complete, clear the flag:
- Update `flags.research_refresh_due` to `false` in `state.json`.
- Log the refresh in `logs/handoffs.md`.

## Key Diagnostic Questions
1. Is this direction truly exhausted, or is it a hyperparameter problem?
2. What is the specific bottleneck (exploration, credit assignment, etc.)?
3. What is the remaining compute budget — is a direction change justified?
4. Are there any techniques already found in the literature that address
   this specific bottleneck?
5. Can the new direction be cheaply probed before committing?

## Common Failure Modes
- **Premature refresh**: Giving up on a direction before fully exploring it.
- **Re-discovering known failures**: Not consulting prior results before
  generating new hypotheses.
- **Direction change with no probe**: Committing full budget to unvalidated
  new approach.
- **Too broad a search**: Finding many unrelated techniques, none directly
  applicable.
- **Ignoring constraints**: Proposing techniques that violate
  reward/env/eval modification constraints from `USER_SPEC.md`.

## Output Format
```
Research Refresh — Date: <ISO timestamp>

Trigger: convergence_patience exceeded | surprise result | manual

Current state:
  Best metric: <value>
  Techniques tried: <N>
  Suspected bottleneck: <one sentence>

Research queries used:
  - <query 1>
  - <query 2>

Candidate new directions:
  1. <technique>: relevance=<H/M/L>, expected gain=<H/M/L>, cost=<H/M/L>
  2. <technique>: ...

Selected direction: <technique>
Rationale: <one sentence>
Probe design: <what the probe will test and pass/fail criterion>
Budget allocated: <N> GPU-hours

IMPLEMENTATION_PLAN.md updated: yes
Queue updated: yes
```
"""


# ===========================================================================
# Console helpers (module-level, no dependency on cli.py)
# ===========================================================================

def _info(msg: str) -> None:  # noqa: F811
    print(f"  [~] {msg}")


def _warn(msg: str) -> None:  # noqa: F811
    print(f"  [!] {msg}", file=sys.stderr)
