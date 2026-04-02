"""
drl_autoresearch.core.init
--------------------------
Scaffold a new DRL AutoResearch project directory.

Creates:
  <project_dir>/
    .drl_autoresearch/
      policy.yaml
      hardware.yaml
      python_env.yaml
      permissions.yaml
      state.json
    logs/
      experiment_registry.tsv
    skills/
      .gitkeep
    NON_NEGOTIABLE_RULES.md
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

from drl_autoresearch.cli import console
from drl_autoresearch.core.state import ProjectState


# ---------------------------------------------------------------------------
# Default file templates
# ---------------------------------------------------------------------------

_POLICY_YAML = textwrap.dedent("""\
    # DRL AutoResearch — policy configuration
    # Modify these values to control the autonomous research loop.

    # Maximum number of experiments to run before stopping.
    max_experiments: 100

    # Primary optimisation metric (higher is better unless flip_sign: true).
    metric:
      name: reward
      flip_sign: false

    # Phase transition thresholds.
    phases:
      baseline_min_runs: 3
      focused_tuning_improvement_threshold: 0.05  # 5 % relative improvement
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
""")

_HARDWARE_YAML = textwrap.dedent("""\
    # DRL AutoResearch — hardware configuration
    # Auto-populated by `drl-autoresearch doctor`.

    gpu:
      backend: auto          # auto | cuda | rocm | mps | cpu
      devices: []            # leave empty to use all available
      mixed_precision: true

    cpu:
      workers: auto          # auto → os.cpu_count()

    memory:
      max_ram_gb: auto       # auto → total RAM
      vram_limit_gb: null    # null → no limit
""")

_PYTHON_ENV_YAML = textwrap.dedent("""\
    # DRL AutoResearch — Python environment configuration

    python_min_version: "3.10"

    required_packages:
      - torch
      - numpy
      - pandas

    optional_packages:
      - aiohttp      # required for dashboard
      - questionary  # required for interactive TUI
""")

_PERMISSIONS_YAML = textwrap.dedent("""\
    # DRL AutoResearch — agent permissions
    # Controls what autonomous agents are allowed to do.

    allow:
      modify_hyperparameters: true
      add_new_experiments: true
      delete_checkpoints: false     # set true to allow pruning old checkpoints
      modify_policy_yaml: false     # set true to allow self-modifying policy
      internet_access: false
      run_shell_commands: false     # custom skill shell commands

    require_human_approval:
      - delete_checkpoints
      - modify_policy_yaml
      - run_shell_commands
""")

_RULES_MD = textwrap.dedent("""\
    # NON_NEGOTIABLE RULES
    <!-- This file is enforced by drl-autoresearch check and may not be overridden
         by policy.yaml or agent instructions. -->

    ## Safety

    1. **Never delete source code** outside of `logs/` or `skills/`.
    2. **Never modify this file** (`NON_NEGOTIABLE_RULES.md`).
    3. **Never disable the permission check** (`drl-autoresearch check`).
    4. **Never exfiltrate experiment data** to external services without explicit
       user configuration in `permissions.yaml`.

    ## Experiment integrity

    5. All experiment results **must be logged** to `logs/experiment_registry.tsv`
       before being used to update the plan.
    6. Metrics must be recorded **as-measured**; no post-hoc manipulation.
    7. A run that crashes must be recorded with `status=crashed`; results from
       incomplete runs must not influence best-model selection.

    ## Resource limits

    8. GPU/CPU usage must stay within limits defined in `hardware.yaml`.
    9. Disk usage for checkpoints must not exceed the project-level quota
       (if set in `policy.yaml`).

    ## Human override

    10. Any action listed under `require_human_approval` in `permissions.yaml`
        **must** pause and wait for explicit human confirmation before proceeding.
""")

from drl_autoresearch.logging.registry import COLUMNS as _REGISTRY_COLUMNS
_REGISTRY_HEADER = "\t".join(_REGISTRY_COLUMNS) + "\n"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(
    project_dir: Path,
    skip_onboarding: bool = False,
    auto: bool = False,
) -> int:
    """Scaffold the project directory.

    Returns 0 on success, non-zero on failure.
    """
    project_dir = Path(project_dir).resolve()

    console(f"Initialising DRL AutoResearch project at: {project_dir}", "info")

    # Ask for confirmation unless --auto or --skip-onboarding.
    if not auto and not skip_onboarding and sys.stdin.isatty():
        try:
            answer = input(
                f"  Scaffold project in '{project_dir}'? [Y/n] "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            console("Aborted.", "warning")
            return 130
        if answer not in ("", "y", "yes"):
            console("Aborted.", "warning")
            return 1

    config_dir = project_dir / ".drl_autoresearch"
    logs_dir   = project_dir / "logs"
    skills_dir = project_dir / "skills"

    try:
        # Create directory tree.
        for d in (config_dir, logs_dir, skills_dir):
            d.mkdir(parents=True, exist_ok=True)

        # Write config files (never overwrite existing ones).
        _write_if_missing(config_dir / "policy.yaml",      _POLICY_YAML)
        _write_if_missing(config_dir / "hardware.yaml",    _HARDWARE_YAML)
        _write_if_missing(config_dir / "python_env.yaml",  _PYTHON_ENV_YAML)
        _write_if_missing(config_dir / "permissions.yaml", _PERMISSIONS_YAML)

        # Hard rules file at project root.
        _write_if_missing(project_dir / "NON_NEGOTIABLE_RULES.md", _RULES_MD)

        # Experiment registry.
        registry = logs_dir / "experiment_registry.tsv"
        if not registry.exists():
            registry.write_text(_REGISTRY_HEADER, encoding="utf-8")
            console(f"Created {registry.relative_to(project_dir)}", "success")

        # Skills placeholder.
        gitkeep = skills_dir / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()

        # Initialise / update state.json.
        state = ProjectState.load(project_dir)
        state.save()
        console(
            f"State written to {(config_dir / 'state.json').relative_to(project_dir)}",
            "success",
        )

    except OSError as exc:
        console(f"File-system error: {exc}", "error")
        return 1

    console("Project initialised successfully.", "success")
    console("Next step: run `drl-autoresearch doctor` to verify your environment.", "info")
    return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_if_missing(path: Path, content: str) -> None:
    """Write *content* to *path* only if it does not already exist."""
    if path.exists():
        console(f"Skipped (already exists): {path.name}", "warning")
        return
    path.write_text(content, encoding="utf-8")
    console(f"Created {path.name}", "success")
