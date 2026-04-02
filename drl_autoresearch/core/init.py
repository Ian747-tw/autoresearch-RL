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

import json
import sys
import textwrap
from pathlib import Path

from drl_autoresearch.cli import console
from drl_autoresearch.core.onboarding import OnboardingFlow
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

_BUNDLED_SKILL_FILES = (
    "investigate.md",
    "reward_shaping.md",
    "exploration.md",
    "env_diagnostics.md",
    "ablation.md",
    "checkpoint_selection.md",
    "compute_budgeting.md",
    "mid_training_research.md",
)

from drl_autoresearch.logging.registry import COLUMNS as _REGISTRY_COLUMNS
_REGISTRY_HEADER = "\t".join(_REGISTRY_COLUMNS) + "\n"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(
    project_dir: Path,
    skip_onboarding: bool = False,
    auto: bool = False,
    plugin: str | None = None,
    skill_pack: str | None = None,
    project_mode: str | None = None,
) -> int:
    """Scaffold the project directory.

    Parameters
    ----------
    project_dir:
        Root of the target project.
    skip_onboarding:
        Skip the interactive questionnaire.
    auto:
        Fully non-interactive mode; implies skip_onboarding.
    plugin:
        ``"cc"`` | ``"codex"`` | ``"both"`` | ``None``.
        When None and not auto, the user is prompted interactively.
        When None and auto, both plugins are installed.
    skill_pack:
        ``"drl"`` keeps the bundled DRL playbooks.
        ``"custom"`` removes the bundled DRL playbooks from the target project
        and installs a backend prompt for generating a compact custom pack.
    project_mode:
        ``"build"`` for from-scratch design/build before normal training loops.
        ``"improve"`` for optimizing an existing working model.

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
    interactive = sys.stdin.isatty() and not auto
    skip_flow = skip_onboarding or not sys.stdin.isatty()

    try:
        # Create directory tree.
        for d in (config_dir, logs_dir, skills_dir):
            d.mkdir(parents=True, exist_ok=True)

        onboarding_result = OnboardingFlow(
            project_dir=project_dir,
            auto=auto,
            skip=skip_flow,
        ).run()

        selected_skill_pack = _resolve_skill_pack(
            explicit_choice=skill_pack,
            auto=auto,
            interactive=interactive,
        )
        selected_project_mode = _resolve_project_mode(
            explicit_choice=project_mode,
            auto=auto,
            interactive=interactive,
        )

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

        _install_skill_pack(
            project_dir=project_dir,
            skills_dir=skills_dir,
            onboarding_result=onboarding_result,
            selected_skill_pack=selected_skill_pack,
        )

        _write_skill_pack_metadata(
            config_dir=config_dir,
            selected_skill_pack=selected_skill_pack,
            selected_project_mode=selected_project_mode,
        )

        # Initialise / update state.json.
        state = ProjectState.load(project_dir)
        state.flags["project_mode"] = selected_project_mode
        state.flags["build_bootstrap_complete"] = (selected_project_mode != "build")
        state.save()
        console(
            f"State written to {(config_dir / 'state.json').relative_to(project_dir)}",
            "success",
        )

    except OSError as exc:
        console(f"File-system error: {exc}", "error")
        return 1
    except Exception as exc:  # noqa: BLE001
        console(f"Init failed: {exc}", "error")
        return 1

    console("Project initialised successfully.", "success")

    # ── Plugin installation ──────────────────────────────────────────────────
    from drl_autoresearch.plugins.installer import install, prompt_and_install

    if plugin == "":           # --plugin none: skip silently
        pass
    elif plugin is not None:   # explicit choice: cc / codex / both
        install(project_dir, plugin)
    else:                      # no flag: prompt (or auto-install both)
        prompt_and_install(project_dir, auto=auto)

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


def _resolve_skill_pack(
    explicit_choice: str | None,
    auto: bool,
    interactive: bool,
) -> str:
    if explicit_choice in {"drl", "custom"}:
        return explicit_choice

    if auto or not interactive:
        return "drl"

    print()
    print("  Skill pack mode:")
    print("  [1] Directly use the provided DRL skill pack")
    print("  [2] Delete the bundled DRL skill pack in this project and build a compact custom pack")
    try:
        raw = input("  Choice [1]: ").strip() or "1"
    except (EOFError, KeyboardInterrupt):
        print()
        return "drl"

    return "custom" if raw == "2" else "drl"


def _resolve_project_mode(
    explicit_choice: str | None,
    auto: bool,
    interactive: bool,
) -> str:
    if explicit_choice in {"build", "improve"}:
        return explicit_choice

    if auto or not interactive:
        return "improve"

    print()
    print("  Project mode:")
    print("  [1] Build mode   (design architecture/reward/features from scratch before training loops)")
    print("  [2] Improve mode (existing working model, focus on improvement)")
    try:
        raw = input("  Choice [2]: ").strip() or "2"
    except (EOFError, KeyboardInterrupt):
        print()
        return "improve"

    return "build" if raw == "1" else "improve"


def _install_skill_pack(
    project_dir: Path,
    skills_dir: Path,
    onboarding_result: object,
    selected_skill_pack: str,
) -> None:
    _generate_bundled_drl_skills(project_dir, onboarding_result)

    if selected_skill_pack == "custom":
        _remove_bundled_drl_skills(skills_dir)
        _install_custom_skill_generator(project_dir, onboarding_result)
        console(
            "Selected custom skill-pack mode. Bundled DRL skills were removed from this project.",
            "info",
        )
    else:
        _remove_custom_skill_generator(project_dir)


def _generate_bundled_drl_skills(project_dir: Path, onboarding_result: object) -> None:
    try:
        from drl_autoresearch.scaffold.generator import (
            OnboardingResult as ScaffoldOnboardingResult,
            ScaffoldGenerator,
        )

        project = getattr(onboarding_result, "project", {}) or {}
        permissions = getattr(onboarding_result, "permissions", {}) or {}
        hard_rules = getattr(onboarding_result, "hard_rules", []) or []
        hardware = getattr(onboarding_result, "hardware", None)
        python_env = getattr(onboarding_result, "python_env", {}) or {}

        gen = ScaffoldGenerator(
            project_dir=project_dir,
            onboarding_result=ScaffoldOnboardingResult(
                project_name=project.get("name") or project_dir.name,
                environment=project.get("env") or "unknown",
                obs_type=project.get("obs_type") or "unknown",
                action_space=project.get("action_space") or "unknown",
                objective=project.get("objective") or "maximize episode reward",
                success_metric=project.get("success_metric") or "mean_eval_reward",
                target_value="tool_decides",
                wall_clock_budget=str(project.get("wall_clock_goal_hours") or "auto"),
                compute_budget=str(project.get("compute_budget") or "auto"),
                reward_modification_allowed=permissions.get("policy") != "locked",
                user_rules=[rule for rule in hard_rules if rule and rule != "none"],
                hardware_summary=_format_hardware_summary(hardware),
                python_env_summary=_format_python_env_summary(python_env),
            ),
        )
        gen.generate_skills()
    except Exception as exc:  # noqa: BLE001
        console(f"Skills generation skipped: {exc}", "warning")
        gitkeep = project_dir / "skills" / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()


def _remove_bundled_drl_skills(skills_dir: Path) -> None:
    for filename in _BUNDLED_SKILL_FILES:
        path = skills_dir / filename
        if path.exists():
            path.unlink()
            console(f"Removed bundled skill: skills/{filename}", "info")

    gitkeep = skills_dir / ".gitkeep"
    if not gitkeep.exists():
        gitkeep.touch()


def _install_custom_skill_generator(project_dir: Path, onboarding_result: object) -> None:
    from drl_autoresearch.plugins.installer import install_skill_generator_backend

    install_skill_generator_backend(
        project_dir=project_dir,
        context=_build_skill_generator_context(project_dir, onboarding_result),
    )


def _remove_custom_skill_generator(project_dir: Path) -> None:
    path = project_dir / ".drl_autoresearch" / "backend" / "skill_generator.md"
    if path.exists():
        path.unlink()
        console("Removed stale custom skill generator backend.", "info")


def _build_skill_generator_context(project_dir: Path, onboarding_result: object) -> str:
    project = getattr(onboarding_result, "project", {}) or {}
    permissions = getattr(onboarding_result, "permissions", {}) or {}
    hard_rules = getattr(onboarding_result, "hard_rules", []) or []

    rules_block = "\n".join(f"- {rule}" for rule in hard_rules if rule and rule != "none")
    if not rules_block:
        rules_block = "- No extra hard rules were provided during onboarding."

    return textwrap.dedent(
        f"""\
        - Project name: {project.get("name") or project_dir.name}
        - Training domain / environment: {project.get("env") or "unspecified"}
        - Observation type: {project.get("obs_type") or "unspecified"}
        - Action space: {project.get("action_space") or "unspecified"}
        - Objective: {project.get("objective") or "unspecified"}
        - Success metric: {project.get("success_metric") or "unspecified"}
        - Modification policy: {project.get("modifications_allowed") or "unspecified"}
        - Permission policy: {permissions.get("policy") or "prompted"}

        Hard rules:
        {rules_block}
        """
    ).strip()


def _write_skill_pack_metadata(
    config_dir: Path,
    selected_skill_pack: str,
    selected_project_mode: str,
) -> None:
    metadata_path = config_dir / "skill_pack.json"
    metadata = {
        "selected_pack": selected_skill_pack,
        "project_mode": selected_project_mode,
        "bundled_pack_available": True,
        "custom_generator_backend": (
            ".drl_autoresearch/backend/skill_generator.md"
            if selected_skill_pack == "custom"
            else None
        ),
        "build_plan_folder": (
            "implementation_plan"
            if selected_project_mode == "build"
            else None
        ),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    console(f"Created {metadata_path.relative_to(config_dir.parent)}", "success")


def _format_hardware_summary(hardware: object) -> str:
    if hardware is None:
        return ""

    fields = [
        f"detected_cpu_model: {getattr(hardware, 'cpu_model', 'unknown')}",
        f"detected_cpu_cores: {getattr(hardware, 'cpu_cores', 0)}",
        f"detected_cpu_threads: {getattr(hardware, 'cpu_threads', 0)}",
        f"detected_ram_gb: {getattr(hardware, 'ram_gb', 0)}",
        f"detected_has_gpu: {str(getattr(hardware, 'has_gpu', False)).lower()}",
        f"detected_gpu_count: {getattr(hardware, 'gpu_count', 0)}",
        f"detected_cuda_available: {str(getattr(hardware, 'cuda_available', False)).lower()}",
        f"detected_cuda_version: {getattr(hardware, 'cuda_version', None)}",
        f"detected_multi_gpu_allowed: {str(getattr(hardware, 'multi_gpu_allowed', False)).lower()}",
    ]
    return "\n".join(fields) + "\n"


def _format_python_env_summary(python_env: dict[str, object]) -> str:
    if not python_env:
        return ""

    lines = [
        f"package_manager: {python_env.get('package_manager', 'unknown')}",
        f"venv_path: {python_env.get('venv_path')}",
        f"python_version: {python_env.get('python_version', 'unknown')}",
        f"create_new_env: {python_env.get('create_new_env', 'auto')}",
    ]
    return "\n".join(lines) + "\n"
