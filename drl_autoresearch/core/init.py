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
import re
import shutil
import sys
import textwrap
from datetime import datetime, timezone
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

    # Minimum completed runs between stuck-triggered refreshes.
    refresh_cooldown_runs: 3
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
      - textual      # required for interactive TUI
""")

_PERMISSIONS_YAML = textwrap.dedent("""\
    # DRL AutoResearch — agent permissions
    # Controls what autonomous agents are allowed to do.
    mode: open  # locked | prompted | bootstrap-only | open | project-only

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
    refresh: bool = False,
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
    refresh:
        Remove DRL AutoResearch-managed config, state, logs, and plugin files
        in the target project before re-initializing. Does not delete user code.
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
    if refresh and not auto and sys.stdin.isatty():
        try:
            answer = input(
                "  Refresh will remove DRL AutoResearch-managed files and re-run onboarding. Continue? [y/N] "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            console("Aborted.", "warning")
            return 130
        if answer not in ("y", "yes"):
            console("Aborted.", "warning")
            return 1
    elif not auto and not skip_onboarding and sys.stdin.isatty():
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
        if refresh:
            _refresh_project_managed_files(project_dir)

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
        _sync_policy_config(config_dir, onboarding_result)
        _sync_permission_mode(config_dir, onboarding_result)

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
        _sync_onboarding_docs(project_dir=project_dir, onboarding_result=onboarding_result)
        _write_compact_spec_artifacts(project_dir=project_dir)

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

    # Best-effort environment remediation from onboarding preferences.
    try:
        from drl_autoresearch.core import doctor as doctor_mod

        console(
            "Auto-configuring Python environment from onboarding preferences...",
            "info",
        )
        doctor_mod.run(project_dir=project_dir, fix=True)
    except Exception as exc:  # noqa: BLE001
        console(f"Automatic environment setup skipped: {exc}", "warning")

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


def _sync_permission_mode(config_dir: Path, onboarding_result: object) -> None:
    """Ensure permissions config carries the onboarding-selected runtime mode."""
    permissions = getattr(onboarding_result, "permissions", {}) or {}
    selected_mode = permissions.get("policy") or "open"
    if not isinstance(selected_mode, str) or not selected_mode.strip():
        selected_mode = "open"

    yaml_path = config_dir / "permissions.yaml"
    if yaml_path.exists():
        text = yaml_path.read_text(encoding="utf-8")
        mode_line = (
            f"mode: {selected_mode}  # locked | prompted | bootstrap-only | open | project-only"
        )
        if re.search(r"(?m)^mode:\s*", text):
            updated = re.sub(r"(?m)^mode:\s*.*$", mode_line, text, count=1)
        else:
            lines = text.splitlines()
            insert_at = 0
            while insert_at < len(lines) and (
                not lines[insert_at].strip() or lines[insert_at].lstrip().startswith("#")
            ):
                insert_at += 1
            lines.insert(insert_at, mode_line)
            updated = "\n".join(lines).rstrip("\n") + "\n"
        if updated != text:
            yaml_path.write_text(updated, encoding="utf-8")
            console(f"Updated {yaml_path.name} mode to {selected_mode}", "success")
        return

    json_path = config_dir / "permissions.json"
    if json_path.exists():
        data = json.loads(json_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            data = {}
        if data.get("mode") != selected_mode:
            data["mode"] = selected_mode
            json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            console(f"Updated {json_path.name} mode to {selected_mode}", "success")


def _sync_policy_config(config_dir: Path, onboarding_result: object) -> None:
    """Ensure runtime policy config carries onboarding-selected execution values."""
    project = getattr(onboarding_result, "project", {}) or {}
    selected_cooldown = _coerce_positive_int(
        project.get("refresh_cooldown_runs"),
        default=3,
    )

    yaml_path = config_dir / "policy.yaml"
    if yaml_path.exists():
        text = yaml_path.read_text(encoding="utf-8")
        key_line = f"refresh_cooldown_runs: {selected_cooldown}"
        if re.search(r"(?m)^refresh_cooldown_runs:\s*", text):
            updated = re.sub(
                r"(?m)^refresh_cooldown_runs:\s*.*$",
                key_line,
                text,
                count=1,
            )
        else:
            updated = text.rstrip("\n") + f"\n\n# Minimum completed runs between stuck-triggered refreshes.\n{key_line}\n"
        if updated != text:
            yaml_path.write_text(updated, encoding="utf-8")
            console(
                f"Updated {yaml_path.name} refresh cooldown to {selected_cooldown} run(s)",
                "success",
            )
        return

    json_path = config_dir / "policy.json"
    if json_path.exists():
        data = json.loads(json_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            data = {}
        if data.get("refresh_cooldown_runs") != selected_cooldown:
            data["refresh_cooldown_runs"] = selected_cooldown
            json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            console(
                f"Updated {json_path.name} refresh cooldown to {selected_cooldown} run(s)",
                "success",
            )


def _coerce_positive_int(value: object, default: int) -> int:
    text = str(value or "").strip()
    if not text:
        return default
    match = re.search(r"\d+", text)
    if not match:
        return default
    try:
        parsed = int(match.group(0))
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _refresh_project_managed_files(project_dir: Path) -> None:
    """Remove DRL AutoResearch-managed files without touching user code."""
    from drl_autoresearch.plugins.installer import remove_managed_plugin_files

    config_dir = project_dir / ".drl_autoresearch"
    skills_dir = project_dir / "skills"
    logs_dir = project_dir / "logs"
    dashboard_dir = project_dir / "dashboard"

    if config_dir.exists():
        shutil.rmtree(config_dir)
        console("Removed .drl_autoresearch/", "info")

    for rel_path in (
        Path("NON_NEGOTIABLE_RULES.md"),
        Path("CLAUDE.md"),
        Path("ORCHESTRATOR.md"),
        Path("USER_SPEC.md"),
        Path("IMPLEMENTATION_PLAN.md"),
    ):
        path = project_dir / rel_path
        if path.exists():
            path.unlink()
            console(f"Removed {rel_path.as_posix()}", "info")

    removed_plugins = remove_managed_plugin_files(project_dir)
    for path in removed_plugins:
        console(f"Removed {path.relative_to(project_dir).as_posix()}", "info")

    for filename in _BUNDLED_SKILL_FILES:
        path = skills_dir / filename
        if path.exists():
            path.unlink()
            console(f"Removed skills/{filename}", "info")

    if skills_dir.is_dir():
        gitkeep = skills_dir / ".gitkeep"
        if gitkeep.exists():
            gitkeep.unlink()
            console("Removed skills/.gitkeep", "info")
        try:
            skills_dir.rmdir()
            console("Removed empty skills/", "info")
        except OSError:
            pass

    for rel_path in (
        Path("experiment_registry.tsv"),
        Path("incidents.md"),
        Path("handoffs.md"),
    ):
        path = logs_dir / rel_path
        if path.exists():
            path.unlink()
            console(f"Removed logs/{rel_path.as_posix()}", "info")

    for rel_dir in (
        Path("checkpoints"),
        Path("tensorboard"),
        Path("videos"),
    ):
        dir_path = logs_dir / rel_dir
        gitkeep = dir_path / ".gitkeep"
        if gitkeep.exists():
            gitkeep.unlink()
            console(f"Removed logs/{rel_dir.as_posix()}/.gitkeep", "info")
        if dir_path.is_dir():
            try:
                dir_path.rmdir()
                console(f"Removed empty logs/{rel_dir.as_posix()}/", "info")
            except OSError:
                pass

    if logs_dir.is_dir():
        try:
            logs_dir.rmdir()
            console("Removed empty logs/", "info")
        except OSError:
            pass

    config_path = dashboard_dir / "config.json"
    if config_path.exists():
        config_path.unlink()
        console("Removed dashboard/config.json", "info")
    if dashboard_dir.is_dir():
        try:
            dashboard_dir.rmdir()
            console("Removed empty dashboard/", "info")
        except OSError:
            pass


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
        from drl_autoresearch.scaffold.generator import ScaffoldGenerator

        scaffold_result = _build_scaffold_onboarding_result(project_dir, onboarding_result)
        gen = ScaffoldGenerator(
            project_dir=project_dir,
            onboarding_result=scaffold_result,
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


def _sync_onboarding_docs(project_dir: Path, onboarding_result: object) -> None:
    """Rewrite managed onboarding-derived docs from the latest answers."""
    from drl_autoresearch.scaffold.generator import ScaffoldGenerator

    gen = ScaffoldGenerator(
        project_dir=project_dir,
        onboarding_result=_build_scaffold_onboarding_result(project_dir, onboarding_result),
    )
    user_spec = project_dir / "USER_SPEC.md"
    user_spec.write_text(gen._render_user_spec_md(), encoding="utf-8")  # noqa: SLF001
    console(f"Updated {user_spec.relative_to(project_dir)}", "success")


def _build_scaffold_onboarding_result(project_dir: Path, onboarding_result: object) -> object:
    from drl_autoresearch.scaffold.generator import OnboardingResult as ScaffoldOnboardingResult

    project = getattr(onboarding_result, "project", {}) or {}
    hard_rules = getattr(onboarding_result, "hard_rules", []) or []
    hardware = getattr(onboarding_result, "hardware", None)
    python_env = getattr(onboarding_result, "python_env", {}) or {}

    return ScaffoldOnboardingResult(
        project_name=project.get("name") or project_dir.name,
        environment=project.get("env") or "unknown",
        obs_type=project.get("obs_type") or "unknown",
        action_space=project.get("action_space") or "unknown",
        objective=project.get("objective") or "maximize episode reward",
        success_metric=project.get("success_metric") or "mean_eval_reward",
        target_value="tool_decides",
        wall_clock_budget=str(project.get("wall_clock_goal_hours") or "auto"),
        compute_budget=str(project.get("compute_budget") or "auto"),
        other_information=project.get("other_information") or "",
        modification_policy=str(project.get("modifications_allowed") or "unspecified"),
        offline_data_policy=str(project.get("offline_data_allowed") or "unspecified"),
        imitation_policy=str(project.get("imitation_learning_allowed") or "unspecified"),
        user_rules=[rule for rule in hard_rules if rule and rule != "none"],
        hardware_summary=_format_hardware_summary(hardware),
        python_env_summary=_format_python_env_summary(python_env),
    )


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
        - Other information: {project.get("other_information") or "none provided"}
        - Stuck refresh cooldown: {project.get("refresh_cooldown_runs") or "3"} run(s)
        - Modification policy: {project.get("modifications_allowed") or "unspecified"}
        - Imitation learning policy: {project.get("imitation_learning_allowed") or "unspecified"}
        - Permission policy: {permissions.get("policy") or "open"}
        - Token philosophy: keep outputs compact and token-efficient; prefer targeted reads over loading full files.

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


def _write_compact_spec_artifacts(project_dir: Path) -> None:
    """Generate compact spec summary + source pointers after init.

    Artifacts:
      - .drl_autoresearch/spec_compact.md
      - .drl_autoresearch/spec_index.json
    """
    config_dir = project_dir / ".drl_autoresearch"
    config_dir.mkdir(parents=True, exist_ok=True)

    sources = _discover_spec_sources(project_dir)
    if not sources:
        return

    index = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "strategy": {
            "schema": "source-driven",
            "notes": (
                "Sections are inferred from source document structure "
                "(headings/ordered items/top-level keys), not fixed categories."
            ),
        },
        "sources": [],
    }

    compact_lines: list[str] = [
        "# Compact Spec (Auto-Generated)\n",
        "",
        "This file is a token-saving navigation layer. It is **not** the source of truth.",
        "For detailed clarification, always open the original source file and lines listed below.",
        "",
        "## Reading Policy",
        "",
        "1. Read this compact file first.",
        "2. Follow pointers (`path:line`) for detailed clarification.",
        "3. If compact/original conflict, trust original source documents.",
        "",
        "## Source Index",
        "",
    ]

    for src in sources:
        try:
            text = src.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        lines = text.splitlines()
        anchors = _extract_source_anchors(lines)

        rel = src.relative_to(project_dir).as_posix()
        index["sources"].append(
            {
                "path": rel,
                "line_count": len(lines),
                "anchors": anchors,
            }
        )

        compact_lines.append(f"### `{rel}` ({len(lines)} lines)")
        if not anchors:
            compact_lines.append("- no structured anchors detected; use file-level read")
            compact_lines.append(f"- pointer: `{rel}:1`")
            compact_lines.append("")
            continue

        for a in anchors:
            pointer = f"{rel}:{a['line_start']}"
            title = a["title"]
            preview = a["preview"]
            line_range = f"{a['line_start']}-{a['line_end']}"
            if preview:
                compact_lines.append(
                    f"- `{pointer}` [{line_range}] **{title}** — {preview}"
                )
            else:
                compact_lines.append(
                    f"- `{pointer}` [{line_range}] **{title}**"
                )
        compact_lines.append("")

    compact_path = config_dir / "spec_compact.md"
    index_path = config_dir / "spec_index.json"
    compact_path.write_text("\n".join(compact_lines).rstrip() + "\n", encoding="utf-8")
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    console(f"Created {compact_path.relative_to(project_dir)}", "success")
    console(f"Created {index_path.relative_to(project_dir)}", "success")


def _discover_spec_sources(project_dir: Path) -> list[Path]:
    """Return candidate source documents for compact spec generation."""
    config_dir = project_dir / ".drl_autoresearch"

    prioritized = [
        config_dir / "onboarding.yaml",
        config_dir / "onboarding.json",
        project_dir / "NON_NEGOTIABLE_RULES.md",
        project_dir / "USER_SPEC.md",
        project_dir / "SPEC.md",
        project_dir / "spec.md",
        project_dir / "RULES.md",
        project_dir / "rules.md",
    ]

    discovered: list[Path] = []
    seen: set[Path] = set()

    def _add(path: Path) -> None:
        if path in seen or not path.is_file():
            return
        seen.add(path)
        discovered.append(path)

    for p in prioritized:
        _add(p)

    # Additional root-level candidate docs (spec-driven, bounded scope).
    for p in sorted(project_dir.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".md", ".txt"}:
            continue
        if p.name.startswith("."):
            continue
        if p.stat().st_size > 2_000_000:
            continue
        name_l = p.name.lower()
        if any(
            key in name_l
            for key in ("spec", "rule", "homework", "assignment", "rubric", "submission")
        ):
            _add(p)

    return discovered


def _extract_source_anchors(lines: list[str]) -> list[dict[str, object]]:
    """Extract structural anchors with pointers and compact previews."""
    heading_pat = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$")
    ordered_pat = re.compile(r"^\s{0,3}(\d+[\.\)])\s+(.+?)\s*$")
    key_pat = re.compile(r"^\s{0,2}([A-Za-z0-9_\-]+)\s*:\s*(.*?)\s*$")

    markers: list[tuple[int, str, str]] = []
    for idx, raw in enumerate(lines, start=1):
        line = raw.rstrip()
        if not line:
            continue

        m = heading_pat.match(line)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            markers.append((idx, f"h{level}", title))
            continue

        m = ordered_pat.match(line)
        if m and len(m.group(2).split()) <= 14:
            markers.append((idx, "ordered", m.group(2).strip()))
            continue

        # For YAML/JSON-like files with no markdown headings.
        if not any(x[1].startswith("h") for x in markers):
            m = key_pat.match(line)
            if m and m.group(1) not in {"-", "..."}:
                key = m.group(1).strip()
                val = m.group(2).strip()
                title = f"{key}" if not val else f"{key}: {val[:60]}"
                markers.append((idx, "key", title))

    if not markers:
        return []

    anchors: list[dict[str, object]] = []
    for i, (start, kind, title) in enumerate(markers):
        end = markers[i + 1][0] - 1 if i + 1 < len(markers) else len(lines)
        if end < start:
            end = start

        preview = _section_preview(lines, start, end, heading_title=title)
        anchors.append(
            {
                "kind": kind,
                "title": title[:120],
                "line_start": start,
                "line_end": end,
                "preview": preview,
            }
        )

    # Keep compact: cap to 40 anchors per source.
    if len(anchors) > 40:
        anchors = anchors[:40]
    return anchors


def _section_preview(
    lines: list[str],
    start: int,
    end: int,
    heading_title: str,
) -> str:
    """Return a short preview line for the section."""
    cleaned: list[str] = []
    for i in range(start, min(end + 1, len(lines) + 1)):
        line = lines[i - 1].strip()
        if not line:
            continue
        if line.lstrip().startswith("#"):
            continue
        if line == heading_title:
            continue
        cleaned.append(line)
        if len(cleaned) >= 3:
            break

    if not cleaned:
        return ""
    preview = " ".join(cleaned)
    preview = re.sub(r"\s+", " ", preview).strip()
    if len(preview) > 160:
        preview = preview[:157].rstrip() + "..."
    return preview


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
