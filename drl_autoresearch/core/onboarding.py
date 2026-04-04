"""
onboarding.py — Interactive onboarding flow for DRL AutoResearch.

Uses stdin/stdout only (no external TUI libraries).
Config is persisted under .drl_autoresearch/ inside the target project.

Usage
-----
    flow = OnboardingFlow(project_dir=Path("."), auto=False, skip=False)
    result = flow.run()
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from drl_autoresearch.core.hardware import HardwareDetector, HardwareInfo

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONFIG_DIR = ".drl_autoresearch"
ONBOARDING_YAML = "onboarding.yaml"
ONBOARDING_JSON = "onboarding.json"
PYTHON_ENV_YAML = "python_env.yaml"
PYTHON_ENV_JSON = "python_env.json"

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AssumptionRecord:
    field: str
    value: Any
    source: str          # "auto-detected" / "inferred" / "default" / "user"
    confidence: str      # "high" / "medium" / "low"
    note: str


@dataclass
class OnboardingResult:
    project: dict
    hardware: HardwareInfo
    python_env: dict
    permissions: dict
    hard_rules: list[str]
    assumptions: list[AssumptionRecord] = field(default_factory=list)


# ---------------------------------------------------------------------------
# YAML helpers (stdlib-safe fallback to JSON)
# ---------------------------------------------------------------------------


def _yaml_dump(data: Any) -> Optional[str]:
    try:
        import yaml  # type: ignore
        return yaml.dump(data, default_flow_style=False, allow_unicode=True)
    except ImportError:
        return None


def _yaml_load(text: str) -> Optional[Any]:
    try:
        import yaml  # type: ignore
        return yaml.safe_load(text)
    except ImportError:
        return None


def _atomic_write(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".ob_tmp_", suffix=path.suffix)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _save_yaml_or_json(config_dir: Path, stem: str, data: dict) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    yaml_text = _yaml_dump(data)
    if yaml_text is not None:
        _atomic_write(config_dir / f"{stem}.yaml", yaml_text)
    else:
        _atomic_write(config_dir / f"{stem}.json",
                      json.dumps(data, indent=2, ensure_ascii=False))


def _load_yaml_or_json(config_dir: Path, stem: str) -> Optional[dict]:
    yaml_path = config_dir / f"{stem}.yaml"
    if yaml_path.exists():
        try:
            text = yaml_path.read_text(encoding="utf-8")
            data = _yaml_load(text)
            if isinstance(data, dict):
                return data
        except OSError:
            pass

    json_path = config_dir / f"{stem}.json"
    if json_path.exists():
        try:
            return json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass

    return None


# ---------------------------------------------------------------------------
# TUI helpers
# ---------------------------------------------------------------------------

SEP = "━" * 55


def _print_header(title: str) -> None:
    print()
    print(SEP)
    print(f"  {title}")
    print(SEP)


def _collect_multiline_text(prompt: str, default: Optional[str] = None) -> Optional[str]:
    try:
        from textual.app import App, ComposeResult
        from textual.binding import Binding
        from textual.containers import Container
        from textual.widgets import Footer, Static, TextArea

        class TextPromptApp(App[str | None]):
            CSS = """
            Screen {
                align: center middle;
                background: $surface;
            }

            #dialog {
                width: 88;
                max-width: 92vw;
                height: 22;
                border: round $primary;
                padding: 1 2;
                background: $panel;
            }

            #title {
                width: 100%;
                text-style: bold;
                margin-bottom: 1;
            }

            #hint {
                width: 100%;
                color: $text-muted;
                margin-bottom: 1;
            }

            TextArea {
                width: 100%;
                height: 1fr;
            }
            """

            BINDINGS = [
                Binding("enter", "submit", "Submit", priority=True),
                Binding("ctrl+j", "newline", "New Line", priority=True),
                Binding("escape", "cancel", "Cancel", priority=True),
                Binding("ctrl+c", "cancel", "Cancel", priority=True),
            ]

            def compose(self) -> ComposeResult:
                yield Container(
                    Static(prompt, id="title"),
                    Static("Enter submits. Ctrl+J inserts a new line.", id="hint"),
                    TextArea(default or "", id="answer"),
                    Footer(),
                    id="dialog",
                )

            def on_mount(self) -> None:
                self.query_one(TextArea).focus()

            def action_submit(self) -> None:
                value = self.query_one(TextArea).text.rstrip("\n").strip()
                self.exit(value or None)

            def action_newline(self) -> None:
                self.query_one(TextArea).insert("\n")

            def action_cancel(self) -> None:
                self.exit(None)

        text = TextPromptApp().run()
        if text is None:
            return None
        return str(text).strip() or None
    except Exception:
        print(f"{prompt}")
        print("  Multi-line input box unavailable; falling back to terminal input.")
        print("  Type `END` on its own line when done.")
        lines: list[str] = []
        while True:
            try:
                line = input("| ")
            except (EOFError, KeyboardInterrupt):
                print()
                return "\n".join(lines).strip() or None
            if line == "END":
                break
            lines.append(line)

        return "\n".join(lines).strip() or None


def _tui_select(
    title: str,
    values: list[tuple[str, str]],
    default: Optional[str] = None,
) -> Optional[str]:
    try:
        from textual.app import App, ComposeResult
        from textual.binding import Binding
        from textual.containers import Container
        from textual.widgets import Footer, OptionList, Static

        class SelectPromptApp(App[str | None]):
            CSS = """
            Screen {
                align: center middle;
                background: $surface;
            }

            #dialog {
                width: 88;
                max-width: 92vw;
                height: auto;
                max-height: 24;
                border: round $primary;
                padding: 1 2;
                background: $panel;
            }

            #title {
                width: 100%;
                text-style: bold;
                margin-bottom: 1;
            }

            OptionList {
                width: 100%;
                height: auto;
                max-height: 12;
                margin-bottom: 1;
            }
            """

            BINDINGS = [
                Binding("enter", "submit", "Select", priority=True),
                Binding("escape", "cancel", "Cancel", priority=True),
                Binding("ctrl+c", "cancel", "Cancel", priority=True),
            ]

            def compose(self) -> ComposeResult:
                option_list = OptionList(*(label for _, label in values), id="options")
                yield Container(
                    Static(title, id="title"),
                    option_list,
                    Footer(),
                    id="dialog",
                )

            def on_mount(self) -> None:
                option_list = self.query_one(OptionList)
                option_list.focus()
                initial = default if default is not None else (values[0][0] if values else None)
                if initial is None:
                    return
                for idx, (value, _) in enumerate(values):
                    if value == initial:
                        option_list.highlighted = idx
                        break

            def action_submit(self) -> None:
                option_list = self.query_one(OptionList)
                idx = option_list.highlighted
                if idx is None:
                    self.exit(None)
                    return
                self.exit(values[idx][0])

            def action_cancel(self) -> None:
                self.exit(None)

        return SelectPromptApp().run()
    except Exception:
        return None


def _ask(
    prompt: str,
    default: Optional[str] = None,
    options: Optional[list[str]] = None,
    allow_auto: bool = False,
    allow_skip: bool = True,
    allow_decide: bool = False,
    allow_multiline: bool = True,
) -> Optional[str]:
    """
    Ask a single question and return the answer.

    Hints shown: [a=auto-detect] [s=skip] [d=let tool decide]
    Returns None if skipped.
    Returns "__auto__" if user chose auto-detect.
    Returns "__decide__" if user chose let-tool-decide.
    """
    hints = []
    if default is not None:
        hints.append(f"default: {default!r}")
    if options:
        hints.append("options: " + " / ".join(options))
    if allow_auto:
        hints.append("a=auto-detect")
    if allow_skip:
        hints.append("s=skip")
    if allow_decide:
        hints.append("d=let tool decide")
    if not allow_multiline:
        hint_str = f"  [{', '.join(hints)}]" if hints else ""
        full_prompt = f"{prompt}{hint_str}\n> "

    while True:
        if allow_multiline:
            raw = _collect_multiline_text(prompt, default=default)
            if raw is None:
                if default is not None:
                    return default
                if allow_skip:
                    return None
                print("  (A value is required. Press Ctrl+C to abort.)")
                continue
            raw = raw.strip()
        else:
            try:
                raw = input(full_prompt).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return None

        if raw.lower() == "s" and allow_skip:
            return None
        if raw.lower() == "a" and allow_auto:
            return "__auto__"
        if raw.lower() == "d" and allow_decide:
            return "__decide__"
        if raw == "" and default is not None:
            return default
        if raw:
            return raw
        if default is None and not allow_skip:
            print("  (A value is required. Press Ctrl+C to abort.)")
            continue
        return None


def _ask_choice(
    prompt: str,
    choices: list[str],
    default: Optional[str] = None,
    allow_skip: bool = True,
    allow_decide: bool = False,
) -> Optional[str]:
    """Numbered-choice prompt. Returns chosen value or None."""
    values = [(choice, choice) for choice in choices]
    if allow_decide:
        values.append(("__decide__", "let tool decide"))
    if allow_skip:
        values.append(("__skip__", "skip"))

    tui_result = _tui_select(prompt, values, default=default)
    if tui_result == "__skip__":
        return None
    if tui_result in {choice for choice in choices} or tui_result == "__decide__":
        return tui_result

    print()
    print(prompt)
    for i, c in enumerate(choices, 1):
        marker = " (default)" if c == default else ""
        print(f"  {i}. {c}{marker}")

    hints = []
    if default:
        hints.append(f"default: {default!r}")
    if allow_skip:
        hints.append("s=skip")
    if allow_decide:
        hints.append("d=let tool decide")
    hint_str = f"  [{', '.join(hints)}]" if hints else ""

    full_prompt = f"Enter number or value{hint_str}\n> "

    while True:
        try:
            raw = input(full_prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return default

        if raw == "" and default is not None:
            return default
        if raw.lower() == "s" and allow_skip:
            return None
        if raw.lower() == "d" and allow_decide:
            return "__decide__"

        # Numeric selection
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(choices):
                return choices[idx]
            print(f"  Please enter a number between 1 and {len(choices)}.")
            continue

        # Direct text value (may or may not be in choices)
        if raw:
            return raw

        if default is None and not allow_skip:
            print("  (A value is required.)")
            continue
        return default


def _ask_confirm(prompt: str, default: bool = True) -> bool:
    """Yes/No confirmation. Returns bool."""
    default_value = "yes" if default else "no"
    tui_result = _tui_select(
        prompt,
        [("yes", "yes"), ("no", "no")],
        default=default_value,
    )
    if tui_result == "yes":
        return True
    if tui_result == "no":
        return False

    hint = "[Y/n]" if default else "[y/N]"
    full_prompt = f"{prompt} {hint}\n> "
    try:
        raw = input(full_prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    if raw == "":
        return default
    return raw in ("y", "yes")


# ---------------------------------------------------------------------------
# Package-manager / Python env detection
# ---------------------------------------------------------------------------


def _detect_package_manager(project_dir: Path) -> tuple[Optional[str], str]:
    """
    Return (manager_name, confidence).
    Checks for lock/config files in project_dir.
    """
    checks = [
        ("uv", "uv.lock"),
        ("poetry", "poetry.lock"),
        ("pipenv", "Pipfile.lock"),
        ("pipenv", "Pipfile"),
        ("pip", "requirements.txt"),
        ("conda", "environment.yml"),
        ("conda", "environment.yaml"),
    ]
    for manager, fname in checks:
        if (project_dir / fname).exists():
            return manager, "high"

    # Check pyproject.toml build-backend
    pyproject = project_dir / "pyproject.toml"
    if pyproject.exists():
        try:
            text = pyproject.read_text(encoding="utf-8")
            if "poetry" in text.lower():
                return "poetry", "medium"
            if "hatch" in text.lower():
                return "hatch", "medium"
            if "flit" in text.lower():
                return "flit", "medium"
        except OSError:
            pass
        return "pip", "low"

    return None, "low"


def _detect_active_venv() -> Optional[str]:
    """Return the path of the currently active virtual environment, if any."""
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        return venv
    conda = os.environ.get("CONDA_PREFIX")
    if conda:
        return conda
    return None


def _detect_python_version() -> str:
    import sys
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


# ---------------------------------------------------------------------------
# OnboardingFlow
# ---------------------------------------------------------------------------


class OnboardingFlow:
    """
    Interactive onboarding wizard for a DRL AutoResearch project.

    Parameters
    ----------
    project_dir : Path
        Root of the target project (where .drl_autoresearch/ will be written).
    auto : bool
        Auto-detect / infer everything possible; minimise prompts.
    skip : bool
        Use all safe defaults; skip every optional question.
    """

    def __init__(
        self,
        project_dir: Path,
        auto: bool = False,
        skip: bool = False,
    ) -> None:
        self.project_dir = Path(project_dir).resolve()
        self.auto = auto
        self.skip = skip
        self._assumptions: list[AssumptionRecord] = []
        self._detector = HardwareDetector()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> OnboardingResult:
        """Run the full onboarding flow and return an OnboardingResult."""
        if self.skip:
            print("[onboarding] --skip-onboarding: using safe defaults.")
            result = self._build_skip_defaults()
            self.save_results(result)
            return result

        print()
        print(SEP)
        print("  DRL AutoResearch — Project Onboarding")
        print(SEP)
        if self.auto:
            print("  Running in --auto mode. Detecting everything possible.")

        project = self.run_project_group()
        hardware = self.run_hardware_group()
        python_env = self.run_python_env_group()
        permissions = self.run_permissions_group()
        hard_rules = self.run_hard_rules_group()

        result = OnboardingResult(
            project=project,
            hardware=hardware,
            python_env=python_env,
            permissions=permissions,
            hard_rules=hard_rules,
            assumptions=list(self._assumptions),
        )
        self.save_results(result)

        # Summary
        print()
        print(SEP)
        print("  Onboarding complete!")
        print(f"  Config saved to {self.project_dir / CONFIG_DIR}/")
        if self._assumptions:
            print(f"  {len(self._assumptions)} assumption(s) logged.")
        print(SEP)
        return result

    # ------------------------------------------------------------------
    # Group 1: Project / Task
    # ------------------------------------------------------------------

    def run_project_group(self) -> dict:
        _print_header("Group 1 / 5 — Project & Task")

        default_name = self.project_dir.name

        if self.auto or self.skip:
            name = default_name
            self._log_assumption("project.name", name, "inferred", "high",
                                 "Derived from project directory name.")
        else:
            name = _ask("Project name", default=default_name,
                        allow_skip=False) or default_name

        if self.skip:
            return self._default_project(name)

        env = _ask(
            "RL environment / benchmark / simulator",
            allow_auto=False,
            allow_skip=True,
            allow_decide=True,
            allow_multiline=True,
        )
        if env is None:
            self._log_assumption("project.env", None, "default", "low",
                                 "User skipped; environment unknown.")
        elif env == "__decide__":
            env = None
            self._log_assumption("project.env", None, "inferred", "low",
                                 "Tool will decide based on context.")

        obs_type = _ask(
            "Observation type",
            default="continuous_vector",
            allow_skip=True,
            allow_decide=True,
            allow_multiline=True,
        )
        if obs_type in (None, "__decide__"):
            self._log_assumption("project.obs_type", obs_type, "default", "medium",
                                 "Observation type not specified; will infer.")
            obs_type = obs_type

        action_space = _ask(
            "Action space",
            default="discrete",
            allow_skip=True,
            allow_decide=True,
            allow_multiline=True,
        )
        if action_space in (None, "__decide__"):
            self._log_assumption("project.action_space", action_space, "default",
                                 "medium", "Action space not specified; will infer.")

        objective = _ask(
            "Main objective (e.g. 'maximize episode reward')",
            default="maximize episode reward",
            allow_skip=True,
            allow_decide=True,
            allow_multiline=True,
        )
        if objective in (None, "__decide__"):
            objective = "maximize episode reward"
            self._log_assumption("project.objective", objective, "default", "medium",
                                 "Defaulting to reward maximisation.")

        success_metric = _ask(
            "Real success metric (what actually matters, not just reward)",
            allow_skip=True,
            allow_decide=True,
            allow_multiline=True,
        )
        if success_metric in (None, "__decide__"):
            self._log_assumption("project.success_metric", None, "default", "low",
                                 "Success metric not provided; will use reward as proxy.")
            success_metric = None

        modifications = _ask(
            "Allowed modification scope",
            default="all",
            allow_skip=True,
            allow_decide=True,
            allow_multiline=True,
        )
        offline_data = _ask_choice(
            "Offline data allowed?",
            ["yes", "no", "limited"],
            default="no",
            allow_skip=True,
            allow_decide=True,
        )
        imitation = _ask(
            "Imitation learning allowed",
            default="no",
            allow_skip=True,
            allow_decide=True,
            allow_multiline=True,
        )
        wall_clock = _ask(
            "Wall-clock goal (hours, e.g. '8' for overnight)",
            allow_skip=True,
            allow_decide=True,
            allow_multiline=True,
        )
        compute_budget = _ask(
            "Compute budget (GPU-hours or dollars, e.g. '50 GPU-hours')",
            allow_skip=True,
            allow_decide=True,
            allow_multiline=True,
        )
        refresh_cooldown_runs = _ask(
            "Stuck refresh cooldown (runs between refreshes)",
            default="3",
            allow_skip=True,
            allow_decide=True,
        )
        if refresh_cooldown_runs in (None, "__decide__"):
            refresh_cooldown_runs = "3"
            self._log_assumption(
                "project.refresh_cooldown_runs",
                refresh_cooldown_runs,
                "default",
                "medium",
                "Defaulting stuck refresh cooldown to 3 runs.",
            )
        other_information = _ask(
            "Other information (project quirks, known issues, extra context)",
            allow_skip=True,
            allow_decide=True,
            allow_multiline=True,
        )
        if other_information in (None, "__decide__"):
            self._log_assumption(
                "project.other_information",
                None,
                "default",
                "low",
                "No extra project context was provided.",
            )
            other_information = None

        return {
            "name": name,
            "env": env,
            "obs_type": obs_type,
            "action_space": action_space,
            "objective": objective,
            "success_metric": success_metric,
            "modifications_allowed": modifications,
            "offline_data_allowed": offline_data,
            "imitation_learning_allowed": imitation,
            "wall_clock_goal_hours": wall_clock,
            "compute_budget": compute_budget,
            "refresh_cooldown_runs": refresh_cooldown_runs,
            "other_information": other_information,
        }

    def _default_project(self, name: str) -> dict:
        self._log_assumption("project", "all defaults", "default", "medium",
                             "--skip-onboarding active; project details unset.")
        return {
            "name": name,
            "env": None,
            "obs_type": None,
            "action_space": None,
            "objective": "maximize episode reward",
            "success_metric": None,
            "modifications_allowed": "all",
            "offline_data_allowed": "no",
            "imitation_learning_allowed": "no",
            "wall_clock_goal_hours": None,
            "compute_budget": None,
            "refresh_cooldown_runs": "3",
            "other_information": None,
        }

    # ------------------------------------------------------------------
    # Group 2: Hardware
    # ------------------------------------------------------------------

    def run_hardware_group(self) -> HardwareInfo:
        _print_header("Group 2 / 5 — Hardware")
        print("  Detecting hardware…")

        info = self._detector.detect()

        print()
        print(f"  CPU   : {info.cpu_model}")
        print(f"  Cores : {info.cpu_cores} physical / {info.cpu_threads} threads")
        print(f"  RAM   : {info.ram_gb} GB")
        if info.has_gpu:
            print(f"  GPUs  : {info.gpu_count}")
            for g in info.gpus:
                cc = f"  CC {g.compute_capability}" if g.compute_capability else ""
                print(f"    [{g.index}] {g.name}  {g.vram_gb} GB VRAM{cc}")
            print(f"  CUDA  : {info.cuda_version or 'unknown'}")
        else:
            print("  GPUs  : none detected")

        self._log_assumption("hardware", "auto-detected", "auto-detected", "high",
                             "Hardware probed via torch / nvidia-smi / /proc.")

        if self.auto or self.skip:
            return info

        confirmed = _ask_confirm("Is this correct?", default=True)
        if not confirmed:
            print("  You can override individual values:")
            cpu_model = _ask("CPU model", default=info.cpu_model)
            if cpu_model and cpu_model != info.cpu_model:
                info.cpu_model = cpu_model
                self._log_assumption("hardware.cpu_model", cpu_model, "user",
                                     "high", "User overrode CPU model.")

            ram_raw = _ask("RAM (GB)", default=str(info.ram_gb))
            if ram_raw:
                try:
                    info.ram_gb = float(ram_raw)
                    self._log_assumption("hardware.ram_gb", info.ram_gb, "user",
                                         "high", "User overrode RAM.")
                except ValueError:
                    pass

        if info.gpu_count > 1 and not self.skip:
            info.multi_gpu_allowed = _ask_confirm(
                f"  Allow multi-GPU training across all {info.gpu_count} GPUs?",
                default=False,
            )
        elif info.gpu_count > 1:
            info.multi_gpu_allowed = False
            self._log_assumption("hardware.multi_gpu_allowed", False, "default",
                                 "medium", "Multi-GPU not enabled by default.")

        # Persist hardware separately
        self._detector.save(self.project_dir, info)
        return info

    # ------------------------------------------------------------------
    # Group 3: Python Environment
    # ------------------------------------------------------------------

    def run_python_env_group(self) -> dict:
        _print_header("Group 3 / 5 — Python Environment")

        # Package manager
        pkg_manager, pm_conf = _detect_package_manager(self.project_dir)
        if pkg_manager:
            print(f"  Detected package manager: {pkg_manager!r} (confidence: {pm_conf})")
            self._log_assumption("python_env.package_manager", pkg_manager,
                                 "auto-detected", pm_conf,
                                 f"Lock/config file found for {pkg_manager}.")
        else:
            print("  No package manager detected.")

        if self.auto or self.skip:
            final_pm = pkg_manager or "pip"
            if not pkg_manager:
                self._log_assumption("python_env.package_manager", "pip",
                                     "default", "low", "Defaulting to pip.")
        else:
            if pkg_manager:
                confirmed = _ask_confirm(
                    f"  Use {pkg_manager!r} as the package manager?", default=True
                )
                if confirmed:
                    final_pm = pkg_manager
                else:
                    final_pm = _ask_choice(
                        "  Select package manager:",
                        ["pip", "uv", "poetry", "pipenv", "conda", "hatch"],
                        default="pip",
                    ) or "pip"
            else:
                final_pm = _ask_choice(
                    "  Select package manager:",
                    ["pip", "uv", "poetry", "pipenv", "conda", "hatch"],
                    default="pip",
                    allow_skip=False,
                ) or "pip"

        # Active virtual environment
        active_venv = _detect_active_venv()
        if active_venv:
            print(f"  Active venv/conda: {active_venv}")
            self._log_assumption("python_env.venv", active_venv, "auto-detected",
                                 "high", "VIRTUAL_ENV / CONDA_PREFIX env var.")
        else:
            print("  No active virtual environment detected.")

        if self.auto or self.skip:
            final_venv = active_venv
        else:
            if active_venv:
                use_active = _ask_confirm(
                    "  Use this virtual environment?", default=True
                )
                final_venv = active_venv if use_active else _ask(
                    "  Virtual env path (leave blank to skip)", allow_skip=True
                )
            else:
                final_venv = _ask(
                    "  Virtual env path (or skip to create later)", allow_skip=True
                )

        # Python version
        current_py = _detect_python_version()
        print(f"  Current Python: {current_py}")
        if self.auto or self.skip:
            py_version = current_py
            self._log_assumption("python_env.python_version", py_version,
                                 "auto-detected", "high",
                                 "Using currently running Python version.")
        else:
            py_version = _ask(
                "  Python version preference", default=current_py, allow_skip=True
            ) or current_py

        # Create new env?
        if self.auto or self.skip:
            create_env = "auto"
            self._log_assumption("python_env.create_new_env", "auto", "default",
                                 "medium", "Will decide at init time.")
        else:
            create_env = _ask_choice(
                "  Create a new virtual environment?",
                ["yes", "no", "auto"],
                default="auto",
                allow_skip=True,
            ) or "auto"

        result = {
            "package_manager": final_pm,
            "venv_path": str(final_venv) if final_venv else None,
            "python_version": py_version,
            "create_new_env": create_env,
        }

        # Save separately per spec
        config_dir = self.project_dir / CONFIG_DIR
        _save_yaml_or_json(config_dir, "python_env", result)

        return result

    # ------------------------------------------------------------------
    # Group 4: Permission Policy
    # ------------------------------------------------------------------

    def run_permissions_group(self) -> dict:
        _print_header("Group 4 / 5 — Permission Policy")

        print("""
  locked        — nothing changes without explicit approval
  prompted      — ask before each install/change
  bootstrap-only — only allow changes during init
  open          — research mode, allow everything  (default)
  project-only  — only install within project venv
""")

        if self.auto or self.skip:
            policy = "open"
            self._log_assumption("permissions.policy", policy, "default", "high",
                                 "Execution default: open mode.")
        else:
            policy = _ask_choice(
                "  Choose permission policy:",
                ["locked", "prompted", "bootstrap-only", "open", "project-only"],
                default="open",
                allow_skip=True,
                allow_decide=False,
            ) or "open"

        return {"policy": policy}

    # ------------------------------------------------------------------
    # Group 5: Hard Rules (MANDATORY)
    # ------------------------------------------------------------------

    def run_hard_rules_group(self) -> list[str]:
        print()
        print(SEP)
        print("  FINAL QUESTION — HARD RULES")
        print(SEP)

        if self.skip:
            rules = ["none"]
            self._log_assumption("hard_rules", rules, "default", "low",
                                 "--skip-onboarding: user acknowledged with 'none'.")
            print("  (--skip-onboarding active; hard rules set to 'none')")
            return rules

        raw = _collect_multiline_text(
            "Hard rules (one per line). Type `none` if there are no extra hard rules.",
            default="",
        )
        rules = [line.strip() for line in (raw or "").splitlines() if line.strip()]

        if not rules:
            rules = ["none"]

        return rules

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_results(self, result: OnboardingResult) -> None:
        """Persist OnboardingResult to .drl_autoresearch/onboarding.yaml (or .json)."""
        config_dir = self.project_dir / CONFIG_DIR
        config_dir.mkdir(parents=True, exist_ok=True)

        def _hardware_to_saveable(hw: HardwareInfo) -> dict:
            gpus = [
                {
                    "index": g.index,
                    "name": g.name,
                    "vram_gb": g.vram_gb,
                    "compute_capability": g.compute_capability,
                }
                for g in hw.gpus
            ]
            return {
                "cpu_model": hw.cpu_model,
                "cpu_cores": hw.cpu_cores,
                "cpu_threads": hw.cpu_threads,
                "ram_gb": hw.ram_gb,
                "has_gpu": hw.has_gpu,
                "gpu_count": hw.gpu_count,
                "gpus": gpus,
                "cuda_available": hw.cuda_available,
                "cuda_version": hw.cuda_version,
                "multi_gpu_allowed": hw.multi_gpu_allowed,
            }

        assumptions_list = [
            {
                "field": a.field,
                "value": str(a.value) if not isinstance(a.value, (str, int, float, bool, type(None))) else a.value,
                "source": a.source,
                "confidence": a.confidence,
                "note": a.note,
            }
            for a in result.assumptions
        ]

        data = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "project": result.project,
            "hardware": _hardware_to_saveable(result.hardware),
            "python_env": result.python_env,
            "permissions": result.permissions,
            "hard_rules": result.hard_rules,
            "assumptions": assumptions_list,
        }

        _save_yaml_or_json(config_dir, "onboarding", data)

    # ------------------------------------------------------------------
    # Skip-mode defaults
    # ------------------------------------------------------------------

    def _build_skip_defaults(self) -> OnboardingResult:
        name = self.project_dir.name
        self._log_assumption("all", "safe defaults", "default", "medium",
                             "--skip-onboarding: all fields set to safe defaults.")

        detected_hw = self._detector.detect()
        self._log_assumption("hardware", "auto-detected", "auto-detected", "high",
                             "Hardware probed even in skip mode.")

        pkg_manager, pm_conf = _detect_package_manager(self.project_dir)
        if not pkg_manager:
            pkg_manager = "pip"

        active_venv = _detect_active_venv()
        py_version = _detect_python_version()

        return OnboardingResult(
            project={
                "name": name,
                "env": None,
                "obs_type": None,
                "action_space": None,
                "objective": "maximize episode reward",
                "success_metric": None,
                "modifications_allowed": "all",
                "offline_data_allowed": "no",
                "imitation_learning_allowed": "no",
                "wall_clock_goal_hours": None,
                "compute_budget": None,
                "refresh_cooldown_runs": "3",
                "other_information": None,
            },
            hardware=detected_hw,
            python_env={
                "package_manager": pkg_manager,
                "venv_path": str(active_venv) if active_venv else None,
                "python_version": py_version,
                "create_new_env": "auto",
            },
            permissions={"policy": "open"},
            hard_rules=["none"],
            assumptions=list(self._assumptions),
        )

    # ------------------------------------------------------------------
    # Internal: assumption logger
    # ------------------------------------------------------------------

    def _log_assumption(
        self,
        field: str,
        value: Any,
        source: str,
        confidence: str,
        note: str,
    ) -> None:
        self._assumptions.append(
            AssumptionRecord(
                field=field,
                value=value,
                source=source,
                confidence=confidence,
                note=note,
            )
        )
