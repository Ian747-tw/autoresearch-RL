"""
DRL AutoResearch CLI — argparse-based entry point.

Each subcommand delegates to the appropriate module in
``drl_autoresearch.core`` or ``drl_autoresearch.dashboard``.
"""

from __future__ import annotations

import argparse
import json
import sys
import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Console helpers (ANSI colour only when stdout is a real TTY)
# ---------------------------------------------------------------------------

_USE_COLOUR: bool = sys.stdout.isatty() and os.environ.get("NO_COLOR", "") == ""

_RESET  = "\033[0m"  if _USE_COLOUR else ""
_GREEN  = "\033[32m" if _USE_COLOUR else ""
_YELLOW = "\033[33m" if _USE_COLOUR else ""
_RED    = "\033[31m" if _USE_COLOUR else ""
_BLUE   = "\033[34m" if _USE_COLOUR else ""


def console(message: str, level: str = "info") -> None:
    """Print a prefixed, optionally coloured message to stdout.

    Parameters
    ----------
    message:
        The text to display.
    level:
        One of ``"success"``, ``"warning"``, ``"error"``, ``"info"``
        (default).  Any other value falls back to plain output.
    """
    prefixes = {
        "success": f"{_GREEN}[✓]{_RESET}",
        "warning": f"{_YELLOW}[!]{_RESET}",
        "error":   f"{_RED}[✗]{_RESET}",
        "info":    f"{_BLUE}[~]{_RESET}",
    }
    prefix = prefixes.get(level, "   ")
    print(f"{prefix} {message}")


# ---------------------------------------------------------------------------
# Lazy imports — keeps startup fast and avoids hard errors if optional deps
# are missing when running unrelated subcommands.
# ---------------------------------------------------------------------------

def _core_init():
    from drl_autoresearch.core import init as _init
    return _init

def _core_doctor():
    from drl_autoresearch.core import doctor as _doctor
    return _doctor

def _core_run():
    from drl_autoresearch.core import run as _run
    return _run

def _core_status():
    from drl_autoresearch.core import status as _status
    return _status

def _core_plan():
    from drl_autoresearch.core import plan as _plan
    return _plan

def _core_research():
    from drl_autoresearch.core import research as _research
    return _research

def _core_resume():
    from drl_autoresearch.core import resume as _resume
    return _resume

def _core_check():
    from drl_autoresearch.core import check as _check
    return _check

def _core_stop():
    from drl_autoresearch.core import stop as _stop
    return _stop

def _dashboard_mod():
    from drl_autoresearch import dashboard as _dash
    return _dash


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def _cmd_install(args: argparse.Namespace) -> int:
    console("DRL AutoResearch installed successfully.", "success")
    console("Quick-start tips:", "info")
    print("  1.  cd <your-project>")
    print("  2.  drl-autoresearch init")
    print("  3.  drl-autoresearch doctor")
    print("  4.  drl-autoresearch run")
    print()
    console("Run `drl-autoresearch --help` for full command list.", "info")
    return 0


def _cmd_init(args: argparse.Namespace) -> int:
    project_dir = Path(args.project_dir).resolve()
    mod = _core_init()
    plugin = getattr(args, "plugin", None)
    if plugin == "none":
        plugin = ""          # empty string = skip install but don't prompt
    skill_pack = getattr(args, "skill_pack", None)
    project_mode = getattr(args, "project_mode", None)
    return mod.run(
        project_dir=project_dir,
        skip_onboarding=args.skip_onboarding,
        auto=args.auto,
        refresh=getattr(args, "refresh", False),
        plugin=plugin,
        skill_pack=skill_pack,
        project_mode=project_mode,
    )


def _cmd_doctor(args: argparse.Namespace) -> int:
    project_dir = Path(args.project_dir).resolve()
    mod = _core_doctor()
    return mod.run(project_dir=project_dir, fix=getattr(args, "fix", False))


def _cmd_dashboard(args: argparse.Namespace) -> int:
    project_dir = Path(args.project_dir).resolve()
    mod = _dashboard_mod()
    return mod.run(project_dir=project_dir, port=args.port)


def _cmd_run(args: argparse.Namespace) -> int:
    project_dir = Path(args.project_dir).resolve()
    mod = _core_run()
    return mod.run(
        project_dir=project_dir,
        parallel=args.parallel,
        dry_run=args.dry_run,
        once=getattr(args, "once", False),
        agent_backend=getattr(args, "agent_backend", "auto"),
    )


def _cmd_status(args: argparse.Namespace) -> int:
    project_dir = Path(args.project_dir).resolve()
    mod = _core_status()
    return mod.run(project_dir=project_dir)


def _cmd_plan(args: argparse.Namespace) -> int:
    project_dir = Path(args.project_dir).resolve()
    mod = _core_plan()
    return mod.run(project_dir=project_dir, refresh=args.refresh)


def _cmd_research(args: argparse.Namespace) -> int:
    project_dir = Path(args.project_dir).resolve()
    mod = _core_research()
    return mod.run(project_dir=project_dir)


def _cmd_resume(args: argparse.Namespace) -> int:
    project_dir = Path(args.project_dir).resolve()
    mod = _core_resume()
    return mod.run(
        project_dir=project_dir,
        parallel=args.parallel,
        dry_run=args.dry_run,
        no_run=args.no_run,
        agent_backend=getattr(args, "agent_backend", "auto"),
    )


def _cmd_check(args: argparse.Namespace) -> int:
    project_dir = Path(args.project_dir).resolve()
    details: dict = {}
    if args.details:
        try:
            details = json.loads(args.details)
        except json.JSONDecodeError as exc:
            console(f"--details is not valid JSON: {exc}", "error")
            return 2
    mod = _core_check()
    return mod.run(
        project_dir=project_dir,
        action=args.action,
        details=details,
    )


def _cmd_stop(args: argparse.Namespace) -> int:
    project_dir = Path(args.project_dir).resolve()
    mod = _core_stop()
    return mod.run(project_dir=project_dir, brief=getattr(args, "brief", "") or "")


# ---------------------------------------------------------------------------
# Parser construction
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="drl-autoresearch",
        description=(
            "DRL AutoResearch — autonomous deep reinforcement learning "
            "research workflow manager."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  drl-autoresearch init --project-dir ~/my-drl-project\n"
            "  drl-autoresearch doctor\n"
            "  drl-autoresearch run --parallel 4\n"
            "  drl-autoresearch dashboard --port 8765\n"
        ),
    )
    parser.add_argument(
        "--version", action="version", version="%(prog)s 0.1.0"
    )

    sub = parser.add_subparsers(
        dest="command", metavar="COMMAND", title="Available commands"
    )
    sub.required = True

    # -- install -------------------------------------------------------------
    p_install = sub.add_parser(
        "install",
        help="Print install confirmation and usage tips.",
        description=(
            "Confirm that DRL AutoResearch is installed correctly and "
            "display quick-start tips."
        ),
    )
    p_install.set_defaults(func=_cmd_install)

    # -- init ----------------------------------------------------------------
    p_init = sub.add_parser(
        "init",
        help="Scaffold a new DRL AutoResearch project.",
        description=(
            "Initialise the target project directory by creating the "
            ".drl_autoresearch/ config folder, logs/, skills/, and "
            "default config files."
        ),
    )
    p_init.add_argument(
        "--project-dir",
        default=".",
        metavar="DIR",
        help="Root of the target project (default: current directory).",
    )
    p_init.add_argument(
        "--skip-onboarding",
        action="store_true",
        help="Skip the interactive onboarding questionnaire.",
    )
    p_init.add_argument(
        "--auto",
        action="store_true",
        help=(
            "Non-interactive mode: accept all defaults without prompting. "
            "Implies --skip-onboarding."
        ),
    )
    p_init.add_argument(
        "--refresh",
        action="store_true",
        help=(
            "Remove DRL AutoResearch-managed config, state, and plugin files in "
            "the target project before initializing again. Does not delete user code."
        ),
    )
    p_init.add_argument(
        "--plugin",
        choices=["cc", "codex", "both", "none"],
        default=None,
        metavar="PLUGIN",
        help=(
            "AI agent plugin(s) to install: cc (Claude Code slash commands), "
            "codex (AGENT.md), both, or none. "
            "If omitted, prompts interactively (or installs both with --auto)."
        ),
    )
    p_init.add_argument(
        "--skill-pack",
        choices=["drl", "custom"],
        default=None,
        metavar="PACK",
        help=(
            "Skill-pack mode: drl keeps the bundled DRL playbooks; "
            "custom removes the bundled DRL playbooks and installs a compact "
            "skill generator backend for building a domain-specific pack. "
            "If omitted, prompts interactively when possible."
        ),
    )
    p_init.add_argument(
        "--project-mode",
        choices=["build", "improve"],
        default=None,
        metavar="MODE",
        help=(
            "Project mode: build (empty/incomplete project, design from scratch) "
            "or improve (existing working model, optimize and iterate). "
            "If omitted, prompts interactively when possible."
        ),
    )
    p_init.set_defaults(func=_cmd_init)

    # -- doctor --------------------------------------------------------------
    p_doctor = sub.add_parser(
        "doctor",
        help="Check environment, hardware, and config validity.",
        description=(
            "Run a series of health checks: Python version, GPU/CUDA "
            "availability, config file presence and schema, experiment "
            "registry integrity, and hard-rules file."
        ),
    )
    p_doctor.add_argument(
        "--project-dir",
        default=".",
        metavar="DIR",
        help="Root of the target project (default: current directory).",
    )
    p_doctor.add_argument(
        "--fix",
        action="store_true",
        help=(
            "Attempt environment remediation automatically (create/use project "
            "venv per onboarding prefs and install required packages) before checks."
        ),
    )
    p_doctor.set_defaults(func=_cmd_doctor)

    # -- dashboard -----------------------------------------------------------
    p_dashboard = sub.add_parser(
        "dashboard",
        help="Start the local web dashboard.",
        description=(
            "Launch the DRL AutoResearch web dashboard which streams live "
            "experiment metrics, logs, and plan status."
        ),
    )
    p_dashboard.add_argument(
        "--project-dir",
        default=".",
        metavar="DIR",
        help="Root of the target project (default: current directory).",
    )
    p_dashboard.add_argument(
        "--port",
        type=int,
        default=8765,
        metavar="PORT",
        help="TCP port for the dashboard server (default: 8765).",
    )
    p_dashboard.set_defaults(func=_cmd_dashboard)

    # -- run -----------------------------------------------------------------
    p_run = sub.add_parser(
        "run",
        help="Start the autonomous experiment loop.",
        description=(
            "Begin executing the research plan: propose hypotheses, run "
            "DRL training experiments, evaluate results, and iterate."
        ),
    )
    p_run.add_argument(
        "--project-dir",
        default=".",
        metavar="DIR",
        help="Root of the target project (default: current directory).",
    )
    p_run.add_argument(
        "--parallel",
        type=int,
        default=1,
        metavar="N",
        help="Maximum number of experiments to run concurrently (default: 1).",
    )
    p_run.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate the plan and print what would be executed without "
            "actually running any experiments."
        ),
    )
    p_run.add_argument(
        "--once",
        action="store_true",
        help="Run a single autonomous agent cycle and exit.",
    )
    p_run.add_argument(
        "--agent-backend",
        choices=["auto", "codex", "claude"],
        default="auto",
        help="Which coding-agent CLI to use for autonomous cycles (default: auto).",
    )
    p_run.set_defaults(func=_cmd_run)

    # -- status --------------------------------------------------------------
    p_status = sub.add_parser(
        "status",
        help="Print current project status summary.",
        description=(
            "Display a summary of the project: active experiments, "
            "completed runs, best-performing checkpoint, and next planned "
            "action."
        ),
    )
    p_status.add_argument(
        "--project-dir",
        default=".",
        metavar="DIR",
        help="Root of the target project (default: current directory).",
    )
    p_status.set_defaults(func=_cmd_status)

    # -- plan ----------------------------------------------------------------
    p_plan = sub.add_parser(
        "plan",
        help="Show or refresh the implementation plan.",
        description=(
            "Display the current research plan stored in "
            ".drl_autoresearch/. Pass --refresh to regenerate the plan "
            "from the latest experiment results."
        ),
    )
    p_plan.add_argument(
        "--project-dir",
        default=".",
        metavar="DIR",
        help="Root of the target project (default: current directory).",
    )
    p_plan.add_argument(
        "--refresh",
        action="store_true",
        help="Regenerate the plan from the latest results before displaying.",
    )
    p_plan.set_defaults(func=_cmd_plan)

    # -- research ------------------------------------------------------------
    p_research = sub.add_parser(
        "research",
        help="Trigger a mid-training literature/research refresh.",
        description=(
            "Query the research knowledge base and update the plan with "
            "newly relevant techniques or findings. Can be called while "
            "training is running."
        ),
    )
    p_research.add_argument(
        "--project-dir",
        default=".",
        metavar="DIR",
        help="Root of the target project (default: current directory).",
    )
    p_research.set_defaults(func=_cmd_research)

    # -- resume --------------------------------------------------------------
    p_resume = sub.add_parser(
        "resume",
        help="Recover a dropped session with compact sync, then continue run loop.",
        description=(
            "Run a token-saving session sync for interrupted/new sessions: "
            "print status, tail key logs (registry/journal/handoffs/incidents), "
            "emit a compact checkpoint, then continue autonomous run by default."
        ),
    )
    p_resume.add_argument(
        "--project-dir",
        default=".",
        metavar="DIR",
        help="Root of the target project (default: current directory).",
    )
    p_resume.add_argument(
        "--parallel",
        type=int,
        default=1,
        metavar="N",
        help="Maximum number of experiments to run concurrently when continuing (default: 1).",
    )
    p_resume.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "After resume sync, validate and print planned execution without "
            "running experiments."
        ),
    )
    p_resume.add_argument(
        "--no-run",
        action="store_true",
        help="Only perform resume sync/checkpoint; do not start the run loop.",
    )
    p_resume.add_argument(
        "--agent-backend",
        choices=["auto", "codex", "claude"],
        default="auto",
        help="Which coding-agent CLI to use when continuing after resume (default: auto).",
    )
    p_resume.set_defaults(func=_cmd_resume)

    # -- stop ----------------------------------------------------------------
    p_stop = sub.add_parser(
        "stop",
        help="Request the autonomous gateway to stop and save a compact resume brief.",
        description=(
            "Signal the active autonomous run loop to stop after the current cycle, "
            "and record a compact summary of what it was doing and what should happen next."
        ),
    )
    p_stop.add_argument(
        "--project-dir",
        default=".",
        metavar="DIR",
        help="Root of the target project (default: current directory).",
    )
    p_stop.add_argument(
        "--brief",
        default="",
        metavar="TEXT",
        help="Optional compact override for the 'what next' part of the saved stop brief.",
    )
    p_stop.set_defaults(func=_cmd_stop)

    # -- check ---------------------------------------------------------------
    p_check = sub.add_parser(
        "check",
        help="Check whether an agent action violates the hard rules.",
        description=(
            "Used internally by autonomous agents to gate potentially "
            "destructive actions against NON_NEGOTIABLE_RULES.md and "
            "permissions.yaml before execution."
        ),
    )
    p_check.add_argument(
        "--project-dir",
        default=".",
        metavar="DIR",
        help="Root of the target project (default: current directory).",
    )
    p_check.add_argument(
        "--action",
        required=True,
        metavar="ACTION",
        help=(
            "Short string identifying the action to validate, e.g. "
            "\"delete_checkpoint\" or \"modify_policy\"."
        ),
    )
    p_check.add_argument(
        "--details",
        default=None,
        metavar="JSON",
        help=(
            "Optional JSON object with additional context for the rule "
            "check, e.g. '{\"path\": \"checkpoints/run_42\"}'."
        ),
    )
    p_check.set_defaults(func=_cmd_check)

    return parser


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    """Parse arguments and dispatch to the appropriate subcommand.

    Parameters
    ----------
    argv:
        Argument list (defaults to ``sys.argv[1:]`` when ``None``).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        exit_code: int = args.func(args)
    except KeyboardInterrupt:
        print()
        console("Interrupted by user.", "warning")
        sys.exit(130)
    except Exception as exc:  # noqa: BLE001
        console(f"Unexpected error: {exc}", "error")
        if os.environ.get("DRL_DEBUG"):
            import traceback
            traceback.print_exc()
        sys.exit(1)

    sys.exit(exit_code if isinstance(exit_code, int) else 0)
