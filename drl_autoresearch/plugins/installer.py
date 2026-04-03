"""
Plugin installer — copies bundled Claude Code / Codex plugin files into the
user's project directory.

Called by ``drl-autoresearch init --plugin [cc|codex|both]``.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from drl_autoresearch.cli import console

_PLUGINS_DIR = Path(__file__).parent
_BACKEND_DIR = _PLUGINS_DIR / "backend"
_SKILL_GENERATOR_TEMPLATE = _BACKEND_DIR / "skill_generator.md"


def install(project_dir: Path, plugin: str) -> int:
    """Install plugins into *project_dir*.

    Parameters
    ----------
    project_dir:
        Root of the user's DRL project.
    plugin:
        ``"cc"`` — Claude Code only.
        ``"codex"`` — Codex only.
        ``"both"`` — both.

    Returns 0 on success, 1 on error.
    """
    project_dir = Path(project_dir).resolve()
    ok = True

    if plugin in ("cc", "both"):
        ok = _install_cc(project_dir) and ok

    if plugin in ("codex", "both"):
        ok = _install_codex(project_dir) and ok

    return 0 if ok else 1


def install_skill_generator_backend(project_dir: Path, context: str) -> bool:
    """Install the backend skill generator prompt into the target project."""
    project_dir = Path(project_dir).resolve()
    src = _SKILL_GENERATOR_TEMPLATE
    dst = project_dir / ".drl_autoresearch" / "backend" / "skill_generator.md"

    if not src.is_file():
        console("Skill generator backend source not found in package.", "error")
        return False

    template = src.read_text(encoding="utf-8").rstrip()
    content = f"{template}\n\n## Project Context\n\n{context.strip()}\n"

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(content, encoding="utf-8")
    console("Installed: .drl_autoresearch/backend/skill_generator.md", "success")
    return True


def remove_managed_plugin_files(project_dir: Path) -> list[Path]:
    """Remove plugin files managed by DRL AutoResearch from the target project."""
    project_dir = Path(project_dir).resolve()
    removed: list[Path] = []

    for rel_path in (
        Path("AGENT.md"),
        Path(".claude/commands/drl-init.md"),
        Path(".claude/commands/drl-run.md"),
        Path(".claude/commands/drl-plan.md"),
        Path(".claude/commands/drl-diagnose.md"),
        Path(".claude/commands/drl-research.md"),
    ):
        path = project_dir / rel_path
        if path.exists():
            path.unlink()
            removed.append(path)

    for rel_dir in (Path(".claude/commands"), Path(".claude")):
        path = project_dir / rel_dir
        if path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass

    return removed


def _install_cc(project_dir: Path) -> bool:
    """Copy bundled Claude Code command files to <project>/.claude/commands/."""
    src = _PLUGINS_DIR / "claude_code" / "commands"
    dst = project_dir / ".claude" / "commands"

    if not src.is_dir():
        console("Claude Code plugin source not found in package.", "error")
        return False

    dst.mkdir(parents=True, exist_ok=True)
    count = 0
    for md_file in src.glob("*.md"):
        dest_file = dst / md_file.name
        if dest_file.exists():
            console(f"Skipped (exists): .claude/commands/{md_file.name}", "warning")
        else:
            shutil.copy2(md_file, dest_file)
            console(f"Installed: .claude/commands/{md_file.name}", "success")
            count += 1

    if count:
        console(
            f"Claude Code plugin ready — {count} slash commands installed in "
            f".claude/commands/. Open this project in Claude Code and use "
            f"/drl-init, /drl-run, /drl-plan, /drl-diagnose, /drl-research.",
            "info",
        )
    return True


def _install_codex(project_dir: Path) -> bool:
    """Copy bundled AGENT.md to <project>/AGENT.md."""
    src = _PLUGINS_DIR / "codex" / "AGENT_TEMPLATE.md"
    if not src.is_file():
        # Backward compatibility for older local plugin bundles.
        src = _PLUGINS_DIR / "codex" / "AGENT.md"
    dst = project_dir / "AGENT.md"

    if not src.is_file():
        console("Codex plugin source not found in package.", "error")
        return False

    if dst.exists():
        console("Skipped (exists): AGENT.md", "warning")
    else:
        shutil.copy2(src, dst)
        console(
            "Codex plugin ready — AGENT.md installed. "
            "Run `codex` or `codex exec` in this directory — "
            "Codex will read AGENT.md automatically.",
            "success",
        )
    return True


def prompt_and_install(project_dir: Path, auto: bool = False) -> int:
    """Interactively ask which plugin(s) to install, then install them.

    In ``--auto`` mode skips the prompt and installs both.
    Returns 0 on success.
    """
    if auto:
        return install(project_dir, "both")

    print()
    print("  Which AI agent plugin(s) do you want to install?")
    print("  [1] Claude Code  (/drl-init, /drl-run, … slash commands)")
    print("  [2] Codex        (AGENT.md operating guide)")
    print("  [3] Both         (recommended)")
    print("  [4] None         (skip)")
    try:
        raw = input("  Choice [3]: ").strip() or "3"
    except (EOFError, KeyboardInterrupt):
        print()
        return 0  # non-fatal; just skip plugin install

    mapping = {"1": "cc", "2": "codex", "3": "both", "4": None}
    choice = mapping.get(raw)
    if choice is None:
        console("Plugin install skipped.", "info")
        return 0

    return install(project_dir, choice)
