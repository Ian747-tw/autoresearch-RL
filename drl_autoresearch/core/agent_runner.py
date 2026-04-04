"""
Agent runner for DRL AutoResearch.

Bridges the controller loop to an external coding agent CLI such as Codex or
Claude Code while preserving the existing project file backbone.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


_AUTONOMOUS_POLICIES = {"open", "project-only", "bootstrap-only"}
_DEFAULT_TIMEOUT_SECONDS = 4 * 60 * 60


@dataclass
class AgentRunResult:
    backend: str
    ok: bool
    exit_code: int
    stdout_log: Path
    stderr_log: Path
    command: list[str]


def load_onboarding_context(project_dir: Path) -> dict[str, Any]:
    config_dir = Path(project_dir) / ".drl_autoresearch"
    for path in (config_dir / "onboarding.yaml", config_dir / "onboarding.json"):
        if not path.exists():
            continue
        try:
            if path.suffix == ".yaml":
                import yaml  # type: ignore

                data = yaml.safe_load(path.read_text(encoding="utf-8"))
            else:
                data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            continue
    return {}


def load_permission_policy(project_dir: Path) -> str:
    onboarding = load_onboarding_context(project_dir)
    permissions = onboarding.get("permissions", {})
    if isinstance(permissions, dict):
        policy = permissions.get("policy")
        if isinstance(policy, str) and policy.strip():
            return policy.strip()
    return "open"


def detect_backend(project_dir: Path, preferred: Optional[str] = None) -> Optional[str]:
    project_dir = Path(project_dir)
    candidates: list[str] = []
    if preferred and preferred != "auto":
        candidates.append(preferred)
    else:
        if (project_dir / "AGENT.md").exists():
            candidates.append("codex")
        if (project_dir / ".claude" / "commands").is_dir():
            candidates.append("claude")
        candidates.extend(["codex", "claude"])

    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate == "codex" and shutil.which("codex"):
            return "codex"
        if candidate == "claude" and shutil.which("claude"):
            return "claude"
    return None


def autonomous_policy_allowed(project_dir: Path) -> bool:
    return load_permission_policy(project_dir) in _AUTONOMOUS_POLICIES


def build_agent_prompt(
    project_dir: Path,
    state: dict[str, Any],
    experiment: dict[str, Any],
    project_mode: str,
) -> str:
    run_id = str(experiment.get("run_id", "agent-cycle"))
    hypothesis = str(experiment.get("hypothesis", "No hypothesis provided."))
    metric_name = str(experiment.get("metric_name", state.get("best_metric_name", "reward")))
    params = experiment.get("params", {})
    flags = state.get("flags", {}) if isinstance(state.get("flags"), dict) else {}
    build_bootstrap_complete = bool(flags.get("build_bootstrap_complete", project_mode != "build"))

    mode_line = (
        "Build mode is active. Prioritize making the project runnable, implementing "
        "missing training/build pieces, and executing the next meaningful build or "
        "training step before moving to tuning."
        if project_mode == "build" and not build_bootstrap_complete
        else "Improve mode is active. Prioritize the next highest-signal improvement iteration."
    )

    return f"""You are the execution engine for DRL AutoResearch.

Work inside this project directory: {project_dir}

This is a single autonomous cycle. Do not call `drl-autoresearch run` or `drl-autoresearch resume` recursively.
Use the existing project backbone instead of redesigning structure:
- `.drl_autoresearch/`
- `logs/`
- `skills/`
- `NON_NEGOTIABLE_RULES.md`
- `USER_SPEC.md` / `spec_compact.md` if present
- dashboard artifact paths under `logs/artifacts/`

Before acting:
1. Read `NON_NEGOTIABLE_RULES.md`.
2. Read `.drl_autoresearch/spec_compact.md` if present, then follow pointed source files when needed.
3. Read `AGENT.md` if present.
4. If `.claude/commands/drl-run.md` exists, follow its project workflow guidance.
5. Read the latest project state from `.drl_autoresearch/state.json` and recent registry/journal tails.
6. Inspect `skills/` and consult any relevant project skills before acting. Do not assume any specific skill filenames.
7. At the very start of the cycle, verify whether this run should use GPU or CPU and record that decision before the main work.

Execution mode:
- Current project mode: {project_mode}
- Current phase: {state.get("current_phase", "research")}
- Best metric: {state.get("best_metric_name", "reward")}={state.get("best_metric_value")}
- {mode_line}

Starting posture:
- The platform provides backbone, rules, logs, compact context, and orchestration. You own the actual research/build direction.
- In build mode, start from the spec and current codebase, identify what is missing, and decide the smallest high-signal research/build step yourself.
- In build mode, create or update `implementation_plan/IMPLEMENTATION_PLAN.md` as a simple free-form plan the user can inspect. Do not use a canned template; just keep it concise and current.
- Prefer GPU when it is available and materially beneficial. For very short or lightweight training/eval where CPU is genuinely the faster or better choice, CPU is allowed, but record the choice and rationale explicitly.
- If GPU should be usable but is currently unresolved, fix that first before doing the main experiment loop work.
- If useful skills are present, use them naturally. If a reusable skill is missing and would materially help future cycles, you may create it.
- Avoid canned algorithm templates unless the project context clearly justifies them.

Assigned experiment candidate:
- run_id: {run_id}
- metric_name: {metric_name}
- hypothesis: {hypothesis}
- params_json: {json.dumps(params, ensure_ascii=False)}

Required outcome for this cycle:
- Perform one meaningful autonomous coding/training/research iteration.
- If code changes are needed, make them directly.
- Run the relevant build/training/eval commands.
- Before any risky action or code/config edit, run `drl-autoresearch check --project-dir . --action <type> --details '<json>'`.
- If a check is blocked, do not proceed with that action.
- Use helper APIs only for backbone writes:
  - `drl_autoresearch.logging.registry.ExperimentRegistry`
  - `drl_autoresearch.logging.journal.ProjectJournal`
  - `drl_autoresearch.logging.incidents.IncidentLog`
  - `drl_autoresearch.logging.handoffs.HandoffLog`
- Update the experiment registry through `ExperimentRegistry`, never by raw TSV append.
- If training/eval metrics exist, write `logs/artifacts/<run_id>/metrics.json` when appropriate for dashboard curves.
- Update journal/incidents/handoffs through their helper APIs only when the cycle warrants it.
- After consulting or creating a skill, record that consultation with:
  `python - <<'PY'\nfrom drl_autoresearch.core.agent_contract import record_skill_consultation\nrecord_skill_consultation('skills/<relevant-file>.md', 'why it was relevant')\nPY`
- Update live progress during the cycle with:
  `python - <<'PY'\nfrom drl_autoresearch.core.agent_contract import update_runtime_activity\nupdate_runtime_activity('building', 'implementing missing training loop')\nPY`
  Use short activity labels like `reading_spec`, `building`, `training`, `evaluating`, `debugging`, `writing_plan`.
- Update runtime device / GPU resolution state with:
  `python - <<'PY'\nfrom drl_autoresearch.core.agent_contract import update_runtime_gpu_status\nupdate_runtime_gpu_status('gpu', 'solved', 'cuda available and selected for training')\nPY`
  Use `device` in `{{gpu,cpu,unknown}}` and `resolution_status` in `{{solving,solved}}`.
- Do not edit `.drl_autoresearch/state.json` manually; the controller will sync state after this run.

Important:
- Never stop just because a skill script is missing; do the work directly as the coding agent.
- Keep moving the project forward within the current backbone.
- This cycle is audited automatically. If you bypass `drl-autoresearch check` or the helper APIs, the controller will mark the cycle failed.
- Final response should be a concise summary of what changed, what ran, and the registry status.
"""


def run_agent_cycle(
    project_dir: Path,
    backend: str,
    prompt: str,
    env: Optional[dict[str, str]] = None,
    dangerous: bool = True,
    timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
) -> AgentRunResult:
    project_dir = Path(project_dir).resolve()
    logs_dir = project_dir / "logs" / "agent_sessions"
    logs_dir.mkdir(parents=True, exist_ok=True)

    stamp = subprocess.run(
        ["date", "+%Y%m%d-%H%M%S"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip() or "session"

    stdout_log = logs_dir / f"{stamp}.{backend}.stdout.log"
    stderr_log = logs_dir / f"{stamp}.{backend}.stderr.log"

    if backend == "codex":
        command = [
            "codex",
            "exec",
            "--cd",
            str(project_dir),
            "--skip-git-repo-check",
        ]
        if dangerous:
            command.append("--dangerously-bypass-approvals-and-sandbox")
        else:
            command.append("--full-auto")
        command.append(prompt)
    elif backend == "claude":
        command = [
            "claude",
            "-p",
        ]
        if dangerous:
            command.append("--dangerously-skip-permissions")
        else:
            command.extend(["--permission-mode", "auto"])
        command.append(prompt)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    completed = subprocess.run(
        command,
        cwd=project_dir,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        env=merged_env,
    )
    stdout_log.write_text(completed.stdout or "", encoding="utf-8")
    stderr_log.write_text(completed.stderr or "", encoding="utf-8")

    return AgentRunResult(
        backend=backend,
        ok=completed.returncode == 0,
        exit_code=completed.returncode,
        stdout_log=stdout_log,
        stderr_log=stderr_log,
        command=command,
    )
