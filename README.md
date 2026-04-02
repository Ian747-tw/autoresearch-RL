# DRL AutoResearch

**Autonomous deep reinforcement learning research — from spec to results overnight**

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Platform: Linux / GPU](https://img.shields.io/badge/platform-Linux%20%7C%20GPU-lightgrey)

---

![DRL AutoResearch Dashboard](edited-photo(1).png)

*Live dashboard showing 6 CartPole REINFORCE experiments: training curves, eval curves, resource usage, experiment timeline, and top-models leaderboard.*

---

## Overview

DRL AutoResearch is an installable Python package (`drl-autoresearch`) that brings the AutoResearch loop — spec → plan → change → run → evaluate → log → iterate — to deep reinforcement learning projects. It is designed to run overnight, unattended, and to hand you a structured summary of what was tried, what worked, and what to do next.

Where the original Karpathy-style AutoResearch workflow gives you a general-purpose research automation scaffold, DRL AutoResearch adds the infrastructure that DRL specifically demands: reward shaping protocols, exploration failure diagnosis, checkpoint selection without data leakage, environment debugging playbooks, and a hard rule enforcement layer that prevents the autonomous loop from making expensive or irreversible mistakes.

The core philosophy is preserved: local-first, repo-native, markdown-guided, minimal ceremony, fast iteration. No databases, no daemon processes, no cloud APIs. All state lives inside your project as plain files.

---

## Architecture

```
User Spec → Onboarding → Research Plan → Experiment Loop → Results
                                               ↕
                               Orchestrator (single authority)
                                     ↕           ↕
                               Policy Engine   Dashboard
                               (hard rules)   (live, port 8765)
                                     ↕
                               Experiment Registry + Journal
```

The orchestrator owns phase progression (`research` → `baseline` → `experimenting` → `focused_tuning` → `ablation` → `converged`), experiment scheduling, and research refresh triggers. Every action an agent proposes is checked against `NON_NEGOTIABLE_RULES.md` by the PolicyEngine before it executes. Results are written to the experiment registry. The dashboard reads those files and streams updates to the browser in real time.

---

## Key Features

### 1. Interactive Onboarding TUI

Running `drl-autoresearch init` scaffolds all required project files. With `questionary` installed (`[tui]` extra), it launches an interactive terminal questionnaire that collects:

- Project name, environment description, and research objective
- Hardware configuration — GPUs, VRAM, CPU count, RAM — auto-detected with manual override
- Python environment path and package manager preference
- Permission policy (see [Permission Modes](#permission-modes))

Pass `--auto` for fully non-interactive CI-compatible scaffolding.

### 2. Single Orchestrator

One decision authority manages the entire lifecycle. The orchestrator reads project state from `.drl_autoresearch/state.json`, selects the next experiment based on the current research phase and the experiment registry, schedules worker processes, and decides when to trigger a research refresh. There are no competing agents and no distributed coordination.

Phase transitions are explicit and logged. The orchestrator will not move to `focused_tuning` until a viable baseline is established, and will not declare `converged` until ablations have been run.

### 3. Hard Rule Enforcement via PolicyEngine

Rules defined in `NON_NEGOTIABLE_RULES.md` are enforced before every agent action:

```bash
drl-autoresearch check --action <type> [--details '<json>']
```

The PolicyEngine returns `ALLOWED` (exit 0) or `BLOCKED` (exit 1). The permission mode controls which action types require confirmation. Violations are printed to stderr with the specific rule that was violated and appended to the audit log.

### 4. Structured Experiment Registry

Every run is appended to `logs/experiment_registry.tsv` — a tab-separated file with 27 fields per row:

`run_id`, `parent_run_id`, `timestamp`, `agent`, `branch`, `commit`, `environment`, `algorithm`, `config_summary`, `change_summary`, `hypothesis`, `rules_checked`, `train_reward_mean`, `train_reward_std`, `eval_reward_mean`, `eval_reward_std`, `custom_metric_name`, `custom_metric_value`, `success_rate`, `constraint_violations`, `seed_count`, `wall_clock_seconds`, `gpu_memory_gb`, `ram_gb`, `status`, `keep_decision`, `notes`

Appends are file-locked and atomic. Multiple workers can write concurrently without corruption.

### 5. Live Localhost Dashboard

The dashboard runs on port 8765 using Python's standard library HTTP server with Server-Sent Events for live updates. No external web framework required. Chart.js provides dark-theme charts in the browser.

The dashboard shows:

- Training reward curves (all runs, per-experiment)
- Evaluation curves with live updates
- GPU and CPU resource utilization
- Experiment timeline with keep/discard decisions
- Top models leaderboard (sorted by eval reward)
- Open incidents with severity
- Next planned experiment from orchestrator state

### 6. Optional Parallel Background Experiments

The orchestrator manages multiple isolated experiment workers running concurrently. Each worker receives a dedicated GPU assignment (`CUDA_VISIBLE_DEVICES`), an explicit VRAM budget enforced before the run starts, and health monitoring for hangs and OOM kills. Parallelism is opt-in: `drl-autoresearch run --parallel N`.

### 7. Claude Code Slash Commands

Five built-in commands live in `.claude/commands/` and are available in any Claude Code session opened from inside this repo:

| Command | Description |
|---------|-------------|
| `/drl-init` | Scaffold a new DRL project interactively |
| `/drl-run` | Start the autonomous overnight experiment loop |
| `/drl-diagnose` | Diagnose a failure or confusing result |
| `/drl-research` | Trigger a mid-training research refresh |
| `/drl-plan` | Review and update the current experiment plan |

---

## Installation

### Requirements

- Python 3.10 or later
- Linux (primary support; macOS has partial support)
- NVIDIA GPU recommended — CPU fallback available but slow
- `uv` (recommended) or `pip`

### Install from Source

```bash
git clone https://github.com/Ian747-tw/autoresearch-RL
cd autoresearch-RL

# with uv (recommended):
uv pip install -e .

# or with pip:
pip install -e .

# verify:
drl-autoresearch --version
```

### Without Installing (uv run)

If you don't want to install the package globally, every command works via `uv run`:

```bash
uv run python -m drl_autoresearch <command>
# e.g.:
uv run python -m drl_autoresearch doctor
uv run python -m drl_autoresearch dashboard --port 8765
```

### Optional Extras

```bash
pip install -e ".[tui]"        # adds questionary for the interactive onboarding TUI
pip install -e ".[full]"       # tui + aiohttp extras
```

The base install has zero non-stdlib runtime dependencies.

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/Ian747-tw/autoresearch-RL
cd autoresearch-RL
pip install -e .

# 2. Go to your DRL project (or stay in autoresearch-RL to test)
cd ~/my-rl-project

# 3. Initialize — scaffolds all config files
drl-autoresearch init

# 4. Verify environment — checks GPU, Python, and config
drl-autoresearch doctor

# 5. Start the live dashboard in the background
drl-autoresearch dashboard &
# open http://localhost:8765

# 6. Run a dry-run first to validate the plan
drl-autoresearch run --dry-run

# 7. Start the autonomous research loop
drl-autoresearch run
```

### Test with the Included Training Script

A self-contained CartPole training harness is included for testing the full pipeline end-to-end:

```bash
# Runs 3 REINFORCE experiments and writes all data to the dashboard
python test_train.py --updates 50

# Fewer updates for a quick smoke test (~10s)
python test_train.py --updates 5
```

The script implements CartPole-v1 physics from scratch — no `gymnasium` dependency required. It writes training curves, eval curves, and registry entries in the exact format the dashboard expects, so you can see the full picture immediately.

---

## Claude Code Setup

After cloning the repo, the `.claude/commands/` directory contains five DRL slash commands that are available immediately in any Claude Code session opened from inside `autoresearch-RL`.

### Step 1 — Open the project in Claude Code

```bash
cd autoresearch-RL
claude  # opens Claude Code in this directory
```

### Step 2 — Initialize your DRL project

```
/drl-init
```

This runs `drl-autoresearch init`, scaffolds all config files, and writes `NON_NEGOTIABLE_RULES.md`. Claude Code will walk you through the setup interactively.

### Step 3 — Verify and start

```
/drl-run
```

The agent checks the environment (`doctor`), reviews the plan, and starts the experiment loop. It checks every proposed action against `NON_NEGOTIABLE_RULES.md` before executing.

### Available slash commands

```
/drl-init       — Scaffold config, initialize state, set permission mode
/drl-run        — Start autonomous experiment loop (plan → run → log → repeat)
/drl-diagnose   — Diagnose a training failure using the skill playbooks
/drl-plan       — Review and refresh the current ranked experiment plan
/drl-research   — Trigger a mid-training literature/implementation refresh
```

### Copy commands to your own project

```bash
cp -r .claude/commands/ ~/my-rl-project/.claude/commands/
```

The commands will then be available in Claude Code sessions opened from `~/my-rl-project`.

---

## Codex Setup

All CLI commands work with Codex exactly as with Claude Code — the shared core (orchestrator, policy engine, logs) is agent-agnostic.

### Step 1 — Install the Codex CLI

```bash
npm install -g @openai/codex
# set your API key:
export OPENAI_API_KEY=sk-...
```

### Step 2 — Run non-interactively

Codex's default sandbox uses `bwrap` which requires unprivileged user namespaces. On some Linux systems (containers, restricted VMs) this is not available. Use `--dangerously-bypass-approvals-and-sandbox` to skip it:

```bash
codex exec --dangerously-bypass-approvals-and-sandbox \
  -c 'shell_environment_policy.inherit=all' \
  "Run drl-autoresearch doctor, then run test_train.py --updates 50 from /path/to/autoresearch-RL, and report the output."
```

### Step 3 — Multi-step research session

```bash
codex exec --dangerously-bypass-approvals-and-sandbox \
  -c 'shell_environment_policy.inherit=all' \
  "You are running a DRL AutoResearch session in /path/to/my-rl-project.

  1. Run: drl-autoresearch doctor
  2. Run: drl-autoresearch plan
  3. Run: drl-autoresearch run --dry-run
  4. If all checks pass, run: drl-autoresearch run --parallel 2
  5. After completion, run: drl-autoresearch status
  
  Check every proposed code change against drl-autoresearch check before executing it.
  Report the output of each step."
```

### Interactive Codex session

```bash
cd autoresearch-RL
codex  # opens interactive session; Codex will read AGENT.md if present
```

---

## Project Scaffold Structure

After `drl-autoresearch init`, the following files are created inside your project:

```
your-drl-project/
├── .drl_autoresearch/
│   ├── state.json          ← live project state (phase, best run, counters)
│   ├── policy.yaml         ← permission policy and action overrides
│   ├── hardware.yaml       ← auto-detected hardware configuration
│   ├── python_env.yaml     ← Python environment and package manager config
│   └── permissions.yaml    ← current permission mode
│
├── NON_NEGOTIABLE_RULES.md ← hard rules enforced by PolicyEngine
│
├── skills/                 ← agent playbooks (add your own .md files here)
│
└── logs/
    ├── experiment_registry.tsv   ← 27-column run history (append-only)
    └── runs/
        └── <run_id>/
            ├── run_meta.json     ← run parameters and git state
            ├── run_result.json   ← final metrics and outcome
            ├── metrics.jsonl     ← per-step metric stream
            └── checkpoints/      ← checkpoint metadata (max 3 per run)
```

The scaffold does not touch your existing source files. Everything is additive.

### Artifact files for dashboard curves

The dashboard reads full training curves from `logs/artifacts/<run_id>/metrics.json`. Write this file from your training script to populate the charts:

```json
{
  "steps":        [100, 200, 300, ...],
  "rewards":      [12.3, 18.7, 24.1, ...],
  "losses":       [0.42, 0.38, 0.31, ...],
  "eval_steps":   [500, 1000, ...],
  "eval_rewards": [22.5, 31.0, ...]
}
```

See `test_train.py` for a complete working example.

---

## CLI Reference

```
drl-autoresearch install
    Print install confirmation and quick-start tips.

drl-autoresearch init [--project-dir DIR] [--skip-onboarding] [--auto]
    Scaffold project files: .drl_autoresearch/, NON_NEGOTIABLE_RULES.md,
    logs/experiment_registry.tsv, skills/.
    --skip-onboarding   Skip the interactive questionnaire, use defaults.
    --auto              Fully non-interactive; implies --skip-onboarding.

drl-autoresearch doctor [--project-dir DIR]
    Run 14 health checks: Python version, PyTorch, GPU/CUDA, NumPy, pandas,
    config files, state.json, registry, rules file, skills directory.
    Prints pass/fail for each check. Exits 0 only if all pass.

drl-autoresearch dashboard [--project-dir DIR] [--port PORT]
    Start the live dashboard HTTP server (stdlib-only, no external framework).
    Default port: 8765. Open http://localhost:8765 in any browser.

drl-autoresearch run [--project-dir DIR] [--parallel N] [--dry-run]
    Start the autonomous experiment loop.
    --parallel N    Run up to N experiments concurrently (default: 1).
    --dry-run       Print what would run without executing. Safe to use anytime.

drl-autoresearch status [--project-dir DIR]
    Show current phase, run counters, best run, recent experiments, active workers.

drl-autoresearch plan [--project-dir DIR] [--refresh]
    Display the ranked experiment plan from .drl_autoresearch/plan.json.
    --refresh   Regenerate the plan from the latest state before displaying.

drl-autoresearch research [--project-dir DIR]
    Trigger a mid-training research refresh.

drl-autoresearch check --action ACTION [--details JSON] [--project-dir DIR]
    Check whether an action is permitted under the current policy.
    Exits 0 (ALLOWED) or 1 (BLOCKED).
    --action    Action type: edit_reward | edit_eval | edit_env | install_package |
                update_package | global_install | exceed_compute | gpu_memory_risk |
                silent_cpu_fallback | eval_protocol_change | use_privileged_info | custom
    --details   Optional JSON with action context, e.g. '{"phase": "init"}'.
```

---

## Permission Modes

Set in `.drl_autoresearch/permissions.yaml` during init or updated manually.

| Mode | Description | Use case |
|------|-------------|----------|
| `locked` | All sensitive actions require explicit human approval | **Default** — shared machines, sensitive codebases |
| `prompted` | Ask before each install or file change; allow reads freely | Research projects where you want visibility |
| `bootstrap-only` | Installs allowed during `init` only; lock down afterward | After initial setup on a shared machine |
| `open` | Allow all actions; fully autonomous operation | Dedicated overnight research sessions |
| `project-only` | Installs and changes restricted to the project's virtualenv | Shared machines where system packages must not be touched |

Change the mode by editing `.drl_autoresearch/permissions.yaml`:

```yaml
mode: open   # or: locked | prompted | bootstrap-only | project-only
```

---

## Design Principles

**Lightweight.** A single installable package. No services, no databases, no cloud accounts. `pip install -e .` and you are done.

**Repo-native.** All state lives inside the target project as plain files. The project is self-contained. You can `git add logs/` to version-control your research history, or `.gitignore` it.

**Markdown-guided.** `NON_NEGOTIABLE_RULES.md` and the `skills/` playbooks are the interface between the system and the agent. They are human-readable and human-editable. Changing agent behavior means editing a markdown file, not touching Python internals.

**Autonomous but controlled.** Hard rules are enforced, not suggested. The PolicyEngine does not print a warning and continue — it blocks. The system is designed to be left running overnight with confidence that it will not install arbitrary packages or violate your stated constraints.

**Fast iteration.** The experiment loop is the core product. Everything else — the dashboard, the TUI, the CLI — exists to make the loop faster, safer, and easier to understand after the fact.

---

## FAQ

**Q: Does this replace the original `program.md` / `train.py` workflow?**

No. It extends it. The original AutoResearch loop still works exactly as before. DRL AutoResearch adds a structured scaffold on top: orchestration, policy enforcement, and durable logging. You can adopt it incrementally.

**Q: Can I use this with non-DRL ML projects?**

Yes. The orchestrator, policy engine, and experiment registry are general. The DRL skills are just markdown files in `skills/`. Add your own playbooks, remove the DRL-specific ones, or replace them entirely.

**Q: What happens if an agent tries to violate a hard rule?**

`drl-autoresearch check` exits 1 and prints which rule was violated. The agent must stop or choose a different action. The violation is appended to `.drl_autoresearch/policy_audit.log`.

**Q: Is the dashboard required?**

No. All state is in files — `logs/experiment_registry.tsv` is spreadsheet-importable. The dashboard is a convenience layer, not a dependency.

**Q: Can multiple AI agents work on the same project?**

Yes. Claude Code and Codex share the same log files, state, and policy. One agent can initialize and run experiments; another can pick up where it left off by reading `state.json` and the registry.

**Q: What if a run crashes mid-training?**

The RunManager logs the incident. On the next `drl-autoresearch run`, the orchestrator reads the registry and decides whether to retry or skip. You can inspect the state with `drl-autoresearch status`.

---

## Contributing

The skills pack is the easiest place to contribute. If you have a DRL failure mode that is not covered, add a markdown file to `skills/`. Pull requests adding new skills are welcome without requiring changes to any Python code.

For bugs in the orchestrator, policy engine, or CLI, open an issue with the output of `drl-autoresearch doctor` and a description of the failure.

---

## License

MIT. See `LICENSE` for details.
