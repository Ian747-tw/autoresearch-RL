# DRL AutoResearch

**Autonomous deep reinforcement learning research — from spec to results overnight**

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Platform: Linux / GPU](https://img.shields.io/badge/platform-Linux%20%7C%20GPU-lightgrey)

---

## Overview

DRL AutoResearch is an installable Python plugin (`drl-autoresearch`) that brings the AutoResearch loop — spec → plan → change → run → evaluate → log → iterate — to deep reinforcement learning projects. It is designed to run overnight, unattended, and to hand you a structured summary of what was tried, what worked, and what to do next.

Where the original Karpathy-style AutoResearch workflow gives you a general-purpose research automation scaffold, DRL AutoResearch adds the infrastructure that DRL specifically demands: reward shaping protocols, exploration failure diagnosis, checkpoint selection without data leakage, environment debugging playbooks, and a hard rule enforcement layer that prevents the autonomous loop from making expensive or irreversible mistakes. It is not a rewrite — it is a strict extension. The original loop still works unchanged.

The core philosophy is preserved: local-first, repo-native, markdown-guided, minimal ceremony, fast iteration. No databases, no daemon processes, no cloud APIs. All state lives inside your project as plain files. The orchestrator is a single decision authority — no competing agents, no chaos. The skills are markdown files you can read, edit, or replace without touching any Python.

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

The orchestrator owns phase progression (`research` → `baseline` → `experimenting` → `focused_tuning` → `ablation` → `converged`), experiment scheduling, and research refresh triggers. Every action an agent proposes is checked against `NON_NEGOTIABLE_RULES.md` by the PolicyEngine before it executes. Results are written to the experiment registry and project journal. The dashboard reads those files and streams updates to a browser in real time.

---

## Key Features

### 1. Interactive Onboarding TUI

Running `drl-autoresearch init` launches a terminal questionnaire that collects everything the system needs to operate autonomously:

- Project name, environment description, and research objective
- Hardware configuration — GPUs, VRAM, CPU count, RAM — auto-detected with manual override
- Python environment path and package manager preference
- Permission policy (see [Permission Modes](#permission-modes))
- Hard rules acknowledgment — the researcher confirms or customizes `NON_NEGOTIABLE_RULES.md`

Every question that is skipped is logged with the inferred default value and a confidence score. Nothing is silently assumed.

### 2. Single Orchestrator

One decision authority manages the entire lifecycle. The orchestrator reads project state from `.drl_autoresearch/state.json`, selects the next experiment based on the current research phase and the experiment registry, schedules worker processes, and decides when to trigger a research refresh (a mid-training literature and implementation review). There are no competing agents and no distributed coordination — complexity that DRL research does not need.

Phase transitions are explicit and logged. The orchestrator will not move to `focused_tuning` until a viable baseline is established, and will not declare `converged` until ablations have been run.

### 3. Hard Rule Enforcement via PolicyEngine

Rules defined in `NON_NEGOTIABLE_RULES.md` are enforced before every action. Agents check rules with:

```bash
drl-autoresearch check --action <type> [--details '<json>']
```

The PolicyEngine returns `ALLOWED`, `BLOCKED`, or `CONFIRM_REQUIRED` depending on the permission mode and the rule set. Violations are never silently ignored — they are printed to stderr with the specific rule that was violated and appended to `logs/incidents.md`.

Permission modes control strictness. In `locked` mode, nothing changes without explicit human approval. In `open` mode, the loop runs fully autonomously. The default is `prompted`.

### 4. DRL Skill Pack

Eight playbooks live in `skills/` and serve as the agent's operating manual for common DRL failure modes. They are plain markdown — readable, editable, and replaceable.

| Skill | Purpose |
|-------|---------|
| `investigate.md` | Regression diagnosis — systematic checklist for performance drops |
| `reward_shaping.md` | Reward modification protocol — safe incremental changes, anti-hacking checks |
| `exploration.md` | Exploration failure diagnosis — entropy collapse, stuck policies |
| `env_diagnostics.md` | Environment debugging — wrapper bugs, reward signal sanity checks |
| `ablation.md` | Scientific ablation protocol — controlled component isolation |
| `checkpoint_selection.md` | Checkpoint management without evaluation data leakage |
| `compute_budgeting.md` | Compute trade-off reasoning — when to cut a run, when to extend |
| `mid_training_research.md` | Mid-training research refresh — when and how to incorporate new findings |

### 5. Structured Experiment Registry

Every run is appended to `logs/experiment_registry.tsv` — a tab-separated log with 26 fields per row:

`run_id`, `parent_run_id`, `agent`, `hypothesis`, `algorithm`, `hyperparams`, `train_reward_mean`, `train_reward_std`, `eval_reward_mean`, `eval_reward_std`, `best_checkpoint`, `steps_trained`, `wall_time_hours`, `gpu_hours`, `peak_vram_gb`, `cpu_util_pct`, `keep`, `discard_reason`, `notes`, `git_commit`, `start_time`, `end_time`, `phase`, `skill_used`, `rule_checks_passed`, `incidents`

Appends are file-locked and atomic. Multiple workers can write concurrently without corruption. The registry is the ground truth for the dashboard and for the orchestrator's scheduling decisions.

### 6. Project Journal, Incident Log, and Handoff Memory

`logs/project_journal.md` is a human-readable running narrative of every significant decision, observation, and phase transition. It is written by agents and readable by humans at any time.

`logs/incidents.md` tracks structured incident records: reward hacking detected, out-of-memory crash, training divergence, broken assumption, and rule violation. Each incident has a timestamp, severity, run ID, description, and resolution status.

`logs/handoffs.md` enables multi-agent continuity. Before any agent session ends — whether that is a Claude Code session, a Codex run, or an overnight job — it writes a structured handoff: current phase, last experiment, open hypotheses, next planned action, and any caveats the next agent must know. The next agent reads the latest handoff entry before doing anything else.

### 7. Live Localhost Dashboard

The dashboard runs on port 8765 and requires no external web framework — it uses Python's standard library HTTP server with Server-Sent Events for live updates. Chart.js provides dark-theme charts in the browser.

The dashboard shows:

- Training reward curves (all active and recent runs)
- Evaluation curves with confidence intervals
- GPU and CPU resource utilization over time
- Experiment timeline (Gantt-style, parallel runs visible)
- Top models table (sorted by eval reward, with hyperparams)
- Open incidents with severity and resolution status
- Next planned experiment (from orchestrator state)
- Morning summary — generated after overnight runs, showing what happened, what to review, and what to do next

Start it with:

```bash
drl-autoresearch dashboard &
# open http://localhost:8765
```

### 8. Optional Parallel Background Experiments

The orchestrator can manage multiple isolated experiment workers running concurrently. Each worker receives:

- A dedicated GPU assignment (`CUDA_VISIBLE_DEVICES`)
- An explicit VRAM budget enforced before the run starts
- Health monitoring — the orchestrator detects hangs, OOM kills, and divergence
- Kill, pause, and resume support via `drl-autoresearch status` and signals

The orchestrator refuses to schedule a new worker if doing so would oversubscribe detected GPU memory. Parallelism is opt-in: `drl-autoresearch run --parallel N`.

### 9. Overnight Nonstop Training

The system is designed to run for 8 or more hours without human intervention:

- Checkpointing at configurable intervals with crash-safe atomic writes
- Automatic resume from the latest valid checkpoint on restart
- Durable structured logs (append-only, never overwritten)
- Periodic evaluation runs at configurable step intervals
- Incident detection with automatic severity classification
- Morning summary generated at a configurable wall-clock time, consolidating all overnight activity into a single readable report

If an experiment crashes, the orchestrator logs the incident, attempts a resume if the failure is recoverable, and moves to the next experiment if it is not.

### 10. Claude Code Slash Commands

Five built-in commands are available after cloning the repo (they live in `.claude/commands/`):

| Command | Description |
|---------|-------------|
| `/drl-init` | Onboard and scaffold a new DRL project interactively |
| `/drl-run` | Start the autonomous overnight experiment loop |
| `/drl-diagnose` | Diagnose a failure or confusing result using the skill pack |
| `/drl-research` | Trigger a mid-training research refresh |
| `/drl-plan` | Review and update the current experiment plan |

---

## Installation

### Requirements

- Python 3.10 or later
- Linux (primary support; macOS has partial support)
- NVIDIA GPU recommended — CPU fallback is available but slow
- `uv` or `pip`

### Install from Source

```bash
git clone https://github.com/your-org/autoresearch-RL
cd autoresearch-RL
pip install -e .
# or with uv:
uv pip install -e .
# verify:
drl-autoresearch --version
```

### Optional Extras

```bash
pip install -e ".[dashboard]"   # adds aiohttp for a faster dashboard server
pip install -e ".[tui]"         # adds questionary for an enhanced onboarding TUI
pip install -e ".[full]"        # both dashboard and tui extras
```

The base install has zero non-stdlib runtime dependencies. The extras are genuinely optional.

---

## Quickstart

```bash
# 1. Go to your DRL project
cd ~/my-rl-project

# 2. Initialize — runs onboarding, scaffolds all files
drl-autoresearch init

# 3. Verify environment — checks GPU, Python env, and config
drl-autoresearch doctor

# 4. Start the live dashboard (optional — opens http://localhost:8765)
drl-autoresearch dashboard &

# 5. Start the autonomous research loop
drl-autoresearch run
```

After `init`, open `CLAUDE.md` to review the generated agent operating guide, and `NON_NEGOTIABLE_RULES.md` to review or edit your hard rules before the first run.

---

## Claude Code Integration

After cloning the repo, the `.claude/commands/` directory contains the five DRL slash commands. They are available immediately in any Claude Code session opened from inside `autoresearch-RL`, and can be copied into any target project's `.claude/commands/` directory.

```
# In Claude Code, after installing the plugin:

/drl-init       # Set up a new DRL project interactively
/drl-run        # Start the autonomous overnight experiment loop
/drl-diagnose   # Diagnose a failure or confusing result
/drl-plan       # Review and update the experiment plan
/drl-research   # Trigger a mid-training research refresh
```

**How it fits together:**

- Running `/drl-init` in your target project calls `drl-autoresearch init`, generates `CLAUDE.md` with project-specific agent operating instructions, and scaffolds all logs and skills directories.
- The PolicyEngine checks are integrated into the run loop. When the agent proposes an action, `drl-autoresearch check` is called automatically. The agent cannot proceed past a `BLOCKED` result.
- The dashboard can be started in the background (`drl-autoresearch dashboard &`) while Claude Code runs experiments. The dashboard reads the same log files the agent writes to — no separate data pipeline.
- `CLAUDE.md` is regenerated each time `drl-autoresearch init` is re-run with updated project state. You can also edit it manually; the system will not overwrite manual edits unless you pass `--force`.

---

## Codex / OpenAI Integration

DRL AutoResearch generates an `AGENT.md` in your project root that serves as the primary guidance file for Codex and other OpenAI-based coding assistants. All CLI commands work independently of which AI tool is running experiments — the shared core (orchestrator, policy engine, logs) is identical for Claude Code and Codex sessions.

```bash
# AGENT.md is generated in your project root and serves as Codex instructions.
# Point Codex at: AGENT.md, CLAUDE.md, IMPLEMENTATION_PLAN.md

# Run the orchestrator to decide what experiment to run next:
drl-autoresearch plan --project-dir .

# Check if a planned action is allowed before executing it:
drl-autoresearch check --action edit_reward --details '{"file": "reward.py"}'

# After an experiment completes, log the result:
# Append a row to logs/experiment_registry.tsv (format described in AGENT.md)

# Trigger a mid-training research refresh:
drl-autoresearch research --project-dir .
```

`AGENT.md` describes the full experiment registry format, the incident reporting protocol, the handoff writing standard, and the expected behavior at each research phase. A Codex agent reading `AGENT.md` has enough context to participate in a multi-session research campaign without losing continuity.

---

## Project Scaffold Structure

After running `drl-autoresearch init` in your DRL project, the following files and directories are created:

```
your-drl-project/
├── .drl_autoresearch/
│   ├── state.json          ← live project state (phase, active runs, counters)
│   ├── policy.yaml         ← permission policy and rule overrides
│   ├── hardware.yaml       ← auto-detected hardware configuration
│   ├── python_env.yaml     ← Python environment and package manager config
│   └── permissions.yaml    ← current permission mode
│
├── CLAUDE.md               ← agent operating guide (auto-generated, project-specific)
├── AGENT.md                ← agent behavior standard (Codex / general AI)
├── ORCHESTRATOR.md         ← orchestrator decision logic and phase definitions
├── USER_SPEC.md            ← clarified project specification from onboarding
├── NON_NEGOTIABLE_RULES.md ← hard rules enforced by PolicyEngine
├── IMPLEMENTATION_PLAN.md  ← current research plan with ranked hypotheses
│
├── skills/
│   ├── investigate.md
│   ├── reward_shaping.md
│   ├── exploration.md
│   ├── env_diagnostics.md
│   ├── ablation.md
│   ├── checkpoint_selection.md
│   ├── compute_budgeting.md
│   └── mid_training_research.md
│
└── logs/
    ├── experiment_registry.tsv
    ├── project_journal.md
    ├── incidents.md
    ├── handoffs.md
    └── runs/
        └── <run_id>/
            ├── config.yaml
            ├── stdout.log
            ├── metrics.jsonl
            └── checkpoints/
```

The scaffold does not touch your existing source files. Everything is additive.

---

## CLI Reference

```
drl-autoresearch install
    Install plugin dependencies into the current environment.

drl-autoresearch init [--project-dir DIR] [--skip-onboarding] [--auto] [--force]
    Run onboarding TUI and scaffold project files.
    --skip-onboarding   Use all defaults, skip interactive questions.
    --auto              Fully non-interactive; suitable for CI.
    --force             Regenerate CLAUDE.md and AGENT.md even if manually edited.

drl-autoresearch doctor [--project-dir DIR]
    Check GPU availability, Python environment, config validity, and log writability.
    Prints a pass/warn/fail summary for each check.

drl-autoresearch dashboard [--project-dir DIR] [--port PORT]
    Start the live dashboard HTTP server.
    Default port: 8765. Reads logs from --project-dir (default: current directory).

drl-autoresearch run [--project-dir DIR] [--parallel N] [--dry-run]
    Start the autonomous experiment loop.
    --parallel N    Run up to N experiments concurrently (default: 1).
    --dry-run       Plan and print the next experiment without executing it.

drl-autoresearch status [--project-dir DIR]
    Show current phase, active workers, recent results, and open incidents.

drl-autoresearch plan [--project-dir DIR] [--refresh]
    Print the current experiment plan from IMPLEMENTATION_PLAN.md.
    --refresh   Trigger a research refresh and update the plan before printing.

drl-autoresearch research [--project-dir DIR]
    Trigger a mid-training research refresh using the mid_training_research.md skill.

drl-autoresearch check --action ACTION [--details JSON] [--project-dir DIR]
    Check whether an action is permitted under the current policy.
    Returns: ALLOWED | CONFIRM_REQUIRED | BLOCKED
    Exits 0 for ALLOWED, 1 for BLOCKED, 2 for CONFIRM_REQUIRED.
    --action    Action type string (e.g. edit_reward, install_package, delete_checkpoint).
    --details   Optional JSON string with action-specific context.
```

---

## Permission Modes

Set in `.drl_autoresearch/permissions.yaml` during onboarding or updated manually.

| Mode | Description | Use case |
|------|-------------|----------|
| `locked` | Nothing changes without explicit human approval for each action | Sensitive production codebases, shared research environments |
| `prompted` | Ask before each install or file change; allow reads freely | **Default** — research projects where you want visibility |
| `bootstrap-only` | Allow installs during `init` only; lock down afterward | After initial setup on a shared machine |
| `open` | Allow all actions; fully autonomous operation | Dedicated overnight research sessions |
| `project-only` | Installs and changes restricted to the project's own virtualenv | Shared machines where system packages must not be touched |

Change the mode at any time:

```bash
# Edit .drl_autoresearch/permissions.yaml directly, or re-run init:
drl-autoresearch init --project-dir . --skip-onboarding
```

---

## Design Principles

These principles were inherited from the original AutoResearch workflow and are maintained without compromise:

**Lightweight.** A single installable package. No services to run, no databases to configure, no cloud accounts required. `pip install -e .` and you are done.

**Repo-native.** All state — project config, experiment registry, journal, incidents, handoffs, checkpoints index — lives inside the target project as plain files. The project is self-contained. You can `git add logs/` to version-control your research history, or `.gitignore` it. Your choice.

**Markdown-guided.** `CLAUDE.md`, `AGENT.md`, and the `skills/*.md` playbooks are the interface between the system and the agent. They are human-readable and human-editable. Changing agent behavior means editing a markdown file, not touching Python internals.

**Autonomous but controlled.** Hard rules are enforced, not suggested. The PolicyEngine does not print a warning and continue — it blocks. The system is designed to be left running overnight with confidence that it will not delete checkpoints, overwrite baselines, install arbitrary system packages, or run experiments that violate your stated constraints.

**Fast iteration.** The experiment loop is the core product. Everything else — the dashboard, the TUI, the CLI, the skill pack — exists to make the loop faster, safer, and easier to understand after the fact. There is no overhead that does not pay for itself within the first overnight run.

---

## FAQ / Gotchas

**Q: Does this replace the original `program.md` / `train.py` workflow?**

No. It extends it. The original AutoResearch loop — edit `program.md`, run `train.py`, read results, repeat — still works exactly as before. DRL AutoResearch adds a structured scaffold on top: orchestration, policy enforcement, DRL-specific skills, and durable logging. You can adopt it incrementally.

**Q: Can I use this with non-DRL ML projects?**

Yes. The core loop — orchestrator, policy engine, experiment registry, journal — is general. The DRL skills are just markdown files in `skills/`. You can add project-specific skills, remove the DRL-specific ones, or replace them entirely. The system does not hardcode any DRL assumptions outside of the skill pack and the onboarding questionnaire.

**Q: What happens if an agent tries to violate a hard rule?**

`drl-autoresearch check` returns `BLOCKED`, exits with code 1, and prints a clear message identifying the specific rule that was violated. The violation is also appended to `logs/incidents.md` with a timestamp and the action that was attempted. The agent must stop or choose a different experiment. In `prompted` mode, the human can override a block interactively. In `locked` mode, overrides are not possible.

**Q: Is the dashboard required?**

No. It is entirely optional. All state is in files — `logs/project_journal.md` is a readable narrative of everything that happened, and `logs/experiment_registry.tsv` is a spreadsheet-importable record of every run. The dashboard is a convenience layer, not a dependency.

**Q: Can multiple AI agents work on the same project?**

Yes — that is the explicit purpose of `logs/handoffs.md`. Before any agent session ends, the agent writes a structured handoff entry: current phase, last experiment result, open hypotheses, next planned action, and any caveats the next agent must be aware of. The next agent reads the latest entry before doing anything. This enables a Claude Code session to hand off to an overnight Codex job and back again without losing context.

**Q: How do I add a custom hard rule?**

Edit `NON_NEGOTIABLE_RULES.md` in your project. Rules are matched by the PolicyEngine using the action type strings documented in `ORCHESTRATOR.md`. You can add rules in plain English — the policy file supports both structured YAML rules and free-text rules that the agent is expected to interpret. Structured rules are checked programmatically; free-text rules are included in the context given to the agent before each action.

**Q: What if a run crashes mid-training?**

The incident is logged to `logs/incidents.md` with the run ID, failure type, and last checkpoint path. On the next `drl-autoresearch run` invocation, the orchestrator reads the incident log and decides whether to resume (recoverable crash) or skip to the next experiment (unrecoverable). You can inspect and manually resolve open incidents with `drl-autoresearch status`.

---

## Contributing

The skills pack is the easiest place to contribute. If you have a DRL failure mode that is not covered by the existing eight playbooks, add a markdown file to `skills/` following the format in any existing skill. Pull requests adding new skills are welcome without requiring changes to any Python code.

For bugs in the orchestrator, policy engine, or CLI, open an issue with the output of `drl-autoresearch doctor` and the relevant section of `logs/incidents.md`.

---

## License

MIT. See `LICENSE` for details.
