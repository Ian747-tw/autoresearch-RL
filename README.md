# DRL AutoResearch

**Autonomous deep reinforcement learning research — from spec to results overnight**

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Platform: Linux / GPU](https://img.shields.io/badge/platform-Linux%20%7C%20GPU-lightgrey)

---

![DRL AutoResearch Dashboard](edited-photo(1).png)

*Live dashboard: training curves, eval curves, GPU/CPU usage, experiment timeline, and top-models leaderboard — all from your own project, no cloning required.*

---

## Overview

DRL AutoResearch is a globally-installed CLI tool (`drl-autoresearch`) that plugs into **your existing DRL training project**. Install it once, then run `init` inside any project to activate the autonomous research loop.

Where the original Karpathy-style AutoResearch workflow gives you a general-purpose research automation scaffold, DRL AutoResearch adds the infrastructure that DRL specifically demands: reward shaping protocols, exploration failure diagnosis, checkpoint selection without data leakage, environment debugging playbooks, and a hard rule enforcement layer that prevents the autonomous loop from making expensive or irreversible mistakes. It is not a rewrite — it is a strict extension. The original loop still works unchanged.

The core philosophy is preserved: local-first, repo-native, markdown-guided, minimal ceremony, fast iteration. No databases, no daemon processes, no cloud APIs. All state lives inside your project as plain files. The orchestrator is a single decision authority — no competing agents, no chaos. The skills are markdown files you can read, edit, or replace without touching any Python.

---

## Architecture

```
Your DRL Project
       │
       ├── .drl_autoresearch/state.json    ← phase, best run, counters
       ├── NON_NEGOTIABLE_RULES.md         ← hard rules (never overridden)
       ├── logs/experiment_registry.tsv   ← full 27-column run history
       ├── skills/*.md                     ← agent playbooks (8 bundled)
       ├── .claude/commands/drl-*.md       ← Claude Code plugin (optional)
       └── AGENT.md                        ← Codex plugin (optional)
               │
               ▼
drl-autoresearch (globally installed — one install, any project)
               │
    ┌──────────┼──────────────────┬───────────────────┐
    │          │                  │                   │
Orchestrator  PolicyEngine    RunManager          Dashboard
(phase mgmt,  (hard rules,    (per-run dirs,      (stdlib HTTP+SSE
 scheduling,   audit log,      metrics.jsonl,      port 8765,
 refresh       check command)  checkpoints,        Chart.js,
 triggers)                     keep/discard)       live updates)
```

Research phases: `research` → `baseline` → `experimenting` → `focused_tuning` → `ablation` → `converged`

Each phase transition is logged. The orchestrator will not advance to `focused_tuning` until a viable baseline exists, and will not declare `converged` until ablations have run.

---

## Key Features

### 1. Plugin-Based — Install Once, Use Anywhere

`drl-autoresearch init` in any DRL project scaffolds all required files and installs your chosen AI agent plugin(s) from the bundled package — no cloning required per-project.

```bash
pip install drl-autoresearch          # once, globally
cd ~/any-rl-project
drl-autoresearch init                 # scaffold + choose plugins interactively
drl-autoresearch init --auto          # scaffold + install both plugins silently
drl-autoresearch init --plugin cc     # Claude Code only
drl-autoresearch init --plugin codex  # Codex only
drl-autoresearch init --skill-pack custom
drl-autoresearch init --project-mode build
```

The Claude Code plugin copies five `/drl-*` slash commands into `.claude/commands/`. The Codex plugin writes `AGENT.md` to your project root. Both files are read automatically by their respective agents.
During init, after onboarding captures the user's spec and hard rules, users can either keep the provided DRL pack or remove it from that project and install `.drl_autoresearch/backend/skill_generator.md` to drive a compact domain-specific replacement pack.
Init also supports two project modes: `build` (from-scratch design/build workflow before normal loops) and `improve` (existing working model, optimize directly).

### 2. Single Orchestrator

One decision authority manages the entire lifecycle. It reads project state from `.drl_autoresearch/state.json`, selects the next experiment based on the current research phase and the experiment registry, schedules worker processes, and decides when to trigger a research refresh. There are no competing agents and no distributed coordination.

### 3. Hard Rule Enforcement via PolicyEngine

Rules defined in `NON_NEGOTIABLE_RULES.md` are enforced before every agent action. Agents gate changes with:

```bash
drl-autoresearch check --action <type> [--details '<json>']
```

The PolicyEngine returns `ALLOWED` (exit 0) or `BLOCKED` (exit 1) depending on the permission mode and the rule set. Violations are never silently ignored — they are logged to `.drl_autoresearch/policy_audit.log` with the specific rule violated.

Permission modes control strictness: `locked` (default) blocks anything sensitive; `open` lets the loop run fully autonomously overnight. Set in `.drl_autoresearch/permissions.yaml`.

### 4. DRL Skill Pack — 8 Playbooks

Eight playbooks are installed into `skills/` during `init`. They serve as the agent's operating manual for common DRL failure modes — plain markdown, readable and replaceable.

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

Every run is appended to `logs/experiment_registry.tsv` — a tab-separated file with 27 fields per row:

`run_id`, `parent_run_id`, `timestamp`, `agent`, `branch`, `commit`, `environment`, `algorithm`, `config_summary`, `change_summary`, `hypothesis`, `rules_checked`, `train_reward_mean`, `train_reward_std`, `eval_reward_mean`, `eval_reward_std`, `custom_metric_name`, `custom_metric_value`, `success_rate`, `constraint_violations`, `seed_count`, `wall_clock_seconds`, `gpu_memory_gb`, `ram_gb`, `status`, `keep_decision`, `notes`

Appends are file-locked and atomic. Multiple workers can write concurrently without corruption. The registry is the ground truth for the dashboard and for all orchestrator scheduling decisions.

### 6. Project Journal, Incident Log, and Handoff Memory

`logs/project_journal.md` is a human-readable running narrative of every significant decision, observation, and phase transition — written by agents, readable by humans at any time.

`logs/incidents.md` tracks structured incident records: reward hacking detected, OOM crash, training divergence, broken assumption, rule violation. Each has a timestamp, severity, run ID, description, and resolution status.

`logs/handoffs.md` enables multi-agent continuity. Before any agent session ends — Claude Code, Codex, or an overnight job — it writes a structured handoff: current phase, last experiment, open hypotheses, next planned action, and caveats the next agent must know. The next agent reads the latest entry before doing anything else.

### 7. Live Localhost Dashboard

The dashboard runs on port 8765 using Python's standard library HTTP server with Server-Sent Events for live updates. No external web framework required. Chart.js provides dark-theme charts.

The dashboard shows:

- Training reward curves (all active and recent runs, per-experiment)
- Evaluation curves with live updates every 5 seconds
- GPU and CPU resource utilization over time
- Experiment timeline with keep/discard decisions highlighted
- Top models leaderboard (sorted by eval reward, with hypotheses)
- Open incidents with severity and resolution status
- Next planned experiment from orchestrator state
- Morning summary — generated after overnight runs, showing what happened, what to review, and what to do next
- Workflow snapshot: current mode, bootstrap state, refresh cooldown, last refresh reason

```bash
drl-autoresearch dashboard &
# open http://localhost:8765
```

### 8. Optional Parallel Background Experiments

The orchestrator manages multiple isolated experiment workers running concurrently. Each worker gets a dedicated GPU assignment (`CUDA_VISIBLE_DEVICES`), an explicit VRAM budget enforced before the run starts, and health monitoring for hangs and OOM kills. Kill, pause, and resume are supported.

The orchestrator refuses to schedule a new worker if doing so would oversubscribe detected GPU memory. Parallelism is opt-in:

```bash
drl-autoresearch run --parallel N
```

### 9. Overnight Nonstop Operation

The system is designed to run for 8+ hours unattended:

- Checkpointing at configurable intervals with crash-safe atomic writes
- Automatic incident logging with severity classification
- Durable append-only logs (never overwritten)
- Morning summary generated after overnight runs
- If an experiment crashes, the orchestrator logs the incident and moves to the next experiment

### 10. Claude Code Slash Commands

Five commands are installed into `.claude/commands/` by the Claude Code plugin:

| Command | Description |
|---------|-------------|
| `/drl-init` | Scaffold and initialize a new DRL project interactively |
| `/drl-run` | Start the autonomous overnight experiment loop |
| `/drl-diagnose` | Diagnose a failure or confusing result using the skill pack |
| `/drl-research` | Trigger a mid-training research refresh |
| `/drl-plan` | Review and update the current experiment plan |

---

## Quickstart — 3 commands

```bash
# 1. Install globally (once)
pip install drl-autoresearch

# 2. Go to YOUR training project and init
cd ~/my-rl-project
drl-autoresearch init

# 3. Verify the environment, then run
drl-autoresearch doctor
drl-autoresearch run
```

During `init` you will be asked which AI agent plugin to install:

- **Claude Code** — installs `/drl-*` slash commands into `.claude/commands/`
- **Codex** — installs an `AGENT.md` operating guide in your project root
- **Both** (recommended)
- **Project mode** — `build` (from-scratch design/build bootstrap) or `improve` (optimize an existing model)
- **Skill-pack mode** — `drl` (provided compact pack) or `custom` (generate compact domain pack via backend prompt)

That's it. No cloning. No config editing. Your project is now wired into the autonomous loop.

---

## Installation

### Requirements

- Python 3.10 or later
- Linux (primary); macOS has partial support
- NVIDIA GPU recommended — CPU fallback available but slow

### Install globally with pip

```bash
pip install drl-autoresearch
drl-autoresearch --version
```

### Install globally with uv

```bash
uv tool install drl-autoresearch
drl-autoresearch --version
```

### Install from source (development / latest)

```bash
git clone https://github.com/Ian747-tw/autoresearch-RL
cd autoresearch-RL
pip install -e .
```

### Without a global install (uv run)

Every command also works without installing:

```bash
uv run python -m drl_autoresearch <command>
```

### Optional extras

```bash
pip install "drl-autoresearch[tui]"   # interactive onboarding TUI (questionary)
pip install "drl-autoresearch[full]"  # tui + aiohttp extras
```

The base install has zero non-stdlib runtime dependencies.

---

## Claude Code Setup

### 1. Install the package

```bash
pip install drl-autoresearch
```

### 2. Open YOUR project in Claude Code and init

```bash
cd ~/my-rl-project
claude  # open Claude Code in your project
```

Inside the Claude Code session:

```
/drl-init
```

This runs `drl-autoresearch init`, scaffolds all config files, and installs the
Claude Code plugin — five `/drl-*` slash commands are copied into your project's
`.claude/commands/` directory. They are available immediately.

### 3. Use the slash commands

```
/drl-init       Initialize or re-initialize this project
/drl-run        Start the autonomous overnight experiment loop
/drl-plan       Review and refresh the ranked experiment plan
/drl-diagnose   Diagnose a training failure using playbooks
/drl-research   Trigger a mid-training literature/implementation refresh
```

### Non-interactive init (CI / scripted)

```bash
# Install both plugins, skip all prompts
drl-autoresearch init --auto

# Install Claude Code plugin only
drl-autoresearch init --plugin cc --skip-onboarding

# Install Codex plugin only
drl-autoresearch init --plugin codex --skip-onboarding

# Skip plugin install entirely
drl-autoresearch init --plugin none --skip-onboarding
```

---

## Codex Setup

### 1. Install the package and Codex CLI

```bash
pip install drl-autoresearch
npm install -g @openai/codex
export OPENAI_API_KEY=sk-...
```

### 2. Init your project with the Codex plugin

```bash
cd ~/my-rl-project
drl-autoresearch init --plugin codex
```

This creates `AGENT.md` in your project root. Codex reads it automatically when
you open an interactive session or run `codex exec` from that directory.

### 3. Interactive session

```bash
cd ~/my-rl-project
codex
# Codex reads AGENT.md → knows the CLI commands, experiment protocol,
# registry format, and policy check requirements automatically
```

### 4. Non-interactive / scripted

```bash
codex exec --dangerously-bypass-approvals-and-sandbox \
  -c 'shell_environment_policy.inherit=all' \
  "Run drl-autoresearch doctor, then drl-autoresearch run --dry-run, then start the experiment loop. Check AGENT.md for the full protocol."
```

> **Note on bwrap:** Codex's default sandbox uses `bwrap`, which requires
> unprivileged user namespaces. On some Linux systems (containers, restricted VMs)
> this is unavailable. Use `--dangerously-bypass-approvals-and-sandbox` to skip it.

---

## Project Scaffold Structure

After `drl-autoresearch init`, these files are created **inside your own project**:

```
your-rl-project/
├── .drl_autoresearch/
│   ├── state.json          ← live state (phase, best run, counters)
│   ├── policy.yaml         ← permission policy and action overrides
│   ├── hardware.yaml       ← auto-detected hardware config
│   ├── python_env.yaml     ← Python environment config
│   └── permissions.yaml    ← current permission mode
│
├── NON_NEGOTIABLE_RULES.md ← hard rules enforced by PolicyEngine
│
├── .claude/
│   └── commands/           ← Claude Code plugin (if --plugin cc/both)
│       ├── drl-init.md
│       ├── drl-run.md
│       ├── drl-plan.md
│       ├── drl-diagnose.md
│       └── drl-research.md
│
├── AGENT.md                ← Codex plugin (if --plugin codex/both)
│
├── skills/                 ← DRL playbooks (installed automatically)
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
    ├── experiment_registry.tsv   ← 27-column run history
    └── runs/
        └── <run_id>/
            ├── run_meta.json
            ├── run_result.json
            ├── metrics.jsonl
            └── checkpoints/
```

The scaffold **does not touch your existing source files**. Everything is additive.

---

## Dashboard

Start the dashboard from your project directory:

```bash
drl-autoresearch dashboard &
# open http://localhost:8765
```

The dashboard uses Python's stdlib HTTP server with SSE — no extra framework needed.
It shows training curves, eval curves, resource usage, timeline, top-models
leaderboard, and open incidents — all read from the same files the experiment loop writes.

To populate the training curve charts from your own training script, write:

```python
# logs/artifacts/<run_id>/metrics.json
import json
from pathlib import Path

artifact_dir = Path("logs/artifacts") / run_id
artifact_dir.mkdir(parents=True, exist_ok=True)
(artifact_dir / "metrics.json").write_text(json.dumps({
    "steps":        [100, 200, 300, ...],
    "rewards":      [12.3, 18.7, 24.1, ...],
    "losses":       [0.42, 0.38, 0.31, ...],
    "eval_steps":   [500, 1000, ...],
    "eval_rewards": [22.5, 31.0, ...],
}))
```

Or use the included `RunManager` + `ExperimentRegistry` classes directly — see
`test_train.py` for a complete working example.

---

## CLI Reference

```
drl-autoresearch install
    Print install confirmation and quick-start tips.

drl-autoresearch init [--project-dir DIR] [--skip-onboarding] [--auto] [--plugin PLUGIN] [--skill-pack PACK] [--project-mode MODE]
    Scaffold project files and optionally install AI agent plugins.
    --skip-onboarding   Skip the interactive questionnaire, use defaults.
    --auto              Fully non-interactive; installs both plugins.
    --plugin PLUGIN     cc | codex | both | none
                        Which plugin(s) to install. If omitted, prompts interactively.
    --skill-pack PACK   drl | custom
                        drl keeps the bundled DRL skills.
                        custom removes the bundled DRL skills from that project
                        and installs a backend prompt for generating a compact
                        domain-specific replacement pack.
    --project-mode MODE build | improve
                        build runs compact deep-research + compact step plans
                        before entering normal training loops.
                        improve assumes a working model already exists.

drl-autoresearch doctor [--project-dir DIR]
    Run 14 health checks: Python, PyTorch, GPU/CUDA, config files, registry, rules.
    Exits 0 only if all pass.

drl-autoresearch dashboard [--project-dir DIR] [--port PORT]
    Start the live dashboard HTTP server. Default port: 8765.

drl-autoresearch run [--project-dir DIR] [--parallel N] [--dry-run]
    Start the autonomous experiment loop (orchestrator-controlled).
    --parallel N    Run up to N experiments concurrently (default: 1).
    --dry-run       Print the next orchestrator-selected experiment without executing it.

drl-autoresearch status [--project-dir DIR]
    Show current phase, counters, best run, recent experiments, active workers.

drl-autoresearch plan [--project-dir DIR] [--refresh]
    Display the ranked experiment plan.
    --refresh   Regenerate the plan from the latest state before displaying.

drl-autoresearch research [--project-dir DIR]
    Trigger a mid-training research refresh.

drl-autoresearch check --action ACTION [--details JSON] [--project-dir DIR]
    Gate an agent action against the current policy.
    Exits 0 (allowed) or 1 (blocked).
    Action types: edit_reward | edit_eval | edit_env | install_package |
                  update_package | global_install | exceed_compute |
                  gpu_memory_risk | eval_protocol_change | use_privileged_info | custom
```

---

## Permission Modes

Set in `.drl_autoresearch/permissions.yaml`. Change at any time by editing the file.

| Mode | Description | Use case |
|------|-------------|----------|
| `locked` | All sensitive actions blocked until explicit approval | **Default** |
| `prompted` | Ask before each install or file change | Visible supervised runs |
| `bootstrap-only` | Installs allowed during init only | Shared machines |
| `open` | Allow all actions; fully autonomous | Dedicated overnight runs |
| `project-only` | Changes restricted to project virtualenv | Shared machines |

---

## Architecture

```
User Project (your-rl-project/)
       │
       ├── .drl_autoresearch/state.json   ← phase + best run + counters
       ├── NON_NEGOTIABLE_RULES.md        ← hard rules
       ├── logs/experiment_registry.tsv  ← all run history
       │
       ▼
drl-autoresearch (globally installed)
       │
       ├── Orchestrator  ← phase management, experiment scheduling
       ├── PolicyEngine  ← hard rule enforcement (check command)
       ├── RunManager    ← per-run directories, metric logging, checkpoints
       ├── Dashboard     ← stdlib HTTP + SSE, reads logs directly
       │
       └── Plugins (bundled, installed into your project on init)
           ├── Claude Code  →  .claude/commands/drl-*.md
           └── Codex        →  AGENT.md
```

Research phases: `research` → `baseline` → `experimenting` → `focused_tuning` → `ablation` → `converged`

---

## Design Principles

**Install once, use everywhere.** `pip install drl-autoresearch` once, then `init` in any DRL project. No cloning. No per-project setup beyond one command.

**Repo-native.** All state lives inside your project as plain files. `git add logs/` to version-control your research history, or `.gitignore` it.

**Markdown-guided.** `NON_NEGOTIABLE_RULES.md`, `AGENT.md`, and `skills/*.md` are the interface between the system and the agent. Changing agent behavior means editing a markdown file, not Python.

**Autonomous but controlled.** The PolicyEngine blocks, it does not warn. Designed to run overnight without touching checkpoints, baselines, or system packages unless explicitly permitted.

---

## FAQ

**Q: Do I need to clone autoresearch-RL into every project?**

No. Install it once globally with `pip install drl-autoresearch`, then run
`drl-autoresearch init` inside your own project. The plugin files are bundled
inside the installed package and copied into your project automatically.

**Q: Can I use this with a non-DRL project?**

Yes. The orchestrator, policy engine, and registry are general. The DRL-specific
parts are the `skills/*.md` playbooks — replace or extend them for your domain.

**Q: What if a run crashes?**

The RunManager logs the incident. On the next `drl-autoresearch run`, the
orchestrator reads the registry and decides whether to retry. Inspect with
`drl-autoresearch status`.

**Q: Can Claude Code and Codex work on the same project?**

Yes. Both share the same `state.json`, `experiment_registry.tsv`, and
`NON_NEGOTIABLE_RULES.md`. One agent can initialize; another can continue.
Write handoffs to `logs/handoffs.md` for continuity across sessions.

**Q: Is the dashboard required?**

No. All state is in files. The dashboard is a convenience layer, not a
dependency.

---

## Contributing

The easiest contribution is a new skill playbook. Add a `.md` file to `skills/`
following the format of an existing playbook — no Python changes required.

For bugs in the CLI or core modules, open an issue with `drl-autoresearch doctor`
output and a description of the failure at:
https://github.com/Ian747-tw/autoresearch-RL/issues

---

## License

MIT. See `LICENSE` for details.
