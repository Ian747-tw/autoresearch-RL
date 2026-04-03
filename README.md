# DRL AutoResearch

Autonomous DRL research workflow manager with orchestrator-led agent loops, hard-rule guardrails, structured logs, and a local dashboard.

## Project Lineage

This repository is forked from Karpathy AutoResearch and keeps the original AutoResearch training-loop spirit as the core.  
The main research/training orchestration logic remains centered on the original local, iterative AutoResearch workflow; this fork adds DRL-focused operational features around init, environment handling, mode control, plugin scaffolding, and dashboard visibility.

## Core Principles

- Local-first, file-based workflow in your own project directory.
- Safety and policy enforcement before risky actions.
- Reproducible experiment tracking via logs and state files.
- Token-aware orchestration: compact planning/research artifacts and controlled refresh behavior.

## Feature Overview

### 1. Orchestrator-controlled agent loop

- `drl-autoresearch run` is orchestrator-first and continuous by default.
- `run` launches Codex or Claude Code as the execution backend for each cycle.
- Orchestrator selects the next experiment candidate.
- If orchestrator cannot produce one, local fallback generation is used.
- Phase and loop state are persisted in `.drl_autoresearch/state.json`.

### 2. Init mode selection (`build` vs `improve`)

During init, users choose one project mode:

- `build` mode:
  - For empty or incomplete training projects.
  - System performs deep research and creates compact plan steps in `implementation_plan/*.md` before build/training loop starts.
  - After bootstrap completes successfully, the plan folder is removed and a compact summary is logged to keep future context short.
- `improve` mode:
  - For existing working models.
  - System starts optimization-focused iteration directly.

Both modes support stuck-driven research/plan refresh without disabling existing capabilities.

### 3. Skill-pack strategy selection (`drl` vs `custom`)

During init, users choose one skill-pack strategy:

- `drl`:
  - Keep the provided compact DRL skill pack in project scope.
- `custom`:
  - Remove bundled DRL skills in the target project.
  - Install `.drl_autoresearch/backend/skill_generator.md` to generate domain-specific compact skills from user spec/rules.

Skill generation policy for custom pack:
- Keep generated skills compact (token-efficient).
- Keep skills general/basic technique-level (like DRL pack style).
- Avoid heavy, overly detailed implementation plans inside skills.

### 4. Onboarding questionnaire + policy capture

Init onboarding captures key inputs including:

- project/task basics (domain, objective, constraints)
- other information (quirks, known issues, extra context)
- hardware and runtime assumptions
- Python environment preferences
- permission policy
- hard rules and non-negotiables

These settings are saved into `.drl_autoresearch/` and reused by init/run/doctor flows.

### 5. Automatic compact spec indexing (token-saving)

After init, the system auto-generates:

- `.drl_autoresearch/spec_compact.md` (compact navigator)
- `.drl_autoresearch/spec_index.json` (machine-readable pointer index)

Design intent:
- structure is source-driven (derived from the actual spec/rules document structure), not a fixed hard-coded schema
- compact file is for fast navigation only
- original source files remain the source of truth for detailed clarification
- entries include line pointers (for example `USER_SPEC.md:300`) so agents can jump directly to relevant details without loading whole files

### 6. Environment doctor with auto-fix path

- `drl-autoresearch doctor` validates runtime health.
- `drl-autoresearch doctor --fix` attempts remediation automatically:
  - uses onboarding env preferences
  - creates/uses project venv when needed
  - installs missing dependencies
  - includes fallback handling for externally-managed Python environments (PEP 668)
- Init triggers best-effort remediation so setup issues are handled early.

### 7. Stuck refresh with cooldown (token saving)

- Stuck detection can trigger research/plan refresh when progress stalls.
- Refresh trigger is orchestrator-first, with local fallback heuristics.
- Cooldown state prevents repeated refresh loops that waste tokens.
- State includes mode/bootstrap/refresh metadata for observability.

### 8. Dashboard workflow visibility

Dashboard includes workflow snapshot fields such as:

- current mode
- loop running / idle state
- active run id
- current agent backend
- bootstrap started/completed state
- refresh cooldown remaining runs
- last refresh reason

This is in addition to normal training/registry status views.

## Installation

### Requirements

- Python 3.10+
- Linux recommended (macOS partial support)
- GPU optional (CPU works, typically slower)

### Install from source

```bash
git clone https://github.com/Ian747-tw/autoresearch-RL
cd autoresearch-RL
uv tool update-shell
uv tool install --force .
```

This installs `drl-autoresearch` once for your user account so it can be run
from any directory.

Verify CLI:

```bash
which drl-autoresearch
drl-autoresearch --help
```

If `which drl-autoresearch` returns nothing after install, open a new shell once
or run `source ~/.bashrc`.

Alternative if you prefer `pipx`:

```bash
git clone https://github.com/Ian747-tw/autoresearch-RL
cd autoresearch-RL
pipx install --force .
```

### Update existing local clone (no re-download)

```bash
cd ~/autoresearch-RL
git pull --rebase origin master
uv tool install --force .
```

If dependencies changed, run:

```bash
cd ~/autoresearch-RL
git pull --rebase origin master
uv tool install --force .
```

## End-to-End Workflow

### Step 0: Prepare a project folder

- Create or choose your training project directory.
- Put project spec/rules/constraints in place (or provide them during onboarding).

### Step 1: Initialize project

```bash
cd /path/to/your-project
drl-autoresearch init
```

Init flow:

1. Creates baseline structure (`.drl_autoresearch/`, `logs/`, `skills/`, config files).
2. Runs onboarding questionnaire (unless skipped).
3. Captures skill-pack choice (`drl` or `custom`).
4. Captures project mode (`build` or `improve`).
5. Installs selected plugin assets (`cc`, `codex`, `both`, `none`).
6. Auto-generates compact spec/index artifacts with source line pointers.
7. Runs best-effort environment remediation.
8. Persists startup state for orchestrator and dashboard.

Non-interactive examples:

```bash
drl-autoresearch init --auto
drl-autoresearch init --project-mode build --skill-pack custom --plugin codex --skip-onboarding
drl-autoresearch init --refresh
```

Refresh behavior:
- `drl-autoresearch init --refresh` removes DRL AutoResearch-managed config,
  runtime state, skill-pack files, and plugin scaffolding from the target
  project, then runs init again from scratch.
- It does not delete your source code or arbitrary project files.

### Step 2: Validate/fix environment

```bash
drl-autoresearch doctor
drl-autoresearch doctor --fix
```

Use `--fix` if dependencies/interpreter setup are not healthy.

### Step 3: Start orchestrated loop

```bash
drl-autoresearch run
```

Behavior summary:

- In `build` mode:
  - bootstrap research + compact implementation plan happens first
  - then the controller keeps launching agent-driven coding/training cycles
  - bootstrap plan folder is removed after build bootstrap completion
  - compact summary is written to project log/journal
- In `improve` mode:
  - immediate continuous agent-driven optimization loop

Runtime notes:

- `run` now stays alive until interrupted, convergence is reached, or a hard policy block stops it.
- The controller launches Codex or Claude Code in autonomous mode for each cycle.
- Use `--once` if you want a single cycle and exit.
- Use `--agent-backend codex` or `--agent-backend claude` to force one backend.
- Autonomous runs require onboarding permission policy `open`, `project-only`, or `bootstrap-only`.
- During autonomous cycles, risky actions must go through `drl-autoresearch check`.
- Registry/journal/incidents/handoffs are expected to be written through the project helper APIs, not raw file edits.

### Step 4: Monitor and intervene when needed

```bash
drl-autoresearch status
drl-autoresearch plan --refresh
drl-autoresearch research
```

If your agent session drops or you start a new session, use one-line resume:

```bash
drl-autoresearch resume --project-dir .
```

This performs compact sync (status + tail windows of registry/journal/handoffs/incidents)
and then continues the run loop automatically.

### Step 5: Open dashboard

```bash
drl-autoresearch dashboard --port 8765
```

Open `http://localhost:8765`.

## Plugin Artifacts Installed by Init

### Claude Code plugin (`--plugin cc` or `both`)

Installs to `.claude/commands/`:

- `drl-init.md`
- `drl-run.md`
- `drl-plan.md`
- `drl-diagnose.md`
- `drl-research.md`

### Codex plugin (`--plugin codex` or `both`)

- Installs `AGENT.md` in project root.
- Codex integration is file-based; no separate project-side `codex/` directory is required.

## Command Reference

### Init

```bash
drl-autoresearch init \
  [--project-dir DIR] \
  [--skip-onboarding] \
  [--auto] \
  [--plugin {cc,codex,both,none}] \
  [--skill-pack {drl,custom}] \
  [--project-mode {build,improve}]
```

### Doctor

```bash
drl-autoresearch doctor [--project-dir DIR] [--fix]
```

### Run

```bash
drl-autoresearch run [--project-dir DIR] [--parallel N] [--dry-run] [--once] [--agent-backend {auto,codex,claude}]
```

### Resume

```bash
drl-autoresearch resume [--project-dir DIR] [--parallel N] [--dry-run] [--no-run]
```

### Other core commands

```bash
drl-autoresearch status    [--project-dir DIR]
drl-autoresearch plan      [--project-dir DIR] [--refresh]
drl-autoresearch research  [--project-dir DIR]
drl-autoresearch resume    [--project-dir DIR] [--parallel N] [--dry-run] [--no-run]
drl-autoresearch check     --action ACTION [--details JSON] [--project-dir DIR]
drl-autoresearch dashboard [--project-dir DIR] [--port PORT]
```

## Token-Efficiency and Stability Notes

- Build bootstrap plans are intentionally compact and temporary.
- Stuck refresh is cooldown-controlled to prevent repetitive research churn.
- Custom skills are designed to be concise and technique-oriented, not verbose implementation documents.
- Existing backbone files remain the source of truth: `.drl_autoresearch/`, `logs/`, `skills/`, compact spec files, dashboard artifacts, and `NON_NEGOTIABLE_RULES.md`.

## Known Limits

- `doctor --fix` is best-effort and depends on package index/network reachability.
- Checks reflect the interpreter context used to run the CLI.
- Dashboard is local HTTP and intended for local/private usage.
- Continuous autonomous runs depend on a working `codex` or `claude` CLI being installed on the machine.
