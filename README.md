# DRL AutoResearch

Autonomous DRL research workflow manager with orchestrator-led training loops, hard-rule guardrails, structured logs, and a local dashboard.

## Project Lineage

This repository is forked from Karpathy AutoResearch and keeps the original AutoResearch training-loop spirit as the core.  
The main research/training orchestration logic remains centered on the original local, iterative AutoResearch workflow; this fork adds DRL-focused operational features around init, environment handling, mode control, plugin scaffolding, and dashboard visibility.

## Core Principles

- Local-first, file-based workflow in your own project directory.
- Safety and policy enforcement before risky actions.
- Reproducible experiment tracking via logs and state files.
- Token-aware orchestration: compact planning/research artifacts and controlled refresh behavior.

## Feature Overview

### 1. Orchestrator-controlled training loop

- `drl-autoresearch run` is orchestrator-first.
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
- hardware and runtime assumptions
- Python environment preferences
- permission policy
- hard rules and non-negotiables

These settings are saved into `.drl_autoresearch/` and reused by init/run/doctor flows.

### 5. Environment doctor with auto-fix path

- `drl-autoresearch doctor` validates runtime health.
- `drl-autoresearch doctor --fix` attempts remediation automatically:
  - uses onboarding env preferences
  - creates/uses project venv when needed
  - installs missing dependencies
  - includes fallback handling for externally-managed Python environments (PEP 668)
- Init triggers best-effort remediation so setup issues are handled early.

### 6. Stuck refresh with cooldown (token saving)

- Stuck detection can trigger research/plan refresh when progress stalls.
- Refresh trigger is orchestrator-first, with local fallback heuristics.
- Cooldown state prevents repeated refresh loops that waste tokens.
- State includes mode/bootstrap/refresh metadata for observability.

### 7. Dashboard workflow visibility

Dashboard includes workflow snapshot fields such as:

- current mode
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
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

Ubuntu/Debian note:
- If you see `externally-managed-environment` (PEP 668), install inside `.venv` as shown above.

Verify CLI:

```bash
drl-autoresearch --help
```

### Update existing local clone (no re-download)

```bash
cd ~/autoresearch-RL
git pull --rebase origin master
source .venv/bin/activate
pip install -e .
```

If dependencies changed, run:

```bash
cd ~/autoresearch-RL
git pull --rebase origin master
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
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
6. Runs best-effort environment remediation.
7. Persists startup state for orchestrator and dashboard.

Non-interactive examples:

```bash
drl-autoresearch init --auto
drl-autoresearch init --project-mode build --skill-pack custom --plugin codex --skip-onboarding
```

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
  - then normal experiment loop begins
  - bootstrap plan folder is removed after build bootstrap completion
  - compact summary is written to project log/journal
- In `improve` mode:
  - immediate iterative optimization loop

### Step 4: Monitor and intervene when needed

```bash
drl-autoresearch status
drl-autoresearch plan --refresh
drl-autoresearch research
```

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
drl-autoresearch run [--project-dir DIR] [--parallel N] [--dry-run]
```

### Other core commands

```bash
drl-autoresearch status    [--project-dir DIR]
drl-autoresearch plan      [--project-dir DIR] [--refresh]
drl-autoresearch research  [--project-dir DIR]
drl-autoresearch check     --action ACTION [--details JSON] [--project-dir DIR]
drl-autoresearch dashboard [--project-dir DIR] [--port PORT]
```

## Token-Efficiency and Stability Notes

- Build bootstrap plans are intentionally compact and temporary.
- Stuck refresh is cooldown-controlled to prevent repetitive research churn.
- Custom skills are designed to be concise and technique-oriented, not verbose implementation documents.
- Existing core training loop behavior is preserved; new features are additive and workflow-structuring.

## Known Limits

- `doctor --fix` is best-effort and depends on package index/network reachability.
- Checks reflect the interpreter context used to run the CLI.
- Dashboard is local HTTP and intended for local/private usage.
