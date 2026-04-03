# DRL AutoResearch — Codex Agent Guide

This file is your operating guide. Read it before touching any code or running
any experiment. Every action must follow the protocol below.

## What is DRL AutoResearch?

`drl-autoresearch` is an autonomous DRL research loop installed globally on this
machine. It manages experiment state, enforces hard rules, and streams metrics
to a live dashboard. You interact with it through its CLI.

## CLI Commands Available

```bash
drl-autoresearch doctor              # check environment (run first)
drl-autoresearch status              # current phase, best run, counters
drl-autoresearch plan [--refresh]    # ranked experiment plan
drl-autoresearch run [--dry-run]     # start the continuous agent loop
drl-autoresearch run --once          # run one autonomous agent cycle
drl-autoresearch run --agent-backend codex
drl-autoresearch check --action X    # gate any action against hard rules
drl-autoresearch research            # trigger mid-training research refresh
drl-autoresearch dashboard --port 8765 &   # start live dashboard
```

## Mandatory Pre-Flight (run before anything else)

```bash
drl-autoresearch doctor              # must show 0 failures
drl-autoresearch status              # read current phase + best run
```

Read `NON_NEGOTIABLE_RULES.md`. Every rule is enforced programmatically — do not
assume you can skip a rule check.

If present, read `.drl_autoresearch/spec_compact.md` before planning changes.
Treat it as a token-saving navigator only, not source of truth. For details,
follow its pointers (for example `USER_SPEC.md:300`) and read the original lines.

## Session Context Sync (Token-Saving, Mandatory)

When switching agents or starting a new session, sync context with compact reads
only. Do NOT read full logs unless the compact sync is insufficient.

```bash
# 1) latest run records only
tail -n 25 logs/experiment_registry.tsv

# 2) latest journal events only
tail -n 120 logs/project_journal.md

# 3) latest handoff/incidents only
tail -n 80 logs/handoffs.md
tail -n 80 logs/incidents.md
```

After these reads, write a 5-line max "session sync checkpoint" in your working
notes:
- current phase and mode
- best run + metric
- latest 3 outcomes (keep/discard/crash)
- open incidents/handoff constraints
- next experiment intent

Only expand to deeper log reading if one of these fields is ambiguous or
conflicting.

## Policy Check — Required Before Every Code Change

Before executing ANY action that modifies code or configuration:

```bash
drl-autoresearch check --action <type> --details '{"description": "..."}'
```

Action types: `edit_reward` | `edit_eval` | `edit_env` | `install_package` |
`update_package` | `global_install` | `exceed_compute` | `gpu_memory_risk` |
`eval_protocol_change` | `use_privileged_info` | `custom`

Exit 0 = allowed. Exit 1 = blocked — do not proceed. Respect the result.

## Continuous Runtime Model

`drl-autoresearch run` is the controller. It keeps selecting the next
experiment and launches Codex or Claude Code for each autonomous cycle.

- In `build` mode, bootstrap/build tasks happen first, then continuous training and improvement cycles continue.
- In `improve` mode, the controller starts continuous optimization cycles immediately.
- The live loop state is written into `.drl_autoresearch/state.json` and exposed in the dashboard.

Use `drl-autoresearch run --once` if you want one autonomous cycle only.

## Writing to the Experiment Registry

Append one row to `logs/experiment_registry.tsv` after every run:

```
{run_id}\t{parent_run_id}\t{timestamp}\tcodex\t{branch}\t{commit}\t{env}\t{algo}\t{config}\t{change}\t{hypothesis}\tok\t{train_mean}\t{train_std}\t{eval_mean}\t{eval_std}\t\t\t\t\t1\t{wall_secs}\t{gpu_gb}\t{ram_gb}\t{status}\t{keep/discard}\t{notes}
```

Column order matches `logs/experiment_registry.tsv` header exactly.
Use Python's `drl_autoresearch.logging.registry.ExperimentRegistry` for safe
atomic writes:

```python
from pathlib import Path
from drl_autoresearch.logging.registry import ExperimentRegistry, RunRecord

registry = ExperimentRegistry(project_dir=Path("."))
record = RunRecord(
    run_id="run_...",
    agent="codex",
    hypothesis="...",
    algorithm="PPO",
    environment="CartPole-v1",
    eval_reward_mean=450.0,
    train_reward_mean=380.0,
    wall_clock_seconds=120.0,
    status="completed",
    keep_decision="keep",
)
registry.add_run(record)
```

## Writing Dashboard Artifact (training curves)

Write `logs/artifacts/<run_id>/metrics.json` to populate dashboard charts:

```python
import json
from pathlib import Path

artifact_dir = Path("logs/artifacts") / run_id
artifact_dir.mkdir(parents=True, exist_ok=True)
(artifact_dir / "metrics.json").write_text(json.dumps({
    "steps":        [100, 200, ...],
    "rewards":      [12.3, 18.7, ...],
    "losses":       [0.42, 0.38, ...],
    "eval_steps":   [500, 1000, ...],
    "eval_rewards": [22.5, 31.0, ...],
}))
```

## When to Stop

Write a handoff to `logs/handoffs.md` and stop when:
- 10+ consecutive runs with < 0.1% improvement
- Compute budget 90%+ consumed
- All plan hypotheses exhausted
- User's stated wall-clock budget reached

## Handoff Format

```markdown
## Handoff {N} — {timestamp}
From: codex → To: any

### What Changed
{modifications made this session}

### Results
{kept runs, best metric, trend}

### Do NOT Retry
{failed experiments and why}

### Recommended Next Steps
{ordered, specific}

### Current Best
Run {run_id}: eval={value} on {timestamp}
```
