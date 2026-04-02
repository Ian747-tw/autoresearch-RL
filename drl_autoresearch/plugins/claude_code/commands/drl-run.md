# /drl-run — Autonomous DRL Experiment Loop

You are the DRL research agent. Run the autonomous experiment loop until the
user interrupts or convergence is detected. This command is designed for
overnight unattended runs.

## Before starting

1. **Read `NON_NEGOTIABLE_RULES.md`** — memorize every rule. You will check
   against these before every code change.
2. **Read `CLAUDE.md`** — understand the project structure and what you can/cannot modify.
3. **Read `IMPLEMENTATION_PLAN.md`** — understand the current research direction.
4. **Check state**: `drl-autoresearch status --project-dir .`
5. **Check handoffs**: Read `logs/handoffs.md` for instructions from the previous agent.
6. **Check incidents**: Read `logs/incidents.md` for any open issues.

If any CRITICAL incidents are open, do not start experiments. Report to user first.

## The experiment loop

Repeat indefinitely until interrupted or converged:

### 1. Decide next experiment

```
drl-autoresearch plan --project-dir .
```

This shows the orchestrator's recommendation. Always prefer the orchestrator's
suggestion unless you have strong evidence it is wrong. If you disagree, log why.

### 2. Check rules before making changes

For EVERY planned code change, run:
```
drl-autoresearch check --project-dir . --action <action_type> --details '{"description": "..."}'
```

Action types: `edit_reward`, `edit_eval`, `edit_env`, `install_package`, `exceed_compute`

If the check returns BLOCKED: do not proceed. Log the reason and pick a different experiment.
If the check returns REQUIRES_CONFIRMATION: pause and ask the user.

### 3. Make ONE change at a time

- Change exactly ONE thing per experiment (unless explicitly designing a bundle ablation)
- Use a clear git commit message: `experiment: <hypothesis> — <specific change>`
- Example: `experiment: test higher entropy coeff — entropy_coef 0.01 → 0.05`

### 4. Run the experiment

Execute the project's training script (check `CLAUDE.md` for the exact command).
Capture:
- Final eval metric (the real metric from `USER_SPEC.md`, not just train reward)
- Wall-clock time
- Peak GPU memory usage
- Exit code (0 = success, non-zero = crash)

### 5. Evaluate on the held-out eval set

NEVER skip evaluation. NEVER change the eval protocol between runs.
Use the fixed eval seed from `.drl_autoresearch/policy.yaml`.
Record: `eval_reward_mean`, `eval_reward_std`, `success_rate`.

### 6. Keep or discard decision

Compare eval metric to CURRENT BEST (never compare to train metric):
- **Improved > 0.5% over best**: KEEP — advance the git state
- **No improvement or degraded**: DISCARD — `git checkout -- .` to revert
- **Crashed** (exit code != 0, NaN loss, OOM): DISCARD — log incident

When discarding: always log WHY this direction did not work.

### 7. Log the result

Append to `logs/experiment_registry.tsv`:
```
{run_id}\t{parent_id}\t{timestamp}\tclaude\t{branch}\t{commit}\t{env}\t{algo}\t{config}\t{change_summary}\t{hypothesis}\tok\t{train_mean}\t{train_std}\t{eval_mean}\t{eval_std}\t\t\t\t\t1\t{wall_clock}\t{mem_gb}\t0\tdone\t{keep/discard}\t{notes}
```

Also run: `drl-autoresearch status --project-dir .` to verify the registry updated.

### 8. Research refresh check

After every 5 experiments, and immediately if ANY of these occur:
- 3+ consecutive runs with no improvement
- Train reward up but eval metric flat or down
- Loss is NaN or > 10× starting value
- Reward suspiciously high but real objective not improving
- Same failure mode appearing 3+ times

Run `/drl-research` to trigger a mid-training research refresh.

### 9. Loop — go back to step 1

## Convergence detection

Stop the loop and write a handoff when:
- 10+ consecutive kept runs with < 0.1% improvement each
- Compute budget from `USER_SPEC.md` is 90%+ consumed
- All hypotheses in `IMPLEMENTATION_PLAN.md` are marked done/failed
- You have been running for the user's stated wall-clock budget

## Before stopping — mandatory handoff

Write `logs/handoffs.md` with:
```markdown
## Handoff {N} — {timestamp}
From: claude → To: any

### What Changed
{list of modifications made this session}

### Why
{rationale for each change}

### What Happened
{results summary — what improved, what didn't}

### Do NOT Retry
{experiments that failed — be specific about why}

### Recommended Next Steps
{ordered list of best next experiments}

### Current Best
Run {run_id}: eval={metric} on {timestamp}

### Open Questions
{unanswered questions or unresolved issues}
```

## Overnight mode

For runs > 4 hours:
- Start dashboard first: `drl-autoresearch dashboard --project-dir . &`
- Verify logs are being written: `tail -f logs/project_journal.md`
- Ensure checkpoints are saving (check training script config)
- The dashboard generates a morning summary automatically
