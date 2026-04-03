# /drl-run — Autonomous DRL Runtime

You are the DRL research agent. Use the DRL AutoResearch controller runtime.
It is designed for unattended runs and will keep launching autonomous cycles
until interrupted or convergence is detected.

## Before starting

1. **Read `NON_NEGOTIABLE_RULES.md`** — memorize every rule. You will check
   against these before every code change.
2. **Read `CLAUDE.md`** — understand the project structure and what you can/cannot modify.
3. **Check state**: `drl-autoresearch status --project-dir .`
4. **Read compact spec index first (if present)**:
   - `.drl_autoresearch/spec_compact.md`
   - Use pointers inside it to open original source lines only when needed.
5. **Token-saving context sync (mandatory, compact reads only)**:
   - `tail -n 25 logs/experiment_registry.tsv`
   - `tail -n 120 logs/project_journal.md`
   - `tail -n 80 logs/handoffs.md`
   - `tail -n 80 logs/incidents.md`
6. **Read `IMPLEMENTATION_PLAN.md`** only if needed to resolve ambiguity from the compact sync.
7. Write a compact session checkpoint (max 5 lines):
   - phase/mode
   - best run + metric
   - latest 3 outcomes
   - open incidents/handoff constraints
   - next experiment intent

If any CRITICAL incidents are open, do not start experiments. Report to user first.

## Runtime entrypoint

Start the controller from the project root:

```bash
drl-autoresearch run --project-dir .
```

Useful variants:

```bash
drl-autoresearch run --project-dir . --once
drl-autoresearch run --project-dir . --agent-backend claude
drl-autoresearch run --project-dir . --agent-backend codex
```

Controller behavior:

- `build` mode: bootstrap/build work first, then continuous coding and training cycles
- `improve` mode: immediate continuous optimization cycles
- live loop status is written to `.drl_autoresearch/state.json` and shown in the dashboard
- autonomous execution requires permissive onboarding policy such as `open`

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
