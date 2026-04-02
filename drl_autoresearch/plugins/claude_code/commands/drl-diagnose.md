# /drl-diagnose — Diagnose DRL Training Failures

Run this when: training is not working, results are confusing, a run crashed
unexpectedly, or you want to understand a failure pattern before proceeding.

## Step 1: Gather evidence

Read these files in order:
1. `logs/experiment_registry.tsv` — focus on the last 10 runs
2. `logs/incidents.md` — any open incidents
3. `logs/project_journal.md` — "Current State" section

Run:
```
drl-autoresearch status --project-dir .
```

Collect:
- Last N run results (metric values, statuses)
- Any crashes and their exit codes / error messages
- Train vs eval metric gap over time
- Resource usage patterns

## Step 2: Classify the failure pattern

Map your evidence to one of these patterns:

| Pattern | Key indicators | Skill to apply |
|---------|---------------|----------------|
| **Crash loop** | status=crash 3+ times, non-zero exit | `skills/env_diagnostics.md` |
| **Flat training** | train_reward constant or near-zero across all runs | `skills/exploration.md` |
| **Train/eval gap** | train improving, eval flat or diverging | `skills/checkpoint_selection.md` |
| **Instability** | reward/loss with very high variance, intermittent NaN | `skills/investigate.md` |
| **Reward hacking** | reward_mean high, but real metric (SUCCESS_RATE) not improving | `skills/reward_shaping.md` |
| **Slow convergence** | improving but extremely slowly, low sample efficiency | `skills/compute_budgeting.md` |
| **Random pattern** | no trend visible, results appear random | `skills/env_diagnostics.md` |
| **Regression** | was improving, now degraded compared to a known-good run | `skills/investigate.md` |

If multiple patterns apply, address the most severe first (crash loop > reward
hacking > instability > train/eval gap > flat training).

## Step 3: Apply the skill

Read the appropriate skill file from `skills/` and follow the diagnostic
procedure step by step. The skill files contain:
- Specific questions to ask
- Specific measurements to take
- Common root causes for the pattern
- Suggested interventions

## Step 4: Generate discriminating hypotheses

From the diagnosis, produce 2–3 hypotheses. For each:
- State the hypothesis precisely: "The problem is X because of evidence Y"
- Design the cheapest experiment that would confirm or deny it
- Estimate compute cost (fraction of normal budget)
- State a clear pass/fail criterion

Rank by: (expected information gain) / (compute cost). Cheapest informative experiment first.

## Step 5: Check hypotheses against rules

For each proposed experiment:
```
drl-autoresearch check --project-dir . --action <action_type> --details '{"hypothesis": "..."}'
```

Remove any experiments that are blocked by hard rules.

## Step 6: Log the diagnosis

Append to `logs/project_journal.md`:

```markdown
## Diagnosis — {timestamp}
**Trigger**: {what caused this diagnosis}
**Pattern detected**: {pattern name from table above}
**Evidence**:
- {observation 1}
- {observation 2}
**Root cause hypothesis**: {most likely cause}
**Alternative hypotheses**:
1. {hypothesis}: test by {experiment}, cost ~{fraction} of budget
2. {hypothesis}: ...
**Recommended experiments** (ordered):
1. {experiment 1} — tests {hypothesis}, expected cost {N}
2. {experiment 2} — ...
**Ruled out**:
- {approach}: ruled out because {evidence}
```

## Step 7: Update the plan

Run:
```
drl-autoresearch plan --refresh --project-dir .
```

This updates `IMPLEMENTATION_PLAN.md` with the new hypotheses and experiment order.

If the diagnosis is severe (reward hacking, eval bug, broken assumption), also
run `/drl-research` to do a full mid-training research refresh.
