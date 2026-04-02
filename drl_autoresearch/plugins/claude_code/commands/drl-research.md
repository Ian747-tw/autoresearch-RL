# /drl-research — Mid-Training Research Refresh

Run this when the current strategy is failing and you need to step back, look
at what the evidence says, and find a new direction before continuing.

## When is a research refresh justified?

A refresh is justified when ANY of these are true:
- No improvement in the last 5+ kept runs (plateau)
- Train reward improves but eval metric does not (possible overfitting/hacking)
- Instability that hyperparameter search has not fixed after 3+ attempts
- Reward appears shaped away from the real success criterion
- You have tried 3+ architectural changes with no consistent gain
- A broken core assumption has been discovered
- The orchestrator's `should_trigger_research_refresh()` returns True

Do NOT trigger a refresh just because one run failed. Require a clear pattern.

## Step 1: Evidence summary

Read `logs/experiment_registry.tsv`. Summarize:
- Total runs: kept vs discarded vs crashed
- Best metric achieved and when
- What categories of changes were tried (algo, reward, arch, hyperparams)
- Which categories helped and which did not
- The trend over last 10 runs (improving / flat / degrading)

Produce a concise 5–10 line summary. Be specific about numbers.

## Step 2: List invalidated assumptions

Open `IMPLEMENTATION_PLAN.md` and find the assumptions section.
For each assumption, state whether the experiment history supports or refutes it.

Format:
```
✗ INVALIDATED: "Higher learning rate will improve convergence speed"
  Evidence: 4 runs with LR 3e-4 to 1e-3 — no consistent improvement (runs 012–015)

✓ STILL VALID: "PPO is appropriate for this discrete action space"
  Evidence: no systematic evidence against it

? UNTESTED: "Value network capacity is the bottleneck"
  Evidence: not yet tested
```

## Step 3: Generate alternative hypotheses

For each invalidated assumption, generate 1–2 alternative hypotheses.

Be concrete. Not: "try a different algorithm."
Instead: "Switch from PPO to SAC because the environment has a smooth,
dense reward signal. SAC's off-policy learning should be more sample-efficient.
Expected improvement: 10–20% better sample efficiency. Cost: 1× normal budget."

Include:
- The hypothesis (what and why)
- The expected mechanism (why it should work)
- Compute cost estimate
- Pass criterion (what result would confirm it)
- Fail criterion (what result would deny it)

## Step 4: Design the probe experiments

For each hypothesis, design the MINIMUM experiment that would confirm/deny it:
- Use 1/4 to 1/2 of normal compute budget if possible
- Change exactly one thing
- Include a control run if the comparison is ambiguous
- State clearly what result = confirmed and what result = denied

Rank by: (expected information if confirmed) / (cost). Best ratio first.

## Step 5: Update IMPLEMENTATION_PLAN.md

Rewrite the "Next Experiments" section. Keep the history of what was tried.
Add the new hypotheses and experiment order.

Mark old failed directions clearly:
```markdown
## ~~Direction 1: Learning Rate Search~~ — EXHAUSTED
Tried: 5 runs across 1e-4 to 1e-3. No consistent improvement. Ruled out.

## Direction 2: Algorithm Switch (NEW — Research Refresh v2)
Hypothesis: ...
```

## Step 6: Run the automated refresh

```
drl-autoresearch research --project-dir .
```

This updates the orchestrator's queue with the new experiments.

## Step 7: Log the refresh

Append to `logs/project_journal.md`:

```markdown
## Research Refresh — {timestamp}
**Trigger**: {reason — plateau / train-eval gap / etc.}
**Runs analyzed**: {N} ({kept} kept, {discarded} discarded, {crashed} crashed)
**Best so far**: {run_id} with {metric}={value}

**Invalidated assumptions**:
- {assumption 1}: {evidence}
- {assumption 2}: {evidence}

**New hypotheses**:
1. {hypothesis}: {rationale}, cost={estimate}
2. {hypothesis}: {rationale}, cost={estimate}

**New experiment order**:
1. {experiment} — tests {hypothesis}
2. {experiment} — tests {hypothesis}

**Directions ruled out**:
- {direction}: {reason}
```
