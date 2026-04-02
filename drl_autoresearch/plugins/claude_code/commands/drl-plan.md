# /drl-plan — Review and Update Experiment Plan

Use this to understand the current research state, see what the orchestrator
recommends, and optionally queue or override experiments.

## Step 1: Current status

Run:
```
drl-autoresearch status --project-dir .
```

Show the user:
- Current phase (research / baseline / experimenting / focused_tuning / ablation / converged)
- Best run so far (run_id, metric value)
- Total runs (kept / discarded / crashed)
- Active workers (if any)
- Open incidents (if any — these should be addressed before new experiments)

## Step 2: Summarize current research direction

Read `IMPLEMENTATION_PLAN.md`. Summarize in 3–5 sentences:
- What approach is being pursued and why
- What has been tried already
- What the main open questions are
- What compute has been used and how much remains

## Step 3: Get orchestrator recommendation

Run:
```
drl-autoresearch plan --project-dir .
```

Show the user:
- What the orchestrator recommends next and why
- The hypothesis being tested
- The specific change(s) to make
- Expected effect and risk level
- Estimated compute cost

## Step 4: User decision

Ask the user:

**Option A — Accept orchestrator recommendation**
"Proceed with the recommended experiment?"

**Option B — Override with custom experiment**
Capture from the user:
- Hypothesis: what are we testing and why?
- Specific change: what exactly will be modified?
- Expected effect: what result would confirm the hypothesis?
- Risk level: low / medium / high

Then validate against `NON_NEGOTIABLE_RULES.md`:
```
drl-autoresearch check --project-dir . --action <action_type> --details '...'
```

If blocked: explain why and suggest a compliant alternative.

**Option C — Research refresh first**
If the user is not confident in any direction: run `/drl-research` before queueing.

## Step 5: Queue approved experiments

If the user approves one or more experiments, add them to the queue:
```
drl-autoresearch plan --queue --project-dir .
```

Confirm to the user:
- What is queued
- In what order
- Estimated total compute cost
- When to expect results (based on typical run time)

## Step 6: Optional — pre-flight check

Before running, verify:
- [ ] Hard rules checked for all queued experiments
- [ ] Eval protocol unchanged
- [ ] Baseline run exists (if not: first experiment should be a baseline)
- [ ] Hardware resources available
- [ ] No critical incidents open

Run:
```
drl-autoresearch doctor --project-dir .
```

If doctor shows any blocking issues, fix them before starting.
