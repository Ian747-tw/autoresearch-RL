# NON_NEGOTIABLE RULES
<!-- This file is enforced by drl-autoresearch check and may not be overridden
     by config files, policy mode, or agent instructions. -->

## Safety

1. **Never delete source code** outside of `logs/` or `skills/`.
2. **Never modify this file** (`NON_NEGOTIABLE_RULES.md`).
3. **Never disable the permission check** (`drl-autoresearch check`).
4. **Never exfiltrate experiment data** to external services unless the user
   has explicitly allowed it.
5. **Never perform global package installs**.

## Experiment integrity

6. All experiment results **must be logged** to `logs/experiment_registry.tsv`
   before being used to update the plan.
7. Metrics must be recorded **as-measured**; no post-hoc manipulation.
8. A run that crashes must be recorded with `status=crashed`; results from
   incomplete runs must not influence best-model selection.
9. **Never modify evaluation code or protocol** without explicit human approval.

## Resource limits

10. GPU/CPU usage must stay within the configured project limits.
11. Disk usage for checkpoints must stay within the configured project limits.

## Human override

12. **Never edit policy or permission config files** without explicit human approval.
13. **Never delete checkpoints** without explicit human approval.
14. **Never run ad hoc shell commands outside normal project execution** without
    explicit human approval.
