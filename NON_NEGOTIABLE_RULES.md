# NON_NEGOTIABLE RULES
<!-- This file is enforced by drl-autoresearch check and may not be overridden
     by policy.yaml or agent instructions. -->

## Safety

1. **Never delete source code** outside of `logs/` or `skills/`.
2. **Never modify this file** (`NON_NEGOTIABLE_RULES.md`).
3. **Never disable the permission check** (`drl-autoresearch check`).
4. **Never exfiltrate experiment data** to external services without explicit
   user configuration in `permissions.yaml`.

## Experiment integrity

5. All experiment results **must be logged** to `logs/experiment_registry.tsv`
   before being used to update the plan.
6. Metrics must be recorded **as-measured**; no post-hoc manipulation.
7. A run that crashes must be recorded with `status=crashed`; results from
   incomplete runs must not influence best-model selection.

## Resource limits

8. GPU/CPU usage must stay within limits defined in `hardware.yaml`.
9. Disk usage for checkpoints must not exceed the project-level quota
   (if set in `policy.yaml`).

## Human override

10. Any action listed under `require_human_approval` in `permissions.yaml`
    **must** pause and wait for explicit human confirmation before proceeding.
