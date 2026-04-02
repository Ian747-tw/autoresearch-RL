# Skill Generator Backend

Use this backend only when the user explicitly chooses the custom skill-pack path during `drl-autoresearch init`.

## Goal

Replace the bundled DRL pack with a compact custom skill pack tailored to the user's project specification, hard rules, and training domain.

## Required Inputs

Read these first:

1. `USER_SPEC.md` if present.
2. `NON_NEGOTIABLE_RULES.md`.
3. `.drl_autoresearch/onboarding.yaml` or `.drl_autoresearch/onboarding.json` if present.
4. Any project training entrypoints or environment-specific docs the user already has.

If the spec is incomplete, infer cautiously from the repo and keep the skills generic.

## Process

1. Delete only the bundled DRL pack files that shipped with init.
2. Research the user's training domain and common failure modes before writing skills.
3. Generate a small set of compact markdown skills in `skills/`.
4. Keep each skill short, reusable, and domain-general.
5. Avoid detailed implementation plans, repo-specific step lists, or token-heavy background sections.

## Skill Design Rules

- Match the DRL pack style: compact, direct, operational.
- Focus on basic techniques and recurring failure patterns.
- Prefer checklists, heuristics, and decision rules over long explanations.
- Do not bake in detailed code edits, exact experiment plans, or large literature dumps.
- Keep the total pack small. Add only the minimum set of skills needed for the user's domain.
- Use names that describe a general capability, not a one-off project task.

## Output Shape

For each skill:

- A short title.
- When to use it.
- A concise diagnostic or execution checklist.
- Common failure modes or tradeoffs.
- Clear stop conditions or escalation criteria.

## Do Not Do

- Do not recreate the original DRL pack verbatim.
- Do not write giant tutorial-style skills.
- Do not include detailed implementation plans.
- Do not include project-specific one-time instructions unless the user asked for them separately.
