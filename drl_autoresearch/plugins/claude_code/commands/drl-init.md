# /drl-init — Initialize DRL AutoResearch Project

You are setting up a DRL AutoResearch project. The user has opened their own
training project directory — this is NOT the autoresearch-RL repo. Follow these
steps to initialize it.

## Step 1: Check if drl-autoresearch is installed

Run:
```bash
drl-autoresearch --version
```

If the command is not found, install it globally:
```bash
# with pip:
pip install drl-autoresearch

# or with uv:
uv tool install drl-autoresearch

# or from source (development):
pip install -e /path/to/autoresearch-RL
```

## Step 2: Check initialization status

Check if `.drl_autoresearch/state.json` exists in the current directory.

- **YES**: Tell the user the project is already initialized. Show current state:
  `drl-autoresearch status`
  Ask: "Do you want to re-initialize? This will regenerate config but NOT touch
  your logs or experiment history."
- **NO**: Proceed with initialization.

## Step 3: Run init with plugin selection

```bash
drl-autoresearch init
```

This will:
1. Scaffold `.drl_autoresearch/`, `logs/`, `skills/`, `NON_NEGOTIABLE_RULES.md`
2. Run onboarding to capture the user's project spec and hard rules
3. Ask which skill-pack mode to use:
   - **Provided DRL pack** — keep the bundled compact DRL skills
   - **Custom pack** — remove the bundled DRL skills from this project and install `.drl_autoresearch/backend/skill_generator.md` so the agent can research and generate a compact domain-specific replacement pack
4. Ask which project mode to use:
   - **Build mode** — from-scratch design/build workflow; run compact deep research and create compact `implementation_plan/*.md` before normal training loops
   - **Improve mode** — assume a working model already exists and optimize it directly
5. Ask which AI agent plugin(s) to install:
   - **Claude Code** — installs `/drl-*` slash commands into `.claude/commands/`
   - **Codex** — installs `AGENT.md` operating guide
   - **Both** (recommended)
   - **None**
6. Run best-effort environment remediation from onboarding preferences (venv + required package setup), then advise running doctor.

For non-interactive CI environments:
```bash
drl-autoresearch init --auto                  # installs both plugins, keeps DRL pack
drl-autoresearch init --plugin cc             # Claude Code only
drl-autoresearch init --plugin codex          # Codex only
drl-autoresearch init --plugin both           # both, skip prompt
drl-autoresearch init --plugin none           # no plugins
drl-autoresearch init --skill-pack custom     # use custom compact skill-pack path
drl-autoresearch init --project-mode build    # build-from-scratch mode
```

## Step 4: Review generated files

After init, confirm with the user:

1. **`NON_NEGOTIABLE_RULES.md`** — Read rules aloud.
   Ask: "Any rules to add or change?"

2. **`.drl_autoresearch/permissions.yaml`** — Confirm permission mode.
   Permission modes: `locked` (default) | `prompted` | `open` | `bootstrap-only` | `project-only`
   Ask: "Is `locked` the right policy? For overnight autonomous runs, consider `open`."

## Step 5: Run doctor

```bash
drl-autoresearch doctor
# if dependencies are missing:
drl-autoresearch doctor --fix
```

Show the full output. All 14 checks must pass before running experiments.
If any fail, explain what to fix.

## Step 6: Confirm ready

Tell the user:
- ✓ Project initialized at: `{project_dir}`
- ✓ Hard rules active: `NON_NEGOTIABLE_RULES.md`
- ✓ Plugin(s) installed: `{list}`
- ✓ Dashboard: `drl-autoresearch dashboard &` then open http://localhost:8765

**Next steps:**
```bash
# Start dashboard (optional but recommended)
drl-autoresearch dashboard &

# Run a dry-run to validate the plan
drl-autoresearch run --dry-run

# Start the autonomous experiment loop
/drl-run
```
