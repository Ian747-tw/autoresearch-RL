"""
RunManager — manages lifecycle of a single experiment run.

Responsibilities:
- Create per-run log directories
- Track git state before/after change
- Manage checkpoints (max 3 per run, select best by eval metric)
- keep() advances git, updates state; discard() reverts git changes
- Log everything to registry + journal
"""

from __future__ import annotations

import datetime
import json
import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from drl_autoresearch.logging.registry import ExperimentRegistry, RunRecord
    from drl_autoresearch.logging.journal import ProjectJournal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MAX_CHECKPOINTS = 3
CONFIG_DIR = ".drl_autoresearch"


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _make_run_id() -> str:
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid.uuid4().hex[:6]
    return f"run_{ts}_{suffix}"


def _make_checkpoint_id() -> str:
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid.uuid4().hex[:4]
    return f"chkpt_{ts}_{suffix}"


def _first_present(source: dict, *keys: str) -> object:
    for key in keys:
        if key in source:
            return source[key]
    return None


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


# ---------------------------------------------------------------------------
# RunContext
# ---------------------------------------------------------------------------


@dataclass
class RunContext:
    run_id: str
    experiment: dict
    project_dir: Path
    started_at: str
    branch: Optional[str]
    commit_before: Optional[str]
    log_dir: Path           # logs/runs/{run_id}/
    checkpoint_dir: Path    # logs/runs/{run_id}/checkpoints/
    metrics_file: Path      # logs/runs/{run_id}/metrics.jsonl
    status: str = "running"


# ---------------------------------------------------------------------------
# RunManager
# ---------------------------------------------------------------------------


class RunManager:
    """
    Manages the full lifecycle of a single experiment run: directory setup,
    metric logging, checkpoint management, and keep/discard decisions.
    """

    def __init__(self, project_dir: Path) -> None:
        self.project_dir = Path(project_dir)
        self._config_dir = self.project_dir / CONFIG_DIR

    # ------------------------------------------------------------------
    # Start
    # ------------------------------------------------------------------

    def start_run(self, experiment: dict) -> RunContext:
        """
        Prepare directories and capture git state, then return a RunContext.
        """
        run_id = experiment.get("run_id") or _make_run_id()

        log_dir = self.project_dir / "logs" / "runs" / run_id
        checkpoint_dir = log_dir / "checkpoints"
        metrics_file = log_dir / "metrics.jsonl"

        log_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        branch = self._get_current_branch()
        commit_before = self._get_current_commit()

        ctx = RunContext(
            run_id=run_id,
            experiment=experiment,
            project_dir=self.project_dir,
            started_at=_now_iso(),
            branch=branch,
            commit_before=commit_before,
            log_dir=log_dir,
            checkpoint_dir=checkpoint_dir,
            metrics_file=metrics_file,
            status="running",
        )

        # Write a run_meta.json for traceability
        meta = {
            "run_id": run_id,
            "experiment": experiment,
            "started_at": ctx.started_at,
            "branch": branch,
            "commit_before": commit_before,
        }
        (log_dir / "run_meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        return ctx

    # ------------------------------------------------------------------
    # Finish
    # ------------------------------------------------------------------

    def finish_run(self, ctx: RunContext, result: dict) -> "RunRecord":
        """
        Finalise a run: evaluate safeguards, build a RunRecord, return it.

        result dict expected keys (all optional):
            eval_reward_mean, eval_reward_std,
            train_reward_mean, train_reward_std,
            status, notes, wall_clock_seconds,
            algorithm, environment, change_summary, agent, branch, commit,
            custom_metric_name, custom_metric_value,
        """
        # Deferred import to avoid circular dependencies at module load time
        from drl_autoresearch.logging.registry import RunRecord

        raw_status = str(result.get("status", "completed") or "completed").strip().lower()
        if raw_status in {"discard", "discarded", "keep"}:
            raw_status = "completed"
        ctx.status = raw_status

        # ── Eval safeguards ──
        eval_reward = result.get("eval_reward_mean")
        train_reward = result.get("train_reward_mean")

        warnings: list[str] = []

        if (
            eval_reward is not None
            and train_reward is not None
            and train_reward != 0.0
            and eval_reward > train_reward * 2
        ):
            msg = (
                f"[run_manager] WARNING: potential data leak detected in run {ctx.run_id}: "
                f"eval_reward ({eval_reward:.4f}) > 2x train_reward ({train_reward:.4f})."
            )
            warnings.append(msg)
            self._report_safeguard_incident(ctx, "data_leak", msg, {
                "eval_reward_mean": eval_reward,
                "train_reward_mean": train_reward,
            })

        if (
            eval_reward is not None and eval_reward == 0.0
            and train_reward is not None and train_reward > 0.0
        ):
            msg = (
                f"[run_manager] WARNING: possible eval bug in run {ctx.run_id}: "
                f"eval_reward is exactly 0.0 but train_reward={train_reward:.4f}."
            )
            warnings.append(msg)
            self._report_safeguard_incident(ctx, "invalid_eval", msg, {
                "eval_reward_mean": eval_reward,
                "train_reward_mean": train_reward,
            })

        for w in warnings:
            print(w)

        # ── Build RunRecord ──
        notes = result.get("notes", "")
        if warnings:
            extra = " | ".join(warnings)
            notes = (notes + " | SAFEGUARD: " + extra).strip(" | ") if notes else "SAFEGUARD: " + extra

        run = RunRecord(
            run_id=ctx.run_id,
            parent_run_id=ctx.experiment.get("parent_run_id", ""),
            agent=result.get("agent", ctx.experiment.get("agent", "")),
            branch=result.get("branch", ctx.branch or ""),
            commit=result.get("commit", ctx.commit_before or ""),
            environment=result.get("environment", ctx.experiment.get("environment", "")),
            algorithm=result.get("algorithm", ctx.experiment.get("algorithm", "")),
            config_summary=result.get("config_summary", ""),
            change_summary=result.get("change_summary", ctx.experiment.get("change_summary", "")),
            hypothesis=ctx.experiment.get("hypothesis", ""),
            rules_checked=result.get("rules_checked", ""),
            train_reward_mean=_maybe_float(train_reward),
            train_reward_std=_maybe_float(result.get("train_reward_std")),
            eval_reward_mean=_maybe_float(eval_reward),
            eval_reward_std=_maybe_float(result.get("eval_reward_std")),
            custom_metric_name=result.get("custom_metric_name", ""),
            custom_metric_value=_maybe_float(result.get("custom_metric_value")),
            success_rate=_maybe_float(result.get("success_rate")),
            constraint_violations=_maybe_float(result.get("constraint_violations")),
            seed_count=_maybe_float(result.get("seed_count")),
            wall_clock_seconds=_maybe_float(result.get("wall_clock_seconds")),
            gpu_memory_gb=_maybe_float(result.get("gpu_memory_gb")),
            ram_gb=_maybe_float(result.get("ram_gb")),
            status=ctx.status,
            keep_decision="",
            notes=notes,
        )

        # Persist a copy of the result alongside the run metadata
        result_path = ctx.log_dir / "run_result.json"
        result_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

        return run

    def should_publish_to_registry(
        self,
        ctx: RunContext,
        result: Optional[dict] = None,
    ) -> bool:
        """
        Return whether this run should be promoted from raw logs into the
        experiment registry/dashboard.

        Raw per-run logs are always written. Agents decide whether a run is a
        full baseline/local eval or only a temporary, partial, or specific test
        by setting `publish_to_registry`/`registry_publish`/`promote_to_registry`
        on the result or experiment metadata.

        If no explicit decision is present, keep the old behavior and publish so
        existing controller fallback/crash accounting continues to work.
        """
        result = result or {}

        for source in (result, ctx.experiment):
            explicit = _first_present(
                source,
                "publish_to_registry",
                "registry_publish",
                "promote_to_registry",
            )
            if explicit is not None:
                return _truthy(explicit)

        return True

    def _report_safeguard_incident(
        self, ctx: RunContext, incident_type: str, description: str, evidence: dict
    ) -> None:
        """Best-effort: report a safeguard incident to IncidentLog."""
        try:
            from drl_autoresearch.logging.incidents import IncidentLog
            incident_log = IncidentLog(self.project_dir)
            incident_log.report(
                incident_type=incident_type,
                run_id=ctx.run_id,
                description=description,
                evidence=evidence,
            )
        except Exception:
            pass  # non-fatal; warning was already printed

    # ------------------------------------------------------------------
    # Metric logging
    # ------------------------------------------------------------------

    def log_metric(self, ctx: RunContext, step: int, metrics: dict) -> None:
        """Append a metrics snapshot to logs/runs/{run_id}/metrics.jsonl."""
        entry = {
            "step": step,
            "timestamp": _now_iso(),
            **{k: v for k, v in metrics.items()},
        }
        with ctx.metrics_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    def checkpoint(self, ctx: RunContext, model_path: Path, metrics: dict) -> str:
        """
        Register a checkpoint.  Keeps at most MAX_CHECKPOINTS per run;
        when the limit is exceeded the worst checkpoint (by eval metric) is
        deleted from disk and the registry.

        Returns the checkpoint_id.
        """
        checkpoint_id = _make_checkpoint_id()
        eval_metric = _maybe_float(
            metrics.get("eval_reward_mean") or metrics.get("eval_metric") or metrics.get("metric")
        )

        meta = {
            "checkpoint_id": checkpoint_id,
            "run_id": ctx.run_id,
            "step": metrics.get("step", 0),
            "timestamp": _now_iso(),
            "metrics": metrics,
            "model_path": str(model_path),
            "eval_metric": eval_metric,
        }
        meta_path = ctx.checkpoint_dir / f"{checkpoint_id}.json"
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

        # Prune if over limit
        self._prune_checkpoints(ctx)

        return checkpoint_id

    def _prune_checkpoints(self, ctx: RunContext) -> None:
        """Delete the worst checkpoint when MAX_CHECKPOINTS is exceeded."""
        chkpt_files = sorted(ctx.checkpoint_dir.glob("chkpt_*.json"))
        if len(chkpt_files) <= MAX_CHECKPOINTS:
            return

        # Load all checkpoint metadata
        checkpoints: list[dict] = []
        for f in chkpt_files:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                data["_meta_path"] = str(f)
                checkpoints.append(data)
            except (OSError, json.JSONDecodeError):
                continue

        if len(checkpoints) <= MAX_CHECKPOINTS:
            return

        # Sort by eval_metric (descending); None treated as -inf
        checkpoints.sort(key=lambda c: c.get("eval_metric") or float("-inf"), reverse=True)

        # Remove the worst one(s)
        to_remove = checkpoints[MAX_CHECKPOINTS:]
        for chkpt in to_remove:
            # Delete model file if it exists and is inside project dir
            model_path_str = chkpt.get("model_path", "")
            if model_path_str:
                model_path = Path(model_path_str)
                if model_path.exists() and self._is_safe_to_delete(model_path):
                    try:
                        if model_path.is_dir():
                            shutil.rmtree(model_path)
                        else:
                            model_path.unlink()
                    except OSError:
                        pass

            # Delete the meta file
            meta_path = Path(chkpt["_meta_path"])
            try:
                meta_path.unlink(missing_ok=True)
            except OSError:
                pass

    def _is_safe_to_delete(self, path: Path) -> bool:
        """Only delete files that are inside project_dir to avoid accidents."""
        try:
            path.resolve().relative_to(self.project_dir.resolve())
            return True
        except ValueError:
            return False

    def get_best_checkpoint(self, run_id: str) -> Optional[Path]:
        """Return the model_path of the checkpoint with the best eval_metric."""
        chkpt_dir = self.project_dir / "logs" / "runs" / run_id / "checkpoints"
        if not chkpt_dir.exists():
            return None

        best_metric = float("-inf")
        best_model_path: Optional[Path] = None

        for meta_file in chkpt_dir.glob("chkpt_*.json"):
            try:
                data = json.loads(meta_file.read_text(encoding="utf-8"))
                metric = data.get("eval_metric")
                if metric is None:
                    continue
                if float(metric) > best_metric:
                    best_metric = float(metric)
                    model_path_str = data.get("model_path", "")
                    if model_path_str:
                        best_model_path = Path(model_path_str)
            except (OSError, json.JSONDecodeError, ValueError):
                continue

        return best_model_path

    # ------------------------------------------------------------------
    # Keep / Discard
    # ------------------------------------------------------------------

    def keep(
        self,
        ctx: RunContext,
        reason: str,
        registry: "ExperimentRegistry",
        journal: "ProjectJournal",
    ) -> None:
        """
        Mark a run as kept:
        1. Update registry with keep decision.
        2. Log to journal.
        3. Update state.json best model if this run improved the best metric.

        Git advancement is intentionally NOT performed here — the orchestrator/
        agent handles that so it can make informed decisions.
        """
        ctx.status = "completed"

        # 1. Update registry
        try:
            registry.update_run(
                ctx.run_id,
                {"status": "completed", "keep_decision": "keep", "notes": reason},
            )
        except KeyError:
            pass  # run not yet in registry — non-fatal

        # 2. Log to journal
        try:
            run_record = registry.get_run(ctx.run_id)
            if run_record is not None:
                journal.log_experiment_result(run_record, "keep")
            else:
                journal.log_event(
                    "keep_decision",
                    f"Run `{ctx.run_id}` kept. Reason: {reason}",
                )
        except Exception:
            pass  # non-fatal

        # 3. Update state.json best model
        self._maybe_update_best_in_state(ctx, registry)

    def discard(
        self,
        ctx: RunContext,
        reason: str,
        registry: "ExperimentRegistry",
        journal: "ProjectJournal",
    ) -> None:
        """
        Mark a run as discarded:
        1. Revert uncommitted changes via git checkout -- . (safe; only if HEAD unchanged).
        2. Update registry with discard decision.
        3. Log to journal.
        """
        ctx.status = "completed"

        # 1. Git revert
        self._git_revert_changes(ctx)

        # 2. Update registry
        try:
            registry.update_run(
                ctx.run_id,
                {"status": "completed", "keep_decision": "discard", "notes": reason},
            )
        except KeyError:
            pass  # run not yet in registry — non-fatal

        # 3. Log to journal
        try:
            run_record = registry.get_run(ctx.run_id)
            if run_record is not None:
                journal.log_experiment_result(run_record, "discard")
            else:
                journal.log_event(
                    "discard_decision",
                    f"Run `{ctx.run_id}` discarded. Reason: {reason}",
                )
        except Exception:
            pass  # non-fatal

    def _maybe_update_best_in_state(
        self, ctx: RunContext, registry: "ExperimentRegistry"
    ) -> None:
        """Update state.json if this run's reward beats the current best."""
        run_record = None
        try:
            run_record = registry.get_run(ctx.run_id)
        except Exception:
            pass

        if run_record is None:
            return

        reward_val = run_record.eval_reward_mean
        if reward_val is None:
            return

        state_path = self._config_dir / "state.json"
        if not state_path.exists():
            return

        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            current_best = state.get("best_metric_value")
            if current_best is None or float(reward_val) > float(current_best):
                state["best_run_id"] = ctx.run_id
                state["best_metric_value"] = reward_val
                state["best_metric_name"] = "reward"
                # Atomic write
                tmp = state_path.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
                tmp.replace(state_path)
        except (OSError, json.JSONDecodeError, ValueError):
            pass  # non-fatal

    # ------------------------------------------------------------------
    # Summary generation
    # ------------------------------------------------------------------

    def generate_run_summary(self, ctx: RunContext) -> str:
        """
        Generate a human-readable summary string for the run.
        Reads from run_result.json and metrics.jsonl if available.
        """
        lines: list[str] = [
            f"# Run Summary: {ctx.run_id}",
            f"Status    : {ctx.status}",
            f"Started   : {ctx.started_at}",
            f"Branch    : {ctx.branch or 'unknown'}",
            f"Commit    : {ctx.commit_before or 'unknown'}",
            f"Hypothesis: {ctx.experiment.get('hypothesis', 'N/A')}",
        ]

        result_path = ctx.log_dir / "run_result.json"
        if result_path.exists():
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
                if result.get("eval_reward_mean") is not None:
                    lines.append(f"Eval reward  : {result['eval_reward_mean']:.4f}")
                if result.get("train_reward_mean") is not None:
                    lines.append(f"Train reward : {result['train_reward_mean']:.4f}")
                if result.get("wall_clock_seconds") is not None:
                    secs = int(result["wall_clock_seconds"])
                    lines.append(f"Wall clock   : {secs//3600}h {(secs%3600)//60}m {secs%60}s")
                if result.get("notes"):
                    lines.append(f"Notes        : {result['notes']}")
            except (OSError, json.JSONDecodeError):
                pass

        # Count logged metric steps
        if ctx.metrics_file.exists():
            try:
                step_count = sum(1 for _ in ctx.metrics_file.open(encoding="utf-8"))
                lines.append(f"Metric steps : {step_count}")
            except OSError:
                pass

        best_ckpt = self.get_best_checkpoint(ctx.run_id)
        if best_ckpt:
            lines.append(f"Best chkpt   : {best_ckpt}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Git helpers
    # ------------------------------------------------------------------

    def _get_current_branch(self) -> Optional[str]:
        """Return the current git branch name, or None if unavailable."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=str(self.project_dir),
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                branch = result.stdout.strip()
                return branch if branch else None
        except (FileNotFoundError, subprocess.SubprocessError, OSError):
            pass
        return None

    def _get_current_commit(self) -> Optional[str]:
        """Return the current HEAD commit SHA, or None if unavailable."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(self.project_dir),
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                commit = result.stdout.strip()
                return commit if commit else None
        except (FileNotFoundError, subprocess.SubprocessError, OSError):
            pass
        return None

    def _git_revert_changes(self, ctx: RunContext) -> None:
        """
        Revert uncommitted changes using `git checkout -- .`.

        Safety check: only revert if the current HEAD still matches
        commit_before (to avoid reverting changes made by a previous kept run).
        Logs a warning instead of raising if anything goes wrong.
        """
        if ctx.commit_before is None:
            print(
                f"[run_manager] Skipping git revert for {ctx.run_id}: "
                "commit_before is unknown."
            )
            return

        current_commit = self._get_current_commit()
        if current_commit != ctx.commit_before:
            print(
                f"[run_manager] Skipping git revert for {ctx.run_id}: "
                f"HEAD has moved from {ctx.commit_before[:8]} to "
                f"{(current_commit or 'unknown')[:8]}. "
                "Not reverting to avoid data loss."
            )
            return

        try:
            result = subprocess.run(
                ["git", "checkout", "--", "."],
                cwd=str(self.project_dir),
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                print(
                    f"[run_manager] WARNING: git revert for {ctx.run_id} exited "
                    f"with code {result.returncode}: {result.stderr.strip()}"
                )
            else:
                print(f"[run_manager] Reverted changes for discarded run {ctx.run_id}.")
        except (FileNotFoundError, subprocess.SubprocessError, OSError) as exc:
            print(
                f"[run_manager] WARNING: git revert failed for {ctx.run_id}: {exc}. "
                "Continuing without revert."
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"RunManager(project_dir={self.project_dir!r})"


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------


def _maybe_float(val) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None
