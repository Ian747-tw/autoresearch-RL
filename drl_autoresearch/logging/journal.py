"""
Project journal — human-readable markdown log appended to over time.
All writes are atomic (write to .tmp then rename) for section updates,
and append-only for log events.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path

from .registry import RunRecord

_AUTO_START = "<!-- AUTO-UPDATED -->"
_AUTO_END = "<!-- END AUTO-UPDATED -->"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _now_human() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


class ProjectJournal:
    JOURNAL_PATH = "logs/project_journal.md"

    def __init__(self, project_dir: Path) -> None:
        self.project_dir = Path(project_dir)
        self.journal_path = self.project_dir / self.JOURNAL_PATH

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def initialize(self, project_name: str, spec: dict) -> None:
        """Create journal with standard header if not already present."""
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        if self.journal_path.exists():
            return

        environment = spec.get("environment", "unknown")
        algorithm = spec.get("algorithm", "unknown")
        metric_name = spec.get("metric_name", "eval_reward_mean")
        higher_is_better = spec.get("higher_is_better", True)
        direction = "higher" if higher_is_better else "lower"

        content = (
            f"# DRL AutoResearch Journal — {project_name}\n\n"
            f"**Started**: {_now_human()}\n"
            f"**Environment**: {environment}\n"
            f"**Algorithm**: {algorithm}\n"
            f"**Metric**: {metric_name} ({direction} is better)\n\n"
            "## Current State\n"
            f"{_AUTO_START}\n"
            "- **Phase**: research\n"
            "- **Best run**: none\n"
            f"- **Best {metric_name}**: n/a\n"
            "- **Total runs**: 0\n"
            "- **Kept / Discarded / Crashed**: 0 / 0 / 0\n"
            "- **Current bottleneck**: unknown\n"
            "- **Known dead ends**: none\n"
            f"{_AUTO_END}\n\n"
            "## Experiment Log\n\n"
            "## Decisions & Lessons\n\n"
            "## Overnight Summaries\n"
        )

        tmp = self.journal_path.with_suffix(".md.tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(self.journal_path)

    # ------------------------------------------------------------------
    # Generic event logging (append-only)
    # ------------------------------------------------------------------

    def log_event(
        self,
        event_type: str,
        content: str,
        metadata: dict = None,
    ) -> None:
        """Append a timestamped event block to the Experiment Log section."""
        lines = [
            f"\n### [{_now_human()}] {event_type}\n",
            f"{content}\n",
        ]
        if metadata:
            lines.append("\n**Metadata**:\n")
            for k, v in metadata.items():
                lines.append(f"- {k}: {v}\n")
        lines.append("\n---\n")
        self._append_to_section("## Experiment Log", "".join(lines))

    # ------------------------------------------------------------------
    # Typed log helpers
    # ------------------------------------------------------------------

    def log_experiment_result(self, run: RunRecord, decision: str) -> None:
        eval_mean = run.eval_reward_mean
        eval_std = run.eval_reward_std
        metric_str = (
            f"{eval_mean:.4f} ± {eval_std:.4f}"
            if eval_mean is not None and eval_std is not None
            else str(eval_mean)
        )
        content = (
            f"**Run**: `{run.run_id}`  \n"
            f"**Algorithm**: {run.algorithm}  \n"
            f"**Status**: {run.status}  \n"
            f"**Eval reward**: {metric_str}  \n"
            f"**Decision**: **{decision}**  \n"
            f"**Change summary**: {run.change_summary}  \n"
            f"**Hypothesis**: {run.hypothesis}  \n"
            f"**Notes**: {run.notes}"
        )
        self.log_event("experiment_result", content)

    def log_research_refresh(self, plan: dict, reason: str) -> None:
        plan_lines = "\n".join(f"- {k}: {v}" for k, v in plan.items())
        content = f"**Reason**: {reason}\n\n**New plan**:\n{plan_lines}"
        self.log_event("research_refresh", content)

    def log_phase_change(
        self, old_phase: str, new_phase: str, reason: str
    ) -> None:
        content = (
            f"**Transition**: {old_phase} → **{new_phase}**  \n"
            f"**Reason**: {reason}"
        )
        self.log_event("phase_change", content)

    def log_best_model_update(
        self, run_id: str, metric: float, metric_name: str
    ) -> None:
        content = (
            f"New best model: `{run_id}`  \n"
            f"**{metric_name}** = {metric:.6f}"
        )
        self.log_event("best_model_update", content)

    def log_incident(self, incident_type: str, description: str) -> None:
        content = (
            f"**Incident type**: {incident_type}  \n"
            f"**Description**: {description}"
        )
        self.log_event("incident", content)

    def log_morning_summary(self) -> str:
        """Generate an overnight summary and append it to the Overnight Summaries section."""
        summary = (
            f"\n### Overnight Summary — {_now_human()}\n\n"
            "_Auto-generated summary checkpoint._\n\n"
            "---\n"
        )
        self._append_to_section("## Overnight Summaries", summary)
        return summary

    # ------------------------------------------------------------------
    # Current State section (atomic replace)
    # ------------------------------------------------------------------

    def update_current_state_section(self, state: dict) -> None:
        """
        Atomically replace the AUTO-UPDATED block with new state fields.
        `state` is a dict whose keys match the bullet labels in the template.
        """
        text = self.read()

        start_idx = text.find(_AUTO_START)
        end_idx = text.find(_AUTO_END)
        if start_idx == -1 or end_idx == -1:
            return  # markers missing — do nothing

        # Build replacement lines
        lines = [f"{_AUTO_START}\n"]
        for k, v in state.items():
            lines.append(f"- **{k}**: {v}\n")
        lines.append(f"{_AUTO_END}")

        new_text = text[:start_idx] + "".join(lines) + text[end_idx + len(_AUTO_END):]

        tmp = self.journal_path.with_suffix(".md.tmp")
        tmp.write_text(new_text, encoding="utf-8")
        tmp.replace(self.journal_path)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def read(self) -> str:
        if not self.journal_path.exists():
            return ""
        return self.journal_path.read_text(encoding="utf-8")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _append_to_section(self, section_header: str, content: str) -> None:
        """
        Append `content` immediately after the `section_header` line.
        Falls back to appending at the end of the file if section not found.
        Uses an atomic write to avoid partial reads.
        """
        text = self.read()
        idx = text.find(f"\n{section_header}\n")
        if idx == -1:
            idx = text.find(f"\n{section_header}")
        if idx == -1:
            # Section not found — just append to end
            new_text = text + content
        else:
            # Insert after the section header line
            insert_at = text.find("\n", idx + 1) + 1
            new_text = text[:insert_at] + content + text[insert_at:]

        tmp = self.journal_path.with_suffix(".md.tmp")
        tmp.write_text(new_text, encoding="utf-8")
        tmp.replace(self.journal_path)
