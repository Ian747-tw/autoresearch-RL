"""
Handoff log — tracks agent handoffs so multiple agents can work over time
without losing context.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def _now_human() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


# ---------------------------------------------------------------------------
# HandoffRecord dataclass
# ---------------------------------------------------------------------------

@dataclass
class HandoffRecord:
    handoff_id: str
    timestamp: str
    from_agent: str
    to_agent: str                   # "any" if unspecified
    what_changed: str               # What was modified this session
    why: str                        # Rationale for changes
    what_happened: str              # Results / outcomes
    do_not_retry: list[str]         # Experiments that failed and should not be repeated
    next_steps: list[str]           # Recommended next experiments
    current_best: str               # Best run ID and metric
    open_questions: list[str]       # Unanswered questions

    def to_markdown_section(self, n: int) -> str:
        def bullets(lst: list[str]) -> str:
            if not lst:
                return "- (none)\n"
            return "".join(f"- {item}\n" for item in lst)

        return (
            f"## Handoff {n} — {self.timestamp}\n"
            f"**From**: {self.from_agent} → **To**: {self.to_agent}\n\n"
            f"### What Changed\n{self.what_changed}\n\n"
            f"### Why\n{self.why}\n\n"
            f"### What Happened\n{self.what_happened}\n\n"
            f"### Do NOT Retry\n{bullets(self.do_not_retry)}\n"
            f"### Recommended Next Steps\n{bullets(self.next_steps)}\n"
            f"### Current Best\n{self.current_best}\n\n"
            f"### Open Questions\n{bullets(self.open_questions)}\n"
        )


# ---------------------------------------------------------------------------
# HandoffLog
# ---------------------------------------------------------------------------

class HandoffLog:
    HANDOFFS_PATH = "logs/handoffs.md"

    def __init__(self, project_dir: Path) -> None:
        self.project_dir = Path(project_dir)
        self.handoffs_path = self.project_dir / self.HANDOFFS_PATH
        self._project_name: str = project_dir.name

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        self.handoffs_path.parent.mkdir(parents=True, exist_ok=True)
        if self.handoffs_path.exists():
            return
        content = f"# Agent Handoffs — {self._project_name}\n\n"
        tmp = self.handoffs_path.with_suffix(".md.tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(self.handoffs_path)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def record_handoff(self, handoff: HandoffRecord) -> None:
        """Append a new handoff to the file atomically."""
        if not self.handoffs_path.exists():
            self.initialize()

        existing = self._read_all()
        existing.append(handoff)
        self._rewrite(existing)
        try:
            from drl_autoresearch.core.agent_contract import audit_event

            audit_event(
                "handoff_record",
                {
                    "handoff_id": handoff.handoff_id,
                    "from_agent": handoff.from_agent,
                    "to_agent": handoff.to_agent,
                },
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_latest(self) -> Optional[HandoffRecord]:
        all_handoffs = self._read_all()
        return all_handoffs[-1] if all_handoffs else None

    def get_all(self) -> list[HandoffRecord]:
        return self._read_all()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _read_all(self) -> list[HandoffRecord]:
        """
        Parse handoffs from the markdown file using embedded JSON comment blocks.
        """
        if not self.handoffs_path.exists():
            return []
        text = self.handoffs_path.read_text(encoding="utf-8")
        handoffs = []
        pattern = re.compile(r"<!-- HANDOFF_JSON: (\{.*?\}) -->", re.DOTALL)
        for match in pattern.finditer(text):
            try:
                data = json.loads(match.group(1))
                handoffs.append(HandoffRecord(**data))
            except Exception:
                pass
        return handoffs

    def _rewrite(self, handoffs: list[HandoffRecord]) -> None:
        """Atomically rewrite the full handoffs file."""
        lines = [f"# Agent Handoffs — {self._project_name}\n\n"]
        for n, handoff in enumerate(handoffs, start=1):
            lines.append("---\n\n")
            lines.append(handoff.to_markdown_section(n))
            lines.append(f"\n<!-- HANDOFF_JSON: {json.dumps(handoff.__dict__)} -->\n\n")
        lines.append("---\n")

        content = "".join(lines)
        tmp = self.handoffs_path.with_suffix(".md.tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(self.handoffs_path)
