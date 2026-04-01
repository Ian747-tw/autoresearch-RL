"""
Incident log — tracks serious failures that need human attention.
Backed by a human-readable markdown file.
"""

from __future__ import annotations

import json
import random
import string
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

INCIDENT_TYPES = [
    "reward_hacking",    # reward improves but real objective doesn't
    "invalid_eval",      # evaluation protocol violated
    "divergence",        # loss exploded or NaN
    "oom",               # GPU/CPU out of memory
    "crash_loop",        # repeated crashes on similar config
    "broken_assumption", # key assumption proved wrong
    "rule_violation",    # hard rule was violated or almost violated
    "data_leak",         # potential information leak in eval
    "reproducibility",   # results not reproducible across seeds
]

# Severity auto-assignment by type
_AUTO_SEVERITY: dict[str, str] = {
    "divergence": "critical",
    "oom": "critical",
    "crash_loop": "critical",
    "rule_violation": "critical",
    "reward_hacking": "high",
    "invalid_eval": "high",
    "data_leak": "high",
    "broken_assumption": "medium",
    "reproducibility": "medium",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _now_human() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _rand4() -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=4))


# ---------------------------------------------------------------------------
# Incident dataclass
# ---------------------------------------------------------------------------

@dataclass
class Incident:
    incident_id: str                 # inc_{timestamp}_{type[:4]}
    incident_type: str
    severity: str                    # "low" / "medium" / "high" / "critical"
    run_id: str
    timestamp: str
    description: str
    evidence: dict
    status: str                      # "open" / "resolved" / "wont_fix"
    resolution: Optional[str] = None

    def to_markdown_block(self, index: int) -> str:
        evidence_lines = "\n".join(f"- {k}: {v}" for k, v in self.evidence.items())
        resolution_line = (
            f"**Resolution**: {self.resolution}\n" if self.resolution else ""
        )
        return (
            f"### [INC-{index:03d}] {self.incident_type} — {self.severity} — {self.timestamp}\n"
            f"**ID**: {self.incident_id}  \n"
            f"**Run**: {self.run_id}  \n"
            f"**Status**: {self.status}  \n"
            f"**Description**: {self.description}  \n"
            f"**Evidence**:\n{evidence_lines}\n"
            f"{resolution_line}"
            "---\n"
        )


# ---------------------------------------------------------------------------
# IncidentLog
# ---------------------------------------------------------------------------

class IncidentLog:
    INCIDENTS_PATH = "logs/incidents.md"

    def __init__(self, project_dir: Path) -> None:
        self.project_dir = Path(project_dir)
        self.incidents_path = self.project_dir / self.INCIDENTS_PATH
        # In-memory list; always re-read from file for persistence
        self._project_name: str = project_dir.name

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        self.incidents_path.parent.mkdir(parents=True, exist_ok=True)
        if self.incidents_path.exists():
            return
        content = (
            f"# Incident Log — {self._project_name}\n\n"
            "## Open Incidents\n\n"
            "## Resolved Incidents\n"
        )
        tmp = self.incidents_path.with_suffix(".md.tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(self.incidents_path)

    # ------------------------------------------------------------------
    # Report a new incident
    # ------------------------------------------------------------------

    def report(
        self,
        incident_type: str,
        run_id: str,
        description: str,
        evidence: dict,
        severity: str = "medium",
    ) -> str:
        """
        Create and persist a new incident. Returns the incident_id.
        Severity is auto-upgraded based on type if the provided severity
        is lower than the type's default.
        """
        if not self.incidents_path.exists():
            self.initialize()

        # Auto-severity logic
        type_severity = _AUTO_SEVERITY.get(incident_type, "low")
        _order = ["low", "medium", "high", "critical"]
        effective_severity = (
            type_severity
            if _order.index(type_severity) > _order.index(severity)
            else severity
        )

        ts = _now_iso()
        incident_id = f"inc_{ts}_{incident_type[:4]}"
        incident = Incident(
            incident_id=incident_id,
            incident_type=incident_type,
            severity=effective_severity,
            run_id=run_id,
            timestamp=_now_human(),
            description=description,
            evidence=evidence,
            status="open",
            resolution=None,
        )

        # Read all incidents, add new one, rewrite atomically
        all_incidents = self._read_all()
        all_incidents.append(incident)
        self._rewrite(all_incidents)
        return incident_id

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_open_incidents(self) -> list[Incident]:
        return [i for i in self._read_all() if i.status == "open"]

    def get_by_type(self, incident_type: str) -> list[Incident]:
        return [i for i in self._read_all() if i.incident_type == incident_type]

    def count_recent(self, hours: int = 24) -> int:
        """Count incidents created in the last `hours` hours."""
        now = datetime.now(timezone.utc)
        count = 0
        for incident in self._read_all():
            try:
                ts = datetime.strptime(incident.timestamp, "%Y-%m-%d %H:%M UTC").replace(
                    tzinfo=timezone.utc
                )
                diff = (now - ts).total_seconds() / 3600.0
                if diff <= hours:
                    count += 1
            except ValueError:
                pass
        return count

    # ------------------------------------------------------------------
    # Resolve
    # ------------------------------------------------------------------

    def resolve(self, incident_id: str, resolution: str) -> None:
        all_incidents = self._read_all()
        found = False
        for incident in all_incidents:
            if incident.incident_id == incident_id:
                incident.status = "resolved"
                incident.resolution = resolution
                found = True
                break
        if not found:
            raise KeyError(f"incident_id {incident_id!r} not found")
        self._rewrite(all_incidents)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _read_all(self) -> list[Incident]:
        """
        Parse incidents from the markdown file.
        We store structured data as JSON comment blocks embedded in the markdown.
        """
        if not self.incidents_path.exists():
            return []
        text = self.incidents_path.read_text(encoding="utf-8")
        incidents = []
        # Extract JSON blocks: <!-- INCIDENT_JSON: {...} -->
        import re
        pattern = re.compile(r"<!-- INCIDENT_JSON: (\{.*?\}) -->", re.DOTALL)
        for match in pattern.finditer(text):
            try:
                data = json.loads(match.group(1))
                incidents.append(Incident(**data))
            except Exception:
                pass
        return incidents

    def _rewrite(self, incidents: list[Incident]) -> None:
        """Atomically rewrite the markdown file with all incidents."""
        open_incidents = [i for i in incidents if i.status == "open"]
        resolved_incidents = [i for i in incidents if i.status != "open"]

        lines = [f"# Incident Log — {self._project_name}\n\n"]

        lines.append("## Open Incidents\n\n")
        for idx, inc in enumerate(open_incidents, start=1):
            lines.append(inc.to_markdown_block(idx))
            lines.append(f"<!-- INCIDENT_JSON: {json.dumps(inc.__dict__)} -->\n\n")

        lines.append("## Resolved Incidents\n\n")
        for idx, inc in enumerate(resolved_incidents, start=1):
            lines.append(inc.to_markdown_block(len(open_incidents) + idx))
            lines.append(f"<!-- INCIDENT_JSON: {json.dumps(inc.__dict__)} -->\n\n")

        content = "".join(lines)
        tmp = self.incidents_path.with_suffix(".md.tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(self.incidents_path)
