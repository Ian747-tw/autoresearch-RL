"""
PolicyEngine — enforces hard rules and permission modes for DRL AutoResearch.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Optional yaml import with json fallback
# ---------------------------------------------------------------------------
try:
    import yaml  # type: ignore

    _YAML_AVAILABLE = True
except ImportError:
    import json as _json_fallback  # noqa: F401

    _YAML_AVAILABLE = False


CONFIG_DIR = ".drl_autoresearch"
PERMISSIONS_FILE = "permissions.yaml"
AUDIT_LOG_FILE = "policy_audit.log"
HARD_RULES_FILE = "NON_NEGOTIABLE_RULES.md"

# ---------------------------------------------------------------------------
# Permission modes
# ---------------------------------------------------------------------------
PERMISSION_MODES = {
    "locked",
    "prompted",
    "bootstrap-only",
    "open",
    "project-only",
}

# ---------------------------------------------------------------------------
# Action types
# ---------------------------------------------------------------------------
ACTION_TYPES = {
    "edit_reward",
    "edit_eval",
    "edit_env",
    "install_package",
    "update_package",
    "global_install",
    "exceed_compute",
    "gpu_memory_risk",
    "silent_cpu_fallback",
    "eval_protocol_change",
    "use_privileged_info",
    "custom",
}

# Actions that are always blocked regardless of mode
_ALWAYS_BLOCKED = {
    "global_install",
    "use_privileged_info",
}

# Actions that require confirmation in non-open modes
_CONFIRMATION_REQUIRED = {
    "edit_reward",
    "edit_eval",
    "edit_env",
    "install_package",
    "update_package",
    "eval_protocol_change",
    "exceed_compute",
    "gpu_memory_risk",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_yaml_or_json(path: Path) -> Any:
    """Load a YAML file, falling back to JSON parsing if PyYAML is unavailable."""
    text = path.read_text(encoding="utf-8")
    if _YAML_AVAILABLE:
        return yaml.safe_load(text)
    # Attempt JSON parse as fallback (works for simple YAML that is valid JSON)
    import json

    return json.loads(text)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PolicyDecision:
    allowed: bool
    requires_confirmation: bool
    reason: str
    violated_rules: List[str] = field(default_factory=list)
    mode: str = "locked"


# ---------------------------------------------------------------------------
# PolicyEngine
# ---------------------------------------------------------------------------


class PolicyEngine:
    """Loads hard rules and permission config; checks every action against them."""

    def __init__(self, project_dir: Path) -> None:
        self.project_dir = Path(project_dir)
        self._mode: str = "locked"
        self._hard_rules: List[str] = []
        self._permissions_config: Dict[str, Any] = {}
        # Per-action-type overrides loaded from permissions.yaml
        self._action_overrides: Dict[str, str] = {}

        self._audit_log_path = self.project_dir / CONFIG_DIR / AUDIT_LOG_FILE

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load hard rules from NON_NEGOTIABLE_RULES.md and permissions from yaml."""
        self._hard_rules = self._parse_hard_rules()
        self._load_permissions()

    def _parse_hard_rules(self) -> List[str]:
        """Parse bullet/numbered list items from NON_NEGOTIABLE_RULES.md."""
        rules_path = self.project_dir / HARD_RULES_FILE
        if not rules_path.exists():
            return []

        rules: List[str] = []
        text = rules_path.read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            # Match markdown bullets (-, *, +) or numbered lists (1., 2. …)
            m = re.match(r"^(?:[-*+]|\d+\.)\s+(.+)", stripped)
            if m:
                rules.append(m.group(1).strip())
        return rules

    def _load_permissions(self) -> None:
        """Load .drl_autoresearch/permissions.yaml (or .json fallback)."""
        perm_path = self.project_dir / CONFIG_DIR / PERMISSIONS_FILE
        if not perm_path.exists():
            # Try json extension if yaml not present
            json_path = perm_path.with_suffix(".json")
            if json_path.exists():
                import json

                self._permissions_config = json.loads(
                    json_path.read_text(encoding="utf-8")
                )
            else:
                self._permissions_config = {}
        else:
            self._permissions_config = _load_yaml_or_json(perm_path) or {}

        raw_mode = self._permissions_config.get("mode", "locked")
        self._mode = raw_mode if raw_mode in PERMISSION_MODES else "locked"

        # Per-action overrides: {action_type: "allow" | "deny" | "prompt"}
        self._action_overrides = self._permissions_config.get("action_overrides", {})

    # ------------------------------------------------------------------
    # Rule checking
    # ------------------------------------------------------------------

    def check(
        self,
        action_type: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> PolicyDecision:
        """
        Evaluate whether an action is permitted under current policy.

        Always checks hard rules first. Then applies permission-mode logic.
        Appends a record to the audit log.
        """
        details = details or {}
        violated_rules: List[str] = []

        # 1. Check hard rules — these always win
        for rule in self._hard_rules:
            if self._rule_matches_action(rule, action_type, details):
                violated_rules.append(rule)

        if violated_rules:
            decision = PolicyDecision(
                allowed=False,
                requires_confirmation=False,
                reason=f"Hard rule violation for action '{action_type}'",
                violated_rules=violated_rules,
                mode=self._mode,
            )
            self._audit(action_type, details, decision, "hard_rule")
            return decision

        # 2. Always-blocked action types (regardless of mode)
        if action_type in _ALWAYS_BLOCKED:
            decision = PolicyDecision(
                allowed=False,
                requires_confirmation=False,
                reason=f"Action type '{action_type}' is unconditionally blocked.",
                violated_rules=[],
                mode=self._mode,
            )
            self._audit(action_type, details, decision, "always_blocked")
            return decision

        # 3. Per-action overrides from permissions.yaml
        override = self._action_overrides.get(action_type)
        if override == "deny":
            decision = PolicyDecision(
                allowed=False,
                requires_confirmation=False,
                reason=f"Action '{action_type}' explicitly denied in permissions config.",
                violated_rules=[],
                mode=self._mode,
            )
            self._audit(action_type, details, decision, "config_override_deny")
            return decision
        if override == "allow":
            decision = PolicyDecision(
                allowed=True,
                requires_confirmation=False,
                reason=f"Action '{action_type}' explicitly allowed via config override.",
                violated_rules=[],
                mode=self._mode,
            )
            self._audit(action_type, details, decision, "config_override_allow")
            return decision

        # 4. Permission-mode logic
        decision = self._apply_mode(action_type, details)
        self._audit(action_type, details, decision, "mode_check")
        return decision

    def _rule_matches_action(
        self,
        rule: str,
        action_type: str,
        details: Dict[str, Any],
    ) -> bool:
        """
        Heuristic: a hard rule matches if the action type keyword or any detail
        value appears verbatim in the rule text (case-insensitive).
        """
        rule_lower = rule.lower()
        if action_type.replace("_", " ") in rule_lower or action_type in rule_lower:
            return True
        for val in details.values():
            if isinstance(val, str) and val.lower() in rule_lower:
                return True
        return False

    def _apply_mode(
        self, action_type: str, details: Dict[str, Any]
    ) -> PolicyDecision:
        """Return a PolicyDecision based purely on the current permission mode."""
        mode = self._mode

        if mode == "open":
            return PolicyDecision(
                allowed=True,
                requires_confirmation=False,
                reason=f"Mode 'open': all actions permitted.",
                mode=mode,
            )

        if mode == "locked":
            requires = action_type in _CONFIRMATION_REQUIRED
            if requires:
                return PolicyDecision(
                    allowed=False,
                    requires_confirmation=True,
                    reason=(
                        f"Mode 'locked': action '{action_type}' requires explicit "
                        "approval before proceeding."
                    ),
                    mode=mode,
                )
            return PolicyDecision(
                allowed=False,
                requires_confirmation=False,
                reason=f"Mode 'locked': action '{action_type}' is not permitted.",
                mode=mode,
            )

        if mode == "prompted":
            # Everything is allowed but gated behind a confirmation prompt
            return PolicyDecision(
                allowed=True,
                requires_confirmation=True,
                reason=(
                    f"Mode 'prompted': action '{action_type}' requires user "
                    "confirmation before execution."
                ),
                mode=mode,
            )

        if mode == "bootstrap-only":
            # Installs only allowed during init phase; other actions need confirmation
            is_install = action_type in ("install_package", "update_package")
            bootstrapping = details.get("phase") == "init"
            if is_install and bootstrapping:
                return PolicyDecision(
                    allowed=True,
                    requires_confirmation=False,
                    reason="Mode 'bootstrap-only': install permitted during init phase.",
                    mode=mode,
                )
            if is_install and not bootstrapping:
                return PolicyDecision(
                    allowed=False,
                    requires_confirmation=True,
                    reason=(
                        "Mode 'bootstrap-only': installs only permitted during init. "
                        "Explicit approval required."
                    ),
                    mode=mode,
                )
            # Non-install actions need confirmation
            return PolicyDecision(
                allowed=True,
                requires_confirmation=True,
                reason=(
                    f"Mode 'bootstrap-only': action '{action_type}' requires "
                    "confirmation."
                ),
                mode=mode,
            )

        if mode == "project-only":
            # Global installs are blocked; project-scoped ops allowed with confirmation
            if action_type == "global_install":
                return PolicyDecision(
                    allowed=False,
                    requires_confirmation=False,
                    reason="Mode 'project-only': global installs are forbidden.",
                    mode=mode,
                )
            requires = action_type in _CONFIRMATION_REQUIRED
            return PolicyDecision(
                allowed=True,
                requires_confirmation=requires,
                reason=(
                    f"Mode 'project-only': action '{action_type}' permitted within "
                    "project scope."
                    + (" Confirmation required." if requires else "")
                ),
                mode=mode,
            )

        # Unknown mode — default to locked
        return PolicyDecision(
            allowed=False,
            requires_confirmation=False,
            reason=f"Unknown permission mode '{mode}'; defaulting to locked.",
            mode=mode,
        )

    # ------------------------------------------------------------------
    # Audit log
    # ------------------------------------------------------------------

    def _audit(
        self,
        action_type: str,
        details: Dict[str, Any],
        decision: PolicyDecision,
        rule_matched: str,
    ) -> None:
        """Append a single line to the audit log atomically."""
        import json

        self._audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": _now_iso(),
            "action_type": action_type,
            "details": details,
            "decision": {
                "allowed": decision.allowed,
                "requires_confirmation": decision.requires_confirmation,
                "reason": decision.reason,
                "violated_rules": decision.violated_rules,
            },
            "rule_matched": rule_matched,
            "mode": decision.mode,
        }
        line = json.dumps(record, ensure_ascii=False) + "\n"

        # Append atomically: write to tmp then append-via-read trick isn't
        # truly atomic on all filesystems, but a named tmp + rename would
        # truncate previous content. We use a simple file lock via O_APPEND.
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        fd = os.open(str(self._audit_log_path), flags, 0o644)
        try:
            os.write(fd, line.encode("utf-8"))
        finally:
            os.close(fd)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_hard_rules(self) -> List[str]:
        """Return the parsed list of hard rules from NON_NEGOTIABLE_RULES.md."""
        return list(self._hard_rules)

    def format_violation(self, action_type: str, rule: str) -> str:
        """Return a human-readable violation message."""
        return (
            f"[POLICY VIOLATION] Action '{action_type}' violates hard rule:\n"
            f"  >> {rule}\n"
            f"This action is blocked unconditionally."
        )

    @property
    def mode(self) -> str:
        return self._mode

    def __repr__(self) -> str:
        return (
            f"PolicyEngine(project={self.project_dir.name!r}, "
            f"mode={self._mode!r}, "
            f"hard_rules={len(self._hard_rules)})"
        )
