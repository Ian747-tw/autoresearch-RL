"""
drl_autoresearch.core.doctor
-----------------------------
Environment, hardware, and config health checks.

Exit codes
----------
0  — all checks passed
1  — one or more checks failed (errors present)
"""

from __future__ import annotations

import importlib.util
import json
import platform
import sys
from pathlib import Path
from typing import Callable, List, NamedTuple

from drl_autoresearch.cli import console


# ---------------------------------------------------------------------------
# Check result type
# ---------------------------------------------------------------------------

class CheckResult(NamedTuple):
    name: str
    ok: bool
    message: str


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _check_python_version(project_dir: Path) -> CheckResult:
    """Require Python >= 3.10."""
    vi = sys.version_info
    ok = (vi.major, vi.minor) >= (3, 10)
    msg = f"Python {vi.major}.{vi.minor}.{vi.micro}"
    if not ok:
        msg += "  (need >= 3.10)"
    return CheckResult("Python version", ok, msg)


def _check_torch(project_dir: Path) -> CheckResult:
    """Verify PyTorch is importable."""
    spec = importlib.util.find_spec("torch")
    if spec is None:
        return CheckResult("PyTorch", False, "torch not found — install it.")
    import torch  # noqa: PLC0415
    return CheckResult("PyTorch", True, f"torch {torch.__version__}")


def _check_cuda(project_dir: Path) -> CheckResult:
    """Report CUDA / GPU availability (informational, not a hard failure)."""
    spec = importlib.util.find_spec("torch")
    if spec is None:
        return CheckResult("CUDA / GPU", False, "torch not importable; skipping GPU check.")
    import torch  # noqa: PLC0415
    if torch.cuda.is_available():
        device_names = [
            torch.cuda.get_device_name(i)
            for i in range(torch.cuda.device_count())
        ]
        msg = f"CUDA {torch.version.cuda} — {', '.join(device_names)}"
        return CheckResult("CUDA / GPU", True, msg)
    # MPS (Apple Silicon) as fallback
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return CheckResult("CUDA / GPU", True, "MPS (Apple Silicon) available.")
    return CheckResult(
        "CUDA / GPU",
        True,  # not a hard failure — CPU training is valid
        "No GPU found — will run on CPU (slow for large models).",
    )


def _check_config_dir(project_dir: Path) -> CheckResult:
    """Verify .drl_autoresearch/ directory exists."""
    config_dir = project_dir / ".drl_autoresearch"
    ok = config_dir.is_dir()
    msg = str(config_dir) if ok else f"{config_dir} missing — run `drl-autoresearch init`."
    return CheckResult(".drl_autoresearch/ directory", ok, msg)


def _make_yaml_checker(filename: str) -> Callable[[Path], CheckResult]:
    """Return a checker that verifies a YAML config file is present and parseable."""

    def _check(project_dir: Path) -> CheckResult:
        path = project_dir / ".drl_autoresearch" / filename
        if not path.exists():
            return CheckResult(
                filename,
                False,
                f"{path} not found — run `drl-autoresearch init`.",
            )
        # Try to parse with PyYAML if available, otherwise just confirm it's readable.
        try:
            import yaml  # noqa: PLC0415
            with path.open("r", encoding="utf-8") as fh:
                yaml.safe_load(fh)
            return CheckResult(filename, True, f"{filename} is valid YAML.")
        except ModuleNotFoundError:
            # yaml not available — just check the file is non-empty text
            try:
                text = path.read_text(encoding="utf-8").strip()
                if not text:
                    return CheckResult(filename, False, f"{filename} is empty.")
                return CheckResult(filename, True, f"{filename} present (PyYAML not installed; skipped schema check).")
            except OSError as exc:
                return CheckResult(filename, False, str(exc))
        except Exception as exc:  # noqa: BLE001
            return CheckResult(filename, False, f"{filename} parse error: {exc}")

    _check.__name__ = f"_check_{filename}"
    return _check


def _check_state_json(project_dir: Path) -> CheckResult:
    """Verify state.json is present and valid JSON."""
    path = project_dir / ".drl_autoresearch" / "state.json"
    if not path.exists():
        return CheckResult(
            "state.json",
            False,
            "state.json missing — run `drl-autoresearch init`.",
        )
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        phase = data.get("current_phase", "unknown")
        return CheckResult("state.json", True, f"state.json valid (phase={phase!r}).")
    except json.JSONDecodeError as exc:
        return CheckResult("state.json", False, f"state.json corrupted: {exc}")
    except OSError as exc:
        return CheckResult("state.json", False, str(exc))


def _check_experiment_registry(project_dir: Path) -> CheckResult:
    """Verify logs/experiment_registry.tsv exists."""
    path = project_dir / "logs" / "experiment_registry.tsv"
    if not path.exists():
        return CheckResult(
            "experiment_registry.tsv",
            False,
            "logs/experiment_registry.tsv missing — run `drl-autoresearch init`.",
        )
    return CheckResult("experiment_registry.tsv", True, "logs/experiment_registry.tsv present.")


def _check_hard_rules(project_dir: Path) -> CheckResult:
    """Verify NON_NEGOTIABLE_RULES.md exists."""
    path = project_dir / "NON_NEGOTIABLE_RULES.md"
    if not path.exists():
        return CheckResult(
            "NON_NEGOTIABLE_RULES.md",
            False,
            "NON_NEGOTIABLE_RULES.md missing — run `drl-autoresearch init`.",
        )
    return CheckResult("NON_NEGOTIABLE_RULES.md", True, "NON_NEGOTIABLE_RULES.md present.")


def _check_skills_dir(project_dir: Path) -> CheckResult:
    """Verify skills/ directory exists."""
    path = project_dir / "skills"
    ok = path.is_dir()
    msg = "skills/ directory present." if ok else "skills/ missing — run `drl-autoresearch init`."
    return CheckResult("skills/ directory", ok, msg)


def _check_numpy(project_dir: Path) -> CheckResult:
    """Verify numpy is importable."""
    spec = importlib.util.find_spec("numpy")
    if spec is None:
        return CheckResult("NumPy", False, "numpy not found.")
    import numpy as np  # noqa: PLC0415
    return CheckResult("NumPy", True, f"numpy {np.__version__}")


def _check_pandas(project_dir: Path) -> CheckResult:
    """Verify pandas is importable."""
    spec = importlib.util.find_spec("pandas")
    if spec is None:
        return CheckResult("pandas", False, "pandas not found.")
    import pandas as pd  # noqa: PLC0415
    return CheckResult("pandas", True, f"pandas {pd.__version__}")


# ---------------------------------------------------------------------------
# Check registry
# ---------------------------------------------------------------------------

_ALL_CHECKS: List[Callable[[Path], CheckResult]] = [
    _check_python_version,
    _check_torch,
    _check_cuda,
    _check_numpy,
    _check_pandas,
    _check_config_dir,
    _make_yaml_checker("policy.yaml"),
    _make_yaml_checker("hardware.yaml"),
    _make_yaml_checker("python_env.yaml"),
    _make_yaml_checker("permissions.yaml"),
    _check_state_json,
    _check_experiment_registry,
    _check_hard_rules,
    _check_skills_dir,
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(project_dir: Path) -> int:
    """Execute all doctor checks and print results.

    Returns 0 if all checks pass, 1 if any check fails.
    """
    project_dir = Path(project_dir).resolve()
    console(f"Running doctor checks for: {project_dir}", "info")
    print()

    results: List[CheckResult] = []
    for check_fn in _ALL_CHECKS:
        result = check_fn(project_dir)
        results.append(result)
        level = "success" if result.ok else "error"
        console(f"{result.name}: {result.message}", level)

    print()
    passed = sum(1 for r in results if r.ok)
    total  = len(results)
    failed = total - passed

    if failed == 0:
        console(f"All {total} checks passed.", "success")
        return 0
    else:
        console(
            f"{passed}/{total} checks passed — {failed} issue(s) found. "
            "Run `drl-autoresearch init` to fix missing files.",
            "error",
        )
        return 1
