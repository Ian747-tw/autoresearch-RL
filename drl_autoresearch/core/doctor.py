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
import os
import shutil
import subprocess
import sys
import venv
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


class EnvFixResult(NamedTuple):
    ok: bool
    steps: list[str]


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


_DEFAULT_REQUIRED_PACKAGES = ["torch", "numpy", "pandas"]


def _load_yaml_or_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except ModuleNotFoundError:
        pass
    except Exception:
        pass

    if path.suffix == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}


def _load_env_preferences(project_dir: Path) -> dict:
    config_dir = project_dir / ".drl_autoresearch"
    # onboarding.{yaml,json} has package manager + venv + create_new_env choices
    onboarding = _load_yaml_or_json(config_dir / "onboarding.yaml")
    if not onboarding:
        onboarding = _load_yaml_or_json(config_dir / "onboarding.json")
    onboarding_py = onboarding.get("python_env", {}) if isinstance(onboarding, dict) else {}
    if not isinstance(onboarding_py, dict):
        onboarding_py = {}

    py_cfg = _load_yaml_or_json(config_dir / "python_env.yaml")
    if not py_cfg:
        py_cfg = _load_yaml_or_json(config_dir / "python_env.json")
    if not isinstance(py_cfg, dict):
        py_cfg = {}

    required = py_cfg.get("required_packages")
    if not isinstance(required, list):
        required = _DEFAULT_REQUIRED_PACKAGES
    required = [str(p).strip() for p in required if str(p).strip()]
    if not required:
        required = _DEFAULT_REQUIRED_PACKAGES

    package_manager = onboarding_py.get("package_manager", "pip")
    venv_path = onboarding_py.get("venv_path")
    create_new_env = onboarding_py.get("create_new_env", "auto")

    if not isinstance(package_manager, str):
        package_manager = "pip"
    if venv_path is not None and not isinstance(venv_path, str):
        venv_path = None
    if not isinstance(create_new_env, str):
        create_new_env = "auto"

    return {
        "package_manager": package_manager,
        "venv_path": venv_path,
        "create_new_env": create_new_env,
        "required_packages": required,
    }


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _resolve_target_python(project_dir: Path, prefs: dict, steps: list[str]) -> Path:
    venv_path = prefs.get("venv_path")
    create_new_env = prefs.get("create_new_env", "auto")

    candidate: Path | None = None
    if isinstance(venv_path, str) and venv_path.strip():
        candidate = Path(venv_path)
        if not candidate.is_absolute():
            candidate = (project_dir / candidate).resolve()
    else:
        candidate = (project_dir / ".venv").resolve()

    py = _venv_python(candidate)
    venv_exists = py.exists()

    should_create = (create_new_env == "yes") or (
        create_new_env == "auto" and not venv_exists
    )
    if should_create and not venv_exists:
        steps.append(f"Creating virtual environment at {candidate}")
        candidate.parent.mkdir(parents=True, exist_ok=True)
        venv.EnvBuilder(with_pip=True).create(candidate)
        py = _venv_python(candidate)
        venv_exists = py.exists()

    if venv_exists:
        steps.append(f"Using environment python: {py}")
        return py

    steps.append("Using current interpreter (no project venv available).")
    return Path(sys.executable)


def _install_packages_with_python(
    python_exec: Path,
    package_manager: str,
    packages: list[str],
    steps: list[str],
) -> None:
    if not packages:
        return

    if package_manager == "uv" and shutil.which("uv"):
        cmd = ["uv", "pip", "install", "--python", str(python_exec), *packages]
        steps.append(f"Installing packages via uv: {' '.join(packages)}")
        completed = subprocess.run(cmd, capture_output=True, text=True)
        if completed.returncode == 0:
            return
        steps.append("uv install failed, falling back to pip.")

    steps.append(f"Installing packages via pip: {' '.join(packages)}")
    ensure = subprocess.run(
        [str(python_exec), "-m", "ensurepip", "--upgrade"],
        capture_output=True,
        text=True,
    )
    _ = ensure  # best effort
    completed = subprocess.run(
        [str(python_exec), "-m", "pip", "install", *packages],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "").strip().splitlines()[-1:]
        reason = tail[0] if tail else "unknown error"
        steps.append(f"Package installation failed: {reason}")
        raise RuntimeError(reason)


def _missing_packages(python_exec: Path, packages: list[str]) -> list[str]:
    missing: list[str] = []
    for pkg in packages:
        code = (
            "import importlib.util,sys;"
            f"sys.exit(0 if importlib.util.find_spec('{pkg}') is not None else 1)"
        )
        rc = subprocess.run(
            [str(python_exec), "-c", code],
            capture_output=True,
            text=True,
        ).returncode
        if rc != 0:
            missing.append(pkg)
    return missing


def fix_environment(project_dir: Path) -> EnvFixResult:
    """Best-effort environment remediation based on onboarding preferences."""
    project_dir = Path(project_dir).resolve()
    steps: list[str] = []

    try:
        prefs = _load_env_preferences(project_dir)
        package_manager = str(prefs.get("package_manager", "pip")).lower()
        required_packages = prefs.get("required_packages", _DEFAULT_REQUIRED_PACKAGES)
        if not isinstance(required_packages, list):
            required_packages = _DEFAULT_REQUIRED_PACKAGES
        required_packages = [str(p).strip() for p in required_packages if str(p).strip()]

        py_exec = _resolve_target_python(project_dir, prefs, steps)
        missing = _missing_packages(py_exec, required_packages)
        if not missing:
            steps.append("All required packages already available.")
            steps.append("Environment remediation completed.")
            return EnvFixResult(True, steps)

        try:
            _install_packages_with_python(
                python_exec=py_exec,
                package_manager=package_manager,
                packages=missing,
                steps=steps,
            )
        except RuntimeError as exc:
            err = str(exc)
            # System-managed Python (PEP 668): recover by creating local .venv.
            if "PEP 668" in err and py_exec == Path(sys.executable):
                fallback_dir = (project_dir / ".venv").resolve()
                steps.append(
                    "Detected externally-managed system Python; creating project .venv fallback."
                )
                venv.EnvBuilder(with_pip=True).create(fallback_dir)
                fallback_py = _venv_python(fallback_dir)
                _install_packages_with_python(
                    python_exec=fallback_py,
                    package_manager="pip",
                    packages=missing,
                    steps=steps,
                )
            else:
                raise
        steps.append("Environment remediation completed.")
        return EnvFixResult(True, steps)
    except Exception as exc:  # noqa: BLE001
        steps.append(f"Environment remediation failed: {exc}")
        return EnvFixResult(False, steps)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(project_dir: Path, fix: bool = False) -> int:
    """Execute all doctor checks and print results.

    Returns 0 if all checks pass, 1 if any check fails.
    """
    project_dir = Path(project_dir).resolve()
    if fix:
        console("Doctor fix mode enabled — attempting environment remediation first.", "info")
        fix_result = fix_environment(project_dir)
        for step in fix_result.steps:
            console(step, "info" if fix_result.ok else "warning")

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
