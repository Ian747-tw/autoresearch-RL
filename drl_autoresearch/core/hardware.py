"""
hardware.py — Detect and report hardware information for DRL AutoResearch.

Config is saved to .drl_autoresearch/hardware.yaml (fallback: hardware.json)
inside the target project directory.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class GPUInfo:
    index: int
    name: str
    vram_gb: float
    compute_capability: Optional[str]


@dataclass
class HardwareInfo:
    cpu_model: str
    cpu_cores: int
    cpu_threads: int
    ram_gb: float
    has_gpu: bool
    gpu_count: int
    gpus: list[GPUInfo]
    cuda_available: bool
    cuda_version: Optional[str]
    multi_gpu_allowed: bool  # Set by user during onboarding


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CONFIG_DIR = ".drl_autoresearch"
HARDWARE_YAML = "hardware.yaml"
HARDWARE_JSON = "hardware.json"


def _try_yaml_dump(data: Any) -> Optional[str]:
    """Attempt to serialise *data* with PyYAML; return None if unavailable."""
    try:
        import yaml  # type: ignore

        return yaml.dump(data, default_flow_style=False, allow_unicode=True)
    except ImportError:
        return None


def _try_yaml_load(text: str) -> Optional[Any]:
    """Attempt to deserialise YAML; return None if PyYAML unavailable."""
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text)
    except ImportError:
        return None


def _gpu_info_to_dict(gpu: GPUInfo) -> dict:
    return {
        "index": gpu.index,
        "name": gpu.name,
        "vram_gb": gpu.vram_gb,
        "compute_capability": gpu.compute_capability,
    }


def _gpu_info_from_dict(d: dict) -> GPUInfo:
    return GPUInfo(
        index=int(d["index"]),
        name=str(d["name"]),
        vram_gb=float(d["vram_gb"]),
        compute_capability=d.get("compute_capability"),
    )


def _hardware_to_dict(info: HardwareInfo) -> dict:
    return {
        "cpu_model": info.cpu_model,
        "cpu_cores": info.cpu_cores,
        "cpu_threads": info.cpu_threads,
        "ram_gb": info.ram_gb,
        "has_gpu": info.has_gpu,
        "gpu_count": info.gpu_count,
        "gpus": [_gpu_info_to_dict(g) for g in info.gpus],
        "cuda_available": info.cuda_available,
        "cuda_version": info.cuda_version,
        "multi_gpu_allowed": info.multi_gpu_allowed,
    }


def _hardware_from_dict(d: dict) -> HardwareInfo:
    gpus = [_gpu_info_from_dict(g) for g in d.get("gpus", [])]
    return HardwareInfo(
        cpu_model=str(d.get("cpu_model", "")),
        cpu_cores=int(d.get("cpu_cores", 0)),
        cpu_threads=int(d.get("cpu_threads", 0)),
        ram_gb=float(d.get("ram_gb", 0.0)),
        has_gpu=bool(d.get("has_gpu", False)),
        gpu_count=int(d.get("gpu_count", 0)),
        gpus=gpus,
        cuda_available=bool(d.get("cuda_available", False)),
        cuda_version=d.get("cuda_version"),
        multi_gpu_allowed=bool(d.get("multi_gpu_allowed", False)),
    )


# ---------------------------------------------------------------------------
# HardwareDetector
# ---------------------------------------------------------------------------


class HardwareDetector:
    """Detects hardware information from the running host."""

    # ------------------------------------------------------------------
    # CPU
    # ------------------------------------------------------------------

    def detect_cpu(self) -> tuple[str, int, int]:
        """Return (model_name, physical_cores, logical_threads)."""
        model = self._read_cpu_model()
        cores, threads = self._count_cpu_cores()
        return model, cores, threads

    def _read_cpu_model(self) -> str:
        # Linux: parse /proc/cpuinfo
        proc_cpuinfo = Path("/proc/cpuinfo")
        if proc_cpuinfo.exists():
            try:
                text = proc_cpuinfo.read_text(encoding="utf-8", errors="replace")
                for line in text.splitlines():
                    if line.startswith("model name"):
                        _, _, value = line.partition(":")
                        cleaned = value.strip()
                        if cleaned:
                            return cleaned
            except OSError:
                pass

        # macOS / Windows fallback
        model = platform.processor().strip()
        if model:
            return model

        # Final fallback
        return platform.machine() or "Unknown CPU"

    def _count_cpu_cores(self) -> tuple[int, int]:
        """Return (physical_cores, logical_threads)."""
        threads = os.cpu_count() or 1

        # Try psutil first for accurate physical core count
        try:
            import psutil  # type: ignore

            physical = psutil.cpu_count(logical=False) or threads
            return physical, threads
        except ImportError:
            pass

        # Linux: count unique physical id + core id pairs in /proc/cpuinfo
        proc_cpuinfo = Path("/proc/cpuinfo")
        if proc_cpuinfo.exists():
            try:
                text = proc_cpuinfo.read_text(encoding="utf-8", errors="replace")
                cores_set: set[tuple[str, str]] = set()
                current_physical = "0"
                current_core = "0"
                for line in text.splitlines():
                    if line.startswith("physical id"):
                        _, _, v = line.partition(":")
                        current_physical = v.strip()
                    elif line.startswith("core id"):
                        _, _, v = line.partition(":")
                        current_core = v.strip()
                        cores_set.add((current_physical, current_core))
                if cores_set:
                    return len(cores_set), threads
            except OSError:
                pass

        # Best-effort: assume HT/SMT doubles physical core count
        physical = max(1, threads // 2)
        return physical, threads

    # ------------------------------------------------------------------
    # RAM
    # ------------------------------------------------------------------

    def detect_ram(self) -> float:
        """Return total RAM in GB."""
        # Try psutil
        try:
            import psutil  # type: ignore

            return round(psutil.virtual_memory().total / (1024 ** 3), 2)
        except ImportError:
            pass

        # Linux: /proc/meminfo
        proc_meminfo = Path("/proc/meminfo")
        if proc_meminfo.exists():
            try:
                text = proc_meminfo.read_text(encoding="utf-8", errors="replace")
                for line in text.splitlines():
                    if line.startswith("MemTotal:"):
                        parts = line.split()
                        # Value is in kB
                        kb = int(parts[1])
                        return round(kb / (1024 ** 2), 2)
            except (OSError, ValueError, IndexError):
                pass

        # macOS: sysctl
        if platform.system() == "Darwin":
            try:
                result = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    return round(int(result.stdout.strip()) / (1024 ** 3), 2)
            except (subprocess.SubprocessError, ValueError, OSError):
                pass

        return 0.0

    # ------------------------------------------------------------------
    # GPUs
    # ------------------------------------------------------------------

    def detect_gpus(self) -> list[GPUInfo]:
        """Try torch, then nvidia-smi, then return empty list."""
        gpus = self._detect_gpus_via_torch()
        if gpus:
            return gpus

        gpus = self._detect_gpus_via_nvidia_smi()
        if gpus:
            return gpus

        return []

    def _detect_gpus_via_torch(self) -> list[GPUInfo]:
        try:
            import torch  # type: ignore

            if not torch.cuda.is_available():
                return []

            gpus: list[GPUInfo] = []
            n = torch.cuda.device_count()
            for i in range(n):
                props = torch.cuda.get_device_properties(i)
                vram_gb = round(props.total_memory / (1024 ** 3), 2)
                cc = f"{props.major}.{props.minor}"
                gpus.append(
                    GPUInfo(
                        index=i,
                        name=props.name,
                        vram_gb=vram_gb,
                        compute_capability=cc,
                    )
                )
            return gpus
        except Exception:
            return []

    def _detect_gpus_via_nvidia_smi(self) -> list[GPUInfo]:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,compute_cap",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                # Try without compute_cap (older nvidia-smi versions)
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=name,memory.total",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
            if result.returncode != 0:
                return []

            gpus: list[GPUInfo] = []
            for i, line in enumerate(result.stdout.strip().splitlines()):
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]
                name = parts[0] if len(parts) > 0 else f"GPU {i}"
                try:
                    # Memory is in MiB
                    vram_gb = round(float(parts[1]) / 1024, 2) if len(parts) > 1 else 0.0
                except ValueError:
                    vram_gb = 0.0
                cc = parts[2] if len(parts) > 2 else None
                gpus.append(
                    GPUInfo(
                        index=i,
                        name=name,
                        vram_gb=vram_gb,
                        compute_capability=cc,
                    )
                )
            return gpus
        except (FileNotFoundError, subprocess.SubprocessError, OSError):
            return []

    # ------------------------------------------------------------------
    # CUDA version
    # ------------------------------------------------------------------

    def _detect_cuda_version(self) -> Optional[str]:
        # From torch
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                v = torch.version.cuda
                if v:
                    return str(v)
        except Exception:
            pass

        # From nvcc
        try:
            result = subprocess.run(
                ["nvcc", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if "release" in line.lower():
                        # e.g. "Cuda compilation tools, release 11.8, V11.8.89"
                        parts = line.split("release")
                        if len(parts) > 1:
                            version_part = parts[1].split(",")[0].strip()
                            return version_part
        except (FileNotFoundError, subprocess.SubprocessError, OSError):
            pass

        # From nvidia-smi
        try:
            result = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if "CUDA Version" in line:
                        parts = line.split("CUDA Version:")
                        if len(parts) > 1:
                            return parts[1].strip().split()[0]
        except (FileNotFoundError, subprocess.SubprocessError, OSError):
            pass

        return None

    # ------------------------------------------------------------------
    # Full detection
    # ------------------------------------------------------------------

    def detect(self) -> HardwareInfo:
        """Auto-detect all hardware. multi_gpu_allowed defaults to False."""
        cpu_model, cpu_cores, cpu_threads = self.detect_cpu()
        ram_gb = self.detect_ram()
        gpus = self.detect_gpus()

        cuda_version: Optional[str] = None
        cuda_available = False

        if gpus:
            cuda_version = self._detect_cuda_version()
            cuda_available = cuda_version is not None or self._probe_cuda_available()

        return HardwareInfo(
            cpu_model=cpu_model,
            cpu_cores=cpu_cores,
            cpu_threads=cpu_threads,
            ram_gb=ram_gb,
            has_gpu=len(gpus) > 0,
            gpu_count=len(gpus),
            gpus=gpus,
            cuda_available=cuda_available,
            cuda_version=cuda_version,
            multi_gpu_allowed=False,  # Set by user during onboarding
        )

    def _probe_cuda_available(self) -> bool:
        """Secondary check for CUDA availability without torch."""
        try:
            import torch  # type: ignore

            return torch.cuda.is_available()
        except ImportError:
            pass
        # If nvidia-smi returned GPUs, assume CUDA is available
        return False

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, project_dir: Path, info: HardwareInfo) -> None:
        """Save HardwareInfo to .drl_autoresearch/hardware.yaml (or .json)."""
        config_dir = Path(project_dir) / CONFIG_DIR
        config_dir.mkdir(parents=True, exist_ok=True)

        data = _hardware_to_dict(info)

        # Try YAML first
        yaml_text = _try_yaml_dump(data)
        if yaml_text is not None:
            dest = config_dir / HARDWARE_YAML
            _atomic_write(dest, yaml_text)
            return

        # Fallback to JSON
        dest = config_dir / HARDWARE_JSON
        _atomic_write(dest, json.dumps(data, indent=2, ensure_ascii=False))

    def load(self, project_dir: Path) -> Optional[HardwareInfo]:
        """Load HardwareInfo from disk. Returns None if not found."""
        config_dir = Path(project_dir) / CONFIG_DIR

        # Try YAML
        yaml_path = config_dir / HARDWARE_YAML
        if yaml_path.exists():
            try:
                text = yaml_path.read_text(encoding="utf-8")
                data = _try_yaml_load(text)
                if data is not None:
                    return _hardware_from_dict(data)
            except (OSError, KeyError, ValueError):
                pass

        # Try JSON
        json_path = config_dir / HARDWARE_JSON
        if json_path.exists():
            try:
                text = json_path.read_text(encoding="utf-8")
                data = json.loads(text)
                return _hardware_from_dict(data)
            except (OSError, json.JSONDecodeError, KeyError, ValueError):
                pass

        return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _atomic_write(path: Path, text: str) -> None:
    """Write *text* to *path* atomically using a temp file + rename."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".hw_tmp_", suffix=path.suffix)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
