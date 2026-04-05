from __future__ import annotations

from pathlib import Path

from drl_autoresearch.cli import console

from .metrics import MetricsCollector
from .server import DashboardServer


def run(project_dir: Path, port: int = 8765, clear_offline: bool = False) -> int:
    """Start the dashboard server or clear its offline backend cache."""
    project_dir = Path(project_dir)
    if clear_offline:
        MetricsCollector(project_dir=project_dir).clear_offline_backend()
        console("Dashboard offline backend cleared. Logs were preserved.", "success")
        return 0
    server = DashboardServer(project_dir=Path(project_dir), port=port)
    server.start()
    return 0
