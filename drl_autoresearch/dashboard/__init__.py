from __future__ import annotations

from pathlib import Path

from .server import DashboardServer


def run(project_dir: Path, port: int = 8765) -> int:
    """Start the dashboard server (blocking). Returns exit code."""
    server = DashboardServer(project_dir=Path(project_dir), port=port)
    server.start()
    return 0
