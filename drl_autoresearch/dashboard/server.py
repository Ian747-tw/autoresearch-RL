"""
DashboardServer — stdlib-only HTTP server with SSE live updates.
"""

from __future__ import annotations

import http.server
import importlib.resources
import json
import socketserver
import threading
import time
import traceback
from pathlib import Path
from typing import Optional

from .metrics import MetricsCollector

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

_STATIC_DIR = Path(__file__).parent / "static"
_INDEX_HTML = _STATIC_DIR / "index.html"

_SSE_INTERVAL = 5  # seconds between SSE pushes

# ---------------------------------------------------------------------------
# Threaded HTTP server
# ---------------------------------------------------------------------------


class _ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """HTTP server that spawns a new thread for each connection."""

    daemon_threads = True

    def __init__(self, server_address, RequestHandlerClass, collector: MetricsCollector):
        self.collector = collector
        super().__init__(server_address, RequestHandlerClass)


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------


class _DashboardHandler(http.server.BaseHTTPRequestHandler):
    """Handle all dashboard HTTP requests."""

    # Suppress default request logging; dashboard uses its own minimal output.
    def log_message(self, fmt: str, *args) -> None:  # type: ignore[override]
        pass

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]  # strip query string

        if path == "/" or path == "/index.html":
            self._serve_index()
        elif path == "/api/data":
            self._serve_api_data()
        elif path == "/api/status":
            self._serve_api_status()
        elif path == "/events":
            self._serve_sse()
        elif path.startswith("/api/run/"):
            run_id = path[len("/api/run/"):]
            self._serve_run_detail(run_id)
        else:
            self._send_404()

    # ------------------------------------------------------------------
    # Route handlers
    # ------------------------------------------------------------------

    def _serve_index(self) -> None:
        body = _load_index_html()
        if body is None:
            self._send_error(500, "index.html not found")
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._send_no_cache_headers()
        self.end_headers()
        self.wfile.write(body)

    def _serve_api_data(self) -> None:
        try:
            data = self.server.collector.collect()  # type: ignore[attr-defined]
            body = json.dumps(data.to_dict(), ensure_ascii=False, default=str).encode("utf-8")
        except Exception:
            body = json.dumps({"error": traceback.format_exc()}).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._send_cors_headers()
        self._send_no_cache_headers()
        self.end_headers()
        self.wfile.write(body)

    def _serve_api_status(self) -> None:
        try:
            data = self.server.collector.collect().to_dict()  # type: ignore[attr-defined]
            payload = {
                "current_phase": data.get("current_phase", "research"),
                "best_metric_value": data.get("best_metric_value"),
                "best_metric_name": data.get("best_metric_name", "reward"),
                "total_runs": data.get("total_runs", 0),
                "kept_runs": data.get("kept_runs", 0),
                "discarded_runs": data.get("discarded_runs", 0),
                "crashed_runs": data.get("crashed_runs", 0),
                "ok": True,
            }
        except Exception:
            payload = {"ok": False, "error": traceback.format_exc()}

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._send_cors_headers()
        self._send_no_cache_headers()
        self.end_headers()
        self.wfile.write(body)

    def _serve_run_detail(self, run_id: str) -> None:
        try:
            timeline = self.server.collector.collect_experiment_timeline()  # type: ignore[attr-defined]
            match = next((r for r in timeline if r.get("run_id") == run_id), None)
            if match is None:
                self._send_404()
                return
            body = json.dumps(match, ensure_ascii=False, default=str).encode("utf-8")
        except Exception:
            body = json.dumps({"error": traceback.format_exc()}).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._send_cors_headers()
        self._send_no_cache_headers()
        self.end_headers()
        self.wfile.write(body)

    def _serve_sse(self) -> None:
        """
        Long-lived SSE stream. Pushes a full DashboardData JSON event every
        _SSE_INTERVAL seconds. Terminates cleanly when the client disconnects.
        """
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self._send_cors_headers()
        self.end_headers()

        try:
            # Initial ping so the browser knows the connection is alive
            self._sse_write("ping", "{}")

            while True:
                try:
                    data = self.server.collector.collect()  # type: ignore[attr-defined]
                    payload = json.dumps(data.to_dict(), ensure_ascii=False, default=str)
                    self._sse_write("update", payload)
                except Exception:
                    err_payload = json.dumps({"error": traceback.format_exc()})
                    self._sse_write("error", err_payload)

                # Sleep in small increments so Ctrl+C is responsive
                for _ in range(_SSE_INTERVAL * 10):
                    time.sleep(0.1)

        except (BrokenPipeError, ConnectionResetError, OSError):
            # Client disconnected — clean exit
            pass

    def _sse_write(self, event: str, data: str) -> None:
        """Write a single SSE message to the response stream."""
        message = f"event: {event}\ndata: {data}\n\n"
        self.wfile.write(message.encode("utf-8"))
        self.wfile.flush()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")

    def _send_no_cache_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")

    def _send_404(self) -> None:
        body = b"404 Not Found"
        self.send_response(404)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, code: int, message: str) -> None:
        body = message.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _load_index_html() -> Optional[bytes]:
    if _INDEX_HTML.exists():
        return _INDEX_HTML.read_bytes()
    try:
        resource = (
            importlib.resources.files("drl_autoresearch.dashboard")
            .joinpath("static")
            .joinpath("index.html")
        )
        if resource.is_file():
            return resource.read_bytes()
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# DashboardServer public API
# ---------------------------------------------------------------------------


class DashboardServer:
    """
    Lightweight HTTP dashboard server using Python stdlib only.

    Usage::

        server = DashboardServer(project_dir=Path("."), port=8765)
        server.start()          # blocking — Ctrl+C to stop
        # or:
        thread = server.start_background()
    """

    def __init__(self, project_dir: Path, port: int = 8765) -> None:
        self.project_dir = Path(project_dir)
        self.port = port
        self._collector = MetricsCollector(project_dir=self.project_dir)
        self._httpd: Optional[_ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the dashboard server (blocking). Press Ctrl+C to stop."""
        self._start_server()
        self._print_banner()
        try:
            self._httpd.serve_forever()  # type: ignore[union-attr]
        except KeyboardInterrupt:
            print("\nShutting down dashboard...")
        finally:
            self.stop()

    def start_background(self) -> threading.Thread:
        """Start the server in a daemon thread. Returns the thread."""
        self._start_server()
        self._print_banner()
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,  # type: ignore[union-attr]
            daemon=True,
            name="drl-dashboard",
        )
        self._thread.start()
        return self._thread

    def stop(self) -> None:
        """Shut the server down gracefully."""
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _start_server(self) -> None:
        self._httpd = _ThreadingHTTPServer(
            ("", self.port),
            _DashboardHandler,
            self._collector,
        )

    def _print_banner(self) -> None:
        print(
            "\n"
            "\u2501" * 49 + "\n"
            "  DRL AutoResearch Dashboard\n"
            f"  http://localhost:{self.port}\n"
            "  Press Ctrl+C to stop\n"
            "\u2501" * 49
        )
