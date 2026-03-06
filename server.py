"""
server.py — HTTP dashboard server.

Serves the static dashboard.html and JSON/JPEG API endpoints.
Import and call start_dashboard(robot, port) from main.py.
"""
import json
import os
import threading
from http.server import SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn, TCPServer
from pathlib import Path


class HTTPServer(ThreadingMixIn, TCPServer):
    """Multi-threaded HTTP server — handles each request in its own thread."""
    allow_reuse_address = True
    daemon_threads = True

# Absolute path to dashboard.html (same directory as this file)
_HTML_PATH = Path(__file__).parent / "dashboard.html"

_robot = None   # set by start_dashboard()


class DashHandler(SimpleHTTPRequestHandler):

    # ── GET ──────────────────────────────────────────────────────
    def do_GET(self):
        if self.path == '/':
            try:
                html = _HTML_PATH.read_text()
                self._html(html)
            except FileNotFoundError:
                self.send_response(500); self.end_headers()
                self.wfile.write(b"dashboard.html not found")

        elif self.path == '/data':
            self._json(_robot.get_dash_data() if _robot else {})

        elif self.path == '/map':
            self._json(_robot.get_occ_grid() if _robot else {})

        elif self.path == '/camera':
            jpeg = _robot.get_cam_jpeg() if _robot else None
            if jpeg:
                self.send_response(200)
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Cache-Control', 'no-cache')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                try:
                    self.wfile.write(jpeg)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.send_response(204); self.end_headers()

        elif self.path == '/stream3d':
            # Serve pre-serialised payload built by the mapping loop
            payload = _robot.get_stream3d_bytes() if _robot else b'{}'
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            try:
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass

        elif self.path == '/map3d':
            pts = _robot.get_map3d() if _robot else None
            self._json(pts or {})

        elif self.path == '/mesh':
            mesh = _robot.get_mesh() if _robot else None
            self._json(mesh or {})

        elif self.path == '/calibrate':
            # Return current camera mounting angles
            cfg = _robot.config if _robot else {}
            self._json({
                "CAMERA_YAW_DEG":   cfg.get("CAMERA_YAW_DEG",   0.0),
                "CAMERA_PITCH_DEG": cfg.get("CAMERA_PITCH_DEG", 0.0),
                "CAMERA_ROLL_DEG":  cfg.get("CAMERA_ROLL_DEG",  0.0),
            })

        else:
            self.send_response(404); self.end_headers()

    # ── POST ─────────────────────────────────────────────────────
    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        body   = json.loads(self.rfile.read(length)) if length > 0 else {}

        if self.path == '/cmd':
            if body.get("action") == "stop" and _robot:
                _robot._brake()
            self._json({"ok": True})

        elif self.path == '/calibrate':
            # Live-update camera mounting angles without restart
            from config import save_config_keys, build_camera_transform
            import numpy as np
            updates = {}
            for key in ("CAMERA_YAW_DEG", "CAMERA_PITCH_DEG", "CAMERA_ROLL_DEG"):
                if key in body:
                    v = float(body[key])
                    updates[key] = v
                    if _robot:
                        _robot.config[key] = v
            if _robot and updates:
                _robot.T_base_cam = build_camera_transform(_robot.config)
            try:
                save_config_keys(updates)
                self._json({"ok": True, "saved": "robot_config.json"})
            except Exception as e:
                self._json({"ok": False, "error": str(e)})

        else:
            self.send_response(404); self.end_headers()

    # ── Helpers ──────────────────────────────────────────────────
    def _html(self, content: str):
        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.end_headers()
        try:
            self.wfile.write(content.encode())
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, data):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        try:
            self.wfile.write(json.dumps(data or {}, default=str).encode())
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *_):
        pass   # suppress request logs

    def handle_error(self, request, client_address):
        pass   # suppress all connection-level errors (BrokenPipe etc.)


# ── Public entry point ────────────────────────────────────────────
def start_dashboard(robot, port: int = 8080):
    global _robot
    _robot = robot

    def serve():
        HTTPServer(('0.0.0.0', port), DashHandler).serve_forever()

    threading.Thread(target=serve, daemon=True).start()
    print(f"[Dashboard] http://localhost:{port}")