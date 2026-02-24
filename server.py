"""
server.py — HTTP dashboard server.

Serves the static dashboard.html and JSON/JPEG API endpoints.
Import and call start_dashboard(robot, port) from main.py.
"""
import json
import math
import os
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

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
                self.wfile.write(jpeg)
            else:
                self.send_response(204); self.end_headers()

        elif self.path == '/stream3d':
            # Live: current base-frame camera points + current lidar scan outline
            xyz, rgb = (_robot.get_live_frame() if _robot else (None, None))
            pts = []
            if xyz is not None and len(xyz) > 0:
                pts = [
                    [round(float(xyz[i,0]),3), round(float(xyz[i,1]),3), round(float(xyz[i,2]),3),
                     int(rgb[i,0]),            int(rgb[i,1]),            int(rgb[i,2])]
                    for i in range(len(xyz))
                ]
            lidar_outline = []
            if _robot:
                scan = _robot.lidar.get_scan()
                for ang, dist_mm in scan.items():
                    if 10 < dist_mm < 8000:
                        rad = math.radians(float(ang))
                        d   = dist_mm / 1000.0
                        lidar_outline.append([
                            round(d * math.cos(rad), 3),
                            round(d * math.sin(rad), 3),
                        ])
            self._json({"pts": pts, "count": len(pts), "lidar_outline": lidar_outline})

        elif self.path == '/map3d':
            pts = _robot.get_map3d() if _robot else None
            self._json(pts or {})

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
        self.wfile.write(content.encode())

    def _json(self, data):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data or {}, default=str).encode())

    def log_message(self, *_):
        pass   # suppress request logs


# ── Public entry point ────────────────────────────────────────────
def start_dashboard(robot, port: int = 8080):
    global _robot
    _robot = robot

    def serve():
        HTTPServer(('0.0.0.0', port), DashHandler).serve_forever()

    threading.Thread(target=serve, daemon=True).start()
    print(f"[Dashboard] http://localhost:{port}")