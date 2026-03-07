#!/usr/bin/env python3
"""
mapping_gui.py — SLAM Mapping desktop controller.
Optimised for 7-inch touchscreen (800×480 / 1024×600).

Layout
------
Left sidebar  (~220 px, fixed) : title · status · mode selector · start/stop · log
Right panel   (fills rest)     : large live 2-D map canvas
"""

import os
import sys
import json
import math
import subprocess
import threading
import urllib.request
import tkinter as tk
from tkinter import scrolledtext

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON     = sys.executable
DASH_URL   = "http://localhost:8080"
API_MAP3D  = f"{DASH_URL}/map3d"
API_DATA   = f"{DASH_URL}/data"
POLL_MS    = 2000

# ── Palette ────────────────────────────────────────────────────────
BG         = "#0d1117"
SIDEBAR    = "#111520"
ACCENT     = "#2563eb"
ACCENT_H   = "#3b82f6"
STOP_C     = "#dc2626"
STOP_H     = "#ef4444"
GREEN      = "#22c55e"
GREEN_D    = "#14532d"
YELLOW     = "#eab308"
TEXT       = "#e2e8f0"
TEXT_DIM   = "#64748b"
LOG_BG     = "#090c12"
LOG_FG     = "#64748b"
LOG_HIGH   = "#38bdf8"
MAP_BG     = "#06080e"
GRID_C     = "#161a27"
LIDAR_C    = "#cbd5e1"
ROBOT_C    = "#22c55e"
MODE_ACT   = "#1e3a5f"
MODE_ACT_B = "#2563eb"
MODE_IDLE  = "#181d2e"
DIVIDER    = "#1e2235"


# ── Jet colormap (blue=near → red=far) ────────────────────────────
def _jet(t: float) -> str:
    t = max(0.0, min(1.0, t))
    r = min(1.0, max(0.0, 1.5 - abs(4*t - 3)))
    g = min(1.0, max(0.0, 1.5 - abs(4*t - 2)))
    b = min(1.0, max(0.0, 1.5 - abs(4*t - 1)))
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

_JET = [_jet(i / 63) for i in range(64)]

def _depth_color(d, d_max):
    if d_max < 0.01:
        return _JET[0]
    return _JET[int(max(0.0, min(1.0, d / d_max)) * 63)]


# ── HTTP helper ────────────────────────────────────────────────────
def _fetch_json(url, timeout=1.5):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════
class MappingGUI:
    def __init__(self, root: tk.Tk):
        self.root     = root
        self._proc    = None
        self._running = False
        self._poll_id = None
        self._mode    = tk.StringVar(value="map_only")

        self._map_lidar  = []
        self._map_camera = []
        self._robot_pos  = None
        self._robot_yaw  = 0.0

        self._view_scale = 60.0
        self._view_ox    = 0.0
        self._view_oy    = 0.0
        self._drag_start = None

        self._build_ui()
        root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ════════════════════════════════════════════════════════════════
    #  UI
    # ════════════════════════════════════════════════════════════════
    def _build_ui(self):
        root = self.root
        root.title("SLAM Mapping")
        root.configure(bg=BG)
        root.minsize(800, 460)

        # ── Root split: sidebar | map ──────────────────────────────
        root.columnconfigure(0, weight=0)   # sidebar: fixed
        root.columnconfigure(1, weight=1)   # map: expands
        root.rowconfigure(0, weight=1)

        # ── Sidebar ───────────────────────────────────────────────
        sb = tk.Frame(root, bg=SIDEBAR, width=220)
        sb.grid(row=0, column=0, sticky="nsew")
        sb.pack_propagate(False)   # hold fixed width

        # Title
        tk.Label(sb, text="SLAM Robot", bg=SIDEBAR, fg=TEXT,
                 font=("Helvetica", 17, "bold"),
                 pady=14).pack(fill="x", padx=14)

        tk.Frame(sb, bg=DIVIDER, height=1).pack(fill="x")

        # Status badge
        self._badge_var = tk.StringVar(value="Stopped")
        self._badge = tk.Label(sb, textvariable=self._badge_var,
                               bg=TEXT_DIM, fg="#ffffff",
                               font=("Helvetica", 11, "bold"),
                               pady=6)
        self._badge.pack(fill="x", padx=14, pady=(10, 2))

        # Status text
        self._status_var = tk.StringVar(value="Select mode, then Start.")
        tk.Label(sb, textvariable=self._status_var,
                 bg=SIDEBAR, fg=TEXT_DIM,
                 font=("Helvetica", 9),
                 wraplength=192, justify="left",
                 pady=2).pack(fill="x", padx=14)

        tk.Frame(sb, bg=DIVIDER, height=1).pack(fill="x", pady=8)

        # Mode label
        tk.Label(sb, text="MODE", bg=SIDEBAR, fg=TEXT_DIM,
                 font=("Helvetica", 8, "bold"),
                 pady=0).pack(fill="x", padx=14)

        # Mode buttons (tall, full sidebar width)
        self._mode_map_btn = _ModeButton(
            sb,
            text="Map Only",
            sub="motors off",
            value="map_only",
            var=self._mode,
            command=lambda: self._select_mode("map_only"))
        self._mode_map_btn.pack(fill="x", padx=14, pady=(4, 3))

        self._mode_nav_btn = _ModeButton(
            sb,
            text="Navigate + Map",
            sub="motors on",
            value="navigate",
            var=self._mode,
            command=lambda: self._select_mode("navigate"))
        self._mode_nav_btn.pack(fill="x", padx=14, pady=(0, 4))

        tk.Frame(sb, bg=DIVIDER, height=1).pack(fill="x")

        # Start / Stop (very large for touch)
        self._start_btn = _SidebarButton(
            sb, text="START",
            bg=ACCENT, hover=ACCENT_H,
            command=self._start,
            font=("Helvetica", 16, "bold"),
            fg="#ffffff", pady=18)
        self._start_btn.pack(fill="x", padx=14, pady=(10, 6))

        self._stop_btn = _SidebarButton(
            sb, text="STOP",
            bg="#252535", hover=STOP_C,
            command=self._stop,
            font=("Helvetica", 16, "bold"),
            fg=TEXT_DIM, pady=18)
        self._stop_btn.pack(fill="x", padx=14, pady=(0, 10))

        tk.Frame(sb, bg=DIVIDER, height=1).pack(fill="x")

        # Dashboard link
        dash_lbl = tk.Label(sb, text="Open Dashboard",
                            bg=SIDEBAR, fg=ACCENT,
                            font=("Helvetica", 9), cursor="hand2", pady=6)
        dash_lbl.pack(fill="x", padx=14)
        dash_lbl.bind("<Button-1>", lambda _: self._open_browser())

        tk.Frame(sb, bg=DIVIDER, height=1).pack(fill="x")

        # Log (takes remaining space)
        log_hdr = tk.Frame(sb, bg=SIDEBAR)
        log_hdr.pack(fill="x", padx=14, pady=(6, 2))
        tk.Label(log_hdr, text="LOG", bg=SIDEBAR, fg=TEXT_DIM,
                 font=("Helvetica", 8, "bold")).pack(side="left")
        tk.Button(log_hdr, text="Clear", bg=SIDEBAR, fg=TEXT_DIM,
                  font=("Helvetica", 8), relief="flat", cursor="hand2",
                  activebackground=BG, activeforeground=TEXT,
                  command=self._clear_log).pack(side="right")

        self._log = scrolledtext.ScrolledText(
            sb, bg=LOG_BG, fg=LOG_FG,
            font=("Courier", 8), relief="flat", bd=0,
            wrap="word", state="disabled")
        self._log.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        self._log.tag_config("info",  foreground=LOG_HIGH)
        self._log.tag_config("warn",  foreground=YELLOW)
        self._log.tag_config("error", foreground=STOP_C)
        self._log.tag_config("dim",   foreground=TEXT_DIM)

        # ── Map panel (right, fills all remaining space) ───────────
        mp = tk.Frame(root, bg=BG)
        mp.grid(row=0, column=1, sticky="nsew")
        mp.rowconfigure(1, weight=1)
        mp.columnconfigure(0, weight=1)

        # Map header bar
        mh = tk.Frame(mp, bg=BG)
        mh.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 0))

        tk.Label(mh, text="Live 2D Map", bg=BG, fg=TEXT,
                 font=("Helvetica", 13, "bold")).pack(side="left")

        self._map_info = tk.Label(mh, text="", bg=BG, fg=TEXT_DIM,
                                  font=("Helvetica", 9))
        self._map_info.pack(side="left", padx=10)

        # Zoom controls (big touch targets)
        zf = tk.Frame(mh, bg=BG)
        zf.pack(side="right")
        for sym, delta, tip in (("−", -20, "Zoom out"), ("+", +20, "Zoom in"), ("⌂", 0, "Reset")):
            cmd = self._reset_view if delta == 0 else lambda d=delta: self._zoom(d)
            tk.Button(zf, text=sym, bg="#1a1d2e", fg=TEXT,
                      font=("Helvetica", 14, "bold"), relief="flat",
                      width=3, cursor="hand2", pady=4,
                      activebackground=ACCENT, activeforeground="#ffffff",
                      command=cmd).pack(side="left", padx=2)

        # Canvas — fills everything
        self._canvas = tk.Canvas(mp, bg=MAP_BG, relief="flat", bd=0,
                                 highlightthickness=0)
        self._canvas.grid(row=1, column=0, sticky="nsew", padx=6, pady=6)
        self._canvas.bind("<ButtonPress-1>", self._drag_start_cb)
        self._canvas.bind("<B1-Motion>",     self._drag_move_cb)
        self._canvas.bind("<MouseWheel>",    self._scroll_cb)
        self._canvas.bind("<Button-4>",      self._scroll_cb)
        self._canvas.bind("<Button-5>",      self._scroll_cb)

        # Legend bar at bottom of map
        leg = tk.Frame(mp, bg=BG)
        leg.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 6))

        tk.Label(leg, text="Near", bg=BG, fg=_JET[0],
                 font=("Helvetica", 8)).pack(side="left")
        bar = tk.Canvas(leg, bg=BG, height=10, width=140,
                        relief="flat", bd=0, highlightthickness=0)
        bar.pack(side="left", padx=6)
        for i in range(140):
            bar.create_line(i, 0, i, 10, fill=_jet(i / 139))
        tk.Label(leg, text="Far", bg=BG, fg=_JET[63],
                 font=("Helvetica", 8)).pack(side="left")
        tk.Label(leg, text="    ━ LiDAR", bg=BG, fg=LIDAR_C,
                 font=("Helvetica", 8)).pack(side="left", padx=12)
        tk.Label(leg, text="▲ Robot", bg=BG, fg=ROBOT_C,
                 font=("Helvetica", 8)).pack(side="left")

        self._draw_placeholder()
        self._select_mode("map_only")

    # ════════════════════════════════════════════════════════════════
    #  Mode
    # ════════════════════════════════════════════════════════════════
    def _select_mode(self, mode):
        if self._running:
            return
        self._mode.set(mode)
        self._mode_map_btn.refresh()
        self._mode_nav_btn.refresh()
        self._status_var.set(
            "Map Only — motors off." if mode == "map_only"
            else "Navigate + Map — robot will move.")

    def _get_cmd(self):
        base = [PYTHON, os.path.join(SCRIPT_DIR, "main.py")]
        if self._mode.get() == "map_only":
            base.append("--map-only")
        return base

    # ════════════════════════════════════════════════════════════════
    #  Process control
    # ════════════════════════════════════════════════════════════════
    def _start(self):
        if self._running:
            return
        cmd   = self._get_cmd()
        label = "Map Only" if self._mode.get() == "map_only" else "Navigate + Map"
        self._log_line(f"[GUI] Starting — {label}", "info")
        try:
            self._proc = subprocess.Popen(
                cmd, cwd=SCRIPT_DIR,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1)
        except Exception as e:
            self._log_line(f"[GUI] Failed: {e}", "error")
            return
        self._running = True
        self._update_state()
        threading.Thread(target=self._stream_output, daemon=True).start()
        self._schedule_poll()

    def _stop(self):
        if not self._running or self._proc is None:
            return
        self._log_line("[GUI] Stopping…", "warn")
        if self._poll_id:
            self.root.after_cancel(self._poll_id)
            self._poll_id = None
        # Brake wheels first
        try:
            req = urllib.request.Request(
                f"{DASH_URL}/cmd",
                data=b'{"action":"stop"}',
                headers={"Content-Type": "application/json"},
                method="POST")
            urllib.request.urlopen(req, timeout=1.0)
            self._log_line("[GUI] Brake sent — wheels stopping.", "warn")
        except Exception:
            pass
        try:
            self._proc.terminate()
            def _wait():
                try:
                    self._proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait()
                self.root.after(0, self._on_proc_exit)
            threading.Thread(target=_wait, daemon=True).start()
        except Exception as e:
            self._log_line(f"[GUI] Error: {e}", "error")

    def _stream_output(self):
        try:
            for line in self._proc.stdout:
                line = line.rstrip("\n")
                low  = line.lower()
                if any(k in low for k in ("error", "exception", "traceback", "failed")):
                    tag = "error"
                elif any(k in low for k in ("warn", "skip")):
                    tag = "warn"
                elif any(k in low for k in ("[slam]", "[lidar]", "[camera]",
                                             "[imu]", "[cuda]", "[jetson]",
                                             "[config]", "dashboard")):
                    tag = "info"
                else:
                    tag = "dim"
                self.root.after(0, self._log_line, line, tag)
        except Exception:
            pass
        self.root.after(0, self._on_proc_exit)

    def _on_proc_exit(self):
        if not self._running:
            return
        self._running = False
        if self._poll_id:
            self.root.after_cancel(self._poll_id)
            self._poll_id = None
        code = getattr(self._proc, "returncode", None)
        self._log_line(f"[GUI] Exited (code {code}).",
                       "warn" if code else "info")
        self._update_state()
        self._draw_placeholder("Process stopped.")

    # ════════════════════════════════════════════════════════════════
    #  Map polling
    # ════════════════════════════════════════════════════════════════
    def _schedule_poll(self):
        if self._running:
            self._poll_id = self.root.after(POLL_MS, self._poll_map)

    def _poll_map(self):
        def _fetch():
            m3d  = _fetch_json(API_MAP3D)
            data = _fetch_json(API_DATA)
            self.root.after(0, self._on_map_data, m3d, data)
        threading.Thread(target=_fetch, daemon=True).start()

    def _on_map_data(self, m3d, data):
        if m3d:
            self._map_lidar  = m3d.get("lidar")  or []
            self._map_camera = m3d.get("camera") or []
        if data:
            self._robot_pos = data.get("position")
            n_lid = data.get("lidar_pts", 0)
            n_cam = data.get("cam_pts",   0)
            frm   = data.get("frame", 0)
            self._map_info.configure(
                text=f"frame {frm}  ·  lidar {n_lid} pts  ·  cam {n_cam} pts")
        self._render_map()
        if self._running:
            self._schedule_poll()

    # ════════════════════════════════════════════════════════════════
    #  Map rendering
    # ════════════════════════════════════════════════════════════════
    def _render_map(self):
        c  = self._canvas
        c.delete("all")
        cw = c.winfo_width()
        ch = c.winfo_height()
        if cw < 10 or ch < 10:
            return

        cx, cy = cw / 2, ch / 2
        s = self._view_scale

        def w2c(wx, wy):
            return (cx + (wx - self._view_ox) * s,
                    cy - (wy - self._view_oy) * s)

        # ── Grid ──────────────────────────────────────────────────
        step_m = _nice_grid_step(90.0 / s)
        x0 = self._view_ox - cw / (2*s);  x1 = self._view_ox + cw / (2*s)
        y0 = self._view_oy - ch / (2*s);  y1 = self._view_oy + ch / (2*s)

        gx = math.floor(x0 / step_m) * step_m
        while gx <= x1:
            px, _ = w2c(gx, 0)
            col = "#252840" if abs(gx) < step_m * 0.01 else GRID_C
            c.create_line(px, 0, px, ch, fill=col)
            c.create_text(px + 3, 6, text=f"{gx:.3g}", fill="#2a3050",
                          font=("Helvetica", 7), anchor="nw")
            gx += step_m

        gy = math.floor(y0 / step_m) * step_m
        while gy <= y1:
            _, py = w2c(0, gy)
            col = "#252840" if abs(gy) < step_m * 0.01 else GRID_C
            c.create_line(0, py, cw, py, fill=col)
            c.create_text(4, py - 8, text=f"{gy:.3g}", fill="#2a3050",
                          font=("Helvetica", 7), anchor="nw")
            gy += step_m

        # Scale bar
        bar_px = int(step_m * s)
        bx, by = 14, ch - 16
        c.create_line(bx, by, bx + bar_px, by, fill=TEXT_DIM, width=2)
        c.create_line(bx, by-4, bx, by+4, fill=TEXT_DIM, width=2)
        c.create_line(bx+bar_px, by-4, bx+bar_px, by+4, fill=TEXT_DIM, width=2)
        c.create_text(bx + bar_px//2, by - 10,
                      text=f"{step_m:.3g} m", fill=TEXT_DIM,
                      font=("Helvetica", 9, "bold"))

        # ── Camera points ─────────────────────────────────────────
        cam = self._map_camera
        if cam:
            d_max = max((p[3] for p in cam if len(p) >= 4), default=1.0)
            step  = max(1, len(cam) // 3000)
            for p in cam[::step]:
                if len(p) < 4:
                    continue
                px, py = w2c(p[0], p[1])
                if px < -4 or px > cw+4 or py < -4 or py > ch+4:
                    continue
                col = _depth_color(p[3], d_max)
                c.create_rectangle(px-1, py-1, px+2, py+2, fill=col, outline="")

        # ── LiDAR points ──────────────────────────────────────────
        lidar = self._map_lidar
        if lidar:
            step = max(1, len(lidar) // 2000)
            for p in lidar[::step]:
                if len(p) < 2:
                    continue
                px, py = w2c(p[0], p[1])
                if px < -4 or px > cw+4 or py < -4 or py > ch+4:
                    continue
                c.create_rectangle(px-1, py-1, px+2, py+2,
                                   fill=LIDAR_C, outline="")

        # ── Robot ─────────────────────────────────────────────────
        rpos = self._robot_pos
        if rpos and len(rpos) >= 2:
            rx, ry = w2c(rpos[0], rpos[1])
            yaw = self._robot_yaw
            r   = 10
            tip_x = rx + r*2 * math.cos(yaw);  tip_y = ry - r*2 * math.sin(yaw)
            lx  = rx + r * math.cos(yaw + math.radians(140))
            ly  = ry - r * math.sin(yaw + math.radians(140))
            rx2 = rx + r * math.cos(yaw - math.radians(140))
            ry2 = ry - r * math.sin(yaw - math.radians(140))
            c.create_polygon(tip_x, tip_y, lx, ly, rx2, ry2,
                             fill=ROBOT_C, outline="#ffffff", width=2)
            c.create_oval(rx-4, ry-4, rx+4, ry+4,
                          fill=ROBOT_C, outline="")

        if not cam and not lidar:
            c.create_text(cw//2, ch//2,
                          text="Waiting for map data…",
                          fill=TEXT_DIM, font=("Helvetica", 14))

    def _draw_placeholder(self, msg="Start mapping to see the live map."):
        c = self._canvas
        c.delete("all")
        cw = c.winfo_width() or 500
        ch = c.winfo_height() or 400
        c.create_text(cw//2, ch//2, text=msg,
                      fill=TEXT_DIM, font=("Helvetica", 13))

    # ── View controls ─────────────────────────────────────────────
    def _zoom(self, delta):
        self._view_scale = max(10.0, min(500.0, self._view_scale + delta))
        self._render_map()

    def _reset_view(self):
        self._view_ox = 0.0; self._view_oy = 0.0; self._view_scale = 60.0
        self._render_map()

    def _drag_start_cb(self, evt):
        self._drag_start = (evt.x, evt.y, self._view_ox, self._view_oy)

    def _drag_move_cb(self, evt):
        if not self._drag_start:
            return
        x0, y0, ox, oy = self._drag_start
        self._view_ox = ox - (evt.x - x0) / self._view_scale
        self._view_oy = oy + (evt.y - y0) / self._view_scale
        self._render_map()

    def _scroll_cb(self, evt):
        if evt.num == 4 or (hasattr(evt, "delta") and evt.delta > 0):
            self._zoom(+15)
        else:
            self._zoom(-15)

    # ── UI state ──────────────────────────────────────────────────
    def _update_state(self):
        lbl = "Map Only" if self._mode.get() == "map_only" else "Navigate+Map"
        if self._running:
            self._badge_var.set(f"  RUNNING  ")
            self._badge.configure(bg=GREEN)
            self._status_var.set(f"{lbl}\n{DASH_URL}")
            self._start_btn.set_colors(bg=GREEN_D, hover=GREEN_D, fg=TEXT_DIM)
            self._stop_btn.set_colors(bg=STOP_C,   hover=STOP_H,  fg="#ffffff")
            self._mode_map_btn.set_disabled(True)
            self._mode_nav_btn.set_disabled(True)
        else:
            self._badge_var.set("  STOPPED  ")
            self._badge.configure(bg=TEXT_DIM)
            self._status_var.set("Select mode, then Start.")
            self._start_btn.set_colors(bg=ACCENT,   hover=ACCENT_H, fg="#ffffff")
            self._stop_btn.set_colors(bg="#252535",  hover=STOP_C,   fg=TEXT_DIM)
            self._mode_map_btn.set_disabled(False)
            self._mode_nav_btn.set_disabled(False)
            self._mode_map_btn.refresh()
            self._mode_nav_btn.refresh()

    def _log_line(self, text, tag="dim"):
        self._log.configure(state="normal")
        self._log.insert("end", text + "\n", tag)
        self._log.see("end")
        self._log.configure(state="disabled")

    def _clear_log(self):
        self._log.configure(state="normal")
        self._log.delete("1.0", "end")
        self._log.configure(state="disabled")

    def _open_browser(self):
        import webbrowser
        webbrowser.open(DASH_URL)

    def _on_close(self):
        if self._running:
            self._stop()
        self.root.destroy()


# ══════════════════════════════════════════════════════════════════
#  Mode toggle button
# ══════════════════════════════════════════════════════════════════
class _ModeButton(tk.Frame):
    def __init__(self, parent, text, sub, value, var, command):
        super().__init__(parent, bg=MODE_IDLE,
                         highlightthickness=2,
                         highlightbackground=MODE_IDLE,
                         cursor="hand2")
        self._value    = value
        self._var      = var
        self._cmd      = command
        self._disabled = False

        self._lbl = tk.Label(self, text=text, bg=MODE_IDLE, fg=TEXT_DIM,
                             font=("Helvetica", 12, "bold"),
                             padx=10, pady=10)
        self._lbl.pack(fill="x")
        self._sub = tk.Label(self, text=sub, bg=MODE_IDLE, fg=TEXT_DIM,
                             font=("Helvetica", 8), padx=10, pady=3)
        self._sub.pack(fill="x")

        for w in (self, self._lbl, self._sub):
            w.bind("<Button-1>", self._on_click)
            w.bind("<Enter>",    self._on_enter)
            w.bind("<Leave>",    self._on_leave)

    def _on_click(self, _=None):
        if not self._disabled:
            self._cmd()

    def _on_enter(self, _=None):
        if not self._disabled and self._var.get() != self._value:
            self._set_bg("#1e2540")

    def _on_leave(self, _=None):
        if not self._disabled:
            self.refresh()

    def refresh(self):
        active = self._var.get() == self._value
        bg  = MODE_ACT   if active else MODE_IDLE
        bdr = MODE_ACT_B if active else MODE_IDLE
        fg  = TEXT       if active else TEXT_DIM
        self._set_bg(bg)
        self.configure(highlightbackground=bdr)
        self._lbl.configure(fg=fg)

    def set_disabled(self, disabled):
        self._disabled = disabled
        if disabled:
            self._set_bg("#0f111a")
            self.configure(highlightbackground="#0f111a")
            self._lbl.configure(fg="#252840")
            self._sub.configure(fg="#252840")
        else:
            self._sub.configure(fg=TEXT_DIM)

    def _set_bg(self, color):
        self.configure(bg=color)
        self._lbl.configure(bg=color)
        self._sub.configure(bg=color)


# ══════════════════════════════════════════════════════════════════
#  Large sidebar button
# ══════════════════════════════════════════════════════════════════
class _SidebarButton(tk.Label):
    def __init__(self, parent, text, bg, hover, command,
                 fg="#ffffff", font=None, pady=16):
        self._bg  = bg; self._hover = hover; self._cmd = command
        super().__init__(parent, text=text, bg=bg, fg=fg,
                         font=font, pady=pady,
                         cursor="hand2", relief="flat")
        self.bind("<Enter>",    lambda _: self.configure(bg=self._hover))
        self.bind("<Leave>",    lambda _: self.configure(bg=self._bg))
        self.bind("<Button-1>", lambda _: self._cmd())

    def set_colors(self, bg, hover, fg):
        self._bg = bg; self._hover = hover
        self.configure(bg=bg, fg=fg)


# ── Grid step ──────────────────────────────────────────────────────
def _nice_grid_step(target_m):
    if target_m <= 0:
        return 1.0
    exp  = math.floor(math.log10(target_m))
    base = 10 ** exp
    for m in (1, 2, 5, 10):
        if base * m >= target_m:
            return base * m
    return base * 10


# ── Entry point ────────────────────────────────────────────────────
def main():
    root = tk.Tk()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    if sw <= 1024:
        root.geometry(f"{sw}x{sh}+0+0")
    else:
        w, h = 1100, 620
        root.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")
    root.resizable(True, True)
    try:
        root.iconphoto(True, tk.PhotoImage(
            file=os.path.join(SCRIPT_DIR, "icon.png")))
    except Exception:
        pass
    MappingGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
