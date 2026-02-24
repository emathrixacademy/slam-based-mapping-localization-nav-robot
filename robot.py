"""
robot.py — SLAMAvoidanceRobot and VFHNavigator.

Ties together all hardware modules (motors, lidar, imu, camera) and
maintains the SLAM state machine: mapping + VFH obstacle avoidance.
Does NOT contain any HTTP / dashboard code — see server.py.
"""
import math
import os
import sys
import time
import random
import threading
from collections import deque

from config import (
    _import_numpy, _import_cv2,
    build_camera_transform, build_lidar_transform,
)
from motors import DualMotors, probe_motor_ports
from lidar  import (
    LD06, lidar_to_3d, transform_points,
    voxel_downsample, numpy_icp, save_pcd,
)
from imu    import ICM20948
from camera import RealSenseCamera


# ═══════════════════════════════════════════════════════════════════
#  VFH Gap-Finding Navigator
# ═══════════════════════════════════════════════════════════════════
class VFHNavigator:
    def __init__(self, bin_size=5.0, threshold=500, min_gap=35.0):
        self.bin_size  = bin_size
        self.threshold = threshold
        self.min_gap   = min_gap
        self.n_bins    = int(360 / bin_size)

    def find_gaps(self, hist):
        blocked = [h < self.threshold for h in hist]
        n   = self.n_bins
        ext = blocked + blocked
        gaps, visited = [], set()
        in_gap = False; gap_start = 0; i = 0

        while i < len(ext):
            if not ext[i] and not in_gap:
                gap_start = i % n; in_gap = True
            elif ext[i] and in_gap:
                gap_end = (i - 1) % n
                w  = i - (gap_start if gap_start <= (i - 1) else gap_start - n)
                wd = w * self.bin_size
                if wd >= self.min_gap:
                    center = (gap_start * self.bin_size + wd / 2) % 360.0
                    key    = (gap_start, gap_end)
                    if key not in visited:
                        gaps.append({
                            "start":  gap_start * self.bin_size,
                            "end":    ((gap_end + 1) * self.bin_size) % 360.0,
                            "width":  wd,
                            "center": center,
                        })
                        visited.add(key)
                in_gap = False
            i += 1

        if not any(blocked):
            gaps = [{"start": 0, "end": 360, "width": 360, "center": 0}]
        return gaps

    def best_heading(self, hist, desired=0.0):
        gaps = self.find_gaps(hist)
        if not gaps:
            return None, None
        best = None; best_score = -999
        for g in gaps:
            diff  = abs(((g["center"] - desired + 180) % 360) - 180)
            ws    = min(g["width"] / 90.0, 1.0)
            hs    = 1.0 - diff / 180.0
            fb    = 0.3 if diff < 45 else 0.0
            score = ws * 0.4 + hs * 0.6 + fb
            if score > best_score:
                best_score = score; best = g
        return (best["center"], best) if best else (None, None)


# ═══════════════════════════════════════════════════════════════════
#  SLAM Avoidance Robot
# ═══════════════════════════════════════════════════════════════════
class SLAMAvoidanceRobot:
    # Speed constants (RPM)
    MAX_SPEED     = 15
    CRUISE_SPEED  = 5
    SLOW_SPEED    = 5
    TURN_SPEED    = 5
    REVERSE_SPEED = 5

    # Distance thresholds (mm)
    EMERGENCY_DIST = 200
    STOP_DIST      = 350
    SLOW_DIST      = 800
    CRUISE_DIST    = 1500
    OPEN_DIST      = 2500

    # Timing
    REVERSE_TIME = 0.6
    TURN_TIME    = 0.5
    LOOP_RATE    = 0.05
    STUCK_TIMEOUT = 4.0

    # VFH params
    VFH_BIN   = 5.0
    VFH_THRESH = 500
    VFH_GAP   = 35.0

    # Map params
    MAP_VOXEL   = 0.05
    MAX_MAP_PTS = 500_000

    def __init__(self, config: dict, map_only: bool = False):
        _import_numpy()
        from config import np
        print("[SLAM] Initializing…")
        self.config   = config
        self.map_only = map_only
        self._running = False

        # ── Motors ──
        self.motors = None
        if not map_only:
            lp = config["LEFT_MOTOR_PORT"]
            rp = config["RIGHT_MOTOR_PORT"]
            if not os.path.exists(lp) or not os.path.exists(rp):
                print("[SLAM] Auto-detecting motor ports…")
                dl, dr = probe_motor_ports(
                    config["LEFT_MOTOR_ID"], config["RIGHT_MOTOR_ID"])
                if dl: config["LEFT_MOTOR_PORT"]  = dl
                if dr: config["RIGHT_MOTOR_PORT"] = dr
            self.motors = DualMotors(
                config["LEFT_MOTOR_PORT"],  config["RIGHT_MOTOR_PORT"],
                config["LEFT_MOTOR_ID"],    config["RIGHT_MOTOR_ID"],
                config["LEFT_SIGN"],        config["RIGHT_SIGN"],
                config["TRIM"])
            print(f"[SLAM] Motors: "
                  f"L={config['LEFT_MOTOR_PORT']}(ID{config['LEFT_MOTOR_ID']}) "
                  f"R={config['RIGHT_MOTOR_PORT']}(ID{config['RIGHT_MOTOR_ID']})")

        # ── LiDAR ──
        self.lidar = LD06("/dev/ttyTHS1", offset=config["LIDAR_OFFSET"])
        self.lidar.start()
        print(f"[SLAM] LiDAR: offset={config['LIDAR_OFFSET']}°")
        time.sleep(0.5)

        # ── IMU ──
        self.imu = None
        try:
            self.imu = ICM20948()
            self.imu.init()
            print(f"[SLAM] IMU: ICM-20948 (mag={'yes' if self.imu.mag_ok else 'no'})")
        except Exception as e:
            print(f"[SLAM] IMU not available: {e}")

        # ── Camera ──
        self.camera = None
        try:
            self.camera = RealSenseCamera(max_depth=3.0)
            print("[SLAM] Camera: RealSense online")
        except Exception as e:
            print(f"[SLAM] Camera not available: {e}")

        # ── VFH ──
        self.vfh = VFHNavigator(self.VFH_BIN, self.VFH_THRESH, self.VFH_GAP)

        # ── Sensor-to-base transforms (with proper optical axis remap) ──
        self.T_base_cam = build_camera_transform(config)
        self.T_base_lid = build_lidar_transform(config)
        self.T_map_base = np.eye(4, dtype=np.float64)
        print(f"[SLAM] Camera transform: "
              f"pos=({config['CAMERA_X']:.2f},{config['CAMERA_Y']:.2f},{config['CAMERA_Z']:.2f})m  "
              f"yaw={config.get('CAMERA_YAW_DEG',0):.1f}° "
              f"pitch={config.get('CAMERA_PITCH_DEG',0):.1f}° "
              f"roll={config.get('CAMERA_ROLL_DEG',0):.1f}°")

        # ── Map state ──
        self.map_points  = np.empty((0, 3), dtype=np.float32)
        self.lidar_map   = np.empty((0, 3), dtype=np.float32)
        self.cam_map_xyz = np.empty((0, 3), dtype=np.float32)
        self.cam_map_rgb = np.empty((0, 3), dtype=np.uint8)
        self.robot_trail = []
        self.prev_cloud  = None
        self.frame_count = 0

        # ── Occupancy grid ──
        self.OCC_RES    = 0.05
        self.OCC_SIZE   = 600
        self.OCC_ORIGIN = np.array([-15.0, -15.0])
        self.log_odds   = np.zeros((self.OCC_SIZE, self.OCC_SIZE), dtype=np.float32)
        self.L_OCC = 0.85;  self.L_FREE = -0.42
        self.L_MAX = 5.0;   self.L_MIN  = -5.0

        # ── Avoidance state ──
        self.state        = "IDLE"
        self.state_detail = ""
        self.current_l    = self.current_r = 0
        self._front_history  = deque(maxlen=40)
        self._last_stuck_time = 0
        self._stuck_count    = 0
        self._last_turn_dir  = 0
        self._turn_persist   = 0

        # ── Dashboard buffers ──
        self._dash_data  = {}; self._dash_lock  = threading.Lock()
        self._occ_grid   = None; self._occ_lock = threading.Lock()
        self._cam_jpeg   = None; self._cam_lock = threading.Lock()
        self._map3d_data = None; self._map3d_lock = threading.Lock()

        # ── Live 3-D stream buffer (single raw frame in base frame) ──
        self._live_frame_xyz  = None   # Nx3 float32
        self._live_frame_rgb  = None   # Nx3 uint8
        self._live_frame_lock = threading.Lock()

        print("[SLAM] Ready!")

    # ── Motor helpers ─────────────────────────────────────────────
    def _drive(self, l, r):
        self.current_l = int(l); self.current_r = int(r)
        if self.motors: self.motors.drive(self.current_l, self.current_r)

    def _stop(self):
        self._drive(0, 0)

    def _brake(self):
        if self.motors: self.motors.brake()
        self.current_l = self.current_r = 0

    def _scale_speed(self, fd):
        if   fd <= self.STOP_DIST:   return 0
        elif fd <= self.SLOW_DIST:
            t = (fd - self.STOP_DIST) / (self.SLOW_DIST - self.STOP_DIST)
            return int(self.SLOW_SPEED + t * (self.CRUISE_SPEED - self.SLOW_SPEED))
        elif fd <= self.OPEN_DIST:
            t = (fd - self.SLOW_DIST) / (self.OPEN_DIST - self.SLOW_DIST)
            return int(self.CRUISE_SPEED + t * (self.MAX_SPEED - self.CRUISE_SPEED))
        return self.MAX_SPEED

    def _steer(self, target, speed):
        err = (target + 180) % 360 - 180
        s   = max(-1.0, min(1.0, err / 60.0 * 1.5))
        l   = max(-self.MAX_SPEED, min(self.MAX_SPEED, int(speed * (1 + s))))
        r   = max(-self.MAX_SPEED, min(self.MAX_SPEED, int(speed * (1 - s))))
        if speed > 0:
            if 0 < l < self.SLOW_SPEED: l = self.SLOW_SPEED
            if 0 < r < self.SLOW_SPEED: r = self.SLOW_SPEED
        self._drive(l, r)

    def _check_stuck(self, fd):
        self._front_history.append((time.time(), fd))
        if len(self._front_history) < 20: return False
        if time.time() - self._front_history[0][0] < self.STUCK_TIMEOUT: return False
        dists = [d for _, d in self._front_history]
        avg   = sum(dists) / len(dists)
        var   = sum((d - avg)**2 for d in dists) / len(dists)
        if var < 300 and avg < self.CRUISE_DIST:
            if time.time() - self._last_stuck_time > 5.0:
                self._last_stuck_time = time.time()
                self._stuck_count += 1
                return True
        return False

    def _unstuck(self):
        self.state = "UNSTUCK"
        self._drive(-self.REVERSE_SPEED, -self.REVERSE_SPEED)
        time.sleep(self.REVERSE_TIME * 1.5)
        d = random.choice([-1, 1])
        self._drive(d * self.TURN_SPEED, -d * self.TURN_SPEED)
        time.sleep(0.5 + random.random() * 0.8)
        self._front_history.clear()

    # ── Mapping tick ──────────────────────────────────────────────
    def _mapping_tick(self):
        """One mapping cycle: read sensors → update live stream → ICP → accumulate map."""
        from config import np
        self.frame_count += 1

        # IMU
        imu_data = None
        if self.imu:
            try: imu_data = self.imu.read_all()
            except Exception: pass

        # LiDAR → base frame
        scan    = self.lidar.get_scan()
        lid_3d  = lidar_to_3d(scan, height=0.0)   # Z=0 in sensor frame; T adds height
        lid_base = (transform_points(lid_3d, self.T_base_lid)
                    if len(lid_3d) > 0 else lid_3d)

        # Camera → base frame (with correct optical axis remap via T_base_cam)
        cam_base = np.empty((0, 3), dtype=np.float32)
        cam_rgb  = np.empty((0, 3), dtype=np.uint8)
        cam_img  = None
        if self.camera:
            try:
                cam_xyz, cam_colors, cam_img = self.camera.get_colored_points_and_image()
                if cam_xyz is not None and len(cam_xyz) > 0:
                    cam_base = transform_points(cam_xyz, self.T_base_cam)
                    cam_rgb  = cam_colors
            except Exception:
                pass

        # Live stream snapshot (base frame, up to 40k points)
        with self._live_frame_lock:
            if len(cam_base) > 0:
                step = max(1, len(cam_base) // 40000)
                self._live_frame_xyz = cam_base[::step].copy()
                self._live_frame_rgb = cam_rgb[::step].copy()
            else:
                self._live_frame_xyz = None
                self._live_frame_rgb = None

        # Downsample camera for map accumulation
        if len(cam_base) > 20000:
            step    = len(cam_base) // 20000
            cam_base = cam_base[::step]
            cam_rgb  = cam_rgb[::step]

        # ICP odometry
        parts = [p for p in [cam_base, lid_base] if len(p) > 0]
        if not parts:
            return scan, imu_data, cam_img
        fused = np.vstack(parts) if len(parts) > 1 else parts[0]
        fused = voxel_downsample(fused, 0.03)

        if self.prev_cloud is not None and len(fused) > 50:
            try:
                dT, fitness = numpy_icp(fused, self.prev_cloud, max_dist=0.5)
                if fitness > 0.3:
                    self.T_map_base = self.T_map_base @ np.linalg.inv(dT)
            except Exception:
                pass
        self.prev_cloud = fused.copy()

        # Robot trail
        pos = self.T_map_base[:3, 3]
        if (not self.robot_trail or
                math.hypot(pos[0] - self.robot_trail[-1][0],
                           pos[1] - self.robot_trail[-1][1]) > 0.05):
            self.robot_trail.append([float(pos[0]), float(pos[1])])
            if len(self.robot_trail) > 5000:
                self.robot_trail = self.robot_trail[-5000:]

        # Accumulate LiDAR wall map
        if len(lid_base) > 0:
            lid_world = transform_points(lid_base, self.T_map_base)
            self.lidar_map = (np.vstack([self.lidar_map, lid_world])
                              if len(self.lidar_map) > 0 else lid_world)

        # Accumulate camera coloured map
        if len(cam_base) > 0:
            cam_world = transform_points(cam_base, self.T_map_base)
            self.cam_map_xyz = (np.vstack([self.cam_map_xyz, cam_world])
                                if len(self.cam_map_xyz) > 0 else cam_world)
            self.cam_map_rgb = (np.vstack([self.cam_map_rgb, cam_rgb])
                                if len(self.cam_map_rgb) > 0 else cam_rgb)

        parts_all = [p for p in [self.lidar_map, self.cam_map_xyz] if len(p) > 0]
        self.map_points = (np.vstack(parts_all) if len(parts_all) > 1
                           else (parts_all[0] if parts_all
                                 else np.empty((0, 3), dtype=np.float32)))

        # Periodic voxel downsample
        if self.frame_count % 20 == 0:
            if len(self.lidar_map) > 0:
                self.lidar_map = voxel_downsample(self.lidar_map, self.MAP_VOXEL)
            if len(self.cam_map_xyz) > 0:
                keys = (self.cam_map_xyz / self.MAP_VOXEL).astype(np.int64)
                hashes = keys[:,0]*73856093 ^ keys[:,1]*19349669 ^ keys[:,2]*83492791
                _, idx = np.unique(hashes, return_index=True)
                self.cam_map_xyz = self.cam_map_xyz[idx]
                self.cam_map_rgb = self.cam_map_rgb[idx]

        # Cap sizes
        if len(self.lidar_map) > 200_000:
            self.lidar_map = voxel_downsample(self.lidar_map, self.MAP_VOXEL * 1.5)
        if len(self.cam_map_xyz) > 300_000:
            keys = (self.cam_map_xyz / (self.MAP_VOXEL * 1.5)).astype(np.int64)
            hashes = keys[:,0]*73856093 ^ keys[:,1]*19349669 ^ keys[:,2]*83492791
            _, idx = np.unique(hashes, return_index=True)
            self.cam_map_xyz = self.cam_map_xyz[idx]
            self.cam_map_rgb = self.cam_map_rgb[idx]

        # Occupancy grid
        if scan:
            self._update_occupancy_from_scan(scan)
        if self.frame_count % 10 == 0:
            with self._occ_lock:
                self._occ_grid = self._get_occupancy_for_dashboard()

        # Camera JPEG
        if cam_img is not None:
            try:
                _import_cv2()
                from config import cv2
                _, buf = cv2.imencode('.jpg', cam_img, [cv2.IMWRITE_JPEG_QUALITY, 60])
                with self._cam_lock:
                    self._cam_jpeg = buf.tobytes()
            except Exception:
                pass

        # 3-D dashboard data
        if self.frame_count % 15 == 0:
            self._prepare_3d_dashboard_data()

        return scan, imu_data, cam_img

    # ── Occupancy grid helpers ────────────────────────────────────
    def _update_occupancy_from_scan(self, scan):
        rx  = float(self.T_map_base[0, 3])
        ry  = float(self.T_map_base[1, 3])
        yaw = float(math.atan2(self.T_map_base[1, 0], self.T_map_base[0, 0]))
        res = self.OCC_RES
        ox, oy = self.OCC_ORIGIN[0], self.OCC_ORIGIN[1]
        sz  = self.OCC_SIZE
        rx_g = int((rx - ox) / res)
        ry_g = int((ry - oy) / res)
        if not (0 <= rx_g < sz and 0 <= ry_g < sz): return

        for angle_deg, dist_mm in scan.items():
            if dist_mm <= 10 or dist_mm > 8000: continue
            angle_rad = yaw - math.radians(float(angle_deg))
            d = dist_mm / 1000.0
            hx_g = int((rx + d*math.cos(angle_rad) - ox) / res)
            hy_g = int((ry + d*math.sin(angle_rad) - oy) / res)
            self._bresenham_ray(rx_g, ry_g, hx_g, hy_g, sz)

    def _bresenham_ray(self, x0, y0, x1, y1, sz):
        dx = abs(x1-x0); dy = abs(y1-y0)
        sx = 1 if x0<x1 else -1; sy = 1 if y0<y1 else -1
        err = dx - dy; cx, cy = x0, y0
        lo  = self.log_odds
        while True:
            if 0<=cx<sz and 0<=cy<sz:
                if cx==x1 and cy==y1:
                    lo[cy,cx] = min(self.L_MAX, lo[cy,cx] + self.L_OCC)
                else:
                    lo[cy,cx] = max(self.L_MIN, lo[cy,cx] + self.L_FREE)
            if cx==x1 and cy==y1: break
            e2 = 2*err
            if e2>-dy: err-=dy; cx+=sx
            if e2< dx: err+=dx; cy+=sy

    def _get_occupancy_for_dashboard(self):
        from config import np
        sz   = self.OCC_SIZE
        grid = np.full(sz*sz, -1, dtype=np.int8)
        flat = self.log_odds.flatten()
        grid[flat < -0.4] = 0
        grid[flat >  0.4] = 100
        grid[(flat >= -0.4) & (flat <= 0.4) & (flat != 0.0)] = 50
        return {
            "grid":   grid.tolist(),
            "res":    self.OCC_RES,
            "origin": self.OCC_ORIGIN.tolist(),
            "size":   sz,
            "trail":  self.robot_trail[-1000:],
            "robot":  [float(self.T_map_base[0,3]), float(self.T_map_base[1,3])],
            "yaw":    float(math.atan2(self.T_map_base[1,0], self.T_map_base[0,0])),
        }

    # ── 3-D dashboard data ────────────────────────────────────────
    def _prepare_3d_dashboard_data(self):
        from config import np
        data = {"lidar": None, "lidar_loop": None, "camera": None}

        if len(self.lidar_map) > 0:
            ds = voxel_downsample(self.lidar_map, 0.12)
            if len(ds) > 6000: ds = ds[::len(ds)//6000]
            WALL_BOTTOM = -0.05; WALL_TOP = 2.20; WALL_STEPS = 8
            heights  = [WALL_BOTTOM + (WALL_TOP-WALL_BOTTOM)*k/(WALL_STEPS-1)
                        for k in range(WALL_STEPS)]
            extruded = []
            for pt in ds:
                for h in heights:
                    extruded.append([round(float(pt[0]),3),
                                     round(float(pt[1]),3),
                                     round(h,3)])
            data["lidar"] = extruded

            if len(ds) > 2:
                cx = float(ds[:,0].mean()); cy = float(ds[:,1].mean())
                order = np.argsort(np.arctan2(ds[:,1]-cy, ds[:,0]-cx))
                loop  = [[round(float(p[0]),3),round(float(p[1]),3),round(float(p[2]),3)]
                         for p in ds[order]]
                if loop: loop.append(loop[0])
                data["lidar_loop"] = loop

        if len(self.cam_map_xyz) > 0:
            step = max(1, len(self.cam_map_xyz) // 25000)
            xyz  = self.cam_map_xyz[::step]
            rgb  = self.cam_map_rgb[::step]
            data["camera"] = [
                [round(float(xyz[i,0]),3), round(float(xyz[i,1]),3), round(float(xyz[i,2]),3),
                 int(rgb[i,0]), int(rgb[i,1]), int(rgb[i,2])]
                for i in range(len(xyz))
            ]

        with self._map3d_lock:
            self._map3d_data = data

    # ── Public data accessors (used by server.py) ─────────────────
    def _update_dash(self, scan, hist, zones, target, gap, imu_data):
        pos = self.T_map_base[:3,3].tolist()
        with self._dash_lock:
            self._dash_data = {
                "points":         {str(k): v for k, v in scan.items()},
                "histogram":      hist,
                "zones":          zones,
                "state":          self.state,
                "detail":         self.state_detail,
                "target_heading": target if target is not None else -1,
                "gap":            gap,
                "left_rpm":       self.current_l,
                "right_rpm":      self.current_r,
                "stuck_count":    self._stuck_count,
                "map_points":     len(self.map_points),
                "lidar_pts":      len(self.lidar_map),
                "cam_pts":        len(self.cam_map_xyz),
                "frame":          self.frame_count,
                "position":       pos,
                "imu": {
                    "roll":  math.degrees(imu_data["roll"]),
                    "pitch": math.degrees(imu_data["pitch"]),
                    "yaw":   math.degrees(imu_data["yaw"]),
                    "temp":  imu_data["temp"],
                } if imu_data else None,
            }

    def get_dash_data(self):
        with self._dash_lock:   return self._dash_data.copy()
    def get_occ_grid(self):
        with self._occ_lock:    return self._occ_grid
    def get_cam_jpeg(self):
        with self._cam_lock:    return self._cam_jpeg
    def get_map3d(self):
        with self._map3d_lock:  return self._map3d_data

    def get_live_frame(self):
        """Returns (xyz Nx3 float32, rgb Nx3 uint8) or (None, None)."""
        with self._live_frame_lock:
            if self._live_frame_xyz is None: return None, None
            return self._live_frame_xyz.copy(), self._live_frame_rgb.copy()

    # ── Main loop ─────────────────────────────────────────────────
    def run(self):
        self._running = True
        mode = 'Map-only mode' if self.map_only else 'Avoidance + Mapping'
        print(f"\n[SLAM] {mode} started! Ctrl+C to stop.\n")

        try:
            while self._running:
                scan, imu_data, cam_img = self._mapping_tick()
                if not scan:
                    self.state = "WAIT_LIDAR"
                    self._stop(); time.sleep(0.1); continue

                hist   = self.lidar.get_histogram(self.VFH_BIN, 6000)
                zones  = self.lidar.get_zone_distances(scan)
                f  = zones["FRONT"]; fl = zones["FRONT_LEFT"]; fr = zones["FRONT_RIGHT"]
                fmin = min(f, fl, fr)
                target, gap = self.vfh.best_heading(hist, 0.0)
                self._update_dash(scan, hist, zones, target, gap, imu_data)

                if self.map_only:
                    self.state        = "MAPPING"
                    self.state_detail = f"pts={len(self.map_points)} f={self.frame_count}"
                    time.sleep(self.LOOP_RATE)
                    self._print_status(zones, target, gap)
                    continue

                if self._check_stuck(fmin):
                    self._unstuck(); continue

                if fmin < self.EMERGENCY_DIST:
                    self.state        = "EMERGENCY"
                    self.state_detail = f"fmin={fmin}mm"
                    self._drive(-self.REVERSE_SPEED, -self.REVERSE_SPEED)
                    time.sleep(self.REVERSE_TIME)
                    if fl < fr: self._drive( self.TURN_SPEED, -self.TURN_SPEED)
                    else:       self._drive(-self.TURN_SPEED,  self.TURN_SPEED)
                    time.sleep(self.TURN_TIME)
                    self._front_history.clear()
                    continue

                elif target is None:
                    self.state = "TRAPPED"
                    d = self._last_turn_dir or 1
                    self._drive(d*self.TURN_SPEED, -d*self.TURN_SPEED)

                elif fmin < self.STOP_DIST:
                    self.state = "STOP_TURN"
                    herr = (target + 180) % 360 - 180
                    if self._turn_persist > 0:
                        self._turn_persist -= 1; d = self._last_turn_dir
                    else:
                        d = 1 if herr > 0 else -1
                        self._last_turn_dir = d; self._turn_persist = 8
                    self.state_detail = f"tgt={target:.0f}°"
                    self._drive(d*self.TURN_SPEED, -d*self.TURN_SPEED)

                else:
                    speed = self._scale_speed(fmin)
                    herr  = abs((target + 180) % 360 - 180)
                    if   herr < 10: self.state = "FORWARD"
                    elif herr < 45: self.state = "STEER"
                    else:           self.state = "TURN_GAP"; speed = max(speed, self.SLOW_SPEED)
                    self.state_detail = f"→{target:.0f}° {speed}rpm f={fmin}mm"
                    self._steer(target, speed)
                    if herr < 20: self._turn_persist = 0

                self._print_status(zones, target, gap)
                time.sleep(self.LOOP_RATE)

        except KeyboardInterrupt:
            print("\n[SLAM] Stopping…")
        finally:
            self._brake()
            self._save_map()

    def _print_status(self, zones, target, gap):
        f = zones["FRONT"]; fl = zones["FRONT_LEFT"]; fr = zones["FRONT_RIGHT"]
        tgt = f"{target:.0f}°" if target else "---"
        gw  = f"{gap['width']:.0f}°" if gap else "---"
        mp  = len(self.map_points)
        sys.stdout.write(
            f"\r[{self.state:<12}] {self.state_detail:<24} "
            f"F:{f:5d} FL:{fl:5d} FR:{fr:5d} "
            f"Tgt:{tgt:>5} Gap:{gw:>5} "
            f"L:{self.current_l:+4d} R:{self.current_r:+4d} "
            f"Map:{mp:6d} Frm:{self.frame_count}  "
        )
        sys.stdout.flush()

    def _save_map(self):
        from config import np
        sz  = self.OCC_SIZE; res = self.OCC_RES
        ox, oy = self.OCC_ORIGIN
        pgm_path  = os.path.expanduser("~/slam_map.pgm")
        yaml_path = os.path.expanduser("~/slam_map.yaml")
        pgm = np.full((sz, sz), 205, dtype=np.uint8)
        pgm[self.log_odds < -0.4] = 254
        pgm[self.log_odds >  0.4] = 0
        with open(pgm_path, 'wb') as f:
            f.write(f"P5\n{sz} {sz}\n255\n".encode())
            f.write(np.flipud(pgm).tobytes())
        print(f"\n[SLAM] Map image → {pgm_path}")
        with open(yaml_path, 'w') as f:
            f.write(f"image: slam_map.pgm\nresolution: {res}\n")
            f.write(f"origin: [{ox}, {oy}, 0.0]\nnegate: 0\n")
            f.write("occupied_thresh: 0.65\nfree_thresh: 0.196\n")
        print(f"[SLAM] Map YAML  → {yaml_path}")
        combined = [p for p in [self.lidar_map, self.cam_map_xyz] if len(p) > 0]
        if combined:
            all_pts  = np.vstack(combined)
            pcd_path = os.path.expanduser("~/slam_map.pcd")
            save_pcd(pcd_path, all_pts)
            print(f"[SLAM] 3D cloud  → {pcd_path} ({len(all_pts)} pts)")
        if len(self.lidar_map)   > 0: np.save(os.path.expanduser("~/slam_lidar.npy"),      self.lidar_map)
        if len(self.cam_map_xyz) > 0:
            np.save(os.path.expanduser("~/slam_camera.npy"),     self.cam_map_xyz)
            np.save(os.path.expanduser("~/slam_camera_rgb.npy"), self.cam_map_rgb)

    def close(self):
        self._running = False
        self._brake()
        if self.motors: self.motors.close()
        self.lidar.close()
        if self.imu:
            try: self.imu.close()
            except Exception: pass
        if self.camera:
            try: self.camera.close()
            except Exception: pass
        print("[SLAM] Shutdown complete")