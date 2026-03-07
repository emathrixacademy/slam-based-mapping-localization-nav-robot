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
import contextlib
from collections import deque

from config import (
    _import_numpy, _import_cv2, _import_o3d, _import_scipy, _import_cupy,
    build_camera_transform, build_lidar_transform,
)
from motors import DualMotors, probe_motor_ports
from lidar  import (
    LD06, lidar_to_3d, transform_points,
    voxel_downsample, voxel_downsample_idx, numpy_icp, save_pcd,
)
from slam import HectorSLAM2D, RGBDOdometry, LoopClosureDetector
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
    CRUISE_DIST    = 2500
    OPEN_DIST      = 5000

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
            self.camera = RealSenseCamera(max_depth=6.0)
            print("[SLAM] Camera: RealSense online")
        except Exception as e:
            print(f"[SLAM] Camera not available: {e}")

        # ── Camera frame buffer (background capture thread) ──
        # get_full_frame() blocks ~67 ms waiting for the sensor — running it in a
        # dedicated thread lets the main loop tick at LiDAR rate (~20 Hz) instead
        # of being throttled to camera rate (15 fps).
        self._cam_latest_xyz     = None   # last captured xyz (camera optical frame)
        self._cam_latest_img     = None   # last captured BGR image
        self._cam_latest_depth_m = None   # last captured depth map (HxW float32)
        self._cam_latest_yaw     = 0.0    # robot heading (rad) at camera capture time
        self._cam_frame_lock     = threading.Lock()

        # ── VFH ──
        self.vfh = VFHNavigator(self.VFH_BIN, self.VFH_THRESH, self.VFH_GAP)

        # ── Sensor-to-base transforms (with proper optical axis remap) ──
        self.T_base_cam = build_camera_transform(config)
        self.T_base_lid = build_lidar_transform(config)
        self.T_map_base = np.eye(4, dtype=np.float64)

        # ── IMU-assisted heading ──
        # _map_yaw: robot heading in map frame (radians), updated from IMU yaw each tick.
        # T_map_base rotation is kept consistent with this so the map stays floor-fixed
        # even during turns where ICP fails.
        self._map_yaw       = 0.0   # current heading in map frame (rad)
        self._imu_yaw_ref   = None  # IMU yaw reading at last heading reset
        self._imu_yaw_base  = 0.0   # map_yaw at last heading reset
        print(f"[SLAM] Camera transform: "
              f"pos=({config['CAMERA_X']:.2f},{config['CAMERA_Y']:.2f},{config['CAMERA_Z']:.2f})m  "
              f"yaw={config.get('CAMERA_YAW_DEG',0):.1f}° "
              f"pitch={config.get('CAMERA_PITCH_DEG',0):.1f}° "
              f"roll={config.get('CAMERA_ROLL_DEG',0):.1f}°")

        # ── Hector SLAM 2D (primary odometry) ─────────────────────
        self.hector         = HectorSLAM2D(res=0.05, size=600, levels=3)
        self._hector_frames = 0   # bootstrap guard (needs ≥5 map updates)

        # ── RGB-D Visual Odometry (backup / Z-axis) ───────────────
        self.rgbd_odom = None
        if self.camera:
            try:
                fx, fy, cx, cy_, w, h = self.camera.get_intrinsics()
                self.rgbd_odom = RGBDOdometry(w, h, fx, fy, cx, cy_,
                                              depth_trunc=3.0)
                print(f"[SLAM] RGB-D odometry: {w}×{h} "
                      f"fx={fx:.1f} fy={fy:.1f}")
            except Exception as e:
                print(f"[SLAM] RGB-D odometry unavailable: {e}")

        # ── Loop closure (runs in background) ─────────────────────
        self.loop_det         = LoopClosureDetector()
        self._loop_building   = False
        self._loop_correction = None   # np.array[dx, dy, dyaw] or None
        self._loop_lock       = threading.Lock()

        # ── TSDF 3-D volume fusion (replaces BPA on accumulated points) ─
        self._tsdf           = None
        self._tsdf_intrinsic = None
        self._tsdf_lock      = threading.Lock()
        # TSDF (ScalableTSDFVolume) disabled — uses CUDA internally and
        # segfaults on Jetson Orin Nano when GPU memory is under pressure.
        # BPA full-rebuild is used for meshing instead.
        self._tsdf           = None
        self._tsdf_intrinsic = None

        # ── Map state ──
        self.map_points  = np.empty((0, 3), dtype=np.float32)
        self.lidar_map   = np.empty((0, 3), dtype=np.float32)
        self.cam_map_xyz   = np.empty((0, 3), dtype=np.float32)
        self.cam_map_depth = np.empty((0,),   dtype=np.float32)  # raw optical-Z depth per point
        self.robot_trail = []
        self.frame_count = 0

        # ── Occupancy grid (60 m × 60 m @ 2.5 cm/cell) ──
        self.OCC_RES    = 0.025
        self.OCC_SIZE   = 2400
        self.OCC_ORIGIN = np.array([-30.0, -30.0])
        from config import cuda_ok
        if cuda_ok:
            import cupy as cp
            self.log_odds = cp.zeros((self.OCC_SIZE, self.OCC_SIZE), dtype=cp.float32)
            self._gpu_occ = True
            print("[SLAM] Occupancy grid: GPU (CuPy — persistent, zero-copy UMA)")
        else:
            self.log_odds = np.zeros((self.OCC_SIZE, self.OCC_SIZE), dtype=np.float32)
            self._gpu_occ = False
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

        # ── Perimeter exploration ──────────────────────────────────
        # Phase 1: stay still for INIT_SCAN_FRAMES to build the initial room map.
        # Phase 2: navigate generated waypoints around the inner edge of the room.
        self.INIT_SCAN_FRAMES = 5       # ~0.25 s at 20 Hz — enough for stable map
        _rl = config.get("ROBOT_LENGTH", 0.370)
        _rw = config.get("ROBOT_WIDTH",  0.305)
        self.robot_length = _rl
        self.robot_width  = _rw
        # WALL_MARGIN = robot half-diagonal + 0.15 m safety buffer
        import math as _m
        self.WALL_MARGIN  = round(_m.sqrt((_rl/2)**2 + (_rw/2)**2) + 0.15, 3)
        self.WP_REACH_DIST = max(0.30, _rw / 2 + 0.10)   # metres — waypoint considered reached
        self.WP_TIMEOUT       = 25.0    # seconds — skip stuck waypoint
        self._perim_waypoints  = []     # [wx, wy] list in world frame
        self._perim_wp_idx     = 0
        self._perim_done       = False
        self._perim_wp_t0      = 0.0    # time we started heading for current wp

        # ── Dashboard buffers ──
        self._dash_data  = {}; self._dash_lock  = threading.Lock()
        self._occ_grid   = None; self._occ_lock = threading.Lock()
        self._cam_jpeg   = None; self._cam_lock   = threading.Lock()
        self._depth_jpeg = None; self._depth_lock = threading.Lock()
        self._map3d_data = None; self._map3d_lock = threading.Lock()

        # ── Mesh reconstruction buffer ──
        # Rebuilt incrementally in background every MESH_INTERVAL frames.
        self._mesh_data      = None
        self._mesh_lock      = threading.Lock()
        self._mesh_building  = False
        self.MESH_INTERVAL   = 20      # check for new geometry every N frames

        # Accumulated mesh — BPA chunks from each new region, merged as robot explores
        self._acc_mesh_verts  = np.empty((0, 3), dtype=np.float32)
        self._acc_mesh_faces  = np.empty((0, 3), dtype=np.int32)
        self._acc_mesh_colors = np.empty((0, 3), dtype=np.float32)
        # Voxel tracking: packed int64 keys of world cells already BPA'd (0.05 m grid)
        # Finer than before → smaller patches → more detail per BPA chunk
        self.TRACK_VOXEL    = 0.05
        self._meshed_voxels = np.empty(0, dtype=np.int64)   # sorted
        # Maximum optical-Z depth (metres) for meshing.  Set to camera max_depth so
        # walls at 2–5 m are included.  Previously 1.5 m silently excluded far walls.
        self.MESH_MAX_DIST  = 5.0

        # ── IMU heading prediction state ──
        self._last_tick_time = time.time()   # for computing dt between ticks
        # Hector motion gate: only update map when robot has moved ≥3 cm or ≥2°
        self._last_map_update_pose = np.array([0.0, 0.0, 0.0])  # [x, y, yaw]
        # Stale-scan guard: track last lidar scan timestamp to detect lidar hangs
        self._last_lid_scan_ts   = 0.0
        self._stale_warn_printed = False
        # Scan motion de-skew: correct per-point heading error during rotation
        self._prev_map_yaw    = 0.0   # heading at end of previous tick
        self._prev_yaw_rate   = 0.0   # rad/s yaw-rate estimate (from previous tick)
        self._prev_lid_scan_ts = 0.0  # scan_ts of previous fresh scan (for T_scan)

        # ── Live 3-D stream buffer (single raw frame in base frame) ──
        self._live_frame_xyz   = None   # Nx3 float32
        self._live_frame_depth = None   # N   float32  (optical-Z depth, metres)
        self._live_frame_lock  = threading.Lock()

        # ── Pre-serialised stream3d payload (served directly by HTTP thread) ──
        self._stream3d_cache      = b'{}'
        self._stream3d_cache_lock = threading.Lock()

        # ── GPU acceleration (CuPy + Open3D CUDA) ──
        _import_cupy()
        _import_o3d()   # also checks o3d_cuda_ok
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

    # ── LiDAR scan motion de-skew ─────────────────────────────────
    def _deskew_scan(self, scan_dict, omega_z, T_scan):
        """
        Correct each scan point for the robot rotation that occurred DURING the scan.

        The LD06 spins CW at ~10 Hz: a full revolution takes T_scan seconds.
        A point at CW angle α (0–360°) was captured (1 – α/360) × T_scan seconds
        before the scan completed.  If the robot was rotating at omega_z rad/s
        (CCW positive) it had a DIFFERENT heading at capture time.

        Correction: rotate each point by  –omega_z × (1 – α/360) × T_scan
        so all points are expressed in the sensor frame at scan-end time.
        """
        from config import np
        if not scan_dict:
            return scan_dict
        angles = np.fromiter(scan_dict.keys(),   dtype=np.float32, count=len(scan_dict))
        dists  = np.fromiter(scan_dict.values(), dtype=np.float32, count=len(scan_dict))

        # Time elapsed before scan end for each point (s)
        t_before = (1.0 - angles / 360.0) * np.float32(T_scan)

        # Per-point correction angle (CCW radians): cancel the heading change
        corr = (-omega_z * t_before).astype(np.float32)

        # Sensor-frame XY  (CW convention: x=forward, y=left)
        d   = dists * np.float32(1e-3)       # mm → m
        ar  = np.deg2rad(angles)
        px  =  d * np.cos(ar)
        py  = -d * np.sin(ar)

        # Rotate each point back to capture-time sensor frame
        cos_c, sin_c = np.cos(corr), np.sin(corr)
        px_c = px * cos_c - py * sin_c
        py_c = px * sin_c + py * cos_c

        # Back to CW polar
        new_d_mm = np.hypot(px_c, py_c) * 1000.0
        new_ang  = np.degrees(np.arctan2(-py_c, px_c)) % 360.0
        return {round(float(a), 1): round(float(d))
                for a, d in zip(new_ang, new_d_mm)}

    # ── Background camera capture ─────────────────────────────────
    def _camera_loop(self):
        """
        Dedicated thread: calls get_full_frame() continuously so the main loop
        never blocks waiting for the camera sensor (~67 ms/frame at 15 fps).
        The latest frame is stored in _cam_latest_* and read by _mapping_tick.
        """
        while self._running:
            if self.camera is None:
                break
            try:
                xyz, _rgb, img, depth_m = self.camera.get_full_frame()
                if xyz is not None:
                    with self._cam_frame_lock:
                        self._cam_latest_xyz     = xyz
                        self._cam_latest_img     = img
                        self._cam_latest_depth_m = depth_m
                        self._cam_latest_yaw     = self._map_yaw  # snapshot heading at capture time
            except Exception:
                time.sleep(0.05)

    # ── GPU-accelerated transform (main-thread only, safe on Jetson UMA) ────
    def _xform(self, pts, T):
        """
        transform_points with CuPy acceleration for large arrays.
        On Jetson Orin Nano the CPU/GPU share unified memory — cp.asarray /
        cp.asnumpy are near-zero-cost remaps, not DMA copies.
        Only use from the MAIN thread; background threads must call transform_points().
        """
        from config import cuda_ok
        if cuda_ok and len(pts) > 1000:
            import cupy as cp
            pts_cp = cp.asarray(pts, dtype=cp.float32)
            T_cp   = cp.asarray(T,   dtype=cp.float32)
            ones   = cp.ones((len(pts_cp), 1), dtype=cp.float32)
            hom    = cp.hstack([pts_cp, ones])
            result = (T_cp[:3, :] @ hom.T).T
            return cp.asnumpy(result)
        return transform_points(pts, T)

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

        # LiDAR → base frame, then filter robot-body reflections (< 150 mm)
        scan         = self.lidar.get_scan()
        _lid_scan_ts = self.lidar._scan_ts
        _lid_fresh   = _lid_scan_ts > self._last_lid_scan_ts
        if _lid_fresh:
            self._last_lid_scan_ts   = _lid_scan_ts
            self._stale_warn_printed = False
        else:
            # Lidar has not produced a new scan since the last tick.
            # DO NOT clear scan — the main loop checks `if not scan` and would
            # enter WAIT_LIDAR (stopping motors + all dashboard updates) if empty.
            # Instead keep scan for navigation/display and use _lid_fresh=False
            # to skip only the Hector registration and map accumulation below.
            if not self._stale_warn_printed:
                age = self.lidar.get_scan_age()
                print(f"[LiDAR] WARNING: stale scan ({age:.2f}s old) — "
                      f"skipping Hector/map update until fresh data arrives")
                self._stale_warn_printed = True

        # ── Scan motion de-skew ────────────────────────────────────
        # Best angular-velocity source: IMU gyro Z (direct measurement).
        # Fallback: Hector yaw-rate estimate from the previous tick.
        _omega_z = self._prev_yaw_rate
        if imu_data is not None:
            _gz = float(imu_data["gyro"][2]) * self.config.get("IMU_GYRO_SIGN", 1.0)
            if abs(_gz) > 0.02:   # 1.1°/s low deadband — de-skew benefits even slow turns
                _omega_z = _gz
        if _lid_fresh and scan and abs(_omega_z) > math.radians(1.0):
            # T_scan from consecutive scan timestamps (real period, not fixed 0.1s)
            _T_scan = (_lid_scan_ts - self._prev_lid_scan_ts
                       if self._prev_lid_scan_ts > 0
                       and 0.05 < _lid_scan_ts - self._prev_lid_scan_ts < 0.30
                       else 0.10)
            scan = self._deskew_scan(scan, _omega_z, _T_scan)
        if _lid_fresh:
            self._prev_lid_scan_ts = _lid_scan_ts

        lid_3d   = lidar_to_3d(scan, height=0.0)
        lid_base = (transform_points(lid_3d, self.T_base_lid)
                    if len(lid_3d) > 0 else lid_3d)
        if len(lid_base) > 0:
            horiz = np.linalg.norm(lid_base[:, :2], axis=1)
            keep  = horiz >= 0.15
            lid_base = lid_base[keep]

        # Camera → base frame (read latest frame from background capture thread)
        cam_base  = np.empty((0, 3), dtype=np.float32)
        cam_depth = np.empty((0,),   dtype=np.float32)  # optical-Z depth (metres)
        cam_img   = None
        depth_m   = None   # raw depth in metres for RGB-D odometry
        cam_yaw = self._map_yaw   # heading at camera capture time (updated below)
        if self.camera:
            with self._cam_frame_lock:
                cam_xyz = self._cam_latest_xyz
                cam_img = self._cam_latest_img
                depth_m = self._cam_latest_depth_m
                cam_yaw = self._cam_latest_yaw   # heading when this frame was captured
            if cam_xyz is not None and len(cam_xyz) > 0:
                cam_base  = self._xform(cam_xyz, self.T_base_cam)
                cam_depth = cam_xyz[:, 2].astype(np.float32)  # Z = depth in optical frame

        # Downsample camera for map accumulation (also caps live-view bandwidth)
        if len(cam_base) > 40000:
            step      = len(cam_base) // 40000
            cam_base  = cam_base[::step]
            cam_depth = cam_depth[::step]

        # Must have at least LiDAR or camera to proceed
        if len(lid_base) == 0 and len(cam_base) == 0:
            return scan, imu_data, cam_img

        lid_world = np.empty((0, 3), dtype=np.float32)   # filled after pose update

        # ── RGB-D Visual Odometry (6-DOF backup / Z-axis) ─────────
        # Skip during fast rotation: if heading changed >5° since camera capture,
        # the frame is too stale for reliable odometry and would cause drift.
        _yaw_delta_cam = abs(self._map_yaw - cam_yaw)
        _fast_rotation = _yaw_delta_cam > math.radians(5.0)
        T_rgbd_cam = np.eye(4, dtype=np.float64)
        rgbd_ok    = False
        if (self.rgbd_odom is not None and depth_m is not None
                and cam_img is not None and not _fast_rotation):
            try:
                T_rgbd_cam, rgbd_ok = self.rgbd_odom.push_frame(cam_img, depth_m)
            except Exception:
                pass

        # ── IMU gyro heading prediction ────────────────────────────
        # (1) Seeds Hector with a better starting yaw before registration
        # (2) Acts as fallback pose update when Hector fitness is low
        now = time.time()
        dt  = min(now - self._last_tick_time, 0.5)   # cap at 500 ms (sanity)
        self._last_tick_time = now
        imu_predicted_yaw = None
        if imu_data is not None and dt > 0:
            gz  = float(imu_data["gyro"][2])
            gz *= self.config.get("IMU_GYRO_SIGN", 1.0)
            if abs(gz) < 0.03:    # 1.7°/s deadband — wider to kill bias drift at rest
                gz = 0.0
            imu_predicted_yaw = self._map_yaw + gz * dt
            if self._hector_frames >= 5:
                self.hector.set_pose(
                    float(self.T_map_base[0, 3]),
                    float(self.T_map_base[1, 3]),
                    imu_predicted_yaw)

        # ── Hector SLAM 2D (primary odometry: scan-to-map) ────────
        # Only register a fresh scan — re-registering the same scan causes drift.
        h_fitness = 0.0
        if _lid_fresh and len(lid_base) >= 20 and self._hector_frames >= 5:
            scan_2d   = lid_base[:, :2].astype(np.float64)
            h_fitness = self.hector.register(scan_2d)

        # ── Apply best available pose update ──────────────────────
        if h_fitness > 0.35:
            # Hector SLAM wins: update T_map_base from 2D pose
            hx, hy, hyaw = self.hector.get_pose()
            _prev_x = float(self.T_map_base[0, 3])
            _prev_y = float(self.T_map_base[1, 3])
            _moved  = math.hypot(hx - _prev_x, hy - _prev_y)
            _turned = abs(hyaw - self._map_yaw)
            # Reject impossible jumps — robot physically can't move >6 cm per tick.
            # Hector drifting while stationary shows as large per-tick jumps → reject.
            # Exception: when Hector is very confident (fitness > 0.55) allow up to
            # 25 cm — this lets Hector snap back to the map when the robot returns
            # to a previously visited area with accumulated drift.
            _max_jump = 0.25 if h_fitness > 0.55 else 0.06
            if _moved > _max_jump:
                self.hector.set_pose(_prev_x, _prev_y, self._map_yaw)
            # Accept only meaningful updates — filter sub-5mm/sub-0.5° scan jitter
            elif _moved > 0.005 or _turned > math.radians(0.5):
                cy_, sy_ = math.cos(hyaw), math.sin(hyaw)
                self.T_map_base[0, 0] =  cy_; self.T_map_base[0, 1] = -sy_
                self.T_map_base[1, 0] =  sy_; self.T_map_base[1, 1] =  cy_
                self.T_map_base[0, 3] =  hx;  self.T_map_base[1, 3] =  hy
                self._map_yaw = hyaw
        elif imu_predicted_yaw is not None:
            # IMU fallback: only apply if yaw actually changed (gz passed deadband)
            # This prevents gyro bias from drifting the pose when stationary
            dyaw = abs(imu_predicted_yaw - self._map_yaw)
            if dyaw > math.radians(1.0):   # must turn > 1° to trigger IMU update
                hyaw = imu_predicted_yaw
                cy_, sy_ = math.cos(hyaw), math.sin(hyaw)
                self.T_map_base[0, 0] =  cy_; self.T_map_base[0, 1] = -sy_
                self.T_map_base[1, 0] =  sy_; self.T_map_base[1, 1] =  cy_
                self._map_yaw = hyaw
                self.hector.set_pose(float(self.T_map_base[0, 3]),
                                     float(self.T_map_base[1, 3]), hyaw)
        elif rgbd_ok:
            # RGB-D backup: camera-frame delta → base-frame delta
            T_cam_base   = np.linalg.inv(self.T_base_cam)
            T_base_delta = self.T_base_cam @ np.linalg.inv(T_rgbd_cam) @ T_cam_base
            self.T_map_base = self.T_map_base @ T_base_delta
            self._map_yaw   = float(math.atan2(
                self.T_map_base[1, 0], self.T_map_base[0, 0]))
            self.hector.set_pose(float(self.T_map_base[0, 3]),
                                 float(self.T_map_base[1, 3]), self._map_yaw)

        # ── Apply pending loop-closure correction (from background) ─
        with self._loop_lock:
            corr = self._loop_correction
            self._loop_correction = None
        if corr is not None:
            dx, dy, dyaw = corr
            self.T_map_base[0, 3] += dx
            self.T_map_base[1, 3] += dy
            self._map_yaw += dyaw
            cy_, sy_ = math.cos(self._map_yaw), math.sin(self._map_yaw)
            self.T_map_base[0, 0] =  cy_; self.T_map_base[0, 1] = -sy_
            self.T_map_base[1, 0] =  sy_; self.T_map_base[1, 1] =  cy_
            self.hector.set_pose(float(self.T_map_base[0, 3]),
                                 float(self.T_map_base[1, 3]), self._map_yaw)

        # ── Update yaw-rate estimate for next tick's scan de-skew ─
        # Use dt computed earlier in the IMU section.
        self._prev_yaw_rate = (self._map_yaw - self._prev_map_yaw) / max(dt, 0.02)
        self._prev_map_yaw  = self._map_yaw

        # ── Update Hector map from world-frame LiDAR points ───────
        # Motion gate: only update the map if the robot has moved ≥3 cm or
        # rotated ≥2° — prevents vibration from smearing occupied cells.
        # Also only update on fresh scans — re-painting the same scan smears walls.
        if len(lid_base) > 0:
            lid_world = transform_points(lid_base, self.T_map_base)
            if _lid_fresh:
                _cx = float(self.T_map_base[0, 3])
                _cy = float(self.T_map_base[1, 3])
                _moved = math.hypot(_cx - self._last_map_update_pose[0],
                                     _cy - self._last_map_update_pose[1])
                _rotated = abs(self._map_yaw - self._last_map_update_pose[2])
                if _moved > 0.03 or _rotated > math.radians(2) or self._hector_frames < 10:
                    self.hector.update_map(lid_world[:, :2])
                    self._hector_frames += 1
                    self._last_map_update_pose = np.array([_cx, _cy, self._map_yaw])

        # ── Loop closure: keyframe + detection (background thread) ─
        if (_lid_fresh and self.frame_count % 30 == 0 and not self._loop_building
                and len(lid_base) > 0):
            self._loop_building = True
            pose_snap = self.hector.get_pose()
            scan_snap = dict(scan)
            pts_snap  = lid_base[:, :2].copy()
            self.loop_det.maybe_add_keyframe(pose_snap, scan_snap, pts_snap)

            def _run_lc():
                result = self.loop_det.detect(pose_snap, scan_snap, pts_snap)
                if result is not None:
                    _, corr_lc = result
                    with self._loop_lock:
                        self._loop_correction = corr_lc
                    print(f"\n[LC] Loop closed — correction "
                          f"Δx={corr_lc[0]:.2f} Δy={corr_lc[1]:.2f} "
                          f"Δyaw={math.degrees(corr_lc[2]):.1f}°")
                self._loop_building = False
            threading.Thread(target=_run_lc, daemon=True).start()

        # TSDF integration removed — using BPA full-rebuild instead

        # ── Build capture-time T_map_base for camera→world transforms ──
        # Camera frames are captured asynchronously; during rotation the heading at
        # capture time (cam_yaw) differs from the post-Hector heading (_map_yaw).
        # Use the capture-time yaw so camera points land in the right world position.
        _cy, _sy = math.cos(cam_yaw), math.sin(cam_yaw)
        T_map_base_cam = self.T_map_base.copy()
        T_map_base_cam[0, 0] =  _cy; T_map_base_cam[0, 1] = -_sy
        T_map_base_cam[1, 0] =  _sy; T_map_base_cam[1, 1] =  _cy

        # ── Live stream in world frame (pose already updated) ──────
        # LiDAR: angle-sort world-frame scan points so polyline closes correctly
        _lidar_scan_world = []
        if len(lid_world) > 0:
            rx = float(self.T_map_base[0, 3])
            ry = float(self.T_map_base[1, 3])
            rel = lid_world[:, :2] - np.array([rx, ry], dtype=np.float32)
            order = np.argsort(np.arctan2(rel[:, 1], rel[:, 0]))
            _lidar_scan_world = [[round(float(lid_world[i, 0]), 3),
                                   round(float(lid_world[i, 1]), 3)]
                                  for i in order]
        # Camera: world-frame live cloud (use capture-time rotation to fix desync)
        _cam_world_live = (self._xform(cam_base, T_map_base_cam)
                           if len(cam_base) > 0 else None)
        with self._live_frame_lock:
            self._live_frame_xyz   = _cam_world_live
            self._live_frame_depth = cam_depth if _cam_world_live is not None else None
        _pts = []
        if _cam_world_live is not None and len(_cam_world_live) > 0:
            # Cap stream payload to 20k pts for bandwidth; full 40k goes to mesh
            _stream_xyz = _cam_world_live
            _stream_dep = cam_depth[:len(_cam_world_live)]
            if len(_stream_xyz) > 20000:
                _s = len(_stream_xyz) // 20000
                _stream_xyz = _stream_xyz[::_s]
                _stream_dep = _stream_dep[::_s]
            _xyz_r = np.round(_stream_xyz.astype(np.float32), 3)
            _dep_r = np.round(_stream_dep.astype(np.float32), 3).reshape(-1, 1)
            _pts = np.hstack([_xyz_r, _dep_r]).tolist()
        import json as _json
        _rx   = round(float(self.T_map_base[0, 3]), 3)
        _ry   = round(float(self.T_map_base[1, 3]), 3)
        _ryaw = round(float(self._map_yaw), 4)
        _payload = _json.dumps({"pts": _pts, "count": len(_pts),
                                 "lidar_scan": _lidar_scan_world,
                                 "robot_x": _rx, "robot_y": _ry,
                                 "robot_yaw": _ryaw}).encode()
        with self._stream3d_cache_lock:
            self._stream3d_cache = _payload

        # Robot trail
        pos = self.T_map_base[:3, 3]
        if (not self.robot_trail or
                math.hypot(pos[0] - self.robot_trail[-1][0],
                           pos[1] - self.robot_trail[-1][1]) > 0.05):
            self.robot_trail.append([float(pos[0]), float(pos[1])])
            if len(self.robot_trail) > 5000:
                self.robot_trail = self.robot_trail[-5000:]

        # Accumulate LiDAR wall map — only on fresh scans to avoid duplicating the same
        # points at the same world position every tick (wastes memory, no new info).
        if _lid_fresh and len(lid_world) > 0:
            self.lidar_map = (np.vstack([self.lidar_map, lid_world])
                              if len(self.lidar_map) > 0 else lid_world.copy())

        # Accumulate camera depth map — pre-downsample each new batch so only
        # unique voxels enter the map (prevents runaway 22k-pts/frame growth).
        if len(cam_base) > 0:
            cam_world    = self._xform(cam_base, T_map_base_cam)  # capture-time rotation
            ds_idx       = voxel_downsample_idx(cam_world, self.MAP_VOXEL)
            cam_world    = cam_world[ds_idx]
            cam_depth_ds = cam_depth[ds_idx]
            self.cam_map_xyz = (np.vstack([self.cam_map_xyz, cam_world])
                                if len(self.cam_map_xyz) > 0 else cam_world)
            self.cam_map_depth = (np.concatenate([self.cam_map_depth, cam_depth_ds])
                                  if len(self.cam_map_depth) > 0 else cam_depth_ds)

        # Periodic voxel downsample (merges duplicates across frames)
        if self.frame_count % 20 == 0:
            if len(self.lidar_map) > 0:
                self.lidar_map = voxel_downsample(self.lidar_map, self.MAP_VOXEL)
            if len(self.cam_map_xyz) > 0:
                idx = voxel_downsample_idx(self.cam_map_xyz, self.MAP_VOXEL)
                self.cam_map_xyz   = self.cam_map_xyz[idx]
                self.cam_map_depth = self.cam_map_depth[idx]

        # Cap sizes
        if len(self.lidar_map) > 200_000:
            self.lidar_map = voxel_downsample(self.lidar_map, self.MAP_VOXEL * 1.5)
        if len(self.cam_map_xyz) > 300_000:
            idx = voxel_downsample_idx(self.cam_map_xyz, self.MAP_VOXEL * 1.5)
            self.cam_map_xyz   = self.cam_map_xyz[idx]
            self.cam_map_depth = self.cam_map_depth[idx]

        # map_points computed AFTER downsample — shows stable post-merge count
        parts_all = [p for p in [self.lidar_map, self.cam_map_xyz] if len(p) > 0]
        self.map_points = (np.vstack(parts_all) if len(parts_all) > 1
                           else (parts_all[0] if parts_all
                                 else np.empty((0, 3), dtype=np.float32)))

        # Occupancy grid — only ray-cast on fresh scans (same scan re-cast smears walls)
        if _lid_fresh and scan:
            self._update_occupancy_from_scan(scan)
        if self.frame_count % 3 == 0:
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

        # Depth JPEG — colorize depth_m (float32 metres) with jet colormap.
        # depth_m may be at decimated resolution (320×240 after magnitude=2 filter).
        # Pixels with no depth reading (<5 cm) are rendered black.
        if depth_m is not None:
            try:
                _import_cv2()
                from config import cv2 as _cv2, np as _np
                _d = _np.clip(depth_m, 0.0, self.camera.max_depth)
                _d_norm = (_d / self.camera.max_depth * 255).astype(_np.uint8)
                _colored = _cv2.applyColorMap(_d_norm, _cv2.COLORMAP_JET)
                _colored[depth_m < 0.05] = 0   # no-data pixels → black
                _, _buf = _cv2.imencode('.jpg', _colored, [_cv2.IMWRITE_JPEG_QUALITY, 65])
                with self._depth_lock:
                    self._depth_jpeg = _buf.tobytes()
            except Exception:
                pass

        # 3-D dashboard data
        if self.frame_count % 15 == 0:
            self._prepare_3d_dashboard_data()

        # Mesh reconstruction — live camera view, launched every MESH_INTERVAL frames
        if (self.frame_count % self.MESH_INTERVAL == 0
                and not self._mesh_building
                and self._live_frame_xyz is not None):
            self._mesh_building = True
            threading.Thread(target=self._rebuild_mesh, daemon=True).start()

        return scan, imu_data, cam_img

    # ── Occupancy grid helpers ────────────────────────────────────
    def _update_occupancy_from_scan(self, scan):
        """
        Vectorised log-odds ray casting — no Python loops over scan rays.
        All rays processed simultaneously with numpy outer-product sampling.
        ~100× faster than the old per-ray Python Bresenham loop.
        """
        from config import np
        rx  = float(self.T_map_base[0, 3])
        ry  = float(self.T_map_base[1, 3])
        yaw = float(math.atan2(self.T_map_base[1, 0], self.T_map_base[0, 0]))
        res = self.OCC_RES
        orig_x, orig_y = self.OCC_ORIGIN[0], self.OCC_ORIGIN[1]
        sz  = self.OCC_SIZE
        rx_g = int((rx - orig_x) / res)
        ry_g = int((ry - orig_y) / res)
        if not (0 <= rx_g < sz and 0 <= ry_g < sz): return

        # Build angle/distance arrays, filter invalid readings + isolated spikes
        # Sort by angle first so neighbour comparison is meaningful
        sorted_items = sorted(scan.items(), key=lambda kv: float(kv[0]))
        raw_a = [float(a) for a, d in sorted_items if 150 <= float(d) <= 12000]
        raw_d = [float(d) for a, d in sorted_items if 150 <= float(d) <= 12000]
        ang_list = []; dist_list = []
        n_raw = len(raw_d)
        for i in range(n_raw):
            d      = raw_d[i]
            d_prev = raw_d[(i - 1) % n_raw]
            d_next = raw_d[(i + 1) % n_raw]
            # Spike: differs from BOTH neighbours by > 200 mm — isolated glitch
            if abs(d - d_prev) > 200 and abs(d - d_next) > 200:
                continue
            ang_list.append(raw_a[i]); dist_list.append(d)
        if not ang_list: return

        angles  = np.radians(np.array(ang_list,  dtype=np.float64))
        dists_m = np.array(dist_list, dtype=np.float64) / 1000.0
        angle_rads = yaw - angles

        # Endpoint grid coordinates (n_rays,)
        hx = np.clip(((rx + dists_m * np.cos(angle_rads) - orig_x) / res
                      ).astype(np.int32), 0, sz - 1)
        hy = np.clip(((ry + dists_m * np.sin(angle_rads) - orig_y) / res
                      ).astype(np.int32), 0, sz - 1)

        # Sample N_STEPS points along every ray simultaneously via outer product.
        # N=480 → 1 sample/cell for a 12 m ray at 0.025 m/cell — full coverage.
        N = 480
        if self._gpu_occ:
            import cupy as cp
            hx_cp = cp.asarray(hx, dtype=cp.int32)
            hy_cp = cp.asarray(hy, dtype=cp.int32)
            t_cp  = cp.linspace(0.0, 1.0, N + 1, dtype=cp.float32)
            gx_cp = cp.clip((rx_g + cp.outer(hx_cp - rx_g, t_cp)).astype(cp.int32), 0, sz - 1)
            gy_cp = cp.clip((ry_g + cp.outer(hy_cp - ry_g, t_cp)).astype(cp.int32), 0, sz - 1)
            lo = self.log_odds   # CuPy array — stays on GPU
            cp.add.at(lo, (gy_cp[:, :-1].ravel(), gx_cp[:, :-1].ravel()), self.L_FREE)
            cp.add.at(lo, (gy_cp[:, -1],           gx_cp[:, -1]),          self.L_OCC)
            cp.clip(lo, self.L_MIN, self.L_MAX, out=lo)
        else:
            t  = np.linspace(0.0, 1.0, N + 1)              # (N+1,)
            gx = np.clip((rx_g + np.outer(hx - rx_g, t)).astype(np.int32), 0, sz - 1)
            gy = np.clip((ry_g + np.outer(hy - ry_g, t)).astype(np.int32), 0, sz - 1)
            # gx, gy: (n_rays, N+1)
            lo = self.log_odds
            # Free-space: all samples except the endpoint
            np.add.at(lo, (gy[:, :-1].ravel(), gx[:, :-1].ravel()), self.L_FREE)
            # Occupied: endpoint only
            np.add.at(lo, (gy[:, -1], gx[:, -1]), self.L_OCC)
            np.clip(lo, self.L_MIN, self.L_MAX, out=lo)

    def _get_occupancy_for_dashboard(self):
        from config import np
        res = self.OCC_RES
        sz  = self.OCC_SIZE
        ox, oy = self.OCC_ORIGIN

        # Pull log_odds to CPU once — on Jetson UMA this is a near-zero-cost remap
        if self._gpu_occ:
            import cupy as cp
            lo_np = cp.asnumpy(self.log_odds)
        else:
            lo_np = self.log_odds

        rx = float(self.T_map_base[0, 3])
        ry = float(self.T_map_base[1, 3])
        cx = int((rx - ox) / res)
        cy = int((ry - oy) / res)

        # ── Viewport = bounding box of the explored area ─────────────
        # "Explored" = any cell that has ever been observed (log_odds ≠ 0).
        # This gives a stable floor-plan view: the map doesn't scroll when
        # the robot moves within an already-mapped room.  The box only grows
        # when the robot reaches a genuinely new area.
        MARGIN   = int(0.3 / res)    # 0.3 m padding — tight fit so room fills the canvas
        MAX_HALF = int(20.0 / res)   # hard cap: never show more than ±20 m

        explored = lo_np != 0.0
        rows_any = np.any(explored, axis=1)  # (sz,) bool — which rows have data
        cols_any = np.any(explored, axis=0)  # (sz,) bool — which cols have data

        if rows_any.any():
            rmin = int(np.argmax(rows_any))
            rmax = int(sz - 1 - np.argmax(rows_any[::-1]))
            cmin = int(np.argmax(cols_any))
            cmax = int(sz - 1 - np.argmax(cols_any[::-1]))
            y0 = max(0, rmin - MARGIN)
            y1 = min(sz, rmax + MARGIN + 1)
            x0 = max(0, cmin - MARGIN)
            x1 = min(sz, cmax + MARGIN + 1)
        else:
            # No observations yet — tiny initial window around robot
            INIT = int(5.0 / res)
            x0 = max(0, cx - INIT);  x1 = min(sz, cx + INIT)
            y0 = max(0, cy - INIT);  y1 = min(sz, cy + INIT)

        # Cap to MAX_HALF around the robot so the map never becomes tiny
        x0 = max(x0, cx - MAX_HALF);  x1 = min(x1, cx + MAX_HALF)
        y0 = max(y0, cy - MAX_HALF);  y1 = min(y1, cy + MAX_HALF)

        crop   = lo_np[y0:y1, x0:x1]
        crop_h, crop_w = crop.shape

        # Flip Y axis so display shows: left=UP, right=DOWN (world +Y → canvas UP).
        # Row 0 of the flipped image = max world_Y; row N = min world_Y.
        crop_display = crop[::-1, :]   # shape unchanged (crop_h, crop_w)
        flat   = crop_display.flatten()

        # Encode as uint8: 0=unknown, 85=free, 170=uncertain, 255=occupied.
        # Base64 binary is ~2.5× smaller than JSON int array at this size.
        import base64
        u8 = np.zeros(len(flat), dtype=np.uint8)
        u8[flat < -0.4] = 85             # free
        u8[flat >  0.4] = 255            # occupied / wall
        u8[(flat >= -0.4) & (flat <= 0.4) & (flat != 0.0)] = 170  # uncertain

        # Origin of the crop in world metres.
        # origin[0] = min world_X (left edge of image, unchanged).
        # origin[1] = max world_Y (TOP of flipped image — largest Y value in crop).
        crop_ox      = ox + x0 * res
        crop_oy_max  = oy + (y1 - 1) * res   # max Y = top of flipped image

        return {
            "grid_b64": base64.b64encode(u8.tobytes()).decode('ascii'),
            "res":    res,
            "origin": [float(crop_ox), float(crop_oy_max)],
            "size_w": crop_w,
            "size_h": crop_h,
            "trail":     self.robot_trail[-500:],
            "robot":        [rx, ry],
            "yaw":          float(math.atan2(self.T_map_base[1,0], self.T_map_base[0,0])),
            "robot_length": self.robot_length,
            "robot_width":  self.robot_width,
            "waypoints": self._perim_waypoints,
            "wp_idx":    self._perim_wp_idx,
        }

    # ── 3-D dashboard data ────────────────────────────────────────
    def _prepare_3d_dashboard_data(self):
        from config import np
        data = {"lidar": None, "camera": None}

        if len(self.lidar_map) > 0:
            ds = voxel_downsample(self.lidar_map, 0.12)
            if len(ds) > 6000: ds = ds[::len(ds)//6000]
            # Send flat XY wall points only (no extrusion — dashboard handles height)
            _lid_r = np.round(ds[:, :2].astype(np.float32), 3)
            data["lidar"] = _lid_r.tolist()

        if len(self.cam_map_xyz) > 0:
            step  = max(1, len(self.cam_map_xyz) // 25000)
            xyz   = self.cam_map_xyz[::step]
            depth = self.cam_map_depth[::step]
            _xyz_r = np.round(xyz.astype(np.float32), 3)
            _dep_r = np.round(depth[:len(xyz)].astype(np.float32), 3).reshape(-1, 1)
            data["camera"] = np.hstack([_xyz_r, _dep_r]).tolist()

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
    def get_cam_depth_jpeg(self):
        with self._depth_lock: return self._depth_jpeg
    def get_map3d(self):
        with self._map3d_lock:  return self._map3d_data

    def get_live_frame(self):
        """Returns (xyz Nx3 float32, depth N float32) or (None, None)."""
        with self._live_frame_lock:
            if self._live_frame_xyz is None: return None, None
            return self._live_frame_xyz.copy(), self._live_frame_depth.copy()

    def get_stream3d_bytes(self):
        """Returns pre-serialised JSON bytes for /stream3d endpoint."""
        with self._stream3d_cache_lock:
            return self._stream3d_cache

    def get_mesh(self):
        with self._mesh_lock: return self._mesh_data

    # ── Perimeter waypoint generation ─────────────────────────────
    def _generate_perimeter_waypoints(self, scan):
        """
        From the initial LiDAR scan, create one waypoint per 18° angular sector,
        placed WALL_MARGIN metres inside the nearest detected wall in that sector.
        Waypoints are sorted by angle → the robot traces the room perimeter in order.
        Returns a list of [wx, wy] world-frame positions.
        """
        rx  = float(self.T_map_base[0, 3])
        ry  = float(self.T_map_base[1, 3])
        yaw = float(math.atan2(self.T_map_base[1, 0], self.T_map_base[0, 0]))

        BIN_DEG = 18            # degrees per sector
        N_BINS  = int(360 / BIN_DEG)   # 20 sectors
        bins    = {}            # bin_idx → (angle_deg, dist_mm)

        for angle_deg, dist_mm in scan.items():
            if dist_mm < 300 or dist_mm > 12000: continue
            b = int(float(angle_deg) / BIN_DEG) % N_BINS
            # Keep the closest wall in each angular sector
            if b not in bins or dist_mm < bins[b][1]:
                bins[b] = (float(angle_deg), dist_mm)

        if len(bins) < 4:
            print("[Perimeter] Too few scan points — aborting perimeter generation")
            return []

        waypoints = []
        for b in sorted(bins.keys()):
            angle_deg, dist_mm = bins[b]
            d = dist_mm / 1000.0 - self.WALL_MARGIN
            if d < 0.2: continue          # wall too close to robot
            # Convert from sensor-frame angle to world-frame coordinates
            ar = yaw - math.radians(angle_deg)
            waypoints.append([rx + d * math.cos(ar),
                               ry + d * math.sin(ar)])

        print(f"[Perimeter] {len(waypoints)} waypoints — room traversal ready")
        return waypoints

    # ── Mesh reconstruction (incremental BPA camera mesh) ───────────
    def _rebuild_mesh(self):
        """
        Incremental BPA mesh — enhanced for 3D quality:

        Sources from both the live camera frame (fine detail at current view)
        AND the accumulated cam_map_xyz (multi-frame context density).  A 1-voxel
        border of already-meshed cells is included so new patches stitch cleanly
        to existing geometry.

        Key upgrades vs old version:
          • Accumulated map supplement → far denser BPA input per patch
          • 1-voxel context border → no seam gaps at patch boundaries
          • TRACK_VOXEL 0.05 m → finer mesh resolution
          • 5-tier BPA radii (micro → macro gap fill)
          • orient_normals_consistent_tangent_plane → smoother surfaces
          • 15 Taubin iterations → better surface quality
          • MAX_ACC_VERTS 300k + voxel-reset on trim → no ghost holes on revisit
        """
        from config import np
        _import_o3d()
        from config import o3d

        if not self._running:
            self._mesh_building = False
            return

        # ── Grab live camera frame (world coords, high-density current view) ─
        with self._live_frame_lock:
            if self._live_frame_xyz is None or len(self._live_frame_xyz) < 100:
                self._mesh_building = False
                return
            xyz_live   = self._live_frame_xyz.copy()
            depth_live = (self._live_frame_depth.copy()
                          if self._live_frame_depth is not None else None)
        if depth_live is None or len(depth_live) != len(xyz_live):
            self._mesh_building = False
            return

        from lidar import voxel_downsample_idx

        # ── Voxel downsample live frame ────────────────────────────────
        BPA_MAX   = 12000   # balanced density vs. BPA runtime
        BPA_VOXEL = 0.015   # 15 mm voxels — fine enough for walls at 4 m

        ds_idx    = voxel_downsample_idx(xyz_live.astype(np.float32), BPA_VOXEL)
        bpa_xyz   = xyz_live[ds_idx].astype(np.float64)
        bpa_depth = depth_live[ds_idx]

        if len(bpa_xyz) > BPA_MAX:
            coarser   = BPA_VOXEL * (len(bpa_xyz) / BPA_MAX) ** (1/3)
            ds2       = voxel_downsample_idx(bpa_xyz.astype(np.float32), float(coarser))
            bpa_xyz   = bpa_xyz[ds2]
            bpa_depth = bpa_depth[ds2]

        # ── Depth gate ────────────────────────────────────────────────
        # Keep points from 0.1 m (near noise floor) up to MESH_MAX_DIST (5 m).
        # This includes far walls — the old 1.5 m cap silently excluded them.
        depth_mask = (bpa_depth >= 0.1) & (bpa_depth <= self.MESH_MAX_DIST)
        if not depth_mask.any():
            self._mesh_building = False
            return
        bpa_xyz   = bpa_xyz[depth_mask]
        bpa_depth = bpa_depth[depth_mask]

        # ── New-voxel gate — skip already-meshed cells ────────────────
        TRACK = self.TRACK_VOXEL   # 0.05 m — finer than old 0.08 m
        P1, P2 = np.int64(1_000_003), np.int64(1_000_033)
        vk     = np.floor(bpa_xyz / TRACK).astype(np.int64)
        packed = (vk[:, 0] * P1 + vk[:, 1] * P2 + vk[:, 2]).astype(np.int64)
        is_new = ~np.isin(packed, self._meshed_voxels)
        if not is_new.any():
            self._mesh_building = False
            return
        new_voxel_keys = np.unique(packed[is_new])

        # ── Build expanded region: new voxels + 1-cell border ─────────
        # Border provides context points so BPA stitches cleanly to
        # existing geometry — no seam gaps at patch boundaries.
        unique_new_vk = np.unique(vk[is_new], axis=0)   # M×3
        ofs = np.array([[a, b, c]
                        for a in (-1, 0, 1)
                        for b in (-1, 0, 1)
                        for c in (-1, 0, 1)], dtype=np.int64)   # 27×3
        exp        = (unique_new_vk[:, None, :] + ofs[None, :, :]).reshape(-1, 3)
        exp_packed = (exp[:, 0] * P1 + exp[:, 1] * P2
                      + exp[:, 2]).astype(np.int64)

        # Include all live-frame points inside the expanded region
        in_region = np.isin(packed, exp_packed)
        bpa_xyz   = bpa_xyz[in_region]
        bpa_depth = bpa_depth[in_region]

        # ── Supplement with accumulated camera map (context density) ──
        # cam_map_xyz holds ALL past frames merged at MAP_VOXEL resolution.
        # Adding these points to the new-voxel patch densifies BPA input
        # and improves normal quality near patch boundaries.
        # CPython GIL makes attribute reads effectively atomic — safe here.
        acc_xyz   = self.cam_map_xyz
        acc_depth = self.cam_map_depth
        if len(acc_xyz) > 50:
            acc_xyz   = acc_xyz.copy()
            acc_depth = acc_depth.copy()
            na        = (acc_depth >= 0.1) & (acc_depth <= self.MESH_MAX_DIST)
            acc_xyz   = acc_xyz[na].astype(np.float64)
            acc_depth = acc_depth[na]
            if len(acc_xyz) > 0:
                acc_vk = np.floor(acc_xyz / TRACK).astype(np.int64)
                acc_pk = (acc_vk[:, 0] * P1 + acc_vk[:, 1] * P2
                          + acc_vk[:, 2]).astype(np.int64)
                acc_in = np.isin(acc_pk, exp_packed)
                if acc_in.any():
                    bpa_xyz   = np.vstack([bpa_xyz,   acc_xyz[acc_in]])
                    bpa_depth = np.concatenate([bpa_depth, acc_depth[acc_in]])

        if len(bpa_xyz) < 30:
            self._mesh_building = False
            return

        # Deduplicate combined live+accumulated source at BPA_VOXEL
        ds_comb   = voxel_downsample_idx(bpa_xyz.astype(np.float32), BPA_VOXEL)
        bpa_xyz   = bpa_xyz[ds_comb]
        bpa_depth = bpa_depth[ds_comb]
        if len(bpa_xyz) > BPA_MAX:
            coarser   = BPA_VOXEL * (len(bpa_xyz) / BPA_MAX) ** (1/3)
            ds3       = voxel_downsample_idx(bpa_xyz.astype(np.float32), float(coarser))
            bpa_xyz   = bpa_xyz[ds3]
            bpa_depth = bpa_depth[ds3]

        # ── Jet depth colormap ────────────────────────────────────────
        d_min = float(bpa_depth.min()); d_max = float(bpa_depth.max())
        d_rng = max(d_max - d_min, 0.1)
        t = np.clip((bpa_depth - d_min) / d_rng, 0.0, 1.0).astype(np.float64)
        colors = np.stack([
            np.clip(1.5 - np.abs(4*t - 3), 0, 1),
            np.clip(1.5 - np.abs(4*t - 2), 0, 1),
            np.clip(1.5 - np.abs(4*t - 1), 0, 1),
        ], axis=1)

        # ── Point cloud + two-pass outlier removal ────────────────────
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(bpa_xyz)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        if len(bpa_xyz) >= 30:
            pcd, inlier_idx = pcd.remove_statistical_outlier(
                nb_neighbors=20, std_ratio=2.0)   # tighter → cleaner surface
            bpa_xyz   = bpa_xyz[inlier_idx]
            bpa_depth = bpa_depth[inlier_idx]
            if len(bpa_xyz) >= 30:
                pcd, inlier_idx2 = pcd.remove_radius_outlier(
                    nb_points=5, radius=0.07)
                bpa_xyz   = bpa_xyz[inlier_idx2]
                bpa_depth = bpa_depth[inlier_idx2]

        if len(np.asarray(pcd.points)) < 50:
            self._mesh_building = False
            return

        # ── Normal estimation ─────────────────────────────────────────
        nn_dists    = np.asarray(pcd.compute_nearest_neighbor_distance())
        mean_d      = float(np.median(nn_dists)) if len(nn_dists) > 0 else 0.04
        mean_d      = float(np.clip(mean_d, 0.008, 0.08))
        norm_radius = float(np.clip(mean_d * 10.0, 0.06, 0.40))
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=norm_radius, max_nn=30))
        # Propagate normal consistency within the local patch first,
        # then orient all normals toward the camera for correct facing.
        try:
            pcd.orient_normals_consistent_tangent_plane(k=25)
        except Exception:
            pass
        T_map_cam = (self.T_map_base @ self.T_base_cam).astype(np.float64)
        pcd.orient_normals_towards_camera_location(T_map_cam[:3, 3])

        # ── BPA reconstruction (5-tier radii) ─────────────────────────
        # Tier 1: micro-detail   Tier 5: macro gap-fill
        bpa_radii = [
            mean_d * 1.0,
            mean_d * 2.0,
            mean_d * 3.5,
            mean_d * 6.0,
            min(mean_d * 10.0, 0.18),
        ]
        try:
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, o3d.utility.DoubleVector(bpa_radii))
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_duplicated_vertices()
            mesh.remove_non_manifold_edges()
            mesh.remove_unreferenced_vertices()
            if len(np.asarray(mesh.vertices)) > 10:
                mesh = mesh.filter_smooth_taubin(number_of_iterations=8)
            mesh.compute_vertex_normals()
        except Exception as e:
            print(f"[Mesh] BPA failed: {e}")
            self._mesh_building = False
            return

        new_verts = np.asarray(mesh.vertices,      dtype=np.float32)
        new_faces = np.asarray(mesh.triangles,     dtype=np.int32)
        new_vcols = np.asarray(mesh.vertex_colors, dtype=np.float32)
        if len(new_verts) == 0 or len(new_faces) == 0:
            self._mesh_building = False
            return
        if len(new_vcols) == 0:
            new_vcols = np.tile([0.15, 0.75, 0.90], (len(new_verts), 1)).astype(np.float32)

        # ── Merge into accumulated mesh ───────────────────────────────
        MAX_ACC_VERTS = 300_000   # increased for full-room coverage
        face_off = len(self._acc_mesh_verts)
        self._acc_mesh_verts  = np.vstack([self._acc_mesh_verts,  new_verts])
        self._acc_mesh_faces  = np.vstack([self._acc_mesh_faces,  new_faces + face_off])
        self._acc_mesh_colors = np.vstack([self._acc_mesh_colors, new_vcols])

        if len(self._acc_mesh_verts) > MAX_ACC_VERTS:
            keep    = MAX_ACC_VERTS
            v_start = len(self._acc_mesh_verts) - keep
            valid   = (self._acc_mesh_faces >= v_start).all(axis=1)
            self._acc_mesh_faces  = self._acc_mesh_faces[valid] - v_start
            self._acc_mesh_verts  = self._acc_mesh_verts[-keep:]
            self._acc_mesh_colors = self._acc_mesh_colors[-keep:]
            # Reset voxel tracking so trimmed regions are re-meshed on revisit
            # (without this, trimmed areas remain permanently unpatched)
            self._meshed_voxels   = np.empty(0, dtype=np.int64)

        self._meshed_voxels = np.union1d(self._meshed_voxels, new_voxel_keys)

        acc_v = self._acc_mesh_verts
        acc_f = self._acc_mesh_faces
        acc_c = self._acc_mesh_colors
        with self._mesh_lock:
            self._mesh_data = {"cam": {
                "verts":  acc_v.round(3).tolist(),
                "faces":  acc_f.tolist(),
                "colors": (np.clip(acc_c, 0, 1) * 255).astype(np.uint8).tolist(),
            }}
        self._mesh_building = False
        print(f"[Mesh] +{len(new_faces)} tris | total {len(self._acc_mesh_faces)} tris"
              f" | pts {len(bpa_xyz)} | depth {d_min:.1f}–{d_max:.1f} m")

    # ── Main loop ─────────────────────────────────────────────────
    def run(self):
        self._running = True
        if self.camera:
            threading.Thread(target=self._camera_loop, daemon=True).start()
        mode = 'Map-only mode' if self.map_only else 'Avoidance + Mapping'
        print(f"\n[SLAM] {mode} started! Ctrl+C to stop.\n")

        try:
            while self._running:
                scan, imu_data, cam_img = self._mapping_tick()
                if not scan:
                    self.state = "WAIT_LIDAR"
                    self._stop(); time.sleep(0.1); continue

                hist   = self.lidar.get_histogram(self.VFH_BIN, 12000)
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

                # ── Phase 1: initial still scan ───────────────────────────
                if not self._perim_done and self.frame_count < self.INIT_SCAN_FRAMES:
                    self.state        = "INIT_SCAN"
                    self.state_detail = f"{self.frame_count}/{self.INIT_SCAN_FRAMES} building map…"
                    self._brake()
                    time.sleep(self.LOOP_RATE)
                    self._print_status(zones, target, gap)
                    continue

                # ── Phase 2: generate waypoints once ─────────────────────
                if not self._perim_done and not self._perim_waypoints:
                    self._perim_waypoints = self._generate_perimeter_waypoints(scan)
                    self._perim_wp_t0     = time.time()
                    if not self._perim_waypoints:
                        self._perim_done = True   # nothing to traverse; fall through

                # ── Phase 3: perimeter traversal ─────────────────────────
                if not self._perim_done and self._perim_waypoints:
                    wp  = self._perim_waypoints[self._perim_wp_idx]
                    rx_ = float(self.T_map_base[0, 3])
                    ry_ = float(self.T_map_base[1, 3])
                    dx, dy      = wp[0] - rx_, wp[1] - ry_
                    dist_to_wp  = math.sqrt(dx * dx + dy * dy)

                    self.state        = "PERIMETER"
                    self.state_detail = (
                        f"wp {self._perim_wp_idx+1}/{len(self._perim_waypoints)}"
                        f" d={dist_to_wp:.1f}m"
                    )

                    # Waypoint reached or timed-out → advance
                    if (dist_to_wp < self.WP_REACH_DIST
                            or time.time() - self._perim_wp_t0 > self.WP_TIMEOUT):
                        self._perim_wp_idx += 1
                        self._perim_wp_t0   = time.time()
                        if self._perim_wp_idx >= len(self._perim_waypoints):
                            print(f"\n[Perimeter] Complete — switching to normal navigation")
                            self._perim_done = True
                        time.sleep(self.LOOP_RATE)
                        self._print_status(zones, target, gap)
                        continue

                    # Compute robot-relative heading to waypoint
                    robot_yaw    = math.atan2(self.T_map_base[1, 0], self.T_map_base[0, 0])
                    wp_angle     = math.atan2(dy, dx)
                    delta        = (wp_angle - robot_yaw + math.pi) % (2 * math.pi) - math.pi
                    wp_target    = (-math.degrees(delta)) % 360  # CW robot-frame degrees

                    if self._check_stuck(fmin):
                        self._unstuck()
                    elif fmin < self.EMERGENCY_DIST:
                        self.state = "PERIM_EMRG"
                        self._drive(-self.REVERSE_SPEED, -self.REVERSE_SPEED)
                        time.sleep(self.REVERSE_TIME)
                        side = self.TURN_SPEED
                        if fl < fr: self._drive( side, -side)
                        else:       self._drive(-side,  side)
                        time.sleep(self.TURN_TIME)
                        self._front_history.clear()
                    elif fmin < self.STOP_DIST:
                        # Blocked — spin toward VFH gap heading
                        self.state = "PERIM_TURN"
                        herr = ((target or 0) + 180) % 360 - 180
                        d    = 1 if herr >= 0 else -1
                        self._drive(d * self.TURN_SPEED, -d * self.TURN_SPEED)
                    else:
                        speed = self._scale_speed(fmin)
                        self._steer(wp_target, speed)

                    self._print_status(zones, target, gap)
                    time.sleep(self.LOOP_RATE)
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
            self.close()

    def _print_status(self, zones, target, gap):
        f = zones["FRONT"]; fl = zones["FRONT_LEFT"]; fr = zones["FRONT_RIGHT"]
        tgt = f"{target:.0f}°" if target else "---"
        gw  = f"{gap['width']:.0f}°" if gap else "---"
        mp  = len(self.map_points)
        sys.stdout.write(
            f"\r[{self.state:<12}] {self.state_detail:<24} "
            f"F:{int(f):5d} FL:{int(fl):5d} FR:{int(fr):5d} "
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
        _lo = (self.log_odds if not self._gpu_occ
               else __import__('cupy').asnumpy(self.log_odds))
        pgm = np.full((sz, sz), 205, dtype=np.uint8)
        pgm[_lo < -0.4] = 254
        pgm[_lo >  0.4] = 0
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
        if len(self.lidar_map)   > 0: np.save(os.path.expanduser("~/slam_lidar.npy"),        self.lidar_map)
        if len(self.cam_map_xyz) > 0:
            np.save(os.path.expanduser("~/slam_camera.npy"),       self.cam_map_xyz)
            np.save(os.path.expanduser("~/slam_camera_depth.npy"), self.cam_map_depth)

    def close(self):
        self._running = False
        self._brake()

        # Wait for background mesh thread to finish before tearing down
        # Open3D objects — otherwise TSDF/BPA C++ destructor segfaults
        deadline = time.time() + 5.0
        while self._mesh_building and time.time() < deadline:
            time.sleep(0.05)

        # Safely destroy TSDF volume under lock so no thread can access it
        # after free
        if self._tsdf is not None:
            with self._tsdf_lock:
                self._tsdf = None

        if self.motors: self.motors.close()
        self.lidar.close()
        if self.imu:
            try: self.imu.close()
            except Exception: pass
        if self.camera:
            try: self.camera.close()
            except Exception: pass
        print("[SLAM] Shutdown complete")