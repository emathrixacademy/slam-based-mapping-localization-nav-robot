"""
slam.py — SLAM algorithms for IMU-less robots (RTAB-Map-style).

Classes
-------
HectorSLAM2D         Scan-to-map 2D pose estimator (replaces scan-to-scan ICP).
RGBDOdometry         Open3D RGB-D frame-to-frame 6-DOF delta (backup / Z-axis).
LoopClosureDetector  Histogram descriptor + 2D-ICP loop closure.

Pose hierarchy (primary → fallback):
  1. Hector SLAM 2D  — scan registered against accumulated map (no drift)
  2. RGB-D odometry  — 6-DOF camera delta (backup when LiDAR scan is weak)
  3. Loop closure     — applied as a one-shot correction on revisit
"""

import math
import threading
import numpy as np


# ═══════════════════════════════════════════════════════════════════
#  Hector SLAM 2D
# ═══════════════════════════════════════════════════════════════════

class HectorSLAM2D:
    """
    Scan-to-map 2D pose estimator.

    Each new scan is registered against the accumulated probability map
    (not just the previous scan), eliminating the frame-to-frame drift of
    plain ICP.  Coarse-to-fine Gauss-Newton with vectorised numpy bilinear
    interpolation for speed.

    Map convention: probability in [0, 1].  0.5 = unknown, →1 = occupied.
    Pose convention: (x_m, y_m, yaw_rad) in the robot base / map frame.
    """

    def __init__(self, res: float = 0.05, size: int = 600, levels: int = 3):
        self._res_0  = res
        self._size_0 = size
        self._levels = levels

        # Per-level probability maps.  Level 0 = finest, level L = coarsest.
        self._maps = [
            np.full((size >> lvl, size >> lvl), 0.5, dtype=np.float32)
            for lvl in range(levels)
        ]

        self._pose  = np.zeros(3, dtype=np.float64)   # [x, y, yaw]
        self._lock  = threading.Lock()
        self._nframes = 0   # map update count (bootstrap guard)

    # ── Pose accessors (thread-safe) ──────────────────────────────
    def get_pose(self):
        with self._lock:
            return self._pose.copy()

    def set_pose(self, x: float, y: float, yaw: float):
        with self._lock:
            self._pose[:] = (x, y, yaw)

    # ── Grid helpers ──────────────────────────────────────────────
    def _res(self, lvl):
        return self._res_0 * (1 << lvl)

    def _half(self, lvl):
        return (self._size_0 >> lvl) * 0.5

    def _to_grid(self, wx, wy, lvl):
        """World metres → grid float coords (both arrays or scalars)."""
        r = self._res(lvl)
        h = self._half(lvl)
        return wx / r + h, wy / r + h

    # ── Vectorised bilinear helpers ───────────────────────────────
    @staticmethod
    def _bilinear_batch(grid, gx, gy):
        """
        Bilinear interpolation at float grid coords (gx, gy) — both 1-D arrays.
        Returns (values N, dm_dgx N, dm_dgy N).  Points outside grid → 0.5, 0, 0.
        """
        h, w = grid.shape
        x0 = gx.astype(np.int32)
        y0 = gy.astype(np.int32)
        x1, y1 = x0 + 1, y0 + 1

        valid = (x0 >= 0) & (y0 >= 0) & (x1 < w) & (y1 < h)
        # Default: unknown map value, zero gradient
        m     = np.full(len(gx), 0.5, dtype=np.float64)
        dm_dx = np.zeros(len(gx), dtype=np.float64)
        dm_dy = np.zeros(len(gx), dtype=np.float64)
        if not valid.any():
            return m, dm_dx, dm_dy

        x0v, y0v = x0[valid], y0[valid]
        x1v, y1v = x1[valid], y1[valid]
        fx = (gx - x0)[valid]
        fy = (gy - y0)[valid]

        # Corner values
        c00 = grid[y0v, x0v].astype(np.float64)
        c10 = grid[y0v, x1v].astype(np.float64)
        c01 = grid[y1v, x0v].astype(np.float64)
        c11 = grid[y1v, x1v].astype(np.float64)

        m[valid] = ((1-fx)*(1-fy)*c00 + fx*(1-fy)*c10
                    + (1-fx)*fy*c01  + fx*fy*c11)
        dm_dx[valid] = (1-fy)*(c10 - c00) + fy*(c11 - c01)
        dm_dy[valid] = (1-fx)*(c01 - c00) + fx*(c11 - c10)
        return m, dm_dx, dm_dy

    # ── Map update ────────────────────────────────────────────────
    def update_map(self, pts_world_2d: np.ndarray):
        """
        Stamp obstacle endpoints into all map levels.
        pts_world_2d: Nx2 array in map frame (metres).

        Vectorised: replaces the inner Python for-loop with np.add.at scatter
        (or cp.add.at on Jetson GPU via UMA).  Typical speed-up: 50-200×.
        """
        if len(pts_world_2d) == 0:
            return

        # Try CuPy once; fall back to numpy on any error
        try:
            from config import cuda_ok
            _use_gpu = cuda_ok
        except Exception:
            _use_gpu = False

        if _use_gpu:
            import cupy as cp

        for lvl in range(self._levels):
            grid = self._maps[lvl]
            h, w = grid.shape
            gx, gy = self._to_grid(pts_world_2d[:, 0], pts_world_2d[:, 1], lvl)
            ix = np.round(gx).astype(np.int32)
            iy = np.round(gy).astype(np.int32)
            valid = (ix >= 0) & (iy >= 0) & (ix < w) & (iy < h)
            vx = ix[valid];  vy = iy[valid]
            if len(vx) == 0:
                continue

            if _use_gpu:
                # On Jetson UMA, cp.asarray / cp.asnumpy are near-zero-cost
                # pointer remaps — no DMA copy.
                g_cp  = cp.asarray(grid)
                vx_cp = cp.asarray(vx, dtype=cp.int32)
                vy_cp = cp.asarray(vy, dtype=cp.int32)

                cp.add.at(g_cp, (vy_cp, vx_cp), cp.float32(0.07))
                for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny = vy_cp + di;  nx = vx_cp + dj
                    nm = (nx >= 0) & (ny >= 0) & (nx < w) & (ny < h)
                    if nm.any():
                        cp.add.at(g_cp, (ny[nm], nx[nm]), cp.float32(-0.02))
                cp.clip(g_cp, 0.01, 0.99, out=g_cp)
                # Write back — UMA means this is a no-op pointer update
                self._maps[lvl][:] = cp.asnumpy(g_cp)
            else:
                # Vectorised numpy path — still 50-100× faster than Python loop
                np.add.at(grid, (vy, vx), np.float32(0.07))
                for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny = vy + di;  nx = vx + dj
                    nm = (nx >= 0) & (ny >= 0) & (nx < w) & (ny < h)
                    np.add.at(grid, (ny[nm], nx[nm]), np.float32(-0.02))
                np.clip(grid, 0.01, 0.99, out=grid)

        self._nframes += 1

    # ── Scan registration ─────────────────────────────────────────
    def register(self, scan_pts_base_2d: np.ndarray,
                 max_iter: int = 15, step_limit: float = 0.3) -> float:
        """
        Register scan_pts_base_2d (sensor frame Nx2, metres) against the map.
        Optimises self._pose in place via coarse-to-fine Gauss-Newton.
        Returns fitness in [0, 1] (mean map probability at scan endpoints).
        """
        if len(scan_pts_base_2d) < 20 or self._nframes < 5:
            return 0.0

        with self._lock:
            pose = self._pose.copy()

        px = scan_pts_base_2d[:, 0]
        py = scan_pts_base_2d[:, 1]

        fitness = 0.0
        for lvl in range(self._levels - 1, -1, -1):
            grid    = self._maps[lvl]
            inv_res = 1.0 / self._res(lvl)

            for _ in range(max_iter):
                x, y, yaw = pose
                cy, sy = math.cos(yaw), math.sin(yaw)

                # Transform scan to world frame
                wx = cy * px - sy * py + x
                wy = sy * px + cy * py + y

                # Grid coordinates
                gx, gy = self._to_grid(wx, wy, lvl)

                m, dm_dgx, dm_dgy = self._bilinear_batch(grid, gx, gy)

                # Convert grid gradient to world gradient
                dm_wx = dm_dgx * inv_res
                dm_wy = dm_dgy * inv_res

                # Jacobian yaw component
                dth_wx = -sy * px - cy * py
                dth_wy =  cy * px - sy * py
                J_yaw  = dm_wx * dth_wx + dm_wy * dth_wy

                J  = np.column_stack([dm_wx, dm_wy, J_yaw])   # N×3
                e  = 1.0 - m                                    # N

                H  = J.T @ J + np.eye(3) * 1e-5
                g  = J.T @ e

                try:
                    dp = np.linalg.solve(H, g)
                except np.linalg.LinAlgError:
                    break

                # Clamp step
                xy_norm = math.hypot(float(dp[0]), float(dp[1]))
                if xy_norm > step_limit:
                    dp[:2] *= step_limit / xy_norm
                dp[2] = max(-0.2, min(0.2, float(dp[2])))

                pose += dp
                pose[2] = (pose[2] + math.pi) % (2 * math.pi) - math.pi

                if xy_norm < 1e-4 and abs(dp[2]) < 1e-4:
                    break

                fitness = float(m.mean())

        with self._lock:
            self._pose = pose

        return fitness


# ═══════════════════════════════════════════════════════════════════
#  RGB-D Visual Odometry
# ═══════════════════════════════════════════════════════════════════

class RGBDOdometry:
    """
    Frame-to-frame 6-DOF pose estimation using Open3D hybrid RGB-D odometry.
    Used as a backup / Z-axis supplement when Hector SLAM fitness is low.

    Convention: push_frame returns T_cam_delta such that
        p_cam_new = T_cam_delta @ p_cam_old
    The robot.py caller converts this to a base-frame delta.
    """

    def __init__(self, width: int = 640, height: int = 480,
                 fx: float = 384.0, fy: float = 384.0,
                 cx: float = 320.0, cy_: float = 240.0,
                 depth_trunc: float = 3.0):
        from config import _import_o3d, _import_numpy
        _import_o3d()
        _import_numpy()
        from config import o3d

        self._intr = o3d.camera.PinholeCameraIntrinsic(
            width, height, fx, fy, cx, cy_)
        self._depth_trunc = depth_trunc
        self._jacobian = o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()
        self._option   = o3d.pipelines.odometry.OdometryOption()
        self._prev     = None   # previous o3d.geometry.RGBDImage

    def push_frame(self, color_bgr: np.ndarray,
                   depth_m: np.ndarray):
        """
        Push a new (color BGR uint8, depth float32 metres) frame.
        Returns (T_delta 4×4 float64, success bool).
        T_delta maps source camera frame to target camera frame.
        """
        from config import o3d, np as _np

        # BGR uint8 → RGB float32 [0,1]
        rgb_f = color_bgr[:, :, ::-1].astype(np.float32) / 255.0
        # Clip depth
        d_clip = depth_m.copy()
        d_clip[d_clip > self._depth_trunc] = 0.0

        o3d_color = o3d.geometry.Image(rgb_f)
        o3d_depth = o3d.geometry.Image(d_clip)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color, o3d_depth,
            depth_scale=1.0,
            depth_trunc=self._depth_trunc,
            convert_rgb_to_intensity=True)

        if self._prev is None:
            self._prev = rgbd
            return np.eye(4, dtype=np.float64), False

        success, T, _info = o3d.pipelines.odometry.compute_rgbd_odometry(
            self._prev, rgbd,
            self._intr,
            np.eye(4),
            self._jacobian,
            self._option)

        self._prev = rgbd
        if success:
            return T.astype(np.float64), True
        return np.eye(4, dtype=np.float64), False


# ═══════════════════════════════════════════════════════════════════
#  Loop Closure Detector
# ═══════════════════════════════════════════════════════════════════

class LoopClosureDetector:
    """
    Keyframe-based loop closure for IMU-less robots.

    Descriptor: 72-bin angular range histogram (L2-normalised).
    Detection: cosine similarity > threshold + spatial proximity gate.
    Verification: lightweight 2D point-to-point ICP.
    """

    def __init__(self, n_bins: int = 72, sim_thresh: float = 0.85,
                 min_kf_dist: float = 0.5, min_kf_gap: int = 20,
                 icp_max_dist: float = 0.4):
        self._n_bins      = n_bins
        self._sim_thresh  = sim_thresh
        self._min_kf_dist = min_kf_dist
        self._min_kf_gap  = min_kf_gap
        self._icp_max_dist = icp_max_dist
        self._keyframes   = []   # list of (pose_2d, descriptor, pts_2d)
        self._lock        = threading.Lock()

    # ── Descriptor ────────────────────────────────────────────────
    def _descriptor(self, scan_dict: dict) -> np.ndarray:
        """72-bin range histogram, L2-normalised."""
        desc = np.full(self._n_bins, 12.0, dtype=np.float32)
        for angle_deg, dist_mm in scan_dict.items():
            if dist_mm < 20 or dist_mm >= 12000:
                continue
            b = int(float(angle_deg) / 360.0 * self._n_bins) % self._n_bins
            d = dist_mm / 1000.0
            if d < desc[b]:
                desc[b] = d
        desc = np.clip(desc, 0.0, 12.0)
        n = float(np.linalg.norm(desc))
        return (desc / n) if n > 1e-6 else desc

    # ── Keyframe management ───────────────────────────────────────
    def maybe_add_keyframe(self, pose_2d: np.ndarray,
                           scan: dict, pts_2d: np.ndarray) -> bool:
        """Add keyframe if the robot has moved far enough from the last one."""
        with self._lock:
            if self._keyframes:
                last_p = self._keyframes[-1][0]
                if math.hypot(pose_2d[0] - last_p[0],
                              pose_2d[1] - last_p[1]) < self._min_kf_dist:
                    return False
            self._keyframes.append((
                pose_2d.copy(),
                self._descriptor(scan),
                pts_2d.copy()))
            return True

    # ── Loop detection ────────────────────────────────────────────
    def detect(self, pose_2d: np.ndarray, scan: dict,
               pts_2d: np.ndarray):
        """
        Try to detect a loop closure.
        Returns (kf_idx, correction np.array[dx, dy, dyaw]) or None.
        Skip most-recent min_kf_gap keyframes to avoid false positives.
        """
        with self._lock:
            kfs = list(self._keyframes)

        candidates = kfs[:max(0, len(kfs) - self._min_kf_gap)]
        if len(candidates) < 5:
            return None

        desc = self._descriptor(scan)

        best_sim = self._sim_thresh
        best_idx = -1
        for i, (kf_pose, kf_desc, _) in enumerate(candidates):
            sim = float(np.dot(desc, kf_desc))
            if sim > best_sim:
                dist = math.hypot(pose_2d[0] - kf_pose[0],
                                  pose_2d[1] - kf_pose[1])
                if dist < 3.0:
                    best_sim = sim
                    best_idx = i

        if best_idx < 0:
            return None

        kf_pose, _, kf_pts = candidates[best_idx]
        if len(pts_2d) < 20 or len(kf_pts) < 20:
            return None

        T4, fitness = self._icp_2d(pts_2d, kf_pts)
        if fitness < 0.5:
            return None

        # Correction = difference between ICP-refined pose and current pose
        dx   = float(T4[0, 3])
        dy   = float(T4[1, 3])
        dyaw = math.atan2(float(T4[1, 0]), float(T4[0, 0]))
        return best_idx, np.array([dx, dy, dyaw], dtype=np.float64)

    # ── 2D ICP for verification ───────────────────────────────────
    def _icp_2d(self, src_2d: np.ndarray, tgt_2d: np.ndarray,
                max_iter: int = 15):
        """Lightweight 2D point-to-point ICP.  Returns (T4×4, fitness)."""
        from config import _import_scipy
        _import_scipy()
        from scipy.spatial import cKDTree

        # Embed 2D points as 3D for reuse of existing matrix maths
        s = np.column_stack([src_2d, np.zeros(len(src_2d))])
        t = np.column_stack([tgt_2d, np.zeros(len(tgt_2d))])

        tree  = cKDTree(t)
        T_acc = np.eye(4, dtype=np.float64)
        s_w   = s.copy()

        for _ in range(max_iter):
            d, idx = tree.query(s_w, k=1)
            mask   = d < self._icp_max_dist
            if mask.sum() < 15:
                break
            ms, mt = s_w[mask], t[idx[mask]]
            cs, ct = ms.mean(0), mt.mean(0)
            H      = (ms - cs).T @ (mt - ct)
            U, _, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt[-1] *= -1
                R = Vt.T @ U.T
            tr = ct - R @ cs
            T_step = np.eye(4, dtype=np.float64)
            T_step[:3, :3] = R
            T_step[:3,  3] = tr
            s_w   = (R @ s_w.T).T + tr
            T_acc = T_step @ T_acc

        d2, _ = tree.query(s_w, k=1)
        fitness = float((d2 < self._icp_max_dist).sum() / max(len(s_w), 1))
        return T_acc, fitness
