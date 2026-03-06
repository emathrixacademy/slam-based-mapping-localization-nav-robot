"""
lidar.py — LD06 LiDAR driver and 3-D geometry helpers.

Exports:
    LD06          — serial driver (threaded background scan loop)
    lidar_to_3d   — convert scan dict → Nx3 float32 array
    transform_points, voxel_downsample, numpy_icp, save_pcd  — point-cloud utils
"""
import math
import struct
import threading
import time
import serial

from config import (
    CRC_TABLE_LD06,
    _import_numpy, _import_scipy,
)


class LD06:
    """LD06 LiDAR serial driver. Runs a background thread to collect scans."""

    # Max bytes kept in the parse buffer = 40 packets × 47 bytes.
    # If the thread falls behind (GIL starvation), older bytes are discarded
    # instead of letting the buffer grow unboundedly and the O(N) index() scan
    # getting slower with each iteration (positive-feedback hang).
    _BUFFER_MAX = 40 * 47   # ~1 880 bytes ≈ 1 full LiDAR revolution

    def __init__(self, port="/dev/ttyTHS1", offset=270):
        self.offset    = offset
        # timeout=0.05: non-blocking read; never stall the thread >50 ms per call
        self.ser       = serial.Serial(port, 230400, timeout=0.05)
        self.ser.reset_input_buffer()
        self._buffer   = bytearray()
        self.scan_data = {}
        self.raw_scan  = {}
        self._lock     = threading.Lock()
        self._scan_ts  = 0.0   # monotonic time of last published scan
        self._running  = False
        self._thread   = None

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        last_angle   = -1
        current      = {}
        current_raw  = {}

        while self._running:
            # Read whatever the OS has buffered (non-blocking; timeout=0.05 s).
            n_wait   = self.ser.in_waiting
            to_read  = max(n_wait, 47)
            incoming = self.ser.read(to_read)
            if not incoming:
                time.sleep(0.005)   # nothing yet — yield briefly then retry
                continue
            self._buffer.extend(incoming)

            # ── Buffer cap ────────────────────────────────────────────
            # If the thread fell behind (GIL starvation, heavy main-loop tick),
            # drop the oldest bytes rather than letting the buffer grow and making
            # the index() scan O(N²).  Keep only the most recent _BUFFER_MAX bytes,
            # re-aligned to the next 0x54 sync byte so we don't parse mid-packet.
            if len(self._buffer) > self._BUFFER_MAX:
                tail = self._buffer[-self._BUFFER_MAX:]
                try:
                    sync = tail.index(0x54)
                    self._buffer = tail[sync:]
                except ValueError:
                    self._buffer.clear()

            while len(self._buffer) >= 47:
                try:
                    idx = self._buffer.index(0x54)
                except ValueError:
                    self._buffer.clear()
                    break

                if idx > 0:
                    self._buffer = self._buffer[idx:]
                if len(self._buffer) < 47:
                    break
                if self._buffer[1] != 0x2C:
                    self._buffer = self._buffer[1:]
                    continue

                packet = bytes(self._buffer[:47])
                self._buffer = self._buffer[47:]

                # CRC check
                crc = 0
                for b in packet[:-1]:
                    crc = CRC_TABLE_LD06[(crc ^ b) & 0xFF]
                if crc != packet[-1]:
                    continue

                sa   = struct.unpack_from("<H", packet, 4)[0]  / 100.0
                ea   = struct.unpack_from("<H", packet, 42)[0] / 100.0
                diff = (ea - sa) % 360.0
                step = diff / 11.0

                for i in range(12):
                    off  = 6 + i * 3
                    dist = struct.unpack_from("<H", packet, off)[0]
                    conf = packet[off + 2]
                    raw_angle = (sa + step * i) % 360.0
                    corrected = (raw_angle - self.offset) % 360.0

                    if dist > 0 and conf > 20:
                        current[round(corrected, 1)]     = dist
                        current_raw[round(corrected, 1)] = (dist, conf)

                    if corrected < last_angle and last_angle > 300 and corrected < 60:
                        if len(current) > 100:
                            with self._lock:
                                self.scan_data = current.copy()
                                self.raw_scan  = current_raw.copy()
                                self._scan_ts  = time.monotonic()  # timestamp this scan
                            current.clear()
                            current_raw.clear()

                    last_angle = corrected

    def get_scan(self) -> dict:
        with self._lock:
            return self.scan_data.copy()

    def get_scan_age(self) -> float:
        """Seconds since the last complete scan was published. Returns inf if none yet."""
        ts = self._scan_ts
        return (time.monotonic() - ts) if ts > 0 else float('inf')

    def is_stale(self, max_age: float = 0.5) -> bool:
        """True if no new scan has arrived within max_age seconds."""
        return self.get_scan_age() > max_age

    def get_raw_scan(self) -> dict:
        with self._lock:
            return self.raw_scan.copy()

    def get_histogram(self, bin_size=5.0, max_range=6000) -> list:
        scan = self.get_scan()
        n    = int(360 / bin_size)
        hist = [max_range] * n
        for angle, dist in scan.items():
            if 0 < dist <= max_range:
                idx = int(angle / bin_size) % n
                if dist < hist[idx]:
                    hist[idx] = dist
        return hist

    def get_zone_distances(self, scan=None) -> dict:
        _import_numpy()
        from config import np
        if scan is None:
            scan = self.get_scan()
        _empty = {"FRONT":9999,"FRONT_RIGHT":9999,"RIGHT":9999,"BACK_RIGHT":9999,
                  "BACK":9999,"BACK_LEFT":9999,"LEFT":9999,"FRONT_LEFT":9999}
        if not scan:
            return _empty
        angles = np.fromiter(scan.keys(),   dtype=np.float32, count=len(scan))
        dists  = np.fromiter(scan.values(), dtype=np.float32, count=len(scan))
        valid  = dists > 0
        angles = angles[valid]; dists = dists[valid]
        zones  = {
            "FRONT":      (337.5, 22.5),
            "FRONT_RIGHT":(22.5,  67.5),
            "RIGHT":      (67.5,  112.5),
            "BACK_RIGHT": (112.5, 157.5),
            "BACK":       (157.5, 202.5),
            "BACK_LEFT":  (202.5, 247.5),
            "LEFT":       (247.5, 292.5),
            "FRONT_LEFT": (292.5, 337.5),
        }
        result = {}
        for name, (s, e) in zones.items():
            mask = (angles >= s) | (angles < e) if s > e else (angles >= s) & (angles < e)
            result[name] = float(dists[mask].min()) if mask.any() else 9999
        return result

    def close(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self.ser and self.ser.is_open:
            self.ser.close()


# ── Point-cloud utilities ─────────────────────────────────────────

def lidar_to_3d(scan_dict: dict, height: float = 0.0):
    """Convert {angle_deg: dist_mm} → Nx3 float32 array (x, y, height).
    Vectorised — ~50x faster than per-point Python loop.
    CW angle convention: 0°=forward, 90°=right, 270°=left.
    """
    _import_numpy()
    from config import np
    if not scan_dict:
        return np.empty((0, 3), dtype=np.float32)
    angles = np.fromiter(scan_dict.keys(),   dtype=np.float32, count=len(scan_dict))
    dists  = np.fromiter(scan_dict.values(), dtype=np.float32, count=len(scan_dict))
    valid  = (dists >= 20) & (dists < 12000)
    angles = angles[valid];  dists = dists[valid]
    if len(angles) == 0:
        return np.empty((0, 3), dtype=np.float32)
    rad = np.deg2rad(angles)
    d   = dists * (1.0 / 1000.0)
    pts = np.empty((len(d), 3), dtype=np.float32)
    pts[:, 0] =  d * np.cos(rad)   # x = forward
    pts[:, 1] = -d * np.sin(rad)   # y = left  (negate for CW)
    pts[:, 2] =  height
    return pts


def transform_points(pts, T):
    """Apply 4×4 homogeneous transform T to Nx3 point array. GPU-accelerated when CuPy is available."""
    if len(pts) == 0:
        return pts
    from config import np, cp, cuda_ok
    if cuda_ok and cp is not None and len(pts) > 1000:
        try:
            g = cp.asarray(pts,        dtype=cp.float32)
            R = cp.asarray(T[:3, :3],  dtype=cp.float32)
            t = cp.asarray(T[:3, 3],   dtype=cp.float32)
            return cp.asnumpy(g @ R.T + t).astype(np.float32)
        except Exception:
            pass
    return (pts @ T[:3, :3].T + T[:3, 3]).astype(np.float32)


def voxel_downsample(pts, voxel_size: float):
    """Hash-based voxel grid downsampling. GPU-accelerated when CuPy is available."""
    if len(pts) == 0:
        return pts
    from config import np, cp, cuda_ok
    if cuda_ok and cp is not None and len(pts) > 500:
        try:
            g      = cp.asarray(pts)
            keys   = (g / voxel_size).astype(cp.int64)
            hashes = (keys[:, 0] * cp.int64(73856093)
                      ^ keys[:, 1] * cp.int64(19349669)
                      ^ keys[:, 2] * cp.int64(83492791))
            _, idx = cp.unique(hashes, return_index=True)
            return cp.asnumpy(g[idx])
        except Exception:
            pass
    keys   = (pts / voxel_size).astype(np.int64)
    hashes = (keys[:, 0] * 73856093
              ^ keys[:, 1] * 19349669
              ^ keys[:, 2] * 83492791)
    _, idx = np.unique(hashes, return_index=True)
    return pts[idx]


def voxel_downsample_idx(pts, voxel_size: float):
    """Returns index array for voxel downsampling — keeps parallel arrays (e.g. depth) in sync."""
    if len(pts) == 0:
        from config import np
        return np.arange(0, dtype=np.int64)
    from config import np, cp, cuda_ok
    if cuda_ok and cp is not None and len(pts) > 500:
        try:
            g      = cp.asarray(pts)
            keys   = (g / voxel_size).astype(cp.int64)
            hashes = (keys[:, 0] * cp.int64(73856093)
                      ^ keys[:, 1] * cp.int64(19349669)
                      ^ keys[:, 2] * cp.int64(83492791))
            _, idx = cp.unique(hashes, return_index=True)
            return cp.asnumpy(idx)
        except Exception:
            pass
    keys   = (pts / voxel_size).astype(np.int64)
    hashes = (keys[:, 0] * 73856093
              ^ keys[:, 1] * 19349669
              ^ keys[:, 2] * 83492791)
    _, idx = np.unique(hashes, return_index=True)
    return idx


def numpy_icp(source, target, max_iter=25, tol=1e-5, max_dist=0.5):
    """
    Point-to-point ICP.  Returns (T_4x4, fitness_ratio).
    Tries Open3D CUDA tensor ICP first; falls back to CPU scipy cKDTree.
    """
    _import_numpy()
    _import_scipy()
    from config import np, cKDTree, o3d_cuda_ok, _import_o3d

    if len(source) < 50 or len(target) < 50:
        return np.eye(4, dtype=np.float64), 0.0

    src = voxel_downsample(source, 0.05).astype(np.float64)
    tgt = voxel_downsample(target, 0.05).astype(np.float64)

    if len(src) < 30 or len(tgt) < 30:
        return np.eye(4, dtype=np.float64), 0.0

    # ── Open3D CUDA tensor ICP ────────────────────────────────────
    if o3d_cuda_ok:
        try:
            _import_o3d()
            from config import o3d
            dev  = o3d.core.Device("CUDA:0")
            f32  = o3d.core.float32
            src_t = o3d.t.geometry.PointCloud(device=dev)
            src_t.point.positions = o3d.core.Tensor(
                src.astype(np.float32), dtype=f32, device=dev)
            tgt_t = o3d.t.geometry.PointCloud(device=dev)
            tgt_t.point.positions = o3d.core.Tensor(
                tgt.astype(np.float32), dtype=f32, device=dev)
            tgt_t.estimate_normals(max_nn=20, radius=0.3)
            result = o3d.t.pipelines.registration.icp(
                src_t, tgt_t, max_dist,
                o3d.core.Tensor(np.eye(4, dtype=np.float32)),   # init on CPU
                o3d.t.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.t.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=max_iter,
                    relative_fitness=tol,
                    relative_rmse=tol),
            )
            T_out   = result.transformation.numpy().astype(np.float64)
            fitness = float(result.fitness)
            return T_out, fitness
        except Exception:
            pass   # fall through to CPU ICP

    # ── CPU fallback: point-to-point ICP via scipy cKDTree ────────
    tree     = cKDTree(tgt)
    T_acc    = np.eye(4, dtype=np.float64)
    src_w    = src.copy()
    prev_err = float('inf')

    for _ in range(max_iter):
        dists, indices = tree.query(src_w, k=1)
        mask = dists < max_dist
        if mask.sum() < 20:
            break

        ms, mt = src_w[mask], tgt[indices[mask]]
        cs, ct = ms.mean(0), mt.mean(0)
        H      = (ms - cs).T @ (mt - ct)
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        t = ct - R @ cs

        T_step = np.eye(4, dtype=np.float64)
        T_step[:3, :3] = R
        T_step[:3, 3]  = t
        src_w = (R @ src_w.T).T + t
        T_acc = T_step @ T_acc

        err = dists[mask].mean()
        if abs(prev_err - err) < tol:
            break
        prev_err = err

    fd, _ = tree.query(src_w, k=1)
    fitness = float((fd < max_dist).sum() / len(src_w))
    return T_acc, fitness


def save_pcd(filepath: str, points):
    """Write a minimal ASCII .pcd file."""
    n = len(points)
    with open(filepath, 'w') as f:
        f.write("# .PCD v0.7\nVERSION 0.7\nFIELDS x y z\n"
                "SIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n")
        f.write(f"WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {n}\nDATA ascii\n")
        for p in points:
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")