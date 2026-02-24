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
import serial

from config import (
    CRC_TABLE_LD06,
    _import_numpy, _import_scipy,
)


class LD06:
    """LD06 LiDAR serial driver. Runs a background thread to collect scans."""

    def __init__(self, port="/dev/ttyTHS1", offset=270):
        self.offset    = offset
        self.ser       = serial.Serial(port, 230400, timeout=1.0)
        self.ser.reset_input_buffer()
        self._buffer   = bytearray()
        self.scan_data = {}
        self.raw_scan  = {}
        self._lock     = threading.Lock()
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
            incoming = self.ser.read(128)
            if not incoming:
                continue
            self._buffer.extend(incoming)

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
                            current.clear()
                            current_raw.clear()

                    last_angle = corrected

    def get_scan(self) -> dict:
        with self._lock:
            return self.scan_data.copy()

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
        if scan is None:
            scan = self.get_scan()
        zones = {
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
            dists = []
            for a, d in scan.items():
                if d <= 0: continue
                in_zone = (a >= s or a < e) if s > e else (s <= a < e)
                if in_zone:
                    dists.append(d)
            result[name] = min(dists) if dists else 9999
        return result

    def close(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self.ser and self.ser.is_open:
            self.ser.close()


# ── Point-cloud utilities ─────────────────────────────────────────

def lidar_to_3d(scan_dict: dict, height: float = 0.0):
    """Convert {angle_deg: dist_mm} → Nx3 float32 array (x, y, height)."""
    _import_numpy()
    from config import np
    pts = []
    for ang, dist in scan_dict.items():
        if 10 < dist < 12000:
            rad = math.radians(float(ang))
            d   = dist / 1000.0
            pts.append([d * math.cos(rad), d * math.sin(rad), height])
    return (np.array(pts, dtype=np.float32)
            if pts else np.empty((0, 3), dtype=np.float32))


def transform_points(pts, T):
    """Apply 4×4 homogeneous transform T to Nx3 point array."""
    if len(pts) == 0:
        return pts
    from config import np
    return (pts @ T[:3, :3].T + T[:3, 3]).astype(np.float32)


def voxel_downsample(pts, voxel_size: float):
    """Hash-based voxel grid downsampling — fast O(N)."""
    if len(pts) == 0:
        return pts
    from config import np
    keys   = (pts / voxel_size).astype(np.int64)
    hashes = (keys[:, 0] * 73856093
              ^ keys[:, 1] * 19349669
              ^ keys[:, 2] * 83492791)
    _, idx = np.unique(hashes, return_index=True)
    return pts[idx]


def numpy_icp(source, target, max_iter=25, tol=1e-5, max_dist=0.5):
    """
    Point-to-point ICP using scipy cKDTree.
    Returns (T_4x4, fitness_ratio).
    """
    _import_numpy()
    _import_scipy()
    from config import np, cKDTree

    if len(source) < 50 or len(target) < 50:
        return np.eye(4, dtype=np.float64), 0.0

    src = voxel_downsample(source, 0.05).astype(np.float64)
    tgt = voxel_downsample(target, 0.05).astype(np.float64)

    if len(src) < 30 or len(tgt) < 30:
        return np.eye(4, dtype=np.float64), 0.0

    tree  = cKDTree(tgt)
    T_acc = np.eye(4, dtype=np.float64)
    src_w = src.copy()
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