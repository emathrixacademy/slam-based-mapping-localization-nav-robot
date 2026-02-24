"""
camera.py — Intel RealSense D4xx depth camera driver.

Provides depth + RGB point clouds aligned to the color frame.
All returned XYZ coordinates are in the camera's optical frame;
the robot module applies T_base_cam to convert to base frame.
"""
from config import _import_realsense, _import_numpy


class RealSenseCamera:
    """
    Wraps a RealSense pipeline for depth + color streaming.

    Optical frame convention (RealSense):
        X = right,  Y = down,  Z = forward (depth direction)
    """

    def __init__(self, max_depth: float = 3.0):
        _import_realsense()
        _import_numpy()
        from config import rs, np as _np

        self.max_depth = max_depth
        self.connected = False

        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,  15)
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
        self.pipeline.start(cfg)

        self.pc    = rs.pointcloud()
        self.align = rs.align(rs.stream.color)
        self.connected = True

    # ── Internal helpers ─────────────────────────────────────────
    def _get_aligned_frames(self):
        try:
            frames = self.pipeline.wait_for_frames(2000)
        except Exception:
            return None, None
        aligned = self.align.process(frames)
        depth   = aligned.get_depth_frame()
        color   = aligned.get_color_frame()
        if not depth or not color:
            return None, None
        return depth, color

    # ── Public API ───────────────────────────────────────────────
    def get_points_and_image(self):
        """
        Returns (xyz Nx3 float32, BGR image) or (None, None).
        XYZ in camera optical frame, filtered to [0.1, max_depth] metres.
        """
        from config import np
        depth, color = self._get_aligned_frames()
        if depth is None:
            return None, None

        pts  = self.pc.calculate(depth)
        self.pc.map_to(color)
        verts = (np.asanyarray(pts.get_vertices())
                   .view(np.float32).reshape(-1, 3))
        mask  = (verts[:, 2] > 0.1) & (verts[:, 2] < self.max_depth)
        img   = np.asanyarray(color.get_data())
        return verts[mask], img

    def get_colored_points_and_image(self):
        """
        Returns (xyz Nx3 float32, rgb Nx3 uint8, BGR image) or (None, None, None).
        Each point carries the colour sampled from the aligned colour frame.
        """
        from config import np
        depth, color = self._get_aligned_frames()
        if depth is None:
            return None, None, None

        pts  = self.pc.calculate(depth)
        self.pc.map_to(color)

        verts = (np.asanyarray(pts.get_vertices())
                   .view(np.float32).reshape(-1, 3))
        tex   = (np.asanyarray(pts.get_texture_coordinates())
                   .view(np.float32).reshape(-1, 2))

        mask = (verts[:, 2] > 0.1) & (verts[:, 2] < self.max_depth)
        xyz  = verts[mask]
        uv   = tex[mask]

        img  = np.asanyarray(color.get_data())   # BGR H×W×3
        h, w = img.shape[:2]
        px   = np.clip((uv[:, 0] * w).astype(int), 0, w - 1)
        py   = np.clip((uv[:, 1] * h).astype(int), 0, h - 1)
        bgr  = img[py, px]
        rgb  = bgr[:, ::-1].copy()

        return xyz, rgb, img

    def close(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass