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

        # Retry up to 3 times — handles dirty USB state from previous crashes
        last_err = None
        for attempt in range(3):
            try:
                self.pipeline = rs.pipeline()
                cfg = rs.config()
                cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,  30)
                cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                profile = self.pipeline.start(cfg)
                # High Accuracy preset — ignore if camera rejects XU control
                try:
                    depth_sensor = profile.get_device().first_depth_sensor()
                    if depth_sensor.supports(rs.option.visual_preset):
                        depth_sensor.set_option(rs.option.visual_preset, 3)  # High Density
                    # Cache depth scale — avoids IPC call every frame
                    self._depth_scale = depth_sensor.get_depth_scale()
                except Exception:
                    self._depth_scale = 0.001   # standard RealSense default
                last_err = None
                break
            except Exception as e:
                last_err = e
                try: self.pipeline.stop()
                except Exception: pass
                import time as _t; _t.sleep(2)
        if last_err:
            raise last_err

        self.pc    = rs.pointcloud()
        self.align = rs.align(rs.stream.color)

        # ── Depth post-processing pipeline ───────────────────────────
        # Order: align → depth_to_disparity → spatial → temporal
        #        → disparity_to_depth → hole_fill
        #
        # Running spatial + temporal in DISPARITY space is critical:
        # the filters are designed for disparity (1/depth), not raw depth.
        # In disparity space, far objects have small values so the filter
        # step-size thresholds behave consistently across the depth range.
        self._depth_to_disparity = rs.disparity_transform(True)
        self._disparity_to_depth = rs.disparity_transform(False)

        # Spatial — edge-preserving fill, 3 passes for more hole coverage
        self._spatial = rs.spatial_filter()
        self._spatial.set_option(rs.option.filter_magnitude,    3)    # passes (1-5)
        self._spatial.set_option(rs.option.filter_smooth_alpha, 0.5)  # smoothing weight
        self._spatial.set_option(rs.option.filter_smooth_delta, 20)   # step threshold

        # Temporal — blends up to 4 frames; alpha=0.4 gives strong smoothing
        self._temporal = rs.temporal_filter()
        self._temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
        self._temporal.set_option(rs.option.filter_smooth_delta, 20)

        # Hole fill — nearest_from_around (2) best for object interior surfaces
        self._hole_fill = rs.hole_filling_filter()
        self._hole_fill.set_option(rs.option.holes_fill, 2)

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
        # Disparity-space filter pipeline (much better quality than raw-depth filtering)
        depth = self._depth_to_disparity.process(depth)
        depth = self._spatial.process(depth)
        depth = self._temporal.process(depth)
        depth = self._disparity_to_depth.process(depth)
        depth = self._hole_fill.process(depth).as_depth_frame()
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

    def get_intrinsics(self):
        """
        Returns (fx, fy, cx, cy, width, height) from the live color stream.
        Call after the pipeline is started.
        """
        from config import rs
        profile = self.pipeline.get_active_profile()
        intr = (profile.get_stream(rs.stream.color)
                       .as_video_stream_profile()
                       .get_intrinsics())
        return intr.fx, intr.fy, intr.ppx, intr.ppy, intr.width, intr.height

    def get_full_frame(self):
        """
        Single pipeline read returning everything needed for SLAM.

        Returns (xyz Nx3 float32, rgb Nx3 uint8, bgr_img HxWx3 uint8,
                 depth_m HxW float32) or (None, None, None, None).

        depth_m: aligned depth in METRES (float32) — for RGB-D odometry.
        xyz:     point cloud in camera optical frame.
        """
        from config import rs, np
        depth, color = self._get_aligned_frames()
        if depth is None:
            return None, None, None, None

        # Raw depth image in metres (use cached scale — no IPC overhead)
        depth_m = np.asanyarray(depth.get_data()).astype(np.float32) * self._depth_scale

        # Point cloud + per-point colour (reuse existing logic)
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

        return xyz, rgb, img, depth_m

    def close(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass