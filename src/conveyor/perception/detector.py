"""OpenCV and calibrated depth multi-object detector for conveyor RGB-D frames.

This module does not import RAI or the simulation package. It accepts any
duck-typed camera observation with ``simulation_time_s``, ``rgb``, ``depth_m``
and ``calibration`` (matching :class:`conveyor.simulation.camera.CameraObservation`).
"""

from __future__ import annotations

from math import atan2, cos, hypot, isfinite, pi, sin
from typing import Any

from conveyor.grasping.geometry import wrap_half_turn
from conveyor.types import ObjectObservation

from .models import DetectedObject, PerceptionConfig


class RGBDObjectDetector:
    """Detect multiple conveyor objects and estimate their 3-D pose and geometry.

    Pipeline per frame:
    1. Vectorised deprojection of valid optical-axis depth pixels into world
       ``(x, y, z)`` using the supplied camera intrinsics and ``camera_to_world``.
    2. Spatial ROI filtering over the conveyor belt bounds and height above the
       belt top surface (``min_object_height_m <= z - belt_top_z_m <= max_object_height_m``).
    3. Connected-component segmentation combined with colour/height clustering
       so adjacent objects in heavy clutter are cleanly separated even if their
       pixel masks touch by 1-2 pixels.
    4. World-frame footprint analysis (oriented minimum-area bounding box vs
       enclosing circle, contour IoU and circularity) to estimate ``box`` vs
       ``cylinder``, 3-D centroid ``(x, y, z_center)``, full extents
       ``(dx, dy, dz)`` and planar orientation ``yaw_rad``.
    """

    def __init__(self, config: PerceptionConfig | None = None) -> None:
        self.config = config or PerceptionConfig()

    def detect(self, observation: Any) -> list[DetectedObject]:
        """Detect all objects on the conveyor belt from an RGB-D observation."""
        import cv2
        import numpy as np

        sim_time = float(observation.simulation_time_s)
        if not isfinite(sim_time) or sim_time < 0.0:
            raise ValueError("observation.simulation_time_s must be finite and non-negative")

        rgb = np.asarray(observation.rgb, dtype=np.uint8)
        depth_m = np.asarray(observation.depth_m, dtype=np.float32)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("observation.rgb must have shape (H, W, 3)")
        if depth_m.ndim != 2 or depth_m.shape != rgb.shape[:2]:
            raise ValueError("observation.depth_m must have shape (H, W) matching rgb")

        calibration = observation.calibration
        height, width = depth_m.shape
        intrinsics = np.asarray(calibration.intrinsics, dtype=np.float64)
        cam_to_world = np.asarray(calibration.camera_to_world, dtype=np.float64)
        fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
        cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])

        # 1. Deproject all valid depth pixels to world coordinates in one vectorised pass.
        valid_mask = np.isfinite(depth_m) & (depth_m > 0.0)
        if not np.any(valid_mask):
            return []

        v_idx, u_idx = np.nonzero(valid_mask)
        z_cam = depth_m[v_idx, u_idx].astype(np.float64)
        x_cam = (u_idx.astype(np.float64) - cx) * z_cam / fx
        y_cam = (v_idx.astype(np.float64) - cy) * z_cam / fy

        rot = cam_to_world[:3, :3]
        trans = cam_to_world[:3, 3]
        pts_cam = np.stack((x_cam, y_cam, z_cam), axis=0)  # (3, N)
        pts_world = rot @ pts_cam + trans[:, None]         # (3, N)

        wx, wy, wz = pts_world[0], pts_world[1], pts_world[2]
        cfg = self.config
        x_min = cfg.belt_x_limits_m[0] + cfg.roi_margin_m
        x_max = cfg.belt_x_limits_m[1] - cfg.roi_margin_m
        y_min = cfg.belt_y_limits_m[0] + cfg.roi_margin_m
        y_max = cfg.belt_y_limits_m[1] - cfg.roi_margin_m
        z_min = cfg.belt_top_z_m + cfg.min_object_height_m
        z_max = cfg.belt_top_z_m + cfg.max_object_height_m

        in_roi = (wx >= x_min) & (wx <= x_max) & (wy >= y_min) & (wy <= y_max) & (wz >= z_min) & (wz <= z_max)
        if not np.any(in_roi):
            return []

        roi_v = v_idx[in_roi]
        roi_u = u_idx[in_roi]
        roi_wx = wx[in_roi]
        roi_wy = wy[in_roi]
        roi_wz = wz[in_roi]

        # Build dense world-coordinate and foreground maps over the image grid.
        fg_mask = np.zeros((height, width), dtype=np.uint8)
        fg_mask[roi_v, roi_u] = 255
        world_x_map = np.full((height, width), np.nan, dtype=np.float64)
        world_y_map = np.full((height, width), np.nan, dtype=np.float64)
        world_z_map = np.full((height, width), np.nan, dtype=np.float64)
        world_x_map[roi_v, roi_u] = roi_wx
        world_y_map[roi_v, roi_u] = roi_wy
        world_z_map[roi_v, roi_u] = roi_wz

        rgb_norm = rgb.astype(np.float32) / 255.0

        # 2. Connected components + colour/height splitting for cluttered blobs.
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg_mask, connectivity=8)
        component_masks: list[np.ndarray] = []
        for label_id in range(1, num_labels):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if area < cfg.min_contour_pixels:
                continue
            x0 = int(stats[label_id, cv2.CC_STAT_LEFT])
            y0 = int(stats[label_id, cv2.CC_STAT_TOP])
            w0 = int(stats[label_id, cv2.CC_STAT_WIDTH])
            h0 = int(stats[label_id, cv2.CC_STAT_HEIGHT])
            sub_labels = labels[y0 : y0 + h0, x0 : x0 + w0]
            comp_local = (sub_labels == label_id)
            split_submasks = self._split_cluttered_component(
                comp_local,
                rgb_norm[y0 : y0 + h0, x0 : x0 + w0],
                world_z_map[y0 : y0 + h0, x0 : x0 + w0],
            )
            for sub_mask in split_submasks:
                if int(np.count_nonzero(sub_mask)) < cfg.min_contour_pixels:
                    continue
                full_mask = np.zeros((height, width), dtype=np.uint8)
                full_mask[y0 : y0 + h0, x0 : x0 + w0][sub_mask] = 255
                component_masks.append(full_mask)

        # 3. Estimate 3-D pose and geometry for each separated object mask.
        raw_detections: list[DetectedObject] = []
        for mask in component_masks:
            det = self._estimate_object_from_mask(
                mask=mask,
                sim_time=sim_time,
                rgb_norm=rgb_norm,
                world_x_map=world_x_map,
                world_y_map=world_y_map,
                world_z_map=world_z_map,
                fx=fx,
                fy=fy,
                camera_z_m=float(cam_to_world[2, 3]),
            )
            if det is not None:
                raw_detections.append(det)

        # Sort deterministically from upstream (-x) to downstream (+x), then by y.
        raw_detections.sort(key=lambda d: (round(d.position_m[0], 4), round(d.position_m[1], 4)))
        final_detections: list[DetectedObject] = []
        for idx, det in enumerate(raw_detections, start=1):
            final_detections.append(
                DetectedObject(
                    detection_id=f"det_{idx:03d}",
                    simulation_time_s=det.simulation_time_s,
                    position_m=det.position_m,
                    dimensions_m=det.dimensions_m,
                    yaw_rad=det.yaw_rad,
                    geometry_type=det.geometry_type,
                    color_rgb=det.color_rgb,
                    pixel_centroid_uv=det.pixel_centroid_uv,
                    pixel_area=det.pixel_area,
                    confidence=det.confidence,
                )
            )
        return final_detections

    def detect_observations(self, observation: Any) -> list[ObjectObservation]:
        """Return shared :class:`ObjectObservation` records for tracking."""
        return [det.to_observation() for det in self.detect(observation)]

    def _split_cluttered_component(
        self,
        comp_local: Any,
        rgb_patch: Any,
        z_patch: Any,
    ) -> list[Any]:
        """Split a connected component if it merges multiple distinct colours or heights."""
        import cv2
        import numpy as np

        ys, xs = np.nonzero(comp_local)
        if len(xs) < 2 * self.config.min_contour_pixels:
            return [comp_local]

        colors = rgb_patch[ys, xs]  # (N, 3)
        z_vals = z_patch[ys, xs]    # (N,)

        # Greedy quantization of colour + top height modes within the blob.
        color_tol = self.config.color_split_distance_rgb
        z_tol = self.config.depth_layer_tolerance_m * 1.5

        med_color = np.median(colors, axis=0)
        med_z = float(np.median(z_vals))
        max_color_dev = float(np.max(np.linalg.norm(colors - med_color[None, :], axis=1)))
        max_z_dev = float(np.max(np.abs(z_vals - med_z)))

        if max_color_dev <= color_tol and max_z_dev <= z_tol:
            return [comp_local]

        # Seed clusters from well-separated pixels.
        Remaining = np.ones(len(xs), dtype=bool)
        submasks: list[Any] = []
        while np.count_nonzero(Remaining) >= self.config.min_contour_pixels:
            rem_idx = np.nonzero(Remaining)[0]
            seed_c = colors[rem_idx[0]]
            seed_z = z_vals[rem_idx[0]]
            dist_c = np.linalg.norm(colors[rem_idx] - seed_c[None, :], axis=1)
            dist_z = np.abs(z_vals[rem_idx] - seed_z)
            in_cluster = (dist_c <= color_tol) & (dist_z <= z_tol)
            chosen = rem_idx[in_cluster]
            if len(chosen) >= self.config.min_contour_pixels:
                # Re-center once around cluster median for clean boundaries.
                c_med = np.median(colors[chosen], axis=0)
                z_med = float(np.median(z_vals[chosen]))
                dist_c2 = np.linalg.norm(colors[rem_idx] - c_med[None, :], axis=1)
                dist_z2 = np.abs(z_vals[rem_idx] - z_med)
                in_cluster2 = (dist_c2 <= color_tol) & (dist_z2 <= z_tol)
                chosen = rem_idx[in_cluster2]

            Remaining[chosen] = False
            if len(chosen) < self.config.min_contour_pixels:
                continue
            cluster_img = np.zeros_like(comp_local, dtype=np.uint8)
            cluster_img[ys[chosen], xs[chosen]] = 255
            n_sub, sub_lbl, sub_stats, _ = cv2.connectedComponentsWithStats(cluster_img, connectivity=8)
            for s_id in range(1, n_sub):
                if int(sub_stats[s_id, cv2.CC_STAT_AREA]) >= self.config.min_contour_pixels:
                    submasks.append(sub_lbl == s_id)

        return submasks if submasks else [comp_local]

    def _estimate_object_from_mask(
        self,
        *,
        mask: Any,
        sim_time: float,
        rgb_norm: Any,
        world_x_map: Any,
        world_y_map: Any,
        world_z_map: Any,
        fx: float,
        fy: float,
        camera_z_m: float,
    ) -> DetectedObject | None:
        import cv2
        import numpy as np

        cfg = self.config
        ys, xs = np.nonzero(mask > 0)
        pixel_area = int(len(xs))
        if pixel_area < cfg.min_contour_pixels:
            return None

        z_all = world_z_map[ys, xs]
        # Top surface height: use the 80th percentile to ignore any edge-pixel graze.
        z_top_ref = float(np.percentile(z_all, 80))
        top_layer = np.abs(z_all - z_top_ref) <= cfg.depth_layer_tolerance_m
        if int(np.count_nonzero(top_layer)) < cfg.min_contour_pixels:
            top_layer = np.ones_like(z_all, dtype=bool)

        top_ys = ys[top_layer]
        top_xs = xs[top_layer]
        z_top = float(np.median(z_all[top_layer]))
        height_m = z_top - cfg.belt_top_z_m
        if not (cfg.min_object_height_m <= height_m <= cfg.max_object_height_m):
            return None
        z_center = cfg.belt_top_z_m + 0.5 * height_m

        top_mask = np.zeros_like(mask, dtype=np.uint8)
        top_mask[top_ys, top_xs] = 255

        # Mean RGB colour of the top surface.
        mean_rgb = tuple(float(np.clip(c, 0.0, 1.0)) for c in np.median(rgb_norm[top_ys, top_xs], axis=0))
        u_mean = float(np.mean(top_xs))
        v_mean = float(np.mean(top_ys))

        # World (x, y) coordinates of all top-surface pixels.
        wx_pts = world_x_map[top_ys, top_xs].astype(np.float32)
        wy_pts = world_y_map[top_ys, top_xs].astype(np.float32)

        # Pixel size in metres at the object's top depth.
        top_depth_m = max(0.1, camera_z_m - z_top)
        pixel_pitch_x = top_depth_m / fx
        pixel_pitch_y = top_depth_m / fy
        pixel_pitch_m = 0.5 * (pixel_pitch_x + pixel_pitch_y)
        pixel_area_m2 = pixel_pitch_x * pixel_pitch_y

        # Fit minimum-area rectangle in world (x, y) coordinates.
        # Because pixels represent cell centers of width pixel_pitch_m, expand by 1 pixel pitch.
        pts_xy = np.stack((wx_pts, wy_pts), axis=1).reshape(-1, 1, 2)
        (rect_cx, rect_cy), (rect_w, rect_h), rect_angle_deg = cv2.minAreaRect(pts_xy)

        # OpenCV minAreaRect angle is in [-90, 0) or (0, 90] degrees in the (x, y) plane.
        theta_rad = float(rect_angle_deg) * pi / 180.0
        # Compute exact rotated projections around the median/mean to avoid quantization bias.
        axis_u = np.array([cos(theta_rad), sin(theta_rad)], dtype=np.float64)
        axis_v = np.array([-sin(theta_rad), cos(theta_rad)], dtype=np.float64)
        pts_xy64 = np.stack((wx_pts.astype(np.float64), wy_pts.astype(np.float64)), axis=1)
        proj_u = pts_xy64 @ axis_u
        proj_v = pts_xy64 @ axis_v

        u_min, u_max = float(np.min(proj_u)), float(np.max(proj_u))
        v_min, v_max = float(np.min(proj_v)), float(np.max(proj_v))
        span_u = (u_max - u_min) + pixel_pitch_m
        span_v = (v_max - v_min) + pixel_pitch_m
        center_u = 0.5 * (u_min + u_max)
        center_v = 0.5 * (v_min + v_max)
        box_cx = float(center_u * axis_u[0] + center_v * axis_v[0])
        box_cy = float(center_u * axis_u[1] + center_v * axis_v[1])

        # Canonicalise box dimensions so dx >= dy, adjusting yaw accordingly.
        if span_u >= span_v:
            dx_box, dy_box = span_u, span_v
            yaw_box = wrap_half_turn(theta_rad)
        else:
            dx_box, dy_box = span_v, span_u
            yaw_box = wrap_half_turn(theta_rad + pi / 2.0)

        # Shape classification: compare filled mask against ideal box vs ideal circle in pixel space.
        geometry_type, diameter_m, shape_conf = self._classify_shape(
            top_mask=top_mask,
            top_xs=top_xs,
            top_ys=top_ys,
            pts_xy64=pts_xy64,
            box_center=(box_cx, box_cy),
            dx_box=dx_box,
            dy_box=dy_box,
            yaw_box=yaw_box,
            pixel_pitch_m=pixel_pitch_m,
            pixel_area_m2=pixel_area_m2,
        )

        if geometry_type == "cylinder":
            cx_world = float(np.mean(pts_xy64[:, 0]))
            cy_world = float(np.mean(pts_xy64[:, 1]))
            dimensions_m = (diameter_m, diameter_m, float(height_m))
            yaw_rad = 0.0
            position_m = (cx_world, cy_world, float(z_center))
        else:
            dimensions_m = (float(dx_box), float(dy_box), float(height_m))
            yaw_rad = float(yaw_box)
            position_m = (box_cx, box_cy, float(z_center))

        return DetectedObject(
            detection_id="det_raw",
            simulation_time_s=sim_time,
            position_m=position_m,
            dimensions_m=dimensions_m,
            yaw_rad=yaw_rad,
            geometry_type=geometry_type,
            color_rgb=mean_rgb,
            pixel_centroid_uv=(u_mean, v_mean),
            pixel_area=pixel_area,
            confidence=shape_conf,
        )

    def _classify_shape(
        self,
        *,
        top_mask: Any,
        top_xs: Any,
        top_ys: Any,
        pts_xy64: Any,
        box_center: tuple[float, float],
        dx_box: float,
        dy_box: float,
        yaw_box: float,
        pixel_pitch_m: float,
        pixel_area_m2: float,
    ) -> tuple[str, float, float]:
        """Distinguish upright boxes from upright cylinders using 2-D footprint geometry."""
        import numpy as np

        n_pts = len(pts_xy64)
        measured_area_m2 = n_pts * pixel_area_m2
        aspect_ratio = dx_box / max(dy_box, 1e-6)

        # 1. Diameter estimate from equivalent area (exact up to pixelation)
        cx_mean = float(np.mean(pts_xy64[:, 0]))
        cy_mean = float(np.mean(pts_xy64[:, 1]))
        diam_from_area = 2.0 * float(np.sqrt(measured_area_m2 / pi))
        diameter_m = diam_from_area

        # A cylinder viewed from straight above has aspect ratio ~ 1.0.
        if aspect_ratio > 1.18:
            return "box", diameter_m, 0.96

        # 2. Check across multiple orientations whether the 2-D footprint has square corners
        # or is radially isotropic (a disc has extent ~ pi/4 = 0.7854 in EVERY bounding box orientation,
        # whereas a square box reaches extent ~ 1.0 along its true axes and ~ 0.5 at 45 degrees).
        extents: list[float] = []
        best_rect_extent = 0.0
        for step in range(18):
            angle = step * (pi / 36.0)
            c, s = cos(angle), sin(angle)
            pu = c * (pts_xy64[:, 0] - cx_mean) + s * (pts_xy64[:, 1] - cy_mean)
            pv = -s * (pts_xy64[:, 0] - cx_mean) + c * (pts_xy64[:, 1] - cy_mean)
            su = float(np.max(pu) - np.min(pu)) + pixel_pitch_m
            sv = float(np.max(pv) - np.min(pv)) + pixel_pitch_m
            ext = measured_area_m2 / max(su * sv, 1e-9)
            extents.append(ext)
            if ext > best_rect_extent:
                best_rect_extent = ext

        min_rect_extent = min(extents)
        extent_range = best_rect_extent - min_rect_extent

        # Also measure circular IoU: how well the pixel set matches a disc of equal area centered at (cx_mean, cy_mean).
        radius_eq = 0.5 * diam_from_area + 0.35 * pixel_pitch_m
        radii = np.hypot(pts_xy64[:, 0] - cx_mean, pts_xy64[:, 1] - cy_mean)
        circle_inlier_fraction = float(np.count_nonzero(radii <= radius_eq)) / max(n_pts, 1)

        # For a true circle, extent_range across 0..90 deg is small (< 0.16) and circle_inlier_fraction is near 1.0.
        # For a square/near-square box, extent_range is large (~ 0.25-0.45) and corners stick outside radius_eq.
        if (extent_range < 0.175 and best_rect_extent < 0.885) or (circle_inlier_fraction >= 0.965 and extent_range < 0.22):
            conf = float(min(1.0, max(0.65, circle_inlier_fraction)))
            return "cylinder", diameter_m, conf

        conf = float(min(1.0, max(0.65, best_rect_extent)))
        return "box", diameter_m, conf
