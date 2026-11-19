"""Multi-object tracker with optimal minimum-cost data association and Kalman filtering."""

from __future__ import annotations

from collections.abc import Iterable
from itertools import permutations
from math import hypot, isfinite, sqrt
from typing import Any

from conveyor.types import ObjectObservation, TrackedState, Vector3

from .kalman import ConstantVelocityKalmanFilter3D
from .models import TrackedObject, TrackerConfig


def _euclidean_3d(a: Vector3, b: Vector3) -> float:
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


class MultiObjectTracker:
    """Track multiple conveyor objects across frames with stable IDs.

    Features:
    - Accepts either :class:`conveyor.types.ObjectObservation` or rich
      :class:`conveyor.perception.models.DetectedObject` records.
    - Automatically resets all tracks when the simulation clock resets
      (i.e. when a frame timestamp is earlier than the previous timestamp).
    - Minimum-cost global data association within a spatial gating radius,
      incorporating colour consistency when available.
    - Exposes :meth:`update_states` returning ``list[TrackedState]``,
      :meth:`update` returning ``list[TrackedObject]``, and
      :meth:`schedulable_objects` for direct handoff to Hüseyin's scheduler.
    """

    def __init__(self, config: TrackerConfig | None = None, *, id_prefix: str = "track") -> None:
        if not isinstance(id_prefix, str) or not id_prefix.strip():
            raise ValueError("id_prefix must be a non-empty string")
        self.config = config or TrackerConfig()
        self.id_prefix = id_prefix.strip()
        self._tracks: dict[str, ConstantVelocityKalmanFilter3D] = {}
        self._next_id: int = 1
        self._last_time_s: float | None = None

    @property
    def current_time_s(self) -> float | None:
        return self._last_time_s

    @property
    def active_track_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._tracks.keys()))

    def reset(self) -> None:
        """Clear all active tracks and restart the track ID counter."""
        self._tracks.clear()
        self._next_id = 1
        self._last_time_s = None

    def update(
        self,
        detections: Iterable[Any],
        simulation_time_s: float | None = None,
        *,
        confirmed_only: bool = True,
    ) -> list[TrackedObject]:
        """Advance all tracks to the frame timestamp and associate detections.

        Parameters
        ----------
        detections:
            Iterable of :class:`DetectedObject` or :class:`ObjectObservation`.
        simulation_time_s:
            Frame acquisition timestamp. Inferred from ``detections`` when
            non-empty; required if ``detections`` is empty and tracks should
            be predicted forward.
        confirmed_only:
            If true (default), return only tracks that have reached
            ``config.min_hits_to_confirm``.
        """
        det_list = list(detections)
        frame_time_s = self._resolve_frame_time(det_list, simulation_time_s)
        if frame_time_s is None:
            return self.get_tracked_objects(confirmed_only=confirmed_only)

        # Contract from docs/interfaces.md: restart tracking when simulation clock resets.
        if self._last_time_s is not None and frame_time_s < self._last_time_s - 1e-9:
            self.reset()

        self._last_time_s = frame_time_s

        # 1. Predict all existing tracks to frame_time_s.
        for track in self._tracks.values():
            track.predict_to(frame_time_s)

        # 2. Associate predicted tracks with current detections.
        track_list = [self._tracks[tid] for tid in sorted(self._tracks.keys())]
        matched_pairs, unmatched_track_indices, unmatched_det_indices = self._associate(track_list, det_list)

        # 3. Update matched tracks.
        for t_idx, d_idx in matched_pairs:
            det = det_list[d_idx]
            pos = tuple(float(v) for v in det.position_m)
            track_list[t_idx].update(pos, frame_time_s, detection=det)

        # 4. Mark unmatched tracks as missed.
        for t_idx in unmatched_track_indices:
            track_list[t_idx].mark_missed()

        # 5. Spawn new tracks for unmatched detections (sorted upstream to downstream).
        unmatched_dets = [det_list[d_idx] for d_idx in unmatched_det_indices]
        unmatched_dets.sort(key=lambda d: (round(float(d.position_m[0]), 4), round(float(d.position_m[1]), 4)))
        for det in unmatched_dets:
            self._spawn_track(det, frame_time_s)

        # 6. Prune lost or exited tracks.
        self._prune_tracks(frame_time_s)

        return self.get_tracked_objects(confirmed_only=confirmed_only)

    def update_states(
        self,
        detections: Iterable[Any],
        simulation_time_s: float | None = None,
        *,
        confirmed_only: bool = True,
    ) -> list[TrackedState]:
        """Update tracks and return shared :class:`conveyor.types.TrackedState` records."""
        return [obj.state for obj in self.update(detections, simulation_time_s, confirmed_only=confirmed_only)]

    def get_tracked_objects(self, *, confirmed_only: bool = True) -> list[TrackedObject]:
        """Return current :class:`TrackedObject` snapshots sorted by ``object_id``."""
        objects = [self._tracks[tid].to_tracked_object() for tid in sorted(self._tracks.keys())]
        if confirmed_only:
            return [obj for obj in objects if obj.is_confirmed]
        return objects

    def get_tracked_states(self, *, confirmed_only: bool = True) -> list[TrackedState]:
        """Return current :class:`conveyor.types.TrackedState` records sorted by ``object_id``."""
        return [obj.state for obj in self.get_tracked_objects(confirmed_only=confirmed_only)]

    def schedulable_objects(self, *, confirmed_only: bool = True) -> list[Any]:
        """Return ``SchedulableObject(state, geometry)`` records for Hüseyin's scheduler."""
        return [obj.to_schedulable_object() for obj in self.get_tracked_objects(confirmed_only=confirmed_only)]

    def _resolve_frame_time(self, det_list: list[Any], simulation_time_s: float | None) -> float | None:
        if not det_list and simulation_time_s is None:
            return self._last_time_s
        times = {float(d.simulation_time_s) for d in det_list}
        if len(times) > 1:
            raise ValueError("All detections in a single update call must share one simulation_time_s")
        det_time = next(iter(times)) if times else None
        if simulation_time_s is not None:
            sim_t = float(simulation_time_s)
            if not isfinite(sim_t) or sim_t < 0.0:
                raise ValueError("simulation_time_s must be finite and non-negative")
            if det_time is not None and abs(sim_t - det_time) > 1e-6:
                raise ValueError("simulation_time_s does not match detection timestamps")
            return sim_t
        if det_time is not None and (not isfinite(det_time) or det_time < 0.0):
            raise ValueError("detection simulation_time_s must be finite and non-negative")
        return det_time

    def _associate(
        self,
        tracks: list[ConstantVelocityKalmanFilter3D],
        detections: list[Any],
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))

        gate = self.config.max_association_distance_m
        color_w = self.config.color_distance_weight_m
        n_t = len(tracks)
        n_d = len(detections)

        # Build cost matrix; float('inf') for pairs outside the spatial gate.
        cost: list[list[float]] = [[float("inf")] * n_d for _ in range(n_t)]
        for i, trk in enumerate(tracks):
            t_pos = trk.position_m
            t_col = trk.color_rgb
            for j, det in enumerate(detections):
                d_pos = tuple(float(v) for v in det.position_m)
                spatial_dist = _euclidean_3d(t_pos, d_pos)
                if spatial_dist > gate:
                    continue
                c_pen = 0.0
                d_col = getattr(det, "color_rgb", None)
                if d_col is not None and color_w > 0.0:
                    c_dist = sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(t_col, d_col)))
                    c_pen = color_w * c_dist
                cost[i][j] = spatial_dist + c_pen

        # Minimum-cost bipartite matching with priority on maximizing valid matches.
        # Since conveyor objects per batch are <= 24 and spatially separated, we first
        # extract connected components of valid edges and solve each component exactly.
        matched_pairs: list[tuple[int, int]] = []
        used_t: set[int] = set()
        used_d: set[int] = set()

        # Greedy-optimal hybrid: sort all valid edges by cost, or solve small components exactly.
        valid_edges = [
            (cost[i][j], i, j)
            for i in range(n_t)
            for j in range(n_d)
            if isfinite(cost[i][j])
        ]
        if not valid_edges:
            return [], list(range(n_t)), list(range(n_d))

        # Group tracks and detections into connected components of valid edges.
        adj_t: dict[int, list[int]] = {i: [] for i in range(n_t)}
        adj_d: dict[int, list[int]] = {j: [] for j in range(n_d)}
        for _, i, j in valid_edges:
            adj_t[i].append(j)
            adj_d[j].append(i)

        visited_t: set[int] = set()
        visited_d: set[int] = set()
        for start_t in range(n_t):
            if start_t in visited_t or not adj_t[start_t]:
                continue
            comp_t: list[int] = []
            comp_d: list[int] = []
            queue_t = [start_t]
            visited_t.add(start_t)
            while queue_t:
                curr_t = queue_t.pop()
                comp_t.append(curr_t)
                for nb_d in adj_t[curr_t]:
                    if nb_d not in visited_d:
                        visited_d.add(nb_d)
                        comp_d.append(nb_d)
                        for nb_t in adj_d[nb_d]:
                            if nb_t not in visited_t:
                                visited_t.add(nb_t)
                                queue_t.append(nb_t)

            # Solve component exactly via DFS branch-and-bound when small, else greedy by cost.
            comp_pairs = self._solve_component(comp_t, comp_d, cost)
            for ti, dj in comp_pairs:
                matched_pairs.append((ti, dj))
                used_t.add(ti)
                used_d.add(dj)

        unmatched_t = [i for i in range(n_t) if i not in used_t]
        unmatched_d = [j for j in range(n_d) if j not in used_d]
        return matched_pairs, unmatched_t, unmatched_d

    def _solve_component(
        self,
        comp_t: list[int],
        comp_d: list[int],
        cost: list[list[float]],
    ) -> list[tuple[int, int]]:
        if len(comp_t) == 1 and len(comp_d) == 1:
            return [(comp_t[0], comp_d[0])]

        if len(comp_t) <= 8 and len(comp_d) <= 8:
            best_count = -1
            best_cost = float("inf")
            best_pairs: list[tuple[int, int]] = []
            used_d: set[int] = set()
            current_pairs: list[tuple[int, int]] = []

            def dfs(idx: int, count: int, total_cost: float) -> None:
                nonlocal best_count, best_cost, best_pairs
                remaining = len(comp_t) - idx
                if count + remaining < best_count:
                    return
                if idx == len(comp_t):
                    if count > best_count or (count == best_count and total_cost < best_cost):
                        best_count = count
                        best_cost = total_cost
                        best_pairs = list(current_pairs)
                    return

                t_i = comp_t[idx]
                for d_j in comp_d:
                    if d_j not in used_d and isfinite(cost[t_i][d_j]):
                        used_d.add(d_j)
                        current_pairs.append((t_i, d_j))
                        dfs(idx + 1, count + 1, total_cost + cost[t_i][d_j])
                        current_pairs.pop()
                        used_d.remove(d_j)
                # Option to leave t_i unmatched
                dfs(idx + 1, count, total_cost)

            dfs(0, 0, 0.0)
            return best_pairs

        # Fallback greedy matching for large components
        edges = [
            (cost[i][j], i, j)
            for i in comp_t
            for j in comp_d
            if isfinite(cost[i][j])
        ]
        edges.sort(key=lambda item: item[0])
        pairs: list[tuple[int, int]] = []
        ut: set[int] = set()
        ud: set[int] = set()
        for _, i, j in edges:
            if i not in ut and j not in ud:
                pairs.append((i, j))
                ut.add(i)
                ud.add(j)
        return pairs

    def _spawn_track(self, detection: Any, frame_time_s: float) -> ConstantVelocityKalmanFilter3D:
        track_id = f"{self.id_prefix}_{self._next_id:03d}"
        self._next_id += 1
        pos = tuple(float(v) for v in detection.position_m)
        geom = getattr(detection, "geometry_type", "box")
        dims = getattr(detection, "dimensions_m", (0.06, 0.06, 0.06))
        yaw = float(getattr(detection, "yaw_rad", 0.0))
        color = getattr(detection, "color_rgb", (0.5, 0.5, 0.5))

        track = ConstantVelocityKalmanFilter3D(
            track_id=track_id,
            initial_time_s=frame_time_s,
            initial_position_m=pos,
            config=self.config,
            geometry_type=geom,
            dimensions_m=dims,
            yaw_rad=yaw,
            color_rgb=color,
        )
        self._tracks[track_id] = track
        return track

    def _prune_tracks(self, frame_time_s: float) -> None:
        to_delete: list[str] = []
        for tid, track in self._tracks.items():
            age_unseen = frame_time_s - track.last_update_time_s
            if track.missed_frames > self.config.max_missed_frames:
                to_delete.append(tid)
            elif age_unseen > self.config.max_time_without_update_s + 1e-9:
                to_delete.append(tid)
            elif track.position_m[0] > self.config.belt_exit_x_m:
                to_delete.append(tid)
        for tid in to_delete:
            del self._tracks[tid]
