"""Exact signed distances for shape pairs near a clearance threshold."""


class PairDistances:
    """Precompute local bounds for a fixed collision topology."""

    def __init__(self, config, pairs):
        import numpy as np
        import robotic as ry

        self.config = config
        self.pairs = tuple(pairs)
        self.pair_keys = tuple(frozenset(pair) for pair in self.pairs)
        names = tuple(dict.fromkeys(name for pair in self.pairs for name in pair))
        indices = {name: index for index, name in enumerate(names)}
        self._first, self._second = np.array([(indices[a], indices[b]) for a, b in self.pairs]).T

        frame_ids, half_sizes, bounded = [], [], []
        for name in names:
            frame = config.getFrame(name)
            frame_ids.append(frame.ID)
            shape, size = frame.getShapeType(), frame.getSize()
            if shape in (ry.ST.box, ry.ST.ssBox) and len(size) >= 3:
                half_size = tuple(size[:3] / 2)
            elif shape in (ry.ST.cylinder, ry.ST.capsule) and len(size) == 2:
                length, radius = size
                end_cap = radius if shape == ry.ST.capsule else 0.0
                half_size = (radius, radius, length / 2 + end_cap)
            else:
                half_size = None
            has_bounds = half_size is not None and all(np.isfinite(value) and value >= 0 for value in half_size)
            bounded.append(has_bounds)
            half_sizes.append(half_size if has_bounds else (0.0, 0.0, 0.0))
        self._frame_ids = np.array(frame_ids)
        self._half_sizes = np.array(half_sizes)
        supported = np.array(bounded)
        self._bounded_pairs = supported[self._first] & supported[self._second]

    def candidate_distances(self, limits):
        """Yield exact (pair index, distance) for every pair that might be within its limit."""
        import numpy as np
        import robotic as ry

        poses = self.config.getFrameState()[self._frame_ids]
        centers = poses[:, :3]
        w, x, y, z = poses[:, 3:].T
        rotations = np.stack((
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ), axis=1).reshape(-1, 3, 3)
        extents = (np.abs(rotations) @ self._half_sizes[..., None]).squeeze(axis=2)

        # Any positive gap between the enclosing boxes is a lower bound on
        # distance. Overlapping boxes still require the exact signed distance.
        gaps = np.abs(centers[self._first] - centers[self._second]) - extents[self._first] - extents[self._second]
        lower_bounds = gaps.max(axis=1)
        needs_exact = (~self._bounded_pairs | ~np.isfinite(lower_bounds)
                       | (lower_bounds <= np.maximum(limits, 0.0) + 1e-9))
        for index in np.flatnonzero(needs_exact):
            first, second = self.pairs[index]
            yield index, -float(self.config.eval(ry.FS.negDistance, [first, second])[0][0])
