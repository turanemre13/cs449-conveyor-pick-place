"""Compare conservative collision bounds with exact RAI distances."""

from math import cos, sin

import pytest

from conveyor.planning.clearance import PairDistances


def test_bounds_keep_near_pairs_and_fall_back_for_other_shapes():
    ry = pytest.importorskip("robotic")
    config = ry.Config()
    config.addFrame("belt").setShape(ry.ST.ssBox, [0.4, 0.4, 0.04, 0.006])
    capsule = config.addFrame("link").setShape(ry.ST.capsule, [0.2, 0.026])
    capsule.setQuaternion([cos(0.45), 0, sin(0.45), 0])
    cylinder = config.addFrame("load").setShape(ry.ST.cylinder, [0.1, 0.038])
    cylinder.setQuaternion([cos(0.27), sin(0.27), 0, 0])
    config.addFrame("other").setShape(ry.ST.sphere, [0.03]).setPosition([0, 0, 2])

    pairs = (("belt", "link"), ("belt", "load"), ("link", "load"), ("belt", "other"))
    limits = (0.005, -0.01, 0.03, 0.005)
    distances = PairDistances(config, pairs)
    for height in (0.0, 0.08, 0.11, 0.24, 0.8):
        capsule.setPosition([0, 0, height])
        cylinder.setPosition([0.06, 0.02, 0.08 + height / 2])
        exact = [-float(config.eval(ry.FS.negDistance, list(pair))[0][0]) for pair in pairs]
        checked = dict(distances.candidate_distances(limits))
        assert {i for i, (gap, limit) in enumerate(zip(exact, limits)) if gap <= limit} <= checked.keys()
        assert checked == pytest.approx({i: exact[i] for i in checked})
        assert 3 in checked

    assert set(checked) == {3}
