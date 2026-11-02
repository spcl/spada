import random

import pytest

from spada.lowering.regions import partition
from spada.syntax.common.symbolic import ConstraintLog, Coord, evaluate
from spada.syntax.spatial_ir.grid_geometry import Rectangle


def _random_rects(rng: random.Random, count: int, size: int) -> list[Rectangle]:
    rects = []
    for k in range(count):
        x0, x1 = sorted(rng.sample(range(size + 1), 2))
        y0, y1 = sorted(rng.sample(range(size + 1), 2))
        rects.append(Rectangle((x0, x1, 1), (y0, y1, 1), k))
    return rects


def _covering(rects, x, y) -> set:
    return {
        r.metadata
        for r in rects
        if r.x_range[0] <= x < r.x_range[1] and r.y_range[0] <= y < r.y_range[1]
    }


@pytest.mark.parametrize("seed", range(30))
def test_partition_per_pe_signature(seed):
    """Lemma 4: every PE lies in exactly one region, whose group holds exactly its covering rectangles."""
    rng = random.Random(seed)
    rects = _random_rects(rng, rng.randint(1, 6), 9)
    groups = partition(rects)
    for x in range(10):
        for y in range(10):
            containing = [
                g
                for g in groups
                if g[0].x_range[0] <= x < g[0].x_range[1]
                and g[0].y_range[0] <= y < g[0].y_range[1]
            ]
            expected = _covering(rects, x, y)
            if not expected:
                assert not containing
                continue
            assert len(containing) == 1
            group = containing[0]
            assert [r.metadata for r in group] == sorted(expected)
            assert all(
                r.x_range == group[0].x_range and r.y_range == group[0].y_range
                for r in group
            )


def _symbolic_rects(rng: random.Random, count: int) -> list[Rectangle]:
    def coord(axis):
        c = rng.randint(-1, 3)
        return Coord(axis, c) if rng.random() < 0.5 else c

    rects = []
    for k in range(count):
        x = sorted(
            [coord("I"), coord("I")],
            key=lambda v: (isinstance(v, Coord), getattr(v, "offset", v)),
        )
        y = sorted(
            [coord("J"), coord("J")],
            key=lambda v: (isinstance(v, Coord), getattr(v, "offset", v)),
        )
        rects.append(Rectangle((x[0], x[1], 1), (y[0], y[1], 1), k))
    return rects


def _evaluate_groups(groups, values):
    return [
        [
            (
                evaluate(r.x_range[0], values),
                evaluate(r.x_range[1], values),
                evaluate(r.y_range[0], values),
                evaluate(r.y_range[1], values),
                r.metadata,
            )
            for r in group
        ]
        for group in groups
    ]


@pytest.mark.parametrize("seed", range(30))
def test_symbolic_partition_simulates_concrete(seed):
    """Lemma 2 for the partition: under the logged constraints, evaluating the symbolic result gives the integer result."""
    rng = random.Random(seed)
    with ConstraintLog() as log:
        rects = _symbolic_rects(rng, rng.randint(1, 5))
        groups = partition(rects)
    checked = 0
    for n_i in range(1, 12):
        for n_j in range(1, 12):
            values = {"I": n_i, "J": n_j}
            if not log.satisfied_by(values):
                continue
            concrete = [
                Rectangle(
                    (evaluate(r.x_range[0], values), evaluate(r.x_range[1], values), 1),
                    (evaluate(r.y_range[0], values), evaluate(r.y_range[1], values), 1),
                    r.metadata,
                )
                for r in rects
            ]
            assert _evaluate_groups(groups, values) == _evaluate_groups(
                partition(concrete), values
            )
            checked += 1
    assert checked > 0
