import copy

import pytest

from spada.syntax.common.symbolic import (
    ConstraintLog,
    Coord,
    SymbolicCoordError,
    evaluate,
)
from spada.syntax.spatial_ir.symbolic_grid import KeyedList


def test_arithmetic():
    i = Coord("I")
    assert repr(i + 3) == "I+3"
    assert repr(i - 1) == "I-1"
    assert repr(2 + i) == "I+2"
    assert (i + 3) - (i - 1) == 4
    assert evaluate(i + 2, {"I": 10}) == 12
    assert evaluate(5, {"I": 10}) == 5


@pytest.mark.parametrize(
    "op",
    [
        lambda a: a + Coord("I"),
        lambda a: a - Coord("J"),
        lambda a: 3 - a,
        lambda a: a * 2,
        lambda a: a // 2,
        lambda a: a % 2,
        lambda a: -a,
        lambda a: bool(a),
        lambda a: int(a),
        lambda a: range(a),
        lambda a: hash(a),
        lambda a: {a},
        lambda a: {a: 1},
    ],
)
def test_unsupported_operations_raise(op):
    with ConstraintLog(), pytest.raises(TypeError):
        op(Coord("I"))


def test_mixed_comparisons_log_constraints():
    i = Coord("I")
    with ConstraintLog() as log:
        assert 3 < i + 1
        assert log.bounds == {"I": 3}
        assert not (i == 7)
        assert log.bounds == {"I": 8}
        assert i - 2 != 0
        assert max(1, i) is i
        assert min(i + 5, 2) == 2
    assert log.bounds == {"I": 8}
    assert log.satisfied_by({"I": 8}) and not log.satisfied_by({"I": 7})


@pytest.mark.parametrize("n", range(1, 12))
def test_comparisons_agree_with_evaluation(n):
    """Lemma 1: under the logged constraints, symbolic comparisons agree with integer ones."""
    values = [0, 1, 4, 7, Coord("I"), Coord("I", 1), Coord("I", -2)]
    ops = [
        lambda a, b: a < b,
        lambda a, b: a <= b,
        lambda a, b: a > b,
        lambda a, b: a >= b,
        lambda a, b: a == b,
        lambda a, b: a != b,
    ]
    with ConstraintLog() as log:
        answers = [(a, b, op, op(a, b)) for a in values for b in values for op in ops]
    if not log.satisfied_by({"I": n}):
        return
    for a, b, op, answer in answers:
        assert op(evaluate(a, {"I": n}), evaluate(b, {"I": n})) == answer


def test_same_anchor_comparisons_log_nothing():
    with ConstraintLog() as log:
        assert Coord("I") < Coord("I", 1)
        assert Coord("I", 2) == Coord("I", 2)
    assert log.bounds == {}


def test_different_anchors_raise():
    with ConstraintLog(), pytest.raises(SymbolicCoordError):
        Coord("I") < Coord("J")


def test_comparison_requires_log():
    with pytest.raises(SymbolicCoordError):
        Coord("I") < 3


def test_non_coordinates():
    """Comparisons against non-coordinates behave as for int and log nothing."""
    with ConstraintLog() as log:
        assert Coord("I") != "?"
        assert not (Coord("I") == None)  # noqa: E711
        assert Coord("I") not in ["?", None]
        with pytest.raises(TypeError):
            Coord("I") < "?"
    assert log.bounds == {}


def test_immutable_and_copyable():
    i = Coord("I", 2)
    with pytest.raises(AttributeError):
        i.offset = 3
    assert copy.deepcopy(i) is i


def test_keyed_list_matches_dict_for_integers():
    keys = [(0, 1, 0, 1), (1, 4, 0, 1), (0, 1, 0, 1), (2, 3, 2, 3)]
    keyed, plain = KeyedList(list), {}
    for value, key in enumerate(keys):
        keyed[key].append(value)
        plain.setdefault(key, []).append(value)
    assert keyed.items() == list(plain.items())


def test_keyed_list_with_symbolic_keys():
    i = Coord("I")
    with ConstraintLog() as log:
        keyed = KeyedList(list)
        keyed[(0, 1)].append("a")
        keyed[(1, i)].append("b")
        keyed[(1, i)].append("c")
        keyed[(0, 1)].append("d")
        assert (2, i + 1) not in keyed
        assert log.bounds == {}
        assert (1, 5) not in keyed
    assert keyed.values() == [["a", "d"], ["b", "c"]]
    assert log.bounds == {"I": 6}
