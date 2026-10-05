"""
Helpers that read subgrid geometry as integers or anchored symbolic coordinates.

On concrete kernels these return the same integers as ``eval()``; on kernels whose subgrid bounds are
parameter expressions of the form ``N + d`` they return :class:`~spada.syntax.common.symbolic.Coord`
values, so that geometric decisions can be made for all sufficiently large parameter values.
"""

from __future__ import annotations

import spada.syntax.spatial_ir.irnodes as spa
from spada.syntax.common.symbolic import Coord, SymbolicCoordError

DEFAULT_PARAMETERS = ("I", "J", "K")


def coord_of_expr(
    expr: spa.Expression | spa.SpatialNode | int,
    parameters: tuple[str, ...] = DEFAULT_PARAMETERS,
) -> int | Coord:
    """
    Reads an integer or anchored coordinate from an expression.

    :param expr: An expression built from integer literals, parameter identifiers, ``+`` and ``-``.
    :param parameters: The names of the kernel parameters.
    :return: The integer or :class:`Coord` value.
    :raises SymbolicCoordError: If the expression is not of that form.
    """
    if isinstance(expr, int):
        return expr
    if isinstance(expr, spa.Expression):
        return coord_of_expr(expr.value, parameters)
    if isinstance(expr, spa.ConstantLiteral) and isinstance(expr.value, int):
        return expr.value
    if (
        isinstance(expr, spa.Identifier)
        and expr.version == 0
        and expr.name in parameters
    ):
        return Coord(expr.name)
    if isinstance(expr, spa.UnaryOperator) and expr.op == "-":
        value = coord_of_expr(expr.value, parameters)
        if isinstance(value, int):
            return -value
    if isinstance(expr, spa.BinaryOperator) and expr.op in ("+", "-"):
        left = coord_of_expr(expr.left, parameters)
        right = coord_of_expr(expr.right, parameters)
        return left + right if expr.op == "+" else left - right
    raise SymbolicCoordError(
        f"Expression is not an anchored coordinate: {expr.as_ir()}"
    )


def range_bounds(rng: spa.RangeExpression) -> tuple[int | Coord, int | Coord, int]:
    """
    :param rng: A range expression.
    :return: ``(start, stop, step)`` of the range, where a missing stop denotes a single point and a
             missing step is 1.
    """
    start = coord_of_expr(rng.start)
    stop = start + 1 if rng.stop is None else coord_of_expr(rng.stop)
    step = 1 if rng.step is None else coord_of_expr(rng.step)
    if not isinstance(step, int):
        raise SymbolicCoordError(f"Range step must be an integer: {rng.as_ir()}")
    return start, stop, step


def subgrid_bounds(subgrid: spa.SubgridExpression) -> tuple[tuple, tuple]:
    """
    :param subgrid: A subgrid expression.
    :return: The x and y ranges as ``(start, stop, step)`` tuples.
    """
    return range_bounds(subgrid.x_range), range_bounds(subgrid.y_range)


def grid_key(block) -> tuple:
    """
    A key identifying the PE rectangle of a block, comparable with ``==``.

    For concrete subgrids this is ``block.get_grid_rect()``. For symbolic subgrids it is the tuple of
    (unnormalized) bounds; symbolic subgrids must have stride 1 here, since canonicalizing a stop
    value to the stride boundary is not an anchored operation.

    :param block: A place, dataflow, or compute block.
    :return: ``(x_start, x_stop, y_start, y_stop)``.
    """
    try:
        return block.get_grid_rect()
    except (TypeError, AttributeError):
        pass
    (x0, x1, sx), (y0, y1, sy) = subgrid_bounds(block.subgrid)
    if sx != 1 or sy != 1:
        raise SymbolicCoordError("Symbolic grid keys require stride 1")
    return (x0, x1, y0, y1)


class KeyedList:
    """
    An insertion-ordered mapping for keys that may contain symbolic coordinates.

    Symbolic coordinates are unhashable, so that every equality decision between them is made by (and
    recorded through) ``Coord.__eq__``. Lookups therefore compare against every unhashable key with
    ``==``; hashable keys (all keys of a concrete kernel) are found through a dict, so concrete kernels
    behave and perform exactly like a ``dict``.
    """

    def __init__(self, default_factory=None):
        self._keys: list = []
        self._values: list = []
        self._hashed: dict = {}
        self._unhashable: list[int] = []
        self._default_factory = default_factory

    @staticmethod
    def _hashable(key) -> bool:
        try:
            hash(key)
            return True
        except TypeError:
            return False

    def _index(self, key) -> int | None:
        if self._hashable(key):
            for i in self._unhashable:
                if self._keys[i] == key:
                    return i
            return self._hashed.get(key)
        for i, k in enumerate(self._keys):
            if k == key:
                return i
        return None

    def _append(self, key, value) -> None:
        if self._hashable(key):
            self._hashed[key] = len(self._keys)
        else:
            self._unhashable.append(len(self._keys))
        self._keys.append(key)
        self._values.append(value)

    def __contains__(self, key) -> bool:
        return self._index(key) is not None

    def __getitem__(self, key):
        i = self._index(key)
        if i is None:
            if self._default_factory is None:
                raise KeyError(key)
            self._append(key, self._default_factory())
            return self._values[-1]
        return self._values[i]

    def __setitem__(self, key, value) -> None:
        i = self._index(key)
        if i is None:
            self._append(key, value)
        else:
            self._values[i] = value

    def values(self) -> list:
        return list(self._values)

    def items(self) -> list:
        return list(zip(self._keys, self._values))
