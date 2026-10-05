"""
Anchored symbolic coordinates for parametric (domain-size independent) lowering.

A coordinate in the stencil lowering is either a plain ``int`` (anchored at the low edge of the
domain) or a :class:`Coord` ``N + d`` (anchored at the high edge of axis ``N``, e.g. ``I + 1``).
:class:`Coord` supports exactly the operations whose results agree with integer evaluation for every
sufficiently large parameter value: adding integers, subtracting a coordinate of the same anchor or a
low-anchored one, and comparisons. Whenever a comparison mixes a low- and a high-anchored value, the
answer for large domains is returned and the corresponding lower bound on the parameter is recorded in
the active :class:`ConstraintLog`. Every other use (truth tests, hashing, ``%``, ``//``, ``int()``,
...) raises :class:`SymbolicCoordError`, so a lowering that completes on symbolic input only made
decisions that are valid for all sizes satisfying the recorded constraints.
"""

from __future__ import annotations

from typing import Optional


class SymbolicCoordError(TypeError):
    """
    Raised when a symbolic coordinate is used in an operation whose result depends on the
    concrete parameter value in a way that cannot be decided for all large domains.
    """


class ConstraintLog:
    """
    Collects lower bounds on parameters that are implied by symbolic comparisons.

    Used as a context manager; the innermost active log receives the constraints::

        with ConstraintLog() as log:
            lower(...)
        log.bounds  # e.g. {'I': 3, 'J': 3, 'K': 2}
    """

    _active: list["ConstraintLog"] = []

    def __init__(self):
        self.bounds: dict[str, int] = {}

    def __enter__(self) -> "ConstraintLog":
        ConstraintLog._active.append(self)
        return self

    def __exit__(self, *exc) -> None:
        popped = ConstraintLog._active.pop()
        assert popped is self

    def require(self, anchor: str, minimum: int) -> None:
        """
        Records the constraint ``anchor >= minimum``.

        :param anchor: The parameter name.
        :param minimum: The smallest admissible value.
        """
        self.bounds[anchor] = max(self.bounds.get(anchor, minimum), minimum)

    def satisfied_by(self, values: dict[str, int]) -> bool:
        """
        :param values: Concrete parameter values.
        :return: Whether all recorded constraints hold for ``values``.
        """
        return all(values[anchor] >= minimum for anchor, minimum in self.bounds.items())

    @staticmethod
    def current() -> "ConstraintLog":
        """
        :return: The innermost active log.
        :raises SymbolicCoordError: If no log is active.
        """
        if not ConstraintLog._active:
            raise SymbolicCoordError(
                "Comparing symbolic coordinates requires an active ConstraintLog"
            )
        return ConstraintLog._active[-1]


class Coord:
    """
    A high-anchored coordinate ``anchor + offset``, e.g. ``I + 1``.

    Low-anchored coordinates are always represented by plain ``int`` values, so a :class:`Coord`
    never denotes a constant.
    """

    __slots__ = ("anchor", "offset")

    def __init__(self, anchor: str, offset: int = 0):
        assert isinstance(anchor, str)
        assert isinstance(offset, int) and not isinstance(offset, bool)
        object.__setattr__(self, "anchor", anchor)
        object.__setattr__(self, "offset", offset)

    def __setattr__(self, key, value):
        raise AttributeError("Coord is immutable")

    def __copy__(self) -> "Coord":
        return self

    def __deepcopy__(self, memo) -> "Coord":
        return self

    def __reduce__(self):
        return (Coord, (self.anchor, self.offset))

    # Evaluation and printing
    def evaluate(self, values: dict[str, int]) -> int:
        """
        :param values: Concrete parameter values.
        :return: The integer value of this coordinate.
        """
        return values[self.anchor] + self.offset

    def __repr__(self) -> str:
        if self.offset == 0:
            return self.anchor
        if self.offset > 0:
            return f"{self.anchor}+{self.offset}"
        return f"{self.anchor}-{-self.offset}"

    __str__ = __repr__

    # Arithmetic
    def __add__(self, other):
        if _is_int(other):
            return Coord(self.anchor, self.offset + other)
        if isinstance(other, Coord):
            raise SymbolicCoordError(
                f"Cannot add symbolic coordinates {self} and {other}"
            )
        return NotImplemented

    __radd__ = __add__

    def __sub__(self, other):
        if _is_int(other):
            return Coord(self.anchor, self.offset - other)
        if isinstance(other, Coord):
            if other.anchor != self.anchor:
                raise SymbolicCoordError(
                    f"Cannot subtract coordinates of different axes: {self} - {other}"
                )
            return self.offset - other.offset
        return NotImplemented

    def __rsub__(self, other):
        if _is_int(other):
            raise SymbolicCoordError(f"{other} - {self} is not an anchored coordinate")
        return NotImplemented

    def _unsupported(self, *args):
        raise SymbolicCoordError(
            f"Operation is not supported on symbolic coordinate {self}"
        )

    __mul__ = __rmul__ = __truediv__ = __rtruediv__ = _unsupported
    __floordiv__ = __rfloordiv__ = __mod__ = __rmod__ = __divmod__ = __rdivmod__ = (
        _unsupported
    )
    __neg__ = __pos__ = __abs__ = __lshift__ = __rshift__ = __pow__ = _unsupported

    def __bool__(self):
        raise SymbolicCoordError(
            f"Truth value of symbolic coordinate {self} is undefined"
        )

    def __int__(self):
        raise SymbolicCoordError(f"Symbolic coordinate {self} has no integer value")

    __index__ = __float__ = __int__

    # Comparisons
    def _sign(self, other) -> Optional[int]:
        """
        :return: The sign of ``self - other`` for all admissible parameter values, or None if
                 ``other`` is not a coordinate.
        """
        if _is_int(other):
            # anchor + offset > other  <=>  anchor >= other - offset + 1
            ConstraintLog.current().require(self.anchor, other - self.offset + 1)
            return 1
        if isinstance(other, Coord):
            if other.anchor != self.anchor:
                raise SymbolicCoordError(
                    f"Cannot compare coordinates of different axes: {self}, {other}"
                )
            return (self.offset > other.offset) - (self.offset < other.offset)
        return None

    def __eq__(self, other):
        sign = self._sign(other)
        return NotImplemented if sign is None else sign == 0

    def __ne__(self, other):
        sign = self._sign(other)
        return NotImplemented if sign is None else sign != 0

    def __lt__(self, other):
        sign = self._sign(other)
        return NotImplemented if sign is None else sign < 0

    def __le__(self, other):
        sign = self._sign(other)
        return NotImplemented if sign is None else sign <= 0

    def __gt__(self, other):
        sign = self._sign(other)
        return NotImplemented if sign is None else sign > 0

    def __ge__(self, other):
        sign = self._sign(other)
        return NotImplemented if sign is None else sign >= 0

    __hash__ = None


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def is_symbolic(value) -> bool:
    """
    :return: Whether ``value`` is a symbolic (high-anchored) coordinate.
    """
    return isinstance(value, Coord)


def evaluate(value, values: dict[str, int]):
    """
    Evaluates an ``int`` or :class:`Coord` for concrete parameter values.

    :param value: The coordinate.
    :param values: Concrete parameter values.
    :return: The integer value.
    """
    return value.evaluate(values) if isinstance(value, Coord) else value
