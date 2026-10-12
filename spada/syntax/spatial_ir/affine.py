"""
Affine decomposition of Spatial IR index expressions.
"""
from typing import Optional
from spada.syntax.spatial_ir import irnodes as spir

#: ``(coefficients, constant)``: the expression equals ``sum(coefficients[v] * v) + constant``.
AffineForm = tuple[dict[spir.Identifier, int], int]


def affine_form(expr: spir.SpatialNode) -> Optional[AffineForm]:
    """
    Decomposes an index expression into an integer-affine form over its identifiers.

    Every identifier is treated as a variable; callers decide which ones are loop (axis) variables
    and which are free. Only ``+``, ``-``, unary ``-`` and multiplication by a constant are
    affine. Anything else (division, modulo, products of variables, float constants, calls)
    returns None.

    :param expr: The expression to decompose.
    :return: The affine form, or None if the expression is not integer-affine.
    """
    e = expr.value if isinstance(expr, spir.Expression) else expr
    if isinstance(e, spir.Identifier):
        return {e: 1}, 0
    if isinstance(e, (spir.ConstantLiteral, spir.Parameter)):
        try:
            v = e.eval()
        except (ValueError, TypeError, AttributeError):
            return None
        return ({}, v) if isinstance(v, int) and not isinstance(v, bool) else None
    if isinstance(e, spir.UnaryOperator) and e.op == '-':
        sub = affine_form(e.value)
        return None if sub is None else ({v: -c for v, c in sub[0].items()}, -sub[1])
    if isinstance(e, spir.BinaryOperator) and e.op in ('+', '-', '*'):
        left, right = affine_form(e.left), affine_form(e.right)
        if left is None or right is None:
            return None
        if e.op in ('+', '-'):
            sign = 1 if e.op == '+' else -1
            coeffs = dict(left[0])
            for v, c in right[0].items():
                coeffs[v] = coeffs.get(v, 0) + sign * c
            return {v: c for v, c in coeffs.items() if c != 0}, left[1] + sign * right[1]
        if not left[0]:
            return {v: c * left[1] for v, c in right[0].items() if c * left[1] != 0}, right[1] * left[1]
        if not right[0]:
            return {v: c * right[1] for v, c in left[0].items() if c * right[1] != 0}, left[1] * right[1]
    return None


def const_range(rng: spir.RangeExpression) -> Optional[tuple[int, int, int]]:
    """
    Evaluates a range expression whose bounds are all integer constants.

    :param rng: The range expression.
    :return: ``(start, stop, step)``, or None if any bound is not a compile-time integer.
    """
    if rng.stop is None:
        return None
    try:
        start = rng.start.eval()
        stop = rng.stop.eval()
        step = rng.step.eval() if rng.step is not None else 1
    except (ValueError, TypeError, AttributeError):
        return None
    if not all(isinstance(v, int) for v in (start, stop, step)) or step == 0:
        return None
    return start, stop, step
