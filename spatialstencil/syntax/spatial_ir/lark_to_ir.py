from dataclasses import dataclass
import lark

from spatialstencil.syntax.spatial_ir import irnodes


class TreeToSpatialIR(lark.Transformer):
    # Low-level literal syntax
    digit = lambda self, val: int(val[0])
    digits = lambda self, val: int(val[0])
    hex_digit = lambda self, val: str(val[0])
    hex_digits = lambda self, val: str(val[0])
    letter = lambda self, val: str(val[0])
    letters = lambda self, val: str(val[0])
    underscore = lambda self, val: str(val[0])
    true = lambda self, _: True
    false = lambda self, _: False

    # Literals
    @lark.v_args(inline=True)
    def decimal_literal(self, *digits):
        return int(''.join(str(d) for d in digits))

    @lark.v_args(inline=True)
    def hexadecimal_literal(self, *digits):
        return '0x' + ''.join(digits)

    negated_integer_literal = lambda self, value: -value[0]
    float_literal = lambda self, value: float(value[0])

    @lark.v_args(inline=True)
    def string_literal(self, s):
        return irnodes.StringLiteral(s[1:-1].replace('\\"', '"'))

    @lark.v_args(inline=True)
    def bare_id(self, *elements):
        return ''.join(str(s) for s in elements)

    @lark.v_args(inline=True)
    def suffix_id(self, *suffix):
        return ''.join(str(s) for s in suffix)

    # List types
    statement_body = list
    computation_body = list
    program_body = list

    # TODO: Implement


# Helper functions


def _expr(val: irnodes.SpatialNode | int | float | str) -> irnodes.Expression:
    if isinstance(val, irnodes.Expression):
        return val
    return irnodes.Expression(val)
