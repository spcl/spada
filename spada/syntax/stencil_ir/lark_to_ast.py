from dataclasses import dataclass
import lark

from spada.syntax.stencil_ir import irnodes


class TreeToAST(lark.Transformer):
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
    unknown_dim_literal = lambda self, _: '?'

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
    extent_offset_tuple = tuple
    extent_tuple = tuple
    extent_tuple_list = list
    dim_list = list
    id_list = list
    type_list = list
    subscript_slice = tuple
    multi_interval_type = list
    computation_interval = list
    attr = tuple
    call_arguments = list
    domain_list = list

    statement_body = list
    computation_body = list
    program_body = list

    def attributes(self, args, meta=None):
        result = {}
        for attr_name, attr_val in list(args):
            result[attr_name] = attr_val
        return result

    # Scalar types
    float_type = int_type = uint_type = bool_type = unknown_type = lambda self, args: getattr(
        irnodes.ScalarType, str(args[0]))
    schedule_type = lambda self, args: getattr(irnodes.ComputationType, str(args[0]))

    def schedule_keyword(self, args, meta=None):
        return "schedule"

    def interval_keyword(self, args, meta=None):
        return "interval"

    def dim_or_end(self, args, meta=None):
        dim = args[0]
        # Dimension can be explicit, end (None), or indeterminate ("?")
        if str(dim) == "None":
            return None
        return dim

    def value_expr(self, args, meta=None):
        # Contract/inline value expressions that only contain another value expression
        if len(args) == 1 and isinstance(args[0], lark.Tree) and args[0].data == 'value_expr':
            return args[0]
        return irnodes.Expression(*args)

    def unknown_domain_literal(self, args, meta=None):
        return irnodes.Interval()

    # Data types
    def domain_type(self, args, meta=None):
        intervals = []
        for interval in args[0]:
            if isinstance(interval, irnodes.Interval):
                intervals.append(interval)
            else:
                intervals.append(irnodes.Interval())
        return irnodes.Cartesian(*intervals)

    def extent_type(self, args, meta=None):
        return irnodes.Extent([_make_offset(a) for a in args[0]])

    view_type = irnodes.ViewType.from_lark
    interval_type = irnodes.Interval.from_lark
    field_type = irnodes.FieldType.from_lark
    any_type = lambda self, args: irnodes.AnyType()

    # Basic types
    def identifier(self, args, meta=None):
        if len(args) == 2 and args[1] == 0:
            raise SyntaxError('Explicit version 0 (%x#0) is not allowed, please use %x')
        return irnodes.Identifier(*args)

    type_info = lambda self, args: irnodes.OperationType(*([arg] for arg in args))
    type_list_info = lambda self, args: irnodes.OperationType(*args)
    return_type_info = lambda self, args: irnodes.OperationType(*args)

    def subscript(self, args, meta=None):
        return irnodes.Subscript(args[0], list(args[1]))

    # Operators
    def unary_op(self, args, meta=None):
        return irnodes.UnaryOperator(str(args[0]), _expr(args[1]))

    def not_test(self, args, meta=None):
        return irnodes.UnaryOperator('not', _expr(args[0]))

    def binary_op(self, args, meta=None):
        return irnodes.BinaryOperator(_expr(args[0]), str(args[1]), _expr(args[2]))

    def binary_op_logical_or(self, args, meta=None):
        return irnodes.BinaryOperator(_expr(args[0]), 'or', _expr(args[1]))

    def binary_op_or(self, args, meta=None):
        return irnodes.BinaryOperator(_expr(args[0]), '|', _expr(args[1]))

    def binary_op_logical_and(self, args, meta=None):
        return irnodes.BinaryOperator(_expr(args[0]), 'and', _expr(args[1]))

    def binary_op_and(self, args, meta=None):
        return irnodes.BinaryOperator(_expr(args[0]), '&', _expr(args[1]))

    def binary_op_xor(self, args, meta=None):
        return irnodes.BinaryOperator(_expr(args[0]), '^', _expr(args[1]))

    def binary_op_pow(self, args, meta=None):
        return irnodes.BinaryOperator(_expr(args[0]), '**', _expr(args[1]))

    def comparison(self, args, meta=None):
        return irnodes.BinaryOperator(_expr(args[0]), str(args[1]), _expr(args[2]))

    def ternary_op(self, args, meta=None):
        return irnodes.TernaryOperator(_expr(args[0]), _expr(args[1]), _expr(args[2]))

    def call(self, args, meta=None):
        return irnodes.MathCall(args[0], [_expr(arg) for arg in args[1]])

    # Operations
    def return_expr(self, args, meta=None):

        # Find the operation type arguments
        optype = next((i for i in range(len(args)) if isinstance(args[i], irnodes.OperationType)), None)
        if optype:
            return irnodes.ReturnOp(args[:optype], args[optype])
        else:
            optype = irnodes.OperationType([irnodes.AnyType() for _ in range(len(args))],
                                           None)
            return irnodes.ReturnOp(args, optype)
    materialize_op = irnodes.MaterializeOp.from_lark

    def assign_expr(self, args, meta=None):
        if len(args) == 3:  # With type info
            return irnodes.AssignOp(args[0][0], args[1], args[2])
        return irnodes.AssignOp(args[0][0], args[1])

    # Blocks
    def if_op(self, args, meta=None):
        results = args[0]
        test = args[1]
        offset = 2
        if isinstance(args[offset], irnodes.OperationType):
            operation_type = args[offset]
            offset += 1
        else:
            operation_type = irnodes.OperationType([irnodes.ViewType.empty()], [irnodes.ViewType.empty()])

        body = args[offset]
        offset += 1

        else_ifs = []
        for arg in args[offset:]:
            if isinstance(arg, lark.Tree) and arg.data == 'elif_block':
                elif_test, elif_body = arg.children
                else_ifs.append(irnodes.ElseIfBlock(elif_test, elif_body))
            elif isinstance(arg, lark.Tree) and arg.data == 'else_block':
                else_ifs.append(irnodes.ElseIfBlock(None, arg.children[0]))

        return irnodes.IfBlock(results, test, operation_type, body, else_ifs)

    def statement(self, args, meta=None):
        if len(args) == 5:
            results, inputs, attributes, operation_type, body = args
        elif len(args) == 4:
            results, inputs, operation_type, body = args
            attributes = {}
        else:
            raise ValueError(f'Unexpected number of arguments to spst.statement: {len(args)}')
        return irnodes.StatementBlock(results, inputs, attributes, operation_type, body)

    def computation(self, args, meta=None):
        results, inputs, attributes, operation_type, body = args
        schedule = interval = None
        for k, v in attributes.items():
            if k == 'schedule':
                schedule = v
            elif k == 'interval':
                interval = v
            else:
                raise NameError(f'Unexpected spst.computation attribute "{k}"')
        if schedule is None:
            raise ValueError('spst.computation is missing the "schedule" attribute')
        if interval is None:
            raise ValueError('spst.computation is missing the "interval" attribute')

        if len(interval) != 3:
            raise ValueError(f'Expected 3-tuple for interval, got {interval}')

        return irnodes.ComputationBlock(results, inputs, schedule, interval, operation_type, body)

    def program(self, args, meta=None):
        outputs = args[0]
        offset = 1
        if isinstance(args[1], str):  # Named program
            name = args[1]
            offset = 2
        else:
            name = None
        inputs, attributes, operation_type, computations = args[offset:]
        return irnodes.Program(outputs, name, inputs, attributes, operation_type, computations)


# Helper functions
def _make_dimtuple(tup):
    return tuple(tup)


def _make_offset(extent_tuple):
    if len(extent_tuple) != 1 and len(extent_tuple[0]) != 3:
        raise ValueError(f"Expected 3-tuple nested in a length 1 sequence, got {extent_tuple}")
    assert isinstance(extent_tuple, tuple)
    return irnodes.Offset(extent_tuple[0])


def _expr(val: irnodes.Node | int | float | str) -> irnodes.Expression:
    if isinstance(val, irnodes.Expression):
        return val
    return irnodes.Expression(val)
