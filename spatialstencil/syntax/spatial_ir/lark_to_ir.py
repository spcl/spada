from dataclasses import dataclass
import lark

from spatialstencil.syntax.spatial_ir import irnodes


@dataclass
class ObjectType:
    typename: str


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
    def NEWLINE(self, args):
        return None

    # Literals
    @lark.v_args(inline=True)
    def decimal_literal(self, *digits):
        return int(''.join(str(d) for d in digits))

    @lark.v_args(inline=True)
    def hexadecimal_literal(self, *digits):
        return '0x' + ''.join(digits)

    negated_integer_literal = lambda self, value: -value[0]
    float_literal = lambda self, value: float(value[0])
    bool_literal = lambda self, value: bool(value[0])

    @lark.v_args(inline=True)
    def string_literal(self, s):
        return irnodes.StringLiteral(s[1:-1].replace('\\"', '"'))

    @lark.v_args(inline=True)
    def bare_id(self, *elements):
        return ''.join(str(s) for s in elements)

    def annotation(self, element):
        return str(element[0])

    @lark.v_args(inline=True)
    def suffix_id(self, *suffix):
        return ''.join(str(s) for s in suffix)

    # Basic types
    def identifier(self, args):
        if len(args) == 1:
            try:
                return irnodes.ConstantLiteral(int(args[0]), None)
            except ValueError:
                try:
                    return irnodes.ConstantLiteral(float(args[0]), None)
                except ValueError:
                    return irnodes.Identifier(args[0], 0)
        return irnodes.Identifier(*args)

    def typed_var(self, args):
        dtype, ident = args
        return irnodes.TypedIdentifier(dtype, ident.name, ident.version)

    float_type = int_type = uint_type = bool_type = lambda self, args: getattr(irnodes.ScalarType, str(args[0]))
    object_type = lambda self, args: ObjectType(str(args[0]))
    stream_type = irnodes.StreamType.from_lark

    def array_type(self, args):
        return irnodes.ArrayType(args[0], args[1:])

    def value_expr(self, args):
        # Contract/inline value expressions that only contain another value expression
        if len(args) == 1 and isinstance(args[0], lark.Tree) and args[0].data == 'value_expr':
            return args[0]
        return irnodes.Expression(*args)

    # Expressions
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

    def function_call(self, args, meta=None):
        func, arguments = args
        if func == 'send':
            return irnodes.SendStatement(*arguments)
        elif func == 'receive':
            return irnodes.ReceiveStatement(*arguments)
        raise SyntaxError(f'Unrecognized free function call to "{func}"')

    subscript = irnodes.ArraySlice.from_lark
    subscript_expr = irnodes.ArraySlice.from_lark

    # Grid/Subgrid expressions
    range_expression = irnodes.RangeExpression.from_lark

    def range_and_stream(self, args):
        rng = args[:-1]
        stream = args[-1]
        return rng, stream

    # Declarations and routing
    hop = irnodes.RoutingHop.from_lark
    routing = irnodes.RoutingDeclaration.from_lark
    field_declaration = irnodes.FieldDeclaration.from_lark
    stream_declaration = irnodes.RelativeStreamDeclaration.from_lark
    subgrid_expression_2d = irnodes.SubgridExpression.from_lark

    # Scopes
    for_stmt = irnodes.ForStatement.from_lark
    map_stmt = irnodes.MapStatement.from_lark
    async_stmt = irnodes.AsyncBlock.from_lark

    def foreach_stmt(self, args):
        iters, (rng, stream), body = args
        return irnodes.ForeachStatement(iters, stream, body, parameter_range=rng)

    # Await variants
    def await_stmt(self, args):
        return irnodes.AwaitStatement(irnodes.Completion(args[0]))

    # Definitions and assignments (combines syntax and semantics)
    def definition(self, args):
        dtype, identifier, rhs = args
        if isinstance(rhs, (irnodes.MapStatement, irnodes.ForeachStatement, irnodes.SendStatement,
                            irnodes.ReceiveStatement, irnodes.AsyncBlock)):
            if isinstance(dtype, ObjectType) and dtype.typename == 'completion':
                rhs.completion_name = irnodes.Completion(identifier)
                return rhs
            else:
                raise SyntaxError('Only completions can be assigned to from scopes')
        return irnodes.DefinitionStatement(dtype, identifier, rhs)

    def assignment(self, args):
        lhs, rhs = args
        if isinstance(rhs, (irnodes.MapStatement, irnodes.ForeachStatement, irnodes.SendStatement,
                            irnodes.ReceiveStatement, irnodes.AsyncBlock)):
            raise SyntaxError('Cannot reassign completions to scopes')
        return irnodes.AssignmentStatement(lhs, rhs)

    def typed_argument(self, args):
        if len(args) == 2:
            dtype, name = args
            annotations = []
        else:
            dtype, annotations, name = args

        return irnodes.KernelArgument(
            dtype,
            irnodes.Identifier(name, 0),
            readonly='readonly' in annotations,
            writeonly='writeonly' in annotations,
            compiletime='compiletime' in annotations)

    # Block types
    def kernel(self, args):
        if len(args) == 4:
            name, parameters, arguments, body = args
        else:
            name = None
            parameters, arguments, body = args

        return irnodes.Kernel(name, parameters, arguments, body)

    place_block = irnodes.PlaceBlock.from_lark
    dataflow_block = irnodes.DataflowBlock.from_lark
    compute_block = irnodes.ComputeBlock.from_lark

    def phase(self, args):
        body = args[0]
        place = [a for a in body if isinstance(a, irnodes.PlaceBlock)]
        dataflow = [a for a in body if isinstance(a, irnodes.DataflowBlock)]
        compute = [a for a in body if isinstance(a, irnodes.ComputeBlock)]
        return irnodes.Phase(place, dataflow, compute)

    def parameters(self, args):
        return [irnodes.Parameter(a) for a in args]

    # List types
    annotations = list
    call_arguments = list
    subscript_slice = list
    subgrid_expression = list
    hops = list
    vars = list
    typed_vars = list
    arguments = list
    kernel_body = list
    place_body = list
    dataflow_body = list
    phase_body = list

    def compute_body(self, args):
        if len(args) == 1 and isinstance(args[0], list):
            return args[0]
        return list(args)

    # Statements is a special list where newlines can appear as tokens
    def statements(self, args):
        return [a for a in args if a is not None]

# Helper functions


def _expr(val: irnodes.SpatialNode | int | float | str) -> irnodes.Expression:
    if isinstance(val, irnodes.Expression):
        return val
    return irnodes.Expression(val)
