import lark

from spatialstencil.syntax.common.types import ScalarType
from spatialstencil.syntax.spatial_ir import irnodes
from spatialstencil.syntax.spatial_ir.irnodes import StreamType, Identifier


class TreeToSpatialIR(lark.Transformer):

    def __init__(self, filename: str = None):
        super().__init__()
        self.filename = filename

    def _call_userfunc(self, tree, new_children=None):
        """
        Override the default _call_userfunc to add source line information.
        """
        # Call the original function with the transformed children
        result = super()._call_userfunc(tree, new_children)

        # Add source line information to the result
        if isinstance(result, irnodes.SpatialNode):
            try:
                result.lineinfo = irnodes.LineInfo(self.filename, tree.meta.line, tree.meta.column)
            except AttributeError:
                result.lineinfo = None

        return result

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
    prefix = lambda self, _: None
    auto = lambda self, _: 'auto'

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
                return irnodes.ConstantLiteral(int(args[0]), ScalarType.UNKNOWN)
            except ValueError:
                try:
                    return irnodes.ConstantLiteral(float(args[0]), ScalarType.UNKNOWN)
                except ValueError:
                    return irnodes.Identifier(args[0], 0)
        return irnodes.Identifier(*args)

    typed_var = irnodes.TypedIdentifier.from_lark

    float_type = int_type = uint_type = bool_type = lambda self, args: getattr(irnodes.ScalarType, str(args[0]))

    def stream_type(self, args):
        return irnodes.StreamType(args[0], args[1] if len(args) > 1 else None)

    def array_type(self, args):
        return irnodes.ArrayType(args[0], args[1:])

    def value_expr(self, args):
        # Contract/inline value expressions that only contain another value expression
        if len(args) == 1 and isinstance(args[0], lark.Tree) and args[0].data == 'value_expr':
            return args[0]
        if len(args) == 1 and isinstance(args[0], (int, bool, float, str)):
            return irnodes.Expression(irnodes.ConstantLiteral(args[0], ScalarType.UNKNOWN))
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

    def call(self, args):
        if args[0] == 'fmac':
            return irnodes.MultiplyAccumulateOperator(_expr(args[1][0]), _expr(args[1][1]), _expr(args[1][2]))
        raise SyntaxError(f'Unrecognized function call to "{args[0]}"')

    # Free function call to builtins
    def function_call(self, args, meta=None):
        if isinstance(args[0], irnodes.Completion):
            completion = args[0]
            func, arguments = args[1:]
        else:
            completion = None
            func, arguments = args[1:]

        if func == 'send':
            return irnodes.SendStatement(*arguments, completion_name=completion)
        elif func == 'receive':
            return irnodes.ReceiveStatement(*arguments, completion_name=completion)
        raise SyntaxError(f'Unrecognized free function call to "{func}"')

    subscript = irnodes.ArraySlice.from_lark
    subscript_expr = irnodes.ArraySlice.from_lark

    # Grid/Subgrid expressions
    range_expression = irnodes.RangeExpression.from_lark

    # Declarations and routing
    def hop(self, args):
        return irnodes.RoutingHop(tuple(args))

    routing = irnodes.RoutingDeclaration.from_lark
    field_declaration = irnodes.FieldDeclaration.from_lark
    subgrid_expression_2d = irnodes.SubgridExpression.from_lark

    def hop(self, args):
        o = (args[0], args[1])
        return irnodes.RoutingHop(o)

    def stream_declaration(self, args):
        return irnodes.RelativeStreamDeclaration(*args)

    # Scopes
    def _scope_wrapper(self, cls, args):
        """
        A scope wrapper that handles `completion` assignments and `await` statements
        """
        completion = None
        if isinstance(args[0], irnodes.Completion) or args[0] is None:
            completion = args[0]
            args = args[1:]
        return cls(*args, completion_name=completion)

    for_stmt = lambda self, args: irnodes.ForStatement.from_lark(args)
    map_stmt = lambda self, args: self._scope_wrapper(irnodes.MapStatement, args)
    async_stmt = irnodes.AsyncBlock.from_lark
    awaitall_stmt = irnodes.AwaitAllStatement.from_lark

    # Foreach statements and generators
    receive_generator = irnodes.ReceiveGenerator.from_lark

    def foreach_stmt(self, args):
        completion = None
        if isinstance(args[0], irnodes.Completion) or args[0] is None:
            completion = args[0]
            args = args[1:]

        iters, generators, body = args

        # Split out generator from potential zipped range iterator(s)
        # Semantic check: a foreach must have at least one stream generator
        try:
            stream_varind, stream_gen = next(
                (i, gen) for i, gen in enumerate(generators) if isinstance(gen, irnodes.ReceiveGenerator))
        except StopIteration:
            raise SyntaxError('A foreach statement must have at least one stream `receive` generator')

        other_gens = [gen for gen in generators if gen is not stream_gen]
        itervars = [it for i, it in enumerate(iters) if i != stream_varind]
        if len(other_gens) > 1:
            raise NotImplementedError('Only one foreach zipped range is supported at the moment')
        if not other_gens:
            other_gens = [[]]

        return irnodes.ForeachStatement(
            itervars, other_gens[0], iters[stream_varind], stream_gen, body, completion_name=completion)

    # Await for a completion object
    def await_completion(self, args):
        if args[0].name == 'all':
            return irnodes.AwaitAllStatement()
        return irnodes.AwaitCompletionStatement.from_lark(args)

    # Definitions and assignments
    completion = irnodes.Completion.from_lark
    assignment = irnodes.AssignmentStatement.from_lark

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
    generators = list
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
    if isinstance(val, int):
        return irnodes.Expression(irnodes.ConstantLiteral(val, irnodes.ScalarType.UNKNOWN))
    if isinstance(val, float):
        return irnodes.Expression(irnodes.ConstantLiteral(val, irnodes.ScalarType.UNKNOWN))
    return irnodes.Expression(val)
