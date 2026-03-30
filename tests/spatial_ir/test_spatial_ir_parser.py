from spatialstencil.syntax.spatial_ir import irnodes as spast, parser
import os


def test_spatial_roundtrip_laplacian():
    """
    Tests a roundtrip IR->parse->IR->parse->IR for differences.
    """
    file = os.path.join(os.path.dirname(__file__), '..', '..', 'samples', 'spatial', 'stencils', 'laplacian.sptl')
    program = parser.parse_file(file)
    ir_1 = program.as_ir()
    program2 = parser.parse_string(ir_1)
    ir_2 = program2.as_ir()
    assert ir_1 == ir_2


def test_spatial_visitor():
    """
    Tests the IR node visitor for the spatial IR.
    """
    file = os.path.join(os.path.dirname(__file__), '..', '..', 'samples', 'spatial', 'stencils', 'laplacian.sptl')
    program = parser.parse_file(file)

    visitor = StreamCollector()
    visitor.visit(program)
    assert visitor.streams == [(1, 0), (-1, 0), (0, -1), (0, 1)]


class StreamCollector(spast.NodeVisitor):
    """
    Test helper class that counts computation blocks
    """

    def __init__(self):
        super().__init__()
        self.streams = []

    def visit_RelativeStreamDeclaration(self, node: spast.RelativeStreamDeclaration):
        self.streams.append((self._expand_ops(node.dx.value), self._expand_ops(node.dy.value)))
        return self.generic_visit(node)

    def _expand_ops(self, op: spast.UnaryOperator | spast.BinaryOperator | spast.ConstantLiteral) -> int:
        if isinstance(op, (int, float)):
            return op
        if isinstance(op, spast.ConstantLiteral):
            return op.value
        if isinstance(op, spast.Expression):
            return self._expand_ops(op.value)
        if isinstance(op, spast.UnaryOperator):
            if op.op == '+':
                return self._expand_ops(op.value)
            elif op.op == '-':
                return -self._expand_ops(op.value)
            raise TypeError(f'Unsupported unary op "{op.op}"')
        elif isinstance(op, spast.BinaryOperator):
            if op.op == '+':
                return self._expand_ops(op.left) + self._expand_ops(op.right)
            elif op.op == '-':
                return self._expand_ops(op.left) - self._expand_ops(op.right)
            raise TypeError(f'Unsupported binary op "{op.op}"')
        raise TypeError(f'Unsupported node type "{type(op).__name__}"')


def test_spatial_roundtrip_two_phase():
    """
    Tests a roundtrip IR->parse->IR->parse->IR for differences.
    """
    file = os.path.join(os.path.dirname(__file__), 'samples', 'two_phase.sptl')
    _rountrip_test(file)


def test_spatial_roundtrip_forward():
    """
    Tests a roundtrip IR->parse->IR->parse->IR for differences.
    """
    file = os.path.join(os.path.dirname(__file__), '..', '..', 'samples', 'spatial', 'simple', 'forward_sum.sptl')
    _rountrip_test(file)


def test_spatial_roundtrip_backward():
    """
    Tests a roundtrip IR->parse->IR->parse->IR for differences.
    """
    file = os.path.join(os.path.dirname(__file__), '..', '..', 'samples', 'spatial', 'simple', 'backward_sum.sptl')
    _rountrip_test(file)


def _rountrip_test(file):
    """
    Tests a roundtrip IR->parse->IR->parse->IR for differences.

    :param file:
    :return:
    """
    program = parser.parse_file(file)
    ir_1 = program.as_ir()
    program2 = parser.parse_string(ir_1)
    ir_2 = program2.as_ir()
    assert ir_1 == ir_2


def test_spatial_roundtrip_two_phase_unrouted():
    file = os.path.join(os.path.dirname(__file__), 'samples', 'two_phase_unrouted.sptl')
    _rountrip_test(file)


def test_spatial_roundtrip_two_phase_split():
    file = os.path.join(os.path.dirname(__file__), 'samples', 'two_phase_split.sptl')
    _rountrip_test(file)


def test_extern_field():
    """
    Tests parsing the ``extern`` field qualifier in place blocks.
    """
    code = """
    kernel @test<N>(f32 coeff) {
        place u16 i, u16 j in [0:N, 0:N] {
            extern f32[1] a;
            f32 local_a;
            extern f32[1] out;
        }
    }"""
    kernel = parser.parse_string(code)
    place_block = next(stmt for stmt in kernel.body if isinstance(stmt, spast.PlaceBlock))
    a_decl, local_a_decl, out_decl = place_block.statements
    assert not local_a_decl.is_extern
    assert a_decl.is_extern
    assert out_decl.is_extern


def test_extern_stream():
    """
    Tests parsing the ``extern_stream`` stream in dataflow blocks.
    """
    code = """
    kernel @test<N>(f32 coeff) {
        dataflow u16 i, u16 j in [0:N, 0:N] {
            stream<f32> in_stream = extern_stream(in);
            stream<f32> out_stream = extern_stream(out) {
                hops = auto,
                channel = 3
            };
        }
    }"""
    kernel = parser.parse_string(code)
    df_block = next(stmt for stmt in kernel.body if isinstance(stmt, spast.DataflowBlock))
    in_decl, out_decl = df_block.statements
    assert isinstance(in_decl.stream, spast.ExternStreamDeclaration)
    assert isinstance(out_decl.stream, spast.ExternStreamDeclaration)
    assert in_decl.stream.direction == 'in'
    assert out_decl.stream.direction == 'out'
    assert out_decl.stream.routing.resolved_channel == 3


if __name__ == '__main__':
    test_spatial_roundtrip_laplacian()
    test_spatial_visitor()
    test_spatial_roundtrip_two_phase()
    test_spatial_roundtrip_two_phase_unrouted()
    test_spatial_roundtrip_two_phase_split()
    test_spatial_roundtrip_forward()
    test_spatial_roundtrip_backward()
    test_extern_field()
    test_extern_stream()
