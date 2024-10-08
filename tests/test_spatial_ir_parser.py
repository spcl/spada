from spatialstencil.syntax.spatial_ir import irnodes as spast, parser
import os


def test_spatial_roundtrip_laplacian():
    """
    Tests a roundtrip IR->parse->IR->parse->IR for differences.
    """
    file = os.path.join(os.path.dirname(__file__), '..', 'samples', 'spatial', 'laplacian.sptl')
    program = parser.parse_file(file)
    ir_1 = program.as_ir()
    program2 = parser.parse_string(ir_1)
    ir_2 = program2.as_ir()
    assert ir_1 == ir_2


def test_spatial_visitor():
    """
    Tests the IR node visitor for the spatial IR.
    """
    file = os.path.join(os.path.dirname(__file__), '..', 'samples', 'spatial', 'laplacian.sptl')
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
    file = os.path.join(os.path.dirname(__file__), '..', 'samples', 'spatial', 'two_phase.sptl')
    _rountrip_test(file)


def test_spatial_roundtrip_forward():
    """
    Tests a roundtrip IR->parse->IR->parse->IR for differences.
    """
    file = os.path.join(os.path.dirname(__file__), '..', 'samples', 'spatial', 'forward_sum.sptl')
    _rountrip_test(file)


def test_spatial_roundtrip_backward():
    """
    Tests a roundtrip IR->parse->IR->parse->IR for differences.
    """
    file = os.path.join(os.path.dirname(__file__), '..', 'samples', 'spatial', 'backward_sum.sptl')
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
    file = os.path.join(os.path.dirname(__file__), '..', 'samples', 'spatial', 'two_phase_unrouted.sptl')
    _rountrip_test(file)


def test_spatial_roundtrip_two_phase_split():
    file = os.path.join(os.path.dirname(__file__), '..', 'samples', 'spatial', 'two_phase_split.sptl')
    _rountrip_test(file)


if __name__ == '__main__':
    test_spatial_roundtrip_laplacian()
    test_spatial_visitor()
    test_spatial_roundtrip_two_phase()
    test_spatial_roundtrip_two_phase_unrouted()
    test_spatial_roundtrip_two_phase_split()
    test_spatial_roundtrip_forward()
    test_spatial_roundtrip_backward()
