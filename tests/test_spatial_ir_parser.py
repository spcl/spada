import unittest
from spatialstencil.syntax.spatial_ir import irnodes as spast, parser
import os


class TestSpatialIRParser(unittest.TestCase):

    def test_roundtrip_hdiff(self):
        """
        Tests a roundtrip IR->parse->IR->parse->IR for differences.
        """
        file = os.path.join(os.path.dirname(__file__), '..', 'samples', 'spatial', 'laplacian.sptl')
        program = parser.parse_file(file)
        ir_1 = program.as_ir()
        program2 = parser.parse_string(ir_1)
        ir_2 = program2.as_ir()
        print(ir_1)
        assert ir_1 == ir_2

    def test_visitor(self):
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
        self.streams.append((node.dx.value, node.dy.value))
        return self.generic_visit(node)


if __name__ == '__main__':
    unittest.main()
