import unittest
from spada.syntax.stencil_ir import irnodes as sast, parser
import os


class TestStencilIRParser(unittest.TestCase):

    def test_if_block(self):
        """
        Tests parsing an if block
        """
        src = '''
        %out = spst.program (%inp) {} : 
          field<domain<?, ?, ?>, bool> ->
          field<domain<?, ?, ?>, f32> {
            %out = spst.computation(%inp) {
              schedule = PARALLEL,
              interval = [interval<?, ?>, interval<?, ?>, interval<?, ?>]
            } : view<domain<?, ?, ?>, extent<(?, ?, ?)>, f32> ->
                view<domain<?, ?, ?>, extent<(?, ?, ?)>, f32> {                    
                    %b = spst.if (%inp) : view<domain<?, ?, ?>, extent<(?, ?, ?)>, bool> -> view<domain<?, ?, ?>, extent<(?, ?, ?)>, f32> {
                        spst.return %inp
                    } elif (%arg1) {
                        spst.return %inp + 1
                    } else {
                        spst.return 0
                    }
                    %out = spst.if (%b) : view<domain<?, ?, ?>, extent<(?, ?, ?)>, bool> -> view<domain<?, ?, ?>, extent<(?, ?, ?)>, f32> {
                        spst.return %inp
                    } else {
                        spst.return 0
                    }
                    spst.return %out
            }
            spst.return %out
        }
        '''
        program = parser.parse_string(src)
        comp = program.computations[0]

        # Branch 1
        assert isinstance(comp.body[0], sast.IfBlock)
        assert len(comp.body[0].outputs) == 1
        assert comp.body[0].outputs[0].as_ir() == '%b'
        assert comp.body[0].condition.as_ir() == '%inp'
        assert len(comp.body[0].else_ifs) == 2
        assert comp.body[0].else_ifs[-1].condition is None

        # Branch 2
        assert isinstance(comp.body[1], sast.IfBlock)
        assert len(comp.body[1].outputs) == 1
        assert comp.body[1].outputs[0].as_ir() == '%out'
        assert comp.body[1].condition.as_ir() == '%b'
        assert len(comp.body[1].else_ifs) == 1
        assert comp.body[1].else_ifs[0].condition is None

    def test_mathcall(self):
        """
        Tests parsing a math call
        """
        src = '''
        %out = spst.program (%inp) {} : 
          field<domain<?, ?, ?>, f32> ->
          field<domain<?, ?, ?>, f32> {
            %out = spst.computation(%inp) {
              schedule = PARALLEL,
              interval = [interval<?, ?>, interval<?, ?>, interval<?, ?>]
            } : view<domain<?, ?, ?>, extent<(?, ?, ?)>, f32> ->
                view<domain<?, ?, ?>, extent<(?, ?, ?)>, f32> {                    
                    %out = spst.statement (%inp) {} : 
                      spst.view<spst.cartesian<?, ?, ?>, spst.extent<(?, ?, ?)>, f32> -> 
                      spst.view<spst.cartesian<?, ?, ?>, spst.extent<(?, ?, ?)>, f32> {
                          spst.return sqrt(%inp) : f32
                    }
                    spst.return %out
            }
            spst.return %out
        }
        '''
        program = parser.parse_string(src)
        comp = program.computations[0]

        assert isinstance(comp.body[0], sast.StatementBlock)
        stmt = comp.body[0]
        assert len(stmt.body) == 1
        assert isinstance(stmt.body[0], sast.ReturnOp)
        retop = stmt.body[0]
        assert len(retop.values) == 1
        assert isinstance(retop.values[0].value, sast.MathCall)
        mathcall = retop.values[0].value
        assert mathcall.func == 'sqrt'
        assert [arg.as_ir() for arg in mathcall.arguments] == ['%inp']

    def test_roundtrip_hdiff(self):
        """
        Tests a roundtrip IR->parse->IR->parse->IR for differences.
        """
        file = os.path.join(os.path.dirname(__file__), '..', '..', 'samples', 'spst', 'hdiff.spst')
        program = parser.parse_file(file)
        ir_1 = program.as_ir()
        program2 = parser.parse_string(ir_1)
        ir_2 = program2.as_ir()
        assert ir_1 == ir_2

    def test_roundtrip_vadv(self):
        """
        Tests a roundtrip IR->parse->IR->parse->IR for differences.
        """
        file = os.path.join(os.path.dirname(__file__), '..', '..', 'samples', 'spst', 'vadv.spst')
        program = parser.parse_file(file)
        ir_1 = program.as_ir()
        program2 = parser.parse_string(ir_1)
        ir_2 = program2.as_ir()
        assert ir_1 == ir_2

    def test_shorthand_notation(self):
        file = os.path.join(os.path.dirname(__file__), '..', '..', 'samples', 'spst', 'laplacian_mat_ext.spst')
        program = parser.parse_file(file)
        file2 = os.path.join(os.path.dirname(__file__), '..', '..', 'samples', 'spst', 'laplacian_mat_sh_ext.spst')
        program2 = parser.parse_file(file2)
        assert program.as_ir() == program2.as_ir()

    def test_visitor(self):
        """
        Tests the IR node visitor for the stencil IR.
        """
        file = os.path.join(os.path.dirname(__file__), '..', '..', 'samples', 'spst', 'vadv.spst')
        program = parser.parse_file(file)

        visitor = IntervalCounter()
        visitor.visit(program)
        assert visitor.counter == 5


class IntervalCounter(sast.NodeVisitor):
    """
    Test helper class that counts computation blocks
    """

    def __init__(self):
        super().__init__()
        self.counter = 0

    def visit_ComputationBlock(self, node: sast.ComputationBlock):
        self.counter += 1
        return self.generic_visit(node)


if __name__ == '__main__':
    unittest.main()
