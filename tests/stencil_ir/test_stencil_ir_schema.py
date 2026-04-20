import unittest
from spada.syntax.stencil_ir.irnodes import Program

class TestStencilIR(unittest.TestCase):
    def test_validate_stencil_schema(self):
        Program.validate_schema()


if __name__ == '__main__':
    unittest.main()
