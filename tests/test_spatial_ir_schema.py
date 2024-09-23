import unittest

from spatialstencil.syntax.spatial_ir.irnodes import Kernel


class TestStencilIR(unittest.TestCase):
    def test_validate_stencil_schema(self):
        Kernel.validate_schema()


if __name__ == '__main__':
    unittest.main()
