import unittest

from spada.syntax.spatial_ir.irnodes import Kernel


class TestSpatialIR(unittest.TestCase):

    def test_validate_stencil_schema(self):
        Kernel.validate_schema()


if __name__ == '__main__':
    unittest.main()
