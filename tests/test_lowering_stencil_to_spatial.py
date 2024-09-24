import unittest
from pathlib import Path


from spatialstencil.syntax.stencil_ir import type_inference, parser

import spatialstencil.syntax.stencil_ir.irnodes as sast
import spatialstencil.syntax.spatial_ir.irnodes as spast

from spatialstencil.lowering.stencil_to_spatial import lower_stencil_to_spatial

class TestTypeInference(unittest.TestCase):


    def test_lowering_finishes(self):
        # For every file, run the parser, infer_extents, infer_domains,
        # lower_stencil_to_spatial, and print the result
        # This a basic check that the lowering finishes without errors

        files = [
            Path(__file__).parent / Path('../samples/spst/laplacian_3ac.spst')#,
            #Path(__file__).parent / Path('../samples/spst/if_else_ext.spst'),
            #Path(__file__).parent / Path('../samples/spst/multiple_returns_ext.spst'),
            #Path(__file__).parent / Path('../samples/spst/laplacian_mat_sh_ext.spst')
        ]

        for file in files:
            with open(file, 'r') as f:
                program = parser.parse_file(f)

            type_inference.infer_field_extents(program)
            domain = sast.Cartesian(x=sast.Interval(0, 250), y=sast.Interval(0, 250), z=sast.Interval(0, 80))
            type_inference.infer_field_domains(program, domain)
            print(program.as_ir())
            spatial_program = lower_stencil_to_spatial(program)
            print(spatial_program.as_ir())


if __name__ == '__main__':
    unittest.main()
