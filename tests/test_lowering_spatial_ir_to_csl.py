import os
from spatialstencil.lowering.spatial_ir_to_csl import lower_spatial_ir_to_csl
from spatialstencil.syntax.spatial_ir import parser, passes
import pytest


def test_non_concrete_program():
    file = os.path.join(os.path.dirname(__file__), '..', 'samples', 'spatial', 'laplacian_routed.sptl')
    kernel = parser.parse_file(file)
    with pytest.raises(ValueError, match='parameter value'):
        lower_spatial_ir_to_csl(kernel)


def test_laplacian():
    file = os.path.join(os.path.dirname(__file__), '..', 'samples', 'spatial', 'laplacian_routed.sptl')
    kernel = parser.parse_file(file)
    kernel = passes.concretize_parameters(kernel, I=32, J=32, K=20)
    kernel = passes.constexpr_propagation(kernel)
    print(kernel.as_ir())
    csl_files = lower_spatial_ir_to_csl(kernel)
    print(csl_files)


if __name__ == '__main__':
    test_non_concrete_program()
    test_laplacian()
