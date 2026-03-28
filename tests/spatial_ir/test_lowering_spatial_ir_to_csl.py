import os
from spatialstencil.lowering.spatial_ir_to_csl import lower_spatial_ir_to_csl
from spatialstencil.syntax.spatial_ir import parser, passes
import pytest


def test_non_concrete_program():
    file = os.path.join(os.path.dirname(__file__), '..', '..', 'samples', 'spatial', 'add.sptl')
    kernel = parser.parse_file(file)
    with pytest.raises(ValueError, match='parameter value'):
        lower_spatial_ir_to_csl(kernel)


def test_add():
    file = os.path.join(os.path.dirname(__file__), '..', '..', 'samples', 'spatial', 'add.sptl')
    kernel = parser.parse_file(file)
    kernel = passes.concretize_parameters(kernel, N=32)
    kernel = passes.constexpr_propagation(kernel)
    csl_files = lower_spatial_ir_to_csl(kernel)
    for f in csl_files:
        print('=============')
        print(f.filename, ':')
        print(f.code)
        print('=============')


_COLLECTIVES_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'samples', 'spatial', 'collectives')

_COLLECTIVES = [
    ('scalar_reduce_1D.sptl',    dict(N=4)),
    ('chain_reduce_1D.sptl',     dict(N=4,  K=2)),
    ('tree_reduce_1D.sptl',      dict(L=3,  K=2)),
    ('twophase_reduce_1D.sptl',  dict(G=3,  S=4, K=2)),
    ('broadcast_1D.sptl',        dict(N=4,  K=4)),
    ('chain_reduce_2D.sptl',     dict(NX=4, NY=4, K=2)),
    ('tree_reduce_2D.sptl',      dict(LX=2, LY=2, K=2)),
    ('twophase_reduce_2D.sptl',  dict(GX=2, SX=4, GY=2, SY=4, K=2)),
    ('broadcast_2D.sptl',        dict(NX=4, NY=4, K=2)),
]


@pytest.mark.parametrize('filename,params', _COLLECTIVES, ids=[c[0] for c in _COLLECTIVES])
def test_collective(filename, params):
    file = os.path.join(_COLLECTIVES_DIR, filename)
    kernel = parser.parse_file(file)
    kernel = passes.concretize_parameters(kernel, **params)
    kernel = passes.constexpr_propagation(kernel)
    print(kernel.as_ir())
    csl_files = lower_spatial_ir_to_csl(kernel)
    for f in csl_files:
        print('=============')
        print(f.filename, ':')
        print(f.code)
        print('='*13)


def test_two_phase_split():
    file = os.path.join(os.path.dirname(__file__), '..', '..', 'samples', 'spatial', 'two_phase_split.sptl')
    kernel = parser.parse_file(file)
    kernel = passes.concretize_parameters(kernel, K=32)
    kernel = passes.constexpr_propagation(kernel)
    print(kernel.as_ir())
    csl_files = lower_spatial_ir_to_csl(kernel)
    for f in csl_files:
        print('=============')
        print(f.filename, ':')
        print(f.code)
        print('=============')


def test_laplacian():
    file = os.path.join(os.path.dirname(__file__), '..', '..', 'samples', 'spatial', 'laplacian_routed.sptl')
    kernel = parser.parse_file(file)
    kernel = passes.constexpr_propagation(kernel)
    print(kernel.as_ir())
    csl_files = lower_spatial_ir_to_csl(kernel)
    for f in csl_files:
        print('=============')
        print(f.filename, ':')
        print(f.code)
        print('=============')


def test_forward_sum():
    file = os.path.join(os.path.dirname(__file__), '..', '..', 'samples', 'spatial', 'forward_sum.sptl')
    kernel = parser.parse_file(file)
    kernel = passes.concretize_parameters(kernel, N=31, K=30)
    kernel = passes.constexpr_propagation(kernel)
    print(kernel.as_ir())
    csl_files = lower_spatial_ir_to_csl(kernel)
    for f in csl_files:

        print('=============')
        print(f.filename, ':')
        print(f.code)
        print('=============')


if __name__ == '__main__':
    test_non_concrete_program()
    test_add()
    for _fname, _params in _COLLECTIVES:
        test_collective(_fname, _params)
    test_two_phase_split()
    test_laplacian()
    test_forward_sum()
