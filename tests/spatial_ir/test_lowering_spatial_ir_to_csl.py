import os
from spatialstencil.lowering.spatial_ir_to_csl import lower_spatial_ir_to_csl
from spatialstencil.syntax.spatial_ir import parser, passes
import pytest


def test_non_concrete_program():
    file = os.path.join(os.path.dirname(__file__), '..', '..', 'samples', 'spatial', 'simple', 'add.sptl')
    kernel = parser.parse_file(file)
    with pytest.raises(ValueError, match='parameter value'):
        lower_spatial_ir_to_csl(kernel)


def test_add():
    file = os.path.join(os.path.dirname(__file__), '..', '..', 'samples', 'spatial', 'simple', 'add.sptl')
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

_COLLECTIVES_1D = [
    ('scalar_reduce_1D.sptl',    dict(N=4)),
    ('chain_reduce_1D.sptl',     dict(N=4,  K=2)),
    ('tree_reduce_1D.sptl',      dict[str, int](L=1,  K=2)),
    ('tree_reduce_1D.sptl',      dict[str, int](L=2,  K=2)),
    ('tree_reduce_1D.sptl',      dict[str, int](L=3,  K=2)),
    ('twophase_reduce_1D.sptl',  dict(G=3,  S=4, K=2)),
    ('twophase_reduce_1D.sptl',  dict(G=4,  S=4, K=2)),
    ('broadcast_1D.sptl',        dict(N=4,  K=4))
]

_COLLECTIVES_2D = [
    ('chain_reduce_2D.sptl',     dict(NX=4, NY=4, K=2)),
    ('tree_reduce_2D.sptl',      dict(LX=2, LY=2, K=2)),
    ('twophase_reduce_2D.sptl',  dict[str, int](GX=3, SX=4, GY=3, SY=4, K=2)),
    ('twophase_reduce_2D.sptl',  dict[str, int](GX=4, SX=4, GY=4, SY=4, K=2)),
    ('broadcast_2D.sptl',        dict(NX=4, NY=4, K=2)),
]

@pytest.mark.parametrize('filename,params', _COLLECTIVES_1D, ids=[c[0] for c in _COLLECTIVES_1D])
def test_collective_1d(filename, params):
    file = os.path.join(_COLLECTIVES_DIR, filename)
    kernel = parser.parse_file(file)
    kernel = passes.concretize_parameters(kernel, **params)
    kernel = passes.constexpr_propagation(kernel)
    print(kernel.as_ir())
    csl_files = lower_spatial_ir_to_csl(kernel, copy_elision=True, prune_memory=True)
    for f in csl_files:
        print('=============')
        print(f.filename, ':')
        print(f.code)
        print('='*13)


def test_tree_reduce_1d_compiles_512_pes():
    """Lowering succeeds for a 512-wide (L=9) 1-D tree reduce; guards large-PE regressions."""
    file = os.path.join(_COLLECTIVES_DIR, 'tree_reduce_1D.sptl')
    kernel = parser.parse_file(file)
    # K must be >= 2: K=1 breaks foreach/receive lowering (empty DSD slot for __x).
    kernel = passes.concretize_parameters(kernel, L=9, K=2)
    kernel = passes.constexpr_propagation(kernel)
    csl_files = lower_spatial_ir_to_csl(kernel, copy_elision=True, prune_memory=True)
    assert csl_files, 'expected at least one generated CSL file'
    assert all(f.code.strip() for f in csl_files), 'expected non-empty CSL bodies'


@pytest.mark.parametrize('filename,params', _COLLECTIVES_2D, ids=[c[0] for c in _COLLECTIVES_2D])
def test_collective_2d(filename, params):
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
    file = os.path.join(os.path.dirname(__file__), 'samples', 'two_phase_split.sptl')
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


def test_neighbor_exchange_lowers():
    """``neighbor_exchange.sptl`` lowers to CSL without errors (neighbor send/receive)."""
    file = os.path.join(os.path.dirname(__file__), 'samples', 'neighbor_exchange.sptl')
    kernel = parser.parse_file(file)
    kernel = passes.concretize_parameters(kernel, K=4)
    kernel = passes.constexpr_propagation(kernel)
    csl_files = lower_spatial_ir_to_csl(kernel)
    assert csl_files, 'expected at least one generated CSL file'
    assert all(f.code.strip() for f in csl_files), 'expected non-empty CSL bodies'


def test_laplacian():
    file = os.path.join(os.path.dirname(__file__), '..', '..', 'samples', 'spatial', 'stencils', 'laplacian_routed.sptl')
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
    file = os.path.join(os.path.dirname(__file__), '..', '..', 'samples', 'spatial', 'simple', 'forward_sum.sptl')
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


def test_prints_place_block_bytes_per_rectangle(capsys):
    kernel = parser.parse_string(
        """
        kernel @test<>() {
            place u16 i, u16 j in [0:1, 0:1] {
                i16 a
            }
            compute u16 i, u16 j in [0:1, 0:1] {
                a = 1
            }

            place u16 i, u16 j in [1:2, 0:1] {
                f32 b
                f32 c
            }
            compute u16 i, u16 j in [1:2, 0:1] {
                b = 1.0
                c = b
            }
        }
        """,
        "test_place_block_stats.sptl",
    )

    lower_spatial_ir_to_csl(kernel, disable_benchmarking=True, prune_memory=False)

    captured = capsys.readouterr()
    assert 'Stats P0,0: 2 bytes/PE' in captured.out
    assert 'Stats P1,0: 8 bytes/PE' in captured.out


if __name__ == '__main__':
    test_non_concrete_program()
    test_add()
    for _fname, _params in _COLLECTIVES_1D:
        test_collective_1d(_fname, _params)
    for _fname, _params in _COLLECTIVES_2D:
        test_collective_2d(_fname, _params)
    test_two_phase_split()
    test_neighbor_exchange_lowers()
    test_laplacian()
    test_forward_sum()
