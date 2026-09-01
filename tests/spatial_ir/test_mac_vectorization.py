"""Tests for the targeted vectorization of nested multiply-accumulate loops.

The lowering should turn

    for (k, l) in [0:K, 0:K]:
        z[k] = z[k] + A[k*K + l] * x[l]

into a strided base DSD plus a per-column ``@increment_dsd_offset`` and ``@fmacs``,
and must fall back to scalar loops for every shape it cannot prove safe
(aliasing, non-affine indices, non-f32 dtypes, or the explicit disable flag).
"""
import pytest
from spada.lowering.spatial_ir_to_csl import lower_spatial_ir_to_csl
from spada.syntax.csl import statements as cslstmt
from spada.syntax.spatial_ir import parser, passes


def _kernel(body: str, decls: str = None) -> str:
    decls = decls if decls is not None else '''
            f32[K*K] A_flat
            f32[K]   x
            f32[K]   z
    '''
    return f'''
    kernel @t<K>(
        stream<f32, K*K>[2, 2] readonly  inp_A,
        stream<f32, K>[2, 2]   readonly  inp_x,
        stream<f32, K>[2, 2]   writeonly out
    ) {{
        place i16 i, i16 j in [0:2, 0:2] {{
            {decls}
        }}
        phase {{
            compute i16 i, i16 j in [0:2, 0:2] {{
                await receive(A_flat, inp_A[i, j])
                await receive(x, inp_x[i, j])
                {body}
                await send(z, out[i, j])
            }}
        }}
    }}
    '''


def _lower(code: str, **lower_kwargs):
    kernel = parser.parse_string(code, 'test.sptl')
    kernel = passes.concretize_parameters(kernel, K=4)
    kernel = passes.constexpr_propagation(kernel)
    return lower_spatial_ir_to_csl(kernel, **lower_kwargs)


def _all_code(csl_files) -> str:
    return '\n'.join(f.code for f in csl_files)


MAC_BODY = '''
                for i16 k in [0:K] {
                    z[k] = 0.0
                }
                for i16 k, i16 l in [0:K, 0:K] {
                    z[k] = z[k] + A_flat[k*K + l] * x[l]
                }
'''


def test_mac_loop_is_vectorized():
    code = _all_code(_lower(_kernel(MAC_BODY)))
    assert '@fmacs(' in code
    assert '@increment_dsd_offset(' in code


def test_scalar_fallback_when_disabled():
    try:
        code = _all_code(_lower(_kernel(MAC_BODY), disable_mac_vectorization=True))
        assert '@increment_dsd_offset(' not in code
    finally:
        cslstmt.DISABLE_MAC_VECTORIZATION = False   # module flag is sticky, reset for other tests


def test_no_vectorization_when_accumulator_aliases_matrix():
    # z appears as the matrix operand: DSD reordering would change semantics.
    body = '''
                for i16 k in [0:K] {
                    z[k] = 1.0
                }
                for i16 k, i16 l in [0:K, 0:K] {
                    z[k] = z[k] + z[k*1 + l*0] * x[l]
                }
    '''
    code = _all_code(_lower(_kernel(body)))
    assert '@increment_dsd_offset(' not in code


def test_no_vectorization_when_accumulator_aliases_vector():
    body = '''
                for i16 k in [0:K] {
                    z[k] = 1.0
                }
                for i16 k, i16 l in [0:K, 0:K] {
                    z[k] = z[k] + A_flat[k*K + l] * z[l]
                }
    '''
    code = _all_code(_lower(_kernel(body)))
    assert '@increment_dsd_offset(' not in code


def test_no_vectorization_for_nonaffine_index():
    body = '''
                for i16 k in [0:K] {
                    z[k] = 0.0
                }
                for i16 k, i16 l in [0:K, 0:K] {
                    z[k] = z[k] + A_flat[k*l] * x[l]
                }
    '''
    code = _all_code(_lower(_kernel(body)))
    assert '@increment_dsd_offset(' not in code


def test_no_vectorization_when_accumulator_differs():
    body = '''
                for i16 k in [0:K] {
                    z[k] = 0.0
                }
                for i16 k, i16 l in [0:K, 0:K] {
                    z[k] = x[k] + A_flat[k*K + l] * x[l]
                }
    '''
    code = _all_code(_lower(_kernel(body)))
    assert '@increment_dsd_offset(' not in code
