"""Tests for the targeted vectorization of nested multiply-accumulate loops.

The lowering should turn

    for (k, l) in [0:K, 0:K]:
        z[k] = z[k] + A[k*K + l] * x[l]

(in either loop-header order) into a strided base DSD plus a per-column
``@increment_dsd_offset`` and the dtype-matched ``@fmac*`` builtin, and must
fall back to scalar loops for every shape it cannot prove safe (aliasing,
non-affine indices, unsupported dtype combinations, or ``--disable-dsd``).
"""
from typing import Optional

from spada.lowering.spatial_ir_to_csl import lower_spatial_ir_to_csl
from spada.syntax.csl import dsd_ops
from spada.syntax.spatial_ir import parser, passes


def _kernel(body: str, decls: Optional[str] = None,
            a_t: str = 'f32', x_t: str = 'f32', z_t: str = 'f32') -> str:
    decls = decls if decls is not None else f'''
            {a_t}[K*K] A_flat
            {x_t}[K]   x
            {z_t}[K]   z
    '''
    return f'''
    kernel @t<K>(
        stream<{a_t}, K*K>[2, 2] readonly  inp_A,
        stream<{x_t}, K>[2, 2]   readonly  inp_x,
        stream<{z_t}, K>[2, 2]   writeonly out
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

# Same computation with the loop-header order swapped: l is the outer variable.
MAC_BODY_SWAPPED = '''
                for i16 k in [0:K] {
                    z[k] = 0.0
                }
                for i16 l, i16 k in [0:K, 0:K] {
                    z[k] = z[k] + A_flat[k*K + l] * x[l]
                }
'''


def test_mac_loop_is_vectorized():
    code = _all_code(_lower(_kernel(MAC_BODY)))
    assert '@fmacs(' in code
    assert '@increment_dsd_offset(' in code


def test_swapped_loop_order_is_vectorized():
    # Operand roles come from the accumulator index, not from the loop-header
    # position, so `for (l, k)` vectorizes identically (issue #69).
    code = _all_code(_lower(_kernel(MAC_BODY_SWAPPED)))
    assert '@fmacs(' in code
    assert '@increment_dsd_offset(' in code


def test_f16_mac_uses_fmach():
    # An all-f16 MAC dispatches to @fmach and offsets the DSD as f16.
    code = _all_code(_lower(_kernel(MAC_BODY, a_t='f16', x_t='f16', z_t='f16')))
    assert '@fmach(' in code
    assert '@fmacs(' not in code
    assert ', f16);' in code


def test_mixed_precision_mac_uses_fmachs():
    # An f32 accumulate with an f16 scalar operand dispatches to @fmachs,
    # mirroring FMADSDOp._as_csl.
    code = _all_code(_lower(_kernel(MAC_BODY, a_t='f32', x_t='f16', z_t='f32')))
    assert '@fmachs(' in code


def test_unsupported_dtype_combo_falls_back():
    # f16 accumulator with f32 scalar operand has no @fmac* builtin: scalar loop.
    code = _all_code(_lower(_kernel(MAC_BODY, a_t='f16', x_t='f32', z_t='f16')))
    assert '@increment_dsd_offset(' not in code


def test_scalar_fallback_when_dsd_disabled():
    # MAC vectorization is part of DSD detection and is disabled by
    # --disable-dsd alone; there is no separate flag.
    try:
        code = _all_code(_lower(_kernel(MAC_BODY), disable_dsd=True))
        assert '@increment_dsd_offset(' not in code
        assert '@fmacs(' not in code
    finally:
        dsd_ops.DISABLE_DSD = False   # module flag is sticky, reset for other tests


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


def test_no_vectorization_when_accumulator_aliases_matrix_swapped_order():
    body = '''
                for i16 k in [0:K] {
                    z[k] = 1.0
                }
                for i16 l, i16 k in [0:K, 0:K] {
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
