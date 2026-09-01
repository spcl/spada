"""Tests for the targeted vectorization of nested multiply-accumulate loops.

The lowering should turn

    for (k, l) in [0:K, 0:K]:
        z[k] = z[k] + A[k*K + l] * x[l]

into a strided base DSD plus a per-column ``@increment_dsd_offset`` and one of
``@fmach``/``@fmachs``/``@fmacs`` (dtype-dependent, mirroring FMADSDOp), and must fall back to
scalar loops for every shape it cannot prove safe (aliasing, non-affine indices, unsupported
dtype combinations, or ``--disable-dsd``). There is no separate disable flag for this pass: it is
gated solely by ``dsd_ops.DISABLE_DSD``, same as every other DSD operation.
"""
import pytest
from spada.lowering.spatial_ir_to_csl import lower_spatial_ir_to_csl
from spada.syntax.csl import dsd_ops
from spada.syntax.spatial_ir import parser, passes


def _kernel(body: str, decls: str = None, mat_ty: str = 'f32', out_ty: str = None) -> str:
    # mat_ty covers A_flat/x (the two @fmac* multiplicands, which must share a dtype); out_ty
    # covers z (the accumulator/destination), defaulting to mat_ty. Stream types must match the
    # local arrays they feed (receive/send are otherwise rejected as a dtype mismatch upstream of
    # MAC vectorization entirely), so both need to vary together with the local declarations.
    out_ty = out_ty if out_ty is not None else mat_ty
    decls = decls if decls is not None else f'''
            {mat_ty}[K*K] A_flat
            {mat_ty}[K]   x
            {out_ty}[K]   z
    '''
    return f'''
    kernel @t<K>(
        stream<{mat_ty}, K*K>[2, 2] readonly  inp_A,
        stream<{mat_ty}, K>[2, 2]   readonly  inp_x,
        stream<{out_ty}, K>[2, 2]   writeonly out
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
    # MAC vectorization has no dedicated disable flag -- it is folded into --disable-dsd, the
    # same switch that disables every other DSD operation (issue #69 ask 5).
    try:
        code = _all_code(_lower(_kernel(MAC_BODY), disable_dsd=True))
        assert '@increment_dsd_offset(' not in code
    finally:
        dsd_ops.DISABLE_DSD = False   # module flag is sticky, reset for other tests


# --- Loop-variable-order matrix (issue #69 ask 2: (k, l) and (l, k) must both be detected) -----

def _order_decl(order: str) -> str:
    """The `for <decl> in [0:K, 0:K]` variable declaration for a given loop-variable order."""
    return {'kl': 'i16 k, i16 l', 'lk': 'i16 l, i16 k'}[order]


def _mac_body(order: str) -> str:
    return f'''
                for i16 k in [0:K] {{
                    z[k] = 0.0
                }}
                for {_order_decl(order)} in [0:K, 0:K] {{
                    z[k] = z[k] + A_flat[k*K + l] * x[l]
                }}
    '''


ORDERS = ['kl', 'lk']

# (mat_ty, out_ty, expected op) for the dtype combinations FMADSDOp actually models
# (dsd_ops.DSD_ASSIGNMENT_MAPPING has no integer @fmac* builtin, so only f16/f32 are covered).
SUPPORTED_DTYPE_COMBOS = [
    ('f32', 'f32', '@fmacs('),   # both multiplicands and accumulator f32
    ('f16', 'f16', '@fmach('),   # both multiplicands and accumulator f16
    ('f32', 'f16', '@fmachs('),  # f32 multiplicands, f16 accumulate (mixed)
]

# (mat_ty, out_ty) combinations that must NOT vectorize: either FMADSDOp itself has no dispatch
# branch for the combination (f16 multiplicands / f32 accumulate), or no @fmac* builtin exists
# for the dtype at all (integer).
UNSUPPORTED_DTYPE_COMBOS = [
    ('f16', 'f32'),
    ('i16', 'i16'),
]


@pytest.mark.parametrize('order', ORDERS)
@pytest.mark.parametrize('mat_ty, out_ty, expected_op', SUPPORTED_DTYPE_COMBOS)
def test_mac_loop_vectorized_for_order_and_dtype(order, mat_ty, out_ty, expected_op):
    code = _all_code(_lower(_kernel(_mac_body(order), mat_ty=mat_ty, out_ty=out_ty)))
    assert '@increment_dsd_offset(' in code
    assert expected_op in code


@pytest.mark.parametrize('order', ORDERS)
@pytest.mark.parametrize('mat_ty, out_ty', UNSUPPORTED_DTYPE_COMBOS)
def test_no_vectorization_for_unsupported_dtype_combo(order, mat_ty, out_ty):
    code = _all_code(_lower(_kernel(_mac_body(order), mat_ty=mat_ty, out_ty=out_ty)))
    assert '@increment_dsd_offset(' not in code


@pytest.mark.parametrize('order', ORDERS)
def test_no_vectorization_when_accumulator_aliases_matrix(order):
    # z appears as the matrix operand: DSD reordering would change semantics.
    body = f'''
                for i16 k in [0:K] {{
                    z[k] = 1.0
                }}
                for {_order_decl(order)} in [0:K, 0:K] {{
                    z[k] = z[k] + z[k*1 + l*0] * x[l]
                }}
    '''
    code = _all_code(_lower(_kernel(body)))
    assert '@increment_dsd_offset(' not in code


@pytest.mark.parametrize('order', ORDERS)
def test_no_vectorization_when_accumulator_aliases_vector(order):
    body = f'''
                for i16 k in [0:K] {{
                    z[k] = 1.0
                }}
                for {_order_decl(order)} in [0:K, 0:K] {{
                    z[k] = z[k] + A_flat[k*K + l] * z[l]
                }}
    '''
    code = _all_code(_lower(_kernel(body)))
    assert '@increment_dsd_offset(' not in code


@pytest.mark.parametrize('order', ORDERS)
def test_no_vectorization_for_nonaffine_index(order):
    body = f'''
                for i16 k in [0:K] {{
                    z[k] = 0.0
                }}
                for {_order_decl(order)} in [0:K, 0:K] {{
                    z[k] = z[k] + A_flat[k*l] * x[l]
                }}
    '''
    code = _all_code(_lower(_kernel(body)))
    assert '@increment_dsd_offset(' not in code


@pytest.mark.parametrize('order', ORDERS)
def test_no_vectorization_when_accumulator_differs(order):
    body = f'''
                for i16 k in [0:K] {{
                    z[k] = 0.0
                }}
                for {_order_decl(order)} in [0:K, 0:K] {{
                    z[k] = x[k] + A_flat[k*K + l] * x[l]
                }}
    '''
    code = _all_code(_lower(_kernel(body)))
    assert '@increment_dsd_offset(' not in code
