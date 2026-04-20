import pytest
from spada.lowering import spatial_ir_to_csl as s2c
from spada.syntax.spatial_ir import parser, passes
from spada.syntax.spatial_ir.canonicalization import PEBlock
from spada.syntax.csl import dsd_ops


def test_dsd_op_detection():
    kernel = parser.parse_string(code=f"""
kernel @tester<K> (stream<f32>[4] readonly in,
                          stream<f32> readonly out ) {{

    place i16 i, i16 j in [0, 0] {{
        f32[K] a32
        f16[K] a16
        f32 localval32
        f16 localval16
    }}
    dataflow i32 i, i32 j in [0, 0] {{
        stream<f32> stream = relative_stream(-1, 0) {{
            hops = [(-1, 0)],
            channel = 0
        }}
    }}
    compute i32 i, i32 j in [0, 0] {{
        // Test foreach
        await foreach i32 k, f32 x in [0:K], receive(stream) {{
            a32[k] = a32[k] + x
        }}
        await foreach i32 k#1, f16 x#1 in [0:K], receive(stream) {{
            a32[k#1] = a32[k#1] + x#1
        }}
        await foreach i32 k#2, i32 x#2 in [0:K], receive(stream) {{
            a16[k#2] = x#2
        }}
        await foreach i32 k#3, f32 x#3 in [0:K], receive(stream) {{
            a32[k#3] = x#3
        }}
        await foreach i32 k#4, f32 x#4 in [0:K], receive(stream) {{
            a32[k#4] = fmac(a32[k#4], x#4, localval32)
        }}
        await foreach i32 k#5, f16 x#5 in [0:K], receive(stream) {{
            a32[k#5] = fmac(a32[k#5], x#5, localval32)
        }}
        await foreach i32 k#6, f32 x#6 in [0:K], receive(stream) {{
            a32[k#6] = fmac(a32[k#6], x#6, localval16)
        }}
        // Test map
        await map i32 m in [0:K] {{
            a32[m] = a32[m] + a16[m]
        }}
        await map i32 k#7 in [0:K] {{
            a32[k#7] = a32[k#7]
        }}
        await map i32 k#8 in [0:K] {{
            a32[k#8] = localval16
        }}
    }}
}}""")
    place, dataflow, compute = kernel.body
    dtypes = s2c._collect_identifier_types(PEBlock(place, dataflow, compute), [])
    assert len(compute.statements) == 10
    assert dsd_ops.get_dsd_op(dtypes, compute.statements[0]) == "@fadds"
    assert dsd_ops.get_dsd_op(dtypes, compute.statements[1]) == "@faddhs"
    assert dsd_ops.get_dsd_op(dtypes, compute.statements[2]) is None
    assert dsd_ops.get_dsd_op(dtypes, compute.statements[3]) == "@fmovs"
    assert dsd_ops.get_dsd_op(dtypes, compute.statements[4]) == "@fmacs"
    assert dsd_ops.get_dsd_op(dtypes, compute.statements[5]) is None
    assert dsd_ops.get_dsd_op(dtypes, compute.statements[6]) == "@fmachs"
    assert dsd_ops.get_dsd_op(dtypes, compute.statements[7]) == "@faddhs"
    assert dsd_ops.get_dsd_op(dtypes, compute.statements[8]) == "@fmovs"
    assert dsd_ops.get_dsd_op(dtypes, compute.statements[9]) == "@fh2s"


def test_dsd_op_detection_constant_folding():
    kernel = parser.parse_string(code=f"""
kernel @tester<K> () {{

    place i16 i, i16 j in [0, 0] {{
        f32[K] a32
    }}
    dataflow i16 i, i16 j in [0, 0] {{
    }}
    compute i32 i, i32 j in [0, 0] {{
        await map i32 k#7 in [0:80] {{
            a32[k#7] = (-4.0 * a32[k#7])
        }}
    }}
}}""")
    kernel = passes.concretize_parameters(kernel, K=80)
    place, dataflow, compute = kernel.body
    dtypes = s2c._collect_identifier_types(PEBlock(place, dataflow, compute), [])
    assert dsd_ops.get_dsd_op(dtypes, compute.statements[0]) is None

    dtypes = s2c._collect_identifier_types(PEBlock(place, dataflow, compute), [])
    kernel = passes.constexpr_propagation(kernel)
    assert dsd_ops.get_dsd_op(dtypes, compute.statements[0]) == "@fmuls"


def test_dsd_op_detection_receive_op_send():
    kernel = parser.parse_string(code="""
kernel @tester<K>() {

    place i16 i, i16 j in [0, 0] {
        f32[K] a32
    }
    dataflow i16 i, i16 j in [0, 0] {
        stream<f32> red = relative_stream(-1, 0) {
            hops = [(-1, 0)],
            channel = 0
        }
        stream<f32> blue = relative_stream(1, 0) {
            hops = [(1, 0)],
            channel = 1
        }
    }
    compute i16 i, i16 j in [0, 0] {
        await foreach i16 k, f32 x in [0:K], receive(red) {
            a32[k] = a32[k] + x
            await send(a32[k], blue)
        }
    }
}""")
    kernel = passes.concretize_parameters(kernel, K=8)
    place, dataflow, compute = kernel.body
    dtypes = s2c._collect_identifier_types(PEBlock(place, dataflow, compute), [])

    assert dsd_ops.get_dsd_op(dtypes, compute.statements[0]) == "@fadds"
    dsd_stmt = dsd_ops.get_dsd_statement(dtypes, compute.statements[0])
    assert dsd_stmt is not None
    assert dsd_stmt.destination.as_ir() == 'blue'


if __name__ == '__main__':
    test_dsd_op_detection()
    test_dsd_op_detection_constant_folding()
    test_dsd_op_detection_receive_op_send()
