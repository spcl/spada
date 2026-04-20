import pytest
from spada.lowering import spatial_ir_to_csl as s2c
from spada.syntax.spatial_ir import analysis, parser
from spada.syntax.spatial_ir.canonicalization import PEBlock
from spada.syntax.csl import tasks as tdag


def _create_tasks(peblock: PEBlock):
    dtypes = s2c._collect_identifier_types(peblock, [])
    completion_dag = analysis.to_completion_dag(peblock.compute)
    return tdag.create_csl_tasks(completion_dag, peblock.compute, dtypes)


@pytest.mark.parametrize('dsd_op', (False, True))
def test_tasks_with_dsd_ops(dsd_op: bool):
    # An f32+i16 operation cannot be generated as a DSD operation
    dtype = 'f32' if dsd_op else 'i16'
    kernel = parser.parse_string(code=f"""
kernel @two_phase<K> (stream<f32>[4] readonly in,
                          stream<f32> readonly out ) {{

    place i16 i, i16 j in [0, 0] {{
        f32[K] a
    }}
    dataflow i32 i, i32 j in [0, 0] {{
        stream<{dtype}> hop1 = relative_stream(-1, 0) {{
            hops = [(-1, 0)],
            channel = 0
        }}
        stream<{dtype}> hop2 = relative_stream(-2, 0) {{
            hops = [(-1, 0), (-1, 0)],
            channel = 0
        }}
    }}
    compute i32 i, i32 j in [0, 0] {{
        await receive(a, in[i])
        await foreach i32 k, {dtype} x in [0:K], receive(hop1) {{
            a[k] = a[k] + x
        }}
        await foreach i32 k#1, {dtype} x#1 in [0:K], receive(hop2) {{
            a[k#1] = a[k#1] + x#1
        }}
        await send(a, out)
    }}
}}""")
    place, dataflow, compute = kernel.body
    block = PEBlock(place, dataflow, compute)
    tasks = _create_tasks(block)
    assert len(tasks) == 4
    if dsd_op:
        assert tasks[1].task_type == 'local'
        assert tasks[2].task_type == 'local'
    else:
        assert tasks[1].task_type == 'data'
        assert tasks[2].task_type == 'data'


def test_wait_tree():
    # A subset of the full code for testing
    kernel = parser.parse_string(code="""
kernel @reduce<N>(stream<f32>[N] readonly inp, stream<f32> writeonly out) {
    place u16 i, u16 j in [1:N-1, 0:1] {
        f32 local
        f32 rcv_val1
        f32 rcv_val2
        f32 rcv_val3
        f32 rcv_val4
        f32 rcv_val5
    }
    dataflow u16 i, u16 j in [1:N-1, 0:1] {
        stream<f32> westwards = relative_stream(-1, 0) {
            hops = [(-1, 0)],
            channel = 0
        }
    }
    compute u16 i, u16 j in [1:N-1, 0:1] {
        completion c1 = receive(local, inp[i])
        completion c2 = receive(rcv_val1, westwards)
        completion c3 = receive(rcv_val2, westwards)
        completion c4 = receive(rcv_val3, westwards)
        completion c5 = receive(rcv_val4, westwards)
        awaitall
        await send(rcv_val1, westwards)
    }
}""")
    place, dataflow, compute = kernel.body
    block = PEBlock(place, dataflow, compute)
    tasks = _create_tasks(block)
    # The receives should exist in the first task, followed by a tree of waits, followed by the last send
    assert len(tasks) in (5, 6)
    assert len(tasks[0].statements) == 5
    assert len(tasks[-1].statements) == 1


def test_tasks_with_relay_dsd_op():
    kernel = parser.parse_string(code="""
kernel @relay<>() {

    place i16 i, i16 j in [0, 0] {
        f32[8] a
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
        await foreach i16 k, f32 x in [0:8], receive(red) {
            a[k] = a[k] + x
            await send(a[k], blue)
        }
    }
}""")
    place, dataflow, compute = kernel.body
    block = PEBlock(place, dataflow, compute)
    tasks = _create_tasks(block)

    assert len(tasks) == 1
    assert tasks[0].task_type == 'local'


@pytest.mark.parametrize('async_first_task', (False, True))
def test_activate_unblock(async_first_task: bool):
    if async_first_task:
        first_task_code = """
        completion c1 = receive(local, inp[i])
        completion c2 = receive(rcv_val, westwards)
        await c1
        await c2"""
    else:
        first_task_code = """
        await receive(local, inp[i])
        await receive(rcv_val, westwards)"""

    # A subset of the full code for testing
    kernel = parser.parse_string(code=f"""
kernel @reduce<N>(stream<f32>[N] readonly inp, stream<f32> writeonly out) {{
    place u16 i, u16 j in [1:N-1, 0:1] {{
        f32 local
        f32 rcv_val
    }}
    dataflow u16 i, u16 j in [1:N-1, 0:1] {{
        stream<f32> westwards = relative_stream(-1, 0) {{
            hops = [(-1, 0)],
            channel = 0
        }}
    }}
    compute u16 i, u16 j in [1:N-1, 0:1] {{
        {first_task_code}
        rcv_val = rcv_val + local
        await send(rcv_val, westwards)
    }}
}}""")
    place, dataflow, compute = kernel.body
    block = PEBlock(place, dataflow, compute)
    tasks = _create_tasks(block)
    # Tasks should have the first two receives and the two following operations blocked by both an
    # @activate operation and an @unblock operation
    if async_first_task:
        assert len(tasks) in (2, 3)
        assert {tasks[0].outgoing[0][1],
                tasks[0].outgoing[1][1]} == {tdag.InterTaskEdge.ACTIVATE, tdag.InterTaskEdge.UNBLOCK}
    else:
        assert len(tasks) == 3


def test_await_sequence():
    kernel = parser.parse_string(code=f"""
kernel @test<N>(stream<f32>[N-2] readonly inp, stream<f32> writeonly out) {{
    place u16 i, u16 j in [1:N-1, 0:1] {{
        f32 local
    }}
    dataflow u16 i, u16 j in [1:N-1, 0:1] {{
    }}
    compute u16 i, u16 j in [1:N-1, 0:1] {{
        completion c1 = receive(local, inp[i])
        await c1
        completion c2 = receive(local, inp[i])
        await c2
        completion c3 = send(local, out[i])
        await c3
    }}
}}""")
    place, dataflow, compute = kernel.body
    block = PEBlock(place, dataflow, compute)
    tasks = _create_tasks(block)
    assert len(tasks) == 3


def test_await_sequence_with_async():
    kernel = parser.parse_string(code="""
    kernel @test_nested_async<N>(stream<f32, 1>[N] readonly a, stream<f32, 1>[N] readonly b,
                                 stream<f32, 1>[N] writeonly result) {
        place u16 i, u16 j in [0:N, 0:1] {
            f32 val_a;
            f32 val_b;
            f32 intermediate;
            f32 final_result;
        }
        dataflow u16 i, u16 j in [0:N, 0:1] {}
        compute u16 i, u16 j in [0:N, 0:1] {
            await receive(val_a, a[i]);
            await receive(val_b, b[i]);
            
            completion computation = async {
                intermediate = fmac(val_a, val_b, 1.0);
                final_result = intermediate if intermediate > 0.0 else 0.0;
            };
            
            await computation;
            await send(final_result, result[i]);
        }
    }""")
    place, dataflow, compute = kernel.body
    block = PEBlock(place, dataflow, compute)
    tasks = _create_tasks(block)
    assert len(tasks) == 3


def test_data_local_task_combo():
    kernel = parser.parse_string(code=f'''
    kernel @test_foreach_range<N>(stream<f32, 4>[N] readonly input) {{
        place u16 i, u16 j in [0:N, 0:1] {{
            f32 accumulator;
        }}
        dataflow u16 i, u16 j in [0:N, 0:1] {{
        }}
        compute u16 i, u16 j in [0:N, 0:1] {{
            accumulator = 0.0;

            await foreach u16 k, f32 value in [0:4], receive(input[i]) {{
                accumulator = accumulator + value;
            }};
        }}
    }}
    ''')
    place, dataflow, compute = kernel.body
    block = PEBlock(place, dataflow, compute)
    tasks = _create_tasks(block)
    assert len(tasks) == 2
    assert tasks[0].task_type == 'local'
    assert tasks[1].task_type == 'data'
    assert len(tasks[0].statements) == 1  # Initialization
    assert tasks[0].outgoing[0][1] == tdag.InterTaskEdge.UNBLOCK


if __name__ == '__main__':
    test_tasks_with_dsd_ops(False)
    test_tasks_with_dsd_ops(True)
    test_wait_tree()
    test_activate_unblock(False)
    test_activate_unblock(True)
    test_await_sequence()
    test_await_sequence_with_async()
    test_data_local_task_combo()
