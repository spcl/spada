"""
Tests for the stream lifetime passes (implicit closes, bound verification, use-after-close,
channel conflicts, and close elision).
"""
import os

import pytest

from spada.lowering import spatial_ir_to_csl as s2c
from spada.syntax.spatial_ir import canonicalization, irnodes as spir, parser, passes, stream_lifetime

SAMPLES = os.path.join(os.path.dirname(__file__), 'samples')


def _canonicalized(code: str, **parameters) -> spir.Kernel:
    kernel = parser.parse_string(code, 'test.sptl')
    if parameters:
        kernel = passes.concretize_parameters(kernel, **parameters)
    kernel = passes.constexpr_propagation(kernel)
    return s2c.canonicalize_kernel(kernel)


def _rectangles(code: str, **parameters):
    kernel = _canonicalized(code, **parameters)
    return canonicalization.consolidate_rectangles_to_equivalence_classes(kernel)


def _rectangles_from_file(filename: str, **parameters):
    with open(os.path.join(SAMPLES, filename)) as fp:
        return _rectangles(fp.read(), **parameters)


def _closed_streams(compute: spir.ComputeBlock) -> list[str]:
    """
    Names of the streams closed in a compute block, without the version suffix that
    ``inline_phases`` adds when it freshens colliding stream names per rectangle.
    """
    return [
        statement.stream_name.as_ir().split('#')[0] for statement in compute.statements
        if isinstance(statement, spir.CloseStatement)
    ]


###
# insert_implicit_closes
###

_TWO_STREAM_KERNEL = """
kernel @test<K>(stream<f32, K>[2] readonly inp) {
    place u16 i, u16 j in [0:2, 0:1] {
        f32[K] a
    }
    dataflow u16 i, u16 j in [0:2, 0:1] {
        stream<f32> eastwards = relative_stream(1, 0) {
            hops = [(1, 0)],
            channel = 0
        }
    }
    compute u16 i, u16 j in [0:1, 0:1] {
        await receive(a, inp[i])
        await send(a, eastwards)
    }
    compute u16 i, u16 j in [1:2, 0:1] {
        await foreach i32 k, f32 x in [0:K], receive(eastwards) {
            a[k] = x
        }
    }
}
"""


def test_implicit_close_is_inserted_for_every_participant():
    """Both the sending and the receiving PE close the stream: closing is collective."""
    rects = _rectangles(_TWO_STREAM_KERNEL, K=4)
    assert len(rects) == 2
    for rect in rects:
        assert _closed_streams(rect.metadata.compute) == ['eastwards']


def test_implicit_close_comes_after_the_phase_barrier():
    """
    The close must follow the implicit awaits: they may be waiting on operations that are still
    using the stream.
    """
    rects = _rectangles(_TWO_STREAM_KERNEL, K=4)
    statements = rects[0].metadata.compute.statements
    close_index = next(i for i, s in enumerate(statements) if isinstance(s, spir.CloseStatement))
    barrier_index = next(i for i, s in enumerate(statements) if isinstance(s, spir.AwaitAllStatement))
    assert barrier_index < close_index


def test_phase_barrier_is_not_duplicated():
    """
    ``insert_implicit_closes`` ends a phase with ``awaitall`` + closes, and ``inline_phases`` would
    otherwise append a second barrier right behind it. The closes are awaited, so nothing is
    outstanding and the second barrier is redundant.
    """
    code = """
    kernel @test<K>(stream<f32, K>[2] readonly inp) {
        place u16 i, u16 j in [0:2, 0:1] { f32[K] a }
        phase {
            dataflow u16 i, u16 j in [0:2, 0:1] {
                stream<f32> e1 = relative_stream(1, 0) { hops = [(1, 0)], channel = 0 }
            }
            compute u16 i, u16 j in [0:1, 0:1] { await send(a, e1) }
        }
        phase {
            dataflow u16 i, u16 j in [0:2, 0:1] {
                stream<f32> e2 = relative_stream(1, 0) { hops = [(1, 0)], channel = 1 }
            }
            compute u16 i, u16 j in [0:1, 0:1] { await send(a, e2) }
        }
    }
    """
    rects = _rectangles(code, K=4)
    sender = next(rect for rect in rects if rect.x_range[0] == 0)
    statements = sender.metadata.compute.statements
    barriers = [i for i, s in enumerate(statements) if isinstance(s, spir.AwaitAllStatement)]
    assert len(barriers) == 2, sender.metadata.compute.as_ir()
    assert all(second - first > 1 for first, second in zip(barriers, barriers[1:]))


def test_implicit_close_is_not_duplicated():
    """A stream the user already closed is not closed a second time."""
    code = _TWO_STREAM_KERNEL.replace('await send(a, eastwards)', 'await send(a, eastwards)\n'
                                      '        await eastwards.close()')
    rects = _rectangles(code, K=4)
    sender = next(rect for rect in rects if rect.x_range[0] == 0)
    assert _closed_streams(sender.metadata.compute) == ['eastwards']


def test_stream_used_in_two_phases_is_closed_after_its_last_use():
    """
    A stream declared at kernel level stays in scope across phases, so it is closed only in the
    phase that uses it last.
    """
    code = """
    kernel @test<K>(stream<f32, K>[2] readonly inp) {
        place u16 i, u16 j in [0:2, 0:1] {
            f32[K] a
        }
        dataflow u16 i, u16 j in [0:2, 0:1] {
            stream<f32> eastwards = relative_stream(1, 0) {
                hops = [(1, 0)],
                channel = 0
            }
        }
        phase {
            compute u16 i, u16 j in [0:1, 0:1] {
                await receive(a, inp[i])
                await send(a, eastwards)
            }
        }
        phase {
            compute u16 i, u16 j in [0:1, 0:1] {
                await send(a, eastwards)
            }
        }
    }
    """
    rects = _rectangles(code, K=4)
    sender = next(rect for rect in rects if rect.x_range[0] == 0)
    # Exactly one close, and it comes after the second send
    statements = sender.metadata.compute.statements
    closes = [i for i, s in enumerate(statements) if isinstance(s, spir.CloseStatement)]
    sends = [i for i, s in enumerate(statements) if isinstance(s, spir.SendStatement)]
    assert len(closes) == 1
    assert closes[0] > max(sends)


###
# verify_stream_bounds
###


def _bounded_kernel(bound: str, elements: str) -> str:
    return f"""
    kernel @test<K>(stream<f32, K>[2] readonly inp) {{
        place u16 i, u16 j in [0:2, 0:1] {{
            f32[{elements}] a
        }}
        dataflow u16 i, u16 j in [0:2, 0:1] {{
            stream<f32, {bound}> eastwards = relative_stream(1, 0) {{
                hops = [(1, 0)],
                channel = 0
            }}
        }}
        compute u16 i, u16 j in [0:1, 0:1] {{
            await send(a, eastwards)
        }}
        compute u16 i, u16 j in [1:2, 0:1] {{
            await foreach i32 k, f32 x in [0:{elements}], receive(eastwards) {{
                a[k] = x
            }}
        }}
    }}
    """


def test_matching_bound_is_accepted():
    stream_lifetime.verify_stream_bounds(_rectangles(_bounded_kernel('4', '4'), K=4))


def test_mismatched_bound_is_rejected():
    with pytest.raises(SyntaxError, match='bound 8, but 4 element'):
        stream_lifetime.verify_stream_bounds(_rectangles(_bounded_kernel('8', '4'), K=4))


def test_unanalyzable_bound_is_silently_accepted():
    """A ``foreach`` without a range receives until the sender is done, so nothing can be checked."""
    code = """
    kernel @test<K>(stream<f32, K>[2] readonly inp) {
        place u16 i, u16 j in [0:2, 0:1] {
            f32[K] a
        }
        dataflow u16 i, u16 j in [0:2, 0:1] {
            stream<f32, 12345> eastwards = relative_stream(1, 0) {
                hops = [(1, 0)],
                channel = 0
            }
        }
        compute u16 i, u16 j in [1:2, 0:1] {
            await foreach f32 x in receive(eastwards) {
                a[0] = x
            }
        }
    }
    """
    stream_lifetime.verify_stream_bounds(_rectangles(code, K=4))


def test_bound_counts_enclosing_loop_trips():
    """A send inside a ``for`` loop transfers ``trips * size`` elements."""
    code = """
    kernel @test<K>(stream<f32, K>[2] readonly inp) {
        place u16 i, u16 j in [0:2, 0:1] {
            f32[4] a
        }
        dataflow u16 i, u16 j in [0:2, 0:1] {
            stream<f32, 8> eastwards = relative_stream(1, 0) {
                hops = [(1, 0)],
                channel = 0
            }
        }
        compute u16 i, u16 j in [0:1, 0:1] {
            for i32 t in [0:3] {
                await send(a, eastwards)
            }
        }
    }
    """
    with pytest.raises(SyntaxError, match='bound 8, but 12 element'):
        stream_lifetime.verify_stream_bounds(_rectangles(code, K=4))


###
# check_use_after_close
###


def _use_after_close_kernel(body: str) -> str:
    return f"""
    kernel @test<K>(stream<f32, K>[2] readonly inp) {{
        place u16 i, u16 j in [0:2, 0:1] {{
            f32[K] a
        }}
        dataflow u16 i, u16 j in [0:2, 0:1] {{
            stream<f32> eastwards = relative_stream(1, 0) {{
                hops = [(1, 0)],
                channel = 0
            }}
        }}
        compute u16 i, u16 j in [0:1, 0:1] {{
{body}
        }}
        compute u16 i, u16 j in [1:2, 0:1] {{
            await foreach i32 k, f32 x in [0:K], receive(eastwards) {{
                a[k] = x
            }}
        }}
    }}
    """


def test_send_after_close_is_rejected():
    code = _use_after_close_kernel("""
            await send(a, eastwards)
            await eastwards.close()
            await send(a, eastwards)""")
    with pytest.raises(SyntaxError, match='used in a `send` after it was closed'):
        stream_lifetime.check_use_after_close(_rectangles(code, K=4))


def test_receive_after_close_is_rejected():
    code = _use_after_close_kernel("""
            await send(a, eastwards)
            await eastwards.close()
            await foreach i32 k, f32 x in [0:K], receive(eastwards) {
                a[k] = x
            }""")
    with pytest.raises(SyntaxError, match='used in a `receive` after it was closed'):
        stream_lifetime.check_use_after_close(_rectangles(code, K=4))


def test_double_close_is_rejected():
    code = _use_after_close_kernel("""
            await send(a, eastwards)
            await eastwards.close()
            await eastwards.close()""")
    with pytest.raises(SyntaxError, match='closed again after it was closed'):
        stream_lifetime.check_use_after_close(_rectangles(code, K=4))


def test_close_after_last_use_is_accepted():
    code = _use_after_close_kernel("""
            await send(a, eastwards)
            await eastwards.close()""")
    stream_lifetime.check_use_after_close(_rectangles(code, K=4))


###
# check_channel_conflicts
###

_SHARED_CHANNEL_KERNEL = """
kernel @test<K>(stream<f32, K>[4] readonly inp) {{
    place i16 i, i16 j in [0:4, 0:1] {{
        f32[K] a
    }}
    dataflow i32 i, i32 j in [0:4, 0:1] {{
        stream<f32> hop1 = relative_stream(-1, 0) {{
            hops = [(-1, 0)],
            channel = 0
        }}
        stream<f32> hop2 = relative_stream(-2, 0) {{
            hops = [(-1, 0), (-1, 0)],
            channel = {second_channel}
        }}
    }}
    compute i32 i, i32 j in [3:4, 0:1] {{
        await receive(a, inp[i])
        await send(a, hop1)
    }}
    compute i32 i, i32 j in [2:3, 0:1] {{
        await foreach i32 k, f32 x in [0:K], receive(hop1) {{
            a[k] = x
        }}
        {close}
        await send(a, hop2)
    }}
    compute i32 i, i32 j in [0:1, 0:1] {{
        await foreach i32 k, f32 x in [0:K], receive(hop2) {{
            a[k] = x
        }}
    }}
}}
"""


def test_unordered_channel_reuse_is_rejected():
    code = _SHARED_CHANNEL_KERNEL.format(second_channel=0, close='')
    with pytest.raises(SyntaxError, match="both use channel 0 and share PE"):
        stream_lifetime.check_channel_conflicts(_rectangles(code, K=4))


def test_channel_reuse_after_close_is_accepted():
    code = _SHARED_CHANNEL_KERNEL.format(second_channel=0, close='await hop1.close()')
    stream_lifetime.check_channel_conflicts(_rectangles(code, K=4))


def test_distinct_channels_never_conflict():
    code = _SHARED_CHANNEL_KERNEL.format(second_channel=1, close='')
    stream_lifetime.check_channel_conflicts(_rectangles(code, K=4))


def test_channel_reuse_across_phases_is_accepted():
    """``two_phase.sptl`` reuses channel 0 in two phases; the phase barrier orders the streams."""
    rects = _rectangles_from_file('two_phase.sptl', K=4)
    stream_lifetime.check_channel_conflicts(rects)


def test_streams_on_one_channel_with_disjoint_paths_are_accepted():
    """Two streams may share a channel without ever meeting on a PE."""
    code = """
    kernel @test<K>(stream<f32, K>[4] readonly inp) {
        place i16 i, i16 j in [0:4, 0:1] {
            f32[K] a
        }
        dataflow i32 i, i32 j in [0:2, 0:1] {
            stream<f32> left = relative_stream(1, 0) {
                hops = [(1, 0)],
                channel = 0
            }
        }
        dataflow i32 i, i32 j in [2:4, 0:1] {
            stream<f32> right = relative_stream(1, 0) {
                hops = [(1, 0)],
                channel = 0
            }
        }
        compute i32 i, i32 j in [0:1, 0:1] {
            await receive(a, inp[i])
            await send(a, left)
        }
        compute i32 i, i32 j in [1:2, 0:1] {
            await foreach i32 k, f32 x in [0:K], receive(left) {
                a[k] = x
            }
        }
        compute i32 i, i32 j in [2:3, 0:1] {
            await receive(a, inp[i])
            await send(a, right)
        }
        compute i32 i, i32 j in [3:4, 0:1] {
            await foreach i32 k, f32 x in [0:K], receive(right) {
                a[k] = x
            }
        }
    }
    """
    stream_lifetime.check_channel_conflicts(_rectangles(code, K=4))


def test_systolic_forwarding_is_not_a_channel_conflict():
    """
    One stream that is received and then forwarded on the same PE needs two router configurations,
    but is ordered by local order and therefore not a conflict.
    """
    rects = _rectangles_from_file('multihop.sptl', K=4)
    stream_lifetime.check_channel_conflicts(rects)


###
# elide_redundant_closes
###


def test_closes_are_elided_when_the_channel_is_never_reused():
    rects = _rectangles(_TWO_STREAM_KERNEL, K=4)
    assert stream_lifetime.elide_redundant_closes(rects) == 2
    assert all(not _closed_streams(rect.metadata.compute) for rect in rects)


def test_closes_are_kept_when_the_channel_is_reused():
    code = _SHARED_CHANNEL_KERNEL.format(second_channel=0, close='await hop1.close()')
    rects = _rectangles(code, K=4)
    assert stream_lifetime.elide_redundant_closes(rects) == 0
    assert any('hop1' in _closed_streams(rect.metadata.compute) for rect in rects)


def test_auto_channel_kernels_emit_no_closes():
    """
    Kernels produced by the GT4Py path use ``auto`` channels, one per declaration, so every close is
    elided and code generation is unaffected by this feature.
    """
    rects = _rectangles_from_file('two_phase_unrouted.sptl', K=4)
    stream_lifetime.elide_redundant_closes(rects)
    assert all(not _closed_streams(rect.metadata.compute) for rect in rects)


###
# assign_fabric_queues
###


def test_sequential_spans_share_a_queue():
    """A queue is remapped only when the previous color's span on this PE has ended."""
    assigned = stream_lifetime.assign_fabric_queues(
        {'channel 0': (0, 2), 'channel 1': (3, 5)}, [0, 1],
        kind='input', architecture='wse2', location='PE (0, 0)')
    assert assigned == {'channel 0': 0, 'channel 1': 0}


def test_overlapping_spans_take_distinct_queues():
    assigned = stream_lifetime.assign_fabric_queues(
        {'channel 0': (0, 4), 'channel 1': (2, 6)}, [0, 1],
        kind='input', architecture='wse2', location='PE (0, 0)')
    assert assigned['channel 0'] != assigned['channel 1']


def test_a_channel_keeps_its_queue_across_a_gap():
    """
    Wavelets of a reused color can still arrive between its epochs, so a different color that
    sits in the gap cannot steal the queue.
    """
    assigned = stream_lifetime.assign_fabric_queues(
        {'channel 0': (0, 10), 'channel 1': (3, 5)}, [0, 1],
        kind='input', architecture='wse2', location='PE (0, 0)')
    assert assigned['channel 0'] != assigned['channel 1']


def test_three_overlapping_spans_exhaust_two_queues():
    with pytest.raises(SyntaxError, match='concurrent input queues'):
        stream_lifetime.assign_fabric_queues(
            {'channel 0': (0, 10), 'channel 1': (2, 8), 'channel 2': (4, 6)}, [0, 1],
            kind='input', architecture='wse2', location='PE (0, 0)')


def test_exclusive_keys_do_not_share_a_queue_across_a_gap():
    """WSE-3 data-task colors cannot share a queue even when their spans are disjoint."""
    assigned = stream_lifetime.assign_fabric_queues(
        {'channel 0': (0, 2), 'channel 1': (3, 5)}, [2, 3],
        kind='input', architecture='wse3', location='PE (0, 0)',
        exclusive_keys=frozenset({'channel 0', 'channel 1'}))
    assert assigned['channel 0'] != assigned['channel 1']


def test_exclusive_keys_exhaust_queues_when_too_many_data_tasks():
    with pytest.raises(SyntaxError, match='data-task ID'):
        stream_lifetime.assign_fabric_queues(
            {'channel 0': (0, 1), 'channel 1': (2, 3), 'channel 2': (4, 5)}, [2, 3],
            kind='input', architecture='wse3', location='PE (0, 0)',
            exclusive_keys=frozenset({'channel 0', 'channel 1', 'channel 2'}))


def test_microthreads_are_shared_across_directions_when_transfers_do_not_overlap():
    """A microthread is held only while a transfer is in flight, so turns may be taken."""
    assigned = stream_lifetime.assign_microthreads(
        {'in channel 0': [(0, 2)], 'out channel 1': [(3, 5)]}, [2, 3], location='PE (0, 0)')
    assert assigned == {'in channel 0': 2, 'out channel 1': 2}


def test_a_receive_and_a_send_in_flight_together_get_distinct_microthreads():
    assigned = stream_lifetime.assign_microthreads(
        {'in channel 0': [(0, 4)], 'out channel 0': [(0, 4)]}, [2, 3], location='PE (0, 0)')
    assert assigned['in channel 0'] != assigned['out channel 0']


def test_a_group_that_comes_back_after_a_gap_does_not_hold_its_microthread_across_it():
    """
    A channel used again much later is not in flight in between, unlike a fabric queue, which stays
    bound to it. The group that runs in the gap may take the same microthread.
    """
    assigned = stream_lifetime.assign_microthreads(
        {'in channel 0': [(0, 1), (8, 9)], 'out channel 1': [(4, 5)]}, [2, 3], location='PE (0, 0)')
    assert assigned == {'in channel 0': 2, 'out channel 1': 2}


def test_microthreads_run_out_when_too_many_transfers_overlap():
    with pytest.raises(SyntaxError, match='concurrent microthreads'):
        stream_lifetime.assign_microthreads(
            {'in channel 0': [(0, 10)], 'out channel 0': [(2, 8)], 'out channel 1': [(4, 6)]},
            [2, 3], location='PE (0, 0)')


def test_microthreads_are_left_to_the_hardware_when_the_target_cannot_name_them():
    assert stream_lifetime.assign_microthreads({'in channel 0': [(0, 2)]}, [],
                                               location='PE (0, 0)') == {}


if __name__ == '__main__':
    pytest.main([__file__])
