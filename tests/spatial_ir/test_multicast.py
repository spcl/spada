"""
Unit tests for the multicast stream feature (relative_stream(dx, [start:stop])).

Covers:
  - Parser: new grammar and IR node construction
  - Roundtrip: as_ir() stability
  - Canonicalization: default routing injection, validation errors
  - CSL lowering: correct routing instructions in generated layout code
"""
import os
import pytest
from spatialstencil.syntax.spatial_ir import irnodes as spir, parser, passes, canonicalization
from spatialstencil.lowering.spatial_ir_to_csl import lower_spatial_ir_to_csl

_TESTING_DIR = os.path.join(os.path.dirname(__file__), 'samples')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_concretize(code: str, **params):
    kernel = parser.parse_string(code)
    if params:
        kernel = passes.concretize_parameters(kernel, **params)
        kernel = passes.constexpr_propagation(kernel)
    return kernel


def _layout_code(csl_files) -> str:
    """Return the string of the layout CSL file (contains @set_color_config)."""
    for f in csl_files:
        if '@set_color_config' in f.code or 'layout' in f.filename:
            return f.code
    return ''.join(f.code for f in csl_files)


def _multicast_kernel(K: int | str = 'K', channel: int = 0) -> str:
    return f"""
    kernel @test<K>() {{
        place u16 i, u16 j in [0:1, 0:K] {{
            f32[1] val
        }}
        dataflow u16 i, u16 j in [0:1, 0:K] {{
            stream<f32> s = relative_stream(0, [1:K]) {{
                hops = auto,
                channel = {channel}
            }}
        }}
        compute u16 i, u16 j in [0:1, 0:1] {{
            await send(val, s)
        }}
        compute u16 i, u16 j in [0:1, 1:K] {{
            await receive(val, s)
        }}
    }}
    """


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------

def test_multicast_dy_parsed_as_multicast_node():
    """relative_stream(0, [1:K]) produces a MulticastRangeStreamDeclaration."""
    kernel = parser.parse_string(_multicast_kernel())
    df = next(b for b in kernel.body if isinstance(b, spir.DataflowBlock))
    decl = df.statements[0]
    assert isinstance(decl.stream, spir.MulticastRangeStreamDeclaration)


def test_multicast_dy_range_values():
    """dy is a RangeExpression with start=1, stop=K; dx is a scalar 0."""
    kernel = _parse_concretize(_multicast_kernel(), K=5)
    df = next(b for b in kernel.body if isinstance(b, spir.DataflowBlock))
    decl = df.statements[0]
    s = decl.stream
    assert isinstance(s.dy, spir.RangeExpression)
    assert isinstance(s.dx, spir.Expression)
    assert s.dx.eval() == 0
    assert s.dy.start.eval() == 1
    assert s.dy.stop.eval() == 5


def test_multicast_dx_range():
    """relative_stream([1:K], 0) also produces a MulticastRangeStreamDeclaration (x-axis)."""
    code = """
    kernel @test<K>() {
        dataflow u16 i, u16 j in [0:K, 0:1] {
            stream<f32> s = relative_stream([1:K], 0) {
                hops = auto, channel = 0
            }
        }
        compute u16 i, u16 j in [0:1, 0:1] { await send(val, s) }
        compute u16 i, u16 j in [1:K, 0:1] { await receive(val, s) }
    }
    """
    kernel = parser.parse_string(code)
    df = next(b for b in kernel.body if isinstance(b, spir.DataflowBlock))
    decl = df.statements[0]
    assert isinstance(decl.stream, spir.MulticastRangeStreamDeclaration)
    assert decl.stream.multicast_axis == 'x'
    assert isinstance(decl.stream.dx, spir.RangeExpression)
    assert isinstance(decl.stream.dy, spir.Expression)


def test_regular_relative_stream_unchanged():
    """relative_stream(1, 0) still produces RelativeStreamDeclaration (regression guard)."""
    code = """
    kernel @test<>() {
        dataflow u16 i, u16 j in [0:4, 0:1] {
            stream<f32> s = relative_stream(1, 0) { hops = [(1, 0)], channel = 0 }
        }
        compute u16 i, u16 j in [0:3, 0:1] { await send(val, s) }
        compute u16 i, u16 j in [1:4, 0:1] { await receive(val, s) }
    }
    """
    kernel = parser.parse_string(code)
    df = next(b for b in kernel.body if isinstance(b, spir.DataflowBlock))
    assert isinstance(df.statements[0].stream, spir.RelativeStreamDeclaration)


def test_multicast_roundtrip():
    """as_ir() output re-parses to the same as_ir() (roundtrip stability)."""
    file = os.path.join(_TESTING_DIR, 'multicast_generalized_y.sptl')
    kernel = parser.parse_file(file)
    ir1 = kernel.as_ir()
    kernel2 = parser.parse_string(ir1)
    ir2 = kernel2.as_ir()
    assert ir1 == ir2


def test_multicast_properties():
    """MulticastRangeStreamDeclaration properties return correct axis / range / fixed_offset."""
    kernel = _parse_concretize(_multicast_kernel(), K=4)
    df = next(b for b in kernel.body if isinstance(b, spir.DataflowBlock))
    s = df.statements[0].stream
    assert s.multicast_axis == 'y'
    assert s.multicast_range is s.dy
    assert s.fixed_offset is s.dx


# ---------------------------------------------------------------------------
# Canonicalization tests
# ---------------------------------------------------------------------------

def test_multicast_auto_routing_injected():
    """Without explicit routing, _AutoHopResolver injects a default RoutingDeclaration."""
    code = """
    kernel @test<K>() {
        dataflow u16 i, u16 j in [0:1, 0:K] {
            stream<f32> s = relative_stream(0, [1:K])
        }
        compute u16 i, u16 j in [0:1, 0:1] { await send(val, s) }
        compute u16 i, u16 j in [0:1, 1:K] { await receive(val, s) }
    }
    """
    kernel = _parse_concretize(code, K=4)
    kernel = canonicalization.resolve_auto_hops(kernel)
    df = next(b for b in kernel.body if isinstance(b, spir.DataflowBlock))
    routing = df.statements[0].stream.routing
    assert routing is not None
    assert routing.hops == []


def test_multicast_explicit_channel_preserved():
    """Explicit channel is preserved through canonicalization."""
    kernel = _parse_concretize(_multicast_kernel(channel=3), K=4)
    kernel = canonicalization.resolve_auto_hops(kernel)
    df = next(b for b in kernel.body if isinstance(b, spir.DataflowBlock))
    assert df.statements[0].stream.routing.resolved_channel == 3


def test_multicast_error_start_zero():
    """start=0 means the sender is its own receiver — must be rejected."""
    code = """
    kernel @test<>() {
        dataflow u16 i, u16 j in [0:1, 0:4] {
            stream<f32> s = relative_stream(0, [0:4]) { hops = auto, channel = 0 }
        }
        compute u16 i, u16 j in [0:1, 0:1] { await send(val, s) }
        compute u16 i, u16 j in [0:1, 0:4] { await receive(val, s) }
    }
    """
    kernel = _parse_concretize(code)
    with pytest.raises(ValueError, match='start must be >= 1'):
        canonicalization.resolve_auto_hops(kernel)


def test_multicast_error_empty_range():
    """stop <= start is an empty range and must be rejected."""
    code = """
    kernel @test<>() {
        dataflow u16 i, u16 j in [0:1, 0:4] {
            stream<f32> s = relative_stream(0, [3:1]) { hops = auto, channel = 0 }
        }
        compute u16 i, u16 j in [0:1, 0:1] { await send(val, s) }
        compute u16 i, u16 j in [0:1, 1:4] { await receive(val, s) }
    }
    """
    kernel = _parse_concretize(code)
    with pytest.raises(ValueError, match='empty'):
        canonicalization.resolve_auto_hops(kernel)


def test_multicast_error_nonzero_fixed_offset():
    """Fixed offset != 0 is not yet supported."""
    code = """
    kernel @test<>() {
        dataflow u16 i, u16 j in [0:1, 0:4] {
            stream<f32> s = relative_stream(2, [1:4]) { hops = auto, channel = 0 }
        }
        compute u16 i, u16 j in [0:1, 0:1] { await send(val, s) }
        compute u16 i, u16 j in [0:1, 1:4] { await receive(val, s) }
    }
    """
    kernel = _parse_concretize(code)
    with pytest.raises(ValueError, match='non-zero fixed offset'):
        canonicalization.resolve_auto_hops(kernel)


# ---------------------------------------------------------------------------
# CSL lowering / routing tests
# ---------------------------------------------------------------------------

def _lower(K: int, **kwargs):
    kernel = _parse_concretize(_multicast_kernel(), K=K)
    return lower_spatial_ir_to_csl(kernel, **kwargs)


def test_multicast_error_send_and_receive_same_rectangle():
    """A compute rectangle that both sends and receives on a multicast stream must be rejected."""
    code = """
    kernel @test<>() {
        place u16 i, u16 j in [0:1, 0:4] { f32[1] val }
        dataflow u16 i, u16 j in [0:1, 0:4] {
            stream<f32> s = relative_stream(0, [1:3]) { hops = auto, channel = 0 }
        }
        compute u16 i, u16 j in [0:1, 0:4] {
            await send(val, s)
            await receive(val, s)
        }
    }
    """
    kernel = _parse_concretize(code)
    with pytest.raises(ValueError, match='both sent and received'):
        lower_spatial_ir_to_csl(kernel)


def test_multicast_K2_no_intermediate():
    """K=2: one receiver, no intermediate forwarding — no SOUTH+RAMP combination."""
    csl_files = _lower(K=2)
    layout = _layout_code(csl_files)
    assert 'SOUTH' in layout       # sender tx
    assert 'NORTH' in layout       # receiver rx
    assert 'SOUTH, RAMP' not in layout  # no intermediate forwarding when K=2


def test_multicast_K3_one_intermediate():
    """K=3: one intermediate that forwards SOUTH+RAMP."""
    csl_files = _lower(K=3)
    layout = _layout_code(csl_files)
    assert 'SOUTH, RAMP' in layout  # intermediate multicast


def test_multicast_sender_routing():
    """Sender PE gets rx=RAMP, tx=SOUTH."""
    csl_files = _lower(K=4)
    layout = _layout_code(csl_files)
    assert '.rx = .{RAMP}, .tx = .{SOUTH}' in layout


def test_multicast_last_receiver_routing():
    """Last receiver gets rx=NORTH, tx=RAMP (no forwarding)."""
    csl_files = _lower(K=4)
    layout = _layout_code(csl_files)
    assert '.rx = .{NORTH}, .tx = .{RAMP}' in layout


def test_multicast_intermediate_count():
    """Exactly K-2 intermediate multicast instructions for K receivers."""
    for K in [3, 4, 5, 8]:
        csl_files = _lower(K=K)
        layout = _layout_code(csl_files)
        n_intermediate = layout.count('SOUTH, RAMP')
        assert n_intermediate == K - 2, f'K={K}: expected {K-2} intermediates, got {n_intermediate}'


def test_multicast_single_color_pair():
    """Only one color_out and one color_in declaration are emitted per code file."""
    csl_files = _lower(K=5)
    for f in csl_files:
        # Count only the 'const … _color_out: color' declaration lines, not usage sites.
        out_decls = f.code.count('_color_out: color')
        in_decls = f.code.count('_color_in: color')
        assert out_decls <= 1
        assert in_decls <= 1


def test_multicast_x_axis_routing():
    """relative_stream([1:K], 0): routing uses EAST/WEST instead of SOUTH/NORTH."""
    code = """
    kernel @test<K>() {
        place u16 i, u16 j in [0:K, 0:1] { f32 val }
        dataflow u16 i, u16 j in [0:K, 0:1] {
            stream<f32> s = relative_stream([1:K], 0) { hops = auto, channel = 0 }
        }
        compute u16 i, u16 j in [0:1, 0:1] { await send(val, s) }
        compute u16 i, u16 j in [1:K, 0:1] { await receive(val, s) }
    }
    """
    kernel = _parse_concretize(code, K=4)
    csl_files = lower_spatial_ir_to_csl(kernel)
    layout = _layout_code(csl_files)
    assert 'EAST' in layout
    assert 'WEST' in layout
    assert 'SOUTH' not in layout
    assert 'NORTH' not in layout


def test_multicast_sample_file_lowers():
    """The multicast_generalized_y.sptl sample file lowers without error for several N/START values."""
    file = os.path.join(_TESTING_DIR, 'multicast_generalized_y.sptl')
    for N, START in [(2, 1), (3, 1), (5, 1), (8, 1), (5, 2)]:
        kernel = parser.parse_file(file)
        kernel = passes.concretize_parameters(kernel, N=N, START=START)
        kernel = passes.constexpr_propagation(kernel)
        csl_files = lower_spatial_ir_to_csl(kernel)
        assert csl_files, f'No output files for N={N}, START={START}'


def test_multicast_x_sample_file_lowers():
    """The multicast_generalized_x.sptl sample file lowers without error for several N/START values."""
    file = os.path.join(_TESTING_DIR, 'multicast_generalized_x.sptl')
    for N, START in [(2, 1), (3, 1), (5, 1), (8, 1), (5, 2)]:
        kernel = parser.parse_file(file)
        kernel = passes.concretize_parameters(kernel, N=N, START=START)
        kernel = passes.constexpr_propagation(kernel)
        csl_files = lower_spatial_ir_to_csl(kernel)
        assert csl_files, f'No output files for N={N}, START={START}'
        layout = _layout_code(csl_files)
        assert 'EAST' in layout, f'No EAST routing for N={N}, START={START}'
        assert 'WEST' in layout, f'No WEST routing for N={N}, START={START}'


# ---------------------------------------------------------------------------
# Negative multicast tests
# ---------------------------------------------------------------------------

def _neg_multicast_kernel(K: int | str = 'K', channel: int = 0) -> str:
    """
    Sender at j=K-1, multicasts NORTH to j=K-2 … j=0 using [-1:-(K)].
    Receivers cover [0:K-1].
    """
    return f"""
    kernel @test_neg<K>() {{
        place u16 i, u16 j in [0:1, 0:K] {{
            f32[1] val
        }}
        dataflow u16 i, u16 j in [0:1, 0:K] {{
            stream<f32> s = relative_stream(0, [-1:-K]) {{
                hops = auto,
                channel = {channel}
            }}
        }}
        compute u16 i, u16 j in [0:1, K-1:K] {{
            await send(val, s)
        }}
        compute u16 i, u16 j in [0:1, 0:K-1] {{
            await receive(val, s)
        }}
    }}
    """


def _lower_neg(K: int, **kwargs):
    kernel = _parse_concretize(_neg_multicast_kernel(), K=K)
    return lower_spatial_ir_to_csl(kernel, **kwargs)


def test_multicast_negative_parsed():
    """relative_stream(0, [-1:-K]) produces a MulticastRangeStreamDeclaration."""
    kernel = parser.parse_string(_neg_multicast_kernel())
    df = next(b for b in kernel.body if isinstance(b, spir.DataflowBlock))
    decl = df.statements[0]
    assert isinstance(decl.stream, spir.MulticastRangeStreamDeclaration)
    assert decl.stream.multicast_axis == 'y'
    assert isinstance(decl.stream.dy, spir.RangeExpression)


def test_multicast_negative_routing_injected():
    """Without explicit routing, _AutoHopResolver injects a default RoutingDeclaration for negative range."""
    code = """
    kernel @test<K>() {
        dataflow u16 i, u16 j in [0:1, 0:K] {
            stream<f32> s = relative_stream(0, [-1:-K])
        }
        compute u16 i, u16 j in [0:1, K-1:K] { await send(val, s) }
        compute u16 i, u16 j in [0:1, 0:K-1] { await receive(val, s) }
    }
    """
    kernel = _parse_concretize(code, K=4)
    kernel = canonicalization.resolve_auto_hops(kernel)
    df = next(b for b in kernel.body if isinstance(b, spir.DataflowBlock))
    routing = df.statements[0].stream.routing
    assert routing is not None
    assert routing.hops == []


def test_multicast_negative_K2_no_intermediate():
    """K=2: one receiver, no intermediate forwarding — no NORTH+RAMP combination."""
    csl_files = _lower_neg(K=2)
    layout = _layout_code(csl_files)
    assert 'NORTH' in layout       # sender tx
    assert 'SOUTH' in layout       # receiver rx
    assert 'NORTH, RAMP' not in layout  # no intermediate forwarding when K=2


def test_multicast_negative_K3_one_intermediate():
    """K=3: one intermediate that forwards NORTH+RAMP."""
    csl_files = _lower_neg(K=3)
    layout = _layout_code(csl_files)
    assert 'NORTH, RAMP' in layout


def test_multicast_negative_sender_routing():
    """Sender PE gets rx=RAMP, tx=NORTH."""
    csl_files = _lower_neg(K=4)
    layout = _layout_code(csl_files)
    assert '.rx = .{RAMP}, .tx = .{NORTH}' in layout


def test_multicast_negative_last_receiver_routing():
    """Last (farthest) receiver gets rx=SOUTH, tx=RAMP."""
    csl_files = _lower_neg(K=4)
    layout = _layout_code(csl_files)
    assert '.rx = .{SOUTH}, .tx = .{RAMP}' in layout


def test_multicast_negative_intermediate_count():
    """Exactly K-2 intermediate multicast instructions for K receivers."""
    for K in [3, 4, 5, 8]:
        csl_files = _lower_neg(K=K)
        layout = _layout_code(csl_files)
        n_intermediate = layout.count('NORTH, RAMP')
        assert n_intermediate == K - 2, f'K={K}: expected {K-2} intermediates, got {n_intermediate}'


def test_multicast_negative_error_empty():
    """[-1:0] has stop=0 >= start=-1, which is empty for negative multicast."""
    code = """
    kernel @test<>() {
        dataflow u16 i, u16 j in [0:1, 0:4] {
            stream<f32> s = relative_stream(0, [-1:0]) { hops = auto, channel = 0 }
        }
        compute u16 i, u16 j in [0:1, 3:4] { await send(val, s) }
        compute u16 i, u16 j in [0:1, 0:3] { await receive(val, s) }
    }
    """
    kernel = _parse_concretize(code)
    with pytest.raises(ValueError, match='empty'):
        canonicalization.resolve_auto_hops(kernel)


def test_multicast_negative_sample_file_lowers():
    """The multicast_generalized_y_neg.sptl sample file lowers without error for several N/START values."""
    file = os.path.join(_TESTING_DIR, 'multicast_generalized_y_neg.sptl')
    for N, START in [(2, 1), (3, 1), (5, 1), (8, 1), (5, 2)]:
        kernel = parser.parse_file(file)
        kernel = passes.concretize_parameters(kernel, N=N, START=START)
        kernel = passes.constexpr_propagation(kernel)
        csl_files = lower_spatial_ir_to_csl(kernel)
        assert csl_files, f'No output files for N={N}, START={START}'
        layout = _layout_code(csl_files)
        assert 'NORTH' in layout, f'No NORTH routing for N={N}, START={START}'


if __name__ == '__main__':
    test_multicast_dy_parsed_as_multicast_node()
    test_multicast_dy_range_values()
    test_multicast_dx_range()
    test_regular_relative_stream_unchanged()
    test_multicast_roundtrip()
    test_multicast_properties()
    test_multicast_auto_routing_injected()
    test_multicast_explicit_channel_preserved()
    test_multicast_error_start_zero()
    test_multicast_error_empty_range()
    test_multicast_error_nonzero_fixed_offset()
    test_multicast_error_send_and_receive_same_rectangle()
    test_multicast_K2_no_intermediate()
    test_multicast_K3_one_intermediate()
    test_multicast_sender_routing()
    test_multicast_last_receiver_routing()
    test_multicast_intermediate_count()
    test_multicast_single_color_pair()
    test_multicast_x_axis_routing()
    test_multicast_sample_file_lowers()
    test_multicast_x_sample_file_lowers()
    test_multicast_negative_parsed()
    test_multicast_negative_routing_injected()
    test_multicast_negative_K2_no_intermediate()
    test_multicast_negative_K3_one_intermediate()
    test_multicast_negative_sender_routing()
    test_multicast_negative_last_receiver_routing()
    test_multicast_negative_intermediate_count()
    test_multicast_negative_error_empty()
    test_multicast_negative_sample_file_lowers()
