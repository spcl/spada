"""Tests for count= routing and counted 1D shift-bundle switching."""
import os
import re

import pytest

from spada.lowering.spatial_ir_to_csl import lower_spatial_ir_to_csl
from spada.syntax.spatial_ir import canonical_subgrids, canonicalization, irnodes as spir, parser, passes
from spada.syntax.spatial_ir.shift_bundles import (
    detect_shift_bundles,
    schedule_counted_switch,
    coalesce_shift_bundles,
    switch_advance_for_bundle,
)


def _prepare(src: str, **params: int):
    kernel = parser.parse_string(src)
    kernel = passes.concretize_parameters(kernel, **params)
    kernel = passes.constexpr_propagation(kernel)
    kernel = canonicalization.inline_metaprogramming(kernel)
    kernel = canonicalization.canonicalize_phases(kernel)
    kernel = canonicalization.reduce_streams(kernel)
    kernel = canonical_subgrids.canonicalize_subgrids(kernel)
    kernel = canonicalization.resolve_auto_hops(kernel)
    return kernel


_SHIFT = """
kernel @shift_k<N>(
    stream<f32, 1>[N, 1] readonly inp,
    stream<f32, 1>[N, 1] writeonly out
) {
    place i16 i, i16 j in [0:N, 0] {
        f32 val
    }
    phase {
        compute i16 i, i16 j in [0:N, 0] {
            await receive(val, inp[i, j])
        }
    }
    phase {
        dataflow i16 i, i16 j in [0:4, 0] {
            stream<f32> fwd = relative_stream(4, 0) {
                hops = auto,
                channel = auto,
                count = 1
            }
        }
        dataflow i16 i, i16 j in [4:8, 0] {
            stream<f32> fwd = relative_stream(4, 0) {
                hops = auto,
                channel = auto,
                count = 1
            }
        }
        compute i16 i, i16 j in [0:4, 0] {
            await send(val, fwd)
        }
        compute i16 i, i16 j in [4:8, 0] {
            await receive(val, fwd)
        }
    }
    phase {
        compute i16 i, i16 j in [0:N, 0] {
            await send(val, out[i, j])
        }
    }
}
"""

_SHIFT_AUTO = _SHIFT.replace("count = 1", "count = auto")

_SHIFT_OMITTED = _SHIFT.replace(",\n                count = 1", "")

_SHIFT_TOO_LONG = _SHIFT.replace("[0:4, 0]", "[0:6, 0]").replace("relative_stream(4, 0)", "relative_stream(2, 0)")


def test_parse_count_roundtrip():
    kernel = parser.parse_string(_SHIFT)
    ir_1 = kernel.as_ir()
    assert "count = 1" in ir_1
    ir_2 = parser.parse_string(ir_1).as_ir()
    assert ir_1 == ir_2


def test_omitted_count_has_no_count_line():
    kernel = parser.parse_string(_SHIFT_OMITTED)
    assert "count =" not in kernel.as_ir()


def test_detect_interval_shift():
    kernel = _prepare(_SHIFT, N=8)
    bundles = detect_shift_bundles(kernel)
    assert len(bundles) == 1
    b = bundles[0]
    assert (b.start, b.length, b.dist, b.count, b.sign, b.axis) == (0, 4, 4, 1, 1, "x")


def test_auto_count_is_not_rewritten():
    kernel = _prepare(_SHIFT_AUTO, N=8)
    assert detect_shift_bundles(kernel) == []
    coalesce_shift_bundles(kernel)
    assert kernel.shift_schedules == []


def test_omitted_count_is_not_rewritten():
    kernel = _prepare(_SHIFT_OMITTED, N=8)
    assert detect_shift_bundles(kernel) == []


def test_m_greater_than_d_is_rejected():
    kernel = _prepare(_SHIFT_TOO_LONG, N=8)
    assert detect_shift_bundles(kernel) == []


def test_schedule_source_and_dest_halves():
    kernel = _prepare(_SHIFT, N=8)
    bundle = detect_shift_bundles(kernel)[0]
    by_pe = {(s.x, s.y): s.steps for s in schedule_counted_switch(bundle)}
    assert [(st.rx, st.tx, st.waves) for st in by_pe[(0, 0)]] == [("RAMP", "EAST", 1)]
    assert [(st.rx, st.tx, st.waves) for st in by_pe[(1, 0)]] == [("WEST", "EAST", 1), ("RAMP", "EAST", 1)]
    assert [(st.rx, st.tx, st.waves) for st in by_pe[(3, 0)]] == [("WEST", "EAST", 3), ("RAMP", "EAST", 1)]
    assert [(st.rx, st.tx, st.waves) for st in by_pe[(4, 0)]] == [("WEST", "RAMP", 1), ("WEST", "EAST", 3)]
    assert [(st.rx, st.tx, st.waves) for st in by_pe[(7, 0)]] == [("WEST", "RAMP", 1)]
    adv = switch_advance_for_bundle(bundle)
    assert adv.last_injector == 3
    assert adv.opcodes == ("SWITCH_ADV", "NOP", "NOP", "SWITCH_ADV")


def test_westbound_schedule():
    west = """
kernel @shift_w<N>(
    stream<f32, 1>[N, 1] readonly inp,
    stream<f32, 1>[N, 1] writeonly out
) {
    place i16 i, i16 j in [0:N, 0] { f32 val }
    phase {
        compute i16 i, i16 j in [0:N, 0] { await receive(val, inp[i, j]) }
    }
    phase {
        dataflow i16 i, i16 j in [4:8, 0] {
            stream<f32> bwd = relative_stream(-4, 0) {
                hops = auto,
                channel = auto,
                count = 1
            }
        }
        compute i16 i, i16 j in [4:8, 0] { await send(val, bwd) }
        compute i16 i, i16 j in [0:4, 0] { await receive(val, bwd) }
    }
    phase {
        compute i16 i, i16 j in [0:N, 0] { await send(val, out[i, j]) }
    }
}
"""
    kernel = _prepare(west, N=8)
    bundle = detect_shift_bundles(kernel)[0]
    assert bundle.sign == -1 and bundle.start == 4 and bundle.length == 4
    by_pe = {(s.x, s.y): s.steps for s in schedule_counted_switch(bundle)}
    assert [(st.rx, st.tx, st.waves) for st in by_pe[(7, 0)]] == [("RAMP", "WEST", 1)]
    assert [(st.rx, st.tx, st.waves) for st in by_pe[(4, 0)]] == [("EAST", "WEST", 3), ("RAMP", "WEST", 1)]
    assert [(st.rx, st.tx, st.waves) for st in by_pe[(0, 0)]] == [("EAST", "RAMP", 1)]
    assert [(st.rx, st.tx, st.waves) for st in by_pe[(3, 0)]] == [("EAST", "RAMP", 1), ("EAST", "WEST", 3)]


def _batcher_prepared(n_log: int):
    path = os.path.join(
        os.path.dirname(__file__), "..", "..", "samples", "spatial", "sort", "batcher_oddeven_1D.sptl"
    )
    kernel = parser.parse_file(path)
    kernel = passes.concretize_parameters(kernel, L=n_log)
    kernel = passes.constexpr_propagation(kernel)
    kernel = canonicalization.inline_metaprogramming(kernel)
    kernel = canonicalization.canonicalize_phases(kernel)
    kernel = canonicalization.reduce_streams(kernel)
    kernel = canonical_subgrids.canonicalize_subgrids(kernel)
    kernel = canonicalization.resolve_auto_hops(kernel)
    return kernel


def test_batcher_n16_p1_d8_bundle():
    kernel = _batcher_prepared(4)
    bundles = detect_shift_bundles(kernel)
    east = [b for b in bundles if b.sign > 0 and b.dist == 8]
    west = [b for b in bundles if b.sign < 0 and b.dist == 8]
    assert len(east) == 1 and east[0].start == 0 and east[0].length == 8
    assert len(west) == 1 and west[0].start == 8 and west[0].length == 8


def test_batcher_n16_p2_d4_bundle():
    kernel = _batcher_prepared(4)
    bundles = detect_shift_bundles(kernel)
    east = [b for b in bundles if b.sign > 0 and b.dist == 4]
    assert any(b.start == 4 and b.length == 4 for b in east)


def test_batcher_d1_has_no_counted_switch():
    kernel = _batcher_prepared(3)
    bundles = detect_shift_bundles(kernel)
    assert all(b.dist != 1 for b in bundles)


def _xy(bundle, coord):
    return (coord, bundle.fixed) if bundle.axis == "x" else (bundle.fixed, coord)


def _consume(states, xy, role):
    steps = states[xy]
    assert steps, f"PE {xy} has no remaining config but needs {role}"
    rx, tx, waves = steps[0]
    if role == "inject":
        assert rx == "RAMP" and tx != "RAMP", (xy, steps[0], role)
    elif role == "absorb":
        assert tx == "RAMP" and rx != "RAMP", (xy, steps[0], role)
    elif role == "forward":
        assert rx != "RAMP" and tx != "RAMP", (xy, steps[0], role)
    else:
        raise ValueError(role)
    waves -= 1
    if waves == 0:
        steps.pop(0)
    else:
        steps[0] = (rx, tx, waves)


def simulate_bundle_delivery(bundle, drop_second_step=False):
    """
    West-first (eastbound) / east-first (westbound) serial delivery.

    Each fabric word consumes one wave at every PE on its path. After the
    source-half forward quota the PE must have switched to inject; after the
    dest-half absorb quota it must have switched to forward. Exhausting every
    quota means the two-state program matches the matching.
    """
    states = {}
    for sched in schedule_counted_switch(bundle):
        steps = [(st.rx, st.tx, st.waves) for st in sched.steps]
        if drop_second_step:
            steps = steps[:1]
        states[(sched.x, sched.y)] = steps
    order = range(bundle.length) if bundle.sign > 0 else range(bundle.length - 1, -1, -1)
    for j in order:
        src = bundle.start + j
        dest = src + bundle.sign * bundle.dist
        for _ in range(bundle.count):
            _consume(states, _xy(bundle, src), "inject")
            hop = src + bundle.sign
            while hop != dest:
                _consume(states, _xy(bundle, hop), "forward")
                hop += bundle.sign
            _consume(states, _xy(bundle, dest), "absorb")
    leftover = {pe: steps for pe, steps in states.items() if steps}
    assert leftover == {}, leftover


def test_two_state_switch_delivers_eastbound():
    kernel = _prepare(_SHIFT, N=8)
    simulate_bundle_delivery(detect_shift_bundles(kernel)[0])


def test_two_state_switch_delivers_westbound():
    west = """
kernel @shift_w<N>(
    stream<f32, 1>[N, 1] readonly inp,
    stream<f32, 1>[N, 1] writeonly out
) {
    place i16 i, i16 j in [0:N, 0] { f32 val }
    phase {
        compute i16 i, i16 j in [0:N, 0] { await receive(val, inp[i, j]) }
    }
    phase {
        dataflow i16 i, i16 j in [4:8, 0] {
            stream<f32> bwd = relative_stream(-4, 0) {
                hops = auto,
                channel = auto,
                count = 1
            }
        }
        compute i16 i, i16 j in [4:8, 0] { await send(val, bwd) }
        compute i16 i, i16 j in [0:4, 0] { await receive(val, bwd) }
    }
    phase {
        compute i16 i, i16 j in [0:N, 0] { await send(val, out[i, j]) }
    }
}
"""
    kernel = _prepare(west, N=8)
    simulate_bundle_delivery(detect_shift_bundles(kernel)[0])


def test_two_state_switch_delivers_count_2():
    kernel = _prepare(_SHIFT.replace("count = 1", "count = 2"), N=8)
    bundle = detect_shift_bundles(kernel)[0]
    assert bundle.count == 2
    simulate_bundle_delivery(bundle)


def test_without_the_switch_delivery_fails():
    kernel = _prepare(_SHIFT, N=8)
    with pytest.raises(AssertionError):
        simulate_bundle_delivery(detect_shift_bundles(kernel)[0], drop_second_step=True)


def test_batcher_d8_switch_delivers():
    kernel = _batcher_prepared(4)
    bundles = detect_shift_bundles(kernel)
    east = next(b for b in bundles if b.sign > 0 and b.dist == 8)
    west = next(b for b in bundles if b.sign < 0 and b.dist == 8)
    simulate_bundle_delivery(east)
    simulate_bundle_delivery(west)


def test_two_state_lemma_never_needs_a_third_config():
    kernel = _prepare(_SHIFT, N=8)
    bundle = detect_shift_bundles(kernel)[0]
    for sched in schedule_counted_switch(bundle):
        assert 1 <= len(sched.steps) <= 2
        if len(sched.steps) == 2:
            first, second = sched.steps
            forward_then_inject = first.rx != "RAMP" and first.tx != "RAMP" and second.rx == "RAMP"
            absorb_then_forward = first.tx == "RAMP" and second.rx != "RAMP" and second.tx != "RAMP"
            assert forward_then_inject or absorb_then_forward


def _onchip_channels_by_phase(kernel):
    phases = []
    for block in kernel.body:
        if not isinstance(block, spir.Phase):
            continue
        chans = set()
        for dataflow in block.dataflow:
            for stmt in dataflow.statements:
                routing = getattr(stmt.stream, "routing", None)
                if routing is None or routing.resolved_channel == "auto":
                    continue
                chans.add(routing.resolved_channel)
        phases.append(chans)
    return phases


def test_batcher_two_colors_per_phase_not_globally():
    kernel = _batcher_prepared(3)
    coalesce_shift_bundles(kernel)
    routed = [chans for chans in _onchip_channels_by_phase(kernel) if chans]
    assert len(routed) == 6  # L=3 has 6 (l,p) CAS phases
    for chans in routed:
        assert len(chans) == 2
    used = [c for chans in routed for c in chans]
    assert len(used) == len(set(used))
    assert set(used) == set(range(12))


def test_batcher_scalar_receive_lowers_to_data_task():
    path = os.path.join(
        os.path.dirname(__file__), "..", "..", "samples", "spatial", "sort", "batcher_oddeven_1D.sptl"
    )
    kernel = parser.parse_file(path)
    kernel = passes.concretize_parameters(kernel, L=1)
    kernel = passes.constexpr_propagation(kernel)
    files = lower_spatial_ir_to_csl(kernel, disable_benchmarking=True)
    pe_codes = [f.code for f in files if "code_" in f.filename]
    assert pe_codes
    for code in pe_codes:
        assert "tmp = bwd" not in code
        assert ".async = true" not in code
    pe0 = next(f.code for f in files if "code_0_0" in f.filename)
    assert "task dtask_" in pe0
    assert "tmp = __x" in pe0


def test_lowering_encodes_intra_phase_switch():
    kernel = parser.parse_string(_SHIFT)
    kernel = passes.concretize_parameters(kernel, N=8)
    kernel = passes.constexpr_propagation(kernel)
    files = lower_spatial_ir_to_csl(kernel, disable_benchmarking=True)
    layout = next(f.code for f in files if "layout" in f.filename)
    assert "spa_phase_reload" not in layout
    # PE 1: forward 1 wave W→E, then switch to inject R→E.
    assert re.search(
        r"@set_color_config\(1, 0, @get_color\(0\), "
        r"\.\{ \.routes = \.\{ \.rx = \.\{WEST\}, \.tx = \.\{EAST\} \}",
        layout,
    )
    assert re.search(
        r"spa_switch_after phase=\d+ pe=1,0 ch=0 waves=1 rx=RAMP tx=EAST",
        layout,
    )
    assert ".pos1 = .{ .rx = RAMP }" in layout
    assert ".pop_mode = .{ .pop_on_advance_nop = true }" in layout
    assert ".pop_mode = .{ .no_pop = true }" in layout
    # PE 4: absorb 1 wave W→R, then switch to forward W→E.
    assert re.search(
        r"@set_color_config\(4, 0, @get_color\(0\), "
        r"\.\{ \.routes = \.\{ \.rx = \.\{WEST\}, \.tx = \.\{RAMP\} \}",
        layout,
    )
    assert re.search(
        r"spa_switch_after phase=\d+ pe=4,0 ch=0 waves=1 rx=WEST tx=EAST",
        layout,
    )
    assert ".pos1 = .{ .tx = EAST }" in layout
    pe0 = next(f.code for f in files if "code_0_0" in f.filename)
    assert "ctrl.opcode.SWITCH_ADV" in pe0
    assert "encode_payload" in pe0
    assert "get_fabric_coord" in pe0


def test_batcher_lowering_two_colors_per_phase():
    path = os.path.join(
        os.path.dirname(__file__), "..", "..", "samples", "spatial", "sort", "batcher_oddeven_1D.sptl"
    )
    for n_log, n_cas, old_static in ((3, 6, 14), (4, 10, 30)):
        kernel = parser.parse_file(path)
        kernel = passes.concretize_parameters(kernel, L=n_log)
        kernel = passes.constexpr_propagation(kernel)
        files = lower_spatial_ir_to_csl(kernel, disable_benchmarking=True)
        layout = next(f.code for f in files if "layout" in f.filename)
        assert "spa_color_schedule" in layout
        assert "spa_switch_after" in layout
        assert "spa_phase_reload" not in layout
        colors = {int(c) for c in re.findall(r"@get_color\((\d+)\)", layout)}
        assert colors == set(range(2 * n_cas)), (
            f"L={n_log} used colors {sorted(colors)}, expected 2 per phase "
            f"({2 * n_cas}), not a kernel-wide pair and not {old_static} static"
        )
        # Every two-step schedule has a matching counted switch onto the second pair.
        schedules = re.findall(
            r"spa_color_schedule phase=(\d+) pe=(\d+),(\d+) ch=(\d+) : ([^\n]+)",
            layout,
        )
        switches = {
            (int(ph), int(x), int(y), int(ch)): (int(w), rx, tx)
            for ph, x, y, ch, w, rx, tx in re.findall(
                r"spa_switch_after phase=(\d+) pe=(\d+),(\d+) ch=(\d+) "
                r"waves=(\d+) rx=(\w+) tx=(\w+)",
                layout,
            )
        }
        short = {"R": "RAMP", "E": "EAST", "W": "WEST", "N": "NORTH", "S": "SOUTH"}
        for ph, x, y, ch, step_txt in schedules:
            parts = [p.strip() for p in step_txt.split(";")]
            key = (int(ph), int(x), int(y), int(ch))
            if len(parts) < 2:
                assert key not in switches
                continue
            first, second = parts[0], parts[1]
            first_waves = int(re.search(r"waves=(\d+)", first).group(1))
            pair = re.search(r"(\w+)->(\w+)", second)
            rx = short.get(pair.group(1), pair.group(1))
            tx = short.get(pair.group(2), pair.group(2))
            assert switches[key] == (first_waves, rx, tx)


_SAMPLE_SHIFT = os.path.join(
    os.path.dirname(__file__), "..", "..", "samples", "spatial", "simple", "shift_bundle_1D.sptl"
)

_DUAL_PORT = re.compile(
    r"\.(?:rx|tx) = \.\{[A-Z]+, [A-Z]+\}"
)
_ABS_COLOR_CONFIG = re.compile(
    r"@set_color_config\((\d+), (\d+), @get_color\((\d+)\),"
)


def _sample_shift(m: int = 4):
    kernel = parser.parse_file(_SAMPLE_SHIFT)
    kernel = passes.concretize_parameters(kernel, M=m)
    kernel = passes.constexpr_propagation(kernel)
    return kernel


def test_sample_shift_bundle_is_one_eastbound_color():
    kernel = _prepare(open(_SAMPLE_SHIFT).read(), M=4)
    bundles = detect_shift_bundles(kernel)
    assert len(bundles) == 1
    b = bundles[0]
    assert (b.start, b.length, b.dist, b.sign, b.axis, b.count) == (0, 4, 4, 1, "x", 1)
    coalesce_shift_bundles(kernel)
    routed = [chans for chans in _onchip_channels_by_phase(kernel) if chans]
    assert routed == [{0}]


def test_sample_shift_bundle_one_rx_tx_pair_at_a_time():
    """A color installs one (rx, tx) pair; pos1 changes rx or tx, never both."""
    kernel = _prepare(open(_SAMPLE_SHIFT).read(), M=4)
    bundle = detect_shift_bundles(kernel)[0]
    for sched in schedule_counted_switch(bundle):
        assert 1 <= len(sched.steps) <= 2
        for step in sched.steps:
            assert step.rx in {"RAMP", "WEST", "EAST"}
            assert step.tx in {"RAMP", "WEST", "EAST"}
            assert step.rx != step.tx
        if len(sched.steps) == 2:
            first, second = sched.steps
            changed_rx = first.rx != second.rx
            changed_tx = first.tx != second.tx
            assert changed_rx ^ changed_tx, (
                f"PE ({sched.x},{sched.y}) changes both rx and tx: "
                f"{first.as_pair()} then {second.as_pair()}"
            )


def test_sample_shift_bundle_layout_does_not_union_ports():
    kernel = _sample_shift(4)
    files = lower_spatial_ir_to_csl(kernel, disable_benchmarking=True)
    layout = next(f.code for f in files if "layout" in f.filename)
    assert _DUAL_PORT.search(layout) is None
    keys = _ABS_COLOR_CONFIG.findall(layout)
    assert keys
    assert len(keys) == len(set(keys)), f"duplicate (PE, color) configs: {keys}"
    assert "spa_switch_after" in layout
    assert ".pos1 = .{ .rx = RAMP }" in layout
    assert ".pos1 = .{ .tx = EAST }" in layout
    simulate_bundle_delivery(detect_shift_bundles(_prepare(open(_SAMPLE_SHIFT).read(), M=4))[0])
