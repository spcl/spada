"""
Tests for bundling overlapping 1D interval shifts onto one color.

The mechanism these check the lowering against is measured in
``tests/csl_runtime/test_shift_bundle_filters.sh``, which runs a hand-written version of the same
layout on the simulator.
"""
import os
import re

import pytest

from spada.lowering.spatial_ir_to_csl import canonicalize_kernel, lower_spatial_ir_to_csl
from spada.syntax.csl import routing as cslrouting
from spada.syntax.spatial_ir import canonicalization, parser, passes
from spada.syntax.spatial_ir.shift_bundles import detect_shift_bundles

_SHIFT = """
kernel @shift<M, D, K>(
    stream<f32, K>[D + M, 1] readonly inp,
    stream<f32, K>[D + M, 1] writeonly out
) {
    place i16 i, i16 j in [0:D + M, 0] {
        f32[K] val
    }
    phase {
        compute i16 i, i16 j in [0:D + M, 0] {
            await receive(val, inp[i, j])
        }
    }
    phase {
        dataflow i16 i, i16 j in [0:D + M, 0] {
            stream<f32, K> fwd = relative_stream(D, 0) {
                hops = auto,
                channel = 0
            }
        }
        compute i16 i, i16 j in [0:M, 0] {
            await send(val, fwd)
        }
        compute i16 i, i16 j in [D:D + M, 0] {
            await receive(val, fwd)
        }
    }
    phase {
        compute i16 i, i16 j in [0:D + M, 0] {
            await send(val, out[i, j])
        }
    }
}
"""

_WESTBOUND = _SHIFT.replace('relative_stream(D, 0)', 'relative_stream(-D, 0)') \
                   .replace('compute i16 i, i16 j in [0:M, 0] {\n            await send(val, fwd)',
                            'compute i16 i, i16 j in [D:D + M, 0] {\n            await send(val, fwd)') \
                   .replace('compute i16 i, i16 j in [D:D + M, 0] {\n            await receive(val, fwd)',
                            'compute i16 i, i16 j in [0:M, 0] {\n            await receive(val, fwd)')

_UNBOUNDED = _SHIFT.replace('stream<f32, K> fwd', 'stream<f32> fwd')


def _rectangles(source: str, **params: int):
    kernel = parser.parse_string(source)
    kernel = passes.concretize_parameters(kernel, **params)
    kernel = passes.constexpr_propagation(kernel)
    kernel = canonicalize_kernel(kernel)
    return canonicalization.consolidate_rectangles_to_equivalence_classes(kernel)


def _bundles(source: str, **params: int):
    return detect_shift_bundles(_rectangles(source, **params))


def _layout(source: str, **params: int) -> str:
    kernel = parser.parse_string(source)
    kernel = passes.concretize_parameters(kernel, **params)
    kernel = passes.constexpr_propagation(kernel)
    files = lower_spatial_ir_to_csl(kernel, disable_benchmarking=True)
    return next(f.code for f in files if 'layout' in f.filename)


def _codes(source: str, **params: int) -> dict[str, str]:
    kernel = parser.parse_string(source)
    kernel = passes.concretize_parameters(kernel, **params)
    kernel = passes.constexpr_propagation(kernel)
    files = lower_spatial_ir_to_csl(kernel, disable_benchmarking=True)
    return {f.filename: f.code for f in files if f.filename.startswith('code_')}


def _configs(layout: str) -> dict[tuple[int, int], str]:
    """
    Maps each ``@set_color_config`` loop in a layout to its configuration, keyed by the PE range.

    Only the standalone loops a shift bundle produces are keyed this way; they hold one call each.
    """
    found = {}
    for start, stop, body in re.findall(
            r'for \(@range\(i16, (\d+), (\d+), 1\)\) \|pe_x\| \{\s*'
            r'for \(@range\(i16, \d+, \d+, 1\)\) \|pe_y\| \{\s*'
            r'(@set_color_config\([^\n]*\);)', layout):
        found[(int(start), int(stop))] = body
    return found


def test_detects_the_overlapping_shift():
    bundles = _bundles(_SHIFT, M=3, D=3, K=1)
    assert len(bundles) == 1
    bundle = bundles[0]
    assert (bundle.axis, bundle.sign, bundle.start, bundle.length, bundle.dist, bundle.words) \
        == ('x', 1, 0, 3, 3, 1)
    assert (bundle.sources(), bundle.destinations(), bundle.relays()) == ((0, 3), (3, 6), (3, 3))


def test_a_gap_between_the_halves_becomes_relays():
    bundle = _bundles(_SHIFT, M=3, D=5, K=1)[0]
    assert (bundle.sources(), bundle.relays(), bundle.destinations()) == ((0, 3), (3, 5), (5, 8))


def test_westbound_shift_is_detected_mirrored():
    bundle = _bundles(_WESTBOUND, M=3, D=5, K=1)[0]
    assert (bundle.sign, bundle.sources(), bundle.relays(), bundle.destinations()) \
        == (-1, (5, 8), (3, 5), (0, 3))
    # The stream arrives on the destinations' east side, so it is the westmost one it reaches last.
    assert bundle.destination_order() == (2, -1)


def test_an_unbounded_stream_is_not_bundled():
    # Without a bound there is no self-close, so nothing would advance the sources' switches.
    assert _bundles(_UNBOUNDED, M=3, D=3, K=1) == []


def test_a_single_source_is_not_bundled():
    assert _bundles(_SHIFT, M=1, D=3, K=1) == []


def test_a_shift_of_one_is_not_bundled():
    # Consecutive sources at distance one form a chain, which the ordinary receive-then-send switch
    # positions already sequence.
    assert _bundles(_SHIFT, M=1, D=1, K=1) == []


def test_sources_longer_than_the_shift_are_not_bundled():
    # Sources would be destinations of the same bundle, which this arrangement cannot express.
    assert _bundles(_SHIFT, M=4, D=2, K=1) == []


def test_sources_inject_then_relay():
    configs = _configs(_layout(_SHIFT, M=3, D=3, K=1))
    sources = configs[(0, 3)]
    assert '.routes = .{ .rx = .{RAMP}, .tx = .{EAST} }' in sources
    assert '.switches = .{ .pos1 = .{ .rx = WEST } }' in sources
    assert 'ring_mode' not in sources
    # One call covers the whole run, the westmost source included: a switch position it never uses
    # is cheaper than a second configuration.
    assert '.filter' not in sources


def test_destinations_are_static_and_filtered():
    configs = _configs(_layout(_SHIFT, M=3, D=3, K=1))
    passing = configs[(3, 5)]
    assert '.routes = .{ .rx = .{WEST}, .tx = .{RAMP, EAST} }' in passing
    assert '.switches' not in passing
    # Destination 3 keeps the last of the three words, destination 4 the second: the counter has to
    # start one and two words short of the end of the cycle respectively.
    assert '.init_counter = pe_x - 2' in passing
    assert '.limit1 = 2, .max_counter = 0' in passing


def test_the_last_destination_terminates_the_stream():
    configs = _configs(_layout(_SHIFT, M=3, D=3, K=1))
    terminal = configs[(5, 6)]
    assert '.routes = .{ .rx = .{WEST}, .tx = .{RAMP} }' in terminal
    assert '.init_counter = 0' in terminal


def test_relays_pass_the_stream_through_unchanged():
    configs = _configs(_layout(_SHIFT, M=3, D=5, K=1))
    relays = configs[(3, 5)]
    assert '.routes = .{ .rx = .{WEST}, .tx = .{EAST} }' in relays
    assert '.switches' not in relays and '.filter' not in relays


def test_westbound_layout_mirrors_the_eastbound_one():
    configs = _configs(_layout(_WESTBOUND, M=3, D=3, K=1))
    assert '.routes = .{ .rx = .{RAMP}, .tx = .{WEST} }' in configs[(3, 6)]
    assert '.switches = .{ .pos1 = .{ .rx = EAST } }' in configs[(3, 6)]
    assert '.routes = .{ .rx = .{EAST}, .tx = .{RAMP, WEST} }' in configs[(1, 3)]
    assert '.init_counter = 3 - pe_x' in configs[(1, 3)]
    assert '.routes = .{ .rx = .{EAST}, .tx = .{RAMP} }' in configs[(0, 1)]
    assert '.init_counter = 0' in configs[(0, 1)]


def test_windows_are_as_wide_as_the_stream_bound():
    configs = _configs(_layout(_SHIFT, M=3, D=3, K=2))
    # Six words in the cycle, two of which each destination keeps.
    assert '.limit1 = 5, .max_counter = 1' in configs[(3, 5)]
    assert '.init_counter = (pe_x - 2) * 2' in configs[(3, 5)]
    assert '.limit1 = 5, .max_counter = 1' in configs[(5, 6)]


def test_each_source_advances_its_own_switch_once():
    codes = _codes(_SHIFT, M=3, D=3, K=1)
    sending = codes['code_0_0.csl']
    assert sending.count('ctrl.opcode.SWITCH_ADV') == 1
    assert '.control = true' in sending
    # The destinations do not switch, so nothing is emitted there.
    assert 'SWITCH_ADV' not in codes['code_3_0.csl']


def test_one_color_carries_the_whole_bundle():
    layout = _layout(_SHIFT, M=3, D=3, K=1)
    routes = layout[layout.index('// Routes'):]
    assert {int(color) for color in re.findall(r'@get_color\((\d+)\)', routes)} == {0}


def test_no_port_is_a_union_on_the_receiving_side():
    # A switch position accepts a single rx direction, and only a destination unions its tx.
    layout = _layout(_SHIFT, M=3, D=3, K=1)
    assert re.search(r'\.rx = \.\{[A-Z]+, [A-Z]+\}', layout) is None
    assert re.search(r'\.pos\d = \.\{ \.rx = [A-Z]+, ', layout) is None


def test_filter_renders_as_a_color_config_field():
    plan = cslrouting.ColorSwitchPlan(
        [cslrouting.RouteConfig(('WEST', ), ('RAMP', 'EAST'))],
        filter=cslrouting.FilterConfig('pe_x - 2', '2', '0'))
    assert plan.as_csl() == ('.{ .routes = .{ .rx = .{WEST}, .tx = .{RAMP, EAST} }, '
                             '.filter = .{ .kind = .{ .counter = true }, .count_data = true, '
                             '.init_counter = pe_x - 2, .limit1 = 2, .max_counter = 0 } }')


def test_a_pe_cannot_use_more_filters_than_the_hardware_has():
    from spada.syntax.csl import constants

    def site(color: int):
        return cslrouting._RouteSite(color=color, x_range=(0, 4, 1), y_range=(0, 1, 1), absolute=True)

    def entry():
        return cslrouting._RouteEntry(cslrouting.RouteConfig(('WEST', ), ('RAMP', )), (0, 0, 0), 0,
                                     (0, 0), None, '', cslrouting.FilterConfig('0', '1', '0'))

    entries = {site(color): [entry()] for color in range(constants.FILTERS_PER_PE + 1)}
    with pytest.raises(SyntaxError, match='wavelet filters'):
        cslrouting._check_filter_budget(entries)


def _bundled_batcher(l: int, k: int = 1):
    path = os.path.join(os.path.dirname(__file__), '..', '..', 'samples', 'spatial', 'sort',
                        'batcher_oddeven_bundled_1D.sptl')
    kernel = parser.parse_file(path)
    kernel = passes.concretize_parameters(kernel, L=l, K=k)
    kernel = passes.constexpr_propagation(kernel)
    return lower_spatial_ir_to_csl(kernel, disable_benchmarking=True)


def _colors_of(layout: str, pattern: str = '') -> set[int]:
    routes = layout[layout.index('// Routes'):]
    return {int(color) for color in re.findall(r'@get_color\((\d+)\)[^;]*' + pattern, routes)}


@pytest.mark.parametrize('l, colors', [(2, 6), (3, 10)])
def test_the_batcher_fits_the_colors_it_has(l: int, colors: int):
    # A bundled phase puts all of its matchings on one color pair; the phases left unbundled take a
    # pair per matching, but share those across phases wherever their sources agree mod 2d.
    from spada.syntax.csl import constants

    used = _colors_of(next(f.code for f in _bundled_batcher(l) if 'layout' in f.filename))
    assert len(used) == colors
    assert len(used) <= len(constants.COLORS)


def test_sixteen_keys_need_three_overlapping_input_queues():
    """
    At L = 4 a reused inbound color stays live across a gap that already holds two other colors.
    WSE-2 has two input queues, so lowering must refuse rather than remap a busy queue.
    """
    from spada.syntax.csl import constants

    if len(constants.INPUT_QUEUE_IDS) >= 3:
        pytest.skip(f'{constants.ARCH} has {len(constants.INPUT_QUEUE_IDS)} input queues, enough for L=4')
    with pytest.raises(SyntaxError, match='concurrent input queues'):
        _bundled_batcher(4)


def test_only_the_widest_batcher_phases_are_bundled():
    # Bundling costs one filter at every participating PE and a PE has three, so the sample bundles
    # every phase that satisfies 4d >= N. At L = 3 that is already three phases (d = 4, 2, 2).
    from spada.syntax.csl import constants

    layout = next(f.code for f in _bundled_batcher(3) if 'layout' in f.filename)
    used, filtered, switched = _colors_of(layout), _colors_of(layout, r'\.filter'), _colors_of(layout, r'\.switches')

    assert len(filtered) == 2 * constants.FILTERS_PER_PE  # three phases, two directions each
    assert filtered == switched  # a bundled color is one whose sources hand over to relay mode
    assert not (used - filtered) & switched  # the pooled ones hold a single static configuration


def test_a_wider_block_costs_wavelets_not_colors():
    # The Batcher trades K keys per comparator instead of one. A bundle of M sources then carries
    # M*K wavelets per epoch, of which each destination keeps the K its filter windows out -- the
    # colors and the filters stay as they are, only the counters grow.
    narrow = next(f.code for f in _bundled_batcher(3, 1) if 'layout' in f.filename)
    wide = next(f.code for f in _bundled_batcher(3, 4) if 'layout' in f.filename)

    assert _colors_of(wide) == _colors_of(narrow)
    assert _colors_of(wide, r'\.filter') == _colors_of(narrow, r'\.filter')
    # At L=3 the bundled phases have two and four sources, so cycles of 8 and 16 words.
    assert '.limit1 = 7, .max_counter = 3' in wide
    assert '.limit1 = 15, .max_counter = 3' in wide
    assert '.init_counter = (pe_x - 1) * 4' in wide


def test_a_scalar_receive_lowers_to_a_data_task():
    # A bundle destination that keeps a single key holds it in a scalar, which arrives as the
    # argument of a data task rather than as a move out of the fabric.
    path = os.path.join(
        os.path.dirname(__file__), '..', '..', 'samples', 'spatial', 'simple', 'exchange_bundle_1D.sptl'
    )
    kernel = parser.parse_file(path)
    kernel = passes.concretize_parameters(kernel, M=2, D=2, R=1)
    kernel = passes.constexpr_propagation(kernel)
    files = lower_spatial_ir_to_csl(kernel, disable_benchmarking=True)
    pe_codes = [f.code for f in files if 'code_' in f.filename]
    assert pe_codes
    for code in pe_codes:
        # A scalar receive becomes a data task, not an undeclared assignment, and a scalar source
        # cannot be moved asynchronously.
        assert 'tmp = bwd' not in code
        assert '.async = true' not in code
    pe0 = next(f.code for f in files if 'code_0_0' in f.filename)
    assert 'task dtask_' in pe0
    assert 'tmp = __x' in pe0
