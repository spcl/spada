"""
Tests for router switch planning: route configuration merging, switch positions, the control
wavelets that retire a configuration, and the hardware capacity limits.
"""
import os
import re

import pytest

from spada.lowering import spatial_ir_to_csl as s2c
from spada.syntax.csl import constants as csl, routing as cslrouting
from spada.syntax.spatial_ir import canonicalization, parser, passes

SAMPLES = os.path.join(os.path.dirname(__file__), 'samples')


def _lower(filename: str, **parameters) -> dict[str, str]:
    kernel = parser.parse_file(os.path.join(SAMPLES, filename))
    if parameters:
        kernel = passes.concretize_parameters(kernel, **parameters)
    kernel = passes.constexpr_propagation(kernel)
    return {f.filename: f.code for f in s2c.lower_spatial_ir_to_csl(kernel)}


def _lower_string(code: str, **parameters) -> dict[str, str]:
    kernel = parser.parse_string(code, 'test.sptl')
    if parameters:
        kernel = passes.concretize_parameters(kernel, **parameters)
    kernel = passes.constexpr_propagation(kernel)
    return {f.filename: f.code for f in s2c.lower_spatial_ir_to_csl(kernel)}


def _rectangles(code: str, **parameters):
    kernel = parser.parse_string(code, 'test.sptl')
    if parameters:
        kernel = passes.concretize_parameters(kernel, **parameters)
    kernel = passes.constexpr_propagation(kernel)
    kernel = s2c.canonicalize_kernel(kernel)
    return canonicalization.consolidate_rectangles_to_equivalence_classes(kernel)


def _turnaround(direction: str) -> str:
    """
    The ``.switches`` text for a router that stops receiving and starts sending in ``direction``.

    Changing both the input and the output direction takes two switch positions wherever a position
    carries only one of them.
    """
    if csl.SWITCH_POSITION_ALLOWS_BOTH:
        return '.pos1 = .{ .rx = RAMP, .tx = .{%s} }' % direction
    return '.pos1 = .{ .tx = .{%s} }, .pos2 = .{ .rx = RAMP }' % direction


###
# RouteConfig / ColorSwitchPlan
###


def test_single_configuration_emits_no_switches():
    plan = cslrouting.ColorSwitchPlan()
    plan.add(cslrouting.RouteConfig(('EAST', ), ('RAMP', )))
    assert plan.as_csl() == '.{ .routes = .{ .rx = .{EAST}, .tx = .{RAMP} } }'
    assert not plan.uses_switches


def test_identical_configurations_collapse():
    """Two streams that route identically through a PE consume a single switch position."""
    plan = cslrouting.ColorSwitchPlan()
    config = cslrouting.RouteConfig(('EAST', ), ('RAMP', ))
    plan.add(config)
    plan.add(cslrouting.RouteConfig(('EAST', ), ('RAMP', )))
    assert len(plan.positions) == 1
    assert not plan.uses_switches


def test_switch_positions_are_emitted():
    plan = cslrouting.ColorSwitchPlan()
    plan.add(cslrouting.RouteConfig(('RAMP', ), ('WEST', )))
    plan.add(cslrouting.RouteConfig(('EAST', ), ('WEST', )))
    # Only the side that changes is written: a switch position carries an input or an output
    assert plan.as_csl() == ('.{ .routes = .{ .rx = .{RAMP}, .tx = .{WEST} }, '
                             '.switches = .{ .pos1 = .{ .rx = EAST } } }')


def test_too_many_configurations_is_rejected():
    plan = cslrouting.ColorSwitchPlan()
    for direction in ('NORTH', 'SOUTH', 'EAST', 'WEST', 'RAMP'):
        plan.add(cslrouting.RouteConfig((direction, ), ('RAMP', )))
    with pytest.raises(SyntaxError, match='requires 5 switch positions'):
        plan.validate(0, 'PEs [2:3, 0:1]')


def test_both_sided_transition_becomes_two_positions():
    """
    A router that changes its input *and* its output direction cannot express that as one switch
    position on WSE-2, so it goes through an intermediate pure-relay configuration.
    """
    receive = cslrouting.RouteConfig(('EAST', ), ('RAMP', ))
    send = cslrouting.RouteConfig(('RAMP', ), ('WEST', ))
    positions, index_of = cslrouting.expand_positions([receive, send])

    if csl.SWITCH_POSITION_ALLOWS_BOTH:
        assert positions == [receive, send]
        assert index_of == [0, 1]
    else:
        assert positions == [receive, cslrouting.RouteConfig(('EAST', ), ('WEST', )), send]
        assert index_of == [0, 2]

        plan = cslrouting.ColorSwitchPlan()
        plan.add(receive)
        plan.add(send)
        # Each position names only the side it changes, and they compose incrementally.
        assert plan.as_csl() == ('.{ .routes = .{ .rx = .{EAST}, .tx = .{RAMP} }, '
                                 '.switches = .{ .pos1 = .{ .tx = .{WEST} }, .pos2 = .{ .rx = RAMP } } }')


def test_both_sided_transition_counts_against_capacity():
    """Two both-sided transitions take four positions, which exactly fills a router."""
    plan = cslrouting.ColorSwitchPlan()
    plan.add(cslrouting.RouteConfig(('EAST', ), ('RAMP', )))
    plan.add(cslrouting.RouteConfig(('RAMP', ), ('WEST', )))
    plan.add(cslrouting.RouteConfig(('NORTH', ), ('RAMP', )))

    if csl.SWITCH_POSITION_ALLOWS_BOTH:
        plan.validate(0, 'PEs [1:2, 0:1]')
    else:
        assert len(plan.hardware_positions) == 5
        with pytest.raises(SyntaxError, match='requires 5 switch positions'):
            plan.validate(0, 'PEs [1:2, 0:1]')


def test_exactly_four_configurations_is_accepted():
    plan = cslrouting.ColorSwitchPlan()
    for direction in ('NORTH', 'SOUTH', 'EAST', 'WEST'):
        plan.add(cslrouting.RouteConfig((direction, ), ('RAMP', )))
    plan.validate(0, 'PEs [2:3, 0:1]')
    assert plan.as_csl().count('.pos') == 3


def test_non_switchable_color_is_rejected(monkeypatch):
    """WSE-3 only implements switches on a subset of colors."""
    monkeypatch.setattr(csl, 'SWITCHABLE_COLORS', [0, 1, 2])
    plan = cslrouting.ColorSwitchPlan()
    plan.add(cslrouting.RouteConfig(('RAMP', ), ('WEST', )))
    plan.add(cslrouting.RouteConfig(('EAST', ), ('WEST', )))
    plan.validate(1, 'PEs [0:1, 0:1]')
    with pytest.raises(SyntaxError, match='needs router switches'):
        plan.validate(7, 'PEs [0:1, 0:1]')


###
# Control wavelets
###


def test_switch_advance_uses_the_single_command_payload():
    """
    The hardware applies a control wavelet's single command at every switch-configured router it
    reaches, so there is nothing to index per router.
    """
    assert cslrouting.switch_advance_payload() == \
        'ctrl.encode_single_payload(ctrl.opcode.SWITCH_ADV, true, {}, 0)'


def test_one_advance_emits_one_wavelet():
    assert cslrouting.switch_advance_statements('s_switch_dsd', 1) == \
        '@mov32(s_switch_dsd, ctrl.encode_single_payload(ctrl.opcode.SWITCH_ADV, true, {}, 0));'


def test_both_sided_transition_emits_two_wavelets():
    text = cslrouting.switch_advance_statements('s_switch_dsd', 2)
    assert text.count('@mov32(s_switch_dsd,') == 2
    assert text.count('\n') == 1


def test_empty_advance_is_rejected():
    with pytest.raises(ValueError, match='at least one position'):
        cslrouting.switch_advance_statements('s_switch_dsd', 0)


###
# two_phase_split: per-PE elision
###


def test_two_phase_split_switch_plans():
    """
    ``hop1`` and ``hop2`` share channel 0. Only the PEs whose configuration actually changes get a
    switch position:

    * PE 0 receives both streams identically -> one configuration, no switches.
    * PE 1 sends ``hop1`` and relays ``hop2`` -> two positions. The relay configuration comes from
      rectangle 2's loop, so this PE also proves that configurations are merged across rectangles.
    * PE 2 receives ``hop1`` and sends ``hop2`` -> two positions.
    * PE 3 only sends ``hop1`` -> one configuration, no switches.
    """
    files = _lower('two_phase_split.sptl', K=32)
    layout = files['layout.csl']
    configs = [line.strip() for line in layout.splitlines() if '@set_color_config' in line]
    assert len(configs) == 4

    with_switches = [line for line in configs if '.switches' in line]
    assert len(with_switches) == 2
    # PE 1 only changes where it receives from, which is one position on every architecture.
    assert any('.rx = .{RAMP}, .tx = .{WEST} }, .switches = .{ .pos1 = .{ .rx = EAST } }' in line
               for line in with_switches), layout
    # PE 2 turns around from receiving to sending, changing both sides at once.
    if csl.SWITCH_POSITION_ALLOWS_BOTH:
        turnaround = '.switches = .{ .pos1 = .{ .rx = RAMP, .tx = .{WEST} } }'
    else:
        turnaround = '.switches = .{ .pos1 = .{ .tx = .{WEST} }, .pos2 = .{ .rx = RAMP } }'
    assert any('.rx = .{EAST}, .tx = .{RAMP} }, ' + turnaround in line
               for line in with_switches), layout


def test_two_phase_split_emits_two_control_wavelets():
    """
    Of the six closes in the sample, only the two on PEs that *send* a stream whose path contains a
    router that must advance survive elision.
    """
    files = _lower('two_phase_split.sptl', K=32)
    emitting = {name: code for name, code in files.items() if 'switch_dsd' in code}
    assert sorted(emitting) == ['code_1_0.csl', 'code_3_0.csl']

    payload = 'ctrl.encode_single_payload(ctrl.opcode.SWITCH_ADV, true, {}, 0)'
    # PE 1 moves its own router one position; PE 3 retires PE 2's incoming configuration, and PE 2
    # has to turn around, which takes two positions where a switch carries only one direction.
    assert emitting['code_1_0.csl'].count(payload) == 1
    assert emitting['code_3_0.csl'].count(payload) == (1 if csl.SWITCH_POSITION_ALLOWS_BOTH else 2)
    for code in emitting.values():
        assert 'const ctrl = @import_module("<control>");' in code
        assert '.control = true' in code


def test_two_phase_split_without_switching_falls_back():
    """``--disable-switching`` emits one configuration per stream, as before switches existed."""
    kernel = parser.parse_file(os.path.join(SAMPLES, 'two_phase_split.sptl'))
    kernel = passes.constexpr_propagation(passes.concretize_parameters(kernel, K=32))
    layout = next(f.code for f in s2c.lower_spatial_ir_to_csl(kernel, disable_switching=True)
                  if f.filename == 'layout.csl')
    assert '.switches' not in layout
    # One call per configuration instead of one per PE: the later call silently wins, which is the
    # behaviour switches replace.
    assert len([line for line in layout.splitlines() if '@set_color_config' in line]) == 6


###
# Kernels that need no switching are unaffected
###


def test_auto_channel_kernels_emit_no_switches_and_no_control_wavelets():
    """
    Kernels from the GT4Py path use ``auto`` channels, one per stream declaration, so nothing is
    ever shared and code generation is exactly as it was before this feature.
    """
    code = """
    kernel @auto_channels<K>(stream<f32, K>[2] readonly inp) {
        place u16 i, u16 j in [0:2, 0:1] {
            f32[K] a
        }
        dataflow u16 i, u16 j in [0:2, 0:1] {
            stream<f32> eastwards = relative_stream(1, 0) {
                hops = auto,
                channel = auto
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
    files = _lower_string(code, K=4)
    for name, code in files.items():
        assert '.switches' not in code, name
        assert 'switch_dsd' not in code, name
        assert '<control>' not in code, name


def test_systolic_forwarding_gets_two_positions():
    """
    A PE that receives a stream and forwards it on the same channel needs two configurations, and
    the upstream PE's close is what advances it onto the second.
    """
    code = """
    kernel @chain<K>(stream<f32, K>[3] readonly inp, stream<f32, K> writeonly out) {
        place u16 i, u16 j in [0:3, 0:1] {
            f32[K] a
        }
        dataflow u16 i, u16 j in [0:3, 0:1] {
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
            await send(a, eastwards)
        }
        compute u16 i, u16 j in [2:3, 0:1] {
            await foreach i32 k, f32 x in [0:K], receive(eastwards) {
                a[k] = x
            }
            await send(a, out)
        }
    }
    """
    files = _lower_string(code, K=4)
    layout = files['layout.csl']
    middle = [line for line in layout.splitlines() if '@set_color_config' in line and '.switches' in line]
    assert len(middle) == 1, layout
    assert _turnaround('EAST') in middle[0], middle[0]

    # The first PE retires its outgoing configuration so that the middle PE advances to sending
    assert any('switch_dsd' in code for name, code in files.items() if name == 'code_0_0.csl')


def test_bounded_chain_sample_lowers_with_switches():
    """
    ``bounded_chain.sptl`` backs ``tests/csl_runtime/test_bounded_chain.sh``: a bounded stream that
    closes itself after its bound, which is what advances the forwarding PE's router.
    """
    files = _lower('bounded_chain.sptl', K=4)
    layout = files['layout.csl']
    switched = [line for line in layout.splitlines() if '@set_color_config' in line and '.switches' in line]
    assert len(switched) == 1, layout
    assert _turnaround('EAST') in switched[0], switched[0]

    # The head of the chain retires the incoming configuration for the PE that forwards
    assert 'ctrl.opcode.SWITCH_ADV' in files['code_0_0.csl']
    assert not any('switch_dsd' in code for name, code in files.items() if name == 'code_2_0.csl')


def test_scalar_reduce_1d_sample_lowers_with_switches():
    """
    ``scalar_reduce_1D.sptl`` used to carry a warning that the CSL backend could not lower it: every
    middle PE receives and sends on one channel. It backs
    ``tests/csl_runtime/test_scalar_reduce_1d.sh``.
    """
    path = os.path.join(os.path.dirname(__file__), '..', '..', 'samples', 'spatial', 'collectives')
    kernel = parser.parse_file(os.path.join(path, 'scalar_reduce_1D.sptl'))
    kernel = passes.constexpr_propagation(passes.concretize_parameters(kernel, N=4))
    files = {f.filename: f.code for f in s2c.lower_spatial_ir_to_csl(kernel)}
    assert '.switches' in files['layout.csl']


###
# Capacity stress
###

_STRESS_PHASES = [
    # (declaration, sender subgrid, receiver subgrid) -- all on channel 0, all crossing PE 2
    ('relative_stream(-1, 0) {{ hops = [(-1, 0)], channel = 0 }}', '3:4', '2:3'),
    ('relative_stream(-1, 0) {{ hops = [(-1, 0)], channel = 0 }}', '2:3', '1:2'),
    ('relative_stream(1, 0) {{ hops = [(1, 0)], channel = 0 }}', '1:2', '2:3'),
    ('relative_stream(1, 0) {{ hops = [(1, 0)], channel = 0 }}', '2:3', '3:4'),
    ('relative_stream(-2, 0) {{ hops = [(-1, 0), (-1, 0)], channel = 0 }}', '3:4', '1:2'),
]


def _stress_kernel(phases: int) -> str:
    body = ''
    for index, (declaration, sender, receiver) in enumerate(_STRESS_PHASES[:phases]):
        body += f"""
        phase {{
            dataflow u16 i, u16 j in [0:5, 0:1] {{
                stream<f32> s{index} = {declaration.format()}
            }}
            compute u16 i, u16 j in [{sender}, 0:1] {{
                await send(a, s{index})
                await s{index}.close()
            }}
            compute u16 i, u16 j in [{receiver}, 0:1] {{
                await foreach i32 k, f32 x in [0:K], receive(s{index}) {{
                    a[k] = x
                }}
                await s{index}.close()
            }}
        }}"""
    return f"""
    kernel @stress<K>(stream<f32, K>[5] readonly inp) {{
        place u16 i, u16 j in [0:5, 0:1] {{
            f32[K] a
        }}
{body}
    }}
    """


# Every phase of the stress kernel turns PE 2 around between receiving and sending, so on an
# architecture where a switch position carries a single direction each phase costs two positions.
_MAX_STRESS_PHASES = 4 if csl.SWITCH_POSITION_ALLOWS_BOTH else 2


@pytest.mark.parametrize('phases', [2, 3, 4])
def test_switch_positions_within_capacity(phases):
    """A router holds four switch positions; how many phases that is depends on the architecture."""
    if phases > _MAX_STRESS_PHASES:
        pytest.skip(f'{csl.ARCH} fits at most {_MAX_STRESS_PHASES} turnarounds in one router')
    files = _lower_string(_stress_kernel(phases), K=4)
    assert any('.switches' in code for code in files.values())


def test_switch_positions_beyond_capacity_are_rejected():
    """One configuration too many on a color has nowhere to go."""
    with pytest.raises(SyntaxError, match='switch positions'):
        _lower_string(_stress_kernel(_MAX_STRESS_PHASES + 1), K=4)


###
# bitonic_sort_1D: the heaviest channel reuse in the samples
###

_BITONIC = os.path.join(os.path.dirname(__file__), '..', '..', 'samples', 'spatial', 'sort',
                        'bitonic_sort_1D.sptl')


def _lower_bitonic(L: int, K: int = 4) -> dict[str, str]:
    kernel = parser.parse_file(_BITONIC)
    kernel = passes.constexpr_propagation(passes.concretize_parameters(kernel, L=L, K=K))
    return {f.filename: f.code for f in s2c.lower_spatial_ir_to_csl(kernel)}


@pytest.mark.skipif(not csl.SWITCH_POSITION_ALLOWS_BOTH,
                    reason=f'{csl.ARCH} cannot reverse a router within four switch positions')
def test_bitonic_sort_uses_one_channel_per_distance():
    """
    A bitonic network on 2^L keys needs L(L+1)/2 exchange steps but only L channels: one per
    exchange distance, reused by every lane, every stage and both directions of travel.

    L is 2 here because that is what the router budget allows: at distance 2^d the channel is
    reused by 2^d lanes in two directions each, so an interior router cycles through 2^(d+1)
    configurations, and four positions run out at d = 2.
    """
    files = _lower_bitonic(2)
    colors = set(re.findall(r'@get_color\((\d+)\)', files['layout.csl']))
    assert len(colors) <= 2, sorted(colors)

    layout = files['layout.csl']
    assert '.switches' in layout
    # The lane pattern repeats, so the configuration sequence closes into a ring.
    assert 'ring_mode' in layout
    # Every router stays inside its four positions.
    for line in layout.splitlines():
        if '.switches' in line:
            assert len(re.findall(r'\.pos\d', line)) < csl.SWITCH_POSITIONS, line


@pytest.mark.skipif(csl.SWITCH_POSITION_ALLOWS_BOTH,
                    reason='this architecture can reverse a router in a single switch position')
def test_bitonic_sort_is_rejected_on_wse2():
    """
    Reversing a router costs two positions where a position carries one direction, and the interior
    routers of the network reverse often enough to exhaust them.
    """
    with pytest.raises(SyntaxError, match='switch positions'):
        _lower_bitonic(2)


###
# odd_even_sort_1D_looped: N rounds as a runtime loop on four static channels
###

_ODD_EVEN_LOOPED = os.path.join(os.path.dirname(__file__), '..', '..', 'samples', 'spatial',
                                'sort', 'odd_even_sort_1D_looped.sptl')


def _lower_odd_even_looped(L: int, K: int = 4) -> dict[str, str]:
    kernel = parser.parse_file(_ODD_EVEN_LOOPED)
    kernel = passes.constexpr_propagation(passes.concretize_parameters(kernel, L=L, K=K))
    return {f.filename: f.code for f in s2c.lower_spatial_ir_to_csl(kernel, disable_benchmarking=True)}


def test_odd_even_sort_looped_uses_four_static_channels():
    """
    One channel per (round parity, direction). Roles never change, so no router switches and the
    rounds stay a CSL loop of N/2 iterations rather than N unrolled phases.
    """
    files = _lower_odd_even_looped(3)
    colors = set(int(c) for c in re.findall(r'@get_color\((\d+)\)', files['layout.csl']))
    assert colors == {0, 1, 2, 3}, sorted(colors)
    assert '.switches' not in files['layout.csl']

    # L = 1 drops the interior rectangles (N = 2 has only the two endpoints).
    ends = _lower_odd_even_looped(1, K=1)
    assert 'code_0_0.csl' in ends and 'code_1_0.csl' in ends
    assert 'code_2_0.csl' not in ends

    interior = files['code_2_0.csl']
    assert 'for (@range(i32, 0, 4, 1))' in interior, interior
    # The loop body is emitted once: two even-round transfers and two odd-round transfers, not
    # four copies of each for the four even/odd pairs at N = 8.
    assert interior.count('fabout_dsd') == 2, interior
    assert interior.count('fabin_dsd') == 2, interior


def test_odd_even_sort_looped_code_is_independent_of_n():
    """Lowering cost and the interior PE program stay flat as N grows."""
    small = _lower_odd_even_looped(3)
    large = _lower_odd_even_looped(6)
    assert 'for (@range(i32, 0, 32, 1))' in large['code_2_0.csl']
    # Same four PE roles, so the same number of code files; the loop trip count is the only
    # difference that scales with L.
    assert len(small) == len(large)
    assert abs(len(small['code_2_0.csl']) - len(large['code_2_0.csl'])) < 64


if __name__ == '__main__':
    pytest.main([__file__])
