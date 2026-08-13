"""
Tests for router switch planning: route configuration merging, switch positions, the control
wavelets that retire a configuration, and the hardware capacity limits.
"""
import os

import pytest

from spada.lowering import spatial_ir_to_csl as s2c
from spada.syntax.csl import constants as csl, switching as cslswitch
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


###
# RouteConfig / ColorSwitchPlan
###


def test_single_configuration_emits_no_switches():
    plan = cslswitch.ColorSwitchPlan()
    plan.add(cslswitch.RouteConfig(('EAST', ), ('RAMP', )))
    assert plan.as_csl() == '.{ .routes = .{ .rx = .{EAST}, .tx = .{RAMP} } }'
    assert not plan.uses_switches


def test_identical_configurations_collapse():
    """Two streams that route identically through a PE consume a single switch position."""
    plan = cslswitch.ColorSwitchPlan()
    config = cslswitch.RouteConfig(('EAST', ), ('RAMP', ))
    plan.add(config)
    plan.add(cslswitch.RouteConfig(('EAST', ), ('RAMP', )))
    assert len(plan.positions) == 1
    assert not plan.uses_switches


def test_switch_positions_are_emitted():
    plan = cslswitch.ColorSwitchPlan()
    plan.add(cslswitch.RouteConfig(('RAMP', ), ('WEST', )))
    plan.add(cslswitch.RouteConfig(('EAST', ), ('WEST', )))
    assert plan.as_csl() == ('.{ .routes = .{ .rx = .{RAMP}, .tx = .{WEST} }, '
                             '.switches = .{ .pos1 = .{ .rx = EAST, .tx = .{WEST} } } }')


def test_too_many_configurations_is_rejected():
    plan = cslswitch.ColorSwitchPlan()
    for direction in ('NORTH', 'SOUTH', 'EAST', 'WEST', 'RAMP'):
        plan.add(cslswitch.RouteConfig((direction, ), ('RAMP', )))
    with pytest.raises(SyntaxError, match='requires 5 route configurations'):
        plan.validate(0, 'PEs [2:3, 0:1]')


def test_exactly_four_configurations_is_accepted():
    plan = cslswitch.ColorSwitchPlan()
    for direction in ('NORTH', 'SOUTH', 'EAST', 'WEST'):
        plan.add(cslswitch.RouteConfig((direction, ), ('RAMP', )))
    plan.validate(0, 'PEs [2:3, 0:1]')
    assert plan.as_csl().count('.pos') == 3


def test_non_switchable_color_is_rejected(monkeypatch):
    """WSE-3 only implements switches on a subset of colors."""
    monkeypatch.setattr(csl, 'SWITCHABLE_COLORS', [0, 1, 2])
    plan = cslswitch.ColorSwitchPlan()
    plan.add(cslswitch.RouteConfig(('RAMP', ), ('WEST', )))
    plan.add(cslswitch.RouteConfig(('EAST', ), ('WEST', )))
    plan.validate(1, 'PEs [0:1, 0:1]')
    with pytest.raises(SyntaxError, match='needs router switches'):
        plan.validate(7, 'PEs [0:1, 0:1]')


###
# Control wavelets
###


def test_single_router_advance_uses_the_single_payload_helper():
    assert cslswitch.switch_advance_payload([True]) == \
        'ctrl.encode_single_payload(ctrl.opcode.SWITCH_ADV, true, {}, 0)'


def test_routers_that_keep_their_configuration_get_a_nop():
    payload = cslswitch.switch_advance_payload([False, True])
    assert '.opcodes = .{ctrl.opcode.NOP, ctrl.opcode.SWITCH_ADV}' in payload


def test_path_longer_than_the_control_wavelet_is_rejected():
    commands = [True] * (csl.MAX_CONTROL_COMMANDS + 1)
    with pytest.raises(SyntaxError, match='at most 8'):
        cslswitch.switch_advance_payload(commands)


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
    assert any('.rx = .{RAMP}, .tx = .{WEST} }, .switches = .{ .pos1 = .{ .rx = EAST, .tx = .{WEST} } }' in line
               for line in with_switches), layout
    assert any('.rx = .{EAST}, .tx = .{RAMP} }, .switches = .{ .pos1 = .{ .rx = RAMP, .tx = .{WEST} } }' in line
               for line in with_switches), layout


def test_two_phase_split_emits_two_control_wavelets():
    """
    Of the six closes in the sample, only the two on PEs that *send* a stream whose path contains a
    router that must advance survive elision.
    """
    files = _lower('two_phase_split.sptl', K=32)
    emitting = {name: code for name, code in files.items() if 'switch_dsd' in code}
    assert sorted(emitting) == ['code_1_0.csl', 'code_3_0.csl']

    # PE 1 advances its own router, PE 3 advances the router of its receiver
    assert '.opcodes = .{ctrl.opcode.SWITCH_ADV, ctrl.opcode.NOP}' in emitting['code_1_0.csl']
    assert '.opcodes = .{ctrl.opcode.NOP, ctrl.opcode.SWITCH_ADV}' in emitting['code_3_0.csl']
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
    assert '.rx = .{WEST}, .tx = .{RAMP} }, .switches = .{ .pos1 = .{ .rx = RAMP, .tx = .{EAST} } }' in middle[0]

    # The first PE retires its outgoing configuration so that the middle PE advances to sending
    assert any('switch_dsd' in code for name, code in files.items() if name == 'code_0_0.csl')


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


@pytest.mark.parametrize('phases', [2, 3, 4])
def test_switch_positions_within_capacity(phases):
    """Up to four route configurations fit in a router."""
    files = _lower_string(_stress_kernel(phases), K=4)
    assert any('.switches' in code for code in files.values())


def test_switch_positions_beyond_capacity_are_rejected():
    """A fifth distinct configuration on one color has nowhere to go."""
    with pytest.raises(SyntaxError, match='route configurations'):
        _lower_string(_stress_kernel(5), K=4)


if __name__ == '__main__':
    pytest.main([__file__])
