"""
Router route configurations and switch planning for CSL code generation.

Each router holds, per color, one base route configuration plus up to three *switch positions*. A
router advances from one position to the next when a switch-advance control message passes through
it, which is what a stream ``close`` lowers to. This module owns:

* :class:`RouteConfig` -- one ``.rx``/``.tx`` pair, the configuration of one router for one color.
* :class:`ColorSwitchPlan` -- the ordered sequence of configurations a router cycles through.
* :func:`set_color_config` -- the single place where ``@set_color_config`` text is produced.
* :func:`switch_advance_payload` -- the ``<control>`` payload that advances a path's routers.

See ``irspec/docs/spatial/routing.md`` ("Lowering to Switches") for the semantics.
"""
from dataclasses import dataclass, field

from spada.syntax.csl import constants


@dataclass(frozen=True)
class RouteConfig:
    """
    The configuration of one router for one color: the directions it receives from and transmits to.

    ``RAMP`` denotes the PE's own compute element.
    """
    rx: tuple[str, ...]
    tx: tuple[str, ...]

    def as_csl(self) -> str:
        return '.{ .rx = .{%s}, .tx = .{%s} }' % (', '.join(self.rx), ', '.join(self.tx))

    def as_switch_position(self) -> str:
        """
        Renders this configuration as a ``.posN`` struct. The ``rx`` field of a switch position only
        accepts a single direction, unlike the base configuration.
        """
        if len(self.rx) != 1:
            raise ValueError(f'A switch position can only receive from a single direction, got {self.rx}')
        return '.{ .rx = %s, .tx = .{%s} }' % (self.rx[0], ', '.join(self.tx))


@dataclass
class ColorSwitchPlan:
    """
    The ordered route configurations a router cycles through for one color.

    ``positions[0]`` is the base configuration, and every further entry becomes a switch position.
    Consecutive identical configurations are collapsed by :meth:`add`, so a router that keeps the
    same configuration across an epoch boundary consumes no switch position and needs no advance.
    """
    positions: list[RouteConfig] = field(default_factory=list)
    ring_mode: bool = False

    def add(self, config: RouteConfig) -> None:
        if self.positions and self.positions[-1] == config:
            return
        self.positions.append(config)

    @property
    def uses_switches(self) -> bool:
        return len(self.positions) > 1

    def index_of_epoch(self, epoch: int) -> int:
        """
        Returns the switch position an epoch maps to, given that identical configurations collapse.
        """
        return min(epoch, len(self.positions) - 1)

    def as_csl(self) -> str:
        base = self.positions[0].as_csl()
        if not self.uses_switches:
            return '.{ .routes = %s }' % base

        switches = [
            '.pos%d = %s' % (index, config.as_switch_position())
            for index, config in enumerate(self.positions[1:], start=1)
        ]
        if self.ring_mode:
            switches.append('.ring_mode = true')
        return '.{ .routes = %s, .switches = .{ %s } }' % (base, ', '.join(switches))

    def validate(self, color: int, location: str) -> None:
        """
        Raises a ``SyntaxError`` if the plan exceeds what a router can hold.

        :param color: The color the plan is for, used for the diagnostic and for the WSE-3 check.
        :param location: A human-readable description of the PE the plan belongs to.
        """
        if len(self.positions) > constants.SWITCH_POSITIONS:
            raise SyntaxError(
                f'Color {color} at {location} requires {len(self.positions)} route configurations, '
                f'but a router holds at most {constants.SWITCH_POSITIONS} per color on '
                f'{constants.ARCH}.\n'
                '  note: assign a different channel to some of the streams, at the cost of an '
                'additional color')
        if self.uses_switches and color not in constants.SWITCHABLE_COLORS:
            raise SyntaxError(
                f'Color {color} at {location} needs router switches, but {constants.ARCH} only '
                f'supports switches on colors {constants.SWITCHABLE_COLORS}.\n'
                '  note: assign the channel to a switchable color')


def set_color_config(x: str, y: str, color: str, plan: ColorSwitchPlan, indent: str = '') -> str:
    """
    Renders a single ``@set_color_config`` call. This is the only place that produces such text.

    :param x: The PE x coordinate expression (e.g. ``pe_x`` or ``pe_x + -1``).
    :param y: The PE y coordinate expression.
    :param color: The color expression (e.g. ``@get_color(0)``).
    :param plan: The route configurations the router cycles through.
    :param indent: Indentation to prefix the line with.
    """
    return indent + '@set_color_config(%s, %s, %s, %s);\n' % (x, y, color, plan.as_csl())


def switch_advance_payload(commands: list[bool]) -> str:
    """
    Returns the ``<control>`` expression for a switch-advance control wavelet.

    One command is consumed per router the wavelet traverses, in order, so ``commands[i]`` says
    whether the ``i``-th router on the path advances. Routers whose configuration does not change
    are given a ``NOP`` so that they stay on their current position.

    :param commands: Per-router advance flags, starting at the sending PE's own router.
    """
    if not commands:
        raise ValueError('A switch advance needs at least one router command')
    if len(commands) > constants.MAX_CONTROL_COMMANDS:
        raise SyntaxError(
            f'A switch advance along this path needs {len(commands)} router commands, but a control '
            f'wavelet carries at most {constants.MAX_CONTROL_COMMANDS}.\n'
            '  note: shorten the routing path, or split it across two channels')

    if len(commands) == 1:
        opcode = 'ctrl.opcode.SWITCH_ADV' if commands[0] else 'ctrl.opcode.NOP'
        return f'ctrl.encode_single_payload({opcode}, true, {{}}, 0)'

    opcodes = ', '.join('ctrl.opcode.SWITCH_ADV' if advance else 'ctrl.opcode.NOP' for advance in commands)
    ce_ignore = ', '.join('true' for _ in commands)
    return ('ctrl.encode_payload(.{ .opcodes = .{%s}, .ce_ignore = .{%s}, '
            '.ce_ignore_remaining = true })' % (opcodes, ce_ignore))
