"""
Detect consecutive 1D interval shifts and schedule counted two-state color switching.

A stream with an explicit ``count = k`` whose senders form a consecutive interval
``[L, L+m)`` at distance ``d`` (with ``1 < m <= d``) can share one color: each PE
forwards a known number of waves, then injects or absorbs. ``count = auto`` is
unbounded and is never rewritten.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Literal

from spada.syntax.spatial_ir import analysis
from spada.syntax.spatial_ir import irnodes as spir


@dataclass(frozen=True)
class ShiftBundle:
    """A consecutive interval of sources shifted by an axis-aligned distance."""

    phase_index: int
    axis: Literal["x", "y"]
    sign: Literal[1, -1]
    start: int
    length: int
    dist: int
    count: int
    fixed: int
    stream_names: tuple[spir.Identifier, ...]
    # Physical channel, unique to this phase and direction. Detect leaves 0;
    # apply_shift_bundles assigns a per-phase pair (fwd, bwd).
    channel: int = 0


@dataclass(frozen=True)
class ColorScheduleStep:
    """One router config and how many fabric waves it stays active."""

    rx: str
    tx: str
    waves: int

    def as_pair(self) -> str:
        return f"{_short(self.rx)}->{_short(self.tx)}"


@dataclass(frozen=True)
class ColorSchedule:
    """Per-PE counted switch program for one phase and logical channel."""

    phase_index: int
    x: int
    y: int
    channel: int
    steps: tuple[ColorScheduleStep, ...]


@dataclass(frozen=True)
class SwitchAdvance:
    """
    SWITCH_ADV control wavelet sent after a non-last injector's data waves.

    Each downstream router pops one opcode (always_pop): the next source and
    the dest that just absorbed see SWITCH_ADV; hops in between see NOP.
    A control wavelet holds at most 8 opcodes, so ``dist`` must be <= 8.
    """

    channel: int
    axis: Literal["x", "y"]
    last_injector: int
    opcodes: tuple[str, ...]


def _short(port: str) -> str:
    return {"RAMP": "R", "EAST": "E", "WEST": "W", "NORTH": "N", "SOUTH": "S"}.get(port, port)


def _resolved_count(routing: spir.RoutingDeclaration | None) -> int | None:
    if routing is None:
        return None
    count = routing.resolved_count
    if count == "auto":
        return None
    if count < 1:
        raise ValueError(f"Routing count must be a positive integer, got {count}")
    return count


def _axis_offset(stream: spir.RelativeStreamDeclaration) -> tuple[Literal["x", "y"], int] | None:
    dx = stream.dx.eval()
    dy = stream.dy.eval()
    if not isinstance(dx, int) or not isinstance(dy, int):
        return None
    if dx != 0 and dy == 0:
        return "x", dx
    if dy != 0 and dx == 0:
        return "y", dy
    return None


def _hops_are_straight(stream: spir.RelativeStreamDeclaration, axis: str, delta: int) -> bool:
    routing = stream.routing
    if routing is None or routing.hops == "auto":
        return True
    if not isinstance(routing.hops, list):
        return False
    step = 1 if delta > 0 else -1
    expected = [(step, 0)] * abs(delta) if axis == "x" else [(0, step)] * abs(delta)
    actual = [hop.offset for hop in routing.hops]
    return actual == expected


def _block_points(block) -> list[tuple[int, int]]:
    x0, x1, y0, y1 = block.get_grid_rect()
    xs, ys = block.get_grid_stride()
    return [(x, y) for x in range(x0, x1, xs) for y in range(y0, y1, ys)]


def _consecutive_runs(values: list[int]) -> list[tuple[int, int]]:
    if not values:
        return []
    ordered = sorted(set(values))
    runs = []
    start = prev = ordered[0]
    for value in ordered[1:]:
        if value == prev + 1:
            prev = value
            continue
        runs.append((start, prev - start + 1))
        start = prev = value
    runs.append((start, prev - start + 1))
    return runs


def detect_shift_bundles(kernel: spir.Kernel) -> list[ShiftBundle]:
    """
    Find consecutive interval shifts with an explicit count in each phase.

    :param kernel: A kernel whose metaprogramming and auto-hops are already resolved,
                   and whose phases have not yet been inlined.
    """
    bundles: list[ShiftBundle] = []
    phase_index = 0
    for block in kernel.body:
        if not isinstance(block, spir.Phase):
            continue
        bundles.extend(_detect_in_phase(block, phase_index))
        phase_index += 1
    return bundles


def _detect_in_phase(phase: spir.Phase, phase_index: int) -> list[ShiftBundle]:
    decls: dict[spir.Identifier, spir.RelativeStreamDeclaration] = {}
    for dataflow in phase.dataflow:
        for stmt in dataflow.statements:
            stream = stmt.stream
            if not isinstance(stream, spir.RelativeStreamDeclaration):
                continue
            existing = decls.get(stmt.stream_name)
            if existing is None:
                decls[stmt.stream_name] = stream
                continue
            if (existing.dx.eval(), existing.dy.eval()) != (stream.dx.eval(), stream.dy.eval()):
                raise ValueError(
                    f'Stream "{stmt.stream_name.as_ir()}" is declared with conflicting offsets '
                    f"in the same phase."
                )

    # (axis, delta, k, fixed_coord) -> (source coords along the axis, stream names)
    groups: dict[tuple[str, int, int, int], tuple[set[int], set[spir.Identifier]]] = {}
    for compute in phase.compute:
        sent_recv = analysis.sends_and_receives(compute)
        for name, stream in decls.items():
            sent, _received = sent_recv.get(name, (False, False))
            if not sent:
                continue
            k = _resolved_count(stream.routing)
            if k is None:
                continue
            axis_delta = _axis_offset(stream)
            if axis_delta is None:
                continue
            axis, delta = axis_delta
            if not _hops_are_straight(stream, axis, delta):
                continue
            for x, y in _block_points(compute):
                if axis == "x":
                    key = (axis, delta, k, y)
                    coord = x
                else:
                    key = (axis, delta, k, x)
                    coord = y
                bucket = groups.get(key)
                if bucket is None:
                    bucket = (set(), set())
                    groups[key] = bucket
                bucket[0].add(coord)
                bucket[1].add(name)

    bundles: list[ShiftBundle] = []
    for (axis, delta, k, fixed), (coords, names) in groups.items():
        dist = abs(delta)
        sign: Literal[1, -1] = 1 if delta > 0 else -1
        for start, length in _consecutive_runs(list(coords)):
            if length < 2 or length > dist:
                continue
            bundles.append(
                ShiftBundle(
                    phase_index=phase_index,
                    axis=axis,
                    sign=sign,
                    start=start,
                    length=length,
                    dist=dist,
                    count=k,
                    fixed=fixed,
                    stream_names=tuple(names),
                )
            )
    return bundles


def schedule_counted_switch(bundle: ShiftBundle) -> list[ColorSchedule]:
    """
    Build the two-state per-PE schedule for one interval shift.

    East/south (+d): source half forwards then injects; dest half absorbs then forwards.
    West/north (−d): the same lemma with the opposite ports, sources on the far side.
    """
    if bundle.axis == "x":
        forward_rx, forward_tx = ("WEST", "EAST") if bundle.sign > 0 else ("EAST", "WEST")
        inject_tx = "EAST" if bundle.sign > 0 else "WEST"
        absorb_rx = "WEST" if bundle.sign > 0 else "EAST"

        def pe(coord: int) -> tuple[int, int]:
            return coord, bundle.fixed
    else:
        forward_rx, forward_tx = ("NORTH", "SOUTH") if bundle.sign > 0 else ("SOUTH", "NORTH")
        inject_tx = "SOUTH" if bundle.sign > 0 else "NORTH"
        absorb_rx = "NORTH" if bundle.sign > 0 else "SOUTH"

        def pe(coord: int) -> tuple[int, int]:
            return bundle.fixed, coord

    L = bundle.start
    m = bundle.length
    d = bundle.dist
    k = bundle.count
    schedules: dict[tuple[int, int], list[ColorScheduleStep]] = {}

    def add(coord: int, steps: list[ColorScheduleStep]) -> None:
        x, y = pe(coord)
        kept = [step for step in steps if step.waves > 0]
        if kept:
            schedules[(x, y)] = kept

    if bundle.sign > 0:
        for j in range(m):
            src = L + j
            add(src, [
                ColorScheduleStep(forward_rx, forward_tx, j * k),
                ColorScheduleStep("RAMP", inject_tx, k),
            ])
        for j in range(m):
            dest = L + d + j
            add(dest, [
                ColorScheduleStep(absorb_rx, "RAMP", k),
                ColorScheduleStep(forward_rx, forward_tx, (m - 1 - j) * k),
            ])
        for coord in range(L + m, L + d):
            add(coord, [ColorScheduleStep(forward_rx, forward_tx, m * k)])
    else:
        # Sources occupy [L, L+m); dests are at source - d.
        for j in range(m):
            src = L + j
            add(src, [
                ColorScheduleStep(forward_rx, forward_tx, (m - 1 - j) * k),
                ColorScheduleStep("RAMP", inject_tx, k),
            ])
        for j in range(m):
            dest = L - d + j
            add(dest, [
                ColorScheduleStep(absorb_rx, "RAMP", k),
                ColorScheduleStep(forward_rx, forward_tx, j * k),
            ])
        for coord in range(L - d + m, L):
            add(coord, [ColorScheduleStep(forward_rx, forward_tx, m * k)])

    return [
        ColorSchedule(bundle.phase_index, x, y, bundle.channel, tuple(steps))
        for (x, y), steps in sorted(schedules.items())
    ]


_MAX_SWITCH_CMDS = 8


def switch_advance_for_bundle(bundle: ShiftBundle) -> SwitchAdvance:
    """Build the opcode chain popped by each downstream hop (distance ``d``).

    The injecting PE uses ``no_pop``, so the first opcode is for its neighbor.
    """
    d = bundle.dist
    if d > _MAX_SWITCH_CMDS:
        raise ValueError(
            f"Counted switching encodes one opcode per hop and a control wavelet "
            f"holds at most {_MAX_SWITCH_CMDS} commands; got dist={d}."
        )
    opcodes = ("SWITCH_ADV",) + ("NOP",) * max(d - 2, 0) + ("SWITCH_ADV",)
    last_injector = bundle.start + bundle.length - 1 if bundle.sign > 0 else bundle.start
    return SwitchAdvance(bundle.channel, bundle.axis, last_injector, opcodes)


def _phase_has_explicit_count_relative(phase: spir.Phase) -> bool:
    for dataflow in phase.dataflow:
        for stmt in dataflow.statements:
            stream = stmt.stream
            if not isinstance(stream, spir.RelativeStreamDeclaration):
                continue
            if _resolved_count(stream.routing) is not None:
                return True
    return False


def _phase_channel_bases(kernel: spir.Kernel, bundles: list[ShiftBundle]) -> dict[int, int]:
    """
    Give each routed phase its own even/odd color pair.

    Wave quotas depend on the shift distance, so a counted two-state program
    cannot be reused on the same color in a later phase.
    """
    needed = {bundle.phase_index for bundle in bundles}
    phase_index = 0
    for block in kernel.body:
        if not isinstance(block, spir.Phase):
            continue
        if _phase_has_explicit_count_relative(block):
            needed.add(phase_index)
        phase_index += 1
    return {phase: 2 * i for i, phase in enumerate(sorted(needed))}


def apply_shift_bundles(kernel: spir.Kernel, bundles: list[ShiftBundle]) -> list[ColorSchedule]:
    """
    Mark bundled streams for counted switching, assign two colors per phase,
    and return the per-PE schedules.
    """
    bases = _phase_channel_bases(kernel, bundles)
    assigned = [
        replace(bundle, channel=bases[bundle.phase_index] + (0 if bundle.sign > 0 else 1))
        for bundle in bundles
    ]

    names_by_phase: dict[int, dict[spir.Identifier, int]] = defaultdict(dict)
    for bundle in assigned:
        for name in bundle.stream_names:
            names_by_phase[bundle.phase_index][name] = bundle.channel

    phase_index = 0
    for block in kernel.body:
        if not isinstance(block, spir.Phase):
            continue
        rewrite = names_by_phase.get(phase_index, {})
        if rewrite:
            for dataflow in block.dataflow:
                for stmt in dataflow.statements:
                    channel = rewrite.get(stmt.stream_name)
                    if channel is None or stmt.stream.routing is None:
                        continue
                    stmt.stream.routing.channel = channel
                    stmt.stream.routing.counted_switch = True
        phase_index += 1

    schedules: list[ColorSchedule] = []
    advances: list[SwitchAdvance] = []
    for bundle in assigned:
        schedules.extend(schedule_counted_switch(bundle))
        advances.append(switch_advance_for_bundle(bundle))
    _assign_unit_hop_channels(kernel, bases)
    kernel.switch_advances = advances
    return schedules


def _assign_unit_hop_channels(kernel: spir.Kernel, bases: dict[int, int]) -> None:
    """Assign this phase's fwd/bwd pair to explicit-count distance-1 streams."""
    phase_index = 0
    for block in kernel.body:
        if not isinstance(block, spir.Phase):
            continue
        base = bases.get(phase_index)
        phase_index += 1
        if base is None:
            continue
        for dataflow in block.dataflow:
            for stmt in dataflow.statements:
                stream = stmt.stream
                if not isinstance(stream, spir.RelativeStreamDeclaration) or stream.routing is None:
                    continue
                if stream.routing.counted_switch:
                    continue
                if _resolved_count(stream.routing) is None:
                    continue
                axis_delta = _axis_offset(stream)
                if axis_delta is None:
                    continue
                _axis, delta = axis_delta
                if abs(delta) != 1:
                    continue
                stream.routing.channel = base + (0 if delta > 0 else 1)


def coalesce_shift_bundles(kernel: spir.Kernel) -> list[ColorSchedule]:
    """
    Detect shift bundles, rewrite their channels, and attach schedules to ``kernel``.
    """
    bundles = detect_shift_bundles(kernel)
    schedules = apply_shift_bundles(kernel, bundles)
    kernel.shift_schedules = schedules
    return schedules
