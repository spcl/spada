"""
Overlapping 1D interval shifts on one color.

A run of consecutive PEs each shifting the same distance ``d`` along an axis has overlapping paths:
the router of source ``p + 1`` carries source ``p``'s words. One color per direction still suffices,
because the routers can be time-multiplexed -- but only if every switch is triggered by something
the PE that owns it knows locally, since a control wavelet advances *every* switch-configured router
it reaches (see ``irspec/docs/spatial/routing.md``).

Send order is what provides that. The sources go nearest-the-destinations first, so a source's
router changes from injecting to relaying exactly when that source has finished its own send, which
it can signal itself with one switch-advance wavelet. Nothing needs to be told from a distance, and
the order enforces itself: a source further from the destinations cannot push a word through its
neighbour's router while that neighbour is still injecting, so it waits on the link.

The destinations do not switch at all. Each transmits to its ramp *and* onward, so all of them see
the whole stream and a counter filter decides which words each one keeps; the last one transmits to
its ramp alone and thereby takes the stream out of the network.

This is the arrangement Schnyder's 2D reduce-scatter uses ("Distributed Sorting on the Cerebras
Wafer-Scale Engine", fig. 7.6), and ``tests/csl_runtime/test_shift_bundle_filters.sh`` is a
hand-written version of it that pins down the hardware behaviour relied on here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from spada.syntax.spatial_ir import analysis, stream_lifetime
from spada.syntax.spatial_ir import irnodes as spir
from spada.syntax.spatial_ir.canonicalization import PEBlock
from spada.syntax.spatial_ir.grid_geometry import Rectangle

#: A shift of one PE needs no bundling: consecutive sources at distance 1 form a chain, whose
#: routers are sequenced by the ordinary receive-then-send switch positions.
MIN_BUNDLE_DISTANCE = 2


@dataclass(frozen=True)
class ShiftBundle:
    """
    One run of consecutive sources shifted onto an equally long run of destinations.

    The sources occupy ``[start, start + length)`` along ``axis``; source ``c`` sends to
    ``c + sign * dist``. ``length <= dist`` keeps the two runs apart, so no PE is both a source and a
    destination of the same bundle.
    """
    axis: Literal['x', 'y']
    sign: Literal[1, -1]
    start: int
    length: int
    dist: int
    #: Words each source sends, from the stream's bound. The filter windows are this wide.
    words: int
    #: The range of the *other* axis, as ``(start, stop, stride)``. Every PE in it runs an
    #: independent copy of the bundle with the same router configurations.
    cross: tuple[int, int, int]
    channel: int
    #: Routing identity of the stream, as :func:`stream_lifetime.stream_group_key` defines it.
    group: str

    def sources(self) -> tuple[int, int]:
        """The source run, as a half-open interval in ascending coordinates."""
        return self.start, self.start + self.length

    def destinations(self) -> tuple[int, int]:
        """The destination run, as a half-open interval in ascending coordinates."""
        first = self.start + self.sign * self.dist
        return first, first + self.length

    def relays(self) -> tuple[int, int]:
        """
        The PEs between the two runs that only pass the stream through, as a half-open interval.

        Empty when ``length == dist``, which is the densest a bundle gets.
        """
        if self.sign > 0:
            return self.sources()[1], self.destinations()[0]
        return self.destinations()[1], self.sources()[0]

    def destination_order(self) -> tuple[int, int]:
        """
        Returns ``(first, step)``: the destination the stream reaches first, and the step from one
        destination to the next along the direction of travel.

        The stream passes the destination run from the side it arrives on, and each destination sees
        the whole stream, so this is what maps a destination onto the words it should keep.
        """
        low, high = self.destinations()
        return (low, 1) if self.sign > 0 else (high - 1, -1)

    def describe(self) -> str:
        low, high = self.sources()
        direction = {('x', 1): 'east', ('x', -1): 'west',
                     ('y', 1): 'south', ('y', -1): 'north'}[(self.axis, self.sign)]
        return f'{self.axis} in [{low}:{high}] shifted {self.dist} {direction}'


def _straight_shift(declaration: spir.StreamDeclaration) -> Optional[tuple[Literal['x', 'y'], int]]:
    """
    Returns the axis and signed distance of a stream that runs straight along one axis.

    ``None`` for anything else: a stream that is not a relative one, that moves diagonally, or whose
    hop list does not walk the axis one PE at a time.
    """
    stream = declaration.stream
    if not isinstance(stream, spir.RelativeStreamDeclaration) or stream.routing is None:
        return None
    try:
        dx, dy = int(stream.dx.eval()), int(stream.dy.eval())
    except Exception:  # pragma: no cover - defensive: a non-constant offset
        return None
    if (dx == 0) == (dy == 0):
        return None
    axis: Literal['x', 'y'] = 'x' if dy == 0 else 'y'
    delta = dx if axis == 'x' else dy

    hops = stream.routing.hops
    if isinstance(hops, list):
        step = (1 if delta > 0 else -1)
        expected = [(step, 0)] * abs(delta) if axis == 'x' else [(0, step)] * abs(delta)
        if [hop.offset for hop in hops] != expected:
            return None
    return axis, delta


def _bound(declaration: spir.StreamDeclaration) -> Optional[int]:
    if declaration.dtype.bound is None:
        return None
    try:
        value = declaration.dtype.bound.eval()
    except Exception:  # pragma: no cover - defensive: a non-constant bound
        return None
    return value if isinstance(value, int) and value > 0 else None


def _consecutive_runs(values: set[int]) -> list[tuple[int, int]]:
    """Splits a set of coordinates into ``(start, length)`` runs of consecutive values."""
    if not values:
        return []
    runs: list[tuple[int, int]] = []
    ordered = sorted(values)
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value == previous + 1:
            previous = value
            continue
        runs.append((start, previous - start + 1))
        start = previous = value
    runs.append((start, previous - start + 1))
    return runs


def detect_shift_bundles(rectangles: list[Rectangle[PEBlock]]) -> list[ShiftBundle]:
    """
    Finds the interval shifts in a kernel whose paths overlap, and which therefore need bundling.

    Sources are collected per channel and shift, across rectangles: one logical shift is often
    declared by several compute blocks -- a sorting network's matchings at successive offsets, for
    instance -- and only their union shows which PEs form a consecutive run.

    A shift is bundled only if *every* run it decomposes into can be: at least two sources, and no
    longer than the shift distance, so that sources and destinations stay disjoint. A shift that
    fails this is left to the ordinary per-hop lowering, which reports the conflict if there is one.

    :param rectangles: The consolidated PE rectangles of the kernel, with channels already resolved.
    :return: The bundles, in a deterministic order.
    """
    # (channel, axis, signed distance, cross-axis range) -> (source coordinates, words, group)
    groups: dict[tuple[int, str, int, tuple[int, int, int]], tuple[set[int], int, str]] = {}
    for rect in rectangles:
        sends_recvs = analysis.sends_and_receives(rect.metadata.compute)
        for declaration in rect.metadata.dataflow.statements:
            sent, _received = sends_recvs.get(declaration.stream_name, (False, False))
            if not sent:
                continue
            shift = _straight_shift(declaration)
            words = _bound(declaration)
            if shift is None or words is None:
                continue
            axis, delta = shift
            if abs(delta) < MIN_BUNDLE_DISTANCE:
                continue
            channel = declaration.stream.routing.resolved_channel
            if channel == 'auto':
                continue

            along = rect.x_range if axis == 'x' else rect.y_range
            cross = rect.y_range if axis == 'x' else rect.x_range
            key = (channel, axis, delta, cross)
            coords, _, _ = groups.setdefault(key, (set(), words, stream_lifetime.stream_group_key(declaration)))
            coords.update(range(along[0], along[1], along[2]))

    bundles: list[ShiftBundle] = []
    for (channel, axis, delta, cross), (coords, words, group) in sorted(groups.items()):
        runs = _consecutive_runs(coords)
        if not all(2 <= length <= abs(delta) for _start, length in runs):
            continue
        for start, length in runs:
            bundles.append(ShiftBundle(axis=axis, sign=1 if delta > 0 else -1, start=start,
                                       length=length, dist=abs(delta), words=words, cross=cross,
                                       channel=channel, group=group))
    return bundles


def bundles_by_group(bundles: list[ShiftBundle]) -> dict[str, list[ShiftBundle]]:
    """
    Indexes bundles by the routing identity of their stream, for the route collector to look up.
    """
    grouped: dict[str, list[ShiftBundle]] = {}
    for bundle in bundles:
        grouped.setdefault(bundle.group, []).append(bundle)
    return grouped
