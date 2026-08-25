#!/usr/bin/env python3
"""
Visualize the channel assignment of the 1D Batcher samples.

Versions
--------
  static   batcher_oddeven_1D.sptl. No router reconfiguration at all: a
           channel is only reused where every PE keeps its role on it, so each
           cell of the table holds a single switch position. The p = 1 stages
           get their own block, since a PE's role there depends on the stage;
           the p >= 2 stages of one distance d agree on roles and share:
             p = 1  : fwd 2*((d - 1) + r),           bwd fwd + 1
             p >= 2 : fwd 2*(n - 1) + 2*((d - 1) + r), bwd fwd + 1
  bundled  Each phase uses two colors, one per direction. Phases with d >= 2
           whose comparators form a run of at least two sources are a shift
           bundle: sources inject then relay (pos0 / pos1), destinations stay
           put and pick their word with a counter filter. Colors are not
           reused across phases. This is what the sample did before the filter
           budget capped it at L = 3; kept here for comparison.
  hybrid   batcher_oddeven_bundled_1D.sptl as it stands. Only the three widest
           phases bundle, which is exactly the three filters a PE has: the
           rule is 4*d >= n. The rest are routed per matching, on colors
           pooled across phases by the source residue mod 2d *and* the
           direction, so that a PE's role on a pooled color is the same in
           every phase that uses it and no router ever switches:
             fwd 2*((2*d - 2) + c),  bwd fwd + 1,  c = r or d + r
  wse3     batcher_oddeven_wse3_1D.sptl. Bundles the same three phases, and
           pools the rest by the origin's residue alone, dropping the
           agreement on direction:
             fwd (2*d - 2) + c_lo,   bwd (2*d - 2) + c_hi
           A comparator's two messages then share a pooled color whenever
           two phases at one distance meet on it, which halves how many
           colors a PE touches -- what WSE-3 counts, since a queue there
           stays bound to its color for the whole kernel. The price is that
           pooled routers switch: a source alternates R->E and R->W, a
           destination W->R and E->R, both one side at a time.

Views
-----
  network  Knuth-style sorting network. Concurrent matchings (same phase)
           occupy adjacent sub-columns so overlapping spans stay visible.
           Each comparator is two arrows: down = fwd (east, +d), up = bwd
           (west, -d), each colored by its channel.
  table    Per-PE @set_color_config, with switch positions resolved the way
           the version's target stores them: on WSE-2 a both-sides change is
           split through a relay intermediate, on WSE-3 it is one position.
           Stacked bands are pos0, pos1, … in that order.
           A destination filter is the small ``fN`` in the cell, N being
           the filter's init_counter; ``--k`` widens the windows the way K
           keys per PE do, which scales every init_counter by K.

Examples
--------
  python samples/spatial/sort/plot_batcher_routing.py --n 8
  python samples/spatial/sort/plot_batcher_routing.py --n 8 --version bundled --view table
  python samples/spatial/sort/plot_batcher_routing.py --n 8 --version static bundled --view table
  python samples/spatial/sort/plot_batcher_routing.py --n 8 --version hybrid --view table --k 4
  python samples/spatial/sort/plot_batcher_routing.py --n 16 --version wse3 --view table
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace

import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


VERSIONS = ("static", "bundled", "hybrid", "wse3")
MIN_BUNDLE_DISTANCE = 2
MIN_BUNDLE_LENGTH = 2

# The architecture each version is written for, which decides how a both-sides change is resolved.
TARGET_ARCH = {"static": "WSE-2", "bundled": "WSE-2", "hybrid": "WSE-2", "wse3": "WSE-3"}

TX_ORDER = ("RAMP", "EAST", "WEST")
DIR_LETTER = {"RAMP": "R", "EAST": "E", "WEST": "W"}


@dataclass(frozen=True)
class Matching:
    l: int
    p: int
    dist: int
    offset: int
    pairs: tuple[tuple[int, int], ...]
    fwd: int
    bwd: int


@dataclass(frozen=True)
class Phase:
    index: int
    l: int
    p: int
    dist: int
    matchings: tuple[Matching, ...]


@dataclass(frozen=True)
class Route:
    """One router configuration: receive from ``rx``, transmit to ``tx``."""

    rx: str
    tx: tuple[str, ...]

    def label(self) -> str:
        tx = "".join(DIR_LETTER[d] for d in self.tx)
        return f"{DIR_LETTER[self.rx]}→{tx}"


@dataclass(frozen=True)
class Cell:
    """Resolved hardware switch positions of one (PE, channel), plus its filter."""

    positions: tuple[Route, ...]
    filter_init: int | None = None


def fwd_channel_static(dist: int, offset: int, p: int, n: int) -> int:
    """Eastbound channel of matching ``(dist, offset)`` of a stage with the given ``p``.

    The p = 1 stages live in their own block of ``2*(n - 1)`` channels because a
    PE's role on such a channel depends on the stage; the p >= 2 stages of one
    distance agree on roles and so share a channel above that block.
    """
    block = 0 if p == 1 else 2 * (n - 1)
    return block + 2 * ((dist - 1) + offset)


def bwd_channel_static(dist: int, offset: int, p: int, n: int) -> int:
    return fwd_channel_static(dist, offset, p, n) + 1


def batcher_phases(n: int) -> list[Phase]:
    if n < 2 or n & (n - 1):
        raise ValueError(f"n must be a power of two >= 2, got {n}")
    log_n = int(math.log2(n))
    phases: list[Phase] = []
    index = 0
    for l in range(1, log_n + 1):
        dist = 1 << (l - 1)
        matchings = []
        for r in range(dist):
            pairs = tuple((i, i + dist) for i in range(r, n, 1 << l))
            matchings.append(
                Matching(
                    l, 1, dist, r, pairs,
                    fwd_channel_static(dist, r, 1, n),
                    bwd_channel_static(dist, r, 1, n),
                )
            )
        phases.append(Phase(index, l, 1, dist, tuple(matchings)))
        index += 1
        for p in range(2, l + 1):
            dist = 1 << (l - p)
            matchings = []
            for r in range(dist):
                pairs = []
                for b in range(0, n, 1 << l):
                    start = b + dist + r
                    stop = b + (1 << l) - dist + r
                    for lo in range(start, stop, 2 * dist):
                        pairs.append((lo, lo + dist))
                matchings.append(
                    Matching(
                        l, p, dist, r, tuple(pairs),
                        fwd_channel_static(dist, r, p, n),
                        bwd_channel_static(dist, r, p, n),
                    )
                )
            phases.append(Phase(index, l, p, dist, tuple(matchings)))
            index += 1
    return phases


def is_bundled(phase: Phase, n: int, version: str) -> bool:
    """Whether ``version`` serializes a phase's matchings onto one colour pair."""
    if version == "static" or phase.dist < MIN_BUNDLE_DISTANCE:
        return False
    if version == "bundled":
        return True
    return 4 * phase.dist >= n  # hybrid and wse3: the three widest phases, one per filter


def assign_channels(phases: list[Phase], version: str, n: int) -> list[Phase]:
    """Rewrite matching channels to match the sample of ``version``."""
    if version == "static":
        return phases
    log_n = int(math.log2(n))
    assigned = []
    for ph in phases:
        matchings = []
        for m in ph.matchings:
            low_residue = m.offset if ph.p == 1 else ph.dist + m.offset
            high_residue = (low_residue + ph.dist) % (2 * ph.dist)
            if version == "bundled":
                fwd, bwd = 2 * ph.index, 2 * ph.index + 1
            elif is_bundled(ph, n, version):
                fwd = n + 2 * (ph.l * (log_n + 1) + ph.p)
                bwd = fwd + 1
            elif version == "wse3":
                # Pooled by the origin's residue mod 2d alone: each direction takes the colour of
                # the partner that sends it, so the two phases at one distance meet on both.
                fwd = (2 * ph.dist - 2) + low_residue
                bwd = (2 * ph.dist - 2) + high_residue
            else:
                # Pooled by direction as well, which keeps every router configuration static.
                fwd = 2 * ((2 * ph.dist - 2) + low_residue)
                bwd = fwd + 1
            matchings.append(replace(m, fwd=fwd, bwd=bwd))
        assigned.append(replace(ph, matchings=tuple(matchings)))
    return _compact_channels(assigned)


def _compact_channels(phases: list[Phase]) -> list[Phase]:
    """
    Renumbers the channels densely, as the compiler's colour allocation does.

    A hand-written channel formula generally leaves gaps, and only the channels a kernel actually
    uses are given a colour, in ascending order. Renumbering here is what makes the drawn channel
    axis the colour axis of the emitted layout.
    """
    used = sorted({ch for ph in phases for m in ph.matchings for ch in (m.fwd, m.bwd)})
    color_of = {channel: index for index, channel in enumerate(used)}
    return [replace(ph, matchings=tuple(replace(m, fwd=color_of[m.fwd], bwd=color_of[m.bwd])
                                        for m in ph.matchings))
            for ph in phases]


def _channel_labels(phases: list[Phase], n_channels: int) -> list[str]:
    """
    Legend text for each channel: what it carries and which phases put it there.

    Read off the assignment rather than assumed, since which direction a channel carries is what
    the versions disagree about -- pooling by direction gives every channel one of them, pooling by
    origin gives the shared ones both.
    """
    distances: list[set[int]] = [set() for _ in range(n_channels)]
    directions: list[set[str]] = [set() for _ in range(n_channels)]
    users: list[list[str]] = [[] for _ in range(n_channels)]
    for ph in phases:
        for m in ph.matchings:
            if not m.pairs:
                continue
            for channel, direction in ((m.fwd, "fwd"), (m.bwd, "bwd")):
                distances[channel].add(ph.dist)
                directions[channel].add(direction)
                if f"l{ph.l}p{ph.p}" not in users[channel]:
                    users[channel].append(f"l{ph.l}p{ph.p}")

    labels = []
    for ch in range(n_channels):
        if not directions[ch]:
            labels.append(f"  ch {ch} (unused)")
            continue
        arrow = {("fwd", ): "↓", ("bwd", ): "↑"}.get(tuple(sorted(directions[ch])), "↕")
        kind = "+".join(sorted(directions[ch]))
        dist = ",".join(str(d) for d in sorted(distances[ch]))
        phase_text = " ".join(users[ch]) if len(users[ch]) <= 3 else f"{len(users[ch])} phases"
        labels.append(f"{arrow} ch {ch}: d={dist} {kind} ({phase_text})")
    return labels


def channel_count(phases: list[Phase]) -> int:
    used = [ch for ph in phases for m in ph.matchings for ch in (m.fwd, m.bwd)]
    return (max(used) + 1) if used else 0


def channel_color(channel: int, n_channels: int, cmap_name: str = "tab20"):
    cmap = plt.get_cmap(cmap_name)
    if n_channels <= 20:
        return cmap(channel % 20)
    return plt.get_cmap("gist_ncar")(channel / max(n_channels - 1, 1))


def _dir_arrow(ax, x: float, y_from: float, y_to: float, color) -> None:
    ax.annotate(
        "",
        xy=(x, y_to),
        xytext=(x, y_from),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6, mutation_scale=9, shrinkA=1.5, shrinkB=1.5),
        zorder=2,
    )


def _draw_network(ax, phases: list[Phase], n: int, version: str) -> None:
    """Draw concurrent matchings as sub-columns; fwd and bwd as offset arrows."""
    n_channels = channel_count(phases)
    slot = 0.38
    gap = 0.55
    dx = 0.06
    origins = []
    x = 0.0
    for ph in phases:
        origins.append(x)
        x += max(len(ph.matchings), 1) * slot + gap
    x_end = x - gap

    ax.set_xlim(-0.7, x_end + 0.4)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels([str(i) for i in range(n)])
    ax.set_ylabel("PE index")
    ax.set_xlabel("Phase (left arrow ↓ fwd, right arrow ↑ bwd)")
    ax.set_title(f"Batcher network ({version}), n={n}: downward = fwd (east), upward = bwd (west)")

    tick_pos = []
    tick_lab = []
    for ph, x0 in zip(phases, origins):
        n_m = max(len(ph.matchings), 1)
        width = n_m * slot
        ax.axvspan(x0 - 0.10, x0 + width - slot + 0.10, color="0.93", zorder=0)
        tick_pos.append(x0 + (n_m - 1) * slot / 2)
        tick_lab.append(f"l={ph.l}\np={ph.p}\nd={ph.dist}")

    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab, fontsize=8)

    for pe in range(n):
        ax.plot([-0.5, x_end + 0.2], [pe, pe], color="0.78", lw=0.8, zorder=1)

    cap = 0.05
    for ph, x0 in zip(phases, origins):
        for k, m in enumerate(ph.matchings):
            x = x0 + k * slot
            fwd_c = channel_color(m.fwd, n_channels)
            bwd_c = channel_color(m.bwd, n_channels)
            for lo, hi in m.pairs:
                x_fwd = x - dx
                x_bwd = x + dx
                _dir_arrow(ax, x_fwd, lo, hi, fwd_c)
                _dir_arrow(ax, x_bwd, hi, lo, bwd_c)
                ax.plot([x_fwd - cap, x_fwd + cap], [lo, lo], color=fwd_c, lw=1.6, zorder=3)
                ax.plot([x_fwd - cap, x_fwd + cap], [hi, hi], color=fwd_c, lw=1.6, zorder=3)
                ax.plot([x_bwd - cap, x_bwd + cap], [lo, lo], color=bwd_c, lw=1.6, zorder=3)
                ax.plot([x_bwd - cap, x_bwd + cap], [hi, hi], color=bwd_c, lw=1.6, zorder=3)

    handles = [
        Line2D([0], [0], color=channel_color(ch, n_channels), lw=2.0, label=label)
        for ch, label in enumerate(_channel_labels(phases, n_channels))
    ]
    ax.legend(
        handles=handles,
        title="channel",
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=7,
        ncol=1 if n_channels <= 16 else 2,
    )


ROUTE_COLOR = {
    "R→E": "#1f77b4",
    "W→E": "#ff7f0e",
    "W→R": "#2ca02c",
    "W→RE": "#9467bd",
    "R→W": "#5fa8d3",
    "E→W": "#f4a261",
    "E→R": "#6dce6d",
    "E→RW": "#c77dff",
}
ROUTE_ORDER = ("R→E", "W→E", "W→RE", "W→R", "R→W", "E→W", "E→RW", "E→R")


def _tx(*dirs: str) -> tuple[str, ...]:
    wanted = set(dirs)
    return tuple(d for d in TX_ORDER if d in wanted)


def _add(seq: list[Route], config: Route) -> None:
    if not seq or seq[-1] != config:
        seq.append(config)


def _resolve_hardware(configs: list[Route], split_both_sides: bool = True) -> tuple[Route, ...]:
    """
    The positions the hardware stores for a sequence of configurations.

    On WSE-2 a position names one side, so a change of both is split through an intermediate that
    keeps the old input; WSE-3 takes both in one position and needs no splitting.
    """
    if not configs:
        return ()
    positions = [configs[0]]
    for config in configs[1:]:
        previous = positions[-1]
        if split_both_sides and previous.rx != config.rx and previous.tx != config.tx:
            positions.append(Route(previous.rx, config.tx))
        positions.append(config)
    return tuple(positions)


def _shift_runs(pairs: tuple[tuple[int, int], ...], dist: int) -> list[tuple[int, int]]:
    """Group ``(lo, lo+dist)`` pairs into maximal consecutive source runs ``(start, length)``."""
    sources = sorted(lo for lo, hi in pairs if hi - lo == dist)
    runs: list[tuple[int, int]] = []
    i = 0
    while i < len(sources):
        start = sources[i]
        length = 1
        while i + length < len(sources) and sources[i + length] == start + length:
            length += 1
        runs.append((start, length))
        i += length
    return runs


def _install_hop(configs: list[list[list[Route]]], pe: int, ch: int, route: Route) -> None:
    _add(configs[pe][ch], route)


def _install_ordinary_pair(configs: list[list[list[Route]]], lo: int, hi: int, fwd: int, bwd: int) -> None:
    _install_hop(configs, lo, fwd, Route("RAMP", _tx("EAST")))
    _install_hop(configs, hi, fwd, Route("WEST", _tx("RAMP")))
    for mid in range(lo + 1, hi):
        _install_hop(configs, mid, fwd, Route("WEST", _tx("EAST")))
    _install_hop(configs, hi, bwd, Route("RAMP", _tx("WEST")))
    _install_hop(configs, lo, bwd, Route("EAST", _tx("RAMP")))
    for mid in range(lo + 1, hi):
        _install_hop(configs, mid, bwd, Route("EAST", _tx("WEST")))


def _window_init(pe: int, first: int, step: int, words: int) -> int:
    """``init_counter`` of a destination filter keeping ``words`` of them, as ``_window_start`` emits it."""
    offset = 1 - first if step > 0 else first + 1
    return (pe + offset if step > 0 else offset - pe) * words


def _install_bundle(
    configs: list[list[list[Route]]],
    filters: list[list[int | None]],
    start: int,
    length: int,
    dist: int,
    fwd: int,
    bwd: int,
    words: int,
) -> None:
    """Eastbound inject-then-relay plus westbound mirror, with destination filters."""
    src_lo, src_hi = start, start + length
    dst_lo, dst_hi = start + dist, start + dist + length

    for pe in range(src_lo, src_hi):
        _install_hop(configs, pe, fwd, Route("RAMP", _tx("EAST")))
        _install_hop(configs, pe, fwd, Route("WEST", _tx("EAST")))
    for pe in range(src_hi, dst_lo):
        _install_hop(configs, pe, fwd, Route("WEST", _tx("EAST")))
    # Stream travels east: last destination terminates, the others copy-and-forward.
    for pe in range(dst_lo, dst_hi - 1):
        _install_hop(configs, pe, fwd, Route("WEST", _tx("RAMP", "EAST")))
        filters[pe][fwd] = _window_init(pe, dst_lo, 1, words)
    _install_hop(configs, dst_hi - 1, fwd, Route("WEST", _tx("RAMP")))
    filters[dst_hi - 1][fwd] = 0

    for pe in range(dst_lo, dst_hi):
        _install_hop(configs, pe, bwd, Route("RAMP", _tx("WEST")))
        _install_hop(configs, pe, bwd, Route("EAST", _tx("WEST")))
    for pe in range(src_hi, dst_lo):
        _install_hop(configs, pe, bwd, Route("EAST", _tx("WEST")))
    # Stream travels west: lowest destination terminates.
    for pe in range(src_lo + 1, src_hi):
        _install_hop(configs, pe, bwd, Route("EAST", _tx("RAMP", "WEST")))
        filters[pe][bwd] = _window_init(pe, src_hi - 1, -1, words)
    _install_hop(configs, src_lo, bwd, Route("EAST", _tx("RAMP")))
    filters[src_lo][bwd] = 0


def pe_table(phases: list[Phase], n: int, version: str, words: int = 1) -> list[list[Cell]]:
    """Build the resolved per-PE switch table of ``version``, for ``words`` keys per PE."""
    n_channels = channel_count(phases)
    configs: list[list[list[Route]]] = [[[] for _ in range(n_channels)] for _ in range(n)]
    filters: list[list[int | None]] = [[None] * n_channels for _ in range(n)]

    for ph in phases:
        pairs = tuple(pair for m in ph.matchings for pair in m.pairs)
        if not pairs:
            continue
        if is_bundled(ph, n, version):
            fwd, bwd = ph.matchings[0].fwd, ph.matchings[0].bwd
            for start, length in _shift_runs(pairs, ph.dist):
                if length >= MIN_BUNDLE_LENGTH:
                    _install_bundle(configs, filters, start, length, ph.dist, fwd, bwd, words)
                else:
                    _install_ordinary_pair(configs, start, start + ph.dist, fwd, bwd)
        else:
            for m in ph.matchings:
                for lo, hi in m.pairs:
                    _install_ordinary_pair(configs, lo, hi, m.fwd, m.bwd)

    split = TARGET_ARCH[version] == "WSE-2"
    return [
        [Cell(_resolve_hardware(configs[pe][ch], split), filters[pe][ch]) for ch in range(n_channels)]
        for pe in range(n)
    ]


def _draw_table(ax, phases: list[Phase], n: int, version: str, words: int) -> None:
    """Resolved per-PE switch positions; destination filters as a small ``fN``."""
    table = pe_table(phases, n, version, words)
    n_channels = channel_count(phases)
    ax.set_xlim(-0.5, n_channels - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xticks(range(n_channels))
    ax.set_yticks(range(n))
    ax.set_xlabel("Channel")
    ax.set_ylabel("PE")
    ax.set_title(
        f"Resolved switch positions ({version}, {TARGET_ARCH[version]}), n={n}, K={words} "
        f"({n_channels} colors); stacked bands are pos0, pos1, ...; fN is the filter init_counter"
    )
    ax.set_aspect("equal")

    for pe in range(n):
        for ch in range(n_channels):
            cell = table[pe][ch]
            if not cell.positions:
                ax.add_patch(
                    Rectangle(
                        (ch - 0.45, pe - 0.45),
                        0.9,
                        0.9,
                        facecolor="#f4f4f4",
                        edgecolor="0.85",
                        lw=0.4,
                    )
                )
                continue
            n_pos = len(cell.positions)
            # Leave a thin strip at the bottom when a filter is present.
            usable = 0.78 if cell.filter_init is not None else 0.9
            band = usable / n_pos
            fontsize = 6 if n_pos == 1 else 5
            for i, route in enumerate(cell.positions):
                y0 = pe - 0.45 + i * band
                label = route.label()
                ax.add_patch(
                    Rectangle(
                        (ch - 0.45, y0),
                        0.9,
                        band,
                        facecolor=ROUTE_COLOR.get(label, "#bbbbbb"),
                        edgecolor="0.85",
                        lw=0.4,
                    )
                )
                text = f"{i}:{label}" if n_pos > 1 else label
                ax.text(ch, y0 + band / 2, text, ha="center", va="center", fontsize=fontsize, color="0.1")
            if cell.filter_init is not None:
                ax.text(
                    ch,
                    pe + 0.38,
                    f"f{cell.filter_init}",
                    ha="center",
                    va="center",
                    fontsize=5,
                    color="0.15",
                    fontweight="bold",
                )

    handles = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=ROUTE_COLOR[p], markersize=10, label=p)
        for p in ROUTE_ORDER
        if any(route.label() == p for row in table for cell in row for route in cell.positions)
    ]
    if any(cell.filter_init is not None for row in table for cell in row):
        handles.append(
            Line2D([0], [0], marker="$f$", color="0.15", markerfacecolor="w", markersize=10, label="fN filter init")
        )
    ax.legend(handles=handles, title="rx→tx", loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)


def plot(n: int, view: str, version: str, words: int, outfile: str | None, show: bool) -> None:
    phases = assign_channels(batcher_phases(n), version, n)
    if view == "network":
        n_slots = sum(max(len(ph.matchings), 1) for ph in phases)
        fig, ax = plt.subplots(figsize=(max(8, 0.7 * n_slots + 0.8 * len(phases)), max(4, 0.45 * n)))
        _draw_network(ax, phases, n, version)
    elif view == "table":
        n_channels = channel_count(phases)
        fig, ax = plt.subplots(figsize=(max(8, 0.45 * n_channels), max(4, 0.5 * n)))
        _draw_table(ax, phases, n, version, words)
    else:
        raise ValueError(f"unknown view {view}")

    fig.tight_layout()
    if outfile:
        fig.savefig(outfile, bbox_inches="tight")
        if outfile.endswith(".pdf"):
            fig.savefig(outfile[:-4] + ".png", dpi=160, bbox_inches="tight")
        print(f"wrote {outfile}")
    if show:
        plt.show()
    plt.close(fig)


def _expand_versions(requested: list[str]) -> list[str]:
    if "all" in requested:
        return list(VERSIONS)
    # Preserve order, drop duplicates.
    seen: set[str] = set()
    versions = []
    for version in requested:
        if version not in seen:
            seen.add(version)
            versions.append(version)
    return versions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=8, help="number of PEs, power of two (default 8)")
    parser.add_argument(
        "--k",
        type=int,
        default=1,
        help="keys per PE (default 1). A comparator trades K of them, so a bundle's cycle is "
        "K times longer and every destination filter starts K times further back.",
    )
    parser.add_argument(
        "--version",
        nargs="+",
        choices=VERSIONS + ("all",),
        default=["static"],
        help="static (one color per matching), bundled (two per phase), hybrid (the WSE-2 sample: "
        "the three widest phases bundled, the rest pooled by direction and origin) and/or wse3 "
        "(the WSE-3 sample: the same bundles, the rest pooled by origin alone, so the pooled "
        "routers switch). 'all' is every one. Repeatable.",
    )
    parser.add_argument(
        "--view",
        choices=("network", "table"),
        default="network",
        help="network: sorting-network comparators; table: resolved switch positions and filters",
    )
    parser.add_argument("--out", default=None, help="output PDF path (version is inserted before the extension if several)")
    parser.add_argument("--show", action="store_true", help="open an interactive window")
    args = parser.parse_args()
    versions = _expand_versions(args.version)
    for version in versions:
        outfile = args.out
        if outfile is None and not args.show:
            suffix = "" if args.k == 1 else f"_k{args.k}"
            outfile = f"samples/spatial/sort/batcher_routing_{version}_n{args.n}{suffix}_{args.view}.pdf"
        elif outfile is not None and len(versions) > 1:
            if outfile.endswith(".pdf"):
                outfile = f"{outfile[:-4]}_{version}.pdf"
            else:
                outfile = f"{outfile}_{version}"
        plot(args.n, args.view, version, args.k, outfile, args.show)


if __name__ == "__main__":
    main()
