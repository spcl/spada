#!/usr/bin/env python3
"""
Visualize the static channel assignment used by batcher_oddeven_1D.sptl.

At distance d, offset r in [0, d) uses:
  fwd (east,  +d) : channel 2*((d - 1) + r)
  bwd (west,  -d) : channel 2*((d - 1) + r) + 1

Stages that share the same d reuse those colors. Total colors = 2*(n - 1).

Views
-----
  network  Knuth-style sorting network. Concurrent matchings (same phase)
           occupy adjacent sub-columns so overlapping spans stay visible.
           Each comparator is two arrows: down = fwd (east, +d), up = bwd
           (west, -d), each colored by its own channel.
  table    Per-PE color table of the static @set_color_config. Each cell is
           the union of rx→tx pairs on that (PE, channel); a reused color
           may install both RAMP→EAST and WEST→RAMP on the same PE.

Examples
--------
  python samples/spatial/sort/plot_batcher_routing.py --n 8
  python samples/spatial/sort/plot_batcher_routing.py --n 8 --view table --show
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


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


def fwd_channel(dist: int, offset: int) -> int:
    return 2 * ((dist - 1) + offset)


def bwd_channel(dist: int, offset: int) -> int:
    return fwd_channel(dist, offset) + 1


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
                Matching(l, 1, dist, r, pairs, fwd_channel(dist, r), bwd_channel(dist, r))
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
                    Matching(l, p, dist, r, tuple(pairs), fwd_channel(dist, r), bwd_channel(dist, r))
                )
            phases.append(Phase(index, l, p, dist, tuple(matchings)))
            index += 1
    return phases


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


def _draw_network(ax, phases: list[Phase], n: int) -> None:
    """Draw concurrent matchings as sub-columns; fwd and bwd as offset arrows."""
    n_channels = 2 * (n - 1)
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
    ax.set_title(f"Batcher network, n={n}: downward = fwd (east), upward = bwd (west)")

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

    handles = []
    for ch in range(n_channels):
        arrow = "↓" if ch % 2 == 0 else "↑"
        kind = "fwd" if ch % 2 == 0 else "bwd"
        handles.append(
            Line2D(
                [0],
                [0],
                color=channel_color(ch, n_channels),
                lw=2.0,
                label=f"{arrow} ch {ch} ({kind})",
            )
        )
    ax.legend(
        handles=handles,
        title="channel",
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=7,
        ncol=1 if n_channels <= 16 else 2,
    )


PAIR_ORDER = ("R→E", "W→E", "W→R", "R→W", "E→W", "E→R")
PAIR_COLOR = {
    "R→E": "#1f77b4",
    "W→E": "#ff7f0e",
    "W→R": "#2ca02c",
    "R→W": "#5fa8d3",
    "E→W": "#f4a261",
    "E→R": "#6dce6d",
}


def _draw_table(ax, phases: list[Phase], n: int) -> None:
    """Static per-PE @set_color_config: each cell is the union of rx→tx pairs."""
    n_channels = 2 * (n - 1)
    routes: list[list[set[str]]] = [[set() for _ in range(n_channels)] for _ in range(n)]
    for ph in phases:
        for m in ph.matchings:
            for lo, hi in m.pairs:
                routes[lo][m.fwd].add("R→E")
                routes[hi][m.fwd].add("W→R")
                for mid in range(lo + 1, hi):
                    routes[mid][m.fwd].add("W→E")
                routes[hi][m.bwd].add("R→W")
                routes[lo][m.bwd].add("E→R")
                for mid in range(lo + 1, hi):
                    routes[mid][m.bwd].add("E→W")

    ax.set_xlim(-0.5, n_channels - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xticks(range(n_channels))
    ax.set_yticks(range(n))
    ax.set_xlabel("Channel")
    ax.set_ylabel("PE")
    ax.set_title(f"Static rx→tx table, n={n} ({n_channels} colors); stacked pairs are a union")
    ax.set_aspect("equal")

    for pe in range(n):
        for ch in range(n_channels):
            pairs = [p for p in PAIR_ORDER if p in routes[pe][ch]]
            if not pairs:
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
            band = 0.9 / len(pairs)
            fontsize = 6 if len(pairs) == 1 else 5
            for i, pair in enumerate(pairs):
                y0 = pe - 0.45 + i * band
                ax.add_patch(
                    Rectangle(
                        (ch - 0.45, y0),
                        0.9,
                        band,
                        facecolor=PAIR_COLOR[pair],
                        edgecolor="0.85",
                        lw=0.4,
                    )
                )
                ax.text(ch, y0 + band / 2, pair, ha="center", va="center", fontsize=fontsize, color="0.1")

    handles = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=PAIR_COLOR[p], markersize=10, label=p)
        for p in PAIR_ORDER
    ]
    ax.legend(handles=handles, title="rx→tx", loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)


def plot(n: int, view: str, outfile: str | None, show: bool) -> None:
    phases = batcher_phases(n)
    if view == "network":
        n_slots = sum(max(len(ph.matchings), 1) for ph in phases)
        fig, ax = plt.subplots(figsize=(max(8, 0.7 * n_slots + 0.8 * len(phases)), max(4, 0.45 * n)))
        _draw_network(ax, phases, n)
    elif view == "table":
        fig, ax = plt.subplots(figsize=(max(8, 0.45 * 2 * (n - 1)), max(4, 0.45 * n)))
        _draw_table(ax, phases, n)
    else:
        raise ValueError(f"unknown view {view}")

    fig.tight_layout()
    if outfile:
        fig.savefig(outfile, bbox_inches="tight")
        png = outfile[:-4] + ".png" if outfile.endswith(".pdf") else outfile + ".png"
        if outfile.endswith(".pdf"):
            fig.savefig(png, dpi=160, bbox_inches="tight")
        print(f"wrote {outfile}")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=8, help="number of PEs, power of two (default 8)")
    parser.add_argument(
        "--view",
        choices=("network", "table"),
        default="network",
        help="network: sorting-network comparators; table: static color config",
    )
    parser.add_argument("--out", default=None, help="output PDF path")
    parser.add_argument("--show", action="store_true", help="open an interactive window")
    args = parser.parse_args()
    outfile = args.out
    if outfile is None and not args.show:
        outfile = f"samples/spatial/sort/batcher_routing_n{args.n}_{args.view}.pdf"
    plot(args.n, args.view, outfile, args.show)


if __name__ == "__main__":
    main()
