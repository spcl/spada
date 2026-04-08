"""Compare "This work" vs HPDC24 runtime ratios for any collective CSV.

Works for reduce, broadcast, or any CSV that shares the schema:
  family, sweep, method, source, x, time_us, ci_low_us, ci_high_us, px, py, k_count, label

The script:
  1. Filters "This work" rows to an optional (px, py) grid size.
  2. Inner-joins on (method, x) with HPDC24 rows to compute
         ratio = time_this_work / time_hpdc24.
  3. Prints per-method and overall harmonic means.
  4. Plots ratio vs x (log₂ scale) with per-method harmonic-mean lines.
  5. Exports the ratio table and the plot to --output-dir.

Usage examples
--------------
    # 2D reduce, 512×512 grid
    python plot_collective_comparison.py \\
        --csv reduce2d_fixed_pxpy_df.csv --px 512 --py 512 \\
        --title "2D Reduce: This work vs HPDC24 (512×512 grid)" \\
        --output-prefix reduce_comparison

    # Broadcast, 512×1 grid
    python plot_collective_comparison.py \\
        --csv broadcast_fixed_pxpy_df.csv --px 512 --py 1 \\
        --title "Broadcast: This work vs HPDC24 (512×1 grid)" \\
        --output-prefix broadcast_comparison
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_OUT = Path(__file__).parent / "roofline_plots"

# Colour / marker cycles used when methods are auto-discovered.
_COLOR_CYCLE = [
    "#2166AC", "#4DAC26", "#E08214",
    "#762A83", "#D6604D", "#1A9850",
    "#F4A582", "#74ADD1",
]
_MARKER_CYCLE = ["o", "s", "^", "D", "v", "P", "X", "*"]


# ── Analysis ──────────────────────────────────────────────────────────────────

def compute_ratio(
    df: pd.DataFrame,
    px_filter: int | None,
    py_filter: int | None,
) -> tuple[pd.DataFrame, dict[str, str], dict[str, str], dict[str, str]]:
    """Return (ratio_df, method_labels, method_colors, method_markers).

    *ratio* = time_this_work / time_hpdc24.
    CI bounds use the "This work" CI divided by the HPDC24 point estimate.
    Only x-values present in *both* sources are kept (inner join).
    """
    this_work = df[df["source"] == "This work"].copy()
    if px_filter is not None:
        this_work = this_work[this_work["px"] == px_filter]
    if py_filter is not None:
        this_work = this_work[this_work["py"] == py_filter]

    hpdc = df[df["source"] == "HPDC24"].copy()

    # Discover methods in a stable order.
    all_methods = list(dict.fromkeys(
        list(this_work["method"].unique()) + list(hpdc["method"].unique())
    ))

    method_labels:  dict[str, str] = {}
    method_colors:  dict[str, str] = {}
    method_markers: dict[str, str] = {}
    for i, m in enumerate(all_methods):
        method_labels[m]  = m.replace("_", " ").title()
        method_colors[m]  = _COLOR_CYCLE[i % len(_COLOR_CYCLE)]
        method_markers[m] = _MARKER_CYCLE[i % len(_MARKER_CYCLE)]

    rows = []
    for method in all_methods:
        tw = this_work[this_work["method"] == method].set_index("x")
        hp = hpdc[hpdc["method"] == method].set_index("x")

        common_x = tw.index.intersection(hp.index)
        for x in sorted(common_x):
            t_tw = tw.loc[x, "time_us"]
            t_hp = hp.loc[x, "time_us"]
            ratio = t_tw / t_hp

            ci_low_us  = tw.loc[x, "ci_low_us"]
            ci_high_us = tw.loc[x, "ci_high_us"]
            ci_low  = ci_low_us  / t_hp if pd.notna(ci_low_us)  else None
            ci_high = ci_high_us / t_hp if pd.notna(ci_high_us) else None

            rows.append({
                "method":  method,
                "x":       x,
                "ratio":   ratio,
                "ci_low":  ci_low,
                "ci_high": ci_high,
            })

    return pd.DataFrame(rows), method_labels, method_colors, method_markers


# ── Summary statistics ────────────────────────────────────────────────────────

def harmonic_mean(values: pd.Series) -> float:
    return len(values) / (1.0 / values).sum()


def print_harmonic_means(
    ratio_df: pd.DataFrame,
    method_labels: dict[str, str],
) -> dict[str, float]:
    """Print and return the harmonic mean of ratio for each method + overall."""
    print("\nHarmonic mean of runtime ratio (This work / HPDC24):")
    print(f"  {'Method':<30}  {'H-mean':>8}  {'n':>4}")
    print("  " + "-" * 47)
    hmeans: dict[str, float] = {}
    for method, label in method_labels.items():
        sub = ratio_df[ratio_df["method"] == method]["ratio"].dropna()
        if sub.empty:
            continue
        hm = harmonic_mean(sub)
        hmeans[method] = hm
        print(f"  {label:<30}  {hm:>8.4f}  {len(sub):>4}")

    all_ratios = ratio_df["ratio"].dropna()
    overall_hm = harmonic_mean(all_ratios)
    print("  " + "-" * 47)
    print(f"  {'Overall':<30}  {overall_hm:>8.4f}  {len(all_ratios):>4}")
    print()
    hmeans["__overall__"] = overall_hm
    return hmeans


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot(
    ratio_df: pd.DataFrame,
    method_labels:  dict[str, str],
    method_colors:  dict[str, str],
    method_markers: dict[str, str],
    hmeans: dict[str, float],
    title: str,
    output_dir: Path,
    output_prefix: str,
) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 3.8))

    # Work in log₂ space on a plain linear axis to avoid matplotlib's
    # log-base-2 layout bug (produces enormous pixel widths).
    all_x = sorted(ratio_df["x"].unique())
    log2_x = {v: math.log2(v) for v in all_x}

    for method, label in method_labels.items():
        sub = ratio_df[ratio_df["method"] == method].sort_values("x")
        if sub.empty:
            continue

        color  = method_colors[method]
        marker = method_markers[method]
        xs = [log2_x[v] for v in sub["x"]]

        ax.plot(xs, sub["ratio"],
                label=label, color=color, marker=marker,
                linewidth=1.4, markersize=5)

        ci = sub.dropna(subset=["ci_low", "ci_high"])
        if not ci.empty:
            ci_xs = [log2_x[v] for v in ci["x"]]
            ax.fill_between(ci_xs, ci["ci_low"], ci["ci_high"],
                            color=color, alpha=0.18, linewidth=0)

    # Harmonic-mean lines.
    x_right = max(log2_x.values())
    for method, hm in hmeans.items():
        if method == "__overall__":
            color, label_text = "black", f"Overall H-mean: {hm:.3f}"
            lw, ls, alpha = 1.2, "-.", 0.9
        else:
            color = method_colors.get(method, "grey")
            label_text = f"{method_labels.get(method, method)} H-mean: {hm:.3f}"
            lw, ls, alpha = 0.9, ":", 0.75
        ax.axhline(hm, color=color, linewidth=lw, linestyle=ls, alpha=alpha, zorder=1)
        ax.text(x_right, hm, f"  {label_text}",
                va="bottom", ha="right", fontsize=7, color=color)

    # Parity line.
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", zorder=0)
    ax.text(ax.get_xlim()[0], 1.0, "  parity",
            va="bottom", ha="left", fontsize=8, color="black")

    tick_positions = [log2_x[v] for v in all_x]
    tick_labels    = [f"{int(v):,}" for v in all_x]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)

    ax.set_xlabel("Elements per PE  (x)", fontsize=11)
    ax.set_ylabel("Runtime ratio  (This work / HPDC24)", fontsize=11)
    ax.set_title(title, fontsize=11)

    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    ax.grid(axis="y", which="major", linewidth=0.5, alpha=0.4)
    ax.grid(axis="y", which="minor", linewidth=0.3, alpha=0.25)

    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = output_dir / f"{output_prefix}.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved → {out}")
    plt.close(fig)


# ── CSV export ────────────────────────────────────────────────────────────────

def export_csv(ratio_df: pd.DataFrame, output_dir: Path, output_prefix: str) -> None:
    out = output_dir / f"{output_prefix}_data.csv"
    ratio_df.to_csv(out, index=False)
    print(f"Saved data → {out}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--csv", type=Path, required=True, metavar="PATH",
        help="Input CSV (e.g. reduce2d_fixed_pxpy_df.csv).",
    )
    parser.add_argument(
        "--px", type=int, default=None, metavar="INT",
        help="Filter 'This work' rows to this px value (e.g. 512).",
    )
    parser.add_argument(
        "--py", type=int, default=None, metavar="INT",
        help="Filter 'This work' rows to this py value (e.g. 512 or 1).",
    )
    parser.add_argument(
        "--title", type=str, default=None, metavar="STR",
        help="Plot title (auto-derived from family + grid size when omitted).",
    )
    parser.add_argument(
        "--output-prefix", type=str, default=None, metavar="STR",
        help="Stem for output files (auto-derived from family when omitted).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUT, metavar="DIR",
        help=f"Directory for output files (default: {DEFAULT_OUT}).",
    )
    args = parser.parse_args()

    if not args.csv.exists():
        parser.error(f"CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)

    # Auto-derive title / prefix from CSV metadata.
    family = df["family"].iloc[0] if "family" in df.columns else args.csv.stem
    grid_str = ""
    if args.px is not None and args.py is not None:
        grid_str = f" ({args.px}×{args.py} grid)"
    elif args.px is not None:
        grid_str = f" (px={args.px})"
    elif args.py is not None:
        grid_str = f" (py={args.py})"

    title = args.title or f"{family.replace('_', ' ').title()}: This work vs HPDC24{grid_str}"
    output_prefix = args.output_prefix or f"{family}_comparison"

    ratio_df, method_labels, method_colors, method_markers = compute_ratio(
        df, args.px, args.py,
    )

    if ratio_df.empty:
        raise RuntimeError("No paired (This work, HPDC24) rows found.")

    export_csv(ratio_df, args.output_dir, output_prefix)
    hmeans = print_harmonic_means(ratio_df, method_labels)
    plot(ratio_df, method_labels, method_colors, method_markers,
         hmeans, title, args.output_dir, output_prefix)


if __name__ == "__main__":
    main()
