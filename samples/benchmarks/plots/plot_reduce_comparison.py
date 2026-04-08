"""Plot the runtime ratio (This work / HPDC24) for all 2D-reduce methods.

Only the 512×512 PE-grid results are included.
HPDC24 rows have no explicit px/py; they are treated as 512×512 throughout.

Usage
-----
    python plot_reduce_comparison.py                              # default CSV path
    python plot_reduce_comparison.py --csv reduce2d_fixed_pxpy_df.csv
    python plot_reduce_comparison.py --output-dir my_plots
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_CSV = Path(__file__).parent / "reduce2d_fixed_pxpy_df.csv"
DEFAULT_OUT = Path(__file__).parent / "roofline_plots"

METHOD_LABELS: dict[str, str] = {
    "chain_reduce_2d":    "Chain",
    "tree_reduce_2d":     "Tree",
    "twophase_reduce_2d": "Two-phase",
}

# Colours chosen to match the roofline palette (blue / green / orange).
METHOD_COLORS: dict[str, str] = {
    "chain_reduce_2d":    "#2166AC",
    "tree_reduce_2d":     "#4DAC26",
    "twophase_reduce_2d": "#E08214",
}

METHOD_MARKERS: dict[str, str] = {
    "chain_reduce_2d":    "o",
    "tree_reduce_2d":     "s",
    "twophase_reduce_2d": "^",
}


# ── Analysis ──────────────────────────────────────────────────────────────────

def compute_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with columns [method, x, ratio, ci_low, ci_high].

    *ratio* = time_this_work / time_hpdc24.
    CI bounds use the "This work" CI divided by the HPDC24 point estimate.
    Only x-values present in *both* sources are kept (inner join).
    Only 512×512 "This work" rows are used; HPDC24 rows have no px/py.
    """
    # Filter "This work" to 512×512 only.
    this_work = df[
        (df["source"] == "This work") &
        (df["px"] == 512) &
        (df["py"] == 512)
    ].copy()

    hpdc = df[df["source"] == "HPDC24"].copy()

    rows = []
    for method in df["method"].unique():
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
                "method":   method,
                "x":        x,
                "ratio":    ratio,
                "ci_low":   ci_low,
                "ci_high":  ci_high,
            })

    return pd.DataFrame(rows)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot(ratio_df: pd.DataFrame, output_dir: Path,
         hmeans: dict[str, float] | None = None) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 3.8))

    # Work in log₂ space on a plain linear axis to avoid matplotlib's
    # log-base-2 layout bug (produces ~377-inch-wide images).
    all_x = sorted(ratio_df["x"].unique())
    log2_x = {v: math.log2(v) for v in all_x}

    for method, label in METHOD_LABELS.items():
        sub = ratio_df[ratio_df["method"] == method].sort_values("x")
        if sub.empty:
            continue

        color  = METHOD_COLORS[method]
        marker = METHOD_MARKERS[method]
        xs = [log2_x[v] for v in sub["x"]]

        ax.plot(xs, sub["ratio"],
                label=label, color=color, marker=marker,
                linewidth=1.4, markersize=5)

        ci = sub.dropna(subset=["ci_low", "ci_high"])
        if not ci.empty:
            ci_xs = [log2_x[v] for v in ci["x"]]
            ax.fill_between(ci_xs, ci["ci_low"], ci["ci_high"],
                            color=color, alpha=0.18, linewidth=0)

    # Harmonic-mean lines per method and overall.
    if hmeans:
        x_right = max(log2_x.values())
        for method, hm in hmeans.items():
            if method == "__overall__":
                color, label_text = "black", f"Overall H-mean: {hm:.3f}"
                lw, ls, alpha = 1.2, "-.", 0.9
            else:
                color = METHOD_COLORS.get(method, "grey")
                label_text = f"{METHOD_LABELS.get(method, method)} H-mean: {hm:.3f}"
                lw, ls, alpha = 0.9, ":", 0.75
            ax.axhline(hm, color=color, linewidth=lw, linestyle=ls,
                       alpha=alpha, zorder=1)
            ax.text(x_right, hm, f"  {label_text}",
                    va="bottom", ha="right", fontsize=7, color=color)

    # Reference line at ratio = 1 (parity).
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", zorder=0)
    ax.text(ax.get_xlim()[0], 1.0, "  parity",
            va="bottom", ha="left", fontsize=8, color="black")

    tick_positions = [log2_x[v] for v in all_x]
    tick_labels    = [f"{int(v):,}" for v in all_x]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)

    ax.set_xlabel("Elements per PE  (x)", fontsize=11)
    ax.set_ylabel("Runtime ratio  (This work / HPDC24)", fontsize=11)
    ax.set_title("2D Reduce: This work vs HPDC24  (512×512 grid)", fontsize=11)

    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    ax.grid(axis="y", which="major", linewidth=0.5, alpha=0.4)
    ax.grid(axis="y", which="minor", linewidth=0.3, alpha=0.25)

    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = output_dir / f"reduce_comparison.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved → {out}")
    plt.close(fig)


# ── Summary statistics ────────────────────────────────────────────────────────

def harmonic_mean(values: "pd.Series") -> float:
    return len(values) / (1.0 / values).sum()


def print_harmonic_means(ratio_df: pd.DataFrame) -> dict[str, float]:
    """Print and return the harmonic mean of ratio for each method."""
    print("\nHarmonic mean of runtime ratio (This work / HPDC24):")
    print(f"  {'Method':<25}  {'H-mean':>8}  {'n':>4}")
    print("  " + "-" * 42)
    hmeans: dict[str, float] = {}
    for method, label in METHOD_LABELS.items():
        sub = ratio_df[ratio_df["method"] == method]["ratio"].dropna()
        if sub.empty:
            continue
        hm = harmonic_mean(sub)
        hmeans[method] = hm
        print(f"  {label:<25}  {hm:>8.4f}  {len(sub):>4}")

    all_ratios = ratio_df["ratio"].dropna()
    overall_hm = harmonic_mean(all_ratios)
    print("  " + "-" * 42)
    print(f"  {'Overall':<25}  {overall_hm:>8.4f}  {len(all_ratios):>4}")
    print()
    hmeans["__overall__"] = overall_hm
    return hmeans


# ── CSV export ────────────────────────────────────────────────────────────────

def export_csv(ratio_df: pd.DataFrame, output_dir: Path) -> None:
    out = output_dir / "reduce_comparison_data.csv"
    ratio_df.to_csv(out, index=False)
    print(f"Saved data → {out}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--csv", type=Path, default=DEFAULT_CSV, metavar="PATH",
        help=f"Input CSV (default: {DEFAULT_CSV.name})",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUT, metavar="DIR",
        help=f"Directory for output files (default: {DEFAULT_OUT})",
    )
    args = parser.parse_args()

    if not args.csv.exists():
        parser.error(f"CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)
    ratio_df = compute_ratio(df)

    if ratio_df.empty:
        raise RuntimeError("No paired (This work, HPDC24) rows found for any method.")

    export_csv(ratio_df, args.output_dir)
    hmeans = print_harmonic_means(ratio_df)
    plot(ratio_df, args.output_dir, hmeans=hmeans)


if __name__ == "__main__":
    main()
