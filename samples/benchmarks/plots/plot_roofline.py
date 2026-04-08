"""plot_roofline.py — Roofline plot for Cerebras WSE-2 and NVIDIA A100.

QUICK START
-----------
Empty roofline (hardware ceilings only):

    python plot_roofline.py

Add GEMV data (all k values, 750×994 shape):

    python plot_roofline.py --gemv-csv gemv_750_994.csv

Add GEMV data for a single block size k=16:

    python plot_roofline.py --gemv-csv gemv_750_994.csv --k 16

Add 2-D reduce collectives (all k, all methods, both sources):

    python plot_roofline.py --reduce-csv reduce2d_fixed_pxpy_df.csv

Filter reduce to k=2048 elements/PE and only two methods:

    python plot_roofline.py \\
        --reduce-csv reduce2d_fixed_pxpy_df.csv \\
        --reduce-k 2048 \\
        --reduce-methods twophase_reduce_2d tree_reduce_2d

Combine GEMV and reduce with a custom output directory:

    python plot_roofline.py \\
        --gemv-csv gemv_750_994.csv --k 16 \\
        --reduce-csv reduce2d_fixed_pxpy_df.csv --reduce-k 2048 \\
        --output-dir my_plots/

EXPECTED CSV COLUMNS
--------------------
GEMV CSV  (--gemv-csv):
    shape_label, px, py, k, series, source, time_us[, ci_low_us, ci_high_us]
    - series: "WSE-2 GEMV", "WSE-2 GEMV Two-phase", "A100 GEMV CUBLAS"
    - Only rows with shape_label=="750x994" are used.

Reduce CSV  (--reduce-csv):
    x, method, source, time_us[, ci_low_us, ci_high_us]
    - x: message size in bytes (= k_count × 4 for f32)
    - method: chain_reduce_2d | tree_reduce_2d | twophase_reduce_2d
    - source: "This work" (filled marker) | "HPDC24" (hollow marker, dashed)
    - PE grid is assumed to be 512×512 (hardcoded in ReduceMetrics).

EXTENDING TO NEW OPERATIONS
---------------------------
1. Subclass OperationMetrics and implement flops() and bytes_transferred().
2. Define COLOR_MAP, MARKER_MAP, and optionally MARKERFACECOLOR_MAP dicts.
3. Call load_series() with the new metrics instance and add the result to
   data_series before calling plot_roofline().

OUTPUT
------
Saves roofline.png (300 dpi) and roofline.pdf to --output-dir
(default: plots/roofline_plots/ next to this script).
"""
from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "roofline_plots"

# ── Hardware specifications ───────────────────────────────────────────────────

# Cerebras WSE-2
WSE2_PEAK_GFLOPS: float = 1.785e6      # 1.785 PFLOPs
WSE2_MEMORY_BW_GBS: float = 20.0e6    # 20.0 PB/s
WSE2_FABRIC_BW_GBS: float = 3.3e6     # 3.3 PB/s  (off/onramp to fabric)

# NVIDIA A100 (SXM4 80 GB, FP32)
A100_PEAK_GFLOPS: float = 19_500.0    # 19.5 TFLOPs
A100_L1_BW_GBS: float = 2573.8        # GB/s
A100_DRAM_BW_GBS: float = 1224.2      # GB/s


# ── Operation metrics interface ───────────────────────────────────────────────

class OperationMetrics(ABC):
    """Convert a single measurement row into roofline coordinates.

    Subclasses implement `flops` and `bytes_transferred`; everything else
    is derived automatically.
    """

    @abstractmethod
    def flops(self, row: pd.Series) -> float:
        """Total floating-point operations for this measurement."""

    @abstractmethod
    def bytes_transferred(self, row: pd.Series) -> float:
        """Total bytes moved between compute and memory for this measurement."""

    def arithmetic_intensity(self, row: pd.Series) -> float:
        return self.flops(row) / self.bytes_transferred(row)

    def gflops(self, row: pd.Series) -> float:
        return self.flops(row) / (row["time_us"] * 1e-6) / 1e9

    def gflops_ci(self, row: pd.Series) -> tuple[float | None, float | None]:
        """Return (ci_low_gflops, ci_high_gflops).

        CI bounds on time are inverted when converting to GFLOPs:
        faster time (ci_low_us) → higher GFLOPs (ci_high_gflops).
        Returns (None, None) when CI columns are absent or NaN.
        """
        lo_us = row.get("ci_low_us")
        hi_us = row.get("ci_high_us")
        if lo_us is None or hi_us is None:
            return None, None
        if pd.isna(lo_us) or pd.isna(hi_us):
            return None, None
        f = self.flops(row)
        return f / (float(hi_us) * 1e-6) / 1e9, f / (float(lo_us) * 1e-6) / 1e9


class ScaledGEMVMetrics(OperationMetrics):
    """Metrics for the scaled GEMV  y = α·A·x + β·y  (BLAS SGEMV, FP32).

    A is distributed across a px×py PE grid where each PE holds a k×k block,
    so the full matrix dimensions are M = px*k  (rows) and N = py*k  (cols).

    FLOPs  = 2·M·N + 3·M
               ↑ Ax       ↑ scale α·(Ax), scale β·y, add
    Bytes  = (M·N + N + 2·M) · 4
               ↑ A   ↑ x  ↑ read y + write y  (both needed for β·y)
    """

    def _dims(self, row: pd.Series) -> tuple[int, int]:
        m = int(row["px"]) * int(row["k"])
        n = int(row["py"]) * int(row["k"])
        return m, n

    def flops(self, row: pd.Series) -> float:
        m, n = self._dims(row)
        return float(2 * m * n + 3 * m)

    def bytes_transferred(self, row: pd.Series) -> float:
        m, n = self._dims(row)
        return float((m * n + n + 2 * m) * 4)


# ── Plot data types ───────────────────────────────────────────────────────────

@dataclass
class RooflinePoint:
    intensity: float          # FLOP/Byte (x-axis)
    gflops: float             # achieved performance (y-axis)
    ci_low: float | None = None   # lower GFLOPs CI bound
    ci_high: float | None = None  # upper GFLOPs CI bound


@dataclass
class RooflineDataSeries:
    label: str
    points: list[RooflinePoint]   # should be sorted by intensity
    color: str
    marker: str
    linestyle: str = "-"
    fillstyle: str = "full"   # "full" | "none" (hollow) | "left" (half-filled)


# ── Series loader ─────────────────────────────────────────────────────────────

def load_series(
    csv_path: Path,
    metrics: OperationMetrics,
    groupby: list[str],
    color_map: dict[str, str],
    marker: str | dict[str, str] = "o",
    label_fn: Callable[[tuple], str] | None = None,
    filter_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    color_key: str = "primary",
    linestyle_map: dict[str, str] | None = None,
    fillstyle_map: dict[str, str] | None = None,
) -> list[RooflineDataSeries]:
    """Load a CSV and produce one `RooflineDataSeries` per group.

    Visual encoding convention (enforced by the caller via maps):
      - Color   → author / source  (``color_key="secondary"`` for [method, source] groupby)
      - Marker  → problem type     (single string, same for every series in the call)
      - Fillstyle → implementation (``fillstyle_map`` keyed on primary / method)
      - Linestyle → source         (``linestyle_map`` keyed on secondary / source)

    Args:
        csv_path:     Path to the measurements CSV.
        metrics:      ``OperationMetrics`` subclass for FLOPs and bytes.
        groupby:      Column names to group by, e.g. ``["method", "source"]``.
        color_map:    Maps a group-key value to a hex color.
        color_key:    Which key element drives color: ``"primary"`` (first column)
                      or ``"secondary"`` (last column).  Default ``"primary"``.
        marker:       Either a fixed marker string (same for all groups) or a dict
                      keyed on the primary key.
        fillstyle_map: Dict keyed on primary key → matplotlib fillstyle string
                      (``"full"`` / ``"none"`` / ``"left"``).  ``"full"`` if absent.
        linestyle_map: Dict keyed on secondary key → matplotlib linestyle string.
        label_fn:     Function from the group-key tuple to a legend label.
        filter_fn:    Optional pre-filter applied to the full DataFrame.
    """
    df = pd.read_csv(csv_path)
    if filter_fn is not None:
        df = filter_fn(df)

    if label_fn is None:
        label_fn = lambda key: " / ".join(str(v) for v in (key if isinstance(key, tuple) else (key,)))

    result: list[RooflineDataSeries] = []

    for key, group in df.groupby(groupby):
        key_tuple = key if isinstance(key, tuple) else (key,)
        primary   = str(key_tuple[0])
        secondary = str(key_tuple[-1])

        color_lookup = secondary if color_key == "secondary" else primary
        color     = color_map.get(color_lookup, "#333333")
        mkr       = marker if isinstance(marker, str) else marker.get(primary, "o")
        linestyle = (linestyle_map or {}).get(secondary, "-")
        fillstyle = (fillstyle_map or {}).get(primary, "full")
        label     = label_fn(key)

        points = []
        for _, row in group.iterrows():
            intensity = metrics.arithmetic_intensity(row)
            gflops = metrics.gflops(row)
            ci_low, ci_high = metrics.gflops_ci(row)
            points.append(RooflinePoint(intensity, gflops, ci_low, ci_high))

        points.sort(key=lambda p: p.intensity)
        result.append(RooflineDataSeries(
            label=label, points=points,
            color=color, marker=mkr, linestyle=linestyle, fillstyle=fillstyle,
        ))

    return result


class ReduceMetrics(OperationMetrics):
    """Metrics for an element-wise sum reduction across a px×py PE grid.

    Each PE holds k = x/4 f32 elements (x is the message size in bytes from
    the CSV).  Using x directly means HPDC24 rows (which lack k_count/px/py)
    are fully supported.  The PE grid dimensions are supplied at construction
    time (default 512×512).

    FLOPs  = (N − 1) · k    where N = px · py,  k = x / 4
               ↑ one addition per contributing PE, for each of the k elements
    Bytes  = (N · k + k) · 4 = k · (N + 1) · 4
               ↑ read all N input vectors once  +  write result once
    Intensity ≈ 0.25 FLOP/Byte  (independent of k and grid size for large N)
    """

    def __init__(self, px: int = 512, py: int = 512) -> None:
        self.n = px * py

    def _k(self, row: pd.Series) -> int:
        return int(row["x"]) // 4   # x is message size in bytes; k = x / 4 for f32

    def flops(self, row: pd.Series) -> float:
        k = self._k(row)
        return float((self.n - 1) * k)

    def bytes_transferred(self, row: pd.Series) -> float:
        k = self._k(row)
        return float((self.n * k + k) * 4)


# ── Visual encoding ───────────────────────────────────────────────────────────
# Color   → author / source
# Marker  → problem type  (one marker per dataset, same for every series)
# Fillstyle → implementation variant within a problem

# Color per author/source (shared across all datasets).
# Deliberately distinct from the hardware roofline colors
# (WSE-2 = #C84C09 orange, A100 = #005F73 teal).
SOURCE_COLOR: dict[str, str] = {
    "This work": "#2166AC",   # blue  — our WSE-2 results
    "WSE":       "#2166AC",   # same  — GEMV CSV uses "WSE" instead of "This work"
    "HPDC24":    "#762A83",   # purple — HPDC24 baseline
    "A100":      "#1A9641",   # green  — A100 CUBLAS baseline
}

# Marker per problem type (fixed string passed to load_series)
GEMV_MARKER   = "o"
REDUCE_MARKER = "D"

# Fillstyle per GEMV implementation  (keyed on "series" column)
# "full" = filled · "none" = hollow · "left" = half-filled
GEMV_FILLSTYLE_MAP: dict[str, str] = {
    "WSE-2 GEMV":           "full",
    "WSE-2 GEMV Two-phase": "left",
    "A100 GEMV CUBLAS":     "none",
}

# Fillstyle per reduce method  (keyed on "method" column)
REDUCE_FILLSTYLE_MAP: dict[str, str] = {
    "chain_reduce_2d":    "full",
    "tree_reduce_2d":     "none",
    "twophase_reduce_2d": "left",
}

# Human-readable reduce method labels
REDUCE_LABEL_MAP: dict[str, str] = {
    "chain_reduce_2d":    "Chain Reduce 2D",
    "tree_reduce_2d":     "Tree Reduce 2D",
    "twophase_reduce_2d": "Two-phase Reduce 2D",
}

# Linestyle per source (secondary distinguisher for multi-point reduce series)
REDUCE_SOURCE_LINESTYLE: dict[str, str] = {
    "This work": "-",
    "HPDC24":    "--",
}


# ── Hardware roofline spec ────────────────────────────────────────────────────

@dataclass(frozen=True)
class BandwidthRoof:
    label: str
    bandwidth_gbs: float
    label_x: float          # explicit x position for the diagonal label
    linestyle: str = "-"


@dataclass(frozen=True)
class HardwareSpec:
    name: str
    peak_gflops: float
    peak_label: str
    roofs: tuple[BandwidthRoof, ...]
    color: str
    name_xy: tuple[float, float]


HARDWARE: tuple[HardwareSpec, ...] = (
    HardwareSpec(
        name="Cerebras WSE-2",
        peak_gflops=WSE2_PEAK_GFLOPS,
        peak_label="1.785 PFLOPs",
        roofs=(
            BandwidthRoof("Memory\n20.0 PB/s", WSE2_MEMORY_BW_GBS, label_x=0.025, linestyle="--"),
            BandwidthRoof("Off/onramp to fabric\n3.3 PB/s", WSE2_FABRIC_BW_GBS, label_x=0.055),
        ),
        color="#C84C09",
        name_xy=(20.0, WSE2_PEAK_GFLOPS * 0.22),
    ),
    HardwareSpec(
        name="NVIDIA A100",
        peak_gflops=A100_PEAK_GFLOPS,
        peak_label="19.5 TFLOPs",
        roofs=(
            BandwidthRoof("L1 - 2573.8 GB/s", A100_L1_BW_GBS, label_x=0.5, linestyle="--"),
            BandwidthRoof("DRAM - 1224.2 GB/s", A100_DRAM_BW_GBS, label_x=2.0),
        ),
        color="#005F73",
        name_xy=(20.0, A100_PEAK_GFLOPS * 0.06),
    ),
)

X_MIN, X_MAX = 1e-2, 1e2
Y_MIN, Y_MAX = 10.0, 1e7
FIG_W, FIG_H = 5.5, 9.0
X_DECADES = np.log10(X_MAX) - np.log10(X_MIN)   # 4
Y_DECADES = np.log10(Y_MAX) - np.log10(Y_MIN)   # 6


# ── Plot helpers ──────────────────────────────────────────────────────────────

def roofline(x: np.ndarray, peak: float, bandwidth: float) -> np.ndarray:
    return np.minimum(peak, bandwidth * x)


def diagonal_rotation(ax: plt.Axes) -> float:
    """Estimate text rotation (degrees) matching a slope-1 line in log-log space."""
    fig = ax.get_figure()
    renderer = fig.canvas.get_renderer()
    bbox = ax.get_window_extent(renderer=renderer)
    px_per_x_dec = bbox.width / X_DECADES
    px_per_y_dec = bbox.height / Y_DECADES
    return float(np.degrees(np.arctan(px_per_y_dec / px_per_x_dec)))


def label_xy(roof: BandwidthRoof) -> tuple[float, float]:
    x = roof.label_x
    return x, roof.bandwidth_gbs * x


# ── Main plot function ────────────────────────────────────────────────────────

def plot_roofline(
    output_dir: Path,
    data_series: list[RooflineDataSeries] = (),
) -> None:
    sns.set_style("whitegrid")

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    x = np.logspace(np.log10(X_MIN), np.log10(X_MAX), 2000)

    for hw in HARDWARE:
        for roof in hw.roofs:
            ax.plot(
                x,
                roofline(x, hw.peak_gflops, roof.bandwidth_gbs),
                color=hw.color,
                linewidth=2.0,
                linestyle=roof.linestyle,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xlabel("FLOP/Byte", fontsize=12, fontweight="bold")
    ax.set_ylabel("GFLOPs", fontsize=12, fontweight="bold")
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.1)

    plt.tight_layout()

    rot = diagonal_rotation(ax)

    for hw in HARDWARE:
        for roof in hw.roofs:
            lx, ly = label_xy(roof)
            ax.text(
                lx, ly,
                roof.label,
                fontsize=8.5,
                rotation=rot,
                rotation_mode="anchor",
                va="bottom",
                ha="left",
                color=hw.color,
            )

        ridge_min = min(hw.peak_gflops / r.bandwidth_gbs for r in hw.roofs)
        peak_label_x = max(X_MIN * 2.0, ridge_min * 1.25)
        ax.text(
            peak_label_x, hw.peak_gflops * 1.1,
            hw.peak_label,
            fontsize=9,
            va="bottom",
            ha="left",
            color="black",
        )

        ax.text(
            *hw.name_xy,
            hw.name,
            fontsize=11,
            style="italic",
            color="gray",
            alpha=0.65,
            ha="right",
            va="center",
        )

    # ── Data series ───────────────────────────────────────────────────────────
    for series in data_series:
        xs = [p.intensity for p in series.points]
        ys = [p.gflops for p in series.points]

        ax.plot(
            xs, ys,
            color=series.color,
            marker=series.marker,
            linestyle=series.linestyle,
            fillstyle=series.fillstyle,
            linewidth=2.4,
            markersize=7,
            label=series.label,
        )

        ci_lows  = [p.ci_low  for p in series.points]
        ci_highs = [p.ci_high for p in series.points]
        if all(v is not None for v in ci_lows + ci_highs):
            ax.fill_between(
                xs,
                ci_lows,
                ci_highs,
                color=series.color,
                alpha=0.18,
                linewidth=0,
            )

    if data_series:
        ax.legend(fontsize=9, loc="best")

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "roofline.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "roofline.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {output_dir}/roofline.{{png,pdf}}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate roofline plot for WSE-2 and A100. See module docstring for full usage.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Output directory for plots",
    )
    parser.add_argument(
        "--gemv-csv", type=Path, default=None,
        help="CSV file with scaled-GEMV measurements (px, py, k, series, time_us, ...)",
    )
    parser.add_argument(
        "--k", type=int, default=None,
        help="If set, restrict GEMV data to rows with this block size k",
    )
    parser.add_argument(
        "--reduce-csv", type=Path, default=None,
        help="CSV file with reduce measurements (x, method, source, time_us, ...)",
    )
    parser.add_argument(
        "--reduce-k", type=int, default=None,
        help="If set, restrict reduce data to rows with this number of f32 elements per PE (x = k*4 bytes)",
    )
    parser.add_argument(
        "--reduce-methods", nargs="*", default=None,
        metavar="METHOD",
        help="If set, show only these reduce methods "
             "(e.g. chain_reduce_2d tree_reduce_2d twophase_reduce_2d). "
             "Shows all methods when omitted.",
    )
    args = parser.parse_args()

    data_series: list[RooflineDataSeries] = []

    if args.gemv_csv is not None:
        def gemv_filter(df: pd.DataFrame) -> pd.DataFrame:
            df = df[df["shape_label"] == "750x994"]
            if args.k is not None:
                df = df[df["k"] == args.k]
            return df

        gemv_series = load_series(
            csv_path=args.gemv_csv,
            metrics=ScaledGEMVMetrics(),
            groupby=["series", "source"],
            color_map=SOURCE_COLOR,
            color_key="secondary",
            marker=GEMV_MARKER,
            fillstyle_map=GEMV_FILLSTYLE_MAP,
            label_fn=lambda key: key[0],
            filter_fn=gemv_filter,
        )
        k_desc = f", k={args.k}" if args.k is not None else ""
        print(f"Loaded {len(gemv_series)} GEMV series from {args.gemv_csv}{k_desc}")
        data_series += gemv_series

    if args.reduce_csv is not None:
        def reduce_filter(df: pd.DataFrame) -> pd.DataFrame:
            if args.reduce_k is not None:
                x_value = args.reduce_k * 4   # x is message size in bytes; k*4 for f32
                df = df[df["x"] == x_value]
            if args.reduce_methods is not None:
                df = df[df["method"].isin(args.reduce_methods)]
            return df

        reduce_series = load_series(
            csv_path=args.reduce_csv,
            metrics=ReduceMetrics(px=512, py=512),
            groupby=["method", "source"],
            color_map=SOURCE_COLOR,
            color_key="secondary",
            marker=REDUCE_MARKER,
            fillstyle_map=REDUCE_FILLSTYLE_MAP,
            linestyle_map=REDUCE_SOURCE_LINESTYLE,
            label_fn=lambda key: f"{REDUCE_LABEL_MAP.get(key[0], key[0])} ({key[1]})",
            filter_fn=reduce_filter,
        )
        k_desc = f", k={args.reduce_k}" if args.reduce_k is not None else ""
        m_desc = f", methods={args.reduce_methods}" if args.reduce_methods else ""
        print(f"Loaded {len(reduce_series)} reduce series from {args.reduce_csv}{k_desc}{m_desc}")
        data_series += reduce_series

    plot_roofline(args.output_dir, data_series)


if __name__ == "__main__":
    main()
