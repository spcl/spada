"""plot_roofline.py — Roofline plot for Cerebras WSE-2 and NVIDIA A100.

Three problem types are supported: scaled GEMV, 2-D reduce collectives,
and stencils.  All three can be combined in a single plot.  Run from the
samples/benchmarks/plots/ directory so relative CSV paths resolve correctly.

QUICK START
-----------
Empty roofline (hardware ceilings only):

    python plot_roofline.py

All three problem types combined (recommended starting point):

    python plot_roofline.py \\
        --gemv-csv gemv_750_994.csv --k 32 \\
        --reduce-csv reduce2d_fixed_pxpy_df.csv --reduce-k 4096 \\
            --reduce-methods chain_reduce_2d \\
        --stencil-flops-csv stencil_flops.csv \\
        --stencil-scaling-csv stencil_scaling_vertical.csv \\
        --stencil-k 320

GEMV OPTIONS
------------
Add GEMV data (all k values, 750×994 PE grid):

    python plot_roofline.py --gemv-csv gemv_750_994.csv

Restrict to a single block size k=32:

    python plot_roofline.py --gemv-csv gemv_750_994.csv --k 32

REDUCE OPTIONS
--------------
Add 2-D reduce collectives (all k, all methods, both sources):

    python plot_roofline.py --reduce-csv reduce2d_fixed_pxpy_df.csv

Filter to k=4096 elements/PE and a single method:

    python plot_roofline.py \\
        --reduce-csv reduce2d_fixed_pxpy_df.csv \\
        --reduce-k 4096 \\
        --reduce-methods chain_reduce_2d

STENCIL OPTIONS
---------------
Add stencil benchmarks (all programs, all z-depths):

    python plot_roofline.py \\
        --stencil-flops-csv stencil_flops.csv \\
        --stencil-scaling-csv stencil_scaling_vertical.csv

Restrict to z-depth k=320 and specific programs:

    python plot_roofline.py \\
        --stencil-flops-csv stencil_flops.csv \\
        --stencil-scaling-csv stencil_scaling_vertical.csv \\
        --stencil-k 320 \\
        --stencil-programs "2D Laplacian" "Vertical Stencil"

The stencil flops CSV is generated with:

    python -m spatialstencil.cli.count_flop samples/benchmarks/

GENERAL OPTIONS
---------------
--output-dir PATH   Where to write roofline.png and roofline.pdf
                    (default: roofline_plots/ next to this script)

EXPECTED CSV COLUMNS
--------------------
GEMV CSV  (--gemv-csv):
    shape_label, px, py, k, series, source, time_us[, ci_low_us, ci_high_us]
    - series: "WSE-2 GEMV" | "WSE-2 GEMV Two-phase" | "A100 GEMV CUBLAS"
    - Only rows with shape_label == "750x994" are used.
    - Produced by: plot_gemv_sweep.py (saved as <shape>_<px>_<py>.csv)

Reduce CSV  (--reduce-csv):
    x, method, source, time_us[, ci_low_us, ci_high_us]
    - x: message size in bytes (= k_elements × 4 for f32)
    - method: chain_reduce_2d | tree_reduce_2d | twophase_reduce_2d
    - source: "This work" (filled, solid) | "HPDC24" (hollow, dashed)
    - PE grid assumed 512×512.

Stencil flops CSV  (--stencil-flops-csv):
    Program, Flop, Loads, Stores, Bytes, ArithmeticIntensity
    - Produced by: python -m spatialstencil.cli.count_flop <dir>
    - Program names follow the pattern {name}_{x}_{y}_{z}
    - Supported kernels: laplacian, vertical_advection, uvbke

Stencil scaling CSV  (--stencil-scaling-csv):
    Program, time_us, time_ci_low, time_ci_hi, k, flops, ...
    - Program display names: "2D Laplacian" | "Vertical Stencil" | "UVBKE"
    - k: z-dimension (vertical depth); one point per unique k per program

VISUAL ENCODING
---------------
    Color    → author / source (blue = This work / WSE, green = A100, purple = HPDC24)
    Marker   → problem type   (circle = GEMV, diamond = reduce, triangle = stencil)
    Fillstyle → variant       (full = primary, hollow = baseline, half = two-phase)
    Linestyle → source        (solid = This work, dashed = HPDC24 / external)

EXTENDING TO NEW OPERATIONS
---------------------------
1. Subclass OperationMetrics and implement flops() and bytes_transferred().
2. Define COLOR_MAP and optionally MARKER_MAP / FILLSTYLE_MAP dicts.
3. Call load_series() with the new metrics instance and append the result
   to data_series before calling plot_roofline().

OUTPUT
------
Saves roofline.png (300 dpi) and roofline.pdf to --output-dir
(default: roofline_plots/ next to this script).
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
    color_index: int | None = None,
    color_fn: Callable[[tuple], str] | None = None,
    linestyle_map: dict[str, str] | None = None,
    linestyle_index: int | None = None,
    fillstyle_map: dict[str, str] | None = None,
    fillstyle_index: int | None = None,
    marker_map: dict[str, str] | None = None,
    marker_index: int | None = None,
    dropna: bool = True,
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

    for key, group in df.groupby(groupby, dropna=dropna):
        key_tuple = key if isinstance(key, tuple) else (key,)
        primary   = str(key_tuple[0])
        secondary = str(key_tuple[-1])

        if color_fn is not None:
            color = color_fn(key_tuple)
        elif color_index is not None:
            color = color_map.get(str(key_tuple[color_index]), "#333333")
        else:
            color_lookup = secondary if color_key == "secondary" else primary
            color = color_map.get(color_lookup, "#333333")

        # Marker: check override map first (e.g. HPDC24 → "x"), then fall back
        mkr_override = (marker_map or {}).get(
            str(key_tuple[marker_index]) if marker_index is not None else "", None
        )
        mkr = mkr_override if mkr_override else (
            marker if isinstance(marker, str) else marker.get(primary, "o")
        )

        ls_lookup = str(key_tuple[linestyle_index]) if linestyle_index is not None else secondary
        linestyle = (linestyle_map or {}).get(ls_lookup, "-")
        fs_lookup = str(key_tuple[fillstyle_index]) if fillstyle_index is not None else primary
        fillstyle = (fillstyle_map or {}).get(fs_lookup, "full")
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
    Bytes  = (N · k + N · k) · 4 = k · N · 8
               ↑ read all N input vectors once + write all input vectors once
    Intensity ≈ 0.125 FLOP/Byte  (independent of k and grid size for large N)
    """

    def __init__(self, default_px: int = 512, default_py: int = 512) -> None:
        self.default_n = default_px * default_py

    def _n(self, row: pd.Series) -> int:
        """PE count: prefer per-row px/py when available (e.g. 750×994), else fall back to default."""
        if "px" in row.index and pd.notna(row["px"]) and "py" in row.index and pd.notna(row["py"]):
            return int(row["px"]) * int(row["py"])
        return self.default_n

    def _k(self, row: pd.Series) -> int:
        return int(row["x"]) // 4   # x is message size in bytes; k = x / 4 for f32

    def flops(self, row: pd.Series) -> float:
        n = self._n(row)
        k = self._k(row)
        return float((n - 1) * k)

    def bytes_transferred(self, row: pd.Series) -> float:
        n = self._n(row)
        k = self._k(row)
        return float((n * k + k) * 4) * 2


# ── Visual encoding ───────────────────────────────────────────────────────────
# Fill      → source/author   (SpaDA = full, A100 = hollow, HPDC24 = "x" marker)
# Color     → problem group   (GEMV: one shared color; Reduce: by grid size;
#                              Stencil: by program type)
# Marker    → problem type    (square=GEMV, diamond=reduce, triangle=stencil)
#             HPDC24 always overrides to "x" regardless of problem type.

# Fillstyle keyed on source column.
# "full" = filled, "none" = hollow; HPDC24 is overridden to "x" marker anyway.
SOURCE_FILLSTYLE: dict[str, str] = {
    "This work": "full",
    "WSE":       "full",   # GEMV CSV uses "WSE" instead of "This work"
    "A100":      "none",
    "HPDC24":    "full",   # shape is overridden to "x" via SOURCE_MARKER_OVERRIDE
}

# Marker override keyed on source: HPDC24 always renders as "x".
SOURCE_MARKER_OVERRIDE: dict[str, str] = {
    "HPDC24": "x",
}

# Marker per problem type (fixed string passed to load_series)
GEMV_MARKER   = "s"
REDUCE_MARKER = "D"

# Single color for all GEMV series (fill distinguishes SpaDA from A100)
GEMV_COLOR = "#762A83"   # purple

# Color per PE-grid size for reduce series (shared across sources/methods)
REDUCE_GRID_COLOR: dict[str, str] = {
    "512×512":  "#2166AC",   # blue
    "750×994":  "#D6604D",   # red-orange
}

# Human-readable reduce method labels
REDUCE_LABEL_MAP: dict[str, str] = {
    "chain_reduce_2d":    "SpaDA Chain Reduce",
    "tree_reduce_2d":     "SpaDA Tree Reduce",
    "twophase_reduce_2d": "SpaDA Two-phase Reduce",
}

# Linestyle per source (secondary distinguisher for multi-point reduce series)
REDUCE_SOURCE_LINESTYLE: dict[str, str] = {
    "This work": "-",
    "HPDC24":    "--",
}

# ── Stencil visual encoding ───────────────────────────────────────────────────

STENCIL_MARKER = "^"

# Color per stencil program (triangle marker, one series per program)
STENCIL_COLOR_MAP: dict[str, str] = {
    "2D Laplacian":    "#2166AC",   # blue  — consistent with WSE "This work"
    "Vertical Stencil": "#E08214",  # amber
    "UVBKE":            "#4DAC26",  # green
}

# Map from display name (scaling CSV) to program key prefix (flops CSV)
STENCIL_NAME_MAP: dict[str, str] = {
    "2D Laplacian":    "laplacian",
    "Vertical Stencil": "pure_vertical",
    "UVBKE":           "uvbke",
}


def load_stencil_series(
    flops_csv: Path,
    scaling_csv: Path,
    stencil_k: int | None = None,
    stencil_programs: list[str] | None = None,
) -> list[RooflineDataSeries]:
    """Load stencil benchmark series for the roofline plot.

    Arithmetic intensity is taken from the flops CSV (constant per stencil
    pattern, independent of domain size).  Performance (GFLOPs/s) is derived
    from the median runtime in the scaling CSV.  Each (program, k) pair
    becomes one RooflinePoint; all k-points for the same program are grouped
    into one RooflineDataSeries.

    Args:
        flops_csv:        Output of ``python -m spatialstencil.cli.count_flop``.
                          Must have columns: Program, Flop, Bytes, ArithmeticIntensity.
        scaling_csv:      Runtime measurements CSV.
                          Must have columns: Program, time_us, time_ci_low,
                          time_ci_hi, k, flops.
        stencil_k:        If given, include only rows with this z-depth (k).
        stencil_programs: If given, include only these program display names.
    """
    flops_df = pd.read_csv(flops_csv)
    scaling_df = pd.read_csv(scaling_csv)

    # Build AI lookup: strip last three "_x_y_z" parts to recover the base key
    ai_lookup: dict[str, float] = {}
    for _, row in flops_df.iterrows():
        prog = str(row["Program"])
        parts = prog.split("_")
        key = "_".join(parts[:-3]) if len(parts) > 3 else prog
        ai_lookup[key] = float(row["ArithmeticIntensity"])

    if stencil_programs is not None:
        scaling_df = scaling_df[scaling_df["Program"].isin(stencil_programs)]
    if stencil_k is not None:
        scaling_df = scaling_df[scaling_df["k"] == stencil_k]

    result: list[RooflineDataSeries] = []

    for prog_name, prog_group in scaling_df.groupby("Program"):
        prog_str = str(prog_name)
        key = STENCIL_NAME_MAP.get(prog_str, prog_str.lower().replace(" ", "_"))
        if key not in ai_lookup:
            raise KeyError(
                f"No flop data found for stencil '{prog_str}' (looked up key '{key}') in {flops_csv}. "
                f"Available keys: {sorted(ai_lookup)}. "
                f"Re-run 'python -m spatialstencil.cli.count_flop' and update --stencil-flops-csv, "
                f"or add '{prog_str}' to STENCIL_NAME_MAP."
            )

        intensity = ai_lookup[key]
        color = STENCIL_COLOR_MAP.get(prog_str, "#333333")

        points: list[RooflinePoint] = []
        for k_val, k_group in prog_group.groupby("k"):
            flops_val = float(k_group["flops"].iloc[0])
            median_us = float(k_group["time_us"].median())
            ci_lo_us  = float(k_group["time_ci_low"].median())
            ci_hi_us  = float(k_group["time_ci_hi"].median())

            gflops      = flops_val / (median_us * 1e-6) / 1e9
            # Faster time → higher GFLOPs; swap lo/hi when converting
            ci_lo_gflops = flops_val / (ci_hi_us  * 1e-6) / 1e9
            ci_hi_gflops = flops_val / (ci_lo_us  * 1e-6) / 1e9

            points.append(RooflinePoint(intensity, gflops, ci_lo_gflops, ci_hi_gflops))

        # Sort by ascending GFLOPs (k increases along the series)
        points.sort(key=lambda p: p.gflops)
        result.append(RooflineDataSeries(
            label=f"SpaDA {prog_str}",
            points=points,
            color=color,
            marker=STENCIL_MARKER,
            linestyle="",   # no connecting line; each k is an independent point
            fillstyle="full",
        ))

    return result


def load_a100_stencil_series(
    a100_dir: Path,
    flops_csv: Path,
    stencil_k: int | None = None,
    stencil_programs: list[str] | None = None,
    use_streaming_ai: bool = True,
) -> list[RooflineDataSeries]:
    """Load A100 stencil baseline series from a directory of per-stencil CSVs.

    Each CSV must have columns: program_id, program_label, i, j, k, time_us.
    Arithmetic intensity and FLOPs-per-point are derived from *flops_csv*
    (output of ``python -m spatialstencil.cli.count_flop``), whose program
    names are expected to follow the ``<name>_<i>_<j>_<k>`` convention.

    Args:
        a100_dir:          Directory containing per-stencil A100 CSV files.
        flops_csv:         Stencil flops CSV (same one used for WSE-2 stencils).
        stencil_k:         If given, include only rows with this z-depth.
        stencil_programs:  If given, include only these program_label values.
        use_streaming_ai:  If True, use ``StreamingAI`` from *flops_csv* instead
                           of ``ArithmeticIntensity``.  StreamingAI counts each
                           input *field* once (L1-cache-resident model) rather
                           than counting every unique load offset, giving a
                           higher effective AI appropriate for cache-friendly
                           A100 stencil kernels.
    """
    flops_df = pd.read_csv(flops_csv)
    ai_col = "StreamingAI" if (use_streaming_ai and "StreamingAI" in flops_df.columns) else "ArithmeticIntensity"

    ai_lookup: dict[str, float] = {}
    flops_per_point: dict[str, float] = {}
    for _, row in flops_df.iterrows():
        prog = str(row["Program"])
        parts = prog.split("_")
        key = "_".join(parts[:-3]) if len(parts) > 3 else prog
        ai_lookup[key] = float(row[ai_col])
        if len(parts) > 3:
            try:
                ix, jx, kx = int(parts[-3]), int(parts[-2]), int(parts[-1])
                flops_per_point[key] = float(row["Flop"]) / (ix * jx * kx)
            except (ValueError, ZeroDivisionError):
                flops_per_point[key] = 0.0

    frames = [pd.read_csv(p) for p in sorted(a100_dir.glob("*.csv")) if p.is_file()]
    if not frames:
        print(f"Warning: no CSV files found in {a100_dir}")
        return []
    df = pd.concat(frames, ignore_index=True)

    if stencil_programs is not None:
        df = df[df["program_label"].isin(stencil_programs)]
    if stencil_k is not None:
        df = df[df["k"] == stencil_k]

    result: list[RooflineDataSeries] = []

    for prog_name, group in df.groupby("program_label"):
        prog_str = str(prog_name)
        flop_key = STENCIL_NAME_MAP.get(prog_str, prog_str.lower().replace(" ", "_"))
        if flop_key not in ai_lookup:
            raise KeyError(
                f"No flop data found for A100 stencil '{prog_str}' (looked up key '{flop_key}'). "
                f"Available keys: {sorted(ai_lookup)}. "
                f"Re-run 'python -m spatialstencil.cli.count_flop' and update --stencil-flops-csv, "
                f"or add '{prog_str}' to STENCIL_NAME_MAP."
            )

        intensity = ai_lookup[flop_key]
        color = STENCIL_COLOR_MAP.get(prog_str, "#333333")

        points: list[RooflinePoint] = []
        for _, row in group.iterrows():
            actual_flops = (
                flops_per_point[flop_key]
                * float(row["i"]) * float(row["j"]) * float(row["k"])
            )
            gflops = actual_flops / (float(row["time_us"]) * 1e-6) / 1e9
            points.append(RooflinePoint(intensity, gflops, gflops, gflops))

        points.sort(key=lambda p: p.gflops)
        result.append(RooflineDataSeries(
            label=f"A100 {prog_str}",
            points=points,
            color=color,
            marker=STENCIL_MARKER,
            linestyle="",
            fillstyle="none",   # hollow = A100
        ))

    return result


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
FIG_W, FIG_H = 5.8, 8.1
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
    ax.set_xlabel("Arithmetic Intensity [FLOP/Byte]", fontsize=12, fontweight="bold")
    ax.set_ylabel("Performance [GFLOP/s]", fontsize=12, fontweight="bold")
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
        "--gemv-series", type=str, nargs="+", default=None,
        metavar="SERIES",
        help='If set, restrict GEMV data to these series names, e.g. "WSE-2 GEMV Two-phase"',
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
    parser.add_argument(
        "--stencil-flops-csv", type=Path, default=None,
        metavar="PATH",
        help="Stencil flops CSV produced by 'python -m spatialstencil.cli.count_flop'. "
             "Must be paired with --stencil-scaling-csv.",
    )
    parser.add_argument(
        "--stencil-scaling-csv", type=Path, default=None,
        metavar="PATH",
        help="Stencil runtime scaling CSV with columns: Program, time_us, k, flops, ...",
    )
    parser.add_argument(
        "--stencil-k", type=int, default=None,
        metavar="K",
        help="If set, restrict stencil data to rows with this z-depth (k).",
    )
    parser.add_argument(
        "--a100-stencil-dir", type=Path, default=None,
        metavar="DIR",
        help="Directory of A100 stencil baseline CSVs "
             "(columns: program_id, program_label, i, j, k, time_us). "
             "Requires --stencil-flops-csv for FLOPs/AI lookup.",
    )
    parser.add_argument(
        "--a100-stencil-no-streaming-ai", action="store_true", default=False,
        help="Use raw ArithmeticIntensity instead of StreamingAI for A100 stencil "
             "series.  By default StreamingAI is used (L1-cache model: each input "
             "field counted once from DRAM).",
    )
    parser.add_argument(
        "--stencil-programs", nargs="*", default=None,
        metavar="PROG",
        help="If set, show only these stencil program names "
             "(e.g. \"2D Laplacian\" \"Vertical Stencil\"). Shows all when omitted.",
    )
    args = parser.parse_args()

    data_series: list[RooflineDataSeries] = []

    if args.gemv_csv is not None:
        def gemv_filter(df: pd.DataFrame) -> pd.DataFrame:
            df = df[df["shape_label"] == "750x994"]
            if args.k is not None:
                df = df[df["k"] == args.k]
            if args.gemv_series is not None:
                df = df[df["series"].isin(args.gemv_series)]
            return df

        gemv_series = load_series(
            csv_path=args.gemv_csv,
            metrics=ScaledGEMVMetrics(),
            groupby=["series", "source"],
            color_map={},
            color_fn=lambda key: GEMV_COLOR,   # all GEMV same color
            marker=GEMV_MARKER,
            marker_map=SOURCE_MARKER_OVERRIDE,
            marker_index=1,                    # source at key[1]
            fillstyle_map=SOURCE_FILLSTYLE,
            fillstyle_index=1,                 # source at key[1]
            label_fn=lambda key: {
                "WSE-2 GEMV":           "SpaDA GEMV",
                "WSE-2 GEMV Two-phase": "SpaDA GEMV (Two-phase)",
            }.get(key[0], key[0]),
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

        def _reduce_grid_label(key: tuple) -> str:
            """Return 'PX×PY' for the given key tuple, falling back to '512×512'."""
            return (
                f"{int(key[2])}×{int(key[3])}" if not pd.isna(key[2]) else "512×512"
            )

        reduce_series = load_series(
            csv_path=args.reduce_csv,
            metrics=ReduceMetrics(),
            groupby=["method", "source", "px", "py"],
            color_map={},
            color_fn=lambda key: REDUCE_GRID_COLOR.get(_reduce_grid_label(key), "#333333"),
            linestyle_index=1,    # source drives linestyle
            linestyle_map=REDUCE_SOURCE_LINESTYLE,
            dropna=False,         # keep HPDC24 rows that have NaN px/py
            marker=REDUCE_MARKER,
            marker_map=SOURCE_MARKER_OVERRIDE,
            marker_index=1,       # source at key[1]
            fillstyle_map=SOURCE_FILLSTYLE,
            fillstyle_index=1,    # source at key[1]
            label_fn=lambda key: (
                f"{REDUCE_LABEL_MAP.get(key[0], key[0]).removeprefix('SpaDA ')} ({key[1]}, 512×512)"
                if key[1] != "This work"
                else (
                    f"{REDUCE_LABEL_MAP.get(key[0], key[0])} ({int(key[2])}×{int(key[3])})"
                    if not pd.isna(key[2])
                    else REDUCE_LABEL_MAP.get(key[0], key[0])
                )
            ),
            filter_fn=reduce_filter,
        )
        k_desc = f", k={args.reduce_k}" if args.reduce_k is not None else ""
        m_desc = f", methods={args.reduce_methods}" if args.reduce_methods else ""
        print(f"Loaded {len(reduce_series)} reduce series from {args.reduce_csv}{k_desc}{m_desc}")
        data_series += reduce_series

    if args.stencil_flops_csv is not None or args.stencil_scaling_csv is not None:
        if args.stencil_flops_csv is None or args.stencil_scaling_csv is None:
            parser.error("--stencil-flops-csv and --stencil-scaling-csv must be used together")
        if not args.stencil_flops_csv.exists():
            parser.error(f"--stencil-flops-csv does not exist: {args.stencil_flops_csv}")
        if not args.stencil_scaling_csv.exists():
            parser.error(f"--stencil-scaling-csv does not exist: {args.stencil_scaling_csv}")

        stencil_series = load_stencil_series(
            flops_csv=args.stencil_flops_csv,
            scaling_csv=args.stencil_scaling_csv,
            stencil_k=args.stencil_k,
            stencil_programs=args.stencil_programs,
        )
        k_desc = f", k={args.stencil_k}" if args.stencil_k is not None else ""
        p_desc = f", programs={args.stencil_programs}" if args.stencil_programs else ""
        print(f"Loaded {len(stencil_series)} stencil series from {args.stencil_scaling_csv}{k_desc}{p_desc}")
        data_series += stencil_series

    if args.a100_stencil_dir is not None:
        if not args.a100_stencil_dir.is_dir():
            parser.error(f"--a100-stencil-dir does not exist: {args.a100_stencil_dir}")
        if args.stencil_flops_csv is None:
            parser.error("--a100-stencil-dir requires --stencil-flops-csv for FLOPs/AI lookup")
        a100_stencil_series = load_a100_stencil_series(
            a100_dir=args.a100_stencil_dir,
            flops_csv=args.stencil_flops_csv,
            stencil_k=args.stencil_k,
            stencil_programs=args.stencil_programs,
            use_streaming_ai=not args.a100_stencil_no_streaming_ai,
        )
        print(f"Loaded {len(a100_stencil_series)} A100 stencil series from {args.a100_stencil_dir}")
        data_series += a100_stencil_series

    plot_roofline(args.output_dir, data_series)


if __name__ == "__main__":
    main()
