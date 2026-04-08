"""Plot GEMV benchmark sweep results for WSE-2 and (optionally) A100 baselines.

Usage
-----
Plot from a pre-built CSV (skips raw result scanning):
    python plot_gemv_sweep.py --from-csv samples/benchmarks/plots/gemv_blas_sweep_results/gemv_750_994.csv

Basic plot (WSE-2 only, linear y-axis, scans raw results):
    python plot_gemv_sweep.py

With A100 baselines and 95% CI bands:
    python plot_gemv_sweep.py --with-baselines --show-ci

Log y-axis (useful when runtimes span multiple orders of magnitude):
    python plot_gemv_sweep.py --log-y

Log-log plot (both axes logarithmic):
    python plot_gemv_sweep.py --log-log

Key flags
---------
--from-csv PATH         Load a pre-built CSV saved by this script instead of
                        scanning --results-root. Mutually exclusive with
                        --with-baselines.
--results-root PATH     Directory containing local WSE benchmark results
                        (default: …/gemv_blas_sweep_results)
--output-dir PATH       Destination for .png, .pdf, and .csv outputs
                        (default: ./gemv_plots/)
--with-baselines        Overlay A100 CUBLAS timings
--a100-csv PATH         Path to the parsed A100 CSV file
--a100-column           Time column to use: kernel_median_us (default) or app_median_us
--show-ci               Shade 95% bootstrap confidence intervals around WSE lines
--bootstrap-samples N   Bootstrap resamples for CI estimation (default: 2000)
--log-y                 Log scale on the y-axis only
--log-log               Log scale on both axes (mutually exclusive with --log-y)

Output filename suffixes encode the scale mode:
    gemv_750_994.png        -> linear y
    gemv_750_994_logy.png   -> --log-y
    gemv_750_994_loglog.png -> --log-log
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
LOCAL_RESULTS_ROOT = REPO_ROOT / "spatialstencil/samples/benchmarks/gemv_blas_sweep_results"
A100_GEMV_CSV = REPO_ROOT / "a100-results/csv/a100_gemv_all.csv"
OUTPUT_DIR = SCRIPT_DIR / "gemv_plots"
LOCAL_BENCHMARK_OVERHEAD_CYCLES = 10.0
CYCLE_GHZ = 0.85

LOCAL_METHOD_LABELS = {
    "gemv": "WSE-2 GEMV",
    "gemv_twophase": "WSE-2 GEMV Two-phase",
}

COLORS = {
    "gemv": "#005F73",
    "gemv_twophase": "#CA6702",
    "a100": "#7A7A7A",
}

MARKERS = {
    "gemv": "o",
    "gemv_twophase": "s",
    "a100": "D",
}


def cycles_to_us(cycles: np.ndarray | float) -> np.ndarray | float:
    adjusted = np.maximum(np.asarray(cycles, dtype=np.float64) - LOCAL_BENCHMARK_OVERHEAD_CYCLES, 0.0)
    return adjusted / CYCLE_GHZ * 1e-3


def bootstrap_median_ci(values: np.ndarray, rng: np.random.Generator, n_bootstrap: int) -> tuple[float, float]:
    if len(values) <= 1:
        value = float(values[0])
        return value, value

    medians = np.empty(n_bootstrap, dtype=np.float64)
    for idx in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        medians[idx] = np.median(sample)
    return float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))


def canonical_perf_files(case_dir: Path) -> list[Path]:
    numeric_files: dict[int, Path] = {}
    for perf_file in case_dir.glob("perf_cycles_*.npy"):
        suffix = perf_file.stem.removeprefix("perf_cycles_")
        if not suffix.isdigit():
            continue
        idx = int(suffix)
        current = numeric_files.get(idx)
        if current is None or len(suffix) > len(current.stem.removeprefix("perf_cycles_")):
            numeric_files[idx] = perf_file
    return [numeric_files[idx] for idx in sorted(numeric_files)]


def summarize_local_case(case_dir: Path, rng: np.random.Generator, n_bootstrap: int) -> tuple[float, float, float] | None:
    perf_files = canonical_perf_files(case_dir)
    if not perf_files:
        return None

    per_run_max = np.array([np.max(np.load(perf_file)) for perf_file in perf_files], dtype=np.float64)
    median_cycles = float(np.median(per_run_max))
    ci_low_cycles, ci_high_cycles = bootstrap_median_ci(per_run_max, rng, n_bootstrap)
    return float(cycles_to_us(median_cycles)), float(cycles_to_us(ci_low_cycles)), float(cycles_to_us(ci_high_cycles))


def collect_local_dataframe(results_root: Path, n_bootstrap: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(0)

    for grid_dir in sorted(results_root.glob("grid_*_*")):
        try:
            _, px_str, py_str = grid_dir.name.split("_")
        except ValueError:
            continue

        px = int(px_str)
        py = int(py_str)
        shape_label = f"{px}x{py}"

        for method in ("gemv", "gemv_twophase"):
            method_root = grid_dir / method / f"PX_{px}_PY_{py}"
            if not method_root.exists():
                continue

            for case_dir in sorted(method_root.glob("K_*")):
                try:
                    k = int(case_dir.name.split("_")[1])
                except (IndexError, ValueError):
                    continue

                summary = summarize_local_case(case_dir, rng, n_bootstrap)
                if summary is None:
                    continue

                rows.append(
                    {
                        "shape_label": shape_label,
                        "px": px,
                        "py": py,
                        "k": k,
                        "series": LOCAL_METHOD_LABELS[method],
                        "source": "WSE",
                        "time_us": summary[0],
                        "ci_low_us": summary[1],
                        "ci_high_us": summary[2],
                    }
                )

    return pd.DataFrame(rows)


def collect_a100_dataframe(csv_path: Path, column: str) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    return pd.DataFrame(
        {
            "shape_label": df["shape_label"],
            "px": df["px"],
            "py": df["py"],
            "k": df["k"],
            "series": "A100 CUBLAS",
            "source": "A100",
            "time_us": df[column],
            "ci_low_us": np.nan,
            "ci_high_us": np.nan,
        }
    )


MAX_K = 32


def plot_shape(df: pd.DataFrame, shape_label: str, output_dir: Path, show_ci: bool, log_y: bool = False, log_x: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    shape_df = df[(df["shape_label"] == shape_label) & (df["k"] <= MAX_K)].copy()
    shape_df["n_elements"] = shape_df["px"] * shape_df["py"] * shape_df["k"] ** 2
    shape_df = shape_df.sort_values("n_elements")

    local_series = ["WSE-2 GEMV", "WSE-2 GEMV Two-phase"]
    for series in local_series:
        series_df = shape_df[shape_df["series"] == series]
        if series_df.empty:
            continue

        method_key = "gemv" if series == "WSE-2 GEMV" else "gemv_twophase"
        ax.plot(
            series_df["n_elements"],
            series_df["time_us"],
            color=COLORS[method_key],
            marker=MARKERS[method_key],
            linewidth=2.4,
            markersize=7,
            label=series,
        )
        if show_ci:
            ax.fill_between(
                series_df["n_elements"].to_numpy(dtype=float),
                series_df["ci_low_us"].to_numpy(dtype=float),
                series_df["ci_high_us"].to_numpy(dtype=float),
                color=COLORS[method_key],
                alpha=0.18,
                linewidth=0,
            )

    a100_df = shape_df[shape_df["source"] == "A100"]
    if not a100_df.empty:
        a100_label = str(a100_df["series"].iloc[0])
        ax.plot(
            a100_df["n_elements"],
            a100_df["time_us"],
            color=COLORS["a100"],
            marker=MARKERS["a100"],
            markerfacecolor="white",
            linewidth=2.0,
            linestyle="--",
            markersize=7,
            label=a100_label,
        )

    def _draw_speedup_arrow(
        x_annot: float,
        y_lo: float,
        y_hi: float,
        label_side: str = "left",
        mutation_scale: float = 20,
        linestyle: str = "dotted",
    ) -> None:
        speedup = y_hi / y_lo
        y_mid = np.sqrt(y_lo * y_hi) if log_y else (y_lo + y_hi) / 2
        ax.annotate(
            "",
            xy=(x_annot, y_hi),
            xytext=(x_annot, y_lo),
            arrowprops=dict(
                arrowstyle="<->",
                color="#333333",
                lw=1.5,
                linestyle=linestyle,
                mutation_scale=mutation_scale,
            ),
        )
        if label_side == "right":
            x_text, ha = x_annot * 1.08, "left"
        else:
            x_text, ha = x_annot / 1.08, "right"
        ax.text(
            x_text,
            y_mid,
            f"{speedup:.1f}×",
            va="center",
            ha=ha,
            fontsize=10,
            color="#333333",
            fontweight="bold",
        )

    # Speedup annotation at K=MAX_K between fastest WSE series and A100
    wse_at_max_k = shape_df[(shape_df["source"] == "WSE") & (shape_df["k"] == MAX_K)]
    a100_at_max_k = shape_df[(shape_df["source"] == "A100") & (shape_df["k"] == MAX_K)]
    if not wse_at_max_k.empty and not a100_at_max_k.empty:
        best_wse_row = wse_at_max_k.loc[wse_at_max_k["time_us"].idxmin()]
        a100_row = a100_at_max_k.iloc[0]
        _draw_speedup_arrow(
            float(best_wse_row["n_elements"]),
            float(best_wse_row["time_us"]),
            float(a100_row["time_us"]),
        )

    # Speedup annotation at K=min_k between the two WSE implementations
    min_k = int(shape_df[shape_df["source"] == "WSE"]["k"].min())
    gemv_at_min_k = shape_df[(shape_df["series"] == "WSE-2 GEMV") & (shape_df["k"] == min_k)]
    twophase_at_min_k = shape_df[(shape_df["series"] == "WSE-2 GEMV Two-phase") & (shape_df["k"] == min_k)]
    if not gemv_at_min_k.empty and not twophase_at_min_k.empty:
        y_gemv = float(gemv_at_min_k.iloc[0]["time_us"])
        y_twophase = float(twophase_at_min_k.iloc[0]["time_us"])
        x_annot = float(gemv_at_min_k.iloc[0]["n_elements"])
        _draw_speedup_arrow(x_annot, min(y_gemv, y_twophase), max(y_gemv, y_twophase), label_side="right", mutation_scale=12, linestyle="solid")

    # Speedup annotation at K=min_k between slower WSE implementation and A100
    a100_at_min_k = shape_df[(shape_df["source"] == "A100") & (shape_df["k"] == min_k)]
    wse_at_min_k = shape_df[(shape_df["source"] == "WSE") & (shape_df["k"] == min_k)]
    if not wse_at_min_k.empty and not a100_at_min_k.empty:
        slower_wse_row = wse_at_min_k.loc[wse_at_min_k["time_us"].idxmax()]
        a100_row = a100_at_min_k.iloc[0]
        _draw_speedup_arrow(
            float(slower_wse_row["n_elements"]),
            float(slower_wse_row["time_us"]),
            float(a100_row["time_us"]),
            label_side="right",
            mutation_scale=20,
            linestyle="dotted",
        )

    px, py = map(int, shape_label.split("x"))
    # ax.set_title(f"GEMV Sweep, PXxPY={px}x{py}", fontsize=13, fontweight="bold")
    ax.set_xlabel("# Matrix Elements", fontsize=12, fontweight="bold")
    ax.set_ylabel("Runtime [μs]", fontsize=12, fontweight="bold")
    tick_vals = sorted(shape_df["n_elements"].unique())
    def _sci_label(x: float, _pos: object) -> str:
        exp = int(np.floor(np.log10(abs(x)))) if x != 0 else 0
        coeff = x / 10**exp
        if abs(coeff - round(coeff)) < 1e-9:
            coeff = int(round(coeff))
            return rf"${coeff}\times10^{{{exp}}}$"
        return rf"${coeff:.2g}\times10^{{{exp}}}$"

    if log_x:
        ax.set_xscale("log", base=10)
        ax.set_xticks(tick_vals)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_sci_label))
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    else:
        ax.set_xticks(tick_vals)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_sci_label))
    if log_y:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, numticks=15))
        ax.yaxis.set_major_formatter(matplotlib.ticker.LogFormatterSciNotation(labelOnlyBase=False))
        ax.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=15))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.LogFormatterSciNotation(labelOnlyBase=False, minor_thresholds=(2, 0.4)))
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.12)
    ax.legend(title_fontsize=10, fontsize=9, loc="best")

    scale_suffix = ""
    if log_x and log_y:
        scale_suffix = "_loglog"
    elif log_y:
        scale_suffix = "_logy"

    plt.tight_layout()
    base = output_dir / f"gemv_{shape_label.replace('x', '_')}{scale_suffix}"
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    df.to_csv(base.with_suffix(".csv"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot GEMV benchmark sweeps against A100 baselines.")
    parser.add_argument("--results-root", type=Path, default=LOCAL_RESULTS_ROOT, help="Local GEMV result root")
    parser.add_argument("--a100-csv", type=Path, default=A100_GEMV_CSV, help="Parsed A100 GEMV CSV")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory for plots")
    parser.add_argument("--with-baselines", action="store_true", help="Overlay A100 baseline data")
    parser.add_argument("--show-ci", action="store_true", help="Plot 95%% bootstrap confidence intervals for local runs")
    parser.add_argument(
        "--a100-column",
        choices=("kernel_median_us", "app_median_us"),
        default="kernel_median_us",
        help="Which parsed A100 time column to plot",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000, help="Bootstrap samples for local CI estimation")
    parser.add_argument(
        "--from-csv",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Load a pre-built CSV (previously saved by this script) instead of scanning "
            "--results-root.  The file must contain columns: shape_label, px, py, k, "
            "series, source, time_us, ci_low_us, ci_high_us.  "
            "Incompatible with --with-baselines and --results-root."
        ),
    )
    scale_group = parser.add_mutually_exclusive_group()
    scale_group.add_argument("--log-y", action="store_true", help="Use log scale on the y-axis (x remains log base-2)")
    scale_group.add_argument("--log-log", action="store_true", help="Use log scale on both axes (log-log plot)")
    args = parser.parse_args()

    if args.from_csv is not None and args.with_baselines:
        parser.error("--from-csv and --with-baselines are mutually exclusive")

    log_y = args.log_y or args.log_log
    log_x = True  # x is always log base-2; --log-log makes the y label explicit

    sns.set_style("whitegrid")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.from_csv is not None:
        if not args.from_csv.exists():
            parser.error(f"--from-csv does not exist: {args.from_csv}")
        df = pd.read_csv(args.from_csv, index_col=0)
        required = {"shape_label", "px", "py", "k", "series", "source", "time_us", "ci_low_us", "ci_high_us"}
        missing = required - set(df.columns)
        if missing:
            parser.error(f"--from-csv is missing required columns: {', '.join(sorted(missing))}")
    else:
        if not args.results_root.exists():
            parser.error(f"--results-root does not exist: {args.results_root}")

        if args.with_baselines and not args.a100_csv.exists():
            parser.error(f"--a100-csv does not exist: {args.a100_csv}  (omit --with-baselines to skip A100 data)")

        local_df = collect_local_dataframe(args.results_root, args.bootstrap_samples)
        if local_df.empty:
            parser.error(
                f"No benchmark results found under {args.results_root}\n"
                "  Expected subdirectories of the form grid_<PX>_<PY>/<method>/PX_<PX>_PY_<PY>/K_<K>/"
            )

        a100_df = collect_a100_dataframe(args.a100_csv, args.a100_column) if args.with_baselines else pd.DataFrame()
        if args.with_baselines and a100_df.empty:
            print(f"Warning: A100 CSV loaded but contained no rows: {args.a100_csv}")

        df = pd.concat([local_df, a100_df], ignore_index=True)

    for shape_label in sorted(df["shape_label"].unique()):
        plot_shape(df, shape_label, args.output_dir, args.show_ci, log_y=log_y, log_x=log_x)
        print(f"Plotted {shape_label}")


if __name__ == "__main__":
    main()
