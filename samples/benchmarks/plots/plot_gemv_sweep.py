from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
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


def plot_shape(df: pd.DataFrame, shape_label: str, output_dir: Path, show_ci: bool) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    shape_df = df[df["shape_label"] == shape_label].sort_values("k")

    local_series = ["WSE-2 GEMV", "WSE-2 GEMV Two-phase"]
    for series in local_series:
        series_df = shape_df[shape_df["series"] == series]
        if series_df.empty:
            continue

        method_key = "gemv" if series == "WSE-2 GEMV" else "gemv_twophase"
        ax.plot(
            series_df["k"],
            series_df["time_us"],
            color=COLORS[method_key],
            marker=MARKERS[method_key],
            linewidth=2.4,
            markersize=7,
            label=series,
        )
        if show_ci:
            ax.fill_between(
                series_df["k"].to_numpy(dtype=float),
                series_df["ci_low_us"].to_numpy(dtype=float),
                series_df["ci_high_us"].to_numpy(dtype=float),
                color=COLORS[method_key],
                alpha=0.18,
                linewidth=0,
            )

    a100_df = shape_df[shape_df["series"] == "A100 CUBLAS"]
    if not a100_df.empty:
        ax.plot(
            a100_df["k"],
            a100_df["time_us"],
            color=COLORS["a100"],
            marker=MARKERS["a100"],
            markerfacecolor="white",
            linewidth=2.0,
            linestyle="--",
            markersize=7,
            label="A100 CUBLAS",
        )

    px, py = map(int, shape_label.split("x"))
    # ax.set_title(f"GEMV Sweep, PXxPY={px}x{py}", fontsize=13, fontweight="bold")
    ax.set_xlabel("K", fontsize=12, fontweight="bold")
    ax.set_ylabel("Runtime [μs]", fontsize=12, fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(shape_df["k"].unique()))
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(title_fontsize=10, fontsize=9, loc="best")

    plt.tight_layout()
    base = output_dir / f"gemv_{shape_label.replace('x', '_')}"
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
    args = parser.parse_args()

    sns.set_style("whitegrid")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    local_df = collect_local_dataframe(args.results_root, args.bootstrap_samples)
    a100_df = collect_a100_dataframe(args.a100_csv, args.a100_column) if args.with_baselines else pd.DataFrame()
    df = pd.concat([local_df, a100_df], ignore_index=True)

    for shape_label in sorted(df["shape_label"].unique()):
        plot_shape(df, shape_label, args.output_dir, args.show_ci)
        print(f"Plotted {shape_label}")


if __name__ == "__main__":
    main()
