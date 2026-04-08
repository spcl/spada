from __future__ import annotations

import argparse
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BROADCAST_ROOT = REPO_ROOT / "spatialstencil/samples/benchmarks/broadcast_collective_sweep_results"
DEFAULT_REDUCE_ROOT = REPO_ROOT / "spatialstencil/samples/benchmarks/reduce_collective_sweep_results"
DEFAULT_HPDC_ROOT = REPO_ROOT / "hpdc24-results"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "collective_plots"

CYCLE_GHZ = 0.85
F32_BYTES = 4
LOCAL_BENCHMARK_OVERHEAD_CYCLES = 10.0


@dataclass(frozen=True)
class PlotSpec:
    name: str
    title: str
    family: str
    sweep: str
    x_label: str
    local_methods: tuple[str, ...]
    hpdc_files: dict[str, str]
    ymax: float | None = None
    xmin: int | None = None


PLOT_SPECS = (
    PlotSpec(
        name="broadcast_fixed_k",
        title="Broadcast, Fixed K = 1KB",
        family="broadcast",
        sweep="fixed_k",
        x_label="#PEs",
        local_methods=("broadcast_1d_multicast",),
        hpdc_files={"broadcast_1d_multicast": "piotr-broadcast1d-fix-k.csv"},
    ),
    PlotSpec(
        name="broadcast_fixed_pxpy",
        title="Broadcast, Fixed PXxPY = 512x1",
        family="broadcast",
        sweep="fixed_pxpy",
        x_label="Message Size [Bytes]",
        local_methods=("broadcast_1d_multicast",),
        hpdc_files={"broadcast_1d_multicast": "piotr-broadcast1d-fix-px.csv"},
        xmin=8,
    ),
    PlotSpec(
        name="reduce1d_fixed_k",
        title="Reduce 1D, Fixed K = 1KB",
        family="reduce1d",
        sweep="fixed_k",
        x_label="#PEs",
        local_methods=("chain_reduce_1d", "tree_reduce_1d", "twophase_reduce_1d"),
        hpdc_files={
            "chain_reduce_1d": "piotr-reduce1d-chain-fix-k.csv",
            "tree_reduce_1d": "piotr-reduce1d-tree-fix-k.csv",
            "twophase_reduce_1d": "piotr-reduce1d-twophase-fix-k.csv",
        },
    ),
    PlotSpec(
        name="reduce1d_fixed_pxpy",
        title="Reduce 1D, Fixed PXxPY = 512x1",
        family="reduce1d",
        sweep="fixed_pxpy",
        x_label="Message Size [Bytes]",
        local_methods=("chain_reduce_1d", "tree_reduce_1d", "twophase_reduce_1d"),
        hpdc_files={
            "chain_reduce_1d": "piotr-reduce1d-chain-fix-px.csv",
            "tree_reduce_1d": "piotr-reduce1d-tree-fix-px.csv",
            "twophase_reduce_1d": "piotr-reduce1d-twophase-fix-px.csv",
        },
        xmin=8,
        ymax=20,
    ),
    PlotSpec(
        name="reduce2d_fixed_k",
        title="Reduce 2D, Fixed K = 1KB",
        family="reduce2d",
        sweep="fixed_k",
        x_label="#PEs",
        local_methods=("chain_reduce_2d", "tree_reduce_2d", "twophase_reduce_2d"),
        hpdc_files={
            "chain_reduce_2d": "piotr-reduce2d-chain-fix-k.csv",
            "tree_reduce_2d": "piotr-reduce2d-tree-fix-k.csv",
            "twophase_reduce_2d": "piotr-reduce2d-twophase-fix-k.csv",
        },
    ),
    PlotSpec(
        name="reduce2d_fixed_pxpy",
        title="Reduce 2D, Fixed PXxPY = 512x512",
        family="reduce2d",
        sweep="fixed_pxpy",
        x_label="Message Size [Bytes]",
        local_methods=("chain_reduce_2d", "tree_reduce_2d", "twophase_reduce_2d"),
        hpdc_files={
            "chain_reduce_2d": "piotr-reduce2d-chain-fix-pxpy.csv",
            "tree_reduce_2d": "piotr-reduce2d-tree-fix-pxpy.csv",
            "twophase_reduce_2d": "piotr-reduce2d-twophase-fix-pxpy.csv",
        },
        ymax=30,
        xmin=8,
    ),
)


LOCAL_LABELS = {
    "broadcast_1d": "Broadcast 1D",
    "broadcast_1d_multicast": "Broadcast 1D Multicast",
    "chain_reduce_1d": "Chain",
    "tree_reduce_1d": "Tree",
    "twophase_reduce_1d": "Two-phase",
    "chain_reduce_2d": "Chain",
    "tree_reduce_2d": "Tree",
    "twophase_reduce_2d": "Two-phase",
}


COLOR_BY_METHOD = {
    "broadcast_1d": "#0B6E4F",
    "broadcast_1d_multicast": "#C84C09",
    "chain_reduce_1d": "#005F73",
    "tree_reduce_1d": "#9B2226",
    "twophase_reduce_1d": "#EE9B00",
    "chain_reduce_2d": "#005F73",
    "tree_reduce_2d": "#9B2226",
    "twophase_reduce_2d": "#EE9B00",
}


MARKER_BY_METHOD = {
    "broadcast_1d": "o",
    "broadcast_1d_multicast": "s",
    "chain_reduce_1d": "o",
    "tree_reduce_1d": "^",
    "twophase_reduce_1d": "s",
    "chain_reduce_2d": "o",
    "tree_reduce_2d": "^",
    "twophase_reduce_2d": "s",
}


def cycles_to_us(cycles: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(cycles)-10) / CYCLE_GHZ * 1e-3


def bootstrap_median_ci(values: np.ndarray, rng: np.random.Generator, n_bootstrap: int) -> tuple[float, float]:
    if len(values) <= 1:
        value = float(values[0])
        return value, value

    bootstrap_medians = np.empty(n_bootstrap, dtype=np.float64)
    for idx in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        bootstrap_medians[idx] = np.median(sample)
    return float(np.percentile(bootstrap_medians, 2.5)), float(np.percentile(bootstrap_medians, 97.5))


def canonical_perf_files(case_dir: Path) -> list[Path]:
    numeric_files: dict[int, Path] = {}
    for perf_file in case_dir.glob("perf_cycles_*.npy"):
        suffix = perf_file.stem.removeprefix("perf_cycles_")
        if not suffix.isdigit():
            continue
        index = int(suffix)
        current = numeric_files.get(index)
        if current is None or len(suffix) > len(current.stem.removeprefix("perf_cycles_")):
            numeric_files[index] = perf_file

    if numeric_files:
        return [numeric_files[idx] for idx in sorted(numeric_files)]

    single_file = case_dir / "perf_cycles.npy"
    if single_file.exists():
        return [single_file]

    return []


def summarize_case(case_dir: Path, rng: np.random.Generator, n_bootstrap: int) -> tuple[float, float, float] | None:
    perf_files = canonical_perf_files(case_dir)
    if not perf_files:
        return None

    per_run_max = np.array([np.max(np.load(perf_file)) for perf_file in perf_files], dtype=np.float64)
    median_cycles = float(np.median(per_run_max))
    ci_low_cycles, ci_high_cycles = bootstrap_median_ci(per_run_max, rng, n_bootstrap)
    return float(cycles_to_us(median_cycles)), float(cycles_to_us(ci_low_cycles)), float(cycles_to_us(ci_high_cycles))


def parse_case_dir(case_dir: Path) -> tuple[int, int, int]:
    pxpy_match = re.fullmatch(r"PX_(\d+)_PY_(\d+)", case_dir.parent.name)
    if pxpy_match is None:
        raise ValueError(f"Unexpected PX/PY directory: {case_dir.parent}")

    k_match = re.fullmatch(r"K_(\d+)_(.+)", case_dir.name)
    if k_match is None:
        raise ValueError(f"Unexpected K directory: {case_dir}")

    px = int(pxpy_match.group(1))
    py = int(pxpy_match.group(2))
    k_count = int(k_match.group(1))
    return px, py, k_count


def collect_local_results(root: Path, family: str, sweep: str, methods: tuple[str, ...], n_bootstrap: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(0)
    sweep_dir = "sweep1_fixed_1KB" if sweep == "fixed_k" else "sweep2_fixed_pxpy"

    for method in methods:
        method_root = root / sweep_dir / method
        if not method_root.exists():
            continue

        for case_dir in sorted(method_root.glob("PX_*_PY_*/K_*")):
            summary = summarize_case(case_dir, rng, n_bootstrap)
            if summary is None:
                continue

            px, py, k_count = parse_case_dir(case_dir)
            if sweep == "fixed_k":
                x_value = px
            else:
                x_value = k_count * F32_BYTES

            rows.append(
                {
                    "family": family,
                    "sweep": sweep,
                    "method": method,
                    "source": "This work",
                    "x": float(x_value),
                    "time_us": summary[0],
                    "ci_low_us": summary[1],
                    "ci_high_us": summary[2],
                    "px": px,
                    "py": py,
                    "k_count": k_count,
                }
            )

    return pd.DataFrame(rows)


def load_hpdc_csv(csv_path: Path, method: str, family: str, sweep: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [column.strip() for column in df.columns]
    return pd.DataFrame(
        {
            "family": family,
            "sweep": sweep,
            "method": method,
            "source": "HPDC24",
            "x": pd.to_numeric(df["x"]),
            "time_us": pd.to_numeric(df["y"]),
            "ci_low_us": np.nan,
            "ci_high_us": np.nan,
        }
    )


def collect_plot_dataframe(
    spec: PlotSpec,
    broadcast_root: Path,
    reduce_root: Path,
    hpdc_root: Path,
    n_bootstrap: int,
    include_local: bool,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    if include_local:
        if spec.family == "broadcast":
            local_root = broadcast_root
        else:
            local_root = reduce_root
        frames.append(collect_local_results(local_root, spec.family, spec.sweep, spec.local_methods, n_bootstrap))

    for method, filename in spec.hpdc_files.items():
        frames.append(load_hpdc_csv(hpdc_root / filename, method, spec.family, spec.sweep))

    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        return df

    df["label"] = df.apply(
        lambda row: f"{LOCAL_LABELS[row['method']]} ({row['source']})",
        axis=1,
    )
    return df.sort_values(["method", "source", "x"]).reset_index(drop=True)


def set_x_axis(ax: plt.Axes, spec: PlotSpec, x_values: np.ndarray) -> None:
    if len(x_values) == 0:
        return

    positive_values = x_values[x_values > 0]
    if len(positive_values) > 0:
        ax.set_xscale("log", base=2)
        ticks = sorted({int(value) for value in positive_values if math.isclose(value, round(value), rel_tol=0, abs_tol=1e-9)})
        if ticks:
            ax.set_xticks(ticks)
            if spec.sweep == "fixed_pxpy":
                # Convert to B/KB human readable notation
                ax.set_xticklabels([f"{int(tick) // 1024} KB" if int(tick) >= 1024 else f"{int(tick)} B" for tick in ticks])
            else:
                # If 2d, use 512x512, if 1d, use 512x1
                if "2d" in spec.family:
                    ax.set_xticklabels([f"{tick}x{tick}" for tick in ticks])
                else:
                    ax.set_xticklabels([f"{tick}x1" for tick in ticks])


def plot_spec(df: pd.DataFrame, spec: PlotSpec, output_dir: Path, show_ci: bool) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.4))

    methods_in_plot = [method for method in spec.local_methods if method in set(df["method"])]

    for method in methods_in_plot:
        color = COLOR_BY_METHOD[method]
        marker = MARKER_BY_METHOD[method]
        if spec.xmin is not None:
            df = df[df.x >= spec.xmin]


        local = df[(df["method"] == method) & (df["source"] == "This work")].sort_values("x")
        hpdc = df[(df["method"] == method) & (df["source"] == "HPDC24")].sort_values("x")

        if not local.empty:
            ax.plot(
                local["x"],
                local["time_us"],
                color=color,
                marker=marker,
                linewidth=2.4,
                markersize=7,
                label=f"{LOCAL_LABELS[method]} (This work)",
            )
            if show_ci and local["ci_low_us"].notna().all():
                ax.fill_between(
                    local["x"].to_numpy(dtype=float),
                    local["ci_low_us"].to_numpy(dtype=float),
                    local["ci_high_us"].to_numpy(dtype=float),
                    color=color,
                    alpha=0.18,
                    linewidth=0,
                )

        if not hpdc.empty:
            ax.plot(
                hpdc["x"],
                hpdc["time_us"],
                color=color,
                marker=marker,
                linestyle="--",
                linewidth=2.0,
                markersize=6,
                markerfacecolor="white",
                label=f"{LOCAL_LABELS[method]} (HPDC24)",
            )

    # ax.set_title(spec.title, fontsize=13, fontweight="bold")
    ax.set_xlabel(spec.x_label, fontsize=12, fontweight="bold")
    ax.set_ylabel("Runtime [μs]", fontsize=12, fontweight="bold")
    ax.grid(True, which="major", alpha=0.3)
    if spec.ymax is not None:
        ymin, _ = ax.get_ylim()
        ax.set_ylim(ymin, spec.ymax)

    set_x_axis(ax, spec, df["x"].to_numpy(dtype=float))

    ax.legend(fontsize=9, title_fontsize=10, loc="best")

    plt.tight_layout()
    fig.savefig(output_dir / f"{spec.name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{spec.name}.pdf", bbox_inches="tight")
    plt.close(fig)

    df.to_csv(output_dir / f"{spec.name}_df.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot collective benchmark sweeps against HPDC24 baselines.")
    parser.add_argument("--broadcast-root", type=Path, default=DEFAULT_BROADCAST_ROOT, help="Broadcast results root")
    parser.add_argument("--reduce-root", type=Path, default=DEFAULT_REDUCE_ROOT, help="Reduce results root")
    parser.add_argument("--hpdc-root", type=Path, default=DEFAULT_HPDC_ROOT, help="HPDC24 CSV root")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory for plots")
    parser.add_argument("--show-ci", action="store_true", help="Plot 95%% bootstrap confidence intervals for local runs")
    parser.add_argument("--bootstrap-samples", type=int, default=2000, help="Bootstrap samples for local CI estimation")
    parser.add_argument("--baseline-only", action="store_true", help="Plot only HPDC24 baseline series")
    args = parser.parse_args()

    sns.set_style("whitegrid")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for spec in PLOT_SPECS:
        df = collect_plot_dataframe(
            spec,
            args.broadcast_root,
            args.reduce_root,
            args.hpdc_root,
            args.bootstrap_samples,
            include_local=not args.baseline_only,
        )
        if df.empty:
            print(f"Skipping {spec.name}: no data found")
            continue
        print(f"Plotting {spec.name}: {len(df)} series points")
        plot_spec(df, spec, args.output_dir, args.show_ci)


if __name__ == "__main__":
    main()
