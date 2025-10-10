import numpy as np
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
import seaborn as sns
from pathlib import Path
import re


def load_runtime_data(folder_path):
    """Load all numpy files from a folder and return combined runtime data."""
    folder = Path(folder_path)
    runtimes = []

    for npy_file in folder.glob("perf*.npy"):
        data = np.load(npy_file)
        runtimes.extend(data.flatten())

    if not runtimes:
        return None

    return np.array(runtimes)


def compute_speedup_stats(runtimes_a, runtimes_b, n_bootstrap=1000):
    """
    Compute median speedup and 95% CI using bootstrap.
    Speedup = runtime_a / runtime_b (speedup of b over a)
    """
    # NOTE: This was too expensive to compute and all of the confidence
    #       intervals were < 0.01

    # Compute speedups for all combinations
    median_speedup = np.median(runtimes_a) / np.median(runtimes_b)
    # speedups = speedups.flatten()

    # Median speedup
    # median_speedup = np.median(speedups)

    # Bootstrap for 95% CI
    # bootstrap_medians = []
    # for _ in range(n_bootstrap):
    #     sample = np.random.choice(speedups, size=len(speedups), replace=True)
    #     bootstrap_medians.append(np.median(sample))

    # ci_lower = np.percentile(bootstrap_medians, 2.5)
    # ci_upper = np.percentile(bootstrap_medians, 97.5)
    ci_lower = ci_upper = median_speedup

    return median_speedup, ci_lower, ci_upper


def parse_folder_name(folder_name):
    """Extract N and K values from folder name like 'N_100_K_50'."""
    match = re.search(r"N_(\d+)_K_(\d+)", folder_name)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def create_speedup_heatmap(base_path, output_file="speedup_heatmap.pdf"):
    """
    Create a heatmap of speedups from folder structure.

    Parameters:
    -----------
    base_path : str
        Base directory containing N_*_K_* folders
    output_file : str
        Output filename for the heatmap
    """
    base = Path(base_path) / "reduce"

    # Collect all N_*_K_* folders
    folders = [f for f in base.glob("N_*_K_*") if f.is_dir()]

    # Parse N and K values and collect data
    data_dict = {}

    for folder in folders:
        n_val, k_val = parse_folder_name(folder.name)
        if n_val is None or k_val is None:
            continue

        app_a_path = Path(base_path) / "reduce" / folder.name
        app_b_path = Path(base_path) / "reduce_pipelined" / folder.name

        if not app_a_path.exists() or not app_b_path.exists():
            print(f"Warning: Missing reduce or reduce_pipelined in {folder.name}")
            continue

        # Load runtime data
        runtimes_a = load_runtime_data(app_a_path)
        runtimes_b = load_runtime_data(app_b_path)
        if runtimes_a is None or runtimes_b is None:
            print(f"Warning: No data in {folder.name}")
            continue

        # Compute speedup statistics
        median, ci_lower, ci_upper = compute_speedup_stats(runtimes_a, runtimes_b)

        data_dict[(n_val, k_val)] = {
            "median": median,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "ci_width": ci_upper - ci_lower,
        }

        median_a = np.median(runtimes_a)
        median_b = np.median(runtimes_b)

        print(
            f"N={n_val}, K={k_val}: Median cycles: {median_a}, {median_b}; Speedup={median:.2f} [{ci_lower:.2f}, {ci_upper:.2f}]"
        )

    # Extract unique N and K values and sort them
    n_values = sorted(set(nk[0] for nk in data_dict.keys()))
    k_values = sorted(set(nk[1] for nk in data_dict.keys()))

    # Create matrices for heatmap
    median_matrix = np.full((len(k_values), len(n_values)), np.nan)
    ci_width_matrix = np.full((len(k_values), len(n_values)), np.nan)

    for i, k_val in enumerate(k_values):
        for j, n_val in enumerate(n_values):
            if (n_val, k_val) in data_dict:
                median_matrix[i, j] = data_dict[(n_val, k_val)]["median"]
                ci_width_matrix[i, j] = data_dict[(n_val, k_val)]["ci_width"]

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 1, figsize=(12, 6))

    # Create custom colormap: red-yellow for <1, green for >1
    colors_lower = ["darkred", "red", "orange", "yellow"]
    colors_upper = ["lightgreen", "green", "green", "darkgreen"]
    n_bins_lower = 1000
    n_bins_upper = 1000

    cmap = LinearSegmentedColormap.from_list("speedup", colors_lower + colors_upper, N=n_bins_lower + n_bins_upper)

    # Use TwoSlopeNorm to center at 1.0
    vmin = np.nanmin(median_matrix)
    vmax = np.nanmax(median_matrix)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)

    # Plot 1: Median speedup heatmap
    im1 = axes.imshow(median_matrix, aspect="auto", cmap=cmap, norm=norm)

    axes.set_xticks(range(len(n_values)))
    axes.set_yticks(range(len(k_values)))
    axes.set_xticklabels(n_values, fontsize=16)
    axes.set_yticklabels(k_values, fontsize=16)
    axes.set_xlabel("#PEs", fontsize=24)
    axes.set_ylabel("Reduced Elements", fontsize=24)

    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=axes)

    # Set explicit colorbar ticks
    tick_values = [vmin]
    # Add ticks below 1.0
    for val in [0.5, 0.75]:
        if vmin < val < 1.0:
            tick_values.append(val)
    tick_values.append(1.0)
    # Add ticks above 1.0
    for val in [20.0, 40]:
        if 1.0 < val < vmax:
            tick_values.append(val)
    tick_values.append(vmax)
    cbar1.set_ticks(sorted(set(tick_values)))
    cbar1.ax.tick_params(labelsize=16)  # Set tick label font size
    cbar1.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.2f}" if x < 1 else f"{x:.0f}"))
    cbar1.set_label("Median Speedup (pipelined over vectorized)", fontsize=14)

    # Add text annotations
    for i in range(len(k_values)):
        for j in range(len(n_values)):
            if not np.isnan(median_matrix[i, j]):
                axes.text(
                    j,
                    i,
                    f"{median_matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="black" if 0.5 < median_matrix[i, j] < 20 else "white",
                    fontsize=12,
                )

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches="tight")
    plt.savefig(output_file + ".png", dpi=300, bbox_inches="tight")
    print(f"\nHeatmap saved to {output_file}")

    return data_dict, median_matrix, ci_width_matrix


# Example usage:
if __name__ == "__main__":
    base_path = "benchmark_results_sweep"

    results, median_mat, ci_mat = create_speedup_heatmap(base_path)
