import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import re

# Avoid Type 3 fonts for publication-quality figures
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


def load_performance_data(folder_path):
    """Load all perf*.npy files from a folder and return combined performance data."""
    folder = Path(folder_path)
    performance = []

    for npy_file in folder.glob("perf*.npy"):
        data = np.load(npy_file)
        performance.append(data.flatten())

    if not performance:
        return []

    allperf = np.array(performance)
    return allperf.flatten()  # np.median(allperf, axis=1)


def compute_performance_stats(
    performance_data, K, method="subsample_bootstrap", n_bootstrap=1000, subsample_size=10000
):
    """
    Compute median performance and 95% CI using various methods.

    Parameters:
    -----------
    performance_data : array
        Performance measurements
    method : str
        'direct': Direct percentiles (fastest for large N)
        'subsample_bootstrap': Bootstrap on random subsample
        'bootstrap': Full bootstrap (slow for large N)
    n_bootstrap : int
        Number of bootstrap iterations (for bootstrap methods)
    subsample_size : int
        Size of subsample for subsample_bootstrap method
    """
    if len(performance_data) == 0:
        return np.nan, np.nan, np.nan

    # Median performance
    median_perf = np.median(performance_data)

    if method == "direct":
        # Direct percentile method - fastest for large samples
        # With 56M samples, empirical distribution is very stable
        ci_lower = np.percentile(performance_data, 2.5)
        ci_upper = np.percentile(performance_data, 97.5)

    elif method == "subsample_bootstrap":
        # Bootstrap on a random subsample - much faster
        if len(performance_data) > subsample_size:
            subsample = np.random.choice(performance_data, size=subsample_size, replace=False)
        else:
            subsample = performance_data

        bootstrap_medians = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(subsample, size=len(subsample), replace=True)
            bootstrap_medians.append(np.median(sample))

        ci_lower = np.percentile(bootstrap_medians, 2.5)
        ci_upper = np.percentile(bootstrap_medians, 97.5)

    elif method == "bootstrap":
        # Full bootstrap (original method, slow for large N)
        bootstrap_medians = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(performance_data, size=len(performance_data), replace=True)
            bootstrap_medians.append(np.median(sample))

        ci_lower = np.percentile(bootstrap_medians, 2.5)
        ci_upper = np.percentile(bootstrap_medians, 97.5)

    else:
        raise ValueError(f"Unknown method: {method}")

    def to_us(cycles):
        seconds = (cycles / 0.85 * 1e-3) / 1000 / 1000
        bytes_per_sec = 2 * K * 4 / 1024 / 1024 / 1024 / seconds
        return bytes_per_sec

    return to_us(median_perf), to_us(ci_upper), to_us(ci_lower)


def parse_folder_name(folder_name):
    """Extract N and K values from folder name like 'N_100_K_50'."""
    match = re.search(r"N_(\d+)_K_(\d+)", folder_name)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def create_performance_heatmap(
    experiment_path, output_file="performance_heatmap.png", metric_name="Performance", vmin=None, vmax=None
):
    """
    Create a heatmap of performance metrics from experiment folder.

    Parameters:
    -----------
    experiment_path : str
        Path to experiment directory containing N_*_K_* folders
    output_file : str
        Output filename for the heatmap
    metric_name : str
        Name of the metric being plotted (for labels)
    vmin, vmax : float, optional
        Min and max values for colorbar
    """
    exp_path = Path(experiment_path)

    # Collect all N_*_K_* folders
    folders = [f for f in exp_path.glob("N_*_K_*") if f.is_dir()]

    if len(folders) == 0:
        print(f"No N_*_K_* folders found in {experiment_path}")
        return None, None, None

    # Parse N and K values and collect data
    data_dict = {}

    for folder in folders:
        n_val, k_val = parse_folder_name(folder.name)
        if n_val is None or k_val is None:
            continue
        # if n_val > 128:
        #     continue

        # Load performance data
        perf_data = load_performance_data(folder)

        if len(perf_data) == 0:
            print(f"Warning: No data in {folder.name}")
            continue

        # Compute performance statistics
        median, ci_lower, ci_upper = compute_performance_stats(perf_data, k_val)

        data_dict[(n_val, k_val)] = {
            "median": median,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "ci_width": ci_upper - ci_lower,
        }

        print(f"N={n_val}, K={k_val}: {metric_name}={median:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")

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

    # Auto-scale if not provided
    if vmin is None:
        vmin = np.nanmin(median_matrix)
    if vmax is None:
        vmax = np.nanmax(median_matrix)

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 1, figsize=(12, 6))

    # Plot 1: Median performance heatmap
    im1 = axes.imshow(median_matrix, aspect="auto", cmap="Blues_r", vmin=vmin, vmax=vmax)  # cmap='viridis',
    axes.set_xticks(range(len(n_values)))
    axes.set_yticks(range(len(k_values)))
    axes.set_xticklabels(n_values, fontsize=16)
    axes.set_yticklabels(k_values, fontsize=16)
    axes.set_xlabel("#PEs", fontsize=24)
    axes.set_ylabel("Elements", fontsize=24)

    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=axes)
    cbar1.set_label(metric_name, fontsize=24)
    cbar1.ax.tick_params(labelsize=16)

    # Add text annotations
    for i in range(len(k_values)):
        for j in range(len(n_values)):
            if not np.isnan(median_matrix[i, j]):
                # Choose text color for contrast
                normalized_val = median_matrix[i, j]
                if ci_width_matrix[i, j] / 2 >= 0.01:
                    ci = f"\n±{(ci_width_matrix[i,j])/2:.2f}"
                else:
                    ci = ""
                text_color = "white" if normalized_val < 1 else "black"
                axes.text(
                    j,
                    i,
                    f"{median_matrix[i, j]:.3f}{ci}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=12,
                    fontweight="bold",
                )

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.savefig(output_file + ".pdf", bbox_inches="tight")
    print(f"\nHeatmap saved to {output_file}")

    return data_dict, median_matrix, ci_width_matrix, n_values, k_values


# Example usage:
if __name__ == "__main__":
    experiment_path = "./benchmark_results_sweep/copy"

    # Create heatmap
    results, median_mat, ci_mat, n_vals, k_vals = create_performance_heatmap(
        experiment_path,
        metric_name="Throughput [GiB/s / PE]",  # Customize this
    )
