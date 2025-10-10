import numpy as np
import pandas as pd
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# Load flops
flop_df = pd.read_csv("flops.csv")[["Program", "Flop"]]


def cycles_to_us(cycles):
    return cycles / 0.85 * 1e-3


def compute_performance_stats(performance_data, method="subsample_bootstrap", n_bootstrap=1000, subsample_size=10000):
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

    return median_perf, ci_lower, ci_upper


def load_runtime_data(folder_path):
    """Load all numpy files from a folder and return combined runtime data."""
    folder = Path(folder_path)
    runtimes = []

    for npy_file in folder.glob("perf*.npy"):
        data = np.load(npy_file)

        runtimes.append(compute_performance_stats(data.flat))

    if not runtimes:
        return None

    return np.array(runtimes)


base_path = "benchmark_results_stencilsweep"
programs = []
domain_sizes = []
times = []
times_ci_lo = []
times_ci_hi = []
flops = []


def sort_key(k):
    # Extract the name part
    name = " ".join(k.split("_")[:-3])
    # Extract the number part
    number = int(k.split("_")[-2])
    return (name, number)


if not os.path.exists("stensweep_k.pkl"):

    for program in sorted(os.listdir(base_path), key=sort_key):
        print("Processing", program)
        program_path = os.path.join(base_path, program)
        flop_count = flop_df[flop_df.Program == program].Flop.item()
        domain_size = program.split("_")[-3:]
        domain_size_label = "x".join(domain_size)
        # if domain_size[-1] != "80":  # Horizontal sweep
        #     continue
        # if int(domain_size[0]) > 128:
        #     continue  # DEBUG
        if domain_size[0] != "512":  # Vertical sweep
            continue
        program_label = " ".join(program.split("_")[:-3])
        if program_label not in ("pure vertical", "laplacian", "uvbke"):
            continue
        if program_label == "vertical advection":
            program_label = "Vertical Advection"
        if program_label == "pure vertical":
            program_label = "Vertical Stencil"
        if program_label == "laplacian":
            program_label = "2D Laplacian"
        if program_label == "uvbke":
            program_label = "UVBKE"
        print("  Processing", program_label, "x".join(domain_size))

        runtimes = load_runtime_data(program_path)

        if runtimes is not None:
            programs.extend([program_label] * len(runtimes))
            times.extend(cycles_to_us(runtimes[:, 0]))
            times_ci_lo.extend(cycles_to_us(runtimes[:, 1]))
            times_ci_hi.extend(cycles_to_us(runtimes[:, 2]))
            flops.extend([flop_count] * len(runtimes))
            domain_sizes.extend([domain_size_label] * len(runtimes))

    print("Creating dataframe...")
    df = pd.DataFrame(
        {
            "Program": programs,
            "time_us": times,
            "time_ci_low": times_ci_lo,
            "time_ci_hi": times_ci_hi,
            "domain_size": domain_sizes,
            "flops": flops,
        }
    )

    df["time_s"] = df.time_us / 1000 / 1000
    df["flops_per_second"] = df.flops / df.time_s
    df["tflops"] = df.flops_per_second / 1000 / 1000 / 1000 / 1000

    import pickle

    with open("stensweep_k.pkl", "wb") as fp:
        pickle.dump(df, fp)
else:
    import pickle

    with open("stensweep_k.pkl", "rb") as fp:
        df = pickle.load(fp)


print("Computing statistics...")
df["domain_label"] = df["domain_size"].str.removeprefix("512x512x")
# Sort by domain size as integers
df['domain_size'] = df['domain_label'].str.extract('(\d+)').astype(int)
df = df.sort_values('domain_size')

# Set seaborn style
sns.set_style("whitegrid")
sns.set_palette("husl")

# Create figure with two subplots
fig, (ax2) = plt.subplots(1, 1, figsize=(6, 4))


marker_dict = {
    '2D Laplacian': 'p',
    'UVBKE': 's', 
    'Vertical Stencil': 'o',
}
# Plot 2: FLOP/s vs Domain Size
sns.lineplot(data=df, x="domain_size", y="tflops", hue="Program", style="Program", markers=marker_dict, markersize=8, linewidth=2.5, ax=ax2)
ax2.set_xlabel("Pressure Levels (K)", fontsize=12, fontweight="bold")
ax2.set_ylabel("Performance [TFlop/s]", fontsize=12, fontweight="bold")
ax2.legend(title="Program", title_fontsize=11, fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("stencil_scaling_vertical.png", dpi=300, bbox_inches="tight")
plt.savefig("stencil_scaling_vertical.pdf", bbox_inches="tight")
