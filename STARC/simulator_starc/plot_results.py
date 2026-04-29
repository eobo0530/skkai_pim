#!/usr/bin/env python3
"""
Visualization script for STARC Remapping Wall experiment results.

Reads CSV outputs from run_experiment.py and generates:
  1. Stacked bar chart: T_comp, T_cluster, T_copy breakdown
  2. Line chart: Dense vs STARC with inversion point
  3. Time-series: Pruning spike detection
"""

import csv
import os
import sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np
except ImportError:
    print("ERROR: matplotlib and numpy are required. Install with: pip3 install matplotlib numpy")
    sys.exit(1)

RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")


def load_csv(filename):
    filepath = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(filepath):
        print(f"WARNING: {filepath} not found, skipping.")
        return []
    with open(filepath, "r") as f:
        return list(csv.DictReader(f))


def plot_task1_stacked_bar(data):
    """Stacked bar chart: T_comp, T_cluster, T_copy vs context length."""
    if not data:
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    labels = [f"{int(r['context_length'])//1024}K" for r in data]
    t_comp = [float(r.get("T_computation", 0)) for r in data]
    t_cluster = [float(r.get("T_clustering", 0)) for r in data]
    t_copy = [float(r.get("T_copy", 0)) for r in data]

    x = np.arange(len(labels))
    width = 0.5

    bars1 = ax.bar(x, t_comp, width, label=r"$T_{computation}$", color="#3498db", edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x, t_cluster, width, bottom=t_comp, label=r"$T_{clustering}$", color="#e67e22", edgecolor="white", linewidth=0.5)
    bars3 = ax.bar(x, t_copy, width, bottom=[a + b for a, b in zip(t_comp, t_cluster)],
                   label=r"$T_{physical\_copy}$", color="#e74c3c", edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Context Length", fontsize=14, fontweight="bold")
    ax.set_ylabel("Per-Token Latency (ms)", fontsize=14, fontweight="bold")
    ax.set_title("STARC Latency Breakdown: The Remapping Wall\n(DeepSeek-R1-7B, Batch=1, P=1024)",
                 fontsize=16, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.legend(fontsize=12, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    # Add percentage annotations on top
    for i in range(len(data)):
        total = t_comp[i] + t_cluster[i] + t_copy[i]
        if total > 0:
            pct = (t_cluster[i] + t_copy[i]) / total * 100
            ax.text(x[i], total * 1.02, f"{pct:.0f}%\noverhead",
                    ha="center", va="bottom", fontsize=9, color="#555")

    plt.tight_layout()
    outpath = os.path.join(PLOTS_DIR, "task1_latency_breakdown.png")
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"  [Plot 1] Saved: {outpath}")


def plot_task2_inversion(data):
    """Line chart: Dense vs STARC latency with inversion point."""
    if not data:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    contexts = [int(r["context_length"]) for r in data]
    ctx_labels = [f"{c//1024}K" for c in contexts]
    t_dense = [float(r.get("T_dense", 0)) for r in data]
    t_starc = [float(r.get("T_starc_total", 0)) for r in data]
    speedup = [float(r.get("speedup", 0)) for r in data]

    # Left: Latency comparison
    ax1.plot(range(len(contexts)), t_dense, "o-", color="#2ecc71", linewidth=2.5,
             markersize=8, label="Dense (Full Attention)", zorder=3)
    ax1.plot(range(len(contexts)), t_starc, "s-", color="#e74c3c", linewidth=2.5,
             markersize=8, label="Sparse + STARC (RPC)", zorder=3)

    # Find inversion point
    inversion_idx = None
    for i in range(len(speedup)):
        if speedup[i] < 1.0:
            inversion_idx = i
            break

    if inversion_idx is not None:
        ax1.axvline(x=inversion_idx, color="#8e44ad", linestyle="--", linewidth=2, alpha=0.7)
        ax1.annotate(f"Inversion\n@ {ctx_labels[inversion_idx]}",
                     xy=(inversion_idx, t_starc[inversion_idx]),
                     xytext=(inversion_idx + 0.5, t_starc[inversion_idx] * 1.3),
                     fontsize=11, fontweight="bold", color="#8e44ad",
                     arrowprops=dict(arrowstyle="->", color="#8e44ad", linewidth=2))

    ax1.set_xlabel("Context Length", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Per-Token Latency (ms)", fontsize=13, fontweight="bold")
    ax1.set_title("Dense vs STARC+RPC Latency", fontsize=14, fontweight="bold")
    ax1.set_xticks(range(len(ctx_labels)))
    ax1.set_xticklabels(ctx_labels, fontsize=11)
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)

    # Right: Speedup ratio
    colors = ["#2ecc71" if s >= 1.0 else "#e74c3c" for s in speedup]
    bars = ax2.bar(range(len(contexts)), speedup, color=colors, edgecolor="white", linewidth=0.5)
    ax2.axhline(y=1.0, color="#555", linestyle="--", linewidth=1.5, alpha=0.7)
    ax2.set_xlabel("Context Length", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Speedup (Dense / STARC)", fontsize=13, fontweight="bold")
    ax2.set_title("Speedup Ratio (>1 = STARC wins)", fontsize=14, fontweight="bold")
    ax2.set_xticks(range(len(ctx_labels)))
    ax2.set_xticklabels(ctx_labels, fontsize=11)
    ax2.grid(axis="y", alpha=0.3)

    for i, (bar, sp) in enumerate(zip(bars, speedup)):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{sp:.2f}x", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.suptitle("Task 2: Finding the Inversion Point\n(DeepSeek-R1-7B, Batch=1, P=1024)",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    outpath = os.path.join(PLOTS_DIR, "task2_inversion_point.png")
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [Plot 2] Saved: {outpath}")


def plot_task3_spikes(data):
    """Time-series plot: latency spikes at pruning boundaries."""
    if not data:
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    # Group by boundary
    boundaries = sorted(set(int(r["boundary"]) for r in data))

    # Plot all points
    contexts = [int(r["context_length"]) for r in data]
    full_times = [float(r["full_time"]) for r in data]
    sparse_times = [float(r["sparse_time"]) for r in data]
    spike_ratios = [float(r["spike_ratio"]) for r in data]
    offsets = [int(r["offset"]) for r in data]

    # Color by spike magnitude
    colors = []
    for sr in spike_ratios:
        if sr > 2.0:
            colors.append("#e74c3c")  # red: severe spike
        elif sr > 1.5:
            colors.append("#e67e22")  # orange: moderate spike
        else:
            colors.append("#3498db")  # blue: normal

    ax.scatter(contexts, full_times, c=colors, s=80, zorder=3, edgecolors="white", linewidth=0.5)
    ax.plot(contexts, sparse_times, "--", color="#95a5a6", linewidth=1, alpha=0.7, label="Sparse (no overhead)")

    # Annotate boundaries
    for boundary in boundaries:
        ax.axvline(x=boundary, color="#e74c3c", linestyle=":", linewidth=1, alpha=0.4)

    ax.set_xlabel("Context Length", fontsize=13, fontweight="bold")
    ax.set_ylabel("Per-Token Latency (ms)", fontsize=13, fontweight="bold")
    ax.set_title("Task 3: Dynamic Pruning Overhead Spikes\n(DeepSeek-R1-7B, P=1024, cluster_mode=full)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    # Custom legend for spike colors
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c", markersize=10, label="Spike > 2x"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e67e22", markersize=10, label="Spike 1.5-2x"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#3498db", markersize=10, label="Normal"),
        Line2D([0], [0], linestyle="--", color="#95a5a6", label="Sparse baseline"),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc="upper left")

    plt.tight_layout()
    outpath = os.path.join(PLOTS_DIR, "task3_pruning_spikes.png")
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"  [Plot 3] Saved: {outpath}")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("\n[Plotting] Loading experiment results...")

    task1_data = load_csv("task1_summary.csv")

    print("[Plotting] Generating charts...")
    plot_task1_stacked_bar(task1_data)
    plot_task2_inversion(task1_data)
    
    # Task 3 was skipped due to WSL memory limits
    # task3_data = load_csv("task3_summary.csv")
    # plot_task3_spikes(task3_data)

    print(f"\n[DONE] All plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
