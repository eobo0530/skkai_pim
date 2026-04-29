#!/usr/bin/env python3
"""
Experiment automation for STARC Remapping Wall verification.

Runs 3 tasks:
  Task 1: Scalability Stress Test — 4 runs × 6 context lengths
  Task 2: Inversion Point (reuses Task 1 data)
  Task 3: Dynamic Pruning Overhead — spike measurement around pruning boundaries

Uses snapshot simulation: --lin L --lout 2 → 1 Ramulator call per config.
"""

import subprocess
import os
import csv
import math
import sys

# === Configuration ===
MODEL = "DeepSeek-R1-7B"
SYSTEM_ARGS = (
    "--system dgx-attacc --gpu H100 --ngpu 8 "
    "--pim bank --powerlimit --ffopt --pipeopt --batch 1"
)

# RPC parameters
P = 1024       # Pruning period (tokens)
C = 4          # Compression ratio
T = P // C     # Tokens kept per compression = 256
R = 32         # Recent window

# Context lengths for Task 1 & 2
CONTEXT_LENGTHS = [1024, 4096, 16384, 32768, 65536, 131072]

# 4 run modes: (label, sparsity?, cluster_mode, uses kv_budget?)
RUN_MODES = [
    ("dense",          False, "none",         False),
    ("sparse",         True,  "none",         True),
    ("sparse_cluster", True,  "cluster_only", True),
    ("sparse_full",    True,  "full",         True),
]

RESULTS_DIR = "results"
MAIN_PY = os.path.join(os.path.dirname(__file__), "main.py")


def compute_kv_budget(context_len):
    """Compute KV budget at a given context length under RPC pruning schedule.

    RPC compresses every P tokens, keeping T tokens per compression.
    At context length L:
      num_compressions = floor(L / P)
      kv_budget = (num_compressions + 1) * T + R
    """
    num_compressions = context_len // P
    budget = (num_compressions + 1) * T + R
    # Budget cannot exceed context length
    return min(budget, context_len)


def run_simulation(lin, label, sparsity, cluster_mode, kv_budget=0):
    """Run a single snapshot simulation and return the output.csv path."""
    output_file = os.path.join(RESULTS_DIR, f"{label}_L{lin}.csv")

    # Clean stale cache
    for f in ["ramulator.out", "output.csv"]:
        if os.path.exists(f):
            os.remove(f)

    cmd = (
        f"{sys.executable} {MAIN_PY} {SYSTEM_ARGS} "
        f"--model {MODEL} --lin {lin} --lout 2"
    )

    if sparsity:
        cmd += f" --sparsity --kv_budget {kv_budget}"

    if cluster_mode != "none":
        cmd += f" --cluster_mode {cluster_mode}"

    print(f"\n{'='*60}")
    print(f"  [{label}] L={lin}, kv_budget={kv_budget}, cluster_mode={cluster_mode}")
    print(f"  CMD: {cmd}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, shell=True, capture_output=False,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, timeout=3600)
    if result.returncode != 0:
        print(f"  ERROR (rc={result.returncode}): {result.stdout}")
        return None

    print(result.stdout)

    # Move output.csv to results
    if os.path.exists("output.csv"):
        os.rename("output.csv", output_file)
        return output_file
    return None


def parse_output_csv(filepath):
    """Parse the output.csv and extract key latency metrics."""
    if not filepath or not os.path.exists(filepath):
        return None
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            return {
                "g_time_ms": float(row["g_time (ms)"]),
                "g_matmul": float(row["g_matmul"]),
                "g_fc": float(row["g_fc"]),
                "g_comm": float(row["g_comm"]),
                "g_etc": float(row["g_etc"]),
            }
    return None


def run_task1():
    """Task 1: Scalability Stress Test — measure T_comp, T_cluster, T_copy at each context length."""
    print("\n" + "=" * 70)
    print("  TASK 1: Scalability Stress Test")
    print("=" * 70)

    results = []

    for L in CONTEXT_LENGTHS:
        kv_budget = compute_kv_budget(L)

        row = {"context_length": L, "kv_budget": kv_budget}

        for label, sparsity, cluster_mode, uses_budget in RUN_MODES:
            budget = kv_budget if uses_budget else 0
            output_file = run_simulation(L, label, sparsity, cluster_mode, budget)
            metrics = parse_output_csv(output_file)

            if metrics:
                row[f"{label}_g_time"] = metrics["g_time_ms"]
                row[f"{label}_g_matmul"] = metrics["g_matmul"]

        # Compute separated times
        if "sparse_g_time" in row and "dense_g_time" in row:
            row["T_computation"] = row.get("sparse_g_time", 0)
            row["T_clustering"] = row.get("sparse_cluster_g_time", 0) - row.get("sparse_g_time", 0)
            row["T_copy"] = row.get("sparse_full_g_time", 0) - row.get("sparse_cluster_g_time", 0)
            row["T_dense"] = row.get("dense_g_time", 0)
            row["T_starc_total"] = row.get("sparse_full_g_time", 0)
            row["overhead_ratio"] = (row["T_clustering"] + row["T_copy"]) / max(row["T_computation"], 1e-9)
            row["speedup"] = row["T_dense"] / max(row["T_starc_total"], 1e-9)

        results.append(row)

    # Write summary CSV
    summary_file = os.path.join(RESULTS_DIR, "task1_summary.csv")
    if results:
        with open(summary_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\n[Task 1] Summary written to {summary_file}")

    return results


def run_task3():
    """Task 3: Dynamic Pruning Overhead — measure latency spikes around pruning boundaries."""
    print("\n" + "=" * 70)
    print("  TASK 3: Dynamic Pruning Overhead Spike Measurement")
    print("=" * 70)

    # Sample pruning boundaries at various scales
    boundaries = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
    offsets = [-2, -1, 0, 1, 2]

    results = []

    for boundary in boundaries:
        for offset in offsets:
            L = boundary + offset
            if L < 1:
                continue

            kv_budget = compute_kv_budget(L)

            # Run sparse + full (STARC with all overhead)
            output_file = run_simulation(
                L, f"spike_full", True, "full", kv_budget
            )
            metrics = parse_output_csv(output_file)

            # Run sparse (no overhead) for comparison
            output_file_base = run_simulation(
                L, f"spike_sparse", True, "none", kv_budget
            )
            metrics_base = parse_output_csv(output_file_base)

            if metrics and metrics_base:
                row = {
                    "boundary": boundary,
                    "offset": offset,
                    "context_length": L,
                    "kv_budget": kv_budget,
                    "sparse_time": metrics_base["g_time_ms"],
                    "full_time": metrics["g_time_ms"],
                    "overhead": metrics["g_time_ms"] - metrics_base["g_time_ms"],
                    "spike_ratio": metrics["g_time_ms"] / max(metrics_base["g_time_ms"], 1e-9),
                }
                results.append(row)

    # Write summary CSV
    summary_file = os.path.join(RESULTS_DIR, "task3_summary.csv")
    if results:
        with open(summary_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\n[Task 3] Summary written to {summary_file}")

    return results


def print_task1_summary(results):
    """Print a formatted summary of Task 1 results."""
    print("\n" + "=" * 90)
    print("  TASK 1 & 2: Summary — Latency Breakdown & Inversion Point")
    print("=" * 90)
    print(f"{'L':>8} | {'T_dense':>10} | {'T_comp':>10} | {'T_cluster':>10} | {'T_copy':>10} | {'T_STARC':>10} | {'Speedup':>8} | {'Overhead%':>10}")
    print("-" * 90)
    for r in results:
        L = r.get("context_length", 0)
        t_d = r.get("T_dense", 0)
        t_c = r.get("T_computation", 0)
        t_cl = r.get("T_clustering", 0)
        t_cp = r.get("T_copy", 0)
        t_s = r.get("T_starc_total", 0)
        sp = r.get("speedup", 0)
        oh = r.get("overhead_ratio", 0)

        marker = " ← INVERSION" if sp < 1.0 else ""
        print(f"{L:>8} | {t_d:>10.4f} | {t_c:>10.4f} | {t_cl:>10.4f} | {t_cp:>10.4f} | {t_s:>10.4f} | {sp:>8.3f} | {oh*100:>9.1f}%{marker}")
    print("=" * 90)


def print_task3_summary(results):
    """Print a formatted summary of Task 3 results."""
    print("\n" + "=" * 80)
    print("  TASK 3: Dynamic Pruning Overhead Spikes")
    print("=" * 80)
    print(f"{'Boundary':>10} | {'Offset':>6} | {'L':>8} | {'Sparse':>10} | {'Full':>10} | {'Overhead':>10} | {'Spike':>6}")
    print("-" * 80)
    for r in results:
        spike_marker = " ← SPIKE" if r["spike_ratio"] > 1.5 else ""
        print(
            f"{r['boundary']:>10} | {r['offset']:>6} | {r['context_length']:>8} | "
            f"{r['sparse_time']:>10.4f} | {r['full_time']:>10.4f} | "
            f"{r['overhead']:>10.4f} | {r['spike_ratio']:>5.2f}x{spike_marker}"
        )
    print("=" * 80)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Task 1 & 2: Scalability + Inversion Point
    task1_results = run_task1()
    print_task1_summary(task1_results)

    # Task 3: Pruning Spike
    task3_results = run_task3()
    print_task3_summary(task3_results)

    print("\n[DONE] All experiments complete. Results in:", RESULTS_DIR)


if __name__ == "__main__":
    main()
