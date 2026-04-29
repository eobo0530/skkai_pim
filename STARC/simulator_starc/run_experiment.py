#!/usr/bin/env python3
import subprocess
import os
import sys
import argparse

SIMULATOR_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SIMULATOR_DIR, "results")
MAIN_PY = os.path.join(SIMULATOR_DIR, "main.py")

# RPC parameters
P = 1024
C = 4
T = P // C  # 256
R = 32

def compute_kv_budget(L):
    return min((L // P + 1) * T + R, L)

def run_simulation(lin, label, sparsity, cluster_mode, kv_budget=0):
    output_file = os.path.join(RESULTS_DIR, f"{label}_L{lin}.csv")
    for f in ["ramulator.out", "output.csv"]:
        if os.path.exists(f):
            os.remove(f)

    cmd = (
        f"{sys.executable} {MAIN_PY} --system dgx-attacc --gpu H100 --ngpu 8 "
        f"--pim bank --powerlimit --ffopt --pipeopt --batch 1 "
        f"--model DeepSeek-R1-7B --lin {lin} --lout 2"
    )

    if sparsity:
        cmd += f" --sparsity --kv_budget {kv_budget}"
    if cluster_mode != "none":
        cmd += f" --cluster_mode {cluster_mode}"

    print(f"[{label}] L={lin} (budget={kv_budget}, cluster={cluster_mode})")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"FAILED: {cmd}")
        print(f"Stderr: {result.stderr}")
        return False
        
    if os.path.exists("output.csv"):
        os.rename("output.csv", output_file)
        return True
    return False

def main():
    parser = argparse.ArgumentParser(description="Unified Simulation Runner")
    parser.add_argument("--seqlen", nargs="+", type=int, required=True, help="List of context lengths to simulate (e.g., --seqlen 64 128 256)")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    RUN_MODES = [
        ("dense",          False, "none",         False),
        ("sparse",         True,  "none",         True),
        ("sparse_cluster", True,  "cluster_only", True),
        ("sparse_full",    True,  "full",         True),
    ]

    for L in args.seqlen:
        print(f"\n========== Running Context Length L={L} ==========")
        kv_budget = compute_kv_budget(L)
        for label, sparsity, cluster_mode, uses_budget in RUN_MODES:
            budget = kv_budget if uses_budget else 0
            run_simulation(L, label, sparsity, cluster_mode, budget)
            
if __name__ == "__main__":
    main()
