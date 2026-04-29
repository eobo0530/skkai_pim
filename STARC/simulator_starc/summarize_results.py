#!/usr/bin/env python3
"""
Task 1 결과 수집 + 64K/128K 외삽 → 요약 CSV 저장.
"""
import csv
import os
import math

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# RPC parameters
P = 1024
C = 4
T = P // C  # 256
R = 32

def compute_kv_budget(L):
    return min((L // P + 1) * T + R, L)

def parse_csv(filepath):
    if not os.path.exists(filepath):
        return None
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            return float(row["g_time (ms)"])
    return None

def main():
    MEASURED_LENGTHS = [256, 512, 1024, 4096, 16384, 32768]
    results = []
    
    print("=" * 110)
    print("  STARC Remapping Wall — 결과 수집 (실측 및 외삽)")
    print("=" * 110)
    
    for L in MEASURED_LENGTHS:
        t_d = parse_csv(os.path.join(RESULTS_DIR, f"dense_L{L}.csv"))
        t_s = parse_csv(os.path.join(RESULTS_DIR, f"sparse_L{L}.csv"))
        t_c = parse_csv(os.path.join(RESULTS_DIR, f"sparse_cluster_L{L}.csv"))
        t_f = parse_csv(os.path.join(RESULTS_DIR, f"sparse_full_L{L}.csv"))
        
        if any(v is None for v in [t_d, t_s, t_c, t_f]):
            print(f"  WARNING: L={L} 데이터 누락")
            continue
        
        kv_budget = compute_kv_budget(L)
        t_clustering = t_c - t_s
        t_copy = t_f - t_c
        t_comp = t_s
        t_starc = t_f
        speedup = t_d / t_starc if t_starc > 0 else float('inf')
        overhead_pct = (t_clustering + t_copy) / t_comp * 100 if t_comp > 0 else 0
        
        results.append({
            "context_length": L,
            "kv_budget": kv_budget,
            "T_dense": t_d,
            "T_computation": t_comp,
            "T_clustering": t_clustering,
            "T_copy": t_copy,
            "T_starc_total": t_starc,
            "speedup": speedup,
            "overhead_pct": overhead_pct,
            "source": "measured"
        })
        
    # Extrapolation for 64K and 128K based on 32K data
    r32 = next((r for r in results if r["context_length"] == 32768), None)
    if r32:
        for scale, L in [(2, 65536), (4, 131072)]:
            kv_budget = compute_kv_budget(L)
            t_d = r32["T_dense"] * scale
            t_comp = r32["T_computation"] * (kv_budget / r32["kv_budget"])
            t_clustering = r32["T_clustering"] * (scale * scale)
            t_copy = r32["T_copy"] * scale
            t_starc = t_comp + t_clustering + t_copy
            speedup = t_d / t_starc if t_starc > 0 else float('inf')
            overhead_pct = (t_clustering + t_copy) / t_comp * 100 if t_comp > 0 else 0
            
            results.append({
                "context_length": L,
                "kv_budget": kv_budget,
                "T_dense": t_d,
                "T_computation": t_comp,
                "T_clustering": t_clustering,
                "T_copy": t_copy,
                "T_starc_total": t_starc,
                "speedup": speedup,
                "overhead_pct": overhead_pct,
                "source": "extrapolated"
            })
            
    print(f"{'L':>8} | {'KV Budget':>10} | {'T_dense':>8} | {'T_comp':>8} | {'T_cluster':>10} | {'T_copy':>8} | {'T_STARC':>8} | {'Speedup':>8} | {'Overhead':>8} | {'Source':>12}")
    print("-" * 110)
    for r in results:
        marker = " ← INVERSION" if r["speedup"] < 1.0 else ""
        print(
            f"{r['context_length']:>8} | {r['kv_budget']:>10} | "
            f"{r['T_dense']:>7.3f}ms | {r['T_computation']:>7.3f}ms | "
            f"{r['T_clustering']:>9.3f}ms | {r['T_copy']:>7.3f}ms | "
            f"{r['T_starc_total']:>7.3f}ms | {r['speedup']:>7.3f}x | "
            f"{r['overhead_pct']:>6.1f}% | {r['source']:>12}{marker}"
        )
    print("=" * 110)
    
    summary_file = os.path.join(RESULTS_DIR, "task1_summary.csv")
    if results:
        fieldnames = list(results[0].keys())
        with open(summary_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n→ CSV 저장: {summary_file}")

if __name__ == "__main__":
    main()
