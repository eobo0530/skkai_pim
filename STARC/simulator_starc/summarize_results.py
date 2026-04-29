#!/usr/bin/env python3
"""
Task 1 실측 결과 수집 (1K ~ 32K) 요약 CSV 저장 스크립트.
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
    print("  STARC Remapping Wall — 실측 벤치마크 결과 (1K ~ 32K)")
    print("=" * 110)
    print(f"{'L':>8} | {'KV Budget':>10} | {'T_dense':>8} | {'T_comp':>8} | {'T_cluster':>10} | {'T_copy':>8} | {'T_STARC':>8} | {'Speedup':>8} | {'Overhead':>8}")
    print("-" * 110)
    
    for L in MEASURED_LENGTHS:
        t_d = parse_csv(os.path.join(RESULTS_DIR, f"dense_L{L}.csv"))
        t_s = parse_csv(os.path.join(RESULTS_DIR, f"sparse_L{L}.csv"))
        t_c = parse_csv(os.path.join(RESULTS_DIR, f"sparse_cluster_L{L}.csv"))
        t_f = parse_csv(os.path.join(RESULTS_DIR, f"sparse_full_L{L}.csv"))
        
        if any(v is None for v in [t_d, t_s, t_c, t_f]):
            print(f"  WARNING: L={L} 데이터 누락")
            continue
        
        kv_budget = compute_kv_budget(L)
        t_clustering = t_c - t_s      # 클러스터링 연산만
        t_copy = t_f - t_c            # 물리적 데이터 이동만
        t_comp = t_s                  # sparse GEMV 연산
        t_starc = t_f                 # 총 STARC 레이턴시
        speedup = t_d / t_starc if t_starc > 0 else float('inf')
        overhead_pct = (t_clustering + t_copy) / t_comp * 100 if t_comp > 0 else 0
        
        row = {
            "context_length": L,
            "kv_budget": kv_budget,
            "T_dense": t_d,
            "T_computation": t_comp,
            "T_clustering": t_clustering,
            "T_copy": t_copy,
            "T_starc_total": t_starc,
            "speedup": speedup,
            "overhead_pct": overhead_pct
        }
        results.append(row)
        
        marker = " ← INVERSION" if speedup < 1.0 else ""
        print(
            f"{L:>8} | {kv_budget:>10} | "
            f"{t_d:>7.3f}ms | {t_comp:>7.3f}ms | "
            f"{t_clustering:>9.3f}ms | {t_copy:>7.3f}ms | "
            f"{t_starc:>7.3f}ms | {speedup:>7.3f}x | "
            f"{overhead_pct:>6.1f}%{marker}"
        )
    print("=" * 110)
    
    # CSV 저장
    summary_file = os.path.join(RESULTS_DIR, "task1_summary_32k.csv")
    if results:
        fieldnames = list(results[0].keys())
        with open(summary_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n→ CSV 저장: {summary_file}")

if __name__ == "__main__":
    main()
