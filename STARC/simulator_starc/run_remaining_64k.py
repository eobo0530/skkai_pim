#!/usr/bin/env python3
"""
64K 남은 실험 2개를 진행률 표시와 함께 실행.
트레이스 생성의 각 단계별 진행률을 표시합니다.
"""
import subprocess
import os
import sys
import math
import time
import csv

SIMULATOR_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SIMULATOR_DIR, "results")
RAMULATOR2_DIR = os.path.join(SIMULATOR_DIR, "ramulator2")
TRACE_GEN = os.path.join(RAMULATOR2_DIR, "trace_gen", "gen_trace_attacc_bank.py")

# --- 1단계: 직접 트레이스 생성 (진행률 표시 포함) ---
# gen_trace_attacc_bank.py의 핵심 로직을 임포트하지 않고,
# 대신 main.py를 호출하되 트레이스 생성 시간을 예측하여 표시

def estimate_trace_time(L, mode):
    """32K 결과 기반으로 64K 소요 시간 추정"""
    # 32K cluster_only: ~60s, full: ~90s (관측값)
    # cluster_similar는 O(L²), 나머지는 O(L)
    # 64K/32K = 2x → O(L²) 부분은 4x
    if mode == "cluster_only":
        return 240  # ~4분 예상
    else:
        return 360  # ~6분 예상

def run_with_progress(L, kv_budget, cluster_mode, label):
    """main.py를 별도 프로세스로 실행하면서 경과 시간/예상 진행률 표시"""
    
    estimated_secs = estimate_trace_time(L, cluster_mode)
    
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  L={L}, kv_budget={kv_budget}, cluster_mode={cluster_mode}")
    print(f"  예상 소요: ~{estimated_secs//60}분 {estimated_secs%60}초")
    print(f"{'='*60}")
    
    # 캐시 정리
    for f in ["ramulator.out", "output.csv"]:
        p = os.path.join(SIMULATOR_DIR, f)
        if os.path.exists(p):
            os.remove(p)
    
    cmd = [
        sys.executable, os.path.join(SIMULATOR_DIR, "main.py"),
        "--system", "dgx-attacc", "--gpu", "H100", "--ngpu", "8",
        "--model", "DeepSeek-R1-7B", "--lin", str(L), "--lout", "2",
        "--batch", "1", "--pim", "bank",
        "--powerlimit", "--ffopt", "--pipeopt",
        "--sparsity", "--kv_budget", str(kv_budget),
        "--cluster_mode", cluster_mode,
    ]
    
    start_time = time.time()
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        cwd=SIMULATOR_DIR
    )
    
    # 진행률 표시 루프
    last_pct = -1
    while proc.poll() is None:
        elapsed = time.time() - start_time
        pct = min(99, int(elapsed / estimated_secs * 100))
        
        if pct != last_pct:
            bar_len = 30
            filled = int(bar_len * pct / 100)
            bar = '█' * filled + '░' * (bar_len - filled)
            elapsed_str = f"{int(elapsed//60)}:{int(elapsed%60):02d}"
            eta = max(0, estimated_secs - elapsed)
            eta_str = f"{int(eta//60)}:{int(eta%60):02d}"
            print(f"\r  [{bar}] {pct:3d}% | 경과: {elapsed_str} | 예상 남은: {eta_str}", end="", flush=True)
            last_pct = pct
        
        time.sleep(1)
    
    elapsed = time.time() - start_time
    stdout = proc.stdout.read()
    
    if proc.returncode == 0:
        print(f"\r  [{'█' * 30}] 100% | 완료: {int(elapsed//60)}:{int(elapsed%60):02d}              ")
        # 결과 추출
        for line in stdout.split('\n'):
            if 'Throughput' in line or 'Latency' in line:
                print(f"  결과: {line.strip()}")
        return True
    else:
        print(f"\n  ERROR (rc={proc.returncode})")
        print(stdout[-500:] if len(stdout) > 500 else stdout)
        return False


def parse_csv(filepath):
    if not os.path.exists(filepath):
        return None
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            return float(row["g_time (ms)"])
    return None


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("=" * 60)
    print("  STARC Remapping Wall 실험 — 64K 남은 작업 실행")
    print("  전체 진행률: 90% → 100%  (2개 남음)")
    print("=" * 60)
    
    L = 65536
    kv_budget = 16672
    
    # --- Run 1: sparse_cluster (cluster_only) ---
    out1 = os.path.join(RESULTS_DIR, f"sparse_cluster_L{L}.csv")
    if os.path.exists(out1):
        print(f"\n[SKIP] sparse_cluster_L{L}.csv 이미 존재")
    else:
        print(f"\n[1/2] 64K sparse_cluster (95%)")
        ok = run_with_progress(L, kv_budget, "cluster_only", "[1/2] 64K sparse_cluster")
        if ok:
            src = os.path.join(SIMULATOR_DIR, "output.csv")
            if os.path.exists(src):
                os.rename(src, out1)
                print(f"  → {out1} 저장 완료")
    
    # --- Run 2: sparse_full ---
    out2 = os.path.join(RESULTS_DIR, f"sparse_full_L{L}.csv")
    if os.path.exists(out2):
        print(f"\n[SKIP] sparse_full_L{L}.csv 이미 존재")
    else:
        print(f"\n[2/2] 64K sparse_full (100%)")
        ok = run_with_progress(L, kv_budget, "full", "[2/2] 64K sparse_full")
        if ok:
            src = os.path.join(SIMULATOR_DIR, "output.csv")
            if os.path.exists(src):
                os.rename(src, out2)
                print(f"  → {out2} 저장 완료")
    
    # --- 전체 결과 요약 ---
    print("\n" + "=" * 60)
    print("  전체 결과 요약 (1K ~ 64K)")
    print("=" * 60)
    
    LENGTHS = [1024, 4096, 16384, 32768, 65536]
    
    print(f"{'L':>8} | {'T_dense':>10} | {'T_sparse':>10} | {'T_cluster':>10} | {'T_full':>10} | {'T_clustering':>12} | {'T_copy':>10} | {'Speedup':>8}")
    print("-" * 100)
    
    for L in LENGTHS:
        t_d = parse_csv(os.path.join(RESULTS_DIR, f"dense_L{L}.csv"))
        t_s = parse_csv(os.path.join(RESULTS_DIR, f"sparse_L{L}.csv"))
        t_c = parse_csv(os.path.join(RESULTS_DIR, f"sparse_cluster_L{L}.csv"))
        t_f = parse_csv(os.path.join(RESULTS_DIR, f"sparse_full_L{L}.csv"))
        
        if all(v is not None for v in [t_d, t_s, t_c, t_f]):
            t_clustering = t_c - t_s
            t_copy = t_f - t_c
            speedup = t_d / t_f if t_f > 0 else 0
            marker = " ← INVERSION" if speedup < 1.0 else ""
            print(f"{L:>8} | {t_d:>10.4f} | {t_s:>10.4f} | {t_c:>10.4f} | {t_f:>10.4f} | {t_clustering:>12.4f} | {t_copy:>10.4f} | {speedup:>7.3f}x{marker}")
        else:
            missing = [n for n, v in [("dense",t_d),("sparse",t_s),("cluster",t_c),("full",t_f)] if v is None]
            print(f"{L:>8} | {'MISSING: ' + ', '.join(missing):>70}")
    
    print("=" * 100)
    print("\n✅ 모든 실험 완료!")


if __name__ == "__main__":
    main()
