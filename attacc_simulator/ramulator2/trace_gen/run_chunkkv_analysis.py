"""
Full KV vs ChunkKV Comparison Analysis Script for AttAcc PIM Simulator

Automates the end-to-end comparison of Full KV and Naive ChunkKV on AttAcc:
1. Generates traces for both Full KV and ChunkKV (multiple compression ratios)
2. Runs ramulator2 simulation for each trace
3. Parses results and computes key metrics
4. Outputs comparison table

Key metrics:
- Latency (memory_system_cycles): Should be nearly identical, proving PIM bottleneck
- MAC command count: Shows reduced computation but not reduced latency
- Latency improvement rate: Expected ~0% despite 50-90% token reduction

Usage:
  python run_chunkkv_analysis.py --seqlen 2048 --dhead 128 --nhead 64
"""

import argparse
import os
import subprocess
import sys
import math
import csv
from datetime import datetime


def make_yaml_file(yaml_path, trace_path, power_constraint=False):
    """Generate ramulator2 YAML configuration file."""
    line = ""
    line += "Frontend:\n"
    line += "  impl: PIMLoadStoreTrace\n"
    line += "  path: {}\n".format(trace_path)
    line += "  clock_ratio: 1\n"
    line += "\n"
    line += "  Translation:\n"
    line += "    impl: NoTranslation\n"
    line += "    max_addr: 2147483648\n"
    line += "\n"
    line += "MemorySystem:\n"
    line += "  impl: PIMDRAM\n"
    line += "  clock_ratio: 1\n"
    line += "  DRAM:\n"
    line += "    impl: HBM3-PIM\n"
    line += "    org:\n"
    line += "      preset: HBM3_8Gb_2R\n"
    line += "      channel: 16\n"
    line += "    timing:\n"
    if power_constraint:
        line += "      preset: HBM3_5.2Gbps\n"
    else:
        line += "      preset: HBM3_5.2Gbps_NPC\n"
    line += "\n"
    line += "  Controller:\n"
    line += "    impl: HBM3-PIM\n"
    line += "    Scheduler:\n"
    line += "      impl: PIM\n"
    line += "    RefreshManager:\n"
    line += "      impl: AllBankHBM3\n"
    line += "    plugins:\n"
    line += "\n"
    line += "  AddrMapper:\n"
    line += "    impl: HBM3-PIM\n"
    
    with open(yaml_path, 'w') as f:
        f.write(line)


def run_trace_gen(script_path, args_dict, output_path):
    """Run trace generation script."""
    cmd_parts = [sys.executable, script_path]
    for key, val in args_dict.items():
        cmd_parts.extend([f"--{key}", str(val)])
    cmd_parts.extend(["--output", output_path])
    
    print(f"  [TRACE] Running: {' '.join(cmd_parts)}")
    result = subprocess.run(cmd_parts, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [ERROR] Trace generation failed: {result.stderr}")
        return False
    print(result.stdout)
    return True


def run_ramulator(ramulator_path, yaml_path):
    """Run ramulator2 simulation and parse output."""
    cmd = f"{ramulator_path} -f {yaml_path}"
    print(f"  [SIM] Running: {cmd}")
    
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True, shell=True)
        output_lines = result.stdout.strip().split('\n')
        output_list = [line.strip() for line in output_lines]
    except subprocess.CalledProcessError as e:
        print(f"  [ERROR] Ramulator failed: {e}")
        return None
    
    # Parse output
    metrics = {
        "cycle": 0,
        "mac": 0,
        "sfm": 0,
        "mvgb": 0,
        "mvsb": 0,
        "wrgb": 0,
    }
    
    for line in output_list:
        if "memory_system_cycles" in line:
            metrics["cycle"] += int(line.split()[-1])
        elif "mac" in line and "total_num" in line:
            metrics["mac"] += int(line.split()[-1])
        elif "softmax_requests" in line:
            metrics["sfm"] += int(line.split()[-1])
        elif "move_to_gemv_buffer" in line:
            metrics["mvgb"] += int(line.split()[-1])
        elif "move_to_softmax_buffer" in line:
            metrics["mvsb"] += int(line.split()[-1])
        elif "write_to_gemv_buffer" in line:
            metrics["wrgb"] += int(line.split()[-1])
    
    return metrics


def count_trace_lines(trace_path):
    """Count total commands in trace file."""
    with open(trace_path, 'r') as f:
        return sum(1 for _ in f)


def analyze_trace_row_info(trace_path):
    """
    Analyze MAC addresses in trace to estimate row activation patterns.
    
    Parse PIM_MAC_AB commands and decompose addresses into HBM3 hierarchy
    to count unique rows activated.
    """
    # HBM3 address decomposition (matching hbm3_pim_linear_mappers.cpp)
    # Address bits after tx_offset (prefetch_size = 32B → tx_offset = 5 bits):
    # Co(5) | Ro(14) | Ba(2) | BG(2) | Ra(1) | Pch(1) | Ch(4)
    tx_offset = 5  # log2(32) for prefetch size
    
    addr_bits = {
        'col': 5,   # 2^5 = 32 columns
        'row': 14,   # 2^14 = 16384 rows
        'bank': 2,   # 4 banks
        'bg': 2,     # 4 bank groups
        'rank': 1,   # 2 ranks
        'pch': 1,    # 2 pseudo channels
        'ch': 4,     # 16 channels
    }
    
    unique_rows = set()
    total_mac_commands = 0
    col_accesses_per_row = {}
    
    with open(trace_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line.startswith("PIM_MAC_AB"):
                continue
            
            total_mac_commands += 1
            parts = line.split()
            addr_hex = parts[1]
            addr = int(addr_hex, 16)
            
            # Decompose address (HBM3-PIM mapping: Ch pCH Ra BG Ba Ro Co)
            addr_shifted = addr >> tx_offset
            
            col = addr_shifted & ((1 << addr_bits['col']) - 1)
            addr_shifted >>= addr_bits['col']
            
            row = addr_shifted & ((1 << addr_bits['row']) - 1)
            addr_shifted >>= addr_bits['row']
            
            bank = addr_shifted & ((1 << addr_bits['bank']) - 1)
            addr_shifted >>= addr_bits['bank']
            
            bg = addr_shifted & ((1 << addr_bits['bg']) - 1)
            addr_shifted >>= addr_bits['bg']
            
            rank = addr_shifted & ((1 << addr_bits['rank']) - 1)
            addr_shifted >>= addr_bits['rank']
            
            pch = addr_shifted & ((1 << addr_bits['pch']) - 1)
            addr_shifted >>= addr_bits['pch']
            
            ch = addr_shifted & ((1 << addr_bits['ch']) - 1)
            
            row_key = (ch, pch, rank, bg, bank, row)
            unique_rows.add(row_key)
            
            if row_key not in col_accesses_per_row:
                col_accesses_per_row[row_key] = set()
            col_accesses_per_row[row_key].add(col)
    
    # Calculate row utilization
    total_cols_per_row = 2 ** addr_bits['col']  # 32
    if len(col_accesses_per_row) > 0:
        utilizations = [len(cols) / total_cols_per_row * 100 
                       for cols in col_accesses_per_row.values()]
        avg_utilization = sum(utilizations) / len(utilizations)
    else:
        avg_utilization = 0
        utilizations = []
    
    return {
        'total_mac_commands': total_mac_commands,
        'unique_rows': len(unique_rows),
        'avg_row_utilization': avg_utilization,
        'min_row_utilization': min(utilizations) if utilizations else 0,
        'max_row_utilization': max(utilizations) if utilizations else 0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Full KV vs ChunkKV comparison analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-dh", "--dhead", type=int, default=128)
    parser.add_argument("-nh", "--nhead", type=int, default=64)
    parser.add_argument("-l", "--seqlen", type=int, default=2048)
    parser.add_argument("-maxl", "--maxlen", type=int, default=4096)
    parser.add_argument("-db", "--dbyte", type=int, default=2)
    parser.add_argument("-cl", "--chunk_length", type=int, default=20)
    parser.add_argument("--compression_ratios", type=float, nargs='+',
                        default=[0.5, 0.7, 0.9],
                        help="List of compression ratios to test")
    parser.add_argument("--ramulator_dir", type=str, default=None,
                        help="Path to ramulator2 directory")
    parser.add_argument("--output_csv", type=str, default="chunkkv_analysis.csv",
                        help="Output CSV file path")

    args = parser.parse_args()

    # Determine paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ramulator_dir = args.ramulator_dir or os.path.dirname(script_dir)
    ramulator_exe = os.path.join(ramulator_dir, "ramulator2")
    fullkv_script = os.path.join(script_dir, "gen_trace_attacc_bank.py")
    chunkkv_script = os.path.join(script_dir, "gen_trace_chunkkv.py")
    
    if not os.path.exists(ramulator_exe):
        print(f"[ERROR] ramulator2 executable not found at {ramulator_exe}")
        print(f"        Please build ramulator2 first or specify --ramulator_dir")
        sys.exit(1)

    tCK_ns = 0.769  # HBM3 5.2Gbps clock period

    print("=" * 70)
    print(" Full KV vs ChunkKV (Naive) Comparison Analysis")
    print("=" * 70)
    print(f" Sequence Length: {args.seqlen}")
    print(f" dhead: {args.dhead}, nhead: {args.nhead}, dbyte: {args.dbyte}")
    print(f" Chunk Length: {args.chunk_length}")
    print(f" Compression Ratios: {args.compression_ratios}")
    print("=" * 70)

    results = []

    # ---- Step 1: Run Full KV baseline ----
    print("\n[1/2] Running Full KV (baseline)...")
    fullkv_trace = os.path.join(ramulator_dir, "fullkv_baseline.trace")
    fullkv_yaml = os.path.join(ramulator_dir, "fullkv_baseline.yaml")
    
    trace_args = {
        "dhead": args.dhead,
        "nhead": args.nhead,
        "seqlen": args.seqlen,
        "maxlen": args.maxlen,
        "dbyte": args.dbyte,
    }
    
    if run_trace_gen(fullkv_script, trace_args, fullkv_trace):
        make_yaml_file(fullkv_yaml, fullkv_trace)
        fullkv_metrics = run_ramulator(ramulator_exe, fullkv_yaml)
        fullkv_trace_count = count_trace_lines(fullkv_trace)
        fullkv_row_info = analyze_trace_row_info(fullkv_trace)
        
        if fullkv_metrics:
            latency_us = fullkv_metrics['cycle'] * tCK_ns / 1000
            results.append({
                'method': 'Full KV',
                'compression_ratio': 0.0,
                'effective_tokens': args.seqlen,
                'cycles': fullkv_metrics['cycle'],
                'latency_us': latency_us,
                'mac_count': fullkv_metrics['mac'],
                'trace_commands': fullkv_trace_count,
                'unique_rows': fullkv_row_info['unique_rows'],
                'avg_row_util': fullkv_row_info['avg_row_utilization'],
                'total_mac_cmds': fullkv_row_info['total_mac_commands'],
                'latency_improvement': 0.0,
            })
            print(f"  ✓ Cycles: {fullkv_metrics['cycle']}, "
                  f"Latency: {latency_us:.2f} µs, "
                  f"MAC: {fullkv_metrics['mac']}, "
                  f"Unique Rows: {fullkv_row_info['unique_rows']}, "
                  f"Avg Row Util: {fullkv_row_info['avg_row_utilization']:.1f}%")
        
        # Clean up
        os.remove(fullkv_trace)
        os.remove(fullkv_yaml)
    
    # ---- Step 2: Run ChunkKV for each compression ratio ----
    print(f"\n[2/2] Running ChunkKV (compression ratios: {args.compression_ratios})...")
    
    for cr in args.compression_ratios:
        print(f"\n  --- Compression Ratio: {cr} (keep {(1-cr)*100:.0f}%) ---")
        
        chunkkv_trace = os.path.join(ramulator_dir, f"chunkkv_cr{cr}.trace")
        chunkkv_yaml = os.path.join(ramulator_dir, f"chunkkv_cr{cr}.yaml")
        
        chunkkv_args = {
            "dhead": args.dhead,
            "nhead": args.nhead,
            "seqlen": args.seqlen,
            "maxlen": args.maxlen,
            "dbyte": args.dbyte,
            "chunk_length": args.chunk_length,
            "compression_ratio": cr,
        }
        
        if run_trace_gen(chunkkv_script, chunkkv_args, chunkkv_trace):
            make_yaml_file(chunkkv_yaml, chunkkv_trace)
            chunkkv_metrics = run_ramulator(ramulator_exe, chunkkv_yaml)
            chunkkv_trace_count = count_trace_lines(chunkkv_trace)
            chunkkv_row_info = analyze_trace_row_info(chunkkv_trace)
            
            if chunkkv_metrics and results:
                baseline_cycles = results[0]['cycles']
                improvement = (baseline_cycles - chunkkv_metrics['cycle']) / baseline_cycles * 100
                effective_tokens = int(args.seqlen * (1 - cr))
                latency_us = chunkkv_metrics['cycle'] * tCK_ns / 1000
                
                results.append({
                    'method': f'ChunkKV (cr={cr})',
                    'compression_ratio': cr,
                    'effective_tokens': effective_tokens,
                    'cycles': chunkkv_metrics['cycle'],
                    'latency_us': latency_us,
                    'mac_count': chunkkv_metrics['mac'],
                    'trace_commands': chunkkv_trace_count,
                    'unique_rows': chunkkv_row_info['unique_rows'],
                    'avg_row_util': chunkkv_row_info['avg_row_utilization'],
                    'total_mac_cmds': chunkkv_row_info['total_mac_commands'],
                    'latency_improvement': improvement,
                })
                print(f"  ✓ Cycles: {chunkkv_metrics['cycle']}, "
                      f"Latency: {latency_us:.2f} µs, "
                      f"MAC: {chunkkv_metrics['mac']}, "
                      f"Improvement: {improvement:.1f}%, "
                      f"Unique Rows: {chunkkv_row_info['unique_rows']}, "
                      f"Avg Row Util: {chunkkv_row_info['avg_row_utilization']:.1f}%")
            
            # Clean up
            os.remove(chunkkv_trace)
            os.remove(chunkkv_yaml)

    # ---- Step 3: Print comparison table ----
    print("\n" + "=" * 115)
    print(" COMPARISON RESULTS: Full KV vs ChunkKV (Naive) on AttAcc PIM")
    print("=" * 115)
    
    # Table 1: Raw metrics
    header = f"{'Method':<22} {'Tokens':<8} {'Cycles':<12} {'Latency(µs)':<13} {'MAC Cmds':<10} {'Rows':<8} {'Row Util(%)':<12} {'Lat.Impr(%)':<12}"
    print(header)
    print("-" * 115)
    
    for r in results:
        line = f"{r['method']:<22} {r['effective_tokens']:<8} {r['cycles']:<12} {r['latency_us']:<13.2f} {r['mac_count']:<10} {r['unique_rows']:<8} {r['avg_row_util']:<12.1f} {r['latency_improvement']:<12.1f}"
        print(line)
    
    print("-" * 115)

    # Table 2: Ideal vs Actual speedup analysis (KEY for paper)
    if len(results) > 1:
        baseline = results[0]
        print(f"\n{'=' * 115}")
        print(" IDEAL vs ACTUAL SPEEDUP ANALYSIS (Problem Statement)")
        print(f"{'=' * 115}")
        print(f"  {'Method':<22} {'Token Red.':<12} {'Ideal Speedup':<15} {'Actual Speedup':<16} {'PIM Efficiency':<16} {'BW Wasted':<12} {'Row Red.':<10}")
        print(f"  {'-' * 103}")
        
        for r in results[1:]:
            token_reduction = r['compression_ratio'] * 100
            ideal_speedup = 1 / (1 - r['compression_ratio']) if r['compression_ratio'] < 1 else float('inf')
            actual_speedup = baseline['cycles'] / r['cycles'] if r['cycles'] > 0 else 0
            pim_efficiency = actual_speedup / ideal_speedup * 100 if ideal_speedup > 0 else 0
            bw_wasted = 100 - r['avg_row_util']
            row_reduction = (1 - r['unique_rows'] / baseline['unique_rows']) * 100 if baseline['unique_rows'] > 0 else 0
            
            print(f"  {r['method']:<22} {token_reduction:>6.0f}%     {ideal_speedup:>6.1f}x        {actual_speedup:>6.2f}x         {pim_efficiency:>6.1f}%         {bw_wasted:>6.1f}%     {row_reduction:>6.1f}%")
        
        print(f"  {'-' * 103}")

        # Key findings
        print(f"\n Key Findings:")
        print(f"   • Full KV baseline: {baseline['latency_us']:.2f} µs ({baseline['cycles']} cycles)")
        for r in results[1:]:
            ideal_speedup = 1 / (1 - r['compression_ratio'])
            actual_speedup = baseline['cycles'] / r['cycles'] if r['cycles'] > 0 else 0
            pim_efficiency = actual_speedup / ideal_speedup * 100
            bw_wasted = 100 - r['avg_row_util']
            row_reduction = (1 - r['unique_rows'] / baseline['unique_rows']) * 100
            
            print(f"   • {r['method']}:")
            print(f"     - Tokens removed: {r['compression_ratio']*100:.0f}%, "
                  f"but rows only reduced by {row_reduction:.1f}%")
            print(f"     - Ideal speedup: {ideal_speedup:.1f}×, "
                  f"Actual: {actual_speedup:.2f}× → "
                  f"PIM Efficiency: {pim_efficiency:.1f}%")
            print(f"     - Row utilization: {r['avg_row_util']:.1f}% → "
                  f"{bw_wasted:.1f}% of memory bandwidth WASTED")
        
        print(f"\n ⚠ Conclusion:")
        worst_cr = max(r['compression_ratio'] for r in results[1:])
        worst_r = [r for r in results[1:] if r['compression_ratio'] == worst_cr][0]
        ideal_speedup = 1 / (1 - worst_cr)
        actual_speedup = baseline['cycles'] / worst_r['cycles'] if worst_r['cycles'] > 0 else 0
        pim_efficiency = actual_speedup / ideal_speedup * 100
        row_reduction = (1 - worst_r['unique_rows'] / baseline['unique_rows']) * 100
        
        print(f"   ChunkKV removes {worst_cr*100:.0f}% of KV tokens (ideal {ideal_speedup:.0f}× speedup),")
        print(f"   but PIM only achieves {actual_speedup:.1f}× ({pim_efficiency:.1f}% efficiency).")
        print(f"   Row activations reduced by only {row_reduction:.1f}%,")
        print(f"   and {100-worst_r['avg_row_util']:.1f}% of activated memory bandwidth is wasted.")
        print(f"   → Naive sparse attention is INEFFICIENT on PIM architectures.")
        print(f"   → Physical data remapping/packing is needed to exploit sparsity on PIM.")
    
    # ---- Step 4: Save CSV ----
    output_csv_path = os.path.join(ramulator_dir, args.output_csv)
    with open(output_csv_path, 'w', newline='') as csvfile:
        if results:
            # Add computed fields
            for r in results:
                if results[0]['cycles'] > 0 and r['compression_ratio'] > 0:
                    ideal = 1 / (1 - r['compression_ratio'])
                    actual = results[0]['cycles'] / r['cycles'] if r['cycles'] > 0 else 0
                    r['ideal_speedup'] = ideal
                    r['actual_speedup'] = actual
                    r['pim_efficiency'] = actual / ideal * 100
                    r['bw_wasted'] = 100 - r['avg_row_util']
                else:
                    r['ideal_speedup'] = 1.0
                    r['actual_speedup'] = 1.0
                    r['pim_efficiency'] = 100.0
                    r['bw_wasted'] = 0.0
            
            fieldnames = results[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
    
    print(f"\n Results saved to: {output_csv_path}")


if __name__ == "__main__":
    main()
