"""
Row Utilization Analyzer for AttAcc PIM Traces

Analyzes PIM MAC command addresses from trace files to compute:
1. Row Utilization: How many columns are actually used per activated row
2. Row Activation Count: Total unique rows activated
3. Distribution analysis: Per-bank, per-channel breakdown

This demonstrates that ChunkKV's sparse access pattern causes severe 
row under-utilization on PIM hardware, wasting memory bandwidth.

Usage:
  python analyze_row_utilization.py --trace_full fullkv.trace --trace_chunkkv chunkkv.trace
  python analyze_row_utilization.py --trace chunkkv.trace  # single trace analysis
"""

import argparse
import os
import sys
from collections import defaultdict


# HBM3 organization parameters
N_CHANNEL = 16
N_PCH = 2
N_RANK = 2
N_BG = 4
N_BANK = 4
N_ROW = 2**14  # 16384
N_COL = 2**5   # 32
PREFETCH_SIZE = 32  # bytes

# Address bit widths (matching HBM3-PIM mapping: Co Ro Ba BG Ra Pch Ch)
ADDR_BITS = {
    'col':  5,   # 32 columns
    'row':  14,  # 16384 rows
    'bank': 2,   # 4 banks
    'bg':   2,   # 4 bank groups
    'rank': 1,   # 2 ranks
    'pch':  1,   # 2 pseudo channels
    'ch':   4,   # 16 channels
}
TX_OFFSET = 5  # log2(PREFETCH_SIZE)


def decompose_address(addr):
    """
    Decompose a physical address into HBM3 hierarchy components.
    
    HBM3-PIM mapping order (from LSB to MSB after tx_offset):
    Co(5) | Ro(14) | Ba(2) | BG(2) | Ra(1) | Pch(1) | Ch(4)
    """
    addr_shifted = addr >> TX_OFFSET
    
    col  = addr_shifted & ((1 << ADDR_BITS['col']) - 1)
    addr_shifted >>= ADDR_BITS['col']
    
    row  = addr_shifted & ((1 << ADDR_BITS['row']) - 1)
    addr_shifted >>= ADDR_BITS['row']
    
    bank = addr_shifted & ((1 << ADDR_BITS['bank']) - 1)
    addr_shifted >>= ADDR_BITS['bank']
    
    bg   = addr_shifted & ((1 << ADDR_BITS['bg']) - 1)
    addr_shifted >>= ADDR_BITS['bg']
    
    rank = addr_shifted & ((1 << ADDR_BITS['rank']) - 1)
    addr_shifted >>= ADDR_BITS['rank']
    
    pch  = addr_shifted & ((1 << ADDR_BITS['pch']) - 1)
    addr_shifted >>= ADDR_BITS['pch']
    
    ch   = addr_shifted & ((1 << ADDR_BITS['ch']) - 1)
    
    return {
        'ch': ch, 'pch': pch, 'rank': rank, 
        'bg': bg, 'bank': bank, 'row': row, 'col': col
    }


def analyze_trace(trace_path):
    """
    Analyze a trace file for row utilization metrics.
    
    Returns a dict with detailed metrics including:
    - Total MAC commands
    - Unique rows activated
    - Per-row column utilization
    - Per-bank/channel breakdown
    """
    if not os.path.exists(trace_path):
        print(f"[ERROR] Trace file not found: {trace_path}")
        return None
    
    # Data structures for analysis
    row_columns = defaultdict(set)       # (ch,pch,rank,bg,bank,row) -> set of accessed columns
    bank_rows = defaultdict(set)         # (ch,pch,rank,bg,bank) -> set of activated rows
    channel_mac_count = defaultdict(int) # ch -> count
    total_mac = 0
    total_commands = 0
    command_counts = defaultdict(int)
    
    with open(trace_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            total_commands += 1
            parts = line.split()
            cmd_type = parts[0]
            command_counts[cmd_type] += 1
            
            if cmd_type != "PIM_MAC_AB":
                continue
            
            total_mac += 1
            addr = int(parts[1], 16)
            components = decompose_address(addr)
            
            row_key = (components['ch'], components['pch'], components['rank'],
                       components['bg'], components['bank'], components['row'])
            bank_key = (components['ch'], components['pch'], components['rank'],
                        components['bg'], components['bank'])
            
            row_columns[row_key].add(components['col'])
            bank_rows[bank_key].add(components['row'])
            channel_mac_count[components['ch']] += 1
    
    # Compute metrics
    total_cols = N_COL
    
    # Row utilization
    utilizations = []
    for row_key, cols in row_columns.items():
        util = len(cols) / total_cols * 100
        utilizations.append(util)
    
    avg_util = sum(utilizations) / len(utilizations) if utilizations else 0
    min_util = min(utilizations) if utilizations else 0
    max_util = max(utilizations) if utilizations else 0
    
    # Utilization distribution (histogram)
    util_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    util_hist = [0] * (len(util_bins) - 1)
    for u in utilizations:
        for i in range(len(util_bins) - 1):
            if util_bins[i] <= u < util_bins[i + 1] or (i == len(util_bins) - 2 and u == 100):
                util_hist[i] += 1
                break
    
    # Per-bank row activation count
    bank_act_counts = {k: len(v) for k, v in bank_rows.items()}
    avg_bank_acts = sum(bank_act_counts.values()) / len(bank_act_counts) if bank_act_counts else 0
    
    return {
        'trace_path': trace_path,
        'total_commands': total_commands,
        'command_counts': dict(command_counts),
        'total_mac_commands': total_mac,
        'unique_rows_activated': len(row_columns),
        'avg_row_utilization': avg_util,
        'min_row_utilization': min_util,
        'max_row_utilization': max_util,
        'utilization_histogram': util_hist,
        'utilization_bins': util_bins,
        'num_banks_used': len(bank_rows),
        'avg_rows_per_bank': avg_bank_acts,
        'bank_activation_counts': bank_act_counts,
        'channel_mac_distribution': dict(channel_mac_count),
    }


def print_analysis(result, label=""):
    """Pretty print analysis results."""
    if result is None:
        return
    
    title = f" Row Utilization Analysis: {label}" if label else " Row Utilization Analysis"
    print(f"\n{'=' * 70}")
    print(title)
    print(f"{'=' * 70}")
    print(f"  Trace: {result['trace_path']}")
    print(f"  Total commands: {result['total_commands']}")
    print()
    
    # Command breakdown
    print("  Command Breakdown:")
    for cmd, count in sorted(result['command_counts'].items()):
        print(f"    {cmd:<20} {count:>8}")
    print()
    
    # Row activation metrics
    print("  Row Activation Metrics:")
    print(f"    Total MAC commands:     {result['total_mac_commands']}")
    print(f"    Unique rows activated:  {result['unique_rows_activated']}")
    print(f"    Banks used:             {result['num_banks_used']}")
    print(f"    Avg rows per bank:      {result['avg_rows_per_bank']:.1f}")
    print()
    
    # Row utilization
    print("  Row Utilization (columns used per activated row):")
    print(f"    Average: {result['avg_row_utilization']:.1f}%")
    print(f"    Min:     {result['min_row_utilization']:.1f}%")
    print(f"    Max:     {result['max_row_utilization']:.1f}%")
    print()
    
    # Utilization histogram
    print("  Utilization Distribution:")
    bins = result['utilization_bins']
    hist = result['utilization_histogram']
    total = sum(hist)
    if total > 0:
        max_bar = 40
        for i in range(len(hist)):
            pct = hist[i] / total * 100
            bar_len = int(hist[i] / max(1, max(hist)) * max_bar)
            bar = "█" * bar_len
            print(f"    {bins[i]:3d}-{bins[i+1]:3d}%: {bar} ({hist[i]:>5}, {pct:5.1f}%)")
    print()


def print_comparison(full_result, chunkkv_result):
    """Print side-by-side comparison of Full KV vs ChunkKV."""
    print(f"\n{'=' * 70}")
    print(" COMPARISON: Full KV vs ChunkKV (Naive)")
    print(f"{'=' * 70}")
    
    metrics = [
        ("Total MAC commands",     full_result['total_mac_commands'],    chunkkv_result['total_mac_commands']),
        ("Unique rows activated",  full_result['unique_rows_activated'], chunkkv_result['unique_rows_activated']),
        ("Avg row utilization",    full_result['avg_row_utilization'],   chunkkv_result['avg_row_utilization']),
        ("Banks used",             full_result['num_banks_used'],        chunkkv_result['num_banks_used']),
        ("Avg rows per bank",      full_result['avg_rows_per_bank'],     chunkkv_result['avg_rows_per_bank']),
    ]
    
    print(f"\n  {'Metric':<30} {'Full KV':>12} {'ChunkKV':>12} {'Ratio':>10}")
    print(f"  {'-' * 64}")
    
    for name, full_val, chunk_val in metrics:
        if isinstance(full_val, float):
            ratio = chunk_val / full_val if full_val > 0 else 0
            print(f"  {name:<30} {full_val:>12.1f} {chunk_val:>12.1f} {ratio:>10.2f}x")
        else:
            ratio = chunk_val / full_val if full_val > 0 else 0
            print(f"  {name:<30} {full_val:>12} {chunk_val:>12} {ratio:>10.2f}x")
    
    print()
    
    # Key insight
    row_ratio = chunkkv_result['unique_rows_activated'] / max(1, full_result['unique_rows_activated'])
    mac_ratio = chunkkv_result['total_mac_commands'] / max(1, full_result['total_mac_commands'])
    
    print("  Key Insight:")
    print(f"    MAC commands reduced by {(1-mac_ratio)*100:.1f}%")
    print(f"    BUT unique rows activated only reduced by {(1-row_ratio)*100:.1f}%")
    print(f"    Row utilization dropped from {full_result['avg_row_utilization']:.1f}% to {chunkkv_result['avg_row_utilization']:.1f}%")
    print(f"    → Memory bandwidth wasted: {100 - chunkkv_result['avg_row_utilization']:.1f}%")
    print()
    print("  ⚠ Conclusion: Sparse ChunkKV selection FRAGMENTS row access,")
    print("    causing the PIM to activate nearly the same number of rows")
    print("    despite processing far fewer tokens.")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(
        description="Row Utilization Analyzer for AttAcc PIM Traces",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("--trace", type=str, default=None,
                        help="Path to a single trace file to analyze")
    parser.add_argument("--trace_full", type=str, default=None,
                        help="Path to Full KV trace file")
    parser.add_argument("--trace_chunkkv", type=str, default=None,
                        help="Path to ChunkKV trace file")
    
    args = parser.parse_args()
    
    if args.trace:
        # Single trace analysis
        result = analyze_trace(args.trace)
        print_analysis(result, os.path.basename(args.trace))
    
    elif args.trace_full and args.trace_chunkkv:
        # Comparison mode
        full_result = analyze_trace(args.trace_full)
        chunkkv_result = analyze_trace(args.trace_chunkkv)
        
        print_analysis(full_result, "Full KV")
        print_analysis(chunkkv_result, "ChunkKV (Naive)")
        print_comparison(full_result, chunkkv_result)
    
    else:
        print("Usage:")
        print("  Single trace:  python analyze_row_utilization.py --trace <trace_file>")
        print("  Comparison:    python analyze_row_utilization.py --trace_full <full.trace> --trace_chunkkv <chunkkv.trace>")
        sys.exit(1)


if __name__ == "__main__":
    main()
