"""
ChunkKV Sparse Trace Generator for AttAcc PIM Simulator

Generates PIM command traces that simulate the memory access patterns 
when ChunkKV (sparse attention) is naively applied to PIM hardware.

Key insight for problem statement:
  PIM uses All-Bank MAC (MACAB) which activates an ENTIRE row across all banks.
  Even if ChunkKV logically skips tokens, the PIM must still activate the same rows 
  because the selected chunks are scattered across many rows.

  The addresses preserve the original KV cache layout (no physical remapping),
  so sparse chunk selection does NOT reduce the number of row activations.

Two modes:
  1. "ideal" mode: Only access selected chunks' addresses (shows best-case latency)
  2. "realistic" mode (default): Must activate all rows that contain ANY selected chunk.
     Since MACAB activates the entire row, we must read the full row even if only
     a fraction of tokens in that row are selected. This is the realistic PIM behavior.

Usage:
  python gen_trace_chunkkv.py --seqlen 2048 --compression_ratio 0.9 --chunk_length 20 \
         --dhead 128 --nhead 64 --output chunkkv.trace
"""

import argparse
import math
import copy
import numpy as np

model = "gpt-3-175B"

dhead = 128
max_L = 2048
data_size = 16  # FP 16

n_attacc = 8
max_n_hbm = 8
n_hbm = 5
n_channel = 16
n_pch = 2
n_rank = 2
n_bank = 4
n_bg = 4
n_row = pow(2, 14)
n_col = pow(2, 5)
prefetch_size = 32  # byte
n_mac = 16


# Granularity size
HBM_GS = {}
HBM_GS['col']     = prefetch_size
HBM_GS['row']     = n_col * HBM_GS['col']
HBM_GS['ba']      = n_row * HBM_GS['row']
HBM_GS['bg']      = n_bank * HBM_GS['ba']
HBM_GS['rank']    = n_bg * HBM_GS['bg']
HBM_GS['pch']     = n_rank * HBM_GS['rank']
HBM_GS['ch']      = n_pch * HBM_GS['pch']
HBM_GS['hbm']     = n_channel * HBM_GS['ch']
HBM_GS['attacc']  = max_n_hbm * HBM_GS['hbm']


cmd_score_wrgb   = []
cmd_score_mac    = []
cmd_score_mvsb   = []
cmd_sfm          = []
cmd_context_mvgb = []
cmd_context_mac  = []
cmd_context_mvsb = []

valid_channels = []

# ---- ChunkKV analysis metrics ---- #
chunkkv_metrics = {
    'total_chunks': 0,
    'selected_chunks': 0,
    'selected_indices': [],
    'total_L': 0,
    'effective_L': 0,
    'rows_with_selected_tokens': 0,
    'total_rows_in_full_kv': 0,
    'row_utilization_pct': 0.0,
}


def cmd_list_reset():
    global cmd_score_wrgb, cmd_score_mac, cmd_score_mvsb
    global cmd_sfm, cmd_context_mvgb, cmd_context_mac, cmd_context_mvsb
    global valid_channels

    cmd_score_wrgb   = []
    cmd_score_mac    = []
    cmd_score_mvsb   = []
    cmd_sfm          = []
    cmd_context_mvgb = []
    cmd_context_mac  = []
    cmd_context_mvsb = []

    valid_channels = []


def select_chunks(L, chunk_length, compression_ratio, seed=42):
    """
    Simulate ChunkKV chunk selection.
    """
    np.random.seed(seed)
    
    num_complete_chunks = L // chunk_length
    remaining_tokens = L % chunk_length
    total_chunks = num_complete_chunks + (1 if remaining_tokens > 0 else 0)
    
    n_chunks_kept = max(1, int(total_chunks * (1 - compression_ratio)))
    
    chunk_scores = np.random.rand(total_chunks)
    
    selected_chunk_indices = np.argsort(chunk_scores)[-n_chunks_kept:]
    selected_chunk_indices = np.sort(selected_chunk_indices)
    
    selected_token_indices = []
    for chunk_idx in selected_chunk_indices:
        if chunk_idx < num_complete_chunks:
            start = chunk_idx * chunk_length
            end = start + chunk_length
        else:
            start = num_complete_chunks * chunk_length
            end = L
        selected_token_indices.extend(range(start, end))
    
    chunk_info = {
        'total_chunks': total_chunks,
        'selected_chunks': n_chunks_kept,
        'selected_chunk_indices': selected_chunk_indices.tolist(),
        'total_L': L,
        'effective_L': len(selected_token_indices),
        'compression_ratio': compression_ratio,
        'chunk_length': chunk_length,
    }
    
    return selected_token_indices, chunk_info


def compute_row_analysis(L, selected_token_indices, chunk_length):
    """
    Analyze which physical rows contain selected tokens.
    
    In the AttAcc bank-level PIM architecture:
    - Tokens are laid out across (pCH × Rank × BG) parallel units
    - Each parallel unit has n_bank banks
    - Within each bank, tokens are stored in rows, with n_col columns per row
    - dhead dimension is spread across (n_bank × n_mac) = 64 elements
    
    The key insight: Each n_idx in score_mac corresponds to one row's worth of tokens.
    n_idx = token_idx // (n_pch * n_rank * n_bg)
    
    So if token_idx 0..15 map to n_idx=0 (row 0), 
       token_idx 16..31 map to n_idx=1 (row 1), etc.
    
    ChunkKV selects scattered chunks, so the n_idx set touched is nearly as large
    as full KV, even though far fewer tokens within each n_idx range are selected.
    """
    parallel_units = n_pch * n_rank * n_bg  # 16
    
    # Full KV: all n_idx from 0 to L/parallel_units
    full_n_idxs = set(range(math.ceil(L / parallel_units)))
    
    # ChunkKV: only n_idx touched by selected tokens
    selected_n_idxs = set()
    for token_idx in selected_token_indices:
        selected_n_idxs.add(token_idx // parallel_units)
    
    # Each n_idx corresponds to one row activation (per iteration of dhead/bank/mac)
    # The total row activations = |n_idx set| × (dhead / n_bank / n_mac)
    cols_per_n_idx = math.ceil(dhead / n_bank / n_mac)  # 2
    
    full_row_acts = len(full_n_idxs) * cols_per_n_idx
    selected_row_acts = len(selected_n_idxs) * cols_per_n_idx
    
    # Row utilization: for each n_idx, how many of the parallel_units tokens are selected?
    n_idx_utilization = {}
    for n_idx in selected_n_idxs:
        total_tokens_in_n_idx = min(parallel_units, L - n_idx * parallel_units)
        selected_in_n_idx = sum(1 for t in selected_token_indices 
                               if t // parallel_units == n_idx)
        n_idx_utilization[n_idx] = selected_in_n_idx / total_tokens_in_n_idx * 100
    
    avg_utilization = (sum(n_idx_utilization.values()) / len(n_idx_utilization) 
                       if n_idx_utilization else 0)
    
    return {
        'full_kv_n_idxs': len(full_n_idxs),
        'chunkkv_n_idxs': len(selected_n_idxs),
        'n_idx_reduction_pct': (1 - len(selected_n_idxs) / len(full_n_idxs)) * 100,
        'full_kv_row_acts': full_row_acts,
        'chunkkv_row_acts': selected_row_acts,
        'row_act_reduction_pct': (1 - selected_row_acts / full_row_acts) * 100 if full_row_acts > 0 else 0,
        'avg_row_utilization': avg_utilization,
        'n_idx_utilization': n_idx_utilization,
    }


def Attention_FullKV_for_reference(L, key_addr, val_addr, itr, valid_channel=n_channel):
    """
    Full KV attention — identical to gen_trace_attacc_bank.py's Attention().
    This is used to generate the FULL KV trace for comparison.
    """
    cmd_score_wrgb.append([])
    cmd_score_mac.append([])
    cmd_score_mvsb.append([])
    cmd_sfm.append([])
    cmd_context_mvgb.append([])
    cmd_context_mac.append([])
    cmd_context_mvsb.append([])
    valid_channels.append(valid_channel)

    def score_cpvec(addr_offset, L):
        for ba_idx in range(n_bank):
            for col_idx in range(math.ceil(dhead / n_bank / n_mac)):
                for lch in range(math.ceil(valid_channel)):
                    addr = addr_offset + lch * HBM_GS['ch'] + ba_idx * HBM_GS['ba'] + col_idx
                    hex_addr = hex(addr)[2:]
                    cmd_score_wrgb[itr].append("PIM_WR_GB 0x{0:0>8}".format(hex_addr))

    def score_mac(addr_offset, L):
        for n_idx in range(math.ceil(L / n_pch / n_rank / n_bg)):
            cmd_score_mac[itr].append([])
            for k_idx in range(math.ceil(dhead / n_bank / n_mac)):
                idx = k_idx + n_idx * math.ceil(dhead / n_bank / n_mac)
                for lch in range(math.ceil(valid_channel)):
                    addr = addr_offset + lch * HBM_GS['ch'] + idx * HBM_GS['col']
                    hex_addr = hex(addr)[2:]
                    cmd_score_mac[itr][-1].append("PIM_MAC_AB 0x{0:0>8}".format(hex_addr))

            if n_idx % 16 == 15 or n_idx == math.ceil(L / n_pch / n_rank / n_bg) - 1:
                cmd_score_mvsb[itr].append([])
                for bg_idx in range(n_bg):
                    for rank in range(n_rank):
                        for lch in range(math.ceil(valid_channel)):
                            bank_addr = addr_offset + lch * HBM_GS['ch'] + rank * HBM_GS['rank'] + \
                                        bg_idx * HBM_GS['bg']
                            hex_addr = hex(bank_addr)[2:]
                            cmd_score_mvsb[itr][-1].append("PIM_MV_SB 0x{0:0>8}".format(hex_addr))

    def context_cpvec(addr_offset, L):
        for rank in range(n_rank):
            for bg_idx in range(n_bg):
                for col_idx in range(math.ceil(L / (n_pch * n_rank * n_bg * n_mac))):
                    for lch in range(math.ceil(valid_channel)):
                        addr = addr_offset + lch * HBM_GS['ch'] + rank * HBM_GS['rank'] + \
                               bg_idx * HBM_GS['bg'] + col_idx
                        hex_addr = hex(addr)[2:]
                        cmd_context_mvgb[itr].append("PIM_MV_GB 0x{0:0>8}".format(hex_addr))

    def context_mac(addr_offset, L):
        for n_idx in range(math.ceil(dhead / (n_bank * n_mac))):
            cmd_context_mac[itr].append([])
            for k_idx in range(math.ceil(L / (n_pch * n_rank * n_bg))):
                idx = k_idx + n_idx * math.ceil(L / (n_pch * n_rank * n_bg))
                for lch in range(math.ceil(valid_channel)):
                    addr = addr_offset + lch * HBM_GS['ch'] + idx * HBM_GS['col']
                    hex_addr = hex(addr)[2:]
                    cmd_context_mac[itr][-1].append("PIM_MAC_AB 0x{0:0>8}".format(hex_addr))

            cmd_context_mvsb[itr].append([])
            for ba_idx in range(n_bank):
                for rank in range(n_rank):
                    for lch in range(math.ceil(valid_channel)):
                        bank_addr = addr_offset + lch * HBM_GS['ch'] + rank * HBM_GS['rank'] + \
                                    ba_idx * HBM_GS['ba']
                        hex_addr = hex(bank_addr)[2:]
                        cmd_context_mvsb[itr][-1].append("PIM_MV_SB 0x{0:0>8}".format(hex_addr))

    def softmax(L):
        for lch in range(math.ceil(valid_channel)):
            addr = lch * HBM_GS['ch']
            hex_addr = hex(addr)[2:]
            cmd_sfm[itr].append("PIM_SFM 0x{0:0>8}".format(hex_addr))

    score_cpvec(key_addr, L)
    score_mac(key_addr, L)
    softmax(L)
    context_cpvec(val_addr, L)
    context_mac(val_addr, L)


def Attention_ChunkKV(L, key_addr, val_addr, itr, selected_token_indices, valid_channel=n_channel):
    """
    Generate PIM commands for attention with ChunkKV sparse access.
    
    REALISTIC PIM behavior: 
    Since PIM uses All-Bank MAC (MACAB), each MAC command activates the ENTIRE row.
    If ANY selected token is in n_idx range X, the PIM must activate the row for X
    and perform the MAC on the whole row — it CANNOT skip individual tokens within a row.
    
    This means the PIM still needs to:
    1. Activate all rows that contain at least one selected token
    2. MAC the entire row (wasting computation on unselected tokens)
    3. Only the number of MVSB/softmax entries is reduced (to effective_L)
    
    The net effect: Row Activation Count barely decreases even with 90% token pruning,
    because chunks are scattered across many different rows.
    """
    cmd_score_wrgb.append([])
    cmd_score_mac.append([])
    cmd_score_mvsb.append([])
    cmd_sfm.append([])
    cmd_context_mvgb.append([])
    cmd_context_mac.append([])
    cmd_context_mvsb.append([])
    valid_channels.append(valid_channel)

    effective_L = len(selected_token_indices)
    parallel_units = n_pch * n_rank * n_bg

    # Determine which n_idx ranges contain selected tokens
    selected_n_idxs = sorted(set(t // parallel_units for t in selected_token_indices))

    def score_cpvec(addr_offset):
        """Write query vector to GEMV buffer — same as full KV"""
        for ba_idx in range(n_bank):
            for col_idx in range(math.ceil(dhead / n_bank / n_mac)):
                for lch in range(math.ceil(valid_channel)):
                    addr = addr_offset + lch * HBM_GS['ch'] + ba_idx * HBM_GS['ba'] + col_idx
                    hex_addr = hex(addr)[2:]
                    cmd_score_wrgb[itr].append("PIM_WR_GB 0x{0:0>8}".format(hex_addr))

    def score_mac(addr_offset):
        """
        Score MAC: Must activate ALL rows containing selected tokens.
        Each n_idx → MACAB on that row. PIM reads the ENTIRE row,
        cannot skip individual token positions within the activated row.
        """
        for ni, n_idx in enumerate(selected_n_idxs):
            cmd_score_mac[itr].append([])
            for k_idx in range(math.ceil(dhead / n_bank / n_mac)):
                idx = k_idx + n_idx * math.ceil(dhead / n_bank / n_mac)
                for lch in range(math.ceil(valid_channel)):
                    addr = addr_offset + lch * HBM_GS['ch'] + idx * HBM_GS['col']
                    hex_addr = hex(addr)[2:]
                    cmd_score_mac[itr][-1].append("PIM_MAC_AB 0x{0:0>8}".format(hex_addr))

            if ni % 16 == 15 or ni == len(selected_n_idxs) - 1:
                cmd_score_mvsb[itr].append([])
                for bg_idx in range(n_bg):
                    for rank in range(n_rank):
                        for lch in range(math.ceil(valid_channel)):
                            bank_addr = addr_offset + lch * HBM_GS['ch'] + rank * HBM_GS['rank'] + \
                                        bg_idx * HBM_GS['bg']
                            hex_addr = hex(bank_addr)[2:]
                            cmd_score_mvsb[itr][-1].append("PIM_MV_SB 0x{0:0>8}".format(hex_addr))

    def context_cpvec(addr_offset):
        """Write score vector — only for effective_L positions"""
        for rank in range(n_rank):
            for bg_idx in range(n_bg):
                for col_idx in range(math.ceil(effective_L / (n_pch * n_rank * n_bg * n_mac))):
                    for lch in range(math.ceil(valid_channel)):
                        addr = addr_offset + lch * HBM_GS['ch'] + rank * HBM_GS['rank'] + \
                               bg_idx * HBM_GS['bg'] + col_idx
                        hex_addr = hex(addr)[2:]
                        cmd_context_mvgb[itr].append("PIM_MV_GB 0x{0:0>8}".format(hex_addr))

    def context_mac(addr_offset):
        """
        Context MAC: Same as score — must activate rows containing selected tokens.
        The PIM reads value vectors from these rows at their ORIGINAL positions.
        """
        for n_idx in range(math.ceil(dhead / (n_bank * n_mac))):
            cmd_context_mac[itr].append([])
            for k_idx in selected_n_idxs:
                idx = k_idx + n_idx * math.ceil(L / parallel_units)
                for lch in range(math.ceil(valid_channel)):
                    addr = addr_offset + lch * HBM_GS['ch'] + idx * HBM_GS['col']
                    hex_addr = hex(addr)[2:]
                    cmd_context_mac[itr][-1].append("PIM_MAC_AB 0x{0:0>8}".format(hex_addr))

            cmd_context_mvsb[itr].append([])
            for ba_idx in range(n_bank):
                for rank in range(n_rank):
                    for lch in range(math.ceil(valid_channel)):
                        bank_addr = addr_offset + lch * HBM_GS['ch'] + rank * HBM_GS['rank'] + \
                                    ba_idx * HBM_GS['ba']
                        hex_addr = hex(bank_addr)[2:]
                        cmd_context_mvsb[itr][-1].append("PIM_MV_SB 0x{0:0>8}".format(hex_addr))

    def softmax():
        for lch in range(math.ceil(valid_channel)):
            addr = lch * HBM_GS['ch']
            hex_addr = hex(addr)[2:]
            cmd_sfm[itr].append("PIM_SFM 0x{0:0>8}".format(hex_addr))

    score_cpvec(key_addr)
    score_mac(key_addr)
    softmax()
    context_cpvec(val_addr)
    context_mac(val_addr)


def run_attention(dhead, n_head_per_hbm, L, trace_file_name,
                  use_chunkkv=False, chunk_length=20, compression_ratio=0.9):
    """Generate attention trace (Full KV or ChunkKV)."""
    partition_size = math.ceil(max_L * dhead / (n_pch * n_rank * n_bg * n_bank))
    v_offset = pow(2, 23)

    selected_indices = None
    chunk_info = None
    row_analysis = None
    
    if use_chunkkv:
        selected_indices, chunk_info = select_chunks(L, chunk_length, compression_ratio)
        row_analysis = compute_row_analysis(L, selected_indices, chunk_length)
        
        chunkkv_metrics['total_chunks'] = chunk_info['total_chunks']
        chunkkv_metrics['selected_chunks'] = chunk_info['selected_chunks']
        chunkkv_metrics['selected_indices'] = chunk_info['selected_chunk_indices']
        chunkkv_metrics['total_L'] = chunk_info['total_L']
        chunkkv_metrics['effective_L'] = chunk_info['effective_L']
        chunkkv_metrics['rows_with_selected_tokens'] = row_analysis['chunkkv_n_idxs']
        chunkkv_metrics['total_rows_in_full_kv'] = row_analysis['full_kv_n_idxs']
        chunkkv_metrics['row_utilization_pct'] = row_analysis['avg_row_utilization']

    cmd_list_reset()

    num_itr = math.ceil(n_head_per_hbm / n_channel)
    for itr in range(num_itr):
        remainder = 0
        if (n_head_per_hbm / ((itr + 1) * n_channel) < 1):
            remainder = n_head_per_hbm % n_channel
        key_addr = itr * partition_size
        val_addr = key_addr + v_offset
        
        vc = n_channel if remainder == 0 else remainder
        
        if use_chunkkv:
            Attention_ChunkKV(L, key_addr, val_addr, itr, selected_indices, vc)
        else:
            Attention_FullKV_for_reference(L, key_addr, val_addr, itr, vc)

    # Overlapping Commands — same structure as gen_trace_attacc_bank.py
    barrier = []
    for lch in range(n_channel):
        addr = lch * HBM_GS['ch']
        hex_addr = hex(addr)[2:]
        barrier.append("PIM_BARRIER 0x{0:0>8}".format(hex_addr))

    total_cmd = []
    
    for i in range(0, num_itr - 1, 2):
        # Head0: Score
        total_cmd += cmd_score_wrgb[i]
        if i == 0:
            for j in range(valid_channels[i]):
                if len(cmd_score_mac[i]) > 0 and j < len(cmd_score_mac[i][0]):
                    total_cmd.append(cmd_score_mac[i][0][j])
        total_cmd += barrier

        length = math.ceil(len(cmd_score_mac[i]) / 16) if len(cmd_score_mac[i]) > 0 else 1
        for j in range(0, length + 1):
            if not j == length:
                stride = 16
                for k in range(stride):
                    if (j * stride + k) >= len(cmd_score_mac[i]):
                        break
                    total_cmd += cmd_score_mac[i][j * stride + k]
            if not j == 0:
                mvsb_idx = j - 1
                if mvsb_idx < len(cmd_score_mvsb[i]):
                    total_cmd += cmd_score_mvsb[i][mvsb_idx]
            if not j == length:
                stride = max(1, int(n_bank * math.ceil(dhead / n_bank / n_mac) *
                             math.ceil(valid_channels[i + 1]) / length))
                for k in range(stride):
                    if (j * stride + k) >= len(cmd_score_wrgb[i + 1]):
                        break
                    total_cmd.append(cmd_score_wrgb[i + 1][j * stride + k])
            if not j == length:
                total_cmd += barrier

        # Head0: SoftMax, Head1: Score
        length = math.ceil(len(cmd_score_mac[i + 1]) / 16) if len(cmd_score_mac[i + 1]) > 0 else 1
        for j in range(0, length + 1):
            if not j == length:
                stride = 16
                for k in range(stride):
                    if (j * stride + k) >= len(cmd_score_mac[i + 1]):
                        break
                    total_cmd += cmd_score_mac[i + 1][j * stride + k]
            if not j == 0:
                mvsb_idx = j - 1
                if mvsb_idx < len(cmd_score_mvsb[i + 1]):
                    total_cmd += cmd_score_mvsb[i + 1][mvsb_idx]
            if j == 0:
                total_cmd += cmd_sfm[i]
            if not j == length:
                if j >= math.floor(length / 2):
                    half_len = max(1, math.ceil(length / 2))
                    stride = max(1, int(len(cmd_context_mvgb[i]) / half_len))
                    for k in range(stride):
                        offset = (j - math.floor(length / 2)) * stride + k
                        if offset >= len(cmd_context_mvgb[i]):
                            break
                        total_cmd.append(cmd_context_mvgb[i][int(offset)])
            if not j == length:
                total_cmd += barrier

        # Head0: Context, Head1: Softmax
        length = math.ceil(dhead / n_bank / n_mac)
        for j in range(0, length + 1):
            if not j == length and j < len(cmd_context_mac[i]):
                total_cmd += cmd_context_mac[i][j]
            if not j == 0 and (j - 1) < len(cmd_context_mvsb[i]):
                total_cmd += cmd_context_mvsb[i][j - 1]
            if j == 0:
                total_cmd += cmd_sfm[i + 1]
            if not j == length:
                if j >= math.floor(length / 2):
                    half_len = max(1, math.ceil(length / 2))
                    stride = max(1, int(len(cmd_context_mvgb[i + 1]) / half_len))
                    for k in range(stride):
                        offset = (j - math.floor(length / 2)) * stride + k
                        if offset >= len(cmd_context_mvgb[i + 1]):
                            break
                        total_cmd.append(cmd_context_mvgb[i + 1][int(offset)])
            if not j == length:
                total_cmd += barrier

        # Head1: Context
        length = math.ceil(dhead / n_bank / n_mac)
        for j in range(0, length + 1):
            if not j == length and j < len(cmd_context_mac[i]):
                total_cmd += cmd_context_mac[i][j]
            if not j == 0 and (j - 1) < len(cmd_context_mvsb[i]):
                total_cmd += cmd_context_mvsb[i][j - 1]
            if not j == length:
                total_cmd += barrier

    if num_itr % 2 != 0:
        i = num_itr - 1
        total_cmd += cmd_score_wrgb[i]
        total_cmd += barrier

        length = math.ceil(len(cmd_score_mac[i]) / 16) if len(cmd_score_mac[i]) > 0 else 1
        for j in range(0, length + 1):
            if not j == length:
                stride = 16
                for k in range(stride):
                    if (j * stride + k) >= len(cmd_score_mac[i]):
                        break
                    total_cmd += cmd_score_mac[i][j * stride + k]
            if not j == 0:
                mvsb_idx = j - 1
                if mvsb_idx < len(cmd_score_mvsb[i]):
                    total_cmd += cmd_score_mvsb[i][mvsb_idx]
            if not j == length:
                total_cmd += barrier

        total_cmd += cmd_sfm[i]
        total_cmd += cmd_context_mvgb[i]
        total_cmd += barrier

        length = math.ceil(dhead / n_bank / n_mac)
        for j in range(0, length + 1):
            if not j == length and j < len(cmd_context_mac[i]):
                total_cmd += cmd_context_mac[i][j]
            if not j == 0 and (j - 1) < len(cmd_context_mvsb[i]):
                total_cmd += cmd_context_mvsb[i][j - 1]
            if not j == length:
                total_cmd += barrier

    trace_file = open(trace_file_name, 'w')
    for cmd in total_cmd:
        trace_file.write(cmd + "\n")
    trace_file.close()

    return chunk_info, row_analysis


def main():
    global dhead, max_L, data_size, n_mac

    parser = argparse.ArgumentParser(
        description="ChunkKV sparse trace generator for AttAcc PIM simulator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-dh", "--dhead", type=int, default=128,
                        help="dhead, default=128")
    parser.add_argument("-nh", "--nhead", type=int, default=64,
                        help="Number of heads, default=64")
    parser.add_argument("-l", "--seqlen", type=int, default=2048,
                        help="Sequence length L, default=2048")
    parser.add_argument("-maxl", "--maxlen", type=int, default=4096,
                        help="maximum L, default=4096")
    parser.add_argument("-db", "--dbyte", type=int, default=2,
                        help="data type (B), default=2")
    parser.add_argument("-o", "--output", type=str, default="chunkkv.trace",
                        help="output path")
    parser.add_argument("-cl", "--chunk_length", type=int, default=20,
                        help="ChunkKV chunk length, default=20")
    parser.add_argument("-cr", "--compression_ratio", type=float, default=0.9,
                        help="ChunkKV compression ratio (0.9 = keep 10%%), default=0.9")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for chunk selection, default=42")

    args = parser.parse_args()

    dhead = args.dhead
    max_L = args.maxlen
    L = args.seqlen
    n_head_per_hbm = args.nhead

    data_size = args.dbyte
    n_mac = int(HBM_GS['col'] / data_size)

    print("=" * 65)
    print("  ChunkKV Sparse Trace Generator for AttAcc PIM")
    print("=" * 65)
    print(f"  Sequence Length (L): {L}")
    print(f"  Chunk Length: {args.chunk_length}")
    print(f"  Compression Ratio: {args.compression_ratio} (keep {(1-args.compression_ratio)*100:.0f}%)")

    chunk_info, row_analysis = run_attention(
        dhead, n_head_per_hbm, L, args.output,
        use_chunkkv=True,
        chunk_length=args.chunk_length,
        compression_ratio=args.compression_ratio)

    print(f"\n  --- ChunkKV Selection Results ---")
    print(f"  Total chunks: {chunk_info['total_chunks']}")
    print(f"  Selected chunks: {chunk_info['selected_chunks']}")
    print(f"  Total tokens (L): {chunk_info['total_L']}")
    print(f"  Effective tokens: {chunk_info['effective_L']}")
    print(f"  Token reduction: {chunk_info['compression_ratio']*100:.0f}%")
    print(f"  Selected chunk indices: {chunk_info['selected_chunk_indices']}")
    
    if row_analysis:
        print(f"\n  --- Row Activation Analysis (Key Insight!) ---")
        print(f"  Full KV row groups (n_idx):    {row_analysis['full_kv_n_idxs']}")
        print(f"  ChunkKV row groups (n_idx):    {row_analysis['chunkkv_n_idxs']}")
        print(f"  Row group reduction:           {row_analysis['n_idx_reduction_pct']:.1f}%")
        print(f"  Full KV row activations:       {row_analysis['full_kv_row_acts']}")
        print(f"  ChunkKV row activations:       {row_analysis['chunkkv_row_acts']}")
        print(f"  Row activation reduction:      {row_analysis['row_act_reduction_pct']:.1f}%")
        print(f"  Avg row utilization:           {row_analysis['avg_row_utilization']:.1f}%")
        print(f"\n  ⚠ Despite removing {chunk_info['compression_ratio']*100:.0f}% of tokens,")
        print(f"    row activations only reduced by {row_analysis['row_act_reduction_pct']:.1f}%!")
        print(f"    Bandwidth wasted: {100 - row_analysis['avg_row_utilization']:.1f}%")
    
    print(f"\n  Trace written to: {args.output}")
    print("=" * 65)


if __name__ == "__main__":
    main()
