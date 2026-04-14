# ChunkKV PIM 병목 분석 파이프라인 — 구현 완료 Walkthrough

## 목표
ChunkKV sparse attention을 AttAcc PIM 시뮬레이터에 naive하게 적용했을 때, **논리적 희소성이 물리적 Latency 개선으로 이어지지 않는 문제**를 정량적으로 증명하는 분석 파이프라인 구현.

## 생성된 파일

### 1. [gen_trace_chunkkv.py](file:///home/starksa/skkai_pim/attacc_simulator/ramulator2/trace_gen/gen_trace_chunkkv.py)
ChunkKV sparse 접근 패턴 트레이스 생성기.

- ChunkKV 알고리즘을 모사하여 랜덤 중요도 기반 청크 선택
- 선택된 청크의 **원래 메모리 주소를 유지** (물리 remapping 없음)
- PIM MACAB 특성 반영: 선택된 토큰이 포함된 Row 전체를 활성화
- Row Activation 분석 메트릭 내장 (n_idx group 분석)

### 2. [run_chunkkv_analysis.py](file:///home/starksa/skkai_pim/attacc_simulator/ramulator2/trace_gen/run_chunkkv_analysis.py)
Full KV vs ChunkKV 자동 비교 분석 스크립트.

- Full KV 트레이스 생성 → ramulator2 시뮬레이션 → ChunkKV 트레이스 생성 → 시뮬레이션
- **Ideal vs Actual Speedup** 비교 (논문 Problem Statement의 핵심)
- PIM Efficiency, BW Wasted, Row Reduction 자동 계산
- CSV 출력 지원

### 3. [analyze_row_utilization.py](file:///home/starksa/skkai_pim/attacc_simulator/ramulator2/trace_gen/analyze_row_utilization.py)
Row Utilization 심층 분석기.

- 트레이스 MAC 주소를 HBM3 계층(Ch/pCH/Rank/BG/Ba/Row/Col)으로 분해
- Per-row column 활용률 계산 및 히스토그램
- Full KV vs ChunkKV 비교 모드 지원

## 검증 결과

시뮬레이션 설정: `L=2048, dhead=128, nhead=64, chunk_length=20, HBM3_5.2Gbps_NPC`

### Raw Metrics

| Method | Tokens | Cycles | Latency(µs) | MAC Cmds | Unique Rows | Row Util(%) |
|--------|--------|--------|-------------|----------|-------------|-------------|
| Full KV | 2048 | 11,755 | 9.04 | 32,784 | 768 | 100.0% |
| ChunkKV (cr=0.5) | 1024 | 8,501 | 6.54 | 21,520 | 768 | 65.6% |
| ChunkKV (cr=0.7) | 614 | 6,477 | 4.98 | 13,328 | 768 | 40.6% |
| ChunkKV (cr=0.9) | 204 | 3,071 | 2.36 | 4,624 | 512 | 21.1% |

### Ideal vs Actual Speedup (핵심 Problem Statement 데이터)

| Token Reduction | Ideal Speedup | Actual Speedup | PIM Efficiency | BW Wasted |
|---|---|---|---|---|
| 50% | 2.0× | 1.38× | **69.1%** | 34.4% |
| 70% | 3.3× | 1.81× | **54.4%** | 59.4% |
| 90% | 10.0× | 3.83× | **38.3%** | 78.9% |

> [!IMPORTANT]
> **핵심 발견**: 토큰을 90% 제거해도 이상적 10× speedup 대비 실제 3.83×만 달성 (효율 38.3%). 
> Compression ratio가 높아질수록 PIM 효율이 급격히 감소하며, 활성화된 Row 내 대역폭의 최대 78.9%가 낭비됩니다.

## 사용법

```bash
# 개별 ChunkKV 트레이스 생성
python3 trace_gen/gen_trace_chunkkv.py --seqlen 2048 --compression_ratio 0.9 --chunk_length 20 \
    --dhead 128 --nhead 64 --output chunkkv.trace

# Full KV vs ChunkKV 전체 분석 (자동)
python3 trace_gen/run_chunkkv_analysis.py --seqlen 2048 --dhead 128 --nhead 64 \
    --compression_ratios 0.5 0.7 0.9 --chunk_length 20

# Row Utilization 심층 분석
python3 trace_gen/analyze_row_utilization.py --trace_full fullkv.trace --trace_chunkkv chunkkv.trace
```

## 결론 (논문 활용)

이 결과는 다음을 증명합니다:
1. **논리적 희소성 ≠ 물리적 성능 개선**: 소프트웨어의 토큰 제거가 PIM 하드웨어 Latency에 비례적으로 반영되지 않음
2. **Row 단위 접근의 제약**: 흩어진 청크를 읽기 위해 거의 동일한 수의 Row를 활성화해야 함
3. **대역폭 낭비**: 활성화된 Row 내 유효 데이터 비율이 cr=0.9에서 21.1%까지 하락
4. **해결 방향**: 물리적 데이터 재배치(Sparse Packing)가 필수 → 이것이 논문의 제안 방법론의 동기
