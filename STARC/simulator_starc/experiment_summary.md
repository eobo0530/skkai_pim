# STARC "Remapping Wall" 실험 설계 및 구현 내역 요약

본 문서는 Reasoning LLM의 장기 문맥 환경에서 PIM 기반 최적화 기법(STARC)의 한계를 검증하기 위해 수행된 시뮬레이터 환경 구축 및 실험 설계(Methodology)를 정리합니다.

## 1. 실험 목표 및 가설
* **목표**: STARC의 K-means 클러스터링 및 PIM 행 입도(Row-granularity) 물리적 재배치(Remapping) 과정에서 발생하는 오버헤드를 개별적으로 측정하고 병목을 확인.
* **가설 ("Remapping Wall")**: 문맥 길이($L$)가 커짐에 따라 클러스터링($O(L^2)$)과 재배치($O(L)$) 비용이 비선형적으로 증가하여, 어느 시점(Inversion Point)부터는 희소성(Sparsity)으로 인한 연산 가속 이득을 완전히 상쇄하고 Dense 연산보다 느려질 것이다.

## 2. 실험 환경 구성 (Configuration)
1. **대상 모델**: `DeepSeek-R1-Distill-Qwen-7B`
    * 파라미터 구성 (`src/config.py`): 28 Layers, 3584 Hidden Dim, 28 Q-Heads, 4 KV-Heads (GQA Ratio=7).
2. **동적 프루닝(RPC) 정책**: 
    * Pruning Period ($P$) = 1024
    * Compression Ratio ($C$) = 4 (주기마다 하위 75% 토큰 삭제, 256개 유지)
    * Recent Window ($R$) = 32
3. **시스템 파라미터**: Batch Size = 1 (단일 요청 Latency 측정)

## 3. 레이턴시(Latency) 분리 측정 기법 설계
시뮬레이터 내부에서 전체 시간 중 연산, 클러스터링, 복사 오버헤드를 분리 측정하기 위해 총 4가지 형태의 실행 모드(`cluster_mode`)를 구축했습니다.

1. **`dense`**: 최적화 미적용 베이스라인.
2. **`sparse`**: 프루닝된 상태(`kv_budget`)에서 오로지 어텐션 연산(GEMV) 시간만 측정. (클러스터링/복사 트레이스 제거)
3. **`sparse_cluster`**: 프루닝 연산 + **클러스터링 연산($O(L^2)$)** 시간 측정. 물리적 데이터 이동 트레이스는 방지됨.
4. **`sparse_full`**: 프루닝 연산 + 클러스터링 연산 + **물리적 데이터 복사 연산($O(L)$)** 까지 포함된 전체 STARC 파이프라인.

이를 통해 다음과 같이 레이턴시를 도출했습니다.
* $T_{comp}$ = `sparse`
* $T_{clustering}$ = `sparse_cluster` - `sparse`
* $T_{copy}$ = `sparse_full` - `sparse_cluster`

## 4. 코드 구현 상세 내역
* **Frontend 파라미터 확장**: `main.py`와 `run_experiment.py`에 `--cluster_mode` 인자를 추가하여 4가지 모드를 자동화 실행.
* **Trace Generator 분기 로직**: `pim_ramulator_src/trace_gen/gen_trace_attacc_bank.py` 파일의 트레이스 삽입부를 분리.
    * 기존 `add_cluster` 옵션 외에 `add_cluster_only` 옵션을 신설하여, 클러스터링 트레이스 명령어(`PIM_MAC_AB`, `PIM_MV_SB` 등)만 삽입되고 재배치 명령어가 제외되도록 스케줄링 게이트(`if add_cluster or add_cluster_only:`)를 수정함.
* **캐시 충돌 방지**: 실행마다 `ramulator.out` 및 `output.csv`를 정리하여 각 모드가 독립적으로 평가되도록 프로세스를 수정함.

## 5. 실행 중 특이사항 (64K 이상 환경)
* **문제점**: 64K 문맥 길이 시뮬레이션 중 `cluster_similar` 함수의 트레이스 생성 로직이 $O(L^2)$ 의 메모리를 점유하여 (약 1.3억 개 문자열, ~10GB), WSL 시스템 환경 내에서 Linux OOM(Out of Memory) Killer에 의해 프로세스가 강제 종료되는 현상 발생.
* **조치**: 시뮬레이터 자체의 메모리 병목임을 확인하고, 64K 이상의 시뮬레이션 실행은 취소함. 대신 1K~32K 구간의 실측 데이터를 통해 이미 "Remapping Wall" 가설이 입증되었으므로 해당 실측 데이터까지만 확정하여 최종 결과를 작성함.
