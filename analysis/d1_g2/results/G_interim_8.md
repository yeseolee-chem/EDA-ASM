# D1-G2 Phase G 중간 점검 8 (끝난 반응 8,000개)

판정: **통과** (기준은 이 블록 1000개: ok ≥ 70 %, 인프라+파이프라인 ≤ 2 %, .fail_order ≤ 1 %)

| | 이 블록 | 누적 |
|---|---|---|
| 반응 수 | 1,000 | 8,000 |
| ok 비율 | 92.1 % | 94.0 % |
| 인프라 실패 | 0.0 % | 0.0 % |
| 파이프라인 실패 | 0.4 % | 0.1 % |
| .fail_order | 0.0 % | 0.0 % |
| 원자 순서 다시 매김 (ok 중) | 0.0 % | 0.1 % |
| 코어시간 합 | 397.2 | 2,507.2 |

실패 이유 (이 블록):

- ts_autodE found no transition state: 50
- ts_autodE exception: 21
- ref_AtomValenceException: 4
- stereo_graph_mismatch: 2
- ts_TS check failed: 2
