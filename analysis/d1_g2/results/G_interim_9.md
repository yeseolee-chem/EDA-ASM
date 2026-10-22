# D1-G2 Phase G 중간 점검 9 (끝난 반응 9,000개)

판정: **통과** (기준은 이 블록 1000개: ok ≥ 70 %, 인프라+파이프라인 ≤ 2 %, .fail_order ≤ 1 %)

| | 이 블록 | 누적 |
|---|---|---|
| 반응 수 | 1,000 | 9,000 |
| ok 비율 | 85.8 % | 93.1 % |
| 인프라 실패 | 0.0 % | 0.0 % |
| 파이프라인 실패 | 1.0 % | 0.2 % |
| .fail_order | 0.0 % | 0.0 % |
| 원자 순서 다시 매김 (ok 중) | 0.0 % | 0.1 % |
| 코어시간 합 | 459.8 | 2,967.1 |

실패 이유 (이 블록):

- ts_autodE found no transition state: 100
- ts_autodE exception: 27
- ref_AtomValenceException: 9
- ts_TS check failed: 5
- ref_KeyError: 1
