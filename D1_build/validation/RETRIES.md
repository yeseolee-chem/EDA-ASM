# 재실행 기록 (VALIDATION_SPEC §2, 파일럿 SPEC §2: 실패를 다시 돌릴 때는 사유를 보고서에 남긴다)

2026-09-30 06:55 KST, 사용자 결정 (c). 1차 worker pool은 job 996236이다. 사전 등록 파일 4개는 수정하지 않았다.

| task | 1차 결과 | 원인 (로그 확인) | 분류 | 조치 |
|---|---|---|---|---|
| V7_J03 | `$Q/failed` (rc=1, 4.4 h) | ORCA relaxed scan이 100점 중 89점째에서 `orca_prop_mpi` "error termination in PROPERTIES"로 종료. 남은 11점이 없어 minimax 경로가 이어지지 않음 | 인프라 (MPI) | `$Q/failed/V7_J03` 삭제 → scan 처음부터 재실행 |
| V3_0020_r2 | `.fail_ts` "autodE exception: CouldNotGetProperty: Could not get energy" | `transition_states/neb/4/4_0_orca.out`: `orca_leanscf_mpi` "error termination in LEANSCF" | 인프라 (MPI) | `.fail_ts`와 `$Q/done/` 표시 삭제 → autodE checkpoint에서 재개 |
| V6b_J08_r3 | `.fail_ts` "autodE exception: CouldNotGetProperty: Could not find Hessian file" | `ts_guess_vvEPtl_template_2-3_7-8_hess_orca.out`: `orca_scfresp_mpi` "error termination in SCF RESPONSE" | 인프라 (MPI) | 같음 |
| V4_0047 | `.fail_ref` "CouldNotGetProperty: Could not get energy" | `ref/dipole_alt_hess_orca.out`: `orca_startup_mpi` "error termination in Startup" | 인프라 (MPI) | `.fail_ref`와 `$Q/done/` 표시 삭제 → ref 단계 재실행 |
| V6b_J09_r2 | `$Q/failed` (rc=1) | `assemble.py`가 `TypeError: dict() got multiple values for keyword argument 'sum_mismatch_promoted'`로 종료. 파일럿 번들에 원래 있던 버그로, `eda_sum_mismatch`에서 `ok`로 승격하는 라벨은 모두 이 단계에서 멈춘다 | 파이프라인 버그 | `assemble.py` 수정(키를 한 번만 넣음, 판정 기준은 그대로) → `$Q/failed` 삭제 → label 단계만 재실행 (ts, ref, inputs, sp는 이미 완료) |

**재실행하지 않은 것:**
- V4_0500: `.fail_ts` "ValueError: RMSD must be computed between atom lists of the same length: 31 =/= 0".
- ORCA 출력 18개는 autodE 종료 규칙상 모두 정상이다. autodE 내부 오류(원자 0개 conformer)이므로 인프라 실패가 아니다. 결과에 그대로 둔다.

**MPI 오류의 분포:**
- 노드는 n107, n108, n115로 흩어져 있어, 특정 노드 하나의 문제로 보이지 않는다.
- 검증 전체 ORCA 출력 중 MPI 오류 종료는 6건이다.
- 환경 변수는 CLAUDE.md와 같다(`OMPI_MCA_pml=ob1`, `hcoll` off, `UCX_TLS=tcp,self,sm`).
