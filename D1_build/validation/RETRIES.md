# 재실행 기록 (VALIDATION_SPEC §2, 파일럿 SPEC §2: 실패를 다시 돌릴 때는 사유를 보고서에 남긴다)

- 2026-09-30 06:55 KST, 사용자 결정 (c).
- 1차 worker pool은 job 996236, 재실행 worker는 996555, 보고서는 996556이다.
- 사전 등록 파일 4개는 수정하지 않았다.
- 원래의 실패 표시는 `scratch/queue/retry_backup_20260930/`에 보존했다.

| task | 1차 결과 | 원인 (로그 확인) | 분류 | 조치 | 2차 결과 |
|---|---|---|---|---|---|
| V7_J03 | `$Q/failed` (rc=1, 4.4 h) | ORCA relaxed scan이 100점 중 89점째에서 `orca_prop_mpi` "error termination in PROPERTIES"로 종료. 남은 11점이 없어 minimax 경로가 이어지지 않음 | ORCA 하위 프로그램 비정상 종료 (원인 미확인) | scan 처음부터 재실행 | **같은 89점째에서 다시 종료** (rc=1, 4.4 h). stderr: `mpirun ... process [[33845,1],0] Exit code: 1`. 같은 구조에서 재현되므로 우연한 인프라 오류가 아님. A10의 J03 부분은 판정 불가 |
| V3_0020_r2 | `.fail_ts` "autodE exception: CouldNotGetProperty: Could not get energy" | NEB 이미지 `neb/4/4_0_orca.out`: `orca_leanscf_mpi` "error termination in LEANSCF". 직전 SCF는 orbital gradient 기준으로 수렴 | ORCA 하위 프로그램 비정상 종료 (원인 미확인) | `.fail_ts`와 done 표시 삭제 → autodE checkpoint에서 재개 | **같은 오류로 다시 실패** (다른 NEB 이미지 `neb/2/2_1`, LEANSCF). 더 재시도하지 않고 실패로 둠 |
| V6b_J08_r3 | `.fail_ts` "autodE exception: CouldNotGetProperty: Could not find Hessian file" | `ts_guess_vvEPtl_template_2-3_7-8_hess_orca.out`: `orca_scfresp_mpi` "error termination in SCF RESPONSE" (SCF는 10 cycle에 수렴) | ORCA 하위 프로그램 비정상 종료 (원인 미확인) | 같음 | **성공** (label까지) |
| V4_0047 | `.fail_ref` "CouldNotGetProperty: Could not get energy" | `ref/dipole_alt_hess_orca.out`: `orca_startup_mpi` "error termination in Startup" | ORCA 하위 프로그램 비정상 종료 (원인 미확인) | `.fail_ref`와 done 표시 삭제 → ref 단계 재실행 | **성공** (label까지) |
| V6b_J09_r2 | `$Q/failed` (rc=1) | `assemble.py`가 `TypeError: dict() got multiple values for keyword argument 'sum_mismatch_promoted'`로 종료. 파일럿 번들에 원래 있던 버그로, `eda_sum_mismatch`에서 `ok`로 승격하는 라벨은 모두 이 단계에서 멈춘다 | 파이프라인 버그 | `assemble.py` 수정(commit d573111e, 키를 한 번만 넣음, 판정 기준은 그대로) → label 단계만 재실행 | **성공**: 라벨 정상 생성으로 수정 확인 |

**재실행하지 않은 것: V4_0500**
- `.fail_ts` "ValueError: RMSD must be computed between atom lists of the same length: 31 =/= 0".
- 이 job의 ORCA 출력 18개는 autodE 종료 규칙상 모두 정상이다. 원자가 0개인 conformer를 비교한 autodE 내부 오류다.

**분류에 대한 정정**
- 처음에는 이 네 건을 "MPI 오류"로 기록했다. 실제로 확인된 것은 ORCA의 MPI 하위 프로그램이 "error termination"으로 끝났다는 사실뿐이다.
- autodE는 ORCA의 stderr를 남기지 않아 원인을 확인할 수 없다.
- V7은 같은 scan 지점에서, V3_0020_r2는 같은 계산의 NEB에서 반복됐다. 따라서 적어도 이 두 건은 특정 계산에서 재현되는 ORCA 오류다.
- 노드는 n107, n108, n109, n115에 흩어져 있다. 환경 변수는 CLAUDE.md와 같다(`OMPI_MCA_pml=ob1`, `hcoll` off, `UCX_TLS=tcp,self,sm`).
