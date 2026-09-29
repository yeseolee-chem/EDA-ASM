# D1 검증 실행 계획 (VALIDATION_SPEC 기준)

작성 2026-09-29 · 기준 문서 [`VALIDATION_SPEC.md`](VALIDATION_SPEC.md)

**현재 상태:** 코드 작성만 끝났습니다. **아직 아무것도 실행하지 않았습니다** (sbatch 제출 0건, 단위 테스트도 미실행). 아래 1–4단계는 각각 사용자 승인 후에 진행합니다.

---

## 1. 실행 순서

모든 명령은 `D1_build/validation/`에서 로그인 노드 one-shot으로 실행합니다. 계산은 전부 sbatch(48 h)로 돌고, 각 submit 스크립트는 제출 전에 큐의 task 수(`squeue -r`)를 확인합니다.

| 단계 | 명령 | 내용 | 제출 task | 예상 |
|---|---|---|---|---|
| 1. Phase F | `bash submit_phaseF.sh` | 단위 테스트(`tests/test_units.py`) → 통과 시 S0를 `config_val.yaml`로 재실행 (새 scratch) | 2 | 테스트 수 분 · S0 약 6분 (파일럿 실측 5.5분) |
| 2. 표본·task 목록 | `bash submit_make_tasks.sh` (V6a는 commit된 `v6a_manifest.csv` 사용) | `val_d0_set.csv`, `val_manifest.csv`, `tasks.csv` 생성, 파일럿 작은 파일을 `scratch/pilot_copy/`로 복사 | 1 | 수 분 |
| 3. 사전 등록 | `git add config_val.yaml val_d0_set.csv tasks.csv val_manifest.csv` → commit → push | 이 커밋 해시가 판정 기준의 기록 | — | — |
| 4. Phase V | `bash submit_validation.sh` | worker 10개(array `0-9%10`) + 최종 보고 job 1개(afterany) | 11 | SPEC 추정 약 1,100 core-h, 벽시계 15–25 h |
| 5. 중간 점검 | 자동 | 우선순위 0–2 task가 모두 끝나면 worker 하나가 `interim_report.md`를 쓰고, STOP 조건이면 `scratch/queue/STOP` | — | — |
| 6. 최종 | 자동 → 확인 후 커밋·보고 | `results/VALIDATION_REPORT.md` + CSV + `geometries/` | — | — |

- `submit_validation.sh`는 사전 등록 파일 4개가 커밋돼 있고 수정되지 않았을 때만 제출합니다.
- worker가 48 h wall에 걸리면 `submit_validation.sh`를 다시 실행합니다. 죽은 worker의 claim은 새 worker가 인수합니다(15분 경과 + 소유 job이 squeue에 없을 때).
- 진행 확인은 one-shot으로만 합니다: `ls scratch/queue/done | wc -l`, `ls scratch/queue/failed`, `cat scratch/interim_report.md`.

## 2. task 구성 (V6a 포함 시 148개)

| 우선순위 | task | 개수 | 형태 |
|---|---|---|---|
| 0 | V1a (J08, J11 SP 재실행), V1b (eda nprocs 4), **V8**, **V9** | 2 + 2 + 1 + 1 | 스크립트 |
| 1 | V2a EnGrad L0/L1/L2 (반응당 1 task, 3수준) | 24 | 스크립트 |
| 2 | V5 D0_replay | 24 | 체인 |
| 3 | V2b-L0 (+ V2c refopt 단계), V1c (L0 OptTS 2회차) | 24 + 2 | 체인 |
| 4 | V7 (J03 2D scan), V6c (J07, F1) | 1 + 1 | 스크립트 / 체인 |
| 5 | V2b-L2 | 8 | 체인 |
| 6 | V3 D0_control × 3회 | 24 | 체인 |
| 7 | V4 나머지 D0_control | 16 | 체인 |
| 8 | V6a 새 D1 10행 | 10 | 체인 (xlsx 필요) |
| 9 | V6b J00/J04/J08/J09 × 2회 | 8 | 체인 |

체인 = 파일럿 단계 `ts → ref → inputs → sp → label` (+ `refopt`). V1c는 `ts`만 돕니다.

## 3. 파일

| 파일 | 역할 |
|---|---|
| `config_val.yaml` | 파일럿 config + scratch 변경 + F1 gate + `val`(실행 설정) + **`prereg`(§7 판정 기준, 사전 등록)** |
| `make_tasks.py` | §5 표본 추출, manifest, task 목록, 파일럿 파일 복사 |
| `worker.sh` / `run_task.sh` | worker pool (claim·heartbeat·takeover는 `smd_relabel_worker.sh` v5 함수 그대로) / task 1개 실행 |
| `val_determinism.py` | V1a, V1b |
| `val_grad.py` | V2a |
| `val_scan_j03.py` | V7 (2D relaxed scan, minimax 경로, 필요하면 OptTS 시도) |
| `analysis/negative_controls.py` | V8 (N1–N8) |
| `analysis/gate_audit_d0.py` | V9 (D0 5,265건 TS gate) |
| `analysis/rmsd.py`, `analysis/stats.py` | §6.4 RMSD, §6.5 통계 |
| `analysis/validation_report.py` | 중간 점검(`--interim`), 최종 보고서(`--final`), §8 결정표 |
| `tests/test_units.py` | Phase F 완료 조건 테스트 |
| `../d1_autode_pilot/orca_direct.py` | [신규] L0/L1/L2 헤더, ORCA 실행, `.hess`/`.engrad` 파서, 출력 수준 확인 |
| `../d1_autode_pilot/refopt.py` | [신규] V2c `refopt` 단계 |

**Phase F 수정 (파일럿 번들, 모두 config로 끄고 켤 수 있음):**
- **F1** `run_ts.imag_gate`: `gates.extra_imag_tol_cm`이 없으면 파일럿 규칙(허수 정확히 1개), 있으면 autodE 규칙.
- **F2** `report.ref_corrected`: 기본값은 꺼짐(파일럿 재현). `config_val`에서 켬.
- **F3** `stereo_utils`: RDKit 경고 억제. `D1P_RDKIT_WARNINGS=1`이면 다시 켬.
- **F4** `assemble.sp_coreh`: SP별 core-h 기록.
- **F5** `report.copy_geometries`: job별 구조 파일을 결과에 복사.
- **F6** `run_ts.run_reopt` + `make_reference`: `REPLAY_LIKE` 종류 처리, `refopt.py`.
- **추가 수정 두 가지:**
  - `pilot_common.meta_rxn_id`: 검증 job id(`V5_0020` 등)는 숫자가 아니라서, 파일럿의 rxn_id 공식이 `inputs` 단계에서 멈췄을 것입니다. manifest 열 `meta_rxn_id`로 대체합니다.
  - `pilot_common.ade_name`: V6c용입니다.

## 4. SPEC과 다르거나 SPEC에 없는 결정 — 확인 부탁드립니다

1. **L0 헤더의 `modify_internal` 블록 (가장 중요).**
   - 파일럿이 실제로 쓴 autodE OptTS 입력에는 TS마다 형성 결합을 내부 좌표로 추가하는 `%geom modify_internal { B i j A } end end` 블록이 두 개 있습니다. SPEC §6.3의 L0 글자 목록에는 이 블록이 없습니다.
   - SPEC 그대로 비교하면 항상 STOP합니다.
   - 구현한 방식:
     - 비교에서 이 블록만 제외합니다.
     - L0 입력에는 autodE처럼 그 TS의 형성 결합으로 블록을 다시 만들어 넣습니다(`val.l0_modify_internal: true`).
     - 파일럿 J10 결합으로 만들면 파일럿 헤더와 byte 단위로 같아야 한다는 단위 테스트를 넣었습니다.
2. **V6a:** 해결됨. 사용자가 같은 명령으로 로컬에서 만든 `v6a_manifest.csv`를 commit했습니다(§4b F-E).
   - `submit_make_tasks.sh`는 xlsx 인자 없이 실행합니다. xlsx를 주면 이 파일을 다시 생성해 덮어씁니다.
3. **V8, V9 우선순위.** SPEC 표에 없어서 0으로 두었습니다(QM이 없고, A7은 STOP 기준이기 때문).
   - 그래서 중간 점검에 A7, A8을 넣었고, **A7 FAIL도 STOP**으로 처리합니다. §8 첫 행과는 맞지만, §8의 중간 점검 문구는 A6과 A3만 언급합니다.
4. **V2b에서 port 진단 생략** (`val.reopt_port_diagnostic: false`). 기준 규칙 재현은 V5가 판정하므로 DFT 비용만 줄였습니다.
5. **표본 추출의 세부 규칙** (SPEC이 정하지 않은 부분):
   - pool을 rxn_id 순으로 정렬합니다.
   - strata는 S1→S4 순서로 뽑습니다.
   - 24개 추출과 부분집합 추출에 `random.Random(20260930)`을 따로 씁니다.
   - 강제 포함한 rxn 20과 105가 예상 stratum에 없으면 make_tasks가 멈춥니다.
6. **A1 계산 방식.**
   - 파일럿 J07은 V6c 결과(F1 적용)로 바꿔 셉니다.
   - J03은 V7 판정으로 분류합니다.
   - V6a에서 나온 "TS 없음"은 scan으로 확인되지 않았으므로 `no_ts_unconfirmed`로 두고, A1 조건(화학적 원인 또는 gate)을 충족하지 않는 것으로 처리합니다.
7. **A5 표본 수.** 성공한 job만으로 계산하고 n을 표시합니다. 최소 n 기준은 따로 두지 않았습니다. A3만 SPEC대로 24/24를 요구합니다.
8. **A6(c) 비교 방식.** 원자 순서가 같으므로 Kabsch 정렬 후 원자별 최대 편차로 비교하고, 정렬 전 차이도 함께 기록합니다.
9. **worker.**
   - `smd_relabel_worker.sh` v5에서 claim·heartbeat·takeover 함수를 그대로 가져왔습니다.
   - successor 자동 제출과 bad-node 계보는 가져오지 않았습니다(SPEC §6.2: wall에 걸리면 수동 재제출).
   - 48 h wall까지 8 h 미만이 남으면 새 task를 잡지 않습니다.
10. **읽기 전용 준수.** 파일럿 scratch는 읽기만 합니다.
    - 필요한 파일은 `scratch/pilot_copy/`(make_tasks), `v1/<task>/pilot_sp`, `v8/J12`, `v7/J03`, `jobs/V6c_J07/ts`(V6c)로 복사해서 씁니다.

## 4b. FIX_348bc3e 적용 (2026-09-29)

| id | 내용 | 파일 |
|---|---|---|
| F-A | 거리 gate를 D0와 대칭으로. `gates.no_bond_A`가 있으면 `not_product_like`(긴 쪽 ≥ 1.6 Å)와 `forming_bond_present`(짧은 쪽 < 3.3 Å, D0 `NO_BOND_A`)를 hard로, 1.6 Å 미만 결합은 `flag_short_forming`으로만 표시. 없으면 파일럿 규칙(`forming_in_range`). N4 검출 조건, V9·A8(`no_bond_ids` 3400, 5783)도 함께 변경 | `run_ts.py`, `config_val.yaml`, `negative_controls.py`, `gate_audit_d0.py`, `validation_report.py` |
| F-B | `level_ok`에 양성 대조 추가: L0/L1은 `rijcosx_on`이 True여야 함. V2a에서 L0/L1 `rijcosx_on` False 또는 VWN 문자열 없음이면 `result_error.json` + queue STOP | `orca_direct.py`, `val_grad.py` |
| F-C | `result.json`은 성공 시에만. 오류는 `result_error.json`(재실행 시 덮어씀). V7의 "OptTS 미수렴", "gate 실패"는 과학적 결과로 `result.json` | `val_determinism.py`, `val_grad.py`, `val_scan_j03.py` |
| F-D | interim A3: 빠진 V5 job에 `.fail_*`가 없으면 FAIL이 아니라 incomplete(`passed=None`). STOP은 쓰되 사유를 "incomplete"로 기록 | `validation_report.py` |
| F-E | `v6a_manifest.csv` commit (사용자가 로컬에서 생성: `sample_pilot.py --seed 20260930 --n 10 --controls 0 --exclude-ts pilot_manifest.csv --id-prefix V6a_`). 10행, 파일럿 D1과 같은 17열, 패널당 2행, 파일럿 ts_id와 겹침 0건 확인. `make_tasks.py`가 읽으므로 xlsx 불필요 | `v6a_manifest.csv` |

**지시와 다르게 구현한 한 가지 (F-B):** 지시문의 `cosx_seen = rijcosx == "on" or "COSX" in ri_lines`는 NORI 출력에 `RIJ-COSX (...).... off` 줄이 찍히면 COSX를 본 것으로 처리해 L2를 잘못 FAIL시킵니다. 그래서 COSX가 들어간 줄은 값이 `on`으로 끝날 때만 셉니다. 합성 `off` 줄로 확인하는 테스트도 넣었습니다.

**D-1 = allowed + engine_abnormal 가드 (결정 2026-09-29).** 가드는 ORCA 출력에 autodE 1.4.5 종료 규칙을 적용합니다(`prereg.A1.no_ts_unconfirmed_policy: allowed`, `engine_abnormal_is_pipeline: true`). 근거:
- Coley 2023은 실패를 스캔 없이 제외했다(D0 대칭).
- (a)는 결함 없는 파이프라인도 65–73% 확률로 FAIL시킨다.
- (c)의 relaxed scan은 TS의 부재를 증명하지 못한다.
- 본실험 no-TS 표본 스캔 감사는 별도 과제로 한다.

구현: `validation_report.engine_scan`이 no-TS 행의 autodE `ts/` 트리를 읽기만 합니다.
- `*_orca\d*.out` 중 autodE가 비정상으로 볼 출력이 있으면 `engine_abnormal`로 분류하고, pipeline으로 셉니다. 허용 실패가 아닙니다.
- xtb 출력은 보고만 합니다.

## 5. 코드 작성 중 확인한 사실

- **F1 근거:** autodE 1.4.5 `transition_state.py` 237행이 `if all([freq > -50 for freq in self.imaginary_frequencies[1:]])` 입니다.
- **F2 수치:** J10 port `dG_alt_minus_orig` = −4.535입니다.
  - 1.942 − (−4.535) = 6.477이고, G_act 6.160과의 차이는 +0.317입니다(테스트 기대값 +0.32 ± 0.01).
  - J04도 −5.02(dG), −5.51(dE)입니다. 즉 **alt가 원래 반응물보다 낮습니다.**
- **MAE_ref:** SUMMARY.md rev 3 표와 일치합니다(disp는 ESPLEY46 1.25). 테스트에서 다시 대조합니다.
- **J10 `.hess`:** 첫 허수가 −186.7488 cm⁻¹입니다(§6.3 파서 검증 대상).
- **`TS_imag_mode.xyz`:** 40 frame의 애니메이션입니다.
  - frame 10이 TS이고, 원자 순서는 TS 파일과 같습니다.
  - 모드 벡터는 frame 0에서 가장 먼 frame에서 frame 0을 뺀 값으로 구합니다.
- **autodE checkpoint 파일 이름:** 반응 이름의 해시입니다(`Reaction.__str__`). 그래서 V6c는 `ade_name=J07`로 복사본의 checkpoint를 다시 읽습니다.
- **`SUMMARY.md` 위치:** SPEC이 가리키는 `analysis/espley_xtb_repro/results/SUMMARY.md`에 있습니다.

## 6. 위험과 한계

- 코드는 아직 한 번도 실행되지 않았습니다. 1단계(단위 테스트, S0)가 첫 검증입니다.
- ORCA 2D scan의 `relaxscanact.dat` 열 형식은 실물로 확인하지 못했습니다. 형식이 다르면 `scan.out`의 단계별 값으로 대신 읽습니다.
- L2(NORI, DefGrid3, TightSCF) OptTS+Freq의 비용은 실측이 없습니다. 가장 긴 task가 될 가능성이 큽니다.
- SPEC §10의 한계는 그대로 남습니다: 35원자 이하 표본, SMD 이산화 차이는 ORCA 안에서 격리 불가, D0 진동수 부재.
