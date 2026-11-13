# D1-G2R 실행 보고 — 9,252 대상 g2 입력 다시 만들기 (기존 TS 재사용)

대상 9,252개 중 **8,617개(93.1 %)를 사용할 수 있다**(`U_report.json`). 기준 구조 단계 코드 오류 20개 중 **3개를 복구했다**
(k = 3: KeyError 우회 1, 다시 뽑은 conformer가 통과 2). 나머지 17개는 alt conformer에서 일라이드 CH2 탄소가 ortho 받개의 O와
실제로 결합해(1.41–1.43 Å) 바뀌지 않은 Coley 규칙(고리 수 변화)으로 실패했다. 허용치 완화(D-5)로 **114개가 새로 들어왔다**:
특성 112개(engine gate 96, 쌍극자 합 16)와 토큰 2개(rxn 1007205, 1007289).

명령서: `docs/specs/D1G2R.md`. 브랜치 `g2r`. 새 g2 = `$G2R=/gpfs/tmp_cpu2/yeseo1ee/g2r`(TS 원본 `$G2R/jobs`, 모델 입력 `$G2R/lf`).
push한 HEAD: 12절 뒤에 적는다(맨 아래).

## 0. 단계별 수 (`U_funnel.csv`)

| 단계 | 빠짐 | 남음 |
|---|---|---|
| D1 비D″ 라벨 | — | 9,266 |
| short_forming 9, regio_ambiguous 5 (겹침 0) | 14 | 9,252 |
| cr2 기준 초과 > 20, 허용 위생 | 0 | 9,252 |
| g2 D-6: TS 없음, autodE 오류, 입체 불일치, 허수 진동 없음 | 599 | 8,653 |
| g2 D-4: TS 검사, 조각 분할 | 19 | 8,634 |
| g2 기준 구조 코드 오류 (D-3), 20 − k | 17 | 8,617 |
| L 진동수 마무리 | 0 | 8,617 |
| F 전역 182열 | 0 | 8,617 |
| T 원자 토큰 | 0 | **8,617** |

0-4절 기대(8,614 + k)와 같다(U-1).

## 1. 시작 조건 (`S0_start.json`)

1-1–1-8 모두 통과. 디스크 여유 104 GB(300 GB − `du`, compute node에서 mmlsquota 불가). `$G2` 539,013파일 11.64 GB,
`$CH3/lf` 156,181파일 10.98 GB, `$G2/jobs` 122,866파일 0.257 GB. 1-3: 작업 폴더의 단계 표지를 finalize 규칙으로 읽은 상태가
`G_status.csv` `marker`와 행마다 같다. T00322(`.fail_hess_no_imaginary_frequency`)는 세 단계가 모두 `.done_*`인 예외로 기록했다.

## C. TS 원본 복사 (`C_report.json`)

9,266개 폴더, 122,866파일, 256,677,455바이트를 `$G2R/jobs`로 복사. 파일 수·바이트·sha256 불일치 0, symlink 0(C-1, C-2 통과).
목록: `$G2R/x/C_manifest.csv.gz`.

## X. 대상 (`X_report.json`, `X_targets.csv`)

| 검사 | 결과 |
|---|---|
| X-1 비D″ 행 | 9,266 |
| X-2 short_forming / regio_ambiguous / 겹침 / 대상 | 9 / 5 / 0 / 9,252 |
| X-3 그룹 | reuse 8,614, repair 20, excluded_check 19, excluded_ts 599 |
| X-4 C_compare flags 불일치 | 0 |
| X-5 flag_async / ex_orig_gt20 | 209 / 8 |
| X-6 cr2 기준 초과 최대 / 위생 실패 허용·엄격 | 0.050 kcal/mol / 0 · 4 |

## R. 기준 구조 단계 코드 오류 20개 (`R_diagnosis.md`, `R_tests.json`, `R_status.csv`, `R_report.json`)

- 진단: 모두 RDKit이 거리로 결합을 정할 때 쌍극자 음이온 끝(일라이드 CH2⁻, 니트론 O⁻)과 ortho 받개 사이에 여분 결합을 보고,
  `AssignBondOrdersFromTemplate`의 `SanitizeMol`이 원자가를 거부한 것(19개: alt conformer, T10518: TS 단계에서 나서 키가 빠짐).
- 우회(5-2 범위): `stereo_utils.template_match`는 그 예외일 때만 원자가 검사 없이 sanitize, `make_reference.coley_port`는
  `KeyError`일 때만 `run_ts.py:238`과 같은 `su.dipole_backbone(manifest SMILES)`로 두 값을 채워 다시 부른다. Coley 판정·xtb·conformer
  수는 그대로.
- 검사: R-T1(mock + 실제 TS 조각 5개, 결과 같음, 우회 0회)과 R-T2(T10518 TS 조각: 원래 `atom # 1 C, 5` → 우회 통과, 여분 결합
  C–O 1.831 Å; T09757 CH2 기를 카보닐 C 쪽 1.95 Å로 옮긴 구조: 같은 예외 → 우회 통과; T10518 `coley_port` KeyError → 재호출 통과)
  모두 통과.
- 실행(반응당 한 번, 4코어): **ok 3**(T10518, T11058, T11070; 모두 alt_used = 1), **fail 17**(`ring count changed in alt (Coley drops
  such profiles)`). 실패 17개는 우회가 실제로 쓰였고 기록된 여분 결합이 C–O 1.41–1.43 Å(공유 결합 거리)이다. 즉 alt conformer가
  ortho 카보닐/니트로 O와 고리를 만들었고, 이것은 화학적 실패라 그대로 실패로 남긴다(5-4-3). T11058·T11070은 다시 뽑은 conformer가
  고리를 만들지 않아 우회 없이 통과했다(autodE conformer 생성에 seed가 없다).
- 5-5: TS 검사·분할 탈락 19개의 실패 검사 이름과 값은 `R_check_fails.csv`(포함하지 않음). 검사 이름은 `.fail_ts` 문장에서
  옮겼다: extra_imag_small 6, imag_mode_on_forming_bonds 5, forming_bond_present 4, partition 3, no_foreign_bond 2,
  not_product_like 2(한 반응에 둘 이상 있을 수 있음).

## L. 진동수 마무리 (`lf/L_report.json`, `L_check.json`, `L_repaired.csv`)

- L-1: deploy L-1·L-2·L-4가 reuse 8,614개에서 통과(진동수 실패 0).
- L-R: reuse 8,614개의 `ts.xyz`·`rel1.xyz`·`rel2.xyz`가 `$CH3/lf/g2lf_final`과 byte 단위로 같고, 최저 허수 진동수 차 최대
  7.3 × 10⁻⁷ cm⁻¹.
- L-K: 복구 3개 모두 `.done`, 허수 진동 1개(−260.9, −259.7, −248.9 cm⁻¹), 형성 결합 비율 0.98, 0.96, 0.97.

## M. 메타데이터 (`lf/M_report.json`, `M_check.json`)

- 복구 반응이 L `.done`에 도달해, deploy `meta_lf.cmd_run`의 L 전제 조건만 바꾼 `m_meta.py`로 만들었다(`meta_lf.build`는
  그대로; 기록된 결정).
- M-1: 레코드 8,617 = L `.done` 8,617, 전하 ≠ 0 없음. M-2: reuse 8,614개의 레코드와 `input_meta_lf.csv` 행이 `$CH3/lf`와 모두 같다.

## F. 전역 182열 (`lf/F_report.json`, `F_check.json`, `F_newly_passed.csv`, `F_tol/`)

- F-2: `site_g2r/sitecustomize.py`가 site_si를 그대로 돌린 뒤 `GATE_E_EH = 1e-4`, `TOL_DIP_AU = 1e-2`. 모든 F·T job의 시작 기록
  (`F_tol/*.json`)에서 job 프로세스, spawn worker, feat_lf식 자식 프로세스 셋 다 이 값과 Si 패치, 나머지 허용치 원래 값.
  새 허용치보다 작은 값으로 실패한 반응 0.
- F-1 열 이름·순서 = `$CH3/lf/X182_lf.parquet`. F-3 특성 실패 0 / 8,617.
- F-R: 예전에 통과한 8,502개의 182열 최대 |차| 7.3 × 10⁻⁷(`nu_imag`; `mode_share` 4.4 × 10⁻⁹, `mode_async` 2.3 × 10⁻⁹, 나머지
  열은 정확히 같다). L에서 진동수를 다시 계산했기 때문이다.
- F-N: 새로 통과 115 = 예전 탈락 112(engine gate 96, 쌍극자 합 16 중 rel1 5) + 복구 3. 예전 탈락 중 아직 탈락 0.
  rel1의 쌍극자 검산 값은 ext_features가 저장하지 않는다(통과했으므로 < 1e-2).
- F-D: |E_tblite − E_xtb| 최대 9.3 × 10⁻⁶ Eh(0.0059 kcal/mol), 중앙값 3.0 × 10⁻⁸, 99 % 1.2 × 10⁻⁶. 쌍극자 검산(fA/fB) 중앙값
  6.3 × 10⁻⁵, 99 % 3.5 × 10⁻⁴, 최대 1.4 × 10⁻³ a.u.

## T. 원자 토큰 (`lf/T_lf_check.json`, `T_check.json`)

- T-1: 원소 [1, 6, 7, 8, 9, 14, 16, 17, 35], d_in = 29, Q-a 실패 0, Q-b 실패 0, 확장 통과율 1.0, T-a 통과, gate 종료 코드 0.
- T-R: 예전에 사용 가능했던 8,500개의 토큰이 `$CH3/lf/fu2_lf/tokens`와 같다(정수 배열 정확히 같음, 실수 최대 |차| 7.3 × 10⁻⁷,
  같은 `nu_imag`가 glob에 들어 있다).
- T-N: rxn 1007205(쌍극자 검산 3.7 × 10⁻³ a.u.)와 1007289(2.0 × 10⁻³)가 통과해 토큰이 생겼다.

## U. 사용 가능 목록 (`U_usable.csv`, `U_funnel.csv`, `U_report.json`)

사용 가능 8,617 / 9,252. 빠진 635 = X 단계 618(D-6 599 + D-4 19) + R 17. sha256: `labels_lf.json` 3f2bf2cc…,
`input_meta_lf.csv` a3968398…, `X182_lf.parquet` 66c40316…, 토큰 manifest f424d12d….

## D. 기존 g2 삭제

(12절 뒤에 채운다.)

## 비용

| 단계 | core-h | 출처 |
|---|---|---|
| R | 5.9 | `R_report.json` core_h_total (시도 wall × 4코어) |
| L | 38.1 | `lf/L_report.json` budget core_h_freq_total (reuse) |
| F | 6.3 | `lf/F_report.json` F4 0.728 core-h / 1,000반응 × 8,617 |
| T | 약 2 | atoms 배열 작업당 약 0.5분 × 32코어 × 8 (로그) |
| C, X, M, U | < 0.1 | 파일 복사·읽기 |

합계 약 52 core-h(명령서 추정 31 core-h).

## 기록된 결정 (추천안)

1. 그룹과 1-3은 `G_status.csv` `marker` 기준. 작업 폴더에는 단계 표지만 있으므로 finalize 규칙으로 읽은 상태를 행마다 대조.
2. R: 작업 폴더의 `.fail_ref` 파일을 지움(명령서의 `.fail_ref_*`). 반응당 끝난 시도 한 번. 5-2 범위 판단은 범위 안.
3. M: 복구 반응이 L `.done`이므로 `m_meta.py`(deploy `meta_lf`의 L 전제 조건만 교체).
4. F-3은 F_report `F_fail_rate`(모든 실패) ≤ 1 %.
5. D-c는 Phase R이 바꾼 파일(`ref_result.json`, `.fail_ref`)을 빼고 대조, `ref_result.orig.json` sha256 = C_manifest의
   `ref_result.json`.
6. ex_orig_gt20은 `$D1PAIR_SCRATCH/P/rowattr.pkl.all.pkl`(build_rowattr.py, 원래 기준 구조).

## 한계

- TS는 재사용했다. autodE는 seed가 고정되지 않아 다시 찾으면 약 절반이 다른 TS가 된다(D1 검증 V3: 같은 TS 0.50).
- 복구한 반응(3개)은 기준 구조의 conformer 탐색을 이번에 처음 끝까지 돌렸다. 새 표본이므로 같은 코드를 다시 돌리면 다른
  결과가 날 수 있다(T11058·T11070은 예전에는 고리화한 alt를 만들었다).
- TS 검사·분할 탈락 19개와 TS가 없는 595개(TS 없음 487 + autodE 오류 108)는 배포 때도 예측할 수 없는 반응이다.
- 허용치 완화로 들어온 114개의 차이는 반올림 수준이다(F-D 최대 0.0059 kcal/mol, 쌍극자 최대 3.7 × 10⁻³ a.u.).
