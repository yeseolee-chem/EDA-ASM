# DEPLOY-V2 (MLTS) — launcher 일시정지 해제 전 감사 (2026-11-05)

- 코드 `analysis/mlts/verify_data.py`(새 파일, commit 안 함). 수치 전체는 `results/AUDIT_report.json`, 작업 파일은 `$MLT/audit/`.
- 계산 job 4개: 1062847(cpu2 64코어, ORCA 100회), 1062848·1062857(누수; 1062857은 입체 판정을 더한 재실행), 1062849(GPU).
- 모델: `FTpilot_AIMNet2_0`은 아직 학습 중이다(1062600, 손대지 않음). 그래서 **best checkpoint** 사본을 썼다.
  - epoch 112, 11:06 저장, sha256 583416a8…
  - 사본을 만든 시점에 학습은 117–118 epoch까지 진행돼 있었다.
- launcher는 계속 일시정지 상태다(`PIPELINE_PAUSE` 그대로).

## 요약

| 검사 | 결과 | 판정 |
|---|---|---|
| ① 데이터 출처 일치 | 네 출처 모두 범함수·기저·분산·용매·grid가 라벨과 같다. 재계산한 힘 MAE는 0.0005–0.0023 kcal/mol/Å | **네 출처 모두 유지**(제외 없음) |
| ② 누수(정체성) | 입체를 뺀 기준으로 겹침 **1,371구조**(train 50, valid 1,321). 입체를 넣으면 train 31(+판정 불가 19), valid 0(+11) | **0이 아님** → `$MLT/audit/leak_remove.json`. 제거 후 재학습 필요 |
| ③ D3 일치 | 라벨 TS 20개에서 \|ΔE\| 최대 4.1e-8 Eh, \|ΔF\| 최대 7.5e-9 Eh/bohr | **통과**(기준 1e-6) |

## ① 데이터 출처 일치

### 수준 표

수준은 출력의 입력 echo와 출력에 찍힌 설정값에서 읽었다(48파일). 기준은 gen_dft G1 EnGrad, 즉 `mlt_common.orca_grad_input`이다.

| 출처 | 프로그램 | 범함수 | 기저 | 분산 | 용매 | grid | SCF · RIJCOSX | 근거 |
|---|---|---|---|---|---|---|---|---|
| S1 D1 G + Phase I | ORCA 6.1.1. G1·G3는 직렬, G2·G3L·Part B·V는 8 MPI | B3LYP (B88 + LYP, VWN-5, HF 0.20) | def2-SVP + def2/J | D3BJ: s6 1, a1 0.3981, s8 1.9889, a2 4.4211 (DFTD3 V3.1) | CPCM(Water) + `smd true` water, ε 78.355, CDS 있음 | 기본(DefGrid2): IntAcc 4.388 / Lebedev-302, COSX 50·110·194, OptM3 | TolE 1e-8, Thresh 2.5e-11, RIJCOSX 켬 | 27파일, 기준과 차이 0 |
| S2 · S3 Nurion | ORCA 6.1.1. Nurion 빌드이며 UBAI 설치본과 sha256 같음, 8 MPI | 같음 | 같음 | 같음 | 같음 | 같음 | 파일별 기록 없음. 같은 D1 설정으로 돈 d1val 출력(대리)에서 Opt·OptTS는 1e-8, **LooseOpt(path · fwd/bwd)는 TolE 1e-6, Thresh 1e-10** | tar는 삭제됨. 아래 세 가지로 보였다 |
| S4 D0 | ORCA 6.1.1. D0G1·D0G3는 직렬 EnGrad, D0G2·d1val은 8 MPI | 같음 | 같음 | 같음 | 같음 | 같음 | 1e-8. 단 d1val LooseOpt(d0_irc)는 1e-6 | 21파일, 수준 차이 0 |

Nurion(S2 · S3) 근거:
- inv_nurion이 남긴 모든 파일에 `label_level` · `same_grid` · 버전 6.1.1 검사를 적용했다.
- N_report 제외 목록에 수준 때문에 뺀 것이 없다. 제외는 TZVP SP, 기울기 없는 파일, 비정상 종료뿐이다.
- d1val 대리 출력 10개가 기준과 같다.

LooseOpt의 SCF 수렴은 Normal 수준이다. 이것은 범함수·기저·grid 같은 수준 차이가 아니라 수렴 차이다. 아래 재계산에서 영향은 힘 0.01 kcal/mol/Å 수준이다.

### 재계산

방법:
- 출처마다 fold0 **train**에서 시드 고정으로 10쌍, 즉 20구조를 뽑았다. 한 쌍은 같은 반응 또는 같은 분자이고 조성도 같다.
- `gen_dft.engrad`로 직렬 ORCA, 노드 로컬에서 계산했다(run_struct와 같은 방식).
- 오차는 저장된 라벨과 비교했다.

| 출처 | 성공 | \|ΔE_abs\| 최대 / 중앙 (Eh) | 반응 안 상대 E MAE / 최대 (kcal/mol) | 힘 MAE / 최대 (kcal/mol/Å) | 판정 |
|---|---|---|---|---|---|
| S1 | 20/20 | 9.7e-5 / 3.5e-7 | 0.0010 / 0.0028 | 0.0006 / 0.033 | 유지 |
| S2 | 20/20 | 2.9e-5 / 4.0e-6 | 0.0061 / 0.018 | 0.0023 / 0.060 | 유지 |
| S3 | 20/20 | 4.2e-5 / 5.4e-6 | 0.0068 / 0.021 | 0.0019 / 0.058 | 유지 |
| S4 | 20/20 | 3.8e-5 / 2.0e-6 | 0.0039 / 0.019 | 0.0005 / 0.032 | 유지 |

- 뽑은 종류(strata):
  - S1: ts_near, v_conf react_opt, g2_path, ts_imag, G3L ts_path, react_disp, react_ref, ts_final_g2, ts_final, ts_label
  - S2: ts_path, tsguess_opt, ts_final, path, fwd_bwd, neb
  - S3: prod_opt, react_opt, x_ts_path, x_path, truncated, ref_opt, ts_path, x_react_opt, x_tsguess_opt, path
  - S4: d0_g2_path, d0_ts_near, d0_react_disp, d0_ts_imag, d0_react_ref, d0_ts_path, d0_react_opt, d0_prod_opt, d0_ts_guess, d0_ts_final_g2
- 같은 run_struct 코드로 만든 행(G1, G3, D0G1, D0G3)은 \|ΔE\| ≤ 5e-9 Eh로 재현된다.
- 다른 방식으로 만든 행에는 run마다 거의 일정한 에너지 차가 있다. 8 MPI 최적화 궤적이고 최대 1e-4 Eh다(S1의 G2 1008507). 같은 run 안의 상대 에너지에서는 이 차가 상쇄된다.
- LooseOpt에서 온 행(path, fwd_bwd)의 힘 MAE는 0.005–0.012로 가장 크다. 그래도 기준 0.2보다 훨씬 작다.

## ② 누수(정체성 기준)

### 시험 반응 집합

- S_folds에서 fold_T1 == 0인 짝은 1,800개다. a_rxn 1,791개와 b_rxn 53개를 합쳐 1,843반응이다.
- 이는 fold 0 대상 1,874개의 부분집합이다. 빠진 31개는 짝이 없는 반응이다. b_rxn은 모두 fold 0이다.
- LADDER `F_pairs_cr2.csv`와 짝 ID · a · b가 모두 일치한다.

비교한 정체성은 세 가지다.
- (가) 입체를 지운 canonical SMILES 쌍. gen_d0 InChI 쌍도 같은 결과를 냈다.
- (나) 입체를 포함한 isomeric SMILES · InChI 쌍.
- (다) 표지와 무관한 기하 연결 그래프. 저장된 구조에서 두 조각의 연결을 읽는다.
  - (다) 검증: 시험 반응 라벨 TS 1,843개 중 1,831개가 SMILES와 일치했다. 나머지 12개는 형성 결합이 연결로 잡혀 한 조각이 됐다.
  - 2조각 행에서 표지와 기하가 일치하는 비율: S1 100 %, S4 100 %, S2 68,928/68,933. S3 불일치 846개는 대부분 truncated 행이다.

### 반응 쌍 겹침

| 집합 | (가) 입체 제거 | (나) 입체 포함 | 내용 |
|---|---|---|---|
| train | **50** (모두 기하 근거) | **31**, 판정 불가 19 | Nurion `nur_truncated` 45 + `nur_x_truncated` 5. 예: T02573(norbornene + 카보닐 일라이드)의 잘린 계는 연결 구조가 fold 0 시험 1002466 · 1002467 · 1002508((E/Z)-2-butene + NH2 일라이드)과 같다 |
| valid | **1,321** (표지 1,310 + truncated 11) | **0**, 판정 불가 11 | E/Z 형제 core. 예: nitrile oxide/AppE(fold 0, 시험) ↔ AppZ(fold 2, fold0의 검증 core). S1 442, S2 865, S3 14 |

- 제거 목록 `leak_remove.json`:
  - `remove`: (가) 기준 1,371개
  - `remove_isomeric`: (나) 기준 31개
  - `review_geometry_stereo_unresolved`: 30개
  - 행마다 train/valid를 표시했다.
- valid 행은 학습하지 않는다. 다만 checkpoint 선택에 쓰인다.
- fold 사이 입체 형제(참고): 대상 474개가 입체를 지운 쌍 158개를 여러 fold에 걸쳐 공유한다. 이 중 fold 0 ↔ fold 2가 77 ↔ 77이다. 입체를 넣으면 fold에 걸친 쌍은 0이다. 따라서 FT_2(fold 2 시험)에서도 같은 종류의 겹침이 생길 수 있다.

### 분자 겹침과 생성물 겹침 (별도 보고)

- 분자: train 113행이 연결 구조상 fold 0 시험 반응물과 같다. 입체까지 같은 것은 62행이다.
  - Phase I S0의 HCNO 8행. 표지 규칙(d1_rxn_match)이 1000010만 붙여서 남았다. 이 분자는 시험 반응 1000113 · 1000471 · 1000472의 반응물이다.
  - LADDER Q 키가 달라 표지 합집합에 들어가지 않은 G3 분자들.
  - 나머지 51행은 기하에서 InChI를 다시 만들지 못했다.
- 생성물: train 35행은 truncated 경로 끝쪽 구조다. 입체까지 같은 것은 10행이다. valid 321행은 AppZ 생성물이다.

## ③ D3 일치 (라벨 TS 20개: Si 5, S 5, F 2, Cl 2, Br 2, CHNO 4)

| 비교 | 최대 \|ΔE\| (Eh) | 최대 \|ΔF\| (Eh/bohr) | 판정 |
|---|---|---|---|
| AIMNet2 DFTD3, float32 (학습한 모듈) vs ORCA `Edisp` · `DISPERSION GRADIENT` | 4.1e-8 | 7.5e-9 | 통과 |
| 같은 모듈을 전체 모델 안에서(hook) 잰 값 vs ORCA | 4.1e-8 | 7.5e-9 | 통과 |
| 같은 모듈 float64 vs ORCA | 3.2e-8 | 7.6e-9 | 통과 |
| AIMNet2 float32 vs simple-dftd3 1.5.0 (2체, DFTD3 3.1 cutoff) | 4.0e-8 | 4.6e-9 | 통과 |
| simple-dftd3 vs ORCA | 5.0e-9 | 5.6e-9 | 통과 |

- 검사 1의 80구조까지 합친 100구조에서도 최대 5.1e-8 Eh, 9.4e-9 Eh/bohr로 통과했다.
- 매개변수는 s6 1, s8 1.9889, a1 0.3981, a2 4.4211이고 3체 항은 없다. 참조 표(c6ab 등)는 aimnet 상수와 같다.
- 진단: aimnet의 CN 상한(cnmax)을 빼면 S를 포함한 3구조에서 1.7e-2 Eh까지 달라진다. 즉 이 상한이 있어야 ORCA와 같은 값이 나온다.
- **정밀도**: 학습과 추론 모두 float32다.
  - `AIMNet2Base._prepare_dtype`이 좌표를 float32로 바꾼다.
  - `finetune.aim_predict`와 `mlts_bridge.AIMNet2Dense`도 float32를 넘긴다.
  - export는 atomic shift만 float64로 바꾼다.
  - float32에서도 기준을 넘는다.

## 추가 보고

### FT-1 출처별 (best checkpoint)

기준은 TS 근방 힘 ≤ 0.6, 상대 E ≤ 0.3이다.

| 집합 | 출처 | n / TS 근방 | TS 근방 힘 MAE (kcal/mol/Å) | TS 근방 E MAE, 구조당 절대값 (kcal/mol) | 반응 안 상대 E MAE: TS 근방 / 전체 (kcal/mol) |
|---|---|---|---|---|---|
| valid | 전체 | 20,000 / 13,775 | 0.467 | 0.395 | 0.113 / 0.198 |
| valid | S1 | 5,554 / 3,663 | 0.530 | 0.381 | 0.111 / 0.122 |
| valid | S2 | 10,112 / 10,112 | 0.444 | 0.399 | 0.091 / 0.091 |
| valid | S3 | 4,334 / 0 | (전체 0.496) | (전체 0.413) | — / 0.128 |
| test | 전체 | 20,000 / 13,768 | 0.527 | 0.477 | 0.156 / 0.225 |
| test | S1 | 12,128 / 8,259 | 0.558 | 0.478 | 0.139 / 0.147 |
| test | S2 | 5,510 / 5,509 | 0.481 | 0.476 | 0.106 / 0.107 |
| test | S3 | 2,362 / 0 | (전체 0.511) | (전체 0.465) | — / 0.133 |
| D0 holdout TS (S4) | S4 | 807 / 807 (269반응) | **0.917** | 0.889 | 0.209 / 0.209 |

- valid와 test에는 D0 행이 없다. D0는 학습에만 쓴다.
- 그래서 S4는 한 번도 학습하지 않은 D0 holdout TS 근방 행(gen_D0G1)으로 쟀다. 이 집합의 힘 MAE는 0.6을 넘는다.

### FT-2 최적화기와 C-1

- **FT-2 최적화기**: ORCA 6.1.1 `! ExtOpt OptTS`를 `mlts_bridge.optimize_ts`(서버 + `mlts_client` ProgExt)로 돌린다. Sella는 FT-2에 쓰지 않고 C-1 (ii)의 대조 최적화기로만 쓴다.
  - 기하 블록: D1 블록(Trust −0.1, MaxIter 100, 형성 결합 modify_internal).
  - 첫 Hessian: MLIP Hessian을 `InHess Read`로 넣는다. AIMNet2는 힘의 중심 차분(h 2e-3 Å)이다.
  - Recalc: `analytic:30`.
  - 수렴: **TightOpt**(ts_geom `TIGHT_RUN = 'tight'`).
  - 허수 진동수 판정: −50 cm⁻¹ 아래.
- **C-1**:
  - zero-shot AIMNet2는 (ii)에서 실패했다(`pass_C1` false; ORCA가 100 cycle 안에 수렴하지 못한 반응 2/5, 이 두 반응의 \|Δd\|는 0.092 Å와 0.33 Å).
  - 미세조정 AIMNet2로 (ii)를 다시 돌렸다. best checkpoint를 `$MLT/audit/models/FTpilot_AIMNet2_0_best.jpt`로 export했다(export 검사 4e-6 eV).
  - 결과: **통과**. ORCA ExtOpt OptTS와 Sella가 5/5 모두 수렴했고 형성 결합 차 최대 0.00077 Å(기준 0.005)이다. 허수 진동수는 모두 1개다.
  - 이 검사는 tests_c1 기본값(NormalOpt)으로 돌렸다. `results/C1_tests.json`은 고치지 않았다.

## 규칙 기록 (나, login)

- `python3 --version`을 1번 실행했다. login에서 python 금지 규칙 위반이다.
- 읽기 전용 작업:
  - 파일 존재를 보는 `for` 루프 1번(`> /dev/null` 포함)
  - `sleep 0` 1번
  - nurion.extxyz.gz 앞 300 MB에 `zcat | grep` 1번, 큰 출력에 `grep` 여러 번
- 감사 job 4개(한도 6). 다른 job은 취소하지 않았다.
