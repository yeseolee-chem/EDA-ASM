# REV4_XTB_GEOMETRY — espley_xtb_repro feature 기하 교정 (DFT 기하 → xTB 기하)

대상: `main` 브랜치 `analysis/espley_xtb_repro/` (기준 commit 1ada37a).
결과 위치: 새 폴더 `analysis/espley_xtb_repro/results_rev4/`. rev 3 파일은 지우거나 덮어쓰지 않는다.
각 Phase 종료마다 보고하고, STOP 조건이 나오면 다음 Phase로 가지 않는다.

> **실행 메모 (2026-09-30, 사용자 지시):** "이전 실험 결과 전부 삭제" — rev 3 결과(`results/`)와 scratch의
> rev 3 ML 산출물은 rev 4 실행 과정에서 삭제한다(git 이력에는 남는다). G0(rev 3과 같은 DFT 기하) 결과는
> rev 4 코드로 같은 행에서 다시 계산해 `results_rev4/`에 "상한"으로 싣는다. 기존 feature parquet은
> Phase 2의 `--geom dft` 일치 검증이 끝난 뒤 삭제한다. D1(`D1_build/`, `d1-build`)은 별도 작업이므로 건드리지 않는다
> — D1 함수는 `d1-build@d573111e`의 git 객체에서 뜬 읽기 전용 스냅샷에서 import 한다.

---

## 0. 배경 — 확인된 사실 (2026-09-30)

### 0-1. rev 3 feature는 전부 Coley DFT 기하에서 계산됨
`xtb_slice.py` (v4) docstring: "GFN2-xTB / ALPB(water) features on Coley DFT geometries". `process_one()`은
`PROF/<rid>/`의 `ts_file`, `rel1_file`, `rel2_file` — 라벨이 계산된 바로 그 DFT TS와 DFT 참조 구조(`_alt` 선택까지
동일) — 위에서 xTB 단일점 5개, 거리 11개, Mulliken 15, Wiberg valence 15, D3 분산, SASA를 계산한다. `b_disp`는 DFT
기하에서 DFT disp 라벨과 해석적으로 같다(MAE 0.000) — disp 채널에서는 라벨 자체가 feature로 들어가 있다.
`compare_espley.py`, `train_ml*.py`, `analyze_extra.py`, `evaluate_pairs.py`, `aggregate.py`, `refresh_targets.py`,
`rev3_doc_numbers.py`는 모두 이 parquet을 읽는다.

### 0-2. 무엇이 문제인가
rev 3은 "라벨을 계산한 DFT 기하를 입력으로 쓴 상한"(oracle geometry)이다. 새 반응에는 DFT TS가 없으므로 이 모델을
배포할 수 없고, Espley(AM1 기하)와 입력 정보가 같지 않다.

### 0-3. 근거 (로컬 절제, KRR, 5 seed 평균)
Espley AM1 46 feature에 DFT 기하 거리 11개만 더하면(xTB 없이) d1 2.66 → 2.05, Interaction 2.43 → 1.84,
ΔE‡ 3.05 → 2.67 — 우리 모델과의 격차의 28–100%가 이것만으로 사라진다. 같은 데이터에서 AM1 TS 형성 결합 길이 대
DFT: MAE 0.13 / 0.27 Å (짧은 / 긴 결합), r 0.65 / 0.36.
(자료: `claude/review-espley-comparison-fairness.md`, `espley_fairness_ablation.py`,
`espley_fairness_ablation_per_seed.csv`.)

### 0-4. 목표
feature는 xTB 수준에서 얻은 기하에서만 계산한다. 타깃은 바꾸지 않는다(`labels_all.json`의 DFT 라벨, DFT 기하
기준). 과제는 "싼 기하에서 계산한 feature로 비싼 DFT 라벨을 예측" — Espley 2024(AM1 기하 → DFT 타깃)와 같은 정의.

---

## 1. 규칙
- HPC: CLAUDE.md 규칙(sbatch, `--time=48:00:00`, 로그인 노드 python/루프/상주 프로세스 금지, 제출 전
  `squeue -u $USER -h | wc -l`, MaxSubmit 20은 array 원소 단위, idempotent + 원자적 쓰기).
- 보존: `labels_all.json`과 rev 3 결과 파일은 수정하지 않는다. (→ 실행 메모 참조: rev 3 결과는 사용자 지시로 삭제.)
- feature 엔진: 지금과 같은 xtb 6.7.1 (GFN2, ALPB water). 기하 최적화 엔진은 Phase 1-1에서 확정.
- 사전 등록: Phase 3 학습 전에 `results_rev4/PREREG_REV4.md`를 commit하고 이후 바꾸지 않는다.
- 판정 금지: 결과를 본 뒤 arm, 모델, 행을 고르지 않는다. 선택 규칙은 사전 등록에 적힌 것만 쓴다.

## 2. 기하 정의

| 이름 | TS | 반응물 참조 | 남는 DFT 정보 | 용도 |
|---|---|---|---|---|
| G0 (rev 3) | Coley DFT TS | Coley DFT 참조 (orig 또는 `_alt`) | 전부 | 상한. "oracle geometry"로 표기 |
| G1 | DFT TS에서 출발해 GFN2-xTB/ALPB(water)로 OptTS | DFT 참조에서 출발해 xTB opt | conformer와 입체 선택만 | 본실험 교정 1차, Espley 대응 비교 |
| G2 | autodE로 SMILES에서 xTB 수준 새 탐색 | autodE xTB conformer | 없음 | 역설계 배포 성능 (Phase 5, 승인 후) |

## Phase 0 — 감사 (계산 없음)
1. `xtb_features.parquet`과 Coley 프로필 xyz를 입력으로 쓰는 모든 스크립트와 결과 파일을 나열하고 G0 의존 / 무관 표시.
2. `results/SUMMARY.md`, `README.md`, 그림에서 rev 3 수치를 배포 가능한 성능처럼 서술한 문장을 모두 찾는다.
3. `results_rev4/AUDIT_G0.md`로 보고.

## Phase 1 — G1 기하 생성
### 1-1. 엔진 smoke test (sbatch 1건)
대상: rxn 20, 105 + 무작위 3건(seed 20260930, `labels_all` ok 중).
TS: ORCA 6.1.1
```
! XTB2 ALPB(water) OptTS Freq
%geom Calc_Hess true Recalc_Hess 5 MaxIter 200 end
%pal nprocs 2 end
* xyz <q1+q2> 1   (Coley ts_file 좌표)
```
참조: `xtb rel.xyz --opt tight --alpb water --chrg <q> --gfn 2` (rel1_file, rel2_file 각각).
확인: (1) ORCA 출력에 GFN2-xTB와 ALPB(water)가 실제 적용됐는지 문자열 그대로 인용; (2) `.hess`를
`orca_direct.parse_hess`로 읽을 수 있는지, 첫 허수 진동수; (3) 엔진 일치: xtb 6.7.1 `--grad --alpb water`로 최적화된
TS 기울기 max |g| < 5e-4 Eh/bohr, ORCA `otool_xtb` 버전 보고.
STOP: ORCA에서 XTB2 + ALPB(water)를 함께 쓸 수 없거나 확인 3 실패 → 멈추고 사용자가 대안 결정(예: ASE + Sella + xtb).

### 1-2. 전체 실행 (array, 18 slices %10)
대상: `labels_all` ok 5,260 (`xtb_slice.EXCLUDE` 5건 제외). 반응별: (1) TS를 1-1과 같은 입력으로 최적화(출발점 Coley
`ts_file`); (2) 참조 두 개 `xtb --opt tight`(출발점 라벨과 같은 `rel1_file`, `rel2_file`); (3) gate — D1 함수 그대로
(`run_ts.imag_gate`, `forming_share`, `fr.partition`, `fr.audit_forming_bonds`): 첫 허수 ≤ −40 cm⁻¹ 이고 나머지 허수
모두 > −50 cm⁻¹; `formed_pairs_ts` 위 mode share ≥ 0.5 (원자 순서 동일 → 라벨 인덱스 사용); `not_product_like`,
`forming_bond_present` (`forming_min_A 1.6`, `no_bond_A 3.3`); `no_foreign_bond`; G1 TS partition = 라벨 `A_idx`;
최적화 후 참조 구조의 결합 그래프가 DFT 참조와 동형. (4) 기록 `g1/<rid>/result.json`: status, 허수 목록, G1 TS–DFT TS
heavy-atom RMSD(같은 원자 순서, Kabsch), Δd_form, 참조 RMSD, 에너지, 계산 시간. (5) 저장
`/gpfs/tmp_cpu2/yeseo1ee/espley_xtb_g1/<rid>/{ts.xyz,rel1.xyz,rel2.xyz,result.json}` + `.done` 또는 `.fail_<이유>`.
보고: 성공률, 실패 유형별 건수, RMSD·Δd_form 분포(중앙값, 90%).
STOP: 성공률 < 90% 또는 partition 불일치 > 1% → Phase 2로 가지 않고 보고.

## Phase 2 — feature 재계산
1. `xtb_slice.py`에 `--geom {dft,g1,g2}` (기본 `dft` = rev 3 동작). 검증: `--geom dft`로 slice 1개를 다시 돌려 기존
   parquet과 feature 값이 1e-8 안에서 같은지. 다르면 STOP.
2. `g1`: TS·참조를 `espley_xtb_g1/<rid>/`에서 읽음; `formed_pairs_ts`와 역할은 라벨 값 그대로; partition 재계산해
   라벨과 같은지 확인; Phase 1 `.fail`인 반응은 `xtb_status = g1_fail:<이유>`.
3. 출력 `/gpfs/tmp_cpu2/yeseo1ee/espley_xtb/xtb_features_g1.parquet` (rev 3과 다른 이름).
4. 보고: ok 건수, feature별 G0 vs G1 분포 변화(중앙값 차이), G1에서의 `MAE(b_disp − dft_disp_dft)` (0이 아니어야 정상).

## Phase 3 — 본실험 재학습 (rev 4)
### 3-1. 사전 등록 (학습 전 commit) — `results_rev4/PREREG_REV4.md`
- 행: G1 ok ∩ rev 3 hygiene filter(d1 < 0, d2 < 0, d2 > 50 제외). 최종 n 명시.
- G0 재실행: 같은 행 집합에서 G0도 다시 학습(같은 행 비교).
- Protocol A는 rev 3과 동일: 80/10/10, seed 22/23/14/1/2, 각 seed train에서 nested GridSearchCV 5-fold,
  StandardScaler + `TransformedTargetRegressor`.
- arm: ESPLEY46, ESPLEY54, ESPLEY73. 모델: Ridge, KRR(RBF), SVR(RBF), XGB.
- 타깃 9: barrier, d1, d2, elst, pauli, oi, disp, cpcm, cds.
- 헤드라인: G1 · ESPLEY73 · KRR(RBF) (rev 3과 같은 사전 고정 규칙).
- G0는 모든 표·그림에서 "상한(DFT oracle geometry)"으로 표기, 헤드라인으로 쓰지 않는다.
- G0의 disp는 `b_disp` 해석적 항등이므로 ML 성과로 보고하지 않는다.
### 3-2. 실행 — `s03_ml_array.sh`, `train_ml_single.py`에 feature 파일 경로 옵션 추가. 출력 `results_rev4/`.
### 3-3. 하류 분석 재실행 — Phase 0 목록에서 G0 의존 분석을 G1 예측값으로 다시(`analyze_extra.py` group_split,
charge_breakdown, margin_calibration; `evaluate_pairs.py` MMP 등).
### 3-4. 보고 — 표: 타깃 × {G0 상한, G1} × 헤드라인 모델의 MAE, NMAE, r², G1 − G0. arm별·모델별 표는 부록.

## Phase 4 — Espley 비교 재실행
### 4-1. G1으로 동일 조건 비교 — `compare_espley.py`에 `ESPLEY_FEAT` 경로 옵션. 반응·분할·지표·Espley 재채점 동일.
우리 쪽 G0와 G1 두 줄. 주 비교: G1 · ESPLEY46 · KRR 대 Espley SVR. 부록: 모델 중 최고값끼리.
### 4-2. 역할 기준 d1/d2를 양쪽 모두에서 — Espley 쪽: 46 feature, `distortion_energy_1/2_am1` 두 열을 같은 swap 벡터로
역할 기준 재배열, 역할 기준 타깃으로 재학습 (a) 그들의 프로토콜(그들의 grid, ESI Table S3; seed 23에서 한 번 튜닝;
X만 StandardScaler) (b) 우리 파이프라인. 로컬 참고값(KRR, 우리 파이프라인): dipole 2.60, dipolarophile 1.89.
우리 쪽: G1 역할 기준 d1/d2. 문서 규칙: 인덱스 기준 d1/d2 비교는 부록으로; "양쪽이 같은 핸디캡" 문장 삭제
(근거: Espley AM1 distortion feature는 타깃과 같은 인덱스(같은 인덱스 r 0.75, 뒤바뀐 인덱스 0.35) → 인덱스 혼합의
오차 증가가 한쪽에만 크다: 우리 1.03 → 2.45, Espley 2.60 → 2.66).
### 4-3. 기하 정보 절제를 저장소에 기록 — `espley_fairness_ablation.py`를 `analysis/espley_xtb_repro/`에 넣고 sbatch로
재현, `results_rev4/espley_geometry_ablation.csv`. arm: A Espley46 우리 파이프라인; B Espley46 + DFT 거리 11;
B_role (B, 역할 기준 타깃); C Espley46 역할 기준; O 우리 ESPLEY46 seed 23 튜닝; 추가 B_g1: Espley46 + G1 거리 11.
### 4-4. Espley AM1 TS의 출발 구조 확인 — Bath 아카이브(ESI: doi:10.15125/BATH-01398; SUMMARY: "BATH-01480 v2") 중
ds3 파일이 있는 쪽 확인. ds3 AM1 TS 로그 무작위 20건(seed 20260930)의 첫 입력 좌표 vs Coley DFT TS heavy-atom Kabsch
RMSD (`rmsd.mapped_heavy_rmsd`). 중앙값 < 0.01 Å → AM1이 DFT 구조에서 출발 → G1이 Espley와 입력 대응 arm.
아니면 실제 출발 구조 보고. 4.3 GB 전체는 받지 않음(HTTP range만), 로그인 노드 말고 sbatch 안에서.
### 4-5. (선택, 4-4 보고 후 승인 시) AM1 기하 위 xTB feature.

## Phase 5 — G2 (DFT 정보 없음): 파일럿 후 사용자 결정
5-1 파일럿 20건(seed 20260930): D1 venv autodE 1.4.5, hmethod ORCA opt/optts/hess/sp → `XTB2 ALPB(water)`, lmethod
xtb, D1 프로토콜 그대로; 후처리 D1 `run_ts.analyse`(FIX F-A 포함) → `xtb_slice --geom g2`. 5-2 보고: 성공률,
core-h, Coley TS와 같은 TS 비율(`rmsd.same_ts`), G2–G1 feature 차이. 5-3 STOP: 전체 실행 여부·규모는 사용자가 정한다.

## Phase 6 — 문서 정정 (Phase 3·4 결과 후)
SUMMARY.md, README.md: rev 3 수치는 "G0 = DFT oracle geometry 상한"으로 재표기, 헤드라인은 G1(G2 나오면 병기).
삭제·정정: "다른 것은 feature뿐" → 기하 차이 명시; "d1/d2는 양쪽이 같은 핸디캡" → 삭제; `b_disp` → G0에서는 해석적 항등
명시; Espley 비교 결론 → 4-1, 4-2 결과로 다시. CLAUDE.md Current status 갱신. commit:
`espley_xtb_repro rev 4: xTB-geometry features (G1); rev 3 relabelled as DFT-geometry upper bound`

## 보고 형식 (Phase마다)
실행한 job id / 결과 표 / STOP 조건별 판정 / 다음 Phase 진행 여부. 수치는 파일에서 직접 읽은 값만.
