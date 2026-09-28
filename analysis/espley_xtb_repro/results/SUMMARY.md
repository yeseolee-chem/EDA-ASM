# espley_xtb_repro — final results (2026-09-28, rev 3)

**Espley 2024 (D4DD00224E) protocol reproduced on Coley 5,260 rxns with GFN2-xTB / ALPB water.**

> 2026-09-15 rev 1: FIXES.md audit (S1/S3/S5/S8/S9). See §Audit.
> 2026-09-15 rev 2: **B2 + N1–N3 재학습**. `dft_barrier_eda`(8-채널 닫힌 총량) 추가 (rev 3에서 철회: 방법 ② 혼합량), KRR 격자 α[1e-5..1] γ[3e-4..1] 확장, TransformedTargetRegressor(StandardScaler)로 y 표준화, per-seed nested CV, 위생 필터(d1<0/d2<0/d2>50) 26행 제외, `is_charged` AUX 편입 → **ESPLEY73**.
> 2026-09-28 rev 3: **SMD 재계산 라벨 + 방법 ① 조립으로 재학습.** 보고 타깃 9개(barrier, d1, d2, 6채널); `dft_barrier_eda` 제거. xTB 재계산 없음. 아래 §Rev 3가 현재 결과이며, 그 아래 rev 2 절들은 이력으로 남긴다.

## Rev 3 (2026-09-28) — SMD relabel, method ① labels

### 바뀐 것

- **라벨.** repo 루트 `labels_all.json`을 2026-09-28 재생성. 5,265 반응의 EDA를 모두 `SMD(water)` 키워드(헤더 + FRAG1/FRAG2 문자열)로 재계산해 ORCA 내부 fragment SCF(`eda_frag*.out`)에도 SMD-CDS가 들어가도록 했고, **방법 ①**로 조립했다: d1/d2는 자기 basis의 단독 `frag*_dist.out` 기준(순수 변형 에너지), 6채널은 ORCA EDA 표. 전수검사(파일/입력/SMD 통일/3 게이트) 통과 — `label_true/scripts/smd_final_pipeline.sh`, `label_true/SMD_RELABEL.md`.
- **타깃만 교체, xTB 재계산 없음** (`refresh_targets.py`). 변화: d1, d2, disp, cds는 0행; **cpcm, elst, oi, pauli (및 e_bond)는 5,260행 전부** (max |Δ| 38.3 / 24.4 / 11.3 / 2.8 kcal/mol); barrier는 2,978행이 SCF 수렴 수준(max 0.068)으로만 변함.
- **보고 타깃은 9개: barrier, d1, d2, 6채널 (elst, Pauli, OI, disp, CPCM, CDS).**
  - e_bond(= 6채널 합)와 ΔE_int(자기 basis)는 파생값이라 타깃에서 제외. 학습 결과는 `ml_table_espley.csv`에 남아 있고, ΔE_int는 Espley 비교표의 기준선으로만 쓴다.
  - 타깃 12번 `dft_barrier_eda`(= d1 + d2 + e_bond, 방법 ② 혼합량)는 제거하고 그 자리를 `dft_c_ghost_kcal`로 채웠다. c_ghost는 계산 방법이 남긴 흔적(BSSE + cavity)이라 보고 대상이 아니다 → 아래 SI 한 줄.
- 프로토콜(N1–N3, 5-seed nested CV, y-std)과 행 수 동일: **n = 5,234** (위생 필터 26행 동일).
- 제외 5 rxn (5,265 → 5,260): 3090 / 3766 / 4252는 foreign bond, 3400 / 5783은 **형성결합 두 개가 모두 ≥ 3.3 Å**(3.63 / 3.65, 3.34 / 3.56 Å)라 결합이 생기는 TS가 아니다 — 수락된 5,260 rxn은 짧은 쪽 형성결합이 모두 ≤ 3.18 Å. (CPCM 시기의 사유 `oi_dft > 0`은 SMD 라벨에서 성립하지 않아 교체.)
- 그림을 **ESPLEY73, 9 타깃** 기준으로 교체 (`figures/scatter_<model>_ESPLEY73.png` × 4, `figures/mae_bar_espley73.png`). ESPLEY54 그림은 삭제.

### Headline — `ESPLEY73` + KRR(RBF) (a priori), 5-seed 80/10/10, n = 5,234

| Target | test MAE ± sd | NMAE | r² | % of range | group-split MAE (dph / dip) | rev 2 MAE |
|---|---:|---:|---:|---:|---:|---:|
| ΔE‡ (barrier) | **1.45 ± 0.05** | 0.201 | 0.956 | 2.46 | 1.64 / 1.51 | 1.45 |
| d1 (dipole strain) | 1.12 ± 0.03 | 0.187 | 0.960 | 2.31 | 1.09 / 1.11 | 1.12 |
| d2 (dipolarophile strain) | 0.82 ± 0.02 | 0.170 | 0.966 | 1.93 | 1.00 / 0.85 | 0.82 |
| elst | **1.52 ± 0.04** | 0.111 | 0.985 | 1.40 | 1.61 / 1.58 | 1.62 |
| Pauli | 2.02 ± 0.08 | 0.078 | 0.992 | 0.96 | 2.10 / 2.08 | 2.00 |
| OI | **1.16 ± 0.05** | 0.074 | 0.993 | 0.92 | 1.15 / 1.15 | 1.23 |
| CPCM | **1.01 ± 0.02** | 0.251 | 0.945 | 1.81 | 1.06 / 1.01 | 1.46 |
| CDS | 0.23 ± 0.01 | 0.342 | 0.879 | 3.38 | 0.26 / 0.24 | 0.23 |
| *disp* | *0.01 (b_disp와 해석적 등식)* | — | 1.000 | — | — | 0.01 |

### 4개 모델 — Protocol A, `ESPLEY73`, test MAE (kcal/mol), rev 3 [rev 2]

| Target | Ridge | KRR | SVR | XGB |
|---|---:|---:|---:|---:|
| barrier | 1.93 [1.93] | **1.45** [1.45] | 1.48 [1.48] | 1.75 [1.77] |
| d1 | 1.41 [1.41] | 1.12 [1.12] | **1.12** [1.12] | 1.18 [1.18] |
| d2 | 1.05 [1.05] | 0.82 [0.82] | 0.81 [0.81] | **0.77** [0.77] |
| elst | 2.48 [2.62] | **1.52** [1.62] | 1.55 [1.65] | 2.25 [2.41] |
| Pauli | 3.93 [3.90] | **2.02** [2.00] | 2.18 [2.17] | 3.19 [3.18] |
| OI | 1.98 [2.03] | **1.16** [1.23] | 1.23 [1.28] | 1.93 [1.99] |
| CPCM | 1.33 [1.96] | **1.01** [1.46] | 1.02 [1.46] | 1.23 [1.67] |
| CDS | 0.35 [0.35] | 0.23 [0.23] | **0.23** [0.23] | 0.26 [0.26] |
| disp | **0.00** [0.00] | 0.01 [0.01] | 0.11 [0.11] | 0.04 [0.04] |

`ml_table_espley.csv`(324행: 12 학습 타깃 × 3 arm × 9 model, Protocol A + B)에는 보고하지 않는 e_bond, eint_spe, c_ghost 행도 들어 있다. rev 2 대비 열은 `rev3_vs_rev2.csv`.

### 관찰

- **라벨이 바뀐 채널에서만 변화, 전부 개선 또는 동일.** CPCM 1.46 → **1.01** (−31%, r² 0.894 → 0.945, NMAE 0.339 → 0.251), elst 1.62 → 1.52 (−6%), OI 1.23 → 1.16 (−6%), Pauli 2.00 → 2.02 (±1%). 라벨이 (거의) 같은 barrier / d1 / d2 / disp / cds는 KRR 기준 소수 셋째 자리까지 rev 2와 동일, 다른 모델은 ±0.03 이내 (예: XGB barrier 1.77 → 1.75) — 파이프라인 재현성 확인.
- **KRR 사전 고정 유지.** 9 타깃 중 KRR이 최적이 아닌 경우와 손실: d2 +0.044 (XGB), disp +0.009 (Ridge, 해석적 등식), d1 +0.003 (SVR), cds +0.000.
- **Arm ablation (KRR):** barrier 1.68 / 1.50 / 1.45, CPCM 2.93 / 1.18 / 1.01 (46 / 54 / 73).
- **그룹 분할 robustness** (`group_split.csv`): 같은 파일의 random 열 대비 1.20× 이내 (최대 d2 dipolarophile 분할 1.20×; 헤드라인 5-seed MAE 대비로는 d2 1.22×).
- **전하별** (`charge_breakdown.csv`, KRR/73): barrier 중성 1.44 vs 하전 1.67 (q2 = +1: 2.81, 5-seed test fold 합집합의 고유 24 rxn), CPCM 중성 0.97 vs 하전 1.81 — 하전 반응이 여전히 약점.
- **MMP 쌍** (`split_metrics.csv`, random split 1,066 쌍): ΔMAE Pauli 2.54 / elst 1.79 / OI 1.39, dominant-channel 일치 0.826.

### SI 한 줄 — 8채널 합과 장벽

d1 + d2 + Σ(6채널)은 장벽과 평균 |Δ| **0.55 kcal/mol**, 5,260 중 **780 반응(14.8%)에서 1 kcal/mol 넘게** 어긋난다 (최대 6.0). 차이는 `c_ghost = ΔE_int(자기 basis) − E_bond(EDA)`: ORCA EDA의 fragment 기준(상대 fragment의 ghost basis + 복합체 AB cavity)과 자기 basis 단독 fragment 기준의 차이(BSSE + cavity)이며, ORCA 표 closure 잔차(≤ 0.11)가 더해진다. 계산 방법의 흔적이라 연구 내용과 무관하며, 방법 ①에서 `barrier = d1 + d2 + Σ6ch + c_ghost`로 정확히 닫힌다 (`labels_all.json`의 `bsse_shift_kcal`, 평균 +0.10, sd 0.77).

### Rev 3 파일

`ml_report.json`, `ml_table_espley.csv`, `rev3_vs_rev2.csv` (신규), `predictions.parquet`, `xtb_features.parquet` (dft_* 교체본), `ml_targets/target_00…11_*.json` (11 = `dft_c_ghost_kcal`), `figures/*ESPLEY73*.png` (9 타깃), `charge_breakdown.csv`, `group_split.csv`, `split_metrics.csv`, `margin_calibration.csv`, `mmp_pairs_v2.csv`. `verify_s1s5.json`은 rev 1/2 검증 기록.

---

아래 절들의 적용 범위:
- **Method · Feature blocks · Gates · Model selection · disp 등식 · 가정·주의 · 산출물 · 재현**: rev 3에도 그대로 적용 (G3의 derived 열만 rev 3에서 `dft_c_ghost_kcal`, `is_charged`).
- **전하 그룹 · 그룹 분할 · Espley 비교 · Audit S1 / S3+S8 / S5 / S9**: 2026-09-28에 rev 3 결과와 정정된 설명으로 갱신.
- **(rev 2) Final headline model, Audit S4, "rev 2에서 처리 완료"**: rev 2 기록으로만 남김 (CPCM 시기 채널 라벨). 방법 ② 혼합량 `dft_barrier_eda`의 행과 논증은 삭제·철회.

## Method (one-line)

Single engine: **xtb 6.7.1 binary**, one GFN2-xTB / ALPB(water) single point per structure gives term-wise energies, Mulliken charges, per-atom Wiberg valences, HOMO-LUMO gap, dipole moment. tblite not used.

## Feature blocks

- `d^struct` **41** — 11 dist + 15 Mulliken + 15 Wiberg-valence
- `b^xtb`    **5**  — barrier, dist_dipole, dist_dipolarophile, sum, interaction (ALPB totals; q_barrier omitted, no Hessian)
- `b^ch`     **8**  — b_strain_1/2 (gas-part strain), b_elst (frozen monopole Coulomb), b_pauli (rep), b_oi (EHT), b_disp (D3(BJ)/B3LYP), b_cpcm (ΔGelec), b_cds (ΔGsasa+Ghb+Gshift) — zero markers 없음, 전부 실측
- `AUX19`         — b_elst_scc, b_disp_d4, b_axc, b_ct, gap_ts/dip/dph, mu_ts/dip/dph, dmu_complexation, **is_charged**, dsasa_{H,C,N,O,F,Cl,Br}

**Three arms**: `ESPLEY46` (41+5), `ESPLEY54` (41+5+8), **`ESPLEY73`** (54+AUX19).

## Gates (모두 통과)

| # | 항목 | 결과 |
|---|---|---|
| G1 | 슬라이스 로그 | 18/18 accepted=5260, xtb=6.7.1, ok |
| G2 | xtb_solvation | `{'alpb-water': 5260}` (gas 없음) |
| G3 | aggregate | **5,260 rows / 5,260 unique / 100% ok** (+ derived `dft_c_ghost_kcal`, `is_charged`; rev 2에서는 `dft_barrier_eda`, `dft_bsse_gap`) |
| G4 | **해석적 적합성** | `mean\|b_disp − dft_disp_dft\| = 2.4 × 10⁻⁶ kcal/mol`, `max 8 × 10⁻⁶` — 사실상 등식 |
| G5 | feature 품질 | 73 컬럼, NaN 0, 상수 0, 순서 중복 0 |
| G6 | ML 사용 행 | **5,234** (5,260 − 26 위생 필터: d1<0, d2<0, d2>50) |

## Model selection — KRR(RBF)을 사전 고정 (a priori)

**모델 앙상블에서 test MAE 최소를 뽑는 방식은 낙관 편향을 유발.** rev 3 `ESPLEY73`, 보고 타깃 9개 기준 KRR이 5개(barrier, elst, Pauli, OI, CPCM)에서 최적. 나머지 4개의 최적 대비 손실은 d2 +0.044 (XGB), disp +0.009 (Ridge, 해석적 등식), d1 +0.003 (SVR), cds +0.000 (SVR). 이 대가로 cherry-picking 편향 완전 제거.

**Ridge / SVR / XGB의 결과는 `ml_table_espley.csv`에 전부 기록** (324 행: 12 학습 타깃 × 3 arm × 9 model; 보고하지 않는 e_bond, eint_spe, c_ghost 포함).

## (rev 2, superseded by §Rev 3) Final headline model — **`ESPLEY73` + KRR(RBF), 5-seed 80/10/10 (nested CV, y-std)**

rev 2 재학습 후 수치. 각 seed의 자기 train fold 위에서 GridSearchCV(5-fold)를 별도로 실행 (seed 23 단일 튜닝 leak 제거), KRR α∈[1e-5..1] γ∈[3e-4..1] 격자, TransformedTargetRegressor(StandardScaler)로 y 표준화. n=5,234.

| Channel | test MAE ± sd | NMAE | r² | % of range | group-split MAE (dph / dip) |
|---|---:|---:|---:|---:|---:|
| ΔE‡ (barrier) | **1.45 ± 0.05** | 0.201 | 0.956 | 2.46 | 1.64 / 1.51 |
| d1 (dipole strain) | 1.13 ± 0.03 | 0.187 | 0.960 | 2.31 | 1.09 / 1.11 |
| d2 (dipolarophile strain) | 0.82 ± 0.02 | 0.170 | 0.966 | 1.93 | 1.00 / 0.85 |
| ΔE_int (SPE) | **0.77 ± 0.03** | 0.156 | 0.971 | 1.93 | 0.83 / 0.84 |
| Bond E (EDA 6채널 합) | 1.17 ± 0.04 | 0.196 | 0.960 | 2.67 | 1.25 / 1.22 |
| elst | **1.62 ± 0.05** | 0.119 | 0.983 | 1.47 | 1.81 / 1.70 |
| Pauli | **2.00 ± 0.08** | 0.077 | **0.992** | 0.95 | 2.09 / 2.07 |
| OI | **1.23 ± 0.05** | 0.079 | **0.992** | 0.98 | 1.24 / 1.22 |
| CPCM | 1.46 ± 0.06 | 0.339 | 0.894 | 2.76 | 1.56 / 1.51 |
| CDS | 0.23 ± 0.01 | 0.342 | 0.879 | 3.38 | 0.26 / 0.24 |
| *disp* | *0.01 (해석적 등식, 아래 참조)* | — | 1.000 | — | — |

(rev 2의 `ΔE‡_EDA (d1+d2+e_bond)` 행은 삭제 — 방법 ② 혼합량이라 장벽 타깃이 아님. 아래 §B2 참조.)

**Rev 1 → Rev 2 델타** (전부 ±10% 안): barrier 1.49→1.45, d2 0.83→0.82, eint_spe 0.79→0.77, elst 1.68→**1.62** (−3.6%), Pauli 2.17→**2.00** (−7.8%, N1 격자 확장 효과 확인), OI 1.30→**1.23** (−5.4%), CPCM 1.48→1.46, CDS 0.24→0.23. **격자 경계 문제 해소**로 Pauli/elst/OI가 계산 채널 축에서 확실히 개선.

### B2 — 철회 (rev 3)

- rev 2의 `dft_barrier_eda = d1 + d2 + e_bond`는 자기 basis 변형(d1, d2)과 ghost-basis·AB-cavity 기준의 EDA 결합에너지(e_bond)를 섞은 **방법 ② 혼합량**이라 장벽이 아니다. 그 위에 세운 "barrier_eda MAE 1.82 vs quadrature 3.5" 논증과 rev 2 표들의 barrier_eda 행은 삭제한다.
- **유효한 서술 (방법 ①):** 장벽은 `d1 + d2 + ΔE_int(자기 basis)`로 정확히 닫힌다. 8채널 합 `d1 + d2 + Σ6ch`와 장벽의 차이는 c_ghost(BSSE + cavity)로, 평균 |Δ| 0.55 kcal/mol, 14.8%가 > 1 kcal/mol — §Rev 3 "SI 한 줄"에만 보고한다. 장벽은 직접 예측(KRR 1.45)으로 보고한다.

## disp 채널은 ML이 아니라 해석적 등식

`b_disp = D3(BJ)/B3LYP inter-fragment dispersion at TS geometry` = 타깃 `dft_disp_dft`와 **정의상 같은 양**. 검증: `mean|b_disp − dft_disp_dft| = 2.4×10⁻⁶ kcal/mol`.

- ESPLEY54/72의 disp 채널 MAE 0.00, r² 1.000은 "Ridge가 예측했다"가 아니라 "feature를 그대로 읽었다".
- 다른 타깃의 feature로 들어가서 도움이 되지만 (제거 시 e_bond +0.02, eint_spe +0.02, barrier −0.00), 이 표에서는 예측 결과로 취급하지 않음.
- ESPLEY46(b^ch 블록 없음)에서만 disp는 실제 ML 예측이며 MAE 1.25 kcal/mol.

## 전하 그룹별 test MAE (ESPLEY73 / KRR, rev 3)

n = 2,148 고유 rxn (5-seed test fold 합집합), MAE는 폴드 예측 전체 기준. 표 값은 `charge_breakdown.csv`(소수 넷째 자리)에서 반올림.

| target | all | neutral (n=2044) | charged (n=104) | q₂=−2 (80) | q₂=+1 (24) |
|---|---:|---:|---:|---:|---:|
| barrier | 1.45 | 1.44 | 1.67 | 1.36 | 2.81 |
| d1 | 1.12 | 1.11 | 1.33 | 1.15 | 2.02 |
| d2 | 0.82 | 0.82 | 0.72 | 0.57 | 1.27 |
| elst | 1.52 | 1.48 | 2.30 | 1.87 | 3.93 |
| Pauli | 2.02 | 1.97 | 2.99 | 2.42 | 5.16 |
| OI | 1.16 | 1.12 | 1.80 | 1.70 | 2.17 |
| CPCM | 1.01 | 0.97 | 1.81 | 1.74 | 2.05 |
| CDS | 0.23 | 0.23 | 0.20 | 0.15 | 0.36 |

- **rev 2 → rev 3 (charged 그룹):** CPCM 2.56 → 1.81, elst 2.40 → 2.30, OI 1.91 → 1.80, Pauli 3.02 → 2.99. 라벨 교정으로 하전 반응의 용매화 채널이 가장 크게 좋아짐.
- **`is_charged` 편입 효과 (rev 1 → rev 2 기록):** charged 그룹 elst 3.00→2.40 (−20%), Pauli 4.66→3.02 (−35%), OI 3.12→1.91 (−39%).
- 절대치는 여전히 중성 대비 1.5–2배. q₂=+1(24 rxn) 극소 표본은 참고용. 원본: [`charge_breakdown.csv`](charge_breakdown.csv).

## 그룹 분할 robustness (dipolarophile / dipole 완전 분리 홀드아웃, rev 3)

동일 KRR + per-seed HP 재사용 (nested CV의 seed별 best). 1,770 unique dipoles / 713 unique dipolarophiles / 5,234 rxn (위생 필터 후). random 열은 같은 스크립트의 분할 기준이라 headline 5-seed MAE와 약간 다르다.

| target | random | dph 그룹 홀드 | 배수 | dipole 그룹 홀드 | 배수 |
|---|---:|---:|---:|---:|---:|
| barrier | 1.50 | 1.64 | 1.10 | 1.51 | 1.01 |
| d1 | 1.12 | 1.09 | 0.97 | 1.11 | 0.99 |
| d2 | 0.83 | 1.00 | 1.20 | 0.85 | 1.02 |
| elst | 1.54 | 1.61 | 1.05 | 1.58 | 1.03 |
| Pauli | 2.06 | 2.10 | 1.02 | 2.08 | 1.01 |
| OI | 1.18 | 1.15 | 0.97 | 1.15 | 0.97 |
| CPCM | 1.02 | 1.06 | 1.04 | 1.01 | 0.99 |
| CDS | 0.23 | 0.26 | 1.13 | 0.24 | 1.05 |

**주의**: 이 그룹 홀드아웃은 *dipole 자체* 또는 *dipolarophile 자체*가 완전히 새로울 때의 성능이지, **골격 클래스가 처음 보는 것**일 때는 아니다. 골격 외삽은 §Audit S3+S8의 `dipole_class` MMP 분할에서 더 냉정하게 관찰됨 — ΔMAE가 랜덤 대비 1.2–1.6배 증가하고 τ95 커버리지도 0.63→0.39로 하락. 원본: [`group_split.csv`](group_split.csv).

## Espley 2024 대비 (ds3 = Coley [3+2] 데이터셋의 3,510 rxn 부분집합)

비교 조건 — Espley 본문·ESI(ChemRxiv v1)와 저자 GitHub(`the-grayson-group/distortion-interaction_ML`)로 2026-09-28 확인.

**Correction (10.1039/D5DD90005K, 2025-02-06):** RSC·미러 모두 403이라 원문을 직접 읽지는 못했다. 검색 색인에 잡힌 본문 요약과 직접 확인한 Bath 데이터 아카이브 v2 공지(BATH-01480, 2025-01-31: "correct an error with ts_100_dft.log in the malonate data set")가 일치한다: 데이터 아카이브의 malonate 데이터셋 파일 하나를 바로잡은 것이고, "data, results and conclusions presented in the paper are unaffected". ds3 수치·반응 수·계산 수준은 바뀌지 않는다 (간접 확인).

- **데이터 — 같은 원천의 부분집합.** ds3는 Stuyver/Jorner/Coley 2023의 `coleygroup/dipolar_cycloaddition_dataset`(ESI S1.3), 즉 우리와 같은 원천 데이터다. 논문에는 반응 수가 없고, 저자 GitHub ML 로그 기준 **3,510 rxn**(Coley 5,269의 약 67%, 선별 기준 미기재; test n = 351). 이전 판의 "Bath [3+2] 3,980 rxn"은 틀렸다.
- **DFT 수준 — 명목상 같음.** ds3의 장벽(ΔE‡, ΔG‡)과 relaxed 반응물 에너지는 Coley 값(B3LYP-D3(BJ)/def2-TZVP//def2-SVP, SMD(water), Gaussian 16)을 그대로 쓰고, 뒤틀린 fragment 단일점만 "같은 수준·같은 용매"로 새로 계산했다(본문 + `get_energies.py`). ESI S3.3 그림 캡션의 "ωB97X-D/def2-TZVP"는 ds3 본문 서술·코드와 맞지 않는다(다른 데이터셋 문구로 보임). Bath 아카이브 메타데이터도 네 데이터셋 전체에 "AM1 // ωB97X-D/def2-TZVP (IEFPCM=Water)"라는 공통 한 줄만 적고 있어, ds3 DFT .log의 route line을 아카이브(`data_archive_files.zip`)에서 직접 확인하면 확정된다. 우리 라벨도 Coley 기하 위 B3LYP-D3(BJ)/def2-TZVP SMD(water)라, 차이는 프로그램(Gaussian vs ORCA)과 세부 설정 정도다.
- **상호작용 정의 — 같음.** Espley는 ΔE_int = ΔE‡ − ΣΔE_dist(counterpoise 없음, EDA 없음)로, 우리 ΔE_int(자기 basis)와 같은 정의다. 그래서 이 표에서만 ΔE_int를 기준선으로 쓴다.
- **반경험 feature — 다름.** Espley는 AM1(Gaussian 16, 암시적 용매) 46개 수동 선택 feature, 모델은 SVR(RBF). AM1 기하를 재최적화했는지는 본문에 없다(코드상 AM1 opt+freq로 추정). 우리는 DFT TS 기하 위 GFN2-xTB/ALPB 단일점(성능 상한).
- **오차 막대 — 다름.** Espley의 ±는 test set 안 |오차|의 표준오차(seed 평균), 우리의 ±는 seed 간 sd라 직접 비교할 수 없다. %range가 그나마 공정하다.

| 양 | Espley pre-ML AM1 | Espley SVR test MAE ± SE (%range) | 이번 pre-ML GFN2/ALPB | 이번 KRR test MAE ± sd (%range) [SVR] |
|---|---:|---:|---:|---:|
| Dipole distortion (d1) | 3.59 | 2.55 ± 0.13 (6.3%) | 4.69 | **1.12 ± 0.03 (2.31%)** [1.12] |
| Dipolarophile distortion (d2) | 3.81 | 2.37 ± 0.12 (6.7%) | 1.48 | **0.82 ± 0.02 (1.93%)** [0.81] |
| Interaction energy (ΔE_int, 기준선) | 20.01 | 2.46 ± 0.12 (6.9%) | 3.39 | **0.77 ± 0.03 (1.93%)** [0.77] |
| ΔE‡ (전자 에너지 장벽) | 23.07 | 3.09 ± 0.14 (6.1%) | 6.23 | **1.45 ± 0.05 (2.46%)** [1.48] |

관찰:
1. 출발점이 다르다. AM1 pre-ML 장벽 MAE 23.07(AM1 상호작용 에너지가 대부분 양수), GFN2/ALPB는 6.23.
2. Dipole distortion의 pre-ML은 오히려 우리가 더 나쁘다(4.69 vs 3.59). ML이 메우는 폭이 여기서 나온다.
3. Espley ΔE‡ test range는 51.0(−13.46 ~ 37.56, 3,510 부분집합)이고 우리는 약 59 — %range 비교가 우리에게 다소 유리하게 작용한다.
4. 같은 SVR끼리 비교해도 우리 쪽이 낮다(ESPLEY73 SVR 1.12 / 0.81 / 0.77 / 1.48). feature 수를 46개로 맞춘 `ESPLEY46` KRR의 장벽은 1.68.
5. **Espley는 변형/상호작용 2분할(+ ΔE‡, ΔG‡)만 예측하며 EDA는 없다.** 상호작용을 elst / Pauli / OI / disp / CPCM / CDS로 나눠 d1, d2와 함께 예측하는 **8채널 분해는 이 연구의 기여**다 (이전 판에서 이를 Espley의 주장으로 적은 것은 오기).

## 가정·주의

- **5 seed의 test 표본은 5,234 랜덤에서 뽑히므로 서로 겹칠 수 있음.** ±sd는 독립 표본 표준오차가 아니라 seed간 변동성.
- rev 2부터 하이퍼파라미터는 각 seed의 자기 train 분할 위 `GridSearchCV(5-fold)`로 별도 튠 (**nested CV**). rev 1의 seed 23 leak 문제는 해소됨.
- 10% validation 분할은 생성만 되고 사용되지 않음. 그대로 정직한 것.
- Feature는 **DFT TS 기하**에서의 xTB 단일점이다 — 성능 상한(upper bound). React-OT 등 생성 기하에서의 배포 성능은 별도 실험이 필요(§S2 후속).

## 산출물 (`results/`)

- `SUMMARY.md` — 이 문서
- `ml_report.json` — 12 target × 3 arm × (Protocol A 4 + Protocol B 5) 전 metric (NMAE, pre_ml_r 포함; rev 3)
- `ml_table_espley.csv` — 324 행 (rev 3)
- `rev3_vs_rev2.csv` — rev 3 표에 rev 2 test MAE / r² 열을 붙인 비교표
- `predictions.parquet` — 5-seed 폴드 예측
- `xtb_features.parquet` — 5,260 rxn × feature + 12 target(`dft_*`, rev 3 라벨) + 메타
- `ml_targets/` — target당 JSON (병렬 array 원본)
- `charge_breakdown.csv` — 전하 그룹별 test MAE (ESPLEY73 / KRR)
- `group_split.csv` — 반응물 그룹 홀드아웃 robustness (동일)
- `split_metrics.csv` — MMP-Δ 분할별 (random / dipole_class / 10 LOSO) dMAE, dom_agree, τ95/τ99 (§Audit S3+S8)
- `margin_calibration.csv` — 예측 margin τ에 따른 coverage / agreement 곡선 (§Audit S9)
- `mmp_pairs_v2.csv` — 5,209 MMP 페어 (r1, r2, kind, sub_from, sub_to)
- `verify_s1s5.json` — S1(gap statistics) + S5(flag counts) 원본 수치
- `figures/` (ESPLEY73 기준, rev 3):
  - `scatter_Ridge/KRR_rbf/SVR_rbf/XGB_ESPLEY73.png` — 9-패널 산점도 (barrier, d1, d2, 6채널)
  - `mae_bar_espley73.png` — 9 target × 4 모델 grouped bar

## 재현

```bash
sbatch analysis/espley_xtb_repro/s01_xtb_array.sh                         # xTB features
sbatch --dependency=afterok:<JID> analysis/espley_xtb_repro/s02_aggregate.sh
sbatch                            analysis/espley_xtb_repro/s03_ml_array.sh
sbatch --dependency=afterok:<JID> analysis/espley_xtb_repro/s04_aggregate_ml.sh
sbatch --dependency=afterok:<JID> analysis/espley_xtb_repro/s05_plot.sh
sbatch --dependency=afterok:<JID> analysis/espley_xtb_repro/s06_analyze_extra.sh
sbatch --dependency=afterok:<JID> analysis/espley_xtb_repro/s07_evaluate_pairs.sh    # outputs to CWD -> move into results/
```

라벨만 바뀐 경우(rev 3처럼)는 s01/s02 대신 타깃만 교체한다:
`python refresh_targets.py labels_all.json <ROOT>/xtb_features.parquet <ROOT>/xtb_features.parquet` (sbatch로 실행) → s03 smoketest → s03_ml_array → s04 → s05 → s06/s07.

Prerequisites: xtb 6.7.1 in `/home1/yeseo1ee/xtb-dist/`, `reactot` env with dftd3 + morfeus-ml + xgboost, `labels_all.json` (5,265 records, 5,260 accepted).

---

## Audit (2026-09-15, FIXES.md; 2026-09-28 rev 3 갱신)

S1 / S3+S8 / S5 / S9는 rev 3 라벨·결과로 갱신했다 (원본: `split_metrics.csv`, `margin_calibration.csv`, `labels_all.json`). `verify_s1s5.json`은 rev 2 시점의 S1·S5 원본 수치(job 984368)로 남겨 둔다.

### S1 — ASM identity vs 8-channel sum (rev 3에서 정정)

두 관계는 별개다.

- **ASM identity** `d1 + d2 + ΔE_int(자기 basis) ≡ ΔE‡`: 라벨링에서 정의로 강제됨. `max|residual| = 8.6 × 10⁻¹⁰ kcal/mol` — 완전 등식.
- **8-channel sum** `d1 + d2 + Σ(elst, Pauli, OI, disp, CPCM, CDS)`는 장벽과 다르다. 차이는 `c_ghost = ΔE_int − E_bond`(+ ORCA 표 closure ≤ 0.11):

  | 통계 (5,260 rxn) | rev 3 (SMD relabel) | rev 1/2 (CPCM 시기 라벨) |
  |---|---:|---:|
  | mean `ΔE_int − E_bond` | +0.10 kcal/mol | −3.39 |
  | sd | 0.77 | 3.69 |
  | `|·| > 5 kcal/mol` | 2 | 1,923 (36.6%) |
  | 평균, charge2 = 0 / −2 / +1 | +0.11 / −0.10 / +0.28 | −3.97 / +9.55 / +2.42 |
  | Pearson r(·, charge2) | +0.055 | −0.617 |
  | 8채널 합 vs 장벽: 평균 \|Δ\|, > 1 kcal/mol | 0.55, 780 (14.8%) | — |

- **정정.** rev 1/2 문서는 −3.4 kcal/mol 편향의 원인을 "GFN2-xTB의 diffuse 함수 부재 + EDA-NOCV 스킴이 이온 짝에서 새는 구조적 문제"라고 적었으나 **틀렸다.** 라벨 계산에는 xTB가 전혀 쓰이지 않는다. 실제 원인은 CPCM 시기 **라벨 계산의 버그**였다: SMD를 `%cpcm smd true` 블록으로 켰는데 ORCA가 이 블록을 EDA가 자동 생성하는 fragment 입력에 넘기지 않아, 복합체는 SMD / fragment SCF(`eda_frag*.out`)는 SMD 없는 CPCM으로 계산됐다. 그 결과 E_bond와 6채널이 자기 basis 기준(d1, d2, ΔE_int)과 다른 용매화 수준의 fragment를 기준으로 삼았고, 전하 반응에서 차이가 커졌다. SMD 재계산(rev 3, `label_true/SMD_RELABEL.md`) 후 남는 차이는 c_ghost(ghost-basis BSSE + AB cavity)뿐이며 전하와 거의 무상관이다 (r = +0.055).
- 원본: `rev3_doc_numbers.py` (sbatch 실행, rev 2 값은 `rev2_backup_20260915` feature 파일).
- **결론.** 8채널 합과 장벽의 차이는 계산 방법이 남긴 흔적(c_ghost)이라 연구 내용과 무관하며, §Rev 3의 "SI 한 줄"로만 보고한다. 장벽 보고는 직접 예측으로 한다.

### S3 + S8 — MMP-Δ 분할별 채널 예측 (rev 3)

`ESPLEY73` 위 KRR(RBF) OOF. 페어당 두 멤버가 같은 fold(OOF) 인 경우만 평가. Δ = pred(r2) − pred(r1), 4 채널(elst / Pauli / OI / CPCM) + d1 + d2 + disp × 지배 채널 지표(`dom_agree_all` = |Δ|이 가장 큰 채널이 true와 pred에서 일치할 확률).

| split | n pairs | ΔMAE elst | ΔMAE Pauli | ΔMAE OI | ΔMAE CPCM | dom_agree | τ₉₅ | cov₉₅ | τ₉₉ | cov₉₉ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **random** | 1,066 | 1.79 | 2.54 | 1.39 | 1.12 | **0.826** | 2.5 | 0.63 | 5.0 | 0.42 |
| **dipole_class** (골격 외삽) | 4,325 | 2.60 | **3.99** | 2.06 | 1.39 | 0.749 | 5.0 | 0.39 | — | — |
| loso:*C | 157 | 2.33 | 3.53 | 1.54 | 1.19 | 0.885 | 3.5 | 0.66 | 6.5 | 0.42 |
| loso:*c1ccccc1 | 383 | 1.72 | 2.75 | 1.49 | 1.02 | 0.854 | 1.5 | 0.75 | 3.5 | 0.53 |
| loso:*NC | 767 | 1.74 | 2.55 | 1.48 | 1.05 | 0.820 | 2.5 | 0.65 | 4.5 | 0.50 |
| loso:*C(=O)NC | 767 | 1.74 | 2.55 | 1.48 | 1.05 | 0.820 | 2.5 | 0.65 | 4.5 | 0.50 |
| loso:*C(C)=O | 1,203 | 1.69 | 2.43 | 1.36 | 1.01 | 0.886 | 1.5 | 0.83 | 4.5 | 0.61 |
| loso:*C#N | 1,254 | 2.11 | 2.96 | 1.53 | 1.14 | **0.880** | 1.5 | 0.84 | 3.5 | 0.67 |
| loso:*C(=O)OC | 765 | 1.84 | 2.43 | 1.43 | 0.98 | 0.848 | 2.0 | 0.75 | 4.5 | 0.55 |
| loso:*OC | 765 | 1.84 | 2.43 | 1.43 | 0.98 | 0.848 | 2.0 | 0.75 | 4.5 | 0.55 |
| loso:*[N-]c1ccccc1 | 90 | 1.97 | 2.39 | 1.51 | 1.07 | 0.867 | 3.0 | 0.73 | 4.5 | 0.63 |
| loso:*[N+]#CC(C)=O | 169 | 1.74 | 2.82 | 1.40 | 0.95 | 0.864 | 3.0 | 0.67 | 5.0 | 0.46 |

관찰:
1. **`dipole_class` 골격 외삽에서 dom_agree가 0.826 → 0.749로 하락, ΔMAE Pauli가 2.54 → 3.99 (×1.57)**. 랜덤 분할이 성능을 낙관적으로 표시함이 정량 확인됨.
2. **LOSO(치환기 홀드아웃)는 예상보다 견고** — 우리 결과: 치환기 하나를 통째로 뺀 10개 분할 모두 dom_agree 0.82–0.89로 random(0.826)과 비슷하거나 오히려 높다. 이 데이터에서는 새 치환기로의 외삽이 골격 외삽(dipole_class 0.749)보다 훨씬 쉽다. (이전 판은 이를 Espley 논문 주장의 재현으로 적었으나 오기 — Espley는 [3+2]에서 그런 주장을 하지 않았고, 외부 검증은 Diels-Alder 문헌 두 세트에서만 했다.)
3. **dipole_class 분할에서 τ₉₉는 미달** — τ = 8.0에서도 agreement 0.977. 골격 외삽은 99%로는 안전하지 않음.
4. rev 2 대비 CPCM ΔMAE가 전 분할에서 크게 감소 (random 1.62 → 1.12) — 라벨 교정 효과.

원본: [`split_metrics.csv`](split_metrics.csv), [`mmp_pairs_v2.csv`](mmp_pairs_v2.csv).

### S5 — 라벨 위생 플래그

`labels_all.json` 5,260 acceptance 위 (d1/d2와 메타데이터는 rev 3에서 바뀌지 않음):

| 플래그 | count | % |
|---|---:|---:|
| `flag_negative_strain` (d1 < −0.5 OR d2 < −0.5) | 9 | 0.17 |
| d1 < 0 | 7 | 0.13 |
| d2 < 0 | 7 | 0.13 |
| d2 > 50 kcal/mol | 12 | 0.23 |
| `flag_async` (async > 0.5) | 228 | 4.33 |
| `alt_used = 1` (relaxed 기준으로 `r*_alt.xyz` 사용) | **3,033** | **57.66** |
| charged (총전하 ≠ 0) | 258 | 4.90 |

- **위생 필터 (rev 2부터):** d1 < 0 (7) ∪ d2 < 0 (7) ∪ d2 > 50 (12) = **26행을 학습에서 제외** (5,260 → 5,234). 음의 변형과 d2 > 50은 모두 여기에 포함된다 (이전 판의 "학습에서 빼지 않음" 서술은 rev 1 시점 기록이라 삭제).
- **`alt_used`는 BSSE와 무관하다.** Coley 데이터가 반응물마다 `r*.xyz`와 함께 제공하는 `r*_alt.xyz`(TS 입체화학과 맞는 반응물 conformer/tautomer, Coley의 G_act 기준)를 relaxed fragment 기준(`frag*_rel`)으로 썼다는 표시다. 이전 판의 "BSSE re-partition / fallback" 설명은 틀렸다. `_alt`를 쓰면 Coley CSV의 G_act를 3,037 rxn 모두 < 0.01 kcal/mol로 재현한다 (`label_true/SPEC.md` §3.4).
- Async 228건(4.3%)은 대체로 이온·강한 편극 반응. group_split 분석의 `charged` 그룹과 일부 겹침.

원본: `labels_all.json`, `label_true/scripts/stage1_build_inputs.py` (rev 2 시점 수치: [`verify_s1s5.json`](verify_s1s5.json)).

### S9 — Margin gate 커브 (rev 3)

배포 시 dominant channel(가장 큰 |Δ|)만이 필요하고, **|Δ|의 top-1 − top-2 차 (`m_pred`)** 를 gate로 쓰면 신뢰 예측만 선별할 수 있다. `random` split 기준:

| τ (kcal/mol) | coverage | agree | true-margin agree |
|---:|---:|---:|---:|
| 0.0 | 1.00 | 0.826 | 0.826 |
| 1.0 | 0.82 | 0.906 | 0.893 |
| 2.0 | 0.69 | 0.942 | 0.937 |
| 3.0 | 0.59 | 0.962 | 0.963 |
| **3.5** | **0.54** | **0.971** | 0.974 |
| 5.0 | 0.42 | 0.993 | 0.983 |
| 6.5 | 0.34 | **0.997** | 0.990 |

- **τ = 3.0에서 59%의 페어가 통과하고 96%가 지배채널을 맞춤.** τ = 5.0에서는 42%가 통과하고 99.3%로 안전.
- 지배채널 예측 자체는 재현 가능하고 (agree 곡선이 true-margin 곡선과 거의 겹침), **모델 τ가 실 τ의 신뢰할 만한 대리치**.
- `dipole_class` 골격 외삽에서 커브는 아래로 이동 — τ = 5.0에서 coverage 0.39 / agree 0.954, τ = 8.0에서도 agree 0.977. 골격 밖에선 gate 임계를 더 높이 잡을 것.

원본: [`margin_calibration.csv`](margin_calibration.csv).

### S4 재검토 — rev 2에서 해소 (결과 아래)

이전 커밋(`b5e46998`)에서 "S4는 이미 만족"이라고 서술했으나 두 지점에서 성립하지 않았다:

1. **단일-seed 튜닝의 test 누출**: 이전 [`train_ml_single.py:106-112`](../train_ml_single.py#L106-L112) 는 seed 23의 train fold에서만 GridSearchCV(5-fold)를 돌린 뒤 그 best HP를 다른 4개 seed에 재사용. **다른 seed의 test 표본 일부가 seed 23의 튜닝 데이터에 이미 들어갔을 수 있어** 편향이 남았음.
2. **격자 모서리에서 stop**: rev 1 grid는 KRR α∈[1e-3, 1], γ∈[1e-3, 1]. 11 타깃 전부 best가 α=1e-3, γ=1e-3(둘 다 최솟값 = 격자 경계).

**rev 2에서 두 문제 모두 해소**:

- **N3 nested CV** — GridSearchCV를 각 seed의 자기 train fold에서 별도로 실행 (5 seeds × 4 model × 3 feature-set × 12 target).
- **N1 격자 확장** — KRR α[1e-5..1], γ[3e-4..1], Ridge α[1e-4..1e3], SVR γ에 1e-3 추가.

**결과** (KRR 헤드라인 아렌드, ESPLEY73):

| 타깃 | rev 1 α, γ | rev 2 α, γ (per-seed) | rev 1 MAE | rev 2 MAE |
|---|---|---|---:|---:|
| Pauli | 1e-3, 1e-3 (5/5) | {1e-5, 1e-4}, {3e-4, 1e-3} | 2.17 | **2.00** (−7.8%) |
| elst | 1e-3, 1e-3 (5/5) | {1e-5, 1e-4}, {3e-4, 1e-3} | 1.68 | **1.62** (−3.6%) |
| OI | 1e-3, 1e-3 (5/5) | {1e-5}, {3e-4} | 1.30 | **1.23** (−5.4%) |
| barrier | 1e-3, 1e-3 (5/5) | 1e-3, 1e-3 (5/5) | 1.49 | 1.45 |
| barrier_eda (rev 2 전용, 방법 ② 혼합량 → rev 3에서 철회) | — | 1e-3, 1e-3 (5/5) | — | (1.82) |
| CPCM | 1e-3, 1e-3 (5/5) | {1e-3, 1e-2}, {1e-3, 3e-3} | 1.48 | 1.46 |

**grid_edge_hits 총계** (12 target × 3 arm × 4 model × 5 seed × param 수 = 1,620개 best-param slot):

| 모델 | 격자 경계 hit | % |
|---|---:|---:|
| Ridge | 50/180 | 27.8 |
| **KRR_rbf** (헤드라인) | **67/360** | **18.6** |
| SVR_rbf | 365/540 | 67.6 |
| XGB | 435/540 | 80.6 |

- KRR 헤드라인은 18.6%로 대부분 grid 내부에서 결정됨. 경계 히트의 대부분은 여전히 **α=1e-5 / γ=3e-4** (Pauli/OI/elst의 최적 지점) — 추가 확장 여지는 있으나 개선폭이 5% 미만으로 diminishing.
- SVR/XGB는 격자 경계 히트가 높지만 헤드라인이 아니므로 결과에는 영향 없음. 후속 실험에서 격자 추가 확장이 필요하면 관리.

### rev 2에서 처리 완료 (B2 + N1–N3 + S7)

- ~~**B2 `dft_barrier_eda`**: 8-채널 닫힌 총량을 라벨/타깃으로 추가.~~ — **rev 3에서 철회**: d1 + d2 + e_bond는 방법 ② 혼합량이라 장벽이 아님 (§B2 철회 참조).
- **N1 격자 확장**: KRR α[1e-5..1], γ[3e-4..1]. Pauli/OI/elst 최적점이 격자 안쪽으로 이동해 5% 개선.
- **N2 y 표준화**: `TransformedTargetRegressor(StandardScaler)` 로 y를 표준화한 뒤 학습 → 예측 시 역변환.
- **N3 nested CV**: per-seed GridSearchCV 로 seed 23의 튜닝 leak 제거.
- **hygiene 필터**: d1<0(7) ∪ d2<0(7) ∪ d2>50(12) → 총 26행 학습 제외 (5,260 → 5,234).
- **S7 `is_charged`**: AUX19에 이진 지표 추가 → charged 그룹의 Pauli/OI/elst MAE 20–39% 감소.

### 남은 후속 (S2)

- **S2 — xTB 기하 민감도**: 현재는 DFT TS 기하 위 xTB SPE(성능 상한). `xtb --opt` 로 로컬 최적화한 기하에서 재추출 후 5개 seed 재학습, 배포 성능 확인. 5,260 × 3 구조 최적화(~1분/구조) + full SPE 재계산 ≈ 새 array job.
- ~~S6 — 채널 sum 재조정 실험~~ — **철회**: S1의 잘못된 원인 진단(xTB diffuse 함수)에 근거한 계획이었다. 실제 원인은 라벨 계산 버그였고 rev 3 SMD 재계산으로 해결됐다 (§S1 정정).
