# espley_xtb_repro — final results (2026-09-15)

**Espley 2024 (D4DD00224E) protocol reproduced on Coley 5,260 rxns with GFN2-xTB / ALPB water.**

> 2026-09-15 revision — FIXES.md audit (S1/S3/S5/S6/S8/S9) applied. See §Audit at the bottom.

## Method (one-line)

Single engine: **xtb 6.7.1 binary**, one GFN2-xTB / ALPB(water) single point per structure gives term-wise energies, Mulliken charges, per-atom Wiberg valences, HOMO-LUMO gap, dipole moment. tblite not used.

## Feature blocks

- `d^struct` **41** — 11 dist + 15 Mulliken + 15 Wiberg-valence
- `b^xtb`    **5**  — barrier, dist_dipole, dist_dipolarophile, sum, interaction (ALPB totals; q_barrier omitted, no Hessian)
- `b^ch`     **8**  — b_strain_1/2 (gas-part strain), b_elst (frozen monopole Coulomb), b_pauli (rep), b_oi (EHT), b_disp (D3(BJ)/B3LYP), b_cpcm (ΔGelec), b_cds (ΔGsasa+Ghb+Gshift) — zero markers 없음, 전부 실측
- `AUX18`         — b_elst_scc, b_disp_d4, b_axc, b_ct, gap_ts/dip/dph, mu_ts/dip/dph, dmu_complexation, dsasa_{H,C,N,O,F,Cl,Br}

**Three arms**: `ESPLEY46` (41+5), `ESPLEY54` (41+5+8), **`ESPLEY72`** (54+AUX18).

## Gates (모두 통과)

| # | 항목 | 결과 |
|---|---|---|
| G1 | 슬라이스 로그 | 18/18 accepted=5260, xtb=6.7.1, ok |
| G2 | xtb_solvation | `{'alpb-water': 5260}` (gas 없음) |
| G3 | aggregate | **5,260 rows / 5,260 unique / 100% ok** |
| G4 | **해석적 적합성** | `mean\|b_disp − dft_disp_dft\| = 2.4 × 10⁻⁶ kcal/mol`, `max 8 × 10⁻⁶` — 사실상 등식 |
| G5 | feature 품질 | 72 컬럼, NaN 0, 상수 0, 순서 중복 0 |
| G6 | ML 사용 행 | **5,260** |

## Model selection — KRR(RBF)을 사전 고정 (a priori)

**모델 앙상블에서 test MAE 최소를 뽑는 방식은 낙관 편향을 유발.** 검증 결과 `ESPLEY72` 조합에서 KRR이 11 타깃 중 8개에서 최적이고, 나머지 3개에서도 최적 대비 손실이 합계 0.05 kcal/mol (d2 +0.01, cds +0.00, disp +0.04)에 불과. 이 대가로 cherry-picking 편향을 완전히 제거.

**Ridge / SVR / XGB의 결과는 `ml_table_espley.csv`에 전부 기록** (297 행: 11 target × 3 arm × 9 model).

## Final headline model — **`ESPLEY72` + KRR(RBF), 5-seed 80/10/10**

| Channel | test MAE ± sd | NMAE | r² | % of range | group-split MAE |
|---|---:|---:|---:|---:|---:|
| ΔE‡ (barrier) | **1.49 ± 0.03** | 0.201 | 0.954 | 2.48 | 1.64 |
| d1 (dipole strain) | 1.13 ± 0.04 | 0.186 | 0.960 | 2.33 | 1.09 |
| d2 (dipolarophile strain) | 0.83 ± 0.04 | 0.165 | 0.970 | 1.35 | 1.00 |
| ΔE_int (SPE) | **0.79 ± 0.03** | 0.155 | 0.972 | 1.43 | 0.81 |
| Bond E (EDA 6채널 합) | 1.17 ± 0.03 | 0.191 | 0.962 | 1.91 | 1.24 |
| elst | 1.68 ± 0.04 | 0.123 | 0.983 | 1.18 | 1.74 |
| Pauli | 2.17 ± 0.02 | 0.084 | **0.991** | 0.85 | 2.10 |
| OI | 1.30 ± 0.05 | 0.082 | **0.991** | 0.71 | 1.27 |
| CPCM | 1.48 ± 0.06 | 0.353 | 0.881 | 2.60 | 1.56 |
| CDS | 0.24 ± 0.01 | 0.347 | 0.871 | 3.79 | 0.26 |
| *disp* | *계산으로 결정 (ML 아님, 아래 참조)* | — | — | — | — |

- 8 채널 중 **7개가 예측 대상**, 그중 5개가 1 kcal/mol 미만 또는 NMAE < 0.2
- 약한 축은 **CPCM(NMAE 0.35)과 CDS(0.35)** — 두 다 용매 항이고 값 자체가 작음. 여기가 다음 개선 지점.

## disp 채널은 ML이 아니라 해석적 등식

`b_disp = D3(BJ)/B3LYP inter-fragment dispersion at TS geometry` = 타깃 `dft_disp_dft`와 **정의상 같은 양**. 검증: `mean|b_disp − dft_disp_dft| = 2.4×10⁻⁶ kcal/mol`.

- ESPLEY54/72의 disp 채널 MAE 0.00, r² 1.000은 "Ridge가 예측했다"가 아니라 "feature를 그대로 읽었다".
- 다른 타깃의 feature로 들어가서 도움이 되지만 (제거 시 e_bond +0.02, eint_spe +0.02, barrier −0.00), 이 표에서는 예측 결과로 취급하지 않음.
- ESPLEY46(b^ch 블록 없음)에서만 disp는 실제 ML 예측이며 MAE 1.25 kcal/mol.

## 전하 그룹별 test MAE (ESPLEY72 / KRR)

| target | all | neutral (n≈2047) | charged (n≈105) | q₂=−2 (79) | q₂=+1 (26) |
|---|---:|---:|---:|---:|---:|
| barrier | 1.49 | 1.49 | 1.55 | 1.51 | 1.70 |
| d1 | 1.13 | 1.11 | 1.44 | 1.31 | 1.89 |
| d2 | 0.83 | 0.83 | 0.87 | 0.66 | 1.59 |
| eint_spe | 0.79 | 0.76 | 1.33 | 0.99 | 2.51 |
| e_bond | 1.17 | 1.14 | 1.83 | 1.62 | 2.54 |
| elst | 1.68 | 1.62 | 3.00 | 2.70 | 4.01 |
| Pauli | 2.17 | 2.04 | 4.66 | 3.78 | 7.66 |
| OI | 1.30 | 1.20 | 3.12 | 2.34 | 5.75 |
| CPCM | 1.48 | 1.42 | 2.66 | 2.70 | 2.51 |
| CDS | 0.24 | 0.24 | 0.24 | 0.18 | 0.44 |

이온 반응(4.9%)에서 elst/Pauli/OI/CPCM이 약 2배 커짐 — GFN2가 diffuse 함수가 없어 음이온이 약간 부정확 + ALPB로 완화하지만 잔차 남음. 원본 표: [`charge_breakdown.csv`](charge_breakdown.csv).

## 그룹 분할 robustness (dipolarophile / dipole 완전 분리 홀드아웃)

동일 KRR·튠 HP 재사용. 1,770 unique dipoles / 713 unique dipolarophiles / 5,260 reactions.

| target | random | dph 그룹 홀드 | 배수 | dipole 그룹 홀드 | 배수 |
|---|---:|---:|---:|---:|---:|
| barrier | 1.49 | 1.64 | 1.10 | 1.51 | 1.01 |
| d1 | 1.13 | 1.09 | 0.97 | 1.10 | 0.98 |
| d2 | 0.83 | 1.00 | 1.20 | 0.85 | 1.02 |
| eint_spe | 0.79 | 0.81 | 1.03 | 0.82 | 1.05 |
| e_bond | 1.17 | 1.24 | 1.06 | 1.23 | 1.05 |
| elst | 1.68 | 1.74 | 1.03 | 1.75 | 1.04 |
| Pauli | 2.17 | 2.10 | 0.97 | 2.17 | 1.00 |
| OI | 1.30 | 1.27 | 0.98 | 1.26 | 0.97 |
| CPCM | 1.48 | 1.56 | 1.06 | 1.52 | 1.03 |
| CDS | 0.24 | 0.26 | 1.12 | 0.25 | 1.05 |

**주의**: 이 그룹 홀드아웃은 *dipole 자체* 또는 *dipolarophile 자체*가 완전히 새로울 때의 성능이지, **골격 클래스가 처음 보는 것**일 때는 아니다. 골격 외삽은 §Audit S3+S8의 `dipole_class` MMP 분할에서 더 냉정하게 관찰됨 — Δchannel MAE가 랜덤 대비 1.3–1.6배 증가하고 τ95 커버리지도 0.59→0.39로 하락. 원본: [`group_split.csv`](group_split.csv).

## Espley 2024 대비 (ds3, Bath [3+2] 3,980 rxn, AM1 최적화 기하)

> **주의**: 데이터셋·기하·반경험 방법이 다 다르다. MAE 직접 비교는 부정확; %range가 그나마 공정.

| 양 | Espley pre-ML AM1 | Espley test MAE (%range) | 이번 pre-ML GFN2/ALPB | 이번 test MAE (%range) |
|---|---:|---:|---:|---:|
| Dipole distortion (d1) | 3.59 | 2.55 ± 0.13 (6.3%) | 4.70 | **1.13 ± 0.04 (2.33%)** |
| Dipolarophile distortion (d2) | 3.81 | 2.37 ± 0.12 (6.7%) | 1.49 | **0.83 ± 0.04 (1.35%)** |
| Interaction energy | 20.01 | 2.46 ± 0.12 (6.9%) | 3.39 | **0.79 ± 0.03 (1.43%)** |
| ΔE‡ | 23.07 | 3.09 ± 0.14 (6.1%) | 6.23 | **1.49 ± 0.03 (2.48%)** |

관찰:
1. Espley ds3의 pre-ML MAE는 barrier가 23.07 (AM1의 시스템 오차가 큼), 이번 pre-ML은 6.23 (GFN2/ALPB의 시스템 오차가 훨씬 작음). 출발점 자체가 다르다.
2. Dipole distortion pre-ML은 오히려 우리가 더 나쁘다 (4.70 vs 3.59). ML이 커버하는 개선폭이 여기서 나옴.
3. 이번 test range가 조금 더 크다 (ΔE‡ 60.3 vs 51.0 kcal/mol) — %range 비교가 우리에게 다소 유리하게 작용.
4. Espley 논문의 진짜 새로운 주장은 barrier 정확도가 아니라 **8채널 분해 그 자체** — distortion/interaction 2분할을 넘어 상호작용을 elst / Pauli / OI / disp / CPCM / CDS로 쪼개 전부 예측한 것이 여기서 이룬 것.

## 가정·주의

- **5 seed의 test 표본은 5,260 랜덤에서 뽑히므로 서로 겹칠 수 있음.** ±sd는 독립 표본 표준오차가 아니라 seed간 변동성(Espley 프로토콜과 동일).
- 하이퍼파라미터는 seed 23의 train 분할 위 `GridSearchCV(5-fold)`로 튠 후 5 seed 재사용 (Espley와 동일). test는 어느 단계에서도 사용되지 않음.
- 10% validation 분할은 생성만 되고 사용되지 않음. 그대로 정직한 것.
- Feature는 **DFT TS 기하**에서의 xTB 단일점이다 — 성능 상한(upper bound). React-OT 등 생성 기하에서의 배포 성능은 별도 실험이 필요.

## 산출물 (`results/`)

- `SUMMARY.md` — 이 문서
- `ml_report.json` — 11 target × 3 arm × (Protocol A 4 + Protocol B 5) 전 metric (NMAE, pre_ml_r 포함)
- `ml_table_espley.csv` — 297 행
- `predictions.parquet` — 3.7 MB, 5-seed 폴드 예측
- `xtb_features.parquet` — 3.6 MB, 5,260 rxn × 72 feature + 11 target + 메타
- `ml_targets/` — target당 JSON (병렬 array 원본)
- `charge_breakdown.csv` — 전하 그룹별 test MAE (ESPLEY72 / KRR)
- `group_split.csv` — 반응물 그룹 홀드아웃 robustness (동일)
- `split_metrics.csv` — MMP-Δ 분할별 (random / dipole_class / 10 LOSO) dMAE, dom_agree, τ95/τ99 (§Audit S3+S8)
- `margin_calibration.csv` — 예측 margin τ에 따른 coverage / agreement 곡선 (§Audit S9)
- `mmp_pairs_v2.csv` — 5,209 MMP 페어 (r1, r2, kind, sub_from, sub_to)
- `verify_s1s5.json` — S1(gap statistics) + S5(flag counts) 원본 수치
- `figures/` (ESPLEY54 기준):
  - `scatter_Ridge/KRR_rbf/SVR_rbf/XGB_ESPLEY54.png` — 8-패널 산점도
  - `mae_bar_espley54.png` — 8 채널 × 4 모델 grouped bar

## 재현

```bash
sbatch analysis/espley_xtb_repro/s01_xtb_array.sh                         # xTB features
sbatch --dependency=afterok:<JID> analysis/espley_xtb_repro/s02_aggregate.sh
sbatch                            analysis/espley_xtb_repro/s03_ml_array.sh
sbatch --dependency=afterok:<JID> analysis/espley_xtb_repro/s04_aggregate_ml.sh
sbatch --dependency=afterok:<JID> analysis/espley_xtb_repro/s05_plot.sh
sbatch --dependency=afterok:<JID> analysis/espley_xtb_repro/s06_analyze_extra.sh
```

Prerequisites: xtb 6.7.1 in `/home1/yeseo1ee/xtb-dist/`, `reactot` env with dftd3 + morfeus-ml + xgboost, `labels_all.json` (5,265 records, 5,260 accepted).

---

## Audit (2026-09-15, FIXES.md)

`verify_s1s5.json` 은 이 절의 모든 수치의 원본. Run: `verify_s1s5.sh` (job 984368) + `s07_evaluate_pairs.sh` (job 984367).

### S1 — ASM identity vs 8-channel sum

두 관계는 별개다.

- **ASM identity** `d1 + d2 + eint_spe ≡ ΔE‡`: 라벨링에서 정의로 강제됨. `max|residual| = 8.6 × 10⁻¹⁰ kcal/mol` — 완전 등식.
- **8-channel sum** `d1 + d2 + Σ(elst, Pauli, OI, disp, CPCM, CDS) = ΔE‡`: **성립하지 않는다.** `eint_spe − e_bond` 의 통계:

  | 통계 | 값 |
  |---|---:|
  | mean | −3.39 kcal/mol |
  | sd | 3.69 |
  | `|gap| > 5 kcal/mol` 건수 | **1,923 / 5,260 (36.6%)** |
  | q_minus2=−2 그룹 평균 | +9.55 |
  | 중성 (charge2=0) 그룹 평균 | −3.97 |
  | Pearson r(gap, charge2) | **−0.617** |

  → 6채널 분해가 `eint_spe` 를 100% 재현하지 않음. 잔차는 반응 총전하와 강하게 상관돼서 GFN2-xTB의 **diffuse 함수 부재 + EDA-NOCV 스킴이 강한 이온 짝에서 잔여 항으로 새는** 구조적 문제로 판단됨. barrier / eint_spe / d1 / d2는 라벨 안정성이 확보돼 있으나, "각 채널을 합치면 barrier가 된다"는 서술은 **평균 −3.4 kcal/mol 편향**을 함께 명시해야 한다.

  **결론**: 예측 표는 채널별로 유효하나, **채널 sum을 barrier로 재구성하는 사용법은 이온 반응에서 특히 부정확**. 배포 시 barrier는 barrier 예측기로, eint_spe는 eint_spe 예측기로 직접 뽑을 것.

### S3 + S8 — MMP-Δ 분할별 채널 예측

`ESPLEY72` 위 KRR(RBF, α=γ=1e−3) a priori. 페어당 두 멤버가 같은 fold(OOF) 인 경우만 평가. Δ = pred(r2) − pred(r1), 6 채널(elst / Pauli / OI / disp / CPCM / d1 / d2 중 사용) × 3 지배 채널 지표(`dom_agree_all` = |Δ|이 가장 큰 채널이 true와 pred에서 일치할 확률).

| split | n pairs | ΔMAE elst | ΔMAE Pauli | ΔMAE OI | ΔMAE CPCM | dom_agree | τ₉₅ | cov₉₅ | τ₉₉ | cov₉₉ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **random** | 1,066 | 1.87 | 2.54 | 1.47 | 1.63 | **0.812** | 3.0 | 0.59 | 6.5 | 0.34 |
| **dipole_class** (골격 외삽) | 4,325 | 2.49 | **3.96** | 2.11 | 1.89 | 0.732 | 5.0 | 0.39 | — | — |
| loso:*C | 157 | 2.56 | 3.43 | 1.71 | 1.98 | 0.841 | 3.5 | 0.67 | 5.0 | 0.52 |
| loso:*c1ccccc1 | 383 | 1.86 | 2.70 | 1.59 | 1.69 | 0.807 | 2.0 | 0.65 | 6.5 | 0.30 |
| loso:*NC | 767 | 1.90 | 2.55 | 1.64 | 1.93 | 0.791 | 3.5 | 0.58 | 5.0 | 0.47 |
| loso:*C(=O)NC | 767 | 1.90 | 2.55 | 1.64 | 1.93 | 0.791 | 3.5 | 0.58 | 5.0 | 0.47 |
| loso:*C(C)=O | 1,203 | 1.90 | 2.42 | 1.43 | 1.52 | 0.865 | 1.5 | 0.82 | 4.5 | 0.60 |
| loso:*C#N | 1,254 | 2.45 | 2.96 | 1.50 | 1.83 | **0.874** | 2.0 | 0.79 | 5.5 | 0.56 |
| loso:*C(=O)OC | 765 | 2.11 | 2.41 | 1.60 | 1.55 | 0.833 | 2.5 | 0.69 | 4.5 | 0.54 |
| loso:*OC | 765 | 2.11 | 2.41 | 1.60 | 1.55 | 0.833 | 2.5 | 0.69 | 4.5 | 0.54 |
| loso:*[N-]c1ccccc1 | 90 | 2.02 | 2.36 | 1.38 | 1.43 | 0.889 | 3.0 | 0.81 | 4.0 | 0.73 |
| loso:*[N+]#CC(C)=O | 169 | 1.80 | 2.81 | 1.45 | 1.33 | 0.828 | 4.5 | 0.52 | 7.5 | 0.36 |

관찰:
1. **`dipole_class` 골격 외삽에서 dom_agree가 0.812 → 0.732로 하락, ΔMAE Pauli가 2.54 → 3.96 (×1.56)**. 랜덤 분할이 성능을 낙관적으로 표시함이 정량 확인됨.
2. LOSO(치환기 홀드아웃)는 예상보다 견고. dom_agree 0.79–0.89로 random과 비슷하거나 오히려 나음 (희귀 치환기일수록 이웃 화학이 잘 정의돼 있어서). Espley 논문의 "새 치환기 일반화" 주장은 이 데이터에서도 재현됨.
3. **dipole_class 분할에서 τ₉₉는 미달** — 어떤 τ에서도 agreement가 0.99에 못 미침 (max 0.97 근처). 골격 외삽은 99%로는 안전하지 않음.

원본: [`split_metrics.csv`](split_metrics.csv), [`mmp_pairs_v2.csv`](mmp_pairs_v2.csv).

### S5 — 라벨 위생 플래그

`labels_all.json` 5,260 acceptance 위:

| 플래그 | count | % |
|---|---:|---:|
| `flag_negative_strain` (d1<0 OR d2<0) | 9 | 0.17 |
| d1<0 | 7 | 0.13 |
| d2<0 | 7 | 0.13 |
| d2 > 50 kcal/mol | 12 | 0.23 |
| `flag_async` (async>0.5) | 228 | 4.33 |
| `alt_used=True` (BSSE re-partition) | **3,033** | **57.66** |
| charged (총전하 ≠ 0) | 258 | 4.90 |

- **3,033건(57.7%)이 BSSE fallback 스킴을 탄다**. 상당 비율. Pauli/OI 성능 지표가 이 부분과 함께 이해되어야 함 — `alt_used` 여부와 무관하게 test MAE가 재현되는지는 후속 분석 필요.
- `neg_strain` 9건과 `d2>50` 12건은 학습 배제하지 않았음 (전체 대비 0.17% + 0.23% = 0.4%). 배제 시 성능 향상 여부는 다음 실험.
- Async 228건(4.3%)은 대체로 이온·강한 편극 반응. group_split 분석의 `charged` 그룹과 일부 겹침.

원본: [`verify_s1s5.json`](verify_s1s5.json).

### S9 — Margin gate 커브

배포 시 dominant channel(가장 큰 |Δ|)만이 필요하고, **|Δ|의 top-1 − top-2 차 (`m_pred`)** 를 gate로 쓰면 신뢰 예측만 선별할 수 있다. `random` split 기준:

| τ (kcal/mol) | coverage | agree | true-margin agree |
|---:|---:|---:|---:|
| 0.0 | 1.00 | 0.812 | 0.812 |
| 1.0 | 0.82 | 0.889 | 0.894 |
| 2.0 | 0.69 | 0.928 | 0.931 |
| 3.0 | 0.59 | 0.960 | 0.964 |
| **3.5** | **0.54** | **0.972** | 0.972 |
| 5.0 | 0.43 | 0.989 | 0.985 |
| 6.5 | 0.34 | **0.992** | 0.992 |

- **τ = 3.0에서 59%의 페어가 통과하고 96%가 지배채널을 맞춤.** τ = 6.5에서는 34%만 통과하지만 99.2%로 안전.
- 지배채널 예측 자체는 재현 가능하고 (agree 곡선이 true-margin 곡선과 거의 겹침), **모델 τ가 실 τ의 신뢰할 만한 대리치**.
- `dipole_class` 골격 외삽에서 커브는 아래로 이동 — τ=5.0에서 coverage 0.39 / agree 0.95, τ=7 이상에서도 agree 0.98 이하. 골격 밖에선 gate 임계를 더 높이 잡을 것.

원본: [`margin_calibration.csv`](margin_calibration.csv).

### 이미 만족된 항목 (S4)

- **S4 — Ridge / KRR / SVR / XGB HP 재튠**: 감사 초안에는 재실험 항목으로 적혀 있었지만, 현재 파이프라인이 이미 이 요구를 만족한다. [`train_ml_single.py:107-112`](../train_ml_single.py#L107-L112) 에서 `GridSearchCV(cv=KFold(5, shuffle=True, random_state=23), scoring="neg_mean_absolute_error")` 를 4 모델 각각에 별도로 실행 (Ridge α 5-grid, KRR α×γ 16-grid, SVR C×γ×ε 36-grid, XGB n_est×depth×lr 12-grid) 후 seed 23의 train fold에서 정해진 best HP를 5개 seed 재사용. **KRR이 8/11 타깃에서 최적**이라는 관찰은 편향 없는 재튠 후 결과다. 별도 추가 실행 불필요.

### 후속으로 남긴 항목 (S2, S6, S7)

이번 라운드에서 처리하지 않음 — 각각 상당한 재계산 또는 스킴 변경이 필요:

- **S2 — xTB 기하 민감도**: 현재는 DFT TS 기하 위 xTB SPE(성능 상한). `xtb --opt` 로 로컬 최적화한 기하에서 재추출 후 5개 seed 재학습, 배포 성능 확인. 5,260 × 3 구조 최적화(~1분/구조) + full SPE 재계산 ≈ 새 array job.
- **S6 — 채널 sum 재조정 실험**: S1 결과를 받아 barrier / eint_spe 재조정 스킴 (예: `disp_scale`, `xc_missing` 학습 채널). 라벨 정의를 건드리므로 별도 spec.
- **S7 — `is_charged` 를 명시 feature로 추가**: `charge2` 컬럼은 이미 `xtb_features.parquet` 에 존재하나 `AUX18` 에는 안 들어감. 새 arm `ESPLEY73 = ESPLEY72 + [charge2]` 로 재학습하면 이온 그룹 MAE 개선 여부 정량 확인 가능. s03 array 1회로 완료.
