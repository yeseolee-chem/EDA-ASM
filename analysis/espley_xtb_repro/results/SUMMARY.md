# espley_xtb_repro — final results (2026-09-14)

Espley 2024 protocol on Coley 5,269 with **GFN2-xTB (tblite) features instead of AM1**.
Feature set: **ESPLEY54** (54 features).

## Pipeline

1. **원자좌표 → tblite SPE**: 5 species per rxn (gs1, gs2, dist1, dist2, TS) — sbatch array 18 slices × %10 concurrent
2. **54-feature build** = 41 d^struct (11 dist + 15 Mulliken + 15 APT_surrogate) + 5 xTB energies + 8 GFN2-xTB channels
3. **ML training** = 11-target array × Ridge/KRR/SVR/XGB (Protocol A: 80/10/10 × 5 seeds + GridSearchCV) + 5 Protocol B OOF models
4. **Plots** = per-model scatter (8 channels each) + combined MAE bar

Total wall: ~50 min end-to-end.

## 결과 — Protocol A · ESPLEY54, 5-seed test MAE (kcal/mol)

| Target | Ridge | KRR_rbf | SVR_rbf | XGB | Best | r² (best) |
|---|---:|---:|---:|---:|---|---:|
| **ΔE‡ barrier** | 2.86 | **2.03** | 2.06 | 2.16 | KRR_rbf | 0.912 |
| **d1** (dipole strain) | 1.83 | 1.30 | **1.29** | 1.40 | SVR_rbf | 0.940 |
| **d2** (dipolarophile strain) | 1.33 | 1.10 | 1.11 | **1.02** | XGB | 0.936 |
| **ΔE_int (SPE)** | 2.04 | **1.44** | 1.45 | 1.51 | KRR_rbf | 0.877 |
| **Bond E (EDA)** | 2.33 | **1.82** | 1.84 | 1.85 | KRR_rbf | 0.883 |
| **elst** | 4.28 | 2.98 | **2.89** | 3.20 | SVR_rbf | 0.947 |
| **Pauli** | 5.59 | 3.22 | **3.15** | 3.85 | SVR_rbf | 0.972 |
| **OI** | 3.24 | **1.78** | 1.87 | 2.22 | KRR_rbf | **0.984** |
| **disp** | 2.26 | **1.38** | 1.39 | 1.42 | KRR_rbf | 0.840 |
| **CPCM** | 3.77 | 2.98 | 3.00 | **2.95** | XGB | 0.520 |
| **CDS** | 0.49 | 0.33 | 0.33 | **0.31** | XGB | 0.793 |

**하이라이트**:
- **6 EDA-NOCV 채널 전부 예측 가능** — xTB에서 직접 얻지 못하는 물리량이지만 54-feature ML로 재구성 성공 (r² 0.52–0.98)
- **OI r² 0.984** — 최고 (0.8% of range)
- **Pauli r² 0.972 · elst r² 0.947** — range의 1–2% 오차
- **barrier MAE 2.03** — Espley 원논문 AM1 SVR test MAE 2.55 대비 20% 개선 (Coley 5,260 rxn = 다른 dataset이지만 참고)
- **XGB**: sparse channels (CDS, CPCM), d2 에서 최고
- **KRR / SVR**: 물리 규모 큰 채널 (barrier, OI, Pauli, elst)에서 최고

## xTB 매핑 (Espley AM1 → tblite)

- **6 AM1 에너지 → 5개 (q_barrier 제외)** — tblite에 Hessian 없어 quasi-harmonic G(T) 불가
- **15 Mulliken** ✓ 직접 (`tblite.result.get("charges")`)
- **15 APT → APT_surrogate** (Wiberg BO row sum, `bond-orders` 매트릭스 행합)
- **8 채널**: 실 값 5 (strain 2 + elst/Pauli/oi 대체값) + 마커 3 (disp/cpcm/cds = 0.0, tblite 불가)
- 용매: gas phase (tblite ALPB 미지원)
- 지오메트리: Coley DFT-이완 그대로 (xTB 재이완 안 함)

## 파일 (`results/`)

- `SUMMARY.md` — 이 문서
- `ml_report.json` — 11 target × (Protocol A 4 + Protocol B 5) 전 metric
- `ml_table_espley.csv` — 99행 (11 target × 9 model)
- `predictions.parquet` — 5-seed 폴드 예측 (rxn_id, seed, model, target, y, yhat)
- `xtb_features.parquet` — 5,260 rxn × 54 feature + 11 target + 메타
- `ml_targets/` — target당 JSON (병렬 array 원본)
- `figures/`:
  - `scatter_Ridge_ESPLEY54.png` · 8-패널 (8 채널)
  - `scatter_KRR_rbf_ESPLEY54.png` · 8-패널
  - `scatter_SVR_rbf_ESPLEY54.png` · 8-패널
  - `scatter_XGB_ESPLEY54.png` · 8-패널
  - `mae_bar_espley54.png` · 8 채널 × 4 모델 grouped bar

## 재현

```bash
sbatch analysis/espley_xtb_repro/s01_xtb_array.sh                                    # xTB features
sbatch --dependency=afterok:<JID> analysis/espley_xtb_repro/s02_aggregate.sh
sbatch analysis/espley_xtb_repro/s03_ml_array.sh                                     # 11-target ML array
sbatch --dependency=afterok:<JID> analysis/espley_xtb_repro/s04_aggregate_ml.sh
sbatch --dependency=afterok:<JID> analysis/espley_xtb_repro/s05_plot.sh              # 5 figures
```
