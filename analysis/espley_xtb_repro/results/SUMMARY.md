# espley_xtb_repro — final results (2026-09-14)

Espley 2024 protocol on Coley 5269 with **GFN2-xTB (tblite) features instead of AM1**.

## Pipeline

1. **원자좌표 → tblite SPE**: 5 species per rxn (gs1, gs2, dist1, dist2, TS) — sbatch array 18 slices × %10 concurrent
2. **54-feature build** = 41 d^struct (11 dist + 15 Mulliken + 15 APT_surrogate) + 5 xTB energies + 8 GFN2-xTB channels
3. **ML training** = 11-target array × Ridge/KRR/SVR/XGB (Protocol A: 80/10/10 × 5 seeds + GridSearchCV) + 5 Protocol B OOF models
4. **Plots** = per-channel MAE bars + scatter + heatmap

Total wall: ~50 min end-to-end (5260 rxns → 5 SPE each → 4 model × 2 fset × 11 target).

## Best model per target (Protocol A · ESPLEY54)

| Target | Best model | pre-ML MAE | test MAE ± sd | r² | MAE %range | 개선 배수 |
|---|---|---:|---:|---:|---:|---:|
| **ΔE‡ barrier** | KRR_rbf | 8.73 | **2.03 ± 0.08** | 0.912 | 3.0% | **×4.3** |
| **d1** (dipole strain) | SVR_rbf | 5.32 | **1.29 ± 0.03** | 0.940 | 2.4% | **×4.1** |
| **d2** (dipolarophile strain) | XGB | 1.61 | **1.02 ± 0.07** | 0.936 | 1.3% | ×1.6 |
| **ΔE_int(SPE)** | KRR_rbf | 5.06 | **1.44 ± 0.07** | 0.877 | 2.0% | ×3.5 |
| **Bond E (EDA)** | KRR_rbf | 7.35 | **1.83 ± 0.10** | 0.883 | 2.2% | ×4.0 |
| **elst** | SVR_rbf | — | **2.89 ± 0.17** | 0.947 | 1.7% | — |
| **Pauli** | SVR_rbf | — | **3.15 ± 0.21** | 0.972 | 1.1% | — |
| **OI** | KRR_rbf | — | **1.78 ± 0.08** | **0.984** | 0.8% | — |
| **disp** | KRR_rbf | — | **1.38 ± 0.05** | 0.840 | 4.8% | — |
| **CPCM** | XGB | — | **2.95 ± 0.16** | 0.520 | 5.2% | — |
| **CDS** | XGB | — | **0.31 ± 0.00** | 0.793 | 4.9% | — |

**Highlights**:
- **6 EDA-NOCV 채널 전부 예측 가능** — 6채널은 xTB에서 직접 얻지 못하는 물리량인데 54-feature ML로 재구성 성공 (r² 0.52–0.98)
- **OI (Orbital Interaction) r² 0.984** — 채널 중 최고 (0.8% of range)
- **Pauli r² 0.972 · elst r² 0.947** — 모두 range의 1–2% 오차
- **barrier MAE 2.03** — Espley 원논문 AM1 SVR test MAE 2.55 대비 20% 개선 (다른 dataset이지만 참고)
- **XGB가 sparse channels (CDS, CPCM) 에서 최고**, KRR/SVR이 물리 규모 큰 채널에서 최고

## Feature Set 대비 (Protocol A test MAE, kcal/mol)

| target | E5 best | ESPLEY54 best | 54가 얼마나 유리? |
|---|---:|---:|---:|
| barrier | 3.84 | **2.03** | ×1.9 |
| d1 | 2.75 | **1.29** | ×2.1 |
| d2 | 1.51 | **1.02** | ×1.5 |
| eint_spe | 2.91 | **1.44** | ×2.0 |
| e_bond | 3.18 | **1.83** | ×1.7 |
| elst | 7.66 | **2.89** | ×2.7 |
| Pauli | 13.22 | **3.15** | ×4.2 |
| OI | 7.70 | **1.78** | ×4.3 |
| disp | 2.87 | **1.38** | ×2.1 |
| CPCM | 3.99 | **2.95** | ×1.4 |
| CDS | 0.63 | **0.31** | ×2.0 |

54-feature ESPLEY set이 모든 채널에서 E5(5-feature) 대비 우세. Pauli/OI 등 EDA-NOCV 채널은 4배 이상 개선 — Mulliken + WBO + geometric feature가 물리 분해에 결정적 정보를 담음.

## 결정한 xTB 매핑 (Espley AM1 → tblite)

- **6 AM1 에너지 → 5개 (q_barrier 제외)** — tblite에 Hessian 없어 quasi-harmonic G(T) 불가
- **15 Mulliken** ✓ 직접 (`tblite.result.get("charges")`)
- **15 APT → APT_surrogate** (Wiberg BO row sum, `bond-orders` 매트릭스 행합)
- **8 채널**: 실 값 5 (strain 2 + elst/Pauli/oi 대체값) + 마커 3 (disp/cpcm/cds = 0.0, tblite 불가)
- 용매: gas phase (tblite ALPB 미지원)
- 지오메트리: Coley DFT-이완 그대로 (xTB 재이완 안 함)

## 파일

- `ml_report.json` — 78 KB, 11 target × 2 fset × (Protocol A 4 + Protocol B 5) 전 metric
- `ml_table_espley.csv` — 23 KB, 198행 (11 target × 9 model × 2 fset)
- `predictions.parquet` — 2.8 MB, Protocol A 5-seed 폴드 예측 (rxn_id, seed, model, target, y, yhat)
- `xtb_features.parquet` — 3.4 MB, 5260 rxn × 54 feature + 11 target + 메타
- `ml_targets/` — target당 JSON (병렬 array 원본, aggregate 전 상태)
- `figures/` — 15 PNG
  - `mae_per_target.png` — 채널별 4 model × 2 fset 그룹 막대
  - `mae_pct_range.png` — MAE/range % heatmap
  - `pre_vs_post_mae.png` — 개선 배수
  - `scatter_grid_espley54.png` — 11×4 grid
  - `scatter_dft_*.png` × 11 — 채널별 8-패널 상세

## 코드 (git-tracked, `analysis/espley_xtb_repro/`)

- `xtb_slice.py`, `aggregate.py`, `train_ml_single.py`, `aggregate_ml.py`, `plot_results.py`
- `s01_xtb_array.sh`, `s02_aggregate.sh`, `s03_ml_array.sh`, `s04_aggregate_ml.sh`, `s05_plot.sh`
- `fragmenter.py` (그래프 규칙 로컬 사본)

## 재현

```bash
# 1. xTB feature computation (약 40분 wall, 18 slice × 10 concurrent)
sbatch analysis/espley_xtb_repro/s01_xtb_array.sh
sbatch --dependency=afterok:<JID> analysis/espley_xtb_repro/s02_aggregate.sh

# 2. ML (11-target array, 약 10분 wall)
sbatch analysis/espley_xtb_repro/s03_ml_array.sh
sbatch --dependency=afterok:<JID> analysis/espley_xtb_repro/s04_aggregate_ml.sh

# 3. Plots
sbatch --dependency=afterok:<JID> analysis/espley_xtb_repro/s05_plot.sh
```
