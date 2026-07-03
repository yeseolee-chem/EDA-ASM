# V1/analysis — Claisen 15-substrate Hammett + EDA regression

Downstream analysis of `V1/outputs/v1_claisen_asr.parquet` (15 para-substituted
vinyl allyl ethers, wB97X-3c geometries + ZORA-BLYP-D3(BJ)/TZ2P NOCV-EDA).
Reads the 5-channel ASR/EDA vector for each substrate and tests how each
channel — and the total activation barrier — track Hammett σₚ (and the
Swain–Lupton (F, R) decomposition).

## Layout

```
V1/analysis/
├── hammett_plot.py           analysis + plotting script
├── submit_hammett.sh         SLURM submitter (cpu2, 10 min)
├── logs/                     SLURM stdout/stderr
├── figures/
│   ├── hammett_barrier.png             ΔE‡ vs σₚ (single-parameter)
│   ├── hammett_per_channel_grid.png    6-panel grid (barrier + 5 EDA channels)
│   ├── swain_lupton_barrier.png        ΔE‡ predicted vs. observed (F,R fit)
│   └── channel_correlation.png         pairwise Pearson |r| heatmap
└── results/
    ├── regression_summary.csv          slope ρ, R², r, p per channel
    ├── regression_summary.md           human-readable table
    └── correlation_matrix.csv          full Pearson matrix
```

## Reproduce

```bash
sbatch V1/analysis/submit_hammett.sh
```

Runs on cpu2 in <1 minute. No GPU required.

## Headline findings (from the frozen run)

| channel | ρ (kcal · mol⁻¹ · σ⁻¹) | R² | Pearson r |
|---|---|---|---|
| ΔE‡ (wB97X-3c) total | +1.85 ± 3.31 | 0.025 | +0.16 |
| ΔE_strain            | −2.59 ± 0.66 | **0.535** | −0.73 |
| ΔV_elst              | +10.9  ± 3.4 | **0.440** | +0.66 |
| ΔE_Pauli             | −22.5  ± 7.2 | **0.431** | −0.66 |
| ΔE_oi                | +7.4   ± 4.4 | 0.180 | +0.42 |
| ΔE_disp              | −0.17 ± 0.42 | 0.012 | −0.11 |

Swain–Lupton dual-parameter fit of ΔE‡:
`ΔE‡ ≈ −16.2·F + 9.1·R + 30.9  (R² = 0.481)`

### Interpretation

**The total activation barrier is almost independent of σₚ (R² = 0.03).**
That looks like the electronic effects don't matter — but the per-channel
decomposition tells a much richer story:

- **ΔE_Pauli** and **ΔV_elst** each correlate strongly with σₚ (R² ≈ 0.43)
  with *opposite signs* — donors (σₚ < 0) push both channels toward more
  Pauli repulsion and less-negative electrostatic stabilization. These
  two contributions **cancel** in the total barrier.
- **ΔE_strain** shows the tightest single-parameter fit (R² = 0.54, ρ < 0):
  stronger σ-donors give slightly *higher* strain at the TS.
- **ΔE_oi** shows a moderate σₚ dependence (R² = 0.18) but doesn't rescue
  the total.
- **ΔE_disp** is essentially flat — Claisen TS dispersion is not
  substituent-driven at this level.
- The Swain–Lupton fit (R² = 0.48) is much stronger than the single-σₚ
  fit (R² = 0.03), meaning the field and resonance contributions each
  matter separately — a well-known limitation of single-parameter
  Hammett for pericyclic transition states.

The take-home is that the **EDA decomposition** — not the total ΔE‡ —
is what carries the substituent electronic signal. This is the argument
for using EDA channels as features in the downstream m1/m2/m3
Δ-learners (rather than trying to predict the barrier directly from σₚ).
