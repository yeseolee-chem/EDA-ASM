# 4-Phase Experiment Runbook

Order of execution (after strain 48h wall hit):

## 1. Phase 0 — Setup (cleanup + cohort + labels/features)
```bash
sbatch analysis/bath1480_probe/experiments/scripts/sbatch_phase0.sh
```
Wait until complete (~10-30 min).

## 2. Phase 1+2 — Sklearn training (Round A, 10 tasks parallel)
```bash
# Phase 1: Espley baseline (τ=0.05, 46 features, 6 targets)
sbatch --export=ALL,\
PHASE_NAME=phase1_espley,\
LABELS_PKL=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments/cohort_v1/labels_v1.pkl,\
FEATURES_PKL=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments/cohort_v1/features_espley_v1.pkl,\
"TARGET_COLS=distortion_energy_1_dft distortion_energy_2_dft sum_distortion_energies_dft interaction_energies_dft e_barrier_dft q_barrier_dft",\
VAR_TAU=0.05 \
  analysis/bath1480_probe/experiments/scripts/sbatch_phase123.sh

# Phase 2: Arm B, τ=1e-10, 54 features, 8 targets
sbatch --export=ALL,\
PHASE_NAME=phase2_armB_tau1e-10,\
LABELS_PKL=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments/cohort_v1/labels_v1.pkl,\
FEATURES_PKL=/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments/cohort_v1/features_armB_v1.pkl,\
"TARGET_COLS=d1_own d2_own elst_dft pauli_dft oi_dft disp_dft cpcm_dft cds_dft",\
VAR_TAU=1e-10 \
  analysis/bath1480_probe/experiments/scripts/sbatch_phase123.sh
```

Wait for both to complete (~4-6h).

## 3. Phase 3 + Phase 4 — Round B
```bash
# Phase 3: Arm B, τ=0.05 (5 CPU tasks)
sbatch --export=ALL,\
PHASE_NAME=phase3_armB_tau0.05,\
LABELS_PKL=...,FEATURES_PKL=...features_armB_v1.pkl,\
"TARGET_COLS=d1_own d2_own elst_dft pauli_dft oi_dft disp_dft cpcm_dft cds_dft",\
VAR_TAU=0.05 \
  analysis/bath1480_probe/experiments/scripts/sbatch_phase123.sh

# Phase 4a: MACE TS feature extraction (3 GPU tasks)
sbatch analysis/bath1480_probe/experiments/scripts/sbatch_phase4_mace.sh

# Phase 4b (after 4a): physics d1..d24 (small, can run on login-node or short sbatch)
# python3 analysis/bath1480_probe/experiments/scripts/04_compute_physics_24.py

# Phase 4c: XGBoost + delta training (25 cells, 3-way GPU parallel)
sbatch --dependency=afterany:<JID_4a> \
  analysis/bath1480_probe/experiments/scripts/sbatch_phase4_train.sh
```

Wait ~4-6h.

## 4. Phase 5 — Aggregate + figures
```bash
sbatch analysis/bath1480_probe/experiments/scripts/sbatch_phase5.sh
```

## 5. Strain resume (Cycle 2)
```bash
sbatch --partition=cpu1 --export=ALL,STRAIN_BUCKET_OFFSET=0 \
    analysis/bath1480_probe/strain_am1/scripts/sbatch_strain_array.sh
sbatch --partition=cpu1 --export=ALL,STRAIN_BUCKET_OFFSET=5 \
    analysis/bath1480_probe/strain_am1/scripts/sbatch_strain_array.sh
```

## Slot budget

| Phase | Tasks | Type |
|---|---|---|
| Phase 0 | 1 | CPU |
| Phase 1 | 5 | CPU |
| Phase 2 | 5 | CPU |
| Phase 3 | 5 | CPU |
| Phase 4a | 3 | GPU |
| Phase 4c | up to 3 concurrent (via %3) | GPU |
| Phase 5 | 1 | CPU |

Round A (Phase 1+2): 10 CPU tasks, exactly at MaxJobs=10 cap.  
Round B (Phase 3 + Phase 4 chain): 5 CPU + up to 3 GPU = 8 tasks max.  
Round C (Phase 4c training): 3 GPU alone.
