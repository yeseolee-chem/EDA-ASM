#!/usr/bin/env python3
"""Build v2 datasets — clean of oracle leakage + strain frag swap fix.

Fixes applied:
  1. Remove xTB dispersion oracle features (disp_xtb, dispd4_xtb) AND
     eint_total_xtb (arithmetic backdoor: eint_total − Σ(other channels) ≈ dispersion).
     Policy (a): disp channel declared analytical/oracle; excluded from best-model
     claims but reported for reference.
  2. Rename strain_1_xtb ↔ strain_2_xtb (frag index audit: r(strain_1_xtb, d2_own)
     = 0.916 vs r(strain_1_xtb, d1_own) = 0.009; xTB and DFT frag numbering mismatched).
  3. Variance-threshold filters applied AFTER the above.

Outputs:
  cohort_v1/phase2_dataset_v2.pkl  (τ=1e-10)
  cohort_v1/phase3_dataset_v2.pkl  (τ=0.05)
  cohort_v1/phase5_dataset_v2.pkl  (no filter)
  cohort_v1/hps_phase23_v2.pkl     (same aliased HPs)
"""
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

EXP = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments")
BASE = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/tt_eda_kisti_results")

ORACLE_FEATURES = ["disp_xtb", "dispd4_xtb", "eint_total_xtb"]
SWAP_PAIRS = [("strain_1_xtb", "strain_2_xtb")]


def main():
    manual_xtb = pd.read_pickle(BASE / "manual_tt_7ch_xtb.pkl")
    labels_v1 = pd.read_pickle(EXP / "cohort_v1/labels_v1.pkl")
    print(f"manual_xtb: {manual_xtb.shape}, labels_v1: {labels_v1.shape}")

    # Add own labels (rename to _dft for ml_analysis target detection)
    own = labels_v1[["reaction_number", "d1_own", "d2_own",
                       "interaction_own", "barrier_own_reconstructed"]].copy()
    own.columns = ["reaction_number", "d1_own_dft", "d2_own_dft",
                     "interaction_own_dft", "barrier_own_dft"]
    m = manual_xtb.merge(own, on="reaction_number", how="inner")
    print(f"merged: {m.shape}")

    # (1) Remove oracle features
    for c in ORACLE_FEATURES:
        if c in m.columns:
            m = m.drop(columns=[c])
            print(f"  removed oracle: {c}")

    # (2) Rename swapped per-fragment features
    rename_map = {}
    for a, b in SWAP_PAIRS:
        if a in m.columns and b in m.columns:
            rename_map[a] = f"__TMP__{a}"
    m = m.rename(columns=rename_map)
    rename_map2 = {}
    for a, b in SWAP_PAIRS:
        rename_map2[f"__TMP__{a}"] = b
        rename_map2[b] = a
    m = m.rename(columns=rename_map2)
    for a, b in SWAP_PAIRS:
        print(f"  swapped: {a} ↔ {b}")

    print(f"clean base: {m.shape}")
    m.to_pickle(EXP / "cohort_v1/phase5_dataset_v2.pkl")
    print(f"saved phase5_dataset_v2.pkl (no filter)")

    # Feature / label split for τ filters
    label_cols = [c for c in m.columns if "_dft" in c or c == "reaction_number"]
    feature_cols = [c for c in m.columns if c not in label_cols]
    numeric = m[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    # (3) τ filters
    def build_filtered(tau, tag):
        vt = VarianceThreshold(threshold=tau)
        vt.fit(m[numeric])
        kept = np.array(numeric)[vt.get_support()].tolist()
        dropped = np.array(numeric)[~vt.get_support()].tolist()
        cols = ["reaction_number"] + kept + [c for c in label_cols if c != "reaction_number"]
        out = m[cols]
        out.to_pickle(EXP / f"cohort_v1/{tag}_dataset_v2.pkl")
        print(f"saved {tag}_dataset_v2.pkl (τ={tau}): {out.shape}  kept {len(kept)} features")
        if dropped: print(f"  dropped: {dropped}")

    build_filtered(1e-10, "phase2")
    build_filtered(0.05, "phase3")

    # (4) HPs (alias for our own labels)
    hps_arm_a = pickle.load(open(EXP / "results/phase1_espley_paper_reproduction/hps_arm_a.pkl", "rb"))
    hps = dict(hps_arm_a)
    alias = {
        "d1_own_dft": "distortion_energy_1_dft",
        "d2_own_dft": "distortion_energy_2_dft",
        "interaction_own_dft": "interaction_energies_dft",
        "barrier_own_dft": "e_barrier_dft",
    }
    for new_t, ref_t in alias.items():
        for mdl in ["ridge","krr","svr","rf","2_st_nn","4_st_nn"]:
            nk, rk = f"{mdl}_{new_t}", f"{mdl}_{ref_t}"
            if rk in hps: hps[nk] = hps[rk]
    pickle.dump(hps, open(EXP / "cohort_v1/hps_phase23_v2.pkl", "wb"))
    print(f"saved hps_phase23_v2.pkl: {len(hps)} entries")


if __name__ == "__main__":
    main()
