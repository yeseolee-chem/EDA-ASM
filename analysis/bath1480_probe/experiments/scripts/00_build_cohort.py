#!/usr/bin/env python3
"""Phase 0: cleanup scratch + build cohort + assemble labels/features pkl.

Executes after strain 48h wall hit. Uses all strain.json currently on disk.
"""
import json
import re
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

STRAIN = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/strain_am1/work")
KISTI  = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/tt_eda_kisti_results/data")
MANUAL_TT_PKL = "/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/tt_eda_kisti_results/manual_tt_7ch_xtb.pkl"
MANUAL_SOLVENT_PKL = "/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/tt_repro/data/manual_tt_solvent.pkl"
OUT = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments/cohort_v1")

K = 627.5094740631
FSPE_RE = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)")
CLOSURE_THRESHOLD_KCAL = 1e-6

# ---- Step 1: scratch cleanup (existing strain.json-having rxn folders) ----
JUNK_PATTERNS = (
    "*.tmp", "*.tmpg", "*.hess", "*.densities", "*.densitiesinfo",
    "*.gbw", "*.cpcm", "*.cpcm_corr", "*.bibtex", "*.property.txt",
    "*.SHARKINP.tmp", "*.opt", "*.citations.tmp", "*.ginp.tmp", "*.int.tmp",
    "*.smd", "*.smd.grd", "*.engrad", "*.carthess", "*.hostnames",
    "*.0", "*_trj.xyz",
)
JUNK_PATTERNS_EXT_INDEX = ("bas0", "bas1", "bas2", "bas3", "bas4", "bas5",
                             "tmp0", "tmp1", "tmp2", "tmp3")


def cleanup_completed_rxns():
    """Delete scratch files from rxn folders that have strain.json (safe: task not running there)."""
    freed_bytes = 0
    n_files = 0
    for d in STRAIN.glob("rxn_*"):
        if not (d / "strain.json").exists():
            continue  # skip in-progress folders
        for pat in JUNK_PATTERNS:
            for f in d.glob(pat):
                freed_bytes += f.stat().st_size
                f.unlink()
                n_files += 1
        for ext in JUNK_PATTERNS_EXT_INDEX:
            for f in d.glob(f"*.{ext}"):
                freed_bytes += f.stat().st_size
                f.unlink()
                n_files += 1
    print(f"[cleanup] deleted {n_files} files, freed {freed_bytes/1024/1024/1024:.2f} GB")


# ---- Step 2: closure verification ----
def get_eab(out_path):
    if not out_path.exists():
        return None
    m = FSPE_RE.findall(out_path.read_text(errors="replace"))
    return float(m[-1]) if m else None


def verify_and_load_strain():
    rows = []
    passed = 0
    failed = 0
    fail_reasons = []
    for jf in sorted(STRAIN.glob("rxn_*/strain.json")):
        s = json.loads(jf.read_text())
        rxn = int(s["rxn_id"])
        req = ["e_frag1_rel_tzvp_eh", "e_frag2_rel_tzvp_eh",
               "e_frag1_dist_tzvp_eh", "e_frag2_dist_tzvp_eh",
               "distortion_energy_1_dft", "distortion_energy_2_dft"]
        if any(f not in s for f in req):
            failed += 1
            fail_reasons.append((rxn, "missing fields"))
            continue
        eda_out = KISTI / f"rxn_{rxn:04d}" / "eda.out"
        e_ab = get_eab(eda_out)
        if e_ab is None:
            failed += 1
            fail_reasons.append((rxn, "no E_AB"))
            continue
        # closure
        d1, d2 = s["distortion_energy_1_dft"], s["distortion_energy_2_dft"]
        e_f1_dist = s["e_frag1_dist_tzvp_eh"]
        e_f2_dist = s["e_frag2_dist_tzvp_eh"]
        e_f1_rel  = s["e_frag1_rel_tzvp_eh"]
        e_f2_rel  = s["e_frag2_rel_tzvp_eh"]
        interaction_own = (e_ab - e_f1_dist - e_f2_dist) * K
        barrier_recon   = d1 + d2 + interaction_own
        barrier_direct  = (e_ab - e_f1_rel - e_f2_rel) * K
        gap = abs(barrier_direct - barrier_recon)
        if gap > CLOSURE_THRESHOLD_KCAL:
            failed += 1
            fail_reasons.append((rxn, f"closure gap {gap:.3e}"))
            continue
        passed += 1
        rows.append({
            "reaction_number": rxn,
            "e_ab_eh": e_ab,
            "e_frag1_dist_tzvp_eh": e_f1_dist,
            "e_frag2_dist_tzvp_eh": e_f2_dist,
            "e_frag1_rel_tzvp_eh":  e_f1_rel,
            "e_frag2_rel_tzvp_eh":  e_f2_rel,
            "d1_own": d1,
            "d2_own": d2,
            "interaction_own": interaction_own,
            "barrier_own_reconstructed": barrier_recon,
            "closure_gap_kcal": gap,
        })
    print(f"[closure verify] pass={passed}  fail={failed}")
    if fail_reasons:
        for rxn, why in fail_reasons[:10]:
            print(f"  FAIL rxn_{rxn:04d}: {why}")
    return pd.DataFrame(rows)


# ---- Step 3: label + feature assembly ----
def build_labels_pkl(strain_df):
    # Espley original labels from manual_tt_solvent.pkl (or manual_tt_7ch_xtb.pkl — same 6-13 cols)
    espley = pd.read_pickle(MANUAL_TT_PKL)
    espley_cols = ["reaction_number",
                   "d1_dft" if "d1_dft" in espley.columns else "distortion_energy_1_dft",
                   "distortion_energy_2_dft",
                   "sum_distortion_energies_dft",
                   "interaction_energies_dft",
                   "e_barrier_dft",
                   "q_barrier_dft"]
    # rename to consistent
    rename_map = {"distortion_energy_1_dft": "d1_dft_espley",
                  "distortion_energy_2_dft": "d2_dft_espley",
                  "sum_distortion_energies_dft": "sum_dist_espley",
                  "interaction_energies_dft": "interaction_espley",
                  "e_barrier_dft": "barrier_e_espley",
                  "q_barrier_dft": "barrier_q_espley"}
    espley_kept = espley[["reaction_number", "distortion_energy_1_dft", "distortion_energy_2_dft",
                          "sum_distortion_energies_dft", "interaction_energies_dft",
                          "e_barrier_dft", "q_barrier_dft"]].rename(columns=rename_map)

    # ORCA EDA channels (already computed, in manual_tt_7ch_xtb.pkl)
    channels = espley[["reaction_number", "elst_dft", "pauli_dft", "oi_dft",
                       "disp_dft", "cpcm_dft", "cds_dft", "eint_dft"]]

    # merge everything
    m = strain_df.merge(espley_kept, on="reaction_number", how="inner")
    m = m.merge(channels, on="reaction_number", how="inner")
    return m


def build_features_pkls(cohort_ids):
    """Espley 46 features and Arm B 54 features. Filter to cohort."""
    df_solvent = pd.read_pickle(MANUAL_SOLVENT_PKL)
    df_full = pd.read_pickle(MANUAL_TT_PKL)

    # Identify feature columns (exclude labels)
    label_cols = {"d1_dft", "d2_dft", "distortion_energy_1_dft", "distortion_energy_2_dft",
                  "sum_distortion_energies_dft", "interaction_energies_dft",
                  "e_barrier_dft", "q_barrier_dft",
                  "elst_dft", "pauli_dft", "oi_dft", "disp_dft", "cpcm_dft", "cds_dft", "eint_dft",
                  "distortion_energy_1_am1", "distortion_energy_2_am1",
                  "sum_distortion_energies_am1", "interaction_energies_am1",
                  "e_barrier_am1", "q_barrier_am1"}
    # Espley features: from solvent pkl (excludes xTB)
    espley_feats = [c for c in df_solvent.columns
                    if c != "reaction_number" and c not in label_cols
                    and "xtb" not in c.lower()]
    features_espley = df_solvent[["reaction_number"] + espley_feats]
    features_espley = features_espley[features_espley["reaction_number"].isin(cohort_ids)].reset_index(drop=True)

    # Arm B: espley + xTB (from full pkl)
    all_feats = [c for c in df_full.columns
                 if c != "reaction_number" and c not in label_cols]
    features_armB = df_full[["reaction_number"] + all_feats]
    features_armB = features_armB[features_armB["reaction_number"].isin(cohort_ids)].reset_index(drop=True)

    print(f"[features] Espley: {len(espley_feats)} features / Arm B: {len(all_feats)} features")
    return features_espley, features_armB


def assign_folds(df, seed=42, n_folds=5):
    """5-fold stratified on n_atoms quantile. Adds fold_id column."""
    # Get n_atoms from manifest
    m = pd.read_parquet("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/tt_eda_kisti_results/data/manifest.parquet")
    n_atoms_map = dict(zip(m["rxn_id"], m["n_atoms"]))
    df["n_atoms"] = df["reaction_number"].map(n_atoms_map)
    # Quantile bin then round-robin within each bin
    df["_size_bin"] = pd.qcut(df["n_atoms"].rank(method="first"), q=n_folds, labels=False)
    rng = np.random.default_rng(seed)
    df["fold_id"] = -1
    for b in range(n_folds):
        idx = df[df["_size_bin"] == b].index.tolist()
        rng.shuffle(idx)
        for i, ix in enumerate(idx):
            df.loc[ix, "fold_id"] = i % n_folds
    df["fold_id"] = df["fold_id"].astype(int)
    df = df.drop(columns=["_size_bin"])
    return df


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PHASE 0: scratch cleanup + cohort build")
    print("=" * 70)

    # Step 1
    print("\n[1/4] Cleaning scratch files from completed rxn folders...")
    cleanup_completed_rxns()

    # Step 2
    print("\n[2/4] Verifying closure across completed strain rxns...")
    strain_df = verify_and_load_strain()
    print(f"  {len(strain_df)} rxns passed closure check")

    # Step 3
    print("\n[3/4] Assembling labels + features + fold split...")
    labels = build_labels_pkl(strain_df)
    labels = assign_folds(labels)

    features_espley, features_armB = build_features_pkls(labels["reaction_number"].tolist())

    # Cohort subset
    cohort = labels[["reaction_number", "n_atoms", "fold_id"]].copy()
    cohort["strain_complete"] = True

    # Save
    labels.to_pickle(OUT / "labels_v1.pkl")
    features_espley.to_pickle(OUT / "features_espley_v1.pkl")
    features_armB.to_pickle(OUT / "features_armB_v1.pkl")
    cohort.to_parquet(OUT / "cohort_subset.parquet", index=False)

    print(f"\n[4/4] Saved to {OUT}:")
    print(f"  cohort_subset.parquet: {len(cohort)} rxns")
    print(f"  labels_v1.pkl:         {labels.shape}")
    print(f"  features_espley_v1.pkl: {features_espley.shape}")
    print(f"  features_armB_v1.pkl:   {features_armB.shape}")
    print()
    print("Fold size distribution:")
    print(cohort.groupby("fold_id").size().to_string())
    print()
    print("n_atoms distribution:")
    print(cohort["n_atoms"].describe().to_string())


if __name__ == "__main__":
    main()
