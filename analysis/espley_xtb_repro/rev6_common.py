#!/usr/bin/env python3
"""rev6_common.py — shared conventions of rev 6 (docs/specs/REV6_FAIR_ESPLEY_COMPARISON.md).

Single source of truth for the rev 6 paths, seeds, arms, models, targets, row rules and test settings. Everything
already fixed in rev 4/5 is imported from rev5_common (paths of the rev 4/5 inputs, feature blocks, nmae,
nadeau_bengio, write_atomic, sha256, seeds); nothing is copied. Import it; do not copy its constants.
Light on imports on purpose: it is also imported inside the Espley-pinned env (env_espley, python 3.11).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

import rev5_common as R5

HERE = R5.HERE
REPO = R5.REPO

# ---------------------------------------------------------------- paths (rev 6 writes only here + new files)
SCRATCH = Path(os.environ.get("R6_SCRATCH", "/gpfs/tmp_cpu2/yeseo1ee/espley_rev6"))
RES6 = HERE / "results_rev6"
R4_SCRATCH = Path(os.environ.get("R4_SCRATCH", "/gpfs/tmp_cpu2/yeseo1ee/espley_rev4"))
R4_PREP = R4_SCRATCH / "compare" / "espley" / "prep.pkl"          # rev 4 swap vector (P-3)
HYP_SRC = R4_SCRATCH / "compare" / "espley" / "hyp_tuning.py"      # their hyp_tuning.py, fetched in rev 4
ESP_ML_SRC = R5.ESP_DATA / "machine_learning" / "ml_analysis.py"   # their ml_analysis.py
ESP_DS = R5.ESP_DATA / "feature_selection" / "_f_selection" / "tt" / "manual_tt_solvent.pkl"
ESP_RES = R5.ESP_DATA / "machine_learning" / "tt" / "solvent" / "ml_results.pkl"
ESP_HPS = R5.ESP_DATA / "hyperparameter_tuning" / "tt" / "solvent" / "hps.pkl"
# Phase R: recovered geometries / features are NEW files; the rev 4/5 ones (R5.G1_ROOT, R5.FEAT_G1, R5.FEAT_EXT)
# are read-only inputs
G1_ROOT_R6 = Path(os.environ.get("G1_ROOT_R6", "/gpfs/tmp_cpu2/yeseo1ee/espley_xtb_g1_rev6"))
XTB_R6 = SCRATCH / "xtb_r6"
FEAT_G1_R6 = XTB_R6 / "xtb_features_g1.parquet"                  # rev 4 G1 features of the recovered rows
FEAT_EXT_R6 = XTB_R6 / "xtb_features_ext_g1.parquet"             # rev 5 blocks B1..B6 of the recovered rows
FEAT_EXT_4994 = XTB_R6 / "xtb_features_ext_4994.parquet"         # R-4: rxn 4994 (rev 4 geometry, B1 tol 1e-5)
ROWS_R = RES6 / "R_rows_final.csv"                               # R-5 row table (all 5,260 labels + Espley flags)
X182_ALL = XTB_R6 / "X182_all.parquet"                           # rxn_id + the 182 features of every R_feat row
# Phase S inputs of every training (sha256 in prereg_rev6.json)
S_DIR = SCRATCH / "S"
S_TARGETS = S_DIR / "S_targets.parquet"                          # rxn_id + target columns (labels_all.json)
S_X_ESP = S_DIR / "S_X_esp.parquet"                              # rxn_id + Espley 47 AM1 columns, role order (P-3)
S_SPLITS_A = RES6 / "S_splits_appendixA.csv"                     # Phase A appendix (Protocol A, all labels)
R_PLAN = RES6 / "R_plan.md"
PREREG6_MD = RES6 / "PREREG_REV6.md"
PREREG6_JSON = RES6 / "prereg_rev6.json"
S_ROWS = RES6 / "S_rows.csv"
S_SPLITS = RES6 / "S_splits.csv"
# Espley-pinned environment (their requirements.txt; r6_espley_env.sh) for running THEIR code unchanged
ESP_ENV = SCRATCH / "env_espley"
ESP_PY = ESP_ENV / "bin" / "python"
# their requirements.txt pins keras_tuner 1.4.5, which cannot save Hyperband checkpoints under keras 3.3.3
# ("The filename must end in .weights.h5", P-1 2026-10-01); user decision 2026-10-01: keras_tuner 1.4.7, all else as pinned
ESP_PINS = {"tensorflow": "2.16.1", "keras": "3.3.3", "keras_tuner": "1.4.7", "scikit-learn": "1.3.1",
            "numpy": "1.26.0", "pandas": "2.1.1", "matplotlib": "3.8.4"}

# ---------------------------------------------------------------- seeds and split (§2-3)
SEEDS = list(R5.PROTO_A_SEEDS)        # 22, 23, 14, 1, 2 = compare_espley.SEEDS = Espley ml_analysis.py
TUNE_SEED = 23                        # Espley hyp_tuning.py tunes once, on the seed-23 train split
N_ESP = 3510                          # |R_esp|: Espley ds3 ML set, ds3 row order


def espley_split3(n, seed):
    """Espley _perform_train_test_split on positions 0..n-1 (ds3 order): (train, test, val).
    test = FIRST half of the 20 % hold-out, val = second half (their NN validation_data / Hyperband objective).
    Same calls as compare_espley.espley_split, which returns (train, test) only."""
    from sklearn.model_selection import train_test_split
    idx = np.arange(n)
    tr, rest = train_test_split(idx, test_size=0.2, random_state=seed)
    te, va = train_test_split(rest, test_size=0.5, random_state=seed)
    return tr, te, va


# ---------------------------------------------------------------- rows (§2-1)
ROW_SETS = ("R_esp", "R_lab", "R_feat", "R_hyg", "C", "C_all")
MAIN_ROWS, SENS_ROWS = "C", "C_all"   # C = R_lab ∩ R_feat ∩ R_hyg; C_all = R_lab ∩ R_feat
HYG_D2_MAX = 50.0                     # hygiene (rev 4 rule, train_ml_single.hygiene): d1 >= 0, d2 >= 0, d2 <= 50


def hygiene_mask(d1, d2):
    d1, d2 = np.asarray(d1, float), np.asarray(d2, float)
    return (d1 >= 0) & (d2 >= 0) & (d2 <= HYG_D2_MAX)


RECOVERED_BY = ("g1_rev4",            # original rev 4 G1 success (not recovered)
                "label_aidx",         # R-1: partition from input_meta A_idx, all other gates unchanged
                "optts_restart", "exact_hess", "displace_mode", "constrained_pre",   # R-2 ladders (R_plan.md)
                "alt_start", "constrained_ref",                                      # R-2 reference-graph ladder
                "gate_tol_user_decision")                                            # R-4 rxn 4994
SENS_DROP_RECOVERED = ("constrained_ref",)   # sensitivity: rows recovered this way removed
R4_RXN = 4994
R4_TOL_EH = 1e-5                      # R-4 user decision for rxn 4994 only (rev 5 B1 engine gate = R5.GATE_E_EH)

# ---------------------------------------------------------------- arms and models (§3)
FEAT_ESP = "ESP46"                    # Espley AM1 46 (role re-ordered, P-3)
FEAT_OUR = "EXT_ALL"                  # ours: ESPLEY73 + B1..B6 = 182 (R5.arm_columns("EXT_ALL", ...))
N_FEAT = {FEAT_ESP: 46, FEAT_OUR: 182}
PROTO_ESP, PROTO_OUR = "espley", "ours"
ARMS = {"ESP-C": (FEAT_ESP, PROTO_ESP),    # primary, Espley side: their code as-is (+ XGB)
        "OUR-O": (FEAT_OUR, PROTO_OUR),    # primary, our side
        "ESP-O": (FEAT_ESP, PROTO_OUR),    # tuning-matched (our pipeline)
        "OUR-C": (FEAT_OUR, PROTO_ESP)}    # tuning-matched (Espley protocol, espley_protocol())
PRIMARY = ("ESP-C", "OUR-O")
MODELS = ["Ridge", "KRR", "SVR", "XGB", "NN2", "NN4"]
NN_SIZE = {"NN2": 2, "NN4": 4}
# their hps / ml_results key prefixes (hyp_tuning.py :303-308, ml_analysis.py :355-360, :506)
ESP_KEY = {"Ridge": "ridge", "KRR": "krr", "SVR": "svr", "NN2": "2_st_nn", "NN4": "4_st_nn"}

# ---------------------------------------------------------------- targets (§1)
TARGETS_B = {"d1": "dft_d1_kcal", "d2": "dft_d2_kcal", "Int": "dft_eint_spe_kcal", "dEa": "dft_barrier_kcal"}
CHANNELS8 = {"d1": "dft_d1_kcal", "d2": "dft_d2_kcal", "elst": "dft_elst_dft", "pauli": "dft_pauli_dft",
             "oi": "dft_oi_dft", "disp": "dft_disp_dft", "cpcm": "dft_cpcm_dft", "cds": "dft_cds_dft"}
C_GHOST = "dft_c_ghost_kcal"          # eint_spe = e_bond + c_ghost (bsse_shift_kcal)
E_BOND = "dft_e_bond_kcal"            # = sum of the 6 EDA channels
TARGETS_A = {**CHANNELS8, "c_ghost": C_GHOST}
SIX = ("elst", "pauli", "oi", "disp", "cpcm", "cds")
# Espley ds3 target column that carries each comparison target in THEIR code (P-4: only the values are ours;
# d1/d2 = role order dipole / dipolarophile, Int = our sign eint_spe)
ESP_TCOL = {"d1": "distortion_energy_1_dft", "d2": "distortion_energy_2_dft",
            "Int": "interaction_energies_dft", "dEa": "e_barrier_dft"}

# ---------------------------------------------------------------- tests and reporting (§7)
N_BOOT = R5.N_BOOT                    # bias check bootstrap (rng BOOT_SEED)
BOOT_SEED = R5.SEED
ALPHA = 0.05
HOLM_FAMILY = list(TARGETS_B)         # primary: 4 targets, Holm
DECIMALS = {"mae": 2, "mae_fine": 3, "nmae": 3, "r2": 3}
FINE_MAE = ("disp", "cds")            # MAE to 3 decimals
P_SIG = 2                             # p to 2 significant digits
WITHIN_KCAL = 1.0                     # share of |error| <= 1 kcal/mol
N_REP_CV = 5                          # extra 5-fold CV for models without a comparable tuning score (§3-3)

# ---------------------------------------------------------------- figures (§10)
FIG_W = R5.FIG_W2
FONT_PT = R5.FONT_PT
ARM_COLORS = {"ESP-C": "#2a78d6", "ESP-O": "#8fb8ea", "OUR-O": "#eb6834", "OUR-C": "#f4ad8a"}

# re-exported helpers (single source in rev5_common)
sha256, write_atomic, write_json, nmae, nadeau_bengio = R5.sha256, R5.write_atomic, R5.write_json, R5.nmae, \
    R5.nadeau_bengio


def holm(pvals):
    """Holm step-down adjusted p-values (same order as input)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, (m - k) * p[i])
        adj[i] = min(1.0, run)
    return adj


def gate_prereg():
    """Training gate (§7-3, as rev 5 final_eval.gate_prereg): PREREG_REV6.md and prereg_rev6.json exist, are committed
    and unchanged in the working tree, and every input file listed in prereg_rev6.json["inputs_sha256"] still has that
    sha256. Exits 3 otherwise; returns the prereg record."""
    import json
    import subprocess
    import sys

    def git(*a):
        return subprocess.run(["git", "-C", str(HERE), *a], capture_output=True, text=True)
    bad = []
    for p in (PREREG6_MD, PREREG6_JSON):
        if not p.is_file():
            bad.append(f"{p.name} missing")
        elif git("ls-files", "--error-unmatch", str(p)).returncode != 0:
            bad.append(f"{p.name} not committed")
        elif git("diff", "--quiet", "HEAD", "--", str(p)).returncode != 0:
            bad.append(f"{p.name} modified since its commit")
    if not bad:
        rec = json.loads(PREREG6_JSON.read_text())
        for path, sha in rec["inputs_sha256"].items():
            if not Path(path).is_file():
                bad.append(f"input {path} missing")
            elif sha256(path) != sha:
                bad.append(f"input {path} changed (sha256 {sha256(path)[:12]} != prereg {sha[:12]})")
    if bad:
        print("PREREG gate (§7-3: commit PREREG_REV6 before any training):\n  " + "\n  ".join(bad), file=sys.stderr)
        sys.exit(3)
    return dict(rec, git_head=git("rev-parse", "HEAD").stdout.strip())


def our_columns(base_cols):
    """The 182 feature columns of FEAT_OUR (base_cols = train_ml_single.FEATURE_SETS)."""
    cols = R5.arm_columns(FEAT_OUR, base_cols)
    assert len(cols) == N_FEAT[FEAT_OUR], len(cols)
    return cols
