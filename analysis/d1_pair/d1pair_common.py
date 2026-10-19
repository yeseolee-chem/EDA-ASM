#!/usr/bin/env python3
"""d1pair_common.py — shared conventions of D1PAIR rev 9 (docs/specs/D1PAIR_REV9.md).

Single source of truth for the D1PAIR paths, seeds, arms, models, targets, scale rules, stage criteria, gate
tolerances and the T2 family list. Everything fixed in rev 4/5/6 is imported from rev6_common (which re-exports
rev5_common); nothing is copied. Import it; do not copy its constants.
Light on imports on purpose: it is imported in reactot (python 3.10), the rev 5 venv and env_espley (python 3.11).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent                 # analysis/d1_pair
REPO = HERE.parent.parent
ESP_DIR = REPO / "analysis" / "espley_xtb_repro"        # rev 4-6 code (read-only; espley-rev6 957f1fc)
if str(ESP_DIR) not in sys.path:
    sys.path.insert(0, str(ESP_DIR))
import rev6_common as R6  # noqa: E402

R5 = R6.R5

# ---------------------------------------------------------------- paths (D1PAIR writes only under RES and SCRATCH)
RES = HERE / "results"
SCRATCH = Path(os.environ.get("D1PAIR_SCRATCH", "/gpfs/tmp_cpu2/yeseo1ee/d1_pair"))
LOGS = SCRATCH / "logs"
# D1 inputs (never modified)
D1_RUN = Path("/gpfs/tmp_cpu2/yeseo1ee/d1/d1_run")      # = D1_build/d1_run (symlink in the main checkout)
D1_LABELS = D1_RUN / "results" / "labels_d1.json"
D1_STATUS = D1_RUN / "results" / "d1_status.csv"
D1_GEOM = D1_RUN / "results" / "geometries"             # <job_id>/{ts,ref_dipole,ref_dipolarophile,product}.xyz
NURION_TARS = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/D1_build/nurion")   # 23 bundle tars
UBAI_JOBS = Path("/gpfs/tmp_cpu2/yeseo1ee/d1/d1_scratch/jobs")                           # rows not harvested
D1_LABELS_SHA = "1eaa417102e20813beea9ba190ef8140233a4a9b0b1d9a69afac16aeaeb210a7"
D1_STATUS_SHA = "a9df873b5021fb11e0d2bd2b2ba1b012abad9e7981f32b676172ec269cabfcf7"
D1_INTERIM_SHA_PREFIX = "a2df136b"                      # the interim 9,459-row file once uploaded as labels_all.json
N_D1_LABELS = 9496
# <D1_JOBS>: user decision 2026-10-19 (Phase 0 STOP, no unpacked job folder on the server): extracted read-only
# from the bundle tars / UBAI folders by d1_extract_jobs.py
D1_JOBS = SCRATCH / "d1_jobs"
JOB_FILES = ("ts.xyz", "ref_dipole.xyz", "ref_dipolarophile.xyz", "label.json", "sp/eda.inp")
# D0 inputs (rev 6; sha256 in results_rev6/prereg_rev6.json)
D0_LABELS = REPO / "labels_all.json"
D0_SMILES = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/dipolar_cycloaddition/full_dataset.csv")   # = g1_geom.SMILES_CSV
D0_META = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/label_true/work/input_meta.csv")
D0_PREREG6 = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-rev4/analysis/espley_xtb_repro/results_rev6"
                  "/prereg_rev6.json")
N_D0 = 4963                                             # rev 6 appendix A rows = f0 training rows

# Phase P / G / F files
ROWATTR = SCRATCH / "P" / "rowattr.pkl"                 # build_rowattr.py output (kept rows; .all.pkl = all rows)
SIGMA = HERE / "sigma_hansch1991.csv"
PAIRS = RES / "pairs_d1.csv"                            # appendix B (P_all)
ANCHOR = RES / "pairs_d1_anchor.csv"                    # appendix E (A_all)
G_TARGETS = RES / "G_targets.csv"                       # the 8,659 G1 rows in run order
D1G_LABELS = SCRATCH / "G" / "labels_d1_g.json"         # D1 label copy (8,659 rows) for the rev 6 code
D1G_META = SCRATCH / "G" / "input_meta_d1.csv"          # rxn_id, A_idx from sp/eda.inp (1) tags
D1G_SMILES = SCRATCH / "G" / "smiles_d1.csv"            # rxn_id, atom-mapped rxn_smiles (map = TS index + 1)
PROF_D1 = SCRATCH / "prof"                              # <rxn_id> -> D1_JOBS/<job_id>
G1_ROOT_D1 = SCRATCH / "g1"                             # first G1 attempt (g1_geom.process)
G1_ROOT_D1R = SCRATCH / "g1_rec"                        # recovery ladders (d1_recover.py)
G1_ROOT_D1F = SCRATCH / "g1_final"                      # <rxn_id> -> the g1 or g1_rec folder (features read here)
X182_D1 = SCRATCH / "X182_d1.parquet"                   # rxn_id + the 182 features (X182_all column order)

# rev 4-6 locations no D1PAIR step may write into (§7-2 STOP)
FORBIDDEN_WRITE = (Path("/gpfs/tmp_cpu2/yeseo1ee/espley_rev4"), Path("/gpfs/tmp_cpu2/yeseo1ee/espley_rev5"),
                   Path("/gpfs/tmp_cpu2/yeseo1ee/espley_rev6"), Path("/gpfs/tmp_cpu2/yeseo1ee/espley_xtb"),
                   Path("/gpfs/tmp_cpu2/yeseo1ee/espley_xtb_g1"), Path("/gpfs/tmp_cpu2/yeseo1ee/espley_xtb_g1_rev6"),
                   Path("/gpfs/tmp_cpu2/yeseo1ee/espley_xtb_g2"), Path("/gpfs/tmp_cpu2/yeseo1ee/d1"),
                   Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw"),
                   ESP_DIR / "results_rev4", ESP_DIR / "results_rev5", ESP_DIR / "results_rev6",
                   Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-rev4"),
                   Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/D1_build"))

# ---------------------------------------------------------------- seeds (§4, §3-4, §3-3)
SEED = 20261019                         # fold shuffle, bootstraps, sampling of check rxns
CV_SEED = 22                            # f0 KFold(5) and the fixed NN 10 % validation split
TUNE_SEED = 23                          # Hyperband seed (r6_nn.TUNE_SEED)
assert TUNE_SEED == 23

# ---------------------------------------------------------------- targets (§2-2, §3-4)
TARGETS = ["d1", "d2", "elst", "pauli", "oi", "disp", "cpcm", "cds", "c_ghost", "barrier"]
CHANNELS9 = [t for t in TARGETS if t != "barrier"]       # dd_barrier = sum of these 9 (identity)
D0_TCOL = {**R6.TARGETS_A, "barrier": R6.TARGETS_B["dEa"]}   # = r6_train.tcol: dft_* column of each target
assert D0_TCOL["c_ghost"] == "dft_c_ghost_kcal" and D0_TCOL["barrier"] == "dft_barrier_kcal"
D1_TCOL = {"d1": "d1_kcal", "d2": "d2_kcal", "elst": "elst_dft", "pauli": "pauli_dft", "oi": "oi_dft",
           "disp": "disp_dft", "cpcm": "cpcm_dft", "cds": "cds_dft", "c_ghost": "bsse_shift_kcal",
           "barrier": "barrier_kcal"}                    # = d1_pairs.TARGETS
PRIMARY_TARGETS = ["d1", "d2", "elst", "pauli", "oi", "disp", "cpcm", "cds", "barrier"]   # 8 channels + barrier
HEADLINE_EXCLUDE = ("disp",)            # §3-1: reported, not a headline
POSITIONS = ["p", "m", "o", "direct"]
K_LOG = 1.3642                          # Hammett value = -dd / K_LOG (= hammett_eval.K)

# ---------------------------------------------------------------- arms and models (§3)
ARMS_G = ("B", "C")                     # arms with a trained g (+ B_ref, P below)
MODELS = list(R6.MODELS)                # Ridge, KRR, SVR, XGB, NN2, NN4
NN_SIZE = dict(R6.NN_SIZE)
GAMMAS = [3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]   # gamma_delta, gamma_xbar (KRR, SVR)
SVR_C = [1, 10, 100, 1000]
SVR_EPS = [0.05, 0.1, 0.5]
N_INNER = 4                             # GroupKFold(4), group = core
SCALE_EPS = 1e-12                       # dropped columns: s_delta or s_xbar < SCALE_EPS on the training pairs
ANTISYM_TOL = 1e-6                      # gate M-1, kcal/mol
ANTISYM_N = 200


def grid_values(key):
    """values of a rev 6 train_ml_single.GRIDS entry with the pipeline prefix removed: {param: values}."""
    import train_ml_single as T
    _, g = T.GRIDS[key]
    return {k.split("__")[-1]: list(v) for k, v in g.items()}


# ---------------------------------------------------------------- splits (§4)
N_FOLDS = 5
SPLITS = ("T1", "LODO", "LOSO", "T2")
DPP = ("DppC", "DppT")                  # D'' skeletons (excluded rows)
T2_FAMILIES = ["azide/A/R2//N-//", "azide/AppZ/R1////", "carbonyl ylide/Bp/R1/t//endo/", "nitrile oxide/B/R1///endo/",
               "nitrile oxide/B/R1///exo/", "nitrone/A/R2//C//", "nitrone/A/R2//O-//", "nitrone/B/R1/c//exo/cis",
               "nitrone/B/R2//O-/exo/", "nitrone/C/R2//O-//", "nitrone/Cp/R1/t///"]
T2_N_PAIRS_PALL = 827
T2_HIDDEN_V = ["carbonyl ylide/Bp/R1/t//endo/", "nitrone/A/R2//C//", "nitrone/A/R2//O-//", "nitrone/B/R1/c//exo/cis",
               "nitrone/C/R2//O-//", "nitrone/Cp/R1/t///"]
T2_N_HIDDEN_BEND = 30

# ---------------------------------------------------------------- Phase P expected values (§2)
EXPECT_P1 = dict(n_labels=9496, excluded=461, kept=9035,
                 pos={"m": 2321, "p": 2283, "o": 2240, "direct": 2028, "ref_aryl": 110, "ref_direct": 53})
EXPECT_P2 = dict(n_pairs=7253, pos={"p": 1878, "m": 1910, "o": 1734, "direct": 1731}, families=146, cores=46,
                 refs=157, refs_aryl=104, refs_H=53, reactions=7410, para_fam_12=96, shared_refs=35,
                 shared_ref_pairs=1403, hygiene_dropped=75, hygiene_dropped_refs=0,
                 sha256="227063285d65a05195a0d662df040618ac52bd175718837423ded2343c83ecce")
EXPECT_P3 = dict(n_pairs=1159, pos={"o": 354, "p": 293, "m": 284, "direct": 228}, families=75, reactions=1249,
                 cores=33, cores_new=["azomethine ylide/D", "carbonyl ylide/C", "carbonyl ylide/D", "carbonyl ylide/Dp"],
                 hygiene_dropped=37, sha256="2121c684bd11a77d4b59128432d3e6e22d3dab23081bd25bd9864e5172a9bb6c")
N_G_TARGETS = 8659
EXPECT_P4_STEREO_DIFF = 852             # reference pairs whose a / b stereo_index differ
# P-4 user decision 2026-10-19: the literal sign rule flags mirror-image TSs and in-plane sp2 X sites; the pairs that
# mismatch under the stereo rule (v2: possible stereocentres only, sign pattern up to a mirror flip) are dropped from
# P_feat / P_main / P_sens / A_feat / A_main / T_main; P_all / A_all stay the appendix outputs (count gates, appendix D)
P4_DROP = RES / "P4_drop.csv"
P4_DECISION = "drop_v2_mismatch (user 2026-10-19)"
# Si user decision 2026-10-19: rev 4-6 element tables extended by site_si/sitecustomize.py (G / F jobs)
SI_DECISION = "Si added to fragmenter.COV (1.11 A) / Z (14), rev5_common.BONDI (2.10 A) / GFN2_VALENCE (4) (user 2026-10-19)"
# appendix D self-test on P_all DFT values (§5-2)
EXPECT_D = dict(n_fam=96, bent={"barrier": 33, "d1": 34, "d2": 15, "elst": 21, "pauli": 24, "oi": 27, "disp": 5,
                                "cpcm": 26, "cds": 12},
                V={"barrier": 2, "d1": 3, "d2": 0, "elst": 3, "pauli": 3, "oi": 5, "disp": 0, "cpcm": 0, "cds": 0},
                hidden_bend=30, hidden_V=6)
APPENDIX_SHA = {"build_rowattr.py": "81cb660e8b0852fd6d114d179955039b80b422c056752627dca0b432a4ef86d4",
                "d1_pairs.py": "424c2bc24b4ed5111c35a00688a88d43f47dffb2e61e50c5326256f31be603e1",
                "sigma_hansch1991.csv": "2ba63224928301657f09707cb3be1b0e05a6c69ef1e733b490b9a4ac6261d06f",
                "hammett_eval.py": "e87d944ef6b35f1a8ae44884270a2d5530bbc32f1f53a55ef7e194ffbffac74b",
                "d1_anchor_pairs.py": "f979e1e336df850a0b55d6ed762718b035bf2b151468263919b53a5aa40bac18"}

# ---------------------------------------------------------------- gates (§7, §8, §3-4)
G1_TOL_COORD_A = 1e-6                   # G-1 (a) eda.inp coordinates vs ts.xyz
G1_TOL_FORM_A = 1e-3                    # G-1 (c) sorted formed-bond lengths vs [formed_d1, formed_d2]
G2_MAX_FAIL = 87                        # G-2: > 87 (1 %) failures = STOP
G_N_CHECK_D0 = 5                        # G-0 D0 rxns (seed SEED, from rev 6 R_feat)
G_REF_FAIL_STOP = 16                    # >= 16 of the 157 reference reactions failing = STOP (> 10 %)
CORE_LOSS_STOP = 0.5                    # a core losing more than half of its reference pairs = STOP
F0_N_CHECK = 20                         # F-0 D0 rxns (seed SEED, rev 6 R_feat minus rxn 4994)
F0_TOL = 1e-8                           # F-0: every block but B1
F0_TOL_B1 = 1e-6                        # F-0: B1 (tblite)
# gate F0-1: representative 5-fold CV MAE <= 1.2 x the rev 6 appendix test MAE (barrier: rev 6 B OUR-O KRR,
# c_ghost: rev 6 main NN-4)
F0_REF_MAE = {"d1": 1.70, "d2": 1.25, "elst": 3.55, "pauli": 6.45, "oi": 3.98, "disp": 0.470, "cpcm": 1.54,
              "cds": 0.222, "c_ghost": 0.233, "barrier": 1.70}
F0_FACTOR = 1.2

# ---------------------------------------------------------------- stages and tests (§5)
STAGE1_R2 = 0.5
STAGE2_MAE = 0.13
STAGE2_TARGETS = ("oi", "cpcm")         # orbital and solvent channels
STAGE2_SENS = 0.7
STAGE2_FPR = 0.1
N_BOOT = 2000
ALPHA = 0.05
MIN_X_FAM = 12                          # para family with >= 12 points (H included)
FIG3_FAMILIES = ["nitrile oxide/C/R2//O-//", "nitrile oxide/C/R1////", "nitrile oxide/Cp/R1////", "azide/D/R1////",
                 "carbonyl ylide/B/R1/c//exo/trans", "azomethine ylide/Bp/R1/c//endo/"]
FINE3 = ("disp", "cds", "c_ghost")      # MAE to 3 decimals
FIG_W = R6.FIG_W
FONT_PT = R6.FONT_PT

# ---------------------------------------------------------------- helpers
sha256, write_json = R6.sha256, R6.write_json
holm = R6.holm
hygiene_mask = R6.hygiene_mask


def assert_write_ok(path):
    """STOP (exit 3) if path lies inside a rev 4-6 / D1 input location (§7-2)."""
    p = Path(path).resolve() if Path(path).exists() else Path(os.path.abspath(path))
    for f in FORBIDDEN_WRITE:
        try:
            p.relative_to(f)
        except ValueError:
            continue
        print(f"STOP: write path {p} is inside the protected location {f}", file=sys.stderr)
        sys.exit(3)
    return p


def write_atomic(path, data):
    assert_write_ok(path)
    return R5.write_atomic(path, data)


def write_json_safe(path, obj):
    assert_write_ok(path)
    return R5.write_json(path, obj)


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def list_sha(rows) -> str:
    """sha256 of a list of tuples, one comma-joined line each (order kept)."""
    return sha256_text("\n".join(",".join(map(str, r)) for r in rows) + "\n")


def load_d1_labels():
    """labels_d1.json after the sha256 / row checks of §1 (exits 3 otherwise): list of records."""
    if sha256(D1_LABELS) != D1_LABELS_SHA:
        print(f"STOP: {D1_LABELS} sha256 != {D1_LABELS_SHA}", file=sys.stderr)
        sys.exit(3)
    lab = json.loads(D1_LABELS.read_text())
    if len(lab) != N_D1_LABELS or any(d.get("status") != "ok" for d in lab):
        print("STOP: labels_d1.json is not 9,496 status-ok rows", file=sys.stderr)
        sys.exit(3)
    return lab
