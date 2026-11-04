#!/usr/bin/env python3
r"""finetune.py — DEPLOY-V2 (MLTS) Phase FT: fine-tune a foundation MLIP to the D1 label level
(B3LYP-D3BJ/def2-SVP, SMD water) energies + forces (docs/specs/DEPLOY_V2_MLTS.md 5; budget 0-5; leakage C-2 (8);
LODO (12); final committee (13)). Compute nodes only (sbatch / jobs/sbx.sh): python never runs on the login node.

Runs (data split in brackets; suffix _al = + G-4 active-learning data, also set by --with-al)
  FT_<k>            k = 0..4   [fold<k>]   drop every structure with k in folds(tag)          (FT-2 / A evaluation)
  FT_all_s<i>       i = 0..2   [all_s<i>]  all data; validation split and MACE seed differ by i (deployment committee)
  FT_LODO_<d>       d = 0..4   [lodo<d>]   drop every structure with DIPOLES[d] in dipoles(tag)   (Phase X)
  FTpilot_<cand>_<k>           [fold<k>]   Z pilot (spec 4: fold-0 fine-tune of the top-2 Z candidates)
  BENCH_<cand>[_<tag>]         [fold0]     timing / smoke runs (--subset), never used downstream
Candidates: MACE-OMOL-0 (default; mace-torch 0.3.16 ships only the extra-large 1024-channel OMOL model),
  MACE-MH-1-omol (smaller multi-head MACE, omol head; optional size alternative), AIMNet2 (aimnet2_wb97m_d3_0),
  MACE-OFF23-medium / MACE-OFF24-medium (no Si: refused, see CAND_NOTES).

Data (decision FT-d0, user decision 'D0' 2026-11-04): I = $MLT/dft/existing.extxyz, G1/G2/G3 = gen_G{1,2,3}.extxyz
(D1), D0G1/D0G2/D0G3 = gen_D0G{1,2,3}.extxyz (D0, read whenever present), G4 = gen_G4.extxyz (--with-al). Inputs are
complete when I + G1-3 exist and results/G_report.json is complete, and — once $MLT/dft/plan/plan_d0.parquet exists —
results/G_report_d0.json is complete for that plan (train refuses incomplete splits unless --allow-partial). D0 rows have
no rxn_ids (every FT_k keeps them; train only) except the TS fragments of D1-shared molecules (D1 union tags, D1 rules)
and dipoles = their D0 class (FT_LODO_d drops class d); Phase I D0 rows get their class from d0_class.csv; Phase I rows
of the 15 D1-design d1_validation groups (D1-NNNN-rK, no D1 target) are tagged with every T1 fold and every D1 dipole
(FT_all only; DECISIONS FT-d1val). The C-2 check also asserts d0_class within dipoles (STOP otherwise). TS-near (FT-1)
includes the D0 kinds (DECISIONS FT-ts-near). Evaluation stays on the D1 T1 folds.
D0 holdout (user decisions 'D0-plus' + 'D0-gauss', DECISIONS FT-d0hold): every structure whose D0 reaction (info d0_id,
or the Phase I group 'D0-<rid>') is in $MLT/dft/plan/d0_holdout.csv (evaluation reactions + their reactant-pair
siblings) is dropped from EVERY run (all splits, with or without AL) except the reactant molecules (d0_react_ref /
d0_react_disp); the C-2 check also derives the D0 reaction from the struct id (d0g1_<rid>_*, d0g3_<rid>_f*,
d0g2_<rid>_s*), independent of the info, STOPs on a holdout TS-derived row, on a D0 TS kind without any D0 reaction and
on a G4 row that is neither a D1-target row nor d0_id-tagged; the holdout file must match every recorded sha256
(plan, results copy, data jsons; missing / changed = STOP); its sha256 enters the split fingerprint and every data json.
Nurion (user decision 'Nurion', DECISIONS FT-nurion): NUR = $MLT/dft/nurion.extxyz.gz (inv_nurion collect; kinds nur_*
of D1-target jobs, nur_x_* of every other job; leakage tags set by inv_nurion) is a required input of every run (complete
= present and described by results/N_report.json; N_report complete = all 23 tars is recorded, the pipeline N-gate
decides), read once per build as a stream (no gzip seek, no full temp copy) after D0G3 and before G4 (dedupe priority).
Same leakage rules (FT_k / FT_LODO drop by tags OR targets(), test = all-held-out rxn_ids); Nurion rows without rxn_ids
(non-target jobs) carry their core's T1 fold + own dipole and are validation rows when their job core is a validation core.
Size cap (user decision 'FT-cap', DECISIONS FT-cap): every split's train set <= 200,000 and validation set <= 20,000
structures, same for every candidate (test uncapped unless --cap-test): tier 1 = all non-Nurion rows (always kept), tier 2
= Nurion TS-near rows (nur_ts_final, or a nur_ TS kind with dform_label_A <= 0.3 A), tier 3 = every other Nurion row; the
room left by tier 1 goes 70 % / 30 % to tiers 2 / 3 (seeded round-robin over job ids). TS-near (FT-1) includes the nur_ TS
kinds within 0.3 A of the label TS.

CLI (submit from analysis/mlts; exit 0 = done / up to date, 2 = not ready / incomplete (resubmit), 3 = STOP)
  data <run ...> | --all  [--with-al] [--fmax-cap 10] [--valid-frac 0.10] [--cap-train 200000] [--cap-valid 20000]
        [--cap-test 0] [--force] [--dry]
        MLTENV=mlts or reactot, CPU: $MLT/ft/data/<split>/{train,valid[,test]}.xyz + meta.json,
        results/ft/<run>_data.json (counts by kind / source / tier before and after the cap, cap record, C-2 leakage
        check, input sha256 incl. nurion.extxyz.gz + N_report.json); --cap-* 0 = no cap;
        --dry: read + split + cap + C-2 check only, counts printed, nothing written
  fetch <cand>                     download the foundation into $MLT/cache (reuses ~/.cache/mace read-only)
  train <run> <cand> [--with-al] [--epochs 200] [--budget-h 12] [--stop-h 24] [--batch-size B] [--lr X]
        [--ew X] [--fw X] [--patience 6] [--subset N] [--allow-partial] [--force]
        MLTENV=mlts, 1 GPU: $MLT/ft/<run>/ (checkpoints, logs, E0s.json, budget.json, epochs.jsonl),
        $MLT/models/<run>.{model|jpt} + <run>.json (model card), results/ft/<run>_train.json
  ft1 <run> [--with-al] [--dtype float32] [--foundation <cand>]
        MLTENV=mlts, 1 GPU: FT-1 (record only) on valid (+ held-out test side) -> results/ft/<run>_ft1.json
  status                           one line per run

sbatch (from analysis/mlts; login: one-shot submits only; `squeue -u $USER -h | wc -l` first; L=$MLT/logs):
  (data, measured 11-05 with 658,901 Nurion structures: pool 5.6 min (Nurion stream 279 s), peak RSS 6.1 GiB for 2 splits)
  JD=$(sbatch --parsable -p cpu1,cpu2 -J ft_data --time=48:00:00 -c 4 --mem=48G --output=$L/%x_%j.out \
       --dependency=afterok:<G collect job> --kill-on-invalid-dep=yes --export=ALL,MLTENV=mlts \
       jobs/mlt.slurm finetune.py data FTpilot_MACE-OMOL-0_0 FTpilot_AIMNet2_0)
  sbatch -p gpu1,gpu3,gpu4,gpu5 -J ft_p_omol --time=48:00:00 --gres=gpu:1 -c 8 --mem=96G --output=$L/%x_%j.out \
       --dependency=afterok:$JD --kill-on-invalid-dep=yes --export=ALL,MLTENV=mlts \
       jobs/mlt.slurm finetune.py train FTpilot_MACE-OMOL-0_0 MACE-OMOL-0 ::: finetune.py ft1 FTpilot_MACE-OMOL-0_0
  sbatch -p gpu1,gpu2,gpu3,gpu4,gpu5,gpu6 -J ft_p_aim --time=48:00:00 --gres=gpu:1 -c 8 --mem=64G \
       --output=$L/%x_%j.out --dependency=afterok:$JD --kill-on-invalid-dep=yes --export=ALL,MLTENV=mlts \
       jobs/mlt.slurm finetune.py train FTpilot_AIMNet2_0 AIMNet2 ::: finetune.py ft1 FTpilot_AIMNet2_0
  A 48 h wall / node loss: resubmit the same command (budget.json keeps the GPU-h, MACE --restart_latest, AIMNet2
  aim_last.pt). Exit 3 = STOP (> 24 GPU-h or C-2 leakage).

Units: extxyz REF_energy eV (absolute label level), REF_forces eV/A; training energy key E_fit = REF_energy -
sum_i ATOM_REF_EH[Z_i] * EH2EV (fixed per-element constants, only to keep float32 targets small; forces unchanged);
exported models predict absolute label-level energies (E0 += ATOM_REF) — see the model card.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import urllib.request  # noqa: E402
from collections import Counter, deque  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mlt_common as M  # noqa: E402

CODE_VERSION = "mlts-ft-4"          # -2: FT-d1val overlay, D0 class check, AIMNet2 charges (review); splits rebuilt
                                    # -3: D0-plus / D0-gauss holdout drop (+ siblings) + independent C-2 holdout / G4
                                    #     checks (FT-d0hold); splits rebuilt (no production data built with -3 yet)
                                    # -4: Nurion input (FT-nurion) + per-split size cap with tiers (FT-cap), nur_ TS kinds
                                    #     in TS-near (FT-ts-near); splits rebuilt
T_START = time.time()

# ---------------------------------------------------------------- paths
DFT = M.MLT / "dft"
FT = M.MLT / "ft"
FT_DATA = FT / "data"
MODELS = M.MLT / "models"
CACHE = M.MLT / "cache"
RES_FT = M.RES / "ft"
INPUTS = (("I", DFT / "existing.extxyz"), ("G1", DFT / "gen_G1.extxyz"), ("G2", DFT / "gen_G2.extxyz"),
          ("G3", DFT / "gen_G3.extxyz"))
D0_INPUTS = (("D0G1", DFT / "gen_D0G1.extxyz"), ("D0G2", DFT / "gen_D0G2.extxyz"), ("D0G3", DFT / "gen_D0G3.extxyz"))
AL_INPUT = ("G4", DFT / "gen_G4.extxyz")
NUR = "NUR"
NUR_INPUT = (NUR, DFT / "nurion.extxyz.gz")         # inv_nurion collect (user decision 'Nurion'; DECISIONS FT-nurion)
N_REPORT = M.RES / "N_report.json"                  # inv_nurion collect report (extxyz_bytes, n_structures, complete)
D1_MANIFEST = M.D1RUN / "d1_manifest.csv"           # job_id -> dipole / skeleton (core of non-target Nurion jobs)
G_REPORT = M.RES / "G_report.json"
G_REPORT_D0 = M.RES / "G_report_d0.json"
PLAN_D0 = DFT / "plan" / "plan_d0.parquet"           # exists = the D0 data are required (user decision 'D0')
D0_CLASS = DFT / "plan" / "d0_class.csv"            # gen_dft plan_d0: D0 rxn_id -> dipole class (Phase I D0 overlay)
D0_HOLDOUT = DFT / "plan" / "d0_holdout.csv"        # gen_d0 plan_d0 --n-holdout (D0-plus): evaluation-only D0 reactions
D0_HOLDOUT_RES = M.RES / "d0_holdout.csv"           # its results/ copy (a record of its sha256)
G2_PLAN_D0 = DFT / "plan" / "g2_plan_d0.json"       # its 'holdout' record (n, sha256) must match the file
PLAN_D0_META = DFT / "plan" / "plan_d0_meta.json"   # holdout_sha256 must match the file
D0_HOLD_KEEP_KINDS = ("d0_react_ref", "d0_react_disp")   # reactant molecules carry no TS geometry: kept (D0-plus)
D0_HOLD_TS_KINDS = ("d0_ts_near", "d0_ts_imag", "d0_ts_label", "d0_frag", "d0_g2_path", "d0_ts_final_g2")
SID_D0 = re.compile(r"d0g1_(\d+)_(?:n\d+|i|lab)|d0g3_(\d+)_f\d+|d0g2_(\d+)_s\d+")   # gen_d0 struct ids of TS rows
SITE = Path(M.CONDA) / "envs" / "mlts" / "lib" / "python3.10" / "site-packages"
HOME_MACE_CACHE = Path.home() / ".cache" / "mace"

# ---------------------------------------------------------------- constants
EV2KCAL = M.EH2KCAL / M.EH2EV                      # 23.0605 kcal/mol per eV
ATOM_REF_EH = {1: -0.5, 6: -37.8, 7: -54.5, 8: -75.0, 9: -99.7, 14: -289.3, 16: -398.0, 17: -460.1, 35: -2573.9}
DEDUPE_DEC = 5                                     # positions rounded to 1e-5 A for the exact-duplicate key
FMAX_CAP = 10.0                                    # eV/A, per-structure max atomic |F| (decision FT-fcap)
VALID_FRAC = 0.10
TS_NEAR_ALWAYS = ("ts_near", "ts_imag", "ts_label", "d0_ts_near", "d0_ts_imag", "d0_ts_label", "d0_ts_grad")
NUR_TS_KINDS = ("nur_ts_path", "nur_ts_final", "nur_tsguess_opt", "nur_path", "nur_neb", "nur_fwd_bwd")  # inv_nurion
TS_NEAR_IF_CLOSE = ("ts_path", "ts_final", "g2_path", "ts_final_g2", "al_ts", "d0_g2_path", "d0_ts_final_g2") + NUR_TS_KINDS
TS_NEAR_DFORM_A = 0.3
CAP_TRAIN, CAP_VALID, CAP_TEST = 200_000, 20_000, 20_000  # decision FT-cap (structures per split; 0 = no cap);
#   test capped too (lead 11-05: FT-1 test-side metrics are descriptive, FT-2 uses reaction lists; saves ~15-25 GB)
CAP_DEFAULTS = dict(train=CAP_TRAIN, valid=CAP_VALID, test=CAP_TEST)
CAP_TIER2_FRAC = 0.70                               # share of the room left by tier 1 that goes to tier 2 (FT-cap)
CAP_SEED_OFF = 1000                                 # cap seed = SEED + 1000 + the split offset (parse_run)
NUR_KEEP_INFO = ("d0_id",)                          # info a Nurion structure keeps after its row is built (memory)
NUR_READ_CHUNK = 5000                               # frames per parsed chunk of the streamed nurion.extxyz.gz
D1DESIGN_GROUP = re.compile(r"D1-\d+-r\d+")          # Phase I d1_validation D1-design groups (DECISIONS FT-d1val)
ALL_T1_FOLDS = ("0", "1", "2", "3", "4")
FT1_TARGET = dict(force_mae_kcal_A=0.6, rel_e_mae_kcal=0.3)
BUDGET_H, STOP_H = 12.0, 24.0
ALL_RUNS = [f"FT_{k}" for k in range(5)] + [f"FT_all_s{i}" for i in range(3)] + [f"FT_LODO_{d}" for d in range(5)]
B3LYP_D3BJ = dict(s6=1.0, s8=1.9889, a1=0.3981, a2=4.4211)   # Grimme D3(BJ) B3LYP parameters

CANDIDATES = {
    "MACE-OMOL-0": dict(kind="mace", file="MACE-omol-0-extra-large-1024.model", head=None, si=True,
                        url="https://github.com/ACEsuit/mace-foundations/releases/download/mace_omol_0/"
                            "MACE-omol-0-extra-large-1024.model",
                        level="OMol25 wB97M-V/def2-TZVPD"),
    "MACE-MH-1-omol": dict(kind="mace", file="mace-mh-1.model", head="omol", si=True,
                           url="https://github.com/ACEsuit/mace-foundations/releases/download/mace_mh_1/mace-mh-1.model",
                           level="MACE-MH-1 omol head (OMol25)"),
    "MACE-OFF23-medium": dict(kind="mace", file="MACE-OFF23_medium.model", head=None, si=False,
                              url="https://raw.githubusercontent.com/ACEsuit/mace-off/main/mace_off23/MACE-OFF23_medium.model",
                              level="SPICE wB97M-D3(BJ)/def2-TZVPPD"),
    "MACE-OFF24-medium": dict(kind="mace", file="MACE-OFF24_medium.model", head=None, si=False,
                              url="https://github.com/ACEsuit/mace-off/raw/main/mace_off24/MACE-OFF24_medium.model",
                              level="SPICE 2 wB97M-D3(BJ)/def2-TZVPPD"),
    "AIMNet2": dict(kind="aimnet", file="aimnet2_wb97m_d3_0.jpt", head=None, si=True,
                    url="https://storage.googleapis.com/aimnetcentral/AIMNet2/aimnet2_wb97m_d3_0.jpt",
                    yaml="aimnet2_dftd3_wb97m.yaml", level="wB97M-D3/def2-TZVP"),
}
ALIASES = {"mace_omol": "MACE-OMOL-0", "omol": "MACE-OMOL-0", "mace-omol-0": "MACE-OMOL-0", "aimnet2": "AIMNet2",
           "mh-1": "MACE-MH-1-omol", "mace-mh-1-omol": "MACE-MH-1-omol", "off23": "MACE-OFF23-medium",
           "off24": "MACE-OFF24-medium"}
CAND_NOTES = {"no_si": "foundation has no Si; 465 D1 reactions carry SiMe3 and mace load_foundations_elements needs every "
                       "training element in the foundation z-table -> not fine-tunable on the D1 data (spec 4: Si-free "
                       "reference only)"}

DECISIONS = {
    "FT-split": "split = drop (C-2 rule on the tags, OR'ed with the targets() fold / dipole of every rxn_id — both must "
                "agree, a disagreement is counted) -> validation = whole fold_core groups ('<dipole>/<skeleton>', "
                "mlt_common.targets) drawn in a seeded random order until >= 10 % of the remaining D1 reactions; a "
                "structure is validation iff it has D1 rxn_ids and all of them are validation reactions (molecules "
                "shared with training reactions and untagged D0 data -> train); a Nurion row without rxn_ids whose job "
                "core is a validation core is validation too (FT-nurion). Seeds: SEED+100+k (fold k), SEED+200+i "
                "(all_s<i>), SEED+300+d (lodo d). test = dropped structures whose rxn_ids are all held out. Then the "
                "size cap (FT-cap).",
    "FT-dedupe": "exact duplicates: same element sequence and positions rounded to 1e-5 A; the first occurrence (input "
                 "order I, G1, G2, G3, D0G1-3, NUR (Nurion), then G4) is kept and gets the union of the leakage tags; "
                 "|dE|, |dF| recorded. A non-Nurion row duplicated by a Nurion row stays (tier 1) with the united tags; "
                 "a Nurion row that a G4 row duplicates counts as tier 1 in the *_al splits (FT-cap).",
    "FT-fcap": "per-structure max atomic |F| > 10 eV/A (0.19 Eh/bohr; early OptTS / LooseOpt steps of Phase I) is "
               "dropped; the measured distribution is in the split meta (fmax_eV_A).",
    "FT-frame": "training energy E_fit = REF_energy - sum ATOM_REF_EH[Z] x EH2EV (fixed, rounded per-element constants): "
                "float32 total energies of ~2-7e4 eV are quantised at 2-8 meV (0.05-0.18 kcal/mol, Br worst); the shift "
                "keeps |E_fit| ~ 1e2 eV (spacing ~1e-5 eV). Forces untouched. Export adds ATOM_REF back to the E0s, so "
                "exported models predict absolute label-level energies (eV).",
    "FT-E0": "atomic reference energies by least squares: E0(Z) = E0_foundation(Z) + lstsq(counts, E_fit - "
             "E_foundation) on the training structures (<= 40,000, all structures with a rare element kept) — the "
             "formula of mace.data.estimate_e0s_from_foundation (MACE --E0s=estimated); MACE --E0s=average (verified "
             "in mace/data/utils.compute_average_E0s: lstsq of total energies on atom counts) would double count the "
             "foundation's interaction energy because load_foundations_elements copies the foundation scale/shift. "
             "AIMNet2: the same residual least squares with the pretrained atomic shifts zeroed (= aimnet SAE, mode "
             "linreg).",
    "FT-mace": "naive fine-tuning (mace_run_train --foundation_model <local file> --multiheads_finetuning False: no "
               "OMol25 replay data on disk; the deployment domain is D1 only). Loss 'weighted' energy_weight 1, "
               "forces_weight 1 (MACE README fine-tuning example), EMA 0.99, amsgrad, rms_forces_scaling; lr 1e-3 "
               "(README naive-FT example 1e-2 is for MP-small with batch 2; multihead FT forces 1e-4; 1e-3 for the "
               "1024-channel model), batch 8 (RTX 3090 bench: batch 8 = batch 16 structures/s, 9.6 vs 15.6 GB, 1.3x "
               "batch 4), ReduceLROnPlateau factor 0.8 patience 3; float32 training (FP64 "
               "is ~1/64 rate on A6000 / 3090 / A10), export float64. mace 0.3.16 patience does not stop a "
               "non-distributed run (tools/train.py exit_now is None) -> early stopping (6 epochs without a better "
               "validation loss) and the 12 GPU-h budget are enforced here: terminate, then finalise with "
               "--restart_latest --max_num_epochs <best epoch> (MACE loads the best checkpoint (saved with EMA "
               "weights) and writes the model). The final MACE train+valid error table (a full forward/force pass "
               "over the training set, ~20 min) is skipped; FT-1 computes the metrics.",
    "FT-aimnet": "AIMNet2 with the aimnet package parts (model yaml aimnet2_dftd3_wb97m + weights of the released "
                 "aimnet2_wb97m_d3_0.jpt, SizeGroupedDataset / Sampler, MTLoss energy_loss_fn + peratom_loss_fn, "
                 "Forces, pt2jpt export helpers) and a loop here (aimnet train needs wandb for metrics and has no time "
                 "budget). D3 head switched to B3LYP-D3(BJ) (s8 1.9889, a1 0.3981, a2 4.4211) = the label "
                 "dispersion, so the network only learns the electronic difference. Molecular charge input = the "
                 "structure's total_charge (D1 all 0; D0 has total -2 / +1 reactions) in training, E0 fit, export "
                 "check and FT-1. Loss weights energy 1 : forces 0.2 (aimnet default_train.yaml, partial-charges term "
                 "dropped: no label partial charges), RAdam lr 1e-4 (default 4e-4 is from scratch), grad clip 0.4, "
                 "ReduceLROnPlateau 0.75 / 3.",
    "FT-budget": "GPU-h = wall-h of the train command (1 GPU), summed over attempts (budget.json, updated each "
                 "minute). An epoch is not started when it would end past 12 h (target, spec 0-5); > 24 h = STOP "
                 "(exit 3).",
    "FT-ts-near": "TS-near (FT-1) = kind ts_near / ts_imag / ts_label / d0_ts_near / d0_ts_imag / d0_ts_label / "
                  "d0_ts_grad, or kind ts_path / ts_final / g2_path / ts_final_g2 / al_ts / d0_g2_path / d0_ts_final_g2 "
                  "/ nur_ts_path / nur_ts_final / nur_tsguess_opt / nur_path / nur_neb / nur_fwd_bwd "
                  "with dform_label_A <= 0.3 A (max over the two forming bonds of |d - d_label|, label TS + "
                  "formed_pairs_ts; computed here for single-D1-reaction records without one; D0 records carry it from "
                  "gen_dft collect --d0, against the D0 label TS; Nurion target-job records from inv_nurion scan, "
                  "against the label TS). D0 rows without D1 rxn_ids are train-only, so the "
                  "D0 kinds change only the train n_ts_near count, never an FT-1 metric (Phase I d0_ts_path / "
                  "d0_ts_final are therefore not added). nur_x_* (non-target jobs) are never TS-near: no rxn_ids, and "
                  "their dform_own_A is against their own TS, not a label TS. nur_ts_final is TS-near only within 0.3 A "
                  "(as Phase I ts_final; N_report 11-04: 51 of 6,740 are farther), although tier 2 of FT-cap takes it "
                  "always.",
    "FT-d0": "D0 (user decision 'D0', gen_d0.DECISIONS_D0): gen_D0G{1,2,3}.extxyz are read after G3 (before G4) whenever "
             "present; once $MLT/dft/plan/plan_d0.parquet exists the inputs are complete only if results/"
             "G_report_d0.json is complete for the current plan_d0 (plan_sha256) and every D0 group with ok structures "
             "has its extxyz. D0 rows have no rxn_ids (FT_k keeps them; train only) except the TS fragments of "
             "D1-shared molecules, which carry the D1 union tags and follow the D1 rules (validation / test possible); "
             "dipoles = the D0 class (FT_LODO_d drops class d). Phase I D0 rows (kind d0_*, group 'D0-<rid>') get "
             "their D0 class added to dipoles from d0_class.csv (conservative LODO; its sha256 enters the split "
             "fingerprint). C-2 check: + every row's D0 class (info d0_class, or the overlay) must be within its "
             "dipoles tag (STOP otherwise) and no kept FT_LODO_d row may carry D0 class d.",
    "FT-d1val": "Phase I rows of the d1_validation D1-design groups (group D1-NNNN-rK, 15 groups / 14 reactions, kinds "
                "d0_*; not D1 targets, but D1-dipole reactions and same-core siblings, e.g. D1-3740-r1 in core "
                "nitrone/C, fold 2): no reliable core / fold assignment exists, so folds += 0..4 and dipoles += all 5 "
                "D1 dipoles -> dropped from every FT_k and FT_LODO_d, kept in FT_all (train). Applied whether or not D0 "
                "data exist (it fixes the D1-only splits too; CODE_VERSION mlts-ft-2 rebuilds them).",
    "FT-d0hold": "D0 holdout (user decisions 'D0-plus' + 'D0-gauss', gen_d0 D0-hold / D0-sib): the reactions of "
                 "$MLT/dft/plan/d0_holdout.csv (role eval = evaluation-only, role sibling = same reactant pair, not "
                 "evaluated) never train. Every structure whose D0 reaction id (extxyz info d0_id; Phase I rows: group "
                 "'D0-<rid>'; any input incl. G4) is listed is dropped from the pool before any split (so from FT_k, "
                 "FT_all_s*, FT_LODO_*, FTpilot_*, BENCH_*, with or without AL), by d0_id and independent of the "
                 "leakage tags; exception: d0_react_ref / d0_react_disp (reactant molecules; their d0_id is only the "
                 "representative reaction). Exact duplicates among the base inputs inherit the drop; a G4 duplicate of "
                 "a base row marks it for the *_al splits only (a base split never depends on G4). C-2 (every kept "
                 "train + valid row): the D0 reaction is re-derived from the info AND, independently, from the struct "
                 "id (d0g1_<rid>_*, d0g3_<rid>_f*, d0g2_<rid>_s*); a listed TS-derived row, a D0 TS kind with neither, "
                 "or a G4 row that is neither a D1-target row (rxn_ids all D1 targets) nor d0_id-tagged with a dipoles "
                 "tag = STOP (exit 3). The file must match every recorded sha256 (g2_plan_d0.json, plan_d0_meta.json, "
                 "results/d0_holdout.csv, every data json / split meta; a missing file with a record = STOP); its sha256 "
                 "enters the split fingerprint (inputs 'D0holdout') and every data json (d0_holdout).",
    "FT-nurion": "Nurion label-level data (user decision 'Nurion', inv_nurion.py; DECISIONS 'Nurion 추출'): "
                 "$MLT/dft/nurion.extxyz.gz (collect of every processed result tar; kinds nur_* = D1-target jobs, "
                 "nur_x_* = every other job) is an input of every run. Inputs are complete only if it exists and "
                 "results/N_report.json describes it (extxyz_bytes == file size); N_report complete (all 23 result "
                 "tars) is recorded, not required (the pipeline N-gate decides when FT data may be built). Read once "
                 "per data build as a stream: the gzip is decompressed front to back, frames are cut on their atom-count "
                 "lines and parsed in chunks of 5,000 through a node-local ${TMPDIR:-/tmp} file by "
                 "mlt_common.read_extxyz (no seek in the gzip: ase seeks per frame, quadratic); the stream is hashed: "
                 "its sha256 must equal the one recorded for the fingerprint (else not ready: a collect replaced the "
                 "file), and when N_report describes the file read (same bytes) the structure count must equal its "
                 "n_structures (STOP otherwise); a Nurion structure keeps only numbers / positions / REF_forces / info "
                 "d0_id after its row is built (memory). Read after D0G3, before G4 (FT-dedupe). "
                 "Leakage tags = inv_nurion's (target rows: the reaction's rxn_ids / fold / dipole; molecules: the "
                 "union over every D1 target using them; non-target jobs: rxn_ids '', folds = their core's T1 fold "
                 "(none for D'' cores), dipoles = own dipole; D0 controls: d0_id + D0 class), so the C-2 rules are "
                 "unchanged (FT_k / FT_LODO_d drop by tag OR targets(); test = all rxn_ids held out; FT-d0hold by "
                 "d0_id). Validation (agent, recommended option): a Nurion row without rxn_ids whose job core "
                 "('<dipole>/<skeleton>' of $D1RUN/d1_manifest.csv = fold_core) is a validation core is a validation "
                 "row, so whole validation cores stay out of training; other rows without rxn_ids train.",
    "FT-cap": "size cap (user decision 'FT-cap', recommended option): one MACE-OMOL-0 epoch takes 0.7-0.8 h per 62.6k "
              "structures on an RTX 3090, so all Nurion structures (658,901 from 13 tars, ~1.1 M from 23) would make an "
              "epoch ~13 h and break the 12 GPU-h budget -> every split's train set is capped at 200,000 and its "
              "validation set at 20,000 structures (test not capped; --cap-test N optional), the same for every "
              "candidate (the cap belongs to the split: FT_k / FTpilot_<cand>_k / BENCH share fold k). Tier 1 = every "
              "non-Nurion row (I, G1-3, D0G1-3; G4 in the *_al splits, + a Nurion row that absorbed a G4 duplicate) — "
              "always kept; tier 2 = Nurion TS-near rows: kind nur_ts_final, or nur_ts_path / nur_tsguess_opt / "
              "nur_path / nur_neb / nur_fwd_bwd with dform_label_A <= 0.3 A (target jobs, against the label TS); "
              "tier 3 = every other Nurion row (incl. all nur_x_*: their dform_own_A is against their own TS). Over "
              "the cap: room = cap - tier 1; tier 2 gets round(0.7 x room), tier 3 the rest; a tier with fewer rows "
              "gives its room to the other. Within a tier: seeded round-robin over job ids (one target job = one D1 "
              "reaction): groups in a seeded random order, rows of a group in a seeded random order, rank 0 of every "
              "group first, then rank 1, ... (broad coverage; long trajectories do not dominate). Seed = "
              "mlt_common.SEED + 1000 + the split offset of parse_run (fold k 100+k, all_s i 200+i, lodo d 300+d); "
              "rng = numpy default_rng([seed, set (train 0 / valid 1 / test 2), tier]). The validation cap works on "
              "the validation rows only (validation cores). The C-2 leakage check runs on every kept row before the "
              "cap (a superset of what is written). Recorded per split meta and data json: counts before / after "
              "(by input / source / kind / tier, Nurion kinds by tier, jobs covered), room, quotas, seeds, the cap "
              "values, sha256 of nurion.extxyz.gz and of results/N_report.json.",
    "FT-relE": "within-reaction relative energy: structures of one reaction (single rxn_id) and one composition, "
               "E - mean(E) of the group (groups of >= 2), MAE over structures; primary on TS-near structures, also "
               "on all single-reaction structures. MACE uses interaction_energy (no float32 E0 quantisation).",
}


# ================================================================ helpers
def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _jobtag():
    return dict(host=socket.gethostname(), slurm_job=os.environ.get("SLURM_JOB_ID"),
                task=os.environ.get("SLURM_ARRAY_TASK_ID"), time=_now(), code=CODE_VERSION)


def _q(v, qs=(0.0, 0.1, 0.5, 0.9, 0.99, 1.0)):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if not len(v):
        return dict(n=0)
    out = {f"q{int(round(q * 100))}": float(np.quantile(v, q)) for q in qs}
    out.update(n=int(len(v)), mean=float(v.mean()))
    return out


def _sha_obj(o):
    return hashlib.sha256(json.dumps(o, sort_keys=True, default=str).encode()).hexdigest()


def _rj(p):
    try:
        return M.read_json(p)
    except Exception:                                                     # noqa: BLE001
        return None


def atom_ref_ev(numbers):
    return float(sum(ATOM_REF_EH[int(z)] for z in numbers)) * M.EH2EV


def canon_cand(c):
    if c in CANDIDATES:
        return c
    c2 = ALIASES.get(str(c).lower())
    if c2 is None:
        raise SystemExit(f"unknown candidate {c}; known: {sorted(CANDIDATES)}")
    return c2


def parse_run(run, with_al=False):
    r = str(run)
    al = r.endswith("_al")
    base = r[:-3] if al else r
    al = al or bool(with_al)
    cand, bench = None, False
    if m := re.fullmatch(r"FT_([0-4])", base):
        mode, val = "fold", int(m.group(1))
    elif m := re.fullmatch(r"FT_all_s(\d+)", base):
        mode, val = "all", int(m.group(1))
    elif m := re.fullmatch(r"FT_LODO_([0-4])", base):
        mode, val = "lodo", int(m.group(1))
    elif m := re.fullmatch(r"FTpilot_(.+)_([0-4])", base):
        mode, val, cand = "fold", int(m.group(2)), canon_cand(m.group(1))
    elif m := re.fullmatch(r"BENCH_([^_]+)(?:_(.+))?", base):
        mode, val, cand, bench = "fold", 0, canon_cand(m.group(1)), True
    else:
        raise SystemExit(f"unknown run name {run} (FT_<k>, FT_all_s<i>, FT_LODO_<d>, FTpilot_<cand>_<k>, BENCH_<cand>)")
    split = {"fold": f"fold{val}", "all": f"all_s{val}", "lodo": f"lodo{val}"}[mode] + ("_al" if al else "")
    seed_off = {"fold": 100, "all": 200, "lodo": 300}[mode] + val
    return dict(run=base + ("_al" if al else ""), base=base, mode=mode, val=val, cand=cand, split=split, al=al,
                bench=bench, valid_seed=M.SEED + seed_off, train_seed=M.SEED + (val if mode == "all" else 0),
                cap_seed=M.SEED + CAP_SEED_OFF + seed_off)


def _wtorch(obj, path):
    import torch
    path = M.assert_write_ok(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    torch.save(obj, tmp)
    os.replace(tmp, path)
    return path


def _mkdir(p):
    M.assert_write_ok(p)
    Path(p).mkdir(parents=True, exist_ok=True)
    return Path(p)


def _rss_gib():
    """peak resident memory of this process (GiB; Linux ru_maxrss is in KiB)."""
    import resource
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2, 2)


def _du_gb(p):
    tot = 0
    for root, _d, files in os.walk(p):
        for f in files:
            try:
                tot += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return tot / 1e9


def _guard_imports():
    """the mlts env is a clone of reactot: torchvision 0.17.1 (built for torch 2.2) breaks `import mace` under torch 2.8
    ('operator torchvision::nms does not exist'). Nothing here needs torchvision; when its minor version does not match
    torch (0.<minor+15>), a None entry makes `import torchvision` raise ImportError (process-local, env unchanged)."""
    os.environ.setdefault("XDG_CACHE_HOME", str(CACHE))
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    try:
        from importlib.metadata import version
        tv, tt = version("torchvision"), version("torch")
    except Exception:                                                     # noqa: BLE001
        return "torchvision absent"
    if int(tv.split(".")[1]) == int(tt.split(".")[1]) + 15:
        return "torchvision ok"
    blocked = ("torchvision", "torchaudio")
    for b in blocked:
        sys.modules.setdefault(b, None)                  # importlib.util.find_spec -> None, import -> ImportError
    try:                                                 # torchmetrics decides from the dist metadata (RequirementCache)
        from lightning_utilities.core import imports as LI
        orig = LI.RequirementCache._check_available
        if not getattr(orig, "_mlts_guard", False):
            def _check_available(self):
                if hasattr(self, "available"):
                    return
                name = str(self.requirement or self.module or "").strip()
                if re.match(r"(torchvision|torchaudio)\b", name):
                    self.available, self.message = False, f"{name} blocked by finetune._guard_imports"
                    return
                orig(self)
            _check_available._mlts_guard = True
            LI.RequirementCache._check_available = _check_available
    except ImportError:
        pass
    return f"blocked torchvision {tv} / torchaudio (torch {tt}; process-local)"


# ================================================================ data
def input_files(with_al):
    """read order (= dedupe priority): I, G1, G2, G3, D0G1, D0G2, D0G3, NUR, then G4 (decisions FT-d0, FT-nurion)."""
    return list(INPUTS) + list(D0_INPUTS) + [NUR_INPUT] + ([AL_INPUT] if with_al else [])


def nur_state(st):
    """(complete, meta) of the Nurion input (decision FT-nurion): nurion.extxyz.gz present and results/N_report.json
    describing it (extxyz_bytes == file size). N_report complete (all 23 result tars) is recorded, not required."""
    e = st.get(NUR) or {}
    J = _rj(N_REPORT) or {}
    nb = J.get("extxyz_bytes")
    match = bool(e.get("present")) and nb is not None and int(nb) == int(e.get("bytes") or -1)
    return match, dict(rule="FT-nurion", extxyz=str(NUR_INPUT[1]), extxyz_present=bool(e.get("present")),
                       extxyz_sha256=e.get("sha256"), extxyz_bytes=e.get("bytes"), N_report=str(N_REPORT),
                       N_report_present=N_REPORT.is_file(),
                       N_report_sha256=M.sha256(N_REPORT) if N_REPORT.is_file() else None,
                       N_report_complete=J.get("complete"), tars_done=J.get("tars_done"),
                       tars_expected=J.get("tars_expected"), n_tars_not_done=len(J.get("tars_not_done") or []),
                       n_structures_report=J.get("n_structures"), N_report_extxyz_bytes=nb, report_matches_extxyz=match,
                       N_report_signature=J.get("signature"))


class _ShaReader(io.RawIOBase):
    """raw file wrapper: sha256 + byte count of everything read (the gzip stream reads the file once, front to back)."""

    def __init__(self, f):
        self.f, self.h, self.n = f, hashlib.sha256(), 0

    def readable(self):
        return True

    def readinto(self, b):
        d = self.f.read(len(b))
        b[:len(d)] = d
        self.h.update(d)
        self.n += len(d)
        return len(d)


def iread_extxyz_gz(path, chunk=NUR_READ_CHUNK, info=None):
    """mlt_common.read_extxyz of a (multi-member) extxyz.gz as a stream (decision FT-nurion): the gzip is decompressed
    once front to back (never seeked: ase seeks to every frame, quadratic on gzip), frames are cut on their atom-count
    lines, and each chunk of `chunk` frames is written to a node-local ${TMPDIR:-/tmp} file, parsed by
    mlt_common.read_extxyz (identical tag handling) and deleted. Yields ase.Atoms; at the end info gets sha256 + bytes
    of the file actually read (a concurrent collect replaces the file by rename: the open handle keeps the old one)."""
    import gzip
    import tempfile
    tmpdir = os.environ.get("TMPDIR") or "/tmp"
    info = {} if info is None else info
    with open(path, "rb") as fr:
        info["bytes_at_open"] = os.fstat(fr.fileno()).st_size
        hr = _ShaReader(fr)
        with gzip.GzipFile(fileobj=hr, mode="rb") as gz, io.TextIOWrapper(gz, encoding="utf-8") as f:
            eof = False
            while not eof:
                buf, n = [], 0
                while n < chunk:
                    line = f.readline()
                    if not line:
                        eof = True
                        break
                    if not line.strip():
                        continue
                    k = int(line.split()[0])
                    buf.append(line)
                    for _ in range(k + 1):
                        buf.append(f.readline())
                    n += 1
                if not n:
                    break
                fd, tmp = tempfile.mkstemp(suffix=".extxyz", prefix="ftnur_", dir=tmpdir)
                try:
                    with os.fdopen(fd, "w") as g:
                        g.writelines(buf)
                    del buf
                    al = M.read_extxyz(tmp)
                finally:
                    os.unlink(tmp)
                if len(al) != n:
                    M.stop(f"{path}: a chunk of {n} frames parsed to {len(al)} structures")
                yield from al
        while hr.read(1 << 22):                          # the rest (if any) -> the hash covers the whole file
            pass
        info.update(sha256=hr.h.hexdigest(), bytes_read=hr.n)


def d0_state(st):
    """(required, complete, meta) of the D0 inputs (decision FT-d0)."""
    if not PLAN_D0.is_file():
        return False, True, dict(plan_d0=None)
    g = _rj(G_REPORT_D0) or {}
    psha = M.sha256(PLAN_D0)
    grp = g.get("groups") or {}
    need = [n for n, _ in D0_INPUTS if (grp.get(n) or {}).get("n_ok", (grp.get(n) or {}).get("n_structures", 0))]
    ok = bool(g.get("complete")) and g.get("plan_sha256") == psha and all(st[n]["present"] for n in need)
    return True, ok, dict(plan_d0=str(PLAN_D0), plan_d0_sha256=psha, G_report_d0=str(G_REPORT_D0),
                          G_report_d0_complete=g.get("complete"), G_report_d0_plan_sha256=g.get("plan_sha256"),
                          d0_groups_with_data=need)


def d0_holdout_records():
    """decision FT-d0hold: {record: sha256} of every place that recorded the holdout file (non-empty only)."""
    rec = {"g2_plan_d0.json": ((_rj(G2_PLAN_D0) or {}).get("holdout") or {}).get("sha256"),
           "plan_d0_meta.json": (_rj(PLAN_D0_META) or {}).get("holdout_sha256"),
           "results/d0_holdout.csv": M.sha256(D0_HOLDOUT_RES) if D0_HOLDOUT_RES.is_file() else None}
    for p in sorted(RES_FT.glob("*_data.json")) + sorted(FT_DATA.glob("*/meta.json")):
        j = _rj(p) or {}
        rec[str(p)] = (j.get("d0_holdout_sha256") or (j.get("d0_holdout") or {}).get("sha256")
                       or ((j.get("pool") or {}).get("d0_holdout") or {}).get("sha256"))
    return {k: v for k, v in rec.items() if v}


def d0_holdout():
    """decision FT-d0hold: (set of listed d0_ids (eval + sibling), meta). STOP when any record (plan, results copy,
    data json) has a holdout sha256 and the file is missing or has another sha256; no file and no record = no holdout."""
    rec = d0_holdout_records()
    base = dict(rule=DECISIONS["FT-d0hold"], file=str(D0_HOLDOUT), records={k: v[:16] for k, v in rec.items()})
    if not D0_HOLDOUT.is_file():
        if rec:
            M.stop(f"{D0_HOLDOUT} is missing but a D0 holdout is recorded in {sorted(rec)[:4]} (FT-d0hold)")
        return set(), dict(base, present=False, sha256=None, n_reactions=0)
    sha = M.sha256(D0_HOLDOUT)
    bad = {k: v[:12] for k, v in rec.items() if v != sha}
    if bad:
        M.stop(f"{D0_HOLDOUT} sha256 {sha[:12]} differs from {len(bad)} record(s) {dict(list(bad.items())[:4])} "
               f"(FT-d0hold)")
    import pandas as pd
    H = pd.read_csv(D0_HOLDOUT, keep_default_na=False)
    if "d0_id" not in H.columns:
        M.stop(f"{D0_HOLDOUT}: no d0_id column (FT-d0hold)")
    ids = {int(x) for x in H.d0_id}
    if len(ids) != len(H):
        M.stop(f"{D0_HOLDOUT}: duplicate d0_id (FT-d0hold)")
    by = dict(Counter(H.d0_class.astype(str))) if "d0_class" in H.columns else {}
    roles = dict(Counter(H.role.astype(str))) if "role" in H.columns else {}
    return ids, dict(base, present=True, sha256=sha, n_reactions=len(ids), by_class=by, by_role=roles)


def d0id_from_sid(sid):
    """D0 reaction of a gen_d0 TS-derived struct id (d0g1_<rid>_*, d0g3_<rid>_f*, d0g2_<rid>_s*), else -1; independent of
    the structure info (FT-d0hold C-2)."""
    m = SID_D0.fullmatch(str(sid))
    return int(next(g for g in m.groups() if g is not None)) if m else -1


def d0id_of(a, inp):
    """D0 reaction id of a structure (decision FT-d0hold): info d0_id (D0G1-3, any other input carrying it), else the
    Phase I group 'D0-<rid>'; -1 = none."""
    v = a.info.get("d0_id")
    if v is not None and str(v).strip() not in ("", "None"):
        try:
            x = int(v)
            if x >= 0:
                return x
        except (TypeError, ValueError):
            pass
    m = re.fullmatch(r"D0-(\d+)", str(a.info.get("group", ""))) if inp == "I" else None
    return int(m.group(1)) if m else -1


def inputs_state(with_al):
    st = {}
    for name, p in input_files(with_al):
        if p.is_file() and p.stat().st_size > 0:
            st[name] = dict(path=str(p), present=True, sha256=M.sha256(p), bytes=p.stat().st_size)
        else:
            st[name] = dict(path=str(p), present=False)
    if D0_CLASS.is_file():                               # Phase I D0 class overlay: part of the split fingerprint
        st["D0class"] = dict(path=str(D0_CLASS), present=True, sha256=M.sha256(D0_CLASS), bytes=D0_CLASS.stat().st_size)
    if D0_HOLDOUT.is_file():                             # D0-plus holdout list: part of the split fingerprint
        st["D0holdout"] = dict(path=str(D0_HOLDOUT), present=True, sha256=M.sha256(D0_HOLDOUT),
                               bytes=D0_HOLDOUT.stat().st_size)
    if st[NUR]["present"] and D1_MANIFEST.is_file():     # job cores of non-target Nurion rows (validation, FT-nurion)
        st["D1manifest"] = dict(path=str(D1_MANIFEST), present=True, sha256=M.sha256(D1_MANIFEST),
                                bytes=D1_MANIFEST.stat().st_size)
    g = _rj(G_REPORT) or {}
    base = [n for n, _ in INPUTS] + [NUR] + (["G4"] if with_al else [])
    d0_req, d0_ok, d0_meta = d0_state(st)
    nur_ok, nur_meta = nur_state(st)
    complete = all(st[n]["present"] for n in base) and bool(g.get("complete")) and d0_ok and nur_ok
    return st, complete, dict(G_report=str(G_REPORT), G_report_complete=g.get("complete"), d0_required=d0_req,
                              d0_complete=d0_ok, nurion_complete=nur_ok, nurion=nur_meta, **d0_meta)


def split_fingerprint(split, st, params):
    used = {n: v.get("sha256") for n, v in st.items() if v["present"] and (n != "G4" or split.endswith("_al"))}
    return _sha_obj(dict(split=split, inputs=used, params=params))


class Pool:
    """every label-level structure of the inputs after dedupe, level / fmax / electron-count filters (not yet split)."""

    def __init__(self, with_al, fmax_cap):
        import pandas as pd
        T = M.targets()
        self.T = T
        Ti = T.set_index("rxn_id")
        self.rfold = Ti.fold_T1.astype(int).to_dict()
        self.rdip = Ti.dipole.to_dict()
        self.rcore = Ti.fold_core.to_dict()
        self.rpairs = Ti.formed_pairs_ts.to_dict()
        self.hold_ids, self.hold_meta = d0_holdout()     # decision FT-d0hold
        self.atoms, rows = [], []
        self.read_info = {}
        self.jcore, self.jcore_meta = {}, dict(loaded=False)
        self.stats = {}
        for name, p in input_files(with_al):
            if not (p.is_file() and p.stat().st_size > 0):
                continue
            t0 = time.time()
            nur = name == NUR
            rinfo = {}
            if nur:
                self._load_jcore()
            n0 = len(rows)
            for a in (iread_extxyz_gz(p, info=rinfo) if nur else M.read_extxyz(p)):
                rows.append(self._row(a, name))
                rows[-1]["i"] = len(self.atoms)
                if nur:                                  # FT-nurion: nothing else of the info is used after _row
                    a.info = {k: a.info[k] for k in NUR_KEEP_INFO if k in a.info}
                self.atoms.append(a)
            n = len(rows) - n0
            self.read_info[name] = dict(n=n, read_s=round(time.time() - t0, 1), peak_rss_gib=_rss_gib())
            print(f"[data] read {name} {n} structures ({time.time() - t0:.0f} s, peak RSS {_rss_gib()} GiB)",
                  flush=True)
            if nur:
                self._nur_count_check(p, n, rinfo)
        self.n_read = len(rows)
        self._d0_overlay(rows)
        self._d1design_overlay(rows)
        self._fill_dform(rows)
        self._mark_holdout(rows)
        rows = self._dedupe(rows)
        rows = self._drop_holdout(rows)
        P = pd.DataFrame(rows)
        del rows
        P = self._filters(P, fmax_cap)
        self._tag_check(P)
        P = P.reset_index(drop=True)
        P["tier"] = tier_of(P)                           # decision FT-cap (G4 duplicates: build_split, *_al only)
        self.P = P
        self._nur_stats(P)

    def _load_jcore(self):
        """decision FT-nurion: job id -> fold_core '<dipole>/<skeleton>' of the D1 run manifest (the core of a
        non-target Nurion job, whose rows have no rxn_ids: validation iff it is a validation core)."""
        if not D1_MANIFEST.is_file():
            self.jcore_meta = dict(loaded=False, reason=f"{D1_MANIFEST} missing: non-target Nurion rows -> train")
            return
        import pandas as pd
        D = pd.read_csv(D1_MANIFEST, usecols=["job_id", "dipole", "skeleton"], dtype=str).fillna("")
        self.jcore = dict(zip(D.job_id, D.dipole + "/" + D.skeleton))
        self.jcore_meta = dict(loaded=True, path=str(D1_MANIFEST), n_jobs=len(self.jcore))

    def _nur_count_check(self, p, n, rinfo):
        """decision FT-nurion: the streamed structure count must equal N_report n_structures when the report describes
        the file that was read (same bytes); STOP otherwise (a reader or collect problem)."""
        J = _rj(N_REPORT) or {}
        nb = J.get("extxyz_bytes")
        same = nb is not None and int(nb) == rinfo.get("bytes_read")
        self.stats["nurion_read"] = dict(path=str(p), n_read=n, n_report=J.get("n_structures"),
                                         report_describes_file=same, sha256_stream=rinfo.get("sha256"),
                                         bytes_read=rinfo.get("bytes_read"), bytes_at_open=rinfo.get("bytes_at_open"),
                                         chunk_frames=NUR_READ_CHUNK, tmpdir=os.environ.get("TMPDIR") or "/tmp")
        if same and J.get("n_structures") is not None and int(J["n_structures"]) != n:
            M.stop(f"{p}: {n} structures read, N_report n_structures {J['n_structures']} (FT-nurion)")

    def _nur_stats(self, P):
        N = P[P.input == NUR]
        if not len(N) and NUR not in self.read_info:
            self.stats["nurion"] = dict(present=False)
            return
        norx = N[N.rxn.map(len) == 0]
        cores = set(self.rcore.values())
        self.stats["nurion"] = dict(
            present=True, rule=DECISIONS["FT-nurion"], read=self.read_info.get(NUR), n_kept=int(len(N)),
            by_kind=dict(Counter(N.kind)), by_tier={str(k): int(v) for k, v in Counter(N.tier).items()},
            n_ts_near=int(ts_near_mask(N).sum()), n_jobs=int(N.grp.nunique()),
            n_rxns=len({x for r in N.rxn for x in r}), n_rows_without_rxn_ids=int(len(norx)),
            n_rows_without_rxn_ids_in_d1_core=int(norx.jcore.isin(list(cores)).sum()),
            n_rows_without_rxn_ids_by_kind=dict(Counter(norx.kind)), job_cores=self.jcore_meta,
            n_rows_absorbing_g4_duplicate=int(N.g4dup.sum()))

    @staticmethod
    def _tags(s):
        return [x for x in str(s or "").split(",") if x]

    @staticmethod
    def _cls(v):
        """info d0_class -> class names (a comma-joined string, or a list if the extxyz reader split it)."""
        if v is None:
            return frozenset()
        if isinstance(v, (list, tuple, np.ndarray)):
            v = ",".join(str(x) for x in v)
        return frozenset(x for x in str(v).split(",") if x)

    def _row(self, a, inp):
        f = np.asarray(a.arrays["REF_forces"], float)
        fn = np.linalg.norm(f, axis=1)
        rx = tuple(sorted({int(x) for x in self._tags(a.info.get("rxn_ids"))}))
        e_eh = float(a.info.get("e_eh", float(a.info["REF_energy"]) / M.EH2EV))
        dform = a.info.get("dform_label_A")
        job = str(a.info.get("job", "") or "") if inp == NUR else ""
        # grp: round-robin unit of the size cap (Nurion: the job; FT-cap); jcore: core of a Nurion row without rxn_ids
        grp = job or (M.tag_list(rx) if rx else str(a.info.get("struct_id")))
        return dict(grp=grp, jcore=self.jcore.get(job, "") if (job and not rx) else "", g4dup=False,
                    struct_id=str(a.info.get("struct_id")), kind=str(a.info.get("kind")), source=str(a.info.get("source")),
                    input=inp, level=str(a.info.get("level", "label")), n_atoms=len(a),
                    formula=a.get_chemical_formula(mode="hill"), rxn=rx,
                    folds=frozenset(self._tags(a.info.get("folds"))),
                    dipoles=frozenset(self._tags(a.info.get("dipoles"))),
                    d0cls=self._cls(a.info.get("d0_class")) if inp.startswith("D0") else frozenset(),
                    fmax=float(fn.max()) if len(fn) else 0.0, e_eh=e_eh,
                    e_ref_check=float(a.info["REF_energy"]) - e_eh * M.EH2EV,
                    charge=int(a.info.get("charge", 0) or 0), nelec=int(np.asarray(a.numbers).sum()),
                    dform=float(dform) if dform is not None else float("nan"),
                    finite=bool(np.isfinite(f).all() and np.isfinite(e_eh)), d0id=d0id_of(a, inp), hold=False,
                    hold_al=False)

    def _mark_holdout(self, rows):
        """decision FT-d0hold: hold = the row's D0 reaction is listed and it is not a reactant molecule; hold_al =
        hold, or (after dedupe) a G4 duplicate of the row is held (drops the row from the *_al splits only)."""
        H = self.hold_ids
        for r in rows:
            r["hold"] = bool(H) and r["d0id"] in H and r["kind"] not in D0_HOLD_KEEP_KINDS
            r["hold_al"] = r["hold"]

    def _drop_holdout(self, rows):
        H = self.hold_ids
        drop = [r for r in rows if r["hold"]]
        kept_mol = [r for r in rows if not r["hold"] and r["d0id"] in H]
        self.stats["d0_holdout"] = dict(
            self.hold_meta, n_dropped=len(drop), by_kind=dict(Counter(r["kind"] for r in drop)),
            by_input=dict(Counter(r["input"] for r in drop)),
            n_reactions_with_dropped_rows=len({r["d0id"] for r in drop}),
            holdout_reactions_without_rows=sorted(H - {r["d0id"] for r in drop})[:100],
            n_kept_molecule_rows_with_holdout_d0_id=len(kept_mol),
            kept_molecule_rows_by_kind=dict(Counter(r["kind"] for r in kept_mol)),
            n_base_rows_held_for_al_only=sum(1 for r in rows if r["hold_al"] and not r["hold"]))
        if drop:
            print(f"[data] D0 holdout: dropped {len(drop)} structures of {len({r['d0id'] for r in drop})} holdout "
                  f"reactions ({dict(Counter(r['kind'] for r in drop))})", flush=True)
        return [r for r in rows if not r["hold"]]

    def _d0_overlay(self, rows):
        """decision FT-d0: Phase I D0 rows (input I, kind d0_*, group 'D0-<rid>') get their D0 dipole class added to the
        dipoles tag (gen_dft plan_d0 d0_class.csv); nothing else changes."""
        if not D0_CLASS.is_file():
            self.stats["d0_class_overlay"] = dict(applied=False, reason=f"{D0_CLASS} missing")
            return
        import pandas as pd
        C = pd.read_csv(D0_CLASS, usecols=["d0_id", "d0_class"])
        cmap = dict(zip(C.d0_id.astype(int), C.d0_class.astype(str)))
        n, miss, seen = Counter(), 0, 0
        for r in rows:
            if r["input"] != "I" or not r["kind"].startswith("d0_"):
                continue
            m = re.fullmatch(r"D0-(\d+)", str(self.atoms[r["i"]].info.get("group", "")))
            if not m:
                continue
            seen += 1
            c = cmap.get(int(m.group(1)))
            if c is None:
                miss += 1
                continue
            r["d0cls"] = r["d0cls"] | {c}
            if c not in r["dipoles"]:
                r["dipoles"] = r["dipoles"] | {c}
                n[c] += 1
        self.stats["d0_class_overlay"] = dict(applied=True, n_phase_i_d0_rows=seen, n_rows_changed=sum(n.values()),
                                              by_class=dict(n), n_without_class=miss, source=str(D0_CLASS))

    def _d1design_overlay(self, rows):
        """decision FT-d1val: Phase I rows of the D1-design d1_validation groups -> every T1 fold + every D1 dipole."""
        n, groups, tagged = 0, Counter(), 0
        self.d1design_ids = set()
        for r in rows:
            if r["input"] != "I":
                continue
            g = str(self.atoms[r["i"]].info.get("group", ""))
            if not D1DESIGN_GROUP.fullmatch(g):
                continue
            n += 1
            groups[g] += 1
            self.d1design_ids.add(r["struct_id"])
            tagged += bool(r["rxn"])
            r["folds"] = r["folds"] | frozenset(ALL_T1_FOLDS)
            r["dipoles"] = r["dipoles"] | frozenset(M.DIPOLES)
        self.stats["d1design_overlay"] = dict(rule=DECISIONS["FT-d1val"], n_rows=n, n_rows_with_rxn_ids=tagged,
                                              n_groups=len(groups), by_group=dict(groups))

    def _fill_dform(self, rows):
        kinds = set(TS_NEAR_ALWAYS) | set(TS_NEAR_IF_CLOSE)
        need = [r for r in rows if r["kind"] in kinds and not np.isfinite(r["dform"]) and len(r["rxn"]) == 1]
        cache, n_ok, n_bad = {}, 0, 0
        for r in need:
            rid = r["rxn"][0]
            if rid not in self.rpairs:
                n_bad += 1
                continue
            if rid not in cache:
                sym, pos = M.read_xyz(M.label_ts(rid))
                prs = M.formed_pairs(self.rpairs[rid])
                pos = np.asarray(pos, float)
                cache[rid] = (sym, [float(np.linalg.norm(pos[i] - pos[j])) for i, j in prs], prs)
            sym, dl, prs = cache[rid]
            a = self.atoms[r["i"]]
            if list(a.get_chemical_symbols()) != list(sym):
                n_bad += 1
                continue
            d = [float(np.linalg.norm(a.positions[i] - a.positions[j])) for i, j in prs]
            r["dform"] = float(max(abs(x - y) for x, y in zip(d, dl)))
            n_ok += 1
        self.stats["dform_filled"] = dict(n_computed=n_ok, n_not_label_order_or_unknown=n_bad,
                                          by_kind=dict(Counter(r["kind"] for r in need)))

    def _dedupe(self, rows):
        def key(a):
            h = hashlib.sha1((" ".join(a.get_chemical_symbols()) + "|").encode())
            h.update((np.round(np.asarray(a.positions, float), DEDUPE_DEC) + 0.0).tobytes())
            return h.hexdigest()

        first, kept, pairs, de, dfm, ndrop = {}, [], Counter(), [], [], 0
        # AL data is deduplicated after the base inputs: a base split never depends on G4
        for r in [r for r in rows if r["input"] != "G4"] + [r for r in rows if r["input"] == "G4"]:
            k = key(self.atoms[r["i"]])
            if k not in first:
                first[k] = r
                kept.append(r)
                continue
            j = first[k]
            ndrop += 1
            if r["input"] == "G4" and j["input"] == NUR:      # FT-cap: the AL geometry survives as this row -> tier 1
                j["g4dup"] = True                              # in the *_al splits
            if r["input"] != "G4" or j["input"] == "G4":     # FT-d0hold: a duplicate of a held row is dropped too
                j["hold"] = bool(j["hold"] or r["hold"])
            j["hold_al"] = bool(j["hold_al"] or r["hold"] or r["hold_al"])   # G4 -> base: *_al splits only
            pairs[f"{j['kind']}<-{r['kind']}"] += 1
            de.append(abs(j["e_eh"] - r["e_eh"]) * M.EH2KCAL)
            dfm.append(float(np.abs(self.atoms[j["i"]].arrays["REF_forces"]
                                    - self.atoms[r["i"]].arrays["REF_forces"]).max()))
            if r["input"] != "G4" or j["input"] == "G4":
                j["rxn"] = tuple(sorted(set(j["rxn"]) | set(r["rxn"])))
                j["folds"] = j["folds"] | r["folds"]
                j["dipoles"] = j["dipoles"] | r["dipoles"]
                j["d0cls"] = j["d0cls"] | r["d0cls"]
        kept.sort(key=lambda r: r["i"])
        self.stats["dedupe"] = dict(method=DECISIONS["FT-dedupe"], n_read=len(rows), n_dropped=ndrop,
                                    pairs=dict(pairs), de_kcal=_q(de), dfmax_eV_A=_q(dfm))
        return kept

    def _filters(self, P, fmax_cap):
        st = {}
        bad_level = P.level != "label"
        st["not_label_level"] = int(bad_level.sum())
        bad_fin = ~P.finite
        st["non_finite"] = int(bad_fin.sum())
        odd = ((P.nelec - P.charge) % 2) == 1
        st["odd_electron"] = int(odd.sum())
        fm = P.fmax.to_numpy()
        st["fmax_eV_A"] = _q(fm, (0.0, 0.5, 0.9, 0.99, 0.999, 1.0))
        st["fmax_above"] = {str(t): int((fm > t).sum()) for t in (1.0, 2.0, 5.0, 10.0, 20.0)}
        st["fmax_above_by_kind"] = {str(t): dict(Counter(P.kind[fm > t])) for t in (5.0, 10.0)}
        top = P.sort_values("fmax", ascending=False).head(20)
        st["fmax_top20"] = [dict(struct_id=r.struct_id, kind=r.kind, fmax_eV_A=round(r.fmax, 3)) for r in top.itertuples()]
        cap = P.fmax > fmax_cap if fmax_cap and fmax_cap > 0 else P.fmax < -1
        st["fmax_cap_eV_A"] = fmax_cap
        st["dropped_fmax_cap"] = int(cap.sum())
        st["dropped_fmax_cap_by_kind"] = dict(Counter(P.kind[cap]))
        st["fmax_eh_bohr_equiv_cap"] = (fmax_cap or 0) / M.EHBOHR2EVA
        e_eh, ii = P.e_eh.to_numpy(float), P.i.to_numpy()          # (no per-row .iat: ~1e6 rows with Nurion)
        e_fit = np.array([e_eh[n] * M.EH2EV - atom_ref_ev(self.atoms[int(ii[n])].numbers) for n in range(len(P))])
        P["e_fit"] = e_fit
        st["e_fit_eV"] = _q(e_fit, (0.0, 0.5, 1.0))
        st["e_ref_check_max_abs_eV"] = float(np.abs(P.e_ref_check).max()) if len(P) else 0.0
        if st["e_ref_check_max_abs_eV"] > 1e-4:
            M.stop(f"REF_energy != e_eh x EH2EV (max {st['e_ref_check_max_abs_eV']:.2e} eV)")
        if len(P) and np.abs(e_fit).max() > 5000:
            M.stop(f"|E_fit| up to {np.abs(e_fit).max():.0f} eV: ATOM_REF_EH constants wrong")
        drop = bad_level | bad_fin | odd | cap
        st["kept"] = int((~drop).sum())
        self.stats["filters"] = st
        return P[~drop]

    def _tag_check(self, P):
        nf, nd, nx, unknown, d0_cls, d0_none, nur_norx = 0, 0, 0, 0, 0, 0, 0
        nf_by = Counter()                                        # fold mismatches by input:kind (review aid)
        for r in P.itertuples():
            if not r.rxn:
                if r.input == NUR:
                    nur_norx += 1                                # Nurion non-target job / unused molecule: core fold
                elif r.folds:                                    # + own dipole without rxn_ids (inv_nurion rule)
                    nf += 1
                    nf_by[f"{r.input}:{r.kind}"] += 1
                elif r.dipoles and str(r.kind).startswith("d0_"):
                    d0_cls += 1                                  # D0: class only (decision FT-d0), not a mismatch
                elif r.dipoles:
                    nf += 1
                    nf_by[f"{r.input}:{r.kind}"] += 1
                elif str(r.kind).startswith("d0_"):
                    d0_none += 1
                continue
            if any(x not in self.rfold for x in r.rxn):
                unknown += 1
                continue
            if {str(self.rfold[x]) for x in r.rxn} != set(r.folds):
                nf += 1
                nf_by[f"{r.input}:{r.kind}"] += 1
            want = {self.rdip[x] for x in r.rxn}
            if not want <= set(r.dipoles):
                nd += 1
            elif want != set(r.dipoles):
                nx += 1                                          # superset: + a D0 class (shared fragment, overlay)
        bad_cls = [r.struct_id for r in P.itertuples() if not r.d0cls <= r.dipoles]
        n_cls = int(sum(bool(x) for x in P.d0cls))
        self.stats["tag_check"] = dict(fold_tag_mismatch=nf, fold_tag_mismatch_by=dict(nf_by.most_common(12)),
                                       dipole_tag_mismatch=nd, dipole_tag_superset=nx,
                                       unknown_rxn_ids=unknown, d0_class_only=d0_cls, d0_untagged=d0_none,
                                       n_rows_with_d0_class=n_cls, d0_class_not_in_dipoles=len(bad_cls),
                                       nurion_without_rxn_ids=nur_norx,
                                       note="drop rule = tag rule OR targets() rule, so a mismatch cannot leak; "
                                            "dipole_tag_mismatch = a D1 dipole missing from the tag, superset = extra "
                                            "(D0) classes; d0_class_only = D0 rows with only a dipole class; "
                                            "d0_class_not_in_dipoles > 0 = STOP (FT-d0); nurion_without_rxn_ids = "
                                            "Nurion rows with core folds / dipoles but no rxn_ids (FT-nurion)")
        if bad_cls:
            M.stop(f"C-2: {len(bad_cls)} rows whose D0 class is not in their dipoles tag (FT_LODO leak); first "
                   f"{bad_cls[:5]}")


def build_split(pool, split_name, R, valid_frac, fp, st, complete, gmeta, params, write=True):
    """write False (data --dry): the same drop / validation / cap / C-2 leakage logic, nothing written."""
    P, T = pool.P, pool.T
    if not R["al"]:
        P = P[P.input != "G4"]
    else:
        P = P[~P.hold_al.astype(bool)]                  # FT-d0hold: base rows whose G4 duplicate is held (AL only)
        P = P.assign(tier=np.where(P.g4dup.to_numpy(bool), 1, P.tier.to_numpy()))   # FT-cap: AL geometry -> tier 1
    mode, val = R["mode"], R["val"]
    if mode == "fold":
        held = set(T.rxn_id[T.fold_T1 == val].astype(int))
        tag_hit = P.folds.map(lambda s: str(val) in s).astype(bool)
    elif mode == "lodo":
        dn = M.DIPOLES[val]
        held = set(T.rxn_id[T.dipole == dn].astype(int))
        tag_hit = P.dipoles.map(lambda s: dn in s).astype(bool)
    else:
        held = set()
        tag_hit = P.folds.map(lambda s: False).astype(bool)
    rx_hit = P.rxn.map(lambda r: any(x in held for x in r)).astype(bool)
    drop = tag_hit | rx_hit
    keep, dropped = P[~drop], P[drop]
    test = dropped[dropped.rxn.map(lambda r: len(r) > 0 and all(x in held for x in r)).astype(bool)]
    # validation: whole cores, seeded order, until >= valid_frac of the remaining reactions
    Rr = sorted({x for r in keep.rxn for x in r if x in pool.rcore})
    ncore = Counter(pool.rcore[x] for x in Rr)
    cores = sorted(ncore)
    rng = np.random.default_rng(R["valid_seed"])
    target = int(math.ceil(valid_frac * len(Rr)))
    chosen, n = set(), 0
    for i in rng.permutation(len(cores)):
        if n >= target:
            break
        chosen.add(cores[i])
        n += ncore[cores[i]]
    V = {x for x in Rr if pool.rcore[x] in chosen}
    is_val = keep.rxn.map(lambda r: len(r) > 0 and all(x in V for x in r)).astype(bool)
    # FT-nurion: a Nurion row without rxn_ids (non-target job) of a validation core is a validation row
    nt_val = ((keep.input == NUR) & (keep.rxn.map(len) == 0) & keep.jcore.isin(sorted(chosen))).astype(bool)
    is_val = is_val | nt_val
    train, valid = keep[~is_val], keep[is_val]
    # FT-cap: per-set size cap by tiers (the C-2 check below runs on every kept row, a superset of the capped sets)
    train_pre, valid_pre, test_pre = train, valid, test
    cp = params.get("cap") or {}
    frac2 = float(cp.get("tier2_frac", CAP_TIER2_FRAC))
    train, cap_tr = cap_set(train, cp.get("train", 0), R["cap_seed"], 0, frac2)
    valid, cap_va = cap_set(valid, cp.get("valid", 0), R["cap_seed"], 1, frac2)
    test, cap_te = cap_set(test, cp.get("test", 0), R["cap_seed"], 2, frac2)
    if write and (not len(valid) or not len(train)):
        M.not_ready(f"{split_name}: empty train ({len(train)}) or validation ({len(valid)}) set")
    # C-2 leakage check (independent of the drop code: targets() folds / dipoles of every used rxn_id)
    used = {x for r in keep.rxn for x in r}
    if mode == "fold":
        bad_t = {x for x in used if pool.rfold.get(x) == val}
        bad_tag = int(keep.folds.map(lambda s: str(val) in s).sum())
    elif mode == "lodo":
        bad_t = {x for x in used if pool.rdip.get(x) == M.DIPOLES[val]}
        bad_tag = int(keep.dipoles.map(lambda s: M.DIPOLES[val] in s).sum())
    else:
        bad_t, bad_tag = set(), 0
    bad_d0 = int(keep.d0cls.map(lambda s: M.DIPOLES[val] in s).sum()) if mode == "lodo" else 0
    # D0 holdout (FT-d0hold): the list is re-read; the D0 reaction of every kept structure is re-derived from its info
    # and, independently, from its struct id; G4 rows must be D1-target rows or d0_id-tagged (with a dipoles tag)
    H2, hmeta = d0_holdout()
    bad_h, mol_h, unid, bad_g4, n_sid = [], 0, [], [], 0
    for r in keep.itertuples():
        x = d0id_of(pool.atoms[int(r.i)], r.input)
        y = d0id_from_sid(r.struct_id)
        n_sid += y >= 0
        if x in H2 or y in H2:
            if r.kind in D0_HOLD_KEEP_KINDS and y < 0:
                mol_h += 1
            else:
                bad_h.append(r.struct_id)
        if r.kind in D0_HOLD_TS_KINDS and x < 0 and y < 0:
            unid.append(r.struct_id)
        if r.input == "G4":
            d1row = bool(r.rxn) and all(t in pool.rfold for t in r.rxn)
            if (x < 0 and not d1row) or (x >= 0 and not r.dipoles):
                bad_g4.append(r.struct_id)
    leak = dict(rule="FT_k: drop iff k in folds(tag); FT_LODO_d: drop iff d in dipoles(tag); OR targets() of rxn_ids; "
                     "FT_LODO_d: no kept row of D0 class d (FT-d0); every run: no kept TS-derived row of a D0 holdout "
                     "reaction (FT-d0hold)",
                n_heldout_rxns=len(held), n_used_rxns=len(used), used_cap_heldout=len(used & held),
                used_with_heldout_fold_or_dipole=len(bad_t), kept_with_heldout_tag=bad_tag,
                kept_with_heldout_d0_class=bad_d0,
                dropped_by_tag=int(tag_hit.sum()), dropped_by_targets=int(rx_hit.sum()),
                tag_vs_targets_disagree=int((tag_hit != rx_hit).sum()),
                n_kept_d1design=int(keep.struct_id.isin(pool.d1design_ids).sum()),
                d0_holdout_sha256=hmeta.get("sha256"), d0_holdout_n=hmeta.get("n_reactions"),
                d0_holdout_sha_matches_pool=hmeta.get("sha256") == pool.hold_meta.get("sha256"),
                kept_holdout_ts_rows=len(bad_h), kept_holdout_ts_first=bad_h[:5], kept_holdout_molecule_rows=mol_h,
                kept_rows_with_d0_struct_id=int(n_sid), kept_d0_ts_rows_without_d0_reaction=len(unid),
                kept_d0_ts_rows_without_d0_reaction_first=unid[:5], kept_g4_rows_unidentified=len(bad_g4),
                kept_g4_rows_unidentified_first=bad_g4[:5], n_kept_nurion=int((keep.input == NUR).sum()),
                checked_on="every kept row before the size cap (FT-cap): a superset of the written train + valid")
    leak["pass_"] = leak["used_cap_heldout"] == 0 and len(bad_t) == 0 and bad_tag == 0 and bad_d0 == 0 and \
        (mode == "all" or leak["n_kept_d1design"] == 0) and not bad_h and leak["d0_holdout_sha_matches_pool"] and \
        not unid and not bad_g4
    if not leak["pass_"]:
        M.stop(f"C-2 leakage in split {split_name}: {leak}")

    def counts(D):
        nur = D[D.input == NUR]
        return dict(n=int(len(D)), by_kind=dict(Counter(D.kind)), by_source=dict(Counter(D.source)),
                    by_input=dict(Counter(D.input)), by_tier={str(k): int(v) for k, v in sorted(Counter(D.tier).items())},
                    nurion_by_tier_kind={str(t): dict(Counter(nur.kind[nur.tier == t])) for t in (1, 2, 3)
                                         if (nur.tier == t).any()},
                    n_nurion_jobs=int(nur.grp.nunique()), n_rxns=len({x for r in D.rxn for x in r}),
                    n_atoms=_q(D.n_atoms, (0.0, 0.5, 1.0)), n_ts_near=int(ts_near_mask(D).sum()))

    sd = _mkdir(FT_DATA / split_name) if write else FT_DATA / split_name
    files = {}
    for nm, D in (("train", train), ("valid", valid), ("test", test)):
        if nm == "test" and mode == "all" or not write:
            continue
        t0 = time.time()
        M.write_extxyz(sd / f"{nm}.xyz", [export_atoms(pool.atoms[int(r.i)], r) for r in D.itertuples()])
        files[nm] = dict(path=str(sd / f"{nm}.xyz"), n=int(len(D)), bytes=(sd / f"{nm}.xyz").stat().st_size,
                         write_s=round(time.time() - t0, 1))
    meta = dict(split=split_name, mode=mode, val=val, al=R["al"], fingerprint=fp, inputs=st, inputs_complete=complete,
                g_report=gmeta, params=params, files=files, valid_seed=R["valid_seed"], valid_frac_target=valid_frac,
                valid=dict(n_cores=len(chosen), n_rxns=len(V), frac_rxns=len(V) / max(1, len(Rr)),
                           n_cores_total=len(cores), n_rxns_total=len(Rr),
                           n_nurion_rows_without_rxn_ids_valid_core=int(nt_val.sum())),
                counts=dict(train=counts(train), valid=counts(valid), test=counts(test)),
                counts_precap=dict(train=counts(train_pre), valid=counts(valid_pre), test=counts(test_pre)),
                cap=dict(rule=DECISIONS["FT-cap"], params=cp, cap_seed=R["cap_seed"], train=cap_tr, valid=cap_va,
                         test=cap_te),
                nurion=gmeta.get("nurion"), leakage_C2=leak,
                pool=dict(read=pool.read_info, n_read=pool.n_read, **pool.stats), atom_ref_eh=ATOM_REF_EH,
                energy_keys=dict(REF_energy="eV absolute label level", E_fit="eV, REF_energy - sum ATOM_REF_EH x EH2EV "
                                 "(training key)", REF_forces="eV/A"),
                d0_holdout=pool.stats.get("d0_holdout"),
                decisions={k: DECISIONS[k] for k in ("FT-split", "FT-dedupe", "FT-fcap", "FT-frame", "FT-ts-near", "FT-d0",
                                                     "FT-d1val", "FT-d0hold", "FT-nurion", "FT-cap")},
                written=_now(), **_jobtag())
    if write:
        M.write_json(sd / "meta.json", meta)
    print(f"[data{'' if write else ' dry'}] {split_name}: train {len(train)} (pre-cap {len(train_pre)}) valid "
          f"{len(valid)} (pre-cap {len(valid_pre)}) test {len(test)} | valid rxns {len(V)}/{len(Rr)} | leak pass "
          f"{leak['pass_']} | train by input {dict(Counter(train.input))} | train by tier "
          f"{counts(train)['by_tier']} | peak RSS {_rss_gib()} GiB", flush=True)
    return meta


def tier_of(D):
    """decision FT-cap: 1 = non-Nurion row, 2 = Nurion TS-near (nur_ts_final, or a nur_ TS kind with dform_label_A <=
    0.3 A), 3 = any other Nurion row."""
    nur = (D.input == NUR).to_numpy()
    t2 = nur & (ts_near_mask(D).to_numpy() | (D.kind == "nur_ts_final").to_numpy())
    return np.where(~nur, 1, np.where(t2, 2, 3)).astype(np.int8)


def rr_pick(groups, n, rng):
    """decision FT-cap: positions of n rows by a seeded round-robin over groups (seeded group order, seeded order
    within a group; rank 0 of every group, then rank 1, ...)."""
    import pandas as pd
    codes, uniq = pd.factorize(np.asarray(groups, dtype=object))
    gpos = rng.permutation(len(uniq))[codes]
    order = np.lexsort((rng.random(len(codes)), codes))          # by group, then a random key
    sc = codes[order]
    start = np.r_[0, np.flatnonzero(np.diff(sc)) + 1]
    rank = np.empty(len(codes), np.int64)
    rank[order] = np.arange(len(sc)) - np.repeat(start, np.diff(np.r_[start, len(sc)]))
    return np.lexsort((gpos, rank))[:int(n)]                     # by rank, then the group's position


def cap_set(D, cap, seed, set_idx, frac2=CAP_TIER2_FRAC):
    """decision FT-cap: (rows kept, record). Tier 1 always; over the cap, the room left by tier 1 goes round(frac2 x
    room) to tier 2 and the rest to tier 3 (a short tier gives its room to the other), each by rr_pick."""
    cap = int(cap or 0)
    tier = D.tier.to_numpy()
    grp = D.grp.to_numpy()
    nb = {t: int((tier == t).sum()) for t in (1, 2, 3)}

    def groups(mask):
        g = Counter(grp[mask])
        return dict(n_groups=len(g), max_rows_per_group=max(g.values()) if g else 0)

    rec = dict(cap=cap, n_before=int(len(D)), tiers_before={str(t): v for t, v in nb.items()},
               seed=[int(seed), int(set_idx)], rng="numpy default_rng([seed, set, tier])")
    if cap <= 0 or len(D) <= cap:
        return D, dict(rec, applied=False, n_after=int(len(D)), tiers_after=rec["tiers_before"],
                       reason="no cap" if cap <= 0 else "within the cap")
    room = max(0, cap - nb[1])
    q2 = min(nb[2], int(math.floor(frac2 * room + 0.5)))
    q3 = min(nb[3], room - q2)
    q2 = min(nb[2], room - q3)
    keep = tier == 1
    for t, q in ((2, q2), (3, q3)):
        idx = np.flatnonzero(tier == t)
        if q >= len(idx):
            keep[idx] = True
        elif q > 0:
            keep[idx[rr_pick(grp[idx], q, np.random.default_rng([int(seed), int(set_idx), t]))]] = True
    out = D[keep]
    ta = out.tier.to_numpy()
    rec.update(applied=True, room=room, quota={"2": q2, "3": q3}, tier2_frac=frac2, n_after=int(len(out)),
               tiers_after={str(t): int((ta == t).sum()) for t in (1, 2, 3)}, tier1_over_cap=nb[1] > cap,
               groups_before={str(t): groups(tier == t) for t in (2, 3)},
               groups_after={str(t): groups(keep & (tier == t)) for t in (2, 3)})
    return out, rec


def export_atoms(a, r):
    from ase import Atoms
    b = Atoms(numbers=a.numbers, positions=a.positions)
    b.arrays["REF_forces"] = np.asarray(a.arrays["REF_forces"], float)
    b.info.update(struct_id=r.struct_id, kind=r.kind, source=r.source, rxn_ids=M.tag_list(r.rxn),
                  folds=M.tag_list(r.folds), dipoles=M.tag_list(r.dipoles), e_eh=float(r.e_eh),
                  REF_energy=float(r.e_eh) * M.EH2EV, E_fit=float(r.e_fit), total_charge=int(r.charge), total_spin=1,
                  config_type=r.kind, input=r.input)
    if np.isfinite(r.dform):
        b.info["dform_label_A"] = float(r.dform)
    return b


def ts_near_mask(D):
    kind = D.kind if hasattr(D, "kind") else D["kind"]
    dform = D.dform if hasattr(D, "dform") else D["dform"]
    return kind.isin(TS_NEAR_ALWAYS) | (kind.isin(TS_NEAR_IF_CLOSE) & (dform <= TS_NEAR_DFORM_A))


def nur_stream_check(pool, st):
    """decision FT-nurion: the Nurion file the pool streamed must be the one inputs_state hashed (fingerprint, data
    json); a collect that replaced it in between = not ready (resubmit)."""
    rd = pool.stats.get("nurion_read") or {}
    want = (st.get(NUR) or {}).get("sha256")
    if want and rd.get("sha256_stream") != want:
        M.not_ready(f"{NUR_INPUT[1]} changed during the build (stream sha256 {str(rd.get('sha256_stream'))[:12]} != "
                    f"{want[:12]}; collect running?) - resubmit")


def cmd_data(runs, all_, with_al, fmax_cap, valid_frac, force, dry=False, caps=None):
    Rs = [parse_run(r, with_al) for r in runs] + ([parse_run(r, with_al) for r in ALL_RUNS] if all_ else [])
    if not Rs:
        raise SystemExit("data: give run names or --all")
    t_data = time.time()
    any_al = any(R["al"] for R in Rs)
    st, complete, gmeta = inputs_state(any_al)
    if any_al and not st["G4"]["present"]:
        M.not_ready(f"--with-al: {AL_INPUT[1]} missing")
    caps = dict(CAP_DEFAULTS, **(caps or {}))
    params = dict(code=CODE_VERSION, fmax_cap=fmax_cap, valid_frac=valid_frac, dedupe_decimals=DEDUPE_DEC,
                  atom_ref_eh=ATOM_REF_EH, ts_near=[TS_NEAR_ALWAYS, TS_NEAR_IF_CLOSE, TS_NEAR_DFORM_A],
                  cap=dict(caps, tier2_frac=CAP_TIER2_FRAC, seed_base=M.SEED + CAP_SEED_OFF,
                           tiers="1 non-Nurion | 2 Nurion nur_ts_final or TS-near nur_ TS kind | 3 other Nurion",
                           sampler="round-robin over job ids (rr_pick)"))
    splits = {}
    for R in Rs:
        splits.setdefault(R["split"], R)
    if dry:                                            # --dry: pool + split + cap logic + C-2 check, nothing written
        pool = Pool(any_al, fmax_cap)
        nur_stream_check(pool, st)
        t_pool = time.time() - t_data
        out = {}

        def cnt(c):
            return {k: dict(n=v["n"], by_input=v["by_input"], by_tier=v["by_tier"], n_rxns=v["n_rxns"],
                            nurion_by_tier_kind=v["nurion_by_tier_kind"], n_nurion_jobs=v["n_nurion_jobs"],
                            by_kind=v["by_kind"], n_ts_near=v["n_ts_near"]) for k, v in c.items()}

        for s, R in splits.items():
            m = build_split(pool, s, R, valid_frac, split_fingerprint(s, st, params), st, complete, gmeta, params,
                            write=False)
            json.dumps(m, default=M._json_default)    # the meta a real build writes must serialise
            out[s] = dict(counts=cnt(m["counts"]), counts_precap=cnt(m["counts_precap"]),
                          cap={k: v for k, v in m["cap"].items() if k != "rule"}, valid=m["valid"],
                          leakage_C2=m["leakage_C2"])
        print(json.dumps(dict(dry=True, inputs_complete=complete, g_report=gmeta,
                              build=dict(pool_s=round(t_pool, 1), wall_s=round(time.time() - t_data, 1),
                                         peak_rss_gib=_rss_gib(), host=socket.gethostname()),
                              nurion=pool.stats.get("nurion"), nurion_read=pool.stats.get("nurion_read"),
                              dedupe={k: v for k, v in (pool.stats.get("dedupe") or {}).items() if k != "method"},
                              inputs={k: dict(present=v["present"], bytes=v.get("bytes")) for k, v in st.items()},
                              d0_holdout={k: v for k, v in (pool.stats.get("d0_holdout") or {}).items() if k != "rule"},
                              read=pool.read_info, tag_check=pool.stats.get("tag_check"),
                              d0_class_overlay=pool.stats.get("d0_class_overlay"),
                              d1design_overlay={k: v for k, v in (pool.stats.get("d1design_overlay") or {}).items()
                                                if k != "rule"},
                              filters={k: pool.stats["filters"].get(k) for k in ("kept", "odd_electron", "dropped_fmax_cap",
                                                                                 "not_label_level")},
                              dform_filled=pool.stats.get("dform_filled"), splits=out), indent=1, default=str), flush=True)
        return 0
    todo = []
    for s, R in splits.items():
        fp = split_fingerprint(s, st, params)
        meta = _rj(FT_DATA / s / "meta.json")
        ok = meta and meta.get("fingerprint") == fp and all(Path(f["path"]).is_file() for f in meta["files"].values())
        if force or not ok:
            todo.append((s, R, fp))
    metas = {}
    if todo:
        pool = Pool(any_al, fmax_cap)
        nur_stream_check(pool, st)
        for s, R, fp in todo:
            metas[s] = build_split(pool, s, R, valid_frac, fp, st, complete, gmeta, params)
    for R in Rs:
        meta = metas.get(R["split"]) or M.read_json(FT_DATA / R["split"] / "meta.json")
        out = dict(run=R["run"], split=R["split"], split_dir=str(FT_DATA / R["split"]), run_info=R,
                   fingerprint=meta["fingerprint"], inputs_complete=meta["inputs_complete"],
                   counts=meta["counts"], counts_precap=meta.get("counts_precap"),
                   cap={k: v for k, v in (meta.get("cap") or {}).items() if k != "rule"},
                   nurion=meta.get("nurion"), nurion_pool=meta["pool"].get("nurion"),
                   valid=meta["valid"], leakage_C2=meta["leakage_C2"],
                   inputs={k: dict(present=v["present"], sha256=v.get("sha256")) for k, v in meta["inputs"].items()},
                   filters=meta["pool"].get("filters"), dedupe=meta["pool"].get("dedupe"),
                   tag_check=meta["pool"].get("tag_check"),
                   d0_holdout={k: v for k, v in (meta["pool"].get("d0_holdout") or {}).items() if k != "rule"},
                   d0_holdout_sha256=(meta["pool"].get("d0_holdout") or {}).get("sha256"),
                   files=meta.get("files"), build=dict(wall_s=round(time.time() - t_data, 1), peak_rss_gib=_rss_gib(),
                                                       split_built_here=R["split"] in metas),
                   written=_now(), **_jobtag())
        M.write_json(RES_FT / f"{R['run']}_data.json", out)
    print(json.dumps(dict(inputs_complete=complete, wall_s=round(time.time() - t_data, 1), peak_rss_gib=_rss_gib(),
                          runs={R["run"]: dict(split=R["split"], built=R["split"] in metas) for R in Rs})), flush=True)
    return 0


# ================================================================ foundations
def fetch(cand):
    """local path of the foundation file: ~/.cache/mace (read only) or $MLT/cache/<kind>/ (downloaded here)."""
    C = CANDIDATES[cand]
    if C["kind"] == "mace":
        alt = [HOME_MACE_CACHE / C["file"], HOME_MACE_CACHE / "".join(c for c in C["file"] if c.isalnum() or c == "_")]
        dst = CACHE / "mace" / C["file"]
    else:
        alt = [SITE / "aimnet" / "calculators" / "assets" / C["file"]]
        dst = CACHE / "aimnet" / C["file"]
    for p in alt + [dst]:
        if p.is_file() and p.stat().st_size > 1000:
            return p
    _mkdir(dst.parent)
    tmp = dst.with_name(f".{dst.name}.part{os.getpid()}")
    t0 = time.time()
    print(f"[fetch] {C['url']} -> {dst}", flush=True)
    with urllib.request.urlopen(C["url"], timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    if tmp.stat().st_size < 1000:
        raise SystemExit(f"download of {C['url']} failed ({tmp.stat().st_size} bytes)")
    os.replace(tmp, dst)
    print(f"[fetch] {dst.stat().st_size / 1e6:.1f} MB in {time.time() - t0:.0f} s", flush=True)
    return dst


def cmd_fetch(cand):
    cand = canon_cand(cand)
    p = fetch(cand)
    info = dict(cand=cand, path=str(p), bytes=p.stat().st_size, sha256=M.sha256(p))
    if CANDIDATES[cand]["kind"] == "mace":
        _guard_imports()
        import torch
        m = torch.load(p, map_location="cpu", weights_only=False)
        info.update(model_class=type(m).__name__, heads=list(getattr(m, "heads", []) or []),
                    elements=[int(z) for z in m.atomic_numbers], r_max=float(m.r_max),
                    n_params=int(sum(x.numel() for x in m.parameters())), dtype=str(next(m.parameters()).dtype),
                    hidden_irreps=str(m.products[0].linear.irreps_out), n_interactions=int(m.num_interactions))
        miss = sorted(set(M.ELEMENTS) - {_sym(z) for z in info["elements"]})
        info["missing_d1_elements"] = miss
    print(json.dumps(info, indent=1), flush=True)
    M.write_json(RES_FT / f"foundation_{cand}.json", dict(info, written=_now(), **_jobtag()))
    return 0


def _sym(z):
    from ase.data import chemical_symbols
    return chemical_symbols[int(z)]


# ================================================================ MACE prediction (batched)
def mace_predict(model, atoms_list, device, batch_size=16, forces=True, head=None, chunk=2000):
    """per structure: energy (model dtype, eV), interaction_energy (eV), forces (eV/A) — as eval_configs.py."""
    import torch
    from mace import data as mdata
    from mace.tools import torch_geometric
    from mace.tools.utils import AtomicNumberTable
    heads = list(getattr(model, "heads", None) or ["Default"])
    head = head or heads[0]
    torch.set_default_dtype(next(model.parameters()).dtype)
    zt = AtomicNumberTable([int(z) for z in model.atomic_numbers])
    ks = mdata.KeySpecification(info_keys={"total_charge": "total_charge", "total_spin": "total_spin"}, arrays_keys={})
    rmax = float(model.r_max)
    E, I, F = [], [], []
    for p in model.parameters():
        p.requires_grad_(False)
    for s in range(0, len(atoms_list), chunk):
        cf = [mdata.config_from_atoms(a, key_specification=ks, head_name=head) for a in atoms_list[s:s + chunk]]
        ds = [mdata.AtomicData.from_config(c, z_table=zt, cutoff=rmax, heads=heads) for c in cf]
        dl = torch_geometric.dataloader.DataLoader(dataset=ds, batch_size=batch_size, shuffle=False, drop_last=False)
        for b in dl:
            b = b.to(device)
            if forces:
                o = model(b.to_dict(), compute_force=True, training=False)
                fb = o["forces"].detach().cpu().numpy()
                F.extend(np.split(fb, b.ptr[1:-1].cpu().numpy()))
            else:
                with torch.no_grad():
                    o = model(b.to_dict(), compute_force=False, training=False)
            E.append(o["energy"].detach().double().cpu().numpy())
            I.append(o["interaction_energy"].detach().double().cpu().numpy())
    return np.concatenate(E), np.concatenate(I), (F if forces else None)


def mace_head_e0(model, head=None):
    import torch
    ae = torch.atleast_2d(model.atomic_energies_fn.atomic_energies).double().cpu().numpy()
    heads = list(getattr(model, "heads", None) or ["Default"])
    hi = heads.index(head) if head in heads else 0
    return {int(z): float(ae[hi, j]) for j, z in enumerate(model.atomic_numbers)}


def lstsq_e0(atoms_list, resid, base=None):
    zs = sorted({int(z) for a in atoms_list for z in a.numbers})
    A = np.array([[int((a.numbers == z).sum()) for z in zs] for a in atoms_list], float)
    c, res, rank, _sv = np.linalg.lstsq(A, np.asarray(resid, float), rcond=None)
    fit = A @ c - resid
    out = {z: float(c[j]) + (base[z] if base else 0.0) for j, z in enumerate(zs)}
    return out, dict(n=len(atoms_list), rank=int(rank), n_elements=len(zs),
                     resid_after_fit_kcal=_q(np.abs(fit) * EV2KCAL, (0.5, 0.9, 1.0)),
                     resid_before_fit_kcal=_q(np.abs(np.asarray(resid) - np.mean(resid)) * EV2KCAL, (0.5, 0.9, 1.0)))


def e0_sample(atoms_list, nmax=40000, seed=M.SEED):
    if len(atoms_list) <= nmax:
        return list(range(len(atoms_list)))
    cnt = Counter(int(z) for a in atoms_list for z in set(a.numbers.tolist()))
    rare = {z for z, n in cnt.items() if n < 2000}
    keep = [i for i, a in enumerate(atoms_list) if rare & set(a.numbers.tolist())]
    rest = sorted(set(range(len(atoms_list))) - set(keep))
    rng = np.random.default_rng(seed)
    keep += list(rng.choice(rest, size=max(0, nmax - len(keep)), replace=False))
    return sorted(keep)


# ================================================================ budget / process control
class Budget:
    def __init__(self, rd, budget_h, stop_h):
        self.path = rd / "budget.json"
        self.d = _rj(self.path) or dict(attempts=[])
        self.prev_h = float(sum(a.get("wall_h", 0.0) for a in self.d["attempts"]))
        self.t0, self.last = time.time(), 0.0
        self.budget_h, self.stop_h = budget_h, stop_h
        self.d["attempts"].append(dict(start=_now(), wall_h=0.0, **_jobtag()))
        self.tick(True)

    def now_h(self):
        return self.prev_h + (time.time() - self.t0) / 3600.0

    def tick(self, force=False, note=None):
        if force or time.time() - self.last > 60:
            self.d["attempts"][-1]["wall_h"] = (time.time() - self.t0) / 3600.0
            if note:
                self.d["attempts"][-1]["note"] = note
            self.d["total_h"] = self.now_h()
            M.write_json(self.path, self.d)
            self.last = time.time()


class Proc:
    """subprocess with stdout teed to a log; '[FTEPOCH] {json}' lines collected."""

    def __init__(self, cmd, log, env):
        self.p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                  env=env, cwd=str(HERE))
        self.log, self.epochs, self.tail = log, [], deque(maxlen=80)
        self.th = threading.Thread(target=self._read, daemon=True)
        self.th.start()

    def _read(self):
        with open(M.assert_write_ok(self.log), "a") as f:
            for line in self.p.stdout:
                f.write(line)
                f.flush()
                self.tail.append(line.rstrip())
                if line.startswith("[FTEPOCH] "):
                    try:
                        rec = json.loads(line[10:])
                        rec["t_wall"] = time.time()
                        self.epochs.append(rec)
                    except ValueError:
                        pass

    def stop(self, grace=90):
        if self.p.poll() is None:
            self.p.terminate()
            try:
                self.p.wait(grace)
            except subprocess.TimeoutExpired:
                self.p.kill()
                self.p.wait()
        self.th.join(10)


def epoch_stats(hist):
    ep = [h for h in hist if h.get("epoch") is not None]
    if not ep:
        return dict(n=0)
    best = min(ep, key=lambda h: h["loss"])
    dts = [b["t_wall"] - a["t_wall"] for a, b in zip(hist, hist[1:]) if b.get("epoch") is not None
           and a.get("attempt") == b.get("attempt")]
    return dict(n=len(ep), last=ep[-1]["epoch"], best_epoch=best["epoch"], best=best,
                since_best=sum(1 for h in ep if h["epoch"] > best["epoch"]),
                epoch_s=_q(dts, (0.0, 0.5, 1.0)), epoch_s_median=float(np.median(dts)) if dts else None)


# ================================================================ train (MACE)
def mace_cmd_args(R, C, rd, fpath, train_file, valid_file, e0, o, max_epochs):
    e0s = "{" + ", ".join(f"{z}: {v!r}" for z, v in sorted(e0.items())) + "}"
    a = ["--name", R["run"], "--seed", str(R["train_seed"]), "--work_dir", str(rd), "--model_dir", str(rd / "mace_model"),
         "--checkpoints_dir", str(rd / "checkpoints"), "--results_dir", str(rd / "results"),
         "--log_dir", str(rd / "logs"), "--downloads_dir", str(rd / "downloads"),
         "--foundation_model", str(fpath), "--multiheads_finetuning", "False",
         "--train_file", str(train_file), "--valid_file", str(valid_file),
         "--energy_key", "E_fit", "--forces_key", "REF_forces", "--E0s", e0s,
         "--loss", "weighted", "--energy_weight", str(o.ew), "--forces_weight", str(o.fw),
         "--lr", str(o.lr), "--scaling", "rms_forces_scaling",
         "--batch_size", str(o.batch_size), "--valid_batch_size", str(2 * o.batch_size),
         "--max_num_epochs", str(int(max_epochs)), "--patience", "100000",
         "--ema", "--ema_decay", "0.99", "--amsgrad",
         "--scheduler", "ReduceLROnPlateau", "--lr_factor", "0.8", "--scheduler_patience", "3",
         "--default_dtype", "float32", "--device", "cuda", "--error_table", "PerAtomMAE", "--eval_interval", "1",
         "--restart_latest", "--save_cpu", "--clip_grad", "10", "--num_workers", str(o.num_workers),
         "--pin_memory", "True", "--plot", "False", "--enable_cueq", "False"]
    if C.get("head"):
        a += ["--foundation_head", C["head"]]
    return a


def ckpt_epochs(rd, tag):
    out = {}
    d = rd / "checkpoints"
    if d.is_dir():
        for p in d.iterdir():
            m = re.fullmatch(re.escape(tag) + r"_epoch-(\d+)\.pt", p.name)
            if m:
                out[int(m.group(1))] = p
    return out


def subset_files(R, rd, sd, n, seed):
    trn, val = rd / "sub_train.xyz", rd / "sub_valid.xyz"
    if not (trn.is_file() and val.is_file()):
        tr = M.read_extxyz(sd / "train.xyz")
        va = M.read_extxyz(sd / "valid.xyz")
        rng = np.random.default_rng(seed)
        it = sorted(rng.choice(len(tr), size=min(n, len(tr)), replace=False))
        iv = sorted(rng.choice(len(va), size=min(max(n // 5, 20), len(va)), replace=False))
        M.write_extxyz(trn, [tr[i] for i in it])
        M.write_extxyz(val, [va[i] for i in iv])
    return trn, val


def train_mace(R, cand, C, sd, meta, rd, o):
    import torch
    guard = _guard_imports()
    out_json = RES_FT / f"{R['run']}_train.json"
    model_out = model_path(R, C)
    bud = Budget(rd, o.budget_h, o.stop_h)
    if bud.prev_h >= o.stop_h:
        _write_train(out_json, dict(run=R["run"], cand=cand, status="STOP", gpu_h=bud.prev_h))
        M.stop(f"{R['run']}: {bud.prev_h:.1f} GPU-h already used (> {o.stop_h})")
    fpath = fetch(cand)
    if o.subset:
        train_file, valid_file = subset_files(R, rd, sd, o.subset, R["valid_seed"])
    else:
        train_file, valid_file = sd / "train.xyz", sd / "valid.xyz"
    dev = "cuda"
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    # ---- E0s (least squares on the foundation residual)
    e0p = rd / "E0s.json"
    e0j = _rj(e0p)
    key = dict(fp=meta["fingerprint"], cand=cand, subset=o.subset, code=CODE_VERSION)
    if not e0j or e0j.get("key") != key:
        t0 = time.time()
        found = torch.load(fpath, map_location="cpu", weights_only=False).float().to(dev)   # e3nn consts: .to()
        tr = M.read_extxyz(train_file)
        idx = e0_sample(tr)
        sub = [tr[i] for i in idx]
        e_pred, _i, _f = mace_predict(found, sub, dev, batch_size=max(8, 2 * o.batch_size), forces=False,
                                      head=C.get("head"))
        e_fit = np.array([a.info["E_fit"] for a in sub], float)
        base = mace_head_e0(found, C.get("head"))
        e0, fitst = lstsq_e0(sub, e_fit - e_pred, base=base)
        e0j = dict(key=key, E0s_eV=e0, foundation_E0s_eV={z: base[z] for z in e0}, fit=fitst, n_sample=len(sub),
                   seconds=round(time.time() - t0, 1), method=DECISIONS["FT-E0"], frame="E_fit (eV)")
        M.write_json(e0p, e0j)
        del found
        torch.cuda.empty_cache()
    e0 = {int(z): float(v) for z, v in e0j["E0s_eV"].items()}
    # ---- training phases
    tag = f"{R['run']}_run-{R['train_seed']}"
    hist_p = rd / "epochs.jsonl"
    hist = [json.loads(x) for x in hist_p.read_text().splitlines()] if hist_p.is_file() else []
    attempt = len(bud.d["attempts"])
    env = dict(os.environ, XDG_CACHE_HOME=str(CACHE), PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    mace_model = rd / "mace_model" / f"{R['run']}.model"
    reason, rc = None, None
    if not mace_model.is_file():
        args = mace_cmd_args(R, C, rd, fpath, train_file, valid_file, e0, o, o.epochs)
        cmd = [sys.executable, str(HERE / "finetune.py"), "_mace", *args]
        M.write_json(rd / "mace_cmd.json", dict(cmd=cmd, written=_now()))
        pr = Proc(cmd, rd / f"mace_attempt{attempt}.log", env)
        seen = 0
        while True:
            rc = pr.p.poll()
            bud.tick()
            new = pr.epochs[seen:]
            seen += len(new)
            for rec in new:
                rec["attempt"] = attempt
                hist.append(rec)
                with open(M.assert_write_ok(hist_p), "a") as f:
                    f.write(json.dumps(rec) + "\n")
                es = epoch_stats(hist)
                if rec.get("epoch") is not None:
                    print(f"[train] epoch {rec['epoch']} loss {rec['loss']:.5g} MAE_F {rec.get('mae_f', float('nan')) * 1e3:.2f} "
                          f"meV/A MAE_E {rec.get('mae_e_per_atom', float('nan')) * 1e3:.3f} meV/atom | best "
                          f"{es['best_epoch']} | {bud.now_h():.2f} GPU-h | mem {rec.get('max_mem_gb')} GB", flush=True)
                    ep_h = (es.get("epoch_s_median") or 0) / 3600.0
                    if es["since_best"] >= o.patience:
                        reason = f"early stop: {es['since_best']} epochs without improvement"
                    elif bud.now_h() + 1.3 * ep_h > o.budget_h:
                        reason = f"budget: next epoch would end past {o.budget_h} GPU-h"
            if rc is not None:
                break
            if reason is None and bud.now_h() > o.budget_h:
                reason = f"budget: {bud.now_h():.2f} > {o.budget_h} GPU-h mid-epoch"
            if bud.now_h() > o.stop_h:
                pr.stop()
                bud.tick(True, note="STOP > stop_h")
                _write_train(out_json, dict(run=R["run"], cand=cand, status="STOP", gpu_h=bud.now_h()))
                M.stop(f"{R['run']}: > {o.stop_h} GPU-h")
            if reason:
                print(f"[train] {reason} -> terminate + finalise", flush=True)
                pr.stop()
                rc = pr.p.returncode
                break
            time.sleep(10)
        bud.tick(True)
        if not mace_model.is_file():
            ck = ckpt_epochs(rd, tag)
            if not ck:
                _write_train(out_json, dict(run=R["run"], cand=cand, status="failed", rc=rc, reason=reason,
                                            tail=list(pr.tail)[-40:], gpu_h=bud.now_h()))
                print("\n".join(list(pr.tail)[-40:]), file=sys.stderr)
                raise SystemExit(1 if reason is None else 2)
            es = epoch_stats(hist)
            best = es.get("best_epoch")
            best = best if best in ck else max(ck)
            moved = []
            for ep, p in sorted(ck.items()):
                if ep > best:                                  # a newer non-best checkpoint (restart artefact)
                    dst = _mkdir(rd / "ckpt_superseded") / p.name
                    os.replace(p, dst)
                    moved.append(p.name)
            args = mace_cmd_args(R, C, rd, fpath, train_file, valid_file, e0, o, best)
            cmd = [sys.executable, str(HERE / "finetune.py"), "_mace", *args]
            print(f"[train] finalise from checkpoint epoch {best} (moved aside {moved})", flush=True)
            pf = Proc(cmd, rd / f"mace_attempt{attempt}_finalise.log", env)
            while pf.p.poll() is None:
                bud.tick()
                if bud.now_h() > o.stop_h:
                    pf.stop()
                    M.stop(f"{R['run']}: > {o.stop_h} GPU-h during finalise")
                time.sleep(10)
            pf.th.join(10)
            bud.tick(True)
            if not mace_model.is_file():
                _write_train(out_json, dict(run=R["run"], cand=cand, status="failed_finalise", tail=list(pf.tail)[-40:],
                                            gpu_h=bud.now_h()))
                print("\n".join(list(pf.tail)[-40:]), file=sys.stderr)
                raise SystemExit(1)
    # ---- export: float64, E0 += ATOM_REF (absolute label-level energies)
    exp = export_mace(mace_model, model_out, R, cand, meta, o)
    es = epoch_stats(hist)
    gpu_h = bud.now_h()
    res = dict(run=R["run"], cand=cand, kind="mace", status="done", model=str(model_out), model_card=str(model_out.with_suffix(".json")),
               foundation=str(fpath), split=R["split"], data_fingerprint=meta["fingerprint"],
               inputs_complete=meta["inputs_complete"], subset=o.subset,
               n_train=_count_xyz(train_file), n_valid=_count_xyz(valid_file), gpu=gpu, gpu_h=gpu_h,
               budget_h=o.budget_h, stop_h=o.stop_h, stop_reason=reason, epochs=es, E0s=e0j,
               hyper=dict(lr=o.lr, ew=o.ew, fw=o.fw, batch_size=o.batch_size, max_epochs=o.epochs, patience=o.patience,
                          ema=0.99, scheduler="ReduceLROnPlateau 0.8/3", dtype_train="float32", seed=R["train_seed"],
                          weights_round=o.weights_round),
               projection=_projection(es, train_file, valid_file, sd, o), export=exp, import_guard=guard,
               disk_gb=dict(run_dir=_du_gb(rd), model=model_out.stat().st_size / 1e9),
               decisions={k: DECISIONS[k] for k in ("FT-mace", "FT-E0", "FT-frame", "FT-budget")},
               written=_now(), **_jobtag())
    _write_train(out_json, res)
    print(json.dumps(dict(run=R["run"], status="done", gpu_h=round(gpu_h, 3), epochs=es.get("n"),
                          best=es.get("best", {}).get("loss"), model=str(model_out)), indent=1), flush=True)
    return 0


def _count_xyz(p):
    n = 0
    with open(p) as f:
        while True:
            line = f.readline()
            if not line:
                break
            k = int(line.split()[0])
            n += 1
            for _ in range(k + 1):
                f.readline()
    return n


def _projection(es, train_file, valid_file, sd, o):
    """epoch time measured on this run's data, scaled (linearly in structures) to the full split."""
    if not es.get("epoch_s_median"):
        return None
    n_tr, n_va = _count_xyz(train_file), _count_xyz(valid_file)
    full = M.read_json(sd / "meta.json")["counts"]
    f_tr, f_va = full["train"]["n"], full["valid"]["n"]
    scale = (f_tr + f_va / 3.0) / max(1.0, (n_tr + n_va / 3.0))
    ep_full = es["epoch_s_median"] * scale
    return dict(epoch_s_measured=es["epoch_s_median"], n_train_measured=n_tr, n_valid_measured=n_va,
                n_train_full=f_tr, n_valid_full=f_va, epoch_s_full=ep_full, epoch_h_full=ep_full / 3600,
                epochs_in_budget=int(o.budget_h * 3600 * 0.9 // ep_full) if ep_full > 0 else None,
                note="valid pass ~1/3 of a train step per structure; setup/E0/finalise excluded")


def export_mace(src, dst, R, cand, meta, o):
    import torch
    m = torch.load(src, map_location="cpu", weights_only=False)
    m = m.double()
    zs = [int(z) for z in m.atomic_numbers]
    add = torch.tensor([ATOM_REF_EH[z] * M.EH2EV for z in zs], dtype=torch.float64)
    with torch.no_grad():
        m.atomic_energies_fn.atomic_energies += add
    _wtorch(m, dst)
    card = dict(run=R["run"], cand=cand, kind="mace", file=str(dst), dtype="float64", heads=list(getattr(m, "heads", [])),
                elements=zs, r_max=float(m.r_max), energy="eV, absolute label level (B3LYP-D3BJ/def2-SVP SMD water)",
                forces="eV/A", charge_spin="atoms.info total_charge = molecular charge (training data: D1 0, D0 also "
                                           "-2 / +1), total_spin 1 (multiplicity, closed shell)",
                atom_ref_added_eh=ATOM_REF_EH, split=R["split"], data_fingerprint=meta["fingerprint"],
                usage="mace.calculators.MACECalculator(model_paths=<file>, device='cuda', default_dtype='float64')",
                written=_now())
    M.write_json(dst.with_suffix(".json"), card)
    return dict(file=str(dst), bytes=dst.stat().st_size, n_params=int(sum(p.numel() for p in m.parameters())))


def _write_train(path, d):
    M.write_json(path, d)


def cmd_mace_wrapper(argv):
    """child process: mace_run_train with an epoch hook ([FTEPOCH] json lines: loss, MAE, peak GPU memory)."""
    _guard_imports()
    import torch
    import mace.tools  # noqa: F401  (mace.tools.train is shadowed by the train() function in the package namespace)
    MT = sys.modules["mace.tools.train"]
    orig = MT.valid_err_log

    def hook(valid_loss, eval_metrics, logger, log_errors, epoch=None, valid_loader_name="Default"):
        orig(valid_loss, eval_metrics, logger, log_errors, epoch, valid_loader_name)
        rec = dict(epoch=epoch, head=valid_loader_name, loss=float(valid_loss))
        for k in ("mae_e_per_atom", "mae_f", "rmse_e_per_atom", "rmse_f", "mae_e"):
            v = eval_metrics.get(k)
            if v is not None:
                try:
                    rec[k] = float(v)
                except (TypeError, ValueError):
                    pass
        if torch.cuda.is_available():
            rec["max_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        print("[FTEPOCH] " + json.dumps(rec), flush=True)

    MT.valid_err_log = hook
    from mace.cli import run_train
    if os.environ.get("MLT_MACE_ERROR_TABLE", "0") != "1":     # final train+valid table = a full forward/force pass
        run_train.create_error_table = lambda **kw: "skipped by finetune.py (FT-1 computes the metrics)"
    sys.argv = ["mace_run_train", *argv]
    run_train.main()
    return 0


# ================================================================ train (AIMNet2)
def aim_cfg(C, d3="label"):
    import yaml
    cfg = yaml.safe_load((SITE / "aimnet" / "models" / C["yaml"]).read_text())
    if d3 == "label":
        cfg["kwargs"]["outputs"]["dftd3"]["kwargs"].update(B3LYP_D3BJ)
    return cfg


def aim_build(C, sd, d3="label", shift_double=False):
    from aimnet.config import build_module
    model = build_module(aim_cfg(C, d3))
    if shift_double:                                     # as the released jpt (pt2jpt.add_sae_to_shifts)
        model.outputs.atomic_shift.double()
    res = model.load_state_dict(sd, strict=False)
    return model, dict(missing=list(res.missing_keys), unexpected=list(res.unexpected_keys))


def aim_groups(atoms_list, idx=None):
    idx = list(range(len(atoms_list))) if idx is None else list(idx)
    by = {}
    for i in idx:
        by.setdefault(len(atoms_list[i]), []).append(i)
    return by


def aim_predict(model, atoms_list, device, batch=64, forces=True, idx=None):
    import torch
    n = len(atoms_list) if idx is None else len(idx)
    order = list(range(len(atoms_list))) if idx is None else list(idx)
    pos_of = {g: k for k, g in enumerate(order)}
    E = np.zeros(n)
    F = [None] * n
    for nat, ii in aim_groups(atoms_list, order).items():
        for s in range(0, len(ii), batch):
            jj = ii[s:s + batch]
            c = torch.tensor(np.stack([atoms_list[j].positions for j in jj]), dtype=torch.float32, device=device)
            z = torch.tensor(np.stack([atoms_list[j].numbers for j in jj]), dtype=torch.int64, device=device)
            q = torch.tensor([float(atoms_list[j].info.get("total_charge", 0) or 0) for j in jj], dtype=torch.float32,
                             device=device)                      # molecular charge (FT-aimnet; D0 has -2 / +1)
            if forces:
                c.requires_grad_(True)
                out = model(dict(coord=c, numbers=z, charge=q))
                e = out["energy"]
                g = torch.autograd.grad(e.sum(), c)[0]
                fb = (-g).detach().cpu().numpy()
            else:
                with torch.no_grad():
                    out = model(dict(coord=c, numbers=z, charge=q))
                e = out["energy"]
            ev = e.detach().double().cpu().numpy()
            for k, j in enumerate(jj):
                E[pos_of[j]] = ev[k]
                if forces:
                    F[pos_of[j]] = fb[k]
    return E, F


def train_aimnet(R, cand, C, sd, meta, rd, o):
    import torch
    from aimnet.data import SizeGroupedDataset, SizeGroupedSampler
    from aimnet.modules import Forces
    from aimnet.train.loss import MTLoss
    out_json = RES_FT / f"{R['run']}_train.json"
    model_out = model_path(R, C)
    bud = Budget(rd, o.budget_h, o.stop_h)
    if bud.prev_h >= o.stop_h:
        M.stop(f"{R['run']}: {bud.prev_h:.1f} GPU-h already used (> {o.stop_h})")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if dev == "cuda" else None
    jpt = fetch(cand)
    if o.subset:
        train_file, valid_file = subset_files(R, rd, sd, o.subset, R["valid_seed"])
    else:
        train_file, valid_file = sd / "train.xyz", sd / "valid.xyz"
    tr, va = M.read_extxyz(train_file), M.read_extxyz(valid_file)
    # ---- weights of the released model + parity (built module with the wB97M D3 head == jpt)
    jm = torch.jit.load(str(jpt), map_location=dev)
    sd0 = {k: v.detach().clone() for k, v in jm.state_dict().items()}
    chk = [tr[i] for i in range(min(8, len(tr)))]
    m_par, lk = aim_build(C, sd0, d3="foundation", shift_double=True)
    m_par = m_par.to(dev).eval()
    e_j, f_j = aim_predict(jm, chk, dev, forces=True)
    e_b, f_b = aim_predict(m_par, chk, dev, forces=True)
    parity = dict(max_abs_dE_eV=float(np.abs(e_j - e_b).max()),
                  max_abs_dF_eV_A=float(max(np.abs(a - b).max() for a, b in zip(f_j, f_b))), load=lk)
    real_missing = [k for k in lk["missing"] if not k.endswith(("c6ab", "r4r2", "rcov", "cnmax"))]
    if parity["max_abs_dE_eV"] > 1e-3 or parity["max_abs_dF_eV_A"] > 1e-3 or real_missing:
        _write_train(out_json, dict(run=R["run"], cand=cand, status="infeasible", parity=parity,
                                    reason="released AIMNet2 weights do not load into the aimnet yaml model"))
        raise SystemExit(f"AIMNet2 weight parity failed: {parity}")
    del m_par, jm
    # ---- training model: label D3 head, shifts zeroed, NaN (masked species) rows -> 0
    sd0["outputs.atomic_shift.shifts.weight"] = torch.zeros_like(sd0["outputs.atomic_shift.shifts.weight"])
    model, _lk = aim_build(C, sd0, d3="label")
    with torch.no_grad():
        model.afv.weight.nan_to_num_(0.0)
    model = model.to(dev)
    sae_p = rd / "sae.json"
    key = dict(fp=meta["fingerprint"], cand=cand, subset=o.subset, code=CODE_VERSION)
    saej = _rj(sae_p)
    if not saej or saej.get("key") != key:
        t0 = time.time()
        idx = e0_sample(tr)
        model.eval()
        e_pred, _ = aim_predict(model, tr, dev, forces=False, idx=idx)
        sub = [tr[i] for i in idx]
        sae, fitst = lstsq_e0(sub, np.array([a.info["E_fit"] for a in sub]) - e_pred)
        saej = dict(key=key, sae_eV=sae, fit=fitst, n_sample=len(sub), seconds=round(time.time() - t0, 1),
                    method=DECISIONS["FT-E0"], frame="E_fit (eV)")
        M.write_json(sae_p, saej)
    sae = {int(z): float(v) for z, v in saej["sae_eV"].items()}

    def dataset(al):
        d = {}
        for nat, ii in aim_groups(al).items():
            d[nat] = dict(coord=np.stack([al[i].positions for i in ii]).astype(np.float32),
                          numbers=np.stack([al[i].numbers for i in ii]).astype(np.int64),
                          charge=np.array([float(al[i].info.get("total_charge", 0) or 0) for i in ii], np.float32),
                          energy=np.array([al[i].info["E_fit"] - sum(sae[int(z)] for z in al[i].numbers) for i in ii],
                                          np.float64).astype(np.float32),
                          forces=np.stack([al[i].arrays["REF_forces"] for i in ii]).astype(np.float32))
        return SizeGroupedDataset(d)

    ds_tr, ds_va = dataset(tr), dataset(va)
    bs = o.batch_size
    ds_tr.merge_groups(min_size=min(8 * bs, max(1, len(ds_tr) // 4)))     # aimnet get_loaders: 8 x batch
    torch.manual_seed(R["train_seed"])
    np.random.seed(R["train_seed"] % (2 ** 32))
    ld_tr = ds_tr.get_loader(SizeGroupedSampler(ds_tr, batch_size=bs, shuffle=True), ["coord", "numbers", "charge"],
                             ["energy", "forces"], num_workers=0, pin_memory=True)
    ld_va = ds_va.get_loader(SizeGroupedSampler(ds_va, batch_size=2 * bs, shuffle=False),
                             ["coord", "numbers", "charge"], ["energy", "forces"], num_workers=0, pin_memory=True)
    fmodel = Forces(model)
    shifts = [p for n, p in model.named_parameters() if n.endswith("atomic_shift.shifts.weight")]
    other = [p for n, p in model.named_parameters() if not n.endswith("atomic_shift.shifts.weight")]
    opt = torch.optim.RAdam([dict(params=other), dict(params=shifts, weight_decay=0.0)], lr=o.lr, weight_decay=1e-8)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.75, patience=3)
    loss_fn = MTLoss(components=dict(
        energy=dict(fn="aimnet.train.loss.energy_loss_fn", weight=o.ew),
        forces=dict(fn="aimnet.train.loss.peratom_loss_fn", weight=o.fw,
                    kwargs=dict(key_true="forces", key_pred="forces"))))
    last_p, best_p = rd / "aim_last.pt", rd / "aim_best.pt"
    hist, start, best_loss, best_epoch = [], 0, float("inf"), None
    if last_p.is_file():
        ck = torch.load(last_p, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sch.load_state_dict(ck["sch"])
        hist, start, best_loss, best_epoch = ck["hist"], ck["epoch"] + 1, ck["best_loss"], ck["best_epoch"]
    reason = None
    ep_times = [h["epoch_s"] for h in hist]
    for epoch in range(start, o.epochs):
        if ep_times and bud.now_h() + 1.3 * float(np.median(ep_times)) / 3600 > o.budget_h:
            reason = f"budget: next epoch would end past {o.budget_h} GPU-h"
            break
        t0 = time.time()
        model.train()
        tl, nb = 0.0, 0
        for x, y in ld_tr:
            x = {k: v.to(dev, non_blocking=True) for k, v in x.items()}
            y = {k: v.to(dev, non_blocking=True) for k, v in y.items()}
            opt.zero_grad()
            pred = fmodel(x)
            loss = loss_fn(pred, y)["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 0.4)
            opt.step()
            tl += float(loss.item())
            nb += 1
            bud.tick()
            if bud.now_h() > o.stop_h:
                M.stop(f"{R['run']}: > {o.stop_h} GPU-h")
        model.eval()
        vl, vb, ae, af, na = 0.0, 0, [], [], 0
        for x, y in ld_va:
            x = {k: v.to(dev) for k, v in x.items()}
            y = {k: v.to(dev) for k, v in y.items()}
            pred = fmodel(x)
            vl += float(loss_fn(pred, y)["loss"].item())
            vb += 1
            ae.append((pred["energy"] - y["energy"]).abs().detach().cpu().numpy())
            mask = (x["numbers"] > 0).unsqueeze(-1).expand_as(y["forces"])
            af.append((pred["forces"] - y["forces"]).abs()[mask].detach().cpu().numpy())
        vl /= max(1, vb)
        sch.step(vl)
        rec = dict(epoch=epoch, loss=vl, train_loss=tl / max(1, nb), mae_e_kcal=float(np.concatenate(ae).mean() * EV2KCAL),
                   mae_f_kcal_A=float(np.concatenate(af).mean() * EV2KCAL), lr=opt.param_groups[0]["lr"],
                   epoch_s=time.time() - t0, gpu_h=bud.now_h(),
                   max_mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2) if dev == "cuda" else None)
        hist.append(rec)
        ep_times.append(rec["epoch_s"])
        if vl < best_loss:
            best_loss, best_epoch = vl, epoch
            _wtorch(model.state_dict(), best_p)
        _wtorch(dict(model=model.state_dict(), opt=opt.state_dict(), sch=sch.state_dict(), hist=hist, epoch=epoch,
                     best_loss=best_loss, best_epoch=best_epoch), last_p)
        bud.tick(True)
        print(f"[train] epoch {epoch} val loss {vl:.5g} MAE_E {rec['mae_e_kcal']:.3f} kcal/mol MAE_F "
              f"{rec['mae_f_kcal_A']:.3f} kcal/mol/A | best {best_epoch} | {bud.now_h():.2f} GPU-h | {rec['epoch_s']:.0f} s",
              flush=True)
        if epoch - best_epoch >= o.patience:
            reason = f"early stop: {epoch - best_epoch} epochs without improvement"
            break
        if opt.param_groups[0]["lr"] < 1e-6:
            reason = "lr < 1e-6"
            break
    exp = export_aimnet(C, best_p, model_out, sae, R, cand, meta, dev, tr)
    bud.tick(True)
    es = dict(n=len(hist), best_epoch=best_epoch, best=next((h for h in hist if h["epoch"] == best_epoch), None),
              epoch_s_median=float(np.median(ep_times)) if ep_times else None, history=hist)
    res = dict(run=R["run"], cand=cand, kind="aimnet", status="done", model=str(model_out),
               model_card=str(model_out.with_suffix(".json")), foundation=str(jpt), split=R["split"],
               data_fingerprint=meta["fingerprint"], inputs_complete=meta["inputs_complete"], subset=o.subset,
               n_train=len(tr), n_valid=len(va), gpu=gpu, gpu_h=bud.now_h(), budget_h=o.budget_h, stop_h=o.stop_h,
               stop_reason=reason, epochs=es, sae=saej, parity_released_vs_built=parity,
               hyper=dict(lr=o.lr, ew=o.ew, fw=o.fw, batch_size=bs, max_epochs=o.epochs, patience=o.patience,
                          optimizer="RAdam wd 1e-8 (shifts 0)", scheduler="ReduceLROnPlateau 0.75/3", clip_value=0.4,
                          d3=B3LYP_D3BJ, seed=R["train_seed"], weights_round=o.weights_round),
               projection=_projection(es, train_file, valid_file, sd, o), export=exp,
               disk_gb=dict(run_dir=_du_gb(rd), model=model_out.stat().st_size / 1e9),
               decisions={k: DECISIONS[k] for k in ("FT-aimnet", "FT-E0", "FT-frame", "FT-budget")},
               written=_now(), **_jobtag())
    _write_train(out_json, res)
    print(json.dumps(dict(run=R["run"], status="done", gpu_h=round(bud.now_h(), 3), epochs=len(hist),
                          best=best_loss, model=str(model_out)), indent=1), flush=True)
    return 0


def export_aimnet(C, best_p, dst, sae, R, cand, meta, dev, tr):
    import torch
    from aimnet.train.pt2jpt import add_cutoff, mask_not_implemented_species, set_eval
    sdb = torch.load(best_p, map_location="cpu", weights_only=False)
    m, _ = aim_build(C, sdb, d3="label")
    m = set_eval(m)
    m = add_cutoff(m, cutoff_lr=float("inf"))
    m.outputs.atomic_shift.double()
    with torch.no_grad():
        for z, v in sae.items():
            m.outputs.atomic_shift.shifts.weight[int(z)] += v + ATOM_REF_EH[int(z)] * M.EH2EV
    species = sorted(sae)
    m = mask_not_implemented_species(m, species)
    m.register_buffer("impemented_species", torch.tensor(species, dtype=torch.int64))
    js = torch.jit.script(m)
    M.assert_write_ok(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.tmp{os.getpid()}")
    js.save(str(tmp))
    os.replace(tmp, dst)
    # check: exported (absolute) = training frame + sum(sae + ATOM_REF)
    chk = tr[:4]
    mt, _ = aim_build(C, sdb, d3="label")
    mt = mt.to(dev).eval()
    e_t, _ = aim_predict(mt, chk, dev, forces=False)
    e_x, _ = aim_predict(torch.jit.load(str(dst), map_location=dev), chk, dev, forces=False)
    off = np.array([sum(sae[int(z)] for z in a.numbers) + atom_ref_ev(a.numbers) for a in chk])
    dmax = float(np.abs(e_x - (e_t + off)).max())
    card = dict(run=R["run"], cand=cand, kind="aimnet", file=str(dst), format="TorchScript (aimnet jitcompile layout)",
                elements=species, d3=B3LYP_D3BJ, energy="eV, absolute label level (B3LYP-D3BJ/def2-SVP SMD water)",
                forces="eV/A (autograd)", split=R["split"], data_fingerprint=meta["fingerprint"],
                usage="aimnet.calculators.AIMNet2Calculator(<file>) / orca-mlips aimnet2 backend with a local path",
                export_check_max_abs_eV=dmax, written=_now())
    M.write_json(dst.with_suffix(".json"), card)
    if dmax > 1e-3:
        raise SystemExit(f"AIMNet2 export check failed: {dmax} eV")
    return dict(file=str(dst), bytes=dst.stat().st_size, check_max_abs_eV=dmax)


# ================================================================ train dispatcher
def model_path(R, C):
    """deployable runs -> $MLT/models/<run>.<ext>; BENCH runs stay in their run dir (never used downstream)."""
    ext = "model" if C["kind"] == "mace" else "jpt"
    return (FT / R["run"] / f"{R['run']}.{ext}") if R["bench"] else (MODELS / f"{R['run']}.{ext}")


def cmd_train(run, cand, o):
    R = parse_run(run, o.with_al)
    cand = canon_cand(cand)
    if R["cand"] and R["cand"] != cand:
        raise SystemExit(f"run {R['run']} is for candidate {R['cand']}, not {cand}")
    C = CANDIDATES[cand]
    if not C["si"]:
        raise SystemExit(f"{cand}: {CAND_NOTES['no_si']}")
    sd = FT_DATA / R["split"]
    meta = _rj(sd / "meta.json")
    if not meta or not (sd / "train.xyz").is_file():
        M.not_ready(f"split {R['split']} not built: run `finetune.py data {R['run']}` first")
    if not meta["inputs_complete"] and not (o.allow_partial or R["bench"]):
        M.not_ready(f"split {R['split']} was built from incomplete inputs (G collect, or G collect --d0 once plan_d0 "
                    f"exists, not complete); rebuild with `data` after the collects, or pass --allow-partial")
    _guard_imports()
    if o.epochs is None:
        o.epochs = 200 if C["kind"] == "mace" else 1000
    if o.patience is None:
        o.patience = 6 if C["kind"] == "mace" else 25
    if o.batch_size is None:
        o.batch_size = 8 if C["kind"] == "mace" else 64
    if o.lr is None:
        o.lr = 1e-3 if C["kind"] == "mace" else 1e-4
    if o.ew is None:
        o.ew = 1.0
    if o.fw is None:
        o.fw = 1.0 if C["kind"] == "mace" else 0.2
    o.weights_round = 0 if (o.ew, o.fw) == ((1.0, 1.0) if C["kind"] == "mace" else (1.0, 0.2)) else 1
    rd = FT / R["run"]
    prev = _rj(RES_FT / f"{R['run']}_train.json")
    model_out = model_path(R, C)
    if prev and prev.get("status") == "done" and model_out.is_file() and not o.force:
        if prev.get("data_fingerprint") == meta["fingerprint"] and prev.get("cand") == cand:
            print(f"[train] {R['run']} done already ({model_out})", flush=True)
            return 0
    old = _rj(rd / "run_key.json")
    key = dict(cand=cand, fp=meta["fingerprint"], subset=o.subset, lr=o.lr, ew=o.ew, fw=o.fw, bs=o.batch_size)
    if rd.exists() and (old is not None) and (old != key or o.force):
        stale = rd.with_name(f"{rd.name}.stale_{time.strftime('%Y%m%d_%H%M%S')}")
        os.replace(rd, stale)
        print(f"[train] run key changed / --force: previous run dir moved to {stale} (not deleted)", flush=True)
    _mkdir(rd)
    M.write_json(rd / "run_key.json", key)
    if C["kind"] == "mace":
        return train_mace(R, cand, C, sd, meta, rd, o)
    return train_aimnet(R, cand, C, sd, meta, rd, o)


# ================================================================ FT-1
def ft1_metrics(D, Fref, Fpred):
    """D: DataFrame per structure (kind, dform, rxn, formula, n_atoms, e_ref, e_rel_pred); forces lists (eV/A)."""
    ts = ts_near_mask(D).to_numpy()
    fae = [np.abs(a - b).ravel() for a, b in zip(Fref, Fpred)]

    def fmae(mask):
        v = [fae[i] for i in np.where(mask)[0]]
        return float(np.concatenate(v).mean() * EV2KCAL) if v else None

    def rel(mask):
        S = D[mask & (D.rxn.map(len) == 1).to_numpy()].copy()
        if not len(S):
            return dict(n=0, n_groups=0, mae_kcal=None)
        S["g"] = S.rxn.map(lambda r: r[0]).astype(str) + "|" + S.formula
        S = S[S.groupby("g").g.transform("size") >= 2]
        if not len(S):
            return dict(n=0, n_groups=0, mae_kcal=None)
        dr = S.e_ref - S.groupby("g").e_ref.transform("mean")
        dp = S.e_rel_pred - S.groupby("g").e_rel_pred.transform("mean")
        err = np.abs(dr - dp) * EV2KCAL
        return dict(n=int(len(S)), n_groups=int(S.g.nunique()), mae_kcal=float(err.mean()),
                    q90_kcal=float(np.quantile(err, 0.9)), n_rxns=int(S.rxn.map(lambda r: r[0]).nunique()))

    allm = np.ones(len(D), bool)
    out = dict(n=int(len(D)), n_ts_near=int(ts.sum()),
               force_mae_kcal_A=dict(ts_near=fmae(ts), all=fmae(allm)),
               rel_e=dict(ts_near=rel(ts), all_single_rxn=rel(allm)),
               force_mae_by_kind_kcal_A={k: fmae((D.kind == k).to_numpy()) for k in sorted(D.kind.unique())},
               n_by_kind=dict(Counter(D.kind)))
    if "e_abs_pred" in D:
        out["abs_e_mae_kcal_per_atom"] = float((np.abs(D.e_abs_pred - D.e_ref) / D.n_atoms).mean() * EV2KCAL)
    fm, em = out["force_mae_kcal_A"]["ts_near"], out["rel_e"]["ts_near"]["mae_kcal"]
    out["targets"] = FT1_TARGET
    out["meets"] = dict(force=None if fm is None else fm <= FT1_TARGET["force_mae_kcal_A"],
                        rel_e=None if em is None else em <= FT1_TARGET["rel_e_mae_kcal"])
    if fm is not None and em is not None:
        rf, re_ = fm / FT1_TARGET["force_mae_kcal_A"], em / FT1_TARGET["rel_e_mae_kcal"]
        out["ratio_to_target"] = dict(force=rf, rel_e=re_)
        out["weights_hint"] = ("raise energy weight" if re_ > 1 and re_ > 1.5 * rf else
                               "raise forces weight" if rf > 1 and rf > 1.5 * re_ else "keep")
    return out


def frame(al):
    import pandas as pd
    rows = []
    for a in al:
        rx = tuple(sorted({int(x) for x in str(a.info.get("rxn_ids") or "").split(",") if x}))
        d = a.info.get("dform_label_A")
        rows.append(dict(struct_id=a.info.get("struct_id"), kind=a.info.get("kind"), rxn=rx,
                         formula=a.get_chemical_formula(mode="hill"), n_atoms=len(a),
                         e_ref=float(a.info["e_eh"]) * M.EH2EV, dform=float(d) if d is not None else float("nan")))
    return pd.DataFrame(rows)


def cmd_ft1(run, o):
    import torch
    _guard_imports()
    R = parse_run(run, o.with_al)
    sd = FT_DATA / R["split"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if o.foundation:
        cand = canon_cand(o.foundation)
        tj = None
        label = f"foundation_{cand}"
    else:
        tj = _rj(RES_FT / f"{R['run']}_train.json")
        if not tj or tj.get("status") != "done":
            M.not_ready(f"{R['run']}: no finished training (results/ft/{R['run']}_train.json)")
        cand, label = tj["cand"], "finetuned"
    C = CANDIDATES[cand]
    meta = M.read_json(sd / "meta.json")
    if tj and tj.get("data_fingerprint") != meta["fingerprint"]:
        print("[ft1] WARNING: split rebuilt after training (fingerprint differs)", flush=True)
    sets = [("valid", sd / "valid.xyz")] + ([("test", sd / "test.xyz")] if (sd / "test.xyz").is_file() else [])
    if tj and tj.get("subset"):
        sets = [("valid_subset", FT / R["run"] / "sub_valid.xyz")] + [s for s in sets if s[0] == "test"]
    if C["kind"] == "mace":
        mp = Path(tj["model"]) if tj else fetch(cand)
        model = torch.load(mp, map_location="cpu", weights_only=False)
        model = (model.float() if o.dtype == "float32" else model.double()).to(dev)
        head = None if tj else C.get("head")
    else:
        mp = Path(tj["model"]) if tj else fetch(cand)
        model = torch.jit.load(str(mp), map_location=dev)
        head = None
    out = dict(run=R["run"], cand=cand, model=str(mp), which=label, dtype=o.dtype if C["kind"] == "mace" else
               "float32 (shifts float64)", device=torch.cuda.get_device_name(0) if dev == "cuda" else "cpu", sets={},
               definitions=dict(ts_near=DECISIONS["FT-ts-near"], rel_e=DECISIONS["FT-relE"],
                                force_mae="mean |F_pred - F_ref| over Cartesian components, kcal mol-1 A-1"),
               note="FT-1 is recorded only (spec 5), the gate is FT-2 geometry")
    for nm, p in sets:
        al = M.read_extxyz(p)
        if o.max_n:
            al = al[: o.max_n]
        D = frame(al)
        t0 = time.time()
        if C["kind"] == "mace":
            e, ei, F = mace_predict(model, al, dev, batch_size=o.batch_size, forces=True, head=head)
            D["e_abs_pred"] = e
            D["e_rel_pred"] = ei
        else:
            e, F = aim_predict(model, al, dev, batch=max(16, o.batch_size * 4), forces=True)
            D["e_abs_pred"] = e
            D["e_rel_pred"] = e
        dt = time.time() - t0
        if tj is None:
            D = D.drop(columns=["e_abs_pred"])                 # foundation: different level, absolute E meaningless
        Fref = [np.asarray(a.arrays["REF_forces"], float) for a in al]
        met = ft1_metrics(D, Fref, F)
        met.update(path=str(p), predict_s=round(dt, 1), structures_per_s=round(len(al) / max(dt, 1e-9), 2))
        out["sets"][nm] = met
        D["f_mae_kcal_A"] = [float(np.abs(a - b).mean() * EV2KCAL) for a, b in zip(Fref, F)]
        D["rxn"] = D.rxn.map(M.tag_list)
        M.write_parquet(FT / R["run"] / f"ft1_{label}_{nm}.parquet", D)
        print(f"[ft1] {R['run']} {nm}: n {len(al)} | TS-near F MAE {met['force_mae_kcal_A']['ts_near']} kcal/mol/A | "
              f"rel E MAE {met['rel_e']['ts_near']['mae_kcal']} kcal/mol | {len(al) / max(dt, 1e-9):.1f} struct/s",
              flush=True)
    out.update(written=_now(), **_jobtag())
    name = f"{R['run']}_ft1.json" if tj else f"{R['run']}_ft1_{label}.json"
    M.write_json(RES_FT / name, out)
    return 0


# ================================================================ status
def cmd_status():
    runs = sorted({p.name.rsplit("_", 1)[0] for p in RES_FT.glob("*_data.json")} |
                  {p.name[: -len("_train.json")] for p in RES_FT.glob("*_train.json")})
    print(f"{'run':28s} {'split':10s} {'n_train':>8s} {'n_valid':>7s} {'complete':>8s} {'cand':16s} {'status':10s} "
          f"{'ep':>4s} {'GPU-h':>6s} {'F_TS':>6s} {'relE':>6s}")
    for r in runs:
        d = _rj(RES_FT / f"{r}_data.json") or {}
        t = _rj(RES_FT / f"{r}_train.json") or {}
        f = _rj(RES_FT / f"{r}_ft1.json") or {}
        v = (f.get("sets") or {}).get("valid") or (f.get("sets") or {}).get("valid_subset") or {}
        fts = (v.get("force_mae_kcal_A") or {}).get("ts_near")
        rel = ((v.get("rel_e") or {}).get("ts_near") or {}).get("mae_kcal")
        c = d.get("counts") or {}
        print(f"{r:28s} {d.get('split', ''):10s} {c.get('train', {}).get('n', ''):>8} {c.get('valid', {}).get('n', ''):>7} "
              f"{str(d.get('inputs_complete', '')):>8s} {t.get('cand', ''):16s} {t.get('status', ''):10s} "
              f"{(t.get('epochs') or {}).get('n', ''):>4} {('%.2f' % t['gpu_h']) if t.get('gpu_h') is not None else '':>6s} "
              f"{('%.3f' % fts) if fts is not None else '':>6s} {('%.3f' % rel) if rel is not None else '':>6s}")
    return 0


# ================================================================ main
def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "_mace":
        return cmd_mace_wrapper(argv[1:])
    ap = argparse.ArgumentParser(description="DEPLOY-V2 (MLTS) Phase FT")
    sub = ap.add_subparsers(dest="cmd")
    a = sub.add_parser("data")
    a.add_argument("runs", nargs="*")
    a.add_argument("--all", action="store_true")
    a.add_argument("--with-al", action="store_true")
    a.add_argument("--fmax-cap", type=float, default=FMAX_CAP)
    a.add_argument("--valid-frac", type=float, default=VALID_FRAC)
    a.add_argument("--cap-train", type=int, default=CAP_TRAIN, help="train structures per split (FT-cap; 0 = no cap)")
    a.add_argument("--cap-valid", type=int, default=CAP_VALID, help="validation structures per split (0 = no cap)")
    a.add_argument("--cap-test", type=int, default=CAP_TEST, help="test structures per split (default 0 = no cap)")
    a.add_argument("--force", action="store_true")
    a.add_argument("--dry", action="store_true")
    a = sub.add_parser("fetch")
    a.add_argument("cand")
    a = sub.add_parser("train")
    a.add_argument("run")
    a.add_argument("cand")
    a.add_argument("--with-al", action="store_true")
    a.add_argument("--epochs", type=int, default=None, help="max epochs (MACE 200, AIMNet2 1000)")
    a.add_argument("--budget-h", type=float, default=BUDGET_H)
    a.add_argument("--stop-h", type=float, default=STOP_H)
    a.add_argument("--batch-size", type=int, default=None,
                   help="MACE 8 (OMOL-0 bench: same structures/s as 16, 9.6 vs 15.6 GB), AIMNet2 64")
    a.add_argument("--lr", type=float, default=None, help="MACE 1e-3, AIMNet2 1e-4")
    a.add_argument("--ew", type=float, default=None, help="energy weight (1)")
    a.add_argument("--fw", type=float, default=None, help="forces weight (MACE 1, AIMNet2 0.2)")
    a.add_argument("--patience", type=int, default=None, help="epochs without a better valid loss (MACE 6, AIMNet2 25)")
    a.add_argument("--subset", type=int, default=0)
    a.add_argument("--num-workers", dest="num_workers", type=int, default=0)
    a.add_argument("--allow-partial", action="store_true")
    a.add_argument("--force", action="store_true")
    a = sub.add_parser("ft1")
    a.add_argument("run")
    a.add_argument("--with-al", action="store_true")
    a.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    a.add_argument("--foundation", default=None)
    a.add_argument("--batch-size", type=int, default=16)
    a.add_argument("--max-n", dest="max_n", type=int, default=0)
    sub.add_parser("status")
    o = ap.parse_args(argv)
    if o.cmd == "data":
        return cmd_data(o.runs, o.all, o.with_al, o.fmax_cap, o.valid_frac, o.force, o.dry,
                        caps=dict(train=o.cap_train, valid=o.cap_valid, test=o.cap_test))
    if o.cmd == "fetch":
        return cmd_fetch(o.cand)
    if o.cmd == "train":
        return cmd_train(o.run, o.cand, o)
    if o.cmd == "ft1":
        return cmd_ft1(o.run, o)
    if o.cmd == "status":
        return cmd_status()
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
