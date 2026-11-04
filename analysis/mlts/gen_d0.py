#!/usr/bin/env python3
r"""gen_d0.py — DEPLOY-V2 (MLTS) Phase G, D0 extension (user decision 'D0', 2026-11-04, results/DECISIONS.md): the
Coley 5,269 dipolar cycloadditions (repo-root labels_all.json, 5,260 ok) go into the MLIP training data with the D1
G rules; evaluation stays on the D1 T1 folds. Called through gen_dft.py (never run directly):
  gen_dft.py plan_d0 [--limit N] [--n-g2 auto|K] [--n-near 2|1] [--force]   -> cmd_plan_d0
  gen_dft.py g0_d0 [--n-rand 30] [--n-label 10] [--n-g2 auto|K] [--label-rxns a,b] [--max-atoms N]   -> cmd_g0_d0
  gen_dft.py collect --d0                                    -> cmd_collect_d0
  (run D0G1 / run D0G3 / run_g2 --d0 are gen_dft.cmd_run / cmd_run_g2 on the D0 plan)
  gen_d0.py plan_d0 [--limit N] [--n-g2 auto|K] [--n-near 2|1] [--n-holdout H] [--replace-holdout] [--unpin]
                    [--force]
        the same plan_d0 with the D0-plus options (user decisions 'D0-plus' + 'D0-gauss', 2026-11-04): gen_dft.main has
        no --n-holdout, so this file has its own CLI for plan_d0 (every other command is passed on to gen_dft.main).
        An explicit --n-g2 K here pins K (decision D0-pin); without --n-holdout the holdout list is kept unchanged (so
        `gen_dft.py plan_d0 --n-g2 K` keeps it). --force = re-plan even if up to date and skip the in-flight guard
        (as in gen_dft); --replace-holdout = the only way to change an existing holdout list (decision D0-hold-fix).
Decisions: DECISIONS_D0 (also in results/G_plan_d0.json, G0_estimate_d0.json, G_report_d0.json).

D0-plus / D0-gauss (decisions D0-plus, D0-gauss, D0-hold, D0-sib, D0-hold-fix, D0-hold-ref, D0-cap, D0-pin): D0G2 = K
training reactions (prefix of the D0-G2 ranking; label-level OptTS from the G2c xTB TS) only; H evaluation-only
("holdout") reactions are NOT refined (D0-gauss: their reference TS is the Coley G16 TS). The holdout list
$MLT/dft/plan/d0_holdout.csv (d0_id, d0_class, n_atoms, reason, role; copy results/d0_holdout.csv) = the H evaluation
reactions (role eval) + every other planned D0 reaction with the same reactant pair (role sibling, decision D0-sib); it
is fixed once written (sha256 in plan_d0_meta.json, g2_plan_d0.json, d0_holdout_ref.json, the results copy and every FT
data json; any mismatch or a missing file = exit 3). Every TS-derived structure of a listed reaction (d0_ts_near / _imag /
_label, d0_frag, d0_g2_path, d0_ts_final_g2, any other d0_id-tagged kind except the reactant molecules d0_react_ref /
d0_react_disp) is dropped from every FT run by finetune.py; collect --d0 writes holdout=True into their extxyz info;
$MLT/dft/plan/d0_holdout_ref.json lists the evaluation reactions with their reference TS (Coley) and starts.

Groups (struct_id prefix, kind)
  D0G1  per D0 reaction: d0g1_<rid>_n1/_n2 d0_ts_near, d0g1_<rid>_i d0_ts_imag (displaced in the run step from an xTB
        Freq at the D0 label TS); d0g1_<rid>_lab d0_ts_label = undisplaced label TS (g0_d0 parity rows only)
  D0G3  per D0 reaction: d0g3_<rid>_f1/_f2 d0_frag (TS cut by the eda.inp (1)/(2) tags); per D0-only molecule:
        d0g3_<mid>_ref d0_react_ref + d0g3_<mid>_d1/_d2 d0_react_disp
  D0G2  N training reactions: label-level OptTS from the G2c D0 xTB TS (G3L input) -> every step d0_g2_path, the last
        of a converged run d0_ts_final_g2 (struct ids d0g2_<rid>_s###); g2_plan_d0.json 'rxns' = the training units
        only (the holdout is in its 'holdout' record, never run by run_g2 --d0)
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import socket  # noqa: E402
import time  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mlt_common as M  # noqa: E402
import gen_dft as G  # noqa: E402

CODE_D0 = "mlts-gen-d0-2"          # -2: resonance-invariant identities, budget basis, n_near / pilot / parity (review)
# CODE_D0 is the PLAN-ROW code inside the plan fingerprint: bump it only when D0G1 / D0G3 rows change (a fingerprint
# change after D0 records exist is a STOP, decision D0-nnear / D0-scope). Report-level revisions use CODE_D0_REV.
CODE_D0_REV = "mlts-gen-d0-3"      # -3: D0-plus / D0-gauss (explicit pinned D0G2 count, holdout list with reactant-pair
#                                    siblings, not refined, Coley reference; content-fixed; cap check; holdout marks)

# ---------------------------------------------------------------- inputs (read only)
D0_LABELS = Path(os.environ.get("MLT_D0_LABELS", str(M.REPO / "labels_all.json")))
D0_RAW = Path(os.environ.get("MLT_D0_RAW", "/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/dipolar_cycloaddition"))
D0_PROF = D0_RAW / "extracted" / "full_dataset_profiles"            # <rid>/{TS_*.xyz, r*.xyz, ...} (Coley, G16)
D0_SMILES = D0_RAW / "full_dataset.csv"                              # rxn_id, rxn_smiles (atom-mapped)
D0_EDA = Path(os.environ.get("MLT_D0_EDA", "/gpfs/tmp_cpu2/yeseo1ee/eda_smd_relabel/inputs"))   # rxn_%04d/eda.inp
D0_Q = Path(os.environ.get("MLT_D0_Q", "/gpfs/tmp_cpu2/yeseo1ee/geo1b/d0/labels_d0_q.json"))    # D0 Q keys (GEO1b)
D0_Q_AUDIT = D0_Q.parent / "Q_audit_d0.csv"
D0_FOLDS = D0_Q.parent / "d0_folds.csv"                              # overlap True = same reaction as a D1 job
D0_G2_FINAL = Path(os.environ.get("MLT_D0_G2", "/gpfs/tmp_cpu2/yeseo1ee/g2c/d0/g2_final"))     # G2c D0 xTB TSs
D1_MANIFEST = M.D1RUN / "d1_manifest.csv"
I_SOURCES = HERE / "results" / "I_sources.csv"                       # Phase I (d0_ts_grad parity rows), read only
D1_G0 = G.G0_JSON if G.G0_JSON.is_file() else HERE / "results" / "G0_estimate.json"
D1_RUNS = Path(os.environ.get("MLT_D1_RUNS", "/gpfs/tmp_cpu2/yeseo1ee/mlts/dft/runs"))          # observed D1 rates

# ---------------------------------------------------------------- constants
D0_SEED_OFF = 2_000_000           # D0 reaction seeds SEED + 2,000,000 + rid (D1 seeds are SEED + 1,000,001.., +2, +3)
BUDGET_CORE_H, MARGIN = G.BUDGET_CORE_H, 0.10
G0_N_RAND, G0_N_LABEL, G0_N_G2 = 30, 10, "auto"   # auto D0G2 pilots: 1 if the plan's rule selects > 0 reactions, else 0
PARITY_D1_X = 10.0                # D0 parity median > 10 x the measured D1 label-TS median = level flag (never a stop)
FIT_MIN_D0 = 10                   # own log-log fit only with >= 10 ok D0 EnGrads
XTB_PRIOR = dict(a=2.5e-6, b=2.0)  # core-h = a N^b per xTB Freq (geo2b T-D0 smoke: 1.8e-3 - 1.5e-2 core-h, N 30-60)
TASK_H = 12.0                     # suggested array size: ceil(core-h / (64 cores x 12 h))
KIND_SRC = dict(d0_ts_near="d0_label_ts+xtb_nm", d0_ts_imag="d0_label_ts+xtb_imag", d0_ts_label="d0_label_ts",
                d0_frag="d0_label_ts_frag", d0_react_ref="d0_ref", d0_react_disp="d0_ref+xtb_nm",
                d0_g2_path="d0_g2cts_label_optts", d0_ts_final_g2="d0_g2cts_label_optts")
D1_CLASSES = tuple(M.DIPOLES)
# D0-plus / D0-gauss (user decisions 2026-11-04, results/DECISIONS.md 'D0-plus', 'D0-gauss')
HOLD_CSV = G.PLAN_DIR / "d0_holdout.csv"          # d0_id, d0_class, n_atoms, reason, role (fixed once written)
HOLD_CSV_RES = M.RES / "d0_holdout.csv"            # copy in results/
HOLD_REF = G.PLAN_DIR / "d0_holdout_ref.json"     # evaluation reactions: reference TS (Coley) + starts
HOLD_COLS = ("d0_id", "d0_class", "n_atoms", "reason", "role")
HOLD_ROLES = ("eval", "sibling")  # eval = evaluation-only reaction; sibling = same reactant pair (excluded, not evaluated)
HOLD_MAX_FRAC = 0.5               # a class gives at most floor(0.5 x its G2c pool size) evaluation reactions
D1DESIGN_TS = re.compile(r"D1-\d+-r\d+")   # Phase I d1_validation D1-design reactions (finetune FT-d1val)
GAUSS_FLOOR_NOTE = ("Coley G16 TS vs the ORCA label-level stationary point: ~0.009 A RMSD (DECISIONS D0-gauss); report "
                    "it as the floor of every D0 geometry metric")
HOLD_KEEP_KINDS = ("d0_react_ref", "d0_react_disp")   # reactant molecules: no TS geometry -> kept in FT
HOLD_TS_KINDS = ("d0_ts_near", "d0_ts_imag", "d0_ts_label", "d0_frag", "d0_g2_path", "d0_ts_final_g2")
G4_RESERVE_CORE_H = 150.0         # G-4 (spec 3: ~100 core-h for 1,000 D1 reactions) + ~50 for the D0 training side (D0-plus)

DECISIONS_D0 = {
    "D0-scope": "D0 = labels_all.json status ok (5,260) minus every reaction that is the same reaction as a D1 target "
                "by any of (a) the stereo-free canonical reaction SMILES (geo1b canon_rxn, copied unchanged), (b) the "
                "resonance-invariant reaction identity (D0-ident: sorted reactant identities >> sorted product "
                "identities), (c) the geo1b d0_folds.csv overlap rows (vs the D1 manifest); skipped entirely (counted). "
                "(b) adds 3270 (= D1 1001155/56/96/97; its ylide is written in another resonance form, so (a) and (c) "
                "miss it). STOP if an identity cannot be built, if a planned reaction still has a D1 identity, or if a "
                "d1_validation D0 reaction is in the skip set (its Phase I tags would be incomplete). Evaluation stays "
                "on the D1 T1 folds.",
    "D0-ident": "molecule identity = RDKit standard InChI of the map-free, stereo-free molecule (invariant to resonance "
                "form / charge placement, stereo, InChI mobile H; 'smi:' + stereo-free canonical SMILES if no InChI). "
                "Every D0 <-> D1 comparison uses it (reaction skip, D1-shared molecules and TS fragments, the D1 dipole of "
                "a shared dipole molecule) and the class union of D0-only molecules; the pre-review string rule (raw "
                "SMILES part) is reported only. Over-matching (stereo variants) only adds tags or skips; on these data "
                "identity = string + one resonance pair (C=[N+](C)[CH-]C(C)=O = [CH2-][N+](C)=CC(C)=O).",
    "D0-level": "D0 label TS = Coley G16 B3LYP-D3BJ/def2-SVP SMD(water) OptTS (profile TS_*.xyz, 5 decimals = the eda.inp "
                "coordinates); the reactants are the Coley r*.xyz (G16 level, inferred). Not stationary at the ORCA "
                "label level (Phase I d0_ts_grad median max|g| 4.7e-4 Eh/bohr vs D1 1.4e-5, ~30x): a program / level "
                "difference (G16 B3LYP = VWN3 vs ORCA B3LYP = VWN5, RIJCOSX def2/J, ORCA SMD, grids), not rounding; the "
                "D1 config 'parity with the D0 labels' refers to the TZVP SP / EDA stage, not to this gradient level. "
                "Every D0 structure is ordinary label-level E/F data; nothing assumes zero force; g0_d0 flags the parity "
                "against 10 x the measured D1 label-TS median (recorded, never a stop).",
    "D0-G1": "per D0 reaction 2 d0_ts_near + 1 d0_ts_imag with the G-near / G-imag rules (gen_dft) on an xTB Freq "
             "(gen_dft XTB_HEADER = the D1 TS Hessian header) at the D0 label TS, computed in the run step and cached as "
             "$MLT/dft/hess_d0/d0ts_<rid>.hess.gz; default_rng(SEED + 2,000,000 + rid), draws in the G1 order (the n1 "
             "and i rows do not depend on --n-near). Deviation from spec 3 ('label TS Hessian if one exists'): a G16 "
             "B3LYP/def2-SVP Hessian exists at every Coley TS (profile frequency_logs.tar.gz) but is not used (other "
             "program and level, archive orientation, no validated parser); xTB is the mode source of D1 G1 as well.",
    "D0-G3": "TS fragments = the (1)/(2) tags of the SMD-relabel eda.inp ($D0_EDA/rxn_%04d, (1) = dipole) in TS atom "
             "order, charge = labels charge{k}. Molecules = D0 Q keys (labels_d0_q cr.key1/key2: mirror-merged SMILES "
             "+ |alt|cfg + |s3d, the D0 analogue of the D1 cr.key: distinct reference states stay distinct); a key "
             "whose identity (D0-ident) is a D1 molecule identity is skipped (D1 G3 already has 3 structures of it); "
             "geometry = member with the lowest labels_all e_frag{k}_rel_eh (ties: smallest rid) = profile "
             "rel{k}_file; 3 structures (ref + 2 G-near displacements), seed keyhash('D0|' + key), Hessians "
             "$MLT/dft/hess_d0/d0m<sha>. A Q key without a SMILES (no identity) is a STOP.",
    "D0-G2": "pool = G2c D0 TSs ($G2C/d0/g2_final: lowest of 3 autodE xTB runs, label atom order) with result.json "
             "status ok, the G-2sel Hessian rules (gen_dft.g2_hess_check) and the label element order, minus overlaps "
             "and the 24 d1_validation D0 reactions; strata = D0 dipole class; ranking = per-stratum shuffled positions "
             "(j + u_s) / n_s merged (default_rng([SEED, 2,000,000, 2])), so every prefix is ~proportional; N = the "
             "largest prefix with D1 + D0 G1 + G3 + D0 G2 <= BUDGET_CORE_H (gen_dft; 8,000 since D0-plus) x (1 - 0.10) "
             "core-h on the budget basis (D0-budget; units already on disk cost 0, their measured core-h is fixed), or "
             "an explicit / pinned --n-g2 K (D0-cap, D0-pin); every reaction of the holdout list (D0-hold, D0-sib) is "
             "skipped in the prefix and is never refined (D0-gauss); input = the G3L builder with charge1 + charge2 and "
             "the label forming bonds; records runs/D0G2/<rid>. A re-plan that changes the selection while a selected "
             "unit has an OptTS start without unit.json is refused (exit 2) unless --force.",
    "D0-budget": "budget basis (plan_d0 and g0_d0): D1 = max(D1 G-0 total, G-0 group prediction x max(1, observed / "
                 "fitted core-h of the running D1 G1 / G2 / G3 records)); D0 EnGrad + xTB costs = fit x the D1 "
                 "production slowdown (G-0-weighted max(1, observed ratio) of G1 + G3: g0 nodes run 16 workers, "
                 "production 64); D0G2 = D1 g2 fit x max(1, D1 G-0 pilot ratio, D1 observed G2 ratio, D0 pilot / fit). "
                 "Fits: the g0_d0 fits when G0_estimate_d0.json belongs to this plan (fingerprint without n_near), "
                 "else the D1 G-0 fits + the xTB prior. Total > BUDGET_CORE_H (5,000 in spec 0-5; 5,500 D0-n3; 8,000 "
                 "D0-plus) = g0_d0 exit 3 (spec STOP); g0_d0 also reports the total with --n-near 1.",
    "D0-nnear": "--n-near 1|2 (default 2 = the G1 rule) may change after D0 records exist (same fingerprint apart from "
                "n_near): the n1 / i rows are identical for both values; records of dropped n2 rows stay on disk and "
                "collect --d0 keeps them (reported as extra).",
    "D0-tags": "rxn_ids / folds '' on every D0 structure except below (no D1 reaction; FT_k keeps them); dipoles = the "
               "D0 dipole class (FT_LODO_d drops D0 reactions of class d). TS fragments whose molecule identity is a D1 "
               "molecule identity: + the union of the D1 tags of every D1 target using that identity (all |fixed "
               "states merged; these rows follow the D1 rules, so they can be validation / test). D0-only molecules: "
               "dipoles = union of the classes of every planned D0 reaction using that identity (all Q keys). Extra "
               "info: d0_id (D0 rxn_id, never in rxn_ids), d0_class, charge.",
    "D0-class": "stereo_utils.dipole_backbone (ring rule, = d0_manifest dipole_smiles) -> (t1, c, t2); allyl type = "
                "central degree >= 3, O centre, or aromatic backbone (mesoionic rings); propargyl = central N of degree "
                "2. N allyl: CC azomethine ylide, CN d0_azomethine imine, CO nitrone; N propargyl: CC d0_nitrile ylide, "
                "CN d0_diazo (terminal N degree 1) / d0_nitrile imine, CO nitrile oxide, NN azide; O: CC carbonyl "
                "ylide, CN d0_carbonyl imine; anything else d0_other_<c><t1t2>. Must reproduce the D1 dipole of all "
                "9,266 targets and the D1 dipole of every shared dipole SMILES (else STOP). Mesoionic rings take the "
                "class of their backbone (conservative for LODO).",
    "D0-kinds": "new kind names d0_ts_near, d0_ts_imag, d0_ts_label, d0_frag, d0_react_ref, d0_react_disp, d0_g2_path, "
                "d0_ts_final_g2 (prefix as the Phase I d0_* kinds), groups D0G1 / D0G2 / D0G3; finetune TS-near = "
                "d0_ts_near / d0_ts_imag / d0_ts_label (+ Phase I d0_ts_grad) always, d0_g2_path / d0_ts_final_g2 when "
                "dform_label_A (written by collect --d0 against the D0 label TS) <= 0.3 A.",
    "D0-sep": "separate plan ($MLT/dft/plan/plan_d0.parquet, g2_plan_d0.json, g2_pool_d0.json, d0_class.csv), records "
              "runs/D0G1, D0G2, D0G3, Hessians hess_d0/, STOP_D0, outputs gen_D0G{1,2,3}.extxyz + "
              "results/G_report_d0.json; the D1 plan / run / collect never read them.",
    "D0-g0": "g0_d0: 30 EnGrads (6 per n_atoms quintile of D0G1 + D0G3) + 10 undisplaced D0 label TSs (n_atoms quantiles "
             "0.05-0.95; not the 24 Phase I d0_ts_grad reactions) + D0G2 pilots (--n-g2 auto: 1 = the first ranked "
             "pool unit with n_atoms <= the pool median, only if the plan selects > 0 D0G2 reactions; their measured "
             "core-h enters the total); fit = D0 log-log over every ok D0 EnGrad on disk if >= 10, else the D1 G-0 "
             "fit x the mean D0 / D1-fit ratio; xTB Freq c N^2 from the D0 Hessians (>= 3) else the prior; budget "
             "basis D0-budget; parity: median > 1e-3 Eh/bohr = off_stationary, > 10 x the D1 label-TS median = "
             "above_10x_d1_median (flags, never a stop); total > 5,000 core-h = exit 3 (spec STOP).",
    "D0-d1val": "the 24 D0 reactions of d1_validation (Phase I d0_* data) stay in D0G1 / D0G3 (no displaced or fragment "
                "data exist for them), are no parity picks (Phase I d0_ts_grad) and are not in the D0G2 pool.",
    "D0-charge": "charge = labels_all charge1 + charge2 (0/0 5,002, 0/-2 197, 0/+1 61), multiplicity 1; written to "
                 "records and extxyz info 'charge' (finetune total_charge, odd-electron filter).",
    "D0-stop": "D0 commands refuse to start under the D1 STOP; their own systematic failures write $MLT/dft/STOP_D0 "
               "only, so D0 never blocks a D1 resubmission.",
    "D0-plus": "user decision 'D0-plus' (2026-11-04, DECISIONS.md) as amended by 'D0-gauss': D0G2 = label-level OptTS "
               "paths from the G2c D0 xTB TS (G3L input, unchanged) for K = 200 training reactions; H = 100 "
               "evaluation-only (holdout) reactions: `gen_d0.py plan_d0 --n-near 2 --n-g2 200 --n-holdout 100`. "
               "Training = the first K units of the D0-G2 ranking outside the holdout list (= exactly the rank prefix "
               "of K: the holdout is drawn from reactant-pair groups without a training unit). D0G1 / D0G3 rows do not "
               "depend on K / H (same plan fingerprint).",
    "D0-gauss": "user decision 'D0-gauss' (2026-11-04 20:2x, 'd0도 gaussian을 라벨로 생각하고 하기'): the holdout "
                "reactions are not OptTS-refined (not in g2_plan_d0.json 'rxns', no D0G2 cost in the cap check, never "
                "'missing' in collect --d0); their reference TS = the Coley G16 TS (D0-hold-ref); their TS-derived "
                "structures stay out of every FT run; run_g2 --d0 runs exactly the K training units.",
    "D0-hold": "holdout (H evaluation reactions = H distinct reactant pairs, D0-sib): strata = the D0 dipole classes of "
               "the G2c pool (14 classes: the 5 D1 dipoles + 9 d0_* classes; the '20 per dipole class' of the request "
               "assumed 5 classes and is infeasible: nitrile oxide has 15 pool reactions in 8 reactant pairs), equal "
               "allocation by water-filling: quota_s = min(cap_s, L) with the largest L such that sum_s min(cap_s, L) "
               "<= H, the remainder +1 to the classes with cap_s > L in decreasing pool size (ties by name); cap_s = "
               "min(eligible groups_s, floor(0.5 x planned reactant pairs_s)) so a class keeps at least half of its "
               "reactant pairs for training; eligible group = no training D0G2 unit, no d1_validation reaction, no D1 "
               "target / D1-design reaction with the same reactant pair, one class; per class the eligible groups in "
               "D0-G2 rank order of their best pool member (seeded by the D0-G2 ranking rng), evaluation reaction = "
               "that best-ranked pool member, every other planned member = sibling. finetune.py drops every "
               "structure of a listed d0_id (eval + sibling) except d0_react_ref / d0_react_disp (by d0_id and by "
               "struct id, independent of the tags) and records the file sha256 in every data json; collect --d0 "
               "writes holdout=True into the extxyz info of their TS-derived rows.",
    "D0-sib": "reactant-pair siblings (review): a D0 reaction's group = the sorted D0-ident identities of its reactant "
              "side (stereo-free, map-free; other regio / stereo channels of the same two molecules share it; 5,254 "
              "planned reactions in 2,638 groups, 4,210 in groups of >= 2). Every planned D0 reaction in the group of "
              "an evaluation reaction is listed (role sibling) and its TS-derived rows are excluded from FT like the "
              "evaluation reaction's (the analogue of the D1 T1 folds, which keep a whole core together); siblings "
              "are not evaluated and never D0G2 training units (the training prefix skips every listed reaction). "
              "No planned D0 reactant pair is a D1 target pair or a Phase I D1-design pair (checked each plan; such a "
              "group would be ineligible).",
    "D0-hold-fix": "the holdout list is fixed by content: its sha256 is recorded in plan_d0_meta.json, g2_plan_d0.json "
                   "'holdout', d0_holdout_ref.json, results/d0_holdout.csv and every FT data json / split meta; plan_d0, "
                   "collect --d0 and finetune STOP (exit 3) when the file is missing while any record exists or when "
                   "its sha256 differs from any record. A later plan_d0 keeps the list whenever --n-holdout is omitted "
                   "or equals its evaluation count; a different list needs --replace-holdout (gen_d0 CLI; exit 3 "
                   "without) and is refused (exit 3) once any FT data json / split meta was built with a holdout or "
                   "with D0 inputs; a new list needs an explicit or pinned --n-g2 (the training prefix it avoids).",
    "D0-hold-ref": "reference TS of an evaluation reaction = the Coley G16 TS (profile TS_*.xyz = the D0 label TS of "
                   "D0G1, decision D0-gauss), not an ORCA re-optimised TS; starts for the supplementary evaluation: the "
                   "G2c xTB TS (xtb_start, label atom order) and the reference TS itself; metrics as FT-2; report the "
                   "Gaussian-ORCA stationary-point difference (~0.009 A RMSD) as the floor. collect --d0 checks that "
                   "both files exist and have the plan TS element order (reported, never replaced).",
    "D0-cap": "explicit or pinned --n-g2 K bypasses the auto prefix rule; plan_d0 still predicts the G total = D1 "
              "(budget basis) + D0 G1 + G3 (plan rows) + measured D0G2 on disk + predicted D0G2 of the training units "
              "without unit.json (D1 g2 fit x g2_ratio; the holdout costs 0, D0-gauss) + G-4 reserve 150 core-h and "
              "STOPs (exit 3, nothing written) only when it exceeds gen_dft.BUDGET_CORE_H (8,000: DECISIONS 'D0 계획 · "
              "실행 (20:1x)' recommendation after the measured D0G2 cost; the user's D0-plus text said 7,000 at the "
              "4.7 core-h guess); the auto rule is unchanged (prefix <= cap x 0.9; reported).",
    "D0-pin": "`gen_d0.py plan_d0 --n-g2 K` (explicit, gen_d0 CLI) pins K in plan_d0_meta.json; later plan_d0 calls "
              "without the gen_d0 CLI keep it (auto = K; the same explicit K = up to date) and an explicit other K "
              "is a STOP (exit 3: the launcher's `plan_d0 --n-g2 <n_g2_rule>` cannot silently drop the user's 200); "
              "`gen_d0.py plan_d0 --n-g2 K'` re-pins, `gen_d0.py plan_d0 --unpin` returns to the auto rule.",
}


# ================================================================ helpers
def canon_rxn(smi, stereo=False):
    """copied unchanged from analysis/geo1b/data.py canon_rxn (= analysis/geo/data.py; decision D-1)."""
    from rdkit import Chem
    out = []
    for side in smi.split(">>"):
        ms = []
        for s in side.split("."):
            m = Chem.MolFromSmiles(s)
            if m is None:
                return None
            for a in m.GetAtoms():
                a.SetAtomMapNum(0)
            m = Chem.RemoveHs(m)
            if not stereo:
                Chem.RemoveStereochemistry(m)
            ms.append(Chem.MolToSmiles(m))
        out.append(".".join(sorted(ms)))
    return ">>".join(out)


_IDENT = {}


def mol_ident(smi):
    """molecule identity (decision D0-ident): standard InChI of the map-free, stereo-free molecule; 'smi:' + its
    canonical SMILES if RDKit gives no InChI; None if the SMILES does not parse. Cached per process."""
    s = str(smi or "").strip()
    if not s:
        return None
    if s in _IDENT:
        return _IDENT[s]
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    x = None
    m = Chem.MolFromSmiles(s)
    if m is not None:
        for a in m.GetAtoms():
            a.SetAtomMapNum(0)
        m = Chem.RemoveHs(m)
        Chem.RemoveStereochemistry(m)
        try:
            x = Chem.MolToInchi(m) or None
        except Exception:                                                      # noqa: BLE001
            x = None
        if x is None:
            x = "smi:" + Chem.MolToSmiles(m)
    _IDENT[s] = x
    return x


def rxn_ident(smi):
    """reaction identity (decision D0-scope (b)): sorted reactant identities >> sorted product identities; None if a
    part has no identity."""
    sides = str(smi).split(">>")
    if len(sides) != 2:
        return None
    out = []
    for side in sides:
        xs = [mol_ident(s) for s in side.split(".")]
        if None in xs:
            return None
        out.append(".".join(sorted(xs)))
    return ">>".join(out)


def pair_ident(smi):
    """reactant-pair identity (decision D0-sib): the reactant side of rxn_ident; None if it has no identity."""
    y = rxn_ident(smi)
    return y.split(">>")[0] if y else None


def d1design_ids():
    """(reaction identities, reactant-pair identities, ts_ids) of the Phase I D1-design d1_validation reactions (ts_id
    D1-NNNN-rK in jobs/<job>/label.json | ts_result.json, v7 via pilot_copy; the inv_dft val_info sources)."""
    rx, pr, ids = set(), set(), set()
    metas = []
    jd = M.D1VAL / "jobs"
    for j in (sorted(p for p in jd.iterdir() if p.is_dir()) if jd.is_dir() else []):
        for n in ("label.json", "ts_result.json"):
            m = G._rj(j / n)
            if m:
                metas.append(m)
                break
    for d in (sorted((M.D1VAL / "v7").glob("*")) if (M.D1VAL / "v7").is_dir() else []):
        m = G._rj(M.D1VAL / "pilot_copy" / "jobs" / d.name / "label.json")
        if m:
            metas.append(m)
    for m in metas:
        if D1DESIGN_TS.fullmatch(str(m.get("ts_id", ""))) and m.get("rxn_smiles"):
            y = rxn_ident(m["rxn_smiles"])
            if y is None:
                M.stop(f"D1-design reaction {m['ts_id']} without an identity (decision D0-sib)")
            rx.add(y)
            pr.add(y.split(">>")[0])
            ids.add(str(m["ts_id"]))
    return rx, pr, sorted(ids)


_SU = None


def _su():
    global _SU
    if _SU is None:
        if str(M.PILOT) not in sys.path:
            sys.path.insert(0, str(M.PILOT))
        import stereo_utils                                                    # noqa: E402  (D1 copy, read only)
        _SU = stereo_utils
    return _SU


_ALLYL_N = {("C", "C"): "azomethine ylide", ("C", "N"): "d0_azomethine imine", ("C", "O"): "nitrone",
            ("N", "N"): "d0_azimine", ("N", "O"): "d0_azoxy", ("O", "O"): "d0_nitro"}
_PROP_N = {("C", "C"): "d0_nitrile ylide", ("C", "O"): "nitrile oxide", ("N", "N"): "azide", ("N", "O"): "d0_nitrous oxide"}
_O = {("C", "C"): "carbonyl ylide", ("C", "N"): "d0_carbonyl imine", ("C", "O"): "d0_carbonyl oxide"}


def classify(mapped):
    """(dipole class, info) of an atom-mapped reaction SMILES (decision D0-class)."""
    from rdkit import Chem
    kdip, (t1, c, t2), _ = _su().dipole_backbone(mapped)
    m = Chem.MolFromSmiles(mapped.split(">>")[0].split(".")[kdip])
    by = {a.GetAtomMapNum(): a for a in m.GetAtoms()}
    A, C, B = by[t1], by[c], by[t2]
    bA, bB = m.GetBondBetweenAtoms(A.GetIdx(), C.GetIdx()), m.GetBondBetweenAtoms(B.GetIdx(), C.GetIdx())
    arom = bool(C.GetIsAromatic() or bA.GetIsAromatic() or bB.GetIsAromatic())
    ce, term, deg = C.GetSymbol(), tuple(sorted([A.GetSymbol(), B.GetSymbol()])), int(C.GetTotalDegree())
    allyl = bool(arom or ce != "N" or deg >= 3)
    if ce == "N" and allyl:
        cls = _ALLYL_N.get(term)
    elif ce == "N":
        if term == ("C", "N"):
            tn = A if A.GetSymbol() == "N" else B
            cls = "d0_diazo" if tn.GetTotalDegree() == 1 else "d0_nitrile imine"
        else:
            cls = _PROP_N.get(term)
    elif ce == "O":
        cls = _O.get(term)
    else:
        cls = None
    cls = cls or f"d0_other_{ce}{''.join(term)}"
    return cls, dict(central=ce, termini="".join(term), central_degree=deg, allyl=allyl, aromatic=arom,
                     bonds=f"{bA.GetBondType()}/{bB.GetBondType()}", dipole_index=int(kdip))


def _classify_safe(mapped):
    try:
        return classify(mapped)
    except Exception as e:                                                     # noqa: BLE001
        return "d0_unclassified", dict(error=f"{type(e).__name__}: {e}"[:200])


def _stats(v):
    return G._stats(v)


def smi_part(key):
    return str(key).split(":", 1)[1].split("|")[0]


def d1val_ids():
    d = M.D1VAL / "v2a"
    return sorted(int(p.name) for p in d.iterdir() if p.is_dir() and p.name.isdigit()) if d.is_dir() else []


def any_records_d0():
    for g in ("D0G1", "D0G3"):
        d = G.RUNS / g
        if d.is_dir() and next(d.glob("*.json"), None) is not None:
            return True
    d = G.RUNS / "D0G2"
    return d.is_dir() and next(d.glob("*/unit.json"), None) is not None


def d1_g0():
    e = G._rj(D1_G0)
    if not e or not e.get("engrad_fit") or not e.get("g2_fit"):
        M.not_ready(f"D1 G-0 estimate {D1_G0} missing or without fits (needed for the D0 budget)")
    return e


def pw(fit, n):
    return float(fit["a"]) * float(n) ** float(fit["b"])


def d1_observed(e1, nmax=1500):
    """observed / fitted core-h of the running D1 records (report only; sampled evenly by file name)."""
    out = {}
    for g in ("G1", "G3"):
        d = D1_RUNS / g
        files = sorted(d.glob("*.json")) if d.is_dir() else []
        take = files[::max(1, len(files) // nmax)][:nmax] if files else []
        ch, ft = 0.0, 0.0
        n = 0
        for f in take:
            r = G._rj(f)
            if r and r.get("status") == "ok" and r.get("core_h") and r.get("n_atoms"):
                ch += float(r["core_h"])
                ft += pw(e1["engrad_fit"], r["n_atoms"])
                n += 1
        out[g] = dict(n_sampled=n, n_files=len(files), ratio=ch / ft if ft else None)
    d = D1_RUNS / "G2"
    ch, ft, n = 0.0, 0.0, 0
    for p in (sorted(d.glob("*/unit.json")) if d.is_dir() else []):
        u = G._rj(p)
        if u and u.get("core_h") and u.get("n_atoms") and u.get("status") in ("ok", "fail", "not_converged"):
            ch += float(u["core_h"])
            ft += pw(e1["g2_fit"], u["n_atoms"])
            n += 1
    out["G2"] = dict(n_units=n, ratio=ch / ft if ft else None)
    pred = (e1.get("predicted_core_h") or {}).get("by_group") or {}
    proj = {g: float(pred.get(g) or 0.0) * max(1.0, float(out[g]["ratio"] or 1.0)) for g in ("G1", "G2", "G3")}
    out["d1_projected_core_h"] = dict(by_group=proj, total=float(sum(proj.values())),
                                      rule="G-0 group prediction x max(1, observed ratio)")
    return out


def budget_basis(e1):
    """decision D0-budget: the D1 total and the production slowdowns every D0 budget number uses."""
    obs = d1_observed(e1)
    pred = (e1.get("predicted_core_h") or {}).get("by_group") or {}
    w = {g: float(pred.get(g) or 0.0) for g in ("G1", "G3")}
    r = {g: max(1.0, float(obs[g]["ratio"] or 1.0)) for g in ("G1", "G3")}
    slow = sum(w[g] * r[g] for g in w) / sum(w.values()) if sum(w.values()) > 0 else 1.0
    d1_g0 = float(e1["predicted_core_h"]["total"])
    d1_proj = float(obs["d1_projected_core_h"]["total"])
    return dict(d1_total=max(d1_g0, d1_proj), d1_g0_total=d1_g0, d1_projected_total=d1_proj,
                engrad_slowdown=float(slow), g2_slowdown=max(1.0, float(obs["G2"]["ratio"] or 1.0)),
                rule=DECISIONS_D0["D0-budget"], d1_observed=obs)


def fp_core(fp):
    """the plan fingerprint without n_near (decision D0-nnear)."""
    return {k: v for k, v in (fp or {}).items() if k != "n_near"}


def plan_fits(fp, e1, basis):
    """EnGrad / xTB fits + D0G2 ratio for the plan's budget rule (decision D0-budget) and their source string."""
    e = G._rj(G.G0_D0_JSON)
    if e and fp_core(e.get("plan_fingerprint")) == fp_core(fp) and e.get("engrad_fit_used") and e.get("g2_ratio"):
        ef = {k: e["engrad_fit_used"][k] for k in ("a", "b")}
        return ef, dict((e.get("xtb_freq") or {}).get("fit") or XTB_PRIOR), float(e["g2_ratio"]), \
            f"g0_d0:{e.get('written')}"
    g2r = max(1.0, float(e1.get("g2_pilot_ratio") or 1.0), basis["g2_slowdown"])
    return {k: e1["engrad_fit"][k] for k in ("a", "b")}, dict(XTB_PRIOR), g2r, "d1_g0_fit"


# ================================================================ D0-plus holdout (decisions D0-plus, D0-gauss, D0-hold,
#                                                                   D0-sib, D0-hold-fix, D0-hold-ref, D0-cap, D0-pin)
def read_holdout(path=None):
    """rows of the holdout list (dicts in file order), [] for a header-only file, None if there is no file."""
    import pandas as pd
    p = Path(path or HOLD_CSV)
    if not p.is_file():
        return None
    H = pd.read_csv(p, keep_default_na=False)
    if tuple(H.columns) != HOLD_COLS:
        M.stop(f"{p}: columns {list(H.columns)} != {list(HOLD_COLS)} (decision D0-hold)")
    rows = [dict(d0_id=int(r.d0_id), d0_class=str(r.d0_class), n_atoms=int(r.n_atoms), reason=str(r.reason),
                 role=str(r.role)) for r in H.itertuples(index=False)]
    if len({r["d0_id"] for r in rows}) != len(rows):
        M.stop(f"{p}: duplicate d0_id (decision D0-hold)")
    bad = [r["d0_id"] for r in rows if r["role"] not in HOLD_ROLES]
    if bad:
        M.stop(f"{p}: role not in {HOLD_ROLES} for {bad[:5]} (decision D0-hold)")
    return rows


def holdout_ids(role=None):
    """d0_ids of the holdout list (every role, or one role); empty without a file."""
    return {int(r["d0_id"]) for r in (read_holdout() or []) if role is None or r["role"] == role}


def holdout_records(old_meta=None, with_ft=True):
    """decision D0-hold-fix: {record: sha256} of every place that recorded the holdout file (non-empty only)."""
    meta = old_meta if old_meta is not None else (G._rj(G.PLAN_D0_META) or {})
    out = {"plan_d0_meta.json": meta.get("holdout_sha256"),
           "g2_plan_d0.json": ((G._rj(G.G2_PLAN_D0) or {}).get("holdout") or {}).get("sha256"),
           "d0_holdout_ref.json": (G._rj(HOLD_REF) or {}).get("holdout_sha256"),
           "results/d0_holdout.csv": M.sha256(HOLD_CSV_RES) if HOLD_CSV_RES.is_file() else None}
    if with_ft:
        for b in ft_data_built():
            out[b["path"]] = b["holdout_sha256"]
    return {k: v for k, v in out.items() if v}


def holdout_integrity(old_meta=None):
    """(current sha256 or None, records, problem or None) of the holdout file (decision D0-hold-fix)."""
    rec = holdout_records(old_meta)
    cur = M.sha256(HOLD_CSV) if HOLD_CSV.is_file() else None
    if cur is None and rec:
        return cur, rec, (f"{HOLD_CSV} is missing but a holdout was recorded in {sorted(rec)[:4]}; restore it (e.g. from "
                          f"{HOLD_CSV_RES} if its sha256 {rec.get('results/d0_holdout.csv', '?')[:12]} matches the "
                          f"other records)")
    bad = {k: v[:12] for k, v in rec.items() if cur is not None and v != cur}
    if bad:
        first = dict(list(bad.items())[:4])
        return cur, rec, f"{HOLD_CSV} sha256 {cur[:12]} differs from {len(bad)} record(s) {first} (edited or replaced)"
    return cur, rec, None


def holdout_check(where):
    """collect / g0 guard: STOP (exit 3) on a missing or changed holdout file (decision D0-hold-fix)."""
    _, _, prob = holdout_integrity()
    if prob:
        M.stop(f"{where}: {prob} (decision D0-hold-fix)")


def holdout_pick(ranked, train_ids, k, g):
    """decisions D0-hold / D0-sib: (rows: role eval + role sibling, info). ranked = the D0-G2 pool in rank order;
    train_ids = the training prefix; g = dict(pair0 = planned d0_id -> reactant-pair identity, by_pair = identity ->
    planned d0_ids, cls / natoms = planned d0_id -> class / n_atoms, block = identity -> reason (D1 target / D1-design
    pair), d1val = d1_validation d0_ids). One evaluation reaction per reactant-pair group (its best-ranked pool member,
    the other planned members = siblings); equal allocation by water-filling over the classes, cap_s = min(eligible
    groups_s, floor(HOLD_MAX_FRAC x planned reactant pairs_s)); per class the eligible groups in rank order."""
    train_ids = {int(x) for x in train_ids}
    pair0, by_pair, cls, natoms = g["pair0"], g["by_pair"], g["cls"], g["natoms"]
    why = {}
    for p, mem in by_pair.items():
        if p in g["block"]:
            why[p] = g["block"][p]
        elif any(m in train_ids for m in mem):
            why[p] = "has_training_d0g2_unit"
        elif any(m in g["d1val"] for m in mem):
            why[p] = "has_d1_validation_reaction"
        elif len({cls[m] for m in mem}) > 1:
            why[p] = "mixed_class"
        else:
            why[p] = None
    strata = defaultdict(list)
    for u in ranked:
        strata[u["d0_class"]].append(u)
    names = sorted(strata)
    npool = {s: len(strata[s]) for s in names}
    ngrp = {s: len({pair0[m] for m in cls if cls[m] == s}) for s in names}          # planned reactant pairs per class
    best = {s: {} for s in names}                       # class -> eligible group -> its best-ranked pool member
    for s in names:
        for u in strata[s]:
            p = pair0[int(u["rxn_id"])]
            if why[p] is None and p not in best[s]:
                best[s][p] = u
    cap = {s: min(len(best[s]), int(math.floor(HOLD_MAX_FRAC * ngrp[s]))) for s in names}
    if sum(cap.values()) < int(k):
        M.stop(f"the D0G2 pool can give only {sum(cap.values())} holdout reactant pairs under the class caps, {k} asked "
               f"(decision D0-hold): {cap}")
    lev = 0
    while any(cap[s] > lev for s in names) and sum(min(cap[s], lev + 1) for s in names) <= int(k):
        lev += 1
    quota = {s: min(cap[s], lev) for s in names}
    rem = int(k) - sum(quota.values())
    for s in sorted([s for s in names if cap[s] > lev], key=lambda s: (-npool[s], s))[:max(0, rem)]:
        quota[s] += 1
    if sum(quota.values()) != int(k):
        M.stop(f"holdout allocation {quota} does not sum to {k} (decision D0-hold)")
    rows, gi, groups, by = [], 0, [], {}
    for s in names:
        n_sib = 0
        for j, (p, u) in enumerate(list(best[s].items())[:quota[s]]):    # dict order = rank order of the best member
            gi += 1
            ev = int(u["rxn_id"])
            rows.append(dict(d0_id=ev, d0_class=s, n_atoms=int(u["n_atoms"]), role="eval",
                             reason=f"D0-plus holdout {j + 1}/{quota[s]} of class {s} (G2c pool {npool[s]}, reactant "
                                    f"pairs {ngrp[s]}, eligible {len(best[s])}, cap {cap[s]}): best-ranked pool member of "
                                    f"reactant-pair group g{gi:03d} ({len(by_pair[p])} planned), pool rank "
                                    f"{int(u['rank'])}"))
            sib = sorted(m for m in by_pair[p] if m != ev)
            for m in sib:
                rows.append(dict(d0_id=int(m), d0_class=cls[m], n_atoms=int(natoms[m]), role="sibling",
                                 reason=f"D0-sib: same reactant pair as holdout {ev} (group g{gi:03d}); TS-derived rows "
                                        f"excluded from FT, not evaluated"))
            n_sib += len(sib)
            groups.append(dict(group=f"g{gi:03d}", d0_class=s, eval=ev, siblings=sib, pair=p))
        nplan = sum(1 for m in cls if cls[m] == s)
        by[s] = dict(pool=npool[s], planned=nplan, reactant_pairs=ngrp[s], eligible_pairs=len(best[s]), cap=cap[s],
                     holdout=quota[s], siblings=n_sib, excluded=quota[s] + n_sib,
                     excluded_frac=round((quota[s] + n_sib) / max(1, nplan), 3))
    info = dict(rule=DECISIONS_D0["D0-hold"], sib_rule=DECISIONS_D0["D0-sib"], n=int(k), level=lev,
                n_classes=len(names), max_frac=HOLD_MAX_FRAC, n_siblings=sum(r["role"] == "sibling" for r in rows),
                n_groups=len(groups), groups_ineligible=dict(Counter(str(v) for v in why.values() if v)),
                n_groups_planned=len(by_pair), by_class=by, groups=groups)
    return rows, info


def write_holdout(rows):
    """d0_holdout.csv (fixed columns, deterministic bytes: the same list always has the same sha256)."""
    import pandas as pd
    M.write_csv(HOLD_CSV, pd.DataFrame([{c: r[c] for c in HOLD_COLS} for r in rows], columns=list(HOLD_COLS)))


def sync_holdout_copy():
    """results/d0_holdout.csv = byte copy of $MLT/dft/plan/d0_holdout.csv (rewritten only when it differs)."""
    if HOLD_CSV.is_file() and (not HOLD_CSV_RES.is_file() or M.sha256(HOLD_CSV_RES) != M.sha256(HOLD_CSV)):
        M.write_text(HOLD_CSV_RES, HOLD_CSV.read_text())


def ft_data_built():
    """FT data jsons (results/ft/*_data.json) and split metas ($MLT/ft/data/*/meta.json): the holdout sha256 each was
    built with and whether it read D0 inputs (decision D0-hold: a holdout change after either = exit 3)."""
    out = []
    for p in sorted((M.RES / "ft").glob("*_data.json")) + sorted((M.MLT / "ft" / "data").glob("*/meta.json")):
        j = G._rj(p) or {}
        ho = j.get("d0_holdout") or (j.get("pool") or {}).get("d0_holdout") or {}
        inp = j.get("inputs") or {}
        out.append(dict(path=str(p), holdout_sha256=ho.get("sha256") or j.get("d0_holdout_sha256"),
                        d0_inputs=bool(any((inp.get(n) or {}).get("present") for n in G.D0_GROUPS))))
    return out


def write_holdout_ref(eval_units, hold_rows, sha, pair0):
    """decision D0-hold-ref: the evaluation reactions with their reference TS (Coley G16 TS) and starts + the siblings.
    Content-only (no timestamp): rewritten only when it changes."""
    sib = defaultdict(list)
    for r in hold_rows:
        if r["role"] == "sibling":
            sib[pair0.get(int(r["d0_id"]))].append(int(r["d0_id"]))
    units = []
    for u in eval_units:
        rid = int(u["rxn_id"])
        units.append(dict(d0_id=rid, d0_class=u["d0_class"], n_atoms=int(u["n_atoms"]), charge=int(u["charge"]),
                          pairs=u["pairs"], pool_rank=int(u["rank"]),
                          reference=dict(kind="coley_g16_ts", xyz=u["label_ts"],
                                         level="Coley G16 B3LYP-D3BJ/def2-SVP SMD(water) OptTS (= the D0 label TS of "
                                               "D0G1; not stationary at the ORCA label level, D0-level)",
                                         floor=GAUSS_FLOOR_NOTE),
                          starts=dict(xtb_g2c=u["start"], reference=u["label_ts"]), g2c_hess=u["hess"],
                          siblings=sorted(sib.get(pair0.get(rid), []))))
    obj = dict(code=CODE_D0_REV, rule=DECISIONS_D0["D0-hold-ref"], gauss=DECISIONS_D0["D0-gauss"],
               holdout_csv=str(HOLD_CSV), holdout_sha256=sha, n=len(units),
               n_siblings=sum(r["role"] == "sibling" for r in hold_rows),
               note="evaluation-only D0 reactions (not OptTS-refined, D0-gauss); siblings are excluded from FT and not "
                    "evaluated (D0-sib)", units=units)
    old = G._rj(HOLD_REF)
    if old != json.loads(json.dumps(obj, default=str)):
        M.write_json(HOLD_REF, obj)


# ================================================================ plan
def rxn_d0_job(t):
    """plan rows of one D0 reaction: D0G1 (3, displaced in the run step) + D0G3 frag (2) + checks (pool worker)."""
    rid = int(t["d0_id"])
    out = dict(d0_id=rid, rows=[], g1_err=None, frag_err=None, chk={}, ref_syms=None, ts_syms=None)
    tags = dict(rxn_ids="", folds="", dipoles=t["cls"])
    base = dict(d0_id=rid, d0_class=t["cls"], fold_T1=-9)
    q = int(t["charge1"]) + int(t["charge2"])
    try:
        syms, pos = M.read_xyz(t["ts"])
        pos = np.asarray(pos, float)
        if len(syms) != int(t["n_atoms"]):
            raise ValueError(f"TS xyz has {len(syms)} atoms, label n_atoms {t['n_atoms']}")
        if any(not (0 <= i < len(syms) and 0 <= j < len(syms)) for i, j in t["pairs"]):
            raise ValueError(f"forming pairs {t['pairs']} outside the TS")
        out["ts_syms"] = list(syms)
    except Exception as e:                                                     # noqa: BLE001
        out["g1_err"] = out["frag_err"] = f"label_ts: {type(e).__name__}: {e}"[:300]
        return out
    try:
        out["ref_syms"] = [M.read_xyz(p)[0] for p in t["refs"]]
    except Exception as e:                                                     # noqa: BLE001
        out["chk"]["ref_err"] = f"{type(e).__name__}: {e}"[:200]
    # ---- D0G1 (positions = the undisplaced label TS; pos_final False: ts_disp_d0 displaces in the run step)
    com = dict(hess_src=str(G.HESS_D0 / f"d0ts_{rid}.hess.gz"), mol_id=f"d0ts_{rid}", seed=M.SEED + D0_SEED_OFF + rid,
               pos_final=False, **base)
    for suf, kind, j in (("n1", "d0_ts_near", 1), ("n2", "d0_ts_near", 2), ("i", "d0_ts_imag", 0)):
        if j > int(t.get("n_near", 2)):
            continue                                     # --n-near 1: n2 not planned (n1 / i draws unchanged)
        out["rows"].append(G.row(f"d0g1_{rid}_{suf}", "D0G1", kind, syms, pos, tags, t["ts"], q,
                                 disp=json.dumps(dict(e_index=j, pairs=t["pairs"], displacement="run step (D0-G1)")),
                                 **com))
    # ---- D0G3 fragments (eda.inp partition)
    try:
        es, tg, ex = G.eda_atoms(t["eda"])
        if es != list(syms):
            raise ValueError("eda.inp elements != D0 label TS")
        cdev = float(np.abs(ex - pos).max())
        if cdev > 1e-6:
            raise ValueError(f"eda.inp coordinates != D0 label TS ({cdev:.2e} A)")
        if set(tg) != {1, 2}:
            raise ValueError(f"eda.inp fragment tags {sorted(set(tg))}")
        fr = {k: [i for i, g in enumerate(tg) if g == k] for k in (1, 2)}
        for k in (1, 2):
            if len(fr[k]) != int(t[f"n_f{k}"]):
                raise ValueError(f"eda.inp fragment {k}: {len(fr[k])} atoms, label n_f{k} {t[f'n_f{k}']}")
        f1 = set(fr[1])
        for k in (1, 2):
            fs = [syms[i] for i in fr[k]]
            same_ref = bool(out["ref_syms"] and Counter(out["ref_syms"][k - 1]) == Counter(fs))
            same_order = bool(out["ref_syms"] and out["ref_syms"][k - 1] == fs)
            ft = t["frag_tags"][str(k)]
            out["rows"].append(G.row(f"d0g3_{rid}_f{k}", "D0G3", "d0_frag", fs, pos[fr[k]], ft, t["eda"], t[f"charge{k}"],
                                     disp=json.dumps(dict(partition="D0 SMD-relabel eda.inp tags", frag=k, atoms=fr[k],
                                                          ref_formula_equal=same_ref, ref_order_equal=same_order,
                                                          shared_d1_smiles=bool(t["frag_shared"][str(k)]))), **base))
            out["chk"][f"frag{k}_ref_formula_equal"] = same_ref
        out["chk"].update(first_nf1_is_frag1=all(g == 1 for g in tg[:int(t["n_f1"])]),
                          pairs_cross_fragments=all((i in f1) != (j in f1) for i, j in t["pairs"]))
    except Exception as e:                                                     # noqa: BLE001
        out["frag_err"] = f"{type(e).__name__}: {e}"[:300]
    return out


def molecules_d0(rids, L, Qk, cls, refsyms, d1ids, d1raw=()):
    """D0G3 d0_react_ref / d0_react_disp rows of the D0-only molecules (decisions D0-G3, D0-ident, D0-tags) + stats.
    d1ids = D1 molecule identities (skip), d1raw = D1 raw SMILES parts (pre-review string rule, report only)."""
    groups, nokey, id_cls = {}, [], defaultdict(set)
    for rid in rids:
        d = L[rid]
        for k in (1, 2):
            key = Qk.get(rid, {}).get(f"key{k}")
            x = mol_ident(smi_part(key)) if key else None
            if not key or x is None:
                nokey.append(dict(d0_id=rid, k=k, key=key))
                continue
            rs = refsyms.get(rid)
            groups.setdefault(key, []).append(dict(d0_id=rid, k=k, e=float(d[f"e_frag{k}_rel_eh"]), ident=x,
                                                   path=str(D0_PROF / str(rid) / str(d[f"rel{k}_file"])),
                                                   charge=int(d.get(f"charge{k}") or 0), cls=cls[rid],
                                                   syms=rs[k - 1] if rs else None))
            id_cls[x].add(cls[rid])
    if nokey:
        M.stop(f"D0 Q keys without a molecule identity (decision D0-G3): {len(nokey)}; first {nokey[:3]}")
    rows, shared, bad_formula, bad_charge, sizes = [], [], [], [], []
    ident = {k: mem[0]["ident"] for k, mem in groups.items()}
    if any(len({m["ident"] for m in mem}) > 1 for mem in groups.values()):
        M.stop("a D0 Q key with members of different identities")
    shared_set = {k for k in groups if ident[k] in d1ids}
    raw_set = {k for k in groups if smi_part(k) in d1raw}
    n_union_wider = 0
    for key in sorted(groups):
        mem = groups[key]
        if key in shared_set:
            shared.append(key)
            continue
        rep = min(mem, key=lambda m: (m["e"], m["d0_id"]))
        if len({G.formula(m["syms"]) for m in mem if m["syms"]}) > 1:
            bad_formula.append(key)
        if len({m["charge"] for m in mem}) > 1:
            bad_charge.append(key)
        syms, pos = M.read_xyz(rep["path"])
        own = {m["cls"] for m in mem}
        n_union_wider += bool(id_cls[ident[key]] - own)
        dips = M.tag_list(own | id_cls[ident[key]])
        tags = dict(rxn_ids="", folds="", dipoles=dips)
        mid = "d0m" + G.mol_id("D0|" + key)[1:]
        com = dict(mol_id=mid, mol_key=key, seed=G.keyhash("D0|" + key), d0_id=rep["d0_id"], d0_class=dips, fold_T1=-9)
        info = dict(rep_d0_id=rep["d0_id"], rep_frag=rep["k"], e_rel_orig_eh=rep["e"], n_members=len(mem),
                    members=sorted({m["d0_id"] for m in mem})[:50], ident=ident[key])
        rows.append(G.row(f"d0g3_{mid}_ref", "D0G3", "d0_react_ref", syms, pos, tags, rep["path"], rep["charge"],
                          disp=json.dumps(info), **com))
        for j in (1, 2):
            rows.append(G.row(f"d0g3_{mid}_d{j}", "D0G3", "d0_react_disp", syms, pos, tags, rep["path"], rep["charge"],
                              pos_final=False, hess_src=str(G.HESS_D0 / f"{mid}.hess.gz"),
                              disp=json.dumps(dict(info, e_index=j)), **com))
        sizes.append(len(mem))
    new_ids = {ident[k] for k in groups if k not in shared_set}
    st = dict(method=DECISIONS_D0["D0-G3"], identity=DECISIONS_D0["D0-ident"], n_keys=len(groups),
              n_occurrences=sum(len(v) for v in groups.values()),
              n_shared_with_d1_skipped=len(shared), shared_keys=shared[:60],
              n_shared_string_rule=len(raw_set), shared_identity_only=sorted(shared_set - raw_set),
              shared_string_only=sorted(raw_set - shared_set),
              shared_by_kind=dict(Counter(k.split(":", 1)[0] for k in shared)), n_new=len(groups) - len(shared),
              new_by_kind=dict(Counter(k.split(":", 1)[0] for k in groups if k not in shared_set)),
              n_alt_keys_new=sum("|alt" in k for k in groups if k not in shared_set), n_without_key=len(nokey),
              n_new_identities=len(new_ids), n_new_keys_class_union_wider=n_union_wider,
              members_per_molecule=_stats(sizes), n_keys_formula_differs=len(bad_formula),
              keys_formula_differs=bad_formula[:20], n_keys_charge_differs=len(bad_charge),
              keys_charge_differs=bad_charge[:20], n_structures=len(rows))
    return rows, st, shared_set


def _g2_check_d0(rid):
    """gen_dft.g2_hess_check on the G2c D0 root + the G2c result status (pool worker)."""
    g2d = D0_G2_FINAL / str(rid)
    c = G.g2_hess_check(rid, D0_G2_FINAL)
    res = G._rj(g2d / "result.json") or {}
    c.update(g2c_status=res.get("status"), g2c_n_imag=res.get("n_imag"),
             g2c_chosen=(G._rj(g2d / "g2c.json") or {}).get("chosen"))
    if c["ok"] and res.get("status") != "ok":
        c.update(ok=False, reason="g2c_status_not_ok")
    return c


def g2_pool_d0(rids, L, cls, ts_syms, nproc):
    """ranked D0G2 pool (decision D0-G2): list of unit dicts in rank order + info."""
    skip_val = set(d1val_ids())
    cand = [r for r in rids if r not in skip_val and (D0_G2_FINAL / str(r)).is_dir()]
    with ProcessPoolExecutor(max(1, nproc)) as ex:
        checks = dict(zip(cand, ex.map(_g2_check_d0, cand, chunksize=16)))
    units, why = [], Counter()
    for rid in cand:
        c = checks[rid]
        if not c["ok"]:
            why[c["reason"]] += 1
            continue
        syms, pos = M.read_xyz(c["start"])
        if ts_syms.get(rid) is None or list(syms) != list(ts_syms[rid]):
            why["element_order_vs_label"] += 1
            continue
        d = L[rid]
        q = int(d.get("charge1") or 0) + int(d.get("charge2") or 0)
        units.append(dict(rxn_id=rid, d0_id=rid, job_id=f"D0G2_{rid:04d}", struct_id=f"d0g2_{rid}", run_group="D0G2",
                          n_atoms=len(syms), n_heavy=G.n_heavy(syms), dipole=cls[rid], d0_class=cls[rid],
                          g1_class="none", fold=-9, charge=q,
                          pairs=[list(p) for p in G.label_pairs(d["formed_pairs_ts"])], start=c["start"],
                          hess=c["hess"], hess_sha256=M.sha256(c["hess"]), hess_translated=bool(c.get("translated")),
                          hess_shift_A=c.get("shift_A"), g2c_chosen=c.get("g2c_chosen"),
                          label_ts=str(D0_PROF / str(rid) / str(d["ts_file"])), rxn_ids="", folds="",
                          dipoles=cls[rid]))
    rng = np.random.default_rng([M.SEED, D0_SEED_OFF, 2])
    strata = defaultdict(list)
    for u in units:
        strata[u["d0_class"]].append(u)
    keyed = []
    for s in sorted(strata):
        mem = strata[s]
        perm = rng.permutation(len(mem))
        off = float(rng.uniform())
        keyed += [((j + off) / len(mem), s, mem[int(p)]) for j, p in enumerate(perm)]
    keyed.sort(key=lambda x: (x[0], x[1]))
    ranked = [dict(x[2], rank=i) for i, x in enumerate(keyed)]
    info = dict(rule=DECISIONS_D0["D0-G2"], n_candidates=len(cand), n_d1val_excluded=len(set(rids) & skip_val),
                n_no_g2c_dir=sum(1 for r in rids if r not in skip_val and not (D0_G2_FINAL / str(r)).is_dir()),
                excluded=dict(why), n_pool=len(ranked), strata={s: len(v) for s, v in sorted(strata.items())},
                n_translated_hess=sum(u["hess_translated"] for u in ranked),
                g2c_chosen=dict(Counter(str(u["g2c_chosen"]) for u in ranked)),
                n_atoms=_stats([u["n_atoms"] for u in ranked]))
    return ranked, info


def cost_rows(P, efit, xfit, slow=1.0):
    """predicted core-h of the D0G1 + D0G3 plan rows (EnGrads) + their xTB Freqs, by item, x the production slowdown
    (decision D0-budget); '_n2_part' = the share of the d0_ts_near n2 rows (dropped by --n-near 1; not summed)."""
    pred = {}
    E = P[P.group.isin(["D0G1", "D0G3"])]
    for (g, k), x in E.groupby(["group", "kind"]):
        pred[f"{g}/{k}"] = float(slow) * float(sum(pw(efit, n) for n in x.n_atoms))
    ts = E[E.kind.isin(G.D0_TS_DISP_KINDS)].drop_duplicates("mol_id")
    mol = E[E.kind == "d0_react_disp"].drop_duplicates("mol_id")
    pred["D0G1/xtb_ts_hessians"] = float(slow) * float(sum(pw(xfit, n) for n in ts.n_atoms))
    pred["D0G3/xtb_mol_hessians"] = float(slow) * float(sum(pw(xfit, n) for n in mol.n_atoms))
    n2 = E[(E.kind == "d0_ts_near") & E.struct_id.str.endswith("_n2")]
    return pred, float(slow) * float(sum(pw(efit, n) for n in n2.n_atoms))


def n_g2_rule(d1_total, d0_fixed, ranked, gfit, ratio, done=()):
    """largest ranked prefix with D1 + D0 fixed (G1 + G3 + measured D0G2 units) + predicted D0G2 <= BUDGET x
    (1 - MARGIN) (decision D0-G2); units in done (record on disk) cost 0 in the prefix."""
    cap = BUDGET_CORE_H * (1.0 - MARGIN)
    d0_g1g3 = float(d0_fixed)
    room = cap - float(d1_total) - float(d0_g1g3)
    done = {int(x) for x in done}
    costs = [0.0 if int(u["rxn_id"]) in done else float(ratio) * pw(gfit, u["n_atoms"]) for u in ranked]
    n, cum = 0, 0.0
    for c in costs:
        if cum + c > room:
            break
        cum += c
        n += 1
    per = _stats(costs)
    return n, dict(cap_with_margin=cap, margin=MARGIN, d1_total=float(d1_total), d0_g1_g3=float(d0_g1g3),
                   room_for_g2=room, margin_met_without_g2=room >= 0, n_g2=n, g2_core_h=cum,
                   g2_core_h_per_rxn=per, g2_ratio=float(ratio),
                   total_with_g2=float(d1_total) + float(d0_g1g3) + cum,
                   cap_met=float(d1_total) + float(d0_g1g3) + cum <= BUDGET_CORE_H,
                   n_g2_at_cap_no_margin=_prefix(costs, BUDGET_CORE_H - float(d1_total) - float(d0_g1g3)))


def _prefix(costs, room):
    n, cum = 0, 0.0
    for c in costs:
        if cum + c > room:
            break
        cum += c
        n += 1
    return n


def fingerprint_d0(limit, n_near=2):
    files = [D0_LABELS, D0_SMILES, D0_Q, D0_FOLDS, M.LABELS, M.S_FOLDS, D1_MANIFEST]
    fp = dict(code=CODE_D0, files={str(p): M.sha256(p) for p in files}, limit=limit, seed=M.SEED, seed_off=D0_SEED_OFF,
              roots=dict(prof=str(D0_PROF), eda=str(D0_EDA), g2c=str(D0_G2_FINAL)), d1val=d1val_ids())
    if int(n_near) != 2:
        fp["n_near"] = int(n_near)
    return fp


def plan_stats_d0(df):
    st = dict(n_rows=int(len(df)), n_engrad_structures=int(df.group.isin(["D0G1", "D0G3"]).sum()),
              n_g2_reactions=int((df.group == "D0G2").sum()),
              by_group={g: int(n) for g, n in df.groupby("group").size().items()},
              by_group_kind={f"{g}/{k}": int(n) for (g, k), n in df.groupby(["group", "kind"]).size().items()},
              n_atoms={f"{g}/{k}": _stats(x.n_atoms.tolist()) for (g, k), x in df.groupby(["group", "kind"])},
              by_charge={str(k): int(n) for k, n in df.charge.value_counts().sort_index().items()},
              with_d1_rxn_ids={f"{g}/{k}": int((x.rxn_ids != "").sum()) for (g, k), x in df.groupby(["group", "kind"])})
    folds = {}
    for k in (-1, 0, 1, 2, 3, 4):
        tagged = df.folds.map(lambda s, k=k: str(k) in str(s).split(","))
        folds[str(k)] = {f"{g}/{kk}": int(n) for (g, kk), n in df[tagged].groupby(["group", "kind"]).size().items()}
    st["fold_tagged"] = folds
    dips = Counter()
    for s in df.dipoles:
        for x in str(s).split(","):
            if x:
                dips[x] += 1
    st["dipole_tag_counts"] = dict(dips)
    return st


def _int_arg(v, name):
    try:
        return int(v)
    except (TypeError, ValueError):
        M.stop(f"plan_d0: {name} {v!r} is not an integer")


def resolve_n_g2(n_g2, fit_src, n_hold=0):
    """('auto' | int) -> (N or None (= the budget rule on the plan fits), source string); '+holdout:H' when the plan
    has H evaluation reactions (decision D0-hold). A bad value is a STOP (exit 3: a retry cannot fix it)."""
    if str(n_g2) != "auto":
        n = _int_arg(n_g2, "--n-g2")
        if n < 0:
            M.stop("plan_d0: --n-g2 must be >= 0 or auto")
        s = f"explicit:{n}"
    else:
        n, s = None, f"rule:{fit_src}"
    return n, s + (f"+holdout:{int(n_hold)}" if n_hold else "")


def resolve_pin(old, n_g2, pin):
    """decision D0-pin: (n_g2 to use, pinned K or None). pin True = the gen_d0 CLI pins an explicit K (auto keeps the
    recorded pin), False = unpin (gen_d0 --unpin), None = gen_dft CLI / launcher: keep the recorded pin (auto -> K, the
    same K -> K, another K -> STOP exit 3)."""
    old_pin = (old or {}).get("pinned_n_g2")
    old_pin = None if old_pin is None else int(old_pin)
    auto = str(n_g2) == "auto"
    if pin is False:
        return n_g2, None
    if pin is True:
        if auto:
            return (old_pin if old_pin is not None else n_g2), old_pin
        return n_g2, _int_arg(n_g2, "--n-g2")
    if old_pin is None:
        return n_g2, None
    if auto:
        return old_pin, old_pin
    if _int_arg(n_g2, "--n-g2") != old_pin:
        M.stop(f"plan_d0 --n-g2 {n_g2}: the D0G2 training count is pinned at {old_pin} (user decision D0-plus, "
               f"`gen_d0.py plan_d0 --n-g2 {old_pin}`); pass --n-g2 {old_pin} (launcher: `param G-D0f.n_g2={old_pin}`) "
               f"or re-pin with `gen_d0.py plan_d0 --n-g2 K` (decision D0-pin)")
    return n_g2, old_pin


def cmd_plan_d0(limit=None, n_g2="auto", force=False, n_near=2, n_holdout=None, replace_holdout=False, pin=None):
    """n_near (--n-near 1|2, default 2 = the D1 G1 rule): d0_ts_near rows per D0 reaction (budget knob; 1 drops n2;
    may change after D0 records exist, decision D0-nnear). N(D0G2) = the budget rule on the plan fits (D0-budget) or
    an explicit / pinned --n-g2 (cap check D0-cap, pin D0-pin). n_holdout (--n-holdout H, D0-plus): H evaluation-only
    D0 reactions + their reactant-pair siblings (decisions D0-hold, D0-sib, D0-gauss); None = keep the existing list.
    replace_holdout (--replace-holdout) = the only way to change an existing list (decision D0-hold-fix). force = re-plan
    even if up to date and skip the in-flight guard (it never touches the holdout or the pin)."""
    import pandas as pd
    if _int_arg(n_near, "--n-near") not in (1, 2):
        M.stop("plan_d0: --n-near must be 1 or 2")
    n_near = int(n_near)
    if n_holdout is not None and _int_arg(n_holdout, "--n-holdout") < 0:
        M.stop("plan_d0: --n-holdout must be >= 0")
    fp = fingerprint_d0(limit, n_near)
    old = G._rj(G.PLAN_D0_META)
    # ---- holdout file (decision D0-hold-fix): content-checked before anything else
    H_old = read_holdout()
    cur_sha, hrec, hprob = holdout_integrity(old or {})
    n_eval_old = sum(r["role"] == "eval" for r in H_old) if H_old is not None else 0
    want = n_holdout is not None and int(n_holdout) != n_eval_old
    replace = bool(replace_holdout and n_holdout is not None)
    if hprob and not replace:
        M.stop(f"plan_d0: {hprob}; not re-planned (decision D0-hold-fix: restore the file, or `gen_d0.py plan_d0 --n-g2 "
               f"K --n-holdout H --replace-holdout` while no FT data were built with a holdout / D0 inputs)")
    if want and H_old is not None and not replace:
        M.stop(f"plan_d0: {HOLD_CSV} ({n_eval_old} evaluation reactions) is fixed once written; --n-holdout {n_holdout} "
               f"needs --replace-holdout (gen_d0 CLI; decision D0-hold-fix)")
    new_hold = want or replace
    k_hold = int(n_holdout) if new_hold else n_eval_old
    e1 = d1_g0()
    basis = budget_basis(e1)
    efit, xfit, g2r, fit_src = plan_fits(fp, e1, basis)
    n_g2_eff, new_pin = resolve_pin(old, n_g2, pin)
    n_fix, src = resolve_n_g2(n_g2_eff, fit_src, k_hold)
    need = [G.PLAN_D0_PQ, G.G2_PLAN_D0, G.G2_POOL_D0, G.D0_CLASS_CSV] + ([HOLD_CSV, HOLD_REF] if k_hold else [])
    have = all(p.is_file() for p in need)
    hold_same = not new_hold and ((not k_hold) or (cur_sha is not None and (old or {}).get("holdout_sha256") == cur_sha))
    if old is not None and have and hold_same and old.get("fingerprint") == fp and old.get("n_g2_source") == src \
            and old.get("pinned_n_g2") == new_pin and not force:
        sync_holdout_copy()
        print(f"[plan_d0] up to date ({G.PLAN_D0_PQ}, D0G2 N {old.get('n_g2')} from {src}"
              f"{f' (pinned {new_pin})' if new_pin is not None else ''}, holdout {old.get('n_holdout', 0)} evaluation "
              f"reactions)", flush=True)
        return 0
    if old is not None and any_records_d0() and old.get("fingerprint") != fp:
        if fp_core(old.get("fingerprint")) != fp_core(fp):
            M.stop(f"D0 plan inputs changed after D0 records were written ({old.get('fingerprint')} -> {fp}); not "
                   f"re-planned")
        print(f"[plan_d0] --n-near {(old.get('fingerprint') or {}).get('n_near', 2)} -> {n_near} with D0 records on "
              f"disk: allowed (decision D0-nnear: the n1 / i rows are identical)", flush=True)
    t0 = time.time()
    nproc = G._cpus()
    # ---- D1 side (decision D0-ident: every comparison by molecule / reaction identity)
    T = M.targets()
    Ti = T.set_index("rxn_id")
    labs1 = M.read_json(M.LABELS)
    id_rxns, raw_rxns, dip_of_id, bad_id1 = defaultdict(set), defaultdict(set), defaultdict(set), []
    for d in labs1:
        cr = d.get("cr") or {}
        for k in (1, 2):
            key = cr.get(f"key{k}")
            if key:
                s = smi_part(key)
                x = mol_ident(s)
                if x is None:
                    bad_id1.append(key)
                    continue
                id_rxns[x].add(int(d["rxn_id"]))
                raw_rxns[s].add(int(d["rxn_id"]))
                if key.startswith("dipole:"):
                    dip_of_id[x].add(str(d["dipole"]))
    canon1, ident1 = defaultdict(set), defaultdict(set)
    for d in labs1:
        c = canon_rxn(d["rxn_smiles"])
        if c:
            canon1[c].add(int(d["rxn_id"]))
        y = rxn_ident(d["rxn_smiles"])
        if y is None:
            bad_id1.append(d["rxn_smiles"])
        else:
            ident1[y].add(int(d["rxn_id"]))
    if bad_id1:
        M.stop(f"D1 molecules / reactions without an identity (decision D0-ident): {len(bad_id1)}; first {bad_id1[:3]}")
    man = pd.read_csv(D1_MANIFEST, usecols=["kind", "mapped_smiles", "dipole", "meta_rxn_id"], keep_default_na=False)
    man = man[man.kind == "D1"]
    mapped1 = {int(r): s for r, s in zip(man.meta_rxn_id, man.mapped_smiles)}
    # ---- classifier check on every D1 target (decision D0-class)
    bad1, nmiss1 = [], 0
    for t in T.itertuples(index=False):
        s = mapped1.get(int(t.rxn_id))
        if s is None:
            nmiss1 += 1
            continue
        c, inf = _classify_safe(s)
        if c != t.dipole:
            bad1.append(dict(rxn_id=int(t.rxn_id), dipole=t.dipole, got=c, **inf))
    if bad1 or nmiss1:
        M.stop(f"D0 dipole classifier does not reproduce the D1 dipoles: {len(bad1)} mismatches, {nmiss1} D1 targets "
               f"without a manifest mapped SMILES; first {bad1[:3]}")
    # ---- D0 side
    L = {int(d["rxn_id"]): d for d in M.read_json(D0_LABELS)}
    ok = sorted(r for r, d in L.items() if d.get("status") == "ok")
    S = pd.read_csv(D0_SMILES, usecols=["rxn_id", "rxn_smiles"])
    smi0 = dict(zip(S.rxn_id.astype(int), S.rxn_smiles))
    Qk = {int(d["rxn_id"]): (d.get("cr") or {}) for d in M.read_json(D0_Q)}
    cls, cinfo = {}, {}
    for r in ok:
        cls[r], cinfo[r] = _classify_safe(smi0.get(r, ""))
    bad0, mid0 = [], {}
    for r in ok:
        mid0[r] = tuple(mol_ident(smi_part(Qk.get(r, {}).get(f"key{k}", "x:"))) for k in (1, 2))
        s, x = smi_part(Qk.get(r, {}).get("key1", "x:")), mid0[r][0]
        if x in dip_of_id and dip_of_id[x] != {cls[r]}:
            bad0.append(dict(d0_id=r, smiles=s, d1=sorted(dip_of_id[x]), got=cls[r]))
    if bad0:
        M.stop(f"D0 class of a shared dipole molecule != its D1 dipole: {len(bad0)}; first {bad0[:3]}")
    roles = Counter(f"{L[r].get('role1')}/{L[r].get('role2')}" for r in ok)
    keykinds = Counter(f"{str(Qk.get(r, {}).get('key1', '')).split(':')[0]}/{str(Qk.get(r, {}).get('key2', '')).split(':')[0]}"
                       for r in ok)
    if set(roles) != {"dipole/dipolarophile"} or set(keykinds) != {"dipole/dipolarophile"}:
        M.stop(f"D0 frag1 is not always the dipole: roles {dict(roles)}, Q key kinds {dict(keykinds)}")
    # ---- overlap with D1 targets (decision D0-scope: (a) canon string, (b) reaction identity, (c) d0_folds.csv)
    ov, ov_id, nofail = {}, {}, []
    for r in ok:
        s = smi0.get(r)
        y = rxn_ident(s) if s else None
        if y is None:
            nofail.append(r)
            continue
        c = canon_rxn(s)
        if c and c in canon1:
            ov[r] = sorted(canon1[c])
        if y in ident1:
            ov_id[r] = sorted(ident1[y])
    if nofail:
        M.stop(f"D0 reactions without a reaction identity (decision D0-scope): {len(nofail)}; first {nofail[:5]}")
    F = pd.read_csv(D0_FOLDS, keep_default_na=False)
    ov_folds = {int(r) for r, o in zip(F.rxn_id, F.overlap) if str(o) == "True"}
    skip = set(ov) | set(ov_id) | ov_folds
    d1val = set(d1val_ids())
    if skip & d1val:
        M.stop(f"d1_validation D0 reactions {sorted(skip & d1val)} are the same reaction as D1 targets: their Phase I "
               f"tags must be checked first (decision D0-scope)")
    rids = [r for r in ok if r not in skip]
    if limit:
        rids = rids[:int(limit)]

    # ---- per reaction (process pool)
    def ftags(r, k):
        x, raw = mid0[r][k - 1], smi_part(Qk.get(r, {}).get(f"key{k}", "x:")) in raw_rxns
        if x is not None and x in id_rxns:
            rx = sorted(y for y in id_rxns[x] if y in Ti.index)
            return True, raw, dict(rxn_ids=M.tag_list(rx), folds=M.tag_list(Ti.loc[rx, "fold_T1"].tolist()),
                                   dipoles=M.tag_list(Ti.loc[rx, "dipole"].tolist() + [cls[r]]))
        return False, raw, dict(rxn_ids="", folds="", dipoles=cls[r])

    jobs, n_frag_raw = [], 0
    for r in rids:
        d = L[r]
        sh, raws, tg = zip(*(ftags(r, k) for k in (1, 2)))
        n_frag_raw += sum(raws)
        jobs.append(dict(d0_id=r, cls=cls[r], ts=str(D0_PROF / str(r) / str(d["ts_file"])),
                         eda=str(D0_EDA / f"rxn_{r:04d}" / "eda.inp"),
                         refs=[str(D0_PROF / str(r) / str(d[f"rel{k}_file"])) for k in (1, 2)],
                         n_atoms=int(d["n_atoms"]), n_f1=int(d["n_f1"]), n_f2=int(d["n_f2"]),
                         charge1=int(d.get("charge1") or 0), charge2=int(d.get("charge2") or 0),
                         pairs=[list(p) for p in M.formed_pairs(d["formed_pairs_ts"])], n_near=int(n_near),
                         frag_tags={"1": tg[0], "2": tg[1]}, frag_shared={"1": sh[0], "2": sh[1]}))
    with ProcessPoolExecutor(max(1, nproc)) as ex:
        res = list(ex.map(rxn_d0_job, jobs, chunksize=8))
    rows = [x for r in res for x in r["rows"]]
    g1_err = {str(r["d0_id"]): r["g1_err"] for r in res if r["g1_err"]}
    fr_err = {str(r["d0_id"]): r["frag_err"] for r in res if r["frag_err"]}
    chk = [r["chk"] for r in res]
    refsyms = {r["d0_id"]: r["ref_syms"] for r in res}
    ts_syms = {r["d0_id"]: r["ts_syms"] for r in res}
    mrows, mstats, _ = molecules_d0(rids, L, Qk, cls, refsyms, set(id_rxns), set(raw_rxns))
    # ---- D0G2 pool, budget (decision D0-budget), N
    ranked, pinfo = g2_pool_d0(rids, L, cls, ts_syms, nproc)
    P0 = pd.DataFrame(rows + mrows)
    pred, n2_part = cost_rows(P0, efit, xfit, basis["engrad_slowdown"])
    d0_g1g3 = float(sum(pred.values()))
    d2 = G.RUNS / "D0G2"
    done = {int(p.parent.name): (G._rj(p) or {}) for p in d2.glob("*/unit.json")} if d2.is_dir() else {}
    done_ch = float(sum(float(u.get("core_h") or 0.0) for u in done.values()))
    # ---- D0-plus holdout (decisions D0-hold, D0-sib, D0-hold-fix): kept as written unless asked (guards above)
    pool_ids = {int(u["rxn_id"]) for u in ranked}
    pair0 = {r: pair_ident(smi0[r]) for r in rids}
    if any(v is None for v in pair0.values()):
        M.stop("D0 reactions without a reactant-pair identity (decision D0-sib)")
    by_pair = defaultdict(list)
    for r in rids:
        by_pair[pair0[r]].append(r)
    d1_pairs = {y.split(">>")[0] for y in ident1}
    des_rx, des_pairs, des_ids = d1design_ids()
    block = {p: "d1_target_pair" for p in by_pair if p in d1_pairs}
    block.update({p: "d1_design_pair" for p in by_pair if p in des_pairs and p not in block})
    sib_chk = dict(n_pair_groups=len(by_pair), group_size=dict(sorted(Counter(len(v) for v in by_pair.values()).items())),
                   n_planned_with_d1_target_pair=sum(pair0[r] in d1_pairs for r in rids),
                   d1design=dict(n_ts_ids=len(des_ids), ts_ids=des_ids,
                                 n_planned_same_reaction=sum(rxn_ident(smi0[r]) in des_rx for r in rids),
                                 n_planned_same_pair=sum(pair0[r] in des_pairs for r in rids)))
    gctx = dict(pair0=pair0, by_pair=by_pair, cls={r: cls[r] for r in rids}, natoms={r: int(L[r]["n_atoms"]) for r in rids},
                block=block, d1val=d1val)
    hinfo, train0 = None, None
    if not new_hold:
        hold_rows, hold_change = list(H_old or []), False
    else:
        if n_fix is None:
            M.stop("a new or replaced D0 holdout needs an explicit or pinned --n-g2 (the training prefix whose "
                   "reactant-pair groups it avoids; decision D0-hold-fix)")
        train0 = [int(u["rxn_id"]) for u in ranked[:max(0, min(int(n_fix), len(ranked)))]]
        hold_rows, hinfo = holdout_pick(ranked, set(train0), k_hold, gctx) if k_hold else ([], None)
        key = [(int(h["d0_id"]), h["d0_class"], h["role"]) for h in hold_rows]
        hold_change = (key != [(int(h["d0_id"]), h["d0_class"], h["role"]) for h in H_old]) if H_old is not None \
            else bool(hold_rows)
    Hall = {int(h["d0_id"]) for h in hold_rows}
    Heval = {int(h["d0_id"]) for h in hold_rows if h["role"] == "eval"}
    hold_cls = {int(h["d0_id"]): h["d0_class"] for h in hold_rows}
    hviol = [f"eval {r} not in the G2c pool" for r in sorted(Heval - pool_ids)]
    hviol += [f"{r} not planned" for r in sorted(Hall - set(rids))]
    hviol += [f"{r} class {hold_cls[r]} != {cls[r]}" for r in sorted(Hall & set(rids)) if hold_cls[r] != cls[r]]
    hviol += [f"{r} {block[pair0[r]]}" for r in sorted(Hall & set(rids)) if pair0[r] in block]
    hviol += [f"{r} d1_validation" for r in sorted(Hall & d1val)]
    hviol += [f"sibling {m} of {r} not listed" for r in sorted(Hall & set(rids)) for m in by_pair[pair0[r]]
              if m not in Hall]                                     # D0-sib closure
    if hviol:
        M.stop(f"D0 holdout list checks: {len(hviol)} violations; first {hviol[:5]} (decisions D0-hold, D0-sib)")
    rest = [u for u in ranked if int(u["rxn_id"]) not in Hall]
    eval_units = [u for u in ranked if int(u["rxn_id"]) in Heval]

    def ucost(u):
        return 0.0 if int(u["rxn_id"]) in done else float(g2r) * pw(e1["g2_fit"], u["n_atoms"])

    n_rule, bud = n_g2_rule(basis["d1_total"], d0_g1g3 + done_ch, rest, e1["g2_fit"], g2r, done=set(done))
    bud["d0_fixed_parts"] = dict(note="rule key d0_g1_g3 = the fixed D0 part = the sum of these; the holdout is not "
                                      "refined (D0-gauss)", d0_g1_g3=d0_g1g3, d0g2_measured_on_disk=done_ch)
    n_sel = n_rule if n_fix is None else n_fix
    n_sel = max(0, min(int(n_sel), len(rest)))
    train = [dict(u, role="train") for u in rest[:n_sel]]
    if train0 is not None and k_hold and [int(u["rxn_id"]) for u in train] != train0:
        M.stop("internal: the training prefix changed when the new holdout was removed from the ranking (D0-plus)")
    sel = train                                                     # D0-gauss: the holdout is never run by run_g2
    train_ch = float(sum(ucost(u) for u in train))
    total = float(basis["d1_total"]) + d0_g1g3 + done_ch + train_ch + G4_RESERVE_CORE_H
    cap_chk = dict(rule=DECISIONS_D0["D0-cap"], explicit=n_fix is not None, pinned=new_pin,
                   d1_total=float(basis["d1_total"]), d0_g1_g3=d0_g1g3, d0g2_measured_on_disk=done_ch,
                   d0g2_train_pred=train_ch, d0g2_holdout_pred=0.0, g4_reserve=G4_RESERVE_CORE_H, total=total,
                   cap=BUDGET_CORE_H, over_cap=total > BUDGET_CORE_H, headroom=BUDGET_CORE_H - total,
                   n_train=len(train), n_holdout=len(Heval), n_holdout_listed=len(Hall),
                   per_rxn_train=_stats([ucost(u) for u in train]), d0g2_pred_total=train_ch,
                   suggested_d0g2_tasks=(max(1, int(math.ceil(train_ch / (64 * TASK_H)))) if sel and train_ch > 0 else 0),
                   task_rule=f"ceil(predicted D0G2 core-h / (64 cores x {TASK_H:g} h)); 64-core cpu2 tasks = 8 OptTS "
                             f"x 8 ranks")
    if n_fix is not None and total > BUDGET_CORE_H:
        M.stop(f"explicit D0 selection ({len(train)} training D0G2 reactions; holdout {len(Heval)} not refined): "
               f"predicted G total {total:,.0f} core-h > cap {BUDGET_CORE_H:,.0f} (D1 {basis['d1_total']:,.0f} + D0 "
               f"G1/G3 {d0_g1g3:,.0f} + D0G2 on disk {done_ch:,.0f} + D0G2 {train_ch:,.0f} + G-4 "
               f"{G4_RESERVE_CORE_H:,.0f}); nothing written (decision D0-cap)")
    old_sel = [int(u["rxn_id"]) for u in (G._rj(G.G2_PLAN_D0) or {}).get("rxns") or []]
    if old_sel and old_sel != [int(u["rxn_id"]) for u in sel] and not force:
        infl = [r for r in old_sel if (d2 / str(r) / "starts.json").is_file() and not (d2 / str(r) / "unit.json").is_file()]
        if infl:
            M.not_ready(f"the D0G2 selection would change while units {infl[:10]} have an OptTS start without unit.json "
                        f"(possibly running); re-plan after they finish, or --force (decision D0-G2)")
    if hold_change or (replace and hprob):
        built = [b for b in ft_data_built() if b["holdout_sha256"] or b["d0_inputs"]]
        if built:
            M.stop(f"the D0 holdout list would change ({'none' if H_old is None else n_eval_old} -> {k_hold} evaluation "
                   f"reactions) but FT data were already built with a holdout or with D0 inputs ({len(built)}: "
                   f"{[b['path'] for b in built[:3]]}); refused (decision D0-hold-fix)")
    g2rows = []
    for u in sel:
        syms, pos = M.read_xyz(u["start"])
        g2rows.append(G.row(u["struct_id"], "D0G2", "d0_g2_path", syms, pos, dict(rxn_ids="", folds="", dipoles=u["dipoles"]),
                            u["start"], u["charge"], hess_src=u["hess"], d0_id=u["d0_id"], d0_class=u["d0_class"], fold_T1=-9,
                            disp=json.dumps(dict(hess_sha256=u["hess_sha256"], rank=u["rank"], role=u["role"]))))
    df = pd.DataFrame(rows + mrows + g2rows)
    if df.struct_id.duplicated().any():
        M.stop(f"duplicate struct_id in the D0 plan: {df.struct_id[df.struct_id.duplicated()].tolist()[:5]}")
    # ---- identity guard (decisions D0-scope / D0-ident / D0-tags): nothing planned may still be a D1 reaction, a
    #      D1-shared fragment without the D1 tags, a D1 molecule, or miss its own D0 class in the dipoles tag
    viol = [f"rxn {r}" for r in rids if rxn_ident(smi0[r]) in ident1]
    for j in jobs:
        for k in ("1", "2"):
            if (mid0[j["d0_id"]][int(k) - 1] in id_rxns) != bool(j["frag_tags"][k]["rxn_ids"]):
                viol.append(f"frag {j['d0_id']}/{k}")
    viol += [f"mol {x['struct_id']}" for x in mrows if json.loads(x["disp"])["ident"] in id_rxns]
    viol += [f"class {x.struct_id}" for x in df.itertuples(index=False)
             if not set(str(x.d0_class).split(",")) <= set(str(x.dipoles).split(","))]
    if viol:
        M.stop(f"D0 identity / tag guard: {len(viol)} violations; first {viol[:5]}")
    # ---- write
    rset = set(rids)
    C = pd.DataFrame([dict(d0_id=r, d0_class=cls[r], dipole_smiles=smi_part(Qk.get(r, {}).get("key1", "x:")),
                           overlap=r in skip, overlap_d1=",".join(str(x) for x in ov.get(r, [])),
                           overlap_d1_ident=",".join(str(x) for x in ov_id.get(r, [])), overlap_d0_folds=r in ov_folds,
                           d1val=r in d1val, planned=r in rset,
                           dipole_mol_n_d1_rxns=len(id_rxns.get(mid0[r][0], ())),
                           dipolarophile_mol_n_d1_rxns=len(id_rxns.get(mid0[r][1], ())), **cinfo[r]) for r in ok])
    if hold_change:
        write_holdout(hold_rows)                         # decisions D0-hold / D0-sib / D0-hold-fix (guards above)
    hold_sha = M.sha256(HOLD_CSV) if HOLD_CSV.is_file() else None
    if hold_sha is not None:
        sync_holdout_copy()
        write_holdout_ref(eval_units, hold_rows, hold_sha, pair0)
    sib_ids = sorted(Hall - Heval)
    hold_rec = dict(n=len(Heval), n_listed=len(Hall), n_siblings=len(sib_ids), sha256=hold_sha,
                    file=str(HOLD_CSV) if hold_sha else None, ref=str(HOLD_REF) if hold_sha else None,
                    rule=DECISIONS_D0["D0-hold"], sib_rule=DECISIONS_D0["D0-sib"], gauss=DECISIONS_D0["D0-gauss"],
                    refined=False, reference="coley_g16_ts", ids=sorted(Heval), sibling_ids=sib_ids,
                    n_groups=len({pair0[r] for r in Hall if r in pair0}),
                    by_class=dict(sorted(Counter(cls[r] for r in Heval).items())),
                    siblings_by_class=dict(sorted(Counter(cls[r] for r in sib_ids).items())))
    M.write_csv(G.D0_CLASS_CSV, C)
    M.write_parquet(G.PLAN_D0_PQ, df)
    M.write_json(G.G2_POOL_D0, dict(code=CODE_D0_REV, written=G._now(), fingerprint=fp, rule=DECISIONS_D0["D0-G2"],
                                    ranked=ranked))
    M.write_json(G.G2_PLAN_D0, dict(code=CODE_D0_REV, written=G._now(), fingerprint=fp, rule=DECISIONS_D0["D0-G2"],
                                    n_g2=n_sel, n_holdout=len(Heval), n_units=len(sel), n_g2_source=src,
                                    pinned_n_g2=new_pin, n_g2_rule=n_rule, fit_source=fit_src, holdout=hold_rec,
                                    rxns=sel))
    ccount = Counter(cls[r] for r in ok)

    def by_cls(us):
        return dict(sorted(Counter(u["d0_class"] for u in us).items()))

    planned_cls = Counter(cls[r] for r in rids)
    excl = {s: dict(planned=planned_cls[s], eval=sum(cls[r] == s for r in Heval),
                    sibling=sum(cls[r] == s for r in sib_ids),
                    excluded_frac=round(sum(cls[r] == s for r in Hall) / planned_cls[s], 3))
            for s in sorted(planned_cls)}
    sel_rep = dict(rule=[DECISIONS_D0["D0-plus"], DECISIONS_D0["D0-gauss"], DECISIONS_D0["D0-hold"],
                         DECISIONS_D0["D0-sib"]], n_train=len(train), n_holdout=len(Heval), n_siblings=len(sib_ids),
                   pinned_n_g2=new_pin, train_by_class=by_cls(train), holdout_by_class=by_cls(eval_units),
                   pool_by_class=by_cls(ranked), excluded_by_class=excl,
                   train_n_atoms=_stats([u["n_atoms"] for u in train]),
                   holdout_n_atoms=_stats([u["n_atoms"] for u in eval_units]),
                   pool_n_atoms=_stats([u["n_atoms"] for u in ranked]),
                   holdout_allocation=({k: v for k, v in hinfo.items() if k != "groups"} if hinfo else None),
                   holdout_groups=([{k: v for k, v in x.items() if k != "pair"} for x in hinfo["groups"]] if hinfo
                                   else None),
                   sibling_checks=sib_chk, holdout_changed=bool(hold_change), holdout=hold_rec)
    rep = dict(phase="G plan D0", code=CODE_D0_REV, written=G._now(), host=socket.gethostname(), wall_s=time.time() - t0,
               fingerprint=fp, limit=limit, decisions=DECISIONS_D0,
               d0=dict(n_records=len(L), n_ok=len(ok), n_overlap_skipped=len(skip), overlap_canon_vs_d1_targets=ov,
                       overlap_ident_vs_d1_targets=ov_id, overlap_d0_folds=sorted(ov_folds),
                       overlap_only_in_d0_folds=sorted(ov_folds - set(ov)), overlap_only_canon=sorted(set(ov) - ov_folds),
                       overlap_ident_not_canon_or_folds=sorted(set(ov_id) - set(ov) - ov_folds),
                       overlap_canon_not_ident=sorted(set(ov) - set(ov_id)), n_planned_reactions=len(rids),
                       n_d1val_reactions=len(d1val), d1val_ids=sorted(d1val)),
               classes=dict(by_class_ok=dict(ccount), by_class_planned=dict(Counter(cls[r] for r in rids)),
                            n_d1_class_ok=sum(v for k, v in ccount.items() if k in D1_CLASSES),
                            n_unclassified=ccount.get("d0_unclassified", 0),
                            n_aromatic=sum(bool(cinfo[r].get("aromatic")) for r in ok),
                            aromatic_by_class=dict(Counter(cls[r] for r in ok if cinfo[r].get("aromatic"))),
                            d1_reproduced=dict(n=int(len(T)), mismatches=len(bad1)),
                            shared_dipole_molecules=dict(n_identities=len({mid0[r][0] for r in ok} & set(dip_of_id)),
                                                         n_reactions_planned=sum(mid0[r][0] in dip_of_id for r in rids),
                                                         mismatches=len(bad0))),
               counts=plan_stats_d0(df),
               excluded=dict(n_g1=len(g1_err), g1=dict(list(g1_err.items())[:50]), n_frag=len(fr_err),
                             frag=dict(list(fr_err.items())[:50])),
               checks=dict(frag_partition=dict(
                   n_first_nf1_is_frag1=sum(bool(c.get("first_nf1_is_frag1")) for c in chk),
                   n_pairs_cross_fragments=sum(bool(c.get("pairs_cross_fragments")) for c in chk),
                   n_frag1_ref_formula_equal=sum(bool(c.get("frag1_ref_formula_equal")) for c in chk),
                   n_frag2_ref_formula_equal=sum(bool(c.get("frag2_ref_formula_equal")) for c in chk)),
                   n_ref_read_errors=sum("ref_err" in c for c in chk),
                   freq_dev_cm="computed in the run step (records disp.freq_dev_cm; G_report_d0.json)",
                   n_frag_shared_d1=dict(f1=sum(bool(j["frag_shared"]["1"]) for j in jobs),
                                         f2=sum(bool(j["frag_shared"]["2"]) for j in jobs), string_rule=n_frag_raw),
                   identity_guard=dict(violations=0, rule="planned reactions / fragments / molecules vs D1 identities, "
                                                          "d0_class within dipoles")),
               molecules=mstats, g2_pool=pinfo,
               budget=dict(d1_g0=str(D1_G0), basis={k: v for k, v in basis.items() if k != "d1_observed"},
                           d1_total=basis["d1_total"], fits=dict(engrad=efit, xtb=xfit, g2_ratio=g2r, source=fit_src),
                           d1_engrad_fit=e1["engrad_fit"], d1_g2_fit=e1["g2_fit"], d0_predicted_by_item=pred,
                           d0_g1_g3=d0_g1g3, d0_n2_rows_share=n2_part, d0g2_units_on_disk=sorted(done),
                           d0g2_measured_core_h=done_ch, rule=bud, n_g2_selected=n_sel, n_g2_source=src,
                           n_holdout=len(Heval), d0_g2_selected_core_h=train_ch, cap_check=cap_chk,
                           d1_observed=basis["d1_observed"]),
               d0g2_selection=sel_rep,
               outputs={str(p): M.sha256(p) for p in (G.PLAN_D0_PQ, G.G2_PLAN_D0, G.G2_POOL_D0, G.D0_CLASS_CSV)
                        + ((HOLD_CSV, HOLD_REF) if hold_sha else ())})
    M.write_json(G.PLAN_D0_META, dict(fingerprint=fp, written=G._now(), n_rows=int(len(df)), code=CODE_D0_REV, n_g2=n_sel,
                                      n_g2_source=src, pinned_n_g2=new_pin, n_holdout=len(Heval),
                                      n_holdout_listed=len(Hall), holdout_sha256=hold_sha))
    M.write_json(G.G_PLAN_D0_JSON, rep)
    print(json.dumps(dict(by_group_kind=rep["counts"]["by_group_kind"], n_planned=len(rids), n_overlap_skipped=len(skip),
                          overlap_ident_only=rep["d0"]["overlap_ident_not_canon_or_folds"],
                          molecules=dict(new=mstats["n_new"], shared_skipped=mstats["n_shared_with_d1_skipped"],
                                         identity_only=mstats["shared_identity_only"]),
                          frag_shared_d1=rep["checks"]["n_frag_shared_d1"],
                          excluded_g1=len(g1_err), excluded_frag=len(fr_err), classes=rep["classes"]["by_class_planned"],
                          g2_pool=pinfo["n_pool"], n_g2=n_sel, n_g2_source=src, pinned_n_g2=new_pin,
                          d0g2_selection={k: sel_rep[k] for k in ("n_train", "n_holdout", "n_siblings", "train_by_class",
                                                                  "holdout_by_class", "excluded_by_class",
                                                                  "holdout_changed", "sibling_checks")},
                          holdout_sha256=hold_sha,
                          cap_check={k: cap_chk[k] for k in ("explicit", "pinned", "d1_total", "d0_g1_g3",
                                                             "d0g2_measured_on_disk", "d0g2_train_pred",
                                                             "d0g2_holdout_pred", "g4_reserve", "total", "cap",
                                                             "headroom", "over_cap", "suggested_d0g2_tasks")},
                          budget=dict({k: bud[k] for k in ("d1_total", "d0_g1_g3", "room_for_g2", "margin_met_without_g2",
                                                           "cap_met", "n_g2_at_cap_no_margin")},
                                      slowdown=basis["engrad_slowdown"], d1_g0_total=basis["d1_g0_total"],
                                      d1_projected=basis["d1_projected_total"], n2_share=n2_part, fits=fit_src),
                          wall_s=round(rep["wall_s"], 1)), indent=1, default=str), flush=True)
    return 0


# ================================================================ G-0 (D0)
def label_row_d0(x):
    """undisplaced D0 label TS (parity) from a d0_ts_near plan row (its pos_A is the undisplaced label TS)."""
    d = json.loads(x["disp"])
    syms = str(x["symbols"]).split()
    pos = np.asarray(x["pos_A"], float).reshape(-1, 3)
    rid = int(x["d0_id"])
    return G.row(f"d0g1_{rid}_lab", "D0G1", "d0_ts_label", syms, pos, dict(rxn_ids=x["rxn_ids"], folds=x["folds"],
                                                                            dipoles=x["dipoles"]),
                 x["src"], int(x["charge"]), d0_id=rid, d0_class=x["d0_class"], fold_T1=-9,
                 disp=json.dumps(dict(undisplaced=True, pairs=d.get("pairs"))))


def pick_pilots_d0(ranked, k):
    if k <= 0 or not ranked:
        return []
    med = float(np.median([u["n_atoms"] for u in ranked]))
    return [int(u["rxn_id"]) for u in ranked if u["n_atoms"] <= med][:k]


def phase_i_parity():
    import pandas as pd
    if not I_SOURCES.is_file():
        return dict(n=0)
    S = pd.read_csv(I_SOURCES, usecols=["owner", "kinds", "fmax_last"], keep_default_na=False)
    S = S[S.kinds.astype(str).str.contains("d0_ts_grad")]
    v = [float(x) for x in S.fmax_last if str(x) != ""]
    return dict(n=len(v), median_max_abs_grad=float(np.median(v)) if v else None,
                min=float(np.min(v)) if v else None, max=float(np.max(v)) if v else None,
                source=f"{I_SOURCES} (d1_validation v2a L0 EnGrad at the Coley TS, kinds d0_ts_grad)")


def d0_records(groups=("D0G1", "D0G3")):
    out = []
    for g in groups:
        d = G.RUNS / g
        for f in (sorted(d.glob("*.json")) if d.is_dir() else []):
            r = G._rj(f)
            if r:
                out.append(r)
    return out


def g0_estimate_d0(P, meta, rows, pil, run_info):
    e1 = d1_g0()
    recs = d0_records()
    ok = [r for r in recs if r.get("status") == "ok" and r.get("core_h")]
    fails = [dict(struct_id=r["struct_id"], reason=r.get("reason")) for r in recs if r.get("status") != "ok"]
    ratio_e = float(np.mean([r["core_h"] / pw(e1["engrad_fit"], r["n_atoms"]) for r in ok])) if ok else 1.0
    own = G.fit_power([r["n_atoms"] for r in ok], [r["core_h"] for r in ok]) if len(ok) >= FIT_MIN_D0 else None
    if own and own.get("b") is not None and own["n"] >= FIT_MIN_D0:
        efit, efit_src = own, f"D0 log-log fit ({own['n']} ok D0 EnGrads)"
    else:
        efit = dict(a=float(e1["engrad_fit"]["a"]) * ratio_e, b=float(e1["engrad_fit"]["b"]))
        efit_src = f"D1 G-0 fit x mean D0/D1-fit ratio {ratio_e:.3f} ({len(ok)} ok D0 EnGrads < {FIT_MIN_D0})"
    hj = [G._rj(p) for p in sorted(G.HESS_D0.glob("*.json"))] if G.HESS_D0.is_dir() else []
    hok = [h for h in hj if h and h.get("status") == "ok" and h.get("n_atoms") and h.get("core_h")]
    if len(hok) >= 3:
        xfit = dict(a=float(np.mean([h["core_h"] / h["n_atoms"] ** 2 for h in hok])), b=2.0)
        xsrc = f"mean core_h / N^2 over {len(hok)} D0 xTB Freqs"
    else:
        xfit, xsrc = dict(XTB_PRIOR), f"prior ({len(hok)} D0 xTB Freqs < 3)"
    basis = budget_basis(e1)
    slow = basis["engrad_slowdown"]
    pred, n2_part = cost_rows(P, efit, xfit, slow)
    lab = [r for r in ok if r["kind"] == "d0_ts_label"]
    pred["D0G1/d0_ts_label_measured"] = float(sum(r["core_h"] for r in lab))
    d0_g1g3 = float(sum(pred.values()))
    ranked = (G._rj(G.G2_POOL_D0) or {}).get("ranked") or []
    pp, ondisk_ch = [], 0.0
    d2 = G.RUNS / "D0G2"
    ondisk = {int(p.parent.name) for p in d2.glob("*/unit.json")} if d2.is_dir() else set()
    for rid in sorted(ondisk):
        u = G._rj(d2 / str(rid) / "unit.json") or {}
        ondisk_ch += float(u.get("core_h") or 0.0)
        if u.get("core_h") and u.get("n_atoms") and rid in {int(x["rxn_id"]) for x in ranked}:
            pp.append(dict(d0_id=rid, n_atoms=int(u["n_atoms"]), core_h=float(u["core_h"]), status=u.get("status"),
                           n_steps=u.get("n_steps"), fit_core_h=pw(e1["g2_fit"], u["n_atoms"])))
    g2_ratio = max(1.0, float(e1.get("g2_pilot_ratio") or 1.0), basis["g2_slowdown"],
                   float(np.mean([x["core_h"] / x["fit_core_h"] for x in pp])) if pp else 1.0)
    d1_total = basis["d1_total"]
    hset = holdout_ids()                         # D0-hold / D0-sib: listed reactions are never in the rule prefix
    rest = [u for u in ranked if int(u["rxn_id"]) not in hset]
    hold_ch = 0.0                                # D0-gauss: the holdout is not refined
    n_rule, bud = n_g2_rule(d1_total, d0_g1g3 + ondisk_ch + hold_ch, rest, e1["g2_fit"], g2_ratio, done=ondisk)
    g2_sel = hold_ch + float(sum(0.0 if int(u["rxn_id"]) in ondisk else g2_ratio * pw(e1["g2_fit"], u["n_atoms"])
                                 for u in rest[:n_rule]))
    by_group = {"D0G1": float(sum(v for k, v in pred.items() if k.startswith("D0G1/"))),
                "D0G3": float(sum(v for k, v in pred.items() if k.startswith("D0G3/"))), "D0G2": g2_sel + ondisk_ch}
    total = d1_total + float(sum(by_group.values()))
    n_rule1, bud1 = n_g2_rule(d1_total, d0_g1g3 - n2_part + ondisk_ch + hold_ch, rest, e1["g2_fit"], g2_ratio,
                              done=ondisk)
    g2_sel1 = hold_ch + float(sum(0.0 if int(u["rxn_id"]) in ondisk else g2_ratio * pw(e1["g2_fit"], u["n_atoms"])
                                  for u in rest[:n_rule1]))
    total1 = total - n2_part - g2_sel + g2_sel1
    alt1 = dict(note="the same estimate with --n-near 1 (plan_d0 --n-near 1; decision D0-nnear)", n2_rows_core_h=n2_part,
                total=total1, stop=total1 > BUDGET_CORE_H, n_g2_rule=n_rule1, d0g2_core_h=g2_sel1 + ondisk_ch,
                margin_met_without_g2=bud1["margin_met_without_g2"], applies=bool(n2_part > 0))
    mg = [float(r["max_abs_grad"]) for r in lab]
    d1med = ((G._rj(D1_G0) or {}).get("parity") or {}).get("median_max_abs_grad")
    par = dict(n=len(mg), median_max_abs_grad=float(np.median(mg)) if mg else None,
               max_max_abs_grad=float(np.max(mg)) if mg else None, min_max_abs_grad=float(np.min(mg)) if mg else None,
               per=[dict(struct_id=r["struct_id"], d0_id=r.get("d0_id"), n_atoms=r["n_atoms"],
                         max_abs_grad=r["max_abs_grad"], rms_grad=r.get("rms_grad")) for r in lab],
               d1_expect=G.PARITY_EXPECT, flag_above=G.PARITY_STOP, phase_i_d0_ts_grad=phase_i_parity(),
               d1_label_ts_median=d1med, d1_median_factor=PARITY_D1_X)
    par["off_stationary"] = bool(mg) and par["median_max_abs_grad"] > G.PARITY_STOP
    par["above_d1_expectation"] = bool(mg) and par["median_max_abs_grad"] > G.PARITY_EXPECT
    par["ratio_to_d1_median"] = par["median_max_abs_grad"] / float(d1med) if mg and d1med else None
    par["above_10x_d1_median"] = bool(mg and d1med) and par["median_max_abs_grad"] > PARITY_D1_X * float(d1med)
    par["note"] = ("D0 label TSs are Coley G16 TSs: above_10x_d1_median = not stationary at the ORCA label level "
                   "(program / level difference, decision D0-level); d0_ts_label / d0_ts_near / d0_ts_imag are still "
                   "label-level E/F data; recorded, never a stop")
    kinds = defaultdict(list)
    for r in ok:
        kinds[r["kind"]].append(r["core_h"])
    sugg = {g: max(1, int(math.ceil(by_group[g] / (64 * TASK_H)))) for g in ("D0G1", "D0G3")}
    sugg["D0G2"] = max(1, int(math.ceil(g2_sel / (64 * TASK_H)))) if n_rule and g2_sel > 0 else 0
    sugg["D0G1_if_n_near_1"] = max(1, int(math.ceil((by_group["D0G1"] - n2_part) / (64 * TASK_H))))
    return dict(phase="G-0 D0", code=CODE_D0, written=G._now(), host=socket.gethostname(), slurm_job=G._jobtag(),
                plan_fingerprint=meta.get("fingerprint"), decisions={k: DECISIONS_D0[k] for k in ("D0-g0", "D0-G2",
                                                                                                   "D0-level",
                                                                                                   "D0-budget",
                                                                                                   "D0-nnear")},
                budget_basis={k: v for k, v in basis.items() if k != "d1_observed"}, if_n_near_1=alt1,
                d0g2_units_on_disk=sorted(ondisk), d0g2_measured_core_h=ondisk_ch,
                n_planned_engrads=int(P.group.isin(["D0G1", "D0G3"]).sum()), n_pool_g2=len(ranked),
                n_engrad_ok=len(ok), engrad_failures=fails,
                per_engrad=sorted([dict(struct_id=r["struct_id"], kind=r["kind"], n_atoms=r["n_atoms"], core_h=r["core_h"],
                                        scf_cycles=r.get("scf_cycles"), max_abs_grad=r.get("max_abs_grad"),
                                        ratio_to_d1_fit=r["core_h"] / pw(e1["engrad_fit"], r["n_atoms"])) for r in ok],
                                  key=lambda x: x["n_atoms"]),
                per_kind_core_h={k: _stats(v) for k, v in kinds.items()}, d0_vs_d1_fit_ratio=ratio_e,
                engrad_fit_used=dict(efit, source=efit_src), engrad_fit_d0_own=own, engrad_fit_d1=e1["engrad_fit"],
                xtb_freq=dict(fit=xfit, source=xsrc, n=len(hok), core_h=_stats([h["core_h"] for h in hok])),
                g2_points_pilot=pp, g2_fit=e1["g2_fit"], g2_ratio=g2_ratio, g2_pilots=pil,
                predicted_core_h=dict(by_item=pred, by_group=by_group, d1_total=d1_total,
                                      d1_g0_total=basis["d1_g0_total"], d1_projected_total=basis["d1_projected_total"],
                                      engrad_slowdown=slow, total=total, budget=BUDGET_CORE_H),
                n_g2_rule=n_rule, budget_rule=bud, stop=total > BUDGET_CORE_H, parity=par,
                suggested_tasks=dict(rule=f"ceil(predicted core-h / (64 cores x {TASK_H:g} h)) per group, 64-core cpu2 "
                                          f"tasks (D0G2: 8 OptTS x 8 ranks per task)", **sugg),
                d1_observed=basis["d1_observed"], run_info=run_info)


def cmd_g0_d0(n_rand=G0_N_RAND, n_label=G0_N_LABEL, n_g2=G0_N_G2, label_rxns=None, max_atoms=None):
    if G.d0_stopped():
        return 3
    holdout_check("g0_d0")                       # decision D0-hold-fix
    P = G.load_plan(d0=True)
    meta = G._rj(G.PLAN_D0_META) or {}
    rng = np.random.default_rng([M.SEED, D0_SEED_OFF, 3])
    pool = P[P.group.isin(["D0G1", "D0G3"])].reset_index(drop=True)
    if max_atoms:
        pool = pool[pool.n_atoms <= int(max_atoms)].reset_index(drop=True)
    pick = []
    if n_rand > 0 and len(pool):
        nb = min(5, n_rand)
        rank = pool.n_atoms.rank(method="first").to_numpy()
        b = np.minimum(nb - 1, np.floor((rank - 1) * nb / len(pool)).astype(int))
        per = [n_rand // nb + (1 if k < n_rand % nb else 0) for k in range(nb)]
        for k in range(nb):
            idx = np.flatnonzero(b == k)
            pick += [int(x) for x in rng.choice(idx, size=min(per[k], len(idx)), replace=False)]
    sample = pool.iloc[sorted(pick)].to_dict("records")
    g1 = P[P.kind == "d0_ts_near"].drop_duplicates("d0_id")
    g1 = g1[~g1.d0_id.isin(d1val_ids())]
    if max_atoms:
        g1 = g1[g1.n_atoms <= int(max_atoms)]
    if label_rxns:
        lab = g1[g1.d0_id.isin(label_rxns)]
    else:
        o = g1.sort_values(["n_atoms", "d0_id"]).reset_index(drop=True)
        qs = np.linspace(0.05, 0.95, n_label) if n_label > 1 else np.array([0.5] * max(0, n_label))
        lab = o.iloc[sorted({int(round(q * (len(o) - 1))) for q in qs})] if len(o) and n_label else o.iloc[0:0]
    lrows = [label_row_d0(x) for x in lab.to_dict("records")]
    rows = sorted(sample + lrows, key=lambda r: (-int(r["n_atoms"]), r["struct_id"]))
    if str(n_g2) == "auto":                      # decision D0-g0: a pilot only when the plan selects D0G2 reactions
        n_plan = int((G._rj(G.G2_PLAN_D0) or {}).get("n_g2") or 0)
        n_g2, why = (1 if n_plan > 0 else 0), f"auto (plan selects {n_plan} D0G2 reactions)"
    else:
        n_g2, why = int(n_g2), "explicit"
    print(f"[g0_d0] {len(sample)} plan EnGrads + {len(lrows)} D0 label TSs on {G._cpus()} workers; D0G2 pilots {n_g2} "
          f"({why})", flush=True)
    ctx = G.Ctx("g0d0")
    t0 = time.time()
    G.run_items([r for r in rows if not G.rec_path(r["group"], r["struct_id"]).is_file()], G._cpus(), ctx, G.RESERVE_H,
                G.run_struct, lambda r: r["struct_id"])
    if ctx.halt:
        return 3
    wall_e = time.time() - t0
    pil, rc_g2, wall_g2 = [], None, 0.0
    if n_g2 > 0 and G.G2_POOL_D0.is_file():
        hset = holdout_ids()                     # D0-hold: a pilot is never a holdout reaction
        pil = pick_pilots_d0([u for u in M.read_json(G.G2_POOL_D0)["ranked"] if int(u["rxn_id"]) not in hset], n_g2)
        pend = [x for x in pil if not (G.RUNS / "D0G2" / str(x) / "unit.json").is_file()]
        if pend:
            print(f"[g0_d0] D0G2 pilots {pend} (MLTENV=d1 subprocess)", flush=True)
            rc_g2, wall_g2 = G.run_g2_subprocess(pend, d0=True)
            if rc_g2 == 3:
                return 3
    est = g0_estimate_d0(P, meta, rows, pil, dict(wall_engrads_s=wall_e, wall_g2_s=wall_g2, rc_g2=rc_g2,
                                                   stats=dict(ctx.stats), sample=[r["struct_id"] for r in sample],
                                                   label=[r["struct_id"] for r in lrows], max_atoms=max_atoms))
    M.write_json(G.G0_D0_JSON, est)
    print(json.dumps(dict(predicted_core_h=est["predicted_core_h"]["by_group"], d1_total=est["predicted_core_h"]["d1_total"],
                          slowdown=est["predicted_core_h"]["engrad_slowdown"],
                          total=est["predicted_core_h"]["total"], stop=est["stop"], n_g2_rule=est["n_g2_rule"],
                          margin_met=est["budget_rule"]["margin_met_without_g2"],
                          if_n_near_1={k: est["if_n_near_1"][k] for k in ("total", "stop", "n_g2_rule")},
                          parity={k: est["parity"].get(k) for k in ("n", "median_max_abs_grad", "max_max_abs_grad",
                                                                    "off_stationary", "above_d1_expectation",
                                                                    "ratio_to_d1_median", "above_10x_d1_median")},
                          phase_i_parity=est["parity"]["phase_i_d0_ts_grad"].get("median_max_abs_grad"),
                          fit=est["engrad_fit_used"], ratio=est["d0_vs_d1_fit_ratio"],
                          suggested_tasks=est["suggested_tasks"]), indent=1, default=str), flush=True)
    if est["n_engrad_ok"] < 3:
        return 2
    return 3 if est["stop"] else 0


# ================================================================ collect (D0)
def _dform(pos, ref, pairs):
    pos, ref = np.asarray(pos, float), np.asarray(ref, float)
    return float(max(abs(np.linalg.norm(pos[i] - pos[j]) - np.linalg.norm(ref[i] - ref[j])) for i, j in pairs))


def cmd_collect_d0():
    P = G.load_plan(d0=True)
    meta = G._rj(G.PLAN_D0_META) or {}
    g2p = G._rj(G.G2_PLAN_D0) or {"rxns": []}
    ranked = (G._rj(G.G2_POOL_D0) or {}).get("ranked") or []
    holdout_check("collect --d0")                                    # decision D0-hold-fix
    hset = holdout_ids()                                             # eval + sibling (every TS-derived row marked)
    heval = holdout_ids("eval")
    nmark = Counter()
    rep = dict(phase="G collect D0", code=CODE_D0_REV, written=G._now(), host=socket.gethostname(), groups={},
               plan_fingerprint=meta.get("fingerprint"),
               plan_sha256=M.sha256(G.PLAN_D0_PQ), g2_plan_sha256=M.sha256(G.G2_PLAN_D0) if G.G2_PLAN_D0.is_file() else None,
               decisions=DECISIONS_D0, molecules=(G._rj(G.G_PLAN_D0_JSON) or {}).get("molecules"))
    lts = {}                                                         # d0_id -> (symbols, undisplaced label TS, pairs)
    for r in P[P.kind == "d0_ts_near"].drop_duplicates("d0_id").itertuples(index=False):
        lts[int(r.d0_id)] = (str(r.symbols).split(), np.asarray(r.pos_A, float).reshape(-1, 3),
                             json.loads(r.disp)["pairs"])
    missing_total = 0
    e_by_rxn = {}
    for g in ("D0G1", "D0G3"):
        d = G.RUNS / g
        files = sorted(d.glob("*.json")) if d.is_dir() else []
        atoms, fails, core, nk, bad = [], [], Counter(), Counter(), 0
        de, seen, dfs, fdev, nimag = [], set(), defaultdict(list), [], Counter()
        for f in files:
            r = G._rj(f)
            if r is None:
                bad += 1
                continue
            seen.add(r["struct_id"])
            core[r["kind"]] += float(r.get("core_h") or 0.0)
            if r.get("status") != "ok":
                fails.append(dict(struct_id=r["struct_id"], kind=r["kind"], reason=r.get("reason")))
                continue
            nk[r["kind"]] += 1
            dd = r.get("disp") or {}
            if dd.get("freq_dev_cm") is not None:
                fdev.append(float(dd["freq_dev_cm"]))
            if r["kind"] in G.D0_TS_DISP_KINDS and dd.get("n_imag_xtb") is not None:
                nimag[str(dd["n_imag_xtb"])] += 1
            if r.get("de_out_engrad_eh") is not None:
                de.append(abs(r["de_out_engrad_eh"]))
            extra = {}
            rid = int(r.get("d0_id", -1))
            if r["kind"] in ("d0_ts_near", "d0_ts_imag", "d0_ts_label") and rid in lts \
                    and list(r["symbols"]) == lts[rid][0]:
                extra["dform_label_A"] = _dform(r["pos_A"], lts[rid][1], lts[rid][2])
                dfs[r["kind"]].append(extra["dform_label_A"])
                e_by_rxn.setdefault(rid, {})[r["struct_id"]] = (r["kind"], r["energy_eh"],
                                                                (r.get("disp") or {}).get("e_harm_kcal"))
            if rid in hset and r["kind"] not in HOLD_KEEP_KINDS:     # D0-hold: TS-derived row of a holdout reaction
                extra["holdout"] = True
                nmark[g] += 1
            atoms.append(M.ext_atoms(r["symbols"], r["pos_A"], r["energy_eh"], r["grad_eh_bohr"], struct_id=r["struct_id"],
                                     kind=r["kind"], source=KIND_SRC.get(r["kind"], r["kind"]), rxn_ids=r["rxn_ids"],
                                     folds=r["folds"], dipoles=r["dipoles"], level="label", group=g, d0_id=rid,
                                     d0_class=r.get("d0_class"), charge=int(r.get("charge") or 0), **extra))
        M.write_extxyz(G.EXT_D0[g], atoms)
        planned = set(P.loc[P.group == g, "struct_id"])
        miss = sorted(planned - seen)
        missing_total += len(miss)
        rep["groups"][g] = dict(n_records=len(files), n_unreadable=bad, n_ok=len(atoms), n_ok_by_kind=dict(nk),
                                n_fail=len(fails), fail_reasons=dict(Counter(x["reason"] for x in fails)),
                                failures=fails[:200], core_h_by_kind=dict(core), core_h=float(sum(core.values())),
                                n_planned=len(planned), n_missing=len(miss), missing=miss[:100],
                                n_extra_not_in_plan=len(seen - planned), extra_not_in_plan=sorted(seen - planned)[:50],
                                out_engrad_energy_abs_diff_eh=_stats(de),
                                dform_label_A={k: _stats(v) for k, v in dfs.items()}, freq_dev_cm=_stats(fdev),
                                ts_disp_n_imag_xtb=dict(nimag), extxyz=str(G.EXT_D0[g]))
    # ---- D0G2: the selected (training) units + any pool unit with a record (g0_d0 pilot, a manual --only run)
    sel = {int(u["rxn_id"]): u for u in g2p["rxns"] if int(u["rxn_id"]) not in hset}
    if len(sel) != len(g2p["rxns"]):
        M.stop(f"g2_plan_d0.json 'rxns' lists {len(g2p['rxns']) - len(sel)} holdout reactions (decision D0-gauss: the "
               f"holdout is never refined); re-run plan_d0")
    pool = {int(u["rxn_id"]): u for u in ranked}
    ondisk = {int(p.parent.name) for p in (G.RUNS / "D0G2").glob("*/unit.json")} if (G.RUNS / "D0G2").is_dir() else set()
    units = dict(sel)
    units.update({r: pool[r] for r in ondisk if r in pool and r not in sel})
    atoms, st, core, nsteps, miss2, conv_label, dfs = [], Counter(), 0.0, [], [], [], []
    hunits = []                                                      # listed reactions with a D0G2 record (marked)
    for rid, u in sorted(units.items()):
        ud = G.RUNS / "D0G2" / str(rid)
        uj = G._rj(ud / "unit.json")
        hold = rid in hset
        if uj is None:
            if rid in sel:
                miss2.append(rid)
            continue
        if hold:
            hunits.append(rid)
        st[uj.get("status")] += 1
        core += float(uj.get("core_h") or 0.0)
        if uj.get("converged_to_label") is not None:
            conv_label.append(bool(uj["converged_to_label"]))
        sj = G._rj(ud / "steps.json")
        if not sj:
            continue
        nsteps.append(len(sj["steps"]))
        ls, lx = M.read_xyz(u["label_ts"])
        same = list(ls) == list(sj["symbols"])
        for j, s in enumerate(sj["steps"]):
            kind = "d0_" + s["kind"]
            extra = {}
            if same:
                extra["dform_label_A"] = _dform(s["pos_A"], lx, u["pairs"])
                dfs.append(extra["dform_label_A"])
            atoms.append(M.ext_atoms(sj["symbols"], s["pos_A"], s["energy_eh"], s["grad_eh_bohr"],
                                     struct_id=f"d0g2_{rid}_s{j:03d}", kind=kind, source=KIND_SRC[kind],
                                     rxn_ids=u["rxn_ids"], folds=u["folds"], dipoles=u["dipoles"], level="label",
                                     group="D0G2", cycle=int(s["cycle"]), d0_id=rid, d0_class=u["d0_class"],
                                     charge=int(u["charge"]), holdout=hold, **extra))
            nmark["D0G2"] += hold
    M.write_extxyz(G.EXT_D0["D0G2"], atoms)
    # ---- holdout (decisions D0-hold-ref / D0-gauss): reference = Coley TS, starts = G2c xTB TS; files + element order
    href = G._rj(HOLD_REF) or {}
    if hset and href.get("holdout_sha256") != M.sha256(HOLD_CSV):
        M.stop(f"{HOLD_REF} does not belong to {HOLD_CSV} (decision D0-hold-fix); re-run plan_d0")
    rchk, rbad = Counter(), []
    for x in href.get("units") or []:
        rid = int(x["d0_id"])
        want = lts.get(rid, (None,))[0]
        for nm, p in (("reference", x["reference"]["xyz"]), ("xtb_start", x["starts"]["xtb_g2c"])):
            try:
                ok_ = list(M.read_xyz(p)[0]) == list(want) if want is not None else False
            except Exception:                                          # noqa: BLE001
                ok_ = False
            rchk[f"{nm}_ok"] += ok_
            if not ok_:
                rbad.append(f"{rid}:{nm}")
    rep["holdout"] = dict(rule=DECISIONS_D0["D0-hold"], sib_rule=DECISIONS_D0["D0-sib"],
                          ref_rule=DECISIONS_D0["D0-hold-ref"], file=str(HOLD_CSV),
                          sha256=M.sha256(HOLD_CSV) if HOLD_CSV.is_file() else None, n_eval=len(heval),
                          n_listed=len(hset), n_rows_marked=dict(nmark), n_d0g2_units_with_record=len(hunits),
                          d0g2_units_with_record=hunits[:50], n_ref_units=len(href.get("units") or []),
                          reference_checks=dict(rchk), reference_problems=rbad[:100],
                          ref_json=str(HOLD_REF) if HOLD_REF.is_file() else None)
    missing_total += len(miss2)
    rep["groups"]["D0G2"] = dict(n_planned=len(sel), n_extra_pool_units=len(units) - len(sel), n_units=sum(st.values()),
                                 by_status=dict(st), core_h=core, n_structures=len(atoms),
                                 n_by_kind=dict(Counter(a.info["kind"] for a in atoms)), steps_per_rxn=_stats(nsteps),
                                 n_missing=len(miss2), missing=miss2[:100],
                                 converged_to_label=dict(n=sum(conv_label), of=len(conv_label)),
                                 dform_label_A=_stats(dfs), n_ts_near_0p3=sum(x <= 0.3 for x in dfs),
                                 extxyz=str(G.EXT_D0["D0G2"]))
    hj = [G._rj(p) for p in sorted(G.HESS_D0.glob("*.json"))] if G.HESS_D0.is_dir() else []
    rep["xtb_hessians"] = dict(n=len(hj), by_status=dict(Counter((x or {}).get("status") for x in hj)),
                               core_h=float(sum((x or {}).get("core_h") or 0.0 for x in hj)),
                               n_ts=sum(str((x or {}).get("mol_id", "")).startswith("d0ts_") for x in hj),
                               n_ts_n_imag=dict(Counter(str(len((x or {}).get("imag_cm") or [])) for x in hj
                                                        if str((x or {}).get("mol_id", "")).startswith("d0ts_"))))
    spread, vs_label = [], []
    for rid, d in e_by_rxn.items():
        es = [v[1] for v in d.values()]
        if len(es) > 1:
            spread.append((max(es) - min(es)) * M.EH2KCAL)
        lab = [v[1] for v in d.values() if v[0] == "d0_ts_label"]
        if lab:
            vs_label += [dict(struct_id=s, kind=k, de_dft_kcal=(e - lab[0]) * M.EH2KCAL, e_harm_xtb_kcal=eh)
                         for s, (k, e, eh) in d.items() if k != "d0_ts_label"]
    rep["energy_checks"] = dict(d0g1_spread_within_rxn_kcal=_stats(spread), d0g1_vs_label_ts=vs_label[:100],
                                note="de_dft = E(displaced) - E(D0 label TS) at the label level; the D0 label TS is not "
                                     "stationary at this level (decision D0-level), so de_dft can be negative")
    g0 = G._rj(G.G0_D0_JSON) or {}
    rep["g0_d0"] = dict(parity=g0.get("parity"), predicted_core_h=g0.get("predicted_core_h"), n_g2_rule=g0.get("n_g2_rule"))
    rep["core_h_measured"] = dict(D0G1=rep["groups"]["D0G1"]["core_h"], D0G2=core, D0G3=rep["groups"]["D0G3"]["core_h"],
                                  xtb_hessians=rep["xtb_hessians"]["core_h"])
    rep["core_h_measured"]["total"] = float(sum(rep["core_h_measured"].values()))
    rep["n_missing"] = missing_total
    rep["complete"] = missing_total == 0
    rep["roundtrip"] = roundtrip_d0()
    rep["stop_file"] = G.STOP_D0.read_text().strip() if G.STOP_D0.is_file() else None
    M.write_json(G.G_REPORT_D0, rep)
    print(json.dumps(dict(complete=rep["complete"], n_missing=missing_total, core_h=rep["core_h_measured"],
                          ok={g: rep["groups"][g].get("n_ok", rep["groups"][g].get("n_structures")) for g in rep["groups"]},
                          roundtrip=rep["roundtrip"]), indent=1, default=str), flush=True)
    return 0 if rep["complete"] else 2


def roundtrip_d0():
    """read every D0 extxyz back with mlt_common.read_extxyz: count, tag strings, energy / forces of the first."""
    out = {}
    for g, p in G.EXT_D0.items():
        if not p.is_file() or p.stat().st_size == 0:
            out[g] = dict(n=0)
            continue
        try:
            al = M.read_extxyz(p)
            a = al[0]
            out[g] = dict(n=len(al), struct_id=a.info.get("struct_id"), kind=a.info.get("kind"),
                          dipoles=a.info.get("dipoles"), rxn_ids=a.info.get("rxn_ids"), d0_id=a.info.get("d0_id"),
                          charge=a.info.get("charge"), dform_label_A=a.info.get("dform_label_A"),
                          REF_energy_eV=a.info.get("REF_energy"), e_eh_x_EH2EV=float(a.info.get("e_eh")) * M.EH2EV,
                          has_forces="REF_forces" in a.arrays,
                          tag_types={k: type(a.info.get(k)).__name__ for k in ("rxn_ids", "folds", "dipoles")},
                          dipole_tags=dict(Counter(x for b in al for x in str(b.info.get("dipoles", "")).split(",") if x)),
                          n_holdout_true=sum(b.info.get("holdout") is True or str(b.info.get("holdout")) == "True"
                                             for b in al),
                          holdout_types=sorted({type(b.info["holdout"]).__name__ for b in al if "holdout" in b.info}))
        except Exception as e:                                                 # noqa: BLE001
            out[g] = dict(error=f"{type(e).__name__}: {e}")
    return out


def main(argv=None):
    """`gen_d0.py plan_d0 ... [--n-holdout H] [--replace-holdout] [--unpin]` (D0-plus); any other command ->
    gen_dft.main (same arguments). An explicit --n-g2 K here pins K (decision D0-pin). Bad arguments = exit 3 (a retry
    cannot fix them; PIPELINE.md 2 = retry, 3 = HALT)."""
    import argparse
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] != "plan_d0":
        return G.main(argv)
    ap = argparse.ArgumentParser(prog="gen_d0.py plan_d0", description="D0 plan (gen_dft plan_d0 + D0-plus options)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n-g2", default="auto", help="training D0G2 reactions (explicit K pins K, decision D0-pin)")
    ap.add_argument("--n-near", type=int, default=2)
    ap.add_argument("--n-holdout", type=int, default=None,
                    help="evaluation-only D0 reactions (decision D0-hold); omitted = keep the existing list")
    ap.add_argument("--replace-holdout", action="store_true",
                    help="allow a different holdout list (refused once FT data exist; decision D0-hold-fix)")
    ap.add_argument("--unpin", action="store_true", help="drop the pinned --n-g2 (back to the auto rule)")
    ap.add_argument("--force", action="store_true", help="re-plan even if up to date; skip the in-flight guard")
    try:
        x = ap.parse_args(argv[1:])
    except SystemExit as e:
        return 0 if e.code == 0 else 3
    if x.unpin and str(x.n_g2) != "auto":
        M.stop("plan_d0: --unpin with an explicit --n-g2 (pass one of them)")
    pin = False if x.unpin else (True if str(x.n_g2) != "auto" else None)
    return cmd_plan_d0(x.limit, x.n_g2, x.force, x.n_near, x.n_holdout, x.replace_holdout, pin)


if __name__ == "__main__":
    sys.modules.setdefault("gen_d0", sys.modules[__name__])   # gen_dft.main imports gen_d0: one module copy
    sys.exit(main())
