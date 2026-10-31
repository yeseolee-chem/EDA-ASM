#!/usr/bin/env python3
r"""inputs.py — D1-REF·LADDER Phase P (docs/specs/D1LADDER.md 7; rung columns 4-1, arms 4-2, folds 4-3) and the shared
data loader of the training / test code (spec 8: L-4, L-5, L-6, L-10, L-11 read through it). sbatch only; nothing in
this file was run when it was written (login node), so every path is untested until it runs inside a job.

CLI (exit 0 = done / up to date, 2 = not ready (an input phase has not finished; nothing written), 3 = STOP (spec 16 P)
or hard error; every command writes its report atomically, a rerun recomputes and overwrites it):
  python inputs.py build <dft|g1|g2> [--force] [--accept-f0-stop] [--g2-q-keys]
        rung input table $LAD/inputs/<rung>.parquet + results/P_build_<rung>.json (gate P-a, re-referencing, tokens)
  python inputs.py pa <dft|g1|g2>
        gate P-a only (no re-referencing, no V / F0 precondition) -> results/P_a_<rung>.json (early check while V runs)
  python inputs.py pairs
        pair sets P_dft, P_g1, P_01 (+ P_g2, P_012 once 'build g2' passed) -> $LAD/inputs/sets/P_<set>.csv,
        fold copies $LAD/inputs/fu2_splits/, results/P_report.json (STOP: P_g1 != FU2 pairs, P_dft < 0.98 x P_g1)
Environment: build / pa / pairs in LADENV=reactot (conda reactot: pandas + pyarrow, the env of refs_q.py and of the D1-HF
build.py runs); the loader (LadderData / load) in env_d1fu2 (FU2's torch pin; FU2 job wrapper fu2/jobs/fu2.slurm,
PYENV=fu2); the frame API (pairs_frame / reaction_frame / fold_lists / read_labels) in any env with pandas + pyarrow.
sbatch (one-shot, from analysis/d1_ladder; first `squeue -u $USER -h -r | wc -l` (MaxSubmit 20 incl. array elements,
LADDER <= 4 jobs while D1-G2 runs) and `sinfo -p cpu1,cpu2 -o "%.9P %.6t %C"`):
  now (independent of V / F0):
    sbatch -p cpu1,cpu2 -J lad_P1 --time=48:00:00 --cpus-per-task=2 --mem=16G --export=ALL,LADENV=reactot \
        jobs/lad.slurm inputs.py build g1 ';;;' inputs.py pa dft
  after Phase V collect passed and the F0 STOP is resolved by the user (see P16); 'build g1' first is a no-op when
  current and rebuilds it when its fingerprint went stale (P22):
    sbatch -p cpu1,cpu2 -J lad_P2 --time=48:00:00 --cpus-per-task=2 --mem=16G --export=ALL,LADENV=reactot \
        jobs/lad.slurm inputs.py build g1 ::: inputs.py build dft [--accept-f0-stop] ::: inputs.py pairs
  stage 2 (D1-G2 F / H passed, committed, origin/d1-g2 merged; P20 decided by the user):
    sbatch ... jobs/lad.slurm inputs.py build g2 [--g2-q-keys] ::: inputs.py pairs

Outputs (written only under $LAD and analysis/d1_ladder/results, spec 15)
  $LAD/inputs/<rung>.parquet   one row per target (9,266, rxn_id order): rxn_id, job_id, has_tok, tok_path, tok_sha256,
                               tok_n_atoms, has_x182, has_h, usable, the 182 X182 columns (X182 names and order), H_<t>
                               (re-referenced, spec 4-1), H0_<t> (= labels_d1_hf values), h_changed, per fragment k:
                               ref<k>_src (own | Q | V | min), ref<k>_src_job, ref<k>_src_frag, ref<k>_key,
                               e_ref<k>_hf_orig_eh, e_ref<k>_hf_eh;  t in H_TARGETS = d1 d2 elst pauli oi disp cpcm cds barrier
  results/P_build_<rung>.json  inputs + sha256, P-a, re-referencing checks and counts, tokens, outputs (sha256)
  $LAD/inputs/sets/P_<set>.csv pair_id + pair attributes (S_folds.csv order) of P_dft, P_g1, P_01 [, P_g2, P_012]
  $LAD/inputs/fu2_splits/      byte copies of FU2 splits/pairs.parquet and T1__f0..4.json (training reads these)
  results/P_report.json        counts by set (reference / anchor / para, P_main, per fold), dropped pairs with reasons,
                               H values changed by re-referencing per rung, FU2 check, 2 % rule, folds, labels, sha256

Loader API (training / tests)
  D = LadderData(rung, arm="S"|"SH", label_path=None, h="cr"|"orig")    # a fu2.train.Data (subclass)
  D, sp = load(rung, arm, label_path, fold, h)                           # sp = D.split("T1", fold) (FU2 dict)
  pairs_frame(rung, label_path, h, subset="usable"|"has_h") / reaction_frame(rung, arm, h) / fold_lists(rung, fold)
                                                                         # XGB / LINE / evaluation (no torch)
  read_labels(path), p_report(rung), read_table(rung), read_set(name), check_paths(rung, paths)
  PATHLOG: L-4 open-path log; a training / test process calls inputs.PATHLOG.install() first (LadderData installs it
  too) and PATHLOG.dump(<$LAD or results path>) at the end; PATHLOG.paths() = data files read.
  label_path default = $LAD/F_pairs_cr2.csv (spec 4-1 y); ORIG_LABELS = D1PAIR results/F_pairs.csv (L-6, L-10, L-11).

Found (also in the reports 'found')
  * D1-HF assembly = analysis/d1_hf/build.py parse_rxn(T, geom, level="hf") on $HF/runs/<geom>/<T>/ (ORCA outputs
    <calc>.out.gz read by hf_common.read_text, energies by hf_common.RE_FSPE.findall): d1 = (E(frag1_dist) - E(frag1_rel))
    x 627.5094740631, d2 likewise, barrier = (E(eda) - E(frag1_rel) - E(frag2_rel)) x 627.5094740631; elst, pauli, oi,
    disp, cpcm, cds from the EDA table (no reference energy). results/labels_d1_hf.parquet = build.py cmd_all: flat(rec)
    of the status-ok records + rxn_id (dft 9,266 and g1 8,433 rows, hf_status.csv). g2: the same function with
    D1HF_SCRATCH = $G2 (runs $G2/runs/g2, inputs $G2/inputs/<T>/g2), analysis/d1_g2/results/labels_d1_hf_g2.parquet.
  * reference HF energies: dft = $LAD/refs_cr2.parquet e_hf_eh (refs_v V14: own -> labels_d1_hf dft of the row, Q ->
    of the source row, V -> the V HF single point); g1 / g2 = labels_d1_hf(_g2) e_frag{k}_rel_eh at the rung's
    rel{k}.xyz.
  * FU2 data: fu2.train.Data reads fu2_common FU2_RES/{T_report.json, Q_report.json, T_manifest.csv, S_splits.json},
    TOKENS/<rxn_id>.npz (sha256 = T_manifest), SPLITS/pairs.parquet (= d1_train.data() S rows: S_folds.csv order, KEEP
    columns, dd_* from F_pairs.csv) and SPLITS/<split>__f<k>.json; y = pairs.parquet dd_<10 targets>; has_atoms = both
    reactions in T_manifest; the token npz 'glob' = the X182 row of FU2_X182 (tokens.py build_one).
  * rung roots: dft FU2_RES = analysis/d1_ladder/results/fu2, TOKENS = $LAD/fu2/tokens, versions fu2-*-3-dft (g0f0 F0);
    g1 = the D1PAIR ones (fu2-*-3); g2 FU2_RES = analysis/d1_g2/results/fu2, TOKENS = $G2/fu2/tokens, fu2-*-3-g2
    (phase_fh.py). atoms.py root tokens copies the FU2 S_splits.json byte for byte into each FU2_RES.

Decisions (user 2026-10-22: every branch = the recommended option; also in the reports 'decisions')
  P1  environments as above (build / pa / pairs: reactot; loader: env_d1fu2).
  P2  assembly adapter (spec 4-1 'build.py with only the reference energies replaced', the refs_q.assemble pattern):
      build.py parse_rxn is called unchanged on the real (read-only) D1-HF run folders; for the call
      hf_common.read_text returns the output text tagged with its calculation name and hf_common.RE_FSPE is a proxy
      whose findall returns [repr(E_new)] for a frag{k}_rel output with a replacement (that output must have exactly one
      FSPE line; every replacement must be consumed exactly once); hf_common.RUNS / INPUTS are rebound to the rung's
      roots during the call (no-op for dft / g1, $G2 for g2). ORCA output texts are cached per reaction, so the
      replaced call re-reads only the small .inp / timing.tsv files (parse_rxn reads those directly).
  P3  gate P-a = the same call without replacement, row by row against every labels_d1_hf(_g2).parquet row of the rung:
      every float column <= 1e-9 (bookkeeping coreh_* / n_attempts reported, not gated), every other column equal,
      status ok, no parse error. P-a fails -> STOP (exit 3), no table written.
  P4  re-assembled H: rows without a replacement = the labels_d1_hf values verbatim; with a replacement: d1, d2, barrier
      from the replaced call, the 6 channels verbatim. Checks (Q-c / Q-b analogues, every replaced row): the call's
      e_frag{k}_rel_eh = the replacement exactly, delta d_k and delta barrier = the reference energy differences
      (<= 1e-9 kcal/mol), identity / seam residuals unchanged (<= 1e-9), every other returned field identical, status ok.
  P5  dft references = $LAD/refs_cr2.parquet (Phase V V-4 / V14); requires results/V_report.json all_pass with unchanged
      outputs (refs_cr2.parquet, F_pairs_cr2.csv included) and Phase Q ready (refs_v.q_ready). Checked: every target
      fragment once; own -> e_hf = the row's own labels_d1_hf value (no replacement); Q -> e_hf = the source row's
      labels_d1_hf value; V -> finite. V not collected -> exit 2.
  P6  g1 / g2 references: the Q-2 keys of $LAD/Q_keys.parquet (Phase Q ready), per key the lowest e_frag{k}_rel_eh among
      the rung's labels_d1_hf(_g2) rows (refs_q.audit, ties = smallest job_id); every fragment whose own value is above
      it gets it (no 0.05 kcal/mol threshold: spec 4-1 'the lowest per key', deployment computes one reference per
      molecule); counts with dev > 0.05 are reported. The g2 keys inherit the refs_q fixed-dihedral suffix of
      constrained dipoles, computed from the D1 (DFT) reference geometry (task: 'refs_q keys'); its count is reported.
  P7  usable reaction = in the rung T_manifest (each npz sha256 = manifest, token 'glob' = the X182 row exactly, token
      count = n_atoms) and an all-finite X182 row and 9 finite H values; a token reaction without an X182 row = STOP.
      T_report / Q_report stop false, token / atom versions of the rung, T_report sha256 links (manifest, Q_report,
      X182, splits = the FU2 S_splits.json) or STOP; token layout (num_cols, bin_cols, glob_cols, ext_used) = the g1
      rung's or STOP (elements reported).
  P8  pair universe = D1PAIR results/S_folds.csv (8,389 rows) = FU2 pairs.parquet (pair ids, order, ends, ref_kind,
      core checked; else STOP); P_<rung> = both ends usable; P_01 = P_dft & P_g1; P_012 = P_01 & P_g2.
  P9  FU2 check: FU2 used pairs = pairs.parquet rows with both ends in the FU2 T_manifest (= fu2.train.Data has_atoms);
      P_g1 must be the same pair set, else STOP (spec 16 P). The T1 train / test lists are compared too (reported).
  P10 STOP if |P_dft| < 0.98 x |P_g1| (spec 7 / 16 P).
  P11 folds: FU2 splits T1__f0..4.json (= d1_train.masks + first inner GroupKFold(4) fold) and pairs.parquet are
      copied byte for byte (sha256 = FU2 S_splits.json) to $LAD/inputs/fu2_splits and checked against S_folds.csv
      (train = fold_T1 != k & in_T_main, test = fold_T1 == k & (in_P_sens | in_A_main), inner = partition of train
      with disjoint cores); this module's loader API (LadderData, fold_lists) reads the copies. The trainers
      (train_atoms.py / train_xgb.py) read the FU2 originals; those exact files (pair-id lists, the same folds for
      every rung, spec 4-3) and FU2 S_splits.json are 'shared' in rung_roots (fix 2026-10-23 of review P-4 / M-1;
      the earlier text 'training reads the copies' was wrong).
  P12 loader: fu2.train.Data.__init__ runs UNCHANGED with the fu2_common roots FU2_RES / TOKENS / SPLITS and the
      atoms.ATOM_VERSION / tokens.TOKEN_VERSION constants rebound to the rung for the call (the atoms.py root_bind
      pattern of D1-G2 Phase T); then only: y <- the label file (read exactly as d1_train.data reads F_pairs.csv), pair
      ids / ends checked; has_atoms <- pair in P_<rung> (= the set file, checked); arm SH: glob <- [glob | 9 H] (Std
      standardises them like the 182 with the fit's training reactions); dH [n_pairs, 10] = H(a) - H(b) in fu2
      TARGETS order, c_ghost column 0 (spec 4-2: beta = 0), NaN for pairs outside P_<rung>. split() = Data.split under
      the same rebinding. Token rows without H (not usable) get H = 0 and are in no usable pair (checked).
  P13 label file argument (spec 8): default $LAD/F_pairs_cr2.csv; pair ids and a_rxn / b_rxn must equal the universe.
  P14 h = 'cr' (re-referenced, default) or 'orig' (= labels_d1_hf values; L-10 'original H').
  P15 L-4 path log: a sys.addaudithook 'open' hook (every Python-level open) + explicit notes of the parquet paths this
      module reads (pyarrow opens files in C++ without an audit event); check_paths classifies paths by rung column
      (longest matching root).
  P16 F0: results/F0_report.json has stop = true (F-1 132 / 9,266 = 1.42 % > 1 %, spec 16 F0 STOP) at the time of
      writing. build dft exits 3 on an F0 stop unless --accept-f0-stop is given (to be used only after a user decision;
      recorded in P_build_dft.json). G0_report stop must be false.
  P17 g2 (spec 12): build g2 requires analysis/d1_g2/results/F_report.json and H_report.json complete, stop false, the
      H_report outputs unchanged, and those files + labels_d1_hf_g2.parquet + fu2/T_report.json / T_manifest.csv
      tracked by git and unmodified; else exit 2.
  P18 H column names H_<t> / H0_<t> (no collision with the 182 names, checked).
  P19 file name / layout: spec 8 lists inputs.py; the pair-set files are csv (readable on the login node with awk).
Review fixes 2026-10-23 (FIXER; every judgment = the recommended option, user 2026-10-22)
  P6b g1 / g2 reference graphs (review P-2): before the per-key minimum, every rung reference geometry with an energy
      (the frag{k}_rel input of its HF single point) must have its key's composition and bond-graph WL hash (the D1
      label reference of the same row) and be isomorphic within its key (refs_q.check_graphs); else STOP (spec Q-2
      '다르면 STOP'), report 'reference_graphs'.
  P20 g2 constrained keys (review P-3): the Q-2 '|fixed:' suffix of constrained dipoles is computed from the D1 (DFT)
      reference geometry; using it for g2 puts DFT-only information into the g2 H values. That is a method question
      not settled by the spec (spec 16 '언제든'): build g2 STOPs while any g2 row sits in a constrained key, unless the
      user chose the spec-literal Q-2 keys (--g2-q-keys, recorded in the fingerprint and the report).
      Decision 2026-10-23 (user: every judgment = the recommended option; ADD3 10 left it to the supplement): option
      (c) --g2-cons-own: g2 rows in constrained keys are never replaced (each reaction keeps its own G2 constrained
      dipole reference, as at deployment); non-constrained keys take the per-key minimum. The DFT-derived '|fixed:'
      state never enters a g2 value (a constrained key holds constrained rows only, refs_q.molecule_keys). Limitation
      (review 2026-10-23; recorded, not changed): WHICH g2 rows are kept is the per-row `constrained` flag of
      Q_keys.parquet = the D1 label ref_port.constrained (D1 / DFT pipeline), not G2's own ref_result.json
      port.constrained (not a committed input); a G2-constrained / D1-free g2 row still takes its free key's minimum.
      Mode recorded as fingerprint / report g2_q_keys = 'own' (True = spec-literal Q-2 keys); --g2-q-keys /
      --g2-cons-own are refused for rungs other than g2.
  P21 g2 H_report output key (review P-1): D1-G2 keys its outputs by its own worktree's absolute path; matched by the
      repo-relative path (exactly one entry).
  P22 build currency (review P-6 / P-8): pairs and the loader accept a build only if its fingerprint (inputs + code,
      now incl. refs_v.py) recomputed now equals the recorded one (stale -> exit 2, rebuild); a build STOP stays a STOP
      (exit 3) in pairs; readiness errors write a STOP report (P-7); a missing label file / P_report = exit 2.
  P23 L-4 roots (review P-4): the FU2 T1 fold files + FU2 S_splits.json + V_report / Q_report are 'shared'; the DFT-only
      phase results (G0_*, F0_*, Q_audit, Q6, labels_d1_cr*.parquet, V_* but V_report) are 'dft'; check_paths ok also
      needs no unclassified path.
GEO2b additions 2026-10-31 (docs/specs/GEO2.md 8-1 read with GEO1B.md 4-3 and the user decision D0D1; role FEAT,
analysis/geo2/results/DECISIONS_feat.md). Added only; dft / g1 / g2, RUNG_ORDER, the CLI and DECISIONS are unchanged.
  PG-1 RUNGS['geo'] is fixed here (= analysis/geo2/feat_geo.py rung_cfg('geo'); feat_geo p compares them by string
       values): the 'g2' geometry slot of the D1-HF runner, fu2_res / h_parquet under analysis/geo2/results, tokens /
       x182 / hf_runs / hf_inputs under $GEO2B = $GEO2_SCRATCH (default /gpfs/tmp_cpu2/yeseo1ee/geo2b), versions
       fu2-*-3-geo, refs min, plus two keys only geo has: targets (the combined D1 + D0 labels
       $GEO2B/comb/labels_d0d1.json) and keys ($LAD/Q_keys.parquet, $GEO1B/d0/Q_keys_d0.parquet). The geo table is
       $GEO2B/inputs/geo.parquet (feat_geo.py p), not table_path('geo'); 'geo' is not in RUNG_ORDER (build / pa / pairs
       and the loader CLI keep dft, g1, g2).
  PG-2 rung_roots: $GEO2B, analysis/geo2/results and $GEO1B/d0 (the D0 labels / pairs / keys) form the column 'geo'
       (L-4 of the geo rung; for the other rungs such a path was unclassified before and is foreign now: check_paths ok
       stays false either way).
  PG-3 targets_of(rung): L.targets() for every rung without a 'targets' entry (dft / g1 / g2: unchanged); for geo the
       combined labels, whose D1 rows must be exactly L.targets() (job_id, rxn_id) and whose D0 rows have job_id
       D0G2_<rid:04d>; make_table loops over targets_of(rung) (dft / g1 / g2: the same 9,266 rows as before).
  PG-4 _load_keys(rung): refs_q.load_keys() ($LAD/Q_keys.parquet) for every rung without a 'keys' entry; for geo the
       concatenation of the key tables (D0 keys carry '|s3d:' / '|err:' and never equal a D1 key, checked; (job_id,
       frag) unique, checked), so refs 'min' and the P6b graph check stay per dataset; refs_min reads it.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True        # no __pycache__ in analysis/d1_pair, d1_hf, d1_g2 (read only)

import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lad_common as L  # noqa: E402

DP = L.DP
KCAL = L.KCAL
CODE_VERSION = "d1-ladder-p1"

# ---------------------------------------------------------------- targets / columns
TARGETS = list(DP.TARGETS)                       # fu2 / d1pair order: d1 d2 elst pauli oi disp cpcm cds c_ghost barrier
H_TARGETS = list(DP.PRIMARY_TARGETS)             # the 9 HF values (spec 4-1); c_ghost has no HF counterpart
assert H_TARGETS == ["d1", "d2", "elst", "pauli", "oi", "disp", "cpcm", "cds", "barrier"], H_TARGETS
H_COLS = [f"H_{t}" for t in H_TARGETS]
H0_COLS = [f"H0_{t}" for t in H_TARGETS]
H_REREF = ("d1", "d2", "barrier")                # the H values a reference energy enters
CH6 = ("elst", "pauli", "oi", "disp", "cpcm", "cds")
DD_COLS = [f"dd_{t}" for t in TARGETS]
ARMS = ("S", "SH")
H_SOURCES = ("cr", "orig")
RUNG_ORDER = ("dft", "g1", "g2")
TOL = 1e-9                                       # P-a, re-assembly checks (kcal/mol)
MAX_DROP_VS_G1 = 0.02                            # spec 7 / 16 P
N_T1 = 5
REREF_FIELDS = {"d1", "d2", "barrier", "identity_residual", "seam_residual", "e_frag1_rel_eh", "e_frag2_rel_eh"}
SET_COLS = ["pair_id", "ref_kind", "a_job", "a_rxn", "b_job", "b_rxn", "core", "dipole", "skeleton", "fam", "pos",
            "fold_T1", "in_P_main", "in_P_sens", "in_A_main", "in_T_main"]
META_COLS = ["rxn_id", "job_id", "has_tok", "tok_path", "tok_sha256", "tok_n_atoms", "has_x182", "has_h", "usable"]
PROV_COLS = [f"{p}{k}{s}" for k in (1, 2) for p, s in (("ref", "_src"), ("ref", "_src_job"), ("ref", "_src_frag"),
                                                       ("ref", "_key"), ("e_ref", "_hf_orig_eh"), ("e_ref", "_hf_eh"))]

# ---------------------------------------------------------------- paths
INP = L.LAD / "inputs"
SETS = INP / "sets"
SPLITS_COPY = INP / "fu2_splits"                 # training-time F.SPLITS (byte copies, P11)
P_REPORT = L.RES / "P_report.json"
S_FOLDS = DP.RES / "S_folds.csv"
ORIG_LABELS = DP.RES / "F_pairs.csv"
F_PAIRS_CR = L.LAD / "F_pairs_cr.csv"
F_PAIRS_CR2 = L.LAD / "F_pairs_cr2.csv"
DEFAULT_LABELS = F_PAIRS_CR2
FU2_SPLITS = DP.SCRATCH / "fu2" / "splits"       # = fu2_common.SPLITS (read only)
FU2_RES_G1 = DP.RES / "fu2"                      # = fu2_common.FU2_RES (read only)
FU2_S_SPLITS = FU2_RES_G1 / "S_splits.json"
FU2_DIR = L.P_DIR / "fu2"
REFS_CR2 = L.LAD / "refs_cr2.parquet"            # = refs_v.REFS_CR2 (checked in build)
V_REPORT = L.RES / "V_report.json"               # = refs_v.RES_REPORT
F0_REPORT = L.RES / "F0_report.json"
G0_REPORT = L.RES / "G0_report.json"
Q_KEYS = L.LAD / "Q_keys.parquet"                # = refs_q.LAD_KEYS
G2_RES = L.G2_DIR / "results"
# GEO2b (PG-1 .. PG-4; read only here): $GEO2B scratch and analysis/geo2/results of the geo rung, the GEO1b D0 tables
GEO2 = Path(os.environ.get("GEO2_SCRATCH", "/gpfs/tmp_cpu2/yeseo1ee/geo2b"))   # = analysis/geo2/g3c.G3
GEO2_RES = L.REPO / "analysis" / "geo2" / "results"
GEO1B_D0 = Path("/gpfs/tmp_cpu2/yeseo1ee/geo1b/d0")                            # d0_labels.py / d0_pairs.py outputs
Q_KEYS_D0 = GEO1B_D0 / "Q_keys_d0.parquet"

RUNGS = {
    "dft": dict(geom="dft", fu2_res=L.RES / "fu2", tokens=L.LAD / "fu2" / "tokens",
                x182=L.LAD / "X182_d1_dft.parquet", h_parquet=L.D1HF_LABELS,
                hf_runs=L.HF_SCR / "runs", hf_inputs=L.HF_SCR / "inputs",
                atom_version="fu2-atoms-3-dft", token_version="fu2-tokens-3-dft", refs="cr2"),
    "g1": dict(geom="g1", fu2_res=FU2_RES_G1, tokens=DP.SCRATCH / "fu2" / "tokens", x182=DP.X182_D1,
               h_parquet=L.D1HF_LABELS, hf_runs=L.HF_SCR / "runs", hf_inputs=L.HF_SCR / "inputs",
               atom_version="fu2-atoms-3", token_version="fu2-tokens-3", refs="min"),
    "g2": dict(geom="g2", fu2_res=G2_RES / "fu2", tokens=L.G2 / "fu2" / "tokens", x182=L.G2 / "X182_d1_g2.parquet",
               h_parquet=G2_RES / "labels_d1_hf_g2.parquet", hf_runs=L.G2 / "runs", hf_inputs=L.G2 / "inputs",
               atom_version="fu2-atoms-3-g2", token_version="fu2-tokens-3-g2", refs="min"),
    # GEO2b PG-1: = analysis/geo2/feat_geo.py rung_cfg("geo") (fixed here; targets / keys only for geo, PG-3 / PG-4)
    "geo": dict(geom="g2", fu2_res=GEO2_RES / "fu2_geo", tokens=GEO2 / "fu2_geo" / "tokens",
                x182=GEO2 / "X182_d1_geo.parquet", h_parquet=GEO2_RES / "hf_geo" / "labels_d1_hf_g2.parquet",
                hf_runs=GEO2 / "hf_geo" / "runs", hf_inputs=GEO2 / "hf_geo" / "inputs",
                atom_version="fu2-atoms-3-geo", token_version="fu2-tokens-3-geo", refs="min",
                targets=GEO2 / "comb" / "labels_d0d1.json", keys=(Q_KEYS, Q_KEYS_D0)),
}

DECISIONS = {
    "P1_env": "build / pa / pairs: LADENV=reactot; loader (LadderData, load): env_d1fu2; frames: any pandas + pyarrow env",
    "P2_adapter": "build.py parse_rxn unchanged on the D1-HF run folders; hf_common.read_text tags texts, RE_FSPE proxy "
                  "returns repr(E_new) for replaced frag{k}_rel outputs (one FSPE line, consumed once); RUNS / INPUTS "
                  "rebound to the rung roots",
    "P3_Pa": "same call without replacement vs every labels_d1_hf(_g2) row: floats <= 1e-9 (coreh_* / n_attempts "
             "reported only), other columns equal, status ok",
    "P4_H": "unchanged rows = labels_d1_hf verbatim; replaced rows: d1 d2 barrier from the call, channels verbatim; "
            "Q-c / Q-b analogue checks",
    "P5_dft_refs": "refs_cr2.parquet (V14) after V_report all_pass + Q ready; own / Q / V e_hf checked",
    "P6_g_refs": "Q_keys keys; per key lowest rung HF e_frag{k}_rel_eh (refs_q.audit), no threshold; > 0.05 reported",
    "P7_usable": "token (sha256, glob = X182 row, n_atoms) + finite X182 row + 9 finite H",
    "P8_universe": "S_folds.csv = FU2 pairs.parquet (checked); P_r = both ends usable; P_01, P_012 intersections",
    "P9_fu2": "P_g1 == pairs.parquet rows with both ends in FU2 T_manifest, else STOP",
    "P10_2pct": "|P_dft| < 0.98 |P_g1| -> STOP",
    "P11_folds": "FU2 T1 split json + pairs.parquet byte copies in $LAD/inputs/fu2_splits, checked vs S_folds masks",
    "P12_loader": "fu2.train.Data.__init__ unchanged under rung roots; y from label file; has_atoms = P_<rung>; SH glob "
                  "+ 9 H; dH in TARGETS order (c_ghost 0)",
    "P13_labels": "label_path argument, default F_pairs_cr2.csv; ids / ends = universe",
    "P14_h": "h = cr (re-referenced) | orig (labels_d1_hf values)",
    "P15_pathlog": "audit-hook 'open' events + explicit parquet notes; check_paths by longest rung root",
    "P16_f0": "build dft exits 3 while F0_report stop is true unless --accept-f0-stop (after a user decision)",
    "P17_g2": "build g2 needs D1-G2 F_report / H_report complete, stop false, committed",
    "P18_names": "H_<t> / H0_<t> column names",
    "P6b_graphs": "g rungs: every reference geometry (frag{k}_rel input of its HF single point) = its Q-2 key's "
                  "composition + WL bond-graph hash, isomorphic within the key, else STOP (spec Q-2)",
    "P20_g2_keys": "g2 rows in constrained (DFT-derived '|fixed:') keys -> STOP for a user decision unless --g2-q-keys "
                   "(spec-literal keys) or --g2-cons-own (option (c), decision 2026-10-23: rows of constrained keys keep "
                   "their own G2 reference; kept rows = Q_keys 'constrained' = the D1 label ref_port flag, a recorded "
                   "limitation)",
    "P21_g2_key_path": "D1-G2 H_report outputs matched by repo-relative path",
    "P22_currency": "build accepted only with an unchanged fingerprint (stale -> exit 2); build STOP -> pairs exit 3",
    "P23_l4_roots": "FU2 T1 fold files / S_splits.json / V_report / Q_report shared; DFT-phase results dft; "
                    "unknown paths fail check_paths",
}


# ---------------------------------------------------------------- small helpers
class _Stop(Exception):
    """a spec 16 P STOP inside a command (message = the reason)."""


def _now():
    return time.strftime("%F %T")


def _log(t0, msg):
    print(f"[inputs {time.time() - t0:8.1f}s] {msg}", flush=True)


def _vc(s):
    return {str(k): int(v) for k, v in pd.Series(s).value_counts().sort_index().items()}


def _q(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {}
    return dict(n=int(len(x)), p50=float(np.quantile(x, 0.5)), p90=float(np.quantile(x, 0.9)),
                p99=float(np.quantile(x, 0.99)), max=float(x.max()), mean=float(x.mean()))


def _f(x):
    """float, or None for None / NaN / non-numbers."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _norm(x):
    """plain python value for comparisons (numpy -> python, NaN / None -> None, arrays -> lists)."""
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if isinstance(x, (list, tuple)):
        return [_norm(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _norm(v) for k, v in x.items()}
    if isinstance(x, np.generic):
        x = x.item()
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return x


def _isnum(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _sha(p):
    return L.sha256(p) if Path(p).is_file() else None


def _env():
    out = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, host=platform.node(),
               job=os.environ.get("SLURM_JOB_ID"))
    try:
        import pyarrow
        out["pyarrow"] = pyarrow.__version__
    except ImportError:
        out["pyarrow"] = None
    return out


def _git_rc(*a):
    return subprocess.run(["git", "-C", str(L.REPO), *a], capture_output=True, text=True).returncode


def _committed(path):
    """tracked by git and identical to HEAD (spec 12 'committed')."""
    p = str(path)
    return _git_rc("ls-files", "--error-unmatch", p) == 0 and _git_rc("diff", "--quiet", "HEAD", "--", p) == 0


def table_path(rung):
    return INP / f"{rung}.parquet"


def build_report_path(rung):
    return L.RES / f"P_build_{rung}.json"


def pa_report_path(rung):
    return L.RES / f"P_a_{rung}.json"


def set_path(name):
    return SETS / f"{name}.csv"


def _code_files():
    return [Path(__file__).resolve(), HERE / "lad_common.py", HERE / "refs_q.py", HERE / "refs_v.py",
            L.HF_DIR / "build.py", L.HF_DIR / "hf_common.py"]


# ---------------------------------------------------------------- L-4: open-path log
class PathLog:
    """L-4 (spec 8): input file paths opened by this process. install() adds a sys.addaudithook hook on the 'open'
    event (builtins / io.open, os.open, hence np.load, pandas read_csv, gzip, json files, torch.load, sha256 reads);
    note() records the paths this module hands to pyarrow (parquet reads open files in C++ without an audit event;
    every parquet read here is preceded by a sha256 read, which the hook sees anyway). Audit hooks cannot be removed;
    pause() / resume() stop / restart recording."""

    SYS_PREFIXES = ("/proc/", "/sys/", "/dev/", "/etc/", "/usr/", "/lib/", "/lib64/", "/run/")
    CODE_SUFFIXES = (".py", ".pyc", ".pyo", ".so", ".pth", ".pyd", ".pyi")

    def __init__(self):
        self.rec = []                         # (abs path, 'r' | 'w', source)
        self.installed = False
        self.active = True

    def install(self):
        if not self.installed:
            sys.addaudithook(self._hook)
            self.installed = True
        return self

    def pause(self):
        self.active = False

    def resume(self):
        self.active = True

    def _hook(self, event, args):
        if event != "open" or not self.active:
            return
        try:
            p = args[0]
            if p is None or isinstance(p, int):
                return
            p = p.decode(errors="replace") if isinstance(p, bytes) else os.fspath(p)
            mode = args[1] if len(args) > 1 else None
            flags = args[2] if len(args) > 2 else 0
            w = (isinstance(mode, str) and any(c in mode for c in "wax+")) or \
                (isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT)))
            self.rec.append((os.path.abspath(p), "w" if w else "r", "audit"))
        except Exception:                                              # noqa: BLE001  (never raise inside a hook)
            pass

    def note(self, path, source="explicit"):
        if self.active:
            self.rec.append((os.path.abspath(os.fspath(path)), "r", source))

    def _is_code(self, p):
        pre = tuple(os.path.abspath(x) + os.sep for x in {sys.prefix, sys.base_prefix, sys.exec_prefix})
        return (p.endswith(self.CODE_SUFFIXES) or f"{os.sep}__pycache__{os.sep}" in p or p.startswith(pre)
                or p.startswith(self.SYS_PREFIXES))

    def paths(self, data_only=True, mode="r"):
        """sorted unique paths ('r' reads, 'w' writes, None both); data_only drops code / interpreter / system files."""
        out = {p for p, m, _ in self.rec if (mode is None or m == mode) and not (data_only and self._is_code(p))}
        return sorted(out)

    def dump(self, path):
        return L.write_json(path, dict(reads=self.paths(), writes=self.paths(mode="w"), n_events=len(self.rec),
                                       installed=self.installed))


PATHLOG = PathLog()


def rung_roots():
    """(root, label) pairs for check_paths; label = the rungs whose column (spec 4-1) the root belongs to, or 'shared'
    (allowed for every rung). Longest matching root wins."""
    R = [
        # dft column
        (L.LAD / "dft_final", "dft"), (L.LAD / "fu2", "dft"), (L.RES / "fu2", "dft"), (RUNGS["dft"]["x182"], "dft"),
        (L.LAD / "F0", "dft"), (L.LAD / "G0", "dft"), (REFS_CR2, "dft"), (L.LAD / "V", "dft"),
        (L.HF_SCR / "runs" / "dft", "dft"), (L.HF_SCR / "runs" / "dft_orig1", "dft"), (DP.D1_JOBS, "dft"),
        (DP.D1_RUN, "dft"), (L.LAD / "labels_d1_cr.json", "dft"), (L.LAD / "labels_d1_cr2.json", "dft"),
        (L.LAD / "Q_keys.parquet", "dft"), (table_path("dft"), "dft"), (set_path("P_dft"), "dft"),
        # g1 column
        (DP.G1_ROOT_D1F, "g1"), (DP.G1_ROOT_D1, "g1"), (DP.G1_ROOT_D1R, "g1"), (DP.SCRATCH / "fu2", "g1"),
        (DP.RES / "fu2", "g1"), (DP.X182_D1, "g1"), (L.HF_SCR / "runs" / "g1", "g1"), (table_path("g1"), "g1"),
        (set_path("P_g1"), "g1"),
        # g2 column
        (L.G2, "g2"), (G2_RES, "g2"), (table_path("g2"), "g2"), (set_path("P_g2"), "g2"),
        # geo column (GEO2b PG-2): every geo input lives under $GEO2B, analysis/geo2/results or the GEO1b D0 tables
        (GEO2, "geo"), (GEO2_RES, "geo"), (GEO1B_D0, "geo"),
        # files holding more than one rung
        (L.D1HF_LABELS, "dft+g1"), (L.HF_SCR / "inputs", "dft+g1"), (L.HF_SCR / "labels_d1_hf.json", "dft+g1"),
        (set_path("P_01"), "dft+g1"), (set_path("P_012"), "dft+g1+g2"),
        # rung-independent training inputs
        (SPLITS_COPY, "shared"), (P_REPORT, "shared"), (S_FOLDS, "shared"), (ORIG_LABELS, "shared"),
        (F_PAIRS_CR, "shared"), (F_PAIRS_CR2, "shared"), (L.LAD / "units", "shared"), (L.LAD / "logs", "shared"),
        (L.RES, "shared"),
        # the FU2 T1 fold files and their index (pair-id lists, the same folds for every rung, spec 4-3): the trainers
        # read these originals for every rung (fix of review P-4 / M-1; exact files, the rest of the FU2 trees stay g1)
        (FU2_S_SPLITS, "shared"), (FU2_SPLITS / "pairs.parquet", "shared"),
        # label provenance reports read by every rung's trainer (V_report / Q_report outputs = the label files)
        (V_REPORT, "shared"), (L.RES / "Q_report.json", "shared"),
    ]
    R += [(FU2_SPLITS / f"T1__f{k}.json", "shared") for k in range(N_T1)]
    # results of the DFT-only phases (G0, F0, Q, V) and of the dft build: dft column, not 'shared' (fix of review P-4)
    R += [(L.RES / n, "dft") for n in ("G0_report.json", "G0_status.csv", "G0_steps", "F0_report.json",
                                       "F0_estimate.json", "F0_steps", "F0_tokens_manifest.csv", "Q_audit.csv",
                                       "Q6_azide_D_para.csv", "labels_d1_cr.parquet", "labels_d1_cr2.parquet",
                                       "V_conformers.csv", "V_cyclooctyne_conformers.csv", "V_decide.json",
                                       "V_molecules.csv", "V_summary.md", "V_targets.csv", "V_targets_report.json",
                                       "V_trial.csv")]
    R += [(build_report_path(r), r) for r in RUNG_ORDER] + [(pa_report_path(r), r) for r in RUNG_ORDER]
    return [(Path(os.path.abspath(p)), lab) for p, lab in R]


def check_paths(rung, paths):
    """L-4 helper: {ok, foreign: [(path, label)], unknown: [path], n}; foreign = a path whose longest matching root
    belongs to another rung's column only; ok needs no foreign and no unclassified path (fix of review P-4)."""
    roots = rung_roots()
    foreign, unknown = [], []
    for p in paths:
        q = Path(os.path.abspath(p))
        best = None
        for root, lab in roots:
            if q == root or root in q.parents:
                if best is None or len(str(root)) > len(str(best[0])):
                    best = (root, lab)
        if best is None:
            unknown.append(str(q))
        elif best[1] != "shared" and rung not in best[1].split("+"):
            foreign.append((str(q), best[1]))
    return dict(ok=not foreign and not unknown, rung=rung, n=len(paths), foreign=foreign, unknown=unknown)


# ---------------------------------------------------------------- read-only modules (build side)
def _Q():
    import refs_q as Q
    return Q


def _V():
    import refs_v as V
    return V


def _hf_modules():
    """(refs_q, hf_common, D1-HF build.py) - build.py imported once under the name d1hf_build (refs_q._load), its
    folder removed from sys.path again; hf_common must be the default D1-HF scratch."""
    Q = _Q()
    HC = Q.hf_common()
    if Path(os.path.abspath(HC.HF)) != Path(os.path.abspath(L.HF_SCR)):
        L.stop(f"hf_common.HF = {HC.HF} != {L.HF_SCR} (D1HF_SCRATCH must not be set for inputs.py)")
    d = L.HF_DIR
    B = Q._load("d1hf_build", d / "build.py")
    for p in (str(d), str(d.resolve())):
        while p in sys.path:
            sys.path.remove(p)
    if B.H is not HC:
        L.stop("analysis/d1_hf/build.py imported a hf_common other than refs_q.hf_common()")
    if not callable(getattr(B, "parse_rxn", None)) or not callable(getattr(B, "flat", None)):
        L.stop("build.py parse_rxn / flat not found (spec 4-1 assembly function)")
    if tuple(HC.CALCS) != ("frag1_dist", "frag2_dist", "frag1_rel", "frag2_rel", "eda") or HC.KCAL != KCAL:
        L.stop(f"hf_common CALCS {HC.CALCS} / KCAL {HC.KCAL} differ from what this adapter assumes")
    return Q, HC, B


class _Out(str):
    """an ORCA output text tagged with its calculation name (attribute .calc, e.g. 'frag1_rel')."""


class _FspeProxy:
    """hf_common.RE_FSPE during an adapter call: findall = the compiled pattern's, except for the outputs named in
    .rel ({'frag1_rel': Eh, ...}), for which it returns [repr(E)] (that output must carry exactly one FSPE)."""

    def __init__(self, orig):
        self.orig, self.rel, self.used = orig, {}, []

    def findall(self, text, *a, **k):
        res = self.orig.findall(text, *a, **k)
        c = getattr(text, "calc", None)
        if c in self.rel:
            if len(res) != 1:
                raise RuntimeError(f"{c}: {len(res)} FINAL SINGLE POINT ENERGY lines (need 1 to replace)")
            self.used.append(c)
            return [repr(float(self.rel[c]))]
        return res

    def __getattr__(self, name):
        return getattr(self.orig, name)


class _Adapter:
    """P2: build.py parse_rxn with only the reference energies replaced."""

    def __init__(self, B, HC):
        self.B, self.HC, self.cache, self.proxy = B, HC, {}, None

    @contextlib.contextmanager
    def bound(self, runs, inputs):
        HC = self.HC
        saved = (HC.read_text, HC.RE_FSPE, HC.RUNS, HC.INPUTS)
        orig_read = saved[0]

        def reader(path):
            key = str(path)
            if key not in self.cache:
                self.cache[key] = orig_read(path)
            t = _Out(self.cache[key])
            t.calc = Path(path).name.split(".")[0]
            return t

        self.proxy = _FspeProxy(saved[1])
        HC.read_text, HC.RE_FSPE, HC.RUNS, HC.INPUTS = reader, self.proxy, Path(runs), Path(inputs)
        try:
            yield self
        finally:
            HC.read_text, HC.RE_FSPE, HC.RUNS, HC.INPUTS = saved
            self.proxy = None

    def parse(self, T, geom, rel=None):
        """parse_rxn(T, geom) record; rel = {k: Eh} replaces E(frag{k}_rel)."""
        self.proxy.rel = {f"frag{int(k)}_rel": float(e) for k, e in (rel or {}).items()}
        self.proxy.used = []
        try:
            rec = self.B.parse_rxn(T, geom)
            if sorted(self.proxy.used) != sorted(self.proxy.rel):
                raise RuntimeError(f"{T}: replacements {sorted(self.proxy.rel)} consumed {self.proxy.used}")
            return rec
        finally:
            self.proxy.rel, self.proxy.used = {}, []


def _bookkeeping(c):
    return c.startswith("coreh_") or c == "n_attempts"


def _cmp_row(new, old, acc):
    """P-a: one flat(parse_rxn) record vs its labels_d1_hf row (both dicts)."""
    for c, b in old.items():
        a, nb = _norm(new.get(c)), _norm(b)
        if _isnum(a) and _isnum(nb):
            d = 0.0 if a == nb else abs(float(a) - float(nb))
            if not math.isfinite(d):
                d = float("inf")
            acc["float"][c] = max(acc["float"].get(c, 0.0), d)
        elif a != nb:
            acc["other"][c] += 1
    for c in set(new) - set(old):
        acc["extra"][c] += 1


def _upd(cur, x):
    """running maximum in which a non-finite value counts as infinite (a NaN never passes a tolerance)."""
    x = float(x)
    return float("inf") if not math.isfinite(x) else max(cur, x)


def _check_reref(T, rec0, rec1, ov, rr):
    """P4 checks of one replaced call against the unreplaced one."""
    de = {}
    for k in (1, 2):
        e0, e1 = rec0.get(f"e_frag{k}_rel_eh"), rec1.get(f"e_frag{k}_rel_eh")
        want = float(ov[k]) if k in ov else e0
        if e1 != want:
            rr["e_not_override"].append(f"{T}/frag{k}")
        de[k] = (e0 - e1) * KCAL
    for k in (1, 2):
        rr["d_max"] = _upd(rr["d_max"], abs((rec1[f"d{k}"] - rec0[f"d{k}"]) - de[k]))
    rr["barrier_max"] = _upd(rr["barrier_max"], abs((rec1["barrier"] - rec0["barrier"]) - (de[1] + de[2])))
    for c in ("identity_residual", "seam_residual"):
        rr["resid_max"] = _upd(rr["resid_max"], abs(rec1[c] - rec0[c]))
    for c in set(rec0) | set(rec1):
        if c not in REREF_FIELDS and _norm(rec1.get(c)) != _norm(rec0.get(c)):
            rr["other_changed"][c] += 1
    if rec1.get("status") != "ok":
        rr["status_bad"].append(f"{T}: {rec1.get('status')} {rec1.get('reason')}")
    rr["n"] += 1


def assemble_rung(rung, Hq, override, rxn_of, t0):
    """P-a on every labels_d1_hf(_g2) row of the rung + the replaced calls -> (pa, rr, Hn {job_id: {t: H}})."""
    cfg = RUNGS[rung]
    geom = cfg["geom"]
    _, HC, B = _hf_modules()
    acc = dict(float={}, other=Counter(), extra=Counter(), bad_status=[], errors=[])
    rr = dict(n=0, d_max=0.0, barrier_max=0.0, resid_max=0.0, other_changed=Counter(), e_not_override=[],
              status_bad=[], errors=[])
    unknown = sorted(set(override) - set(Hq.job_id))
    Hn = {}
    ad = _Adapter(B, HC)
    with ad.bound(cfg["hf_runs"], cfg["hf_inputs"]):
        for i, r0 in enumerate(Hq.to_dict(orient="records")):
            T = r0["job_id"]
            ad.cache.clear()
            try:
                rec0 = ad.parse(T, geom)
            except Exception as e:                                     # noqa: BLE001
                acc["errors"].append(f"{T}: {type(e).__name__}: {e}")
                continue
            if rec0.get("status") != "ok":
                acc["bad_status"].append(f"{T}: {rec0.get('status')} {rec0.get('reason')}")
            fl = B.flat(rec0)
            fl["rxn_id"] = rxn_of.get(T)                               # = build.py cmd_all (added before flat there)
            _cmp_row(fl, r0, acc)
            h = {t: _f(r0.get(t)) for t in H_TARGETS}
            ov = override.get(T) or {}
            if ov:
                try:
                    rec1 = ad.parse(T, geom, rel=ov)
                except Exception as e:                                 # noqa: BLE001
                    rr["errors"].append(f"{T}: {type(e).__name__}: {e}")
                    continue
                _check_reref(T, rec0, rec1, ov, rr)
                for t in H_REREF:
                    h[t] = _f(rec1.get(t))
            Hn[T] = h
            if i % 2000 == 0:
                _log(t0, f"{rung}: {i}/{len(Hq)} rows parsed")
    ad.cache.clear()
    gated = {c: v for c, v in acc["float"].items() if not _bookkeeping(c)}
    other = {c: n for c, n in acc["other"].items() if not _bookkeeping(c)}
    pa = dict(ok=bool(not acc["errors"] and not acc["bad_status"] and all(v <= TOL for v in gated.values())
                      and not other and len(Hq)),
              tol=TOL, source=str(cfg["h_parquet"]), geom=geom, n_rows=int(len(Hq)),
              n_parsed=int(len(Hq) - len(acc["errors"])), max_abs_gated=max(gated.values()) if gated else None,
              float_max_abs=gated, bookkeeping_max_abs={c: v for c, v in acc["float"].items() if _bookkeeping(c)},
              n_columns_other_mismatch=other, extra_columns=dict(acc["extra"]),
              n_bad_status=len(acc["bad_status"]), bad_status=acc["bad_status"][:30],
              n_errors=len(acc["errors"]), errors=acc["errors"][:30],
              h_max_abs={t: gated.get(t) for t in H_TARGETS})
    rr = dict(rr, other_changed=dict(rr["other_changed"]), n_override_jobs=len(override),
              override_jobs_without_h_row=unknown[:30], n_override_jobs_without_h_row=len(unknown))
    rr["ok"] = bool(rr["n"] == len(override) and not unknown and rr["d_max"] <= TOL and rr["barrier_max"] <= TOL
                    and rr["resid_max"] <= TOL and not rr["other_changed"] and not rr["e_not_override"]
                    and not rr["status_bad"] and not rr["errors"])
    rr["e_not_override"], rr["status_bad"], rr["errors"] = rr["e_not_override"][:30], rr["status_bad"][:30], \
        rr["errors"][:30]
    return pa, rr, Hn


# ---------------------------------------------------------------- inputs of a rung (build side)
def targets_of(rung):
    """PG-3: the target rows of a rung table, job_id order. L.targets() (the 9,266 D1 targets) unless the rung has a
    'targets' entry (geo: the combined D1 + D0 labels, D1 rows = L.targets() by (job_id, rxn_id), D0 job_id
    D0G2_<rid:04d>, rxn_id unique)."""
    cfg = RUNGS[rung]
    if "targets" not in cfg:
        return L.targets()
    p = Path(cfg["targets"])
    PATHLOG.note(p, f"{rung} targets")
    rows = json.loads(p.read_text())
    d1 = {(str(d["job_id"]), int(d["rxn_id"])) for d in L.targets()}
    got1 = {(str(d["job_id"]), int(d["rxn_id"])) for d in rows if int(d["rxn_id"]) > 1_000_000}
    d0 = [d for d in rows if int(d["rxn_id"]) <= 1_000_000]
    bad0 = [int(d["rxn_id"]) for d in d0 if str(d.get("job_id")) != f"D0G2_{int(d['rxn_id']):04d}"]
    ids = [int(d["rxn_id"]) for d in rows]
    if got1 != d1 or bad0 or len(set(ids)) != len(ids):
        raise _Stop(f"{p}: D1 rows != lad_common.targets() ({len(got1 ^ d1)} differ), D0 job_id rule broken "
                    f"{bad0[:5]}, duplicate rxn_id {len(ids) - len(set(ids))} (PG-3)")
    return sorted(rows, key=lambda d: d["job_id"])


def _load_keys(rung):
    """PG-4: the Q-2 key table of a rung: refs_q.load_keys() unless the rung has a 'keys' entry (geo: the D1 and D0
    key tables concatenated; key strings disjoint across the tables, (job_id, frag) unique)."""
    cfg = RUNGS[rung]
    if "keys" not in cfg:
        return _Q().load_keys()
    parts = []
    for p in cfg["keys"]:
        p = Path(p)
        PATHLOG.note(p, f"{rung} Q-2 keys")
        K = pd.read_parquet(p)
        need = ["job_id", "rxn_id", "frag", "kind", "key", "constrained", "formula", "wl_graph"]
        miss = [c for c in need if c not in K.columns]
        if miss or K[need].isna().any().any():
            raise _Stop(f"{p}: key columns missing {miss} or null values (PG-4)")
        if "dataset" not in K.columns:
            K["dataset"] = "D1" if Path(p) == Q_KEYS else Path(p).stem
        parts.append(K)
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            both = set(parts[i].key) & set(parts[j].key)
            if both:
                raise _Stop(f"Q-2 key tables {cfg['keys']} share {len(both)} keys (PG-4: keys are per dataset): "
                            f"{sorted(both)[:3]}")
    K = pd.concat(parts, ignore_index=True, sort=False)
    K["constrained"] = K.constrained.astype(bool)
    if K.duplicated(["job_id", "frag"]).any():
        raise _Stop(f"Q-2 key tables {cfg['keys']}: duplicate (job_id, frag) rows (PG-4)")
    return K


def _read_h(rung):
    """labels_d1_hf(_g2).parquet rows of the rung's geometry, job_id order (P3 source)."""
    cfg = RUNGS[rung]
    H = pd.read_parquet(cfg["h_parquet"])
    H = H[H.geom == cfg["geom"]].copy()
    need = ["job_id", "rxn_id", "status", "e_frag1_rel_eh", "e_frag2_rel_eh"] + H_TARGETS
    miss = [c for c in need if c not in H.columns]
    if not len(H) or miss:
        raise _Stop(f"{cfg['h_parquet']}: {len(H)} geom == {cfg['geom']} rows, missing columns {miss}")
    if H.job_id.duplicated().any():
        raise _Stop(f"{cfg['h_parquet']}: duplicate job_id in geom {cfg['geom']}")
    if (H.status != "ok").any():
        raise _Stop(f"{cfg['h_parquet']}: rows with status != ok in geom {cfg['geom']}")
    return H.sort_values("job_id").reset_index(drop=True)


def refs_dft(Hq, tj):
    """P5: replacements of the dft rung from $LAD/refs_cr2.parquet -> (override {job: {k: Eh}}, prov, report)."""
    R = pd.read_parquet(REFS_CR2)
    need = ["job_id", "frag", "key", "src", "src_job", "src_frag", "e_hf_eh"]
    miss = [c for c in need if c not in R.columns]
    if miss:
        raise _Stop(f"{REFS_CR2}: columns {miss} missing")
    e_par = {}
    for j, e1, e2 in zip(Hq.job_id, Hq.e_frag1_rel_eh, Hq.e_frag2_rel_eh):
        e_par[(j, 1)], e_par[(j, 2)] = _f(e1), _f(e2)
    want = {(j, k) for j in tj for k in (1, 2)}
    have = Counter((j, int(k)) for j, k in zip(R.job_id, R.frag))
    dup = [f"{j}/frag{k}" for (j, k), n in have.items() if n != 1]
    if dup or set(have) != want:
        raise _Stop(f"{REFS_CR2}: not every target fragment exactly once (dup {dup[:5]}, missing "
                    f"{len(want - set(have))}, extra {len(set(have) - want)})")
    ov, prov, bad = {}, {}, []
    for r in R.itertuples(index=False):
        j, k = r.job_id, int(r.frag)
        e0, e = e_par.get((j, k)), _f(r.e_hf_eh)
        if r.src == "own":
            if e is None or e != e0:
                bad.append(f"{j}/frag{k}: own e_hf {e} != labels_d1_hf {e0}")
            prov[(j, k)] = dict(src="own", src_job=j, src_frag=k, key=r.key, e_orig=e0, e_new=e0)
            continue
        if r.src == "Q":
            sj, sf = r.src_job, int(r.src_frag)
            es = e_par.get((sj, sf))
            if e is None or es is None or e != es:
                bad.append(f"{j}/frag{k}: Q e_hf {e} != labels_d1_hf of the source {sj}/frag{sf} {es}")
        elif r.src == "V":
            sj, sf = None, k
            if e is None:
                bad.append(f"{j}/frag{k}: V reference without its HF energy")
        else:
            bad.append(f"{j}/frag{k}: unknown src {r.src!r}")
            continue
        if e0 is None:
            bad.append(f"{j}/frag{k}: no labels_d1_hf dft value")
        ov.setdefault(j, {})[k] = e
        prov[(j, k)] = dict(src=r.src, src_job=sj, src_frag=sf, key=r.key, e_orig=e0, e_new=e)
    if bad:
        raise _Stop(f"refs_cr2.parquet HF energies inconsistent ({len(bad)}): {bad[:3]}")
    src = Counter(p["src"] for p in prov.values())
    d = [(p["e_orig"] - p["e_new"]) * KCAL for p in prov.values() if p["src"] != "own"]
    rep = dict(source=str(REFS_CR2), sha256=L.sha256(REFS_CR2), by_src=dict(src), n_jobs_replaced=len(ov),
               n_fragments_replaced=int(sum(len(v) for v in ov.values())),
               n_hf_up=int(sum(x < 0 for x in d)), n_hf_equal=int(sum(x == 0 for x in d)),
               dref_kcal_orig_minus_new=_q(d),
               note="dft rung H uses the HF energy of the Q / V final reference (spec 4-1); HF can be higher than the "
                    "row's own reference (HF and DFT orderings differ): n_hf_up")
    return ov, prov, rep


def ref_graphs(rung, A):
    """P6b (Q-2 on the rung's own references; fix of review P-2): the bond graph (refs_q.bond_graph) of every rung
    reference geometry with an energy = the frag{k}_rel input of its HF single point (runs/<geom>/<T>/frag{k}_rel.inp,
    the file build.py parse_rxn reads) must have the element composition and the WL graph hash of its Q-2 key (K formula
    / wl_graph of the same row = the D1 label reference that defines the molecule), and within each key be isomorphic to
    the first member (refs_q.check_graphs). Any mismatch = STOP (spec Q-2 '다르면 STOP')."""
    import networkx as nx
    Q = _Q()
    _, HC, _ = _hf_modules()
    cfg = RUNGS[rung]
    d = Path(cfg["hf_runs"]) / cfg["geom"]
    graphs, rows, bad = {}, [], []
    for j, k, f0, w0, key in zip(A.job_id, A.frag, A.formula, A.wl_graph, A.key):
        k = int(k)
        p = d / j / f"frag{k}_rel.inp"
        try:
            at = HC.parse_inp(p.read_text())["atoms"]
            syms, xyz = [a[0] for a in at], np.array([a[2:] for a in at], float)
            G = Q.bond_graph(syms, xyz)
        except Exception as e:                                         # noqa: BLE001
            bad.append(dict(row=f"{j}/frag{k}", key=key, why=f"{type(e).__name__}: {e}"))
            continue
        f, w = Q._fstr(Counter(syms)), nx.weisfeiler_lehman_graph_hash(G, node_attr="el")
        graphs[(j, k)] = G
        rows.append(dict(job_id=j, frag=k, key=key, formula=f, wl_graph=w))
        if f != f0 or w != w0:
            bad.append(dict(row=f"{j}/frag{k}", key=key, why="composition differs from the key" if f != f0
                            else "bond graph (WL hash) differs from the key's label reference"))
    within = Q.check_graphs(pd.DataFrame(rows), graphs) if rows else dict(ok=False, n_fail=0, fail=[])
    rep = dict(rule=DECISIONS["P6b_graphs"], n_checked=len(rows), n_vs_key_fail=len(bad), vs_key_fail=bad[:50],
               within_key=within, ok=bool(not bad and within["ok"] and len(rows) == len(A)))
    return rep


def _g2_mode(g2_q_keys):
    """P20 mode: False (none chosen), True (spec-literal Q-2 keys) or 'own' (option (c): constrained rows kept)."""
    return "own" if g2_q_keys == "own" else bool(g2_q_keys)


def refs_min(rung, Hq, rep_out=None, g2_q_keys=False):
    """P6: replacements of a g rung: per Q-2 key the lowest rung HF reference energy -> (override, prov, report).
    g2 with g2_q_keys == 'own' (P20 option (c)): rows in constrained keys are never replaced."""
    own = rung == "g2" and g2_q_keys == "own"
    Q = _Q()
    if Path(Q.LAD_KEYS) != Q_KEYS:
        raise _Stop(f"refs_q.LAD_KEYS {Q.LAD_KEYS} != {Q_KEYS}")
    K = _load_keys(rung)                                              # PG-4 (= Q.load_keys() but for geo)
    E = {}
    for j, e1, e2 in zip(Hq.job_id, Hq.e_frag1_rel_eh, Hq.e_frag2_rel_eh):
        for k, e in ((1, e1), (2, e2)):
            if _f(e) is not None:
                E[(j, k)] = float(e)
    missing = sorted({j for j, _ in E} - set(K.job_id))
    if missing:
        raise _Stop(f"{len(missing)} {rung} reactions have no Q-2 key in {Q_KEYS}: {missing[:5]}")
    A = Q.audit(K, E, key="key")
    n_cons = int(A.constrained.astype(bool).sum())
    if rung == "g2" and n_cons and not g2_q_keys:                     # P20 (fix of review P-3)
        raise _Stop(f"spec 16 '언제든' (P20): {n_cons} g2 reference rows sit in constrained-dipole Q-2 keys whose "
                    f"'|fixed:' suffix comes from the D1 (DFT) reference geometry, so grouping them would put DFT-only "
                    f"information into the g2 H values. Options: (a) Q-2 keys as the spec text says (rerun with "
                    f"--g2-q-keys), (b) SMILES-level key_plain for g2, (c) constrained g2 references never merged. "
                    f"User decision needed")
    gr = ref_graphs(rung, A)
    if rep_out is not None:
        rep_out["reference_graphs"] = gr
    if not gr["ok"]:
        raise _Stop(f"{rung}: {gr['n_vs_key_fail']} reference geometries differ from their Q-2 key "
                    f"(composition / bond graph), within-key check {gr['within_key'].get('n_fail')} failures "
                    f"(spec Q-2 STOP; P6b): {gr['vs_key_fail'][:3]}")
    ov, prov = {}, {}
    for r in A.itertuples(index=False):
        j, k = r.job_id, int(r.frag)
        rep_ = r.dev_kcal > 0 and not (own and bool(r.constrained))
        prov[(j, k)] = dict(src="min" if rep_ else "own", src_job=r.src_job if rep_ else j,
                            src_frag=int(r.src_frag) if rep_ else k, key=r.key, e_orig=float(r.e_eh),
                            e_new=float(r.e_min_eh) if rep_ else float(r.e_eh))
        if rep_:
            ov.setdefault(j, {})[k] = float(r.e_min_eh)
    rm = (A.dev_kcal > 0) & ~(A.constrained.astype(bool) & own)
    kept = (A.dev_kcal > 0) & A.constrained.astype(bool) & own      # P20 (c): above the key minimum, not replaced
    rule = DECISIONS["P6_g_refs"] + ("; P20 option (c) 'own': rows of constrained keys (Q_keys 'constrained' = D1 "
                                     "label ref_port flag) keep their own reference" if own else "")
    # dev_gt counts over the replaced rows (= every row with dev > 0 unless 'own'); kept-own rows counted separately
    rep = dict(source=str(Q_KEYS), sha256=L.sha256(Q_KEYS), rule=rule, g2_mode=_g2_mode(g2_q_keys),
               n_constrained_kept_own=int(kept.sum()) if own else 0,
               dev_kcal_kept_own=_q(A.dev_kcal[kept]) if own else None,
               n_fragments_with_energy=int(len(A)), n_keys=int(A.key.nunique()),
               n_keys_multi_row=int((A.groupby("key").size() >= 2).sum()),
               n_fragments_replaced=int(rm.sum()), n_jobs_replaced=len(ov),
               n_fragments_dev_gt_0_05=int((A.dev_kcal[rm] > 0.05).sum()),
               n_fragments_dev_gt_1=int((A.dev_kcal[rm] > 1.0).sum()),
               by_kind=_vc(A.kind[rm]), n_constrained_replaced=int((rm & A.constrained.astype(bool)).sum()),
               n_rows_constrained_keys=int(A.constrained.astype(bool).sum()),
               dev_kcal=_q(A.dev_kcal[rm]))
    return ov, prov, rep


def token_info(rung, tg1=None):
    """P7: the rung's FU2 tokens + X182 -> (tok {rxn_id: (path, sha256, n)}, X (index rxn_id), xcols, info)."""
    cfg = RUNGS[rung]
    res, tokdir = Path(cfg["fu2_res"]), Path(cfg["tokens"])
    tp, qp, mp, sp = res / "T_report.json", res / "Q_report.json", res / "T_manifest.csv", res / "S_splits.json"
    trep, qrep = json.loads(tp.read_text()), json.loads(qp.read_text())
    sha = {str(p): L.sha256(p) for p in (tp, qp, mp, sp, Path(cfg["x182"]))}
    fu2_ss = L.sha256(FU2_S_SPLITS)
    bad = []
    for cond, msg in ((not trep.get("stop"), "T_report stop"), (not qrep.get("stop"), "Q_report stop"),
                      (trep.get("token_version") == cfg["token_version"], f"token_version {trep.get('token_version')}"),
                      (qrep.get("atom_version") == cfg["atom_version"], f"atom_version {qrep.get('atom_version')}"),
                      (trep.get("manifest_sha256") == sha[str(mp)], "T_manifest.csv sha256 != T_report"),
                      (trep.get("q_report_sha256") == sha[str(qp)], "Q_report.json sha256 != T_report"),
                      (trep.get("x182_sha256") == sha[str(cfg["x182"])], "X182 sha256 != T_report"),
                      (trep.get("splits_sha256") == fu2_ss, "T_report splits_sha256 != FU2 S_splits.json"),
                      (sha[str(sp)] == fu2_ss, f"{sp} != FU2 S_splits.json")):
        if not cond:
            bad.append(msg)
    if tg1 is not None:
        for k in ("num_cols", "bin_cols", "glob_cols", "ext_used"):
            if trep.get(k) != tg1.get(k):
                bad.append(f"token layout {k} differs from the g1 rung (spec 4-2: same model inputs)")
    if bad:
        raise _Stop(f"{rung} tokens: {bad}")
    man = pd.read_csv(mp)
    X = pd.read_parquet(cfg["x182"])
    if "rxn_id" not in X.columns:
        raise _Stop(f"{cfg['x182']}: no rxn_id column")
    xcols = [c for c in X.columns if c != "rxn_id"]
    if len(xcols) != 182 or xcols != list(trep.get("glob_cols") or []):
        raise _Stop(f"{cfg['x182']}: {len(xcols)} columns, order == T_report glob_cols: {xcols == trep.get('glob_cols')}")
    clash = sorted(set(xcols) & set(META_COLS + H_COLS + H0_COLS + PROV_COLS + ["h_changed"]))
    if clash:
        raise _Stop(f"X182 column names collide with the table columns: {clash}")
    X["rxn_id"] = X.rxn_id.astype(int)
    if X.rxn_id.duplicated().any():
        raise _Stop(f"{cfg['x182']}: duplicate rxn_id")
    X = X.set_index("rxn_id")
    xv = X[xcols].to_numpy(float)
    pos = {int(r): i for i, r in enumerate(X.index)}
    tok, sbad, gbad = {}, [], []
    for rid, s0, n in zip(man.rxn_id.astype(int), man.sha256, man.n_atoms.astype(int)):
        p = tokdir / f"{rid}.npz"
        s = _sha(p)
        if s != s0:
            sbad.append(int(rid))
            continue
        with np.load(p) as z:
            g, nz = np.asarray(z["glob"], float), len(z["Z"])
        if rid not in pos or nz != n or not np.array_equal(g, xv[pos[rid]]):
            gbad.append(int(rid))
        tok[int(rid)] = (str(p), s, int(n))
    if sbad or gbad:
        raise _Stop(f"{rung} tokens: {len(sbad)} npz sha256 != T_manifest {sbad[:5]}, {len(gbad)} glob != X182 row / "
                    f"token count != n_atoms {gbad[:5]}")
    info = dict(fu2_res=str(res), tokens=str(tokdir), t_report=str(tp), t_report_sha256=sha[str(tp)],
                q_report_sha256=sha[str(qp)], manifest_sha256=sha[str(mp)], s_splits_sha256=sha[str(sp)],
                x182=str(cfg["x182"]), x182_sha256=sha[str(cfg["x182"])], token_version=trep.get("token_version"),
                atom_version=qrep.get("atom_version"), n_tokens=len(tok), n_x182_rows=int(len(X)),
                elements=trep.get("elements"), num_cols=trep.get("num_cols"), bin_cols=trep.get("bin_cols"),
                ext_used=trep.get("ext_used"), elements_same_as_g1=None if tg1 is None else
                trep.get("elements") == tg1.get("elements"))
    return tok, X, xcols, info


def _q_ready(rep):
    why = _V().q_ready(rep)
    return (3, f"Phase Q not ready: {why}") if why else (0, "")


def _v_ready(rep):
    V = _V()
    if Path(V.REFS_CR2) != REFS_CR2 or Path(V.RES_REPORT) != V_REPORT:
        return 3, f"refs_v paths {V.REFS_CR2} / {V.RES_REPORT} != {REFS_CR2} / {V_REPORT}"
    if not V_REPORT.is_file():
        return 2, "results/V_report.json missing (Phase V collect has not passed)"
    v = json.loads(V_REPORT.read_text())
    rep["v_report_sha256"] = L.sha256(V_REPORT)
    if not v.get("all_pass") or v.get("stop"):
        return (3 if v.get("stop") else 2), f"Phase V not passed (all_pass {v.get('all_pass')}, stop {v.get('stop')})"
    outs = v.get("outputs") or {}
    miss = [str(p) for p in (REFS_CR2, F_PAIRS_CR2) if str(p) not in outs]
    bad = [p for p, s in outs.items() if not Path(p).is_file() or L.sha256(p) != s]
    if miss or bad:
        return 3, f"V_report outputs: missing {miss}, changed {bad[:5]}"
    return 0, ""


def _f0_ready(rep, accept):
    f0 = json.loads(F0_REPORT.read_text()) if F0_REPORT.is_file() else {}
    g0 = json.loads(G0_REPORT.read_text()) if G0_REPORT.is_file() else {}
    rep["f0"] = dict(sha256=_sha(F0_REPORT), complete=f0.get("complete"), stop=f0.get("stop"),
                     stop_reasons=f0.get("stop_reasons"), counts=f0.get("counts"), accept_f0_stop=bool(accept))
    rep["g0"] = dict(sha256=_sha(G0_REPORT), complete=g0.get("complete"), stop=g0.get("stop"))
    if not f0.get("complete") or not g0.get("complete"):
        return 2, f"G0 / F0 not complete (G0 {g0.get('complete')}, F0 {f0.get('complete')})"
    if g0.get("stop"):
        return 3, f"G0_report stop: {g0.get('stop_reasons')}"
    if f0.get("stop") and not accept:
        return 3, (f"F0_report.json stop {f0.get('stop_reasons')} (spec 16 F0 STOP): resolve it with the user; "
                   f"--accept-f0-stop only after that decision")
    rep["f0"]["f0_stop_accepted"] = bool(f0.get("stop") and accept)
    return 0, ""


G2_F_DECISION = G2_RES / "F_DECISION.md"            # D1-G2 F gate STOP accepted as a deviation (2026-10-23)


def _g2_ready(rep, accept=False):
    """accept (--accept-g2f-stop, decision 2026-10-23: every judgment = the recommended option): a D1-G2 F_report STOP
    whose only reason is the F gate failure rate passes when G2_F_DECISION is committed; recorded in g2_gates."""
    fr, hr = G2_RES / "F_report.json", G2_RES / "H_report.json"
    info = {}
    for p in (fr, hr):
        if not p.is_file():
            return 2, f"{p} missing (D1-G2 Phase F / H not finished)"
        r = json.loads(p.read_text())
        info[p.name] = dict(complete=r.get("complete"), stop=r.get("stop"), sha256=L.sha256(p))
        reasons = [str(x) for x in (r.get("stop_reasons") or [])]
        if p == fr and r.get("complete") and r.get("stop") and accept and reasons and \
                all(x.startswith("F gate failure rate") for x in reasons) and G2_F_DECISION.is_file() and \
                _committed(G2_F_DECISION):
            info["F_stop_accepted"] = dict(stop_reasons=reasons, decision=str(G2_F_DECISION),
                                           decision_sha256=L.sha256(G2_F_DECISION))
            continue
        if not r.get("complete") or r.get("stop"):
            return (3 if r.get("stop") else 2), f"{p.name}: complete {r.get('complete')}, stop {r.get('stop')}"
        if p == hr:
            # D1-G2 keys its outputs by the absolute path of the worktree that wrote them (eda-asm-prediction-d1g2):
            # match the entry by its repo-relative path (exactly one), then compare sha256 (fix of review P-1)
            outs = r.get("outputs") or {}
            hp = Path(RUNGS["g2"]["h_parquet"])
            rel = str(hp.relative_to(L.REPO))
            hit = [k for k in outs if k == str(hp) or k.endswith(os.sep + rel)]
            info["H_report_output_key"] = hit
            if len(hit) != 1:
                return 3, f"H_report outputs: {len(hit)} entries end with {rel} (need exactly one): {hit[:3]}"
            if outs[hit[0]] != _sha(hp):
                return 3, f"H_report outputs: {hit[0]} sha256 differs from {hp}"
    files = [fr, hr, RUNGS["g2"]["h_parquet"], G2_RES / "fu2" / "T_report.json", G2_RES / "fu2" / "T_manifest.csv"]
    unc = [str(p) for p in files if not _committed(p)]
    info["not_committed"] = unc
    rep["g2_gates"] = info
    if unc:
        return 2, f"D1-G2 outputs not committed / modified (spec 12): {unc}"
    return 0, ""


def _needs(rung):
    cfg = RUNGS[rung]
    res = Path(cfg["fu2_res"])
    p = [cfg["h_parquet"], cfg["x182"], res / "T_report.json", res / "Q_report.json", res / "T_manifest.csv",
         res / "S_splits.json", cfg["tokens"], Path(cfg["hf_runs"]) / cfg["geom"], FU2_S_SPLITS]
    if cfg["refs"] == "cr2":
        p += [F0_REPORT, G0_REPORT]
    else:
        p += [Q_KEYS]
    return [Path(x) for x in p]


def _fingerprint(rung, accept, g2_q_keys=False):
    cfg = RUNGS[rung]
    res = Path(cfg["fu2_res"])
    files = _code_files() + [cfg["h_parquet"], cfg["x182"], res / "T_report.json", res / "Q_report.json",
                             res / "T_manifest.csv", res / "S_splits.json", FU2_S_SPLITS]
    if cfg["refs"] == "cr2":
        files += [REFS_CR2, V_REPORT, F0_REPORT, G0_REPORT, L.RES / "Q_report.json"]
    else:
        files += [Q_KEYS, L.RES / "Q_report.json"]
    if rung == "g2":
        files += [G2_RES / "F_report.json", G2_RES / "H_report.json"]
    fp = {str(p): _sha(p) for p in files}
    fp["accept_f0_stop"] = bool(accept) if rung in ("dft", "g2") else None    # g2: --accept-g2f-stop
    fp["g2_q_keys"] = _g2_mode(g2_q_keys) if rung == "g2" else None
    fp["code_version"] = CODE_VERSION
    return fp


def _ready(rung, rep, accept):
    if rung == "dft":
        for c, why in (_q_ready(rep), _v_ready(rep), _f0_ready(rep, accept)):
            if c:
                return c, why
    else:
        c, why = _q_ready(rep)
        if c:
            return c, why
        if rung == "g2":
            return _g2_ready(rep, accept)
    return 0, ""


def make_table(rung, Hq, Hn, prov, tok, X, xcols):
    """the rung input table (P7 flags, 182 columns, H / H0, provenance), rxn_id order."""
    h0 = {j: {t: _f(v) for t, v in zip(H_TARGETS, vals)} for j, *vals in
          zip(Hq.job_id, *[Hq[t] for t in H_TARGETS])}
    xfin = pd.Series(np.isfinite(X[xcols].to_numpy(float)).all(1), index=X.index)
    rows = []
    for d in targets_of(rung):                                         # PG-3 (= L.targets() but for geo)
        j, rid = d["job_id"], int(d["rxn_id"])
        t = tok.get(rid)
        r = dict(rxn_id=rid, job_id=j, has_tok=t is not None, tok_path=t[0] if t else "",
                 tok_sha256=t[1] if t else "", tok_n_atoms=t[2] if t else -1,
                 has_x182=bool(rid in xfin.index and xfin.loc[rid]))
        hn, ho = Hn.get(j), h0.get(j)
        r["has_h"] = bool(hn is not None and ho is not None and all(hn[x] is not None for x in H_TARGETS)
                          and all(ho[x] is not None for x in H_TARGETS))
        r["usable"] = bool(r["has_tok"] and r["has_x182"] and r["has_h"])
        for x in H_TARGETS:
            r[f"H_{x}"] = hn[x] if hn and hn[x] is not None else np.nan
            r[f"H0_{x}"] = ho[x] if ho and ho[x] is not None else np.nan
        r["h_changed"] = bool(r["has_h"] and any(r[f"H_{x}"] != r[f"H0_{x}"] for x in H_REREF))
        for k in (1, 2):
            p = prov.get((j, k)) or {}
            r.update({f"ref{k}_src": p.get("src", ""), f"ref{k}_src_job": p.get("src_job") or "",
                      f"ref{k}_src_frag": int(p["src_frag"]) if p.get("src_frag") is not None else -1,
                      f"ref{k}_key": p.get("key") or "",
                      f"e_ref{k}_hf_orig_eh": p["e_orig"] if p.get("e_orig") is not None else np.nan,
                      f"e_ref{k}_hf_eh": p["e_new"] if p.get("e_new") is not None else np.nan})
        rows.append(r)
    M = pd.DataFrame(rows).sort_values("rxn_id").reset_index(drop=True)
    Xr = X.reindex(M.rxn_id.to_numpy())[xcols].reset_index(drop=True)
    T = pd.concat([M[META_COLS], Xr, M[H_COLS + H0_COLS + ["h_changed"] + PROV_COLS]], axis=1)
    return T


def table_summary(T, prov):
    tok_no_x = T[T.has_tok & ~T.has_x182]
    hc = {}
    for t in H_REREF:
        d = (T[f"H_{t}"] - T[f"H0_{t}"]).abs()
        m = T.has_h
        hc[t] = dict(n_changed=int((m & (d > 0)).sum()), n_gt_1e9=int((m & (d > TOL)).sum()),
                     n_gt_0_05=int((m & (d > 0.05)).sum()), n_changed_usable=int((T.usable & (d > 0)).sum()),
                     abs_change=_q(d[m & (d > 0)]))
    ch_same = {t: bool(((T[f"H_{t}"] == T[f"H0_{t}"]) | (T[f"H_{t}"].isna() & T[f"H0_{t}"].isna())).all())
               for t in CH6}
    kinds = Counter()
    for (j, k), p in prov.items():
        if p.get("src") not in ("own", None) and p.get("e_new") != p.get("e_orig"):
            kinds[str(p.get("key") or "").split(":", 1)[0]] += 1
    return dict(n_targets=int(len(T)), n_tok=int(T.has_tok.sum()), n_x182=int(T.has_x182.sum()),
                n_h=int(T.has_h.sum()), n_usable=int(T.usable.sum()),
                n_tok_without_h=int((T.has_tok & ~T.has_h).sum()), n_tok_without_x182=int(len(tok_no_x)),
                tok_without_x182=tok_no_x.rxn_id.tolist()[:30],
                h_changed=dict(n_reactions=int(T.h_changed.sum()), n_reactions_usable=int((T.h_changed & T.usable).sum()),
                               by_value=hc, n_fragments_changed_by_kind=dict(kinds)),
                channels_unchanged=ch_same)


# ---------------------------------------------------------------- commands: build / pa
def _stop_report(path, rep, msg, code=3, stale=()):
    rep.update(stop=code == 3, all_pass=False, stop_reason=msg, finished=_now(), running=False)
    moved = []
    for p in stale:
        p = Path(p)
        if p.is_file():
            q = L.assert_path(p)
            os.replace(q, q.with_name(q.name + ".stale"))
            moved.append(str(q))
    rep["invalidated"] = moved
    L.write_json(path, rep)
    print(f"{'STOP' if code == 3 else 'NOT READY'}: {msg}", file=sys.stderr, flush=True)
    return code


def _base_report(step, rung=None):
    return dict(phase="P", step=step, rung=rung, code_version=CODE_VERSION, started=_now(),
                git_head=L.git("rev-parse", "HEAD").strip(), env=_env(), decisions=DECISIONS,
                cfg={k: str(v) for k, v in RUNGS[rung].items()} if rung else None, stop=False, all_pass=False)


def cmd_build(rung, force=False, accept_f0_stop=False, g2_q_keys=False):
    t0 = time.time()
    out, rp = table_path(rung), build_report_path(rung)
    miss = [str(p) for p in _needs(rung) if not p.exists()]
    if miss:
        print(f"NOT READY: {rung} inputs missing: {miss}", file=sys.stderr)
        return 2
    rep = dict(phase="P", step=f"build {rung}", rung=rung, code_version=CODE_VERSION, started=_now(), stop=False,
               all_pass=False)
    try:                                       # readiness inside the STOP-report path too (fix of review P-7)
        rep = _base_report(f"build {rung}", rung)
        code, why = _ready(rung, rep, accept_f0_stop)
        fp = None if code else _fingerprint(rung, accept_f0_stop, g2_q_keys)
    except SystemExit as e:
        return _stop_report(rp, rep, f"exit {e.code} in the readiness checks (message in the job log)", 3)
    except Exception as e:                                             # noqa: BLE001
        traceback.print_exc()
        return _stop_report(rp, rep, f"unexpected error in the readiness checks {type(e).__name__}: {e}", 3)
    if code:
        print(f"{'STOP' if code == 3 else 'NOT READY'}: {why}", file=sys.stderr)
        if code == 3:
            return _stop_report(rp, rep, why, 3)
        return 2
    rep["g2_q_keys"] = _g2_mode(g2_q_keys) if rung == "g2" else None
    if not force and rp.is_file():
        try:
            o = json.loads(rp.read_text())
        except ValueError:
            o = {}
        if o.get("all_pass") and o.get("fingerprint") == fp and out.is_file() and \
                (o.get("outputs") or {}).get(str(out)) == L.sha256(out):
            print(f"{rung} input table up to date ({out}); --force to rebuild", flush=True)
            return 0
    rep["fingerprint"] = fp
    L.write_json(rp, dict(rep, running=True))
    try:
        T = _build(rung, rep, t0, g2_q_keys)
    except _Stop as e:
        return _stop_report(rp, rep, str(e), 3, stale=[out])
    except SystemExit as e:
        return _stop_report(rp, rep, f"exit {e.code} inside a helper (message in the job log)", 3, stale=[out])
    except Exception as e:                                             # noqa: BLE001
        traceback.print_exc()
        return _stop_report(rp, rep, f"unexpected error {type(e).__name__}: {e} (traceback in the job log)", 3,
                            stale=[out])
    L.write_atomic(out, lambda t: T.to_parquet(t, index=False))
    rep.update(outputs={str(out): L.sha256(out)}, all_pass=True, stop=False, running=False, finished=_now(),
               wall_s=round(time.time() - t0, 1))
    L.write_json(rp, rep)
    s = rep["table"]
    _log(t0, f"{rung}: table {out} ({s['n_usable']} usable of {s['n_targets']}; H changed in "
             f"{s['h_changed']['n_reactions']} reactions)")
    return 0


def _build(rung, rep, t0, g2_q_keys=False):
    cfg = RUNGS[rung]
    if cfg["refs"] == "cr2" and Path(_V().REFS_CR2) != REFS_CR2:
        raise _Stop("refs_v.REFS_CR2 differs from inputs.REFS_CR2")
    targets = L.targets()
    tj = {d["job_id"]: int(d["rxn_id"]) for d in targets}
    Hq = _read_h(rung)
    extra = sorted(set(Hq.job_id) - set(tj))
    if extra:
        raise _Stop(f"{len(extra)} {cfg['h_parquet']} rows are not targets: {extra[:5]}")
    rep["h_source"] = dict(path=str(cfg["h_parquet"]), sha256=L.sha256(cfg["h_parquet"]), n_rows=int(len(Hq)))
    _log(t0, f"{rung}: {len(Hq)} H rows; references ({cfg['refs']})")
    if cfg["refs"] == "cr2":
        ov, prov, rrep = refs_dft(Hq, tj)
    else:
        ov, prov, rrep = refs_min(rung, Hq, rep, g2_q_keys)
    rep["references"] = rrep
    _log(t0, f"{rung}: {len(ov)} reactions with a replaced reference; P-a + re-assembly")
    pa, rr, Hn = assemble_rung(rung, Hq, ov, tj, t0)
    rep["P-a"], rep["reassembly"] = pa, rr
    _log(t0, f"{rung}: P-a ok={pa['ok']} (max {pa['max_abs_gated']}), re-assembly ok={rr['ok']}")
    if not pa["ok"]:
        raise _Stop(f"gate P-a failed for {rung} (spec 16 P): see P_build_{rung}.json 'P-a'")
    if not rr["ok"]:
        raise _Stop(f"re-assembly checks failed for {rung}: see P_build_{rung}.json 'reassembly'")
    tg1 = None
    if rung != "g1":
        tg1 = json.loads((Path(RUNGS["g1"]["fu2_res"]) / "T_report.json").read_text())
    _log(t0, f"{rung}: tokens + X182")
    tok, X, xcols, tinfo = token_info(rung, tg1)
    rep["tokens"] = tinfo
    T = make_table(rung, Hq, Hn, prov, tok, X, xcols)
    s = table_summary(T, prov)
    rep["table"] = s
    rep["x182_cols"] = xcols
    if s["n_tok_without_x182"]:
        raise _Stop(f"{s['n_tok_without_x182']} {rung} token reactions without an all-finite X182 row "
                    f"{s['tok_without_x182'][:5]}")
    if not all(s["channels_unchanged"].values()):
        raise _Stop(f"H channels changed by the re-referencing: {s['channels_unchanged']}")
    if s["n_usable"] == 0:
        raise _Stop(f"{rung}: no usable reaction")
    return T


def cmd_pa(rung):
    t0 = time.time()
    rp = pa_report_path(rung)
    cfg = RUNGS[rung]
    if not Path(cfg["h_parquet"]).is_file() or not (Path(cfg["hf_runs"]) / cfg["geom"]).is_dir():
        print(f"NOT READY: {cfg['h_parquet']} / {Path(cfg['hf_runs']) / cfg['geom']} missing", file=sys.stderr)
        return 2
    rep = dict(phase="P", step=f"pa {rung}", rung=rung, code_version=CODE_VERSION, started=_now(), stop=False,
               all_pass=False)
    try:
        rep = _base_report(f"pa {rung}", rung)
        tj = {d["job_id"]: int(d["rxn_id"]) for d in L.targets()}
        Hq = _read_h(rung)
        pa, _, _ = assemble_rung(rung, Hq, {}, tj, t0)
    except _Stop as e:
        return _stop_report(rp, rep, str(e))
    except SystemExit as e:
        return _stop_report(rp, rep, f"exit {e.code} inside a helper (message in the job log)")
    except Exception as e:                                             # noqa: BLE001
        traceback.print_exc()
        return _stop_report(rp, rep, f"unexpected error {type(e).__name__}: {e}")
    rep.update({"P-a": pa, "h_source": dict(path=str(cfg["h_parquet"]), sha256=L.sha256(cfg["h_parquet"])),
                "all_pass": pa["ok"], "stop": not pa["ok"], "finished": _now(), "wall_s": round(time.time() - t0, 1)})
    L.write_json(rp, rep)
    _log(t0, f"P-a {rung}: ok={pa['ok']} max {pa['max_abs_gated']} over {pa['n_rows']} rows")
    return 0 if pa["ok"] else 3


# ---------------------------------------------------------------- command: pairs
def _build_ok(rung):
    """(ok, build report | reason, code): code 0 = passed and current, 2 = not built / running / stale (rebuild), 3 = the
    build ended in a STOP or its table changed afterwards. Current = the build fingerprint (inputs + code sha256)
    recomputed now equals the recorded one (fix of review P-6 / P-8)."""
    rp, out = build_report_path(rung), table_path(rung)
    if not rp.is_file():
        return False, f"{rp.name} missing (run inputs.py build {rung})", 2
    br = json.loads(rp.read_text())
    if not br.get("all_pass"):
        return (False, f"{rp.name}: all_pass false (stop {br.get('stop')}: {br.get('stop_reason')})",
                3 if br.get("stop") else 2)
    want = (br.get("outputs") or {}).get(str(out))
    if not out.is_file() or want != L.sha256(out):
        return False, f"{out} missing or changed since {rp.name}", 3
    fp0 = br.get("fingerprint") or {}
    fp1 = _fingerprint(rung, fp0.get("accept_f0_stop"), fp0.get("g2_q_keys"))
    if fp1 != fp0:
        diff = sorted(k for k in set(fp0) | set(fp1) if fp0.get(k) != fp1.get(k))
        return False, f"{rp.name}: inputs / code changed since the build (stale; rerun inputs.py build {rung}): {diff}", 2
    return True, br, 0


def _reason(row):
    miss = [x for x, ok in (("no_tokens", row.has_tok), ("no_x182", row.has_x182), ("no_h", row.has_h)) if not ok]
    return "+".join(miss) if miss else "ok"


def _copy_folds(S):
    """P11: byte copies of the FU2 T1 split files + pairs.parquet, checked against S_folds -> (info, lists)."""
    src = json.loads(FU2_S_SPLITS.read_text())
    info = dict(source=str(FU2_SPLITS), s_splits=str(FU2_S_SPLITS), s_splits_sha256=L.sha256(FU2_S_SPLITS),
                dir=str(SPLITS_COPY), files={}, checks={})
    for n in ["pairs.parquet"] + [f"T1__f{k}.json" for k in range(N_T1)]:
        s = FU2_SPLITS / n
        want = src["pairs_sha256"] if n == "pairs.parquet" else src["units"][n]["sha256"]
        if L.sha256(s) != want:
            raise _Stop(f"{s} sha256 != FU2 S_splits.json")
        d = SPLITS_COPY / n
        if not d.is_file() or L.sha256(d) != want:
            L.write_atomic(d, s.read_bytes())
        if L.sha256(d) != want:
            raise _Stop(f"copy {d} sha256 != {want}")
        info["files"][str(d)] = want
    core = dict(zip(S.pair_id, S.core))
    lists, bad = {}, []
    for k in range(N_T1):
        rec = json.loads((SPLITS_COPY / f"T1__f{k}.json").read_text())
        tr = set(S.pair_id[(S.fold_T1 != k) & S.in_T_main])
        te = set(S.pair_id[(S.fold_T1 == k) & (S.in_P_sens | S.in_A_main)])
        it, iv = set(rec["inner_train"]), set(rec["inner_val"])
        c = dict(train_eq_masks=set(rec["train"]) == tr, test_eq_masks=set(rec["test"]) == te,
                 no_duplicates=all(len(rec[x]) == len(set(rec[x])) for x in ("train", "test", "inner_train", "inner_val")),
                 inner_partition=(it | iv) == set(rec["train"]) and not (it & iv),
                 inner_cores_disjoint=not ({core[p] for p in it} & {core[p] for p in iv}),
                 n=dict(train=len(rec["train"]), test=len(rec["test"]), inner_train=len(it), inner_val=len(iv)))
        info["checks"][f"T1__f{k}"] = c
        if not all(v for x, v in c.items() if x != "n"):
            bad.append(f"T1__f{k}: {c}")
        lists[k] = rec
    if bad:
        raise _Stop(f"FU2 T1 split files disagree with S_folds.csv: {bad[:2]}")
    return info, lists


def _set_summary(S, m, lists):
    sub = S[m]
    para = sub.pos == "p"
    d = dict(n=int(m.sum()), by_ref_kind=_vc(sub.ref_kind), para=int(para.sum()), para_by_ref_kind=_vc(sub.ref_kind[para]),
             by_pos=_vc(sub.pos), in_P_main=int(sub.in_P_main.sum()), in_P_main_para=int((sub.in_P_main & para).sum()),
             in_P_sens=int(sub.in_P_sens.sum()), in_A_main=int(sub.in_A_main.sum()), in_T_main=int(sub.in_T_main.sum()))
    keep = set(sub.pair_id)
    fo = {}
    for k, rec in lists.items():
        f = {x: int(sum(p in keep for p in rec[x])) for x in ("train", "test", "inner_train", "inner_val")}
        f["dropped_from_fu2_lists"] = {x: len(rec[x]) - f[x] for x in ("train", "test", "inner_train", "inner_val")}
        tm = (sub.fold_T1 == k) & sub.in_P_main
        f["test_P_main"], f["test_P_main_para"] = int(tm.sum()), int((tm & para).sum())
        f["test_anchor"] = int(((sub.fold_T1 == k) & sub.in_A_main).sum())
        fo[str(k)] = f
    d["folds_T1"] = fo
    return d


def _dropped(S, m, reason_of):
    sub = S[~m]
    used = (sub.in_T_main | sub.in_P_sens | sub.in_A_main).to_numpy()
    rs = [reason_of(a, b) for a, b in zip(sub.a_rxn, sub.b_rxn)]
    rows = [dict(pair_id=p, ref_kind=rk, pos=po, used_in_T1=bool(u), reason=r)
            for p, rk, po, u, r in zip(sub.pair_id, sub.ref_kind, sub.pos, used, rs)]
    return dict(n=len(rows), by_reason=dict(Counter(rs)), n_used_in_T1=int(used.sum()),
                used_by_reason=dict(Counter(r for r, u in zip(rs, used) if u)), pairs=rows)


def cmd_pairs():
    t0 = time.time()
    rungs, brs, skipped = [], {}, {}
    for r in RUNG_ORDER:
        ok, br, code = _build_ok(r)
        if ok:
            rungs.append(r)
            brs[r] = br
        elif r in ("dft", "g1"):
            print(f"{'STOP' if code == 3 else 'NOT READY'}: {r}: {br}", file=sys.stderr)
            return code                                # a build STOP stays a STOP (exit 3), fix of review P-6
        else:
            skipped[r] = dict(reason=br, code=code)
            print(f"note: {r} not used in the pair sets: {br}", file=sys.stderr)
    rep = _base_report("pairs")
    rep["rungs_built"] = rungs
    rep["rungs_not_built"] = skipped
    L.write_json(P_REPORT, dict(rep, running=True))
    try:
        return _pairs(rep, rungs, brs, t0)
    except _Stop as e:
        return _stop_report(P_REPORT, rep, str(e))
    except SystemExit as e:
        return _stop_report(P_REPORT, rep, f"exit {e.code} inside a helper (message in the job log)")
    except Exception as e:                                             # noqa: BLE001
        traceback.print_exc()
        return _stop_report(P_REPORT, rep, f"unexpected error {type(e).__name__}: {e} (traceback in the job log)")


def _pairs(rep, rungs, brs, t0):
    stop = []
    S = pd.read_csv(S_FOLDS)                                           # = d1_train.data()
    rep["inputs"] = {str(S_FOLDS): L.sha256(S_FOLDS)}
    if S.pair_id.duplicated().any():
        raise _Stop("S_folds.csv pair_id not unique")
    # rung tables
    tabs, U, why = {}, {}, {}
    rep["rungs"] = {}
    for r in rungs:
        T = pd.read_parquet(table_path(r))
        tabs[r] = T
        U[r] = set(T.rxn_id[T.usable].astype(int))
        why[r] = {int(x.rxn_id): _reason(x) for x in T[["rxn_id", "has_tok", "has_x182", "has_h"]].itertuples(index=False)}
        br = brs[r]
        rep["rungs"][r] = dict(table=str(table_path(r)), table_sha256=L.sha256(table_path(r)),
                               build_report=str(build_report_path(r)), build_report_sha256=L.sha256(build_report_path(r)),
                               x182_cols=br["x182_cols"], tokens=br["tokens"], counts=br["table"],
                               references=br["references"], P_a_max_abs=br["P-a"]["max_abs_gated"],
                               reassembly={k: br["reassembly"].get(k) for k in ("n", "d_max", "barrier_max", "resid_max")},
                               f0=br.get("f0"))

    def end_reason(r):
        def f(a, b):
            ra, rb = why[r].get(int(a), "not_target"), why[r].get(int(b), "not_target")
            return f"a:{ra};b:{rb}"
        return f
    inset = {r: (S.a_rxn.isin(U[r]) & S.b_rxn.isin(U[r])).to_numpy() for r in rungs}
    sets = {"P_dft": inset["dft"], "P_g1": inset["g1"], "P_01": inset["dft"] & inset["g1"]}
    if "g2" in rungs:
        sets["P_g2"] = inset["g2"]
        sets["P_012"] = sets["P_01"] & inset["g2"]
    # folds (P11)
    finfo, lists = _copy_folds(S)
    rep["folds"] = finfo
    # FU2 check (P8 / P9)
    FP = pd.read_parquet(FU2_SPLITS / "pairs.parquet")
    fsrep = json.loads(FU2_S_SPLITS.read_text())
    fman = pd.read_csv(FU2_RES_G1 / "T_manifest.csv")
    ftrep = json.loads((FU2_RES_G1 / "T_report.json").read_text())
    same_univ = bool(len(FP) == len(S) and FP.pair_id.tolist() == S.pair_id.tolist()
                     and (FP.a_rxn.astype(int).to_numpy() == S.a_rxn.astype(int).to_numpy()).all()
                     and (FP.b_rxn.astype(int).to_numpy() == S.b_rxn.astype(int).to_numpy()).all()
                     and FP.ref_kind.tolist() == S.ref_kind.tolist() and FP.core.tolist() == S.core.tolist())
    have = set(fman.rxn_id.astype(int))
    fused_m = (FP.a_rxn.isin(have) & FP.b_rxn.isin(have)).to_numpy()
    f_used, p_g1 = set(FP.pair_id[fused_m]), set(S.pair_id[sets["P_g1"]])
    unit_cmp = {}
    for k, rec in lists.items():
        for x in ("train", "test", "inner_train", "inner_val"):
            a = [p for p in rec[x] if p in f_used]
            b = [p for p in rec[x] if p in p_g1]
            unit_cmp[f"T1__f{k}:{x}"] = dict(n_fu2=len(a), n_P_g1=len(b), same=a == b)
    Fo = pd.read_csv(ORIG_LABELS, usecols=["pair_id"] + DD_COLS).set_index("pair_id")
    dd_vs = {c: float(np.nanmax(np.abs(FP.set_index("pair_id")[c].to_numpy(float) - Fo.loc[FP.pair_id, c].to_numpy(float))))
             for c in DD_COLS}
    fu2 = dict(pairs=str(FU2_SPLITS / "pairs.parquet"), pairs_sha256=L.sha256(FU2_SPLITS / "pairs.parquet"),
               pairs_sha256_ok=L.sha256(FU2_SPLITS / "pairs.parquet") == fsrep["pairs_sha256"],
               t_manifest_sha256_ok=L.sha256(FU2_RES_G1 / "T_manifest.csv") == ftrep.get("manifest_sha256"),
               universe_same_as_S_folds=same_univ, n_pairs=int(len(FP)), n_fu2_used=len(f_used), n_P_g1=len(p_g1),
               only_in_P_g1=sorted(p_g1 - f_used)[:50], n_only_in_P_g1=len(p_g1 - f_used),
               only_in_fu2=sorted(f_used - p_g1)[:50], n_only_in_fu2=len(f_used - p_g1),
               t1_unit_lists=unit_cmp, dd_pairs_parquet_vs_F_pairs_max_abs=dd_vs,
               rule=DECISIONS["P9_fu2"])
    fu2["ok"] = bool(fu2["pairs_sha256_ok"] and fu2["t_manifest_sha256_ok"] and same_univ and f_used == p_g1)
    rep["fu2_check"] = fu2
    if not (fu2["pairs_sha256_ok"] and fu2["t_manifest_sha256_ok"] and same_univ):
        stop.append("FU2 pairs.parquet / T_manifest / universe inconsistent with S_folds.csv")
    if f_used != p_g1:
        stop.append(f"P_g1 ({len(p_g1)}) != FU2 pair list ({len(f_used)}): {len(p_g1 - f_used)} only in P_g1, "
                    f"{len(f_used - p_g1)} only in FU2 (spec 16 P)")
    # 2 % rule (P10)
    n_dft, n_g1 = int(sets["P_dft"].sum()), int(sets["P_g1"].sum())
    rep["dft_vs_g1"] = dict(n_P_dft=n_dft, n_P_g1=n_g1, ratio=n_dft / n_g1 if n_g1 else None,
                            limit=1 - MAX_DROP_VS_G1, ok=bool(n_g1 and n_dft >= (1 - MAX_DROP_VS_G1) * n_g1))
    if not rep["dft_vs_g1"]["ok"]:
        stop.append(f"P_dft {n_dft} < {1 - MAX_DROP_VS_G1:.2f} x P_g1 {n_g1} (spec 16 P)")
    # sets: summaries, dropped pairs, dH changes, files
    rep["sets"] = {}
    for name, m in sets.items():
        d = _set_summary(S, m, lists)
        if name in ("P_dft", "P_g1", "P_g2"):
            d["dropped"] = _dropped(S, m, end_reason(name[2:]))
        else:
            parts = [("P_dft", sets["P_dft"]), ("P_g1", sets["P_g1"])] + ([("P_g2", sets["P_g2"])] if name == "P_012"
                                                                        else [])
            rs = ["+".join(f"not_in_{n}" for n, mm in parts if not mm[i]) for i in np.nonzero(~m)[0]]
            sub = S[~m]
            used = (sub.in_T_main | sub.in_P_sens | sub.in_A_main).to_numpy()
            d["dropped"] = dict(n=int((~m).sum()), by_reason=dict(Counter(rs)), n_used_in_T1=int(used.sum()),
                                used_by_reason=dict(Counter(r for r, u in zip(rs, used) if u)))
        dh = {}
        for r in [x for x in rungs if x in {"P_dft": ["dft"], "P_g1": ["g1"], "P_g2": ["g2"], "P_01": ["dft", "g1"],
                                            "P_012": ["dft", "g1", "g2"]}[name]]:
            Tr = tabs[r].set_index("rxn_id")
            sub = S[m]
            dh[r] = {}
            for t in H_REREF:
                new = Tr.loc[sub.a_rxn, f"H_{t}"].to_numpy(float) - Tr.loc[sub.b_rxn, f"H_{t}"].to_numpy(float)
                old = Tr.loc[sub.a_rxn, f"H0_{t}"].to_numpy(float) - Tr.loc[sub.b_rxn, f"H0_{t}"].to_numpy(float)
                dd = np.abs(new - old)
                dh[r][t] = dict(n_changed=int((dd > 0).sum()), n_gt_1e9=int((dd > TOL).sum()),
                                n_gt_0_05=int((dd > 0.05).sum()), n_para_gt_0_05=int(((dd > 0.05) & (sub.pos == "p")
                                                                                       .to_numpy()).sum()),
                                abs_change=_q(dd[dd > 0]))
        d["dH_changed_by_rereferencing"] = dh
        p = set_path(name)
        df = S.loc[m, SET_COLS]
        L.write_atomic(p, lambda t, df=df: df.to_csv(t, index=False))
        d.update(file=str(p), sha256=L.sha256(p))
        rep["sets"][name] = d
    # label files (information; the loader checks the file it is given)
    lab = {}
    for p in (F_PAIRS_CR2, F_PAIRS_CR, ORIG_LABELS):
        if not p.is_file():
            lab[str(p)] = None
            continue
        Y = pd.read_csv(p, usecols=["pair_id", "a_rxn", "b_rxn"] + DD_COLS)
        Yi = Y.set_index("pair_id")
        ok = bool(len(Y) == len(S) and set(Y.pair_id) == set(S.pair_id)
                  and (Yi.loc[S.pair_id, "a_rxn"].to_numpy() == S.a_rxn.to_numpy()).all()
                  and (Yi.loc[S.pair_id, "b_rxn"].to_numpy() == S.b_rxn.to_numpy()).all())
        dmax = {c: float(np.nanmax(np.abs(Yi.loc[S.pair_id, c].to_numpy(float) - Fo.loc[S.pair_id, c].to_numpy(float))))
                for c in DD_COLS}
        lab[str(p)] = dict(sha256=L.sha256(p), ids_ok=ok, dd_max_abs_vs_F_pairs=dmax)
    rep["labels"] = lab
    rep["h_changed_by_rereferencing"] = {r: brs[r]["table"]["h_changed"] for r in rungs}
    rep.update(stop=bool(stop), stop_reasons=stop, all_pass=not stop, running=False, finished=_now(),
               wall_s=round(time.time() - t0, 1),
               outputs={**{rep["sets"][n]["file"]: rep["sets"][n]["sha256"] for n in rep["sets"]}, **finfo["files"]})
    L.write_json(P_REPORT, rep)
    _log(t0, "pairs: " + ", ".join(f"{n} {rep['sets'][n]['n']}" for n in rep["sets"])
         + (f"; STOP: {stop}" if stop else "; all pass"))
    if stop:
        print("STOP: " + "; ".join(stop), file=sys.stderr, flush=True)
        return 3
    return 0


# ---------------------------------------------------------------- loader side (training / tests)
def _not_ready(msg):
    """exit 2 = an input phase has not finished (the loader-side counterpart of L.stop; fix of review P-7)."""
    print(f"NOT READY: {msg}", file=sys.stderr, flush=True)
    sys.exit(2)


def p_report(rung=None):
    """results/P_report.json (all_pass required); with rung: that rung's build report passed, is current (fingerprint)
    and its table sha256 is the one P_report recorded. Exit 2 when P_report / the build is missing or stale, 3 on a
    STOP or a changed table."""
    PATHLOG.note(P_REPORT, "p_report")
    if not P_REPORT.is_file():
        _not_ready(f"{P_REPORT} missing (run inputs.py pairs)")
    rep = json.loads(P_REPORT.read_text())
    if rep.get("running"):
        _not_ready(f"{P_REPORT} is being written (inputs.py pairs running)")
    if not rep.get("all_pass") or rep.get("stop"):
        L.stop(f"{P_REPORT}: all_pass {rep.get('all_pass')}, stop {rep.get('stop_reasons')}")
    if rung is not None:
        if rung not in (rep.get("rungs") or {}):
            _not_ready(f"rung {rung} not in {P_REPORT} (build {rung}, then pairs)")
        ok, br, code = _build_ok(rung)
        if not ok:
            (L.stop if code == 3 else _not_ready)(f"rung {rung}: {br}")
        if br["outputs"][str(table_path(rung))] != rep["rungs"][rung]["table_sha256"]:
            L.stop(f"{table_path(rung)} differs from the table P_report.json was made from (rerun inputs.py pairs)")
    return rep


def _verified(path, sha, what):
    PATHLOG.note(path, what)
    got = _sha(path)
    if got != sha:
        L.stop(f"{path} sha256 {got} != {sha} recorded in {P_REPORT.name} ({what})")
    return Path(path)


def read_table(rung, rep=None):
    """the rung input table (sha256 = P_report)."""
    rep = rep or p_report(rung)
    p = _verified(table_path(rung), rep["rungs"][rung]["table_sha256"], f"{rung} input table")
    return pd.read_parquet(p)


def read_set(name, rep=None):
    """pair-set file P_<name> (name 'dft' / 'P_dft' ...), sha256 = P_report."""
    rep = rep or p_report()
    name = name if name.startswith("P_") else f"P_{name}"
    s = (rep.get("sets") or {}).get(name)
    if s is None:
        L.stop(f"pair set {name} not in {P_REPORT}")
    p = _verified(s["file"], s["sha256"], f"pair set {name}")
    return pd.read_csv(p, dtype={"pair_id": str, "a_job": str, "b_job": str})


def read_labels(path=None):
    """(Y, sha256): pair_id + dd_<10 targets> read exactly as d1_train.data() reads F_pairs.csv (pandas default float
    parser, so the FU2 y of L-6 is reproduced), merged with a_job / a_rxn / b_job / b_rxn of the same file."""
    path = Path(path or DEFAULT_LABELS)
    PATHLOG.note(path, "pair labels")
    if not path.is_file():
        _not_ready(f"label file {path} missing")
    Y = pd.read_csv(path, usecols=["pair_id"] + DD_COLS)
    ids = pd.read_csv(path, usecols=["pair_id", "a_job", "a_rxn", "b_job", "b_rxn"],
                      dtype={"pair_id": str, "a_job": str, "b_job": str})
    if Y.pair_id.duplicated().any() or len(ids) != len(Y):
        L.stop(f"{path}: duplicate pair_id")
    Y["pair_id"] = Y.pair_id.astype(str)
    return Y.merge(ids, on="pair_id", how="left", validate="1:1"), L.sha256(path)


def _read_s_folds(rep):
    p = _verified(S_FOLDS, rep["inputs"][str(S_FOLDS)], "S_folds.csv")
    return pd.read_csv(p)


def _hcols(h):
    if h not in H_SOURCES:
        raise ValueError(f"h must be one of {H_SOURCES}")
    return H_COLS if h == "cr" else H0_COLS


def reaction_frame(rung, arm="S", h="cr", rep=None):
    """usable reactions of the rung, index rxn_id: the 182 X182 columns (arm S) or 182 + H_<t> (arm SH; values of the
    h source, column names H_COLS either way). For the XGB arm (rev 9 arm C input: 182; SH: 182 + H 9)."""
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    rep = rep or p_report(rung)
    T = read_table(rung, rep)
    xcols = rep["rungs"][rung]["x182_cols"]
    Tu = T[T.usable].set_index("rxn_id")
    X = Tu[xcols].copy()
    if arm == "SH":
        hc = _hcols(h)
        X = X.join(Tu[hc].rename(columns=dict(zip(hc, H_COLS))))
    return X


def pairs_frame(rung, label_path=None, h="cr", rep=None, subset="usable"):
    """S_folds.csv rows (original S index and order, every S_folds column) + dd_<t> from label_path + dH_<t> = H(a) -
    H(b) for the 9 H targets (h source). subset 'usable' = P_<rung> (training / evaluation); 'has_h' = both ends have
    the 9 H values, tokens / X182 not required (the D1-HF Phase A pair rule, for L-10). attrs: label_path,
    label_sha256, rung, h, subset."""
    rep = rep or p_report(rung)
    S = _read_s_folds(rep)
    if subset == "usable":
        keep = set(read_set(rung, rep).pair_id)
    elif subset == "has_h":
        T0 = read_table(rung, rep)
        hh = set(T0.rxn_id[T0.has_h].astype(int))
        keep = set(S.pair_id[S.a_rxn.isin(hh) & S.b_rxn.isin(hh)])
    else:
        raise ValueError("subset must be 'usable' or 'has_h'")
    Y, ysha = read_labels(label_path)
    Yi = Y.set_index("pair_id")
    if set(Yi.index) != set(S.pair_id) or \
            not (Yi.loc[S.pair_id, "a_rxn"].to_numpy() == S.a_rxn.to_numpy()).all() or \
            not (Yi.loc[S.pair_id, "b_rxn"].to_numpy() == S.b_rxn.to_numpy()).all():
        L.stop(f"label file {label_path or DEFAULT_LABELS}: pair ids / ends differ from S_folds.csv")
    S = S[S.pair_id.isin(keep)].copy()
    for c in DD_COLS:
        S[c] = Yi.loc[S.pair_id, c].to_numpy(float)
    T = read_table(rung, rep).set_index("rxn_id")
    for t, c in zip(H_TARGETS, _hcols(h)):
        S[f"dH_{t}"] = T.loc[S.a_rxn, c].to_numpy(float) - T.loc[S.b_rxn, c].to_numpy(float)
    S.attrs.update(label_path=str(label_path or DEFAULT_LABELS), label_sha256=ysha, rung=rung, h=h, subset=subset)
    return S


def fold_lists(rung, fold, rep=None):
    """FU2 T1 fold `fold` (train / test / inner_train / inner_val pair ids, FU2 order) restricted to P_<rung>, + dropped
    counts (as fu2.train.Data.split drops pairs without atom data)."""
    rep = rep or p_report(rung)
    keep = set(read_set(rung, rep).pair_id)
    name = f"T1__f{int(fold)}.json"
    p = SPLITS_COPY / name
    _verified(p, rep["folds"]["files"][str(p)], f"fold copy {name}")
    rec = json.loads(p.read_text())
    out = {x: [q for q in rec[x] if q in keep] for x in ("train", "test", "inner_train", "inner_val")}
    out["dropped"] = {x: len(rec[x]) - len(out[x]) for x in ("train", "test", "inner_train", "inner_val")}
    out.update(split="T1", fold=int(fold), rung=rung)
    return out


# ---------------------------------------------------------------- FU2 Data on the rung roots (P12)
def fu2_modules():
    """(train, fu2_common, atoms, tokens) of analysis/d1_pair/fu2 (imported as FU2's own tests do)."""
    d = str(FU2_DIR)
    if d not in sys.path:
        sys.path.insert(0, d)
    import train as TR
    import fu2_common as F
    import atoms as AT
    import tokens as TK
    for m, f in ((TR, "train.py"), (F, "fu2_common.py"), (AT, "atoms.py"), (TK, "tokens.py")):
        if Path(m.__file__).resolve() != (FU2_DIR / f).resolve():
            L.stop(f"module {m.__name__} imported from {m.__file__}, expected {FU2_DIR / f}")
    if TR.F is not F:
        L.stop("fu2 train.F is not fu2_common")
    return TR, F, AT, TK


@contextlib.contextmanager
def fu2_roots(rung):
    """fu2_common FU2_RES / TOKENS / SPLITS and atoms.ATOM_VERSION / tokens.TOKEN_VERSION rebound to the rung for the
    duration of the block (the D1-G2 atoms.py root_bind pattern; restored afterwards)."""
    TR, F, AT, TK = fu2_modules()
    cfg = RUNGS[rung]
    saved = (F.FU2_RES, F.TOKENS, F.SPLITS, AT.ATOM_VERSION, TK.TOKEN_VERSION)
    F.FU2_RES, F.TOKENS, F.SPLITS = Path(cfg["fu2_res"]), Path(cfg["tokens"]), SPLITS_COPY
    AT.ATOM_VERSION, TK.TOKEN_VERSION = cfg["atom_version"], cfg["token_version"]
    try:
        yield
    finally:
        F.FU2_RES, F.TOKENS, F.SPLITS, AT.ATOM_VERSION, TK.TOKEN_VERSION = saved


_CLS = {}


def _ladder_cls():
    if "LadderData" in _CLS:
        return _CLS["LadderData"]
    TR, F, AT, TK = fu2_modules()
    if list(F.TARGETS) != TARGETS:
        L.stop(f"fu2 TARGETS {F.TARGETS} != d1pair TARGETS {TARGETS}")

    class LadderData(TR.Data):
        """fu2.train.Data of one rung x arm x label file (P12). FU2's __init__ / split run unchanged on the rung roots;
        afterwards only: P[dd_*], y <- label_path; P.has_atoms <- pair in P_<rung> (P.has_atoms_fu2 = FU2's own flag);
        arm SH: glob <- [glob | H] (d_glob 182 + 9). Added attributes: rung, arm, h_source, label_path, label_sha256,
        H [n_token_rows, 9] (H_TARGETS order, 0 for rows that are not usable), row_usable, dH [n_pairs, 10] (TARGETS
        order; c_ghost 0; NaN outside P_<rung>), h_mask (TARGETS with an H), glob_cols, y_fu2_max_abs_diff (label file
        y vs pairs.parquet y), inputs {path: sha256}."""

        def __init__(self, rung, arm="S", label_path=None, h="cr"):
            if rung not in RUNGS or arm not in ARMS or h not in H_SOURCES:
                raise ValueError(f"rung {rung} / arm {arm} / h {h}")
            PATHLOG.install()
            rep = p_report(rung)
            rr = rep["rungs"][rung]
            for p, s in rep["folds"]["files"].items():
                _verified(p, s, "fold copy")
            self.rung, self.arm, self.h_source = rung, arm, h
            with fu2_roots(rung):
                PATHLOG.note(SPLITS_COPY / "pairs.parquet", "fu2 Data pairs.parquet (copy)")
                super().__init__()
            if self.t_report_sha != rr["tokens"]["t_report_sha256"] or \
                    self.trep.get("manifest_sha256") != rr["tokens"]["manifest_sha256"]:
                L.stop(f"{rung} T_report / T_manifest changed since {P_REPORT.name}")
            # labels (P13)
            self.label_path = Path(label_path or DEFAULT_LABELS)
            Y, self.label_sha256 = read_labels(self.label_path)
            Yi = Y.set_index("pair_id")
            if set(Yi.index) != set(self.P.index) or \
                    not (Yi.loc[self.P.index, "a_rxn"].to_numpy() == self.P.a_rxn.to_numpy()).all() or \
                    not (Yi.loc[self.P.index, "b_rxn"].to_numpy() == self.P.b_rxn.to_numpy()).all():
                L.stop(f"label file {self.label_path}: pair ids / ends differ from FU2 pairs.parquet")
            y_fu2 = self.y.copy()
            for c in DD_COLS:
                self.P[c] = Yi.loc[self.P.index, c].to_numpy(float)
            self.y = self.P[[f"dd_{t}" for t in F.TARGETS]].to_numpy(float)          # = Data.__init__
            self.y_fu2_max_abs_diff = float(np.nanmax(np.abs(self.y - y_fu2)))
            # usable pairs (P_<rung>)
            T = read_table(rung, rep)
            Tu = T[T.usable]
            usable = set(Tu.rxn_id.astype(int))
            self.P["has_atoms_fu2"] = self.P.has_atoms
            self.P["has_atoms"] = self.P.a_rxn.isin(usable) & self.P.b_rxn.isin(usable)
            sset = read_set(rung, rep)
            if set(self.P.index[self.P.has_atoms]) != set(sset.pair_id):
                L.stop(f"usable pairs of {table_path(rung)} != {set_path('P_' + rung)}")
            bad = self.P.has_atoms & ~self.P.has_atoms_fu2
            if bad.any():
                L.stop(f"{int(bad.sum())} usable pairs without token rows (usable must imply tokens)")
            # H per token row, dH per pair
            hv = Tu.set_index("rxn_id")[_hcols(h)]
            self.row_usable = np.array([int(r) in usable for r in self.rids], dtype=bool)
            self.H = np.zeros((len(self.rids), len(H_TARGETS)))
            ix = np.nonzero(self.row_usable)[0]
            self.H[ix] = hv.loc[self.rids[ix]].to_numpy(float)
            hp = self.P.has_atoms.to_numpy()
            if not (self.row_usable[self.ra[hp]].all() and self.row_usable[self.rb[hp]].all()):
                L.stop("a usable pair points at a token row without H")
            if not np.isfinite(self.H[ix]).all():
                L.stop(f"non-finite H in the usable rows of {table_path(rung)}")
            self.dH = np.full((len(self.P), len(F.TARGETS)), np.nan)
            for j, t in enumerate(F.TARGETS):
                if t in H_TARGETS:
                    c = H_TARGETS.index(t)
                    self.dH[hp, j] = self.H[self.ra[hp], c] - self.H[self.rb[hp], c]
                else:
                    self.dH[hp, j] = 0.0                                              # c_ghost: beta = 0 (spec 4-2)
            self.h_mask = np.array([t in H_TARGETS for t in F.TARGETS])
            self.glob_cols = list(self.trep["glob_cols"])
            if arm == "SH":
                self.glob = np.concatenate([self.glob, self.H], 1)
                self.glob_cols += H_COLS
                self.d_glob = self.glob.shape[1]
            self.inputs = {str(table_path(rung)): rr["table_sha256"], rep["sets"][f"P_{rung}"]["file"]:
                           rep["sets"][f"P_{rung}"]["sha256"], str(self.label_path): self.label_sha256,
                           str(Path(rr["tokens"]["t_report"])): self.t_report_sha, "S_splits": self.s_splits_sha,
                           "Q_report": self.q_report_sha, "pairs.parquet": self.srep["pairs_sha256"]}

        def split(self, split, k):
            """FU2 Data.split (pairs outside P_<rung> dropped through has_atoms) on the fold copies; T1 only."""
            if split != "T1":
                L.stop(f"split {split}: LADDER uses T1 only (spec 4-3)")
            with fu2_roots(self.rung):
                PATHLOG.note(SPLITS_COPY / f"T1__f{int(k)}.json", "fold copy")
                return super().split(split, k)

    _CLS["LadderData"] = LadderData
    return LadderData


def __getattr__(name):                     # PEP 562: inputs.LadderData imports torch / FU2 only when used
    if name == "LadderData":
        return _ladder_cls()
    raise AttributeError(f"module 'inputs' has no attribute {name!r}")


def load(rung, arm="S", label_path=None, fold=0, h="cr"):
    """(LadderData, its D.split('T1', fold) or None)."""
    D = _ladder_cls()(rung, arm, label_path, h)
    return D, (D.split("T1", fold) if fold is not None else None)


# ---------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description="D1LADDER Phase P (inputs.py)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("rung", choices=RUNG_ORDER)
    b.add_argument("--force", action="store_true")
    b.add_argument("--accept-f0-stop", action="store_true", help="only after a user decision on the F0 STOP (P16)")
    b.add_argument("--g2-q-keys", action="store_true", help="g2 only, after a user decision on P20 (spec-literal Q-2 keys)")
    b.add_argument("--accept-g2f-stop", action="store_true",
                   help="g2 only: accept the D1-G2 F gate STOP recorded in analysis/d1_g2/results/F_DECISION.md")
    b.add_argument("--g2-cons-own", action="store_true",
                   help="g2 only, P20 option (c): constrained-key rows keep their own G2 reference (decision 2026-10-23)")
    a = sub.add_parser("pa")
    a.add_argument("rung", choices=RUNG_ORDER)
    sub.add_parser("pairs")
    o = ap.parse_args(argv)
    if o.cmd == "build":
        if o.g2_q_keys and o.g2_cons_own:
            ap.error("--g2-q-keys and --g2-cons-own exclude each other")
        if (o.g2_q_keys or o.g2_cons_own) and o.rung != "g2":
            ap.error("--g2-q-keys and --g2-cons-own are for g2 only")
        if (o.accept_g2f_stop and o.rung != "g2") or (o.accept_f0_stop and o.rung != "dft"):
            ap.error("--accept-f0-stop is for dft, --accept-g2f-stop for g2")
        return cmd_build(o.rung, o.force, o.accept_f0_stop or o.accept_g2f_stop,
                         "own" if o.g2_cons_own else o.g2_q_keys)
    if o.cmd == "pa":
        return cmd_pa(o.rung)
    if o.cmd == "pairs":
        return cmd_pairs()
    return 2


if __name__ == "__main__":
    sys.exit(main())
