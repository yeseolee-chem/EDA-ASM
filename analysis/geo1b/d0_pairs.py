#!/usr/bin/env python3
"""GEO1b D0 pairs and folds = design step 2 of results/D0D1_design_investigation.json (DECISIONS.md 'D0D1', user
2026-10-31). D1 run env (python 3.9, rdkit, pandas, numpy; jobs/cpu.slurm), sbatch only:

  cd analysis/geo1b && sbatch -p <cpu1|cpu2> -J gb_d0pairs --cpus-per-task=2 --mem=8G jobs/cpu.slurm d0_pairs.py build
  python d0_pairs.py build [--force] [--accept-twin-conflicts]
                                        # exit 0 = outputs written, 3 = STOP (results/D0_pairs_report.json), 2 = usage

Reads (never a model output):
  $GEO1B/d0/labels_d0_q.json, results/Q_d0_report.json   D0 Q labels (d0_labels.py): 10 targets (D1_TCOL names),
                                                          status, charge1/2, flag_async. Used only when the report has
                                                          all_pass true, stop false and records this file's sha256
                                                          under 'outputs' (d0_labels.py reader contract)
  labels_all.json (main checkout)                         Coley-reference energies e_frag{1,2}_rel_eh (ex rule), roles,
                                                          cross-checks of the reference-free targets / charges / flags
  $G2C/d0_r1/d0_manifest.csv                              dipole_smiles, dipolarophile_smiles, d0_rxn_id
  D1 manifest + geo_common.d1_core_table()                D1 reactant sets and T1 folds (twins)
  data.d0_overlap_ids()                                   the 5 GEO1 D-1 overlaps
  which D0 reactions have a G2 TS (anchor priority only): the G2c selection = the .done folders of GC.D0_G2_FINAL
  ($D0G2_SCRATCH/g2_final, default $G2C/d0/g2_final) checked against GC.G2C_SELECT (g2c worktree results/C_select.csv)
  once select_g2c.py build has written all of its output (folders, C_select.csv, C_select.json). Before the build has
  written any of it: the D0-G2 orig folders ($G2C_ORIG_D0, default /gpfs/tmp_cpu2/yeseo1ee/d0_g2/g2_final), report
  g2_source 'd0_g2_orig' = preliminary (GEO1b data.py refuses it). Part of the build output only = STOP. Folds and
  components never depend on it, pairs (anchors) do.

Writes (atomic; the same inputs give byte-identical files):
  $GEO1B/d0/d0_folds.csv       rxn_id, component, fold (0-4; -9 = D-1 overlap, in no set), twin_of, overlap
                               (read twin_of with dtype=str, keep_default_na=False: '' or a D1 rxn_id)
  $GEO1B/d0/D0_pairs.csv       F_pairs.csv columns (pair_id D#####, ref_kind 'd0', a_job/b_job 'D0G2_<rid:04d>',
                               fam = core = 'D0/<component>', dipole 'D0', skeleton <component>, pos 'd0', the other
                               D1-only columns empty) + edge_kind (sibling / dipole / leaf), ok
  $GEO1B/d0/d0_components.csv  one row per component
  $GEO1B/d0/d0_twins.csv       one row per D0 reaction with a stereo-free D1 twin: rxn_id, kind (forced / iso_resolved
                               / m1_only / conflict), fold, component, twin_of, m1 (a D1 twin of T1 fold -1), n_twins,
                               twins_ns, twins_iso, bind ('<D1 rxn_id>:<T1 fold>' space separated; bind = the twins that
                               set the fold). Read with dtype=str, keep_default_na=False
  results/D0_pairs_report.json counts per fold / edge kind / charge, twins, gates, sha256 of inputs and outputs
Frozen folds: an existing d0_folds.csv that would change is a STOP (GEO1b may have read it); --force overwrites it.

Rules (design 'D0 PAIRS'; judgment calls in results/DECISIONS_PAIR.md):
  population  labels_all status-ok rows (5,260) minus the 5 D-1 overlaps = 5,255
  component   = the fold unit = one dipole molecule (stereo-free canonical dipole_smiles): all its reactions, of every
              charge, share one component and one fold. Only twins split a dipole: one component per twin-forced fold,
              the unforced reactions join the fold that forces most of its reactions. Sibling sets = same dipolarophile
              molecule (canonical isomeric SMILES) and charges = regio / stereo siblings.
  anchor tree one tree per charge class (charge1, charge2) of a component (edges only between equal charges, never
              across folds): siblings -> their sibling-set anchor ('sibling'); sibling-set anchors -> the charge-class
              anchor ('dipole'). A reaction alone in its fold for its dipole (its dipole occurs nowhere else, or a twin
              forced it away from the rest of its dipole) attaches to a same-dipolarophile, same-charge, fold-compatible
              reaction of a dipole component and joins that component ('leaf'); such reactions without one but sharing
              a dipolarophile and charges among themselves form a leaf component; the rest are singletons (no pair).
              n members in m charge classes -> n - m pairs. Anchor priority: has a G2 TS, then passes the row rules,
              then lowest rxn_id. The component a leaf joins: partner passes the row rules on the Coley labels
              (labels_all, never the Q labels or G2), then lowest rxn_id, so folds depend on neither.
  twins       a D0 reaction with the same stereo-free reactant set as a D1 target takes that D1 reaction's T1 fold
              (0-4); several folds -> the D1 reactions with the same isomeric reactant set decide (iso_resolved);
              still several -> conflict = STOP for a user decision ('--accept-twin-conflicts' = lowest D1 rxn_id, the
              leak stays listed). D1 fold -1 twins do not force a fold (m1, listed).
  folds       forced components first; the other components with a pair in rng(20261019) permutation order into the
              fold with the fewest pairs so far (d1_split T1 rule, ties lower fold); then the components without a
              pair in a second permutation of the same generator into the fold with the fewest reactions so far
  pairs       dd_t = y_a - y_b on the Q labels, a = child, b = its anchor. hyg = both rows pass the rev 6 hygiene
              rule (d1 >= 0, d2 >= 0, d2 <= 50); ok = both rows pass flag_async false, ex1 <= 20, ex2 <= 20 (ex = row
              reference - lowest reference of the same molecule, Coley references, build_rowattr.py rule) and hygiene
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import geo_common as GC  # noqa: E402

CODE_VERSION = "geo1b-d0pairs-2"

# ---------------------------------------------------------------- constants (source named on each line)
KCAL = 627.5094740631                   # stage3_parse.EH_TO_KCAL = build_rowattr.H
SEED = 20261019                         # d1pair_common.SEED (d1_split T1 permutation)
N_FOLDS = 5                             # d1pair_common.N_FOLDS
EX_MAX = 20.0                           # build_rowattr.py: ex1 > 20 or ex2 > 20 kcal/mol -> row dropped
HYG_D2_MAX = 50.0                       # rev6_common.HYG_D2_MAX (hygiene: d1 >= 0, d2 >= 0, d2 <= 50)
N_POP_OK = 5260                         # labels_all.json status-ok rows
N_OVERLAP = 5                           # GEO1 decision D-1 (analysis/geo/results/D_dataset_report.json d0_overlap)
FOLD_OVERLAP = GC.D0_FOLD_OVERLAP       # = -9: d0_folds.csv fold of a D-1 overlap (in no set)
EQ_TOL = 1e-9                           # reference-free targets: labels_d0_q vs labels_all (kcal/mol)
D1_TCOL = {"d1": "d1_kcal", "d2": "d2_kcal", "elst": "elst_dft", "pauli": "pauli_dft", "oi": "oi_dft",
           "disp": "disp_dft", "cpcm": "cpcm_dft", "cds": "cds_dft", "c_ghost": "bsse_shift_kcal",
           "barrier": "barrier_kcal"}   # = d1pair_common.D1_TCOL = d1_anchor_pairs.TARGETS
REF_FREE = ("elst", "pauli", "oi", "disp", "cpcm", "cds", "c_ghost")   # targets Q does not change
F_COLS = ["pair_id", "ref_kind", "a_job", "a_rxn", "b_job", "b_rxn", "Xb", "pos", "dipole", "skeleton", "series", "cfg",
          "orient", "face", "c3", "fam", "core", "sigma"] + [f"dd_{t}" for t in D1_TCOL] + \
         ["feat", "hyg", "p4_drop", "in_P_all", "in_P_feat", "in_P_main", "in_P_sens", "anchor_Xb", "anchor_pos",
          "in_A_feat", "in_A_main"]     # = header of analysis/d1_pair/results/F_pairs.csv (checked at run time)
EXTRA_COLS = ["edge_kind", "ok"]
FOLD_COLS = ["rxn_id", "component", "fold", "twin_of", "overlap"]
COMP_COLS = ["component", "kind", "key", "charges", "n_rxn", "n_pairs", "n_trees", "n_sets", "n_leaves", "fold",
             "forced", "anchor", "anchors"]
TWIN_COLS = ["rxn_id", "kind", "fold", "component", "twin_of", "m1", "n_twins", "twins_ns", "twins_iso", "bind"]
TWIN_KINDS = ("forced", "iso_resolved", "m1_only", "conflict")
EDGE_KINDS = ("sibling", "dipole", "leaf")

# ---------------------------------------------------------------- paths
D0DIR = GC.D0_DIR                                                       # $GEO1B/d0
LABELS_Q = D0DIR / "labels_d0_q.json"                                   # d0_labels.py OUT_LABELS
Q_REPORT = GC.RES / "Q_d0_report.json"                                  # d0_labels.py REPORT
FOLDS_CSV = GC.D0_FOLDS                                                 # $GEO1B/d0/d0_folds.csv (GEO1b data.py)
PAIRS_CSV = D0DIR / "D0_pairs.csv"
COMP_CSV = D0DIR / "d0_components.csv"
TWINS_CSV = D0DIR / "d0_twins.csv"
REPORT = GC.D0_PAIRS_REPORT                                             # results/D0_pairs_report.json
OUTPUTS = (FOLDS_CSV, PAIRS_CSV, COMP_CSV, TWINS_CSV)
LABELS_ALL = Path(os.environ.get("D0_LABELS_ALL", "/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/labels_all.json"))
D0_MANIFEST = GC.G2C / "d0_r1" / "d0_manifest.csv"                      # $G2C/d0_r1/d0_manifest.csv
G2C_D0_FINAL = GC.D0_G2_FINAL                                           # select_g2c.py build output (D0) = data.py input
G2C_SELECT = GC.G2C_SELECT                                              # select_g2c.RES / C_select.csv (written 2nd)
G2C_SELECT_JS = GC.G2C_RES / "C_select.json"                            # select_g2c.RES / C_select.json (written last)
ORIG_D0_FINAL = Path(os.environ.get("G2C_ORIG_D0", "/gpfs/tmp_cpu2/yeseo1ee/d0_g2/g2_final"))  # = select_g2c.ORIG['d0']


# ---------------------------------------------------------------- helpers
def write_text_atomic(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


_CANON = {}


def canon(smi, stereo):
    """RDKit canonical SMILES (atom maps cleared, explicit H removed as data.canon_rxn); stereo=False also removes the
    stereochemistry. None if RDKit cannot parse it."""
    key = (str(smi), bool(stereo))
    if key not in _CANON:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
        m = Chem.MolFromSmiles(str(smi)) if str(smi).strip() else None     # '' would parse to an empty molecule
        if m is None:
            _CANON[key] = None
        else:
            for a in m.GetAtoms():
                a.SetAtomMapNum(0)
            m = Chem.RemoveHs(m)
            if not stereo:
                Chem.RemoveStereochemistry(m)
            _CANON[key] = Chem.MolToSmiles(m)
    return _CANON[key]


def rset(dip, dph):
    """reactant set = the two canonical SMILES sorted (None if either is None)."""
    return None if dip is None or dph is None else tuple(sorted((dip, dph)))


def _vc(values):
    return {str(k): int(v) for k, v in sorted(Counter(values).items(), key=lambda kv: str(kv[0]))}


def _main_fold(counter):
    """the forced fold holding the most forced reactions (ties: lower fold); None if nothing is forced."""
    return min(counter, key=lambda f: (-counter[f], f)) if counter else None


def _fkey(f):
    return -1 if f is None else int(f)


def stop(rep, fails):
    rep.update(stop=True, status="STOP", fails=fails, finished=time.strftime("%Y-%m-%dT%H:%M:%S"))
    GC.write_json(REPORT, rep)
    print("STOP (results/D0_pairs_report.json):\n  " + "\n  ".join(map(str, fails)), file=sys.stderr)
    return 3


# ---------------------------------------------------------------- G2 TS availability (anchor priority only)
def g2_availability(pop, fails):
    """('g2c' | 'g2c_incomplete' | 'd0_g2_orig', {rid: has a G2 TS}, info). select_g2c.py build writes the folders
    $G2C/d0/g2_final first, then C_select.csv, then C_select.json. All three present -> 'g2c' (the D0 'chosen' must
    agree with the .done markers); none present -> the D0-G2 orig .done folders ('d0_g2_orig', preliminary: GEO1b
    data.py refuses it); only some present -> STOP (the build is running or died; never a silent fallback)."""
    sel, sel_js = G2C_SELECT, G2C_SELECT_JS
    have = dict(g2c_final=G2C_D0_FINAL.is_dir(), c_select=sel.is_file(), c_select_json=sel_js.is_file())
    info = dict(c_select=str(sel), c_select_json=str(sel_js), g2c_final=str(G2C_D0_FINAL), orig_final=str(ORIG_D0_FINAL),
                present=have)
    if all(have.values()):
        S = pd.read_csv(sel, keep_default_na=False, dtype=str)
        S = S[S.src == "d0"]
        info.update(c_select_sha256=GC.sha256(sel), c_select_json_sha256=GC.sha256(sel_js), n_d0_rows=int(len(S)))
        if not len(S):
            fails.append(f"G2c {sel} exists but has no 'd0' rows (wrong or partial C_select.csv)")
            return "g2c", {r: False for r in pop}, info
        ch, dup, badc = {}, [], []
        for r, c in zip(S.rxn_id, S.chosen):
            rid = int(float(r))
            if rid in ch:
                dup.append(rid)
            if c not in ("",) + tuple(GC.G2C_RUNS):
                badc.append(rid)
            ch[rid] = c != ""
        miss = [r for r in pop if r not in ch]
        mism = [r for r in pop if r in ch and ch[r] != (G2C_D0_FINAL / str(r) / ".done").is_file()]
        for what, ids in (("lists twice", dup), ("has an unknown 'chosen' run for", badc), ("lacks", miss)):
            if ids:
                fails.append(f"G2c C_select.csv {what} {len(ids)} D0 reactions, e.g. {ids[:10]}")
        if mism:
            fails.append(f"G2c C_select.csv 'chosen' disagrees with {G2C_D0_FINAL} .done for {len(mism)} D0 "
                         f"reactions, e.g. {mism[:10]}")
        info.update(n_missing=len(miss), n_marker_mismatch=len(mism))
        return "g2c", {r: bool(ch.get(r, False)) for r in pop}, info
    if any(have.values()):
        fails.append(f"G2c build output incomplete ({have}): select_g2c.py build is running or stopped. Rerun "
                     f"d0_pairs.py once it has finished (no fallback to the orig folders while a build is under way)")
        return "g2c_incomplete", {r: False for r in pop}, info
    info["reason"] = ("G2c build not started (no $G2C/d0/g2_final, C_select.csv or C_select.json): D0-G2 orig .done "
                      "folders used. Preliminary: GEO1b data.py STOPs on it; rerun after select_g2c.py build")
    return "d0_g2_orig", {r: (ORIG_D0_FINAL / str(r) / ".done").is_file() for r in pop}, info


# ---------------------------------------------------------------- build
def cmd_build(force=False, accept_conflicts=False):
    t0 = time.time()
    rep = dict(code_version=CODE_VERSION, spec="results/D0D1_design_investigation.json design step 2; DECISIONS.md "
               "D0D1 (user 2026-10-31)", started=time.strftime("%Y-%m-%dT%H:%M:%S"), stop=True, status="running",
               options=dict(force=bool(force), accept_twin_conflicts=bool(accept_conflicts)))
    GC.write_json(REPORT, rep)                          # a killed run never leaves an old pass behind
    need = [LABELS_Q, Q_REPORT, LABELS_ALL, GC.D0_LABELS, D0_MANIFEST, GC.D1_MANIFEST, GC.C_COMPARE, GC.S_FOLDS,
            GC.F_PAIRS]
    miss = [str(p) for p in need if not Path(p).is_file()]
    if miss:
        return stop(rep, [f"missing input {m}" for m in miss])
    rep["inputs_sha256"] = {str(p): GC.sha256(p) for p in need}
    fails = []

    # ---- G-0 file-level checks
    hdr = list(pd.read_csv(GC.F_PAIRS, nrows=0).columns)
    if hdr != F_COLS:
        fails.append(f"F_pairs.csv header differs from F_COLS: {hdr}")
    if GC.sha256(LABELS_ALL) != GC.sha256(GC.D0_LABELS):
        fails.append(f"{LABELS_ALL} differs from {GC.D0_LABELS} (GEO1b data.py reads the latter)")
    # d0_labels.py reader contract: all_pass true, stop false, and labels_d0_q.json = the file that pass wrote
    qrep = json.loads(Q_REPORT.read_text())
    q_out = [v for k, v in (qrep.get("outputs") or {}).items() if Path(k).name == LABELS_Q.name]
    q_sha = rep["inputs_sha256"][str(LABELS_Q)]
    rep["q_report"] = dict(all_pass=qrep.get("all_pass"), stop=qrep.get("stop"), labels_sha256=q_sha,
                           labels_sha256_recorded=q_out)
    if qrep.get("all_pass") is not True or qrep.get("stop") is not False:
        fails.append(f"results/Q_d0_report.json is not a pass: all_pass {qrep.get('all_pass')!r}, stop "
                     f"{qrep.get('stop')!r} (both must be true / false)")
    if len(q_out) != 1 or q_out[0] != q_sha:
        fails.append(f"{LABELS_Q} sha256 {q_sha} is not the one Q_d0_report.json records under 'outputs' ({q_out}): "
                     f"the labels changed after the Q pass, or another d0_labels.py output")

    # ---- labels
    A = {int(d["rxn_id"]): d for d in json.loads(LABELS_ALL.read_text()) if d.get("status") == "ok"}
    if len(A) != N_POP_OK:
        fails.append(f"labels_all.json has {len(A)} status-ok rows, expected {N_POP_OK}")
    Q, dup_q = {}, []
    for d in json.loads(LABELS_Q.read_text()):
        rid = int(d["rxn_id"])
        if rid in Q:
            dup_q.append(rid)
        Q[rid] = d
    if dup_q:
        fails.append(f"labels_d0_q.json repeats {len(dup_q)} rxn_id, e.g. {dup_q[:10]}")

    # ---- G-1 the 5 D-1 overlaps (GEO1 rule, unchanged code)
    try:
        import data as DATA                             # lazy: data.py imports rmsd (networkx, fragmenter)
        ov_all, ov_stats = DATA.d0_overlap_ids()
    except Exception as e:                              # noqa: BLE001
        return stop(rep, fails + [f"data.d0_overlap_ids() failed: {type(e).__name__}: {e}"])
    ov = sorted(int(r) for r in ov_all if int(r) in A)
    ovs = set(ov)
    rep["overlap"] = dict(ids=ov, n=len(ov), n_outside_status_ok=len(ov_all) - len(ov), stats=ov_stats)
    if len(ov) != N_OVERLAP:
        fails.append(f"{len(ov)} D-1 overlaps among the status-ok D0 rows, expected {N_OVERLAP}")
    pop = sorted(set(A) - ovs)

    # ---- G-2 per-reaction label checks (Q rows complete; reference-free targets, charges, flags, roles = labels_all)
    bad = {k: [] for k in ("missing_q", "status", "fields", "nonfinite", "charge", "flag_async", "role", "ref_free")}
    maxdiff = dict.fromkeys(REF_FREE, 0.0)
    for rid in pop:
        q, a = Q.get(rid), A[rid]
        if q is None:
            bad["missing_q"].append(rid)
            continue
        try:
            if q.get("status") != "ok":
                bad["status"].append(rid)
            vals = [float(q[c]) for c in D1_TCOL.values()]
            if not all(math.isfinite(v) for v in vals):
                bad["nonfinite"].append(rid)
            if (int(q["charge1"]), int(q["charge2"])) != (int(a["charge1"]), int(a["charge2"])):
                bad["charge"].append(rid)
            if bool(q["flag_async"]) != bool(a["flag_async"]):
                bad["flag_async"].append(rid)
            if (a.get("role1"), a.get("role2")) != ("dipole", "dipolarophile"):
                bad["role"].append(rid)
            for t in REF_FREE:
                dv = abs(float(q[D1_TCOL[t]]) - float(a[D1_TCOL[t]]))
                maxdiff[t] = max(maxdiff[t], dv) if math.isfinite(dv) else float("inf")
                if not dv <= EQ_TOL:
                    bad["ref_free"].append(rid)
            float(a["e_frag1_rel_eh"]), float(a["e_frag2_rel_eh"])
        except (KeyError, TypeError, ValueError):
            bad["fields"].append(rid)
    rep["label_checks"] = dict(ref_free_max_abs=maxdiff, tol=EQ_TOL, n_q_rows=len(Q),
                               bad={k: dict(n=len(set(v)), ids=sorted(set(v))[:20]) for k, v in bad.items()})
    for k, v in bad.items():
        if v:
            fails.append(f"label check {k}: {len(set(v))} D0 reactions, e.g. {sorted(set(v))[:10]}")

    # ---- manifest, molecule keys
    man = pd.read_csv(D0_MANIFEST, keep_default_na=False, dtype=str)
    mrid = pd.to_numeric(man.d0_rxn_id, errors="coerce")
    if mrid.isna().any() or mrid.duplicated().any():
        fails.append("d0_manifest.csv d0_rxn_id has blanks or repeats")
    M = {int(r): (dp, dph) for r, dp, dph in zip(mrid.fillna(-1), man.dipole_smiles, man.dipolarophile_smiles)}
    miss_m = sorted(r for r in A if r not in M)
    if miss_m:
        fails.append(f"d0_manifest.csv lacks {len(miss_m)} status-ok D0 reactions, e.g. {miss_m[:10]}")
    if fails:
        return stop(rep, fails)
    K = {}
    for rid in sorted(A):
        dp, dph = M[rid]
        K[rid] = dict(dip_ns=canon(dp, False), dip_iso=canon(dp, True), dph_ns=canon(dph, False),
                      dph_iso=canon(dph, True))
    unparsed = sorted(rid for rid, k in K.items() if any(v is None for v in k.values()))
    if unparsed:
        return stop(rep, [f"RDKit cannot parse the manifest SMILES of {len(unparsed)} D0 reactions, e.g. "
                          f"{unparsed[:10]}"])

    # ---- row rules: ex on the Coley references (labels_all, every status-ok row; build_rowattr.py keys)
    def ex_of(k, keyname):
        e = {rid: float(A[rid][f"e_frag{k}_rel_eh"]) for rid in A}
        emin = {}
        for rid, v in e.items():
            kk = K[rid][keyname]
            emin[kk] = min(emin.get(kk, v), v)
        return {rid: (e[rid] - emin[K[rid][keyname]]) * KCAL for rid in A}
    ex1, ex2 = ex_of(1, "dip_iso"), ex_of(2, "dph_iso")
    R = {}
    for rid in pop:
        q = Q[rid]
        d1v, d2v = float(q[D1_TCOL["d1"]]), float(q[D1_TCOL["d2"]])
        hyg = bool(d1v >= 0 and d2v >= 0 and d2v <= HYG_D2_MAX)
        asy = bool(q["flag_async"])
        # ok0 = the same row rules on the Coley labels (labels_all d1 / d2): only the leaf choice (pr_nog2), and so the
        # component membership and folds, reads it, so a d0_labels.py rerun never moves a fold (LAB review finding)
        a0 = A[rid]
        d1c, d2c = float(a0[D1_TCOL["d1"]]), float(a0[D1_TCOL["d2"]])
        h0 = bool(d1c >= 0 and d2c >= 0 and d2c <= HYG_D2_MAX)
        R[rid] = dict(dip=K[rid]["dip_ns"], dph=K[rid]["dph_iso"], c1=int(q["charge1"]), c2=int(q["charge2"]), hyg=hyg,
                      asy=asy, ex1=ex1[rid], ex2=ex2[rid],
                      ok=bool(hyg and not asy and ex1[rid] <= EX_MAX and ex2[rid] <= EX_MAX),
                      ok0=bool(h0 and not bool(a0["flag_async"]) and ex1[rid] <= EX_MAX and ex2[rid] <= EX_MAX),
                      g2=False, forced=None, twin_of=None)

    # ---- G2 TS availability (anchor priority)
    g2_source, has, g2_info = g2_availability(pop, fails)
    for rid in pop:
        R[rid]["g2"] = bool(has[rid])
    rep["g2_source"], rep["g2_info"] = g2_source, dict(g2_info, n_with_ts=int(sum(has.values())), n_pop=len(pop))
    rep["preliminary"] = g2_source != "g2c"

    # ---- twins (D1 targets with the same stereo-free reactant set)
    T1 = GC.d1_core_table()                             # D1 rxn_id -> (core 'dipole/skeleton', T1 fold, None)
    m1 = pd.read_csv(GC.D1_MANIFEST, keep_default_na=False, dtype=str)
    m1 = m1[m1.kind == "D1"]
    d1_ns, d1_iso, d1_fold, d1_unparsed = {}, {}, {}, []
    for rid_s, dp, dph in zip(m1.meta_rxn_id, m1.dipole_smiles, m1.dipolarophile_smiles):
        if not str(rid_s).strip():
            continue
        rid = int(float(rid_s))
        if rid not in T1:
            continue
        ns, iso = rset(canon(dp, False), canon(dph, False)), rset(canon(dp, True), canon(dph, True))
        if ns is None or iso is None:
            d1_unparsed.append(rid)
            continue
        d1_fold[rid] = int(T1[rid][1])
        d1_ns.setdefault(ns, set()).add(rid)
        d1_iso.setdefault(iso, set()).add(rid)
    if d1_unparsed:
        fails.append(f"RDKit cannot parse the D1 manifest SMILES of {len(d1_unparsed)} D1 targets")
    if len(d1_fold) != len(T1):
        fails.append(f"the D1 manifest covers {len(d1_fold)} of the {len(T1)} D1 targets (C_compare)")

    def fmt(ts):                                        # '<D1 rxn_id>:<T1 fold>' space separated (d0_twins.csv)
        return " ".join(f"{t}:{d1_fold[t]}" for t in sorted(ts))

    tw = {k: [] for k in ("any", "iso", "m1_any") + TWIN_KINDS}
    d1_matched, twin_rows = set(), {}
    for rid in pop:
        T = sorted(d1_ns.get(rset(K[rid]["dip_ns"], K[rid]["dph_ns"]), ()))
        if not T:
            continue
        tw["any"].append(rid)
        d1_matched |= set(T)
        Ti = sorted(d1_iso.get(rset(K[rid]["dip_iso"], K[rid]["dph_iso"]), ()))
        if Ti:
            tw["iso"].append(rid)
        has_m1 = any(not 0 <= d1_fold[t] < N_FOLDS for t in T)          # a twin that is never a T1 test reaction
        if has_m1:
            tw["m1_any"].append(rid)
        Tf = [t for t in T if 0 <= d1_fold[t] < N_FOLDS]
        src = Tf
        if not Tf:
            kind = "m1_only"
        elif len({d1_fold[t] for t in Tf}) == 1:
            kind = "forced"
        else:
            Tfi = [t for t in Tf if t in set(Ti)]
            if Tfi and len({d1_fold[t] for t in Tfi}) == 1:
                src, kind = Tfi, "iso_resolved"
            else:
                src, kind = (Tfi or Tf), "conflict"
        tw[kind].append(rid)
        tt = None
        if kind in ("forced", "iso_resolved") or (kind == "conflict" and accept_conflicts):
            tt = min(src)
            R[rid]["forced"], R[rid]["twin_of"] = d1_fold[tt], tt
        twin_rows[rid] = dict(rxn_id=rid, kind=kind, twin_of="" if tt is None else str(int(tt)), m1=bool(has_m1),
                              n_twins=len(T), twins_ns=fmt(T), twins_iso=fmt(Ti), bind=fmt(src))
    ov_ns = {rset(K[r]["dip_ns"], K[r]["dph_ns"]) for r in ov}
    sib_ns = Counter(rset(K[r]["dip_ns"], K[r]["dph_ns"]) for r in pop)
    n_forced = int(sum(R[r]["forced"] is not None for r in pop))
    rep["twins"] = dict(
        rule="same stereo-free reactant set as a D1 target (C_compare 9,266) -> its T1 fold (forced); several folds -> "
             "the isomeric reactant-set matches decide (iso_resolved; the stereo-only twins of the other folds then "
             "see this D0 reaction in training, as D1 itself trains AppE while AppZ is tested); still several -> "
             "conflict (STOP unless --accept-twin-conflicts: lowest D1 rxn_id); D1 fold -1 twins only -> no forced "
             "fold (m1_only). m1 = any stereo-free twin of T1 fold -1: such a D0 reaction is tested in its D0 fold "
             "while the D1 twin trains in every unit, and GEO1b Y (train_m1, unchanged) trains on it",
        conflict_policy="accepted: lowest D1 rxn_id (--accept-twin-conflicts)" if accept_conflicts else "stop",
        file=str(TWINS_CSV), n_twins=len(tw["any"]), n_twins_isomeric=len(tw["iso"]), n_forced=n_forced,
        by_kind={k: len(tw[k]) for k in TWIN_KINDS}, n_m1_any=len(tw["m1_any"]),
        n_m1_mixed=int(sum(R[r]["forced"] is not None for r in tw["m1_any"])), m1_any=tw["m1_any"],
        m1_only=tw["m1_only"], iso_resolved=[twin_rows[r] for r in tw["iso_resolved"]],
        conflicts=[twin_rows[r] for r in tw["conflict"]], n_conflict=len(tw["conflict"]),
        n_d1_reactions_matched=len(d1_matched), d1_matched_by_fold=_vc(d1_fold[t] for t in d1_matched),
        forced_by_fold=_vc(R[r]["forced"] for r in pop if R[r]["forced"] is not None),
        report_only=dict(
            n_d0_sharing_reactants_with_an_overlap=int(sum(rset(K[r]["dip_ns"], K[r]["dph_ns"]) in ov_ns for r in pop)),
            d0_reactant_sets_with_2plus=int(sum(v >= 2 for v in sib_ns.values())),
            d0_reactions_in_those_sets=int(sum(v for v in sib_ns.values() if v >= 2)),
            n_d0_reactant_sets=len(sib_ns)))
    if tw["conflict"] and not accept_conflicts:
        fails.append(f"{len(tw['conflict'])} D0 twins have D1 twins in several T1 folds that the isomeric match cannot "
                     f"settle (report twins.conflicts), e.g. {tw['conflict'][:10]}: each would be trained while one "
                     f"of its twins is tested. User decision needed; '--accept-twin-conflicts' takes the lowest D1 "
                     f"rxn_id and keeps them listed in d0_twins.csv (kind 'conflict')")

    if fails:
        return stop(rep, fails)

    # ---- components = fold units (one dipole molecule; twin-forced folds split it)
    def pr(rid):                                        # anchor priority
        return (not R[rid]["g2"], not R[rid]["ok"], rid)

    def pr_nog2(rid):                                   # G2- and Q-label-independent order (which component a leaf
        return (not R[rid]["ok0"], rid)                 # joins = component membership = folds; ok0 = Coley labels)

    def chg(rid):
        return (R[rid]["c1"], R[rid]["c2"])

    by_dip = {}
    for rid in pop:
        by_dip.setdefault(R[rid]["dip"], []).append(rid)
    comps, seekers = [], []
    st = Counter()
    for dk in sorted(by_dip):
        rids = sorted(by_dip[dk])
        if len(rids) == 1:
            seekers.append(rids[0])
            st["dipole_singletons"] += 1
            continue
        st["dipole_groups"] += 1
        sets = {}
        for rid in rids:
            sets.setdefault((R[rid]["dph"], R[rid]["c1"], R[rid]["c2"]), []).append(rid)
        forced_n = Counter()                            # over the WHOLE dipole group (every charge class)
        for sk in sorted(sets):
            mem = sets[sk]
            if len({_fkey(R[r]["forced"]) for r in mem}) > 1:
                fails.append(f"sibling set {dk} | {sk} has members with different forced folds")
            f = R[mem[0]]["forced"]
            if f is not None:
                forced_n[f] += len(mem)
        main = _main_fold(forced_n)                     # the unforced sets (any charge) go to this fold
        parts = {}
        for sk in sorted(sets):
            f = R[sets[sk][0]]["forced"]
            parts.setdefault(f if f is not None else main, []).append(sk)
        st["dipole_groups_split_by_twins"] += int(len(parts) > 1)
        for f in sorted(parts, key=_fkey):
            psets = [sorted(sets[sk]) for sk in parts[f]]
            if sum(map(len, psets)) == 1:               # only possible for a twin-forced part
                seekers.append(psets[0][0])
                st["forced_part_singletons"] += 1
                continue
            comps.append(dict(kind="dipole", key=dk, sets=psets, leaves=[], forced=f))
    # leaf attachment of the reactions alone in their fold for their dipole (same dipolarophile molecule, same charges,
    # compatible fold) to any member of a dipole component; they join that component (and its fold)
    cand = {}
    for ci, c in enumerate(comps):
        for mem in c["sets"]:
            for rid in mem:
                cand.setdefault((R[rid]["dph"], R[rid]["c1"], R[rid]["c2"]), []).append((rid, ci))
    rest = []
    for rid in sorted(seekers):
        k = (R[rid]["dph"], R[rid]["c1"], R[rid]["c2"])
        f = R[rid]["forced"]
        cs = [(b, ci) for b, ci in cand.get(k, []) if f is None or comps[ci]["forced"] is None
              or comps[ci]["forced"] == f]
        if not cs:
            rest.append(rid)
            continue
        ci = min(cs, key=lambda t: pr_nog2(t[0]))[1]
        b = min((bb for bb, cj in cs if cj == ci), key=pr)
        comps[ci]["leaves"].append((rid, b))
        if f is not None:
            comps[ci]["forced"] = f
        cand.setdefault(k, []).append((rid, ci))
        st["leaves_attached"] += 1
    # the rest: leaf components among themselves (same dipolarophile molecule and charges), else singletons
    groups = {}
    for rid in rest:
        groups.setdefault((R[rid]["dph"], R[rid]["c1"], R[rid]["c2"]), []).append(rid)
    singles = []
    for k in sorted(groups):
        rids = sorted(groups[k])
        if len(rids) == 1:
            singles.append(rids[0])
            continue
        forced_n = Counter(R[r]["forced"] for r in rids if R[r]["forced"] is not None)
        main = _main_fold(forced_n)
        parts = {}
        for r in rids:
            parts.setdefault(R[r]["forced"] if R[r]["forced"] is not None else main, []).append(r)
        for f in sorted(parts, key=_fkey):
            mem = sorted(parts[f])
            if len(mem) == 1:
                singles.append(mem[0])
                continue
            comps.append(dict(kind="leaf", key=k[0], sets=[mem], leaves=[], forced=f))
            st["leaf_components"] += 1
    for rid in sorted(singles):
        comps.append(dict(kind="single", key=R[rid]["dip"], sets=[[rid]], leaves=[], forced=R[rid]["forced"]))
    for c in comps:
        c["members"] = sorted([r for mem in c["sets"] for r in mem] + [a for a, _ in c["leaves"]])
    comps.sort(key=lambda c: c["members"][0])
    for i, c in enumerate(comps):
        c["id"] = f"c{i:04d}"
    cid_of = {r: c["id"] for c in comps for r in c["members"]}
    comp_by_id = {c["id"]: c for c in comps}

    # ---- anchor trees: one per charge class of a component (n members, m charge classes -> n - m edges)
    for c in comps:
        out, trees = [], {}
        if c["kind"] == "dipole":
            by_ch = {}
            for mem in c["sets"]:                       # a sibling set has one charge class (its key)
                by_ch.setdefault(chg(mem[0]), []).append(mem)
            for ch in sorted(by_ch):
                heads = []
                for mem in by_ch[ch]:
                    h = min(mem, key=pr)
                    heads.append(h)
                    out += [(a, h, "sibling") for a in mem if a != h]
                top = min(heads, key=pr)
                out += [(h, top, "dipole") for h in sorted(heads) if h != top]
                trees[ch] = top
            out += [(a, b, "leaf") for a, b in c["leaves"]]
        elif c["kind"] == "leaf":
            top = min(c["sets"][0], key=pr)
            out += [(a, top, "leaf") for a in c["sets"][0] if a != top]
            trees[chg(top)] = top
        else:
            trees[chg(c["sets"][0][0])] = c["sets"][0][0]
        size = Counter(chg(r) for r in c["members"])
        c["trees"], c["edges"] = trees, out
        c["anchor"] = min(trees.values(), key=lambda a: (-size[chg(a)], pr(a)))   # anchor of the largest tree
        st["charge_trees"] += len(trees)
        st["charge_trees_without_pair"] += int(sum(size[ch] == 1 for ch in trees))
        st["components_multi_charge"] += int(len(trees) > 1)

    # ---- folds: forced components; then the components with a pair by pair load (d1_split rule); then the components
    # without a pair by reaction count. Loads use the structural pair count, never anchors / G2 (P-fold-load)
    fold_of, load_p, load_r = {}, [0] * N_FOLDS, [0] * N_FOLDS
    npairs = {c["id"]: len(c["edges"]) for c in comps}
    nmem = {c["id"]: len(c["members"]) for c in comps}
    for c in comps:
        if c["forced"] is not None:
            k = int(c["forced"])
            fold_of[c["id"]] = k
            load_p[k] += npairs[c["id"]]
            load_r[k] += nmem[c["id"]]
    rng = np.random.default_rng(SEED)
    free_pair = sorted(c["id"] for c in comps if c["forced"] is None and npairs[c["id"]] >= 1)
    perm_pair = [str(x) for x in rng.permutation(np.array(free_pair, dtype=object))]
    for cid in perm_pair:
        k = int(np.argmin(load_p))
        fold_of[cid] = k
        load_p[k] += npairs[cid]
        load_r[k] += nmem[cid]
    free_nopair = sorted(c["id"] for c in comps if c["forced"] is None and npairs[c["id"]] == 0)
    perm_nopair = [str(x) for x in rng.permutation(np.array(free_nopair, dtype=object))]
    for cid in perm_nopair:
        k = int(np.argmin(load_r))
        fold_of[cid] = k
        load_r[k] += nmem[cid]

    # ---- G-3 structure gates
    cnt = Counter(r for c in comps for r in c["members"])
    if set(cnt) != set(pop) or max(cnt.values()) != 1:
        fails.append("components do not partition the population exactly once")
    if set(fold_of) != set(comp_by_id) or any(not 0 <= v < N_FOLDS for v in fold_of.values()):
        fails.append("a component has no fold in 0-4")
    for c in comps:
        mem, E = set(c["members"]), c["edges"]
        charges = {chg(r) for r in mem}
        anchors = set(c["trees"].values())
        if set(c["trees"]) != charges or any(chg(a) != ch or a not in mem for ch, a in c["trees"].items()):
            fails.append(f"{c['id']}: not one anchor per charge class")
        if c["kind"] != "dipole" and len(charges) != 1:
            fails.append(f"{c['id']}: {c['kind']} component with charges {sorted(charges)}")
        if len(E) != len(mem) - len(charges):
            fails.append(f"{c['id']}: {len(E)} edges for {len(mem)} members in {len(charges)} charge classes")
        ac = Counter(a for a, _, _ in E)
        if set(ac) != mem - anchors or (ac and max(ac.values()) != 1):
            fails.append(f"{c['id']}: every member but the tree anchors must be 'a' exactly once")
        parent = {r: r for r in mem}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for a, b, kind in E:
            if a == b or a not in mem or b not in mem or kind not in EDGE_KINDS:
                fails.append(f"{c['id']}: bad edge {a} -> {b} ({kind})")
                continue
            if chg(a) != chg(b):
                fails.append(f"{c['id']}: edge {a} -> {b} links different charges")
            if kind == "sibling" and (R[a]["dip"], R[a]["dph"]) != (R[b]["dip"], R[b]["dph"]):
                fails.append(f"{c['id']}: sibling edge {a} -> {b} with different reactants")
            if kind == "dipole" and R[a]["dip"] != R[b]["dip"]:
                fails.append(f"{c['id']}: dipole edge {a} -> {b} with different dipoles")
            if kind == "leaf" and R[a]["dph"] != R[b]["dph"]:
                fails.append(f"{c['id']}: leaf edge {a} -> {b} with different dipolarophiles")
            parent[find(a)] = find(b)
        if mem and (len({find(r) for r in mem}) != len(charges) or len({find(a) for a in anchors}) != len(anchors)):
            fails.append(f"{c['id']}: the anchor trees are not one connected tree per charge class")
    for rid in pop:
        tt = R[rid]["twin_of"]
        if tt is not None and fold_of[cid_of[rid]] != d1_fold[tt]:
            fails.append(f"twin {rid}: fold {fold_of[cid_of[rid]]} != fold {d1_fold[tt]} of D1 {tt}")
    # one dipole molecule = one fold, except the folds its twins force (finding 1 of the PAIR review)
    span = []
    for dk in sorted(by_dip):
        rids = by_dip[dk]
        folds = {fold_of[cid_of[r]] for r in rids}
        if len(folds) == 1:
            continue
        free = {fold_of[cid_of[r]] for r in rids if R[r]["forced"] is None}
        forced = {int(R[r]["forced"]) for r in rids if R[r]["forced"] is not None}
        span.append(dict(dipole=dk, n_rxn=len(rids), folds=sorted(folds), forced_folds=sorted(forced),
                         unforced_fold=sorted(free)))
        if len(free) > 1 or not free <= forced or not folds <= forced | free:
            fails.append(f"dipole {dk}: reactions in folds {sorted(folds)} not explained by its twins (forced "
                         f"{sorted(forced)}, unforced {sorted(free)})")
    if fails:
        return stop(rep, fails[:200])

    # ---- tables
    frows = []
    for rid in sorted(A):
        if rid in ovs:
            frows.append(dict(rxn_id=rid, component="", fold=FOLD_OVERLAP, twin_of="", overlap=True))
        else:
            tt = R[rid]["twin_of"]
            frows.append(dict(rxn_id=rid, component=cid_of[rid], fold=fold_of[cid_of[rid]],
                              twin_of="" if tt is None else str(int(tt)), overlap=False))
    folds_text = pd.DataFrame(frows, columns=FOLD_COLS).to_csv(index=False)
    prow = []
    for c in comps:
        for a, b, kind in c["edges"]:
            d = dict.fromkeys(F_COLS + EXTRA_COLS, "")
            d.update(ref_kind="d0", a_job=f"D0G2_{a:04d}", a_rxn=int(a), b_job=f"D0G2_{b:04d}", b_rxn=int(b), pos="d0",
                     dipole="D0", skeleton=c["id"], fam=f"D0/{c['id']}", core=f"D0/{c['id']}", sigma=float("nan"),
                     hyg=bool(R[a]["hyg"] and R[b]["hyg"]), edge_kind=kind, ok=bool(R[a]["ok"] and R[b]["ok"]))
            for t, col in D1_TCOL.items():
                d[f"dd_{t}"] = float(Q[a][col]) - float(Q[b][col])
            prow.append(d)
    for i, d in enumerate(prow):
        d["pair_id"] = f"D{i:05d}"
    P = pd.DataFrame(prow, columns=F_COLS + EXTRA_COLS)
    pairs_text = P.to_csv(index=False)
    crow = []
    for c in comps:
        n_leaf = sum(kind == "leaf" for _, _, kind in c["edges"])
        crow.append(dict(component=c["id"], kind=c["kind"], key=c["key"],
                         charges=" ".join(f"{a}/{b}" for a, b in sorted(c["trees"])), n_rxn=len(c["members"]),
                         n_pairs=len(c["edges"]), n_trees=len(c["trees"]),
                         n_sets=len(c["sets"]) if c["kind"] == "dipole" else 0, n_leaves=n_leaf,
                         fold=fold_of[c["id"]], forced=c["forced"] is not None, anchor=c["anchor"],
                         anchors=" ".join(str(c["trees"][ch]) for ch in sorted(c["trees"]))))
    comp_text = pd.DataFrame(crow, columns=COMP_COLS).to_csv(index=False)
    trow = [dict(twin_rows[rid], fold=fold_of[cid_of[rid]], component=cid_of[rid]) for rid in sorted(twin_rows)]
    twins_text = pd.DataFrame(trow, columns=TWIN_COLS).to_csv(index=False)

    # ---- G-4 frozen folds
    if FOLDS_CSV.is_file() and FOLDS_CSV.read_text() != folds_text and not force:
        return stop(rep, ["d0_folds.csv exists and would change (frozen folds: GEO1b may have read it). Find the "
                          "input that changed; '--force' only if no GEO1b D / R output has used the old file"])
    rep["folds_rewritten"] = bool(FOLDS_CSV.is_file() and FOLDS_CSV.read_text() != folds_text)

    write_text_atomic(FOLDS_CSV, folds_text)
    write_text_atomic(PAIRS_CSV, pairs_text)
    write_text_atomic(COMP_CSV, comp_text)
    write_text_atomic(TWINS_CSV, twins_text)

    # ---- report
    P["fold"] = [fold_of[cid_of[a]] for a in P.a_rxn]
    P["g2_both"] = [R[a]["g2"] and R[b]["g2"] for a, b in zip(P.a_rxn, P.b_rxn)]
    P["charge"] = [f"{R[a]['c1']}/{R[a]['c2']}" for a in P.a_rxn]
    P["m1_twin"] = [twin_rows.get(a, {}).get("m1", False) or twin_rows.get(b, {}).get("m1", False)
                    for a, b in zip(P.a_rxn, P.b_rxn)]

    def psum(g):
        return dict(n_pairs=int(len(g)), n_hyg=int(g.hyg.sum()), n_ok=int(g.ok.sum()),
                    n_ok_g2_both=int((g.ok & g.g2_both).sum()), n_ok_m1_twin=int((g.ok & g.m1_twin).sum()),
                    by_edge_kind=_vc(g.edge_kind), ok_by_edge_kind=_vc(g.edge_kind[g.ok]))
    per_fold = {}
    for k in range(N_FOLDS):
        cs = [c for c in comps if fold_of[c["id"]] == k]
        rr = [r for c in cs for r in c["members"]]
        per_fold[str(k)] = dict(n_rxn=len(rr), n_rxn_g2=int(sum(R[r]["g2"] for r in rr)),
                                n_rxn_ok=int(sum(R[r]["ok"] for r in rr)), n_components=len(cs),
                                n_forced_components=int(sum(c["forced"] is not None for c in cs)),
                                n_singletons=int(sum(len(c["members"]) == 1 for c in cs)),
                                n_rxn_m1_twin=int(sum(twin_rows.get(r, {}).get("m1", False) for r in rr)),
                                **psum(P[P.fold == k]))
    okp = P[P.ok]
    chs = sum(P[f"dd_{t}"] for t in D1_TCOL if t != "barrier")
    rep.update(
        stop=False, status="ok" if g2_source == "g2c" else "ok_preliminary",
        finished=time.strftime("%Y-%m-%dT%H:%M:%S"), elapsed_s=round(time.time() - t0, 1),
        population=dict(n_status_ok=len(A), n_overlap=len(ov), n=len(pop),
                        n_charged=int(sum((R[r]["c1"], R[r]["c2"]) != (0, 0) for r in pop)),
                        charge_classes=_vc(f"{R[r]['c1']}/{R[r]['c2']}" for r in pop)),
        row_rules=dict(n_flag_async=int(sum(R[r]["asy"] for r in pop)),
                       n_ex1_gt20=int(sum(R[r]["ex1"] > EX_MAX for r in pop)),
                       n_ex2_gt20=int(sum(R[r]["ex2"] > EX_MAX for r in pop)),
                       n_hygiene_fail=int(sum(not R[r]["hyg"] for r in pop)), n_ok=int(sum(R[r]["ok"] for r in pop)),
                       ex_rule="ex_k = (e_frag{k}_rel_eh - lowest e_frag{k}_rel_eh of the same molecule over the 5,260 "
                               "status-ok labels_all rows) x 627.5094740631; molecule = RDKit canonical isomeric SMILES "
                               "of the manifest dipole / dipolarophile (build_rowattr.py rule)"),
        components=dict(n=len(comps), by_kind=_vc(c["kind"] for c in comps),
                        n_forced=int(sum(c["forced"] is not None for c in comps)), structure=dict(st),
                        size_quantiles=[float(x) for x in np.quantile([len(c["members"]) for c in comps],
                                                                      [0.5, 0.9, 0.99, 1.0])],
                        n_rxn_in_multi=int(sum(len(c["members"]) for c in comps if len(c["members"]) >= 2)),
                        dipole_fold_span=dict(n_dipoles_multi_fold=len(span),
                                              n_rxn=int(sum(s["n_rxn"] for s in span)),
                                              rule="a dipole molecule spans several folds only where its twins force "
                                                   "them (gated)", dipoles=span)),
        folds=dict(rule="forced (twin) components first; the other components with a pair in rng(20261019) "
                        "permutation order into the fold with the fewest pairs (ties lower fold); the components "
                        "without a pair in a second permutation of the same generator into the fold with the fewest "
                        "reactions", load_pairs=load_p, load_rxn=load_r, perm_pair_head=perm_pair[:20],
                   per_fold=per_fold),
        pairs=dict(**psum(P), by_charge={ch: psum(g) for ch, g in P.groupby("charge")},
                   n_b_without_g2=int(sum(not R[b]["g2"] for b in P.b_rxn)),
                   n_a_g2_b_without_g2=int(sum(R[a]["g2"] and not R[b]["g2"] for a, b in zip(P.a_rxn, P.b_rxn))),
                   mean_abs_dd_ok={k: {t: float(g[f"dd_{t}"].abs().mean()) for t in D1_TCOL}
                                   for k, g in okp.groupby("edge_kind")},
                   max_abs_barrier_minus_sum9=float((P.dd_barrier - chs).abs().max()) if len(P) else 0.0,
                   expected="about 3,000-3,500 (design); fixed by this run"),
        decisions="results/DECISIONS_PAIR.md",
        outputs={str(p): dict(sha256=GC.sha256(p), bytes=p.stat().st_size) for p in OUTPUTS},
        fails=[])
    GC.write_json(REPORT, rep)
    print(json.dumps(dict(g2_source=g2_source, population=rep["population"], row_rules=rep["row_rules"],
                          components=rep["components"]["by_kind"], twins={k: v for k, v in rep["twins"].items()
                                                                          if k.startswith("n_")},
                          dipole_fold_span=len(span),
                          pairs={k: rep["pairs"][k] for k in ("n_pairs", "n_ok", "n_ok_g2_both", "by_edge_kind")},
                          per_fold={k: (v["n_rxn"], v["n_pairs"], v["n_ok"]) for k, v in per_fold.items()}),
                     indent=1, default=str), flush=True)
    if g2_source != "g2c":
        print("PRELIMINARY: anchors from the D0-G2 orig folders; rerun after select_g2c.py build", file=sys.stderr)
    return 0


def main_build(force, accept_conflicts):
    try:
        return cmd_build(force, accept_conflicts)
    except Exception as e:                              # noqa: BLE001
        GC.write_json(REPORT, dict(code_version=CODE_VERSION, stop=True, status="crash",
                                   error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-4000:]))
        traceback.print_exc()
        return 3


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["build"] and set(args[1:]) <= {"--force", "--accept-twin-conflicts"}:
        sys.exit(main_build("--force" in args[1:], "--accept-twin-conflicts" in args[1:]))
    print(__doc__, file=sys.stderr)
    sys.exit(2)
