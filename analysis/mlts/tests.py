#!/usr/bin/env python3
r"""tests.py — DEPLOY-V2 (MLTS) Phase C aggregate (docs/specs/DEPLOY_V2_MLTS.md 8) -> results/C_tests.json, and the
K-unit leakage check C2-K -> results/C2K_report.json. Compute nodes only (jobs/mlt.slurm or jobs/sbx.sh).

  python tests.py all [options]          C-1 .. C-7 -> results/C_tests.json {complete, all_pass, stop, n_todo, est,
                                         stand_in, C1..C7 {pass, detail}}   (MLTENV=mlts, 1 GPU, 16 CPUs, 64 GB)
  python tests.py c1|c2|c3|c4|c5|c6|c7 [options]   one section, merged into the same json
  python tests.py c2_k [--units-dir D]   K channel units: f{k} training pairs ∩ fold k = ∅ -> results/C2K_report.json
                                         {complete, pass, n_units, n_overlap, ...}               (MLTENV=reactot, CPU)
  python tests.py c3_files <file.inp|.tar[.gz]> ...   the C-3 word check on given files (prints JSON, writes nothing)
  internal: _c7pred <variant root> <folds> [--src label --out D]  (re-executes itself in env fu2 = env_d1fu2; the
            C1 atom-unit path; --src label = the H-1 label root, deployed yhat vs L_predictions: re-expansion test)
options (tests / stand-ins): --model M, --c1-json P, --no-rerun, --c1-force, --c2-splits a,b, --c2-data DIR (negative
  control), --c3-root DIR, --c3-n N, --c7-src prod|g2, --c7-root DIR, --c7-pairs N, --c7-delta A, --c7-max-atoms N,
  --c7-stage all|plan|chain|pred|report, --c7-variant v0|p|m, --c7-steps a,b, --src-results DIR (read H1 / C6 /
  C1_tests / ft/*_data / K_plan / K_report from DIR; outputs always go to results/ = $MLT_RESULTS).
exit 0 done (C-1..C-6 all pass), 2 not ready / unfinished (same command again), 3 STOP (a C-1..C-6 check failed; C2-K
overlap). Every write is atomic (mlt_common.write_*); reruns reuse what is still valid; with < 2 h of the 48 h wall left
(MLT_JOB_T0) no new C-7 chain step or C-1 rerun is started (exit 2).

Sections (spec 8 table; the stand-ins used when production inputs do not exist yet are recorded under 'stand_in')
  C-1  tests_c1.py (A-1: bridge, optimizer, Hessian) for the C-1 model: --model, else FT_0_al, else the newest of
       FT_0 / FTpilot_<cand>_0 (+_al) model cards in $MLT/models (recorded). Sections of results/C1_tests.json for that
       model are reused when their 'time' is not older than the model card ('written', else file mtime); otherwise the
       missing / stale sections are rerun in-process (tests_c1.main). pass = pass_C1.
  C-2  FT leakage, three independent ways, for every fold<k> / lodo<d> split [_al] used by a model card in $MLT/models
       (or --c2-splits): (a) finetune's own check results/ft/<run>_data.json leakage_C2.pass_; (b) recomputed here from
       $MLT/ft/data/<split>/{train,valid}.xyz tags vs mlt_common.targets(): no rxn_id of the held-out fold / dipole, no
       held-out fold / dipole tag; (c) geometric: no training structure with the same element sequence as a held-out
       label TS ($LAD/dft_final, label order) within Kabsch RMSD 0.10 A (untagged copy). Model card data_fingerprint =
       split meta.json fingerprint (the model was trained on this split). The K-unit half of C-2 is `c2_k` (after K).
  C-3  50 autodE-MLTS ORCA inputs: 50 job folders of $MLT/ade/jobs (seeded), one random .inp member of each inputs
       tar (+ every .inp of those tars, 'extended'): no functional / basis / solvation (CPCM, SMD, ALPB, COSMO) /
       dispersion-RI / xTB word on '!' lines, no %cpcm / %basis / smdsolvent / solvent word in blocks (quoted ProgExt /
       Ext_Params strings excluded, listed separately), and 'ExtOpt' on a '!' line.
  C-4  tests_c1.py units section of the C-1 model (MLTS E / F vs ORCA label-level EnGrad on water, methane, 3
       displaced label TSs). pass = pass_C4.
  C-5  results/H1_report.json pass (H-1 reproduction).        C-6  results/C6_report.json passed (C1 regression).
  C-7  report only (spec 8 '보고'): N para P_main test pairs (3 per dipole, seeded, both ends with a usable MLTS input
       row), the a-side MLTS TS pushed along both forming bonds by +-delta (each forming bond +-0.01 A, the two bond
       atoms moved symmetrically), three variant roots v0 / p / m under $MLT/c7 with the A-2 layout, the Phase H chain
       (inputs_mlts.py ... --root --ids, refs own), the C1 atom-unit path (geo2 predict_atoms.unit_run on the fold's
       LADDER dft units), disp = D3_int (H-0). Prediction changes |yhat(+-) - yhat(v0)| and HF input changes are compared
       with the SUMMARY_G3 E-2 noise table (d1_g3/results/E_noise.csv, scope common, class same) raw and linearly scaled
       to delta (x delta / its |dd| median). No label y is stored and no error is computed (not a channel result).

Design decisions (also in the json 'decisions'; for results/DECISIONS.md) — see DECISIONS below.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import csv  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import tarfile  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
THIS = Path(__file__).resolve()
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import mlt_common as M  # noqa: E402            (stdlib only at import)
import inputs_mlts as IM  # noqa: E402          (stdlib only at import; src_cfg / ensure_env / the Phase H chain)

CODE_VERSION = "mlts-c-1"
OUT = M.RES / "C_tests.json"
C2K_OUT = M.RES / "C2K_report.json"
SECTIONS = ("C1", "C2", "C3", "C4", "C5", "C6", "C7")
GATED = ("C1", "C2", "C3", "C4", "C5", "C6")          # a failure = exit 3; C-7 is report only
T0 = float(os.environ.get("MLT_JOB_T0") or time.time())
WALL_H, RESERVE_H = 48.0, 2.0
REPO = M.REPO
FU2_S_SPLITS = REPO / "analysis" / "d1_pair" / "results" / "fu2" / "S_splits.json"
G3_RES = REPO / "analysis" / "d1_g3" / "results"
E2_NOISE, E2_GEOM = G3_RES / "E_noise.csv", G3_RES / "E_geometry.csv"
C1_REPS = M.MLT / "preds" / "C1_representatives.json"
MODELS_DIR = M.MLT / "models"
FT_DATA = M.MLT / "ft" / "data"
SRC = {"res": M.RES}                 # where existing reports are READ (--src-results, tests); outputs always go to M.RES


def src_res():
    return Path(SRC["res"])
H_TARGETS = ("d1", "d2", "elst", "pauli", "oi", "disp", "cpcm", "cds", "barrier")
C2_RMSD_A = 0.10                   # = mlt_common.SAME_TS rmsd (D1 same-TS rule)
C2_SPEC_A = 0.03                   # RMS of sorted interatomic distances (atom-order free); same-core position isomers
                                   # of the negative control sat at 0.043-0.050 A, held-out TS-path frames down to 0.006
C3_N = 50
C7_DELTA_A = 0.01
C7_PAIRS = 15
C7_VARIANTS = ("v0", "p", "m")
C7_SIGN = {"v0": 0.0, "p": 1.0, "m": -1.0}
C7_MAX_FAILS = 2                   # C-7-terminal: chain failures in this many runs -> ready, pass false (report only)
C1_SECTIONS = ("bridge", "optimizer", "hessian", "units")

DECISIONS = {
    "C-agg-exit": "exit 3 when any of C-1 .. C-6 fails (spec 14 lists C-2, C-6 and H-1 = C-5; PIPELINE lists C-1; C-3 / "
                  "C-4 failures are bugs that spec header says to report before fixing = HALT); C-7 is report only "
                  "(pass = the report was produced for >= 80 % of the sampled pairs, judged false).",
    "C-1-model": "--model, else FT_0_al, else the newest FT_0 / FTpilot_<cand>_0 (+_al) model card (task: 'FT_0_al or the "
                 "newest FT_0 model'); C1_tests.json sections are reused when not older than the model card.",
    "C-2-three-ways": "finetune leakage_C2 + recomputation from the split extxyz tags vs targets() + a geometric check "
                      "(same element sequence as a held-out label TS and Kabsch RMSD <= 0.10 A = untagged copy); train "
                      "AND valid files count as used (validation steers early stopping); model card fingerprint must "
                      "equal the split meta fingerprint.",
    "C-2-scope": "splits = those of the model cards in $MLT/models with split fold<k>|lodo<d>[_al] (FT_all splits hold "
                 "nothing out: counted, not checked); none -> not ready unless --c2-splits.",
    "C-3-sample": "50 seeded job folders, one random .inp member of each inputs tar = the 50-file sample; every .inp of "
                  "those tars is checked too ('extended'); both must be clean. Words: functional (incl. HF / MP2 / "
                  "semi-empirical), basis, solvation (CPCM SMD ALPB COSMO GBSA), dispersion / RI (D3BJ D4 RIJCOSX ...), "
                  "xTB on '!' lines; %cpcm / %basis / smdsolvent / cpcm / smd / alpb / b3lyp / def2 in blocks; quoted "
                  "ProgExt / Ext_Params strings are paths and arguments: excluded, hits there listed separately.",
    "C-7-perturbation": "mode 'bond': each forming bond (ts_result.json formed_pairs_ts, autodE order) lengthened (p) or "
                        "shortened (m) by delta = 0.01 A, its two atoms moved by delta/2 along the bond (the literal "
                        "'push along the forming bonds'); the two [3+2] forming bonds share no atom (checked).",
    "C-7-side": "only the a (substituted) reaction of each pair is perturbed; the reference b keeps its MLTS TS, so the "
                "pair prediction change is the per-reaction sensitivity (no cancellation between a and b).",
    "C-7-pairs": "15 para P_main T1 test pairs, 3 per dipole, seeded (SEED + 7), distinct a, a never a selected b; both "
                 "ends usable in the production rung table (stand-in: G2 ok).",
    "C-7-chain": "three full Phase H chains (v0, p, m roots; inputs_mlts.py --root --ids, refs own, no h2) so every "
                 "input (xTB Freq tokens, X182, HF/6-31G* 9 values, D3_int) is recomputed on the pushed TS; v0 is "
                 "compared with the production rung table (reproduction info).",
    "C-7-elements": "a small subset lacks some of the 9 elements, so its token report lists fewer elements than the "
                    "LADDER dft rung; the element one-hot is re-expanded to the dft element order in-process (absent "
                    "elements = zero columns, exact) before the layout check and unit_run.",
    "C-7-models": "atom units AW / GNN / TF of both arms on the pair's fold (geo2 unit_run, saved seeds, recomputed "
                  "std); REP8 per target from C1_representatives.json; REP8 = XGB targets: disp -> D3_int (H-0), cds -> "
                  "the arm-S REP atom model (XGB final refits not repeated for a sensitivity report).",
    "C-7-E2": "reference = E_noise.csv scope common, class same (the SUMMARY_G3 E-2 'same' table); linear scaling "
              "noise x delta / |dd| median of that level (E_geometry.csv, first order: channels are not stationary at "
              "the TS). Category per target: ratio to the scaled g3r value < 1/3 smaller, <= 3 same order, else larger.",
    "C-7-tmpdir": "each variant chain runs with its own node-local TMPDIR (removed afterwards): the D1-HF runner's work "
                  "folder $TMPDIR/d1hf_<SLURM_JOB_ID>/<run>/<T> is otherwise the same for v0 / p / m.",
    "C-7-no-labels":"C-7 files keep yhat / base / yhat_seed_sd only (y dropped) and no error is computed, so C-7 is not "
                     "a channel result for the pre-registration.",
    "C-7-terminal": "C-7 is report only: a Phase H chain that fails in 2 separate runs ends the section as ready, pass "
                    "false, not judged (C_tests complete; never a HALT, never an endless exit 2).",
    "C2-K-source": "training pairs of a K unit: an explicit list in the unit json (train_pair_ids | train_pairs | "
                   "pairs_train | fit_pair_ids) or a sidecar <bare unit name>__train_pairs.{parquet,csv,json} next to "
                   "the unit json (a unit with a train_k K record and no sidecar = unresolved = fail); else the FU2 T1 "
                   "split train list of its fold (superset of the usable training pairs; verified when the unit "
                   "fingerprint s_splits equals the FU2 S_splits.json sha256). Overlap = training pair in the fold-k "
                   "test list or touching a fold-k reaction (fold_T1 = k; anchors -1 never do); K test predictions "
                   "outside the fold-k test list are counted too.",
}


# ================================================================ helpers
def now():
    return M.now()


def log(msg):
    print(f"[C {time.time() - T0:7.0f}s] {msg}", flush=True)


def hours_left():
    return WALL_H - (time.time() - T0) / 3600.0


class NotReady(Exception):
    """a section prerequisite is missing: the section is recorded not ready (reason kept), the command exits 2."""


def not_ready(msg):
    print(f"[NOT READY] {msg}", file=sys.stderr, flush=True)
    raise NotReady(msg)


def need_time(what):
    if hours_left() < RESERVE_H:
        not_ready(f"< {RESERVE_H} h of the {WALL_H} h wall left: {what} not started (resubmit)")


def rj(p):
    try:
        return M.read_json(p)
    except Exception:                                               # noqa: BLE001
        return None


def jobtag():
    j, a = os.environ.get("SLURM_JOB_ID"), os.environ.get("SLURM_ARRAY_TASK_ID")
    return f"{j}_{a}" if j and a else (j or f"pid{os.getpid()}")


def clean(o):
    return IM._clean(o)


def qs(v):
    import numpy as np
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], dtype=float)
    if not len(v):
        return dict(n=0)
    return dict(n=int(len(v)), median=float(np.median(v)), p90=float(np.quantile(v, 0.9)), max=float(v.max()),
                mean=float(v.mean()))


def section(ok, detail, ready=True, **kw):
    return dict(ready=bool(ready), **{"pass": (None if not ready else (None if ok is None else bool(ok)))},
                detail=detail, **kw)


def not_ready_sec(why, **detail):
    return section(None, dict(why=why, **detail), ready=False)


# ================================================================ C-1 / C-4 (tests_c1.py)
C1_NAME = re.compile(r"(FT_0|FT_0_al|FTpilot_.+_0|FTpilot_.+_0_al)")


def pick_c1_model(o):
    if o.model:
        return o.model, "given (--model)"
    if (MODELS_DIR / "FT_0_al.json").is_file():
        return "FT_0_al", "FT_0_al model card"
    c = sorted((p for p in MODELS_DIR.glob("*.json") if C1_NAME.fullmatch(p.stem)), key=lambda p: p.stat().st_mtime)
    if c:
        return c[-1].stem, f"newest FT_0 model card ({c[-1].name}; FT_0_al absent)"
    return None, "no FT_0_al / FT_0 / FTpilot_*_0 model card in $MLT/models"


def model_written(model):
    """ISO time the model was written: its card 'written', else the weights' mtime."""
    card = MODELS_DIR / f"{model}.json"
    if card.is_file():
        c = rj(card) or {}
        if c.get("written"):
            return str(c["written"]), str(card)
        return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(card.stat().st_mtime)), str(card)
    p = Path(model)
    if p.is_file():
        return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(p.stat().st_mtime)), str(p)
    return None, None


def c1_name(model):
    import mlts_bridge as B
    return B.spec_name(B.resolve_spec(model))


def c1_entry(o, model, rerun_missing=True):
    """the C1_tests.json entry of `model` with bridge / optimizer / hessian / units not older than the model."""
    name = c1_name(model)
    src = Path(o.c1_json) if o.c1_json else src_res() / "C1_tests.json"
    wt, wsrc = model_written(model)
    J = rj(src) or {}
    m = (J.get("models") or {}).get(name) or {}
    stale = [s for s in C1_SECTIONS if getattr(o, "c1_force", False) or s not in m
             or (wt and str(m[s].get("time") or "") < wt) or "error" in m[s]]
    info = dict(model=model, name=name, written=wt, written_src=wsrc, source=str(src), reused=[s for s in C1_SECTIONS
                                                                                              if s not in stale])
    if stale and rerun_missing and not o.no_rerun:
        need_time(f"tests_c1 {stale}")
        import tests_c1 as TC
        info["rerun"] = {}
        for s in stale:
            t = time.time()
            rc = TC.main([s, model])
            info["rerun"][s] = dict(rc=rc, wall_s=round(time.time() - t, 1))
        J = rj(TC.OUT) or {}
        info["source"] = str(TC.OUT)
        m = (J.get("models") or {}).get(name) or {}
        stale = [s for s in C1_SECTIONS if s not in m or (wt and str(m[s].get("time") or "") < wt)]
    info["stale_or_missing"] = stale
    return m, info


def _c1_brief(m):
    b, op, h = m.get("bridge") or {}, m.get("optimizer") or {}, m.get("hessian") or {}
    rows_b = b.get("rows") or []
    rows_o = op.get("rows") or []
    rows_h = h.get("rows") or []
    return dict(
        bridge=dict(passed=b.get("pass"), max_dE_eh=max((r.get("dE_eh", math.inf) for r in rows_b), default=None),
                    max_dG_eh_bohr=max((r.get("dG_eh_bohr", math.inf) for r in rows_b), default=None),
                    n=len(rows_b), time=b.get("time"), gpu=b.get("gpu")),
        optimizer=dict(passed=op.get("pass"), pass_production=op.get("pass_production"),
                       dform_max_A=[r.get("dform_max_A") for r in rows_o],
                       max_dform_where_both_converged_A=op.get("max_dform_where_both_converged_A"), n=len(rows_o),
                       time=op.get("time")),
        hessian=dict(passed=h.get("pass"), n_imag=[((r.get("orca") or {}).get("n_imag"), (r.get("mlip") or {}).get("n_imag"))
                                                   for r in rows_h], time=h.get("time")),
        tolerances=dict(bridge_dE_eh=1e-6, bridge_dG_eh_bohr=1e-5, optimizer_dform_A=0.005))


def run_c1(o, ctx):
    model, why = pick_c1_model(o)
    if model is None:
        return not_ready_sec(why)
    m, info = c1_entry(o, model)
    ctx["c1"] = (m, info, why)
    if info["stale_or_missing"] and any(s in info["stale_or_missing"] for s in C1_SECTIONS[:3]):
        return not_ready_sec(f"C1_tests.json lacks fresh {info['stale_or_missing']} for {info['name']}", **info)
    ok = bool(m.get("bridge", {}).get("pass") and m.get("optimizer", {}).get("pass") and m.get("hessian", {}).get("pass"))
    return section(ok, dict(selection=why, **info, checks=_c1_brief(m), pass_C1=m.get("pass_C1")))


def run_c4(o, ctx):
    if "c1" not in ctx:
        model, why = pick_c1_model(o)
        if model is None:
            return not_ready_sec(why)
        m, info = c1_entry(o, model)
        ctx["c1"] = (m, info, why)
    m, info, why = ctx["c1"]
    u = m.get("units")
    if not u or "units" in info["stale_or_missing"]:
        return not_ready_sec(f"C1_tests.json lacks a fresh units section for {info['name']}", **info)
    ts = [r for r in u.get("rows") or [] if str(r.get("tag", "")).startswith("ts_")]
    det = dict(model=info["model"], name=info["name"], selection=why, time=u.get("time"),
               rows=[{k: r.get(k) for k in ("tag", "n_atoms", "dE_eh", "rms_ratio", "cosine", "sign_agree")}
                     for r in u.get("rows") or []],
               fd_consistency=u.get("fd_consistency"), rule="TS force RMS ratio in [1/3, 3] and cosine > 0.8 "
                                                           "(eV/A <-> Eh/bohr error = ratio 51.4 or 1/51.4)",
               ratio_range=[min((r["rms_ratio"] for r in ts), default=None), max((r["rms_ratio"] for r in ts), default=None)])
    return section(u.get("pass"), det)


# ================================================================ C-2 (FT leakage)
SPLIT_RE = re.compile(r"(fold|lodo)([0-4])(_al)?")


def c2_splits(o):
    """[(split, run, card fingerprint, card path)] to check."""
    out = {}
    if o.c2_splits:
        for s in o.c2_splits.split(","):
            if not SPLIT_RE.fullmatch(s):
                raise SystemExit(f"--c2-splits: {s} is not fold<k>|lodo<d>[_al]")
            out[(s, None)] = (s, None, None, None)
    else:                                       # one check per (split, data fingerprint); the first card names the run
        for p in sorted(MODELS_DIR.glob("*.json")):
            c = rj(p) or {}
            s = str(c.get("split") or "")
            if SPLIT_RE.fullmatch(s):
                out.setdefault((s, c.get("data_fingerprint")), (s, c.get("run") or p.stem, c.get("data_fingerprint"),
                                                                str(p)))
    return list(out.values())


def formula_key(syms):
    return " ".join(f"{e}{n}" for e, n in sorted(Counter(syms).items()))


def dist_spectrum(X):
    import numpy as np
    n = len(X)
    i, j = np.triu_indices(n, 1)
    return np.sort(np.linalg.norm(X[i] - X[j], axis=1))


def kabsch_rmsd(P, Q):
    import numpy as np
    P = P - P.mean(0)
    Q = Q - Q.mean(0)
    U, _, Vt = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return float(np.sqrt(((P @ R.T - Q) ** 2).sum(1).mean()))


def _c2_one(args):
    """one split (process pool worker): the recomputation (b) and the geometric check (c)."""
    split, run, fp_card, card, res_dir, data_dir = args
    import numpy as np
    m = SPLIT_RE.fullmatch(split)
    mode, val = m.group(1), int(m.group(2))
    T = M.targets()
    rfold = dict(zip(T.rxn_id.astype(int), T.fold_T1.astype(int)))
    rdip = dict(zip(T.rxn_id.astype(int), T.dipole.astype(str)))
    if mode == "fold":
        held = {r for r, f in rfold.items() if f == val}
        tag_key, tag_val = "folds", str(val)
    else:
        dn = M.DIPOLES[val]
        held = {r for r, d in rdip.items() if d == dn}
        tag_key, tag_val = "dipoles", dn
    sd = Path(data_dir) if data_dir else FT_DATA / split          # data_dir: --c2-data (negative-control test only)
    meta = rj(sd / "meta.json")
    rec = dict(split=split, run=run, card=card, mode=mode, held_out=val if mode == "fold" else M.DIPOLES[val],
               n_held_rxns=len(held), split_dir=str(sd))
    if meta is None:
        rec.update(error=f"{sd}/meta.json missing", pass_=False)
        return rec
    rec["meta_fingerprint"] = meta.get("fingerprint")
    rec["card_fingerprint"] = fp_card
    rec["fingerprint_equal"] = None if fp_card is None else bool(fp_card == meta.get("fingerprint"))
    rec["meta_leakage_C2_pass"] = (meta.get("leakage_C2") or {}).get("pass_")
    dj = [rj(p) for p in sorted((Path(res_dir) / "ft").glob("*_data.json"))]
    dj = [d for d in dj if d and d.get("split") == split and (run is None or d.get("run") == run)]
    rec["data_json"] = [dict(run=d.get("run"), fingerprint=d.get("fingerprint"),
                             leakage_C2_pass=(d.get("leakage_C2") or {}).get("pass_")) for d in dj]
    # held-out label TSs by element sequence (label order: Kabsch) and by formula (any order: distance spectrum)
    hts, hspec = {}, {}
    for r in sorted(held):
        p = M.label_ts(r)
        if p.is_file():
            s, x = M.read_xyz(p)
            x = np.asarray(x, float)
            hts.setdefault(" ".join(s), []).append((r, x))
            hspec.setdefault(formula_key(s), []).append((r, dist_spectrum(x)))
    rec["n_held_label_ts"] = sum(len(v) for v in hts.values())
    rec["geom_rule"] = (f"same element sequence and Kabsch RMSD <= {C2_RMSD_A} A, or same formula and RMS of the sorted "
                        f"interatomic distances <= {C2_SPEC_A} A (atom-order free)")
    n, used, bad_rxn, bad_tag, geo, files = 0, set(), [], [], [], {}
    for nm in ("train", "valid"):
        f = sd / f"{nm}.xyz"
        if not f.is_file():
            rec.update(error=f"{f} missing", pass_=False)
            return rec
        files[nm] = dict(path=str(f), sha256=M.sha256(f))
        for a in M.read_extxyz(f):
            n += 1
            rx = {int(x) for x in str(a.info.get("rxn_ids") or "").split(",") if x}
            used |= rx
            sid = str(a.info.get("struct_id"))
            if rx & held:
                bad_rxn.append(dict(file=nm, struct_id=sid, rxn=sorted(rx & held)[:5]))
            tg = {x for x in str(a.info.get(tag_key) or "").split(",") if x}
            if tag_val in tg:
                bad_tag.append(dict(file=nm, struct_id=sid, tag=sorted(tg)))
            syms = a.get_chemical_symbols()
            key, fk = " ".join(syms), formula_key(syms)
            hit = {}
            if key in hts:
                P = np.asarray(a.positions, float)
                for r, X in hts[key]:
                    d = kabsch_rmsd(P, X)
                    if d <= C2_RMSD_A:
                        hit[int(r)] = dict(rmsd_A=round(d, 4))
            if fk in hspec:
                ds = dist_spectrum(np.asarray(a.positions, float))
                for r, dh in hspec[fk]:
                    v = float(np.sqrt(((ds - dh) ** 2).mean()))
                    if v <= C2_SPEC_A:
                        hit.setdefault(int(r), {})["spectrum_rms_A"] = round(v, 4)
            for r, h in hit.items():
                geo.append(dict(file=nm, struct_id=sid, kind=str(a.info.get("kind")), held_rxn=r, rxn_ids=sorted(rx)[:5],
                                **h))
    rec.update(files=files, n_structures=n, n_used_rxns=len(used), n_rxn_hits=len(bad_rxn), rxn_hits=bad_rxn[:20],
               n_tag_hits=len(bad_tag), tag_hits=bad_tag[:20], n_geom_hits=len(geo),
               n_geom_hits_untagged=sum(1 for g in geo if g["held_rxn"] not in g["rxn_ids"]),
               geom_hits=sorted(geo, key=lambda g: g["held_rxn"] in g["rxn_ids"])[:20],
               n_unknown_rxn_ids=len(used - set(rfold)))
    rec["pass_"] = bool(not bad_rxn and not bad_tag and not geo and rec["meta_leakage_C2_pass"] is True
                        and all(d["leakage_C2_pass"] is True for d in rec["data_json"])
                        and rec["fingerprint_equal"] is not False and rec["n_unknown_rxn_ids"] == 0)
    return rec


def run_c2(o, ctx):
    sp = c2_splits(o)
    if not sp:
        return not_ready_sec("no model card with a fold<k> / lodo<d> split in $MLT/models (use --c2-splits for a "
                             "stand-in)")
    from concurrent.futures import ProcessPoolExecutor
    w = max(1, min(len(sp), 4, IM.cpus()))
    t = time.time()
    with ProcessPoolExecutor(w) as ex:
        recs = list(ex.map(_c2_one, [tuple(s) + (str(src_res()), o.c2_data) for s in sp]))
    folds_cov = sorted({int(SPLIT_RE.fullmatch(r["split"]).group(2)) for r in recs if r["mode"] == "fold"})
    k_report = rj(C2K_OUT)
    det = dict(rule="FT_k: no rxn_id with fold_T1 = k, no folds tag k; FT_LODO_d: no rxn_id of dipole d, no dipoles "
                    "tag d; no training structure = held-out label TS (same element sequence, Kabsch RMSD <= "
                    f"{C2_RMSD_A} A); finetune leakage_C2.pass_ true; card fingerprint = split fingerprint",
               splits=recs, n_splits=len(recs), folds_checked=folds_cov, wall_s=round(time.time() - t, 1),
               k_units=None if k_report is None else {k: k_report.get(k) for k in ("complete", "pass", "n_units",
                                                                                  "n_overlap", "written")},
               note="the K channel-unit half of C-2 is `tests.py c2_k` (C2-K stage, after K)")
    return section(all(r.get("pass_") for r in recs), det)


# ================================================================ C-3 (autodE-MLTS input words)
C3_BANG = {
    "functional": re.compile(
        r"(?:[a-z0-9-]*b3lyp[a-z0-9()/+-]*|pbe0?|pbeh-3c|revpbe0?|rpbe|bp86|blyp|olyp|b97(?:-3c|-d3?|m-v|m-d3bj)?|tpssh?|"
        r"tpss0|r2scan(?:-3c|0|h|50)?|scan|m06(?:-2x|-l|-hf)?|m062x|m06l|mn15|w?b97x(?:-d3(?:bj)?|-d4|-v)?|wb97m-v|"
        r"wb97m-d3bj|ωb97.*|cam-b3lyp|lc-.*|pw6b95|b2plyp.*|dsd-.*|pwpb95|bhandhlyp|x3lyp|o3lyp|hf-3c|hf|rhf|uhf|rohf|"
        r"ks|rks|uks|mp2|ri-mp2|scs-mp2|dlpno-.*|ccsd.*|cisd|casscf|nevpt2|am1|pm3|pm6|mndo|zindo.*|dft)", re.I),
    "xtb": re.compile(r"(?:g?xtb[0-9]?|gfn[0-9]?(?:-xtb)?|gfn-ff|native-.*xtb.*)", re.I),
    "basis": re.compile(
        r"(?:(?:ma-)?def2-[a-z0-9+()*]+|def2/jk?c?|def2/c|(?:aug-|ma-)?cc-p[cw]?v[dtq56]z.*|6-31\+*g.*|6-311\+*g.*|"
        r"3-21g.*|sto-3g|svp|tzvpp?|qzvpp?|minix|minis|pcseg-[0-4]|aug-pcseg-[0-4]|def-svp|def-tzvp|def-tzvpp|sv|"
        r"sv\(p\)|autoaux|decontract.*|gcp\(.*\))", re.I),
    "solvation": re.compile(r"(?:c?pcm(?:\(.*\))?|smd(?:\(.*\))?|alpb(?:\(.*\))?|ddcosmo(?:\(.*\))?|cosmo(?:\(.*\))?|"
                            r"gbsa(?:\(.*\))?|cpcmc(?:\(.*\))?|cosmors.*)", re.I),
    "dispersion_ri": re.compile(r"(?:d3(?:bj|zero|0|m?bj|tz)?|d4|d2|nl|vv10|sc-nl|rijcosx|rijk|ri-jk|rij|ri|cosx|"
                                r"rijonx|nori|nocosx|gcp.*)", re.I),
}
C3_BLOCK = re.compile(r"(%cpcm|%basis|%method\s+functional|smdsolvent|\bcpcm\b|\bsmd\b|\balpb\b|\bcosmo\b|\bsolvent\b|"
                      r"\bb3lyp\b|\bdef2[-/]|\bxtb\d?\b|\bgfn\d?\b)", re.I)
QUOTED_LINE = re.compile(r"^\s*(progext|ext_params)\b", re.I)


QUOTED = re.compile(r"\"([^\"]*)\"|'([^']*)'")


def inp_check(text):
    """C-3 verdict of one ORCA input text: ok = 'ExtOpt' on a '!' line and no forbidden word (decision C-3-sample)."""
    hits, quoted, extopt, progext = [], [], False, False
    for ln, raw in enumerate(text.splitlines(), 1):
        s = raw.split("#", 1)[0].strip()
        if not s:
            continue
        if s.startswith("!"):
            for tok in s[1:].split():
                t = tok.strip("'\"")
                if t.lower() == "extopt":
                    extopt = True
                    continue
                for cat, rx in C3_BANG.items():
                    if rx.fullmatch(t):
                        hits.append(dict(line=ln, cat=cat, word=t, text=s[:160]))
                        break
            continue
        if QUOTED_LINE.match(s):                                    # program path / arguments: listed, not judged
            progext |= s.lower().startswith("progext")
            for q in QUOTED.findall(s):
                for mt in C3_BLOCK.finditer(q[0] or q[1]):
                    quoted.append(dict(line=ln, word=mt.group(1), text=s[:160]))
            s = QUOTED.sub(" ", s)
        for mt in C3_BLOCK.finditer(s):
            hits.append(dict(line=ln, cat="block", word=mt.group(1), text=s[:160]))
    return dict(ok=bool(extopt and not hits), extopt=extopt, progext=progext, hits=hits, quoted_hits=quoted)


def inp_kind(name):
    b = name.rsplit("/", 1)[-1].lower()
    for k in ("optts", "opt", "hess", "freq", "grad", "sp"):
        if f"_{k}_" in b or b.endswith(f"_{k}.inp"):
            return k
    return "other"


ORCA_INPUT = re.compile(r"(?m)^\s*(!|\*\s*xyz)", re.I)


def _tar_inps(tp, skipped=None):
    """the ORCA inputs (.inp with a '!' line or a '* xyz' block) of a tarball; other .inp names go to `skipped`."""
    out = []
    with tarfile.open(tp, "r:*") as tf:
        for mem in tf.getmembers():
            if mem.isfile() and mem.name.endswith(".inp"):
                f = tf.extractfile(mem)
                t = f.read().decode("utf-8", errors="replace") if f else ""
                if ORCA_INPUT.search(t):
                    out.append((mem.name, t))
                elif skipped is not None:
                    skipped.append(f"{Path(tp).parent.name}:{mem.name}")
    return out


def input_tars(job_dir):
    """every tarball of an A-2 job folder (the layout has one: the autodE inputs tar; G2 / g2_autode: inputs.tar.gz)."""
    return sorted(p for p in Path(job_dir).iterdir() if p.is_file()
                  and re.search(r"\.tar(\.gz|\.bz2|\.xz)?$|\.tgz$", p.name))


def c3_default_root():
    """the autodE-MLTS (engine A / Ap) job folders: $MLT/ade/jobs, or $MLT/ade_pre/jobs when A'' is the main engine
    ($MLT/pipeline/arms.json engine App: mlts_autode keeps A in ade_pre and writes the DFT-refined folders to ade)."""
    arms = rj(M.MLT / "pipeline" / "arms.json") or {}
    if str(arms.get("engine")) == "App":
        return M.MLT / "ade_pre" / "jobs", "arms.json engine App -> the A run in $MLT/ade_pre"
    return M.MLT / "ade" / "jobs", f"main engine {arms.get('engine', 'unknown (no arms.json)')}"


def run_c3(o, ctx):
    import numpy as np
    root, why_root = (Path(o.c3_root), "--c3-root") if o.c3_root else c3_default_root()
    jobs = sorted(p for p in root.glob("T*") if p.is_dir()) if root.is_dir() else []
    jobs = [j for j in jobs if input_tars(j)]
    if not jobs:
        return not_ready_sec(f"no job folder with an inputs tar under {root} (A-2 / A-4 not run)")
    rng = np.random.default_rng(M.SEED + 3)
    order = rng.permutation(len(jobs))
    sample, ext, kinds, nfail_ext, used, empty, skipped = [], [], Counter(), 0, [], [], []
    for i in order:
        if len(sample) >= o.c3_n:
            break
        j = jobs[int(i)]
        inps = []
        for tp in input_tars(j):
            inps += [(f"{tp.name}:{n}", t) for n, t in _tar_inps(tp, skipped)]
        if not inps:
            empty.append(j.name)
            continue
        used.append(j.name)
        pick = inps[int(rng.integers(len(inps)))]
        r = inp_check(pick[1])
        sample.append(dict(job=j.name, member=pick[0], **{k: r[k] for k in ("ok", "extopt", "progext")},
                           hits=r["hits"][:6], quoted_hits=r["quoted_hits"][:3]))
        for nm, t in inps:
            rr = inp_check(t)
            kinds[inp_kind(nm)] += 1
            if not rr["ok"]:
                nfail_ext += 1
                if len(ext) < 30:
                    ext.append(dict(job=j.name, member=nm, extopt=rr["extopt"], hits=rr["hits"][:4]))
    n_ok = sum(s["ok"] for s in sample)
    det = dict(root=str(root), root_rule=why_root, n_job_folders=len(jobs), n_sampled=len(sample), n_sample_ok=n_ok,
               target_n=o.c3_n,
               sample_failures=[s for s in sample if not s["ok"]][:20], sample=sample[:o.c3_n],
               extended=dict(n_files=int(sum(kinds.values())), n_fail=nfail_ext, failures=ext,
                             by_kind=dict(kinds.most_common(20))),
               jobs_without_inp=empty[:20], n_non_orca_inp_skipped=len(skipped), non_orca_inp=skipped[:20],
               words=sorted(C3_BANG) + ["block"],
               rule="sample of 50 (one random .inp of 50 random job tars) and every .inp of those tars: ExtOpt on a "
                    "'!' line, no functional / basis / solvation / dispersion-RI / xTB word")
    ok = len(sample) >= o.c3_n and n_ok == len(sample) and nfail_ext == 0
    if len(sample) < o.c3_n:
        det["why"] = f"only {len(sample)} job tars with .inp members (< {o.c3_n})"
    return section(ok, det)


# ================================================================ C-5 / C-6
def run_c5(o, ctx):
    p = src_res() / "H1_report.json"
    J = rj(p)
    if J is None:
        return not_ready_sec(f"{p} missing (inputs_mlts.py h1)")
    return section(J.get("pass"), dict(path=str(p), sha256=M.sha256(p), n=J.get("n"), gates=J.get("gates"),
                                       tol=J.get("tol"), written=J.get("written"), rule=J.get("gate_rule")))


def run_c6(o, ctx):
    p = src_res() / "C6_report.json"
    J = rj(p)
    if J is None:
        return not_ready_sec(f"{p} missing (predict_c1.py c6)")
    return section(J.get("passed"), dict(path=str(p), sha256=M.sha256(p), max_abs_yhat_kcal=J.get("max_abs_yhat"),
                                         tol_kcal=J.get("tol_kcal"), n_units=J.get("n_units"),
                                         worst_unit=J.get("worst_unit"), gpu=J.get("gpu"), written=J.get("written")))


# ================================================================ C-7 (input sensitivity)
def c7_root(o):
    """run folder <base>/<src>_n<pairs>_d<delta>[_a<max atoms>]: other parameters never mix with an earlier run."""
    base = Path(o.c7_root) if o.c7_root else M.MLT / "c7"
    tag = f"{o.c7_src}_n{int(o.c7_pairs)}_d{float(o.c7_delta):g}" + (f"_a{o.c7_max_atoms}" if o.c7_max_atoms else "")
    return base / tag


def c7_source(o):
    """(A-2 source folder of a job, eligibility set of rxn_ids, description)."""
    import pandas as pd
    if o.c7_src == "g2":
        G = pd.read_csv(IM.G2_STATUS)
        ok = set(G.rxn_id[G.status == "ok"].astype(int))
        return (lambda job: M.G2_JOBS / job), ok, "stand-in: G2 xTB TSs ($G2/jobs, A-2 layout, autodE order)"
    tab = M.MLT / "inputs" / "mlts.parquet"
    if not tab.is_file():
        not_ready(f"{tab} missing (Phase H not built)")
    T = pd.read_parquet(tab, columns=["rxn_id", "usable"])
    ok = set(T.rxn_id[T.usable.astype(bool)].astype(int))
    return (lambda job: M.MLT / "ade" / "jobs" / job), ok, f"production A-2 folders, usable in {tab}"


def c7_plan(o):
    """seeded pair sample -> <root>/plan.json (reused when the parameters are the same)."""
    import numpy as np
    import pandas as pd
    R = c7_root(o)
    pp = R / "plan.json"
    params = dict(src=o.c7_src, n_pairs=int(o.c7_pairs), delta_A=float(o.c7_delta), mode="bond", seed=M.SEED + 7,
                  max_atoms=o.c7_max_atoms, code=CODE_VERSION)
    old = rj(pp)
    if old and old.get("params") == params:
        return old
    src, ok, desc = c7_source(o)
    S = pd.read_csv(M.S_FOLDS, usecols=["pair_id", "ref_kind", "a_job", "a_rxn", "b_job", "b_rxn", "dipole", "fam",
                                        "pos", "in_P_main", "fold_T1"])
    S = S[(S.in_P_main.astype(bool)) & (S.pos == "p") & (S.fold_T1 >= 0)]
    S = S[S.a_rxn.isin(ok) & S.b_rxn.isin(ok)]
    S = S[[(src(a) / ".done_ts").is_file() and (src(b) / ".done_ts").is_file() for a, b in zip(S.a_job, S.b_job)]]
    if o.c7_max_atoms:
        T = M.targets().set_index("rxn_id")
        S = S[(S.a_rxn.map(T.n_atoms) <= o.c7_max_atoms) & (S.b_rxn.map(T.n_atoms) <= o.c7_max_atoms)]
    rng = np.random.default_rng(params["seed"])
    n = int(o.c7_pairs)
    dips = list(M.DIPOLES)
    per = {d: (n // len(dips)) + (1 if k < n % len(dips) else 0) for k, d in
           enumerate([dips[int(i)] for i in rng.permutation(len(dips))])}
    pick, A, B = [], set(), set()
    for d in M.DIPOLES:
        g = S[S.dipole == d].sort_values("pair_id").reset_index(drop=True)
        for i in rng.permutation(len(g)):
            if sum(1 for x in pick if x["dipole"] == d) >= per[d]:
                break
            r = g.iloc[int(i)]
            a, b = int(r.a_rxn), int(r.b_rxn)
            if a in A or a in B or b in A:
                continue
            pick.append(dict(pair_id=str(r.pair_id), a_rxn=a, a_job=str(r.a_job), b_rxn=b, b_job=str(r.b_job),
                             dipole=d, fam=str(r.fam), fold=int(r.fold_T1), ref_kind=str(r.ref_kind)))
            A.add(a)
            B.add(b)
    if not pick:
        not_ready(f"C-7: no eligible para P_main pair ({len(S)} after the filters)")
    rx = sorted({(p["a_rxn"], p["a_job"]) for p in pick} | {(p["b_rxn"], p["b_job"]) for p in pick}, key=lambda t: t[1])
    plan = dict(params=params, source=desc, n_eligible_pairs=int(len(S)), pairs=pick, rxns=[dict(rxn_id=r, job_id=j)
                                                                                           for r, j in rx],
                perturbed=sorted(A), folds=sorted({p["fold"] for p in pick}), written=now(),
                decisions={k: DECISIONS[k] for k in ("C-7-perturbation", "C-7-side", "C-7-pairs")})
    M.write_json(pp, plan)
    return plan


def push_bonds(pos, pairs, delta):
    """each forming bond (i, j) changed by delta (A), i and j moved by delta / 2 along the bond."""
    import numpy as np
    X = np.array(pos, dtype=float)
    atoms = [a for p in pairs for a in p]
    if len(set(atoms)) != len(atoms):
        raise ValueError(f"forming bonds share an atom: {pairs}")
    d0 = [float(np.linalg.norm(X[j] - X[i])) for i, j in pairs]
    for i, j in pairs:
        e = (X[j] - X[i]) / np.linalg.norm(X[j] - X[i])
        X[i] -= 0.5 * delta * e
        X[j] += 0.5 * delta * e
    d1 = [float(np.linalg.norm(X[j] - X[i])) for i, j in pairs]
    if max(abs((b - a) - delta) for a, b in zip(d0, d1)) > 1e-9:
        raise ValueError("bond push check failed")
    return X, d0, d1


def write_xyz(path, syms, X, title):
    lines = [str(len(syms)), title] + [f"{s:<3s}{x:16.10f}{y:16.10f}{z:16.10f}" for s, (x, y, z) in zip(syms, X)]
    M.write_text(path, "\n".join(lines) + "\n")


def c7_roots(o, plan):
    """<root>/<v>/ade/jobs/<job>/ (copies of the source A-2 folders without tars; ts.xyz pushed for perturbed a's in
    p / m) + ids.csv; per-reaction push record in <root>/push.json."""
    R = c7_root(o)
    src, _, _ = c7_source(o)
    pert = set(plan["perturbed"])
    push = rj(R / "push.json") or {}
    for v in C7_VARIANTS:
        Rv = R / v
        M.write_text(Rv / "ids.csv", "rxn_id,job_id\n" + "".join(f"{x['rxn_id']},{x['job_id']}\n" for x in plan["rxns"]))
        for x in plan["rxns"]:
            rid, job = int(x["rxn_id"]), str(x["job_id"])
            dst = Rv / "ade" / "jobs" / job
            if (dst / ".c7_ok").is_file():
                continue
            s = src(job)
            IM.mkdir(dst)
            for p in sorted(s.iterdir()):
                if p.is_file() and not re.search(r"\.tar(\.gz)?$|\.tgz$", p.name):
                    shutil.copyfile(p, dst / p.name)
            if v != "v0" and rid in pert:
                syms, pos = M.read_xyz(s / "ts.xyz")
                fp = [tuple(int(a) for a in q) for q in (rj(s / "ts_result.json") or {})["formed_pairs_ts"]]
                X, d0, d1 = push_bonds(pos, fp, C7_SIGN[v] * float(o.c7_delta))
                write_xyz(dst / "ts.xyz", syms, X, f"C-7 {v} rxn {rid}: forming bonds {fp} {C7_SIGN[v] * o.c7_delta:+.3f} A")
                push[f"{v}:{rid}"] = dict(pairs=fp, d_before_A=d0, d_after_A=d1)
            M.write_text(dst / ".c7_ok", now() + "\n")
    M.write_json(R / "push.json", push)
    return push


C7_CHAIN = (("finalize", ["0", "1", "1"], (0, 2)), ("meta", [], (0,)), ("x182", ["0", "1"], (0,)),
            ("x182_merge", [], (0,)), ("atoms", ["0", "1"], (0, 2)), ("tokens", [], (0,)), ("hf_inputs", [], (0,)),
            ("hf_run", ["0", "1", "W"], (0, 2)), ("hf_build", [], (0, 2)), ("d3", [], (0,)), ("build", [], (0, 3)))


def c7_chain(o, v, steps=None):
    """the Phase H chain on variant root v (each inputs_mlts.py step re-executes itself in its env; idempotent)."""
    Rv = c7_root(o) / v
    sp = Rv / "chain.json"
    st = rj(sp) or {}
    env = {k: val for k, val in os.environ.items() if k != "MLT_IN_ENV"}
    # own node-local TMPDIR per variant: the D1-HF runner works in $TMPDIR/d1hf_<SLURM_JOB_ID>/<run>/<T>, the same path
    # for v0 / p / m (same job, same T) if two variants ever ran at once in one allocation (decision C-7-tmpdir)
    tmp = Path(os.environ.get("TMPDIR") or "/tmp") / f"mlts_c7_{v}_{os.getpid()}"
    tmp.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = str(tmp)
    try:
        return _c7_chain_steps(o, v, steps, Rv, sp, st, env)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _c7_chain_steps(o, v, steps, Rv, sp, st, env):
    for name, args, ok_rc in C7_CHAIN:
        if steps and name not in steps:
            continue
        if (st.get(name) or {}).get("ok"):
            continue
        need_time(f"C-7 chain {v}:{name}")
        a = [str(IM.cpus()) if x == "W" else x for x in args]
        argv = [sys.executable, str(IM.THIS), name, "mlts"] + a + ["--root", str(Rv), "--ids", str(Rv / "ids.csv")]
        lp = Rv / "logs" / f"{name}.log"
        IM.mkdir(lp.parent)
        tries = 2 if name in ("hf_run",) else 1
        rc, t = None, time.time()
        for _ in range(tries):
            with open(lp, "a") as fh:
                fh.write(f"=== {now()} {' '.join(argv)}\n")
                fh.flush()
                rc = subprocess.run(argv, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
            if rc == 0:
                break
        ok = rc in ok_rc
        if name == "build" and rc == 3:
            ok = (Rv / "inputs" / "mlts.parquet").is_file()        # H-2 > 2 % on a subset: table written, tolerated
        st[name] = dict(rc=rc, ok=bool(ok), wall_s=round(time.time() - t, 1), log=str(lp), time=now())
        M.write_json(sp, st)
        log(f"C-7 {v} {name}: exit {rc} ({st[name]['wall_s']} s)")
        if not ok:
            return False, st
    return all((st.get(n) or {}).get("ok") for n, _, _ in C7_CHAIN), st


def c7_pred(o, v, folds):
    """the C1 atom-unit predictions on variant root v (child process in env fu2)."""
    Rv = c7_root(o) / v
    out = Rv / "c7_pred.json"
    tab = Rv / "inputs" / "mlts.parquet"
    J = rj(out)
    if J and J.get("table_sha256") == M.sha256(tab) and J.get("folds") == list(folds) and (Rv / "c7_pred.parquet").is_file():
        return J
    need_time(f"C-7 predictions {v}")
    env = {k: val for k, val in os.environ.items() if k != "MLT_IN_ENV"}
    lp = Rv / "logs" / "c7pred.log"
    IM.mkdir(lp.parent)
    with open(lp, "a") as fh:
        rc = subprocess.run([sys.executable, str(THIS), "_c7pred", str(Rv), ",".join(str(k) for k in folds)], env=env,
                            stdout=fh, stderr=subprocess.STDOUT).returncode
    J = rj(out)
    if rc != 0 or not J:
        raise RuntimeError(f"_c7pred {v} exit {rc} (log {lp})")
    return J


def cmd_c7pred(root, folds, src="mlts", out=None):
    """child: env fu2. Root R as the 'mlts' source (inputs_mlts._OPT) -> predict_c1.register_mlts -> predict_atoms
    unit_run of every atom unit on the given folds; y dropped. src 'label' (test of the element re-expansion only):
    the H-1 label root ($MLT/h1, inputs bit-equal to the LADDER dft rung) -> every deployable pair, and the deployed
    yhat is compared with L_predictions (must be equal: same inputs, same units) -> <out>/c7_pred.json."""
    IM.ensure_env("fu2", script=THIS, argv=["_c7pred", root, folds, "--src", src] + (["--out", out] if out else []))
    import numpy as np
    R = Path(root)
    if src == "mlts":
        with open(R / "ids.csv", newline="") as fh:
            ids = sorted({int(r["rxn_id"]) for r in csv.DictReader(fh)})
        IM._OPT.update(root=R, ids=ids)
    import predict_c1 as PC
    PC._geo2_import()
    import predict_atoms as PA
    DC, TA, FT = PA.DC, PA.TA, PA.FT
    reg = PC.register_mlts(TA, "atoms", src)
    folds = [int(k) for k in str(folds).split(",") if k != ""]
    want = set(str(p["pair_id"]) for p in (rj(R.parent / "plan.json") or {}).get("pairs", [])) if src == "mlts" else None
    if src != "mlts":
        R = Path(out)
    Lp, _ = DC.load_ladder_dft(DC.ATOM_MODELS)
    deps, units, remap, layout, vs_lp = [], {}, {}, {}, []
    t0 = time.time()
    for arm in TA.ARMS:
        Dd, Dg = TA.get_data("dft", arm), TA.get_data("mlts", arm)
        ed, eg = [int(z) for z in Dd.elements], [int(z) for z in Dg.elements]
        remap[arm] = dict(dft=ed, subset=eg, remapped=ed != eg)
        if ed != eg:                                                # decision C-7-elements
            if not set(eg) <= set(ed):
                M.stop(f"C-7: subset elements {eg} not within the dft elements {ed}")
            idx = [ed.index(z) for z in eg]
            oh = np.zeros(tuple(Dg.onehot.shape[:2]) + (len(ed),), dtype=Dg.onehot.dtype)
            oh[..., idx] = Dg.onehot
            Dg.onehot, Dg.elements, Dg.d_in = oh, Dd.elements, Dd.d_in
            if Dg.d_in != len(Dg.elements) + len(Dg.bin_cols) + len(Dg.num_cols):
                M.stop("C-7: d_in after the element re-expansion differs")
        layout[arm] = PA.layout_check(Dd, Dg)
        for model in TA.MODELS:
            for k in folds:
                u = dict(rung="dft", arm=arm, model=model, fold=k)
                rec, dep, _ = PA.unit_run(Dd, Dg, u, Lp)
                if want is not None:
                    dep = dep[dep.pair_id.astype(str).isin(want)].copy()
                else:                                               # label source: deployed = L_predictions?
                    lr = DC.ladder_rows(Lp, arm, model, k)[["pair_id", "target", "yhat"]].rename(columns={"yhat": "lp"})
                    mm = dep[["pair_id", "target", "yhat"]].merge(lr, on=["pair_id", "target"], how="inner")
                    vs_lp.append(dict(unit=TA.uname(u), n=int(len(mm)), n_dep=int(len(dep)),
                                      max_abs=float((mm.yhat - mm.lp).abs().max()) if len(mm) else None))
                units[TA.uname(u)] = dict(n_deploy_all=rec["n_deploy"], n_kept=int(dep.pair_id.nunique()),
                                          regression_max_abs_yhat=rec["vs_l_predictions"]["max_abs_yhat"])
                deps.append(dep.drop(columns=["y"]))                 # decision C-7-no-labels
    import pandas as pd
    D = pd.concat(deps, ignore_index=True) if deps else pd.DataFrame()
    M.write_parquet(R / "c7_pred.parquet", D)
    tab = Path(reg["table"])
    J = dict(root=str(R), source=src, registered=reg, folds=folds, table_sha256=M.sha256(tab),
             n_rows=int(len(D)), n_pairs=int(D.pair_id.nunique()) if len(D) else 0, units=units, remap=remap,
             layout_ok={a: bool(v.get("ok")) for a, v in layout.items()}, versions=FT.versions(),
             seconds=round(time.time() - t0, 1), written=now(), note="y dropped (C-7 is not a channel result)",
             label_source_vs_l_predictions=vs_lp or None)
    M.write_json(R / "c7_pred.json", clean(J))
    print(f"[c7pred] {R.name}: {len(D)} rows, {J['n_pairs']} pairs, {J['seconds']} s", flush=True)
    return 0


def e2_table():
    """E-2 'same' table (scope common): {target: {level: noise_median}} and {level: |dd| median}."""
    noise, dd = {}, {}
    with open(E2_NOISE, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["scope"] == "common" and r["class"] == "same":
                noise.setdefault(r["channel"], {})[r["level"]] = float(r["noise_median"])
    with open(E2_GEOM, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["scope"] == "common" and r["class"] == "same":
                dd[r["level"]] = float(r["dform_abs_median"])
    return noise, dd


def rep8_choice():
    J = rj(C1_REPS) or {}
    rep8 = ((J.get("REP8") or {}).get("mlts") or (J.get("REP8") or {}).get("dft") or {})
    reps = ((J.get("REP") or {}).get("mlts") or (J.get("REP") or {}).get("dft") or {})
    out, notes = {}, {}
    for t in H_TARGETS:
        a, m = (rep8.get(t) or [None, None])[:2]
        if t == "disp" and m == "XGB":
            out[t], notes[t] = ("D3_int", "D3_int"), "REP8 S/XGB; H-0: disp predicted by D3_int"
        elif m == "XGB":
            alt = next(((arm, reps.get(arm, {}).get(t)) for arm in ("S", "SH") if reps.get(arm, {}).get(t) in
                        ("AW", "GNN", "TF")), ("S", "AW"))
            out[t], notes[t] = alt, f"REP8 {a}/XGB not run (decision C-7-models): {alt[0]}/{alt[1]}"
        elif a and m:
            out[t] = (a, m)
        else:
            out[t], notes[t] = ("SH", "AW"), "no C1_representatives.json: SH/AW"
    return out, notes, str(C1_REPS)


def c7_report(o, plan, chains, preds):
    import numpy as np
    import pandas as pd
    R = c7_root(o)
    pairs = pd.DataFrame(plan["pairs"])
    P = {v: pd.read_parquet(R / v / "c7_pred.parquet") for v in C7_VARIANTS}
    Tb = {v: pd.read_parquet(R / v / "inputs" / "mlts.parquet").set_index("rxn_id") for v in C7_VARIANTS}
    key = ["pair_id", "arm", "model", "target"]
    base = P["v0"][key + ["yhat"]].rename(columns={"yhat": "y0"})
    W = base
    for v in ("p", "m"):
        W = W.merge(P[v][key + ["yhat"]].rename(columns={"yhat": f"y_{v}"}), on=key, how="inner")
    W["d_p"], W["d_m"] = W.y_p - W.y0, W.y_m - W.y0
    complete = sorted(set(W.pair_id) & set(pairs.pair_id))
    noise, dd = e2_table()
    sc = float(o.c7_delta) / dd["g3r"] if dd.get("g3r") else None
    rep, notes, rep_src = rep8_choice()
    a_of = dict(zip(pairs.pair_id, pairs.a_rxn))

    def stats(dp, dm):
        both = np.abs(np.concatenate([dp, dm]))
        return dict(n_pairs=int(len(dp)), median_abs_plus=float(np.median(np.abs(dp))) if len(dp) else None,
                    median_abs_minus=float(np.median(np.abs(dm))) if len(dm) else None,
                    median_abs=float(np.median(both)) if len(both) else None,
                    p90_abs=float(np.quantile(both, 0.9)) if len(both) else None,
                    max_abs=float(both.max()) if len(both) else None,
                    median_abs_nonlinear=float(np.median(np.abs(dp + dm))) if len(dp) else None)

    def inputs_delta(col):
        dp, dm = [], []
        for pid in complete:
            a = int(a_of[pid])
            if all(a in Tb[v].index for v in C7_VARIANTS):
                x0 = float(Tb["v0"].at[a, col])
                dp.append(float(Tb["p"].at[a, col]) - x0)
                dm.append(float(Tb["m"].at[a, col]) - x0)
        return np.asarray(dp), np.asarray(dm)

    per, units = {}, {}
    for (arm, model, t), g in W[W.pair_id.isin(complete)].groupby(["arm", "model", "target"]):
        units[f"{arm}/{model}/{t}"] = stats(g.d_p.to_numpy(), g.d_m.to_numpy())
    for t in H_TARGETS:
        arm, model = rep[t]
        if arm == "D3_int":
            s = stats(*inputs_delta("D3_int"))
        else:
            g = W[(W.arm == arm) & (W.model == model) & (W.target == t) & W.pair_id.isin(complete)]
            s = stats(g.d_p.to_numpy(), g.d_m.to_numpy())
        hi = stats(*inputs_delta(f"H_{t}"))
        e = noise.get(t, {})
        ref = e.get("g3r")
        scaled = ref * sc if (ref is not None and sc is not None) else None
        r_raw = s["median_abs"] / ref if (s.get("median_abs") is not None and ref) else None
        r_sc = s["median_abs"] / scaled if (s.get("median_abs") is not None and scaled) else None
        cat = None if r_sc is None else ("smaller" if r_sc < 1 / 3 else ("same order" if r_sc <= 3 else "larger"))
        per[t] = dict(model=f"{arm}/{model}" if arm != "D3_int" else "D3_int", note=notes.get(t), prediction=s,
                      hf_input=hi, E2_noise_median=e, E2_g3r_scaled_to_delta=scaled, ratio_to_E2_g3r=r_raw,
                      ratio_to_E2_g3r_scaled=r_sc, category=cat,
                      hf_input_ratio_to_E2_g3r_scaled=(hi["median_abs"] / scaled if hi.get("median_abs") is not None
                                                       and scaled else None))
    prod = None
    if o.c7_src == "prod":                                          # v0 = production inputs? (reproduction info)
        tab = M.MLT / "inputs" / "mlts.parquet"
        Tp = pd.read_parquet(tab).set_index("rxn_id")
        rid = [r for r in Tb["v0"].index if r in Tp.index]
        num = [c for c in Tb["v0"].columns if c in Tp.columns and pd.api.types.is_numeric_dtype(Tb["v0"][c])
               and pd.api.types.is_numeric_dtype(Tp[c])]
        hcols = [c for c in num if c.startswith(("H_", "H0_")) or c == "D3_int"]
        d = (Tb["v0"].loc[rid, num].astype(float) - Tp.loc[rid, num].astype(float)).abs() if rid else None
        prod = dict(table=str(tab), n_rxn=len(rid),
                    max_abs_numeric=float(np.nanmax(d.to_numpy())) if d is not None and d.size else None,
                    max_abs_H_D3=float(np.nanmax(d[hcols].to_numpy())) if d is not None and hcols else None,
                    worst={k: float(v) for k, v in d.max().sort_values(ascending=False).head(5).items()} if d is not None
                    else {})
    gate = len(complete) >= max(1, math.ceil(0.8 * len(pairs)))
    head = [t for t in ("oi", "cpcm", "pauli", "elst", "d1", "barrier") if per[t]["prediction"].get("median_abs") is not None]
    summ = (f"+-{o.c7_delta} A on {len(complete)}/{len(pairs)} para pairs: median |dyhat| "
            + ", ".join(f"{t} {per[t]['prediction']['median_abs']:.3f}" for t in head)
            + (f"; E-2 g3r x{sc:.2f} ({dd['g3r']:.3f} A -> {o.c7_delta} A): "
               + ", ".join(f"{t} {per[t]['E2_g3r_scaled_to_delta']:.3f}" for t in head if per[t]['E2_g3r_scaled_to_delta'] is not None)
               if sc else "")
            + "; category " + ", ".join(f"{t} {per[t]['category']}" for t in head))
    det = dict(source=plan["source"], params=plan["params"], n_pairs=len(pairs), n_pairs_complete=len(complete),
               pairs=plan["pairs"], perturbed=plan["perturbed"], delta_A=o.c7_delta, per_target=per, all_units=units,
               representatives=rep_src, E2=dict(source=[str(E2_NOISE), str(E2_GEOM)], scope="common", cls="same",
                                                dform_abs_median_A=dd, scale_to_delta=sc),
               chains={v: {k: dict(rc=x.get("rc"), ok=x.get("ok"), wall_s=x.get("wall_s")) for k, x in c.items()}
                       for v, c in chains.items()},
               predictions={v: {k: preds[v].get(k) for k in ("n_rows", "n_pairs", "folds", "seconds", "layout_ok",
                                                             "remap", "table_sha256")} for v in preds},
               regression_max_abs_yhat=max((u["regression_max_abs_yhat"] for p in preds.values()
                                            for u in (p.get("units") or {}).values()), default=None),
               v0_vs_production=prod, push=str(R / "push.json"),
               decisions={k: DECISIONS[k] for k in DECISIONS if k.startswith("C-7")})
    return section(gate, det, judged=False, summary=summ)


def run_c7(o, ctx):
    plan = c7_plan(o)
    if o.c7_stage == "plan":
        return not_ready_sec("C-7 stage 'plan' only", plan=str(c7_root(o) / "plan.json"), n_pairs=len(plan["pairs"]))
    c7_roots(o, plan)
    variants = [o.c7_variant] if o.c7_variant else list(C7_VARIANTS)
    steps = set(o.c7_steps.split(",")) if o.c7_steps else None
    chains, failed = {}, []
    if o.c7_stage in ("all", "chain"):
        for v in variants:
            ok, st = c7_chain(o, v, steps)
            chains[v] = st
            if not ok:
                failed.append(v)
    for v in C7_VARIANTS:
        chains.setdefault(v, rj(c7_root(o) / v / "chain.json") or {})
    if failed and o.c7_stage == "all":
        # C-7-terminal: report only, so a chain that fails in two separate runs ends the section (ready, pass false,
        # not judged) instead of keeping C-agg / R / K waiting at exit 2
        ap = c7_root(o) / "chain_failures.json"
        A = rj(ap) or {}
        for v in failed:
            A[v] = int(A.get(v) or 0) + 1
        M.write_json(ap, A)
        if any(int(A.get(v) or 0) >= C7_MAX_FAILS for v in failed):
            return section(False, dict(why=f"C-7 chain failed in {C7_MAX_FAILS} runs for {failed} (report only: not "
                                           f"produced)", failures=A,
                                       chains={v: {k: x.get("rc") for k, x in c.items()} for v, c in chains.items()}),
                           judged=False, summary=f"C-7 not produced: chain failures {A}")
    if o.c7_stage == "chain" or failed:
        return not_ready_sec(f"C-7 chain {'failed for ' + str(failed) if failed else 'stage only'}",
                             chains={v: {k: x.get("rc") for k, x in c.items()} for v, c in chains.items()})
    need = [o.c7_variant] if (o.c7_stage == "pred" and o.c7_variant) else list(C7_VARIANTS)
    if not all(all((chains[v].get(n) or {}).get("ok") for n, _, _ in C7_CHAIN) for v in need):
        return not_ready_sec("C-7 chains incomplete", chains={v: {k: x.get("rc") for k, x in c.items()}
                                                              for v, c in chains.items()})
    preds = {}
    if o.c7_stage == "pred" and o.c7_variant:                       # one variant per call (test granularity)
        J = c7_pred(o, o.c7_variant, plan["folds"])
        return not_ready_sec(f"C-7 predictions of {o.c7_variant} only", n_rows=J.get("n_rows"), seconds=J.get("seconds"))
    for v in C7_VARIANTS:
        if o.c7_stage == "report":
            preds[v] = rj(c7_root(o) / v / "c7_pred.json") or {}
        else:
            preds[v] = c7_pred(o, v, plan["folds"])
    return c7_report(o, plan, chains, preds)


# ================================================================ C_tests.json
RUNNERS = dict(C1=run_c1, C2=run_c2, C3=run_c3, C4=run_c4, C5=run_c5, C6=run_c6, C7=run_c7)


def merge_out(res, o):
    J = rj(OUT) or {}
    J.update(spec="docs/specs/DEPLOY_V2_MLTS.md 8", code=CODE_VERSION)
    for k, v in res.items():
        J[k] = clean(v)
    secs = {k: J.get(k) or {} for k in SECTIONS}
    ready = {k: bool(s.get("ready")) for k, s in secs.items()}
    gated_fail = [k for k in GATED if ready[k] and secs[k].get("pass") is False]
    J["complete"] = all(ready.values())
    J["n_todo"] = sum(not r for r in ready.values())
    J["all_pass"] = (bool(all(secs[k].get("pass") is True for k in SECTIONS)) if J["complete"] else
                     (False if gated_fail else None))
    J["stop"] = (f"Phase C check failed: {gated_fail} (spec 14 / PIPELINE HALT)" if gated_fail else None)
    J["est"] = dict(units_total=1, unit_wall_h=2.5, workers=1,
                    note="C-7 dominates: 3 Phase H chains on ~30 reactions (serial HF EDA 5-10 min at 25-30 atoms, "
                         "longer for 50) + 3 x 6 atom units (~35 s each) ~2-2.5 h on 16 cores; C-2 ~5 min; C-1 rerun "
                         "~0.3 GPU-h only when C1_tests.json is stale")
    st = dict(J.get("stand_in") or {})
    if "C1" in res:
        d = res["C1"].get("detail") or {}
        st["C1_C4_model"] = d.get("model") or (d.get("why"))
    if "C2" in res:
        st.pop("C2_splits", None)
        if o.c2_splits:
            st["C2_splits"] = f"--c2-splits {o.c2_splits} (no fine-tuned model card)" + (
                f", data {o.c2_data}" if o.c2_data else "")
    if "C3" in res:
        st.pop("C3_root", None)
        if o.c3_root:
            st["C3_root"] = f"--c3-root {o.c3_root}"
    if "C7" in res:
        st["C7_source"] = "G2 xTB TSs as stand-in MLTS TSs" if o.c7_src == "g2" else "production MLTS TSs"
    J["stand_in"] = st
    J["C2"] = J.get("C2") or {}
    J["C7"] = J.get("C7") or {}
    J["decisions"] = DECISIONS
    J.update(written=now(), host=os.uname().nodename, job=jobtag())
    M.write_json(OUT, clean(J))
    return J


def cmd_sections(o, which):
    ctx, res = {}, {}
    for k in which:
        t = time.time()
        try:
            r = RUNNERS[k](o, ctx)
        except NotReady as e:
            r = not_ready_sec(str(e))
        except SystemExit as e:
            if isinstance(e.code, int) and e.code >= 128:            # terminated (signal): stop here
                raise
            if e.code == 2:
                r = not_ready_sec(f"not ready (exit 2) in {k}")
            elif e.code == 3:                                       # a STOP inside the check = that check failed
                r = section(False, dict(stop=f"STOP (exit 3) in {k}: see the job log"))
            else:
                r = not_ready_sec(f"exit {e.code} in {k}", error=True)
        except Exception as e:                                      # noqa: BLE001  (a crash is not a test result)
            r = not_ready_sec(f"{type(e).__name__}: {e}", error=True, trace=traceback.format_exc()[-2500:])
        r["wall_s"] = round(time.time() - t, 1)
        r["time"] = now()
        res[k] = r
        J = merge_out({k: r}, o)
        log(f"{k}: ready {r.get('ready')} pass {r.get('pass')} ({r['wall_s']} s)"
            + (f" — {r.get('summary')}" if r.get("summary") else ""))
    J = rj(OUT) or {}
    if J.get("stop"):
        print(f"STOP: {J['stop']}", file=sys.stderr)
        return 3
    if any((res[k].get("detail") or {}).get("error") for k in which):
        print(f"ERROR in {[k for k in which if (res[k].get('detail') or {}).get('error')]} (exit 1: retry once)",
              file=sys.stderr)
        return 1
    return 0 if all(res[k].get("ready") for k in which) else 2


# ================================================================ C2-K (K channel units)
TRAIN_KEYS = ("train_pair_ids", "train_pairs", "pairs_train", "fit_pair_ids")


def _pair_ids(v):
    out = []
    for x in v or []:
        if isinstance(x, dict):
            x = x.get("pair_id")
        elif isinstance(x, (list, tuple)):
            x = x[0] if x else None
        if x is not None:
            out.append(str(x))
    return out


def k_unit_files(unit, extra):
    dirs = [Path(d) for d in extra] + [M.MLT / "units", M.MLT / "k" / "units", M.MLT / "k", M.MLT / "k_units"]
    for d in dirs:
        for p in (d / f"{unit}.json", d / unit / f"{unit}.json"):
            if p.is_file():
                return p
    for d in dirs:
        if d.is_dir():
            hit = sorted(d.glob(f"**/{unit}.json"))
            if hit:
                return hit[0]
    return None


def k_train_pairs(unit, rec, jp, fold, splits_sha):
    """(pair ids, source label) of a K unit's training pairs (decision C2-K-source). The plan unit name carries its
    config folder ('C2/L__mlts__S__TF__f0'); the sidecar sits next to the unit json under its bare name. A unit with a
    train_k K record (<name>__k.json) must have its sidecar: the FU2 fallback would not see a C3 label copy."""
    name = Path(str(unit)).name
    for k in TRAIN_KEYS:
        if rec.get(k):
            return _pair_ids(rec[k]), f"unit json '{k}'"
    for ext in ("parquet", "csv", "json"):
        for nm in (f"{name}__train_pairs.{ext}", f"{name}.train_pairs.{ext}"):
            p = jp.parent / nm
            if p.is_file():
                if ext == "parquet":
                    import pandas as pd
                    return [str(x) for x in pd.read_parquet(p).pair_id], f"sidecar {p.name}"
                if ext == "csv":
                    with open(p, newline="") as fh:
                        return [str(r["pair_id"]) for r in csv.DictReader(fh)], f"sidecar {p.name}"
                return _pair_ids(M.read_json(p)), f"sidecar {p.name}"
    if (jp.parent / f"{name}__k.json").is_file():
        return None, f"K record {name}__k.json without its __train_pairs sidecar (C2-K-source)"
    sp = Path(M.read_json(FU2_S_SPLITS)["pairs"]).parent / f"T1__f{int(fold)}.json"
    fp = rec.get("fingerprint") or {}
    ver = "verified" if fp.get("s_splits") == splits_sha else ("assumed" if fp.get("s_splits") is None else "MISMATCH")
    return [str(x) for x in M.read_json(sp)["train"]], f"FU2 T1 split train list {sp.name} ({ver})"


def cmd_c2k(o):
    import pandas as pd
    kp, kr = src_res() / "K_plan.json", src_res() / "K_report.json"
    plan, krep = rj(kp), rj(kr)
    if not plan or not plan.get("complete"):
        M.not_ready("results/K_plan.json missing or incomplete (train_k.py plan)")
    if not krep or not krep.get("complete"):
        M.not_ready("results/K_report.json missing or incomplete (train_k.py select)")
    S = pd.read_csv(M.S_FOLDS, usecols=["pair_id", "a_rxn", "b_rxn", "fold_T1"])
    ab = {str(p): (int(a), int(b)) for p, a, b in zip(S.pair_id, S.a_rxn, S.b_rxn)}
    T = M.targets()
    rfold = dict(zip(T.rxn_id.astype(int), T.fold_T1.astype(int)))
    spdir = Path(M.read_json(FU2_S_SPLITS)["pairs"]).parent
    test = {k: set(str(x) for x in M.read_json(spdir / f"T1__f{k}.json")["test"]) for k in range(5)}
    ssha = M.sha256(FU2_S_SPLITS)
    units, n_over, overlaps, unresolved = [], 0, [], []
    for x in plan.get("units") or []:
        arm, unit = str(x.get("arm")), str(x.get("unit"))
        jp = k_unit_files(unit, o.units_dir or [])
        if jp is None:
            unresolved.append(unit)
            units.append(dict(arm=arm, unit=unit, error="unit json not found"))
            continue
        rec = M.read_json(jp)
        m = re.search(r"__f(\d)", unit)
        fold = (rec.get("unit") or {}).get("fold", int(m.group(1)) if m else rec.get("fold"))
        if fold is None or str(fold) in ("all", "-1"):
            units.append(dict(arm=arm, unit=unit, json=str(jp), fold=fold, note="no held-out fold (final unit): skipped"))
            continue
        fold = int(fold)
        ids, src = k_train_pairs(unit, rec, jp, fold, ssha)
        if ids is None:
            unresolved.append(unit)
            units.append(dict(arm=arm, unit=unit, json=str(jp), fold=fold, error=src))
            continue
        unknown = [p for p in ids if p not in ab]
        in_test = [p for p in ids if p in test[fold]]
        touch = [p for p in ids if p in ab and (rfold.get(ab[p][0]) == fold or rfold.get(ab[p][1]) == fold)]
        bad = sorted(set(in_test) | set(touch) | set(unknown))
        n_over += len(bad)
        if bad:
            overlaps.append(dict(unit=unit, n=len(bad), first=bad[:10]))
        units.append(dict(arm=arm, unit=unit, json=str(jp), fold=fold, n_train_pairs=len(ids), source=src,
                          n_in_test_fold=len(in_test), n_touching_fold_rxn=len(touch), n_unknown_pair_ids=len(unknown),
                          split_mismatch="MISMATCH" in src))
    preds = []
    for p in sorted((M.MLT / "preds").glob("C[23]*_predictions.parquet")):
        D = pd.read_parquet(p, columns=["pair_id", "fold"])
        out = int(sum(str(a) not in test.get(int(f), set()) for a, f in zip(D.pair_id, D.fold)))
        preds.append(dict(path=str(p), n_rows=int(len(D)), n_outside_test_fold=out))
        n_over += out
    n_units = sum(1 for u in units if "n_train_pairs" in u)
    ok = bool(n_units > 0 and n_over == 0 and not unresolved and not any(u.get("split_mismatch") for u in units))
    rep = dict(spec="DEPLOY_V2_MLTS.md 8 C-2 (K channel units)", code=CODE_VERSION, complete=True, n_todo=0, stop=None if
               ok else "C-2 (K units) failed", est=dict(units_total=1, unit_wall_h=0.05, workers=1), pass_=ok,
               n_units=n_units, n_overlap=n_over, overlaps=overlaps[:20], unresolved=unresolved, units=units,
               predictions=preds, k_plan_sha256=M.sha256(kp), k_report_sha256=M.sha256(kr), fu2_s_splits_sha256=ssha,
               rule=DECISIONS["C2-K-source"], written=now(), host=os.uname().nodename, job=jobtag())
    rep["pass"] = ok
    M.write_json(C2K_OUT, clean(rep))
    log(f"C2-K: {n_units} units, overlap {n_over}, unresolved {len(unresolved)} -> pass {ok}")
    return 0 if ok else 3


# ================================================================ main
def parser():
    ap = argparse.ArgumentParser(prog="tests.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    com = argparse.ArgumentParser(add_help=False)
    com.add_argument("--model", default=None, help="C-1 / C-4 model (default FT_0_al, else newest FT_0 card)")
    com.add_argument("--c1-json", default=None, help="C1_tests.json to reuse (default results/C1_tests.json)")
    com.add_argument("--no-rerun", action="store_true", help="C-1: never rerun tests_c1 (reuse only)")
    com.add_argument("--c1-force", action="store_true", help="C-1: rerun the 4 tests_c1 sections even when fresh")
    com.add_argument("--c2-splits", default=None, help="C-2: comma list of split names (stand-in)")
    com.add_argument("--c2-data", default=None, help="C-2: read this split folder for every checked split (negative "
                                                     "control test only)")
    com.add_argument("--c3-root", default=None, help="C-3: job folders root (default $MLT/ade/jobs)")
    com.add_argument("--c3-n", type=int, default=C3_N)
    com.add_argument("--c7-src", default="prod", choices=("prod", "g2"))
    com.add_argument("--c7-root", default=None)
    com.add_argument("--c7-pairs", type=int, default=C7_PAIRS)
    com.add_argument("--c7-delta", type=float, default=C7_DELTA_A)
    com.add_argument("--c7-max-atoms", type=int, default=None)
    com.add_argument("--c7-stage", default="all", choices=("all", "plan", "chain", "pred", "report"))
    com.add_argument("--c7-variant", default=None, choices=C7_VARIANTS)
    com.add_argument("--c7-steps", default=None, help="comma list of chain steps (test granularity)")
    com.add_argument("--src-results", default=None, help="read existing reports (H1, C6, C1_tests, ft/*_data, K_plan, "
                                                         "K_report) from DIR instead of results/ (tests)")
    sub.add_parser("all", parents=[com])
    for k in SECTIONS:
        sub.add_parser(k.lower(), parents=[com])
    p = sub.add_parser("c2_k")
    p.add_argument("--units-dir", action="append", default=None)
    p.add_argument("--src-results", default=None)
    p = sub.add_parser("c3_files")
    p.add_argument("files", nargs="+")
    p = sub.add_parser("_c7pred")
    p.add_argument("root")
    p.add_argument("folds")
    p.add_argument("--src", default="mlts", choices=("mlts", "label"))
    p.add_argument("--out", default=None, help="output folder (required with --src label)")
    return ap


def _on_term(signum, frame):                                        # SIGTERM (timeout / scancel / wall): run finally
    raise SystemExit(128 + signum)                                  # blocks (C-7 TMPDIR clean-up), then exit


def main(argv=None):
    import signal
    signal.signal(signal.SIGTERM, _on_term)
    o = parser().parse_args(argv)
    if getattr(o, "src_results", None):
        SRC["res"] = Path(o.src_results)
    if o.cmd == "all":
        return cmd_sections(o, SECTIONS)
    if o.cmd in [k.lower() for k in SECTIONS]:
        return cmd_sections(o, (o.cmd.upper(),))
    if o.cmd == "c2_k":
        return cmd_c2k(o)
    if o.cmd == "c3_files":
        out = {}
        for f in o.files:
            p = Path(f)
            if re.search(r"\.tar(\.gz)?$|\.tgz$", p.name):
                out[f] = {n: inp_check(t) for n, t in _tar_inps(p)}
            else:
                out[f] = inp_check(p.read_text(errors="replace"))
        print(json.dumps(out, indent=1))
        return 0
    if o.cmd == "_c7pred":
        if o.src != "mlts" and not o.out:
            raise SystemExit("_c7pred --src label needs --out <dir under $MLT>")
        return cmd_c7pred(o.root, o.folds, o.src, o.out)
    parser().print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
