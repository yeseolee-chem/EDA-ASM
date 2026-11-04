#!/usr/bin/env python3
r"""g4_al.py — DEPLOY-V2 (MLTS) Phase G-4: one round of active learning after the first fine-tune
(docs/specs/DEPLOY_V2_MLTS.md 3 G-4 / 5 FT; budget 0-5). Compute nodes only (sbatch / jobs/sbx.sh).

Spec 3 G-4: "on 1,000 training-side reactions optimise the TS with MLTS and run one DFT EnGrad at that structure; the
structures with max force > 0.002 Eh/bohr are added to the training data for the second fine-tune (~100 core-h)".
This file is new (gen_dft.py belongs to another workflow): the label-level EnGrad worker of gen_dft (run_struct,
engrad, Ctx, run_items, slice_units, the level checks and $MLT/dft/STOP) and the ts_geom TS evaluator
(evaluate_one with tight = TightOpt and, for D0 rows, explicit label / start files; table, summarize) are imported and
called (decisions G4-tight, G4-d0).

CLI (from analysis/mlts; exit 0 = done / up to date, 2 = incomplete or not ready (same command again), 3 = STOP)
  g4_plan --models m0,m1,m2,m3,m4 [--n-per-model 200] [--n-d0-per-model 100] [--start g2|label] [--max-atoms N]
          [--allow-unchecked] [--force]     MLTENV=reactot, CPU 4c. Entry k = the round-1 model that held out T1 fold k
                                            (FT_0 := FTpilot_<base>_0). -> $MLT/g4/plan.json, results/G4_plan.json
                                            {complete, predicted_core_h{total,...}, est{opt, run}} (no QM). D0 rows
                                            (user decision D0-plus 'G-4에 D0 포함'): + n_d0 per model from the D0
                                            training side (decision G4-d0)
  g4_opt <i|@I> <n> [--models ...] [--workers W]
                                            MLTENV=mlts, 1 GPU + 16c: MLTS TS optimisations of task i of n (contiguous
                                            chunk of the model-major list; one MLTS server per model at a time, W CPU
                                            workers) -> $MLT/g4/opt/k<k>_<model>/<start>/recs/<rxn_id>.json (+ summary)
  run G4 <i|@I> <n> [--only rxn,...]        MLTENV=reactot, 64-core cpu2: serial label-level ORCA EnGrad at each MLTS TS
                                            of slice i (LPT on n_atoms^3 over the plan) -> $MLT/dft/runs/G4/g4_<rxn>.json
  collect [--g4]                            MLTENV=reactot: $MLT/dft/gen_G4.extxyz (kind al_ts, max|F| > 0.002
                                            Eh/bohr) + results/G4_report.json {complete, core_h_measured{total},
                                            n_selected, ...}
  status                                    one line per step

Env: MLT_G4_TIGHT (tight | verytight | none; default tight), MLT_WORKERS, MLT_G4_SRC / MLT_G4_SRC_RES (read-only input
roots for tests; default $MLT / results), and the mlt_common MLT_* overrides.
sbatch (login: one-shot only; `squeue -u $USER -h | wc -l` first) — normally submitted by the launcher (PIPELINE.md):
  sbatch -p cpu1,cpu2 -J mlt_g4plan --time=48:00:00 -c 4 --mem=16G jobs/mlt.slurm g4_al.py g4_plan --models <m0..m4>
  sbatch -p gpu1,gpu2,gpu3,gpu4,gpu5,gpu6 --gres=gpu:1 -c 16 --mem=64G --array=0-<n-1> --export=ALL,MLTENV=mlts \
         --time=48:00:00 jobs/mlt.slurm g4_al.py g4_opt @I <n> --models <m0..m4>
  sbatch -p cpu2 -c 64 --mem=240G --array=0-<n-1> --time=48:00:00 jobs/mlt.slurm g4_al.py run G4 @I <n>
  sbatch -p cpu1,cpu2 -c 4 --mem=32G --time=48:00:00 jobs/mlt.slurm g4_al.py collect --g4
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import socket  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from collections import Counter  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import mlt_common as M  # noqa: E402
import gen_dft as GD  # noqa: E402            (read-only import: EnGrad worker, level checks, STOP file)
import mlts_bridge as B  # noqa: E402
import ts_geom as TG  # noqa: E402            (evaluate_one / table / summarize / g1_classes / target_rows)

CODE_VERSION = "mlts-g4-1"
T_START = time.time()

# ---------------------------------------------------------------- paths (writes: $MLT and results only)
G4 = M.MLT / "g4"
PLAN = G4 / "plan.json"
OPT = G4 / "opt"
STEPS = G4 / "steps"
GROUP = "G4"
RUNS_G4 = GD.RUNS / GROUP                       # gen_dft.run_struct record folder for group G4
EXT_G4 = GD.DFT / "gen_G4.extxyz"               # finetune.AL_INPUT
G4_PLAN_JSON, G4_REPORT = M.RES / "G4_plan.json", M.RES / "G4_report.json"
SRC_MLT = Path(os.environ.get("MLT_G4_SRC", str(M.MLT)))         # read-only inputs (tests point outputs elsewhere)
SRC_RES = Path(os.environ.get("MLT_G4_SRC_RES", str(M.RES)))
D0_PLAN_DIR = SRC_MLT / "dft" / "plan"
D0_POOL, D0_G2PLAN, D0_HOLD = (D0_PLAN_DIR / "g2_pool_d0.json", D0_PLAN_DIR / "g2_plan_d0.json",
                               D0_PLAN_DIR / "d0_holdout.csv")             # gen_d0 (read only)

# ---------------------------------------------------------------- constants
N_MODELS = 5
N_PER_MODEL = 200                               # x 5 = 1,000 reactions (spec 3 G-4)
N_D0_PER_MODEL = 100                            # x 5 = 500 D0 training-side reactions (user decision D0-plus)
SEED_OFF_D0 = 450                               # default_rng(SEED + 450 + k): D0 draw of model k
FMAX_SELECT = 0.002                             # Eh/bohr, max atomic |F| (spec: 최대 힘 > 0.002 Eh/bohr)
SEED_OFF = 400                                  # default_rng(SEED + 400 + k): G-4 draw of model k
GEOM_OK = ("ok", "not_converged")               # MLTS outcomes whose final geometry gets an EnGrad
MIN_PAIR_A = 0.5                                # a collapsed geometry (any two atoms closer) gets no EnGrad
OPT_WORKERS_DEFAULT = 14                        # 16-CPU GPU task - 2 (ts_geom default for MLIPs)
OPT_WALL_DEFAULT_S = 300.0
_TIGHT = os.environ.get("MLT_G4_TIGHT", "tight").strip().lower()
TIGHT = False if _TIGHT in ("", "0", "none", "false", "default") else _TIGHT
MAX_PIPELINE_ERR = 20

DECISIONS = {
    "G4-alloc": "1,000 reactions = 200 per round-1 fold model FT_k (k = 0..4, the --models order; FT_0 = "
                "FTpilot_<base>_0), drawn from that model's own training side: T1 fold != k (fold -1 anchor-only cores "
                "included), minus the reactions of its validation split (rxn_ids of <split_dir>/valid.xyz from "
                "results/ft/<model>_data.json, split must be fold<k>), minus the reactions that already have a "
                "label-level G2->label path (gen_dft G-2 plan, 400) or a Phase I G3L / Part B path (60), minus "
                "reactions already drawn for a smaller k (1,000 distinct reactions). Strata dipole x g1 class "
                "(C_compare same_ts_g1: same / different / none), 200 allocated proportionally to the stratum sizes "
                "of the remaining pool (largest remainder, ties in stratum order), uniform order inside a stratum "
                "with numpy default_rng(SEED + 400 + k). The pool is first restricted to reactions with a label TS "
                "and a start TS file (G2 start: 650 of 9,266 have none); a drawn reaction whose start atom order "
                "differs from the label TS is skipped and the next one in that stratum is taken.",
    "G4-start": "start = the G2 TS in label order ($G2/g2_final/<rxn>/ts.xyz; the deployment path starts from an "
                "xTB-level TS, so G-4 samples the MLTS PES where deployment walks; the label-start neighbourhood is "
                "already covered by G-1 ts_near / ts_imag). --start label is allowed and recorded in the plan; only "
                "reactions with a usable start are drawn (no fallback).",
    "G4-opt": "MLTS optimisation = ts_geom.evaluate_one (the FT-2 evaluator: mlts_bridge.optimize_ts, ORCA ExtOpt "
              "OptTS with the D1 OptTS block, MLIP initial Hessian, Recalc analytic:30, n_imag < -50 cm-1, |dd| / "
              "RMSD / same_ts against the label TS) with the round-1 model of the allocation; one MLTS server per "
              "model per task, W = CPUs - 2 ORCA workers.",
    "G4-tight": "TightOpt (DECISIONS A-1 '수렴 기준': MLTS optimisations use TightOpt): ts_geom.evaluate_one(tight="
                "MLT_G4_TIGHT, default 'tight') (the same convergence as FT-2 / X since decision TSG-tight).",
    "G4-errors": "a python exception in one reaction is retried once in place; a second failure writes a "
                 "status 'pipeline_error' record (no geometry) only while the MLTS server is alive: a dead server writes "
                 "no error record and the task exits 1 (infra retry; the reactions stay to do); > 20 errors: no error "
                 "records, exit 3.",
    "G4-d0": "D0 (user decision D0-plus: 'G-4에 D0 포함'): + 100 D0 reactions per round-1 model (500), the D0 training "
             "side = gen_d0 g2_pool_d0.json ranked units (G2c xTB TS in label order, label TS = the Coley G16 TS, user "
             "decision D0-gauss) minus every reaction of $MLT/dft/plan/d0_holdout.csv (evaluation + siblings; finetune "
             "drops their TS-derived rows anyway; the file must exist (else not ready, exit 2) and match g2_plan_d0.json "
             "holdout.sha256 (else STOP)), minus the D0G2 training units (g2_plan_d0.json rxns: they already "
             "have a label-level path), neutral only (charge1 + charge2 = 0: mlts_bridge.optimize_ts has no charge "
             "argument); strata = D0 class, largest remainder, default_rng(SEED + 450 + k); a D0 reaction belongs to "
             "every FT_k training side, so model k is only its optimiser. Records $MLT/g4/opt/k<k>_<model>/d0_<start>/"
             "recs/<d0_id>.json, EnGrad struct_id g4d0_<d0_id>; extxyz tags rxn_ids '' / folds '' / dipoles = the D0 "
             "class (FT_LODO_d drops class d), info d0_id / d0_class (finetune holdout rule by d0_id).",
    "G4-geom": "the EnGrad structure is the final MLTS geometry of status ok (converged) and not_converged (MaxIter) "
               "optimisations (rounded to 1e-6 A in the record); timeout / orca_error / pipeline_error / no_start "
               "and geometries with two atoms closer than 0.5 A get no EnGrad (counted).",
    "G4-engrad": "EnGrad = gen_dft.run_struct unchanged (mlt_common.orca_grad_input: the D1 'grad' keyword set, "
                 "ORCA 6.1.1 serial, label-level input echo + SMD CDS + water epsilon checks -> $MLT/dft/STOP on a "
                 "level mismatch, one rerun on failure), group G4, struct_id g4_<rxn_id>, kind al_ts, charge = label "
                 "charge1 + charge2, multiplicity 1; slices = LPT on n_atoms^3 over the 1,000 planned reactions "
                 "(gen_dft.slice_units), so the slices never depend on which optimisations finished.",
    "G4-select": "selection: max over atoms of |F_i| (per-atom force norm) > 0.002 Eh/bohr; the max |gradient "
                 "component| is recorded too (count reported). Selected structures -> $MLT/dft/gen_G4.extxyz "
                 "(mlt_common.ext_atoms: REF_energy eV, REF_forces eV/A), kind al_ts, source mlts_<start>_optts, "
                 "tags rxn_ids / folds / dipoles of the reaction (finetune drops it from FT_<fold> and "
                 "FT_LODO_<dipole>), dform_label_A = max |d - d_label| over the two label forming bonds (finetune "
                 "TS-near rule), model / model_k / mlts_status / fmax_eh_bohr info.",
    "G4-cost": "predicted_core_h = sum over the planned reactions of the G-0 EnGrad fit a N^b (results/"
               "G0_estimate.json engrad_fit; an upper bound: every reaction with a geometry). core_h_measured.total = "
               "the label-level EnGrad core-h (all attempts) = what counts against the 5,000 core-h G budget; the "
               "MLTS optimisation cost (GPU s, wall) is reported separately (opt_*), not as DFT core-h.",
    "G4-est": "est.opt.unit_wall_h = median ts_geom wall per reaction of the g2 (else label) start from the round-1 "
              "FT-2 summaries of the listed models ($MLT/ft2/<model>/<start>/summary.json), else the FT2-p pilots, "
              "else Z_report geometry walls (median over candidates), else 300 s; workers 14. est.run.unit_wall_h = "
              "median a N^b (1-core EnGrad), workers 64.",
}


# ================================================================ helpers
def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _rj(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def _cpus():
    return int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1)


def _sha_file(p):
    try:
        return M.sha256(p)
    except OSError:
        return None


def _sha_obj(o):
    return hashlib.sha256(json.dumps(o, sort_keys=True, default=str).encode()).hexdigest()


def _q(v, qs=(0.0, 0.1, 0.5, 0.9, 1.0)):
    a = np.asarray([float(x) for x in v if x is not None and np.isfinite(float(x))], float)
    if not len(a):
        return dict(n=0)
    out = {f"q{int(round(q * 100))}": float(np.quantile(a, q)) for q in qs}
    out.update(n=int(len(a)), mean=float(a.mean()), sum=float(a.sum()))
    return out


def model_dir_name(k, name):
    return f"k{int(k)}_{str(name).replace('/', '_')}"


def opt_dir(plan, k, d0=False):
    return OPT / model_dir_name(k, plan["model_names"][k]) / ((f"d0_{plan['start']}") if d0 else plan["start"])


def opt_rec(plan, k, rxn):
    return TG.rec_path(opt_dir(plan, k), rxn)


def sid_of(rxn):
    return f"g4_{int(rxn)}"


def engrad_rec(rxn):
    return GD.rec_path(GROUP, sid_of(rxn))


# ---- plan rows: D1 (set 'd1', default) and D0 (set 'd0', decision G4-d0)
def is_d0(r):
    return r.get("set") == "d0"


def ukey(r):
    """the unit key of a plan row: the D1 rxn_id (int) or 'd0_<d0_id>'."""
    return f"d0_{int(r['d0_id'])}" if is_d0(r) else int(r["rxn_id"])


def row_opt_rec(plan, r):
    return TG.rec_path(opt_dir(plan, int(r["k"]), is_d0(r)), int(r["d0_id"] if is_d0(r) else r["rxn_id"]))


def row_sid(r):
    return f"g4d0_{int(r['d0_id'])}" if is_d0(r) else sid_of(r["rxn_id"])


def row_engrad_rec(r):
    return GD.rec_path(GROUP, row_sid(r))


def row_label(r):
    return Path(r["label_file"]) if is_d0(r) else M.label_ts(int(r["rxn_id"]))


def load_plan():
    P = _rj(PLAN)
    if not P or not P.get("rxns"):
        M.not_ready(f"{PLAN} missing: run 'g4_al.py g4_plan --models ...' first")
    return P


def parse_models(s):
    ms = [x.strip() for x in str(s or "").split(",")]
    if len(ms) != N_MODELS or not all(ms):
        M.stop(f"--models needs {N_MODELS} comma-separated names (entry k = the model that held out fold k), got {s!r}")
    return ms


def pairs_of(fp):
    return [tuple(p) for p in M.formed_pairs(fp)]


# ================================================================ plan
def valid_rxns(split_dir):
    """rxn_ids of the validation structures of a finetune split (<split_dir>/valid.xyz comment lines; the tag value is
    '"_JSON [\"1005431\", ...]"' (mlt_common.write_extxyz) or a plain string). -> (set, n_structures)."""
    import re
    p = Path(split_dir) / "valid.xyz"
    rx_val = re.compile(r'rxn_ids=("(?:[^"\\]|\\.)*"|\S+)')
    out, n = set(), 0
    with open(p) as f:
        while True:
            l1 = f.readline()
            if not l1:
                break
            if not l1.strip():
                continue
            na = int(l1.split()[0])
            l2 = f.readline()
            m = rx_val.search(l2)
            if m:
                out |= {int(x) for x in re.findall(r"\d+", m.group(1))}
            for _ in range(na):
                f.readline()
            n += 1
    return out, n


def model_check(k, model, allow_unchecked):
    """fold of the model from results/ft/<model>_data.json (must be fold<k>) and its validation reactions."""
    dj = SRC_RES / "ft" / f"{model}_data.json"
    D = _rj(dj)
    if D is None:
        if allow_unchecked:
            return dict(model=model, checked=False, note=f"{dj} missing: fold assumed = list position {k} "
                                                         "(--allow-unchecked, tests / stand-in models)"), set()
        M.not_ready(f"{dj} missing (the fold-{k} model's split): cannot verify the fold or read its validation split")
    split = D.get("split") or (D.get("run_info") or {}).get("split")
    if split != f"fold{k}":
        M.stop(f"--models entry {k} = {model} has split {split!r}, expected fold{k} (entry k must hold out fold k)")
    sd = D.get("split_dir")
    try:
        V, nv = valid_rxns(sd)
    except (OSError, TypeError, ValueError) as e:
        if allow_unchecked:
            return dict(model=model, checked=True, split=split, split_dir=sd, valid_read=False,
                        note=f"valid.xyz unreadable ({type(e).__name__}: {e}); no validation exclusion"), set()
        M.not_ready(f"{model}: validation split {sd}/valid.xyz unreadable ({type(e).__name__}: {e})")
    return dict(model=model, checked=True, split=split, split_dir=sd, valid_read=True, n_valid_structures=nv,
                n_valid_rxns=len(V), data_fingerprint=D.get("fingerprint")), V


def path_rxns():
    """reactions with a label-level path already: gen_dft G-2 plan (400) + Phase I G3L / Part B units (60)."""
    g2 = _rj(SRC_MLT / "dft" / "plan" / "g2_plan.json") or {}
    a = {int(u["rxn_id"]) for u in g2.get("rxns") or []}
    b = set(GD.phase_i_rxns())
    return a, b, bool(g2)


def start_ok(rxn, start):
    """(usable?, reason): the start TS exists and has the label TS element sequence (ts_geom no_start rules)."""
    lab = M.label_ts(rxn)
    sp = TG.start_geometry(rxn, start)
    if not lab.is_file():
        return False, "no_label"
    if not sp.is_file():
        return False, "no_start"
    if list(M.read_xyz(lab)[0]) != list(M.read_xyz(sp)[0]):
        return False, "atom_order"
    return True, ""


def largest_remainder(sizes, n):
    tot = sum(sizes.values())
    if tot <= 0 or n <= 0:
        return {k: 0 for k in sizes}
    n = min(n, tot)
    raw = {k: n * v / tot for k, v in sizes.items()}
    q = {k: min(sizes[k], int(np.floor(x))) for k, x in raw.items()}
    left = n - sum(q.values())
    for k in sorted(sizes, key=lambda s: (-(raw[s] - np.floor(raw[s])), list(sizes).index(s))):
        if left <= 0:
            break
        if q[k] < sizes[k]:
            q[k] += 1
            left -= 1
    while left > 0:                                            # strata exhausted: fill from the largest free ones
        for k in sorted(sizes, key=lambda s: -(sizes[s] - q[s])):
            if left > 0 and q[k] < sizes[k]:
                q[k] += 1
                left -= 1
    return q


def d0_pool(start, max_atoms=None):
    """decision G4-d0: the D0 training-side candidates (dicts) and the exclusion counts."""
    import csv
    P = _rj(D0_POOL)
    if not P or not P.get("ranked"):
        return [], dict(note=f"{D0_POOL} missing / empty: no D0 rows", pool=0)
    g2 = _rj(D0_G2PLAN) or {}
    hrec = g2.get("holdout") or {}
    # the D0 holdout list (evaluation + siblings, gen_d0 D0-hold / D0-sib) must exist and be the one the D0 plan
    # recorded (D0-hold-fix): without it the draw could take an evaluation reaction (decision G4-d0)
    if not D0_HOLD.is_file():
        M.not_ready(f"{D0_HOLD} missing: `gen_d0.py plan_d0 --n-near 2 --n-g2 200 --n-holdout 100` has not run (G-4 "
                    f"D0 rows never draw a holdout reaction)")
    if hrec.get("sha256") and hrec["sha256"] != _sha_file(D0_HOLD):
        M.stop(f"{D0_HOLD} sha256 differs from g2_plan_d0.json holdout.sha256 (edited or replaced; decision D0-hold-fix)")
    with open(D0_HOLD, newline="") as fh:
        hold = {int(float(r["d0_id"])) for r in csv.DictReader(fh) if r.get("d0_id")}
    train = {int(u["rxn_id"]) for u in g2.get("rxns") or []}
    hold |= {int(x) for x in (hrec.get("ids") or hrec.get("d0_ids") or [])}
    hold |= {int(x) for x in (hrec.get("sibling_ids") or [])}
    out, ex = [], Counter()
    for u in P["ranked"]:
        rid = int(u.get("d0_id", u.get("rxn_id")))
        if rid in hold:
            ex["holdout"] += 1
            continue
        if rid in train:
            ex["d0g2_training_path"] += 1
            continue
        if int(u.get("charge") or 0) != 0:
            ex["charged"] += 1
            continue
        if max_atoms and int(u["n_atoms"]) > int(max_atoms):
            ex["max_atoms"] += 1
            continue
        out.append(dict(set="d0", d0_id=rid, rxn_id=rid, job_id=f"D0_{rid}", d0_class=str(u.get("d0_class") or u["dipole"]),
                        dipole=str(u.get("d0_class") or u["dipole"]), n_atoms=int(u["n_atoms"]), fold_T1=-9, g1_class="none",
                        pairs=[[int(a), int(b)] for a, b in u["pairs"]], label_file=str(u["label_ts"]),
                        start_file=str(u["label_ts"] if start == "label" else u["start"]), charge=0))
    return out, dict(pool=len(P["ranked"]), candidates=len(out), excluded=dict(ex), holdout_file=str(D0_HOLD),
                     holdout_sha256=_sha_file(D0_HOLD), g2_plan_d0=_sha_file(D0_G2PLAN), g2_pool_d0=_sha_file(D0_POOL))


def d0_start_ok(r):
    lab, sp = Path(r["label_file"]), Path(r["start_file"])
    if not lab.is_file():
        return False, "no_label"
    if not sp.is_file():
        return False, "no_start"
    if list(M.read_xyz(lab)[0]) != list(M.read_xyz(sp)[0]):
        return False, "atom_order"
    return True, ""


def opt_wall_estimate(models, start):
    def med(paths):
        w = []
        for p in paths:
            a = ((_rj(p) or {}).get("all") or {})
            if a.get("wall_median_s"):
                w.append(float(a["wall_median_s"]))
        return float(np.median(w)) if w else None, len(w)
    for st in dict.fromkeys((start, "label")):
        v, n = med([SRC_MLT / "ft2" / m / st / "summary.json" for m in models])
        if v:
            return v, f"round-1 FT-2 summaries ({n} models, {st} start)"
        v, n = med(sorted((SRC_MLT / "ft2").glob(f"FTpilot_*_0/{st}/summary.json")))
        if v:
            return v, f"FT2-p pilot summaries ({n}, {st} start)"
    Z = _rj(SRC_RES / "Z_report.json") or {}
    w = [float(((c or {}).get("geometry") or {}).get("wall_median_s")) for c in (Z.get("candidates") or {}).values()
         if ((c or {}).get("geometry") or {}).get("wall_median_s")]
    if w:
        return float(np.median(w)), f"Z_report geometry wall_median_s (median of {len(w)} candidates, label start)"
    return OPT_WALL_DEFAULT_S, "default"


def cmd_plan(models, n_per=N_PER_MODEL, start="g2", max_atoms=None, allow_unchecked=False, force=False,
             n_d0=N_D0_PER_MODEL):
    import pandas as pd
    if start not in ("g2", "label"):
        M.stop(f"--start must be g2 or label, not {start!r}")
    models = parse_models(models)
    T = M.targets()
    if max_atoms:
        T = T[T.n_atoms <= int(max_atoms)]
    labs = {int(d["rxn_id"]): d for d in M.read_json(M.LABELS)}
    cls = TG.g1_classes()
    checks, valid = [], []
    for k, m in enumerate(models):
        c, V = model_check(k, m, allow_unchecked)
        checks.append(c)
        valid.append(V)
    g2p, phi, have_g2p = path_rxns()
    d0c, d0info = d0_pool(start, max_atoms) if int(n_d0) > 0 else ([], dict(note="--n-d0-per-model 0"))
    params = dict(code=CODE_VERSION, models=models, n_per_model=int(n_per), start=start, max_atoms=max_atoms,
                  seed=[M.SEED + SEED_OFF + k for k in range(N_MODELS)], labels=_sha_file(M.LABELS),
                  s_folds=_sha_file(M.S_FOLDS), c_compare=_sha_file(M.C_COMPARE),
                  g2_plan=_sha_file(SRC_MLT / "dft" / "plan" / "g2_plan.json"),
                  valid=[_sha_obj(sorted(V)) for V in valid], checks=[c.get("data_fingerprint") for c in checks],
                  n_d0_per_model=int(n_d0), d0={k: d0info.get(k) for k in ("holdout_sha256", "g2_plan_d0", "g2_pool_d0")},
                  d0_seed=[M.SEED + SEED_OFF_D0 + k for k in range(N_MODELS)])
    fp = _sha_obj(params)
    old = _rj(PLAN)
    if old and old.get("fingerprint") == fp and not force and (_rj(G4_PLAN_JSON) or {}).get("complete"):
        print(f"[g4_plan] up to date ({PLAN})", flush=True)
        return 0
    if old and old.get("fingerprint") != fp and any(OPT.glob("*/*/recs/*.json")) and not force:
        M.stop(f"G-4 plan inputs changed after MLTS records were written ({old.get('fingerprint')} -> {fp}); "
               "not re-planned (use --force only after deciding what to do with $MLT/g4/opt)")
    t0 = time.time()
    has = {int(r): M.label_ts(int(r)).is_file() and TG.start_geometry(int(r), start).is_file() for r in T.rxn_id}
    n_no_start = int(sum(not v for v in has.values()))
    T = T[T.rxn_id.map(lambda r: has[int(r)])]
    T = T.assign(g1_class=T.rxn_id.map(lambda r: cls.get(int(r), "none")))
    T = T.assign(stratum=T.dipole + "|" + T.g1_class)
    strata_order = [f"{d}|{c}" for d in M.DIPOLES for c in ("same", "different", "none")]
    taken, rows, alloc = set(), [], {}
    rej = Counter()
    for k, m in enumerate(models):
        pool = T[(T.fold_T1 != k) & ~T.rxn_id.isin(valid[k] | g2p | set(phi) | taken)]
        sizes = {s: int((pool.stratum == s).sum()) for s in strata_order if (pool.stratum == s).any()}
        quota = largest_remainder(sizes, int(n_per))
        rng = np.random.default_rng(M.SEED + SEED_OFF + k)
        got = Counter()
        for s in sizes:
            cand = pool[pool.stratum == s].sort_values("rxn_id").reset_index(drop=True)
            order = rng.permutation(len(cand))
            for j in order:
                if got[s] >= quota[s]:
                    break
                t = cand.iloc[int(j)]
                rid = int(t.rxn_id)
                ok, why = start_ok(rid, start)
                if not ok:
                    rej[why] += 1
                    continue
                d = labs[rid]
                rows.append(dict(rxn_id=rid, job_id=str(t.job_id), k=k, model=m, fold_T1=int(t.fold_T1),
                                 dipole=str(t.dipole), g1_class=str(t.g1_class), n_atoms=int(t.n_atoms),
                                 formed_pairs_ts=str(t.formed_pairs_ts),
                                 charge=int(d.get("charge1") or 0) + int(d.get("charge2") or 0)))
                taken.add(rid)
                got[s] += 1
        short = {s: quota[s] - got[s] for s in quota if got[s] < quota[s]}
        alloc[k] = dict(model=m, heldout_fold=k, pool=int(len(pool)), quota=quota, drawn=dict(got),
                        n=int(sum(got.values())), shortfall=short, seed=M.SEED + SEED_OFF + k,
                        n_valid_excluded=int(len(valid[k] & set(T.rxn_id))), check=checks[k])
    # D0 (decision G4-d0): every model draws n_d0 from the remaining D0 training side, strata = D0 class
    d0rows, d0alloc, d0taken = [], {}, set()
    for k, m in enumerate(models):
        if int(n_d0) <= 0 or not d0c:
            break
        pool = [r for r in d0c if r["d0_id"] not in d0taken]
        sizes = {}
        for r in pool:
            sizes[r["d0_class"]] = sizes.get(r["d0_class"], 0) + 1
        sizes = dict(sorted(sizes.items()))
        quota = largest_remainder(sizes, int(n_d0))
        rng = np.random.default_rng(M.SEED + SEED_OFF_D0 + k)
        got = Counter()
        for s in sizes:
            cand = sorted((r for r in pool if r["d0_class"] == s), key=lambda r: r["d0_id"])
            for j in rng.permutation(len(cand)):
                if got[s] >= quota[s]:
                    break
                r = cand[int(j)]
                ok, why = d0_start_ok(r)
                if not ok:
                    rej[f"d0_{why}"] += 1
                    continue
                d0rows.append(dict(r, k=k, model=m))
                d0taken.add(r["d0_id"])
                got[s] += 1
        d0alloc[k] = dict(model=m, pool=len(pool), quota=quota, drawn=dict(got), n=int(sum(got.values())),
                          seed=M.SEED + SEED_OFF_D0 + k)
    rows.sort(key=lambda r: (r["k"], r["rxn_id"]))
    d0rows.sort(key=lambda r: (r["k"], r["d0_id"]))
    R = pd.DataFrame(rows)
    # leakage self-check: no reaction drawn for the model that held out its fold, none from that model's validation
    bad_fold = int(sum(r["fold_T1"] == r["k"] for r in rows))
    bad_val = int(sum(r["rxn_id"] in valid[r["k"]] for r in rows))
    bad_d0 = len(d0rows) - len({r["d0_id"] for r in d0rows})
    if bad_fold or bad_val or R.rxn_id.duplicated().any() or bad_d0:
        M.stop(f"G-4 plan self-check failed: fold {bad_fold}, validation {bad_val}, duplicates "
               f"{int(R.rxn_id.duplicated().sum())}, D0 duplicates {bad_d0}")
    # cost
    g0 = _rj(SRC_RES / "G0_estimate.json") or {}
    fit = g0.get("engrad_fit")
    if not fit:
        M.not_ready(f"{SRC_RES / 'G0_estimate.json'}: no engrad_fit (G-0 not done)")
    N = np.asarray([float(r["n_atoms"]) for r in rows + d0rows], float)
    ch = fit["a"] * N ** fit["b"] if len(N) else np.zeros(0)
    n_all = len(rows) + len(d0rows)
    wall_s, wall_src = opt_wall_estimate(models, start)
    gpu = []
    for m in models:
        a = ((_rj(SRC_MLT / "ft2" / m / start / "summary.json") or {}).get("all") or {})
        if a.get("gpu_s_median"):
            gpu.append(float(a["gpu_s_median"]))
    est_opt = dict(units_total=int(n_all), unit_wall_h=wall_s / 3600.0, workers=OPT_WORKERS_DEFAULT,
                   source=wall_src, tight=TIGHT)
    est_run = dict(units_total=int(n_all), unit_wall_h=float(np.median(ch)) if len(ch) else 0.05, workers=64,
                   source="G-0 engrad_fit a N^b (1-core EnGrad wall = core-h)")
    P = dict(phase="G-4 plan", code=CODE_VERSION, written=_now(), fingerprint=fp, params=params, models=models,
             model_names=[B.spec_name(B.resolve_spec(m)) if m.endswith((".model", ".jpt")) else m for m in models],
             start=start, n_per_model=int(n_per), n_d0_per_model=int(n_d0), rxns=rows + d0rows)
    M.write_json(PLAN, P)
    by = {}
    for k in range(N_MODELS):
        Rk = R[R.k == k] if len(R) else R
        by[k] = dict(alloc[k], by_dipole=dict(Counter(Rk.dipole)) if len(Rk) else {},
                     by_g1_class=dict(Counter(Rk.g1_class)) if len(Rk) else {},
                     by_fold=dict(Counter(str(x) for x in Rk.fold_T1)) if len(Rk) else {},
                     n_atoms=_q(Rk.n_atoms) if len(Rk) else dict(n=0),
                     d0=dict(d0alloc.get(k) or dict(n=0), by_class=dict(Counter(r["d0_class"] for r in d0rows
                                                                               if r["k"] == k))))
    rep = dict(phase="G-4 plan", spec_section="3 Phase G (G-4)", code=CODE_VERSION, written=_now(),
               host=socket.gethostname(), slurm_job=os.environ.get("SLURM_JOB_ID"), complete=True, n_todo=0, stop=None,
               plan=str(PLAN), fingerprint=fp, models=models, start=start, n_total=int(n_all), n_d1=int(len(rows)),
               n_d0=int(len(d0rows)), n_per_model=int(n_per), n_d0_per_model=int(n_d0), max_atoms=max_atoms, by_model=by,
               exclusions=dict(no_start_file=n_no_start, g2_path_rxns=len(g2p), g2_plan_found=have_g2p,
                               phase_i_rxns=len(phi), start_rejects=dict(rej), d0=d0info),
               leakage_check=dict(drawn_in_heldout_fold=bad_fold, drawn_in_validation=bad_val, d0_duplicates=bad_d0,
                                  pass_=True),
               predicted_core_h=dict(total=float(ch.sum()), engrad=float(ch.sum()), per_engrad=_q(ch),
                                     d1=float(ch[:len(rows)].sum()), d0=float(ch[len(rows):].sum()),
                                     engrad_fit=fit, n_structures_max=int(n_all),
                                     note="upper bound: one EnGrad per planned reaction (G4-cost)"),
               predicted_gpu_h=(float(np.median(gpu)) * n_all / 3600.0) if gpu else None,
               est=dict(units_total=est_opt["units_total"], unit_wall_h=est_opt["unit_wall_h"],
                        workers=est_opt["workers"], opt=est_opt, run=est_run),
               decisions=DECISIONS, wall_s=time.time() - t0)
    M.write_json(G4_PLAN_JSON, rep)
    print(json.dumps(dict(n_total=rep["n_total"], n_d0=rep["n_d0"], per_model={k: v["n"] for k, v in by.items()},
                          per_model_d0={k: (v["d0"] or {}).get("n") for k, v in by.items()},
                          predicted_core_h=round(rep["predicted_core_h"]["total"], 1), est=rep["est"],
                          start_rejects=dict(rej)), indent=1, default=str), flush=True)
    return 0


# ================================================================ g4_opt (MLTENV=mlts, GPU)
def chunk(units, n, i):
    L = len(units)
    return units[(i * L) // n:((i + 1) * L) // n]


def d0_row(r):
    """ts_geom.evaluate_one row of a D0 plan row (label pairs as formed_pairs_ts text)."""
    return dict(job_id=r["job_id"], dipole=r["dipole"], fold_T1=r["fold_T1"], n_atoms=r["n_atoms"],
                formed_pairs_ts=" ".join(f"{int(a)}-{int(b)}" for a, b in r["pairs"]), skeleton="")


def opt_model(plan, k, prow, rows, cls, workers, log=print):
    """MLTS TS optimisations of model k for the plan rows prow (D1 and D0; records skipped when present) -> info."""
    name = plan["models"][k]
    try:
        spec = B.resolve_spec(name)
    except ValueError as e:
        M.not_ready(f"model {name}: {e}")
    if spec.get("weights") and not Path(spec["weights"]).is_file():
        M.not_ready(f"model {name}: weights {spec['weights']} missing (round-1 fine-tune not finished?)")
    card = _rj(spec["card"]) if spec.get("card") else None
    if card and card.get("split") and card["split"] != f"fold{k}":
        M.stop(f"model {name} (entry {k}) was trained on split {card['split']}, expected fold{k}")
    for d0 in (False, True):
        od = M.assert_write_ok(opt_dir(plan, k, d0))
        if d0 and not any(is_d0(r) for r in prow):
            continue
        (od / "recs").mkdir(parents=True, exist_ok=True)
    todo = [r for r in prow if not row_opt_rec(plan, r).is_file()]
    info = dict(k=k, model=name, spec={x: v for x, v in spec.items() if x != "card"}, start=plan["start"],
                n_rxns=len(prow), n_d0=sum(1 for r in prow if is_d0(r)), n_todo=len(todo), tight=TIGHT, started=_now())
    if not todo:
        info["note"] = "all records exist"
        return info
    h = B.start_server(spec)
    info["server_start"] = {x: v for x, v in (h.stats() or {}).items() if x != "spec"}
    work_root = B.local_tmp() / f"mlts_g4_{B.jobtag()}_{os.getpid()}_k{k}"
    work_root.mkdir(parents=True, exist_ok=True)
    mu, it = threading.Lock(), iter(todo)
    st = dict(halt="", timed_out=False, n_err=0, errs={}, counts=Counter())

    def alive():
        return h.proc is None or h.proc.poll() is None

    def claim():
        with mu:
            if st["halt"]:
                return None
            if TG.time_left_h() < TG.RESERVE_H:
                st["timed_out"] = True
                return None
            if not alive():                                            # G4-errors: a dead server stops the task
                st["halt"] = st["halt"] or "MLTS server process died"
                return None
            return next(it, None)

    def evaluate(r):
        if is_d0(r):
            od = opt_dir(plan, k, True)
            rec = TG.evaluate_one(int(r["d0_id"]), d0_row(r), spec, plan["start"], h.socket, od, work_root / "d0", None,
                                  tight=TIGHT, lab=r["label_file"], sp=r["start_file"])
            rec.update(set="d0", d0_id=int(r["d0_id"]), d0_class=r["d0_class"])
            return rec
        return TG.evaluate_one(int(r["rxn_id"]), rows[int(r["rxn_id"])], spec, plan["start"], h.socket,
                               opt_dir(plan, k), work_root, cls, tight=TIGHT)

    def loop():
        while True:
            r = claim()
            if r is None:
                return
            rp = row_opt_rec(plan, r)
            if rp.is_file():
                continue
            rec, err = None, None
            for _ in (1, 2):
                try:
                    rec = evaluate(r)
                    break
                except Exception as e:                                 # noqa: BLE001
                    err = f"{type(e).__name__}: {e}"[:500]
                    print(f"[g4_opt] pipeline error {ukey(r)}: {err}\n{traceback.format_exc(limit=5)}", file=sys.stderr,
                          flush=True)
                    if not alive():
                        break
            if rec is None:
                with mu:
                    st["n_err"] += 1
                    st["errs"][ukey(r)] = (r, err)
                    st["counts"]["pipeline_error"] += 1
                    if st["n_err"] > MAX_PIPELINE_ERR:
                        st["halt"] = f"more than {MAX_PIPELINE_ERR} pipeline errors"
                continue
            rec.update(g4=dict(k=k, model=name, tight=TIGHT, code=CODE_VERSION))
            M.write_text(rp, json.dumps(rec, default=M._json_default) + "\n")
            with mu:
                st["counts"][rec["status"]] += 1
            log(f"[g4_opt k{k}] {ukey(r)} {rec['status']} dd={rec.get('dd_max')} same={rec.get('same_ts')} "
                f"nimag={rec.get('n_imag')} cyc={rec.get('n_cycles')} wall={rec.get('wall_s', 0):.1f}s")
    t0 = time.time()
    try:
        nw = max(1, min(int(workers), len(todo)))
        info["workers"] = nw
        with ThreadPoolExecutor(nw) as ex:
            for f in [ex.submit(loop) for _ in range(nw)]:
                f.result()
    finally:
        shutil.rmtree(work_root, ignore_errors=True)
        info["server_alive_at_end"] = alive()
        info["server_end"] = {x: v for x, v in (h.stats() or {}).items() if x != "spec"}
        h.shutdown()
    if not st["halt"] and info["server_alive_at_end"]:                   # isolated failures: closed as records
        for key, (r, err) in st["errs"].items():
            rid = int(r["d0_id"] if is_d0(r) else r["rxn_id"])
            M.write_text(row_opt_rec(plan, r), json.dumps(dict(
                rxn_id=rid, job_id=r["job_id"], dipole=r["dipole"], fold_T1=r["fold_T1"], n_atoms=r["n_atoms"],
                start=plan["start"], model=B.spec_name(spec), g1_class=cls.get(rid, "none") if not is_d0(r) else "none",
                status="pipeline_error", error=err, host=socket.gethostname(), job=B.jobtag(), time=_now(),
                **(dict(set="d0", d0_id=rid, d0_class=r["d0_class"]) if is_d0(r) else {}),
                g4=dict(k=k, model=name, tight=TIGHT, code=CODE_VERSION))) + "\n")
    info.update(counts=dict(st["counts"]), n_pipeline_error=st["n_err"], halt=st["halt"], timed_out=st["timed_out"],
                wall_s=time.time() - t0, finished=_now())
    for d0 in (False, True):
        od = opt_dir(plan, k, d0)
        if not (od / "recs").is_dir() or not any((od / "recs").glob("*.json")):
            continue
        try:
            TG.summarize(od)
        except (SystemExit, Exception) as e:                             # noqa: BLE001  (diagnostic only)
            info[f"summary_error{'_d0' if d0 else ''}"] = f"{type(e).__name__}: {e}"[:300]
    return info


def cmd_opt(i, n, models=None, workers=None):
    plan = load_plan()
    if models and parse_models(models) != plan["models"]:
        M.stop(f"--models {models} != the G-4 plan models {plan['models']} (re-plan first)")
    if not 0 <= i < n:
        print(f"g4_opt: task {i} outside 0..{n - 1}", file=sys.stderr)
        return 2
    units = sorted(plan["rxns"], key=lambda r: (int(r["k"]), is_d0(r), int(r["d0_id"] if is_d0(r) else r["rxn_id"])))
    mine = chunk(units, n, i)
    rows = TG.target_rows([int(r["rxn_id"]) for r in mine if not is_d0(r)])
    cls = TG.g1_classes()
    nw = int(workers or os.environ.get("MLT_WORKERS") or max(1, _cpus() - 2))
    print(f"[g4_opt {i}/{n}] {len(mine)} reactions ({sum(1 for r in mine if is_d0(r))} D0), models "
          f"{sorted({int(r['k']) for r in mine})}, {nw} workers, tight={TIGHT}, job {B.jobtag()} on "
          f"{socket.gethostname()}, {TG.time_left_h():.1f} h left", flush=True)
    infos, rc = [], 0
    for k in sorted({int(r["k"]) for r in mine}):
        inf = opt_model(plan, k, [r for r in mine if int(r["k"]) == k], rows, cls, nw)
        infos.append(inf)
        if inf.get("halt"):
            rc = 1 if not inf.get("server_alive_at_end") else 3
            break
        if inf.get("timed_out"):
            break
    left = [ukey(r) for r in mine if not row_opt_rec(plan, r).is_file()]
    summ = dict(step="g4_opt", task=i, n_tasks=n, n_units=len(mine), n_left=len(left), left=left[:50], models=infos,
                slurm_job=B.jobtag(), host=socket.gethostname(), finished=_now(), code=CODE_VERSION)
    M.write_json(STEPS / f"opt_{i:02d}of{n:02d}.json", summ)
    print(json.dumps(dict(n_left=len(left), counts=[x.get("counts") for x in infos], halt=[x.get("halt") for x in infos]),
                     default=str), flush=True)
    if rc:
        return rc
    return 0 if not left else 2


# ================================================================ run G4 (MLTENV=reactot, 64-core cpu2)
def geom_of(plan, r):
    """(symbols, pos (N,3), opt record, reason): the EnGrad structure of planned reaction r, or reason."""
    rp = row_opt_rec(plan, r)
    rec = _rj(rp)
    if rec is None:
        return None, None, None, "opt_missing"
    if rec.get("status") not in GEOM_OK or not rec.get("pos_A"):
        return None, None, rec, f"no_geometry:{rec.get('status')}"
    syms = M.read_xyz(row_label(r))[0]
    pos = np.asarray(rec["pos_A"], float).reshape(-1, 3)
    if len(syms) != len(pos):
        return None, None, rec, "atom_count"
    dmin = float(np.min(np.linalg.norm(pos[:, None] - pos[None], axis=2)[np.triu_indices(len(pos), 1)]))
    if dmin < MIN_PAIR_A:
        return None, None, rec, "collapsed"
    return syms, pos, rec, ""


def engrad_row(plan, r, syms, pos, rec):
    disp = dict(model=plan["models"][r["k"]], model_k=int(r["k"]), start=plan["start"], mlts_status=rec.get("status"),
                converged=rec.get("converged"), n_cycles=rec.get("n_cycles"), dd_max=rec.get("dd_max"),
                same_ts=rec.get("same_ts"), n_imag=rec.get("n_imag"), tight=(rec.get("g4") or {}).get("tight"),
                opt_record=str(row_opt_rec(plan, r)))
    if is_d0(r):                    # decision G4-d0: no D1 tags; the D0 class as dipole (LODO); d0_id kept in disp / info
        disp.update(set="d0", d0_id=int(r["d0_id"]), d0_class=r["d0_class"])
        return dict(struct_id=row_sid(r), group=GROUP, kind="al_ts", rxn_ids="", folds="", dipoles=str(r["d0_class"]),
                    rxn_id=int(r["d0_id"]), mol_id="", fold_T1=-9, charge=int(r.get("charge", 0)), mult=1,
                    src=f"mlts:{plan['models'][r['k']]}:d0_{plan['start']}", symbols=" ".join(syms),
                    pos_A=[float(v) for v in pos.ravel()], disp=json.dumps(disp, default=M._json_default))
    return dict(struct_id=sid_of(r["rxn_id"]), group=GROUP, kind="al_ts", rxn_ids=str(r["rxn_id"]),
                folds=str(r["fold_T1"]), dipoles=str(r["dipole"]), rxn_id=int(r["rxn_id"]), mol_id="",
                fold_T1=int(r["fold_T1"]), charge=int(r.get("charge", 0)), mult=1,
                src=f"mlts:{plan['models'][r['k']]}:{plan['start']}", symbols=" ".join(syms),
                pos_A=[float(v) for v in pos.ravel()], disp=json.dumps(disp, default=M._json_default))


def cmd_run(group, i, n, only=None):
    if group != GROUP:
        print(f"run: group must be {GROUP}", file=sys.stderr)
        return 2
    if GD.stopped():
        return 3
    plan = load_plan()
    R = {str(ukey(r)): r for r in plan["rxns"]}
    if only:
        mine = [R[x] for x in sorted(str(y) for y in only) if x in R]
        loads = None
    else:
        if not 0 <= i < n:
            print(f"run: slice {i} outside 0..{n - 1}", file=sys.stderr)
            return 2
        U = [(x, float(r["n_atoms"]) ** 3, r) for x, r in R.items()]
        mine, loads = GD.slice_units(U, n, i)
    todo, why = [], Counter()
    for r in mine:
        if row_engrad_rec(r).is_file():
            why["done"] += 1
            continue
        syms, pos, rec, reason = geom_of(plan, r)
        if reason:
            why[reason.split(":")[0]] += 1
            continue
        todo.append(engrad_row(plan, r, syms, pos, rec))
    print(f"[run {GROUP} {i}/{n}] {len(mine)} planned ({sum(1 for r in mine if is_d0(r))} D0), {len(todo)} EnGrads to "
          f"do, {dict(why)}, {_cpus()} workers, job {GD._jobtag()} on {socket.gethostname()}, {GD.time_left_h():.1f} h "
          f"left", flush=True)
    ctx = GD.Ctx("g4")
    t0 = time.time()
    GD.run_items(todo, _cpus(), ctx, GD.RESERVE_H, GD.run_struct, lambda x: x["struct_id"])
    left = [x["struct_id"] for x in todo if not GD.rec_path(GROUP, x["struct_id"]).is_file()]
    summ = dict(step="run G4", slice=i, n_slices=n, only=sorted(only) if only else None, n_planned=len(mine),
                n_todo_at_start=len(todo), skipped=dict(why), stats=dict(ctx.stats), core_h=ctx.core_h,
                wall_s=time.time() - t0, n_left=len(left), left=left[:50], halt=ctx.halt, timed_out=ctx.timed_out,
                load_this=loads[i] if loads else None, slurm_job=GD._jobtag(), host=socket.gethostname(),
                finished=_now(), code=CODE_VERSION)
    if not only:
        M.write_json(STEPS / f"run_{i:02d}of{n:02d}.json", summ)
    print(json.dumps({x: summ[x] for x in ("stats", "core_h", "n_left", "skipped", "halt", "timed_out")}), flush=True)
    if ctx.halt:
        return 3
    return 0 if not left and not why.get("opt_missing") else 2


# ================================================================ collect
def fmax_norm(g):
    g = np.asarray(g, float).reshape(-1, 3)
    return float(np.linalg.norm(g, axis=1).max())


def cmd_collect():
    import pandas as pd
    plan = load_plan()
    T = {str(ukey(r)): r for r in plan["rxns"]}
    atoms, resolved, n_todo = [], 0, 0
    st = Counter()
    per = []
    fn, fc, core, n_att = [], [], 0.0, 0
    opt_recs = {k: [] for k in range(N_MODELS)}
    for key, r in sorted(T.items()):
        k = int(r["k"])
        d0 = is_d0(r)
        rid = int(r["d0_id"] if d0 else r["rxn_id"])
        orc = _rj(row_opt_rec(plan, r))
        if orc is None:
            st["opt_missing"] += 1
            n_todo += 1
            continue
        opt_recs[k].append(orc)
        syms, pos, _, reason = geom_of(plan, r)
        if reason:
            st[reason.split(":")[0] if not reason.startswith("no_geometry") else reason] += 1
            resolved += 1
            continue
        er = _rj(row_engrad_rec(r))
        if er is None:
            st["engrad_missing"] += 1
            n_todo += 1
            continue
        resolved += 1
        core += float(er.get("core_h") or 0.0)
        n_att += int(er.get("n_attempts") or 0)
        if er.get("status") != "ok":
            st[f"engrad_fail:{er.get('reason')}"] += 1
            continue
        st["engrad_ok"] += 1
        g = np.asarray(er["grad_eh_bohr"], float)
        a, b = fmax_norm(g), float(np.abs(g).max())
        fn.append(a)
        fc.append(b)
        pairs = [tuple(p) for p in r["pairs"]] if d0 else pairs_of(r["formed_pairs_ts"])
        lp = M.read_xyz(row_label(r))[1]
        dform = float(np.max(np.abs(np.subtract(B.dists(er["pos_A"], pairs), B.dists(lp, pairs)))))
        sel = a > FMAX_SELECT
        per.append(dict(key=key, set="d0" if d0 else "d1", rxn_id=rid, k=k, model=plan["models"][k], dipole=r["dipole"],
                        fold_T1=r["fold_T1"], n_atoms=r["n_atoms"], mlts_status=orc.get("status"),
                        same_ts=orc.get("same_ts"), dd_max=orc.get("dd_max"), fmax_norm_eh_bohr=a, fmax_comp_eh_bohr=b,
                        dform_label_A=dform, selected=sel, core_h=er.get("core_h")))
        if sel:
            st["selected"] += 1
            st["selected_d0" if d0 else "selected_d1"] += 1
            tags = (dict(rxn_ids="", folds="", dipoles=str(r["d0_class"]), d0_id=rid, d0_class=str(r["d0_class"]),
                         charge=int(r.get("charge", 0))) if d0 else
                    dict(rxn_ids=str(rid), folds=str(r["fold_T1"]), dipoles=str(r["dipole"])))
            atoms.append(M.ext_atoms(er["symbols"], er["pos_A"], er["energy_eh"], g, struct_id=er["struct_id"],
                                     kind="al_ts", source=f"mlts_{'d0_' if d0 else ''}{plan['start']}_optts",
                                     level="label", group=GROUP, model=str(plan["models"][k]), model_k=k,
                                     mlts_status=str(orc.get("status")), fmax_eh_bohr=a, dform_label_A=dform, **tags))
    if atoms:
        M.write_extxyz(EXT_G4, atoms)
    else:
        M.write_text(EXT_G4, "")
    rt = None
    if atoms:
        try:
            b0 = M.read_extxyz(EXT_G4, index=0)
            rt = dict(struct_id=b0.info.get("struct_id"), kind=b0.info.get("kind"), rxn_ids=b0.info.get("rxn_ids"),
                      folds=b0.info.get("folds"), dipoles=b0.info.get("dipoles"),
                      dE_eV=float(b0.info["REF_energy"]) - float(atoms[0].info["REF_energy"]),
                      dF_max_eV_A=float(np.abs(b0.arrays["REF_forces"] - atoms[0].arrays["REF_forces"]).max()),
                      n_read=len(M.read_extxyz(EXT_G4)))
            rt["pass_"] = (rt["struct_id"] == atoms[0].info["struct_id"] and rt["rxn_ids"] == atoms[0].info["rxn_ids"]
                           and rt["folds"] == atoms[0].info["folds"] and rt["dipoles"] == atoms[0].info["dipoles"]
                           and abs(rt["dE_eV"]) < 1e-6 and rt["dF_max_eV_A"] < 1e-6 and rt["n_read"] == len(atoms))
        except Exception as e:                                                 # noqa: BLE001
            rt = dict(error=f"{type(e).__name__}: {e}", pass_=False)
    D = pd.DataFrame(per)
    by_model = {}
    for k in range(N_MODELS):
        Dk = D[D.k == k] if len(D) else D
        geo = None
        if opt_recs[k]:
            try:
                df = pd.DataFrame(opt_recs[k])
                for c in ("dd_max", "rmsd_heavy", "rmsd_direct", "n_imag", "converged", "same_ts", "n_cycles", "gpu_s",
                          "n_ext_calls", "wall_s", "start_dd_max", "start_same_ts"):
                    if c not in df:
                        df[c] = None
                for c in ("dd_max", "rmsd_heavy", "rmsd_direct", "n_cycles", "gpu_s", "n_ext_calls", "wall_s",
                          "start_dd_max", "n_imag"):
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                geo = TG.table(df)
            except Exception as e:                                             # noqa: BLE001
                geo = dict(error=f"{type(e).__name__}: {e}")
        by_model[k] = dict(model=plan["models"][k], n_planned=sum(1 for r in T.values() if int(r["k"]) == k),
                           n_opt_records=len(opt_recs[k]), n_engrad_ok=int(len(Dk)),
                           n_selected=int(Dk.selected.sum()) if len(Dk) else 0,
                           fmax_norm_eh_bohr=_q(Dk.fmax_norm_eh_bohr) if len(Dk) else dict(n=0),
                           mlts_geometry_vs_label=geo)
    gpu_s = sum(float(o.get("gpu_s") or 0.0) for v in opt_recs.values() for o in v)
    wall = sum(float(o.get("wall_s") or 0.0) for v in opt_recs.values() for o in v)
    stopf = GD.STOP_FILE.read_text().strip() if GD.STOP_FILE.is_file() else None
    complete = n_todo == 0 and stopf is None
    pl = _rj(G4_PLAN_JSON) or {}
    rep = dict(phase="G-4 collect", spec_section="3 Phase G (G-4)", code=CODE_VERSION, written=_now(),
               host=socket.gethostname(), slurm_job=os.environ.get("SLURM_JOB_ID"), complete=complete, n_todo=n_todo,
               stop=stopf, stop_file=stopf, est=pl.get("est"), plan=str(PLAN), models=plan["models"],
               start=plan["start"], n_planned=len(T), n_resolved=resolved, counts=dict(st),
               n_engrad_ok=int(st.get("engrad_ok", 0)), n_selected=int(st.get("selected", 0)),
               n_selected_d1=int(st.get("selected_d1", 0)), n_selected_d0=int(st.get("selected_d0", 0)),
               n_planned_d0=sum(1 for r in T.values() if is_d0(r)),
               n_selected_if_component=int(sum(x > FMAX_SELECT for x in fc)), fmax_select_eh_bohr=FMAX_SELECT,
               fmax_norm_eh_bohr=_q(fn), fmax_comp_eh_bohr=_q(fc),
               core_h_measured=dict(total=core, engrad=core, n_engrad_attempts=n_att,
                                    predicted=(pl.get("predicted_core_h") or {}).get("total"),
                                    opt_gpu_h=gpu_s / 3600.0, opt_rxn_wall_h=wall / 3600.0,
                                    note="total = label-level EnGrad core-h (counts against the G budget); MLTS "
                                         "optimisation GPU / wall reported separately (G4-cost)"),
               by_model=by_model, extxyz=str(EXT_G4), roundtrip=rt, decisions=DECISIONS)
    if len(D):
        M.write_csv(G4 / "g4_structures.csv", D)
        rep["per_structure_csv"] = str(G4 / "g4_structures.csv")
    M.write_json(G4_REPORT, rep)
    print(json.dumps(dict(complete=complete, n_todo=n_todo, counts=dict(st), core_h=round(core, 2),
                          n_selected=rep["n_selected"], roundtrip=(rt or {}).get("pass_")), default=str), flush=True)
    if stopf:
        return 3
    return 0 if complete else 2


def cmd_status():
    P = _rj(PLAN)
    if not P:
        print("plan: missing")
        return 0
    n = len(P["rxns"])
    n0 = sum(1 for r in P["rxns"] if is_d0(r))
    no = sum(row_opt_rec(P, r).is_file() for r in P["rxns"])
    ne = sum(row_engrad_rec(r).is_file() for r in P["rxns"])
    R = _rj(G4_REPORT) or {}
    print(f"plan {n} reactions ({n0} D0; {P['start']} start, models {P['models']}); opt records {no}/{n}; EnGrad records "
          f"{ne}; report complete={R.get('complete')} n_selected={R.get('n_selected')}")
    return 0


# ================================================================ main
def main(argv=None):
    ap = argparse.ArgumentParser(description="DEPLOY-V2 (MLTS) Phase G-4 active learning")
    sub = ap.add_subparsers(dest="cmd")
    a = sub.add_parser("g4_plan")
    a.add_argument("--models", required=True)
    a.add_argument("--n-per-model", type=int, default=N_PER_MODEL)
    a.add_argument("--n-d0-per-model", type=int, default=N_D0_PER_MODEL)
    a.add_argument("--start", default="g2")
    a.add_argument("--max-atoms", type=int, default=None)
    a.add_argument("--allow-unchecked", action="store_true")
    a.add_argument("--force", action="store_true")
    a = sub.add_parser("g4_opt")
    a.add_argument("i")
    a.add_argument("n", type=int)
    a.add_argument("--models", default=None)
    a.add_argument("--workers", type=int, default=None)
    a = sub.add_parser("run")
    a.add_argument("group")
    a.add_argument("i")
    a.add_argument("n", type=int)
    a.add_argument("--only", default=None)
    a = sub.add_parser("collect")
    a.add_argument("--g4", action="store_true")
    sub.add_parser("status")
    try:
        x = ap.parse_args(argv)
    except SystemExit:
        return 2
    if x.cmd == "g4_plan":
        return cmd_plan(x.models, x.n_per_model, x.start, x.max_atoms, x.allow_unchecked, x.force, x.n_d0_per_model)
    if x.cmd == "g4_opt":
        return cmd_opt(M.array_task(x.i), x.n, x.models, x.workers)
    if x.cmd == "run":
        only = {s.strip() for s in str(x.only).split(",") if s.strip()} if x.only else None
        return cmd_run(x.group, M.array_task(x.i) if not only else 0, x.n, only)
    if x.cmd == "collect":
        return cmd_collect()
    if x.cmd == "status":
        return cmd_status()
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
