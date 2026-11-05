#!/usr/bin/env python3
r"""manual_chain.py — DEPLOY-V2 (MLTS) steps 2-5 of the user's consolidated instruction (11-05 20:5x) OUTSIDE the frozen
launcher: hand sbatch (arrays) only. pipeline.py is neither imported nor changed; the existing stage code is reused as
is (ts_geom.py, inputs_mlts.py, predict_c1.py, prereg_v2.py, train_k.py, tests.py, eval_v2.py); only `assemble`,
`gaps` and `ft2_report` below are new code. Compute nodes only (jobs/manual_chain.slurm).

Roots
  $MLT0 = /gpfs/tmp_cpu2/yeseo1ee/mlts (the production scratch; models, lists, FT-2 out dirs)
  step 2   ts_geom out dirs  $MLT0/ft2_full/<model>/<label|g2>  (fold -1 anchors: <model>/g2_fm1)
  steps 3-5 run with MC_ROOT=full: jobs/manual_chain.slurm exports MLT_SCRATCH=$MLT0/full and
           MLT_RESULTS=analysis/mlts/results/full, so every existing script (mlt_common paths) reads and writes the
           deploy root $FULL = $MLT0/full instead of the launcher's $MLT0 / results: A-2 folders $FULL/ade/jobs/<job_id>,
           inputs $FULL/inputs, tokens $FULL/fu2_mlts, HF $FULL/hf/mlts, K units $FULL/units, predictions $FULL/preds,
           reports analysis/mlts/results/full/. Nothing of the launcher's state ($MLT0/ade, results/*.json) is touched.

CLI (python manual_chain.py ..., exit 0 done, 2 not ready (same command again later), 3 STOP)
  cmds <step> [--fold k] [--model M] [--n N] [--starts label,g2]
        print the exact sbatch lines of a step (also written to $FULL/cmds/<step>[_f<k>].sh; stdlib only, any env).
        Dependencies are written as afterok:<JID_name> placeholders: submit in order and paste the job ids.
        steps: checks | ft2 | asm | h3a | h3b | ft2rep | prereg | c1 | kplan | k | select | e | d0 (step_* below)
  assemble <k> <model> [--accept converged|any] [--src DIR] [--list CSV]
        [reactot] deploy TS set of fold k (k = -1: the anchor-only reactions): the g2-start (b) MLIP TS records
        <src>/recs/<rxn_id>.json -> $FULL/ade/jobs/<job_id>/ in the A-2 layout read by inputs_mlts.py (decision MC-asm
        below), $FULL/ids/f<k>.csv, $FULL/pipeline/arms.json (engine 'Ap' = G2 TS -> MLTS OptTS, no A'' arm),
        results/full/ASM_f<k>.json. Exit 2 while < 99.5 % of the list has a record.
  gaps  [reactot] counts for the gap list (no G2 start per fold, fold -1 anchors and their pairs, coverage, D0, the
        launcher-state files steps 3-5 read) -> results/full/GAPS.json + $FULL/lists/fm1.csv (the anchor list).
  ft2_report [--models FT_{k}_full]
        [reactot] FT-2 (a) label start / (b) g2 start re-measured per fold (report only, no gate) + the structure
        criterion |dd| <= 0.010 A (report only) -> results/full/FT2_full_report.json.

Decisions (recommended options; for results/DECISIONS.md when the lead records them)
  MC-asm   A-2 folder of one reaction from its g2-start ts_geom record: ts.xyz = the record's pos_A (label atom order,
           symbols of the G2 start file $G2/g2_final/<rxn>/ts.xyz); ref_dipole.xyz / ref_dipolarophile.xyz = byte
           copies of the G2 xTB references $G2/g2_final/<rxn>/rel1.xyz / rel2.xyz (label order; no DFT, the lf deploy
           convention: references are not re-optimised with the MLIP - existing code has no MLIP reactant optimiser
           outside mlts_autode); A_idx = the label partition (input_meta_d1_g2.csv, a property of the reaction graph,
           = the H-1 label source); formed_pairs_ts = the record's label forming pairs; charges = the label record
           (SMILES charges); checks flag_async / no_foreign_bond and alt_used = the G2 result.json (xTB level, the
           references' own choice); imag_freqs_cm = the MLIP Hessian of the record; markers .done_ts/.done_stereo/
           .done_ref. A reaction without a deploy TS gets a .fail_ts folder ('no_g2_ts', 'not_converged',
           'not_covered', 'no_geometry', 'g2_not_done') so eval_v2 geom and finalize_lf count it.
  MC-accept deploy TS = converged records with a geometry (default --accept converged); n_imag != 1 is kept and
           counted (finalize_lf drops a TS whose xTB Freq has no imaginary mode). --accept any also takes the last
           geometry of an unconverged run.
  MC-root  one deploy root for all folds; per-fold early work (h3a: finalize, meta, hf_inputs, hf_run, all
           idempotent per reaction) may run as folds finish; h3b (x182 slices, atoms, tokens, hf_build, d3, build,
           h2) runs ONCE after every fold and the anchors are assembled (feat_lf slices are tied to their selection).
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import shlex  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import mlt_common as M  # noqa: E402            (stdlib only at import)

CODE_VERSION = "mlts-mc-1"
MLT0 = Path(os.environ.get("MC_BASE", "/gpfs/tmp_cpu2/yeseo1ee/mlts"))   # MC_BASE: a smoke root below $MLT/tmp
FULL = MLT0 / "full"
RESF = HERE / "results" / "full" if "MC_BASE" not in os.environ else FULL / "reports"
FT2_FULL = MLT0 / "ft2_full"
LISTS = MLT0 / "pipeline" / "lists"
SLURM = HERE / "jobs" / "manual_chain.slurm"
PART = dict(C="cpu1,cpu2", G="gpu1,gpu2,gpu3,gpu4,gpu5,gpu6")
NOT_ATTEMPTED = ("no_start", "no_label", "not_covered")
COVER_MIN = 0.995                       # = pipeline GEOM_PARTIAL / ts_geom X_PARTIAL
DD_STRUCT_A = 0.010                     # user 11-05 20:5x step 5: structure criterion, report only
ARMS8 = ("S", "SH")
ATOM_MODELS = ("AW", "GNN", "TF")
K_CONFIGS = {"C2": "mlts", "C2_s1": "mlts", "C2_s2": "mlts", "C3": "mlts_mix", "C1v": "dft"}
K_SUB = {"C2": "c2", "C2_s1": "c2", "C2_s2": "c2", "C3": "c3", "C1v": "c1v"}


def log(msg):
    print(f"[MC {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def list_path(k):
    return FULL / "lists" / "fm1.csv" if int(k) == -1 else LISTS / f"f{int(k)}_test.csv"


def src_dir(k, model, start="g2"):
    return FT2_FULL / model / (f"{start}_fm1" if int(k) == -1 else start)


def default_model(k):
    """FT_<k>_full for fold k; fold -1 (anchors, training side of every FT_k_full): FT_0_full = the A-4 rule
    (pipeline start_a4: fold -1 uses the fold-0 model)."""
    return "FT_0_full" if int(k) == -1 else f"FT_{int(k)}_full"


# ================================================================ cmds
def sb(name, args, part="C", c=4, mem=16, gpu=False, array=None, deps=(), env="reactot", full=True):
    """one sbatch line (jobs/manual_chain.slurm; -J mc_<name>; afterok placeholders <JID_x>)."""
    a = ["sbatch", "-J", f"mc_{name}", "-p", PART[part], "-c", str(c), f"--mem={mem}G"]
    if gpu:
        a.append("--gres=gpu:1")
    if array:
        a.append(f"--array={array}")
    if deps:
        a.append("--dependency=afterok:" + ":".join(f"<JID_{d}>" for d in deps))
    exp = f"ALL,MLTENV={env}" + (",MC_ROOT=full" if full else "")
    a += [f"--export={exp}", str(SLURM)]
    return dict(name=name, tasks=_ntasks(array), line=" ".join(shlex.quote(x) if any(ch in x for ch in "*? ;&|<>")
                                                               else x for x in a + [str(x) for x in args]))


def _ntasks(array):
    if not array:
        return 1
    spec = str(array).split("%")[0]
    n = 0
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            n += int(hi) - int(lo) + 1
        else:
            n += 1
    return n


def chain(*segs):
    """mlt.slurm segments joined with ' ::: ' (next only if exit 0)."""
    out = []
    for s in segs:
        if out:
            out.append(":::")
        out += list(s)
    return out


def im(*a):
    return ["inputs_mlts.py", *a]


def step_checks(o):
    """H-1 (= C-5) and C-6, once each, in the deploy root (fresh: every number recomputed)."""
    h1 = chain(im("label_src", "50"), im("finalize", "label", "0", "1"), im("meta", "label"),
               im("x182", "label", "0", "1"), im("x182_merge", "label"), im("atoms", "label", "0", "1"),
               im("tokens", "label"), im("hf_inputs", "label"), im("hf_run", "label", "0", "1"),
               im("hf_build", "label"), im("d3", "label"), im("build", "label"), im("h1"))
    return [sb("h1", h1, c=32, mem=64),
            sb("c6_atoms", ["predict_c1.py", "atoms", "dft"], part="G", c=8, mem=64, gpu=True),
            sb("c6_xgb", ["predict_c1.py", "xgb", "dft"], c=32, mem=64),
            sb("c6", ["predict_c1.py", "c6"], c=4, mem=32, deps=("c6_atoms", "c6_xgb"))]


def step_ft2(o):
    k, model = o.fold, o.model or default_model(o.fold)
    n = o.n or (1 if k == -1 else 4)
    starts = o.starts.split(",") if o.starts else (["g2"] if k == -1 else ["label", "g2"])
    out = []
    for s in starts:
        od = src_dir(k, model, s)
        out.append(sb(f"ft2_{model}_{s}" + ("_fm1" if k == -1 else ""),
                      ["ts_geom.py", "run", model, str(list_path(k)), s, str(od), "@I", str(n)], part="G", c=16,
                      mem=64, gpu=True, array=f"0-{n - 1}", env="mlts", full=False))
    return out


def step_asm(o):
    k, model = o.fold, o.model or default_model(o.fold)
    dep = f"ft2_{model}_g2" + ("_fm1" if k == -1 else "")
    r = sb(f"asm_f{k}", ["manual_chain.py", "assemble", str(k), model], c=4, mem=16, deps=(dep,), full=False)
    r["line"] = r["line"].replace("--dependency=afterok:", "--dependency=afterany:")   # assemble checks coverage itself
    return [r]


def step_h3a(o):
    k = o.fold
    seg = chain(im("finalize", "mlts", "0", "1"), im("meta", "mlts"), im("hf_inputs", "mlts"),
                im("hf_run", "mlts", "0", "1"))
    return [sb(f"h3a_f{k}", seg, c=64, mem=240, deps=(f"asm_f{k}",))]


def step_h3b(o):
    seg = chain(im("finalize", "mlts", "0", "1"), im("meta", "mlts"), im("x182", "mlts", "0", "1"),
                im("x182_merge", "mlts"), im("atoms", "mlts", "0", "1"), im("tokens", "mlts"),
                im("hf_inputs", "mlts"), im("hf_run", "mlts", "0", "1"), im("hf_build", "mlts"), im("d3", "mlts"),
                im("build", "mlts"))
    # build exits 3 on H-2 > 2 % AFTER writing the table: ' ;;; ' still builds the min table and H2_report (the STOP is
    # the lead's decision, the job exit code stays 3)
    seg += [";;;"] + im("build", "mlts", "--refs", "min") + [";;;"] + im("h2", "mlts")
    return [sb("h3b", seg, c=64, mem=240)]


def step_prereg(o):
    return [sb("prereg", ["prereg_v2.py", "write", "--allow-partial", "--from-results", str(HERE / "results")],
               c=2, mem=8)]


def step_c1(o):
    out = []
    for tag, x in (("", []), ("_2nd", ["--refs", o.refs2 or "own"])):
        out.append(sb(f"c1_atoms{tag}", ["train_k.py", "c1pred", "atoms"] + x, part="G", c=8, mem=64, gpu=True))
        out.append(sb(f"c1_xgb{tag}", ["train_k.py", "c1pred", "xgb"] + x, c=32, mem=64))
        out.append(sb(f"c1_collect{tag}", ["train_k.py", "c1pred", "collect"] + x, c=4, mem=32,
                      deps=(f"c1_atoms{tag}", f"c1_xgb{tag}")))
    return out


def step_kplan(o):
    return [sb("kplan", ["train_k.py", "plan"], c=2, mem=8)]


def step_k(o):
    """one array of 5 folds per (config, arm, model): train_k.py <sub> '<config>/L__<rung>__<arm>__<model>__f*' @I 5
    (K-8 resources: atoms 1 GPU + 8 CPU, 24 GB (C3 / C1v 48 GB); XGB 32 CPU / 64 GB (C3 96 GB, C1v 16 / 32))."""
    cfgs = (o.configs or "C2").split(",")
    out = []
    for cfg in cfgs:
        rung, sub = K_CONFIGS[cfg], K_SUB[cfg]
        for arm in ARMS8:
            for model in ATOM_MODELS + ("XGB",):
                if cfg in ("C2_s1", "C2_s2") and model == "XGB":
                    continue                                        # K-3: XGB seed free (copied from C2)
                pat = f"{cfg}/L__{rung}__{arm}__{model}__f*"
                if model == "XGB":
                    c, mem = (16, 32) if cfg == "C1v" else (32, 96 if cfg == "C3" else 64)
                    out.append(sb(f"k_{cfg}_{arm}_{model}", ["train_k.py", sub, pat, "@I", "5"], c=c, mem=mem,
                                  array="0-4"))
                else:
                    out.append(sb(f"k_{cfg}_{arm}_{model}", ["train_k.py", sub, pat, "@I", "5"], part="G", c=8,
                                  mem=48 if cfg in ("C3", "C1v") else 24, gpu=True, array="0-4"))
    return out


def step_select(o):
    return [sb("select", ["train_k.py", "select"], c=4, mem=16),
            sb("c2k", ["tests.py", "c2_k"], c=4, mem=32, deps=("select",))]


def step_e(o):
    return [sb("e", ["eval_v2.py", "run"], c=16, mem=64)]


def step_ft2rep(o):
    return [sb("ft2rep", ["manual_chain.py", "ft2_report"], c=4, mem=16, full=False)]


def step_d0(o):
    """optional, report only: the existing D0 evaluation (ts_geom d0_eval) on the FT_all_full model(s) (--model a
    comma list; the default models of ts_geom are FT_all_s*_al, so they must be passed)."""
    models = o.model or "FT_all_full"
    n = o.n or 2
    root = MLT0 / "d0eval_full"
    return [sb("d0_run", ["ts_geom.py", "d0_eval", "run", "@I", str(n), "--models", models, "--root", str(root)],
               part="G", c=16, mem=64, gpu=True, array=f"0-{n - 1}", env="mlts", full=False),
            sb("d0_report", ["ts_geom.py", "d0_eval", "report", "--models", models, "--root", str(root), "--out",
                             str(RESF / "D0_eval_full_report.json")], c=4, mem=16, deps=("d0_run",), env="mlts",
               full=False)]


STEPS = dict(checks=step_checks, ft2=step_ft2, asm=step_asm, h3a=step_h3a, h3b=step_h3b, ft2rep=step_ft2rep,
             prereg=step_prereg, c1=step_c1, kplan=step_kplan, k=step_k, select=step_select, e=step_e, d0=step_d0)


def cmd_cmds(o):
    if o.step not in STEPS:
        print(f"unknown step {o.step!r}; steps: {' | '.join(STEPS)}", file=sys.stderr)
        return 2
    if o.step in ("ft2", "asm", "h3a") and o.fold is None:
        print(f"step {o.step} needs --fold k (k = 0..4, -1 = anchors)", file=sys.stderr)
        return 2
    rows = STEPS[o.step](o)
    head = [f"# manual_chain.py cmds {o.step}" + (f" --fold {o.fold}" if o.fold is not None else "")
            + (f" --model {o.model}" if o.model else "") + f"  ({now()}, {CODE_VERSION})",
            f"# {sum(r['tasks'] for r in rows)} tasks; submit in order (each one-shot sbatch); replace <JID_x> by the "
            "job id of line mc_x; check the queue first (<= 20 tasks account-wide, array elements count)"]
    txt = "\n".join(head + [r["line"] for r in rows]) + "\n"
    print(txt, end="")
    try:
        name = o.step + (f"_f{o.fold}" if o.fold is not None else "") + (f"_{o.configs}" if o.configs else "")
        M.assert_write_ok(FULL / "cmds" / "x")
        M.write_text(FULL / "cmds" / f"{name.replace(',', '+')}.sh", txt)
    except SystemExit:
        pass                                                         # read-only use (MC_ROOT unset elsewhere)
    return 0


# ================================================================ assemble (reactot)
def _rj(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def _write_xyz(path, syms, pos, comment):
    lines = [str(len(syms)), comment] + [f"{s:<3s}{x:16.8f}{y:16.8f}{z:16.8f}" for s, (x, y, z) in zip(syms, pos)]
    Path(path).write_text("\n".join(lines) + "\n")


def _publish(out, tmp):
    if out.exists():
        shutil.rmtree(out)
    os.replace(tmp, out)


def _no_final(job, src):
    """an A-2 folder is (re)written only while finalize has not consumed it: final/<rxn_id> (.done or .fail_*) is
    never refreshed by finalize_lf (it skips a marked folder), so a changed source after it = STOP, not silence."""
    f = FULL / "final" / str(M.rxn_of(job))
    if f.is_dir():
        M.stop(f"{job}: A-2 folder would change ({src}) but {f} already exists; move that final folder away first")


def _fail_folder(root, job, reason, detail, src):
    """an A-2 folder that records why the reaction has no deploy TS (.fail_ts 'reason: detail')."""
    out = root / job
    if out.is_dir() and (out / ".fail_ts").is_file() and (out / ".fail_ts").read_text().split(":")[0].strip() == reason:
        return "exists"
    _no_final(job, src)
    tmp = root / f".{job}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    (tmp / ".fail_ts").write_text(f"{reason}: {detail}\n")
    (tmp / "ts_result.json").write_text(json.dumps(dict(job_id=job, status="fail", reason=reason, detail=detail,
                                                        source=str(src), writer=CODE_VERSION), indent=1))
    _publish(out, tmp)
    return "made"


def cmd_assemble(k, model, accept="converged", src=None, lst=None):
    import inputs_mlts as IM
    IM.ensure_env("reactot")
    import hashlib
    import numpy as np
    import pandas as pd
    k = int(k)
    src = Path(src) if src else src_dir(k, model, "g2")
    lst = Path(lst) if lst else list_path(k)
    if accept not in ("converged", "any"):
        M.stop(f"--accept {accept!r}: converged | any")
    if not lst.is_file():
        M.not_ready(f"{lst} missing" + (" (run manual_chain.py gaps first)" if k == -1 else ""))
    rx = [int(x) for x in pd.read_csv(lst).rxn_id]
    recd = src / "recs"
    have = {int(p.stem) for p in recd.glob("*.json")} if recd.is_dir() else set()
    missing = [r for r in rx if r not in have]
    if len(rx) and (len(rx) - len(missing)) / len(rx) < COVER_MIN:
        M.not_ready(f"{src}: {len(rx) - len(missing)} / {len(rx)} records (< {COVER_MIN:.1%}); run / resubmit ts_geom")
    T = M.targets().set_index("rxn_id")
    lab = {int(d["rxn_id"]): d for d in M.read_json(M.LABELS)}
    A_lab = IM.label_A()
    root = FULL / "ade" / "jobs"
    M.assert_write_ok(root / "x")
    root.mkdir(parents=True, exist_ok=True)
    cnt, made, rows = Counter(), Counter(), []
    nimag, dd_ok, conv_n = Counter(), Counter(), 0
    for rid in rx:
        job = str(T.at[rid, "job_id"])
        if int(T.at[rid, "fold_T1"]) != k:
            M.stop(f"{rid} in {lst} has fold_T1 {T.at[rid, 'fold_T1']} != {k}")
        p = recd / f"{rid}.json"
        if rid in missing:
            cnt["no_record"] += 1
            rows.append(dict(rxn_id=rid, job_id=job, deploy="no", reason="no_record"))
            continue
        raw = p.read_bytes()
        r = json.loads(raw)
        sha = hashlib.sha256(raw).hexdigest()
        st = str(r.get("status"))
        g2d = M.G2_FINAL / str(rid)
        reason = None
        if st == "no_start":
            reason = "no_g2_ts"
        elif st in ("not_covered", "no_label"):
            reason = st
        elif not (g2d / ".done").is_file():
            reason = "g2_not_done"
        elif r.get("pos_A") is None:
            reason = "no_geometry"
        elif r.get("converged") is not True and accept == "converged":
            reason = "not_converged"
        if reason:
            cnt[reason] += 1
            made[_fail_folder(root, job, reason, f"{st} ({p})", p)] += 1
            rows.append(dict(rxn_id=rid, job_id=job, deploy="no", reason=reason))
            continue
        out = root / job
        old = _rj(out / "ts_result.json") or {}
        if (out / ".done_ref").is_file() and old.get("source_sha256") == sha:
            made["exists"] += 1
        else:
            _no_final(job, p)
            ss, _ = M.read_xyz(g2d / "ts.xyz")
            pos = np.asarray(r["pos_A"], dtype=float)
            if len(ss) != len(pos) or pos.shape[1:] != (3,):
                M.stop(f"{rid}: record pos_A {pos.shape} vs start {len(ss)} atoms")
            A = sorted(A_lab[rid])
            B = sorted(set(range(len(ss))) - set(A))
            pairs = sorted(tuple(sorted(int(x) for x in q)) for q in (r.get("pairs") or []))
            lp = sorted(tuple(sorted(q)) for q in M.formed_pairs(lab[rid]["formed_pairs_ts"]))
            if pairs != lp:
                M.stop(f"{rid}: record forming pairs {pairs} != label record {lp}")
            r1, r2 = M.read_xyz(g2d / "rel1.xyz"), M.read_xyz(g2d / "rel2.xyz")
            if sorted(ss[i] for i in A) != sorted(r1[0]) or sorted(ss[i] for i in B) != sorted(r2[0]):
                cnt["ref_composition"] += 1
                made[_fail_folder(root, job, "ref_composition", "G2 references differ from the TS fragments", p)] += 1
                rows.append(dict(rxn_id=rid, job_id=job, deploy="no", reason="ref_composition"))
                continue
            g2r = _rj(g2d / "result.json") or {}
            ck = ((g2r.get("ts_result") or {}).get("checks") or {})
            alt = (g2r.get("ref_result") or {}).get("alt_used")
            if "flag_async" not in ck or "no_foreign_bond" not in ck or alt is None:
                M.stop(f"{g2d}/result.json lacks ts_result.checks flag_async / no_foreign_bond or ref_result.alt_used")
            tmp = root / f".{job}.tmp"
            if tmp.exists():
                shutil.rmtree(tmp)
            tmp.mkdir(parents=True)
            _write_xyz(tmp / "ts.xyz", ss, pos.tolist(), f"MLIP TS {model} g2 start rxn {rid} (label order)")
            back = np.asarray(M.read_xyz(tmp / "ts.xyz")[1], dtype=float)
            if float(np.abs(back - pos).max()) > 1e-6:
                M.stop(f"{rid}: ts.xyz round trip {float(np.abs(back - pos).max()):.2e} A")
            shutil.copyfile(g2d / "rel1.xyz", tmp / "ref_dipole.xyz")
            shutil.copyfile(g2d / "rel2.xyz", tmp / "ref_dipolarophile.xyz")
            tsr = dict(job_id=job, source=str(p), source_sha256=sha, model=r.get("model"), start="g2",
                       writer=CODE_VERSION, A_idx=A, B_idx=B, formed_pairs_ts=[list(q) for q in lp],
                       charge_dipole=int(lab[rid]["charge1"]), charge_dipolarophile=int(lab[rid]["charge2"]),
                       checks=dict(flag_async=bool(ck["flag_async"]), no_foreign_bond=bool(ck["no_foreign_bond"]),
                                   source="G2 result.json ts_result.checks (xTB TS)"),
                       imag_freqs_cm=[float(x) for x in (r.get("imag_cm") or [])], n_imag_mlip=r.get("n_imag"),
                       converged=r.get("converged"), n_cycles=r.get("n_cycles"), accept=accept)
            (tmp / "ts_result.json").write_text(json.dumps(tsr, indent=1))
            (tmp / "ref_result.json").write_text(json.dumps(dict(job_id=job, alt_used=int(alt),
                                                                 source="G2 result.json ref_result.alt_used (G2 xTB "
                                                                        "references rel1 / rel2)"), indent=1))
            (tmp / "stereo_result.json").write_text(json.dumps(dict(
                job_id=job, status=str((g2r.get("stereo_result") or {}).get("status") or "g2"),
                source="G2 result.json stereo_result"), indent=1))
            (tmp / "timing.json").write_text(json.dumps(dict(job_id=job, wall_s=r.get("wall_s"), gpu_s=r.get("gpu_s"),
                                                             n_ext_calls=r.get("n_ext_calls"),
                                                             n_hess=r.get("n_hess_calls"), source="ts_geom record"),
                                                        indent=1))
            for s in ("ts", "stereo", "ref"):
                (tmp / f".done_{s}").write_text(now() + "\n")
            _publish(out, tmp)
            made["made"] += 1
        cnt["deploy_ts"] += 1
        conv_n += int(r.get("converged") is True)
        ni = r.get("n_imag")
        nimag[str(int(ni)) if isinstance(ni, (int, float)) else "none"] += 1
        dd = r.get("dd_max")
        dd_ok["le_0.010" if dd is not None and dd <= DD_STRUCT_A else "gt_0.010"] += 1
        rows.append(dict(rxn_id=rid, job_id=job, deploy="yes", reason=""))
    R = pd.DataFrame(rows)
    M.write_csv(FULL / "ids" / f"f{k}.csv", R[R.deploy == "yes"][["rxn_id", "job_id"]])
    M.write_csv(FULL / "ids" / f"f{k}_all.csv", R)
    arms = FULL / "pipeline" / "arms.json"
    if not arms.is_file():
        M.write_json(arms, dict(engine="Ap", extra_arm_App=False, app_root=None, app_usable=None, written=now(),
                                writer=f"manual_chain.py {CODE_VERSION}",
                                note="deploy root $MLT/full: engine = G2 TS -> MLTS OptTS (ts_geom g2 start, FT_k_full); "
                                     "A'' dropped (user 11-05 20:5x)"))
    rep = dict(step="assemble", fold=k, model=model, src=str(src), list=str(lst), accept=accept, n_list=len(rx),
               n_records=len(rx) - len(missing), counts=dict(cnt), folders=dict(made), n_deploy_ts=cnt["deploy_ts"],
               n_converged_in_deploy=conv_n, n_imag_mlip=dict(nimag),
               structure_dd_le_0010_A=dict(dd_ok, note="report only (user step 5); |dd| vs the label TS"),
               n_missing_records=len(missing), missing_records=missing[:50],
               complete=True, coverage_rule=f">= {COVER_MIN:.1%} of the list recorded (= pipeline GEOM_PARTIAL); a "
                                            "missing record (ts_geom pipeline error) gets no A-2 folder",
               written=now(), code=CODE_VERSION, decisions=__doc__.split("Decisions")[1][:2400])
    M.write_json(RESF / f"ASM_f{k}.json", rep)
    log(f"assemble f{k} {model}: {dict(cnt)} folders {dict(made)} -> {root}")
    return 0


# ================================================================ gaps (reactot)
def cmd_gaps():
    import inputs_mlts as IM
    IM.ensure_env("reactot")
    import pandas as pd
    T = M.targets()
    g2ok = {int(r) for r in T.rxn_id if (M.G2_FINAL / str(int(r)) / ".done").is_file()}
    G = pd.read_csv(M.ANALYSIS / "d1_g2" / "results" / "G_status.csv", usecols=["rxn_id", "status"])
    gst = dict(zip(G.rxn_id.astype(int), G.status.astype(str)))
    out = dict(written=now(), code=CODE_VERSION, n_targets=int(len(T)), n_g2_done=len(g2ok),
               n_g2_status_ok=int(sum(v == "ok" for v in gst.values())), folds={})
    n_lists = 0
    for k in range(5):
        p = list_path(k)
        L = [int(x) for x in pd.read_csv(p).rxn_id] if p.is_file() else []
        n_lists += len(L)
        tk = set(int(x) for x in T.rxn_id[T.fold_T1 == k])
        out["folds"][str(k)] = dict(n_list=len(L), list_equals_fold=set(L) == tk, n_g2_start=sum(r in g2ok for r in L),
                                    n_no_g2_start=sum(r not in g2ok for r in L),
                                    n_no_g2_by_dipole=dict(Counter(T.set_index("rxn_id").loc[[r for r in L if r not in g2ok],
                                                                                           "dipole"]).most_common()))
    am = T[T.fold_T1 == -1]
    out["anchors"] = dict(n=int(len(am)), n_g2_start=int(sum(int(r) in g2ok for r in am.rxn_id)),
                          n_cores=int(am.fold_core.nunique()))
    M.write_csv(FULL / "lists" / "fm1.csv", am[["rxn_id", "job_id", "dipole"]].sort_values("rxn_id"))
    out["coverage"] = dict(n_in_fold_lists=n_lists, n_anchor=int(len(am)), n_total=n_lists + int(len(am)),
                           deploy_ts_max_folds=sum(v["n_g2_start"] for v in out["folds"].values()),
                           deploy_ts_max_with_anchors=sum(v["n_g2_start"] for v in out["folds"].values())
                           + out["anchors"]["n_g2_start"])
    S = pd.read_csv(M.S_FOLDS)
    ok = lambda a, b: (int(a) in g2ok) and (int(b) in g2ok)                       # noqa: E731
    S["both_g2"] = [ok(a, b) for a, b in zip(S.a_rxn, S.b_rxn)]
    fold_of = dict(zip(T.rxn_id.astype(int), T.fold_T1.astype(int)))
    S["anchor_end"] = [fold_of.get(int(a)) == -1 or fold_of.get(int(b)) == -1 for a, b in zip(S.a_rxn, S.b_rxn)]
    S["mixed_fold"] = [fold_of.get(int(a)) != fold_of.get(int(b)) for a, b in zip(S.a_rxn, S.b_rxn)]
    tm = S[S.in_T_main]
    out["pairs"] = dict(n=int(len(S)), n_T_main=int(len(tm)), n_T_main_fold_m1=int((tm.fold_T1 == -1).sum()),
                        n_T_main_with_anchor_end=int(tm.anchor_end.sum()), n_pairs_mixed_fold_ends=int(S.mixed_fold.sum()),
                        test_P_main_para_by_fold={}, d0_overlap_pairs=int(S.d0_overlap.astype(str).str.lower()
                                                                          .eq("true").sum()))
    out["pairs"]["ref_kind_counts"] = dict(Counter(S.ref_kind.astype(str)))
    out["pairs"]["pos_counts"] = dict(Counter(S.pos.astype(str)))
    for k in range(5):
        x = S[(S.fold_T1 == k) & S.in_P_main & (S.pos.astype(str) == "p")]
        out["pairs"]["test_P_main_para_by_fold"][str(k)] = dict(n=int(len(x)), n_both_g2=int(x.both_g2.sum()),
                                                                n_with_anchor_end=int(x.anchor_end.sum()))
    D0 = pd.read_csv(HERE / "results" / "d0_holdout.csv")
    out["d0"] = dict(n_holdout=int(len(D0)), roles=dict(Counter(D0.role.astype(str))),
                     note="D0 = Coley reactions outside D1 (ids d0_id), held out of MLIP training; not in the 9,266")
    files = {"$MLT/ade/jobs (A-4 folders)": MLT0 / "ade" / "jobs", "$MLT/pipeline/arms.json": MLT0 / "pipeline" / "arms.json",
             "results/PREREG_V2.md": HERE / "results" / "PREREG_V2.md", "results/C_tests.json": HERE / "results" / "C_tests.json",
             "results/A4_report.json": HERE / "results" / "A4_report.json",
             "results/K_plan.json": HERE / "results" / "K_plan.json", "$MLT/full": FULL}
    st = {}
    for k_, p in files.items():
        st[k_] = (sum(1 for _ in os.scandir(p)) if p.is_dir() else "file") if p.exists() else "absent"
    out["launcher_state_files"] = st
    mods = sorted(p.stem for p in (MLT0 / "models").glob("*.json")) if (MLT0 / "models").is_dir() else []
    out["models_full"] = [m for m in mods if "full" in m or "LODO" in m]
    out["models_all"] = mods[:60]
    out["ft2_full_dirs"] = sorted(f"{a.name}/{b.name}:{sum(1 for _ in (b / 'recs').glob('*.json'))}"
                                  for a in FT2_FULL.glob("*") if a.is_dir() for b in a.glob("*") if (b / "recs").is_dir()) \
        if FT2_FULL.is_dir() else []
    abl = {}
    for tag, d in (("baseline", MLT0 / "ft2" / "FTpilot_AIMNet2_4" / "label"),
                   ("fw1", MLT0 / "ft2" / "FTabl_AIMNet2_4_fw1" / "label"),
                   ("fw5", MLT0 / "ft2" / "FTabl_AIMNet2_4_fw5" / "label")):
        a = (_rj(d / "summary.json") or {}).get("all") or {}
        abl[tag] = {x: a.get(x) for x in ("n_attempted", "same_ts_frac", "dd_median_A", "dd_p90_A", "nimag1_frac")}
    out["ft_abl_first_pass_label"] = abl
    M.write_json(RESF / "GAPS.json", out)
    log(f"gaps -> {RESF / 'GAPS.json'}")
    return 0


# ================================================================ ft2_report (reactot)
def cmd_ft2_report(tmpl="FT_{k}_full"):
    import inputs_mlts as IM
    IM.ensure_env("reactot")
    import pandas as pd
    import ts_geom as TG
    out = dict(written=now(), code=CODE_VERSION, rule="report only (no gate); FT-2 (a) = label start, (b) = g2 start; "
               "structure criterion |dd| <= 0.010 A reported as a fraction (user 11-05 20:5x)", folds={}, pooled={})
    pool = {"label": [], "g2": []}
    for k in range(5):
        m = tmpl.format(k=k)
        e = dict(model=m)
        n_list = len(pd.read_csv(list_path(k))) if list_path(k).is_file() else None
        for s in ("label", "g2"):
            df = TG.load_records(src_dir(k, m, s))
            if df.empty:
                e[s] = dict(present=False)
                continue
            t = TG.table(df)
            att = df[~df.status.isin(list(NOT_ATTEMPTED))]
            t["dd_le_0010_frac_of_attempted"] = float((att.dd_max <= DD_STRUCT_A).sum() / len(att)) if len(att) else None
            t["coverage"] = len(df) / n_list if n_list else None
            t["fail_classes"] = TG.fail_counts(df)
            e[s] = t
            pool[s].append(df)
        out["folds"][str(k)] = e
    for s, fr in pool.items():
        if fr:
            df = pd.concat(fr, ignore_index=True)
            t = TG.table(df)
            att = df[~df.status.isin(list(NOT_ATTEMPTED))]
            t["dd_le_0010_frac_of_attempted"] = float((att.dd_max <= DD_STRUCT_A).sum() / len(att)) if len(att) else None
            out["pooled"][s] = t
    M.write_json(RESF / "FT2_full_report.json", out)
    log(f"ft2_report -> {RESF / 'FT2_full_report.json'}")
    return 0


# ================================================================ main
def main(argv=None):
    ap = argparse.ArgumentParser(prog="manual_chain.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("cmds")
    p.add_argument("step")
    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--starts", default=None)
    p.add_argument("--configs", default=None, help="step k: comma list of C2,C2_s1,C2_s2,C3,C1v (default C2)")
    p.add_argument("--refs2", default=None, help="step c1: the secondary refs mode (default own)")
    p = sub.add_parser("assemble")
    p.add_argument("fold", type=int)
    p.add_argument("model")
    p.add_argument("--accept", default="converged")
    p.add_argument("--src", default=None)
    p.add_argument("--list", default=None)
    sub.add_parser("gaps")
    p = sub.add_parser("ft2_report")
    p.add_argument("--models", default="FT_{k}_full")
    o = ap.parse_args(argv)
    if o.cmd == "cmds":
        return cmd_cmds(o)
    if o.cmd == "assemble":
        return cmd_assemble(o.fold, o.model, o.accept, o.src, o.list)
    if o.cmd == "gaps":
        return cmd_gaps()
    if o.cmd == "ft2_report":
        return cmd_ft2_report(o.models)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
