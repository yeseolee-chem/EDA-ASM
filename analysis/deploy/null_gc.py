#!/usr/bin/env python3
"""null_gc.py — DEPLOY-MODEL G-C null check (user decision 2026-11-04; results/DECISIONS.md 'G-C 영가설 검사 설계').

Is the pre-registered G-C gate passable by a model that differs from LADDER g2 . S . TF only by its random seed?
  python null_gc.py run <i|@I> <n>   GPU (env_d1fu2, jobs/gpu_fu2.slurm): folds [i::n] of the LADDER g2 inputs
                                     (DL.get_data("g2"), the C-2 / C-3 data with the LADDER usable rule) through the
                                     unchanged train_tf.run_unit (4 tuning cells + 5 final seeds), with ONLY the seeds
                                     moved: tuning seed SEED + 1000, final seeds SEED + 1000 + 0..4
                                     -> $DEP/null/units/D__g2__S__TF__f<k>.{json,parquet}, __s<j>.pt
  python null_gc.py gc               CPU (env espley, jobs/cpu_py.slurm): the 5 null unit predictions in place of 'lf' in
                                     eval_tf.evaluate against the LADDER g2 . S . TF rows -> results/NULL_gc.json
                                     (G-C of the null, side by side with the lf G-C of results/E_report.json, and the
                                     pre-registered reading). Exit 0 written, 2 = units missing, 3 = inconsistent.
No DEPLOY-MODEL file is modified; train_tf / fu2_common constants are only rebound in this process."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dep_common as C  # noqa: E402

SEED_OFFSET = 1000
NULL_ROOT = C.DEP / "null"
UNITS_N = NULL_ROOT / "units"
NF = 5


def uname(k):
    return f"D__g2__S__TF__f{k}"


def cmd_run(i, n):
    import data_lf as DL
    import train_tf as TT
    TT.FT.require_gpu()
    F = TT.F
    base = int(F.SEED)
    F.SEED = base + SEED_OFFSET                                   # tuning seed + gate seeds inside run_unit
    seeds = [F.SEED + j for j in range(len(F.FINAL_SEEDS))]
    F.FINAL_SEEDS = list(seeds)                                   # recorded in the unit fingerprint
    i, n = C.array_task(i), int(n)
    folds = list(range(NF))[i::n]
    D = DL.get_data("g2")
    print(f"[null] seeds tune {F.SEED} final {seeds} (base {base} + {SEED_OFFSET}), folds {folds}, "
          f"{TT.FT.versions()}", flush=True)
    for k in folds:
        u = dict(rung="g2", arm=TT.ARM, model=TT.MODEL, fold=k)
        rec = TT.run_unit(D, u, root=UNITS_N, seeds=seeds)
        print(f"[null] {uname(k)} chosen {rec['chosen']} (wall {rec.get('wall_seconds', 0):.0f} s)", flush=True)
    return 0


def cmd_gc():
    import pandas as pd
    import eval_tf as E
    names = [uname(k) for k in range(NF)]
    miss = [x for x in names if not ((UNITS_N / f"{x}.json").is_file() and (UNITS_N / f"{x}.parquet").is_file())]
    if miss:
        C.not_ready(f"null units missing: {miss}")
    recs, parts = {}, []
    for x in names:
        r = C.read_json(UNITS_N / f"{x}.json")
        if C.sha256(UNITS_N / f"{x}.parquet") != r.get("parquet_sha256"):
            C.stop(f"{x}.parquet changed after its json")
        fits = list(r.get("tune") or []) + list(r.get("final") or [])
        if not fits or not all(f["gate_M1"]["ok"] for f in fits) or r.get("smoke") or r.get("limit"):
            C.stop(f"{x}: gate C-4 failed or smoke / limited unit")
        seeds = (r.get("fingerprint") or {}).get("rules", {}).get("seeds")
        if seeds != [C.SEED + SEED_OFFSET + j for j in range(5)]:
            C.stop(f"{x}: final seeds {seeds} are not the null seeds")
        recs[x] = r
        parts.append(pd.read_parquet(UNITS_N / f"{x}.parquet"))
    null = pd.concat(parts, ignore_index=True)
    g2 = E.ladder_g2()
    R = E.evaluate(null, g2)
    gc_null = R["gc"]
    e_rep = C.read_json(C.RES / "E_report.json") if (C.RES / "E_report.json").is_file() else {}
    gc_lf = e_rep.get("G_C") or {}
    null_pass = bool(gc_null["pass"])
    reading = ("null passes G-C: the gate is passable by a seed-only replicate, so the lf failure is real -> STOP and "
               "report (DECISIONS 'G-C 영가설 검사 설계')") if null_pass else (
               "null fails G-C: a seed-only replicate of LADDER g2 . S . TF cannot pass the gate either, so the gate is "
               "underpowered for this comparison -> record the lf G-C failure as a deviation and proceed to W / I and "
               "DEPLOY_MODEL_W (DECISIONS 'G-C 영가설 검사 설계')")
    per = {}
    for t, v in gc_null["per_target"].items():
        lfv = (gc_lf.get("per_target") or {}).get(t, {})
        per[t] = dict(null_rel=v["rel"], null_rel_ci_hi=v["rel_ci_hi"], null_pass=v["pass"],
                      lf_rel=lfv.get("rel"), lf_rel_ci_hi=lfv.get("rel_ci_hi"), lf_pass=lfv.get("pass"))
    out = dict(check="G-C null", seeds=dict(tune=C.SEED + SEED_OFFSET, final=[C.SEED + SEED_OFFSET + j for j in range(5)]),
               null_gc=gc_null, lf_gc_pass=gc_lf.get("pass"), per_target=per, n_pairs=R["n"],
               null_chosen={x: r["chosen"] for x, r in recs.items()},
               ladder_chosen=None, null_pass=null_pass, reading=reading,
               proceed=not null_pass, units_root=str(UNITS_N),
               unit_json_sha256={x: C.sha256(UNITS_N / f"{x}.json") for x in names},
               gpu={x: (r.get("fingerprint") or {}).get("gpu") for x, r in recs.items()},
               gpu_hours=sum(float(r.get("wall_seconds") or 0) for r in recs.values()) / 3600)
    C.write_json(C.RES / "NULL_gc.json", out)
    print(json.dumps(dict(null_pass=null_pass, failed=gc_null["failed"], proceed=not null_pass), indent=1))
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["run"] and len(a) == 3:
        sys.exit(cmd_run(a[1], a[2]))
    if a == ["gc"]:
        sys.exit(cmd_gc())
    raise SystemExit(__doc__)
