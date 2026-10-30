#!/usr/bin/env python3
"""G2c selection and report (docs/specs/GEO1B.md 6-4). D1 run env (rmsd.py: networkx + fragmenter), sbatch only.

  python select_g2c.py check              -> results/C_progress.json (markers per run root; exit 3 if a root is unfinished)
  python select_g2c.py build              -> $G2C/{d1,d0}/g2_final/<rxn_id>/ + results/C_select.csv, C_select.json
  python select_g2c.py report <n_proc>    -> results/C_rxn.csv, C_report.json   (D1 only, descriptive)

Rule (fixed before any result; deployable: reads no label and no G1):
  candidates = the '.done' final folders of the same reaction in run orig (D1-G2 / D0-G2), r1, r2;
  chosen     = lowest result.json e_ts_freq_eh (XTB2 ALPB(water) Freq energy at the final TS, the same ORCA header in
               every run); |dE| <= 1e-7 Eh counts as a tie, broken orig < r1 < r2;
  G2c folder = byte copy of the whole chosen folder (TS, rel1, rel2, ts.hess, work/, result.json) + g2c.json.
'report' compares orig and G2c with the D1 label TS (symmetry-aware heavy-atom RMSD, forming-bond errors)."""
from __future__ import annotations

import json
import math
import os
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RES = HERE / "results"
G2C = Path(os.environ.get("G2C_SCRATCH", "/gpfs/tmp_cpu2/yeseo1ee/g2c"))
ORIG = {"d1": Path(os.environ.get("G2C_ORIG_D1", "/gpfs/tmp_cpu2/yeseo1ee/d1_g2/g2_final")),
        "d0": Path(os.environ.get("G2C_ORIG_D0", "/gpfs/tmp_cpu2/yeseo1ee/d0_g2/g2_final"))}
RUNS = ("orig", "r1", "r2")
TIE_EH, EH2KCAL = 1e-7, 627.509474
D1_LABELS = Path(os.environ.get("G2C_D1_LABELS", "/gpfs/tmp_cpu2/yeseo1ee/d1_g2/G/labels_d1_g2.json"))
D1_JOBS = Path(os.environ.get("G2C_D1_JOBS", "/gpfs/tmp_cpu2/yeseo1ee/d1_pair/d1_jobs"))
C_COMPARE = REPO / "analysis" / "d1_g2" / "results" / "C_compare.csv"
RMSD_DIR = Path(os.environ.get("G2C_RMSD_DIR",
                               "/gpfs/tmp_cpu2/yeseo1ee/d1/home01/d1_env/EDA-ASM/D1_build/validation/analysis"))
N_BOOT, SEED = 2000, 20261030


def root(src, run):
    return ORIG[src] if run == "orig" else G2C / f"{src}_{run}" / "g2_final"


def marker(d):
    if (d / ".done").is_file():
        return ".done"
    f = sorted(p.name for p in d.glob(".fail_*")) if d.is_dir() else []
    return f[0] if f else None


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    os.replace(tmp, path)


def targets(src):
    if src == "d1":
        return sorted(int(d["rxn_id"]) for d in json.loads(D1_LABELS.read_text()))
    lab = G2C / "d0_r1" / "G" / "labels_d0_g2.json"           # written by d0g2.py manifest (D0 targets, 5,260)
    return sorted(int(d["rxn_id"]) for d in json.loads(lab.read_text()))


def cmd_check():
    out, bad = {}, []
    for src in ("d1", "d0"):
        T = targets(src)
        for run in ("r1", "r2"):
            m = [marker(root(src, run) / str(r)) for r in T]
            n_mark = sum(x is not None for x in m)
            out[f"{src}_{run}"] = dict(n_targets=len(T), n_marked=n_mark, n_done=sum(x == ".done" for x in m))
            if n_mark < len(T):
                bad.append(f"{src}_{run}: {len(T) - n_mark} reactions without a marker")
    write_json(RES / "C_progress.json", dict(runs=out, unfinished=bad))
    print(json.dumps(out, indent=1), flush=True)
    return 3 if bad else 0


def energy(d):
    if marker(d) != ".done":
        return None
    e = json.loads((d / "result.json").read_text()).get("e_ts_freq_eh")
    return float(e) if e is not None and math.isfinite(float(e)) else None


def choose(E):
    ok = {k: v for k, v in E.items() if v is not None}
    if not ok:
        return None
    emin = min(ok.values())
    return next(k for k in RUNS if k in ok and ok[k] <= emin + TIE_EH)


def cmd_build():
    if cmd_check():
        print("STOP: a rerun root is unfinished (results/C_progress.json)", file=sys.stderr)
        return 3
    rows = []
    for src in ("d1", "d0"):
        for rid in targets(src):
            E = {run: energy(root(src, run) / str(rid)) for run in RUNS}
            nodone_e = [run for run in RUNS if marker(root(src, run) / str(rid)) == ".done" and E[run] is None]
            ch = choose(E)
            row = dict(src=src, rxn_id=rid, chosen=ch or "", n_done=sum(v is not None for v in E.values()),
                       **{f"e_{k}": E[k] for k in RUNS}, done_without_energy=" ".join(nodone_e),
                       dE_vs_orig_kcal=(E[ch] - E["orig"]) * EH2KCAL if ch and E["orig"] is not None else None)
            rows.append(row)
            if not ch:
                continue
            out = G2C / src / "g2_final" / str(rid)
            meta = dict(rxn_id=rid, src=src, chosen=ch, source=str(root(src, ch) / str(rid)), energies_eh=E,
                        rule="lowest e_ts_freq_eh over .done runs orig/r1/r2, tie 1e-7 Eh -> orig<r1<r2")
            if marker(out) == ".done" and (out / "g2c.json").is_file():
                old = json.loads((out / "g2c.json").read_text())
                if old.get("chosen") == ch and old.get("energies_eh") == E:
                    continue
                print(f"STOP: {out} exists with another choice; move it aside explicitly", file=sys.stderr)
                return 3
            tmp = out.with_name(f"{rid}.tmp")
            if tmp.exists():
                shutil.rmtree(tmp)
            shutil.copytree(root(src, ch) / str(rid), tmp)
            (tmp / "g2c.json").write_text(json.dumps(meta, indent=1))
            if out.exists():
                shutil.rmtree(out)
            tmp.replace(out)
    D = pd.DataFrame(rows)
    RES.mkdir(parents=True, exist_ok=True)
    D.to_csv(RES / "C_select.csv", index=False)
    summ = {}
    for src, g in D.groupby("src"):
        o = g[g.e_orig.notna()]
        ch = g[g.chosen != ""]
        summ[src] = dict(n_targets=int(len(g)), n_g2_orig_done=int(len(o)), n_g2c_done=int(len(ch)),
                         gained=int(((g.chosen != "") & g.e_orig.isna()).sum()),
                         chosen=ch.chosen.value_counts().to_dict(),
                         changed_share_of_orig_done=float((o.chosen != "orig").mean()) if len(o) else None,
                         dE_kcal_quantiles=[float(x) for x in o.dE_vs_orig_kcal.dropna().quantile([0.1, 0.5, 0.9])]
                         if len(o) else None,
                         n_done_without_energy=int((g.done_without_energy != "").sum()))
    write_json(RES / "C_select.json", dict(summary=summ, written=time.strftime("%Y-%m-%dT%H:%M:%S")))
    print(json.dumps(summ, indent=1, default=float), flush=True)
    return 0


# ---------------------------------------------------------------- report (D1, descriptive)
def read_xyz(p):
    L = Path(p).read_text().splitlines()
    n = int(L[0].split()[0])
    rows = [ln.split() for ln in L[2:2 + n]]
    return [r[0] for r in rows], np.array([[float(v) for v in r[1:4]] for r in rows], float)


def _init():
    sys.path.insert(0, str(RMSD_DIR))
    global rm
    import rmsd as rm  # noqa: F401


def errs(x, lab, pairs):
    dl = np.array([np.linalg.norm(lab[1][i] - lab[1][j]) for i, j in pairs])
    dx = np.array([np.linalg.norm(x[1][i] - x[1][j]) for i, j in pairs])
    o = np.argsort(dl)
    return (dx - dl)[o]


def one(a):
    rid, job, pairs, ch = a
    lab = read_xyz(D1_JOBS / job / "ts.xyz")
    o = read_xyz(root("d1", "orig") / str(rid) / "ts.xyz") if ch is not None and \
        marker(root("d1", "orig") / str(rid)) == ".done" else None
    c = read_xyz(G2C / "d1" / "g2_final" / str(rid) / "ts.xyz")

    def m(a_, b_):
        r = float(rm.mapped_heavy_rmsd(a_, b_))
        return r if math.isfinite(r) else float("nan")
    eo = errs(o, lab, pairs) if o else [np.nan, np.nan]
    ec = errs(c, lab, pairs)
    return dict(rxn_id=rid, chosen=ch, map_orig=m(o, lab) if o else np.nan, map_g2c=m(c, lab),
                map_g2c_vs_orig=m(c, o) if o and ch != "orig" else (0.0 if o else np.nan),
                e_short_orig=eo[0], e_long_orig=eo[1], e_short_g2c=ec[0], e_long_g2c=ec[1])


def cmd_report(n_proc):
    S = pd.read_csv(RES / "C_select.csv")
    S = S[(S.src == "d1") & (S.chosen.fillna("") != "")]
    lab = {int(d["rxn_id"]): d for d in json.loads(D1_LABELS.read_text())}
    jobs = [(int(r.rxn_id), lab[int(r.rxn_id)]["job_id"],
             sorted(tuple(sorted(int(x) for x in p.split("-"))) for p in str(lab[int(r.rxn_id)]["formed_pairs_ts"]).split()),
             r.chosen) for r in S.itertuples()]
    with Pool(int(n_proc), initializer=_init) as pool:
        rows = pool.map(one, jobs, chunksize=8)
    D = pd.DataFrame(rows)
    C = pd.read_csv(C_COMPARE, usecols=["rxn_id", "dipole", "skeleton"])
    D = D.merge(C.assign(core=C.dipole + "/" + C.skeleton)[["rxn_id", "core"]], on="rxn_id", how="left")
    D.to_csv(RES / "C_rxn.csv", index=False)
    both = D[D.map_orig.notna() & D.map_g2c.notna()]
    rng = np.random.default_rng(SEED)
    cores = both.core.unique()
    by = {c: g for c, g in both.groupby("core")}
    diffs = {t: [] for t in (0.25, 0.5)}
    for _ in range(N_BOOT):
        g = pd.concat([by[c] for c in rng.choice(cores, len(cores), replace=True)])
        for t in diffs:
            diffs[t].append(float((g.map_g2c <= t).mean() - (g.map_orig <= t).mean()))
    fb = lambda g, k: float(np.nanmedian(np.abs(np.r_[g[f"e_short_{k}"], g[f"e_long_{k}"]])))   # noqa: E731
    rep = dict(n=int(len(D)), n_both=int(len(both)), changed_share=float((both.chosen != "orig").mean()),
               moved_conformer_share=float((both.map_g2c_vs_orig > 0.30).mean()),
               map_median=dict(orig=float(both.map_orig.median()), g2c=float(both.map_g2c.median())),
               share_le={str(t): dict(orig=float((both.map_orig <= t).mean()), g2c=float((both.map_g2c <= t).mean()),
                                      diff_ci=[float(x) for x in np.percentile(diffs[t], [2.5, 97.5])]) for t in diffs},
               form_abs_median=dict(orig=fb(both, "orig"), g2c=fb(both, "g2c")),
               n_boot=N_BOOT, seed=SEED, note="descriptive; no gate. Bins/thresholds fixed in GEO1B.md 6-4.")
    write_json(RES / "C_report.json", rep)
    print(json.dumps(rep, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["check"]:
        sys.exit(cmd_check())
    if a == ["build"]:
        sys.exit(cmd_build())
    if a[:1] == ["report"] and len(a) == 2:
        sys.exit(cmd_report(a[1]))
    raise SystemExit(__doc__)
