#!/usr/bin/env python3
"""validation_report.py — interim check and VALIDATION_REPORT.md (VALIDATION_SPEC §7, §8, §10).

  python analysis/validation_report.py --interim   (worker.sh, once every priority 0-2 task finished)
      A3/A4 (V5), A6 a/b (V1), V2a, A7 (V8), A8 (V9) -> <scratch>/interim_report.md and results/;
      writes <scratch>/queue/STOP when A3, A6(a), A6(b) or A7 FAILs (§8 first row).
  python analysis/validation_report.py --final     (run_report.sh, afterany the worker array)
      everything -> results/VALIDATION_REPORT.md + CSVs + geometries/<job>/
Criteria are read from config_val.yaml `prereg` only; nothing here changes them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

VAL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VAL / "analysis"))
sys.path.insert(0, str(VAL.parent / "d1_autode_pilot"))
import pilot_common as pc  # noqa: E402
import stats as st  # noqa: E402

TARGETS = ["barrier_kcal", "d1_kcal", "d2_kcal", "elst_dft", "pauli_dft", "oi_dft", "disp_dft", "cpcm_dft", "cds_dft"]
SHORT = {t: t.replace("_kcal", "").replace("_dft", "") for t in TARGETS}


def fmt(x, n=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    if isinstance(x, (bool, np.bool_)):
        return "PASS" if x else "FAIL"
    return f"{x:.{n}g}" if isinstance(x, float) and abs(x) < 1e-3 and x != 0 else (f"{x:.{n}f}" if isinstance(x, float) else str(x))


def pf(x):
    return "PASS" if x is True else ("FAIL" if x is False else "not judged")


class Ctx:
    def __init__(self):
        self.cfg = pc.load_config(VAL / "config_val.yaml")
        self.P = self.cfg["prereg"]; self.V = self.cfg["val"]
        self.scr = Path(self.cfg["scratch"]); self.Q = self.scr / "queue"
        self.tasks = pd.read_csv(VAL / "tasks.csv", keep_default_na=False)
        self.man = pd.read_csv(VAL / "val_manifest.csv", keep_default_na=False).set_index("job_id", drop=False)
        self.dset = pd.read_csv(VAL / "val_d0_set.csv")
        self.pilot_jobs = self.scr / "pilot_copy" / "jobs"
        self.MAE = {t: float(self.P["MAE_ref"][t]) for t in TARGETS}

    def jobs(self, exp):
        return self.man.job_id[self.man.exp == exp].tolist()

    def jd(self, job, pilot=False):
        return (self.pilot_jobs if pilot else self.scr / "jobs") / job

    @staticmethod
    def jl(p: Path):
        return json.loads(p.read_text()) if p.is_file() else {}

    def label(self, job, pilot=False):
        d = self.jd(job, pilot)
        lab = self.jl(d / "label.json")
        return lab if (d / ".done_label").is_file() and lab.get("status") == "ok" else None

    def fail(self, job, pilot=False):
        d = self.jd(job, pilot)
        for s in pc.STAGES + ("refopt",):
            f = d / f".fail_{s}"
            if f.is_file():
                return s, f.read_text().strip()
        return None

    def d0(self, job):
        r = self.man.loc[job]
        return {t: float(r[f"d0_{t}"]) for t in TARGETS}

    def state(self, tid):
        return "done" if (self.Q / "done" / tid).exists() else ("failed" if (self.Q / "failed" / tid).exists() else "pending")

    def coley_ts(self, rid):
        pdir = Path(self.cfg["d0_profiles"]) / str(int(rid))
        return pc.read_xyz(sorted(p for p in pdir.glob("TS_*.xyz") if p.name != "TS_imag_mode.xyz")[0])


# ---------------------------------------------------------------- criteria
def deltas(c, jobs):
    rows = []
    for j in jobs:
        lab = c.label(j)
        if lab:
            d0 = c.d0(j)
            rows.append(dict(job_id=j, rxn_id=int(float(c.man.loc[j].d0_rxn_id)), **{t: lab[t] - d0[t] for t in TARGETS}))
    return pd.DataFrame(rows)


def a3(c):
    df = deltas(c, c.jobs("V5"))
    n_req, tol = int(c.P["A3"]["n_required"]), float(c.P["A3"]["max_abs_kcal"])
    mx = float(df[TARGETS].abs().to_numpy().max()) if len(df) else float("nan")
    per = df[TARGETS].abs().max(axis=1).tolist() if len(df) else []
    ok = len(df) == n_req and all(x <= tol for x in per)
    return dict(n_ok=len(df), n_required=n_req, max_abs=mx, passed=ok, table=df)


def a4(c):
    A = c.P["A4"]; rows = []
    for j in c.jobs("V5"):
        port = c.jl(c.jd(j) / "ref_result.json").get("port_diagnostic") or {}
        use = port.get("use_alt"); coley = bool(int(float(c.man.loc[j].d0_alt_used)))
        dG, rm_ = port.get("dG_alt_minus_orig_kcal"), port.get("rmsd_heavy_alt_orig")
        boundary = ((dG is not None and abs(dG - A["dG_thr"]) < A["dG_band"])
                    or (rm_ is not None and abs(rm_ - A["rmsd_thr"]) < A["rmsd_band"]))
        rows.append(dict(job_id=j, coley_alt=coley, port_use_alt=use, agree=None if use is None else use == coley,
                         dG=dG, rmsd=rm_, boundary=boundary, error=port.get("error")))
    df = pd.DataFrame(rows)
    n_agree = int((df.agree == True).sum()) if len(df) else 0                       # noqa: E712
    dis = df[df.agree == False] if len(df) else df                                   # noqa: E712
    ok = n_agree >= int(A["n_agree_min"]) and bool(dis.boundary.all() if len(dis) else True)
    return dict(n_agree=n_agree, n_total=len(df), n_judged=int(df.agree.notna().sum()) if len(df) else 0,
                disagreements=dis.to_dict("records"), passed=ok, table=df)


def a6(c):
    A = c.P["A6"]; out = {}
    ra = [c.jl(c.scr / "v1" / t / "result.json") for t in c.tasks.task_id[c.tasks.exp == "V1a"]]
    rb = [c.jl(c.scr / "v1" / t / "result.json") for t in c.tasks.task_id[c.tasks.exp == "V1b"]]
    if ra and all(r.get("max_abs_dE_eh") is not None for r in ra):
        dE = max(r["max_abs_dE_eh"] for r in ra); dl = max((r.get("max_abs_dlabel_kcal") or 0) for r in ra)
        out["a"] = dict(max_abs_dE_eh=dE, max_abs_dlabel_kcal=dl, passed=dE <= A["dE_eh"] and dl <= A["label_kcal"],
                        runs=ra)
    else:
        out["a"] = dict(passed=None, runs=ra)
    if rb and all(r.get("max_abs_dE_eh") is not None for r in rb):
        dE = max(r["max_abs_dE_eh"] for r in rb)
        out["b"] = dict(max_abs_dE_eh=dE, passed=dE <= A["dE_eh"], runs=rb)
    else:
        out["b"] = dict(passed=None, runs=rb)
    import rmsd as rm
    devs = []
    for rid in c.V["v1c_rxns"]:
        a, b = c.jd(f"V1c_{int(rid):04d}") / "ts.xyz", c.jd(f"V2L0_{int(rid):04d}") / "ts.xyz"
        if a.is_file() and b.is_file():
            xa, xb = pc.read_xyz(a)[1], pc.read_xyz(b)[1]
            devs.append(dict(rxn_id=int(rid), aligned_max_dev_A=rm.aligned_max_dev(xa, xb),
                             raw_max_dev_A=float(np.abs(xa - xb).max())))
    if devs and len(devs) == len(c.V["v1c_rxns"]):
        m = max(d["aligned_max_dev_A"] for d in devs)
        out["c"] = dict(max_dev_A=m, runs=devs, passed=(True if m <= A["coord_pass_A"] else (None if m <= A["coord_fail_A"] else False)),
                        note="between pass and fail limits: size reported, not a FAIL" if A["coord_pass_A"] < m <= A["coord_fail_A"] else "")
    else:
        out["c"] = dict(passed=None, runs=devs)
    return out


def v2a(c):
    rows = []
    for t in c.tasks.task_id[c.tasks.exp == "V2a"]:
        rid = int(t.split("_")[1]); r = c.jl(c.scr / "v2a" / f"{rid:04d}" / "result.json")
        for lv, x in (r.get("levels") or {}).items():
            rows.append(dict(rxn_id=rid, level=lv, max_abs_grad=x.get("max_abs_grad"), rms_grad=x.get("rms_grad"),
                             vwn=x["checks"].get("vwn"), rijcosx=x["checks"].get("rijcosx"), eps=x["checks"].get("eps"),
                             smd_cds=x["checks"].get("smd_cds"), d3=json.dumps(x["checks"].get("d3")),
                             **{f"ok_{k}": v for k, v in (x.get("level_ok") or {}).items()}))
    df = pd.DataFrame(rows)
    if not len(df):
        return dict(table=df, summary={})
    summ = {lv: dict(n=int(g.max_abs_grad.notna().sum()), median_max=float(g.max_abs_grad.median()),
                     max_max=float(g.max_abs_grad.max()), median_rms=float(g.rms_grad.median()))
            for lv, g in df.groupby("level")}
    piv = df.pivot_table(index="rxn_id", columns="level", values="rms_grad")
    dec = int(((piv.get("L0") > piv.get("L1")) & (piv.get("L1") > piv.get("L2"))).sum()) if {"L0", "L1", "L2"} <= set(piv.columns) else None
    return dict(table=df, summary=summ, n_decreasing_L0_L1_L2=dec, n_rxn=int(piv.shape[0]))


def ts_shift(c, jobs):
    import rmsd as rm
    rows = []
    for j in jobs:
        tr = c.jl(c.jd(j) / "ts_result.json")
        f = c.jd(j) / "ts.xyz"
        if not f.is_file() or not tr.get("replay_ts_file"):
            continue
        cts = pc.read_xyz(c.jd(j) / f"d0_{tr['replay_ts_file']}")
        r = rm.mapped_heavy_rmsd(pc.read_xyz(f), cts)
        dd = (np.max(np.abs(np.sort(tr["formed_d"]) - np.sort(tr["coley_ts_formed_d"])))
              if tr.get("formed_d") and tr.get("coley_ts_formed_d") else None)
        rows.append(dict(job_id=j, mapped_rmsd_A=r, max_dform_shift_A=None if dd is None else float(dd),
                         imag=tr.get("imag_freqs_cm")))
    return pd.DataFrame(rows)


def bias_rms(df, c, frac_b, frac_r=None):
    out = {}
    for t in TARGETS:
        x = df[t].to_numpy(float) if len(df) else np.array([])
        m, lo, hi = st.mean_ci(x, c.P["ci_level"])
        r = st.rms(x)
        pi = bool(abs(m) <= frac_b * c.MAE[t] or (np.isfinite(lo) and lo <= 0 <= hi)) if len(x) else None
        pii = (bool(r <= frac_r * c.MAE[t]) if len(x) else None) if frac_r is not None else None
        out[t] = dict(n=len(x), mean=m, ci=(lo, hi), rms=r, pass_bias=pi, pass_rms=pii)
    return out


def a5_eng(c, exp="V2b-L0"):
    A = c.P["A5_eng"]
    df = deltas(c, c.jobs(exp))
    per = bias_rms(df, c, A["bias_frac"], A["rms_frac"])
    sh = ts_shift(c, c.jobs(exp))
    med = float(sh.mapped_rmsd_A.median()) if len(sh) else float("nan")
    ok_t = all(v["pass_bias"] and v["pass_rms"] for v in per.values()) if len(df) else None
    ok = None if ok_t is None else bool(ok_t and med <= A["ts_rmsd_median_max_A"])
    return dict(n=len(df), n_jobs=len(c.jobs(exp)), per_target=per, ts_rmsd_median=med, shift=sh, table=df, passed=ok)


def run_groups(c):
    """{target: [array per reaction]} for V3 (D0) and V6b + pilot run (D1)."""
    d0, d1 = {t: [] for t in TARGETS}, {t: [] for t in TARGETS}
    v3 = c.man[c.man.exp == "V3"]
    for rid, g in v3.groupby(v3.d0_rxn_id.astype(float).astype(int)):
        labs = [c.label(j) for j in g.job_id]
        labs = [l for l in labs if l]
        for t in TARGETS:
            if len(labs) >= 2:
                d0[t].append(np.array([l[t] for l in labs]))
    for j in c.V["v6b_jobs"]:
        labs = [c.label(j, pilot=True)] + [c.label(f"V6b_{j}_r{k}") for k in range(2, 2 + int(c.V["v6b_extra_runs"]))]
        labs = [l for l in labs if l]
        for t in TARGETS:
            if len(labs) >= 2:
                d1[t].append(np.array([l[t] for l in labs]))
    return d0, d1


def a5_run(c):
    d0, d1 = run_groups(c)
    out = {}
    for t in TARGETS:
        s0, s1, sa = st.pooled_sd(d0[t]), st.pooled_sd(d1[t]), st.pooled_sd(d0[t] + d1[t])
        out[t] = dict(sd_d0=s0, sd_d1=s1, sd_all=sa, nf_d0=s0 / c.MAE[t], nf_d1=s1 / c.MAE[t], nf_all=sa / c.MAE[t],
                      n_groups_d0=len(d0[t]), n_groups_d1=len(d1[t]))
    nf_gt1 = any(np.isfinite(v["nf_all"]) and v["nf_all"] > c.P["A5_run"]["nf_decision"] for v in out.values())
    return dict(per_target=out, nf_gt1=nf_gt1, groups_d0=d0)


def e2e_jobs(c):
    sub = set(c.dset.rxn_id[c.dset.subset8])
    out = []
    for rid in c.dset.rxn_id:
        out.append(f"V3_{rid:04d}_r1" if rid in sub else f"V4_{rid:04d}")
    return out


def a5_e2e(c, eng, run):
    A = c.P["A5_e2e"]
    df = deltas(c, e2e_jobs(c))
    per = bias_rms(df, c, A["bias_frac"])
    rng = np.random.default_rng(int(c.P["bootstrap_seed"]))
    B = int(c.P["bootstrap_n"])
    eng_df = eng["table"]; groups = run["groups_d0"]
    for t in TARGETS:
        e = df[t].to_numpy(float) if len(df) else np.array([])
        g_eng = eng_df[t].to_numpy(float) if len(eng_df) else np.array([])
        grp = groups[t]
        den = st.rms(g_eng) ** 2 + 2 * st.pooled_sd(grp) ** 2 if len(grp) else float("nan")
        R = st.rms(e) ** 2 / den if len(e) and np.isfinite(den) and den > 0 else float("nan")
        boots = []
        if np.isfinite(R):
            for _ in range(B):
                eb = rng.choice(e, len(e)); gb = rng.choice(g_eng, len(g_eng))
                gi = [grp[i] for i in rng.integers(0, len(grp), len(grp))]
                d = st.rms(gb) ** 2 + 2 * st.pooled_sd(gi) ** 2
                boots.append(st.rms(eb) ** 2 / d if d > 0 else np.nan)
        lo, hi = st.percentile_ci(boots, c.P["ci_level"])
        fail_ii = bool(np.isfinite(lo) and lo > A["R_ci_lower_fail"])
        per[t].update(R=R, R_ci=(lo, hi), fail_excess=fail_ii)
    tol = float(A["ref_barrier_tol"])
    ref = float((df.barrier_kcal.abs() <= tol).mean()) if len(df) else float("nan")
    # (i) bias fails or (ii) excess variance fails -> FAIL; R not computable for a target -> not judged
    if not len(df):
        passed = None
    elif any(per[t]["pass_bias"] is False or per[t]["fail_excess"] for t in TARGETS):
        passed = False
    elif not all(np.isfinite(per[t]["R"]) for t in TARGETS):
        passed = None
    else:
        passed = True
    return dict(n=len(df), per_target=per, frac_barrier_within=ref, table=df, passed=passed)


def classify(stage, reason, v7_verdict, job, v7_job):
    r = reason or ""
    if stage == "ts" and r.startswith("autodE found no transition state"):
        return "chemical_no_saddle" if (job == v7_job and v7_verdict == "no_saddle_confirmed") else "no_ts_unconfirmed"
    if stage == "ts" and r.startswith("TS check failed"):
        return "gate"
    if stage == "ref" and ("dipolarophile configuration" in r or "ring count" in r):
        return "gate"
    return "pipeline"


def a1(c, v7):
    A = c.P["A1"]
    pm = pd.read_csv(c.V["pilot_manifest"], keep_default_na=False)
    rows = []
    for j in pm.job_id[pm.kind == "D1"]:
        src, pil = (f"V6c_{j}", False) if j == c.V["v6c_job"] else (j, True)
        lab, fl = c.label(src, pil), c.fail(src, pil)
        rows.append(dict(row=j, source=("V6c (F1)" if not pil else "pilot"), ok=lab is not None,
                         fail_stage=fl[0] if fl else None, fail_reason=fl[1] if fl else None))
    for j in c.jobs("V6a"):
        lab, fl = c.label(j), c.fail(j)
        rows.append(dict(row=j, source="V6a", ok=lab is not None, fail_stage=fl[0] if fl else None,
                         fail_reason=fl[1] if fl else None))
    df = pd.DataFrame(rows)
    df["class"] = [None if r.ok else (classify(r.fail_stage, r.fail_reason, v7.get("verdict"), r.row, c.V["v7_job"])
                                      if r.fail_stage else "incomplete") for r in df.itertuples()]
    k, n = int(df.ok.sum()), len(df)
    lo, hi = st.wilson(k, n, c.P["ci_level"])
    fails_ok = bool(df["class"].dropna().isin(["chemical_no_saddle", "gate"]).all())
    judged = n == int(A["n_total"])
    return dict(k=k, n=n, wilson=(lo, hi), table=df, pipeline_bugs=int((df["class"] == "pipeline").sum()),
                passed=(bool(k >= int(A["n_ok_min"]) and fails_ok) if judged else None),
                note="" if judged else f"{n} rows instead of {A['n_total']} (V6a missing: D1_설계_v8.xlsx)")


def a2(c):
    bad, n = [], 0
    for j in c.man.job_id:
        lab = c.jl(c.jd(j) / "label.json")
        if not lab:
            continue
        n += 1
        failed = [k for k, v in (lab.get("gates") or {}).items() if not v]
        if lab.get("status") != "ok" or failed:
            bad.append(dict(job_id=j, status=lab.get("status"), failed_gates=failed))
    return dict(n_labels=n, bad=bad, passed=(len(bad) == 0) if n else None)


def a9(c):
    def coreh(d):
        t = 0.0
        f = d / "timing.tsv"
        if f.is_file():
            for line in f.read_text().splitlines():
                p = line.split("\t")
                if len(p) >= 5 and p[3] == "0":
                    t += (int(p[2]) - int(p[1])) * int(p[4]) / 3600.0
        return t
    pm = pd.read_csv(c.V["pilot_manifest"], keep_default_na=False)
    rows = [dict(job=j, src="pilot", coreh=coreh(c.jd(j, True))) for j in pm.job_id[pm.kind == "D1"]]
    rows += [dict(job=j, src=c.man.loc[j].exp, coreh=coreh(c.jd(j))) for j in c.jobs("V6a") + c.jobs("V6b")]
    df = pd.DataFrame(rows)
    df = df[df.coreh > 0]
    m, lo, hi = st.mean_ci(df.coreh, c.P["ci_level"])
    N, cores = int(c.V["n_d1_total"]), int(c.cfg["cpus_per_task"])
    return dict(n=len(df), mean=m, ci=(lo, hi), median=float(df.coreh.median()) if len(df) else float("nan"),
                max=float(df.coreh.max()) if len(df) else float("nan"), total_extrap=m * N,
                wall_days=m * N / (10 * cores) / 24, table=df)


def a10(c, v7):
    j = f"V6c_{c.V['v6c_job']}"
    j07, j07_fail = c.label(j) is not None, c.fail(j)
    v = v7.get("verdict")
    j03 = v == "no_saddle_confirmed" or (isinstance(v, str) and v.startswith("reclassified"))
    finished = v is not None and (j07 or j07_fail is not None)
    return dict(j07_ok=j07, j07_fail=j07_fail, j03_verdict=v, passed=(j07 and j03) if finished else None)


def ts_match(c):
    import rmsd as rm
    S = c.P["same_ts"]; out = {}
    pairs = []
    v3 = c.man[c.man.exp == "V3"]
    lowest = []
    for rid, g in v3.groupby(v3.d0_rxn_id.astype(float).astype(int)):
        runs = []
        for j in g.job_id:
            tr = c.jl(c.jd(j) / "ts_result.json")
            if (c.jd(j) / "ts.xyz").is_file() and tr.get("formed_d"):
                runs.append((j, pc.read_xyz(c.jd(j) / "ts.xyz"), tr["formed_d"], (tr.get("ts") or {}).get("e_sp_eh")))
        for i in range(len(runs)):
            for k in range(i + 1, len(runs)):
                s, r, dd = rm.same_ts(runs[i][1], runs[k][1], runs[i][2], runs[k][2], S["rmsd_A"], S["dform_A"])
                pairs.append(dict(rxn_id=rid, a=runs[i][0], b=runs[k][0], same=s, rmsd=r, dform=dd))
        e = [r for r in runs if r[3] is not None]
        if e:
            lo = min(e, key=lambda r: r[3])
            d0f = [float(g.iloc[0].d0_formed_d1), float(g.iloc[0].d0_formed_d2)]
            s, r, dd = rm.same_ts(lo[1], c.coley_ts(rid), lo[2], d0f, S["rmsd_A"], S["dform_A"])
            lowest.append(dict(rxn_id=rid, job=lo[0], same_as_coley=s, rmsd=r, dform=dd))
    ctrl = []
    for j in e2e_jobs(c):
        tr = c.jl(c.jd(j) / "ts_result.json")
        if (c.jd(j) / "ts.xyz").is_file() and tr.get("formed_d"):
            row = c.man.loc[j]
            s, r, dd = rm.same_ts(pc.read_xyz(c.jd(j) / "ts.xyz"), c.coley_ts(row.d0_rxn_id), tr["formed_d"],
                                  [float(row.d0_formed_d1), float(row.d0_formed_d2)], S["rmsd_A"], S["dform_A"])
            ctrl.append(dict(job_id=j, same_as_coley=s, rmsd=r, dform=dd))
    for name, rows, key in (("v3_pairs", pairs, "same"), ("v4_vs_coley", ctrl, "same_as_coley"),
                            ("lowest_of_3_vs_coley", lowest, "same_as_coley")):
        df = pd.DataFrame(rows)
        out[name] = dict(n=len(df), rate=float(df[key].mean()) if len(df) else float("nan"), table=df)
    return out


def extra_imag(c):
    lo, hi = c.P["extra_imag_band_cm"]; rows = []
    for exp in ("V2b-L0", "V2b-L2", "V3", "V4"):
        for j in c.jobs(exp):
            im = c.jl(c.jd(j) / "ts_result.json").get("imag_freqs_cm")
            if im:
                extra = [v for v in sorted(im)[1:] if lo < v < hi]
                rows.append(dict(exp=exp, job_id=j, n_imag=len(im), extra=extra))
    df = pd.DataFrame(rows)
    ex = df[df.extra.map(len) > 0] if len(df) else df
    return dict(n_ts=len(df), n_with_extra=len(ex), extra_values=sorted(v for e in ex.extra for v in e) if len(ex) else [])


def prereg_hash():
    files = ["config_val.yaml", "val_d0_set.csv", "tasks.csv", "val_manifest.csv"]
    out = {}
    for f in files:
        p = VAL / f
        out[f] = dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.is_file() else None)
        try:
            out[f]["commit"] = subprocess.run(["git", "-C", str(VAL), "log", "-1", "--format=%H", "--", f],
                                              capture_output=True, text=True).stdout.strip() or None
            out[f]["clean"] = subprocess.run(["git", "-C", str(VAL), "diff", "--quiet", "HEAD", "--", f]).returncode == 0
        except OSError:
            pass
    return out


def worker_jobs(c):
    ids = set()
    for d in ("done", "failed"):
        for f in (c.Q / d).glob("*"):
            t = f.read_text().split()
            if t:
                ids.add(t[0])
    return sorted(ids)


# ---------------------------------------------------------------- output
def interim(c):
    r3, r4, r6, g, v8, v9 = a3(c), a4(c), a6(c), v2a(c), c.jl(c.scr / "v8" / "result.json"), c.jl(c.scr / "v9" / "result.json")
    a7 = (v8.get("n_detected") == int(c.P["A7"]["n_required"])) if v8 else None
    L = ["# D1 validation — interim report (priority 0–2)\n",
         f"| check | value | result |\n|---|---|---|",
         f"| A3 (V5 replay) | {r3['n_ok']}/{r3['n_required']} ok, max \\|Δ\\| {fmt(r3['max_abs'], 4)} kcal/mol | {pf(r3['passed'])} |",
         f"| A4 (V5 port) | {r4['n_agree']}/{r4['n_total']} agree | {pf(r4['passed'])} |",
         f"| A6(a) same input | max \\|ΔE\\| {fmt(r6['a'].get('max_abs_dE_eh'))} Eh, labels {fmt(r6['a'].get('max_abs_dlabel_kcal'))} | {pf(r6['a']['passed'])} |",
         f"| A6(b) nprocs 1 vs 4 | max \\|ΔE\\| {fmt(r6['b'].get('max_abs_dE_eh'))} Eh | {pf(r6['b']['passed'])} |",
         f"| A7 (V8) | {v8.get('n_detected')}/{v8.get('n_controls')} detected | {pf(a7)} |",
         f"| A8 (V9) | ok hard fails {v9.get('n_ok_hard_fail')}, foreign {v9.get('foreign_detected')} | (final report) |",
         f"| V2a | {json.dumps(g.get('summary'))} | diagnostic |"]
    stop = [k for k, v in (("A3", r3["passed"]), ("A6(a)", r6["a"]["passed"]), ("A6(b)", r6["b"]["passed"]), ("A7", a7))
            if v is False]
    L.append(f"\n**STOP conditions (§8 row 1): {'FAIL ' + ', '.join(stop) + ' -> queue STOP written' if stop else 'none'}**")
    text = "\n".join(L) + "\n"
    (c.scr / "interim_report.md").write_text(text)
    (VAL / "results").mkdir(exist_ok=True)
    (VAL / "results" / "interim_report.md").write_text(text)
    if stop:
        c.Q.mkdir(parents=True, exist_ok=True)
        (c.Q / "STOP").write_text(f"interim: {', '.join(stop)} FAIL (VALIDATION_SPEC §8)\n")
    print(text)


def final(c):
    res = VAL / "results"; res.mkdir(exist_ok=True)
    v7 = c.jl(c.scr / "v7" / c.V["v7_job"] / "result.json")
    v8 = c.jl(c.scr / "v8" / "result.json"); v9 = c.jl(c.scr / "v9" / "result.json")
    r1, r2, r3, r4, r6 = a1(c, v7), a2(c), a3(c), a4(c), a6(c)
    eng, eng2 = a5_eng(c, "V2b-L0"), a5_eng(c, "V2b-L2")
    run = a5_run(c); e2e = a5_e2e(c, eng, run)
    a7 = (v8.get("n_detected") == int(c.P["A7"]["n_required"])) if v8 else None
    fdet = v9.get("foreign_detected") or {}
    a8 = (v9.get("n_ok_hard_fail", 1) <= int(c.P["A8"]["ok_hard_fail_max"]) and all(fdet.values())) if v9 else None
    r9, r10, g, tm, xi = a9(c), a10(c, v7), v2a(c), ts_match(c), extra_imag(c)
    a6_all = (r6["a"]["passed"] and r6["b"]["passed"] and r6["c"]["passed"] is not False) if (
        r6["a"]["passed"] is not None and r6["b"]["passed"] is not None) else None
    ph = prereg_hash()

    # CSV + geometries
    for name, df in (("a1_rows", r1["table"]), ("a3_v5_deltas", r3["table"]), ("a4_v5_port", r4["table"]),
                     ("a5_eng_L0_deltas", eng["table"]), ("a5_eng_L2_deltas", eng2["table"]),
                     ("a5_eng_L0_ts_shift", eng["shift"]), ("a5_e2e_deltas", e2e["table"]), ("a9_cost", r9["table"]),
                     ("v2a_grad", g["table"]), ("ts_match_v3_pairs", tm["v3_pairs"]["table"]),
                     ("ts_match_v4", tm["v4_vs_coley"]["table"])):
        if len(df):
            df.to_csv(res / f"{name}.csv", index=False)
    rows = []
    for t in c.tasks.itertuples():
        rows.append(dict(task_id=t.task_id, exp=t.exp, priority=t.priority, state=c.state(t.task_id),
                         job_fail=(c.fail(t.job_id) or ("", ""))[1] if t.job_id else ""))
    pd.DataFrame(rows).to_csv(res / "task_status.csv", index=False)
    if (c.scr / "v9" / "v9_gate_audit.csv").is_file():
        shutil.copy(c.scr / "v9" / "v9_gate_audit.csv", res / "v9_gate_audit.csv")
    for f, d in ((c.scr / "v8" / "result.json", "v8_negative_controls.json"), (c.scr / "v7" / c.V["v7_job"] / "result.json", "v7_scan_j03.json")):
        if f.is_file():
            shutil.copy(f, res / d)
    import report as prep
    for j in c.man.job_id:
        prep.copy_geometries(c.jd(j), res / "geometries" / j)
    refopt = [dict(job_id=j, **c.jl(c.jd(j) / "label_refopt.json").get("delta_refopt_minus_label", {}))
              for j in c.jobs("V2b-L0") if (c.jd(j) / "label_refopt.json").is_file()]
    if refopt:
        pd.DataFrame(refopt).to_csv(res / "v2c_refopt_deltas.csv", index=False)

    # §8 decision table
    stop_pipe = any(x is False for x in (r3["passed"], r6["a"]["passed"], r6["b"]["passed"], a7))
    go = a6_all is True and eng["passed"] is True and e2e["passed"] is True
    rows8 = [
        ("A3, A6(a)(b), A7 중 하나라도 FAIL", stop_pipe, "STOP. `$Q/STOP`, 본실험 불가"),
        ("A6 PASS, A5-eng PASS, A5-e2e PASS", go, "GO (ORCA L0). A5-run은 따로 판단"),
        ("A5-eng FAIL (L0), L2 기준 충족", eng["passed"] is False and eng2["passed"] is True, "사용자 결정: L2 설정 / D0 재최적화"),
        ("A5-eng FAIL (L0, L2 모두)", eng["passed"] is False and eng2["passed"] is False, "사용자 결정: G16 / D0 전체 ORCA 재라벨"),
        ("A5-e2e FAIL, A5-eng PASS", e2e["passed"] is False and eng["passed"] is True, "STOP 후 보고 (autodE 1.2→1.4.5 가능성)"),
        ("A5-run: NF > 1 채널 있음", run["nf_gt1"], "사용자 결정: 라벨 정의·잡음 하한 서술"),
        ("A1 FAIL", r1["passed"] is False, "실패 유형별 대책 후 재검증"),
        ("A2, A4, A8, A10 FAIL", any(x is False for x in (r2["passed"], r4["passed"], a8, r10["passed"])), "원인 보고, 수정안, 부분 재검증"),
    ]
    if stop_pipe or (e2e["passed"] is False and eng["passed"] is True):
        rec = "STOP"
    elif go and all(x is not False for x in (r1["passed"], r2["passed"], r4["passed"], a8, r10["passed"])):
        rec = "GO (ORCA L0)" + (" — A5-run NF > 1: 라벨 정의 결정 필요" if run["nf_gt1"] else "")
    else:
        rec = "조건부 (해당 행의 사용자 결정 필요)"

    L = ["# D1 validation — VALIDATION_REPORT\n",
         "## 1. 사전 등록과 실행\n",
         "| file | sha256[:16] | commit | clean |\n|---|---|---|---|"]
    L += [f"| {f} | {v.get('sha256')} | {v.get('commit')} | {v.get('clean')} |" for f, v in ph.items()]
    L.append(f"\nworker job ids: {', '.join(worker_jobs(c)) or '—'}")
    counts = pd.read_csv(res / "task_status.csv").state.value_counts().to_dict()
    L.append(f"\ntasks: {len(c.tasks)} — {counts}\n")

    L.append("## 2. 판정표 (A1–A10)\n\n| id | value | 95% CI | result | note |\n|---|---|---|---|---|")
    L.append(f"| A1 | {r1['k']}/{r1['n']} ok, pipeline bugs {r1['pipeline_bugs']} | {fmt(r1['wilson'][0])}–{fmt(r1['wilson'][1])} | {pf(r1['passed'])} | {r1['note']} |")
    L.append(f"| A2 | {len(r2['bad'])} of {r2['n_labels']} labels with a failed gate | | {pf(r2['passed'])} | |")
    L.append(f"| A3 | {r3['n_ok']}/{r3['n_required']}, max \\|Δ\\| {fmt(r3['max_abs'], 4)} kcal/mol | | {pf(r3['passed'])} | |")
    L.append(f"| A4 | {r4['n_agree']}/{r4['n_total']} agree | | {pf(r4['passed'])} | disagreements: {len(r4['disagreements'])} |")
    L.append(f"| A5-eng | n = {eng['n']}/{eng['n_jobs']}, TS RMSD median {fmt(eng['ts_rmsd_median'], 4)} Å | see §3 | {pf(eng['passed'])} | |")
    L.append(f"| A5-run | NF max {fmt(max((v['nf_all'] for v in run['per_target'].values() if np.isfinite(v['nf_all'])), default=float('nan')))} | | report only | NF > 1: {run['nf_gt1']} |")
    L.append(f"| A5-e2e | n = {e2e['n']}; \\|Δbarrier\\| ≤ 1.0: {fmt(e2e['frac_barrier_within'])} | see §3 | {pf(e2e['passed'])} | |")
    L.append(f"| A6 | (a) {fmt(r6['a'].get('max_abs_dE_eh'))} Eh (b) {fmt(r6['b'].get('max_abs_dE_eh'))} Eh (c) {fmt(r6['c'].get('max_dev_A'))} Å | | {pf(a6_all)} | {r6['c'].get('note', '')} |")
    L.append(f"| A7 | {v8.get('n_detected')}/{v8.get('n_controls')} | | {pf(a7)} | |")
    L.append(f"| A8 | ok hard fails {v9.get('n_ok_hard_fail')}; foreign {fdet} | | {pf(a8)} | |")
    L.append(f"| A9 | {fmt(r9['mean'], 2)} core-h/row (n={r9['n']}); 2,846 rows ≈ {fmt(r9['total_extrap'], 0)} core-h, {fmt(r9['wall_days'], 1)} d | {fmt(r9['ci'][0], 2)}–{fmt(r9['ci'][1], 2)} | report only | |")
    L.append(f"| A10 | J07 ok: {r10['j07_ok']}; J03: {r10['j03_verdict']} | | {pf(r10['passed'])} | |")

    L.append("\n## 3. 가설 판정 자료 (H0 / H1 / H2)\n")
    L.append(f"**V1 (H0):** A6(a) runs {json.dumps([{k: r.get(k) for k in ('src_job', 'max_abs_dE_eh', 'max_abs_dlabel_kcal')} for r in r6['a']['runs']])}; "
             f"A6(b) {json.dumps([{k: r.get(k) for k in ('src_job', 'max_abs_dE_eh')} for r in r6['b']['runs']])}; A6(c) {json.dumps(r6['c'].get('runs'))}\n")
    L.append(f"**V2a:** {json.dumps(g.get('summary'))}; rms gradient decreasing L0 > L1 > L2 in {g.get('n_decreasing_L0_L1_L2')}/{g.get('n_rxn')}\n")
    for name, e in (("V2b L0 (A5-eng)", eng), ("V2b L2", eng2), ("V4 e2e (A5-e2e)", e2e)):
        L.append(f"**{name}** (n = {e['n']})\n\n| target | MAE_ref | mean Δ | 95% CI | RMS Δ | bias ok | RMS ok | R | R CI |\n|---|---|---|---|---|---|---|---|---|")
        for t in TARGETS:
            v = e["per_target"][t]
            L.append(f"| {SHORT[t]} | {c.MAE[t]} | {fmt(v['mean'])} | {fmt(v['ci'][0])}–{fmt(v['ci'][1])} | {fmt(v['rms'])} | "
                     f"{pf(v['pass_bias'])} | {pf(v.get('pass_rms'))} | {fmt(v.get('R'))} | "
                     f"{fmt((v.get('R_ci') or (None, None))[0])}–{fmt((v.get('R_ci') or (None, None))[1])} |")
        L.append("")
    L.append("**V3 / V6b (A5-run):**\n\n| target | SD_run D0 | SD_run D1 | SD all | NF all |\n|---|---|---|---|---|")
    for t in TARGETS:
        v = run["per_target"][t]
        L.append(f"| {SHORT[t]} | {fmt(v['sd_d0'])} | {fmt(v['sd_d1'])} | {fmt(v['sd_all'])} | {fmt(v['nf_all'])} |")
    L.append(f"\n**TS 일치율:** V3 pairs {fmt(tm['v3_pairs']['rate'])} (n={tm['v3_pairs']['n']}); "
             f"V4 control = Coley {fmt(tm['v4_vs_coley']['rate'])} (n={tm['v4_vs_coley']['n']}); "
             f"lowest of 3 = Coley {fmt(tm['lowest_of_3_vs_coley']['rate'])} (n={tm['lowest_of_3_vs_coley']['n']})")
    L.append(f"\n**새 허수 규칙 영향:** {xi['n_with_extra']}/{xi['n_ts']} TS with extra modes in (−50, 0) cm⁻¹: {xi['extra_values']}")
    if refopt:
        d = pd.DataFrame(refopt)
        L.append(f"\n**V2c refopt (refopt − label, kcal/mol):** " + ", ".join(
            f"{k} mean {d[k].mean():+.3f} / max \\|·\\| {d[k].abs().max():.3f}" for k in ("d1_kcal", "d2_kcal", "barrier_kcal") if k in d))

    L.append("\n## 4. 교수님 명제에 대한 답 (데이터만)\n")
    L.append(f"같은 ORCA 입력 → 최대 차이 {fmt(r6['a'].get('max_abs_dE_eh'))} Eh (nprocs 1 vs 4: {fmt(r6['b'].get('max_abs_dE_eh'))} Eh; "
             f"같은 OptTS 두 번: {fmt(r6['c'].get('max_dev_A'))} Å). Coley 구조에서 ORCA L0 재최적화 → TS 이동 중앙값 "
             f"{fmt(eng['ts_rmsd_median'], 4)} Å, 채널 \\|mean Δ\\| 최대 "
             f"{fmt(max((abs(v['mean']) for v in eng['per_target'].values() if np.isfinite(v['mean'])), default=float('nan')))} kcal/mol. "
             f"반복 실행 → pooled SD 최대 {fmt(max((v['sd_all'] for v in run['per_target'].values() if np.isfinite(v['sd_all'])), default=float('nan')))} kcal/mol.\n")

    L.append("## 5. §8 결정표\n\n| 조건 | 해당 | 조치 |\n|---|---|---|")
    L += [f"| {a} | {'**예**' if b else ('아니오' if b is False else '판정 불가')} | {x} |" for a, b, x in rows8]
    L.append(f"\n**권고: {rec}**\n")
    L.append("## 6. 비용\n")
    L.append(f"D1 행당 {fmt(r9['mean'], 2)} core-h (중앙값 {fmt(r9['median'], 2)}, 최대 {fmt(r9['max'], 2)}, n={r9['n']}); "
             f"2,846행 외삽 {fmt(r9['total_extrap'], 0)} core-h, 10 job × {c.cfg['cpus_per_task']}코어로 약 {fmt(r9['wall_days'], 1)}일.\n")
    L.append("## 7. 한계\n\n- 표본은 35원자 이하 (D0 중앙값 44원자)\n- SMD 이산화 차이는 ORCA 안에서 격리할 수 없음 (L2도 ORCA SMD)\n"
             "- D0 진동수 파일이 없어 V9에서 허수 개수 gate는 적용 불가\n" + ("- " + r1["note"] + "\n" if r1["note"] else ""))
    (res / "VALIDATION_REPORT.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--interim", action="store_true"); g.add_argument("--final", action="store_true")
    a = ap.parse_args()
    c = Ctx()
    interim(c) if a.interim else final(c)


if __name__ == "__main__":
    main()
