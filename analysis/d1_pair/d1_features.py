#!/usr/bin/env python3
"""d1_features.py — D1PAIR Phase F (§8): the 182 rev 6 features of every D1 G1 row, the F-0 reproduction check and
the pair sets of §2-5.

  python d1_features.py run <i> <n>     # slice i of n of the G1-ok rows: g1feat (72 cols) then ext (B1..B6) per rxn
  python d1_features.py merge           # aggregate.derive, X182_d1.parquet, F-1, row flow -> results/F_report.json
  python d1_features.py f0check         # F-0: 20 D0 rxns recomputed through this path vs X182_all.parquet (exit 3 = STOP)
  python d1_features.py pairs           # P_feat / P_main / P_sens / A_feat / A_main (+ P-4 drops) -> results/F_pairs.csv

Same order and code as rev 6 r6_features.cmd_g1feat / cmd_ext: r6_features._proc_one (= xtb_slice.process_one(label,
"g1")) in a spawn pool, aggregate.derive on the merged frame, columns reindexed to the rev 4 G1 parquet, then
r6_features._ext (= ext_features.run_pool + assemble_row, CODE_VERSION r5-ext-3, all six blocks) on the xtb_status ok
rows. Environment = jobs/g_env.sh with D1PAIR_PHASE=F: G1_ROOT = $D1PAIR_SCRATCH/g1_final, ESPLEY_PARTITION=label,
ext cache = $R5_SCRATCH/ext = $D1PAIR_SCRATCH/r5/ext, slices under $ESPLEY_OUT = $D1PAIR_SCRATCH/xtb.
X182_d1.parquet = rxn_id + the 182 columns (rev6_common.our_columns order = X182_all.parquet) of the rows with
xtb_status ok, ext_status ok and 182 finite values (F-1; a B1 engine-gate failure stays a failure).
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import d1pair_common as C  # noqa: E402
import d1_g1 as DG  # noqa: E402

FEAT_G1_REV4 = Path("/gpfs/tmp_cpu2/yeseo1ee/espley_xtb/xtb_features_g1.parquet")       # read-only: column order
REV4_G1 = Path("/gpfs/tmp_cpu2/yeseo1ee/espley_xtb_g1")
REV6_G1 = Path("/gpfs/tmp_cpu2/yeseo1ee/espley_xtb_g1_rev6")
D0_PROF = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/dipolar_cycloaddition/extracted/full_dataset_profiles")
D0_X182 = Path("/gpfs/tmp_cpu2/yeseo1ee/espley_rev6/xtb_r6/X182_all.parquet")


def slice_dir():
    return Path(os.environ["ESPLEY_OUT"]) / "f_slices"


VENV5_PY = Path("/gpfs/tmp_cpu2/yeseo1ee/espley_rev5/venv/bin/python")


def check_engines():
    """exit 3 unless the rev 6 feature engines are in use: the rev 5 venv interpreter, tblite 0.7.0, xtb 6.7.1 (edcfbbe)
    (the ext_features smoke test; a cache record carries no engine version, so this runs before any cache access)."""
    import ext_features as ef
    v = ef.versions()
    ok = (v.get("tblite") == C.R5.TBLITE_VERSION and any("6.7.1" in l and "edcfbbe" in l for l in v.get("xtb", []))
          and Path(sys.executable).resolve() == VENV5_PY.resolve())
    if not ok:
        print(f"STOP: feature engines {v} / {sys.executable} are not the rev 6 ones (tblite {C.R5.TBLITE_VERSION}, "
              f"xtb {C.R5.XTB_VERSION}, {VENV5_PY}); submit with PYENV=venv5", file=sys.stderr)
        sys.exit(3)
    return v


def record_versions(sdir, tag, v):
    f = sdir / f"{tag}_versions.json"
    if f.is_file() and json.loads(f.read_text()) != json.loads(json.dumps(v)):
        print(f"STOP: {f.name} was made under other engines: {f.read_text()[:300]}", file=sys.stderr)
        sys.exit(3)
    if not f.is_file():
        C.write_json_safe(f, v)


def g1_done_ids():
    root = Path(os.environ["G1_ROOT"])
    return sorted(int(p.parent.name) for p in root.glob("*/.done"))


def run_ids(ids, tag, sdir, cache_root, workers):
    """g1feat + ext of ids -> sdir/{tag}_g1.parquet, sdir/{tag}_ext.parquet (each skipped when present)."""
    import r6_features as RF
    g1p, exp = sdir / f"{tag}_g1.parquet", sdir / f"{tag}_ext.parquet"
    C.assert_write_ok(sdir)
    sdir.mkdir(parents=True, exist_ok=True)
    record_versions(sdir, tag, check_engines())
    if not g1p.is_file():
        t0 = time.time()
        with mp.get_context("spawn").Pool(workers) as pool:
            rows = pool.map(RF._proc_one, ids, chunksize=1)
        import aggregate
        df = pd.DataFrame(rows)
        aggregate.derive(df)                                       # as r6_features.cmd_g1feat (again after the merge)
        ref_cols = list(pd.read_parquet(FEAT_G1_REV4).columns)
        extra = sorted(set(df.columns) - set(ref_cols))
        if extra:
            raise SystemExit(f"STOP: columns not in the rev 4 parquet: {extra}")
        df = df.reindex(columns=ref_cols)
        C.write_atomic(g1p, lambda t: df.to_parquet(t, index=False))
        print(f"[{tag}] g1feat {len(df)} rows {df.xtb_status.value_counts().to_dict()} ({time.time() - t0:.0f} s)", flush=True)
    g1 = pd.read_parquet(g1p, columns=["rxn_id", "xtb_status"])
    ok = sorted(int(r) for r in g1.loc[g1.xtb_status == "ok", "rxn_id"])
    if not exp.is_file():
        os.environ["SLURM_CPUS_PER_TASK"] = str(workers)          # r6_features._ext sizes its pool from this
        df, ver = RF._ext(ok, g1p, cache_root, exp, f"ext-{tag}")
        print(f"[{tag}] ext {len(df)} rows {df.ext_status.value_counts().to_dict()}", flush=True)
    return g1p, exp


def cmd_run(i, n):
    DG.check_env()
    if os.environ.get("D1PAIR_PHASE") != "F" or Path(os.environ["G1_ROOT"]) != C.G1_ROOT_D1F:
        print("STOP: Phase F needs D1PAIR_PHASE=F (G1_ROOT = g1_final)", file=sys.stderr); return 3
    ids = g1_done_ids()[i::n]
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    print(f"[F slice {i}/{n}] {len(ids)} rxns, {workers} workers, G1_ROOT {os.environ['G1_ROOT']}", flush=True)
    import rev5_common as R5
    run_ids(ids, f"slice{i:02d}of{n:02d}", slice_dir(), R5.SCRATCH / "ext", workers)
    return 0


def merge_rows(g1_paths, ext_paths):
    """the rev 6 merge: concat, aggregate.derive, reindex to the rev 4 G1 columns; ext rows; X182 (182 cols)."""
    import aggregate
    import train_ml_single as T
    g = pd.concat([pd.read_parquet(p) for p in g1_paths], ignore_index=True)
    aggregate.derive(g)
    ref_cols = list(pd.read_parquet(FEAT_G1_REV4).columns)
    extra = sorted(set(g.columns) - set(ref_cols))
    if extra:
        raise SystemExit(f"STOP: columns not in the rev 4 parquet: {extra}")
    g = g.reindex(columns=ref_cols)
    e = pd.concat([pd.read_parquet(p) for p in ext_paths], ignore_index=True)
    cols = C.R6.our_columns(T.FEATURE_SETS)
    gcols = [c for c in cols if c in g.columns]
    ecols = [c for c in cols if c not in g.columns]
    m = g[["rxn_id", "xtb_status"] + gcols].merge(e[["rxn_id", "ext_status"] + ecols], on="rxn_id", how="left")
    vals = m[cols].to_numpy(float)
    m["n_nonfinite"] = (~np.isfinite(vals)).sum(axis=1)
    m["F_ok"] = (m.xtb_status == "ok") & (m.ext_status == "ok") & (m.n_nonfinite == 0)
    return g, e, m, cols


def cmd_merge():
    DG.check_env()
    sd = slice_dir()
    g1p, exp = sorted(sd.glob("slice*_g1.parquet")), sorted(sd.glob("slice*_ext.parquet"))
    n = {p.name.split("of")[1][:2] for p in g1p}
    if len(n) != 1 or len(g1p) != int(n.pop()) or len(exp) != len(g1p):
        print(f"GATE: incomplete slices {len(g1p)} g1 / {len(exp)} ext", file=sys.stderr); return 2
    vers = {p.name: json.loads(p.read_text()) for p in sorted(sd.glob("slice*_versions.json"))}
    v0 = check_engines()
    if len(vers) != len(g1p) or any(json.loads(json.dumps(v)) != json.loads(json.dumps(v0)) for v in vers.values()):
        print(f"STOP: slice engine versions differ or are missing: {vers}", file=sys.stderr); return 3
    g, e, m, cols = merge_rows(g1p, exp)
    done = g1_done_ids()
    if sorted(g.rxn_id) != done:
        print(f"STOP: g1feat rows {len(g)} != G1 .done rows {len(done)}", file=sys.stderr); return 3
    x = m.loc[m.F_ok, ["rxn_id"] + cols].sort_values("rxn_id").reset_index(drop=True)
    C.write_atomic(Path(os.environ["ESPLEY_OUT"]) / "xtb_features_g1_d1.parquet", lambda t: g.to_parquet(t, index=False))
    C.write_atomic(Path(os.environ["ESPLEY_OUT"]) / "xtb_features_ext_g1_d1.parquet", lambda t: e.to_parquet(t, index=False))
    C.write_atomic(C.X182_D1, lambda t: x.to_parquet(t, index=False))
    x0 = pd.read_parquet(D0_X182)
    prep = pd.read_csv(DG.PREP_CSV)
    grows = pd.read_csv(C.RES / "G_rows.csv")
    why = []
    for r in prep.itertuples():
        gr = grows[grows.rxn_id == r.rxn_id]
        if not (r.g1_ok and r.g2_ok):
            why.append("g2_fail")
        elif gr.empty or gr.status.iloc[0] != "ok":
            why.append("g1_" + (gr.status.iloc[0] if len(gr) else "missing"))
        else:
            mr = m[m.rxn_id == r.rxn_id]
            if mr.empty:
                why.append("g1feat_missing")
            elif mr.xtb_status.iloc[0] != "ok":
                why.append(f"g1feat:{mr.xtb_status.iloc[0]}")
            elif mr.ext_status.iloc[0] != "ok":
                why.append(f"ext:{mr.ext_status.iloc[0]}")
            elif mr.n_nonfinite.iloc[0]:
                why.append("nonfinite_182")
            else:
                why.append("ok")
    prep["F_reason"] = why
    C.write_atomic(C.RES / "F_rows.csv", lambda t: prep[["order", "job_id", "rxn_id", "role", "core", "F_reason"]]
                   .to_csv(t, index=False))
    import ext_features as ef
    rep = dict(n_g1_targets=len(prep), n_g1_done=len(done), n_feat=len(x), x182_d1=dict(path=str(C.X182_D1),
               sha256=C.sha256(C.X182_D1), n=len(x), columns_equal_X182_all=list(x.columns) == list(x0.columns)),
               reasons=pd.Series(why).value_counts().to_dict(),
               ref_fail=int(((prep.role == "reference") & (prep.F_reason != "ok")).sum()),
               xtb_status=m.xtb_status.value_counts().to_dict(), ext_status=m.ext_status.value_counts().to_dict(),
               F1_all_finite=bool(np.isfinite(x[cols].to_numpy(float)).all()), versions=ef.versions(), versions_slices=vers,
               si_decision=C.SI_DECISION)
    rep["stop"] = not rep["x182_d1"]["columns_equal_X182_all"] or not rep["F1_all_finite"]
    C.write_json_safe(C.RES / "F_report.json", rep)
    print(json.dumps(rep, indent=1, default=str))
    return 3 if rep["stop"] else 0


# ================================================================ F-0 (D0 reproduction)
def f0_ids():
    rows = pd.read_csv(C.ESP_DIR / "results_rev6" / "R_rows_final.csv")
    feat = rows[rows.R_feat & (rows.rxn_id != C.R6.R4_RXN)]
    rng = np.random.default_rng(C.SEED)
    ids = sorted(int(i) for i in rng.choice(sorted(feat.rxn_id.astype(int)), size=C.F0_N_CHECK, replace=False))
    src = dict(zip(feat.rxn_id.astype(int), feat.recovered_by))
    return ids, src


def cmd_f0check():
    DG.check_env()
    v0 = check_engines()
    ids, src = f0_ids()
    base = C.SCRATCH / "F0chk"
    root = base / "g1root"
    C.assert_write_ok(root)
    root.mkdir(parents=True, exist_ok=True)
    for r in ids:
        tgt = REV4_G1 / str(r) if src[r] == "g1_rev4" else REV6_G1 / str(r)
        link = root / str(r)
        if not link.is_symlink():
            os.symlink(tgt, link)
    env = {**os.environ, "ESPLEY_LABELS": str(C.D0_LABELS), "ESPLEY_META": str(C.D0_META), "ESPLEY_PROF": str(D0_PROF),
           "ESPLEY_SMILES": str(C.D0_SMILES), "G1_ROOT": str(root), "R5_SCRATCH": str(base / "r5"),
           "ESPLEY_OUT": str(base / "xtb"), "D1PAIR_PHASE": "F0CHK"}
    r = subprocess.run([sys.executable, str(HERE / "d1_features.py"), "_f0run", ",".join(map(str, ids))], env=env,
                       capture_output=True, text=True)
    print(r.stdout[-4000:], r.stderr[-4000:], flush=True)
    if r.returncode != 0:
        print(f"STOP: F-0 run exit {r.returncode}", file=sys.stderr); return 3
    g, e, m, cols = merge_rows([base / "xtb" / "f_slices" / "f0chk_g1.parquet"], [base / "xtb" / "f_slices" / "f0chk_ext.parquet"])
    x0 = pd.read_parquet(D0_X182).set_index("rxn_id")
    mm = m.set_index("rxn_id")
    b1 = set(C.R5.BLOCKS["B1"])
    out, worst = [], {"B1": 0.0, "other": 0.0}
    for rid in ids:
        rec = dict(rxn_id=rid, src=src[rid], xtb_status=mm.loc[rid, "xtb_status"], ext_status=mm.loc[rid, "ext_status"])
        if mm.loc[rid, "F_ok"]:
            d = (mm.loc[rid, cols].astype(float) - x0.loc[rid, cols].astype(float)).abs()
            rec["max_B1"] = float(d[[c for c in cols if c in b1]].max())
            rec["max_other"] = float(d[[c for c in cols if c not in b1]].max())
            rec["worst_col"] = str(d.idxmax())
            worst["B1"] = max(worst["B1"], rec["max_B1"]); worst["other"] = max(worst["other"], rec["max_other"])
        out.append(rec)
    all_ok = all(mm.loc[r, "F_ok"] for r in ids)
    strict = all_ok and worst["B1"] <= C.F0_TOL_B1 and worst["other"] <= C.F0_TOL
    ok = strict or (all_ok and worst["B1"] <= C.F0_TOL_B1 and worst["other"] <= C.F0_ACCEPT_MAX)
    over = [dict(rxn_id=r["rxn_id"], col=r.get("worst_col"), diff=r.get("max_other")) for r in out
            if (r.get("max_other") or 0) > C.F0_TOL]
    rep = dict(rxn_ids=ids, rows=out, max_abs_diff=worst, tol=dict(B1=C.F0_TOL_B1, other=C.F0_TOL), pass_strict=strict,
               over_tolerance=over, user_decision=None if strict else C.F0_DECISION, versions=v0, pass_=ok, stop=not ok)
    C.write_json_safe(C.RES / "F0_report.json", rep)
    print(json.dumps(rep, indent=1, default=str))
    return 0 if ok else 3


def _f0run(ids):
    import rev5_common as R5
    ids = [int(x) for x in ids.split(",")]
    run_ids(ids, "f0chk", Path(os.environ["ESPLEY_OUT"]) / "f_slices", R5.SCRATCH / "ext",
            int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    return 0


# ================================================================ pair sets (§2-5)
def p4_v2(pairs):
    """stereo rule v2 of d1_phase_p for (a_job, b_job) pairs -> list of v2 status."""
    import d1_phase_p as PP
    lab = {d["job_id"]: d for d in C.load_d1_labels()}
    cache, out = {}, []
    for a, b in pairs:
        for j in (a, b):
            if j not in cache:
                try:
                    cache[j] = PP.ring_info(j, lab[j])
                except Exception as e:                                  # noqa: BLE001
                    cache[j] = f"{type(e).__name__}:{e}"
        ia, ib = cache[a], cache[b]
        out.append("ring_fail" if isinstance(ia, str) or isinstance(ib, str) else PP.compare(ia, ib)["v2"])
    return out


def cmd_pairs():
    x = pd.read_parquet(C.X182_D1, columns=["rxn_id"])
    have = set(x.rxn_id.astype(int))
    r = pd.read_pickle(C.ROWATTR)
    okh = dict(zip(r.job_id, C.hygiene_mask(r.d1_kcal, r.d2_kcal)))
    feat_job = {j for j, rid in zip(r.job_id, r.rxn_id) if int(rid) in have}
    drop = pd.read_csv(C.P4_DROP)
    dropk = set(zip(drop.a_job, drop.b_job))
    P = pd.read_csv(C.PAIRS).assign(ref_kind="reference")
    P["feat"] = P.a_job.isin(feat_job) & P.b_job.isin(feat_job)
    P["hyg"] = [okh[a] and okh[b] for a, b in zip(P.a_job, P.b_job)]
    P["p4_drop"] = [(a, b) in dropk for a, b in zip(P.a_job, P.b_job)]
    P["in_P_all"] = True
    P["in_P_sens"] = P.feat & ~P.p4_drop
    P["in_P_feat"] = P.in_P_sens
    P["in_P_main"] = P.in_P_sens & P.hyg
    # A_feat: appendix E rerun on the rows with the 182 features (§2-5; user decision 2026-10-19 C.AFEAT_DECISION): its
    # pairs file (used by its overlap assert) = the reference pairs that exist among the feature rows (P_feat), so a family
    # whose reference has no features becomes an anchor family; its own stdout appended to the appendix log
    rf = r[r.job_id.isin(feat_job)]
    pk = C.SCRATCH / "P" / "rowattr_feat.pkl"
    C.write_atomic(pk, lambda t: rf.to_pickle(t))
    pf = C.SCRATCH / "P" / "pairs_d1_feat.csv"
    pcols = list(pd.read_csv(C.PAIRS, nrows=0).columns)
    C.write_atomic(pf, lambda t: P.loc[P.feat, pcols].to_csv(t, index=False))
    out = C.RES / "pairs_d1_anchor_feat.csv"
    res = subprocess.run([sys.executable, "-I", str(HERE / "d1_anchor_pairs.py"), str(pk), str(C.SIGMA), str(pf),
                          str(out)], capture_output=True, text=True)
    log = C.RES / "P_appendix_stdout.txt"
    C.write_atomic(log, log.read_text() + f"\n$ python -I d1_anchor_pairs.py rowattr_feat.pkl sigma_hansch1991.csv "
                   f"pairs_d1_feat.csv pairs_d1_anchor_feat.csv   (A_feat, exit {res.returncode})\n{res.stdout}{res.stderr}")
    if res.returncode != 0:
        print(f"STOP: A_feat appendix E: {res.stderr[-2000:]}", file=sys.stderr); return 3
    A = pd.read_csv(out)
    if (set(A.a_job) | set(A.b_job)) & (set(P.loc[P.feat, "a_job"]) | set(P.loc[P.feat, "b_job"])):
        print("STOP: A_feat rows overlap the P_feat reference pairs", file=sys.stderr); return 3
    Aall = pd.read_csv(C.ANCHOR)
    A["hyg"] = [okh[a] and okh[b] for a, b in zip(A.a_job, A.b_job)]
    v2 = p4_v2(list(zip(A.a_job, A.b_job)))
    A["p4_v2"] = v2
    A["p4_drop"] = [(a, b) in dropk or s == "mismatch" for (a, b), s in zip(zip(A.a_job, A.b_job), v2)]
    A["in_A_feat"] = ~A.p4_drop
    A["in_A_main"] = A.in_A_feat & A.hyg
    A["feat"], A["in_P_all"] = True, False
    keep = ["pair_id", "ref_kind", "a_job", "a_rxn", "b_job", "b_rxn", "Xb", "pos", "dipole", "skeleton", "series", "cfg",
            "orient", "face", "c3", "fam", "core", "sigma"] + [f"dd_{t}" for t in C.TARGETS]
    T = pd.concat([P[keep + ["feat", "hyg", "p4_drop", "in_P_all", "in_P_feat", "in_P_main", "in_P_sens"]],
                   A[keep + ["anchor_Xb", "anchor_pos", "feat", "hyg", "p4_drop", "in_A_feat", "in_A_main"]]],
                  ignore_index=True)
    for c in ("in_P_all", "in_P_feat", "in_P_main", "in_P_sens", "in_A_feat", "in_A_main"):
        T[c] = T[c].fillna(False).astype(bool)
    for c in ("cfg", "orient", "face", "c3"):
        T[c] = T[c].fillna("").astype(str)
    C.write_atomic(C.RES / "F_pairs.csv", lambda t: T.to_csv(t, index=False))
    # flows
    aall_k = set(zip(Aall.a_job, Aall.b_job))
    afeat_k = set(zip(A.a_job, A.b_job))
    flow = dict(P_all=len(P), P_feat_missing_feature=int((~P.feat).sum()), P_p4_drop=int((P.feat & P.p4_drop).sum()),
                P_feat=int(P.in_P_feat.sum()), P_hygiene_dropped=int((P.in_P_feat & ~P.hyg).sum()),
                P_main=int(P.in_P_main.sum()), P_sens=int(P.in_P_sens.sum()),
                A_all=len(Aall), A_all_rows_without_feature=int(sum(not (a in feat_job and b in feat_job) for a, b in aall_k)),
                A_feat_rerun=len(A), A_feat_same_pairs_as_A_all=len(afeat_k & aall_k), A_feat_new_pairs=len(afeat_k - aall_k),
                A_p4_drop=int(A.p4_drop.sum()), A_p4_v2=pd.Series(v2).value_counts().to_dict(),
                A_feat=int(A.in_A_feat.sum()), A_hygiene_dropped=int((A.in_A_feat & ~A.hyg).sum()), A_main=int(A.in_A_main.sum()))
    ach = Aall.groupby("fam").b_job.agg(lambda s: tuple(sorted(set(s))))
    acf = A.groupby("fam").b_job.agg(lambda s: tuple(sorted(set(s))))
    changed = sorted(f for f in acf.index if f in ach.index and ach[f] != acf[f])
    loss = 1 - P.groupby("core").in_P_feat.mean()
    reac = set(P.loc[P.in_P_main, "a_job"]) | set(P.loc[P.in_P_main, "b_job"]) | set(A.loc[A.in_A_main, "a_job"]) | \
        set(A.loc[A.in_A_main, "b_job"])
    over = sorted(c for c, v in loss.items() if v > C.CORE_LOSS_STOP)
    newf = sorted(set(A.fam) - set(Aall.fam))
    lost_ref = P[~P.feat].groupby("fam").b_job.agg(lambda s: sorted(set(s))).to_dict()
    rep = dict(flow=flow, anchor_changed_families=changed, n_anchor_changed=len(changed),
               A_feat_new_families=newf, A_feat_new_family_pairs=int(A.fam.isin(newf).sum()),
               A_feat_new_family_lost_reference={f: lost_ref.get(f.replace("/anchor", ""), []) for f in newf},
               core_ref_pair_loss=loss.round(4).to_dict(), core_ref_pair_loss_max=float(loss.max()),
               cores_losing_over_half=over, core_loss_user_decision=C.CORE_LOSS_DECISION, afeat_decision=C.AFEAT_DECISION,
               stop_core_loss=bool(set(over) - set(C.CORE_LOSS_ACCEPTED)), d1_reactions_in_T_main_pairs=len(reac),
               P_main_pos=P[P.in_P_main].pos.value_counts().to_dict(), A_main_pos=A[A.in_A_main].pos.value_counts().to_dict(),
               sha256=dict(F_pairs=C.sha256(C.RES / "F_pairs.csv"), A_feat=C.sha256(out)),
               p4_decision=C.P4_DECISION)
    rep["stop"] = rep["stop_core_loss"]
    fr = json.loads((C.RES / "F_report.json").read_text())
    fr["pairs"] = rep
    C.write_json_safe(C.RES / "F_report.json", fr)
    print(json.dumps(rep, indent=1, default=str))
    return 3 if rep["stop"] else 0


if __name__ == "__main__":
    a = sys.argv[1:]
    cmd = a[0] if a else ""
    if cmd == "run":
        sys.exit(cmd_run(int(a[1]), int(a[2])))
    if cmd == "_f0run":
        sys.exit(_f0run(a[1]))
    fn = {"merge": cmd_merge, "f0check": cmd_f0check, "pairs": cmd_pairs}.get(cmd)
    if fn is None:
        raise SystemExit(__doc__)
    sys.exit(fn())
