#!/usr/bin/env python3
"""d1_recover.py — D1PAIR §7-4: rev 6 Phase R ladders (results_rev6/R_plan.md) on the D1 G1 failures.

  python d1_recover.py run <i> <n>    # slice i of n of the first-attempt failures (references -> anchors -> job_id)
  python d1_recover.py infra          # reset rows whose ladder had an ORCA MPI start-up abort (rerun the same ladder)
  python d1_recover.py summary        # results/G_attempts.csv, G_by_reason.csv, G_rows.csv, G_report.json (exit 3 = STOP)
                                      # + $D1PAIR_SCRATCH/g1_final/<rxn_id> -> the g1 or g1_rec folder of every G1 row

The ladder runner and every gate are r6_recover's own code (LADDERS, REF_STEPS, the R_plan settings, parse_ts,
evaluate = the g1_geom.process gate block with the rev 6 label partition, run_ts_attempt, mode_start, xtb_opt_args,
process): r6_recover.process is called per rxn with its two module roots pointing at D1PAIR scratch (G1_ROOT_R6 ->
OUT_ROOT = $D1PAIR_SCRATCH/g1_rec, G1_ROOT_REV4 -> REV4_ROOT = $D1PAIR_SCRATCH/g1 = the first D1 G1 attempt, the start
structures). r6_recover's own commands (inventory / run / summary) are never called: they write results_rev6/.
Its targets() / load_labels() are not used either; the targets are the failures under $D1PAIR_SCRATCH/g1.
STOP (§7-4, §13): >= 16 of the 157 reference reactions fail after recovery; a core loses more than half of its
reference pairs; an attempt used a setting that is not in R_plan (r6_recover.cmd_summary's check, repeated here);
a gate record is missing; partition_label_map_failed.
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import d1pair_common as C  # noqa: E402
import d1_g1 as DG  # noqa: E402


def rx():
    DG.check_env()
    import r6_recover as RX
    if Path(RX.OUT_ROOT) != C.G1_ROOT_D1R or Path(RX.REV4_ROOT) != C.G1_ROOT_D1:
        print(f"STOP: r6_recover roots {RX.OUT_ROOT}, {RX.REV4_ROOT} are not the D1PAIR scratch", file=sys.stderr)
        sys.exit(3)
    C.assert_write_ok(RX.OUT_ROOT)
    return RX


def first_results():
    """rxn_id -> first-attempt result.json for every G1 row (run order); exits 2 if one has no marker."""
    out, unmarked = {}, []
    for lab in DG.targets_ok():
        rid = int(lab["rxn_id"])
        d = C.G1_ROOT_D1 / str(rid)
        if not ((d / ".done").is_file() or list(d.glob(".fail_*"))):
            unmarked.append(rid); continue
        out[rid] = json.loads((d / "result.json").read_text())
    if unmarked:
        print(f"GATE: {len(unmarked)} G1 rows without a marker under {C.G1_ROOT_D1}: {unmarked[:10]}", file=sys.stderr)
        sys.exit(2)
    return out


def cmd_run(i, n):
    RX = rx()
    import g1_geom as G
    first = first_results()
    labs = {int(d["rxn_id"]): d for d in DG.targets_ok()}
    order = [r for r in labs if first[r].get("status") != "ok"]          # targets_ok() is already in run order
    mine = order[i::n]
    smiles, meta = G.load_smiles(), G.load_meta()
    workers = int(os.environ.get("R6_WORKERS", max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "2")) // RX.NPROCS)))
    print(f"[d1 R slice {i}/{n}] {len(mine)} of {len(order)} failures, {workers} workers; out {RX.OUT_ROOT}", flush=True)
    C.G1_ROOT_D1R.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(RX.process, labs[r], first[r], smiles, meta) for r in mine]
        for k, f in enumerate(futs):
            rid, st = f.result()
            print(f"[{k + 1}/{len(futs)}] rxn {rid}: {st}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    return 0


def cmd_infra():
    """recovery rows whose ladder had an ORCA run that died in orca_util before any cycle (MPI start-up, not
    chemistry): marker / result.json / attempts.json moved to g1_rec_infra/<rid>.<n>/, so `run` repeats the same ladder;
    r6_recover.orca_run reuses every normally terminated step and reruns the aborted one."""
    rx()
    dest = C.SCRATCH / "g1_rec_infra"
    C.assert_write_ok(dest)
    rows = []
    for d in sorted(C.G1_ROOT_D1R.iterdir()):
        marks = list(d.glob(".fail_*"))
        if not marks:
            continue
        bad = []
        for out in d.rglob("*.out"):
            if out.parent.name.startswith("ref"):
                continue
            t = out.read_text(errors="replace")
            if "error termination in orca_util" in t and "OPTIMIZATION CYCLE" not in t:
                bad.append(str(out.relative_to(d)))
        if not bad:
            continue
        k = len(list(dest.glob(f"{d.name}.*")))
        tgt = dest / f"{d.name}.{k}"
        tgt.mkdir(parents=True)
        for f in marks + [d / "result.json", d / "attempts.json"]:
            if f.exists():
                os.replace(f, tgt / f.name)
        rows.append(dict(rxn_id=int(d.name), aborted_runs=";".join(bad), moved_to=str(tgt), time=time.strftime("%Y-%m-%dT%H:%M:%S")))
    f = C.RES / "G_infra_retry_rec.csv"
    old = pd.read_csv(f) if f.is_file() else pd.DataFrame()
    C.write_atomic(f, lambda t: pd.concat([old, pd.DataFrame(rows)], ignore_index=True).to_csv(t, index=False))
    print(f"[rec infra] {len(rows)} rows reset for a rerun of the same ladder: {[r['rxn_id'] for r in rows]}")
    return 0


def cmd_summary():
    RX = rx()
    first = first_results()
    prep = pd.read_csv(DG.PREP_CSV).set_index("rxn_id")
    rows, atts, unmarked = [], [], []
    for rid, r1 in first.items():
        base = dict(rxn_id=rid, job_id=prep.loc[rid, "job_id"], role=prep.loc[rid, "role"], core=prep.loc[rid, "core"],
                    first_status=r1.get("status"), first_reasons=";".join(r1.get("fail_reasons") or []))
        if r1.get("status") == "ok":
            dd = r1.get("dd_form") or [np.nan, np.nan]
            rows.append(dict(base, status="ok", recovered_by="g1_first", fail_reasons="",
                             rmsd_ts_heavy_A=r1.get("rmsd_ts_heavy_A"), dd_form_short_A=dd[0], dd_form_long_A=dd[1],
                             g1_dir=str(C.G1_ROOT_D1 / str(rid))))
            continue
        d = C.G1_ROOT_D1R / str(rid)
        if not ((d / ".done").is_file() or list(d.glob(".fail_*"))):
            unmarked.append(rid); continue
        r = json.loads((d / "result.json").read_text())
        a = json.loads((d / "attempts.json").read_text()) if (d / "attempts.json").is_file() else []
        atts += a
        dd = r.get("dd_form") or [np.nan, np.nan]
        rows.append(dict(base, status=r["status"], recovered_by=r.get("recovered_by") if r["status"] == "ok" else None,
                         fail_reasons=";".join(r.get("fail_reasons") or []), ts_ladder=r.get("ts_ladder"),
                         rmsd_ts_heavy_A=r.get("rmsd_ts_heavy_A"), dd_form_short_A=dd[0], dd_form_long_A=dd[1],
                         map_source=(r.get("partition_diag") or {}).get("map_source"),
                         gate_keys_missing=";".join(r.get("gate_keys_missing") or []), wall_s=r.get("wall_s"),
                         g1_dir=str(d)))
    if unmarked:
        print(f"GATE: {len(unmarked)} failures without a recovery marker under {C.G1_ROOT_D1R}: {unmarked[:10]}",
              file=sys.stderr)
        return 2
    df, at = pd.DataFrame(rows), pd.DataFrame(atts)
    C.write_atomic(C.RES / "G_rows.csv", lambda t: df.to_csv(t, index=False))
    C.write_atomic(C.RES / "G_attempts.csv", lambda t: at.to_csv(t, index=False))
    fail1 = df[df.first_status != "ok"].copy()
    fail1["category"] = fail1.first_status.str.replace("fail:", "", regex=False)
    fail1["first_method"] = fail1.recovered_by.fillna("").str.split(";").str[0]
    if len(fail1):
        tab = fail1.pivot_table(index="category", columns="first_method", values="rxn_id", aggfunc="count", fill_value=0)
        tab["n_first_fail"] = fail1.groupby("category").size()
        tab["n_recovered"] = fail1[fail1.status == "ok"].groupby("category").size().reindex(tab.index).fillna(0).astype(int)
        tab["n_remaining"] = tab["n_first_fail"] - tab["n_recovered"]
        tab = tab.drop(columns=[""], errors="ignore").reset_index()
    else:
        tab = pd.DataFrame()
    C.write_atomic(C.RES / "G_by_reason.csv", lambda t: tab.to_csv(t, index=False))
    # r6_recover.cmd_summary STOP checks
    allowed = {s for lad in RX.LADDERS.values() for _, _, s in lad} | {s for _, s in RX.REF_STEPS} | {"rev4"}
    bad_setting = sorted(set(at.get("setting", pd.Series(dtype=str)).dropna()) - allowed)
    hdr_ok = {RX.H_COMMON, RX.H_EXACT, RX.H_EXACT_TIGHT}
    bad_hdr = [h for h in at.get("header", pd.Series(dtype=str)).dropna().unique()
               if h not in hdr_ok and not h.startswith("! XTB2 ALPB(water) Opt\n%geom\n Constraints\n")]
    rec = df[df.first_status != "ok"]
    missing = rec.loc[(rec.status == "ok") & (rec.gate_keys_missing.fillna("") != ""), "rxn_id"].tolist()
    lmf = rec.loc[rec.fail_reasons.str.contains("partition_label_map_failed", na=False), "rxn_id"].tolist()
    ok = df.status == "ok"
    okj = set(df.loc[ok, "job_id"])
    P = pd.read_csv(C.PAIRS)
    P["kept"] = P.a_job.isin(okj) & P.b_job.isin(okj)
    core_loss = P.groupby("core").kept.agg(lambda s: 1 - s.mean())
    ref_fail = int(((df.role == "reference") & ~ok).sum())
    rep = dict(n=len(df), n_ok_first=int((df.first_status == "ok").sum()), n_first_fail=int(len(rec)),
               n_recovered=int(((df.first_status != "ok") & ok).sum()), n_ok=int(ok.sum()), n_fail=int((~ok).sum()),
               recovered_by=df.loc[ok & (df.first_status != "ok"), "recovered_by"].value_counts().to_dict(),
               remaining_reasons=df.loc[~ok, "status"].value_counts().to_dict(),
               first_fail_reasons=rec.first_status.value_counts().to_dict(),
               ref_fail=ref_fail, ref_fail_rows=df.loc[(df.role == "reference") & ~ok, ["job_id", "status"]].to_dict("records"),
               anchor_fail=int(((df.role == "anchor") & ~ok).sum()),
               core_ref_pair_loss_max=float(core_loss.max()), core_ref_pair_loss=core_loss.round(4).to_dict(),
               ref_pairs_kept=int(P.kept.sum()), n_attempts=len(at),
               settings_used=at.get("setting", pd.Series(dtype=str)).value_counts().to_dict(),
               stop_setting_not_in_plan=bad_setting + bad_hdr, stop_gate_record_missing=missing,
               stop_label_map_failed=lmf)
    for c in ("rmsd_ts_heavy_A", "dd_form_short_A", "dd_form_long_A"):
        for tag, m in (("first", df.first_status == "ok"), ("recovered", ok & (df.first_status != "ok"))):
            v = df.loc[m, c].astype(float)
            a = v.abs() if c.startswith("dd") else v
            rep[f"{c}_{tag}"] = dict(n=int(v.notna().sum()), median=float(v.median()) if len(v) else None,
                                     median_abs=float(a.median()) if len(v) else None,
                                     p90_abs=float(a.quantile(0.9)) if len(v) else None,
                                     max_abs=float(a.max()) if len(v) else None)
    rep["stop_ref_fail"] = ref_fail >= C.G_REF_FAIL_STOP
    over = sorted(c for c, v in core_loss.items() if v > C.CORE_LOSS_STOP)
    rep["cores_losing_over_half"] = over
    rep["core_loss_user_decision"] = C.CORE_LOSS_DECISION if set(over) & set(C.CORE_LOSS_ACCEPTED) else None
    rep["stop_core_loss"] = bool(set(over) - set(C.CORE_LOSS_ACCEPTED))
    rep["stop"] = bool(bad_setting or bad_hdr or missing or lmf or rep["stop_ref_fail"] or rep["stop_core_loss"])
    # g1_final: one folder per G1 row (features read here)
    C.assert_write_ok(C.G1_ROOT_D1F)
    C.G1_ROOT_D1F.mkdir(parents=True, exist_ok=True)
    for r in df.itertuples():
        link = C.G1_ROOT_D1F / str(r.rxn_id)
        if link.is_symlink() and os.readlink(link) != r.g1_dir:
            link.unlink()
        if not link.is_symlink():
            os.symlink(r.g1_dir, link)
    rep["g1_final"] = dict(root=str(C.G1_ROOT_D1F), n_links=len(df), n_done=int(ok.sum()))
    C.write_json_safe(C.RES / "G_report.json", rep)
    print(json.dumps(rep, indent=1, default=str))
    return 3 if rep["stop"] else 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["run"]:
        sys.exit(cmd_run(int(a[1]), int(a[2])))
    if a[:1] == ["infra"]:
        sys.exit(cmd_infra())
    if a[:1] == ["summary"]:
        sys.exit(cmd_summary())
    raise SystemExit(__doc__)
