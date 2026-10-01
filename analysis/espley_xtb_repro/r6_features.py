#!/usr/bin/env python3
"""r6_features.py — rev 6 R-3 / R-4 features and the R-5 row table (results_rev6/R_plan.md §5-§6).

  python r6_features.py g1feat     # rev 4 xtb_slice.process_one on every recovered rxn -> FEAT_G1_R6
  python r6_features.py ext        # rev 5 ext_features.run_rxn (r5-ext-3) on the recovered rxns -> FEAT_EXT_R6
  python r6_features.py ext4994    # R-4: rxn 4994 on its rev 4 geometry, B1 gate tol 1e-5 -> FEAT_EXT_4994
  python r6_features.py report     # R-5: row flow R_esp -> R_lab -> R_feat -> C, R_rows_final.csv, R5_report.json

The rev 4 / rev 5 code runs unchanged; the environment (set by r6_F.sh BEFORE python starts, so the spawned ext
workers inherit it) points it at the rev 6 inputs:
  g1feat / ext   G1_ROOT=$G1_ROOT_R6, ESPLEY_PARTITION=label (R_plan §2)
  ext4994        G1_ROOT=rev 4 root, ESPLEY_PARTITION=redetect (= rev 5), R6_B1_TOL=4994:1e-5
The rows that passed rev 4 keep their rev 4 / rev 5 parquet values (not recomputed). Outputs are new files only.
Exit codes: 0 ok, 2 bad inputs / environment, 3 STOP.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rev6_common as R6  # noqa: E402

R5 = R6.R5
REV4_G1_ROOT = Path("/gpfs/tmp_cpu2/yeseo1ee/espley_xtb_g1")
FEAT_G1_REV4 = Path("/gpfs/tmp_cpu2/yeseo1ee/espley_xtb/xtb_features_g1.parquet")       # read-only
FEAT_EXT_REV5 = Path("/gpfs/tmp_cpu2/yeseo1ee/espley_xtb/xtb_features_ext_g1.parquet")  # read-only
EXT_CACHE = R6.SCRATCH / "ext"
EXT_CACHE_4994 = R6.SCRATCH / "ext_4994"


def need_env(**kw):
    bad = {k: (os.environ.get(k), v) for k, v in kw.items() if os.environ.get(k) != v}
    if bad:
        print(f"environment not as required (got, want): {bad}", file=sys.stderr)
        sys.exit(2)


def recovered_ids():
    """rxn_ids with a rev 6 .done under G1_ROOT_R6."""
    return sorted(int(p.parent.name) for p in R6.G1_ROOT_R6.glob("*/.done"))


_LABELS = None


def _proc_one(rid):
    global _LABELS
    import xtb_slice as xs
    if _LABELS is None:
        _LABELS = {int(d["rxn_id"]): d for d in json.loads(Path(R5.LABELS).read_text())}
    return xs.process_one(_LABELS[rid], "g1")


def cmd_g1feat():
    need_env(G1_ROOT=str(R6.G1_ROOT_R6), ESPLEY_PARTITION="label")
    out = R6.FEAT_G1_R6
    if out.exists():
        print(f"{out} exists - skip"); return 0
    ids = recovered_ids()
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    print(f"[g1feat] {len(ids)} recovered rxns, {workers} workers, G1_ROOT={os.environ['G1_ROOT']}", flush=True)
    with mp.get_context("spawn").Pool(workers) as pool:
        rows = pool.map(_proc_one, ids, chunksize=1)
    df = pd.DataFrame(rows)
    ref_cols = list(pd.read_parquet(FEAT_G1_REV4).columns)
    miss, extra = sorted(set(ref_cols) - set(df.columns)), sorted(set(df.columns) - set(ref_cols))
    # a failed row only has the columns up to its failure: add them as NaN; any unknown column is an error
    if extra:
        print(f"STOP: columns not in the rev 4 parquet: {extra}", file=sys.stderr); return 3
    df = df.reindex(columns=ref_cols)
    R5.write_atomic(out, lambda t: df.to_parquet(t, index=False))
    rep = dict(n=len(df), xtb_status=df["xtb_status"].value_counts().to_dict(), missing_cols_filled=miss[:20],
               sha256=R6.sha256(out))
    R6.write_json(R6.RES6 / "R3_g1feat.json", rep)
    print(json.dumps(rep, indent=1, default=str))
    return 0


def _ext(ids, g1_parquet, cache_root, out, tag):
    import ext_features as ef
    labels = ef.load_labels()
    g1 = pd.read_parquet(g1_parquet, columns=ef.G1_COLS)
    g1 = {int(r["rxn_id"]): r for r in g1.to_dict("records")}
    tasks = [(labels[r], g1.get(r)) for r in ids if not ef.rxn_complete(labels[r], cache_root, list(R5.BLOCK_ORDER))]
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    print(f"[{tag}] {len(ids)} rxns, {len(tasks)} to compute, {workers} workers, cache {cache_root}", flush=True)
    ef.run_pool(tasks, cache_root, workers, list(R5.BLOCK_ORDER), False, tag=tag)
    df = pd.DataFrame([ef.assemble_row(labels[r], cache_root) for r in ids], columns=ef.ROW_COLS)
    R5.write_atomic(out, lambda t: df.to_parquet(t, index=False))
    return df, ef.versions()


def cmd_ext():
    need_env(G1_ROOT=str(R6.G1_ROOT_R6), ESPLEY_PARTITION="label")
    if not R6.FEAT_G1_R6.is_file():
        print("run g1feat first", file=sys.stderr); return 2
    if R6.FEAT_EXT_R6.exists():
        print(f"{R6.FEAT_EXT_R6} exists - skip"); return 0
    g1 = pd.read_parquet(R6.FEAT_G1_R6, columns=["rxn_id", "xtb_status"])
    ids = sorted(int(r) for r in g1.loc[g1.xtb_status == "ok", "rxn_id"])
    df, ver = _ext(ids, R6.FEAT_G1_R6, EXT_CACHE, R6.FEAT_EXT_R6, "ext-r6")
    rep = dict(n=len(df), ext_status=df["ext_status"].value_counts().to_dict(), versions=ver,
               sha256=R6.sha256(R6.FEAT_EXT_R6))
    R6.write_json(R6.RES6 / "R3_ext.json", rep)
    print(json.dumps(rep, indent=1, default=str))
    return 0


def cmd_ext4994():
    need_env(G1_ROOT=str(REV4_G1_ROOT), ESPLEY_PARTITION="redetect")
    import ext_features as ef
    if ef.B1_TOL_OVERRIDE != {R6.R4_RXN: R6.R4_TOL_EH}:
        print(f"R6_B1_TOL must give {{{R6.R4_RXN}: {R6.R4_TOL_EH}}}, got {ef.B1_TOL_OVERRIDE}", file=sys.stderr)
        return 2
    if R6.FEAT_EXT_4994.exists():
        print(f"{R6.FEAT_EXT_4994} exists - skip"); return 0
    df, ver = _ext([R6.R4_RXN], FEAT_G1_REV4, EXT_CACHE_4994, R6.FEAT_EXT_4994, "ext-4994")
    old = pd.read_parquet(FEAT_EXT_REV5)
    old = old[old.rxn_id == R6.R4_RXN].iloc[0]
    new = df.iloc[0]
    same = {}
    for b in R5.BLOCK_ORDER:
        if b == "B1":
            continue
        diff = [abs(float(new[c]) - float(old[c])) for c in R5.BLOCKS[b]]
        same[b] = float(max(diff))
    rep = dict(rxn_id=R6.R4_RXN, ext_status=str(new["ext_status"]), status_B1=str(new["status_B1"]),
               qc_b1_dE_ts_eh=float(new["qc_b1_dE_ts_eh"]), tol_eh=R6.R4_TOL_EH,
               rev5_ext_status=str(old["ext_status"]), max_absdiff_vs_rev5_other_blocks=same, versions=ver,
               recovered_by="gate_tol_user_decision")
    R6.write_json(R6.RES6 / "R4_rxn4994.json", rep)
    print(json.dumps(rep, indent=1, default=str))
    return 0 if rep["ext_status"] == "ok" else 3


def cmd_report():
    import train_ml_single as T
    cols = R6.our_columns(T.FEATURE_SETS)
    labels = json.loads(Path(R5.LABELS).read_text())
    lab = pd.DataFrame([dict(rxn_id=int(d["rxn_id"]), status=d["status"], d1=d.get("d1_kcal"), d2=d.get("d2_kcal"))
                        for d in labels])
    esp = pd.read_pickle(R6.ESP_DS)["reaction_number"].astype(int).tolist()
    assert len(esp) == R6.N_ESP and len(set(esp)) == R6.N_ESP, len(esp)
    # features: rev 4 / rev 5 rows, then the rev 6 recovered rows and rxn 4994
    g4, e5 = pd.read_parquet(FEAT_G1_REV4), pd.read_parquet(FEAT_EXT_REV5)
    g6 = pd.read_parquet(R6.FEAT_G1_R6) if R6.FEAT_G1_R6.is_file() else g4.iloc[:0]
    e6 = pd.read_parquet(R6.FEAT_EXT_R6) if R6.FEAT_EXT_R6.is_file() else e5.iloc[:0]
    e4994 = pd.read_parquet(R6.FEAT_EXT_4994) if R6.FEAT_EXT_4994.is_file() else e5.iloc[:0]
    rec = pd.read_csv(R6.RES6 / "R_rows.csv") if (R6.RES6 / "R_rows.csv").is_file() else None
    if rec is None or not R6.FEAT_G1_R6.is_file() or not R6.FEAT_EXT_R6.is_file() or not R6.FEAT_EXT_4994.is_file():
        print("report needs R_rows.csv (r6_recover summary), g1feat, ext and ext4994 outputs", file=sys.stderr)
        return 2
    overlap = set(g6.rxn_id) & set(g4.loc[g4.xtb_status == "ok", "rxn_id"])
    if overlap:
        print(f"STOP: recovered rxns already ok in rev 4: {sorted(overlap)[:10]}", file=sys.stderr); return 3
    g = pd.concat([g4[~g4.rxn_id.isin(g6.rxn_id)].assign(feat_src="rev4"), g6.assign(feat_src="rev6")])
    e = pd.concat([e5[~e5.rxn_id.isin(set(e6.rxn_id) | {R6.R4_RXN})].assign(ext_src="rev5"),
                   e6.assign(ext_src="rev6"), e4994.assign(ext_src="rev6_4994")])
    gcols = [c for c in cols if c in g.columns]
    ecols = [c for c in cols if c not in g.columns]
    assert set(ecols) <= set(e.columns), sorted(set(ecols) - set(e.columns))[:5]
    m = lab.merge(g[["rxn_id", "xtb_status", "feat_src"] + gcols], on="rxn_id", how="left") \
        .merge(e[["rxn_id", "ext_status", "ext_src"] + ecols], on="rxn_id", how="left")
    rb = dict(zip(rec.rxn_id, rec.recovered_by.where(rec.status == "ok")))
    m["recovered_by"] = [rb.get(r) if r in rb else ("g1_rev4" if s == "ok" else None)
                         for r, s in zip(m.rxn_id, m.xtb_status)]
    m.loc[m.rxn_id == R6.R4_RXN, "recovered_by"] = "gate_tol_user_decision"
    m["in_esp"] = m.rxn_id.isin(esp)
    m["R_lab"] = m.status == "ok"
    m["nan182"] = m[cols].isna().sum(axis=1)
    m["R_feat"] = m.R_lab & (m.xtb_status == "ok") & (m.ext_status == "ok") & (m.nan182 == 0)
    m["R_hyg"] = R6.hygiene_mask(m.d1, m.d2)
    m["C_all"] = m.in_esp & m.R_lab & m.R_feat
    m["C"] = m.C_all & m.R_hyg
    # reason a labelled row is not in R_feat
    g1_fail = dict(zip(rec.rxn_id, rec.status))

    def why(r):
        if not r.R_lab:
            return "label_not_ok"
        if r.R_feat:
            return "in_R_feat"
        if r.rxn_id in g1_fail and g1_fail[r.rxn_id] != "ok":
            return "g1_" + str(g1_fail[r.rxn_id])
        if r.xtb_status != "ok":
            return f"g1feat:{r.xtb_status}"
        if r.ext_status != "ok":
            return f"ext:{r.ext_status}"
        return "nan_in_182"
    m["not_feat_reason"] = [why(r) for r in m.itertuples()]
    keep = ["rxn_id", "in_esp", "R_lab", "R_feat", "R_hyg", "C", "C_all", "recovered_by", "feat_src", "ext_src",
            "xtb_status", "ext_status", "not_feat_reason", "d1", "d2"]
    out = m[keep].sort_values("rxn_id")
    R6.write_atomic(R6.ROWS_R, lambda t: out.to_csv(t, index=False))
    es = m[m.in_esp]
    flow = dict(R_esp=int(R6.N_ESP), R_esp_in_labels=int(len(es)), R_lab=int(es.R_lab.sum()),
                R_feat=int((es.R_lab & es.R_feat).sum()), C_all=int(es.C_all.sum()), C=int(es.C.sum()),
                hygiene_dropped=int((es.C_all & ~es.R_hyg).sum()),
                not_in_R_feat_reasons=es.loc[es.R_lab & ~es.R_feat, "not_feat_reason"].value_counts().to_dict(),
                recovered_in_C=es.loc[es.C, "recovered_by"].value_counts().to_dict())
    allrows = dict(R_lab=int(m.R_lab.sum()), R_feat=int(m.R_feat.sum()), R_feat_hyg=int((m.R_feat & m.R_hyg).sum()),
                   not_in_R_feat_reasons=m.loc[m.R_lab & ~m.R_feat, "not_feat_reason"].value_counts().to_dict(),
                   recovered_by=m.loc[m.R_feat, "recovered_by"].value_counts().to_dict())
    g4sum = json.loads((R5.RES4 / "g1_geometry_summary.json").read_text())
    r2 = json.loads((R6.RES6 / "R2_report.json").read_text())
    rep = dict(flow_espley=flow, all_labels=allrows, rows_sha256=R6.sha256(R6.ROWS_R),
               geometry_original_success={k: g4sum.get(k) for k in ("rmsd_ts_heavy_A", "dd_form_short_A",
                                                                     "dd_form_long_A")},
               geometry_recovered={k: r2.get(k + "_recovered") for k in ("rmsd_ts_heavy_A", "dd_form_short_A",
                                                                         "dd_form_long_A")},
               inputs=dict(g1_rev4=R6.sha256(FEAT_G1_REV4), ext_rev5=R6.sha256(FEAT_EXT_REV5),
                           g1_rev6=R6.sha256(R6.FEAT_G1_R6), ext_rev6=R6.sha256(R6.FEAT_EXT_R6),
                           ext_4994=R6.sha256(R6.FEAT_EXT_4994), labels=R6.sha256(R5.LABELS)))
    R6.write_json(R6.RES6 / "R5_report.json", rep)
    print(json.dumps(rep, indent=1, default=str))
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    fn = {"g1feat": cmd_g1feat, "ext": cmd_ext, "ext4994": cmd_ext4994, "report": cmd_report}.get(cmd)
    if fn is None:
        raise SystemExit(__doc__)
    sys.exit(fn())
