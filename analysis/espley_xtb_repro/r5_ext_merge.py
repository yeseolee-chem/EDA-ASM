#!/usr/bin/env python3
"""r5_ext_merge.py — rev 5 Phase B-7 (docs/specs/REV5_FEATURES_FIGURES.md): merge the ext_features.py slices,
apply the B-7 gates and fix the rev 5 rows.

  python r5_ext_merge.py [--force] [--slices-dir DIR] [--n-slices 18]

1. Reads <slices-dir>/slice_00..17.parquet (default $R5_SCRATCH/ext/slices; every slice must exist, else exit 2).
   Their union must be exactly the G1-ok rxns (accepted labels with $G1_ROOT/<rid>/.done), one row each, geom 'g1',
   every ext_features.ROW_COLS column present (else exit 2).
2. Writes rev5_common.FEAT_EXT ($ESPLEY_OUT/xtb_features_ext_g1.parquet) atomically, ascending rxn_id. An existing
   file with the same content is left untouched; a different one is replaced only with --force (else it is kept,
   rows_rev5.csv is not written and the run exits 3 after the report).
3. B-7 gates:
     fail share   rows_rev4 rows whose ext_status != ok (or that are absent) / |rows_rev4| > MAX_EXT_FAIL -> STOP
     NaN          (a) any rows_rev4 rxn with a block that failed as nonfinite:<cols> (ext_features turns a NaN /
                  inf feature into that block failure, so without this gate a NaN would only be a dropped row);
                  (b) any NaN / inf in the 112 EXT_COLS of an ext_status-ok row (consistency)             -> STOP
     B5 gate      any status_B5 == scan_gate_fail (δ = 0 recomputation != the G1 features)                -> STOP
4. Always writes results_rev5/B7_report.json (failures per block and their reasons, core-h, n_de_floored stats,
   self-check summaries, per-feature median / 1 % / 99 % quantiles, gates, how the parquet was written) and
   results_rev5/B7_feature_quantiles.csv (over rows_rev4 ∩ ext ok), also when step 2 or a gate STOPs. Only when every
   gate passes and FEAT_EXT holds the merged slices: results_rev5/rows_rev5.csv = rows_rev4 ∩ ext_status ok,
   ascending rxn_id, the one-column format of rows_rev4.csv; an existing different rows_rev5.csv is replaced only with
   --force (else exit 3, nothing replaced).
Exit codes: 0 every gate passed, 2 bad / incomplete inputs, 3 a gate failed (STOP) or an existing output differs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rev5_common as R5  # noqa: E402
import ext_features as E  # noqa: E402  (column layout and cache paths; importing it computes nothing)
import xtb_slice as xs  # noqa: E402

REPORT = R5.RES5 / "B7_report.json"
QUANT_CSV = R5.RES5 / "B7_feature_quantiles.csv"


def die(code, msg):
    print(msg, file=sys.stderr, flush=True)
    sys.exit(code)


def expected_ids():
    """The G1-ok rxn_ids (accepted labels with .done); exit 2 if G1 is incomplete."""
    labels = json.loads(Path(R5.LABELS).read_text())
    acc = sorted(int(d["rxn_id"]) for d in labels if int(d["rxn_id"]) not in xs.EXCLUDE)
    marks = {r: xs.geom_marker("g1", r) for r in acc}
    unmarked = [r for r, m in marks.items() if m is None]
    if unmarked:
        die(2, f"G1 incomplete: {len(unmarked)} accepted rxns without .done / .fail_*: {unmarked[:10]}")
    return sorted(r for r, m in marks.items() if m == ".done")


def write_parquet_checked(df, dst, force):
    """'new' / 'unchanged' / 'replaced', or 'differs' (an existing different file, no --force: left untouched)."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".merge.tmp")
    df.to_parquet(tmp, index=False)
    if dst.exists():
        if pd.read_parquet(tmp).equals(pd.read_parquet(dst)):
            tmp.unlink()
            return "unchanged"
        if not force:
            tmp.unlink()
            return "differs"
        os.replace(tmp, dst)
        return "replaced"
    os.replace(tmp, dst)
    return "new"


def _num(x):
    return None if x is None or not np.isfinite(x) else float(x)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="replace a different existing parquet / rows_rev5.csv")
    ap.add_argument("--slices-dir", type=Path, default=E.SLICE_DIR)
    ap.add_argument("--n-slices", type=int, default=E.N_SLICES)
    a = ap.parse_args()

    # ---- 1. slices
    paths = [a.slices_dir / f"slice_{k:02d}.parquet" for k in range(a.n_slices)]
    miss = [str(p) for p in paths if not p.is_file()]
    if miss:
        die(2, f"STOP: {len(miss)}/{a.n_slices} slice parquets missing (r5_ext_array.sh incomplete): {miss}")
    parts = [pd.read_parquet(p) for p in paths]
    slices = [dict(path=str(p), n_rows=len(d), sha256=R5.sha256(p)) for p, d in zip(paths, parts)]
    df = pd.concat(parts, ignore_index=True)
    lack = [c for c in E.ROW_COLS if c not in df.columns]
    if lack:
        die(2, f"slices lack {len(lack)} columns: {lack[:10]}")
    df["rxn_id"] = df["rxn_id"].astype("int64")
    if df["rxn_id"].duplicated().any():
        die(2, f"{int(df['rxn_id'].duplicated().sum())} duplicate rxn_id across the slices")
    tags = sorted(set(df["geom"].astype(str)))
    if tags != ["g1"]:
        die(2, f"geom tags {tags}, expected ['g1']")
    exp = expected_ids()
    got = set(df["rxn_id"].tolist())
    missing, extra = sorted(set(exp) - got), sorted(got - set(exp))
    if missing or extra:
        die(2, f"slices hold {len(df)} rows vs {len(exp)} G1-ok rxns: missing {len(missing)} {missing[:10]}, "
               f"unexpected {len(extra)} {extra[:10]}")
    df = df.sort_values("rxn_id", kind="stable").reset_index(drop=True)
    df = df[E.ROW_COLS + [c for c in df.columns if c not in E.ROW_COLS]]

    # ---- 2. the block parquet
    how = write_parquet_checked(df, R5.FEAT_EXT, a.force)
    if how == "differs":
        print(f"STOP: {R5.FEAT_EXT} exists and differs from the merged slices; kept as it is (--force / MERGE_FORCE=1 "
              f"replaces it). The report is still written; rows_rev5.csv is not.", file=sys.stderr, flush=True)
    else:
        print(f"{R5.FEAT_EXT}: {how} ({len(df)} rows)", flush=True)

    # ---- 3. gates
    rows4 = pd.read_csv(R5.ROWS4)["rxn_id"].astype(int).tolist()
    ok = df["ext_status"] == "ok"
    in4 = df["rxn_id"].isin(rows4)
    ok_ids = set(df.loc[ok, "rxn_id"].tolist())
    fail4 = [r for r in rows4 if r not in ok_ids]
    share = len(fail4) / len(rows4)
    X = df.loc[ok, R5.EXT_COLS].to_numpy(dtype=float)
    nonfin = df.loc[ok, "rxn_id"][~np.isfinite(X).all(axis=1)].tolist()
    stat = df[[f"status_{b}" for b in R5.BLOCK_ORDER]].astype(str)
    nf_any = stat.apply(lambda s: s.str.startswith("nonfinite:")).any(axis=1)      # per row: a NaN / inf feature
    nf4 = df.loc[in4 & nf_any, "rxn_id"].tolist()
    nf_blocks = {b: int((stat[f"status_{b}"].str.startswith("nonfinite:") & in4).sum()) for b in R5.BLOCK_ORDER}
    scan_fail = df.loc[df["status_B5"].astype(str) == "scan_gate_fail", "rxn_id"].tolist()
    reasons4 = df.loc[in4 & ~ok, "ext_status"].astype(str).value_counts().head(20).to_dict()
    gates = {"fail_share_rows_rev4": dict(value=share, n_fail=len(fail4), n_rows_rev4=len(rows4),
                                          limit=R5.MAX_EXT_FAIL, passed=share <= R5.MAX_EXT_FAIL,
                                          reasons=reasons4, rxns=fail4[:300]),
             "nonfinite_features_rows_rev4": dict(n=len(nf4), n_all_rows=int(nf_any.sum()), per_block=nf_blocks,
                                                  rxns=nf4[:300], passed=not nf4),
             "nan_in_ok_rows": dict(n=len(nonfin), rxns=nonfin[:300], passed=not nonfin),
             "b5_scan_gate": dict(n=len(scan_fail), rxns=scan_fail[:300], passed=not scan_fail)}
    passed = all(g["passed"] for g in gates.values())

    # ---- 4. report
    per_block = {}
    for b in R5.BLOCK_ORDER:
        st = df[f"status_{b}"].astype(str)
        bad = st != "ok"
        per_block[b] = dict(n_fail_all=int(bad.sum()), n_fail_rows_rev4=int((bad & in4).sum()),
                            reasons=st[bad].value_counts().head(15).to_dict(),
                            core_h=float(df[f"t_{b}"].sum(skipna=True)) / 3600.0,
                            median_s_per_rxn=_num(df.loc[ok, f"t_{b}"].median()))
    geom_fail = df.loc[df["ext_status"].astype(str).str.startswith("geom:"), "ext_status"].value_counts().to_dict()
    sel = ok & in4

    def stats(col):
        s = df.loc[sel, col].astype(float)
        return dict(n=int(s.notna().sum()), total=_num(s.sum()), mean=_num(s.mean()), median=_num(s.median()),
                    max=_num(s.max()), n_rxn_gt0=int((s > 0).sum()))

    qc = {}
    for col in E.EXTRA_COLS:
        if col in E.STR_EXTRA or col.startswith("n_de_"):
            continue
        s = df.loc[ok, col].astype(float).abs()
        qc[col] = dict(max_abs=_num(s.max()), median_abs=_num(s.median()))
    dE = df.loc[ok, ["qc_b1_dE_fA_eh", "qc_b1_dE_fB_eh", "qc_b1_dE_ts_eh"]].astype(float).abs()
    qc["n_ok_rows_tblite_minus_xtb_ge_GATE_E_EH"] = int((dE >= R5.GATE_E_EH).any(axis=1).sum())
    qc["quad_orders"] = df.loc[ok, "qc_b3_quad_order"].astype(str).value_counts().to_dict()
    qc["b6_map_identity_share"] = _num(df.loc[ok, "qc_b6_map_identity"].astype(float).mean())
    qc["b6_product_file_p0_share"] = _num(df.loc[ok, "qc_b6_product_file"].astype(str).str.startswith("p0_").mean())

    qrows = []
    for b in R5.BLOCK_ORDER:
        for col in R5.BLOCKS[b]:
            s = df.loc[sel, col].astype(float)
            fin = s[np.isfinite(s)]
            qrows.append(dict(block=b, feature=col, n=int(fin.size), n_nonfinite=int(s.size - fin.size),
                              median=_num(fin.median()), q01=_num(fin.quantile(0.01)), q99=_num(fin.quantile(0.99)),
                              mean=_num(fin.mean()), std=_num(fin.std()), min=_num(fin.min()), max=_num(fin.max())))
    qdf = pd.DataFrame(qrows)
    R5.write_atomic(QUANT_CSV, qdf.to_csv(index=False))

    t_tot = df["t_total_s"].astype(float)
    rep = dict(spec="REV5 B-7", slices=slices, n_rows=len(df), n_g1_ok=len(exp), n_ext_ok=int(ok.sum()),
               n_rows_rev4=len(rows4), rows_rev4_sha256=R5.sha256(R5.ROWS4),
               ext_parquet=str(R5.FEAT_EXT),
               ext_parquet_write=(how if how != "differs" else "NOT replaced: the existing file differs from the "
                                  "merged slices (--force / MERGE_FORCE=1 replaces it); this report describes the slices"),
               ext_parquet_sha256=R5.sha256(R5.FEAT_EXT) if how != "differs" else None,
               ext_parquet_on_disk_sha256=R5.sha256(R5.FEAT_EXT),
               ext_status_top=df.loc[~ok, "ext_status"].astype(str).value_counts().head(25).to_dict(),
               geometry_failures=geom_fail, failures_per_block=per_block,
               core_h=dict(total=_num(t_tot.sum() / 3600.0), per_rxn_mean_s=_num(t_tot.mean()),
                           per_rxn_median_s=_num(t_tot.median()),
                           per_block={b: per_block[b]["core_h"] for b in R5.BLOCK_ORDER}),
               n_de_floored=stats("n_de_floored"), n_de_floored_win=stats("n_de_floored_win"), de_floor_ev=R5.DE_FLOOR_EV,
               qc=qc, gates=gates, passed=passed, quantiles_csv=str(QUANT_CSV),
               feature_quantiles={b: {r["feature"]: dict(median=r["median"], q01=r["q01"], q99=r["q99"])
                                      for r in qrows if r["block"] == b} for b in R5.BLOCK_ORDER})

    # ---- 5. rows_rev5 (only when every gate passed and FEAT_EXT holds the merged slices)
    code = 0 if passed and how != "differs" else 3
    rows_info = dict(path=str(R5.ROWS5))
    if how == "differs":
        rows_info.update(n=None, write=f"not written: {R5.FEAT_EXT.name} on disk differs from the merged slices and "
                                       f"was not replaced (--force / MERGE_FORCE=1)")
    elif passed:
        ids5 = sorted(set(rows4) & ok_ids)
        text = pd.DataFrame({"rxn_id": ids5}).to_csv(index=False)
        old = R5.ROWS5.read_text() if R5.ROWS5.exists() else None
        if old is not None and old != text and not a.force:
            rows_info.update(n=len(ids5), write="NOT written: an existing different rows_rev5.csv (--force replaces it)")
            code = 3
        else:
            if old != text:
                R5.write_atomic(R5.ROWS5, text)
            rows_info.update(n=len(ids5), write="unchanged" if old == text else ("replaced" if old is not None else "new"),
                             sha256=R5.sha256(R5.ROWS5))
    else:
        rows_info.update(n=None, write="not written: a B-7 gate failed")
    rep["rows_rev5"] = rows_info
    R5.write_json(REPORT, rep)
    print(json.dumps(dict(n_rows=rep["n_rows"], n_ext_ok=rep["n_ext_ok"], core_h=rep["core_h"],
                          gates={k: {kk: vv for kk, vv in g.items() if kk != "rxns"} for k, g in gates.items()},
                          failures_per_block={b: per_block[b]["n_fail_rows_rev4"] for b in R5.BLOCK_ORDER},
                          n_de_floored=rep["n_de_floored"], rows_rev5=rows_info), indent=1, default=str), flush=True)
    print(f"wrote {REPORT} and {QUANT_CSV}", flush=True)
    if code:
        why = []
        if not passed:
            why.append("a gate failed: " + ", ".join(k for k, g in gates.items() if not g["passed"]))
        if how == "differs" or passed:
            why.append(f"rows_rev5.csv {rows_info['write']}")
        die(code, "STOP (B-7): " + "; ".join(why))
    return 0


if __name__ == "__main__":
    sys.exit(main())
