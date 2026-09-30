#!/usr/bin/env python3
"""make_rows_rev4.py — REV4 §3-1 pre-registered ML rows; run after both feature parquets, BEFORE the PREREG commit.

  python make_rows_rev4.py [--g0 PARQUET] [--g1 PARQUET] [--out-dir DIR]
Defaults: $ESPLEY_OUT/xtb_features_{g0,g1}.parquet -> results_rev4/rows_rev4.csv (single column rxn_id, ascending)
+ rows_rev4.json (counts at each step, dropped rxn_ids, sha256 of the inputs and of the csv).
rows = G1 xtb_status ok ∩ G0 xtb_status ok ∩ no NaN in any ESPLEY73 feature or rev 4 target in either parquet
       ∩ hygiene (train_ml_single.hygiene: d1 >= 0, d2 >= 0, d2 <= 50).
G0 and G1 are both trained on exactly these rows in this order (ESPLEY_ROWS), so their 80/10/10 splits are identical.
Gates (exit 1, nothing written): unique rxn_ids, geom tags dft / g1, dft_* targets identical in G0 and G1, n > 0.
rows_rev4.csv is pre-registered: an existing one is never rewritten (exit 3 if the recomputed rows differ);
rows_rev4.json is rewritten atomically (deterministic).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train_ml_single import FEATURE_SETS, TARGET_SETS, hygiene, sha256  # noqa: E402

ROOT = Path(os.environ.get("ESPLEY_OUT", "/gpfs/tmp_cpu2/yeseo1ee/espley_xtb"))
F73 = FEATURE_SETS["ESPLEY73"]                       # ⊇ ESPLEY46, ESPLEY54
TGT = TARGET_SETS["rev4"]
GEOM_TAG = {"g0": "dft", "g1": "g1"}
TOL_TARGET = 1e-9


def write_atomic(path, text):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def load(g, path):
    df = pd.read_parquet(path)
    bad = []
    tags = sorted(set(df["geom"].astype(str))) if "geom" in df else ["<no geom column>"]
    if tags != [GEOM_TAG[g]]:
        bad.append(f"geom tags {tags} != ['{GEOM_TAG[g]}']")
    lacking = [c for c in ["rxn_id", "xtb_status"] + F73 + TGT if c not in df]
    if lacking:
        bad.append(f"columns missing: {lacking}")
    elif df.rxn_id.duplicated().any():
        bad.append(f"{int(df.rxn_id.duplicated().sum())} duplicate rxn_id")
    if bad:
        sys.exit(f"GATE {g} ({path}): " + "; ".join(bad))
    return df.assign(rxn_id=df.rxn_id.astype(int)).set_index("rxn_id")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g0", type=Path, default=ROOT / "xtb_features_g0.parquet")
    ap.add_argument("--g1", type=Path, default=ROOT / "xtb_features_g1.parquet")
    ap.add_argument("--out-dir", type=Path, default=HERE / "results_rev4")
    a = ap.parse_args()
    paths = {"g0": a.g0, "g1": a.g1}
    G = {g: load(g, p) for g, p in paths.items()}
    ok = {g: {int(r) for r in df.index[(df.xtb_status == "ok").to_numpy()]} for g, df in G.items()}
    both = sorted(ok["g0"] & ok["g1"])
    g1_only = sorted(ok["g1"] - ok["g0"])
    g0_only = sorted(ok["g0"] - ok["g1"])
    nan = {}
    for g, df in G.items():
        m = df.loc[both, F73 + TGT].isna().any(axis=1)
        nan[g] = sorted(int(r) for r in m[m].index)
    nan_any = set(nan["g0"]) | set(nan["g1"])
    keep = [r for r in both if r not in nan_any]
    if not keep:
        sys.exit("GATE: no rxn is ok and NaN-free on both geometries")
    tdiff = {c: float(np.max(np.abs(G["g0"].loc[keep, c].to_numpy(float) - G["g1"].loc[keep, c].to_numpy(float))))
             for c in TGT}
    if max(tdiff.values()) > TOL_TARGET:
        sys.exit(f"GATE: dft_* targets differ between G0 and G1 (max |diff| > {TOL_TARGET}): {tdiff}")
    sub = G["g1"].loc[keep]
    hyg = hygiene(sub)
    rows = sorted(int(r) for r in hyg[hyg].index)
    if not rows:
        sys.exit("GATE: no row survives hygiene")
    d1, d2 = sub["dft_d1_kcal"], sub["dft_d2_kcal"]

    native = lambda x: x.item() if hasattr(x, "item") else str(x)  # noqa: E731
    rep = dict(rule="G1 xtb_status ok ∩ G0 xtb_status ok ∩ no NaN in ESPLEY73 features + rev 4 targets (both parquets)"
                    " ∩ hygiene (dft_d1_kcal >= 0, dft_d2_kcal >= 0, dft_d2_kcal <= 50); ascending rxn_id",
               inputs={g: dict(path=str(p), sha256=sha256(p), n_rows=len(G[g]),
                               xtb_status={str(s): int(n) for s, n in G[g].xtb_status.value_counts().items()})
                       for g, p in paths.items()},
               targets=TGT, n_features_checked=len(F73),
               n_g1_ok=len(ok["g1"]), n_g0_ok=len(ok["g0"]), n_ok_both=len(both),
               n_g1_ok_not_g0_ok=len(g1_only),
               g1_ok_not_g0_ok=[dict(rxn_id=int(r), g0_status=str(G["g0"].xtb_status.get(r, "absent")))
                                for r in g1_only],
               n_g0_ok_not_g1_ok=len(g0_only),
               g1_status_of_g0_ok_not_g1_ok={str(s): int(n) for s, n in G["g1"].xtb_status.reindex(g0_only)
                                              .fillna("absent").value_counts().items()},
               n_nan_dropped=len(nan_any), nan_dropped=nan,
               n_after_nan=len(keep), max_abs_target_diff_g0_g1=tdiff,
               n_hygiene_dropped=int((~hyg).sum()), hygiene_dropped=sorted(int(r) for r in hyg[~hyg].index),
               hygiene_breakdown=dict(d1_lt_0=int((d1 < 0).sum()), d2_lt_0=int((d2 < 0).sum()),
                                      d2_gt_50=int((d2 > 50).sum())),
               n_final=len(rows))

    a.out_dir.mkdir(parents=True, exist_ok=True)
    csv, js = a.out_dir / "rows_rev4.csv", a.out_dir / "rows_rev4.json"
    if csv.exists():
        old = pd.read_csv(csv)["rxn_id"].astype(int).tolist()
        if old != rows:
            print(f"STOP: {csv} exists (pre-registered, {len(old)} rows) but the recomputed rows differ "
                  f"({len(rows)} rows; only in old {len(set(old) - set(rows))}, only in new {len(set(rows) - set(old))})."
                  f" Not overwritten.")
            sys.exit(3)
        print(f"{csv} exists and equals the recomputed rows — kept")
    else:
        write_atomic(csv, pd.DataFrame({"rxn_id": rows}).to_csv(index=False))
        print(f"wrote {csv}")
    rep.update(rows_csv=str(csv), rows_sha256=sha256(csv))
    write_atomic(js, json.dumps(rep, indent=1, default=native))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("nan_dropped", "hygiene_dropped", "inputs")},
                     indent=1, default=native))
    print(f"-> {csv} ({len(rows)} rows), {js}")


if __name__ == "__main__":
    main()
