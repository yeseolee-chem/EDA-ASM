#!/usr/bin/env python3
"""make_rows_rev4.py — REV4 §3-1 pre-registered ML rows, from the G1 feature parquet.

  python make_rows_rev4.py [--g1 PARQUET] [--out-dir DIR]        # registration -> results_rev4/rows_rev4.{csv,json}
  python make_rows_rev4.py --check-g1-only [--g1 PARQUET]         # REV5 A-4 -> results_rev5/A_rows_check.json
Default --g1 $ESPLEY_OUT/xtb_features_g1.parquet; --out-dir (where rows_rev4.csv lives) results_rev4/.
rows = G1 xtb_status ok ∩ no NaN in any ESPLEY73 feature or rev 4 target ∩ hygiene (train_ml_single.hygiene:
       d1 >= 0, d2 >= 0, d2 <= 50); single column rxn_id, ascending.
The rev 4 registration also required the second, since discarded geometry to be ok; it was ok for every one of the
5,260 rxns, so this G1-only rule must give exactly the registered rows_rev4.csv (4,839 rows): --check-g1-only asserts it.
Gates (exit 1, nothing written): unique rxn_ids, geom tag g1, required columns present, n > 0.
registration: rows_rev4.csv is pre-registered — an existing one is never rewritten (exit 3 if the recomputed rows
  differ); rows_rev4.json (the registration record) is written only together with a new rows_rev4.csv.
--check-g1-only: rewrites results_rev5/A_rows_check.json atomically on every run (cheap, deterministic); exit 3 if the
  recomputed rows differ from rows_rev4.csv (content or order), 0 if identical.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train_ml_single import FEATURE_SETS, TARGET_SETS, hygiene, sha256  # noqa: E402
from rev5_common import RES5, write_json  # noqa: E402

ROOT = Path(os.environ.get("ESPLEY_OUT", "/gpfs/tmp_cpu2/yeseo1ee/espley_xtb"))
F73 = FEATURE_SETS["ESPLEY73"]                       # ⊇ ESPLEY46, ESPLEY54
TGT = TARGET_SETS["rev4"]
GEOM = "g1"                                          # parquet `geom` tag (xtb_slice.py --geom g1)
RULE = ("G1 xtb_status ok ∩ no NaN in ESPLEY73 features + rev 4 targets ∩ hygiene (dft_d1_kcal >= 0, "
        "dft_d2_kcal >= 0, dft_d2_kcal <= 50); ascending rxn_id")


def write_atomic(path, text):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def load(path):
    if not path.is_file():
        sys.exit(f"GATE: {path} missing")
    df = pd.read_parquet(path)
    bad = []
    tags = sorted(set(df["geom"].astype(str))) if "geom" in df else ["<no geom column>"]
    if tags != [GEOM]:
        bad.append(f"geom tags {tags} != ['{GEOM}']")
    lacking = [c for c in ["rxn_id", "xtb_status"] + F73 + TGT if c not in df]
    if lacking:
        bad.append(f"columns missing: {lacking}")
    elif df.rxn_id.duplicated().any():
        bad.append(f"{int(df.rxn_id.duplicated().sum())} duplicate rxn_id")
    if bad:
        sys.exit(f"GATE ({path}): " + "; ".join(bad))
    return df.assign(rxn_id=df.rxn_id.astype(int)).set_index("rxn_id")


def compute(path):
    """(rows, record of every step) from the G1 parquet alone."""
    G = load(path)
    ok = sorted(int(r) for r in G.index[(G.xtb_status == "ok").to_numpy()])
    m = G.loc[ok, F73 + TGT].isna().any(axis=1)
    nan = sorted(int(r) for r in m[m].index)
    nan_set = set(nan)
    keep = [r for r in ok if r not in nan_set]
    if not keep:
        sys.exit("GATE: no rxn is G1 ok and NaN-free")
    sub = G.loc[keep]
    hyg = hygiene(sub)
    rows = sorted(int(r) for r in hyg[hyg].index)
    if not rows:
        sys.exit("GATE: no row survives hygiene")
    d1, d2 = sub["dft_d1_kcal"], sub["dft_d2_kcal"]
    rep = dict(rule=RULE,
               input=dict(path=str(path), sha256=sha256(path), n_rows=len(G),
                          xtb_status={str(s): int(n) for s, n in G.xtb_status.value_counts().items()}),
               targets=TGT, n_features_checked=len(F73), n_g1_ok=len(ok),
               n_nan_dropped=len(nan), nan_dropped=nan, n_after_nan=len(keep),
               n_hygiene_dropped=int((~hyg).sum()), hygiene_dropped=sorted(int(r) for r in hyg[~hyg].index),
               hygiene_breakdown=dict(d1_lt_0=int((d1 < 0).sum()), d2_lt_0=int((d2 < 0).sum()),
                                      d2_gt_50=int((d2 > 50).sum())),
               n_final=len(rows))
    return rows, rep


def check_g1_only(rows, rep, csv):
    """REV5 A-4: the rows rebuilt from G1 alone must equal the registered rows_rev4.csv."""
    if not csv.is_file():
        sys.exit(f"GATE: {csv} missing — nothing to check against")
    old = pd.read_csv(csv)["rxn_id"].astype(int).tolist()
    js = csv.with_suffix(".json")
    reg = json.loads(js.read_text()) if js.is_file() else {}
    equal = old == rows
    out = RES5 / "A_rows_check.json"
    write_json(out, dict(check="REV5 A-4: rows rebuilt from the G1 parquet alone == results_rev4/rows_rev4.csv "
                               "(same rxn_ids, same order)",
                         verdict="PASS" if equal else "FAIL", equal=equal,
                         rows_rev4=str(csv), rows_rev4_sha256=sha256(csv), n_rows_rev4=len(old),
                         rows_rev4_json_n_final=reg.get("n_final"), n_recomputed=len(rows),
                         same_set=set(old) == set(rows),
                         only_in_rows_rev4=sorted(set(old) - set(rows)),
                         only_in_recomputed=sorted(set(rows) - set(old)),
                         recomputed=rep))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("nan_dropped", "hygiene_dropped", "input")}, indent=1))
    print(f"rows_rev4.csv {len(old)} rows, recomputed from G1 alone {len(rows)}: "
          f"{'IDENTICAL' if equal else 'DIFFERENT'} -> {out}")
    if not equal:
        print(f"STOP: only in rows_rev4.csv {len(set(old) - set(rows))}, only in recomputed "
              f"{len(set(rows) - set(old))} (order differs: {set(old) == set(rows)})")
        sys.exit(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g1", type=Path, default=ROOT / "xtb_features_g1.parquet")
    ap.add_argument("--out-dir", type=Path, default=HERE / "results_rev4")
    ap.add_argument("--check-g1-only", action="store_true",
                    help="assert the G1-only rows == <out-dir>/rows_rev4.csv; report results_rev5/A_rows_check.json")
    a = ap.parse_args()
    rows, rep = compute(a.g1)
    csv, js = a.out_dir / "rows_rev4.csv", a.out_dir / "rows_rev4.json"
    if a.check_g1_only:
        check_g1_only(rows, rep, csv)
        return

    a.out_dir.mkdir(parents=True, exist_ok=True)
    if csv.exists():
        old = pd.read_csv(csv)["rxn_id"].astype(int).tolist()
        if old != rows:
            print(f"STOP: {csv} exists (pre-registered, {len(old)} rows) but the recomputed rows differ "
                  f"({len(rows)} rows; only in old {len(set(old) - set(rows))}, only in new {len(set(rows) - set(old))})."
                  f" Not overwritten.")
            sys.exit(3)
        print(f"{csv} exists and equals the recomputed rows — kept, {js.name} (registration record) not rewritten")
        return
    write_atomic(csv, pd.DataFrame({"rxn_id": rows}).to_csv(index=False))
    rep.update(rows_csv=str(csv), rows_sha256=sha256(csv))
    write_atomic(js, json.dumps(rep, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("nan_dropped", "hygiene_dropped", "input")}, indent=1))
    print(f"-> {csv} ({len(rows)} rows), {js}")


if __name__ == "__main__":
    main()
