#!/usr/bin/env python3
"""SPEC18 Step 2b — merge per-shard CSVs into `xtb_path_{MODE}.csv`
and write GATE-2 status.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
BASE = REPO / "analysis" / "otfm_guess"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="pilot", choices=["pilot", "full"])
    args = ap.parse_args()

    shards = sorted((BASE / "artifacts").glob(f"xtb_path_{args.mode}_shard*.csv"))
    if not shards:
        print(f"no shard CSVs for mode={args.mode}", file=sys.stderr)
        return 1
    D = pd.concat([pd.read_csv(p) for p in shards], ignore_index=True)
    # De-duplicate on rxn_id (later shards win — should not happen but safe)
    D = D.drop_duplicates("rxn_id", keep="last").sort_values("rxn_id")

    out = BASE / "artifacts" / f"xtb_path_{args.mode}.csv"
    D.to_csv(out, index=False)
    print(f"merged {len(shards)} shards -> {len(D)} rxns  ->  {out}")

    n_ok = int((D.status == "ok").sum())
    ok_rate = n_ok / max(1, len(D))
    counts = D.status.value_counts().to_dict()
    print(f"ok={n_ok}/{len(D)} ({ok_rate:.1%})")
    for k, v in counts.items():
        print(f"  {k}: {v}")

    # STOP conditions per SPEC §4 Step 2 (+ audit-added path-validity gates)
    order_mm = int(counts.get("order_mismatch", 0))
    n_notconn = int(counts.get("path_not_connected", 0))
    n_bad_barrier = int(counts.get("barrier_unphysical", 0))
    n_bad_dE = int(counts.get("dE_sign_wrong", 0))
    if order_mm > 0:
        print(f"⚠ order_mismatch={order_mm} — this is a WRITE bug. STOP.",
              file=sys.stderr)
    if n_notconn > 0 or n_bad_barrier > 0 or n_bad_dE > 0:
        print(f"[note] path validity: not_connected={n_notconn}  "
              f"barrier_unphysical={n_bad_barrier}  dE_sign_wrong={n_bad_dE}  "
              f"— these rxns fall back to (R+P)/2 at Step 4.")

    gate = "PASS" if (ok_rate >= 0.80 and order_mm == 0) else "FAIL"
    lines = [gate, f"n={len(D)}", f"ok={n_ok}", f"ok_rate={ok_rate:.4f}",
             f"order_mismatch={order_mm}",
             f"path_not_connected={n_notconn}",
             f"barrier_unphysical={n_bad_barrier}",
             f"dE_sign_wrong={n_bad_dE}"]
    lines += [f"{k}={v}" for k, v in counts.items()]
    (BASE / "artifacts" / f"GATE2_{args.mode}_STATUS.txt").write_text(
        "\n".join(lines) + "\n"
    )
    return 0 if gate == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
