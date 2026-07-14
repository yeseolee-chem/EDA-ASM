"""Build the GFN2-xTB cache for all 789 v6_multifamily reactions.

Reads `runs/orca_recompute/results/orca_eda_labels.parquet` (for the canonical
source_dir mapping), runs three GFN2-xTB single-points per reaction (complex,
fragA, fragB at TS-frozen geometries), and writes:

    analysis/exp_6arm_redesign_v2/xtb_cache/xtb_features.parquet

Idempotent: skips reactions already present unless --rebuild is passed.
Parallelised over processes (default = os.cpu_count() // 2, capped at 16).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure imports work whether invoked from repo root or from analysis dir.
HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE))

from src.inventory import Reaction, load_reaction  # noqa: E402
from src.xtb_features import compute_xtb_features  # noqa: E402


def _worker(orca_row_dict: dict) -> dict:
    """Picklable wrapper: orca row → xtb result dict."""
    row = pd.Series(orca_row_dict)
    rxn = load_reaction(row)
    t0 = time.time()
    res = compute_xtb_features(rxn)
    dd = res.to_dict()
    dd["family"] = rxn.family
    dd["wall_s"] = round(time.time() - t0, 3)
    return dd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--orca-labels", type=Path,
                    default=REPO / "runs/orca_recompute/results/orca_eda_labels.parquet")
    ap.add_argument("--out", type=Path,
                    default=HERE / "xtb_cache/xtb_features.parquet")
    ap.add_argument("--workers", type=int,
                    default=min(16, max(1, (os.cpu_count() or 4) // 2)))
    ap.add_argument("--limit", type=int, default=0,
                    help="Process only the first N reactions (smoke test).")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    orca = pd.read_parquet(args.orca_labels)
    if args.limit:
        orca = orca.head(args.limit).copy()
    print(f"[xtb] {len(orca)} reactions to process; workers = {args.workers}")

    # Idempotency
    have = set()
    if args.out.exists() and not args.rebuild:
        prev = pd.read_parquet(args.out)
        have = set(prev["rid"].to_list())
        print(f"[xtb] cache exists with {len(have)} entries — skipping those.")
    else:
        prev = pd.DataFrame()

    todo = orca[~orca["reaction_id"].isin(have)].copy()
    print(f"[xtb] {len(todo)} to compute.")
    if len(todo) == 0:
        print("[xtb] nothing to do."); return

    payload = [r.to_dict() for _, r in todo.iterrows()]

    results: list[dict] = []
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_worker, p): p["reaction_id"] for p in payload}
        done = 0
        for fut in as_completed(futs):
            rid = futs[fut]
            try:
                results.append(fut.result())
            except Exception as e:
                results.append({"rid": rid, "ok": False, "fail_reason": f"worker_exc:{e}"})
            done += 1
            if done % 20 == 0 or done == len(futs):
                elapsed = time.time() - t_start
                eta = elapsed / done * (len(futs) - done)
                print(f"[xtb] {done}/{len(futs)}  elapsed={elapsed:6.1f}s  eta={eta:6.1f}s", flush=True)

    new_df = pd.DataFrame(results)
    out_df = pd.concat([prev, new_df], ignore_index=True) if len(prev) else new_df

    out_df.to_parquet(args.out, index=False)
    n_ok = int(out_df["ok"].sum())
    n_fail = len(out_df) - n_ok
    fail_reasons = out_df.loc[~out_df["ok"], "fail_reason"].value_counts().head(10).to_dict()
    print(f"[xtb] cache written: {args.out}")
    print(f"[xtb] ok = {n_ok}/{len(out_df)}; failed = {n_fail}")
    if fail_reasons:
        print(f"[xtb] top fail reasons: {fail_reasons}")


if __name__ == "__main__":
    main()
