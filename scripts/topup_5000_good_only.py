"""Top up the 5000 working set with 'good' (n_fragments >= 2) reactions only.

After Stage 5a runs on all 19,175 eligible trajectories, this script:
  1) Identifies all reactions where n_fragments == 1 (fragment = whole molecule).
  2) Reports counts across the full pool and the current 5000 selection.
  3) Removes bad reactions from the 5000 selection.
  4) Tops up the deficit by sampling from the eligible good pool, stratified
     by source × ea_tertile, matching the original sampling strategy.
  5) Writes updated CSV selected_reactions_5000_good.csv and merges any new
     Stage 5a results into outputs/stage5a/.

Run AFTER outputs/stage5a_remaining/ is fully populated.
"""
from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
S5A_DIR = REPO / "outputs/stage5a"
S5A_REM = REPO / "outputs/stage5a_remaining"
INDEX_FP = REPO / "data/halo8_index/index.parquet"
SEED = 20260516


def main():
    # 1) Load all Stage 5a summaries we have so far
    summary_main = json.loads((S5A_DIR / "fragmentation_summary.json").read_text())
    summary_rem = []
    if (S5A_REM / "fragmentation_summary.json").exists():
        summary_rem = json.loads((S5A_REM / "fragmentation_summary.json").read_text())
    print(f"main summary: {len(summary_main)}")
    print(f"remaining summary: {len(summary_rem)}")

    all_summary = summary_main + summary_rem
    by_id = {s["reaction_id"]: s for s in all_summary}
    print(f"combined: {len(by_id)}")

    # 2) Identify "fragment = whole molecule" (n_fragments == 1)
    bad_ids = {rid for rid, s in by_id.items() if s["n_fragments"] == 1}
    print(f"\n=== Classified pool ===")
    print(f"  Total classified:      {len(by_id):>6,}")
    print(f"  n_fragments==1 (bad):  {len(bad_ids):>6,}")
    print(f"  n_fragments>=2 (good): {len(by_id) - len(bad_ids):>6,}")

    by_pat_bad = Counter()
    by_pat_total = Counter()
    for rid, s in by_id.items():
        by_pat_total[s["pattern"]] += 1
        if rid in bad_ids:
            by_pat_bad[s["pattern"]] += 1
    print(f"\n  Bad by pattern:")
    for pat, n in by_pat_bad.most_common():
        pct = 100 * n / by_pat_total[pat]
        print(f"    {pat:12s} {n:>5d}/{by_pat_total[pat]:>5d} ({pct:.1f}%)")

    # 3) Current 5000 set
    sel = pd.read_csv(REPO / "outputs/phase1/selected_reactions_5000.csv")
    print(f"\n=== Current 5000 selection ===")
    cur_ids = set(sel["reaction_id"])
    cur_bad = cur_ids & bad_ids
    cur_good = cur_ids - bad_ids
    print(f"  Total:        {len(cur_ids):>5d}")
    print(f"  Bad in set:   {len(cur_bad):>5d}")
    print(f"  Good in set:  {len(cur_good):>5d}")

    # 4) Top-up: sample (5000 - len(cur_good)) from eligible pool, stratified
    deficit = 5000 - len(cur_good)
    if deficit <= 0:
        print(f"\nNo top-up needed (already at/above 5000 good).")
        return
    print(f"\n=== Top-up: need {deficit} more good reactions ===")

    # Eligible: classified, good, not already in current good set
    idx = pd.read_parquet(INDEX_FP)
    eligible = idx[
        idx["reaction_id"].isin(set(by_id.keys()) - bad_ids - cur_good)
    ].copy()
    print(f"  Eligible top-up pool: {len(eligible):>5d}")

    # Stratify by source × ea_tertile (computed on the current good set for proportional matching)
    cur_good_df = sel[sel["reaction_id"].isin(cur_good)]
    # Use the same ea_tertile bins as in the good set
    if "ea_tertile" not in cur_good_df.columns:
        cur_good_df["ea_tertile"] = pd.qcut(
            cur_good_df["activation_energy"], 3, labels=["low", "mid", "high"]
        )

    rng = np.random.default_rng(SEED)
    add_rows = []
    eligible["ea_tertile"] = pd.qcut(
        eligible["activation_energy"], 3, labels=["low", "mid", "high"]
    )

    # Per-cell quotas based on cur_good distribution
    target_dist = cur_good_df.groupby(["source", "ea_tertile"], observed=True).size()
    target_total = target_dist.sum()
    for (src, tert), n_cur in target_dist.items():
        quota = round(deficit * n_cur / target_total)
        cell = eligible[(eligible["source"] == src) &
                          (eligible["ea_tertile"] == tert)]
        take = min(quota, len(cell))
        if take == 0:
            continue
        picked = cell.sample(n=take, random_state=rng.integers(2**31))
        add_rows.append(picked)
        print(f"    {src:10s} × {tert:5s}: pool={len(cell):>5d} quota={quota:>4d} picked={take}")
    add = pd.concat(add_rows, ignore_index=True) if add_rows else pd.DataFrame()
    # If still short, fill randomly from remaining good eligible
    if len(add) < deficit:
        more = eligible[~eligible["reaction_id"].isin(add["reaction_id"])]
        topup = more.sample(n=deficit - len(add), random_state=rng.integers(2**31))
        add = pd.concat([add, topup], ignore_index=True)
    elif len(add) > deficit:
        add = add.sample(n=deficit, random_state=rng.integers(2**31))

    print(f"  Final added: {len(add)}")

    # 5) Build new 5000_good CSV
    add["cohort"] = "phase2_topup"
    add["seed"] = SEED
    good_kept = sel[sel["reaction_id"].isin(cur_good)]
    combo = pd.concat([good_kept, add], ignore_index=True)
    out = REPO / "outputs/phase1/selected_reactions_5000_good.csv"
    combo.to_csv(out, index=False)
    print(f"\n[OK] → {out} ({len(combo)} rows)")

    # 6) Merge any new Stage 5a results into outputs/stage5a/
    n_merged = 0
    for rid in add["reaction_id"]:
        src = S5A_REM / "per_reaction" / rid
        dst = S5A_DIR / "per_reaction" / rid
        if src.exists() and not dst.exists():
            shutil.copytree(src, dst)
            n_merged += 1
    print(f"  Merged {n_merged} new per-reaction dirs into outputs/stage5a/per_reaction/")

    # 7) Update fragmentation_summary.json to reflect 5000_good
    main_summary = json.loads((S5A_DIR / "fragmentation_summary.json").read_text())
    main_ids = {s["reaction_id"] for s in main_summary}
    add_summary = [by_id[r] for r in add["reaction_id"] if r in by_id and r not in main_ids]
    new_summary = main_summary + add_summary
    (S5A_DIR / "fragmentation_summary.json").write_text(json.dumps(new_summary, indent=2))
    print(f"  fragmentation_summary.json: {len(main_summary)} → {len(new_summary)}")

    # 8) Also remove the bad ones from review_log.json so dashboard doesn't show them
    log_fp = S5A_DIR / "review_log.json"
    if log_fp.exists():
        log = json.loads(log_fp.read_text())
        # Mark bad reactions as 'excluded:no_fragmentation' (we won't remove the
        # accepted-by-user 500 originals though — leave their status alone if any
        # were already accepted)
        excluded = 0
        for rid in cur_bad:
            if rid in log:
                cur_status = log[rid].get("review_status")
                if cur_status in (None, "not_reviewed"):
                    log[rid]["review_status"] = "excluded_no_fragmentation"
                    excluded += 1
        log_fp.write_text(json.dumps(log, indent=2))
        print(f"  Marked {excluded} bad reactions as excluded_no_fragmentation in review_log")

    # 9) Final report
    print(f"\n{'='*60}\n  FINAL COUNTS\n{'='*60}")
    print(f"  Halo8 pool (interior_ts=True):  {len(idx[idx['interior_ts']]):>5,}")
    print(f"  Classified so far:               {len(by_id):>5,}")
    print(f"  Excluded (n_fragments==1):       {len(bad_ids):>5,}")
    print(f"  Eligible for ASM (n_frag>=2):    {len(by_id) - len(bad_ids):>5,}")
    print(f"  New 5000_good working set:       {len(combo):>5,}")
    print(f"  Bad removed from old 5000:       {len(cur_bad):>5,}")
    print(f"  Top-up added:                    {len(add):>5,}")


if __name__ == "__main__":
    main()
