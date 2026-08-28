#!/usr/bin/env python3
"""SPEC14 Step 5 — compare B3LYP-geometry channels vs existing (low-level TS)
channels from phase5_dataset_v2.pkl.

Main deliverable: artifacts/channel_comparison.csv
GATE-4 verdict per channel:
   δ-propagated < 1.0 kcal/mol            → PASS      (target achievable)
   1.0 ≤ δ-propagated < 3.0               → CONDITIONAL
   δ-propagated ≥ 3.0                     → FAIL      (recompute all 3504)
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/geometry_validation")
COH = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments/cohort_v1")

CHANNELS = ["elst", "pauli", "oi", "disp", "cpcm", "cds"]


def verdict(delta_prop: float) -> str:
    if delta_prop < 1.0:
        return "PASS"
    if delta_prop < 3.0:
        return "CONDITIONAL"
    return "FAIL"


def main():
    new = pd.read_csv(BASE / "artifacts" / "channels_b3lyp.csv").set_index("rxn_id")
    old = pd.read_pickle(COH / "phase5_dataset_v2.pkl")
    old["reaction_number"] = old["reaction_number"].astype(int)
    old = old.set_index("reaction_number")
    meta = pd.read_csv(BASE / "artifacts" / "input_meta.csv").set_index("rxn_id")
    sel = pd.read_csv(BASE / "artifacts" / "sample.csv").set_index("rxn_id")

    idx = new.index.intersection(old.index)
    print(f"comparable rxns : {len(idx)}")

    rows = []
    for c in CHANNELS:
        a = old.loc[idx, f"{c}_dft"].values.astype(float)      # low-level TS
        b = new.loc[idx, c].values.astype(float)               # B3LYP TS
        mask = ~(np.isnan(a) | np.isnan(b))
        a, b = a[mask], b[mask]
        d = b - a
        mae = float(np.abs(d).mean())
        rows.append(dict(
            channel=c,
            n=int(mask.sum()),
            mean_signed=float(d.mean()),
            mae=mae,
            median_abs=float(np.median(np.abs(d))),
            p95_abs=float(np.percentile(np.abs(d), 95)),
            max_abs=float(np.abs(d).max()),
            std=float(d.std()),
            rel_to_scale=float(mae / max(np.abs(a).mean(), 1e-9)),
            pearson_r=float(np.corrcoef(a, b)[0, 1]) if len(a) >= 3 else np.nan,
            delta_propagated=float(mae * np.sqrt(2)),
            verdict=verdict(mae * np.sqrt(2)),
        ))
    R = pd.DataFrame(rows)
    R.to_csv(BASE / "artifacts" / "channel_comparison.csv", index=False)
    pd.set_option("display.width", 160)
    print("\n=== Channel comparison (B3LYP TS − low-level TS, kcal/mol) ===")
    print(R.round(3).to_string(index=False))

    # Bond-length shift correlation: does formed-bond distance change
    # explain channel shifts?
    gm = meta.loc[idx.intersection(meta.index)]
    if len(gm) > 5:
        # Load low-level TS distances by re-reading existing eda.inp
        ext = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments/cohort_v1/reactions")

        def read_eda_xyz(path):
            lines = path.read_text().splitlines()
            in_xyz = False
            syms, xyz = [], []
            for ln in lines:
                s = ln.strip()
                if s.startswith("* xyz"):
                    in_xyz = True
                    continue
                if in_xyz:
                    if s.startswith("*") or not s:
                        break
                    parts = s.split()
                    syms.append(parts[0].split("(")[0])
                    xyz.append([float(x) for x in parts[1:4]])
            return np.array(xyz)

        corr_rows = []
        for rid, r in gm.iterrows():
            p = ext / f"rxn_{int(rid):04d}" / "eda.inp"
            if not p.exists():
                continue
            xyz_low = read_eda_xyz(p)
            # formed_pair1/2 stored as "(i, j)" strings
            def parse_pair(s):
                s = s.strip("() ")
                a, b = s.split(",")
                return int(a), int(b)
            p1 = parse_pair(r["formed_pair1"])
            p2 = parse_pair(r["formed_pair2"])
            d1_low = float(np.linalg.norm(xyz_low[p1[0]] - xyz_low[p1[1]]))
            d2_low = float(np.linalg.norm(xyz_low[p2[0]] - xyz_low[p2[1]]))
            corr_rows.append(dict(
                rxn_id=rid,
                d1_low=d1_low, d1_b3lyp=r["formed_d1_b3lyp"], d1_diff=r["formed_d1_b3lyp"] - d1_low,
                d2_low=d2_low, d2_b3lyp=r["formed_d2_b3lyp"], d2_diff=r["formed_d2_b3lyp"] - d2_low,
            ))
        cdf = pd.DataFrame(corr_rows).set_index("rxn_id")
        cdf.to_csv(BASE / "artifacts" / "bond_length_shifts.csv")
        print("\n=== Formed-bond length shift (B3LYP − low-level, Å) ===")
        print(f"  d1: mean={cdf['d1_diff'].mean():.3f}  |diff|_med={cdf['d1_diff'].abs().median():.3f}  p95={cdf['d1_diff'].abs().quantile(0.95):.3f}")
        print(f"  d2: mean={cdf['d2_diff'].mean():.3f}  |diff|_med={cdf['d2_diff'].abs().median():.3f}  p95={cdf['d2_diff'].abs().quantile(0.95):.3f}")

        # Correlation between bond-length shift and channel shift
        corr_ch = []
        for c in ["pauli", "elst", "oi"]:
            common = idx.intersection(cdf.index)
            if len(common) < 5:
                continue
            ch_diff = (new.loc[common, c].values - old.loc[common, f"{c}_dft"].values)
            bd_diff = np.mean(np.abs(cdf.loc[common, ["d1_diff", "d2_diff"]].values), axis=1)
            r_val = float(np.corrcoef(np.abs(ch_diff), bd_diff)[0, 1])
            corr_ch.append((c, r_val))
        print("\n|Δchannel| vs |Δbond-len| Pearson r:")
        for c, r_val in corr_ch:
            print(f"  {c}: r = {r_val:+.3f}")

    # GATE-4 status
    worst = R["delta_propagated"].max()
    n_fail = int((R["verdict"] == "FAIL").sum())
    n_cond = int((R["verdict"] == "CONDITIONAL").sum())
    n_pass = int((R["verdict"] == "PASS").sum())
    with open(BASE / "artifacts" / "GATE4_STATUS.txt", "w") as f:
        f.write(f"worst_delta_prop={worst:.3f} PASS={n_pass} CONDITIONAL={n_cond} FAIL={n_fail}\n")
        for _, r in R.iterrows():
            f.write(f"  {r['channel']:>6}: δ_prop={r['delta_propagated']:.3f} {r['verdict']}\n")
    if n_fail > 0:
        overall = "FAIL"
    elif n_cond > 0:
        overall = "CONDITIONAL"
    else:
        overall = "PASS"
    print(f"\n=== GATE-4 {overall} (worst δ_propagated = {worst:.3f}) ===")


if __name__ == "__main__":
    main()
