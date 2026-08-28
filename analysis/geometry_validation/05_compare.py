#!/usr/bin/env python3
"""SPEC14 Step 5 (revised per precheck §4) — compare B3LYP-geometry channels
vs existing (low-level TS) channels; extract low-level bond distances directly
from existing reactions/rxn_XXXX/eda.inp so the geometry-shift correlation is
concrete rather than a placeholder.

GATE-4 verdict per channel (δ-propagation basis, ×√2):
    δ-propagated < 1.0 → PASS
    1.0 ≤ δ < 3.0     → CONDITIONAL
    δ ≥ 3.0           → FAIL   (recompute all 3504 at B3LYP TS)
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/geometry_validation")
COH = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments/cohort_v1")
EXT_EDA = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments/cohort_v1/reactions")

CHANNELS = ["elst", "pauli", "oi", "disp", "cpcm", "cds"]

_ATOM_RE = re.compile(r"^\s*([A-Za-z]+)\((\d)\)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)")


def read_eda_xyz(path: Path):
    """Return (syms, frags, xyz) from an eda.inp xyz block."""
    txt = path.read_text()
    m = re.search(r"\*\s*xyz\s+\S+\s+\S+\s*\n(.*?)\n\s*\*", txt, re.S)
    if not m:
        return None, None, None
    syms, frags, coords = [], [], []
    for line in m.group(1).splitlines():
        g = _ATOM_RE.match(line)
        if g:
            syms.append(g.group(1))
            frags.append(int(g.group(2)))
            coords.append([float(g.group(i)) for i in (3, 4, 5)])
    return syms, np.array(frags), np.array(coords)


def closest_two_heavy_contacts(syms, frags, xyz):
    """Return the two smallest heavy-atom distances between fragment 1 and 2."""
    hv = np.array([s != "H" for s in syms])
    idx_a = np.where((frags == 1) & hv)[0]
    idx_b = np.where((frags == 2) & hv)[0]
    if len(idx_a) == 0 or len(idx_b) == 0:
        return None, None
    D = np.linalg.norm(xyz[idx_a][:, None, :] - xyz[idx_b][None, :, :], axis=-1)
    flat = np.sort(D.ravel())
    return float(flat[0]), float(flat[1] if len(flat) >= 2 else flat[0])


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

    idx = new.index.intersection(old.index)
    print(f"comparable rxns : {len(idx)}")

    rows = []
    for c in CHANNELS:
        a = old.loc[idx, f"{c}_dft"].values.astype(float)   # low-level TS
        b = new.loc[idx, f"{c}_dft"].values.astype(float)   # B3LYP TS (phase5-style col in new)
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

    # ---- Bond-length shift: low-level TS (from existing eda.inp) vs B3LYP TS ----
    corr_rows = []
    for rid in idx.intersection(meta.index):
        rid_int = int(rid)
        p_low = EXT_EDA / f"rxn_{rid_int:04d}" / "eda.inp"
        if not p_low.exists():
            continue
        syms, frags, xyz = read_eda_xyz(p_low)
        if syms is None:
            continue
        d1_low, d2_low = closest_two_heavy_contacts(syms, frags, xyz)
        if d1_low is None:
            continue
        r_meta = meta.loc[rid_int]
        # meta stored B3LYP distances but on the *formed-bond* indices from
        # the ds3 filename convention (not necessarily the two closest heavy
        # contacts). Both proxies for "forming bonds" — use as-is.
        d1_b, d2_b = float(r_meta["formed_d1_b3lyp"]), float(r_meta["formed_d2_b3lyp"])
        # Match by nearest (avoid label swap): assign each low→b to minimize error
        pair_a = abs(d1_low - d1_b) + abs(d2_low - d2_b)
        pair_b = abs(d1_low - d2_b) + abs(d2_low - d1_b)
        if pair_b < pair_a:
            d1_b, d2_b = d2_b, d1_b
        corr_rows.append(dict(rxn_id=rid_int,
                              d1_low=d1_low, d1_b3lyp=d1_b, d1_diff=d1_b - d1_low,
                              d2_low=d2_low, d2_b3lyp=d2_b, d2_diff=d2_b - d2_low))

    if corr_rows:
        cdf = pd.DataFrame(corr_rows).set_index("rxn_id")
        cdf.to_csv(BASE / "artifacts" / "bond_length_shifts.csv")
        print("\n=== Formed-bond length shift (B3LYP − low-level, Å) ===")
        print(f"  d1: mean={cdf['d1_diff'].mean():+.3f}  |diff|_med={cdf['d1_diff'].abs().median():.3f}  p95={cdf['d1_diff'].abs().quantile(0.95):.3f}")
        print(f"  d2: mean={cdf['d2_diff'].mean():+.3f}  |diff|_med={cdf['d2_diff'].abs().median():.3f}  p95={cdf['d2_diff'].abs().quantile(0.95):.3f}")

        # Correlation between |Δbond| (mean of |d1_diff|, |d2_diff|) and |Δchannel|
        print("\n|Δchannel| vs |Δbond-len| Pearson r:")
        for c in ["pauli", "elst", "oi"]:
            common = new.index.intersection(old.index).intersection(cdf.index)
            if len(common) < 5:
                continue
            ch_diff = np.abs(new.loc[common, f"{c}_dft"].values.astype(float)
                             - old.loc[common, f"{c}_dft"].values.astype(float))
            bd_diff = 0.5 * (cdf.loc[common, "d1_diff"].abs()
                             + cdf.loc[common, "d2_diff"].abs()).values
            m = ~(np.isnan(ch_diff) | np.isnan(bd_diff))
            r_val = float(np.corrcoef(ch_diff[m], bd_diff[m])[0, 1]) if m.sum() >= 3 else float("nan")
            print(f"  {c}: r = {r_val:+.3f}  (n={int(m.sum())})")

    # GATE-4 status
    worst = float(R["delta_propagated"].max())
    n_fail = int((R["verdict"] == "FAIL").sum())
    n_cond = int((R["verdict"] == "CONDITIONAL").sum())
    n_pass = int((R["verdict"] == "PASS").sum())
    with open(BASE / "artifacts" / "GATE4_STATUS.txt", "w") as f:
        f.write(f"worst_delta_prop={worst:.3f} PASS={n_pass} CONDITIONAL={n_cond} FAIL={n_fail}\n")
        for _, r in R.iterrows():
            f.write(f"  {r['channel']:>6}: δ_prop={r['delta_propagated']:.3f} {r['verdict']}\n")
    overall = "FAIL" if n_fail else ("CONDITIONAL" if n_cond else "PASS")
    print(f"\n=== GATE-4 {overall} (worst δ_propagated = {worst:.3f}) ===")


if __name__ == "__main__":
    main()
