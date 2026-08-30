#!/usr/bin/env python3
"""SPEC16rev Step 4 — 8-check QC gate. All must pass to authorize v2 label export.

References (218-rxn spec15 measurement):
  ① 6-channel sum vs e_bond    max|Δ| 0.098 kcal/mol
  ② strain algebra              max|Δ| 7e-15
  ③ closure |direct − recon|    max 2.1e-10
  ④ sign convention             100%
  ⑤ strain negative ratio       0.46%   (allow ≤ 2%)
  ⑥ missing / dup               0
  ⑦ Coley G_act correlation     r=0.9307 (need > 0.85)
  ⑧ BSSE audit                  mean(interaction_own − Σchannels) −3.41 ± 2.70
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/b3lyp_full")
COLEY_CSV = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/dipolar_cycloaddition/full_dataset.csv")

CHANNELS = ["elst_dft", "pauli_dft", "oi_dft", "disp_dft", "cpcm_dft", "cds_dft"]


def main():
    R = pd.read_csv(BASE / "artifacts" / "labels_b3lyp_full.csv")
    n = len(R)
    print(f"labels rows: {n}")

    checks = []

    # ① 6-channel sum vs e_bond
    ch_sum = R[CHANNELS].sum(axis=1)
    gap = (ch_sum - R["e_bond_kcal"]).abs()
    c1 = gap.max() < 0.5
    checks.append(("6ch sum == e_bond", c1, f"max|Δ|={gap.max():.4f}"))

    # ② strain algebra
    HK = 627.5094740631
    lhs1 = (R["e_frag1_dist_eh"] - R["e_frag1_rel_eh"]) * HK
    lhs2 = (R["e_frag2_dist_eh"] - R["e_frag2_rel_eh"]) * HK
    a1 = (lhs1 - R["d1_b3lyp"]).abs().max()
    a2 = (lhs2 - R["d2_b3lyp"]).abs().max()
    c2 = max(a1, a2) < 1e-8
    checks.append(("strain algebra", c2, f"max|Δ|={max(a1, a2):.2e}"))

    # ③ energy closure
    c3 = R["closure_gap"].max() < 0.01
    checks.append(("closure |direct−recon|", c3, f"max={R['closure_gap'].max():.2e}"))

    # ④ sign convention
    sign_ok = ((R["pauli_dft"] > 0) & (R["elst_dft"] < 0) &
               (R["oi_dft"] < 0) & (R["disp_dft"] < 0)).sum()
    c4 = sign_ok == n
    checks.append(("sign convention", c4, f"{sign_ok}/{n}"))

    # ⑤ strain negative ratio ≤ 2%
    neg = ((R["d1_b3lyp"] < 0) | (R["d2_b3lyp"] < 0)).sum()
    neg_frac = neg / n
    c5 = neg_frac <= 0.02
    checks.append(("strain negative ≤2%", c5, f"{neg}/{n} = {neg_frac*100:.2f}%"))

    # ⑥ missing / dup
    n_nan = R[CHANNELS + ["d1_b3lyp", "d2_b3lyp"]].isna().sum().sum()
    n_dup = R["rxn_id"].duplicated().sum()
    c6 = (n_nan == 0) and (n_dup == 0)
    checks.append(("no NaN / dup", c6, f"nan={n_nan} dup={n_dup}"))

    # ⑦ Coley G_act correlation vs our barrier_direct
    try:
        coley = pd.read_csv(COLEY_CSV).set_index("rxn_id")
        idx = R["rxn_id"][R["rxn_id"].isin(coley.index)]
        if len(idx) >= 30:
            a = coley.loc[idx, "G_act"].values
            b = R.set_index("rxn_id").loc[idx, "barrier_direct"].values
            m = ~(np.isnan(a) | np.isnan(b))
            r_val = float(np.corrcoef(a[m], b[m])[0, 1])
            c7 = r_val > 0.85
            checks.append(("Coley G_act correlation", c7,
                           f"r={r_val:.4f} (n={m.sum()})"))
        else:
            c7 = False
            checks.append(("Coley G_act correlation", c7, "insufficient overlap"))
    except Exception as e:
        c7 = False
        checks.append(("Coley G_act correlation", c7, f"error: {e}"))

    # ⑧ BSSE audit: interaction_own − Σchannels
    bsse = R["interaction_own"] - R[CHANNELS].sum(axis=1)
    bsse_mean = float(bsse.mean())
    bsse_std = float(bsse.std())
    c8 = abs(bsse_mean) < 10 and bsse_std < 5
    checks.append(("BSSE audit", c8, f"mean={bsse_mean:.3f} std={bsse_std:.3f}"))

    print("\n=== QC checks ===")
    all_ok = True
    for name, ok, detail in checks:
        tag = "✅" if ok else "❌"
        print(f"  {tag} {name:<30} {detail}")
        all_ok = all_ok and ok

    status = "PASS" if all_ok else "FAIL"
    with open(BASE / "artifacts" / "GATE3_STATUS.txt", "w") as f:
        f.write(f"{status}\n")
        for name, ok, detail in checks:
            f.write(f"  [{'OK' if ok else 'FAIL'}] {name}: {detail}\n")

    print(f"\n=== GATE-3 {status} ===")
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
