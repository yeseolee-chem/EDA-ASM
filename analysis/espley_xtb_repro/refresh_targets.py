#!/usr/bin/env python3
"""refresh_targets.py — replace the dft_* target columns in xtb_features.parquet with values from a new
labels_all.json (method ① assembly). xTB features are untouched, so no xTB re-run is needed.

usage: python refresh_targets.py <labels_all.json> <xtb_features.parquet in> <xtb_features.parquet out>
"""
import json, sys
import numpy as np, pandas as pd
EH2KCAL = 627.5094740631
CH = ("elst_dft", "pauli_dft", "oi_dft", "disp_dft", "cpcm_dft", "cds_dft")

lab_path, fin, fout = sys.argv[1:4]
L = {int(r["rxn_id"]): r for r in json.load(open(lab_path))}
df = pd.read_parquet(fin)
miss = [i for i in df.rxn_id if int(i) not in L]
if miss:
    sys.exit(f"GATE: {len(miss)} rxn_ids missing from labels, e.g. {miss[:5]}")

def g(i, k): return L[int(i)][k]
e_ab = df.rxn_id.map(lambda i: g(i, "e_ab_eh"))
f1d = df.rxn_id.map(lambda i: g(i, "e_frag1_dist_eh")); f2d = df.rxn_id.map(lambda i: g(i, "e_frag2_dist_eh"))
f1r = df.rxn_id.map(lambda i: g(i, "e_frag1_rel_eh"));  f2r = df.rxn_id.map(lambda i: g(i, "e_frag2_rel_eh"))
old = df.filter(like="dft_").copy()
df["dft_barrier_kcal"] = (e_ab - f1r - f2r) * EH2KCAL
df["dft_eint_spe_kcal"] = (e_ab - f1d - f2d) * EH2KCAL
for k in ("d1_kcal", "d2_kcal", "e_bond_kcal"):
    df["dft_" + k] = df.rxn_id.map(lambda i: g(i, k))
for ch in CH:
    df["dft_" + ch] = df.rxn_id.map(lambda i: g(i, ch))
df["dft_c_ghost_kcal"] = df["dft_barrier_kcal"] - (df["dft_d1_kcal"] + df["dft_d2_kcal"] + df["dft_e_bond_kcal"])
df = df.drop(columns=[c for c in ("dft_barrier_eda", "dft_bsse_gap") if c in df])
df["flag_negative_strain"] = (df["dft_d1_kcal"] < -0.5) | (df["dft_d2_kcal"] < -0.5)

# ---- gates (method ①)
g1 = (df.dft_barrier_kcal - (df.dft_d1_kcal + df.dft_d2_kcal + df.dft_eint_spe_kcal)).abs().max()
chk_d1 = (df.dft_d1_kcal - (f1d - f1r) * EH2KCAL).abs().max()
if g1 > 1e-6 or chk_d1 > 1e-6:
    sys.exit(f"GATE: method-① identity failed (g1 {g1:.2e}, d1 {chk_d1:.2e}) — labels are not own-basis assembled")
print(f"g1 max {g1:.2e} | d1 check {chk_d1:.2e} | c_ghost mean {df.dft_c_ghost_kcal.mean():+.3f} sd {df.dft_c_ghost_kcal.std():.3f}")
for c in sorted(set(old.columns) & set(df.columns)):
    d = (df[c] - old[c]).abs()
    print(f"  {c:20s} changed rows {int((d > 1e-6).sum()):5d}   max|Δ| {d.max():8.3f}")
df.to_parquet(fout, index=False)
print(f"wrote {fout}  ({len(df)} rows)")
