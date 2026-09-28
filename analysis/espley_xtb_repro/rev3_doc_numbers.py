#!/usr/bin/env python3
"""Numbers for the rev 3 SUMMARY.md corrections (read-only on results).

1. B2 replacement (method ①): barrier reconstructed from predicted components,
   same rxns/seeds as the direct barrier test folds, vs the direct barrier model.
2. c_ghost statistics (rev 3) and the rev 2 gap it replaces, by charge.
3. Markdown tables from the rev 3 CSVs: charge breakdown, group split,
   MMP split metrics (S3+S8), margin calibration (S9, random split).
"""
import numpy as np
import pandas as pd

R = "/gpfs/tmp_cpu2/yeseo1ee/espley_xtb"
RES = "/home1/yeseo1ee/projects/eda-asm-prediction/analysis/espley_xtb_repro/results"
MODELS = ["Ridge", "KRR_rbf", "SVR_rbf", "XGB"]
CH6 = ["dft_elst_dft", "dft_pauli_dft", "dft_oi_dft", "dft_disp_dft", "dft_cpcm_dft", "dft_cds_dft"]

# ---------------------------------------------------------------- 1. composite barrier
p = pd.read_parquet(f"{R}/predictions.parquet")
p = p[p.feature_set == 73]
print("## 1. barrier reconstructed from predicted components (ESPLEY73, Protocol A test folds)")
for m in MODELS:
    q = p[p.model == m]
    y = q.pivot_table(index=["seed", "rxn_id"], columns="target", values="y")
    yh = q.pivot_table(index=["seed", "rxn_id"], columns="target", values="yhat")
    assert y.notna().all().all() and yh.notna().all().all(), "targets do not share test folds"
    ident = (y["dft_d1_kcal"] + y["dft_d2_kcal"] + y["dft_e_bond_kcal"] + y["dft_c_ghost_kcal"]
             - y["dft_barrier_kcal"]).abs().max()
    comp = {
        "direct barrier": yh["dft_barrier_kcal"],
        "d1+d2+eint_spe": yh["dft_d1_kcal"] + yh["dft_d2_kcal"] + yh["dft_eint_spe_kcal"],
        "d1+d2+e_bond+c_ghost": yh["dft_d1_kcal"] + yh["dft_d2_kcal"] + yh["dft_e_bond_kcal"] + yh["dft_c_ghost_kcal"],
        "d1+d2+Σ6ch+c_ghost": yh["dft_d1_kcal"] + yh["dft_d2_kcal"] + yh[CH6].sum(axis=1) + yh["dft_c_ghost_kcal"],
    }
    parts9 = ["dft_d1_kcal", "dft_d2_kcal"] + CH6 + ["dft_c_ghost_kcal"]
    quad = np.sqrt(sum((yh[c] - y[c]).abs().groupby(level=0).mean().mean() ** 2 for c in parts9))
    lin = sum((yh[c] - y[c]).abs().groupby(level=0).mean().mean() for c in parts9)
    print(f"### {m}  (true identity residual {ident:.1e}; n per seed {len(y) // 5})")
    for k, v in comp.items():
        per_seed = (v - y["dft_barrier_kcal"]).abs().groupby(level=0).mean()
        print(f"  {k:24s} MAE {per_seed.mean():.2f} ± {per_seed.std(ddof=0):.2f}")
    print(f"  9-part component MAEs: sum {lin:.2f}, quadrature {quad:.2f}")

# ---------------------------------------------------------------- 2. c_ghost / rev 2 gap
print("\n## 2. c_ghost (rev 3) vs rev 2 gap eint_spe - e_bond, by charge2")
new = pd.read_parquet(f"{R}/xtb_features.parquet")
old = pd.read_parquet(f"{R}/rev2_backup_20260915/xtb_features.parquet")
for name, df, col in (("rev3 c_ghost", new, None), ("rev2 gap", old, None)):
    g = df["dft_eint_spe_kcal"] - df["dft_e_bond_kcal"]
    q2 = df["charge2"].fillna(0).astype(int)
    r = np.corrcoef(g, q2)[0, 1]
    print(f"  {name:13s} mean {g.mean():+.2f} sd {g.std():.2f} |.|>5: {int((g.abs() > 5).sum())}/{len(g)}  "
          f"q2=0 {g[q2 == 0].mean():+.2f}  q2=-2 {g[q2 == -2].mean():+.2f}  q2=+1 {g[q2 == 1].mean():+.2f}  r(.,charge2) {r:+.3f}")

# ---------------------------------------------------------------- 3. tables
LBL = {"dft_barrier_kcal": "barrier", "dft_d1_kcal": "d1", "dft_d2_kcal": "d2", "dft_eint_spe_kcal": "eint_spe",
       "dft_e_bond_kcal": "e_bond", "dft_elst_dft": "elst", "dft_pauli_dft": "Pauli", "dft_oi_dft": "OI",
       "dft_disp_dft": "disp", "dft_cpcm_dft": "CPCM", "dft_cds_dft": "CDS", "dft_c_ghost_kcal": "c_ghost"}
ORDER = list(LBL)
print("\n## 3a. charge breakdown (charge_breakdown.csv)")
cb = pd.read_csv(f"{RES}/charge_breakdown.csv").set_index("target")
print("| target | all | neutral (n={}) | charged (n={}) | q₂=−2 ({}) | q₂=+1 ({}) |".format(
    int(cb.n_neutral.iloc[0]), int(cb.n_charged.iloc[0]), int(cb.n_q_minus2.iloc[0]), int(cb.n_q_plus1.iloc[0])))
print("|---|---:|---:|---:|---:|---:|")
for t in ORDER:
    if t in cb.index and t != "dft_disp_dft":
        r = cb.loc[t]
        print(f"| {LBL[t]} | {r.mae_all:.2f} | {r.mae_neutral:.2f} | {r.mae_charged:.2f} | {r.mae_q_minus2:.2f} | {r.mae_q_plus1:.2f} |")
print("\n## 3b. group split (group_split.csv)")
gs = pd.read_csv(f"{RES}/group_split.csv").set_index("target")
print("| target | random | dph 그룹 홀드 | 배수 | dipole 그룹 홀드 | 배수 |\n|---|---:|---:|---:|---:|---:|")
for t in ORDER:
    if t in gs.index:
        r = gs.loc[t]
        print(f"| {LBL[t]} | {r.random:.2f} | {r.group_dipolarophile:.2f} | {r.ratio_dph:.2f} | {r.group_dipole:.2f} | {r.ratio_dip:.2f} |")
print("\n## 3c. MMP split metrics (split_metrics.csv)")
sm = pd.read_csv(f"{RES}/split_metrics.csv")
print("| split | n pairs | ΔMAE elst | ΔMAE Pauli | ΔMAE OI | ΔMAE CPCM | dom_agree | τ₉₅ | cov₉₅ | τ₉₉ | cov₉₉ |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
f = lambda v, d=2: "—" if pd.isna(v) else f"{v:.{d}f}"
for _, r in sm.iterrows():
    print(f"| {r.split} | {int(r.n_pairs):,} | {f(r.dmae_elst)} | {f(r.dmae_pauli)} | {f(r.dmae_oi)} | {f(r.dmae_cpcm)} | "
          f"{f(r.dom_agree_all, 3)} | {f(r.tau_95, 1)} | {f(r.cov_95)} | {f(r.tau_99, 1)} | {f(r.cov_99)} |")
print("\n## 3d. margin calibration, random split (margin_calibration.csv)")
mc = pd.read_csv(f"{RES}/margin_calibration.csv")
mr = mc[mc.split == "random"]
print("| τ (kcal/mol) | coverage | agree | true-margin agree |\n|---:|---:|---:|---:|")
for _, r in mr.iterrows():
    print(f"| {r.tau:.1f} | {r.coverage:.2f} | {r.agree:.3f} | {r.agree_true_margin:.3f} |")
dc = mc[mc.split == "dipole_class"]
print("\ndipole_class:", "; ".join(f"τ {r.tau:.1f}: cov {r.coverage:.2f} agree {r.agree:.3f}" for _, r in dc.iterrows()))
