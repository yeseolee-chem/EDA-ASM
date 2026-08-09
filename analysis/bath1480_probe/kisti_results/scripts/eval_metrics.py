#!/usr/bin/env python3
"""SPEC_EVAL_01 Step 1-2: metrics per (arm, model, target, seed, split) + bootstrap CI + sum recon."""
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/tt_eda_kisti_results/eval")

HIT_THRESH = 1.0
N_INT_CH = 6
BUDGET = HIT_THRESH / np.sqrt(N_INT_CH)   # 0.4082 kcal/mol
B_BOOT = 1000
CI = 0.95
CH_INT = ["elst_dft","pauli_dft","oi_dft","disp_dft","cpcm_dft","cds_dft"]
CH_STRAIN = ["distortion_energy_1_dft","distortion_energy_2_dft"]

ESPLEY_ANCHOR = {
    "distortion_energy_1_dft": {"mae": 2.55, "mad_cohort": 5.60, "nmae": 0.455},
    "distortion_energy_2_dft": {"mae": 2.37, "mad_cohort": 5.17, "nmae": 0.459},
    "interaction_energies_dft": {"mae": 2.46, "mad_cohort": 4.80, "nmae": 0.513},
}

def bootstrap_ci(y_true, y_pred, B=B_BOOT, ci=CI, rng=None):
    if rng is None: rng = np.random.default_rng(42)
    n = len(y_true)
    maes = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        maes[b] = np.mean(np.abs(y_true[idx] - y_pred[idx]))
    lo = np.percentile(maes, 100*(1-ci)/2)
    hi = np.percentile(maes, 100*(1+ci)/2)
    return maes.mean(), lo, hi

def main():
    preds = pd.read_parquet(BASE / "preds_all.parquet")
    print(f"loaded preds: {len(preds)}, arms={sorted(preds['arm'].unique())}, models={sorted(preds['model'].unique())}, targets={preds['target'].nunique()}")

    # 1. per-(arm, model, target, seed, split) metrics
    rows = []
    for (arm, model, target, seed, split), g in preds.groupby(['arm','model','target','seed','split']):
        y_t = g['y_true'].to_numpy(); y_p = g['y_pred'].to_numpy()
        err = y_p - y_t
        mae = np.mean(np.abs(err))
        rmse = np.sqrt(np.mean(err**2))
        mad = np.mean(np.abs(y_t - y_t.mean()))
        nmae = mae/mad if mad > 0 else np.nan
        y_range = y_t.max() - y_t.min()
        prange = 100*mae/y_range if y_range > 0 else np.nan
        rows.append({
            'arm': arm, 'model': model, 'target': target, 'seed': seed, 'split': split,
            'mae': mae, 'rmse': rmse, 'mad': mad, 'nmae': nmae, 'prange': prange, 'n': len(y_t)
        })
    metrics = pd.DataFrame(rows)

    # 2. Seed-averaged with bootstrap CI (test only)
    agg_rows = []
    for (arm, model, target), g in metrics[metrics['split']=='test'].groupby(['arm','model','target']):
        # naive: mean + SD across seeds
        mae_mean = g['mae'].mean(); mae_sd = g['mae'].std(ddof=0)
        nmae_mean = g['nmae'].mean(); nmae_sd = g['nmae'].std(ddof=0)
        # bootstrap: pool all seed-test preds, resample
        subset = preds[(preds['arm']==arm)&(preds['model']==model)&(preds['target']==target)&(preds['split']=='test')]
        if len(subset) > 10:
            m, lo, hi = bootstrap_ci(subset['y_true'].to_numpy(), subset['y_pred'].to_numpy())
        else:
            m = lo = hi = np.nan
        agg_rows.append({
            'arm': arm, 'model': model, 'target': target,
            'mae_seed_mean': mae_mean, 'mae_seed_sd': mae_sd,
            'nmae_seed_mean': nmae_mean, 'nmae_seed_sd': nmae_sd,
            'mae_boot': m, 'mae_boot_lo': lo, 'mae_boot_hi': hi,
            'n_seeds': len(g),
        })
    agg = pd.DataFrame(agg_rows)

    # 3. Sum reconstruction (headline): predict sum of 6 interaction channels, compare to eint_dft
    #    yhat_sum_int per (arm, model, seed) by summing channel predictions across CH_INT
    sum_rows = []
    for (arm, model, seed), g in preds[preds['split']=='test'].groupby(['arm','model','seed']):
        # pivot per rxn
        gp = g[g['target'].isin(CH_INT + ['eint_dft'])].pivot_table(
            index='reaction_number', columns='target', values=['y_true','y_pred'], aggfunc='first')
        if not all(f'y_pred' in gp.columns.get_level_values(0) for _ in [0]):
            continue
        try:
            yhat_sum = np.zeros(len(gp))
            for ch in CH_INT:
                if ('y_pred', ch) in gp.columns:
                    yhat_sum = yhat_sum + gp[('y_pred', ch)].fillna(0).to_numpy()
            if ('y_true','eint_dft') in gp.columns:
                y_eint = gp[('y_true','eint_dft')].to_numpy()
                e_sum = yhat_sum - y_eint
                mae_sum = np.mean(np.abs(e_sum))
                rmse_sum = np.sqrt(np.mean(e_sum**2))
                hit1 = np.mean(np.abs(e_sum) <= HIT_THRESH)
                mad_eint = np.mean(np.abs(y_eint - y_eint.mean()))
                sum_rows.append({
                    'arm': arm, 'model': model, 'seed': seed,
                    'mae_sum': mae_sum, 'rmse_sum': rmse_sum,
                    'nmae_sum': mae_sum/mad_eint if mad_eint > 0 else np.nan,
                    'hit1': hit1, 'n_rxn': len(gp),
                })
        except Exception as e:
            print(f"  sum warn: {arm}/{model}/seed={seed}: {e}")
    sum_df = pd.DataFrame(sum_rows)
    # Aggregate sum metrics per (arm, model)
    if len(sum_df):
        sum_agg = sum_df.groupby(['arm','model']).agg(
            mae_sum_mean=('mae_sum','mean'), mae_sum_sd=('mae_sum','std'),
            rmse_sum_mean=('rmse_sum','mean'), nmae_sum_mean=('nmae_sum','mean'),
            hit1_mean=('hit1','mean'), n_seeds=('seed','count')
        ).reset_index()
    else:
        sum_agg = pd.DataFrame()

    # Save
    metrics.to_parquet(BASE / "metrics_master.parquet")
    agg.to_parquet(BASE / "metrics_agg.parquet")
    if len(sum_agg): sum_agg.to_parquet(BASE / "metrics_sum.parquet")
    if len(sum_df): sum_df.to_parquet(BASE / "metrics_sum_perseed.parquet")

    print(f"\n[ok] metrics_master.parquet: {len(metrics)} rows")
    print(f"[ok] metrics_agg.parquet: {len(agg)} (arm, model, target) combos")
    print(f"[ok] metrics_sum.parquet: {len(sum_agg)} sum-headline rows")

if __name__ == "__main__":
    main()
