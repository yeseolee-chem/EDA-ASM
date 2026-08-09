#!/usr/bin/env python3
"""SPEC_EVAL_01 Step 3: figures with Arm A/B side-by-side per metric."""
import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/tt_eda_kisti_results/eval")
FIG = BASE / "fig"; FIG.mkdir(exist_ok=True)

# color spec: Arm A tomato, Arm B teal, Espley anchor gray-dashed
ARM_COLORS = {"A": "#e07a5f", "B": "#3d9d9b"}
ARM_LABELS = {"A": "Arm A (AM1)", "B": "Arm B (AM1+xTB)"}
ESPLEY_ANCHOR = {
    "distortion_energy_1_dft": 2.55,
    "distortion_energy_2_dft": 2.37,
    "interaction_energies_dft": 2.46,
}
CH_INT = ["elst_dft","pauli_dft","oi_dft","disp_dft","cpcm_dft","cds_dft"]
CH_STRAIN = ["distortion_energy_1_dft","distortion_energy_2_dft"]
CH_ESPLEY = ["e_barrier_dft","q_barrier_dft","interaction_energies_dft","sum_distortion_energies_dft"]
BUDGET = 1.0 / np.sqrt(6)

def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=140, bbox_inches='tight')
    fig.savefig(FIG / f"{name}.pdf", bbox_inches='tight')
    plt.close(fig)
    print(f"  saved: {name}.png / .pdf")

def fig_bar_side(df, targets, metric, ylabel, title, name, anchor=None, model="svr"):
    """Side-by-side Arm A vs B bars per target for given metric."""
    sub = df[(df['model']==model) & (df['target'].isin(targets))]
    if sub.empty:
        print(f"  [skip] no data for {name} (model={model})")
        return
    pivot = sub.pivot_table(index='target', columns='arm', values=metric).reindex(targets)
    err = sub.pivot_table(index='target', columns='arm', values=f'{metric}_seed_sd' if f'{metric}_seed_sd' in sub.columns else metric).reindex(targets) * 0
    fig, ax = plt.subplots(figsize=(max(6, 0.7*len(targets)+2), 4.5))
    x = np.arange(len(targets)); w = 0.35
    for i, arm in enumerate(['A','B']):
        if arm in pivot.columns:
            vals = pivot[arm].values
            ax.bar(x + (i-0.5)*w, vals, w, label=ARM_LABELS[arm], color=ARM_COLORS[arm])
    # Espley anchor overlay
    if anchor:
        for t in targets:
            if t in anchor:
                ti = targets.index(t)
                ax.axhline(anchor[t], xmin=(ti+0.05)/len(targets), xmax=(ti+0.95)/len(targets),
                          color='gray', linestyle='--', linewidth=1.5)
        # anchor legend entry
        ax.plot([], [], 'k--', label='Espley anchor')
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace('_dft','').replace('_energies','').replace('energy_','') for t in targets],
                     rotation=30, ha='right', fontsize=9)
    ax.set_ylabel(ylabel); ax.set_title(f"{title} (model: {model.upper()})")
    ax.legend(loc='best', fontsize=9); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    save(fig, name)

def main():
    agg = pd.read_parquet(BASE / "metrics_agg.parquet")
    print(f"loaded: {len(agg)} rows")

    all_targets = sorted(agg['target'].unique())
    # ordering: strain (2) + interaction (6) + Espley reference (4) + eint
    ordered = [t for group in [CH_STRAIN, CH_INT, CH_ESPLEY, ['eint_dft']] for t in group if t in all_targets]

    # F1: NMAE per channel, Arm A vs B (test), SVR
    fig_bar_side(agg, ordered, 'nmae_seed_mean', 'test NMAE',
                 "F1: NMAE per channel (5-seed mean, SVR)", "F1_nmae_svr")
    # Also for KRR
    fig_bar_side(agg, ordered, 'nmae_seed_mean', 'test NMAE',
                 "F1b: NMAE per channel (5-seed mean, KRR)", "F1b_nmae_krr", model="krr")

    # F2: MAE per channel (raw)
    fig_bar_side(agg, ordered, 'mae_seed_mean', 'test MAE (kcal/mol)',
                 "F2: MAE per channel (5-seed mean, SVR)", "F2_mae_svr",
                 anchor={"distortion_energy_1_dft":2.55,"distortion_energy_2_dft":2.37,
                         "interaction_energies_dft":2.46})

    # F3: budget check — RMSE per channel vs BUDGET (0.41)
    # rmse_seed_mean not aggregated. Compute from master.
    master = pd.read_parquet(BASE / "metrics_master.parquet")
    test = master[master['split']=='test']
    rmse_agg = test.groupby(['arm','model','target'])['rmse'].mean().reset_index()
    rmse_agg = rmse_agg.rename(columns={'rmse':'rmse_seed_mean'})
    rmse_pivot = rmse_agg[(rmse_agg['model']=='svr') & (rmse_agg['target'].isin(CH_INT))].pivot_table(
        index='target', columns='arm', values='rmse_seed_mean').reindex(CH_INT)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(CH_INT)); w = 0.35
    for i, arm in enumerate(['A','B']):
        if arm in rmse_pivot.columns:
            ax.bar(x + (i-0.5)*w, rmse_pivot[arm].values, w, label=ARM_LABELS[arm], color=ARM_COLORS[arm])
    ax.axhline(BUDGET, color='red', linestyle=':', linewidth=1.5, label=f'Budget (0.41 kcal)')
    ax.set_xticks(x); ax.set_xticklabels([c.replace('_dft','') for c in CH_INT], rotation=20, ha='right')
    ax.set_ylabel('test RMSE (kcal/mol)')
    ax.set_title('F3: Channel budget check (SVR, 5-seed mean) — 6 interaction channels only')
    ax.legend(); ax.grid(axis='y', alpha=0.3); plt.tight_layout()
    save(fig, "F3_budget_svr")

    # F4: sum reconstruction headline (MAE_sum, HIT1)
    sum_path = BASE / "metrics_sum.parquet"
    if sum_path.exists():
        S = pd.read_parquet(sum_path)
        for model in ['svr','krr','ridge']:
            sm = S[S['model']==model]
            if sm.empty: continue
            fig, ax1 = plt.subplots(figsize=(6, 4.5))
            arms = sorted(sm['arm'].unique()); x = np.arange(len(arms))
            colors = [ARM_COLORS[a] for a in arms]
            ax1.bar(x - 0.2, sm.set_index('arm').reindex(arms)['mae_sum_mean'].values, 0.4,
                    color=colors, label='MAE_sum')
            ax1.axhline(2.46, color='gray', linestyle='--', label='Espley int MAE 2.46')
            ax1.set_ylabel('MAE_sum (kcal/mol)'); ax1.grid(axis='y', alpha=0.3)
            ax2 = ax1.twinx()
            ax2.bar(x + 0.2, 100*sm.set_index('arm').reindex(arms)['hit1_mean'].values, 0.4,
                    color=[ARM_COLORS[a] for a in arms], alpha=0.5, label='hit@1kcal %')
            ax2.set_ylabel('hit@1kcal (%)')
            ax1.set_xticks(x); ax1.set_xticklabels([ARM_LABELS[a] for a in arms])
            ax1.set_title(f"F4: sum-of-6-channels reconstruction ({model.upper()})")
            ax1.legend(loc='upper left'); ax2.legend(loc='upper right')
            plt.tight_layout(); save(fig, f"F4_sum_{model}")

    # F6: train vs test NMAE scatter (overfit check) — skip if no train split in export
    nmae = master.groupby(['arm','model','target','split'])['nmae'].mean().reset_index()
    tr = nmae[nmae['split']=='train'].rename(columns={'nmae':'nmae_train'}).drop(columns='split')
    te = nmae[nmae['split']=='test'].rename(columns={'nmae':'nmae_test'}).drop(columns='split')
    if len(tr) == 0:
        print("  [skip F6] train split not in exports; skipping overfit-check figure")
        return
    j = tr.merge(te, on=['arm','model','target'])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True, sharex=True)
    for ax, arm in zip(axes, ['A','B']):
        jj = j[j['arm']==arm].dropna(subset=['nmae_train','nmae_test'])
        for model in sorted(jj['model'].unique()):
            m = jj[jj['model']==model]
            ax.scatter(m['nmae_train'], m['nmae_test'], label=model, s=45, alpha=0.75)
        lim = max(jj['nmae_train'].max() if len(jj) else 1.0,
                  jj['nmae_test'].max() if len(jj) else 1.0, 1.1) * 1.05
        ax.plot([0,lim],[0,lim],'k--',alpha=0.4)
        ax.set_xlim(0,lim); ax.set_ylim(0,lim)
        ax.set_xlabel('train NMAE'); ax.set_ylabel('test NMAE' if arm=='A' else '')
        ax.set_title(f'{ARM_LABELS[arm]}')
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle('F6: train vs test NMAE (overfit check)')
    plt.tight_layout(); save(fig, "F6_train_vs_test")

if __name__ == "__main__":
    main()
