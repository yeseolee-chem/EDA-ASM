#!/usr/bin/env python3
"""Extract per-reaction test/train predictions from ml_results.pkl for both arms.
Output long-format parquet: rxn × arm × model × target × seed × split × y_true × y_pred."""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split

BASE = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/tt_eda_kisti_results")
OUT_DIR = BASE / "eval"
OUT_DIR.mkdir(exist_ok=True)

ARMS = {
    "A": {"pkl": BASE / "manual_tt_7ch.pkl",     "ml_dir": BASE / "pipeline_arm_a/ml_results"},
    "B": {"pkl": BASE / "manual_tt_7ch_xtb.pkl", "ml_dir": BASE / "pipeline_7ch/ml_results"},
}
SEEDS = [22, 23, 14, 1, 2]

def get_dft_targets(df):
    return [c for c in df.columns if '_dft' in c]

def do_split(X, y, seed):
    """Reproduce ml_analysis split: 80/10/10 with random_state=seed."""
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)
    X_te, X_val, y_te, y_val = train_test_split(X_te, y_te, test_size=0.5, random_state=seed)
    return X_tr, X_val, X_te, y_tr, y_val, y_te

def main():
    all_rows = []
    for arm, cfg in ARMS.items():
        df = pd.read_pickle(cfg["pkl"])
        rxns = df['reaction_number'].astype(int).values
        target_cols = get_dft_targets(df)
        # Split predictor/target
        y = df[target_cols]
        X = df.drop(columns=target_cols + ['reaction_number'])

        for seed in SEEDS:
            X_tr, X_val, X_te, y_tr, y_val, y_te = do_split(X, y, seed)
            tr_idx = y_tr.index; val_idx = y_val.index; te_idx = y_te.index
            tr_rxn = df.loc[tr_idx, 'reaction_number'].astype(int).values
            val_rxn = df.loc[val_idx, 'reaction_number'].astype(int).values
            te_rxn = df.loc[te_idx, 'reaction_number'].astype(int).values

            ml_pkl = cfg["ml_dir"] / f"seed_{seed}" / "ml_results.pkl"
            if not ml_pkl.exists():
                print(f"[skip] {arm} seed={seed}: no ml_results.pkl")
                continue
            R = pd.read_pickle(ml_pkl)

            for _, r in R.iterrows():
                model_target = r['model_target']
                # parse model/target
                if model_target.startswith('2_st_nn_'):
                    model, target = "2L_NN", model_target[len("2_st_nn_"):]
                elif model_target.startswith('4_st_nn_'):
                    model, target = "4L_NN", model_target[len("4_st_nn_"):]
                else:
                    model, target = model_target.split('_', 1)
                # Extract predictions
                try:
                    y_true_te = np.asarray(r['y_test_true'][0]).flatten()
                    y_pred_te = np.asarray(r['y_test_pred_values'][0]).flatten()
                    n_te = min(len(y_true_te), len(te_rxn), len(y_pred_te))
                    for i in range(n_te):
                        all_rows.append({
                            'reaction_number': int(te_rxn[i]),
                            'arm': arm, 'model': model, 'target': target,
                            'seed': int(seed), 'split': 'test',
                            'y_true': float(y_true_te[i]), 'y_pred': float(y_pred_te[i]),
                        })
                except Exception as e:
                    print(f"  [warn] {arm} seed={seed} {model_target}: {type(e).__name__}: {e}")
        print(f"[arm {arm}] processed {len(SEEDS)} seeds")

    out = pd.DataFrame(all_rows)
    out.to_parquet(OUT_DIR / "preds_all.parquet")
    print(f"\n[ok] eval/preds_all.parquet: {len(out)} rows")
    print(f"     arms: {out['arm'].unique().tolist()}")
    print(f"     models: {out['model'].unique().tolist()}")
    print(f"     targets: {out['target'].nunique()}")
    print(f"     seeds: {sorted(out['seed'].unique())}")

if __name__ == "__main__":
    main()
