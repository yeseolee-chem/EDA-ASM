#!/usr/bin/env python3
"""train_ml_single.py — process ONE target for parallel ML array.

Usage:
    python train_ml_single.py <target_idx>   # 0..11 (ESPLEY_TARGET_SET=rev4: 0..8)

Runs Protocol A (Ridge/KRR/SVR/XGB with per-seed GridSearchCV; y standardized
via TransformedTargetRegressor) + Protocol B (Linear/Ridge/RF/GBR/XGB, pooled
OOF) on three feature sets (ESPLEY46 / ESPLEY54 / ESPLEY73) for a single
target. Writes per-target JSON + preds parquet to <ML_OUT>/ml_targets/.
aggregate_ml.py merges them.

rev 4 env (all optional; unset = rev 3 behaviour):
  ESPLEY_FEAT        feature parquet (default $ESPLEY_OUT/xtb_features.parquet)
  ESPLEY_ML_OUT      output root, ml_targets/ under it (default $ESPLEY_OUT)
  ESPLEY_ROWS        csv of rxn_id: exactly these rows in file order. Every one must be xtb_status ok, NaN-free in
                     the feature sets + targets and pass hygiene, else exit; no further row is dropped.
  ESPLEY_TARGET_SET  rev4 = the 9 reported targets, TARGET_SETS["rev4"] (default all 12)
  ESPLEY_PROTOCOLS   A = Protocol A only (default A,B)
  ESPLEY_GEOM        g0|g1|g2: recorded in the JSON and as preds column `geom`; must match the parquet `geom` tag
A target whose JSON + preds exist is skipped, or exits if they came from another geom / target set / protocols / feature
file (path or sha256) / rows file; both are written atomically (preds first).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

ROOT = Path(os.environ.get("ESPLEY_OUT", "/gpfs/tmp_cpu2/yeseo1ee/espley_xtb"))
FEAT_PATH = Path(os.environ.get("ESPLEY_FEAT") or ROOT / "xtb_features.parquet")
ML_OUT = Path(os.environ.get("ESPLEY_ML_OUT") or ROOT)
TARGETS_DIR = ML_OUT / "ml_targets"                  # created in main() (no side effect on import)
ROWS_PATH = os.environ.get("ESPLEY_ROWS") or None    # None = rev 3 row filter (ok ∩ hygiene)
GEOM = os.environ.get("ESPLEY_GEOM") or None
GEOM_TAG = {"g0": "dft", "g1": "g1", "g2": "g2"}     # ESPLEY_GEOM -> parquet `geom` value (xtb_slice --geom)
PROTOCOLS = {p.strip().upper() for p in (os.environ.get("ESPLEY_PROTOCOLS") or "A,B").split(",") if p.strip()}
N_JOBS = int(os.environ.get("ESPLEY_NJOBS", "8"))

DIST11 = ["dist_R_dip_ab", "dist_R_dip_bc", "dist_R_dip_ac", "dist_R_dph_ab",
          "dist_TS_dip_ab", "dist_TS_dip_bc", "dist_TS_dip_ac", "dist_TS_dph_ab",
          "dist_TS_form_ad", "dist_TS_form_be", "dist_TS_diag_ae"]
MULLIKEN15 = ([f"mulliken_R_dip_{k}" for k in (0, 1, 2)]
              + [f"mulliken_R_dph_{k}" for k in (0, 1)]
              + [f"mulliken_TSdip_{k}" for k in (0, 1, 2)]
              + [f"mulliken_TSdph_{k}" for k in (0, 1)]
              + [f"mulliken_TS_{k}" for k in (0, 1, 2, 3, 4)])
VALENCE15 = ([f"wbo_valence_R_dip_{k}" for k in (0, 1, 2)]
             + [f"wbo_valence_R_dph_{k}" for k in (0, 1)]
             + [f"wbo_valence_TSdip_{k}" for k in (0, 1, 2)]
             + [f"wbo_valence_TSdph_{k}" for k in (0, 1)]
             + [f"wbo_valence_TS_{k}" for k in (0, 1, 2, 3, 4)])
D_STRUCT41 = DIST11 + MULLIKEN15 + VALENCE15
E5 = ["xtb_e_barrier_kcal", "xtb_dist_dipole_kcal", "xtb_dist_dipolarophile_kcal",
      "xtb_sum_distortion_kcal", "xtb_interaction_kcal"]
CHAN8 = ["b_strain_1", "b_strain_2", "b_elst", "b_pauli", "b_oi", "b_disp", "b_cpcm", "b_cds"]
AUX19 = (["b_elst_scc", "b_disp_d4", "b_axc", "b_ct",
          "gap_ts", "gap_dip", "gap_dph", "mu_ts", "mu_dip", "mu_dph", "dmu_complexation",
          "is_charged"]                                         # N: is_charged added
         + [f"dsasa_{el}" for el in ("H", "C", "N", "O", "F", "Cl", "Br")])
ESPLEY46 = D_STRUCT41 + E5                       # ablation: 채널 블록 없음
ESPLEY54 = D_STRUCT41 + E5 + CHAN8               # 옛 55의 대응 (q_barrier만 제외)
ESPLEY73 = ESPLEY54 + AUX19                      # was ESPLEY72; +is_charged
FEATURE_SETS = {"ESPLEY46": ESPLEY46, "ESPLEY54": ESPLEY54, "ESPLEY73": ESPLEY73}

TARGETS = ["dft_barrier_kcal", "dft_d1_kcal", "dft_d2_kcal", "dft_eint_spe_kcal", "dft_e_bond_kcal",
           "dft_elst_dft", "dft_pauli_dft", "dft_oi_dft", "dft_disp_dft", "dft_cpcm_dft", "dft_cds_dft",
           "dft_c_ghost_kcal"]                   # 9th term: barrier = d1 + d2 + e_bond + c_ghost (method ①)
# rev 4 pre-registered report set (REV4 §3-1): barrier, d1, d2 + 6 EDA channels, index 0..8 in this order
TARGETS_REV4 = ["dft_barrier_kcal", "dft_d1_kcal", "dft_d2_kcal", "dft_elst_dft", "dft_pauli_dft", "dft_oi_dft",
                "dft_disp_dft", "dft_cpcm_dft", "dft_cds_dft"]
TARGET_SETS = {"all": TARGETS, "rev4": TARGETS_REV4}
PRE_ML = {"dft_barrier_kcal": "xtb_e_barrier_kcal", "dft_d1_kcal": "xtb_dist_dipole_kcal",
          "dft_d2_kcal": "xtb_dist_dipolarophile_kcal", "dft_eint_spe_kcal": "xtb_interaction_kcal",
          "dft_e_bond_kcal": "xtb_interaction_kcal",
          "dft_elst_dft": "b_elst", "dft_pauli_dft": "b_pauli", "dft_oi_dft": "b_oi",
          "dft_disp_dft": "b_disp", "dft_cpcm_dft": "b_cpcm", "dft_cds_dft": "b_cds",
          }
SEEDS = [22, 23, 14, 1, 2]
TUNE_SEED = 23                                    # kept only for record; N3 nested CV tunes per seed

# N1 — grids expanded down to the corners we hit in the previous run.
# N2 — every estimator wrapped in TransformedTargetRegressor(StandardScaler) → prefix "regressor__"
GRIDS = {
    "Ridge": (Ridge(),
              {"regressor__ridge__alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]}),
    "KRR_rbf": (KernelRidge(kernel="rbf"),
                {"regressor__kernelridge__alpha": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
                 "regressor__kernelridge__gamma": [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1]}),
    "SVR_rbf": (SVR(kernel="rbf"),
                {"regressor__svr__C": [1, 10, 100, 1000],
                 "regressor__svr__gamma": ["scale", 1e-3, 1e-2, 1e-1],
                 "regressor__svr__epsilon": [0.05, 0.1, 0.5]}),
    "XGB": (XGBRegressor(random_state=42, n_jobs=1, verbosity=0, tree_method="hist"),
            {"regressor__xgbregressor__n_estimators": [200, 500],
             "regressor__xgbregressor__max_depth": [3, 5, 7],
             "regressor__xgbregressor__learning_rate": [0.03, 0.1]}),
}


def _make_pipe(est):
    """N2: y-standardization via TransformedTargetRegressor wrapping (StandardScaler(x) → est)."""
    return TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(with_mean=True, with_std=True), est),
        transformer=StandardScaler(with_mean=True, with_std=True))


def _edge_hit(param_name, value, grid_values):
    """True if numeric value equals min or max of grid_values (ignores non-numeric like 'scale')."""
    numeric = [v for v in grid_values if isinstance(v, (int, float))]
    if not numeric or not isinstance(value, (int, float)):
        return False
    return value == min(numeric) or value == max(numeric)


def mets(y, p):
    r = float(np.corrcoef(y, p)[0, 1]) if (np.std(y) > 0 and np.std(p) > 0) else float("nan")
    mae = float(mean_absolute_error(y, p))
    mad = float(np.mean(np.abs(y - np.mean(y))))          # mean absolute deviation of the target
    return dict(mae=mae, nmae=mae / mad if mad > 0 else float("nan"),
                rmse=float(np.sqrt(mean_squared_error(y, p))),
                r2=float(r2_score(y, p)), pearson_r=r)


def split_80_10_10(n, seed):
    idx = np.arange(n)
    tr, rest = train_test_split(idx, test_size=0.20, random_state=seed)
    _, te = train_test_split(rest, test_size=0.50, random_state=seed)
    return tr, te


def protocol_A(df, feats, tgt, preds_store):
    """N3: nested CV — GridSearchCV is run on each seed's own train fold (no shared-tuning leak)."""
    X, y = df[feats].values, df[tgt].values
    out = {}
    for name, (est, grid) in GRIDS.items():
        tr_m, te_m, ranges = [], [], []
        per_seed_best, per_seed_edge = [], []
        for seed in SEEDS:
            tr, te = split_80_10_10(len(df), seed)
            gs = GridSearchCV(_make_pipe(est), grid,
                              cv=KFold(5, shuffle=True, random_state=seed),
                              scoring="neg_mean_absolute_error",
                              n_jobs=N_JOBS, error_score="raise")
            gs.fit(X[tr], y[tr])
            best = gs.best_params_
            mdl = gs.best_estimator_
            tr_m.append(mets(y[tr], mdl.predict(X[tr])))
            te_m.append(mets(y[te], mdl.predict(X[te])))
            ranges.append(float(y[te].max() - y[te].min()))
            preds_store.append(pd.DataFrame({"rxn_id": df.rxn_id.values[te], "seed": seed, "model": name,
                                             "target": tgt, "feature_set": len(feats),
                                             "y": y[te], "yhat": mdl.predict(X[te])}))
            per_seed_best.append(best)
            per_seed_edge.append({k: _edge_hit(k, v, grid[k]) for k, v in best.items()})
        test_mae = float(np.mean([m["mae"] for m in te_m]))
        edge_hits_total = sum(sum(1 for hit in d.values() if hit) for d in per_seed_edge)
        out[name] = dict(per_seed_best_params=per_seed_best,
                         per_seed_edge_hits=per_seed_edge,
                         edge_hits_total=edge_hits_total,
                         train_mae=float(np.mean([m["mae"] for m in tr_m])),
                         train_nmae=float(np.mean([m["nmae"] for m in tr_m])),
                         test_mae=test_mae, test_mae_sd=float(np.std([m["mae"] for m in te_m])),
                         test_nmae=float(np.mean([m["nmae"] for m in te_m])),
                         test_rmse=float(np.mean([m["rmse"] for m in te_m])),
                         test_r2=float(np.mean([m["r2"] for m in te_m])),
                         test_pearson_r=float(np.mean([m["pearson_r"] for m in te_m])),
                         test_range=float(np.mean(ranges)),
                         test_mae_pct_range=100 * test_mae / max(float(np.mean(ranges)), 1e-9))
    return out


def protocol_B(df, feats, tgt):
    X, y = df[feats].values, df[tgt].values
    kf = KFold(5, shuffle=True, random_state=42)
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge_alpha1": Ridge(alpha=1.0),
        "RandomForest_200": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=N_JOBS),
        "GBR_200_lr0.05": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42),
        "XGB_default": XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5,
                                    random_state=42, n_jobs=N_JOBS, verbosity=0, tree_method="hist"),
    }
    out = {}
    for name, est in models.items():
        oof = np.zeros_like(y)
        for tr, va in kf.split(X):
            oof[va] = make_pipeline(StandardScaler(with_mean=True, with_std=True), est).fit(X[tr], y[tr]).predict(X[va])
        out[name] = dict(pooled_oof=mets(y, oof))
    return out


def hygiene(df):
    """N (hygiene): keep d1 >= 0, d2 >= 0, d2 <= 50 — 라벨 위생 필터."""
    return (df["dft_d1_kcal"] >= 0) & (df["dft_d2_kcal"] >= 0) & (df["dft_d2_kcal"] <= 50)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_ml(feat_path=FEAT_PATH, rows_path=ROWS_PATH, targets=TARGETS, geom=GEOM):
    """(ML rows, record of how they were chosen).
    rows_path None: rev 3 filter = xtb_status ok, no NaN in the feature sets + all 12 TARGETS, hygiene.
    rows_path set: exactly its rxn_ids in file order; a row that is missing, duplicated, not ok, NaN in a feature /
    `targets` column or failing hygiene is a hard error — nothing is dropped."""
    df = pd.read_parquet(feat_path)
    feats = sorted(set(sum(FEATURE_SETS.values(), [])))
    if geom is not None:
        if geom not in GEOM_TAG:
            sys.exit(f"ESPLEY_GEOM must be one of {sorted(GEOM_TAG)}, got {geom!r}")
        tags = sorted(set(df["geom"].astype(str))) if "geom" in df else ["<no geom column: rev 3 parquet?>"]
        if tags != [GEOM_TAG[geom]]:
            sys.exit(f"GATE: ESPLEY_GEOM={geom} expects geom tag {GEOM_TAG[geom]!r}, {feat_path} has {tags}")
    info = dict(features_file=str(feat_path), features_sha256=sha256(feat_path), rows_file=None, rows_sha256=None)
    if rows_path is None:
        all_cols = sorted(set(feats) | set(TARGETS))
        ok = (df["xtb_status"] == "ok") & df[all_cols].notna().all(axis=1)
        hyg = hygiene(df)
        dropped = int((ok & ~hyg).sum())
        ml = df[ok & hyg].reset_index(drop=True)
        print(f"ML-ready rows: {len(ml)}  (hygiene dropped {dropped} rows: d1<0 / d2<0 / d2>50)", flush=True)
        return ml, dict(info, hygiene_dropped=dropped)

    ids = pd.read_csv(rows_path)["rxn_id"].astype(int).tolist()
    rid = df["rxn_id"].astype(int)
    need = sorted(set(feats) | set(targets))
    lacking = [c for c in need + ["xtb_status"] if c not in df]
    bad = [f"columns missing in {feat_path}: {lacking}"] if lacking else []
    if not ids:
        bad.append("rows file is empty")
    if len(set(ids)) != len(ids):
        bad.append(f"{len(ids) - len(set(ids))} duplicate rxn_id in the rows file")
    if rid.duplicated().any():
        bad.append(f"{int(rid.duplicated().sum())} duplicate rxn_id in {feat_path}")
    miss = sorted(set(ids) - set(rid))
    if miss:
        bad.append(f"{len(miss)} rxn_id absent from {feat_path}: {miss[:10]}")
    if bad:
        sys.exit(f"GATE ESPLEY_ROWS={rows_path}: " + "; ".join(bad))
    ml = df.set_index(rid.to_numpy()).loc[ids].reset_index(drop=True)
    for what, m in (("xtb_status != ok", ml["xtb_status"] != "ok"),
                    ("NaN in a feature / target column", ml[need].isna().any(axis=1)),
                    ("failing hygiene (d1<0 / d2<0 / d2>50)", ~hygiene(ml))):
        if m.any():
            bad.append(f"{int(m.sum())} rows {what}: {ml.loc[m, 'rxn_id'].astype(int).tolist()[:10]}")
    if bad:
        sys.exit(f"GATE ESPLEY_ROWS={rows_path}: " + "; ".join(bad))
    sha = sha256(rows_path)
    print(f"ML rows: {len(ml)} = {rows_path} in file order (sha256 {sha[:12]}); gated, nothing dropped", flush=True)
    return ml, dict(info, rows_file=str(Path(rows_path).resolve()), rows_sha256=sha)


def main():
    tidx = int(sys.argv[1])
    tset = os.environ.get("ESPLEY_TARGET_SET") or "all"
    if tset not in TARGET_SETS:
        sys.exit(f"ESPLEY_TARGET_SET must be one of {sorted(TARGET_SETS)}, got {tset!r}")
    if "A" not in PROTOCOLS or PROTOCOLS - {"A", "B"}:
        sys.exit(f"ESPLEY_PROTOCOLS must be A or A,B, got {sorted(PROTOCOLS)}")
    targets = TARGET_SETS[tset]
    if not 0 <= tidx < len(targets):
        sys.exit(f"target_idx {tidx} out of range 0..{len(targets) - 1} (ESPLEY_TARGET_SET={tset})")
    tgt = targets[tidx]
    print(f"[target {tidx}/{len(targets)}] {tgt}  set={tset} geom={GEOM} protocols={','.join(sorted(PROTOCOLS))}"
          f"\n  features {FEAT_PATH}\n  rows {ROWS_PATH}\n  out {TARGETS_DIR}", flush=True)

    out_json = TARGETS_DIR / f"target_{tidx:02d}_{tgt}.json"
    out_preds = TARGETS_DIR / f"preds_{tidx:02d}_{tgt}.parquet"
    if out_json.exists() and out_preds.exists():
        old = json.loads(out_json.read_text())
        want = dict(geom=GEOM, target_set=tset, protocols=sorted(PROTOCOLS), features_file=str(FEAT_PATH),
                    features_sha256=sha256(FEAT_PATH), rows_sha256=sha256(ROWS_PATH) if ROWS_PATH else None)
        stale = {k: (old.get(k), v) for k, v in want.items() if old.get(k) != v}
        if stale:
            sys.exit(f"{out_json} exists but was made with other inputs (existing, now): {stale} — move it away")
        print(f"skip: {out_json.name} + {out_preds.name} exist", flush=True)
        return

    ml, rows_info = load_ml(FEAT_PATH, ROWS_PATH, targets, GEOM)

    pre_ml = None
    if tgt in PRE_ML:
        feat = PRE_ML[tgt]
        d = ml[feat] - ml[tgt]
        pre_ml = dict(feature=feat, mae=float(d.abs().mean()), bias=float(d.mean()),
                      pearson_r=float(np.corrcoef(ml[feat], ml[tgt])[0, 1]))
        print(f"pre-ML {tgt} <- {feat} MAE {pre_ml['mae']:.2f}  bias {pre_ml['bias']:+.2f}", flush=True)

    result = dict(target=tgt, target_idx=tidx, target_set=tset, targets_in_set=targets, geom=GEOM,
                  protocols=sorted(PROTOCOLS), n_rows_ml=len(ml), **rows_info, pre_ml=pre_ml,
                  protocol_A={}, protocol_B={})
    preds = []
    for fs_name, feats in FEATURE_SETS.items():
        print(f"\n=== [{fs_name}] target={tgt} ({len(feats)} features) ===", flush=True)
        A = protocol_A(ml, feats, tgt, preds)
        B = protocol_B(ml, feats, tgt) if "B" in PROTOCOLS else {}
        result["protocol_A"][fs_name] = A
        if "B" in PROTOCOLS:
            result["protocol_B"][fs_name] = B
        pre = (pre_ml or {}).get("mae", float("nan"))
        for mname, r in A.items():
            print(f"  A  {mname:8s} pre-ML {pre:6.2f}  test MAE {r['test_mae']:6.2f} ± {r['test_mae_sd']:.2f}"
                  f"  ({r['test_mae_pct_range']:.1f}% of range {r['test_range']:.1f})  r2 {r['test_r2']:+.3f}", flush=True)
        for mname, r in B.items():
            print(f"  B  {mname:20s}                 test MAE {r['pooled_oof']['mae']:6.2f}                 r2 {r['pooled_oof']['r2']:+.3f}", flush=True)

    TARGETS_DIR.mkdir(parents=True, exist_ok=True)
    p = pd.concat(preds, ignore_index=True)
    if GEOM:
        p["geom"] = GEOM
    tmp = out_preds.with_name(out_preds.name + ".tmp")
    p.to_parquet(tmp, index=False)
    os.replace(tmp, out_preds)
    tmp = out_json.with_name(out_json.name + ".tmp")          # JSON last: its presence marks a finished target
    tmp.write_text(json.dumps(result, indent=2))
    os.replace(tmp, out_json)
    print(f"\nsaved -> {out_json}")


if __name__ == "__main__":
    main()
