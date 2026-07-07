"""Aggregate A/B/C OOF predictions → abc_metrics.csv + REPORT.md.

- Reads arm A parquet (single file, wide already) and arm B/C fold JSONs
  (results/cells/{B,C}/fold{F}.json) → assembles per-arm OOF parquets.
- Computes NMAE / RMSE / R² / slope per channel + barrier + bootstrap 95% CI.
- Computes the arm-diff ΔNMAE(B − C) 95% CI (paired bootstrap) + Wilcoxon.
- Emits:
    results/oof_pred_{A,B,C}.parquet
    results/abc_metrics.csv
    results/REPORT.md
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
CH = ["strain", "Pauli", "V_elst", "oi", "disp"]
SEED = 42
B_BOOT = 1000


# ============================================================
# Metric helpers (pooled OOF)
# ============================================================


def mad_per_channel(y: np.ndarray) -> np.ndarray:
    """Mean absolute deviation from per-channel mean (fixed on full pool)."""
    return np.mean(np.abs(y - y.mean(axis=0)), axis=0)


def nmae(yt: np.ndarray, yp: np.ndarray, mad: np.ndarray) -> np.ndarray:
    return np.abs(yp - yt).mean(axis=0) / np.where(mad < 1e-9, np.nan, mad)


def rmse(yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((yp - yt) ** 2, axis=0))


def r2(yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
    ss = np.sum((yt - yt.mean(axis=0)) ** 2, axis=0)
    return 1.0 - np.sum((yp - yt) ** 2, axis=0) / np.where(ss < 1e-9, np.nan, ss)


def slope(yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
    x = yt - yt.mean(axis=0); y = yp - yp.mean(axis=0)
    d = np.sum(x * x, axis=0)
    return np.sum(x * y, axis=0) / np.where(d < 1e-9, np.nan, d)


def cancellation_ratio(y: np.ndarray) -> float:
    """SPEC §5: ρ = |Σ_c ê_c| / Σ_c |ê_c| where ê_c = y_c - mean(y_c)."""
    e = y - y.mean(axis=0)
    return float(np.mean(np.abs(e.sum(axis=1)) / np.maximum(np.abs(e).sum(axis=1), 1e-9)))


# ============================================================
# Bootstrap (reaction-level, seeded)
# ============================================================


def bootstrap_ci(fn, yt: np.ndarray, yp: np.ndarray, mad: np.ndarray | None = None,
                 B: int = B_BOOT, seed: int = SEED, alpha: float = 0.05):
    """Return (point, ci_low, ci_high) arrays over channels.

    fn(yt, yp[, mad]) → per-channel ndarray."""
    rng = np.random.default_rng(seed)
    n = yt.shape[0]
    if mad is not None:
        point = fn(yt, yp, mad)
    else:
        point = fn(yt, yp)
    boots = np.zeros((B, point.shape[0] if np.ndim(point) else 1))
    for i in range(B):
        idx = rng.integers(0, n, size=n)
        if mad is not None:
            v = fn(yt[idx], yp[idx], mad)   # mad kept fixed on full pool
        else:
            v = fn(yt[idx], yp[idx])
        boots[i] = v
    lo = np.nanquantile(boots, alpha / 2, axis=0)
    hi = np.nanquantile(boots, 1 - alpha / 2, axis=0)
    return point, lo, hi


def bootstrap_scalar_ci(fn_scalar, yt: np.ndarray, yp: np.ndarray, mad_scalar=None,
                        B: int = B_BOOT, seed: int = SEED, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    n = yt.shape[0]
    if mad_scalar is not None:
        point = fn_scalar(yt, yp, mad_scalar)
    else:
        point = fn_scalar(yt, yp)
    boots = np.zeros(B)
    for i in range(B):
        idx = rng.integers(0, n, size=n)
        if mad_scalar is not None:
            boots[i] = fn_scalar(yt[idx], yp[idx], mad_scalar)
        else:
            boots[i] = fn_scalar(yt[idx], yp[idx])
    return point, float(np.nanquantile(boots, alpha / 2)), float(np.nanquantile(boots, 1 - alpha / 2))


# ============================================================
# I/O — assemble per-arm OOF parquet from per-fold JSONs
# ============================================================


def assemble_arm(arm: str) -> pd.DataFrame:
    cells = sorted((RESULTS / "cells" / arm).glob("fold*.json"))
    rows = []
    for c in cells:
        d = json.load(open(c))
        yt = np.array(d["y_true_test"])
        yp = np.array(d["y_pred_test"])
        for rid, y, p in zip(d["test_rids"], yt, yp):
            rows.append({
                "reaction_id": rid, "fold": d["fold"],
                **{f"y_{CH[i]}": float(y[i]) for i in range(5)},
                **{f"yhat_{CH[i]}": float(p[i]) for i in range(5)},
            })
    df = pd.DataFrame(rows).sort_values("reaction_id").reset_index(drop=True)
    df.to_parquet(RESULTS / f"oof_pred_{arm}.parquet", index=False)
    return df


def load_arm(arm: str) -> pd.DataFrame:
    p = RESULTS / f"oof_pred_{arm}.parquet"
    return pd.read_parquet(p)


# ============================================================
# Main
# ============================================================


def per_arm_metrics(arm: str, df: pd.DataFrame, mad_global: np.ndarray) -> list[dict]:
    y = df[[f"y_{c}" for c in CH]].values.astype(np.float64)
    p = df[[f"yhat_{c}" for c in CH]].values.astype(np.float64)
    rows = []

    # Per-channel bootstrap
    for c, ch in enumerate(CH):
        yt = y[:, c:c+1]; yp = p[:, c:c+1]
        # NMAE / RMSE / R² / slope with global mad
        (nm, nm_lo, nm_hi) = bootstrap_ci(
            lambda a, b, m: nmae(a, b, m[c:c+1]), yt, yp, mad=mad_global)
        (rm, rm_lo, rm_hi) = bootstrap_ci(lambda a, b: rmse(a, b), yt, yp)
        (r_, r_lo, r_hi) = bootstrap_ci(lambda a, b: r2(a, b), yt, yp)
        (sl, sl_lo, sl_hi) = bootstrap_ci(lambda a, b: slope(a, b), yt, yp)
        rows.append({"arm": arm, "channel": ch,
                     "NMAE": float(nm[0]), "NMAE_lo": float(nm_lo[0]), "NMAE_hi": float(nm_hi[0]),
                     "RMSE": float(rm[0]), "RMSE_lo": float(rm_lo[0]), "RMSE_hi": float(rm_hi[0]),
                     "R2": float(r_[0]), "R2_lo": float(r_lo[0]), "R2_hi": float(r_hi[0]),
                     "slope": float(sl[0]), "slope_lo": float(sl_lo[0]), "slope_hi": float(sl_hi[0])})

    # Barrier (sum of channels)
    bt = y.sum(axis=1); bp = p.sum(axis=1)
    mad_b = float(np.mean(np.abs(bt - bt.mean())))
    (nm, nm_lo, nm_hi) = bootstrap_scalar_ci(
        lambda a, b, m: float(np.abs(b - a).mean() / max(m, 1e-9)),
        bt, bp, mad_scalar=mad_b)
    (rm, rm_lo, rm_hi) = bootstrap_scalar_ci(
        lambda a, b: float(np.sqrt(np.mean((b - a) ** 2))), bt, bp)
    (r_, r_lo, r_hi) = bootstrap_scalar_ci(
        lambda a, b: 1.0 - float(np.sum((b - a) ** 2) / max(np.sum((a - a.mean()) ** 2), 1e-9)),
        bt, bp)
    (sl, sl_lo, sl_hi) = bootstrap_scalar_ci(
        lambda a, b: float(np.sum((a - a.mean()) * (b - b.mean())) /
                            max(np.sum((a - a.mean()) ** 2), 1e-9)),
        bt, bp)
    rows.append({"arm": arm, "channel": "barrier",
                 "NMAE": nm, "NMAE_lo": nm_lo, "NMAE_hi": nm_hi,
                 "RMSE": rm, "RMSE_lo": rm_lo, "RMSE_hi": rm_hi,
                 "R2": r_, "R2_lo": r_lo, "R2_hi": r_hi,
                 "slope": sl, "slope_lo": sl_lo, "slope_hi": sl_hi})
    return rows


def paired_bc_delta_nmae(dfB: pd.DataFrame, dfC: pd.DataFrame,
                          mad_global: np.ndarray) -> list[dict]:
    """Paired ΔNMAE(B − C) 95% CI + Wilcoxon per channel + barrier."""
    dfBs = dfB.set_index("reaction_id")
    dfCs = dfC.set_index("reaction_id")
    common = dfBs.index.intersection(dfCs.index)
    assert len(common) == len(dfBs) == len(dfCs), "arm B/C OOF pool mismatch"
    dfBs = dfBs.loc[common]; dfCs = dfCs.loc[common]

    yB = dfBs[[f"yhat_{c}" for c in CH]].values.astype(np.float64)
    yC = dfCs[[f"yhat_{c}" for c in CH]].values.astype(np.float64)
    ytrue = dfBs[[f"y_{c}" for c in CH]].values.astype(np.float64)

    rows = []
    rng = np.random.default_rng(SEED)
    n = ytrue.shape[0]
    for c, ch in enumerate(CH):
        aeB = np.abs(yB[:, c] - ytrue[:, c])
        aeC = np.abs(yC[:, c] - ytrue[:, c])
        # ΔNMAE(B - C) = (mean(aeB) - mean(aeC)) / MAD_c
        mad_c = mad_global[c]
        point = (aeB.mean() - aeC.mean()) / mad_c
        boots = np.empty(B_BOOT)
        for i in range(B_BOOT):
            idx = rng.integers(0, n, size=n)
            boots[i] = (aeB[idx].mean() - aeC[idx].mean()) / mad_c
        lo, hi = float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))
        # Wilcoxon on |errB| - |errC|
        try:
            w = wilcoxon(aeB, aeC, alternative="two-sided")
            wstat = float(w.statistic); wp = float(w.pvalue)
        except ValueError:
            wstat, wp = np.nan, np.nan
        rows.append({"channel": ch, "delta_NMAE_B_minus_C": float(point),
                     "ci_low": lo, "ci_high": hi,
                     "wilcoxon_stat": wstat, "wilcoxon_p": wp})

    # barrier
    bB = yB.sum(axis=1); bC = yC.sum(axis=1); bt = ytrue.sum(axis=1)
    aeB = np.abs(bB - bt); aeC = np.abs(bC - bt)
    mad_b = float(np.mean(np.abs(bt - bt.mean())))
    point = (aeB.mean() - aeC.mean()) / mad_b
    boots = np.empty(B_BOOT)
    for i in range(B_BOOT):
        idx = rng.integers(0, n, size=n)
        boots[i] = (aeB[idx].mean() - aeC[idx].mean()) / mad_b
    lo, hi = float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))
    try:
        w = wilcoxon(aeB, aeC, alternative="two-sided")
        wstat, wp = float(w.statistic), float(w.pvalue)
    except ValueError:
        wstat, wp = np.nan, np.nan
    rows.append({"channel": "barrier", "delta_NMAE_B_minus_C": float(point),
                 "ci_low": lo, "ci_high": hi,
                 "wilcoxon_stat": wstat, "wilcoxon_p": wp})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-arm", default=None, choices=["A", "B", "C"],
                    help="Assemble a single arm parquet then exit (no CSV/REPORT).")
    args = ap.parse_args()

    if args.only_arm in ("B", "C"):
        assemble_arm(args.only_arm)
        return

    # Arm A already emits its own parquet; but make sure B and C are assembled.
    if (RESULTS / "cells" / "B").exists():
        assemble_arm("B")
    if (RESULTS / "cells" / "C").exists():
        assemble_arm("C")

    dfs = {}
    for arm in ("A", "B", "C"):
        p = RESULTS / f"oof_pred_{arm}.parquet"
        if not p.exists():
            print(f"[warn] {p} missing — skipping arm {arm}")
            continue
        dfs[arm] = load_arm(arm)

    if not dfs:
        raise SystemExit("no arm parquets found")

    # Global MAD_c fixed on 787 (any arm's y-columns work; take from first).
    df0 = next(iter(dfs.values()))
    y_all = df0[[f"y_{c}" for c in CH]].values.astype(np.float64)
    mad_global = mad_per_channel(y_all)

    print(f"pooled N = {len(df0)}; mad per channel = {mad_global.round(3).tolist()}", flush=True)
    print(f"cancellation ratio ρ (labels) = {cancellation_ratio(y_all):.3f}", flush=True)

    all_rows = []
    for arm, df in dfs.items():
        print(f"metrics: arm {arm}  N={len(df)}", flush=True)
        all_rows.extend(per_arm_metrics(arm, df, mad_global))
    metrics_df = pd.DataFrame(all_rows)
    metrics_df.to_csv(RESULTS / "abc_metrics.csv", index=False)
    print(f"wrote → {RESULTS / 'abc_metrics.csv'}", flush=True)

    # BC delta
    if "B" in dfs and "C" in dfs:
        bc = pd.DataFrame(paired_bc_delta_nmae(dfs["B"], dfs["C"], mad_global))
        bc.to_csv(RESULTS / "delta_BC.csv", index=False)
        print("BC ΔNMAE(B−C) per channel:")
        for _, row in bc.iterrows():
            print(f"  {row['channel']}: Δ={row['delta_NMAE_B_minus_C']:+.3f} "
                  f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]  "
                  f"Wilcoxon p={row['wilcoxon_p']:.3g}")


if __name__ == "__main__":
    main()
