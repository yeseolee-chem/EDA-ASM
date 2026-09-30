#!/usr/bin/env python3
"""phase2_report.py — REV4 Phase 2-4 report: xtb feature shift from G0 (DFT geometry) to G1 (xTB geometry).

  python phase2_report.py [--g0 PARQUET] [--g1 PARQUET] [--out-dir DIR]
Defaults: $ESPLEY_OUT/xtb_features_{g0,g1}.parquet -> results_rev4/phase2_feature_shift.csv + phase2_report.json.

Per geometry: rows, ok count, xtb_status value counts, ok rows with NaN in an ESPLEY73 column.
Per ESPLEY73 feature, on the rxns ok in both: median G0, median G1, median of the per-rxn G1 − G0, MAE G1 vs G0
(+ SD of G0 and MAE / SD as a scale-free shift, Pearson r G0–G1).
MAE(b_disp − dft_disp_dft): ~0 on G0 (analytic identity with the DFT disp label), not 0 on G1.
Rows: the make_rows_rev4.py rule (G1 ok ∩ G0 ok ∩ no NaN in ESPLEY73 + rev 4 targets in either parquet ∩ hygiene
d1 >= 0, d2 >= 0, d2 <= 50) and, if present, whether $ESPLEY_ROWS equals it; plus the looser G1 ok ∩ hygiene count.
Outputs are rewritten atomically on every run (cheap and deterministic).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train_ml_single import FEATURE_SETS, TARGET_SETS, hygiene  # noqa: E402

ROOT = Path(os.environ.get("ESPLEY_OUT", "/gpfs/tmp_cpu2/yeseo1ee/espley_xtb"))
ROWS = Path(os.environ.get("ESPLEY_ROWS", HERE / "results_rev4" / "rows_rev4.csv"))
F73 = FEATURE_SETS["ESPLEY73"]
TGT = TARGET_SETS["rev4"]


def arm_of(c):
    return next(k for k in ("ESPLEY46", "ESPLEY54", "ESPLEY73") if c in FEATURE_SETS[k])


def rows_rule(ok):
    """make_rows_rev4.py: ok in both ∩ no NaN in ESPLEY73 + rev 4 targets in either parquet ∩ hygiene (G1 values)."""
    ix = {k: df.assign(rxn_id=df.rxn_id.astype(int)).set_index("rxn_id") for k, df in ok.items()}
    both = sorted(set(ix["g0"].index) & set(ix["g1"].index))
    clean = ix["g0"].loc[both, F73 + TGT].notna().all(axis=1) & ix["g1"].loc[both, F73 + TGT].notna().all(axis=1)
    sub = ix["g1"].loc[clean[clean].index]
    return sorted(int(r) for r in sub.index[hygiene(sub).to_numpy()])


def disp_mae(df):
    d = (df["b_disp"] - df["dft_disp_dft"]).dropna()
    return dict(n=int(len(d)), mae=float(d.abs().mean()) if len(d) else None, bias=float(d.mean()) if len(d) else None)


def write_atomic(path, text):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g0", type=Path, default=ROOT / "xtb_features_g0.parquet")
    ap.add_argument("--g1", type=Path, default=ROOT / "xtb_features_g1.parquet")
    ap.add_argument("--out-dir", type=Path, default=HERE / "results_rev4")
    a = ap.parse_args()
    G = {"g0": pd.read_parquet(a.g0), "g1": pd.read_parquet(a.g1)}
    ok = {k: df[df.xtb_status == "ok"] for k, df in G.items()}
    rep = dict(g0=str(a.g0), g1=str(a.g1), n_rows={k: len(df) for k, df in G.items()},
               n_ok={k: len(df) for k, df in ok.items()},
               xtb_status={k: {str(s): int(n) for s, n in df.xtb_status.value_counts().items()} for k, df in G.items()},
               n_ok_with_nan_espley73={k: int(df[F73].isna().any(axis=1).sum()) for k, df in ok.items()})

    both = ok["g0"][["rxn_id"] + F73].merge(ok["g1"][["rxn_id"] + F73], on="rxn_id", suffixes=("_g0", "_g1"))
    g1_st = G["g1"].set_index("rxn_id").xtb_status.reindex(ok["g0"].rxn_id)
    rep.update(n_ok_both=len(both),
               g1_status_among_g0_ok={str(s): int(n) for s, n in g1_st.fillna("absent").value_counts().items()})
    rows = []
    for c in F73:
        x, y = both[c + "_g0"].to_numpy(dtype=float), both[c + "_g1"].to_numpy(dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        sd = float(np.std(x, ddof=1)) if len(x) > 1 else float("nan")
        mae = float(np.mean(np.abs(y - x))) if len(x) else float("nan")
        r = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 and np.std(x) > 0 and np.std(y) > 0 else float("nan")
        rows.append(dict(feature=c, arm=arm_of(c), n_both=int(len(x)),
                         median_g0=float(np.median(x)) if len(x) else float("nan"),
                         median_g1=float(np.median(y)) if len(x) else float("nan"),
                         median_diff_g1_minus_g0=float(np.median(y - x)) if len(x) else float("nan"),
                         mae_g1_vs_g0=mae, sd_g0=sd, mae_over_sd_g0=mae / sd if sd > 0 else float("nan"),
                         pearson_r_g0_g1=r))
    shift = pd.DataFrame(rows)

    okid = set(ok["g0"].rxn_id) & set(ok["g1"].rxn_id)
    rep["mae_b_disp_minus_dft_disp"] = {"g0_ok": disp_mae(ok["g0"]), "g1_ok": disp_mae(ok["g1"]),
                                        "g0_ok_both": disp_mae(ok["g0"][ok["g0"].rxn_id.isin(okid)]),
                                        "g1_ok_both": disp_mae(ok["g1"][ok["g1"].rxn_id.isin(okid)])}
    sel = ok["g1"][hygiene(ok["g1"])]
    rep["n_g1_ok_hygiene"] = len(sel)                                   # looser than the rows rule (G1 only)
    rep["n_g1_ok_hygiene_nan_espley73"] = int(sel[F73].isna().any(axis=1).sum())
    rule = rows_rule(ok)
    rep["n_rows_rule"] = len(rule)
    if ROWS.is_file():
        pre = pd.read_csv(ROWS).rxn_id.astype(int).tolist()
        rep["rows_csv"] = dict(path=str(ROWS), n=len(pre), equals_rows_rule=pre == rule)
    rep["largest_shift_mae_over_sd_g0"] = (shift.sort_values("mae_over_sd_g0", ascending=False).head(10)
                                           [["feature", "mae_over_sd_g0", "median_diff_g1_minus_g0"]].to_dict("records"))

    a.out_dir.mkdir(parents=True, exist_ok=True)
    write_atomic(a.out_dir / "phase2_feature_shift.csv", shift.to_csv(index=False))
    native = lambda x: x.item() if hasattr(x, "item") else str(x)  # noqa: E731
    write_atomic(a.out_dir / "phase2_report.json", json.dumps(rep, indent=1, default=native))
    print(json.dumps({k: v for k, v in rep.items() if k != "largest_shift_mae_over_sd_g0"}, indent=1, default=native))
    print(shift.round(4).to_string(index=False))
    print(f"-> {a.out_dir}/phase2_feature_shift.csv, phase2_report.json")


if __name__ == "__main__":
    main()
