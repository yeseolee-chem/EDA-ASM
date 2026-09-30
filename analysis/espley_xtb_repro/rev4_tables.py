#!/usr/bin/env python3
"""rev4_tables.py — REV4 §3-4 tables from the G1 ML run (after aggregate_ml.py on g1).

  python rev4_tables.py [--g1 DIR] [--out-dir DIR]
Defaults: $ESPLEY_OUT/rev4/g1/ (ml_report.json, predictions.parquet) -> results_rev4/:
  rev4_headline.csv / .md     target row of the pre-registered headline G1 · ESPLEY73 · KRR_rbf: test MAE ± sd, NMAE, r²
  rev4_appendix_by_arm.csv    KRR_rbf: target × arm (ESPLEY46 / 54 / 73)
  rev4_appendix_by_model.csv  ESPLEY73: target × model (Ridge / KRR_rbf / SVR_rbf / XGB)
  rev4_appendix_full.csv      every target × arm × model
  rev4_pre_ml.csv             pre-ML baselines (xTB counterpart vs DFT label, on the ML rows)
Metrics: Protocol A test folds, mean over seeds 22/23/14/1/2 (MAE sd over seeds). Columns g1_* = G1 = GFN2-xTB/ALPB
geometry (the only geometry).
Gates (exit 1, nothing written): the report is complete for the rev 4 target set and tagged g1, trained on
$ESPLEY_ROWS (sha256, n); the test rows of every (target, arm, model, seed) are the split_80_10_10 test rows of
$ESPLEY_ROWS in file order.
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
from train_ml_single import FEATURE_SETS, GRIDS, SEEDS, TARGET_SETS, sha256, split_80_10_10  # noqa: E402

ROOT = Path(os.environ.get("ESPLEY_OUT", "/gpfs/tmp_cpu2/yeseo1ee/espley_xtb"))
ROWS = Path(os.environ.get("ESPLEY_ROWS") or HERE / "results_rev4" / "rows_rev4.csv")
TGT = TARGET_SETS["rev4"]
ARMS, MODELS = list(FEATURE_SETS), list(GRIDS)
HEAD_ARM, HEAD_MODEL = "ESPLEY73", "KRR_rbf"           # pre-registered headline (REV4 §3-1)
GEOM, GEOM_NOTE = "g1", "G1 = GFN2-xTB/ALPB geometry"
LABEL = {"dft_barrier_kcal": "ΔE‡ (barrier)", "dft_d1_kcal": "d1 (dipole strain)",
         "dft_d2_kcal": "d2 (dipolarophile strain)", "dft_elst_dft": "elst", "dft_pauli_dft": "Pauli",
         "dft_oi_dft": "OI", "dft_disp_dft": "disp", "dft_cpcm_dft": "CPCM", "dft_cds_dft": "CDS"}
METRICS = ["test_mae", "test_mae_sd", "test_nmae", "test_r2"]
PRED_KEYS = ["target", "feature_set", "model", "seed"]


def write_atomic(path, text):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def load(d):
    rep = json.loads((d / "ml_report.json").read_text())
    pt = rep.get("per_target", {})
    bad = []
    if rep.get("geom") != GEOM:
        bad.append(f"geom {rep.get('geom')!r} != {GEOM!r}")
    if rep.get("target_set") != "rev4":
        bad.append(f"target_set {rep.get('target_set')!r} != 'rev4'")
    if sorted(pt) != sorted(TGT):
        bad.append(f"targets missing {sorted(set(TGT) - set(pt))}, unexpected {sorted(set(pt) - set(TGT))}")
    gaps = [f"{t}/{a}/{m}" for t in TGT if t in pt for a in ARMS for m in MODELS
            if m not in pt[t].get("protocol_A", {}).get(a, {})]
    if gaps:
        bad.append(f"{len(gaps)} Protocol A cells missing: {gaps[:10]}")
    if bad:
        sys.exit(f"GATE {GEOM} ({d}): " + "; ".join(bad))
    return rep


def check_test_rows(d, rows):
    """Every (target, arm, model, seed) was scored on the split_80_10_10 test rows of `rows` (file order)."""
    P = pd.read_parquet(d / "predictions.parquet", columns=PRED_KEYS + ["rxn_id"])
    want = {s: np.sort(rows[split_80_10_10(len(rows), s)[1]]) for s in SEEDS}
    fs = {len(FEATURE_SETS[a]) for a in ARMS}
    bad, n_groups = [], 0
    for (t, f, m, s), sub in P.groupby(PRED_KEYS):
        if t not in TGT or f not in fs or m not in MODELS:
            continue
        n_groups += 1
        if s not in want or not np.array_equal(np.sort(sub["rxn_id"].to_numpy(int)), want[s]):
            bad.append((t, f, m, s))
    if n_groups != len(TGT) * len(ARMS) * len(MODELS) * len(SEEDS):
        bad.append(f"{n_groups} (target, arm, model, seed) prediction groups, expected "
                   f"{len(TGT) * len(ARMS) * len(MODELS) * len(SEEDS)}")
    return bad


def cube(rep):
    pt = rep["per_target"]
    return pd.DataFrame([dict(target=t, arm=a, model=m, **{f"g1_{k}": pt[t]["protocol_A"][a][m][k] for k in METRICS})
                         for t in TGT for a in ARMS for m in MODELS])


def fmt(v, spec=".2f"):
    return "—" if v is None or not np.isfinite(v) else format(v, spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g1", type=Path, default=ROOT / "rev4" / "g1")
    ap.add_argument("--out-dir", type=Path, default=HERE / "results_rev4")
    a = ap.parse_args()
    R = load(a.g1)

    rows_sha = sha256(ROWS)
    rows = pd.read_csv(ROWS)["rxn_id"].astype(int).to_numpy()
    bad = []
    if R.get("rows_sha256") != rows_sha:
        bad.append(f"trained on rows sha256 {R.get('rows_sha256')}, {ROWS} is {rows_sha}")
    if R.get("n_rows_ml") != len(rows):
        bad.append(f"n_rows_ml {R.get('n_rows_ml')} != {len(rows)} rows in {ROWS}")
    if not bad:
        off = check_test_rows(a.g1, rows)
        if off:
            bad.append(f"test rows differ from split_80_10_10 of {ROWS.name} in {len(off)} groups: {off[:5]}")
    if bad:
        sys.exit("GATE: " + "; ".join(bad))
    n = len(rows)

    full = cube(R)
    full.insert(1, "label", full.target.map(LABEL))
    full.insert(4, "n_rows", n)
    head = full[(full.arm == HEAD_ARM) & (full.model == HEAD_MODEL)].copy()

    pre = []
    for t in TGT:
        p = R["per_target"][t].get("pre_ml") or {}
        pre.append(dict(target=t, label=LABEL[t], feature=p.get("feature"), n_rows=n,
                        **{f"g1_pre_ml_{k}": p.get(k, np.nan) for k in ("mae", "bias", "pearson_r")}))
    pre = pd.DataFrame(pre)

    md = [f"# rev 4 headline — G1 · {HEAD_ARM} · {HEAD_MODEL} (pre-registered)", "",
          "Protocol A test folds, mean over seeds 22/23/14/1/2 (80/10/10, nested per-seed GridSearchCV 5-fold): "
          "MAE ± sd over seeds (kcal/mol), NMAE = MAE / mean absolute deviation of the test targets, r². "
          f"n = {n} rxns (`rows_rev4.csv`, sha256 `{rows_sha[:12]}`).",
          "",
          f"- **{GEOM_NOTE}**: TS and references re-optimised with GFN2-xTB/ALPB(water) from the DFT structures "
          "(conformer and stereo choice retained); the Espley-comparable arm, not a deployment result.",
          "",
          "| Target | MAE | NMAE | r² |",
          "|---|---:|---:|---:|"]
    for r in head.itertuples():
        md.append("| " + " | ".join([r.label, f"{fmt(r.g1_test_mae)} ± {fmt(r.g1_test_mae_sd)}",
                                      fmt(r.g1_test_nmae, ".3f"), fmt(r.g1_test_r2, ".3f")]) + " |")
    md += ["", f"Appendix: `rev4_appendix_by_arm.csv` ({HEAD_MODEL} × {' / '.join(ARMS)}), `rev4_appendix_by_model.csv` "
           f"({HEAD_ARM} × {' / '.join(MODELS)}), `rev4_appendix_full.csv` (all cells), pre-ML baselines "
           "`rev4_pre_ml.csv`.", ""]

    a.out_dir.mkdir(parents=True, exist_ok=True)
    outs = {"rev4_headline.csv": head, "rev4_appendix_by_arm.csv": full[full.model == HEAD_MODEL],
            "rev4_appendix_by_model.csv": full[full.arm == HEAD_ARM], "rev4_appendix_full.csv": full,
            "rev4_pre_ml.csv": pre}
    for name, df in outs.items():
        write_atomic(a.out_dir / name, df.to_csv(index=False))
    write_atomic(a.out_dir / "rev4_headline.md", "\n".join(md))
    print("\n".join(md))
    print(pre.round(3).to_string(index=False))
    print(f"-> {a.out_dir}: {', '.join(outs)}, rev4_headline.md")


if __name__ == "__main__":
    main()
