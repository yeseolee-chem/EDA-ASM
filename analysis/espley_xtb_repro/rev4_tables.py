#!/usr/bin/env python3
"""rev4_tables.py — REV4 §3-4 tables from the two per-geometry ML runs (after aggregate_ml.py on g0 and g1).

  python rev4_tables.py [--g0 DIR] [--g1 DIR] [--out-dir DIR]
Defaults: $ESPLEY_OUT/rev4/{g0,g1}/ (ml_report.json, predictions.parquet) -> results_rev4/:
  rev4_headline.csv / .md     target × {G0 upper bound, G1} for the pre-registered headline G1 · ESPLEY73 · KRR_rbf
                              (same arm / model on G0): test MAE ± sd, NMAE, r², G1 − G0
  rev4_appendix_by_arm.csv    KRR_rbf: target × arm (ESPLEY46 / 54 / 73)
  rev4_appendix_by_model.csv  ESPLEY73: target × model (Ridge / KRR_rbf / SVR_rbf / XGB)
  rev4_appendix_full.csv      every target × arm × model
  rev4_pre_ml.csv             pre-ML baselines (xTB counterpart vs DFT label, on the ML rows) per geometry
Metrics: Protocol A test folds, mean over seeds 22/23/14/1/2 (MAE sd over seeds). Columns g0_* = G0 = DFT oracle
geometry (upper bound), g1_* = G1 = GFN2-xTB/ALPB geometry.
Gates (exit 1, nothing written): both reports complete for the rev 4 target set and tagged g0 / g1, both trained on
$ESPLEY_ROWS (sha256), identical test rows and targets per (target, arm, model, seed).
G0 disp in the arms with b_disp (ESPLEY54 / 73): b_disp = the disp label (analytic identity) — marked in g0_note,
G1 − G0 left empty; the headline CSV blanks its g0_* metrics and the .md shows n/a (values kept in the appendices).
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
from train_ml_single import FEATURE_SETS, GRIDS, TARGET_SETS, sha256  # noqa: E402

ROOT = Path(os.environ.get("ESPLEY_OUT", "/gpfs/tmp_cpu2/yeseo1ee/espley_xtb"))
ROWS = Path(os.environ.get("ESPLEY_ROWS") or HERE / "results_rev4" / "rows_rev4.csv")
TGT = TARGET_SETS["rev4"]
ARMS, MODELS = list(FEATURE_SETS), list(GRIDS)
HEAD_ARM, HEAD_MODEL = "ESPLEY73", "KRR_rbf"           # pre-registered headline (REV4 §3-1), same cell on G0
GEOMS = {"g0": "G0 = DFT oracle geometry (upper bound)", "g1": "G1 = GFN2-xTB/ALPB geometry"}
LABEL = {"dft_barrier_kcal": "ΔE‡ (barrier)", "dft_d1_kcal": "d1 (dipole strain)",
         "dft_d2_kcal": "d2 (dipolarophile strain)", "dft_elst_dft": "elst", "dft_pauli_dft": "Pauli",
         "dft_oi_dft": "OI", "dft_disp_dft": "disp", "dft_cpcm_dft": "CPCM", "dft_cds_dft": "CDS"}
DISP = "dft_disp_dft"
DISP_NOTE = "analytic identity (b_disp) — not an ML result"
METRICS = ["test_mae", "test_mae_sd", "test_nmae", "test_r2"]
DELTA = ["test_mae", "test_nmae", "test_r2"]
PRED_KEYS = ["target", "feature_set", "model", "seed", "rxn_id"]


def write_atomic(path, text):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def load(g, d):
    rep = json.loads((d / "ml_report.json").read_text())
    pt = rep.get("per_target", {})
    bad = []
    if rep.get("geom") != g:
        bad.append(f"geom {rep.get('geom')!r} != {g!r}")
    if rep.get("target_set") != "rev4":
        bad.append(f"target_set {rep.get('target_set')!r} != 'rev4'")
    if sorted(pt) != sorted(TGT):
        bad.append(f"targets missing {sorted(set(TGT) - set(pt))}, unexpected {sorted(set(pt) - set(TGT))}")
    gaps = [f"{t}/{a}/{m}" for t in TGT if t in pt for a in ARMS for m in MODELS
            if m not in pt[t].get("protocol_A", {}).get(a, {})]
    if gaps:
        bad.append(f"{len(gaps)} Protocol A cells missing: {gaps[:10]}")
    if bad:
        sys.exit(f"GATE {g} ({d}): " + "; ".join(bad))
    return rep


def cube(rep, g):
    pt = rep["per_target"]
    return pd.DataFrame([dict(target=t, arm=a, model=m, **{f"{g}_{k}": pt[t]["protocol_A"][a][m][k] for k in METRICS})
                         for t in TGT for a in ARMS for m in MODELS])


def fmt(v, spec=".2f"):
    return "—" if v is None or not np.isfinite(v) else format(v, spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g0", type=Path, default=ROOT / "rev4" / "g0")
    ap.add_argument("--g1", type=Path, default=ROOT / "rev4" / "g1")
    ap.add_argument("--out-dir", type=Path, default=HERE / "results_rev4")
    a = ap.parse_args()
    D = {"g0": a.g0, "g1": a.g1}
    R = {g: load(g, d) for g, d in D.items()}

    rows_sha = sha256(ROWS)
    bad = [f"{g} trained on rows sha256 {R[g].get('rows_sha256')}, {ROWS} is {rows_sha}"
           for g in R if R[g].get("rows_sha256") != rows_sha]
    if R["g0"].get("n_rows_ml") != R["g1"].get("n_rows_ml"):
        bad.append(f"n_rows_ml g0 {R['g0'].get('n_rows_ml')} != g1 {R['g1'].get('n_rows_ml')}")
    P = {g: pd.read_parquet(d / "predictions.parquet", columns=PRED_KEYS + ["y"]).sort_values(PRED_KEYS)
         .reset_index(drop=True) for g, d in D.items()}
    same = (len(P["g0"]) == len(P["g1"]) and P["g0"][PRED_KEYS].equals(P["g1"][PRED_KEYS])
            and np.allclose(P["g0"]["y"], P["g1"]["y"], rtol=0, atol=1e-9))
    if not same:
        bad.append("test rows or targets per (target, arm, model, seed) differ between G0 and G1 predictions")
    if bad:
        sys.exit("GATE: " + "; ".join(bad))
    n = int(R["g1"]["n_rows_ml"])

    full = cube(R["g0"], "g0").merge(cube(R["g1"], "g1"), on=["target", "arm", "model"], validate="one_to_one")
    full.insert(1, "label", full.target.map(LABEL))
    full.insert(4, "n_rows", n)
    dcols = [f"g1_minus_g0_{k[len('test_'):]}" for k in DELTA]
    for k, c in zip(DELTA, dcols):
        full[c] = full[f"g1_{k}"] - full[f"g0_{k}"]
    disp = (full.target == DISP) & full.arm.map(lambda arm: "b_disp" in FEATURE_SETS[arm])   # ESPLEY46: no b_disp
    full.loc[disp, dcols] = np.nan
    full["g0_note"] = np.where(disp, DISP_NOTE, "")
    head = full[(full.arm == HEAD_ARM) & (full.model == HEAD_MODEL)].copy()
    head.loc[head.g0_note != "", [f"g0_{k}" for k in METRICS]] = np.nan   # REV4 §3-1: not reported as an ML result

    pre = []
    for t in TGT:
        p = {g: R[g]["per_target"][t].get("pre_ml") or {} for g in R}
        pre.append(dict(target=t, label=LABEL[t], feature=p["g1"].get("feature") or p["g0"].get("feature"), n_rows=n,
                        **{f"{g}_pre_ml_{k}": p[g].get(k, np.nan) for g in R for k in ("mae", "bias", "pearson_r")},
                        g0_note="b_disp = the disp label on G0 (analytic identity)" if t == DISP else ""))
    pre = pd.DataFrame(pre)

    md = [f"# rev 4 headline — G1 · {HEAD_ARM} · {HEAD_MODEL} (pre-registered)", "",
          "Protocol A test folds, mean over seeds 22/23/14/1/2 (80/10/10, nested per-seed GridSearchCV 5-fold): "
          "MAE ± sd over seeds (kcal/mol), NMAE = MAE / mean absolute deviation of the test targets, r². "
          f"n = {n} rxns (`rows_rev4.csv`, sha256 `{rows_sha[:12]}`), identical rows and test splits on both geometries.",
          "",
          f"- **{GEOMS['g0']}**: features on the DFT TS and references the labels were computed on; not deployable, "
          "never the headline.",
          f"- **{GEOMS['g1']}**: TS and references re-optimised with GFN2-xTB/ALPB(water) from the DFT structures "
          "(conformer and stereo choice retained); the Espley-comparable arm, not a deployment result.",
          "",
          "| Target | G0 upper bound MAE | G0 NMAE | G0 r² | G1 MAE | G1 NMAE | G1 r² | G1 − G0 MAE |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in head.itertuples():
        g0 = (["n/a †"] * 3 if r.g0_note else
              [f"{fmt(r.g0_test_mae)} ± {fmt(r.g0_test_mae_sd)}", fmt(r.g0_test_nmae, ".3f"), fmt(r.g0_test_r2, ".3f")])
        g1 = [f"{fmt(r.g1_test_mae)} ± {fmt(r.g1_test_mae_sd)}", fmt(r.g1_test_nmae, ".3f"), fmt(r.g1_test_r2, ".3f")]
        md.append("| " + " | ".join([r.label, *g0, *g1, fmt(r.g1_minus_g0_mae, "+.2f")]) + " |")
    md += ["", f"† G0 disp: {DISP_NOTE}. On the DFT geometry the b_disp feature equals the disp label; the G0 numbers "
           "are blank in `rev4_headline.csv` and kept, flagged in `g0_note`, in the appendix CSVs for traceability "
           "only. G1 disp, and G0 disp in ESPLEY46 (no b_disp), are ordinary predictions.", "",
           f"Appendix: `rev4_appendix_by_arm.csv` ({HEAD_MODEL} × {' / '.join(ARMS)}), `rev4_appendix_by_model.csv` "
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
