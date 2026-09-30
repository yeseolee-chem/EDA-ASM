#!/usr/bin/env python3
"""aggregate_ml.py — merge per-target JSON/parquet into unified report + table + predictions.

  [ESPLEY_ML_OUT=<dir>] [ESPLEY_GEOM=g1|g2] [ESPLEY_REPORT_PREFIX=<path prefix>] python aggregate_ml.py
<ML_OUT>/ml_targets/ -> <ML_OUT>/ml_report.json, ml_table_espley.csv, predictions.parquet (ML_OUT default $ESPLEY_OUT).
Any target set (rev 3: 12 targets, rev 4 / 5: 9); Protocol B rows only where it was run. Table rows carry `arm`
(name) and `n_features` (rev 5 JSONs); `feature_set` keeps the arm name as in rev 4.
Gates (exit != 0, nothing written): a preds parquet for every JSON, no duplicate target, the whole recorded target
set present (rev 4 JSONs carry targets_in_set), one geom (= ESPLEY_GEOM if set), protocols, row set, feature file
(path + sha256), arms and their columns, block file and pre-registration; the same arms x models in every target;
no duplicate (target, arm, model, seed, rxn_id) prediction.
ESPLEY_REPORT_PREFIX (rev 5 D-2, e.g. <code>/results_rev5/D2_protocolA_) also writes
  <prefix>table.csv      every Protocol A cell (target x arm x model), `note` on EXT_SEL
  <prefix>headline.csv   KRR_rbf x arms: MAE, sd, NMAE, r², difference to ESPLEY73, `note` on EXT_SEL
  <prefix>headline.md    the same as tables, with the selection-bias note on EXT_SEL (spec D-2)
  <prefix>meta.json      inputs (rows / feature / block files + sha256, arms, EXT_SEL blocks, pre-registration)
  (title line: ESPLEY_REPORT_TITLE, default "Protocol A").
Outputs are rewritten atomically on every run (cheap and deterministic).
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev5_common as R5  # noqa: E402

ROOT = Path(os.environ.get("ESPLEY_OUT", "/gpfs/tmp_cpu2/yeseo1ee/espley_xtb"))
ML_OUT = Path(os.environ.get("ESPLEY_ML_OUT") or ROOT)
GEOM = os.environ.get("ESPLEY_GEOM") or None
PREFIX = os.environ.get("ESPLEY_REPORT_PREFIX") or None
TITLE = os.environ.get("ESPLEY_REPORT_TITLE") or "Protocol A"
TARGETS_DIR = ML_OUT / "ml_targets"
OUT_REPORT = ML_OUT / "ml_report.json"
OUT_TABLE = ML_OUT / "ml_table_espley.csv"
OUT_PREDS = ML_OUT / "predictions.parquet"
SHARED = ("geom", "target_set", "protocols", "features_file", "features_sha256", "rows_sha256",
          "n_rows_ml", "ext_features_file", "ext_features_sha256", "arms", "arm_columns", "arm_n_features",
          "ext_sel_blocks", "prereg_rev5b_sha256", "arm_same_columns_as")  # identical in every per-target JSON
HEAD_MODEL = "KRR_rbf"
TARGET_LABEL = {"dft_barrier_kcal": "ΔE‡ (barrier)", "dft_d1_kcal": "d1 (dipole strain)",
                "dft_d2_kcal": "d2 (dipolarophile strain)", "dft_elst_dft": "elst", "dft_pauli_dft": "Pauli",
                "dft_oi_dft": "OI", "dft_disp_dft": "disp", "dft_cpcm_dft": "CPCM", "dft_cds_dft": "CDS"}
SEL_NOTE = ("EXT_SEL: its blocks were selected in Phase C by dev-CV on the dev rows, which overlap these Protocol A "
            "test rows, so these numbers carry selection bias; the unbiased estimate is D-1 (lockbox).")
PRED_KEYS = ["target", "arm", "model", "seed", "rxn_id"]
HEAD_COLS = ["target", "label", "arm", "n_features", "model", "test_mae", "test_mae_sd", "test_nmae", "test_r2",
             "train_mae", "train_nmae"]


def gates(per_target, preds_paths):
    bad = []
    tg = [t["target"] for t in per_target]
    dup = sorted({x for x in tg if tg.count(x) > 1})
    if dup:
        bad.append(f"duplicate targets {dup}")
    for key in SHARED:
        vals = sorted({json.dumps(t.get(key), sort_keys=True, default=str) for t in per_target})
        if len(vals) > 1:
            bad.append(f"per-target JSONs disagree on {key}: {[v[:120] for v in vals]}")
    if GEOM and per_target[0].get("geom") != GEOM:
        bad.append(f"ESPLEY_GEOM={GEOM} but the JSONs say geom={per_target[0].get('geom')}")
    want = per_target[0].get("targets_in_set")
    if want and sorted(tg) != sorted(want):
        bad.append(f"target set {per_target[0].get('target_set')}: missing {sorted(set(want) - set(tg))},"
                   f" unexpected {sorted(set(tg) - set(want))}")
    cells = {json.dumps({a: sorted(m) for a, m in t.get("protocol_A", {}).items()}) for t in per_target}
    if len(cells) > 1:
        bad.append(f"arms x models of protocol_A differ between targets: {sorted(cells)}")
    arms = per_target[0].get("arms")
    if arms is not None and list(per_target[0].get("protocol_A", {})) != list(arms):
        bad.append(f"protocol_A arms {list(per_target[0].get('protocol_A', {}))} != recorded arms {arms}")
    miss = [p.name for p in preds_paths if not p.exists()]
    if miss:
        bad.append(f"preds parquet missing: {miss}")
    return bad


def fmt(v, spec):
    return "n/a" if v is None or not np.isfinite(v) else format(v, spec)


def write_reports(report, df, t0, preds):
    """rev 5 repo tables (ESPLEY_REPORT_PREFIX): full Protocol A table, KRR headline (+ md), input record."""
    prefix = str(PREFIX)
    if "arm" not in preds:
        sys.exit("ESPLEY_REPORT_PREFIX: predictions have no `arm` column (needs rev 5 train_ml_single.py output)")
    arms = list(t0.get("arms") or list(t0["protocol_A"]))
    targets = [t for t in (t0.get("targets_in_set") or list(report["per_target"])) if t in report["per_target"]]
    base = R5.BASE
    pa = df[df.protocol == "A"].drop(columns=["protocol"]).copy()
    pa["note"] = np.where(pa.arm == "EXT_SEL", SEL_NOTE, "")
    head = pa[pa.model == HEAD_MODEL].copy()
    if head.empty:
        sys.exit(f"ESPLEY_REPORT_PREFIX: no {HEAD_MODEL} cells in the Protocol A table")
    head["label"] = head.target.map(TARGET_LABEL).fillna(head.target)
    cols = list(HEAD_COLS)
    if base in arms:
        ref = head[head.arm == base].set_index("target")
        for k in ("mae", "nmae"):
            head[f"delta_{k}_vs_{base}"] = head[f"test_{k}"] - head.target.map(ref[f"test_{k}"])
            cols.append(f"delta_{k}_vs_{base}")
    order = {ta: i for i, ta in enumerate((t, a) for t in targets for a in arms)}
    head["_o"] = [order.get(ta, len(order)) for ta in zip(head.target, head.arm)]
    head = head.sort_values("_o")[cols + ["note"]]

    n_feat = t0.get("arm_n_features") or {}
    cell = {(r.target, r.arm): r for r in head.itertuples(index=False)}

    def arm_head(a):
        return f"{a}{' †' if a == 'EXT_SEL' else ''} ({n_feat.get(a, '?')})"

    deltas = [a for a in ("EXT_SEL", "EXT_ALL") if a in arms and base in arms]
    rows_name = Path(t0.get("rows_file") or "?").name
    lines = [f"# {TITLE} — {t0.get('geom')} · {HEAD_MODEL} · arms {', '.join(arms)}", "",
             f"Protocol A test folds: 80/10/10 over seeds {'/'.join(map(str, R5.PROTO_A_SEEDS))}, nested per-seed "
             f"GridSearchCV 5-fold, StandardScaler on X, y standardised. n = {t0.get('n_rows_ml')} rxns "
             f"(`{rows_name}`, sha256 `{str(t0.get('rows_sha256'))[:12]}`). Arm column counts in parentheses.", "",
             "## test MAE ± sd over seeds (kcal/mol)", "",
             "| Target | " + " | ".join(arm_head(a) for a in arms) + "".join(f" | {a} − {base}" for a in deltas) + " |",
             "|---|" + "---:|" * (len(arms) + len(deltas))]
    for t in targets:
        vals = []
        for a in arms:
            r = cell.get((t, a))
            vals.append("n/a" if r is None else f"{fmt(r.test_mae, '.2f')} ± {fmt(r.test_mae_sd, '.2f')}")
        for a in deltas:
            r = cell.get((t, a))
            vals.append("n/a" if r is None else fmt(getattr(r, f"delta_mae_vs_{base}"), "+.2f"))
        lines.append(f"| {TARGET_LABEL.get(t, t)} | " + " | ".join(vals) + " |")
    for title, col in (("NMAE (MAE / mean absolute deviation of the test targets)", "test_nmae"), ("r²", "test_r2")):
        lines += ["", f"## {title}", "", "| Target | " + " | ".join(arm_head(a) for a in arms) + " |",
                  "|---|" + "---:|" * len(arms)]
        for t in targets:
            rs = [cell.get((t, a)) for a in arms]
            lines.append(f"| {TARGET_LABEL.get(t, t)} | "
                         + " | ".join("n/a" if r is None else fmt(getattr(r, col), ".3f") for r in rs) + " |")
    lines.append("")
    if "EXT_SEL" in arms:
        lines += [f"† {SEL_NOTE}", "",
                  f"EXT_SEL = {base} + blocks {t0.get('ext_sel_blocks')} (`prereg_rev5b.json`, sha256 "
                  f"`{str(t0.get('prereg_rev5b_sha256'))[:12]}`)."]
    same = t0.get("arm_same_columns_as") or {}
    if same:
        lines += ["", "Arms with the same columns as an earlier arm (results copied, not refitted): "
                  + ", ".join(f"{a} = {b}" for a, b in same.items()) + "."]
    lines += ["", f"Every model and arm: `{Path(prefix).name}table.csv`; inputs: `{Path(prefix).name}meta.json`.", ""]

    meta = dict({k: t0.get(k) for k in SHARED}, rows_file=t0.get("rows_file"))
    meta.update(targets=targets, seeds=R5.PROTO_A_SEEDS, models=sorted(set(pa.model)), headline_model=HEAD_MODEL,
                ext_sel_note=SEL_NOTE if "EXT_SEL" in arms else None, ml_out=str(ML_OUT))
    R5.write_atomic(f"{prefix}table.csv", lambda f: pa.to_csv(f, index=False))
    R5.write_atomic(f"{prefix}headline.csv", lambda f: head.to_csv(f, index=False))
    R5.write_atomic(f"{prefix}headline.md", "\n".join(lines))
    R5.write_json(f"{prefix}meta.json", meta)
    print(f"reports         : {prefix}{{table.csv, headline.csv, headline.md, meta.json}}")


def main():
    json_paths = sorted(TARGETS_DIR.glob("target_*.json"))
    if not json_paths:
        sys.exit(f"no per-target JSON files in {TARGETS_DIR}")

    per_target = [json.loads(p.read_text()) for p in json_paths]
    preds_paths = [TARGETS_DIR / f"preds_{p.stem[len('target_'):]}.parquet" for p in json_paths]
    bad = gates(per_target, preds_paths)
    if bad:
        sys.exit(f"GATE ({TARGETS_DIR}): " + "; ".join(bad))

    t0 = per_target[0]
    geom = t0.get("geom")
    report = dict(n_targets=len(per_target), **{k: t0.get(k) for k in SHARED}, rows_file=t0.get("rows_file"),
                  per_target={})
    rows = []
    for t in per_target:
        tgt = t["target"]
        report["per_target"][tgt] = t
        pre_info = t.get("pre_ml") or {}
        pre_mae = pre_info.get("mae", float("nan"))
        pre_r = pre_info.get("pearson_r", float("nan"))
        n_feat = t.get("arm_n_features") or {}
        for fs_name, A in t.get("protocol_A", {}).items():
            for mname, r in A.items():
                rows.append(dict(geom=geom, protocol="A", feature_set=fs_name, arm=fs_name,
                                 n_features=n_feat.get(fs_name, np.nan), target=tgt, model=mname,
                                 pre_ml_mae=pre_mae, pre_ml_r=pre_r,
                                 train_mae=r["train_mae"], train_nmae=r.get("train_nmae", float("nan")),
                                 test_mae=r["test_mae"], test_mae_sd=r["test_mae_sd"],
                                 test_nmae=r.get("test_nmae", float("nan")),
                                 test_r2=r["test_r2"], test_range=r["test_range"],
                                 test_mae_pct_range=r["test_mae_pct_range"]))
        for fs_name, B in t.get("protocol_B", {}).items():
            for mname, r in B.items():
                rows.append(dict(geom=geom, protocol="B", feature_set=fs_name, arm=fs_name,
                                 n_features=n_feat.get(fs_name, np.nan), target=tgt, model=mname,
                                 pre_ml_mae=pre_mae, pre_ml_r=pre_r,
                                 train_mae=float("nan"), train_nmae=float("nan"),
                                 test_mae=r["pooled_oof"]["mae"], test_mae_sd=float("nan"),
                                 test_nmae=r["pooled_oof"].get("nmae", float("nan")),
                                 test_r2=r["pooled_oof"]["r2"], test_range=float("nan"),
                                 test_mae_pct_range=float("nan")))

    df = pd.DataFrame(rows)
    preds = pd.concat([pd.read_parquet(p) for p in preds_paths], ignore_index=True)
    if "arm" in preds:
        dup = preds.duplicated(PRED_KEYS)
        if dup.any():
            sys.exit(f"GATE ({TARGETS_DIR}): {int(dup.sum())} duplicate {PRED_KEYS} prediction rows")
    for out, write in ((OUT_REPORT, lambda f: f.write_text(json.dumps(report, indent=2))),
                       (OUT_TABLE, lambda f: df.to_csv(f, index=False)),
                       (OUT_PREDS, lambda f: preds.to_parquet(f, index=False))):
        R5.write_atomic(out, write)
    print(f"merged {len(preds_paths)} preds parquets ({len(preds)} rows) -> {OUT_PREDS}")

    print(f"geom            : {geom}  (target set {t0.get('target_set')}, n_rows_ml {t0.get('n_rows_ml')})")
    print(f"arms            : {t0.get('arms') or list(t0.get('protocol_A', {}))}")
    print(f"targets merged  : {len(per_target)}")
    print(f"report          : {OUT_REPORT}")
    print(f"table (rows={len(df)}): {OUT_TABLE}")
    if PREFIX:
        write_reports(report, df, t0, preds)


if __name__ == "__main__":
    main()
