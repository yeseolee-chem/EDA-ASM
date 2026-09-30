#!/usr/bin/env python3
"""analyze_extra.py — 재학습 없이 보고용 표 2개를 만든다 (기하 하나씩, rev 4).

  (A) charge_breakdown_<geom>.csv : 전하 그룹별 test MAE  (predictions.parquet만 사용, 추가 계산 0)
  (B) group_split_<geom>.csv      : dipolarophile / dipole 그룹 분할 robustness (KRR, 기존 튠 HP 재사용, ~2분)

행 = --rows (rows_rev4.csv, ML과 같은 행·순서) → (B)의 random 열은 헤드라인과 같은 분할·seed별 HP·파이프라인(_make_pipe).
disp: G0에서는 b_disp == 라벨이라 예측 과제가 아니므로 (B)에서 뺀다. G1에서는 넣는다 (데이터로 확인, 어긋나면 중단).
ml_report.json의 geom·rows_sha256이 --geom·--rows와 다르거나 없으면 중단. 이미 있는 출력은 건너뛴다.

실행 (sbatch r4_downstream.sh):
  python analyze_extra.py [--geom g1] [--feat PARQUET] [--ml-out DIR] [--rows CSV] [--csv COLEY_CSV] [--out DIR]
기본값: $ESPLEY_GEOM, $ESPLEY_FEAT, $ESPLEY_ML_OUT (predictions.parquet, ml_report.json),
        $ESPLEY_ROWS (results_rev4/rows_rev4.csv), $ESPLEY_COLEY_CSV, results_rev4/downstream_<geom>/
"""
import argparse
import hashlib
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from rdkit import Chem, RDLogger                                    # noqa: E402
from sklearn.base import clone                                      # noqa: E402
from sklearn.metrics import mean_absolute_error                     # noqa: E402
from sklearn.model_selection import GroupShuffleSplit               # noqa: E402

from train_ml_single import FEATURE_SETS, GRIDS, _make_pipe, split_80_10_10  # noqa: E402

RDLogger.DisableLog("rdApp.*")
HERE = Path(__file__).resolve().parent
CSV = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/dipolar_cycloaddition/full_dataset.csv")
SEEDS = [22, 23, 14, 1, 2]
HEADLINE_MODEL, HEADLINE_ARM = "KRR_rbf", 73         # predictions.feature_set = len(feats)
FEAT_TAG = {"dft": "g0", "g1": "g1", "g2": "g2"}      # parquet `geom` column -> geometry label
DISP_IDENTITY_TOL = 1e-3                              # kcal/mol; G0 MAE(b_disp − dft_disp_dft) = 2.4e-6
GS_COLS = ["target", "random", "group_dipolarophile", "ratio_dph", "group_dipole", "ratio_dip"]


def parse_args():
    env = os.environ.get
    ap = argparse.ArgumentParser()
    ap.add_argument("--geom", default=env("ESPLEY_GEOM"), help="g0 | g1 | g2 (output label)")
    ap.add_argument("--feat", type=Path, default=env("ESPLEY_FEAT"))
    ap.add_argument("--ml-out", type=Path, default=env("ESPLEY_ML_OUT"), help="dir with predictions.parquet, ml_report.json")
    ap.add_argument("--rows", type=Path, default=env("ESPLEY_ROWS", HERE / "results_rev4" / "rows_rev4.csv"))
    ap.add_argument("--csv", type=Path, default=env("ESPLEY_COLEY_CSV", CSV))
    ap.add_argument("--out", type=Path, default=None, help="default results_rev4/downstream_<geom>/")
    a = ap.parse_args()
    if a.geom not in FEAT_TAG.values() or a.feat is None or a.ml_out is None:
        ap.error("need --geom g0|g1|g2, --feat, --ml-out (or ESPLEY_GEOM, ESPLEY_FEAT, ESPLEY_ML_OUT)")
    return a


def load_feat(path, rows_csv, geom, cols):
    """Feature rows in the pre-registered order; every row must be ok, with no NaN in cols."""
    rows = pd.read_csv(rows_csv)[["rxn_id"]]
    feat = rows.merge(pd.read_parquet(path), on="rxn_id", how="left", validate="1:1")
    bad = (feat.xtb_status != "ok") | feat[list(dict.fromkeys(cols))].isna().any(axis=1)
    if bad.any():
        sys.exit(f"{int(bad.sum())} of {len(feat)} rows of {rows_csv} are missing / not ok / NaN in {path}: "
                 f"{feat.rxn_id[bad].tolist()[:10]}")
    tags = set(feat.geom) if "geom" in feat else {"<no geom column: rev 3 parquet?>"}
    if {FEAT_TAG.get(t) for t in tags} != {geom}:
        sys.exit(f"{path} has geom {sorted(tags)}, not --geom {geom}")
    return feat


def write_csv(df, path):
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)
    print(f"wrote {path}")


# ---------------------------------------------------------------- (A) charge breakdown
def charge_breakdown(feat, pred, geom):
    if "geom" in pred and set(pred.geom) != {geom}:
        sys.exit(f"predictions have geom {sorted(set(pred.geom))}, not --geom {geom}")
    f = feat.set_index("rxn_id")
    p = pred[(pred.model == HEADLINE_MODEL) & (pred.feature_set == HEADLINE_ARM)].copy()
    if p.empty:
        print(f"  no {HEADLINE_MODEL} / {HEADLINE_ARM}-feature predictions — (A) skipped")
        return None
    extra = set(p.rxn_id) - set(f.index)
    if extra:
        sys.exit(f"predictions contain {len(extra)} rxn_ids outside --rows, e.g. {sorted(extra)[:5]}")
    p["charged"] = p.rxn_id.map((f.charge1 != 0) | (f.charge2 != 0))
    p["q2"] = p.rxn_id.map(f.charge2)
    rows = []
    for tg, g in p.groupby("target"):
        r = {"target": tg}
        for lab, sub in [("all", g), ("neutral", g[~g.charged]), ("charged", g[g.charged]),
                         ("q_minus2", g[g.q2 == -2]), ("q_plus1", g[g.q2 == 1])]:
            if len(sub):
                r[f"mae_{lab}"] = mean_absolute_error(sub.y, sub.yhat)
                r[f"n_{lab}"] = int(sub.rxn_id.nunique())
        rows.append(r)
    return pd.DataFrame(rows).round(4)


# ---------------------------------------------------------------- (B) grouped split
def components(rxn_ids, csv):
    """dipole / dipolarophile canonical SMILES per rxn.

    dipole = the reactant contributing 3 atoms to the new 5-ring (same rule as stage 1).
    """
    full = pd.read_csv(csv).set_index("rxn_id")
    dip, dph = {}, {}

    def canon(s):
        m = Chem.MolFromSmiles(s)
        for a in m.GetAtoms():
            a.SetAtomMapNum(0)
        return Chem.MolToSmiles(m)

    for rid in rxn_ids:
        smi = full.loc[rid, "rxn_smiles"]
        rs = smi.split(">>")[0].split(".")
        rmols = [Chem.MolFromSmiles(x) for x in rs]
        pmol = Chem.MolFromSmiles(smi.split(">>")[1])

        def bonds(m):
            return {tuple(sorted((b.GetBeginAtom().GetAtomMapNum(), b.GetEndAtom().GetAtomMapNum())))
                    for b in m.GetBonds()}
        formed = bonds(pmol) - set().union(*[bonds(m) for m in rmols])
        owner = {a.GetAtomMapNum(): k for k, m in enumerate(rmols) for a in m.GetAtoms()}
        idx2map = {a.GetIdx(): a.GetAtomMapNum() for a in pmol.GetAtoms()}
        rings = [set(idx2map[i] for i in r) for r in pmol.GetRingInfo().AtomRings()]
        rings = [r for r in rings if all(a in r and b in r for a, b in formed)]
        ring = min(rings, key=len)
        cnt = {}
        for m_ in ring:
            cnt[owner[m_]] = cnt.get(owner[m_], 0) + 1
        i = [k for k, v in cnt.items() if v == 3][0]
        dip[rid], dph[rid] = canon(rs[i]), canon(rs[1 - i])
    return dip, dph


def group_split(feat, report, csv, disp_exact):
    dip, dph = components(feat.rxn_id.tolist(), csv)
    feat = feat.assign(dipole=feat.rxn_id.map(dip), dph=feat.rxn_id.map(dph))
    print(f"unique dipoles {feat.dipole.nunique()}, dipolarophiles {feat.dph.nunique()}, rxns {len(feat)}")
    arm = f"ESPLEY{HEADLINE_ARM}"
    X = feat[FEATURE_SETS[arm]].values
    rows = []
    for tg, node in report.items():
        if tg == "dft_disp_dft" and disp_exact:        # b_disp == target (G0): not a prediction task
            continue
        res = node.get("protocol_A", {}).get(arm, {}).get(HEADLINE_MODEL)
        if res is None or len(res.get("per_seed_best_params", [])) != len(SEEDS):
            print(f"  {tg}: no protocol_A {arm} {HEADLINE_MODEL} per-seed HP in ml_report.json — skipped")
            continue
        per_seed = res["per_seed_best_params"]          # HP is per-seed (nested CV), in SEEDS order
        y = feat[tg].values
        out = {}
        for mode, groups in [("random", None), ("group_dipolarophile", feat.dph.values),
                             ("group_dipole", feat.dipole.values)]:
            maes = []
            for si, seed in enumerate(SEEDS):
                if groups is None:                      # = the ML split (same rows, same order)
                    tr, te = split_80_10_10(len(feat), seed)
                else:
                    tr, te = next(GroupShuffleSplit(1, test_size=0.10, random_state=seed).split(X, y, groups=groups))
                mdl = _make_pipe(clone(GRIDS[HEADLINE_MODEL][0])).set_params(**per_seed[si]).fit(X[tr], y[tr])
                maes.append(mean_absolute_error(y[te], mdl.predict(X[te])))
            out[mode] = float(np.mean(maes))
        out.update(target=tg, ratio_dph=out["group_dipolarophile"] / out["random"],
                   ratio_dip=out["group_dipole"] / out["random"])
        rows.append(out)
        print(f"  {tg:20s} random {out['random']:.2f} (headline {res.get('test_mae', np.nan):.2f})  "
              f"grp-dph {out['group_dipolarophile']:.2f} ({out['ratio_dph']:.2f}x)  "
              f"grp-dip {out['group_dipole']:.2f} ({out['ratio_dip']:.2f}x)", flush=True)
    return pd.DataFrame(rows, columns=GS_COLS).round(4)


def main():
    a = parse_args()
    out = a.out or HERE / "results_rev4" / f"downstream_{a.geom}"
    out.mkdir(parents=True, exist_ok=True)
    out_cb, out_gs = out / f"charge_breakdown_{a.geom}.csv", out / f"group_split_{a.geom}.csv"
    if out_cb.exists() and out_gs.exists():
        print(f"{out_cb.name}, {out_gs.name} exist in {out} — skip")
        return
    rep_path = a.ml_out / "ml_report.json"
    if not rep_path.exists():
        sys.exit(f"{rep_path} missing — run r4_ml_aggregate.sh first")
    rep = json.load(open(rep_path))
    want = dict(geom=a.geom, rows_sha256=hashlib.sha256(Path(a.rows).read_bytes()).hexdigest())
    stale = {k: (rep.get(k), v) for k, v in want.items() if rep.get(k) != v}
    if stale:                                          # also refuses reports without geom / rows_sha256 (rev 3)
        sys.exit(f"{rep_path} was made with other inputs (report, now): {stale}")
    report = rep["per_target"]
    F = FEATURE_SETS[f"ESPLEY{HEADLINE_ARM}"]
    feat = load_feat(a.feat, a.rows, a.geom, F + ["charge1", "charge2", "b_disp", "dft_disp_dft"] + list(report))
    disp_mae = float((feat.b_disp - feat.dft_disp_dft).abs().mean())
    disp_exact = disp_mae < DISP_IDENTITY_TOL
    if disp_exact != (a.geom == "g0"):                 # pre-registered: identity at G0 only (REV4 §3-1)
        sys.exit(f"MAE(b_disp − dft_disp_dft) {disp_mae:.2e} inconsistent with --geom {a.geom}")
    print(f"geom {a.geom}: {len(feat)} rows ({a.rows}), feat {a.feat}, ml {a.ml_out}\n"
          f"MAE(b_disp − dft_disp_dft) {disp_mae:.2e} -> disp {'= label, left out of (B)' if disp_exact else 'predicted'}; "
          f"ml_report targets: {list(report)}", flush=True)

    if out_cb.exists():
        print(f"{out_cb} exists — skip (A)")
    else:
        cb = charge_breakdown(feat, pd.read_parquet(a.ml_out / "predictions.parquet"), a.geom)
        if cb is not None:
            write_csv(cb, out_cb)

    if out_gs.exists():
        print(f"{out_gs} exists — skip (B)")
    else:
        write_csv(group_split(feat, report, a.csv, disp_exact), out_gs)


if __name__ == "__main__":
    main()
