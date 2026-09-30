#!/usr/bin/env python3
"""analyze_extra.py — 재학습 없이 보고용 표 2개를 만든다 (기하 하나, arm 하나, 모델 KRR_rbf).

  (A) charge_breakdown_<geom>.csv : 전하 그룹별 test MAE  (predictions.parquet만 사용, 추가 계산 0)
  (B) group_split_<geom>.csv      : dipolarophile / dipole 그룹 분할 robustness (KRR, 기존 튠 HP 재사용, ~2분)
  + analyze_extra_<geom>.json     : 입력 기록 (arm, 열 수, 파일 경로와 sha256)

arm = $ESPLEY_DOWNSTREAM_ARM (기본 ESPLEY73; rev 5에서는 EXT_SEL 등), 열 = train_ml_single.resolve_arm(arm).
확장 블록(B1..B6)을 쓰는 arm이면 rev5_common.FEAT_EXT를 rxn_id로 합친다 (모든 행 ext_status ok, 필요한 열에 NaN 없음).
행 = --rows (ML과 같은 행·순서) → (B)의 random 열은 헤드라인과 같은 분할·seed별 HP·파이프라인(_make_pipe).
ml_report.json의 geom·rows_sha256·features_sha256(확장 arm이면 ext_features_sha256, EXT_SEL이면 prereg_rev5b_sha256도)이
이번 입력과 다르거나, 기록된 그 arm의 열 목록이 resolve_arm과 다르거나, 어느 타깃이든 그 arm의 KRR_rbf seed별 HP가 없으면
중단. EXT_SEL(또는 prereg_rev5b.json이 생긴 뒤의 확장 arm)이면 --feat·블록 파일·(rows_rev5.csv인) --rows의 sha256이
prereg_rev5b.json `inputs`(Phase C 선별 입력)와 같아야 한다 (train_ml_single.check_prereg_inputs, 건너뛰기 전에 확인).
이미 있는 출력은 건너뛴다.

실행 (sbatch r4_downstream.sh / r5_downstream.sh):
  python analyze_extra.py [--geom g1] [--feat PARQUET] [--ml-out DIR] [--rows CSV] [--csv COLEY_CSV] [--out DIR]
기본값: $ESPLEY_GEOM, $ESPLEY_FEAT, $ESPLEY_ML_OUT (predictions.parquet, ml_report.json), $ESPLEY_ROWS, $ESPLEY_COLEY_CSV.
  ESPLEY_RES5=1: rows results_rev5/rows_rev5.csv, 출력 results_rev5/downstream_<arm>/
  그 외:         rows results_rev4/rows_rev4.csv, 출력 results_rev4/downstream_<geom>/ (arm ESPLEY73만; 다른 arm은 --out 필요)
"""
import argparse
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev5_common as R5                                            # noqa: E402
from train_ml_single import (FEATURE_SETS, GEOM_TAG, GRIDS, SEEDS, _make_pipe, arm_needs_ext,  # noqa: E402
                             check_prereg_inputs, load_ext, merge_ext, resolve_arm, sha256, split_80_10_10)

RDLogger.DisableLog("rdApp.*")
CSV = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/dipolar_cycloaddition/full_dataset.csv")
HEADLINE_MODEL = "KRR_rbf"
ARM = os.environ.get("ESPLEY_DOWNSTREAM_ARM") or "ESPLEY73"
RES5_ENV = os.environ.get("ESPLEY_RES5") or "0"
RES5 = RES5_ENV == "1"
GS_COLS = ["target", "random", "group_dipolarophile", "ratio_dph", "group_dipole", "ratio_dip"]


def parse_args():
    env = os.environ.get
    ap = argparse.ArgumentParser()
    ap.add_argument("--geom", default=env("ESPLEY_GEOM"), help="g1 | g2 (output label)")
    ap.add_argument("--feat", type=Path, default=env("ESPLEY_FEAT"))
    ap.add_argument("--ml-out", type=Path, default=env("ESPLEY_ML_OUT"), help="dir with predictions.parquet, ml_report.json")
    ap.add_argument("--rows", type=Path, default=env("ESPLEY_ROWS") or (R5.ROWS5 if RES5 else R5.ROWS4))
    ap.add_argument("--csv", type=Path, default=env("ESPLEY_COLEY_CSV", CSV))
    ap.add_argument("--out", type=Path, default=None,
                    help="default results_rev5/downstream_<arm>/ (ESPLEY_RES5=1) or results_rev4/downstream_<geom>/")
    a = ap.parse_args()
    if a.geom not in GEOM_TAG or a.feat is None or a.ml_out is None:
        ap.error(f"need --geom {'|'.join(GEOM_TAG)}, --feat, --ml-out (or ESPLEY_GEOM, ESPLEY_FEAT, ESPLEY_ML_OUT)")
    if RES5_ENV not in ("0", "1"):
        ap.error(f"ESPLEY_RES5 must be 0 or 1, got {RES5_ENV!r}")
    return a


def out_dir(a, arm):
    if a.out is not None:
        return a.out
    if RES5:
        return R5.RES5 / f"downstream_{arm}"
    if arm != "ESPLEY73":                              # results_rev4/downstream_<geom>/ holds the rev 4 ESPLEY73 outputs
        sys.exit(f"ESPLEY_DOWNSTREAM_ARM={arm} needs ESPLEY_RES5=1 or --out")
    return R5.RES4 / f"downstream_{a.geom}"


def load_feat(path, rows_csv, geom, cols, ext):
    """Feature rows in the pre-registered order; every row must be ok (and ext_status ok), with no NaN in cols."""
    rows = pd.read_csv(rows_csv)[["rxn_id"]]
    if rows.rxn_id.duplicated().any():
        sys.exit(f"{int(rows.rxn_id.duplicated().sum())} duplicate rxn_id in {rows_csv}")
    feat = rows.merge(pd.read_parquet(path), on="rxn_id", how="left", validate="1:1")
    ext_info = None
    if ext:
        blocks, ext_info = load_ext()
        feat = merge_ext(feat, blocks, how="left")
    bad = (feat.xtb_status != "ok") | feat[list(dict.fromkeys(cols))].isna().any(axis=1)
    if ext:
        bad |= feat.ext_status != "ok"
    if bad.any():
        sys.exit(f"{int(bad.sum())} of {len(feat)} rows of {rows_csv} are missing / not ok / NaN in {path}"
                 f"{' + ' + ext_info['ext_features_file'] if ext else ''}: {feat.rxn_id[bad].tolist()[:10]}")
    tags = set(feat.geom.astype(str)) if "geom" in feat else {"<no geom column: rev 3 parquet?>"}
    if tags != {GEOM_TAG[geom]}:
        sys.exit(f"{path} has geom {sorted(tags)}, not --geom {geom}")
    return feat, ext_info


def write_csv(df, path):
    R5.write_atomic(path, lambda f: df.to_csv(f, index=False))
    print(f"wrote {path}")


# ---------------------------------------------------------------- (A) charge breakdown
def charge_breakdown(feat, pred, geom, arm, n_cols):
    if "geom" in pred and set(pred.geom) != {geom}:
        sys.exit(f"predictions have geom {sorted(set(pred.geom))}, not --geom {geom}")
    f = feat.set_index("rxn_id")
    sel = (pred.arm == arm) if "arm" in pred else (pred.feature_set == n_cols)
    p = pred[(pred.model == HEADLINE_MODEL) & sel].copy()
    if p.empty:
        sys.exit(f"no {HEADLINE_MODEL} / {arm} predictions in the ML output")
    if set(p.feature_set) != {n_cols}:
        sys.exit(f"{arm} predictions have feature_set {sorted(set(p.feature_set))}, the arm has {n_cols} columns")
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


def group_split(feat, report, csv, arm, cols):
    dip, dph = components(feat.rxn_id.tolist(), csv)
    feat = feat.assign(dipole=feat.rxn_id.map(dip), dph=feat.rxn_id.map(dph))
    print(f"unique dipoles {feat.dipole.nunique()}, dipolarophiles {feat.dph.nunique()}, rxns {len(feat)}")
    X = feat[cols].values
    rows = []
    for tg, node in report.items():
        res = node["protocol_A"][arm][HEADLINE_MODEL]    # presence checked in main()
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
    arm = ARM
    cols = resolve_arm(arm)
    ext = arm_needs_ext(arm)
    if ext and a.geom != "g1":
        sys.exit(f"arm {arm} uses the blocks B1..B6, which exist for G1 only (--geom {a.geom})")
    if ext and not R5.FEAT_EXT.exists():
        sys.exit(f"arm {arm} needs the block parquet {R5.FEAT_EXT} (Phase B), missing")
    checked = check_prereg_inputs([arm], a.feat, R5.FEAT_EXT if ext else None, a.rows)   # EXT_SEL: files C used
    if checked:
        print(f"prereg_rev5b.json inputs match: {', '.join(checked)}", flush=True)
    out = out_dir(a, arm)
    out.mkdir(parents=True, exist_ok=True)
    out_cb, out_gs = out / f"charge_breakdown_{a.geom}.csv", out / f"group_split_{a.geom}.csv"
    if out_cb.exists() and out_gs.exists():
        print(f"{out_cb.name}, {out_gs.name} exist in {out} — skip")
        return
    rep_path = a.ml_out / "ml_report.json"
    if not rep_path.exists():
        sys.exit(f"{rep_path} missing — run aggregate_ml.py first")
    rep = json.loads(rep_path.read_text())
    want = dict(geom=a.geom, rows_sha256=sha256(a.rows), features_sha256=sha256(a.feat))
    if ext:
        want["ext_features_sha256"] = sha256(R5.FEAT_EXT)
    if arm == "EXT_SEL":
        want["prereg_rev5b_sha256"] = sha256(R5.PREREG5B_JSON)
    stale = {k: (rep.get(k), v) for k, v in want.items() if rep.get(k) != v}
    if stale:                                          # also refuses reports without geom / rows_sha256 (rev 3)
        sys.exit(f"{rep_path} was made with other inputs (report, now): {stale}")
    report = rep["per_target"]
    rec = (rep.get("arm_columns") or {}).get(arm)
    if rec is None and arm in FEATURE_SETS and not rep.get("arms"):
        rec = cols                                     # rev 4 report: arms = FEATURE_SETS, columns not recorded
    if rec != cols:
        sys.exit(f"{rep_path}: columns recorded for arm {arm} differ from resolve_arm({arm}) "
                 f"({None if rec is None else len(rec)} vs {len(cols)})")
    gaps = [tg for tg, node in report.items()
            if len((node.get("protocol_A", {}).get(arm, {}).get(HEADLINE_MODEL) or {})
                   .get("per_seed_best_params", [])) != len(SEEDS)]
    if gaps:
        sys.exit(f"{rep_path}: no protocol_A {arm} {HEADLINE_MODEL} per-seed HP for {gaps}")
    feat, ext_info = load_feat(a.feat, a.rows, a.geom, cols + ["charge1", "charge2"] + list(report), ext)
    print(f"geom {a.geom}, arm {arm} ({len(cols)} columns), model {HEADLINE_MODEL}: {len(feat)} rows ({a.rows}), "
          f"feat {a.feat}{' + ' + ext_info['ext_features_file'] if ext else ''}, ml {a.ml_out} -> {out}\n"
          f"ml_report targets: {list(report)}", flush=True)

    if out_cb.exists():
        print(f"{out_cb} exists — skip (A)")
    else:
        write_csv(charge_breakdown(feat, pd.read_parquet(a.ml_out / "predictions.parquet"), a.geom, arm, len(cols)),
                  out_cb)

    if out_gs.exists():
        print(f"{out_gs} exists — skip (B)")
    else:
        write_csv(group_split(feat, report, a.csv, arm, cols), out_gs)

    R5.write_json(out / f"analyze_extra_{a.geom}.json",
                  dict(arm=arm, model=HEADLINE_MODEL, n_features=len(cols), geom=a.geom, n_rows=len(feat),
                       rows_file=str(Path(a.rows).resolve()), rows_sha256=want["rows_sha256"],
                       features_file=str(a.feat), features_sha256=want["features_sha256"],
                       ext_features_file=ext_info["ext_features_file"] if ext else None,
                       ext_features_sha256=ext_info["ext_features_sha256"] if ext else None,
                       ml_report=str(rep_path), ml_report_sha256=sha256(rep_path),
                       ext_sel_blocks=rep.get("ext_sel_blocks") if arm == "EXT_SEL" else None,
                       prereg_inputs_checked=checked))


if __name__ == "__main__":
    main()
