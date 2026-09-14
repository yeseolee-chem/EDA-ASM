#!/usr/bin/env python3
"""analyze_extra.py — 재학습 없이 보고용 표 2개를 만든다.

  (A) charge_breakdown.csv : 전하 그룹별 test MAE  (predictions.parquet만 사용, 추가 계산 0)
  (B) group_split.csv      : dipolarophile 그룹 분할 robustness (KRR, 기존 튠 HP 재사용, ~2분)

실행:  cd analysis/espley_xtb_repro && python analyze_extra.py
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from rdkit import Chem, RDLogger                                    # noqa: E402
from rdkit.Chem.rdMolDescriptors import CalcMolFormula              # noqa: E402
from sklearn.kernel_ridge import KernelRidge                        # noqa: E402
from sklearn.metrics import mean_absolute_error                     # noqa: E402
from sklearn.model_selection import GroupShuffleSplit, train_test_split  # noqa: E402
from sklearn.pipeline import make_pipeline                          # noqa: E402
from sklearn.preprocessing import StandardScaler                    # noqa: E402

RDLogger.DisableLog("rdApp.*")
HERE = Path(__file__).resolve().parent
RES = HERE / "results"
CSV = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/dipolar_cycloaddition/full_dataset.csv")
SEEDS = [22, 23, 14, 1, 2]
HEADLINE_MODEL, HEADLINE_ARM = "KRR_rbf", 72

feat = pd.read_parquet(RES / "xtb_features.parquet")
pred = pd.read_parquet(RES / "predictions.parquet")

# ---------------------------------------------------------------- (A) charge breakdown
f = feat.set_index("rxn_id")
p = pred[(pred.model == HEADLINE_MODEL) & (pred.feature_set == HEADLINE_ARM)].copy()
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
pd.DataFrame(rows).round(3).to_csv(RES / "charge_breakdown.csv", index=False)
print("wrote results/charge_breakdown.csv")

# ---------------------------------------------------------------- (B) grouped split
def components(rxn_ids):
    """dipole / dipolarophile canonical SMILES per rxn.

    dipole = the reactant contributing 3 atoms to the new 5-ring (same rule as stage 1).
    """
    full = pd.read_csv(CSV).set_index("rxn_id")
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

dip, dph = components(feat.rxn_id.tolist())
feat["dipole"], feat["dph"] = feat.rxn_id.map(dip), feat.rxn_id.map(dph)
print(f"unique dipoles {feat.dipole.nunique()}, dipolarophiles {feat.dph.nunique()}")

src = (HERE / "train_ml_single.py").read_text()
ns = {}
exec(src[src.index("DIST11"):src.index("TARGETS =")], ns)
F = ns["FEATURE_SETS"][f"ESPLEY{HEADLINE_ARM}"]
report = json.load(open(RES / "ml_report.json"))["per_target"]

rows = []
for tg, node in report.items():
    if tg == "dft_disp_dft":                       # b_disp == target: not a prediction task
        continue
    bp = node["protocol_A"][f"ESPLEY{HEADLINE_ARM}"][HEADLINE_MODEL]["best_params"]
    P = dict(alpha=bp["kernelridge__alpha"], gamma=bp["kernelridge__gamma"])
    X, y = feat[F].values, feat[tg].values
    out = {}
    for mode, groups in [("random", None), ("group_dipolarophile", feat.dph.values),
                         ("group_dipole", feat.dipole.values)]:
        maes = []
        for seed in SEEDS:
            if groups is None:
                tr, rest = train_test_split(np.arange(len(feat)), test_size=0.20, random_state=seed)
                _, te = train_test_split(rest, test_size=0.50, random_state=seed)
            else:
                tr, te = next(GroupShuffleSplit(1, test_size=0.10, random_state=seed).split(X, y, groups=groups))
            mdl = make_pipeline(StandardScaler(), KernelRidge(kernel="rbf", **P)).fit(X[tr], y[tr])
            maes.append(mean_absolute_error(y[te], mdl.predict(X[te])))
        out[mode] = float(np.mean(maes))
    out.update(target=tg, ratio_dph=out["group_dipolarophile"] / out["random"],
               ratio_dip=out["group_dipole"] / out["random"])
    rows.append(out)
    print(f"  {tg:20s} random {out['random']:.2f}  grp-dph {out['group_dipolarophile']:.2f} "
          f"({out['ratio_dph']:.2f}x)  grp-dip {out['group_dipole']:.2f} ({out['ratio_dip']:.2f}x)", flush=True)
pd.DataFrame(rows)[["target", "random", "group_dipolarophile", "ratio_dph",
                    "group_dipole", "ratio_dip"]].round(3).to_csv(RES / "group_split.csv", index=False)
print("wrote results/group_split.csv")
