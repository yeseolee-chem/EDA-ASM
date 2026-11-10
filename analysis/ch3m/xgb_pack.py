#!/usr/bin/env python3
"""xgb_pack.py — CH3M Phase XP (docs/specs/CH3M_MODELS.md 4): the XGB input packs, built in env_d1fu2 (DeployData needs
torch) for xgb_m.py in env_espley (no torch). Same pairs / splits / labels as the TF · GNN units.

  python xgb_pack.py           -> $CH3M/xgb_pack/{lf,g2}.npz + .json, results/XP_report.json (exit 3 on a failed check)

Pack content (pair-table order = DeployData P order = F_pairs_cr2 rows):
  pair_id, ref_kind, pos, a_rxn, b_rxn, core (S_folds.csv; '' when absent), y10 (dd_<10 targets>), y4 = y10 @ M,
  ra, rb (token-row indices, -1 = no token), has_tok, usable_pair; rids (token-row order); X = the source's 182-column
  parquet values reindexed to the token rows, columns = T_report glob_cols (the LADDER XData rule: train_xgb.load_x182,
  not the token glob); split lists (original split-file ids) as f<k>_<train|test|inner_train|inner_val>.
Checks (STOP): y10 bit-equal to the label file dd_<t>; the usable filter applied to each split list = DeployData.split(k);
X rows equal to the token glob (<= 1e-12 relative, finite rows); lf pair_list_sha = ch3 V-6 value.
Sources: lf = DeployData('lf') (data_ch3 binds $CH3/lf); g2 = DeployData('g2') (LADDER usable rule, = deploy C-2)."""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ch3m_common as C  # noqa: E402

LF_PAIR_LIST_SHA = "7de6e33ac550656a26c6b2cbfdf63069cfe3ab996d4a65a9811e39a15116d146"
S_FOLDS = C.REPO / "analysis" / "d1_pair" / "results" / "S_folds.csv"
KEYS = ("train", "test", "inner_train", "inner_val")


def build(source):
    import data_ch3 as DC3
    DL = DC3.DL
    D = DL.DeployData(source)
    cfg = DL.SOURCES[source]
    cols = list(D.trep["glob_cols"])
    Xp = pd.read_parquet(cfg["x182"])
    Xp = Xp.assign(rxn_id=Xp.rxn_id.astype(int)).set_index("rxn_id")
    if list(Xp.columns) != cols:
        C.stop(f"{cfg['x182']}: columns differ from T_report glob_cols")
    X = Xp.reindex([int(r) for r in D.rids]).to_numpy(dtype=float)
    fin = np.isfinite(X).all(1)
    dev = np.abs(X - D.glob) > 1e-12 * np.maximum(1.0, np.abs(D.glob))
    n_dev = int((dev & np.isfinite(X)).any(1).sum())
    L = pd.read_csv(C.LABELS)
    L["pair_id"] = L.pair_id.astype(str)
    if L.pair_id.tolist() != [str(q) for q in D.P.index]:
        C.stop(f"{source}: label file row order differs from the DeployData pair table")
    y10 = L[[f"dd_{t}" for t in C.TARGETS10]].to_numpy(float)
    y_eq = bool(np.array_equal(y10, D.y, equal_nan=True))
    S = pd.read_csv(S_FOLDS)
    core = S.assign(pair_id=S.pair_id.astype(str)).set_index("pair_id").core.astype(str).reindex(D.P.index).fillna("")
    arrays = dict(pair_id=np.array([str(q) for q in D.P.index]), ref_kind=D.P.ref_kind.to_numpy().astype(str),
                  pos=D.P.pos.to_numpy().astype(str), a_rxn=D.P.a_rxn.to_numpy(int), b_rxn=D.P.b_rxn.to_numpy(int),
                  core=core.to_numpy().astype(str), y10=y10, y4=y10 @ C.M(), ra=D.ra.astype(int), rb=D.rb.astype(int),
                  has_tok=D.has_tok, usable_pair=D.usable_pair, rids=np.asarray(D.rids, dtype=int), X=X)
    split_ok = {}
    for k in range(C.NF):
        rec = D.folds[k]
        sp = D.split(k)
        for key in KEYS:
            ids = [str(q) for q in rec[key]]
            arrays[f"f{k}_{key}"] = np.array(ids)
            rows = np.array([D.pid_row[q] for q in rec[key]], dtype=int)
            split_ok[f"{k}/{key}"] = bool(np.array_equal(rows[D.usable_pair[rows]], sp[key]))
    out = C.XPACK / f"{source}.npz"
    C.write_npz(out, **arrays)
    info = dict(source=source, written=C.now(), n_pairs=int(len(D.P)), n_tokens=int(len(D.rids)),
                n_usable_rxn=int(D.usable_rxn.sum()), n_pairs_P=len(D.pair_list), pair_list_sha=D.pair_list_sha,
                glob_cols=cols, sources_sha256=dict(D.sha, x182_file=C.sha256(cfg["x182"]), labels_file=C.sha256(C.LABELS),
                                                    s_folds=C.sha256(S_FOLDS)),
                inputs=dict(D.inputs), usable_rule=D.usable_rule, npz=str(out), npz_sha256=C.sha256(out),
                checks=dict(y10_bit_equal_labels=y_eq, splits_equal_DeployData=all(split_ok.values()),
                            n_split_lists_checked=len(split_ok), X_rows_vs_token_glob_n_dev=n_dev,
                            X_finite_rows=int(fin.sum()), X_equals_token_glob=n_dev == 0,
                            pair_list_sha_lf=(D.pair_list_sha == LF_PAIR_LIST_SHA) if source == "lf" else None,
                            core_missing_in_fold_lists=int(sum(1 for k in range(C.NF) for key in KEYS
                                                               for q in D.folds[k][key] if core.get(str(q), "") == ""))))
    info["ok"] = bool(y_eq and all(split_ok.values()) and n_dev == 0 and info["checks"]["core_missing_in_fold_lists"] == 0
                      and (source != "lf" or info["checks"]["pair_list_sha_lf"]))
    C.write_json(C.XPACK / f"{source}.json", info)
    return info


def main():
    rep = dict(spec="docs/specs/CH3M_MODELS.md 4", written=C.now(), packs={})
    for s in ("lf", "g2"):
        info = build(s)
        rep["packs"][s] = dict(ok=info["ok"], checks=info["checks"], npz_sha256=info["npz_sha256"],
                               json_sha256=C.sha256(C.XPACK / f"{s}.json"), n_pairs_P=info["n_pairs_P"],
                               pair_list_sha=info["pair_list_sha"], n_tokens=info["n_tokens"])
        print(f"[XP] {s}: {json.dumps(rep['packs'][s]['checks'])}", flush=True)
    rep["ok"] = all(v["ok"] for v in rep["packs"].values())
    C.write_json(C.RES / "XP_report.json", rep)
    print(json.dumps(dict(ok=rep["ok"]), indent=1))
    return 0 if rep["ok"] else 3


if __name__ == "__main__":
    sys.exit(main())
