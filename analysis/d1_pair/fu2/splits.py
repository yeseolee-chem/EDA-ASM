"""FU2 M-1 pair splits — the arm C train / test pairs of d1_train.masks and the rev 9 NN validation fold, exported for the
torch env (which does not import d1_train / sklearn-side code).

  python fu2/splits.py export        # (env_espley) -> $D1PAIR_SCRATCH/fu2/splits/{pairs.parquet, <split>__f<k>.json},
                                     #                 results/fu2/S_splits.json

Per unit u = dict(split, fold, arm="C") (arm C: train = outer-train T_main pairs incl. anchor pairs, y = dd, no f0):
  tr, te = d1_train.masks(u)                  (in-unit leak checks of rev 9: test core / reaction never in training)
  itr, ite = d1_train.inner_folds(tr)[0]      (GroupKFold(4) by core; rev 9 NN rule: first inner fold = validation)
pairs.parquet keeps only what training needs: pair_id, ref_kind, a_rxn, b_rxn, core (grouping), dd_<10 targets>.
sigma, family, Xb, positions are NOT exported (never model inputs; evaluation reads S_folds directly).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fu2_common as F  # noqa: E402

C = F.C
NF = {"T1": 5, "LODO": 5, "LOSO": 10}
KEEP = ["pair_id", "ref_kind", "a_rxn", "b_rxn", "core"] + [f"dd_{t}" for t in F.TARGETS]


def cmd_export():
    import d1_train as T
    S, X, f0, f0rep, sha = T.data()
    P = S[KEEP].copy()
    P["a_rxn"], P["b_rxn"] = P.a_rxn.astype(int), P.b_rxn.astype(int)
    out = F.SPLITS / "pairs.parquet"
    F.write_atomic(out, lambda t: P.to_parquet(t, index=False))
    rep = dict(pairs=str(out), pairs_sha256=F.sha256(out), n_pairs=len(P), inputs_sha256=sha, units={})
    for split, nf in NF.items():
        for k in range(nf):
            u = dict(group=split, split=split, arm="C", model="FU2", target="all", fold=k)
            tr, te = T.masks(u)
            itr, ite = T.inner_folds(tr)[0]
            pid = lambda idx: [str(x) for x in S.loc[idx, "pair_id"]]           # noqa: E731  (P00000 / A00000)
            rec = dict(split=split, fold=k, train=pid(tr), test=pid(te), inner_train=pid(itr), inner_val=pid(ite))
            # leak checks repeated on pair ids and reactions (C7 / C8 also re-check these in the torch env)
            st = {key: set(v) for key, v in rec.items() if isinstance(v, list)}
            assert not st["test"] & st["train"], "test pair in train"
            assert st["inner_train"] | st["inner_val"] == st["train"] and not st["inner_train"] & st["inner_val"]
            rx = lambda ids: set(P.set_index("pair_id").loc[list(ids), ["a_rxn", "b_rxn"]].to_numpy().ravel())  # noqa
            assert not rx(st["test"]) & rx(st["train"]), "test reaction in train"
            name = f"{split}__f{k}.json"
            F.write_json(F.SPLITS / name, rec)
            rep["units"][name] = dict(n_train=len(tr), n_test=len(te), n_inner_train=len(itr), n_inner_val=len(ite),
                                      sha256=F.sha256(F.SPLITS / name))
    F.write_json(F.FU2_RES / "S_splits.json", rep)
    print(json.dumps({k: v for k, v in rep.items() if k != "inputs_sha256"}, indent=1))
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["export"]:
        sys.exit(cmd_export())
    print(__doc__); sys.exit(2)
