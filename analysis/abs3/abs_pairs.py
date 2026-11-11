"""abs_pairs.py — ABS3 pair ΔΔ from reaction predictions (docs/specs/ABS3_MODELS.md 5 C-P, 7). numpy / pandas only (used
by xgb_abs.py collect in env_espley and by tests_abs.py in both environments).

assemble(rx, pairs, label): for every fold k and its usable test pairs, yhat = yhat(a) - yhat(b) from the fold-k
reaction predictions; yhat_seed_sd = std (ddof 0) over the seeds of the seed-level ΔΔ (NaN without seed columns: XGB);
y = the pair label dd_<10> @ M (F_pairs_cr2), columns = the pair models' K_predictions columns, rung abs3, arm S.
direct(rx, pairs): the same ΔΔ by an independent merge (C-P compares the two, <= 1e-9)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import abs_common as C  # noqa: E402  (stdlib; STOP = exit 3)

TARGETS3 = ["strain", "inter", "solv", "barrier"]
PAIR_COLS = ["pair_id", "ref_kind", "pos", "fold", "rung", "arm", "model", "target", "y", "yhat", "base", "beta",
             "yhat_seed_sd"]


def pack_test_pairs(z, k):
    """fold k's usable test pairs from the pack: pair_id, ref_kind, pos, a_rxn, b_rxn, fold, dd4 columns y_<t>."""
    pid = z["pair_id"].astype(str)
    row = {q: i for i, q in enumerate(pid)}
    rows = np.array([row[q] for q in z[f"p{k}_test"].astype(str)], dtype=int)
    rows = rows[z["usable_pair"][rows].astype(bool)]
    d = pd.DataFrame(dict(pair_id=pid[rows], ref_kind=z["ref_kind"][rows].astype(str), pos=z["pos"][rows].astype(str),
                          a_rxn=z["a_rxn"][rows].astype(np.int64), b_rxn=z["b_rxn"][rows].astype(np.int64), fold=int(k)))
    for j, t in enumerate(TARGETS3):
        d[f"y_{t}"] = z["dd4"][rows, j]
    return d


def assemble(rx, pairs, label, rung="abs3", arm="S"):
    """rx: reaction predictions of ONE model (rxn_id, fold, target, yhat[, yhat_s<j>]); pairs: pack_test_pairs of every
    fold concatenated. Returns the pair frame (PAIR_COLS)."""
    seed_cols = sorted([c for c in rx.columns if c.startswith("yhat_s") and c[6:].isdigit()], key=lambda c: int(c[6:]))
    out = []
    for k, P in pairs.groupby("fold", sort=True):
        R = rx[rx.fold == k]
        for t in TARGETS3:
            q = R[R.target == t].set_index("rxn_id")
            if q.index.duplicated().any():
                C.stop(f"duplicated reaction predictions (fold {k}, {t})")
            ia = q.index.get_indexer(P.a_rxn.to_numpy())
            ib = q.index.get_indexer(P.b_rxn.to_numpy())
            if (ia < 0).any() or (ib < 0).any():
                C.stop(f"C-P: {int((ia < 0).sum() + (ib < 0).sum())} test-pair reactions without a fold-{k} prediction "
                       f"({label}, {t})")
            yh = q.yhat.to_numpy(float)
            d = pd.DataFrame(dict(pair_id=P.pair_id.to_numpy(), ref_kind=P.ref_kind.to_numpy(), pos=P.pos.to_numpy(),
                                  fold=int(k), rung=rung, arm=arm, model=label, target=t, y=P[f"y_{t}"].to_numpy(float),
                                  yhat=yh[ia] - yh[ib], base=0.0, beta=0.0))
            if seed_cols:
                S = q[seed_cols].to_numpy(float)
                d["yhat_seed_sd"] = (S[ia] - S[ib]).std(axis=1)
            else:
                d["yhat_seed_sd"] = np.nan
            out.append(d)
    return pd.concat(out, ignore_index=True)[PAIR_COLS]


def direct(rx, pairs):
    """the ΔΔ yhat(a) - yhat(b) by merging the reaction predictions on a and on b (independent of assemble)."""
    base = pairs.melt(id_vars=["pair_id", "fold", "a_rxn", "b_rxn"], value_vars=[f"y_{t}" for t in TARGETS3],
                      var_name="target", value_name="y")
    base["target"] = base.target.str[2:]
    r = rx[["rxn_id", "fold", "target", "yhat"]]
    a = base.merge(r.rename(columns=dict(rxn_id="a_rxn", yhat="ya")), on=["a_rxn", "fold", "target"], how="left")
    b = a.merge(r.rename(columns=dict(rxn_id="b_rxn", yhat="yb")), on=["b_rxn", "fold", "target"], how="left")
    b["yhat"] = b.ya - b.yb
    return b[["pair_id", "target", "y", "yhat"]]


def labels_truth(labels_csv, M):
    """the pair truth from F_pairs_cr2.csv itself: dd_<10 targets> @ M, long (pair_id, target, y)."""
    T10 = ["d1", "d2", "elst", "pauli", "oi", "disp", "cpcm", "cds", "c_ghost", "barrier"]
    L = pd.read_csv(labels_csv)
    y4 = L[[f"dd_{t}" for t in T10]].to_numpy(float) @ np.asarray(M, dtype=float)
    return pd.concat([pd.DataFrame(dict(pair_id=L.pair_id.astype(str), target=t, y=y4[:, j]))
                      for j, t in enumerate(TARGETS3)], ignore_index=True)


def cp_check(assembled, rx, pairs, truth, tol=1e-9):
    """C-P: assembled yhat = direct yhat(a) - yhat(b) and assembled y = F_pairs_cr2 dd @ M (truth, read from the csv)
    (<= tol)."""
    d = direct(rx, pairs)[["pair_id", "target", "yhat"]].merge(
        assembled[["pair_id", "target", "y", "yhat"]], on=["pair_id", "target"], suffixes=("_direct", "_assembled"),
        how="outer", indicator=True)
    same = bool((d._merge == "both").all())
    d = d.drop(columns="_merge").merge(truth.rename(columns=dict(y="y_truth")), on=["pair_id", "target"], how="left")
    dyh = float(np.abs(d.yhat_direct - d.yhat_assembled).max()) if same and len(d) else float("inf")
    dy = float(np.abs(d.y_truth - d.y).max()) if same and len(d) and d.y_truth.notna().all() else float("inf")
    fin = bool(np.isfinite(d.yhat_direct).all()) if same else False
    return dict(n_rows=int(len(d)), n_pairs=int(d.pair_id.nunique()), same_rows=same, finite=fin, max_abs_dyhat=dyh,
                max_abs_dy_vs_labels=dy, tol=tol, ok=bool(same and fin and dyh <= tol and dy <= tol))
