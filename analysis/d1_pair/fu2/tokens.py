"""FU2 Phase T — atom tokens + T-2 graph per reaction (docs/specs/D1PAIR_FU2.md, Phase T).

  python fu2/tokens.py build          # Q_manifest use == True -> $D1PAIR_SCRATCH/fu2/tokens/<rxn_id>.npz + T_report.json

One reaction = n atom tokens (TS order, no padding) + one global token (its X182_d1 row, 182 columns, used by "A+G"):
  Z      [n]      atomic numbers (element one-hot is built at load time from T_report "elements")
  bin    [n, 6]   frag_B (0 = A dipole, 1 = B dipolarophile), dip_a, dip_mid, dip_b, dph_a, dph_b
  num    [n, m]   bond_dist (covalent-graph shortest path to the nearest reacting atom, capped at 8),
                  q_rel, q_frag, q_TS, dq_dist = q_frag - q_rel, dq_int = q_TS - q_frag,
                  v_rel, v_frag, v_TS, dv_dist = v_frag - v_rel, dv_int = v_TS - v_frag,
                  [mu_abs, f_plus, f_minus]   (TS atomic |dipole| and Fukui; only if Q_report ext_used)
  pos    [n, 3]   TS Cartesian coordinates (A)
  edges  [E, 2]   i < j; etype [E]: 1 covalent (rel1 / rel2 bond_matrix bonds in TS numbering), 2 formed (formed_pairs_ts)
  glob   [182]    X182_d1 row (rev6_common.our_columns order)
Numeric standardisation is NOT done here (train.py, per fit, training reactions only; T-3).
Gate T-a: tokens = atoms, no NaN / inf, Q-a / Q-b failures excluded. Idempotent (existing files are re-verified).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fu2_common as F  # noqa: E402

C = F.C
TOKEN_VERSION = "fu2-tokens-2"               # -2: built from fu2-atoms-2 (symmetry-corrected maps)
BIN_COLS = ["frag_B", "dip_a", "dip_mid", "dip_b", "dph_a", "dph_b"]
NUM_BASE = ["bond_dist", "q_rel", "q_frag", "q_TS", "dq_dist", "dq_int", "v_rel", "v_frag", "v_TS", "dv_dist", "dv_int"]
NUM_EXT = ["mu_abs", "f_plus", "f_minus"]
NUM_STANDARDISE = None        # all num columns are standardised in train.py (bond_dist included); bin / one-hot are not


def graph_dist(n, bonds, sources, cap):
    adj = [[] for _ in range(n)]
    for i, j in bonds:
        adj[i].append(j); adj[j].append(i)
    d = np.full(n, cap, dtype=float)
    dq = deque()
    for s in sources:
        d[s] = 0; dq.append(s)
    while dq:
        u = dq.popleft()
        for v in adj[u]:
            if d[v] > d[u] + 1:
                d[v] = d[u] + 1; dq.append(v)
    return np.minimum(d, cap)


def build_one(a, glob, ext_used):
    n = int(a["n"])
    react = [int(x) for x in a["react"]]
    bonds = [tuple(map(int, b)) for b in a["bonds"]]
    formed = [tuple(sorted(map(int, p))) for p in a["formed"]]
    binm = np.zeros((n, len(BIN_COLS)), dtype=np.int8)
    binm[:, 0] = a["frag"]
    for k, i in enumerate(react):
        binm[i, 1 + k] = 1
    cols = dict(bond_dist=graph_dist(n, bonds, react, F.BOND_CAP),
                q_rel=a["q_rel"], q_frag=a["q_frag"], q_TS=a["q_TS"],
                dq_dist=a["q_frag"] - a["q_rel"], dq_int=a["q_TS"] - a["q_frag"],
                v_rel=a["v_rel"], v_frag=a["v_frag"], v_TS=a["v_TS"],
                dv_dist=a["v_frag"] - a["v_rel"], dv_int=a["v_TS"] - a["v_frag"])
    if ext_used:
        cols.update(mu_abs=a["mu_abs"], f_plus=a["fk"][:, 0], f_minus=a["fk"][:, 1])
    names = NUM_BASE + (NUM_EXT if ext_used else [])
    num = np.stack([np.asarray(cols[c], dtype=float) for c in names], 1)
    if set(bonds) & set(formed):
        raise ValueError("formed pair also a reactant covalent bond")
    edges = np.array(sorted(bonds) + sorted(formed), dtype=np.int64).reshape(-1, 2)
    etype = np.array([1] * len(bonds) + [2] * len(formed), dtype=np.int8)
    return dict(rxn_id=int(a["rxn_id"]), Z=np.asarray(a["ts_Z"], dtype=np.int64), bin=binm, num=num,
                pos=np.asarray(a["ts_pos"], dtype=float), edges=edges, etype=etype, glob=np.asarray(glob, dtype=float),
                n_bonds_rel=np.array([int(a["n_bonds_rel"][0])]), token_version=np.array([TOKEN_VERSION])), names


def save_npz(path, arrs):
    F.assert_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.", suffix=".npz")
    os.close(fd)
    try:
        np.savez(tmp, **arrs)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def cmd_build():
    qrep = F.read_json(F.FU2_RES / "Q_report.json")
    if qrep.get("stop"):
        print(f"STOP: Q_report.json stop = {qrep.get('stop_reason')}", file=sys.stderr); return 3
    man = pd.read_csv(F.FU2_RES / "Q_manifest.csv")
    if F.sha256(F.FU2_RES / "Q_manifest.csv") != qrep["manifest_sha256"]:
        print("STOP: Q_manifest.csv changed after Q_report.json", file=sys.stderr); return 3
    use = man[man.use].rxn_id.astype(int).tolist()
    ext_used = bool(qrep["ext_used"])
    X = pd.read_parquet(F.X182).set_index("rxn_id")
    gcols = [c for c in X.columns]
    assert len(gcols) == 182, len(gcols)
    rows, elems, bad, names = [], set(), [], None
    for rid in use:
        apath = F.ATOMS / f"{rid}.npz"
        if F.sha256(apath) != man.loc[man.rxn_id == rid, "sha256"].iloc[0]:
            print(f"STOP: {apath} sha256 differs from Q_manifest", file=sys.stderr); return 3
        a = dict(np.load(apath))
        t, names = build_one(a, X.loc[rid, gcols].to_numpy(float), ext_used)
        n_ok = len(t["Z"]) == int(a["n"]) == len(t["num"]) == len(t["bin"]) == len(t["pos"])
        fin = bool(np.all(np.isfinite(t["num"])) and np.all(np.isfinite(t["glob"])) and np.all(np.isfinite(t["pos"])))
        n_cov, n_form = int((t["etype"] == 1).sum()), int((t["etype"] == 2).sum())
        rec = dict(rxn_id=rid, n_atoms=int(a["n"]), n_ok=bool(n_ok), finite=fin, n_cov=n_cov, n_formed=n_form,
                   n_bonds_rel=int(a["n_bonds_rel"][0]))
        if not (n_ok and fin and n_form == 2):
            bad.append(rec)
        out = F.TOKENS / f"{rid}.npz"
        save_npz(out, t)
        rec["sha256"] = F.sha256(out)
        rows.append(rec)
        elems |= set(int(z) for z in t["Z"])
    Tm = pd.DataFrame(rows)
    F.write_atomic(F.FU2_RES / "T_manifest.csv", lambda p: Tm.to_csv(p, index=False))
    excl = man.loc[~man.use, "rxn_id"].astype(int).tolist()
    # T-a: pairs with a or b lacking atom data (all pairs, and per split / fold train / test as M-1 will see them)
    have = set(Tm.rxn_id.astype(int))
    P = pd.read_parquet(F.SPLITS / "pairs.parquet").set_index("pair_id")
    miss = ~(P.a_rxn.isin(have) & P.b_rxn.isin(have))
    pair_missing = dict(all=int(miss.sum()), n_pairs=len(P), by_ref_kind=miss.groupby(P.ref_kind).sum().astype(int).to_dict())
    srep = F.read_json(F.FU2_RES / "S_splits.json")
    for name in srep["units"]:
        rec = F.read_json(F.SPLITS / name)
        pair_missing[name.replace(".json", "")] = {k: int(miss.loc[rec[k]].sum()) for k in ("train", "test", "inner_train",
                                                                                            "inner_val")}
    rep = dict(token_version=TOKEN_VERSION, n_reactions=len(rows), elements=sorted(elems), bin_cols=BIN_COLS,
               num_cols=names, ext_used=ext_used, glob_cols=gcols, bond_cap=F.BOND_CAP,
               n_atoms=dict(min=int(Tm.n_atoms.min()), median=float(Tm.n_atoms.median()), max=int(Tm.n_atoms.max())),
               n_cov_edges=dict(min=int(Tm.n_cov.min()), max=int(Tm.n_cov.max())),
               cov_edges_equal_rel_bonds=bool((Tm.n_cov == Tm.n_bonds_rel).all()),
               excluded_from_Q=excl, n_excluded_from_Q=len(excl), pairs_missing_atoms=pair_missing,
               splits_sha256=F.sha256(F.FU2_RES / "S_splits.json"),
               gate_Ta=dict(tokens_eq_atoms=bool(Tm.n_ok.all()), all_finite=bool(Tm.finite.all()),
                            two_formed_edges=bool((Tm.n_formed == 2).all()),
                            q_excluded_absent=not bool(set(excl) & set(Tm.rxn_id))),
               bad=bad, manifest_sha256=F.sha256(F.FU2_RES / "T_manifest.csv"),
               q_report_sha256=F.sha256(F.FU2_RES / "Q_report.json"),
               x182_sha256=F.sha256(F.X182))
    rep["stop"] = not all(rep["gate_Ta"].values())
    F.write_json(F.FU2_RES / "T_report.json", rep)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("glob_cols", "excluded_from_Q")}, indent=1, default=str))
    return 3 if rep["stop"] else 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["build"]:
        sys.exit(cmd_build())
    print(__doc__); sys.exit(2)
