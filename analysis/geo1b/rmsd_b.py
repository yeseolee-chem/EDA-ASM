#!/usr/bin/env python3
"""GEO1b Phase E2 (D1 run env, CPU): symmetry-aware heavy-atom RMSD and sanity of the test predictions, report only
(docs/specs/GEO1B.md 4-2).   python rmsd_b.py <n_proc>
Reads $GEO1B/data/records.pkl and $GEO1B/preds/final_f{k}_s{0,1,2}.pkl (3-seed mean = the evaluated model).
Writes results/E_rmsd.csv (one row per test reaction) and results/E_rmsd.json (summary).
  map_g2    = rmsd.mapped_heavy_rmsd(G2 raw, label)      (= records map_rmsd_g2; recomputed as a check)
  map_model = rmsd.mapped_heavy_rmsd(model, label)
  map_move  = rmsd.mapped_heavy_rmsd(model, G2 raw)      (> 0.30 A = autodE's distinct-conformer threshold)
  sanity    = the GEO2 build_geo rules (close contact, bond graph change vs G2, forming distance 1.4-4.5 A)."""
from __future__ import annotations

import json
import math
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geo_common as GC  # noqa: E402
from metrics import rmsd_bin  # noqa: E402

sys.path.insert(0, str(GC.RMSD_DIR))
import rmsd as rm  # noqa: E402

MIN_CONTACT, BOND_IN, BOND_OUT, FORM = 0.60, 1.10, 1.40, (1.4, 4.5)   # build_geo rules; 4.5 A = GEO1 decision I-1
MOVE_A = 0.30


def mapped(a, b):
    r = float(rm.mapped_heavy_rmsd(a, b))
    return r if math.isfinite(r) else float("nan")


def sanity(syms, X, Xg2, pairs):
    r = np.array([GC.COV[s] for s in syms])
    S = r[:, None] + r[None, :]
    D, Dg = GC.dmat(X), GC.dmat(Xg2)
    iu = np.triu_indices(len(syms), 1)
    if (D / S)[iu].min() < MIN_CONTACT:
        return "close_contact"
    fp = {tuple(sorted(p)) for p in pairs}
    mask = np.array([(i, j) not in fp for i, j in zip(*iu)])
    rg, rx = (Dg / S)[iu][mask], (D / S)[iu][mask]
    if ((rg < BOND_IN) & (rx > BOND_OUT)).any() or ((rg > BOND_OUT) & (rx < BOND_IN)).any():
        return "bond_graph_changed"
    d = GC.form_d(X, [tuple(p) for p in pairs])
    if min(d) < FORM[0] or max(d) > FORM[1]:
        return "forming_distance_range"
    return ""


def one(a):
    uid, rid, core, syms, lab, g2, mod, pairs, map_rec = a
    L, G, M = (syms, lab), (syms, g2), (syms, mod)
    return dict(uid=uid, rxn_id=rid, core=core, map_g2_rec=map_rec, map_g2=mapped(G, L), map_model=mapped(M, L),
                map_move=mapped(M, G), sanity=sanity(syms, mod, g2, pairs))


def main(n_proc):
    recs = GC.load_records()
    test = [r for r in recs if r["src"] == "D1" and r["fold"] in range(5) and r["xyz_g2"] is not None]
    P = {}
    for k in range(5):
        parts = []
        for s in (0, 1, 2):
            with open(GC.GEO / "preds" / f"final_f{k}_s{s}.pkl", "rb") as f:
                parts.append(pickle.load(f))
        P.update({u: np.mean([np.asarray(q[u], float) for q in parts], axis=0) for u in parts[0]})
    jobs = [(r["uid"], r["rxn_id"], r["core"], [GC.ELEM[t] for t in r["types"]], r["lab"], r["xyz_g2"],
             P[r["uid"]], r["pairs"].tolist(), r["map_rmsd_g2"]) for r in test]
    with Pool(int(n_proc)) as pool:
        rows = pool.map(one, jobs, chunksize=8)
    D = pd.DataFrame(rows)
    D["bin"] = [rmsd_bin(v) for v in D.map_g2]
    GC.RES.mkdir(parents=True, exist_ok=True)
    D.to_csv(GC.RES / "E_rmsd.csv", index=False)

    def summ(g):
        f = lambda c: g[c].dropna()                                   # noqa: E731
        return dict(n=int(len(g)), median_map_g2=float(f("map_g2").median()), median_map_model=float(f("map_model").median()),
                    share_le_025=dict(g2=float((f("map_g2") <= 0.25).mean()), model=float((f("map_model") <= 0.25).mean())),
                    share_le_050=dict(g2=float((f("map_g2") <= 0.5).mean()), model=float((f("map_model") <= 0.5).mean())),
                    share_moved_gt_030=float((f("map_move") > MOVE_A).mean()),
                    n_nan=dict(g2=int(g.map_g2.isna().sum()), model=int(g.map_model.isna().sum())))
    dev = (D.map_g2 - D.map_g2_rec).abs().max()
    out = dict(all=summ(D), by_bin={b: summ(g) for b, g in D.groupby("bin")},
               sanity_fail_share=float((D.sanity != "").mean()), sanity_reasons=D.sanity[D.sanity != ""].value_counts().to_dict(),
               map_g2_vs_records_max_dev=float(dev) if np.isfinite(dev) else None, move_threshold_A=MOVE_A)
    GC.write_json(GC.RES / "E_rmsd.json", out)
    print(json.dumps(out, indent=1, default=float), flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    sys.exit(main(sys.argv[1]))
