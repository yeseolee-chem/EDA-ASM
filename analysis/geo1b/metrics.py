"""GEO1b metrics (docs/specs/GEO1B.md 4-1). Bond identity across a pair = rank by LABEL length (short, long), as GEO1.
Primary (inner validation and test) = share of ALL para P_main reference pairs with pair error <= 0.005 A. No pair
class, no G1-derived flag. map_rmsd_g2 (records) only labels descriptive bins."""
from __future__ import annotations

import numpy as np
import pandas as pd

import geo_common as GC

_FP = None
BINS = (0.25, 0.75)                                       # mapped RMSD (G2 raw vs label, A): low <= 0.25 < mid <= 0.75 < high


def f_pairs():
    global _FP
    if _FP is None:
        F = pd.read_csv(GC.F_PAIRS, usecols=["pair_id", "ref_kind", "a_rxn", "b_rxn", "pos", "in_P_main"])
        _FP = F[(F.ref_kind == "reference") & F.in_P_main.astype(bool)].copy()
    return _FP


def rxn_errors(pred, lab, pairs):
    dl = np.array([np.linalg.norm(lab[i] - lab[j]) for i, j in pairs])
    dp = np.array([np.linalg.norm(pred[i] - pred[j]) for i, j in pairs])
    o = np.argsort(dl)
    return dp[o] - dl[o]                                  # (e_short, e_long)


def rmsd_bin(v):
    if v is None or not np.isfinite(v):
        return "na"
    return "low" if v <= BINS[0] else ("mid" if v <= BINS[1] else "high")


def pair_bin(a, b):
    """the worse of the two reactions' bins (na if either is na)."""
    order = dict(low=0, mid=1, high=2)
    if "na" in (a, b):
        return "na"
    return max((a, b), key=lambda s: order[s])


def evaluate_preds(P, recs, with_rmsd=True):
    """P: uid -> predicted xyz (any frame); recs: records of the evaluated reactions (D1 only)."""
    R = {r["rxn_id"]: r for r in recs if r["src"] == "D1" and r["uid"] in P}
    E, rows = {}, []
    for rid, r in R.items():
        e = rxn_errors(P[r["uid"]], r["lab"], r["pairs"])
        E[rid] = e
        row = dict(rxn_id=rid, core=r["core"], e_short=float(e[0]), e_long=float(e[1]), abs_max=float(np.abs(e).max()),
                   map_rmsd_g2=float(r.get("map_rmsd_g2", np.nan)), bin=rmsd_bin(r.get("map_rmsd_g2")))
        if with_rmsd:
            heavy = r["types"] != 0
            row["rmsd_heavy"] = GC.kabsch_align(P[r["uid"]][heavy], r["lab"][heavy], reflect=True)[1]
        rows.append(row)
    F = f_pairs()
    F = F[F.a_rxn.isin(E) & F.b_rxn.isin(E)].copy()
    F["pe"] = [float(np.max(np.abs(E[a] - E[b]))) for a, b in zip(F.a_rxn, F.b_rxn)]
    F["core"] = [R[a]["core"] for a in F.a_rxn]
    F["dipole"] = [R[a]["core"].split("/")[0] for a in F.a_rxn]
    F["bin"] = [pair_bin(rmsd_bin(R[a].get("map_rmsd_g2")), rmsd_bin(R[b].get("map_rmsd_g2")))
                for a, b in zip(F.a_rxn, F.b_rxn)]
    prim = F[F.pos == "p"]
    D = pd.DataFrame(rows)
    if not with_rmsd:
        D["rmsd_heavy"] = np.nan
    nan = float("nan")
    return dict(primary_share=float((prim.pe <= 0.005).mean()) if len(prim) else nan, n_primary=int(len(prim)),
                all_pairs_share=float((F.pe <= 0.005).mean()) if len(F) else nan, n_pairs=int(len(F)),
                pair_pe_median_para=float(prim.pe.median()) if len(prim) else nan,
                rxn_form_abs_median=float(np.median(np.abs(np.concatenate([D.e_short, D.e_long])))) if len(D) else nan,
                rmsd_heavy_median=float(D.rmsd_heavy.median()) if len(D) else nan, table_pairs=F, table_rxn=D)
