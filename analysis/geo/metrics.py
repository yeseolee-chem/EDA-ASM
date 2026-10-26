"""GEO-1 metrics. Bond identity across a pair = rank by LABEL length (short, long), fixed before results."""
from __future__ import annotations

import numpy as np
import pandas as pd

import geo_common as GC

_FP = None


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


def pair_class(sa, sb):
    if sa is True and sb is True:
        return "same"
    if sa is False and sb is False:
        return "different"
    return "mixed"                                        # one True / one False, or a None


def evaluate_preds(P, recs, with_rmsd=True):
    """P: uid -> predicted xyz (any frame); recs: records of the evaluated reactions (D1 only).
    Primary metric (spec 12): share of 'same'-class para P_main reference pairs with pair error <= 0.005 A, where
    pair error = max over (short, long) of |e_a - e_b| and e = predicted - label forming distance."""
    R = {r["rxn_id"]: r for r in recs if r["src"] == "D1" and r["uid"] in P}
    E, rows = {}, []
    for rid, r in R.items():
        e = rxn_errors(P[r["uid"]], r["lab"], r["pairs"])
        E[rid] = e
        row = dict(rxn_id=rid, core=r["core"], e_short=float(e[0]), e_long=float(e[1]),
                   abs_max=float(np.abs(e).max()), same=r["same"])
        if with_rmsd:
            heavy = r["types"] != 0
            row["rmsd_heavy"] = GC.kabsch_align(P[r["uid"]][heavy], r["lab"][heavy], reflect=True)[1]
        rows.append(row)
    F = f_pairs()
    F = F[F.a_rxn.isin(E) & F.b_rxn.isin(E)].copy()
    F["pe"] = [float(np.max(np.abs(E[a] - E[b]))) for a, b in zip(F.a_rxn, F.b_rxn)]
    F["cls"] = [pair_class(R[a]["same"], R[b]["same"]) for a, b in zip(F.a_rxn, F.b_rxn)]
    F["same"] = F.cls == "same"
    F["core"] = [R[a]["core"] for a in F.a_rxn]
    F["dipole"] = [R[a]["core"].split("/")[0] for a in F.a_rxn]
    prim = F[(F.pos == "p") & F.same]
    D = pd.DataFrame(rows)
    if not with_rmsd:
        D["rmsd_heavy"] = np.nan
    return dict(primary_share=float((prim.pe <= 0.005).mean()) if len(prim) else float("nan"), n_primary=int(len(prim)),
                all_pairs_share=float((F.pe <= 0.005).mean()) if len(F) else float("nan"), n_pairs=int(len(F)),
                pair_pe_median_same_para=float(prim.pe.median()) if len(prim) else float("nan"),
                rxn_form_abs_median=float(np.median(np.abs(np.concatenate([D.e_short, D.e_long])))),
                rmsd_heavy_median=float(D.rmsd_heavy.median()), table_pairs=F, table_rxn=D)
