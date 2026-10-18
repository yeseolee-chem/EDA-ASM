"""d1_anchor_pairs.py — extra D1 pairs for families that have no usable reference (training data for g).

Why: d1_pairs.py pairs every substituted row with the X = Ph (aryl) or X = H (direct) reference of its family.
Rows whose family has no such reference are left out (2026-10-19: 1,011 aryl + 297 direct rows). A family of n rows
holds n - 1 independent differences, so one anchor row per family keeps all of them.

family key F = (dipole, skeleton, series, cfg, orient, face, c3), separately for aryl rows (pos p / m / o) and direct rows
  aryl families:   F has no row with pos ref_aryl at all (families whose reference is duplicated are stereo-mixed -> skipped)
  direct families: direct rows with no H candidate in d1_pairs.py (same dipole, skeleton, face) -> grouped by F
rows duplicated within F + Xb + pos are dropped (keep=False), as in d1_pairs.py
anchor b = first row after sorting by: passes the rev 6 hygiene rule (d1 >= 0, d2 >= 0, d2 <= 50) first,
  then |sigma_p(Xb)| (missing sigma last), pos p < m < o < direct, Xb; a = every other row
families with one row give no pair
output columns = pairs_d1.csv columns + ref_kind ("anchor"); pair_id A00000...; the Hammett analysis never uses these rows
usage: python -I d1_anchor_pairs.py rowattr.pkl sigma.csv pairs_d1.csv out_anchor.csv"""
import sys, os, numpy as np, pandas as pd

TARGETS = {"d1": "d1_kcal", "d2": "d2_kcal", "elst": "elst_dft", "pauli": "pauli_dft", "oi": "oi_dft",
           "disp": "disp_dft", "cpcm": "cpcm_dft", "cds": "cds_dft", "c_ghost": "bsse_shift_kcal",
           "barrier": "barrier_kcal"}
F = ["dipole", "skeleton", "series", "cfg", "orient", "face", "c3"]
POS_ORDER = {"p": 0, "m": 1, "o": 2, "direct": 3}


def build(r, sg):
    r = r.copy()
    for c in F:
        r[c] = r[c].fillna("").astype(str)
    refk = set(map(tuple, r.loc[r.pos == "ref_aryl", F].to_numpy()))
    ar = r[r.pos.isin(["p", "m", "o"])]
    ar = ar[[tuple(t) not in refk for t in ar[F].to_numpy()]]
    H = r[r.pos == "ref_direct"]
    dr = r[r.pos == "direct"]
    noh = [len(H[(H.dipole == x.dipole) & (H.skeleton == x.skeleton) & (H.face == x.face)]) == 0 for _, x in dr.iterrows()]
    dr = dr[noh]
    rows = []
    for kind, q in (("aryl", ar), ("direct", dr)):
        q = q[~q.duplicated(F + ["Xb", "pos"], keep=False)].copy()
        q["_s"] = q.Xb.map(sg.sigma_p).abs().fillna(np.inf)
        q["_p"] = q.pos.map(POS_ORDER)
        q["_h"] = (~((q.d1_kcal >= 0) & (q.d2_kcal >= 0) & (q.d2_kcal <= 50))).astype(int)
        for _, g in q.groupby(F, sort=True):
            if len(g) < 2:
                continue
            g = g.sort_values(["_h", "_s", "_p", "Xb"])
            b = g.iloc[0]
            for _, a in g.iloc[1:].iterrows():
                d = {"a_job": a.job_id, "a_rxn": a.rxn_id, "Xb": a.Xb, "pos": a.pos, **{c: a[c] for c in F},
                     "b_job": b.job_id, "b_rxn": b.rxn_id, "anchor_Xb": b.Xb, "anchor_pos": b.pos}
                for t, col in TARGETS.items():
                    d[f"dd_{t}"] = a[col] - b[col]
                rows.append(d)
    A = pd.DataFrame(rows)
    A["fam"] = A[F].agg("/".join, axis=1) + "/anchor"
    A["core"] = A.dipole + "/" + A.skeleton
    A["sigma"] = np.where(A.pos == "m", A.Xb.map(sg.sigma_m), A.Xb.map(sg.sigma_p))
    A["ref_kind"] = "anchor"
    A.insert(0, "pair_id", [f"A{i:05d}" for i in range(len(A))])
    return A


if __name__ == "__main__":
    r = pd.read_pickle(sys.argv[1]); sg = pd.read_csv(sys.argv[2]).set_index("X"); P = pd.read_csv(sys.argv[3])
    A = build(r, sg)
    assert not (set(A.a_job) | set(A.b_job)) & (set(P.a_job) | set(P.b_job)), "anchor rows overlap the reference pairs"
    tmp = sys.argv[4] + ".tmp"; A.to_csv(tmp, index=False); os.replace(tmp, sys.argv[4])
    print("anchor pairs", len(A), A.pos.value_counts().to_dict())
    print("anchor families", A.fam.nunique(), "| rows used", len(set(A.a_job) | set(A.b_job)))
    print("cores", A.core.nunique(), "| cores not in reference pairs:", sorted(set(A.core) - set(P.core)))
    print("pairs by dipole:", A.dipole.value_counts().to_dict())
