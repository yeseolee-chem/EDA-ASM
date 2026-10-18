"""d1_pairs.py — D1 matched pairs for the pair-difference model.

pair p = (a, b): a = substituted reaction, b = its reference in the same reaction family.
  family key F = (dipole, skeleton, series, cfg, orient, face, c3)   (columns of rowattr.pkl, build_rowattr.py)
  aryl X (pos p / m / o): b = the unsubstituted-phenyl row (pos ref_aryl) with the same F; F must be unique
  direct X (pos direct):  b = the H row (pos ref_direct) with the same dipole, skeleton, face, narrowed by orient,
                          cfg, c3 only when every candidate H row has that field; exactly one candidate required
  rows / references that are duplicated within their key are dropped (keep=False)
dd_<t> = y_a - y_b in kcal/mol for every target t (raw energies, NOT log units; Hammett value = -dd / 1.3642)
core = (dipole, skeleton) -> outer CV group: every pair whose a or b belongs to a core is in that core's fold
usage: python -I d1_pairs.py rowattr.pkl sigma.csv out_pairs.csv"""
import sys, numpy as np, pandas as pd

TARGETS = {"d1": "d1_kcal", "d2": "d2_kcal", "elst": "elst_dft", "pauli": "pauli_dft", "oi": "oi_dft",
           "disp": "disp_dft", "cpcm": "cpcm_dft", "cds": "cds_dft", "c_ghost": "bsse_shift_kcal",
           "barrier": "barrier_kcal"}
F = ["dipole", "skeleton", "series", "cfg", "orient", "face", "c3"]


def build(r, sg):
    r = r.copy()
    for c in F:
        r[c] = r[c].fillna("").astype(str)
    keep = ["job_id", "rxn_id", "Xb", "pos"] + F + list(TARGETS.values())
    ref = r[r.pos == "ref_aryl"]
    ref = ref[~ref.duplicated(F, keep=False)].set_index(F)
    out = []
    for pos in ("p", "m", "o"):
        q = r[r.pos == pos]
        q = q[~q.duplicated(F + ["Xb"], keep=False)]
        j = q[keep].join(ref[["job_id", "rxn_id"] + list(TARGETS.values())].add_suffix("_b"), on=F, how="inner")
        out.append(j)
    H = r[r.pos == "ref_direct"]
    rows = []
    for _, x in r[r.pos == "direct"].iterrows():
        c = H[(H.dipole == x.dipole) & (H.skeleton == x.skeleton) & (H.face == x.face)]
        for col in ("orient", "cfg", "c3"):
            if len(c) > 1 and (c[col] != "").all():
                c = c[c[col] == x[col]]
        if len(c) == 1:
            h = c.iloc[0]
            rows.append({**{k: x[k] for k in keep}, **{f"{v}_b": h[v] for v in ["job_id", "rxn_id"] + list(TARGETS.values())}})
    out.append(pd.DataFrame(rows))
    P = pd.concat(out, ignore_index=True)
    P = P.rename(columns={"job_id": "a_job", "rxn_id": "a_rxn", "job_id_b": "b_job", "rxn_id_b": "b_rxn"})
    for t, col in TARGETS.items():
        P[f"dd_{t}"] = P[col] - P[col + "_b"]
    P["fam"] = P[F].agg("/".join, axis=1)
    P["core"] = P.dipole + "/" + P.skeleton
    P["sigma"] = np.where(P.pos == "m", P.Xb.map(sg.sigma_m), P.Xb.map(sg.sigma_p))
    P = P.drop(columns=[c for c in P.columns if c.endswith("_b") and c not in ("b_job", "b_rxn")] + list(TARGETS.values()))
    P.insert(0, "pair_id", [f"P{i:05d}" for i in range(len(P))])
    return P


if __name__ == "__main__":
    r = pd.read_pickle(sys.argv[1]); sg = pd.read_csv(sys.argv[2]).set_index("X")
    P = build(r, sg)
    tmp = sys.argv[3] + ".tmp"; P.to_csv(tmp, index=False)
    import os; os.replace(tmp, sys.argv[3])
    print("pairs", len(P), P.pos.value_counts().to_dict())
    print("families", P.fam.nunique(), "| para families with >= 12 X having sigma_p:",
          int((P[(P.pos == "p") & P.sigma.notna()].groupby("fam").Xb.nunique() >= 11).sum()))
    print("cores", P.core.nunique(), "| pairs per core min/median/max:", P.groupby("core").size().agg(["min", "median", "max"]).to_dict())
    print("pairs by dipole:", P.dipole.value_counts().to_dict())
    print("distinct a rows", P.a_job.nunique(), "| distinct references", P.b_job.nunique())
