"""d1s_prep.py — D1 라벨 로드, QC 표식, regio 방향(RDKit), H 부모 짝 / Ph 기준 아릴 짝 생성.

모든 함수는 순수 함수다(파일을 쓰지 않음). run_all.py가 호출한다.
에너지 단위 kcal/mol, 길이 Å.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

CH = ["d1_kcal", "d2_kcal", "elst_dft", "pauli_dft", "oi_dft", "disp_dft", "cpcm_dft", "cds_dft"]
BUD = CH + ["bsse_shift_kcal"]          # 9 terms; Σ = barrier (closure ≤ 0.11)
CHN = {"d1_kcal": "d1", "d2_kcal": "d2", "elst_dft": "elst", "pauli_dft": "Pauli", "oi_dft": "OI",
       "disp_dft": "disp", "cpcm_dft": "CPCM", "cds_dft": "CDS", "bsse_shift_kcal": "bsse", "barrier_kcal": "ΔE‡"}
VALS = CH + ["bsse_shift_kcal", "barrier_kcal", "eint_spe_kcal", "e_bond_kcal", "zeta", "asyn"]

# QC thresholds (pre-registered; do not change after seeing results)
Q_BADREF_KCAL = -5.0      # d1 or d2 below this -> broken reference structure
Q_MIN_FORMED = 1.65       # shorter forming bond below this -> product-like / stepwise-like
Q_MAX_FORMED = 3.0        # longer forming bond above this -> stepwise-like (= flag_async)


def load_labels(path: str) -> pd.DataFrame:
    d = json.load(open(path))
    rows = []
    for r in d:
        x = {k: v for k, v in r.items() if not isinstance(v, (dict, list))}
        x["gates_ok"] = bool(all((r.get("gates") or {"missing": False}).values()))
        for f, v in (r.get("flags") or {}).items():
            if f != "audit_flags":
                x[f if f.startswith("flag_") else "flag_" + f] = v
        rows.append(x)
    del d
    df = pd.DataFrame(rows)
    df["face"] = df["face"].fillna("")
    df["stereo_index"] = df["stereo_index"].astype(str)
    df["zeta"] = (df.formed_d1 + df.formed_d2) / 2
    df["asyn"] = (df.formed_d1 - df.formed_d2).abs()
    df["side"] = np.where(df.series == "R1", "dipole", "dipolarophile")
    df["R"] = np.where(df.series == "R1", df.R1, df.R2)
    df["X"] = df.R.str.replace(r"^[pmo]-", "", regex=True)
    df["R2p"] = np.where(df.series == "R2", "H", df.R2)
    df["is_parent"] = (df.R1 == "H") & (df.R2.isin(["H", "-"]))
    mf = np.minimum(df.formed_d1, df.formed_d2)
    Mf = np.maximum(df.formed_d1, df.formed_d2)
    df["q_badref"] = (df.d1_kcal < Q_BADREF_KCAL) | (df.d2_kcal < Q_BADREF_KCAL)
    df["q_stepwise"] = (mf < Q_MIN_FORMED) | (Mf > Q_MAX_FORMED)
    df["q_bad"] = df.q_badref | df.q_stepwise
    df["closure"] = df[BUD].sum(axis=1) - df.barrier_kcal
    df["orient"] = ""
    need = df.regio.isin(["r1/2", "r2/2"])
    df.loc[need, "orient"] = [orientation(s, dp) for s, dp in zip(df.loc[need, "rxn_smiles"], df.loc[need, "dipole"])]
    return df


# ---------------------------------------------------------------- regio orientation
def _new_ring(m, dip):
    for ring in m.GetRingInfo().AtomRings():
        if len(ring) != 5:
            continue
        s = set(ring)
        el = [m.GetAtomWithIdx(i).GetSymbol() for i in ring]
        if dip in ("nitrile oxide", "nitrone"):
            for i in ring:
                a = m.GetAtomWithIdx(i)
                if a.GetSymbol() == "O" and any(nb.GetSymbol() == "N" and nb.GetIdx() in s for nb in a.GetNeighbors()):
                    return ring
        elif dip == "azide":
            for i in ring:
                a = m.GetAtomWithIdx(i)
                if a.GetSymbol() == "N" and sum(nb.GetIdx() in s and nb.GetSymbol() == "N" for nb in a.GetNeighbors()) == 2:
                    return ring
        elif dip == "azomethine ylide" and "N" in el:
            return ring
        elif dip == "carbonyl ylide" and "O" in el:
            return ring
    return None


def _partition(m, ring, dip):
    s = set(ring)
    A = m.GetAtomWithIdx
    rnb = lambda i: [nb.GetIdx() for nb in A(i).GetNeighbors() if nb.GetIdx() in s]
    if dip in ("nitrile oxide", "nitrone"):
        O = [i for i in ring if A(i).GetSymbol() == "O" and any(A(j).GetSymbol() == "N" for j in rnb(i))][0]
        N = [j for j in rnb(O) if A(j).GetSymbol() == "N"][0]
        Cb = [j for j in rnb(N) if j != O][0]
        return [O, N, Cb], O
    if dip == "azide":
        mid = [i for i in ring if A(i).GetSymbol() == "N" and sum(A(j).GetSymbol() == "N" for j in rnb(i)) == 2][0]
        t = rnb(mid)
        sub = [i for i in t if A(i).GetTotalNumHs() > 0 or any(nb.GetIdx() not in s for nb in A(i).GetNeighbors())]
        return [mid] + t, (sub[0] if len(sub) == 1 else None)
    c = [i for i in ring if A(i).GetSymbol() == ("N" if dip == "azomethine ylide" else "O")][0]
    t = rnb(c)
    hv = [i for i in t if any(nb.GetIdx() not in s and nb.GetAtomicNum() > 1 for nb in A(i).GetNeighbors())]
    return [c] + t, (hv[0] if len(hv) == 1 else None)


def orientation(rxn_smiles: str, dip: str) -> str:
    """'a' = dipolarophile ring atom nearest the anchor is bonded to terminus a
    (O for nitrile oxide/nitrone; substituted N1 for azide; R1-bearing C for ylides)."""
    try:
        m = Chem.MolFromSmiles(rxn_smiles.split(">>")[1])
        ring = _new_ring(m, dip)
        if ring is None:
            return "noring"
        dset, a = _partition(m, ring, dip)
        if a is None:
            return "sym"
        dpo = [i for i in ring if i not in dset]
        dm = Chem.GetDistanceMatrix(m)
        ds = set(dset)
        cands = []
        for at in m.GetAtoms():
            i = at.GetIdx()
            if i in ds:
                continue
            for bd in at.GetBonds():
                o = bd.GetOtherAtom(at)
                if not bd.IsInRing() and o.GetAtomicNum() > 1 and o.GetIdx() not in ds:
                    cands.append(i)
                    break
        if not cands:
            return "noanchor"
        best = min(cands, key=lambda c: min(dm[c, dpo[0]], dm[c, dpo[1]]))
        d0, d1 = dm[best, dpo[0]], dm[best, dpo[1]]
        if d0 == d1:
            return "tie"
        prox = dpo[0] if d0 < d1 else dpo[1]
        return "a" if m.GetBondBetweenAtoms(prox, a) is not None else "b"
    except Exception as e:  # noqa: BLE001
        return "err:" + type(e).__name__


# ---------------------------------------------------------------- pairs
def pairs_vs_H(df: pd.DataFrame) -> pd.DataFrame:
    """child − H parent (same core, same fixed R2, same face; orientation if parent has two)."""
    par = df[df.is_parent]
    pg = {k: v for k, v in par.groupby(["core", "R2p", "face"])}
    out = []
    for _, r in df[~df.is_parent].iterrows():
        p = pg.get((r.core, r.R2p, r.face))
        if p is None:
            continue
        if len(p) > 1 and r.orient in ("a", "b") and r.orient in set(p.orient):
            p = p[p.orient == r.orient]
        if len(p) != 1:
            continue
        p = p.iloc[0]
        x = {c: r[c] - p[c] for c in VALS}
        x.update(job_id=r.job_id, parent_job=p.job_id, core=r.core, dipole=r.dipole, skeleton=r.skeleton,
                 side=r.side, R=r.R, X=r.X, attach=r.attach, regio=r.regio, orient=r.orient, face=r.face,
                 stereo_index=r.stereo_index, cell_id=r.cell_id,
                 q_bad=bool(r.q_bad or p.q_bad))
        out.append(x)
    P = pd.DataFrame(out)
    P["ctx"] = P.core + "|" + P.orient + "|" + P.face + "|" + P.stereo_index
    return P


def pairs_vs_Ph(df: pd.DataFrame) -> pd.DataFrame:
    """aryl (p-/m-/o-X-C6H4) − Ph in the same core/series/face/orientation/stereo slot (clean rows only)."""
    d = df[~df.q_bad]
    key = ["core", "series", "R2p", "face", "orient", "stereo_index"]
    ph = {k: v for k, v in d[d.R == "Ph"].groupby(key)}
    out = []
    for _, r in d[d.attach.isin(["p", "m", "o"])].iterrows():
        k = tuple(r[c] for c in key)
        p = ph.get(k)
        if p is None or len(p) != 1:
            continue
        p = p.iloc[0]
        x = {c: r[c] - p[c] for c in VALS}
        x.update(job_id=r.job_id, ref_job=p.job_id, core=r.core, dipole=r.dipole, skeleton=r.skeleton,
                 side=r.side, X=r.X, pos=r.attach, ctx="|".join(map(str, k)))
        out.append(x)
    return pd.DataFrame(out)


def sibling_pairs(df: pd.DataFrame, key, col, A, B) -> pd.DataFrame:
    """A − B inside groups that hold exactly one clean A row and one clean B row."""
    d = df[~df.q_bad]
    out = []
    for _, g in d.groupby(key):
        a, b = g[g[col] == A], g[g[col] == B]
        if len(a) == 1 and len(b) == 1:
            a, b = a.iloc[0], b.iloc[0]
            x = {c: a[c] - b[c] for c in VALS}
            x.update(job_a=a.job_id, job_b=b.job_id, core=a.core, dipole=a.dipole, skeleton=a.skeleton,
                     series=a.series, R=a.R, X=a.X, attach=a.attach)
            out.append(x)
    return pd.DataFrame(out)


def budget(D: pd.DataFrame, target: str = "barrier_kcal") -> dict:
    v = D[target].var()
    return {CHN[c]: D[c].cov(D[target]) / v for c in BUD}
