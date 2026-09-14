"""mmp_utils.py — dipole/dipolarophile 분리, 단일 절단 MMP 쌍, 치환기 집합, 분할 생성기.

사용:
    from mmp_utils import annotate, build_pairs, splits
    feat = annotate(feat, full_csv)          # dipole, dph, regio, dclass, subs_dip 컬럼 추가
    pairs = build_pairs(feat)                # r1, r2, kind, sub_from, sub_to
    for name, folds in splits(feat, pairs): ...
"""
from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from sklearn.model_selection import GroupKFold, KFold

RDLogger.DisableLog("rdApp.*")
MAX_SUB_HEAVY = 8


def _canon(m):
    m = Chem.Mol(m)
    for a in m.GetAtoms():
        a.SetAtomMapNum(0)
    return Chem.MolToSmiles(m)


def annotate(feat: pd.DataFrame, full_csv: str) -> pd.DataFrame:
    """rxn_smiles(원자 매핑)에서 dipole/dph SMILES, regio 서명, dipole 클래스, 치환기 집합을 만든다."""
    full = pd.read_csv(full_csv).set_index("rxn_id")
    dip, dph, regio, dclass = {}, {}, {}, {}
    for rid in feat.rxn_id:
        smi = full.loc[rid, "rxn_smiles"]
        rmols = [Chem.MolFromSmiles(x) for x in smi.split(">>")[0].split(".")]
        pmol = Chem.MolFromSmiles(smi.split(">>")[1])

        def bonds(m):
            return {tuple(sorted((b.GetBeginAtom().GetAtomMapNum(), b.GetEndAtom().GetAtomMapNum())))
                    for b in m.GetBonds()}
        formed = bonds(pmol) - set().union(*[bonds(m) for m in rmols])
        owner = {a.GetAtomMapNum(): k for k, m in enumerate(rmols) for a in m.GetAtoms()}
        idx2map = {a.GetIdx(): a.GetAtomMapNum() for a in pmol.GetAtoms()}
        rings = [set(idx2map[i] for i in r) for r in pmol.GetRingInfo().AtomRings()]
        rings = [r for r in rings if all(a in r and b in r for a, b in formed)]
        ring = min(rings, key=len)
        cnt = {}
        for m_ in ring:
            cnt[owner[m_]] = cnt.get(owner[m_], 0) + 1
        i = [k for k, v in cnt.items() if v == 3][0]
        dip[rid], dph[rid] = _canon(rmols[i]), _canon(rmols[1 - i])
        # regio 서명: 형성 결합 양단 원소의 (소유 분자, 원소, 차수)
        sig = []
        for a, b in formed:
            ea = [x for x in pmol.GetAtoms() if x.GetAtomMapNum() == a][0]
            eb = [x for x in pmol.GetAtoms() if x.GetAtomMapNum() == b][0]
            sig.append(tuple(sorted([(owner[a], ea.GetSymbol(), ea.GetDegree()),
                                     (owner[b], eb.GetSymbol(), eb.GetDegree())])))
        regio[rid] = str(sorted(sig))
        # dipole 클래스: 고리에 들어가는 3원자 원소 + 말단(*) 표시
        term = {x for pair in formed for x in pair}
        lab = [a.GetSymbol() + ("*" if a.GetAtomMapNum() in term else "")
               for a in rmols[i].GetAtoms() if a.GetAtomMapNum() in ring]
        dclass[rid] = "".join(sorted(lab))
    out = feat.copy()
    out["dipole"] = out.rxn_id.map(dip)
    out["dph"] = out.rxn_id.map(dph)
    out["regio"] = out.rxn_id.map(regio)
    out["dclass"] = out.rxn_id.map(dclass)
    cache = {}
    out["subs_dip"] = [frozenset(s for _, s in single_cuts(x, cache)) for x in out.dipole]
    out["subs_dph"] = [frozenset(s for _, s in single_cuts(x, cache)) for x in out.dph]
    return out


def single_cuts(smi: str, cache: dict | None = None):
    """비고리 단일결합 1개를 끊어 (core, substituent) 쌍을 만든다. 치환기 ≤ MAX_SUB_HEAVY 중원자."""
    if cache is not None and smi in cache:
        return cache[smi]
    m = Chem.MolFromSmiles(smi)
    out = set()
    for b in m.GetBonds():
        if b.IsInRing() or b.GetBondType() != Chem.BondType.SINGLE:
            continue
        fm = Chem.FragmentOnBonds(m, [b.GetIdx()], addDummies=True, dummyLabels=[(0, 0)])
        frags = Chem.GetMolFrags(fm, asMols=True, sanitizeFrags=False)
        if len(frags) != 2:
            continue
        for k in (0, 1):
            sub, core = frags[k], frags[1 - k]
            if sum(1 for a in sub.GetAtoms() if a.GetAtomicNum() > 0) <= MAX_SUB_HEAVY:
                out.add((_canon(core), _canon(sub)))
    if cache is not None:
        cache[smi] = out
    return out


def build_pairs(feat: pd.DataFrame) -> pd.DataFrame:
    """같은 파트너·같은 regio 서명에서, 다른 성분이 단일 절단 MMP인 반응 쌍."""
    cache = {}
    cuts = {s: single_cuts(s, cache) for s in set(feat.dipole) | set(feat.dph)}
    core_index = {}
    for s, cs in cuts.items():
        for core, sub in cs:
            core_index.setdefault(core, {})[s] = sub
    lookup = feat.set_index("rxn_id")
    rows = []
    for other, mine, kind in (("dph", "dipole", "dipole_change"), ("dipole", "dph", "dph_change")):
        for _, rids in feat.groupby([other, "regio"]).rxn_id.apply(list).items():
            if len(rids) < 2:
                continue
            for r1, r2 in itertools.combinations(rids, 2):
                a, b = lookup.loc[r1, mine], lookup.loc[r2, mine]
                for core, members in ((c, core_index[c]) for c, _ in cuts[a]):
                    if a in members and b in members:
                        rows.append((r1, r2, kind, members[a], members[b]))
                        break
    return pd.DataFrame(rows, columns=["r1", "r2", "kind", "sub_from", "sub_to"]).drop_duplicates(["r1", "r2"])


def splits(feat: pd.DataFrame, pairs: pd.DataFrame, n_random=5, n_loso=10, seed=7):
    """(name, list of (train_idx, test_idx)) 를 yield.

    random        : KFold — 보간(같은 화학 공간) 성능
    dipole_class  : GroupKFold(클래스) — 골격 외삽
    loso:<sub>    : 치환기 <sub>를 포함한 반응 전부를 test — 새 치환기 외삽 (논문의 EDG/EWG 일반화 주장)
    """
    n = len(feat)
    yield "random", list(KFold(n_random, shuffle=True, random_state=seed).split(np.arange(n)))
    g = feat.dclass.values
    yield "dipole_class", list(GroupKFold(n_splits=feat.dclass.nunique()).split(np.arange(n), groups=g))
    counts = pd.Series([s for ss in feat.subs_dip for s in ss]).value_counts()
    for sub in counts.index[:n_loso]:
        te = np.array([i for i, ss in enumerate(feat.subs_dip) if sub in ss])
        tr = np.setdiff1d(np.arange(n), te)
        yield f"loso:{sub}", [(tr, te)]
