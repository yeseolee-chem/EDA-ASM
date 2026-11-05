#!/usr/bin/env python3
"""audit_review.py - read-only reviewer check of the verify_data leak result (prints JSON, writes nothing).

(a) truncated Nurion systems: the audit flags single structures whose stored geometry splits into the 2 fragments of a
    fold-0 test reactant pair. Frames of the SAME truncated system that are one fragment (forming bonds within 1.3 x
    covalent radii) are not flagged. Here every train / valid row under a Nurion '.../truncated/' subtree gets an identity
    with the forming bonds named in the subtree's autodE file names (_i-j_k-l_) removed from the connectivity, and is
    compared with the fold-0 test reactant-pair graphs.
(b) 'frag' kind rows (single molecules, classed 'rxn' by verify_data): 1-fragment graph vs fold-0 test reactant graphs.
"""
import sys

sys.dont_write_bytecode = True
import json  # noqa: E402
import re  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import mlt_common as M  # noqa: E402
import verify_data as V  # noqa: E402

RX_SUB = re.compile(r"^(N:T\d{5}:.*?/truncated/)")
RX_BONDS = re.compile(r"_(\d+)-(\d+)_(\d+)-(\d+)_")


def broken_pair(syms, pos, bonds):
    Chem = V._rd()
    from rdkit.Chem import rdDetermineBonds
    m = Chem.MolFromXYZBlock(V._xyz_block(syms, pos))
    rdDetermineBonds.DetermineConnectivity(m)
    rw = Chem.RWMol(m)
    n = rw.GetNumAtoms()
    for i, j in bonds:
        if i < n and j < n and rw.GetBondBetweenAtoms(int(i), int(j)) is not None:
            rw.RemoveBond(int(i), int(j))
    g = rw.GetMol()
    return tuple(sorted(V.gkey_mol(f) for f in Chem.GetMolFrags(g, asMols=True, sanitizeFrags=False)))


def main():
    T = M.targets()
    lab = dict(zip(T["rxn_id"].astype(int), T["rxn_smiles"]))
    Sf = pd.read_csv(M.S_FOLDS)
    P0 = Sf[Sf["fold_T1"] == 0]
    test = set(P0["a_rxn"].astype(int)) | set(P0["b_rxn"].astype(int))
    PG, RG = defaultdict(set), defaultdict(set)
    for r in test:
        idn = V.rxn_idents(lab[r])
        PG[idn["pair_graph"]].add(r)
        for x in idn["reac_graph"]:
            RG[x].add(r)
    LR = M.read_json(V.LEAK_REMOVE)
    rm = set(LR["remove"])
    hit_sub = Counter(RX_SUB.match(h["struct_id"]).group(1) for h in LR["detail"] if RX_SUB.match(h["struct_id"]))
    out = dict(n_test=len(test), n_remove=len(rm), hit_subtrees=len(hit_sub))
    for nm in ("train", "valid"):
        al = M.read_extxyz(V.SPLIT / f"{nm}.xyz")
        sids = [str(a.info.get("struct_id")) for a in al]
        bonds = defaultdict(set)
        for s in sids:
            m = RX_SUB.match(s)
            if m:
                b = RX_BONDS.search(s.split("/truncated/", 1)[1])
                if b:
                    x = [int(v) for v in b.groups()]
                    bonds[m.group(1)].add(((x[0], x[1]), (x[2], x[3])))
        st = Counter()
        sub_rows = defaultdict(Counter)
        missed = []
        for i, (a, s) in enumerate(zip(al, sids)):
            m = RX_SUB.match(s)
            kind = str(a.info.get("kind"))
            if kind == "frag" or kind.endswith("_frag"):
                st["frag_rows"] += 1
                try:
                    fr = V.geom_frags(list(a.get_chemical_symbols()), np.asarray(a.positions, float))
                except Exception:                                          # noqa: BLE001
                    fr = None
                if fr and len(fr) == 1 and fr[0] in RG:
                    st["frag_rows_equal_test_reactant"] += 1
                    rx = {int(x) for x in str(a.info.get("rxn_ids") or "").split(",") if x.strip()}
                    if rx & test:
                        st["frag_rows_equal_test_reactant_tagged_test"] += 1
            if not m:
                continue
            sub = m.group(1)
            st["trunc_rows"] += 1
            syms, pos = list(a.get_chemical_symbols()), np.asarray(a.positions, float)
            ident, how = None, "no_bond_names"
            for bp in sorted(bonds.get(sub, ())):
                try:
                    p = broken_pair(syms, pos, bp)
                except Exception:                                          # noqa: BLE001
                    continue
                if len(p) == 2:
                    ident, how = p, "2frag"
                    break
                how = f"{len(p)}frag"
            st[f"trunc_ident_{how}"] += 1
            hit = ident is not None and ident in PG
            flagged = s in rm
            sub_rows[sub]["n"] += 1
            sub_rows[sub]["flagged"] += int(flagged)
            sub_rows[sub]["ident_hit"] += int(hit)
            if hit:
                st["trunc_ident_hit"] += 1
                st["trunc_ident_hit_flagged" if flagged else "trunc_ident_hit_NOT_flagged"] += 1
                if not flagged:
                    try:
                        nf = len(V.geom_frags(syms, pos) or [])
                    except Exception:                                      # noqa: BLE001
                        nf = -1
                    st[f"trunc_ident_hit_NOT_flagged_geomfrags_{nf}"] += 1
                    if len(missed) < 40:
                        missed.append(dict(struct_id=s, kind=kind, test_rxns=sorted(PG[ident])[:5], geom_frags=nf))
            elif flagged:
                st["flagged_but_no_bond_ident_hit"] += 1
        hs = {k: dict(v) for k, v in sub_rows.items() if k in hit_sub or v["ident_hit"]}
        out[nm] = dict(stats=dict(st), n_subtrees_with_hits=len(hs),
                       subtrees_with_ident_hits_but_no_flag=sum(1 for v in hs.values() if v["ident_hit"] and not v["flagged"]),
                       rows_in_hit_subtrees=sum(v["n"] for v in hs.values()),
                       rows_in_hit_subtrees_flagged=sum(v["flagged"] for v in hs.values()),
                       subtrees=dict(list(hs.items())[:60]), missed_examples=missed)
        del al
    print("REVIEW_JSON " + json.dumps(out, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
