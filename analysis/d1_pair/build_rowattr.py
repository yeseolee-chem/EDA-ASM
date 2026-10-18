"""Rebuild the row-attribute table from labels_all.json (container was replaced; earlier rowattr.pkl lost).
Columns rebuilt: dipole/dipolarophile SMILES, reference excess ex1/ex2, flags, Xfull/Xb/pos/sigma, cfg (= TS dipole
configuration tag), orient (which dipole terminus the more-substituted dipolarophile carbon bonds to; from the
product structure, no atom map needed), c3 (cis/trans of X on C3 vs the neighbouring ring carbon's substituent).
usage: python -I build_rowattr.py labels_all.json sigma.csv out.pkl"""
import sys, json, numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")
H = 627.5094740631

d = pd.DataFrame(json.load(open(sys.argv[1])))
sg = pd.read_csv(sys.argv[2]).set_index("X")
for k in ["flag_async", "flag_short_forming", "flag_regio_ambiguous", "flag_small_extra_imag"]:
    d["fl_" + k] = d["flags"].map(lambda f: bool(f.get(k)) if isinstance(f, dict) else False)

DIP = {"nitrile oxide": "[#6]#[N+]-[O-]", "azide": "[#7]~[N+]~[#7-,#7]", "nitrone": "[#6]=[N+](-[#6])-[O-]",
       "azomethine ylide": "[#6-]-[N+]=[#6]", "carbonyl ylide": "[#6-]-[O+]=[#6]"}
PAT = {k: Chem.MolFromSmarts(v) for k, v in DIP.items()}


def split(row):
    react, prod = row.rxn_smiles.split(">>")
    comps = react.split(".")
    dip = [c for c in comps if Chem.MolFromSmiles(c).HasSubstructMatch(PAT[row.dipole])]
    assert len(dip) == 1, (row.ts_id, comps)
    return dip[0], [c for c in comps if c != dip[0]][0], prod


sp = d.apply(split, axis=1, result_type="expand"); d["dip_smi"], d["dph_smi"], d["prod_smi"] = sp[0], sp[1], sp[2]
d["dip_smi"] = d.dip_smi.map(lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s)))
d["dph_smi"] = d.dph_smi.map(lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s)))
# reference excess: this row's relaxed reference vs the lowest reference of the same molecule (kcal/mol)
d["ex1"] = (d.e_frag1_rel_eh - d.groupby("dip_smi").e_frag1_rel_eh.transform("min")) * H
d["ex2"] = (d.e_frag2_rel_eh - d.groupby("dph_smi").e_frag2_rel_eh.transform("min")) * H

d["cfg"] = d.ts_dipole_config.fillna("").astype(str)
d["Xfull"] = d.R1.where(d.series == "R1", d.R2)
d["Xb"] = d.Xfull.str.replace(r"^[omp]-", "", regex=True)
d["pos"] = np.where(d.Xfull == "Ph", "ref_aryl", np.where(d.Xfull == "H", "ref_direct", d.attach))
smap, pmap = d.Xb.map(sg.sigma_m), d.Xb.map(sg.sigma_p)
d["sigma"] = np.select([d.pos.isin(["ref_aryl", "ref_direct"]), d.pos == "m", d.pos.isin(["p", "direct", "o"])], [0.0, smap, pmap], np.nan)


# ---------------------------------------------------------------- orient (needs only the product + reactants)
def any_bond_query(m):
    """same heavy atoms (element only), any bond order: finds the reactant skeleton inside the product"""
    m2 = Chem.RWMol(m)
    for at in m2.GetAtoms():
        at.SetFormalCharge(0); at.SetNumExplicitHs(0); at.SetNoImplicit(False)
    m2 = m2.GetMol(); m2.UpdatePropertyCache(False)
    prm = Chem.AdjustQueryParameters.NoAdjustments(); prm.makeBondsGeneric = True
    return Chem.AdjustQueryProperties(m2, prm)


def backbone_in_product(row):
    """-> product mol, (pa, pc, pb) product indices of the dipole backbone, labels, 5-ring"""
    dip = Chem.MolFromSmiles(row.dip_smi); prod = Chem.MolFromSmiles(row.prod_smi)
    (a, c, b), lab = termini(dip, row.dipole)
    for mt in prod.GetSubstructMatches(any_bond_query(dip), useChirality=False, uniquify=False, maxMatches=5000):
        pa, pc, pb = mt[a], mt[c], mt[b]
        # every dipole atom keeps its hydrogens (HN3: NH end vs bare N end; nitrone/ylide: ring CH2 vs N-CH3)
        if any(prod.GetAtomWithIdx(mt[i]).GetTotalNumHs() != dip.GetAtomWithIdx(i).GetTotalNumHs() for i in range(dip.GetNumAtoms())):
            continue
        ring = [set(r) for r in prod.GetRingInfo().AtomRings() if len(r) == 5 and {pa, pc, pb} <= set(r)]
        if ring:
            return prod, (pa, pc, pb), {pa: lab[a], pb: lab[b]}, ring[0]
    return None


def termini(dip, name):
    """backbone atoms of the dipole (a, centre, b) and the terminus labels: anionic end -> 'O-'/'N-'/'C-', other -> symbol"""
    mt = dip.GetSubstructMatches(PAT[name])[0]
    if name == "nitrone":
        a, c, b = mt[0], mt[1], mt[3]
    else:
        a, c, b = mt[0], mt[1], mt[2]
    lab = lambda i: dip.GetAtomWithIdx(i).GetSymbol() + ("-" if dip.GetAtomWithIdx(i).GetFormalCharge() < 0 else "")
    return (a, c, b), {a: lab(a), b: lab(b)}


def heaviness(m, i, exclude):
    """substituent weight around atom i (radius 1, then 2, 3) excluding the given atoms"""
    out = []
    for rad in (1, 2, 3):
        env = set(); front = {i}; seen = {i} | set(exclude)
        for _ in range(rad):
            nxt = set()
            for f in front:
                for n in m.GetAtomWithIdx(f).GetNeighbors():
                    if n.GetIdx() not in seen and n.GetAtomicNum() > 1:
                        nxt.add(n.GetIdx())
            seen |= nxt; env |= nxt; front = nxt
        out.append(sum(m.GetAtomWithIdx(j).GetMass() for j in env))
    return tuple(out)


def orient(row):
    try:
        bb = backbone_in_product(row)
        if bb is None:
            return "?"
        prod, (pa, pc, pb), lab, ring = bb
        dc = [i for i in ring if i not in (pa, pc, pb)]
        ca = [i for i in dc if prod.GetBondBetweenAtoms(i, pa)][0]
        cb = [i for i in dc if prod.GetBondBetweenAtoms(i, pb)][0]
        ha, hb = heaviness(prod, ca, ring), heaviness(prod, cb, ring)
        if ha == hb:
            return "sym"
        return lab[pa] if ha > hb else lab[pb]
    except Exception:
        return "?"


two = d.regio.isin(["r1/2", "r2/2"])
d["orient"] = ""
d.loc[two, "orient"] = d[two].apply(orient, axis=1)


# ---------------------------------------------------------------- C3 orientation (nitrone / AY / CY, X on the dipole)
def c3_orientation(row):
    try:
        bb = backbone_in_product(row)
        if bb is None:
            return "?"
        prod, (pa, pc, pb), lab, ring = bb
        p = Chem.AddHs(prod)
        c3 = None
        for t in (pa, pb):
            at = p.GetAtomWithIdx(t)
            if at.GetSymbol() == "C" and any(n.GetAtomicNum() > 1 and n.GetIdx() not in ring for n in at.GetNeighbors()):
                c3 = at
        if c3 is None:
            return ""
        xs = [n for n in c3.GetNeighbors() if n.GetAtomicNum() > 1 and n.GetIdx() not in ring][0]
        c4 = [n for n in c3.GetNeighbors() if n.GetIdx() in ring and n.GetIdx() not in (pa, pc, pb)][0]
        ms4 = [n for n in c4.GetNeighbors() if n.GetAtomicNum() > 1 and n.GetIdx() not in ring]
        if len(ms4) != 1 or c4.GetHybridization() != Chem.HybridizationType.SP3:
            return ""
        if AllChem.EmbedMolecule(p, randomSeed=7) != 0:
            return "?"
        AllChem.MMFFOptimizeMolecule(p); pos = p.GetConformer().GetPositions()
        rp = pos[sorted(ring)]; nrm = np.linalg.svd(rp - rp.mean(0))[2][-1]
        u = np.dot(pos[xs.GetIdx()] - pos[c3.GetIdx()], nrm); v = np.dot(pos[ms4[0].GetIdx()] - pos[c4.GetIdx()], nrm)
        return "cis" if u * v > 0 else "trans"
    except Exception:
        return "?"


need = d.dipole.isin(["nitrone", "azomethine ylide", "carbonyl ylide"]) & (d.series == "R1") & d.skeleton.isin(["AppE", "AppZ", "B", "Bp"])
d["c3"] = ""
d.loc[need, "c3"] = d[need].apply(c3_orientation, axis=1)

drop = (d.skeleton.isin(["DppC", "DppT"]) | (d.ex1 > 20) | (d.ex2 > 20) | d.fl_flag_async | d.fl_flag_short_forming | d.fl_flag_regio_ambiguous)
print("rows", len(d), "D''", int(d.skeleton.isin(['DppC', 'DppT']).sum()), "ref excess>20 (non-D'')",
      int((((d.ex1 > 20) | (d.ex2 > 20)) & ~d.skeleton.isin(['DppC', 'DppT'])).sum()), "excluded", int(drop.sum()), "kept", int((~drop).sum()))
print("orient:", d.loc[two, "orient"].value_counts().to_dict())
print("c3:", d.loc[need, "c3"].value_counts().to_dict())
d.to_pickle(sys.argv[3] + ".all.pkl")
d[~drop].to_pickle(sys.argv[3])
print(d[~drop].pos.value_counts().to_dict())
