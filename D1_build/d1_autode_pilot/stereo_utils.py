#!/usr/bin/env python3
"""stereo_utils.py — dipole backbone dihedrals and configuration tags from 3D geometries.

Why this exists
---------------
autodE locates the TS of an addition from the PRODUCT side (autodE 1.4.5,
reaction.py::locate_transition_state: "If there are more bonds in the product e.g. an
addition reaction then switch"), so the dipole's in-plane configuration at the TS (E/Z
about the dipole C=X, W/U/S shape) is inherited from the product conformer, not from
the reactant SMILES. Coley 2023 (Sci. Data 10:66, doi:10.1038/s41597-023-01977-8, Fig. 7
and postprocess_reaction_profiles.py) therefore re-derives a TS-compatible dipole
reference afterwards. This module supplies the geometric pieces:

  * template_match()    SMILES map number -> geometry index (bond orders + charges from a
                        mapped template, connectivity from the coordinates)
  * dipole_backbone()   (t1, c, t2) map numbers of the dipole atoms that enter the ring
  * tracked_dihedrals() the dihedrals about the two backbone bonds (Coley's rule, with the
                        poles taken from the ring instead of from formal charges — see
                        SPEC §5.3 deviation D-1)
  * config_tags()       'c'/'t' per tracked dihedral, '' when there is nothing to track
  * ez_tags()           CIP E/Z of every stereogenic double bond (reporting only)
"""
from __future__ import annotations

import collections

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdDetermineBonds

LINEAR_DEG = 165.0      # an sp centre in a dihedral makes it undefined -> '?'


# ---------------------------------------------------------------- geometry <-> template
def _mol_from_xyz(xyz_block: str, charge: int = 0) -> Chem.Mol:
    m = Chem.MolFromXYZBlock(xyz_block)
    rdDetermineBonds.DetermineConnectivity(m, charge=charge)
    return m


def _clean_template(smiles: str) -> Chem.Mol:
    t = Chem.MolFromSmiles(smiles)
    for b in t.GetBonds():
        b.SetStereo(Chem.BondStereo.STEREONONE)
    for a in t.GetAtoms():
        a.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    return Chem.AddHs(t)


def template_match(xyz_block: str, template_smiles: str, charge: int = 0):
    """Return (mol with 3D conformer, {template_idx: geometry_idx}, template mol with Hs)."""
    tmpl = _clean_template(template_smiles)
    geo = _mol_from_xyz(xyz_block, charge)
    m = AllChem.AssignBondOrdersFromTemplate(tmpl, geo)
    match = m.GetSubstructMatch(tmpl)
    if not match:
        raise ValueError("geometry does not match the template graph")
    return m, dict(enumerate(match)), tmpl


def dihedral_deg(p0, p1, p2, p3) -> float:
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    x = np.dot(v, w)
    y = np.dot(np.cross(b1n, v), w)
    return float(np.degrees(np.arctan2(y, x)))


def angle_deg(p0, p1, p2) -> float:
    a, b = p0 - p1, p2 - p1
    c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


# ---------------------------------------------------------------- dipole backbone
def dipole_backbone(mapped_rxn_smiles: str):
    """(dipole reactant index, (t1, c, t2) map numbers) from an atom-mapped reaction SMILES.

    The ring atoms contributed by the dipole are those of the smallest product ring that
    contains both forming bonds (same rule as fragmenter.dipole_smiles_index); c is the one
    bonded to the other two inside the dipole.
    """
    r_str, p_str = mapped_rxn_smiles.split(">>")
    rmols = [Chem.MolFromSmiles(s) for s in r_str.split(".")]
    pmol = Chem.MolFromSmiles(p_str)

    def bonds(m):
        return {tuple(sorted((b.GetBeginAtom().GetAtomMapNum(), b.GetEndAtom().GetAtomMapNum())))
                for b in m.GetBonds()}
    formed = bonds(pmol) - set().union(*[bonds(m) for m in rmols])
    owner = {a.GetAtomMapNum(): k for k, m in enumerate(rmols) for a in m.GetAtoms()}
    idx2map = {a.GetIdx(): a.GetAtomMapNum() for a in pmol.GetAtoms()}
    rings = [set(idx2map[i] for i in r) for r in pmol.GetRingInfo().AtomRings()]
    rings = [r for r in rings if all(a in r and b in r for a, b in formed)]
    if not rings:
        raise ValueError("no ring contains both forming bonds")
    ring = min(rings, key=len)
    cnt = collections.Counter(owner[m] for m in ring)
    dip = [k for k, v in cnt.items() if v == 3]
    if len(ring) != 5 or len(dip) != 1:
        raise ValueError("not a [3+2] ring")
    dmol = rmols[dip[0]]
    bb = [m for m in ring if owner[m] == dip[0]]
    by_map = {a.GetAtomMapNum(): a for a in dmol.GetAtoms()}
    nb = {m: {n.GetAtomMapNum() for n in by_map[m].GetNeighbors()} & set(bb) for m in bb}
    c = [m for m in bb if len(nb[m]) == 2]
    if len(c) != 1:
        raise ValueError("dipole backbone is not a 3-atom chain")
    t1, t2 = sorted(m for m in bb if m != c[0])
    return dip[0], (t1, c[0], t2), Chem.MolToSmiles(dmol)


def tracked_dihedrals(dipole_mapped_smiles: str, backbone):
    """Coley's dihedrals-to-track (postprocess_reaction_profiles.py::get_dihedral_angles_to_track).

    For the two backbone bonds (t1,c) and (c,t2): quadruple (n_u, u, v, n_v) with n_x the
    first heavy neighbour of x other than its partner (Coley: first *mapped* neighbour;
    all heavy atoms are mapped in the Coley-style SMILES). Nothing is tracked when a
    backbone bond is triple or aromatic (propargyl/allenyl-type or cyclic dipoles), as in
    Coley. Returned as template atom indices of the H-added template.
    """
    tmpl = _clean_template(dipole_mapped_smiles)
    idx = {a.GetAtomMapNum(): a.GetIdx() for a in tmpl.GetAtoms() if a.GetAtomMapNum()}
    t1, c, t2 = (idx[m] for m in backbone)
    for u, v in ((t1, c), (c, t2)):
        bt = tmpl.GetBondBetweenAtoms(u, v).GetBondType()
        if bt in (Chem.BondType.TRIPLE, Chem.BondType.AROMATIC):
            return []
    out = []
    for u, v in ((t1, c), (c, t2)):
        nu = [n.GetIdx() for n in tmpl.GetAtomWithIdx(u).GetNeighbors()
              if n.GetIdx() != v and n.GetAtomicNum() > 1]
        nv = [n.GetIdx() for n in tmpl.GetAtomWithIdx(v).GetNeighbors()
              if n.GetIdx() != u and n.GetAtomicNum() > 1]
        if nu and nv:
            out.append((nu[0], u, v, nv[0]))
    return out


def config_tags(xyz_block: str, dipole_mapped_smiles: str, backbone, charge: int = 0):
    """('ct' style tag string, [dihedral values], [geometry-index quadruples])."""
    quads = tracked_dihedrals(dipole_mapped_smiles, backbone)
    if not quads:
        return "", [], []
    m, t2g, _ = template_match(xyz_block, dipole_mapped_smiles, charge)
    pos = m.GetConformer().GetPositions()
    tags, vals, gq = [], [], []
    for q in quads:
        g = [t2g[i] for i in q]
        p = [pos[i] for i in g]
        if angle_deg(*p[0:3]) > LINEAR_DEG or angle_deg(*p[1:4]) > LINEAR_DEG:
            tags.append("?"); vals.append(float("nan"))
        else:
            d = dihedral_deg(*p)
            tags.append("c" if abs(d) < 90 else "t"); vals.append(d)
        gq.append(g)
    return "".join(tags), vals, gq


def tags_compatible(a: str, b: str) -> bool:
    """Equal wherever both are defined ('?' = undefined, e.g. a linear reactant dipole)."""
    return len(a) == len(b) and all(x == y or "?" in (x, y) for x, y in zip(a, b))


# ---------------------------------------------------------------- reporting
def ez_tags(xyz_block: str, template_smiles: str, charge: int = 0) -> str:
    """CIP E/Z of every stereogenic double bond of the template, in template bond order."""
    m, t2g, tmpl = template_match(xyz_block, template_smiles, charge)
    Chem.AssignStereochemistryFrom3D(m)
    tags = []
    for tb in tmpl.GetBonds():
        if tb.GetBondType() != Chem.BondType.DOUBLE:
            continue
        st = m.GetBondBetweenAtoms(t2g[tb.GetBeginAtomIdx()], t2g[tb.GetEndAtomIdx()]).GetStereo()
        if st in (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOTRANS):
            tags.append("E")
        elif st in (Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOCIS):
            tags.append("Z")
    return "".join(tags)


def xyz_block_from_atoms(symbols, coords) -> str:
    lines = [str(len(symbols)), ""]
    lines += [f"{s} {x:.6f} {y:.6f} {z:.6f}" for s, (x, y, z) in zip(symbols, coords)]
    return "\n".join(lines) + "\n"
