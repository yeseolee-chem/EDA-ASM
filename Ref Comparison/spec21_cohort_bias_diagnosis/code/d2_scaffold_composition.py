"""spec21 D2 — topology-based scaffold classification of the dipolarophile.

Classify each reaction by its dipolarophile's reacting bond topology:
  alkyne_in_ring       (cyclooctyne-like)
  bridged_alkene       (norbornene / norbornadiene / oxanorbornadiene)
  other_cyclic_alkene
  acyclic_alkene
  acyclic_alkyne

The dipole side is coarser:
  mesoionic_ring       (reacting atoms inside a 5-ring)
  acyclic_dipole

Uses the atom-mapped `rxn_smiles`. The DIPOLAROPHILE is identified as
the reactant fragment with exactly 2 atoms forming a new
inter-fragment bond in the product (spec19's convention).

Wilson 95% CI on fractions. G21-C spotcheck (20 rows, 10 per half,
seed-fixed) written to `results/D2_spotcheck.csv`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
STAGE = REPO / "Ref Comparison/spec21_cohort_bias_diagnosis"
IN_JOINED = STAGE / "results/cohort_joined.parquet"
IN_STUYVER = STAGE / "results/stuyver_full.parquet"
OUT_FRACT = STAGE / "results/D2_scaffold_fractions.csv"
OUT_PER = STAGE / "results/D2_per_reaction.csv"
OUT_SPOT = STAGE / "results/D2_spotcheck.csv"
OUT_FIG = STAGE / "figures/D2_scaffold_composition.png"

CLASSES = ["alkyne_in_ring", "bridged_alkene", "other_cyclic_alkene",
           "acyclic_alkene", "acyclic_alkyne"]
DIPOLE_CLASSES = ["mesoionic_ring", "acyclic_dipole"]


def _mol(smi: str):
    """MolFromSmiles that also tries sanitize=False fallback for charge-
    separated dipoles like [N+]#[C-] / [C-]/[N+]. If sanitize=True fails,
    return an unsanitized-but-connectivity-intact Mol.
    """
    m = Chem.MolFromSmiles(smi)
    if m is not None:
        return m
    m = Chem.MolFromSmiles(smi, sanitize=False)
    if m is not None:
        try:
            # Sanitize with kekulization + valence checks skipped
            Chem.SanitizeMol(
                m,
                sanitizeOps=(Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE
                             ^ Chem.SANITIZE_PROPERTIES),
            )
        except Exception:
            pass
    return m


def _reactant_fragments(rxn_smiles: str):
    rct_str, prd_str = rxn_smiles.split(">>")
    rct_mols = [_mol(s) for s in rct_str.split(".")]
    prd_mol = _mol(prd_str)
    return rct_mols, prd_mol


def _bonds_by_map(m):
    s = set()
    for b in m.GetBonds():
        a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
        m1, m2 = a1.GetAtomMapNum(), a2.GetAtomMapNum()
        if m1 and m2:
            s.add(frozenset((m1, m2)))
    return s


def bond_forming_map_ids(rmol, prd_mol):
    rct_bonds = _bonds_by_map(rmol)
    prd_bonds = _bonds_by_map(prd_mol)
    map_ids_in_frag = {a.GetAtomMapNum() for a in rmol.GetAtoms() if a.GetAtomMapNum()}
    bond_forming = set()
    for m in map_ids_in_frag:
        for pb in prd_bonds:
            if m in pb:
                other = next(x for x in pb if x != m)
                if frozenset((m, other)) not in rct_bonds and other not in map_ids_in_frag:
                    bond_forming.add(m)
                    break
    return sorted(bond_forming)


def classify_dipolarophile(rmol, bond_forming_map: list[int]) -> tuple[str, dict]:
    """Return (class, features)."""
    if len(bond_forming_map) != 2:
        return "unresolved", {"reason": f"n_bond_forming={len(bond_forming_map)}"}
    map2idx = {a.GetAtomMapNum(): a.GetIdx() for a in rmol.GetAtoms() if a.GetAtomMapNum()}
    idx_a, idx_b = map2idx[bond_forming_map[0]], map2idx[bond_forming_map[1]]
    bond = rmol.GetBondBetweenAtoms(idx_a, idx_b)
    if bond is None:
        return "unresolved", {"reason": "bond_between_reacting_atoms_absent"}
    order = bond.GetBondTypeAsDouble()  # 2.0 = double, 3.0 = triple, 1.5 = aromatic
    in_ring = bond.IsInRing()
    ring_info = rmol.GetRingInfo()
    smallest_ring = None
    if in_ring:
        smallest_ring = min(
            (len(r) for r in ring_info.AtomRings()
             if idx_a in r and idx_b in r), default=None,
        )
    # Bridged: the reacting bond is in a ring AND at least one neighbour of a
    # reacting atom is a bridgehead atom (member of ≥ 2 SSSR rings).
    # Norbornene C=C atoms are in ONE SSSR 5-ring each, but their neighbours
    # (the bicyclic bridgeheads) are in two.
    n_rings_at_a = ring_info.NumAtomRings(idx_a)
    n_rings_at_b = ring_info.NumAtomRings(idx_b)
    bridgehead_neighbor = False
    for ridx in (idx_a, idx_b):
        atom = rmol.GetAtomWithIdx(ridx)
        for nb in atom.GetNeighbors():
            if ring_info.NumAtomRings(nb.GetIdx()) >= 2:
                bridgehead_neighbor = True
                break
        if bridgehead_neighbor:
            break
    bridged = in_ring and (n_rings_at_a >= 2 or n_rings_at_b >= 2 or bridgehead_neighbor)

    features = {
        "bond_order": order,
        "in_ring": in_ring,
        "smallest_ring": smallest_ring,
        "n_rings_at_a": n_rings_at_a,
        "n_rings_at_b": n_rings_at_b,
        "bridged": bridged,
    }

    if abs(order - 3.0) < 0.1 and in_ring:
        return "alkyne_in_ring", features
    if abs(order - 3.0) < 0.1 and not in_ring:
        return "acyclic_alkyne", features
    if abs(order - 2.0) < 0.1 and bridged:
        return "bridged_alkene", features
    if abs(order - 2.0) < 0.1 and in_ring:
        return "other_cyclic_alkene", features
    if abs(order - 2.0) < 0.1 and not in_ring:
        return "acyclic_alkene", features
    return "unresolved", features


def classify_dipole(rmol, bond_forming_map: list[int]) -> str:
    if len(bond_forming_map) != 2:
        return "unresolved"
    map2idx = {a.GetAtomMapNum(): a.GetIdx() for a in rmol.GetAtoms() if a.GetAtomMapNum()}
    idx_a, idx_b = map2idx[bond_forming_map[0]], map2idx[bond_forming_map[1]]
    ring_info = rmol.GetRingInfo()
    # if either reacting atom is in a 5-membered ring → mesoionic; else acyclic
    for ring in ring_info.AtomRings():
        if len(ring) == 5 and (idx_a in ring or idx_b in ring):
            return "mesoionic_ring"
    return "acyclic_dipole"


def process_reaction(rxn_smiles: str) -> dict:
    try:
        rct_mols, prd_mol = _reactant_fragments(rxn_smiles)
        if prd_mol is None or any(m is None for m in rct_mols) or len(rct_mols) != 2:
            return {"dipolarophile_class": "unresolved",
                    "dipole_class": "unresolved",
                    "reason": "smiles_parse_or_frag_count"}
        bf_per_frag = [bond_forming_map_ids(m, prd_mol) for m in rct_mols]
        # dipolarophile: 2 bond-forming atoms AND they're directly bonded in reactant
        classes = []
        for i, (m, bf) in enumerate(zip(rct_mols, bf_per_frag)):
            if len(bf) != 2:
                classes.append(None)
                continue
            map2idx = {a.GetAtomMapNum(): a.GetIdx() for a in m.GetAtoms() if a.GetAtomMapNum()}
            bond = m.GetBondBetweenAtoms(map2idx[bf[0]], map2idx[bf[1]])
            classes.append("dipolarophile" if bond is not None else "dipole")
        if classes.count("dipolarophile") == 1 and classes.count("dipole") == 1:
            i_dp = classes.index("dipolarophile")
            i_dl = classes.index("dipole")
        else:
            return {"dipolarophile_class": "unresolved",
                    "dipole_class": "unresolved",
                    "reason": f"could_not_split_roles: {classes}"}

        dp_class, dp_feat = classify_dipolarophile(rct_mols[i_dp], bf_per_frag[i_dp])
        dl_class = classify_dipole(rct_mols[i_dl], bf_per_frag[i_dl])
        return {
            "dipolarophile_class": dp_class,
            "dipole_class": dl_class,
            "dipolarophile_smiles": Chem.MolToSmiles(rct_mols[i_dp]),
            "dipole_smiles": Chem.MolToSmiles(rct_mols[i_dl]),
            "features": dp_feat,
        }
    except Exception as e:
        return {"dipolarophile_class": "unresolved",
                "dipole_class": "unresolved",
                "reason": f"exception:{type(e).__name__}:{e}"}


def wilson_ci(count: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 0.0
    p = count / n
    denom = 1.0 + z*z/n
    centre = p + z*z/(2*n)
    half = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n))
    lo = (centre - half) / denom
    hi = (centre + half) / denom
    return p, max(0.0, lo), min(1.0, hi)


def main() -> int:
    STAGE.joinpath("figures").mkdir(exist_ok=True)
    joined = pd.read_parquet(IN_JOINED)
    stuyver = pd.read_parquet(IN_STUYVER)

    def _process_group(df: pd.DataFrame, has_our_meta: bool) -> pd.DataFrame:
        rows = []
        for _, row in df.iterrows():
            smi = row["rxn_smiles_ours"] if has_our_meta else row["rxn_smiles"]
            r = process_reaction(str(smi))
            entry = {"rxn_smiles": smi,
                     "dipolarophile_class": r["dipolarophile_class"],
                     "dipole_class": r["dipole_class"],
                     "reason": r.get("reason", "")}
            if has_our_meta:
                entry["reaction_id"] = row["reaction_id"]
                entry["sub_source"] = row["sub_source"]
                entry["source_id"] = row["source_id"]
            else:
                entry["source_id"] = row["rxn_id"]
            rows.append(entry)
        return pd.DataFrame(rows)

    per_ours = _process_group(joined, has_our_meta=True)
    per_ours.to_csv(OUT_PER, index=False)
    print(f"[write] {OUT_PER}")

    per_full = _process_group(stuyver, has_our_meta=False)

    groups = {
        "full_5269":  per_full,
        "ours_400":   per_ours,
        "locked_192": per_ours[per_ours["sub_source"] == "locked_778"],
        "spec16_208": per_ours[per_ours["sub_source"] == "spec16"],
    }

    rows = []
    for cls in CLASSES + ["unresolved"]:
        for name, df in groups.items():
            n_cls = int((df["dipolarophile_class"] == cls).sum())
            p, lo, hi = wilson_ci(n_cls, len(df))
            rows.append({"axis": "dipolarophile", "class": cls, "group": name,
                          "n": n_cls, "n_group": len(df),
                          "fraction": p, "ci95_lo": lo, "ci95_hi": hi})
    for cls in DIPOLE_CLASSES + ["unresolved"]:
        for name, df in groups.items():
            n_cls = int((df["dipole_class"] == cls).sum())
            p, lo, hi = wilson_ci(n_cls, len(df))
            rows.append({"axis": "dipole", "class": cls, "group": name,
                          "n": n_cls, "n_group": len(df),
                          "fraction": p, "ci95_lo": lo, "ci95_hi": hi})
    fract = pd.DataFrame(rows)
    fract.to_csv(OUT_FRACT, index=False)
    print(f"[write] {OUT_FRACT}")
    print(fract[fract["axis"] == "dipolarophile"].pivot(
          index="class", columns="group", values="fraction").to_string())

    # Spotcheck: 20 rows, 10 per half, seed-fixed
    rng = np.random.default_rng(42)
    parts = []
    for sub in ("locked_778", "spec16"):
        s = per_ours[per_ours["sub_source"] == sub]
        pick = rng.choice(len(s), min(10, len(s)), replace=False)
        parts.append(s.iloc[pick])
    spot = pd.concat(parts).reset_index(drop=True)
    spot["reviewed"] = ""    # user fills this
    spot["correct?"] = ""
    spot.to_csv(OUT_SPOT, index=False)
    print(f"[write] {OUT_SPOT}")

    # Bar chart: dipolarophile classes
    fig, ax = plt.subplots(figsize=(11, 5))
    class_order = CLASSES + ["unresolved"]
    x = np.arange(len(class_order))
    width = 0.2
    for i, name in enumerate(["full_5269", "ours_400", "locked_192", "spec16_208"]):
        sub = fract[(fract["axis"] == "dipolarophile") & (fract["group"] == name)]
        y = np.array([float(sub[sub["class"] == c]["fraction"].iloc[0]) for c in class_order])
        lo = np.array([float(sub[sub["class"] == c]["ci95_lo"].iloc[0]) for c in class_order])
        hi = np.array([float(sub[sub["class"] == c]["ci95_hi"].iloc[0]) for c in class_order])
        # Clip clip errors to non-negative (defensive against Wilson boundary + clip interaction)
        err_lo = np.clip(y - lo, 0.0, None)
        err_hi = np.clip(hi - y, 0.0, None)
        err = np.vstack([err_lo, err_hi])
        ax.bar(x + (i - 1.5) * width, y, width, label=name, yerr=err, capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels(class_order, rotation=25)
    ax.set_ylabel("fraction of reactions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=140)
    print(f"[write] {OUT_FIG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
