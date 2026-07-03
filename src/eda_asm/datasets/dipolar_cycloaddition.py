"""Loader for the Stuyver / Jorner / Coley dipolar cycloaddition dataset.

Reference: doi:10.1038/s41597-023-01977-8
Archive:   https://doi.org/10.6084/m9.figshare.21707888.v5

Layout (after extracting full_dataset_profiles.tar.gz under root/extracted/):
  full_dataset_profiles/{rxn_id}/
    r0_{formulaA}.xyz                   fragment A reactant (dipole, relaxed)
    r1_{formulaB}.xyz                   fragment B reactant (dipolarophile, relaxed)
    TS_{hash}_{annotation}.xyz          transition state
    TS_imag_mode.xyz                    (skipped — imaginary-mode vector, not a geometry)
    p0_{formulaP}.xyz                   product (relaxed)
    energies.csv                        Species, E_opt, G_cont, H_cont, E_sp  (all Hartree)
    frequency_logs.tar.gz               nested Gaussian outputs — left compressed
    single_point_logs.tar.gz            nested Gaussian outputs — left compressed

Root-level full_dataset.csv provides per-reaction metadata: atom-mapped SMILES,
solvent, temperature, G_act and G_r (kcal/mol).

ASM fragmentation is unambiguous: fragment A = r0 atoms, fragment B = r1 atoms,
concatenated in that order to match the TS atom ordering (autodE convention).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem.rdDetermineBonds import DetermineConnectivity

from .base import HARTREE_TO_EV, Geometry, ReactionRecord, read_xyz


def _fragment_indices_via_atommap(
    ts_path: Path, rxn_smiles: str, n_atoms_total: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Determine (fragA_indices, fragB_indices) into the TS atom array by
    matching each LHS-SMILES fragment as a subgraph of the TS connectivity.

    autodE's TS atom order is *not* `r0 atoms followed by r1 atoms` (the
    convention is `all heavy atoms followed by all hydrogens` after a
    permutation that's not externally documented), so naive index slicing
    yields garbage fragments. The atom-mapped reaction SMILES + RDKit's
    GetSubstructMatch is a robust assignment.
    """
    lhs, _ = rxn_smiles.split(">>", 1)
    smi_parts = lhs.split(".")
    if len(smi_parts) != 2:
        raise ValueError(f"expected exactly two LHS fragments in {rxn_smiles!r}")
    smi_a, smi_b = smi_parts

    mol_a = Chem.AddHs(Chem.MolFromSmiles(smi_a))
    mol_b = Chem.AddHs(Chem.MolFromSmiles(smi_b))
    if mol_a is None or mol_b is None:
        raise ValueError(f"could not parse one of the LHS fragments: {smi_parts}")

    raw_ts = Chem.MolFromXYZFile(str(ts_path))
    if raw_ts is None:
        raise ValueError(f"MolFromXYZFile failed on {ts_path}")
    ts_mol = Chem.Mol(raw_ts)
    DetermineConnectivity(ts_mol, charge=0)

    match_a = ts_mol.GetSubstructMatch(mol_a, useChirality=False)
    match_b = ts_mol.GetSubstructMatch(mol_b, useChirality=False)
    if not match_a or not match_b:
        raise ValueError(
            f"substructure match failed for TS={ts_path.name}: "
            f"|A|={len(match_a)} expected {mol_a.GetNumAtoms()}, "
            f"|B|={len(match_b)} expected {mol_b.GetNumAtoms()}"
        )
    set_a, set_b = set(match_a), set(match_b)
    if set_a & set_b:
        raise ValueError(f"fragA and fragB matches overlap in {ts_path.name}")
    if len(set_a) + len(set_b) != n_atoms_total:
        raise ValueError(
            f"fragments cover {len(set_a) + len(set_b)} atoms but TS has "
            f"{n_atoms_total} ({ts_path.name})"
        )
    return (
        np.array(sorted(set_a), dtype=np.int64),
        np.array(sorted(set_b), dtype=np.int64),
    )


class DipolarCycloadditionLoader:
    """Iterate reactions in the dipolar cycloaddition dataset as (R, TS, P) triples.

    Parameters
    ----------
    root : Path
        Directory containing full_dataset.csv and the `extracted/` subdir
        with full_dataset_profiles/ inside.
    profiles_subdir : str
        Path (relative to root) to the directory holding {rxn_id}/ subdirs.
    energy_kind : {"E_opt", "E_sp"}
        Which Hartree energy column from energies.csv to use. E_sp is the
        single-point energy at B3LYP-D3(BJ)/def2-TZVP; E_opt is the
        optimisation-level energy at def2-SVP.
    """

    def __init__(
        self,
        root: str | Path,
        profiles_subdir: str = "extracted/full_dataset_profiles",
        energy_kind: str = "E_sp",
    ):
        self.root = Path(root)
        if energy_kind not in {"E_opt", "E_sp"}:
            raise ValueError(f"energy_kind must be E_opt|E_sp, got {energy_kind!r}")
        self.energy_kind = energy_kind
        self._profiles_dir = self.root / profiles_subdir
        if not self._profiles_dir.is_dir():
            raise FileNotFoundError(f"profiles dir not found: {self._profiles_dir}")
        self._metadata = self._load_metadata()

    def _load_metadata(self) -> pd.DataFrame:
        df = pd.read_csv(self.root / "full_dataset.csv", index_col=0)
        df["rxn_id"] = df["rxn_id"].astype(int)
        return df.set_index("rxn_id")

    def list_reaction_ids(self) -> list[int]:
        meta_ids = set(self._metadata.index.tolist())
        avail = [
            int(p.name)
            for p in self._profiles_dir.iterdir()
            if p.is_dir() and p.name.isdigit()
        ]
        return sorted(rid for rid in avail if rid in meta_ids)

    def __len__(self) -> int:
        return len(self.list_reaction_ids())

    def __iter__(self) -> Iterator[ReactionRecord]:
        for rid in self.list_reaction_ids():
            try:
                yield self.get(rid)
            except (FileNotFoundError, ValueError):
                continue

    def get(self, rxn_id: int) -> ReactionRecord:
        rxn_dir = self._profiles_dir / str(rxn_id)
        if not rxn_dir.is_dir():
            raise KeyError(f"no profile directory for reaction {rxn_id}")

        energies = pd.read_csv(
            rxn_dir / "energies.csv", skiprows=1, skipinitialspace=True
        ).set_index("Species")

        def eV(species: str) -> float | None:
            if species not in energies.index:
                return None
            return float(energies.loc[species, self.energy_kind]) * HARTREE_TO_EV

        r0_path = _single_match(rxn_dir, "r0_*.xyz")
        r1_path = _single_match(rxn_dir, "r1_*.xyz")
        p0_path = _single_match(rxn_dir, "p0_*.xyz")
        ts_candidates = [p for p in rxn_dir.glob("TS_*.xyz") if p.name != "TS_imag_mode.xyz"]
        if len(ts_candidates) != 1:
            raise ValueError(
                f"expected exactly one TS_*.xyz in {rxn_dir}, found "
                f"{[p.name for p in ts_candidates]}"
            )
        ts_path = ts_candidates[0]

        r0 = read_xyz(r0_path, energy=eV(r0_path.stem))
        r1 = read_xyz(r1_path, energy=eV(r1_path.stem))
        TS = read_xyz(ts_path, energy=eV(ts_path.stem))
        P = read_xyz(p0_path, energy=eV(p0_path.stem))

        R_energy = (
            None if r0.energy is None or r1.energy is None else r0.energy + r1.energy
        )
        R = Geometry(
            numbers=np.concatenate([r0.numbers, r1.numbers]),
            positions=np.concatenate([r0.positions, r1.positions], axis=0),
            energy=R_energy,
        )

        if R.natoms != TS.natoms:
            raise ValueError(
                f"reactant atom count {R.natoms} (r0={r0.natoms}+r1={r1.natoms}) "
                f"!= TS atom count {TS.natoms} for reaction {rxn_id}"
            )

        # NOTE: fragments are placeholder indices into the r0||r1-concat R
        # ordering — NOT the autodE TS atom ordering. Downstream code that
        # uses these to slice TS.positions/TS.numbers will produce WRONG
        # fragmentations. The correct fragmentation logic is being reworked
        # per the user's referenced pattern; see TODO.
        fragments = {
            "A": np.arange(r0.natoms, dtype=np.int64),
            "B": np.arange(r0.natoms, r0.natoms + r1.natoms, dtype=np.int64),
        }

        meta = self._metadata.loc[rxn_id]
        return ReactionRecord(
            reaction_id=f"dipolar-{rxn_id}",
            family="dipolar",
            R=R,
            TS=TS,
            P=P,
            fragments=fragments,
            rxn_smiles=str(meta["rxn_smiles"]),
            activation_energy_kcal=float(meta["G_act"]),
            extra={
                "rxn_id": int(rxn_id),
                "solvent": str(meta["solvent"]),
                "temp_K": float(meta["temp"]),
                "G_r_kcal": float(meta["G_r"]),
                "r0_species": r0_path.stem,
                "r1_species": r1_path.stem,
                "p0_species": p0_path.stem,
                "ts_species": ts_path.stem,
                "r0_natoms": r0.natoms,
                "r1_natoms": r1.natoms,
                "energy_kind": self.energy_kind,
            },
        )


def _single_match(rxn_dir: Path, pattern: str) -> Path:
    matches = list(rxn_dir.glob(pattern))
    if len(matches) == 1:
        return matches[0]
    # Some dipolar reactions ship both canonical and `_alt` stereo variants.
    # Prefer the canonical one (without `_alt` in the stem).
    primary = [p for p in matches if "_alt" not in p.stem]
    if len(primary) == 1:
        return primary[0]
    raise ValueError(
        f"expected exactly one {pattern} in {rxn_dir}, found {[p.name for p in matches]}"
    )
