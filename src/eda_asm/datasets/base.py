"""Common dataclasses + XYZ reader shared by all native reaction-dataset loaders.

The pipeline downstream (ADF EDA-ASM input prep) only needs an R/TS/P triple
per reaction plus optional fragmentation hints and metadata. This module
defines that minimal record shape; concrete loaders (QMrxn20, dipolar
cycloaddition, ...) populate it from their native on-disk layout.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import ase
import ase.data
import ase.io
import numpy as np

HARTREE_TO_EV = 27.211386245988
KCAL_PER_MOL_TO_EV = 0.04336410390059322


@dataclass(slots=True)
class Geometry:
    """A single molecular geometry — minimal subset of ase.Atoms we need downstream."""

    numbers: np.ndarray            # (natoms,) int
    positions: np.ndarray          # (natoms, 3) float, Å
    energy: float | None = None    # total energy in eV, optional

    @property
    def natoms(self) -> int:
        return int(self.numbers.shape[0])

    @property
    def symbols(self) -> list[str]:
        return [ase.data.chemical_symbols[int(z)] for z in self.numbers]

    @property
    def formula(self) -> str:
        return ase.Atoms(numbers=self.numbers).get_chemical_formula()

    def to_ase(self) -> ase.Atoms:
        atoms = ase.Atoms(numbers=self.numbers, positions=self.positions)
        if self.energy is not None:
            atoms.info["energy_eV"] = self.energy
        return atoms


@dataclass(slots=True)
class ReactionRecord:
    """A reactant / TS / product triple for one reaction, ready for ADF EDA-ASM input prep."""

    reaction_id: str
    family: str                                # "QMrxn20-e2" | "QMrxn20-sn2" | "dipolar"
    R: Geometry
    TS: Geometry
    P: Geometry | None
    fragments: dict[str, np.ndarray] = field(default_factory=dict)  # name -> indices into TS
    rxn_smiles: str | None = None
    activation_energy_kcal: float | None = None
    extra: dict = field(default_factory=dict)

    @property
    def Ea_from_energies_eV(self) -> float | None:
        if self.R.energy is None or self.TS.energy is None:
            return None
        return self.TS.energy - self.R.energy


def read_xyz(path: Path, energy: float | None = None) -> Geometry:
    """Read a single-frame XYZ file via ase.io and wrap in Geometry."""
    atoms = ase.io.read(str(path), format="xyz")
    return Geometry(
        numbers=np.asarray(atoms.numbers, dtype=np.int64),
        positions=np.asarray(atoms.positions, dtype=np.float64),
        energy=energy,
    )
