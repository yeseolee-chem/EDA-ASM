"""SPEC_11 - extended tblite driver.

Extension of pipeline_rebuild.spec_v1.stage3._run_xtb that also returns
electronic primitives needed for d29..d33:

  - dipole          (3,)    e * Bohr        -> d29 (isolated fragments)
  - gradient        (nat,3) Hartree/Bohr    -> d33 (isolated fragments)
  - overlap-matrix  (norb,norb) unitless    -> d31 (complex)
  - hamiltonian-matrix (norb,norb) Hartree  -> d32 (complex core H0)
  - charges         (nat,)  e               -> reused
  - bond-orders     (nat,nat) unitless      -> reused

We do NOT modify shared pipeline_rebuild.spec_v1.stage3 (per spec section 6).
"""
from __future__ import annotations

# tblite BEFORE torch (libgomp GOMP_5.0 quirk documented in CLAUDE.md).
from tblite.interface import Calculator as _TbliteCalculator  # noqa: F401

import numpy as np

BOHR_TO_ANG = 0.5291772108


def run_xtb_extended(numbers: np.ndarray, positions_ang: np.ndarray,
                     charge: int, mult: int,
                     want_bond_orders: bool = False,
                     want_matrices: bool = False,
                     want_gradient: bool = False) -> dict:
    """Single-point GFN2-xTB with selective heavy-property extraction.

    Positions in Angstrom (converted to Bohr internally, matching stage3).
    """
    n_unpaired = mult - 1
    calc = _TbliteCalculator("GFN2-xTB", numbers, positions_ang / BOHR_TO_ANG,
                             charge=float(charge), uhf=n_unpaired)
    calc.set("verbosity", 0)
    calc.set("max-iter", 500)
    calc.set("mixer-damping", 0.2)
    calc.set("accuracy", 0.1)
    if want_matrices:
        # tblite only writes the overlap / core Hamiltonian into the Result
        # object when 'save-integrals' is enabled before singlepoint().
        calc.set("save-integrals", 1)
    res = calc.singlepoint()

    # Save orbital->atom map from the calculator (uses shell-map + orbital-map,
    # so no single-atom probing needed for AO block slicing).
    orbital_map = None
    if want_matrices:
        shell_to_atom = np.asarray(calc.get("shell-map")).ravel().astype(np.int64)
        orbital_to_shell = np.asarray(calc.get("orbital-map")).ravel().astype(np.int64)
        orbital_map = shell_to_atom[orbital_to_shell]

    E_h = float(res.get("energy"))
    orb_E = np.asarray(res.get("orbital-energies")).ravel()
    orb_occ = np.asarray(res.get("orbital-occupations")).ravel()
    is_occ = orb_occ > 1.0
    homo_idx = int(np.max(np.where(is_occ)[0])) if is_occ.any() else 0
    lumo_idx = int(homo_idx + 1) if (homo_idx + 1) < len(orb_E) else homo_idx
    HOMO = float(orb_E[homo_idx])
    LUMO = float(orb_E[lumo_idx])
    dipole = np.asarray(res.get("dipole")).ravel()   # (3,), e*Bohr
    charges = np.asarray(res.get("charges")).ravel()  # (nat,), e
    out = dict(
        E_h=E_h, HOMO_h=HOMO, LUMO_h=LUMO,
        dipole_norm=float(np.linalg.norm(dipole)),
        dipole=dipole.astype(np.float64),
        charges=charges.astype(np.float64),
        n_orb=int(len(orb_E)),
    )
    if want_bond_orders:
        bo = np.asarray(res.get("bond-orders"))
        if bo.ndim == 3:
            bo = bo.sum(axis=-1)
        out["bond_orders"] = bo.astype(np.float64)
    if want_gradient:
        g = np.asarray(res.get("gradient"))  # (nat, 3), Hartree/Bohr
        out["gradient"] = g.astype(np.float64)
    if want_matrices:
        S = np.asarray(res.get("overlap-matrix"))       # (norb, norb)
        H0 = np.asarray(res.get("hamiltonian-matrix"))  # (norb, norb), Hartree
        out["overlap"] = S.astype(np.float64)
        out["hamiltonian"] = H0.astype(np.float64)
        out["orbital_map"] = orbital_map  # (norb,) atom index per orbital
    return out


# AO -> atom mapping is read directly from the tblite calculator
# (shell-map + orbital-map) in run_xtb_extended(..., want_matrices=True).
# This avoids probing individual atoms and works for any element that
# tblite supports without hardcoding the GFN2 minimal-basis table.
