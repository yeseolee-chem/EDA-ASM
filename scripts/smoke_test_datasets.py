"""Smoke test for the native dataset loaders.

Loads one record from each of QMrxn20 and dipolar_cycloaddition and prints
a summary. Run from the repo root:

    python scripts/smoke_test_datasets.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from eda_asm.datasets import DipolarCycloadditionLoader, QMrxn20Loader  # noqa: E402

QMRXN20_ROOT = REPO / "data" / "raw" / "QMrxn20"
DIPOLAR_ROOT = REPO / "data" / "raw" / "dipolar_cycloaddition"


def _fmt(v, spec=".4f"):
    return format(v, spec) if v is not None else "None"


def smoke_qmrxn20() -> None:
    print("=" * 70)
    print("QMrxn20  (method=mp2, reactant_kind=rcc)")
    print("=" * 70)
    loader = QMrxn20Loader(QMRXN20_ROOT, method="mp2", reactant_kind="rcc")
    ids = loader.list_reaction_ids()
    counts: dict[str, int] = {}
    for r, _ in ids:
        counts[r] = counts.get(r, 0) + 1
    print(f"  reactions available: {len(ids)}  ({counts})")

    for r, label in ids[:1]:
        rec = loader.get(r, label)
        print()
        print(f"  sample: {rec.reaction_id}")
        print(f"    family               : {rec.family}")
        print(f"    natoms (R/TS/P)      : {rec.R.natoms} / {rec.TS.natoms} / "
              f"{rec.P.natoms if rec.P else None}")
        print(f"    formula (TS)         : {rec.TS.formula}")
        print(f"    energy R  (eV)       : {_fmt(rec.R.energy)}")
        print(f"    energy TS (eV)       : {_fmt(rec.TS.energy)}")
        print(f"    energy P  (eV)       : {_fmt(rec.P.energy if rec.P else None)}")
        print(f"    Ea computed (eV)     : {_fmt(rec.Ea_from_energies_eV)}")
        print(f"    Ea native  (kcal/mol): {_fmt(rec.activation_energy_kcal)}")
        print(f"    extra                : {rec.extra}")


def smoke_dipolar() -> None:
    print()
    print("=" * 70)
    print("Dipolar Cycloaddition  (energy_kind=E_sp)")
    print("=" * 70)
    loader = DipolarCycloadditionLoader(DIPOLAR_ROOT, energy_kind="E_sp")
    ids = loader.list_reaction_ids()
    print(f"  reactions available: {len(ids)}")

    for rid in ids[:1]:
        rec = loader.get(rid)
        smiles = rec.rxn_smiles or ""
        print()
        print(f"  sample: {rec.reaction_id}")
        print(f"    family               : {rec.family}")
        print(f"    natoms (R/TS/P)      : {rec.R.natoms} / {rec.TS.natoms} / {rec.P.natoms}")
        print(f"    formula (TS)         : {rec.TS.formula}")
        print(f"    fragments (A | B)    : "
              f"{rec.fragments['A'].size} atoms | {rec.fragments['B'].size} atoms")
        print(f"    rxn_smiles[:90]      : {smiles[:90]}{'...' if len(smiles) > 90 else ''}")
        print(f"    energy R  (eV)       : {_fmt(rec.R.energy)}")
        print(f"    energy TS (eV)       : {_fmt(rec.TS.energy)}")
        print(f"    energy P  (eV)       : {_fmt(rec.P.energy)}")
        print(f"    Ea computed (eV)     : {_fmt(rec.Ea_from_energies_eV)}")
        print(f"    Ea native  (kcal/mol): {_fmt(rec.activation_energy_kcal)}")
        print(f"    extra                : {rec.extra}")


if __name__ == "__main__":
    smoke_qmrxn20()
    smoke_dipolar()
