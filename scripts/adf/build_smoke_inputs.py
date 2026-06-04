#!/usr/bin/env python
"""Build ADF EDA-NOCV input deck for the smoke-test reaction.

Default: dipolar cycloaddition reaction id 0.
The reaction record has explicit r0/r1 fragments, so fragmentation is trivial.

Writes everything (XYZ snapshots + run_eda.sh) to
    ADF_250/smoke_test/<reaction_id>/

Run with `python scripts/adf/build_smoke_inputs.py`. No heavy compute.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from eda_asm.adf import ADFRunSpec, FragmentSpec, generate_run_script  # noqa: E402
from eda_asm.datasets import DipolarCycloadditionLoader  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rxn-id", type=int, default=0,
                        help="dipolar reaction id to use (default: 0)")
    parser.add_argument("--out-root", type=Path,
                        default=REPO / "ADF_800" / "smoke_test",
                        help="parent dir for the run dir")
    args = parser.parse_args()

    loader = DipolarCycloadditionLoader(
        REPO / "data" / "raw" / "dipolar_cycloaddition",
        energy_kind="E_sp",
    )
    rec = loader.get(args.rxn_id)
    reaction_id = f"smoke_dipolar_{args.rxn_id}"
    print(f"reaction       : {rec.reaction_id}")
    print(f"family         : {rec.family}")
    print(f"natoms (R/TS/P): {rec.R.natoms} / {rec.TS.natoms} / {rec.P.natoms}")
    print(f"formula        : {rec.TS.formula}")
    print(f"fragments A|B  : {rec.fragments['A'].size}|{rec.fragments['B'].size}")

    fA_indices = rec.fragments["A"]
    fB_indices = rec.fragments["B"]
    # By construction (autodE) the TS atom order is r0 atoms then r1 atoms.
    # So positions_at_TS = TS.positions[indices], and positions_relaxed = R.positions[indices].
    fragA = FragmentSpec(
        name="fA",
        indices=fA_indices,
        numbers=rec.TS.numbers[fA_indices],
        positions_at_TS=rec.TS.positions[fA_indices],
        positions_relaxed=rec.R.positions[fA_indices],
        charge=0,
    )
    fragB = FragmentSpec(
        name="fB",
        indices=fB_indices,
        numbers=rec.TS.numbers[fB_indices],
        positions_at_TS=rec.TS.positions[fB_indices],
        positions_relaxed=rec.R.positions[fB_indices],
        charge=0,
    )
    spec = ADFRunSpec(
        reaction_id=reaction_id,
        fragA=fragA,
        fragB=fragB,
        ts_numbers=rec.TS.numbers,
        ts_positions=rec.TS.positions,
        total_charge=0,
        extra_provenance={"dipolar_rxn_id": int(args.rxn_id),
                          "rxn_smiles": rec.rxn_smiles},
    )

    out_dir = args.out_root / reaction_id
    run_path = generate_run_script(spec, out_dir)
    meta = {
        "reaction_id": reaction_id,
        "source": "dipolar_cycloaddition",
        "dipolar_rxn_id": int(args.rxn_id),
        "rxn_smiles": rec.rxn_smiles,
        "ts_natoms": int(rec.TS.natoms),
        "fragA_natoms": int(len(fA_indices)),
        "fragB_natoms": int(len(fB_indices)),
        "has_Br": spec.has_Br,
        "total_charge": spec.total_charge,
        "functional": f"Hybrid {spec.functional}",
        "dispersion": spec.dispersion,
        "basis": spec.basis_type,
        "numerical_quality": spec.numerical_quality,
        "Ea_native_kcal": float(rec.activation_energy_kcal)
            if rec.activation_energy_kcal else None,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"\nwrote inputs to: {out_dir}")
    print(f"run script     : {run_path}")
    print(f"meta           : {out_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
