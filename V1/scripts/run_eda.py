"""Phase 3 — ADF EDA at TS + isolated-fragment optimizations.

Run via $AMSBIN/amspython.

Per spec §6 Phase 3, the workflow is:
  1. Read TS geom + fragments.json.
  2. AMSJob (GeometryOptimization) for each fragment in isolation
     to get the relaxed (preparation-energy reference) geometries.
  3. ADFFragmentJob with (TS-geom frag, relaxed-geom frag) for both —
     PLAMS internally runs 5 child jobs (f1, f2, full, f1_opt, f2_opt)
     and exposes the EDA decomposition via get_energy_decomposition().
  4. Compute strain = ΔE_prep_A + ΔE_prep_B from fragment SP energies.
  5. Write runs/<id>/eda/asr_vector.json.

Settings per spec §3.2:
  ZORA-BLYP-D3(BJ)/TZ2P, all-electron, NoSym; unrestricted doublet for fragments.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from scm.plams import (
    Atom,
    Molecule,
    Settings,
    AMSJob,
    ADFFragmentJob,
    init,
    finish,
    config,
)

PROJECT = Path(__file__).resolve().parent.parent
RUNS = PROJECT / "runs"


# ---------- ADF settings ----------------------------------------------------

def adf_full_settings() -> Settings:
    s = Settings()
    s.input.ams.Task = "SinglePoint"
    s.input.adf.basis.type = "TZ2P"
    s.input.adf.basis.core = "None"
    s.input.adf.xc.gga = "BLYP"
    s.input.adf.xc.dispersion = "Grimme3 BJDAMP"
    s.input.adf.relativity.level = "Scalar"
    s.input.adf.symmetry = "NOSYM"
    s.input.adf.numericalquality = "Good"
    return s


def adf_frag1_settings() -> Settings:
    """Fragment A: doublet with α excess (+1)."""
    s = adf_full_settings()
    s.input.adf.unrestricted = "yes"
    s.input.adf.spinpolarization = 1
    s.input.adf.scf.mixing = 0.05
    return s


def adf_frag2_settings() -> Settings:
    """Fragment B: doublet with β excess (-1) for anti-FM BS coupling with A."""
    s = adf_full_settings()
    s.input.adf.unrestricted = "yes"
    s.input.adf.spinpolarization = -1
    s.input.adf.scf.mixing = 0.05
    return s


def adf_full_bs_settings() -> Settings:
    """Full complex: BS-singlet (Sz=0) from anti-FM coupling of two doublets."""
    s = adf_full_settings()
    s.input.adf.unrestricted = "yes"
    s.input.adf.unrestrictedfragments = "yes"
    s.input.adf.spinpolarization = 0
    s.input.adf.scf.mixing = 0.05
    return s


def opt_settings_for_frag(sign: int) -> Settings:
    """Isolated fragment GeoOpt with given spin sign (+1 for A, -1 for B)."""
    s = adf_full_settings()
    s.input.adf.unrestricted = "yes"
    s.input.adf.spinpolarization = sign
    s.input.adf.scf.mixing = 0.05
    s.input.ams.Task = "GeometryOptimization"
    return s


# ---------- I/O -------------------------------------------------------------

def mol_from_xyz(path: Path) -> Molecule:
    m = Molecule()
    lines = path.read_text().splitlines()
    n = int(lines[0])
    for L in lines[2 : 2 + n]:
        parts = L.split()
        m.add_atom(Atom(symbol=parts[0],
                        coords=(float(parts[1]), float(parts[2]), float(parts[3]))))
    return m


# ---------- main routine ---------------------------------------------------

def run_one(rxn_id: str, workdir: Path) -> dict:
    eda_dir = RUNS / rxn_id / "eda"
    sentinel = eda_dir / ".done_eda"
    if sentinel.exists():
        print(f"[{rxn_id}] .done_eda exists; skipping")
        return json.loads((eda_dir / "asr_vector.json").read_text())

    molA = mol_from_xyz(eda_dir / "frag_A.xyz")
    molB = mol_from_xyz(eda_dir / "frag_B.xyz")

    init(path=str(workdir), folder=f"plams_{rxn_id}")
    config.preview = False

    # --- Step 1: isolated fragment GeometryOptimization (gives molA_opt, molB_opt)
    optA = AMSJob(molecule=molA, settings=opt_settings_for_frag(+1), name="optA")
    print(f"[{rxn_id}] running optA (fragment A relax in isolation, doublet)")
    optA_res = optA.run()
    molA_opt = optA_res.get_main_molecule()

    optB = AMSJob(molecule=molB, settings=opt_settings_for_frag(-1), name="optB")
    print(f"[{rxn_id}] running optB (fragment B relax in isolation, doublet ↓)")
    optB_res = optB.run()
    molB_opt = optB_res.get_main_molecule()

    # --- Step 2: ADFFragmentJob — runs 5 children: f1, f2, full, f1_opt, f2_opt
    # Full job uses BS-singlet coupling (Sz=0) of two anti-parallel doublets.
    eda = ADFFragmentJob(
        fragment1=molA,
        fragment2=molB,
        fragment1_opt=molA_opt,
        fragment2_opt=molB_opt,
        full_settings=adf_full_bs_settings(),
        frag1_settings=adf_frag1_settings(),
        frag2_settings=adf_frag2_settings(),
        name=f"{rxn_id}_eda",
    )
    print(f"[{rxn_id}] running ADFFragmentJob (5 children)")
    eda_res = eda.run()

    # --- Step 3: extract EDA channels (kcal/mol).
    # PLAMS recipe `get_energy_decomposition` has a split_position bug: only
    # E_Pauli is parsed correctly (it has more words before the value). For
    # others, PLAMS picks the kcal/mol column then erroneously converts au→kcal/mol.
    # Parse the full.out 4-column block ourselves: au, eV, kcal/mol, kJ/mol.
    import re as _re

    full_out_path = Path(eda.full.path) / "full.out"
    text = full_out_path.read_text(errors="ignore")

    def grep_kcal(pattern: str, *, last: bool = True) -> float:
        """Find lines matching pattern with 4 trailing floats and return col 3 (kcal/mol)."""
        rx = _re.compile(pattern + r".*?(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$",
                         _re.MULTILINE)
        matches = list(rx.finditer(text))
        if not matches:
            return float("nan")
        m = matches[-1] if last else matches[0]
        return float(m.group(3))  # kcal/mol column

    E_Pauli = grep_kcal(r"Pauli Repulsion \(Delta E\^Pauli\):")
    E_elstat = grep_kcal(r"^\s+Electrostatic Interaction:")
    E_orb = grep_kcal(r"Total Orbital Interactions:")
    E_disp = grep_kcal(r"Dispersion Energy:")
    E_int_total = grep_kcal(r"Total Bonding Energy:")

    # Fragment SP energies at TS geom (these are total energies, not interaction)
    E_A_TS = eda.f1.results.get_energy(unit="kcal/mol")
    E_B_TS = eda.f2.results.get_energy(unit="kcal/mol")

    # f1_opt and f2_opt are children of the eda job
    E_A_relax = eda.f1_opt.results.get_energy(unit="kcal/mol")
    E_B_relax = eda.f2_opt.results.get_energy(unit="kcal/mol")

    finish()

    dEprep_A = E_A_TS - E_A_relax
    dEprep_B = E_B_TS - E_B_relax
    E_strain = dEprep_A + dEprep_B

    E_int_sum = E_Pauli + E_elstat + E_orb + E_disp
    dE_frag_barrier_eda = E_strain + E_int_sum

    asr = {
        "rxn_id": rxn_id,
        "level": "ZORA-BLYP-D3(BJ)/TZ2P all-electron NOSYM Good (ADF)",
        # ASR 5-vector (kcal/mol)
        "E_strain": E_strain,
        "E_Pauli": E_Pauli,
        "E_elstat": E_elstat,
        "E_oi": E_orb,
        "E_disp": E_disp,
        # Derived
        "E_int": E_int_sum,
        "E_int_total_kcal": E_int_total,  # ADF's own total (should match E_int_sum)
        "dE_frag_barrier_eda": dE_frag_barrier_eda,
        # Intermediates
        "E_A_TS_kcal": E_A_TS,
        "E_B_TS_kcal": E_B_TS,
        "E_A_relax_kcal": E_A_relax,
        "E_B_relax_kcal": E_B_relax,
        "dEprep_A_kcal": dEprep_A,
        "dEprep_B_kcal": dEprep_B,
    }
    (eda_dir / "asr_vector.json").write_text(json.dumps(asr, indent=2))
    sentinel.touch()

    print(f"[{rxn_id}] ASR vector (kcal/mol):")
    for k in ("E_strain", "E_Pauli", "E_elstat", "E_oi", "E_disp", "E_int", "dE_frag_barrier_eda"):
        v = asr[k]
        print(f"  {k:20s} = {v:>10.3f}")
    return asr


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--id", required=True)
    p.add_argument("--workdir", default=None)
    args = p.parse_args()

    wd = Path(args.workdir) if args.workdir else Path(
        os.environ.get("TMPDIR", f"/gpfs/tmp_cpu2/plams_{os.environ.get('USER', 'yeseo1ee')}")
    )
    wd.mkdir(parents=True, exist_ok=True)
    run_one(args.id, wd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
