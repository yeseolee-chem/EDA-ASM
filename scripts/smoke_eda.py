"""Smoke test for AMS EDA-NOCV pipeline using H2O dimer.

Two closed-shell H2O monomers, then EDA-NOCV on the H-bonded complex.
Closed-shell throughout, avoiding open-shell coupling complications that
arise in the ethane->2CH3 test. The real 500-reaction set is mostly
closed-shell (mult=1), so this is more representative.

If this completes, the EDA-NOCV machinery is wired correctly.

Run:
    module load mpi/2021.9.0
    source $HOME/ams2026.103/amsbashrc.sh
    NSCM=1 $AMSBIN/amspython scripts/smoke_eda.py
"""
import json
from pathlib import Path

from scm.plams import AMSJob, Atom, Molecule, Settings, init, finish


HARTREE_TO_EV = 27.2114079527

OUT_DIR = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/outputs/stage5b/smoke_eda")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def base_settings(has_heavy_relativistic: bool = False) -> Settings:
    s = Settings()
    s.input.ams.Task = "SinglePoint"
    s.input.adf.XC.GGA = "BLYP"
    s.input.adf.XC.Dispersion = "Grimme3 BJDAMP"
    s.input.adf.Basis.Type = "TZ2P"
    s.input.adf.Basis.Core = "None"
    s.input.adf.NumericalQuality = "Good"
    s.input.adf.Symmetry = "NoSym"
    s.input.adf.SCF.Iterations = 300
    s.input.adf.SCF.Converge = "1e-6"
    if has_heavy_relativistic:
        s.input.adf.Relativity.Level = "Scalar"
        s.input.adf.Relativity.Formalism = "ZORA"
    return s


def _mol(atoms_xyz):
    mol = Molecule()
    for sym, x, y, z in atoms_xyz:
        mol.add_atom(Atom(symbol=sym, coords=(x, y, z)))
    return mol


def water_a() -> Molecule:
    """Water donor (H-bond donor)."""
    return _mol([
        ("O", 0.000,  0.000,  0.000),
        ("H", 0.957,  0.000,  0.000),
        ("H", -0.240, 0.927,  0.000),
    ])


def water_b() -> Molecule:
    """Water acceptor, ~1.85 A H-bond from water_a's H1."""
    return _mol([
        ("O", 2.80,  0.000,  0.000),
        ("H", 3.04,  0.927,  0.000),
        ("H", 3.04, -0.464,  0.802),
    ])


def water_dimer() -> Molecule:
    """Concatenated geometry: atoms 1-3 = donor, atoms 4-6 = acceptor."""
    return _mol([
        ("O", 0.000,  0.000,  0.000),
        ("H", 0.957,  0.000,  0.000),
        ("H", -0.240, 0.927,  0.000),
        ("O", 2.80,  0.000,  0.000),
        ("H", 3.04,  0.927,  0.000),
        ("H", 3.04, -0.464,  0.802),
    ])


def main():
    init(folder=str(OUT_DIR / "plams_workdir"), use_existing_folder=True)
    try:
        # 1) Two H2O fragments at the geometry they hold in the dimer
        s_h2o = base_settings()  # closed shell singlet, restricted
        job_a = AMSJob(molecule=water_a(), settings=s_h2o, name="frag_A_H2O")
        job_b = AMSJob(molecule=water_b(), settings=s_h2o, name="frag_B_H2O")
        res_a = job_a.run()
        res_b = job_b.run()
        e_a = res_a.get_energy(unit="eV")
        e_b = res_b.get_energy(unit="eV")
        print(f"[OK] E(H2O_A) = {e_a:.4f} eV")
        print(f"[OK] E(H2O_B) = {e_b:.4f} eV")

        rkf_a = res_a.rkfpath(file="adf")
        rkf_b = res_b.rkfpath(file="adf")
        print(f"[OK] rkf A: {rkf_a}")
        print(f"[OK] rkf B: {rkf_b}")

        # 2) Complex with fragment-based EDA-NOCV
        s_complex = base_settings()
        s_complex.input.adf.Fragments.frag_A = str(rkf_a)
        s_complex.input.adf.Fragments.frag_B = str(rkf_b)
        s_complex.input.adf.ETSNOCV.Enabled = "Yes"
        s_complex.input.adf.ETSNOCV.ENOCV = 0.01
        s_complex.input.adf.Print = "ETSLOWDIN"

        mol = water_dimer()
        for i, at in enumerate(mol):
            at.properties.suffix = "adf.f=frag_A" if i < 3 else "adf.f=frag_B"

        job_c = AMSJob(molecule=mol, settings=s_complex, name="complex_EDA")
        res_c = job_c.run()
        e_c = res_c.get_energy(unit="eV")
        print(f"[OK] E(H2O dimer w/ EDA) = {e_c:.4f} eV")

        # 3) Discover EDA component keys in the rkf
        from scm.plams import KFFile
        rkf_complex = res_c.rkfpath(file="adf")
        kf = KFFile(rkf_complex)
        sections = kf.sections()
        all_keys = []
        for sec in sections:
            for var in kf.read_section(sec):
                all_keys.append((sec, var))

        # Dump for review
        dump_path = OUT_DIR / "rkf_key_dump.txt"
        with open(dump_path, "w") as f:
            for sec, var in sorted(all_keys):
                f.write(f"{sec}::{var}\n")
        print(f"[OK] dumped {len(all_keys)} keys to {dump_path}")

        # Find EDA-like keys
        eda_hints = ["pauli", "elstat", "orbital", "orb int", "dispersion",
                     "bond energy", "interaction", "steric", "kin"]
        hits = []
        for sec, var in all_keys:
            low = var.lower()
            if any(h in low for h in eda_hints):
                try:
                    val = kf.read(sec, var)
                except Exception:
                    val = "<unreadable>"
                hits.append((sec, var, val))

        print(f"\n[INFO] {len(hits)} EDA-candidate keys:")
        for sec, var, val in hits:
            if isinstance(val, float):
                print(f"  {sec}::{var:40s} = {val:14.6f}  ({val * HARTREE_TO_EV:10.4f} eV if Ha)")
            else:
                vrepr = repr(val)[:60]
                print(f"  {sec}::{var:40s} = {vrepr}")

        # 4) Save summary + canonical key map for production
        summary = {
            "status": "ok",
            "E_H2O_A_eV": e_a,
            "E_H2O_B_eV": e_b,
            "E_dimer_complex_eV": e_c,
            "n_rkf_sections": len(sections),
            "n_rkf_keys": len(all_keys),
            "eda_candidate_keys": [
                {"section": sec, "var": var, "value": str(val)}
                for sec, var, val in hits
            ],
        }
        with open(OUT_DIR / "smoke_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[DONE] summary at {OUT_DIR / 'smoke_summary.json'}")

    finally:
        finish()


if __name__ == "__main__":
    main()
