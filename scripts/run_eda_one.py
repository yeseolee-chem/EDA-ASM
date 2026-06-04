"""Per-reaction EDA-ASM workflow (spec §4).

For one stage5a reaction (identified by --rxn_id), this runs:
  A.   3 whole-molecule SPs (R, TS, P)
  B,C. 2 × n_fragments fragment SPs (at R-geom, at TS-geom)
  D,E. 2 EDA-NOCV jobs (complex at R-geom, complex at TS-geom)
       using B/C fragment rkf as references.

Outputs `eda_result.json` per spec §5 schema, including:
  - whole-molecule energies (R / TS / P)
  - per-fragment energies and strain
  - EDA channels at R and at TS
  - GPR labels = Δ(TS − R) of each channel
  - ASM closure validation

Usage:
    module load mpi/2021.9.0
    source $HOME/ams2026.103/amsbashrc.sh
    NSCM=1 $AMSBIN/amspython scripts/run_eda_one.py --rxn_id <ID>
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from scm.plams import AMSJob, Atom, KFFile, Molecule, Settings, init, finish


HA_TO_EV = 27.2114079527

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
STAGE5A_DIR = REPO / "outputs" / "stage5a" / "per_reaction"
STAGE5B_DIR = REPO / "outputs" / "stage5b" / "per_reaction"

# Canonical AMS 2026 EDA rkf keys (discovered by smoke_eda + verified)
EDA_KEY_MAP = {
    "Pauli":     ("Energy", "Pauli Total"),
    "elstat":    ("Energy", "Electrostatic Interaction"),
    "orb":       ("Energy", "Orb.Int. Total"),
    "disp":      ("Energy", "Dispersion Energy"),
    "int_total": ("Energy", "Bond Energy"),
}

# Candidate rkf keys for σ/π orbital decomposition (only present if the
# complex SCF resolves into A'/A'' or A1/A2/B1/B2 irreps — i.e. when
# AUTO-detected symmetry is Cs or C2v). Probed at parse time.
ORB_IRREP_CANDIDATES = {
    "orb_Aprime":  ("Energy", "Orb.Int. A'"),
    "orb_Adblpr":  ("Energy", "Orb.Int. A\""),
    "orb_A1":      ("Energy", "Orb.Int. A1"),
    "orb_A2":      ("Energy", "Orb.Int. A2"),
    "orb_B1":      ("Energy", "Orb.Int. B1"),
    "orb_B2":      ("Energy", "Orb.Int. B2"),
    "orb_A":       ("Energy", "Orb.Int. A"),
}

# Symmetry-related SCF/INPUT error patterns that warrant a NoSym retry.
SYMMETRY_ERROR_PATTERNS = (
    "symmetry",
    "Symmetry",
    "SYMMETRY",
    "point group",
    "irreducible",
)


def load_xyz(path: Path) -> Molecule:
    """Read an XYZ file into a PLAMS Molecule (atoms only; no bonds/charge)."""
    mol = Molecule()
    with open(path) as f:
        n = int(f.readline().strip())
        f.readline()  # comment line
        for _ in range(n):
            parts = f.readline().split()
            sym = parts[0]
            x, y, z = (float(p) for p in parts[1:4])
            mol.add_atom(Atom(symbol=sym, coords=(x, y, z)))
    return mol


def base_settings(symmetry: str = "AUTO") -> Settings:
    """Option-1 ADF settings — guide-compatible where feasible.

    PBE-D3(BJ)/TZ2P, all-electron, ZORA scalar always, VeryGood numerics,
    tight SCF. Symmetry is AUTO by default (caller can pass "NoSym" as
    fallback after a symmetry-related SCF failure).
    """
    s = Settings()
    s.input.ams.Task = "SinglePoint"
    s.input.adf.XC.GGA = "PBE"
    s.input.adf.XC.Dispersion = "Grimme3 BJDAMP"
    s.input.adf.Basis.Type = "TZ2P"
    s.input.adf.Basis.Core = "None"
    s.input.adf.NumericalQuality = "VeryGood"
    s.input.adf.BeckeGrid.Quality = "VeryGood"
    s.input.adf.Symmetry = symmetry
    s.input.adf.SCF.Iterations = 500
    s.input.adf.SCF.Converge = "1e-8 1e-8"
    # ZORA scalar always — for consistency across the full 500-set, since
    # Br/I are present in some fractions; guide §5 recommends always-on.
    s.input.adf.Relativity.Level = "Scalar"
    s.input.adf.Relativity.Formalism = "ZORA"
    return s


def open_shell_settings(multiplicity: int, symmetry: str = "AUTO",
                        spin_sign: int = 1) -> Settings:
    s = base_settings(symmetry)
    s.input.adf.Unrestricted = "Yes"
    s.input.adf.SpinPolarization = spin_sign * (multiplicity - 1)
    s.input.adf.SCF.Mixing = 0.05
    return s


def settings_for(mult: int, symmetry: str = "AUTO",
                 spin_sign: int = 1) -> Settings:
    """Single-point settings. For open-shell fragments, callers should
    pass spin_sign = +1 for the first and -1 for the second open-shell
    fragment, so the two rkfs hold opposite-spin reference states. When
    these are combined in the complex EDA, antiparallel coupling yields a
    closed-shell singlet (the "UNRESTRICTED FRAGMENT" error otherwise)."""
    if mult > 1:
        return open_shell_settings(mult, symmetry, spin_sign)
    return base_settings(symmetry)


def assign_spin_signs(fragments):
    """Alternate +1/-1 sign across open-shell fragments only.

    closed-shell (mult=1) fragments get sign=+1 but contribute SpinPol=0.
    """
    signs = []
    open_idx = 0
    for f in fragments:
        if f["multiplicity"] > 1:
            signs.append(1 if open_idx % 2 == 0 else -1)
            open_idx += 1
        else:
            signs.append(1)
    return signs


def parse_eda(rkf_path: str) -> dict:
    """Read EDA components from an ADF complex-EDA rkf. Returns eV.

    Always extracts the 5 core channels. Additionally probes for σ/π
    orbital decomposition keys (A'/A'' for Cs, A1/A2/B1/B2 for C2v) and
    includes any that are present.
    """
    kf = KFFile(rkf_path)
    out = {}
    for channel, (sec, var) in EDA_KEY_MAP.items():
        val_ha = kf.read(sec, var)
        out[channel] = val_ha * HA_TO_EV
    # σ/π breakdown if symmetry-resolved
    irrep_decomp = {}
    for label, (sec, var) in ORB_IRREP_CANDIDATES.items():
        try:
            val_ha = kf.read(sec, var)
            irrep_decomp[label] = val_ha * HA_TO_EV
        except Exception:
            pass
    if irrep_decomp:
        out["orb_irrep_decomposition_eV"] = irrep_decomp
    return out


def _looks_symmetry_related(err_text: str) -> bool:
    return any(p in err_text for p in SYMMETRY_ERROR_PATTERNS)


def read_s2(rkf_path: str) -> float | None:
    """Read ⟨S²⟩ from an unrestricted-fragment rkf. Returns None if absent."""
    try:
        kf = KFFile(rkf_path)
        # AMS 2026 stores <S²> under "Properties" or "S2"
        for sec, var in [("Properties", "SpinSquared"),
                         ("Properties", "<S^2>"),
                         ("General", "S2")]:
            try:
                return float(kf.read(sec, var))
            except Exception:
                continue
    except Exception:
        return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rxn_id", required=True)
    args = ap.parse_args()
    rxn_id = args.rxn_id

    in_dir = STAGE5A_DIR / rxn_id
    out_dir = STAGE5B_DIR / rxn_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Idempotent: if a complete result.json already exists, skip.
    result_path = out_dir / "eda_result.json"
    if result_path.exists():
        try:
            with open(result_path) as f:
                prev = json.load(f)
            if prev.get("metadata", {}).get("status") == "ok":
                print(f"[SKIP] {rxn_id} already completed "
                      f"(Ea_adf={prev['whole_eV'].get('Ea_adf')} eV, "
                      f"closure={prev['validation'].get('asm_closure_eV')} eV)")
                return
            else:
                print(f"[RETRY] {rxn_id} previous status={prev.get('metadata', {}).get('status')}, re-running")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[RETRY] {rxn_id} cannot parse previous result ({e}), re-running")

    with open(in_dir / "result.json") as f:
        meta = json.load(f)
    pattern = meta["result"]["pattern"]
    fragments = meta["result"]["fragments"]

    r_mol = load_xyz(in_dir / "R.xyz")
    symbols = sorted({a.symbol for a in r_mol})

    print(f"[{rxn_id}]")
    print(f"  pattern   = {pattern}")
    print(f"  formula   = {meta.get('formula', '?')}  natoms={meta['n_atoms']}")
    print(f"  fragments = {[f['role'] for f in fragments]} mult={[f['multiplicity'] for f in fragments]}")
    print(f"  elements  = {symbols}")

    t0 = time.time()
    init(folder=str(out_dir / "plams_workdir"), use_existing_folder=True)

    # Track which symmetry each job actually used (AUTO vs NoSym fallback)
    sym_used = {"whole_R": None, "whole_TS": None, "whole_P": None}
    s2_diag = {}  # ⟨S²⟩ per fragment job (when unrestricted)

    def run_with_fallback(mol, mult, name, spin_sign=1):
        """Run AMSJob; if it fails with a symmetry-related error, retry NoSym."""
        for sym_try in ("AUTO", "NoSym"):
            s = settings_for(mult, sym_try, spin_sign)
            job = AMSJob(molecule=mol, settings=s, name=name if sym_try == "AUTO" else name + "_NoSym")
            res = job.run()
            if res.ok():
                return res, sym_try
            # Failed — check error
            err_files = []
            try:
                wd = Path(job.path)
                for ef in wd.glob("*.err"):
                    err_files.append(ef.read_text(errors="ignore"))
                err_text = " ".join(err_files)
            except Exception:
                err_text = ""
            if sym_try == "AUTO" and _looks_symmetry_related(err_text):
                print(f"    [retry NoSym] {name}: symmetry-related failure, falling back")
                continue
            return res, sym_try  # genuine failure, return the failed result
        return None, None

    try:
        # ===== A. Whole molecule SPs at R, TS, P =====
        whole_E = {}
        for state in ["R", "TS", "P"]:
            mol = load_xyz(in_dir / f"{state}.xyz")
            n_e = sum(a.atnum for a in mol)
            mult = 1 if n_e % 2 == 0 else 2
            res, used = run_with_fallback(mol, mult, f"whole_{state}")
            sym_used[f"whole_{state}"] = used
            if res is not None and res.ok():
                whole_E[state] = res.get_energy(unit="eV")
                print(f"  [A] whole {state}: {whole_E[state]:.4f} eV (mult={mult}, sym={used})")
            else:
                whole_E[state] = None
                print(f"  [A] whole {state}: FAILED")

        # ===== B,C. Fragment SPs at R-geom and TS-geom =====
        # Pre-compute per-fragment SpinPolarization sign so two open-shell
        # fragments combine into a closed-shell singlet (alternating ±).
        spin_signs = assign_spin_signs(fragments)
        print(f"  spin-sign assignment: " + ", ".join(
            f"{f['role']}=mult{f['multiplicity']}/{'+' if s>0 else '-'}{abs(f['multiplicity']-1)}"
            for f, s in zip(fragments, spin_signs)
        ))
        frag_rkf = {"R": {}, "TS": {}}
        frag_E = {"R": {}, "TS": {}}
        for state in ["R", "TS"]:
            for f, ssign in zip(fragments, spin_signs):
                role = f["role"]
                mult = f["multiplicity"]
                fpath = in_dir / f"{role}_{state}.xyz"
                if not fpath.exists():
                    print(f"  [B/C] {role} @ {state}: xyz file missing — {fpath}")
                    continue
                mol = load_xyz(fpath)
                jname = f"frag_{role}_{state}"
                res, used = run_with_fallback(mol, mult, jname, spin_sign=ssign)
                if res is not None and res.ok():
                    frag_E[state][role] = res.get_energy(unit="eV")
                    frag_rkf[state][role] = res.rkfpath(file="adf")
                    if mult > 1:
                        s2 = read_s2(frag_rkf[state][role])
                        if s2 is not None:
                            s2_diag[jname] = s2
                    print(f"  [{state}] frag {role}: {frag_E[state][role]:.4f} eV "
                          f"(mult={mult}, spin_sign={ssign:+d}, sym={used})")
                else:
                    frag_E[state][role] = None
                    print(f"  [{state}] frag {role}: FAILED")

        # ===== D,E. Complex EDA at R and TS =====
        eda = {"R": {}, "TS": {}}
        eda_sym = {"R": None, "TS": None}
        for state in ["R", "TS"]:
            if any(frag_rkf[state].get(f["role"]) is None for f in fragments):
                print(f"  [{state}] skipping EDA: missing fragment rkf")
                continue
            mol = load_xyz(in_dir / f"{state}.xyz")
            for i, at in enumerate(mol):
                role = None
                for f in fragments:
                    if i in f["atom_indices"]:
                        role = f["role"]
                        break
                if role is None:
                    raise RuntimeError(f"atom {i} not in any fragment for {rxn_id}")
                at.properties.suffix = f"adf.f={role}"

            total_unpaired = sum(f["multiplicity"] - 1 for f in fragments)
            complex_mult = 1 if total_unpaired % 2 == 0 else 2

            # Try AUTO then NoSym for the complex too. For open-shell
            # fragments, the complex needs a per-fragment SpinPolarization
            # Float List (antiparallel coupling). AMS docs explicitly say
            # this is the way to combine unrestricted fragments into a
            # singlet complex without the "UNRESTRICTED FRAGMENT" error.
            for sym_try in ("AUTO", "NoSym"):
                s = settings_for(complex_mult, sym_try, spin_sign=1)
                if total_unpaired > 0 and complex_mult == 1:
                    s.input.adf.Unrestricted = "Yes"
                    # Per-fragment SpinPolarization list (sum must equal complex SP).
                    # spin_signs is aligned with `fragments` order.
                    sp_list = [
                        ssign * (fi["multiplicity"] - 1)
                        for fi, ssign in zip(fragments, spin_signs)
                    ]
                    s.input.adf.SpinPolarization = " ".join(
                        f"{v:.1f}" for v in sp_list
                    )
                    s.input.adf.SCF.Mixing = 0.05
                for f in fragments:
                    s.input.adf.Fragments[f["role"]] = str(frag_rkf[state][f["role"]])
                s.input.adf.ETSNOCV.Enabled = "Yes"
                # AMS 2026 ETSNOCV block uses ENOCV (threshold) — the guide's
                # `EKEEP 5` is pre-2018 syntax and is rejected. ENOCV=0.01
                # keeps all NOCV pairs with |eigenvalue| > 0.01.
                s.input.adf.ETSNOCV.ENOCV = 0.01
                s.input.adf.Print = "ETSLOWDIN"

                name = f"complex_EDA_{state}" if sym_try == "AUTO" else f"complex_EDA_{state}_NoSym"
                job = AMSJob(molecule=mol, settings=s, name=name)
                res = job.run()
                if res.ok():
                    rkf = res.rkfpath(file="adf")
                    eda[state] = parse_eda(rkf)
                    eda_sym[state] = sym_try
                    has_irreps = "orb_irrep_decomposition_eV" in eda[state]
                    print(f"  [{state}] EDA: Pauli={eda[state]['Pauli']:+.4f} "
                          f"elstat={eda[state]['elstat']:+.4f} orb={eda[state]['orb']:+.4f} "
                          f"disp={eda[state]['disp']:+.4f} int={eda[state]['int_total']:+.4f} eV  "
                          f"(sym={sym_try}, σ/π={'yes' if has_irreps else 'no'})")
                    break
                # check if symmetry-related
                err_text = ""
                try:
                    wd = Path(job.path)
                    for ef in wd.glob("*.err"):
                        err_text += ef.read_text(errors="ignore")
                except Exception:
                    pass
                if sym_try == "AUTO" and _looks_symmetry_related(err_text):
                    print(f"    [retry NoSym] complex EDA @ {state}: symmetry fallback")
                    continue
                print(f"  [{state}] EDA: FAILED ({sym_try})")
                break

        # ===== Compute strain, Δ-labels, ASM closure =====
        strain_per_frag = {}
        if all(frag_E["TS"].get(f["role"]) is not None for f in fragments) and \
           all(frag_E["R"].get(f["role"]) is not None for f in fragments):
            for f in fragments:
                role = f["role"]
                strain_per_frag[role] = frag_E["TS"][role] - frag_E["R"][role]
        dE_strain = sum(strain_per_frag.values()) if strain_per_frag else None

        # Δ(TS - R) per channel
        dE = {}
        if eda["R"] and eda["TS"]:
            for ch in EDA_KEY_MAP:
                dE[ch] = eda["TS"][ch] - eda["R"][ch]

        Ea_adf = (whole_E["TS"] - whole_E["R"]) if whole_E.get("R") is not None and whole_E.get("TS") is not None else None
        if dE_strain is not None and dE.get("int_total") is not None and Ea_adf is not None:
            closure = (dE_strain + dE["int_total"]) - Ea_adf
        else:
            closure = None

        # ===== Assemble result =====
        result = {
            "rxn_id": rxn_id,
            "schema_version": "stage5b_v2_option1",
            "pattern": pattern,
            "p2_subtype": meta["result"].get("p2_subtype"),
            "n_fragments": len(fragments),
            "level_of_theory": {
                "functional": "PBE",
                "dispersion": "D3(BJ)",
                "basis": "TZ2P",
                "core": "all-electron",
                "relativity": "ZORA-scalar",
                "software": "AMS 2026.103",
                "symmetry_requested": "AUTO (fallback NoSym)",
                "numerical_quality": "VeryGood",
                "becke_grid_quality": "VeryGood",
                "scf_converge": "1e-8 1e-8",
                "scf_iterations": 500,
                "etsnocv_enocv": 0.01,
            },
            "symmetry_used": {
                **sym_used,
                "complex_EDA_R": eda_sym["R"],
                "complex_EDA_TS": eda_sym["TS"],
            },
            "spin_diagnostics": s2_diag,
            "geometry_source": {
                "level": "wB97X-3c",
                "software": "ORCA 6.0.1",
                "dataset": "Halo8",
                "doi": "10.1038/s41597-025-05944-3",
            },
            "whole_eV": {
                "E_R": whole_E.get("R"),
                "E_TS": whole_E.get("TS"),
                "E_P": whole_E.get("P"),
                "Ea_adf": Ea_adf,
                "dE_rxn_adf": (whole_E["P"] - whole_E["R"])
                              if whole_E.get("R") is not None and whole_E.get("P") is not None else None,
                "Ea_halo8_reference": meta.get("activation_energy"),
                "E_R_halo8_reference": meta.get("energy_R"),
                "E_TS_halo8_reference": meta.get("energy_TS"),
                "E_P_halo8_reference": meta.get("energy_P"),
            },
            "fragments": [
                {
                    "role": f["role"],
                    "multiplicity": f["multiplicity"],
                    "n_atoms": len(f["atom_indices"]),
                    "atom_indices": f["atom_indices"],
                    "E_at_R_eV": frag_E["R"].get(f["role"]),
                    "E_at_TS_eV": frag_E["TS"].get(f["role"]),
                    "strain_eV": strain_per_frag.get(f["role"]),
                }
                for f in fragments
            ],
            "eda_at_R_eV": eda.get("R") or {},
            "eda_at_TS_eV": eda.get("TS") or {},
            "labels_for_gpr_eV": {
                "dE_strain_ddag": dE_strain,
                "dE_Pauli_ddag": dE.get("Pauli"),
                "dE_elstat_ddag": dE.get("elstat"),
                "dE_orb_ddag": dE.get("orb"),
                "dE_disp_ddag": dE.get("disp"),
                "dE_int_ddag": dE.get("int_total"),
            },
            "validation": {
                "asm_closure_eV": closure,
                "asm_closure_within_0p05eV": (abs(closure) < 0.05) if closure is not None else None,
                "eda_sum_check_at_TS_eV": (
                    (eda["TS"]["Pauli"] + eda["TS"]["elstat"]
                     + eda["TS"]["orb"] + eda["TS"]["disp"] - eda["TS"]["int_total"])
                    if eda.get("TS") else None
                ),
                "sign_checks": {
                    "strain_positive": (dE_strain is not None and dE_strain > 0),
                    "pauli_destabilizing": (dE.get("Pauli") is not None and dE["Pauli"] > 0),
                    "elstat_stabilizing":  (dE.get("elstat") is not None and dE["elstat"] < 0),
                    "orb_stabilizing":     (dE.get("orb") is not None and dE["orb"] < 0),
                } if dE else {},
            },
            "metadata": {
                "ams_version": "2026.103",
                "ams_functional": "PBE-D3(BJ)/TZ2P",
                "wall_time_seconds": time.time() - t0,
                "nscm": int(os.environ.get("NSCM", "1")),
                "host": os.environ.get("HOSTNAME", os.uname().nodename),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "status": "ok" if (closure is not None and abs(closure) < 0.05) else "partial",
            },
        }

        out_path = out_dir / "eda_result.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n[DONE] {out_path}")
        print(f"       wall = {result['metadata']['wall_time_seconds']:.1f} s")
        print(f"       Ea_adf = {Ea_adf} eV   closure = {closure} eV")

    finally:
        finish()


if __name__ == "__main__":
    main()
