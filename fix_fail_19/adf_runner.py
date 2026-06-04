"""ADF / SCM.PLAMS wrappers — the only place that touches scm.plams APIs.

All other modules in fix_fail_19 import from here. Keeping this module thin
so the pipeline can be tested with monkey-patched stubs (see tests/).
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import Config


@dataclass
class SPResult:
    """Single-point ADF result, JSON-friendly."""
    ok: bool
    energy_kcal: float | None        # total bond energy in kcal/mol
    rkf_path: str | None             # path to ams.rkf (or adf.rkf for EDA)
    s2: float | None = None          # <S²> if available
    error: str = ""


@dataclass
class OptResult:
    """GeometryOptimization result."""
    ok: bool
    energy_kcal: float | None
    final_coords: list[list[float]] | None    # (n_atoms, 3)
    rkf_path: str | None
    converged: bool = False
    error: str = ""


def _import_plams():
    """Late import so callers can monkey-patch with stubs."""
    from scm import plams                                          # type: ignore
    return plams


def _base_settings(cfg: Config):
    """Construct the base ADF Settings object matching cfg."""
    plams = _import_plams()
    s = plams.Settings()
    s.input.ams.Task = "SinglePoint"
    s.input.adf.XC.GGA = cfg.adf_functional
    s.input.adf.XC.Dispersion = "Grimme3 BJDAMP" if cfg.adf_dispersion == "D3BJ" else cfg.adf_dispersion
    s.input.adf.Basis.Type = cfg.adf_basis
    s.input.adf.Basis.Core = "None"
    s.input.adf.NumericalQuality = "Good"
    s.input.adf.Symmetry = "NoSym"
    s.input.adf.SCF.Iterations = 200
    s.input.adf.SCF.Converge = str(cfg.adf_scf_thr)
    if cfg.adf_relativity == "ZORA_scalar":
        s.input.adf.Relativity.Level = "Scalar"
        s.input.adf.Relativity.Formalism = "ZORA"
    return s


def _mol_from(symbols, coords):
    """Build a PLAMS Molecule from symbols + (n,3) coords."""
    plams = _import_plams()
    m = plams.Molecule()
    for sym, (x, y, z) in zip(symbols, coords):
        m.add_atom(plams.Atom(symbol=str(sym), coords=(float(x), float(y), float(z))))
    return m


def _with_spin(s, spin_polarization: int):
    """Clone settings and add Unrestricted + SpinPolarization."""
    plams = _import_plams()
    s2 = s.copy() if hasattr(s, "copy") else plams.Settings(s.as_dict())
    if spin_polarization != 0:
        s2.input.adf.Unrestricted = "Yes"
        s2.input.adf.SpinPolarization = spin_polarization
        s2.input.adf.SCF.Mixing = 0.05
    return s2


def _parse_s2(rkf_path: str | None) -> float | None:
    """Extract <S²> from an ADF rkf if present; None otherwise."""
    if not rkf_path or not os.path.exists(rkf_path):
        return None
    plams = _import_plams()
    try:
        kf = plams.KFFile(rkf_path)
        for section, var in (("Properties", "SpinSquared"),
                              ("General", "S2"),
                              ("Properties", "S-squared")):
            try:
                v = kf.read(section, var)
                if v is not None:
                    return float(v)
            except Exception:
                continue
    except Exception:
        return None
    return None


HA_TO_KCAL = 627.5094740631


def run_sp(symbols, coords, charge: int, spin_polarization: int,
            workdir: str, jobname: str, cfg: Config) -> SPResult:
    """Run one ADF SinglePoint (unrestricted iff spin_polarization != 0).

    Returns SPResult; never raises (failures become ok=False with .error set).
    """
    plams = _import_plams()
    try:
        if not plams.config.get("init_called"):
            plams.init(folder=workdir, use_existing_folder=True)
            plams.config["init_called"] = True
        mol = _mol_from(symbols, coords)
        mol.properties.charge = int(charge)
        s = _with_spin(_base_settings(cfg), spin_polarization)
        job = plams.AMSJob(molecule=mol, settings=s, name=jobname)
        r = job.run()
        if not r.ok():
            return SPResult(ok=False, energy_kcal=None, rkf_path=None,
                            error="SCF/AMS run failed")
        e_kcal = r.get_energy(unit="kcal/mol")
        rkf = r.rkfpath(file="adf")
        return SPResult(ok=True, energy_kcal=float(e_kcal), rkf_path=str(rkf),
                        s2=_parse_s2(rkf))
    except Exception as exc:
        return SPResult(ok=False, energy_kcal=None, rkf_path=None, error=str(exc))


def run_geo_opt(symbols, coords, charge: int, spin_polarization: int,
                 workdir: str, jobname: str, cfg: Config) -> OptResult:
    """Run an ADF GeometryOptimization until cfg.endpoint_grad_thr."""
    plams = _import_plams()
    try:
        if not plams.config.get("init_called"):
            plams.init(folder=workdir, use_existing_folder=True)
            plams.config["init_called"] = True
        mol = _mol_from(symbols, coords)
        mol.properties.charge = int(charge)
        s = _with_spin(_base_settings(cfg), spin_polarization)
        s.input.ams.Task = "GeometryOptimization"
        s.input.ams.GeometryOptimization.MaxIterations = cfg.endpoint_max_step
        s.input.ams.GeometryOptimization.Convergence.Gradients = cfg.endpoint_grad_thr
        job = plams.AMSJob(molecule=mol, settings=s, name=jobname)
        r = job.run()
        if not r.ok():
            return OptResult(ok=False, energy_kcal=None, final_coords=None,
                              rkf_path=None, error="opt run failed")
        e_kcal = r.get_energy(unit="kcal/mol")
        # final geometry from ams.rkf
        rkf = r.rkfpath(file="ams")
        ams = plams.KFFile(rkf)
        try:
            coords_flat = ams.read("Molecule", "Coords")
            n_atoms = ams.read("Molecule", "nAtoms")
            arr = list(coords_flat) if not hasattr(coords_flat, "__iter__") else list(coords_flat)
            arr3 = [[float(arr[3*i]), float(arr[3*i+1]), float(arr[3*i+2])]
                    for i in range(int(n_atoms))]
        except Exception:
            arr3 = None
        try:
            term = ams.read("General", "termination status")
            converged = "NORMAL" in str(term).upper()
        except Exception:
            converged = True
        return OptResult(ok=True, energy_kcal=float(e_kcal), final_coords=arr3,
                          rkf_path=str(rkf), converged=converged)
    except Exception as exc:
        return OptResult(ok=False, energy_kcal=None, final_coords=None,
                          rkf_path=None, error=str(exc))


def run_eda(symbols, coords, charge: int, frags: list[dict],
             frag_states: list[dict], workdir: str, jobname: str,
             cfg: Config) -> dict:
    """Run a coupled EDA-NOCV using prepared fragment .t21 references.

    frags: [{role, atom_indices}, ...].
    frag_states: [{role, t21_path, multiplicity, spin_sign}, ...].
    Returns {ok, components_kcal: {strain,elst,Pauli,oi,disp}, s2, rkf_path, error}.
    The components are extracted from the ADF rkf using the same keys as
    run_asr_spec.parse_eda_kcal.
    """
    plams = _import_plams()
    try:
        if not plams.config.get("init_called"):
            plams.init(folder=workdir, use_existing_folder=True)
            plams.config["init_called"] = True
        mol = _mol_from(symbols, coords)
        mol.properties.charge = int(charge)
        # Tag each atom with its fragment role
        for i, at in enumerate(mol):
            for f in frags:
                if i in f["atom_indices"]:
                    at.properties.suffix = f"adf.f={f['role']}"
                    break
        s = _base_settings(cfg)
        s.input.adf.Unrestricted = "Yes"
        s.input.adf.UnrestrictedFragments = "Yes"
        s.input.adf.Fragments = {fs["role"]: fs["t21_path"] for fs in frag_states}
        s.input.adf.ETSNOCV = ""
        # Total spin polarization
        total_sp = sum(fs.get("spin_sign", 1) * (fs.get("multiplicity", 1) - 1)
                       for fs in frag_states)
        if total_sp != 0:
            s.input.adf.SpinPolarization = total_sp
        job = plams.AMSJob(molecule=mol, settings=s, name=jobname)
        r = job.run()
        if not r.ok():
            return {"ok": False, "error": "EDA run failed"}
        rkf = r.rkfpath(file="adf")
        kf = plams.KFFile(rkf)
        # Extract EDA component energies (same convention as run_asr_spec)
        keys = {
            "int_total": ("Energy", "Bond Energy"),
            "Pauli":     ("Energy", "Pauli Total"),
            "elst":      ("Energy", "Electrostatic Interaction"),
            "oi":        ("Energy", "Orb.Int. Total"),
            "disp":      ("Energy", "Dispersion Energy"),
        }
        comp = {}
        for k, (sec, var) in keys.items():
            try:
                comp[k] = float(kf.read(sec, var)) * HA_TO_KCAL
            except Exception:
                comp[k] = None
        return {"ok": True, "components_kcal": comp, "rkf_path": str(rkf),
                "s2": _parse_s2(str(rkf))}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
