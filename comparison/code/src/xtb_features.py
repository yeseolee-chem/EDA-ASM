"""GFN2-xTB single-point feature extraction for ASR Δ-baseline.

Per reaction, we compute single-points on three states (all at the TS-frozen
geometry, matching ADF's EDA convention):
    1) complex@TS   — full supermolecule
    2) fragA@TS     — fragment A frozen at its TS positions
    3) fragB@TS     — fragment B frozen at its TS positions

From these we derive scalar features that map onto EDA channels:
    E_int^xtb        = E(complex) − E(fragA) − E(fragB)
    dipole_complex   = ‖μ‖ of the supermolecule
    dipole_frag_diff = ‖μ(complex)‖ − ‖μ(fragA)‖ − ‖μ(fragB)‖
    HOMO/LUMO/gap of complex / fragA / fragB
    charge_transfer  = Σ q(fragA atoms in complex) − charge_a (Mulliken-like)

Caveat: this is a TS-only feature extractor. The full activation-strain
definition (ΔE_strain^xTB = E(fragA@TS) − E(fragA@R) + …) needs relaxed
fragment geometries, which require xtb optimization on the fragment
inputs. That step is intentionally deferred — the spec acknowledges
"strain" is already captured by the geom6 Kabsch-RMSD features, and the
xtb baseline's primary value-add is the **interaction-block** signal
(V_elst / Pauli / E_oi).
"""
from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import ase.io
import numpy as np

# 1 Bohr in Å
BOHR_TO_A = 0.5291772108
HARTREE_TO_KCAL = 627.509474

try:
    from tblite.interface import Calculator
    _HAS_TBLITE = True
except Exception:  # pragma: no cover
    _HAS_TBLITE = False


@dataclass
class XtbResult:
    rid: str
    ok: bool
    fail_reason: str = ""
    # complex
    E_complex_kcal: float = float("nan")
    HOMO_complex: float = float("nan")
    LUMO_complex: float = float("nan")
    dipole_complex_norm: float = float("nan")
    # fragA
    E_fragA_kcal: float = float("nan")
    HOMO_fragA: float = float("nan")
    LUMO_fragA: float = float("nan")
    dipole_fragA_norm: float = float("nan")
    # fragB
    E_fragB_kcal: float = float("nan")
    HOMO_fragB: float = float("nan")
    LUMO_fragB: float = float("nan")
    dipole_fragB_norm: float = float("nan")
    # derived
    E_int_kcal: float = float("nan")
    gap_complex: float = float("nan")
    gap_fragA: float = float("nan")
    gap_fragB: float = float("nan")
    dipole_int: float = float("nan")
    sum_q_A_frag_atoms: float = float("nan")  # charges on first n_A atoms
    n_atoms: int = 0

    def to_dict(self) -> dict:
        return self.__dict__


def _run_one(numbers: np.ndarray, coords_A: np.ndarray, charge: int, mult: int,
             want_bond_orders: bool = False) -> dict:
    """Single-point GFN2-xTB. coords_A in Å, returned energies in kcal/mol."""
    uhf = max(0, mult - 1)  # unpaired electrons = mult - 1
    calc = Calculator("GFN2-xTB", numbers, coords_A / BOHR_TO_A,
                      charge=float(charge), uhf=int(uhf))
    calc.set("verbosity", 0)
    calc.set("max-iter", 250)
    res = calc.singlepoint()
    E_h = res.get("energy")
    orb_E = res.get("orbital-energies")
    occ = res.get("orbital-occupations")
    # HOMO/LUMO indices: largest occ > 1 (closed-shell) or > 0 (alpha for open-shell)
    occ_arr = np.asarray(occ).ravel()
    homo_idx = int(np.where(occ_arr > 1e-3)[0].max()) if (occ_arr > 1e-3).any() else 0
    lumo_idx = homo_idx + 1 if homo_idx + 1 < len(orb_E) else homo_idx
    HOMO = float(orb_E[homo_idx])
    LUMO = float(orb_E[lumo_idx])
    dipole = np.asarray(res.get("dipole")).ravel()
    charges = np.asarray(res.get("charges")).ravel()
    out = dict(
        E_kcal=float(E_h * HARTREE_TO_KCAL),
        HOMO_h=HOMO, LUMO_h=LUMO,
        dipole_norm=float(np.linalg.norm(dipole)),
        charges=charges,
    )
    if want_bond_orders:
        bo = np.asarray(res.get("bond-orders"))
        # tblite returns (natoms, natoms, nspin); collapse spin axis if present.
        if bo.ndim == 3:
            bo = bo.sum(axis=-1)
        out["bond_orders"] = bo
    return out


def compute_xtb_features(reaction, max_atoms: int = 200) -> XtbResult:
    """Run the 3 single-points and pack into XtbResult."""
    if not _HAS_TBLITE:
        return XtbResult(rid=reaction.rid, ok=False, fail_reason="no_tblite")

    try:
        ts_atoms = ase.io.read(reaction.ts_xyz)
        a_atoms = ase.io.read(reaction.frag_a_xyz)
        b_atoms = ase.io.read(reaction.frag_b_xyz)
    except Exception as e:
        return XtbResult(rid=reaction.rid, ok=False, fail_reason=f"xyz_read:{e}")

    n_a, n_b = len(a_atoms), len(b_atoms)
    n_ts = len(ts_atoms)
    if n_ts > max_atoms:
        return XtbResult(rid=reaction.rid, ok=False,
                         fail_reason=f"too_big:{n_ts}>{max_atoms}", n_atoms=n_ts)

    out = XtbResult(rid=reaction.rid, ok=True, n_atoms=n_ts)

    try:
        rc = _run_one(ts_atoms.numbers, ts_atoms.get_positions(),
                      charge=reaction.total_charge, mult=1)
    except Exception as e:
        return XtbResult(rid=reaction.rid, ok=False, fail_reason=f"complex_scf:{e}", n_atoms=n_ts)
    out.E_complex_kcal = rc["E_kcal"]
    out.HOMO_complex = rc["HOMO_h"]; out.LUMO_complex = rc["LUMO_h"]
    out.gap_complex = rc["LUMO_h"] - rc["HOMO_h"]
    out.dipole_complex_norm = rc["dipole_norm"]
    q_complex = rc["charges"]

    try:
        ra = _run_one(a_atoms.numbers, a_atoms.get_positions(),
                      charge=reaction.charge_a, mult=reaction.mult_a)
    except Exception as e:
        out.ok = False; out.fail_reason = f"fragA_scf:{e}"; return out
    out.E_fragA_kcal = ra["E_kcal"]
    out.HOMO_fragA = ra["HOMO_h"]; out.LUMO_fragA = ra["LUMO_h"]
    out.gap_fragA = ra["LUMO_h"] - ra["HOMO_h"]
    out.dipole_fragA_norm = ra["dipole_norm"]

    try:
        rb = _run_one(b_atoms.numbers, b_atoms.get_positions(),
                      charge=reaction.charge_b, mult=reaction.mult_b)
    except Exception as e:
        out.ok = False; out.fail_reason = f"fragB_scf:{e}"; return out
    out.E_fragB_kcal = rb["E_kcal"]
    out.HOMO_fragB = rb["HOMO_h"]; out.LUMO_fragB = rb["LUMO_h"]
    out.gap_fragB = rb["LUMO_h"] - rb["HOMO_h"]
    out.dipole_fragB_norm = rb["dipole_norm"]

    out.E_int_kcal = out.E_complex_kcal - out.E_fragA_kcal - out.E_fragB_kcal
    out.dipole_int = out.dipole_complex_norm - out.dipole_fragA_norm - out.dipole_fragB_norm
    # First n_a atoms in TS xyz correspond to fragA (the EDA atom_permutation
    # already groups frag atoms; geometry_fragA.xyz matches the leading slice).
    out.sum_q_A_frag_atoms = float(q_complex[:n_a].sum())
    return out


# === SPEC_xtb_descriptor_expansion §3 — 6 new scalar descriptors =============
# Derived from a single TS-complex SP that also exposes Mulliken charges and
# Wiberg bond orders (tblite "bond-orders" result field).

@dataclass
class XtbExtraResult:
    rid: str
    ok: bool
    fail_reason: str = ""
    # All scalars below are evaluated at the TS-frozen complex geometry,
    # except xtb_dwbo_interfrag which probes inter-fragment WBO (a R-state
    # proxy that needs no extra SP — see SPEC §3 note on strain).
    xtb_gap: float = float("nan")        # ε_LUMO − ε_HOMO   (Hartree)
    xtb_mu: float = float("nan")         # (ε_HOMO + ε_LUMO)/2
    xtb_omega: float = float("nan")      # μ² / (2η),  η = gap/2  (Parr electrophilicity)
    xtb_dipole: float = float("nan")     # ‖μ_vec‖
    xtb_qpol: float = float("nan")       # Σ_a q_a² (charge-polarisation)
    xtb_dwbo_interfrag: float = float("nan")  # Σ_{a∈A, b∈B} WBO_ab(TS)
    n_atoms_a: int = 0
    n_atoms_b: int = 0
    wall_s: float = 0.0

    def to_dict(self) -> dict:
        return self.__dict__


def compute_xtb_extra(reaction, max_atoms: int = 200) -> XtbExtraResult:
    """Run one TS-complex GFN2-xTB SP and derive the 6 new SPEC scalars.

    Only one SP per reaction (the complex@TS); fragment SPs are not re-run.
    Bond-order matrix is requested from tblite Result.
    """
    if not _HAS_TBLITE:
        return XtbExtraResult(rid=reaction.rid, ok=False, fail_reason="no_tblite")

    import time as _t
    t0 = _t.time()
    try:
        ts_atoms = ase.io.read(reaction.ts_xyz)
        a_atoms = ase.io.read(reaction.frag_a_xyz)
        b_atoms = ase.io.read(reaction.frag_b_xyz)
    except Exception as e:
        return XtbExtraResult(rid=reaction.rid, ok=False, fail_reason=f"xyz_read:{e}")

    n_a, n_b = len(a_atoms), len(b_atoms)
    n_ts = len(ts_atoms)
    if n_ts > max_atoms:
        return XtbExtraResult(rid=reaction.rid, ok=False,
                              fail_reason=f"too_big:{n_ts}>{max_atoms}",
                              n_atoms_a=n_a, n_atoms_b=n_b)

    try:
        rc = _run_one(ts_atoms.numbers, ts_atoms.get_positions(),
                      charge=reaction.total_charge, mult=1,
                      want_bond_orders=True)
    except Exception as e:
        return XtbExtraResult(rid=reaction.rid, ok=False,
                              fail_reason=f"complex_scf:{e}",
                              n_atoms_a=n_a, n_atoms_b=n_b,
                              wall_s=round(_t.time() - t0, 3))

    HOMO = rc["HOMO_h"]; LUMO = rc["LUMO_h"]
    gap = LUMO - HOMO
    mu = 0.5 * (HOMO + LUMO)
    eta = 0.5 * gap
    # electrophilicity ω = μ²/(2η); guard against tiny gap.
    omega = float((mu * mu) / (2.0 * eta)) if abs(eta) > 1e-8 else float("nan")

    charges = np.asarray(rc["charges"]).ravel()
    qpol = float(np.sum(charges * charges))

    bo = np.asarray(rc.get("bond_orders"))
    # Inter-fragment WBO sum: rows ∈ fragA atoms, cols ∈ fragB atoms.
    # The TS xyz orders fragA atoms first (n_a), then fragB (n_b); matches
    # the same convention used for sum_q_A_frag_atoms upstream.
    if bo.shape[0] >= n_a + n_b:
        dwbo_inter = float(np.abs(bo[:n_a, n_a:n_a + n_b]).sum())
    else:
        dwbo_inter = float("nan")

    return XtbExtraResult(
        rid=reaction.rid, ok=True,
        xtb_gap=float(gap), xtb_mu=float(mu), xtb_omega=omega,
        xtb_dipole=float(rc["dipole_norm"]),
        xtb_qpol=qpol,
        xtb_dwbo_interfrag=dwbo_inter,
        n_atoms_a=n_a, n_atoms_b=n_b,
        wall_s=round(_t.time() - t0, 3),
    )


def parse_orca_eda_inp(path: str | Path) -> dict:
    """Parse an ORCA EDA-NOCV input file and return (numbers, coords_A,
    total_charge, frag_a_indices, frag_b_indices).

    Expected layout:
        ! BLYP D3BJ def2-TZVP NoSym EDA TightSCF
        %EDA
          FRAG1_C <int>
          FRAG1_M <int>
          FRAG2_C <int>
          FRAG2_M <int>
        end
        * xyz <total_charge> <total_mult>
         <Elem>(<frag_idx>)  x  y  z
         ...
        *
    """
    import re
    from ase.data import atomic_numbers

    text = Path(path).read_text()
    # Extract * xyz <charge> <mult>  ... *  block
    m = re.search(r"\*\s*xyz\s+(-?\d+)\s+(\d+)\s*\n(.*?)\n\s*\*", text, re.DOTALL)
    if not m:
        raise ValueError(f"no xyz block in {path}")
    total_charge = int(m.group(1))
    atom_block = m.group(3)

    nums, coords = [], []
    frag_a_idx, frag_b_idx = [], []
    for i, line in enumerate(atom_block.strip().splitlines()):
        # Format: " C(2)  -0.10  -0.00  -0.42"
        mm = re.match(r"\s*([A-Z][a-z]?)\(([12])\)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)", line)
        if not mm:
            raise ValueError(f"line {i} parse fail: {line!r}")
        elem, frag, x, y, z = mm.group(1), int(mm.group(2)), float(mm.group(3)), float(mm.group(4)), float(mm.group(5))
        nums.append(atomic_numbers[elem])
        coords.append([x, y, z])
        (frag_a_idx if frag == 1 else frag_b_idx).append(i)
    return dict(
        numbers=np.asarray(nums, dtype=int),
        coords_A=np.asarray(coords, dtype=float),
        total_charge=total_charge,
        frag_a_indices=frag_a_idx,
        frag_b_indices=frag_b_idx,
    )


def compute_xtb_extra_from_orca_inp(rid: str, eda_inp: str | Path, max_atoms: int = 200) -> XtbExtraResult:
    """Run xtb SP on the TS-complex geometry stored in an ORCA EDA input file.

    Use this when neither the canonical ADF source_dir (with status.json +
    geometry_fragA.xyz) nor the orca_eda_label staging copy survives — but
    `runs/orca_recompute/inputs/<rid>/eda.inp` still contains the geometry +
    fragment assignment as written by the ORCA recompute pipeline.
    """
    if not _HAS_TBLITE:
        return XtbExtraResult(rid=rid, ok=False, fail_reason="no_tblite")
    import time as _t
    t0 = _t.time()
    try:
        parsed = parse_orca_eda_inp(eda_inp)
    except Exception as e:
        return XtbExtraResult(rid=rid, ok=False, fail_reason=f"orca_inp_parse:{e}")

    numbers = parsed["numbers"]
    coords_A = parsed["coords_A"]
    idx_a = parsed["frag_a_indices"]
    idx_b = parsed["frag_b_indices"]
    n_a, n_b = len(idx_a), len(idx_b)
    n_ts = len(numbers)
    if n_ts > max_atoms:
        return XtbExtraResult(rid=rid, ok=False,
                              fail_reason=f"too_big:{n_ts}>{max_atoms}",
                              n_atoms_a=n_a, n_atoms_b=n_b)

    try:
        rc = _run_one(numbers, coords_A,
                      charge=parsed["total_charge"], mult=1,
                      want_bond_orders=True)
    except Exception as e:
        return XtbExtraResult(rid=rid, ok=False, fail_reason=f"complex_scf:{e}",
                              n_atoms_a=n_a, n_atoms_b=n_b,
                              wall_s=round(_t.time() - t0, 3))

    HOMO = rc["HOMO_h"]; LUMO = rc["LUMO_h"]
    gap = LUMO - HOMO
    mu = 0.5 * (HOMO + LUMO)
    eta = 0.5 * gap
    omega = float((mu * mu) / (2.0 * eta)) if abs(eta) > 1e-8 else float("nan")
    charges = np.asarray(rc["charges"]).ravel()
    qpol = float(np.sum(charges * charges))
    bo = np.asarray(rc.get("bond_orders"))
    if bo.shape[0] >= max(max(idx_a, default=-1), max(idx_b, default=-1)) + 1:
        dwbo_inter = float(np.abs(bo[np.ix_(idx_a, idx_b)]).sum())
    else:
        dwbo_inter = float("nan")
    return XtbExtraResult(
        rid=rid, ok=True,
        xtb_gap=float(gap), xtb_mu=float(mu), xtb_omega=omega,
        xtb_dipole=float(rc["dipole_norm"]),
        xtb_qpol=qpol, xtb_dwbo_interfrag=dwbo_inter,
        n_atoms_a=n_a, n_atoms_b=n_b,
        wall_s=round(_t.time() - t0, 3),
    )


def compute_xtb_extra_from_status(
    rid: str, ts_xyz: str | Path, status_json: str | Path, max_atoms: int = 200,
) -> XtbExtraResult:
    """Variant of compute_xtb_extra that does NOT require fragment xyz files.

    Reads fragment atom indices directly from status.json:
        status["fragment_atoms_a"], status["fragment_atoms_b"]  (lists of int)
        status["total_charge"]                                  (int)

    The TS atoms keep their original ts.xyz ordering; the bond-orders matrix
    is sliced by the explicit indices, so this also works for reactions where
    fragA/B atoms are NOT contiguous in ts.xyz (e.g. several qmrxn20_e2 cases).

    Use this when the original `geometry_fragA.xyz`/`geometry_fragB.xyz` files
    are missing (archive directories deleted) but the staging copy of
    status.json + ts.xyz survives.
    """
    if not _HAS_TBLITE:
        return XtbExtraResult(rid=rid, ok=False, fail_reason="no_tblite")

    import time as _t
    t0 = _t.time()
    try:
        with open(status_json) as f:
            status = json.load(f)
        ts_atoms = ase.io.read(str(ts_xyz))
    except Exception as e:
        return XtbExtraResult(rid=rid, ok=False, fail_reason=f"read:{e}")

    # Schema A (dipolar / qmrxn20): status.json carries fragment atom indices
    # explicitly. Schema B (rgd1): no indices in status.json — fall back to
    # reading geometry_fragA.xyz / geometry_fragB.xyz from the same directory
    # (assumes the TS xyz orders fragA atoms first, then fragB, which is how
    # the ADF EDA prep workflow writes them).
    src_dir = Path(status_json).parent
    if "fragment_atoms_a" in status and "fragment_atoms_b" in status:
        idx_a = list(status["fragment_atoms_a"])
        idx_b = list(status["fragment_atoms_b"])
        slicing = "indexed"
    else:
        frag_a_xyz = src_dir / "geometry_fragA.xyz"
        frag_b_xyz = src_dir / "geometry_fragB.xyz"
        if not (frag_a_xyz.exists() and frag_b_xyz.exists()):
            return XtbExtraResult(
                rid=rid, ok=False,
                fail_reason=f"no_indices_no_fragxyz:{src_dir}",
            )
        try:
            a_atoms = ase.io.read(str(frag_a_xyz))
            b_atoms = ase.io.read(str(frag_b_xyz))
        except Exception as e:
            return XtbExtraResult(rid=rid, ok=False, fail_reason=f"fragxyz_read:{e}")
        n_a_frag, n_b_frag = len(a_atoms), len(b_atoms)
        # Contiguous slice assumption: ts.xyz has fragA atoms first (n_a_frag),
        # then fragB (n_b_frag). Same convention as compute_xtb_extra().
        idx_a = list(range(n_a_frag))
        idx_b = list(range(n_a_frag, n_a_frag + n_b_frag))
        slicing = "contiguous_from_fragxyz"
    n_a, n_b = len(idx_a), len(idx_b)
    n_ts = len(ts_atoms)
    if n_ts > max_atoms:
        return XtbExtraResult(rid=rid, ok=False,
                              fail_reason=f"too_big:{n_ts}>{max_atoms}",
                              n_atoms_a=n_a, n_atoms_b=n_b)
    # Dual schema: dipolar/qmrxn20 has total_charge; rgd1 has charge_a + charge_b.
    if "total_charge" in status:
        total_charge = int(status["total_charge"])
    else:
        total_charge = int(status.get("charge_a", 0)) + int(status.get("charge_b", 0))

    try:
        rc = _run_one(ts_atoms.numbers, ts_atoms.get_positions(),
                      charge=total_charge, mult=1, want_bond_orders=True)
    except Exception as e:
        return XtbExtraResult(rid=rid, ok=False, fail_reason=f"complex_scf:{e}",
                              n_atoms_a=n_a, n_atoms_b=n_b,
                              wall_s=round(_t.time() - t0, 3))

    HOMO = rc["HOMO_h"]; LUMO = rc["LUMO_h"]
    gap = LUMO - HOMO
    mu = 0.5 * (HOMO + LUMO)
    eta = 0.5 * gap
    omega = float((mu * mu) / (2.0 * eta)) if abs(eta) > 1e-8 else float("nan")

    charges = np.asarray(rc["charges"]).ravel()
    qpol = float(np.sum(charges * charges))

    bo = np.asarray(rc.get("bond_orders"))
    # Index-based inter-fragment WBO — no contiguity assumption.
    if bo.shape[0] >= max(max(idx_a, default=-1), max(idx_b, default=-1)) + 1:
        dwbo_inter = float(np.abs(bo[np.ix_(idx_a, idx_b)]).sum())
    else:
        dwbo_inter = float("nan")

    return XtbExtraResult(
        rid=rid, ok=True,
        xtb_gap=float(gap), xtb_mu=float(mu), xtb_omega=omega,
        xtb_dipole=float(rc["dipole_norm"]),
        xtb_qpol=qpol,
        xtb_dwbo_interfrag=dwbo_inter,
        n_atoms_a=n_a, n_atoms_b=n_b,
        wall_s=round(_t.time() - t0, 3),
    )
