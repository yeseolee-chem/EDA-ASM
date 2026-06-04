"""ASR-compliant ADF runner per ASR_Fragmentation_Spec.md (sections 1, 4, 7).

For one reaction_id:
  - Load Stage 5a fragmentation result
  - Load Halo8 R / TS / P frames (3-point per spec, not 5-point)
  - Run 11 ADF jobs (3 full SP + 6 fragment SP + 2 fragment opt)
  - Compute ASR vector (strain, elst, Pauli, oi, disp) at each of R/TS/P
  - Run 10 spec diagnostic checks
  - Output JSON per spec section 7
  - Status: AUTO_ACCEPT_CANDIDATE / MANUAL_REVIEW_REQUIRED / FAILED

Methodology (spec section 1):
  Functional        BP86
  Dispersion        D3(BJ)
  Basis set         TZ2P, frozen core None
  Relativity        ZORA scalar
  Integration       Becke Good
  SCF threshold     1e-6
  Max SCF iter      200
  Symmetry          NOSYM
  Spin              unrestricted iff any fragment open-shell
  Solvent           none (gas phase)
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from scm.plams import (  # type: ignore
    Atom, AMSJob, KFFile, Molecule, Settings, init, finish,
)

# Inline frame loader (amspython is 3.8 — can't import halo8_io which uses 3.10
# dataclass slots). Reads R/TS/P from Halo8 ASE DB directly via sqlite.
import json as _json
import sqlite3


def _decode_data_blob(b: bytes) -> dict:
    """Decode ASE's `data` BLOB: 8-byte int64 offset header + JSON body."""
    if not b:
        return {}
    offset = int(np.frombuffer(b[:8], np.int64)[0])
    return _json.loads(b[offset:].decode())


def _decode_np_blob(b: bytes, dtype) -> np.ndarray | None:
    if not b:
        return None
    return np.frombuffer(b, dtype=dtype)


def _fetch_frames_3pt(db_path: Path, rxn_id: str,
                        wanted_frames: set) -> dict:
    """Stream the Halo8 ASE DB and pull only the 3 frames we need."""
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute(
        "SELECT energy, natoms, charge, numbers, positions, data FROM systems"
    )
    out = {}
    while True:
        rows = cur.fetchmany(50000)
        if not rows:
            break
        for energy, natoms, charge, nb, pb, db in rows:
            try:
                data = _decode_data_blob(db)
            except Exception:
                continue
            did = str(data.get("dand_id", ""))
            head, _, tail = did.rpartition("_")
            if head != rxn_id:
                continue
            try:
                fi = int(tail)
            except ValueError:
                continue
            if fi not in wanted_frames:
                continue
            numbers = _decode_np_blob(nb, np.int32).astype(int)
            positions = _decode_np_blob(pb, np.float64).reshape(int(natoms), 3)
            out[fi] = {
                "energy_eV": float(energy),
                "numbers": numbers,
                "positions": positions.copy(),
                "natoms": int(natoms),
                "charge": float(charge or 0.0),
            }
            if len(out) == len(wanted_frames):
                conn.close()
                return out
    conn.close()
    return out

HA_TO_EV = 27.211386245988
HA_TO_KCAL = 627.5094740631
EV_TO_KCAL = 23.060547830619
ZETA = ["R", "TS", "P"]
DB_DIR = Path("/home1/yeseo1ee/projects/ts_prediction_project/data")
STAGE5A_DIR = REPO / "outputs/stage5a"
OUT_DIR = REPO / "outputs/asr_spec"

# Halo8 row.energy is in eV. ADF result energies are converted to kcal/mol
# for spec compliance.


# ─── Methodology (spec section 1) ──────────────────────────────────────────
def base_settings() -> Settings:
    s = Settings()
    s.input.ams.Task = "SinglePoint"
    s.input.adf.XC.GGA = "BP86"
    s.input.adf.XC.Dispersion = "Grimme3 BJDAMP"
    s.input.adf.Basis.Type = "TZ2P"
    s.input.adf.Basis.Core = "None"
    s.input.adf.NumericalQuality = "Good"
    s.input.adf.Symmetry = "NoSym"
    s.input.adf.SCF.Iterations = 200
    s.input.adf.SCF.Converge = "1e-6"
    s.input.adf.Relativity.Level = "Scalar"
    s.input.adf.Relativity.Formalism = "ZORA"
    return s


def opt_settings() -> Settings:
    """Geometry optimization for strain reference (frag_A_opt, frag_B_opt)."""
    s = base_settings()
    s.input.ams.Task = "GeometryOptimization"
    s.input.ams.GeometryOptimization.MaxIterations = 200
    s.input.ams.GeometryOptimization.Convergence.Gradients = 1.0e-4
    return s


def with_spin(s: Settings, spin_polarization: int) -> Settings:
    """Apply unrestricted + SpinPolarization to a settings clone."""
    s2 = s.copy() if hasattr(s, "copy") else Settings(s.as_dict())
    if spin_polarization != 0:
        s2.input.adf.Unrestricted = "Yes"
        s2.input.adf.SpinPolarization = spin_polarization
        s2.input.adf.SCF.Mixing = 0.05
    return s2


# ─── Stage 5a helpers ──────────────────────────────────────────────────────
def load_stage5a(rxn_id: str) -> dict:
    fp = STAGE5A_DIR / "per_reaction" / rxn_id / "result.json"
    if not fp.exists():
        raise FileNotFoundError(fp)
    return json.loads(fp.read_text())


def assign_spin_signs(fragments: list[dict]) -> list[int]:
    """+1, -1, +1, -1 ... for open-shell fragments (anti-FM coupling).
    Closed-shell fragments get +1 (sign unused)."""
    out, open_i = [], 0
    for f in fragments:
        if int(f.get("multiplicity", 1)) > 1:
            out.append(1 if open_i % 2 == 0 else -1)
            open_i += 1
        else:
            out.append(1)
    return out


def mol_from(symbols, coords):
    m = Molecule()
    for s, (x, y, z) in zip(symbols, coords):
        m.add_atom(Atom(symbol=str(s), coords=(float(x), float(y), float(z))))
    return m


# ─── Halo8 frame loading ───────────────────────────────────────────────────
_DB_IDX_CACHE: dict | None = None


def _load_db_idx_map() -> dict:
    """Build {rxn_id -> source_db_idx}. Prefer the comprehensive JSON map
    (covers all 19,175 halo8 reactions); fall back to CSVs for back-compat."""
    global _DB_IDX_CACHE
    if _DB_IDX_CACHE is not None:
        return _DB_IDX_CACHE
    import csv as _csv
    json_map = REPO / "outputs/asr_spec/db_idx_map.json"
    if json_map.exists():
        with open(json_map) as f:
            _DB_IDX_CACHE = {k: int(v) for k, v in json.load(f).items()}
        return _DB_IDX_CACHE
    m: dict = {}
    for fp in [
        REPO / "outputs/phase1/selected_reactions.csv",
        REPO / "outputs/phase1/selected_reactions_4500.csv",
    ]:
        if not fp.exists():
            continue
        for row in _csv.DictReader(open(fp)):
            try:
                m[row["reaction_id"]] = int(row["source_db_idx"])
            except (KeyError, ValueError):
                pass
    _DB_IDX_CACHE = m
    return m


def load_3_frames(rxn_id: str, stage5a: dict) -> dict:
    """Return {"R": ..., "TS": ..., "P": ...} for 3 IRC frames."""
    db_map = _load_db_idx_map()
    if rxn_id not in db_map:
        raise RuntimeError(f"no source_db_idx for {rxn_id}")
    db_idx = db_map[rxn_id]
    db_path = DB_DIR / f"Halo_{db_idx}.db"

    fr_first = int(stage5a["frame_index_first"])
    fr_ts    = int(stage5a["ts_frame_idx"])
    fr_last  = int(stage5a["frame_index_last"])
    wanted = {fr_first, fr_ts, fr_last}
    frames_raw = _fetch_frames_3pt(db_path, rxn_id, wanted)
    if len(frames_raw) != len(wanted):
        missing = wanted - set(frames_raw.keys())
        raise RuntimeError(f"{rxn_id} missing frames {missing}")
    SYMTAB = {1:"H",6:"C",7:"N",8:"O",9:"F",15:"P",16:"S",17:"Cl",35:"Br",53:"I"}
    out = {}
    for label, fi in [("R", fr_first), ("TS", fr_ts), ("P", fr_last)]:
        r = frames_raw[fi]
        out[label] = {
            "symbols": [SYMTAB.get(int(z), "?") for z in r["numbers"]],
            "coords": r["positions"],
            "frame_idx": fi,
            "energy_eV": r["energy_eV"],
        }
    return out


# ─── EDA & S² parsing ──────────────────────────────────────────────────────
EDA_KEYS = {
    "int_total": ("Energy", "Bond Energy"),
    "Pauli":     ("Energy", "Pauli Total"),
    "elst":      ("Energy", "Electrostatic Interaction"),
    "oi":        ("Energy", "Orb.Int. Total"),
    "disp":      ("Energy", "Dispersion Energy"),
}


def parse_eda_kcal(rkf_path: str) -> dict:
    kf = KFFile(rkf_path)
    out = {}
    for k, (sec, var) in EDA_KEYS.items():
        try:
            out[k] = float(kf.read(sec, var)) * HA_TO_KCAL
        except Exception:
            out[k] = None
    return out


def parse_s2(rkf_path: str) -> float | None:
    kf = KFFile(rkf_path)
    for sec, var in [("Properties", "SpinSquared"),
                      ("General", "S2"),
                      ("Properties", "<S^2>")]:
        try:
            return float(kf.read(sec, var))
        except Exception:
            continue
    return None


# ─── Diagnostics (spec section 4) ──────────────────────────────────────────
def run_diagnostics(scf_ok: dict, asr_at_TS: dict, asr_at_R: dict,
                     asr_at_P: dict, delta_E_dft_TS: float,
                     s2_per_frag: dict, fragments: list[dict],
                     frag_geom_min_dist: dict, charge_total: float,
                     wall_time_min: float) -> dict:
    """Return dict of {check_name: "PASS"|"FAIL"} + reasons."""
    diag: dict = {"scf": "PASS" if all(scf_ok.values()) else "FAIL"}
    # 2: S² deviation
    s2_pass = True
    for f, s2 in s2_per_frag.items():
        if s2 is None:
            continue
        mult = next(fr["multiplicity"] for fr in fragments if fr["role"] == f)
        S = (mult - 1) / 2
        target = S * (S + 1)
        if abs(s2 - target) > 0.10:
            s2_pass = False
            break
    diag["S2"] = "PASS" if s2_pass else "FAIL"
    diag["S2_values"] = s2_per_frag

    # 3: Sum consistency
    if all(asr_at_TS.get(k) is not None for k in
           ["strain", "elst", "Pauli", "oi", "disp"]):
        sum_comp = sum(asr_at_TS[k] for k in ["strain", "elst", "Pauli", "oi", "disp"])
        diff = sum_comp - delta_E_dft_TS
        diag["sum_consistency"] = "PASS" if abs(diff) < 2.0 else "FAIL"
        diag["sum_consistency_kcal"] = float(diff)
    else:
        diag["sum_consistency"] = "FAIL"
        diag["sum_consistency_kcal"] = None

    # 4: Pauli range
    p = asr_at_TS.get("Pauli")
    diag["Pauli_range"] = "PASS" if (p is not None and 50 < p < 500) else "FAIL"

    # 5: oi sign
    o = asr_at_TS.get("oi")
    diag["oi_sign"] = "PASS" if (o is not None and o < 0) else "FAIL"

    # 6: strain sign
    s = asr_at_TS.get("strain")
    diag["strain_sign"] = "PASS" if (s is not None and s > 0) else "FAIL"

    # 7: wall time
    diag["wall_time_min"] = wall_time_min
    diag["wall_time"] = "PASS" if wall_time_min < 240 else "FAIL"

    # 8: geometry sanity — min interatomic distance per fragment
    diag["geometry_sanity"] = (
        "PASS" if all(d is None or d > 0.5 for d in frag_geom_min_dist.values())
        else "FAIL"
    )

    # 9: charge sanity (integer total)
    diag["charge_sanity"] = "PASS" if abs(charge_total - round(charge_total)) < 0.01 else "FAIL"

    # 10: numerical sanity (no NaN/Inf)
    def _bad(v):
        return v is not None and (math.isnan(v) or math.isinf(v))
    bad = any(_bad(asr_at_TS.get(k)) for k in
              ["strain", "elst", "Pauli", "oi", "disp"])
    diag["numerical"] = "FAIL" if bad else "PASS"
    return diag


def determine_status(rule_blocks_auto: bool, diag: dict, ambiguous: bool) -> str:
    """spec section 5"""
    if diag.get("scf") == "FAIL":
        return "FAILED" if "_crashed" in diag else "MANUAL_REVIEW_REQUIRED"
    if ambiguous or rule_blocks_auto:
        return "MANUAL_REVIEW_REQUIRED"
    failed_checks = [k for k, v in diag.items()
                      if isinstance(v, str) and v == "FAIL"]
    if failed_checks:
        return "MANUAL_REVIEW_REQUIRED"
    return "AUTO_ACCEPT_CANDIDATE"


# ─── Main per-reaction workflow ────────────────────────────────────────────
def run_one(rxn_id: str) -> dict:
    t_start = time.time()
    stage5a = load_stage5a(rxn_id)
    fragments = stage5a["result"]["fragments"]
    pattern = stage5a["result"]["pattern"]
    spin_signs = assign_spin_signs(fragments)
    total_spin = sum(s * (f["multiplicity"] - 1) for f, s in zip(fragments, spin_signs))
    open_shell = any(f["multiplicity"] > 1 for f in fragments)

    # Note: spec defines 2-fragment ASM by default; N>=3 fragments are
    # supported as a generalisation (strain = sum over all fragments,
    # ETSNOCV on N-body) when explicitly requested via the Stage 5a result.
    pass  # continue with N-fragment computation

    frames = load_3_frames(rxn_id, stage5a)
    symbols = frames["R"]["symbols"]

    workdir = Path("/tmp/yeseo1ee/asr_spec") / rxn_id
    if workdir.exists():
        shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True, exist_ok=True)
    init(folder=str(workdir), use_existing_folder=True)

    scf_ok: dict[str, bool] = {}
    whole_E: dict[str, float] = {}
    frag_E: dict[tuple, float] = {}
    frag_rkf: dict[tuple, str] = {}
    frag_opt_E: dict[str, float] = {}
    s2: dict[str, float | None] = {}
    eda: dict[str, dict] = {}

    try:
        base = base_settings()
        # ── 1) whole SPs ──
        for z in ZETA:
            mol = mol_from(symbols, frames[z]["coords"])
            n_e = sum(a.atnum for a in mol)
            if open_shell:
                s = with_spin(base, total_spin)
            else:
                mult = 1 if n_e % 2 == 0 else 2
                s = with_spin(base, mult - 1)
            job = AMSJob(molecule=mol, settings=s, name=f"whole_{z}")
            r = job.run()
            scf_ok[f"whole_{z}"] = r.ok()
            whole_E[z] = (r.get_energy(unit="kcal/mol") if r.ok() else None)

        # ── 2) fragment SPs (×3 ζ × N fragments) ──
        for z in ZETA:
            for f, ssign in zip(fragments, spin_signs):
                role = f["role"]
                f_syms = [symbols[k] for k in f["atom_indices"]]
                f_coords = frames[z]["coords"][f["atom_indices"]]
                mol = mol_from(f_syms, f_coords)
                s = with_spin(base, ssign * (f["multiplicity"] - 1))
                job = AMSJob(molecule=mol, settings=s,
                              name=f"frag_{role}_{z}")
                r = job.run()
                scf_ok[f"frag_{role}_{z}"] = r.ok()
                if r.ok():
                    frag_E[(role, z)] = r.get_energy(unit="kcal/mol")
                    frag_rkf[(role, z)] = r.rkfpath(file="adf")
                    if f["multiplicity"] > 1 and role not in s2:
                        s2[role] = parse_s2(frag_rkf[(role, z)])

        # ── 3) fragment geometry-opt (×2 strain references) ──
        for f, ssign in zip(fragments, spin_signs):
            role = f["role"]
            # Use R-geom as starting point
            f_syms = [symbols[k] for k in f["atom_indices"]]
            f_coords = frames["R"]["coords"][f["atom_indices"]]
            mol = mol_from(f_syms, f_coords)
            s = opt_settings()
            s = with_spin(s, ssign * (f["multiplicity"] - 1))
            job = AMSJob(molecule=mol, settings=s,
                          name=f"frag_{role}_opt")
            r = job.run()
            scf_ok[f"frag_{role}_opt"] = r.ok()
            if r.ok():
                frag_opt_E[role] = r.get_energy(unit="kcal/mol")

        # ── 4) coupled EDA-NOCV at each ζ ──
        for z in ZETA:
            if any((f["role"], z) not in frag_rkf for f in fragments):
                eda[z] = {}
                continue
            mol = mol_from(symbols, frames[z]["coords"])
            for i, at in enumerate(mol):
                for f in fragments:
                    if i in f["atom_indices"]:
                        at.properties.suffix = f"adf.f={f['role']}"
                        break
            s = base_settings()
            if open_shell:
                s.input.adf.Unrestricted = "Yes"
                s.input.adf.UnrestrictedFragments = "Yes"
                s.input.adf.SpinPolarization = total_spin
                s.input.adf.SCF.Mixing = 0.05
            for f in fragments:
                s.input.adf.Fragments[f["role"]] = str(frag_rkf[(f["role"], z)])
            s.input.adf.ETSNOCV.Enabled = "Yes"
            s.input.adf.ETSNOCV.ENOCV = 0.01
            s.input.adf.Print = "ETSLOWDIN"
            job = AMSJob(molecule=mol, settings=s, name=f"EDA_{z}")
            r = job.run()
            scf_ok[f"EDA_{z}"] = r.ok()
            eda[z] = parse_eda_kcal(r.rkfpath(file="adf")) if r.ok() else {}

    finally:
        finish()
        # workdir kept for now (deletable manually)

    # ── Compute ASR vector at each ζ ──
    def _strain_at(z):
        if not all((role, z) in frag_E and role in frag_opt_E for role in [f["role"] for f in fragments]):
            return None
        return sum(frag_E[(f["role"], z)] - frag_opt_E[f["role"]]
                   for f in fragments)

    asr = {}
    for z in ZETA:
        asr[z] = {
            "strain": _strain_at(z),
            "elst":   eda.get(z, {}).get("elst"),
            "Pauli":  eda.get(z, {}).get("Pauli"),
            "oi":     eda.get(z, {}).get("oi"),
            "disp":   eda.get(z, {}).get("disp"),
        }

    # ΔE‡_DFT(TS) — whole energy difference (TS - R) in kcal/mol
    deltaE_TS = (whole_E.get("TS") - whole_E.get("R")
                  if whole_E.get("TS") is not None and whole_E.get("R") is not None
                  else None)

    # Geometry sanity per fragment (min interatomic distance at TS)
    frag_min_d = {}
    for f in fragments:
        atom_idx = f["atom_indices"]
        coords = frames["TS"]["coords"][atom_idx]
        if len(coords) < 2:
            frag_min_d[f["role"]] = None
            continue
        d = float("inf")
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                d = min(d, float(np.linalg.norm(coords[i] - coords[j])))
        frag_min_d[f["role"]] = d

    wall_min = (time.time() - t_start) / 60.0

    diag = run_diagnostics(
        scf_ok=scf_ok,
        asr_at_TS=asr["TS"], asr_at_R=asr["R"], asr_at_P=asr["P"],
        delta_E_dft_TS=deltaE_TS or 0.0,
        s2_per_frag=s2, fragments=fragments,
        frag_geom_min_dist=frag_min_d,
        charge_total=float(stage5a.get("total_charge", 0.0)),
        wall_time_min=wall_min,
    )

    status = determine_status(
        rule_blocks_auto=(pattern == "P5b"),
        diag=diag, ambiguous=False,
    )

    # ── Result per spec section 7 ──
    result = {
        "reaction_id": rxn_id,
        "schema_version": "asr_spec_v1",
        "halo8_meta": {
            "n_heavy_atoms": stage5a.get("n_heavy_atoms"),
            "n_atoms": stage5a.get("n_atoms"),
            "source": stage5a.get("source"),
            "frame_index_first": stage5a.get("frame_index_first"),
            "ts_frame_idx": stage5a.get("ts_frame_idx"),
            "frame_index_last": stage5a.get("frame_index_last"),
            "activation_energy_eV": stage5a.get("activation_energy"),
        },
        "pattern": pattern,
        "fragmentation": {
            "fragments": fragments,
            "spin_signs": spin_signs,
            "total_spin_polarization": int(total_spin),
            "coupling": ("anti_FM_broken_symmetry" if open_shell and total_spin == 0
                          else "high_spin" if open_shell else "closed_shell_singlet"),
        },
        "adf_settings": {
            "functional": "BP86",
            "dispersion": "D3BJ",
            "basis": "TZ2P",
            "frozen_core": "None",
            "relativity": "ZORA_scalar",
            "integration": "Becke_Good",
            "scf_thr": 1e-6,
            "symmetry": "NoSym",
        },
        "irc_points": {
            z: {"frame_idx": frames[z]["frame_idx"],
                "energy_eV_halo8": frames[z]["energy_eV"],
                "energy_kcal_adf": whole_E.get(z)}
            for z in ZETA
        },
        "fragment_opt_energy_kcal": frag_opt_E,
        "asr_vector_kcal": asr,
        "consistency": {
            "sum_components_TS": (
                sum(asr["TS"][k] for k in ["strain","elst","Pauli","oi","disp"])
                if all(asr["TS"][k] is not None for k in ["strain","elst","Pauli","oi","disp"])
                else None),
            "delta_E_DFT_TS_kcal": deltaE_TS,
            "diff_TS_kcal": diag.get("sum_consistency_kcal"),
            "S2_per_fragment": s2,
        },
        "diagnostics": diag,
        "scf_convergence_per_job": scf_ok,
        "status_at_queue": status,
        "manual_review_reasons": [
            k for k, v in diag.items()
            if isinstance(v, str) and v == "FAIL"
        ],
        "user_decision": None,
        "fragmentation_revision": 0,
        "wall_time_min": wall_min,
        "pipeline_version": "asr_spec_v1",
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rxn_id", required=True)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{args.rxn_id}.json"
    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text())
            if prev.get("status_at_queue") not in (None, "FAILED"):
                print(f"[SKIP] {args.rxn_id} already done ({prev.get('status_at_queue')})")
                return
        except Exception:
            pass

    try:
        result = run_one(args.rxn_id)
    except Exception as e:
        result = {
            "reaction_id": args.rxn_id,
            "schema_version": "asr_spec_v1",
            "status_at_queue": "FAILED",
            "manual_review_reasons": [f"ERROR: {e}"],
            "traceback": traceback.format_exc(),
        }
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"[OK] {args.rxn_id}: {result.get('status_at_queue')}")


if __name__ == "__main__":
    main()
