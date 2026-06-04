"""Per-reaction 5-point EDA-ASM workflow (v3 — eda_asm_benchmark_methodology.md).

For a reaction `<rxn_id>` already processed by stage5a, this script:

  1. Reads the Halo8 trajectory from the appropriate Halo_*.db.
  2. Selects 5 ζ-points via arc-length-equidistant sampling:
        ζ0 = R (frame 0)
        ζ1 = arc midpoint of R → TS
        ζ2 = TS (max-energy interior frame, from stage5a)
        ζ3 = arc midpoint of TS → P
        ζ4 = P (last frame)
  3. For each ζ-point, runs:
        - Whole-molecule SP
        - 2 fragment SPs (slicing atoms by stage5a's atom_indices)
        - Fragment-based EDA (using same-ζ fragment rkfs as references)
  4. Extracts the v3 schema:
        - 5-point energetics (Δ-from-R) for total / strain / int / elstat / Pauli / orb / disp
        - per-fragment strain
        - per-fragment ⟨S²⟩ + VDD + Hirshfeld + HOMO/LUMO
        - NOCV pair energies + eigenvalues (top 5 per ζ)
        - 6-stage QC flags

Settings follow methodology §4: PBE-D3(BJ)/TZ2P, all-electron, ZORA scalar,
NumericalQuality Normal, SCF 1e-6, SYMMETRY NOSYM (σ/π separation dropped),
ETSNOCV ENOCV 0.01 (top-N selection done at parsing).

Usage:
    module load mpi/2021.9.0
    source $HOME/ams2026.103/amsbashrc.sh
    NSCM=1 $AMSBIN/amspython scripts/run_eda_one_v3.py --rxn_id <ID>
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scm.plams import AMSJob, Atom, KFFile, Molecule, Settings, init, finish


HA_TO_EV = 27.2114079527

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
STAGE5A_DIR = REPO / "outputs" / "stage5a" / "per_reaction"
STAGE5B_DIR = REPO / "outputs" / "stage5b" / "per_reaction"
ZETA_BUNDLES_DIR = REPO / "outputs" / "stage5b" / "zeta_bundles"

ZETA_LABELS = ["R", "pre_TS", "TS", "post_TS", "P"]


def load_zeta_bundle(rxn_id: str) -> dict:
    """Load the pre-extracted 5-point ζ bundle for `rxn_id`.

    Bundle is produced once by scripts/extract_5point_v3.py and contains
    coords/energies/forces/numbers/symbols/etc. for 5 ζ-points (R, pre_TS,
    TS, post_TS, P) along the IRC arc-length.
    """
    npz_path = ZETA_BUNDLES_DIR / f"{rxn_id}.npz"
    if not npz_path.exists():
        raise RuntimeError(
            f"ζ bundle missing for {rxn_id}: {npz_path}\n"
            f"Run `python3 scripts/extract_5point_v3.py` first."
        )
    return dict(np.load(npz_path, allow_pickle=True))

EDA_KEY_MAP = {
    "Pauli":     ("Energy", "Pauli Total"),
    "elstat":    ("Energy", "Electrostatic Interaction"),
    "orb":       ("Energy", "Orb.Int. Total"),
    "disp":      ("Energy", "Dispersion Energy"),
    "int_total": ("Energy", "Bond Energy"),
}


# ============================================================================
# Halo8 trajectory access — read full trajectory frames for arc-length sampling
# ============================================================================



# ============================================================================
# Settings — methodology §4
# ============================================================================

def base_settings() -> Settings:
    """v3 settings per methodology §4."""
    s = Settings()
    s.input.ams.Task = "SinglePoint"
    s.input.adf.XC.GGA = "PBE"
    s.input.adf.XC.Dispersion = "Grimme3 BJDAMP"
    s.input.adf.Basis.Type = "TZ2P"
    s.input.adf.Basis.Core = "None"
    s.input.adf.NumericalQuality = "Normal"
    s.input.adf.Symmetry = "NoSym"   # methodology §2 — enforced
    s.input.adf.SCF.Iterations = 200
    s.input.adf.SCF.Converge = "1e-6 1e-6"
    s.input.adf.Relativity.Level = "Scalar"
    s.input.adf.Relativity.Formalism = "ZORA"
    return s


def fragment_settings(multiplicity: int, spin_sign: int = 1) -> Settings:
    s = base_settings()
    if multiplicity > 1:
        s.input.adf.Unrestricted = "Yes"
        s.input.adf.SpinPolarization = spin_sign * (multiplicity - 1)
        s.input.adf.SCF.Mixing = 0.05
    return s


def complex_settings_for_open_shell(fragments, spin_signs) -> Settings:
    """Complex EDA settings for open-shell fragmentation.

    Sets `SpinPolarization` as a Python list. PLAMS renders this as
    multiple lines (one per element), which AMS rejects as "duplicate
    unique entry". We use a custom AMSJob subclass below (FixSpinJob) to
    rewrite the .in file before submission and collapse the duplicate
    lines into a single Float-List line.
    """
    s = base_settings()
    s.input.adf.Unrestricted = "Yes"
    sp_list = [float(ssign * (f["multiplicity"] - 1))
               for f, ssign in zip(fragments, spin_signs)]
    s.input.adf.SpinPolarization = sp_list
    s.input.adf.SCF.Mixing = 0.05
    return s


import re


class FixSpinJob(AMSJob):
    """AMSJob that rewrites the .in file before submission to collapse
    consecutive duplicate `SpinPolarization` lines into a single
    Float-List line."""

    _RE = re.compile(
        r"^(\s*)SpinPolarization\s+(\S+)\n(?:\1SpinPolarization\s+(\S+)\n)+",
        re.MULTILINE,
    )

    def _rewrite_input(self):
        from pathlib import Path
        in_file = Path(self.path) / f"{self.name}.in"
        if not in_file.exists():
            return
        text = in_file.read_text()
        # Find any consecutive "SpinPolarization X" lines and merge.
        # Manual collapse since re alternation is fiddly:
        lines = text.split("\n")
        out = []
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            if stripped.startswith("SpinPolarization "):
                values = [stripped.split(None, 1)[1]]
                indent = line[:len(line) - len(line.lstrip())]
                j = i + 1
                while j < len(lines):
                    s2 = lines[j].strip()
                    if s2.startswith("SpinPolarization "):
                        values.append(s2.split(None, 1)[1])
                        j += 1
                    else:
                        break
                out.append(f"{indent}SpinPolarization {' '.join(values)}")
                i = j
            else:
                out.append(line)
                i += 1
        new_text = "\n".join(out)
        if new_text != text:
            in_file.write_text(new_text)

    def run(self, *args, **kwargs):
        # PLAMS lifecycle: get_ready writes the input file, then run
        # actually executes. Override run() to splice our rewrite in.
        # Use the standard PLAMS path: _prepare which calls get_input()
        # and writes the file; we re-write right after that.
        result = super().run(*args, **kwargs)
        return result

    def _get_ready(self):
        super()._get_ready()
        self._rewrite_input()


def assign_spin_signs(fragments):
    signs = []
    open_idx = 0
    for f in fragments:
        if f["multiplicity"] > 1:
            signs.append(1 if open_idx % 2 == 0 else -1)
            open_idx += 1
        else:
            signs.append(1)
    return signs


# ============================================================================
# Result parsing — read all v3 schema fields from an ADF rkf
# ============================================================================

def parse_eda(rkf_path: str) -> dict:
    kf = KFFile(rkf_path)
    out = {}
    for channel, (sec, var) in EDA_KEY_MAP.items():
        val_ha = kf.read(sec, var)
        out[channel] = val_ha * HA_TO_EV
    return out


def parse_fragment_diag(rkf_path: str, mult: int) -> dict:
    """Extract diagnostic info from a fragment SP rkf:
    ⟨S²⟩, VDD/Hirshfeld charges, HOMO/LUMO energies."""
    kf = KFFile(rkf_path)
    diag = {}
    # ⟨S²⟩ (only meaningful for unrestricted)
    if mult > 1:
        for sec, var in [("Properties", "SpinSquared"), ("General", "S2"),
                         ("Properties", "<S^2>")]:
            try:
                diag["s2"] = float(kf.read(sec, var))
                break
            except Exception:
                continue
    # VDD charges
    for sec, var in [("Properties", "AtomCharges_VDD"),
                     ("Properties", "VDD charges"),
                     ("Properties", "FragmentCharges VDD")]:
        try:
            diag["vdd"] = list(kf.read(sec, var))
            break
        except Exception:
            continue
    # Hirshfeld
    for sec, var in [("Properties", "AtomCharges_Hirshfeld"),
                     ("Properties", "Hirshfeld charges")]:
        try:
            diag["hirshfeld"] = list(kf.read(sec, var))
            break
        except Exception:
            continue
    # HOMO / LUMO
    try:
        homo_eV = kf.read("Properties", "HOMO") * HA_TO_EV
        diag["homo_eV"] = homo_eV
    except Exception:
        pass
    try:
        lumo_eV = kf.read("Properties", "LUMO") * HA_TO_EV
        diag["lumo_eV"] = lumo_eV
    except Exception:
        pass
    return diag


def parse_nocv_pairs(rkf_path: str, top_n: int = 5) -> list:
    """Pull the top-|eigenvalue| NOCV pairs with energy contributions."""
    kf = KFFile(rkf_path)
    pairs = []
    try:
        # Section names in AMS 2026: 'NOCV'
        for sec in kf.sections():
            if sec.lower() != "nocv":
                continue
            keys = list(kf.read_section(sec))
            # Find eigenvalue list and energy contributions
            eigs, energies = None, None
            for k in keys:
                low = k.lower()
                if "eigenvalue" in low and "list" in low:
                    eigs = kf.read(sec, k)
                elif "energy contribution" in low:
                    energies = kf.read(sec, k)
            if eigs is not None and energies is not None:
                # Build (eigenvalue, energy) pairs and pick top by |eig|
                triples = []
                for i, (e, en) in enumerate(zip(eigs, energies)):
                    triples.append({"index": i, "eigenvalue": float(e),
                                    "energy_eV": float(en) * HA_TO_EV})
                triples.sort(key=lambda t: abs(t["eigenvalue"]), reverse=True)
                pairs = triples[:top_n]
                break
    except Exception:
        pass
    return pairs


# ============================================================================
# Main per-reaction workflow
# ============================================================================

def _mol_from_arrays(symbols, coords):
    mol = Molecule()
    for s, (x, y, z) in zip(symbols, coords):
        mol.add_atom(Atom(symbol=s, coords=(float(x), float(y), float(z))))
    return mol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rxn_id", required=True)
    args = ap.parse_args()
    rxn_id = args.rxn_id

    in_dir = STAGE5A_DIR / rxn_id
    out_dir = STAGE5B_DIR / rxn_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Idempotency
    result_path = out_dir / "eda_result.json"
    if result_path.exists():
        try:
            with open(result_path) as f:
                prev = json.load(f)
            if (prev.get("schema_version") == "stage5b_v3_5point"
                    and prev.get("metadata", {}).get("status") == "ok"):
                print(f"[SKIP] {rxn_id} already v3-complete")
                return
        except Exception:
            pass

    with open(in_dir / "result.json") as f:
        meta = json.load(f)
    pattern = meta["result"]["pattern"]
    fragments = meta["result"]["fragments"]
    ts_frame_idx = meta["ts_frame_idx"]

    print(f"[{rxn_id}]  pattern={pattern}  n_frag={len(fragments)}")

    # Load pre-extracted 5-point bundle (from extract_5point_v3.py)
    bundle = load_zeta_bundle(rxn_id)
    coords_5pts = bundle["coords_5pts"]          # (5, N, 3)
    energies_5pts = bundle["energies_5pts"]      # (5,)
    symbols_arr = bundle["symbols"]               # (N,) str
    frame_indices = bundle["frame_indices"]       # (5,) int
    print(f"  ζ-points (frame indices): {frame_indices.tolist()}")
    print(f"  ζ-point energies (eV):    {[round(float(e), 3) for e in energies_5pts]}")

    spin_signs = assign_spin_signs(fragments)
    print(f"  spin signs: " + ", ".join(
        f"{f['role']}={s:+d}" for f, s in zip(fragments, spin_signs)))

    # PLAMS workdir on LOCAL disk (/tmp), not GPFS — the cluster's GPFS has
    # an inode-count limit shared across users (we hit 95% at ~250 rxns).
    # Final eda_result.json (the only persistent artifact) still lands on
    # GPFS via `out_dir / "eda_result.json"`. The raw rkfs in /tmp are
    # deleted at the end of this script.
    import shutil
    tmp_root = Path("/tmp/yeseo1ee/eda_v3")
    tmp_root.mkdir(parents=True, exist_ok=True)
    tmp_workdir = tmp_root / rxn_id
    if tmp_workdir.exists():
        shutil.rmtree(tmp_workdir, ignore_errors=True)
    tmp_workdir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    init(folder=str(tmp_workdir), use_existing_folder=True)

    # Per-ζ storage
    whole_E = {}        # zlabel → eV
    frag_E = {}         # (zlabel, role) → eV
    frag_rkf = {}       # (zlabel, role) → rkf path
    frag_diag = {}      # (zlabel, role) → {s2, vdd, hirshfeld, homo, lumo}
    eda_at = {}         # zlabel → {Pauli, elstat, orb, disp, int_total}
    nocv_at = {}        # zlabel → list of NOCV pairs
    scf_ok = {}         # job_name → bool

    try:
        # ===== Whole-molecule SPs at each ζ =====
        symbols_list = [str(s) for s in symbols_arr]
        for zi, zlabel in enumerate(ZETA_LABELS):
            coords = coords_5pts[zi]
            mol = _mol_from_arrays(symbols_list, coords)
            n_e = sum(a.atnum for a in mol)
            mult = 1 if n_e % 2 == 0 else 2
            s = fragment_settings(mult)
            job = AMSJob(molecule=mol, settings=s, name=f"whole_{zlabel}")
            res = job.run()
            scf_ok[f"whole_{zlabel}"] = res.ok()
            if res.ok():
                whole_E[zlabel] = res.get_energy(unit="eV")
                print(f"  [whole {zlabel:<7}] {whole_E[zlabel]:+.4f} eV")
            else:
                whole_E[zlabel] = None
                print(f"  [whole {zlabel:<7}] FAILED")

        # ===== Fragment SPs at each ζ =====
        for zi, zlabel in enumerate(ZETA_LABELS):
            coords = coords_5pts[zi]
            for f, ssign in zip(fragments, spin_signs):
                role = f["role"]
                mult = f["multiplicity"]
                atom_idx = f["atom_indices"]
                f_symbols = [symbols_list[k] for k in atom_idx]
                f_coords = coords[atom_idx]
                mol = _mol_from_arrays(f_symbols, f_coords)
                s = fragment_settings(mult, ssign)
                job = AMSJob(molecule=mol, settings=s,
                             name=f"frag_{role}_{zlabel}")
                res = job.run()
                key = (zlabel, role)
                scf_ok[f"frag_{role}_{zlabel}"] = res.ok()
                if res.ok():
                    frag_E[key] = res.get_energy(unit="eV")
                    frag_rkf[key] = res.rkfpath(file="adf")
                    frag_diag[key] = parse_fragment_diag(frag_rkf[key], mult)
                    print(f"  [frag {role:<14} @ {zlabel:<7}] "
                          f"{frag_E[key]:+.4f} eV  "
                          f"⟨S²⟩={frag_diag[key].get('s2', '—')}")
                else:
                    frag_E[key] = None
                    print(f"  [frag {role} @ {zlabel}] FAILED")

        # ===== Complex EDA at each ζ =====
        # AMS 2026 fragment-based EDA-NOCV with unrestricted fragments requires
        # FragOccupations + ModifyStartPotential setup that does not generalise
        # to single-atom fragments (e.g. P5_HSHIFT migrating H). For 55% of the
        # 500-set (any fragment with mult>1), we run only whole + fragment SPs
        # and emit strain values; the EDA channel breakdown is left empty.
        total_unpaired = sum(f["multiplicity"] - 1 for f in fragments)
        if total_unpaired > 0:
            print(f"  [EDA] open-shell fragments — skipping complex EDA "
                  f"(strain still computed from frag SPs)")
            # populate eda_at with empty dicts so downstream code handles None
            for zlabel in ZETA_LABELS:
                eda_at[zlabel] = {}
                nocv_at[zlabel] = []
            # Skip the EDA loop entirely
            skip_eda = True
        else:
            skip_eda = False
        for zi, zlabel in enumerate(ZETA_LABELS):
            if skip_eda:
                break
            if any(frag_rkf.get((zlabel, f["role"])) is None for f in fragments):
                print(f"  [EDA @ {zlabel}] skipping: missing fragment rkf")
                continue
            coords = coords_5pts[zi]
            mol = _mol_from_arrays(symbols_list, coords)
            for i, at in enumerate(mol):
                role = None
                for f in fragments:
                    if i in f["atom_indices"]:
                        role = f["role"]
                        break
                if role is None:
                    raise RuntimeError(f"atom {i} unassigned")
                at.properties.suffix = f"adf.f={role}"

            if total_unpaired > 0:
                s = complex_settings_for_open_shell(fragments, spin_signs)
                job_cls = FixSpinJob  # post-process .in to collapse duplicate SpinPolarization lines
            else:
                s = base_settings()
                job_cls = AMSJob
            for f in fragments:
                s.input.adf.Fragments[f["role"]] = str(frag_rkf[(zlabel, f["role"])])
            s.input.adf.ETSNOCV.Enabled = "Yes"
            s.input.adf.ETSNOCV.ENOCV = 0.01
            s.input.adf.Print = "ETSLOWDIN"

            job = job_cls(molecule=mol, settings=s, name=f"complex_EDA_{zlabel}")
            res = job.run()
            scf_ok[f"complex_EDA_{zlabel}"] = res.ok()
            if res.ok():
                rkf = res.rkfpath(file="adf")
                eda_at[zlabel] = parse_eda(rkf)
                nocv_at[zlabel] = parse_nocv_pairs(rkf, top_n=5)
                e = eda_at[zlabel]
                print(f"  [EDA  @ {zlabel:<7}]  "
                      f"Pauli{e['Pauli']:+.3f}  el{e['elstat']:+.3f}  "
                      f"orb{e['orb']:+.3f}  disp{e['disp']:+.3f}  "
                      f"int{e['int_total']:+.3f} eV")
            else:
                print(f"  [EDA  @ {zlabel:<7}] FAILED")

        # ===== Δ-from-R schema =====
        def _delta(at_R, at_z):
            if at_R is None or at_z is None:
                return None
            return at_z - at_R

        energetics = {ch: [None] * 5 for ch in
                      ["total", "strain", "int", "elstat", "Pauli",
                       "orb", "disp"]}
        frag_strain = {f["role"]: [None] * 5 for f in fragments}
        for zi, zlabel in enumerate(ZETA_LABELS):
            energetics["total"][zi] = _delta(whole_E.get("R"), whole_E.get(zlabel))
            # strain = sum_i (E_frag_i(ζ) - E_frag_i(R))
            sum_strain = 0.0
            valid_strain = True
            for f in fragments:
                eR = frag_E.get(("R", f["role"]))
                ez = frag_E.get((zlabel, f["role"]))
                if eR is None or ez is None:
                    valid_strain = False
                    frag_strain[f["role"]][zi] = None
                else:
                    fs = ez - eR
                    frag_strain[f["role"]][zi] = fs
                    sum_strain += fs
            energetics["strain"][zi] = sum_strain if valid_strain else None
            for ch_src, ch_dst in [("int_total", "int"),
                                   ("elstat", "elstat"),
                                   ("Pauli", "Pauli"),
                                   ("orb", "orb"),
                                   ("disp", "disp")]:
                eR = eda_at.get("R", {}).get(ch_src)
                ez = eda_at.get(zlabel, {}).get(ch_src)
                energetics[ch_dst][zi] = _delta(eR, ez)

        # ===== Quality control (methodology §6.3) =====
        flags = []
        # 1. SCF
        if not all(scf_ok.values()):
            flags.append("SCF_FAIL")
        # 2. EDA sum closure at each ζ
        for zi, zlabel in enumerate(ZETA_LABELS):
            ed = eda_at.get(zlabel, {})
            if ed and all(k in ed for k in
                          ["Pauli", "elstat", "orb", "disp", "int_total"]):
                recon = ed["Pauli"] + ed["elstat"] + ed["orb"] + ed["disp"]
                if abs(recon - ed["int_total"]) > 0.001:
                    flags.append(f"EDA_SANITY_FAIL_{zlabel}")
        # 3. ASM closure: total = strain + int  (deltas)
        for zi in range(5):
            t = energetics["total"][zi]
            st = energetics["strain"][zi]
            it = energetics["int"][zi]
            if t is not None and st is not None and it is not None:
                if abs(t - (st + it)) > 0.05:  # 50 meV tolerance
                    flags.append(f"ASM_SANITY_FAIL_{ZETA_LABELS[zi]}")
        # 4. Strain monotonicity (TS should be max) — informational
        strains = [s for s in energetics["strain"] if s is not None]
        info_flags = []
        if len(strains) == 5:
            if not (strains[2] >= max(strains[0], strains[4])):
                info_flags.append("STRAIN_NONMONOTONIC_P_HIGHER_THAN_TS")
        # 5. Spin contamination — informational
        s2_max = 0.0
        for d in frag_diag.values():
            if "s2" in d:
                s2_max = max(s2_max, d["s2"])
        if s2_max > 0.85:
            info_flags.append(f"SPIN_CONTAMINATION_MAX_{s2_max:.2f}")
        if skip_eda:
            info_flags.append("EDA_SKIPPED_OPEN_SHELL")

        # Status: "ok" if no hard failures (SCF/EDA/ASM sanity).
        # Informational flags (strain shape, spin contamination, EDA skip)
        # are recorded but do not downgrade status.
        status = "ok" if not flags else "partial"
        flags = flags + info_flags

        # ===== Assemble result =====
        result = {
            "rxn_id": rxn_id,
            "schema_version": "stage5b_v3_5point",
            "metadata": {
                "stage5a_pattern": pattern,
                "p2_subtype": meta["result"].get("p2_subtype"),
                "fragmentation_source": f"outputs/stage5a/per_reaction/{rxn_id}/result.json",
                "adf_version": "2026.103",
                "ams_functional": "PBE-D3(BJ)/TZ2P",
                "numerical_quality": "Normal",
                "scf_converge": "1e-6 1e-6",
                "symmetry": "NoSym",
                "wall_time_seconds": time.time() - t0,
                "nscm": int(os.environ.get("NSCM", "1")),
                "host": os.environ.get("HOSTNAME", os.uname().nodename),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "status": status,
            },
            "zeta_frame_indices": [int(x) for x in frame_indices],
            "zeta_labels": ZETA_LABELS,
            "halo8_reference_energies_eV": [float(x) for x in energies_5pts],
            "energetics_delta_from_R_eV": energetics,
            "fragment_strain_eV": frag_strain,
            "absolute_whole_energies_eV": {k: whole_E[k] for k in ZETA_LABELS},
            "absolute_fragment_energies_eV": {
                f"{role}_{zlabel}": frag_E.get((zlabel, role))
                for role in [f["role"] for f in fragments]
                for zlabel in ZETA_LABELS
            },
            "absolute_eda_at_zeta_eV": {
                zlabel: eda_at.get(zlabel, {}) for zlabel in ZETA_LABELS
            },
            "nocv_top5_per_zeta": {
                zlabel: nocv_at.get(zlabel, []) for zlabel in ZETA_LABELS
            },
            "spin_diagnostics_s2": {
                f"{role}_{zlabel}": frag_diag.get((zlabel, role), {}).get("s2")
                for role in [f["role"] for f in fragments]
                for zlabel in ZETA_LABELS
            },
            "vdd_charges_per_frag_per_zeta": {
                f"{role}_{zlabel}": frag_diag.get((zlabel, role), {}).get("vdd")
                for role in [f["role"] for f in fragments]
                for zlabel in ZETA_LABELS
            },
            "hirshfeld_charges_per_frag_per_zeta": {
                f"{role}_{zlabel}": frag_diag.get((zlabel, role), {}).get("hirshfeld")
                for role in [f["role"] for f in fragments]
                for zlabel in ZETA_LABELS
            },
            "fragment_homo_lumo_eV": {
                f"{role}_{zlabel}": {
                    "homo": frag_diag.get((zlabel, role), {}).get("homo_eV"),
                    "lumo": frag_diag.get((zlabel, role), {}).get("lumo_eV"),
                }
                for role in [f["role"] for f in fragments]
                for zlabel in ZETA_LABELS
            },
            "fragments": [
                {"role": f["role"],
                 "atom_indices": f["atom_indices"],
                 "multiplicity": f["multiplicity"],
                 "spin_sign": spin_signs[i],
                 "n_atoms": len(f["atom_indices"])}
                for i, f in enumerate(fragments)
            ],
            "halo8_reference": {
                "Ea_eV": meta.get("activation_energy"),
                "E_R": meta.get("energy_R"),
                "E_TS": meta.get("energy_TS"),
                "E_P": meta.get("energy_P"),
                "ts_frame_idx": ts_frame_idx,
            },
            "quality_flags": flags,
            "eda_available": not skip_eda,
            "scf_convergence_per_job": scf_ok,
        }
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n[DONE] {result_path}")
        print(f"       status={status}  flags={flags}")
        print(f"       wall={result['metadata']['wall_time_seconds']/60:.1f} min")

    finally:
        finish()
        # Clean up local /tmp workdir to free inodes — only the persisted
        # eda_result.json on GPFS is retained.
        try:
            shutil.rmtree(tmp_workdir, ignore_errors=True)
        except Exception:
            pass


if __name__ == "__main__":
    main()
