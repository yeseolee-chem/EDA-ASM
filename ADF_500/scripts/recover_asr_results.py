"""Recover spec result JSONs from /tmp/yeseo1ee/asr_spec/<rxn_id>/ workdirs.

Each workdir contains AMS subdirs: whole_R, whole_TS, whole_P,
frag_<role>_<R/TS/P>, frag_<role>_opt, EDA_R, EDA_TS, EDA_P.

For each completed workdir we re-parse energies + EDA channels and
build the same JSON schema as scripts/run_asr_spec.py.
"""
import json
import os
import sys
from pathlib import Path

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO/"src"))

# We need PLAMS KFFile for rkf reading
from scm.plams import KFFile  # noqa

HA_TO_KCAL = 627.5094740631
EDA_KEYS = {
    "int_total": ("Energy", "Bond Energy"),
    "Pauli":     ("Energy", "Pauli Total"),
    "elst":      ("Energy", "Electrostatic Interaction"),
    "oi":        ("Energy", "Orb.Int. Total"),
    "disp":      ("Energy", "Dispersion Energy"),
}
ZETA = ["R", "TS", "P"]
TMP_ROOT = Path("/tmp/yeseo1ee/asr_spec")
OUT_DIR = REPO / "outputs/asr_spec"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_energy_kcal(workdir, name):
    """Read bond energy (Hartree) from <workdir>/<name>/adf.rkf, return kcal."""
    rkf = workdir / name / "adf.rkf"
    if not rkf.exists():
        return None
    try:
        kf = KFFile(str(rkf))
        return float(kf.read("Energy", "Bond Energy")) * HA_TO_KCAL
    except Exception:
        return None


def parse_eda_kcal(workdir, name):
    rkf = workdir / name / "adf.rkf"
    if not rkf.exists():
        return None
    out = {}
    try:
        kf = KFFile(str(rkf))
        for k, (sec, var) in EDA_KEYS.items():
            try:
                out[k] = float(kf.read(sec, var)) * HA_TO_KCAL
            except Exception:
                out[k] = None
    except Exception:
        return None
    return out


def list_frag_roles(workdir):
    """Find unique fragment roles from subdirs like frag_<role>_<R|TS|P|opt>."""
    roles = set()
    for d in workdir.iterdir():
        if not d.is_dir(): continue
        name = d.name
        if not name.startswith("frag_"): continue
        body = name[len("frag_"):]
        for suffix in ["_R","_TS","_P","_opt"]:
            if body.endswith(suffix):
                roles.add(body[:-len(suffix)])
                break
    return sorted(roles)


def recover(rxn_id):
    wd = TMP_ROOT / rxn_id
    if not wd.exists():
        return None
    # Quick test: does it have a basic complete signature?
    if not (wd / "EDA_TS").exists():
        return None  # not finished
    out_fp = OUT_DIR / f"{rxn_id}.json"
    if out_fp.exists():
        return "skip_already"

    # Whole energies
    whole_E = {z: parse_energy_kcal(wd, f"whole_{z}") for z in ZETA}
    # Fragment SPs
    roles = list_frag_roles(wd)
    frag_E = {}
    frag_opt_E = {}
    for role in roles:
        for z in ZETA:
            frag_E[(role, z)] = parse_energy_kcal(wd, f"frag_{role}_{z}")
        frag_opt_E[role] = parse_energy_kcal(wd, f"frag_{role}_opt")
    # EDA
    eda = {z: parse_eda_kcal(wd, f"EDA_{z}") or {} for z in ZETA}

    # ASR vector at each ζ (per spec: strain = E(frag z) - E(frag opt))
    asr = {}
    for z in ZETA:
        if all((role, z) in frag_E and frag_E[(role, z)] is not None
               and role in frag_opt_E and frag_opt_E[role] is not None
               for role in roles):
            strain = sum(frag_E[(role, z)] - frag_opt_E[role] for role in roles)
        else:
            strain = None
        e = eda.get(z, {})
        asr[z] = {
            "strain": strain,
            "elst":   e.get("elst"),
            "Pauli":  e.get("Pauli"),
            "oi":     e.get("oi"),
            "disp":   e.get("disp"),
        }
    # deltaE
    deltaE_TS = (whole_E.get("TS") - whole_E.get("R")
                  if whole_E.get("TS") is not None and whole_E.get("R") is not None
                  else None)
    sum_TS = (sum(asr["TS"][k] for k in ["strain","elst","Pauli","oi","disp"])
               if all(asr["TS"][k] is not None
                      for k in ["strain","elst","Pauli","oi","disp"])
               else None)

    result = {
        "reaction_id": rxn_id,
        "schema_version": "asr_spec_v1_recovered",
        "adf_settings": {
            "functional": "BP86", "dispersion": "D3BJ",
            "basis": "TZ2P", "frozen_core": "None",
            "relativity": "ZORA_scalar", "integration": "Becke_Good",
            "scf_thr": 1e-6, "symmetry": "NoSym",
        },
        "irc_points": {
            z: {"energy_kcal_adf": whole_E.get(z)} for z in ZETA
        },
        "fragment_opt_energy_kcal": frag_opt_E,
        "asr_vector_kcal": asr,
        "consistency": {
            "sum_components_TS": sum_TS,
            "delta_E_DFT_TS_kcal": deltaE_TS,
            "diff_TS_kcal": (sum_TS - deltaE_TS) if sum_TS is not None and deltaE_TS is not None else None,
        },
        "status_at_queue": "RECOVERED_FROM_RKF",
        "recovery_note": "Result JSON re-generated from /tmp workdir rkfs (orig JSON was deleted).",
    }
    out_fp.write_text(json.dumps(result, indent=2))
    return "ok"


# Walk all workdirs
n_ok = n_skip = n_partial = 0
for d in TMP_ROOT.iterdir():
    if not d.is_dir(): continue
    r = recover(d.name)
    if r == "ok": n_ok += 1
    elif r == "skip_already": n_skip += 1
    else: n_partial += 1
print(f"recovered: {n_ok}  already-exists: {n_skip}  partial/in-progress: {n_partial}")
