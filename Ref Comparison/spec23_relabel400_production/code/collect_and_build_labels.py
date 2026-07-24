"""spec23 collector — parse ORCA outputs into the label parquet.

Reads:
  results/job_manifest.csv     — 1200 rows
  {workdir}/{eda|fragA_opt|fragB_opt}/*.out

Writes:
  outputs/spec23_wb97x3c/labels/dipolar_400_wb97x3c.parquet
  results/attempts.csv                — per-rxn timing + attempt count
  results/needs_attention.csv         — rxns whose EDA never converged
  results/failed_jobs.csv             — every abnormal termination
  results/asm_residual.csv            — G23-F distribution

Parsing rules
  EDA channels (kcal/mol) — read the block:
    Bond Energy     ...       -0.39
    Orbital Energy  ...       -7.40
    Electrostatic   ...       -9.40
    Pauli Energy    ...       56.28
    Delta E^0(XC)   ...      -35.05
    Delta Dispersion...       -4.82
  Total interaction = Bond Energy. ASM identity closure:
    strain + (pauli + elst + orb_raw + xc + disp) ≈ act
  E_CP(D_A) = last "FINAL SINGLE POINT ENERGY" in eda_frag1.out (Hartree)
  E_CP(D_B) = same in eda_frag2.out
  E_TS      = last "FINAL SINGLE POINT ENERGY" in eda.out
  E_R_A_opt = last "FINAL SINGLE POINT ENERGY" in fragA_opt.out
  E_R_B_opt = same in fragB_opt.out
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
STAGE = REPO / "Ref Comparison/spec23_relabel400_production"
JOB_MANIFEST = STAGE / "results/job_manifest.csv"
OUT_LABELS_DIR = REPO / "outputs/spec23_wb97x3c/labels"
OUT_LABELS = OUT_LABELS_DIR / "dipolar_400_wb97x3c.parquet"

OUT_ATTEMPTS = STAGE / "results/attempts.csv"
OUT_NEEDS = STAGE / "results/needs_attention.csv"
OUT_FAILED = STAGE / "results/failed_jobs.csv"
OUT_ASM = STAGE / "results/asm_residual.csv"

HARTREE_TO_KCAL = 627.5094740631
NORMAL_TERM = "****ORCA TERMINATED NORMALLY****"
FSPE_RE = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)")

EDA_LABELS = {
    "Bond Energy":         "bond_kcal",
    "Orbital Energy":      "orb_raw_kcal",
    "Electrostatic Energy": "elst_kcal",
    "Pauli Energy":        "pauli_kcal",
    "Delta E^0(XC)":       "xc_kcal",
    "Delta Dispersion":    "disp_kcal",
}


def last_fspe(path: Path) -> float | None:
    if not path.exists():
        return None
    last = None
    with open(path) as f:
        for line in f:
            m = FSPE_RE.search(line)
            if m:
                last = float(m.group(1))
    return last


def parse_eda_channels(path: Path) -> dict | None:
    """Return the 6-channel breakdown in kcal/mol from the summary block."""
    if not path.exists():
        return None
    text = path.read_text()
    out = {}
    for label, key in EDA_LABELS.items():
        # match e.g. "     Orbital Energy                -0.0117904826        -7.40"
        pat = re.compile(re.escape(label) + r"\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)")
        m = pat.search(text)
        if m:
            out[key] = float(m.group(2))
    return out if len(out) == len(EDA_LABELS) else None


def terminated_normally(path: Path) -> bool:
    if not path.exists():
        return False
    with open(path) as f:
        for line in f:
            if NORMAL_TERM in line:
                return True
    return False


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    jm = pd.read_csv(JOB_MANIFEST)
    # per reaction, gather three outputs
    per_rxn = {}
    failed = []
    for _, r in jm.iterrows():
        rid = r["reaction_id"]
        d = per_rxn.setdefault(rid, {"reaction_id": rid,
                                      "sub_source": r["sub_source"],
                                      "reaction_number": int(r["reaction_number"])})
        out = Path(r["out"])
        ok = terminated_normally(out)
        d[r["jobtype"] + "_ok"] = ok
        if not ok:
            failed.append({"reaction_id": rid, "jobtype": r["jobtype"],
                            "out": str(out), "exists": out.exists()})

    labels = []
    asm_rows = []
    for rid, r in per_rxn.items():
        need = ("eda_ok", "fragA_opt_ok", "fragB_opt_ok")
        if not all(r.get(k) for k in need):
            continue
        eda_out = Path([row["out"] for _, row in jm.iterrows()
                        if row["reaction_id"] == rid and row["jobtype"] == "eda"][0])
        eda_dir = eda_out.parent
        e_ts = last_fspe(eda_out)
        e_da = last_fspe(eda_dir / "eda_frag1.out")
        e_db = last_fspe(eda_dir / "eda_frag2.out")
        chans = parse_eda_channels(eda_out)
        opt_a = Path([row["out"] for _, row in jm.iterrows()
                       if row["reaction_id"] == rid and row["jobtype"] == "fragA_opt"][0])
        opt_b = Path([row["out"] for _, row in jm.iterrows()
                       if row["reaction_id"] == rid and row["jobtype"] == "fragB_opt"][0])
        e_ra = last_fspe(opt_a)
        e_rb = last_fspe(opt_b)

        if None in (e_ts, e_da, e_db, e_ra, e_rb) or chans is None:
            failed.append({"reaction_id": rid, "jobtype": "parse",
                            "out": str(eda_out), "exists": True})
            continue

        # strain, act, int
        strain_A = (e_da - e_ra) * HARTREE_TO_KCAL
        strain_B = (e_db - e_rb) * HARTREE_TO_KCAL
        strain = strain_A + strain_B
        act = (e_ts - e_ra - e_rb) * HARTREE_TO_KCAL
        # channels — orb_kcal = orb_raw + xc (paper convention)
        orb = chans["orb_raw_kcal"] + chans["xc_kcal"]
        int_eda = chans["pauli_kcal"] + chans["elst_kcal"] + orb + chans["disp_kcal"]

        labels.append({
            "reaction_id":     rid,
            "sub_source":      r["sub_source"],
            "reaction_number": r["reaction_number"],
            "pauli_kcal":      chans["pauli_kcal"],
            "elst_kcal":       chans["elst_kcal"],
            "orb_raw_kcal":    chans["orb_raw_kcal"],
            "xc_kcal":         chans["xc_kcal"],
            "orb_kcal":        orb,       # modelling column
            "disp_kcal":       chans["disp_kcal"],
            "int_eda_kcal":    int_eda,
            "strain_A_kcal":   strain_A,
            "strain_B_kcal":   strain_B,
            "strain_kcal":     strain,
            "act_kcal":        act,
            "E_TS_hartree":    e_ts,
            "E_DA_hartree":    e_da,
            "E_DB_hartree":    e_db,
            "E_RA_hartree":    e_ra,
            "E_RB_hartree":    e_rb,
            "bond_energy_kcal_from_orca": chans["bond_kcal"],
        })

        # G23-F: ASM identity residual
        four_sum = chans["pauli_kcal"] + chans["elst_kcal"] + chans["orb_raw_kcal"] + chans["xc_kcal"] + chans["disp_kcal"]
        resid = abs(strain + four_sum - act)
        asm_rows.append({"reaction_id": rid, "sub_source": r["sub_source"],
                          "resid_kcal": resid})

    # Write outputs
    OUT_LABELS_DIR.mkdir(parents=True, exist_ok=True)
    STAGE.joinpath("results").mkdir(exist_ok=True)
    if labels:
        pd.DataFrame(labels).to_parquet(OUT_LABELS, index=False)
        print(f"[write] {OUT_LABELS}  n={len(labels)}  sha256={sha256(OUT_LABELS)[:16]}")
    if asm_rows:
        pd.DataFrame(asm_rows).to_csv(OUT_ASM, index=False)
    if failed:
        pd.DataFrame(failed).to_csv(OUT_FAILED, index=False)
    print(f"[summary] labels={len(labels)}  failed_or_incomplete={len(failed)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
