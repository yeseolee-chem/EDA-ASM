#!/usr/bin/env python3
"""SPEC16rev Step 6 — export labels_b3lypgeom_v2.pkl + metadata JSON."""
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

BASE = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/b3lyp_full")


def main():
    R = pd.read_csv(BASE / "artifacts" / "labels_b3lyp_full.csv")
    bad_build = pd.read_csv(BASE / "artifacts" / "build_failures.csv")
    bad_parse = pd.read_csv(BASE / "artifacts" / "parse_failures.csv")

    strain_invalid = int((~R["strain_valid"]).sum())

    R.to_pickle(BASE / "artifacts" / "labels_b3lypgeom_v2.pkl")
    print(f"wrote labels_b3lypgeom_v2.pkl ({len(R)} rows)")

    try:
        commit = subprocess.check_output(
            ["git", "-C", str(BASE), "rev-parse", "HEAD"],
            text=True).strip()
    except Exception:
        commit = "unknown"

    meta = {
        "geometry_source": "coleygroup/dipolar_cycloaddition_dataset",
        "geometry_level": "B3LYP-D3BJ/def2-SVP (autodE)",
        "energy_level": "B3LYP-D3BJ/def2-TZVP, CPCM(SMD, water)",
        "notation": "B3LYP-D3BJ/def2-TZVP // B3LYP-D3BJ/def2-SVP",
        "reoptimized": False,
        "reoptimization_rationale": (
            "Relaxed and distorted fragment geometries are taken from the same "
            "source at an identical level of theory; re-optimizing only the "
            "relaxed reference would systematically inflate distortion energies "
            "and break the // convention."
        ),
        "n_reactions": int(len(R)),
        "excluded": {
            "build_fail": int(len(bad_build)),
            "parse_fail": int(len(bad_parse)),
            "strain_invalid": strain_invalid,
        },
        "orca_version": "6.1.1",
        "channels": ["elst_dft", "pauli_dft", "oi_dft",
                     "disp_dft", "cpcm_dft", "cds_dft"],
        "strain_fields": ["d1_b3lyp", "d2_b3lyp"],
        "generated": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
    }
    with open(BASE / "artifacts" / "labels_b3lypgeom_v2_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote metadata JSON")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
