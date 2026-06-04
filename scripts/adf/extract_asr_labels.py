#!/usr/bin/env python
"""Extract 5-channel ASR vector per reaction from BLYP-D3(BJ) ADF outputs.

For each reaction with status.json exit_code=0, parse:
  c1_fragA_ts.out      → E(A@TS)        ('Total Bonding Energy' in kcal/mol)
  c2_fragB_ts.out      → E(B@TS)
  c3_eda.out           → Pauli, V_elst, Orb, Disp (kcal/mol)
  c4_fragA_opt.out     → E(A_relaxed)   [skip if single-atom]
  c5_fragB_opt.out     → E(B_relaxed)   [skip if single-atom]

Assemble:
  strain_A = E(A@TS) - E(A_relaxed)   [0 if single-atom]
  strain_B = E(B@TS) - E(B_relaxed)   [0 if single-atom]
  strain   = strain_A + strain_B
  Ea_ASM   = strain + Pauli + V_elst + Orb + Disp

Output: adf_outputs/parsed/asr_labels.parquet
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]

# Matches lines like:
#   Total Bonding Energy:           -0.522911684824970        -14.2292         -328.13         -1372.90
_VAL = re.compile(
    r"^\s*(?P<label>.+?):\s+(?P<ha>-?\d+\.\d+)\s+(?P<ev>-?\d+\.\d+)\s+"
    r"(?P<kcal>-?\d+\.\d+)\s+(?P<kj>-?\d+\.\d+)\s*$"
)


def first_value(text: str, label_regex: str) -> float | None:
    rex = re.compile(label_regex, re.IGNORECASE)
    for line in text.splitlines():
        m = _VAL.match(line)
        if m and rex.search(m["label"].strip()):
            return float(m["kcal"])
    return None


def parse_fragment_bond(path: Path) -> float | None:
    if not path.is_file():
        return None
    text = path.read_text(errors="replace")
    return first_value(text, r"^Total\s+Bonding\s+Energy$")


def parse_eda(path: Path) -> dict | None:
    if not path.is_file():
        return None
    text = path.read_text(errors="replace")
    pauli = first_value(text, r"^Total\s+Pauli\s+Repulsion$")
    elst = first_value(text, r"^Electrostatic\s+Interaction$")
    orb = first_value(text, r"^Total\s+Orbital\s+Interactions$")
    total = first_value(text, r"^Total\s+Bonding\s+Energy$")
    disp = (
        first_value(text, r"^Dispersion\s+Energy$")
        or first_value(text, r"^Total\s+Dispersion\s+Energy$")
        or first_value(text, r"^Dispersion$")
        or 0.0
    )
    if pauli is None or elst is None or orb is None or total is None:
        return None
    return {
        "Pauli_kcal": pauli, "V_elst_kcal": elst,
        "E_orb_kcal": orb, "E_disp_kcal": disp,
        "E_int_total_kcal": total,
    }


def extract_one(rxn_dir: Path) -> dict | None:
    status = rxn_dir / "status.json"
    if not status.is_file():
        return None
    try:
        s = json.loads(status.read_text())
    except Exception:
        return None
    if s.get("exit_code") != 0:
        return None

    eda = parse_eda(rxn_dir / "c3_eda.out")
    if eda is None:
        return None
    e_fA_TS = parse_fragment_bond(rxn_dir / "c1_fragA_ts.out")
    e_fB_TS = parse_fragment_bond(rxn_dir / "c2_fragB_ts.out")
    if e_fA_TS is None or e_fB_TS is None:
        return None

    single_a = s["calc_status"].get("c4_fragA_opt") == "n/a_single_atom"
    single_b = s["calc_status"].get("c5_fragB_opt") == "n/a_single_atom"
    e_fA_rel = e_fA_TS if single_a else parse_fragment_bond(rxn_dir / "c4_fragA_opt.out")
    e_fB_rel = e_fB_TS if single_b else parse_fragment_bond(rxn_dir / "c5_fragB_opt.out")
    if e_fA_rel is None or e_fB_rel is None:
        return None

    strain_A = 0.0 if single_a else (e_fA_TS - e_fA_rel)
    strain_B = 0.0 if single_b else (e_fB_TS - e_fB_rel)
    strain = strain_A + strain_B
    Ea_asm = strain + eda["Pauli_kcal"] + eda["V_elst_kcal"] + eda["E_orb_kcal"] + eda["E_disp_kcal"]

    return {
        "reaction_id": s["reaction_id"],
        "family": s["reaction_id"].split("_")[0] + (
            "_" + s["reaction_id"].split("_")[1]
            if s["reaction_id"].startswith("qmrxn20_") else ""
        ),
        # 5-channel ASR vector
        "E_strain_kcal": strain,
        "Pauli_kcal": eda["Pauli_kcal"],
        "V_elst_kcal": eda["V_elst_kcal"],
        "E_orb_kcal": eda["E_orb_kcal"],
        "E_disp_kcal": eda["E_disp_kcal"],
        # auxiliary
        "E_int_total_kcal": eda["E_int_total_kcal"],
        "Ea_ASM_kcal": Ea_asm,
        "strain_A_kcal": strain_A,
        "strain_B_kcal": strain_B,
        "E_fA_TS_kcal": e_fA_TS,
        "E_fB_TS_kcal": e_fB_TS,
        "E_fA_relaxed_kcal": e_fA_rel,
        "E_fB_relaxed_kcal": e_fB_rel,
        # provenance
        "single_atom_a": single_a,
        "single_atom_b": single_b,
        "functional": s.get("functional"),
        "basis": s.get("basis"),
        "wallclock_s": s.get("wallclock_s"),
        "ams_version": s.get("ams_version"),
        "dataset_delta_Ea": s.get("dataset_delta_Ea"),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adf-root", type=Path, default=REPO / "adf_outputs")
    p.add_argument("--out", type=Path,
                   default=REPO / "adf_outputs" / "parsed" / "asr_labels.parquet")
    args = p.parse_args()

    rows = []
    n_skip = 0
    for rxn_dir in sorted(args.adf_root.glob("batch_*/*")):
        if not rxn_dir.is_dir():
            continue
        rec = extract_one(rxn_dir)
        if rec is None:
            n_skip += 1
            continue
        rows.append(rec)

    if not rows:
        print("no parseable status.json found", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame.from_records(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, compression="zstd")
    print(f"parsed {len(df)} reactions (skipped {n_skip})  →  {args.out}")
    print(df[["reaction_id", "family", "E_strain_kcal", "Pauli_kcal",
              "V_elst_kcal", "E_orb_kcal", "E_disp_kcal", "Ea_ASM_kcal"]]
          .head(10).to_string(index=False))


if __name__ == "__main__":
    main()
