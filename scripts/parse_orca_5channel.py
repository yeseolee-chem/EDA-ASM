"""Parse ORCA EDA pass + strain pass and emit orca_eda_labels_v7.parquet with
5 channels per reaction (Pauli+ΔXC, V_elst, E_orb, E_disp, E_strain).

Convention (ADF-style, per project memory):
  Pauli_channel  = ORCA Pauli Energy + Delta E^0(XC)   (kcal/mol)
  V_elst_channel = ORCA Electrostatic Energy           (kcal/mol)
  E_orb_channel  = ORCA Orbital Energy                 (kcal/mol)
  E_disp_channel = ORCA Delta Dispersion               (kcal/mol)
  E_strain_channel = E(fragA at TS) - E(fragA opt)
                   + E(fragB at TS) - E(fragB opt)      (kcal/mol)

Inputs:
  outputs/orca_eda/inputs/<rid>/eda.out                (EDA — 4 base channels)
  outputs/orca_eda/inputs/<rid>/eda_frag{1,2}.out      (fragment SP @ TS geometry)
  outputs/orca_strain/inputs/<rid>__f{A,B}/opt.out     (fragment opt)

Output:
  labels/orca/orca_eda_labels_v7.parquet
"""
from __future__ import annotations
import re
from pathlib import Path

import pandas as pd

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
EDA_ROOT = REPO / "outputs/orca_eda/inputs"
STRAIN_ROOT = REPO / "outputs/orca_strain/inputs"
OUT_PARQUET = REPO / "labels/orca/orca_eda_labels_v7.parquet"

HARTREE_TO_KCAL = 627.5094740631

# EDA output: "     Bond Energy                   -1.0950039866      -687.13"
# NOTE: no ^ anchor because we scan whole file text with finditer (multiline).
EDA_CHANNEL_RE = {
    "bond":  re.compile(r"Bond Energy\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)"),
    "orb":   re.compile(r"Orbital Energy\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)"),
    "elst":  re.compile(r"Electrostatic Energy\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)"),
    "pauli": re.compile(r"Pauli Energy\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)"),
    "dxc":   re.compile(r"Delta E\^0\(XC\)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)"),
    "disp":  re.compile(r"Delta Dispersion\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)"),
}
FSP_RE = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)")


def _tail_terminated(p: Path) -> bool:
    if not p.exists(): return False
    with open(p, "rb") as f:
        f.seek(0, 2); size = f.tell(); f.seek(max(0, size - 4000))
        return b"ORCA TERMINATED NORMALLY" in f.read()


def _parse_eda_channels(eda_out: Path) -> dict:
    """Return per-channel kcal/mol values (bond, orb, elst, pauli, dxc, disp)."""
    out = {}
    if not eda_out.exists(): return out
    text = eda_out.read_text(errors="ignore")
    for name, rx in EDA_CHANNEL_RE.items():
        for m in rx.finditer(text):
            out[name] = float(m.group(2))  # kcal/mol
    return out


def _parse_final_sp(out_file: Path) -> float:
    """Return the LAST FINAL SINGLE POINT ENERGY (Hartree)."""
    if not out_file.exists(): return float("nan")
    text = out_file.read_text(errors="ignore")
    matches = FSP_RE.findall(text)
    return float(matches[-1]) if matches else float("nan")


def main():
    rows = []
    for rid_dir in sorted(EDA_ROOT.iterdir()):
        if not rid_dir.is_dir() or "broken" in rid_dir.name: continue
        rid = rid_dir.name
        eda_out = rid_dir / "eda.out"
        if not _tail_terminated(eda_out):
            continue
        ch = _parse_eda_channels(eda_out)
        if not all(k in ch for k in ("pauli", "dxc", "elst", "orb", "disp")):
            continue

        # Strain: E(fragX @ TS) - E(fragX relaxed)
        # E(fragX @ TS) is the FINAL SP in eda_frag{1,2}.out (from the EDA aux run)
        # E(fragX relaxed) is the FINAL SP in orca_strain/<rid>__f{A,B}/opt.out
        # For monatomic fragments (typical QMrxn20 nucleophile), opt is impossible
        # and E_strain = 0 by definition (nothing to relax).
        e_ts_A = _parse_final_sp(rid_dir / "eda_frag1.out")
        e_ts_B = _parse_final_sp(rid_dir / "eda_frag2.out")
        e_opt_A = _parse_final_sp(STRAIN_ROOT / f"{rid}__fA" / "opt.out")
        e_opt_B = _parse_final_sp(STRAIN_ROOT / f"{rid}__fB" / "opt.out")

        # Count atoms per fragment from eda.inp fragment tags
        n_A = n_B = 0
        try:
            in_xyz = False
            for line in (rid_dir / "eda.inp").read_text().splitlines():
                s = line.strip()
                if s.startswith("* xyz"): in_xyz = True; continue
                if in_xyz and s == "*": break
                if in_xyz:
                    m = re.match(r"^\s*[A-Z][a-z]?\((\d+)\)", line)
                    if m:
                        if m.group(1) == "1": n_A += 1
                        else: n_B += 1
        except Exception:
            pass

        strain_A_kcal = float("nan")
        strain_B_kcal = float("nan")
        if n_A == 1:
            strain_A_kcal = 0.0
        elif all(x == x for x in (e_ts_A, e_opt_A)):
            strain_A_kcal = (e_ts_A - e_opt_A) * HARTREE_TO_KCAL
        if n_B == 1:
            strain_B_kcal = 0.0
        elif all(x == x for x in (e_ts_B, e_opt_B)):
            strain_B_kcal = (e_ts_B - e_opt_B) * HARTREE_TO_KCAL

        # 5-channel labels (kcal/mol)
        rows.append({
            "reaction_id": rid,
            "family": rid.split("_")[0] if not rid.startswith("qmrxn20")
                      else "_".join(rid.split("_")[:2]),
            # 5 channels
            "Pauli_kcal":      ch["pauli"] + ch["dxc"],  # merged with ΔE^0(XC)
            "V_elst_kcal":     ch["elst"],
            "E_orb_kcal":      ch["orb"],
            "E_disp_kcal":     ch["disp"],
            "E_strain_kcal":   strain_A_kcal + strain_B_kcal,
            # Bookkeeping
            "E_int_bond_kcal": ch["bond"],                 # ΣΔchannels = Bond Energy
            "E_strain_A_kcal": strain_A_kcal,
            "E_strain_B_kcal": strain_B_kcal,
            "Ea_ASM_kcal":     ch["bond"] + strain_A_kcal + strain_B_kcal,
            "E_ts_fragA_Eh":   e_ts_A,
            "E_ts_fragB_Eh":   e_ts_B,
            "E_opt_fragA_Eh":  e_opt_A,
            "E_opt_fragB_Eh":  e_opt_B,
        })
    df = pd.DataFrame(rows)
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"wrote {len(df)} rows → {OUT_PARQUET}")
    # Summary
    if len(df):
        complete = df.dropna(subset=["E_strain_kcal"])
        print(f"complete rows (all 5 channels): {len(complete)}")
        print(df.family.value_counts().to_string())


if __name__ == "__main__":
    main()
