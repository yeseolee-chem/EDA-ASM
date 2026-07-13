"""Parse ORCA EDA + strain SP outputs into a 5-channel v8 label parquet.

Channels (kcal/mol, following CLAUDE.md ASM convention):
  Pauli    := Pauli Energy + Delta E^0(XC)   (steric wall)
  elst     := Electrostatic Energy
  orb      := Orbital Energy
  disp     := Delta Dispersion
  strain   := [E_A(TS) - E_A(R)] + [E_B(TS) - E_B(R)]  ← from separate SPs
  int_eda  := Bond Energy (total interaction at TS geom, sanity check)
  act      := int_eda + strain (total activation proxy)

Output: outputs/v8_review/labels/labels_v8_5channel.parquet
"""
from __future__ import annotations
import re, json
from pathlib import Path
import pandas as pd

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
V8 = REPO / "outputs/v8_review"
ORCA_ROOT = V8 / "orca_inputs"
SP_ROOT = V8 / "strain_sp"
OUT_DIR = V8 / "labels"; OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PQ = OUT_DIR / "labels_v8_5channel.parquet"

HARTREE_TO_KCAL = 627.5094740631


def _fam(rid):
    if rid.startswith("qmrxn20"):
        return rid.split("_")[0] + "_" + rid.split("_")[1]
    return rid.split("_")[0]


def parse_eda_out(out_path):
    """Parse the 'Energy Decomposition Analysis' table from eda.out.
    Returns dict with kcal/mol values or None if section missing."""
    txt = out_path.read_text(errors="ignore")
    if "ORCA TERMINATED NORMALLY" not in txt:
        return None
    m = re.search(r'Energy Decomposition Analysis(.*?)NOCV analysis',
                  txt, re.DOTALL)
    if not m:
        return None
    section = m.group(1)
    def _grab(key):
        rx = rf'{re.escape(key)}\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)'
        m2 = re.search(rx, section)
        if not m2:
            return None
        return float(m2.group(2))  # kcal/mol column
    return {
        "int_eda":   _grab("Bond Energy"),
        "orb":       _grab("Orbital Energy"),
        "elst":      _grab("Electrostatic Energy"),
        "pauli_raw": _grab("Pauli Energy"),
        "dE0_xc":    _grab("Delta E^0(XC)"),
        "disp":      _grab("Delta Dispersion"),
    }


def parse_final_energy(path):
    """Get the last 'FINAL SINGLE POINT ENERGY' (hartree) from an ORCA out."""
    if not path.exists():
        return None
    txt = path.read_text(errors="ignore")
    if "ORCA TERMINATED NORMALLY" not in txt:
        return None
    m = re.findall(r'FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)', txt)
    if not m:
        return None
    return float(m[-1])


def main():
    rows = []
    n_ok = 0; n_miss_eda = 0; n_miss_sp = 0
    for d in sorted(ORCA_ROOT.iterdir()):
        if not d.is_dir():
            continue
        rid = d.name
        eda_out = d / "eda.out"
        eda = parse_eda_out(eda_out)
        if eda is None:
            n_miss_eda += 1
            continue
        # Fragment SPs at TS geometry (embedded in EDA output)
        E_A_TS = parse_final_energy(d / "eda_frag1.out")
        E_B_TS = parse_final_energy(d / "eda_frag2.out")
        # Fragment SPs at R geometry
        E_A_R = parse_final_energy(SP_ROOT / rid / "fragA_R.out")
        E_B_R = parse_final_energy(SP_ROOT / rid / "fragB_R.out")
        if None in (E_A_TS, E_B_TS, E_A_R, E_B_R):
            n_miss_sp += 1
            continue

        strain_A = (E_A_TS - E_A_R) * HARTREE_TO_KCAL
        strain_B = (E_B_TS - E_B_R) * HARTREE_TO_KCAL
        strain = strain_A + strain_B

        pauli = eda["pauli_raw"] + (eda["dE0_xc"] or 0.0)
        act = eda["int_eda"] + strain

        rows.append({
            "reaction_id": rid,
            "family": _fam(rid),
            "pauli_kcal": pauli,
            "elst_kcal": eda["elst"],
            "orb_kcal": eda["orb"],
            "disp_kcal": eda["disp"],
            "strain_kcal": strain,
            "int_eda_kcal": eda["int_eda"],
            "act_kcal": act,
            "strain_A_kcal": strain_A,
            "strain_B_kcal": strain_B,
            "E_A_TS_hartree": E_A_TS,
            "E_B_TS_hartree": E_B_TS,
            "E_A_R_hartree": E_A_R,
            "E_B_R_hartree": E_B_R,
        })
        n_ok += 1

    df = pd.DataFrame(rows)
    df.to_parquet(OUT_PQ, index=False)
    print(f"Assembled: {n_ok} rxns   missing_eda={n_miss_eda}   missing_sp={n_miss_sp}")
    print(f"Output: {OUT_PQ}")
    if n_ok:
        print(df.groupby("family").size().to_string())
        print()
        print("=== channel summary (kcal/mol) ===")
        for ch in ["pauli_kcal","elst_kcal","orb_kcal","disp_kcal","strain_kcal","act_kcal"]:
            print(f"  {ch:15}  mean={df[ch].mean():>8.2f}  std={df[ch].std():>7.2f}  min={df[ch].min():>8.2f}  max={df[ch].max():>8.2f}")


if __name__ == "__main__":
    main()
