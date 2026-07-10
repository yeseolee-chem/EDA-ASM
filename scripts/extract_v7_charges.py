"""Extract per-reaction fragment charges + multiplicities from v7 eda.inp files.

For each rid in outputs/final_776_v7/orca_eda/<rid>/eda.inp, pull
  FRAG1_C, FRAG1_M, FRAG2_C, FRAG2_M and the total charge from
  the `* xyz {total_charge} {mult}` line.

Output: labels/orca/orca_eda_charges_v7.parquet with columns
  reaction_id, total_charge, total_mult,
  fragment_charge_a, fragment_mult_a, fragment_charge_b, fragment_mult_b
"""
from __future__ import annotations
import re
from pathlib import Path

import pandas as pd

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
EDA_ROOT = REPO / "outputs/final_776_v7/orca_eda"
OUT_PARQUET = REPO / "labels/orca/orca_eda_charges_v7.parquet"


def parse_one(inp_path: Path) -> dict:
    txt = inp_path.read_text()
    def _one(rx):
        m = re.search(rx, txt)
        return int(m.group(1)) if m else None
    fA_c = _one(r"FRAG1_C\s+(-?\d+)")
    fA_m = _one(r"FRAG1_M\s+(-?\d+)")
    fB_c = _one(r"FRAG2_C\s+(-?\d+)")
    fB_m = _one(r"FRAG2_M\s+(-?\d+)")
    m = re.search(r"\*\s+xyz\s+(-?\d+)\s+(\d+)", txt)
    tot_c = int(m.group(1)) if m else (fA_c + fB_c if fA_c is not None else None)
    tot_m = int(m.group(2)) if m else 1
    return dict(
        total_charge=tot_c, total_mult=tot_m,
        fragment_charge_a=fA_c, fragment_mult_a=fA_m,
        fragment_charge_b=fB_c, fragment_mult_b=fB_m,
    )


def main():
    rows = []
    for rid_dir in sorted(EDA_ROOT.iterdir()):
        inp = rid_dir / "eda.inp"
        if not inp.exists():
            continue
        row = {"reaction_id": rid_dir.name, **parse_one(inp)}
        rows.append(row)
    df = pd.DataFrame(rows)
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"wrote {len(df)} rows -> {OUT_PARQUET}")
    print(df.head())
    print("nan counts:", df.isna().sum().to_dict())


if __name__ == "__main__":
    main()
