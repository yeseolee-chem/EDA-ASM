"""Extract R/TS/P XYZ triples for v7 rgd1 rxns whose extracted_xyz dir
is missing (36 rxns per audit). Reads from RGD1_CHNO.h5.
"""
from pathlib import Path
import h5py
import numpy as np
import pandas as pd

RAW = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/rgd1")
H5 = RAW / "RGD1_CHNO.h5"
OUT = RAW / "extracted_xyz"
LABELS_V7 = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/labels/orca/orca_eda_labels_v7.parquet")

NUM2SYM = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}


def write_xyz(path, elements, coords, comment=""):
    with open(path, "w") as f:
        f.write(f"{len(elements)}\n{comment}\n")
        for e, xyz in zip(elements, coords):
            sym = NUM2SYM.get(int(e), "X")
            f.write(f"{sym:>2s}  {xyz[0]:12.6f} {xyz[1]:12.6f} {xyz[2]:12.6f}\n")


def main():
    df = pd.read_parquet(LABELS_V7)
    rgd1_rids = [r for r, f in zip(df.reaction_id, df.family) if f == "rgd1"]
    missing = [r for r in rgd1_rids if not (OUT / r).exists()]
    print(f"v7 rgd1: {len(rgd1_rids)}   missing raw: {len(missing)}")
    if not missing:
        return
    got = 0; still_missing = []
    with h5py.File(str(H5), "r") as hf:
        for rid in missing:
            key = rid.replace("rgd1_", "", 1)
            if key not in hf:
                still_missing.append(rid); continue
            rxn = hf[key]
            elements = np.array(rxn["elements"])
            rid_dir = OUT / rid
            rid_dir.mkdir(exist_ok=True)
            write_xyz(rid_dir / "R.xyz", elements, np.array(rxn["RG"]), rid + " reactant")
            write_xyz(rid_dir / "TS.xyz", elements, np.array(rxn["TSG"]), rid + " TS")
            write_xyz(rid_dir / "P.xyz", elements, np.array(rxn["PG"]), rid + " product")
            got += 1
    print(f"wrote {got} rxns to {OUT}")
    if still_missing:
        print(f"still missing (not in h5): {len(still_missing)}: {still_missing[:5]}")


if __name__ == "__main__":
    main()
