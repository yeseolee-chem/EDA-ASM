"""Regenerate inputs for the 10 OOD rxns with tightened electronic settings.

- eda.inp:      VeryTightSCF + SlowConv + SOSCF + defgrid3
- fragA_R.inp:  VeryTightSCF + defgrid3
- fragB_R.inp:  VeryTightSCF + defgrid3
Deletes .out for all so runner retries.

Preserves the CURRENT TS/R partitions (reflects any user edits from the app).
Writes manifest_ood_retry_eda.txt and manifest_ood_retry_sp.txt.
"""
from __future__ import annotations
import json
from pathlib import Path
import ase.io
import pandas as pd
from ase.data import chemical_symbols

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
V8 = REPO / "outputs/v8_review"
RAW = V8 / "raw_geoms"
ORCA_ROOT = V8 / "orca_inputs"
SP_ROOT = V8 / "strain_sp"
MP = V8 / "manual_partitions.json"
OOD_CSV = V8 / "labels/ood_report_v8.csv"
MANIFEST_EDA = ORCA_ROOT / "manifest_ood_retry_eda.txt"
MANIFEST_SP = SP_ROOT / "manifest_ood_retry_sp.txt"


def _fam(rid):
    if rid.startswith("qmrxn20"):
        return rid.split("_")[0] + "_" + rid.split("_")[1]
    return rid.split("_")[0]


def write_eda_tight(rid, A, B, fA_c, fB_c, tc):
    ts_at = ase.io.read(str(RAW / rid / "TS.xyz"))
    Z = ts_at.get_atomic_numbers()
    pos = ts_at.get_positions()
    n = len(Z)
    frag_of = [None] * n
    for i in A: frag_of[i] = 1
    for i in B: frag_of[i] = 2
    if any(f is None for f in frag_of):
        raise ValueError(f"{rid}: unassigned")
    lines = [
        "! BLYP D3BJ def2-TZVP NoSym EDA VeryTightSCF SlowConv SOSCF defgrid3",
        "%maxcore 3500",
        "",
        "%scf",
        "  MaxIter 500",
        "end",
        "",
        "%eda",
        '  FRAG1 "BLYP D3BJ def2-TZVP NoSym VeryTightSCF SlowConv SOSCF defgrid3"',
        '  FRAG2 "BLYP D3BJ def2-TZVP NoSym VeryTightSCF SlowConv SOSCF defgrid3"',
        f"  FRAG1_C {fA_c}",
        "  FRAG1_M 1",
        f"  FRAG2_C {fB_c}",
        "  FRAG2_M 1",
        "end",
        "",
        f"* xyz {tc} 1",
    ]
    for i in range(n):
        sym = chemical_symbols[int(Z[i])]
        lines.append(f"{sym}({frag_of[i]})   {pos[i,0]:15.8f}   {pos[i,1]:15.8f}   {pos[i,2]:15.8f}")
    lines += ["*", ""]
    inp = ORCA_ROOT / rid / "eda.inp"
    inp.write_text("\n".join(lines))
    (ORCA_ROOT / rid / "eda.out").unlink(missing_ok=True)
    (ORCA_ROOT / rid / "eda.err").unlink(missing_ok=True)


def write_sp_tight(rid, frag_idx, charge, out_path):
    r_at = ase.io.read(str(RAW / rid / "R.xyz"))
    Z = r_at.get_atomic_numbers()
    pos = r_at.get_positions()
    lines = [
        "! BLYP D3BJ def2-TZVP NoSym VeryTightSCF SlowConv SOSCF defgrid3",
        "%maxcore 3500",
        "",
        "%scf",
        "  MaxIter 500",
        "end",
        "",
        f"* xyz {charge} 1",
    ]
    for i in frag_idx:
        sym = chemical_symbols[int(Z[i])]
        lines.append(f"{sym}   {pos[i,0]:15.8f}   {pos[i,1]:15.8f}   {pos[i,2]:15.8f}")
    lines += ["*", ""]
    out_path.write_text("\n".join(lines))


def main():
    m = json.loads(MP.read_text())
    ood = pd.read_csv(OOD_CSV)
    rids = ood.reaction_id.tolist()
    eda_manifest = []
    sp_manifest = []
    for rid in rids:
        e = m.get(rid, {})
        A = e.get("frag_A_indices", []); B = e.get("frag_B_indices", [])
        A_R = e.get("frag_A_indices_R", []); B_R = e.get("frag_B_indices_R", [])
        if not A or not B:
            print(f"SKIP {rid}: no TS partition"); continue
        fam = _fam(rid)
        tc = 0; fA_c = 0; fB_c = 0
        if fam in ("qmrxn20_sn2", "qmrxn20_e2"):
            fB_c = -1; tc = -1
        # For dipolar rxns with historical odd-electron issue, keep charge=-1
        if rid in ("dipolar_004594", "dipolar_005435"):
            fA_c = -1; fB_c = 0; tc = -1

        write_eda_tight(rid, A, B, fA_c, fB_c, tc)
        eda_manifest.append(rid)
        print(f"EDA tight: {rid}  |A|={len(A)}  |B|={len(B)}  charge=(A:{fA_c}, B:{fB_c}, tot:{tc})")

        # Strain SPs
        sp_dir = SP_ROOT / rid
        sp_dir.mkdir(exist_ok=True)
        # fragA charge
        cA = fA_c
        cB = 0 if fam not in ("qmrxn20_sn2","qmrxn20_e2") else -1
        write_sp_tight(rid, A_R, cA, sp_dir / "fragA_R.inp")
        write_sp_tight(rid, B_R, cB, sp_dir / "fragB_R.inp")
        (sp_dir / "fragA_R.out").unlink(missing_ok=True)
        (sp_dir / "fragB_R.out").unlink(missing_ok=True)
        (sp_dir / "fragA_R.err").unlink(missing_ok=True)
        (sp_dir / "fragB_R.err").unlink(missing_ok=True)
        sp_manifest.append(f"{rid} fragA")
        sp_manifest.append(f"{rid} fragB")

    MANIFEST_EDA.write_text("\n".join(eda_manifest) + "\n")
    MANIFEST_SP.write_text("\n".join(sp_manifest) + "\n")
    print(f"\nEDA manifest ({len(eda_manifest)}): {MANIFEST_EDA}")
    print(f"SP manifest ({len(sp_manifest)}):  {MANIFEST_SP}")


if __name__ == "__main__":
    main()
