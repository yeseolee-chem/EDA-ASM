"""Generate counterpoise-corrected fragment SP inputs at R geometry.

Fix for the v8 BSSE bug (see memory/v8_strain_bugs.md):
  eda_frag{1,2}.out (numerator) include ghost basis from the opposing
  fragment at TS positions. The reference strain_sp/frag{A,B}_R.out
  used to be ghost-free — leaking pure BSSE (~-15 kcal/mol per
  monatomic nucleophile) into the label.
  Fix: put the opposing fragment's atoms at R positions as ghost basis.

Partition convention:
  * R.xyz atom ordering is INDEPENDENT of TS.xyz (raw datasets ship
    with no atom-mapping between R and TS). The user has manually
    reviewed BOTH partitions, so:
      TS-side: use `frag_A_indices` / `frag_B_indices` (already used
        by eda_frag{1,2}.out at TS geometry).
      R-side:  use `frag_A_indices_R` / `frag_B_indices_R` at R.xyz.
  * Approximate BSSE cancellation still works because the ghost basis
    at R has the same atom composition (same fragment) as the ghost
    at TS — only the position differs.

Theory level AND per-fragment charge are read from the actual v8
eda.inp for each rid (FRAG1_C / FRAG2_C fields) — matching whatever
charge the EDA numerator used. Hardcoded defaults per family are
only a fallback if parse fails.

Output:
  outputs/v9_review/strain_sp_cp/{rid}/fragA_R.inp
  outputs/v9_review/strain_sp_cp/{rid}/fragB_R.inp
  outputs/v9_review/manifest_sp.txt   ← "SP <rid> <fragA|fragB>"

Charges match EDA convention:
  dipolar / rgd1:   (fA=0, fB=0)
  qmrxn20 e2/sn2:   (fA=0, fB=-1)
"""
from __future__ import annotations
import json
from pathlib import Path
import ase.io
from ase.data import chemical_symbols

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
V8   = REPO / "outputs/v8_review"
V9   = REPO / "outputs/v9_review"
RAW  = V8 / "raw_geoms"
MANUAL_JSON = V8 / "manual_partitions.json"
LABELS_LOCKED = V8 / "labels/labels_v8_5channel.LOCKED_799.parquet"

SP_ROOT  = V9 / "strain_sp_cp"; SP_ROOT.mkdir(parents=True, exist_ok=True)
MANIFEST = V9 / "manifest_sp.txt"


def family(rid: str) -> str:
    if rid.startswith("qmrxn20"):
        return rid.split("_")[0] + "_" + rid.split("_")[1]
    return rid.split("_")[0]


def read_theory_line(edp_frag_inp: Path) -> str:
    """Return the first '!' line from eda_frag*.inp so we match SCF settings."""
    if not edp_frag_inp.exists():
        # Reasonable default matching most rids
        return "! BLYP D3BJ def2-TZVP NoSym VeryTightSCF SlowConv SOSCF defgrid3"
    for line in edp_frag_inp.read_text().splitlines():
        s = line.strip()
        if s.startswith("!"):
            # Normalize spacing: "!X..." → "! X..."
            body = s[1:].lstrip()
            return f"! {body}"
    return "! BLYP D3BJ def2-TZVP NoSym VeryTightSCF SlowConv SOSCF defgrid3"


def read_fragment_charges(eda_inp: Path, fam: str) -> tuple[int, int]:
    """Return (fragA_charge, fragB_charge) parsed from v8 eda.inp FRAG*_C
    fields. Falls back to hardcoded family defaults if either is missing.
    Fixed 2026-07-13: 2 dipolar rxns have anionic fragA (charge=-1); the
    old hardcoded fA=0 default produced 41-electron singlet errors."""
    import re
    fA_default = 0
    fB_default = -1 if fam in ("qmrxn20_sn2", "qmrxn20_e2") else 0
    if not eda_inp.exists():
        return fA_default, fB_default
    txt = eda_inp.read_text()
    mA = re.search(r"FRAG1_C\s+(-?\d+)", txt)
    mB = re.search(r"FRAG2_C\s+(-?\d+)", txt)
    fA = int(mA.group(1)) if mA else fA_default
    fB = int(mB.group(1)) if mB else fB_default
    return fA, fB


def write_cp_sp(rid: str,
                real_idx: list[int],
                ghost_idx: list[int],
                real_charge: int,
                r_atoms,
                theory: str,
                out_path: Path) -> None:
    """Write ORCA SP for real_idx atoms + ghost basis from ghost_idx atoms,
    all at R coordinates. Charge / multiplicity are the isolated-fragment
    values (spin=1 singlet per EDA convention). Ghost atoms carry no
    electrons and don't affect charge."""
    Z = r_atoms.get_atomic_numbers()
    pos = r_atoms.get_positions()

    lines = [
        theory,
        "%maxcore 3500",
        "",
        "%scf",
        "  MaxIter 500",
        "end",
        "",
        f"* xyz {real_charge} 1",
    ]
    # Real atoms
    for i in real_idx:
        sym = chemical_symbols[int(Z[i])]
        x, y, z = pos[i]
        lines.append(f"{sym:<3s}   {x:15.8f}   {y:15.8f}   {z:15.8f}")
    # Ghost atoms (ORCA syntax: 'Element:' immediately followed by colon)
    for i in ghost_idx:
        sym = chemical_symbols[int(Z[i])]
        x, y, z = pos[i]
        lines.append(f"{sym+':':<3s}   {x:15.8f}   {y:15.8f}   {z:15.8f}")
    lines.append("*")
    lines.append("")
    out_path.write_text("\n".join(lines))


def main() -> None:
    manual = json.loads(MANUAL_JSON.read_text())

    # v9 partition override: applies to rids the user has since corrected
    # (e.g. 4 SN2 rids where R-fragB was mistagged as the wrong element).
    override_path = V9 / "partition_override_v9.json"
    if override_path.exists():
        override = json.loads(override_path.read_text())
        for rid, patch in override.items():
            if rid in manual:
                manual[rid] = {**manual[rid], **patch}
        print(f"applied partition override for {len(override)} rids: {sorted(override)}")

    # Restrict to the 799 rids in LOCKED_799 so v9 and v8 line up 1:1.
    import pandas as pd
    locked_rids = set(pd.read_parquet(LABELS_LOCKED).reaction_id)

    manifest_lines = []
    n_written = 0
    n_skipped = 0
    n_missing_partition = 0
    for rid in sorted(locked_rids):
        entry = manual.get(rid)
        if entry is None:
            n_missing_partition += 1
            print(f"[MISS] {rid}: no partition entry")
            continue
        # R.xyz partition is INDEPENDENT of TS.xyz (different atom ordering).
        # Use the R-side manual partition for R.xyz atom selection + ghost basis.
        A = entry.get("frag_A_indices_R")
        B = entry.get("frag_B_indices_R")
        if not A or not B:
            n_missing_partition += 1
            print(f"[MISS] {rid}: empty R partition")
            continue

        try:
            R_atoms = ase.io.read(str(RAW / rid / "R.xyz"))
        except Exception as exc:
            print(f"[ERR ] {rid}: cannot read R.xyz ({exc})")
            n_skipped += 1
            continue
        # Sanity: max index < len(R_atoms)
        if max(A + B) >= len(R_atoms):
            print(f"[ERR ] {rid}: index out of range for R.xyz "
                  f"(max={max(A+B)}, len(R)={len(R_atoms)})")
            n_skipped += 1
            continue

        fam = family(rid)
        fA_c, fB_c = read_fragment_charges(V8 / "orca_inputs" / rid / "eda.inp", fam)

        rid_dir = SP_ROOT / rid
        rid_dir.mkdir(exist_ok=True)

        # Match theory to the actual eda_frag{1,2}.inp used at TS
        theory_A = read_theory_line(V8 / "orca_inputs" / rid / "eda_frag1.inp")
        theory_B = read_theory_line(V8 / "orca_inputs" / rid / "eda_frag2.inp")

        # fragA_R.inp: fragA real, fragB ghost
        write_cp_sp(rid, A, B, fA_c, R_atoms, theory_A, rid_dir / "fragA_R.inp")
        # fragB_R.inp: fragB real, fragA ghost
        write_cp_sp(rid, B, A, fB_c, R_atoms, theory_B, rid_dir / "fragB_R.inp")

        manifest_lines.append(f"SP {rid} fragA")
        manifest_lines.append(f"SP {rid} fragB")
        n_written += 1

    MANIFEST.write_text("\n".join(manifest_lines) + "\n")
    print()
    print(f"wrote CP-SP inputs for {n_written} rxns")
    print(f"  missing partition: {n_missing_partition}")
    print(f"  skipped (error):   {n_skipped}")
    print(f"manifest: {MANIFEST}  ({len(manifest_lines)} entries)")


if __name__ == "__main__":
    main()
