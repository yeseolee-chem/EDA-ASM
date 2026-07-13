"""Recover the TS-based partition from eda_frag1.inp historical files
and store R-review results in separate fields.

Fields added to manual_partitions.json (write in place):
  frag_A_indices   := TS-based partition (used for EDA calc)
  frag_B_indices   := TS-based partition
  frag_A_indices_R := R-based partition  (used for strain SP)
  frag_B_indices_R := R-based partition
  TS_recoverable   := True/False  (whether we had historical eda_frag1.inp)
  needs_TS_review  := True if partition ambiguous
  needs_R_review   := (already exists via R_reviewed)

Also writes a manifest of rxns that require user review:
  outputs/v8_review/needs_review.txt
"""
from __future__ import annotations
import json, re
from pathlib import Path

V8 = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/outputs/v8_review")
ORCA = V8 / "orca_inputs"
MP = V8 / "manual_partitions.json"
NEEDS = V8 / "needs_review.txt"


def parse_frag1_inp(path: Path):
    A, B = [], []
    i = 0
    xyz_started = False
    for line in path.read_text().splitlines():
        s = line.strip()
        if s.startswith("*xyz"):
            xyz_started = True
            continue
        if not xyz_started:
            continue
        if s.startswith("*") or not s:
            break
        m = re.match(r'^\s*[A-Za-z]+\s*(:?)\(([12])\)', line)
        if not m:
            continue
        is_ghost = m.group(1) == ":"
        frag = int(m.group(2))
        if not is_ghost and frag == 1:
            A.append(i)
        elif is_ghost and frag == 2:
            B.append(i)
        i += 1
    return sorted(A), sorted(B)


def main():
    manual = json.loads(MP.read_text())
    n_recovered_same = 0
    n_recovered_diff = 0
    n_no_history = 0
    needs = []

    for rid, entry in list(manual.items()):
        cur_A = sorted(entry.get("frag_A_indices", []))
        cur_B = sorted(entry.get("frag_B_indices", []))

        # Always preserve current (R-review) values in the _R keys
        entry["frag_A_indices_R"] = cur_A
        entry["frag_B_indices_R"] = cur_B

        hist_path = ORCA / rid / "eda_frag1.inp"
        if hist_path.exists():
            try:
                A_hist, B_hist = parse_frag1_inp(hist_path)
            except Exception:
                A_hist, B_hist = None, None
        else:
            A_hist, B_hist = None, None

        if A_hist is not None and A_hist and B_hist:
            # Restore TS partition from historical file
            entry["frag_A_indices"] = A_hist
            entry["frag_B_indices"] = B_hist
            entry["TS_recoverable"] = True
            if A_hist == cur_A and B_hist == cur_B:
                entry["needs_TS_review"] = False
                n_recovered_same += 1
            else:
                # Divergent — user needs to confirm which is right
                entry["needs_TS_review"] = True
                needs.append(rid)
                n_recovered_diff += 1
        else:
            # No historical file → TS partition unknown (was overwritten)
            # Keep current (R-review) as best guess, mark needs review
            entry["frag_A_indices"] = cur_A
            entry["frag_B_indices"] = cur_B
            entry["TS_recoverable"] = False
            entry["needs_TS_review"] = True
            needs.append(rid)
            n_no_history += 1

    tmp = MP.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manual, indent=1))
    tmp.replace(MP)

    NEEDS.write_text("\n".join(sorted(needs)) + "\n")
    print(f"Recovered same (TS=R, no action needed):   {n_recovered_same}")
    print(f"Recovered divergent (needs user decision): {n_recovered_diff}")
    print(f"No historical file (needs TS review):      {n_no_history}")
    print(f"TOTAL needs review: {len(needs)}  -> {NEEDS}")


if __name__ == "__main__":
    main()
