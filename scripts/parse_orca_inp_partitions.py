"""Parse ORCA .inp files to extract fragment IDs per atom (elem(N) tags)
and emit a review-app-compatible partitions JSON.

Reads:   outputs/orca_eda/inputs/<rid>/eda.inp
Writes:  outputs/frag_review/orca_inp_partitions.json
"""
from __future__ import annotations
import json, re
from pathlib import Path

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
INPUT_ROOT = REPO / "outputs/orca_eda/inputs"
OUT = REPO / "outputs/frag_review/orca_inp_partitions.json"


ATOM_RE = re.compile(r"^\s*([A-Z][a-z]?)\((\d+)\)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*$")


def parse_inp(path: Path):
    """Return (frag_A_indices, frag_B_indices) parsed from the xyz block."""
    frag_A, frag_B = [], []
    in_xyz = False
    idx = 0
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("* xyz"):
            in_xyz = True
            continue
        if in_xyz and stripped == "*":
            break
        if not in_xyz:
            continue
        m = ATOM_RE.match(line)
        if not m:
            continue
        _elem, fid, _x, _y, _z = m.groups()
        if fid == "1":
            frag_A.append(idx)
        elif fid == "2":
            frag_B.append(idx)
        idx += 1
    return frag_A, frag_B


def main():
    out = {}
    n_ok = n_err = 0
    for d in sorted(INPUT_ROOT.iterdir()):
        if not d.is_dir(): continue
        inp = d / "eda.inp"
        if not inp.exists(): continue
        try:
            A, B = parse_inp(inp)
            if not A or not B:
                raise RuntimeError(f"empty fragment (A={len(A)}, B={len(B)})")
            out[d.name] = {
                "frag_A_indices": A,
                "frag_B_indices": B,
                "reviewed": False,
                "note": "from ORCA .inp",
            }
            n_ok += 1
        except Exception as exc:
            n_err += 1
            print(f"[ERR] {d.name}: {exc}", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"parsed {n_ok} inputs, {n_err} errors → {OUT}")


if __name__ == "__main__":
    main()
