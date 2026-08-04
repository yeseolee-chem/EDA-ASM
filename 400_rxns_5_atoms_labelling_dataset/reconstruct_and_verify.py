#!/usr/bin/env python3
"""
Reconstruct + verify 400-reaction 5-atom labeling dataset.

Self-contained: needs only files inside this folder. No repo imports.

Loads
    5_atom_picks.json      : per-rxn 5-atom picks (user-labeled via viz app)
    fragment_assignment.json: per-rxn frag A/B atom lists on TS complex + type
    manifest.csv           : rn ↔ reaction_id ↔ sub_source table
    structures/rxn_XXXX/   : xyz files (ts, r_A, r_B, d_A, d_B)

Emits
    (dict) full_labels[reaction_id] = {
        rn, reaction_id, sub_source, is_A_dipole,
        di_ts, dp_ts,                              # 5-atom picks on TS complex
        di_local_in_r, dp_local_in_r,              # 5-atom picks on isolated R fragment
        ts_idx_A, ts_idx_B, type_A, type_B,        # fragment assignment
        n_atoms_ts, n_atoms_A, n_atoms_B,
        charge_total, charge_A, charge_B, mult_A, mult_B,
        xyz_paths                                  # dict of local xyz paths
    }

Verifies
    * all 5-atom picks landed on atoms inside the correct fragment
    * di_ts ∪ dp_ts are all valid TS complex indices
    * XYZ files exist and atom counts match manifest

Run
    python reconstruct_and_verify.py                # print stats + verify
    python reconstruct_and_verify.py --dump out.pkl # also dump full dict as pkl
"""
from __future__ import annotations
import argparse
import json
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _xyz_natoms(path: Path) -> int:
    with path.open() as fh:
        return int(fh.readline().strip())


def build() -> dict:
    picks = json.loads((HERE / "5_atom_picks.json").read_text())
    frag = json.loads((HERE / "fragment_assignment.json").read_text())

    out: dict = {}
    for rid, pick in picks.items():
        if rid not in frag:
            raise KeyError(f"{rid} in picks but not in fragment_assignment")
        f = frag[rid]

        rxn_num = f["reaction_number"]
        rxn_dir = HERE / "structures" / f"rxn_{rxn_num:04d}"
        xyz_paths = {
            "ts": rxn_dir / "ts.xyz",
            "r_A": rxn_dir / "r_A.xyz",
            "r_B": rxn_dir / "r_B.xyz",
            "d_A": rxn_dir / "d_A.xyz",
            "d_B": rxn_dir / "d_B.xyz",
        }
        entry = {
            "rn": rxn_num,
            "reaction_id": rid,
            "sub_source": f["sub_source"],
            "is_A_dipole": bool(pick["is_A_dipole"]),
            "di_ts": list(map(int, pick["di_ts"])),
            "dp_ts": list(map(int, pick["dp_ts"])),
            "di_local_in_r": list(map(int, pick["di_local_in_r"])),
            "dp_local_in_r": list(map(int, pick["dp_local_in_r"])),
            "ts_idx_A": f["ts_idx_A"],
            "ts_idx_B": f["ts_idx_B"],
            "type_A": f["type_A"],
            "type_B": f["type_B"],
            "n_atoms_ts": f["n_atoms_ts"],
            "n_atoms_A": f["n_atoms_A"],
            "n_atoms_B": f["n_atoms_B"],
            "charge_total": f["charge_total"],
            "charge_A": f["charge_A"],
            "charge_B": f["charge_B"],
            "mult_A": f["mult_A"],
            "mult_B": f["mult_B"],
            "xyz_paths": {k: str(v.relative_to(HERE)) for k, v in xyz_paths.items()},
        }
        out[rid] = entry
    return out


def verify(labels: dict) -> tuple[int, int, list[str]]:
    ok = 0
    problems: list[str] = []
    for rid, e in labels.items():
        # 5-atom picks are valid TS indices
        n_ts = e["n_atoms_ts"]
        for slot, idxs in (("di_ts", e["di_ts"]), ("dp_ts", e["dp_ts"])):
            for i in idxs:
                if not (0 <= i < n_ts):
                    problems.append(f"{rid}: {slot}[{i}] out of range [0, {n_ts})")
        # 5-atom picks fall on correct fragment
        di_set = set(e["di_ts"])
        dp_set = set(e["dp_ts"])
        A_set = set(e["ts_idx_A"])
        B_set = set(e["ts_idx_B"])
        if e["is_A_dipole"]:
            dipole_set, dipolarophile_set = A_set, B_set
        else:
            dipole_set, dipolarophile_set = B_set, A_set
        stray_di = di_set - dipole_set
        stray_dp = dp_set - dipolarophile_set
        if stray_di:
            problems.append(f"{rid}: di_ts atoms {stray_di} not in dipole fragment")
        if stray_dp:
            problems.append(f"{rid}: dp_ts atoms {stray_dp} not in dipolarophile fragment")
        # required xyz files: ts, d_A, d_B  (r_A/r_B optional — see README)
        for k in ("ts", "d_A", "d_B"):
            p = HERE / e["xyz_paths"][k]
            if not p.exists():
                problems.append(f"{rid}: missing {k}.xyz (required)")
                continue
            n = _xyz_natoms(p)
            expected = e["n_atoms_ts"] if k == "ts" else (e["n_atoms_A"] if k.endswith("_A") else e["n_atoms_B"])
            if n != expected:
                problems.append(f"{rid}: {k}.xyz has {n} atoms, manifest says {expected}")
        # optional r_A/r_B — verify only if present
        for k in ("r_A", "r_B"):
            p = HERE / e["xyz_paths"][k]
            if not p.exists():
                continue  # optional
            n = _xyz_natoms(p)
            expected = e["n_atoms_A"] if k.endswith("_A") else e["n_atoms_B"]
            if n != expected:
                problems.append(f"{rid}: {k}.xyz has {n} atoms, manifest says {expected}")
        if not any(rid in p for p in problems[-8:]):
            ok += 1
    return ok, len(labels), problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path, help="also dump full dict as pickle to this path")
    args = ap.parse_args()

    labels = build()
    print(f"loaded {len(labels)} reactions")

    ok, total, problems = verify(labels)
    print(f"verify: {ok}/{total} rxns pass all consistency checks")
    if problems:
        print(f"{len(problems)} problem(s):")
        for p in problems[:20]:
            print(f"  - {p}")
        if len(problems) > 20:
            print(f"  ... ({len(problems) - 20} more)")
        return 1

    # summary stats
    n_A_dipole = sum(1 for e in labels.values() if e["is_A_dipole"])
    print(f"summary:")
    print(f"  is_A_dipole=True : {n_A_dipole}")
    print(f"  is_A_dipole=False: {total - n_A_dipole}")
    from collections import Counter
    src_counts = Counter(e["sub_source"] for e in labels.values())
    for src, c in src_counts.most_common():
        print(f"  sub_source={src}: {c}")

    if args.dump:
        args.dump.parent.mkdir(parents=True, exist_ok=True)
        with args.dump.open("wb") as fh:
            pickle.dump(labels, fh)
        print(f"dumped to {args.dump}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
