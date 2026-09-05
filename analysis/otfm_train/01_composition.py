#!/usr/bin/env python3
"""SPEC17rev2 Step 1 — element inventory.

GATE-1: supported == 100% (H C N O F Cl Br). Any other element must be added
to ATOM_MAPPING / Z / node_nfs before Step 4.

Reference numbers (spec):
  heavy atom median 24, halogen-containing 31.2%, CHNO-only 68.8%.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

BASE = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/otfm_train")
PROF = BASE / "coley_profiles/full_dataset_profiles"
SUPPORTED = {"H", "C", "N", "O", "F", "Cl", "Br"}


def read_syms(path: Path):
    lines = path.read_text().split("\n")
    n = int(lines[0])
    return [l.split()[0] for l in lines[2:2 + n]]


def main() -> int:
    if not PROF.exists():
        print(f"[FATAL] profiles missing: {PROF}. Run Step 0 first.", file=sys.stderr)
        return 1

    rows = []
    for d in sorted(PROF.iterdir()):
        if not (d.is_dir() and d.name.isdigit()):
            continue
        rid = int(d.name)
        ts = [f for f in d.iterdir()
              if f.name.startswith("TS_") and f.name != "TS_imag_mode.xyz"]
        if not ts:
            rows.append(dict(rxn_id=rid, status="no_ts"))
            continue
        syms = read_syms(ts[0])
        els = set(syms)
        rows.append(dict(
            rxn_id=rid, status="ok",
            supported=(els <= SUPPORTED), n_atoms=len(syms),
            elements="".join(sorted(els)),
            has_halogen=bool(els & {"Cl", "Br"}),
        ))

    C = pd.DataFrame(rows)
    C.to_csv(BASE / "artifacts" / "composition.csv", index=False)
    ok = C[C.status == "ok"]
    n_ok = len(ok)
    sup_rate = ok.supported.mean() if n_ok else 0.0
    halo_rate = ok.has_halogen.mean() if n_ok else 0.0
    print(f"TS present: {n_ok} / {len(C)}")
    print(f"supported-elements-only: {sup_rate:.2%}   halogen-containing: {halo_rate:.2%}")

    un = ok[~ok.supported] if n_ok else ok
    if len(un):
        print(f"\n[WARN] unsupported-element rxns: {len(un)}")
        print(un.elements.value_counts().head(10).to_string())

    passed = (sup_rate == 1.0)
    (BASE / "artifacts" / "GATE1_STATUS.txt").write_text(
        ("PASS" if passed else "FAIL") + "\n"
        f"ok_count={n_ok}\n"
        f"supported_rate={sup_rate:.6f}\n"
        f"halogen_rate={halo_rate:.6f}\n"
        f"unsupported_count={len(un)}\n"
    )
    if not passed:
        print("[GATE-1 FAIL] add missing elements to ATOM_MAPPING/Z/node_nfs.",
              file=sys.stderr)
        return 1
    print("=== GATE-1 PASS ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
