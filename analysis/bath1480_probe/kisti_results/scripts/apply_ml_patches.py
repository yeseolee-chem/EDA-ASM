#!/usr/bin/env python3
"""
apply_ml_patches.py — minimal in-place patches to the-grayson-group ML code
for the 7-channel run. Idempotent; writes .bak backups on first application.

Patch 1 (f_select.py): VarianceThreshold(0.05) on UNSTANDARDIZED features
  deletes 14/41 descriptors (12 of 15 Mulliken charges + both dp C=C
  distances) purely because their units are small. -> threshold=1e-10
  (removes true constants only).

Patch 2 (tt_ml/hyp_tuning.py): SVR grid.
  - epsilon ceiling 1.0 swallows small-scale channels (cpcm ~ -0.5 kcal/mol
    -> constant model). Extend log-scale to 20.
  - C ceiling 50 was hit by 4/6 paper targets. Add 100, 300.
  - coef0/degree are IGNORED by sklearn SVR(kernel='rbf'); shrinking them to
    single values changes nothing but cuts the grid 6x.
  Net grid per target: 288 -> 120 combinations, wider where it matters.

Usage: python apply_ml_patches.py /path/to/gh_repo
"""
import shutil, sys
from pathlib import Path

PATCHES = {
    "f_select.py": [
        ("selector = VarianceThreshold(threshold=0.05)",
         "selector = VarianceThreshold(threshold=1e-10)  # PATCH-7ch: scale-free (constants only); 0.05 on raw units killed 14/41 descriptors"),
    ],
    "tt_ml/hyp_tuning.py": [
        ("'epsilon':[0.001, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1],",
         "'epsilon':[0.001, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20],  # PATCH-7ch: channel scales span ~50x"),
        ("'C':[1, 30, 50],",
         "'C':[1, 30, 50, 100, 300],  # PATCH-7ch: 4/6 paper targets sat at old ceiling"),
        ("'coef0': [0, 1], ",
         "'coef0': [0],  # PATCH-7ch: ignored by rbf SVR"),
        ("'degree': [1, 2, 3],",
         "'degree': [3],  # PATCH-7ch: ignored by rbf SVR"),
    ],
}

def main(repo: str) -> None:
    root = Path(repo)
    for rel, subs in PATCHES.items():
        p = root / rel
        if not p.exists():
            sys.exit(f"[fatal] {p} not found")
        text = p.read_text()
        changed = 0
        for old, new in subs:
            if new in text:
                print(f"[skip] {rel}: already patched -> {old[:40]}...")
            elif old in text:
                text = text.replace(old, new); changed += 1
            else:
                sys.exit(f"[fatal] {rel}: anchor not found:\n  {old}\n"
                         "  (upstream code differs — patch by hand)")
        if changed:
            bak = p.with_suffix(p.suffix + ".bak")
            if not bak.exists():
                shutil.copy2(p, bak)
            p.write_text(text)
            print(f"[ok] {rel}: {changed} patch(es) applied (backup: {bak.name})")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
