#!/usr/bin/env python3
"""Mirror per-reaction raw inputs into git-visible tree.

For each rxn in cohort_v1/labels_v1.pkl (N=3504):
  cohort_v1/reactions/rxn_XXXX/
    strain.json   (verbatim copy of strain_am1/work/rxn_XXXX/strain.json)
    eda.inp       (verbatim copy of tt_eda_kisti_results/data/rxn_XXXX/eda.inp)
    e_ab.txt      (single FSPE line extracted from eda.out)

Idempotent: skip rxn dirs where all 3 target files already exist.
"""
import re
import shutil
from pathlib import Path

import pandas as pd

EXP = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/bath1480_probe/experiments")
STRAIN_SRC = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/strain_am1/work")
KISTI_SRC = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/tt_eda_kisti_results/data")
OUT_ROOT = EXP / "cohort_v1/reactions"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

FSPE_RE = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)")


def extract_fspe(eda_out: Path) -> str | None:
    last = None
    with eda_out.open() as f:
        for line in f:
            m = FSPE_RE.search(line)
            if m:
                last = m.group(1)
    return last


def main():
    labels = pd.read_pickle(EXP / "cohort_v1/labels_v1.pkl")
    rxns = sorted(labels["reaction_number"].tolist())
    print(f"Cohort: {len(rxns)} reactions")

    done = skipped = missing = fspe_fail = 0
    for i, rxn in enumerate(rxns):
        tag = f"rxn_{rxn:04d}"
        out_dir = OUT_ROOT / tag
        out_strain = out_dir / "strain.json"
        out_inp = out_dir / "eda.inp"
        out_eab = out_dir / "e_ab.txt"
        if out_strain.exists() and out_inp.exists() and out_eab.exists():
            skipped += 1
            continue

        src_strain = STRAIN_SRC / tag / "strain.json"
        src_inp = KISTI_SRC / tag / "eda.inp"
        src_out = KISTI_SRC / tag / "eda.out"
        if not src_strain.exists() or not src_inp.exists() or not src_out.exists():
            missing += 1
            print(f"MISSING sources for {tag}", flush=True)
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_strain, out_strain)
        shutil.copy2(src_inp, out_inp)
        eab = extract_fspe(src_out)
        if eab is None:
            fspe_fail += 1
            print(f"FSPE not found in {src_out}", flush=True)
            continue
        out_eab.write_text(f"{eab}\n")
        done += 1

        if (i + 1) % 500 == 0:
            print(f"  progress {i+1}/{len(rxns)}: done={done} skipped={skipped} missing={missing} fspe_fail={fspe_fail}", flush=True)

    print(f"\n=== DONE ===  new={done}  skipped={skipped}  missing={missing}  fspe_fail={fspe_fail}")


if __name__ == "__main__":
    main()
