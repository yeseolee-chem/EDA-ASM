#!/usr/bin/env python3
"""Build 5,260 SMD-keyword eda.inp files under /gpfs/tmp_cpu2/yeseo1ee/eda_smd_relabel/inputs/.

Rewrites the old eda.inp per rxn:
    !B3LYP D3BJ def2-TZVP CPCM(water) NoSym EDA TightSCF  (+ separate %cpcm smd block)
into:
    !B3LYP D3BJ def2-TZVP SMD(water) NoSym EDA TightSCF   (no %cpcm block)
and rewrites the FRAG1/FRAG2 method strings the same way. This propagates SMD
to ORCA's auto-generated fragment SCFs (eda_frag*.out), which were the only
files missing SMD-CDS in the original run.

Idempotent: skips reactions whose input file already exists.
"""
import re
import sys
from pathlib import Path

SRC_ROOT = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/label_true/work/inputs")
DST_ROOT = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_smd_relabel/inputs")
CPCM_BLOCK_RE = re.compile(r"^%cpcm\s*\n(?:.*?\n)*?^end\s*\n", re.MULTILINE)


def rewrite(text: str) -> str:
    # 1. header + FRAG strings: CPCM(water) → SMD(water)
    out = text.replace("CPCM(water)", "SMD(water)")
    # 2. drop the standalone %cpcm block that used to enable SMD (redundant now)
    out = CPCM_BLOCK_RE.sub("", out)
    return out


def main():
    src_dirs = sorted(SRC_ROOT.glob("rxn_*"))
    print(f"src rxn dirs: {len(src_dirs)}")
    made = skipped = missing = 0
    for src in src_dirs:
        rid = src.name
        src_inp = src / "eda.inp"
        if not src_inp.is_file():
            missing += 1
            continue
        dst_dir = DST_ROOT / rid
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_inp = dst_dir / "eda.inp"
        if dst_inp.is_file():
            skipped += 1
            continue
        dst_inp.write_text(rewrite(src_inp.read_text()))
        made += 1
    print(f"made={made}  skipped={skipped}  missing_src={missing}")
    if missing:
        sys.exit(f"GATE: {missing} rxn dirs missing eda.inp")


if __name__ == "__main__":
    main()
