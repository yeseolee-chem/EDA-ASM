#!/usr/bin/env python3
"""SPEC17rev2 Step 0 — environment prep.

GATE-0: pretrained ckpt 42,699,153 bytes, 5,269 profile dirs, CSV 5,269 rows,
        networkx importable.

Idempotent — safe to re-run.
- Symlinks Coley extracted profiles from /gpfs/tmp_cpu2 (already extracted)
  into analysis/otfm_train/coley_profiles/ instead of re-extracting (saves
  ~4 GB in $HOME quota).
- Clones react-ot into external/react-ot/ if missing.
- Verifies the pretrained checkpoint size.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
BASE = REPO / "analysis/otfm_train"
COLEY_TMP = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/dipolar_cycloaddition")
EXTRACTED = COLEY_TMP / "extracted/full_dataset_profiles"
CSV_PATH = COLEY_TMP / "full_dataset.csv"

EXTERNAL = REPO / "external"
ROT = EXTERNAL / "react-ot"
CKPT = ROT / "reactot-pretrained.ckpt"
CKPT_SIZE_EXPECT = 42_699_153


def main() -> int:
    # ---- directories --------------------------------------------------------
    for d in ("artifacts", "data", "data/reactant_complex",
              "ckpt", "generated", "generated/crossfit",
              "logs", "figures", "external_stage"):
        (BASE / d).mkdir(parents=True, exist_ok=True)
    EXTERNAL.mkdir(parents=True, exist_ok=True)

    # ---- Coley profile symlink ---------------------------------------------
    prof_link = BASE / "coley_profiles"
    prof_link.mkdir(exist_ok=True)
    tgt = prof_link / "full_dataset_profiles"
    if tgt.is_symlink() or tgt.exists():
        if tgt.is_symlink() and Path(tgt).resolve() != EXTRACTED.resolve():
            tgt.unlink()
    if not tgt.exists():
        if not EXTRACTED.exists():
            print(f"[FATAL] extracted profiles not found: {EXTRACTED}", file=sys.stderr)
            print(f"        expected pre-extracted archive at {COLEY_TMP}/full_dataset_profiles.tar.gz",
                  file=sys.stderr)
            return 1
        tgt.symlink_to(EXTRACTED)
        print(f"symlinked {tgt} -> {EXTRACTED}")

    prof_dirs = sorted(d for d in tgt.iterdir()
                       if d.is_dir() and d.name.isdigit())
    print(f"Coley profile dirs: {len(prof_dirs)}")
    if len(prof_dirs) != 5269:
        print(f"[FATAL] expected 5269 profile dirs, got {len(prof_dirs)}", file=sys.stderr)
        return 1

    # ---- CSV ----------------------------------------------------------------
    if not CSV_PATH.exists():
        print(f"[FATAL] CSV not found: {CSV_PATH}", file=sys.stderr)
        return 1
    csv = pd.read_csv(CSV_PATH)
    print(f"Coley CSV rows: {len(csv)}")
    if len(csv) != 5269:
        print(f"[FATAL] expected 5269 CSV rows, got {len(csv)}", file=sys.stderr)
        return 1

    # ---- react-ot clone -----------------------------------------------------
    if not ROT.exists():
        print(f"cloning react-ot into {ROT} ...")
        r = subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/deepprinciple/react-ot.git", str(ROT)],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            print("[FATAL] git clone failed:", file=sys.stderr)
            print(r.stderr, file=sys.stderr)
            return 1
    else:
        print(f"react-ot already present at {ROT}")

    # ---- pretrained ckpt ----------------------------------------------------
    if not CKPT.exists():
        print(f"[FATAL] pretrained ckpt missing: {CKPT}", file=sys.stderr)
        print(f"        download from zenodo:14836384 or 13119868 into {ROT}",
              file=sys.stderr)
        return 1
    size = CKPT.stat().st_size
    print(f"pretrained ckpt: {size:,} bytes (expect {CKPT_SIZE_EXPECT:,})")
    if size != CKPT_SIZE_EXPECT:
        print(f"[FATAL] ckpt size mismatch", file=sys.stderr)
        return 1

    # ---- python env sanity --------------------------------------------------
    import networkx
    import numpy
    import torch
    print(f"networkx {networkx.__version__}  numpy {numpy.__version__}  torch {torch.__version__}")

    # ---- record gate status -------------------------------------------------
    (BASE / "artifacts" / "GATE0_STATUS.txt").write_text(
        "PASS\n"
        f"profile_dirs={len(prof_dirs)}\n"
        f"csv_rows={len(csv)}\n"
        f"ckpt_bytes={size}\n"
        f"networkx={networkx.__version__}\n"
        f"numpy={numpy.__version__}\n"
        f"torch={torch.__version__}\n"
    )
    print("=== GATE-0 PASS ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
