#!/usr/bin/env python3
"""SPEC18 Step 0 — environment + prior-stage prerequisite check.

Verifies:
  - xtb binary discoverable (must run inside `reactot` conda env)
  - prior spec17rev2 outputs: coley_all.pkl, folds.csv, reactant_build.csv,
    5 fold ckpt dirs, final ckpt dir, 5,261+ reactant_complex npz files
  - Coley profile dirs (integer names) present for RID resolution in Step 3
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
BASE = REPO / "analysis" / "otfm_guess"
PREV = REPO / "analysis" / "otfm_train"

for d in ("artifacts", "xtb_runs", "data", "ckpt", "generated", "logs", "figures"):
    (BASE / d).mkdir(parents=True, exist_ok=True)

xtb = shutil.which("xtb")
assert xtb, "xtb 실행 파일이 PATH에 없음. `conda activate reactot` 필요"
v = subprocess.run([xtb, "--version"], capture_output=True, text=True)
print(f"xtb: {xtb}")
print(v.stdout.strip().split("\n")[0] if v.stdout else v.stderr[:200])

for rel in (
    "data/coley_all.pkl",
    "artifacts/folds.csv",
    "artifacts/reactant_build.csv",
    "artifacts/generation_quality.csv",
):
    p = PREV / rel
    assert p.exists(), f"이전 단계 산출물 없음: {p}"
    print(f"[prev] {rel}  {p.stat().st_size} B")

npz_dir = PREV / "data" / "reactant_complex"
n_npz = len(list(npz_dir.glob("rxn_*.npz")))
print(f"reactant_complex npz: {n_npz}")
assert n_npz >= 5261, f"npz {n_npz} < 5261"

for k in range(5):
    d = PREV / "ckpt" / f"fold{k}"
    ckpts = list(d.rglob("sb-*.ckpt")) if d.exists() else []
    assert ckpts, f"ckpt/fold{k} 아래 sb-*.ckpt 없음"
    print(f"[ckpt] fold{k}: {len(ckpts)} files")

fin = PREV / "ckpt" / "final"
fin_ckpts = list(fin.rglob("sb-*.ckpt")) if fin.exists() else []
assert fin_ckpts, "ckpt/final 아래 sb-*.ckpt 없음"
print(f"[ckpt] final: {len(fin_ckpts)} files")

prof = PREV / "coley_profiles" / "full_dataset_profiles"
if prof.exists():
    n_prof = sum(1 for p in prof.iterdir() if p.is_dir() and p.name.isdigit())
    print(f"coley profile dirs: {n_prof}")
    assert n_prof >= 5269, f"coley profile dirs {n_prof} < 5269"

(BASE / "artifacts" / "GATE0_STATUS.txt").write_text(
    "PASS\n"
    f"xtb_path={xtb}\n"
    f"reactant_complex_npz={n_npz}\n"
)
print("=== GATE-0 PASS ===")
