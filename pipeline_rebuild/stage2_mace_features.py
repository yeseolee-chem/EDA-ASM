"""Stage 2 — MACE-OFF23_medium feature extraction for the 789-reaction cohort.

For every reaction_id in labels/adf/adf_labels_v6_multifamily.parquet, load the
R / TS / P XYZ triple from the family's on-disk layout (populated by stages
1 / 1b / 1d) and run the frozen MACE-OFF23 backbone. Save one .pt file per
reaction under features/{reaction_id}.pt with keys {R, TS, P} → per-atom
invariant features + atomic numbers + coordinates.

Output layout:
  /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/mace_off23_medium/{rid}.pt
  /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/mace_off23_medium/_progress.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import ase.io
import numpy as np
import pandas as pd
import torch

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
sys.path.insert(0, str(REPO / "src"))

from eda_asm.asr_v1.backbone_maceoff import MACEOFFFeatureExtractor

RAW = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw")
DIP_ROOT = RAW / "dipolar_cycloaddition" / "extracted" / "full_dataset_profiles"
QMR_ROOT = RAW / "QMrxn20"
RGD1_ROOT = RAW / "rgd1" / "extracted_xyz"

OUT_DIR = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/mace_off23_medium")
OUT_DIR.mkdir(parents=True, exist_ok=True)
PROGRESS = OUT_DIR / "_progress.jsonl"


def _single_match(rxn_dir: Path, pattern: str) -> Optional[Path]:
    matches = list(rxn_dir.glob(pattern))
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    primary = [p for p in matches if "_alt" not in p.stem]
    return primary[0] if len(primary) == 1 else matches[0]


def load_triple(rid: str, family: str) -> tuple[ase.Atoms, ase.Atoms, ase.Atoms]:
    if family == "dipolar":
        idx = int(rid.split("_")[-1])
        d = DIP_ROOT / str(idx)
        r0 = ase.io.read(str(_single_match(d, "r0_*.xyz")))
        r1_path = _single_match(d, "r1_*.xyz")
        R = r0 + ase.io.read(str(r1_path)) if r1_path else r0
        ts_path = _single_match(d, "TS_imag_mode_*.xyz") or _single_match(d, "TS_*.xyz")
        TS = ase.io.read(str(ts_path))
        P = ase.io.read(str(_single_match(d, "p0_*.xyz")))
        return R, TS, P

    if family in ("qmrxn20_e2", "qmrxn20_sn2"):
        subfam = "e2" if "e2" in family else "sn2"
        label = "_".join(rid.split("_")[2:])                     # A_B_A_A_C_B
        TS = ase.io.read(str(QMR_ROOT / "transition-states" / subfam / f"{label}.xyz"))
        # reactant complex — pick conformer 00
        rc = QMR_ROOT / "reactant-complex-constrained-conformers" / subfam / label
        r_path = rc / "00.xyz"
        if not r_path.exists():
            r_path = next(iter(rc.glob("*.xyz")), None)
        R = ase.io.read(str(r_path))
        # product — derive product label
        parts = label.split("_")
        if subfam == "e2":
            plabel = "_".join([*parts[:4], "0", "0"])
        else:
            plabel = "_".join([*parts[:4], "0", parts[5]])
        pd_dir = QMR_ROOT / "product-conformers" / subfam / plabel
        p_path = pd_dir / "00.xyz"
        if not p_path.exists():
            p_path = next(iter(pd_dir.glob("*.xyz")), None)
        P = ase.io.read(str(p_path))
        return R, TS, P

    if family == "rgd1":
        d = RGD1_ROOT / rid
        R = ase.io.read(str(d / "R.xyz"))
        TS = ase.io.read(str(d / "TS.xyz"))
        P = ase.io.read(str(d / "P.xyz"))
        return R, TS, P

    raise ValueError(f"unknown family {family!r}")


def run_one(fe: MACEOFFFeatureExtractor, rid: str, family: str) -> dict:
    R, TS, P = load_triple(rid, family)
    feat_R = fe.extract(R)
    feat_TS = fe.extract(TS)
    feat_P = fe.extract(P)
    return {
        "reaction_id":       rid,
        "family":            family,
        "R":  {"z": R.get_atomic_numbers().tolist(),
               "pos": R.get_positions().astype(np.float32),
               "feat": feat_R.cpu().numpy().astype(np.float32)},
        "TS": {"z": TS.get_atomic_numbers().tolist(),
               "pos": TS.get_positions().astype(np.float32),
               "feat": feat_TS.cpu().numpy().astype(np.float32)},
        "P":  {"z": P.get_atomic_numbers().tolist(),
               "pos": P.get_positions().astype(np.float32),
               "feat": feat_P.cpu().numpy().astype(np.float32)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-size", default="medium",
                    help="MACE-OFF23 size: small|medium|large (default medium)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop",  type=int, default=-1)
    args = ap.parse_args()

    df = pd.read_parquet(REPO / "labels/adf/adf_labels_v6_multifamily.parquet")
    print(f"[{time.strftime('%H:%M:%S')}] cohort = {len(df)} reactions")
    print(df.family.value_counts())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{time.strftime('%H:%M:%S')}] loading MACE-OFF23 {args.model_size} on {device}")
    fe = MACEOFFFeatureExtractor(model_size=args.model_size, device=device)
    print(f"[{time.strftime('%H:%M:%S')}] feature_dim = {fe.feature_dim}")

    if args.stop == -1: args.stop = len(df)
    subset = df.iloc[args.start:args.stop]
    print(f"[{time.strftime('%H:%M:%S')}] processing rows {args.start}..{args.stop}")

    done = fail = 0
    t0 = time.time()
    with open(PROGRESS, "a") as pf:
        for i, row in subset.iterrows():
            rid, fam = row.reaction_id, row.family
            out = OUT_DIR / f"{rid}.pt"
            if out.exists():
                done += 1
                continue
            try:
                d = run_one(fe, rid, fam)
                torch.save(d, out)
                done += 1
            except Exception as e:
                fail += 1
                pf.write(json.dumps({"rid": rid, "family": fam,
                                     "err": f"{type(e).__name__}: {e}"}) + "\n")
                pf.flush()
            if (done + fail) % 25 == 0:
                elapsed = time.time() - t0
                rate = (done + fail) / max(elapsed, 1e-6)
                print(f"[{time.strftime('%H:%M:%S')}] {done+fail}/{len(subset)} "
                      f"done={done} fail={fail} rate={rate:.2f}/s")

    elapsed = time.time() - t0
    print(f"[{time.strftime('%H:%M:%S')}] DONE  elapsed={elapsed:.1f}s "
          f"done={done} fail={fail} of {len(subset)}")


if __name__ == "__main__":
    main()
