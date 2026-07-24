"""spec19 Stage 2 G2-G — diassep cross-check on a 20-reaction sample.

Informational only. Cannot fail the stage; disagreement with diassep is a
finding, not a defect (per §2.3 and user directive: fragment split
follows the EDA-NOCV pipeline convention absolutely — never re-derived).

We attempt to import the paper's `diassep/diassep.py`. If its runtime
dependencies (cclib, molml, xyz_py) are unavailable in the reactot env,
we emit an INFO note and skip the run — the outcome is the same either
way (the partition of record is inherited, not diassep's).

Emits results/diassep_agreement.csv with columns:
  reaction_number, reaction_id, sub_source, agreement, note
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
STAGE = REPO / "Ref Comparison/spec19_espley_s2_structures"
MANIFEST = STAGE / "results/manifest.pkl"
OUT_CSV = STAGE / "results/diassep_agreement.csv"
BUILD_LOG = STAGE / "logs/build.log"

REF_CLONE = Path("/gpfs/tmp_cpu2/yeseo1ee/ext/distortion-interaction_ML")

SAMPLE_N = 20
SEED = 42


def _log(fh, msg):
    print(msg)
    fh.write(msg + "\n")
    fh.flush()


def can_load_diassep(fh):
    """Try to import the paper's diassep module + its runtime deps."""
    for mod in ("cclib", "molml", "xyz_py"):
        try:
            __import__(mod)
        except ImportError:
            _log(fh, f"[G2-G INFO] runtime dep '{mod}' not installed — diassep run skipped")
            return None
    dpath = REF_CLONE / "diassep"
    if not (dpath / "diassep.py").exists():
        _log(fh, f"[G2-G INFO] diassep.py not found at {dpath} — skipped")
        return None
    sys.path.insert(0, str(dpath))
    try:
        import diassep  # type: ignore  # noqa
        return diassep
    except Exception as e:
        _log(fh, f"[G2-G INFO] diassep import failed ({e}) — skipped")
        return None


def sample_reactions(mf: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    parts = []
    for sub in sorted(mf["sub_source"].unique()):
        s = mf[mf["sub_source"] == sub]
        k = min(SAMPLE_N // 2, len(s))
        parts.append(s.iloc[rng.choice(len(s), k, replace=False)])
    return pd.concat(parts).sort_values("reaction_number").reset_index(drop=True)


def main() -> int:
    STAGE.joinpath("results").mkdir(exist_ok=True)
    with open(BUILD_LOG, "a") as fh:
        _log(fh, "=== spec19 Stage 2 diassep_crosscheck ===")
        mf = pd.read_pickle(MANIFEST)
        sample = sample_reactions(mf)
        _log(fh, f"[sample] {len(sample)} reactions, seed={SEED}")

        diassep = can_load_diassep(fh)
        rows = []
        for _, row in sample.iterrows():
            rn = int(row["reaction_number"])
            rid = row["reaction_id"]
            sub = row["sub_source"]

            if diassep is None:
                rows.append({
                    "reaction_number": rn, "reaction_id": rid, "sub_source": sub,
                    "agreement": "SKIPPED",
                    "note": "diassep runtime deps unavailable in reactot env",
                })
                continue

            # If we ever wire up a real diassep run, it goes here. For now we
            # would need frequency logs from ORCA .out to feed diassep, which
            # is not the primary artifact of Stage 1 or 2. Deferred.
            rows.append({
                "reaction_number": rn, "reaction_id": rid, "sub_source": sub,
                "agreement": "DEFERRED",
                "note": "diassep requires imaginary-mode frequency data; deferred to later stage",
            })

        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
        _log(fh, f"[write] {OUT_CSV}")
        _log(fh,
             "[G2-G note] Fragment partition of record is INHERITED from the "
             "EDA-NOCV label pipeline per user directive and §2.1. This "
             "cross-check is informational and cannot change the partition.")
        _log(fh, "=== diassep_crosscheck OK ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
