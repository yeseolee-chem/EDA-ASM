"""Per-reaction source-dir inventory utilities.

Each of the 789 v6_multifamily reactions has a paired source directory
containing the TS / fragment xyz files and a status.json with fragment
partition + charge/multiplicity. The status.json schema varies between
the dipolar/qmrxn20 batches (`fragment_charge_a`, `fragment_mult_a`,
`fragment_atoms_a`, …) and the rgd1 batches (shorter schema:
`charge_a`, `charge_b`). This module normalises both into a single
record per reaction.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import ase.io
import numpy as np
import pandas as pd


@dataclass
class Reaction:
    rid: str
    family: str
    source_dir: Path
    ts_xyz: Path
    frag_a_xyz: Path
    frag_b_xyz: Path
    frag_a_opt_out: Path | None
    frag_b_opt_out: Path | None
    charge_a: int
    charge_b: int
    mult_a: int
    mult_b: int
    total_charge: int

    @property
    def n_atoms_a(self) -> int:
        return int(ase.io.read(self.frag_a_xyz).get_global_number_of_atoms())

    @property
    def n_atoms_b(self) -> int:
        return int(ase.io.read(self.frag_b_xyz).get_global_number_of_atoms())


def _load_status(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_reaction(orca_row: pd.Series) -> Reaction:
    src = Path(orca_row["source_dir"])
    rid = orca_row["reaction_id"]
    fam = rid.split("_")[0] if rid.startswith(("dipolar", "rgd1")) else "_".join(rid.split("_")[:2])
    if fam not in {"dipolar", "rgd1", "qmrxn20_e2", "qmrxn20_sn2"}:
        # rgd1 ids like rgd1_MR_…
        for cand in ("qmrxn20_e2", "qmrxn20_sn2", "dipolar", "rgd1"):
            if rid.startswith(cand):
                fam = cand
                break
    status = _load_status(src / "status.json")

    # Dual schema normalisation.
    if "fragment_charge_a" in status:
        ca = int(status["fragment_charge_a"])
        cb = int(status["fragment_charge_b"])
        ma = int(status.get("fragment_mult_a", 1))
        mb = int(status.get("fragment_mult_b", 1))
        tot = int(status.get("total_charge", ca + cb))
    else:
        ca = int(status.get("charge_a", 0))
        cb = int(status.get("charge_b", 0))
        ma = int(status.get("mult_a", 1))
        mb = int(status.get("mult_b", 1))
        tot = int(status.get("total_charge", ca + cb))

    # ORCA EDA stored fragment_charge/mult as ints — prefer it if it exists.
    if not pd.isna(orca_row.get("fragment_charge_a")):
        ca = int(orca_row["fragment_charge_a"]); cb = int(orca_row["fragment_charge_b"])
        ma = int(orca_row["fragment_mult_a"]); mb = int(orca_row["fragment_mult_b"])
        tot = int(orca_row["total_charge"])

    return Reaction(
        rid=rid, family=fam, source_dir=src,
        ts_xyz=src / "ts.xyz",
        frag_a_xyz=src / "geometry_fragA.xyz",
        frag_b_xyz=src / "geometry_fragB.xyz",
        frag_a_opt_out=(src / "c4_fragA_opt.out") if (src / "c4_fragA_opt.out").exists() else None,
        frag_b_opt_out=(src / "c5_fragB_opt.out") if (src / "c5_fragB_opt.out").exists() else None,
        charge_a=ca, charge_b=cb, mult_a=ma, mult_b=mb, total_charge=tot,
    )


def load_all_reactions(orca_parquet: Path) -> list[Reaction]:
    df = pd.read_parquet(orca_parquet)
    return [load_reaction(row) for _, row in df.iterrows()]
