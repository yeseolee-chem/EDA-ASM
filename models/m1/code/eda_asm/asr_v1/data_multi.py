"""Multi-family (dipolar + qmrxn20 e2/sn2) geometry loaders.

Reads the seed CSV (``ADF_250/seed_selection/initial_seed_v1/selected_reactions.csv``)
as the single source of truth for (reaction_id, family, path_r, path_p, path_ts)
and returns ASE Atoms triples for any labelled or candidate reaction.

Per-family conventions:
  - dipolar: path_r → r0; the matching r1_*.xyz lives in the same directory.
             R = r0 ⊕ r1 (matches the dipolar-cycloaddition convention used
             by the existing Δ-learning best). TS = TS_imag_mode_*.xyz in the
             same directory (DFT-converged, NOT the autodE-template path_ts).
             P = path_p (= p0_*.xyz).
  - qmrxn20 e2/sn2: path_r is the reactant-complex XYZ (already a single
             file containing both nucleophile + substrate); path_ts is the
             QM TS XYZ; path_p is the product XYZ. R = path_r, TS = path_ts,
             P = path_p.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import ase
import ase.io
import numpy as np
import pandas as pd


# Same component order as data.ASR_COMPONENTS / sign convention.
ASR_COMPONENTS = (
    "E_strain_kcal", "Pauli_kcal", "V_elst_kcal", "E_orb_kcal", "E_disp_kcal",
)


@dataclass
class GeomTriple:
    """A single reaction's geometry + label."""
    reaction_id: str
    family: str
    R_atoms: ase.Atoms
    TS_atoms: ase.Atoms
    P_atoms: ase.Atoms
    label: Optional[np.ndarray]              # (5,) or None for pool entries
    path_r: str
    path_ts: str
    path_p: str


def load_seed_csv(path: str | Path) -> pd.DataFrame:
    """Returns the master seed CSV with all 800 (or extended) rows."""
    df = pd.read_csv(path)
    return df


def _read_xyz(p: str | Path) -> ase.Atoms:
    return ase.io.read(str(p))


def _single_match(rxn_dir: Path, pattern: str) -> Path:
    """Pick exactly one match for ``pattern``; if multiple, prefer non-_alt."""
    matches = list(rxn_dir.glob(pattern))
    if len(matches) == 1:
        return matches[0]
    primary = [p for p in matches if "_alt" not in p.stem]
    if len(primary) == 1:
        return primary[0]
    raise FileNotFoundError(
        f"expected exactly one {pattern} in {rxn_dir}, found "
        f"{[p.name for p in matches]}"
    )


def _load_dipolar(row: pd.Series) -> tuple[ase.Atoms, ase.Atoms, ase.Atoms]:
    """For dipolar: R = r0 ⊕ r1, TS = TS_imag_mode_*.xyz, P = path_p.
    Filters out ``*_alt.xyz`` alternates when both primary and alt exist."""
    path_r = Path(row["path_r"])
    rxn_dir = path_r.parent
    r0 = _read_xyz(path_r)
    R = r0 + _read_xyz(_single_match(rxn_dir, "r1_*.xyz"))
    # Prefer the DFT-converged TS over the autodE template guess in path_ts.
    try:
        ts_path = _single_match(rxn_dir, "TS_imag_mode*.xyz")
    except FileNotFoundError:
        ts_path = Path(row["path_ts"])
    TS = _read_xyz(ts_path)
    P = _read_xyz(row["path_p"])
    return R, TS, P


def _load_qmrxn(row: pd.Series) -> tuple[ase.Atoms, ase.Atoms, ase.Atoms]:
    """For e2 / sn2: each path points directly to a single XYZ file."""
    R = _read_xyz(row["path_r"])
    TS = _read_xyz(row["path_ts"])
    P = _read_xyz(row["path_p"])
    return R, TS, P


def load_geom(row: pd.Series) -> tuple[ase.Atoms, ase.Atoms, ase.Atoms]:
    """Dispatch on row.family."""
    fam = row["family"]
    if fam == "dipolar":
        return _load_dipolar(row)
    if fam in ("e2", "sn2", "qmrxn20_e2", "qmrxn20_sn2"):
        return _load_qmrxn(row)
    raise ValueError(f"unsupported family: {fam!r} (row id={row['reaction_id']})")


def iter_seed_rows(
    seed_df: pd.DataFrame,
    labels_df: Optional[pd.DataFrame] = None,
    include_labeled: bool = True,
    include_unlabeled: bool = True,
    families: Optional[list[str]] = None,
) -> Iterator[GeomTriple]:
    """Yield :class:`GeomTriple` for every row in the seed CSV, joining with
    labels_df if provided."""
    if families is not None:
        seed_df = seed_df[seed_df["family"].isin(families)]

    if labels_df is not None:
        # Normalize family names so qmrxn20_e2/qmrxn20_sn2 join cleanly with the
        # e2/sn2 family used in the seed CSV.
        lab = labels_df.copy()
        rid_to_label = {}
        for _, r in lab.iterrows():
            rid_to_label[r["reaction_id"]] = r[list(ASR_COMPONENTS)].to_numpy(dtype=np.float32)
    else:
        rid_to_label = {}

    for _, row in seed_df.iterrows():
        rid = row["reaction_id"]
        labelled = rid in rid_to_label
        if labelled and not include_labeled:
            continue
        if not labelled and not include_unlabeled:
            continue
        try:
            R, TS, P = load_geom(row)
        except Exception:
            continue
        yield GeomTriple(
            reaction_id=rid,
            family=row["family"],
            R_atoms=R, TS_atoms=TS, P_atoms=P,
            label=rid_to_label.get(rid),
            path_r=row["path_r"], path_ts=row["path_ts"], path_p=row["path_p"],
        )


# Match label-parquet family names (qmrxn20_e2, qmrxn20_sn2) to seed CSV (e2, sn2).
def normalize_family(name: str) -> str:
    if name == "qmrxn20_e2": return "e2"
    if name == "qmrxn20_sn2": return "sn2"
    return name


def labels_to_seed_join(labels_df: pd.DataFrame) -> pd.DataFrame:
    """Return labels with a 'seed_family' column suitable for joining with the
    seed CSV's family column."""
    df = labels_df.copy()
    df["seed_family"] = df["family"].map(normalize_family)
    return df
