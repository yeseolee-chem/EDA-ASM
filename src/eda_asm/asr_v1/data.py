"""Dataset assembly for ASR v1 (dipolar POC).

Joins the 134 dipolar labels in ``asr_labels.parquet`` with the
corresponding R / P geometries from the dipolar-cycloaddition source
(``data/raw/dipolar_cycloaddition/extracted/full_dataset_profiles``).

R is built by concatenating ``r0_*.xyz`` + ``r1_*.xyz`` (the two
relaxed reactants, side-by-side at their standalone geometries).
P is the relaxed product (``p0_*.xyz``).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import ase
import ase.io
import numpy as np
import pandas as pd


# Order matches the parquet columns; (sign, name) per spec §5 of CLAUDE.md.
ASR_COMPONENTS: tuple[str, ...] = (
    "E_strain_kcal",
    "Pauli_kcal",
    "V_elst_kcal",
    "E_orb_kcal",
    "E_disp_kcal",
)
# +1 → softplus(raw),  -1 → -softplus(raw)
ASR_COMPONENT_SIGNS: tuple[int, ...] = (+1, +1, -1, -1, -1)


_RXN_ID_RE = re.compile(r"^dipolar_(\d+)$")


def _parse_dipolar_rxn_id(reaction_id: str) -> Optional[int]:
    m = _RXN_ID_RE.match(reaction_id)
    return int(m.group(1)) if m else None


@dataclass
class AsrSample:
    reaction_id: str
    rxn_id_int: int
    R_atoms: ase.Atoms
    P_atoms: ase.Atoms
    label: np.ndarray  # shape (5,), kcal/mol, ordered per ASR_COMPONENTS


@dataclass
class AsrSampleRTSP:
    """3-way variant: reactant, TS, product geometries + label."""
    reaction_id: str
    rxn_id_int: int
    R_atoms: ase.Atoms
    TS_atoms: ase.Atoms
    P_atoms: ase.Atoms
    label: np.ndarray  # (5,), kcal/mol


def load_label_table(
    parquet_path: str | Path,
    family: str = "dipolar",
) -> pd.DataFrame:
    """Load and filter the labels parquet to a single family."""
    df = pd.read_parquet(parquet_path)
    df = df[df["family"] == family].reset_index(drop=True).copy()
    df["rxn_id_int"] = df["reaction_id"].map(_parse_dipolar_rxn_id)
    if family == "dipolar" and df["rxn_id_int"].isna().any():
        bad = df[df["rxn_id_int"].isna()]["reaction_id"].tolist()
        raise ValueError(f"failed to parse dipolar reaction ids: {bad[:5]}")
    return df


def _single_match(rxn_dir: Path, pattern: str) -> Path:
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


def _build_R(rxn_dir: Path) -> ase.Atoms:
    """Concatenate r0 + r1 standalone-relaxed geometries into one Atoms."""
    r0 = ase.io.read(_single_match(rxn_dir, "r0_*.xyz"))
    r1 = ase.io.read(_single_match(rxn_dir, "r1_*.xyz"))
    return r0 + r1  # ASE supports Atoms concatenation


def _build_P(rxn_dir: Path) -> ase.Atoms:
    return ase.io.read(_single_match(rxn_dir, "p0_*.xyz"))


def _build_TS(rxn_dir: Path) -> ase.Atoms:
    """Load the DFT-converged TS (TS_imag_mode.xyz). Falls back to the
    autodE template guess if the converged file is missing."""
    matches = list(rxn_dir.glob("TS_imag_mode*.xyz"))
    if not matches:
        matches = list(rxn_dir.glob("TS_ts_guess*.xyz"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one TS_*.xyz in {rxn_dir}, found "
            f"{[p.name for p in matches]}"
        )
    return ase.io.read(matches[0])


def iter_reaction_pairs(
    df: pd.DataFrame,
    dipolar_root: str | Path = "data/raw/dipolar_cycloaddition/extracted/full_dataset_profiles",
    skip_errors: bool = False,
) -> Iterator[AsrSample]:
    """Yield (R, P, label) tuples for each labelled dipolar reaction in df."""
    root = Path(dipolar_root)
    labels = df[list(ASR_COMPONENTS)].to_numpy(dtype=np.float32)
    for i, row in df.reset_index(drop=True).iterrows():
        rxn_dir = root / str(int(row["rxn_id_int"]))
        try:
            R = _build_R(rxn_dir)
            P = _build_P(rxn_dir)
        except FileNotFoundError as exc:
            if skip_errors:
                continue
            raise
        yield AsrSample(
            reaction_id=row["reaction_id"],
            rxn_id_int=int(row["rxn_id_int"]),
            R_atoms=R,
            P_atoms=P,
            label=labels[i].copy(),
        )


def iter_reaction_triples(
    df: pd.DataFrame,
    dipolar_root: str | Path = "data/raw/dipolar_cycloaddition/extracted/full_dataset_profiles",
    skip_errors: bool = False,
) -> Iterator[AsrSampleRTSP]:
    """Yield (R, TS, P, label) quadruples for the RTSP pipeline."""
    root = Path(dipolar_root)
    labels = df[list(ASR_COMPONENTS)].to_numpy(dtype=np.float32)
    for i, row in df.reset_index(drop=True).iterrows():
        rxn_dir = root / str(int(row["rxn_id_int"]))
        try:
            R = _build_R(rxn_dir)
            TS = _build_TS(rxn_dir)
            P = _build_P(rxn_dir)
        except FileNotFoundError:
            if skip_errors:
                continue
            raise
        yield AsrSampleRTSP(
            reaction_id=row["reaction_id"],
            rxn_id_int=int(row["rxn_id_int"]),
            R_atoms=R,
            TS_atoms=TS,
            P_atoms=P,
            label=labels[i].copy(),
        )
