"""Direct SQLite access for Halo8 ASE-DB files.

We bypass ase.db for the bulk index pass because the row API is ~5x slower
than reading the relevant BLOBs directly. The blobs use a simple ASE
format: little-endian int64 offset in the first 8 bytes, then numpy arrays
(if any), then JSON. For Halo8 the data BLOB has no numpy arrays inside,
so JSON starts at byte 8.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


def _decode_blob_simple(b: bytes) -> object:
    """Decode an ASE blob that contains only JSON (no embedded numpy arrays)."""
    if b is None:
        return None
    offset = int(np.frombuffer(b[:8], np.int64)[0])
    return json.loads(b[offset:].decode())


def _decode_numpy_blob(b: bytes, dtype: np.dtype) -> np.ndarray | None:
    """Decode an ASE BLOB that holds a single raw numpy array.

    In Halo8's ASE-DB version 9 the numbers/positions/forces BLOBs store the
    raw bytes of the array with no 8-byte offset header (unlike the `data`
    BLOB, which is a JSON tree with optional @nparray references). So we
    just deserialise from the start.
    """
    if b is None:
        return None
    return np.frombuffer(b, dtype=dtype)


def decode_data(b: bytes) -> dict:
    obj = _decode_blob_simple(b)
    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise TypeError(f"expected dict in data BLOB, got {type(obj).__name__}")
    return obj


def decode_numbers(b: bytes) -> np.ndarray:
    arr = _decode_numpy_blob(b, np.dtype(np.int32))
    if arr is None:
        raise ValueError("empty numbers BLOB")
    return arr.astype(np.int64, copy=False)


def decode_positions(b: bytes, natoms: int) -> np.ndarray:
    arr = _decode_numpy_blob(b, np.dtype(np.float64))
    if arr is None:
        raise ValueError("empty positions BLOB")
    return arr.reshape(natoms, 3).copy()


def decode_forces(b: bytes, natoms: int) -> np.ndarray | None:
    arr = _decode_numpy_blob(b, np.dtype(np.float64))
    if arr is None:
        return None
    return arr.reshape(natoms, 3).copy()


@dataclass(slots=True)
class FrameRow:
    """All Halo8 fields needed downstream for a single frame."""

    dand_id: str
    trajectory_id: str
    frame_idx: int
    energy: float
    natoms: int
    charge: float
    numbers: np.ndarray
    positions: np.ndarray
    forces: np.ndarray | None
    data: dict

    @property
    def formula(self) -> str:
        return _formula_from_numbers(self.numbers)


def parse_trajectory_id(dand_id: str) -> tuple[str, int]:
    """Return (trajectory_id, frame_idx)."""
    head, _, tail = dand_id.rpartition("_")
    if not head or not tail:
        raise ValueError(f"unexpected dand_id: {dand_id}")
    return head, int(tail)


def family_from_traj_id(traj_id: str) -> str:
    low = traj_id.lower()
    if low.startswith("t1x"):
        return "T1x"
    if low.startswith("halo"):
        return "Halogen"
    return "Other"


def halogen_subfamily(numbers: np.ndarray) -> str | None:
    """Return one of {"Halo_F","Halo_Cl","Halo_Br"} or None if no halogen present.

    Priority Br > Cl > F (heaviest halogen wins) so a F+Br molecule is bucketed
    under Halo_Br to track the most distinctive heavy element.
    """
    z = set(int(x) for x in numbers.tolist())
    if 35 in z:
        return "Halo_Br"
    if 17 in z:
        return "Halo_Cl"
    if 9 in z:
        return "Halo_F"
    return None


def assign_source(traj_id: str, numbers: np.ndarray) -> str:
    fam = family_from_traj_id(traj_id)
    if fam == "T1x":
        return "T1x"
    if fam == "Halogen":
        sub = halogen_subfamily(numbers)
        return sub or "Halogen_other"
    return "Other"


_ELEMENTS = [
    "X", "H", "He",
    "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe",
]


def _formula_from_numbers(numbers: np.ndarray) -> str:
    """Hill-system formula from atomic numbers."""
    counts: dict[str, int] = {}
    for z in numbers.tolist():
        sym = _ELEMENTS[int(z)] if int(z) < len(_ELEMENTS) else f"Z{int(z)}"
        counts[sym] = counts.get(sym, 0) + 1
    out = []
    for sym in ("C", "H"):
        if sym in counts:
            n = counts.pop(sym)
            out.append(sym + (str(n) if n > 1 else ""))
    for sym in sorted(counts):
        n = counts[sym]
        out.append(sym + (str(n) if n > 1 else ""))
    return "".join(out)


def iter_index_rows(db_path: Path, batch: int = 50000) -> Iterator[tuple[int, str, float, int, float, np.ndarray, dict]]:
    """Yield (id, dand_id, energy, natoms, charge, numbers, data_dict) per row.

    Streams in batches to keep memory low. Suitable for the indexing pass that
    only needs metadata (no positions / forces).
    """
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute(
        "SELECT id, energy, natoms, charge, numbers, data FROM systems ORDER BY id"
    )
    while True:
        rows = cur.fetchmany(batch)
        if not rows:
            break
        for rid, energy, natoms, charge, numbers_blob, data_blob in rows:
            data = decode_data(data_blob)
            numbers = decode_numbers(numbers_blob)
            yield rid, str(data["dand_id"]), float(energy), int(natoms), float(charge or 0.0), numbers, data
    conn.close()


def fetch_frames(
    db_path: Path,
    trajectory_id: str,
) -> list[FrameRow]:
    """Return all frames of a single trajectory, sorted by frame index.

    Used by Stage 3.4 (5-point extraction). Reads positions/forces/data.
    """
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    # ASE row id is stored separately; we filter via the data BLOB. The data
    # column is a BLOB (not searchable via SQL JSON), so we scan and filter.
    # This is fine: each DB has ~2M rows but only one trajectory is rare per call.
    cur.execute(
        "SELECT energy, natoms, charge, numbers, positions, forces, data FROM systems"
    )
    out: list[FrameRow] = []
    while True:
        rows = cur.fetchmany(50000)
        if not rows:
            break
        for energy, natoms, charge, nb, pb, fb, db in rows:
            data = decode_data(db)
            did = str(data["dand_id"])
            traj, frame_idx = parse_trajectory_id(did)
            if traj != trajectory_id:
                continue
            numbers = decode_numbers(nb)
            positions = decode_positions(pb, int(natoms))
            forces = decode_forces(fb, int(natoms))
            out.append(
                FrameRow(
                    dand_id=did,
                    trajectory_id=traj,
                    frame_idx=frame_idx,
                    energy=float(energy),
                    natoms=int(natoms),
                    charge=float(charge or 0.0),
                    numbers=numbers,
                    positions=positions,
                    forces=forces,
                    data=data,
                )
            )
    conn.close()
    out.sort(key=lambda r: r.frame_idx)
    return out


def fetch_frames_multi(
    db_path: Path,
    trajectory_ids: set[str],
) -> dict[str, list[FrameRow]]:
    """Return {traj_id: [FrameRow,...]} for any trajectory_ids found in this DB."""
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute(
        "SELECT energy, natoms, charge, numbers, positions, forces, data FROM systems"
    )
    out: dict[str, list[FrameRow]] = {tid: [] for tid in trajectory_ids}
    while True:
        rows = cur.fetchmany(50000)
        if not rows:
            break
        for energy, natoms, charge, nb, pb, fb, db_ in rows:
            data = decode_data(db_)
            did = str(data["dand_id"])
            traj, frame_idx = parse_trajectory_id(did)
            if traj not in out:
                continue
            numbers = decode_numbers(nb)
            positions = decode_positions(pb, int(natoms))
            forces = decode_forces(fb, int(natoms))
            out[traj].append(
                FrameRow(
                    dand_id=did,
                    trajectory_id=traj,
                    frame_idx=frame_idx,
                    energy=float(energy),
                    natoms=int(natoms),
                    charge=float(charge or 0.0),
                    numbers=numbers,
                    positions=positions,
                    forces=forces,
                    data=data,
                )
            )
    conn.close()
    for tid in out:
        out[tid].sort(key=lambda r: r.frame_idx)
    return {tid: rows for tid, rows in out.items() if rows}
