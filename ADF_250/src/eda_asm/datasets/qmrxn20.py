"""Loader for the QMrxn20 dataset (von Rudorff et al., 2020).

Reference: doi:10.1088/2632-2153/aba822
Archive:   https://archive.materialscloud.org/record/2020.55

Layout (after extracting geometries.tgz under root):
  transition-states/{e2|sn2}/{label}.xyz
  reactant-complex-constrained-conformers/{e2|sn2}/{label}/{nn}.xyz
  reactant-complex-unconstrained-conformers/{e2|sn2}/{label}/{nn}.xyz
  product-conformers/{e2|sn2}/{label_product}/{nn}.xyz
  reactant-conformers/{label_substrate}/{nn}.xyz

Per-reaction join via barriers.txt (TS↔reactant complex pairing with
activation energy in kcal/mol). Product label is derived from the TS label:
  e2:  A_B_C_D_E_F  →  A_B_C_D_0_0   (leaving group X and proton H both gone)
  sn2: A_B_C_D_E_F  →  A_B_C_D_0_F   (only X gone; nucleophile Y stays)
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd

from .base import HARTREE_TO_EV, Geometry, ReactionRecord, read_xyz


class QMrxn20Loader:
    """Iterate reactions in QMrxn20 as (R_complex, TS, P) triples.

    Parameters
    ----------
    root : Path
        Directory containing energies.txt, barriers.txt, and the
        transition-states/, reactant-complex-*/, product-conformers/ trees.
    method : {"mp2", "hf", "lccsd"}
        Level of theory used for both energies and barrier filtering.
    reactant_kind : {"rcc", "rcu"}
        rcc = constrained reactant complex (closer to TS topology, preferred
        for ASM); rcu = unconstrained reactant complex.
    """

    def __init__(self, root: str | Path, method: str = "mp2", reactant_kind: str = "rcc"):
        self.root = Path(root)
        if method not in {"mp2", "hf", "lccsd"}:
            raise ValueError(f"method must be mp2|hf|lccsd, got {method!r}")
        if reactant_kind not in {"rcc", "rcu"}:
            raise ValueError(f"reactant_kind must be rcc|rcu, got {reactant_kind!r}")
        self.method = method
        self.reactant_kind = reactant_kind
        self._energies = self._load_table("energies.txt")
        self._barriers = self._load_table("barriers.txt")

    def _load_table(self, name: str) -> pd.DataFrame:
        path = self.root / name
        if not path.exists():
            gz = self.root / f"{name}.gz"
            if gz.exists():
                return pd.read_csv(gz)
            raise FileNotFoundError(f"neither {path} nor {gz} found")
        return pd.read_csv(path)

    def list_reaction_ids(self) -> list[tuple[str, str]]:
        """Return (reaction_class, label) pairs available for the current method/reactant_kind."""
        b = self._barriers
        mask = (b["method"] == self.method) & (b["reactant"] == self.reactant_kind)
        sel = b.loc[mask, ["reaction", "label"]].drop_duplicates()
        return list(sel.itertuples(index=False, name=None))

    def __len__(self) -> int:
        return len(self.list_reaction_ids())

    def __iter__(self) -> Iterator[ReactionRecord]:
        for reaction, label in self.list_reaction_ids():
            try:
                yield self.get(reaction, label)
            except (FileNotFoundError, ValueError):
                continue

    def get(self, reaction: str, label: str) -> ReactionRecord:
        b = self._barriers
        rows = b[
            (b["reaction"] == reaction)
            & (b["label"] == label)
            & (b["method"] == self.method)
            & (b["reactant"] == self.reactant_kind)
        ]
        if rows.empty:
            raise KeyError(
                f"no barriers row for {reaction}/{label} "
                f"({self.method}/{self.reactant_kind})"
            )
        row = rows.iloc[0]

        ts_path = self.root / row["filename_ts"]
        r_path = self.root / row["filename_r"]

        TS = read_xyz(ts_path, energy=float(row["energy_ts"]) * HARTREE_TO_EV)
        R = read_xyz(r_path, energy=float(row["energy_r"]) * HARTREE_TO_EV)

        product_label = _product_label(reaction, label)
        P = self._read_product(reaction, product_label)

        return ReactionRecord(
            reaction_id=f"QMrxn20-{reaction}-{label}",
            family=f"QMrxn20-{reaction}",
            R=R,
            TS=TS,
            P=P,
            activation_energy_kcal=float(row["activation"]),
            extra={
                "label": label,
                "reaction_class": reaction,
                "method": self.method,
                "reactant_kind": self.reactant_kind,
                "ts_conformer_num": int(row["number_ts"]),
                "r_conformer_num": int(row["number_r"]),
                "product_label": product_label,
            },
        )

    def _read_product(self, reaction: str, product_label: str) -> Geometry | None:
        p_dir = self.root / "product-conformers" / reaction / product_label
        if not p_dir.is_dir():
            return None
        p_xyz = p_dir / "00.xyz"
        if not p_xyz.is_file():
            return None
        energy_ha = self._lookup_energy(reaction, product_label, geometry="pc", number=0)
        energy_eV = None if energy_ha is None else energy_ha * HARTREE_TO_EV
        return read_xyz(p_xyz, energy=energy_eV)

    def _lookup_energy(
        self, reaction: str, label: str, geometry: str, number: int
    ) -> float | None:
        e = self._energies
        mask = (
            (e["reaction"] == reaction)
            & (e["label"] == label)
            & (e["geometry"] == geometry)
            & (e["number"] == number)
            & (e["method"] == self.method)
        )
        match = e.loc[mask, "energy"]
        return None if match.empty else float(match.iloc[0])


def _product_label(reaction: str, ts_label: str) -> str:
    parts = ts_label.split("_")
    if len(parts) != 6:
        raise ValueError(f"unexpected QMrxn20 TS label format: {ts_label}")
    if reaction == "e2":
        return "_".join(parts[:4] + ["0", "0"])
    if reaction == "sn2":
        return "_".join(parts[:4] + ["0", parts[5]])
    raise ValueError(f"unknown QMrxn20 reaction class: {reaction!r}")
