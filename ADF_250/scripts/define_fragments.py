#!/usr/bin/env python
"""Per-reaction TS supermolecule partitioning into fragments A and B.

Implements ASR_Fragment_Partitioning_Spec_v1.0.md:
  - SMARTS-based partition for SN2/E2 (anion + substrate) and dipolar
    cycloaddition (dipole + dipolarophile)
  - D2AF fallback (connected-components after removing reacting bonds)
  - Manual-override hook
  - Hard + soft constraint validation
  - Writes fragments.parquet, fragments_audit.csv, failures.csv,
    manifest.json, partitioning_log.txt

KEY DEPARTURE from spec §5.3 code:
  autodE's TS XYZ atom order is NOT the naive concatenation of SMILES
  atoms (it's `all heavy atoms followed by all hydrogens` after autodE's
  internal permutation). We therefore use RDKit's `GetSubstructMatch`
  against the TS connectivity graph to assign each SMILES atom to its
  TS atom index, rather than assuming `range(0, n_dipole)`.

Exit codes: 0 ok, 1 preflight, 2 input malformed, 3 too many failures, 4 IO.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import sys
import traceback
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from rdkit import Chem
from rdkit.Chem.rdDetermineBonds import DetermineBonds, DetermineConnectivity

REPO = Path(__file__).resolve().parents[1]

# Pyykkö covalent radii (pm), spec §6.2
COVALENT_RADII_PM = {
    "H": 32, "He": 28,
    "C": 75, "N": 71, "O": 63, "F": 64,
    "P": 111, "S": 103, "Cl": 99, "Br": 114, "I": 133,
    "B": 85, "Al": 126, "Si": 116, "Na": 166, "K": 203,
}

EXIT_PREFLIGHT = 1
EXIT_INPUT = 2
EXIT_FAILURES = 3
EXIT_IO = 4

SPEC_VERSION = "1.0"
SCRIPT_VERSION = "smarts_v1.0"


# --------------------------- data ---------------------------
@dataclass
class PartitionResult:
    reaction_id: str = ""
    family: str = ""
    partition_method: str = ""              # "smarts" | "d2af" | "manual_override" | "failed"
    partition_status: str = ""              # "ok" | "warning" | "failed"
    failure_reason: str | None = None
    warning_codes: list[str] = field(default_factory=list)

    n_atoms_total: int = 0
    fragment_atoms_a: list[int] = field(default_factory=list)
    fragment_atoms_b: list[int] = field(default_factory=list)
    fragment_smiles_a: str | None = None
    fragment_smiles_b: str | None = None
    fragment_charge_a: int = 0
    fragment_charge_b: int = 0
    fragment_mult_a: int = 1
    fragment_mult_b: int = 1
    n_atoms_a: int = 0
    n_atoms_b: int = 0

    min_interfragment_dist_ts: float = -1.0
    min_interfragment_dist_r: float = -1.0
    reacting_bonds: list[tuple[int, int, str]] = field(default_factory=list)

    partition_method_version: str = SCRIPT_VERSION
    partitioned_utc: str = ""
    spec_version: str = SPEC_VERSION


# --------------------------- utilities ---------------------------
def setup_logging(log_path: Path, level: str) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("fragments")
    log.setLevel(level)
    for h in list(log.handlers):
        log.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s")
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)
    return log


def read_xyz(path: Path) -> tuple[list[str], np.ndarray]:
    """Read xyz file; return (elements, positions in Å)."""
    text = Path(path).read_text()
    lines = text.strip().splitlines()
    n = int(lines[0].strip())
    elements: list[str] = []
    positions = np.zeros((n, 3), dtype=np.float64)
    for i, raw in enumerate(lines[2:2 + n]):
        toks = raw.split()
        elements.append(toks[0])
        positions[i] = [float(t) for t in toks[1:4]]
    return elements, positions


def xyz_to_connectivity_mol(path: Path, charge: int = 0) -> Chem.Mol | None:
    raw = Chem.MolFromXYZFile(str(path))
    if raw is None:
        return None
    mol = Chem.Mol(raw)
    try:
        DetermineConnectivity(mol, charge=charge)
    except (ValueError, RuntimeError):
        return None
    return mol


def to_skeleton_query(mol: Chem.Mol) -> Chem.Mol:
    """Strip bond orders / formal charges / explicit Hs so the resulting Mol
    matches a connectivity-only target. RDKit's DetermineConnectivity leaves
    all bonds as SINGLE; the source SMILES Mol has explicit DOUBLE / TRIPLE,
    which would cause an exact-substructure miss. We pin all bonds to SINGLE
    and clear formal charges so substructure matching considers the skeleton
    only.
    """
    rw = Chem.RWMol(Chem.Mol(mol))
    for bond in rw.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
    for atom in rw.GetAtoms():
        atom.SetFormalCharge(0)
        atom.SetNumExplicitHs(0)
        atom.SetNoImplicit(True)
    return rw.GetMol()


def xyz_to_bonded_mol(path: Path, charge: int = 0) -> Chem.Mol | None:
    raw = Chem.MolFromXYZFile(str(path))
    if raw is None:
        return None
    mol = Chem.Mol(raw)
    try:
        DetermineBonds(mol, charge=charge, embedChiral=False)
        Chem.SanitizeMol(mol)
    except (ValueError, RuntimeError, Chem.rdchem.AtomValenceException,
            Chem.rdchem.KekulizeException):
        return None
    return mol


def min_interfragment_dist(positions: np.ndarray,
                           atoms_a: list[int], atoms_b: list[int]) -> float:
    if not atoms_a or not atoms_b:
        return -1.0
    A = positions[atoms_a]
    B = positions[atoms_b]
    diff = A[:, None, :] - B[None, :, :]
    d = np.sqrt((diff * diff).sum(axis=-1))
    return float(d.min())


def is_bonded(e1: str, e2: str, d: float, tolerance: float = 0.30) -> bool:
    r1 = COVALENT_RADII_PM.get(e1, 100)
    r2 = COVALENT_RADII_PM.get(e2, 100)
    return d <= (r1 + r2) / 100.0 * (1 + tolerance)


def detect_bonds(elements: list[str], positions: np.ndarray,
                 tolerance: float = 0.30) -> set[tuple[int, int]]:
    """Distance-based covalent bond detection per spec §6.2."""
    n = len(elements)
    bonds: set[tuple[int, int]] = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(positions[i] - positions[j]))
            if is_bonded(elements[i], elements[j], d, tolerance):
                bonds.add((i, j))
    return bonds


# --------------------------- step 0: atom-order validation ---------------------------
def validate_atom_order(row: pd.Series, log: logging.Logger) -> bool:
    """Verify R/TS share atom count (P may differ for QMrxn20 — leaving group
    stripped from product file)."""
    try:
        r_el, _ = read_xyz(Path(row["path_r"]))
        ts_el, _ = read_xyz(Path(row["path_ts"]))
        p_el, _ = read_xyz(Path(row["path_p"]))
    except Exception as e:
        log.error(f"  {row['reaction_id']}: xyz read failed: {e}")
        return False

    fam = row["family"]
    if fam == "dipolar":
        # Stuyver dataset: path_r holds just one isolated fragment (r0),
        # path_p holds the cycloaddition product. Only TS spans the full
        # system. Skip strict R/TS count check; downstream substructure
        # match uses TS alone.
        return True

    # QMrxn20 (e2/sn2): R is the reactant *complex* (substrate + Y), same
    # atom count as TS. Product file has the leaving group X stripped, so
    # P count is smaller — that's expected.
    if len(r_el) != len(ts_el):
        log.error(
            f"  {row['reaction_id']}: R/TS atom counts differ R={len(r_el)} TS={len(ts_el)}"
        )
        return False
    if r_el != ts_el:
        log.debug(
            f"  {row['reaction_id']}: R/TS element orderings differ — "
            "downstream uses substructure matching, OK"
        )
    return True


# --------------------------- partition methods ---------------------------
def smarts_partition(row: pd.Series, cfg: dict,
                     log: logging.Logger) -> PartitionResult | None:
    family = row["family"]
    if family in ("e2", "sn2"):
        return _smarts_partition_qmrxn20(row, cfg, log)
    if family == "dipolar":
        return _smarts_partition_dipolar(row, cfg, log)
    return None


def _smarts_partition_qmrxn20(row: pd.Series, cfg: dict,
                              log: logging.Logger) -> PartitionResult | None:
    """SMARTS-based SN2/E2 partition (spec §5.1, §5.2).

    Strategy: use the *TS* xyz (path_ts). QMrxn20's reactant-complex and TS
    files do not share atom ordering, so partitioning on path_r then applying
    the indices to path_ts produces a wrong fragmentation (e.g. assigning a
    hydrogen to the anion slot). The TS itself has the forming bond
    Y...substrate at ~2.3-2.7 Å, which exceeds RDKit's default covalent
    cutoff, so DetermineConnectivity gives two disconnected fragments:
    the standalone anion (Y) and the substrate (R-X).
    """
    ts_mol = xyz_to_connectivity_mol(Path(row["path_ts"]), charge=int(row["charge"]))
    if ts_mol is None:
        log.debug(f"  {row['reaction_id']}: DetermineConnectivity(TS) failed")
        return None

    frags = Chem.GetMolFrags(ts_mol, asMols=False)

    if len(frags) >= 2:
        sized = sorted(frags, key=len)
        atoms_a = sorted(int(i) for i in sized[0])
        atoms_b = sorted(int(i) for c in sized[1:] for i in c)
    else:
        # TS is a single connected component — Y is already bonded to the
        # substrate at TS distance. Cut at the longest bond, which is the
        # partially-formed Y...substrate bond.
        ts_el, ts_pos = read_xyz(Path(row["path_ts"]))
        bonds = detect_bonds(ts_el, ts_pos, tolerance=cfg["d2af"]["bond_detection_tolerance"])
        bonds_with_len = [(i, j, float(np.linalg.norm(ts_pos[i] - ts_pos[j])))
                          for i, j in bonds]
        if not bonds_with_len:
            return None
        bonds_with_len.sort(key=lambda t: -t[2])
        longest = bonds_with_len[0]
        from rdkit.Chem.rdchem import EditableMol
        em = EditableMol(ts_mol)
        em.RemoveBond(longest[0], longest[1])
        cut = em.GetMol()
        frags = Chem.GetMolFrags(cut, asMols=False)
        if len(frags) < 2:
            return None
        sized = sorted(frags, key=len)
        atoms_a = sorted(int(i) for i in sized[0])
        atoms_b = sorted(int(i) for c in sized[1:] for i in c)

    total_charge = int(row["charge"])
    res = PartitionResult(
        reaction_id=row["reaction_id"],
        family=row["family"],
        partition_method="smarts",
        partition_status="ok",
        n_atoms_total=int(ts_mol.GetNumAtoms()),
        fragment_atoms_a=atoms_a,
        fragment_atoms_b=atoms_b,
        n_atoms_a=len(atoms_a),
        n_atoms_b=len(atoms_b),
        fragment_charge_a=-1,                    # anion convention
        fragment_charge_b=total_charge + 1,      # remainder
        fragment_mult_a=1,
        fragment_mult_b=1,
    )
    return res


def _smarts_partition_dipolar(row: pd.Series, cfg: dict,
                              log: logging.Logger) -> PartitionResult | None:
    """Dipolar cycloaddition partition (spec §5.3) using SMILES dot-split +
    RDKit substructure match against TS connectivity graph.

    This corrects spec §5.3's "atom order in TS XYZ matches SMILES" assumption,
    which is invalid for the Stuyver dataset (autodE re-orders to heavy-first
    then hydrogens).
    """
    smiles_r = str(row["smiles_r"])
    parts = smiles_r.split(".")
    if len(parts) != 2:
        log.debug(f"  {row['reaction_id']}: SMILES has {len(parts)} parts, expected 2")
        return None

    mol_a = Chem.MolFromSmiles(parts[0])
    mol_b = Chem.MolFromSmiles(parts[1])
    if mol_a is None or mol_b is None:
        return None
    mol_a_h = Chem.AddHs(mol_a)
    mol_b_h = Chem.AddHs(mol_b)

    ts_mol = xyz_to_connectivity_mol(Path(row["path_ts"]), charge=int(row["charge"]))
    if ts_mol is None:
        log.debug(f"  {row['reaction_id']}: TS connectivity failed")
        return None
    n_total = ts_mol.GetNumAtoms()
    if mol_a_h.GetNumAtoms() + mol_b_h.GetNumAtoms() != n_total:
        log.debug(f"  {row['reaction_id']}: SMILES atom count "
                  f"{mol_a_h.GetNumAtoms()}+{mol_b_h.GetNumAtoms()} != TS {n_total}")
        return None

    query_a = to_skeleton_query(mol_a_h)
    query_b = to_skeleton_query(mol_b_h)
    match_a = ts_mol.GetSubstructMatch(query_a, useChirality=False)
    match_b = ts_mol.GetSubstructMatch(query_b, useChirality=False)
    if not match_a or not match_b:
        log.debug(f"  {row['reaction_id']}: substructure match failed "
                  f"(|A|={len(match_a)}/{mol_a_h.GetNumAtoms()}, "
                  f"|B|={len(match_b)}/{mol_b_h.GetNumAtoms()})")
        return None

    set_a = set(match_a)
    set_b = set(match_b)
    if set_a & set_b:
        return None
    if len(set_a) + len(set_b) != n_total:
        return None

    charge_a = Chem.GetFormalCharge(mol_a)
    charge_b = Chem.GetFormalCharge(mol_b)

    return PartitionResult(
        reaction_id=row["reaction_id"],
        family=row["family"],
        partition_method="smarts",
        partition_status="ok",
        n_atoms_total=int(n_total),
        fragment_atoms_a=sorted(int(i) for i in set_a),
        fragment_atoms_b=sorted(int(i) for i in set_b),
        fragment_smiles_a=Chem.MolToSmiles(mol_a),
        fragment_smiles_b=Chem.MolToSmiles(mol_b),
        n_atoms_a=len(set_a),
        n_atoms_b=len(set_b),
        fragment_charge_a=charge_a,
        fragment_charge_b=charge_b,
        fragment_mult_a=1,
        fragment_mult_b=1,
    )


def d2af_partition(row: pd.Series, cfg: dict,
                   log: logging.Logger) -> PartitionResult | None:
    """D2AF fallback (spec §6) — connected-components after removing reacting bonds."""
    try:
        r_el, r_pos = read_xyz(Path(row["path_r"]))
        p_el, p_pos = read_xyz(Path(row["path_p"]))
    except Exception:
        return None
    if r_el != p_el:
        # Atom orders differ; D2AF assumes same atom ordering for R and P.
        return None

    tol = float(cfg["d2af"]["bond_detection_tolerance"])
    bonds_r = detect_bonds(r_el, r_pos, tolerance=tol)
    bonds_p = detect_bonds(p_el, p_pos, tolerance=tol)
    reacting = bonds_r.symmetric_difference(bonds_p)
    g_r_minus = bonds_r - reacting   # R bonds excluding reacting

    n = len(r_el)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, j in g_r_minus:
        union(i, j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    comps = sorted(groups.values(), key=len)

    if len(comps) < 2:
        return None  # cannot split, spec §6.1 unimolecular_no_partition
    # Merge tiny components into nearest larger until 2 remain (spec §6.1)
    while len(comps) > 2:
        smallest = comps.pop(0)
        # Add to nearest by minimum atom-pair distance
        nearest = min(
            comps,
            key=lambda c: min(
                np.linalg.norm(r_pos[i] - r_pos[j]) for i in smallest for j in c
            ),
        )
        nearest.extend(smallest)
        comps = sorted(comps, key=len)

    atoms_a = sorted(int(i) for i in comps[0])
    atoms_b = sorted(int(i) for i in comps[1])

    total_charge = int(row["charge"])

    # Reacting bonds, labeled
    rb = []
    for i, j in reacting:
        kind = "form" if (i, j) in bonds_p else "break"
        rb.append((i, j, kind))

    return PartitionResult(
        reaction_id=row["reaction_id"],
        family=row["family"],
        partition_method="d2af",
        partition_status="ok",
        n_atoms_total=n,
        fragment_atoms_a=atoms_a,
        fragment_atoms_b=atoms_b,
        n_atoms_a=len(atoms_a),
        n_atoms_b=len(atoms_b),
        fragment_charge_a=-1 if row["family"] in ("e2", "sn2") else 0,
        fragment_charge_b=total_charge - (-1 if row["family"] in ("e2", "sn2") else 0),
        fragment_mult_a=1,
        fragment_mult_b=1,
        reacting_bonds=rb,
    )


# --------------------------- validation ---------------------------
def validate_partition(res: PartitionResult, row: pd.Series, cfg: dict,
                       log: logging.Logger) -> dict[str, Any]:
    warnings: list[str] = []

    a, b = set(res.fragment_atoms_a), set(res.fragment_atoms_b)

    if a & b:
        return {"status": "failed", "failure_reason": "overlap"}
    if a | b != set(range(res.n_atoms_total)):
        return {"status": "failed", "failure_reason": "incomplete_coverage"}
    if not a or not b:
        return {"status": "failed", "failure_reason": "empty_fragment"}
    if res.fragment_charge_a + res.fragment_charge_b != int(row["charge"]):
        return {"status": "failed", "failure_reason": "charge_mismatch"}

    total_mult = int(row.get("multiplicity", 1))
    if (res.fragment_mult_a + res.fragment_mult_b - 1) % 2 != total_mult % 2:
        return {"status": "failed", "failure_reason": "mult_parity_mismatch"}

    # HC-6: no covalent bond in R crosses partition
    try:
        ts_el, ts_pos = read_xyz(Path(row["path_ts"]))
    except Exception:
        return {"status": "failed", "failure_reason": "atom_order_mismatch"}

    fam = res.family
    if fam == "dipolar":
        # Stuyver: R is isolated r0+r1, HC-6 is trivially satisfied.
        res.min_interfragment_dist_r = float("inf")
        res.min_interfragment_dist_ts = min_interfragment_dist(
            ts_pos, list(a), list(b)
        )
    else:
        # QMrxn20: R and TS have different atom orderings; partition is on TS
        # indices. Skip path_r-based bond-crossing check (would compare apples
        # to oranges). Bond-crossing in TS is by construction satisfied:
        # connected-components partition cannot cross a TS bond.
        res.min_interfragment_dist_r = float("nan")
        res.min_interfragment_dist_ts = min_interfragment_dist(
            ts_pos, list(a), list(b)
        )
    upper = float(cfg["validation"]["min_interfragment_dist_ts_warn_upper"])
    lower = float(cfg["validation"]["min_interfragment_dist_ts_warn_lower"])
    if res.min_interfragment_dist_ts > upper:
        warnings.append("fragments_too_far")
    if 0 < res.min_interfragment_dist_ts < lower:
        warnings.append("fragments_too_close")

    # SC-3: imbalance
    n = res.n_atoms_total
    smaller = min(res.n_atoms_a, res.n_atoms_b)
    if smaller > 1 and (smaller / n) < 0.05:
        warnings.append("very_imbalanced")

    # family-specific SC
    fam = res.family
    if fam in ("e2", "sn2"):
        if res.n_atoms_a != 1:
            warnings.append("anion_not_single_atom")
    elif fam == "dipolar":
        if not (3 <= res.n_atoms_a <= 30 or 3 <= res.n_atoms_b <= 30):
            warnings.append("dipolar_size_outside_range")

    res.warning_codes = warnings
    return {"status": "warning" if warnings else "ok", "warning_codes": warnings}


# --------------------------- main loop ---------------------------
def partition_reaction(row: pd.Series, cfg: dict, manual_overrides: dict,
                       log: logging.Logger) -> PartitionResult:
    rid = row["reaction_id"]

    if rid in manual_overrides:
        m = manual_overrides[rid]
        res = PartitionResult(
            reaction_id=rid, family=row["family"],
            partition_method="manual_override", partition_status="ok",
            n_atoms_total=int(m["n_atoms_total"]),
            fragment_atoms_a=m["atoms_a"], fragment_atoms_b=m["atoms_b"],
            fragment_charge_a=int(m["charge_a"]), fragment_charge_b=int(m["charge_b"]),
            fragment_mult_a=int(m["mult_a"]), fragment_mult_b=int(m["mult_b"]),
            n_atoms_a=len(m["atoms_a"]), n_atoms_b=len(m["atoms_b"]),
        )
        v = validate_partition(res, row, cfg, log)
        if v["status"] == "failed":
            res.partition_status = "failed"
            res.failure_reason = v["failure_reason"]
        return res

    if not validate_atom_order(row, log):
        return PartitionResult(
            reaction_id=rid, family=row["family"],
            partition_method="failed", partition_status="failed",
            failure_reason="atom_order_mismatch",
        )

    for method in cfg["method_order"]:
        if method == "manual_override":
            continue
        if method == "smarts":
            res = smarts_partition(row, cfg, log)
        elif method == "d2af":
            res = d2af_partition(row, cfg, log)
        else:
            continue
        if res is None:
            continue
        v = validate_partition(res, row, cfg, log)
        if v["status"] == "failed":
            log.debug(f"  {rid}: {method} validation failed {v['failure_reason']}")
            continue
        res.partition_status = v["status"]
        return res

    return PartitionResult(
        reaction_id=rid, family=row["family"],
        partition_method="failed", partition_status="failed",
        failure_reason="no_method_succeeded",
    )


def to_record(res: PartitionResult) -> dict:
    d = asdict(res)
    d["fragment_atoms_a"] = json.dumps(res.fragment_atoms_a)
    d["fragment_atoms_b"] = json.dumps(res.fragment_atoms_b)
    d["warning_codes"] = ",".join(res.warning_codes) if res.warning_codes else None
    d["reacting_bonds"] = json.dumps(res.reacting_bonds) if res.reacting_bonds else None
    return d


# --------------------------- CLI ---------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--seed-csv", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--manual-overrides", type=Path, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    seed_csv = args.seed_csv or (REPO / cfg["inputs"]["seed_csv"])
    out_dir = args.output_dir or (REPO / cfg["output_dir"])
    out_dir = Path(out_dir).resolve()

    if out_dir.exists() and any(out_dir.iterdir()):
        non_log = [p for p in out_dir.iterdir() if p.name != "partitioning_log.txt"]
        if non_log and not args.force:
            sys.stderr.write(f"ERROR: output dir non-empty: {out_dir} (--force)\n")
            sys.exit(EXIT_PREFLIGHT)
        if args.force:
            import shutil
            shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(out_dir / "partitioning_log.txt", args.log_level)
    log.info("=" * 70)
    log.info("ASR Fragment Partitioning Spec v1.0")
    log.info("=" * 70)
    log.info(f"config       : {args.config}")
    log.info(f"seed_csv     : {seed_csv}")
    log.info(f"output_dir   : {out_dir}")

    manual_overrides: dict[str, dict] = {}
    if args.manual_overrides and Path(args.manual_overrides).is_file():
        mo_df = pd.read_csv(args.manual_overrides)
        for _, r in mo_df.iterrows():
            manual_overrides[r["reaction_id"]] = {
                "atoms_a": json.loads(r["fragment_atoms_a"]),
                "atoms_b": json.loads(r["fragment_atoms_b"]),
                "charge_a": int(r["fragment_charge_a"]),
                "charge_b": int(r["fragment_charge_b"]),
                "mult_a": int(r["fragment_mult_a"]),
                "mult_b": int(r["fragment_mult_b"]),
                "n_atoms_total": int(r.get("n_atoms_total", 0)),
            }
        log.info(f"manual overrides loaded: {len(manual_overrides)}")

    seed = pd.read_csv(seed_csv)
    seed = seed.sort_values("reaction_id", kind="stable").reset_index(drop=True)
    log.info(f"reactions to partition: {len(seed)}")

    results: list[PartitionResult] = []
    method_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    fail_counts: Counter[str] = Counter()
    t0 = datetime.now()

    for idx, row in seed.iterrows():
        try:
            res = partition_reaction(row, cfg, manual_overrides, log)
        except Exception as e:
            log.error(f"  unhandled: {row['reaction_id']}: {e}")
            log.debug(traceback.format_exc())
            res = PartitionResult(
                reaction_id=row["reaction_id"], family=row["family"],
                partition_method="failed", partition_status="failed",
                failure_reason="unhandled_exception",
            )
        res.partitioned_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        results.append(res)
        method_counts[res.partition_method] += 1
        status_counts[res.partition_status] += 1
        if res.partition_status == "failed":
            fail_counts[res.failure_reason or "unknown"] += 1
        if (idx + 1) % 100 == 0:
            log.info(f"  progress: {idx + 1}/{len(seed)}  "
                     f"methods={dict(method_counts)}  statuses={dict(status_counts)}")

    log.info(f"done in {(datetime.now() - t0).total_seconds():.1f}s")
    log.info(f"methods : {dict(method_counts)}")
    log.info(f"statuses: {dict(status_counts)}")
    log.info(f"failures: {dict(fail_counts)}")

    df = pd.DataFrame.from_records([to_record(r) for r in results])
    df.to_parquet(out_dir / "fragments.parquet",
                  compression=cfg["io"]["parquet_compression"])
    df.to_csv(out_dir / "fragments_audit.csv", index=False)
    fails = df[df["partition_status"] == "failed"].copy()
    fails.to_csv(out_dir / "failures.csv", index=False)

    fail_rate = len(fails) / max(len(df), 1)
    max_rate = float(cfg["failure_limits"]["max_failure_rate"])
    log.info(f"failure rate: {fail_rate:.3f}  (max allowed {max_rate:.3f})")

    import sklearn  # noqa
    manifest = {
        "spec_version": SPEC_VERSION,
        "partition_method_version": SCRIPT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "config": cfg,
        "config_path": str(args.config),
        "inputs": {"seed_csv": str(seed_csv)},
        "pool_stats": {
            "n_reactions": int(len(df)),
            "methods": dict(method_counts),
            "statuses": dict(status_counts),
            "failures_by_reason": dict(fail_counts),
            "failure_rate": float(fail_rate),
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "rdkit": Chem.rdBase.rdkitVersion,
        },
        "output_hash": hashlib.sha256(
            (out_dir / "fragments.parquet").read_bytes()
        ).hexdigest(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    log.info(f"wrote manifest.json + fragments.parquet")

    if fail_rate > max_rate:
        log.error(f"failure rate {fail_rate:.3f} exceeds threshold {max_rate:.3f}")
        sys.exit(EXIT_FAILURES)
    log.info("SUCCESS")


if __name__ == "__main__":
    main()
