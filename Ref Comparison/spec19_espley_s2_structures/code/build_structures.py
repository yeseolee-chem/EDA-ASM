"""spec19 Stage 2 — assemble the 5 DIAS structures per reaction.

Reads:
  logs/discovery.json  (per-reaction path table + charge/mult)

Writes:
  structures/rxn_{NNNN}/{r_A.xyz, r_B.xyz, ts.xyz, d_A.xyz, d_B.xyz}
  results/manifest.pkl

Fragment A/B convention (USER-MANDATED — INHERITED, NEVER RE-DERIVED):
  Fragment A = ORCA EDA `(1)` atoms = strain_A_kcal   = dict key `{rxn_number}_1`
  Fragment B = ORCA EDA `(2)` atoms = strain_B_kcal   = dict key `{rxn_number}_2`

Per-sub-source r_A/r_B provenance:
  spec16:     opt.xyz (isolated fragment optimized at BLYP-D3BJ/def2-TZVP)
  locked_778: R.xyz atom subset (embedded in strain_sp_cp/{rid}/fragA_R.inp);
              NOT independently optimized — the label pipeline used this
              geometry to compute strain_A_kcal, so we inherit it.
"""

from __future__ import annotations

import json
import platform
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
STAGE = REPO / "Ref Comparison/spec19_espley_s2_structures"
DISCOVERY_JSON = STAGE / "logs/discovery.json"
STRUCT_ROOT = STAGE / "structures"
MANIFEST = STAGE / "results/manifest.pkl"
BUILD_LOG = STAGE / "logs/build.log"

COORD_MATCH_TOL = 1e-3   # Å; eda.inp and TS.xyz coords are stored to 5-8 decimals
XYZ_COMMENT_PREFIX = "spec19_s2"


def _log(fh, msg: str) -> None:
    print(msg)
    fh.write(msg + "\n")
    fh.flush()


def read_xyz(path: Path) -> tuple[list[str], np.ndarray, str]:
    lines = path.read_text().splitlines()
    n = int(lines[0].strip())
    comment = lines[1] if len(lines) > 1 else ""
    elems, coords = [], []
    for ln in lines[2 : 2 + n]:
        parts = ln.split()
        elems.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return elems, np.array(coords, dtype=np.float64), comment


def write_xyz(path: Path, elems: list[str], coords: np.ndarray, comment: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(elems)
    assert coords.shape == (n, 3)
    with open(path, "w") as f:
        f.write(f"{n}\n{comment}\n")
        for e, xyz in zip(elems, coords):
            f.write(f"{e:2s} {xyz[0]:>16.8f} {xyz[1]:>16.8f} {xyz[2]:>16.8f}\n")


# ---------------------------------------------------------------------------
# eda.inp parsing — element labels of the form `C(1)`, `N(2)`, etc.
# ---------------------------------------------------------------------------
EDA_ATOM_RE = re.compile(
    r"^\s*([A-Z][a-z]?)\s*\(([12])\)\s+([-+.\deE]+)\s+([-+.\deE]+)\s+([-+.\deE]+)\s*$"
)
EDA_XYZ_START = re.compile(r"^\s*\*\s*xyz\s+([-+]?\d+)\s+(\d+)\s*$", re.IGNORECASE)
EDA_XYZ_END = re.compile(r"^\s*\*\s*$")


def parse_eda_inp(path: Path) -> tuple[list[str], list[int], np.ndarray]:
    """Parse an ORCA EDA input file. Return (elements, frag_labels, coords).
    frag_labels[i] ∈ {1, 2} — the fragment membership assigned by the EDA
    pipeline. This is the USER'S SPLIT and must be inherited.
    """
    lines = path.read_text().splitlines()
    in_block = False
    elems, frags, coords = [], [], []
    for ln in lines:
        if not in_block:
            if EDA_XYZ_START.match(ln):
                in_block = True
            continue
        if EDA_XYZ_END.match(ln):
            break
        m = EDA_ATOM_RE.match(ln)
        if m:
            elems.append(m.group(1))
            frags.append(int(m.group(2)))
            coords.append([float(m.group(3)), float(m.group(4)), float(m.group(5))])
    if not elems:
        raise RuntimeError(f"no (1)/(2)-labeled atoms found in {path}")
    return elems, frags, np.array(coords, dtype=np.float64)


# ---------------------------------------------------------------------------
# Match eda.inp atoms (arbitrary order, (1)/(2) labels) back to TS.xyz order.
# Returns TS-order boolean masks for fragment A (1) and fragment B (2).
# ---------------------------------------------------------------------------
def eda_to_ts_masks(
    ts_elems: list[str], ts_coords: np.ndarray,
    eda_elems: list[str], eda_frags: list[int], eda_coords: np.ndarray,
    rid: str, fh,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(ts_elems)
    if len(eda_elems) != n:
        raise RuntimeError(f"[{rid}] eda.inp has {len(eda_elems)} atoms, TS.xyz has {n}")
    # For each TS atom, find a matching eda atom (same element + coord within tol).
    ts_frag = np.zeros(n, dtype=np.int8)
    matched = np.zeros(len(eda_elems), dtype=bool)
    for i in range(n):
        te, tc = ts_elems[i], ts_coords[i]
        best_j, best_d = -1, np.inf
        for j in range(len(eda_elems)):
            if matched[j] or eda_elems[j] != te:
                continue
            d = np.linalg.norm(eda_coords[j] - tc)
            if d < best_d:
                best_d, best_j = d, j
        if best_j < 0 or best_d > COORD_MATCH_TOL:
            raise RuntimeError(
                f"[{rid}] TS atom {i} ({te} {tc.tolist()}) has no eda.inp match "
                f"within {COORD_MATCH_TOL} Å (best d={best_d:.5f} Å)")
        matched[best_j] = True
        ts_frag[i] = eda_frags[best_j]
    return (ts_frag == 1), (ts_frag == 2)


# ---------------------------------------------------------------------------
# r_A / r_B geometry sources
# ---------------------------------------------------------------------------
def read_locked_frag_R(path: Path) -> tuple[list[str], np.ndarray]:
    """Parse the fragment geometry embedded in `strain_sp_cp/{rid}/frag{A,B}_R.inp`.
    Format: ORCA-style header + `* xyz C M` + atom lines + `*`.

    CP-corrected inputs contain REAL atoms (e.g. `C`) and GHOST atoms
    (e.g. `C:`) — the ghost atoms are the OTHER fragment's atom positions
    supplying basis functions for BSSE correction. We keep only the REAL
    atoms (no trailing `:`).
    """
    lines = path.read_text().splitlines()
    in_block = False
    elems, coords = [], []
    for ln in lines:
        if not in_block:
            if EDA_XYZ_START.match(ln) or re.match(r"^\s*\*\s*xyz\b", ln):
                in_block = True
            continue
        if EDA_XYZ_END.match(ln):
            break
        parts = ln.split()
        if len(parts) < 4:
            continue
        elem = parts[0]
        if elem.endswith(":"):
            # ghost atom (BSSE basis); belongs to the other fragment
            continue
        try:
            xyz = [float(parts[1]), float(parts[2]), float(parts[3])]
        except ValueError:
            continue
        elems.append(elem)
        coords.append(xyz)
    if not elems:
        raise RuntimeError(f"no real atoms parsed from {path}")
    return elems, np.array(coords, dtype=np.float64)


# ---------------------------------------------------------------------------
def build_reaction(rec: dict, fh) -> dict:
    rid = rec["reaction_id"]
    rn = rec["reaction_number"]
    sub = rec["sub_source"]
    paths = rec["paths"]
    charge_info = rec["charge_info"]

    # --- ts.xyz ---
    ts_elems, ts_coords, _ = read_xyz(Path(paths["ts_xyz"]["path"]))
    natoms_ts = len(ts_elems)

    # --- fragment split (INHERITED from eda.inp, USER-MANDATED) ---
    eda_e, eda_f, eda_c = parse_eda_inp(Path(paths["eda_inp"]["path"]))
    mask_A, mask_B = eda_to_ts_masks(ts_elems, ts_coords, eda_e, eda_f, eda_c, rid, fh)
    if mask_A.sum() + mask_B.sum() != natoms_ts:
        raise RuntimeError(f"[{rid}] mask_A + mask_B does not cover all TS atoms")

    ts_idx_A = np.where(mask_A)[0].tolist()
    ts_idx_B = np.where(mask_B)[0].tolist()

    # --- d_A / d_B (TS atom subsets) ---
    d_A_elems = [ts_elems[i] for i in ts_idx_A]
    d_A_coords = ts_coords[ts_idx_A]
    d_B_elems = [ts_elems[i] for i in ts_idx_B]
    d_B_coords = ts_coords[ts_idx_B]

    # --- r_A / r_B ---
    # Load both candidate files and check which matches the (1)/(2) split from
    # eda.inp by ELEMENT MULTISET. The source naming (`__fA`/`fragA_R`) is not
    # guaranteed to match the user-convention fragment A in all rows (verified
    # on dipolar_000658 et al.). Fragment A = eda `(1)` atoms is INVARIANT.
    if sub == "spec16":
        cand_A_elems, cand_A_coords, _ = read_xyz(Path(paths["r_A_source"]["path"]))
        cand_B_elems, cand_B_coords, _ = read_xyz(Path(paths["r_B_source"]["path"]))
        r_A_kind_base = "opt.xyz_isolated_fragment_opt"
    else:
        cand_A_elems, cand_A_coords = read_locked_frag_R(Path(paths["r_A_source"]["path"]))
        cand_B_elems, cand_B_coords = read_locked_frag_R(Path(paths["r_B_source"]["path"]))
        r_A_kind_base = "R.xyz_atom_subset_NOT_optimized"

    dA_ms = sorted(d_A_elems)
    dB_ms = sorted(d_B_elems)
    a_matches_A = (sorted(cand_A_elems) == dA_ms)
    a_matches_B = (sorted(cand_A_elems) == dB_ms)
    b_matches_A = (sorted(cand_B_elems) == dA_ms)
    b_matches_B = (sorted(cand_B_elems) == dB_ms)

    swap_needed = None
    if a_matches_A and b_matches_B:
        r_A_elems, r_A_coords = cand_A_elems, cand_A_coords
        r_B_elems, r_B_coords = cand_B_elems, cand_B_coords
        swap_needed = False
    elif a_matches_B and b_matches_A:
        # Source naming swaps A/B; assign per the user-mandated convention.
        r_A_elems, r_A_coords = cand_B_elems, cand_B_coords
        r_B_elems, r_B_coords = cand_A_elems, cand_A_coords
        swap_needed = True
    else:
        raise RuntimeError(
            f"[{rid}] neither r_A candidate matches d_A/d_B element multiset: "
            f"cand_A={sorted(cand_A_elems)} cand_B={sorted(cand_B_elems)} "
            f"d_A={dA_ms} d_B={dB_ms}")

    r_A_kind = r_A_kind_base + ("_AB_SWAPPED_vs_source_filename" if swap_needed else "")
    r_B_kind = r_A_kind_base + ("_AB_SWAPPED_vs_source_filename" if swap_needed else "")

    # --- charge / multiplicity ---
    if charge_info is None:
        raise RuntimeError(f"[{rid}] missing charge info")
    ca, cb = charge_info["fragment_charge_a"], charge_info["fragment_charge_b"]
    ma, mb = charge_info["fragment_mult_a"], charge_info["fragment_mult_b"]
    ct = charge_info["total_charge"]

    # --- write 5 files ---
    dirname = STRUCT_ROOT / f"rxn_{rn:04d}"
    prov = (f"{XYZ_COMMENT_PREFIX} rid={rid} rn={rn} sub={sub} "
            f"frag_split=inherited_from_EDA_NOCV(1)/(2)_in_eda.inp")
    write_xyz(dirname / "ts.xyz",
              ts_elems, ts_coords, f"{prov} kind=ts charge={ct} mult=1")
    write_xyz(dirname / "d_A.xyz",
              d_A_elems, d_A_coords, f"{prov} kind=d_A charge={ca} mult={ma}")
    write_xyz(dirname / "d_B.xyz",
              d_B_elems, d_B_coords, f"{prov} kind=d_B charge={cb} mult={mb}")
    write_xyz(dirname / "r_A.xyz",
              r_A_elems, r_A_coords, f"{prov} kind=r_A src={r_A_kind} charge={ca} mult={ma}")
    write_xyz(dirname / "r_B.xyz",
              r_B_elems, r_B_coords, f"{prov} kind=r_B src={r_B_kind} charge={cb} mult={mb}")

    return {
        "reaction_number": rn,
        "reaction_id": rid,
        "sub_source": sub,
        "dir": str(dirname),
        "natoms": {
            "ts":  natoms_ts,
            "r_A": len(r_A_elems), "r_B": len(r_B_elems),
            "d_A": len(d_A_elems), "d_B": len(d_B_elems),
        },
        "charge": {"total": ct, "A": ca, "B": cb},
        "mult":   {"A": ma, "B": mb},
        "ts_idx_A": ts_idx_A,   # 0-based indices into ts.xyz
        "ts_idx_B": ts_idx_B,
        "r_A_provenance": r_A_kind,
        "r_B_provenance": r_B_kind,
    }


def main() -> int:
    (STAGE / "results").mkdir(parents=True, exist_ok=True)
    (STAGE / "logs").mkdir(exist_ok=True)

    with open(BUILD_LOG, "a") as fh:
        _log(fh, "=== spec19 Stage 2 build_structures ===")
        _log(fh, f"[env] python={platform.python_version()} pandas={pd.__version__}")

        with open(DISCOVERY_JSON) as jf:
            disc = json.load(jf)
        records = disc["records"]
        _log(fh, f"[load] n_records={len(records)}")

        manifest_rows = []
        errors = []
        for rec in records:
            try:
                m = build_reaction(rec, fh)
                manifest_rows.append(m)
            except Exception as e:
                errors.append((rec["reaction_id"], str(e)))
                _log(fh, f"[error] {rec['reaction_id']}: {e}")

        if errors:
            _log(fh, f"[HALT] {len(errors)} reactions failed; not writing manifest")
            for rid, e in errors[:10]:
                _log(fh, f"       {rid}: {e}")
            raise RuntimeError(f"{len(errors)} build errors — see build.log")

        manifest_df = pd.DataFrame(manifest_rows).sort_values("reaction_number").reset_index(drop=True)
        tmp = MANIFEST.with_suffix(".pkl.tmp")
        manifest_df.to_pickle(tmp)
        tmp.replace(MANIFEST)
        _log(fh, f"[write] {MANIFEST}  n={len(manifest_df)} size={MANIFEST.stat().st_size} bytes")
        _log(fh, "=== build_structures OK ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
