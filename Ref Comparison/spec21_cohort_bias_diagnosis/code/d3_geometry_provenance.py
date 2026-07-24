"""spec21 D3 (G21-B) — TS geometry provenance vs. Stuyver's originals.

For each of 400: read our TS.xyz + Stuyver's TS_imag_mode.xyz for the
same source_id, resolve heavy-atom ordering, Kabsch-align, and record
heavy-atom RMSD.

Atom-order resolution:
  1. Try direct order (both files came from the same optimization →
     atoms are in the same order → RMSD will be tiny).
  2. If element sequence matches, direct order is OK.
  3. Otherwise: try Hungarian-style matching over element-preserving
     assignments (elements first, then coords). Above 5 Å RMSD from
     Hungarian, record as `unresolved` — do NOT fall back to positional
     comparison per §5.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
STAGE = REPO / "Ref Comparison/spec21_cohort_bias_diagnosis"
IN_JOINED = STAGE / "results/cohort_joined.parquet"
STUYVER_PROFILES = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/dipolar_cycloaddition/extracted/full_dataset_profiles")

OUT_CSV = STAGE / "results/D3_geometry_provenance.csv"
OUT_FIG = STAGE / "figures/D3_rmsd_by_half.png"
HALT_FLAG = STAGE / "logs/G21_B_HALT.flag"
PASS_FLAG = STAGE / "logs/G21_B_PASS.flag"
GATES_LOG = STAGE / "logs/gates.log"


def read_xyz(path: Path) -> tuple[list[str], np.ndarray]:
    lines = path.read_text().splitlines()
    n = int(lines[0].strip())
    elems, coords = [], []
    for ln in lines[2 : 2 + n]:
        parts = ln.split()
        elems.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return elems, np.array(coords, dtype=np.float64)


def keep_heavy(elems: list[str], coords: np.ndarray) -> tuple[list[str], np.ndarray]:
    idx = [i for i, e in enumerate(elems) if e.upper() != "H"]
    return [elems[i] for i in idx], coords[idx]


def kabsch_rmsd(A: np.ndarray, B: np.ndarray) -> float:
    Ac = A - A.mean(0)
    Bc = B - B.mean(0)
    H = Ac.T @ Bc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    Ac_rot = Ac @ R.T
    return float(np.sqrt(np.mean(np.sum((Ac_rot - Bc) ** 2, axis=1))))


def hungarian_reorder(a_elems, a_coords, b_elems, b_coords):
    """Element-preserving Hungarian matching in centred, PCA-aligned frame."""
    a_c = a_coords - a_coords.mean(0)
    b_c = b_coords - b_coords.mean(0)
    # cost = squared distance if element matches, else +inf
    n = a_c.shape[0]
    C = np.full((n, n), 1e9)
    for i in range(n):
        for j in range(n):
            if a_elems[i] == b_elems[j]:
                C[i, j] = np.sum((a_c[i] - b_c[j]) ** 2)
    r, c = linear_sum_assignment(C)
    if np.any(C[r, c] > 1e8):
        return None, None
    return b_elems, b_coords[c]


def compare(rid: str, sub: str, our_ts: Path, stuyver_ts: Path) -> dict:
    if not stuyver_ts.exists():
        return {"reaction_id": rid, "sub_source": sub, "n_heavy": None,
                "rmsd_ang": None, "verdict": "stuyver_TS_missing"}
    a_e, a_x = read_xyz(our_ts)
    b_e, b_x = read_xyz(stuyver_ts)
    a_eh, a_xh = keep_heavy(a_e, a_x)
    b_eh, b_xh = keep_heavy(b_e, b_x)
    if len(a_eh) != len(b_eh) or sorted(a_eh) != sorted(b_eh):
        return {"reaction_id": rid, "sub_source": sub, "n_heavy": len(a_eh),
                "rmsd_ang": None, "verdict": "atom_count_or_element_mismatch"}
    # Direct order first
    if a_eh == b_eh:
        rmsd = kabsch_rmsd(a_xh, b_xh)
    else:
        # element sequences differ → Hungarian
        _, b_xh_re = hungarian_reorder(a_eh, a_xh, b_eh, b_xh)
        if b_xh_re is None:
            return {"reaction_id": rid, "sub_source": sub, "n_heavy": len(a_eh),
                    "rmsd_ang": None, "verdict": "hungarian_infeasible"}
        rmsd = kabsch_rmsd(a_xh, b_xh_re)
        if rmsd > 5.0:
            return {"reaction_id": rid, "sub_source": sub, "n_heavy": len(a_eh),
                    "rmsd_ang": rmsd, "verdict": "unresolved"}
    if rmsd < 0.01:
        v = "identical_lineage"
    elif rmsd < 0.05:
        v = "same_structure_precision_diff"
    else:
        v = "reoptimized_or_different"
    return {"reaction_id": rid, "sub_source": sub, "n_heavy": len(a_eh),
            "rmsd_ang": rmsd, "verdict": v}


def main() -> int:
    STAGE.joinpath("logs").mkdir(exist_ok=True)
    STAGE.joinpath("figures").mkdir(exist_ok=True)
    joined = pd.read_parquet(IN_JOINED)

    rows = []
    for _, row in joined.iterrows():
        rid = row["reaction_id"]
        sub = row["sub_source"]
        sid = int(row["source_id"])
        our_ts = Path(row["ts_xyz_path"])
        stuyver_ts = STUYVER_PROFILES / str(sid) / "TS_imag_mode.xyz"
        rec = compare(rid, sub, our_ts, stuyver_ts)
        rec["source_id"] = sid
        rec["stuyver_path"] = str(stuyver_ts)
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"[write] {OUT_CSV}")

    verdict_counts = df.groupby(["sub_source", "verdict"]).size().unstack(fill_value=0)
    print("=== verdict counts ===")
    print(verdict_counts.to_string())

    # G21-B: halves must land in the same regime.
    # Compare "regime" per half using the modal verdict.
    modal = {}
    for sub in ("locked_778", "spec16"):
        sub_df = df[df["sub_source"] == sub]
        counts = sub_df["verdict"].value_counts()
        modal[sub] = counts.idxmax() if len(counts) else "empty"
    with open(GATES_LOG, "a") as gf:
        gf.write(f"[G21-B] locked_778 modal verdict = {modal.get('locked_778')} "
                 f"({int((df[df.sub_source=='locked_778']['verdict']==modal.get('locked_778')).sum())}/"
                 f"{int((df.sub_source=='locked_778').sum())})\n")
        gf.write(f"[G21-B] spec16     modal verdict = {modal.get('spec16')} "
                 f"({int((df[df.sub_source=='spec16']['verdict']==modal.get('spec16')).sum())}/"
                 f"{int((df.sub_source=='spec16').sum())})\n")
        homogeneous_regimes = {
            "identical_lineage", "same_structure_precision_diff",
        }
        both_in_same = (
            modal.get("locked_778") == modal.get("spec16")
            or (modal.get("locked_778") in homogeneous_regimes
                and modal.get("spec16") in homogeneous_regimes)
        )
        if both_in_same:
            PASS_FLAG.write_text(f"G21-B PASS: locked={modal.get('locked_778')}, spec16={modal.get('spec16')}\n")
            gf.write("[G21-B PASS] halves in same regime — geometry-homogeneous\n")
        else:
            HALT_FLAG.write_text(
                f"G21-B HALT: locked_778 modal='{modal.get('locked_778')}', "
                f"spec16 modal='{modal.get('spec16')}'. Halves sit on different regimes; "
                f"re-labelling all 400 under one protocol will not unify them.\n"
            )
            gf.write(f"[G21-B HALT] {modal}\n")

    # Histogram by half
    fig, ax = plt.subplots(figsize=(8, 4.6))
    vals = df["rmsd_ang"].dropna().values
    if len(vals):
        bins = np.logspace(np.log10(max(1e-4, vals.min())),
                            np.log10(max(vals.max(), 1e-2)), 40)
        for sub in ("locked_778", "spec16"):
            v = df.loc[(df["sub_source"] == sub) & df["rmsd_ang"].notna(), "rmsd_ang"].values
            ax.hist(v, bins=bins, alpha=0.6, label=f"{sub} (n={len(v)})",
                    edgecolor="black", linewidth=0.4)
        ax.set_xscale("log")
        ax.set_xlabel("heavy-atom Kabsch RMSD to Stuyver TS  [Å]")
        ax.set_ylabel("count")
        for v, name in [(0.01, "identical<0.01"),
                        (0.05, "precision<0.05")]:
            ax.axvline(v, color="k", linewidth=0.6, linestyle="--")
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=140)
    print(f"[write] {OUT_FIG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
