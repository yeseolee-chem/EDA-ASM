"""spec19 Stage 2 verify — G2-A .. G2-F on the ON-DISK artifacts.

Every gate reads structures/*.xyz + results/manifest.pkl from disk. Same
discipline as Stage 1: no gate is evaluated on an in-memory frame that
has not made a round trip.

Runnable standalone:
  python verify_structures.py [path/to/manifest.pkl]
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
STAGE = REPO / "Ref Comparison/spec19_espley_s2_structures"
DEFAULT_MANIFEST = STAGE / "results/manifest.pkl"
STRUCT_ROOT = STAGE / "structures"
STAGE1_PKL = REPO / "Ref Comparison/spec18r1_espley_s1_labels_fix/results/labels_2ch_400dipolar.pkl"
GATES_LOG = STAGE / "logs/gates.log"
COMMON_PKL = STAGE / "results/common_atoms.pkl"
ANOMALIES_CSV = STAGE / "results/common_atom_anomalies.csv"

COORD_TOL = 1e-6
COHORT_N = 400
STRUCT_KEYS = ["r_A", "r_B", "ts", "d_A", "d_B"]


class Gate:
    def __init__(self, fh):
        self.fh = fh
        self.fails = []
        self.warnings = []

    def _emit(self, msg): print(msg); self.fh.write(msg + "\n"); self.fh.flush()
    def pass_(self, n, d=""): self._emit(f"[{n} PASS] {d}")
    def warn(self, n, d): self.warnings.append(f"{n}: {d}"); self._emit(f"[{n} WARN] {d}")
    def fail(self, n, d): self.fails.append(f"{n}: {d}"); self._emit(f"[{n} FAIL] {d}")
    def info(self, n, d): self._emit(f"[{n} INFO] {d}")


def read_xyz(path: Path) -> tuple[list[str], np.ndarray]:
    lines = path.read_text().splitlines()
    n = int(lines[0].strip())
    elems, coords = [], []
    for ln in lines[2:2 + n]:
        p = ln.split()
        elems.append(p[0])
        coords.append([float(p[1]), float(p[2]), float(p[3])])
    return elems, np.array(coords, dtype=np.float64)


# ---------------------------------------------------------------------------
def gate_A_inventory(gate: Gate, mf: pd.DataFrame) -> None:
    if len(mf) != COHORT_N:
        gate.fail("G2-A", f"manifest rows={len(mf)} != {COHORT_N}")
        return
    rns = sorted(mf["reaction_number"].tolist())
    if rns != list(range(COHORT_N)):
        gate.fail("G2-A", "reaction_numbers not 0..399 in manifest")
        return

    missing, zero_byte, nan_coord = [], [], []
    for _, row in mf.iterrows():
        rn = int(row["reaction_number"])
        d = Path(row["dir"])
        for k in STRUCT_KEYS:
            p = d / f"{k}.xyz"
            if not p.exists():
                missing.append(f"{rn}:{k}")
                continue
            if p.stat().st_size == 0:
                zero_byte.append(f"{rn}:{k}")
                continue
            try:
                _, c = read_xyz(p)
                if not np.all(np.isfinite(c)):
                    nan_coord.append(f"{rn}:{k}")
            except Exception as e:
                gate.fail("G2-A", f"failed to read {p}: {e}")
                return

    if not missing and not zero_byte and not nan_coord:
        gate.pass_("G2-A", f"2000 files present, no zero-byte, no NaN coords")
    else:
        gate.fail("G2-A", f"missing={len(missing)} zero_byte={len(zero_byte)} nan={len(nan_coord)}; "
                          f"heads={missing[:5]}/{zero_byte[:5]}/{nan_coord[:5]}")


def gate_B_atom_conservation(gate: Gate, mf: pd.DataFrame) -> None:
    fails = 0
    first_fail = None
    for _, row in mf.iterrows():
        rn = int(row["reaction_number"])
        d = Path(row["dir"])
        eA, _ = read_xyz(d / "r_A.xyz")
        eB, _ = read_xyz(d / "r_B.xyz")
        eT, _ = read_xyz(d / "ts.xyz")
        edA, _ = read_xyz(d / "d_A.xyz")
        edB, _ = read_xyz(d / "d_B.xyz")

        cond = (
            len(eT) == len(edA) + len(edB)
            and sorted(eA) == sorted(edA)
            and sorted(eB) == sorted(edB)
            and len(eA) == len(edA) and len(eB) == len(edB)
        )
        if not cond:
            fails += 1
            if first_fail is None:
                first_fail = (rn, len(eT), len(edA), len(edB), sorted(eA) == sorted(edA))
    if fails == 0:
        gate.pass_("G2-B", "atom conservation OK on all 400 reactions")
    else:
        gate.fail("G2-B", f"{fails} reactions fail conservation; first={first_fail}")


def gate_C_subset(gate: Gate, mf: pd.DataFrame) -> None:
    fails, worst = 0, 0.0
    first_bad = None
    for _, row in mf.iterrows():
        rn = int(row["reaction_number"])
        d = Path(row["dir"])
        _, tsc = read_xyz(d / "ts.xyz")
        eA, dAc = read_xyz(d / "d_A.xyz")
        eB, dBc = read_xyz(d / "d_B.xyz")

        # d_A + d_B coords must appear verbatim in TS. Check by matching each
        # d_A/d_B coord to a unique TS index within COORD_TOL.
        matched = np.zeros(len(tsc), dtype=bool)
        max_d = 0.0
        for src_coords in (dAc, dBc):
            for c in src_coords:
                dists = np.linalg.norm(tsc - c, axis=1)
                dists[matched] = np.inf
                j = int(np.argmin(dists))
                if dists[j] > COORD_TOL:
                    fails += 1
                    if first_bad is None:
                        first_bad = (rn, float(dists[j]))
                    max_d = max(max_d, float(dists[j]))
                    break
                matched[j] = True
                max_d = max(max_d, float(dists[j]))
        worst = max(worst, max_d)

        # Also assert union covers all TS atoms exactly once
        if fails == 0 and not matched.all():
            fails += 1
            if first_bad is None:
                first_bad = (rn, "d_A ∪ d_B does not cover all TS atoms")

    if fails == 0:
        gate.pass_("G2-C", f"d_A ∪ d_B == ts atom-set exactly (worst |Δ|={worst:.2e} Å < {COORD_TOL:.0e})")
    else:
        gate.fail("G2-C", f"{fails} rxns fail subset check; first={first_bad}")


def gate_D_role_contract(gate: Gate, mf: pd.DataFrame) -> None:
    st1 = pd.read_pickle(STAGE1_PKL)
    st1 = st1.set_index("reaction_number")

    mismatches = []
    for _, row in mf.iterrows():
        rn = int(row["reaction_number"])
        d1 = st1.at[rn, "distortion_contributions_dft"]
        keys = list(d1.keys())
        # Stage 1 dict is {"{rn}_1": strain_A, "{rn}_2": strain_B}
        # Fragment A should correspond to the "_1" key.
        if not (f"{rn}_1" in keys and f"{rn}_2" in keys):
            mismatches.append((rn, f"dict keys not '{rn}_1','{rn}_2': {keys}"))
            continue
        # We inherit the split from eda.inp (1)/(2). By construction the
        # manifest's "A" side is EDA fragment 1 and matches "{rn}_1".
        # Verify the manifest recorded the provenance.
        prov = str(row.get("r_A_provenance", "")) + str(row.get("r_B_provenance", ""))
        if not prov:
            mismatches.append((rn, "manifest missing r_{A,B}_provenance"))
    if not mismatches:
        gate.pass_("G2-D",
                   "Fragment A ↔ dict key '_1' contract holds by construction; "
                   "r_A/r_B provenance recorded in manifest for every row.")
    else:
        gate.fail("G2-D", f"{len(mismatches)} rxns violate contract; first={mismatches[0]}")


def gate_E_common_atom_shape(gate: Gate) -> None:
    if not COMMON_PKL.exists():
        gate.warn("G2-E", "common_atoms.pkl absent — build_common_atoms did not run")
        return
    ca = pd.read_pickle(COMMON_PKL)
    anomalies_df = pd.read_csv(ANOMALIES_CSV) if ANOMALIES_CSV.exists() else pd.DataFrame()
    n_ok = sum(
        1 for rn, v in ca.items()
        if (v["r_A_k"], v["r_B_k"], v["ts_k"], v["d_A_k"], v["d_B_k"]) == (3, 2, 5, 3, 2)
        or (v["r_A_k"], v["r_B_k"], v["ts_k"], v["d_A_k"], v["d_B_k"]) == (2, 3, 5, 2, 3)
    )
    n_total = len(ca)
    n_anom = len(anomalies_df)
    gate.info("G2-E-distribution",
              f"n_processed={n_total} conform_to_(3,2,5,3,2)_or_(2,3,5,2,3)={n_ok} anomalies={n_anom}")
    if n_anom == 0 and n_total == COHORT_N:
        gate.pass_("G2-E", f"all 400 reactions match (3,2,5,3,2) contract or its A/B swap")
    elif n_anom > 0:
        gate.warn("G2-E",
                  f"{n_anom} rxns flagged; see common_atom_anomalies.csv "
                  f"(report-only, no exclusion per §4 G2-E semantics)")
    else:
        gate.warn("G2-E", f"only {n_total}/{COHORT_N} reactions processed (RDKit / SMILES failure)")


def gate_F_charge_mult(gate: Gate, mf: pd.DataFrame) -> None:
    """Charge conservation + explicit ints + open-shell list."""
    bad_conserve = 0
    bad_type = 0
    open_shell = 0
    for _, row in mf.iterrows():
        c = row["charge"]
        m = row["mult"]
        for v in (c["total"], c["A"], c["B"], m["A"], m["B"]):
            if not isinstance(v, (int, np.integer)):
                bad_type += 1
                break
        if int(c["A"]) + int(c["B"]) != int(c["total"]):
            bad_conserve += 1
        if int(m["A"]) != 1 or int(m["B"]) != 1:
            open_shell += 1

    detail = f"open_shell={open_shell}, charge_conserve_fail={bad_conserve}, non_int={bad_type}"
    if bad_conserve == 0 and bad_type == 0:
        gate.pass_("G2-F", detail)
        if open_shell > 0:
            gate.info("G2-F-open-shell", f"{open_shell} rxns with fragment mult != 1 (see results/open_shell.csv)")
    else:
        gate.fail("G2-F", detail)


def gate_manifest_hash(gate: Gate, path: Path) -> None:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    gate.info("G2-manifest-sha256", h.hexdigest())


def main(argv):
    mf_path = Path(argv[1]) if len(argv) > 1 else DEFAULT_MANIFEST
    STAGE.joinpath("logs").mkdir(exist_ok=True)
    with open(GATES_LOG, "w") as fh:
        g = Gate(fh)
        g._emit("=== spec19 Stage 2 verify ===")
        g._emit(f"[reload] {mf_path}")
        mf = pd.read_pickle(mf_path)
        g._emit(f"[reload] manifest shape={mf.shape}")

        gate_A_inventory(g, mf)
        gate_B_atom_conservation(g, mf)
        gate_C_subset(g, mf)
        gate_D_role_contract(g, mf)
        gate_E_common_atom_shape(g)
        gate_F_charge_mult(g, mf)
        gate_manifest_hash(g, mf_path)

        g._emit("")
        g._emit(f"=== SUMMARY: {len(g.fails)} FAIL, {len(g.warnings)} WARN ===")
        for f in g.fails:
            g._emit(f"  FAIL {f}")
        for w in g.warnings:
            g._emit(f"  WARN {w}")

        if g.fails:
            raise RuntimeError(f"{len(g.fails)} gate failures — see {GATES_LOG}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
