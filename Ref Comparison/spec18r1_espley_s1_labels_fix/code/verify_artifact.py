"""spec18r1 Stage 1 verify — G-A .. G-G on the RELOADED artifact.

Every gate reads the pickle from disk (fixes F2). Numeric checks
re-derive from the source parquet's four channels (fixes F3–F5).
The Rev 0 tautological checks are kept as *regression guards* and
explicitly labelled as such.

Runnable standalone:
  python verify_artifact.py [path/to/artifact.pkl]
Default path = the stage's canonical pickle.
"""

from __future__ import annotations

import hashlib
import platform
import re
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
SRC_PARQUET = REPO / "outputs/spec16_orca/labels/dipolar_400_merged.parquet"
STAGE = REPO / "Ref Comparison/spec18r1_espley_s1_labels_fix"
DEFAULT_PKL = STAGE / "results/labels_2ch_400dipolar.pkl"
INSPECTION_PARQUET = STAGE / "results/labels_2ch_400dipolar.INSPECTION_ONLY.parquet"
GATES_LOG = STAGE / "logs/gates.log"
ANOMALY_CSV = STAGE / "results/anomaly_triage.csv"
RESID_OUTLIERS_CSV = STAGE / "results/asm_residual_outliers.csv"
RESID_HIST = STAGE / "figures/asm_residual_hist.png"

REF_CLONE = Path("/gpfs/tmp_cpu2/yeseo1ee/ext/distortion-interaction_ML")

COHORT_N = 400
CONTRIBS_TOL = 5e-3
KEY_RE = re.compile(r"^\d+_[12]$")


class Gate:
    def __init__(self, fh):
        self.fh = fh
        self.fails: list[str] = []
        self.warnings: list[str] = []

    def _emit(self, msg: str) -> None:
        print(msg)
        self.fh.write(msg + "\n")
        self.fh.flush()

    def pass_(self, name: str, detail: str = "") -> None:
        self._emit(f"[{name} PASS] {detail}")

    def warn(self, name: str, detail: str) -> None:
        self.warnings.append(f"{name}: {detail}")
        self._emit(f"[{name} WARN] {detail}")

    def fail(self, name: str, detail: str) -> None:
        self.fails.append(f"{name}: {detail}")
        self._emit(f"[{name} FAIL] {detail}")

    def info(self, name: str, detail: str) -> None:
        self._emit(f"[{name} INFO] {detail}")


# -----------------------------------------------------------------------------
# G-A — round-trip fidelity of the pickle
# -----------------------------------------------------------------------------
def gate_A(gate: Gate, df: pd.DataFrame) -> None:
    if len(df) != COHORT_N:
        gate.fail("G-A", f"len(df)={len(df)} expected {COHORT_N}")
        return

    dc = df["distortion_contributions_dft"].tolist()
    rns = df["reaction_number"].tolist()
    sd = df["sum_distortion_energies_dft"].tolist()

    bad_shape, bad_none, bad_keyfmt, bad_prefix = 0, 0, 0, 0
    max_sum_diff = 0.0
    for rn, d, sdv in zip(rns, dc, sd):
        if not isinstance(d, dict) or len(d) != 2:
            bad_shape += 1
            continue
        if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in d.values()):
            bad_none += 1
        for k in d:
            if not KEY_RE.match(str(k)):
                bad_keyfmt += 1
                continue
            if int(str(k).split("_")[0]) != int(rn):
                bad_prefix += 1
        max_sum_diff = max(max_sum_diff, abs(sum(d.values()) - float(sdv)))

    ok = (bad_shape == 0 and bad_none == 0 and bad_keyfmt == 0
          and bad_prefix == 0 and max_sum_diff < CONTRIBS_TOL)
    detail = (f"n=400, dict-shape errors={bad_shape}, None values={bad_none}, "
              f"key-format errors={bad_keyfmt}, prefix mismatches={bad_prefix}, "
              f"max|sum(dict) − sum_distortion|={max_sum_diff:.3e} "
              f"(tol {CONTRIBS_TOL:.0e})")
    if ok:
        gate.pass_("G-A", detail)
    else:
        gate.fail("G-A", detail)

    # dtypes
    dt = {c: str(df[c].dtype) for c in
          ["reaction_number", "e_barrier_dft", "sum_distortion_energies_dft",
           "interaction_energies_dft", "distortion_contributions_dft"]}
    dtype_ok = (dt["reaction_number"] == "int32"
                and dt["e_barrier_dft"] == "float64"
                and dt["sum_distortion_energies_dft"] == "float64"
                and dt["interaction_energies_dft"] == "float64"
                and dt["distortion_contributions_dft"] == "object")
    if dtype_ok:
        gate.pass_("G-A-dtypes", f"dtypes={dt}")
    else:
        gate.fail("G-A-dtypes", f"unexpected dtypes: {dt}")


# -----------------------------------------------------------------------------
# G-B — file sanity
# -----------------------------------------------------------------------------
def gate_B(gate: Gate, pkl_path: Path) -> None:
    ok = (pkl_path.suffix == ".pkl") and pkl_path.exists()
    gate.info("G-B", f"artifact of record: {pkl_path.name} size={pkl_path.stat().st_size} bytes")
    if not ok:
        gate.fail("G-B", f"artifact suffix / existence: suffix={pkl_path.suffix} exists={pkl_path.exists()}")
        return
    gate.pass_("G-B-artifact", f"suffix '.pkl' and file exists")

    if INSPECTION_PARQUET.exists():
        insp = pd.read_parquet(INSPECTION_PARQUET)
        if "distortion_contributions_dft" in insp.columns:
            gate.fail("G-B-inspection", "inspection parquet contains dict column")
        else:
            gate.pass_("G-B-inspection", f"inspection parquet dict-column-free, size={INSPECTION_PARQUET.stat().st_size} bytes")


# -----------------------------------------------------------------------------
# G-C — independent interaction check (fixes F3, F4)
# -----------------------------------------------------------------------------
def gate_C(gate: Gate, df: pd.DataFrame, src: pd.DataFrame) -> None:
    # Independent path: re-sum the four channels FROM THE SOURCE parquet.
    src_sorted = src.sort_values("reaction_id", kind="mergesort").reset_index(drop=True)
    resum = (src_sorted["pauli_kcal"] + src_sorted["elst_kcal"]
             + src_sorted["orb_kcal"] + src_sorted["disp_kcal"]).values

    # Align via reaction_id — the artifact is already sorted, so index parity is fine,
    # but do it defensively.
    df_sorted = df.sort_values("reaction_id", kind="mergesort").reset_index(drop=True)
    itn = df_sorted["interaction_energies_dft"].values

    # sign flip is applied ⇒ itn + resum ≈ 0 (both up to source rounding).
    diff = itn + resum
    absd = np.abs(diff)
    stats = {
        "max": float(absd.max()),
        "mean": float(absd.mean()),
        "p99": float(np.quantile(absd, 0.99)),
    }
    detail = (f"|interaction_dft + resum| max={stats['max']:.3e} "
              f"mean={stats['mean']:.3e} p99={stats['p99']:.3e}")
    if stats["max"] < 0.05:
        gate.pass_("G-C", detail)
    else:
        gate.fail("G-C", detail)


# -----------------------------------------------------------------------------
# G-D — ASM identity residual on the four-channel sum (fixes F5)
# -----------------------------------------------------------------------------
def gate_D(gate: Gate, df: pd.DataFrame, src: pd.DataFrame) -> None:
    src_sorted = src.sort_values("reaction_id", kind="mergesort").reset_index(drop=True)
    resum = (src_sorted["pauli_kcal"] + src_sorted["elst_kcal"]
             + src_sorted["orb_kcal"] + src_sorted["disp_kcal"]).values
    strain = src_sorted["strain_kcal"].values
    act = src_sorted["act_kcal"].values
    resid = np.abs(strain + resum - act)

    stats = {
        "n": int(resid.size),
        "min": float(resid.min()),
        "median": float(np.median(resid)),
        "mean": float(resid.mean()),
        "p95": float(np.quantile(resid, 0.95)),
        "p99": float(np.quantile(resid, 0.99)),
        "max": float(resid.max()),
    }
    gate.info("G-D-distribution", ", ".join(f"{k}={v:.4f}" for k, v in stats.items()))

    # per sub_source
    for sub, mask in df_masks_by_sub(df.sort_values("reaction_id", kind="mergesort").reset_index(drop=True)).items():
        r = resid[mask]
        s = {
            "n": int(r.size),
            "median": float(np.median(r)),
            "mean": float(r.mean()),
            "p95": float(np.quantile(r, 0.95)),
            "max": float(r.max()),
        }
        gate.info(f"G-D-{sub}", ", ".join(f"{k}={v:.4f}" for k, v in s.items()))

    # outlier list, always written with a schema (empty file still has a header)
    df_sorted = df.sort_values("reaction_id", kind="mergesort").reset_index(drop=True)
    outlier_cols = ["reaction_id", "sub_source", "reaction_number",
                    "strain_kcal", "resum_channels", "act_kcal", "residual_kcal"]
    out_rows = []
    for i, r in enumerate(resid):
        if r > 0.1:
            out_rows.append({
                "reaction_id":       df_sorted.at[i, "reaction_id"],
                "sub_source":        df_sorted.at[i, "sub_source"],
                "reaction_number":   int(df_sorted.at[i, "reaction_number"]),
                "strain_kcal":       float(strain[i]),
                "resum_channels":    float(resum[i]),
                "act_kcal":          float(act[i]),
                "residual_kcal":     float(r),
            })
    pd.DataFrame(out_rows, columns=outlier_cols).to_csv(RESID_OUTLIERS_CSV, index=False)
    gate.info("G-D-outliers", f"n_outliers_above_0.1kcal={len(out_rows)} written to {RESID_OUTLIERS_CSV.name}")

    # histogram
    STAGE.joinpath("figures").mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.logspace(-6, 2, 40) if resid.max() > 0 else np.linspace(0, 1, 40)
    # avoid log(0): floor at machine epsilon for display
    display = np.where(resid > 0, resid, 1e-8)
    ax.hist(display, bins=bins, color="#3b7dbf", edgecolor="black", linewidth=0.4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("|strain + resum(channels) − act_kcal|  [kcal/mol]")
    ax.set_ylabel("count (log)")
    ax.axvline(0.05, color="#666", linewidth=1.0, linestyle="--", label="G-D silent floor 0.05")
    ax.axvline(0.1, color="#e07b00", linewidth=1.0, linestyle="--", label="G-D escalation 0.1")
    ax.axvline(1.0, color="#c00", linewidth=1.0, linestyle="--", label="G-D HALT 1.0")
    ax.axvline(33.88, color="#000", linewidth=0.6, linestyle=":", label="SPEC_10 CONTAM scale 33.88")
    ax.legend(fontsize=8)
    ax.set_title("G-D ASM residual (kcal/mol) on the 4-channel sum, n=400")
    fig.tight_layout()
    fig.savefig(RESID_HIST, dpi=140)

    # gate semantics
    if stats["max"] > 1.0:
        gate.fail("G-D", f"max residual {stats['max']:.4f} > 1.0 kcal/mol — HALT (label not a barrier decomposition)")
    elif stats["max"] < 0.05:
        gate.pass_("G-D", f"max residual {stats['max']:.4f} < 0.05 — source rounding floor confirmed")
    else:
        gate.warn("G-D", f"max residual {stats['max']:.4f} in (0.05, 1.0] — {len(out_rows)} rows above 0.1; see outlier CSV + summary.md")


def df_masks_by_sub(df: pd.DataFrame) -> dict:
    return {sub: (df["sub_source"] == sub).values for sub in sorted(df["sub_source"].unique())}


# -----------------------------------------------------------------------------
# G-E — end-to-end contract test using their own code
# -----------------------------------------------------------------------------
def gate_E(gate: Gate, df: pd.DataFrame) -> None:
    fs_path = REF_CLONE / "feature_selection"
    if not (fs_path / "f_select.py").exists():
        gate.warn("G-E", f"reference clone missing at {REF_CLONE} — G-E DEGRADED (import skipped)")
        return
    sys.path.insert(0, str(fs_path))
    # f_select.py imports `inquirer` at top-level for interactive CLI prompts.
    # _clean_dist_contr and _manual_runner don't use it — stub the module so the
    # top-level import succeeds without adding a real dep to the reactot env.
    import types
    for mod_name in ("inquirer",):
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)
    try:
        from f_select import General, Manual  # type: ignore  # noqa: E402
    except Exception as e:
        gate.warn("G-E", f"failed to import f_select ({e}) — G-E DEGRADED")
        return

    # 1. dict expansion — the check Rev 0 skipped and would have failed on the parquet artifact
    try:
        expanded = General._clean_dist_contr(df.copy(deep=True))
    except Exception as e:
        gate.fail("G-E-clean", f"_clean_dist_contr raised {type(e).__name__}: {e}")
        return

    if "reaction_number" not in expanded.columns:
        gate.fail("G-E-clean", "expanded frame missing reaction_number")
        return

    rns = sorted(expanded["reaction_number"].tolist())
    if rns != list(range(COHORT_N)):
        # exhibit the pathology explicitly (Rev 0 would show [399]*400)
        unique = pd.Series(rns).value_counts().head(3).to_dict()
        gate.fail("G-E-clean", f"reaction_number after expansion != 0..399 — head counts {unique}")
        return
    gate.pass_("G-E-clean-rxnum", f"reaction_number == 0..399 on expanded frame")

    # The Rev 1 spec §3 G-E asked for columns '1' and '2'. The actual code renames
    # these to 'distortion_energy_1_dft' / 'distortion_energy_2_dft' (f_select.py
    # line 113). Assert against the actual names.
    e1, e2 = "distortion_energy_1_dft", "distortion_energy_2_dft"
    if not (e1 in expanded.columns and e2 in expanded.columns):
        gate.fail("G-E-clean", f"expected columns {e1!r}, {e2!r} not both present; have {list(expanded.columns)}")
        return
    v1, v2 = expanded[e1].values, expanded[e2].values
    if pd.isna(v1).any() or pd.isna(v2).any():
        n_nan = int(pd.isna(v1).sum() + pd.isna(v2).sum())
        gate.fail("G-E-clean", f"NaN in expanded strain columns ({n_nan} total)")
        return
    # Sum must reproduce sum_distortion up to source rounding.
    # Note: _clean_dist_contr merges on reaction_number, so align on that.
    merged = expanded[["reaction_number", e1, e2]].merge(
        df[["reaction_number", "sum_distortion_energies_dft"]], on="reaction_number")
    sum_err = float(np.max(np.abs((merged[e1] + merged[e2]).values
                                   - merged["sum_distortion_energies_dft"].values)))
    if sum_err < CONTRIBS_TOL:
        gate.pass_("G-E-clean-sum",
                   f"({e1}+{e2}) matches sum_distortion within {CONTRIBS_TOL:.0e} (max diff {sum_err:.3e})")
    else:
        gate.fail("G-E-clean-sum", f"per-row sum mismatch max {sum_err:.3e}")

    # 2. selection contract — inject dummy SQM cols to exercise Manual._manual_runner
    probe = df.copy()
    probe["e_barrier_gfn2"] = 0.0
    probe["interaction_energies_gfn2"] = 0.0
    try:
        kept = Manual._manual_runner(probe)
    except Exception as e:
        gate.fail("G-E-runner", f"_manual_runner raised {type(e).__name__}: {e}")
        return
    kept_cols = set(kept.columns)
    survives = {"sum_distortion_energies_dft", "interaction_energies_dft", "e_barrier_dft", "reaction_number"}
    missing = survives - kept_cols
    if missing:
        gate.fail("G-E-runner", f"dft targets dropped by _manual_runner: {missing}")
    else:
        gate.pass_("G-E-runner", f"all dft targets survive _manual_runner ({len(kept_cols)} cols kept)")
    # Deviation #6 tripwire
    dropped_gfn2 = [c for c in ["e_barrier_gfn2", "interaction_energies_gfn2"] if c not in kept_cols]
    if dropped_gfn2:
        gate.info("G-E-deviation6",
                  f"_gfn2 columns dropped as predicted ({dropped_gfn2}) — "
                  "Deviation #6 (f_select.py line 226 patch) confirmed necessary at Stage 3")


# -----------------------------------------------------------------------------
# G-F — reaction_number stability under re-sort
# -----------------------------------------------------------------------------
def gate_F(gate: Gate, df: pd.DataFrame) -> None:
    resorted = df.sort_values("reaction_id", kind="mergesort").reset_index(drop=True)
    if list(resorted["reaction_number"].values) == list(range(COHORT_N)):
        gate.pass_("G-F-sort", "reaction_number == 0..399 under sort by reaction_id")
    else:
        first_bad = next(i for i, v in enumerate(resorted["reaction_number"].values) if int(v) != i)
        gate.fail("G-F-sort",
                  f"reaction_number mismatch at first index {first_bad} "
                  f"(got {int(resorted.at[first_bad, 'reaction_number'])})")

    # Byte-identical rebuild check: rebuild once, sha-compare pickle bytes.
    # sbatch submit_s1.sh runs `build → verify`, so verify's disk pickle is
    # already the fresh build. We rebuild once more here and hash-compare.
    def _sha(p: Path) -> str:
        h = hashlib.sha256()
        with open(p, "rb") as bfh:
            for chunk in iter(lambda: bfh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    try:
        sha_before = _sha(DEFAULT_PKL)
        result = subprocess.run(
            [sys.executable, str(STAGE / "code/build_2ch_labels.py")],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            gate.warn("G-F-rebuild", f"rebuild exited {result.returncode}; stderr tail: {result.stderr[-200:]}")
            return
        sha_after = _sha(DEFAULT_PKL)
    except Exception as e:
        gate.warn("G-F-rebuild", f"rebuild probe failed to run: {e}")
        return
    if sha_before == sha_after:
        gate.pass_("G-F-rebuild", f"pickle sha256 stable across rebuilds: {sha_before[:16]}…")
    else:
        gate.warn("G-F-rebuild",
                  f"pickle bytes drifted across rebuilds: {sha_before[:16]}… vs {sha_after[:16]}… "
                  "(soft check — frame content stability via G-F-sort is the primary guarantee)")


# -----------------------------------------------------------------------------
# G-G — anomaly triage (report only, never exclude)
# -----------------------------------------------------------------------------
def gate_G(gate: Gate, df: pd.DataFrame, src: pd.DataFrame) -> None:
    d = df.sort_values("reaction_id", kind="mergesort").reset_index(drop=True)
    src_sorted = src.sort_values("reaction_id", kind="mergesort").reset_index(drop=True)
    resum = (src_sorted["pauli_kcal"] + src_sorted["elst_kcal"]
             + src_sorted["orb_kcal"] + src_sorted["disp_kcal"]).values
    strain = src_sorted["strain_kcal"].values
    act = src_sorted["act_kcal"].values
    resid = np.abs(strain + resum - act)

    flags = {
        "interaction_lt_0": (d["interaction_energies_dft"] < 0).values,
        "e_barrier_lt_0":   (d["e_barrier_dft"] < 0).values,
        "e_barrier_lt_neg20": (d["e_barrier_dft"] < -20).values,
        "sum_distortion_gt_100": (d["sum_distortion_energies_dft"] > 100).values,
    }

    rows = []
    for i in range(len(d)):
        row_flags = [name for name, mask in flags.items() if bool(mask[i])]
        if not row_flags:
            continue
        rows.append({
            "reaction_id": d.at[i, "reaction_id"],
            "sub_source":  d.at[i, "sub_source"],
            "reaction_number": int(d.at[i, "reaction_number"]),
            "flags":       "|".join(row_flags),
            "e_barrier_dft": float(d.at[i, "e_barrier_dft"]),
            "sum_distortion_energies_dft": float(d.at[i, "sum_distortion_energies_dft"]),
            "interaction_energies_dft": float(d.at[i, "interaction_energies_dft"]),
            "asm_residual_kcal": float(resid[i]),
        })
    tri = pd.DataFrame(rows)
    tri.to_csv(ANOMALY_CSV, index=False)

    # Per-flag counts + cross-tabs
    gate.info("G-G", f"flagged rows total: {len(tri)}  ({ANOMALY_CSV.name})")
    for name, mask in flags.items():
        n = int(mask.sum())
        by_sub = {sub: int(((d["sub_source"] == sub).values & mask).sum())
                  for sub in sorted(d["sub_source"].unique())}
        # correlation with large residual
        n_and_big_resid = int((mask & (resid > 0.05)).sum())
        gate.info(f"G-G-{name}", f"n={n} by sub_source={by_sub}  (with G-D residual>0.05: {n_and_big_resid})")


# -----------------------------------------------------------------------------
# Regression guards — kept from Rev 0 but explicitly labelled tautological
# -----------------------------------------------------------------------------
def regression_guards(gate: Gate, df: pd.DataFrame) -> None:
    n_sd_pos = int((df["sum_distortion_energies_dft"] > 0).sum())
    if n_sd_pos == len(df):
        gate.pass_("REG-sd-positive", "sum_distortion > 0 in all rows")
    else:
        gate.fail("REG-sd-positive", f"only {n_sd_pos}/{len(df)} rows with sum_distortion > 0")

    lhs = df["e_barrier_dft"].values
    rhs = (df["sum_distortion_energies_dft"] - df["interaction_energies_dft"]).values
    d3 = float(np.max(np.abs(lhs - rhs)))
    gate.info("REG-identity-tautology",
              f"max|e_barrier − (sum_dist − int)|={d3:.3e}  "
              "(guard, not evidence — algebraic identity by construction)")

    # sign check on the interaction (paper's convention is >0 for stabilising)
    n_int_pos = int((df["interaction_energies_dft"] > 0).sum())
    gate.info("REG-interaction-positive-fraction",
              f"interaction > 0 in {n_int_pos}/{len(df)} rows "
              f"({n_int_pos / len(df):.4f})  (physical, not tautological)")


# -----------------------------------------------------------------------------
def main(argv: list[str]) -> int:
    pkl_path = Path(argv[1]) if len(argv) > 1 else DEFAULT_PKL
    STAGE.joinpath("logs").mkdir(exist_ok=True)
    STAGE.joinpath("results").mkdir(exist_ok=True)
    STAGE.joinpath("figures").mkdir(exist_ok=True)

    with open(GATES_LOG, "w") as fh:
        gate = Gate(fh)
        gate._emit("=== spec18r1 verify ===")
        gate._emit(f"[env] python={platform.python_version()} pandas={pd.__version__} numpy={np.__version__}")
        gate._emit(f"[reload] {pkl_path}")

        df = pd.read_pickle(pkl_path)
        src = pd.read_parquet(SRC_PARQUET)
        gate._emit(f"[reload] artifact shape={df.shape}; source shape={src.shape}")

        gate_A(gate, df)
        gate_B(gate, pkl_path)
        gate_C(gate, df, src)
        gate_D(gate, df, src)
        gate_E(gate, df)
        gate_F(gate, df)
        gate_G(gate, df, src)
        regression_guards(gate, df)

        gate._emit("")
        gate._emit(f"=== SUMMARY: {len(gate.fails)} FAIL, {len(gate.warnings)} WARN ===")
        for f in gate.fails:
            gate._emit(f"  FAIL {f}")
        for w in gate.warnings:
            gate._emit(f"  WARN {w}")

        if gate.fails:
            raise RuntimeError(f"{len(gate.fails)} gate failures — see {GATES_LOG}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
