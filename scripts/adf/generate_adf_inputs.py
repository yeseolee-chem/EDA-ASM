#!/usr/bin/env python
"""Generate ADF C1-C5 input decks per ASR_ADF_Computation_Spec_v1.0.

Per reaction (`partition_status` ∈ {'ok', 'warning'}) under
`adf_outputs/batch_NNN/{reaction_id}/`:

  c1_fragA_ts.in       Fragment A single-point at TS-frozen geometry
  c2_fragB_ts.in       Fragment B single-point at TS-frozen geometry
  c3_eda.in            Supermolecule EDA fragment-analysis (atoms in A-then-B order)
  c4_fragA_opt.in      Fragment A geometry optimization (relaxed reference)  [skip if 1 atom]
  c5_fragB_opt.in      Fragment B geometry optimization (relaxed reference)  [skip if 1 atom]
  ts.xyz               TS geometry (full system, original ordering)
  geometry_fragA.xyz   fragment A xyz (TS-frozen positions, in A-fragment order)
  geometry_fragB.xyz   fragment B xyz (TS-frozen positions, in B-fragment order)
  run_reaction.sh      executable driver (5-job chain, NSCM=1, nice -n 10)

Also writes:
  adf_outputs/batch_NNN/                 (8 batches of 100 reactions each)
  adf_outputs/batch_manifest.json        provenance + batch inventory
  adf_outputs/submit_all.sh              master driver (bash-loops over batches)

Protocol (BLYP-D3(BJ)/TZ2P/Normal, NoSym, all-electron, no relativity, GGA so
dispersion is a separable EDA channel).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import ase.data
import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]

SPEC_VERSION = "1.0"
SCRIPT_VERSION = "generate_v1.0"


# --------------------------------- helpers ---------------------------------
def read_xyz(path: Path) -> tuple[list[str], np.ndarray]:
    lines = Path(path).read_text().strip().splitlines()
    n = int(lines[0].strip())
    elements: list[str] = []
    positions = np.zeros((n, 3), dtype=np.float64)
    for i, raw in enumerate(lines[2:2 + n]):
        toks = raw.split()
        elements.append(toks[0])
        positions[i] = [float(t) for t in toks[1:4]]
    return elements, positions


def write_xyz(path: Path, elements: list[str], positions: np.ndarray,
              comment: str = "") -> None:
    n = len(elements)
    out = [str(n), comment]
    for sym, (x, y, z) in zip(elements, positions):
        out.append(f"{sym:<3s} {x: .8f} {y: .8f} {z: .8f}")
    path.write_text("\n".join(out) + "\n")


# --------------------------------- input blocks ---------------------------------
def _engine_adf_block(cfg: dict, *, is_unrestricted: bool, spin_polarization: int,
                      fragments_block: str | None = None,
                      include_eda: bool = False, scf_iters: int = 200) -> str:
    prot = cfg["adf_protocol"]
    lines: list[str] = []
    lines.append("Engine ADF")
    lines.append("  Basis")
    lines.append(f"    Type {prot['basis']}")
    lines.append(f"    Core {prot['core']}")
    lines.append("  End")
    lines.append("  XC")
    lines.append(f"    GGA {prot['functional']}")
    lines.append(f"    Dispersion {prot['dispersion']}")
    lines.append("  End")
    lines.append(f"  Symmetry {prot['symmetry']}")
    lines.append(f"  NumericalQuality {prot['numerical_quality']}")
    lines.append(f"  Unrestricted {'Yes' if is_unrestricted else 'No'}")
    if spin_polarization > 0:
        lines.append(f"  SpinPolarization {spin_polarization}")
    lines.append("  SCF")
    lines.append(f"    Iterations {scf_iters}")
    lines.append("    Converge 1.0e-6")
    lines.append("  End")
    if fragments_block:
        lines.append("  " + fragments_block.replace("\n", "\n  "))
    if include_eda:
        # ETS-NOCV energy decomposition (matches AMS 2026 example syntax)
        lines.append("  etsnocv")
        lines.append("    enocv 0.05")
        lines.append("    ekmin 0.5")
        lines.append("  end")
        lines.append("  print etslowdin")
    lines.append("  Save TAPE21")
    lines.append("EndEngine")
    return "\n".join(lines)


def _atoms_block(elements: list[str], positions: np.ndarray,
                 indent: str = "    ") -> str:
    out = []
    for sym, (x, y, z) in zip(elements, positions):
        out.append(f"{indent}{sym:<3s} {x: .8f} {y: .8f} {z: .8f}")
    return "\n".join(out)


def make_c1c2_input(elements: list[str], positions: np.ndarray, charge: int,
                    multiplicity: int, cfg: dict, title: str = "") -> str:
    """Fragment single-point at TS-frozen geometry (C1 or C2)."""
    is_unr = multiplicity > 1
    spin_p = max(0, multiplicity - 1)
    engine = _engine_adf_block(cfg, is_unrestricted=is_unr,
                               spin_polarization=spin_p)
    atoms = _atoms_block(elements, positions, indent="    ")
    return (
        f"# {title}\n"
        "Task SinglePoint\n\n"
        "System\n"
        "  Atoms\n"
        f"{atoms}\n"
        "  End\n"
        f"  Charge {charge}\n"
        "End\n\n"
        f"{engine}\n"
    )


def make_c4c5_input(elements: list[str], positions: np.ndarray, charge: int,
                    multiplicity: int, cfg: dict, title: str = "") -> str:
    """Fragment geometry optimization (C4 or C5) — relaxed reference."""
    is_unr = multiplicity > 1
    spin_p = max(0, multiplicity - 1)
    engine = _engine_adf_block(cfg, is_unrestricted=is_unr,
                               spin_polarization=spin_p)
    atoms = _atoms_block(elements, positions, indent="    ")
    go = cfg["geometry"]["fragment_opt_convergence"]
    return (
        f"# {title}\n"
        "Task GeometryOptimization\n\n"
        "System\n"
        "  Atoms\n"
        f"{atoms}\n"
        "  End\n"
        f"  Charge {charge}\n"
        "End\n\n"
        f"{engine}\n\n"
        "GeometryOptimization\n"
        "  Convergence\n"
        f"    Energy {go['energy']}\n"
        f"    Gradients {go['gradients']}\n"
        "  End\n"
        f"  MaxIterations {go['max_iterations']}\n"
        "End\n"
    )


def make_c3_eda_input(elements: list[str], positions: np.ndarray,
                      n_a: int, n_b: int, total_charge: int,
                      total_mult: int, cfg: dict) -> str:
    """Supermolecule EDA fragment-analysis (C3).

    AMS 2026.103 syntax (matches Hplus_CO_etsnocv example):
      - atoms get `adf.f=fA` / `adf.f=fB` tags
      - `fragments` engine-block uses simple `label rkf_path` pairs
      - `etsnocv` block triggers the ETS-NOCV energy decomposition
    """
    is_unr = total_mult > 1
    spin_p = max(0, total_mult - 1)

    # Atom block with adf.f tags — first n_a → fA, next n_b → fB
    lines = []
    for i, (sym, (x, y, z)) in enumerate(zip(elements, positions)):
        tag = "fA" if i < n_a else "fB"
        lines.append(f"    {sym:<3s} {x: .8f} {y: .8f} {z: .8f}  adf.f={tag}")
    atoms = "\n".join(lines)

    fragments = (
        "fragments\n"
        "  fA c1_fragA_ts.results/adf.rkf\n"
        "  fB c2_fragB_ts.results/adf.rkf\n"
        "end"
    )
    engine = _engine_adf_block(cfg, is_unrestricted=is_unr,
                               spin_polarization=spin_p,
                               fragments_block=fragments, include_eda=True)
    return (
        "# C3: EDA supermolecule fragment-analysis (atom order: fragA then fragB)\n"
        "Task SinglePoint\n\n"
        "System\n"
        "  Atoms\n"
        f"{atoms}\n"
        "  End\n"
        f"  Charge {total_charge}\n"
        "End\n\n"
        f"{engine}\n"
    )


# --------------------------------- run_reaction.sh template ---------------------------------
RUN_REACTION_TEMPLATE = """#!/bin/bash
# ASR ADF run for reaction {rid}
# Per ASR_ADF_Computation_Spec_v1.0 §7.1 (single-job C1-C5 chain).
# C4/C5 backgrounded; C1, C2 sequential; C3 after C1+C2; wait for C4/C5.
#SBATCH --job-name=asr_{rid}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --output=slurm.log
#SBATCH --error=slurm.err

# NO `set -e` — we want all 5 calcs to attempt, then write_status to record
# what passed and what failed. Failures should not abort the chain.
set -o pipefail
source {ams_home}/amsbashrc.sh

cd "$(dirname "$0")"

export NSCM=${{SLURM_NTASKS:-1}}
ulimit -s unlimited 2>/dev/null || true

START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "=== {rid} START $START_UTC host=$(hostname) NSCM=$NSCM ==="

# C4 + C5: fragment geometry optimization (independent, backgrounded)
{c4_launch}
{c5_launch}

# C1, C2: fragment SP at TS geometry (sequential — share gate1 license slot)
AMS_JOBNAME=c1_fragA_ts "$AMSBIN/ams" <c1_fragA_ts.in > c1_fragA_ts.out 2>&1
C1_RC=$?
AMS_JOBNAME=c2_fragB_ts "$AMSBIN/ams" <c2_fragB_ts.in > c2_fragB_ts.out 2>&1
C2_RC=$?

# C3: EDA, depends on C1 + C2 rkfs
if [[ $C1_RC -eq 0 && $C2_RC -eq 0 ]]; then
    AMS_JOBNAME=c3_eda "$AMSBIN/ams" <c3_eda.in > c3_eda.out 2>&1
    C3_RC=$?
else
    echo "skipping C3 (C1 rc=$C1_RC, C2 rc=$C2_RC)" >&2
    C3_RC=2
fi

# Wait for the background optimizations
{c4_wait}
{c5_wait}

END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "=== {rid} END $END_UTC ==="

# Generate status.json with calc statuses derived from output parsing
python {repo}/scripts/adf/write_status.py \\
    --rid "{rid}" \\
    --rxn-dir "$(pwd)" \\
    --start "$START_UTC" \\
    --end "$END_UTC" \\
    --functional "{functional}" \\
    --basis "{basis}" \\
    --frag-method "{frag_method}" \\
    --atoms-a '{atoms_a_json}' \\
    --atoms-b '{atoms_b_json}' \\
    --charge-a {charge_a} \\
    --charge-b {charge_b} \\
    --mult-a {mult_a} \\
    --mult-b {mult_b} \\
    --total-charge {total_charge} \\
    --dataset-delta-Ea {dataset_dEa} \\
    --atom-permutation '{atom_perm_json}' \\
    --single-atom-a {single_atom_a} \\
    --single-atom-b {single_atom_b} \\
    || true   # write_status exit code is informational; don't abort the .sh

# Cleanup binary results dirs (kept .out and .in for audit + re-parse).
# Saves ~100MB per reaction; ~80GB across 794 reactions.
find . -maxdepth 1 -type d -name '*.results' -exec rm -rf {{}} + 2>/dev/null || true

echo "=== {rid} cleanup done ==="
"""


def render_c4_lines(single_atom_a: bool) -> tuple[str, str]:
    if single_atom_a:
        return ("# C4 skipped: fragA is a single atom (strain_a = 0 by definition)",
                "# (no C4 to wait on)")
    return (
        'AMS_JOBNAME=c4_fragA_opt "$AMSBIN/ams" <c4_fragA_opt.in > c4_fragA_opt.out 2>&1 &\n'
        "PID_C4=$!",
        "wait ${PID_C4:-1} 2>/dev/null || true",
    )


def render_c5_lines(single_atom_b: bool) -> tuple[str, str]:
    if single_atom_b:
        return ("# C5 skipped: fragB is a single atom (strain_b = 0)",
                "# (no C5 to wait on)")
    return (
        'AMS_JOBNAME=c5_fragB_opt "$AMSBIN/ams" <c5_fragB_opt.in > c5_fragB_opt.out 2>&1 &\n'
        "PID_C5=$!",
        "wait ${PID_C5:-1} 2>/dev/null || true",
    )


# --------------------------------- per-reaction builder ---------------------------------
def build_reaction(rxn_dir: Path, sel_row: pd.Series, frag_row: pd.Series,
                   cfg: dict) -> dict:
    rxn_dir.mkdir(parents=True, exist_ok=True)
    rid = sel_row["reaction_id"]

    ts_elements, ts_positions = read_xyz(Path(sel_row["path_ts"]))
    fragA_idx = json.loads(frag_row["fragment_atoms_a"])
    fragB_idx = json.loads(frag_row["fragment_atoms_b"])
    if not isinstance(fragA_idx, list) or not isinstance(fragB_idx, list):
        raise ValueError("fragment_atoms_{a,b} must be JSON arrays")
    n_a = len(fragA_idx)
    n_b = len(fragB_idx)
    if n_a + n_b != len(ts_elements):
        raise ValueError(
            f"fragA+fragB={n_a+n_b} atoms but TS has {len(ts_elements)}"
        )

    # Build A-then-B atom permutation (preserving order within each fragment)
    perm = list(fragA_idx) + list(fragB_idx)
    is_identity_perm = perm == list(range(len(perm)))
    permuted_elements = [ts_elements[i] for i in perm]
    permuted_positions = ts_positions[perm]

    fragA_elements = [ts_elements[i] for i in fragA_idx]
    fragA_positions = ts_positions[fragA_idx]
    fragB_elements = [ts_elements[i] for i in fragB_idx]
    fragB_positions = ts_positions[fragB_idx]

    # Write xyz audit files
    write_xyz(rxn_dir / "ts.xyz", ts_elements, ts_positions,
              comment=f"TS for {rid}")
    write_xyz(rxn_dir / "geometry_fragA.xyz", fragA_elements, fragA_positions,
              comment=f"fragA at TS for {rid}")
    write_xyz(rxn_dir / "geometry_fragB.xyz", fragB_elements, fragB_positions,
              comment=f"fragB at TS for {rid}")

    charge_a = int(frag_row["fragment_charge_a"])
    charge_b = int(frag_row["fragment_charge_b"])
    mult_a = int(frag_row["fragment_mult_a"])
    mult_b = int(frag_row["fragment_mult_b"])
    total_charge = int(sel_row["charge"])
    total_mult = int(sel_row.get("multiplicity", 1))

    single_atom_a = (n_a == 1)
    single_atom_b = (n_b == 1)

    # Write C1, C2 input files
    (rxn_dir / "c1_fragA_ts.in").write_text(
        make_c1c2_input(fragA_elements, fragA_positions, charge_a, mult_a, cfg,
                        title=f"C1: fragA SP at TS for {rid}")
    )
    (rxn_dir / "c2_fragB_ts.in").write_text(
        make_c1c2_input(fragB_elements, fragB_positions, charge_b, mult_b, cfg,
                        title=f"C2: fragB SP at TS for {rid}")
    )

    # C3: EDA on permuted atoms
    (rxn_dir / "c3_eda.in").write_text(
        make_c3_eda_input(permuted_elements, permuted_positions,
                          n_a, n_b, total_charge, total_mult, cfg)
    )

    # C4, C5: geometry optimization (skip if single-atom)
    if not single_atom_a:
        (rxn_dir / "c4_fragA_opt.in").write_text(
            make_c4c5_input(fragA_elements, fragA_positions, charge_a, mult_a, cfg,
                            title=f"C4: fragA opt for {rid}")
        )
    if not single_atom_b:
        (rxn_dir / "c5_fragB_opt.in").write_text(
            make_c4c5_input(fragB_elements, fragB_positions, charge_b, mult_b, cfg,
                            title=f"C5: fragB opt for {rid}")
        )

    # run_reaction.sh
    c4_launch, c4_wait = render_c4_lines(single_atom_a)
    c5_launch, c5_wait = render_c5_lines(single_atom_b)
    script = RUN_REACTION_TEMPLATE.format(
        rid=rid,
        ams_home=cfg["cluster"]["ams_home"],
        repo=str(REPO),
        functional=f"BLYP-D3(BJ)",
        basis=cfg["adf_protocol"]["basis"],
        frag_method=str(frag_row["partition_method"]),
        atoms_a_json=json.dumps(fragA_idx),
        atoms_b_json=json.dumps(fragB_idx),
        charge_a=charge_a, charge_b=charge_b,
        mult_a=mult_a, mult_b=mult_b,
        total_charge=total_charge,
        dataset_dEa=float(sel_row.get("delta_Ea", 0.0)),
        atom_perm_json="null" if is_identity_perm else json.dumps(perm),
        single_atom_a=str(int(single_atom_a)),
        single_atom_b=str(int(single_atom_b)),
        c4_launch=c4_launch, c5_launch=c5_launch,
        c4_wait=c4_wait, c5_wait=c5_wait,
    )
    sh = rxn_dir / "run_reaction.sh"
    sh.write_text(script)
    sh.chmod(0o755)

    return {
        "reaction_id": rid,
        "family": sel_row["family"],
        "n_atoms_total": len(ts_elements),
        "n_atoms_a": n_a, "n_atoms_b": n_b,
        "charge_a": charge_a, "charge_b": charge_b,
        "mult_a": mult_a, "mult_b": mult_b,
        "total_charge": total_charge,
        "atom_permutation": None if is_identity_perm else perm,
        "single_atom_a": single_atom_a,
        "single_atom_b": single_atom_b,
        "calcs": ["c1", "c2", "c3"]
                 + ([] if single_atom_a else ["c4"])
                 + ([] if single_atom_b else ["c5"]),
        "out_dir": str(rxn_dir),
    }


# --------------------------------- main ---------------------------------
def _sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path,
                   default=REPO / "configs" / "adf_computation.yaml")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="overrides cfg.output.adf_root")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    adf_root = args.output_dir or (REPO / cfg["output"]["adf_root"])
    adf_root = Path(adf_root).resolve()

    if adf_root.exists() and any(adf_root.iterdir()) and not args.force:
        sys.stderr.write(f"ERROR: {adf_root} non-empty (use --force)\n")
        sys.exit(1)
    if adf_root.exists() and args.force:
        shutil.rmtree(adf_root)
    adf_root.mkdir(parents=True, exist_ok=True)

    frag_path = REPO / cfg["input"]["fragments_parquet"]
    seed_path = REPO / cfg["input"]["seed_csv"]
    fragments = pd.read_parquet(frag_path)
    seed = pd.read_csv(seed_path)
    ok = fragments[fragments["partition_status"].isin(["ok", "warning"])]
    merged = seed.merge(
        ok[["reaction_id", "fragment_atoms_a", "fragment_atoms_b",
            "fragment_charge_a", "fragment_charge_b",
            "fragment_mult_a", "fragment_mult_b",
            "partition_method", "partition_status"]],
        on="reaction_id", how="inner"
    ).sort_values("reaction_id", kind="stable").reset_index(drop=True)
    print(f"reactions to build: {len(merged)}  by family: "
          f"{merged['family'].value_counts().to_dict()}")

    batch_size = int(cfg["output"]["batch_size"])
    n_batches = math.ceil(len(merged) / batch_size)

    manifest: dict = {
        "spec_version": SPEC_VERSION,
        "script_version": SCRIPT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "adf_protocol": {
            "functional": "BLYP-D3(BJ)",
            "basis": cfg["adf_protocol"]["basis"],
            "core": cfg["adf_protocol"]["core"],
            "symmetry": cfg["adf_protocol"]["symmetry"],
            "numerical_quality": cfg["adf_protocol"]["numerical_quality"],
            "relativity": cfg["adf_protocol"]["relativity"],
            "ams_home": cfg["cluster"]["ams_home"],
            "workflow_tool": "manual",
        },
        "n_batches": n_batches,
        "batch_ids": [],
        "reactions_per_batch": {},
        "reaction_ids_by_batch": {},
        "total_reactions": int(len(merged)),
        "source_fragments_parquet_sha256": _sha256_of(frag_path),
        "source_seed_csv_sha256": _sha256_of(seed_path),
    }

    failures_path = adf_root / "_failures_build.jsonl"
    failures = failures_path.open("w")
    n_ok = 0
    n_fail = 0

    for b in range(n_batches):
        batch_id = f"batch_{b:03d}"
        batch_dir = adf_root / batch_id
        batch_dir.mkdir(exist_ok=True)
        slice_df = merged.iloc[b * batch_size: (b + 1) * batch_size]
        batch_rids: list[str] = []
        for _, row in slice_df.iterrows():
            sel_dict = row.to_dict()
            sel_series = pd.Series(sel_dict)
            try:
                meta = build_reaction(batch_dir / sel_dict["reaction_id"],
                                      sel_series, sel_series, cfg)
                batch_rids.append(meta["reaction_id"])
                n_ok += 1
            except Exception as e:
                n_fail += 1
                failures.write(json.dumps({
                    "reaction_id": sel_dict["reaction_id"],
                    "family": sel_dict["family"],
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }) + "\n")
        manifest["batch_ids"].append(batch_id)
        manifest["reactions_per_batch"][batch_id] = len(batch_rids)
        manifest["reaction_ids_by_batch"][batch_id] = batch_rids
        print(f"  {batch_id}: {len(batch_rids)} reactions")
    failures.close()

    (adf_root / "batch_manifest.json").write_text(json.dumps(manifest, indent=2))

    # submit_all.sh — master driver (gate1 bash mode)
    submit_sh = adf_root / "submit_all.sh"
    submit_sh.write_text(f"""#!/bin/bash
# Master driver — runs all batches sequentially on gate1.
# Per-batch concurrency PAR (default 4) controls how many reactions run at once.
PAR="${{1:-4}}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO={REPO}
for batch_dir in "$ROOT"/batch_*; do
    [[ -d "$batch_dir" ]] || continue
    echo ""
    echo "=== running $(basename $batch_dir) at $(date -Iseconds) ==="
    bash "$REPO/adf_outputs/run_batch.sh" "$batch_dir" "$PAR"
done
echo ""
echo "=== all batches done at $(date -Iseconds) ==="
""")
    submit_sh.chmod(0o755)

    # run_batch.sh — per-batch xargs runner
    run_batch_sh = adf_root / "run_batch.sh"
    run_batch_sh.write_text(f"""#!/bin/bash
# Run all reactions in a single batch via xargs -P N (gate1, NSCM=1, nice 10).
# Idempotent: reactions whose status.json shows exit_code:0 are skipped.
# NO `set -e` — individual reaction failures must not abort the batch.
set -o pipefail
BATCH_DIR="${{1:?usage: $0 <batch_dir> [parallelism=4]}}"
PAR="${{2:-4}}"

run_one() {{
    local rxn_dir="$1"
    local rid="$(basename "$rxn_dir")"
    local status="$rxn_dir/status.json"
    if [[ -f "$status" ]] && python3 -c "
import json, sys
try:
    s = json.load(open('$status'))
    sys.exit(0 if s.get('exit_code', 1) == 0 else 1)
except Exception:
    sys.exit(1)
" 2>/dev/null; then
        return  # already done
    fi
    nice -n 10 bash "$rxn_dir/run_reaction.sh" 2>/dev/null || true
}}
export -f run_one

find "$BATCH_DIR" -mindepth 1 -maxdepth 1 -type d -print | \\
    xargs -P "$PAR" -I {{}} bash -c 'run_one "$@"' _ {{}} || true

echo "=== run_batch done at $(date -Iseconds) ==="
""")
    run_batch_sh.chmod(0o755)

    print(f"\nbuilt {n_ok} reactions across {n_batches} batches; failed {n_fail}")
    print(f"manifest -> {adf_root / 'batch_manifest.json'}")
    print(f"submit_all -> {submit_sh}")


if __name__ == "__main__":
    main()
