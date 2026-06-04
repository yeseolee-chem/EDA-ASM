"""For a given AL round's ``picks.csv``, produce ADF input decks under
``ADF_AL/round_<R>/batch_000/<reaction_id>/`` and a submit script.

Steps:
  1. Run ``define_fragments.py`` on the round's mini-seed CSV (the 40
     picks) → ``round_<R>/fragments/fragments.parquet``.
  2. Call into the generator logic of ``ADF_250/scripts/adf/generate_adf_inputs.py``
     (imported as a module) with a custom output dir, using the round's
     fragments.parquet + seed CSV.
  3. Produce ``round_<R>/run_batch.sh`` and a Slurm array submission script.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--round-dir", default=None)
    ap.add_argument("--adf-root", default=None,
                    help="default outputs/asr_v1/al/round_<R>/adf_inputs")
    ap.add_argument("--frag-config", default="configs/fragment_partitioning.yaml")
    ap.add_argument("--adf-config", default="ADF_250/configs/adf_computation.yaml")
    args = ap.parse_args()

    repo = Path.cwd()
    round_dir = Path(args.round_dir or f"outputs/asr_v1/al/round_{args.round:02d}")
    picks_csv = round_dir / "picks.csv"
    if not picks_csv.exists():
        raise FileNotFoundError(f"picks.csv missing: {picks_csv}")
    print(f"[prep] round={args.round}  picks={picks_csv}")

    # 1) Fragmentation on the K picks --------------------------------------------------
    frag_out = round_dir / "fragments"
    frag_out.mkdir(parents=True, exist_ok=True)
    print("[prep] running define_fragments.py on the K picks...")
    res = subprocess.run(
        [
            sys.executable, "scripts/define_fragments.py",
            "--config", args.frag_config,
            "--seed-csv", str(picks_csv),
            "--output-dir", str(frag_out),
            "--force",
        ],
        check=False, capture_output=True, text=True,
    )
    print("--- fragmentation stdout (tail) ---")
    print((res.stdout or "")[-1500:])
    if res.returncode != 0:
        print("--- fragmentation stderr ---")
        print((res.stderr or "")[-2000:])
        raise RuntimeError("define_fragments.py failed")
    frag_parquet = frag_out / "fragments.parquet"
    if not frag_parquet.exists():
        raise FileNotFoundError(f"expected {frag_parquet}, missing")
    frags = pd.read_parquet(frag_parquet)
    print(f"[prep] fragments produced: {len(frags)}  by status: "
          f"{frags['partition_status'].value_counts().to_dict()}")

    # 2) Generate ADF inputs via a custom adf_computation.yaml --------------------------
    adf_root = Path(args.adf_root or round_dir / "adf_inputs").resolve()
    adf_root.parent.mkdir(parents=True, exist_ok=True)

    # Build a one-off cfg that points to round-specific fragments + seed.
    base_cfg = yaml.safe_load(Path(args.adf_config).read_text())
    # ADF_250's generator uses paths RELATIVE to its REPO (= ADF_250 parent).
    # We instead point it at our round files via absolute paths.
    round_cfg = dict(base_cfg)
    round_cfg["input"] = {
        "fragments_parquet": str(frag_parquet.resolve()),
        "seed_csv": str(picks_csv.resolve()),
    }
    round_cfg["output"] = {
        "adf_root": str(adf_root),
        "batch_size": int(base_cfg["output"].get("batch_size", 100)),
    }
    round_cfg_path = round_dir / "adf_computation.yaml"
    round_cfg_path.write_text(yaml.safe_dump(round_cfg))

    # Generator expects to import eda_asm.adf — make sure path works from CWD.
    # Use ADF_250's generate_adf_inputs.py as the canonical source; it reads
    # the cfg's `input.fragments_parquet` directly when the path is absolute.
    print(f"[prep] generating ADF input decks under {adf_root} ...")
    if adf_root.exists():
        shutil.rmtree(adf_root)
    res = subprocess.run(
        [
            sys.executable, "ADF_250/scripts/adf/generate_adf_inputs.py",
            "--config", str(round_cfg_path),
            "--output-dir", str(adf_root),
            "--force",
        ],
        check=False, capture_output=True, text=True,
    )
    print("--- generate_adf_inputs stdout (tail) ---")
    print((res.stdout or "")[-1500:])
    if res.returncode != 0:
        print("--- generate_adf_inputs stderr ---")
        print((res.stderr or "")[-3000:])
        raise RuntimeError("generate_adf_inputs.py failed")

    # 3) Write a submission helper -----------------------------------------------------
    submit_sh = round_dir / "submit_round_adf.sh"
    submit_sh.write_text(f"""#!/bin/bash
# AL round {args.round}: run all reactions in the round's adf_inputs/batch_000/
# via the same xargs -P pattern as ADF_250 (gate1, NSCM=1, nice 10).

#SBATCH --job-name=al_r{args.round:02d}_adf
#SBATCH --partition=cpu1,cpu2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output={round_dir}/slurm-adf-%j.out
#SBATCH --error={round_dir}/slurm-adf-%j.err

set -o pipefail
cd "{repo}"
source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh
conda activate reactot

PAR="${{PAR:-8}}"
echo "[$(date)] === AL round {args.round} ADF batch start (job $SLURM_JOB_ID, par=$PAR) ==="
for batch_dir in {adf_root}/batch_*; do
    [[ -d "$batch_dir" ]] || continue
    echo "--- batch: $batch_dir ---"
    bash ADF_250/adf_outputs/run_batch.sh "$batch_dir" "$PAR"
done
echo "[$(date)] === AL round {args.round} ADF batch done ==="
""")
    submit_sh.chmod(0o755)
    print(f"[prep] wrote submit script: {submit_sh}")

    # Per-round summary update
    summary_path = round_dir / "picks_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["adf_inputs_root"] = str(adf_root)
    summary["fragments_ok"] = int((frags["partition_status"] == "ok").sum())
    summary["fragments_warning"] = int((frags["partition_status"] == "warning").sum())
    summary["fragments_failed"] = int((frags["partition_status"] == "failed").sum())
    summary["submit_script"] = str(submit_sh)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[prep] done. Submit with: sbatch {submit_sh}")


if __name__ == "__main__":
    main()
