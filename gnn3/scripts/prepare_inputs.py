"""Build GNN3 inputs (182 global features + 29-wide atom tokens) for D1 reactions, by running the existing code.

    python gnn3/scripts/prepare_inputs.py <stage> [<stage> ...] --root ROOT --jobs LIST.csv [--cores 4]
                                          [--reuse-finalized FROM_ROOT] [--manifest M.csv --n-parallel N]

Stages, in order: finalize metadata features tokens (and ts-search for a new TS search). Each stage needs its job
environment, so run it through the original job wrappers (or use the chain gnn3/jobs/prepare_inputs.sh):

    cd analysis/deploy && sbatch -p cpu1,cpu2 --cpus-per-task=8 --time=48:00:00 -o LOG jobs/cpu_d1.slurm \
        $REPO/gnn3/scripts/prepare_inputs.py finalize metadata --root ROOT --jobs LIST.csv
    cd analysis/d1_pair && sbatch -p cpu1,cpu2 --cpus-per-task=8 --time=48:00:00 -o LOG \
        --export=ALL,PYENV=venv5,D1PAIR_PHASE=F jobs/g.slurm \
        $REPO/gnn3/scripts/prepare_inputs.py features tokens --root ROOT --jobs LIST.csv

LIST.csv has a job_id column (D1 job ids, e.g. T00017; G_status.csv maps them to rxn_id). Outputs under ROOT:
X182_lf.parquet (rxn_id + 182 columns), fu2_lf/tokens/<rxn_id>.npz, and the stage reports in ROOT/results.
--reuse-finalized copies finished finalize folders from another root (e.g. the published /gpfs/tmp_cpu2/yeseo1ee/ch3/lf)
instead of rerunning the ORCA frequency job. The TS search itself (autodE) is random and is not rerun for D1 reactions:
their TSs come from the existing D1-G2 folders (/gpfs/tmp_cpu2/yeseo1ee/d1_g2/jobs/<job_id>).
Exit 0 done, 2 not ready (resubmit the same command), 3 STOP.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gnn3.inputs.pipeline import STAGES, run_stage  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stages", nargs="+", choices=list(STAGES))
    parser.add_argument("--root", type=Path, required=True, help="output root (fresh folder; everything is written here)")
    parser.add_argument("--jobs", type=Path, required=True, help="csv with a job_id column")
    parser.add_argument("--cores", type=int, default=4, help="cores per ORCA / autodE run (D1-G2 used 4)")
    parser.add_argument("--reuse-finalized", type=Path, default=None, help="root whose g2lf_final folders are copied")
    parser.add_argument("--manifest", type=Path, default=None, help="ts-search only: D1-format manifest")
    parser.add_argument("--n-parallel", type=int, default=1, help="ts-search only: reactions at a time")
    args = parser.parse_args()
    for stage in args.stages:
        rc = run_stage(stage, args.root, args.jobs, args.cores, args.manifest, args.n_parallel,
                       args.reuse_finalized if stage == "finalize" else None)
        if rc != 0:
            print(f"stage {stage}: exit {rc}", flush=True)
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
