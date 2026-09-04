# CLAUDE.md — eda-asm-prediction

## HPC ground rules (READ FIRST — non-negotiable)

- **ALL non-trivial compute MUST run via `sbatch` on compute nodes.**
  The login node `gate1.hpc` is **read-only** for us. No `python`,
  no `xtb`, no matplotlib rendering, no MACE feature extraction on
  login. Even a 15-row Hammett plot goes through sbatch.
- **Every `sbatch` submission MUST use `#SBATCH --time=48:00:00`.**
  No shorter walltimes. If the underlying work only needs a few
  minutes, still ask for 48h — the extra time costs nothing while a
  too-tight walltime silently kills long tails.
- **All work must be idempotent on resubmit.** Each cell / shard /
  batch must (a) write its output atomically and (b) skip work whose
  output file already exists. If a 48h wall clips a partial cell,
  losing that cell is acceptable — losing the *rest* is not.
- **If a job hits the 48h wall, just re-`sbatch` the same script.**
  Idempotency + `if out_path.exists(): return` in the runner makes
  this safe.
- **SLURM quota: MaxJobs=10 running, MaxSubmit=20 in queue.**
  Both caps are on TASKS, not on wrapper submissions —
  **an array counts as one task per array element** on this cluster,
  not "one submission". Practical consequences:
    - A single `sbatch --array=0-N%K` submission uses N+1 slots
      against the 20-cap (not 1). If N+1 > 20, submission fails.
    - Design arrays with `N < 20 − (currently_pending)` and fan out
      the rest through `--dependency=afterany` chains or by
      resubmitting once the first batch drains.
    - Before every `sbatch`, check `squeue -u $USER -h | wc -l` —
      that count is task-level and is what the 20-cap tests against.
- **Partition selection: pick whichever has idle capacity, don't
  be picky about which one.** Within a given resource class (CPU vs
  GPU), the specific partition does not matter — cpu1 and cpu2 are
  interchangeable for our workloads, and gpu3 / gpu4 / gpu5 are
  interchangeable for training / MACE feature extraction. The right
  answer is *whichever partition currently has idle nodes and no
  pending queue*, so the job starts immediately instead of sitting
  in `PENDING (Priority)` on a busy partition while a sister
  partition sits idle. Before `sbatch`:
    - CPU jobs: `sinfo -p cpu1,cpu2 -o "%.9P %.6t %C"` and pick the
      one with the largest idle CPU count (the `I` column of
      `A/I/O/T`). If both are packed, either is fine — take one.
    - GPU jobs: `sinfo -p gpu3,gpu4,gpu5 -o "%.9P %.6t %.10G %C"`
      and target the partition with a free GPU. Same idea.
- **NO long-running processes on the login node — ever.** This is
  a stricter reading of the "login is read-only" rule and includes
  everything below, not just python/xtb/MACE:
    - No `nohup … &` on gate1.hpc for any purpose.
    - No shell loops (`while true; do sbatch …; sleep …; done`),
      no launchers, no pollers, no watchers, no `tail -F` daemons,
      no `screen` / `tmux` sessions used for background work.
    - A one-shot `sbatch script.sh` command is fine (it returns
      immediately). Anything that stays resident past that submit
      is not.
- **If you need automated / throttled job submission, use one of
  these — never a login-node daemon:**
    1. **SLURM job array with a concurrency cap.** `sbatch
       --array=0-N%K script.sh` runs at most `K` array tasks at a
       time. `K = 10` matches our RUNNING concurrency limit.
       **Array tasks each count against the 20-task submit cap.**
       To fan out beyond 20 tasks, split into sequential arrays
       chained with `--dependency=afterany` (see (3)) or resubmit
       the next batch once the first drains.
    2. **Submit the launcher itself as an sbatch job** on a small
       cpu partition (`--time=48:00:00`, `--mem=2G`, `--cpus-per-task=1`,
       e.g. cpu2). The launcher's polling + `sbatch` calls then
       run on a compute node, not the login node. Idempotent
       runners make this safe to restart at the 48h wall.
    3. **`--dependency=afterany:<jid>` chains** for strict
       ordering — one job releases the next when it finishes,
       with no polling needed.
  If none of these fit the task, stop and ask before doing
  anything that would leave a process on gate1.hpc.

## Project overview

EDA (Energy Decomposition Analysis) + ASM (Activation Strain Model)
proxy prediction of 5-channel decomposed activation energies.

**Current status (2026-09-04):** Full B3LYP-EDA relabeling of the
Coley 5269-reaction dipolar cycloaddition dataset (spec16rev). All
previous labeling experiments (789-rxn ADF, 3504-rxn cohort,
spec14/15 validations, m1/m2/m3 delta learners) have been removed —
those labels turned out to be incorrect and would cause confusion
with the current authoritative labels being produced now.

Folder/distribution name: `eda-asm-prediction` (hyphenated).
Python import name: `eda_asm`.

## Repository layout

```
analysis/
  b3lyp_full/            spec16rev — current label pipeline
                         (5262 rxns × 5 SPEs, B3LYP EDA-NOCV)
  bath1480_probe/        experiment scripts only (results removed;
                         code kept for future reuse)
src/eda_asm/             shared Python package
  datasets/              dataset loaders (dipolar, qmrxn20)
  asr_v1/                model / backbone / training utilities
  adf/                   ADF input builder + parser (legacy, reusable)
scripts/                 utility scripts (fragment tools, ORCA input
                         builders, etc.)
backbone_ft/             MACE-OFF23 fine-tune experiment (gitignored)
docs/                    documentation
CLAUDE.md                this file
```

## Current label pipeline (spec16rev b3lyp_full)

Full B3LYP-D3(BJ)/def2-TZVP CPCM(water) EDA-NOCV on the Coley 5269
dipolar cycloaddition dataset. Per reaction:
- 5 ORCA single-point calculations run in parallel:
  `eda` + `frag1_dist` + `frag2_dist` + `frag1_rel` + `frag2_rel`
- 5-channel EDA labels + strain corrections
- Self-chaining orchestrator (`orch_b3full.sh`) handles the 47h
  wall-time limit by automatically re-submitting itself
  (`--dependency=afterany:$SLURM_JOB_ID`)

Pipeline stages under `analysis/b3lyp_full/`:

| stage | script | what it does |
|---|---|---|
| 0 | `00_setup.py` | contamination check + Coley 5269 verification |
| 1 | `01_build_inputs.py` | build 5 ORCA inputs per rxn (Coley r*.xyz) |
| 2 | `02_submit.sh` | per-rxn ORCA runner (SLURM array element) |
|   | `orch_b3full.sh` | continuous-fill + self-chaining orchestrator |
| 3 | `03_parse.py` | parse ORCA outputs → per-rxn EDA channels |
| 4 | `04_qc.py` | quality-check labels (post-run) |
| 5 | `05_compare.py` | compare vs external references (post-run) |
| 6 | `06_export.py` | export final labels parquet (post-run) |
| 7 | `07_report.py` | REPORT.md with figures + gate status |

Cleanup policy (`02_submit.sh`): after all 5 SPEs terminate normally,
delete densities/CPCM/tmp files but keep `.inp/.out/.err/.gbw`.
Wavefunctions (~10 MB/rxn, ~52 GB total for 5262) retained for
future re-analysis without SCF re-run.

Idempotency: each SPE is skipped if its `.out` already contains
`ORCA TERMINATED NORMALLY`. Safe to resubmit at any time.

## Datasets — where the geometries come from

Raw data under `/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/` (regenerable,
gitignored).

| family | count | source | archive |
|---|---|---|---|
| **dipolar (current label target)** | **5269** | Stuyver / Jorner / Coley 2023 | figshare 21707888 v5 |
| qmrxn20_e2 | 200 | von Rudorff 2020 | materialscloud 2020.55 |
| qmrxn20_sn2 | 196 | von Rudorff 2020 | (same) |
| rgd1 | (available) | Zhao & Savoie 2023 | figshare 21066901 v6 |

Only the **dipolar 5269** dataset is currently being labeled
(spec16rev). Others are available raw for future labeling.

## Environment

- Conda env: `reactot` (Python 3.10, torch 2.2.1, mace, nequip,
  tblite, rdkit, ase, pandas, matplotlib).
- Activation: `source /home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh;
  conda activate reactot`. Every sbatch script does this.
- QM tools: ORCA 6.1.1 at `$HOME/orca_6_1_1_avx2/` (used for b3lyp_full).
- MPI: OpenMPI 4.1.7a1 at `/usr/mpi/gcc/openmpi-4.1.7a1/`. ORCA
  parallel EDA (`%pal nprocs 5`) requires `OMPI_MCA_pml=ob1`,
  `OMPI_MCA_coll_hcoll_enable=0`, `UCX_TLS=tcp,self,sm` — UCX/hcoll
  crash MPI mid-run on some nodes.

## Home directory quota

- Home quota: **300 GB** (raised from 100 GB on 2026-09-02).
- Current usage: ~130 GB (`analysis/b3lyp_full/` scratch + .gbw
  retention, ~52 GB predicted at completion).
- Test quota: `dd if=/dev/zero of=~/qtest.bin bs=1M count=500` should
  succeed. "Disk quota exceeded" means the ORCA basis-write error
  (`TBasis::WriteElement`) will resurface.

## Conventions

- Random seeds always come from the config (default seed = 42).
- All energies stored in **kcal/mol** unless the column name says
  otherwise (`_Eh` = hartree, `_h` = hartree).
- Geometries in Å.
- Never commit `/gpfs/tmp_cpu2/yeseo1ee/...` outputs. Raw datasets
  and any large regenerable data live there and are gitignored.
