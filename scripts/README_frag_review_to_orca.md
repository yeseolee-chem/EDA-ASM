# Manual fragment review → ORCA EDA-NOCV recompute

End-to-end workflow to fix wrong auto-partitions on 787 reactions and
regenerate ORCA EDA-NOCV labels from manually verified fragments.

## Files

| step | script | purpose |
|---|---|---|
| 1 | `scripts/frag_review_app.py` | Flask+3Dmol.js review UI |
| 1 | `scripts/frag_review_app.sh` | sbatch launcher (48h, cpu2) |
| 2 | `scripts/make_orca_eda_inputs.py` | emit 787 ORCA `.inp` from reviewed JSON |
| 3 | `scripts/run_orca_eda_array.sh` | SLURM array (%10 concurrent, 48h) |

## Step 1 — Launch the review app on a compute node

```bash
sbatch scripts/frag_review_app.sh
```

Check the job's stdout for the compute node + tunnel instructions:

```bash
tail -f /gpfs/tmp_cpu2/yeseo1ee/eda_asm_features/spec_v1_logs/frag_review.<JOBID>.out
```

You'll see something like:

```
Fragment review app starting on compute node: n014
Port:                                         5788
Port-forward from your LAPTOP terminal:
    ssh -N -L 5788:n014:5788 yeseo1ee@gate1.hpc
Then open in your browser:
    http://localhost:5788
```

Run the `ssh -N -L …` command in a **local laptop terminal** (leave it
open), then browse to `http://localhost:5788`.

### Review UI cheatsheet

- Click atoms to select (highlighted yellow)
- `a` — assign selected atoms to fragment A (blue)
- `b` — assign selected atoms to fragment B (orange)
- `c` — clear current selection
- `s` — swap A ↔ B (if the auto guessed the sides swapped)
- `r` — mark current reaction reviewed and go to next
- `n` / `p` — next / previous reaction (no save)
- Family + review-status filters on the top-left
- Auto-saves after every action to
  `outputs/frag_review/manual_partitions.json`

Progress is preserved. Stop and resume anytime — the sbatch job stays
alive up to 48h; re-`sbatch` to restart if needed.

## Step 2 — Generate ORCA EDA-NOCV inputs

After all (or a subset of) reactions are marked reviewed:

```bash
# Dry-run: first 5 only
python scripts/make_orca_eda_inputs.py --limit 5 --only-reviewed
# Full run: 787 reactions (only those marked reviewed)
python scripts/make_orca_eda_inputs.py --only-reviewed
# Or emit all, ignoring the reviewed flag (uses whatever manual_partitions.json says):
python scripts/make_orca_eda_inputs.py
```

**Warning:** the input generator runs quickly but should still be
executed on a compute node (CLAUDE.md rule):

```bash
srun --partition=cpu2 --time=48:00:00 --mem=8G --pty \
    python scripts/make_orca_eda_inputs.py --only-reviewed
```

Outputs: `outputs/orca_eda/inputs/<rid>/{eda.inp,meta.json}`

## Step 3 — Run 787 ORCA EDA-NOCV jobs

```bash
sbatch scripts/run_orca_eda_array.sh
```

- Array size: `0-786%10` (10 concurrent, ≤ CLAUDE.md limit)
- Each task: 8 CPUs, 32 GB RAM, 48 h walltime
- Idempotent: skips if `eda.out` contains "ORCA TERMINATED NORMALLY"
- If the 48 h wall clips a task, just re-`sbatch` — remaining tasks
  restart from scratch, done tasks are skipped.

## Step 4 — Parse results

Reuse the existing parser (from the `runs/orca_recompute/` legacy
pipeline). Adapt paths or write a fresh parser targeting
`outputs/orca_eda/inputs/<rid>/eda.out` — output should be a parquet
that matches `labels/orca/orca_eda_labels.parquet` schema.

## Charge / multiplicity convention

Baked into `make_orca_eda_inputs.py:charge_and_mult`:

| family | total | fragA (charge, mult) | fragB (charge, mult) |
|---|---|---|---|
| dipolar | 0 | (0, 1) | (0, 1) |
| rgd1 | 0 | (0, 1) | (0, 1) |
| qmrxn20_e2 (closed-shell) | −1 | (−1, 1) | (0, 1) |
| qmrxn20_sn2 | −1 | (−1, 1) | (0, 1) |
| qmrxn20_e2 (open-shell) | −1 | (−1, 2) | (0, 2) + `FRAG2_SF=TRUE` |

Open-shell hint is currently `False` by default — the parser can flag
open-shell retry candidates by SCF non-convergence, and the second
pass sets `--functional BLYP` + `open_shell_hint=True`. Wire this up if
the initial pass produces spin-contamination outliers.

## What if I want to re-review one reaction?

Reopen the app (or leave it running), navigate to the reaction in the
list, edit atoms → save. The `.inp` will need re-generation:

```bash
rm -rf outputs/orca_eda/inputs/<rid>
python scripts/make_orca_eda_inputs.py --only-reviewed
```

Then the SLURM array's idempotency check will re-run only the missing
`<rid>`.
