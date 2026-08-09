# CLAUDE.md — tt EDA-NOCV Labeling Package (self-contained)

> This file instructs a **fresh Claude Code session** on a new HPC (typically KISTI)
> to execute an ORCA EDA-NOCV labeling pipeline for the "three_two_cycloaddition"
> (tt, [3+2] cycloaddition) subset of the Bath 1480 dataset (Espley 2024, DOI:
> 10.1039/D4DD00224E). All data and scripts required are inside this package.
> Read this file FIRST before doing anything.

---

## 1. Immediate context

**Task**: Run ORCA 6.x EDA-NOCV single-point calculations on ~3,662 pre-built
input files (`data/rxn_XXXX/eda.inp`), then aggregate the 5-channel energy
decomposition results into a parquet.

**Why it matters**: These are supervised labels for a downstream ML model that
predicts distortion/interaction decomposition of [3+2] cycloaddition barriers.
The original Espley et al. paper reports 4-channel data (their custom scheme);
we produce 5-channel EDA-NOCV (Pauli, Elstat, Orb, Disp, + implicit Prep).

**Origin**: The package was built on a companion HPC and copied to this one.
The upstream Bath 1480 zip (4 GB) is NOT included — only the derived ORCA inputs
and manifest.

---

## 2. HPC ground rules (READ FIRST — non-negotiable)

- **ALL non-trivial compute MUST run via `sbatch`.**  No direct `python` /
  `orca` invocations on the login node. Even 5-minute jobs go through sbatch.
- **Every `sbatch` uses `#SBATCH --time=48:00:00`** (or KISTI equivalent max
  per queue). No shorter walltimes.
- **All work is idempotent.** Each ORCA reaction writes `eda.out` atomically;
  if `eda.out` contains "ORCA TERMINATED NORMALLY", skip that reaction on
  resubmit.
- **If a 48h wall clips a shard, just re-`sbatch` the same script.** Idempotency
  makes this safe — completed reactions are preserved.

### Queue caps (⚠ FILL IN BEFORE RUNNING)
KISTI Nurion / Neuron / Alpha 5 have different limits per queue and per
allocation. Before submitting, the human user MUST provide:

- Max concurrent RUNNING jobs (tasks): `<N_RUN>`
- Max concurrent PENDING + RUNNING (tasks): `<N_SUBMIT>`
- Recommended partition for CPU ORCA jobs: `<PARTITION>`
- Whether SLURM `--array=` supports `%K` concurrency cap: yes/no

Design the array so `N + 1 ≤ N_SUBMIT` and concurrency cap `K ≤ N_RUN`.
If exceeded, chain arrays via `--dependency=afterany:<jid>` or wait for the
current batch to drain before submitting the next.

### Long-running processes on login node
FORBIDDEN. No `nohup … &`, no shell loops, no launchers, no `tail -F`
watchers, no persistent screen/tmux jobs. A one-shot `sbatch script.sh` is
fine; anything that stays resident past that submit is not.

---

## 3. Package layout

```
tt_eda_kisti_package/
├── README.md                     — human-readable overview
├── CLAUDE.md                     — this file
├── VERSION.md                    — package version + provenance
├── data/
│   ├── manifest.parquet          — 3,662 rows: {rxn_id, n_atoms, symbols_str,
│   │                                ts_coords_x/y/z, imag_freq_cm1,
│   │                                frag_A_indices, frag_B_indices, ...}
│   ├── rxn_XXXX/                 — one dir per reaction (padded to 4 digits)
│   │   └── eda.inp               — pre-built ORCA EDA-NOCV input
│   └── espley_reference.parquet  — Espley's DFT energies for sanity check
├── scripts/
│   ├── sbatch_run_orca.sh        — SLURM sbatch template (edit for KISTI)
│   ├── aggregate_channels.py     — parse eda.out → 5-channel parquet
│   ├── verify_vs_espley.py       — cross-check our results vs Espley refs
│   └── run_orca_single.sh        — helper: run ORCA on one rxn_XXXX/eda.inp
└── env/
    ├── requirements.txt          — pip deps for parsing/aggregation
    └── setup_env.sh              — conda env recipe
```

---

## 4. ORCA input details (already baked in)

Every `data/rxn_XXXX/eda.inp` is built for:

- Method: **B3LYP-D3(BJ) / def2-TZVP** (matches Espley 2024 SPE conditions)
- Solvation: **CPCM shell with SMD parametrization, water** (matches Gaussian
  `scrf=(smd,solvent=water)` in Espley outputs)
- ORCA `EDA` keyword: produces Pauli / Elstat / Orb (NOCV) / Disp channels
- Fragment charges/multiplicities: **(0, 1) + (0, 1)** — tt reactions are all
  neutral closed-shell singlets (verified via audit of Bath 1480 file
  metadata; no ionic dipoles present in this subset)
- Fragment assignment: `C(1)`, `C(2)`, `H(1)` etc. — ORCA reads fragment IDs
  from parentheses on atom labels

If any reaction is later found to need different charge/mult, edit that specific
`eda.inp` and re-submit.

---

## 5. Workflow steps (exact commands)

### Step 0 — set up environment
```bash
cd tt_eda_kisti_package/
bash env/setup_env.sh                # creates conda env "tt_eda"
conda activate tt_eda
```

### Step 1 — verify package integrity
```bash
python scripts/verify_vs_espley.py --mode=preflight
# checks: manifest row count == number of rxn_XXXX dirs == expected 3662
```

### Step 2 — locate ORCA binary
Edit `scripts/sbatch_run_orca.sh`:
- Set `ORCA_BIN=` to your ORCA 6.x executable path
- Set `#SBATCH --partition=<PARTITION>`
- Set `#SBATCH --cpus-per-task=<N_CORES>` (recommend 4-8 for 20-atom systems)
- Set `#SBATCH --mem=<MEM>` (recommend 16-32G)
- Set `#SBATCH --array=0-3661%<K>` where `K = min(N_RUN, N_SUBMIT-1)`

### Step 3 — dry-run one reaction (sanity)
```bash
sbatch --array=0 scripts/sbatch_run_orca.sh
squeue -u $USER
# wait for completion, then:
cat data/rxn_0000/eda.out | tail -50
# expect "ORCA TERMINATED NORMALLY" and an EDA table
```

### Step 4 — full submission
```bash
sbatch scripts/sbatch_run_orca.sh
# if it hits array size limits, chain via afterany:
#   JID=$(sbatch --parsable scripts/sbatch_run_orca.sh)
#   sbatch --dependency=afterany:$JID scripts/sbatch_run_orca.sh
```

### Step 5 — aggregate results
```bash
sbatch scripts/sbatch_aggregate.sh
# writes results/tt_eda_5channel.parquet
```

### Step 6 — verify vs Espley reference
```bash
sbatch scripts/sbatch_verify.sh
# writes results/verification_report.md
# spot-checks: our (Elstat + Orb + Disp) ≈ Espley interaction_energy_dft
```

---

## 6. Expected outputs

Under `results/` (created by aggregation step):

- `tt_eda_5channel.parquet` — one row per reaction:
  - `rxn_id`, `n_atoms`, `frag_A_count`, `frag_B_count`
  - `e_pauli_kcal`, `e_elst_kcal`, `e_orb_kcal`, `e_disp_kcal`
  - `e_int_kcal` (= sum of 4 above)
  - `e_prep_kcal` (from Espley refs: distortion energy sum)
  - `terminated_normally` (bool)
  - `nocv_top_channels` (JSON: top 5 NOCV pairs with their energies)

- `verification_report.md` — summary: 
  - N reactions succeeded / failed
  - MAE(our_e_int, espley_interaction_dft) — should be small (<3 kcal/mol) if
    ORCA and Gaussian give consistent EDA
  - List of anomalous reactions (large disagreement) for manual review

---

## 7. Common failure modes

1. **SCF didn't converge**: In `eda.out` search `WARNING: SCF NOT CONVERGED`.
   Fix: edit `eda.inp` for that reaction, add `SlowConv` or `%scf maxiter 300 end`.

2. **CPCM(SMD) parameter issue**: ORCA versions before 5.0.4 have partial SMD
   support. If seeing errors about `smdsolvent`, verify ORCA version ≥ 5.0.4
   (6.x preferred).

3. **Fragment assignment mismatch**: If ORCA rejects `C(1) / C(2)` labels,
   the ORCA version may parse fragments differently. Check with a small
   test input: `data/rxn_0000/eda.inp` on a single core (no MPI).

4. **48h wall hit mid-array**: Just resubmit — idempotency preserves progress.

---

## 8. What NOT to modify

- **Do not rebuild `eda.inp` files** — their partition + coord data are
  authoritative (built and validated against Espley's frag counts).
- **Do not change the ORCA method line** — deviating from B3LYP-D3(BJ)/def2-TZVP
  breaks comparability with Espley reference.
- **Do not commit `data/rxn_XXXX/eda.out`** to git — these are large. If you
  need to share results, ship `results/tt_eda_5channel.parquet` instead.

---

## 9. Task completion checklist (report back to user when done)

- [ ] N reactions attempted: X
- [ ] N reactions terminated normally: X
- [ ] N reactions failed: X — list rxn_ids and top failure category
- [ ] MAE(our_e_int, espley_ref) = X kcal/mol
- [ ] `results/tt_eda_5channel.parquet` written, row count = X
- [ ] Wall time consumed: X hours
- [ ] Total compute cost: X core-hours

---

## 10. Provenance & versioning

Package source: `/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/bath_1480/tt_kisti_package/`
on the originating HPC (contact user for details). Built from:
- **Data**: `data_archive_files.zip` from https://researchdata.bath.ac.uk/1480/
  (SHA1/MD5 in `VERSION.md`)
- **Filter**: audit of complete 5-file sets (am1_ts + splits/dft × 4) yielded
  3,662 candidate reactions from 3,980 total tt entries
- **Partition**: diassep.py algorithm (Espley's own tool) applied to AM1 TS
  imaginary mode; 100% agreement with Espley's own fragment atom counts on
  audited subset
- **Method**: matches Espley 2024 DFT SPE protocol

See `VERSION.md` for exact commits, dates, and reproducibility hashes.
