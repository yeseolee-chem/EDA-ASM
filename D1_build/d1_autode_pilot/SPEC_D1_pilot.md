# SPEC — D1 autodE pilot: TS search → d1, d2, EDA in one chain (10 random D1 TS rows + D0 controls)

Owner: Yeseo (yeseolee-chem/EDA-ASM). Executor: Claude Code on the UBAI cluster.
Bundle: `analysis/d1_autode_pilot/` (this file + the code next to it). Date: 2026-09-29.

---

## 0. What to do, in order (read §2 first)

1. Copy the bundle into the repo as `analysis/d1_autode_pilot/` and copy `D1_설계_v8.xlsx` next to it
   (only `sample_pilot.py` needs it; the pilot manifest is already built and committed with the bundle).
2. **S-1 environment check** (§3.2). If autodE 1.4.5 is not importable in the configured env, STOP and ask
   the user before installing anything.
3. Edit only the `host` block of `config.yaml` if a path differs (§3.3). Never edit the `method` blocks.
4. Submit once from the login node: `bash submit_pilot.sh` (§7). It chains
   **S0 preflight → 14-task array (runs only if S0 exits 0) → report**. 16 tasks in total.
5. When the report job has finished, hand back `results/pilot_report.md` + `results/S0_REPORT.md`
   with the §9 summary. Commit code + the small result files (§7.5). Do not start the full D1 run.

If S0 fails, the array never starts (`afterok` + `--kill-on-invalid-dep=yes`). Read
`$SCRATCH/s0/S0_REPORT.md`, report the failing check, and apply §8 — do not patch around it.

---

## 1. Objective and definition of done

Per D1 TS row: **autodE TS search → strain references → 5 ORCA single points → method-① labels**
(barrier, d1, d2, elst, pauli, oi, disp, cpcm, cds), in one sbatch chain, at exactly the level
and with exactly the code that produced the D0 labels (`labels_all.json`).

The pilot answers five questions before any D1-scale spend:

| # | question | measured by |
|---|---|---|
| Q1 | Does autodE 1.4.5 + ORCA 6.1.1 find D1 TSs at Coley's success rate (87.7 %)? | A1 |
| Q2 | Do D1 inputs/outputs pass every D0 gate unchanged? | A2 + S0 parity |
| Q3 | Is the SP → label half of the chain identical to D0 on identical geometry? | A3 (D0_replay) |
| Q4 | Does the ported stereo-reference rule make Coley's decision? | A4 (+ S0 config validation) |
| Q5 | How close is a from-scratch autodE rerun of a D0 reaction to its D0 label, and what does D1 cost? | A5, cost table |

Done = the report job has written `results/pilot_report.md` and every acceptance line (§8) is either
PASS or explained with a concrete cause.

---

## 2. Ground rules (non-negotiable; CLAUDE.md applies in full)

- All compute through `sbatch` on compute nodes; `#SBATCH --time=48:00:00` always. The login node
  runs only one-shot commands (`sbatch`, `squeue`, `sinfo`, `sacct`, `git`, `cat`). No python there,
  no loops, no pollers, no `nohup`.
- MaxSubmit 20 counts array tasks. `submit_pilot.sh` refuses if `squeue` count + 16 > 20.
- Read-only for this pilot: `labels_all.json`, `label_true/**`, `analysis/b3lyp_full/**`,
  `analysis/espley_xtb_repro/**`, the D0 raw data and the D0 scratch trees. The pilot imports repo
  code from those folders; it never writes there.
- Heavy output only under `scratch` (`/gpfs/tmp_cpu2/yeseo1ee/d1_autode_pilot`), never committed.
- Method parameters are frozen (`config.yaml` → `ts_stage`, `sp_stage`, `gates`). Changing a level of
  theory, a keyword, a threshold or the reference rule is a user decision, not an executor decision.
- A stage that failed leaves `.fail_<stage>` with its reason and is never retried automatically.
  Retrying means deleting that marker, and the reason must be written into the report.
- Do not modify autodE. If autodE misbehaves, stop and report (§8).

---

## 3. Inputs, environment, placement

### 3.1 Bundle

| file | role |
|---|---|
| `SPEC_D1_pilot.md` | this document |
| `config.yaml` | host paths (editable) + frozen method/gate parameters |
| `pilot_manifest.csv` | the 14 pilot jobs (§6), built by `sample_pilot.py`, seed 20260929 |
| `manifest_s0.csv` | the S0 smoke job (HCNO + C2H4 → 2-isoxazoline) |
| `sample_pilot.py` | rebuilds the manifest from `D1_설계_v8.xlsx` (not needed to run) |
| `pilot_common.py` | config, markers, autodE configuration (Coley protocol) |
| `stereo_utils.py` | dipole backbone, dihedral tracking, configuration tags |
| `run_ts.py` | stage `ts` (autodE profile, or D0 replay) + TS checks |
| `make_reference.py` | stage `ref` (port of Coley's stereo-compatible dipole reference) |
| `build_sp_inputs.py` | stage `inputs` (the repo's D0 builder text) |
| `assemble.py` | stage `label` (repo `stage3_parse` + repo `smd_relabel_audit.check`) |
| `report.py` | aggregation, acceptance, cost |
| `pilot_env.sh` | env block (copied from `smd_relabel_worker.sh`) + `run_stage` / `run_sp` |
| `run_job.sh`, `s0_preflight.sh`, `run_report.sh`, `submit_pilot.sh` | SLURM entry points |
| `s0_check_smoke.py`, `tests/*.py` | S0 checks (`fake_orca_outputs.py` is local-test only; do not run on the cluster) |

### 3.2 S-1: environment (login node: read-only checks only)

Required in the env named by `conda_env` (default `reactot`): **autodE 1.4.5** (checked against commit
`e7e71b33`, tag v1.4.5), RDKit ≥ 2022.09 (for `rdDetermineBonds`), networkx ≥ 3, numpy, pandas, PyYAML.
Required binaries: xtb 6.7.1 (`xtb_bin`), ORCA 6.1.1 (`orca_bin`), OpenMPI 4.1.7a1 (`mpi_root`).

The check itself is python, so it runs inside S0 (`versions` step). Before submitting, you may check
with a one-line sbatch:
`sbatch -p cpu2 --time=48:00:00 -c 1 --mem=2G --wrap "source <conda_sh>; conda activate reactot; python -c 'import autode; print(autode.__version__)'"`.

**If autodE is missing or the version differs, STOP and ask the user.** Proposed remedy, for approval:
clone the env (`conda create -n d1ade --clone reactot`, inside an sbatch job), then install autodE
v1.4.5 from a `git clone --branch v1.4.5 https://github.com/duartegroup/autodE` checkout. The clone
itself is a one-shot login-node command. Install with `pip install --no-deps .` in the clone, again
inside sbatch; it needs Cython and a C++ compiler for the extensions. Then set `conda_env: d1ade`.
Do not install into `reactot` without explicit approval, because `reactot` runs the label and ML
pipelines.

### 3.3 `config.yaml` host block

Check that these paths exist, and edit only if they differ:
`repo_root`, `scratch`, `conda_sh`, `conda_env`, `orca_bin`, `xtb_bin`, `mpi_root`, `d0_profiles`,
`d0_csv`, `d0_inputs_standalone`, `d0_inputs_eda_smd`.

`g16_bin` stays `null` unless the user decides to switch the TS stage to Gaussian 16 (§10).

---

## 4. The chain (one array task per job; every stage is idempotent)

```
run_job.sh  (8 cores, 32 GB, 48 h)
  ts      run_ts.py          autodE calculate_reaction_profile(free_energy=True)   -> ts.xyz, r_*_autode.xyz, ts_result.json
  ref     make_reference.py  Coley-port dipole reference (+ dipolarophile check)    -> ref_dipole.xyz, ref_dipolarophile.xyz, ref_result.json
  inputs  build_sp_inputs.py graph partition + repo builder text (5 inputs)         -> sp/*.inp, sp/input_meta.json
  sp      run_sp (bash)      5 ORCA SPs in parallel, serial each                    -> sp/*.out (+ eda_frag{1,2}.out by ORCA)
  label   assemble.py        stage3_parse.parse_one_rxn + D0 audit + uniformity     -> label.json
```

- Each stage writes `.done_<stage>` last, or `.fail_<stage>` with the reason.
- `run_stage` skips done stages, refuses failed ones, and appends wall time to `timing.tsv`.
- A stage killed at the 48 h wall leaves no marker, so it resumes on resubmit. autodE checkpoints its
  profile steps (`checkpoints/`), and `run_sp` skips outputs that terminated normally.

**Hard failures (the job stops at that stage)**

| stage | failures |
|---|---|
| ts | autodE exception; no TS; n_imag ≠ 1; ν_imag > −40 cm⁻¹; graph partition ≠ ok; forming-bond audit impossible; foreign inter-fragment bond; forming bond < 1.6 Å; imaginary-mode forming-bond share < 0.5 |
| ref | ring count changes in the alternative dipole (Coley drops such profiles); xtb scan unparsable; TS vs reference configuration of the dipolarophile's reacting C=C differs |
| inputs | partition with the references ≠ ok; odd electron count; method-line gate (`check_method_lines`) |
| sp | any of the 7 outputs not terminated normally |
| label | parser status ≠ ok after promotion; any gate false (§8 A2 list) |

**Flags only (reported, not failed):** asynchronous TS (a forming bond > 3.0 Å, as in D0); ambiguous
regiochemistry (crossed terminus pairing < 0.3 Å longer than the SMILES pairing); d1 or d2
< −0.5 kcal/mol; Pauli ≤ 0; OI ≥ 0; audit flags.

---

## 5. Method definitions and the evidence behind them

### 5.1 TS stage — Coley 2023 protocol, run on ORCA

Stuyver, Jorner & Coley, *Sci. Data* 10, 66 (2023), doi:10.1038/s41597-023-01977-8. They used
autodE (Young et al., *Angew. Chem. Int. Ed.* 2021, doi:10.1002/anie.202011941) with Gaussian 16 at
B3LYP-D3(BJ)/def2-SVP (opt/TS/freq) and def2-TZVP (SP), SMD water, `hmethod_conformers = False`
and `calculate_reaction_profile(free_energy=True)`. Their G16 keyword lists (from
`high_throughput_reaction_profiles/lib/input_generation.py`) map to our ORCA settings as follows:

| Coley (G16) | pilot (ORCA 6.1.1 via autodE) |
|---|---|
| `B3LYP def2svp EmpiricalDispersion=GD3BJ` | `B3LYP D3BJ def2-SVP` (+ autodE default `RIJCOSX def2/J`) |
| SP `def2tzvp` | SP `def2-TZVP` |
| `SCRF=(SMD,Solvent=water)` | `CPCM(Water)` + `%cpcm smd true SMDsolvent "water" end` |
| `Opt=(TS,CalcFC,NoEigenTest,MaxCycles=100,MaxStep=10,NoTrustUpdate,RecalcFC=30)` | `OptTS Freq` + `%geom Calc_Hess true Recalc_Hess 30 Trust -0.1 MaxIter 100 end` |
| `hmethod_conformers = False` | same |

`Trust -0.1` is a fixed 0.1-bohr step, the same as G16 `MaxStep=10` with `NoTrustUpdate`.

We checked that autodE writes exactly these inputs by generating OptTS/Opt/SP inputs locally with
`pilot_common.configure_autode` (no ORCA run). S0 checks the same strings on the cluster.

Known differences from D0 (quantified by the D0_control jobs, not removed):
- autodE 1.4.5 vs 1.2.
- xtb 6.7.1 vs 6.3.
- ORCA RIJCOSX/SMD vs G16 SMD geometries.
- Stochastic conformer sampling. Coley report that two autodE runs of the same SMILES, with
  GFN2-xTB conformer selection (the setting used here), reproduce activation energies with
  MAE ≈ 0.7 and RMSE ≈ 1.1 kcal/mol (azide test set, Fig. 10e).

### 5.2 SP stage and labels — identical to D0 by construction

- `build_sp_inputs.render()` calls the repo's `build_inputs_graph.eda_header/frag_header/write_block`
  and `smd_relabel_build_inputs.rewrite`. Only the `%pal nprocs` line of `eda.inp` comes from config.
- Labels come from `stage3_parse.parse_one_rxn` (method ①: d from standalone `frag*_dist`; c_ghost
  kept as `bsse_shift_kcal`). Gates come from `smd_relabel_audit.check`, pointed at the job's `sp/`
  through a symlink view.
- Local verification:
  - The pilot path reproduces the repo builder byte for byte on D0 rxn 0, 20, 105 and 3232.
  - On synthetic outputs built from `labels_all` energies, `assemble.py` returns the `labels_all`
    values to 3 × 10⁻⁶ kcal/mol, with every gate true.
- S0 repeats the parity test against the real D0 input files on disk.

### 5.3 Strain references — why a stereo rule is needed, and what it is

**Fact 1 (autodE source, v1.4.5).** `Reaction.locate_transition_state` switches reactants and
products when the product has more bonds, as in an addition. So a cycloaddition TS is searched from
the product side. Coley state the same: "In the case of addition reactions, the product side
stereochemistry is selected by default" (p. 8).

Consequences:
- The dipole's in-plane configuration at the TS (E/Z about C=X, W/U/S shape) comes from the product
  conformer, not from the reactant SMILES.
- **A reactant-side E/Z tag does not select the TS isomer.** An earlier draft of this pilot expanded
  E/Z; that is therefore dropped.
- One job per TS row, with the workbook's policy "미지정 — autodE 최저 입체 선택 (Coley 규약)".

**Fact 2 (validation on all 5,260 D0 ok reactions, `tests/validate_config_on_coley.py`).**
- 3,575 have trackable dipole backbone dihedrals.
- In 2,509 of them (70 %), the plain reactant dipole's configuration differs from the TS dipole's.
- Coley's `r*_alt.xyz` exists for 2,940 of them and matches the TS in 2,777 (94.5 %).
- 0 undefined tags on TS geometries.
- All 6 D0 records with d1 < −0.5 kcal/mol (24, 25, 788, 1271, 1479, 4111) have a dipole reference
  whose configuration differs from the TS.

So the reference rule decides d1 for most allyl-type dipoles. It must be the rule D0 used.

**The rule** is a port of Coley's `postprocess_reaction_profiles.py` + `finalize_reaction_profiles.py`.
Steps 1–7 are in the `make_reference.py` docstring:
1. xtb-optimise the TS-cut dipole and the original dipole.
2. Pick the dihedrals about the two backbone bonds.
3. Scan each with xtb (ALPB water, k = 0.1, 60 steps / 360°). A bond is *rotatable* iff
   |E₀−E_end| ≤ 5, max bias ≤ 20 and 2 ≤ barrier ≤ 20 kcal/mol.
4. Run 1,000 randomise-and-relax conformers, with unrotatable dihedrals frozen by 4 distances each.
5. `to_run` if anything was frozen, else only if ΔE_xtb < −0.1.
6. DFT opt + freq + SP of the alternative.
7. Use it if frozen and heavy-atom RMSD > 0.05 Å, or if unfrozen and ΔG < −0.239 kcal/mol. A
   ring-count change drops the job.

Deviations (each recorded in `ref_result.json`):

| id | deviation | reason |
|---|---|---|
| D-1 | Dipole poles taken from the new ring, not from formal charges | Coley's charge loop keeps the *last* +1/−1 atoms, i.e. a nitro N⁺/O⁻ when the dipole carries NO₂; pilot rows J03 and J09 do |
| D-2 | A dihedral through an sp centre (angle > 165°) is not tracked | undefined dihedral |
| D-3 | Step 6 is run even when `to_run` is False (diagnostic only; never changes the label) | shows how often Coley's xtb gate hides a lower, TS-compatible reference |

Checked locally with xtb standing in for DFT:
- xtb 6.7.1 prints the two lines Coley's parser reads (`unbiased energy:`, `    bias energy:`),
  60 each.
- On D0 rxn 20, the port freezes the C=N⁺ dihedral (xtb barrier 51 kcal/mol), gives RMSD 0.45 Å,
  uses the alternative, and so agrees with Coley.
- On D0 rxn 105 (cyclic dipole, nothing to track), it keeps the original, and so agrees with Coley.
- The DFT-level agreement is A4.

The dipolarophile reference is autodE's reactant, unchanged, as in Coley (they fix dipolarophile
stereo through product stereotags). The TS-vs-reference configuration of its reacting C=C is a hard
check.

### 5.4 TS checks — thresholds validated on D0

All numbers below come from Coley's `TS_imag_mode.xyz` over all 5,260 D0 ok TSs.

| check | D0 evidence | use |
|---|---|---|
| Imaginary-mode forming-bond share (top-2 normalised) | min 0.546, 0.1st percentile 0.640, median 1.000 | hard gate at 0.5 |
| Crossed-pairing regio margin | 4/5,260 < 0, 1st percentile 0.885 Å | flag at 0.3 Å (autodE already checks that the mode forms the product bonds) |

---

## 6. The pilot set (`pilot_manifest.csv`)

**Draw.** `sample_pilot.py`, seed 20260929:
- Stratified, 2 TS rows per design panel, from the backbone pool: 1,560 cells / 2,846 TS rows with no
  '복제' substituent role.
- Panels: P1 A-grid 2,465, P2 B-aryl 80, P3 C1–6 96, P4 C7–10 64, P5 D 141.
- `--force-stereo 1` guarantees one A1/A3 row with an unspecified dipole C=X, so the reference rule
  is exercised on the A grid.

| job | kind | TS id | core | panel | regio | atoms | dipole C=X unspecified | reaction SMILES (autodE input) |
|---|---|---|---|---|---|---|---|---|
| J00 | D1 | D1-2823-r1 | A5 | P1 | r1/2 | 38 | 0 | `CO[N-][N+]#N.C=CC12CC3CC(CC(C3)C1)C2>>CON1CC(C23CC4CC(CC(C4)C2)C3)N=N1` |
| J01 | D1 | D1-3422-r2 | B2 | P2 | r2/2 | 40 | 0 | `[O-][N+]#Cc1ccccc1.C#Cc1ccc(C(C)(C)C)cc1>>…` |
| J02 | D1 | D1-3449-r2 | B3 | P2 | r2/2 | 29 | 0 | `N#Cc1ccc(C#[N+][O-])cc1.C#Cc1ccccc1>>…` |
| J03 | D1 | D1-3499-r1 | C2 | P3 | r1/1 | 15 | 1 | `[CH2-][O+]=C[N+](=O)[O-].C=C>>O=[N+]([O-])C1CCCO1` |
| J04 | D1 | D1-3556-r1 | C4 | P3 | r1/1 | 24 | 1 | `CN(C)C=[N+](C)[NH-].C=C>>CN(C)C1CCNN1C` |
| J05 | D1 | D1-3667-r1 | C8 | P4 | r1/1 | 48 | 0 | `N#[N+][N-]C12CC3CC(CC(C3)C1)C2.C1#CCCCCCC1>>…` |
| J06 | D1 | D1-3693-r1 | C9 | P4 | r1/1 | 32 | 0 | `[O-][N+]#CC12CC3CC(CC(C3)C1)C2.C#C>>…` |
| J07 | D1 | D1-3842-r1 | D5 | P5 | r1/2 | 35 | 0 | `C[N-][N+]#N.CN(C)C1C#CCCCCC1>>…` |
| J08 | D1 | D1-3740-r1 | D1 | P5 | r1/2 | 17 | 0 | `C=[N+](C)[O-].C#CCC#N>>CN1CC(CC#N)=CO1` |
| J09 | D1 | D1-1417-r1 | A3 | P1 | r1/2 | 31 | 1 | `[CH2-][N+](C)=C[N+](=O)[O-].C=CC(=O)c1ccccc1>>…` |
| J10 | D0_control | rxn 20 | — | — | — | 25 | 1 | autodE from SMILES; Coley used `_alt` (plain `c`, TS `t`) |
| J11 | D0_control | rxn 105 | — | — | — | 27 | 0 | autodE from SMILES; cyclic dipole, no `_alt` |
| J12 | D0_replay | rxn 20 | — | — | — | 25 | 1 | Coley TS + Coley references → SP → label |
| J13 | D0_replay | rxn 105 | — | — | — | 27 | 0 | same |

**What each control isolates**

| kind | isolates | expected |
|---|---|---|
| D0_replay | inputs → SP → parse → gates, on D0 geometry and D0 references | = `labels_all` within SCF noise (≤ 0.1 kcal/mol; D0 saw ≤ 1.1 × 10⁻⁴ Eh SCF differences) |
| D0_replay port diagnostic | the reference rule at DFT level | same alt decision as Coley |
| D0_control | TS search + reference rule + ORCA-vs-G16 geometry, end to end | within ~1 kcal/mol (Coley's own rerun reproducibility is 0.7 / 1.1 kcal/mol MAE / RMSE) |

**Caveat to report, not fix.** The B block in `D1_설계_v8.xlsx` is not the PPT plan (80 cells / 160 TS
with meta/para). The pilot samples the workbook as it is.

---

## 7. Execution

### 7.1 Before submitting (login node, one-shot)

```bash
cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/d1_autode_pilot
ls "$(grep -E '^orca_bin:' config.yaml | awk '{print $2}')"     # binaries and paths exist
squeue -u $USER -h | wc -l                                        # must be <= 4
sinfo -p cpu1,cpu2 -o "%.9P %.6t %C"
```

### 7.2 Submit (one command, returns immediately)

```bash
bash submit_pilot.sh
# -> partition=<idle>  S0=<jid>  ARRAY=<jid> (0-13%10)  REPORT=<jid>  tasks=16
```

### 7.3 Monitoring

Monitor with one-shot commands only, when the user asks or at natural checkpoints. Never with a loop.

```bash
squeue -u $USER
sacct -j <ARRAY> --format=JobID,State,Elapsed,NodeList
cat /gpfs/tmp_cpu2/yeseo1ee/d1_autode_pilot/s0/S0_REPORT.md
ls /gpfs/tmp_cpu2/yeseo1ee/d1_autode_pilot/jobs/J*/.{done,fail}_* 2>/dev/null
```

### 7.4 Resubmitting a single job

Use this after a wall kill, or after an approved retry. `idx` is the manifest row index (J00 = 0):

```bash
SCR=/gpfs/tmp_cpu2/yeseo1ee/d1_autode_pilot
sbatch -p <idle> --array=<idx> -o $SCR/logs/job_%A_%a.log --export=ALL,PILOT_DIR=$PWD run_job.sh
```

Then rerun the report:

```bash
sbatch -p <idle> -o $SCR/logs/report_%j.log --export=ALL,PILOT_DIR=$PWD run_report.sh
```

### 7.5 After the report

- Commit to a branch:
  - the bundle code;
  - `pilot_manifest.csv`;
  - `results/{pilot_report.md,pilot_summary.csv,labels_d1_pilot.json,S0_REPORT.md}`.
- Commit message:

  ```
  d1 autodE pilot: <n_ok>/10 D1 labels, controls <PASS/FAIL>
  ```

- Nothing from `scratch`.

---

## 8. Acceptance and stop conditions

### Acceptance (computed by `report.py`)

| id | criterion | threshold |
|---|---|---|
| A1 | D1 TS rows ending with an ok label | ≥ 8 / 10. Coley's autodE + post-processing failure rate was 12.3 % |
| A2 | every produced label passes all gates | see the gate list below |
| A3 | D0_replay reproduces `labels_all` | 9 targets, max \|Δ\| ≤ 0.1 kcal/mol |
| A4 | the Coley-port decision equals Coley's `alt_used` | both replay controls |
| A5 | D0_control \|Δbarrier\| | ≤ 1.0 kcal/mol. **Diagnose, not a hard stop.** Report the TS RMSD, the forming-distance Δ, the reference choice and ΔG‡ vs `G_act` |

The A2 gates are the D0 relabel gates:
- g1 identity ≤ 1e-6;
- Bond-reference residual ≤ 0.02;
- \|Σ6ch − Bond\| ≤ 0.11;
- 8-channel closure ≤ 0.11;
- D0 audit problems = none;
- uniformity: ORCA 6.1.1, SMD, water, ε = 78.3550, CDS present, terminated.

### Stop conditions (stop, report, wait for the user)

| trigger | what to report | options for the user |
|---|---|---|
| S0 `versions` fails (autodE missing or ≠ 1.4.5, RDKit too old) | the `versions` block | §3.2 remedy |
| S0 smoke fails in `ts` with an autodE parse error on ORCA 6 output | the traceback, the offending `*_orca.out` section | (i) G16 engine if it is on the cluster (exact Coley parity); (ii) a minimal autodE ORCA-parser patch shown as a diff; (iii) ORCA 5.0.x for the TS stage only (SPs stay 6.1.1) |
| S0 parity fails | the diff | nothing runs until explained |
| S0 config validation fails (errors, or `_alt` agreement < 0.90) | the counts | — |
| ≥ 3 D1 jobs fail at the same stage for the same reason | the reasons, and one representative log | do not resubmit |
| A3 fails | the per-target Δ | the SP/label half differs from D0: this blocks D1 |
| any wish to change `ts_stage` / `sp_stage` / `gates` or the reference rule | — | user decision |

---

## 9. What to hand back (paste into the reply, in this order)

1. Acceptance table (A1–A5) with PASS/FAIL, and one line of cause for every FAIL.
2. Failures: job, stage, reason, and whether it is a TS-search failure or a pipeline bug.
3. Labels table for the ok jobs (9 targets + autodE ΔE‡(sp), ΔG‡). Note any flag: async, regio,
   d < −0.5, Pauli ≤ 0, OI ≥ 0.
4. Controls table: Δ(label − `labels_all`) for replay and control jobs; Δd_form; ΔG‡ − `G_act`;
   alt decision (Coley / ours).
5. Configuration/reference table, for the jobs with tracked dihedrals:
   - TS config vs reactant config;
   - frozen dihedrals;
   - `to_run`, alt used;
   - what the lowest-compatible rule (D-3) would have done.
   One paragraph: what this implies for the 1,022 D1 backbone rows with an unspecified dipole C=X.
6. Cost: core-hours per stage and per job (mean / median / max), the extrapolation to 2,846 TS rows,
   and the wall time at MaxJobs = 10.
7. Go / no-go recommendation for the full D1 run, and open decisions (§10).

---

## 10. Open decisions (for the user; the pilot informs them)

1. **TS engine.** ORCA (default, available) or G16 (exact Coley parity, if licensed on UBAI). A5 and
   the D0_control geometry deltas are the evidence.
2. **Reference rule for D1.**
   - Coley-faithful (default; parity with D0), or
   - lowest TS-compatible (D-3), which would remove the negative-d cases. Choosing it means
     re-referencing D0 too, or the two datasets stop being comparable.
3. **B block.** Regenerate it to the PPT plan before the full run (the pilot uses v8 as it is).
4. **`eda_nprocs`.** It is 1 for parity with the relabel runs, and the energies do not depend on it.
   Raise it only if EDA wall time dominates.

---

## 11. References (DOIs verified)

- Stuyver, T.; Jorner, K.; Coley, C. W. Reaction profiles for quantum chemistry-computed [3+2]
  cycloaddition reactions. *Sci. Data* **2023**, 10, 66. doi:10.1038/s41597-023-01977-8.
  Protocol, keywords, 12.3 % failure rate, stereo-compatibility post-processing, reproducibility.
- Young, T. A.; Silcock, J. J.; Sterling, A. J.; Duarte, F. autodE: Automated Calculation of Reaction
  Energy Profiles — Application to Organic and Organometallic Reactions. *Angew. Chem. Int. Ed.*
  **2021**, 60, 4266. doi:10.1002/anie.202011941.
- Code (the version used, the file that holds each rule):
  - autodE v1.4.5 (commit e7e71b33): `autode/reactions/reaction.py::locate_transition_state`.
  - coleygroup/dipolar_cycloaddition_dataset:
    - `high_throughput_reaction_profiles/lib/input_generation.py`
    - `postprocess_reaction_profiles/postprocess_reaction_profiles.py`
    - `postprocess_reaction_profiles/finalize_reaction_profiles.py`
