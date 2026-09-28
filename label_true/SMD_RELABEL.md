# SMD relabel and method-① label assembly (2026-09-15 → 2026-09-28)

This rebuilt the repo-root `labels_all.json` (5,265 records). The scripts are in `label_true/scripts/`. Every compute step ran through sbatch. Scratch data lives at
`/gpfs/tmp_cpu2/yeseo1ee/eda_smd_relabel/` (not committed).

## 1. Why

- **The fault.** In the old runs, SMD was enabled with `CPCM(water)` plus a `%cpcm smd true smdsolvent "water" end` block. ORCA does not copy that block into the fragment inputs it generates for EDA. So the fragment SCFs of every EDA run (`eda_frag1.out`, `eda_frag2.out`) had no SMD-CDS term.
- **The consequence.** Bond Energy and the 6 EDA channels were therefore defined against fragments solvated at a different level from the supermolecule.

## 2. What was recomputed

- **Re-run for all 5,265 reactions:** only `eda.inp` (and so `eda.out`, `eda_frag1.out`, `eda_frag2.out`).
- **The fix.** `smd_relabel_build_inputs.py` puts the `SMD(water)` keyword in the `!` header and in the FRAG1/FRAG2 strings, and drops the `%cpcm` block.
  - Every new input is byte-identical to this rewrite applied to the old input.
  - The method is otherwise unchanged: B3LYP D3BJ def2-TZVP NoSym TightSCF, ORCA 6.1.1, `%pal nprocs 1`, `%maxcore 3500`.
- **Not re-run:** `frag{1,2}_dist.out` and `frag{1,2}_rel.out` in `label_true/work/inputs/`. They are standalone runs, where the `%cpcm` block does apply, so they already use the same SMD model.

## 3. How it was run

- **`smd_relabel_worker.sh` (v5)** is a pool of SLURM workers.
  - Each worker claims a reaction atomically with `mkdir`, and a heartbeat keeps the claim fresh.
  - A claim is taken over only when its owner job has left squeue.
  - Every reaction starts clean, and the worker rescans the list after each reaction.
  - Each worker submits a successor without a dependency. The successor waits on AssocMaxJobsLimit until a slot frees.
  - Nodes that fail fast are flagged and excluded.
- **Other accounts on UBAI.** Part of the work ran there as self-contained bundles.
  - Built with `smd_bundle_*.py`/`.sh`; each bundle is started with one `launch.sh`.
  - Returned archives were extracted on a compute node (`smd_bundle_extract.sh`).
  - Before merging, every returned reaction was checked: complete, SMD input, no clash with local reactions.
  - Reactions that came back unfinished were re-run locally from `eda.inp` alone.

## 4. Audit (all passed; `smd_final_pipeline.sh` steps 1–3)

- **`smd_relabel_audit.py`** runs per reaction.
  - File checks: all files present; every output terminated normally with exactly one FSPE; no SCF non-convergence.
  - Input checks: the input matches the rewrite; the eda_frag / frag_dist / frag_rel charge, multiplicity, coordinates and method lines are consistent. frag_dist coordinates equal the TS fragment coordinates.
  - It also evaluates the gates.
- **`smd_relabel_uniformity.sh`** checks that every output feeding a label is uniform: eda, eda_frag1/2, frag1/2_dist, frag1/2_rel. All of them show ORCA 6.1.1, the SMD module, the SMD CDS term, water, ε 78.3550, and normal termination.
- **`smd_relabel_gate.py pre`** gives the verdict.

| Gate | Result |
|---|---|
| (1) d1 + d2 + eint_spe = barrier | 5265/5265 (max 8.6e-10 kcal/mol) |
| (2) Bond Energy = E(AB) − E(eda_frag1) − E(eda_frag2) | 5265/5265 (max 3.2e-8) |
| (3) d1 + d2 + Σ6ch + c_ghost = barrier | 5209 within 0.03. The other 56 are ORCA's own EDA-table closure (rows − Bond Energy, max 0.108 kcal/mol, RIJCOSX/grid); the same residual appears in the old runs. |

An independent hand recomputation of 36 sample reactions from the raw ORCA files matched the pipeline to ≤ 5e-13 kcal/mol.

## 5. Assembly: method ① (`stage3_parse.py`, `smd_stage3_build.py`, `postprocess_labels_all.py`)

| Quantity | Definition |
|---|---|
| d1, d2 | E(frag_dist, own basis, own cavity) − E(frag_rel): pure deformation energies |
| eint_spe | E(AB) − E(frag1_dist) − E(frag2_dist), so d1 + d2 + eint_spe = barrier exactly |
| e_bond, 6 channels | ORCA EDA table. It refers to eda_frag*.out, which uses the ghost basis and the AB cavity. |
| `bsse_shift_kcal` (= c_ghost) | eint_spe − e_bond, so d1 + d2 + e_bond + c_ghost = barrier exactly |

**Rejected options:**
- ② counterpoise-correct only the interaction: then d1 + d2 + bond ≠ barrier.
- ③ take d1/d2 from eda_frag*.out (the 2026-09-15 build): the 8-channel sum closes, but d1/d2 then contain ghost-basis and AB-cavity terms.

**c_ghost is not pure BSSE.** It is the ghost-basis BSSE (median −0.9 kcal/mol per fragment) plus the change of moving the fragment into the AB cavity (median +0.9, up to +8.4).

## 6. Output

`labels_all.json` has 5,265 records.
- **ok: 5,260.** 167 of them had |Σ6ch − Bond| > 0.02 and were promoted, with `sum_mismatch_promoted` set and the residual kept.
- **excluded: 5.**
  - 3090, 3766, 4252: `flag_foreign_bond`.
  - 3400, 5783: `oi_dft>0`. That reason came from the CPCM-era labels; with SMD, oi_dft = −2.84 and −3.07, so `exclusion_note` asks for review.

**Other notes:**
- In 212 reactions the supermolecule energy differs from the old run by up to 1.1e-4 Eh. This is SCF convergence noise, and the new runs are better converged.
- 9 ok records have d1 or d2 < −0.5 kcal/mol, in both the old and new standalone references. This is a relaxed-geometry issue in the dataset.

## 7. Rebuild

```bash
sbatch label_true/scripts/smd_final_pipeline.sh   # audit -> uniformity -> gate -> stage3 -> postprocess -> crosscheck
```

The script writes `labels_all.json` only if the gate and the crosscheck both pass. The reports go to `.../eda_smd_relabel/audit/GATE_REPORT.txt`.
