#!/usr/bin/env python3
r"""gen_dft.py — DEPLOY-V2 (MLTS) Phase G: label-level DFT energies + forces for MLIP fine-tuning
(docs/specs/DEPLOY_V2_MLTS.md 3; budget 0-5). sbatch only: python and ORCA never run on the login node.

Level (spec 3): mlt_common.orca_grad_input = the D1 autodE 'grad' keywords ('! EnGrad B3LYP RIJCOSX D3BJ def2-SVP
def2/J CPCM(Water)' + %cpcm smd true SMDsolvent "water" + %scf maxiter 250, %maxcore 3000), ORCA 6.1.1, serial
(nprocs 1). Every finished EnGrad is checked: input echo at the label level (orca_io.label_level), SMD CDS line and
water epsilon 78.3550 in the output (a failure is systematic -> $MLT/dft/STOP, exit 3).

Groups
  G1  per D1 target, from the label TS $LAD/dft_final/<rxn>/ts.xyz and the xTB Hessian at exactly that geometry
      (work/ts.hess, ORCA XTB2 ALPB(water) Freq; no DFT TS Hessian survives -> the spec's xTB fallback):
        ts_near  x2  random real normal-mode displacements, harmonic energy E ~ U[0.5, 3] kcal/mol
        ts_imag  x1  along the imaginary mode, mean change of the two forming bonds = +-U[0.05, 0.15] A
        ts_label     the undisplaced label TS (only the 10 G-0 parity structures, written by 'g0')
  G2  400 reactions (G2 ok minus the 60 G3L / Part B reactions of Phase I, stratified dipole x g1 class): label-level
      OptTS from the G2 TS with the G3L input (analysis/d1_g3/g3l.g3l_text, imported); every optimisation step with an
      energy and a gradient = one record (g2_path; the last step of a converged run = ts_final_g2)
  G3  per unique label reactant molecule (LADDER Q key) react_ref + 2 react_disp; per reaction 2 frag (TS cut with the
      D1 sp/eda.inp partition)

CLI (exit 0 = done / up to date, 2 = incomplete (resubmit) or usage, 3 = STOP; submit from analysis/mlts)
  plan [--limit N] [--n-g2 K] [--force]     MLTENV=reactot. $MLT/dft/plan/{plan.parquet, g2_plan.json,
                                            plan_meta.json}, results/G_plan.json (no QM)
  g0 [--n-rand 50] [--n-label 10] [--n-g2 2] [--label-rxns a,b]
                                            MLTENV=reactot, 16 cores: G-0 EnGrads (normal G records) + the G-2 pilots
                                            (run_g2 in a MLTENV=d1 subprocess, 2 x 8 ranks) -> results/G0_estimate.json;
                                            exit 3 if the prediction > 5,000 core-h (spec STOP) or the parity is broken
  run <G1|G3> <i|@I> <n> [--only ids]       MLTENV=reactot, 64-core cpu2 array element: slice i of n
  run_g2 <i|@I> <n> [--only rxns]           MLTENV=d1, 64-core cpu2 array element: 8 OptTS x 8 MPI ranks
  g2_check [--rxns a,b]                     MLTENV=d1: the G-2 input builder against every G3L unit input
                                            ($G3/O/units/L.*/optts/ts.inp) + the step parser on their ts.out.gz ->
                                            results/G2_check.json
  g2_probe <rxn> [--maxiter 2]              MLTENV=d1, 8 cores, diagnostic: first OptTS cycles with g2.hess and with a
                                            rigidly translated copy (identical = ORCA ignores the .hess $atoms
                                            coordinates; decision G-2sel) -> results/G2_probe_<rxn>.json
  collect                                   MLTENV=reactot: $MLT/dft/gen_{G1,G2,G3}.extxyz + results/G_report.json
                                            (D1 only: never reads the D0 plan or D0 records)

D0 extension (user decision 'D0' 2026-11-04, results/DECISIONS.md; code in gen_d0.py, decisions gen_d0.DECISIONS_D0):
the Coley 5,269 D0 reactions (repo-root labels_all.json, 5,260 ok) go into the MLIP training data with the D1 rules;
a SEPARATE plan, separate records / Hessians / STOP file / outputs; evaluation stays on the D1 T1 folds.
  plan_d0 [--limit N] [--n-g2 auto|K] [--n-near 2|1] [--force]
                                            MLTENV=reactot: $MLT/dft/plan/{plan_d0.parquet, g2_plan_d0.json,
                                            g2_pool_d0.json, d0_class.csv, plan_d0_meta.json}, results/G_plan_d0.json
                                            (no QM). Re-running with another --n-g2 only changes D0G2 (prefix of the
                                            stratified ranking); auto = budget rule on the g0_d0 fits when present;
                                            --n-near may change after D0 records exist (the n1 / i rows are identical)
  g0_d0 [--n-rand 30] [--n-label 10] [--n-g2 auto|K] [--label-rxns a,b] [--max-atoms N]
                                            MLTENV=reactot, 16 cores: D0 G-0 -> results/G0_estimate_d0.json (exit 3 =
                                            D1 (max of G-0 and observed) + D0 prediction > 5,000 core-h, also reported
                                            for --n-near 1; D0 parity is recorded, never a stop; auto pilots: 1 D0G2
                                            pilot only if the plan selects D0G2 reactions)
  run <D0G1|D0G3> <i|@I> <n> [--only ids] [--rev] [--dry]     as run G1/G3, records $MLT/dft/runs/<D0G1|D0G3>/
  run_g2 <i|@I> <n> --d0 [--only rxns] [--dry]                 as run_g2, units g2_plan_d0.json, $MLT/dft/runs/D0G2/
  collect --d0                              MLTENV=reactot: $MLT/dft/gen_{D0G1,D0G2,D0G3}.extxyz +
                                            results/G_report_d0.json (D1 outputs untouched)

Outputs ($MLT = $MLT_SCRATCH, default /gpfs/tmp_cpu2/yeseo1ee/mlts)
  $MLT/dft/plan/plan.parquet   struct_id, group, kind, rxn_ids, folds, dipoles (leakage tags, mlt_common), n_atoms,
        n_heavy, charge, mult, symbols (space separated), pos_A (flattened A; for react_disp the reference geometry,
        pos_final False: the displacement is made in the run step from the molecule's xTB Hessian), src, hess_src,
        mol_id, mol_key, seed, disp (JSON: displacement record), rxn_id, fold_T1, g1_class
  $MLT/dft/runs/<G1|G3>/<struct_id>.json (+ .out.gz of the last attempt)   one record per structure, written last
  $MLT/dft/runs/G2/<rxn_id>/{unit.json, steps.json, starts.json, ts.inp, ts.out.gz, ts.xyz, ts.hess.gz}
  $MLT/dft/hess/<mol_id>.{hess.gz,json}     xTB Hessians of the G3 molecules (run step)
  $MLT/dft/steps/run_<group>_<i>of<n>.json  per array task summary;  $MLT/dft/STOP (level failure)
  results/G_plan.json, G0_estimate.json, G2_check.json, G_report.json

sbatch (login node: one-shot commands only; `squeue -u $USER -h | wc -l` first: MaxSubmit 20 tasks, MaxJobs 10;
64-core jobs only on cpu2). From /gpfs/home1/yeseo1ee/projects/eda-asm-prediction-mlts/analysis/mlts, L=$MLT/logs:
  JP=$(sbatch --parsable -p cpu1,cpu2 -J g_plan --time=48:00:00 --cpus-per-task=16 --mem=32G \
       --output=$L/%x_%j.out jobs/mlt.slurm gen_dft.py plan)
  J0=$(sbatch --parsable -p cpu1,cpu2 -J g_g0 --time=48:00:00 --cpus-per-task=16 --mem=64G \
       --dependency=afterok:$JP --kill-on-invalid-dep=yes --output=$L/%x_%j.out jobs/mlt.slurm gen_dft.py g0)
  (after G0_estimate.json go; n from its 'suggested_tasks')
  sbatch -p cpu2 -J g_run1 --time=48:00:00 --cpus-per-task=64 --mem=240G --array=0-5 --output=$L/%x_%A_%a.out \
       jobs/mlt.slurm gen_dft.py run G1 @I 6
  sbatch -p cpu2 -J g_run3 --time=48:00:00 --cpus-per-task=64 --mem=240G --array=0-1 --output=$L/%x_%A_%a.out \
       jobs/mlt.slurm gen_dft.py run G3 @I 2
  sbatch -p cpu2 -J g_run2 --time=48:00:00 --cpus-per-task=64 --mem=240G --array=0-1 --output=$L/%x_%A_%a.out \
       --export=ALL,MLTENV=d1 jobs/mlt.slurm gen_dft.py run_g2 @I 2
  sbatch -p cpu1,cpu2 -J g_collect --time=48:00:00 --cpus-per-task=4 --mem=32G --dependency=afterany:<run jobs> \
       --output=$L/%x_%j.out jobs/mlt.slurm gen_dft.py collect
  Resubmission is idempotent (records are skipped; n may even change: slices are recomputed, records are per
  structure). A task stops claiming when < 2 h (run) / 6 h (run_g2) of the 48 h wall remain and exits 2.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True        # nothing compiled into the read-only code trees (d1_g3, d1_ladder, D1 copy)

import argparse  # noqa: E402
import gzip  # noqa: E402
import hashlib  # noqa: E402
import heapq  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from collections import Counter  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mlt_common as M  # noqa: E402
import orca_io as OI  # noqa: E402

sys.modules.setdefault("gen_dft", sys.modules[__name__])   # gen_d0 imports this module: one copy when run as __main__

CODE_VERSION = "mlts-gen-dft-1"
T_START = time.time()

# ---------------------------------------------------------------- paths
DFT = M.MLT / "dft"
PLAN_DIR = DFT / "plan"
PLAN_PQ, PLAN_META, G2_PLAN = PLAN_DIR / "plan.parquet", PLAN_DIR / "plan_meta.json", PLAN_DIR / "g2_plan.json"
RUNS, HESS_DIR, STEPS = DFT / "runs", DFT / "hess", DFT / "steps"
STOP_FILE = DFT / "STOP"
STOP_D1 = STOP_FILE                           # the D1 STOP path (STOP_FILE becomes STOP_D0 in a D0 process)
D1G3_SHADOW = DFT / "g2" / "_d1g3"            # D1G3_SCRATCH for the d1_g3 import (nothing is written there)
EXT = {g: DFT / f"gen_{g}.extxyz" for g in ("G1", "G2", "G3")}
G_PLAN_JSON, G0_JSON = M.RES / "G_plan.json", M.RES / "G0_estimate.json"
G_REPORT, G2_CHECK = M.RES / "G_report.json", M.RES / "G2_check.json"
G_STATUS = M.ANALYSIS / "d1_g2" / "results" / "G_status.csv"
G3_UNITS = M.G3 / "O" / "units"               # Phase I data: L.<rxn> (G3L, 20) and B.<rxn> (Part B, 40)
# D0 (gen_d0.py): separate plan / Hessian cache / STOP file / outputs, so the D1 plan, run and collect paths never see them
D0_GROUPS = ("D0G1", "D0G2", "D0G3")
PLAN_D0_PQ, PLAN_D0_META = PLAN_DIR / "plan_d0.parquet", PLAN_DIR / "plan_d0_meta.json"
G2_PLAN_D0, G2_POOL_D0, D0_CLASS_CSV = PLAN_DIR / "g2_plan_d0.json", PLAN_DIR / "g2_pool_d0.json", PLAN_DIR / "d0_class.csv"
HESS_D0 = DFT / "hess_d0"                    # xTB Freq at the D0 label TSs (d0ts_<rid>) and the D0 molecules (d0m...)
STOP_D0 = DFT / "STOP_D0"
EXT_D0 = {g: DFT / f"gen_{g}.extxyz" for g in D0_GROUPS}
G_PLAN_D0_JSON, G0_D0_JSON, G_REPORT_D0 = M.RES / "G_plan_d0.json", M.RES / "G0_estimate_d0.json", M.RES / "G_report_d0.json"
UNIT_KINDS = ("react_ref", "react_disp", "d0_react_ref", "d0_react_disp", "d0_ts_near", "d0_ts_imag")   # sliced by mol_id
D0_TS_DISP_KINDS = ("d0_ts_near", "d0_ts_imag")

# ---------------------------------------------------------------- constants
E_RANGE = (0.5, 3.0)              # kcal/mol, harmonic energy of a normal-mode displacement
DELTA_RANGE = (0.05, 0.15)        # A, mean forming-bond change of the imaginary-mode displacement
LOW_CM = 50.0                     # modes below this are not displaced
MAX_DISP_A = 0.3                  # no atom moves more than this in a normal-mode displacement
MAX_DRAWS = 50
IMAG_MIN_COEF = 0.2               # forming-bond coefficient (A per A of max atom motion) of an accepted imaginary mode
IMAG_CAND_CM = 300.0              # fallback candidates for the reaction mode
HESS_TOL_A = 1e-4                 # .hess $atoms vs the structure (after removing a common shift)
N_G2 = 400
G0_N_RAND, G0_N_LABEL, G0_N_G2 = 50, 10, 2
BUDGET_CORE_H = 5500.0            # spec 0-5: G total 5,000 core-h; above = STOP. Raised to 5,500 by the user's D0 decision
                                  # (2026-11-04 'D0도 반응당 3구조': D1 ~3,280 + D0 G1/G3 ~1,960 + G-4 ~100; DECISIONS 'D0-n3')
PARITY_EXPECT = 5e-4              # Eh/bohr, max |gradient| at the label TS (stationary at this level)
PARITY_STOP = 1e-3                # median above this = level mismatch (added guard)
G2_NPROCS = 8                     # g3l / Part B %pal (opt.PARTB_NPROCS, checked)
WALL_H = float(os.environ.get("MLT_WALL_H", "48"))
RESERVE_H = float(os.environ.get("MLT_RESERVE_H", "2"))
RESERVE_G2_H = float(os.environ.get("MLT_RESERVE_G2_H", "6"))
MAX_STARTS = 2                    # an OptTS started twice without finishing (48 h wall) is closed as wall_clipped
G2_FINAL_STEP_TOL_A = 0.05        # g2_check: last gradient step vs the converged geometry (one sub-threshold step)
MAX_PIPELINE_ERR = 20
XTB_HEADER = "! XTB2 ALPB(water) Freq\n%pal nprocs 1 end\n%maxcore 3000\n"   # = the LADDER G0 TS Hessian header
CDS_MARK, EPS_MARK = "SMD CDS free energy correction energy", "78.3550"
_EH_J, _BOHR_M, _AMU_KG, _C_CM = 4.3597447222071e-18, 5.29177210903e-11, 1.66053906660e-27, 2.99792458e10
AU2CM = float(np.sqrt(_EH_J / (_BOHR_M ** 2 * _AMU_KG)) / (2 * np.pi * _C_CM))   # sqrt(Eh/(bohr^2 amu)) -> cm-1
SOURCE = dict(ts_near="label_ts+xtb_nm", ts_imag="label_ts+xtb_imag", ts_label="label_ts", react_ref="d1_ref",
              react_disp="d1_ref+xtb_nm", frag="label_ts_frag", g2_path="g2ts_label_optts",
              ts_final_g2="g2ts_label_optts")
RE_SCF_OK = re.compile(r"SCF CONVERGED AFTER\s+\d+\s+CYCLES")
RE_SCF_BAD = re.compile(r"SCF NOT CONVERGED|NOT CONVERGED AFTER")
RE_SCF_N = re.compile(r"SCF CONVERGED AFTER\s+(\d+)\s+CYCLES")
RE_FSPE = re.compile(r"^FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", re.M)
RE_OUTFREQ = re.compile(r"^\s*\d+:\s+(-?\d+\.\d+) cm\*\*-1", re.M)
RE_EDA_ATOM = re.compile(r"^\s*([A-Z][a-z]?)\((\d+)\)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*$")

DECISIONS = {
    "G-hess": "TS normal modes from $LAD/dft_final/<rxn>/work/ts.hess (ORCA 6.1.1 XTB2 ALPB(water) Freq at the exact label "
              "TS; no DFT TS Hessian survives = spec fallback). Molecule Hessians: ORCA 6.1.1 '! XTB2 ALPB(water) Freq' "
              "(bundled otool_xtb = xtb 6.7.1; the same header as the TS Hessians), computed in the run step, cached in "
              "$MLT/dft/hess/<mol_id>.hess.gz (one per molecule, under a per-molecule lock; the 3 structures of a "
              "molecule are in one slice).",
    "G-modes": "normal modes: symmetrised Cartesian Hessian, mass weighted with the .hess masses, restricted to the "
               "orthogonal complement of the mass-weighted translations + rotations (SVD rank: 6, or 5 if linear), "
               "eigendecomposed; nu = sign(l) sqrt|l| x 5140.49 cm-1 (checked against the .hess frequencies).",
    "G-near": "ts_near / react_disp: modes with nu >= 50 cm-1 only (imaginary and soft modes skipped); amplitudes "
              "c_i = z_i / sqrt(l_i), z ~ N(0,1) (equal expected energy per mode), scaled so E = 1/2 sum l_i c_i^2 = "
              "E_target ~ U[0.5, 3] kcal/mol; a draw that moves any atom > 0.3 A is redrawn (<= 50 draws), then the "
              "draw with the smallest max displacement is scaled down to 0.3 A (capped; the smaller E_harm recorded).",
    "G-imag": "ts_imag: the xTB imaginary mode if there is exactly one and its forming-bond coefficient (mean over the 2 "
              "label forming bonds of the bond-length change per A of max atom motion) >= 0.2; otherwise the mode with "
              "nu < 300 cm-1 (imaginary included) with the largest |coefficient| (choice recorded). Step s solved "
              "(Newton on exact distances) so mean(d_k(s) - d_k(0)) = sign x U[0.05, 0.15] A, sign +-1 (p = 0.5).",
    "G-rng": "numpy default_rng(SEED + rxn_id) per reaction (draws: E1, E2, |delta|, sign, then the directions); "
             "default_rng([SEED, int(sha256(mol_key)[:12], 16)]) per molecule (E1, E2, directions); G-2 sample "
             "default_rng(SEED + 2); G-0 sample default_rng(SEED + 3). SEED = mlt_common.SEED.",
    "G-refs": "label reactant geometries = the D1 ref_dipole.xyz / ref_dipolarophile.xyz ($D1PAIR_SCRATCH/d1_jobs/T#####, "
              "as the spec says); $LAD/dft_final rel1 / rel2 are the D1-HF per-reaction copies of the same coordinates.",
    "G-dedupe": "unique molecule = the LADDER Phase Q molecule key of (reaction, fragment) as stored in labels_d1_cr2 "
                "cr.key1 / cr.key2 ('<kind>:' + RDKit canonical isomeric SMILES, enantiomers merged, constrained "
                "dipoles '|fixed:<state>'; refs_q D4). Geometry = the key member with the lowest original label "
                "e_frag{k}_rel_eh (ties: smallest job_id) = the Q:label_min reference of the cr2 labels. Tags = union "
                "over all targets with that key (mlt_common.tag_list).",
    "G-frag": "TS fragments: atoms tagged (1) / (2) in the D1 sp/eda.inp (authoritative; the tags are not the first n_f1 "
              "atoms in general), coordinates exactly as in the label TS, TS atom order; n_f{k} and the element "
              "sequence of the D1 reference are checked; charge = label charge{k}, multiplicity 1.",
    "G-2sel": "G-2 pool: G_status.csv status ok, a usable G2 xTB Hessian (g3l.hess_check rules: $G2/g2_final/<rxn>/.done, "
              "ts.xyz, work/ts.hess $atoms in ts.xyz order, dim 3N, XTB2 on the ts.inp '!' line; geometry within 1e-4 A "
              "AFTER removing a common translation — deviation from g3l's no-shift rule: 464 G2-ok Hessians are pure "
              "rigid translations of ts.xyz (residual <= 5e-8 A), which leave a Cartesian Hessian unchanged, and "
              "excluding them would under-sample the 'different' class; the raw shift is recorded), minus the "
              "L.* (G3L) and B.* (Part B) reactions of $G3/O/units. Strata dipole x same_ts_g1 (C_compare.csv True = "
              "same, False = different, blank = none); 400 allocated proportionally (largest remainder, ties in "
              "stratum order); uniform draw without replacement in each stratum.",
    "G-2run": "G-2 OptTS input = g3l.g3l_text(opt.d1(), G2 TS (label order), label charge, label forming bonds) "
              "(Part B text with InHess Read / InHessName \"g2.hess\", no Calc_Hess / Recalc_Hess, Trust -0.1, MaxIter "
              "100, %pal 8, %maxcore 3000, + Freq); g2.hess = byte copy of $G2/g2_final/<rxn>/work/ts.hess (sha256 from "
              "the plan); d1_g3 imported with D1G3_SCRATCH = $MLT/dft/g2/_d1g3 (nothing written under $G3); ORCA run "
              "by this file in node-local tmp with opt.d1() env (OpenMPI 4.1.8f, binding none); infra death (no SCF) "
              "rerun once; no reopt. The input builder is checked against a G3L unit input before every run_g2.",
    "G-2steps": "every OptTS geometry with an energy and a CARTESIAN GRADIENT (orca_io.read_out_steps; consecutive "
                "duplicates dropped) is a record whatever the run outcome; the last step of a converged run has kind "
                "ts_final_g2 (one record, not duplicated as g2_path): path = g2_path + ts_final_g2. ORCA's converged "
                "geometry (ts.xyz, energy-only final evaluation, no gradient) is one sub-threshold step past that last "
                "gradient step (G3L outputs: 0.001-0.011 A; recorded per unit as final_vs_last_step_A); no extra EnGrad.",
    "G-run": "run: the group's structures are packed into n slices by LPT on n_atoms^3 (a molecule's ref + 2 disp stay "
             "together); a slice runs largest first with SLURM_CPUS_PER_TASK serial ORCA workers in node-local "
             "$TMPDIR; no new structure when < 2 h of the 48 h wall remain (MLT_JOB_T0).",
    "G-retry": "EnGrad failure (no normal termination, SCF not converged, missing / inconsistent .engrad) -> one rerun "
               "of the same input, then a fail record with the reason (kept: not rerun on resubmission). Unexpected "
               "python errors write no record (> 20 per task = halt, exit 3).",
    "G-0": "G-0: 50 EnGrads drawn from the G1 + G3 plan, 10 per n_atoms quintile (seeded), + 10 undisplaced label TSs "
           "at the n_atoms quantiles 0.05 .. 0.95 (kind ts_label, kept in G1), + 2 G-2 pilots = the 'different' "
           "class reactions of the G-2 sample at its n_atoms tertiles (G3L covered only 'same'); all are normal records.",
    "G-0fit": "cost: EnGrad core-h = wall x 1, fitted core-h = a N^b (log-log least squares, ok G-0 EnGrads) summed "
              "over every planned G1 + G3 structure, + molecule Hessians at their measured mean, + the 10 ts_label; "
              "G-2 core-h = wall x 8, a N^b fitted on the 20 G3L units + the pilots, x max(1, mean pilot / fit); "
              "total > 5,000 core-h -> exit 3 (spec STOP).",
    "G-parity": "parity: max |gradient| of the undisplaced label TS (median expected <= 5e-4 Eh/bohr); median > 1e-3 = "
                "level mismatch -> exit 3 (guard added here; not a spec STOP).",
}


# ================================================================ small helpers
def _now():
    return time.strftime("%F %T")


def _cpus():
    return int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1)


def _jobtag():
    j, a = os.environ.get("SLURM_JOB_ID"), os.environ.get("SLURM_ARRAY_TASK_ID")
    return f"{j}_{a}" if j and a else (j or f"pid{os.getpid()}")


def _rj(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def time_left_h():
    t0 = float(os.environ.get("MLT_JOB_T0") or T_START)
    return WALL_H - (time.time() - t0) / 3600.0


def write_bytes(path, data):
    path = M.assert_write_ok(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}_{threading.get_ident()}")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return path


def write_gz(path, src):
    return write_bytes(path, gzip.compress(Path(src).read_bytes(), 6))


class StopAll(Exception):
    """systematic problem (level check, G-2 input builder): $MLT/dft/STOP, no new structures anywhere."""


class HessFail(Exception):
    pass


_STOP_MU = threading.Lock()


def write_stop(msg):
    with _STOP_MU:                       # mlt_common's atomic writers use a per-process temp name: one thread at a time
        if not STOP_FILE.is_file():
            M.write_text(STOP_FILE, f"{_now()} {_jobtag()} {msg}\n")
    print(f"[STOP] {msg}", file=sys.stderr, flush=True)


def stopped():
    """the D1 STOP always counts (also in a D0 process, whose STOP_FILE is STOP_D0 after d0_stopped)."""
    for f in dict.fromkeys((STOP_D1, STOP_FILE)):
        if f.is_file():
            print(f"[STOP] {f} exists: {f.read_text().strip()}", file=sys.stderr, flush=True)
            return True
    return False


def d0_stopped():
    """D0 commands: the D1 STOP (systematic, e.g. the level check) blocks them; their own failures go to STOP_D0 only
    (write_stop), so a D0 problem never blocks a D1 resubmission (decision D0-stop). stopped() keeps checking the D1
    STOP after the switch."""
    global STOP_FILE
    if stopped():
        return True
    STOP_FILE = STOP_D0
    return stopped()


def n_heavy(syms):
    return sum(1 for s in syms if s != "H")


def xyz_block(syms, pos):
    return "".join(f"{s:<3s}{x:16.10f}{y:16.10f}{z:16.10f}\n" for s, (x, y, z) in zip(syms, np.asarray(pos, float)))


def keyhash(key):
    return int(hashlib.sha256(key.encode()).hexdigest()[:12], 16)


def mol_id(key):
    return "m" + hashlib.sha256(key.encode()).hexdigest()[:10]


def label_pairs(s):
    """formed_pairs_ts -> sorted sorted pairs (= d1_g3 opt._pairs, the G3L / Part B order)."""
    return sorted(tuple(sorted(int(a) for a in p.split("-"))) for p in str(s).split())


def formula(syms):
    c = Counter(syms)
    return "".join(f"{e}{c[e]}" for e in sorted(c))


def rec_path(group, sid):
    return RUNS / group / f"{sid}.json"


def _stats(v):
    a = np.asarray([float(x) for x in v if x is not None and np.isfinite(float(x))], float)
    if not len(a):
        return dict(n=0)
    return dict(n=int(len(a)), median=float(np.median(a)), mean=float(a.mean()), min=float(a.min()),
                max=float(a.max()), sum=float(a.sum()), p10=float(np.percentile(a, 10)), p90=float(np.percentile(a, 90)))


# ================================================================ Hessians and normal modes
def hess_freqs(path):
    """$vibrational_frequencies of an ORCA .hess (cm-1); [] if absent."""
    with OI._open(path) as f:
        lines = f.read().splitlines()
    i = next((k for k, ln in enumerate(lines) if ln.strip() == "$vibrational_frequencies"), None)
    if i is None:
        return []
    n = int(lines[i + 1].split()[0])
    return [float(lines[i + 2 + k].split()[1]) for k in range(n)]


def hess_atoms_only(path):
    """(symbols, pos_A, hessian dim) from the $atoms block and the $hessian header (no matrix parse)."""
    with OI._open(path) as f:
        lines = f.read().splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.strip() == "$hessian")
    dim = int(lines[i + 1].split()[0])
    j = next(k for k, ln in enumerate(lines) if ln.strip() == "$atoms")
    n = int(lines[j + 1].split()[0])
    rows = [lines[j + 2 + k].split() for k in range(n)]
    return [r[0] for r in rows], np.array([[float(v) for v in r[2:5]] for r in rows]) * M.BOHR2A, dim


def vib_modes(H, masses, pos_bohr):
    """(lam Eh/(bohr^2 amu), L (3N x nvib, orthonormal mass-weighted modes), nu cm-1 signed, sqrt(m) per coordinate,
    number of removed translation / rotation vectors)."""
    H = np.asarray(H, float)
    m = np.asarray(masses, float)
    x = np.asarray(pos_bohr, float)
    n = len(m)
    sm = np.sqrt(np.repeat(m, 3))
    Hmw = 0.5 * (H + H.T) / np.outer(sm, sm)
    r = x - (m[:, None] * x).sum(0) / m.sum()
    B = []
    for a in range(3):
        v = np.zeros((n, 3))
        v[:, a] = 1.0
        B.append((v * np.sqrt(m)[:, None]).ravel())
    for a in range(3):
        e = np.zeros(3)
        e[a] = 1.0
        B.append((np.cross(e, r) * np.sqrt(m)[:, None]).ravel())
    U, S, _ = np.linalg.svd(np.array(B).T, full_matrices=True)
    ntr = int((S > 1e-6 * S.max()).sum())
    Q = U[:, ntr:]
    lam, W = np.linalg.eigh(Q.T @ Hmw @ Q)
    nu = np.sign(lam) * np.sqrt(np.abs(lam)) * AU2CM
    return lam, Q @ W, nu, sm, ntr


def freq_dev(nu, ntr, ref):
    """max |nu - .hess frequency| (the ntr smallest |f| of the .hess list are its projected zeros); None if unusable."""
    if not ref or len(ref) != len(nu) + ntr:
        return None
    r = np.asarray(ref, float)
    keep = np.sort(r[np.argsort(np.abs(r))[ntr:]])
    return float(np.max(np.abs(keep - np.sort(nu))))


def cart(L, sm, c):
    """mass-weighted mode coefficients c -> Cartesian displacement (N x 3, A)."""
    return ((L @ c) / sm).reshape(-1, 3) * M.BOHR2A


def nm_displace(rng, lam, L, sm, nu, e_target):
    """random real-mode displacement with harmonic energy e_target kcal/mol (decision G-near)."""
    idx = np.flatnonzero(nu >= LOW_CM)
    if not len(idx):
        raise ValueError(f"no mode >= {LOW_CM} cm-1")
    best = None
    for att in range(1, MAX_DRAWS + 1):
        z = rng.standard_normal(len(idx))
        c = z / np.sqrt(lam[idx])
        e = 0.5 * float(np.sum(lam[idx] * c ** 2)) * M.EH2KCAL
        c = c * np.sqrt(e_target / e)
        full = np.zeros(L.shape[1])
        full[idx] = c
        dx = cart(L, sm, full)
        md = float(np.linalg.norm(dx, axis=1).max())
        if md <= MAX_DISP_A:
            return dx, dict(e_target_kcal=float(e_target), e_harm_kcal=float(e_target), max_disp_A=md, draws=att,
                            capped=False, n_modes=int(len(idx)), low_cm=LOW_CM)
        if best is None or md < best[1]:
            best = (dx, md)
    dx, md = best
    f = MAX_DISP_A / md
    return dx * f, dict(e_target_kcal=float(e_target), e_harm_kcal=float(e_target) * f * f, max_disp_A=MAX_DISP_A,
                        draws=MAX_DRAWS, capped=True, n_modes=int(len(idx)), low_cm=LOW_CM)


def imag_displace(L, sm, nu, pos_A, pairs, delta):
    """displacement along the reaction (imaginary) mode with mean forming-bond change delta A (decision G-imag)."""
    pos = np.asarray(pos_A, float)
    n = len(pos)
    V = (L / sm[:, None]).T.reshape(-1, n, 3)
    V = V / np.linalg.norm(V, axis=2).max(1)[:, None, None]          # max atom motion 1 A per unit step
    pairs = [tuple(int(a) for a in p) for p in pairs]
    rij = np.array([pos[i] - pos[j] for i, j in pairs])
    d0 = np.linalg.norm(rij, axis=1)
    u = rij / d0[:, None]
    coef = np.array([np.mean([u[k] @ (V[m, i] - V[m, j]) for k, (i, j) in enumerate(pairs)]) for m in range(len(V))])
    neg = np.flatnonzero(nu < 0)
    if len(neg) == 1 and abs(coef[neg[0]]) >= IMAG_MIN_COEF:
        j, choice = int(neg[0]), "imag"
    else:
        cand = np.flatnonzero(nu < IMAG_CAND_CM)
        if not len(cand):
            cand = np.arange(len(nu))
        j = int(cand[np.argmax(np.abs(coef[cand]))])
        if nu[j] >= 0:
            choice = "lowest_forming"
        elif len(neg) == 1:
            choice = "imag_low_coef"                   # the only imaginary mode, still the most forming-like one
        else:
            choice = f"imag_of_{len(neg)}"
    v = V[j]
    vij = np.array([v[i] - v[k] for i, k in pairs])

    def f(s):
        return float(np.mean(np.linalg.norm(rij + s * vij, axis=1) - d0))

    s = delta / coef[j] if abs(coef[j]) > 1e-8 else 0.0
    conv = False
    for _ in range(50):
        g = f(s) - delta
        if abs(g) < 1e-10:
            conv = True
            break
        rr = rij + s * vij
        der = float(np.mean(np.sum(rr * vij, axis=1) / np.linalg.norm(rr, axis=1)))
        if abs(der) < 1e-12:
            break
        s -= g / der
    dx = s * v
    dd = (np.linalg.norm(rij + s * vij, axis=1) - d0).tolist()
    return dx, dict(choice=choice, mode_cm=float(nu[j]), n_imag_xtb=int(len(neg)), imag_cm=[float(x) for x in nu[neg]],
                    forming_coef=float(coef[j]), delta_target_A=float(delta), delta_A=float(np.mean(dd)),
                    dd_bonds_A=dd, s_A=float(s), max_disp_A=float(abs(s)), solved=conv)


# ================================================================ plan
def row(sid, group, kind, syms, pos, tags, src, charge=0, mult=1, **kw):
    r = dict(struct_id=sid, group=group, kind=kind, rxn_ids=tags["rxn_ids"], folds=tags["folds"],
             dipoles=tags["dipoles"], n_atoms=len(syms), n_heavy=n_heavy(syms), charge=int(charge), mult=int(mult),
             symbols=" ".join(syms), pos_A=[float(v) for v in np.asarray(pos, float).ravel()], pos_final=True,
             src=str(src), hess_src="", mol_id="", mol_key="", seed=-1, disp="{}", rxn_id=0, fold_T1=-9, g1_class="")
    r.update(kw)
    return r


def eda_atoms(path):
    """(symbols, fragment tags, coordinates A) of the coordinate block of a D1 eda.inp ('C(1)  x y z')."""
    syms, tags, xyz, inb = [], [], [], False
    for ln in Path(path).read_text().splitlines():
        s = ln.strip()
        if not inb:
            if s.lower().replace(" ", "").startswith("*xyz"):
                inb = True
            continue
        if s == "*":
            break
        m = RE_EDA_ATOM.match(ln)
        if not m:
            raise ValueError(f"{path}: coordinate line without a fragment tag: {ln!r}")
        syms.append(m.group(1))
        tags.append(int(m.group(2)))
        xyz.append([float(m.group(3)), float(m.group(4)), float(m.group(5))])
    return syms, tags, np.array(xyz)


def rxn_job(t):
    """plan rows of one target reaction: G1 (3) + G3 frag (2), checks (process-pool worker)."""
    rid = int(t["rxn_id"])
    out = dict(rxn_id=rid, rows=[], g1_err=None, frag_err=None, chk={}, ref_syms=None)
    tags = dict(rxn_ids=str(rid), folds=str(t["fold"]), dipoles=t["dipole"])
    base = dict(rxn_id=rid, fold_T1=int(t["fold"]), g1_class=t["g1_class"])
    try:
        syms, pos = M.read_xyz(t["ts"])
        pos = np.asarray(pos, float)
        if len(syms) != int(t["n_atoms"]):
            raise ValueError(f"label ts.xyz has {len(syms)} atoms, label n_atoms {t['n_atoms']}")
    except Exception as e:                                                     # noqa: BLE001
        out["g1_err"] = out["frag_err"] = f"label_ts: {type(e).__name__}: {e}"[:300]
        return out
    try:
        out["ref_syms"] = [M.read_xyz(p)[0] for p in t["refs"]]
    except Exception as e:                                                     # noqa: BLE001
        out["chk"]["ref_err"] = f"{type(e).__name__}: {e}"[:200]
    # ---- G1
    try:
        hd = OI.read_hess(t["hess"])
        if list(hd["symbols"]) != list(syms):
            raise ValueError("hess $atoms order != label ts.xyz")
        sh = hd["pos_bohr"] * M.BOHR2A - pos
        dev = float(np.abs(sh - sh.mean(0)).max())
        if dev > HESS_TOL_A:
            raise ValueError(f"hess geometry != label ts.xyz ({dev:.2e} A)")
        lam, L, nu, sm, ntr = vib_modes(hd["hessian"], hd["masses"], hd["pos_bohr"])
        rng = np.random.default_rng(M.SEED + rid)
        e12 = rng.uniform(*E_RANGE, size=2)
        dmag = float(rng.uniform(*DELTA_RANGE))
        sgn = 1.0 if rng.random() < 0.5 else -1.0
        dx1, i1 = nm_displace(rng, lam, L, sm, nu, float(e12[0]))
        dx2, i2 = nm_displace(rng, lam, L, sm, nu, float(e12[1]))
        dxi, ii = imag_displace(L, sm, nu, pos, t["pairs"], sgn * dmag)
        com = dict(hess_src=str(t["hess"]), seed=M.SEED + rid, **base)
        q = int(t["charge1"]) + int(t["charge2"])
        out["rows"] += [row(f"g1_{rid}_n1", "G1", "ts_near", syms, pos + dx1, tags, t["ts"], q, disp=json.dumps(i1), **com),
                        row(f"g1_{rid}_n2", "G1", "ts_near", syms, pos + dx2, tags, t["ts"], q, disp=json.dumps(i2), **com),
                        row(f"g1_{rid}_i", "G1", "ts_imag", syms, pos + dxi, tags, t["ts"], q, disp=json.dumps(ii), **com)]
        out["chk"].update(hess_geom_dev_A=dev, freq_dev_cm=freq_dev(nu, ntr, hess_freqs(t["hess"])), ntr=ntr,
                          n_imag_xtb=int((nu < 0).sum()), imag_cm=[float(x) for x in nu[nu < 0]])
    except Exception as e:                                                     # noqa: BLE001
        out["g1_err"] = f"{type(e).__name__}: {e}"[:300]
    # ---- G3 fragments (D1 eda.inp partition)
    try:
        es, tg, ex = eda_atoms(t["eda"])
        if es != list(syms):
            raise ValueError("eda.inp elements != label ts.xyz")
        cdev = float(np.abs(ex - pos).max())
        if cdev > 1e-6:
            raise ValueError(f"eda.inp coordinates != label ts.xyz ({cdev:.2e} A)")
        fr = {}
        for k in (1, 2):
            idx = [i for i, g in enumerate(tg) if g == k]
            if len(idx) != int(t[f"n_f{k}"]):
                raise ValueError(f"eda.inp fragment {k}: {len(idx)} atoms, label n_f{k} {t[f'n_f{k}']}")
            fr[k] = idx
        if set(tg) != {1, 2}:
            raise ValueError(f"eda.inp fragment tags {sorted(set(tg))}")
        f1 = set(fr[1])
        cross = all((i in f1) != (j in f1) for i, j in t["pairs"])
        for k in (1, 2):
            fs = [syms[i] for i in fr[k]]
            same_ref = bool(out["ref_syms"] and out["ref_syms"][k - 1] == fs)
            out["rows"].append(row(f"g3_{rid}_f{k}", "G3", "frag", fs, pos[fr[k]], tags, t["eda"], t[f"charge{k}"],
                                   disp=json.dumps(dict(partition="d1 sp/eda.inp tags", frag=k, atoms=fr[k],
                                                        ref_elements_equal=same_ref)), **base))
            out["chk"][f"frag{k}_ref_elements_equal"] = same_ref
        out["chk"].update(first_nf1_is_frag1=all(g == 1 for g in tg[:int(t["n_f1"])]), pairs_cross_fragments=cross)
    except Exception as e:                                                     # noqa: BLE001
        out["frag_err"] = f"{type(e).__name__}: {e}"[:300]
    return out


def e_orig(d, k):
    """original label e_frag{k}_rel_eh (before the LADDER cr2 replacement)."""
    cr = d.get("cr") or {}
    if f"frag{k}" in (cr.get("changed") or []):
        return float(cr["orig"][f"e_frag{k}_rel_eh"])
    return float(d[f"e_frag{k}_rel_eh"])


def molecules(T, labs, res):
    """G3 react_ref / react_disp rows of the unique label reactant molecules (decision G-dedupe) + stats."""
    refsyms = {r["rxn_id"]: r.get("ref_syms") for r in res}
    groups, nokey = {}, 0
    for t in T.itertuples(index=False):
        d = labs[int(t.rxn_id)]
        cr = d.get("cr") or {}
        for k in (1, 2):
            kind = "dipole" if k == 1 else "dipolarophile"
            key = cr.get(f"key{k}")
            if not key:
                key, nokey = f"{kind}:nokey:{t.job_id}", nokey + 1
            rs = refsyms.get(int(t.rxn_id))
            changed = f"frag{k}" in (cr.get("changed") or [])
            groups.setdefault(key, []).append(dict(
                rxn_id=int(t.rxn_id), job_id=str(t.job_id), k=k, e=e_orig(d, k), path=str(M.label_refs(t.rxn_id)[k - 1]),
                fold=int(t.fold_T1), dipole=str(t.dipole), charge=int(d.get(f"charge{k}") or 0),
                cr_src=cr.get(f"src{k}_job") if changed else None, syms=rs[k - 1] if rs else None))
    rows, n_formula_bad, cr_agree, cr_n, sizes = [], [], 0, 0, []
    for key in sorted(groups):
        mem = groups[key]
        rep = min(mem, key=lambda m: (m["e"], m["job_id"]))
        fs = {formula(m["syms"]) for m in mem if m["syms"]}
        if len(fs) > 1:
            n_formula_bad.append(key)
        for m in mem:
            if m["cr_src"] is not None:
                cr_n += 1
                cr_agree += int(m["cr_src"] == rep["job_id"])
        syms, pos = M.read_xyz(rep["path"])
        tags = dict(rxn_ids=M.tag_list(m["rxn_id"] for m in mem), folds=M.tag_list(m["fold"] for m in mem),
                    dipoles=M.tag_list(m["dipole"] for m in mem))
        mid = mol_id(key)
        com = dict(mol_id=mid, mol_key=key, seed=keyhash(key))
        info = dict(rep_job=rep["job_id"], rep_rxn=rep["rxn_id"], rep_frag=rep["k"], e_rel_orig_eh=rep["e"],
                    n_members=len(mem))
        rows.append(row(f"g3_{mid}_ref", "G3", "react_ref", syms, pos, tags, rep["path"], rep["charge"],
                        disp=json.dumps(info), **com))
        for j in (1, 2):
            rows.append(row(f"g3_{mid}_d{j}", "G3", "react_disp", syms, pos, tags, rep["path"], rep["charge"],
                            pos_final=False, hess_src=str(HESS_DIR / f"{mid}.hess.gz"),
                            disp=json.dumps(dict(info, e_index=j)), **com))
        sizes.append(len(mem))
    kinds = Counter(k.split(":", 1)[0] for k in groups)
    st = dict(method=DECISIONS["G-dedupe"], n_occurrences=sum(sizes), n_unique=len(groups), by_kind=dict(kinds),
              n_constrained_keys=sum("|fixed:" in k for k in groups), n_without_key=nokey,
              members_per_molecule=_stats(sizes), n_keys_formula_differs=len(n_formula_bad),
              keys_formula_differs=n_formula_bad[:20],
              cr2_label_min_agreement=dict(n_changed_fragments=cr_n, n_src_equals_representative=cr_agree),
              n_structures=len(rows))
    return rows, st


def phase_i_rxns():
    out = {}
    if G3_UNITS.is_dir():
        for p in G3_UNITS.iterdir():
            m = re.match(r"^(L|B)\.(\d+)$", p.name)
            if m:
                out[int(m.group(2))] = m.group(1)
    return out


def g2_hess_check(rid, base=None):
    """the g3l.hess_check rules without the D1 modules (process-pool worker); base = the g2_final root (D1 $G2 default,
    gen_d0 passes the G2c D0 root)."""
    g2d = Path(base or M.G2_FINAL) / str(rid)
    hp, xp, ip = g2d / "work" / "ts.hess", g2d / "ts.xyz", g2d / "work" / "ts.inp"
    out = dict(rxn_id=rid, ok=False, reason="", hess=str(hp), start=str(xp))
    if not ((g2d / ".done").is_file() and xp.is_file()):
        return dict(out, reason="no_g2_ts")
    if not hp.is_file():
        return dict(out, reason="no_hess")
    try:
        syms, pos = M.read_xyz(xp)
        hs, hx, dim = hess_atoms_only(hp)
        kw = next((ln.strip() for ln in ip.read_text().splitlines() if ln.strip().startswith("!")), None) \
            if ip.is_file() else None
    except Exception as e:                                                     # noqa: BLE001
        return dict(out, reason=f"hess_unreadable: {type(e).__name__}: {e}"[:200])
    if list(hs) != list(syms):
        return dict(out, reason="hess_atom_order")
    sh = hx - np.asarray(pos, float)
    raw = float(np.abs(sh).max())
    dev = float(np.abs(sh - sh.mean(0)).max())             # a rigid translation leaves a Cartesian Hessian unchanged
    if dev > HESS_TOL_A:
        return dict(out, reason="hess_geometry", max_dev_A=raw, max_dev_shift_removed_A=dev)
    out.update(max_dev_raw_A=raw, translated=raw > HESS_TOL_A, shift_A=[float(v) for v in sh.mean(0)])
    if dim != 3 * len(syms):
        return dict(out, reason="hess_dimension")
    if not kw or "XTB2" not in kw.replace("!", " ").split():
        return dict(out, reason="hess_not_xtb2")
    return dict(out, ok=True, max_dev_A=dev, n_atoms=len(syms), n_heavy=n_heavy(syms))


def g2_select(T, labs, cls, n_g2, nproc):
    import pandas as pd
    S = pd.read_csv(G_STATUS, usecols=["rxn_id", "status"])
    ok = {int(r) for r, s in zip(S.rxn_id, S.status) if str(s) == "ok"}
    ph = phase_i_rxns()
    tset = set(int(x) for x in T.rxn_id)
    pool0 = sorted(tset & ok)
    cand = [r for r in pool0 if r not in ph]
    with ProcessPoolExecutor(max(1, nproc)) as ex:
        checks = dict(zip(cand, ex.map(g2_hess_check, cand, chunksize=32)))
    usable = [r for r in cand if checks[r]["ok"]]
    tr = T.set_index("rxn_id")
    strata = {}
    for r in usable:
        strata.setdefault((str(tr.at[r, "dipole"]), cls.get(r, "none")), []).append(r)
    keys = sorted(strata)
    n = min(int(n_g2), len(usable))
    quota = {k: n * len(strata[k]) / max(1, len(usable)) for k in keys}
    alloc = {k: int(np.floor(quota[k])) for k in keys}
    for k in sorted(keys, key=lambda k: (-(quota[k] - alloc[k]), keys.index(k)))[:n - sum(alloc.values())]:
        alloc[k] += 1
    rng = np.random.default_rng(M.SEED + 2)
    pick = []
    for k in keys:
        if alloc[k]:
            pick += [int(x) for x in rng.choice(np.asarray(sorted(strata[k]), dtype=np.int64), size=alloc[k],
                                                replace=False)]
    pick = sorted(pick)
    rows, units = [], []
    for rid in pick:
        d, c = labs[rid], checks[rid]
        syms, pos = M.read_xyz(c["start"])
        q = int(d.get("charge1") or 0) + int(d.get("charge2") or 0)
        fold = int(tr.at[rid, "fold_T1"])
        tags = dict(rxn_ids=str(rid), folds=str(fold), dipoles=str(tr.at[rid, "dipole"]))
        u = dict(rxn_id=rid, job_id=str(tr.at[rid, "job_id"]), struct_id=f"g2_{rid}", n_atoms=len(syms),
                 n_heavy=n_heavy(syms), dipole=tags["dipoles"], g1_class=cls.get(rid, "none"), fold=fold, charge=q,
                 pairs=[list(p) for p in label_pairs(d["formed_pairs_ts"])], start=c["start"], hess=c["hess"],
                 hess_sha256=M.sha256(c["hess"]), hess_translated=bool(c.get("translated")),
                 hess_shift_A=c.get("shift_A"), label_ts=str(M.label_ts(rid)), **tags)
        units.append(u)
        rows.append(row(u["struct_id"], "G2", "g2_path", syms, pos, tags, c["start"], q, hess_src=c["hess"],
                        rxn_id=rid, fold_T1=fold, g1_class=u["g1_class"],
                        disp=json.dumps(dict(hess_sha256=u["hess_sha256"]))))
    info = dict(rule=DECISIONS["G-2sel"], n_target=int(n_g2), n_picked=len(pick), n_g2_ok=len(pool0),
                n_phase_i_excluded=len(set(pool0) & set(ph)), phase_i_units=dict(Counter(ph.values())),
                n_hess_excluded=len(cand) - len(usable),
                hess_excluded=dict(Counter(checks[r]["reason"] for r in cand if not checks[r]["ok"])),
                hess_geometry_dev_A=_stats([checks[r].get("max_dev_A") for r in cand
                                            if checks[r]["reason"] == "hess_geometry"]),
                hess_geometry_dev_shift_removed_A=_stats([checks[r].get("max_dev_shift_removed_A") for r in cand
                                                          if checks[r]["reason"] == "hess_geometry"]),
                hess_excluded_by_stratum=dict(Counter(f"{tr.at[r, 'dipole']}/{cls.get(r, 'none')}" for r in cand
                                                      if not checks[r]["ok"])),
                hess_excluded_rxns=sorted(r for r in cand if not checks[r]["ok"]),
                n_translated_hess_pool=sum(bool(checks[r].get("translated")) for r in usable),
                n_translated_hess_picked=sum(bool(u["hess_translated"]) for u in units),
                strata=[dict(dipole=k[0], g1_class=k[1], n_pool=len(strata[k]), quota=quota[k], n=alloc[k])
                        for k in keys],
                picked_by_fold=dict(Counter(str(u["fold"]) for u in units)),
                picked_by_class=dict(Counter(u["g1_class"] for u in units)))
    return rows, units, info


def plan_fingerprint(limit, n_g2):
    return dict(code=CODE_VERSION, labels=M.sha256(M.LABELS), s_folds=M.sha256(M.S_FOLDS),
                c_compare=M.sha256(M.C_COMPARE), g_status=M.sha256(G_STATUS), limit=limit, n_g2=int(n_g2), seed=M.SEED,
                phase_i_units=sorted(f"{v}.{k}" for k, v in phase_i_rxns().items()))


def any_records():
    for g in ("G1", "G3"):
        d = RUNS / g
        if d.is_dir() and next(d.glob("*.json"), None) is not None:
            return True
    d = RUNS / "G2"
    return d.is_dir() and next(d.glob("*/unit.json"), None) is not None


def _cls(v):
    s = str(v).strip().lower()
    return "same" if s == "true" else "different" if s == "false" else "none"


def plan_stats(df):
    st = dict(n_rows=int(len(df)), n_engrad_structures=int(df.group.isin(["G1", "G3"]).sum()),
              n_g2_reactions=int((df.group == "G2").sum()),
              by_group={g: int(n) for g, n in df.groupby("group").size().items()},
              by_group_kind={f"{g}/{k}": int(n) for (g, k), n in df.groupby(["group", "kind"]).size().items()},
              n_atoms={f"{g}/{k}": _stats(x.n_atoms.tolist()) for (g, k), x in df.groupby(["group", "kind"])})
    folds = {}
    for k in (-1, 0, 1, 2, 3, 4):
        tagged = df.folds.map(lambda s, k=k: str(k) in str(s).split(","))
        folds[str(k)] = dict(
            tagged={f"{g}/{kk}": int(n) for (g, kk), n in df[tagged].groupby(["group", "kind"]).size().items()},
            usable_for_FT_k=({f"{g}/{kk}": int(n) for (g, kk), n in df[~tagged].groupby(["group", "kind"]).size().items()}
                             if k >= 0 else None))
    st["folds"] = folds
    st["note_folds"] = ("tagged = structures whose folds tag contains k (FT_k drops them); a molecule used by reactions "
                        "of several folds is dropped from each of those FT_k")
    dd = {}
    for kind in ("ts_near", "ts_imag", "frag"):
        J = [json.loads(s) for s in df.loc[df.kind == kind, "disp"]]
        if kind == "ts_near":
            dd[kind] = dict(e_target_kcal=_stats([j["e_target_kcal"] for j in J]),
                            e_harm_kcal=_stats([j["e_harm_kcal"] for j in J]), max_disp_A=_stats([j["max_disp_A"] for j in J]),
                            n_capped=sum(bool(j["capped"]) for j in J), draws=_stats([j["draws"] for j in J]),
                            n_modes=_stats([j["n_modes"] for j in J]))
        elif kind == "ts_imag":
            dd[kind] = dict(choice=dict(Counter(j["choice"] for j in J)),
                            n_imag_xtb=dict(Counter(str(j["n_imag_xtb"]) for j in J)),
                            abs_delta_A=_stats([abs(j["delta_A"]) for j in J]),
                            delta_minus_target_A=_stats([abs(j["delta_A"] - j["delta_target_A"]) for j in J]),
                            max_disp_A=_stats([j["max_disp_A"] for j in J]), mode_cm=_stats([j["mode_cm"] for j in J]),
                            forming_coef=_stats([abs(j["forming_coef"]) for j in J]),
                            n_unsolved=sum(not j["solved"] for j in J),
                            n_sign_positive=sum(j["delta_target_A"] > 0 for j in J))
        else:
            dd[kind] = dict(n_ref_elements_differ=sum(not j.get("ref_elements_equal") for j in J))
    st["displacements"] = dd
    return st


def cmd_plan(limit=None, n_g2=N_G2, force=False):
    import pandas as pd
    fp = plan_fingerprint(limit, n_g2)
    old = _rj(PLAN_META)
    if old is not None and PLAN_PQ.is_file() and G2_PLAN.is_file() and old.get("fingerprint") == fp and not force:
        print(f"[plan] up to date ({PLAN_PQ})", flush=True)
        return 0
    if old is not None and any_records() and old.get("fingerprint") != fp:
        M.stop(f"plan inputs changed after records were written ({old.get('fingerprint')} -> {fp}); not re-planned")
    t0 = time.time()
    nproc = _cpus()
    T = M.targets()
    if limit:
        T = T.head(int(limit)).copy()
    labs = {int(d["rxn_id"]): d for d in M.read_json(M.LABELS)}
    Cc = pd.read_csv(M.C_COMPARE, usecols=["rxn_id", "same_ts_g1"], dtype={"same_ts_g1": str}, keep_default_na=False)
    cls = {int(r): _cls(v) for r, v in zip(Cc.rxn_id, Cc.same_ts_g1)}
    jobs = []
    for t in T.itertuples(index=False):
        rid = int(t.rxn_id)
        d = labs[rid]
        jobs.append(dict(rxn_id=rid, job_id=str(t.job_id), dipole=str(t.dipole), fold=int(t.fold_T1),
                         g1_class=cls.get(rid, "none"), pairs=[list(p) for p in M.formed_pairs(t.formed_pairs_ts)],
                         n_atoms=int(t.n_atoms), n_f1=int(t.n_f1), n_f2=int(t.n_f2),
                         charge1=int(d.get("charge1") or 0), charge2=int(d.get("charge2") or 0),
                         ts=str(M.label_ts(rid)), hess=str(M.label_xtb_hess(rid)),
                         eda=str(M.D1_JOBS / str(t.job_id) / "sp" / "eda.inp"), refs=[str(p) for p in M.label_refs(rid)]))
    with ProcessPoolExecutor(max(1, nproc)) as ex:
        res = list(ex.map(rxn_job, jobs, chunksize=8))
    rows = [x for r in res for x in r["rows"]]
    g1_err = {str(r["rxn_id"]): r["g1_err"] for r in res if r["g1_err"]}
    fr_err = {str(r["rxn_id"]): r["frag_err"] for r in res if r["frag_err"]}
    chk = [r["chk"] for r in res]
    mrows, dedupe = molecules(T, labs, res)
    g2rows, g2units, g2info = g2_select(T, labs, cls, n_g2, nproc)
    df = pd.DataFrame(rows + mrows + g2rows)
    if df.struct_id.duplicated().any():
        M.stop(f"duplicate struct_id in the plan: {df.struct_id[df.struct_id.duplicated()].tolist()[:5]}")
    M.write_parquet(PLAN_PQ, df)
    M.write_json(G2_PLAN, dict(code=CODE_VERSION, written=_now(), fingerprint=fp, rule=DECISIONS["G-2sel"],
                               rxns=g2units))
    fdev = [c.get("freq_dev_cm") for c in chk if c.get("freq_dev_cm") is not None]
    rep = dict(phase="G plan", code=CODE_VERSION, written=_now(), host=socket.gethostname(), wall_s=time.time() - t0,
               fingerprint=fp, limit=limit, n_targets=int(len(T)), cores_without_fold=T.attrs.get("cores_without_fold"),
               targets_by_fold={str(k): int(n) for k, n in T.fold_T1.value_counts().sort_index().items()},
               hessian_source=DECISIONS["G-hess"], decisions=DECISIONS, parameters=dict(
                   E_range_kcal=E_RANGE, delta_range_A=DELTA_RANGE, low_cm=LOW_CM, max_disp_A=MAX_DISP_A,
                   max_draws=MAX_DRAWS, imag_min_coef=IMAG_MIN_COEF, imag_cand_cm=IMAG_CAND_CM, seed=M.SEED),
               counts=plan_stats(df),
               excluded=dict(n_g1=len(g1_err), g1=dict(list(g1_err.items())[:50]), n_frag=len(fr_err),
                             frag=dict(list(fr_err.items())[:50])),
               checks=dict(hess_geom_dev_A=_stats([c.get("hess_geom_dev_A") for c in chk]),
                           freq_dev_vs_hess_cm=_stats(fdev),
                           n_imag_xtb_label_ts=dict(Counter(str(c.get("n_imag_xtb")) for c in chk if "n_imag_xtb" in c)),
                           n_tr_removed=dict(Counter(str(c.get("ntr")) for c in chk if "ntr" in c)),
                           frag_partition=dict(
                               n_first_nf1_is_frag1=sum(bool(c.get("first_nf1_is_frag1")) for c in chk),
                               n_not_contiguous=sum(c.get("first_nf1_is_frag1") is False for c in chk),
                               n_pairs_cross_fragments=sum(bool(c.get("pairs_cross_fragments")) for c in chk),
                               n_frag1_ref_elements_equal=sum(bool(c.get("frag1_ref_elements_equal")) for c in chk),
                               n_frag2_ref_elements_equal=sum(bool(c.get("frag2_ref_elements_equal")) for c in chk)),
                           n_ref_read_errors=sum("ref_err" in c for c in chk)),
               dedupe=dedupe, g2=g2info,
               outputs={str(PLAN_PQ): M.sha256(PLAN_PQ), str(G2_PLAN): M.sha256(G2_PLAN)})
    M.write_json(PLAN_META, dict(fingerprint=fp, written=_now(), n_rows=int(len(df)), code=CODE_VERSION))
    M.write_json(G_PLAN_JSON, rep)
    print(json.dumps(dict(by_group_kind=rep["counts"]["by_group_kind"], excluded_g1=len(g1_err), excluded_frag=len(fr_err),
                          n_unique_molecules=dedupe["n_unique"], g2_picked=g2info["n_picked"],
                          freq_dev_cm=rep["checks"]["freq_dev_vs_hess_cm"],
                          ts_imag_choice=rep["counts"]["displacements"]["ts_imag"]["choice"],
                          wall_s=round(rep["wall_s"], 1)), indent=1), flush=True)
    return 0


def load_plan(d0=False):
    import pandas as pd
    p = PLAN_D0_PQ if d0 else PLAN_PQ
    if not p.is_file():
        M.not_ready(f"{p} missing: run 'gen_dft.py {'plan_d0' if d0 else 'plan'}' first")
    return pd.read_parquet(p)


# ================================================================ EnGrad worker (run, g0)
class Ctx:
    def __init__(self, tag="g"):
        self.mu = threading.Lock()
        self.halt = ""
        self.timed_out = False
        self.n_pipeline = 0
        self.locks = {}
        self.stats = Counter()
        self.core_h = 0.0
        self.local = Path(os.environ.get("TMPDIR") or "/tmp") / f"mlt_{tag}_{_jobtag()}_{os.getpid()}"
        self.env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")

    def lock(self, key):
        with self.mu:
            return self.locks.setdefault(key, threading.Lock())


def run_items(items, nw, ctx, reserve_h, fn, name):
    """claim items in order with nw threads (fn(item, ctx) -> (how, core_h)); stop claiming on halt / time."""
    it = iter(items)

    def claim():
        with ctx.mu:
            if ctx.halt:
                return None
            if time_left_h() < reserve_h:
                ctx.timed_out = True
                return None
            return next(it, None)

    def loop():
        while True:
            x = claim()
            if x is None:
                return
            try:
                how, ch = fn(x, ctx)
                with ctx.mu:
                    ctx.stats[how] += 1
                    ctx.core_h += ch
            except StopAll as e:
                with ctx.mu:
                    ctx.halt = ctx.halt or str(e)
                write_stop(str(e))
                return
            except Exception as e:                                             # noqa: BLE001
                print(f"[gen_dft] pipeline error {name(x)}: {type(e).__name__}: {e}\n{traceback.format_exc(limit=4)}",
                      file=sys.stderr, flush=True)
                with ctx.mu:
                    ctx.stats["pipeline_error"] += 1
                    ctx.n_pipeline += 1
                    if ctx.n_pipeline > MAX_PIPELINE_ERR and not ctx.halt:
                        ctx.halt = f"more than {MAX_PIPELINE_ERR} pipeline errors in this task"
    if not items:
        return
    ctx.local.mkdir(parents=True, exist_ok=True)
    try:
        with ThreadPoolExecutor(max(1, min(nw, len(items)))) as ex:
            for f in [ex.submit(loop) for _ in range(max(1, min(nw, len(items))))]:
                f.result()
    finally:
        shutil.rmtree(ctx.local, ignore_errors=True)


def engrad(work, text, syms, pos, env, name="g"):
    """one serial EnGrad in work -> dict(ok, level_bad, reason, att, energy_eh, grad, ...)."""
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    (work / f"{name}.inp").write_text(text)
    t0 = time.time()
    with open(work / f"{name}.out", "w") as fo, open(work / f"{name}.err", "w") as fe:
        p = subprocess.run([str(M.ORCA), f"{name}.inp"], cwd=str(work), stdout=fo, stderr=fe, env=env)
    wall = time.time() - t0
    txt = (work / f"{name}.out").read_text(errors="replace")
    term = "ORCA TERMINATED NORMALLY" in txt
    att = dict(rc=p.returncode, wall_s=wall, core_h=wall / 3600.0, terminated=term, host=socket.gethostname())
    res = dict(ok=False, level_bad=False, reason="", att=att)
    if term:
        lv, why = OI.label_level(OI.input_echo(work / f"{name}.out"))
        if not lv:
            return dict(res, level_bad=True, reason=f"input echo: {why}")
        if CDS_MARK not in txt or EPS_MARK not in txt:
            return dict(res, level_bad=True, reason="no SMD CDS line / water epsilon 78.3550 in the output")
    if not term:
        return dict(res, reason=f"orca_not_terminated (rc {p.returncode})")
    if RE_SCF_BAD.search(txt) or not RE_SCF_OK.search(txt):
        return dict(res, reason="scf_not_converged")
    ep = work / f"{name}.engrad"
    if not ep.is_file():
        return dict(res, reason="no_engrad")
    eg = OI.read_engrad(ep)
    zz = [OI.SYM2Z.get(s) for s in syms]
    dev = float(np.abs(eg["pos_bohr"] * M.BOHR2A - np.asarray(pos, float)).max()) if eg["n_atoms"] == len(syms) else None
    if eg["n_atoms"] != len(syms) or eg["Z"] != zz or dev is None or dev > 1e-4:
        return dict(res, reason=f"engrad_mismatch (n {eg['n_atoms']}, dev {dev})")
    fs = RE_FSPE.findall(txt)
    e_out = float(fs[-1]) if fs else None
    g = np.asarray(eg["grad_eh_bohr"], float)
    if not np.all(np.isfinite(g)) or not np.isfinite(eg["energy_eh"]):
        return dict(res, reason="non_finite")
    ns = RE_SCF_N.findall(txt)
    return dict(res, ok=True, energy_eh=float(eg["energy_eh"]), grad=g, e_out_eh=e_out,
                de_out_engrad_eh=(e_out - float(eg["energy_eh"])) if e_out is not None else None, pos_dev_A=dev,
                scf_cycles=int(ns[-1]) if ns else None)


def mol_hess(mid, syms, pos, charge, mult, ctx, hdir=None, extra=None):
    """xTB Hessian of a G3 molecule at its reference geometry, cached ($MLT/dft/hess/<mid>.hess.gz + .json).
    D0 (gen_d0): hdir = $MLT/dft/hess_d0 (D0 label TSs d0ts_<rid> and D0 molecules), extra = fields added to the json."""
    hdir = Path(hdir or HESS_DIR)
    hz, hj = hdir / f"{mid}.hess.gz", hdir / f"{mid}.json"
    with ctx.lock("hess:" + mid):
        info = _rj(hj)
        if info is not None and info.get("status") == "ok" and hz.is_file():
            return hz, info
        if info is not None and info.get("status") == "fail":
            raise HessFail(info.get("reason"))
        work = ctx.local / f"xtb_{mid}"
        text = XTB_HEADER + f"* xyz {int(charge)} {int(mult)}\n" + xyz_block(syms, pos) + "*\n"
        runs, ok = [], False
        try:
            for k in (1, 2):
                shutil.rmtree(work, ignore_errors=True)
                work.mkdir(parents=True)
                (work / "h.inp").write_text(text)
                t0 = time.time()
                with open(work / "h.out", "w") as fo, open(work / "h.err", "w") as fe:
                    p = subprocess.run([str(M.ORCA), "h.inp"], cwd=str(work), stdout=fo, stderr=fe, env=ctx.env)
                wall = time.time() - t0
                txt = (work / "h.out").read_text(errors="replace")
                ok = "ORCA TERMINATED NORMALLY" in txt and (work / "h.hess").is_file()
                runs.append(dict(run=k, rc=p.returncode, wall_s=wall, ok=ok, host=socket.gethostname()))
                if ok:
                    break
            if not ok:
                M.write_json(hj, dict(status="fail", reason="xTB Freq failed twice", mol_id=mid, runs=runs,
                                      core_h=sum(r["wall_s"] for r in runs) / 3600, finished=_now(), **(extra or {})))
                raise HessFail("xTB Freq failed twice")
            write_gz(hz, work / "h.hess")
            fr = hess_freqs(work / "h.hess")
            info = dict(status="ok", mol_id=mid, header=XTB_HEADER, runs=runs, core_h=sum(r["wall_s"] for r in runs) / 3600,
                        imag_cm=[f for f in fr if f < 0], n_freq=len(fr), finished=_now(), slurm_job=_jobtag())
            if extra:
                info.update(extra)
            M.write_json(hj, info)
            return hz, info
        finally:
            shutil.rmtree(work, ignore_errors=True)


def _hess_extra(r, syms):
    return dict(n_atoms=len(syms), d0_id=int(r["d0_id"]), kind=r["kind"]) if r["kind"].startswith("d0_") else None


def mol_disp(r, syms, pos, ctx, hdir=None):
    """react_disp geometry: decision G-near on the molecule's xTB Hessian with its own seeded RNG."""
    hz, info = mol_hess(r["mol_id"], syms, pos, int(r["charge"]), int(r["mult"]), ctx, hdir=hdir,
                        extra=_hess_extra(r, syms))
    hd = OI.read_hess(hz)
    if list(hd["symbols"]) != list(syms):
        raise HessFail("hess atom order != reference")
    sh = hd["pos_bohr"] * M.BOHR2A - pos
    dev = float(np.abs(sh - sh.mean(0)).max())
    if dev > HESS_TOL_A:
        raise HessFail(f"hess geometry != reference ({dev:.2e} A)")
    lam, L, nu, sm, ntr = vib_modes(hd["hessian"], hd["masses"], hd["pos_bohr"])
    rng = np.random.default_rng([M.SEED, int(r["seed"])])
    e12 = rng.uniform(*E_RANGE, size=2)
    d1 = nm_displace(rng, lam, L, sm, nu, float(e12[0]))
    d2 = nm_displace(rng, lam, L, sm, nu, float(e12[1]))
    j = int(json.loads(r["disp"])["e_index"])
    dx, inf = d1 if j == 1 else d2
    inf.update(hess=str(hz), hess_core_h=info.get("core_h"), n_imag_xtb=int((nu < 0).sum()), ntr=ntr,
               freq_dev_cm=freq_dev(nu, ntr, hess_freqs(hz)), hess_geom_dev_A=dev, e_index=j)
    return pos + dx, inf


def ts_disp_d0(r, syms, pos, ctx):
    """d0_ts_near / d0_ts_imag geometry (decision D0-G1): the G1 rules on an xTB Freq at the D0 label TS, computed here
    (no xTB Hessian exists at the Coley TS) and cached as $MLT/dft/hess_d0/d0ts_<rid>.hess.gz. Draws exactly in the G1
    order with default_rng(seed = SEED + D0 offset + rid): E1, E2, |delta|, sign, direction 1, direction 2 (the imaginary
    displacement uses no draw), so the 3 rows of a reaction are reproducible independently of each other."""
    hz, info = mol_hess(r["mol_id"], syms, pos, int(r["charge"]), int(r["mult"]), ctx, hdir=HESS_D0,
                        extra=_hess_extra(r, syms))
    hd = OI.read_hess(hz)
    if list(hd["symbols"]) != list(syms):
        raise HessFail("hess atom order != D0 label TS")
    sh = hd["pos_bohr"] * M.BOHR2A - pos
    dev = float(np.abs(sh - sh.mean(0)).max())
    if dev > HESS_TOL_A:
        raise HessFail(f"hess geometry != D0 label TS ({dev:.2e} A)")
    lam, L, nu, sm, ntr = vib_modes(hd["hessian"], hd["masses"], hd["pos_bohr"])
    d = json.loads(r["disp"])
    j = int(d["e_index"])
    rng = np.random.default_rng(int(r["seed"]))
    e12 = rng.uniform(*E_RANGE, size=2)
    dmag = float(rng.uniform(*DELTA_RANGE))
    sgn = 1.0 if rng.random() < 0.5 else -1.0
    if j == 0:
        dx, inf = imag_displace(L, sm, nu, pos, d["pairs"], sgn * dmag)
    else:
        dx, inf = nm_displace(rng, lam, L, sm, nu, float(e12[0]))
        if j == 2:
            dx, inf = nm_displace(rng, lam, L, sm, nu, float(e12[1]))
    inf.update(hess=str(hz), hess_core_h=info.get("core_h"), n_imag_xtb=int((nu < 0).sum()),
               imag_cm=[float(x) for x in nu[nu < 0]], ntr=ntr, freq_dev_cm=freq_dev(nu, ntr, hess_freqs(hz)),
               hess_geom_dev_A=dev, e_index=j, seed=int(r["seed"]))
    return pos + dx, inf


def run_struct(r, ctx):
    """one plan row -> runs/<group>/<struct_id>.json (+ .out.gz); skipped when the record exists."""
    group, sid = r["group"], r["struct_id"]
    rp = rec_path(group, sid)
    if rp.is_file():
        return "skip", 0.0
    syms = str(r["symbols"]).split()
    pos = np.asarray(r["pos_A"], float).reshape(-1, 3)
    rec = dict(struct_id=sid, group=group, kind=r["kind"], rxn_ids=r["rxn_ids"], folds=r["folds"], dipoles=r["dipoles"],
               rxn_id=int(r["rxn_id"]), mol_id=r["mol_id"], fold_T1=int(r["fold_T1"]), n_atoms=len(syms),
               n_heavy=n_heavy(syms), charge=int(r["charge"]), mult=int(r["mult"]), src=r["src"],
               plan_disp=json.loads(r["disp"] or "{}"), code=CODE_VERSION, level="label",
               keywords=M.LABEL_GRAD_LINE.strip(), slurm_job=_jobtag(), host=socket.gethostname())
    if r["kind"].startswith("d0_"):                              # D0 plan rows only (gen_d0): D0 id / class
        rec.update(d0_id=int(r["d0_id"]), d0_class=str(r["d0_class"]))
    if r["kind"] in ("react_disp", "d0_react_disp") + D0_TS_DISP_KINDS:
        try:
            if r["kind"] == "react_disp":
                pos, rec["disp"] = mol_disp(r, syms, pos, ctx)
            elif r["kind"] == "d0_react_disp":
                pos, rec["disp"] = mol_disp(r, syms, pos, ctx, hdir=HESS_D0)
            else:
                pos, rec["disp"] = ts_disp_d0(r, syms, pos, ctx)
        except HessFail as e:
            rec.update(status="fail", reason=f"xtb_hessian: {e}", symbols=syms, pos_A=pos.tolist(), core_h=0.0,
                       attempts=[], finished=_now())
            M.write_json(rp, rec)
            return "fail", 0.0
    text = M.orca_grad_input(syms, pos, nprocs=1, maxcore=M.LABEL_MAXCORE_MB, charge=int(r["charge"]),
                             mult=int(r["mult"]), title=sid)
    work = ctx.local / sid
    atts, res = [], None
    try:
        for _ in (1, 2):
            res = engrad(work, text, syms, pos, ctx.env)
            atts.append(dict(res["att"], ok=res["ok"], reason=res["reason"]))
            if res["level_bad"]:
                raise StopAll(f"{sid}: EnGrad not at the label level ({res['reason']})")
            if res["ok"]:
                break
        if (work / "g.out").is_file():
            write_gz(RUNS / group / f"{sid}.out.gz", work / "g.out")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    core = sum(a["core_h"] for a in atts)
    rec.update(symbols=syms, pos_A=np.asarray(pos, float).tolist(), attempts=atts, n_attempts=len(atts), core_h=core,
               wall_s=sum(a["wall_s"] for a in atts), finished=_now())
    if res["ok"]:
        g = res["grad"]
        rec.update(status="ok", reason="", energy_eh=res["energy_eh"], grad_eh_bohr=g.tolist(),
                   max_abs_grad=float(np.abs(g).max()), rms_grad=float(np.sqrt(np.mean(g ** 2))),
                   e_out_eh=res["e_out_eh"], de_out_engrad_eh=res["de_out_engrad_eh"], pos_dev_A=res["pos_dev_A"],
                   scf_cycles=res["scf_cycles"])
    else:
        rec.update(status="fail", reason=res["reason"])
    M.write_json(rp, rec)
    return rec["status"], core


def slice_units(units, n, i):
    """LPT: units = [(key, weight, payload)] -> payloads of slice i (largest first), load per slice."""
    U = sorted(units, key=lambda u: (-u[1], u[0]))
    heap = [(0.0, b) for b in range(n)]
    bins = [[] for _ in range(n)]
    loads = [0.0] * n
    for u in U:
        ld, b = heapq.heappop(heap)
        bins[b].append(u)
        loads[b] = ld + u[1]
        heapq.heappush(heap, (loads[b], b))
    return [u[2] for u in bins[i]], loads


def dry_report(name, units, n):
    """--dry: slice sizes and n_atoms^3 loads of every slice (nothing computed)."""
    out = []
    for b in range(n):
        mine, loads = slice_units(units, n, b)
        out.append(dict(slice=b, n_units=len(mine), load_rel=loads[b] / max(1e-12, sum(loads) / n)))
    print(json.dumps(dict(dry=name, n_slices=n, n_units=len(units), slices=out)), flush=True)
    return 0


def cmd_run(group, i, n, only=None, dry=False, rev=False):
    """rev (--rev, decision G-helper): the same slice in reverse order (smallest first). A helper task
    `run <group> 0 1 --rev` walks the whole group from the small end while the n-slice tasks walk theirs from the
    large end; run_struct skips structures whose record exists when it claims them, so the two meet without
    redoing finished work (only the structures in flight at the meeting point can run twice).
    D0G1 / D0G3 (gen_d0 plan_d0): the same code on $MLT/dft/plan/plan_d0.parquet, records runs/<D0G1|D0G3>/, a D0
    reaction's 3 TS rows (one xTB TS Hessian) and a D0 molecule's 3 rows stay in one slice (UNIT_KINDS)."""
    if group not in ("G1", "G3", "D0G1", "D0G3"):
        print("run: group must be G1, G3, D0G1 or D0G3", file=sys.stderr)
        return 2
    d0 = group.startswith("D0")
    if (d0_stopped() if d0 else stopped()):
        return 3
    P = load_plan(d0=d0)
    G = P[P.group == group]
    if only:
        rows = [r for r in G.to_dict("records") if r["struct_id"] in only]
        loads = None
    else:
        if not 0 <= i < n:
            print(f"run: slice {i} outside 0..{n - 1}", file=sys.stderr)
            return 2
        units = {}
        for r in G.to_dict("records"):
            key = r["mol_id"] if r["kind"] in UNIT_KINDS else r["struct_id"]
            units.setdefault(key, []).append(r)
        U = [(k, sum(float(x["n_atoms"]) ** 3 for x in v), v) for k, v in units.items()]
        if dry:
            return dry_report(f"run {group}", U, n)
        mine, loads = slice_units(U, n, i)
        rows = [x for v in mine for x in v]
        if rev:
            rows = [x for v in mine[::-1] for x in v]                 # unit order reversed, a molecule's rows together
    todo = [r for r in rows if not rec_path(group, r["struct_id"]).is_file()]
    print(f"[run {group} {i}/{n}{' rev' if rev else ''}] {len(rows)} structures, {len(todo)} to do, {_cpus()} workers, job {_jobtag()} on "
          f"{socket.gethostname()}, {time_left_h():.1f} h left", flush=True)
    ctx = Ctx("g")
    t0 = time.time()
    run_items(todo, _cpus(), ctx, RESERVE_H, run_struct, lambda r: r["struct_id"])
    left = [r["struct_id"] for r in rows if not rec_path(group, r["struct_id"]).is_file()]
    summ = dict(group=group, slice=i, n_slices=n, only=sorted(only) if only else None, n_rows=len(rows),
                n_todo_at_start=len(todo), stats=dict(ctx.stats), core_h=ctx.core_h, wall_s=time.time() - t0,
                n_left=len(left), left=left[:50], halt=ctx.halt, timed_out=ctx.timed_out,
                load_this=loads[i] if loads else None, load_all=loads, slurm_job=_jobtag(), host=socket.gethostname(),
                finished=_now(), code=CODE_VERSION)
    if not only:
        M.write_json(STEPS / f"run_{group}{'_rev' if rev else ''}_{i:02d}of{n:02d}.json", summ)
    print(json.dumps({k: summ[k] for k in ("stats", "core_h", "n_left", "halt", "timed_out")}), flush=True)
    if ctx.halt:
        return 3
    return 0 if not left else 2


# ================================================================ G-2 (MLTENV=d1)
def d1_g3():
    """(D, opt, g3l) of analysis/d1_g3 with D1G3_SCRATCH pointed under $MLT (decision G-2run)."""
    os.environ["D1G3_SCRATCH"] = str(D1G3_SHADOW)
    p = str(M.ANALYSIS / "d1_g3")
    if p not in sys.path:
        sys.path.insert(0, p)
    import g3l as G3L                                                          # noqa: E402
    O = G3L.O
    if Path(os.path.abspath(G3L.C.G3)) != Path(os.path.abspath(D1G3_SHADOW)):
        raise StopAll(f"d1_g3 G3 root {G3L.C.G3} is not the shadow {D1G3_SHADOW}")
    if int(O.PARTB_NPROCS) != G2_NPROCS:
        raise StopAll(f"opt.PARTB_NPROCS {O.PARTB_NPROCS} != {G2_NPROCS}")
    return O.d1(), O, G3L


def g2_text(D, G3L, syms, xyz, q, pairs):
    """the G3L OptTS input (g3l.g3l_text; Part B text with the InHess lines)."""
    return G3L.g3l_text(D, list(syms), np.asarray(xyz, float), int(q), [list(p) for p in pairs])[0]


def parse_g2_out(path):
    """orca_io.read_out_steps + consecutive duplicate geometries dropped."""
    steps, info = OI.read_out_steps(path)
    keep, ndup = [], 0
    for s in steps:
        if (keep and s["pos_A"].shape == keep[-1]["pos_A"].shape
                and float(np.abs(s["pos_A"] - keep[-1]["pos_A"]).max()) < 1e-7
                and abs(s["energy_eh"] - keep[-1]["energy_eh"]) < 1e-9):
            ndup += 1
            continue
        keep.append(s)
    info["n_duplicates_dropped"] = ndup
    return keep, info


def g2_guard(D, G3L, rxns=None, full=False):
    """the G-2 input builder against G3L unit inputs: header (text before '*xyz') must be byte identical; coordinates
    compared too (the G2 TS files may have been rewritten since); full = also parse the G3L outputs."""
    labs = {int(d["rxn_id"]): d for d in M.read_json(M.LABELS)}
    units = sorted(p for p in G3_UNITS.glob("L.*") if (p / "optts" / "ts.inp").is_file())
    if rxns:
        units = [p for p in units if int(p.name[2:]) in rxns]
    elif not full:
        units = units[:1]
    out = []
    for p in units:
        rid = int(p.name[2:])
        d = labs[rid]
        syms, xyz = M.read_xyz(M.G2_FINAL / str(rid) / "ts.xyz")
        q = int(d.get("charge1") or 0) + int(d.get("charge2") or 0)
        text = g2_text(D, G3L, syms, xyz, q, label_pairs(d["formed_pairs_ts"]))
        ref = (p / "optts" / "ts.inp").read_text()
        h1, h2 = text.split("*xyz")[0], ref.split("*xyz")[0]
        r = dict(unit=p.name, rxn_id=rid, identical=text == ref, header_identical=h1 == h2)
        if not r["header_identical"]:
            a, b = h1.splitlines(), h2.splitlines()
            k = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), min(len(a), len(b)))
            r["first_header_diff"] = [k, a[k] if k < len(a) else None, b[k] if k < len(b) else None]
        if full:
            r.update(g2_check_parse(p, syms))
        out.append(r)
    ok = bool(out) and all(x["header_identical"] for x in out) and all(x.get("parse_ok", True) for x in out)
    return dict(ok=ok, n=len(out), n_identical=sum(x["identical"] for x in out),
                n_header_identical=sum(x["header_identical"] for x in out), units=out)


def g2_check_parse(p, syms):
    """parse a G3L ts.out.gz with the run_g2 step parser and compare it with the unit's own record."""
    st = _rj(p / "optts" / "step.json") or {}
    steps, info = parse_g2_out(p / "optts" / "ts.out.gz")
    fin = M.read_xyz(p / "optts" / "ts.xyz")
    r = dict(n_steps=len(steps), n_opt_cycles_unit=st.get("n_opt_cycles"), parse_info=info,
             converged_unit=st.get("converged"))
    if not steps:
        return dict(r, parse_ok=False)
    last = steps[-1]
    e_fin = st.get("e_last_eh")                     # last FINAL SINGLE POINT ENERGY = the stationary-point evaluation
    r.update(last_vs_final_xyz_A=float(np.abs(last["pos_A"] - np.asarray(fin[1], float)).max()),
             last_energy_eh=last["energy_eh"], e_final_eval_eh=e_fin,
             de_final_eval_minus_last_kcal=(e_fin - last["energy_eh"]) * M.EH2KCAL if e_fin is not None else None,
             last_max_abs_grad=float(np.abs(last["grad_eh_bohr"]).max()),
             first_max_abs_grad=float(np.abs(steps[0]["grad_eh_bohr"]).max()),
             symbols_equal=all(list(s["symbols"]) == list(syms) for s in steps),
             energy_drop_kcal=(steps[0]["energy_eh"] - last["energy_eh"]) * M.EH2KCAL)
    # the converged geometry (ts.xyz, energy only) is one sub-threshold step past the last gradient step
    r["parse_ok"] = bool(r["symbols_equal"] and r["n_steps"] == r["n_opt_cycles_unit"]
                         and r["last_vs_final_xyz_A"] < G2_FINAL_STEP_TOL_A
                         and (e_fin is None or abs(r["de_final_eval_minus_last_kcal"]) < 0.1))
    return r


def cmd_g2_check(rxns=None):
    D, O, G3L = d1_g3()
    g = g2_guard(D, G3L, rxns=rxns, full=True)
    rep = dict(phase="G-2 input / parser check", code=CODE_VERSION, written=_now(), host=socket.gethostname(),
               rule="g2_text (= g3l.g3l_text) on $G2/g2_final/<rxn>/ts.xyz with the label charge / forming bonds vs "
                    "$G3/O/units/L.<rxn>/optts/ts.inp; parse_g2_out on the unit's ts.out.gz vs its step.json / ts.xyz",
               **g)
    M.write_json(G2_CHECK, rep)
    print(json.dumps({k: rep[k] for k in ("ok", "n", "n_identical", "n_header_identical")}), flush=True)
    for u in g["units"]:
        print(json.dumps({k: u.get(k) for k in ("unit", "identical", "header_identical", "n_steps", "n_opt_cycles_unit",
                                                "last_vs_final_xyz_A", "last_max_abs_grad", "parse_ok")}), flush=True)
    return 0 if g["ok"] else 3


def shifted_hess_text(path, shift_bohr):
    """the .hess text with every $atoms coordinate shifted by shift_bohr (x, y, z): a rigid translation."""
    lines = Path(path).read_text().splitlines()
    j = next(k for k, ln in enumerate(lines) if ln.strip() == "$atoms")
    n = int(lines[j + 1].split()[0])
    for k in range(j + 2, j + 2 + n):
        t = lines[k].split()
        lines[k] = f" {t[0]:<3s}{float(t[1]):12.5f}{float(t[2]) + shift_bohr:20.12f}{float(t[3]) + shift_bohr:20.12f}" \
                   f"{float(t[4]) + shift_bohr:20.12f}"
    return "\n".join(lines) + "\n"


def cmd_g2_probe(rid, maxiter=2, shift_bohr=1.0):
    """diagnostic (decision G-2sel): the first `maxiter` OptTS cycles of the G-2 input of rid (no Freq) with the
    original g2.hess and with a rigidly translated copy; identical steps = ORCA ignores the $atoms coordinates of an
    InHess file, so translated G2 Hessians are usable. -> results/G2_probe_<rid>.json"""
    D, O, G3L = d1_g3()
    labs = {int(d["rxn_id"]): d for d in M.read_json(M.LABELS)}
    d = labs[int(rid)]
    syms, xyz = M.read_xyz(M.G2_FINAL / str(rid) / "ts.xyz")
    hp = M.G2_FINAL / str(rid) / "work" / "ts.hess"
    text = g2_text(D, G3L, syms, xyz, int(d.get("charge1") or 0) + int(d.get("charge2") or 0),
                   label_pairs(d["formed_pairs_ts"]))
    lines = text.split("\n")
    if lines[0].split()[:3] != ["!", "OptTS", "Freq"] or sum(ln.strip() == "MaxIter 100" for ln in lines) != 1:
        raise StopAll("probe: unexpected G-2 header")
    lines[0] = lines[0].replace(" Freq", "", 1)
    lines = [f"MaxIter {int(maxiter)}" if ln.strip() == "MaxIter 100" else ln for ln in lines]
    probe = "\n".join(lines)
    local = Path(os.environ.get("TMPDIR") or "/tmp") / f"mlt_probe_{_jobtag()}_{os.getpid()}"
    out = {}
    try:
        for tag, htxt in (("original", Path(hp).read_text()), ("translated", shifted_hess_text(hp, shift_bohr))):
            w = local / tag
            w.mkdir(parents=True, exist_ok=True)
            (w / "ts.inp").write_text(probe)
            (w / G3L.HESS_NAME).write_text(htxt)
            t0 = time.time()
            with open(w / "ts.out", "w") as fo, open(w / "ts.err", "w") as fe:
                p = subprocess.run([str(D["orca"]), "ts.inp"], cwd=str(w), stdout=fo, stderr=fe, env=D["env"])
            txt = (w / "ts.out").read_text(errors="replace")
            steps, info = parse_g2_out(w / "ts.out")
            out[tag] = dict(rc=p.returncode, wall_s=time.time() - t0, inhess=G3L.inhess_evidence(txt), info=info,
                            energies=[s["energy_eh"] for s in steps],
                            warnings=[ln.strip() for ln in txt.splitlines() if "WARNING" in ln.upper()][:20], steps=steps)
        a, b = out["original"]["steps"], out["translated"]["steps"]
        n = min(len(a), len(b))
        cmp_ = dict(n_compared=n, max_pos_diff_A=[float(np.abs(a[k]["pos_A"] - b[k]["pos_A"]).max()) for k in range(n)],
                    energy_diff_eh=[float(a[k]["energy_eh"] - b[k]["energy_eh"]) for k in range(n)])
        cmp_["identical"] = bool(n >= 2 and max(cmp_["max_pos_diff_A"]) < 1e-6 and max(abs(x) for x in cmp_["energy_diff_eh"]) < 1e-8)
        for tag in out:
            out[tag].pop("steps")
        rep = dict(phase="G-2 translated Hessian probe", rxn_id=int(rid), maxiter=int(maxiter), shift_bohr=shift_bohr,
                   written=_now(), host=socket.gethostname(), runs=out, compare=cmp_)
        M.write_json(M.RES / f"G2_probe_{int(rid)}.json", rep)
        print(json.dumps(dict(compare=cmp_, inhess={k: v["inhess"].get("read") for k, v in out.items()},
                              energies={k: v["energies"] for k, v in out.items()}), indent=1), flush=True)
        return 0 if cmp_["identical"] else 3
    finally:
        shutil.rmtree(local, ignore_errors=True)


def g2_unit(u, ctx, D, O, G3L):
    """one G-2 OptTS -> runs/G2/<rxn>/{unit.json (last), steps.json, ...}. A D0 unit (gen_d0) carries run_group 'D0G2'
    (runs/D0G2/<d0 rxn>/) and its own label_ts (the Coley TS); D1 units have neither key set differently."""
    rid = int(u["rxn_id"])
    ud = RUNS / u.get("run_group", "G2") / str(rid)
    uj = ud / "unit.json"
    if uj.is_file():
        return "skip", 0.0
    base = dict(rxn_id=rid, struct_id=u["struct_id"], job_id=u["job_id"], n_atoms=u["n_atoms"], dipole=u["dipole"],
                g1_class=u["g1_class"], fold_T1=u["fold"], rxn_ids=u["rxn_ids"], folds=u["folds"], dipoles=u["dipoles"],
                code=CODE_VERSION, level="label", slurm_job=_jobtag(), host=socket.gethostname())
    if "d0_id" in u:
        base.update(d0_id=int(u["d0_id"]), d0_class=u["d0_class"], charge=int(u["charge"]), run_group=u["run_group"])
    sp = ud / "starts.json"
    starts = _rj(sp) or []
    if len(starts) >= MAX_STARTS:
        M.write_json(uj, dict(base, status="wall_clipped", reason=f"started {len(starts)} times without finishing",
                              starts=starts, core_h=0.0, n_steps=0, finished=_now()))
        return "wall_clipped", 0.0
    M.write_json(sp, starts + [dict(job=_jobtag(), host=socket.gethostname(), started=_now())])
    syms, xyz = M.read_xyz(u["start"])
    xyz = np.asarray(xyz, float)
    hp = Path(u["hess"])
    if M.sha256(hp) != u["hess_sha256"]:
        M.write_json(uj, dict(base, status="error", reason="g2_hess_changed_since_plan", core_h=0.0, n_steps=0,
                              finished=_now()))
        return "error", 0.0
    pairs = [tuple(int(a) for a in p) for p in u["pairs"]]
    text = g2_text(D, G3L, syms, xyz, u["charge"], pairs)
    work = ctx.local / f"g2_{rid}"
    runs, txt = [], ""
    try:
        for k in (1, 2):
            shutil.rmtree(work, ignore_errors=True)
            work.mkdir(parents=True)
            (work / "ts.inp").write_text(text)
            shutil.copyfile(hp, work / G3L.HESS_NAME)
            t0 = time.time()
            with open(work / "ts.out", "w") as fo, open(work / "ts.err", "w") as fe:
                p = subprocess.run([str(D["orca"]), "ts.inp"], cwd=str(work), stdout=fo, stderr=fe, env=D["env"])
            wall = time.time() - t0
            txt = (work / "ts.out").read_text(errors="replace")
            term = "ORCA TERMINATED NORMALLY" in txt
            infra = (not term) and (p.returncode < 0 or not txt.strip()
                                    or ("SCF ITERATIONS" not in txt and "GEOMETRY OPTIMIZATION CYCLE" not in txt))
            runs.append(dict(run=k, rc=p.returncode, wall_s=wall, core_h=wall * G2_NPROCS / 3600, infra=infra,
                             host=socket.gethostname()))
            if not infra:
                break
        core = sum(r["core_h"] for r in runs)
        term = "ORCA TERMINATED NORMALLY" in txt
        conv = bool(D["OD"].opt_converged(txt))
        steps, info = parse_g2_out(work / "ts.out")
        ih = G3L.inhess_evidence(txt)
        fin = M.read_xyz(work / "ts.xyz") if (work / "ts.xyz").is_file() else None
        last = txt.rsplit("VIBRATIONAL FREQUENCIES", 1)[-1] if "VIBRATIONAL FREQUENCIES" in txt else ""
        freqs = [float(v) for v in RE_OUTFREQ.findall(last)]
        imag = [f for f in freqs if f < 0]
        shp = {}
        if fin is not None:
            if list(fin[0]) != list(syms):
                raise RuntimeError("ORCA final geometry has a different atom order")
            lts = M.read_xyz(u.get("label_ts") or M.label_ts(rid))
            fx, lx = np.asarray(fin[1], float), np.asarray(lts[1], float)
            shp = dict(label=O.shape(D, (list(fin[0]), fx), (list(lts[0]), lx), pairs),
                       start=O.shape(D, (list(fin[0]), fx), (list(syms), xyz), pairs),
                       d_form=O._dists(fx, pairs), d_form_label=O._dists(lx, pairs))
        n_trj = None
        tj = work / "ts_trj.xyz"
        if tj.is_file():
            n_trj = sum(1 for ln in tj.read_text().splitlines() if ln.strip().startswith("Coordinates from"))
        if runs[-1]["infra"]:
            status, reason = "error", "pipeline_infra"
        elif ih.get("not_read"):
            status, reason = "error", G3L.INHESS_FAIL
        elif not conv:
            status, reason = "not_converged", "optts_not_converged" if term else "orca_not_terminated"
        elif not term:
            status, reason = "fail", "orca_not_terminated"
        elif len(imag) != 1:
            status, reason = "fail", f"n_imag_{len(imag)}"
        else:
            status, reason = "ok", ""
        for j, s in enumerate(steps):
            s["kind"] = "ts_final_g2" if (conv and term and j == len(steps) - 1) else "g2_path"
        ud.mkdir(parents=True, exist_ok=True)
        M.write_text(ud / "ts.inp", text)
        for fn, gz in (("ts.out", True), ("ts.xyz", False), ("ts.hess", True), ("ts.err", True)):
            f = work / fn
            if not f.is_file() or (fn == "ts.err" and f.stat().st_size == 0):
                continue
            if gz:
                write_gz(ud / f"{fn}.gz", f)
            else:
                write_bytes(ud / fn, f.read_bytes())
        M.write_json(ud / "steps.json", dict(rxn_id=rid, symbols=list(syms), n_steps=len(steps), steps=[
            dict(cycle=int(s["cycle"]), kind=s["kind"], energy_eh=float(s["energy_eh"]), pos_A=s["pos_A"].tolist(),
                 grad_eh_bohr=np.asarray(s["grad_eh_bohr"], float).tolist()) for s in steps]))
        rec = dict(base, status=status, reason=reason, runs=runs, core_h=core, wall_s=sum(r["wall_s"] for r in runs),
                   terminated=term, converged=conv, n_steps=len(steps), parse_info=info, n_trj_frames=n_trj,
                   inhess=ih, imag_freqs_cm=imag, n_imag=len(imag) if freqs else None, shape=shp,
                   converged_to_label=(shp.get("label") or {}).get("same_ts"),
                   e_last_eh=float(steps[-1]["energy_eh"]) if steps else None,
                   last_max_abs_grad=float(np.abs(steps[-1]["grad_eh_bohr"]).max()) if steps else None,
                   e_final_eval_eh=float(RE_FSPE.findall(txt)[-1]) if RE_FSPE.findall(txt) else None,
                   final_vs_last_step_A=(float(np.abs(np.asarray(fin[1], float) - steps[-1]["pos_A"]).max())
                                         if (fin is not None and steps) else None),
                   header=text.split("*xyz")[0], hess_source=str(hp), hess_sha256=u["hess_sha256"],
                   orca=str(D["orca"]), finished=_now())
        M.write_json(uj, rec)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if ih.get("not_read"):
        raise StopAll(f"G-2 {rid}: {G3L.INHESS_FAIL} (g2.hess not read by ORCA)")
    return status, core


def cmd_run_g2(i, n, only=None, dry=False, d0=False):
    """d0 (--d0): the D0G2 units of g2_plan_d0.json (with --only: any unit of the ranked pool g2_pool_d0.json, e.g. the
    g0_d0 pilot); records runs/D0G2/<d0 rxn>/; the guard is the same D1 G3L input check (same builder)."""
    if (d0_stopped() if d0 else stopped()):
        return 3
    gp, grp = (G2_PLAN_D0, "D0G2") if d0 else (G2_PLAN, "G2")
    if not gp.is_file():
        M.not_ready(f"{gp} missing: run 'gen_dft.py {'plan_d0' if d0 else 'plan'}' first")
    units = M.read_json(gp)["rxns"]
    if d0 and only and G2_POOL_D0.is_file():
        units = M.read_json(G2_POOL_D0)["ranked"]
    if dry:
        return dry_report(f"run_g2{' --d0' if d0 else ''}", [(str(u["rxn_id"]), float(u["n_atoms"]) ** 3, u)
                                                             for u in units], n)
    try:
        D, O, G3L = d1_g3()
        g = g2_guard(D, G3L)
    except StopAll as e:
        write_stop(str(e))
        return 3
    if not g["ok"]:
        write_stop(f"G-2 input builder differs from the G3L input ({g['units'][:1]})")
        return 3
    if only:
        mine, loads = [u for u in units if int(u["rxn_id"]) in only], None
    else:
        if not 0 <= i < n:
            print(f"run_g2: slice {i} outside 0..{n - 1}", file=sys.stderr)
            return 2
        mine, loads = slice_units([(str(u["rxn_id"]), float(u["n_atoms"]) ** 3, u) for u in units], n, i)
    slots = max(1, _cpus() // G2_NPROCS)
    todo = [u for u in mine if not (RUNS / grp / str(u["rxn_id"]) / "unit.json").is_file()]
    print(f"[run_g2{' --d0' if d0 else ''} {i}/{n}] {len(mine)} reactions, {len(todo)} to do, {slots} slots x {G2_NPROCS} ranks, guard "
          f"{g['n_header_identical']}/{g['n']} header identical, job {_jobtag()} on {socket.gethostname()}, "
          f"{time_left_h():.1f} h left", flush=True)
    ctx = Ctx("g2")
    t0 = time.time()
    run_items(todo, slots, ctx, RESERVE_G2_H, lambda u, c: g2_unit(u, c, D, O, G3L), lambda u: u["rxn_id"])
    left = [u["rxn_id"] for u in mine if not (RUNS / grp / str(u["rxn_id"]) / "unit.json").is_file()]
    summ = dict(group=grp, slice=i, n_slices=n, only=sorted(only) if only else None, n_rxns=len(mine),
                n_todo_at_start=len(todo), stats=dict(ctx.stats), core_h=ctx.core_h, wall_s=time.time() - t0,
                n_left=len(left), left=left, halt=ctx.halt, timed_out=ctx.timed_out, guard=g, load_all=loads,
                slurm_job=_jobtag(), host=socket.gethostname(), finished=_now(), code=CODE_VERSION)
    if not only:
        M.write_json(STEPS / f"run_{grp}_{i:02d}of{n:02d}.json", summ)
    print(json.dumps({k: summ[k] for k in ("stats", "core_h", "n_left", "halt", "timed_out")}), flush=True)
    if ctx.halt:
        return 3
    return 0 if not left else 2


# ================================================================ G-0
def fit_power(N, y):
    N, y = np.asarray(N, float), np.asarray(y, float)
    k = (N > 0) & (y > 0)
    N, y = N[k], y[k]
    if len(N) >= 3 and len(np.unique(N)) >= 2:
        b, la = np.polyfit(np.log(N), np.log(y), 1)
        a, b = float(np.exp(la)), float(b)
        res = np.log(y) - np.log(a * N ** b)
        return dict(a=a, b=b, n=int(len(N)), rms_log=float(np.sqrt(np.mean(res ** 2))),
                    model="core_h = a * n_atoms^b (log-log least squares)")
    if len(N):
        return dict(a=float(np.mean(y / N ** 3)), b=3.0, n=int(len(N)), model="fallback: b = 3, a = mean(core_h / N^3)")
    return None


def g3l_points():
    pts = []
    for p in sorted(G3_UNITS.glob("L.*")):
        u = _rj(p / "unit.json")
        x = p / "optts" / "ts.xyz"
        if u and u.get("core_h") and x.is_file():
            pts.append(dict(src=p.name, n_atoms=int(x.read_text().split()[0]), core_h=float(u["core_h"]),
                            status=u.get("status"), n_opt_cycles=u.get("n_opt_cycles")))
    return pts


def pick_pilots(units, k):
    pool = [u for u in units if u["g1_class"] == "different"] or list(units)
    pool = sorted(pool, key=lambda u: (u["n_atoms"], u["rxn_id"]))
    if not pool or k <= 0:
        return []
    pos = sorted({int(round((j + 1) / (k + 1) * (len(pool) - 1))) for j in range(k)})
    return [int(pool[p]["rxn_id"]) for p in pos]


def run_g2_subprocess(ids, d0=False):
    """the G-2 pilots: 'gen_dft.py run_g2 0 1 --only ...' in the D1 env (jobs/env.sh MLTENV=d1); d0 adds --d0."""
    cmd = ["bash", "-c", 'cd "$0" && export MLTENV=d1 && source jobs/env.sh && exec "$PY" gen_dft.py run_g2 0 1 --only "$1"'
           + (" --d0" if d0 else ""), str(HERE), ",".join(str(x) for x in ids)]
    env = dict(os.environ, SLURM_CPUS_PER_TASK=str(max(G2_NPROCS, _cpus())))
    t0 = time.time()
    p = subprocess.run(cmd, env=env)
    return p.returncode, time.time() - t0


def label_row(x):
    rid = int(x["rxn_id"])
    syms, pos = M.read_xyz(M.label_ts(rid))
    return row(f"g1_{rid}_lab", "G1", "ts_label", syms, pos, dict(rxn_ids=x["rxn_ids"], folds=x["folds"],
                                                                  dipoles=x["dipoles"]),
               M.label_ts(rid), int(x["charge"]), rxn_id=rid, fold_T1=int(x["fold_T1"]), g1_class=x["g1_class"],
               disp=json.dumps(dict(undisplaced=True)))


def cmd_g0(n_rand=G0_N_RAND, n_label=G0_N_LABEL, n_g2=G0_N_G2, label_rxns=None):
    if stopped():
        return 3
    P = load_plan()
    rng = np.random.default_rng(M.SEED + 3)
    pool = P[P.group.isin(["G1", "G3"])].reset_index(drop=True)
    pick = []
    if n_rand > 0 and len(pool):
        nb = min(5, n_rand)
        rank = pool.n_atoms.rank(method="first").to_numpy()
        b = np.minimum(nb - 1, np.floor((rank - 1) * nb / len(pool)).astype(int))
        per = [n_rand // nb + (1 if k < n_rand % nb else 0) for k in range(nb)]
        for k in range(nb):
            idx = np.flatnonzero(b == k)
            pick += [int(x) for x in rng.choice(idx, size=min(per[k], len(idx)), replace=False)]
    sample = pool.iloc[sorted(pick)].to_dict("records")
    g1 = P[P.kind == "ts_near"].drop_duplicates("rxn_id")
    if label_rxns:
        lab = g1[g1.rxn_id.isin(label_rxns)]
    else:
        o = g1.sort_values(["n_atoms", "rxn_id"]).reset_index(drop=True)
        qs = np.linspace(0.05, 0.95, n_label) if n_label > 1 else np.array([0.5] * max(0, n_label))
        lab = o.iloc[sorted({int(round(q * (len(o) - 1))) for q in qs})] if len(o) and n_label else o.iloc[0:0]
    lrows = [label_row(x) for x in lab.to_dict("records")]
    rows = sorted(sample + lrows, key=lambda r: (-int(r["n_atoms"]), r["struct_id"]))
    print(f"[g0] {len(sample)} plan EnGrads + {len(lrows)} label TSs on {_cpus()} workers; G-2 pilots {n_g2}", flush=True)
    ctx = Ctx("g0")
    t0 = time.time()
    run_items([r for r in rows if not rec_path(r["group"], r["struct_id"]).is_file()], _cpus(), ctx, RESERVE_H,
              run_struct, lambda r: r["struct_id"])
    if ctx.halt:
        return 3
    wall_e = time.time() - t0
    pil, rc_g2, wall_g2 = [], None, 0.0
    if n_g2 > 0 and G2_PLAN.is_file():
        pil = pick_pilots(M.read_json(G2_PLAN)["rxns"], n_g2)
        pend = [x for x in pil if not (RUNS / "G2" / str(x) / "unit.json").is_file()]
        if pend:
            print(f"[g0] G-2 pilots {pend} (MLTENV=d1 subprocess)", flush=True)
            rc_g2, wall_g2 = run_g2_subprocess(pend)
            if rc_g2 == 3:
                return 3
    est = g0_estimate(P, rows, pil, dict(wall_engrads_s=wall_e, wall_g2_s=wall_g2, rc_g2=rc_g2, stats=dict(ctx.stats)))
    M.write_json(G0_JSON, est)
    print(json.dumps({k: est[k] for k in ("predicted_core_h", "stop", "parity", "suggested_tasks")}, indent=1,
                     default=str), flush=True)
    if est["n_engrad_ok"] < 3:
        return 2
    return 3 if (est["stop"] or est["parity"].get("stop")) else 0


def g0_estimate(P, rows, pil, run_info):
    recs = [_rj(rec_path(r["group"], r["struct_id"])) for r in rows]
    eng = [x for x in recs if x and x.get("status") == "ok"]
    fails = [dict(struct_id=x["struct_id"], reason=x.get("reason")) for x in recs if x and x.get("status") != "ok"]
    per = [dict(struct_id=x["struct_id"], kind=x["kind"], n_atoms=x["n_atoms"], n_heavy=x.get("n_heavy"),
                core_h=x["core_h"], scf_cycles=x.get("scf_cycles"), n_attempts=x.get("n_attempts"),
                max_abs_grad=x.get("max_abs_grad")) for x in eng]
    fe = fit_power([x["n_atoms"] for x in eng], [x["core_h"] for x in eng])
    pool = P[P.group.isin(["G1", "G3"])]
    pred = {}
    if fe:
        for (g, k), x in pool.groupby(["group", "kind"]):
            pred[f"{g}/{k}"] = float((fe["a"] * x.n_atoms.astype(float) ** fe["b"]).sum())
    lab = [x for x in eng if x["kind"] == "ts_label"]
    hj = [_rj(p) for p in sorted(HESS_DIR.glob("*.json"))] if HESS_DIR.is_dir() else []
    hc = [x["core_h"] for x in hj if x and x.get("status") == "ok"]
    n_mols = int((P.kind == "react_ref").sum())
    pred["G3/xtb_hessians"] = float(np.mean(hc) * n_mols) if hc else 0.0
    pred["G1/ts_label_measured"] = float(sum(x["core_h"] for x in lab))
    pts = g3l_points()
    pp = []
    for rid in pil:
        u = _rj(RUNS / "G2" / str(rid) / "unit.json")
        if u and u.get("core_h"):
            pp.append(dict(src=f"G2.{rid}", n_atoms=int(u["n_atoms"]), core_h=float(u["core_h"]), status=u.get("status"),
                           n_steps=u.get("n_steps"), g1_class=u.get("g1_class")))
    fg = fit_power([x["n_atoms"] for x in pts + pp], [x["core_h"] for x in pts + pp])
    ratio = 1.0
    if fg and pp:
        ratio = max(1.0, float(np.mean([x["core_h"] / (fg["a"] * x["n_atoms"] ** fg["b"]) for x in pp])))
    g2p = P[P.group == "G2"]
    pred["G2/optts"] = float(ratio * (fg["a"] * g2p.n_atoms.astype(float) ** fg["b"]).sum()) if fg else None
    total = float(sum(v for v in pred.values() if v is not None))
    by_group = {g: float(sum(v for k, v in pred.items() if k.startswith(g + "/") and v is not None))
                for g in ("G1", "G2", "G3")}
    mg = [x["max_abs_grad"] for x in lab]
    parity = dict(n=len(mg), median_max_abs_grad=float(np.median(mg)) if mg else None,
                  max_max_abs_grad=float(np.max(mg)) if mg else None,
                  rms_grad=[x.get("rms_grad") for x in lab], per=[dict(struct_id=x["struct_id"], n_atoms=x["n_atoms"],
                                                                       max_abs_grad=x["max_abs_grad"]) for x in lab],
                  expect=PARITY_EXPECT, stop_above=PARITY_STOP)
    parity["pass"] = bool(mg) and parity["median_max_abs_grad"] <= PARITY_EXPECT
    parity["stop"] = bool(mg) and parity["median_max_abs_grad"] > PARITY_STOP
    cap = 10 * 64
    sugg = {g: max(1, int(np.ceil(by_group[g] / (64 * 24.0)))) for g in ("G1", "G3")}
    sugg["G2"] = max(1, int(np.ceil(by_group["G2"] / (64 * 24.0))))
    kinds = {}
    for x in per:
        kinds.setdefault(x["kind"], []).append(x["core_h"])
    return dict(phase="G-0", code=CODE_VERSION, written=_now(), host=socket.gethostname(), slurm_job=_jobtag(),
                decisions={k: DECISIONS[k] for k in ("G-0", "G-0fit", "G-parity")}, n_planned_engrads=int(len(pool)),
                n_planned_g2=int(len(g2p)), n_molecules=n_mols, n_engrad_measured=len(recs),
                n_engrad_ok=len(eng), engrad_failures=fails, per_engrad=sorted(per, key=lambda x: x["n_atoms"]),
                per_kind_core_h={k: _stats(v) for k, v in kinds.items()}, engrad_fit=fe, xtb_hessian_core_h=_stats(hc),
                g2_points_g3l=pts, g2_points_pilot=pp, g2_fit=fg, g2_pilot_ratio=ratio, g2_pilots=pil,
                predicted_core_h=dict(by_item=pred, by_group=by_group, total=total, budget=BUDGET_CORE_H),
                predicted_wall_h_10x64=total / cap, stop=total > BUDGET_CORE_H, parity=parity,
                suggested_tasks=dict(rule="ceil(predicted core-h / (64 cores x 24 h)) per group (64-core cpu2 tasks)",
                                     **sugg),
                run_info=run_info)


# ================================================================ collect
def cmd_collect():
    P = load_plan()
    G2P = _rj(G2_PLAN) or {"rxns": []}
    rep = dict(phase="G collect", code=CODE_VERSION, written=_now(), host=socket.gethostname(), groups={},
               hessian_source=DECISIONS["G-hess"], dedupe=(_rj(G_PLAN_JSON) or {}).get("dedupe"),
               decisions=DECISIONS)
    missing_total = 0
    e_by_rxn = {}
    for g in ("G1", "G3"):
        d = RUNS / g
        files = sorted(d.glob("*.json")) if d.is_dir() else []
        atoms, fails, core, nk, bad = [], [], Counter(), Counter(), 0
        de, seen = [], set()
        for f in files:
            r = _rj(f)
            if r is None:
                bad += 1
                continue
            seen.add(r["struct_id"])
            core[r["kind"]] += float(r.get("core_h") or 0.0)
            if r.get("status") != "ok":
                fails.append(dict(struct_id=r["struct_id"], kind=r["kind"], reason=r.get("reason")))
                continue
            nk[r["kind"]] += 1
            if r.get("de_out_engrad_eh") is not None:
                de.append(abs(r["de_out_engrad_eh"]))
            if g == "G1":
                e_by_rxn.setdefault(r["rxn_id"], {})[r["struct_id"]] = (r["kind"], r["energy_eh"],
                                                                         (r.get("plan_disp") or {}).get("e_harm_kcal"))
            atoms.append(M.ext_atoms(r["symbols"], r["pos_A"], r["energy_eh"], r["grad_eh_bohr"], struct_id=r["struct_id"],
                                     kind=r["kind"], source=SOURCE.get(r["kind"], r["kind"]), rxn_ids=r["rxn_ids"],
                                     folds=r["folds"], dipoles=r["dipoles"], level="label", group=g))
        M.write_extxyz(EXT[g], atoms)
        planned = set(P.loc[P.group == g, "struct_id"])
        miss = sorted(planned - seen)
        missing_total += len(miss)
        rep["groups"][g] = dict(n_records=len(files), n_unreadable=bad, n_ok=len(atoms), n_ok_by_kind=dict(nk),
                                n_fail=len(fails), fail_reasons=dict(Counter(x["reason"] for x in fails)),
                                failures=fails[:200], core_h_by_kind=dict(core), core_h=float(sum(core.values())),
                                n_planned=len(planned), n_missing=len(miss), missing=miss[:100],
                                n_extra_not_in_plan=len(seen - planned), out_engrad_energy_abs_diff_eh=_stats(de),
                                extxyz=str(EXT[g]))
    # G-2
    atoms, st, core, nsteps, ref = [], Counter(), 0.0, [], {}
    for u in G2P["rxns"]:
        ref[int(u["rxn_id"])] = u
    miss2 = []
    conv_label = []
    for rid, u in sorted(ref.items()):
        ud = RUNS / "G2" / str(rid)
        uj = _rj(ud / "unit.json")
        if uj is None:
            miss2.append(rid)
            continue
        st[uj.get("status")] += 1
        core += float(uj.get("core_h") or 0.0)
        if uj.get("converged_to_label") is not None:
            conv_label.append(bool(uj["converged_to_label"]))
        sj = _rj(ud / "steps.json")
        if not sj:
            continue
        nsteps.append(len(sj["steps"]))
        for j, s in enumerate(sj["steps"]):
            atoms.append(M.ext_atoms(sj["symbols"], s["pos_A"], s["energy_eh"], s["grad_eh_bohr"],
                                     struct_id=f"g2_{rid}_s{j:03d}", kind=s["kind"], source=SOURCE[s["kind"]],
                                     rxn_ids=u["rxn_ids"], folds=u["folds"], dipoles=u["dipoles"], level="label",
                                     group="G2", cycle=int(s["cycle"])))
    M.write_extxyz(EXT["G2"], atoms)
    missing_total += len(miss2)
    rep["groups"]["G2"] = dict(n_planned=len(ref), n_units=sum(st.values()), by_status=dict(st), core_h=core,
                               n_structures=len(atoms), n_by_kind=dict(Counter(a.info["kind"] for a in atoms)),
                               steps_per_rxn=_stats(nsteps), n_missing=len(miss2), missing=miss2[:100],
                               converged_to_label=dict(n=sum(conv_label), of=len(conv_label)), extxyz=str(EXT["G2"]))
    # Hessians, energies, G-0
    hj = [_rj(p) for p in sorted(HESS_DIR.glob("*.json"))] if HESS_DIR.is_dir() else []
    rep["xtb_hessians"] = dict(n=len(hj), by_status=dict(Counter((x or {}).get("status") for x in hj)),
                               core_h=float(sum((x or {}).get("core_h") or 0.0 for x in hj)),
                               n_with_imag=sum(bool((x or {}).get("imag_cm")) for x in hj))
    spread, vs_label = [], []
    for rid, d in e_by_rxn.items():
        es = [v[1] for v in d.values()]
        if len(es) > 1:
            spread.append((max(es) - min(es)) * M.EH2KCAL)
        lab = [v[1] for v in d.values() if v[0] == "ts_label"]
        if lab:
            for sid, (k, e, eh) in d.items():
                if k != "ts_label":
                    vs_label.append(dict(struct_id=sid, kind=k, de_dft_kcal=(e - lab[0]) * M.EH2KCAL, e_harm_xtb_kcal=eh))
    rep["energy_checks"] = dict(g1_spread_within_rxn_kcal=_stats(spread), g1_vs_label_ts=vs_label[:100],
                                note="de_dft = E(displaced) - E(label TS) at the label level; e_harm = xTB harmonic "
                                     "energy of the displacement (ts_near only) — same order expected (unit check)")
    g0 = _rj(G0_JSON) or {}
    rep["g0"] = dict(parity=g0.get("parity"), predicted_core_h=g0.get("predicted_core_h"), engrad_fit=g0.get("engrad_fit"),
                     g2_fit=g0.get("g2_fit"))
    rep["core_h_measured"] = dict(G1=rep["groups"]["G1"]["core_h"], G2=core, G3=rep["groups"]["G3"]["core_h"],
                                  xtb_hessians=rep["xtb_hessians"]["core_h"])
    rep["core_h_measured"]["total"] = float(sum(rep["core_h_measured"].values()))
    rep["n_missing"] = missing_total
    rep["complete"] = missing_total == 0
    rep["roundtrip"] = roundtrip_check()
    rep["stop_file"] = STOP_FILE.read_text().strip() if STOP_FILE.is_file() else None
    M.write_json(G_REPORT, rep)
    print(json.dumps(dict(complete=rep["complete"], n_missing=missing_total, core_h=rep["core_h_measured"],
                          ok={g: rep["groups"][g].get("n_ok", rep["groups"][g].get("n_structures")) for g in rep["groups"]}),
                     indent=1), flush=True)
    return 0 if rep["complete"] else 2


def roundtrip_check():
    """read the first structure of each extxyz back (ase) and compare energy / forces / tag types."""
    from ase.io import read
    out = {}
    for g, p in EXT.items():
        if not p.is_file() or p.stat().st_size == 0:
            continue
        try:
            a = read(str(p), index=0)
            out[g] = dict(struct_id=a.info.get("struct_id"), REF_energy_eV=a.info.get("REF_energy"),
                          e_eh_x_EH2EV=float(a.info.get("e_eh")) * M.EH2EV, has_forces="REF_forces" in a.arrays,
                          info_types={k: type(a.info.get(k)).__name__ for k in ("rxn_ids", "folds", "dipoles")})
        except Exception as e:                                                 # noqa: BLE001
            out[g] = dict(error=f"{type(e).__name__}: {e}")
    return out


# ================================================================ main
def _ints(s):
    return {int(x) for x in str(s).split(",") if x.strip()} if s else None


def main(argv=None):
    ap = argparse.ArgumentParser(description="DEPLOY-V2 (MLTS) Phase G", add_help=True)
    sub = ap.add_subparsers(dest="cmd")
    a = sub.add_parser("plan")
    a.add_argument("--limit", type=int, default=None)
    a.add_argument("--n-g2", type=int, default=N_G2)
    a.add_argument("--force", action="store_true")
    a = sub.add_parser("g0")
    a.add_argument("--n-rand", type=int, default=G0_N_RAND)
    a.add_argument("--n-label", type=int, default=G0_N_LABEL)
    a.add_argument("--n-g2", type=int, default=G0_N_G2)
    a.add_argument("--label-rxns", default=None)
    a = sub.add_parser("run")
    a.add_argument("group")
    a.add_argument("i")
    a.add_argument("n", type=int)
    a.add_argument("--only", default=None)
    a.add_argument("--dry", action="store_true")
    a.add_argument("--rev", action="store_true")
    a = sub.add_parser("run_g2")
    a.add_argument("i")
    a.add_argument("n", type=int)
    a.add_argument("--only", default=None)
    a.add_argument("--dry", action="store_true")
    a.add_argument("--d0", action="store_true")
    a = sub.add_parser("g2_check")
    a.add_argument("--rxns", default=None)
    a = sub.add_parser("g2_probe")
    a.add_argument("rxn", type=int)
    a.add_argument("--maxiter", type=int, default=2)
    a = sub.add_parser("collect")
    a.add_argument("--d0", action="store_true")
    a = sub.add_parser("plan_d0")
    a.add_argument("--limit", type=int, default=None)
    a.add_argument("--n-g2", default="auto")
    a.add_argument("--n-near", type=int, default=2)
    a.add_argument("--force", action="store_true")
    a = sub.add_parser("g0_d0")
    a.add_argument("--n-rand", type=int, default=30)
    a.add_argument("--n-label", type=int, default=10)
    a.add_argument("--n-g2", default="auto")
    a.add_argument("--label-rxns", default=None)
    a.add_argument("--max-atoms", type=int, default=None)
    try:
        x = ap.parse_args(argv)
    except SystemExit:
        return 2
    if x.cmd == "plan":
        return cmd_plan(x.limit, x.n_g2, x.force)
    if x.cmd == "g0":
        return cmd_g0(x.n_rand, x.n_label, x.n_g2, _ints(x.label_rxns))
    if x.cmd == "run":
        only = {s for s in str(x.only).split(",") if s} if x.only else None
        return cmd_run(x.group, M.array_task(x.i) if not (only or x.dry) else 0, x.n, only, x.dry, x.rev)
    if x.cmd == "run_g2":
        only = _ints(x.only)
        return cmd_run_g2(M.array_task(x.i) if not (only or x.dry) else 0, x.n, only, x.dry, x.d0)
    if x.cmd == "g2_check":
        return cmd_g2_check(_ints(x.rxns))
    if x.cmd == "g2_probe":
        return cmd_g2_probe(x.rxn, x.maxiter)
    if x.cmd == "collect":
        if x.d0:
            import gen_d0                                                      # noqa: E402  (D0 only)
            return gen_d0.cmd_collect_d0()
        return cmd_collect()
    if x.cmd == "plan_d0":
        import gen_d0                                                          # noqa: E402
        return gen_d0.cmd_plan_d0(x.limit, x.n_g2, x.force, x.n_near)
    if x.cmd == "g0_d0":
        import gen_d0                                                          # noqa: E402
        return gen_d0.cmd_g0_d0(x.n_rand, x.n_label, x.n_g2, _ints(x.label_rxns), x.max_atoms)
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
