#!/usr/bin/env python3
"""refs_v.py — D1-REF·LADDER Phase V (docs/specs/D1LADDER.md 3, V-1 .. V-5): confirmation calculations for the reference
structures ("is the lowest reference found = the lowest there is?"). sbatch only (python, xtb, ORCA, RDKit never on the
login node). Phase Q (refs_q.py) must have passed; its functions are imported, not copied.

  python refs_v.py targets                  # V-1 (LADENV=reactot): results/V_targets.csv, V_trial.csv, V_targets_report.json,
                                            #   $LAD/V/mols/<mid>/spec.json, $LAD/V/plan.json (trial part)
  python refs_v.py check                    # V-3 preflight (LADENV=d1): autodE ORCA input text, versions -> $LAD/V/check.json
  python refs_v.py work <trial|main> --nshards N [--shard i]
                                            # V-2 + V-3 for every molecule of shard i (default $SLURM_ARRAY_TASK_ID) of the list
                                            #   (LADENV=d1, --cpus-per-task=8: the D1 autodE %pal)
  python refs_v.py conf <mid> | dft <mid>   # one molecule: V-2 only / V-3 (+ summary) only (LADENV=d1)
  python refs_v.py chain <trial|main> [--round r] [--max-run K]
                                            # launcher (LADENV=reactot, compute node): resubmits unfinished shards (<= MAX_ROUNDS),
                                            #   trial complete -> decide (V-5) -> main array + next chain; main complete -> collect
  python refs_v.py decide                   # V-5 only, no submission -> results/V_decide.json
  python refs_v.py collect                  # V-4 (LADENV=reactot): $LAD/labels_d1_cr2.json, results/labels_d1_cr2.parquet,
                                            #   $LAD/F_pairs_cr2.csv, $LAD/refs_cr2.parquet, results/V_report.json, V_summary.md, ...
  python refs_v.py status                   # progress counts
Exit codes: 0 ok / up to date, 2 incomplete (resubmit) or usage, 3 STOP (spec 16 V; $LAD/V/STOP is written and every later
work / chain call exits 3 until it is removed).

Outputs (written only under $LAD and analysis/d1_ladder/results, atomically; every step is idempotent):
  $LAD/V/mols/<mid>/spec.json            the molecule (key, anchor, D1 references, label / HF templates)
                   conf.json, conf_all.xyz.gz, forms/f<n>.xyz, chosen/c<n>.xyz      V-2
                   c<n>/{result.json, opt1.*, freq1.*, [disp.xyz, opt2.*, freq2.*], final.xyz, sp_label.*, sp_hf.*}   V-3
                   repro/{repro_label,repro_hf}.*   (trial molecules only: the anchor label inputs rerun unchanged)
                   summary.json, .done   (outputs *.out.gz / *.hess.gz; no .gbw / .densities / .tmp are kept)
  results/V_targets.csv, V_trial.csv, V_targets_report.json, V_decide.json, V_report.json, V_summary.md (Korean),
          V_molecules.csv, V_conformers.csv, V_cyclooctyne_conformers.csv, labels_d1_cr2.parquet
  $LAD/labels_d1_cr2.json, $LAD/F_pairs_cr2.csv (the ground truth of every later phase), $LAD/refs_cr2.parquet (final
  reference of every (reaction, fragment): source, label energy, HF/6-31G* energy, geometry, inputs — spec 4-1 dft rung H)

Found (V-2 / V-3 tools and levels; also in V_report.json 'found')
  * RR tool: autodE 1.4.5 conformers/conf_gen.py get_simanl_conformer = make_reference.coley_port step 4 (D1 code copy) and
    autodE's own RR conformer generator (species/molecule.py _generate_conformers, used by autodE for SMILES with a ring of
    >= 8 atoms, i.e. every cyclooctyne: smiles/smiles.py init_organic_smiles 'max_ring_n >= 8' -> init_smiles), followed
    by Conformers.prune_on_energy(e_tol=1e-10) and prune_on_rmsd() (Config.rmsd_threshold 0.3 A). Its random state is
    np.random.RandomState() without a seed (conf_gen.py get_simanl_atoms) -> D1's conformers are not reproducible.
  * reactant DFT level (d1_run/config_run.yaml ts_stage + pilot_common.configure_autode, recorded in every
    ts_result.json 'keywords'): opt = 'Opt' B3LYP RIJCOSX D3BJ def2-SVP def2/J (+ def2-ECP, printed as an empty word),
    hess = 'Freq' with the same words; ORCA implicit solvation smd -> autodE wrappers/ORCA.py writes 'CPCM(Water)' on the
    '!' line and '%cpcm smd true SMDsolvent "water" end'; print_default_params (%output xyzfile, %scf maxiter 250,
    Hirshfeld, '% maxcore' 3000), '%pal nprocs 8' (cpus_per_task 8). autodE's Species.optimise is one ORCA Opt run and
    calc_thermo one ORCA Freq run (no convergence or imaginary-frequency check for reactants).
  * label SP inputs: $HF/inputs/<job>/orig/frag{k}_rel.inp = byte copies of sp/frag{k}_rel.inp of the D1 job archives
    (D1-HF prep.py, from the bundle tars / UBAI archive.tgz); written by build_inputs_graph.write_block, whose coordinate
    format is the one D1-HF prep.replace_coords writes. HF rule = prep.hf_text ('B3LYP' -> 'HF', 'def2-TZVP' -> '6-31G*'
    on the '!' line); $HF/inputs/<job>/hf/frag{k}_rel.inp = prep.hf_text of the orig input.

Decisions (user 2026-10-22: every branch = the recommended option; also in V_report.json 'decisions')
  V1  environments: targets / chain / decide / collect in reactot (pyarrow for Q_keys.parquet and the outputs); check /
      work / conf / dft in the D1 run environment (LADENV=d1: autodE 1.4.5 + its cconf_gen, RDKit 2024.3.2 = the version D1
      ran with, xtb 6.7.1, ORCA 6.1.1, OpenMPI 4.1.8f and the MPI variables of d1_run/nurion/env.sh).
  V2  V-1 targets (keys = Q-2 keys of $LAD/Q_keys.parquet): (a) keys with a reaction dev_kcal > 0.5; (b) dipolarophile keys
      with a row of skeleton D or Dp; constrained dipole keys are excluded from (a), (b) and (c) (counts reported); (c) the
      keys in neither, not constrained, sorted by key per kind; ONE numpy default_rng(20261023): first 10 dipolarophiles
      (rng.choice(n, 10, replace=False)), then 10 dipoles with the same generator. Molecule ids V0001.. in key order.
  V3  V-5 trial = the keys of cyclooctyne (C1#CCCCCCC1) and 3,3-dimethylcyclooctyne (CC1(C)C#CCCCCC1) + 8 keys of (a) (sorted,
      those two removed) drawn with a fresh default_rng(20261023).choice(n, 8, replace=False), in draw order.
      Either parent missing from the targets = STOP.
  V4  anchor = the lowest-label-energy D1 reference of the key (ties: smallest job_id) that has both D1-HF inputs (orig, hf);
      every structure (D1 references, ETKDG, RR, xTB, DFT) is put into the anchor's atom order by bond-graph isomorphism
      (fragmenter bond matrix, Si 1.11 A); the anchor's inputs are the SP templates. targets proves for every target that
      replace_coords(template, anchor xyz) == template byte for byte and hf template == prep.hf_text(template).
  V5  D1 forms = the key's D1 references deduplicated with the V-2 rule on the label energy (|dE| < 0.05 kcal/mol or heavy-atom
      RMSD < 0.1 A with mirror image and graph symmetry), lowest energy first. Forms are kept with their D1 DFT values (not
      recomputed, spec V-2); they are also xTB-relaxed so that they enter the V-2 dedupe pool first. The 5 DFT conformers =
      the 5 lowest-xTB search conformers that duplicate neither a form nor each other (so DFT is never spent on a structure
      D1 already has; '5 lowest + the D1 forms' of the spec).
  V6  ETKDG: RDKit ETKDGv3 defaults, randomSeed 20261023, 300 conformers, pruneRmsThresh -1 (no pre-pruning), with H,
      numThreads = cpus (results do not depend on it); zero conformers = 'ETKDG failed', RR only, recorded.
  V7  RR: get_simanl_conformer(species, dist_consts=None, conf_n=n, save_xyz=False) on make_reference.ade_mol of a form
      (no frozen dihedral: constrained keys are excluded), 300 starts round robin over the forms (lowest first), then the two
      autodE prunes above, as coley_port does. Deviation (recorded): start n uses RandomState(20261023 + n) instead of OS
      entropy, so V is reproducible; nothing else in the algorithm or its parameters changes.
  V8  xTB: xtb --opt tight --alpb water --gfn 2 --chrg q --uhf 0, one thread per process, cpus processes in parallel;
      not converged = dropped; bond graph not isomorphic to the anchor = dropped; stereo = refs_q.stereo3d_sig (mirror-merged
      RDKit 3D perception) must equal the key's reference signature computed in the same environment: the single valid
      signature of the key's D1 references, or, when they carry several, the signature of the Q E_min source (the lowest
      D1 reference); no valid signature at all, or several with an unusable E_min-source signature = the stereo of the key
      cannot be checked -> molecule failed (no V replacement, recorded; spec V-3 'stereo != key -> discard' cannot be
      applied otherwise; fixer 2026-10-22, the check could be switched off before). Dedupe on xTB energies (spec rule).
  V9  DFT text: autode_orca_text() reproduces autodE's input; check proves it byte for byte on the D1 run's own autodE inputs
      (T03433 archive ref/dipole_alt_{opt,hess}_orca.inp) and the pilot reactant inputs (J10 r0 opt / hess), and targets
      checks that ts_result.json 'keywords' opt / hess of every anchor job are the strings of EXPECTED_KW. ORCA is run
      directly (as autodE does: one Opt, then one Freq on the optimised geometry).
  V10 imaginary frequencies (ORCA .hess, projected), spec V-3 as written: any frequency < 0 -> one displacement along the
      most negative mode (unit-norm 3N vector x 1.0 A = autodE displaced_species_along_mode default disp_factor) and Opt +
      Freq again; any frequency < 0 after that = failure 'fail:imag' (discarded, counted in the 10 % gate of spec 16 V).
      Sensitivity only (reported, never used): how many of those would pass the D1 run's TS-stage tolerance
      gates.extra_imag_tol_cm (-50 cm-1) and the failure rate under it. (fixer 2026-10-22: the builder kept modes above
      -50 cm-1, a method change not in the spec; the D1 references keep their D1 values unchecked, as spec V-2 says.)
      DFT failure = ORCA error, Opt not converged, or 'imag'; failure rate = failures / conformers whose Opt started;
      > 10 % = STOP (trial and collect).
  V11 SPs: label SP = the anchor template with the coordinates replaced by prep.replace_coords (checked: same line count,
      every non-coordinate line identical, same '!' words / blocks / charge / multiplicity, elements in order), serial ORCA
      like the label SPs, no retry (keywords unchanged; SCF failure = discarded); output must show SMD water (Epsilon
      78.3550). HF SP: route A = prep.hf_text(label SP input), route B = replace_coords(D1-HF hf template); A == B byte for
      byte or STOP; D1-HF retry rule (run.retry_text once on SCF non-convergence). Bond graph or stereo of the final DFT
      geometry not the key's = discarded ('discard:graph' / 'discard:stereo', not DFT failures). A conformer is valid
      ('ok', can become V best) only when both SPs are ok: the final reference needs its HF value (spec 4-1 dft rung H);
      label ok + HF failed = 'discard:sp_hf' (fixer 2026-10-22). Every ORCA output (Opt, Freq, SPs) must report
      Program Version 6.1.1 (GIT 487d211c, hf_common) or STOP (spec V-3 'ORCA 6.1.1').
  V12 trial reproduction (added, spec 16 'any time'): the anchor label and HF inputs of the 10 trial molecules are rerun
      unchanged; |E - label| and |E - labels_d1_hf| must be <= 0.05 kcal/mol (the Q-4 SCF reproduction level) or STOP.
  V13 V-4: V best = the lowest label SP of a valid V conformer; the key's final reference changes iff E_min(Q) - E_V > 0.05
      kcal/mol; then every row of that key is replaced (its own reference is >= E_min > E_V + 0.05); other keys keep the Q
      override. refs_q.reassemble / gate_qa / gates / cr_table / write_outputs / write_pairs with tag cr2; added gates V-a
      (the Q audit recomputed from labels_d1.json = Q_keys.parquet), V-c (fragments of non-V keys identical to
      labels_d1_cr.json, whole rows when neither key changed), V-e (V references below E_min - 0.05 and <= cr references).
  V14 HF of the final references ($LAD/refs_cr2.parquet): own reference -> labels_d1_hf.parquet geom=dft e_frag{k}_rel_eh
      of the row; Q source -> that of the source row; V source -> the V HF SP.
  V15 cost (V-5): allocated core-h = wall x allocated CPUs of each stage; estimate = max(mean trial core-h x N, trial
      core-h x sum(n_atoms^3) / sum over the trial (n_atoms^3)) over all targets; <= 5,000 -> main runs, else STOP.
  V16 layout: trial = 2 array tasks (%2, 5 molecules each, largest first round robin); main = K = --max-run (default 4)
      tasks, molecules dealt round robin by size; each task 8 CPUs / 48G (the D1 worker slot); a heavy step is not
      started after 45 h (V_STOP_AFTER_H); the launcher 'chain' (2 CPUs, 24G, afterany) resubmits unfinished shards
      after waiting on the compute node for queue room (squeue -r, MaxSubmit 20); submissions use the launcher's
      environment without SLURM_* / SBATCH_* variables. Concurrency (spec 15: LADDER <= 4 jobs): the array throttle of a
      round = min(--max-run, 4 - LADDER tasks of the other streams in the queue (names lad_* / ladQ), at least 1).
      Round limits: trial 2; main max(4, ceil(remaining estimate / (K x 8 x 40 core-h)) + 2), fixed in plan.json when
      main is planned. Hitting the limit with unfinished shards = $LAD/V/STOP listing the unfinished molecules and their
      error.txt (fixer 2026-10-22: it used to stop silently with exit 2). A second 'chain trial' never re-plans or
      re-submits main once chain_log.json has a main round.
  V17 file name: spec 8 says refs.py (Q·V); the task names refs_v.py; Phase Q lives in refs_q.py.
  V18 per-molecule lock (fixer 2026-10-22): a work task holds $LAD/V/mols/<mid>/.lock (O_EXCL, its SLURM_JOB_ID) while
      it runs the molecule; a molecule locked by a job still in squeue is skipped ('locked', shard incomplete -> the next
      round); a lock of a job no longer in squeue is taken over. So a manual re-sbatch next to the chain is safe.
  V19 shared files of parallel array tasks ($LAD/V/check.json, $LAD/V/STOP) are written with a per-process temporary
      name (lad_common.write_atomic uses a fixed '<name>.tmp', which races when two tasks write the same file).
  V20 check also requires the mpirun on PATH to be the one under $MPI_ROOT (OpenMPI 4.1.8f of env.sh).
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True        # nothing compiled into the read-only code trees

import argparse  # noqa: E402
import getpass  # noqa: E402
import gzip  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import multiprocessing  # noqa: E402
import os  # noqa: E402
import platform  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import tarfile  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lad_common as L  # noqa: E402
import refs_q as Q  # noqa: E402

DP = L.DP
KCAL = L.KCAL
CODE_VERSION = "d1-ladder-v1"

# ---------------------------------------------------------------- paths
VDIR = L.LAD / "V"
MOLS = VDIR / "mols"
STOP_FILE = VDIR / "STOP"
PLAN = VDIR / "plan.json"
CHECK = VDIR / "check.json"
REFS_CR2 = L.LAD / "refs_cr2.parquet"
RES_TARGETS = L.RES / "V_targets.csv"
RES_TRIAL = L.RES / "V_trial.csv"
RES_TREP = L.RES / "V_targets_report.json"
RES_DECIDE = L.RES / "V_decide.json"
RES_REPORT = L.RES / "V_report.json"
RES_SUMMARY = L.RES / "V_summary.md"
RES_MOLS = L.RES / "V_molecules.csv"
RES_CONFS = L.RES / "V_conformers.csv"
RES_COT = L.RES / "V_cyclooctyne_conformers.csv"
D1_CONFIG = DP.D1_RUN / "config_run.yaml"
LAD_SLURM = "jobs/lad.slurm"

# ---------------------------------------------------------------- rules (spec 3)
DEV_A = 0.5                         # V-1 (a)
SKEL_B = ("D", "Dp")                # V-1 (b)
N_CTRL = 10                         # V-1 (c), per kind
N_ETKDG = 300                       # V-2 (1)
N_RR = 300                          # V-2 (2)
N_KEEP = 5                          # V-2: lowest 5
RMSD_DUP = 0.1                      # V-2 dedupe, A (heavy atoms, mirror + symmetry)
DE_DUP = 0.05                       # V-2 dedupe, kcal/mol
THR = Q.THR_REPLACE                 # V-4: 0.05 kcal/mol
REPRO_TOL = 0.05                    # V12, kcal/mol
CORE_H_MAX = 5000.0                 # V-5
DFT_FAIL_MAX = 0.10                 # V-5 / spec 16 V
N_TRIAL_RANDOM = 8                  # V-5
TRIAL_PARENTS = (("cyclooctyne", "C1#CCCCCCC1"), ("3,3-dimethylcyclooctyne", "CC1(C)C#CCCCCC1"))
TRIAL_SHARDS = 2
MAX_ROUNDS = {"trial": 2, "main": 4}
QUEUE_CAP = 20                      # MaxSubmit (array elements count)
DISP_FACTOR = 1.0                   # V10, A (autodE displaced_species_along_mode default)
XTB_TIMEOUT_S = 1800
MAX_AUTOMORPHISMS = 2000
STOP_AFTER_H = float(os.environ.get("V_STOP_AFTER_H", "45"))
T0 = float(os.environ.get("D1LADDER_JOB_T0", time.time()))
WORK_RES = dict(cpus=8, mem="48G")  # the D1 worker slot (ubai/bin/qsub: 8 cores, 48G)
CHAIN_RES = dict(cpus=2, mem="24G")

# autodE 1.4.5 ORCA text for the D1 run keywords (V9); proven in cmd_check on real D1 autodE inputs
BANG = {"opt": "! Opt B3LYP RIJCOSX D3BJ def2-SVP def2/J  CPCM(Water)",
        "hess": "! Freq B3LYP RIJCOSX D3BJ def2-SVP def2/J  CPCM(Water)"}
EXPECTED_KW = {
    "opt": "'Opt' Functional(B3LYP) ResolutionOfIdentity(rijcosx) DispersionCorrection(D3BJ) BasisSet(def2-SVP) "
           "'def2/J' EffectiveCorePotential(def2-ECP)",
    "hess": "'Freq' Functional(B3LYP) ResolutionOfIdentity(rijcosx) DispersionCorrection(D3BJ) BasisSet(def2-SVP) "
            "'def2/J' EffectiveCorePotential(def2-ECP)"}
ORCA_TEXT_SOURCES = (
    dict(name="d1_run T03433 (UBAI archive; make_reference.dft_refine of the Coley alt dipole)", d1_run=True,
         tar=DP.UBAI_JOBS / "T03433" / "archive.tgz",
         members={"opt": "ref/dipole_alt_opt_orca.inp", "hess": "ref/dipole_alt_hess_orca.inp"}),
    dict(name="pilot J10 (autodE reaction profile, reactant r0)", d1_run=False,
         dir=Path("/gpfs/tmp_cpu2/yeseo1ee/d1_autode_pilot/jobs/J10/ts/J10"),
         members={"opt": "reactants_and_products/r0_C6H3N3_opt_orca.inp", "hess": "thermal/r0_C6H3N3_hess_orca.inp"}),
)
RE_XTB_E = re.compile(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh")
RE_GEL = re.compile(r"G-E\(el\)\s+\.\.\.\s+(-?\d+\.\d+)\s+Eh")
RE_OPTCYC = re.compile(r"GEOMETRY OPTIMIZATION CYCLE\s+\d+")

FOUND = dict(
    rr_tool="autodE 1.4.5 conf_gen.get_simanl_conformer (make_reference.coley_port step 4; autodE RR conformer generator) "
            "+ Conformers.prune_on_energy(e_tol=1e-10), prune_on_rmsd() (0.3 A)",
    rr_seed_in_d1="np.random.RandomState() without a seed (autode/conformers/conf_gen.py get_simanl_atoms)",
    autode_rr_for_8_rings="autode/smiles/smiles.py init_organic_smiles: max_ring_n >= 8 -> init_smiles (RR conformers)",
    dft_level="autodE opt / hess keywords of the D1 run (ts_result.json 'keywords'): " + EXPECTED_KW["opt"] + " | "
              + EXPECTED_KW["hess"],
    orca_text="autode/wrappers/ORCA.py generate_input_for: '!' words, %cpcm smd block, %output / %scf / Hirshfeld, "
              "'% maxcore' 3000, %pal nprocs 8, '*xyz q m' with '{:<3} {:^12.8f} x3' lines",
    label_inputs="$HF/inputs/<job>/orig/frag{k}_rel.inp (D1-HF prep.py byte copies of the archive sp/*.inp)",
    hf_rule="D1-HF prep.hf_text (hf_common.SUB on the '!' line); $HF/inputs/<job>/hf/frag{k}_rel.inp",
)
DECISIONS = {
    "V1_env": "targets/chain/decide/collect reactot; check/work/conf/dft = D1 run env (autodE 1.4.5, RDKit 2024.3.2, "
              "xtb 6.7.1, ORCA 6.1.1, OpenMPI 4.1.8f)",
    "V2_targets": "(a) dev > 0.5; (b) dipolarophile keys with a D / Dp row; constrained excluded; (c) one "
                  "default_rng(20261023): 10 dipolarophiles then 10 dipoles from the sorted rest",
    "V3_trial": "cyclooctyne + 3,3-dimethylcyclooctyne + 8 of sorted (a) with a fresh default_rng(20261023)",
    "V4_anchor": "lowest label reference with D1-HF orig + hf inputs; all structures in its atom order; its inputs = templates",
    "V5_forms": "D1 references deduped (dE < 0.05 or RMSD < 0.1) = forms, kept with D1 values, xTB-relaxed into the dedupe "
                "pool first; DFT on the 5 lowest new search conformers",
    "V6_etkdg": "ETKDGv3 defaults, randomSeed 20261023, 300, no pre-pruning, with H",
    "V7_rr": "autodE get_simanl_conformer x 300 round robin over the forms + autodE prunes; seeded RandomState(20261023+n)",
    "V8_xtb": "--opt tight --alpb water --gfn 2 --chrg q --uhf 0; graph iso to anchor; stereo3d_sig = the key's signature "
              "(unique D1 signature, else the Q E_min source's); not checkable -> molecule failed (no V replacement)",
    "V9_dft_text": "autodE ORCA text reproduced byte for byte (checked on D1 run inputs); ORCA run directly",
    "V10_imag": "any freq < 0 -> one 1.0 A displacement + Opt/Freq; any freq < 0 after that = failure 'imag' (spec V-3); "
                "the -50 cm-1 (config extra_imag_tol_cm) count is a reported sensitivity only",
    "V11_sp": "label SP = template with coordinates replaced (checked), no retry; HF route A == route B, D1-HF retry rule; "
              "valid conformer = both SPs ok; every ORCA output = 6.1.1 (487d211c) or STOP",
    "V12_repro": "trial anchors rerun unchanged: |dE| <= 0.05 kcal/mol vs label and labels_d1_hf, else STOP",
    "V13_v4": "replace iff E_min(Q) - E_V > 0.05; Q-4/Q-5 via refs_q with tag cr2; gates V-a, V-c, V-e added",
    "V14_hf_refs": "HF of final references: own / Q source / V SP -> $LAD/refs_cr2.parquet",
    "V15_cost": "estimate = max(mean x N, n_atoms^3 scaling) of allocated core-h; <= 5000 -> go",
    "V16_layout": "trial 2 tasks %2; main K tasks (K = --max-run, default 4), throttle min(K, 4 - other LADDER tasks); "
                  "8 CPUs 48G; chain launcher resubmits; round limit (main from the estimate) -> STOP when hit",
    "V17_name": "refs_v.py (task) instead of refs.py (spec 8)",
    "V18_lock": "per-molecule O_EXCL lock with the SLURM job id; held by a live job -> skipped; dead owner -> taken over",
    "V19_shared_writes": "check.json / STOP written with a per-process temporary name (parallel array tasks)",
    "V20_mpirun": "check requires mpirun on PATH under $MPI_ROOT",
}
LADDER_MAX = 4                      # spec 15: LADDER uses at most 4 jobs at once (V and G0 / F0 / Q streams together)
LADDER_OTHER_PREFIX = ("lad_", "ladQ")   # job names of the G0 / F0 (lad_g0, lad_g0r, lad_f0s, lad_f0r) and Q (ladQ) jobs
ROUND_CORE_H = 40.0                 # usable hours of one 48 h round per CPU (heavy steps stop at 45 h; margin)


class MolFail(Exception):
    """deterministic failure of one molecule (it is complete, with no V conformer)."""


class Infra(Exception):
    """infrastructure problem (process killed): the molecule is retried by the next round."""


class TimeUp(Exception):
    pass


class VStop(Exception):
    """spec 16 V STOP condition found inside a worker."""


# ---------------------------------------------------------------- small helpers
def _now():
    return time.strftime("%F %T")


def _f(x):
    """float, or None for None / NaN / non-numbers."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _load_json(p, default=None):
    p = Path(p)
    if not p.is_file():
        return default
    return json.loads(p.read_text())


def _sha_text(s):
    import hashlib
    return hashlib.sha256(s.encode()).hexdigest()


def _q(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    if not len(x):
        return {}
    return dict(n=int(len(x)), min=float(x.min()), p10=float(np.quantile(x, 0.1)), p50=float(np.quantile(x, 0.5)),
                p90=float(np.quantile(x, 0.9)), max=float(x.max()), mean=float(x.mean()))


def _xyz_text(syms, xyz, title=""):
    """pilot_common.write_xyz format."""
    body = "\n".join(f"{s:<2} {x:15.8f} {y:15.8f} {z:15.8f}" for s, (x, y, z) in zip(syms, np.asarray(xyz, float)))
    return f"{len(syms)}\n{title}\n{body}\n"


def read_xyz(p):
    syms, xyz = Q.fragmenter().read_xyz(p)
    return list(syms), np.asarray(xyz, float)


def write_xyz(p, syms, xyz, title=""):
    return L.write_atomic(p, _xyz_text(syms, xyz, title))


def write_gz(dest, src):
    data = Path(src).read_bytes()
    return L.write_atomic(dest, lambda t: t.write_bytes(gzip.compress(data, 6)))


def n_cpus():
    return int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))


def code_sha():
    return L.sha256(Path(__file__).resolve())


def time_ok():
    return time.time() - T0 < STOP_AFTER_H * 3600


def write_shared(path, data):
    """V19: atomic write with a per-process temporary name, for files that parallel array tasks may write at the same
    time (lad_common.write_atomic always uses '<name>.tmp'). The last os.replace wins; nothing half-written is published."""
    path = L.assert_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{socket.gethostname()}.{os.getpid()}.tmp")
    tmp.write_bytes(data.encode() if isinstance(data, str) else data)
    os.replace(tmp, path)
    return path


def write_stop(msg):
    write_shared(STOP_FILE, f"{_now()} {msg}\n")


# ---------------------------------------------------------------- V18 per-molecule lock
def _job_tag():
    return os.environ.get("SLURM_JOB_ID") or f"pid{os.getpid()}@{socket.gethostname()}"


def _job_alive(jid):
    """True if SLURM job `jid` is still in the queue; unknown (not a job id, squeue error) = True (conservative: the
    molecule is skipped this round, never run twice)."""
    if not jid or not jid.isdigit():
        return True
    try:
        p = subprocess.run(["squeue", "-h", "-j", jid, "-o", "%T"], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return True
    if p.returncode == 0:
        return bool(p.stdout.strip())
    return "invalid job id" not in (p.stderr or "").lower()       # a job id the controller no longer knows = gone


def lock_molecule(mid):
    """take $LAD/V/mols/<mid>/.lock for this job (O_EXCL); False if a live job holds it. A dead owner's lock is taken
    over by os.replace and then re-read (the last writer wins, the others see a foreign owner and skip)."""
    p = L.assert_path(mdir(mid) / ".lock")
    p.parent.mkdir(parents=True, exist_ok=True)
    me = _job_tag()
    try:
        fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        with os.fdopen(fd, "w") as f:
            f.write(f"{me} {_now()} {socket.gethostname()}\n")
        return True
    except FileExistsError:
        pass
    try:
        owner = p.read_text().split()[0]
    except (OSError, IndexError):
        owner = ""
    if owner == me:
        return True
    if _job_alive(owner):
        return False
    write_shared(p, f"{me} {_now()} {socket.gethostname()} (took over from {owner})\n")
    try:
        return p.read_text().split()[0] == me
    except (OSError, IndexError):
        return False


def unlock_molecule(mid):
    p = mdir(mid) / ".lock"
    try:
        if p.read_text().split()[0] == _job_tag():
            p.unlink()
    except (OSError, IndexError):
        pass


def stop_report(path, rep, msg, hard=True):
    """report with stop true, exit 3; hard = also write $LAD/V/STOP (spec 16 V; a missing Phase Q is not hard)."""
    rep.update(stop=True, all_pass=False, stop_reason=msg, finished=_now())
    L.write_json(path, rep)
    if hard:
        write_stop(msg)
    print(f"STOP: {msg}", file=sys.stderr, flush=True)
    return 3


def mdir(mid):
    return MOLS / mid


def done(mid):
    return (mdir(mid) / ".done").is_file()


def load_spec(mid):
    p = mdir(mid) / "spec.json"
    return json.loads(p.read_text()), _sha_text(p.read_text())


# ---------------------------------------------------------------- read-only modules
def hf_prep():
    Q.hf_common()
    return Q._load("d1hf_prep", L.HF_DIR / "prep.py")


def hf_run():
    Q.hf_common()
    return Q._load("d1hf_run", L.HF_DIR / "run.py")


def pc_mod():
    return Q._load("pilot_common", Q.g2c().PILOT / "pilot_common.py")


def od_mod():
    return Q._load("orca_direct", Q.g2c().PILOT / "orca_direct.py")


def mr_mod():
    pc_mod()
    Q.stereo_utils()
    return Q._load("make_reference", Q.g2c().PILOT / "make_reference.py")


def d1_config():
    import yaml
    with open(D1_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_settings():
    """ORCA / xtb settings of the D1 run (config_run.yaml), overridable binaries from env.sh."""
    c = d1_config()
    return dict(n_cores=int(c["cpus_per_task"]), maxcore_mb=int(c["maxcore_mb"]),
                imag_tol_cm=float(c["gates"]["extra_imag_tol_cm"]),
                orca=os.environ.get("ORCA_BIN") or c["orca_bin"], xtb=os.environ.get("XTB_BIN") or c["xtb_bin"],
                ts_stage=c["ts_stage"], sp_stage=c["sp_stage"])


# ---------------------------------------------------------------- V9: autodE's ORCA input text
def autode_orca_text(kind, syms, xyz, charge, mult=1, n_cores=8, maxcore_mb=3000):
    """the input autodE 1.4.5 writes for the D1 run keywords (wrappers/ORCA.py generate_input_for)."""
    out = [BANG[kind], "%cpcm", "smd true", 'SMDsolvent "water"', "end",
           "%output ", "xyzfile=True ", "end ", "%scf ", "maxiter 250 ", "end", "%output", "Print[P_Hirshfeld] = 1",
           "end", "% maxcore", str(int(maxcore_mb))]
    if int(n_cores) > 1:
        out += [f"%pal nprocs {int(n_cores)}", "end"]
    out.append(f"*xyz {int(charge)} {int(mult)}")
    out += [f"{s:<3} {x:^12.8f} {y:^12.8f} {z:^12.8f}" for s, (x, y, z) in zip(syms, np.asarray(xyz, float))]
    out.append("*")
    return "\n".join(out) + "\n"


def parse_autode_inp(text):
    lines = text.split("\n")
    kind = "opt" if lines[0].startswith("! Opt ") else "hess" if lines[0].startswith("! Freq ") else None
    i = next(k for k, ln in enumerate(lines) if ln.startswith("*xyz"))
    _, q, m = lines[i].split()
    syms, xyz = [], []
    for ln in lines[i + 1:]:
        if ln.strip() == "*":
            break
        t = ln.split()
        syms.append(t[0])
        xyz.append([float(v) for v in t[1:4]])
    pal = re.search(r"^%pal nprocs (\d+)$", text, re.M)
    mc = int(lines[lines.index("% maxcore") + 1])
    return dict(kind=kind, syms=syms, xyz=np.array(xyz, float), charge=int(q), mult=int(m),
                n_cores=int(pal.group(1)) if pal else 1, maxcore_mb=mc)


# ---------------------------------------------------------------- geometry
def _edges(G):
    return {tuple(sorted(e)) for e in G.edges()}


def to_anchor(G_a, syms_a, syms, xyz):
    """coordinates in the anchor's atom order (bond-graph isomorphism, element labels), or None if not isomorphic."""
    from networkx.algorithms.isomorphism import GraphMatcher
    xyz = np.asarray(xyz, float)
    if len(syms) != len(syms_a):
        return None
    G = Q.bond_graph(syms, xyz)
    if list(syms) == list(syms_a) and _edges(G) == _edges(G_a):
        return xyz
    gm = GraphMatcher(G_a, G, node_match=lambda a, b: a["el"] == b["el"])
    if not gm.is_isomorphic():
        return None
    m = gm.mapping
    return xyz[[m[i] for i in range(len(syms_a))]]


def automorphisms(syms, xyz):
    """heavy-atom index list and its graph automorphisms (fragmenter.heavy_graph labels = (element, nH))."""
    from networkx.algorithms.isomorphism import GraphMatcher
    Gh = Q.fragmenter().heavy_graph(list(syms), np.asarray(xyz, float))[0]
    heavy = sorted(Gh.nodes)
    perms, capped = [], False
    for m in GraphMatcher(Gh, Gh, node_match=lambda a, b: a["lab"] == b["lab"]).isomorphisms_iter():
        perms.append(np.array([m[h] for h in heavy], int))
        if len(perms) >= MAX_AUTOMORPHISMS:
            capped = True
            break
    return np.array(heavy, int), perms, capped


def rmsd_sym(X, Y, heavy, perms):
    """heavy-atom RMSD minimised over rotations and reflections (O(3)) and the graph automorphisms."""
    if len(heavy) == 0:
        return 0.0
    P = X[heavy] - X[heavy].mean(0)
    nP = float((P * P).sum())
    best = math.inf
    for perm in perms:
        Qm = Y[perm] - Y[perm].mean(0)
        s = float(np.linalg.svd(P.T @ Qm, compute_uv=False).sum())
        v = max(nP + float((Qm * Qm).sum()) - 2 * s, 0.0) / len(heavy)
        if v < best:
            best = v
            if best < 1e-10:
                break
    return math.sqrt(best)


# ---------------------------------------------------------------- molecule context (D1 references, forms, stereo)
def mol_context(spec):
    a = spec["anchor"]
    syms_a, xyz_a = read_xyz(a["xyz"])
    G_a = Q.bond_graph(syms_a, xyz_a)
    heavy, perms, capped = automorphisms(syms_a, xyz_a)
    refs = []
    for r in spec["refs"]:
        s, x = read_xyz(r["xyz"])
        y = to_anchor(G_a, syms_a, s, x)
        if y is None:
            raise MolFail(f"D1 reference {r['job_id']}/frag{r['frag']} is not isomorphic to the anchor")
        refs.append(dict(r, geom=y))
    for r in refs:
        r["sig"] = Q.stereo3d_sig(syms_a, r["geom"], spec["smiles"], spec["charge"])
    refs.sort(key=lambda r: (r["e_eh"], r["job_id"], r["frag"]))    # refs[0] = the Q E_min source (audit tie rule)
    # V8 (spec V-3 'stereo != key -> discard'): the key's stereo = its single valid D1 signature, or the E_min source's
    valid = sorted({r["sig"] for r in refs if not r["sig"].startswith("error")})
    src_sig = refs[0]["sig"] if refs else "error:no_refs"
    if len(valid) == 1:
        allowed, mode = valid, "on"
    elif valid and not src_sig.startswith("error"):
        allowed, mode = [src_sig], f"on (E_min source signature; the D1 references carry {len(valid)})"
    else:
        raise MolFail(f"stereo of the key cannot be checked: {len(valid)} valid D1 signatures, E_min source "
                      f"{src_sig[:60]} (spec V-3; no V replacement)")
    forms = []
    for r in refs:
        hit = None
        for f in forms:
            if abs(r["e_eh"] - f["e_label_eh"]) * KCAL < DE_DUP or rmsd_sym(r["geom"], f["geom"], heavy, perms) < RMSD_DUP:
                hit = f
                break
        if hit is None:
            forms.append(dict(idx=len(forms), rep_job=r["job_id"], rep_frag=int(r["frag"]), rep_rxn=r.get("rxn_id"),
                              e_label_eh=float(r["e_eh"]), e_hf_eh=_f(r.get("e_hf_eh")), geom=r["geom"], sig=r["sig"],
                              members=[f"{r['job_id']}/frag{r['frag']}"]))
        else:
            hit["members"].append(f"{r['job_id']}/frag{r['frag']}")
    return dict(syms=syms_a, xyz=xyz_a, G=G_a, heavy=heavy, perms=perms, perms_capped=capped, refs=refs,
                allowed=allowed, stereo_mode=mode, n_valid_sigs=len(valid), forms=forms, smiles=spec["smiles"],
                charge=int(spec["charge"]))


def validate(ctx, syms, xyz):
    """(coordinates in anchor order, 'ok') or (None, reason): bond graph, then stereo (V8; always checked)."""
    y = to_anchor(ctx["G"], ctx["syms"], list(syms), xyz)
    if y is None:
        return None, "graph"
    s = Q.stereo3d_sig(ctx["syms"], y, ctx["smiles"], ctx["charge"])
    if s.startswith("error"):
        return None, "stereo_error"
    if s not in ctx["allowed"]:
        return None, "stereo"
    return y, "ok"


# ---------------------------------------------------------------- V-2 tools
def etkdg(smiles, n, seed, ncpu):
    from rdkit import Chem, rdBase
    from rdkit.Chem import AllChem
    info = dict(requested=n, seed=seed, method="ETKDGv3", prune_rms=-1.0, rdkit=rdBase.rdkitVersion)
    try:
        m = Chem.AddHs(Chem.MolFromSmiles(smiles))
        p = AllChem.ETKDGv3()
        p.randomSeed = int(seed)
        p.numThreads = int(ncpu)
        p.pruneRmsThresh = -1.0
        ids = list(AllChem.EmbedMultipleConfs(m, numConfs=int(n), params=p))
        syms = [a.GetSymbol() for a in m.GetAtoms()]
        X = [np.asarray(m.GetConformer(i).GetPositions(), float) for i in ids]
    except Exception as e:                                             # noqa: BLE001
        info.update(embedded=0, failed=True, error=f"{type(e).__name__}: {e}")
        return [], [], info
    info.update(embedded=len(X), failed=len(X) == 0)
    return syms, X, info


_RR_MOLS = []


def _rr_one(task):
    """one autodE RR conformer in a forked worker; RandomState() seeded with 20261023 + n (V7)."""
    n, f = task
    import numpy.random as npr
    from autode.conformers.conf_gen import get_simanl_conformer
    orig = npr.RandomState

    def seeded(*a, **k):
        return orig(*a, **k) if (a or k) else orig(L.SEED_V + n)
    npr.RandomState = seeded
    try:
        return get_simanl_conformer(_RR_MOLS[f], dist_consts=None, conf_n=n, save_xyz=False)
    finally:
        npr.RandomState = orig


def rr_conformers(ctx, mid, ncpu, root):
    global _RR_MOLS
    import autode as ade
    from autode.conformers.conformers import Conformers
    mr = mr_mod()
    _RR_MOLS = [mr.ade_mol(ade, f"{mid}_f{f['idx']}", ctx["syms"], f["geom"], ctx["charge"]) for f in ctx["forms"]]
    tasks = [(n, n % len(_RR_MOLS)) for n in range(N_RR)]
    cwd, rd = os.getcwd(), root / "rr"
    rd.mkdir(parents=True, exist_ok=True)
    os.chdir(rd)                          # as coley_port: autodE runs in a scratch folder (nothing is written: save_xyz)
    try:
        with ProcessPoolExecutor(max_workers=max(1, ncpu), mp_context=multiprocessing.get_context("fork")) as ex:
            res = list(ex.map(_rr_one, tasks))
    finally:
        os.chdir(cwd)
    confs = Conformers()
    for c in res:
        confs.append(c)
    n0 = len(confs)
    confs.prune_on_energy(e_tol=1e-10)
    n1 = len(confs)
    confs.prune_on_rmsd()
    out = []
    for c in confs:
        n = int(str(c.name).rsplit("_conf", 1)[1])
        out.append((n, np.asarray(c.coordinates, float)))
    info = dict(requested=N_RR, generated=n0, after_prune_energy=n1, after_prune_rmsd=len(out), n_forms=len(_RR_MOLS),
                seeds=f"RandomState({L.SEED_V} + n)", autode=ade.__version__, rmsd_threshold=float(ade.Config.rmsd_threshold))
    return sorted(out), info


def xtb_opt(task):
    """(tag, idx, syms, xyz, charge, work, xtb) -> result dict (V8). work = node-local folder (removed)."""
    tag, idx, syms, xyz, charge, work, xtb = task
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    (work / "in.xyz").write_text(_xyz_text(syms, xyz, f"{tag} {idx}"))
    cmd = [xtb, "in.xyz", "--opt", "tight", "--alpb", "water", "--gfn", "2", "--chrg", str(int(charge)), "--uhf", "0"]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    t = time.time()
    rc = None
    try:
        with open(work / "xtb.out", "w") as fo:
            rc = subprocess.run(cmd, cwd=work, stdout=fo, stderr=subprocess.STDOUT, env=env,
                                timeout=XTB_TIMEOUT_S).returncode
    except subprocess.TimeoutExpired:
        rc = "timeout"
    txt = (work / "xtb.out").read_text(errors="replace") if (work / "xtb.out").is_file() else ""
    res = dict(tag=tag, idx=int(idx), rc=rc, wall_s=time.time() - t, ok=False, e=None, syms=list(syms), xyz=None)
    try:
        conv = "GEOMETRY OPTIMIZATION CONVERGED" in txt
        es = RE_XTB_E.findall(txt)
        if rc == 0 and conv and es and (work / "xtbopt.xyz").is_file():
            s, x = read_xyz(work / "xtbopt.xyz")
            res.update(ok=True, e=float(es[-1]), syms=s, xyz=x, reason="ok")
        else:
            res["reason"] = "timeout" if rc == "timeout" else ("not_converged" if rc == 0 or "FAILED TO CONVERGE" in txt
                                                               else f"rc_{rc}")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return res


def dedupe_choose(forms_pool, cands, heavy, perms, n_keep=N_KEEP):
    """greedy, lowest xTB energy first: a candidate within DE_DUP kcal/mol or RMSD_DUP A of a kept structure (forms
    first) is a duplicate; the first n_keep new ones are chosen."""
    kept = [dict(kind="form", idx=f["idx"], xyz=f["xyz"], e=f["e"]) for f in forms_pool]
    chosen, n_ex, n_dup_form, n_dup_search = [], 0, 0, 0
    for c in sorted(cands, key=lambda c: (c["e"], c["tag"], c["idx"])):
        if len(chosen) >= n_keep:
            break
        n_ex += 1
        dup = None
        for k in kept:
            if (k["e"] is not None and abs(c["e"] - k["e"]) * KCAL < DE_DUP) \
                    or rmsd_sym(c["xyz"], k["xyz"], heavy, perms) < RMSD_DUP:
                dup = k
                break
        if dup is not None:
            c["dup_of"] = f"{dup['kind']}{dup['idx']}"
            if dup["kind"] == "form":
                n_dup_form += 1
            else:
                n_dup_search += 1
            continue
        c["dup_of"] = ""
        kept.append(dict(kind="c", idx=len(chosen), xyz=c["xyz"], e=c["e"]))
        chosen.append(c)
    return chosen, dict(n_examined=n_ex, n_dup_of_form=n_dup_form, n_dup_of_search=n_dup_search, n_chosen=len(chosen))


# ---------------------------------------------------------------- V-2 stage
def conf_stage(spec, spec_sha, ctx, root, ncpu, rs):
    mid = spec["mid"]
    d = mdir(mid)
    old = _load_json(d / "conf.json")
    if old is not None:
        return old
    t0 = time.time()
    rec = dict(mid=mid, key=spec["key"], spec_sha=spec_sha, code_version=CODE_VERSION, host=socket.gethostname(),
               slurm_job=os.environ.get("SLURM_JOB_ID"), ncpu=ncpu, started=_now(), stereo_check=ctx["stereo_mode"],
               n_valid_d1_sigs=ctx["n_valid_sigs"], allowed_sigs=ctx["allowed"],
               automorphisms=len(ctx["perms"]), automorphisms_capped=ctx["perms_capped"], n_refs=len(ctx["refs"]),
               n_ref_sig_errors=sum(r["sig"].startswith("error") for r in ctx["refs"]))
    syms_e, X_e, rec["etkdg"] = etkdg(spec["smiles"], N_ETKDG, L.SEED_V, ncpu)
    t1 = time.time()
    rec["etkdg"]["wall_s"] = t1 - t0
    rr, rec["rr"] = rr_conformers(ctx, mid, ncpu, root)
    t2 = time.time()
    rec["rr"]["wall_s"] = t2 - t1
    xr = root / "xtb"
    tasks = [("form", f["idx"], ctx["syms"], f["geom"], ctx["charge"], xr / f"form_{f['idx']}", rs["xtb"])
             for f in ctx["forms"]]
    tasks += [("etkdg", i, syms_e, x, ctx["charge"], xr / f"etkdg_{i}", rs["xtb"]) for i, x in enumerate(X_e)]
    tasks += [("rr", n, ctx["syms"], x, ctx["charge"], xr / f"rr_{n}", rs["xtb"]) for n, x in rr]
    with ThreadPoolExecutor(max_workers=max(1, ncpu)) as ex:
        res = list(ex.map(xtb_opt, tasks))
    t3 = time.time()
    counts = dict(n_run=len(res), n_xtb_ok=0, graph=0, stereo=0, stereo_error=0, valid=0, xtb_failed=0)
    forms_pool, cands = [], []
    fx = {f["idx"]: f for f in ctx["forms"]}
    form_xtb = []
    for r in res:
        if r["ok"]:
            counts["n_xtb_ok"] += 1
            y, why = validate(ctx, r["syms"], r["xyz"])
        else:
            counts["xtb_failed"] += 1
            y, why = None, f"xtb_{r['reason']}"
        if why in ("graph", "stereo", "stereo_error"):
            counts[why] += 1
        r["status"] = why
        if r["tag"] == "form":
            f = fx[r["idx"]]
            ok = y is not None
            forms_pool.append(dict(idx=f["idx"], xyz=y if ok else f["geom"], e=r["e"] if ok else None))
            form_xtb.append(dict(idx=f["idx"], status=why, e_xtb_eh=r["e"] if ok else None))
            continue
        if y is not None:
            counts["valid"] += 1
            cands.append(dict(tag=r["tag"], idx=r["idx"], e=r["e"], xyz=y))
    chosen, dstat = dedupe_choose(forms_pool, cands, ctx["heavy"], ctx["perms"])
    t4 = time.time()
    # files
    for f in ctx["forms"]:
        write_xyz(d / "forms" / f"f{f['idx']}.xyz", ctx["syms"], f["geom"],
                  f"{mid} D1 form {f['idx']} {f['rep_job']}/frag{f['rep_frag']} E_label={f['e_label_eh']:.10f}")
    blocks = [_xyz_text(ctx["syms"], c["xyz"], f"{c['tag']} {c['idx']} E_xtb={c['e']:.10f} dup_of={c.get('dup_of', 'n/a')}")
              for c in sorted(cands, key=lambda c: (c["e"], c["tag"], c["idx"]))]
    L.write_atomic(d / "conf_all.xyz.gz", lambda t: t.write_bytes(gzip.compress("".join(blocks).encode(), 6)))
    ch = []
    e_form_min = min([p["e"] for p in forms_pool if p["e"] is not None], default=None)
    for n, c in enumerate(chosen):
        p = d / "chosen" / f"c{n}.xyz"
        write_xyz(p, ctx["syms"], c["xyz"], f"{mid} c{n} {c['tag']} {c['idx']} E_xtb={c['e']:.10f}")
        ch.append(dict(cid=f"c{n}", tag=c["tag"], idx=c["idx"], e_xtb_eh=c["e"], xyz=str(p),
                       de_xtb_vs_lowest_form_kcal=None if e_form_min is None else (c["e"] - e_form_min) * KCAL))
    statuses = {}
    for r in res:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    rec.update(forms=[dict(idx=f["idx"], rep_job=f["rep_job"], rep_frag=f["rep_frag"], rep_rxn=f["rep_rxn"],
                           e_label_eh=f["e_label_eh"], e_hf_eh=f["e_hf_eh"], sig=f["sig"], n_members=len(f["members"]),
                           members=f["members"], **{k: v for k, v in x.items() if k != "idx"})
                      for f, x in zip(ctx["forms"], sorted(form_xtb, key=lambda z: z["idx"]))],
               n_forms=len(ctx["forms"]), xtb=dict(counts, statuses=statuses, wall_s=t3 - t2,
                                                   busy_s=float(sum(r["wall_s"] for r in res))),
               dedupe=dict(dstat, wall_s=t4 - t3, rule=f"|dE_xtb| < {DE_DUP} kcal/mol or heavy RMSD < {RMSD_DUP} A "
                                                        "(mirror + symmetry); forms first"),
               chosen=ch, wall_s=time.time() - t0, finished=_now())
    L.write_json(d / "conf.json", rec)
    print(f"  {mid}: conf {len(ctx['forms'])} forms, ETKDG {rec['etkdg'].get('embedded')}, RR {rec['rr']['after_prune_rmsd']}"
          f", xTB valid {counts['valid']}, chosen {len(ch)} ({rec['wall_s']:.0f} s)", flush=True)
    return rec


# ---------------------------------------------------------------- ORCA running (node-local work folders)
def orca_run(work, name, text, orca):
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    (work / f"{name}.inp").write_text(text)
    t = time.time()
    with open(work / f"{name}.out", "w") as fo, open(work / f"{name}.err", "w") as fe:
        p = subprocess.run([orca, f"{name}.inp"], cwd=work, stdout=fo, stderr=fe, env=os.environ.copy())
    out = (work / f"{name}.out").read_text(errors="replace")
    return p.returncode, time.time() - t, out, (p.returncode < 0 or not out.strip())


def orca_version_check(out, where):
    """V11: an ORCA output that prints its version must be 6.1.1 / GIT 487d211c (spec V-3; hf_common constants), else
    VStop. Returns the version (None when the output has no header: handled as an ORCA error / killed run)."""
    H = Q.hf_common()
    v, g = H.RE_VER.search(out), H.RE_GIT.search(out)
    if v and (v.group(1) != H.ORCA_VERSION or (g is not None and g.group(1) != H.ORCA_GIT)):
        raise VStop(f"{where}: ORCA {v.group(1)} (GIT {g.group(1) if g else '?'}) != {H.ORCA_VERSION} ({H.ORCA_GIT})")
    return v.group(1) if v else None


def copy_back(work, dest, name, plain=(), gz=()):
    L.write_atomic(dest / f"{name}.inp", (work / f"{name}.inp").read_bytes())
    write_gz(dest / f"{name}.out.gz", work / f"{name}.out")
    e = work / f"{name}.err"
    if e.is_file() and e.stat().st_size:
        write_gz(dest / f"{name}.err.gz", e)
    for ext in plain:
        p = work / f"{name}.{ext}"
        if p.is_file():
            L.write_atomic(dest / p.name, p.read_bytes())
    for ext in gz:
        p = work / f"{name}.{ext}"
        if p.is_file():
            write_gz(dest / f"{p.name}.gz", p)


def opt_step(cdir, root, name, syms, xyz, charge, rs):
    H, od = Q.hf_common(), od_mod()
    text = autode_orca_text("opt", syms, xyz, charge, 1, rs["n_cores"], rs["maxcore_mb"])
    work = root / "orca" / f"{cdir.parent.name}_{cdir.name}_{name}"
    try:
        rc, wall, out, killed = orca_run(work, name, text, rs["orca"])
        if killed:
            raise Infra(f"{cdir}/{name}: ORCA killed or no output (rc {rc})")
        copy_back(work, cdir, name, plain=("xyz",))
        orca_version_check(out, f"{cdir}/{name}")
        term, conv, has_xyz = H.terminated(out), od.opt_converged(out), (work / f"{name}.xyz").is_file()
        ok = term and conv and has_xyz
        st = "ok" if ok else ("opt_not_converged" if term and not conv else "orca_error")
        return dict(status=st, rc=rc, wall_s=wall, ncpu=rs["n_cores"], e_opt_eh=od.last_fspe(out),
                    n_cycles=len(RE_OPTCYC.findall(out)), orca_s=H.run_seconds(out), host=socket.gethostname(),
                    xyz=str(cdir / f"{name}.xyz") if ok else None)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def freq_step(cdir, root, name, syms, xyz, charge, rs):
    H, od = Q.hf_common(), od_mod()
    text = autode_orca_text("hess", syms, xyz, charge, 1, rs["n_cores"], rs["maxcore_mb"])
    work = root / "orca" / f"{cdir.parent.name}_{cdir.name}_{name}"
    try:
        rc, wall, out, killed = orca_run(work, name, text, rs["orca"])
        if killed:
            raise Infra(f"{cdir}/{name}: ORCA killed or no output (rc {rc})")
        copy_back(work, cdir, name, gz=("hess",))
        orca_version_check(out, f"{cdir}/{name}")
        hp = work / f"{name}.hess"
        if not (H.terminated(out) and hp.is_file()):
            return dict(status="orca_error", rc=rc, wall_s=wall, ncpu=rs["n_cores"], host=socket.gethostname())
        freqs, modes = od.parse_hess(hp)
        neg = sorted(float(v) for v in freqs if v < 0)
        rec = dict(status="ok", rc=rc, wall_s=wall, ncpu=rs["n_cores"], orca_s=H.run_seconds(out),
                   host=socket.gethostname(), n_neg=len(neg), neg_cm=neg,
                   n_below_tol=sum(v < rs["imag_tol_cm"] for v in neg), imag_tol_cm=rs["imag_tol_cm"],
                   lowest_real_cm=min([float(v) for v in freqs if v > 1e-6], default=None),
                   g_minus_eel_eh=_f((RE_GEL.findall(out) or [None])[-1]))
        if neg:
            mode = modes[:, int(np.argmin(freqs))].reshape(-1, 3)
            disp = np.asarray(xyz, float) + DISP_FACTOR * mode / np.linalg.norm(mode)
            write_xyz(cdir / "disp.xyz", syms, disp, f"{name} displaced along {neg[0]:.2f} cm-1 x {DISP_FACTOR} A")
            rec["max_atom_disp_A"] = float(np.linalg.norm(disp - np.asarray(xyz, float), axis=1).max())
        return rec
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ---------------------------------------------------------------- V11: label / HF SP inputs
def coord_only_check(template, new):
    """'' if new = template with only coordinate numbers changed, else the reason."""
    H = Q.hf_common()
    a, b = template.splitlines(keepends=True), new.splitlines(keepends=True)
    if len(a) != len(b):
        return "line count"
    for i, (x, y) in enumerate(zip(a, b)):
        mx, my = H.XYZ_LINE.match(x), H.XYZ_LINE.match(y)
        if mx:
            if not my or mx.group(1) != my.group(1) or mx.group(2) != my.group(2):
                return f"coordinate line {i} element / tag"
        elif x != y:
            return f"line {i} differs"
    pa, pb = H.parse_inp(template), H.parse_inp(new)
    for k in ("bang", "blocks", "charge", "mult"):
        if pa[k] != pb[k]:
            return f"{k} differs"
    if [t[0] for t in pa["atoms"]] != [t[0] for t in pb["atoms"]]:
        return "elements differ"
    return ""


def sp_texts(spec, syms, xyz):
    """(label SP text, HF SP text) for coordinates in anchor order; VStop when the inputs cannot be made by replacing
    coordinates only, or the two HF routes differ (spec 16 V)."""
    P = hf_prep()
    a = spec["anchor"]
    lab_t = Path(a["inp_label"]).read_text()
    hf_t = Path(a["inp_hf"]).read_text()
    if L.sha256(a["inp_label"]) != a["sha_label"] or L.sha256(a["inp_hf"]) != a["sha_hf"]:
        raise VStop(f"{spec['mid']}: template sha256 changed since targets")
    atoms = [(s, float(x), float(y), float(z)) for s, (x, y, z) in zip(syms, np.asarray(xyz, float))]
    lab = P.replace_coords(lab_t, atoms)
    why = coord_only_check(lab_t, lab)
    if why:
        raise VStop(f"{spec['mid']}: label SP input is not the template with new coordinates only ({why})")
    route_a = P.hf_text(lab, "frag")[0]
    route_b = P.replace_coords(hf_t, atoms)
    if route_a != route_b:
        raise VStop(f"{spec['mid']}: HF route A (hf_text of the label SP) != route B (D1-HF hf template, coordinates)")
    return lab, route_a


def sp_run(dest, kind, text, root, rs, retry):
    """one serial ORCA SP -> dest/<kind>.json (cached). retry = D1-HF rule (run.retry_text once on SCF non-convergence)."""
    jp = dest / f"{kind}.json"
    old = _load_json(jp)
    if old is not None:
        return old
    H, R = Q.hf_common(), hf_run()
    work = root / "sp" / f"{dest.parent.name}_{dest.name}_{kind}"
    txt, attempts, ok, conv, out, rc, wall_tot = text, [], False, False, "", None, 0.0
    try:
        for att in (1, 2):
            rc, wall, out, killed = orca_run(work, kind, txt, rs["orca"])
            wall_tot += wall
            if killed:
                raise Infra(f"{dest}/{kind}: ORCA killed or no output (rc {rc})")
            eps = H.RE_EPS.findall(out)
            ok = R.scf_good(out) and len(H.RE_FSPE.findall(out)) == 1 and bool(eps) and all(e == "78.3550" for e in eps)
            conv = bool(R.RE_NOCONV.search(out))
            attempts.append(dict(attempt=att, rc=rc, ok=ok, scf_not_converged=conv, wall_s=wall,
                                 change="none" if att == 1 else "SlowConv + %scf MaxIter 500 end (D1-HF run.retry_text)"))
            if ok or not (retry and att == 1 and conv):
                break
            txt = R.retry_text(txt, "frag1_rel")
        copy_back(work, dest, kind)
        orca_version_check(out, f"{dest}/{kind}")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    ver = H.RE_VER.search(out)
    rec = dict(kind=kind, status="ok" if ok else ("scf_not_converged" if conv else "orca_error"),
               e_eh=float(H.RE_FSPE.findall(out)[-1]) if ok else None, attempts=attempts, wall_s=wall_tot, ncpu=1,
               orca_version=ver.group(1) if ver else None, orca_s=H.run_seconds(out), host=socket.gethostname(),
               inp=str(dest / f"{kind}.inp"))
    L.write_json(jp, rec)
    return rec


# ---------------------------------------------------------------- V-3 stage
def dft_one(spec, ctx, c, root, rs):
    cdir = mdir(spec["mid"]) / c["cid"]
    rp = cdir / "result.json"
    R = _load_json(rp) or dict(cid=c["cid"], tag=c["tag"], idx=c["idx"], e_xtb_eh=c["e_xtb_eh"], start_xyz=c["xyz"],
                               status="pending", steps={})
    if R["status"] != "pending":
        return R
    S, syms, q = R["steps"], ctx["syms"], ctx["charge"]
    geom = read_xyz(c["xyz"])[1]
    for att in (1, 2):
        o, f = f"opt{att}", f"freq{att}"
        if att == 2:
            geom = read_xyz(cdir / "disp.xyz")[1]
        if o not in S:
            if not time_ok():
                raise TimeUp()
            S[o] = opt_step(cdir, root, o, syms, geom, q, rs)
            L.write_json(rp, R)
        if S[o]["status"] != "ok":
            R["status"] = f"fail:{S[o]['status']}"
            L.write_json(rp, R)
            return R
        geom = read_xyz(cdir / f"{o}.xyz")[1]
        if f not in S:
            if not time_ok():
                raise TimeUp()
            S[f] = freq_step(cdir, root, f, syms, geom, q, rs)
            L.write_json(rp, R)
        if S[f]["status"] != "ok":
            R["status"] = f"fail:{S[f]['status']}"
            L.write_json(rp, R)
            return R
        if S[f]["n_neg"] == 0:
            break
        if att == 2:
            # V10 (spec V-3): still imaginary after the one displacement + re-optimisation -> discarded (DFT failure)
            R["strict_imag"] = True
            R["imag_after_reopt_cm"] = S[f]["neg_cm"]
            R["imag_within_d1_tol"] = bool(S[f]["n_below_tol"] == 0)     # sensitivity only (-50 cm-1 rule)
            R["status"] = "fail:imag"
            L.write_json(rp, R)
            return R
    R["n_neg_first"] = S["freq1"]["n_neg"]
    R["displaced"] = "opt2" in S
    write_xyz(cdir / "final.xyz", syms, geom, f"{spec['mid']} {c['cid']} final DFT geometry (anchor atom order)")
    R["final_xyz"] = str(cdir / "final.xyz")
    y, why = validate(ctx, syms, geom)
    S["check"] = dict(status=why)
    R["status"] = "opt_ok" if why == "ok" else f"discard:{why}"
    L.write_json(rp, R)
    return R


def sp_stage(spec, ctx, conf, root, rs, ncpu):
    d = mdir(spec["mid"])
    tasks = []
    for c in conf["chosen"]:
        R = _load_json(d / c["cid"] / "result.json")
        if not R or R["status"] != "opt_ok":
            continue
        lab, hf = sp_texts(spec, ctx["syms"], read_xyz(R["final_xyz"])[1])
        tasks += [(d / c["cid"], "sp_label", lab, False), (d / c["cid"], "sp_hf", hf, True)]
    if spec.get("trial"):
        lab, hf = sp_texts(spec, ctx["syms"], ctx["xyz"])
        if lab != Path(spec["anchor"]["inp_label"]).read_text() or hf != Path(spec["anchor"]["inp_hf"]).read_text():
            raise VStop(f"{spec['mid']}: replace_coords with the anchor coordinates does not give the template back")
        tasks += [(d / "repro", "repro_label", lab, False), (d / "repro", "repro_hf", hf, True)]
    tasks = [t for t in tasks if not (t[0] / f"{t[1]}.json").is_file()]
    if not tasks:
        return
    if not time_ok():
        raise TimeUp()
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, min(ncpu, len(tasks)))) as ex:
        futs = [ex.submit(sp_run, dd, k, txt, root, rs, retry) for dd, k, txt, retry in tasks]
        errs = []
        for fu in futs:
            try:
                fu.result()
            except (Infra, VStop) as e:
                errs.append(e)
    pool = _load_json(d / "sp_pool.json", [])
    pool.append(dict(n=len(tasks), wall_s=time.time() - t0, ncpu=ncpu, host=socket.gethostname(), at=_now()))
    L.write_json(d / "sp_pool.json", pool)
    for e in errs:
        raise e


def finalize(spec, ctx, conf, ncpu):
    d = mdir(spec["mid"])
    out, n_started, n_fail = [], 0, 0
    for c in conf["chosen"]:
        R = _load_json(d / c["cid"] / "result.json") or dict(cid=c["cid"], status="pending", steps={})
        if R["status"] == "opt_ok":
            lab = _load_json(d / c["cid"] / "sp_label.json")
            hf = _load_json(d / c["cid"] / "sp_hf.json")
            if lab is None or hf is None:
                raise Infra(f"{spec['mid']}/{c['cid']}: SP results missing")
            R["e_label_eh"] = lab["e_eh"]
            R["e_hf_eh"] = hf["e_eh"]
            R["sp_label_status"], R["sp_hf_status"] = lab["status"], hf["status"]
            R["inp_label"], R["inp_hf"] = lab["inp"], hf["inp"]
            # V11: a valid conformer needs both SPs (its HF value is the dft rung H of the final reference)
            R["status"] = ("ok" if hf["status"] == "ok" and hf["e_eh"] is not None else "discard:sp_hf") \
                if lab["status"] == "ok" else "discard:sp_label"
            if R["status"] == "ok":
                g = read_xyz(R["final_xyz"])[1]
                hit = [(rmsd_sym(g, f["geom"], ctx["heavy"], ctx["perms"]), f) for f in ctx["forms"]]
                hit = [h for h in hit if h[0] < RMSD_DUP]
                if hit:
                    r, f = min(hit, key=lambda h: h[0])
                    R["same_as_form"] = dict(idx=f["idx"], rmsd_A=r, de_label_kcal=(R["e_label_eh"] - f["e_label_eh"]) * KCAL)
            L.write_json(d / c["cid"] / "result.json", R)
        if "opt1" in R.get("steps", {}):
            n_started += 1
        if R["status"].startswith("fail:"):
            n_fail += 1
        out.append(R)
    ok = [r for r in out if r["status"] == "ok"]
    best = min(ok, key=lambda r: (r["e_label_eh"], r["cid"])) if ok else None
    e_min = float(spec["e_min_eh"])
    vb = None
    if best:
        vb = dict(cid=best["cid"], e_label_eh=best["e_label_eh"], e_hf_eh=best["e_hf_eh"], xyz=best["final_xyz"],
                  inp_label=best["inp_label"], inp_hf=best["inp_hf"], lowering_kcal=(e_min - best["e_label_eh"]) * KCAL)
    repro = None
    if spec.get("trial"):
        rl, rh = _load_json(d / "repro" / "repro_label.json"), _load_json(d / "repro" / "repro_hf.json")
        a = spec["anchor"]
        repro = dict(label=rl, hf=rh,
                     d_label_kcal=None if not rl or rl["e_eh"] is None else (rl["e_eh"] - a["e_eh"]) * KCAL,
                     d_hf_kcal=None if not rh or rh["e_eh"] is None or a.get("e_hf_eh") is None
                     else (rh["e_eh"] - a["e_hf_eh"]) * KCAL)
    # core hours: allocated = stage wall x allocated CPUs; busy = process wall x processes
    alloc = conf["wall_s"] * conf["ncpu"]
    busy = conf["xtb"]["busy_s"] + conf["rr"]["wall_s"] * conf["ncpu"] + conf["etkdg"]["wall_s"] * conf["ncpu"]
    by = dict(conf=conf["wall_s"] * conf["ncpu"], opt=0.0, freq=0.0, sp=0.0)
    for r in out:
        for k, s in r.get("steps", {}).items():
            if k.startswith(("opt", "freq")) and "wall_s" in s:
                v = s["wall_s"] * s.get("ncpu", 1)
                alloc += s["wall_s"] * ncpu
                busy += v
                by["opt" if k.startswith("opt") else "freq"] += s["wall_s"] * ncpu
        for k in ("sp_label", "sp_hf"):
            sj = _load_json(d / r["cid"] / f"{k}.json")
            if sj:
                busy += sj["wall_s"]
    for k in ("repro_label", "repro_hf"):
        sj = _load_json(d / "repro" / f"{k}.json")
        if sj:
            busy += sj["wall_s"]
    for p in _load_json(d / "sp_pool.json", []):
        alloc += p["wall_s"] * p["ncpu"]
        by["sp"] += p["wall_s"] * p["ncpu"]
    summ = dict(mid=spec["mid"], key=spec["key"], kind=spec["kind"], groups=spec["groups"], trial=bool(spec.get("trial")),
                status="ok", code_version=CODE_VERSION, n_atoms=spec["n_atoms"], n_rxn=spec["n_rxn"],
                e_min_eh=e_min, n_forms=conf["n_forms"], forms=conf["forms"], conf_counts=conf["xtb"],
                etkdg=conf["etkdg"], rr=conf["rr"], dedupe=conf["dedupe"], stereo_check=conf["stereo_check"],
                n_chosen=len(conf["chosen"]), n_dft_started=n_started, n_dft_fail=n_fail,
                n_ok=len(ok), statuses={s: sum(r["status"] == s for r in out) for s in sorted({r["status"] for r in out})},
                n_strict_imag=sum(bool(r.get("strict_imag")) for r in out),
                n_imag_fail_within_d1_tol=sum(bool(r.get("imag_within_d1_tol")) for r in out),
                conformers=out, v_best=vb, repro=repro,
                core_h=dict(alloc=alloc / 3600, busy=busy / 3600, by_stage={k: v / 3600 for k, v in by.items()}),
                finished=_now())
    L.write_json(d / "summary.json", summ)
    L.write_atomic(d / ".done", _now() + "\n")
    low = "-" if not vb else f"{vb['lowering_kcal']:+.3f}"
    print(f"  {spec['mid']}: done, {len(ok)}/{len(out)} ok, V best {low} kcal/mol below E_min, "
          f"{summ['core_h']['alloc']:.2f} core-h", flush=True)
    return summ


def fail_molecule(spec, why):
    d = mdir(spec["mid"])
    summ = dict(mid=spec["mid"], key=spec["key"], kind=spec["kind"], groups=spec["groups"], trial=bool(spec.get("trial")),
                status=f"fail:{why}", code_version=CODE_VERSION, n_atoms=spec["n_atoms"], n_rxn=spec["n_rxn"],
                e_min_eh=float(spec["e_min_eh"]), n_chosen=0, n_dft_started=0, n_dft_fail=0, n_ok=0, conformers=[],
                v_best=None, repro=None, core_h=dict(alloc=0.0, busy=0.0, by_stage={}), finished=_now())
    L.write_json(d / "summary.json", summ)
    L.write_atomic(d / ".done", _now() + "\n")


def run_molecule(mid, root, ncpu, rs, stages=("conf", "dft")):
    spec, sha = load_spec(mid)
    d = mdir(mid)
    old = _load_json(d / "conf.json")
    if old is not None and old.get("spec_sha") != sha:
        raise VStop(f"{mid}: spec.json changed after V-2 ran (conf.json spec_sha); remove {d} to redo it")
    try:
        ctx = mol_context(spec)
    except MolFail as e:
        fail_molecule(spec, str(e))
        return "fail"
    conf = conf_stage(spec, sha, ctx, root / mid, ncpu, rs) if "conf" in stages else _load_json(d / "conf.json")
    if "dft" not in stages:
        return "conf"
    if conf is None:
        raise RuntimeError(f"{mid}: conf.json missing (run conf first)")
    for c in conf["chosen"]:
        dft_one(spec, ctx, c, root / mid, rs)
    sp_stage(spec, ctx, conf, root / mid, rs, ncpu)
    finalize(spec, ctx, conf, ncpu)
    return "done"


# ---------------------------------------------------------------- lists and shards
def mol_table():
    return pd.read_csv(RES_TARGETS, dtype={"mid": str, "key": str}, keep_default_na=False)


def list_mids(which):
    T = mol_table()
    if which == "trial":
        T = T[T.trial.astype(int) == 1]
    elif which == "main":
        T = T[T.trial.astype(int) == 0]
    T = T.assign(_n=-T.n_atoms.astype(int)).sort_values(["_n", "mid"])
    return T.mid.tolist()


def shard_mids(which, nshards, shard):
    return list_mids(which)[shard::nshards]


# ---------------------------------------------------------------- check (V9 proof, versions)
def cmd_check(force=False):
    old = _load_json(CHECK)
    if not force and old and old.get("ok") and old.get("code_sha") == code_sha():
        print("check up to date", flush=True)
        return 0
    rep = dict(code_sha=code_sha(), code_version=CODE_VERSION, host=socket.gethostname(), started=_now(), sources=[])
    rs = run_settings()
    t = rs["ts_stage"]
    want = dict(engine="orca", functional="B3LYP", dispersion="D3BJ", opt_basis="def2-SVP", solvent="water",
                solvation="smd", hmethod_conformers=False)
    rep["config"] = dict(path=str(D1_CONFIG), sha256=L.sha256(D1_CONFIG), ts_stage={k: t.get(k) for k in want},
                         n_cores=rs["n_cores"], maxcore_mb=rs["maxcore_mb"], imag_tol_cm=rs["imag_tol_cm"])
    cfg_ok = all(t.get(k) == v for k, v in want.items()) and rs["n_cores"] == 8 and rs["maxcore_mb"] == 3000
    n_d1_ok, all_ok = 0, True
    for src in ORCA_TEXT_SOURCES:
        r = dict(name=src["name"], files={})
        texts = {}
        try:
            if "tar" in src:
                want_m = {v: k for k, v in src["members"].items()}
                with tarfile.open(src["tar"], "r:*") as T:
                    for m in T:
                        n = m.name[2:] if m.name.startswith("./") else m.name
                        if m.isfile() and n in want_m:
                            texts[want_m[n]] = T.extractfile(m).read().decode()
                            if len(texts) == len(want_m):
                                break
            else:
                texts = {k: (src["dir"] / v).read_text() for k, v in src["members"].items()}
        except (OSError, tarfile.TarError) as e:
            r["error"] = f"{type(e).__name__}: {e}"
        for kind, txt in texts.items():
            try:
                p = parse_autode_inp(txt)
                regen = autode_orca_text(p["kind"], p["syms"], p["xyz"], p["charge"], p["mult"], p["n_cores"],
                                         p["maxcore_mb"])
            except (ValueError, KeyError, IndexError, StopIteration) as e:
                r["files"][kind] = dict(kind_parsed=None, identical=False, resources_match=False,
                                        error=f"{type(e).__name__}: {e}")
                continue
            r["files"][kind] = dict(kind_parsed=p["kind"], n_atoms=len(p["syms"]), n_cores=p["n_cores"],
                                    maxcore_mb=p["maxcore_mb"], identical=regen == txt,
                                    resources_match=(p["n_cores"], p["maxcore_mb"]) == (rs["n_cores"], rs["maxcore_mb"]))
        r["present"] = len(texts) == len(src["members"])
        r["ok"] = r["present"] and all(f["identical"] and f["resources_match"] and f["kind_parsed"] == k
                                       for k, f in r["files"].items())
        if r["present"] and not r["ok"]:
            all_ok = False
        if src["d1_run"] and r["ok"]:
            n_d1_ok += 1
        rep["sources"].append(r)
    vers = {}
    try:
        import autode as ade
        vers["autode"] = ade.__version__
        import cconf_gen  # noqa: F401
        vers["cconf_gen"] = "ok"
    except Exception as e:                                             # noqa: BLE001
        vers["autode_error"] = f"{type(e).__name__}: {e}"
    try:
        from rdkit import rdBase
        vers["rdkit"] = rdBase.rdkitVersion
    except Exception as e:                                             # noqa: BLE001
        vers["rdkit_error"] = str(e)
    try:
        xv = subprocess.run([rs["xtb"], "--version"], capture_output=True, text=True, timeout=120).stdout
        m = re.search(r"xtb version (\S+)", xv)
        vers["xtb"] = m.group(1) if m else None
    except (OSError, subprocess.SubprocessError) as e:
        vers["xtb_error"] = str(e)
    vers["orca_bin"] = rs["orca"]
    vers["orca_executable"] = os.access(rs["orca"], os.X_OK)
    vers["orca_sha256"] = L.sha256(rs["orca"]) if vers["orca_executable"] else None
    vers["python"] = platform.python_version()
    vers["mpi"] = {k: os.environ.get(k) for k in ("MPI_ROOT", "OMPI_MCA_btl", "OMPI_MCA_pml", "OMPI_MCA_coll_hcoll_enable",
                                                  "OMPI_MCA_rmaps_base_oversubscribe")}
    mr, mpi_root = shutil.which("mpirun"), vers["mpi"]["MPI_ROOT"]
    vers["mpirun"] = mr
    vers["mpirun_under_MPI_ROOT"] = bool(mr and mpi_root and Path(Path(mpi_root).resolve()) in Path(mr).resolve().parents)
    vers["orca_version_rule"] = f"every ORCA output: Program Version {Q.hf_common().ORCA_VERSION} (GIT {Q.hf_common().ORCA_GIT})"
    rep["versions"] = vers
    ver_ok = vers.get("autode") == "1.4.5" and vers.get("cconf_gen") == "ok" and vers.get("xtb") == "6.7.1" \
        and vers["orca_executable"] and mpi_root is not None and vers["mpirun_under_MPI_ROOT"]
    rep.update(config_ok=cfg_ok, text_ok=all_ok and n_d1_ok >= 1, n_d1_run_sources_ok=n_d1_ok, versions_ok=ver_ok,
               finished=_now())
    rep["ok"] = bool(cfg_ok and rep["text_ok"] and ver_ok)
    write_shared(CHECK, json.dumps(rep, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(json.dumps({k: v for k, v in rep.items() if k != "sources"}, indent=1, default=str), flush=True)
    if not rep["ok"]:
        write_stop("check failed: autodE ORCA text / D1 config / versions (see $LAD/V/check.json)")
        return 3
    return 0


# ---------------------------------------------------------------- V-1 targets
def template_check(aj, k, charge, xyz_path):
    """(problems, info) of the anchor's D1-HF orig / hf frag{k}_rel inputs (V4, spec 16 V)."""
    H, P = Q.hf_common(), hf_prep()
    ti, th = H.INPUTS / aj / "orig" / f"frag{k}_rel.inp", H.INPUTS / aj / "hf" / f"frag{k}_rel.inp"
    if not (ti.is_file() and th.is_file()):
        return ["template missing"], {}
    txt, hft = ti.read_text(), th.read_text()
    syms, xyz = read_xyz(xyz_path)
    d = H.parse_inp(txt)
    atoms = [(s, float(x), float(y), float(z)) for s, (x, y, z) in zip(syms, xyz)]
    bad = []
    if (d["charge"], d["mult"]) != (int(charge), 1):
        bad.append(f"charge / mult {d['charge']} {d['mult']}")
    if [a[0] for a in d["atoms"]] != syms:
        bad.append("element order != reference xyz")
    elif float(np.abs(np.array([a[2:] for a in d["atoms"]], float) - xyz).max()) > Q.XYZ_TOL:
        bad.append("coordinates != reference xyz")
    if not H.smd_ok(d, "frag"):
        bad.append("no SMD water")
    if not bad:
        if P.replace_coords(txt, atoms) != txt:
            bad.append("replace_coords(template, reference) != template")
        if P.hf_text(txt, "frag")[0] != hft:
            bad.append("hf template != prep.hf_text(orig template)")
        if P.replace_coords(hft, atoms) != hft:
            bad.append("replace_coords(hf template, reference) != hf template")
    return bad, dict(inp_label=str(ti), inp_hf=str(th), sha_label=L.sha256(ti), sha_hf=L.sha256(th),
                     bang=" ".join(d["bang"]))


def q_ready(rep):
    """Phase Q passed and its outputs / inputs are unchanged -> '' else the reason."""
    if not Q.RES_REPORT.is_file():
        return "results/Q_report.json missing (Phase Q has not run)"
    qr = json.loads(Q.RES_REPORT.read_text())
    rep["q_report_sha256"] = L.sha256(Q.RES_REPORT)
    if not qr.get("all_pass"):
        return "Phase Q did not pass (Q_report.json all_pass false)"
    bad = [p for p, s in (qr.get("outputs") or {}).items() if not Path(p).is_file() or L.sha256(p) != s]
    if bad:
        return f"Phase Q outputs changed since Q_report.json: {bad[:5]}"
    if qr.get("fingerprint") != Q.inputs_fingerprint():
        return "Phase Q inputs or code changed since Q_report.json (fingerprint)"
    return ""


def cmd_targets():
    t0 = time.time()
    rep = dict(phase="V-1", code_version=CODE_VERSION, code_sha=code_sha(), started=_now(),
               git_head=L.git("rev-parse", "HEAD").strip(), decisions=DECISIONS, found=FOUND, stop=False, all_pass=False)
    why = q_ready(rep)
    if why:
        return stop_report(RES_TREP, rep, why, hard=False)
    K = Q.load_keys()
    g = K.groupby("key", sort=True)
    A = pd.DataFrame(dict(kind=g.kind.first(), smiles_key=g.smiles_key.first(), constrained=g.constrained.any(),
                          n_rxn=g.size(), max_dev_kcal=g.dev_kcal.max(), e_min_eh=g.e_min_eh.first(),
                          q_src_job=g.src_job.first(), q_src_frag=g.src_frag.first(), formula=g.formula.first(),
                          n_atoms=g.n_atoms.first(), charge=g.charge.first(),
                          skeletons=g.skeleton.agg(lambda s: ";".join(sorted(set(s)))),
                          dipoles=g.dipole.agg(lambda s: ";".join(sorted(set(s))))))
    if (g.charge.nunique() > 1).any() or (g.formula.nunique() > 1).any() or (g.e_min_eh.nunique() > 1).any():
        return stop_report(RES_TREP, rep, "Q_keys.parquet: charge / formula / e_min not unique within a key")
    skel = g.skeleton.agg(lambda s: set(s))
    cons = A.constrained
    A["in_a"] = (A.max_dev_kcal > DEV_A) & ~cons
    A["in_b"] = (A.kind == "dipolarophile") & skel.map(lambda s: bool(s & set(SKEL_B))) & ~cons
    pool = A[~cons & ~A.in_a & ~A.in_b]
    rng = np.random.default_rng(L.SEED_V)
    c_keys, c_order = [], {}
    for kind in ("dipolarophile", "dipole"):
        cand = sorted(pool.index[pool.kind == kind])
        pick = [cand[i] for i in rng.choice(len(cand), size=min(N_CTRL, len(cand)), replace=False)] if cand else []
        for n, k in enumerate(pick):
            c_order[k] = f"{kind}:{n}"
        c_keys += pick
    A["in_c"] = A.index.isin(c_keys)
    parents = {}
    for name, smi in TRIAL_PARENTS:
        k = f"dipolarophile:{Q.mirror_min(smi)}"
        if k not in A.index or not (A.in_a[k] or A.in_b[k]):
            return stop_report(RES_TREP, rep, f"V-5 trial: {name} ({k}) is not a V target")
        parents[k] = name
    cand = sorted(k for k in A.index[A.in_a] if k not in parents)
    rng2 = np.random.default_rng(L.SEED_V)
    pick = [cand[i] for i in rng2.choice(len(cand), size=min(N_TRIAL_RANDOM, len(cand)), replace=False)] if cand else []
    trial = list(parents) + pick
    V = A[A.in_a | A.in_b | A.in_c].copy()
    V["mid"] = [f"V{i + 1:04d}" for i in range(len(V))]
    V["trial"] = V.index.isin(trial).astype(int)
    V["trial_order"] = [trial.index(k) if k in trial else -1 for k in V.index]
    V["c_order"] = [c_order.get(k, "") for k in V.index]
    V["groups"] = [";".join(x for x, f in (("a", a), ("b", b), ("c", c)) if f) for a, b, c in zip(V.in_a, V.in_b, V.in_c)]
    # anchors, templates, specs
    H = Q.hf_common()
    probs, specs, anchors = {}, {}, {}
    for key, r in V.iterrows():
        R = K[K.key == key].sort_values(["e_eh", "job_id", "frag"], kind="mergesort")
        anc = None
        for x in R.itertuples(index=False):
            k = int(x.frag)
            if (H.INPUTS / x.job_id / "orig" / f"frag{k}_rel.inp").is_file() and \
                    (H.INPUTS / x.job_id / "hf" / f"frag{k}_rel.inp").is_file():
                anc = x
                break
        if anc is None:
            probs[key] = ["no reference of the key has both D1-HF inputs"]
            continue
        bad, info = template_check(anc.job_id, int(anc.frag), int(r["charge"]), anc.xyz)
        if bad:
            probs[key] = bad
            continue
        anchors[key] = anc.job_id
        first = R.iloc[0]
        specs[key] = dict(
            mid=r["mid"], key=key, kind=r["kind"], smiles=anc.smiles, smiles_key=r["smiles_key"], charge=int(r["charge"]),
            formula=r["formula"], n_atoms=int(r["n_atoms"]), n_rxn=int(r["n_rxn"]), groups=r["groups"].split(";"),
            trial=bool(r["trial"]), e_min_eh=float(r["e_min_eh"]), q_src_job=r["q_src_job"],
            q_src_frag=int(r["q_src_frag"]),
            anchor=dict(job_id=anc.job_id, frag=int(anc.frag), rxn_id=int(anc.rxn_id), e_eh=float(anc.e_eh),
                        e_hf_eh=_f(anc.e_hf_eh), xyz=anc.xyz, **info),
            anchor_is_q_src=bool(anc.job_id == first["job_id"] and int(anc.frag) == int(first["frag"])),
            refs=[dict(job_id=x.job_id, frag=int(x.frag), rxn_id=int(x.rxn_id), e_eh=float(x.e_eh), e_hf_eh=_f(x.e_hf_eh),
                       dev_kcal=float(x.dev_kcal), xyz=x.xyz) for x in R.itertuples(index=False)],
            code_version=CODE_VERSION)
    if probs:
        rep["template_problems"] = {k: v for k, v in list(probs.items())[:30]}
        return stop_report(RES_TREP, rep, f"label SP input cannot be made by replacing coordinates for {len(probs)} "
                                          f"molecules (spec 16 V), first {next(iter(probs.items()))}")
    # the anchor jobs' autodE keywords (V9)
    recs, rinfo = Q.d1_job_records(sorted(set(anchors.values())), names=("ts_result.json",))
    kw_bad, kw_missing = [], 0
    for j, d in recs.items():
        ts = d.get("ts_result.json")
        if not ts:
            kw_missing += 1
            continue
        kw = ts.get("keywords") or {}
        if kw.get("opt") != EXPECTED_KW["opt"] or kw.get("hess") != EXPECTED_KW["hess"] or ts.get("autode_version") != "1.4.5":
            kw_bad.append(j)
    rep["autode_keywords"] = dict(n_jobs=len(recs), n_missing=kw_missing, n_mismatch=len(kw_bad), mismatch=kw_bad[:20],
                                  expected=EXPECTED_KW, ledger=dict(bad_tars=rinfo.get("bad_tars"), n_ubai=rinfo.get("n_ubai")))
    if kw_bad or rinfo.get("bad_tars"):
        return stop_report(RES_TREP, rep, f"anchor jobs whose autodE opt / hess keywords differ from the D1 level: "
                                          f"{kw_bad[:5]} (bad ledger tars {rinfo.get('bad_tars')})")
    # write specs (unchanged text is not rewritten; a molecule whose V-2 already ran must not change) and the tables
    texts = {key: json.dumps(s, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
             for key, s in specs.items()}
    changed = [s["mid"] for key, s in specs.items() if (mdir(s["mid"]) / "conf.json").is_file()
               and (mdir(s["mid"]) / "spec.json").read_text() != texts[key]]
    if changed:
        rep["specs_changed_after_conf"] = changed
        return stop_report(RES_TREP, rep, f"spec.json would change for molecules whose V-2 already ran: {changed[:5]} "
                                          "(targets differ from the earlier run; nothing rewritten)")
    for key, s in specs.items():
        p = mdir(s["mid"]) / "spec.json"
        if not (p.is_file() and p.read_text() == texts[key]):
            L.write_atomic(p, texts[key])
    V["anchor_job"] = [specs[k]["anchor"]["job_id"] for k in V.index]
    V["anchor_frag"] = [specs[k]["anchor"]["frag"] for k in V.index]
    V["anchor_is_q_src"] = [specs[k]["anchor_is_q_src"] for k in V.index]
    V["smiles"] = [specs[k]["smiles"] for k in V.index]
    cols = ["mid", "key", "kind", "smiles_key", "smiles", "charge", "formula", "n_atoms", "n_rxn", "max_dev_kcal",
            "e_min_eh", "q_src_job", "q_src_frag", "anchor_job", "anchor_frag", "anchor_is_q_src", "in_a", "in_b", "in_c",
            "groups", "trial", "trial_order", "c_order", "skeletons", "dipoles"]
    out = V.reset_index().rename(columns={"index": "key"})[cols]
    L.write_atomic(RES_TARGETS, lambda t: out.to_csv(t, index=False))
    tr = out[out.trial == 1].sort_values("trial_order")[["mid", "key", "trial_order", "groups", "n_atoms"]].copy()
    tr["why"] = [parents.get(k, "random (a)") for k in tr.key]
    L.write_atomic(RES_TRIAL, lambda t: tr.to_csv(t, index=False))
    plan = _load_json(PLAN, {})
    plan["trial"] = dict(nshards=TRIAL_SHARDS, mids=list_mids("trial"))
    L.write_json(PLAN, plan)
    nk = A.groupby("kind").size().to_dict()
    rep.update(
        n_keys=int(len(A)), n_keys_by_kind={k: int(v) for k, v in nk.items()},
        n_constrained_keys=int(cons.sum()), n_constrained_excluded_from_a=int((cons & (A.max_dev_kcal > DEV_A)).sum()),
        n_a=int(A.in_a.sum()), n_b=int(A.in_b.sum()), n_a_and_b=int((A.in_a & A.in_b).sum()), n_c=int(A.in_c.sum()),
        n_c_pool={k: int((pool.kind == k).sum()) for k in ("dipolarophile", "dipole")}, n_targets=int(len(V)),
        targets_by_kind={k: int(v) for k, v in V.groupby("kind").size().items()},
        a_by_kind={k: int(v) for k, v in A[A.in_a].groupby("kind").size().items()},
        trial=[dict(mid=V.mid[k], key=k, why=parents.get(k, "random (a)")) for k in trial],
        n_anchor_not_q_src=int((~V.anchor_is_q_src).sum()), n_refs_total=int(V.n_rxn.sum()),
        outputs={str(p): L.sha256(p) for p in (RES_TARGETS, RES_TRIAL)},
        all_pass=True, stop=False, finished=_now(), wall_s=round(time.time() - t0, 1))
    L.write_json(RES_TREP, rep)
    print(json.dumps({k: rep[k] for k in ("n_keys", "n_a", "n_b", "n_c", "n_targets", "n_constrained_excluded_from_a")},
                     indent=1), flush=True)
    return 0


# ---------------------------------------------------------------- work
def cmd_work(which, nshards, shard, stages=("conf", "dft"), mids=None):
    if STOP_FILE.is_file():
        print(f"STOP file present: {STOP_FILE.read_text().strip()}", file=sys.stderr)
        return 3
    plan = _load_json(PLAN, {})
    if mids is None:
        want = (plan.get(which) or {}).get("nshards")
        if want is not None and int(want) != int(nshards):
            print(f"--nshards {nshards} != plan.json {which} nshards {want}", file=sys.stderr)
            return 2
        mids = shard_mids(which, nshards, shard)
    rs = run_settings()
    ncpu = n_cpus()
    if "dft" in stages and ncpu < rs["n_cores"]:
        print(f"{ncpu} CPUs < {rs['n_cores']} (the D1 autodE %pal): submit with --cpus-per-task={rs['n_cores']}",
              file=sys.stderr)
        return 2
    rc = cmd_check()
    if rc:
        return rc
    root = Path(os.environ.get("TMPDIR", "/tmp")) / f"ladV_{os.environ.get('SLURM_JOB_ID', os.getpid())}"
    root.mkdir(parents=True, exist_ok=True)
    print(f"[work] {which} shard {shard}/{nshards}: {len(mids)} molecules, {ncpu} CPUs, work {root}", flush=True)
    res = {}
    try:
        for mid in mids:
            if done(mid):
                res[mid] = "skip"
                continue
            if not time_ok():
                res[mid] = "not_started"
                break
            if not lock_molecule(mid):                         # V18: another live job works on it
                res[mid] = "locked"
                print(f"  {mid}: locked by another live job ({mdir(mid) / '.lock'}); skipped", flush=True)
                continue
            if done(mid):                                      # finished by the previous owner just now
                unlock_molecule(mid)
                res[mid] = "skip"
                continue
            try:
                res[mid] = run_molecule(mid, root, ncpu, rs, stages)
            except TimeUp:
                res[mid] = "timeup"
                break
            except Infra as e:
                res[mid] = f"infra: {e}"
                print(f"  {mid}: INFRA {e}", flush=True)
                break
            except VStop as e:
                write_stop(str(e))
                print(f"STOP: {e}", file=sys.stderr, flush=True)
                return 3
            except Exception as e:                                     # noqa: BLE001
                tb = traceback.format_exc()
                L.write_atomic(mdir(mid) / "error.txt", f"{_now()} {type(e).__name__}: {e}\n{tb}")
                res[mid] = f"error: {type(e).__name__}"
                print(f"  {mid}: ERROR {type(e).__name__}: {e}\n{tb}", flush=True)
            finally:
                shutil.rmtree(root / mid, ignore_errors=True)
                unlock_molecule(mid)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print(f"[work] {res}", flush=True)
    complete = all(v in ("skip", "done", "fail", "conf") for v in res.values()) and len(res) == len(mids)
    return 0 if complete else 2


# ---------------------------------------------------------------- V-5 decide
def summaries(mids):
    return {m: _load_json(mdir(m) / "summary.json") for m in mids}


def dft_counts(S):
    n_start = sum(s["n_dft_started"] for s in S.values() if s)
    n_fail = sum(s["n_dft_fail"] for s in S.values() if s)
    by = {}
    for s in S.values():
        for c in (s or {}).get("conformers", []):
            by[c["status"]] = by.get(c["status"], 0) + 1
    strict = sum(s.get("n_strict_imag", 0) for s in S.values() if s)
    tol = sum(s.get("n_imag_fail_within_d1_tol", 0) for s in S.values() if s)
    return dict(n_started=n_start, n_fail=n_fail, rate=n_fail / n_start if n_start else 0.0, statuses=by,
                rule="failure = ORCA error / Opt not converged / any imaginary frequency after the one re-optimisation "
                     "(spec V-3)",
                n_imag_after_reopt=strict, n_imag_fail_within_d1_tol=tol,
                rate_if_d1_tol=(n_fail - tol) / n_start if n_start else 0.0,
                note_d1_tol="sensitivity only: the failure rate if modes above extra_imag_tol_cm (-50 cm-1) were kept")


def cmd_decide():
    rep = dict(phase="V-5", code_version=CODE_VERSION, started=_now(), stop=False)
    T = mol_table()
    tm = list_mids("trial")
    if not all(done(m) for m in tm):
        rep.update(ready=False, missing=[m for m in tm if not done(m)])
        L.write_json(RES_DECIDE, rep)
        return rep
    S = summaries(tm)
    alloc = {m: S[m]["core_h"]["alloc"] for m in tm}
    w = {m: float(n) ** 3 for m, n in zip(T.mid, T.n_atoms)}
    n_all = len(T)
    est_mean = float(np.mean(list(alloc.values()))) * n_all
    est_size = sum(w.values()) * sum(alloc.values()) / max(sum(w[m] for m in tm), 1e-12)
    est = max(est_mean, est_size)
    dc = dft_counts(S)
    rp = []
    for m in tm:
        r = S[m].get("repro") or {}
        rp.append(dict(mid=m, status=S[m]["status"], d_label_kcal=r.get("d_label_kcal"), d_hf_kcal=r.get("d_hf_kcal"),
                       label_status=(r.get("label") or {}).get("status"), hf_status=(r.get("hf") or {}).get("status")))
    # molecules that failed deterministically (no conformer search) have no reproduction run; they are reported
    repro_bad = [x for x in rp if x["status"] == "ok" and (
        x["d_label_kcal"] is None or abs(x["d_label_kcal"]) > REPRO_TOL
        or (x["d_hf_kcal"] is not None and abs(x["d_hf_kcal"]) > REPRO_TOL))]
    rep["trial_failed_molecules"] = [x["mid"] for x in rp if x["status"] != "ok"]
    reasons = []
    if est > CORE_H_MAX:
        reasons.append(f"estimated total {est:.0f} core-h > {CORE_H_MAX:.0f} (spec 16 V)")
    if dc["rate"] > DFT_FAIL_MAX:
        reasons.append(f"trial DFT optimisation failures {dc['n_fail']}/{dc['n_started']} = {dc['rate']:.1%} > 10 % "
                       "(spec 16 V)")
    if repro_bad:
        reasons.append(f"trial anchor reproduction > {REPRO_TOL} kcal/mol or failed for {[x['mid'] for x in repro_bad]}")
    rep.update(ready=True, trial=tm, trial_core_h=alloc, trial_core_h_total=sum(alloc.values()),
               trial_busy_core_h=sum(S[m]["core_h"]["busy"] for m in tm), n_targets=n_all,
               estimate=dict(mean_based=est_mean, size_based_n_atoms3=est_size, used=est, limit=CORE_H_MAX,
                             remaining=est - sum(alloc.values())),
               dft=dc, repro=rp, repro_tol_kcal=REPRO_TOL, go=not reasons, stop=bool(reasons), stop_reasons=reasons,
               finished=_now())
    L.write_json(RES_DECIDE, rep)
    print(json.dumps({k: rep[k] for k in ("trial_core_h_total", "estimate", "go", "stop_reasons")}, indent=1,
                     default=str), flush=True)
    return rep


# ---------------------------------------------------------------- chain (launcher on a compute node)
def _queue_count():
    user = os.environ.get("USER") or getpass.getuser()
    p = subprocess.run(["squeue", "-u", user, "-h", "-r", "-o", "%i"], capture_output=True, text=True, timeout=120)
    if p.returncode != 0:
        return None
    return len([ln for ln in p.stdout.splitlines() if ln.strip()])


def _wait_room(need, max_wait_h=40.0):
    """wait on this compute node until the queue (task level) has room for `need` more tasks (MaxSubmit 20; this
    launcher itself is one of the counted tasks and ends right after submitting)."""
    t = time.time()
    while True:
        n = _queue_count()
        if n is not None and n - 1 + need <= QUEUE_CAP:
            return True
        if time.time() - t > max_wait_h * 3600:
            return False
        print(f"[chain] queue {n} tasks, need {need}: waiting 10 min", flush=True)
        time.sleep(600)


def _sbatch(args, ladenv, name, cpus, mem, array=None, dep=None):
    cmd = ["sbatch", "--parsable", "-J", name, "-p", "cpu1,cpu2", f"--cpus-per-task={cpus}", f"--mem={mem}",
           "--time=48:00:00", f"--export=ALL,LADENV={ladenv}", "--kill-on-invalid-dep=yes"]
    if array:
        cmd.append(f"--array={array}")
    if dep:
        cmd.append(f"--dependency={dep}")
    cmd += [LAD_SLURM, "refs_v.py", *[str(a) for a in args]]
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SBATCH_", "SRUN_"))
           and k not in ("D1LADDER_JOB_T0",)}
    p = subprocess.run(cmd, cwd=HERE, env=env, capture_output=True, text=True, timeout=300)
    jid = p.stdout.strip().split(";")[0]
    if p.returncode != 0 or not jid.isdigit():
        raise RuntimeError(f"sbatch failed (rc {p.returncode}): {' '.join(cmd)}\n{p.stdout}{p.stderr}")
    print(f"[chain] submitted {jid}: {' '.join(cmd)}", flush=True)
    return jid


def _other_ladder_tasks():
    """LADDER tasks of the other streams (G0 / F0 / Q job names lad_* / ladQ) in the queue, pending or running; None if
    squeue fails. V shares the 4-job LADDER cap with them (spec 15)."""
    user = os.environ.get("USER") or getpass.getuser()
    try:
        p = subprocess.run(["squeue", "-u", user, "-h", "-r", "-o", "%j"], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    if p.returncode != 0:
        return None
    return sum(1 for ln in p.stdout.splitlines() if ln.strip().startswith(LADDER_OTHER_PREFIX))


def _submit_round(stage, nshards, shards, rnd, max_run):
    if not _wait_room(len(shards) + 1):
        print("[chain] no queue room after 40 h", file=sys.stderr)
        return 2
    others = _other_ladder_tasks()
    free = LADDER_MAX - (2 if others is None else others)      # unknown -> assume the G0 / F0 stream holds 2
    k = max(1, min(int(max_run), len(shards), free))
    jid = _sbatch(["work", stage, "--nshards", nshards], "d1", f"ladV{stage[0]}", WORK_RES["cpus"], WORK_RES["mem"],
                  array=",".join(map(str, shards)) + f"%{k}")
    _sbatch(["chain", stage, "--round", rnd, "--max-run", max_run], "reactot", f"ladVc{stage[0]}", CHAIN_RES["cpus"],
            CHAIN_RES["mem"], dep=f"afterany:{jid}")
    log = _load_json(VDIR / "chain_log.json", [])
    log.append(dict(at=_now(), stage=stage, round=rnd, work_job=jid, shards=shards, nshards=nshards, max_run=k,
                    other_ladder_tasks=others))
    L.write_json(VDIR / "chain_log.json", log)
    return 0


def cmd_chain(stage, rnd, max_run):
    print(f"[chain] {stage} round {rnd} {_now()}", flush=True)
    if STOP_FILE.is_file():
        print(f"STOP file present: {STOP_FILE.read_text().strip()}", file=sys.stderr)
        return 3
    tr = _load_json(RES_TREP)
    if not tr or not tr.get("all_pass"):
        print("V_targets_report.json missing or not passed", file=sys.stderr)
        return 3
    plan = _load_json(PLAN, {})
    if stage not in plan:
        print(f"plan.json has no {stage} part", file=sys.stderr)
        return 3
    ns = int(plan[stage]["nshards"])
    mids = list_mids(stage)
    todo = sorted({i % ns for i, m in enumerate(mids) if not done(m)})
    if todo:
        # round r = the r-th work submission of this stage (0 = the first); at most the plan's max_rounds of them
        lim = int(plan[stage].get("max_rounds") or MAX_ROUNDS[stage])
        if rnd + 1 >= lim:
            left = [m for m in mids if not done(m)]
            err = [m for m in left if (mdir(m) / "error.txt").is_file()]
            msg = (f"V-{stage}: {len(left)} molecules unfinished after {rnd + 1} rounds (limit {lim}): {left[:20]}; "
                   f"with error.txt: {err[:20]} (resolve, remove {STOP_FILE}, then resubmit 'chain {stage} --round 0')")
            write_stop(msg)
            print(f"STOP: {msg}", file=sys.stderr, flush=True)
            return 3
        return _submit_round(stage, ns, todo, rnd + 1, max_run)
    if stage == "trial":
        d = cmd_decide()
        if d.get("stop"):
            write_stop("V-5: " + "; ".join(d["stop_reasons"]))
            return 3
        main = list_mids("main")
        if not main:
            return cmd_collect()
        log = _load_json(VDIR / "chain_log.json", []) or []
        if any(x.get("stage") == "main" for x in log):          # V7: never re-plan / re-submit a running main stage
            print("[chain] main already submitted (chain_log.json); nothing to do", flush=True)
            return 0
        if "main" in plan and plan["main"].get("mids") == main:
            k = int(plan["main"]["nshards"])                     # planned before, submission did not happen
        else:
            k = max(1, min(int(max_run), len(main)))
            rem = float((d.get("estimate") or {}).get("remaining") or 0.0)
            rounds = max(MAX_ROUNDS["main"], math.ceil(rem / (k * WORK_RES["cpus"] * ROUND_CORE_H)) + 2)
            plan["main"] = dict(nshards=k, max_run=k, max_rounds=int(rounds), mids=main, decided=_now(),
                                rule=f"max_rounds = max({MAX_ROUNDS['main']}, ceil(remaining {rem:.0f} core-h / "
                                     f"({k} x {WORK_RES['cpus']} x {ROUND_CORE_H:g} h)) + 2)")
            L.write_json(PLAN, plan)
        return _submit_round("main", k, list(range(k)), 0, max_run)
    return cmd_collect()


# ---------------------------------------------------------------- V-4 collect
def cmd_collect():
    t0 = time.time()
    rep = dict(phase="V-4", code_version=CODE_VERSION, code_sha=code_sha(), started=_now(),
               git_head=L.git("rev-parse", "HEAD").strip(), decisions=DECISIONS, found=FOUND, stop=False, all_pass=False,
               env=Q.env_info())
    if STOP_FILE.is_file():
        print(f"STOP file present: {STOP_FILE.read_text().strip()}", file=sys.stderr)
        return 3
    T = mol_table()
    missing = [m for m in T.mid if not done(m)]
    if missing:
        print(f"[collect] {len(missing)} molecules not done: {missing[:10]}", file=sys.stderr)
        return 2
    why = q_ready(rep)
    if why:
        return stop_report(RES_REPORT, rep, why)
    rep["targets_report_sha256"] = L.sha256(RES_TREP)
    rep["decide"] = _load_json(RES_DECIDE)
    S = summaries(T.mid.tolist())
    dc = dft_counts(S)
    rep["dft"] = dc
    rep["stereo_check"] = {k: int(v) for k, v in pd.Series([str((S[m] or {}).get("stereo_check") or (S[m] or {}).get(
        "status")) for m in T.mid]).value_counts().items()}
    # per molecule / conformer tables
    mrow, crow = [], []
    for r in T.itertuples(index=False):
        s = S[r.mid]
        vb = s.get("v_best") or {}
        mrow.append(dict(mid=r.mid, key=r.key, kind=r.kind, groups=r.groups, trial=int(r.trial), n_atoms=int(r.n_atoms),
                         n_rxn=int(r.n_rxn), max_dev_q_kcal=float(r.max_dev_kcal), status=s["status"],
                         n_forms=s.get("n_forms"), n_chosen=s["n_chosen"], n_dft_started=s["n_dft_started"],
                         n_dft_fail=s["n_dft_fail"], n_ok=s["n_ok"], e_min_q_eh=s["e_min_eh"],
                         v_best_cid=vb.get("cid"), v_best_e_eh=vb.get("e_label_eh"), v_best_hf_eh=vb.get("e_hf_eh"),
                         lowering_kcal=vb.get("lowering_kcal"), stereo_check=s.get("stereo_check"),
                         replaced=bool(vb and vb["lowering_kcal"] > THR
                                       and str(s.get("stereo_check") or "").startswith("on")),
                         core_h_alloc=s["core_h"]["alloc"], core_h_busy=s["core_h"]["busy"]))
        for c in s.get("conformers", []):
            st = c.get("steps", {})
            crow.append(dict(mid=r.mid, key=r.key, cid=c["cid"], tag=c.get("tag"), src_idx=c.get("idx"),
                             e_xtb_eh=c.get("e_xtb_eh"), status=c["status"], n_neg_first=c.get("n_neg_first"),
                             displaced=c.get("displaced"), imag_after_reopt_cm=json.dumps(c.get("imag_after_reopt_cm")),
                             imag_within_d1_tol=c.get("imag_within_d1_tol"),
                             e_opt_eh=(st.get("opt2") or st.get("opt1") or {}).get("e_opt_eh"),
                             e_label_eh=c.get("e_label_eh"), e_hf_eh=c.get("e_hf_eh"),
                             de_label_vs_emin_kcal=None if c.get("e_label_eh") is None
                             else (c["e_label_eh"] - s["e_min_eh"]) * KCAL,
                             same_as_form=(c.get("same_as_form") or {}).get("idx"),
                             de_vs_same_form_kcal=(c.get("same_as_form") or {}).get("de_label_kcal")))
    M, C = pd.DataFrame(mrow), pd.DataFrame(crow)
    L.write_atomic(RES_MOLS, lambda t: M.to_csv(t, index=False))
    L.write_atomic(RES_CONFS, lambda t: C.to_csv(t, index=False))
    if dc["rate"] > DFT_FAIL_MAX:
        return stop_report(RES_REPORT, rep, f"DFT optimisation failures {dc['n_fail']}/{dc['n_started']} = "
                                            f"{dc['rate']:.1%} > 10 % (spec 16 V)")
    # V-4 re-assembly
    rows = L.targets()
    K = Q.load_keys()
    Edft = {(r["job_id"], k): float(r[Q.REF_E[k]]) for r in rows for k in (1, 2)}
    A = Q.audit(K, Edft)
    Kx = K.set_index(["job_id", "frag"])
    Ax = A.set_index(["job_id", "frag"]).loc[Kx.index]
    va = dict(ok=bool((Ax.e_min_eh.to_numpy() == Kx.e_min_eh.to_numpy()).all()
                      and (Ax.src_job.to_numpy() == Kx.src_job.to_numpy()).all()
                      and (Ax.dev_kcal.to_numpy() == Kx.dev_kcal.to_numpy()).all() and len(A) == len(K)),
              note="Q audit recomputed from labels_d1.json == Q_keys.parquet")
    vbest = {}
    for r in M.itertuples(index=False):
        if r.replaced:
            s = S[r.mid]
            vbest[r.key] = dict(s["v_best"], mid=r.mid)
    A2 = A.copy()
    isv = A2.key.isin(list(vbest))
    A2.loc[isv, "e_min_eh"] = [vbest[k]["e_label_eh"] for k in A2.key[isv]]
    A2.loc[isv, "dev_kcal"] = (A2.e_eh[isv] - A2.e_min_eh[isv]) * KCAL
    ov = Q.choose_override(A2[~isv], K)
    for r in A2[isv].itertuples(index=False):
        if r.dev_kcal > THR:
            v = vbest[r.key]
            ov[(r.job_id, int(r.frag))] = dict(
                e_eh=float(v["e_label_eh"]), e_orig_eh=float(r.e_eh), dev_kcal=float(r.dev_kcal), src="V:label_min",
                src_job=None, src_rxn=None, src_frag=int(r.frag), key=r.key, ref_xyz=v["xyz"], ref_inp=v["inp_label"],
                v_mid=v["mid"], v_cid=v["cid"], e_hf_eh=_f(v.get("e_hf_eh")), ref_inp_hf=v["inp_hf"])
    new_rows, recs = Q.reassemble(rows, ov, K, "cr2")
    text, new_rt = Q.json_roundtrip(new_rows)
    qa = Q.gate_qa(rows)
    G = Q.gates(rows, new_rt, ov, recs, qa)
    G["V-a"] = va
    crp = Q.out_paths("cr")["json"]
    cr = {r["job_id"]: r for r in json.loads(crp.read_text())}
    keys = {(j, int(f)): k for j, f, k in zip(K.job_id, K.frag, K.key)}
    vc_bad, ve_bad = [], []
    for n in new_rt:
        j = n["job_id"]
        o = cr.get(j)
        if o is None:
            vc_bad.append(f"{j}: not in labels_d1_cr.json")
            continue
        kv = [k for k in (1, 2) if keys.get((j, k)) in vbest]
        for k in (1, 2):
            if k not in kv and n[Q.REF_E[k]] != o[Q.REF_E[k]]:
                vc_bad.append(f"{j}/frag{k}")
            if not n[Q.REF_E[k]] <= o[Q.REF_E[k]]:
                ve_bad.append(f"{j}/frag{k}: above the cr reference")
        if not kv:
            if any(json.dumps(n[c]) != json.dumps(o[c]) for c in Q.CHANGED_KEYS) or \
                    {a: b for a, b in n["cr"].items() if a != "tag"} != {a: b for a, b in o["cr"].items() if a != "tag"}:
                vc_bad.append(f"{j}: row differs from labels_d1_cr.json")
    for key, v in vbest.items():
        e_min = float(K.e_min_eh[K.key == key].iloc[0])
        if not (e_min - v["e_label_eh"]) * KCAL > THR:
            ve_bad.append(f"{key}: V reference not below E_min - {THR}")
        if _f(v.get("e_hf_eh")) is None:
            ve_bad.append(f"{key}: V reference without its HF/6-31G* energy (spec 4-1 dft rung H)")
    G["V-c"] = dict(ok=not vc_bad, n_fail=len(vc_bad), fail=vc_bad[:30],
                    note="fragments of non-V keys = labels_d1_cr.json; rows without a V key identical (cr dict but tag)")
    G["V-e"] = dict(ok=not ve_bad, n_fail=len(ve_bad), fail=ve_bad[:30])
    rep["gates"] = G
    if not Q.gates_ok(G):
        L.write_atomic(Q.out_paths("cr2")["rejected"], text)
        return stop_report(RES_REPORT, rep, "V-4 gate failure: " + ", ".join(k for k, v in G.items() if not v.get("ok")))
    T2 = Q.cr_table(rows, new_rt, K, A2, ov)
    for k in (1, 2):
        T2[f"src{k}_v"] = [f"{ov[(j, k)]['v_mid']}/{ov[(j, k)]['v_cid']}" if (j, k) in ov and "v_mid" in ov[(j, k)]
                           else None for j in T2.job_id]
    outputs = Q.write_outputs("cr2", text, T2)
    orig_by_job = {r["job_id"]: r for r in rows}
    new_by_job = {r["job_id"]: r for r in new_rt}
    prep, Fm, dd0, dd1 = Q.write_pairs("cr2", orig_by_job, new_by_job)
    rep["pairs"] = prep
    if not prep.get("ok"):
        return stop_report(RES_REPORT, rep, f"F_pairs_cr2.csv could not be proven ({prep.get('why') or 'see V_report'})")
    outputs[prep["output"]] = prep["output_sha256"]
    # pairs vs F_pairs_cr.csv
    vjobs = {j for (j, k), o in ov.items() if "v_mid" in o}
    Fc = pd.read_csv(Q.out_paths("cr")["pairs"], dtype=str, keep_default_na=False)
    F2 = pd.read_csv(Q.out_paths("cr2")["pairs"], dtype=str, keep_default_na=False)
    touch = (Fc.a_job.isin(vjobs) | Fc.b_job.isin(vjobs)).to_numpy()
    same_rest = bool((Fc[~touch].to_numpy() == F2[~touch].to_numpy()).all()) if len(Fc) == len(F2) else False
    pc = {}
    for t in Q.PAIR_T:
        a, b = pd.to_numeric(Fc[f"dd_{t}"]), pd.to_numeric(F2[f"dd_{t}"])
        d = (b - a).to_numpy()
        ch = np.abs(d) > Q.TOL
        pc[t] = dict(n_changed_vs_cr=int(ch.sum()), in_P_main=int((ch & (Fc.in_P_main == "True").to_numpy()).sum()),
                     abs_change=_q(np.abs(d[ch])), by_skeleton=Q._vc(Fc.skeleton[ch]), by_dipole=Q._vc(Fc.dipole[ch]),
                     by_pos=Q._vc(Fc.pos[ch]))
    rep["pairs_vs_cr"] = dict(n_pairs_touching_v=int(touch.sum()), untouched_pairs_identical=same_rest, dd=pc)
    if not same_rest:
        return stop_report(RES_REPORT, rep, "F_pairs_cr2.csv differs from F_pairs_cr.csv outside the V-changed reactions")
    # final references with HF energies (spec 4-1 dft rung H)
    H = Q.hf_common()
    ehf = {(j, int(f)): _f(e) for j, f, e in zip(K.job_id, K.frag, K.e_hf_eh)}
    kxyz = {(j, int(f)): p for j, f, p in zip(K.job_id, K.frag, K.xyz)}
    rr = []
    for n in new_rt:
        j = n["job_id"]
        for k in (1, 2):
            o = ov.get((j, k))
            if o is None:
                src, sj, sf, e_hf, xyz = "own", j, k, ehf.get((j, k)), kxyz[(j, k)]
                il, ih = str(H.INPUTS / j / "orig" / f"frag{k}_rel.inp"), str(H.INPUTS / j / "hf" / f"frag{k}_rel.inp")
            elif "v_mid" in o:
                src, sj, sf, e_hf, xyz, il, ih = "V", None, k, o["e_hf_eh"], o["ref_xyz"], o["ref_inp"], o["ref_inp_hf"]
            else:
                src, sj, sf, e_hf, xyz = "Q", o["src_job"], int(o["src_frag"]), ehf.get((o["src_job"], int(o["src_frag"]))), o["ref_xyz"]
                il, ih = o["ref_inp"], str(H.INPUTS / o["src_job"] / "hf" / f"frag{int(o['src_frag'])}_rel.inp")
            rr.append(dict(job_id=j, rxn_id=int(n["rxn_id"]), frag=k, key=keys.get((j, k)), src=src, src_job=sj, src_frag=sf,
                           v_mid=(o or {}).get("v_mid"), v_cid=(o or {}).get("v_cid"), e_label_eh=float(n[Q.REF_E[k]]),
                           e_label_orig_eh=float(orig_by_job[j][Q.REF_E[k]]), e_hf_eh=e_hf, xyz=xyz, inp_label=il,
                           inp_hf=ih))
    RR = pd.DataFrame(rr)
    L.write_atomic(REFS_CR2, lambda t: RR.to_parquet(t, index=False))
    outputs[str(REFS_CR2)] = L.sha256(REFS_CR2)
    # report numbers
    groups = {}
    for gname in ("a", "b", "c"):
        sub = M[M["groups"].str.split(";").map(lambda s, g=gname: g in s)]
        low = sub.lowering_kcal.dropna()
        groups[gname] = dict(n=int(len(sub)), n_with_v_conformer=int(sub.v_best_e_eh.notna().sum()),
                             n_lower_gt_0_05=int((low > THR).sum()), frac_lower=float((low > THR).sum() / len(sub))
                             if len(sub) else None,
                             n_lower_gt={f"{x:g}": int((low > x).sum()) for x in (0.05, 0.5, 1.0, 2.0)},
                             lowering_of_lower=_q(low[low > THR]), v_minus_emin_all=_q(-low))
    cot = cyclooctyne_table(T, S)
    rep.update(
        n_targets=int(len(T)), groups=groups, control=groups["c"],
        n_keys_replaced=len(vbest), n_fragments_replaced_by_v=sum("v_mid" in o for o in ov.values()),
        n_rows_changed_vs_cr=len({j for (j, k), o in ov.items() if "v_mid" in o}),
        n_rows_changed_vs_orig=int((T2.changed != "").sum()),
        cyclooctyne_csv=str(RES_COT), n_cyclooctyne_rows=len(cot),
        core_h=dict(alloc=float(M.core_h_alloc.sum()), busy=float(M.core_h_busy.sum()),
                    by_group={g: float(M.core_h_alloc[M["groups"].str.split(";").map(lambda s, g=g: g in s)].sum())
                              for g in ("a", "b", "c")}),
        outputs=dict(outputs, **{str(p): L.sha256(p) for p in (RES_MOLS, RES_CONFS, RES_COT)}),
        all_pass=True, finished=_now(), wall_s=round(time.time() - t0, 1))
    L.write_json(RES_REPORT, rep)
    L.write_atomic(RES_SUMMARY, summary_md(rep, M, cot))
    print(f"[collect] V-4 pass: {len(vbest)} keys lowered, {rep['n_rows_changed_vs_cr']} reactions changed vs cr", flush=True)
    return 0


def cyclooctyne_table(T, S):
    rows = []
    for name, smi in TRIAL_PARENTS:
        key = f"dipolarophile:{Q.mirror_min(smi)}"
        sub = T[T.key == key]
        if not len(sub):
            continue
        s = S[sub.mid.iloc[0]]
        e0 = s["e_min_eh"]
        hf0 = min([f["e_hf_eh"] for f in s.get("forms", []) if f.get("e_hf_eh") is not None], default=None)
        for f in s.get("forms", []):
            rows.append(dict(molecule=name, mid=s["mid"], source="D1 form", id=f"f{f['idx']}", n_d1_refs=f["n_members"],
                             rep=f"{f['rep_job']}/frag{f['rep_frag']}", e_xtb_eh=f.get("e_xtb_eh"),
                             e_label_eh=f["e_label_eh"], de_label_kcal=(f["e_label_eh"] - e0) * KCAL,
                             e_hf_eh=f.get("e_hf_eh"), de_hf_kcal=None if f.get("e_hf_eh") is None or hf0 is None
                             else (f["e_hf_eh"] - hf0) * KCAL, status="kept (D1 value)"))
        for c in s.get("conformers", []):
            rows.append(dict(molecule=name, mid=s["mid"], source=f"V {c.get('tag')}", id=c["cid"], n_d1_refs=None,
                             rep=(c.get("same_as_form") and f"= f{c['same_as_form']['idx']}") or "",
                             e_xtb_eh=c.get("e_xtb_eh"), e_label_eh=c.get("e_label_eh"),
                             de_label_kcal=None if c.get("e_label_eh") is None else (c["e_label_eh"] - e0) * KCAL,
                             e_hf_eh=c.get("e_hf_eh"), de_hf_kcal=None if c.get("e_hf_eh") is None or hf0 is None
                             else (c["e_hf_eh"] - hf0) * KCAL, status=c["status"]))
    D = pd.DataFrame(rows)
    L.write_atomic(RES_COT, lambda t: D.to_csv(t, index=False))
    return D


def summary_md(rep, M, cot):
    tr = _load_json(RES_TREP) or {}
    de = rep.get("decide") or {}
    g = rep["groups"]

    def fmt(x, d=3):
        return "-" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{d}f}"
    L1 = [
        "# D1LADDER Phase V — 기준 구조 확인 계산 요약", "",
        f"- 코드 {CODE_VERSION}, git {rep.get('git_head', '')[:10]}, 작성 {rep.get('finished')}",
        f"- 판정: **{'통과' if rep.get('all_pass') else 'STOP'}** — 정답 파일 `labels_d1_cr2.json`, `F_pairs_cr2.csv`", "",
        "## V-1 대상",
        f"- 분자 키 {tr.get('n_keys')}개 중 대상 {rep['n_targets']}개: (a) dev > 0.5 kcal/mol {tr.get('n_a')}개, "
        f"(b) cyclooctyne 계열(D, Dp) {tr.get('n_b')}개 (겹침 {tr.get('n_a_and_b')}개), (c) 대조군 {tr.get('n_c')}개",
        f"- 제외: constrained dipole 키 {tr.get('n_constrained_keys')}개 (그중 dev > 0.5인 것 "
        f"{tr.get('n_constrained_excluded_from_a')}개)", "",
        "## V-5 시험 묶음과 비용",
        f"- 시험 10개 분자 코어시간 {fmt(de.get('trial_core_h_total'), 1)} (할당 기준), 전체 예상 "
        f"{fmt((de.get('estimate') or {}).get('used'), 0)} 코어시간 (한도 5,000) → "
        f"{'자동 진행' if de.get('go') else 'STOP'}",
        f"- 시험 DFT 최적화 실패율 {fmt(((de.get('dft') or {}).get('rate') or 0) * 100, 1)} %",
        f"- 시험 기준 구조 재현 (라벨·HF 단일점 그대로 다시 계산): 최대 |ΔE| "
        f"{fmt(max([abs(x['d_label_kcal']) for x in de.get('repro', []) if x.get('d_label_kcal') is not None], default=None), 4)}"
        f" / {fmt(max([abs(x['d_hf_kcal']) for x in de.get('repro', []) if x.get('d_hf_kcal') is not None], default=None), 4)}"
        " kcal/mol", "",
        "## 더 낮은 형태를 찾은 분자 (라벨 수준 단일점, E_min − E_V > 0.05 kcal/mol)",
        "| 대상 | 분자 수 | V 형태 있음 | 더 낮음 | > 0.5 | > 1 | > 2 | 낮아진 양 중앙값 | 최대 |", "|---|---|---|---|---|---|---|---|---|"]
    for k, name in (("a", "(a) dev > 0.5"), ("b", "(b) cyclooctyne 계열"), ("c", "(c) 대조군")):
        x = g[k]
        lo = x["lowering_of_lower"] or {}
        L1.append(f"| {name} | {x['n']} | {x['n_with_v_conformer']} | {x['n_lower_gt_0_05']} | {x['n_lower_gt']['0.5']} | "
                  f"{x['n_lower_gt']['1']} | {x['n_lower_gt']['2']} | {fmt(lo.get('p50'))} | {fmt(lo.get('max'))} |")
    c = g["c"]
    L1 += ["", f"- 대조군에서 더 낮은 형태를 찾은 비율: {c['n_lower_gt_0_05']}/{c['n']}"
               f" = {fmt((c['frac_lower'] or 0) * 100, 1)} %", "",
           "## cyclooctyne 부모 분자의 형태별 에너지 (ΔE = E − Q의 E_min, kcal/mol)",
           "| 분자 | 출처 | id | D1 반응 수 | ΔE 라벨 | ΔE HF | 상태 |", "|---|---|---|---|---|---|---|"]
    for r in cot.itertuples(index=False):
        L1.append(f"| {r.molecule} | {r.source} | {r.id} {r.rep} | {'' if r.n_d1_refs is None or r.n_d1_refs != r.n_d1_refs else int(r.n_d1_refs)} | "
                  f"{fmt(r.de_label_kcal)} | {fmt(r.de_hf_kcal)} | {r.status} |")
    d = rep["dft"]
    L1 += ["", "## DFT (V-3)",
           f"- 최적화 시작 {d['n_started']}개, 실패 {d['n_fail']}개 ({fmt(d['rate'] * 100, 1)} %; 명령서 V-3대로 재최적화 뒤 "
           f"허수 진동수가 하나라도 남으면 실패), 참고: D1 TS 단계 허용치(−50 cm⁻¹)를 썼다면 "
           f"{fmt(d['rate_if_d1_tol'] * 100, 1)} % (그중 허용치 안 {d['n_imag_fail_within_d1_tol']}개)",
           "- 상태별: " + ", ".join(f"{k} {v}" for k, v in sorted(d["statuses"].items())), "",
           "## 재조립 (V-4)",
           f"- 기준이 바뀐 분자 키 {rep['n_keys_replaced']}개, V 기준을 쓰게 된 조각 {rep['n_fragments_replaced_by_v']}개, "
           f"cr 대비 바뀐 반응 {rep['n_rows_changed_vs_cr']}개, 원래 라벨 대비 바뀐 반응 {rep['n_rows_changed_vs_orig']}개",
           "- gate: " + ", ".join(f"{k} {'통과' if v.get('ok') else '실패'}" for k, v in rep["gates"].items())]
    for t, x in rep["pairs_vs_cr"]["dd"].items():
        L1.append(f"- dd_{t}: cr 대비 바뀐 짝 {x['n_changed_vs_cr']}개 (P_main {x['in_P_main']}개), "
                  f"바뀐 양 중앙값 {fmt((x['abs_change'] or {}).get('p50'))} kcal/mol")
    ch = rep["core_h"]
    L1 += ["", "## 코어시간", f"- 할당 기준 {fmt(ch['alloc'], 1)}, 실제 사용 {fmt(ch['busy'], 1)} 코어시간 "
                             f"(그룹별 할당: " + ", ".join(f"{k} {fmt(v, 1)}" for k, v in ch["by_group"].items()) + ")",
           "", "## 결정 (사용자 지시: 모든 갈림길은 추천안)"]
    L1 += [f"- {k}: {v}" for k, v in DECISIONS.items()]
    L1 += ["", "## 출력 파일 (sha256)"] + [f"- `{p}` {s[:16]}" for p, s in rep["outputs"].items()]
    return "\n".join(L1) + "\n"


# ---------------------------------------------------------------- status
def cmd_status():
    out = dict(stop=STOP_FILE.read_text().strip() if STOP_FILE.is_file() else None)
    if RES_TARGETS.is_file():
        for which in ("trial", "main"):
            m = list_mids(which)
            out[which] = dict(n=len(m), done=sum(done(x) for x in m),
                              errors=[x for x in m if (mdir(x) / "error.txt").is_file() and not done(x)][:20])
    out["plan"] = {k: {kk: vv for kk, vv in v.items() if kk != "mids"} for k, v in (_load_json(PLAN, {}) or {}).items()}
    out["chain_log"] = (_load_json(VDIR / "chain_log.json", []) or [])[-5:]
    print(json.dumps(out, indent=1, default=str), flush=True)
    return 0


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description="D1LADDER Phase V (refs_v.py)")
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("targets")
    c = sp.add_parser("check")
    c.add_argument("--force", action="store_true")
    w = sp.add_parser("work")
    w.add_argument("which", choices=["trial", "main"])
    w.add_argument("--nshards", type=int, required=True)
    w.add_argument("--shard", type=int, default=None)
    for name in ("conf", "dft"):
        x = sp.add_parser(name)
        x.add_argument("mid")
    ch = sp.add_parser("chain")
    ch.add_argument("stage", choices=["trial", "main"])
    ch.add_argument("--round", type=int, default=0)
    ch.add_argument("--max-run", type=int, default=4)
    sp.add_parser("decide")
    sp.add_parser("collect")
    sp.add_parser("status")
    a = ap.parse_args(argv)
    if a.cmd == "targets":
        return cmd_targets()
    if a.cmd == "check":
        return cmd_check(a.force)
    if a.cmd == "work":
        shard = a.shard if a.shard is not None else os.environ.get("SLURM_ARRAY_TASK_ID")
        if shard is None or not (0 <= int(shard) < a.nshards):
            print("--shard or SLURM_ARRAY_TASK_ID in [0, nshards) required", file=sys.stderr)
            return 2
        return cmd_work(a.which, a.nshards, int(shard))
    if a.cmd in ("conf", "dft"):
        stages = ("conf",) if a.cmd == "conf" else ("dft",)
        return cmd_work("trial", 1, 0, stages=stages, mids=[a.mid])
    if a.cmd == "chain":
        return cmd_chain(a.stage, a.round, a.max_run)
    if a.cmd == "decide":
        r = cmd_decide()
        return 3 if r.get("stop") else (0 if r.get("ready") else 2)
    if a.cmd == "collect":
        return cmd_collect()
    if a.cmd == "status":
        return cmd_status()
    return 2


if __name__ == "__main__":
    sys.exit(main())
