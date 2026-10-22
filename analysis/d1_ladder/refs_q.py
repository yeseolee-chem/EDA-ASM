#!/usr/bin/env python3
"""refs_q.py — D1-REF·LADDER Phase Q (docs/specs/D1LADDER.md 2, Q-1 .. Q-6): reference-structure audit and re-assembly
of d1 / d2 / barrier with the lowest label-level reference of each molecule. No new quantum chemistry. sbatch only:

  cd analysis/d1_ladder && sbatch -p <cpu1|cpu2> -J ladQ --cpus-per-task=2 --mem=24G \
      --export=ALL,LADENV=reactot jobs/lad.slurm refs_q.py run
  python refs_q.py run [--force]      # exit 0 = all gates pass (or up to date), 3 = spec 16 Q STOP, 2 = usage

Outputs (spec 2, 15: written only under $LAD and analysis/d1_ladder/results, atomically; a rerun with unchanged inputs
and code exits 0 'up to date', otherwise it recomputes everything and rewrites the same files):
  $LAD/labels_d1_cr.json          9,266 target rows = labels_d1.json rows; changed rows carry new e_frag{k}_rel_eh,
                                  d1_kcal, d2_kcal, barrier_kcal, identity_residual_kcal, channel_closure_residual_kcal
                                  (all from the label assembly function); every row gets a 'cr' dict (keys, sources)
  results/labels_d1_cr.parquet    rxn_id, job_id, original / new d1 d2 barrier, changed fragment(s), change amounts,
                                  source rxn_id / job_id per fragment, molecule keys, energies
  $LAD/F_pairs_cr.csv             = results/F_pairs.csv of D1PAIR with dd_d1 / dd_d2 / dd_barrier recomputed (Q-6)
  results/Q_audit.csv             one row per molecule key (Q-3)
  results/Q6_azide_D_para.csv     azide/D para pairs, original vs new dd_barrier / dd_d2 / dd_d1
  results/Q_report.json           everything below + gates Q-a .. Q-e + sha256 of inputs / outputs
  $LAD/Q_keys.parquet             one row per (reaction, fragment): key, energies, dev, source (for Phase V and P)
  on a gate failure: results/Q_report.json (stop true) and $LAD/labels_d1_cr.REJECTED.json; exit 3
  A recompute first writes Q_report.json with all_pass false (a killed run never leaves an old pass behind) and renames
  the previous outputs to <name>.stale; any STOP / crash renames the outputs it already wrote to <name>.stale, so after
  a STOP no output file looks valid; a pass deletes the .stale copies and any old REJECTED file. Readers must still
  check Q_report.json all_pass and its output sha256 (refs_v.q_ready does).

Q-1 findings (also in Q_report.json 'found')
  * reference single-point energy keys: e_frag1_rel_eh, e_frag2_rel_eh (Eh; FINAL SINGLE POINT ENERGY of
    frag{1,2}_rel.out at the label level, geometry = rel{1,2}_file = ref_dipole.xyz / ref_dipolarophile.xyz);
    d1_kcal, d2_kcal, barrier key = barrier_kcal. role1 = dipole and role2 = dipolarophile in all 9,496 rows.
  * assembly function = label_true/scripts/stage3_parse.py::parse_one_rxn(rdir, meta_row) of the D1 run code copy
    (/gpfs/tmp_cpu2/yeseo1ee/d1/home01/d1_env/EDA-ASM; byte-identical to label_true/scripts/stage3_parse.py of this
    repo), called by D1_build/d1_autode_pilot/assemble.py main() (rec = s3.parse_one_rxn(spd, meta), s3 from
    pilot_common.import_repo_modules); label.json = its record unchanged (+ status promotion, gates, flags):
        d1 = (E(frag1_dist) - E(frag1_rel)) x 627.5094740631, d2 likewise,
        barrier = (E(eda) - E(frag1_rel) - E(frag2_rel)) x 627.5094740631.
    It reads ORCA outputs, so it cannot take a label row. Thinnest adapter (assemble()): parse_one_rxn is called
    unchanged; only its three module-level readers are swapped for the duration of the call —
    orca_terminated_normally -> True, read_fspe(<calc>.out) -> the label energy of <calc> (eda -> e_ab_eh, eda_frag{k} ->
    e_frag{k}_dist_ghost_eh, frag{k}_dist -> e_frag{k}_dist_eh, frag{k}_rel -> e_frag{k}_rel_eh or the replacement),
    parse_eda_table -> the label's eda_raw_eh; meta_row = the label's own stage-1 meta fields. Gate Q-a proves that the
    original energies reproduce every row (d1, d2, barrier <= 1e-9; every other float it returns is reported).
  * constrained dipole field: ref_result.json 'port' (= label 'ref_port'; assemble.py copies ref_result.port):
    port.constrained (make_reference.coley_port step 5 = some tracked dihedral not rotatable), the frozen dihedral =
    port.scans[i].quad with rotatable false (geometry indices of ref_dipole.xyz), port.use_alt, port.alt_config /
    port.orig_config. G (E_sp + G_cont, autodE level): port.alt_dft.G_eh for the Coley alt, ts_result.json
    reactants[i].e_sp_eh + g_cont_eh for autodE's reactants (the original dipole and the dipolarophile reference).
    The ledger records (summary tars / UBAI folders, read as analysis/d1_g2/prep_x.py does) are compared with the label
    ref_port at run time (STOP if a key-relevant field differs).

Decisions (user 2026-10-22: every branch = the recommended option; recorded here and in Q_report.json 'decisions')
  D1  environment LADENV=reactot (python 3.10, rdkit 2026.3.2, pandas, pyarrow, networkx): the D1 venv has rdkit but no
      pyarrow (labels_d1_hf.parquet, the parquet outputs). The D1 code (stage3_parse / _common / stereo_utils) is
      imported read-only from the D1 code copy with bytecode writing off.
  D2  population = the 9,266 targets (lad_common.targets()). The 230 D'' rows are not used for E_min, not re-assembled
      and not written to labels_d1_cr.json; their keys are computed and the report gives how many keys / rows they share
      with the targets and how many would have lowered a target E_min by > 0.05 kcal/mol.
  D3  which reactant is the dipole: the rxn_smiles (results/X_targets.csv of D1-G2) reactant component whose formula
      (with H) equals the reference xyz of the role1 / role2 fragment (and n_f1 / n_f2); cross-check: the ring rule
      fragmenter.dipole_smiles_index on X_targets mapped_smiles gives the same molecule (formula of the reference xyz
      and the same mirror-min canonical SMILES, map numbers removed). Not unique, or the ring-rule dipole differs in
      formula or SMILES -> STOP (the key's SMILES would be in doubt; fixer 2026-10-22, was formula-only before).
  D4  molecule key = '<kind>:' + RDKit canonical isomeric SMILES of that component, enantiomers merged (the
      lexicographically smaller of the SMILES and its mirror image, every tetrahedral tag inverted); E/Z and other
      diastereomers stay apart through the SMILES. Constrained dipole references (port.constrained true: both the
      constrained alt and the constrained original that the alt reproduced, RMSD <= 0.05 A, are in the TS-compatible
      state) get '|fixed:<bond>:<c|t|s>' per frozen dihedral, computed from the reference geometry itself: bond = the two
      central atoms named by their element + Weisfeiler-Lehman hash on the bond graph (graph invariant), c/t = the
      dihedral through the highest-hash heavy neighbour on each side, 's' when that side has two equivalent heavy
      neighbours (not stereogenic). Reason: the D1 tag strings (alt_config / orig_config) use the first heavy neighbour
      of the mapped SMILES and the map-number order of the backbone, which are not canonical across reactions.
  D5  the Q-2 check (same element composition and bond graph within a key: networkx isomorphism, node label = element,
      on fragmenter.bond_matrix with the D1 Si radius 1.11 A added as analysis/d1_pair/site_si does) is run on the
      D1_JOBS reference xyz, both per key and per SMILES-level key (key_plain = '<kind>:<SMILES>', which holds the
      constrained and free references of one molecule): the fixed-state suffix names bond ends by graph-dependent WL
      hashes, so a bond-graph difference inside one SMILES would otherwise only split the key instead of stopping
      (fixer 2026-10-22). Added checks (spec 16 'any time': E_min would otherwise mix levels or structures): within a
      key the D1-HF copies of the label inputs (inputs/<T>/orig/frag{k}_rel.inp) have the same keywords, blocks
      (except %maxcore / %pal), charge and multiplicity, and their coordinates equal the reference xyz (<= 1e-6 A).
  D6  E_min = the lowest e_frag{k}_rel_eh of the key over the targets (ties: smallest job_id); replaced iff
      (E_orig - E_min) x 627.5094740631 > 0.05 kcal/mol; the replacement energy, the source job_id / rxn_id / fragment
      and the source coordinates (D1_JOBS xyz and D1-HF orig input) are recorded.
  D7  unchanged rows are copied verbatim; changed rows change only the fields named above (flags such as
      d1_below_min, rel{k}_file, gates are not touched; 'cr' holds the new reference). Gates are evaluated on the JSON
      round trip of what is written. Q-c and Q-e are applied to every row (unchanged rows: differences exactly 0).
      Q-b also requires every other field to be identical and the recomputed channels to be within 1e-9.
  D8  Q-6: dd_t = y_a - y_b with the d1_pairs.py / d1_anchor_pairs.py TARGETS map (the code that computed F_pairs.csv
      dd_*; d1_features.py pairs only copies them); proven by reproducing every dd_* column of F_pairs.csv from the
      original labels (<= 1e-9, else STOP). F_pairs_cr.csv = F_pairs.csv byte for byte except dd_d1 / dd_d2 / dd_barrier
      of pairs with a changed endpoint (repr of the float); pair list, flags (hyg, in_P_*), order and S_folds.csv
      unchanged (pairs whose hygiene status would flip are only counted).
  D9  HF pre-check (spec 0-1): the same keys on labels_d1_hf.parquet geom == dft e_frag{k}_rel_eh, minimum over the
      targets with HF values; reaction flag = max fragment dev > 1 kcal/mol; agreement on all targets and on the
      7,410 P_all reactions; the same with SMILES-only keys (no fixed state) for comparison with the reviewer's numbers.
  D10 G vs E: per key with >= 2 rows, the row with the lowest G = E_sp + G_cont (autodE level) vs the row with the
      lowest label E; counted when different and when the G-chosen row is > 0.05 kcal/mol above E_min.
  D11 3D stereo (RDKit perception on the template-matched reference geometry, mirror-merged) is a diagnostic only:
      the spec key is the reaction SMILES; keys holding more than one 3D signature and replacements across signatures
      are counted.
  D12 file name: spec 8 lists refs.py (Q·V); this task names refs_q.py, so Phase Q lives here and Phase V imports it.
  D13 ledger records (fixer 2026-10-22): a summary tar whose sha256 differs from bundles.tsv (or is missing), or a
      d1_status.csv whose sha256 differs from d1pair_common.D1_STATUS_SHA, is a STOP (as analysis/d1_g2/prep_x.py);
      jobs without ref_result.json are only counted (the keys use the label's own ref_port).
  D14 failure handling (fixer 2026-10-22): Q-a is checked before the re-assembly (a Q-a failure is a Q-5 STOP with its
      report, not a crash inside reassemble); any unexpected exception is turned into a STOP with Q_report.json.

Reuse (Phase V V-4 and Phase P rung H):
  K, graphs, geoms, errs = molecule_keys(rows, rxn_smiles, mapped)   # Q-2 keys (or load_keys() = $LAD/Q_keys.parquet)
  A = audit(K, energy, key="key")       # energy {(job_id, frag): Eh} -> e_min / src / dev per (reaction, fragment)
  ov = choose_override(A, K)            # {(job_id, frag): {e_eh, src, src_job, src_rxn, src_frag, ref_xyz, ref_inp}}
  new_rows, recs = reassemble(rows, ov, K, tag)        # stage3_parse.parse_one_rxn with only the references replaced
  text, new_rt = json_roundtrip(new_rows)              # (text written, rows read back)
  G = gates(rows, new_rt, ov, recs, gate_qa(rows))
  write_outputs(tag, text, table) / write_pairs(tag, orig_by_job, new_by_job)   # tag 'cr2' -> labels_d1_cr2.json,
                                        # F_pairs_cr2.csv (V-4); q6_report(..., azd_path=...) needs its own path per tag
  pairs_dd(F, labels_by_job)            # dd_* of D1PAIR pairs from any label set
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True        # no __pycache__ in analysis/d1_pair, d1_hf, d1_g2 or the D1 code copy (read only)

import argparse  # noqa: E402
import contextlib  # noqa: E402
import csv  # noqa: E402
import importlib.util  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import platform  # noqa: E402
import tarfile  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lad_common as L  # noqa: E402

DP = L.DP
KCAL = L.KCAL
CODE_VERSION = "d1-ladder-q1"

# ---------------------------------------------------------------- inputs (read only)
X_TARGETS = L.G2_DIR / "results" / "X_targets.csv"     # D1-G2 Phase X: rxn_smiles, mapped_smiles of the 9,266 targets
F_PAIRS = DP.RES / "F_pairs.csv"                       # D1PAIR pair labels (dd_*)
S_FOLDS = DP.RES / "S_folds.csv"                       # D1PAIR folds (unchanged)
D1_PAIRS_PY = DP.HERE / "d1_pairs.py"                  # dd_* of the reference pairs (appendix B)
D1_ANCHOR_PY = DP.HERE / "d1_anchor_pairs.py"          # dd_* of the anchor pairs (appendix E)

# ---------------------------------------------------------------- outputs
RES_REPORT = L.RES / "Q_report.json"
RES_AUDIT = L.RES / "Q_audit.csv"
RES_AZD = L.RES / "Q6_azide_D_para.csv"
LAD_KEYS = L.LAD / "Q_keys.parquet"


def out_paths(tag="cr"):
    return dict(json=L.LAD / f"labels_d1_{tag}.json", parquet=L.RES / f"labels_d1_{tag}.parquet",
                pairs=L.LAD / f"F_pairs_{tag}.csv", rejected=L.LAD / f"labels_d1_{tag}.REJECTED.json")


# ---------------------------------------------------------------- rules (spec 2)
THR_REPLACE = 0.05                                     # Q-4 kcal/mol
THRS = (0.05, 0.5, 1.0, 2.0)                           # Q-3 report thresholds
TOL = 1e-9                                             # Q-a, Q-c, Q-d (kcal/mol)
XYZ_TOL = 1e-6                                         # A, label input coordinates vs reference xyz
HF_FLAG = 1.0                                          # spec 0-1 reviewer pre-check: dev > 1 kcal/mol
CYCLOOCTYNE = ("D", "Dp")                              # skeletons of the cyclooctyne dipolarophiles (spec 0-1, V-1 b)

# label fields (Q-1)
REF_E = {1: "e_frag1_rel_eh", 2: "e_frag2_rel_eh"}
D_KEY = {1: "d1_kcal", 2: "d2_kcal"}
BARRIER_KEY = "barrier_kcal"
SPE_KEY = {"eda": "e_ab_eh", "eda_frag1": "e_frag1_dist_ghost_eh", "eda_frag2": "e_frag2_dist_ghost_eh",
           "frag1_dist": "e_frag1_dist_eh", "frag2_dist": "e_frag2_dist_eh",
           "frag1_rel": "e_frag1_rel_eh", "frag2_rel": "e_frag2_rel_eh"}        # stage3_parse.SPES -> label key
META_KEYS = ("rxn_id", "ts_file", "n_atoms", "n_f1", "n_f2", "rel1_file", "rel2_file", "role1", "role2", "charge1",
             "charge2", "alt_used", "flag_foreign_bond", "flag_async", "formed_d1", "formed_d2", "formed_pairs_ts",
             "recovery_method")                                              # what parse_one_rxn reads from meta_row
CHANGED_KEYS = ("e_frag1_rel_eh", "e_frag2_rel_eh", "d1_kcal", "d2_kcal", "barrier_kcal", "identity_residual_kcal",
                "channel_closure_residual_kcal")                             # parse_one_rxn outputs that use frag*_rel
EDA7 = ("elst_dft", "pauli_dft", "oi_dft", "disp_dft", "cpcm_dft", "cds_dft", "bsse_shift_kcal")   # Q-b (+ c_ghost)
SUM9 = tuple(DP.D1_TCOL[t] for t in DP.CHANNELS9)                            # Q-d: d1 d2 6 channels c_ghost
ASSEMBLE_OK = ("ok", "eda_sum_mismatch")       # parse_one_rxn status of a label row (assemble.py promoted the second)
PAIR_T = ("d1", "d2", "barrier")               # the dd_* that Q can change

FOUND = dict(
    ref_energy_keys={"frag1": REF_E[1], "frag2": REF_E[2]},
    ref_energy_meaning="Eh, FINAL SINGLE POINT ENERGY of frag{1,2}_rel.out (label level) at rel{1,2}_file "
                       "(ref_dipole.xyz / ref_dipolarophile.xyz of the D1 job)",
    d_keys={"d1": D_KEY[1], "d2": D_KEY[2]}, barrier_key=BARRIER_KEY,
    formulas="d1 = (E(frag1_dist) - E(frag1_rel)) x 627.5094740631; d2 likewise; "
             "barrier = (E(eda) - E(frag1_rel) - E(frag2_rel)) x 627.5094740631",
    assembly_function="label_true/scripts/stage3_parse.py::parse_one_rxn(rdir, meta_row) of the D1 code copy "
                      "/gpfs/tmp_cpu2/yeseo1ee/d1/home01/d1_env/EDA-ASM (= this repo's label_true/scripts/stage3_parse.py)",
    called_by="D1_build/d1_autode_pilot/assemble.py main(): rec = s3.parse_one_rxn(spd, meta), "
              "s3 = pilot_common.import_repo_modules(cfg)[1]; label.json = rec + status promotion / gates / flags",
    adapter="refs_q.assemble(): parse_one_rxn unchanged; for the call its module-level orca_terminated_normally / "
            "read_fspe / parse_eda_table return True / the label energy of each <calc>.out / the label eda_raw_eh",
    frag_roles="role1 = dipole, role2 = dipolarophile (checked at run time)",
    constrained_field="ref_result.json port.constrained (= label ref_port.constrained); frozen dihedral = "
                      "port.scans[i].quad with rotatable false (ref_dipole.xyz indices); port.use_alt, "
                      "port.alt_config / orig_config (make_reference.coley_port steps 3-7)",
    G_field="G = E_sp + G_cont (autodE level): Coley alt = ref_port.alt_dft.G_eh; autodE reactants (original dipole, "
            "dipolarophile reference) = ts_result.json reactants[i].e_sp_eh + g_cont_eh (rule: dG < -0.239 kcal/mol)",
)
DECISIONS = {
    "D1_env": "LADENV=reactot (rdkit + pyarrow); D1 code imported read-only from the D1 code copy, no bytecode",
    "D2_dpp": "230 D'' rows: not used for E_min, not re-assembled, not in labels_d1_cr.json; shared keys reported",
    "D3_dipole": "rxn_smiles component with the formula of the reference xyz (n_f1 / n_f2); cross-check "
                 "fragmenter.dipole_smiles_index(mapped_smiles) = same formula and same mirror-min SMILES; not unique / "
                 "not agreeing -> STOP",
    "D4_key": "'<kind>:' + mirror-min canonical isomeric SMILES; constrained dipole + '|fixed:<bond>:<c|t|s>' from the "
              "reference geometry with WL-hash canonical neighbours (D1 tags are not canonical across reactions)",
    "D5_checks": "Q-2 graph check on D1_JOBS xyz (Si 1.11 A as site_si) per key and per SMILES-level key (key_plain); "
                 "plus label-input level / charge / coordinate identity within a key (D1-HF inputs/<T>/orig) -> STOP "
                 "when violated",
    "D6_emin": "lowest e_frag{k}_rel_eh over the targets, ties = smallest job_id; replace iff dev > 0.05 kcal/mol",
    "D7_fields": "unchanged rows verbatim; changed rows: only the 7 parse_one_rxn outputs that use frag*_rel + 'cr'; "
                 "Q-c / Q-e on every row; Q-b also every other field identical",
    "D8_pairs": "dd_t = y_a - y_b (d1_pairs.py / d1_anchor_pairs.py TARGETS), proven on F_pairs.csv <= 1e-9; only "
                "dd_d1 / dd_d2 / dd_barrier strings of pairs with a changed endpoint rewritten; flags, folds unchanged",
    "D9_hf": "same keys on labels_d1_hf.parquet geom=dft; reaction flag max dev > 1; all targets and P_all reactions; "
             "SMILES-only keys reported too",
    "D10_G": "per key: argmin G (E_sp + G_cont, autodE) vs argmin label E",
    "D11_stereo3d": "3D-perceived stereo is a diagnostic only (spec key = reaction SMILES)",
    "D12_name": "spec 8 says refs.py; the task names refs_q.py (Phase V imports it)",
    "D13_ledger": "summary tar sha256 != bundles.tsv (or missing) or d1_status.csv sha256 changed -> STOP (prep_x.py "
                  "rule); jobs without ref_result.json only counted",
    "D14_failures": "Q-a checked before re-assembly; unexpected exceptions -> STOP with Q_report.json; outputs of a "
                    "STOP / crashed run renamed to .stale; a recompute starts with Q_report.json all_pass false",
}

# ---------------------------------------------------------------- read-only modules
def _load(name, path):
    """import a module from its file without putting its folder on sys.path (stage3_parse adds its own folder; that
    entry is removed again by stage3())."""
    path = Path(path)
    m = sys.modules.get(name)
    if m is not None:
        if Path(m.__file__).resolve() != path.resolve():
            L.stop(f"module {name} already imported from {m.__file__}, expected {path}")
        return m
    if not path.is_file():
        L.stop(f"{path} not found")
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def g2c():
    """analysis/d1_g2/g2_common.py: D1_REPO (the D1 run code copy), PILOT, LEDGER (constants only)."""
    return _load("g2_common", L.G2_DIR / "g2_common.py")


def hf_common():
    """analysis/d1_hf/hf_common.py: INPUTS (D1-HF scratch inputs/<T>/{orig,hf}/*.inp), parse_inp."""
    return _load("hf_common", L.HF_DIR / "hf_common.py")


def s3_dir():
    return g2c().D1_REPO / "label_true" / "scripts"


def stage3():
    """the label assembly module of the D1 run (spec 16 Q STOP if the function cannot be found)."""
    if "stage3_parse" in sys.modules:
        return sys.modules["stage3_parse"]
    d = s3_dir()
    s3 = _load("stage3_parse", d / "stage3_parse.py")
    for p in (str(d), str(d.resolve())):
        while p in sys.path:
            sys.path.remove(p)
    c = sys.modules.get("_common")
    if c is None or Path(c.__file__).resolve() != (d / "_common.py").resolve():
        L.stop(f"stage3_parse imported a _common other than {d / '_common.py'}")
    if not callable(getattr(s3, "parse_one_rxn", None)):
        L.stop("assembly function stage3_parse.parse_one_rxn not found (spec 16 Q)")
    if set(s3.SPES) != set(SPE_KEY):
        L.stop(f"stage3_parse.SPES {s3.SPES} != the label energy map {sorted(SPE_KEY)}")
    if s3.EH_TO_KCAL != KCAL:
        L.stop(f"stage3_parse EH_TO_KCAL {s3.EH_TO_KCAL} != lad_common.KCAL {KCAL}")
    return s3


def stereo_utils():
    """D1_build/d1_autode_pilot/stereo_utils.py of the D1 code copy (template_match, dihedral_deg, xyz block)."""
    return _load("stereo_utils", g2c().PILOT / "stereo_utils.py")


def fragmenter():
    """the espley fragmenter (bond_matrix, read_xyz, dipole_smiles_index) on DP's path, with the D1 Si entries."""
    import fragmenter as fr
    if Path(fr.__file__).resolve() != (DP.ESP_DIR / "fragmenter.py").resolve():
        L.stop(f"fragmenter imported from {fr.__file__}, expected {DP.ESP_DIR / 'fragmenter.py'}")
    fr.COV.setdefault("Si", 1.11)                      # = analysis/d1_pair/site_si and the D1 code copy fragmenter.py
    fr.Z.setdefault("Si", 14)
    return fr


# ---------------------------------------------------------------- small helpers
def _vc(s):
    return {str(k): int(v) for k, v in pd.Series(s).value_counts().sort_index().items()}


def _q(x):
    x = np.asarray(x, float)
    if not len(x):
        return {}
    return dict(n=int(len(x)), p50=float(np.quantile(x, 0.5)), p90=float(np.quantile(x, 0.9)),
                p99=float(np.quantile(x, 0.99)), max=float(x.max()), mean=float(x.mean()))


def _maxabs(a, b):
    """max |a - b|; inf when any value is not finite (a NaN never passes a tolerance)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not len(a):
        return 0.0
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return float("inf")
    return float(np.abs(a - b).max())


def _jd(x):
    return json.dumps(x, sort_keys=True)


def json_roundtrip(rows):
    """(text, rows) as they are written / read back."""
    text = json.dumps(rows)
    return text, json.loads(text)


def env_info():
    import networkx
    out = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
               networkx=networkx.__version__, host=platform.node())
    try:
        import rdkit
        out["rdkit"] = rdkit.__version__
    except ImportError:
        out["rdkit"] = None
    return out


# ---------------------------------------------------------------- Q-1: assembly through the label function
@contextlib.contextmanager
def _record_io(s3, energies, raw):
    saved = s3.orca_terminated_normally, s3.read_fspe, s3.parse_eda_table
    s3.orca_terminated_normally = lambda p: True
    s3.read_fspe = lambda p, expected_count=1: float(energies[Path(p).stem])
    s3.parse_eda_table = lambda p: dict(raw)
    try:
        yield
    finally:
        s3.orca_terminated_normally, s3.read_fspe, s3.parse_eda_table = saved


def assemble(row, rel=None):
    """stage3_parse.parse_one_rxn on a label row; rel = {k: Eh} replaces E(frag{k}_rel). Returns its record."""
    s3 = stage3()
    E = {s: float(row[k]) for s, k in SPE_KEY.items()}
    for k, e in (rel or {}).items():
        E[f"frag{int(k)}_rel"] = float(e)
    meta = {k: row[k] for k in META_KEYS}
    with _record_io(s3, E, row["eda_raw_eh"]):
        rec = s3.parse_one_rxn(Path("/nonexistent/refs_q") / str(row["job_id"]), meta)
    if rec.get("status") not in ASSEMBLE_OK:
        raise RuntimeError(f"{row['job_id']}: parse_one_rxn status {rec.get('status')} {rec.get('_error', '')}")
    return rec


def gate_qa(rows):
    """Q-a: the original energies reproduce d1, d2, barrier of every row (<= 1e-9); other floats reported."""
    gk = (D_KEY[1], D_KEY[2], BARRIER_KEY)
    mx, other, bad, err = dict.fromkeys(gk, 0.0), {}, [], []
    for row in rows:
        try:
            rec = assemble(row)
        except Exception as e:                                         # noqa: BLE001
            err.append(f"{row['job_id']}: {type(e).__name__}: {e}")
            continue
        for k, v in rec.items():
            if k in gk or k.startswith("_") or not isinstance(v, float):
                continue
            o = row.get(k)
            if isinstance(o, (int, float)) and not isinstance(o, bool):
                other[k] = max(other.get(k, 0.0), abs(v - o))
        for k in gk:
            d = abs(rec[k] - row[k])
            mx[k] = max(mx[k], d)
            if not d <= TOL:
                bad.append(row["job_id"])
    bad = sorted(set(bad))
    return dict(ok=not bad and not err, tol=TOL, n_rows=len(rows), max_abs=mx, other_max_abs=other,
                n_fail=len(bad), fail=bad[:30], errors=err[:30], n_errors=len(err))


# ---------------------------------------------------------------- Q-2: molecule keys
def mirror_min(smi):
    """canonical isomeric SMILES, enantiomers merged: min(SMILES, SMILES with every tetrahedral tag inverted)."""
    from rdkit import Chem
    m = Chem.MolFromSmiles(smi)
    if m is None:
        raise ValueError(f"RDKit cannot parse {smi!r}")
    s0 = Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(m)))
    inv, n = Chem.Mol(m), 0
    for a in inv.GetAtoms():
        t = a.GetChiralTag()
        if t == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
            a.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW); n += 1
        elif t == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
            a.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW); n += 1
    if not n:
        return s0
    s1 = Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(inv)))
    return min(s0, s1)


def _formula_smiles(smi):
    from rdkit import Chem
    m = Chem.MolFromSmiles(smi)
    if m is None:
        raise ValueError(f"RDKit cannot parse {smi!r}")
    return Counter(a.GetSymbol() for a in Chem.AddHs(m).GetAtoms())


def _fstr(c):
    return "".join(f"{e}{n}" for e, n in sorted(c.items()))


def bond_graph(syms, xyz):
    """networkx graph of fragmenter.bond_matrix (node attribute el = element)."""
    import networkx as nx
    fr = fragmenter()
    adj, _ = fr.bond_matrix(list(syms), np.asarray(xyz, float))
    G = nx.Graph()
    G.add_nodes_from((i, {"el": s}) for i, s in enumerate(syms))
    ii, jj = np.nonzero(np.triu(adj, 1))
    G.add_edges_from(zip(ii.tolist(), jj.tolist()))
    return G


def wl_atom_hashes(G):
    """graph-invariant atom labels: last Weisfeiler-Lehman subgraph hash (n_atoms iterations, label = element)."""
    import networkx as nx
    h = nx.weisfeiler_lehman_subgraph_hashes(G, node_attr="el", iterations=max(1, G.number_of_nodes()))
    return {n: v[-1] for n, v in h.items()}


def fixed_state(G, syms, xyz, quad):
    """(state, tag on the D1 quad itself) of one frozen dihedral of a constrained dipole reference (D4)."""
    su = stereo_utils()
    a, u, v, d = (int(x) for x in quad)
    for x, y in ((a, u), (u, v), (v, d)):
        if not G.has_edge(x, y):
            raise ValueError(f"frozen dihedral {quad}: bond {x}-{y} is not in the reference bond graph")
    h = wl_atom_hashes(G)

    def side(x, y):
        nb = [n for n in G.neighbors(x) if n != y and syms[n] != "H"]
        if not nb:
            raise ValueError(f"frozen dihedral {quad}: atom {x} has no heavy neighbour besides {y}")
        top = max(h[n] for n in nb)
        best = [n for n in nb if h[n] == top]
        return best[0] if len(best) == 1 else -1

    nu, nv = side(u, v), side(v, u)
    if nu < 0 or nv < 0:
        tag = "s"
    else:
        tag = "c" if abs(su.dihedral_deg(xyz[nu], xyz[u], xyz[v], xyz[nv])) < 90 else "t"
    tq = "c" if abs(su.dihedral_deg(xyz[a], xyz[u], xyz[v], xyz[d])) < 90 else "t"
    ends = sorted((f"{syms[u]}{h[u][:10]}", f"{syms[v]}{h[v][:10]}"))
    return f"{ends[0]}~{ends[1]}:{tag}", tq


def _d1_tag(port, i):
    """D1's own c/t tag of tracked dihedral i of the used reference (alt_config if use_alt else orig_config)."""
    cfg = port.get("alt_config") if port.get("use_alt") else port.get("orig_config")
    pos = [n for n, c in enumerate(port.get("ts_dipole_config") or "") if c != "?"]
    if cfg is None or i >= len(pos) or pos[i] >= len(cfg):
        return ""
    return cfg[pos[i]]


def stereo3d_sig(syms, xyz, smiles, charge):
    """D11 diagnostic: mirror-merged SMILES with stereo perceived from the reference geometry."""
    from rdkit import Chem
    su = stereo_utils()
    try:
        m, _, _ = su.template_match(su.xyz_block_from_atoms(syms, xyz), smiles, int(charge))
        Chem.AssignStereochemistryFrom3D(m)
        return mirror_min(Chem.MolToSmiles(Chem.RemoveHs(m)))
    except Exception as e:                                             # noqa: BLE001
        return f"error:{type(e).__name__}"


def molecule_keys(rows, rxn_smiles=None, mapped=None, stereo3d=True):
    """Q-2 keys of both fragments of every row -> (K frame, graphs {(job_id, frag): nx.Graph},
    geoms {(job_id, frag): (syms, xyz)}, errors [str]). rxn_smiles / mapped: {job_id: SMILES} (default: the label's
    rxn_smiles, no ring cross-check)."""
    import networkx as nx
    fr = fragmenter()
    from rdkit import Chem
    out, graphs, geoms, errors = [], {}, {}, []
    for row in rows:
        j = row["job_id"]
        try:
            smi = (rxn_smiles or {}).get(j) or row["rxn_smiles"]
            comps = smi.split(">>")[0].split(".")
            if len(comps) != 2:
                raise ValueError(f"{len(comps)} reactant components in {smi!r}")
            cform = [_formula_smiles(c) for c in comps]
            geo = {}
            for k in (1, 2):
                p = DP.D1_JOBS / j / row[f"rel{k}_file"]
                syms, xyz = fr.read_xyz(p)
                if len(syms) != int(row[f"n_f{k}"]):
                    raise ValueError(f"{p}: {len(syms)} atoms != n_f{k} = {row[f'n_f{k}']}")
                geo[k] = (p, list(syms), np.asarray(xyz, float))
            kinds = {k: row[f"role{k}"] for k in (1, 2)}
            if sorted(kinds.values()) != ["dipolarophile", "dipole"]:
                raise ValueError(f"roles {kinds}")
            cand = {k: [i for i in (0, 1) if cform[i] == Counter(geo[k][1])] for k in (1, 2)}
            if not (len(cand[1]) == 1 and len(cand[2]) == 1 and cand[1][0] != cand[2][0]):
                raise ValueError(f"reactant components {comps} do not match the reference formulas "
                                 f"{_fstr(Counter(geo[1][1]))} / {_fstr(Counter(geo[2][1]))} one to one")
            comp_of = {1: cand[1][0], 2: cand[2][0]}
            kdip = [k for k in (1, 2) if kinds[k] == "dipole"][0]
            ring = ""
            mp = (mapped or {}).get(j)
            if mp:
                r = fr.dipole_smiles_index(mp)
                if r is None:
                    raise ValueError(f"fragmenter.dipole_smiles_index found no [3+2] ring in {mp!r}")
                mcomp = mp.split(">>")[0].split(".")[r[0]]
                if _formula_smiles(mcomp) != Counter(geo[kdip][1]):
                    raise ValueError(f"ring-rule dipole {mcomp!r} formula != the dipole reference {geo[kdip][0]}")
                mm = Chem.MolFromSmiles(mcomp)
                for at in mm.GetAtoms():
                    at.SetAtomMapNum(0)
                if mirror_min(Chem.MolToSmiles(mm)) != mirror_min(comps[comp_of[kdip]]):
                    raise ValueError(f"ring-rule dipole {mcomp!r} has the reference formula but is not the formula-chosen "
                                     f"rxn_smiles component {comps[comp_of[kdip]]!r} (D3)")
                ring = "same"
            port = row.get("ref_port") or {}
            recs = []
            for k in (1, 2):
                p, syms, xyz = geo[k]
                G = bond_graph(syms, xyz)
                skey = mirror_min(comps[comp_of[k]])
                cons = bool(kinds[k] == "dipole" and port.get("constrained"))
                state, d1tag, tq = "", "", ""
                if cons:
                    frozen = [(i, s["quad"]) for i, s in enumerate(port.get("scans") or []) if s.get("rotatable") is False]
                    if not frozen:
                        raise ValueError("constrained reference without a frozen dihedral in ref_port.scans")
                    tracked = [list(q) for q in (port.get("tracked") or [])]
                    st = [fixed_state(G, syms, xyz, q) for _, q in frozen]
                    state = ";".join(sorted(s for s, _ in st))
                    tq = "".join(t for _, t in st)
                    d1tag = "".join(_d1_tag(port, tracked.index(list(q))) if list(q) in tracked else "" for _, q in frozen)
                key = f"{kinds[k]}:{skey}" + (f"|fixed:{state}" if cons else "")
                recs.append(dict(
                    job_id=j, rxn_id=int(row["rxn_id"]), frag=k, kind=kinds[k], skeleton=row["skeleton"],
                    dipole=row["dipole"], smiles=comps[comp_of[k]], smiles_key=skey, key_plain=f"{kinds[k]}:{skey}",
                    constrained=cons, fixed_state=state, fixed_tag_quad=tq, fixed_tag_d1=d1tag, key=key,
                    formula=_fstr(Counter(syms)), n_atoms=len(syms), charge=int(row[f"charge{k}"]), xyz=str(p),
                    wl_graph=nx.weisfeiler_lehman_graph_hash(G, node_attr="el"),
                    stereo3d=stereo3d_sig(syms, xyz, comps[comp_of[k]], row[f"charge{k}"]) if stereo3d else "",
                    ring_check=ring))
                graphs[(j, k)] = G
                geoms[(j, k)] = (syms, xyz)
            out += recs
        except Exception as e:                                         # noqa: BLE001
            errors.append(f"{j}: {type(e).__name__}: {e}")
    return pd.DataFrame(out), graphs, geoms, errors


def load_keys():
    """the Q-2 keys of the last passing Phase Q run ($LAD/Q_keys.parquet)."""
    return pd.read_parquet(LAD_KEYS)


def check_graphs(K, graphs, by="key"):
    """Q-2: within each group of column `by` ('key', or 'key_plain' = the SMILES-level key that also holds the
    constrained and free references of one molecule, D5) the same element composition and bond graph as its first
    member (job_id order)."""
    import networkx as nx

    def nm(a, b):
        return a["el"] == b["el"]
    bad, n_iso = [], 0
    for key, g in K.sort_values([by, "job_id"]).groupby(by, sort=True):
        j0, k0, f0, h0 = g.job_id.iloc[0], int(g.frag.iloc[0]), g.formula.iloc[0], g.wl_graph.iloc[0]
        G0 = graphs[(j0, k0)]
        for j, k, f, h in zip(g.job_id.iloc[1:], g.frag.iloc[1:], g.formula.iloc[1:], g.wl_graph.iloc[1:]):
            if f != f0:
                bad.append(dict(key=key, ref=f"{j0}/frag{k0}", row=f"{j}/frag{int(k)}", why=f"composition {f} != {f0}"))
                continue
            n_iso += 1
            if h != h0 or not nx.is_isomorphic(G0, graphs[(j, int(k))], node_match=nm):
                bad.append(dict(key=key, ref=f"{j0}/frag{k0}", row=f"{j}/frag{int(k)}", why="bond graph differs"))
    return dict(ok=not bad, by=by, n_keys=int(K[by].nunique()), n_compared=n_iso, n_fail=len(bad), fail=bad[:50])


def check_level(K, geoms):
    """D5: the label inputs (D1-HF copies inputs/<T>/orig/frag{k}_rel.inp) of one key have the same keywords, blocks
    (without %maxcore / %pal), charge and multiplicity; their coordinates equal the reference xyz."""
    HC = hf_common()
    sig, bad_xyz, missing = {}, [], []
    for j, k, key, q in zip(K.job_id, K.frag, K.key, K.charge):
        k = int(k)
        p = HC.INPUTS / j / "orig" / f"frag{k}_rel.inp"
        if not p.is_file():
            missing.append(str(p))
            continue
        d = HC.parse_inp(p.read_text())
        s = (tuple(d["bang"]), tuple(sorted((n, tuple(b)) for n, b in d["blocks"].items() if n not in ("maxcore", "pal"))),
             d["charge"], d["mult"])
        sig.setdefault(key, {}).setdefault(s, []).append(j)
        syms, xyz = geoms[(j, k)]
        at = d["atoms"]
        if (d["charge"] != int(q) or len(at) != len(syms) or any(a[0] != e for a, e in zip(at, syms))
                or _maxabs(np.array([a[2:] for a in at], float), xyz) > XYZ_TOL):
            bad_xyz.append(f"{j}/frag{k}")
    mixed = {key: {repr(s): v[:5] for s, v in dct.items()} for key, dct in sig.items() if len(dct) > 1}
    levels = sorted({repr(s[:2]) for dct in sig.values() for s in dct})
    return dict(ok=not bad_xyz and not mixed, n_checked=int(len(K) - len(missing)), n_missing=len(missing),
                missing=missing[:20], n_inp_xyz_or_charge_mismatch=len(bad_xyz), inp_xyz_or_charge_mismatch=bad_xyz[:30],
                n_keys_mixed_level=len(mixed), keys_mixed_level=dict(list(mixed.items())[:10]), levels=levels)


# ---------------------------------------------------------------- Q-3 / Q-4
def audit(K, energy, key="key"):
    """per (reaction, fragment) of K with an energy: e_eh, e_min_eh of its key, the source row of e_min (lowest energy,
    ties = smallest job_id) and dev_kcal = (e - e_min) x KCAL. energy = {(job_id, frag): Eh}."""
    A = K.copy()
    A["e_eh"] = [energy.get((j, int(f)), np.nan) for j, f in zip(A.job_id, A.frag)]
    A = A[np.isfinite(A.e_eh.astype(float))].copy()
    A["e_eh"] = A.e_eh.astype(float)
    first = A.sort_values([key, "e_eh", "job_id"], kind="mergesort").groupby(key, sort=False).head(1).set_index(key)
    A["e_min_eh"] = A[key].map(first.e_eh)
    A["src_job"] = A[key].map(first.job_id)
    A["src_rxn"] = A[key].map(first.rxn_id).astype(int)
    A["src_frag"] = A[key].map(first.frag).astype(int)
    A["dev_kcal"] = (A.e_eh - A.e_min_eh) * KCAL
    A["n_key"] = A.groupby(key)[key].transform("size")
    return A.sort_values(["job_id", "frag"]).reset_index(drop=True)


def choose_override(A, K, thr=THR_REPLACE, src="Q:label_min"):
    """Q-4: {(job_id, frag): replacement} for dev > thr (E_min and the source's coordinates)."""
    HC = hf_common()
    xyz = {(j, int(f)): p for j, f, p in zip(K.job_id, K.frag, K.xyz)}
    ov = {}
    for r in A[A.dev_kcal > thr].itertuples(index=False):
        ov[(r.job_id, int(r.frag))] = dict(
            e_eh=float(r.e_min_eh), e_orig_eh=float(r.e_eh), dev_kcal=float(r.dev_kcal), src=src, src_job=r.src_job,
            src_rxn=int(r.src_rxn), src_frag=int(r.src_frag), key=r.key, ref_xyz=xyz[(r.src_job, int(r.src_frag))],
            ref_inp=str(HC.INPUTS / r.src_job / "orig" / f"frag{int(r.src_frag)}_rel.inp"))
    return ov


def reassemble(rows, override, K=None, tag="cr"):
    """new label rows: unchanged rows verbatim, changed rows = parse_one_rxn with the replaced references (only
    CHANGED_KEYS taken from it); 'cr' dict on every row. Returns (rows, {job_id: parse_one_rxn record})."""
    keys = {} if K is None else {(j, int(f)): k for j, f, k in zip(K.job_id, K.frag, K.key)}
    out, recs = [], {}
    for row in rows:
        j = row["job_id"]
        ov = {k: override[(j, k)] for k in (1, 2) if (j, k) in override}
        new = dict(row)
        cr = dict(tag=tag, version=CODE_VERSION, changed=[f"frag{k}" for k in sorted(ov)],
                  key1=keys.get((j, 1)), key2=keys.get((j, 2)))
        if ov:
            rec = assemble(row, {k: o["e_eh"] for k, o in ov.items()})
            recs[j] = rec
            for kk in CHANGED_KEYS:
                new[kk] = rec[kk]
            for k, o in ov.items():
                cr.update({f"src{k}": o.get("src"), f"src{k}_job": o.get("src_job"), f"src{k}_rxn": o.get("src_rxn"),
                           f"src{k}_frag": o.get("src_frag"), f"ref{k}_xyz": o.get("ref_xyz"),
                           f"ref{k}_inp": o.get("ref_inp"), f"e_frag{k}_rel_orig_eh": row[REF_E[k]],
                           f"dref{k}_kcal": (row[REF_E[k]] - o["e_eh"]) * KCAL})
            cr["orig"] = {kk: row[kk] for kk in CHANGED_KEYS}
        new["cr"] = cr
        out.append(new)
    return out, recs


# ---------------------------------------------------------------- Q-5
def gates(orig, new, override, recs, qa):
    """Q-a .. Q-e on the written (JSON round-trip) rows `new` against the original rows."""
    G = {"Q-a": qa}
    if len(orig) != len(new) or any(o["job_id"] != n["job_id"] for o, n in zip(orig, new)):
        G["rows"] = dict(ok=False, why="row list differs")
        return G
    # Q-b: 6 EDA channels + c_ghost identical; every other field identical; recomputed channels within 1e-9
    nb, other_bad, rec_mx = 0, [], 0.0
    for o, n in zip(orig, new):
        nb += sum(n[k] != o[k] for k in EDA7)
        ch = CHANGED_KEYS if n["job_id"] in recs else ()
        if set(n) != set(o) | {"cr"} or any(_jd(n[k]) != _jd(o[k]) for k in o if k not in ch):
            other_bad.append(n["job_id"])
    by_job = {o["job_id"]: o for o in orig}
    for j, rec in recs.items():
        rec_mx = max([rec_mx] + [abs(rec[k] - by_job[j][k]) for k in EDA7])
    G["Q-b"] = dict(ok=nb == 0 and not other_bad and rec_mx <= TOL, n_channel_values_different=int(nb),
                    n_rows_other_fields_different=len(other_bad), rows=other_bad[:30],
                    recomputed_channels_max_abs=rec_mx, fields=list(EDA7))
    # Q-c: delta d (per fragment) and delta barrier = reference energy differences (every row)
    dd, db, e_bad = [], [], []
    for o, n in zip(orig, new):
        de = {k: (o[REF_E[k]] - n[REF_E[k]]) * KCAL for k in (1, 2)}
        for k in (1, 2):
            dd.append(abs((n[D_KEY[k]] - o[D_KEY[k]]) - de[k]))
            ov = override.get((o["job_id"], k))
            want = ov["e_eh"] if ov else o[REF_E[k]]
            if n[REF_E[k]] != want:
                e_bad.append(f"{o['job_id']}/frag{k}")
        db.append(abs((n[BARRIER_KEY] - o[BARRIER_KEY]) - (de[1] + de[2])))
    G["Q-c"] = dict(ok=max(dd + db) <= TOL and not e_bad, max_abs_d=max(dd), max_abs_barrier=max(db),
                    n_energy_not_the_override=len(e_bad), energy_not_the_override=e_bad[:30], tol=TOL)
    # Q-d: identity residual barrier - sum of the 9 channels unchanged
    rd = [abs((n[BARRIER_KEY] - sum(n[k] for k in SUM9)) - (o[BARRIER_KEY] - sum(o[k] for k in SUM9)))
          for o, n in zip(orig, new)]
    G["Q-d"] = dict(ok=max(rd) <= TOL, max_abs=max(rd), tol=TOL, sum_of=list(SUM9))
    # Q-e: new reference <= original reference (every row and fragment)
    ex = [f"{o['job_id']}/frag{k}" for o, n in zip(orig, new) for k in (1, 2) if not n[REF_E[k]] <= o[REF_E[k]]]
    G["Q-e"] = dict(ok=not ex, n_exceptions=len(ex), exceptions=ex[:30])
    return G


def gates_ok(G):
    return all(v.get("ok") for v in G.values())


# ---------------------------------------------------------------- outputs
def cr_table(orig, new, K, A, override):
    """results/labels_d1_<tag>.parquet rows."""
    info = {(j, int(f)): (k, c, d) for j, f, k, c, d in zip(A.job_id, A.frag, A.key, A.constrained, A.dev_kcal)}
    kk = {(j, int(f)): k for j, f, k in zip(K.job_id, K.frag, K.key)}
    rows = []
    for o, n in zip(orig, new):
        j = o["job_id"]
        d = dict(rxn_id=int(o["rxn_id"]), job_id=j, skeleton=o["skeleton"], dipole=o["dipole"], series=o.get("series"),
                 attach=o.get("attach"), d1_orig=o[D_KEY[1]], d1_new=n[D_KEY[1]], d2_orig=o[D_KEY[2]],
                 d2_new=n[D_KEY[2]], barrier_orig=o[BARRIER_KEY], barrier_new=n[BARRIER_KEY],
                 changed="+".join(f"frag{k}" for k in (1, 2) if (j, k) in override))
        d["delta_d1"], d["delta_d2"] = d["d1_new"] - d["d1_orig"], d["d2_new"] - d["d2_orig"]
        d["delta_barrier"] = d["barrier_new"] - d["barrier_orig"]
        for k in (1, 2):
            ov = override.get((j, k))
            key, cons, dev = info.get((j, k), (kk.get((j, k)), None, np.nan))
            d.update({f"key{k}": key, f"constrained{k}": cons, f"dev{k}_kcal": dev,
                      f"dref{k}_kcal": (o[REF_E[k]] - n[REF_E[k]]) * KCAL,
                      f"e_frag{k}_rel_orig_eh": o[REF_E[k]], f"e_frag{k}_rel_new_eh": n[REF_E[k]],
                      f"src{k}_rxn": ov["src_rxn"] if ov else None, f"src{k}_job": ov["src_job"] if ov else None,
                      f"src{k}": ov["src"] if ov else None})
        rows.append(d)
    T = pd.DataFrame(rows)
    for k in (1, 2):
        T[f"src{k}_rxn"] = pd.array([r[f"src{k}_rxn"] for r in rows], dtype="Int64")
    return T


def write_outputs(tag, text, table):
    p = out_paths(tag)
    L.write_atomic(p["json"], text)
    L.write_atomic(p["parquet"], lambda t: table.to_parquet(t, index=False))
    return {str(p["json"]): L.sha256(p["json"]), str(p["parquet"]): L.sha256(p["parquet"])}


# ---------------------------------------------------------------- Q-6
def pairs_dd(F, lab_by_job, targets=tuple(DP.TARGETS)):
    """dd_<t> = y_a - y_b of every pair of F (columns a_job, b_job) from {job_id: label row}; index = F.index."""
    out = {}
    for t in targets:
        col = DP.D1_TCOL[t]
        out[f"dd_{t}"] = np.array([lab_by_job[a][col] - lab_by_job[b][col] for a, b in zip(F.a_job, F.b_job)], float)
    return pd.DataFrame(out, index=F.index)


def check_pair_code():
    """the dd_* column map of the code that computed F_pairs.csv (d1_pairs.py, d1_anchor_pairs.py) = DP.D1_TCOL."""
    a = _load("d1_pairs", D1_PAIRS_PY).TARGETS
    b = _load("d1_anchor_pairs", D1_ANCHOR_PY).TARGETS
    return dict(ok=a == DP.D1_TCOL and b == DP.D1_TCOL, d1_pairs=a, d1_anchor_pairs=b,
                sha256={D1_PAIRS_PY.name: L.sha256(D1_PAIRS_PY), D1_ANCHOR_PY.name: L.sha256(D1_ANCHOR_PY)})


def write_pairs(tag, orig_by_job, new_by_job, F_path=F_PAIRS):
    """Q-6: F_pairs_<tag>.csv; returns (report, Fm (str frame), dd0, dd1). STOP-worthy failures -> report ok False."""
    rep = dict(source=str(F_path), source_sha256=L.sha256(F_path), code=check_pair_code())
    Fn = pd.read_csv(F_path, float_precision="round_trip")
    Fm = pd.read_csv(F_path, dtype=str, keep_default_na=False)
    miss = sorted((set(Fn.a_job) | set(Fn.b_job)) - set(orig_by_job))
    if miss:
        rep.update(ok=False, why=f"{len(miss)} pair reactions are not targets", missing=miss[:20])
        return rep, Fm, None, None
    dd0, dd1 = pairs_dd(Fn, orig_by_job), pairs_dd(Fn, new_by_job)
    proof = {t: _maxabs(dd0[f"dd_{t}"], Fn[f"dd_{t}"]) for t in DP.TARGETS}
    rep["proof_original_labels_reproduce_F_pairs"] = dict(max_abs=proof, tol=TOL, ok=max(proof.values()) <= TOL)
    still = {t: _maxabs(dd0[f"dd_{t}"], dd1[f"dd_{t}"]) for t in DP.TARGETS if t not in PAIR_T}
    rep["other_dd_unchanged_max_abs"] = still
    with open(F_path, newline="") as f:
        raw_lines = f.read().split("\n")
    with open(F_path, newline="") as f:
        lines = list(csv.reader(f))
    head, body = lines[0], lines[1:]
    ix = {c: i for i, c in enumerate(head)}
    if [r[ix["pair_id"]] for r in body] != Fn.pair_id.tolist():
        rep.update(ok=False, why="csv reader rows != pandas rows")
        return rep, Fm, dd0, dd1
    ddc = {ix[f"dd_{t}"] for t in PAIR_T}
    outl, mods = [], []
    for i, r in enumerate(body):
        r = list(r)
        a, b = r[ix["a_job"]], r[ix["b_job"]]
        mod = False
        for t in PAIR_T:
            col = DP.D1_TCOL[t]
            if new_by_job[a][col] != orig_by_job[a][col] or new_by_job[b][col] != orig_by_job[b][col]:
                r[ix[f"dd_{t}"]] = repr(float(dd1[f"dd_{t}"].iloc[i]))
                mod = True
        mods.append(mod)
        outl.append(r)
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(head)
    w.writerows(outl)
    text = buf.getvalue()
    out = out_paths(tag)["pairs"]
    # rows with no changed endpoint come out byte for byte; rewritten rows differ at most in the three dd fields
    new_lines = text.split("\n")
    n_diff, bad = 0, []
    for i, r in enumerate(outl):
        if new_lines[i + 1] != raw_lines[i + 1]:
            n_diff += 1
            same_rest = [x for c, x in enumerate(r) if c not in ddc] == [x for c, x in enumerate(body[i]) if c not in ddc]
            if not mods[i] or not same_rest:
                bad.append(r[ix["pair_id"]])
    rep["n_rows_rewritten"] = int(sum(mods))
    rep["n_lines_differing"] = int(n_diff)
    rep["n_lines_differing_outside_dd"] = len(bad)
    rep["lines_differing_outside_dd"] = bad[:20]
    rep["n_lines_total_same"] = len(new_lines) == len(raw_lines)
    rep["header_identical"] = new_lines[0] == raw_lines[0]
    L.write_atomic(out, text)
    Fc = pd.read_csv(out, float_precision="round_trip")
    Fo = pd.read_csv(F_path, float_precision="round_trip")
    same_cols = list(Fc.columns) == list(Fo.columns) and Fc.pair_id.tolist() == Fo.pair_id.tolist()
    oth = [c for c in Fo.columns if c not in {f"dd_{t}" for t in PAIR_T}]
    oth_same = all(Fc[c].equals(Fo[c]) for c in oth) if same_cols else False
    rb = {t: _maxabs(Fc[f"dd_{t}"], dd1[f"dd_{t}"]) for t in PAIR_T} if same_cols else {}
    S = pd.read_csv(S_FOLDS, usecols=["pair_id"])
    rep.update(readback=dict(columns_rows_order_same=same_cols, other_columns_identical=oth_same, dd_vs_new_max_abs=rb),
               s_folds=dict(path=str(S_FOLDS), sha256=L.sha256(S_FOLDS), same_pair_set=set(S.pair_id) == set(Fo.pair_id)),
               output=str(out), output_sha256=L.sha256(out))
    rep["ok"] = bool(rep["proof_original_labels_reproduce_F_pairs"]["ok"] and rep["code"]["ok"]
                     and all(v == 0.0 for v in still.values()) and not bad and rep["n_lines_total_same"]
                     and rep["header_identical"] and same_cols and oth_same and all(v <= 1e-12 for v in rb.values())
                     and rep["s_folds"]["same_pair_set"])
    return rep, Fm, dd0, dd1


def q6_report(Fm, dd0, dd1, new_by_job, azd_path=RES_AZD):
    """Q-6 report; the azide/D para table goes to azd_path (default = Phase Q's table; another tag must pass its own)."""
    rep = {}
    pmain = Fm.in_P_main == "True"
    for t in PAIR_T:
        d = dd1[f"dd_{t}"].to_numpy() - dd0[f"dd_{t}"].to_numpy()
        ch = np.abs(d) > TOL
        sub = Fm[ch]
        rep[t] = dict(n_changed=int(ch.sum()), n_abs_gt={f"{x:g}": int((np.abs(d) > x).sum()) for x in THRS},
                      in_P_main=int((ch & pmain).sum()), by_ref_kind=_vc(sub.ref_kind), by_skeleton=_vc(sub.skeleton),
                      by_dipole=_vc(sub.dipole), by_pos=_vc(sub.pos), by_pos_P_main=_vc(Fm[ch & pmain].pos),
                      abs_change=_q(np.abs(d[ch])), mean_signed_change=float(d[ch].mean()) if ch.any() else 0.0)
    m = ((Fm.dipole == "azide") & (Fm.skeleton == "D") & (Fm.pos == "p") & (Fm.ref_kind == "reference")).to_numpy()
    tab = pd.DataFrame(dict(pair_id=Fm.pair_id[m], fam=Fm.fam[m], Xb=Fm.Xb[m],
                            sigma=pd.to_numeric(Fm.sigma[m], errors="coerce"), a_job=Fm.a_job[m], b_job=Fm.b_job[m],
                            in_P_main=Fm.in_P_main[m],
                            dd_barrier_orig=dd0.dd_barrier[m], dd_barrier_new=dd1.dd_barrier[m],
                            dd_d2_orig=dd0.dd_d2[m], dd_d2_new=dd1.dd_d2[m],
                            dd_d1_orig=dd0.dd_d1[m], dd_d1_new=dd1.dd_d1[m]))
    tab = tab.sort_values(["fam", "sigma", "Xb"]).reset_index(drop=True)
    L.write_atomic(azd_path, lambda p: tab.to_csv(p, index=False))
    rep["azide_D_para"] = dict(csv=str(azd_path), n=len(tab),
                               n_changed_dd_barrier=int((np.abs(tab.dd_barrier_new - tab.dd_barrier_orig) > TOL).sum()),
                               n_changed_dd_d2=int((np.abs(tab.dd_d2_new - tab.dd_d2_orig) > TOL).sum()),
                               rows=tab.round(6).to_dict(orient="records"))
    hyg_new = [bool(DP.hygiene_mask(new_by_job[a][D_KEY[1]], new_by_job[a][D_KEY[2]])
                    and DP.hygiene_mask(new_by_job[b][D_KEY[1]], new_by_job[b][D_KEY[2]])) for a, b in zip(Fm.a_job, Fm.b_job)]
    flip = np.array(hyg_new) != (Fm.hyg == "True").to_numpy()
    rep["hygiene_would_flip"] = dict(n=int(flip.sum()), by_ref_kind=_vc(Fm.ref_kind[flip]), pairs=Fm.pair_id[flip].tolist()[:30],
                                     note="pair list and flags unchanged (spec Q-6); counted only")
    return rep


# ---------------------------------------------------------------- reports
def dev_counts(A):
    R = A.groupby("job_id").agg(dev=("dev_kcal", "max"), skeleton=("skeleton", "first"), dipole=("dipole", "first"))
    out = {}
    for t in THRS:
        m, rm = R.dev > t, A.dev_kcal > t
        out[f"{t:g}"] = dict(reactions=int(m.sum()), by_skeleton=_vc(R.skeleton[m]), by_dipole=_vc(R.dipole[m]),
                             rows_by_kind=_vc(A.kind[rm]), rows_constrained=int((rm & A.constrained).sum()),
                             dipolarophile_rows_cyclooctyne=int((rm & (A.kind == "dipolarophile")
                                                                 & A.skeleton.isin(CYCLOOCTYNE)).sum()),
                             molecules=int(A.key[rm].nunique()),
                             molecules_by_kind={k: int(v) for k, v in A[rm].groupby("kind").key.nunique().items()})
    out["dev_of_rows_above_0.05"] = _q(A.dev_kcal[A.dev_kcal > 0.05])
    return out


def audit_table(K, A):
    g = A.groupby("key", sort=True)
    first = A.sort_values(["key", "e_eh", "job_id"], kind="mergesort").groupby("key").head(1).set_index("key")
    T = pd.DataFrame(dict(kind=g.kind.first(), smiles_key=g.smiles_key.first(), constrained=g.constrained.first(),
                          fixed_state=g.fixed_state.first(), n_rxn=g.size(), e_min_eh=g.e_eh.min(), e_max_eh=g.e_eh.max()))
    T["range_kcal"] = (T.e_max_eh - T.e_min_eh) * KCAL
    T["src_job"], T["src_rxn"] = first.job_id, first.rxn_id
    for t in THRS:
        T[f"n_dev_gt_{t:g}"] = A.assign(_f=A.dev_kcal > t).groupby("key")["_f"].sum().astype(int)
    T["skeletons"] = g.skeleton.agg(lambda s: ";".join(sorted(set(s))))
    T["dipoles"] = g.dipole.agg(lambda s: ";".join(sorted(set(s))))
    T["formula"] = g.formula.first()
    T["n_stereo3d"] = g.stereo3d.agg(lambda s: len({x for x in s if x and not x.startswith("error")}))
    return T


def d1_job_records(job_ids, names=("ts_result.json", "ref_result.json")):
    """{job_id: {name: dict}} from the ledger summary tars (sha256 = bundles.tsv) or the UBAI job folders, as
    analysis/d1_g2/prep_x.py reads ts_result.json."""
    C = g2c()
    st = pd.read_csv(DP.D1_STATUS, dtype=str, keep_default_na=False).set_index("job_id")
    bt = pd.read_csv(C.LEDGER / "bundles.tsv", sep="\t", dtype=str).set_index("bundle")["summary_sha256"]
    out = {j: {} for j in job_ids}
    info = dict(d1_status_sha256_ok=L.sha256(DP.D1_STATUS) == DP.D1_STATUS_SHA, bad_tars=[], n_ubai=0, tars={})
    for b, g in st.loc[list(job_ids)].groupby("harvested"):
        want = set(g.index)
        if b == "":
            info["n_ubai"] += len(want)
            for j in want:
                for n in names:
                    p = DP.UBAI_JOBS / j / n
                    if p.is_file():
                        out[j][n] = json.loads(p.read_text())
            continue
        tar = C.LEDGER / "summary" / f"{b}.small.tar"
        sha = L.sha256(tar) if tar.is_file() else None
        info["tars"][tar.name] = sha
        if sha is None or sha != bt.get(b):
            info["bad_tars"].append(b)
            continue
        with tarfile.open(tar, "r:") as T:
            for m in T:
                p = (m.name[2:] if m.name.startswith("./") else m.name).split("/")
                if m.isfile() and len(p) == 3 and p[0] == "jobs" and p[1] in want and p[2] in names:
                    out[p[1]][p[2]] = json.loads(T.extractfile(m).read())
    info["n_with"] = {n: sum(n in v for v in out.values()) for n in names}
    return out, info


def port_check(rows, recs):
    """label ref_port vs ref_result.json port (assemble.py copy); key-relevant = constrained, use_alt, frozen quads."""
    same, rel_bad, missing = 0, [], 0
    for row in rows:
        rr = recs.get(row["job_id"], {}).get("ref_result.json")
        if not rr:
            missing += 1
            continue
        p = rr.get("port") or rr.get("port_diagnostic") or {}
        q = row.get("ref_port") or {}
        same += _jd(p) == _jd(q)

        def kr(x):
            return (bool(x.get("constrained")), bool(x.get("use_alt")),
                    [list(s["quad"]) for s in x.get("scans") or [] if s.get("rotatable") is False])
        if kr(p) != kr(q) or int(rr.get("alt_used", -1)) != int(row.get("alt_used", -2)):
            rel_bad.append(row["job_id"])
    return dict(ok=not rel_bad, n_compared=len(rows) - missing, n_missing=missing, n_identical=int(same),
                n_key_relevant_mismatch=len(rel_bad), key_relevant_mismatch=rel_bad[:30])


def g_energies(rows, recs):
    """{(job_id, frag): G_eh} of the used references (autodE level) + consistency of the label's dG."""
    G, cons = {}, []
    for row in rows:
        ts = recs.get(row["job_id"], {}).get("ts_result.json")
        if not ts:
            continue
        kd = ts.get("reactant_index_dipole")
        R = ts.get("reactants") or []
        if kd not in (0, 1) or len(R) != 2:
            continue

        def gr(r):
            return r["e_sp_eh"] + r["g_cont_eh"] if r.get("g_cont_eh") is not None and r.get("e_sp_eh") is not None else None
        port = row.get("ref_port") or {}
        alt = (port.get("alt_dft") or {}).get("G_eh")
        g_orig = gr(R[kd])
        if alt is not None and g_orig is not None and port.get("dG_alt_minus_orig_kcal") is not None:
            cons.append(abs((alt - g_orig) * KCAL - port["dG_alt_minus_orig_kcal"]))
        for k in (1, 2):
            if row[f"role{k}"] == "dipole":
                g = alt if port.get("use_alt") else g_orig
            else:
                g = gr(R[1 - kd])
            if g is not None:
                G[(row["job_id"], k)] = float(g)
    return G, dict(n=len(G), dG_label_vs_ts_result_max_abs=max(cons) if cons else None)


def g_vs_e(K, Edft, Gd):
    have = [(j, int(f)) for j, f in zip(K.job_id, K.frag) if (j, int(f)) in Gd and (j, int(f)) in Edft]
    Ks = K.set_index(["job_id", "frag"]).loc[have].reset_index()
    Ae, Ag = audit(Ks, Edft), audit(Ks, Gd)
    se = Ae[Ae.n_key >= 2].groupby("key").src_job.first()
    sg = Ag[Ag.n_key >= 2].groupby("key").src_job.first()
    sgf = Ag[Ag.n_key >= 2].groupby("key").src_frag.first()
    devE = {(j, int(f)): d for j, f, d in zip(Ae.job_id, Ae.frag, Ae.dev_kcal)}
    kind = Ae.groupby("key").kind.first()
    cons = Ae.groupby("key").constrained.first()
    diff = [k for k in se.index if se[k] != sg[k]]
    mat = [k for k in diff if devE[(sg[k], int(sgf[k]))] > THR_REPLACE]
    return dict(n_rows_with_G=len(have), n_keys_compared=int(len(se)), n_argmin_differ=len(diff),
                n_argmin_differ_material=len(mat), differ_by_kind=_vc(kind[diff]) if diff else {},
                material_by_kind=_vc(kind[mat]) if mat else {},
                material_constrained=int(cons[mat].sum()) if mat else 0,
                material_E_dev_of_G_choice=_q([devE[(sg[k], int(sgf[k]))] for k in mat]),
                examples=[dict(key=k, E_src=se[k], G_src=sg[k], E_dev_of_G_src=devE[(sg[k], int(sgf[k]))]) for k in mat[:15]])


def hf_precheck(K, Edft, pall_jobs):
    H = pd.read_parquet(L.D1HF_LABELS, columns=["job_id", "geom", "e_frag1_rel_eh", "e_frag2_rel_eh"])
    H = H[H.geom == "dft"]
    Eh = {}
    for j, e1, e2 in zip(H.job_id, H.e_frag1_rel_eh, H.e_frag2_rel_eh):
        Eh[(j, 1)], Eh[(j, 2)] = float(e1), float(e2)
    out = dict(source=str(L.D1HF_LABELS), sha256=L.sha256(L.D1HF_LABELS), n_hf_reactions=int(len(H)),
               reviewer=dict(P_all_reactions=7410, flagged=523, dipolarophile_rows=313, dipole_rows=221,
                             cyclooctyne_share_of_dipolarophile=0.82, median_dev=3.3))
    for mode, key in (("same_keys", "key"), ("smiles_only_keys", "key_plain")):
        Ad, Ah = audit(K, Edft, key=key), audit(K, Eh, key=key)
        fd = Ad.groupby("job_id").dev_kcal.max() > HF_FLAG
        fh = Ah.groupby("job_id").dev_kcal.max() > HF_FLAG
        res = {}
        for pop, jobs in (("all_targets", None), ("P_all_reactions", pall_jobs)):
            idx = fd.index.intersection(fh.index)
            if jobs is not None:
                idx = idx.intersection(pd.Index(sorted(jobs)))
            a, b = fd[idx], fh[idx]
            rh = Ah[Ah.job_id.isin(idx) & (Ah.dev_kcal > HF_FLAG)]
            mx = Ah[Ah.job_id.isin(idx)].groupby("job_id").dev_kcal.max()
            res[pop] = dict(n=int(len(idx)), dft_flag=int(a.sum()), hf_flag=int(b.sum()), both=int((a & b).sum()),
                            dft_only=int((a & ~b).sum()), hf_only=int((~a & b).sum()),
                            agreement=float((a == b).mean()) if len(idx) else None,
                            hf_flag_rows_by_kind=_vc(rh.kind),
                            hf_cyclooctyne_share_of_dipolarophile_rows=(
                                float(rh[rh.kind == "dipolarophile"].skeleton.isin(CYCLOOCTYNE).mean())
                                if (rh.kind == "dipolarophile").any() else None),
                            hf_median_dev_flagged_reactions=float(mx[mx > HF_FLAG].median()) if (mx > HF_FLAG).any() else None,
                            hf_median_dev_flagged_rows=float(rh.dev_kcal.median()) if len(rh) else None)
        out[mode] = res
    return out, Eh


def dpp_report(Kd, errs, A):
    """D2: the D'' rows' keys vs the target keys (Kd has e_eh = the D'' row's own reference energy)."""
    if Kd is None or not len(Kd):
        return dict(n_rows=0, n_key_errors=len(errs), errors=errs[:10], decision=DECISIONS["D2_dpp"])
    emin = A.groupby("key").e_min_eh.first()
    shared = Kd.key.isin(emin.index).to_numpy()
    em = emin.reindex(Kd.key).to_numpy()
    below = shared & ((em - Kd.e_eh.to_numpy()) * KCAL > THR_REPLACE)
    return {"n_rows": int(Kd.job_id.nunique()), "n_fragment_rows": int(len(Kd)), "n_keys": int(Kd.key.nunique()),
            "n_keys_shared_with_targets": int(Kd.key[shared].nunique()),
            "n_fragment_rows_in_shared_keys": int(shared.sum()), "shared_by_kind": _vc(Kd.kind[shared]),
            "n_fragment_rows_below_target_E_min_by_gt_0.05": int(below.sum()),
            "n_key_errors": len(errs), "errors": errs[:10], "decision": DECISIONS["D2_dpp"]}


def stereo3d_report(K, A, override):
    s = A[A.stereo3d.notna()]
    err = s.stereo3d.str.startswith("error")
    nk = s[~err].groupby("key").stereo3d.nunique()
    multi = nk[nk > 1]
    sig = {(j, int(f)): x for j, f, x in zip(A.job_id, A.frag, A.stereo3d)}
    cross = []
    for (j, k), o in override.items():
        a, b = sig.get((j, k)) or "", sig.get((o["src_job"], o["src_frag"])) or ""
        if a and b and not a.startswith("error") and not b.startswith("error") and a != b:
            cross.append(f"{j}/frag{k}")
    kind = s.groupby("key").kind.first()
    cons = s.groupby("key").constrained.first()
    return dict(n_errors=int(err.sum()), n_keys_multi_signature=int(len(multi)),
                multi_by_kind=_vc(kind[multi.index]) if len(multi) else {},
                multi_constrained=int(cons[multi.index].sum()) if len(multi) else 0,
                n_replacements_across_signatures=len(cross), replacements_across_signatures=cross[:30],
                examples=[dict(key=k, n=int(v)) for k, v in multi.head(15).items()],
                note=DECISIONS["D11_stereo3d"])


# ---------------------------------------------------------------- run
def inputs_fingerprint():
    C = g2c()
    files = [DP.D1_LABELS, X_TARGETS, L.D1HF_LABELS, F_PAIRS, S_FOLDS, DP.D1_STATUS, Path(__file__).resolve(),
             HERE / "lad_common.py", s3_dir() / "stage3_parse.py", s3_dir() / "_common.py", C.PILOT / "stereo_utils.py",
             DP.ESP_DIR / "fragmenter.py", D1_PAIRS_PY, D1_ANCHOR_PY, L.HF_DIR / "hf_common.py"]
    return {str(p): L.sha256(p) for p in files}


def q_outputs():
    """every Phase Q output file (tag cr) that a reader could take for valid."""
    p = out_paths("cr")
    return [p["json"], p["parquet"], p["pairs"], LAD_KEYS, RES_AUDIT, RES_AZD]


def _invalidate():
    """rename the existing Phase Q outputs to <name>.stale (D14); returns the renamed paths."""
    moved = []
    for q in q_outputs():
        if q.is_file():
            q = L.assert_path(q)
            os.replace(q, q.with_name(q.name + ".stale"))
            moved.append(str(q))
    return moved


def _fail(rep, msg):
    rep.update(stop=True, all_pass=False, stop_reason=msg, finished=time.strftime("%F %T"))
    rep["invalidated_on_stop"] = _invalidate()
    L.write_json(RES_REPORT, rep)
    L.stop(msg)


def _log(t0, msg):
    print(f"[refs_q {time.time() - t0:8.1f}s] {msg}", flush=True)


def cmd_run(force=False):
    t0 = time.time()
    fp = inputs_fingerprint()
    if not force and RES_REPORT.is_file():
        try:
            o = json.loads(RES_REPORT.read_text())
        except ValueError:
            o = {}
        if (o.get("all_pass") and o.get("fingerprint") == fp
                and all(Path(p).is_file() and L.sha256(p) == s for p, s in (o.get("outputs") or {}).items())):
            print(f"Phase Q up to date ({RES_REPORT}); --force to recompute", flush=True)
            return 0
    rep = dict(phase="Q", code_version=CODE_VERSION, started=time.strftime("%F %T"), git_head=L.git("rev-parse", "HEAD").strip(),
               env=env_info(), fingerprint=fp, found=dict(FOUND), decisions=DECISIONS, stop=False, all_pass=False)
    # D14: a recompute never leaves an earlier pass (report or outputs) that looks valid
    rep["invalidated_at_start"] = _invalidate()
    rj = out_paths("cr")["rejected"]
    if rj.is_file():
        L.assert_path(rj).unlink()
    L.write_json(RES_REPORT, dict(rep, running=True))
    try:
        return _run(rep, t0)
    except SystemExit:
        if not rep.get("stop"):            # an exit 3 inside a helper (L.stop / load_d1_labels): record it as a STOP
            rep.update(stop=True, all_pass=False, finished=time.strftime("%F %T"),
                       stop_reason="exit inside a helper before a Q gate (message on stderr of the job log)")
            rep["invalidated_on_stop"] = _invalidate()
            L.write_json(RES_REPORT, rep)
        raise
    except Exception as e:                                             # noqa: BLE001
        traceback.print_exc()
        _fail(rep, f"unexpected error {type(e).__name__}: {e} (traceback in the job log)")


def _run(rep, t0):
    s3 = stage3()
    rep["found"]["assembly_sha256"] = {"stage3_parse.py": L.sha256(Path(s3.__file__)),
                                       "_common.py": L.sha256(Path(sys.modules["_common"].__file__))}
    rep["found"]["assembly_same_as_repo"] = (L.sha256(Path(s3.__file__))
                                             == L.sha256(L.REPO / "label_true" / "scripts" / "stage3_parse.py"))
    rows = L.targets()
    lab_all = DP.load_d1_labels()
    dpp = sorted((d for d in lab_all if d["skeleton"] in DP.DPP), key=lambda d: d["job_id"])
    need = set(SPE_KEY.values()) | set(META_KEYS) | set(CHANGED_KEYS) | set(EDA7) | set(SUM9) | {"eda_raw_eh", "ref_port"}
    miss = sorted({k for r in rows for k in need if k not in r})
    if miss:
        _fail(rep, f"label keys not found in labels_d1.json: {miss} (spec 16 Q)")
    roles = Counter((r["role1"], r["role2"]) for r in rows)
    rep["found"]["roles_observed"] = {f"{a}/{b}": n for (a, b), n in roles.items()}
    if set(roles) != {("dipole", "dipolarophile")}:
        rep["found"]["roles_note"] = "not every row has role1 = dipole; keys use role{k} per fragment"
    rep["population"] = dict(n_targets=len(rows), n_dpp=len(dpp), n_labels=len(lab_all))
    X = pd.read_csv(X_TARGETS, dtype=str, keep_default_na=False).set_index("job_id")
    if sorted(X.index) != [r["job_id"] for r in rows]:
        _fail(rep, f"{X_TARGETS} job list != the 9,266 targets")
    smi, mapped = X.rxn_smiles.to_dict(), X.mapped_smiles.to_dict()
    rep["population"]["rxn_smiles_label_vs_X_targets_differ"] = int(sum(r["rxn_smiles"] != smi[r["job_id"]] for r in rows))
    _log(t0, f"{len(rows)} targets, {len(dpp)} D''; molecule keys")

    # Q-2 keys and checks
    K, graphs, geoms, errs = molecule_keys(rows, smi, mapped)
    if errs:
        rep["keys"] = dict(n_errors=len(errs), errors=errs[:50])
        _fail(rep, f"Q-2: molecule keys could not be built for {len(errs)} reactions, first: {errs[0]}")
    gk = check_graphs(K, graphs)
    gks = check_graphs(K, graphs, by="key_plain")       # D5: constrained and free references of one SMILES too
    _log(t0, f"keys {K.key.nunique()}, graph check ok={gk['ok']}, SMILES-level ok={gks['ok']}")
    lv = check_level(K, geoms)
    _log(t0, f"level check ok={lv['ok']}")
    recs, rinfo = d1_job_records([r["job_id"] for r in rows])
    pc = port_check(rows, recs)
    _log(t0, f"ledger records {rinfo['n_with']}, port check ok={pc['ok']}, bad tars {rinfo['bad_tars']}")
    tagmix = K[K.constrained & (K.fixed_tag_d1 != "")]
    rep["keys"] = dict(
        n_keys=int(K.key.nunique()), n_keys_by_kind=_vc(K.groupby("key").kind.first()),
        n_keys_constrained=int(K.groupby("key").constrained.first().sum()),
        n_keys_multi_row=int((K.groupby("key").size() >= 2).sum()),
        n_rows_constrained=int(K.constrained.sum()),
        n_smiles_with_constrained_and_free=int(K[K.kind == "dipole"].groupby("smiles_key").constrained.nunique().gt(1).sum()),
        fixed_tag_s=int(K.fixed_state.str.endswith(":s").sum()),
        d1_tag_vs_quad_recomputed_differ=int((tagmix.fixed_tag_d1 != tagmix.fixed_tag_quad).sum()),
        keys_with_mixed_d1_tags=int(tagmix.groupby("key").fixed_tag_d1.nunique().gt(1).sum()),
        ring_check=_vc(K[K.kind == "dipole"].ring_check),
        graph_check=gk, graph_check_smiles_level=gks, level_check=lv, ledger=rinfo, port_check=pc)
    if not gk["ok"]:
        _fail(rep, f"Q-2: element composition / bond graph differ within a key ({gk['n_fail']} rows), first {gk['fail'][0]}")
    if not gks["ok"]:
        _fail(rep, f"Q-2: element composition / bond graph differ within one SMILES (key_plain; {gks['n_fail']} rows), "
                   f"first {gks['fail'][0]}")
    if rinfo["bad_tars"] or not rinfo["d1_status_sha256_ok"]:
        _fail(rep, f"D13: ledger summary tars with a sha256 != bundles.tsv (or missing): {rinfo['bad_tars'][:5]}; "
                   f"d1_status.csv sha256 ok = {rinfo['d1_status_sha256_ok']}")
    if not lv["ok"]:
        _fail(rep, f"Q-2 (D5): label inputs within a key differ in level / charge or their coordinates differ from the "
                   f"reference xyz ({lv['n_keys_mixed_level']} keys, {lv['n_inp_xyz_or_charge_mismatch']} rows)")
    if not pc["ok"]:
        _fail(rep, f"Q-2: label ref_port differs from ref_result.json in a key-relevant field for "
                   f"{pc['n_key_relevant_mismatch']} rows, first {pc['key_relevant_mismatch'][0]}")

    # Q-3 audit, Q-4 override
    Edft = {(r["job_id"], k): float(r[REF_E[k]]) for r in rows for k in (1, 2)}
    A = audit(K, Edft)
    ov = choose_override(A, K)
    rep["audit"] = dev_counts(A)
    _log(t0, f"audit: {len(ov)} fragment references to replace")

    # Q-a, re-assembly, Q-b .. Q-e (on the JSON round trip of what is written)
    qa = gate_qa(rows)
    _log(t0, f"Q-a ok={qa['ok']} max {qa['max_abs']}")
    if not qa["ok"]:                       # D14: before reassemble (which would raise on a row that cannot be assembled)
        rep["gates"] = {"Q-a": qa}
        _fail(rep, f"Q-5 gate failure: Q-a ({qa['n_fail']} rows above {TOL}, {qa['n_errors']} rows not assembled)")
    new_rows, rrecs = reassemble(rows, ov, K, "cr")
    text, new_rt = json_roundtrip(new_rows)
    G = gates(rows, new_rt, ov, rrecs, qa)
    rep["gates"] = G
    _log(t0, "gates " + " ".join(f"{k}={v.get('ok')}" for k, v in G.items()))
    if not gates_ok(G):
        L.write_atomic(out_paths("cr")["rejected"], text)
        _fail(rep, "Q-5 gate failure: " + ", ".join(k for k, v in G.items() if not v.get("ok")))
    T = cr_table(rows, new_rt, K, A, ov)
    outputs = write_outputs("cr", text, T)
    chg = T[T.changed != ""]
    cmap = {(j, int(f)): bool(c) for j, f, c in zip(K.job_id, K.frag, K.constrained)}
    rep["reassembly"] = dict(
        n_rows_changed=int(len(chg)), by_changed=_vc(chg.changed), by_skeleton=_vc(chg.skeleton), by_dipole=_vc(chg.dipole),
        n_fragments_changed=len(ov), n_fragments_changed_by_kind=_vc([k for (j, f), o in ov.items()
                                                                      for k in [o["key"].split(":", 1)[0]]]),
        n_constrained_fragments_changed=int(sum(cmap[jk] for jk in ov)),
        delta_d1=_q(chg.delta_d1[chg.delta_d1 != 0]), delta_d2=_q(chg.delta_d2[chg.delta_d2 != 0]),
        delta_barrier=_q(chg.delta_barrier), n_source_molecules=int(len({o["key"] for o in ov.values()})))

    # Q-3 tables
    Q = audit_table(K, A)
    Fm0 = pd.read_csv(F_PAIRS, dtype=str, keep_default_na=False)
    ref = Fm0[(Fm0.ref_kind == "reference") & (Fm0.in_P_all == "True")]
    pall = set(ref.a_job) | set(ref.b_job)
    Hrep, Eh = hf_precheck(K, Edft, pall_jobs=pall)
    Hrep["n_P_all_reactions"] = len(pall)
    rep["hf_precheck"] = Hrep
    Ah = audit(K, Eh)
    Q["hf_range_kcal"] = (Ah.groupby("key").e_eh.max() - Ah.groupby("key").e_eh.min()) * KCAL
    Q["hf_src_rxn"] = Ah.groupby("key").src_rxn.first()
    Gd, ginfo = g_energies(rows, recs)
    rep["g_vs_e"] = dict(g_vs_e(K, Edft, Gd), **ginfo) if Gd else dict(available=False, **ginfo)
    if Gd:
        Ag = audit(K.set_index(["job_id", "frag"]).loc[[k for k in zip(K.job_id, K.frag) if k in Gd]].reset_index(), Gd)
        Q["g_src_rxn"] = Ag.groupby("key").src_rxn.first()
    rep["stereo3d"] = stereo3d_report(K, A, ov)
    _log(t0, "D'' keys")
    Kd, _, _, derr = molecule_keys(dpp, None, None, stereo3d=False)
    if len(Kd):
        dby = {r["job_id"]: r for r in dpp}
        Kd["e_eh"] = [float(dby[j][REF_E[int(f)]]) for j, f in zip(Kd.job_id, Kd.frag)]
        Q["n_dpp_rows"] = Kd.groupby("key").size().reindex(Q.index).fillna(0).astype(int)
    rep["dpp"] = dpp_report(Kd, derr, A)
    Q = Q.reset_index()
    L.write_atomic(RES_AUDIT, lambda p: Q.to_csv(p, index=False))
    hf_e = {(j, int(f)): e for j, f, e in zip(Ah.job_id, Ah.frag, Ah.e_eh)}
    hf_d = {(j, int(f)): d for j, f, d in zip(Ah.job_id, Ah.frag, Ah.dev_kcal)}
    keys_out = A.assign(replaced=[(j, int(f)) in ov for j, f in zip(A.job_id, A.frag)],
                        e_hf_eh=[hf_e.get((j, int(f)), np.nan) for j, f in zip(A.job_id, A.frag)],
                        dev_hf_kcal=[hf_d.get((j, int(f)), np.nan) for j, f in zip(A.job_id, A.frag)])
    L.write_atomic(LAD_KEYS, lambda p: keys_out.to_parquet(p, index=False))
    outputs.update({str(RES_AUDIT): L.sha256(RES_AUDIT), str(LAD_KEYS): L.sha256(LAD_KEYS)})

    # Q-6 pair labels
    _log(t0, "Q-6 pairs")
    orig_by_job = {r["job_id"]: r for r in rows}
    new_by_job = {r["job_id"]: r for r in new_rt}
    prep, Fm, dd0, dd1 = write_pairs("cr", orig_by_job, new_by_job)
    rep["Q6"] = prep
    if not prep.get("ok"):
        _fail(rep, f"Q-6: F_pairs_cr.csv could not be proven ({prep.get('why') or 'see Q_report.json Q6'})")
    rep["Q6"].update(q6_report(Fm, dd0, dd1, new_by_job))
    outputs.update({prep["output"]: prep["output_sha256"], str(RES_AZD): L.sha256(RES_AZD)})
    rep.update(outputs=outputs, all_pass=True, stop=False, finished=time.strftime("%F %T"),
               wall_s=round(time.time() - t0, 1))
    L.write_json(RES_REPORT, rep)
    for q in q_outputs():                  # D14: the pass supersedes the .stale copies
        s = q.with_name(q.name + ".stale")
        if s.is_file():
            L.assert_path(s).unlink()
    _log(t0, f"Phase Q pass: {rep['reassembly']['n_rows_changed']} reactions changed; "
             f"pairs dd_barrier changed {rep['Q6']['barrier']['n_changed']}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="D1LADDER Phase Q (refs_q.py)")
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--force", action="store_true", help="recompute even if Q_report.json is up to date")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        return cmd_run(a.force)
    return 2


if __name__ == "__main__":
    sys.exit(main())
