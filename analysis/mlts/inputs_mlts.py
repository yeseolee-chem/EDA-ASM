#!/usr/bin/env python3
r"""inputs_mlts.py — DEPLOY-V2 (MLTS) Phase H (docs/specs/DEPLOY_V2_MLTS.md 7; C-5 = H-1): model inputs on TS
structures of the A-2 layout, in the LADDER rung-table format, + D3_int. Compute nodes only (jobs/mlt.slurm or
jobs/sbx.sh); every command re-executes itself in the environment it needs (ENVS below), so any MLTENV can start it.

Sources (src_cfg)
  mlts    A-2 folders $MLT/ade/jobs/<job_id>/{ts.xyz, ts_result.json (formed_pairs_ts, A_idx, B_idx in autodE order,
          checks), ref_dipole.xyz, ref_dipolarophile.xyz (MLTS-refined references), ref_result.json, stereo_result.json,
          .done_ts / .done_stereo / .done_ref, ts.hess, inputs tar} copied out of the node-local work folder by
          mlts_autode.py (spec 6 A-2). Targets = the 9,266 D1 targets (analysis/d1_g2/results/G_status.csv), or the
          rxn_id column of --ids <csv>; --root <dir> moves every mlts path below <dir> (A-3 pilot). Outputs under $MLT:
          final/<rxn_id>/, inputs/{meta_mlts/, prof_empty/, X182_mlts.parquet, d3_mlts.parquet, mlts.parquet,
          status_mlts.csv}, feat/mlts/, fu2_mlts/{atoms,tokens}/, hf/mlts/; reports results/H_*_mlts.json,
          results/fu2_mlts/ (FU2 Q_* / T_* reports).
  label   (H-1 only) the label TS + label references of $LAD/dft_final/<rxn_id>/ (the LADDER dft-rung structures, label
          order) written into the same A-2 layout by `label_src` ($MLT/h1/ade/<job_id>/: A_idx = the LADDER
          input_meta_d1_g2.csv label partition, formed_pairs_ts / flag_async / flag_foreign_bond / alt_used = the label
          record), then through exactly the mlts code path; everything under $MLT/h1.

CLI (exit 0 done, 2 not ready / unfinished (resubmit the same command), 3 STOP). i n = slice i of n ('@I' = array id).
  label_src [N]               label   pick the H-1 sample (N = 50, seeded, N/5 per dipole among dft-rung usable
                                      reactions) -> $MLT/h1/targets.csv + A-2 folders                      [reactot]
  smoke_src [N] --root R      mlts    test fixture: N small G2-ok targets, $G2/jobs/<T> copied to R/ade/jobs/<T>
                                      (xTB TSs as stand-in MLTS TSs) + R/ids.csv                           [reactot]
  finalize <src> <i> <n> [c]  both    deploy/finalize_lf.finalize_lf (xTB Freq at ts.xyz, autodE order, own
                                      formed_pairs_ts / A_idx) -> final/<rxn_id>/ (G1 folder format); c = ORCA cores
                                      per reaction (default 1 = the LADDER G0 dft_final header)               [d1]
  meta <src>                  both    deploy/meta_lf.build -> labels.json (DFT keys NaN) + input_meta.csv (A_idx)  [d1]
  x182 <src> <i> <n>          both    deploy/feat_lf.run_slice (partition 'redetect', empty prof)             [feat]
  x182_merge <src>            both    feat_lf.merge_slices -> X182 parquet, F status, report                  [feat]
  atoms <src> <i> <n>         both    fu2/atoms.py root run (tokens_lf child path, redetect)                  [feat]
  tokens <src>                both    one retry pass of .err.json, atoms.py root gate, root tokens + layout
                                      check against the LADDER dft T_report                                  [feat]
  hf_inputs <src>             both    the 5 D1-HF HF/6-31G* inputs per reaction (text = the D1-HF dft-rung
                                      inputs, coordinates / tags / references from final/<rxn_id>/)          [reactot]
  hf_run <src> <i> <n> [w]    both    analysis/d1_hf/run.py cmd_run on the 'g2' slot (D1HF_SCRATCH = hf/<src>),
                                      w serial ORCA processes (default $SLURM_CPUS_PER_TASK)                  [reactot]
  hf_build <src>              both    analysis/d1_hf/build.py parse_rxn -> hf/<src>/hf_records.parquet       [reactot]
  d3 <src>                    both    D3_int per reaction -> d3 parquet                                      [reactot]
  build <src> [--refs R]      both    rung table (LADDER $LAD/inputs/dft.parquet columns + D3_int), status csv,
                                      build report (+ H-2 for mlts). R: mlts own (default) | min; label cr2  [reactot]
  h0                                  H-0 (100 label TSs) -> results/H0_report.json                           [reactot]
  h1                                  H-1 = C-5 -> results/H1_report.json (exit 3 on failure)                [reactot]
  h2 <src>                            H-2 -> results/H2_report.json (from the build outputs)                 [reactot]
  internal: _fu2 <run|gate|tokens> [i n]   (child of atoms / tokens: the FU2 root code under fu2_env)

Environments (ensure_env; the command re-executes itself with bash -c '<setup>; exec $PY <this file> <args>')
  d1       the D1 run env (d1_run/nurion/env.sh: venv python 3.9 + autodE 1.4.5 + rdkit, ORCA 6.1.1) = deploy cpu_d1.slurm
  feat     the D1PAIR Phase F env (analysis/d1_pair/jobs/env.sh + g_env.sh, PYENV=venv5, D1PAIR_PHASE=F: tblite 0.7.0,
           xtb 6.7.1, Si entries) = deploy feat_lf / tokens_lf jobs
  reactot  conda reactot + analysis/d1_hf/jobs/hf_env.sh (ORCA 6.1.1 serial, OMP_STACKSIZE 4G, ulimit -s unlimited) =
           the D1-HF runner env; dftd3 (simple-dftd3) python for D3_int
  every env: DEP_SCRATCH=$MLT, DEP_RESULTS=analysis/mlts/results (the deploy write guard then admits only the MLTS
  roots), PYTHONDONTWRITEBYTECODE=1.

Design decisions (recorded in every build report; for results/DECISIONS.md)
  H-ref   reactant references feeding frag*_rel (and d1 / d2 / barrier of the HF 9 values): mlts = the MLTS-refined
          ref_dipole.xyz / ref_dipolarophile.xyz of the A-2 folder ('own', deployable; H = H0). The LADDER dft-rung
          re-referencing (refs_cr2: own 15,233 / Q 3,176 / V 123 fragments; 3,131 reactions with H changed) is NOT
          applied to mlts: Q (lowest reference of the same molecule in another reaction) and V (DFT conformer search)
          use label-level DFT data. Optional --refs min = the LADDER g1 / g2 rule (P6 + P20 option (c)): per Q-2 key of
          $LAD/Q_keys.parquet the lowest HF reference energy among this source's own reactions, constrained-dipole keys
          keep their own reference (no new calculation; written to inputs/mlts_min.parquet). The label source (H-1)
          applies refs_cr2 exactly as LADDER (own = this run's energy, Q / V = the refs_cr2 e_hf_eh), so H / H0 are
          comparable with $LAD/inputs/dft.parquet. Labels y stay F_pairs_cr2.csv in every case.
  H-part  partition: X182 and tokens use feat_lf's 'redetect' (fragmenter.partition on the TS must reproduce the A_idx
          of ts_result.json, else the reaction is dropped and counted, F-2); the HF EDA fragment tags and D3_int use
          the same A_idx (ts_result.json, = result.json / input_meta.csv). LADDER dft used partition 'label'; on the
          DFT structures that is the same partition and the same atom maps whenever fragmenter.partition reproduces the
          label A_idx (r6_partition.label_partition first branch; LADDER F0: xtb_status ok for 9,266 / 9,266), so the
          label source runs the redetect path and H-1 counts the reactions where redetect differs (explanatory).
  H-geom  HF geometry key: D1-HF prep.g1_input / g1_frag_inputs (used by D1-G2 H, GEO2b hf_geo, D1-G3 Part B) need the
          structure in label atom order and take the fragment tags from the DFT-TS template; GEO2b / G3 structures were
          label-order by construction. MLTS structures are autodE-order with their own A_idx, and a deployed reaction
          has no template, so the inputs are written from scratch with the D1-HF dft-rung text (HF_EDA_HEAD /
          HF_FRAG_HEAD below; identical for every D1 target: '! HF D3BJ 6-31G* SMD(water) NoSym EDA TightSCF',
          %maxcore 3500, %pal nprocs 1, FRAG strings, charges 0 / mult 1; prep.replace_coords line format) with tags
          (1) = A_idx, frag<k>_dist = the TS atoms of fragment k in TS order, frag<k>_rel = rel<k>.xyz. H-1 checks the
          generated text byte for byte against $D1HF/inputs/<T>/hf/*.inp. Runner = d1_hf run.py (serial ORCA, one
          SlowConv retry), parser = d1_hf build.parse_rxn (all D1-HF gates), both unchanged on the 'g2' slot with
          D1HF_SCRATCH = $MLT/hf/<src> (the GEO2b / D1-G3 pattern).
  H-d3    D3_int = E_D3(BJ)(TS) - E(frag1 at TS geometry) - E(frag2 at TS geometry), B3LYP parameters s6 1, a1 0.3981,
          s8 1.9889, a2 4.4211 (= the ORCA 6.1.1 'DFTD3 V3.1 Rev 1' printout of the label EDA), two-body only (ORCA
          D3BJ prints E6 + E8, no three-body term; s9 = 0), cutoffs = DFTD3 3.1 defaults (pair 94.87 bohr = sqrt(9000),
          CN 40 bohr); simple-dftd3 python (reactot, version recorded); kcal/mol with 627.5094740631 (label_true).
          The label disp channel is exactly this difference (ORCA EDA 'Delta Dispersion' = Edisp(AB) - Edisp(A) -
          Edisp(B) at the TS geometry, checked on T00312 dft_orig1).
  H-fin   finalize_lf runs ORCA inside final/<rxn_id>.tmp/work; afterwards this wrapper gzips work/ts.out (disk; no H
          step reads it). Folder publication stays finalize_lf's (os.replace of the tmp folder).
  H-fail  failures never stop a slice: each stage records its reason; build lists them per reaction (status csv) and
          H-2 = share of reactions with an MLTS TS (A-2 folder with .done_ts / .done_stereo / .done_ref) that have no
          usable row (token + finite X182 + 9 finite HF values), STOP above 2 %.

sbatch (login node: one-shot submissions only; squeue -u $USER -h | wc -l first; submit from analysis/mlts) — see the
report of this phase / DECISIONS.md for the full production chain.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True                  # nothing compiled into the read-only analysis/* trees

import csv  # noqa: E402
import gzip  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
THIS = Path(__file__).resolve()
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import mlt_common as M  # noqa: E402            (stdlib only)

REPO = M.REPO
DEPLOY_DIR = REPO / "analysis" / "deploy"
D1PAIR_DIR = REPO / "analysis" / "d1_pair"
HF_DIR = REPO / "analysis" / "d1_hf"
LAD_DIR = REPO / "analysis" / "d1_ladder"
G2_STATUS = REPO / "analysis" / "d1_g2" / "results" / "G_status.csv"      # the 9,266 D1 targets (job_id, rxn_id)
LAD_DFT_TABLE = M.LAD / "inputs" / "dft.parquet"
LAD_X182 = M.LAD / "X182_d1_dft.parquet"
LAD_TOKENS = M.LAD / "fu2" / "tokens"
LAD_T_REPORT = LAD_DIR / "results" / "fu2" / "T_report.json"
LAD_REFS_CR2 = M.LAD / "refs_cr2.parquet"
LABEL_META = M.G2 / "G" / "input_meta_d1_g2.csv"                         # label A_idx (LADDER F0 input)
LABEL_LABELS = M.G2 / "G" / "labels_d1_g2.json"                         # LADDER F0 label rows (H-1 meta check)
D1HF_SCR = Path("/gpfs/tmp_cpu2/yeseo1ee/d1_hf")                         # D1-HF scratch (read only): inputs/<T>/hf
D1HF_LABELS = HF_DIR / "results" / "labels_d1_hf.parquet"
CODE_VERSION = "mlts-h-1"
KCAL = 627.5094740631                         # = hf_common.KCAL = lad_common.KCAL (label_true _common)
H_TARGETS = ("d1", "d2", "elst", "pauli", "oi", "disp", "cpcm", "cds", "barrier")
H_REREF = ("d1", "d2", "barrier")
CALCS = ("frag1_dist", "frag2_dist", "frag1_rel", "frag2_rel", "eda")
TOL_H1 = 1e-4                                 # spec 7 H-1 (kcal/mol, X182 / token units)
H0_RULE_KCAL = 0.01                           # spec 7 H-0
H2_MAX = 0.02                                 # spec 7 H-2
T0 = float(os.environ.get("MLT_JOB_T0") or time.time())
STOP_AFTER_H = float(os.environ.get("MLT_STOP_AFTER_H", "44"))
SYM2Z = {"H": 1, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Br": 35, "I": 53}
ANG2BOHR = 1.0 / M.BOHR2A

# ---------------------------------------------------------------- D3(BJ) (decision H-d3)
D3_PARAMS = dict(s6=1.0, s8=1.9889, s9=0.0, a1=0.3981, a2=4.4211, alp=14.0)
D3_CUTOFF_BOHR = dict(disp2=math.sqrt(9000.0), disp3=40.0, cn=40.0)

# ---------------------------------------------------------------- HF inputs (decision H-geom; = D1-HF dft-rung text)
HF_EDA_HEAD = ('! HF D3BJ 6-31G* SMD(water) NoSym EDA TightSCF\n%maxcore 3500\n\n%pal nprocs 1 end\n\n%eda\n'
               '  FRAG1 "HF D3BJ 6-31G* SMD(water) NoSym TightSCF"\n'
               '  FRAG2 "HF D3BJ 6-31G* SMD(water) NoSym TightSCF"\n'
               '  FRAG1_C 0\n  FRAG1_M 1\n  FRAG2_C 0\n  FRAG2_M 1\nend\n\n* xyz 0 1\n')
HF_FRAG_HEAD = ('! HF D3BJ 6-31G* CPCM(water) NoSym TightSCF\n%maxcore 3500\n\n%cpcm\n  smd true\n  smdsolvent "water"\n'
                'end\n\n* xyz 0 1\n')
HF_TAIL = "*\n"

# ---------------------------------------------------------------- environments (ensure_env)
CONDA_SH = "/home1/yeseo1ee/miniconda3/etc/profile.d/conda.sh"
ENV_D1FU2 = Path("/gpfs/tmp_cpu2/yeseo1ee/d1_pair/fu2/env_d1fu2")
ESPLEY_PY = Path("/gpfs/tmp_cpu2/yeseo1ee/espley_rev6/env_espley/bin/python")
ENVS = {
    "d1": 'source /gpfs/tmp_cpu2/yeseo1ee/d1/d1_run/nurion/env.sh && PY="$D1_PY"',
    "feat": (f'export PYENV=venv5 D1PAIR_PHASE=F D1PAIR_CODE={D1PAIR_DIR} && source {D1PAIR_DIR}/jobs/env.sh && '
             f'source {D1PAIR_DIR}/jobs/g_env.sh'),
    "reactot": f'source {HF_DIR}/jobs/hf_env.sh && unset D1HF_SCRATCH && PY="$(command -v python)"',
    "fu2": (f'source {CONDA_SH} && conda activate reactot && test -f {ENV_D1FU2}/.d1fu2_env_ok && '
            f'PY={ENV_D1FU2}/bin/python && export PYTHONNOUSERSITE=1 PYTHONWARNINGS=ignore'),
    "espley": (f'source {CONDA_SH} && conda activate reactot && PY={ESPLEY_PY} && '
               'export PYTHONNOUSERSITE=1 PYTHONWARNINGS=ignore TMPDIR=${SLURM_TMPDIR:-/tmp}'),
}


def env_common():
    return (f"export MLT_SCRATCH={M.MLT} DEP_SCRATCH={M.MLT} DEP_RESULTS={M.RES} "
            "D1PAIR_SCRATCH=/gpfs/tmp_cpu2/yeseo1ee/d1_pair D1LADDER_SCRATCH=/gpfs/tmp_cpu2/yeseo1ee/d1_ladder "
            "GEO2_SCRATCH=/gpfs/tmp_cpu2/yeseo1ee/geo2b PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 "
            "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
            f"MLT_JOB_T0={int(T0)}")


def ensure_env(name, script=None, argv=None):
    """re-execute `script argv` (default: this process) in environment `name` unless it already runs there."""
    if os.environ.get("MLT_IN_ENV") == name:
        return
    script = str(script or Path(sys.argv[0]).resolve())
    argv = list(sys.argv[1:] if argv is None else argv)
    sh = f'{ENVS[name]} && {env_common()} MLT_IN_ENV={name} && exec "$PY" "$@"'
    print(f"[env] re-exec in '{name}': {Path(script).name} {' '.join(argv)}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os.execvp("bash", ["bash", "-c", sh, "_", script] + argv)


# ---------------------------------------------------------------- small helpers
def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    print(f"[H {time.time() - T0:7.0f}s] {msg}", flush=True)


def cpus():
    return int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1)


def read_xyz(path):
    """(symbols, [[x, y, z]] floats as written) of a plain xyz file."""
    return M.read_xyz(path)


def fp_pairs(s):
    """'3-7 5-6' -> [[3, 7], [5, 6]]."""
    return [sorted(int(x) for x in t.split("-")) for t in str(s).split()]


def mkdir(p):
    M.assert_write_ok(Path(p) / "x")
    Path(p).mkdir(parents=True, exist_ok=True)
    return Path(p)


def ids_arg(a):
    """global options --root DIR / --ids CSV (mlts source only) -> (rest, root, ids)."""
    root = ids = None
    out = []
    k = 0
    while k < len(a):
        if a[k] == "--root":
            root = Path(os.path.abspath(a[k + 1]))
            k += 2
        elif a[k] == "--ids":
            with open(a[k + 1], newline="") as fh:
                ids = sorted({int(r["rxn_id"]) for r in csv.DictReader(fh)})
            k += 2
        else:
            out.append(a[k])
            k += 1
    return out, root, ids


_OPT = dict(root=None, ids=None)


def src_cfg(name):
    """paths of a source (module docstring)."""
    if name == "mlts":
        R = _OPT["root"] or M.MLT
        res = M.RES if _OPT["root"] is None else R / "reports"
        return dict(name="mlts", root=R, ade=R / "ade" / "jobs", final=R / "final", meta=R / "inputs" / "meta_mlts",
                    prof=R / "inputs" / "prof_empty", feat=R / "feat" / "mlts", x182=R / "inputs" / "X182_mlts.parquet",
                    fu2_scr=R / "fu2_mlts", fu2_res=res / "fu2_mlts", hf=R / "hf" / "mlts",
                    d3=R / "inputs" / "d3_mlts.parquet", table=R / "inputs" / "mlts.parquet",
                    table_min=R / "inputs" / "mlts_min.parquet", status=R / "inputs" / "status_mlts.csv", rep=res,
                    tag="mlts", refs="own", targets=None, ids=_OPT["ids"])
    if name == "label":
        R = M.MLT / "h1"
        return dict(name="label", root=R, ade=R / "ade", final=R / "final", meta=R / "meta", prof=R / "prof_empty",
                    feat=R / "feat", x182=R / "X182_label.parquet", fu2_scr=R / "fu2", fu2_res=R / "fu2_res",
                    hf=R / "hf", d3=R / "d3_label.parquet", table=R / "label.parquet", table_min=None,
                    status=R / "status_label.csv", rep=R / "reports", tag="mltslab", refs="cr2",
                    targets=R / "targets.csv", ids=None)
    raise SystemExit(f"unknown source {name!r} (mlts | label)")


def rep_path(S, step):
    return S["rep"] / f"H_{step}_{S['name']}.json"


def versions(S):
    return f"fu2-atoms-3-{S['tag']}", f"fu2-tokens-3-{S['tag']}"


def targets(S):
    """[(rxn_id, job_id)] of the source, job_id order."""
    if S["targets"] is None:
        with open(G2_STATUS, newline="") as fh:
            rows = [(int(r["rxn_id"]), r["job_id"]) for r in csv.DictReader(fh)]
        if len(rows) != 9266:
            M.stop(f"{G2_STATUS}: {len(rows)} rows, expected 9,266")
        if S.get("ids"):
            want = set(S["ids"])
            rows = [r for r in rows if r[0] in want]
            if len(rows) != len(want):
                M.stop(f"--ids: {len(want) - len(rows)} rxn_ids are not D1 targets")
    else:
        if not Path(S["targets"]).is_file():
            M.not_ready(f"{S['targets']} missing (run label_src first)")
        with open(S["targets"], newline="") as fh:
            rows = [(int(r["rxn_id"]), r["job_id"]) for r in csv.DictReader(fh)]
    return sorted(rows, key=lambda r: r[1])


def records(S):
    """meta outputs: (labels.json records by rxn_id, A_idx by rxn_id); exit 2 when meta has not run."""
    lp, mp = S["meta"] / "labels.json", S["meta"] / "input_meta.csv"
    if not (lp.is_file() and mp.is_file()):
        M.not_ready(f"{lp} / {mp} missing (run meta {S['name']})")
    recs = {int(d["rxn_id"]): d for d in json.loads(lp.read_text())}
    with open(mp, newline="") as fh:
        A = {int(r["rxn_id"]): [int(x) for x in str(r["A_idx"]).split()] for r in csv.DictReader(fh)}
    return recs, A


def jdump(path, obj):
    M.write_json(path, obj)
    return path


def _clean(o):
    """JSON-able copy (numpy -> python, NaN -> None)."""
    try:
        import numpy as np
    except ImportError:                                      # pragma: no cover
        np = None
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_clean(v) for v in (sorted(o) if isinstance(o, set) else o)]
    if np is not None:
        if isinstance(o, np.ndarray):
            return _clean(o.tolist())
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            o = float(o)
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, Path):
        return str(o)
    return o


def qstats(v):
    import numpy as np
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], dtype=float)
    if not len(v):
        return dict(n=0)
    return dict(n=int(len(v)), median=float(np.median(v)), p90=float(np.quantile(v, 0.9)), max=float(v.max()),
                mean=float(v.mean()))


# ================================================================ label source (H-1)
def label_A():
    with open(LABEL_META, newline="") as fh:
        return {int(r["rxn_id"]): sorted(int(x) for x in r["A_idx"].split()) for r in csv.DictReader(fh)}


def cmd_label_src(n=50):
    """the H-1 sample + its A-2 folders (decision: seeded M.SEED, n/5 per dipole, among dft-rung usable reactions so
    that every compared quantity exists on the LADDER side)."""
    ensure_env("reactot")
    import numpy as np
    import pandas as pd
    S = src_cfg("label")
    n = int(n)
    T = pd.read_parquet(LAD_DFT_TABLE, columns=["rxn_id", "job_id", "usable"])
    lab = {int(d["rxn_id"]): d for d in json.loads(M.LABELS.read_text())}
    T["dipole"] = T.rxn_id.map(lambda r: lab[int(r)]["dipole"])
    tp = S["targets"]
    if tp.is_file():
        old = pd.read_csv(tp)
        if len(old) != n:
            M.stop(f"{tp} has {len(old)} rows != {n}: use the existing sample or remove it first")
        pick = old
    else:
        rng = np.random.default_rng(M.SEED)
        per = n // len(M.DIPOLES)
        rows = []
        for d in M.DIPOLES:
            ids = sorted(int(r) for r in T.rxn_id[T.usable & (T.dipole == d)])
            rows += [int(x) for x in sorted(rng.choice(ids, size=per, replace=False))]
        pick = T.set_index("rxn_id").loc[rows, ["job_id", "dipole"]].reset_index().sort_values("job_id")
        M.write_csv(tp, pick[["rxn_id", "job_id", "dipole"]])
    A = label_A()
    made = Counter()
    for r in pick.itertuples(index=False):
        made[make_label_ade(int(r.rxn_id), str(r.job_id), lab[int(r.rxn_id)], A[int(r.rxn_id)], S["ade"])] += 1
    log(f"label_src: {len(pick)} reactions ({pick.dipole.value_counts().to_dict()}), A-2 folders {dict(made)}")
    return 0


def make_label_ade(rid, job, lab, A, root):
    """$MLT/h1/ade/<job>/ in the A-2 layout from $LAD/dft_final/<rid>/ (+ label record): ts.xyz, ref_*.xyz byte copies,
    ts_result.json (A_idx, B_idx, formed_pairs_ts, checks, charges), ref_result.json (alt_used), stereo_result.json,
    the three stage markers. Existing complete folders are kept."""
    out = Path(root) / job
    if (out / ".done_ref").is_file():
        return "exists"
    src = M.LAD / "dft_final" / str(rid)
    if not (src / ".done").is_file():
        M.stop(f"{src} has no .done")
    res = json.loads((src / "result.json").read_text())
    n = int(open(src / "ts.xyz").readline().split()[0])
    if not (res.get("checks") or {}).get("A_idx_eq_meta") or int(res["checks"]["n_f1"]) != len(A):
        M.stop(f"{src}: result.json A_idx check / n_f1 differs from {LABEL_META}")
    tmp = Path(root) / f".{job}.tmp"
    M.assert_write_ok(tmp)
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for a, b in (("ts.xyz", "ts.xyz"), ("rel1.xyz", "ref_dipole.xyz"), ("rel2.xyz", "ref_dipolarophile.xyz")):
        shutil.copyfile(src / a, tmp / b)
    B = sorted(set(range(n)) - set(A))
    tsr = dict(job_id=job, source=f"label TS {src} (H-1 label source)", A_idx=list(A), B_idx=B,
               formed_pairs_ts=fp_pairs(lab["formed_pairs_ts"]), charge_dipole=int(lab["charge1"]),
               charge_dipolarophile=int(lab["charge2"]),
               checks=dict(flag_async=bool(lab["flag_async"]), no_foreign_bond=not bool(lab["flag_foreign_bond"])))
    (tmp / "ts_result.json").write_text(json.dumps(tsr, indent=1))
    (tmp / "ref_result.json").write_text(json.dumps(dict(job_id=job, alt_used=int(lab["alt_used"]),
                                                         source="label record alt_used"), indent=1))
    (tmp / "stereo_result.json").write_text(json.dumps(dict(job_id=job, status="label"), indent=1))
    for st in ("ts", "stereo", "ref"):
        (tmp / f".done_{st}").write_text(now() + "\n")
    if out.exists():
        shutil.rmtree(out)
    os.replace(tmp, out)
    return "made"


def cmd_smoke_src(n=10):
    """test fixture of the mlts path (needs --root): n G2-ok D1 targets (the smallest n/5 per dipole) with their
    $G2/jobs/<T> folders (xTB TSs in the A-2 layout, autodE order) copied to <root>/ade/jobs/<T> (no inputs tar) and
    <root>/ids.csv; then run the mlts commands with --root <root> --ids <root>/ids.csv."""
    ensure_env("reactot")
    if _OPT["root"] is None or Path(_OPT["root"]).resolve() == M.MLT.resolve():
        M.stop("smoke_src needs --root <dir under $MLT> (never the production root)")
    R = Path(_OPT["root"])
    with open(G2_STATUS, newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["status"] == "ok"]
    pick = []
    for d in M.DIPOLES:
        c = sorted((int(r["n_atoms"]), r["job_id"], int(r["rxn_id"])) for r in rows if r["dipole"] == d)
        pick += c[:max(1, int(n) // len(M.DIPOLES))]
    lines = ["rxn_id,job_id"] + [f"{rid},{job}" for _, job, rid in sorted(pick, key=lambda x: x[1])]
    M.write_text(R / "ids.csv", "\n".join(lines) + "\n")
    for _, job, _ in pick:
        dst = R / "ade" / "jobs" / job
        if (dst / ".done_ref").is_file():
            continue
        mkdir(dst)
        for p in sorted((M.G2_JOBS / job).iterdir()):
            if p.is_file() and p.name != "inputs.tar.gz":
                shutil.copyfile(p, dst / p.name)
    log(f"smoke_src: {len(pick)} reactions -> {R}")
    return 0


# ================================================================ finalize / meta (D1 env)
def _deploy_import():
    if str(DEPLOY_DIR) not in sys.path:
        sys.path.insert(0, str(DEPLOY_DIR))


def _fin_one(job_row, src_dir, out_root, cores):
    import finalize_lf as FL
    rid = FL.rxn_key(job_row)
    if FL.marker(Path(out_root) / str(rid)) is None and time.time() - T0 > STOP_AFTER_H * 3600:
        return str(job_row["job_id"]), "not_started", 0.0
    t = time.time()
    try:
        st = FL.finalize_lf(job_row, src_dir, out_root, cores)
    except SystemExit as e:
        st = f"error_exit_{e.code}"
    except Exception as e:                                          # noqa: BLE001
        st = f"error_{type(e).__name__}:{e}"
    if st == "ok":                                                  # decision H-fin
        p = Path(out_root) / str(rid) / "work" / "ts.out"
        if p.is_file():
            with open(p, "rb") as fi, gzip.open(str(p) + ".gz.tmp", "wb", compresslevel=6) as fo:
                shutil.copyfileobj(fi, fo)
            os.replace(str(p) + ".gz.tmp", str(p) + ".gz")
            p.unlink()
    return str(job_row["job_id"]), st, time.time() - t


def cmd_finalize(src, i, n, cores=1):
    ensure_env("d1")
    _deploy_import()
    import finalize_lf as FL
    from concurrent.futures import ProcessPoolExecutor, as_completed
    S = src_cfg(src)
    i, n, cores = M.array_task(i), int(n), int(cores)
    rows = targets(S)[i::n]
    man = FL.load_manifest()
    mkdir(S["final"])
    todo, cnt = [], Counter()
    for rid, job in rows:
        if FL.marker(S["final"] / str(rid)) is not None:
            cnt["skip"] += 1
            continue
        if not (S["ade"] / job).is_dir():
            cnt["no_ade_folder"] += 1
            continue
        r = man.get(job)
        if r is None or int(r["meta_rxn_id"]) != rid:
            M.stop(f"{job}: manifest row missing or meta_rxn_id != {rid}")
        todo.append(dict(r))
    n_par = max(1, cpus() // cores)
    log(f"finalize {src} slice {i}/{n}: {len(rows)} targets, {len(todo)} to finalise, {n_par} x {cores} cores "
        f"({S['ade']} -> {S['final']}); {dict(cnt)}")
    with ProcessPoolExecutor(max_workers=n_par, initializer=FL._init_worker, initargs=(cores,)) as ex:
        futs = {ex.submit(_fin_one, r, S["ade"] / r["job_id"], S["final"], cores): r["job_id"] for r in todo}
        for f in as_completed(futs):
            try:
                T, st, w = f.result()
            except Exception as e:                                  # noqa: BLE001  (worker crash)
                T, st, w = futs[f], f"error_worker:{type(e).__name__}", 0.0
            key = st.split(":")[0]
            cnt[key] += 1
            if key != "ok":
                print(f"  {T}: {st}", flush=True)
    log(f"finalize {src} slice {i}/{n}: {dict(cnt)}")
    return 2 if cnt.get("not_started") or any(k.startswith("error") for k in cnt) else 0


def cmd_meta(src):
    """meta_lf.build on every .done folder of final/ (targets only) -> labels.json + input_meta.csv. Records that
    cannot be built are dropped and counted (H-2); a reactant charge != 0 is a STOP (spec 0-6: all neutral)."""
    ensure_env("d1")
    _deploy_import()
    import finalize_lf as FL
    import meta_lf as ML
    import pandas as pd
    S = src_cfg(src)
    tg = {rid for rid, _ in targets(S)}
    rids = sorted(int(p.parent.name) for p in S["final"].glob("*/.done")
                  if p.parent.name.isdigit() and int(p.parent.name) in tg)
    if not rids:
        M.not_ready(f"no .done folder under {S['final']}")
    man = FL.load_manifest()
    recs, meta, charges, prob, cross = ML.build(S["final"], man, rids)
    Q = pd.DataFrame(charges)
    nz = Q[Q.nonzero] if len(Q) else Q
    if len(nz):
        M.stop(f"{len(nz)} reactions with a reactant charge != 0: {nz.rxn_id.tolist()[:10]}")
    recs.sort(key=lambda r: r["rxn_id"])
    mkdir(S["meta"])
    M.write_text(S["meta"] / "labels.json", json.dumps(recs, indent=1))      # NaN DFT keys as the JSON token NaN
    M.write_csv(S["meta"] / "input_meta.csv", pd.DataFrame(meta, columns=["rxn_id", "A_idx"]).sort_values("rxn_id"))
    mkdir(S["prof"])
    if any(S["prof"].iterdir()):
        M.stop(f"{S['prof']} is not empty")
    rep = dict(step="meta", source=src, n_done=len(rids), n_records=len(recs), n_problems=len(prob), problems=prob[:100],
               n_pairs_not_across_partition=sum(not x for x in cross), code="analysis/deploy/meta_lf.py build",
               outputs={str(S["meta"] / f): M.sha256(S["meta"] / f) for f in ("labels.json", "input_meta.csv")},
               written=now())
    jdump(rep_path(S, "meta"), rep)
    log(f"meta {src}: {len(recs)} records of {len(rids)} .done folders, {len(prob)} problems")
    return 0


# ================================================================ X182 / tokens (feat env)
def _feat_import():
    _deploy_import()
    import feat_lf as FL                                       # puts analysis/d1_pair and d1_pair/fu2 on sys.path
    return FL


def cmd_x182(src, i, n):
    ensure_env("feat")
    FL = _feat_import()
    FL.parent_env()
    S = src_cfg(src)
    recs, A = records(S)
    mkdir(S["prof"])
    ids = sorted(recs)
    return FL.run_slice(S["final"], S["meta"] / "labels.json", S["meta"] / "input_meta.csv", S["prof"], S["feat"],
                        M.array_task(i), int(n), ids=ids)


def cmd_x182_merge(src):
    ensure_env("feat")
    FL = _feat_import()
    import numpy as np
    import pandas as pd
    v0 = FL.parent_env()
    S = src_cfg(src)
    R = FL.merge_slices(S["feat"])
    g, e, m, cols, ids = R["g"], R["e"], R["m"], R["cols"], R["ids"]
    x = m.loc[m.F_ok, ["rxn_id"] + cols].sort_values("rxn_id").reset_index(drop=True)
    ref = FL.x182_columns_ref()
    f1 = list(x.columns) == ref and len(ref) == 183
    finite = bool(np.isfinite(x[cols].to_numpy(float)).all())
    why = FL.reasons(m, ids)
    st = dict(zip(m.rxn_id.astype(int), m.xtb_status.astype(str)))
    redet = {r: st[r] for r in ids if FL.is_redetect_fail(st[r])}
    other = {r: w for r, w in why.items() if w != "ok" and r not in redet}
    M.write_parquet(S["feat"] / "xtb_features_g1.parquet", g)
    M.write_parquet(S["feat"] / "xtb_features_ext_g1.parquet", e)
    M.write_csv(S["feat"] / "F_status.csv", pd.DataFrame(sorted(why.items()), columns=["rxn_id", "reason"]))
    stop = []
    if not f1:
        stop.append("F-1: X182 columns != the X182_d1_g2 columns")
    if not finite:
        stop.append("non-finite values among the 182 columns of F_ok rows")
    if not stop:
        M.write_parquet(S["x182"], x)
    import ext_features as ef
    rep = dict(step="x182", source=src, written=now(), n_selected=len(ids), n_feat=len(x),
               fail_rate=(len(ids) - len(x)) / len(ids) if ids else None, partition=FL.PARTITION,
               F1_columns_equal=f1, F1_all_finite=finite, reasons=dict(Counter(why.values()).most_common()),
               not_ok={str(r): w for r, w in sorted(why.items()) if w != "ok"},
               F2_redetect=dict(n=len(redet), statuses=dict(Counter(redet.values())), rxn=sorted(redet)),
               F3_other=dict(n=len(other), reasons=dict(Counter(other.values()).most_common(40))),
               F4=FL.timing_summary(R["timing"], e), versions=ef.versions(), engines_job=v0,
               x182=dict(path=str(S["x182"]), sha256=M.sha256(S["x182"]) if not stop else None),
               code=dict(feat_lf=M.sha256(FL.THIS)), stop=bool(stop), stop_reasons=stop)
    jdump(rep_path(S, "F"), rep)
    log(f"x182_merge {src}: {len(x)} / {len(ids)} rows, redetect failures {len(redet)}, other {len(other)}; stop {stop}")
    return 3 if stop else 0


def fu2_env(S):
    """feat_lf.lf_env (partition redetect, empty prof) + the FU2 root variables (tokens_lf.fu2_env with this source's
    roots and versions)."""
    FL = _feat_import()
    env = FL.lf_env(S["final"], S["meta"] / "labels.json", S["meta"] / "input_meta.csv", S["prof"], S["feat"])
    av, tv = versions(S)
    env.update(FU2_SCR=str(S["fu2_scr"]), FU2_RES=str(S["fu2_res"]), FU2_X182=str(S["x182"]), FU2_ATOM_VERSION=av,
               FU2_TOKEN_VERSION=tv)
    for k in ("FU2_SCR", "FU2_RES"):
        M.assert_write_ok(Path(env[k]) / "x")
    return env


def _run_fu2(S, cmd, *args):
    env = fu2_env(S)
    argv = [sys.executable, str(THIS), "_fu2", cmd] + [str(a) for a in args]
    log(f"child {' '.join(argv[2:])}")
    r = subprocess.run(argv, env=env)
    log(f"child exit {r.returncode}")
    return r.returncode


def _child_fu2(cmd, args):
    """= deploy tokens_lf._child with this source's versions: atoms.root_bind, the root check with partition
    'redetect' (tokens_lf._env_root_lf), then atoms.cmd_run / cmd_gate / cmd_root_tokens unchanged."""
    _deploy_import()
    import tokens_lf as TL
    A, D = TL.A, TL.D
    A.root_bind()
    A._env_root = TL._env_root_lf
    A._env()
    for p in (A.F.FU2_SCR, A.F.FU2_RES):
        D.assert_write_ok(p)
        M.assert_write_ok(Path(p) / "x")
    if A.ATOM_VERSION != os.environ.get("FU2_ATOM_VERSION"):
        M.stop(f"atoms.ATOM_VERSION {A.ATOM_VERSION} != {os.environ.get('FU2_ATOM_VERSION')}")
    if cmd == "run":
        return A.cmd_run(*args)
    if cmd == "gate":
        return A.cmd_gate()
    if cmd == "tokens":
        return A.cmd_root_tokens()
    raise SystemExit(f"unknown child command {cmd}")


def _need_x182(S):
    rp = rep_path(S, "F")
    if not (rp.is_file() and S["x182"].is_file()):
        M.not_ready(f"{rp} / {S['x182']} missing (x182_merge {S['name']})")
    r = M.read_json(rp)
    if r.get("stop") or (r.get("x182") or {}).get("sha256") != M.sha256(S["x182"]):
        M.stop(f"{rp}: stop or X182 changed after the report")
    return r


def cmd_atoms(src, i, n):
    ensure_env("feat")
    FL = _feat_import()
    FL.parent_env()
    S = src_cfg(src)
    _need_x182(S)
    return _run_fu2(S, "run", M.array_task(i), int(n))


def cmd_tokens(src):
    ensure_env("feat")
    FL = _feat_import()
    FL.parent_env()
    S = src_cfg(src)
    _need_x182(S)
    av, tv = versions(S)
    rec = dict(step="tokens", source=src, written=now(), atom_version=av, token_version=tv)
    err = sorted((S["fu2_scr"] / "atoms").glob("*.err.json"))
    rec["n_err_before"] = len(err)
    if err:                                                     # one rerun pass of the retryable errors (tokens_lf)
        rec["retry_rc"] = _run_fu2(S, "run", 0, 1)
    rc = _run_fu2(S, "gate")
    rec["gate_rc"] = rc
    qf = S["fu2_res"] / "Q_report.json"
    if rc == 2:
        jdump(rep_path(S, "T"), rec)
        M.not_ready("atoms incomplete: run 'atoms' for every slice")
    if not qf.is_file():
        M.stop(f"no {qf} after the gate (exit {rc})")
    q = M.read_json(qf)
    rec["Q"] = {k: q.get(k) for k in ("n_target", "n_ok", "n_compute_fail", "n_qa_fail", "qa_maxdiff_max", "n_qb_fail",
                                      "ext_pass_rate", "ext_used", "n_use", "fail_reasons", "stop", "stop_reason",
                                      "atom_version")}
    stop = []
    if q.get("stop") or rc == 3:
        stop.append(f"FU2 Q gate stop: {q.get('stop_reason')}")
    if q.get("atom_version") != av:
        stop.append(f"Q_report atom_version {q.get('atom_version')} != {av}")
    if not stop:
        rc2 = _run_fu2(S, "tokens")
        rec["tokens_rc"] = rc2
        tf = S["fu2_res"] / "T_report.json"
        if not tf.is_file():
            stop.append(f"no {tf} after tokens (exit {rc2})")
        else:
            t, ref = M.read_json(tf), M.read_json(LAD_T_REPORT)
            el = [int(z) for z in t.get("elements") or []]
            chk = dict(elements_subset_of_ladder=set(el) <= set(int(z) for z in ref["elements"]),
                       elements_equal_ladder=el == [int(z) for z in ref["elements"]],
                       bin_cols=t.get("bin_cols") == ref["bin_cols"], num_cols=t.get("num_cols") == ref["num_cols"],
                       ext_used=t.get("ext_used") == ref["ext_used"], glob_cols=t.get("glob_cols") == ref["glob_cols"],
                       token_version=t.get("token_version") == tv, x182_sha256=t.get("x182_sha256") == M.sha256(S["x182"]),
                       gate_Ta=bool(t.get("gate_Ta")) and all((t.get("gate_Ta") or {}).values()) and not t.get("stop"))
            rec["T"] = dict(checks=chk, n_reactions=t.get("n_reactions"), n_bad=len(t.get("bad") or []), elements=el,
                            ladder_T_report=str(LAD_T_REPORT), t_report_sha256=M.sha256(tf))
            hard = [k for k in ("elements_subset_of_ladder", "bin_cols", "num_cols", "ext_used", "glob_cols",
                                "token_version", "x182_sha256") if not chk[k]]
            if hard:
                stop.append(f"token layout differs from the LADDER dft rung (the dft units cannot read it): {hard}")
            if rc2 not in (0,):
                stop.append(f"tokens exit {rc2} (T-a: {t.get('gate_Ta')})")
    rec.update(stop=bool(stop), stop_reasons=stop)
    jdump(rep_path(S, "T"), rec)
    log(f"tokens {src}: stop {stop}")
    return 3 if stop else 0


# ================================================================ HF EDA (reactot env)
def hf_mods(S):
    """hf_common (+ run.py, build.py) of analysis/d1_hf on D1HF_SCRATCH = hf/<src> (imported once per process)."""
    os.environ["D1HF_SCRATCH"] = str(S["hf"])
    os.environ["D1HF_RES"] = str(S["hf"] / "results")
    os.environ.setdefault("D1HF_JOB_T0", str(int(T0)))
    M.assert_write_ok(S["hf"] / "x")
    if str(HF_DIR) not in sys.path:
        sys.path.insert(0, str(HF_DIR))
    import hf_common as H
    if os.path.abspath(H.HF) != os.path.abspath(S["hf"]):
        M.stop(f"hf_common.HF = {H.HF} != {S['hf']} (hf_common imported before D1HF_SCRATCH was set?)")
    return H


def _load_mod(name, path):
    m = sys.modules.get(name)
    if m is not None:
        return m
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def xyz_line(el, x, y, z, tag=None):
    """= analysis/d1_hf/prep.replace_coords line format."""
    lab = f"{el}({tag})" if tag is not None else el
    return f"  {lab:<6}{x:16.8f}{y:14.8f}{z:14.8f}\n"


def hf_texts(ts, A, rel1, rel2):
    """the 5 D1-HF inputs (decision H-geom): eda tags (1) = A, (2) = the rest; frag<k>_dist = the TS atoms of fragment
    k in TS order; frag<k>_rel = rel<k> in its own order."""
    syms, X = ts
    A = set(int(a) for a in A)
    out = {"eda.inp": HF_EDA_HEAD + "".join(xyz_line(s, *x, tag=1 if i in A else 2) for i, (s, x) in
                                            enumerate(zip(syms, X))) + HF_TAIL}
    for k, keep in ((1, lambda i: i in A), (2, lambda i: i not in A)):
        out[f"frag{k}_dist.inp"] = HF_FRAG_HEAD + "".join(xyz_line(s, *x) for i, (s, x) in enumerate(zip(syms, X))
                                                          if keep(i)) + HF_TAIL
    for k, (s2, X2) in ((1, rel1), (2, rel2)):
        out[f"frag{k}_rel.inp"] = HF_FRAG_HEAD + "".join(xyz_line(s, *x) for s, x in zip(s2, X2)) + HF_TAIL
    return out


def cmd_hf_inputs(src):
    import pandas as pd
    ensure_env("reactot")
    S = src_cfg(src)
    hf_mods(S)
    recs, A = records(S)
    rows = []
    for rid in sorted(recs):
        r = recs[rid]
        job, d = str(r["job_id"]), S["final"] / str(rid)
        try:
            ts, r1, r2 = (read_xyz(d / f) for f in ("ts.xyz", "rel1.xyz", "rel2.xyz"))
            a = A[rid]
            if int(r["charge1"]) != 0 or int(r["charge2"]) != 0:
                raise ValueError("charged_fragment (the D1-HF input text is neutral singlet only)")
            fa = sorted(ts[0][i] for i in a)
            fb = sorted(ts[0][i] for i in range(len(ts[0])) if i not in set(a))
            if fa != sorted(r1[0]) or fb != sorted(r2[0]):
                raise ValueError("reference_composition_differs_from_ts_fragment")
            texts = hf_texts(ts, a, r1, r2)
            dest = S["hf"] / "inputs" / job / "g2"
            for name in [c + ".inp" for c in CALCS]:              # eda last
                f = dest / name
                if not (f.is_file() and f.read_text() == texts[name]):
                    M.write_text(f, texts[name])
            rows.append(dict(job_id=job, rxn_id=rid, n_atoms=len(ts[0]), status="ok"))
        except Exception as e:                                      # noqa: BLE001
            rows.append(dict(job_id=job, rxn_id=rid, n_atoms=None, status=f"error:{type(e).__name__}:{e}"))
    D = pd.DataFrame(rows)
    M.write_csv(S["hf"] / "inputs.csv", D)
    ok = D[D.status == "ok"].sort_values(["n_atoms", "job_id"], ascending=[False, True])
    M.write_csv(S["hf"] / "H_list.csv", ok[["job_id", "n_atoms"]])     # largest first: balanced slices / pool tail
    log(f"hf_inputs {src}: {len(ok)} / {len(D)} ok {D.status.value_counts().to_dict()}")
    return 0


def cmd_hf_run(src, i, n, workers=None):
    ensure_env("reactot")
    S = src_cfg(src)
    H = hf_mods(S)
    R = _load_mod("d1hf_run", HF_DIR / "run.py")
    if R.H is not H:
        M.stop("d1_hf run.py imported another hf_common")
    lst = S["hf"] / "H_list.csv"
    if not lst.is_file():
        M.not_ready(f"{lst} missing (hf_inputs {src})")
    w = int(workers or cpus())
    return R.cmd_run(str(lst), "g2", "hf", w, M.array_task(i), int(n))


def cmd_hf_build(src):
    import pandas as pd
    ensure_env("reactot")
    S = src_cfg(src)
    H = hf_mods(S)
    B = _load_mod("d1hf_build", HF_DIR / "build.py")
    if B.H is not H:
        M.stop("d1_hf build.py imported another hf_common")
    L = pd.read_csv(S["hf"] / "inputs.csv")
    rows = []
    for r in L.itertuples(index=False):
        if r.status != "ok":
            rows.append(dict(job_id=r.job_id, rxn_id=int(r.rxn_id), status="no_input", reason=r.status))
            continue
        try:
            fl = B.flat(B.parse_rxn(str(r.job_id), "g2"))
        except Exception as e:                                      # noqa: BLE001
            fl = dict(job_id=r.job_id, status="parse_error", reason=f"{type(e).__name__}: {e}")
        fl["rxn_id"] = int(r.rxn_id)
        rows.append(fl)
    D = pd.DataFrame(rows)
    for c in D.columns:                                             # lists / dicts / mixed text -> str (parquet)
        if D[c].dtype == object and c not in ("job_id", "status"):
            if D[c].map(lambda v: isinstance(v, (list, dict, tuple))).any() or c in ("reason", "eda_xrow",
                                                                                      "setting_problems"):
                D[c] = D[c].map(lambda v: "" if v is None else str(v))
    M.write_parquet(S["hf"] / "hf_records.parquet", D)
    st = D.status.value_counts().to_dict()
    done = int((D.status != "not_done").sum())
    rep = dict(step="hf", source=src, written=now(), n=len(D), status=st,
               not_ok=D.loc[D.status != "ok", ["rxn_id", "status", "reason"]].astype(str).to_dict("records")[:100],
               coreh=qstats(D.coreh_total.tolist()) if "coreh_total" in D else None,
               code=dict(run=M.sha256(HF_DIR / "run.py"), build=M.sha256(HF_DIR / "build.py"),
                         hf_common=M.sha256(HF_DIR / "hf_common.py")),
               settings="D1-HF dft-rung text (HF_EDA_HEAD / HF_FRAG_HEAD), serial ORCA 6.1.1, one SlowConv retry")
    jdump(rep_path(S, "HF"), rep)
    log(f"hf_build {src}: {st}")
    return 2 if st.get("not_done") or st.get("incomplete") else 0


# ================================================================ D3_int (reactot env)
def d3_versions():
    try:
        from importlib.metadata import version
        v = version("dftd3")
    except Exception:                                               # noqa: BLE001
        v = None
    import dftd3
    return dict(package="dftd3 (simple-dftd3 python API)", dist_version=v, module_version=getattr(dftd3, "__version__", None),
                params=D3_PARAMS, cutoff_bohr=D3_CUTOFF_BOHR, damping="rational (BJ)", three_body="off (s9 = 0)",
                kcal_per_eh=KCAL, bohr_per_angstrom=ANG2BOHR)


def d3_energy(syms, pos_A, params=None, cutoff=True):
    import numpy as np
    from dftd3.interface import DispersionModel, RationalDampingParam
    m = DispersionModel(np.array([SYM2Z[s] for s in syms]), np.asarray(pos_A, dtype=float) * ANG2BOHR)
    if cutoff:
        m.set_realspace_cutoff(**D3_CUTOFF_BOHR)
    p = RationalDampingParam(**(params or D3_PARAMS))
    return float(m.get_dispersion(p, grad=False)["energy"])


def d3_int(syms, pos, A, params=None, cutoff=True):
    """(E_TS, E_frag1, E_frag2 in Eh, D3_int kcal/mol) at the TS geometry; frag1 = A."""
    A = sorted(set(int(a) for a in A))
    B = [i for i in range(len(syms)) if i not in set(A)]
    e = d3_energy(syms, pos, params, cutoff)
    ea = d3_energy([syms[i] for i in A], [pos[i] for i in A], params, cutoff)
    eb = d3_energy([syms[i] for i in B], [pos[i] for i in B], params, cutoff)
    return e, ea, eb, (e - ea - eb) * KCAL


def cmd_d3(src):
    import pandas as pd
    ensure_env("reactot")
    S = src_cfg(src)
    recs, A = records(S)
    rows = []
    for rid in sorted(recs):
        try:
            syms, pos = read_xyz(S["final"] / str(rid) / "ts.xyz")
            e, ea, eb, di = d3_int(syms, pos, A[rid])
            rows.append(dict(rxn_id=rid, e_d3_ts_eh=e, e_d3_frag1_eh=ea, e_d3_frag2_eh=eb, D3_int=di, d3_status="ok"))
        except Exception as ex:                                     # noqa: BLE001
            rows.append(dict(rxn_id=rid, D3_int=float("nan"), d3_status=f"error:{type(ex).__name__}:{ex}"))
    D = pd.DataFrame(rows)
    M.write_parquet(S["d3"], D)
    jdump(rep_path(S, "D3"), dict(step="d3", source=src, n=len(D), status=D.d3_status.value_counts().to_dict(),
                                  versions=d3_versions(), written=now()))
    log(f"d3 {src}: {D.d3_status.value_counts().to_dict()}")
    return 0


# ================================================================ rung table (reactot env)
def _lad_inputs():
    if str(LAD_DIR) not in sys.path:
        sys.path.insert(0, str(LAD_DIR))
    import inputs as IN                                          # LADDER inputs.py: column names (read only)
    return IN


def token_rows(S, X, xcols):
    """= LADDER inputs.token_info on this source: T_report / Q_report links, npz sha256 = T_manifest, token glob = the
    X182 row exactly, token count = n_atoms -> {rxn_id: (path, sha256, n)}, info."""
    import numpy as np
    import pandas as pd
    res, tok = S["fu2_res"], S["fu2_scr"] / "tokens"
    tp, qp, mp = res / "T_report.json", res / "Q_report.json", res / "T_manifest.csv"
    if not all(p.is_file() for p in (tp, qp, mp)):
        return {}, dict(missing=[str(p) for p in (tp, qp, mp) if not p.is_file()])
    trep, qrep = M.read_json(tp), M.read_json(qp)
    av, tv = versions(S)
    bad = [m for ok, m in ((not trep.get("stop"), "T_report stop"), (not qrep.get("stop"), "Q_report stop"),
                           (trep.get("token_version") == tv, "token_version"), (qrep.get("atom_version") == av, "atom_version"),
                           (trep.get("manifest_sha256") == M.sha256(mp), "T_manifest sha256"),
                           (trep.get("q_report_sha256") == M.sha256(qp), "Q_report sha256"),
                           (trep.get("x182_sha256") == M.sha256(S["x182"]), "X182 sha256"),
                           (list(trep.get("glob_cols") or []) == list(xcols), "glob_cols")) if not ok]
    if bad:
        M.stop(f"{S['name']} tokens: {bad}")
    man = pd.read_csv(mp)
    xv = X[xcols]
    out, sbad, gbad = {}, [], []
    for rid, s0, n in zip(man.rxn_id.astype(int), man.sha256, man.n_atoms.astype(int)):
        p = tok / f"{rid}.npz"
        s = M.sha256(p) if p.is_file() else None
        if s != s0:
            sbad.append(int(rid))
            continue
        with np.load(p) as z:
            g, nz = np.asarray(z["glob"], float), len(z["Z"])
        if rid not in xv.index or nz != n or not np.array_equal(g, xv.loc[rid].to_numpy(float)):
            gbad.append(int(rid))
            continue
        out[int(rid)] = (str(p), s, int(n))
    if sbad or gbad:
        M.stop(f"{S['name']} tokens: {len(sbad)} npz sha256 != T_manifest {sbad[:5]}, {len(gbad)} glob != X182 / n_atoms "
               f"{gbad[:5]}")
    return out, dict(t_report=str(tp), t_report_sha256=M.sha256(tp), n_tokens=len(out), token_version=tv,
                     atom_version=av, elements=trep.get("elements"), ext_used=trep.get("ext_used"))


def refs_override(S, Hq, mode):
    """reference energies per (job, k) -> (override {job: {k: Eh}}, prov {(job, k): dict}, report).
    own: no replacement. cr2 (label source): refs_cr2 rows, own = this run's energy, Q / V = refs_cr2 e_hf_eh
    (= LADDER inputs.refs_dft provenance rules). min: refs_q.audit per Q-2 key over this source's rows, replace when
    above the key minimum, constrained keys keep their own (LADDER P6 + P20 option (c))."""
    import pandas as pd
    e0 = {}
    for j, e1, e2 in zip(Hq.job_id, Hq.e_frag1_rel_eh, Hq.e_frag2_rel_eh):
        e0[(j, 1)], e0[(j, 2)] = float(e1), float(e2)
    ov, prov = {}, {}
    if mode == "own":
        for (j, k), e in e0.items():
            prov[(j, k)] = dict(src="own", src_job=j, src_frag=k, key="", e_orig=e, e_new=e)
        return ov, prov, dict(mode="own", n_fragments=len(prov), n_replaced=0)
    if mode == "cr2":
        R = pd.read_parquet(LAD_REFS_CR2)
        R = R[R.job_id.isin(set(Hq.job_id))]
        for r in R.itertuples(index=False):
            j, k = r.job_id, int(r.frag)
            if (j, k) not in e0:
                continue
            if r.src == "own":
                prov[(j, k)] = dict(src="own", src_job=j, src_frag=k, key=r.key, e_orig=e0[(j, k)], e_new=e0[(j, k)])
                continue
            sj, sf = (r.src_job, int(r.src_frag)) if r.src == "Q" else (None, k)
            ov.setdefault(j, {})[k] = float(r.e_hf_eh)
            prov[(j, k)] = dict(src=r.src, src_job=sj, src_frag=sf, key=r.key, e_orig=e0[(j, k)], e_new=float(r.e_hf_eh))
        miss = sorted(set(e0) - set(prov))
        if miss:
            M.stop(f"refs_cr2.parquet lacks {len(miss)} fragments, e.g. {miss[:3]}")
        return ov, prov, dict(mode="cr2", source=str(LAD_REFS_CR2), sha256=M.sha256(LAD_REFS_CR2),
                              by_src=dict(Counter(p["src"] for p in prov.values())), n_jobs_replaced=len(ov))
    if mode == "min":
        if str(LAD_DIR) not in sys.path:
            sys.path.insert(0, str(LAD_DIR))
        import refs_q as Q
        K = Q.load_keys()
        K = K[K.job_id.isin(set(Hq.job_id))]
        A = Q.audit(K, e0, key="key")
        for r in A.itertuples(index=False):
            j, k = r.job_id, int(r.frag)
            rep_ = r.dev_kcal > 0 and not bool(r.constrained)
            prov[(j, k)] = dict(src="min" if rep_ else "own", src_job=r.src_job if rep_ else j,
                                src_frag=int(r.src_frag) if rep_ else k, key=r.key, e_orig=float(r.e_eh),
                                e_new=float(r.e_min_eh) if rep_ else float(r.e_eh))
            if rep_:
                ov.setdefault(j, {})[k] = float(r.e_min_eh)
        for (j, k), e in e0.items():                                # fragments without a Q-2 key keep their own
            prov.setdefault((j, k), dict(src="own", src_job=j, src_frag=k, key="", e_orig=e, e_new=e))
        return ov, prov, dict(mode="min", source=str(Q.LAD_KEYS), sha256=M.sha256(Q.LAD_KEYS), n_jobs_replaced=len(ov),
                              n_fragments_replaced=int(sum(len(v) for v in ov.values())),
                              rule="LADDER P6 (per Q-2 key lowest HF reference among this source's rows) + P20 (c)")
    M.stop(f"unknown refs mode {mode}")


def reref(rec, e1, e2):
    """d1, d2, barrier of build.parse_rxn with the reference energies e1, e2 (same expressions, same operand order)."""
    return dict(d1=(rec["e_frag1_dist_eh"] - e1) * KCAL, d2=(rec["e_frag2_dist_eh"] - e2) * KCAL,
                barrier=(rec["e_eda_eh"] - e1 - e2) * KCAL)


def _f(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def stage_status(S, tg, recs, F_why, tok, Hq_all, D3):
    """per target: the first stage without a result (H-2 accounting)."""
    import pandas as pd
    hst = {int(r): (s, str(w)) for r, s, w in zip(Hq_all.rxn_id, Hq_all.status, Hq_all.get("reason", Hq_all.status))} \
        if len(Hq_all) else {}
    d3 = dict(zip(D3.rxn_id.astype(int), D3.d3_status)) if len(D3) else {}
    rows = []
    for rid, job in tg:
        a = S["ade"] / job
        ade_ok = all((a / f".done_{s}").is_file() for s in ("ts", "stereo", "ref"))
        mk = None
        f = S["final"] / str(rid)
        if (f / ".done").is_file():
            mk = ".done"
        elif f.is_dir():
            fl = sorted(p.name for p in f.glob(".fail_*"))
            mk = fl[0] if fl else "no_marker"
        r = dict(rxn_id=rid, job_id=job, ade_folder=a.is_dir(), ade_ok=ade_ok, final=mk or "missing",
                 meta=rid in recs, x182=F_why.get(rid, "not_run") if rid in recs else "no_meta",
                 tok=rid in tok, hf=hst.get(rid, ("not_run", ""))[0], hf_reason=hst.get(rid, ("", ""))[1],
                 d3=d3.get(rid, "not_run"))
        stages = [("ade", r["ade_ok"]), ("finalize", mk == ".done"), ("meta", r["meta"]), ("x182", r["x182"] == "ok"),
                  ("tokens", r["tok"]), ("hf", r["hf"] == "ok"), ("d3", r["d3"] == "ok")]
        first = next((n for n, ok in stages if not ok), "")
        r["fail_stage"] = first
        r["fail_reason"] = {"": "", "ade": "no MLTS TS (A-2 stage markers)", "finalize": r["final"], "meta": "meta",
                            "x182": r["x182"], "tokens": "no token", "hf": f"{r['hf']}:{r['hf_reason']}",
                            "d3": r["d3"]}[first]
        rows.append(r)
    return pd.DataFrame(rows)


def cmd_build(src, refs=None):
    import numpy as np
    import pandas as pd
    ensure_env("reactot")
    IN = _lad_inputs()
    S = src_cfg(src)
    refs = refs or S["refs"]
    if src == "mlts" and refs not in ("own", "min"):
        M.stop("mlts references: own | min (refs_cr2 uses label-level DFT data)")
    tg = targets(S)
    recs, A = records(S)
    X = pd.read_parquet(S["x182"]) if S["x182"].is_file() else None
    if X is None:
        M.not_ready(f"{S['x182']} missing")
    xcols = [c for c in X.columns if c != "rxn_id"]
    X = X.assign(rxn_id=X.rxn_id.astype(int)).set_index("rxn_id")
    tok, tinfo = token_rows(S, X, xcols)
    Hall = pd.read_parquet(S["hf"] / "hf_records.parquet") if (S["hf"] / "hf_records.parquet").is_file() else \
        pd.DataFrame(columns=["rxn_id", "job_id", "status"])
    Hq = Hall[Hall.status == "ok"].copy()
    D3 = pd.read_parquet(S["d3"]) if S["d3"].is_file() else pd.DataFrame(columns=["rxn_id", "D3_int", "d3_status"])
    Fs = S["feat"] / "F_status.csv"
    F_why = dict(zip(*[pd.read_csv(Fs)[c] for c in ("rxn_id", "reason")])) if Fs.is_file() else {}
    F_why = {int(k): str(v) for k, v in F_why.items()}
    ov, prov, rrep = refs_override(S, Hq, refs)
    h0, hn = {}, {}
    hrec = {j: r for j, r in zip(Hq.job_id, Hq.to_dict("records"))}
    for j, r in hrec.items():
        h0[j] = {t: _f(r.get(t)) for t in H_TARGETS}
        h = dict(h0[j])
        o = ov.get(j) or {}
        if o:
            e1 = o.get(1, float(r["e_frag1_rel_eh"]))
            e2 = o.get(2, float(r["e_frag2_rel_eh"]))
            h.update({t: _f(v) for t, v in reref(r, e1, e2).items()})
        hn[j] = h
    xfin = pd.Series(np.isfinite(X[xcols].to_numpy(float)).all(1), index=X.index)
    d3 = dict(zip(D3.rxn_id.astype(int), D3.D3_int.astype(float))) if len(D3) else {}
    rows = []
    for rid, j in tg:
        t = tok.get(rid)
        r = dict(rxn_id=rid, job_id=j, has_tok=t is not None, tok_path=t[0] if t else "", tok_sha256=t[1] if t else "",
                 tok_n_atoms=t[2] if t else -1, has_x182=bool(rid in xfin.index and xfin.loc[rid]))
        a, b = hn.get(j), h0.get(j)
        r["has_h"] = bool(a is not None and b is not None and all(a[x] is not None for x in H_TARGETS)
                          and all(b[x] is not None for x in H_TARGETS))
        r["usable"] = bool(r["has_tok"] and r["has_x182"] and r["has_h"])
        for x in H_TARGETS:
            r[f"H_{x}"] = a[x] if a and a[x] is not None else np.nan
            r[f"H0_{x}"] = b[x] if b and b[x] is not None else np.nan
        r["h_changed"] = bool(r["has_h"] and any(r[f"H_{x}"] != r[f"H0_{x}"] for x in H_REREF))
        for k in (1, 2):
            p = prov.get((j, k)) or {}
            r.update({f"ref{k}_src": p.get("src", ""), f"ref{k}_src_job": p.get("src_job") or "",
                      f"ref{k}_src_frag": int(p["src_frag"]) if p.get("src_frag") is not None else -1,
                      f"ref{k}_key": p.get("key") or "",
                      f"e_ref{k}_hf_orig_eh": p["e_orig"] if p.get("e_orig") is not None else np.nan,
                      f"e_ref{k}_hf_eh": p["e_new"] if p.get("e_new") is not None else np.nan})
        r["D3_int"] = d3.get(rid, np.nan)
        rows.append(r)
    Mt = pd.DataFrame(rows).sort_values("rxn_id").reset_index(drop=True)
    Xr = X.reindex(Mt.rxn_id.to_numpy())[xcols].reset_index(drop=True)
    T = pd.concat([Mt[IN.META_COLS], Xr, Mt[IN.H_COLS + IN.H0_COLS + ["h_changed"] + IN.PROV_COLS + ["D3_int"]]], axis=1)
    import pyarrow.parquet as pq
    lad_cols = list(pq.read_schema(LAD_DFT_TABLE).names) if LAD_DFT_TABLE.is_file() else None
    cols_ok = lad_cols is not None and list(T.columns) == lad_cols + ["D3_int"]
    if not cols_ok:
        M.stop(f"rung table columns differ from {LAD_DFT_TABLE} + D3_int")
    out = S["table"] if refs == S["refs"] or S["table_min"] is None else S["table_min"]
    M.write_parquet(out, T)
    St = stage_status(S, tg, recs, F_why, tok, Hall, D3)
    M.write_csv(S["status"] if out == S["table"] else S["status"].with_name(S["status"].stem + "_min.csv"), St)
    rep = dict(step="build", source=src, refs=refs, references=rrep, written=now(), table=str(out),
               table_sha256=M.sha256(out), n_targets=len(T), n_tok=int(T.has_tok.sum()), n_x182=int(T.has_x182.sum()),
               n_h=int(T.has_h.sum()), n_usable=int(T.usable.sum()), n_h_changed=int(T.h_changed.sum()),
               n_d3=int(np.isfinite(T.D3_int.to_numpy(float)).sum()), tokens=tinfo,
               columns_equal_ladder_dft_plus_D3_int=cols_ok, fail_stage=St.fail_stage.value_counts().to_dict(),
               decisions=dict(refs=DECISION_REFS, partition=DECISION_PART, d3=DECISION_D3))
    if src == "mlts":
        rep["H2"] = h2(St)
    jdump(rep_path(S, "build" if out == S["table"] else "build_min"), rep)
    log(f"build {src} ({refs}): {rep['n_usable']} usable of {len(T)}; fail stages {rep['fail_stage']}")
    if src == "mlts" and rep["H2"]["stop"]:
        print(f"STOP: H-2 input failure rate {rep['H2']['rate']:.4f} > {H2_MAX}", file=sys.stderr)
        return 3
    return 0


DECISION_REFS = ("mlts: own MLTS-refined references (H = H0); refs_cr2 (own/Q/V, label-level DFT data) not applied; "
                 "--refs min = LADDER P6 + P20(c) on this source's own HF energies; label source (H-1): refs_cr2 "
                 "exactly as LADDER")
DECISION_PART = ("redetect (feat_lf): fragmenter.partition must reproduce ts_result.json A_idx, else dropped (F-2); "
                 "HF tags / D3_int use that A_idx")
DECISION_D3 = "B3LYP-D3(BJ) two-body (s9 0), DFTD3 3.1 cutoffs, simple-dftd3 python, kcal = Eh x 627.5094740631"


def h2(St):
    """H-2 (spec 7): among reactions with an MLTS TS (A-2 folder with its three stage markers) the share without a
    usable input row; > 2 % = STOP. Also the counts by first failing stage."""
    have = St[St.ade_ok]
    bad = have[have.fail_stage != ""]
    n = len(have)
    rate = len(bad) / n if n else None
    return dict(rule="failures / reactions with an MLTS TS; usable = token + finite X182 + 9 finite HF + D3",
                n_targets=int(len(St)), n_with_ts=n, n_fail=int(len(bad)), rate=rate, max=H2_MAX,
                by_stage=bad.fail_stage.value_counts().to_dict(),
                by_reason=bad.fail_reason.astype(str).value_counts().head(30).to_dict(),
                stop=bool(rate is not None and rate > H2_MAX), judged=bool(n))


def cmd_h2(src="mlts"):
    import pandas as pd
    ensure_env("reactot")
    S = src_cfg(src)
    if not S["status"].is_file():
        M.not_ready(f"{S['status']} missing (build {src})")
    r = h2(pd.read_csv(S["status"]))
    r.update(source=src, status_csv=str(S["status"]), status_sha256=M.sha256(S["status"]), written=now())
    jdump(M.RES / "H2_report.json", r)
    log(f"H-2 {src}: {r}")
    return 3 if r["stop"] else 0


# ================================================================ H-0 (reactot env)
def cmd_h0(n=100):
    """100 label TSs (seeded M.SEED + 1, 20 per dipole, among dft_final .done): D3_int vs the label disp channel."""
    import numpy as np
    import pandas as pd
    ensure_env("reactot")
    n = int(n)
    lab = {int(d["rxn_id"]): d for d in json.loads(M.LABELS.read_text())}
    A = label_A()
    ids = sorted(r for r in lab if (M.LAD / "dft_final" / str(r) / ".done").is_file() and r in A)
    rng = np.random.default_rng(M.SEED + 1)
    pick = []
    for d in M.DIPOLES:
        c = sorted(r for r in ids if lab[r]["dipole"] == d)
        pick += sorted(int(x) for x in rng.choice(c, size=n // len(M.DIPOLES), replace=False))
    rows = []
    for rid in pick:
        syms, pos = read_xyz(M.LAD / "dft_final" / str(rid) / "ts.xyz")
        e, ea, eb, di = d3_int(syms, pos, A[rid])
        _, _, _, di_meth = d3_int(syms, pos, A[rid], params=None, cutoff=False)
        p3 = dict(D3_PARAMS, s9=1.0)
        _, _, _, di_atm = d3_int(syms, pos, A[rid], params=p3)
        L = lab[rid]
        rows.append(dict(rxn_id=rid, job_id=L["job_id"], dipole=L["dipole"], n_atoms=len(syms), D3_int=di,
                         disp_dft=float(L["disp_dft"]), disp_raw_kcal=float(L["eda_raw_eh"]["disp"]) * KCAL,
                         diff=di - float(L["disp_dft"]), diff_raw=di - float(L["eda_raw_eh"]["disp"]) * KCAL,
                         D3_int_default_cutoff=di_meth, D3_int_with_atm=di_atm))
    D = pd.DataFrame(rows)
    med = float(D["diff"].abs().median())
    rule = "disp_predicted_by_D3_int" if med <= H0_RULE_KCAL else "D3_int_input_column_only"
    rep = dict(spec="DEPLOY_V2_MLTS.md 7 H-0", written=now(), n=len(D), sample=f"seed {M.SEED + 1}, "
               f"{n // len(M.DIPOLES)} per dipole among $LAD/dft_final .done", label="labels_d1_cr2.json disp_dft (kcal/mol)",
               structure="$LAD/dft_final/<rxn_id>/ts.xyz (label TS = D1-HF eda.inp coordinates), partition = "
                         "input_meta_d1_g2.csv A_idx (label eda.inp (1) tags)",
               median_abs_diff=med, max_abs_diff=float(D["diff"].abs().max()), mean_diff=float(D["diff"].mean()),
               median_abs_diff_vs_eda_raw_eh=float(D.diff_raw.abs().median()),
               max_abs_diff_vs_eda_raw_eh=float(D.diff_raw.abs().max()),
               cutoff_effect_max_kcal=float((D.D3_int - D.D3_int_default_cutoff).abs().max()),
               atm_effect_median_kcal=float((D.D3_int_with_atm - D.D3_int).abs().median()),
               by_dipole={d: float(g["diff"].abs().median()) for d, g in D.groupby("dipole")},
               rule=f"median |D3_int - disp_dft| <= {H0_RULE_KCAL} kcal/mol -> disp predicted by D3_int directly (E reports "
                    f"both); else D3_int is an input column only", decision=rule, versions=d3_versions(),
               rows=D.to_dict("records"))
    jdump(M.RES / "H0_report.json", _clean(rep))
    log(f"H-0: median |diff| {med:.3g} kcal/mol, max {rep['max_abs_diff']:.3g} -> {rule}")
    return 0


# ================================================================ H-1 = C-5 (reactot env)
def _maxabs(a, b):
    import numpy as np
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or not np.array_equal(np.isnan(a), np.isnan(b)):
        return float("inf")
    m = ~np.isnan(a)
    return float(np.abs(a[m] - b[m]).max()) if m.any() else 0.0


TOKEN_FLOAT = ("bin", "num", "pos", "glob")
TOKEN_EXACT = ("Z", "edges", "etype", "n_bonds_rel")


def cmd_h1():
    """H-1 (= C-5): the label source through this file's code path vs the LADDER dft-rung inputs (module docstring)."""
    import numpy as np
    import pandas as pd
    ensure_env("reactot")
    S = src_cfg("label")
    tg = targets(S)
    rids = [r for r, _ in tg]
    job = dict(tg)
    rep = dict(spec="DEPLOY_V2_MLTS.md 7 H-1 = 8 C-5", written=now(), tol=TOL_H1, n=len(rids), sample=str(S["targets"]),
               definition="same structures (label TS + label references of $LAD/dft_final), same partition (label "
                          "A_idx; this path re-detects it and drops a reaction when fragmenter.partition differs), same "
                          "reference energies (refs_cr2 for H; own for H0) in -> the same X182 / token / HF numbers "
                          "out, through finalize_lf -> meta_lf -> feat_lf (redetect) -> fu2 atoms / tokens -> D1-HF "
                          "inputs (this file) + run.py + build.parse_rxn -> build (this file)",
               compared={})
    gates = {}
    # ---- (a) finalize: xTB Freq at the label TS vs $LAD/dft_final (explanatory)
    fa = []
    for rid in rids:
        mine, lad = S["final"] / str(rid), M.LAD / "dft_final" / str(rid)
        r = dict(rxn_id=rid, done=(mine / ".done").is_file())
        if r["done"]:
            rm, rl = M.read_json(mine / "result.json"), M.read_json(lad / "result.json")
            r["hess_bytes_equal"] = (mine / "work" / "ts.hess").read_bytes() == (lad / "work" / "ts.hess").read_bytes()
            im, il = rm.get("imag_freqs_cm") or [], rl.get("imag_freqs_cm") or []
            r["n_imag_equal"] = len(im) == len(il)
            r["d_imag1_cm"] = abs(im[0] - il[0]) if im and il else None
            r["d_share"] = abs(rm["imag_mode_forming_share"] - rl["imag_mode_forming_share"]) \
                if rm.get("imag_mode_forming_share") is not None and rl.get("imag_mode_forming_share") is not None else None
            r["ts_equal"] = read_xyz(mine / "ts.xyz") == read_xyz(lad / "ts.xyz")
        fa.append(r)
    F = pd.DataFrame(fa)
    rep["compared"]["finalize"] = dict(n_done=int(F.done.sum()), n_hess_bytes_equal=int(F.get("hess_bytes_equal", pd.Series(dtype=bool)).fillna(False).sum()),
                                       n_ts_equal=int(F.get("ts_equal", pd.Series(dtype=bool)).fillna(False).sum()),
                                       max_d_imag1_cm=qstats(F.get("d_imag1_cm", pd.Series(dtype=float)).tolist()).get("max"),
                                       max_d_share=qstats(F.get("d_share", pd.Series(dtype=float)).tolist()).get("max"),
                                       note="explanatory (B6 inputs); not an H-1 gate")
    # ---- (b) meta records vs the LADDER F0 label rows / label A_idx
    recs, A = records(S)
    LL = {int(d["rxn_id"]): d for d in json.loads(LABEL_LABELS.read_text()) if int(d["rxn_id"]) in set(rids)}
    LA = label_A()
    keys = ("n_atoms", "n_f1", "n_f2", "formed_pairs_ts", "alt_used", "flag_async", "flag_foreign_bond", "charge1",
            "charge2", "role1", "role2")
    mb = {k: sum(1 for r in rids if r in recs and str(recs[r][k]) == str(LL[r][k])) for k in keys}
    rep["compared"]["meta"] = dict(n_records=sum(r in recs for r in rids), equal_by_key=mb,
                                   A_idx_equal=sum(1 for r in rids if A.get(r) == LA.get(r)))
    gates["meta"] = all(v == len(rids) for v in mb.values()) and rep["compared"]["meta"]["A_idx_equal"] == len(rids)
    # ---- (c) X182 vs $LAD/X182_d1_dft.parquet
    Fr = M.read_json(rep_path(S, "F"))
    X = pd.read_parquet(S["x182"]).set_index("rxn_id")
    XL = pd.read_parquet(LAD_X182).set_index("rxn_id")
    xcols = [c for c in XL.columns]
    common = [r for r in rids if r in X.index and r in XL.index]
    d = np.abs(X.loc[common, xcols].to_numpy(float) - XL.loc[common, xcols].to_numpy(float)) if common else np.zeros((0, 182))
    redet = Fr.get("F2_redetect") or {}
    rep["compared"]["x182"] = dict(n_mine=int(sum(r in X.index for r in rids)), n_ladder=int(sum(r in XL.index for r in rids)),
                                   n_common=len(common), columns_equal=list(X.columns) == xcols,
                                   max_abs=float(d.max()) if d.size else None,
                                   n_bit_equal_rows=int((d == 0).all(1).sum()) if d.size else 0,
                                   worst_columns={c: float(v) for c, v in sorted(zip(xcols, d.max(0)), key=lambda t: -t[1])[:8]}
                                   if d.size else {},
                                   missing_in_mine={str(r): (Fr.get("not_ok") or {}).get(str(r)) for r in rids
                                                    if r not in X.index},
                                   redetect_differs_from_label_partition=dict(
                                       n=int(redet.get("n", 0)), statuses=redet.get("statuses"), rxn=redet.get("rxn"),
                                       note="explanatory: feat_lf redetect on the label TS vs the label A_idx"))
    gates["x182"] = bool(common and len(common) == len(rids) and rep["compared"]["x182"]["columns_equal"]
                         and d.max() <= TOL_H1)
    # ---- (d) tokens vs $LAD/fu2/tokens
    tmax = {k: 0.0 for k in TOKEN_FLOAT}
    texact = {k: 0 for k in TOKEN_EXACT}
    n_tok = 0
    tmiss = []
    for rid in rids:
        a, b = S["fu2_scr"] / "tokens" / f"{rid}.npz", LAD_TOKENS / f"{rid}.npz"
        if not (a.is_file() and b.is_file()):
            tmiss.append(rid)
            continue
        n_tok += 1
        with np.load(a) as za, np.load(b) as zb:
            for k in TOKEN_FLOAT:
                tmax[k] = max(tmax[k], _maxabs(za[k], zb[k]))
            for k in TOKEN_EXACT:
                texact[k] += int(np.array_equal(za[k], zb[k]))
    rep["compared"]["tokens"] = dict(n_compared=n_tok, missing=tmiss, max_abs=tmax, n_exact_equal=texact,
                                     token_version=dict(mine=versions(S)[1], ladder="fu2-tokens-3-dft"))
    gates["tokens"] = bool(n_tok == len(rids) and all(v <= TOL_H1 for v in tmax.values())
                           and all(v == n_tok for v in texact.values()))
    # ---- (e) HF inputs (bytes vs the D1-HF dft-rung inputs) and HF 9 values vs labels_d1_hf (geom dft)
    nb, nbad = 0, []
    for rid in rids:
        for c in CALCS:
            a, b = S["hf"] / "inputs" / job[rid] / "g2" / f"{c}.inp", D1HF_SCR / "inputs" / job[rid] / "hf" / f"{c}.inp"
            if a.is_file() and b.is_file() and a.read_bytes() == b.read_bytes():
                nb += 1
            else:
                nbad.append(f"{job[rid]}/{c}")
    Hm = pd.read_parquet(S["hf"] / "hf_records.parquet")
    Hm = Hm[Hm.job_id.isin([job[r] for r in rids])].set_index("job_id")
    HL = pd.read_parquet(D1HF_LABELS)
    HL = HL[(HL.geom == "dft") & HL.job_id.isin([job[r] for r in rids])].set_index("job_id")
    jj = [job[r] for r in rids if job[r] in Hm.index and Hm.at[job[r], "status"] == "ok" and job[r] in HL.index]
    ecols = list(H_TARGETS) + ["e_frag1_dist_eh", "e_frag2_dist_eh", "e_frag1_rel_eh", "e_frag2_rel_eh", "e_eda_eh", "bond",
                               "eint_spe", "bsse_shift_kcal"]
    hd = {}
    for c in ecols:
        if c in Hm.columns and c in HL.columns and jj:
            v = _maxabs(Hm.loc[jj, c], HL.loc[jj, c])
            hd[c] = v * KCAL if c.endswith("_eh") else v
    rep["compared"]["hf"] = dict(n_input_files_byte_equal=nb, n_input_files=len(rids) * len(CALCS), not_equal=nbad[:20],
                                 n_status_ok=int((Hm.status == "ok").sum()), n_compared=len(jj),
                                 max_abs_kcal=hd, note="*_eh columns converted to kcal/mol; mine = this rerun "
                                                       "(HF/6-31G* serial ORCA 6.1.1) vs analysis/d1_hf/results/"
                                                       "labels_d1_hf.parquet geom dft")
    gates["hf"] = bool(len(jj) == len(rids) and hd and all(np.isfinite(v) and v <= TOL_H1 for v in hd.values()))
    # ---- (f) rung table vs $LAD/inputs/dft.parquet (H with refs_cr2, H0 own, provenance, flags)
    T = pd.read_parquet(S["table"]).set_index("rxn_id").loc[rids]
    TL = pd.read_parquet(LAD_DFT_TABLE).set_index("rxn_id").loc[rids]
    IN = _lad_inputs()
    flags = {c: int((T[c] == TL[c]).sum()) for c in ("has_tok", "has_x182", "has_h", "usable", "tok_n_atoms", "h_changed")}
    hcols = IN.H_COLS + IN.H0_COLS
    hv = {c: _maxabs(T[c], TL[c]) for c in hcols}
    prov_s = {c: int((T[c].astype(str) == TL[c].astype(str)).sum()) for c in IN.PROV_COLS if not c.startswith("e_ref")}
    prov_e = {c: _maxabs(T[c], TL[c]) * KCAL for c in IN.PROV_COLS if c.startswith("e_ref")}
    x_t = _maxabs(T[xcols].to_numpy(float), TL[xcols].to_numpy(float))
    rep["compared"]["table"] = dict(flags_equal=flags, h_max_abs_kcal=hv, prov_strings_equal=prov_s,
                                    prov_energies_max_abs_kcal=prov_e, x182_max_abs=x_t,
                                    columns_equal_plus_D3_int=list(T.reset_index().columns) ==
                                    list(pd.read_parquet(LAD_DFT_TABLE).columns) + ["D3_int"],
                                    note="tok_path / tok_sha256 differ by construction (other root, token_version tag)")
    gates["table"] = bool(all(v == len(rids) for v in flags.values()) and all(v <= TOL_H1 for v in hv.values())
                          and all(v == len(rids) for v in prov_s.values()) and all(v <= TOL_H1 for v in prov_e.values())
                          and x_t <= TOL_H1 and rep["compared"]["table"]["columns_equal_plus_D3_int"])
    # ---- (g) D3_int vs the label disp channel on the same 50 (explanatory; H-0 is the rule)
    lab = {int(d["rxn_id"]): d for d in json.loads(M.LABELS.read_text()) if int(d["rxn_id"]) in set(rids)}
    dd = (T.D3_int - pd.Series({r: float(lab[r]["disp_dft"]) for r in rids})).abs()
    rep["compared"]["d3_vs_disp_dft"] = dict(median_abs_kcal=float(dd.median()), max_abs_kcal=float(dd.max()))
    rep["gates"] = gates
    rep["pass"] = bool(all(gates.values()))
    rep["gate_rule"] = (f"meta keys equal; X182 / token float arrays / HF 9 values + energies / table H, H0, e_ref: max "
                        f"|diff| <= {TOL_H1}; token integer arrays, table flags and provenance strings equal; every one "
                        f"of the {len(rids)} reactions present in each comparison")
    jdump(M.RES / "H1_report.json", _clean(rep))
    log(f"H-1: gates {gates} -> pass {rep['pass']}")
    if not rep["pass"]:
        print("STOP: H-1 (C-5) reproduction failed: results/H1_report.json", file=sys.stderr)
        return 3
    return 0


# ================================================================ main
def main(argv):
    a, root, ids = ids_arg(list(argv))
    _OPT.update(root=root, ids=ids)
    if not a:
        print(__doc__)
        return 2
    c, r = a[0], a[1:]
    table = {
        "label_src": (cmd_label_src, (0, 1)),
        "smoke_src": (cmd_smoke_src, (0, 1)),
        "finalize": (cmd_finalize, (3, 4)),
        "meta": (cmd_meta, (1, 1)),
        "x182": (cmd_x182, (3, 3)),
        "x182_merge": (cmd_x182_merge, (1, 1)),
        "atoms": (cmd_atoms, (3, 3)),
        "tokens": (cmd_tokens, (1, 1)),
        "hf_inputs": (cmd_hf_inputs, (1, 1)),
        "hf_run": (cmd_hf_run, (3, 4)),
        "hf_build": (cmd_hf_build, (1, 1)),
        "d3": (cmd_d3, (1, 1)),
        "h0": (cmd_h0, (0, 1)),
        "h1": (cmd_h1, (0, 0)),
        "h2": (cmd_h2, (0, 1)),
    }
    if c == "build":
        refs = None
        if "--refs" in r:
            k = r.index("--refs")
            refs = r[k + 1]
            r = r[:k] + r[k + 2:]
        if len(r) != 1:
            print(__doc__)
            return 2
        return cmd_build(r[0], refs)
    if c == "_fu2":
        return _child_fu2(r[0], r[1:])
    if c not in table or not (table[c][1][0] <= len(r) <= table[c][1][1]):
        print(__doc__)
        return 2
    if r and c not in ("label_src", "smoke_src", "h0") and r[0] not in ("mlts", "label"):
        print(f"unknown source {r[0]}")
        return 2
    return table[c][0](*r)


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except SystemExit:
        raise
    except Exception:                                               # noqa: BLE001
        traceback.print_exc()
        sys.exit(3)
