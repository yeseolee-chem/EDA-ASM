#!/usr/bin/env python3
r"""sample.py — D1-G3 Phase 0 (start conditions, docs/specs/D1G3.md 1) and Phase S (sample, spec 2). sbatch only;
nothing in this file was run when it was written (login node), so every path is untested until it runs inside a job.

CLI (env reactot; submit from analysis/d1_g3 through jobs/g3.slurm with G3ENV=reactot; exit 0 = done, 2 = not ready /
usage, 3 = STOP)
  python sample.py start    spec 1 0-1 .. 0-3 -> results/start_report.json (always written; exit 3 if any fails)
  python sample.py sample   spec 2 -> results/S_pairs.csv, results/S_rxns.csv, results/S_report.json (needs a passing
                            results/start_report.json: missing -> exit 2, stop -> exit 3). Deterministic; a rerun
                            rewrites the same files (atomic writes, report last).

sbatch (login node: one-shot commands only; first `squeue -u $USER -h -r | wc -l` (N + 1 <= 20) and
`sinfo -p cpu1,cpu2 -o "%.9P %.6t %C"`):
  cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction-d1g3/analysis/d1_g3
  sbatch -p cpu1,cpu2 -J g3_S --time=48:00:00 --cpus-per-task=2 --mem=8G --export=ALL,G3ENV=reactot \
      jobs/g3.slurm sample.py start ::: sample.py sample
  (log $G3/logs/g3_S_<jobid>.out; ' ::: ' = sample only if start exited 0)

Start conditions (spec 1; any failure = STOP, spec 12 '0')
  0-1  D1LADDER results committed in d1-ladder dcd83ab3 (an ancestor of HEAD) and unchanged in this worktree:
       results/{L_predictions.parquet, L_report.json, L_val.csv, L_xgb_report.json, E1_report.json, E2_report.json};
       sha256 links L_predictions / L_val / L_report = L_report / E2_report.l_report, L_xgb_report = L_report.xgb_check;
       L_report stage 2, rungs dft g1 g2, 120 units, models AW GNN TF XGB; E1 phase E1, E2 phase E2 stage 2;
       E2 representatives for every rung x arm x target (10) with a known model. Unit cache $LAD/units: the 120 unit
       names of E2_report.units (= L_report.units), each <unit>.json / .parquet sha256 = E2_report.units; atom units
       (AW, GNN, TF: 90) each exactly __s0..s4.pt; XGB units (30): json + parquet only, no model file (recorded, not a
       failure: spec T-4 drops XGB from E and reports it); the XGB representatives are listed.
  0-2  D1-G2: analysis/d1_g2/results/C_compare.csv unchanged since dcd83ab3, sha256 = the ADD3 supplement pin
       (prereg_d1pair_add3.json supplement.same_ts_split.source_sha256), columns present; $G2/g2_final a folder; the
       Phase S population (below) builds; every population reaction has g2_status ok and a same_ts_g1 judgment, a
       $G2/g2_final/<rxn_id>/ with .done, ts.xyz, rel1.xyz, rel2.xyz, result.json and no .fail_*, and its D1 job copy
       d1pair_common.D1_JOBS/<job_id>/ (d1pair_common.JOB_FILES).
  0-3  reused code unchanged since dcd83ab3 (analysis/d1_ladder, d1_hf, d1_g2, d1_pair, espley_xtb_repro); ORCA = the
       D1 config orca_bin (= nurion/env.sh ORCA_BIN = hf_common.ORCA_BIN) runs a serial H2 HF/STO-3G single point in
       node-local $TMPDIR and prints Program Version 6.1.1 / GIT 487d211c / ORCA TERMINATED NORMALLY; xtb = the D1
       config xtb_bin (= env.sh XTB_BIN) --version 6.7.1; root entry points present (source parsed with ast, nothing
       imported): d1_features.py cmd_runroot(root, out, sel) / cmd_mergeroot(out, x182_path, report), fu2/atoms.py
       root_bind / cmd_root / cmd_root_tokens + ROOT_USAGE, D1-HF prep.py cmd_geominputs(geom, lst) / g1_input, run.py
       cmd_run / run_rxn, build.py parse_rxn and the hf_common environment roots D1HF_SCRATCH / D1HF_RES /
       D1HF_TEMPLATES / D1HF_G2_ROOT; g0f0.py g0_modules / hess_atoms / feat_env / atoms_env and inputs.py RUNGS /
       assemble_rung / refs_min / token_info / make_table. The g0f0.py / inputs.py CLIs themselves are fixed to the
       D1LADDER roots: the strict reading of 0-3 is NOT met; start_report.json records it under 0-3 strict_reading and
       'deviations' as a STOP condition overridden by decision 0-3 (0-3 ok = every other 0-3 check).

Phase S (spec 2; rule fixed, spec 11)
  population  pairs of $LAD/inputs/sets/P_012.csv (inputs.read_set: sha256 = results/P_report.json) whose S_folds.csv
              row has in_P_main, ref_kind == 'reference' and pos == 'p'; S_folds.csv sha256 = the P_report inputs value;
              the P_012 columns (inputs.SET_COLS) = the S_folds ones; count = P_report sets.P_012.in_P_main_para;
              fold = S_folds fold_T1, which must be the fold whose T1 test list (evaluate.test_lists_all: the
              $LAD/inputs/fu2_splits copies, sha256 = P_report) holds the pair.
  classes     evaluate.read_c_compare / pair_classes (D1LADDER E2 SAME_TS_RULE) on column same_ts_g1: reaction True =
              'same', False = 'different'; pair = same (both ends) / different (neither) / mixed.
  draws       one numpy.random.default_rng(20261024) in this order: Part A 'same' (dipoles sorted by name, 16 each),
              Part A 'different' (same order, 4 each), Part B (same order, 8 each). Per stratum the candidates are
              sorted (pair_id string / rxn_id integer) and rng.choice(n, size=min(k, n), replace=False) gives the
              positions (no call when n = 0); shortfalls are kept and reported.
  Part B pool population reactions (a and b ends of population pairs, any pair class) with same_ts_g1 False; dipole =
              the pair dipole (= C_compare dipole, checked); overlap with Part A allowed (spec 2).
  outputs     results/S_pairs.csv  pair_id, a_rxn, b_rxn, fold, class, dipole, a_job, b_job, core, fam (pair_id order)
              results/S_rxns.csv   rxn_id, job_id, dipole, core, ts_class, roles, part_a, part_b, partA_pairs,
                                   partA_classes, partA_ends (rxn_id order)
              results/S_report.json  population counts, draws (raw positions), shortfalls, unique reactions by part /
                                   class / dipole, folds, input / output sha256, decisions
              every sampled reaction must have its G2 folder files and D1 job copy (0-2 rule), else STOP.

Decisions (user: every judgment = the recommended option): DECISIONS below, also in results/DECISIONS.md
(g3_common.record_decision under the shared fcntl lock $G3/locks/DECISIONS.lock, written when a command starts) and in
the reports.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True        # no __pycache__ in the read-only analysis/d1_ladder, d1_pair, d1_hf trees (spec 8)

import ast  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import platform  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import g3_common as C  # noqa: E402   (puts analysis/d1_ladder on sys.path: L = lad_common, DP = d1pair_common)

DP = C.DP
import evaluate as EV  # noqa: E402   D1LADDER E2: read_c_compare, pair_classes (SAME_TS_RULE), test_lists_all, C_COMPARE
import inputs as IN  # noqa: E402     D1LADDER Phase P: p_report, read_set, SET_COLS, S_FOLDS, SPLITS_COPY

for _m, _f in ((EV, "evaluate.py"), (IN, "inputs.py")):
    if Path(_m.__file__).resolve() != (C.LAD_DIR / _f).resolve():
        C.stop(f"module {_m.__name__} imported from {_m.__file__}, not {C.LAD_DIR / _f}")

# ---------------------------------------------------------------- constants
LADDER_COMMIT = "dcd83ab3"                               # spec header: d1-ladder HEAD the branch d1-g3 was made from
START = C.RES / "start_report.json"
S_PAIRS, S_RXNS, S_REPORT = C.RES / "S_pairs.csv", C.RES / "S_rxns.csv", C.RES / "S_report.json"
SET = "P_012"
SAME_COL = EV.SAME_TS_COLS[0]                            # spec 2 'G2 대 G1 같은 TS 판정' = E2 primary judgment
assert SAME_COL == "same_ts_g1", SAME_COL
N_DIPOLES = 5
PART_A = (("same", 16), ("different", 4))                # spec 2: 80 = 5 x 16, 20 = 5 x 4 (draw order)
N_PARTB = 8                                              # spec 2: 40 = 5 x 8
PAIR_CLASSES = ("same", "mixed", "different", "unclassified")
RUNGS, ARMS, ATOM_MODELS, XGB, NF, N_SEEDS = ("dft", "g1", "g2"), ("S", "SH"), ("AW", "GNN", "TF"), "XGB", 5, 5
LAD_FILES = ("L_predictions.parquet", "L_report.json", "L_val.csv", "L_xgb_report.json", "E1_report.json",
             "E2_report.json")
UNITS = C.LAD / "units"
G2_FINAL = C.G2 / "g2_final"
G2_FILES = (".done", "ts.xyz", "rel1.xyz", "rel2.xyz", "result.json")
C_COMPARE = C.G2_RES / "C_compare.csv"
C_COLS = ("rxn_id", "job_id", "g2_status", "same_ts_g1", "dipole")
ADD3_JSON = C.LAD_RES / "prereg_d1pair_add3.json"
D1_CONFIG = DP.D1_RUN / "config_run.yaml"
D1_ENV_SH = DP.D1_RUN / "nurion" / "env.sh"
REUSED_TREES = ("analysis/d1_ladder", "analysis/d1_hf", "analysis/d1_g2", "analysis/d1_pair",
                "analysis/espley_xtb_repro")
XTB_VERSION = "6.7.1"
ORCA_PROBE_INP = "! HF STO-3G TightSCF NoSym\n%maxcore 500\n* xyz 0 1\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n*\n"
# root entry points (0-3): file -> {function: leading parameter names}
ENTRY = {
    C.P_DIR / "d1_features.py": {"cmd_runroot": ["root", "out", "sel"], "cmd_mergeroot": ["out", "x182_path", "report"]},
    C.P_DIR / "fu2" / "atoms.py": {"root_bind": [], "cmd_root": ["a"], "cmd_root_tokens": []},
    C.HF_DIR / "prep.py": {"cmd_geominputs": ["geom", "lst"], "g1_input": ["T", "rxn_id", "geom"]},
    C.HF_DIR / "run.py": {"cmd_run": ["lst", "geom", "level", "n", "i", "m"], "run_rxn": ["T", "geom", "level", "root"]},
    C.HF_DIR / "build.py": {"parse_rxn": ["T", "geom", "level"]},
    C.LAD_DIR / "g0f0.py": {"g0_modules": ["rec"], "hess_atoms": ["path"], "feat_env": [], "atoms_env": []},
    C.LAD_DIR / "inputs.py": {"assemble_rung": ["rung", "Hq", "override", "rxn_of", "t0"],
                              "refs_min": ["rung", "Hq", "rep_out", "g2_q_keys"], "token_info": ["rung", "tg1"],
                              "make_table": ["rung", "Hq", "Hn", "prov", "tok", "X", "xcols"]},
}
ENTRY_TEXT = {C.P_DIR / "fu2" / "atoms.py": ["ROOT_USAGE", "root run", "root gate", "root tokens", "FU2_ATOM_VERSION"],
              C.P_DIR / "d1_features.py": ['cmd == "runroot"', 'cmd == "mergeroot"'],
              C.HF_DIR / "hf_common.py": ['"D1HF_SCRATCH"', '"D1HF_RES"', '"D1HF_TEMPLATES"', '"D1HF_G2_ROOT"'],
              C.HF_DIR / "prep.py": ['["geominputs"]'],
              C.HF_DIR / "run.py": ['("dft", "g1", "g2")']}
# CLI facts recorded for 0-3 (module-level assignments, unparsed from the source)
FIXED_ROOTS = {C.LAD_DIR / "g0f0.py": ("DFT_FINAL", "HF_INPUTS", "F0_DIR", "X182_DFT", "VERSIONS", "FU2_SCR"),
               C.LAD_DIR / "inputs.py": ("RUNG_ORDER", "INP")}

DECISIONS = {
    "0-1": "D1LADDER 결과 = dcd83ab3에 commit된 analysis/d1_ladder/results의 L_predictions.parquet, L_report.json, "
           "L_val.csv, L_xgb_report.json, E1_report.json, E2_report.json이 이 작업 트리에서 그대로이고 sha256 연결"
           "(L_report / E2_report.l_report, L_report.xgb_check)이 맞으면 '있다'. unit 캐시 = E2_report.units 120개의 "
           "json·parquet sha256 일치 + atom unit 90개(AW, GNN, TF)마다 __s0..s4.pt 5개. XGB 30 unit은 json·parquet만 "
           "있고 모델 파일이 없다: 0-1 실패가 아니라 기록한다(T-4: XGB는 E에서 빼고 보고). XGB 대표(rung·arm·타깃)는 "
           "start_report에 나열한다.",
    "0-2": "D1-G2 결과 = C_compare.csv가 dcd83ab3과 같고 sha256이 ADD3 supplement 고정값과 같으며, Phase S 모집단의 "
           "모든 반응이 g2_status ok, same_ts_g1 판정 있음, $G2/g2_final/<rxn>/{.done, ts.xyz, rel1.xyz, rel2.xyz, "
           "result.json}(.fail_* 없음), D1 job 사본(d1pair_common.JOB_FILES)을 가지면 '있다'. 판정 없는 반응이 하나라도 "
           "있으면 0-2 실패.",
    "0-3": "g0f0.py CLI는 $LAD/dft_final, 라벨 TS 좌표(D1-HF dft 입력), -dft 버전에, inputs.py build는 rung dft|g1|g2와 "
           "$LAD/inputs에 고정돼 CLI 자체에는 루트 인자가 없다. 루트 인자는 그 아래 단계에 있다: d1_features.py "
           "runroot <root> / mergeroot, fu2/atoms.py root run|gate|tokens(G1_ROOT, FU2_SCR, FU2_RES, FU2_X182, "
           "FU2_*_VERSION), D1-HF prep.py geominputs + run.py(D1HF_G2_ROOT, D1HF_SCRATCH, D1HF_RES, D1HF_TEMPLATES; "
           "D1-G2 Phase H 선례). 엄격하게 읽으면 0-3(STOP 조건)은 맞지 않는다. 추천안: 이 진입점과 g0f0(g0_modules, "
           "hess_atoms, feat_env, atoms_env)·inputs(RUNGS, assemble_rung, refs_min, token_info, make_table) 함수가 "
           "dcd83ab3과 같은 코드로 있으면 진행한다; Phase F의 feat.py가 이 함수들을 G3 루트로 import·호출하고, G 작업 "
           "루프만 g0f0.g0_one을 검토해 옮긴 것을 쓴다(feat.py F-2). start_report.json에는 0-3 ok(나머지 검사)와 함께 "
           "strict_reading·deviations에 'STOP 조건을 결정 0-3으로 넘김'으로 적는다.",
    "0-3-bin": "ORCA·xtb는 D1 run config_run.yaml의 orca_bin·xtb_bin(= nurion/env.sh, = hf_common.ORCA_BIN)을 계산 "
               "노드에서 실제로 부른다: ORCA는 노드 로컬 TMPDIR에서 H2 HF/STO-3G 직렬 단일점 1개 -> 출력의 Program "
               "Version 6.1.1, GIT 487d211c, ORCA TERMINATED NORMALLY; xtb는 --version = 6.7.1.",
    "S-1": "모집단 = $LAD/inputs/sets/P_012.csv(sha256 = P_report) 짝 중 S_folds.csv에서 in_P_main, ref_kind == "
           "reference, pos == p. S_folds sha256은 P_report inputs 값과 비교한다(키가 d1ladder 작업 트리 절대 경로라 "
           "repo 상대 경로로 맞춘다). P_012 공통 열 = S_folds 열, 짝 수 = P_report sets.P_012.in_P_main_para, fold = "
           "S_folds fold_T1 = 그 짝을 test로 가진 T1 분할 파일의 fold(E-3 누출 없음 전제). 어긋나면 STOP.",
    "S-2": "반응·짝 분류는 D1LADDER evaluate.read_c_compare / pair_classes(E2 SAME_TS_RULE, 열 same_ts_g1)를 그대로 "
           "import한다: 반응 True = same, False = different; 짝 = same / different / mixed.",
    "S-3": "추출 정의: numpy.random.default_rng(20261024) 하나를 Part A same(dipole 이름 정렬 순, 각 16) -> Part A "
           "different(같은 순, 각 4) -> Part B(같은 순, 각 8) 순서로 쓴다. 층마다 후보를 정렬(pair_id 문자열, rxn_id "
           "정수)하고 rng.choice(n, size=min(k, n), replace=False)로 위치를 뽑는다(n = 0이면 부르지 않음). 모자라면 "
           "있는 만큼 쓰고 S_report shortfalls에 적는다.",
    "S-4": "Part B 후보 = 모집단 짝(a·b 양 끝, 짝 분류 무관)의 고유 반응 중 same_ts_g1 = False; dipole = 그 짝의 dipole"
           "(C_compare dipole과 같음 확인). 모든 반응(9,266) 대신 모집단으로 한정한 이유: 2절 모집단이 Phase S 전체에 "
           "걸리고, G1 TS·G2 TS가 모두 있는 반응만 G2 != G1 판정이 있다. Part A와 겹쳐도 된다.",
    "S-5": "출력 열: S_pairs.csv = pair_id, a_rxn, b_rxn, fold(= fold_T1), class, dipole + a_job, b_job, core, fam "
           "(pair_id 순); S_rxns.csv = rxn_id, job_id, dipole, core, ts_class(반응 same/different), roles(partA / "
           "partB / partA;partB), part_a, part_b, partA_pairs, partA_classes, partA_ends(rxn_id 순). 표본 반응마다 "
           "O에 필요한 입력(G2 폴더 파일, D1 job 사본)이 없으면 STOP.",
}


class Bad(Exception):
    """a start-condition / population inconsistency (start: recorded; sample: STOP)."""


# ---------------------------------------------------------------- small helpers
def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def not_ready(msg):
    print(f"NOT READY: {msg}", file=sys.stderr, flush=True)
    sys.exit(2)


def git(*a):
    return subprocess.run(["git", "-C", str(C.REPO), *a], capture_output=True, text=True)


def git_out(*a):
    r = git(*a)
    if r.returncode != 0:
        raise Bad(f"git {' '.join(a)}: exit {r.returncode}: {r.stderr.strip()[:300]}")
    return r.stdout.strip()


def unchanged(rel):
    """(in LADDER_COMMIT, identical in the working tree) for a repo-relative path."""
    in_c = git("cat-file", "-e", f"{LADDER_COMMIT}:{rel}").returncode == 0
    d = git("diff", "--name-only", LADDER_COMMIT, "--", rel)
    return in_c, d.returncode == 0 and not d.stdout.strip()


def env_rec():
    return dict(host=socket.gethostname(), python=platform.python_version(), numpy=np.__version__,
                pandas=pd.__version__, slurm_job=os.environ.get("SLURM_JOB_ID"),
                cpus=os.environ.get("SLURM_CPUS_PER_TASK"), g3_scratch=str(C.G3),
                git_head=git("rev-parse", "--short", "HEAD").stdout.strip())


def _bool(s):
    return s.astype(str).str.strip().str.lower().eq("true")


def guarded(fn):
    """fn() -> (record, problems); SystemExit (lad_common / inputs stop or not-ready), Bad and errors become problems."""
    try:
        return fn()
    except Bad as e:
        return dict(error=str(e)), [str(e)]
    except SystemExit as e:
        return dict(error=f"SystemExit {e.code} (message on stderr)"), [f"{fn.__name__}: SystemExit {e.code}"]
    except Exception as e:                                 # noqa: BLE001 — recorded, the condition fails
        return dict(error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-3000:]), \
            [f"{fn.__name__}: {type(e).__name__}: {e}"]


def grab(pattern, text):
    m = re.search(pattern, text, re.M)
    return m.group(1) if m else None


def same_file(a, b):
    return bool(a and b) and Path(a).resolve() == Path(b).resolve()


# ---------------------------------------------------------------- population (spec 2; used by 0-2 and Phase S)
def population():
    """-> dict(pop = population pairs (index pair_id, sorted; columns of S_folds + 'class'), R = population reactions
    (index rxn_id: job_id, dipole, core, same_ts_g1, ts_class), info, inputs); raises Bad."""
    rep_p = IN.p_report()                                # all_pass, not running (exit 2 / 3 inside otherwise)
    sp = (rep_p.get("sets") or {}).get(SET)
    if not sp:
        raise Bad(f"{SET} not in {IN.P_REPORT}")
    Pset = IN.read_set(SET, rep_p)                       # sha256 = P_report sets.P_012.sha256
    # S_folds.csv: P_report keys it by the d1ladder worktree's absolute path -> match the repo-relative path
    rel = "analysis/d1_pair/results/S_folds.csv"
    keys = [k for k in (rep_p.get("inputs") or {}) if k.endswith("/" + rel)]
    if len(keys) != 1:
        raise Bad(f"P_report inputs: {len(keys)} entries for {rel}")
    if Path(IN.S_FOLDS).resolve() != (C.REPO / rel).resolve():
        raise Bad(f"inputs.S_FOLDS {IN.S_FOLDS} is not {C.REPO / rel}")
    sf_sha = C.sha256(IN.S_FOLDS)
    if sf_sha != rep_p["inputs"][keys[0]]:
        raise Bad(f"{IN.S_FOLDS} sha256 {sf_sha} != P_report inputs {rep_p['inputs'][keys[0]]}")
    S = pd.read_csv(IN.S_FOLDS, dtype={"pair_id": str, "a_job": str, "b_job": str})
    if S.pair_id.duplicated().any() or Pset.pair_id.duplicated().any():
        raise Bad("duplicated pair_id in S_folds.csv or P_012.csv")
    S = S.set_index("pair_id")
    P = Pset.set_index("pair_id")
    miss = P.index[~P.index.isin(S.index)]
    if len(miss):
        raise Bad(f"{len(miss)} P_012 pairs not in S_folds.csv (e.g. {list(miss[:3])})")
    X = S.loc[P.index]
    diff = [c for c in IN.SET_COLS if c != "pair_id" and
            not P[c].astype(str).reset_index(drop=True).equals(X[c].astype(str).reset_index(drop=True))]
    if diff:
        raise Bad(f"P_012 columns {diff} differ from S_folds.csv")
    m = _bool(X.in_P_main) & X.ref_kind.astype(str).eq("reference") & X.pos.astype(str).eq("p")
    pop = X[m.to_numpy()].copy().sort_index()
    if len(pop) != int(sp.get("in_P_main_para", -1)):
        raise Bad(f"population {len(pop)} pairs != P_report sets.{SET}.in_P_main_para {sp.get('in_P_main_para')}")
    # fold = fold_T1 = the T1 test list holding the pair (FU2 split copies, sha256 = P_report)
    test_of = EV.test_lists_all(IN, rep_p)
    fold = pop.fold_T1.astype(int)
    badf = [p for p, k in zip(pop.index, fold) if test_of.get(str(p)) != k or not 0 <= k < NF]
    if badf:
        raise Bad(f"{len(badf)} population pairs whose fold_T1 is not their T1 test fold (e.g. {badf[:3]})")
    # pair classes (D1LADDER E2 rule) and reaction table
    cc = EV.read_c_compare()
    if Path(EV.C_COMPARE).resolve() != C_COMPARE.resolve():
        raise Bad(f"evaluate.C_COMPARE {EV.C_COMPARE} is not {C_COMPARE}")
    pop["class"] = EV.pair_classes(pop, cc)[SAME_COL].to_numpy()
    Cc = pd.read_csv(C_COMPARE, usecols=list(C_COLS), dtype=str)
    crow = {int(r): (j, d, g) for r, j, d, g in zip(Cc.rxn_id, Cc.job_id, Cc.dipole, Cc.g2_status)}
    rx, problems = {}, []
    for pid, r in pop.iterrows():
        for end in ("a", "b"):
            rid, job = int(r[f"{end}_rxn"]), str(r[f"{end}_job"])
            v = (job, str(r.dipole), str(r.core))
            if rx.setdefault(rid, v) != v:
                problems.append(f"rxn {rid}: job / dipole / core {rx[rid]} vs {v} ({pid})")
    for rid, (job, dip, _) in rx.items():
        c = crow.get(rid)
        if c is None:
            problems.append(f"rxn {rid}: not in C_compare.csv")
            continue
        if c[0] != job or c[1] != dip:
            problems.append(f"rxn {rid}: C_compare job / dipole {c[:2]} != pair {job} / {dip}")
        if c[2] != "ok":
            problems.append(f"rxn {rid}: g2_status {c[2]!r}")
        if cc["same"][SAME_COL].get(rid) is None:
            problems.append(f"rxn {rid}: no {SAME_COL} judgment")
    if problems:
        raise Bad(f"{len(problems)} population reaction problems: " + "; ".join(problems[:8]))
    R = pd.DataFrame([dict(rxn_id=rid, job_id=v[0], dipole=v[1], core=v[2], same_ts_g1=bool(cc["same"][SAME_COL][rid]))
                      for rid, v in rx.items()]).set_index("rxn_id").sort_index()
    R["ts_class"] = np.where(R.same_ts_g1, "same", "different")
    dips = sorted(pop.dipole.astype(str).unique())
    info = dict(
        n_pairs=int(len(pop)), n_P012=int(len(P)), dipoles=dips,
        pairs_by_class={k: int((pop["class"] == k).sum()) for k in PAIR_CLASSES},
        pairs_by_dipole_class={d: {k: int(((pop["class"] == k) & (pop.dipole == d)).sum()) for k in PAIR_CLASSES}
                               for d in dips},
        pairs_by_fold={str(k): int((fold == k).sum()) for k in range(NF)},
        n_reactions=int(len(R)), n_a_reactions=int(pop.a_rxn.nunique()), n_b_reactions=int(pop.b_rxn.nunique()),
        reactions_by_ts_class={k: int((R.ts_class == k).sum()) for k in ("same", "different")},
        reactions_by_dipole_ts_class={d: {k: int(((R.ts_class == k) & (R.dipole == d)).sum())
                                          for k in ("same", "different")} for d in dips})
    inputs = {str(Path(sp["file"])): sp["sha256"], str(IN.S_FOLDS): sf_sha, str(C_COMPARE): C.sha256(C_COMPARE),
              str(IN.P_REPORT): C.sha256(IN.P_REPORT),
              **{str(IN.SPLITS_COPY / f"T1__f{k}.json"): rep_p["folds"]["files"][str(IN.SPLITS_COPY / f"T1__f{k}.json")]
                 for k in range(NF)}}
    return dict(pop=pop, R=R, info=info, inputs=inputs)


def missing_files(rid, job):
    """the O inputs of one reaction: G2 folder files (no .fail_*) and the D1 job copy (d1pair_common.JOB_FILES)."""
    g = G2_FINAL / str(rid)
    miss = [f"g2_final/{rid}/{f}" for f in G2_FILES if not (g / f).is_file()]
    try:
        miss += [f"g2_final/{rid}/{x}" for x in sorted(os.listdir(g)) if x.startswith(".fail_")]
    except OSError:
        miss.append(f"g2_final/{rid}/ (no folder)")
    j = DP.D1_JOBS / job
    miss += [f"d1_jobs/{job}/{f}" for f in DP.JOB_FILES if not (j / f).is_file()]
    return miss


# ---------------------------------------------------------------- 0-1 D1LADDER results + unit cache
def check_01():
    r, bad = {}, []
    r["ladder_commit"] = git_out("rev-parse", LADDER_COMMIT)
    r["ladder_commit_is_ancestor_of_HEAD"] = git("merge-base", "--is-ancestor", LADDER_COMMIT, "HEAD").returncode == 0
    if not r["ladder_commit_is_ancestor_of_HEAD"]:
        bad.append(f"0-1 {LADDER_COMMIT} is not an ancestor of HEAD")
    files = {}
    for f in LAD_FILES:
        p, rel = C.LAD_RES / f, f"analysis/d1_ladder/results/{f}"
        in_c, same = unchanged(rel)
        files[f] = dict(present=p.is_file(), in_ladder_commit=in_c, unchanged=same,
                        sha256=C.sha256(p) if p.is_file() else None)
        if not (p.is_file() and in_c and same):
            bad.append(f"0-1 {rel}: present {p.is_file()}, in {LADDER_COMMIT} {in_c}, unchanged {same}")
    r["files"] = files
    if bad:
        return r, bad
    Lr, E1, E2, X = (C.read_json(C.LAD_RES / f) for f in ("L_report.json", "E1_report.json", "E2_report.json",
                                                          "L_xgb_report.json"))
    el = E2.get("l_report") or {}
    sh = {f: files[f]["sha256"] for f in files}
    links = {"L_predictions = L_report.predictions_sha256": sh["L_predictions.parquet"] == Lr.get("predictions_sha256"),
             "L_predictions = E2_report.l_report.predictions_sha256": sh["L_predictions.parquet"] == el.get(
                 "predictions_sha256"),
             "L_report = E2_report.l_report.l_report_sha256": sh["L_report.json"] == el.get("l_report_sha256"),
             "L_val = L_report.val_csv_sha256": sh["L_val.csv"] == Lr.get("val_csv_sha256"),
             "L_val = E2_report.l_report.l_val_sha256": sh["L_val.csv"] == el.get("l_val_sha256"),
             "L_xgb_report = L_report.xgb_check.sha256": sh["L_xgb_report.json"] == (Lr.get("xgb_check") or {}).get(
                 "sha256")}
    r["sha256_links"] = links
    bad += [f"0-1 sha256 link failed: {k}" for k, v in links.items() if not v]
    state = dict(L_stage=Lr.get("stage"), L_rungs=Lr.get("rungs"), L_n_units=Lr.get("n_units"),
                 L_models=Lr.get("models"), X_ok=X.get("ok"), X_n_units=X.get("n_units"), E1_phase=E1.get("phase"),
                 E1_stage=E1.get("stage"), E2_phase=E2.get("phase"), E2_stage=E2.get("stage"))
    want = dict(L_stage=2, L_rungs=list(RUNGS), L_n_units=120, L_models=list(ATOM_MODELS) + [XGB], X_ok=True,
                X_n_units=30, E1_phase="E1", E1_stage=1, E2_phase="E2", E2_stage=2)
    r["state"] = state
    bad += [f"0-1 {k} = {state[k]!r}, need {v!r}" for k, v in want.items() if state[k] != v]
    # E2 representatives (rung x arm x target)
    reps, xgb_reps, miss = E2.get("representatives") or {}, [], []
    for rg in RUNGS:
        for a in ARMS:
            for t in DP.TARGETS:
                m = (((reps.get(rg) or {}).get(a) or {}).get(t) or {}).get("model")
                if m not in ATOM_MODELS + (XGB,):
                    miss.append(f"{rg}|{a}|{t}")
                elif m == XGB:
                    xgb_reps.append(f"{rg}|{a}|{t}")
    r["representatives"] = dict(n=len(RUNGS) * len(ARMS) * len(DP.TARGETS) - len(miss), missing=miss,
                                xgb=xgb_reps, note="XGB representatives have no model file: left out of E (spec T-4)")
    if miss:
        bad.append(f"0-1 E2 representatives missing / unknown model: {miss[:6]}")
    # unit cache
    names = sorted(f"L__{rg}__{a}__{m}__f{k}" for rg in RUNGS for a in ARMS for m in ATOM_MODELS + (XGB,)
                   for k in range(NF))
    eu = E2.get("units") or {}
    if set(eu) != set(names) or set(Lr.get("units") or {}) != set(names):
        bad.append(f"0-1 unit names: E2_report.units {len(eu)}, L_report.units {len(Lr.get('units') or {})}, "
                   f"expected the {len(names)} L__<rung>__<arm>__<model>__f<k>")
        r["units"] = dict(error="unit names differ")
        return r, bad
    listing = set(os.listdir(UNITS)) if UNITS.is_dir() else set()
    used, probs, per = set(), [], {}
    for nm in names:
        _, rg, a, m, _f = nm.split("__")
        rec = per.setdefault(f"{rg}|{a}|{m}", dict(units=0, json_parquet_ok=0, pt_files=0, pt_bytes=0,
                                                   model_files=0))
        rec["units"] += 1
        ok = True
        for ext, key in ((".json", "json_sha256"), (".parquet", "parquet_sha256")):
            f = nm + ext
            used.add(f)
            if f not in listing or C.sha256(UNITS / f) != eu[nm].get(key):
                ok = False
                probs.append(f"{f}: missing or sha256 != E2_report.units")
        rec["json_parquet_ok"] += int(ok)
        mf = sorted(f for f in listing if f.startswith(nm + "__") or
                    (f.startswith(nm + ".") and f not in (nm + ".json", nm + ".parquet")))
        used.update(mf)
        rec["model_files"] += len(mf)
        if m != XGB:
            want_pt = [f"{nm}__s{j}.pt" for j in range(N_SEEDS)]
            if mf != want_pt:
                probs.append(f"{nm}: model files {mf} != {want_pt}")
            rec["pt_files"] += sum(1 for f in mf if f.endswith(".pt"))
            rec["pt_bytes"] += sum((UNITS / f).stat().st_size for f in mf)
    xgb_mf = {k: v["model_files"] for k, v in per.items() if k.endswith("|" + XGB)}
    xj = UNITS / f"{[n for n in names if n.split('__')[3] == XGB][0]}.json"
    r["units"] = dict(
        dir=str(UNITS), n_units=len(names), n_files_listed=len(listing), per_rung_arm_model=per,
        atom_pt_total=sum(v["pt_files"] for k, v in per.items() if not k.endswith("|" + XGB)),
        atom_pt_expected=len(RUNGS) * len(ARMS) * len(ATOM_MODELS) * NF * N_SEEDS,
        xgb_model_files=xgb_mf, xgb_has_model_files=any(xgb_mf.values()),
        xgb_statement=("XGB units: <unit>.json + <unit>.parquet only, no model file (train_xgb.py saves no booster); "
                       "XGB cannot be re-applied to new inputs and is left out of E, reported (spec T-4; not a STOP)"
                       if not any(xgb_mf.values()) else "XGB model files found: see xgb_model_files"),
        xgb_json_keys_example=dict(file=xj.name, keys=sorted(C.read_json(xj)) if xj.is_file() else None),
        other_files=sorted(listing - used)[:50], n_other_files=len(listing - used), problems=probs[:50],
        n_problems=len(probs))
    bad += [f"0-1 unit cache: {p}" for p in probs[:10]]
    return r, bad


# ---------------------------------------------------------------- 0-2 D1-G2 results
def check_02():
    r, bad = {}, []
    rel = "analysis/d1_g2/results/C_compare.csv"
    in_c, same = unchanged(rel)
    sha = C.sha256(C_COMPARE) if C_COMPARE.is_file() else None
    pin = ((C.read_json(ADD3_JSON).get("supplement") or {}).get("same_ts_split") or {}).get("source_sha256")
    cols = list(pd.read_csv(C_COMPARE, nrows=0).columns) if C_COMPARE.is_file() else []
    r["c_compare"] = dict(path=str(C_COMPARE), present=C_COMPARE.is_file(), in_ladder_commit=in_c, unchanged=same,
                          sha256=sha, pinned_sha256=pin, missing_columns=[c for c in C_COLS if c not in cols])
    if not (C_COMPARE.is_file() and in_c and same and sha == pin) or r["c_compare"]["missing_columns"]:
        bad.append(f"0-2 {rel}: {r['c_compare']}")
        return r, bad
    r["g2_final_dir"] = dict(path=str(G2_FINAL), is_dir=G2_FINAL.is_dir())
    if not G2_FINAL.is_dir():
        bad.append(f"0-2 {G2_FINAL} is not a folder")
        return r, bad
    Pp = population()                                   # Bad -> guarded() records it
    r["population"] = Pp["info"]
    miss = {}
    for rid, row in Pp["R"].iterrows():
        m = missing_files(rid, row.job_id)
        if m:
            miss[str(rid)] = m
    r["reaction_files"] = dict(n_reactions=int(len(Pp["R"])), n_with_missing=len(miss),
                               examples=dict(list(miss.items())[:10]), g2_files=list(G2_FILES),
                               job_files=list(DP.JOB_FILES), d1_jobs=str(DP.D1_JOBS))
    if miss:
        bad.append(f"0-2 {len(miss)} population reactions miss G2 / D1 job files (e.g. {list(miss.items())[:2]})")
    return r, bad


# ---------------------------------------------------------------- 0-3 ORCA, xtb, root entry points
def orca_probe(H):
    """serial H2 HF/STO-3G single point in node-local $TMPDIR (removed afterwards; nothing is copied back)."""
    rec, wd = dict(input=ORCA_PROBE_INP), None
    try:
        wd = Path(tempfile.mkdtemp(prefix=f"d1g3_orca_{os.getpid()}_", dir=os.environ.get("TMPDIR") or "/tmp"))
        rec["work"] = str(wd)
        (wd / "probe.inp").write_text(ORCA_PROBE_INP)
        env = dict(os.environ, PATH=f"{H.ORCA_DIR}:{os.environ.get('PATH', '')}",
                   LD_LIBRARY_PATH=f"{H.ORCA_DIR}:{os.environ.get('LD_LIBRARY_PATH', '')}", OMP_NUM_THREADS="1")
        t0 = time.time()
        with open(wd / "probe.out", "w") as fo, open(wd / "probe.err", "w") as fe:
            p = subprocess.run([str(H.ORCA_BIN), "probe.inp"], cwd=str(wd), stdout=fo, stderr=fe, env=env, timeout=900)
        txt = (wd / "probe.out").read_text(errors="replace")
        ver, gi, e = H.RE_VER.search(txt), H.RE_GIT.search(txt), H.RE_FSPE.findall(txt)
        rec.update(returncode=p.returncode, wall_s=time.time() - t0, version=ver.group(1) if ver else None,
                   git=gi.group(1) if gi else None, terminated=H.terminated(txt), e_eh=float(e[-1]) if e else None,
                   stderr_tail=(wd / "probe.err").read_text(errors="replace")[-500:])
    except (OSError, subprocess.SubprocessError) as ex:
        rec["error"] = f"{type(ex).__name__}: {ex}"
    finally:
        if wd is not None:
            shutil.rmtree(wd, ignore_errors=True)
    return rec


def xtb_probe(xtb):
    rec = dict(bin=xtb)
    try:
        p = subprocess.run([xtb, "--version"], capture_output=True, text=True, timeout=120)
        m = re.search(r"xtb version (\S+)", (p.stdout or "") + (p.stderr or ""))
        rec.update(returncode=p.returncode, version=m.group(1) if m else None)
    except (OSError, subprocess.SubprocessError, TypeError) as ex:
        rec["error"] = f"{type(ex).__name__}: {ex}"
    return rec


def source_defs(path):
    """{function name: parameter names} and {module-level name: unparsed value} of a Python source (not imported)."""
    t = ast.parse(Path(path).read_text(encoding="utf-8"))
    fns = {n.name: [a.arg for a in n.args.args] for n in t.body if isinstance(n, ast.FunctionDef)}
    asg = {}
    for n in t.body:
        if isinstance(n, ast.Assign):
            for tg in n.targets:
                for nm in ([tg] if isinstance(tg, ast.Name) else (tg.elts if isinstance(tg, ast.Tuple) else [])):
                    if isinstance(nm, ast.Name):
                        asg[nm.id] = ast.unparse(n.value)[:300]
    return fns, asg


def check_03():
    r, bad = {}, []
    ch = git_out("diff", "--name-only", LADDER_COMMIT, "--", *REUSED_TREES).split()
    r["reused_code"] = dict(trees=list(REUSED_TREES), changed_since_ladder_commit=ch)
    if ch:
        bad.append(f"0-3 reused code changed since {LADDER_COMMIT}: {ch[:8]}")
    if str(C.HF_DIR) not in sys.path:
        sys.path.insert(0, str(C.HF_DIR))
    import hf_common as H
    if Path(H.__file__).resolve() != (C.HF_DIR / "hf_common.py").resolve():
        raise Bad(f"hf_common imported from {H.__file__}")
    cfg, envsh = D1_CONFIG.read_text(encoding="utf-8"), D1_ENV_SH.read_text(encoding="utf-8")
    bins = dict(orca_config=grab(r"^orca_bin:\s*(\S+)", cfg), orca_env_sh=grab(r"^export ORCA_BIN=(\S+)", envsh),
                orca_hf_common=str(H.ORCA_BIN), xtb_config=grab(r"^xtb_bin:\s*(\S+)", cfg),
                xtb_env_sh=grab(r"^export XTB_BIN=(\S+)", envsh), d1_py=grab(r"^export D1_PY=(\S+)", envsh))
    bins["orca_same"] = same_file(bins["orca_config"], H.ORCA_BIN) and same_file(bins["orca_env_sh"], H.ORCA_BIN)
    bins["xtb_same"] = same_file(bins["xtb_config"], bins["xtb_env_sh"])
    bins["d1_py_executable"] = bool(bins["d1_py"] and os.access(bins["d1_py"], os.X_OK))   # G3ENV=d1 (info)
    r["binaries"] = bins
    po = orca_probe(H)
    r["orca"] = dict(po, want_version=H.ORCA_VERSION, want_git=H.ORCA_GIT,
                     ok=bool(bins["orca_same"] and po.get("version") == H.ORCA_VERSION and po.get("git") == H.ORCA_GIT
                             and po.get("terminated")))
    if not r["orca"]["ok"]:
        bad.append(f"0-3 ORCA: same bin {bins['orca_same']}, version {po.get('version')}, git {po.get('git')}, "
                   f"terminated {po.get('terminated')}, error {po.get('error')}")
    xp = xtb_probe(bins["xtb_config"])
    r["xtb"] = dict(xp, want_version=XTB_VERSION, ok=bool(bins["xtb_same"] and xp.get("version") == XTB_VERSION))
    if not r["xtb"]["ok"]:
        bad.append(f"0-3 xtb: same bin {bins['xtb_same']}, version {xp.get('version')}, error {xp.get('error')}")
    ent = {}
    for path, want in ENTRY.items():
        fns, asg = source_defs(path)
        e = dict(functions={}, text={}, fixed_roots={})
        for fn, params in want.items():
            got = fns.get(fn)
            e["functions"][fn] = dict(params=got, ok=got is not None and got[:len(params)] == params)
            if not e["functions"][fn]["ok"]:
                bad.append(f"0-3 {path.name}: {fn}{tuple(params)} not found (got {got})")
        txt = path.read_text(encoding="utf-8")
        for s in ENTRY_TEXT.get(path, []):
            e["text"][s] = s in txt
            if s not in txt:
                bad.append(f"0-3 {path.name}: '{s}' not in the source")
        for nm in FIXED_ROOTS.get(path, ()):
            e["fixed_roots"][nm] = asg.get(nm)
        ent[str(path.relative_to(C.REPO))] = e
    for path, need in ENTRY_TEXT.items():                # text-only files (hf_common.py)
        k = str(path.relative_to(C.REPO))
        if k not in ent:
            txt = path.read_text(encoding="utf-8")
            ent[k] = dict(text={s: s in txt for s in need})
            bad += [f"0-3 {path.name}: '{s}' not in the source" for s in need if s not in txt]
    r["entry_points"] = ent
    r["hf_geom_roots"] = {k: str(v) for k, v in H.GEOM_ROOT.items()}
    if "g2" not in H.GEOM_ROOT:
        bad.append("0-3 hf_common.GEOM_ROOT has no 'g2' slot")
    _, asg_in = source_defs(C.LAD_DIR / "inputs.py")
    r["cli_facts"] = dict(
        g0f0=("g0 / f0_* commands are fixed to $LAD/dft_final, the D1-HF dft inputs and the -dft versions "
              "(fixed_roots above): no root argument on the g0f0.py CLI"),
        inputs=f"build <rung> choices = RUNG_ORDER {asg_in.get('RUNG_ORDER')}; tables under INP = {asg_in.get('INP')}",
        root_arguments=("d1_features.py runroot / mergeroot <root>; fu2/atoms.py root run|gate|tokens (env G1_ROOT, "
                        "FU2_SCR, FU2_RES, FU2_X182, FU2_ATOM_VERSION, FU2_TOKEN_VERSION); D1-HF prep.py geominputs + "
                        "run.py via D1HF_G2_ROOT / D1HF_SCRATCH / D1HF_RES / D1HF_TEMPLATES"),
        decision="0-3")
    # the strict reading of 0-3 ('g0f0.py / inputs.py ... 루트 인자로 부를 수 있다') is NOT met: their CLIs have no root
    # argument. Recorded as a met STOP condition overridden by decision 0-3 (recommended option), never as a plain pass.
    r["strict_reading"] = dict(
        met=False, overridden=True, override_decision="0-3",
        why=("g0f0.py (g0 / f0_* fixed to $LAD/dft_final, D1-HF dft inputs, -dft versions) and inputs.py (build <rung> in "
             "RUNG_ORDER, tables under $LAD/inputs) have no root argument on their CLI; feat.py runs an adapted copy of "
             "the g0f0 G loop and calls the inputs.py functions with run-time RUNGS entries (feat.py F-2 / F-7)"))
    return r, bad


# ---------------------------------------------------------------- commands
def record_decision(key, text):
    """g3_common.record_decision under an fcntl lock ($G3/locks/DECISIONS.lock, shared by every d1_g3 script):
    concurrent jobs never lose each other's DECISIONS.md lines (g3_common.py itself is not edited)."""
    import fcntl
    lk = C.assert_path(C.G3 / "locks" / "DECISIONS.lock")
    lk.parent.mkdir(parents=True, exist_ok=True)
    with open(lk, "a") as f:
        fcntl.lockf(f, fcntl.LOCK_EX)
        try:
            C.record_decision(key, text)
        finally:
            fcntl.lockf(f, fcntl.LOCK_UN)


def record_decisions(keys):
    for k in keys:
        record_decision(k, DECISIONS[k])


def cmd_start():
    t0 = time.time()
    rep = dict(phase="0", spec="docs/specs/D1G3.md 1", started=now(), env=env_rec(), ladder_commit=LADDER_COMMIT)
    record_decisions(("0-1", "0-2", "0-3", "0-3-bin"))         # spec 11: recorded before they are acted on
    fail = []
    for key, fn in (("0-1", check_01), ("0-2", check_02), ("0-3", check_03)):
        rec, bad = guarded(fn)
        rec["ok"] = not bad
        rep[key] = rec
        fail += bad
        print(f"[start] {key}: {'ok' if not bad else 'FAIL'} ({time.time() - t0:.0f} s)", flush=True)
        for b in bad[:10]:
            print(f"  {b}", flush=True)
    sr = (rep.get("0-3") or {}).get("strict_reading") or {}
    rep["deviations"] = ([dict(condition="0-3", strict_reading_met=False, overridden_by_decision="0-3", why=sr.get("why"),
                               note="spec 12 '0' STOP condition (strict reading) taken past by the recommended option "
                                    "(user: every branch = the recommendation); ok = the other 0-3 checks")]
                         if sr.get("overridden") else [])
    rep.update(fail=fail, stop=bool(fail), finished=now(), wall_s=time.time() - t0,
               decisions={k: DECISIONS[k] for k in ("0-1", "0-2", "0-3", "0-3-bin")})
    C.write_json(START, rep)
    print(json.dumps({k: rep[k] for k in ("fail", "stop", "wall_s")}, indent=1, default=str), flush=True)
    if fail:
        print(f"STOP: start conditions failed ({len(fail)}): see {START}", file=sys.stderr, flush=True)
        return 3
    return 0


def draw(rng, n, k):
    """positions of min(k, n) of n sorted candidates (no rng call when n = 0)."""
    m = min(int(k), int(n))
    return [] if m == 0 else [int(x) for x in rng.choice(int(n), size=m, replace=False)]


def cmd_sample():
    t0 = time.time()
    if not START.is_file():
        not_ready(f"{START} missing (run 'sample.py start' first)")
    st = C.read_json(START)
    if st.get("stop") or st.get("fail") or not all((st.get(k) or {}).get("ok") for k in ("0-1", "0-2", "0-3")):
        C.stop(f"{START}: start conditions not passed ({(st.get('fail') or [])[:3]})")
    record_decisions(("S-1", "S-2", "S-3", "S-4", "S-5"))     # spec 11: recorded before they are acted on
    try:
        Pp = population()
    except Bad as e:
        C.stop(f"population: {e}")
    pop, R = Pp["pop"], Pp["R"]
    dips = Pp["info"]["dipoles"]
    if len(dips) != N_DIPOLES:
        C.stop(f"population has {len(dips)} dipoles {dips}, spec 2 needs {N_DIPOLES}")
    rng = np.random.default_rng(C.SEED_S)
    draws, chosen = [], {}
    for cls, k in PART_A:                                # spec 2 Part A: same, then different
        for d in dips:
            cand = sorted(str(p) for p in pop.index[(pop["class"] == cls).to_numpy() & (pop.dipole == d).to_numpy()])
            pos = draw(rng, len(cand), k)
            ids = sorted(cand[j] for j in pos)
            draws.append(dict(part="A", stratum=cls, dipole=d, need=k, available=len(cand), drawn=len(ids),
                              shortfall=k - len(ids), positions=pos, ids=ids))
            for p in ids:
                chosen[p] = cls
    candB = R[~R.same_ts_g1.astype(bool)]                # spec 2 Part B: G2 TS != G1 TS, same rng, next draws
    partB = []
    for d in dips:
        cand = sorted(int(x) for x in candB.index[(candB.dipole == d).to_numpy()])
        pos = draw(rng, len(cand), N_PARTB)
        ids = sorted(cand[j] for j in pos)
        draws.append(dict(part="B", stratum="G2!=G1", dipole=d, need=N_PARTB, available=len(cand), drawn=len(ids),
                          shortfall=N_PARTB - len(ids), positions=pos, ids=ids))
        partB += ids
    # S_pairs
    SP = pop.loc[sorted(chosen)].copy()
    if not (SP["class"].to_numpy() == np.array([chosen[p] for p in SP.index])).all():
        C.stop("internal: drawn pair classes changed")
    SP.index.name = "pair_id"
    SP = SP.reset_index().rename(columns={"fold_T1": "fold"})
    SP = SP[["pair_id", "a_rxn", "b_rxn", "fold", "class", "dipole", "a_job", "b_job", "core", "fam"]].copy()
    SP["a_rxn"], SP["b_rxn"], SP["fold"] = SP.a_rxn.astype(int), SP.b_rxn.astype(int), SP.fold.astype(int)
    # S_rxns
    ra = {}
    for pid, a_r, b_r, pcls in zip(SP.pair_id, SP.a_rxn, SP.b_rxn, SP["class"]):
        for end, rid in (("a", int(a_r)), ("b", int(b_r))):
            e = ra.setdefault(rid, dict(pairs=[], classes=set(), ends=set()))
            e["pairs"].append(str(pid))
            e["classes"].add(str(pcls))
            e["ends"].add(end)
    setB = set(partB)
    ids = sorted(set(ra) | setB)
    rows = []
    for rid in ids:
        rr = R.loc[rid]
        inA, inB = rid in ra, rid in setB
        rows.append(dict(rxn_id=int(rid), job_id=rr.job_id, dipole=rr.dipole, core=rr.core, ts_class=rr.ts_class,
                         roles=";".join(x for x, f in (("partA", inA), ("partB", inB)) if f), part_a=inA, part_b=inB,
                         partA_pairs=";".join(sorted(ra[rid]["pairs"])) if inA else "",
                         partA_classes=";".join(sorted(ra[rid]["classes"])) if inA else "",
                         partA_ends=";".join(sorted(ra[rid]["ends"])) if inA else ""))
    SR = pd.DataFrame(rows, columns=["rxn_id", "job_id", "dipole", "core", "ts_class", "roles", "part_a", "part_b",
                                     "partA_pairs", "partA_classes", "partA_ends"])
    # every sampled reaction has its O inputs (0-2 rule)
    miss = {str(r.rxn_id): m for r in SR.itertuples(index=False) for m in [missing_files(int(r.rxn_id), r.job_id)] if m}
    if miss:
        C.stop(f"{len(miss)} sampled reactions miss G2 / D1 job files: {list(miss.items())[:3]}")
    # Part B sanity: every Part B reaction is a G2 != G1 reaction of the population
    if not all(not bool(R.loc[x, "same_ts_g1"]) for x in partB):
        C.stop("internal: a Part B reaction has same_ts_g1 True")
    C.write_atomic(S_PAIRS, lambda t: SP.to_csv(t, index=False))
    C.write_atomic(S_RXNS, lambda t: SR.to_csv(t, index=False))
    A_ids = set(ra)
    shortfalls = [{k: d[k] for k in ("part", "stratum", "dipole", "need", "available", "drawn", "shortfall")}
                  for d in draws if d["shortfall"] > 0]

    def by_dip(sel):
        return {d: int(sum(1 for x in sel if R.loc[x, "dipole"] == d)) for d in dips}
    rep = dict(
        phase="S", spec="docs/specs/D1G3.md 2", created=now(), env=env_rec(), seed=C.SEED_S,
        rule=("population = P_012 pairs with S_folds in_P_main, ref_kind reference, pos p; reaction class = C_compare "
              "same_ts_g1 (evaluate.read_c_compare); pair class = evaluate.pair_classes; one "
              "numpy.random.default_rng(20261024): Part A same (dipoles sorted, 16 each), Part A different (4 each), "
              "Part B (population reactions with same_ts_g1 False, sorted by rxn_id, 8 each); per stratum candidates "
              "sorted, rng.choice(n, size=min(k, n), replace=False), no call when n = 0"),
        start_report_sha256=C.sha256(START), inputs=Pp["inputs"], population=Pp["info"],
        draws=draws, shortfalls=shortfalls,
        part_a=dict(n_pairs=int(len(SP)), requested={c: k * N_DIPOLES for c, k in PART_A},
                    pairs_by_class={c: int((SP["class"] == c).sum()) for c, _ in PART_A},
                    pairs_by_dipole_class={d: {c: int(((SP["class"] == c) & (SP.dipole == d)).sum()) for c, _ in PART_A}
                                           for d in dips},
                    pairs_by_fold_class={c: {str(k): int(((SP["class"] == c) & (SP.fold == k)).sum())
                                             for k in range(NF)} for c, _ in PART_A},
                    n_unique_reactions=len(A_ids), n_a_reactions=int(SP.a_rxn.nunique()),
                    n_b_reactions=int(SP.b_rxn.nunique()),
                    n_b_shared=int(sum(1 for v in Counter(SP.b_rxn.tolist()).values() if v > 1)),
                    reactions_by_ts_class={c: int(sum(1 for x in A_ids if R.loc[x, "ts_class"] == c))
                                           for c in ("same", "different")},
                    reactions_by_dipole=by_dip(A_ids)),
        part_b=dict(n_reactions=len(partB), requested=N_PARTB * N_DIPOLES,
                    candidates_by_dipole={d: int((candB.dipole == d).sum()) for d in dips},
                    reactions_by_dipole=by_dip(partB), overlap_with_part_a=int(len(setB & A_ids)),
                    overlap_ids=sorted(setB & A_ids)),
        unique_reactions=dict(part_a=len(A_ids), part_b=len(setB), union=len(ids), overlap=int(len(setB & A_ids)),
                              by_dipole=by_dip(ids),
                              by_ts_class={c: int((SR.ts_class == c).sum()) for c in ("same", "different")}),
        outputs={str(S_PAIRS): C.sha256(S_PAIRS), str(S_RXNS): C.sha256(S_RXNS)},
        decisions={k: DECISIONS[k] for k in ("S-1", "S-2", "S-3", "S-4", "S-5")}, wall_s=time.time() - t0)
    C.write_json(S_REPORT, rep)
    print(f"[S] population {len(pop)} pairs ({Pp['info']['pairs_by_class']}), {len(R)} reactions; Part A "
          f"{len(SP)} pairs ({rep['part_a']['pairs_by_class']}), {len(A_ids)} reactions; Part B {len(partB)} "
          f"reactions (overlap {len(setB & A_ids)}); union {len(ids)}; shortfalls {len(shortfalls)}", flush=True)
    for s in shortfalls:
        print(f"  shortfall: {s}", flush=True)
    return 0


def main(a):
    if a == ["start"]:
        return cmd_start()
    if a == ["sample"]:
        return cmd_sample()
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
