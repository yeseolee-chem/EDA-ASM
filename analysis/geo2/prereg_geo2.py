#!/usr/bin/env python3
r"""prereg_geo2.py — GEO2 Phase R: the pre-registration results/PREREG_GEO2.md + results/prereg_geo2.json
(docs/specs/GEO2.md 8-3), written ONCE, before any stage-3 training (spec 14 R·M3). sbatch only; nothing in this file
was run when it was written (login node, no python allowed): every path is untested until it runs inside a job.

CLI (exit 0 = done, 2 = not ready (an input of the pre-registration is missing; nothing written), 3 = STOP / refused)
  python prereg_geo2.py write   -> results/PREREG_GEO2.md + results/prereg_geo2.json.
        Refused (exit 3) when either file is tracked by git: a committed pre-registration is never rewritten (an
        untracked draft is replaced). Needs (else exit 2): GEO1 analysis/geo/results/E_report.json and Y_plan.json;
        GEO2 results/Z_report.json, P_build_geo.json, P_geo_report.json, P_geo_sets.json, C_tests_geo.json,
        results/selftest/E_report.json + E_metrics.csv, analysis/d1_ladder/results/C_smoke_atoms_geo.json and
        C_smoke_xgb_geo.json; the gated input files. STOP (exit 3): Z / P / C stop or not passed, the geo table sha256
        != P_build_geo.json outputs, the label file sha256 != V_report outputs, pair-set inconsistency (duplicates,
        unknown ids, P_23 != P_g2 & P_geo, P_023 != P_dft & P_g2 & P_geo, |P_23| < 0.98 |P_g2|, list sha256 absent from
        P_geo_report.json while it reports sha256 values), a stage-3 unit / prediction / evaluation file already exists
        (training before the pre-registration), C_tests_geo.json not run on the code / geo table / labels now (D-39),
        a stage-3 smoke not passed, its GPU-hour estimate not computed or > 80 (D-40), a tracked
        analysis/d1_ladder/results file differs from git HEAD (an existing LADDER result changed).
  python prereg_geo2.py check   the stage-3 gate (train_atoms / train_xgb gate_prereg_geo2) without training: both
        files tracked and unchanged since HEAD, sha256(md) = md_sha256, rungs = ["geo"], need ⊆ inputs_sha256, every
        listed file and the LADDER result snapshot unchanged -> results/R_check_geo2.json (exit 0 / 3).
Env: pandas + numpy + git (conda reactot): analysis/d1_ladder/jobs/lad.slurm LADENV=reactot, submitted from
analysis/d1_ladder with the script ../geo2/prereg_geo2.py (analysis/geo2/RUNBOOK_lad.md R-1 / R-2).

json keys the stage-3 gate reads (train_atoms.gate_prereg_geo2 / train_xgb.gate_prereg_geo2 = the gate_add3 rule):
  md_sha256; rungs = ["geo"]; inputs_sha256 {absolute, unresolved path (os.path.abspath): sha256} holding need = the
  label file $LAD/F_pairs_cr2.csv and $GEO2/inputs/geo.parquet (the trainers' str(_abs(labels)) and
  str(_abs(inputs_path("geo")))) plus the other training inputs and the training code (every listed file must keep its
  sha256); ladder_results_sha256 {repo-relative path: sha256} = every analysis/d1_ladder/results file at writing time
  except the GEO2 outputs (paths with 'geo' in the name, E3_*, *.tmp) (train_atoms GA-5; repo-relative keys = D1-G3
  F-13).
json key for evaluate.py e3 (not read by the training gate): code.fingerprint_keys {train_atoms: {CODE_SHA key:
  sha256}, train_xgb: {...}} = the trainers' fingerprint code_sha256 at writing time (D-41).
Everything else in the json is a record (numbers copied from the files, sha256 of evidence files, git states).
Numbers in the md are the file values (floats printed with 6 significant digits; the json keeps them exact).
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import geo2_common as C  # noqa: E402

import pandas as pd  # noqa: E402

CODE_VERSION = "geo2-prereg-1"
REPO, RES = C.REPO, C.RES
SPEC = REPO / "docs" / "specs" / "GEO2.md"
LAD_SPEC = REPO / "docs" / "specs" / "D1LADDER.md"
MD, JS = RES / "PREREG_GEO2.md", RES / "prereg_geo2.json"
CHECK_OUT = RES / "R_check_geo2.json"
LAD = Path(os.environ.get("D1LADDER_SCRATCH", "/gpfs/tmp_cpu2/yeseo1ee/d1_ladder"))        # = lad_common.LAD
D1PAIR_SCR = Path(os.environ.get("D1PAIR_SCRATCH", "/gpfs/tmp_cpu2/yeseo1ee/d1_pair"))     # = d1pair_common.SCRATCH
LAD_DIR = REPO / "analysis" / "d1_ladder"
LAD_RES = LAD_DIR / "results"
P_DIR = REPO / "analysis" / "d1_pair"
GEO1_RES = C.GEO1_RES
LABELS = LAD / "F_pairs_cr2.csv"                    # = train_atoms.DEFAULT_LABELS (LADDER 4-1 y)
TABLE = C.GEO2 / "inputs" / "geo.parquet"           # = train_atoms.inputs_path("geo") (GEO2 8-1)
X182 = C.GEO2 / "X182_d1_geo.parquet"               # = train_xgb.RUNGS["geo"]["x182"]
FU2_GEO = RES / "fu2_geo"                           # = train_atoms.RUNGS["geo"]["fu2_res"]
Z_REPORT, P_BUILD, P_REPORT, P_SETS = (RES / n for n in ("Z_report.json", "P_build_geo.json", "P_geo_report.json",
                                                         "P_geo_sets.json"))
C_TESTS = RES / "C_tests_geo.json"
SELFTEST = RES / "selftest"
S_FOLDS = P_DIR / "results" / "S_folds.csv"
LAD_SETS = LAD / "inputs" / "sets"
SET_KEYS = ("P_geo", "P_23", "P_023", "P_g2", "P_dft")
MAIN_SETS = ("P_geo", "P_23", "P_023")              # spec 8-3 item 3
ARMS, MODELS4, NF = ("S", "SH"), ("AW", "GNN", "TF", "XGB"), 5
TARGETS = ["d1", "d2", "elst", "pauli", "oi", "disp", "cpcm", "cds", "c_ghost", "barrier"]
GEO_LIMIT_GPU_H = 80.0                              # spec 8-4 / 12
CODE = [LAD_DIR / "train_atoms.py", LAD_DIR / "train_xgb.py", LAD_DIR / "lad_common.py", LAD_DIR / "inputs.py",
        P_DIR / "fu2" / "models.py", P_DIR / "fu2" / "train.py", P_DIR / "fu2" / "fu2_common.py",
        P_DIR / "fu2" / "tokens.py", P_DIR / "fu2" / "atoms.py", P_DIR / "d1pair_common.py", P_DIR / "d1_train.py",
        REPO / "analysis" / "espley_xtb_repro" / "train_ml_single.py"]
FU2_DIR = P_DIR / "fu2"
ESP_DIR = REPO / "analysis" / "espley_xtb_repro"
# D-41: the trainers' fingerprint code_sha256 key names (train_atoms.CODE_SHA / train_xgb.CODE_SHA; = LADDER prereg.py
# CODE_ATOMS / CODE_XGB) -> prereg_geo2.json code.fingerprint_keys (for evaluate.py e3, the ADD3 11 E / E2-15 analogue)
CODE_ATOMS = {"train_atoms.py": LAD_DIR / "train_atoms.py", "lad_common.py": LAD_DIR / "lad_common.py",
              "fu2/models.py": FU2_DIR / "models.py", "fu2/train.py": FU2_DIR / "train.py",
              "fu2/fu2_common.py": FU2_DIR / "fu2_common.py", "fu2/tokens.py": FU2_DIR / "tokens.py",
              "d1pair_common.py": P_DIR / "d1pair_common.py"}
CODE_XGB = {"train_xgb.py": LAD_DIR / "train_xgb.py", "lad_common.py": LAD_DIR / "lad_common.py",
            "d1_train.py": P_DIR / "d1_train.py", "d1pair_common.py": P_DIR / "d1pair_common.py",
            "train_ml_single.py": ESP_DIR / "train_ml_single.py", "fu2/fu2_common.py": FU2_DIR / "fu2_common.py",
            "fu2/tokens.py": FU2_DIR / "tokens.py"}
# D-39: tests/run_tests.py GEO_CODE_FILES keys (C_tests_geo.json code_sha256; paths relative to analysis/d1_ladder)
C_CODE_GEO = ("tests/run_tests.py", "inputs.py", "train_atoms.py", "train_xgb.py", "lad_common.py", "evaluate.py")
C_TRAINER_CODE = ("train_atoms.py", "train_xgb.py", "inputs.py", "lad_common.py")    # also in inputs_sha256 (CODE)
AUTHOR_ESTIMATE = "같은 형태 짝 oi −30 % 안팎, 전체 −10 % 안팎; 추정치"      # spec 8-3 item 5 (verbatim)
HEX64 = re.compile(r"^[0-9a-f]{64}$")


# ================================================================ helpers
def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def not_ready(msg):
    print(f"NOT READY: {msg}", file=sys.stderr, flush=True)
    sys.exit(2)


def stop(msg):
    C.stop(msg)                                      # exit 3


def ab(p):
    """absolute unresolved path string (= the trainers' str(_abs(p)))."""
    return os.path.abspath(str(p))


def rel(p):
    q = Path(ab(p))
    try:
        return str(q.relative_to(REPO))
    except ValueError:
        return str(q)


def sha(p):
    return C.sha256(p) if Path(p).is_file() else None


def list_sha(ids):
    """GEO2 pair-set rule: sha256 of '\\n'.join(sorted ids)."""
    return hashlib.sha256("\n".join(sorted(str(q) for q in ids)).encode()).hexdigest()


def git(*a):
    return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True)


def tracked(p):
    return git("ls-files", "--error-unmatch", str(p)).returncode == 0


def git_state(p):
    if not Path(p).is_file():
        return "missing"
    if not tracked(p):
        return "untracked"
    return "committed" if git("diff", "--quiet", "HEAD", "--", str(p)).returncode == 0 else "modified"


def read_json(p, what):
    if not Path(p).is_file():
        not_ready(f"{p} missing ({what})")
    return json.loads(Path(p).read_text())


def f6(x):
    """a number as the md prints it (6 significant digits; ints and None unchanged)."""
    if isinstance(x, bool) or x is None:
        return str(x)
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        return f"{x:.6g}"
    return str(x)


def compact(o, n=20):
    """json value with lists longer than n replaced by {'n': len, 'head': first 5} (for the md / json record)."""
    if isinstance(o, dict):
        return {str(k): compact(v, n) for k, v in o.items()}
    if isinstance(o, list):
        return [compact(v, n) for v in o] if len(o) <= n else {"n": len(o), "head": [compact(v, n) for v in o[:5]]}
    return o


def hex_values(o):
    """every 64-hex string anywhere in a json value."""
    out = set()
    if isinstance(o, dict):
        for v in o.values():
            out |= hex_values(v)
    elif isinstance(o, list):
        for v in o:
            out |= hex_values(v)
    elif isinstance(o, str) and HEX64.match(o):
        out.add(o)
    return out


def quote(text):
    """verbatim spec text as a markdown block quote (every line kept, '> ' prefixed)."""
    return [("> " + ln) if ln.strip() else ">" for ln in text.split("\n")]


def section(text, head, name):
    """verbatim lines of a spec from the line starting with `head` up to (not including) the next heading of the same
    or a higher level, a code fence or a code-introduction line ('`analysis/...'); STOP when `head` is not unique."""
    lines = text.split("\n")
    hits = [i for i, ln in enumerate(lines) if ln.startswith(head)]
    if len(hits) != 1:
        stop(f"{name}: heading {head!r} found {len(hits)} times (the spec changed; nothing written)")
    i0 = hits[0]
    lev = len(lines[i0]) - len(lines[i0].lstrip("#"))
    out = [lines[i0]]
    for ln in lines[i0 + 1:]:
        if ln.startswith("#") and len(ln) - len(ln.lstrip("#")) <= lev:
            break
        if ln.startswith("```") or ln.startswith("`analysis/"):
            break
        out.append(ln)
    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out)


def assign_src(path, name, key=None):
    """verbatim source of the module-level assignment to `name` (tuple targets too), or of its dict entry `key`."""
    t = Path(path).read_text(encoding="utf-8")
    for node in ast.parse(t).body:
        if isinstance(node, ast.Assign):
            tg = node.targets
        elif isinstance(node, ast.AnnAssign):
            tg = [node.target]
        else:
            continue
        names = set()
        for g in tg:
            if isinstance(g, ast.Name):
                names.add(g.id)
            elif isinstance(g, ast.Tuple):
                names |= {e.id for e in g.elts if isinstance(e, ast.Name)}
        if name not in names:
            continue
        if key is None:
            return ast.get_source_segment(t, node)
        if isinstance(node.value, ast.Dict):
            for k, v in zip(node.value.keys, node.value.values):
                if isinstance(k, ast.Constant) and k.value == key:
                    return f'{name}["{key}"]: ' + ast.get_source_segment(t, v)
    stop(f"{path}: no module-level assignment {name} {key or ''} (code changed; nothing written)")


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(f6(c) for c in r) + " |" for r in rows]
    return out


def units():
    return [f"L__geo__{a}__{m}__f{k}" for a in ARMS for m in MODELS4 for k in range(NF)]


def need_paths():
    """the paths the stage-3 gate requires in inputs_sha256 (= train_atoms / train_xgb cmd_run need, default labels)."""
    return [ab(LABELS), ab(TABLE)]


def gated_inputs():
    """(absolute path, role, column) of every file the stage-3 gate re-verifies (inputs_sha256)."""
    sp = D1PAIR_SCR / "fu2" / "splits"
    R = [(LABELS, "labels: F_pairs_cr2.csv (LADDER 4-1 y; need)", "shared"),
         (TABLE, "geo input table (feat_geo.py p; need)", "geo"),
         (X182, "geo 182 global features (XGB input = token glob source)", "geo")]
    R += [(FU2_GEO / n, f"geo FU2 report {n} (token layout / versions / manifest)", "geo")
          for n in ("T_report.json", "T_manifest.csv", "Q_report.json", "S_splits.json")]
    R += [(P_SETS, "pair sets P_geo / P_23 / P_023 / P_g2 / P_dft (feat_geo.py p_report)", "geo"),
          (P_DIR / "results" / "F_pairs.csv", "original F_pairs.csv (L-11 invariant on every label load)", "shared"),
          (P_DIR / "results" / "fu2" / "S_splits.json", "FU2 S_splits.json (split index)", "shared"),
          (S_FOLDS, "S_folds.csv (folds, cores; XGB inner GroupKFold)", "shared")]
    R += [(sp / f"T1__f{k}.json", f"T1 fold {k} split file", "shared") for k in range(NF)]
    R += [(LAD_RES / "V_report.json", "label provenance (V_report outputs = F_pairs_cr2.csv sha256)", "shared"),
          (LAD_RES / "P_report.json", "label file record (train_atoms.gate_p_geo)", "shared")]
    R += [(p, "stage-3 training code", "code") for p in CODE]
    return [(ab(p), r, c) for p, r, c in R]


def evidence_paths():
    """files the md cites (sha256 + git state recorded, not gated)."""
    E = [SPEC, LAD_SPEC, GEO1_RES / "E_report.json", GEO1_RES / "E_interim.json", GEO1_RES / "Y_plan.json",
         GEO1_RES / "E_rxn.csv"] + [GEO1_RES / f"R_select_f{k}.json" for k in range(NF)]
    E += [Z_REPORT, RES / "Z_status.csv", P_BUILD, P_REPORT, RES / "F_report.json", RES / "F_G_status.csv", C_TESTS,
          SELFTEST / "E_report.json", SELFTEST / "E_metrics.csv", HERE / "eval_geo2.py", HERE / "build_geo.py",
          HERE / "geo2_common.py", HERE / "feat_geo.py", HERE / "g3c.py", Path(__file__).resolve(),
          REPO / "analysis" / "geo" / "train_m1.py", LAD_DIR / "evaluate.py", LAD_DIR / "tests" / "run_tests.py",
          LAD_RES / "C_smoke_atoms_geo.json", LAD_RES / "C_smoke_xgb_geo.json", LAD_RES / "C_l6_xgb_geo.json"]
    E += [C.GEO1 / "preds" / f"final_f{k}_s{s}.pkl" for k in range(NF) for s in C.SEEDS]
    E += [C.GEO1 / "preds" / f"final_fm1_s{s}.pkl" for s in C.SEEDS]
    return E


def geo2_output(p):
    """GEO2's own outputs inside analysis/d1_ladder/results, excluded from the LADDER snapshot: every path below it with
    'geo' in its name (stage-3 *_geo.* files of train_atoms / train_xgb / tests; no LADDER file has it), the E3_* files
    of evaluate.py e3 (spec 9-2) and *.tmp files of an atomic write in progress."""
    r = str(Path(p).relative_to(LAD_RES)).lower()
    return "geo" in r or Path(p).name.startswith("E3_") or r.endswith(".tmp")


def ladder_snapshot():
    """{repo-relative path: sha256} of every LADDER result file (GA-5; spec 14 'existing LADDER result file changed')."""
    return {rel(p): C.sha256(p) for p in sorted(LAD_RES.rglob("*")) if p.is_file() and not geo2_output(p)}


def ladder_git_changes():
    """tracked analysis/d1_ladder/results files that differ from git HEAD (modified / deleted / renamed); untracked new
    files ('??', e.g. the stage-3 *_geo.* outputs) are allowed. Non-empty = an existing LADDER result changed (STOP)."""
    r = git("status", "--porcelain", "--untracked-files=no", "--", str(LAD_RES))
    if r.returncode != 0:
        stop(f"git status failed on {LAD_RES}: {r.stderr.strip()[:300]}")
    return [ln for ln in r.stdout.splitlines() if ln.strip()]


# ================================================================ gather (read only)
def pair_sets():
    """P_geo_sets.json -> per set: ids (sorted unique), sha256, counts from S_folds.csv; consistency checks."""
    raw = read_json(P_SETS, "feat_geo.py p_report")
    miss = [k for k in SET_KEYS if not isinstance(raw.get(k), list)]
    if miss:
        stop(f"{P_SETS}: lists {miss} missing (interface: P_geo, P_23, P_023, P_g2, P_dft)")
    S = pd.read_csv(S_FOLDS, dtype={"pair_id": str})
    known = set(S.pair_id)
    ids, bad, warn = {}, [], []
    for k in SET_KEYS:
        v = [str(q) for q in raw[k]]
        if len(set(v)) != len(v):
            bad.append(f"{k} has duplicated ids")
        if v != sorted(v):
            warn.append(f"{k} is not sorted (interface: sorted pair_id lists; the sha256 here sorts anyway)")
        unk = sorted(set(v) - known)
        if unk:
            bad.append(f"{k}: {len(unk)} ids not in S_folds.csv, e.g. {unk[:3]}")
        ids[k] = sorted(set(v))
    g, g2, dft = set(ids["P_geo"]), set(ids["P_g2"]), set(ids["P_dft"])
    if set(ids["P_23"]) != g2 & g:
        bad.append("P_23 != P_g2 & P_geo (spec 7)")
    if set(ids["P_023"]) != dft & g2 & g:
        bad.append("P_023 != P_dft & P_g2 & P_geo (spec 7)")
    if len(ids["P_23"]) < 0.98 * len(ids["P_g2"]):
        bad.append(f"|P_23| {len(ids['P_23'])} < 0.98 |P_g2| {len(ids['P_g2'])} (spec 7 STOP)")
    if bad:
        stop(f"{P_SETS}: " + "; ".join(bad))
    for c in ("in_P_main", "in_P_sens", "in_A_main"):
        S[c] = S[c].astype(str).str.lower().eq("true") if S[c].dtype != bool else S[c]
    out = {}
    for k in SET_KEYS:
        s = S[S.pair_id.isin(set(ids[k]))]
        ref = s.ref_kind.eq("reference")
        pm = ref & s.in_P_main
        para = s.pos.eq("p")
        ev = pm & s.fold_T1.ge(0) & (s.in_P_sens | s.in_A_main)        # the eval_geo2 own / P_23 population
        out[k] = dict(n=len(ids[k]), sha256=list_sha(ids[k]), reference=int(ref.sum()),
                      anchor=int(s.ref_kind.eq("anchor").sum()), para=int(para.sum()),
                      P_main_reference=int(pm.sum()), P_main_reference_para=int((pm & para).sum()),
                      eval_P_main_reference_test=int(ev.sum()), eval_P_main_reference_test_para=int((ev & para).sum()),
                      by_fold_T1={str(f): int(n) for f, n in s.fold_T1.value_counts().sort_index().items()},
                      by_fold_T1_P_main_reference_para={str(f): int(n) for f, n in
                                                        s[pm & para].fold_T1.value_counts().sort_index().items()})
    lad = {}
    for k, f in (("P_g2", LAD_SETS / "P_g2.csv"), ("P_dft", LAD_SETS / "P_dft.csv")):
        if f.is_file():
            lad[k] = dict(file=str(f), sha256=C.sha256(f),
                          equal=sorted(set(pd.read_csv(f, dtype={"pair_id": str}).pair_id)) == ids[k])
        else:
            lad[k] = dict(file=str(f), sha256=None, equal=None)
    missing_g2 = sorted(g2 - g)
    return ids, out, dict(ladder_sets=lad, n_in_P_g2_not_in_P_geo=len(missing_g2), in_P_g2_not_in_P_geo=missing_g2,
                          p23_ratio=len(ids["P_23"]) / max(1, len(ids["P_g2"])), warnings=warn)


def selftest_baseline():
    """LADDER g2 / dft REP8 para MAE on P_23 / P_023 from the eval_geo2 selftest (spec 0-4 numbers, already seen)."""
    rep = read_json(SELFTEST / "E_report.json", "eval_geo2.py selftest (spec 1 0-6)")
    st = rep.get("selftest") or {}
    if not st.get("all_pass"):
        stop(f"{SELFTEST / 'E_report.json'}: selftest all_pass {st.get('all_pass')} (spec 1 0-6)")
    if not (SELFTEST / "E_metrics.csv").is_file():
        not_ready(f"{SELFTEST / 'E_metrics.csv'} missing (eval_geo2.py selftest)")
    M = pd.read_csv(SELFTEST / "E_metrics.csv")
    M = M[(M.sel == "REP8") & (M.pos == "p")]
    rows = []
    for t in TARGETS:
        r = dict(target=t)
        for ps, rung in (("P_23", "g2"), ("P_023", "dft"), ("P_023", "g2")):
            m = M[(M.pair_set == ps) & (M.rung == rung) & (M.target == t)]
            if len(m) != 1:
                stop(f"selftest E_metrics.csv: {len(m)} rows for {ps} {rung} REP8 p {t}")
            m = m.iloc[0]
            r[f"{ps}_{rung}"] = dict(n=int(m.n), mae=float(m.mae), ci_lo=float(m.mae_ci_lo), ci_hi=float(m.mae_ci_hi))
        rows.append(r)
    rep8 = {k: v for k, v in (rep.get("rep8") or {}).items() if k.split("|")[0] in ("dft", "g2")}
    return dict(n_pairs=rep.get("n_pairs"), n_para=rep.get("n_para"), selftest=st, rows=rows, rep8=rep8)


def c_tests():
    """C_tests_geo.json (tests/run_tests.py geo merge) all_pass AND run on what this pre-registration freezes (D-28,
    review fix D-39; = LADDER prereg.py Phase C check): full geo mode (not --regress-only), code_sha256 = the files now
    for every recorded key (keys are paths relative to analysis/d1_ladder; the run_tests.py GEO_CODE_FILES keys
    C_CODE_GEO must be there), rungs.geo ready with table_sha256 = $GEO2/inputs/geo.parquet now, labels = F_pairs_cr2.csv
    now (final), consistency / current ok. STOP otherwise: rerun the C tests (both parts + merge) on the final code."""
    c = read_json(C_TESTS, "tests/run_tests.py geo (spec 8-2)")
    if not c.get("all_pass"):
        stop(f"{C_TESTS}: all_pass {c.get('all_pass')} (spec 8-2 STOP: fix and rerun the tests before the PREREG)")
    bad = []
    if c.get("mode") not in (None, "geo") or c.get("training_gate") is False:
        bad.append(f"mode {c.get('mode')} / training_gate {c.get('training_gate')}: not the full geo mode")
    rec = c.get("code_sha256") if isinstance(c.get("code_sha256"), dict) else {}
    lack = [k for k in C_CODE_GEO if k not in rec]
    if lack:
        bad.append(f"code_sha256 lacks {lack} (interface: tests/run_tests.py GEO_CODE_FILES)")
    code_now = {k: sha(LAD_DIR / k) for k in rec}
    diff = sorted(k for k in rec if code_now.get(k) is None or code_now[k] != rec[k])
    if diff:
        bad.append(f"code changed since the C tests: {diff} (the PREREG would freeze untested code)")
    geo = (c.get("rungs") or {}).get("geo") or {}
    tsha = sha(TABLE)
    if geo.get("status") != "ready" or tsha is None or geo.get("table_sha256") != tsha:
        bad.append(f"rungs.geo status {geo.get('status')}, table_sha256 {str(geo.get('table_sha256'))[:12]} != "
                   f"{TABLE} now {str(tsha)[:12]}")
    lab = c.get("labels") or {}
    lsha = sha(LABELS)
    if c.get("labels_final") is not True or lsha is None or lab.get("sha256") != lsha:
        bad.append(f"labels {lab.get('path')} / {str(lab.get('sha256'))[:12]} (final {c.get('labels_final')}) != "
                   f"{LABELS} now {str(lsha)[:12]}")
    for k in ("consistency", "current"):
        if (c.get(k) or {}).get("ok") is not True:
            bad.append(f"{k}.ok {(c.get(k) or {}).get('ok')}")
    if bad:
        stop(f"{C_TESTS} passed, but not on what this pre-registration freezes (D-39; rerun the C tests, both parts + "
             f"merge, on the final code and inputs): " + "; ".join(bad))
    checks = c.get("checks")
    st = {}
    if isinstance(checks, dict):
        for k, v in checks.items():
            st[str(k)] = (v.get("status") if v.get("status") is not None else v.get("pass")) if isinstance(v, dict) else v
    return dict(file=rel(C_TESTS), sha256=C.sha256(C_TESTS), all_pass=True, checks=st, mode=c.get("mode"),
                code_sha256=rec, code_equal_now=True, table_sha256=tsha, labels_sha256=lsha,
                p_build_geo_same=geo.get("build_report_sha256") == sha(P_BUILD),
                gpu_estimate_geo=c.get("gpu_estimate_geo"), merged=c.get("merged"))


def smoke_estimates():
    """spec 8-4 / 14 '예상 > 80 GPU시간 = STOP' (GA-9; review fix D-40): both stage-3 smoke reports must exist (exit 2
    while missing), be stage 3 and passed (gate M-1'), and the atom estimate must have been computed (estimate_geo.ok);
    STOP when it was not (no fallback to the FU2 upper bound: for one LADDER rung that formula already gives ~92 GPU-h)
    or when estimate_geo.geo_gpu_hours > 80."""
    a, x = LAD_RES / "C_smoke_atoms_geo.json", LAD_RES / "C_smoke_xgb_geo.json"
    for p, what in ((a, "train_atoms.py smoke --stage 3: C tests G-9 or RUNBOOK_lad S-1"),
                    (x, "train_xgb.py smoke --stage 3: C tests G-9 or RUNBOOK_lad S-2")):
        if not p.is_file():
            not_ready(f"{p} missing ({what})")
    sa, sx = json.loads(a.read_text()), json.loads(x.read_text())
    e = sa.get("estimate_geo") if isinstance(sa.get("estimate_geo"), dict) else {}
    out = dict(atoms_file=rel(a), atoms_sha256=C.sha256(a), xgb_file=rel(x), xgb_sha256=C.sha256(x),
               atoms_ok=sa.get("ok"), estimate_geo=sa.get("estimate_geo"),
               upper_bound_gpu_hours=sa.get("stage_upper_bound_gpu_hours"), xgb_ok=sx.get("ok"),
               xgb_cpu_hours=sx.get("stage_cpu_hours_estimate"), limit_gpu_hours=GEO_LIMIT_GPU_H,
               atoms_code_now=sa.get("code_sha256") == {k: sha(p) for k, p in CODE_ATOMS.items()},
               xgb_code_now=sx.get("code_sha256") == {k: sha(p) for k, p in CODE_XGB.items()})
    bad = []
    for name, s in (("atom", sa), ("XGB", sx)):
        if str(s.get("stage")) != "3":
            bad.append(f"{name} smoke stage {s.get('stage')} (not the geo smoke)")
        if s.get("ok") is not True:
            bad.append(f"{name} smoke ok {s.get('ok')} (gate M-1' failed: spec 8-2 smoke)")
    h = float("nan")
    if e.get("ok") is not True:
        bad.append(f"estimate_geo not computed ({e.get('why')}): the 80 GPU-h rule cannot be checked")
    else:
        try:
            h = float(e.get("geo_gpu_hours"))
        except (TypeError, ValueError):
            pass
        if not math.isfinite(h):
            bad.append(f"estimate_geo.geo_gpu_hours {e.get('geo_gpu_hours')!r} is not a finite number")
        elif h > GEO_LIMIT_GPU_H:
            bad.append(f"smoke estimate {h:.1f} GPU-h > {GEO_LIMIT_GPU_H} (spec 8-4 / 12)")
    if bad:
        stop("stage-3 smoke / GPU estimate (spec 8-4 / 14 STOP, D-40): " + "; ".join(bad))
    out.update(geo_gpu_hours=h, within_limit=True)
    return out


def seen():
    """what of stage 3 exists already: units / predictions / evaluation = training before the PREREG (STOP)."""
    ud = LAD / "units"
    geo_units = sorted(p.name for p in ud.glob("L__geo__*")) if ud.is_dir() else []
    files = [p for p in (LAD_RES / "L_predictions_geo.parquet", LAD_RES / "L_report_geo.json",
                         LAD_RES / "L_xgb_report_geo.json", LAD_RES / "L_val_geo.csv", RES / "E_report.json",
                         RES / "E_metrics.csv", RES / "E_compare.csv") if p.is_file()]
    out = dict(geo_unit_files=geo_units, prediction_or_evaluation_files=[rel(p) for p in files],
               e3_files=sorted(p.name for p in LAD_RES.glob("E3_*")),
               smoke_dirs={str(d): d.is_dir() for d in (LAD / "smoke_geo", LAD / "smoke_xgb_geo")})
    if geo_units or files:
        stop(f"stage-3 training / evaluation output exists before the pre-registration (spec 14 R·M3): "
             f"{(geo_units + [str(p) for p in files])[:6]}")
    return out


def gather():
    D = dict(date=time.strftime("%Y-%m-%d"), written=now())
    for p in (MD, JS):
        if p.is_file() and tracked(p):
            stop(f"{rel(p)} is tracked by git: a committed pre-registration is written once and never replaced "
                 f"(a change needs a user decision and a new, separately named addendum)")
    D["replaced_draft"] = {rel(p): sha(p) for p in (MD, JS) if p.is_file()}
    D["seen"] = seen()
    if not SPEC.is_file() or not LAD_SPEC.is_file():
        not_ready(f"{SPEC} / {LAD_SPEC} missing")
    D["spec_text"] = SPEC.read_text(encoding="utf-8")
    D["lad_spec_text"] = LAD_SPEC.read_text(encoding="utf-8")
    # GEO1 (spec 1 0-1) and Y (spec 3)
    D["E_report"] = read_json(GEO1_RES / "E_report.json", "GEO1 Phase E (spec 1 0-1)")
    ei = GEO1_RES / "E_interim.json"
    D["E_interim"] = json.loads(ei.read_text()) if ei.is_file() else None
    D["Y_plan"] = read_json(GEO1_RES / "Y_plan.json", "train_m1.py plan (spec 3)")
    # Z (spec 4)
    z = read_json(Z_REPORT, "build_geo.py build (spec 4)")
    if z.get("stop"):
        stop(f"{Z_REPORT}: stop {z.get('stop_reasons')} (spec 4-3)")
    D["Z_report"] = z
    # P (spec 6-1, 7)
    b = read_json(P_BUILD, "feat_geo.py p (spec 6-1)")
    if not b.get("all_pass") or b.get("stop"):
        stop(f"{P_BUILD}: all_pass {b.get('all_pass')}, stop {b.get('stop')} (spec 6-2 P)")
    want = (b.get("outputs") or {}).get(str(TABLE))
    if not TABLE.is_file() or want != C.sha256(TABLE):
        stop(f"{TABLE} sha256 != {P_BUILD.name} outputs ({want})")
    D["P_build"] = dict(file=rel(P_BUILD), sha256=C.sha256(P_BUILD), table=str(TABLE), table_sha256=want,
                        rung_cfg=b.get("rung_cfg"), table_summary=compact(b.get("table")))
    pr = read_json(P_REPORT, "feat_geo.py p_report (spec 7)")
    if pr.get("stop"):
        stop(f"{P_REPORT}: stop {pr.get('stop_reasons') or pr.get('stop_reason')} (spec 7)")
    D["P_geo_report"] = compact(pr)
    D["ids"], D["sets"], D["sets_extra"] = pair_sets()
    hx = hex_values(pr)
    D["sets_sha_in_report"] = {k: (D["sets"][k]["sha256"] in hx) if hx else None for k in MAIN_SETS}
    if hx and not all(D["sets_sha_in_report"].values()):
        stop(f"{P_REPORT}: the list sha256 computed here (sha256 of the newline-joined sorted ids) are not all in the "
             f"report: {D['sets_sha_in_report']} (interface mismatch between P_geo_sets.json and P_geo_report.json)")
    # labels (spec 8-3 item 3)
    if not LABELS.is_file():
        not_ready(f"{LABELS} missing")
    v = read_json(LAD_RES / "V_report.json", "LADDER Phase V")
    lsha = C.sha256(LABELS)
    if not v.get("all_pass") or (v.get("outputs") or {}).get(str(LABELS)) != lsha:
        stop(f"{LABELS} sha256 {lsha[:12]} != results/V_report.json outputs (or V not all_pass)")
    D["labels"] = dict(path=ab(LABELS), sha256=lsha, v_report_outputs_equal=True)
    # tests, smoke, baselines
    D["C_tests"] = c_tests()
    D["smoke"] = smoke_estimates()
    D["baseline"] = selftest_baseline()
    # gated inputs, evidence, LADDER snapshot
    gi = gated_inputs()
    miss = [p for p, _, _ in gi if not Path(p).is_file()]
    if miss:
        not_ready(f"gated inputs missing: {miss[:6]}")
    D["inputs"] = [(p, r, c, C.sha256(p)) for p, r, c in gi]
    D["code_git"] = {rel(p): git_state(p) for p in CODE}
    # D-41: the trainers' code_sha256 key names (e3 checks the geo units' fingerprints against these)
    D["fingerprint_keys"] = dict(train_atoms={k: C.sha256(p) for k, p in CODE_ATOMS.items()},
                                 train_xgb={k: C.sha256(p) for k, p in CODE_XGB.items()})
    # D-39 / D-41: the frozen code (inputs_sha256) = the tested code (C_tests_geo code_sha256) = fingerprint_keys
    insha = {p: s for p, _, _, s in D["inputs"]}
    drift = [k for k in C_TRAINER_CODE if insha.get(ab(LAD_DIR / k)) != D["C_tests"]["code_sha256"].get(k)]
    drift += [f"{t}:{k}" for t, M in (("train_atoms", CODE_ATOMS), ("train_xgb", CODE_XGB)) for k, p in M.items()
              if insha.get(ab(p)) != D["fingerprint_keys"][t][k]]
    if drift:
        stop(f"code changed while the pre-registration was being gathered, or the gated code list lacks a trainer file: "
             f"{drift} (rerun the C tests on the final code, then prereg_geo2.py write)")
    D["evidence"] = {rel(p): dict(sha256=sha(p), git=git_state(p) if str(ab(p)).startswith(str(REPO)) else None)
                     for p in evidence_paths()}
    lg = ladder_git_changes()
    if lg:
        stop(f"existing LADDER result files differ from git HEAD before the snapshot (spec 14): {lg[:8]}")
    D["ladder_snapshot"] = ladder_snapshot()
    D["git_head"] = git("rev-parse", "HEAD").stdout.strip()
    D["branch"] = git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    D["spec_sha256"], D["spec_git"] = C.sha256(SPEC), git_state(SPEC)
    D["lad_spec_sha256"], D["lad_spec_git"] = C.sha256(LAD_SPEC), git_state(LAD_SPEC)
    D["generator"] = dict(file=rel(Path(__file__).resolve()), sha256=C.sha256(Path(__file__).resolve()),
                          git=git_state(Path(__file__).resolve()), code_version=CODE_VERSION)
    D["grid_src"] = dict(
        atoms=assign_src(P_DIR / "fu2" / "models.py", "GRID"),
        xgb=assign_src(REPO / "analysis" / "espley_xtb_repro" / "train_ml_single.py", "GRIDS", "XGB"),
        fu2=[assign_src(P_DIR / "fu2" / "fu2_common.py", n) for n in ("SEED", "LAMBDA", "LR", "FINAL_SEEDS",
                                                                     "ANTISYM_TOL", "ANTISYM_N")],
        d1pair=[assign_src(P_DIR / "d1pair_common.py", n) for n in ("SEED", "N_INNER", "ANTISYM_TOL", "ANTISYM_N")])
    return D


# ================================================================ md
def md_text(D):
    sp, lsp = D["spec_text"], D["lad_spec_text"]
    sec = lambda h: section(sp, h, "docs/specs/GEO2.md")                 # noqa: E731
    lsec = lambda h: section(lsp, h, "docs/specs/D1LADDER.md")           # noqa: E731
    a = []
    A = a.append
    E = D["E_report"]
    prim = E.get("primary") or {}
    g2r, mod = prim.get("g2_raw") or {}, prim.get("model") or {}
    A(f"# PREREG GEO2 — 보정 구조(geo) rung 사전 등록 (LADDER 3단계: geo, {D['date']})")
    A("")
    A(f"이 문서는 `docs/specs/GEO2.md`(sha256 `{D['spec_sha256']}`, git {D['spec_git']}) 8-3절의 사전 등록이다. "
      f"`{D['generator']['file']}`(sha256 `{D['generator']['sha256']}`, git {D['generator']['git']}, "
      f"{CODE_VERSION})가 결과 파일만 읽어 만들었다(git HEAD `{D['git_head']}`, branch `{D['branch']}`, 작성 "
      f"{D['written']}). 수치는 파일에서 옮겼고(표는 유효숫자 6자리, json에는 원래 값), 규칙은 명령서를 그대로 인용했다"
      f"(인용 블록 = 명령서 원문).")
    A("범위: **rung `geo` (LADDER 3단계)**, 2 arm × 4 모델 × 5 fold = 40 unit. 학습(`train_atoms.py run --stage 3`, "
      "`train_xgb.py run --stage 3`)은 이 md와 `prereg_geo2.json`이 commit된 뒤에만 돈다(7절 gate). "
      "**이 문서는 한 번만 쓴다**: `prereg_geo2.py write`는 commit된 PREREG를 덮어쓰지 않는다.")
    A("")
    s = D["seen"]
    A(f"**지금까지 본 것** (작성 {D['written']}):")
    A(f"- geo 학습은 한 번도 돌지 않았다: `$LAD/units/L__geo__*` {len(s['geo_unit_files'])}개, geo 예측·평가 파일 "
      f"{len(s['prediction_or_evaluation_files'])}개(`L_predictions_geo.parquet`, `L_report_geo.json`, "
      f"`L_xgb_report_geo.json`, `analysis/geo2/results/E_report.json` 없음). E3 파일: {s['e3_files'] or '없음'}.")
    A("- geo에서 지금까지 계산한 것은 구조와 입력뿐이다: Y(fold −1 GEO1 추가 모델), Z(geo 구조 폴더, 1-3절), G(xTB "
      "Freq), F(feature·토큰), H(HF EDA), P(입력표·짝 집합, 3절), 8-2 검사(`C_tests_geo.json`, 5-5절; smoke는 fold 0, "
      "2 epoch이며 그 시험 예측은 평가하지 않았다).")
    A("- `eval_geo2.py selftest`는 g2 예측을 geo로 복사해 LADDER 결과를 재현한 것이다(5-3절). geo 예측은 열지 않았다.")
    A(f"- GEO1 시험 결과(1단계 구조, 5-1절)는 이미 보았다: 주 지표 짝 비율 g2_raw {f6(g2r.get('primary_share'))} → "
      f"model {f6(mod.get('primary_share'))}.")
    if D["replaced_draft"]:
        A(f"- 이 파일은 commit되지 않은 초안을 대체했다: {D['replaced_draft']}")
    A("")
    A("### 명령서 8-3절 (이 문서가 담을 것, 그대로)")
    a += quote(sec("### 8-3."))
    A("")
    # ---------------------------------------------------------- 1
    A("## 1. rung `geo` 정의 (8-3 항목 1)")
    A("")
    A("### 1-1. 정의 (명령서 0-2절 그대로)")
    a += quote(sec("### 0-2."))
    A("")
    A("### 1-2. 상식 검사·대체 규칙·Z gate (명령서 4절 그대로; 코드 `analysis/geo2/build_geo.py`는 명령서 4절 그대로)")
    a += quote(sec("## 4."))
    A("")
    z = D["Z_report"]
    nf = z.get("n_fallback")
    nfold = z.get("n_folders")
    rate = (nf / nfold) if isinstance(nf, (int, float)) and nfold else None
    ec = z.get("e_rxn_check") or {}
    A("### 1-3. Z 결과 (`analysis/geo2/results/Z_report.json`)")
    A(f"- 대상 {f6(z.get('n_targets'))}반응, geo 폴더 {f6(nfold)}개, 상태별 {z.get('by_status')}, 출처(fold)별 "
      f"{z.get('by_source')}")
    A(f"- G2 TS 대체(fallback) {f6(nf)}개 (비율 {f6(rate)}; 상한 0.02), 이유별 {z.get('fallback_reasons')}")
    A(f"- GEO1 `E_rxn.csv` 형성 결합 오차 재현: n {f6(ec.get('n'))}, 최대 차이 {f6(ec.get('max_abs_dev_A'))} Å "
      f"(기준 {f6(ec.get('tol'))}); stop {z.get('stop')}")
    A("")
    y = D["Y_plan"]
    A("### 1-4. Y 설정 (명령서 3절 그대로 + `analysis/geo/results/Y_plan.json`)")
    a += quote(sec("## 3."))
    A("")
    A(f"- `Y_plan.json`: 설정 `{y.get('cfg')}`, epoch {f6(y.get('epochs'))}; fold 0–4에서 고른 설정 {y.get('chosen')}, "
      f"best_ep {y.get('best_ep')}; 규칙 \"{y.get('rule')}\"")
    A("")
    # ---------------------------------------------------------- 2
    A("## 2. 학습 규칙 (8-3 항목 2: LADDER 4-2·4-3 그대로)")
    A("")
    A("### 2-1. 입력 두 가지 (LADDER 명령서 4-2절 그대로)")
    a += quote(lsec("### 4-2."))
    A("")
    A("### 2-2. 모델 네 개 (LADDER 명령서 4-3절 그대로)")
    a += quote(lsec("### 4-3."))
    A("")
    A("### 2-3. grid와 상수 (코드에서 그대로; 파일 sha256은 7-1절 gate 표)")
    A("```python")
    A("# analysis/d1_pair/fu2/models.py (AW / GNN / TF, 변형 A+G)")
    A(D["grid_src"]["atoms"])
    A("# analysis/espley_xtb_repro/train_ml_single.py (XGB; 칸 = d1_train.grid_cells('XGB'), 정렬된 키의 곱 12칸)")
    A(D["grid_src"]["xgb"])
    A("# analysis/d1_pair/fu2/fu2_common.py")
    a += D["grid_src"]["fu2"]
    A("# analysis/d1_pair/d1pair_common.py")
    a += D["grid_src"]["d1pair"]
    A("```")
    A("- 원자 모델: AdamW, cosine(T_max = MAX_EPOCHS), 튜닝 seed = SEED, 최종 = FINAL_SEEDS 5개 평균, 최대 300 epoch, "
      "patience 30, 칸 = 검증 지표 최저(동점은 grid 순서). SH의 β·잔차·벌점 규칙은 2-1절. XGB: 타깃마다 튜닝(안쪽 "
      "GroupKFold(4) MAE 평균 최저), random_state 42 한 번 학습. gate M-1′는 모든 학습에서 확인한다.")
    A("- 분할: FU2 T1 split 파일 fold 0–4 (LADDER와 같은 파일, 7-1절 sha256). geo에서 a·b가 사용 가능하지 않은 짝은 모든 "
      "목록에서 뺀다(LADDER와 같은 규칙) → 두 arm, 네 모델이 같은 짝을 쓴다.")
    A("")
    A("### 2-4. 대표 고르기 (LADDER 명령서 13-1절 그대로; REP8은 4절)")
    a += quote(lsec("### 13-1."))
    A("")
    A("### 2-5. 학습 단위와 실행 (명령서 8-1·8-4절 그대로)")
    a += quote(sec("### 8-1."))
    A(">")
    a += quote(sec("### 8-4."))
    A("")
    A(f"- unit {len(units())}개: " + ", ".join(f"`{u}`" for u in units()[:2]) + " … `" + units()[-1] + "` "
      "(이름 규칙 `L__geo__<arm>__<model>__f<k>`; json `units`).")
    A("")
    # ---------------------------------------------------------- 3
    A("## 3. 짝 집합 (8-3 항목 3, 명령서 7절)")
    A("")
    a += quote(sec("## 7."))
    A("")
    st = D["sets"]
    A("### 3-1. 짝 수와 sha256 (`analysis/geo2/results/P_geo_sets.json`; 수는 `S_folds.csv`로 셈; sha256 = "
      "정렬한 pair_id를 줄바꿈으로 이은 문자열의 sha256)")
    rows = [[k, v["n"], v["reference"], v["anchor"], v["para"], v["P_main_reference"], v["P_main_reference_para"],
             v["eval_P_main_reference_test"], v["eval_P_main_reference_test_para"], "`" + v["sha256"] + "`"]
            for k, v in st.items()]
    a += table(["집합", "짝", "기준", "anchor", "para", "P_main 기준", "P_main 기준 para", "평가 대상(시험 fold)",
                "평가 대상 para", "sha256"], rows)
    A("")
    n23 = (D["baseline"].get("n_pairs") or dict()).get("P_23")
    A("- '평가 대상' = P_main 기준 짝 중 T1 fold 0–4의 시험 쪽(P_sens ∪ A_main)에 있는 짝 = `eval_geo2.py`가 P_23·"
      f"P_023을 세는 모집단(4절; selftest의 P_23 = {f6(n23)}짝).")
    ex = D["sets_extra"]
    A(f"- |P_23| / |P_g2| = {f6(ex['p23_ratio'])} (STOP 기준 < 0.98). P_g2에 있고 P_geo에 없는 짝 "
      f"{ex['n_in_P_g2_not_in_P_geo']}개(json `pair_sets_extra`; 이유는 3-3절 feat_geo 보고).")
    A(f"- LADDER 짝 집합 파일과 같은가: {ex['ladder_sets']}; 경고: {ex['warnings'] or '없음'}")
    A(f"- `P_geo_report.json`에 같은 sha256이 있는가: {D['sets_sha_in_report']} (None = 보고에 sha256 값 없음)")
    A("")
    A("### 3-2. fold별 짝 수 (`fold_T1`; −1 = 어느 fold의 시험도 아닌 반응의 짝)")
    folds = sorted({f for v in st.values() for f in v["by_fold_T1"]}, key=lambda q: int(q))
    a += table(["집합"] + [f"fold {f}" for f in folds] + [f"P_main 기준 para fold {f}" for f in folds],
               [[k] + [v["by_fold_T1"].get(f, 0) for f in folds]
                + [v["by_fold_T1_P_main_reference_para"].get(f, 0) for f in folds] for k, v in st.items()])
    A("")
    A("### 3-3. `P_geo_report.json` (feat_geo.py, 그대로; 20개보다 긴 목록은 개수와 앞 5개)")
    A("```json")
    a += json.dumps(D["P_geo_report"], indent=1, ensure_ascii=False, default=str).split("\n")
    A("```")
    A("")
    A("### 3-4. 라벨과 geo 입력표")
    A(f"- 라벨 `{D['labels']['path']}` sha256 `{D['labels']['sha256']}` (= `results/V_report.json` outputs; LADDER 4-1 "
      "정답 y, 다른 rung과 같은 파일)")
    A(f"- geo 입력표 `{D['P_build']['table']}` sha256 `{D['P_build']['table_sha256']}` (= `{D['P_build']['file']}` "
      f"outputs; rung_cfg {D['P_build']['rung_cfg']})")
    A("")
    # ---------------------------------------------------------- 4
    A("## 4. 지표와 판정 규칙 (8-3 항목 4: 명령서 9·10절 그대로)")
    A("")
    A("### 4-1. 명령서 9절 (평가 코드 앞까지)")
    a += quote(sec("## 9."))
    A("")
    ev = D["evidence"].get("analysis/geo2/eval_geo2.py") or {}
    A(f"- 평가 코드 `analysis/geo2/eval_geo2.py` = 명령서 9절 코드 그대로 (sha256 `{ev.get('sha256')}`, git "
      f"{ev.get('git')}). selftest: {D['baseline']['selftest']}.")
    A("")
    A("### 4-2. 명령서 10절")
    a += quote(sec("## 10."))
    A("")
    # ---------------------------------------------------------- 5
    A("## 5. 이미 본 결과 (8-3 항목 5)")
    A("")
    A("### 5-1. GEO1 시험 결과 (`analysis/geo/results/E_report.json`)")
    ci = (E.get("ci") or {}).get("model_vs_g2_raw") or {}
    A(f"- 판정 `{E.get('verdict')}`; 주 지표(같은 형태 para 짝 중 형성 결합 오차 ≤ {f6(E.get('thr_A'))} Å 비율) "
      f"g2_raw {f6(g2r.get('primary_share'))} → model {f6(mod.get('primary_share'))}; 차이 core bootstrap 95 % CI "
      f"{[f6(q) for q in (ci.get('ci_diff') or [])]} (n_boot {f6(E.get('n_boot'))}); 시험 반응 {f6(E.get('n_test'))}개; "
      f"g2_raw 기대값 {f6(E.get('g2_raw_expected'))}.")
    keys = []
    for v in prim.values():
        for k in (v or {}):
            if k not in keys and k != "folds":
                keys.append(k)
    a += table(["대상"] + keys + ["folds"], [[n] + [(v or {}).get(k) for k in keys] + [(v or {}).get("folds")]
                                           for n, v in prim.items()])
    A("")
    crow = []
    for n, v in (E.get("ci") or {}).items():
        v = v or {}
        crow.append([n, v.get("folds"), v.get("model_share"), v.get("other_share"),
                     [f6(q) for q in (v.get("ci_a") or [])], [f6(q) for q in (v.get("ci_b") or [])],
                     [f6(q) for q in (v.get("ci_diff") or [])]])
    a += table(["비교", "folds", "model", "other", "CI a", "CI b", "CI 차이"], crow)
    A("")
    EI = D["E_interim"]
    A("### 5-2. GEO1 중간 결과 (`analysis/geo/results/E_interim.json`, 보고용)")
    if EI:
        A(f"- {EI.get('note')}; folds {EI.get('folds')}")
        a += table(["fold", "g2_raw", "model", "n_primary"],
                   [[k, ((v or {}).get("g2_raw") or {}).get("primary_share"),
                     ((v or {}).get("model") or {}).get("primary_share"),
                     ((v or {}).get("model") or {}).get("n_primary")] for k, v in (EI.get("per_fold") or {}).items()])
    else:
        A("- 파일 없음")
    A("")
    A("### 5-3. LADDER 기준값 (`analysis/geo2/results/selftest/E_metrics.csv`: REP8, para; 명령서 0-4절)")
    a += quote(sec("### 0-4."))
    A("")
    B = D["baseline"]
    a += table(["타깃", "P_23 g2 n", "P_23 g2 MAE", "CI", "P_023 dft MAE", "CI", "P_023 g2 MAE"],
               [[r["target"], r["P_23_g2"]["n"], r["P_23_g2"]["mae"],
                 f"[{f6(r['P_23_g2']['ci_lo'])}, {f6(r['P_23_g2']['ci_hi'])}]", r["P_023_dft"]["mae"],
                 f"[{f6(r['P_023_dft']['ci_lo'])}, {f6(r['P_023_dft']['ci_hi'])}]", r["P_023_g2"]["mae"]]
                for r in B["rows"]])
    A("")
    A(f"- REP8 (selftest = LADDER E2 검증 MAE 기준 선택): {B['rep8']}")
    A("")
    A("### 5-4. 작성자의 에너지 환산 예측 (추정치; 판정에 쓰지 않음)")
    A(f"- 명령서 8-3 항목 5: \"{AUTHOR_ESTIMATE}\". 근거: GEO1 짝 비율 변화(5-1절)와 0-4절의 같은 형태 / 다른 형태 짝 "
      "g2 oi 차이. 이 수치는 결과를 보기 전의 예상이며 판정 기준이 아니다.")
    A("")
    A("### 5-5. 8-2 검사와 smoke 예상")
    A(f"- `{D['C_tests']['file']}` all_pass {D['C_tests']['all_pass']} (sha256 `{D['C_tests']['sha256']}`, merged "
      f"{D['C_tests'].get('merged')}); 검사별 {D['C_tests']['checks'] or '(checks 블록 없음)'}")
    A(f"- 검사가 돈 코드 = 지금 코드(code_sha256 {sorted(D['C_tests']['code_sha256'])} 모두 같음), 검사의 geo 입력표 "
      f"sha256 = 7-1절의 geo 입력표, 라벨 = 7-1절의 라벨 파일 → 이 문서가 고정하는 코드·입력은 8-2 검사를 통과한 그것이다.")
    sm = D["smoke"]
    e = sm.get("estimate_geo") or {}
    A(f"- smoke (rung geo, fold 0): 원자 모델 ok {sm.get('atoms_ok')}, XGB ok {sm.get('xgb_ok')} (gate M-1′).")
    A(f"- smoke 예상 (train_atoms GA-9): geo 원자 모델 {f6(sm.get('geo_gpu_hours'))} GPU시간 ≤ 상한 {GEO_LIMIT_GPU_H} "
      f"(명령서 8-4; FU2 상한식 {f6(sm.get('upper_bound_gpu_hours'))}; LADDER rung별 실제 "
      f"{e.get('ladder_rung_atom_gpu_hours')}); XGB {f6(sm.get('xgb_cpu_hours'))} CPU시간. 파일: `{sm['atoms_file']}`, "
      f"`{sm['xgb_file']}`.")
    A("")
    # ---------------------------------------------------------- 6
    A("## 6. 알려진 한계 (8-3 항목 6: 명령서 10-3절 그대로)")
    a += quote(sec("### 10-3."))
    A("")
    # ---------------------------------------------------------- 7
    A("## 7. gate와 멈춤 규칙")
    A("")
    A("### 7-1. gate (학습 코드가 매 실행 전에 확인; `train_atoms.py` GA-4 .. GA-7, `train_xgb.py` XG-G3)")
    A("- 이 md와 `prereg_geo2.json`이 git에 있고 HEAD 이후 바뀌지 않았다; sha256(md) = json `md_sha256`; json `rungs` = "
      "[\"geo\"].")
    A("- json `inputs_sha256`에 라벨 파일과 geo 입력표가 있고(need), 아래 표의 파일이 모두 그 sha256 그대로다.")
    A(f"- `analysis/d1_ladder/results`의 LADDER 결과 파일 {len(D['ladder_snapshot'])}개(json `ladder_results_sha256`, "
      "repo 상대 경로; 이름에 'geo'가 든 GEO2 출력·`E3_*`·`*.tmp` 제외; 작성 때 git HEAD와 같음을 확인)가 그대로다"
      "(명령서 14절: 기존 LADDER 결과 파일이 바뀜 = STOP). `collect --stage 3`도 다시 확인한다.")
    A("- GEO2 P: `P_build_geo.json` all_pass·입력표 sha256, `P_geo_report.json` stop false, `P_geo_sets.json`의 "
      "|P_23| ≥ 0.98 |P_g2|, 라벨 파일 = LADDER `P_report.json` 기록.")
    A("- `analysis/geo2/results/C_tests_geo.json` all_pass, 그리고 그 code_sha256의 train_atoms.py·train_xgb.py·"
      "inputs.py·lad_common.py = 지금 파일(학습 코드 GA-14 / XG-G10).")
    A("")
    a += table(["파일", "역할", "열", "sha256"], [[f"`{rel(p)}`", r, c, f"`{s}`"] for p, r, c, s in D["inputs"]])
    A("")
    A(f"- 학습 코드 git 상태(작성 시): {D['code_git']}")
    A("- geo unit 지문의 code_sha256 기준값(json `code.fingerprint_keys`; `evaluate.py e3`가 40 unit을 이것과 비교할 수 "
      "있다): " + "; ".join(f"{t}: " + ", ".join(f"{k} `{v[:12]}`" for k, v in m.items())
                           for t, m in D["fingerprint_keys"].items()))
    A("")
    A("### 7-2. 명령서 13절 (HPC 규칙, 그대로)")
    a += quote(sec("## 13."))
    A("")
    A("### 7-3. 명령서 14절 (결정 기록과 STOP, 그대로)")
    a += quote(sec("## 14."))
    A("")
    A("- **원칙:** 결과를 본 뒤 판정·선택 규칙·짝 집합을 바꾸지 않는다. 바꾸면 이탈로 기록한다(명령서 13절).")
    return "\n".join(a) + "\n"


# ================================================================ json
def json_rec(D, md_sha):
    return dict(
        date=D["date"], written=D["written"], code_version=CODE_VERSION, stage=3, rungs=["geo"],
        spec="docs/specs/GEO2.md", spec_sha256=D["spec_sha256"], spec_git=D["spec_git"],
        ladder_spec="docs/specs/D1LADDER.md", ladder_spec_sha256=D["lad_spec_sha256"], ladder_spec_git=D["lad_spec_git"],
        md=rel(MD), md_sha256=md_sha, generator=D["generator"], git_head=D["git_head"], branch=D["branch"],
        need=need_paths(),
        inputs_sha256={p: s for p, _, _, s in D["inputs"]},
        inputs_roles={p: dict(role=r, column=c) for p, r, c, _ in D["inputs"]},
        ladder_results_sha256=D["ladder_snapshot"],
        gate=dict(files=[rel(MD), rel(JS)],
                  rule="both files tracked + unchanged since HEAD; sha256(md) = md_sha256; rungs == ['geo']; need ⊆ "
                       "inputs_sha256; every listed file unchanged (files only in other rungs' columns are skipped, "
                       "MA-15); every ladder_results_sha256 file unchanged (GA-5)",
                  code="analysis/d1_ladder/train_atoms.py gate_prereg_geo2, train_xgb.py gate_prereg_geo2; also "
                       "gate_p_geo (P_build_geo / P_geo_report / P_geo_sets / label record) and gate_c_geo "
                       "(C_tests_geo.json all_pass + its code_sha256 of train_atoms.py / train_xgb.py / inputs.py / "
                       "lad_common.py = the files now, GA-14 / XG-G10)"),
        units=units(),
        pair_sets={k: dict(v, ids_sha256_rule="sha256 of newline-joined sorted pair_id") for k, v in D["sets"].items()},
        pair_sets_extra=D["sets_extra"], pair_sets_sha_in_p_geo_report=D["sets_sha_in_report"],
        code=dict(fingerprint_keys=D["fingerprint_keys"],
                  rule="every geo unit json fingerprint code_sha256 = fingerprint_keys of its trainer (train_xgb for "
                       "XGB, train_atoms for AW / GNN / TF); key names = the trainers' CODE_SHA (= LADDER prereg.py "
                       "CODE_ATOMS / CODE_XGB; the ADD3 11 E / E2-15 analogue for evaluate.py e3, D-41)",
                  c_tests_geo_code_sha256=D["C_tests"]["code_sha256"]),
        labels=D["labels"], geo_table=D["P_build"], code_git=D["code_git"], evidence=D["evidence"], seen=D["seen"],
        replaced_draft=D["replaced_draft"],
        numbers=dict(geo1_E_report=D["E_report"], geo1_E_interim=D["E_interim"], Y_plan=D["Y_plan"],
                     Z_report=D["Z_report"], P_geo_report=D["P_geo_report"], ladder_baseline_selftest=D["baseline"],
                     C_tests_geo=D["C_tests"], smoke=D["smoke"], author_estimate=AUTHOR_ESTIMATE),
        grid_source=D["grid_src"])


# ================================================================ commands
def cmd_write():
    D = gather()
    text = md_text(D)
    C.write_atomic(MD, text)
    md_sha = C.sha256(MD)
    if md_sha != hashlib.sha256(text.encode()).hexdigest():
        stop(f"{MD}: the written file differs from the generated text")
    rec = json_rec(D, md_sha)
    C.write_json(JS, rec)
    print(json.dumps(dict(md=rel(MD), md_sha256=md_sha, json=rel(JS), json_sha256=C.sha256(JS),
                          n_inputs=len(rec["inputs_sha256"]), n_ladder_results=len(rec["ladder_results_sha256"]),
                          sets={k: v["n"] for k, v in D["sets"].items()},
                          next="commit + push both files (and the code) before train_atoms / train_xgb run --stage 3"),
                     indent=1), flush=True)
    return 0


def cmd_check():
    """= the trainers' gate_prereg_geo2 (every listed file verified; no rung-column skipping is needed: the list holds
    geo, shared and code files only) -> results/R_check_geo2.json."""
    bad = []
    for p in (MD, JS):
        if not p.is_file():
            bad.append(f"{p.name} missing")
        elif not tracked(p):
            bad.append(f"{p.name} not committed")
        elif git("diff", "--quiet", "HEAD", "--", str(p)).returncode != 0:
            bad.append(f"{p.name} modified since its commit")
    rec, changed = {}, []
    if not bad:
        rec = json.loads(JS.read_text())
        if C.sha256(MD) != rec.get("md_sha256"):
            bad.append("md sha256 != json md_sha256")
        if list(rec.get("rungs") or []) != ["geo"]:
            bad.append(f"rungs {rec.get('rungs')} != ['geo']")
        ins = rec.get("inputs_sha256") or {}
        lack = sorted(set(need_paths()) - set(ins))
        if lack:
            bad.append(f"inputs_sha256 lacks {lack}")
        for p, s in ins.items():
            if not Path(p).is_file() or C.sha256(p) != s:
                bad.append(f"input {p} changed or missing")
        snap = rec.get("ladder_results_sha256") or {}
        if not snap:
            bad.append("no ladder_results_sha256 snapshot")
        changed = [r for r, s in sorted(snap.items()) if not (REPO / r).is_file() or C.sha256(REPO / r) != s]
        if changed:
            bad.append(f"LADDER result files changed: {changed[:8]}")
    out = dict(checked=now(), ok=not bad, bad=bad, md=rel(MD), md_sha256=sha(MD), json=rel(JS), json_sha256=sha(JS),
               n_inputs=len(rec.get("inputs_sha256") or {}),
               n_ladder_results=len(rec.get("ladder_results_sha256") or {}), ladder_changed=changed,
               git_head=git("rev-parse", "HEAD").stdout.strip())
    C.write_json(CHECK_OUT, out)
    print(json.dumps(out, indent=1), flush=True)
    return 0 if not bad else 3


if __name__ == "__main__":
    if sys.argv[1:] == ["write"]:
        sys.exit(cmd_write())
    if sys.argv[1:] == ["check"]:
        sys.exit(cmd_check())
    raise SystemExit(__doc__)
