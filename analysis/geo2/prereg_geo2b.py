#!/usr/bin/env python3
r"""prereg_geo2b.py — GEO2b Phase R (design step 11 of analysis/geo1b/results/D0D1_design_investigation.json; docs/specs/
GEO2.md 8-3 read with GEO1B.md 4-3 and the user decision D0D1 2026-10-31): the pre-registration
results/PREREG_GEO2B.md + results/prereg_geo2b.json, written ONCE, before any stage-3 training. No comparison verdict
(GEO2.md 10-1 dropped by D0D1): descriptive results only, D1 and D0 separately, the LADDER goal as the only absolute
target. Compute node only (sbatch / srun); conda reactot (pandas + git): analysis/geo2/jobs/m3_cpu.slurm M3ENV=reactot.

CLI (exit 0 = done, 2 = not ready (an input missing; nothing written), 3 = STOP / refused)
  python prereg_geo2b.py write   -> results/PREREG_GEO2B.md + results/prereg_geo2b.json. Refused (exit 3) when either
        file is tracked by git (a committed pre-registration is never rewritten; an untracked draft is replaced).
        Needs (else exit 2): analysis/geo1b/results/E_report.json (GEO1b E1), results/Z_report.json, P_build_geo.json,
        P_geo_report.json, S_splits_d0d1.json (final), C_tests_geo.json, analysis/d1_ladder/results/C_smoke_atoms_geo.json,
        C_smoke_xgb_geo.json, D0D1_diag_geo.json, the gated input files. STOP (exit 3): Z / P stop, the geo table sha256
        != P_build_geo outputs, the D0D1 index not final / not current (d0d1_splits.load_s_splits), F_pairs_cr2.csv !=
        V_report outputs, C_tests_geo.json not all_pass or not run on the code / geo table / labels now, a stage-3 smoke
        not passed, the GPU projection not computed or > 160 GPU-h, a stage-3 unit / prediction / evaluation file
        already exists, a tracked analysis/d1_ladder/results file differs from git HEAD, a spec heading missing.
  python prereg_geo2b.py check   the stage-3 gate (train_atoms / train_xgb gate_prereg_geo2) without training: both files
        tracked and unchanged since HEAD, sha256(md) = md_sha256, rungs = ["geo"], split = "T1D0", need ⊆ inputs_sha256,
        every listed file and the LADDER result snapshot unchanged -> results/R_check_geo2b.json (exit 0 / 3).

json keys the stage-3 gate reads (train_atoms.gate_prereg_geo2 / train_xgb.gate_prereg_geo2): md_sha256; rungs ["geo"];
  split "T1D0"; inputs_sha256 {absolute unresolved path: sha256} holding need = $GEO2B/F_pairs_d0d1.csv and
  $GEO2B/inputs/geo.parquet plus the other training inputs and the training code; ladder_results_sha256
  {repo-relative path: sha256} = every analysis/d1_ladder/results file except GEO2 outputs ('geo' in the name, E3_*,
  *.tmp). For evaluate.py e3 (not read by the gate): code.fingerprint_keys {train_atoms: {...}, train_xgb: {...}}.
Everything else is a record (numbers copied from the files, sha256 of evidence files, git states). Decisions: results/
DECISIONS_lad.md (LP-*).
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import d0d1_splits as DD  # noqa: E402

import pandas as pd  # noqa: E402

CODE_VERSION = "geo2b-prereg-1"
REPO = DD.REPO
RES = HERE / "results"
GEO2 = DD.GEO2
SPEC, SPEC_B, LAD_SPEC = (REPO / "docs" / "specs" / n for n in ("GEO2.md", "GEO1B.md", "D1LADDER.md"))
DESIGN = REPO / "analysis" / "geo1b" / "results" / "D0D1_design_investigation.json"
GEO1B_DEC = REPO / "analysis" / "geo1b" / "results" / "DECISIONS.md"
MD, JS = RES / "PREREG_GEO2B.md", RES / "prereg_geo2b.json"
CHECK_OUT = RES / "R_check_geo2b.json"
LAD = DD.LAD
D1PAIR_SCR = DD.D1PAIR_SCR
LAD_DIR = REPO / "analysis" / "d1_ladder"
LAD_RES = LAD_DIR / "results"
P_DIR = REPO / "analysis" / "d1_pair"
GEO1B_RES = DD.GEO1B_RES
TABLE = GEO2 / "inputs" / "geo.parquet"             # = train_atoms.inputs_path("geo")
X182 = GEO2 / "X182_d1_geo.parquet"                 # = train_xgb.RUNGS["geo"]["x182"]
FU2_GEO = RES / "fu2_geo"                           # = train_atoms.RUNGS["geo"]["fu2_res"]
Z_REPORT, P_BUILD, P_REPORT = RES / "Z_report.json", RES / "P_build_geo.json", RES / "P_geo_report.json"
C_TESTS = RES / "C_tests_geo.json"
S_FOLDS = P_DIR / "results" / "S_folds.csv"
ARMS, MODELS4, NF = ("S", "SH"), ("AW", "GNN", "TF", "XGB"), 5
TARGETS = ["d1", "d2", "elst", "pauli", "oi", "disp", "cpcm", "cds", "c_ghost", "barrier"]
GEO_LIMIT_GPU_H = 160.0                             # GEO1b DECISIONS D0D1 '상한 조정' (GEO2.md 8-4 had 80)
FU2_DIR = P_DIR / "fu2"
ESP_DIR = REPO / "analysis" / "espley_xtb_repro"
CODE = [LAD_DIR / "train_atoms.py", LAD_DIR / "train_xgb.py", LAD_DIR / "lad_common.py", LAD_DIR / "inputs.py",
        FU2_DIR / "models.py", FU2_DIR / "train.py", FU2_DIR / "fu2_common.py", FU2_DIR / "tokens.py",
        FU2_DIR / "atoms.py", P_DIR / "d1pair_common.py", P_DIR / "d1_train.py", ESP_DIR / "train_ml_single.py",
        HERE / "d0d1_splits.py"]
# the trainers' fingerprint code_sha256 key names (train_atoms.CODE_SHA / train_xgb.CODE_SHA; LADDER prereg.py rule)
CODE_ATOMS = {"train_atoms.py": LAD_DIR / "train_atoms.py", "lad_common.py": LAD_DIR / "lad_common.py",
              "fu2/models.py": FU2_DIR / "models.py", "fu2/train.py": FU2_DIR / "train.py",
              "fu2/fu2_common.py": FU2_DIR / "fu2_common.py", "fu2/tokens.py": FU2_DIR / "tokens.py",
              "d1pair_common.py": P_DIR / "d1pair_common.py"}
CODE_XGB = {"train_xgb.py": LAD_DIR / "train_xgb.py", "lad_common.py": LAD_DIR / "lad_common.py",
            "d1_train.py": P_DIR / "d1_train.py", "d1pair_common.py": P_DIR / "d1pair_common.py",
            "train_ml_single.py": ESP_DIR / "train_ml_single.py", "fu2/fu2_common.py": FU2_DIR / "fu2_common.py",
            "fu2/tokens.py": FU2_DIR / "tokens.py"}
# C_tests_geo.json code_sha256 keys (run_tests.py GEO_CODE_FILES) that must equal the files now (LP-3: evaluate.py is
# recorded, not required: it is evaluation code of the Z / E role, not training code)
C_CODE_REQ = {"tests/run_tests.py": LAD_DIR / "tests" / "run_tests.py", "inputs.py": LAD_DIR / "inputs.py",
              "train_atoms.py": LAD_DIR / "train_atoms.py", "train_xgb.py": LAD_DIR / "train_xgb.py",
              "lad_common.py": LAD_DIR / "lad_common.py", "geo2/d0d1_splits.py": HERE / "d0d1_splits.py"}
C_CODE_REC = {"evaluate.py": LAD_DIR / "evaluate.py"}


# ================================================================ helpers
def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def not_ready(msg):
    print(f"NOT READY: {msg}", file=sys.stderr, flush=True)
    sys.exit(2)


def stop(msg):
    print(f"STOP: {msg}", file=sys.stderr, flush=True)
    sys.exit(3)


def ab(p):
    return os.path.abspath(str(p))


def rel(p):
    q = Path(ab(p))
    try:
        return str(q.relative_to(REPO))
    except ValueError:
        return str(q)


def sha(p):
    return DD.sha256(p) if Path(p).is_file() else None


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


def write_text(p, text):
    DD.write_bytes_atomic(p, text.encode())


def f6(x):
    if isinstance(x, bool) or x is None:
        return str(x)
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        return f"{x:.6g}"
    return str(x)


def compact(o, n=20):
    """json value with lists longer than n replaced by {'n': len, 'head': first 5}."""
    if isinstance(o, dict):
        return {str(k): compact(v, n) for k, v in o.items()}
    if isinstance(o, list):
        return [compact(v, n) for v in o] if len(o) <= n else {"n": len(o), "head": [compact(v, n) for v in o[:5]]}
    return o


def quote(text):
    return [("> " + ln) if ln.strip() else ">" for ln in text.split("\n")]


def section(text, head, name):
    """verbatim lines of a spec from the line starting with `head` up to (not including) the next heading of the same
    or a higher level, a code fence or a code-introduction line; STOP when `head` is not unique."""
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


def bullet(text, prefix, name):
    """a top-level markdown bullet (line starting with `prefix`) with its indented continuation lines; STOP if absent."""
    lines = text.split("\n")
    hits = [i for i, ln in enumerate(lines) if ln.startswith(prefix)]
    if len(hits) != 1:
        stop(f"{name}: bullet {prefix!r} found {len(hits)} times")
    out = [lines[hits[0]]]
    for ln in lines[hits[0] + 1:]:
        if ln.startswith("- ") or ln.startswith("#"):
            break
        out.append(ln)
    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out)


def assign_src(path, name, key=None):
    """verbatim source of the module-level assignment to `name`, or of its dict entry `key`."""
    t = Path(path).read_text(encoding="utf-8")
    for node in ast.parse(t).body:
        tg = node.targets if isinstance(node, ast.Assign) else ([node.target] if isinstance(node, ast.AnnAssign) else [])
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
    """= train_atoms / train_xgb cmd_run need with the stage-3 default labels."""
    return [ab(DD.LABELS), ab(TABLE)]


def gated_inputs():
    """(absolute path, role, column) of every file the stage-3 gate re-verifies (inputs_sha256)."""
    sp = D1PAIR_SCR / "fu2" / "splits"
    R = [(DD.LABELS, "labels: F_pairs_d0d1.csv = F_pairs_cr2.csv + D0 rows (need)", "geo"),
         (TABLE, "geo input table D1 + D0 (inputs role, GEO2.md 6-1 P; need)", "geo"),
         (X182, "geo 182 global features (XGB input = token glob source)", "geo")]
    R += [(FU2_GEO / n, f"geo FU2 report {n} (token layout / versions / manifest)", "geo")
          for n in ("T_report.json", "T_manifest.csv", "Q_report.json", "S_splits.json")]
    R += [(DD.S_SPLITS, "D0D1 split index S_splits_d0d1.json (d0d1_splits.py build)", "geo")]
    R += [(DD.SPLITS / DD.split_name(k), f"T1D0 fold {k} split file", "geo") for k in range(NF)]
    R += [(DD.CR2, "F_pairs_cr2.csv (the D1 rows of the label file, byte-identical; LADDER y)", "shared"),
          (DD.ORIG, "original F_pairs.csv (L-11 invariant of the D1 rows)", "shared"),
          (DD.FU2_S_SPLITS, "FU2 S_splits.json (split index of the D1 lists)", "shared"),
          (S_FOLDS, "S_folds.csv (D1 folds, cores; XGB inner GroupKFold)", "shared"),
          (LAD_RES / "V_report.json", "label provenance of F_pairs_cr2.csv (V_report outputs)", "shared")]
    R += [(sp / f"T1__f{k}.json", f"FU2 T1 fold {k} split file (the D1 part of T1D0)", "shared") for k in range(NF)]
    R += [(DD.D0_PAIRS, "GEO1b D0_pairs.csv (the D0 rows; final build)", "geo"),
          (DD.D0_FOLDS, "GEO1b d0_folds.csv (frozen D0 folds)", "geo"),
          (DD.D0_TWINS, "GEO1b d0_twins.csv (D1 twins of D0 reactions)", "geo")]
    R += [(p, "stage-3 training code", "code") for p in CODE]
    return [(ab(p), r, c) for p, r, c in R]


def evidence_paths():
    """files the md cites (sha256 + git state recorded, not gated)."""
    E = [SPEC, SPEC_B, LAD_SPEC, DESIGN, GEO1B_DEC, DD.D0_REPORT, DD.Q_D0_REPORT, GEO1B_RES / "DECISIONS_PAIR.md",
         GEO1B_RES / "E_report.json", GEO1B_RES / "E_rmsd.json", GEO1B_RES / "E_main.csv", GEO1B_RES / "E_rxn.csv",
         GEO1B_RES / "E_rxn_d0.csv", GEO1B_RES / "Y_plan.json"] + [GEO1B_RES / f"R_select_f{k}.json" for k in range(NF)]
    E += [Z_REPORT, RES / "Z_status.csv", P_BUILD, P_REPORT, C_TESTS, RES / "C_tests_geo_gpu.json",
          RES / "C_tests_geo_cpu.json", HERE / "eval_geo2.py", HERE / "build_geo.py", HERE / "geo2_common.py",
          Path(__file__).resolve(), LAD_DIR / "evaluate.py", LAD_DIR / "tests" / "run_tests.py",
          LAD_RES / "C_smoke_atoms_geo.json", LAD_RES / "C_smoke_xgb_geo.json", LAD_RES / "C_l6_xgb_geo.json",
          LAD_RES / "D0D1_diag_geo.json", RES / "selftest" / "E_report.json"]
    return E


def geo2_output(p):
    """GEO2's own outputs inside analysis/d1_ladder/results (excluded from the LADDER snapshot)."""
    r = str(Path(p).relative_to(LAD_RES)).lower()
    return "geo" in r or Path(p).name.startswith("E3_") or r.endswith(".tmp")


def ladder_snapshot():
    return {rel(p): DD.sha256(p) for p in sorted(LAD_RES.rglob("*")) if p.is_file() and not geo2_output(p)}


def ladder_git_changes():
    r = git("status", "--porcelain", "--untracked-files=no", "--", str(LAD_RES))
    if r.returncode != 0:
        stop(f"git status failed on {LAD_RES}: {r.stderr.strip()[:300]}")
    return [ln for ln in r.stdout.splitlines() if ln.strip()]


# ================================================================ gather (read only)
def d0d1_index():
    """LP-1: the D0D1 index passes the trainers' reader contract now (final) -> record."""
    try:
        rec, lists, files = DD.load_s_splits(allow_preliminary=False)
    except DD.NotReady as e:
        not_ready(str(e))
    except DD.Stop as e:
        stop(f"D0D1 index: {e}")
    return dict(file=rel(DD.S_SPLITS), sha256=DD.sha256(DD.S_SPLITS), labels=rec.get("labels"), d0=rec.get("d0"),
                units={k: dict(path=v.get("path"), sha256=v.get("sha256"), n=v.get("n"), n_d0=v.get("n_d0"))
                       for k, v in (rec.get("units") or {}).items()},
                rules=rec.get("rules"), twins=rec.get("twins"), inputs_sha256=rec.get("inputs_sha256"),
                code_version=rec.get("code_version"), built=rec.get("finished"))


def c_tests():
    """C_tests_geo.json all_pass in the full geo mode AND run on what this pre-registration freezes (LP-3)."""
    c = read_json(C_TESTS, "tests/run_tests.py geo (GEO2.md 8-2)")
    if not c.get("all_pass"):
        stop(f"{C_TESTS}: all_pass {c.get('all_pass')} (8-2 STOP: fix and rerun the tests before the PREREG)")
    bad = []
    if c.get("mode") not in (None, "geo") or c.get("training_gate") is False:
        bad.append(f"mode {c.get('mode')} / training_gate {c.get('training_gate')}: not the full geo mode")
    rec = c.get("code_sha256") if isinstance(c.get("code_sha256"), dict) else {}
    lack = [k for k in C_CODE_REQ if k not in rec]
    if lack:
        bad.append(f"code_sha256 lacks {lack} (interface: tests/run_tests.py GEO_CODE_FILES)")
    diff = sorted(k for k, p in C_CODE_REQ.items() if sha(p) is None or rec.get(k) != sha(p))
    if diff:
        bad.append(f"code changed since the C tests: {diff} (the PREREG would freeze untested code)")
    geo = (c.get("rungs") or {}).get("geo") or {}
    tsha = sha(TABLE)
    if geo.get("status") != "ready" or tsha is None or geo.get("table_sha256") != tsha:
        bad.append(f"rungs.geo status {geo.get('status')}, table_sha256 {str(geo.get('table_sha256'))[:12]} != now "
                   f"{str(tsha)[:12]}")
    lg = c.get("labels_geo") or {}
    if c.get("labels_geo_final") is not True or lg.get("sha256") != sha(DD.LABELS) \
            or lg.get("s_splits_sha256") != sha(DD.S_SPLITS):
        bad.append(f"labels_geo final {c.get('labels_geo_final')}, sha256 / index differ from the files now")
    lab = c.get("labels") or {}
    if c.get("labels_final") is not True or lab.get("sha256") != sha(DD.CR2):
        bad.append(f"LADDER labels {lab.get('path')} (final {c.get('labels_final')}) != F_pairs_cr2.csv now")
    for k in ("consistency", "current"):
        if (c.get(k) or {}).get("ok") is not True:
            bad.append(f"{k}.ok {(c.get(k) or {}).get('ok')}")
    if bad:
        stop(f"{C_TESTS} passed, but not on what this pre-registration freezes (LP-3; rerun the C tests, both parts + "
             f"merge, on the final code and inputs): " + "; ".join(bad))
    st = {}
    for k, v in (c.get("checks") or {}).items():
        st[str(k)] = v.get("status") if isinstance(v, dict) else v
    return dict(file=rel(C_TESTS), sha256=DD.sha256(C_TESTS), all_pass=True, checks=st, mode=c.get("mode"),
                code_sha256=rec, code_equal_now=True, table_sha256=tsha,
                evaluate_py=dict(recorded=rec.get("evaluate.py"), now=sha(C_CODE_REC["evaluate.py"]),
                                 equal=rec.get("evaluate.py") == sha(C_CODE_REC["evaluate.py"])),
                gpu_estimate_geo=c.get("gpu_estimate_geo"), merged=c.get("merged"))


def smoke_estimates():
    """LP-4 (GEO2.md 8-4 / 14 with the 160 GPU-h cap): both stage-3 smoke reports exist (exit 2), stage 3, ok; the atom
    projection computed and <= 160 GPU-h; STOP otherwise."""
    a, x = LAD_RES / "C_smoke_atoms_geo.json", LAD_RES / "C_smoke_xgb_geo.json"
    for p in (a, x):
        if not p.is_file():
            not_ready(f"{p} missing (the C tests' smoke or RUNBOOK_lad S-2 / S-3)")
    sa, sx = json.loads(a.read_text()), json.loads(x.read_text())
    e = sa.get("estimate_geo") if isinstance(sa.get("estimate_geo"), dict) else {}
    out = dict(atoms_file=rel(a), atoms_sha256=DD.sha256(a), xgb_file=rel(x), xgb_sha256=DD.sha256(x),
               atoms_ok=sa.get("ok"), estimate_geo=e, upper_bound_gpu_hours=sa.get("stage_upper_bound_gpu_hours"),
               xgb_ok=sx.get("ok"), xgb_cpu_hours=sx.get("stage_cpu_hours_estimate"), limit_gpu_hours=GEO_LIMIT_GPU_H,
               atoms_code_now=sa.get("code_sha256") == {k: sha(p) for k, p in CODE_ATOMS.items()},
               xgb_code_now=sx.get("code_sha256") == {k: sha(p) for k, p in CODE_XGB.items()})
    bad = []
    for name, s in (("atom", sa), ("XGB", sx)):
        if str(s.get("stage")) != "3":
            bad.append(f"{name} smoke stage {s.get('stage')} (not the geo smoke)")
        if s.get("ok") is not True:
            bad.append(f"{name} smoke ok {s.get('ok')} (gate M-1' failed)")
    h = float("nan")
    if e.get("ok") is not True:
        bad.append(f"estimate_geo not computed ({e.get('why')}): the 160 GPU-h rule cannot be checked")
    else:
        try:
            h = float(e.get("geo_gpu_hours"))
        except (TypeError, ValueError):
            pass
        if not math.isfinite(h):
            bad.append(f"estimate_geo.geo_gpu_hours {e.get('geo_gpu_hours')!r} is not a finite number")
        elif h > GEO_LIMIT_GPU_H:
            bad.append(f"smoke projection {h:.1f} GPU-h > {GEO_LIMIT_GPU_H} (GEO1b DECISIONS D0D1)")
    if bad:
        stop("stage-3 smoke / GPU projection (STOP): " + "; ".join(bad))
    out.update(geo_gpu_hours=h, within_limit=True)
    return out


def diag():
    """the s_y / SH beta diagnostic (train_xgb.py diag --stage 3) -> per arm the median over folds per target."""
    p = LAD_RES / "D0D1_diag_geo.json"
    rep = read_json(p, "train_xgb.py diag --stage 3 (C tests diag)")
    rows = {}
    for key, v in (rep.get("folds") or {}).items():
        _, arm, fold = key.split("/")
        for fit in ("final_T_main", "inner_fold0"):
            e = v.get(fit) or {}
            rows.setdefault((arm, fit), []).append(e)
    summ = {}
    for (arm, fit), es in rows.items():
        r = dict(n_folds=len(es), n_d0_median=float(pd.Series([e.get("n_d0") for e in es]).median()))
        for name in ("s_y",):
            for side in ("d1", "d1_d0"):
                r[f"{name}_{side}_median"] = {t: float(pd.Series([e[side][name][j] for e in es]).median())
                                              for j, t in enumerate(TARGETS)}
        if arm == "SH":
            for side in ("d1", "d1_d0"):
                r[f"beta_{side}_median"] = {t: float(pd.Series([e[side]["beta"][j] for e in es]).median())
                                            for j, t in enumerate(TARGETS)}
        summ[f"{arm}/{fit}"] = r
    return dict(file=rel(p), sha256=DD.sha256(p), summary=summ)


def seen():
    """stage-3 units / predictions / evaluation must not exist yet (training before the PREREG = STOP)."""
    ud = LAD / "units"
    geo_units = sorted(p.name for p in ud.glob("L__geo__*")) if ud.is_dir() else []
    files = [p for p in (LAD_RES / "L_predictions_geo.parquet", LAD_RES / "L_report_geo.json",
                         LAD_RES / "L_xgb_report_geo.json", LAD_RES / "L_val_geo.csv", RES / "E_report.json",
                         RES / "E_metrics.csv", RES / "E_compare.csv", RES / "E_d0_metrics.csv") if p.is_file()]
    out = dict(geo_unit_files=geo_units, prediction_or_evaluation_files=[rel(p) for p in files],
               e3_files=sorted(p.name for p in LAD_RES.glob("E3_*")),
               smoke_dirs={str(d): d.is_dir() for d in (LAD / "smoke_geo", LAD / "smoke_xgb_geo")})
    if geo_units or files:
        stop(f"stage-3 training / evaluation output exists before the pre-registration: "
             f"{(geo_units + [str(p) for p in files])[:6]}")
    return out


def design():
    d = read_json(DESIGN, "GEO1b D0D1 design investigation")
    rd = ((d.get("result") or {}).get("design") or {})
    text = rd.get("recommended_design") or ""

    def part(head, nxt):
        i = text.find(head)
        if i < 0:
            stop(f"{DESIGN.name}: '{head}' not in recommended_design")
        j = text.find(nxt, i + len(head)) if nxt else -1
        return text[i:j if j >= 0 else None].strip()
    return dict(training=part("TRAINING", "RESULTS"), results=part("RESULTS", "RISKS ACCEPTED"),
                risks_accepted=part("RISKS ACCEPTED", None), risks=rd.get("risks") or [],
                step9=(rd.get("steps") or [])[9:13])


def gather():
    D = dict(date=time.strftime("%Y-%m-%d"), written=now())
    for p in (MD, JS):
        if p.is_file() and tracked(p):
            stop(f"{rel(p)} is tracked by git: a committed pre-registration is written once and never replaced "
                 f"(a change needs a user decision and a new, separately named addendum)")
    D["replaced_draft"] = {rel(p): sha(p) for p in (MD, JS) if p.is_file()}
    D["seen"] = seen()
    for p in (SPEC, SPEC_B, LAD_SPEC, GEO1B_DEC):
        if not p.is_file():
            not_ready(f"{p} missing")
    D["spec_text"], D["spec_b_text"] = SPEC.read_text(encoding="utf-8"), SPEC_B.read_text(encoding="utf-8")
    D["lad_spec_text"] = LAD_SPEC.read_text(encoding="utf-8")
    D["d0d1_decision"] = bullet(GEO1B_DEC.read_text(encoding="utf-8"), "- **D0D1 (사용자 2026-10-31",
                                "analysis/geo1b/results/DECISIONS.md")
    D["design"] = design()
    D["E_report"] = read_json(GEO1B_RES / "E_report.json", "GEO1b evaluate.py (E1)")
    er = GEO1B_RES / "E_rmsd.json"
    D["E_rmsd"] = json.loads(er.read_text()) if er.is_file() else None
    z = read_json(Z_REPORT, "build_geo.py build (GEO2.md 4)")
    if z.get("stop"):
        stop(f"{Z_REPORT}: stop {z.get('stop_reasons')} (GEO2.md 4-3)")
    D["Z_report"] = compact(z)
    b = read_json(P_BUILD, "inputs role P (GEO2.md 6-1)")
    if not b.get("all_pass") or b.get("stop"):
        stop(f"{P_BUILD}: all_pass {b.get('all_pass')}, stop {b.get('stop')} (GEO2.md 6-2 P)")
    want = (b.get("outputs") or {}).get(str(TABLE))
    if not TABLE.is_file() or want != DD.sha256(TABLE):
        stop(f"{TABLE} sha256 != {P_BUILD.name} outputs ({want})")
    D["P_build"] = dict(file=rel(P_BUILD), sha256=DD.sha256(P_BUILD), table=str(TABLE), table_sha256=want,
                        rung_cfg=b.get("rung_cfg"), table_summary=compact(b.get("table")))
    pr = read_json(P_REPORT, "inputs role P report (GEO2.md 7)")
    if pr.get("stop"):
        stop(f"{P_REPORT}: stop {pr.get('stop_reasons') or pr.get('stop_reason')} (GEO2.md 7)")
    D["P_geo_report"] = compact(pr)
    v = read_json(LAD_RES / "V_report.json", "LADDER Phase V")
    if not v.get("all_pass") or (v.get("outputs") or {}).get(str(DD.CR2)) != sha(DD.CR2):
        stop(f"{DD.CR2} sha256 != results/V_report.json outputs (or V not all_pass)")
    D["index"] = d0d1_index()
    D["C_tests"] = c_tests()
    D["smoke"] = smoke_estimates()
    D["diag"] = diag()
    st = RES / "selftest" / "E_report.json"
    D["selftest"] = compact(json.loads(st.read_text())) if st.is_file() else None
    gi = gated_inputs()
    miss = [p for p, _, _ in gi if not Path(p).is_file()]
    if miss:
        not_ready(f"gated inputs missing: {miss[:6]}")
    D["inputs"] = [(p, r, c, DD.sha256(p)) for p, r, c in gi]
    D["code_git"] = {rel(p): git_state(p) for p in CODE}
    D["fingerprint_keys"] = dict(train_atoms={k: DD.sha256(p) for k, p in CODE_ATOMS.items()},
                                 train_xgb={k: DD.sha256(p) for k, p in CODE_XGB.items()},
                                 d0d1_code=DD.sha256(HERE / "d0d1_splits.py"))
    insha = {p: s for p, _, _, s in D["inputs"]}
    drift = [k for k, p in C_CODE_REQ.items() if ab(p) in insha and insha[ab(p)] != D["C_tests"]["code_sha256"].get(k)]
    drift += [f"{t}:{k}" for t, M in (("train_atoms", CODE_ATOMS), ("train_xgb", CODE_XGB)) for k, p in M.items()
              if insha.get(ab(p)) != D["fingerprint_keys"][t][k]]
    if (D["index"]["labels"] or {}).get("sha256") != insha.get(ab(DD.LABELS)):
        drift.append("F_pairs_d0d1.csv sha256 != the D0D1 index record")
    if drift:
        stop(f"code / labels changed while the pre-registration was being gathered: {drift} (rerun the C tests on the "
             f"final code, then prereg_geo2b.py write)")
    D["evidence"] = {rel(p): dict(sha256=sha(p), git=git_state(p) if str(ab(p)).startswith(str(REPO)) else None)
                     for p in evidence_paths()}
    lg = ladder_git_changes()
    if lg:
        stop(f"existing LADDER result files differ from git HEAD before the snapshot (GEO2.md 14): {lg[:8]}")
    D["ladder_snapshot"] = ladder_snapshot()
    D["git_head"] = git("rev-parse", "HEAD").stdout.strip()
    D["branch"] = git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    for k, p in (("spec", SPEC), ("spec_b", SPEC_B), ("lad_spec", LAD_SPEC)):
        D[f"{k}_sha256"], D[f"{k}_git"] = DD.sha256(p), git_state(p)
    D["generator"] = dict(file=rel(Path(__file__).resolve()), sha256=DD.sha256(Path(__file__).resolve()),
                          git=git_state(Path(__file__).resolve()), code_version=CODE_VERSION)
    D["grid_src"] = dict(
        atoms=assign_src(FU2_DIR / "models.py", "GRID"),
        xgb=assign_src(ESP_DIR / "train_ml_single.py", "GRIDS", "XGB"),
        fu2=[assign_src(FU2_DIR / "fu2_common.py", n) for n in ("SEED", "LAMBDA", "LR", "FINAL_SEEDS", "ANTISYM_TOL",
                                                               "ANTISYM_N")],
        d1pair=[assign_src(P_DIR / "d1pair_common.py", n) for n in ("SEED", "N_INNER", "ANTISYM_TOL", "ANTISYM_N")])
    return D


# ================================================================ md (Korean, numbers from the files)
def md_text(D):
    sec = lambda h: section(D["spec_text"], h, "docs/specs/GEO2.md")              # noqa: E731
    secb = lambda h: section(D["spec_b_text"], h, "docs/specs/GEO1B.md")          # noqa: E731
    lsec = lambda h: section(D["lad_spec_text"], h, "docs/specs/D1LADDER.md")     # noqa: E731
    a = []
    A = a.append
    ix, d0 = D["index"], D["index"]["d0"] or {}
    A(f"# PREREG GEO2B — 보정 구조(geo, GEO1b) rung, D0 + D1 에너지 학습 사전 등록 ({D['date']})")
    A("")
    A(f"이 문서는 GEO2b의 사전 등록이다(`docs/specs/GEO2.md` 8-3, `docs/specs/GEO1B.md` 4-3, 사용자 결정 D0D1). "
      f"`{D['generator']['file']}`(sha256 `{D['generator']['sha256']}`, git {D['generator']['git']}, {CODE_VERSION})가 "
      f"결과 파일만 읽어 만들었다(git HEAD `{D['git_head']}`, branch `{D['branch']}`, 작성 {D['written']}). 수치는 파일에서 "
      f"옮겼고(표는 유효숫자 6자리, json에는 원래 값), 규칙은 명령서·결정 기록을 그대로 인용했다(인용 블록 = 원문).")
    A("범위: **rung `geo` (LADDER 3단계), D0 + D1 짝, 분할 `T1D0`**, 2 arm × 4 모델 × 5 fold = 40 unit. 학습"
      "(`train_atoms.py run --stage 3`, `train_xgb.py run --stage 3`)은 이 md와 `prereg_geo2b.json`이 commit된 뒤에만 돈다"
      "(8절 gate). **이 문서는 한 번만 쓴다.**")
    A("")
    A("**판정 없음.** 사용자 결정 D0D1로 GEO2.md 10-1절의 주 판정(geo 대 g2)과 비교 모델은 하지 않는다. 결과는 설명용이며, "
      "D1과 D0를 따로 보고하고 합치지 않는다. 절대 기준은 LADDER 목표(5-3절) 하나다.")
    A("")
    s = D["seen"]
    A(f"**지금까지 본 것** (작성 {D['written']}): geo 학습은 한 번도 돌지 않았다(`$LAD/units/L__geo__*` "
      f"{len(s['geo_unit_files'])}개, geo 예측·평가 파일 {len(s['prediction_or_evaluation_files'])}개). geo에서 계산한 것은 "
      f"구조·입력·D0 짝과 8-2 검사(smoke는 fold 0, 2 epoch, 시험 예측 평가 안 함)와 s_y/β 진단뿐이다. GEO1b 구조 결과(7-1절)는 "
      f"이미 보았다.")
    if D["replaced_draft"]:
        A(f"- 이 파일은 commit되지 않은 초안을 대체했다: {D['replaced_draft']}")
    A("")
    A("## 0. 사용자 결정 D0D1과 설계 (그대로)")
    a += quote(D["d0d1_decision"])
    A("")
    A("### 0-1. 설계의 학습·결과 부분 (`analysis/geo1b/results/D0D1_design_investigation.json` result.design, 그대로)")
    a += quote(D["design"]["training"])
    A(">")
    a += quote(D["design"]["results"])
    A("")
    A("### 0-2. 명령서 8-3절 (그대로; 이 문서의 항목 5의 GEO1 숫자는 GEO1B.md 4-3에 따라 GEO1b 숫자로 바꾼다)")
    a += quote(sec("### 8-3."))
    A("")
    A("## 1. rung `geo` 정의")
    A("")
    A("### 1-1. 정의 (GEO2.md 0-2절 그대로)")
    a += quote(sec("### 0-2."))
    A("")
    A("### 1-2. GEO1B.md 0-1·0-2·4-3절 (그대로: G1 미사용, G2 전부로 보정 모델 학습, 바꿔 읽기)")
    a += quote(secb("### 0-1."))
    A(">")
    a += quote(secb("### 0-2."))
    A(">")
    a += quote(secb("### 4-3."))
    A("")
    A("### 1-3. 상식 검사·대체 규칙·Z gate (GEO2.md 4절 그대로) 와 Z 결과 (`analysis/geo2/results/Z_report.json`)")
    a += quote(sec("## 4."))
    A("")
    A("```json")
    a += json.dumps(D["Z_report"], indent=1, ensure_ascii=False, default=str).split("\n")
    A("```")
    A("")
    A("## 2. D0 짝과 fold 규칙")
    A("")
    A(f"- D0 짝: GEO1b `d0_pairs.py`(code {d0.get('report_code_version')}, g2_source `{d0.get('g2_source')}`, preliminary "
      f"{d0.get('preliminary')}) — 규칙은 그 docstring과 `analysis/geo1b/results/DECISIONS_PAIR.md`(7-2절 sha256). D0 짝 "
      f"{f6(d0.get('n_pairs'))}개, 에너지 적격(ok) {f6(d0.get('n_ok'))}개.")
    A("- 라벨·분할 규칙 (`analysis/geo2/d0d1_splits.py` RULES, 그대로):")
    for k, v in (ix.get("rules") or {}).items():
        A(f"  - `{k}`: {v}")
    A("- fold별 D0 짝 (ok, 분할 파일 기준; geo 입력이 없는 짝은 학습 때 두 끝 사용 가능 필터로 빠진다):")
    a += table(["fold", "D0 학습 짝", "D0 시험 짝", "D0 짝 전체(fold)", "시험 짝 edge 종류"],
               [[k, v.get("n_d0_ok_train"), v.get("n_d0_ok_test"), v.get("n_d0_all"), v.get("test_by_edge_kind")]
                for k, v in sorted((d0.get("per_fold") or {}).items())])
    A("")
    A(f"- twins: {compact(ix.get('twins'))}")
    A("- 분할 T1D0: D1 목록은 FU2 T1__f<k>와 똑같고(같은 id, 같은 순서), D0 짝은 그 뒤에 붙는다. inner_val은 D1만이다"
      "(REP·REP8 검증 MAE의 LADDER 13-1 의미 유지). XGB 안쪽 GroupKFold(4)는 D1 부분에서만 돌고, D0 학습 짝은 모든 안쪽 학습 "
      "쪽에 들어간다(arm P 선례).")
    A("")
    A("## 3. 학습 규칙 (LADDER 4-2·4-3 그대로; 바뀐 것은 짝·분할뿐)")
    A("")
    A("### 3-1. 입력 두 가지 (D1LADDER.md 4-2절 그대로)")
    a += quote(lsec("### 4-2."))
    A("")
    A("### 3-2. 모델 네 개 (D1LADDER.md 4-3절 그대로)")
    a += quote(lsec("### 4-3."))
    A("")
    A("### 3-3. grid와 상수 (코드에서 그대로; 파일 sha256은 8-1절 gate 표)")
    A("```python")
    A("# analysis/d1_pair/fu2/models.py (AW / GNN / TF, 변형 A+G)")
    A(D["grid_src"]["atoms"])
    A("# analysis/espley_xtb_repro/train_ml_single.py (XGB)")
    A(D["grid_src"]["xgb"])
    A("# analysis/d1_pair/fu2/fu2_common.py")
    a += D["grid_src"]["fu2"]
    A("# analysis/d1_pair/d1pair_common.py")
    a += D["grid_src"]["d1pair"]
    A("```")
    A("- 모델 구조·grid·손실·β 규칙·seed·epoch·patience는 LADDER와 같다(D0 가중치 없음). gate M-1′는 모든 학습에서 시험 짝 "
      "200개와 D0 학습 짝 200개 모두에 대해 확인한다(GB-5 / XG-B4).")
    A("")
    A("### 3-4. 학습 단위와 실행 (GEO2.md 8-1·8-4절 그대로; 상한은 D0D1 결정으로 160 GPU시간)")
    a += quote(sec("### 8-1."))
    A(">")
    a += quote(sec("### 8-4."))
    A("")
    A(f"- unit {len(units())}개: " + ", ".join(f"`{u}`" for u in units()[:2]) + " … `" + units()[-1] + "`.")
    A("")
    A("## 4. 라벨·분할·입력 sha256")
    A(f"- 라벨 `{(ix.get('labels') or {}).get('path')}` sha256 `{(ix.get('labels') or {}).get('sha256')}` "
      f"(D1 행 = `F_pairs_cr2.csv` 바이트 그대로, D0 행 = `D0_pairs.csv`; L-11′)")
    A(f"- 분할 색인 `{ix['file']}` sha256 `{ix['sha256']}`")
    a += table(["분할 파일", "sha256", "train", "test", "inner_train", "inner_val", "D0 train", "D0 test"],
               [[k, f"`{v['sha256']}`", (v.get("n") or {}).get("train"), (v.get("n") or {}).get("test"),
                 (v.get("n") or {}).get("inner_train"), (v.get("n") or {}).get("inner_val"),
                 (v.get("n_d0") or {}).get("train"), (v.get("n_d0") or {}).get("test")]
                for k, v in sorted(ix["units"].items())])
    A("")
    A(f"- D0 입력: D0_pairs.csv `{d0.get('d0_pairs_sha256')}`, d0_folds.csv `{d0.get('d0_folds_sha256')}`, d0_twins.csv "
      f"`{d0.get('d0_twins_sha256')}`, D0_pairs_report.json `{d0.get('report_sha256')}`, Q_d0_report.json "
      f"`{d0.get('q_report_sha256')}`, labels_d0_q.json `{d0.get('labels_d0_q_sha256')}`")
    A(f"- geo 입력표 `{D['P_build']['table']}` sha256 `{D['P_build']['table_sha256']}` (= `{D['P_build']['file']}` outputs)")
    A("")
    A("## 5. 지표와 선택 규칙")
    A("")
    A("### 5-1. 선택 정의 (GEO2.md 9-1절 그대로; 검증 MAE = D1 inner_val 짝)")
    a += quote(sec("### 9-1."))
    A("")
    A("### 5-2. 대표 고르기 (D1LADDER.md 13-1절 그대로)")
    a += quote(lsec("### 13-1."))
    A("")
    A("### 5-3. 유일한 절대 기준: LADDER 목표 (D1LADDER.md 13-2절 그대로; D1 para 짝, REP 기준)")
    a += quote(lsec("### 13-2."))
    A("")
    A("- GEO2.md 10-1절(geo 대 g2 주 판정)은 하지 않는다(D0D1). D0 결과는 같은 모델·선택 표를 D0 시험 짝(fold 밖 짝)에서 따로 "
      "내고 묶음(component) 단위 bootstrap CI를 쓴다(설계 RESULTS, 0-1절). D1과 D0는 합치지 않는다.")
    A("")
    A("## 6. 알려진 한계")
    a += quote(sec("### 10-3."))
    A("")
    A("- 위 GEO2.md 10-3의 1단계 fold 교차 의존은 D0에도 적용된다(GEO1b final_f<k>가 다른 fold의 D0 라벨 TS를 학습).")
    A("- D1 시험 core와 같은 종류(dipole × 골격)의 D0 반응이 학습에 들어간다(데이터셋 간 전이; 라벨 누수 아님). D1 T1의 "
      "'보지 못한 core' 주장은 약해진다. D-1 겹침 5개는 모든 집합에서 빠지고, 반응물이 같은 D0 twin은 그 D1 twin의 fold를 "
      "따른다.")
    A("- D0 Phase V(DFT 형태 탐색)는 없다(D0 d1·d2·barrier는 Q 규칙만, GEO1b DECISIONS D0D1).")
    A("- D0 라벨 TS는 Gaussian 16, D1은 ORCA RIJCOSX로 최적화했다(A5-eng: pauli·oi RMS 1.6 / 0.8 kcal/mol, n = 24).")
    A("- XGB의 제곱 손실과 SH β(절편 없는 최소제곱)는 더 큰 D0 ΔΔ에 끌릴 수 있다(D0 짝 |dd|가 D1의 2.5–6배). 7-3절의 s_y·β "
      "진단은 설명용이며 학습 규칙을 바꾸지 않는다.")
    A("- 비교 모델이 없어 D0가 D1 정확도를 돕는지 해치는지는 잴 수 없다. 기존 LADDER g2/dft 수치는 다른 구조에서 나온 것이라 "
      "통제된 비교가 아니다.")
    A("")
    A("### 6-1. 설계가 받아들인 위험 (그대로)")
    a += quote(D["design"]["risks_accepted"])
    A("")
    A("## 7. 이미 본 결과 (설명용)")
    A("")
    A("### 7-1. GEO1b 구조 결과 (`analysis/geo1b/results/E_report.json`, `E_rmsd.json`; 판정 없음)")
    A("```json")
    a += json.dumps(compact(D["E_report"]), indent=1, ensure_ascii=False, default=str).split("\n")
    A("```")
    if D["E_rmsd"] is not None:
        A("```json")
        a += json.dumps(compact(D["E_rmsd"]), indent=1, ensure_ascii=False, default=str).split("\n")
        A("```")
    A("")
    A("### 7-2. 입력표 P 보고 (`analysis/geo2/results/P_geo_report.json`, 20개보다 긴 목록은 개수와 앞 5개)")
    A("```json")
    a += json.dumps(D["P_geo_report"], indent=1, ensure_ascii=False, default=str).split("\n")
    A("```")
    A("")
    A("### 7-3. s_y·SH β 진단 (D1 학습 짝만 대 D1 + D0; `analysis/d1_ladder/results/D0D1_diag_geo.json`, fold 중앙값)")
    A("```json")
    a += json.dumps(D["diag"]["summary"], indent=1, ensure_ascii=False, default=str).split("\n")
    A("```")
    A("")
    A("### 7-4. 8-2 검사와 smoke 예상")
    C = D["C_tests"]
    A(f"- `{C['file']}` all_pass {C['all_pass']} (sha256 `{C['sha256']}`, merged {C.get('merged')}); 검사별 {C['checks']}")
    A(f"- 검사가 돈 코드 = 지금 코드({sorted(C_CODE_REQ)}), 검사의 geo 입력표·라벨·분할 색인 = 4절의 파일. evaluate.py는 기록만 "
      f"(검사 때와 같음: {C['evaluate_py']['equal']}).")
    sm = D["smoke"]
    e = sm.get("estimate_geo") or {}
    A(f"- smoke (rung geo, fold 0): 원자 모델 ok {sm.get('atoms_ok')}, XGB ok {sm.get('xgb_ok')} (gate M-1′, D0 학습 짝 포함).")
    A(f"- GPU 예상: {f6(sm.get('geo_gpu_hours'))} GPU시간 ≤ 상한 {GEO_LIMIT_GPU_H} (기준 rung별 {e.get('by_reference')}, "
      f"사용 {e.get('reference_used')}; FU2 상한식 {f6(sm.get('upper_bound_gpu_hours'))}; LADDER rung별 실제 "
      f"{e.get('ladder_rung_atom_gpu_hours')}); 작업자 수별 가장 긴 slice 시간 {e.get('busiest_slice_hours')}; XGB "
      f"{f6(sm.get('xgb_cpu_hours'))} CPU시간.")
    if D["selftest"] is not None:
        A("")
        A("### 7-5. LADDER 기준값 (`analysis/geo2/results/selftest/E_report.json`, eval_geo2 selftest; 참고용)")
        A("```json")
        a += json.dumps({k: D["selftest"].get(k) for k in ("n_pairs", "n_para", "rep8", "goal_mae")}, indent=1,
                        ensure_ascii=False, default=str).split("\n")
        A("```")
    A("")
    A("## 8. gate와 멈춤 규칙")
    A("")
    A("### 8-1. gate (학습 코드가 매 실행 전에 확인; train_atoms GA-4 / GA-5 / GA-7 / GA-14 / GB-2 / GB-6, train_xgb XG-G3)")
    A("- 이 md와 `prereg_geo2b.json`이 git에 있고 HEAD 이후 그대로; sha256(md) = json `md_sha256`; json `rungs` = [\"geo\"], "
      "`split` = \"T1D0\".")
    A("- json `inputs_sha256`에 라벨 파일과 geo 입력표가 있고(need), 아래 표의 파일이 모두 그 sha256 그대로다.")
    A(f"- `analysis/d1_ladder/results`의 LADDER 결과 파일 {len(D['ladder_snapshot'])}개(json `ladder_results_sha256`)가 "
      "그대로다(작성 때 git HEAD와 같음을 확인; `collect --stage 3`도 다시 확인).")
    A("- P: `P_build_geo.json` all_pass·입력표 sha256, `P_geo_report.json` stop false. 분할 색인: d0d1_splits 읽기 규칙 전부"
      "(final). C: `C_tests_geo.json` all_pass(geo 모드)와 그 code_sha256의 학습 코드 5개 = 지금 파일.")
    A("")
    a += table(["파일", "역할", "열", "sha256"], [[f"`{rel(p)}`", r, c, f"`{s}`"] for p, r, c, s in D["inputs"]])
    A("")
    A(f"- 학습 코드 git 상태(작성 시): {D['code_git']}")
    A("- geo unit 지문의 code_sha256 기준값(json `code.fingerprint_keys`): " + "; ".join(
        f"{t}: " + (", ".join(f"{k} `{v[:12]}`" for k, v in m.items()) if isinstance(m, dict) else f"`{str(m)[:12]}`")
        for t, m in D["fingerprint_keys"].items()))
    A("")
    A("### 8-2. GEO2.md 13절 (HPC 규칙, 그대로)")
    a += quote(sec("## 13."))
    A("")
    A("### 8-3. GEO2.md 14절 (결정 기록과 STOP, 그대로; M3 상한은 D0D1로 160 GPU시간)")
    a += quote(sec("## 14."))
    A("")
    A("- **원칙:** 결과를 본 뒤 선택 규칙·짝 집합·분할을 바꾸지 않는다. 바꾸면 이탈로 기록한다(GEO2.md 13절).")
    return "\n".join(a) + "\n"


# ================================================================ json
def json_rec(D, md_sha):
    return dict(
        date=D["date"], written=D["written"], code_version=CODE_VERSION, stage=3, rungs=["geo"], split=DD.SPLIT,
        verdict=None, verdict_note="no comparison verdict (user decision D0D1); descriptive, D1 and D0 separately",
        spec="docs/specs/GEO2.md + docs/specs/GEO1B.md 4-3 + D0D1", spec_sha256=D["spec_sha256"], spec_git=D["spec_git"],
        spec_b_sha256=D["spec_b_sha256"], spec_b_git=D["spec_b_git"], ladder_spec_sha256=D["lad_spec_sha256"],
        ladder_spec_git=D["lad_spec_git"], md=rel(MD), md_sha256=md_sha, generator=D["generator"],
        git_head=D["git_head"], branch=D["branch"], need=need_paths(),
        inputs_sha256={p: s for p, _, _, s in D["inputs"]},
        inputs_roles={p: dict(role=r, column=c) for p, r, c, _ in D["inputs"]},
        ladder_results_sha256=D["ladder_snapshot"],
        gate=dict(files=[rel(MD), rel(JS)],
                  rule="both files tracked + unchanged since HEAD; sha256(md) = md_sha256; rungs == ['geo']; split == "
                       "'T1D0'; need ⊆ inputs_sha256; every listed file unchanged (files only in other rungs' columns "
                       "are skipped, MA-15); every ladder_results_sha256 file unchanged (GA-5)",
                  code="analysis/d1_ladder/train_atoms.py / train_xgb.py gate_prereg_geo2; also gate_p_geo, "
                       "gate_splits_d0d1 and gate_c_geo"),
        units=units(), d0d1_index=D["index"],
        code=dict(fingerprint_keys=D["fingerprint_keys"],
                  rule="every geo unit json fingerprint code_sha256 = fingerprint_keys of its trainer; d0d1_code = "
                       "sha256 of analysis/geo2/d0d1_splits.py (fingerprint key d0d1_code)",
                  c_tests_geo_code_sha256=D["C_tests"]["code_sha256"]),
        selection=dict(rep="LADDER 13-1 per rung x arm x target, lowest 5-fold mean D1 inner-val kcal MAE, ties AW, GNN, "
                           "TF, XGB", rep8="per target over 2 arms x 4 models, ties SH first then the model order",
                       ens4="equal-weight mean of the 4 models' test predictions", goal="LADDER 13-2: oi and cpcm para "
                       "MAE <= 0.13 kcal/mol + bend sensitivity >= 0.7 + false-positive rate <= 0.1 (D1)",
                       reporting="D1 and D0 separately, never pooled; D0 component bootstrap CI"),
        geo_table=D["P_build"], code_git=D["code_git"], evidence=D["evidence"], seen=D["seen"],
        replaced_draft=D["replaced_draft"],
        numbers=dict(geo1b_E_report=D["E_report"], geo1b_E_rmsd=D["E_rmsd"], Z_report=D["Z_report"],
                     P_geo_report=D["P_geo_report"], C_tests_geo=D["C_tests"], smoke=D["smoke"], diag=D["diag"],
                     selftest=D["selftest"]),
        design=D["design"], d0d1_decision=D["d0d1_decision"], grid_source=D["grid_src"])


# ================================================================ commands
def cmd_write():
    D = gather()
    text = md_text(D)
    write_text(MD, text)
    md_sha = DD.sha256(MD)
    if md_sha != hashlib.sha256(text.encode()).hexdigest():
        stop(f"{MD}: the written file differs from the generated text")
    rec = json_rec(D, md_sha)
    write_text(JS, json.dumps(rec, indent=1, ensure_ascii=False, default=str) + "\n")
    print(json.dumps(dict(md=rel(MD), md_sha256=md_sha, json=rel(JS), json_sha256=DD.sha256(JS),
                          n_inputs=len(rec["inputs_sha256"]), n_ladder_results=len(rec["ladder_results_sha256"]),
                          gpu_hours=D["smoke"].get("geo_gpu_hours"),
                          next="commit + push both files (and the code) before train_atoms / train_xgb run --stage 3"),
                     indent=1), flush=True)
    return 0


def cmd_check():
    """= the trainers' gate_prereg_geo2 (every listed file verified) -> results/R_check_geo2b.json."""
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
        if DD.sha256(MD) != rec.get("md_sha256"):
            bad.append("md sha256 != json md_sha256")
        if list(rec.get("rungs") or []) != ["geo"] or rec.get("split") != DD.SPLIT:
            bad.append(f"rungs {rec.get('rungs')} / split {rec.get('split')} != ['geo'] / {DD.SPLIT}")
        ins = rec.get("inputs_sha256") or {}
        lack = sorted(set(need_paths()) - set(ins))
        if lack:
            bad.append(f"inputs_sha256 lacks {lack}")
        for p, s in ins.items():
            if not Path(p).is_file() or DD.sha256(p) != s:
                bad.append(f"input {p} changed or missing")
        snap = rec.get("ladder_results_sha256") or {}
        if not snap:
            bad.append("no ladder_results_sha256 snapshot")
        changed = [r for r, s in sorted(snap.items()) if not (REPO / r).is_file() or DD.sha256(REPO / r) != s]
        if changed:
            bad.append(f"LADDER result files changed: {changed[:8]}")
    out = dict(checked=now(), ok=not bad, bad=bad, md=rel(MD), md_sha256=sha(MD), json=rel(JS), json_sha256=sha(JS),
               n_inputs=len(rec.get("inputs_sha256") or {}),
               n_ladder_results=len(rec.get("ladder_results_sha256") or {}), ladder_changed=changed,
               git_head=git("rev-parse", "HEAD").stdout.strip())
    write_text(CHECK_OUT, json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1), flush=True)
    return 0 if not bad else 3


if __name__ == "__main__":
    if sys.argv[1:] == ["write"]:
        sys.exit(cmd_write())
    if sys.argv[1:] == ["check"]:
        sys.exit(cmd_check())
    raise SystemExit(__doc__)
