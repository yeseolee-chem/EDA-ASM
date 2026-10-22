#!/usr/bin/env python3
"""D1-G2 Phase T (docs/specs/D1G2.md section 4): before any G2 computation, the root-argument paths that Phase F / H
will run on G2 are run on G1 folders and compared with the D1PAIR / D1HF results they must reproduce. Run through
sbatch only, one job per check (they need different environments); each writes results/T_<k>.json and exits 3 on
failure (= STOP, spec 14 T).

  python test_t.py t1 [--inputs d1pair|g2] [--rerun]
      T-1 (d1_pair jobs/g.slurm, PYENV=venv5, D1PAIR_PHASE=F): $G2/t1/g1root/<rxn_id> -> d1_pair/g1_final/<rxn_id> for
      the 5 rxns, d1_features.py runroot <g1root> $G2/t1/out $G2/t1/T1_ids.csv --labels L --meta M --prof P (the
      Phase F form; L / M / P = the --inputs files), then mergeroot -> $G2/t1/X182_T1.parquet; all 182 columns vs the
      same rows of d1_pair/X182_d1.parquet. Pass: max |diff| <= 1e-8 (spec 4), or (user decision 2026-10-22, T-1 =
      the D1PAIR F-0 precedent, d1pair_common.F0_DECISION) every value over 1e-8 is in a scan_curv_* column and max <=
      F0_ACCEPT_MAX = 2.5e-7 (xtb last printed digit x the B5 1/h^2 amplification): verdict pass_f0_precedent, recorded
      as a deviation. Anything else is a fail (STOP, spec 14 T).
  python test_t.py t2 [--inputs d1pair|g2] [--rerun]
      T-2 (same environment): FU2 atoms + tokens through `fu2/atoms.py root ...` (fu2_common.py and tokens.py are not
      edited: fu2/train.py hashes them into the Phase M fingerprints) in children with D1PAIR_PHASE=FROOT, G1_ROOT =
      $G2/t2/g1root (the same 5 links), FU2_SCR = $G2/t2/fu2, FU2_RES = $G2/t2/fu2_res, FU2_X182 =
      $G2/t2/X182_T2.parquet (the 5 rows of X182_d1, exact copy), default versions: atoms.py root run 0 1, atoms.py root
      gate, atoms.py root tokens. Pass: the sha256 of every $G2/t2/fu2/tokens/<rxn_id>.npz = its row of
      analysis/d1_pair/results/fu2/T_manifest.csv (first: the fu2/train.py CODE_SHA files must equal the code_sha256 of
      the cached Phase M units, else fail before any run).
  python test_t.py t3 [--rerun]
      T-3 (d1_hf jobs/hf.slurm, reactot): the D1HF g2 path for T00003, T00019, T00614 in children with D1HF_G2_ROOT =
      d1_pair/g1_final, D1HF_SCRATCH = $G2/t3, D1HF_TEMPLATES = d1_hf/inputs, D1HF_RES = $G2/t3/results:
      prep.py geominputs g2 (the 5 inputs of each must be byte-identical to d1_hf/inputs/<T>/g1/*.inp, else STOP before
      any ORCA run), run.py run g2 hf (the 5 calculations), build.py allgeom / check g2; then parse_rxn(T, "g2") vs
      parse_rxn(T, "g1") of d1_hf/runs/g1/<T> (this process, D1HF environment, read only): d1, d2, elst, pauli, oi,
      disp, cpcm, cds, barrier, bond, eint_spe within 1e-6 kcal/mol, both status ok.
  python test_t.py report [F]
      results/T_1.json + T_2.json + T_3.json + T_1_g2.json + T_2_g2.json -> results/T_report.json: all_pass = T-1, T-2,
      T-3 passed with the current code (the Phase T gate, spec 4; a T_<k>.json written by other code is stale);
      F_ready = all_pass and T-1 / T-2 also passed with --inputs g2. Exit 3 unless all_pass; with F, unless F_ready
      (the gate before Phase F).

--inputs g2: the rev 4-6 label / meta / prof inputs = $G2/G/labels_d1_g2.json, $G2/G/input_meta_d1_g2.csv, $G2/prof (the
Phase F inputs for the 9,266 targets) instead of the D1PAIR ones (default; = the inputs that made X182_d1 and the
tokens); results/T_<k>_g2.json, scratch $G2/t<k>_g2. The 5 label rows are verbatim copies, so both must pass with the
same values: run t1 / t2 --inputs g2 once $G2/G and $G2/prof exist, then `report F`, before Phase F.
A check whose results/T_<k>[_g2].json passed with the same code (sha256 of the code files it runs) and options is not
rerun; otherwise (or with --rerun) that json is first marked 'running', $G2/t<k>[_g2] is moved to
$G2/t<k>[_g2].prev_<time> (never deleted) and the check starts from an empty $G2/t<k>[_g2] (fresh root, out and caches).
Writes only under $G2 (/gpfs/tmp_cpu2/yeseo1ee/d1_g2) and analysis/d1_g2/results (+ the D1HF runner's node-local work
folder, deleted by run.py); d1_pair / d1_hf scratch and results are read only. No __pycache__ is written.

sbatch (from analysis/d1_g2; 4 tasks against MaxSubmit 20: check `squeue -u $USER -h | wc -l` and
`sinfo -p cpu1,cpu2 -o "%.9P %.6t %C"` first). jobs/g.slurm sources $SLURM_SUBMIT_DIR/jobs/env.sh and jobs/hf.slurm
cds to $SLURM_SUBMIT_DIR, so those two are submitted from their own folder (subshell cd); logs go to $G2/logs:
  J1=$(cd ../d1_pair && sbatch --parsable -p cpu1,cpu2 -J d1g2_t1 --time=48:00:00 --cpus-per-task=8 --mem=32G \
       --output=/gpfs/tmp_cpu2/yeseo1ee/d1_g2/logs/%x_%j.out --export=ALL,PYENV=venv5,D1PAIR_PHASE=F \
       jobs/g.slurm ../d1_g2/test_t.py t1)
  J2=$(cd ../d1_pair && sbatch --parsable -p cpu1,cpu2 -J d1g2_t2 --time=48:00:00 --cpus-per-task=8 --mem=32G \
       --output=/gpfs/tmp_cpu2/yeseo1ee/d1_g2/logs/%x_%j.out --export=ALL,PYENV=venv5,D1PAIR_PHASE=F \
       jobs/g.slurm ../d1_g2/test_t.py t2)
  J3=$(cd ../d1_hf && sbatch --parsable -p cpu1,cpu2 -J d1g2_t3 --time=48:00:00 --cpus-per-task=3 --mem=16G \
       --output=/gpfs/tmp_cpu2/yeseo1ee/d1_g2/logs/%x_%j.out jobs/hf.slurm ../d1_g2/test_t.py t3)
  sbatch -p cpu1,cpu2 -J d1g2_trep --time=48:00:00 --cpus-per-task=1 --mem=2G --export=ALL,G2ENV=reactot \
       --dependency=afterany:$J1:$J2:$J3 jobs/g2.slurm test_t.py report
Before Phase F (once $G2/G/labels_d1_g2.json, $G2/G/input_meta_d1_g2.csv and $G2/prof exist): the J1 / J2 commands with
'-J d1g2_t1g / d1g2_t2g' and 't1 --inputs g2' / 't2 --inputs g2', then the report job with 'test_t.py report F'.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True                           # no __pycache__ in analysis/d1_pair, d1_hf, d1_g2

import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g2_common as G  # noqa: E402

DP = G.DP
THIS = Path(__file__).resolve()
P_DIR = G.REPO / "analysis" / "d1_pair"                 # d1_features.py, fu2/ (this worktree; read only)
H_DIR = G.REPO / "analysis" / "d1_hf"                   # prep.py, run.py, build.py (this worktree; read only)
ESP_DIR = G.REPO / "analysis" / "espley_xtb_repro"      # rev 4-6 modules (read only)
VENV5_PY = Path("/gpfs/tmp_cpu2/yeseo1ee/espley_rev5/venv/bin/python")    # = d1_features.VENV5_PY (Phase F engines)
HF_SCR = Path("/gpfs/tmp_cpu2/yeseo1ee/d1_hf")          # D1HF scratch (read only: inputs/<T>/{hf,g1}, runs/g1/<T>)
T_MANIFEST = P_DIR / "results" / "fu2" / "T_manifest.csv"
T_RXN = [1000001, 1000003, 1000018, 1000019, 1000614]   # map 5: first G1 row, Br, S, Si, recovered (g1_rec, exact_hess)
T3_JOBS = ["T00003", "T00019", "T00614"]                 # = rxn 1000003 (Br), 1000019 (Si), 1000614 (g1_rec)
TOL_T1 = 1e-8                                            # spec 4 T-1
# user decision 2026-10-22 (D1G2 T-1 = the D1PAIR F-0 precedent, accepted 2026-10-19): scan_curv_* only, max <= this
PRECEDENT_MAX, PRECEDENT_FAMILY = DP.F0_ACCEPT_MAX, "scan_curv_"
TOL_T3 = 1e-6                                            # spec 4 T-3 (kcal/mol)
T3_KEYS = ["d1", "d2", "elst", "pauli", "oi", "disp", "cpcm", "cds", "barrier", "bond", "eint_spe"]
# the files edited for D1-G2 (no builder could compile them on the login node): compiled first, without writing a .pyc
SYNTAX = {"1": [P_DIR / "d1_features.py"], "2": [P_DIR / "fu2" / "atoms.py"],
          "3": [H_DIR / f for f in ("hf_common.py", "prep.py", "run.py", "build.py")]}
T_NAMES = {"1": "1", "2": "2", "3": "3", "1_g2": "1", "2_g2": "2"}    # results/T_<name>.json -> the check (code files)
T0 = time.time()


class Fail(Exception):
    """a failed step (stage, message): results/T_<k>.json pass false, exit 3."""


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def clean(o):
    """JSON without NaN / Infinity (readable with jq): non-finite floats -> their string; numpy scalars -> python."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else str(float(o))
    return o


def code_files(k):
    """the code check k runs (its combined sha256 decides whether a passed check is rerun, and staleness in report)."""
    f = [THIS, G.HERE / "g2_common.py", P_DIR / "d1pair_common.py"] + sorted(ESP_DIR.glob("*.py"))
    if k in ("1", "2"):
        f += sorted(P_DIR.glob("*.py")) + sorted((P_DIR / "site_si").glob("*.py")) + \
            [P_DIR / "jobs" / n for n in ("env.sh", "g_env.sh", "g.slurm")]
    if k == "2":
        f += sorted((P_DIR / "fu2").glob("*.py"))
    if k == "3":
        f += sorted(H_DIR.glob("*.py")) + sorted(p for p in (H_DIR / "jobs").iterdir() if p.is_file())
    return sorted(set(f))


def code_sha(k):
    h = hashlib.sha256()
    for f in code_files(k):
        h.update(f"{f.relative_to(G.REPO)} {G.sha256(f)}\n".encode())
    return h.hexdigest()


def git_head():
    try:
        r = subprocess.run(["git", "-C", str(G.REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
        return r.stdout.strip() or None
    except OSError:
        return None


def run(argv, cwd, env=None):
    """one child interpreter (this python; its output goes to the job log) -> exit code."""
    argv = [str(a) for a in argv]
    print(f"=== [{now()}] {' '.join(argv)}  (cwd {cwd})", flush=True)
    r = subprocess.run([sys.executable] + argv, cwd=str(cwd), env=env)
    print(f"=== [{now()}] exit {r.returncode}", flush=True)
    return r.returncode


def step(rec, name, argv, cwd, env=None, ok=(0,)):
    rc = run(argv, cwd, env)
    rec.setdefault("exit", {})[name] = rc
    if rc not in ok:
        raise Fail(name, f"exit {rc}")
    return rc


def syntax(k, rec):
    bad = {}
    for f in SYNTAX[k]:
        try:
            compile(f.read_text(), str(f), "exec")
        except SyntaxError as e:
            bad[str(f.relative_to(G.REPO))] = f"line {e.lineno}: {e.msg}"
    rec["syntax_checked"] = [str(f.relative_to(G.REPO)) for f in SYNTAX[k]]
    if bad:
        raise Fail("syntax", bad)


def t_name(k, opts):
    """results/T_<name>.json, $G2/t<name>: k for the default inputs, <k>_g2 for --inputs g2 (both kept)."""
    return k if opts.get("inputs", "d1pair") == "d1pair" else f"{k}_{opts['inputs']}"


def checked(k, env_check, body, opts, rerun):
    """T-<k>: skip when results/T_<name>.json passed with this code and options; else mark it 'running' (a crash never
    leaves an older pass), env_check, a fresh $G2/t<name> (an existing one moved to t<name>.prev_<time>), the syntax
    check, body(base, rec, opts) -> results/T_<name>.json; exit 0 (pass) / 3. name = t_name(k, opts)."""
    name = t_name(k, opts)
    out = G.RES / f"T_{name}.json"
    rec = dict(test=f"T-{k}", spec="docs/specs/D1G2.md 4", options=opts, started=now(), host=socket.gethostname(),
               slurm_job=os.environ.get("SLURM_JOB_ID"), python=sys.executable, git_head=git_head(),
               code_sha256=code_sha(k))
    if out.is_file() and not rerun:
        old = json.loads(out.read_text())
        if old.get("pass") and old.get("code_sha256") == rec["code_sha256"] and old.get("options") == opts:
            print(f"[T-{name}] {out} passed ({old.get('finished')}) with this code and these options: not rerun "
                  f"(--rerun forces it)", flush=True)
            return 0
    G.write_json(out, clean(dict(rec, verdict="running", **{"pass": False})))
    try:
        env_check(rec)
        base = G.G2 / f"t{name}"
        if base.exists() or base.is_symlink():
            prev = G.assert_path(base.with_name(f"{base.name}.prev_{time.strftime('%Y%m%d_%H%M%S')}"))
            os.replace(G.assert_path(base), prev)
            rec["previous_moved_to"] = str(prev)
            print(f"[T-{k}] previous {base} moved to {prev}", flush=True)
        G.assert_path(base).mkdir(parents=True)            # exist_ok False: a concurrent run of this check stops here
        rec["dir"] = str(base)
        syntax(k, rec)
        body(base, rec, opts)
    except Fail as e:
        rec.update(verdict="fail", fail_stage=e.args[0], fail=e.args[1] if isinstance(e.args[1], dict)
                   else str(e.args[1])[:4000], **{"pass": False})
    except Exception as e:                                 # noqa: BLE001 — a crash is a failed check, recorded
        rec.update(verdict="fail", fail_stage="error", fail=f"{type(e).__name__}: {str(e)[:2000]}",
                   traceback=traceback.format_exc()[-4000:], **{"pass": False})
    rec.setdefault("pass", False)
    rec.update(finished=now(), wall_s=round(time.time() - T0, 1), stop=not rec["pass"])
    G.write_json(out, clean(rec))
    print(json.dumps(clean({x: v for x, v in rec.items() if x != "traceback"}), indent=1, default=str), flush=True)
    if rec.get("traceback"):
        print(rec["traceback"], file=sys.stderr, flush=True)
    print(f"[T-{k}] {rec['verdict']} -> {out}", flush=True)
    return 0 if rec["pass"] else 3


# ================================================================ environments
def env_f(rec):
    """t1 / t2: the D1PAIR Phase F job environment of this worktree (d1_pair jobs/g.slurm with PYENV=venv5,
    D1PAIR_PHASE=F); runroot / atoms check g_env(), the engines and the Si entries themselves."""
    bad = []
    if os.environ.get("D1PAIR_PHASE") != "F":
        bad.append(f"D1PAIR_PHASE={os.environ.get('D1PAIR_PHASE')!r} (need F)")
    if Path(sys.executable).resolve() != VENV5_PY.resolve():
        bad.append(f"python {sys.executable} (need {VENV5_PY}: PYENV=venv5)")
    for k, want in (("CODE", P_DIR), ("D1PAIR_ESP_DIR", ESP_DIR)):
        if Path(os.environ.get(k) or "/nonexistent").resolve() != want:
            bad.append(f"{k}={os.environ.get(k)!r} (need {want}: jobs/g.slurm submitted from this worktree's "
                       f"analysis/d1_pair)")
    if os.environ.get("G1_ROOT") != str(DP.G1_ROOT_D1F) or os.environ.get("D1PAIR_SI_PATCH") != "1":
        bad.append(f"G1_ROOT={os.environ.get('G1_ROOT')!r}, D1PAIR_SI_PATCH={os.environ.get('D1PAIR_SI_PATCH')!r} "
                   f"(need jobs/g_env.sh with D1PAIR_PHASE=F)")
    over = [k for k in ("FU2_SCR", "FU2_RES", "FU2_X182", "FU2_ATOM_VERSION", "FU2_TOKEN_VERSION") if os.environ.get(k)]
    if over:
        bad.append(f"{over} set in the job environment (the checks set their own)")
    rec["env"] = {k: os.environ.get(k) for k in ("D1PAIR_PHASE", "PYENV", "CODE", "G1_ROOT", "ESPLEY_LABELS",
                                                 "ESPLEY_META", "ESPLEY_PROF", "ESPLEY_OUT", "R5_SCRATCH", "PYTHONPATH",
                                                 "XTB_BIN", "SLURM_CPUS_PER_TASK")}
    if bad:
        raise Fail("environment", "; ".join(bad))


def env_hf(rec):
    """t3: the D1HF job environment of this worktree (d1_hf jobs/hf.slurm: reactot, ORCA 6.1.1, D1HF_SCRATCH = the
    D1HF scratch) without D1HF_* overrides, so this process's parse_rxn(T, "g1") reads d1_hf/runs/g1 (read only)."""
    bad = []
    if os.environ.get("D1HF_SCRATCH") != str(HF_SCR):
        bad.append(f"D1HF_SCRATCH={os.environ.get('D1HF_SCRATCH')!r} (need {HF_SCR}: jobs/hf_env.sh)")
    over = [k for k in ("D1HF_RES", "D1HF_TEMPLATES", "D1HF_G2_ROOT") if os.environ.get(k)]
    if over:
        bad.append(f"{over} set in the job environment (the check sets them for its children only)")
    if Path.cwd().resolve() != H_DIR or not os.environ.get("D1HF_JOB_T0"):
        bad.append(f"cwd {Path.cwd()} / D1HF_JOB_T0 {os.environ.get('D1HF_JOB_T0')!r} (need jobs/hf.slurm submitted "
                   f"from {H_DIR})")
    orca = os.environ.get("ORCA_BIN", "")
    if not (orca and os.access(orca, os.X_OK)):
        bad.append(f"ORCA_BIN={orca!r} not executable")
    rec["env"] = {k: os.environ.get(k) for k in ("D1HF_SCRATCH", "ORCA_BIN", "CONDA_DEFAULT_ENV", "SLURM_CPUS_PER_TASK",
                                                 "D1HF_JOB_T0")}
    if bad:
        raise Fail("environment", "; ".join(bad))


# ================================================================ shared inputs
def links(root, ids):
    """a fresh root of symlinks root/<rxn_id> -> d1_pair/g1_final/<rxn_id> (each .done) -> {rxn_id: resolved folder}."""
    G.assert_path(root).mkdir()
    for r in ids:
        src = DP.G1_ROOT_D1F / str(r)
        if not (src / ".done").is_file():
            raise Fail("root", f"{src} has no .done")
        os.symlink(src, G.assert_path(root / str(r)))
    return {str(r): os.path.realpath(root / str(r)) for r in ids}


def x182_reference(rec):
    """d1_pair/X182_d1.parquet, checked to be the D1PAIR Phase F file (sha256 = results/F_report.json x182_d1 = the
    fu2 T_report.json x182_sha256, i.e. the global token source)."""
    sha = G.sha256(DP.X182_D1)
    f = json.loads((P_DIR / "results" / "F_report.json").read_text())["x182_d1"]["sha256"]
    t = json.loads((P_DIR / "results" / "fu2" / "T_report.json").read_text())["x182_sha256"]
    rec["reference_x182"] = dict(path=str(DP.X182_D1), sha256=sha, F_report_sha256=f, fu2_T_report_sha256=t)
    if not sha == f == t:
        raise Fail("reference", f"{DP.X182_D1} sha256 {sha} != F_report {f} / fu2 T_report {t}")
    return pd.read_parquet(DP.X182_D1)


def rev_inputs(which):
    """the rev 4-6 label / meta / prof inputs: d1pair (default = jobs/g_env.sh, the inputs of X182_d1 and the tokens)
    or g2 ($G2/G, $G2/prof: the Phase F inputs of the 9,266 targets)."""
    d = dict(labels=DP.D1G_LABELS, meta=DP.D1G_META, prof=DP.PROF_D1) if which == "d1pair" else \
        dict(labels=G.G2 / "G" / "labels_d1_g2.json", meta=G.G2 / "G" / "input_meta_d1_g2.csv", prof=G.G2 / "prof")
    miss = [str(p) for k, p in d.items() if not (p.is_dir() if k == "prof" else p.is_file())]
    if miss:
        raise Fail("inputs", f"--inputs {which}: missing {miss}")
    return d


# ================================================================ T-1
def t1(base, rec, opts):
    """d1_features.py runroot / mergeroot on a fresh root of the 5 G1 links -> all 182 columns vs X182_d1."""
    X0 = x182_reference(rec)
    inp = rev_inputs(opts["inputs"])
    rec["inputs"] = {k: str(v) for k, v in inp.items()}
    root, out, x182, rep = base / "g1root", base / "out", base / "X182_T1.parquet", base / "mergeroot_report.json"
    rec["links"] = links(root, T_RXN)
    ids = G.write_atomic(base / "T1_ids.csv", "rxn_id\n" + "".join(f"{r}\n" for r in T_RXN))
    extra = ["--labels", inp["labels"], "--meta", inp["meta"], "--prof", inp["prof"]]   # the Phase F form (any inputs)
    step(rec, "runroot", [P_DIR / "d1_features.py", "runroot", root, out, ids] + extra, P_DIR)
    step(rec, "mergeroot", [P_DIR / "d1_features.py", "mergeroot", out, x182, rep], P_DIR)
    m = json.loads(rep.read_text())
    rec["mergeroot"] = {k: m.get(k) for k in ("n_requested", "n_done", "n_feat", "F_fail_rate", "reasons", "not_ok",
                                              "x182", "sha256", "F1_all_finite", "env", "versions")}
    X1 = pd.read_parquet(x182)
    cols = [c for c in X0.columns if c != "rxn_id"]
    if list(X1.columns) != list(X0.columns) or len(cols) != 182:
        raise Fail("columns", f"X182_T1 columns ({len(X1.columns)}) != X182_d1 columns ({len(X0.columns)}) or not "
                              f"rxn_id + 182")
    a, b = X1.set_index("rxn_id"), X0.set_index("rxn_id")
    a.index, b.index = a.index.astype(int), b.index.astype(int)
    miss, miss_ref = sorted(set(T_RXN) - set(a.index)), sorted(set(T_RXN) - set(b.index))
    if miss or miss_ref or sorted(a.index) != T_RXN:
        raise Fail("rows", f"no features for {miss} (mergeroot not_ok {m.get('not_ok')}); not in X182_d1 {miss_ref}; "
                           f"X182_T1 rows {sorted(a.index)}")
    A, B = a.loc[T_RXN, cols].to_numpy(float), b.loc[T_RXN, cols].to_numpy(float)
    D = np.abs(A - B)
    D[np.isnan(A) & np.isnan(B)] = 0.0
    D[np.isnan(D)] = np.inf                                # NaN on one side only
    blk = {c: k for k, cs in DP.R5.BLOCKS.items() for c in cs}
    over = sorted((dict(rxn_id=T_RXN[i], col=cols[j], block=blk.get(cols[j], "g1feat"), diff=float(D[i, j]),
                        new=float(A[i, j]), ref=float(B[i, j])) for i, j in zip(*np.nonzero(D > TOL_T1))),
                  key=lambda o: -o["diff"])
    oc, by_block = {}, {}
    for o in over:
        oc[o["col"]] = max(oc.get(o["col"], 0.0), o["diff"])
    for j, c in enumerate(cols):
        k = blk.get(c, "g1feat")
        by_block[k] = max(by_block.get(k, 0.0), float(D[:, j].max()))
    mx = float(D.max())
    i, j = np.unravel_index(int(np.argmax(D)), D.shape)
    strict = mx <= TOL_T1
    within = (not strict) and mx <= PRECEDENT_MAX and all(o["col"].startswith(PRECEDENT_FAMILY) for o in over)
    verdict = "pass" if strict else ("pass_f0_precedent" if within else "fail")
    rec.update(n_rows=len(T_RXN), n_columns=len(cols), n_values=int(D.size), n_exact=int((D == 0).sum()),
               max_abs_diff=mx, worst=dict(rxn_id=T_RXN[i], col=cols[j], diff=mx),
               max_abs_diff_by_block=by_block,
               max_abs_diff_by_rxn={str(r): float(D[n].max()) for n, r in enumerate(T_RXN)},
               tol=TOL_T1, n_over_tol=len(over), over_tolerance_columns=oc, over_tolerance=over,
               f0_precedent=dict(max_abs=PRECEDENT_MAX, family=PRECEDENT_FAMILY + "*", decision=DP.F0_DECISION,
                                 within=bool(within), user_decision="2026-10-22: T-1 = the F-0 precedent (a "
                                 "difference explained by the xtb last digit / scan_curv h^2 amplification passes as a deviation)"),
               verdict=verdict, deviation=bool(within), **{"pass": bool(strict or within)})
    rec["summary"] = dict(verdict=verdict, max_abs_diff=mx, worst=rec["worst"], over_tolerance_columns=oc,
                          within_f0_precedent=bool(within))


# ================================================================ T-2
def npz_diff(a_path, b_path):
    """diagnostic only: per key of two npz files 'equal' | max |diff| | a shape / dtype / missing note; key order."""
    out = {}
    with np.load(a_path) as A, np.load(b_path) as B:
        out["_key_order_equal"] = list(A.files) == list(B.files)
        for k in sorted(set(A.files) | set(B.files)):
            if k not in A.files or k not in B.files:
                out[k] = "missing in " + ("new" if k not in A.files else "reference")
                continue
            x, y = A[k], B[k]
            if x.shape != y.shape or x.dtype != y.dtype:
                out[k] = f"shape / dtype {x.shape} {x.dtype} vs {y.shape} {y.dtype}"
            elif np.array_equal(x, y):
                out[k] = "equal"
            elif np.issubdtype(x.dtype, np.number):
                out[k] = float(np.nanmax(np.abs(x.astype(float) - y.astype(float))))
            else:
                out[k] = "differs"
    return out


def t2(base, rec, opts):
    """FU2 atoms + tokens (`atoms.py root ...` children, D1PAIR_PHASE=FROOT, FU2_* under $G2/t2, default versions) vs
    T_manifest.csv."""
    # fu2/train.py hashes these files into every Phase M unit fingerprint (CODE_SHA): they must equal the cached units'
    u = sorted((DP.SCRATCH / "fu2" / "units").glob("*.json"))
    fp = json.loads(u[0].read_text())["fingerprint"]["code_sha256"] if u else None
    cur = {**{f: G.sha256(P_DIR / "fu2" / f) for f in ("models.py", "train.py", "fu2_common.py", "tokens.py")},
           "d1pair_common.py": G.sha256(P_DIR / "d1pair_common.py")}
    rec["phase_m_code"] = dict(unit=str(u[0]) if u else None, n_units=len(u), equal=fp == cur, current=cur)
    if fp != cur:
        raise Fail("phase_m_code", f"code files differ from the Phase M unit fingerprint {u[0] if u else None}: {cur}")
    X0 = x182_reference(rec)
    inp = rev_inputs(opts["inputs"])
    rec["inputs"] = {k: str(v) for k, v in inp.items()}
    root, fu2, res, x182 = base / "g1root", base / "fu2", base / "fu2_res", base / "X182_T2.parquet"
    rec["links"] = links(root, T_RXN)
    x = X0[X0.rxn_id.astype(int).isin(T_RXN)].sort_values("rxn_id").reset_index(drop=True)
    G.write_atomic(x182, lambda t: x.to_parquet(t, index=False))
    x2 = pd.read_parquet(x182)
    exact = bool(x2.equals(x) and list(x2.columns) == list(X0.columns) and (x2.dtypes == X0.dtypes).all()
                 and sorted(x2.rxn_id.astype(int)) == T_RXN)
    rec["x182_copy"] = dict(path=str(x182), sha256=G.sha256(x182), n=len(x2), exact_copy=exact)
    if not exact:
        raise Fail("x182_copy", f"{x182} is not an exact copy of the 5 X182_d1 rows")
    env = {k: v for k, v in os.environ.items() if k not in ("FU2_ATOM_VERSION", "FU2_TOKEN_VERSION")}   # defaults
    env.update(D1PAIR_PHASE="FROOT", G1_ROOT=str(root), ESPLEY_OUT=str(base / "xtb"), R5_SCRATCH=str(base / "r5"),
               ESPLEY_LABELS=str(inp["labels"]), ESPLEY_META=str(inp["meta"]), ESPLEY_PROF=str(inp["prof"]),
               FU2_SCR=str(fu2), FU2_RES=str(res), FU2_X182=str(x182), PYTHONDONTWRITEBYTECODE="1")
    rec["child_env"] = {k: env[k] for k in ("D1PAIR_PHASE", "G1_ROOT", "ESPLEY_OUT", "R5_SCRATCH", "ESPLEY_LABELS",
                                            "ESPLEY_META", "ESPLEY_PROF", "FU2_SCR", "FU2_RES", "FU2_X182")}
    step(rec, "atoms_run", [P_DIR / "fu2" / "atoms.py", "root", "run", "0", "1"], P_DIR, env)
    step(rec, "atoms_gate", [P_DIR / "fu2" / "atoms.py", "root", "gate"], P_DIR, env)
    step(rec, "tokens_build", [P_DIR / "fu2" / "atoms.py", "root", "tokens"], P_DIR, env)
    q, t = (json.loads((res / f).read_text()) for f in ("Q_report.json", "T_report.json"))
    rec["Q_report"] = {k: q.get(k) for k in ("n_target", "n_ok", "n_use", "excluded", "ext_used", "ext_pass_rate",
                                             "qa_maxdiff_max", "qb_maxdiff_max", "atom_version", "stop", "stop_reason",
                                             "overrides")}
    rec["T_report"] = {k: t.get(k) for k in ("token_version", "n_reactions", "num_cols", "ext_used", "gate_Ta",
                                             "x182_sha256", "splits_sha256", "q_report_sha256", "stop")}
    M = pd.read_csv(T_MANIFEST)
    ref = dict(zip(M.rxn_id.astype(int), M.sha256))
    rows = []
    for r in T_RXN:
        p = fu2 / "tokens" / f"{r}.npz"
        s = G.sha256(p) if p.is_file() else None
        row = dict(rxn_id=r, sha256=s, ref_sha256=ref.get(r), equal=bool(s is not None and s == ref.get(r)))
        d1p = DP.SCRATCH / "fu2" / "tokens" / f"{r}.npz"
        if s and not row["equal"] and d1p.is_file():
            row["diff_vs_d1pair_npz"] = npz_diff(p, d1p)
        rows.append(row)
    n_eq = sum(r["equal"] for r in rows)
    verdict = "pass" if n_eq == len(T_RXN) else "fail"
    rec.update(tokens=rows, n_equal=n_eq, n=len(T_RXN),
               reference=dict(path=str(T_MANIFEST), sha256=G.sha256(T_MANIFEST)), verdict=verdict,
               **{"pass": n_eq == len(T_RXN)})
    rec["summary"] = dict(verdict=verdict, n_equal=n_eq, n=len(T_RXN), ext_used=q.get("ext_used"),
                          atom_version=q.get("atom_version"), token_version=t.get("token_version"))


# ================================================================ T-3
def t3(base, rec, opts):
    """the D1HF g2 path on g1_final: inputs byte-identical to the D1HF G1 inputs, 5 calculations, parse_rxn(T, "g2")
    vs parse_rxn(T, "g1") of d1_hf/runs/g1/<T> (<= 1e-6 kcal/mol)."""
    sys.path.insert(0, str(H_DIR))
    import build as HB                                     # this process: the D1HF paths (read only)
    H = HB.H
    here = dict(HF=(H.HF, HF_SCR), RES=(H.RES, H_DIR / "results"), TEMPLATES=(H.TEMPLATES, HF_SCR / "inputs"),
                GEOM_ROOT_g1=(H.GEOM_ROOT["g1"], DP.G1_ROOT_D1F))
    bad = {k: str(v) for k, (v, w) in here.items() if Path(v) != Path(w)}
    if bad:
        raise Fail("environment", f"hf_common of this process is not on the D1HF paths: {bad}")
    lst = G.write_atomic(base / "t3_list.csv", "job_id\n" + "".join(f"{T}\n" for T in T3_JOBS))
    env = dict(os.environ, D1HF_SCRATCH=str(base), D1HF_RES=str(base / "results"),
               D1HF_TEMPLATES=str(HF_SCR / "inputs"), D1HF_G2_ROOT=str(DP.G1_ROOT_D1F), PYTHONDONTWRITEBYTECODE="1")
    rec["child_env"] = {k: env[k] for k in ("D1HF_SCRATCH", "D1HF_RES", "D1HF_TEMPLATES", "D1HF_G2_ROOT")}
    # 1. the g2 inputs (prep.py geominputs) must be byte-identical to the D1HF G1 inputs before any ORCA run
    step(rec, "geominputs", [H_DIR / "prep.py", "geominputs", "g2", lst], H_DIR, env)
    st = pd.read_csv(base / "results" / "g2_inputs.csv", dtype=str, keep_default_na=False)
    rec["g2_inputs"] = dict(zip(st.job_id, st.status))
    if sorted(st.job_id) != T3_JOBS or (st.status != "ok").any():
        raise Fail("geominputs", f"results/g2_inputs.csv statuses {rec['g2_inputs']}")
    ident = {}
    for T in T3_JOBS:
        a, b = base / "inputs" / T / "g2", HF_SCR / "inputs" / T / "g1"
        fa, fb = sorted(p.name for p in a.iterdir()), sorted(p.name for p in b.iterdir())
        diff = [n for n in fa if n in fb and (a / n).read_bytes() != (b / n).read_bytes()]
        ident[T] = dict(identical=fa == fb == sorted(f"{c}.inp" for c in H.CALCS) and not diff, differ=diff,
                        only_g2=sorted(set(fa) - set(fb)), only_g1=sorted(set(fb) - set(fa)),
                        sha256={n: G.sha256(a / n) for n in fa})
    rec["inputs_identical"] = ident
    if not all(v["identical"] for v in ident.values()):
        raise Fail("inputs", "the g2 inputs are not byte-identical to d1_hf/inputs/<T>/g1/*.inp (no ORCA run)")
    # 2. the 5 calculations, label assembly and checkpoint of the g2 path
    nw = max(1, min(len(T3_JOBS), int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))))
    step(rec, "run", [H_DIR / "run.py", "run", lst, "g2", "hf", nw], H_DIR, env)
    step(rec, "build_all", [H_DIR / "build.py", "allgeom", lst, "g2"], H_DIR, env)
    step(rec, "build_check", [H_DIR / "build.py", "check", lst, "g2"], H_DIR, env, ok=(0, 3))
    pj = base / "parse_g2.json"
    step(rec, "parse_g2", [THIS, "_t3parse", lst, pj], H_DIR, env)
    # 3. parse_rxn(T, "g2") (child) vs parse_rxn(T, "g1") on d1_hf/runs/g1/<T> (this process)
    g2 = json.loads(pj.read_text())
    if g2["env"] != rec["child_env"]:
        raise Fail("parse_g2", f"_t3parse ran with {g2['env']}")
    lab = {r["job_id"]: r for r in json.loads((base / "labels_d1_hf_g2.json").read_text())}
    hst = pd.read_csv(base / "results" / "H_status.csv", dtype=str, keep_default_na=False)
    rec["H_status"] = dict(zip(hst.job_id, hst.status))
    rows = {}
    for T in T3_JOBS:
        a, b = g2["records"][T], HB.parse_rxn(T, "g1")
        d = {k: abs(float(a[k]) - float(b[k])) if (a.get(k) is not None and b.get(k) is not None) else float("inf")
             for k in T3_KEYS}
        mx = float(np.max(np.array(list(d.values()), dtype=float)))          # NaN propagates (= not <= tol)
        ok = a.get("status") == "ok" and b.get("status") == "ok" and mx <= TOL_T3
        info = {k: abs(float(a[k]) - float(b[k])) for k in ("bsse_shift_kcal", "bsse_total", "sum6")
                if k in a and k in b}                     # reported only
        rows[T] = dict(ok=bool(ok), max_abs_diff_kcal=mx, diff=d, status_g2=a.get("status"), reason_g2=a.get("reason"),
                       status_g1=b.get("status"), reason_g1=b.get("reason"), g2={k: a.get(k) for k in T3_KEYS},
                       g1={k: b.get(k) for k in T3_KEYS}, info_diff=info, gates_g2=a.get("gates"),
                       gates_g1=b.get("gates"), labels_json_row_equal=bool(T in lab and all(lab[T].get(k) == a.get(k)
                                                                                         for k in T3_KEYS)))
    mx = float(np.max(np.array([r["max_abs_diff_kcal"] for r in rows.values()], dtype=float)))
    good = all(r["ok"] and r["labels_json_row_equal"] for r in rows.values()) and rec["exit"]["build_check"] == 0
    verdict = "pass" if good else "fail"
    rec.update(compare=rows, keys=T3_KEYS, tol_kcal=TOL_T3, max_abs_diff_kcal=mx, reference=str(HF_SCR / "runs" / "g1"),
               verdict=verdict, **{"pass": good})
    rec["summary"] = dict(verdict=verdict, max_abs_diff_kcal=mx, inputs_identical=True,
                          status={T: (r["status_g2"], r["status_g1"]) for T, r in rows.items()},
                          build_check_exit=rec["exit"]["build_check"])


def _t3parse(lst, out):
    """child of t3 (D1HF_* overrides in its environment): parse_rxn(T, "g2") of the listed reactions -> <out> (json)."""
    sys.path.insert(0, str(H_DIR))
    import build as HB
    H = HB.H
    got = dict(D1HF_SCRATCH=H.HF, D1HF_RES=H.RES, D1HF_TEMPLATES=H.TEMPLATES, D1HF_G2_ROOT=H.GEOM_ROOT["g2"])
    bad = {k: (str(v), os.environ.get(k)) for k, v in got.items()
           if not os.environ.get(k) or Path(v) != Path(os.environ[k])}
    if bad or Path(H.HF) == HF_SCR:
        G.stop(f"_t3parse: hf_common is not on the D1HF_* overrides: {bad}")
    ids = pd.read_csv(lst, dtype=str).job_id.tolist()
    G.write_json(out, dict(env={k: os.environ[k] for k in got}, records={T: HB.parse_rxn(T, "g2") for T in ids}))
    return 0


# ================================================================ report
def report(gate_f=False):
    """results/T_<name>.json (T_NAMES) -> results/T_report.json: all_pass = T-1, T-2, T-3 (default inputs) passed with
    the current code (Phase T, spec 4); F_ready = all_pass and T-1 / T-2 also passed with --inputs g2 (the $G2/G,
    $G2/prof inputs Phase F reads). Exit 3 unless all_pass; with gate_f (report F), unless F_ready."""
    tests, good = {}, {}
    keep = ("pass", "verdict", "fail_stage", "fail", "summary", "options", "started", "finished", "slurm_job", "host",
            "git_head", "code_sha256", "dir")
    for name, k in T_NAMES.items():
        f = G.RES / f"T_{name}.json"
        key = f"T-{name.replace('_', ' ')}"
        if not f.is_file():
            tests[key] = dict(missing=str(f), **{"pass": False})
            good[name] = False
            continue
        r = json.loads(f.read_text())
        stale = r.get("code_sha256") != code_sha(k)
        opts_ok = r.get("options") == ({} if k == "3" else {"inputs": "d1pair" if name == k else "g2"})
        tests[key] = dict({x: r.get(x) for x in keep}, stale_code=stale, options_ok=opts_ok, file=str(f),
                          file_sha256=G.sha256(f))
        good[name] = bool(r.get("pass")) and not stale and opts_ok
    ok = all(good[n] for n in ("1", "2", "3"))
    f_ready = ok and good["1_g2"] and good["2_g2"]
    rep = dict(spec="docs/specs/D1G2.md 4 (T-1, T-2, T-3; any failure = STOP, section 14 T)", tests=tests, all_pass=ok,
               F_ready=f_ready, F_ready_rule="all_pass and T-1 / T-2 passed with --inputs g2 ($G2/G, $G2/prof) with the "
               "current code (required before Phase F)", gate=("F" if gate_f else "T"),
               stop=not (f_ready if gate_f else ok), written=now())
    G.write_json(G.RES / "T_report.json", clean(rep))
    print(json.dumps(clean(rep), indent=1, default=str), flush=True)
    return 3 if rep["stop"] else 0


if __name__ == "__main__":
    a = sys.argv[1:]
    cmd, opt = (a[0] if a else ""), a[1:]
    if cmd == "_t3parse" and len(opt) == 2:
        sys.exit(_t3parse(*opt))
    if cmd == "report" and opt in ([], ["F"]):
        sys.exit(report(gate_f=opt == ["F"]))
    rerun = "--rerun" in opt
    opt = [x for x in opt if x != "--rerun"]
    if cmd in ("t1", "t2") and opt in ([], ["--inputs", "d1pair"], ["--inputs", "g2"]):
        sys.exit(checked(cmd[1], env_f, t1 if cmd == "t1" else t2, dict(inputs=opt[1] if opt else "d1pair"), rerun))
    if cmd == "t3" and not opt:
        sys.exit(checked("3", env_hf, t3, {}, rerun))
    raise SystemExit(__doc__)
