#!/usr/bin/env python3
"""tokens_lf.py — DEPLOY-MODEL Phase T (docs/specs/DEPLOY_MODEL.md 6): per-atom xTB values (fu2/atoms.py) and atom
tokens (fu2/tokens.py) of the X182_lf reactions, with the 5-2 environment (partition mode redetect).

  python tokens_lf.py atoms <i|@I> <n>   # fu2/atoms.py root run i n -> $DEP/fu2_lf/atoms/<rxn_id>.npz (.fail / .err.json)
  python tokens_lf.py gate               # (one rerun pass when <rid>.err.json remain) + fu2/atoms.py root gate
                                         #   -> results/fu2_lf/Q_manifest.csv, Q_report.json (Q-a, Q-b, Q-c as FU2)
  python tokens_lf.py tokens             # fu2/atoms.py root tokens (= fu2/tokens.py build) -> $DEP/fu2_lf/tokens/,
                                         #   results/fu2_lf/T_report.json, T_manifest.csv (T-a); then the spec 6
                                         #   element / column checks against the g2 T_report -> results/T_lf_check.json

Job: analysis/d1_pair/jobs/g.slurm from analysis/d1_pair, --export=ALL,PYENV=venv5,D1PAIR_PHASE=F (as feat_lf.py).
Every command needs results/F_report.json without STOP for the current $DEP/X182_lf.parquet (feat_lf.need_f_report).
The parent checks the job environment (feat_lf.parent_env) and runs the FU2 code in a child interpreter with
feat_lf.lf_env (spec 5-2) + D1PAIR_PHASE=FROOT, FU2_SCR=$DEP/fu2_lf, FU2_RES=results/fu2_lf, FU2_X182=$DEP/X182_lf.parquet,
FU2_ATOM_VERSION=fu2-atoms-3-lf, FU2_TOKEN_VERSION=fu2-tokens-3-lf (atoms.root_bind rebinds the fu2_common roots and
versions from these, in the child and in its spawn workers). The only change to the FU2 code path (spec 6): the root
environment check atoms._env_root is replaced in the child by _env_root_lf = the same checks with partition mode
'redetect' (not 'label') and ESPLEY_PARTITION allowed to differ from g_env() (+ ESPLEY_PROF must be an empty folder).
atoms.cmd_run / cmd_gate / cmd_root_tokens and tokens.cmd_build run unchanged (Q-a QA_TOL 1e-6 / QA_MAX_FAIL, Q-b,
Q-c, T-a as FU2). Extra STOP rules of spec 6 / 16 T, checked here: Q-a failures > 0.5 % of the reactions; T_report
elements != [1, 6, 7, 8, 9, 14, 16, 17, 35]; bin_cols / num_cols / ext_used != the g2 T_report
(analysis/d1_g2/results/fu2/T_report.json); d_in != 29; glob_cols != the X182_d1_g2 columns.
Exit codes: 0 done, 2 not ready (atoms incomplete: resubmit 'atoms'), 3 STOP. Idempotent: atoms.py caches every
reaction (npz / fail.json by atom version; err.json retried), tokens are rebuilt and re-verified.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
THIS = Path(__file__).resolve()
sys.path.insert(0, str(HERE))
import feat_lf as FL  # noqa: E402      (puts analysis/d1_pair and analysis/d1_pair/fu2 on sys.path)

D, A, DG = FL.D, FL.A, FL.DG
C = FL.C
FU2_RES_LF = D.RES / "fu2_lf"
G2_TREP = D.REPO / "analysis" / "d1_g2" / "results" / "fu2" / "T_report.json"
T_CHECK = D.RES / "T_lf_check.json"
QA_RATE_MAX = 0.005                                   # spec 6 / 16 T: Q-a failures <= 0.5 %
D_IN = 29                                             # spec 6: 9 element one-hot + 6 bin + 14 num


def fu2_env(fu2_scr=None, fu2_res=None, x182=None, root=None, labels=None, meta=None, prof=None, fdir=None):
    """spec 6 child environment: feat_lf.lf_env (5-2) + the FU2 root variables of atoms.py root."""
    env = FL.lf_env(root or D.LF_FINAL, labels or D.LABELS_LF, meta or D.META_LF, prof or D.PROF_EMPTY,
                    fdir or FL.F_DIR)
    env.update(FU2_SCR=os.path.abspath(fu2_scr or D.FU2_LF), FU2_RES=os.path.abspath(fu2_res or FU2_RES_LF),
               FU2_X182=os.path.abspath(x182 or D.X182_LF), FU2_ATOM_VERSION=D.ATOM_VERSION,
               FU2_TOKEN_VERSION=D.TOKEN_VERSION)
    for k in ("FU2_SCR", "FU2_RES"):
        D.assert_write_ok(env[k])
        C.assert_write_ok(env[k])
    return env


def run_child(cmd, args, env):
    argv = [sys.executable, str(THIS), "_" + cmd] + [str(x) for x in args]
    print(f"=== [{FL.now()}] {' '.join(argv[1:])}", flush=True)
    r = subprocess.run(argv, env=env)
    print(f"=== [{FL.now()}] exit {r.returncode}", flush=True)
    return r.returncode


# ================================================================ child (FROOT, lf environment)
def _env_root_lf(DG_):
    """= fu2/atoms.py _env_root (atoms.py:132-170) with the spec 6 change only: the partition mode must be 'redetect'
    and ESPLEY_PARTITION may differ from g_env() (feat_lf.ENV_EXCEPT); plus ESPLEY_PROF must be an empty folder."""
    bad = {k: (os.environ.get(k), v) for k, v in DG_.g_env().items() if k not in FL.ENV_EXCEPT and os.environ.get(k) != v}
    bad.update({k: (None, "set") for k in A.ROOT_ENV if not os.environ.get(k)})
    if os.environ.get("ESPLEY_PARTITION") != FL.PARTITION:
        bad["ESPLEY_PARTITION"] = (os.environ.get("ESPLEY_PARTITION"), FL.PARTITION)
    if bad:
        print(f"STOP: D1PAIR_PHASE=FROOT environment differs from g_env() outside {FL.ENV_EXCEPT}: {bad}", file=sys.stderr)
        sys.exit(3)
    import fragmenter as fr
    import r6_partition as RP
    import rev5_common as R5
    import rev6_common as R6
    import xtb_slice as xs
    F = A.F
    si = (os.environ.get("D1PAIR_SI_PATCH"), fr.COV.get("Si"), fr.Z.get("Si"), R5.BONDI.get("Si"), R5.GFN2_VALENCE.get("Si"))
    if si != FL.SI_WANT:
        print(f"STOP: Si element entries not active (site_si/sitecustomize.py): {si}", file=sys.stderr); sys.exit(3)
    E = {k: Path(os.environ[k]) for k in A.ROOT_ENV}
    same = dict(xtb_slice_g1=(xs.GEOM_ROOT["g1"], "G1_ROOT"), R5_G1_ROOT=(R5.G1_ROOT, "G1_ROOT"),
                xtb_slice_LABELS=(xs.LABELS_PATH, "ESPLEY_LABELS"), R5_LABELS=(R5.LABELS, "ESPLEY_LABELS"),
                xtb_slice_META=(xs.META, "ESPLEY_META"), R5_META=(R5.META, "ESPLEY_META"),
                R5_PROF=(R5.PROF, "ESPLEY_PROF"), r6_partition_PROF=(RP.PROF, "ESPLEY_PROF"),
                R5_SCRATCH=(R5.SCRATCH, "R5_SCRATCH"), R5_XTB_ROOT=(R5.XTB_ROOT, "ESPLEY_OUT"))
    bad = {k: (str(p), str(E[v])) for k, (p, v) in same.items() if Path(p) != E[v]}
    bad.update({k: str(p) for k, p, ok in (("G1_ROOT", E["G1_ROOT"], E["G1_ROOT"].is_dir()),
                                           ("ESPLEY_LABELS", E["ESPLEY_LABELS"], E["ESPLEY_LABELS"].is_file()),
                                           ("ESPLEY_META", E["ESPLEY_META"], E["ESPLEY_META"].is_file()),
                                           ("ESPLEY_PROF", E["ESPLEY_PROF"], FL.prof_is_empty(E["ESPLEY_PROF"])),
                                           ("FU2_X182", F.X182, F.X182.is_file())) if not ok})
    if bad or xs.PARTITION_MODE != FL.PARTITION:
        print(f"STOP: modules not on the FROOT roots / inputs missing: {bad}, partition {xs.PARTITION_MODE}", file=sys.stderr)
        sys.exit(3)
    paths = dict(xtb_slice_g1=xs.GEOM_ROOT["g1"], xtb_slice_g2=xs.GEOM_ROOT["g2"], R5_SCRATCH=R5.SCRATCH,
                 R5_XTB_ROOT=R5.XTB_ROOT, R5_G1_ROOT=R5.G1_ROOT, R6_SCRATCH=R6.SCRATCH, R6_G1_ROOT_R6=R6.G1_ROOT_R6,
                 R6_XTB_R6=R6.XTB_R6)
    for k, p in paths.items():
        C.assert_write_ok(p)
    return {k: str(v) for k, v in paths.items()}


def _child(cmd, args):
    A.root_bind()                          # exit 3 unless FROOT + FU2_* set away from D1PAIR + version tags (-lf)
    A._env_root = _env_root_lf             # spec 6: the root check with partition mode 'redetect'
    A._env()
    for p in (A.F.FU2_SCR, A.F.FU2_RES):
        D.assert_write_ok(p)
    if A.ATOM_VERSION != D.ATOM_VERSION:
        D.stop(f"atoms.ATOM_VERSION {A.ATOM_VERSION} != {D.ATOM_VERSION}")
    if cmd == "atoms":
        return A.cmd_run(*args)
    if cmd == "gate":
        return A.cmd_gate()
    if cmd == "tokens":
        return A.cmd_root_tokens()
    raise SystemExit(f"unknown child command {cmd}")


# ================================================================ parent commands
def _update_check(key, obj):
    cur = D.read_json(T_CHECK) if T_CHECK.is_file() else {}
    cur[key] = obj
    D.write_json(T_CHECK, cur)


def q_check(qf):
    """spec 6 / 16 T on a Q_report: FU2 stop (Q-a > QA_MAX_FAIL, Q-b, Q-c) or Q-a failures > 0.5 % of the reactions."""
    q = D.read_json(qf)
    n = int(q.get("n_target") or 0)
    rate = (int(q.get("n_qa_fail") or 0) / n) if n else None
    stop = []
    if q.get("stop"):
        stop.append(f"Q_report stop: {q.get('stop_reason')}")
    if rate is None or rate > QA_RATE_MAX:
        stop.append(f"Q-a failures {q.get('n_qa_fail')} / {n} > {QA_RATE_MAX}")
    if q.get("atom_version") != D.ATOM_VERSION:
        stop.append(f"Q_report atom_version {q.get('atom_version')} != {D.ATOM_VERSION}")
    return dict(n_target=n, n_ok=q.get("n_ok"), n_qa_fail=q.get("n_qa_fail"), qa_fail_rate=rate, qa_rate_max=QA_RATE_MAX,
                n_qb_fail=q.get("n_qb_fail"), n_compute_fail=q.get("n_compute_fail"), n_use=q.get("n_use"),
                ext_used=q.get("ext_used"), ext_pass_rate=q.get("ext_pass_rate"), fail_reasons=q.get("fail_reasons"),
                q_report_sha256=D.sha256(qf), stop=bool(stop), stop_reasons=stop)


def t_check(tf, x182, g2_trep=G2_TREP):
    """spec 6: T_report of this run vs dep_common.ELEMENTS, the g2 T_report columns, d_in = 29, the X182 columns."""
    import tokens as T
    t, g = D.read_json(tf), D.read_json(g2_trep)
    el = [int(z) for z in t.get("elements") or []]
    d_in = len(el) + len(t.get("bin_cols") or []) + len(t.get("num_cols") or [])
    g_din = len(g["elements"]) + len(g["bin_cols"]) + len(g["num_cols"])
    xcols = FL.x182_columns_ref()[1:]
    chk = dict(elements=el == list(D.ELEMENTS) and el == [int(z) for z in g["elements"]],
               bin_cols=t.get("bin_cols") == g["bin_cols"] == T.BIN_COLS,
               num_cols=t.get("num_cols") == g["num_cols"], ext_used=t.get("ext_used") == g["ext_used"] is True,
               d_in=d_in == D_IN == g_din, glob_cols=t.get("glob_cols") == xcols,
               token_version=t.get("token_version") == D.TOKEN_VERSION,
               x182_sha256=t.get("x182_sha256") == D.sha256(x182),
               gate_Ta=bool(t.get("gate_Ta")) and all((t.get("gate_Ta") or {}).values()) and not t.get("stop"))
    return dict(checks=chk, elements=el, d_in=d_in, n_reactions=t.get("n_reactions"), n_bad=len(t.get("bad") or []),
                g2_T_report=str(g2_trep), g2_T_report_sha256=D.sha256(g2_trep), t_report_sha256=D.sha256(tf),
                stop=not all(chk.values()), stop_reasons=[k for k, v in chk.items() if not v])


def cmd_atoms(i, n):
    FL.parent_env()
    FL.need_f_report()
    i, n = D.array_task(i), int(n)
    if not 0 <= i < n:
        D.stop(f"slice {i} of {n}?")
    return run_child("atoms", [i, n], fu2_env())


def cmd_gate():
    FL.parent_env()
    FL.need_f_report()
    env = fu2_env()
    err = sorted((Path(env["FU2_SCR"]) / "atoms").glob("*.err.json"))
    rec = dict(written=FL.now(), n_err_before=len(err), err_before=[p.name for p in err][:200])
    if err:                                              # one rerun pass of the retryable (infra) errors, as D1-G2 f_gate
        rec["retry_rc"] = run_child("atoms", [0, 1], env)
        if rec["retry_rc"] != 0:
            _update_check("gate", rec)
            return rec["retry_rc"]
    rc = run_child("gate", [], env)
    rec["gate_rc"] = rc
    qf = Path(env["FU2_RES"]) / "Q_report.json"
    if rc == 2:
        rec["note"] = "atoms incomplete: run 'atoms' for every slice"
        _update_check("gate", rec)
        return 2
    if not qf.is_file():
        _update_check("gate", rec)
        D.stop(f"no {qf} after the gate (exit {rc})")
    rec.update(q_check(qf))
    _update_check("gate", rec)
    print(json.dumps(rec, indent=1, default=str))
    return 3 if (rec["stop"] or rc == 3) else rc


def cmd_tokens():
    FL.parent_env()
    FL.need_f_report()
    env = fu2_env()
    qf = Path(env["FU2_RES"]) / "Q_report.json"
    if not qf.is_file():
        D.not_ready(f"no {qf} (tokens_lf.py gate)")
    q = q_check(qf)
    if q["stop"]:
        D.stop(f"Q gate STOP: {q['stop_reasons']}")
    rc = run_child("tokens", [], env)
    tf = Path(env["FU2_RES"]) / "T_report.json"
    if not tf.is_file():
        D.stop(f"no {tf} after tokens (exit {rc})")
    rec = dict(written=FL.now(), tokens_rc=rc, **t_check(tf, D.X182_LF))
    rec["stop"] = bool(rec["stop"] or rc != 0)
    _update_check("tokens", rec)
    print(json.dumps(rec, indent=1, default=str))
    return 3 if rec["stop"] else 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["atoms"] and len(a) == 3:
        sys.exit(cmd_atoms(a[1], a[2]))
    if a == ["gate"]:
        sys.exit(cmd_gate())
    if a == ["tokens"]:
        sys.exit(cmd_tokens())
    if a[:1] in (["_atoms"], ["_gate"], ["_tokens"]):
        sys.exit(_child(a[0][1:], a[1:]))
    print(__doc__)
    sys.exit(2)
