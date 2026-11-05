#!/usr/bin/env python3
"""tests/test_pipeline.py — simulator for pipeline.py (the MLTS automatic stage launcher). Compute nodes only:
    bash jobs/sbx.sh reactot tests/test_pipeline.py [scenario ...]          (from analysis/mlts)
Fake SLURM (pipeline.FakeSlurm: squeue / sacct / sbatch in memory), fake git (pipeline.FakeGit), a fake clock and
fake results / scratch roots under $MLT/tmp/pipetest/<scenario>/ (PIPETEST_ROOT overrides). Nothing outside that root
is written; the real results/ and $MLT are only read (code tokens, the G-0 estimate copy, the real targets in s13).
Scenarios (review defect in brackets):
  s1 real 2026-11-04 squeue snapshot: queue full -> nothing; G ends -> lists [S2: queue <= 19 incl. successor]
  s2 adopting the manual collect 1061711, G-D1 done + commit;  s2b collect incomplete -> round 2 (G1 only)
  s3 Z-rep decision (2 / 1 / 0 candidates);  s4 48 h TIMEOUT retry -> no-progress HALT -> RESOLVE retry (+ STOP removal
     committed);  s5 crash after / before sbatch;  s6 MaxSubmit + queue cap;  s7 code missing -> edited -> committed
  s8 D0 budget (recommended refit, HALT, RESOLVE param);  s9 FT-2 gate HALT + RESOLVE accept (+ FT_report / FT_geom
     [P7]);  s10 pause;  s11 lock (live / stale 12 min [c]) + successor + bash launcher (self-check, rc -> ATTENTION)
  s12 every planned job line (48 h, guard segment, no --nice);  s13 lists on the real targets + status
  s14 RESOLVE retry while a sibling still runs [S1/R2];  s15 a manual job never ends the launcher chain [S2]
  s16 FIFO slot policy: eligible pending <= free + 2, filler only into free slots [S3/P3/R5]
  s17 sbatch transient / created-but-timed-out / fixed errors [S4];  s18 RESOLVE + DECISIONS crash-safety [S5]
  s19 job-start guard (temp git repo), exit 4 handling [S6/P8];  s20 A-3 budget + the A'' arm [P1/P2]
  s21 C-2 on the K units [P4];  s22 no submissions without a fresh QUOTA_FREE_GB, disk estimate [P5]
  s23 FT2-p speed tie-break from speed_variants [P6];  s25 not done before prerequisites, Z-run has no G edge [R1]
  s26 lead-cancelled helper: done when complete, STOP + auto-resolve when not [R3];  s27 no log growth in a HALT [R4]
  p1 reconcile with sacct down / without accounting comments [a];  p2 manual job between passes adopted [b]
  p4 LOST only when accounting answers [f];  p5 G_plan_d0 --limit never done [p];  p6 Z coverage [m/o]
  p7 unreadable state [t];  p8 filler HALT is local [i];  p9 ts_geom partial coverage [k];  p11 G-4 budget before
  running, cap read from gen_dft.py [h];  p12 old index.lock -> ATTENTION [e];  p13 git ls-remote (informational)
  Integrate step (DEPLOY-V2 remaining stages): s8 rewritten for the D0 user decisions (8,000 cap, 200 + 100, RESOLVE
  retry of the 5,000-era HALT); n1 A-3 / A-4 progress folders + GPU job size; n2 R commit gate (md + json, sha);
  n3 K-C1 c1pred refs + C-agg decision; n4 A2-smoke completeness / decision HALT; n5 D1 G-2 user stop at 317 / 400
  (decision Nurion); n6 the lead's D0 jobs adopted (d0p_plan, D0G1 / D0G3 / D0G2 arrays, helpers, @I 3 retry);
  n7 N-scan per uploaded Nurion tar + collect + re-arm, FT-dp gate ('N partial ok'); n8 D0-eval (after FT-fin, S waits)
  FT2-p rules (user 11-05): e1 E/Z 'with' / 'without sibling' split (> 0.05 same-TS or > 0.002 A median |dd| -> gate on
  'without' + DEVIATIONS note; list missing -> wait / HALT); e2 failure classes (i)-(iii) + recalc-5 re-measure units
  after the first pass, the gate on the merged records; e3 ts_geom fail_class = pipeline.fail_class, --only-failed,
  --recalc parsing, summary fail_classes / by_ez_sibling; e4 ts_geom task count (wall 2 h, >= 120 s per reaction)"""
from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
MLTS = HERE.parent
sys.path.insert(0, str(MLTS))
import mlt_common as M  # noqa: E402
import pipeline as PL  # noqa: E402

ROOT = Path(os.environ.get("PIPETEST_ROOT", str(M.MLT / "tmp" / "pipetest")))
T0 = time.mktime(time.strptime("2026-11-04T15:40:00", "%Y-%m-%dT%H:%M:%S"))

# squeue -r -u yeseo1ee -h -o '%i|%j|%T|%C|%b|%P|%r|%k' at 2026-11-04 15:3x (login, read-only)
REAL_SQ = """1061711|g_collect|PENDING|4|N/A|cpu1,cpu2|Dependency|(null)
1061798|z_xtb|PENDING|16|N/A|cpu1,cpu2|AssocMaxJobsLimit|(null)
1061800|z_report|PENDING|2|N/A|cpu1,cpu2|Dependency|(null)
1061708_0|g_run1|RUNNING|64|N/A|cpu2|None|(null)
1061708_1|g_run1|RUNNING|64|N/A|cpu2|None|(null)
1061708_2|g_run1|RUNNING|64|N/A|cpu2|None|(null)
1061709_0|g_run3|RUNNING|64|N/A|cpu2|None|(null)
1061710_0|g_run2|RUNNING|64|N/A|cpu2|None|(null)
1061710_1|g_run2|RUNNING|64|N/A|cpu2|None|(null)
1061710_2|g_run2|RUNNING|64|N/A|cpu2|None|(null)
1061779|g_run1rev|RUNNING|64|N/A|cpu2|None|(null)
1061635|mlt_gsandbox|RUNNING|16|gres/gpu:1|gpu1|None|(null)
1061795|z_mh1|PENDING|8|gres/gpu:1|gpu1,gpu2,gpu3,gpu4,gpu5,gpu6|AssocMaxJobsLimit|(null)
1061796|z_aim|PENDING|8|gres/gpu:1|gpu1,gpu2,gpu3,gpu4,gpu5,gpu6|AssocMaxJobsLimit|(null)
1061797|z_off24|PENDING|8|gres/gpu:1|gpu1,gpu2,gpu3,gpu4,gpu5,gpu6|AssocMaxJobsLimit|(null)
1061799|z_speed|PENDING|4|gres/gpu:1|gpu1,gpu2,gpu3,gpu4,gpu5,gpu6|AssocMaxJobsLimit|(null)
1061794|z_omol|RUNNING|8|gres/gpu:1|gpu2|None|(null)""".splitlines()
G_JOBS = ["1061708", "1061709", "1061710", "1061779"]
ALL = [s.name for s in PL.registry()]


class Clock:
    def __init__(self, t=T0):
        self.t = t

    def __call__(self):
        return self.t

    def adv(self, s):
        self.t += s


def stub_code(code, missing=()):
    """one stub file per registry script holding exactly the tokens the code-readiness check looks for."""
    toks = {}
    for S in PL.registry():
        for script, sub, extra in S.code:
            toks.setdefault(script, []).append((sub, extra))
    for script, subs in toks.items():
        L = ["# pipeline test stub"]
        for sub, extra in subs:
            if (script, sub) in set(missing):
                continue
            L.append(f'sub.add_parser("{sub}")')
            if extra:
                L.append(f"# {extra}")
        (code / script).write_text("\n".join(L) + "\n")


class MirrorGit(PL.FakeGit):
    """tracked / clean from the REAL repository (read-only git), commits / pushes faked; names in `assume` count as
    committed (pipeline.py itself: the lead commits it before the first launch)."""

    def __init__(self, assume=(), edited=()):
        super().__init__()
        self.real, self.assume, self.edited = PL.Git(MLTS, readonly=True), set(assume), set(edited)

    def tracked(self, path):
        return Path(path).name in self.assume or Path(path).name in self.committed or self.real.tracked(path)

    def clean(self, path):
        if Path(path).name in self.edited:              # the 2026-11-04 15:3x working tree of the snapshot
            return False
        return Path(path).name in self.assume or self.real.clean(path)


class Env:
    def __init__(self, name, real_code=False, missing=(), dirty=(), quota=150.0):
        self.name = name
        self.d = ROOT / name
        assert str(self.d.resolve()).startswith(str(ROOT.resolve()))
        if self.d.exists():
            shutil.rmtree(self.d)
        self.mlt, self.res = self.d / "mlt", self.d / "res"
        self.code = MLTS if real_code else self.d / "code"
        for p in (self.mlt / "logs", self.res / "ft", self.mlt / "pipeline"):
            p.mkdir(parents=True, exist_ok=True)
        if not real_code:
            self.code.mkdir(parents=True, exist_ok=True)
            stub_code(self.code, missing)
        (self.res / "DECISIONS.md").write_text("# test DECISIONS\n")
        self.clock = Clock()
        self.sl = PL.FakeSlurm(clock=self.clock)
        self.git = PL.FakeGit(dirty=dirty)
        self.quota = quota                         # None: no QUOTA_FREE_GB file; a number: refreshed before each pass

    def ctx(self, launcher=False, job="900"):
        return PL.Ctx(mlt=self.mlt, res=self.res, pdir=self.mlt / "pipeline", ctl=self.mlt, code=self.code, slurm=self.sl,
                      git=self.git, clock=self.clock, job=job, launcher=launcher, nur_upload=self.d / "nur")

    def run(self, dry=False, **kw):
        if self.quota is not None:
            (self.mlt / "QUOTA_FREE_GB").write_text(f"{self.quota} {PL.iso(self.clock())}\n")
        P = PL.Pass(self.ctx(**kw), dry=dry)
        P.run()
        return P

    def state(self):
        p = self.res / "PIPELINE_STATE.json"
        return json.loads(p.read_text()) if p.is_file() else {}

    def stage(self, n):
        return self.state().get("stages", {}).get(n, {})

    def calls(self, pat=None):
        return [" ".join(c) for c in self.sl.calls if pat is None or pat in " ".join(c)]

    def wj(self, path, obj, mtime=None):
        p = path if isinstance(path, Path) else self.res / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj))
        if mtime is not None:
            os.utime(p, (mtime, mtime))
        return p

    def preset(self, done=(), params=None, extra=None):
        st = PL.new_state()
        st["created"] = "test preset"
        for n in done:
            s = PL.new_stage_state()
            s.update(status="done", t_done="preset")
            st["stages"][n] = s
        st["params"].update(params or {})
        st.update(extra or {})
        self.wj(self.res / "PIPELINE_STATE.json", st)

    def lists(self, n=4):
        d = self.mlt / "pipeline" / "lists"
        d.mkdir(parents=True, exist_ok=True)
        files = {}
        for nm in [f"f{k}_test" for k in range(5)] + [f"dip{x}" for x in range(5)] + ["a3"]:
            p = d / f"{nm}.csv"
            p.write_text("rxn_id\n" + "".join(f"{1000000 + i}\n" for i in range(n)))
            files[nm] = dict(path=str(p), n=n)
        self.wj(d / "lists.json", dict(files=files, n_fold_minus1=0))

    def job_of(self, stage, key):
        js = self.stage(stage).get("units", {}).get(key, {}).get("jobs", [])
        return js[-1]["id"] if js else None

    def finish(self, jid, state="COMPLETED", exit=0, dt=60):
        self.sl.set(jid, "RUNNING")
        self.clock.adv(dt)
        self.sl.set(jid, state, exit)
        self.clock.adv(1)

    def dec(self):
        return (self.res / "DECISIONS.md").read_text()

    def live(self, name=None):
        return [l for l in self.sl.squeue() if name is None or l["name"] == name]


RESULTS = []


def check(cond, what):
    RESULTS[-1]["checks"].append(dict(ok=bool(cond), what=what))
    if not cond:
        RESULTS[-1]["ok"] = False
        print(f"    FAIL: {what}", flush=True)


def info(what):
    RESULTS[-1].setdefault("info", []).append(what)
    print(f"    INFO: {what}", flush=True)


def scenario(fn):
    fn._scenario = True
    return fn


def halt_until(E, stage, key, state="TIMEOUT", n=5):
    """end the unit's latest job with `state` and run a pass, until the stage halts."""
    for _ in range(n):
        if E.stage(stage).get("status") == "halted":
            return
        E.finish(E.job_of(stage, key), state, 0)
        E.run()


def _real_snapshot(E, launcher=True):
    for l in REAL_SQ:
        E.sl.add_line(l)
    if launcher:
        E.sl.add("1062000", "mltp_launch", "RUNNING", cpus=1)
        E.sl.add("1062001", "mltp_launch", "PENDING", cpus=1, reason="BeginTime")
    for f in ("G0_estimate.json", "G_plan.json"):
        if (M.RES / f).is_file():
            shutil.copy(M.RES / f, E.res / f)


def _z_json(E, c, n=4, cov=None, finished=True, mtime=None):
    J = dict(candidate=c, available=True, n_rxn=n, geometry=dict(all=dict(status_counts=dict(ok=n if cov is None else cov))))
    if finished:
        J["finished"] = "x"
    return E.wj(E.mlt / "z" / c / "z.json", J, mtime=mtime)


def _ft_done(E, run, fp="fp"):
    E.wj(E.mlt / "models" / f"{run}.json", dict(run=run))
    E.wj(E.res / "ft" / f"{run}_data.json", dict(fingerprint=fp, inputs_complete=True, leakage_C2=dict(pass_=True)))
    E.wj(E.res / "ft" / f"{run}_train.json", dict(status="done", data_fingerprint=fp, cand="MACE-OMOL-0", gpu_h=3.0),
         mtime=E.clock() - 10)
    E.wj(E.res / "ft" / f"{run}_ft1.json", dict(run=run, sets=dict(valid=dict(force_mae_kcal_A=dict(ts_near=0.5)))),
         mtime=E.clock())


def _geom(E, root, run, start, n, dd, cov=None):
    d = E.mlt / root / run / start
    d.mkdir(parents=True, exist_ok=True)
    cov = n if cov is None else cov
    E.wj(d / "summary.json", dict(all=dict(status_counts=dict(ok=cov), same_ts_frac=1.0, dd_median_A=dd, dd_p90_A=dd,
                                           n_attempted=cov, rmsd_median_A=0.01)))
    rows = "rxn_id,status,converged,dd_max,rmsd_heavy,same_ts,n_imag\n" + "".join(
        f"{1000000 + i},ok,True,{dd},0.01,True,1.0\n" for i in range(cov))
    (d / "geom.csv").write_text(rows)


GEOM_COLS = ("rxn_id", "status", "converged", "dd_max", "rmsd_heavy", "same_ts", "n_imag")


def _row(i, cls=None, dd=0.004):
    """one geom.csv row: passing, or failing in class (i) not_converged / (ii) n_imag / (iii) other_saddle."""
    r = dict(rxn_id=1000000 + i, status="ok", converged=True, dd_max=dd, rmsd_heavy=0.01, same_ts=True, n_imag=1.0)
    if cls == "not_converged":
        r.update(status="not_converged", converged=False, dd_max=0.2, same_ts=False)
    elif cls == "n_imag":
        r.update(n_imag=2.0)
    elif cls == "other_saddle":
        r.update(same_ts=False, dd_max=0.25, rmsd_heavy=0.3)
    return r


def _geom_rows(E, run, start, rows, root="ft2"):
    """geom.csv from per-reaction rows (+ a summary.json whose status counts cover them)."""
    d = E.mlt / root / run / start
    d.mkdir(parents=True, exist_ok=True)
    (d / "geom.csv").write_text(",".join(GEOM_COLS) + "\n" + "".join(",".join(str(r.get(c, "")) for c in GEOM_COLS) + "\n"
                                                                     for r in rows))
    E.wj(d / "summary.json", dict(all=dict(status_counts=dict(ok=len(rows)), n_attempted=len(rows))))
    return d


def _ez(E, n, with_ids=()):
    """the E/Z sibling list (ez_groups.py layout) for rxn 1000000 .. 1000000 + n - 1."""
    p = E.mlt / "pipeline" / "lists" / "f0_test_ez.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("rxn_id,role,is_a,is_b,ez_sibling_in_train,matching_train_core\n" + "".join(
        f"{1000000 + i},a,True,False,{i in set(with_ids)},{'nitrile oxide/AppZ' if i in set(with_ids) else ''}\n"
        for i in range(n)))
    return p


# ================================================================ scenarios: the original 13, new policy
@scenario
def s1_fresh_real():
    E = Env("s1_fresh_real", real_code=True)
    # the real code tree, with the 15:3x state of the snapshot: gen_dft.py / gen_d0.py / finetune.py being edited
    E.git = MirrorGit(assume={"pipeline.py"}, edited={"gen_dft.py", "gen_d0.py", "finetune.py"})
    _real_snapshot(E)
    E.run()
    S = E.state()
    g = S["stages"]["G-D1"]
    check(g["status"] == "running", f"G-D1 running (got {g['status']}: {g['why']})")
    check(E.job_of("G-D1", "collect") == "1061711", "manual collect 1061711 adopted as G-D1/collect")
    check(all(E.job_of("G-D1", k) for k in ("G1[0]", "G1[1]", "G1[2]", "G1rev", "G3[0]", "G2[0]", "G2[1]", "G2[2]")),
          "g_run1 / g_run1rev / g_run3 / g_run2 adopted unit by unit")
    check(S["stages"]["Z-run"]["status"] in ("running", "queued"), "Z-run adopted (live)")
    check(all(E.job_of("Z-run", c) for c in PL.Z_CANDS), "z_omol / z_mh1 / z_aim / z_off24 / z_xtb adopted")
    check(E.job_of("Z-speed", "speed") == "1061799" and E.job_of("Z-rep", "report") == "1061800", "z_speed, z_report adopted")
    check(S["stages"]["G-D0p"]["status"] == "waiting_code",
          f"G-D0p waiting: code not committed yet ({S['stages']['G-D0p']['why']})")
    check(S["stages"]["FT-dp"]["status"] in ("blocked", "waiting_code"), f"FT-dp not startable: {S['stages']['FT-dp']['why']}")
    check(not E.calls() and "no room" in S["stages"]["lists"]["why"],
          f"19 tasks incl. the launchers (20 - 1 spare): nothing submitted ({S['stages']['lists']['why'][-160:]})")
    check(S["slots"]["total"] == 19 and S["slots"]["eligible_pending"] == 5, f"slots {S['slots']}")
    check("QUOTA_FREE_GB" not in (E.mlt / "PIPELINE_ATTENTION").read_text(), "fresh quota value -> no quota attention")
    check(not (E.mlt / "pipeline" / "lock").exists(), "lock released")
    # the G runs end: 3 running (sandbox, z_omol, launcher), 5 eligible pending -> 2 free + 2 buffer
    E.clock.adv(3600)
    for j in G_JOBS:
        E.finish(j)
    E.run()
    c = E.calls()
    check(len(c) == 1 and "pipeline.py lists" in c[0], f"slots free -> exactly one submission: lists ({c})")
    check(len(E.sl.squeue()) <= PL.SUBMIT_CAP - PL.SPARE, f"queue after the pass <= 19 ({len(E.sl.squeue())})")
    c0 = c[0] if c else ""
    check("--time=48:00:00" in c0 and "--nice" not in c0 and "--comment=mltp:lists:lists:" in c0 and "-J mltp_lists" in c0
          and "pipeline.py guard mlt_common.py pipeline.py :::" in c0, f"sbatch line: 48 h, no nice, comment, guard ({c0[-160:]})")
    E.clock.adv(600)
    E.run()
    check(len(E.calls()) == 1, "next pass: no duplicate submission")
    before = (E.res / "PIPELINE_STATE.json").read_text()
    P = E.run(dry=True)
    check(len(E.calls()) == 1 and (E.res / "PIPELINE_STATE.json").read_text() == before, "dry pass: no sbatch, no state write")
    check(isinstance(P.c.slurm, PL.ReadOnlySlurm), "dry pass uses the read-only SLURM wrapper")


@scenario
def s2_adopt_collect():
    E = Env("s2_adopt_collect", real_code=True)
    _real_snapshot(E)
    E.run()
    E.clock.adv(3600)
    for j in G_JOBS:
        E.finish(j)
    E.finish("1061711")
    E.wj("G_report.json", dict(complete=True, stop_file=None, n_missing=0, core_h_measured=dict(total=2900.0),
                               groups=dict(G1=dict(n_missing=0, n_planned=27798), G3=dict(n_missing=0, n_planned=21424),
                                           G2=dict(n_missing=0, n_planned=400))))
    E.run()
    g = E.stage("G-D1")
    check(g["status"] == "done", f"G-D1 done after the adopted collect 1061711 finished ({g['status']}: {g['why']})")
    j = [x for x in g["units"]["collect"]["jobs"] if x["id"] == "1061711"][0]
    check((j.get("end") or {}).get("state") == "COMPLETED", "collect end state recorded from sacct")
    check(E.git.commits and any(p.endswith("G_report.json") for p in E.git.commits[-1]["paths"]), "commit has G_report.json")
    check(E.git.commits and "MLTS pipeline: G-D1 done" in E.git.commits[-1]["title"], "commit title")
    check(E.git.commits and PL.TRAILER in E.git.commits[-1]["body"], "commit trailer")
    check(E.git.pushes and E.git.pushes[-1] is True, "pushed")
    check(E.stage("Z-run")["status"] in ("running", "queued"), "Z-run still running (adopted)")


@scenario
def s2b_collect_incomplete():
    E = Env("s2b_collect_incomplete", real_code=True)
    for l in REAL_SQ:
        if "|g_" in l:
            E.sl.add_line(l)
    shutil.copy(M.RES / "G0_estimate.json", E.res / "G0_estimate.json")
    E.run()
    E.clock.adv(3600)
    for j in G_JOBS:
        E.finish(j)
    E.finish("1061711", "FAILED", 2)
    E.wj("G_report.json", dict(complete=False, stop_file=None, n_missing=500,
                               groups=dict(G1=dict(n_missing=500, n_planned=27798), G3=dict(n_missing=0, n_planned=21424),
                                           G2=dict(n_missing=0, n_planned=400))))
    E.run()
    g = E.stage("G-D1")
    runs = E.calls("gen_dft.py run G1")
    check(g["epoch"] == 1, f"collect exit 2 with all inputs final -> round 2 (epoch {g['epoch']})")
    check(len(runs) == 1 and "--array=0" in runs[0] and "-p cpu2" in runs[0] and "--cpus-per-task=64" in runs[0],
          f"round 2: G1 only (n_missing > 0), cpu2 64 cores ({runs})")
    check(runs and "pipeline.py guard gen_d0.py gen_dft.py mlt_common.py orca_io.py :::" in runs[0],
          "guard covers gen_dft.py + the modules it imports (gen_d0, orca_io, mlt_common)")
    check(not E.calls("gen_dft.py collect"), "collect waits until the runs are RUNNING (lookahead 1)")
    jid = E.job_of("G-D1", "G1[0]")
    E.sl.set(jid, "RUNNING")
    E.run()
    col = E.calls("gen_dft.py collect")
    check(len(col) == 1 and f"--dependency=afterany:{jid}" in col[0], f"collect queued afterany the running G1 task ({col})")


@scenario
def s3_stage_done_decision():
    for name, top, expect in (("s3_zrep", ["MACE-OMOL-0", "MACE-MH-1"], ["MACE-OMOL-0", "MACE-MH-1-omol"]),
                              ("s3b_zrep_one", ["AIMNet2"], ["AIMNet2"]), ("s3c_zrep_none", [], None)):
        E = Env(name)
        E.preset(done=["G-D1", "lists"])
        for c in PL.Z_CANDS:
            _z_json(E, c, mtime=E.clock() - 100)
        E.wj(E.mlt / "z" / "speed.json", {t: dict(ef_ms_median=10.0) for t in PL.SPEED_DEFAULT_TAG.values()},
             mtime=E.clock() - 100)
        E.wj("Z_report.json", dict(ranking=top + ["xx"] if top else [], top2_for_ft_pilot=top, candidates={}),
             mtime=E.clock())
        (E.res / "Z_DECISION.md").write_text("# z\n")
        E.run()
        st = E.state()
        if expect:
            check(st["stages"]["Z-rep"]["status"] == "done" and st["params"].get("pilots") == expect,
                  f"{name}: Z-rep done, pilots {st['params'].get('pilots')} == {expect}")
            check("auto Z-rep" in E.dec() and "<!-- mltp:" in E.dec(), f"{name}: DECISIONS line written (with its id)")
            if len(expect) == 1:
                check("하나뿐" in E.dec(), f"{name}: one eligible -> recorded as the recommended option")
            check(E.git.commits and any(p.endswith("Z_report.json") for p in E.git.commits[-1]["paths"]),
                  f"{name}: commit has Z_report.json")
            check(st["stages"]["FT-dp"]["status"] == "blocked", f"{name}: FT-dp waits for G-D0 (user D0 rule)")
        else:
            check(st["stages"]["Z-rep"]["status"] == "halted" and (E.res / "STOP_Z-rep.md").is_file(),
                  f"{name}: no eligible candidate -> HALT + STOP_Z-rep.md")


@scenario
def s4_timeout_retry():
    E = Env("s4_timeout")
    E.preset(done=[n for n in ALL if n not in ("Z-run", "Z-speed", "Z-rep")])
    for c in PL.Z_CANDS[:-1]:
        _z_json(E, c)
    E.run()
    c1 = E.calls("zero_shot.py run xtb")
    check(len(c1) == 1 and "-p cpu1,cpu2" in c1[0] and "--gres" not in c1[0], f"xtb unit submitted on CPU ({c1})")
    jid = E.job_of("Z-run", "xtb")
    rec = E.mlt / "z" / "xtb" / "geom_label" / "recs"
    rec.mkdir(parents=True)
    for i in range(5):
        (rec / f"{i}.json").write_text("{}")
    E.clock.adv(48 * 3600)
    E.finish(jid, "TIMEOUT", 0)
    E.run()
    c2 = E.calls("zero_shot.py run xtb")
    us = E.stage("Z-run")["units"]["xtb"]
    check(len(c2) == 2 and c2[1].split("jobs/mlt.slurm")[1] == c1[0].split("jobs/mlt.slurm")[1],
          "48 h clip (TIMEOUT) with progress -> same command resubmitted")
    check(us["att"]["cont"] == 1 and us["noprog"] == 0, f"attempts cont 1, no-progress 0 ({us['att']}, {us['noprog']})")
    E.finish(E.job_of("Z-run", "xtb"), "TIMEOUT", 0)
    E.run()
    us = E.stage("Z-run")["units"]["xtb"]
    check(len(E.calls("zero_shot.py run xtb")) == 3 and us["noprog"] == 1, "no progress once -> resubmitted, noprog 1")
    E.finish(E.job_of("Z-run", "xtb"), "TIMEOUT", 0)
    E.run()
    st = E.stage("Z-run")
    check(st["status"] == "halted" and st.get("halt_kind") == "ops" and (E.res / "STOP_Z-run.md").is_file(),
          "2 resubmits without progress -> HALT (operational)")
    check(len(E.calls("zero_shot.py run xtb")) == 3, "no 4th submission while halted")
    check("HALT" in E.dec(), "HALT recorded in DECISIONS")
    txt = (E.res / "STOP_Z-run.md").read_text()
    (E.res / "STOP_Z-run.md").write_text("RESOLVE: retry\n" + txt)
    E.run()
    check((E.res / "STOP_Z-run.resolved.md").is_file() and not (E.res / "STOP_Z-run.md").exists(),
          "RESOLVE retry: STOP file renamed to .resolved.md")
    check(len(E.calls("zero_shot.py run xtb")) == 4 and E.stage("Z-run")["status"] in ("queued", "running"),
          "RESOLVE retry: resubmitted")
    check(E.dec().count("RESOLVE: retry") == 1, "RESOLVE recorded once in DECISIONS")
    last = E.git.commits[-1]["paths"] if E.git.commits else []
    check(any(p.endswith("STOP_Z-run.md") for p in last) and any(p.endswith("STOP_Z-run.resolved.md") for p in last),
          "the commit carries the removal of the old tracked STOP_Z-run.md and the .resolved.md [n]")


@scenario
def s5_crash_intent():
    others = [n for n in ALL if n != "lists"]
    E = Env("s5_crash_after")
    E.preset(done=others)
    E.sl.fail = "crash_after"
    try:
        E.run()
        check(False, "pass should have raised (fake crash after sbatch)")
    except RuntimeError:
        pass
    it = E.state()["intents"][-1]
    check(it["status"] == "submitting" and len(E.sl.jobs) == 1, "crash after sbatch: intent left 'submitting', job exists")
    check(not (E.mlt / "pipeline" / "lock").exists(), "lock released by the crashed pass")
    E.sl.fail = None
    E.clock.adv(30)
    E.run()
    it = E.state()["intents"][-1]
    check(it["status"] == "adopted" and E.job_of("lists", "lists") == list(E.sl.jobs)[0], "intent matched by comment -> adopted")
    check(len(E.calls()) == 1, "no duplicate sbatch")
    E2 = Env("s5b_crash_before")
    E2.preset(done=others)
    E2.sl.fail = "crash_before"
    try:
        E2.run()
    except RuntimeError:
        pass
    E2.sl.fail = None
    E2.clock.adv(30)
    E2.run()
    check(len(E2.sl.jobs) == 0 and E2.state()["intents"][-1]["status"] == "submitting",
          "crash before sbatch: < 2 min -> intent kept, unit not resubmitted")
    E2.clock.adv(150)
    E2.run()
    ints = E2.state()["intents"]
    check(ints[0]["status"] == "void" and len(E2.sl.jobs) == 1 and ints[-1]["status"] == "submitted",
          "> 2 min, no job found -> intent void, resubmitted once")


@scenario
def s6_queue_full():
    E = Env("s6_full")
    E.preset(done=[n for n in ALL if n != "lists"])
    for i in range(17):                                   # Dependency-pending: no running slot, but queue places
        E.sl.add(f"20000{i:02d}", "other", "PENDING", cpus=1, reason="Dependency")
    E.sl.add("1062000", "mltp_launch", "RUNNING", cpus=1)
    E.sl.add("1062001", "mltp_launch", "PENDING", cpus=1, reason="BeginTime")
    E.run()
    st = E.stage("lists")
    check(not E.calls() and st["status"] == "ready" and "no room" in st["why"],
          f"19 tasks (17 + 2 launchers) -> nothing submitted ({st['why'][-120:]})")
    E.finish("2000000")
    E.run()
    check(len(E.calls("pipeline.py lists")) == 1 and len(E.sl.squeue()) == 19, "a place frees -> submitted, queue 19")
    E2 = Env("s6b_maxsubmit")
    E2.preset(done=[n for n in ALL if n != "lists"])
    E2.sl.fail = "maxsubmit"
    P = E2.run()
    check(P.stop_submit == "MaxSubmit" and E2.state()["intents"][-1]["status"] == "failed",
          "sbatch MaxSubmit -> no more submissions this pass, intent failed (not counted)")
    E2.sl.fail = None
    E2.run()
    check(len(E2.sl.jobs) == 1, "next pass submits")


@scenario
def s7_code_appears():
    E = Env("s7_code", missing={("gen_dft.py", "plan_d0")})
    E.preset(done=["lists", "G-D1"])
    E.run()
    st = E.stage("G-D0p")
    check(st["status"] == "waiting_code" and "plan_d0" in st["why"] and not E.calls("plan_d0"),
          f"placeholder without code: waiting ({st['why']})")
    stub_code(E.code)
    E.git.dirty = {"gen_dft.py"}
    E.run()
    st = E.stage("G-D0p")
    check(st["status"] == "waiting_code" and "modified" in st["why"], f"code present but being edited -> waiting ({st['why']})")
    E.git.dirty = set()
    E.run()
    c = E.calls("gen_dft.py plan_d0")
    check(len(c) == 1 and "--cpus-per-task=16" in c[0], f"committed -> submitted automatically ({c})")


@scenario
def s8_spec_stop_recommended():
    """D0 (user decisions D0-n3 / D0-plus / D0-gauss, cap gen_dft.BUDGET_CORE_H = 8,000): plan_d0 (auto) -> g0_d0 ->
    the budget check of the user's selection (its 5,000-era stop flag ignored) -> gen_d0.py plan_d0 --n-near 2 --n-g2
    200 --n-holdout 100 -> D0 runs; over the cap -> HALT -> RESOLVE param G-D0f.n_g2=30 -> continue."""
    def est(total, n, stop, d1=2969.0, fp="fp1"):
        return dict(plan_fingerprint=dict(id=fp, n_near=1), predicted_core_h=dict(
                    total=total, d1_total=d1, budget=5000.0,
                    by_group=dict(D0G1=900.0, D0G3=300.0, D0G2=max(0.0, total - d1 - 1200.0))),
                    n_g2_rule=n, stop=stop, suggested_tasks=dict(D0G1=2, D0G3=1, D0G2=0))
    E = Env("s8_g0d0")
    E.preset(done=["lists", "G-D1"])
    E.run()
    c = E.calls("gen_dft.py plan_d0")
    check(len(c) == 1 and "--n-g2" not in c[0], f"plan_d0 (auto) first ({c})")
    E.finish(E.job_of("G-D0p", "plan_d0"))
    E.wj("G_plan_d0.json", dict(fingerprint=dict(id="fp1"), limit=None,
                                budget=dict(n_g2_selected=0, d0_g1_g3=2185.6, basis=dict(d1_total=3411.3),
                                            rule=dict(g2_core_h_per_rxn=dict(mean=5.9)))))
    E.run()
    check(E.stage("G-D0p")["status"] == "done" and len(E.calls("gen_dft.py g0_d0")) == 1, "plan done -> g0_d0 submitted")
    E.wj("G0_estimate_d0.json", est(5071.0, 0, True, d1=3416.0))      # the real 1061949 numbers: stop vs 5,000
    E.finish(E.job_of("G-D0e", "g0_d0"), "FAILED", 3)
    E.run()
    S = E.state()
    p = S["stages"]["G-D0f"]["params"]
    check(S["stages"]["G-D0e"]["status"] == "done" and p.get("n_g2") == 200 and p.get("n_holdout") == 100,
          f"exit 3 against the old 5,000 cap; under 8,000 with 200 D0G2 + G-4 reserve -> user selection ({p})")
    nums = (S["stages"]["G-D0e"].get("decision") or {}).get("numbers") or {}
    check(abs(nums.get("total", 0) - (3411.3 + 2185.6 + 200 * 5.9 + 150)) < 0.2 and nums.get("budget") == 8000.0,
          f"budget basis: plan d1 basis + plan D0 G1/G3 + 200 x 5.9 + 150 vs 8,000 ({nums.get('total')})")
    check("D0-plus" in E.dec() and "--n-holdout 100" in E.dec(), "the decision line names the user decisions")
    c = E.calls("gen_d0.py plan_d0")
    check(len(c) == 1 and c[0].endswith("--n-near 2 --n-g2 200 --n-holdout 100"),
          f"gen_d0.py plan_d0 --n-near 2 --n-g2 200 --n-holdout 100 in the same pass ({c[-1:]})")
    E.finish(E.job_of("G-D0f", "plan_d0"))
    E.wj("G_plan_d0.json", dict(fingerprint=dict(id="fp1"), n_g2=200, n_holdout=99,
                                budget=dict(n_g2_selected=200, d0_g1_g3=2185.6, rule=dict(g2_core_h_per_rxn=dict(mean=5.9)))))
    E.run()
    check(E.stage("G-D0f")["status"] != "done" and len(E.calls("gen_d0.py plan_d0")) == 2,
          "a plan with 99 holdout reactions is not the requested one -> plan_d0 again")
    E.wj("G_plan_d0.json", dict(fingerprint=dict(id="fp1"), n_g2=200, n_holdout=100,
                                budget=dict(n_g2_selected=200, d0_g1_g3=2185.6, rule=dict(g2_core_h_per_rxn=dict(mean=5.9)))))
    E.finish(E.job_of("G-D0f", "plan_d0"))
    E.run()
    c1, c2 = E.calls("run D0G1"), E.calls("run_g2 @I 2 --d0")
    check(E.stage("G-D0f")["status"] == "done" and len(c1) == 1 and "--array=0,1" in c1[0] and "-p cpu2" in c1[0],
          f"D0G1 array of suggested_tasks = 2 on cpu2 ({c1})")
    check(len(c2) == 1 and "MLTENV=d1" in c2[0] and "--array=0,1" in c2[0],
          f"D0G2 (the 200 training units) in the d1 env, 2 tasks ({c2 or E.calls('run_g2')})")
    check(len(E.calls("run D0G3")) == 1 and not E.calls("collect --d0"), "D0G3 queued; collect waits for running runs")
    E2 = Env("s8b_g0d0_halt")
    E2.preset(done=["lists", "G-D1"])
    E2.run()
    E2.finish(E2.job_of("G-D0p", "plan_d0"))
    E2.wj("G_plan_d0.json", dict(fingerprint=dict(id="fp1"), budget=dict(n_g2_selected=0, d0_g1_g3=2185.6,
                                                                         rule=dict(g2_core_h_per_rxn=dict(mean=5.9)))))
    E2.run()
    E2.wj("G0_estimate_d0.json", est(9000.0, 0, True, d1=5600.0))
    E2.finish(E2.job_of("G-D0e", "g0_d0"), "FAILED", 3)
    E2.run()
    S = E2.state()
    check(S["stages"]["G-D0e"]["status"] == "halted" and (E2.res / "STOP_G-D0e.md").is_file(),
          "5,600 + 2,186 + 1,180 + 150 > 8,000 -> HALT")
    check(S["stages"]["G-D0f"]["status"] == "blocked" and len(E2.calls("plan_d0")) == 1, "refit / D0 runs not started")
    txt = (E2.res / "STOP_G-D0e.md").read_text()
    (E2.res / "STOP_G-D0e.md").write_text("RESOLVE: param G-D0f.n_g2=30\n" + txt)
    E2.run()
    S = E2.state()
    c = E2.calls("plan_d0")
    check(S["stages"]["G-D0e"]["status"] == "done" and len(c) == 2 and c[-1].endswith("--n-g2 30 --n-holdout 100"),
          f"RESOLVE param G-D0f.n_g2=30: G-D0e passed, plan_d0 --n-g2 30 submitted ({c[-1:]})")
    E3 = Env("s8c_retry_real_stop")              # the real state: G-D0e halted on 1061949 (5,000 era), lead RESOLVE retry
    E3.preset(done=["lists", "G-D1", "G-D0p"])
    E3.wj("G_plan_d0.json", dict(fingerprint=dict(id="fp1"), budget=dict(n_g2_selected=0, d0_g1_g3=2185.6,
                                                                         rule=dict(g2_core_h_per_rxn=dict(mean=5.9)))))
    E3.wj("G0_estimate_d0.json", est(5071.0, 0, True, d1=3416.0))
    S = E3.state()
    S["stages"]["G-D0e"] = dict(PL.new_stage_state(), status="halted", halt_kind="decision", why="g0: exit 3")
    S["halted"] = ["G-D0e"]
    E3.wj(E3.res / "PIPELINE_STATE.json", S)
    (E3.res / "STOP_G-D0e.md").write_text("RESOLVE: retry\n# STOP: G-D0e\n")
    E3.run()
    check(E3.stage("G-D0e")["status"] == "done" and not E3.calls("g0_d0") and len(E3.calls("gen_d0.py plan_d0")) == 1,
          "RESOLVE retry: estimate fingerprint = plan fingerprint without n_near -> decided again, no g0_d0 rerun, "
          "plan_d0 --n-g2 200 --n-holdout 100 submitted")


@scenario
def s9_halt_resolve():
    E = Env("s9_halt")
    upto = ALL[:ALL.index("FT-d2") + 1] + ["FT-r2"]
    E.preset(done=upto, params=dict(base="MACE-OMOL-0", base_run="FTpilot_MACE-OMOL-0_0",
                                    pilots=["MACE-OMOL-0", "AIMNet2"]))
    E.lists(4)
    for k in range(5):
        _ft_done(E, f"FT_{k}_al")
        for s in PL.STARTS:
            _geom(E, "ft2", f"FT_{k}_al", s, 4, 0.02)
    E.run()
    S = E.state()
    check(S["stages"]["FT2-r2"]["status"] == "halted" and (E.res / "STOP_FT2-r2.md").is_file(),
          "pooled FT-2 (a) fails (|dd| 0.02 > 0.010) -> HALT")
    check("FT2-r2" in S["halted"] and "HALT" in (E.mlt / "PIPELINE_ATTENTION").read_text(), "halted list + attention")
    check(not E.calls("tests_c1.py") and "no new stages" in S["stages"]["C1-ft"]["why"],
          f"HALT: C1-ft (ready, never started) not started ({S['stages']['C1-ft']['why']})")
    check(not E.calls("FT_all_s0"), "HALT: filler FT-fin not started")
    R = json.loads((E.res / "FT_report.json").read_text()) if (E.res / "FT_report.json").is_file() else {}
    rows = (E.res / "FT_geom.csv").read_text().splitlines() if (E.res / "FT_geom.csv").is_file() else []
    check(R.get("gate_round2", {}).get("pass_all") is False and "FT_0_al" in R.get("runs", {}),
          "results/FT_report.json written at the FT-2 decision (runs, FT-1, gate) [P7]")
    check(len(rows) == 1 + 5 * 2 * 4 and rows[0].startswith("run,fold,start,rxn_id"),
          f"results/FT_geom.csv: one row per FT_<k>_al reaction and start ({len(rows)} lines) [P7]")
    check(any(p.endswith("FT_geom.csv") for c in E.git.commits for p in c["paths"]), "FT_report / FT_geom committed")
    txt = (E.res / "STOP_FT2-r2.md").read_text()
    check("RESOLVE" in txt and "추천" in txt and "조건" in txt, "STOP file: condition, options, recommended, how to clear")
    (E.res / "STOP_FT2-r2.md").write_text("RESOLVE: accept\n" + txt)
    E.run()
    S = E.state()
    check(S["stages"]["FT2-r2"]["status"] == "done" and not S["halted"], "RESOLVE accept -> done, HALT cleared")
    check((E.res / "STOP_FT2-r2.resolved.md").is_file(), "STOP renamed (not deleted)")
    c = E.calls("tests_c1.py all FT_0_al")
    check(len(c) == 1 and "gpu3,gpu4,gpu5" in c[0], f"C1-ft starts on G48 after the RESOLVE ({c})")
    fin = E.calls("FT_all_s0")
    check(len(fin) == 1 and "--nice" not in fin[0], "FT-fin filler submitted into free slots (no --nice)")
    check("LODO" in E.dec(), "FT-fin recommended option (LODO with AL) recorded at its start")


@scenario
def s10_pause():
    E = Env("s10_pause")
    E.preset(done=[n for n in ALL if n not in ("lists", "Z-run")])
    E.sl.add("1062001", "mltp_launch", "PENDING", cpus=1, reason="BeginTime")      # the bash-queued successor
    (E.mlt / "PIPELINE_PAUSE").write_text("")
    E.sl.add("3000001", "my_manual_z", "RUNNING", cpus=32, comment="mltp:Z-run:xtb:manual")
    E.run(launcher=True)
    S = E.state()
    check(not E.calls() and S["paused"] is True, "PIPELINE_PAUSE: no submissions")
    check(E.job_of("Z-run", "xtb") == "3000001", "pause: reconcile / adoption continue (manual job by comment)")
    check("paused" in S["stages"]["lists"]["why"], "status says paused")
    (E.mlt / "PIPELINE_PAUSE").write_text("exit\n")
    E.run(launcher=True)
    check(not E.calls(), "PIPELINE_PAUSE=exit: no successor, no submissions")
    (E.mlt / "PIPELINE_PAUSE").unlink()
    E.run()
    check(len(E.calls("pipeline.py lists")) == 1, "pause removed -> submits")


@scenario
def s11_singleton():
    E = Env("s11_lock")
    E.preset(done=[n for n in ALL if n != "lists"])
    (E.mlt / "pipeline" / "lock").write_text(f"777 {PL.iso(E.clock() - 60)} n1 4242\n")
    E.sl.add("777", "mltp_launch", "RUNNING", cpus=1)
    before = (E.res / "PIPELINE_STATE.json").read_text()
    E.run()
    check(not E.calls() and (E.res / "PIPELINE_STATE.json").read_text() == before, "lock held by a live launcher -> no pass")
    E.clock.adv(PL.LOCK_STALE_S + 60)                    # [c] still RUNNING (e.g. the sandbox) but older than 12 min
    E.run()
    log = (E.res / "PIPELINE_LOG.md").read_text()
    check("lock takeover from 777" in log, "a 13-min-old lock of a RUNNING job is stale -> taken over")
    E.sl.set("777", "COMPLETED")
    (E.mlt / "pipeline" / "lock").write_text(f"777 {PL.iso(E.clock() - 60)} n1 4242\n")
    E.run(launcher=True, job="1062100")
    check(not (E.mlt / "pipeline" / "lock").exists(), "stale lock (holder ended) taken over, released")
    succ = E.calls("jobs/pipeline.slurm")
    check(len(succ) == 1 and "--begin=now+10minutes" in succ[0], f"launcher without a pending successor queues one ({succ})")
    E.run(launcher=True, job="1062100")
    check(len(E.calls("jobs/pipeline.slurm")) == 1, "successor pending -> not queued again (singleton)")
    bash_launcher(E)


def bash_launcher(E):
    d = E.d / "bash"
    fb = d / "bin"
    fb.mkdir(parents=True, exist_ok=True)
    (d / "mlt").mkdir(exist_ok=True)
    (d / "res").mkdir(exist_ok=True)
    for nm, body in (("squeue", '#!/bin/bash\n[ -f "$FAKE_SQ_FAIL" ] && exit 1\ncat "$FAKE_SQ" 2>/dev/null\n'),
                     ("sbatch", '#!/bin/bash\necho "$@" >> "$FAKE_SB"\necho 4242\n'),
                     ("sleep", '#!/bin/bash\nexit 0\n')):
        p = fb / nm
        p.write_text(body)
        p.chmod(p.stat().st_mode | stat.S_IXUSR)
    env = dict(os.environ, PATH=f"{fb}:{os.environ.get('PATH', '')}", SLURM_SUBMIT_DIR=str(MLTS), SLURM_JOB_ID="500",
               MLT_SCRATCH=str(d / "mlt"), MLT_RESULTS=str(d / "res"), PIPELINE_NO_PASS="1",
               FAKE_SQ=str(d / "sq.txt"), FAKE_SB=str(d / "sb.txt"), FAKE_SQ_FAIL=str(d / "sq_fail"),
               USER=os.environ.get("USER", "yeseo1ee"))

    def go(sq, pause=None, stop_s=False, sq_fail=False, **extra):
        (d / "sq.txt").write_text(sq)
        for f in ("sb.txt", "sq_fail"):
            if (d / f).exists():
                (d / f).unlink()
        if sq_fail:
            (d / "sq_fail").write_text("1")
        pp = d / "mlt" / "PIPELINE_PAUSE"
        if pause is None:
            pp.unlink() if pp.exists() else None
        else:
            pp.write_text(pause)
        sp = d / "res" / "STOP_S.md"
        if stop_s:
            sp.write_text("# 명령서 S: 멈춤\n")
        elif sp.exists():
            sp.unlink()
        att = d / "mlt" / "PIPELINE_ATTENTION"
        if att.exists():
            att.unlink()
        e2 = dict(env, **extra)
        for k, v in list(e2.items()):
            if v is None:
                e2.pop(k)
        r = subprocess.run(["bash", str(MLTS / "jobs" / "pipeline.slurm")], env=e2, capture_output=True, text=True,
                           timeout=120)
        sb = (d / "sb.txt").read_text() if (d / "sb.txt").exists() else ""
        return r, sb, (att.read_text() if att.exists() else "")

    r, sb, _ = go("501\n")
    check(r.returncode == 0 and sb == "", f"bash: another pending launcher -> no successor ({r.stdout.strip()[-80:]})")
    r, sb, _ = go("")
    check("--parsable --begin=now+10minutes jobs/pipeline.slurm" in sb, f"bash: none pending -> successor queued ({sb!r})")
    r, sb, _ = go("500\n")
    check(sb != "", "bash: only itself listed -> successor queued")
    r, sb, _ = go("", pause="exit\n")
    check(sb == "" and "no successor" in r.stdout, "bash: PIPELINE_PAUSE=exit -> no successor")
    r, sb, _ = go("", pause="")
    check(sb != "", "bash: empty PIPELINE_PAUSE (pause only) -> successor still queued")
    r, sb, _ = go("", stop_s=True)
    check(sb == "" and "STOP_S.md" in r.stdout, "bash: STOP_S.md -> no successor (normal end)")
    r, sb, _ = go("501\n", sq_fail=True)
    check(sb != "" and "squeue failed 3 times" in r.stdout, "bash: squeue failing 3 x -> successor queued anyway [d]")
    # the pass itself: rc != 0 -> PIPELINE_ATTENTION with the last output line
    r, sb, att = go("501\n", PIPELINE_NO_PASS=None, PIPELINE_SKIP_SELFCHECK="1", PIPELINE_TEST_PY="false")
    check("launcher pass exit 1" in att, f"bash: a failing pass writes PIPELINE_ATTENTION ({att.strip()[:120]})")
    # the self-check against the real repository state of pipeline.py / mlt_common.py
    g = PL.Git(MLTS, readonly=True)
    ok = all(g.tracked(MLTS / f) and g.clean(MLTS / f) for f in ("pipeline.py", "mlt_common.py"))
    r, sb, att = go("501\n", PIPELINE_NO_PASS=None, PIPELINE_TEST_PY="true")
    if ok:
        check("pass skipped" not in att and "pass exit 0" in r.stdout, "bash: committed launcher code -> pass runs")
    else:
        check("pass skipped" in att and "pass exit" not in r.stdout,
              f"bash: pipeline.py not committed / modified -> pass skipped + ATTENTION ({att.strip()[:100]})")
    txt = (MLTS / "jobs" / "pipeline.slurm").read_text()
    check("#SBATCH --time=48:00:00" in txt and "#SBATCH --dependency=singleton" in txt and "--mem=2G" in txt,
          "pipeline.slurm: 48 h, singleton, 2G, 1 CPU")


@scenario
def s12_contract_units():
    """every stage with all its code present plans a well-formed submission (48 h, comment, env, guard, no nice)."""
    E = Env("s12_units")
    E.preset(done=["lists"], params=dict(base="MACE-OMOL-0", base_run="FTpilot_MACE-OMOL-0_0", pilots=["MACE-OMOL-0"],
                                         engine="A", extra_arm_App=True))
    E.lists(4)
    P = PL.Pass(E.ctx(), dry=True)
    P.state = json.loads((E.res / "PIPELINE_STATE.json").read_text())
    bad = []
    for S in P.reg:
        if S.gate is not None:
            continue
        st = P.st(S.name)
        try:
            us = S.units(P, st)
        except Exception as e:                                          # noqa: BLE001
            bad.append(f"{S.name}: {type(e).__name__}: {e}")
            continue
        for u in us:
            argv, _ = P.sbatch_argv(S, [u], [], 1.0, "abcd")
            line = " ".join(argv)
            if "--time=48:00:00" not in line or f"MLTENV={u.env}" not in line or u.part not in PL.PART or "--nice" in line \
                    or "jobs/mlt.slurm pipeline.py guard" not in line:
                bad.append(f"{S.name}/{u.key}: {line[:120]}")
            if u.akey is not None and "@I" not in u.cmd:
                bad.append(f"{S.name}/{u.key}: array without @I")
            for d in u.deps:
                if d not in {x.key for x in us}:
                    bad.append(f"{S.name}/{u.key}: dep {d} not a unit")
    check(not bad, f"all registry units well formed ({bad[:5]})")
    check(len(PL.registry()) == 42, f"registry has every spec stage + A4-App / H-App / C2-K / N-scan / D0-eval "
                                    f"({len(PL.registry())})")


@scenario
def s13_lists_status():
    """`lists` (real targets, written under the pipetest root) and `status`; a manual D0 job blocks D0 submissions."""
    E = Env("s13_lists")
    rc = PL.cmd_lists(E.ctx())
    inf = json.loads((E.mlt / "pipeline" / "lists" / "lists.json").read_text())
    f = inf["files"]
    check(rc == 0 and sum(f[f"f{k}_test"]["n"] for k in range(5)) + inf["n_fold_minus1"] == inf["n_targets"] == 9266,
          f"fold lists partition the 9,266 targets (fold -1: {inf['n_fold_minus1']})")
    check(sum(f[f"dip{d}"]["n"] for d in range(5)) == 9266, "dipole lists partition the targets")
    check(f["a3"]["n"] == 300 and all(v == 60 for v in f["a3"]["by_fold"].values()), f"A-3 list 60 per fold ({f['a3']['by_fold']})")
    rc2 = PL.cmd_lists(E.ctx())
    check(rc2 == 0 and json.loads((E.mlt / "pipeline" / "lists" / "lists.json").read_text())["files"]["a3"] == f["a3"],
          "lists are deterministic (seeded)")
    E.preset(done=["G-D1"])
    E.run()
    check(E.stage("lists")["status"] == "done" and "A-3 시범 목록" in E.dec() and "fold" in E.dec(),
          "lists done -> the A-3 split rule recorded in DECISIONS (minor)")
    E.preset(done=["lists", "G-D1", "G-D0p", "G-D0e", "G-D0f"])
    E.wj("G0_estimate_d0.json", dict(plan_fingerprint="fp1", predicted_core_h=dict(total=4000.0, d1_total=2969.0),
                                     n_g2_rule=50, stop=False, suggested_tasks=dict(D0G1=2, D0G3=1, D0G2=1)))
    E.sl.add("3000002", "g_run_d0g1_manual", "RUNNING", cpus=64)
    E.run()
    st = E.stage("G-D0")
    check(not E.calls("D0G1") and "foreign" in st.get("why", ""), f"manual D0 job -> no duplicate D0 submission ({st.get('why')})")
    check(PL.cmd_status(E.ctx()) == 0, "status prints from the state file")


# ================================================================ scenarios: one per review defect
@scenario
def s14_retry_live_sibling():
    """[S1/R2] a HALT while a sibling unit still runs, then RESOLVE retry: the running unit is not submitted again."""
    E = Env("s14_sibling")
    E.preset(done=[n for n in ALL if n not in ("Z-run", "Z-speed", "Z-rep")])
    for c in ("MACE-MH-1", "AIMNet2", "MACE-OFF24"):
        _z_json(E, c)
    E.sl.add("3000010", "z_omol", "RUNNING", cpus=8, gpus=1)
    E.run()
    halt_until(E, "Z-run", "xtb")
    check(E.stage("Z-run")["status"] == "halted", "xtb: 2 no-progress resubmits -> HALT while z_omol still RUNNING")
    n0 = len(E.calls("zero_shot.py run xtb"))
    txt = (E.res / "STOP_Z-run.md").read_text()
    (E.res / "STOP_Z-run.md").write_text("RESOLVE: retry\n" + txt)
    E.run()
    check(not E.calls("zero_shot.py run MACE-OMOL-0") and len(E.live("z_omol")) == 1,
          "RESOLVE retry: the live MACE-OMOL-0 job is carried into the new round, no second copy")
    check(len(E.calls("zero_shot.py run xtb")) == n0 + 1, "only the halted unit (xtb) is resubmitted")
    E.finish("3000010")
    _z_json(E, "MACE-OMOL-0")
    E.run()
    check(not E.calls("zero_shot.py run MACE-OMOL-0"), "the carried job ends with its output -> unit ok, never resubmitted")


@scenario
def s15_manual_job_chain():
    """[S2] a full queue plus one manual job between passes still leaves a place for the next successor."""
    E = Env("s15_chain")
    E.preset(done=[n for n in ALL if n != "lists"])
    for i in range(16):
        E.sl.add(f"21000{i:02d}", "other", "PENDING", cpus=1, reason="Dependency")
    E.sl.add("1062000", "mltp_launch", "RUNNING", cpus=1)
    E.sl.add("1062001", "mltp_launch", "PENDING", cpus=1, reason="BeginTime")
    E.run(launcher=True, job="1062000")
    n_after = len(E.sl.squeue())
    check(len(E.calls("pipeline.py lists")) == 1 and n_after == 19, f"pass fills the queue to 19 incl. both launchers ({n_after})")
    E.sl.set("1062000", "COMPLETED")                     # the launcher ends
    E.sl.add("3100000", "my_manual", "PENDING", cpus=1)  # the lead submits one job between passes
    E.sl.set("1062001", "RUNNING")                       # the successor starts: its bash queues the next one
    check(len(E.sl.squeue()) + 1 <= PL.SUBMIT_CAP, f"room for the next successor ({len(E.sl.squeue())} + 1 <= 20)")
    E.sl.add("1062002", "mltp_launch", "PENDING", cpus=1, reason="BeginTime")
    E.clock.adv(600)
    E.run(launcher=True, job="1062001")
    check(len(E.sl.squeue()) <= PL.SUBMIT_CAP and not E.calls("jobs/pipeline.slurm"),
          f"second pass: no further submissions at the cap, chain intact ({len(E.sl.squeue())})")


@scenario
def s16_fifo_slots():
    """[S3] FIFO: eligible pending <= free running slots + 2 for critical work, filler only into free slots; no nice."""
    E = Env("s16_fifo")
    E.preset(done=[n for n in ALL[:ALL.index("FT-d2") + 1]],
             params=dict(base="MACE-OMOL-0", base_run="FTpilot_MACE-OMOL-0_0", pilots=["MACE-OMOL-0"]))
    E.lists(4)
    for i in range(10):
        E.sl.add(f"22000{i:02d}", "other", "RUNNING", cpus=1)
    E.run()
    r2, fin = E.calls("finetune.py train FT_"), E.calls("FT_all_s") + E.calls("FT_LODO")
    r2 = [c for c in r2 if "--with-al" in c and "FT_all" not in c and "FT_LODO" not in c]
    check(len(r2) == 2 and not fin, f"10 running: critical FT-r2 gets the 2-task buffer, filler FT-fin nothing ({len(r2)}, {len(fin)})")
    check(not any("--nice" in c for c in E.calls()), "no --nice anywhere (priority/basic = FIFO)")
    for i in range(4):
        E.finish(f"22000{i:02d}")
    E.run()
    r2 = [c for c in E.calls("finetune.py train FT_") if "FT_all" not in c and "FT_LODO" not in c]
    fin = E.calls("FT_all_s") + E.calls("FT_LODO")
    check(len(r2) == 5 and not fin, f"4 slots free: the other 3 FT-r2 units, still no filler ({len(r2)}, {len(fin)})")
    for i in range(4, 10):
        E.finish(f"22000{i:02d}")
    E.run()
    fin = [c for c in E.calls("finetune.py train") if "FT_all" in c or "FT_LODO" in c]
    check(len(fin) == 5, f"6 more free, 5 eligible pending: filler fills the 5 free slots only ({len(fin)})")


@scenario
def s17_sbatch_errors():
    """[S4] a controller timeout never HALTs; a job created by a timed-out sbatch is adopted; fixed errors HALT."""
    E = Env("s17_sbatch")
    E.preset(done=[n for n in ALL if n != "lists"])
    E.sl.fail = "transient"
    for _ in range(4):
        E.run()
        E.clock.adv(600)
    st = E.stage("lists")
    check(st["status"] != "halted" and st.get("sbatch_err", 0) == 0, f"4 passes of 'Socket timed out' -> no HALT ({st['status']})")
    check(E.state()["intents"][-1]["status"] in ("submitting", "void"), "the intent of a timed-out sbatch is kept, then voided")
    E.sl.fail = "transient_created"
    E.run()
    E.sl.fail = None
    E.clock.adv(30)
    E.run()
    ids = [k for k in E.sl.jobs if E.sl.jobs[k]["name"] == "mltp_lists"]
    check(len(ids) == 1 and E.job_of("lists", "lists") == ids[0],
          f"a job created by a timed-out sbatch is adopted by its comment, never duplicated ({ids})")
    E2 = Env("s17b_fixed")
    E2.preset(done=[n for n in ALL if n != "lists"])
    E2.sl.fail = "error"
    for _ in range(3):
        E2.run()
        E2.clock.adv(600)
    check(E2.stage("lists")["status"] == "halted", "a fixed sbatch error 3 passes in a row -> HALT (ops)")


@scenario
def s18_resolve_crash():
    """[S5] RESOLVE: a pass that dies after applying it but before the rename never applies it twice; DECISIONS lines
    are written once even when a pass is re-run."""
    E = Env("s18_resolve")
    E.preset(done=[n for n in ALL if n not in ("Z-run", "Z-speed", "Z-rep")])
    for c in PL.Z_CANDS[:-1]:
        _z_json(E, c)
    E.run()
    halt_until(E, "Z-run", "xtb")
    txt = (E.res / "STOP_Z-run.md").read_text()
    (E.res / "STOP_Z-run.md").write_text("RESOLVE: retry\n" + txt)
    ep0 = E.stage("Z-run")["epoch"]
    orig = PL.Pass.rename_stop

    def boom(self, f, name):
        raise RuntimeError("fake crash before the rename")
    PL.Pass.rename_stop = boom
    try:
        E.run()
    except RuntimeError:
        pass
    finally:
        PL.Pass.rename_stop = orig
    check((E.res / "STOP_Z-run.md").is_file() and E.stage("Z-run")["epoch"] == ep0 + 1,
          "crash after the state write: STOP file still there, the retry recorded once")
    E.run()
    check(E.stage("Z-run")["epoch"] == ep0 + 1 and (E.res / "STOP_Z-run.resolved.md").is_file(),
          "next pass: hash already applied -> only renamed (no second epoch bump)")
    check(E.dec().count("RESOLVE: retry") == 1, "the RESOLVE line is in DECISIONS once")
    E2 = Env("s18b_dec")
    E2.preset(done=["G-D1", "lists"])
    for c in PL.Z_CANDS:
        _z_json(E2, c, mtime=E2.clock() - 100)
    E2.wj(E2.mlt / "z" / "speed.json", {t: dict(ef_ms_median=10.0) for t in PL.SPEED_DEFAULT_TAG.values()},
          mtime=E2.clock() - 100)
    E2.wj("Z_report.json", dict(ranking=["AIMNet2"], top2_for_ft_pilot=["AIMNet2"], candidates={}), mtime=E2.clock())
    oc = PL.Pass.commit

    def boom2(self):
        raise RuntimeError("fake crash before the state write")
    PL.Pass.commit = boom2
    try:
        E2.run()
    except RuntimeError:
        pass
    finally:
        PL.Pass.commit = oc
    E2.run()
    check(E2.dec().count("auto Z-rep") == 1, "a decision re-made after a crashed pass is written to DECISIONS once")


@scenario
def s19_guard():
    """[S6/P8] the job-start guard: a temp git repo; exit 4 resubmits without using a retry."""
    d = ROOT / "s19_guard"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    gi = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "init.defaultBranch=main"]
    subprocess.run(gi + ["init", "-q"], cwd=d, check=True)
    (d / "a.py").write_text("x = 1\n")
    (d / "mlt_common.py").write_text("y = 1\n")
    subprocess.run(gi + ["add", "a.py", "mlt_common.py"], cwd=d, check=True)
    subprocess.run(gi + ["commit", "-q", "-m", "t"], cwd=d, check=True)
    g = PL.Git(d, readonly=True)
    check(PL.cmd_guard(["a.py", "mlt_common.py"], code=d, git=g) == 0, "guard: committed and unchanged -> 0")
    (d / "a.py").write_text("x = 2\n")
    check(PL.cmd_guard(["a.py", "mlt_common.py"], code=d, git=g) == 4, "guard: modified since HEAD -> 4")
    (d / "a.py").write_text("x = 1\n")
    (d / "b.py").write_text("z = 1\n")
    check(PL.cmd_guard(["a.py", "b.py"], code=d, git=g) == 4, "guard: untracked file -> 4")
    check(PL.local_closure(MLTS, ["ts_geom.py"]) == ["mlt_common.py", "mlts_bridge.py", "mlts_client.py", "orca_io.py",
                                                    "ts_geom.py"], f"import closure of ts_geom.py {PL.local_closure(MLTS, ['ts_geom.py'])}")
    E = Env("s19b_exit4")
    E.preset(done=[n for n in ALL if n != "lists"])
    E.run()
    E.finish(E.job_of("lists", "lists"), "FAILED", 4)
    E.git.dirty = {"pipeline.py"}
    E.run()
    st = E.stage("lists")
    check(st["status"] == "waiting_code" and len(E.calls("pipeline.py lists")) == 1, "exit 4 + code still edited -> waits")
    E.git.dirty = set()
    E.run()
    us = E.stage("lists")["units"]["lists"]
    check(len(E.calls("pipeline.py lists")) == 2 and us["att"].get("code") == 1 and us["att"].get("infra", 0) == 0,
          f"committed -> resubmitted; exit 4 uses no retry ({us['att']})")


@scenario
def s20_a3_budget():
    """[P1/P2] A-3: the adopted engine must be within budget; the A'' arm runs (A4-App, H-App) only when chosen."""
    base = [n for n in ALL[:ALL.index("A3")]]

    def go(name, arms):
        E = Env(name, quota=1000.0)                       # the A-4 disk gate passes
        E.preset(done=base, params=dict(base="MACE-OMOL-0", base_run="FTpilot_MACE-OMOL-0_0", pilots=["MACE-OMOL-0"]))
        E.lists(4)
        E.wj("A3_report.json", dict(complete=True, arms=arms, disk_gb_per_rxn=0.01))
        E.run()
        return E
    E = go("s20_over", dict(A=dict(same_ts_frac=0.88, core_h_median=0.4, gpu_min_median=2.0),
                            App=dict(same_ts_frac=0.95, core_h_median=1.8, gpu_min_median=2.0)))
    p = E.state()["params"]
    check(p.get("engine") == "A" and not p.get("extra_arm_App") and "예산 초과" in E.dec(),
          f"A'' 0.95 at 1.8 core-h is over budget -> engine A, no extra arm ({p.get('engine')})")
    E = go("s20_app", dict(A=dict(same_ts_frac=0.80, core_h_median=0.4, gpu_min_median=2.0),
                           App=dict(same_ts_frac=0.95, core_h_median=0.9, gpu_min_median=2.0)))
    check(E.state()["params"].get("engine") == "App", "A'' in budget and best -> engine A''")
    E = go("s20_extra", dict(A=dict(same_ts_frac=0.90, core_h_median=0.4, gpu_min_median=2.0),
                             App=dict(same_ts_frac=0.91, core_h_median=0.9, gpu_min_median=2.0)))
    p = E.state()["params"]
    aj = json.loads((E.mlt / "pipeline" / "arms.json").read_text()) if (E.mlt / "pipeline" / "arms.json").is_file() else {}
    check(p.get("engine") == "A" and p.get("extra_arm_App") and aj.get("extra_arm_App") is True,
          "tie within 0.025 -> cheaper A; A'' <= 1 core-h -> extra arm, $MLT/pipeline/arms.json")
    S = E.state()
    for n in ("A4_disk", "A4", "H-early", "H"):
        S["stages"].setdefault(n, PL.new_stage_state())["status"] = "done"
    E.wj(E.res / "PIPELINE_STATE.json", S)
    E.wj("A4_report.json", dict(complete=True, folds={}))
    E.run()
    c = E.calls("refine_dft d1:")
    check(c and all("--root" in x and "/app" in x for x in c) and E.stage("A4-App")["status"] in ("queued", "running"),
          f"A4-App: refine_dft d1:<k> --root $MLT/app for every fold ({len(c)})")
    check(E.stage("H-App")["status"] == "blocked" and E.stage("R")["status"] == "blocked", "H-App and R wait for A4-App")
    E = go("s20_none", dict(A=dict(same_ts_frac=0.90, core_h_median=0.9, gpu_min_median=2.0),
                            App=dict(same_ts_frac=0.95, core_h_median=2.0, gpu_min_median=2.0)))
    check(E.stage("A3")["status"] == "halted" and "예산 안의 엔진이 없다" in (E.res / "STOP_A3.md").read_text(),
          "no arm within budget -> HALT")
    E = go("s20_skip", dict(A=dict(same_ts_frac=0.90, core_h_median=0.4, gpu_min_median=2.0)))
    S = E.state()
    for n in ("A4_disk", "A4"):
        S["stages"].setdefault(n, PL.new_stage_state())["status"] = "done"
    E.wj(E.res / "PIPELINE_STATE.json", S)
    E.run()
    check(E.stage("A4-App")["status"] == "skipped" and E.stage("H-App")["status"] == "skipped",
          "no extra arm -> A4-App and H-App skipped (R can proceed)")


@scenario
def s21_c2k():
    """[P4] C-2 on the K channel units before E."""
    E = Env("s21_c2k")
    E.preset(done=ALL[:ALL.index("K-C2C3") + 1] + ["K-C1"])
    E.wj("K_report.json", dict(complete=True, primary="C2"), mtime=E.clock() - 50)
    E.run()
    c = E.calls("tests.py c2_k")
    check(len(c) == 1 and E.stage("E")["status"] == "blocked", f"C2-K submitted, E waits ({E.stage('E')['why']})")
    E.finish(E.job_of("C2-K", "c2k"))
    E.wj("C2K_report.json", dict(complete=True, **{"pass": False}, n_units=60, n_overlap=3))
    E.run()
    check(E.stage("C2-K")["status"] == "halted" and not E.calls("eval_v2.py"), "C-2 leak in the K units -> HALT, no E")


@scenario
def s22_quota():
    """[P5] no submissions without a fresh QUOTA_FREE_GB or with < 30 GB estimated free; the estimate counts jobs that
    were running at the reading."""
    E = Env("s22_quota", quota=None)
    E.preset(done=[n for n in ALL if n != "lists"])
    E.run()
    check(not E.calls() and "QUOTA_FREE_GB" in (E.mlt / "PIPELINE_ATTENTION").read_text(), "no quota file -> no submissions")
    (E.mlt / "QUOTA_FREE_GB").write_text(f"150 {PL.iso(E.clock() - 25 * 3600)}\n")
    E.run()
    check(not E.calls(), "a 25 h old value -> no submissions")
    (E.mlt / "QUOTA_FREE_GB").write_text(f"30.1 {PL.iso(E.clock())}\n")
    E.sl.add("3200000", "g_run1", "RUNNING", cpus=64)       # adopted G task running at the reading: its estimate counts
    E.quota = None
    E.run()
    q = E.state()["quota"]
    check(not E.calls("pipeline.py lists") and q["est_free"] < 30, f"30.1 GB - a running writer < 30 -> nothing ({q})")
    (E.mlt / "QUOTA_FREE_GB").write_text(f"150 {PL.iso(E.clock())}\n")
    E.run()
    check(len(E.calls("pipeline.py lists")) == 1, "fresh 150 GB -> submits")


@scenario
def s23_speed():
    """[P6] the FT2-p 'same -> faster' tie-break reads speed_variants (one job, one GPU)."""
    E = Env("s23_speed")
    E.preset(done=ALL[:ALL.index("FT2-p")], params=dict(pilots=["MACE-OMOL-0", "AIMNet2"]))
    E.lists(4)
    _ez(E, 4)
    for c in ("MACE-OMOL-0", "AIMNet2"):
        _geom(E, "ft2", f"FTpilot_{c}_0", "label", 4, 0.005)
        _geom(E, "ft2", f"FTpilot_{c}_0", "g2", 4, 0.005)
    E.wj("Z_report.json", dict(candidates={"MACE-OMOL-0": dict(ef_ms=5.0), "AIMNet2": dict(ef_ms=50.0)},
                               speed_variants={"MACE-OMOL-0|float64|": dict(ef_ms_median=82.0),
                                               "AIMNet2||": dict(ef_ms_median=10.0)}))
    E.run()
    p = E.state()["params"]
    check(p.get("base") == "AIMNet2" and "speed.json" in E.dec(),
          f"tie on same-TS and |dd| -> faster on the same GPU (speed_variants): {p.get('base')}")


@scenario
def s25_done_before_deps():
    """[R1] a stage is never done ahead of its prerequisites; Z-run has no G-D1 edge."""
    E = Env("s25_deps")
    E.preset(done=["lists"])
    for c in ("MACE-OMOL-0", "AIMNet2", "MACE-OFF24", "xtb"):
        _z_json(E, c, mtime=E.clock() - 100)
    E.wj(E.mlt / "z" / "speed.json", {t: dict(ef_ms_median=10.0) for t in PL.SPEED_DEFAULT_TAG.values()},
         mtime=E.clock() - 100)
    E.wj("Z_report.json", dict(ranking=["MACE-OMOL-0", "AIMNet2"], top2_for_ft_pilot=["MACE-OMOL-0", "AIMNet2"]),
         mtime=E.clock())
    E.run()
    S = E.state()
    check(S["stages"]["Z-rep"]["status"] == "blocked" and not S["params"].get("pilots"),
          "Z_report written without MACE-MH-1: Z-rep not done while Z-run is not (no pilots chosen)")
    check(len(E.calls("zero_shot.py run MACE-MH-1")) == 1, "Z-run (no G-D1 edge) resubmits the missing candidate")
    E.finish(E.job_of("Z-run", "MACE-MH-1"))
    _z_json(E, "MACE-MH-1")
    E.run()
    check(len(E.calls("zero_shot.py report")) == 1, "Z-run done -> the older report is stale -> zero_shot report again")


@scenario
def s26_cancel_helper():
    """[R3] the lead cancels the G1 helper: done when the outputs are complete; STOP + auto-resolve when not yet."""
    for name, complete in (("s26_complete", True), ("s26b_incomplete", False)):
        E = Env(name, real_code=True)
        for l in REAL_SQ:
            if "|g_" in l:
                E.sl.add_line(l)
        E.run()
        E.sl.set("1061779", "CANCELLED")
        E.clock.adv(60)
        E.run()                                          # a pass while the other G tasks still run
        check(E.stage("G-D1")["status"] in ("running", "queued"),
              f"{name}: helper cancelled while the others run -> still running ({E.stage('G-D1')['status']})")
        for j in ("1061708", "1061709", "1061710"):
            E.finish(j)
        E.finish("1061711")
        E.wj("G_report.json", dict(complete=complete, stop_file=None, n_missing=0 if complete else 5,
                                   groups=dict(G1=dict(n_missing=0 if complete else 5, n_planned=27798))))
        E.run()
        g = E.stage("G-D1")
        if complete:
            check(g["status"] == "done" and not (E.res / "STOP_G-D1.md").exists(),
                  f"{name}: helper cancelled, outputs complete -> done ({g['status']}: {g['why']})")
        else:
            check(g["status"] == "cancelled" and (E.res / "STOP_G-D1.md").is_file() and not E.calls("gen_dft.py run"),
                  f"{name}: outputs incomplete -> STOP (lead decides), nothing resubmitted ({g['status']})")
            E.wj("G_report.json", dict(complete=True, stop_file=None, n_missing=0, groups={}))
            E.run()
            g = E.stage("G-D1")
            check(g["status"] == "done" and (E.res / "STOP_G-D1.resolved.md").is_file(),
                  f"{name}: outputs completed later (manual job) -> auto-resolved, done")


@scenario
def s27_halt_log():
    """[R4] passes during a HALT add no log lines."""
    E = Env("s27_log")
    upto = ALL[:ALL.index("FT-d2") + 1] + ["FT-r2"]
    E.preset(done=upto, params=dict(base="MACE-OMOL-0", base_run="FTpilot_MACE-OMOL-0_0", pilots=["MACE-OMOL-0"]))
    E.lists(4)
    for k in range(5):
        _ft_done(E, f"FT_{k}_al")
        for s in PL.STARTS:
            _geom(E, "ft2", f"FT_{k}_al", s, 4, 0.02)
    E.run()
    E.run()
    n1 = len((E.res / "PIPELINE_LOG.md").read_text().splitlines())
    for _ in range(3):
        E.clock.adv(600)
        E.run()
    n2 = len((E.res / "PIPELINE_LOG.md").read_text().splitlines())
    check(n2 == n1, f"3 passes during the HALT: log {n1} -> {n2} lines")


# ================================================================ plausible findings (verified, fixed)
@scenario
def p1_reconcile_sacct():
    """[a] sacct down: the intent is kept (never voided); without accounting comments it is matched by name + time +
    the array index set."""
    E = Env("p1_reconcile")
    E.preset(done=[n for n in ALL if n != "lists"])
    E.sl.fail = "crash_after"
    try:
        E.run()
    except RuntimeError:
        pass
    E.sl.fail = None
    jid = list(E.sl.jobs)[0]
    E.finish(jid)                                        # the job ends before the next pass: not in squeue
    E.sl.sacct_fail = True
    E.clock.adv(300)
    E.run()
    check(E.state()["intents"][-1]["status"] == "submitting" and len(E.calls()) == 1,
          "sacct failing: intent kept, no second sbatch")
    E.sl.sacct_fail = False
    E.sl.jobs[jid]["comment"] = ""                       # accounting stores no comments on this cluster
    E.run()
    it0 = E.state()["intents"][0]
    js = E.stage("lists")["units"]["lists"]["jobs"]
    check(it0["status"] == "adopted" and it0.get("job") == jid and js[0]["id"] == jid,
          "sacct back, no comments: matched by name + submit time (its outputs missing -> a new round, legit)")


@scenario
def p2_history_scan():
    """[b] a manual job that starts and ends between two passes is adopted from sacct."""
    E = Env("p2_hist")
    E.preset(done=[n for n in ALL if n not in ("Z-run", "Z-speed", "Z-rep")])
    for c in PL.Z_CANDS[:-1]:
        _z_json(E, c)
    E.sl.add("3300000", "z_xtb", "RUNNING", cpus=16)
    E.run()
    E.finish("3300000", "FAILED", 1)
    E.run()
    n = len(E.calls("zero_shot.py run xtb"))
    E.clock.adv(1200)
    E.sl.add("3300001", "z_xtb", "RUNNING", cpus=16)     # the lead's second manual run, between passes
    E.finish("3300001")
    _z_json(E, "xtb")
    E.run()
    check(E.job_of("Z-run", "xtb") in ("3300001",) or E.stage("Z-run")["status"] == "done",
          f"manual z_xtb that ran between passes adopted ({E.job_of('Z-run', 'xtb')}, {E.stage('Z-run')['status']})")
    check(len(E.calls("zero_shot.py run xtb")) == n, "no extra submission")


@scenario
def p4_lost():
    """[f] a job missing from an empty sacct answer is not LOST (accounting outage) until 12 h."""
    E = Env("p4_lost")
    E.preset(done=[n for n in ALL if n != "lists"])
    E.run()
    jid = E.job_of("lists", "lists")
    E.sl.jobs.pop(jid)
    E.clock.adv(2 * 3600)
    E.run()
    j = E.stage("lists")["units"]["lists"]["jobs"][-1]
    check(j.get("end") is None and len(E.calls()) == 1, "empty sacct after 2 h: not LOST, not resubmitted")
    E.clock.adv(11 * 3600)
    E.run()
    j = E.stage("lists")["units"]["lists"]["jobs"][0]
    check((j.get("end") or {}).get("state") == "LOST" and len(E.calls()) == 2, "after 12 h: LOST -> one infra retry")


@scenario
def p5_plan_limit():
    """[p] a test plan_d0 --limit N never counts as the D0 plan."""
    E = Env("p5_limit")
    E.preset(done=["lists", "G-D1"])
    E.wj("G_plan_d0.json", dict(fingerprint=dict(limit=5), limit=5, budget=dict(n_g2_selected=200)))
    E.run()
    check(E.stage("G-D0p")["status"] != "done" and len(E.calls("plan_d0")) == 1, "limit 5 -> not done, plan_d0 submitted")


@scenario
def p6_z_coverage():
    """[m/o] z.json 'finished' with 150 of 200 records -> not done (resubmitted); 199 of 200 -> done (1 % tolerance)."""
    E = Env("p6_zcov")
    E.preset(done=[n for n in ALL if n not in ("Z-run", "Z-speed", "Z-rep")])
    for c in PL.Z_CANDS[:-1]:
        _z_json(E, c, n=200)
    _z_json(E, "xtb", n=200, cov=150)
    E.run()
    check(len(E.calls("zero_shot.py run xtb")) == 1, "150/200 -> not finished -> xtb submitted")
    E.finish(E.job_of("Z-run", "xtb"))
    _z_json(E, "xtb", n=200, cov=199)
    E.run()
    check(E.stage("Z-run")["status"] == "done", "199/200 -> finished (1 % tolerance)")


@scenario
def p7_state_unreadable():
    """[t] an unreadable state file: continue from state.prev.json, or skip the pass (never a fresh start)."""
    E = Env("p7_state")
    E.preset(done=[n for n in ALL if n != "lists"])
    E.run()
    n = len(E.calls())
    (E.res / "PIPELINE_STATE.json").write_text("{not json")
    E.run()
    check(len(E.calls()) == n and E.stage("lists").get("status") in ("queued", "running"),
          "garbage state -> state.prev.json used, no resubmission")
    (E.res / "PIPELINE_STATE.json").write_text("{not json")
    (E.mlt / "pipeline" / "state.prev.json").unlink()
    E.run()
    check(len(E.calls()) == n and "unreadable" in (E.mlt / "PIPELINE_ATTENTION").read_text(),
          "garbage state, no copy -> pass skipped, ATTENTION")


@scenario
def p8_local_halt():
    """[i] a HALT in a report-only filler stage (FT2-r1) does not stop new critical stages."""
    E = Env("p8_local")
    E.preset(done=ALL[:ALL.index("FT-r1")], params=dict(base="MACE-OMOL-0", base_run="FTpilot_MACE-OMOL-0_0",
                                                        pilots=["MACE-OMOL-0"]))
    E.lists(4)
    S = E.state()
    S["stages"]["FT2-r1"] = dict(PL.new_stage_state(), status="halted", halt_kind="ops", why="test")
    S["halted"] = ["FT2-r1"]
    E.wj(E.res / "PIPELINE_STATE.json", S)
    E.run()
    check(len(E.calls("finetune.py train FT_1")) == 1, "FT2-r1 halted (filler, local) -> FT-r1 still starts")


@scenario
def p9_geom_partial():
    """[k] ts_geom: 1 reaction of a list never gets a record (pipeline error) -> accepted after 2 rounds without
    progress, counted as a failure in the pooled numbers."""
    E = Env("p9_partial")
    upto = ALL[:ALL.index("FT-d2") + 1] + ["FT-r2"]
    E.preset(done=upto, params=dict(base="MACE-OMOL-0", base_run="FTpilot_MACE-OMOL-0_0", pilots=["MACE-OMOL-0"]))
    E.lists(400)
    for k in range(5):
        _ft_done(E, f"FT_{k}_al")
        for s in PL.STARTS:
            _geom(E, "ft2", f"FT_{k}_al", s, 400, 0.002, cov=400 if (k, s) != (0, "label") else 399)
    for _ in range(6):
        E.run()
        for j in [l["id"] for l in E.sl.squeue()]:
            E.finish(j)
    st = E.stage("FT2-r2")
    nums = (st.get("decision") or {}).get("numbers") or {}
    check(st["status"] == "done" and st.get("partial"), f"accepted with 1 missing record ({st['status']}: {st['why'][:80]})")
    check((nums.get("label") or {}).get("n_missing_records") == 1, f"the missing reaction counts in the pooled numbers")


@scenario
def p11_g4_budget():
    """[h] G-4: the plan's predicted cost is checked before any G-4 DFT run, against gen_dft.BUDGET_CORE_H."""
    E = Env("p11_g4")
    E.preset(done=ALL[:ALL.index("G4")], params=dict(base="MACE-OMOL-0", base_run="FTpilot_MACE-OMOL-0_0"))
    E.wj("G_report.json", dict(complete=True, core_h_measured=dict(total=3300.0)))
    E.wj("G_report_d0.json", dict(complete=True, core_h_measured=dict(total=3300.0)))
    E.wj("G4_plan.json", dict(complete=True, predicted_core_h=dict(total=300.0), est={}))
    E.run()
    check(E.stage("G4")["status"] != "halted" and len(E.calls("g4_al.py g4_opt")) == 1,
          f"measured 6,600 + 300 <= 8,000 (the user's cap, not 5,000) -> g4_al.py g4_opt submitted "
          f"({E.stage('G4')['why'][:100]})")
    E2 = Env("p11b_g4_over")
    E2.preset(done=ALL[:ALL.index("G4")], params=dict(base="MACE-OMOL-0", base_run="FTpilot_MACE-OMOL-0_0"))
    E2.wj("G_report.json", dict(complete=True, core_h_measured=dict(total=5900.0)))
    E2.wj("G_report_d0.json", dict(complete=True, core_h_measured=dict(total=1900.0)))
    E2.wj("G4_plan.json", dict(complete=True, predicted_core_h=dict(total=300.0), est={}))
    E2.run()
    check(E2.stage("G4")["status"] == "halted" and not E2.calls("run G4") and not E2.calls("g4_opt"),
          "measured 7,800 + predicted 300 > 8,000 -> HALT before g4_opt / run G4")
    (E2.code / "gen_dft.py").write_text((E2.code / "gen_dft.py").read_text() + "BUDGET_CORE_H = 9000.0\n")
    check(PL.g_budget(PL.Pass(E2.ctx(), dry=True)) == 9000.0, "the cap is read from gen_dft.py (BUDGET_CORE_H line)")


@scenario
def n1_a_progress():
    """[A_G4X 3] A-3 / A-4 progress counts the folders mlts_autode writes ($MLT/a3/<arm>/ade/jobs; with A'' as engine
    also $MLT/ade_pre/jobs), so exit-2 rounds that publish folders never trip the no-progress HALT; GPU job 16 cores."""
    E = Env("n1_aprog")
    P = PL.Pass(E.ctx(), dry=True)
    P.state = PL.new_state()
    for a, n in (("A", 3), ("Ap", 2), ("App", 1)):
        for i in range(n):
            (E.mlt / "a3" / a / "ade" / "jobs" / f"T{i:05d}").mkdir(parents=True, exist_ok=True)
    check(PL.prog_a3(P) == 6, f"A-3 progress = folders of the three arms ({PL.prog_a3(P)})")
    (E.mlt / "ade_pre" / "jobs" / "T00001").mkdir(parents=True)
    (E.mlt / "ade" / "jobs" / "T00002").mkdir(parents=True)
    P.state["params"]["engine"] = "A"
    a = PL.prog_a4(P)
    P.state["params"]["engine"] = "App"
    b = PL.prog_a4(P)
    check(a == 1 and b == 2, f"A-4 progress: ade/jobs, + ade_pre/jobs when the engine is A'' ({a}, {b})")
    E2 = Env("n1b_a3_rounds")
    E2.preset(done=ALL[:ALL.index("A3")], params=dict(base="MACE-OMOL-0", base_run="FTpilot_MACE-OMOL-0_0"))
    E2.lists(4)
    E2.run()
    c = E2.calls("mlts_autode.py run a3")
    check(len(c) == 1 and "--cpus-per-task=16" in c[0] and "--mem=64G" in c[0] and "--gres=gpu:1" in c[0],
          f"A-3 GPU job: 16 cores / 64 GB (6 workers x 2 cores + server) ({c[0][:160] if c else c})")
    for r in range(3):
        for k in ("A[0]", "A[1]"):
            (E2.mlt / "a3" / "A" / "ade" / "jobs" / f"T{r}{k[2]}").mkdir(parents=True, exist_ok=True)
            E2.finish(E2.job_of("A3", k), "FAILED", 2)
        E2.run()
    check(E2.stage("A3")["status"] != "halted" and len(E2.calls("mlts_autode.py run a3")) == 4,
          f"3 exit-2 rounds with new A-2 folders -> resubmitted, no HALT ({E2.stage('A3')['status']})")


@scenario
def n2_prereg_commit():
    """[CRK 5] R: done only when PREREG_V2.md AND prereg_v2.json are committed, unchanged and json md_sha256 =
    sha256(md); the launcher's commit carries both files; K then starts."""
    import hashlib
    E = Env("n2_prereg")
    E.preset(done=ALL[:ALL.index("R")])
    names = [p.name for p in E.code.iterdir()] + ["PIPELINE_STATE.json", "PIPELINE_LOG.md", "DECISIONS.md"]
    E.git = PL.FakeGit(tracked=names)
    E.run()
    check(len(E.calls("prereg_v2.py write")) == 1, "prereg_v2.py write submitted")
    md = "# PREREG\n"
    (E.res / "PREREG_V2.md").write_text(md)
    E.wj("prereg_v2.json", dict(md_sha256="0" * 64, criteria=dict(refs=dict(primary="min"))))
    E.finish(E.job_of("R", "write"))
    E.run()
    check(E.stage("R")["status"] != "done", "md_sha256 mismatch -> R never done")
    E.wj("prereg_v2.json", dict(md_sha256=hashlib.sha256(md.encode()).hexdigest(), criteria=dict(refs=dict(primary="min"))))
    E.run()
    paths = [p for cm in E.git.commits for p in cm["paths"]]
    check(any(p.endswith("PREREG_V2.md") for p in paths) and any(p.endswith("prereg_v2.json") for p in paths),
          "commit gate: PREREG_V2.md and prereg_v2.json committed together")
    E.run()
    check(E.stage("R")["status"] == "done" and E.calls("train_k.py c1pred atoms --refs min")
          and E.calls("train_k.py plan"), f"R done after the commit -> K-C1 / K-C2C3 start ({E.stage('R')['why']})")


@scenario
def n3_kc1_cagg():
    """[CRK 4 / CR registry] K-C1 = train_k c1pred for the registered refs (-> C1_report) and the other mode
    (C1_own_report), done only when both reports carry their refs; C-agg: a failed C-1..C-6 = HALT, C-7 alone not."""
    E = Env("n3_kc1")
    E.preset(done=ALL[:ALL.index("K-C1")])
    (E.res / "PREREG_V2.md").write_text("# PREREG\n")
    E.wj("prereg_v2.json", dict(criteria=dict(refs=dict(primary="min"))), mtime=E.clock() - 10)
    E.run()
    a = E.calls("train_k.py c1pred atoms --refs min")
    check(len(a) == 1 and "--gres=gpu:1" in a[0] and len(E.calls("train_k.py c1pred xgb --refs own")) == 1,
          "K-C1: c1pred atoms (GPU) / xgb for the registered min and the descriptive own")
    for k in ("atoms", "xgb", "atoms_own", "xgb_own"):
        E.finish(E.job_of("K-C1", k))
    E.run()
    check(len(E.calls("train_k.py c1pred collect --refs min")) == 1, "collect after atoms + xgb")
    E.wj("C1_report.json", dict(refs="own"))
    E.wj("C1_own_report.json", dict(refs="own"))
    P = PL.Pass(E.ctx(), dry=True)
    P.state = E.state()
    check(not PL.done_kc1(P), "C1_report.json on refs own while min is registered -> not done")
    E.wj("C1_report.json", dict(refs="min"))
    for k in ("collect", "collect_own"):
        E.finish(E.job_of("K-C1", k))
    E.run()
    check(E.stage("K-C1")["status"] == "done", f"both reports with their refs -> K-C1 done ({E.stage('K-C1')['why']})")
    for name, secs, rc, halt in (("n3b_cagg_fail", dict(C3=False), 3, True), ("n3c_cagg_c7", dict(C7=False), 0, False)):
        E2 = Env(name)
        E2.preset(done=ALL[:ALL.index("C-agg")])
        E2.run()
        J = dict(complete=True, **{k: {"pass": secs.get(k, True), "ready": True}
                                   for k in ("C1", "C2", "C3", "C4", "C5", "C6", "C7")})
        J["stop"] = "Phase C check failed: ['C3']" if halt else None
        E2.wj("C_tests.json", J)
        E2.finish(E2.job_of("C-agg", "all"), "FAILED" if rc else "COMPLETED", rc)
        E2.run()
        st = E2.stage("C-agg")["status"]
        check((st == "halted") == halt and (st == "done") == (not halt),
              f"{name}: {'C-3 failed -> HALT' if halt else 'only C-7 false (report only) -> done'} ({st})")


@scenario
def n4_a2_smoke():
    """A2-smoke: not done while A2_smoke.json is incomplete (resubmitted); exit 3 with a complete report (C-3 wording
    failed) -> the decision rule -> HALT."""
    E = Env("n4_smoke")
    E.preset(done=ALL[:ALL.index("A2-smoke")])
    E.run()
    check(len(E.calls("mlts_autode.py smoke --model FT_0_al")) == 1, "smoke submitted")
    E.wj("A2_smoke.json", {"pass": False, "complete": False})
    E.finish(E.job_of("A2-smoke", "smoke"), "FAILED", 2)
    E.run()
    check(E.stage("A2-smoke")["status"] != "halted" and len(E.calls("mlts_autode.py smoke")) == 2,
          "incomplete smoke (exit 2) -> resubmitted, no HALT")
    E.wj("A2_smoke.json", {"pass": False, "complete": True, "inp_checks": {"n_bad": 3}})
    E.finish(E.job_of("A2-smoke", "smoke"), "FAILED", 3)
    E.run()
    check(E.stage("A2-smoke")["status"] == "halted" and "A-2 smoke" in (E.res / "STOP_A2-smoke.md").read_text(),
          "complete smoke with failed C-3 wording -> decision HALT")


@scenario
def n5_gd1_user_stop():
    """[state b, decision Nurion] the user cancelled D1 G-2 (1061710) at 317 / 400: the cancelled G2 units + the collect
    exit 2 -> G-D1 done with a decision line, no STOP_G-D1.md, G-2 never resubmitted; a G-2 short of 317 (not the
    user's stop) still stops and asks."""
    for name, n_units, done in (("n5_gd1_stop", 317, True), ("n5b_gd1_short", 200, False)):
        E = Env(name)
        for l in REAL_SQ:
            if "|g_" in l:
                E.sl.add_line(l)
        E.run()
        E.clock.adv(3600)
        for j in ("1061708", "1061709", "1061779"):
            E.finish(j)
        E.finish("1061710", "CANCELLED")
        E.finish("1061711", "FAILED", 2)
        E.wj("G_report.json", dict(complete=False, stop_file=None, core_h_measured=dict(total=3292.4),
                                   groups=dict(G1=dict(n_missing=0, n_planned=27798), G3=dict(n_missing=0, n_planned=21424),
                                               G2=dict(n_missing=400 - n_units, n_planned=400, n_units=n_units))))
        E.run()
        g = E.stage("G-D1")
        if done:
            check(g["status"] == "done" and not (E.res / "STOP_G-D1.md").exists(),
                  f"{name}: G1 / G3 complete + G-2 stopped by the user at 317 / 400 -> done ({g['status']}: {g['why']})")
            check("Nurion" in E.dec() and "317/400" in E.dec(), f"{name}: the user truncation recorded in DECISIONS")
            check(E.git.commits and any(p.endswith("G_report.json") for p in E.git.commits[-1]["paths"]),
                  f"{name}: G_report.json committed with the stage")
        else:
            check(g["status"] == "cancelled" and (E.res / "STOP_G-D1.md").is_file(),
                  f"{name}: G-2 at {n_units} < 317 -> the lead-cancel STOP, not done ({g['status']})")
        check(not E.calls("run_g2") and not E.calls("gen_dft.py run G"), f"{name}: G-2 (and nothing else) resubmitted")


@scenario
def n6_d0_manual_adopt():
    """[state a] the lead's D0 jobs are adopted, not resubmitted: d0p_plan by G-D0f; g_run_d0g1 0-1 / g_run_d0g3 0 /
    g_run_d0g2 0-2 / g_collect_d0 unit by unit; the rev / shuf helpers block G-D0 submissions while live; D0G2 keeps the
    manual 3-task slicing (@I 3) on a retry; G_report_d0 complete -> done."""
    E = Env("n6_d0adopt")
    E.preset(done=["lists", "G-D1", "G-D0p", "G-D0e", "Z-run", "Z-speed", "Z-rep", "N-scan"])
    S = E.state()
    S["stages"]["G-D0f"] = dict(PL.new_stage_state(), params=dict(n_g2=200, n_holdout=100))
    E.wj(E.res / "PIPELINE_STATE.json", S)
    E.wj("G0_estimate_d0.json", dict(plan_fingerprint=dict(id="fp1"), predicted_core_h=dict(total=5071.0, d1_total=3416.0),
                                     n_g2_rule=0, stop=True, suggested_tasks=dict(D0G1=2, D0G3=1, D0G2=0)))
    E.wj("G_plan_d0.json", dict(fingerprint=dict(id="fp1"), budget=dict(n_g2_selected=0, d0_g1_g3=2185.6)))
    E.sl.add("1062233", "d0p_plan", "PENDING", cpus=16, reason="Priority")
    for jid, nm, stt in (("1062165_0", "g_run_d0g1", "RUNNING"), ("1062165_1", "g_run_d0g1", "RUNNING"),
                         ("1062166_0", "g_run_d0g3", "RUNNING"), ("1062198", "g_run_d0g1rev", "RUNNING"),
                         ("1062205", "g_run_d0g1shuf1", "RUNNING"), ("1062234_0", "g_run_d0g2", "PENDING"),
                         ("1062234_1", "g_run_d0g2", "PENDING"), ("1062234_2", "g_run_d0g2", "PENDING")):
        E.sl.add(jid, nm, stt, cpus=64, reason="None" if stt == "RUNNING" else "Priority")
    E.sl.add("1062235", "g_collect_d0", "PENDING", cpus=4, reason="Dependency")
    E.run()
    check(E.job_of("G-D0f", "plan_d0") == "1062233" and not E.calls("plan_d0"),
          f"d0p_plan adopted by G-D0f, no second plan_d0 ({E.stage('G-D0f').get('why')})")
    keys = ("D0G1[0]", "D0G1[1]", "D0G3[0]", "D0G2[0]", "D0G2[1]", "D0G2[2]", "collect")
    check(all(E.job_of("G-D0", k) for k in keys), f"G-D0 units adopted ({[k for k in keys if not E.job_of('G-D0', k)]})")
    E.finish("1062233")
    E.wj("G_plan_d0.json", dict(fingerprint=dict(id="fp1"), budget=dict(
        n_g2_selected=200, n_holdout=100, d0_g1_g3=2185.6, cap_check=dict(total=7020.0, n_train=200, d1_total=3448.0,
                                                                         d0_g1_g3=2194.0, d0g2_train_pred=1224.0,
                                                                         d0g2_measured_on_disk=0.0, g4_reserve=150.0))))
    E.run()
    st = E.stage("G-D0")
    nn = (st.get("params") or {}).get("n") or {}
    check(E.stage("G-D0f")["status"] == "done" and not E.calls("gen_d"), f"plan written -> G-D0f done, nothing "
                                                                         f"submitted ({E.calls('gen_d')[:2]})")
    check(nn.get("D0G2@0") == 3 and nn.get("D0G1@0") == 2 and nn.get("D0G3@0") == 1,
          f"task counts = the adopted arrays (D0G1 2, D0G3 1, D0G2 3) ({nn})")
    P = PL.Pass(E.ctx(), dry=True)
    P.state = E.state()
    nums, tot = PL.d0_cost_basis(P)
    check("cap_check" in nums["source"] and abs(tot - (3448.0 + 2194.0 + 1224.0 + 150.0)) < 1e-6,
          f"the D0-plus plan's own cap check is the budget basis ({nums['source']}: {tot})")
    check("foreign" in st.get("why", ""), f"rev / shuf helpers recognised: G-D0 submits nothing while they run ({st['why']})")
    for j in ("1062198", "1062205", "1062165", "1062166", "1062234_0", "1062234_2"):
        E.finish(j)
    E.finish("1062234_1", "FAILED", 1)
    E.run()
    c = E.calls("run_g2")
    check(len(c) == 1 and "run_g2 @I 3 --d0" in c[0] and "--array=1" in c[0] and "MLTENV=d1" in c[0],
          f"D0G2[1] infra retry keeps the manual @I 3 slicing ({c})")
    E.finish(E.job_of("G-D0", "D0G2[1]"))
    E.finish("1062235")
    E.wj("G_report_d0.json", dict(complete=True, stop_file=None, core_h_measured=dict(total=3500.0)))
    E.run()
    check(E.stage("G-D0")["status"] == "done" and len(E.calls("run D0G")) == 0,
          f"G_report_d0 complete -> G-D0 done, no D0G1 / D0G3 resubmission ({E.stage('G-D0')['status']})")


def _nur_env(name):
    E = Env(name)
    up = E.d / "nur"
    mani = up / "D1_build" / "nurion"
    (mani / "index").mkdir(parents=True)
    stems = ["d1_20261006_g00_001", "d1_20261010_g01_002", "d1_20261010_g01_003"]
    for s in stems:
        (mani / f"{s}.manifest.tsv").write_text("job\n")
    (mani / "d1_extra_02_jobs_unfinished.sha256").write_text("ab  d1_extra_02_jobs_unfinished.tar.gz\n")
    (mani / "d1_extra_01_code_logs.sha256").write_text("ab  d1_extra_01_code_logs.tar.gz\n")    # no job tree: ignored
    (mani / "index" / "bundles.tsv").write_text("bundle\tn_rows\tbytes\n" + "".join(f"{s}\t1\t10\n" for s in stems))
    old = E.clock() - 3600
    for s, size in ((stems[0], 10), (stems[1], 4)):                     # g01_002: still uploading (size)
        p = up / f"{s}.tar"
        p.write_bytes(b"x" * size)
        os.utime(p, (old, old))
    (up / "d1_extra_01_code_logs.tar.gz").write_bytes(b"x")
    os.utime(up / "d1_extra_01_code_logs.tar.gz", (old, old))
    E.preset(done=["lists", "G-D1", "G-D0p", "G-D0e", "G-D0f", "G-D0", "Z-run", "Z-speed", "Z-rep"],
             params=dict(pilots=["MACE-OMOL-0"]))
    return E, up, stems


@scenario
def n7_nscan():
    """[state c, decision N-gate] N-scan: a unit per uploaded, unscanned tar (scan ::: verify, cpu1/cpu2 8c 24G; the
    user deletes copies by hand), none for a tar still uploading / not uploaded / outside the job-tree bundles / already
    scanned; a manual nur_<tar> job adopted; collect after the scans; re-arms until N_report complete; FT-dp waits for
    N_report complete or '- **N partial ok'."""
    E, up, stems = _nur_env("n7_nscan")
    E.run()
    c = E.calls("inv_nurion.py scan")
    check(len(c) == 1 and f"inv_nurion.py scan {stems[0]} --workers 8 ::: inv_nurion.py verify {stems[0]}" in c[0]
          and "-p cpu1,cpu2" in c[0] and "--cpus-per-task=8" in c[0] and "--mem=24G" in c[0] and "MLTENV=reactot" in c[0]
          and f"::: inv_nurion.py release {stems[0]} --reviewed-by user-auto-delete-2026-11-05" in c[0],
          f"one scan ::: verify ::: release (user auto-delete 11-05) for the uploaded tar only ({c})")
    ft = E.stage("FT-dp")
    check(ft["status"] == "blocked" and "N partial ok" in ft["why"] and not E.calls("finetune.py data"),
          f"FT-dp waits for the Nurion data ({ft['status']}: {ft['why'][:120]})")
    p3 = up / f"{stems[2]}.tar"                                         # uploaded + scanned by hand (inv_nurion pending)
    p3.write_bytes(b"x" * 10)
    os.utime(p3, (E.clock() - 3600, E.clock() - 3600))
    E.sl.add("3100001", "nur_g01_003", "RUNNING", cpus=8, comment=f"mltp:nur:{stems[2]}:manual")
    out = E.mlt / "dft" / "nurion"
    (out / stems[0]).mkdir(parents=True)
    (out / stems[0] / ".done").write_text("{}")                         # scanned; the copy stays until the user deletes it
    E.finish(E.job_of("N-scan", "scan/g00_001"))
    E.run()
    check(E.job_of("N-scan", "scan/g01_003") == "3100001" and len(E.calls("inv_nurion.py scan")) == 1,
          f"manual nur_g01_003 adopted as scan/g01_003, not resubmitted ({E.calls('inv_nurion.py scan')[1:]})")
    col = E.calls("inv_nurion.py collect")
    check(len(col) == 1 and "--dependency=afterany:3100001" in col[0], f"collect queued after the running scan ({col})")
    (out / stems[2]).mkdir(parents=True)
    (out / stems[2] / ".done").write_text("{}")
    p3.unlink()
    E.finish("3100001")
    E.finish(E.job_of("N-scan", "collect@2"))
    E.wj("N_report.json", dict(complete=False, complete_all=False, tars_processed=2))
    E.run()
    st = E.stage("N-scan")
    check(st["status"] == "idle" and E.stage("FT-dp")["status"] == "blocked" and len(E.calls("inv_nurion.py scan")) == 1,
          f"uploaded tars processed (g00_001 copy not deleted yet: no rescan), more expected -> N-scan re-armed "
          f"(idle), FT-dp still waits ({st['status']})")
    (E.res / "DECISIONS.md").write_text(E.dec() + "- **N partial ok (11-05 09:00)**: FT starts with the tars so far.\n")
    E.run()
    check(len(E.calls("finetune.py data FTpilot_MACE-OMOL-0_0")) == 1,
          f"the lead's 'N partial ok' line -> FT-dp submitted ({E.stage('FT-dp')['why'][:120]})")
    E2, up2, stems2 = _nur_env("n7b_nscan_final")
    for s in stems2:
        (E2.mlt / "dft" / "nurion" / s).mkdir(parents=True)
        (E2.mlt / "dft" / "nurion" / s / ".done").write_text("{}")
    for p in up2.glob("d1_2026*.tar"):
        p.unlink()
    E2.wj("N_report.json", dict(complete=True, complete_all=False, tars_processed=3))
    E2.run()
    check(E2.stage("N-scan")["status"] == "done" and len(E2.calls("finetune.py data")) == 1,
          f"every result tar processed (complete; the extras optional) -> N-scan done, FT-dp starts "
          f"({E2.stage('N-scan')['status']})")


@scenario
def n9_nscan_halt_d0f_retry():
    """N-scan HALT is local (a scan / verify exit 3 stops N-scan and FT data, not the other stages); the adopted
    d0p_plan ending exit 2 is resubmitted by G-D0f with exactly --n-near 2 --n-g2 200 --n-holdout 100."""
    E, up, stems = _nur_env("n9_nscan_halt")
    E.run()
    E.finish(E.job_of("N-scan", "scan/g00_001"), "FAILED", 3)
    E.run()
    st = E.stage("N-scan")
    check(st["status"] == "halted" and (E.res / "STOP_N-scan.md").is_file(), f"scan exit 3 -> N-scan HALT ({st['status']})")
    S = E.state()
    S["stages"]["lists"]["status"] = ""                                 # a stage not started yet during the HALT
    E.wj(E.res / "PIPELINE_STATE.json", S)
    E.run()
    check(len(E.calls("pipeline.py lists")) == 1 and E.stage("FT-dp")["status"] == "blocked",
          f"local HALT: lists still submitted, FT-dp still waits for the Nurion data ({E.calls('pipeline.py lists')[:1]})")
    E2 = Env("n9b_d0f_retry")
    E2.preset(done=["lists", "G-D1", "G-D0p", "G-D0e", "Z-run", "Z-speed", "Z-rep", "N-scan"])
    S = E2.state()
    S["stages"]["G-D0f"] = dict(PL.new_stage_state(), params=dict(n_g2=200, n_holdout=100))
    E2.wj(E2.res / "PIPELINE_STATE.json", S)
    E2.wj("G_plan_d0.json", dict(fingerprint=dict(id="fp1"), budget=dict(n_g2_selected=0)))
    E2.sl.add("1062233", "d0p_plan", "RUNNING", cpus=16)
    E2.run()
    E2.finish("1062233", "FAILED", 2)
    E2.run()
    c = E2.calls("gen_d0.py plan_d0")
    check(len(c) == 1 and c[0].endswith("gen_d0.py plan_d0 --n-near 2 --n-g2 200 --n-holdout 100"),
          f"adopted d0p_plan exit 2 -> resubmitted with the user's options ({c})")


@scenario
def n8_d0eval():
    """[state e] D0-eval (D0-plus / D0-gauss, report only): after FT-fin, ts_geom d0_eval run @I 3 (GPU) -> d0_eval
    report; done on a complete D0_eval_report.json; S waits for it."""
    E = Env("n8_d0eval")
    E.preset(done=[n for n in ALL if n not in ("D0-eval", "S")])
    E.run()
    c = E.calls("ts_geom.py d0_eval run")
    check(len(c) == 1 and "--array=0,1,2" in c[0] and "d0_eval run @I 3" in c[0] and "--gres=gpu:1" in c[0]
          and "MLTENV=mlts" in c[0], f"d0_eval run array of 3 on a GPU ({c})")
    check(not E.calls("summary_v2.py"), "S waits for D0-eval")
    for i in range(3):
        E.finish(E.job_of("D0-eval", f"run[{i}]"))
    E.run()
    r = E.calls("ts_geom.py d0_eval report")
    check(len(r) == 1 and "MLTENV=mlts" in r[0], f"d0_eval report after the runs ({r})")
    E.wj("D0_eval_report.json", dict(complete=True, n_units=100))
    E.finish(E.job_of("D0-eval", "report"))
    E.run()
    check(E.stage("D0-eval")["status"] == "done" and len(E.calls("summary_v2.py write")) == 1,
          f"D0-eval done -> S starts ({E.stage('D0-eval')['status']})")


@scenario
def p12_index_lock():
    """[e] a .git/index.lock older than 10 min is reported in PIPELINE_ATTENTION."""
    E = Env("p12_lock")
    E.preset(done=[n for n in ALL if n != "lists"])
    E.git.locked, E.git.lock_age = True, 1800.0
    (E.res / "STOP_x.md").write_text("x")
    S = E.state()
    S["commit_pending"] = [["lists", "done"]]
    E.wj(E.res / "PIPELINE_STATE.json", S)
    E.run()
    check("index.lock" in (E.mlt / "PIPELINE_ATTENTION").read_text() and not E.git.commits, "old index.lock -> ATTENTION")


@scenario
def p13_git_remote():
    """informational: can a compute node reach origin (the launcher pushes from a job)? git ls-remote, read-only."""
    try:
        r = subprocess.run(["git", "ls-remote", "--heads", "origin", "mlts"], cwd=MLTS, capture_output=True, text=True,
                           timeout=60, env=dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_SSH_COMMAND="ssh -o BatchMode=yes"))
        info(f"git ls-remote origin mlts from {os.uname().nodename}: rc {r.returncode} "
             f"{(r.stdout or r.stderr).strip()[:160]}")
    except Exception as e:                                              # noqa: BLE001
        info(f"git ls-remote failed: {type(e).__name__}: {e}")
    check(True, "informational")


# ================================================================ FT2-p rules (user 11-05 (2) E/Z, (3) recalc 5)
def _ft2p_env(name, n, ez_with=None, pilots=("AIMNet2",)):
    E = Env(name)
    E.preset(done=ALL[:ALL.index("FT2-p")], params=dict(pilots=list(pilots)))
    E.lists(n)
    if ez_with is not None:
        _ez(E, n, ez_with)
    return E


def _ft2p_numbers(E):
    return (E.stage("FT2-p").get("decision") or {}).get("numbers") or {}


@scenario
def e1_ft2p_ez_split():
    """[user 11-05 (2)] FT2-p reports FT-2 (a) / (b) for the E/Z 'with' / 'without sibling' groups; groups differing by
    > 0.002 A median |dd| (A) or > 0.05 same-TS (E) -> the fold-0 gate on 'without' only (decision line + DEVIATIONS
    note); within the thresholds -> all reactions (B pass, C HALT); no E/Z list -> FT2-p waits (D) / HALT at the
    decision (D2)."""
    n, run = 20, "FTpilot_AIMNet2_0"
    with_ids = set(range(12))

    def go(name, dd_with, dd_without, cls_with=None):
        E = _ft2p_env(name, n, with_ids)
        for s in PL.STARTS:
            _geom_rows(E, run, s, [_row(i, (cls_with or {}).get(i), dd=dd_with if i in with_ids else dd_without)
                                   for i in range(n)])
        return E
    E = go("e1a_ez_split_dd", 0.012, 0.004)
    E.run()
    st, nums = E.stage("FT2-p"), _ft2p_numbers(E)
    ez = nums.get("ez") or {}
    dev = (E.res / "DEVIATIONS.md").read_text() if (E.res / "DEVIATIONS.md").is_file() else ""
    check(st["status"] == "done" and nums.get("gate_set") == "without_sibling" and ez.get("split") is True,
          f"A: median |dd| 0.012 (with) vs 0.004 (without) > 0.002 A -> gate on 'without' only, passes "
          f"({st['status']}, {nums.get('gate_set')})")
    check(nums.get("label", {}).get("dd_median_A") == 0.012 and nums.get("pass_", {}).get("dd_median_A") is True,
          "A: all reactions would fail (median 0.012 > 0.010); the 'without' group (0.004) passes")
    check((ez.get("label") or {}).get("with", {}).get("n_attempted") == 12
          and (ez.get("label") or {}).get("without", {}).get("n_attempted") == 8
          and (ez.get("g2") or {}).get("with", {}).get("dd_median_A") == 0.012,
          "A: (a) label and (b) G2 numbers reported per group (12 with / 8 without)")
    check("형제 없는" in E.dec() and "형제 없는" in dev and "auto FT2-p" in dev and "<!-- mltp:" in dev,
          "A: the decision line and a DEVIATIONS.md note say the gate used 'without sibling' only")
    check(any(p.endswith("DEVIATIONS.md") for c in E.git.commits for p in c["paths"]), "A: DEVIATIONS.md committed")
    check(E.state()["params"].get("base") == "AIMNet2", "A: base MLIP AIMNet2")
    E.run()
    dev2 = (E.res / "DEVIATIONS.md").read_text()
    check(dev2.count("auto FT2-p") == 1, "A: DEVIATIONS note written once")
    E = go("e1b_ez_same", 0.005, 0.004)
    E.run()
    st, nums = E.stage("FT2-p"), _ft2p_numbers(E)
    check(st["status"] == "done" and nums.get("gate_set") == "all" and (nums.get("ez") or {}).get("split") is False
          and not (E.res / "DEVIATIONS.md").exists(),
          f"B: 0.005 vs 0.004 (diff 0.001 <= 0.002) -> gate on all reactions, no DEVIATIONS note ({st['status']})")
    check("관문은 fold 0 시험 전체" in E.dec(), "B: decision line says the gate used all fold-0 test reactions")
    E = go("e1c_ez_same_fail", 0.020, 0.019)
    E.run()
    st = E.stage("FT2-p")
    check(st["status"] == "halted" and (E.res / "STOP_FT2-p.md").is_file() and _ft2p_numbers(E).get("gate_set") == "all",
          f"C: within the thresholds and median 0.02 > 0.010 -> HALT on all reactions ({st['status']})")
    check(not E.calls("finetune.py data FT_1"), "C: FT-d1 (folds 1-4) not started")
    E = _ft2p_env("e1d_ez_missing", n)
    E.run()
    st = E.stage("FT2-p")
    check(st["status"] == "blocked" and "E/Z" in st["why"] and not E.calls("ts_geom.py"),
          f"D: no E/Z list -> FT2-p waits, nothing submitted ({st['status']}: {st['why'][:90]})")
    E = _ft2p_env("e1d2_ez_missing_done", n)
    for s in PL.STARTS:
        _geom_rows(E, run, s, [_row(i) for i in range(n)])
    E.run()
    st = E.stage("FT2-p")
    check(st["status"] == "halted" and "ez_groups.py" in (E.res / "STOP_FT2-p.md").read_text(),
          f"D2: outputs complete but no E/Z list -> decision HALT naming ez_groups.py ({st['status']})")
    E = go("e1e_ez_split_same", 0.004, 0.004, cls_with={0: "other_saddle", 1: "other_saddle"})
    for s in PL.STARTS:                                 # both re-measured already, still other saddles
        _geom_rows(E, run, f"{s}{PL.RECALC_SUFFIX}", [_row(i, "other_saddle") for i in (0, 1)])
    E.run()
    st, nums = E.stage("FT2-p"), _ft2p_numbers(E)
    ez = nums.get("ez") or {}
    check(not E.calls("--only-failed") and st["status"] == "done" and nums.get("gate_set") == "without_sibling"
          and abs((ez.get("diff_same") or 0) - (1.0 - 10 / 12)) < 1e-9,
          f"E: re-measure already complete (no recalc job); same TS 10/12 (with) vs 8/8 -> diff 0.167 > 0.05 -> "
          f"'without' only ({st['status']}, {nums.get('gate_set')}, {ez.get('diff_same')})")


@scenario
def e2_ft2p_recalc():
    """[user 11-05 (3)] FT2-p: first-pass failures classified (i) not_converged / (ii) n_imag / (iii) other_saddle per
    start; after the first pass the failed reactions are run again with `ts_geom.py run ... --recalc 5 --only-failed
    <out_dir>` -> <start>_recalc5; the gate uses the re-measured record per reaction (fails only if it fails again)."""
    n, run = 40, "FTpilot_AIMNet2_0"
    cls = {**{i: "not_converged" for i in range(4)}, **{i: "n_imag" for i in range(4, 7)},
           **{i: "other_saddle" for i in range(7, 10)}}
    E = _ft2p_env("e2_recalc", n, ())
    for s in PL.STARTS:
        _geom_rows(E, run, s, [_row(i, cls.get(i)) for i in range(n)])
    E.run()
    c = E.calls("--only-failed")
    od = {s: str(E.mlt / "ft2" / run / s) for s in PL.STARTS}
    check(len(c) == 2 and all("--recalc 5 --only-failed" in x and "--gres=gpu:1" in x and "MLTENV=mlts" in x for x in c)
          and any(f"{od['label']}_recalc5 @I 1 --recalc 5 --only-failed {od['label']}" in x for x in c)
          and any(f"{od['g2']}_recalc5 @I 1 --recalc 5 --only-failed {od['g2']}" in x for x in c),
          f"recalc 5 re-measure submitted for both starts after the first pass ({[x[-160:] for x in c]})")
    check(E.stage("FT2-p")["status"] in ("queued", "running") and not E.stage("FT2-p").get("decision"),
          "no decision before the re-measure")
    check(any("comment=mltp:FT2-p:FTpilot_AIMNet2_0/label_recalc5[0]" in x for x in c), "unit key <run>/<start>_recalc5")
    for s in PL.STARTS:
        _geom_rows(E, run, f"{s}{PL.RECALC_SUFFIX}", [_row(i, {0: "not_converged", 4: "n_imag"}.get(i))
                                                      for i in range(10)])
    for l in E.live():
        E.finish(l["id"])
    E.run()
    st, nums = E.stage("FT2-p"), _ft2p_numbers(E)
    fc = (nums.get("fail_classes") or {}).get("label") or {}
    first, aft = fc.get("first_pass") or {}, fc.get("after_recalc") or {}
    check(st["status"] == "done", f"re-measure complete -> FT2-p done ({st['status']}: {st['why'][:100]})")
    check((first.get("not_converged"), first.get("n_imag"), first.get("other_saddle")) == (4, 3, 3)
          and (aft.get("not_converged"), aft.get("n_imag"), aft.get("other_saddle")) == (1, 1, 0)
          and fc.get("n_recovered") == 8 and fc.get("n_remeasured") == 10,
          f"classes first pass 4 / 3 / 3 -> after recalc 5: 1 / 1 / 0, 8 recovered ({first}, {aft})")
    check(((nums.get("fail_classes") or {}).get("g2") or {}).get("n_failed_after") == 2, "classes per start (g2 too)")
    check((nums.get("label_first_pass") or {}).get("same_ts_frac") == 33 / 40
          and (nums.get("label") or {}).get("same_ts_frac") == 39 / 40,
          "the gate uses the merged records: same TS 0.825 (first pass, would fail) -> 0.975 (passes)")
    check("recalc 5" in E.dec() and "재측정" in E.dec() and "not_converged" in E.dec(),
          "the decision line records the classes and the re-measure rule")
    n_calls = len(E.calls("ts_geom.py"))
    E.run()
    check(len(E.calls("ts_geom.py")) == n_calls, "nothing resubmitted after the decision")
    # the first pass is not final while a first-pass task is in squeue (>= 99.5 % recorded)
    P = PL.Pass(E.ctx(), dry=True)
    P.state = E.state()
    P.live = {"777_0": {}}
    st0 = dict(units={f"{run}/label[0]": dict(jobs=[dict(id="777_0")])})
    d = E.mlt / "ft2" / run / "label"
    E.wj(d / "summary.json", dict(all=dict(status_counts=dict(ok=399))))
    check(not PL.first_pass_final(P, st0, d, 400, f"{run}/label") and
          PL.first_pass_final(P, dict(units={}), d, 400, f"{run}/label") and
          not PL.first_pass_final(P, dict(units={}), d, 410, f"{run}/label"),
          "first pass final: 399/400 and no live first-pass task; not with a live task; not below 99.5 %")


@scenario
def e3_ts_geom_rules():
    """[user 11-05 (3)] ts_geom: fail_class = pipeline.fail_class (records and the geom.csv summarize writes);
    --only-failed selection + slicing; --recalc parsing (1..30, analytic); summary fail_classes and by_ez_sibling."""
    import ts_geom as TG
    d = ROOT / "e3_ts_geom"
    if d.exists():
        shutil.rmtree(d)
    prev = d / "prev"
    (prev / "recs").mkdir(parents=True)
    base = dict(model="FTpilot_AIMNet2_0", start="label", dipole="nitrone", fold_T1=0, wall_s=10.0, n_cycles=12,
                rmsd_heavy=0.01, tight="tight", recalc="analytic:30")
    cases = [(dict(status="ok", converged=True, n_imag=1, same_ts=True, dd_max=0.004), None),
             (dict(status="ok", converged=True, n_imag=1, same_ts=True, dd_max=0.030), None),
             (dict(status="ok", converged=True, n_imag=1, same_ts=True, dd_max=0.031), "other_saddle"),
             (dict(status="ok", converged=True, n_imag=1, same_ts=False, dd_max=0.01), "other_saddle"),
             (dict(status="ok", converged=True, n_imag=1, same_ts=None, dd_max=None), "other_saddle"),
             (dict(status="ok", converged=True, n_imag=2, same_ts=True, dd_max=0.004), "n_imag"),
             (dict(status="ok", converged=True, n_imag=0, same_ts=True, dd_max=0.004), "n_imag"),
             (dict(status="ok", converged=True, n_imag=None, same_ts=True, dd_max=0.004), "n_imag"),
             (dict(status="not_converged", converged=False, n_imag=1, same_ts=True, dd_max=0.004), "not_converged"),
             (dict(status="orca_error", converged=False, n_imag=None, same_ts=False, dd_max=0.5), "not_converged"),
             (dict(status="no_start"), None), (dict(status="not_covered"), None)]
    for k, (r, _c) in enumerate(cases):
        rec = dict(base, rxn_id=1000000 + k, **r)
        (prev / "recs" / f"{1000000 + k}.json").write_text(json.dumps(rec))
    want = {1000000 + k: c for k, (_r, c) in enumerate(cases) if c}
    got = TG.failed_ids(prev)
    check(got == want, f"ts_geom.failed_ids on records ({got})")
    ezp = d / "ez.csv"
    ezp.write_text("rxn_id,ez_sibling_in_train\n" + "".join(f"{1000000 + k},{k < 4}\n" for k in range(len(cases))))
    S = TG.summarize(prev, ez_list=ezp)
    rows = PL.read_geom_csv(prev / "geom.csv")
    pl = {int(r["rxn_id"]): PL.fail_class(r) for r in rows}
    check({k: v for k, v in pl.items() if v} == want, f"pipeline.fail_class on the geom.csv summarize wrote ({pl})")
    fc = S.get("fail_classes") or {}
    check((fc.get("n_attempted"), fc.get("n_pass"), fc.get("not_converged"), fc.get("n_imag"), fc.get("other_saddle"))
          == (10, 2, 2, 3, 3), f"summary fail_classes ({fc})")
    ezt = S.get("by_ez_sibling") or {}
    check(ezt.get("with", {}).get("n_attempted") == 4 and ezt.get("without", {}).get("n_attempted") == 6
          and ezt.get("n_listed") == len(cases), f"summary by_ez_sibling with 4 / without 6 attempted ({ezt.keys()})")
    lst = [1000000 + k for k in range(len(cases))]
    r0, f0 = TG.select_rxns(lst, 0, 2, str(prev))
    r1, _f1 = TG.select_rxns(lst, 1, 2, str(prev))
    fl = [x for x in lst if x in want]
    check(sorted(r0 + r1) == fl and r0 == fl[0::2] and f0 == want, f"--only-failed: failed subset sliced k % n ({r0}, {r1})")
    o = TG.parse_run_argv(["run", "FTpilot_AIMNet2_0", "l.csv", "label", "out", "@I", "3", "--recalc", "5",
                           "--only-failed", str(prev)])
    check(o and o["recalc"] == "analytic:5" and o["only_failed"] == str(prev) and o["n"] == 3 and o["i"] == "@I",
          f"run argv with --recalc 5 --only-failed ({o})")
    check(TG.parse_run_argv(["run", "x", "l.csv", "label", "out"])["recalc"] is None, "no --recalc: default (analytic:30)")
    bad = []
    for v in ("31", "0", "numeric:5", "x"):
        try:
            TG.recalc_spec(v)
            bad.append(v)
        except SystemExit:
            pass
    check(not bad and TG.recalc_spec("analytic:30") == "analytic:30", f"--recalc outside 1..30 / not analytic -> STOP {bad}")


@scenario
def e4_geom_tasks():
    """[lead 11-05] ts_geom arrays: n = n_tasks(n_rxn, max(Z wall, 120 s), 14, 4, wall 2 h): f0_test 1,874 reactions
    with AIMNet2's 4.3 s Z wall -> more than one task per start (3), both starts submitted in one pass."""
    n = 1874
    E = _ft2p_env("e4_tasks", n, ())
    E.wj("Z_report.json", dict(candidates={"AIMNet2": dict(geometry=dict(wall_median_s=4.3))}))
    E.run()
    c = E.calls("ts_geom.py run FTpilot_AIMNet2_0")
    k = PL.n_tasks(n, PL.GEOM_UNIT_MIN_S / 3600.0, 14, 4, wall=PL.GEOM_WALL_H)
    arr_ = ",".join(str(i) for i in range(k))
    check(k >= 3 and len(c) == 2 and all(f"--array={arr_}" in x and f"@I {k}" in x for x in c)
          and not E.calls("--only-failed"),
          f"f0_test 1,874: {k} tasks per start (formula), label + g2 submitted, no re-measure yet ({[x[:90] for x in c]})")
    check(PL.n_tasks(400, PL.geom_unit_h(PL.Pass(E.ctx(), dry=True), "AIMNet2"), 14, 4, wall=PL.GEOM_WALL_H) == 1
          and PL.n_tasks(10000, 300 / 3600, 14, 4, wall=PL.GEOM_WALL_H) == 4, "small lists 1 task, nmax 4")


# ================================================================ main
def main(argv):
    ROOT.mkdir(parents=True, exist_ok=True)
    want = set(argv)
    fns = [f for n, f in sorted(globals().items()) if getattr(f, "_scenario", False) and (not want or n in want)]
    for f in fns:
        RESULTS.append(dict(name=f.__name__, ok=True, checks=[]))
        print(f"== {f.__name__}", flush=True)
        t = time.time()
        try:
            f()
        except Exception as e:                                          # noqa: BLE001
            RESULTS[-1]["ok"] = False
            RESULTS[-1]["error"] = f"{type(e).__name__}: {e}"
            traceback.print_exc()
        r = RESULTS[-1]
        n_ok = sum(c["ok"] for c in r["checks"])
        print(f"   {'PASS' if r['ok'] else 'FAIL'} {n_ok}/{len(r['checks'])} checks ({time.time() - t:.1f} s)", flush=True)
    tot = sum(len(r["checks"]) for r in RESULTS)
    ok = sum(c["ok"] for r in RESULTS for c in r["checks"])
    summary = dict(time=PL.iso(time.time()), root=str(ROOT), scenarios=len(RESULTS),
                   scenarios_pass=sum(r["ok"] for r in RESULTS), checks=tot, checks_pass=ok, results=RESULTS)
    (ROOT / "results.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(f"TOTAL scenarios {summary['scenarios_pass']}/{summary['scenarios']} pass, checks {ok}/{tot}", flush=True)
    return 0 if summary["scenarios_pass"] == summary["scenarios"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
