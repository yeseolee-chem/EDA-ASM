#!/usr/bin/env python3
r"""g2_deploy.py — DEPLOY-MODEL 14-2: TS search for new reactions (SMILES -> autodE 1.4.5 + XTB2 ALPB(water), once per
reaction) and the label-free finalisation, with every write under one root. D1 run env (python 3.9.5 venv, autodE
1.4.5, ORCA 6.1.1, xtb 6.7.1, OpenMPI 4.1.8f) = jobs/cpu_d1.slurm. Exit 0 done, 2 not ready / unfinished, 3 STOP.

  python g2_deploy.py one <job_id> <cores> --manifest <manifest.csv> --out <root>
  python g2_deploy.py pool <list.csv> <n_parallel> <cores> --manifest <manifest.csv> --out <root> [--slice i n]
  python g2_deploy.py i3-prep --out <root>       I-3 inputs: 5 P_main para reference pairs, one per dipole kind (10 D1
                                                 reactions) -> <root>/i3_manifest.csv, i3_list.csv, i3_pairs.csv

Flow per reaction (= analysis/d1_g2/g2_autode.py cmd_one 392-423 without the label list / G2_FINAL; spec 14-2):
  node-local $TMPDIR/d1dep_<pid> work folder (config_g2.yaml with scratch = it) -> g2_autode.stage_ts (137-162; run_ts
  run_autode + analyse, D1 retry_infra_once: one infrastructure retry) -> stage_stereo (= g2_autode.stage_stereo 165-171
  with the --manifest file handed to d1_run/stereo_check.py instead of the D1 manifest) -> g2_autode.stage_ref (174-211)
  -> finalize_lf.finalize_lf(job_row, work folder, <root>/g2lf_final, cores) (analysis/deploy/finalize_lf.py) ->
  g2_autode.keep_small(work folder, <root>/jobs/<job_id>) (373-389). One attempt per reaction; a reaction whose
  <root>/g2lf_final/<rxn key> already has .done / .fail_* is skipped (idempotent).
Paths: g2_common reads D1G2_SCRATCH at import and fixes G2 (its write guard assert_path and the template folder
  G2/ts_templates), so D1G2_SCRATCH = <root> is set BEFORE g2_common / g2_autode / finalize_lf are imported and
  G.G2 == <root> is asserted. <root> must lie under $DEP (dep_common write guard; finalize_lf writes there) and is never
  the D1-G2 root. TS templates: on first use $G2/ts_templates (read only) is copied to <root>/ts_templates (atomic
  rename; default of the spec, the deployment spec may change it); autodE then reads / adds templates there only.
Manifest columns (D1 manifest format, spec 14-2): job_id, kind (= D1), ts_id, rxn_smiles, mapped_smiles (required, not
  empty), product_smiles, dipole_smiles, dipolarophile_smiles, charge, mult; meta_rxn_id or rxn_id (optional) names the
  output folder (finalize_lf.rxn_key; else the job_id).
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dep_common as C  # noqa: E402            (stdlib only: safe before D1G2_SCRATCH is set)

REQ_COLS = ("job_id", "kind", "ts_id", "rxn_smiles", "mapped_smiles", "product_smiles", "dipole_smiles",
            "dipolarophile_smiles", "charge", "mult")
STOP_AFTER_H = float(os.environ.get("DEP_STOP_AFTER_H", "44"))
T0 = float(os.environ.get("DEP_JOB_T0", time.time()))
_MODS = {}


def _abs(p):
    return Path(os.path.abspath(str(p)))


def setup(root):
    """D1G2_SCRATCH = root, then import g2_common / g2_autode / finalize_lf; assert G.G2 == root (spec 14-2)."""
    if _MODS:
        if _abs(_MODS["G"].G2) != _abs(root):
            C.stop(f"g2_common already imported with G2 = {_MODS['G'].G2}, not {root}")
        return _MODS
    root = _abs(root)
    C.assert_write_ok(root / "probe")
    if root == _abs(C.G2) or _abs(C.G2) in root.parents:
        C.stop(f"--out {root} is the D1-G2 root or inside it (read only)")
    if "g2_common" in sys.modules:
        C.stop("g2_common was imported before D1G2_SCRATCH was set")
    os.environ["D1G2_SCRATCH"] = str(root)
    sys.path.insert(0, str(C.REPO / "analysis" / "d1_g2"))
    import g2_common as G  # noqa: E402
    if _abs(G.G2) != root:
        C.stop(f"g2_common.G2 = {G.G2} != --out {root}")
    import g2_autode as GA  # noqa: E402
    import finalize_lf as FL  # noqa: E402
    if GA.G is not G or FL.GA is not GA:
        C.stop("g2_autode / finalize_lf did not bind the g2_common module of this root")
    _MODS.update(G=G, GA=GA, FL=FL, root=root)
    return _MODS


def copy_templates(root):
    """$G2/ts_templates -> <root>/ts_templates once (copy to a temporary folder, then an atomic rename)."""
    dst = root / "ts_templates"
    if dst.exists():
        return dict(status="exists", path=str(dst))
    src = C.G2 / "ts_templates"
    if not src.is_dir():
        C.not_ready(f"{src} missing")
    tmp = root / f".ts_templates.{os.getpid()}.tmp"
    C.assert_write_ok(tmp)
    root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, tmp)
    try:
        os.rename(tmp, dst)
        st = "copied"
    except OSError:                                    # another process published it first
        shutil.rmtree(tmp, ignore_errors=True)
        st = "exists"
    return dict(status=st, path=str(dst), source=str(src), n_files=sum(1 for p in dst.rglob("*") if p.is_file()))


def check_manifest(path):
    """the manifest must have the D1 manifest columns, unique job ids, kind D1 and a non-empty mapped_smiles."""
    import pandas as pd
    m = pd.read_csv(path, keep_default_na=False, dtype=str)
    miss = [c for c in REQ_COLS if c not in m.columns]
    if miss:
        C.stop(f"{path}: columns missing {miss}")
    if m.job_id.duplicated().any():
        C.stop(f"{path}: duplicated job_id")
    bad = m.loc[(m.kind != "D1") | (m.mapped_smiles.str.strip() == ""), "job_id"].tolist()
    if bad:
        C.stop(f"{path}: rows with kind != D1 or an empty mapped_smiles: {bad[:10]}")
    return m


def stage_stereo(GA, cfg_path, job, jd, manifest):
    """= g2_autode.stage_stereo (165-171) with the given manifest (not the fixed D1 manifest)."""
    if GA.pc.done(jd, "stereo"):
        return True
    r = subprocess.run([sys.executable, str(GA.G.D1_RUN / "stereo_check.py"), "--job", str(job["job_id"]), "--config",
                        str(cfg_path), "--manifest", str(manifest)], cwd=GA.G.PILOT, capture_output=True, text=True)
    (jd / "stereo.log").write_text(r.stdout + r.stderr)
    return GA.pc.done(jd, "stereo")


def cmd_one(T, cores, manifest, root):
    import yaml
    M = setup(root)
    G, GA, FL, root = M["G"], M["GA"], M["FL"], M["root"]
    copy_templates(root)
    manifest = _abs(manifest)
    check_manifest(manifest)
    cores = int(cores)
    job = GA.pc.load_job(T, manifest)
    fin, jobs = root / "g2lf_final", root / "jobs"
    rk = FL.rxn_key(job)
    if FL.marker(fin / str(rk)) is not None:
        print(f"{T}: final exists ({FL.marker(fin / str(rk))}), skip", flush=True)
        return 0
    os.environ["SLURM_CPUS_PER_TASK"] = str(cores)               # pilot_common.n_cores() -> autodE n_cores
    os.environ["OMP_NUM_THREADS"] = str(cores)
    local = _abs(os.environ.get("TMPDIR", "/tmp")) / f"d1dep_{os.getpid()}"
    cfg = yaml.safe_load(Path(GA.CONFIG).read_text())
    cfg["scratch"] = str(local)
    cfg_path = local / "config_g2_local.yaml"
    local.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(cfg))
    jd = GA.pc.job_dir(cfg, T)
    times, st = {}, None
    try:
        t = time.time(); ok = GA.stage_ts(cfg, job, jd, cores); times["ts"] = time.time() - t
        if ok:
            t = time.time(); ok = stage_stereo(GA, cfg_path, job, jd, manifest); times["stereo"] = time.time() - t
        if ok:
            t = time.time(); ok = GA.stage_ref(cfg, job, jd); times["ref"] = time.time() - t
        os.chdir(local)
        t = time.time()
        st = FL.finalize_lf(job, jd, fin, cores)
        times["finalize"] = time.time() - t
        (jd / "deploy_times.json").write_text(json.dumps(dict(
            job_id=T, rxn_key=str(rk), status=st, cores=cores, wall_s=times,
            core_h={k: v * cores / 3600 for k, v in times.items()}, manifest=str(manifest),
            manifest_sha256=C.sha256(manifest), root=str(root), code="analysis/deploy/g2_deploy.py",
            code_sha256=C.sha256(Path(__file__).resolve())), indent=1))
        GA.keep_small(jd, jobs / T)
    finally:
        os.chdir("/")
        shutil.rmtree(local, ignore_errors=True)
    print(f"{T}: {st}  {json.dumps({k: round(v) for k, v in times.items()})}", flush=True)
    return 0


def cmd_pool(lst, n_par, cores, manifest, root, i=0, n=1):
    import pandas as pd
    M = setup(root)
    FL, root = M["FL"], M["root"]
    tpl = copy_templates(root)                                   # once, before the children start
    manifest = _abs(manifest)
    man = check_manifest(manifest).set_index("job_id")
    ids = pd.read_csv(lst, dtype=str).job_id.tolist()[int(i)::int(n)]
    miss = [T for T in ids if T not in man.index]
    if miss:
        C.stop(f"{lst}: job ids not in {manifest}: {miss[:10]}")
    fin = root / "g2lf_final"
    print(f"[g2_deploy pool] {len(ids)} reactions, {n_par} at a time x {cores} cores, root {root}, templates "
          f"{tpl['status']}", flush=True)

    def one(T):
        rk = FL.rxn_key(dict(man.loc[T], job_id=T))
        if FL.marker(fin / str(rk)) is not None:
            return T, "skip"
        if time.time() - T0 > STOP_AFTER_H * 3600:
            return T, "not_started"
        log = root / "logs" / "rxn" / f"{T}.log"
        C.assert_write_ok(log)
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "w") as f:
            r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "one", T, str(cores), "--manifest",
                                str(manifest), "--out", str(root)], stdout=f, stderr=subprocess.STDOUT)
        mk = FL.marker(fin / str(rk))
        return T, (mk if mk else f"error rc={r.returncode}")
    res = {}
    with ThreadPoolExecutor(max_workers=int(n_par)) as ex:
        for fu in as_completed([ex.submit(one, T) for T in ids]):
            T, s = fu.result()
            key = s.split("_")[0] if s.startswith(".fail") else s
            res[key] = res.get(key, 0) + 1
            print(f"  {T}: {s}  ({sum(res.values())}/{len(ids)}, {time.time() - T0:.0f} s)", flush=True)
    print(f"[g2_deploy pool] {res}", flush=True)
    return 2 if res.get("not_started") or any(k.startswith("error") for k in res) else 0


def cmd_i3_prep(root, seed=C.SEED):
    """I-3 inputs (spec 14-3): one P_main para reference pair per D1 dipole kind (5 kinds -> 5 pairs, 10 reactions incl.
    the 5 reference reactions), seeded choice among pairs whose both reactions are G2 ok (analysis/d1_g2/results/
    G_status.csv) -> <root>/i3_manifest.csv (their D1 manifest rows, verbatim), i3_list.csv (job_id), i3_pairs.csv
    (pair_id, a_rxn, b_rxn, a_job, b_job, dipole). Reads only; writes under <root>."""
    import numpy as np
    import pandas as pd
    root = _abs(root)
    C.assert_write_ok(root / "probe")
    S = pd.read_csv(C.REPO / "analysis" / "d1_pair" / "results" / "S_folds.csv", dtype={"a_job": str, "b_job": str})
    st = pd.read_csv(C.REPO / "analysis" / "d1_g2" / "results" / "G_status.csv", dtype=str)
    ok = set(st.loc[st.status == "ok", "job_id"])
    q = S[(S.ref_kind == "reference") & S.in_P_main.astype(bool) & (S.pos == "p") & S.a_job.isin(ok) & S.b_job.isin(ok)]
    rng = np.random.default_rng(seed)
    pick = []
    for dip, g in sorted(q.groupby("dipole"), key=lambda kv: kv[0]):
        pick.append(g.iloc[int(rng.integers(0, len(g)))])
    P = pd.DataFrame(pick)[["pair_id", "a_rxn", "b_rxn", "a_job", "b_job", "dipole"]].reset_index(drop=True)
    if P.dipole.nunique() != 5 or len(set(P.a_job) | set(P.b_job)) != 10:
        C.stop(f"I-3 selection: {P.dipole.nunique()} dipole kinds, {len(set(P.a_job) | set(P.b_job))} reactions "
               "(spec: 5 kinds x 2 = 10 reactions)")
    man = pd.read_csv(C.D1_MANIFEST, keep_default_na=False, dtype=str)
    jobs = sorted(set(P.a_job) | set(P.b_job))
    M = man[man.job_id.isin(jobs)]
    C.write_csv(root / "i3_manifest.csv", M)
    C.write_csv(root / "i3_list.csv", pd.DataFrame(dict(job_id=jobs)))
    C.write_csv(root / "i3_pairs.csv", P)
    print(P.to_string(index=False), flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd")
    a3 = sub.add_parser("i3-prep")
    a3.add_argument("--out", required=True)
    a1 = sub.add_parser("one")
    a1.add_argument("job_id")
    a1.add_argument("cores", type=int)
    a2 = sub.add_parser("pool")
    a2.add_argument("list_csv")
    a2.add_argument("n_parallel", type=int)
    a2.add_argument("cores", type=int)
    a2.add_argument("--slice", nargs=2, type=int, default=(0, 1))
    for p in (a1, a2):
        p.add_argument("--manifest", required=True)
        p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "i3-prep":
        return cmd_i3_prep(a.out)
    if a.cmd == "one":
        return cmd_one(a.job_id, a.cores, a.manifest, a.out)
    if a.cmd == "pool":
        return cmd_pool(a.list_csv, a.n_parallel, a.cores, a.manifest, a.out, *a.slice)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
