#!/usr/bin/env python3
"""D1-G2 structures (docs/specs/D1G2.md section 2): the D1 autodE protocol from the reaction SMILES with every
high-level ORCA calculation at XTB2 ALPB(water). Run through sbatch only (jobs/g2.slurm, G2ENV=d1 = the D1 run env).

  python g2_autode.py one <job_id> <cores>                          # one reaction: ts -> stereo -> ref -> final
  python g2_autode.py pool <list.csv> <n_parallel> <cores> [i n]    # reactions of the list (slice i of n), one
                                                                    # subprocess each, n_parallel at a time

What is the D1 code, unchanged (imported, not copied): run_ts.run_autode / analyse / imag_gate / forming_share,
stereo_check.py (run as the D1 script), make_reference.coley_port / dipolarophile_check, pilot_common, fragmenter.
What changes (spec 2, user decisions 2026-10-22):
  * config_g2.yaml = d1_run/config_run.yaml with scratch and the OptTS block (G1: Calc_Hess true, Recalc_Hess 5,
    MaxIter 200) changed; pilot_common.configure_autode is called as is and then the ORCA keyword sets are replaced
    by XTB2 ALPB(water) ones (LooseOpt / EnGrad / SP / Opt / OptTS Freq / Freq; no functional, basis, RIJCOSX, def2/J);
  * autodE's ORCA wrapper always adds CPCM(<solvent>) and, with SMD, a %cpcm block (autode/wrappers/ORCA.py
    add_solvent_keyword / print_solvent); user decision (a): those two methods are replaced at run time by no-ops in
    this process only (autodE files untouched), so every ORCA input is exactly 'XTB2 ALPB(water)' + the run type;
    the reaction keeps solvent_name='water' (xtb conformers with GBSA water and template matching as in D1);
  * the TS template library is $G2/ts_templates (autodE's DFT templates in the D1 env are neither read nor written);
  * reference step: make_reference.dft_refine -> xtb_refine (xtb --opt tight --alpb water --gfn 2, then
    xtb --ohess --alpb water for G); both the autodE dipole and the alt conformer go through it (orig_energy=None),
    the Coley rule and thresholds (RMSD > 0.05 A, dG < -0.239 kcal/mol) unchanged. No ORCA call in the reference step.
  * one infra retry (D1 retry_infra_once: 'autodE exception' + d1_run/orca_parse.py --infra evidence) for ts only.
Work happens in node-local ${TMPDIR:-/tmp}; $G2/jobs/<T>/ keeps the small outputs and a tar of every ORCA / xtb input
(S-1 audit); $G2/g2_final/<rxn_id>/ is the G1 folder format: ts.xyz, ts.hess (copy) and work/ts.hess (XTB2 ALPB(water)
Freq at the final TS, the file the features read), rel1.xyz (dipole), rel2.xyz (dipolarophile), result.json, .done or
.fail_<reason>. TS / reference atom order = the label's; a different order is re-mapped by graph isomorphism
(remapped: true) and an impossible map gives .fail_order.
"""
from __future__ import annotations

import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import g2_common as G  # noqa: E402

sys.path.insert(0, str(G.PILOT))
sys.path.insert(0, str(G.D1_RUN))
import pilot_common as pc  # noqa: E402
import run_ts as RT  # noqa: E402
import make_reference as MR  # noqa: E402
import orca_direct as OD  # noqa: E402

CONFIG = HERE / "config_g2.yaml"
XT = ["XTB2", "ALPB(water)"]
FREQ_HEADER = "! XTB2 ALPB(water) Freq\n%pal nprocs {n} end\n%maxcore {m}\n"
STOP_AFTER_H = float(os.environ.get("D1G2_STOP_AFTER_H", "44"))
T0 = float(os.environ.get("D1G2_JOB_T0", time.time()))
_ORIG_CONFIGURE = pc.configure_autode


# ================================================================ autodE at XTB2 ALPB(water)
def g2_configure(cfg, cores=None):
    """pilot_common.configure_autode unchanged, then the G2 changes (module docstring)."""
    ade = _ORIG_CONFIGURE(cfg, cores)
    from autode.wrappers.keywords import KeywordsSet, MaxOptCycles
    import autode.wrappers.ORCA as AO
    ade.Config.ORCA.keywords = KeywordsSet(
        low_opt=["LooseOpt", *XT, MaxOptCycles(10)], grad=["EnGrad", *XT], low_sp=["SP", *XT], opt=["Opt", *XT],
        opt_ts=["OptTS", "Freq", *XT, cfg["ts_stage"]["orca_optts_block"]], hess=["Freq", *XT], sp=["SP", *XT])
    AO.ORCA.add_solvent_keyword = lambda self, molecule, keywords: None      # user decision (a), this process only
    AO.ORCA.print_solvent = lambda self, inp_file, molecule, keywords: None
    tdir = G.G2 / "ts_templates"
    tdir.mkdir(parents=True, exist_ok=True)
    ade.Config.ts_template_folder_path = str(tdir)
    return ade


pc.configure_autode = g2_configure            # run_ts.run_autode and make_reference.coley_port look it up at call time

RE_TOTAL = re.compile(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh")
RE_FREE = re.compile(r"TOTAL FREE ENERGY\s+(-?\d+\.\d+)\s+Eh")
XTB_LOG = []                                   # (name, args) of every xtb CLI call of the reference step


def _xtb(args, wd):
    wd.mkdir(parents=True, exist_ok=True)
    with open(wd / "xtb.out", "w") as fo, open(wd / "xtb.err", "w") as fe:
        subprocess.run([os.environ.get("XTB_BIN", "xtb"), *args], cwd=wd, stdout=fo, stderr=fe, check=False)
    XTB_LOG.append(dict(dir=str(wd.name), args=args))
    return (wd / "xtb.out").read_text(errors="replace")


def xtb_refine(ade, mol, temp):
    """replaces make_reference.dft_refine: xtb --opt tight --alpb water --gfn 2 (coordinates written back to mol),
    then xtb --ohess --alpb water on that geometry (E = TOTAL ENERGY, G = TOTAL FREE ENERGY at 298.15 K)."""
    q = int(mol.charge)
    base = Path.cwd() / f"xtbref_{mol.name}"
    d1 = base / "opt"
    d1.mkdir(parents=True, exist_ok=True)
    syms = [a.label for a in mol.atoms]
    pc.write_xyz(d1 / "in.xyz", syms, np.array([a.coord for a in mol.atoms], dtype=float), mol.name)
    out = _xtb(["in.xyz", "--opt", "tight", "--alpb", "water", "--gfn", "2", "--chrg", str(q), "--uhf", "0"], d1)
    if not (d1 / "xtbopt.xyz").is_file() or "GEOMETRY OPTIMIZATION CONVERGED" not in out:
        raise MR.StageFail(f"xtb --opt tight not converged ({mol.name})")
    s2, x2 = pc.read_xyz(d1 / "xtbopt.xyz")
    if list(s2) != syms:
        raise MR.StageFail(f"xtb changed the atom order ({mol.name})")
    mol.coordinates = np.asarray(x2, dtype=float)
    e_opt = float(RE_TOTAL.findall(out)[-1])
    d2 = base / "ohess"
    d2.mkdir(parents=True, exist_ok=True)
    shutil.copy(d1 / "xtbopt.xyz", d2 / "in.xyz")
    out2 = _xtb(["in.xyz", "--ohess", "--alpb", "water", "--gfn", "2", "--chrg", str(q), "--uhf", "0"], d2)
    if not RE_FREE.findall(out2) or not RE_TOTAL.findall(out2):
        raise MR.StageFail(f"xtb --ohess failed ({mol.name})")
    e, g = float(RE_TOTAL.findall(out2)[-1]), float(RE_FREE.findall(out2)[-1])
    return dict(e_opt_eh=e_opt, e_sp_eh=e, g_cont_eh=g - e, G_eh=g, level="xtb_gfn2_alpb_water")


MR.dft_refine = xtb_refine                    # coley_port looks it up at call time


# ================================================================ stages (node-local job folder jd)
def infra(stage_dir):
    r = subprocess.run([sys.executable, str(G.D1_RUN / "orca_parse.py"), "--infra", str(stage_dir)],
                       capture_output=True, text=True)
    return r.returncode == 0


def stage_ts(cfg, job, jd, cores):
    """run_ts.main for kind D1 (run_autode + analyse), with the D1 retry_infra_once rule."""
    if pc.done(jd, "ts"):
        return True
    fr, _ = pc.import_repo_modules(cfg)
    res = dict(job_id=job["job_id"], kind=job["kind"], ts_id=job["ts_id"], rxn_smiles=job["rxn_smiles"],
               engine=cfg["ts_stage"]["engine"], dry_xtb=False, level="XTB2 ALPB(water)", attempts=[])
    for att in (1, 2):
        t0 = time.time()
        try:
            got = RT.run_autode(cfg, job, jd, res, False)
            res["wall_s"] = time.time() - t0
            RT.analyse(cfg, job, jd, res, got, fr)
        except RT.StageFail as e:
            res["wall_s"] = time.time() - t0
            res["attempts"].append(dict(attempt=att, wall_s=res["wall_s"], fail=str(e)))
            if att == 1 and str(e).startswith("autodE exception") and infra(jd / "ts"):
                continue                                         # D1 retry_infra_once: same input, once
            pc.write_json(jd / "ts_result.json", res)
            pc.mark_fail(jd, "ts", str(e))
            return False
        res["attempts"].append(dict(attempt=att, wall_s=res["wall_s"], fail=None))
        pc.write_json(jd / "ts_result.json", res)
        pc.mark_done(jd, "ts")
        return True
    return False


def stage_stereo(cfg_path, job, jd):
    if pc.done(jd, "stereo"):
        return True
    r = subprocess.run([sys.executable, str(G.D1_RUN / "stereo_check.py"), "--job", job["job_id"], "--config",
                        str(cfg_path), "--manifest", str(G.MANIFEST)], cwd=G.PILOT, capture_output=True, text=True)
    (jd / "stereo.log").write_text(r.stdout + r.stderr)
    return pc.done(jd, "stereo")


def stage_ref(cfg, job, jd):
    """make_reference.main's D1 branch with orig_energy=None (both reactant dipole conformers through xtb_refine)."""
    if pc.done(jd, "ref"):
        return True
    fr, _ = pc.import_repo_modules(cfg)
    ts_res = json.loads((jd / "ts_result.json").read_text())
    res = dict(job_id=job["job_id"], kind=job["kind"], level="xtb CLI (G2)")
    t0 = time.time()
    XTB_LOG.clear()
    try:
        orig = pc.read_xyz(jd / "r_dipole_autode.xyz")
        r_dph = pc.read_xyz(jd / "r_dipolarophile_autode.xyz")
        port, alt = MR.coley_port(cfg, jd, jd / "ref", ts_res, fr, orig, None, False)
        res["port"] = port
        if "abort" in port and port.get("abort", "").startswith("ring count changed"):
            raise MR.StageFail(port["abort"])
        ref_dip = alt if port["use_alt"] else orig
        res.update(source="coley_port_xtb", alt_used=int(port["use_alt"]))
        res["dipolarophile_check"] = MR.dipolarophile_check(fr, jd, ts_res, job, r_dph)
        if res["dipolarophile_check"].get("ok") is False:
            raise MR.StageFail("dipolarophile configuration at the TS differs from its reference")
    except MR.StageFail as e:
        res.update(wall_s=time.time() - t0, xtb_calls=list(XTB_LOG))
        pc.write_json(jd / "ref_result.json", res)
        pc.mark_fail(jd, "ref", str(e))
        return False
    except Exception as e:                                       # noqa: BLE001
        res.update(wall_s=time.time() - t0, traceback=traceback.format_exc(), xtb_calls=list(XTB_LOG))
        pc.write_json(jd / "ref_result.json", res)
        pc.mark_fail(jd, "ref", f"{type(e).__name__}: {e}")
        return False
    pc.write_xyz(jd / "ref_dipole.xyz", *ref_dip, f"{job['job_id']} G2 dipole reference (alt_used={res['alt_used']})")
    pc.write_xyz(jd / "ref_dipolarophile.xyz", *r_dph, f"{job['job_id']} G2 dipolarophile reference")
    res.update(wall_s=time.time() - t0, xtb_calls=list(XTB_LOG),
               n_orca_outputs=len(list((jd / "ref").rglob("*_orca.out"))) if (jd / "ref").is_dir() else 0)
    pc.write_json(jd / "ref_result.json", res)
    pc.mark_done(jd, "ref")
    return True


# ================================================================ atom order vs the label
def _graph(fr, syms, xyz, extra_edges=(), tags=None):
    import networkx as nx
    B = fr.bond_matrix(syms, np.asarray(xyz, dtype=float))
    g = nx.Graph()
    for i, s in enumerate(syms):
        g.add_node(i, lab=(s, None if tags is None else tags[i]))
    n = len(syms)
    for i in range(n):
        for j in range(i + 1, n):
            if B[i, j]:
                g.add_edge(i, j, f=False)
    for i, j in extra_edges:
        g.add_edge(i, j, f=True)
    return g


def remap(fr, a, b, extra_a=(), extra_b=(), tags_a=None, tags_b=None, cap=5000):
    """permutation p with b[p] aligned to a (a = label order): graph isomorphism on all atoms (element + fragment tag
    labels, covalent edges + the forming-bond edges), the one with the lowest proper-rotation RMSD; None if none."""
    from networkx.algorithms.isomorphism import GraphMatcher
    ga, gb = _graph(fr, *a, extra_a, tags_a), _graph(fr, *b, extra_b, tags_b)
    gm = GraphMatcher(ga, gb, node_match=lambda x, y: x["lab"] == y["lab"], edge_match=lambda x, y: x["f"] == y["f"])
    best, bp = np.inf, None
    A = np.asarray(a[1], dtype=float)
    for k, m in enumerate(gm.isomorphisms_iter()):
        p = [m[i] for i in range(len(a[0]))]
        Bp = np.asarray(b[1], dtype=float)[p]
        P, Q = A - A.mean(0), Bp - Bp.mean(0)
        U, S, Vt = np.linalg.svd(P.T @ Q)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
        r = float(np.sqrt(((P @ R.T - Q) ** 2).sum(1).mean()))
        if r < best:
            best, bp = r, p
        if k + 1 >= cap:
            break
    return bp, best


def label_pairs(lab):
    return sorted(tuple(sorted(map(int, p.split("-")))) for p in str(lab["formed_pairs_ts"]).split())


# ================================================================ final folder (G1 format)
def finalize(cfg, job, jd, lab, cores, fin, times):
    """$G2/g2_final/<rxn_id>/ from the node-local job folder; returns the status string."""
    fr, _ = pc.import_repo_modules(cfg)
    rid = int(lab["rxn_id"])
    res = dict(rxn_id=rid, job_id=job["job_id"], level="G2: autodE 1.4.5 from SMILES, hmethod ORCA XTB2 ALPB(water)",
               stage_wall_s=times, cores=cores)
    reasons = []
    for st in ("ts", "stereo", "ref"):
        f = pc.failed(jd, st)
        if f is not None:
            reasons.append(f"{st}_" + (f.split(":")[0] if f else "fail"))
            res[f"{st}_fail"] = f
            break
        if not pc.done(jd, st):
            reasons.append(f"{st}_not_run")
            break
    for f in ("ts_result.json", "stereo_result.json", "ref_result.json"):
        if (jd / f).is_file():
            r = json.loads((jd / f).read_text())
            res[f.split(".")[0]] = {k: r.get(k) for k in ("autode_version", "keywords", "n_cores", "imag_freqs_cm",
                                                          "formed_pairs_ts", "formed_d", "checks", "A_idx", "status",
                                                          "found", "requested", "alt_used", "source", "attempts",
                                                          "n_orca_outputs", "xtb_calls", "wall_s") if k in r}
    if not reasons:
        try:
            ts = pc.read_xyz(jd / "ts.xyz")
            lts = pc.read_xyz(G.DP.D1_JOBS / job["job_id"] / "ts.xyz")
            tsr = json.loads((jd / "ts_result.json").read_text())
            lp = label_pairs(lab)
            g2p = sorted(tuple(sorted(p)) for p in tsr["formed_pairs_ts"])
            res["remapped"] = False
            if list(ts[0]) != list(lts[0]) or g2p != lp:
                tags_l = [1 if i in set(lab_A_idx(job)) else 2 for i in range(len(lts[0]))]
                tags_g = [1 if i in set(tsr["A_idx"]) else 2 for i in range(len(ts[0]))]
                p, r = remap(fr, lts, ts, lp, g2p, tags_l, tags_g)
                if p is None:
                    raise Fail("order")
                ts = (list(lts[0]), np.asarray(ts[1], dtype=float)[p])
                res.update(remapped=True, remap_rmsd_A=r)
            rels = []
            for k, (mine, theirs) in enumerate((("ref_dipole.xyz", "ref_dipole.xyz"),
                                                ("ref_dipolarophile.xyz", "ref_dipolarophile.xyz")), 1):
                g = pc.read_xyz(jd / mine)
                l = pc.read_xyz(G.DP.D1_JOBS / job["job_id"] / theirs)
                if list(g[0]) != list(l[0]):
                    p, r = remap(fr, l, g)
                    if p is None:
                        raise Fail("order")
                    g = (list(l[0]), np.asarray(g[1], dtype=float)[p])
                    res[f"rel{k}_remapped"] = True
                rels.append(g)
            # ts.hess: XTB2 ALPB(water) Freq at the final TS (label order)
            wd = jd / "final_work"
            wd.mkdir(exist_ok=True)
            inp = wd / "ts.inp"
            OD.write_inp(inp, FREQ_HEADER.format(n=cores, m=cfg["maxcore_mb"]), *ts,
                         charge=int(lab["charge1"]) + int(lab["charge2"]), mult=1)
            t1 = time.time()
            txt = OD.run(cfg, inp).read_text(errors="replace")
            times["freq"] = time.time() - t1
            if not OD.terminated(txt) or not (wd / "ts.hess").is_file():
                raise Fail("freq_not_terminated")
            imag, mode = OD.imag_and_mode(wd / "ts.hess")
            heavy = [i for i, s in enumerate(ts[0]) if s != "H"]
            share = None if mode is None else RT.forming_share(np.asarray(ts[1], dtype=float), mode, lp, heavy)
            D = np.linalg.norm(np.asarray(ts[1])[:, None] - np.asarray(ts[1])[None], axis=-1)
            res.update(imag_freqs_cm=imag, n_imag=len(imag), imag_mode_forming_share=share,
                       formed_pairs_label=[list(p) for p in lp], d_form_g2=[float(D[p]) for p in lp],
                       e_ts_freq_eh=OD.last_fspe(txt), freq_header=FREQ_HEADER.strip().splitlines()[0])
            if not imag:
                raise Fail("hess_no_imaginary_frequency")
        except Fail as e:
            reasons.append(str(e))
        except Exception as e:                                   # noqa: BLE001
            res["traceback"] = traceback.format_exc()
            reasons.append(f"error_{type(e).__name__}")
    res["core_h"] = {k: v * cores / 3600 for k, v in times.items()}
    res["core_h_total"] = float(sum(res["core_h"].values()))
    res["fail_reasons"] = reasons
    res["status"] = "ok" if not reasons else "fail:" + reasons[0]
    out = fin / str(rid)
    tmp = fin / f"{rid}.tmp"
    G.assert_path(tmp)
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "work").mkdir(parents=True)
    if not reasons:
        pc.write_xyz(tmp / "ts.xyz", *ts, f"G2 TS rxn {rid}")
        for k, g in enumerate(rels, 1):
            pc.write_xyz(tmp / f"rel{k}.xyz", *g, f"G2 reference {k} rxn {rid}")
        shutil.copy(jd / "final_work" / "ts.hess", tmp / "work" / "ts.hess")
        shutil.copy(jd / "final_work" / "ts.hess", tmp / "ts.hess")
        for f in ("ts.inp", "ts.out"):
            shutil.copy(jd / "final_work" / f, tmp / "work" / f)
    (tmp / "result.json").write_text(json.dumps(res, indent=1, default=pc._jsonable))
    marker = ".done" if not reasons else ".fail_" + re.sub(r"[^A-Za-z0-9_]+", "_", reasons[0])
    (tmp / marker).write_text(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n" + "\n".join(reasons) + "\n")
    if out.exists():
        shutil.rmtree(out)
    os.replace(tmp, out)
    return res["status"]


class Fail(Exception):
    pass


def lab_A_idx(job):
    """(1)-tagged atoms of the label eda.inp (= the label partition, as D1PAIR's input_meta)."""
    txt = (G.DP.D1_JOBS / job["job_id"] / "sp" / "eda.inp").read_text()
    at = [l for l in txt.splitlines() if re.match(r"^\s*[A-Z][a-z]?\(\d\)\s", l)]
    return [i for i, l in enumerate(at) if "(1)" in l.split()[0]]


def keep_small(jd, dest):
    """the small outputs + a tar of every ORCA input (.inp) of the reaction -> $G2/jobs/<T>/ (atomic; quota)."""
    tmp = dest.with_name(dest.name + ".tmp")
    G.assert_path(tmp)
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for p in jd.iterdir():
        if p.is_file() and (p.suffix in (".xyz", ".json", ".csv", ".txt", ".log") or p.name.startswith((".done", ".fail"))):
            shutil.copy(p, tmp / p.name)
    with tarfile.open(tmp / "inputs.tar.gz", "w:gz") as t:
        for p in sorted(jd.rglob("*")):
            if p.is_file() and p.suffix == ".inp":
                t.add(p, arcname=str(p.relative_to(jd)))
    if dest.exists():
        shutil.rmtree(dest)
    os.replace(tmp, dest)


def cmd_one(T, cores):
    cores = int(cores)
    lab = {d["job_id"]: d for d in G.targets()}[T]
    fin = G.G2_FINAL
    if any((fin / str(lab["rxn_id"])).glob(".done")) or any((fin / str(lab["rxn_id"])).glob(".fail_*")):
        print(f"{T}: final exists, skip", flush=True)
        return 0
    os.environ["SLURM_CPUS_PER_TASK"] = str(cores)               # pilot_common.n_cores() -> autodE n_cores
    os.environ["OMP_NUM_THREADS"] = str(cores)
    local = Path(os.environ.get("TMPDIR", "/tmp")) / f"d1g2_{os.getpid()}"
    cfg = yaml.safe_load(CONFIG.read_text())
    cfg["scratch"] = str(local)
    cfg_path = local / "config_g2_local.yaml"
    local.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(cfg))
    job = pc.load_job(T, G.MANIFEST)
    jd = pc.job_dir(cfg, T)
    times = {}
    try:
        t = time.time(); ok = stage_ts(cfg, job, jd, cores); times["ts"] = time.time() - t
        if ok:
            t = time.time(); ok = stage_stereo(cfg_path, job, jd); times["stereo"] = time.time() - t
        if ok:
            t = time.time(); ok = stage_ref(cfg, job, jd); times["ref"] = time.time() - t
        os.chdir(local)
        st = finalize(cfg, job, jd, lab, cores, fin, times)
        keep_small(jd, G.G2_JOBS / T)
    finally:
        os.chdir("/")
        shutil.rmtree(local, ignore_errors=True)
    print(f"{T}: {st}  {json.dumps({k: round(v) for k, v in times.items()})}", flush=True)
    return 0


def cmd_pool(lst, n_par, cores, i=0, m=1):
    ids = pd.read_csv(lst, dtype=str).job_id.tolist()[int(i)::int(m)]
    lab = {d["job_id"]: d for d in G.targets()}
    print(f"[g2 pool] {len(ids)} reactions, {n_par} at a time x {cores} cores", flush=True)

    def one(T):
        rid = lab[T]["rxn_id"]
        if any((G.G2_FINAL / str(rid)).glob(".done")) or any((G.G2_FINAL / str(rid)).glob(".fail_*")):
            return T, "skip"
        if time.time() - T0 > STOP_AFTER_H * 3600:
            return T, "not_started"
        log = G.LOGS / "rxn" / f"{T}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "w") as f:
            r = subprocess.run([sys.executable, __file__, "one", T, str(cores)], stdout=f, stderr=subprocess.STDOUT)
        fs = [p.name for p in (G.G2_FINAL / str(rid)).glob(".*")] if (G.G2_FINAL / str(rid)).is_dir() else []
        return T, (fs[0] if fs else f"error rc={r.returncode}")
    res = {}
    with ThreadPoolExecutor(max_workers=int(n_par)) as ex:
        for f in as_completed([ex.submit(one, T) for T in ids]):
            T, s = f.result()
            key = s.split("_")[0] if s.startswith(".fail") else s
            res[key] = res.get(key, 0) + 1
            print(f"  {T}: {s}  ({sum(res.values())}/{len(ids)}, {time.time() - T0:.0f} s)", flush=True)
    print(f"[g2 pool] {res}", flush=True)
    return 2 if res.get("not_started") or any(k.startswith("error") for k in res) else 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["one"] and len(a) == 3:
        sys.exit(cmd_one(a[1], a[2]))
    if a[:1] == ["pool"] and len(a) in (4, 6):
        sys.exit(cmd_pool(*a[1:]))
    raise SystemExit(__doc__)
