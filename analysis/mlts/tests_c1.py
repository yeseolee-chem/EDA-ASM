#!/usr/bin/env python3
"""tests_c1.py — DEPLOY-V2 C-1 (A-1 bridge / optimizer / Hessian checks) and C-4 (MLIP unit check), with a zero-shot
model now and FT_0 later (docs/specs/DEPLOY_V2_MLTS.md 6 A-1, 8 C-1 / C-4). Results: results/C1_tests.json, one entry
per model and section (each section call merges its own part; reruns overwrite that part only).

  python tests_c1.py <section|all> [model]        (GPU sandbox / GPU job; model default MACE-OMOL-0)
sections:
  bridge     (i)   ORCA `ExtOpt EnGrad` (server + client) vs mlts_bridge.direct_engrad (in-process) on 3 label TSs at
                   the coordinates ORCA used: |dE| < 1e-6 Eh, max |dG| < 1e-5 Eh/bohr
  optimizer  (ii)  ORCA `ExtOpt OptTS` (optimize_ts, production settings) vs Sella (ASE, order=1, same model, MLIP
                   Hessian as initial Hessian, fmax 1e-3 eV/A) from 5 label TSs: max forming-distance difference
                   < 0.005 A (pass_production); a miss is re-run with both tight (ORCA VeryTightOpt, Sella fmax
                   1e-4 eV/A) and pass = every reaction within 0.005 A at production or tight settings
  optimizer_diag <rxn_id>  that tight re-check for one reaction (not in 'all')
  hessian    (iii) ORCA `ExtOpt NumFreq` vs the MLIP analytic Hessian at the ORCA TS of the 2 smallest optimizer
                   reactions: same n_imag (modes < -50 cm-1); also max |dnu|, the numerical-Hessian cost (= one
                   Recalc_Hess through ExtOpt) and analytic vs finite-difference Hessian in-process
  units      C-4   MLIP E/F vs ORCA label-level DFT EnGrad (mlt_common.orca_grad_input, serial ORCA) on distorted water,
                   methane and 3 displaced label TSs (Gaussian 0.03 A, seeded; at the exact label TS the DFT forces
                   are ~0): force RMS ratio in [1/3, 3] and cosine > 0.8 on the TSs; plus an energy / force
                   finite-difference consistency check of the MLIP
  overhead   per-call cost of the ORCA client (subprocess start + socket + GPU), orca-mlips' own client for
                   comparison, ORCA-side cost per ExtOpt call, wall of one OptTS from the label TS
  recalc     Recalc_Hess choice: the 2 smallest reactions from their G2 TS with recalc none / analytic:30 / numeric:30
Reactions: seeded (mlt_common.SEED) one per dipole from the training side of fold 0 with 15-30 atoms, a label TS and a
G2 TS (C1_pick in the json)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import mlt_common as C  # noqa: E402
import mlts_bridge as B  # noqa: E402

OUT = C.RES / "C1_tests.json"
TOL = dict(bridge_dE_eh=1e-6, bridge_dG_eh_bohr=1e-5, optimizer_dform_A=0.005, units_ratio=3.0, units_cos=0.8)
SELLA_FMAX = 1e-3                     # eV/A (ORCA NormalOpt TolMaxG 3e-4 Eh/bohr = 0.015 eV/A)


def pick():
    T = C.targets()
    T = T[(T.fold_T1 != 0) & T.n_atoms.between(15, 30)]
    T = T[[C.label_ts(r).is_file() and C.g2_ts_label_order(r).is_file() for r in T.rxn_id]]
    rng = np.random.default_rng(C.SEED)
    out = []
    for d in C.DIPOLES:
        g = T[T.dipole == d].sort_values("rxn_id")
        if len(g):
            r = g.iloc[int(rng.integers(len(g)))]
            out.append(dict(rxn_id=int(r.rxn_id), dipole=d, n_atoms=int(r.n_atoms), formed_pairs_ts=r.formed_pairs_ts))
    return out


def merge(model, section, data):
    J = C.read_json(OUT) if OUT.is_file() else {}
    J.setdefault("models", {}).setdefault(model, {})[section] = data
    J["tolerances"] = TOL
    J["updated"] = C.now()
    m = J["models"][model]
    for sec, key in (("bridge", "pass"), ("optimizer", "pass"), ("hessian", "pass"), ("units", "pass")):
        if sec in m:
            m.setdefault("passes", {})[sec] = m[sec].get(key)
    if all(s in m for s in ("bridge", "optimizer", "hessian")):
        m["pass_C1"] = bool(m["bridge"].get("pass") and m["optimizer"].get("pass") and m["hessian"].get("pass"))
    if "units" in m:
        m["pass_C4"] = bool(m["units"].get("pass"))
    C.write_json(OUT, J)


def label(rx):
    s, p = C.read_xyz(C.label_ts(rx["rxn_id"]))
    return s, np.asarray(p, float), C.formed_pairs(rx["formed_pairs_ts"])


def work(tag):
    d = B.local_tmp() / f"mlts_c1_{B.jobtag()}_{os.getpid()}" / tag
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)
    return d


# ================================================================ (i) bridge
def t_bridge(spec, sock, picks):
    rows = []
    for rx in picks[:3]:
        s, p, _ = label(rx)
        w = work(f"bridge_{rx['rxn_id']}")
        r = B.run_orca_ext(s, p, "EnGrad", spec, sock, w, base="eg")
        if "grad_eh_bohr" not in r:
            rows.append(dict(rxn_id=rx["rxn_id"], error="no .engrad", tail=r.get("out_tail")))
            continue
        pe = np.asarray(r["engrad_pos_A"], float)
        e, g = B.direct_engrad(spec, s, pe)
        rows.append(dict(rxn_id=rx["rxn_id"], n_atoms=len(s), orca_rc=r["rc"], orca_wall_s=r["wall_s"], ext=r["ext"],
                         coord_shift_A=float(np.abs(pe - p).max()),
                         dE_eh=abs(float(r["engrad_energy_eh"]) - e), dG_eh_bohr=float(np.abs(r["grad_eh_bohr"] - g).max()),
                         gmax_eh_bohr=float(np.abs(g).max())))
    ok = len(rows) == 3 and all("error" not in x and x["dE_eh"] < TOL["bridge_dE_eh"] and
                                x["dG_eh_bohr"] < TOL["bridge_dG_eh_bohr"] for x in rows)
    return {"rows": rows, "pass": ok}


# ================================================================ (ii) optimizer
def sella_ts(spec, s, p, fmax=SELLA_FMAX, steps=400):
    from ase import Atoms
    from sella import Sella
    k = C.EH2EV / C.BOHR2A ** 2

    def hf(at):
        return B.hessian(spec, s, at.positions) * k
    res = {}
    for internal in (True, False):
        at = Atoms(s, positions=p)
        at.calc = B.ase_calculator(spec)
        t0 = time.time()
        try:
            opt = Sella(at, order=1, internal=internal, hessian_function=hf, logfile=None, trajectory=None)
            conv = bool(opt.run(fmax=fmax, steps=steps))
            res = dict(pos=at.positions.copy(), converged=conv, steps=int(opt.nsteps), n_calls=at.calc.n_calls,
                       internal=internal, wall_s=time.time() - t0,
                       fmax_final=float(np.linalg.norm(at.get_forces(), axis=1).max()))
            break
        except Exception as exc:                               # noqa: BLE001
            res = dict(error=f"{type(exc).__name__}: {exc}"[:300], internal=internal)
    return res


def t_optimizer(spec, sock, picks):
    from concurrent.futures import ThreadPoolExecutor
    def orca_one(rx, tight=False):
        s, p, pairs = label(rx)
        return B.optimize_ts(spec, s, p, pairs, work(f"opt_{rx['rxn_id']}_{int(tight)}"), sock, tight=tight)
    with ThreadPoolExecutor(len(picks)) as ex:
        futs = [ex.submit(orca_one, rx) for rx in picks]
        sel = [sella_ts(spec, *label(rx)[:2]) for rx in picks]          # in-process, while ORCA runs
        orc = [f.result() for f in futs]
    rows, cache = [], {}
    for rx, o, sl in zip(picks, orc, sel):
        s, p, pairs = label(rx)
        row = dict(rxn_id=rx["rxn_id"], dipole=rx["dipole"], n_atoms=len(s),
                   orca=dict({k: o.get(k) for k in ("status", "converged", "n_cycles", "n_ext_calls", "n_imag", "imag_cm",
                                                    "gpu_s", "wall_s", "orca_wall_s", "orca_side_ms_per_call",
                                                    "gmax_final_eh_bohr", "recalc", "e_final_eh")},
                              d_form=B.dists(o["pos_A"], pairs)),
                   d_label=B.dists(p, pairs))
        if "error" in sl:
            row.update(sella=sl, dform_max_A=None)
            rows.append(row)
            continue
        e_s, _ = B.direct_engrad(spec, s, sl["pos"])
        hs = B.hessian(spec, s, sl["pos"])
        row["sella"] = dict({k: sl[k] for k in ("converged", "steps", "n_calls", "internal", "wall_s", "fmax_final")},
                            d_form=B.dists(sl["pos"], pairs), e_eh=e_s, **B.imag_summary(B.freqs_cm(hs, s, sl["pos"])))
        row["dform_max_A"] = float(np.max(np.abs(np.subtract(row["orca"]["d_form"], row["sella"]["d_form"]))))
        row["rmsd_direct_A"] = B.heavy_rmsd_direct(s, o["pos_A"], sl["pos"])
        row["dE_kcal"] = (o["e_final_eh"] - e_s) * C.EH2KCAL
        row["pass"] = row["dform_max_A"] < TOL["optimizer_dform_A"]
        cache[rx["rxn_id"]] = o["pos_A"]
        rows.append(row)
    for r in rows:                                               # tolerance check of a miss: both optimisers tight
        r["pass_production"] = bool(r.get("pass"))
        if not r.get("pass"):
            try:
                r["tight"] = t_optimizer_diag(spec, sock, picks, r["rxn_id"])
                r["pass"] = bool(r["tight"].get("pass"))
            except Exception as exc:                             # noqa: BLE001
                r["tight"] = dict(error=str(exc)[:300])
    ok = len(rows) == len(picks) and all(r.get("pass") for r in rows)
    both = [r for r in rows if r["orca"].get("converged") and (r.get("sella") or {}).get("converged")]
    out = {"rows": rows, "sella_fmax_eV_A": SELLA_FMAX, "pass": ok,
           "pass_production": len(rows) == len(picks) and all(r["pass_production"] for r in rows),
           "n_both_converged": len(both),
           "max_dform_where_both_converged_A": max((r["dform_max_A"] for r in both), default=None),
           "note": "pass_production: ORCA (D1 block, NormalOpt) vs Sella (fmax 1e-3 eV/A) within 0.005 A for all 5. "
                   "A miss is re-run with both optimisers tight (ORCA VeryTightOpt, Sella fmax 1e-4 eV/A): on a flat "
                   "PES the NormalOpt stopping point can move ~0.01 A at < 0.01 kcal/mol. pass = every row within "
                   "0.005 A at production or tight settings. Non-convergence of a zero-shot model fails both."}
    return out, cache


# ================================================================ (iii) Hessian
def t_hessian(spec, sock, picks, opt_pos=None):
    rows = []
    sm = sorted(picks, key=lambda x: x["n_atoms"])[:2]
    for rx in sm:
        s, p, pairs = label(rx)
        if opt_pos and rx["rxn_id"] in opt_pos:
            g = np.asarray(opt_pos[rx["rxn_id"]], float)
            src = "ORCA ExtOpt OptTS TS"
        else:
            o = B.optimize_ts(spec, s, p, pairs, work(f"hopt_{rx['rxn_id']}"), sock)
            g, src = np.asarray(o["pos_A"], float), "ORCA ExtOpt OptTS TS (rerun)"
        w = work(f"numfreq_{rx['rxn_id']}")
        r = B.run_orca_ext(s, g, "NumFreq", spec, sock, w, base="nf")
        fr = np.sort(np.asarray(r.get("freqs_cm") or [], float))
        nt = 6
        orca_vib = np.sort(fr[np.argsort(np.abs(fr))[nt:]]) if len(fr) else np.array([])
        t0 = time.time()
        H = B.hessian(spec, s, g, socket=sock)
        t_an = time.time() - t0
        nu = B.freqs_cm(H, s, g)
        row = dict(rxn_id=rx["rxn_id"], n_atoms=len(s), geometry=src, orca_numfreq_wall_s=r["wall_s"],
                   orca_numfreq_ext_calls=r["ext"]["n"], orca_numfreq_gpu_s=r["ext"]["eval_s"], analytic_hessian_s=t_an,
                   orca=B.imag_summary(orca_vib) if len(orca_vib) else dict(error="no frequencies", tail=r.get("out_tail")),
                   mlip=B.imag_summary(nu))
        if len(orca_vib) == len(nu):
            row["max_dnu_cm"] = float(np.abs(orca_vib - nu).max())
            row["dnu_imag_cm"] = float(orca_vib[0] - nu[0])
        row["pass"] = bool(len(orca_vib) and row["orca"]["n_imag"] == row["mlip"]["n_imag"])
        rows.append(row)
    # analytic vs finite-difference Hessian (in-process, smallest reaction, h = 1e-3 A central differences of forces)
    s, p, _ = label(sm[0])
    H = B.hessian(spec, s, p)
    h, n3 = 1e-3, 3 * len(s)
    Hfd = np.zeros((n3, n3))
    for k in range(n3):
        a, c = divmod(k, 3)
        pp, pm = p.copy(), p.copy()
        pp[a, c] += h
        pm[a, c] -= h
        gp = B.direct_engrad(spec, s, pp)[1].ravel()
        gm = B.direct_engrad(spec, s, pm)[1].ravel()
        Hfd[:, k] = (gp - gm) / (2 * h / C.BOHR2A)
    Hfd = 0.5 * (Hfd + Hfd.T)
    fd = dict(rxn_id=sm[0]["rxn_id"], max_abs_diff_au=float(np.abs(H - Hfd).max()), max_abs_au=float(np.abs(H).max()),
              rel=float(np.abs(H - Hfd).max() / np.abs(H).max()),
              max_dnu_cm=float(np.abs(B.freqs_cm(H, s, p) - B.freqs_cm(Hfd, s, p)).max()))
    ok = len(rows) == 2 and all(r["pass"] for r in rows)
    return dict(rows=rows, analytic_vs_fd=fd, imag_tol_cm=B.IMAG_TOL_CM, **{"pass": ok})


# ================================================================ C-4 units
WATER = (["O", "H", "H"], [[0.0, 0.0, 0.12], [0.0, 0.80, -0.47], [0.0, -0.74, -0.52]])
METHANE = (["C", "H", "H", "H", "H"], [[0.0, 0.0, 0.0], [0.68, 0.68, 0.62], [-0.62, -0.66, 0.60],
                                        [-0.63, 0.63, -0.70], [0.66, -0.60, -0.63]])


def dft_engrad(s, p, tag):
    """label-level ORCA EnGrad (serial); cached in $MLT/tmp/c1_dft/<tag>_<geometry hash>.json (model independent)."""
    import hashlib
    key = hashlib.sha256((" ".join(s) + np.array2string(np.round(np.asarray(p, float), 8))).encode()).hexdigest()[:16]
    cp = C.MLT / "tmp" / "c1_dft" / f"{tag}_{key}.json"
    if cp.is_file():
        d = C.read_json(cp)
        return d["energy_eh"], np.asarray(d["grad_eh_bohr"], float), d["wall_s"]
    e, g, t = _dft_engrad(s, p, tag)
    C.write_json(cp, dict(tag=tag, symbols=list(s), pos_A=np.asarray(p, float).tolist(), energy_eh=e,
                          grad_eh_bohr=np.asarray(g).tolist(), wall_s=t, level=C.LABEL_GRAD_LINE.strip()))
    return e, g, t


def _dft_engrad(s, p, tag):
    w = work(f"dft_{tag}")
    (w / "d.inp").write_text(C.orca_grad_input(s, p, nprocs=1, title=f"C-4 {tag}"))
    t0 = time.time()
    with open(w / "d.out", "w") as fo:
        subprocess.run([str(C.ORCA), "d.inp"], cwd=w, stdout=fo, stderr=subprocess.STDOUT, env=B._orca_env(),
                       timeout=3600)
    import orca_io as OI
    d = OI.read_engrad(w / "d.engrad")
    shutil.rmtree(w, ignore_errors=True)
    return d["energy_eh"], np.asarray(d["grad_eh_bohr"], float), time.time() - t0


def t_units(spec, picks):
    from concurrent.futures import ThreadPoolExecutor
    rng = np.random.default_rng(C.SEED)
    items = [("water", *WATER), ("methane", *METHANE)]
    for rx in picks[:3]:
        s, p, _ = label(rx)
        items.append((f"ts_{rx['rxn_id']}", s, p + rng.normal(0.0, 0.03, p.shape)))
    with ThreadPoolExecutor(len(items)) as ex:
        dft = list(ex.map(lambda it: dft_engrad(it[1], np.asarray(it[2], float), it[0]), items))
    rows = []
    for (tag, s, p), (ed, gd, tw) in zip(items, dft):
        p = np.asarray(p, float)
        em, gm = B.direct_engrad(spec, s, p)
        fm, fd = -gm.ravel(), -gd.ravel()
        big = np.abs(fd) > 0.1 * np.abs(fd).max()
        rows.append(dict(tag=tag, n_atoms=len(s), e_mlip_eh=em, e_dft_eh=ed, dE_eh=em - ed, dft_wall_s=tw,
                         f_rms_mlip_eh_bohr=float(np.sqrt((fm ** 2).mean())), f_rms_dft_eh_bohr=float(np.sqrt((fd ** 2).mean())),
                         rms_ratio=float(np.sqrt((fm ** 2).mean() / (fd ** 2).mean())),
                         cosine=float(fm @ fd / (np.linalg.norm(fm) * np.linalg.norm(fd))),
                         sign_agree=float((np.sign(fm[big]) == np.sign(fd[big])).mean())))
    # MLIP internal consistency: -dE/du (central difference, h = 1e-3 A) vs F.u, eV/A
    s, p, _ = label(picks[0])
    u = rng.normal(size=p.shape)
    u /= np.linalg.norm(u)
    h = 1e-3
    ep, _, _ = B.direct_ev(spec, s, p + h * u)
    em_, _, _ = B.direct_ev(spec, s, p - h * u)
    _, f0, _ = B.direct_ev(spec, s, p)
    fdc = dict(rxn_id=picks[0]["rxn_id"], minus_dE_du_eV_A=-(ep - em_) / (2 * h), F_dot_u_eV_A=float((f0 * u).sum()))
    fdc["rel_err"] = abs(fdc["minus_dE_du_eV_A"] - fdc["F_dot_u_eV_A"]) / max(abs(fdc["F_dot_u_eV_A"]), 1e-12)
    ts = [r for r in rows if r["tag"].startswith("ts_")]
    ok = len(ts) == 3 and all(1 / TOL["units_ratio"] <= r["rms_ratio"] <= TOL["units_ratio"] and r["cosine"] > TOL["units_cos"]
                              for r in ts)
    return dict(rows=rows, fd_consistency=fdc, displacement_sigma_A=0.03, **{"pass": ok})


# ================================================================ overhead and recalc
def t_overhead(spec, sock, picks):
    s, p, pairs = label(picks[0])
    w = work("overhead")
    (w / "c_EXT.xyz").write_text(f"{len(s)}\n\n" + B.xyz_lines(s, p))
    (w / "c_EXT.extinp.tmp").write_text("c_EXT.xyz   # xyz\n0   # charge\n1   # mult\n1   # ncores\n1   # do gradient\n")
    exe = B.client_exe(Path(sock).parent)
    n = 40
    t0 = time.time()
    for _ in range(n):
        subprocess.run([str(exe), "c_EXT.extinp.tmp", "--socket", sock], cwd=w, check=True)
    ours = (time.time() - t0) / n
    st = B.ext_log_stats(w / "mlts_ext.log")
    res = dict(rxn_id=picks[0]["rxn_id"], n_atoms=len(s), n_calls=n, client_wall_ms_per_call=1e3 * ours,
               eval_ms_median=st.get("eval_ms_median"),
               client_overhead_ms_per_call=1e3 * ours - (st.get("eval_ms_median") or 0.0))
    om = Path(sys.executable).parent / ("orca-mlips-mace" if spec["backend"] == "mace" else "orca-mlips-aimnet2")
    if om.is_file():
        k = 10
        t0 = time.time()
        rr = None
        for _ in range(k):
            rr = subprocess.run([str(om), "c_EXT.extinp.tmp", "--server-socket", sock], cwd=w, capture_output=True,
                                text=True)
        res["orca_mlips_client_wall_ms_per_call"] = 1e3 * (time.time() - t0) / k
        res["orca_mlips_client_rc"] = rr.returncode if rr else None
        if rr is not None and rr.returncode:
            res["orca_mlips_client_err"] = rr.stderr[-400:]
    o = B.optimize_ts(spec, s, p, pairs, work("overhead_opt"), sock)
    res["optts_from_label"] = {k: o.get(k) for k in ("status", "n_cycles", "n_ext_calls", "wall_s", "orca_wall_s",
                                                     "ext_client_wall_s", "orca_side_ms_per_call", "gpu_s", "n_hess_calls")}
    shutil.rmtree(w, ignore_errors=True)
    return res


def t_recalc(spec, sock, picks, modes=("none", "analytic:30", "numeric:30")):
    """the 2 smallest picks from their G2 TS, each recalc mode in its own thread (ORCA processes share the server)."""
    from concurrent.futures import ThreadPoolExecutor
    jobs = []
    for rx in sorted(picks, key=lambda x: x["n_atoms"])[:2]:
        s, p, pairs = label(rx)
        gp = np.asarray(C.read_xyz(C.g2_ts_label_order(rx["rxn_id"]))[1], float)
        for mode in modes:
            jobs.append((rx, s, p, pairs, gp, mode))

    def one(j):
        rx, s, p, pairs, gp, mode = j
        o = B.optimize_ts(spec, s, gp, pairs, work(f"recalc_{rx['rxn_id']}_{mode.replace(':', '_')}"), sock, recalc=mode)
        d_lab = B.dists(p, pairs)
        return dict(rxn_id=rx["rxn_id"], n_atoms=len(s), recalc=mode,
                    start_dd_A=float(np.max(np.abs(np.subtract(B.dists(gp, pairs), d_lab)))),
                    **{k: o.get(k) for k in ("status", "converged", "n_cycles", "n_segments", "n_ext_calls",
                                             "n_hess_calls", "wall_s", "orca_wall_s", "gpu_s", "n_imag")},
                    dd_max_vs_label_A=float(np.max(np.abs(np.subtract(B.dists(o["pos_A"], pairs), d_lab)))))
    with ThreadPoolExecutor(len(jobs)) as ex:
        rows = list(ex.map(one, jobs))
    return dict(start="g2", rows=rows, default=B.RECALC_DEFAULT)


def t_optimizer_diag(spec, sock, picks, rxn_id):
    """one reaction that missed (ii): ORCA VeryTightOpt vs Sella fmax 1e-4 eV/A (flat-PES / tolerance diagnosis)."""
    rx = next(x for x in picks if x["rxn_id"] == int(rxn_id))
    s, p, pairs = label(rx)
    o = B.optimize_ts(spec, s, p, pairs, work(f"diag_{rx['rxn_id']}"), sock, tight="verytight")
    sl = sella_ts(spec, s, p, fmax=1e-4, steps=1000)
    row = dict(rxn_id=rx["rxn_id"], orca_verytight=dict(converged=o["converged"], n_cycles=o["n_cycles"],
                                                         d_form=B.dists(o["pos_A"], pairs), n_imag=o.get("n_imag"),
                                                         gmax_eh_bohr=o.get("gmax_final_eh_bohr"), e_eh=o.get("e_final_eh")))
    if "error" not in sl:
        row["sella_1e-4"] = dict(converged=sl["converged"], steps=sl["steps"], d_form=B.dists(sl["pos"], pairs),
                                 fmax_final=sl["fmax_final"], e_eh=B.direct_engrad(spec, s, sl["pos"])[0])
        row["dform_max_A"] = float(np.max(np.abs(np.subtract(row["orca_verytight"]["d_form"], row["sella_1e-4"]["d_form"]))))
        row["dE_kcal"] = (row["orca_verytight"]["e_eh"] - row["sella_1e-4"]["e_eh"]) * C.EH2KCAL
        row["pass"] = row["dform_max_A"] < TOL["optimizer_dform_A"]
    else:
        row["sella_1e-4"] = sl
    return row


SECTIONS = ("bridge", "optimizer", "hessian", "units", "overhead", "recalc", "optimizer_diag")


def main(argv=None):
    a = list(sys.argv[1:] if argv is None else argv)
    if not a or a[0] not in SECTIONS + ("all",):
        print(__doc__)
        return 2
    secs = SECTIONS[:-1] if a[0] == "all" else (a[0],)
    spec = B.resolve_spec(a[1] if len(a) > 1 else "MACE-OMOL-0")
    name = B.spec_name(spec)
    picks = pick()
    J = C.read_json(OUT) if OUT.is_file() else {}
    J["C1_pick"] = picks
    C.write_json(OUT, J)
    h = B.start_server(spec) if any(x != "units" for x in secs) else None
    sock = h.socket if h else None
    opt_pos = None
    rc = 0
    try:
        for sec in secs:
            t0 = time.time()
            try:
                if sec == "bridge":
                    d = t_bridge(spec, sock, picks)
                elif sec == "optimizer":
                    d, opt_pos = t_optimizer(spec, sock, picks)
                elif sec == "hessian":
                    d = t_hessian(spec, sock, picks, opt_pos)
                elif sec == "units":
                    d = t_units(spec, picks)
                elif sec == "overhead":
                    d = t_overhead(spec, sock, picks)
                elif sec == "optimizer_diag":
                    d = t_optimizer_diag(spec, sock, picks, a[2])
                else:
                    d = t_recalc(spec, sock, picks)
            except Exception as exc:                             # noqa: BLE001
                d = {"error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-2000:], "pass": False}
                rc = 1
            d.update(section_wall_s=time.time() - t0, host=os.uname().nodename, job=B.jobtag(), time=C.now(),
                     **B.device_info())
            if h:
                d["server"] = {k: v for k, v in (h.stats() or {}).items() if k not in ("spec",)}
            merge(name, sec, d)
            print(f"[tests_c1] {name} {sec} pass={d.get('pass')} wall={d['section_wall_s']:.0f}s", flush=True)
    finally:
        if h:
            h.shutdown()
    print(json.dumps(C.read_json(OUT)["models"][name], indent=1, default=C._json_default)[:6000])
    return rc


if __name__ == "__main__":
    sys.exit(main())
