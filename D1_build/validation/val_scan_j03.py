#!/usr/bin/env python3
"""val_scan_j03.py — V7 (VALIDATION_SPEC §6.6, A10): is the pilot J03 reaction really barrierless?

  python val_scan_j03.py --src J03 --task V7_J03
1. product geometry = the pilot autodE output p0_*.xyz (copied from the read-only pilot scratch)
2. forming-bond atoms: stereo_utils.template_match(product, mapped product SMILES) -> geometry indices
3. ORCA relaxed 2D scan at the L0 Opt level (autodE's Opt header): both forming bonds 1.60 -> 3.40 A,
   10 points each (100 constrained optimisations); energies from scan.relaxscanact.dat (fallback: the
   per-step constrained values and energies in scan.out)
4. minimax path (4-neighbour bottleneck Dijkstra) from (1.6, 1.6) to (3.4, 3.4):
   path maximum at the reactant end and no interior maximum -> no saddle point (confirmed);
   otherwise OptTS (L0, modify_internal on the two bonds) from the highest interior maximum, then the
   pilot TS gate (run_ts.analyse, F1): pass -> reclassified "autodE search failure".
Writes <scratch>/v7/<src>/result.json for a finished analysis (including "OptTS did not converge" or
"gate failed", which are scientific results); FIX F-C: missing scan data -> result_error.json + exit 1.
"""
from __future__ import annotations

import argparse
import heapq
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "d1_autode_pilot"))
import pilot_common as pc  # noqa: E402
import orca_direct as od  # noqa: E402
import stereo_utils as su  # noqa: E402

STEP_RE = re.compile(r"RELAXED SURFACE SCAN STEP\s+(\d+)")
BOND_RE = re.compile(r"Bond \(\s*(\d+),\s*(\d+)\)\s*:\s*(-?\d+\.\d+)")


def forming_atoms(prod, mapped_rxn, fr):
    _, formed, _ = fr.smiles_reactants_and_formed(mapped_rxn)
    psmi = mapped_rxn.split(">>")[1]
    _, t2g, tmpl = su.template_match(su.xyz_block_from_atoms(*prod), psmi, 0)
    g = {a.GetAtomMapNum(): t2g[a.GetIdx()] for a in tmpl.GetAtoms() if a.GetAtomMapNum()}
    return sorted(tuple(sorted((g[u], g[v]))) for u, v in formed)


def read_grid(wd: Path, start, end, n):
    """{(i, j): (E_eh, step_no)} with i, j the grid indices of bond 1 and bond 2."""
    step = (end - start) / (n - 1)
    gi = lambda x: int(round((x - start) / step))            # noqa: E731
    out = {}
    dat = wd / "scan.relaxscanact.dat"
    rows = [[float(x) for x in l.split()] for l in dat.read_text().splitlines() if l.strip()] if dat.is_file() else []
    if rows and all(len(r) == 3 for r in rows):
        for k, (b1, b2, e) in enumerate(rows, start=1):
            out[(gi(b1), gi(b2))] = (e, k)
        return out, "relaxscanact.dat"
    text = (wd / "scan.out").read_text(errors="replace")
    parts = STEP_RE.split(text)
    for k in range(1, len(parts) - 1, 2):
        no, body = int(parts[k]), parts[k + 1]
        vals = [float(m[2]) for m in BOND_RE.findall(body)[:2]]
        e = od.last_fspe(body)
        if len(vals) == 2 and e is not None:
            out[(gi(vals[0]), gi(vals[1]))] = (e, no)
    return out, "scan.out"


def minimax_path(E: np.ndarray):
    """Bottleneck shortest path on the grid from (0,0) to (n-1,n-1); NaN cells are impassable."""
    n = E.shape[0]
    best = np.full(E.shape, np.inf); prev = {}
    best[0, 0] = E[0, 0]
    pq = [(E[0, 0], 0, 0)]
    while pq:
        b, i, j = heapq.heappop(pq)
        if b > best[i, j]:
            continue
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            u, v = i + di, j + dj
            if 0 <= u < n and 0 <= v < n and not np.isnan(E[u, v]):
                nb = max(b, E[u, v])
                if nb < best[u, v]:
                    best[u, v] = nb; prev[(u, v)] = (i, j)
                    heapq.heappush(pq, (nb, u, v))
    path, p = [(n - 1, n - 1)], (n - 1, n - 1)
    while p != (0, 0):
        if p not in prev:
            return None
        p = prev[p]; path.append(p)
    return path[::-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--task", required=True)
    a = ap.parse_args()
    cfg = pc.load_config()
    fr, _ = pc.import_repo_modules(cfg)
    wd = Path(cfg["scratch"]) / "v7" / a.src; wd.mkdir(parents=True, exist_ok=True)
    if (wd / "result.json").is_file():
        print(f"{a.task}: done"); return
    job = pd.read_csv(cfg["val"]["pilot_manifest"], keep_default_na=False).set_index("job_id").loc[a.src].to_dict()
    job["job_id"] = a.src
    outdir = Path(cfg["pilot_scratch"]) / "jobs" / a.src / "ts" / a.src / "output"
    for f in outdir.glob("*.xyz"):                                  # read-only pilot -> copy
        if not (wd / f.name).is_file():
            shutil.copy(f, wd / f.name)
    prod = pc.read_xyz(sorted(wd.glob("p0_*.xyz"))[0])
    pairs = forming_atoms(prod, job["mapped_smiles"], fr)
    s = cfg["val"]["scan"]
    res = dict(task=a.task, src=a.src, forming_pairs=pairs, scan=s)
    inp = wd / "scan.inp"
    if not inp.is_file():
        (i, j), (k, l) = pairs
        od.write_inp(inp, od.opt_header(cfg, pc.n_cores(), od.scan_block(i, j, k, l, s["start"], s["end"], s["n"])),
                     *prod, charge=int(job["charge"]), mult=int(job["mult"]))
    text = od.run(cfg, inp).read_text(errors="replace")
    res["scan_hours"] = od.run_hours(text)

    def error(msg):      # FIX F-C: result.json only for a finished scan; a retry overwrites result_error.json
        res["error"] = msg
        (wd / "result_error.json").write_text(json.dumps(res, indent=1, default=str))
        print(msg, file=sys.stderr); sys.exit(1)

    grid, source = read_grid(wd, float(s["start"]), float(s["end"]), int(s["n"]))
    if not grid:
        error(f"no scan points parsed (ORCA terminated normally: {od.terminated(text)})")
    n = int(s["n"])
    E = np.full((n, n), np.nan)
    for (i, j), (e, _) in grid.items():
        if 0 <= i < n and 0 <= j < n:
            E[i, j] = e
    Ek = (E - np.nanmin(E)) * pc.EH_TO_KCAL
    res.update(grid_source=source, n_points=int(np.isfinite(E).sum()), grid_kcal=np.round(Ek, 3).tolist(),
               orca_terminated=od.terminated(text))
    path = minimax_path(Ek)
    if path is None:
        error("scan incomplete: no connected path")
    pe = [float(Ek[p]) for p in path]
    interior = [q for q in range(1, len(pe) - 1) if pe[q] > pe[q - 1] and pe[q] > pe[q + 1]]
    res.update(path=path, path_kcal=pe, path_argmax=int(np.argmax(pe)), interior_maxima=[path[q] for q in interior])
    if int(np.argmax(pe)) == len(pe) - 1 and not interior:
        res["verdict"] = "no_saddle_confirmed"
    else:
        q = max(interior, key=lambda q: pe[q]) if interior else int(np.argmax(pe))
        step_no = grid[tuple(path[q])][1]
        res["optts_start"] = dict(grid_point=path[q], step=step_no, rel_kcal=pe[q])
        att = optts_attempt(cfg, job, wd, step_no, pairs, fr)
        if att.pop("error", False):
            res.update(att); error(att["verdict"])
        res.update(att)          # "did not converge" / "gate failed" / "reclassified" are scientific results
    (wd / "result.json").write_text(json.dumps(res, indent=1, default=str))
    print(res.get("verdict"))


def optts_attempt(cfg, job, wd, step_no, pairs, fr):
    """L0 OptTS from scan step `step_no`, then the pilot TS gate (run_ts.analyse with F1)."""
    import run_ts
    geo = wd / f"scan.{step_no:03d}.xyz"
    if not geo.is_file():
        return dict(error=True, verdict=f"interior maximum, but the scan geometry {geo.name} is missing")
    ad = wd / "optts"; ad.mkdir(exist_ok=True)
    inp = ad / "reopt.inp"
    if not inp.is_file():
        od.write_inp(inp, od.level_header(cfg, "L0", pairs, n_cores=pc.n_cores()), *pc.read_xyz(geo),
                     charge=int(job["charge"]), mult=int(job["mult"]))
    text = od.run(cfg, inp).read_text(errors="replace")
    if not (od.terminated(text) and od.opt_converged(text)):
        return dict(verdict="interior maximum, OptTS did not converge: not reclassified")
    imag, mode = od.imag_and_mode(ad / "reopt.hess")
    reacs = [pc.read_xyz(p) for p in sorted(wd.glob("r*_*.xyz"))]
    got = dict(ts=pc.read_xyz(ad / "reopt.xyz"), reacs=reacs, prod=None, imag=imag, mode=mode)
    r = {}
    try:
        run_ts.analyse(cfg, job, ad, r, got, fr)
    except run_ts.StageFail as e:
        return dict(verdict=f"interior maximum, OptTS converged but the TS gate failed ({e}): not reclassified",
                    optts_imag=imag, optts_checks=r.get("checks"))
    return dict(verdict="reclassified: a TS exists (autodE search failure)", optts_imag=imag,
                optts_checks=r.get("checks"), optts_formed_d=r.get("formed_d"))


if __name__ == "__main__":
    main()
