#!/usr/bin/env python3
"""g1_geom.py — rev 4 G1 geometries (REV4_XTB_GEOMETRY.md §2, Phase 1).

G1 = GFN2-xTB / ALPB(water) re-optimisation started from the label geometries:
  TS          ORCA 6.1.1  `! XTB2 ALPB(water) OptTS Freq` from the Coley ts_file
  references  xtb 6.7.1   `--opt tight --alpb water --gfn 2` from the label's rel1_file / rel2_file (incl. _alt)
Atom order is kept, so the label's formed_pairs_ts, roles and partition (A_idx) apply unchanged.

  python g1_geom.py smoke                     # Phase 1-1: rxn 20, 105 + 3 random ok rxns (seed 20260930)
  python g1_geom.py run <slice> <n_slices>    # Phase 1-2: one interleaved slice of the 5,260 ok rxns
  python g1_geom.py summary <out_csv>         # Phase 1-2 report: success rate, failure types, RMSD, Δd_form

Per rxn -> $G1_ROOT/<rid>/ : ts.xyz rel1.xyz rel2.xyz result.json, then .done or .fail_<reason> (written last).
A rxn with a marker is skipped (idempotent); work/ keeps the ORCA / xtb inputs and outputs (.inp .out .hess).

Gates (the D1 functions, imported from a read-only snapshot of d1-build@d573111e in $D1_SNAPSHOT):
  run_ts.imag_gate        first imaginary frequency <= -40 cm-1, every other one > -50 cm-1
  run_ts.forming_share    share of the imaginary mode on the label formed_pairs_ts >= 0.5
  not_product_like        longer forming bond >= 1.6 A          (run_ts.analyse FIX F-A rule)
  forming_bond_present    shorter forming bond < 3.3 A
  no_foreign_bond         fragmenter.audit_forming_bonds: no other inter-fragment covalent contact
  partition_match         fragmenter.partition of the G1 TS == label A_idx (input_meta.csv)
  refN_graph_same         optimised reference heavy graph (element + H count) isomorphic to the DFT reference
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fragmenter as fr  # noqa: E402  (imported first: rmsd.py below reuses this module)

D1_SNAPSHOT = Path(os.environ.get("D1_SNAPSHOT", "/gpfs/tmp_cpu2/yeseo1ee/espley_rev4/d1_snapshot_d573111e"))
D1_COMMIT = "d573111eb899fd78c71e8ce4ac5c0f53553d987b"
sys.path.insert(0, str(D1_SNAPSHOT / "D1_build/d1_autode_pilot"))
sys.path.insert(0, str(D1_SNAPSHOT / "D1_build/validation/analysis"))
import orca_direct as od  # noqa: E402
import rmsd as rm  # noqa: E402
import run_ts  # noqa: E402

LABELS = Path(os.environ.get("ESPLEY_LABELS", HERE.parent.parent / "labels_all.json"))
PROF = Path(os.environ.get("ESPLEY_PROF", "/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/dipolar_cycloaddition/extracted/full_dataset_profiles"))
SMILES_CSV = Path(os.environ.get("ESPLEY_SMILES", "/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/dipolar_cycloaddition/full_dataset.csv"))
META = Path(os.environ.get("ESPLEY_META", "/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/label_true/work/input_meta.csv"))
G1_ROOT = Path(os.environ.get("G1_ROOT", "/gpfs/tmp_cpu2/yeseo1ee/espley_xtb_g1"))
ORCA_BIN = os.environ.get("ORCA_BIN", "/home1/yeseo1ee/orca_6_1_1_avx2/orca")
XTB_BIN = os.environ.get("XTB_BIN", "/home1/yeseo1ee/xtb-dist/bin/xtb")
EXCLUDE = {3090, 3766, 4252, 3400, 5783}           # = xtb_slice.EXCLUDE (labels_all non-ok)
SMOKE_FIXED = (20, 105)
SMOKE_SEED = 20260930
ORCA_NPROCS = 2
GATES = dict(min_imag_cm=-40.0, extra_imag_tol_cm=-50.0, mode_share_min=0.5,
             forming_min_A=1.6, no_bond_A=3.3, forming_max_A=3.0)
TS_HEADER = ("! XTB2 ALPB(water) OptTS Freq\n"
             "%geom Calc_Hess true Recalc_Hess 5 MaxIter 200 end\n"
             f"%pal nprocs {ORCA_NPROCS} end\n")
XTB_OPT = ["--opt", "tight", "--alpb", "water", "--gfn", "2"]
XTB_GRAD = ["--grad", "--alpb", "water", "--gfn", "2"]
BOHR = 0.529177210903                               # Å per bohr
EH2KCAL = 627.5094740631
TOTAL_E_RE = re.compile(r"TOTAL ENERGY\s+(-?\d+\.\d+)\s+Eh")


# ---------------------------------------------------------------- inputs
def load_labels():
    labels = json.load(open(LABELS))
    ok = sorted((d for d in labels if int(d["rxn_id"]) not in EXCLUDE), key=lambda d: int(d["rxn_id"]))
    assert all(d["status"] == "ok" for d in ok), "a non-excluded label is not status ok"
    return ok


def load_smiles():
    df = pd.read_csv(SMILES_CSV)
    return dict(zip(df.rxn_id.astype(int), df.rxn_smiles))


def load_meta():
    m = pd.read_csv(META)
    return {int(r.rxn_id): sorted(int(x) for x in str(r.A_idx).split()) for r in m.itertuples()}


def smoke_ids(ok_ids):
    rng = np.random.default_rng(SMOKE_SEED)
    pool = sorted(set(ok_ids) - set(SMOKE_FIXED))
    return list(SMOKE_FIXED) + sorted(int(i) for i in rng.choice(pool, size=3, replace=False))


# ---------------------------------------------------------------- engines
def xtb_run(args, wd: Path, log: str, threads: int = ORCA_NPROCS) -> str:
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), OMP_STACKSIZE="4G")
    r = subprocess.run([XTB_BIN, *args], cwd=wd, capture_output=True, text=True, env=env, timeout=6 * 3600)
    (wd / log).write_text(r.stdout + "\n#### stderr\n" + r.stderr)
    return r.stdout


def xtb_opt(syms, xyz, charge, wd: Path):
    """(syms, xyz, E_Eh) of the xtb --opt tight minimum; None if not converged. Idempotent on xtbopt.xyz."""
    wd.mkdir(parents=True, exist_ok=True)
    out = wd / "opt.out"
    if not ((wd / "xtbopt.xyz").is_file() and out.is_file()
            and "GEOMETRY OPTIMIZATION CONVERGED" in out.read_text(errors="replace")):
        fr_write_xyz(wd / "start.xyz", syms, xyz)
        xtb_run(["start.xyz", *XTB_OPT, "--chrg", str(int(charge))], wd, "opt.out")
    text = out.read_text(errors="replace")
    if "GEOMETRY OPTIMIZATION CONVERGED" not in text or not (wd / "xtbopt.xyz").is_file():
        return None
    s, x = fr.read_xyz(wd / "xtbopt.xyz")
    e = TOTAL_E_RE.findall(text)
    return s, x, float(e[-1]) if e else None


def xtb_grad(syms, xyz, charge, wd: Path):
    """(E_Eh, gradient (N,3) Eh/bohr) from xtb 6.7.1 at a fixed geometry (engine-parity check)."""
    wd.mkdir(parents=True, exist_ok=True)
    fr_write_xyz(wd / "g.xyz", syms, xyz)
    text = xtb_run(["g.xyz", *XTB_GRAD, "--chrg", str(int(charge))], wd, "grad.out")
    lines = (wd / "gradient").read_text().splitlines()
    k = next(i for i, l in enumerate(lines) if l.strip().startswith("$grad"))
    n = len(syms)
    g = np.array([[float(v.replace("D", "E")) for v in l.split()[:3]] for l in lines[k + 2 + n:k + 2 + 2 * n]])
    e = TOTAL_E_RE.findall(text)
    return (float(e[-1]) if e else None), g


def fr_write_xyz(path, syms, xyz, title=""):
    body = "\n".join(f"{s:<2} {x:15.8f} {y:15.8f} {z:15.8f}" for s, (x, y, z) in zip(syms, np.asarray(xyz)))
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(f"{len(syms)}\n{title}\n{body}\n")
    os.replace(tmp, path)


def hess_atoms(path):
    """(syms, xyz Å) from the $atoms block of an ORCA .hess file."""
    lines = Path(path).read_text().splitlines()
    i = next(k for k, l in enumerate(lines) if l.strip() == "$atoms")
    n = int(lines[i + 1].split()[0])
    rows = [lines[i + 2 + k].split() for k in range(n)]
    return [r[0] for r in rows], np.array([[float(v) for v in r[2:5]] for r in rows]) * BOHR


def orca_level_lines(wd: Path):
    """Verbatim lines of every ORCA text output that show the method and the solvent model."""
    pat = re.compile(r"(GFN|gfn|xTB|XTB|xtb|ALPB|alpb|[Ss]olvent|[Ww]ater|otool)")
    got = []
    for f in sorted(wd.glob("*")):
        if f.suffix in (".out", ".log", ".inp") or "xtb" in f.name.lower():
            try:
                for l in f.read_text(errors="replace").splitlines():
                    if pat.search(l) and l.strip() and len(got) < 200:
                        got.append(f"{f.name}: {l.strip()}")
            except (IsADirectoryError, UnicodeDecodeError):
                pass
    seen, uniq = set(), []
    for l in got:
        if l not in seen:
            seen.add(l); uniq.append(l)
    return uniq


def heavy_graph_same(a, b) -> bool:
    Ga = fr.heavy_graph(*a)[0]; Gb = fr.heavy_graph(*b)[0]
    return nx.is_isomorphic(Ga, Gb, node_match=fr._nm)


def heavy_rmsd_same_order(a_xyz, b_xyz, heavy):
    return rm.kabsch_rmsd(np.asarray(a_xyz)[heavy], np.asarray(b_xyz)[heavy])


# ---------------------------------------------------------------- one reaction
def process(label, smiles, meta, root: Path, engine_check: bool = False):
    rid = int(label["rxn_id"])
    d = root / str(rid)
    if (d / ".done").is_file() or list(d.glob(".fail_*")):
        return rid, "skip"
    d.mkdir(parents=True, exist_ok=True)
    wd = d / "work"; wd.mkdir(exist_ok=True)
    q1, q2 = int(label.get("charge1") or 0), int(label.get("charge2") or 0)
    res = dict(rxn_id=rid, charge1=q1, charge2=q2, gates_cfg=GATES, ts_header=TS_HEADER, xtb_opt=XTB_OPT,
               d1_snapshot_commit=D1_COMMIT, orca_bin=ORCA_BIN, xtb_bin=XTB_BIN, host=os.uname().nodename)
    reasons = []
    t0 = time.time()
    try:
        pdir = PROF / str(rid)
        ts0 = fr.read_xyz(pdir / label["ts_file"])
        rel0 = [fr.read_xyz(pdir / label[k]) for k in ("rel1_file", "rel2_file")]
        res.update(ts_file=label["ts_file"], rel1_file=label["rel1_file"], rel2_file=label["rel2_file"])
        pairs = [tuple(sorted(map(int, p.split("-")))) for p in str(label["formed_pairs_ts"]).split()]
        heavy = [i for i, s in enumerate(ts0[0]) if s != "H"]

        # ---- TS: ORCA XTB2 ALPB(water) OptTS Freq from the Coley TS
        inp = wd / "ts.inp"
        if not inp.is_file():
            od.write_inp(inp, TS_HEADER, *ts0, charge=q1 + q2, mult=1)
        t1 = time.time()
        text = od.run({"orca_bin": ORCA_BIN}, inp).read_text(errors="replace")
        res["t_ts_s"] = time.time() - t1
        res.update(ts_terminated=od.terminated(text), ts_converged=od.opt_converged(text),
                   e_ts_eh=od.last_fspe(text), ts_run_hours=od.run_hours(text),
                   n_opt_cycles=len(re.findall(r"GEOMETRY OPTIMIZATION CYCLE\s+\d+", text)))
        if engine_check and (wd / "ts.xyz").is_file():
            # engine parity: xtb 6.7.1 gradient at ORCA's last geometry (pass criterion uses converged TSs only)
            e_x, g = xtb_grad(*fr.read_xyz(wd / "ts.xyz"), q1 + q2, wd / "grad")
            res.update(engine_xtb_e_eh=e_x, engine_maxgrad_eh_bohr=float(np.abs(g).max()),
                       engine_de_eh=None if e_x is None or res["e_ts_eh"] is None else e_x - res["e_ts_eh"])
            res["orca_level_lines"] = orca_level_lines(wd)
        if not res["ts_terminated"]:
            raise Fail("orca_not_terminated")
        if not res["ts_converged"]:
            raise Fail("optts_not_converged")
        if not (wd / "ts.xyz").is_file() or not (wd / "ts.hess").is_file():
            raise Fail("orca_missing_xyz_or_hess")
        ts1 = fr.read_xyz(wd / "ts.xyz")
        # the .hess $atoms block is the final geometry shifted to the centre of mass (smoke 2026-09-30:
        # pure translations of 0.03-12.5 A); the frequencies equal the last Freq block of the output
        hs, hx = hess_atoms(wd / "ts.hess")
        shift = hx - ts1[1]
        res["hess_geom_max_dev_A"] = float(np.abs(shift - shift.mean(0)).max())
        res["hess_shift_A"] = shift.mean(0).tolist()
        last = text.rsplit("VIBRATIONAL FREQUENCIES", 1)[-1]
        f_out = [float(v) for v in re.findall(r"^\s*\d+:\s+(-?\d+\.\d+) cm\*\*-1", last, re.M)]
        f_hess = od.parse_hess(wd / "ts.hess")[0]
        res["hess_freq_max_dev_cm"] = (float(np.abs(np.asarray(f_out) - f_hess).max())
                                       if len(f_out) == len(f_hess) else None)
        if (hs != ts1[0] or res["hess_geom_max_dev_A"] > 1e-4 or res["hess_freq_max_dev_cm"] is None
                or res["hess_freq_max_dev_cm"] > 0.01):
            raise Fail("hess_not_at_final_geometry")
        imag, mode = od.imag_and_mode(wd / "ts.hess")
        res.update(imag_freqs_cm=imag, n_imag=len(imag))

        # ---- references: xtb --opt tight from the label references
        t1 = time.time()
        rel1 = []
        for k in (0, 1):
            got = xtb_opt(*rel0[k], (q1, q2)[k], wd / f"ref{k + 1}")
            if got is None:
                raise Fail(f"ref{k + 1}_opt_not_converged")
            rel1.append((got[0], got[1])); res[f"e_rel{k + 1}_eh"] = got[2]
            res[f"ref{k + 1}_graph_same"] = heavy_graph_same(rel0[k], rel1[k])
            hk = [i for i, s in enumerate(rel0[k][0]) if s != "H"]
            res[f"rmsd_rel{k + 1}_heavy_A"] = heavy_rmsd_same_order(rel0[k][1], rel1[k][1], hk)
        res["t_ref_s"] = time.time() - t1
        for k in (1, 2):
            if not res[f"ref{k}_graph_same"]:
                reasons.append(f"ref{k}_graph_changed")

        # ---- geometry change vs the DFT TS (same atom order)
        res["rmsd_ts_heavy_A"] = heavy_rmsd_same_order(ts0[1], ts1[1], heavy)
        D0, D1 = fr.dist_matrix(ts0[1]), fr.dist_matrix(ts1[1])
        order = sorted(pairs, key=lambda p: D0[p])                   # bond 1 = shorter at the DFT TS
        res.update(formed_pairs_label=[list(p) for p in order],
                   d_form_dft=[float(D0[p]) for p in order], d_form_g1=[float(D1[p]) for p in order])
        res["dd_form"] = [b - a for a, b in zip(res["d_form_dft"], res["d_form_g1"])]
        if all(e is not None for e in (res["e_ts_eh"], res["e_rel1_eh"], res["e_rel2_eh"])):
            res["xtb_barrier_g1_kcal"] = (res["e_ts_eh"] - res["e_rel1_eh"] - res["e_rel2_eh"]) * EH2KCAL

        # ---- gates
        ichecks, ihard = run_ts.imag_gate(imag, GATES)
        res["imag_checks"] = ichecks
        reasons += ihard
        share = None if mode is None else run_ts.forming_share(ts1[1], mode, order, heavy)
        res["imag_mode_forming_share"] = share
        if share is None or share < GATES["mode_share_min"]:
            reasons.append("imag_mode_on_forming_bonds")
        dg = res["d_form_g1"]
        res.update(not_product_like=max(dg) >= GATES["forming_min_A"], forming_bond_present=min(dg) < GATES["no_bond_A"],
                   flag_short_forming=min(dg) < GATES["forming_min_A"], flag_async=max(dg) > GATES["forming_max_A"])
        reasons += [k for k in ("not_product_like", "forming_bond_present") if not res[k]]

        P = fr.partition(ts1, rel1[0], rel1[1])
        res["partition_status"] = P.status
        if P.status != "ok":
            reasons.append(f"partition_{P.status}")
        else:
            a_g1 = sorted(P.A)
            P0 = fr.partition(ts0, rel0[0], rel0[1])
            a_lab = meta.get(rid)
            res.update(A_idx_g1=a_g1, A_idx_label=a_lab, A_idx_dft=sorted(P0.A) if P0.status == "ok" else None)
            ref_a = a_lab if a_lab is not None else res["A_idx_dft"]
            res["partition_match"] = a_g1 == ref_a
            if not res["partition_match"]:
                reasons.append("partition_mismatch")
            audit = fr.audit_forming_bonds(P, ts1, rel1[0], rel1[1], smiles[rid])
            if audit is None:
                reasons.append("forming_bond_map_failed")
            else:
                res.update(foreign=audit.foreign, formed_pairs_audit=sorted(list(p) for p in audit.formed_ts))
                res["formed_pairs_audit_eq_label"] = {tuple(p) for p in audit.formed_ts} == set(order)
                if audit.foreign:
                    reasons.append("no_foreign_bond")

        fr_write_xyz(d / "ts.xyz", *ts1, f"G1 TS rxn {rid}")
        for k in (0, 1):
            fr_write_xyz(d / f"rel{k + 1}.xyz", *rel1[k], f"G1 reference {k + 1} rxn {rid}")
    except Fail as e:
        reasons.insert(0, str(e))
    except Exception as e:                                              # noqa: BLE001
        res["traceback"] = traceback.format_exc()
        reasons.insert(0, f"error_{type(e).__name__}")
    res["wall_s"] = time.time() - t0
    res["fail_reasons"] = reasons
    res["status"] = "ok" if not reasons else "fail:" + reasons[0]
    od_write_json(d / "result.json", res)
    for f in ("hess", "gbw", "densities", "engrad", "opt", "cis", "tmp", "xtbrestart", "charges", "wbo"):
        for p in wd.glob(f"*.{f}"):
            if p.name != "ts.hess":
                p.unlink(missing_ok=True)
    marker = d / (".done" if not reasons else f".fail_{re.sub(r'[^A-Za-z0-9_]+', '_', reasons[0])}")
    marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n" + "\n".join(reasons) + "\n")
    return rid, res["status"]


class Fail(Exception):
    pass


def od_write_json(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))
    os.replace(tmp, path)


# ---------------------------------------------------------------- drivers
def run_many(labels, root, workers, engine_check=False):
    smiles, meta = load_smiles(), load_meta()
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(process, lab, smiles, meta, root, engine_check) for lab in labels]
        for k, f in enumerate(futs):
            rid, st = f.result()
            print(f"[{k + 1}/{len(futs)}] rxn {rid}: {st}  ({(time.time() - t0) / 60:.1f} min)", flush=True)


def smoke():
    ok = load_labels()
    ids = smoke_ids([int(d["rxn_id"]) for d in ok])
    byid = {int(d["rxn_id"]): d for d in ok}
    root = G1_ROOT / "_smoke"
    print("smoke rxns:", ids, flush=True)
    otool = Path(ORCA_BIN).parent / "otool_xtb"
    ver = subprocess.run([str(otool), "--version"], capture_output=True, text=True)
    xver = subprocess.run([XTB_BIN, "--version"], capture_output=True, text=True)
    run_many([byid[i] for i in ids], root, workers=max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "2")) // ORCA_NPROCS),
             engine_check=True)
    rep = dict(rxns=ids, otool_xtb_version=[l for l in (ver.stdout + ver.stderr).splitlines() if "version" in l.lower()][:3],
               xtb_version=[l for l in (xver.stdout + xver.stderr).splitlines() if "version" in l.lower()][:3], per_rxn={})
    for i in ids:
        r = json.load(open(root / str(i) / "result.json"))
        rep["per_rxn"][i] = {k: r.get(k) for k in ("status", "fail_reasons", "imag_freqs_cm", "imag_mode_forming_share",
                                                    "engine_maxgrad_eh_bohr", "engine_de_eh", "rmsd_ts_heavy_A", "dd_form",
                                                    "rmsd_rel1_heavy_A", "rmsd_rel2_heavy_A", "partition_match",
                                                    "n_opt_cycles", "t_ts_s", "t_ref_s", "wall_s", "ts_converged",
                                                    "hess_geom_max_dev_A", "hess_shift_A", "hess_freq_max_dev_cm")}
        rep["per_rxn"][i]["orca_level_lines"] = r.get("orca_level_lines", [])[:40]
    conv = {i: json.load(open(root / str(i) / "result.json")).get("ts_converged") for i in ids}
    g = [rep["per_rxn"][i]["engine_maxgrad_eh_bohr"] for i in ids if conv[i]]
    rep["engine_check_n_converged"] = len(g)
    rep["engine_check_pass"] = len(g) >= 3 and all(v is not None and v < 5e-4 for v in g)
    od_write_json(root / "smoke_report.json", rep)
    print(json.dumps(rep, indent=1, default=str))


def run_slice(slice_id, n_slices):
    ok = load_labels()
    mine = ok[slice_id::n_slices]
    workers = int(os.environ.get("G1_WORKERS", max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "2")) // ORCA_NPROCS)))
    print(f"[slice {slice_id}/{n_slices}] {len(mine)} rxns, {workers} workers x ORCA nprocs {ORCA_NPROCS}", flush=True)
    run_many(mine, G1_ROOT, workers)


def summary(out_csv):
    ok = load_labels()
    rows = []
    for lab in ok:
        rid = int(lab["rxn_id"]); f = G1_ROOT / str(rid) / "result.json"
        if not f.is_file():
            rows.append(dict(rxn_id=rid, status="missing")); continue
        r = json.load(open(f))
        dd = r.get("dd_form") or [np.nan, np.nan]
        rows.append(dict(rxn_id=rid, status=r["status"], fail_reasons=";".join(r.get("fail_reasons", [])),
                         imag1_cm=(r.get("imag_freqs_cm") or [np.nan])[0], n_imag=r.get("n_imag"),
                         forming_share=r.get("imag_mode_forming_share"), rmsd_ts_heavy_A=r.get("rmsd_ts_heavy_A"),
                         dd_form_short_A=dd[0], dd_form_long_A=dd[1],
                         d_form_g1_short_A=(r.get("d_form_g1") or [np.nan])[0],
                         d_form_g1_long_A=(r.get("d_form_g1") or [np.nan, np.nan])[1],
                         rmsd_rel1_heavy_A=r.get("rmsd_rel1_heavy_A"), rmsd_rel2_heavy_A=r.get("rmsd_rel2_heavy_A"),
                         partition_match=r.get("partition_match"), flag_async=r.get("flag_async"),
                         flag_short_forming=r.get("flag_short_forming"), n_opt_cycles=r.get("n_opt_cycles"),
                         wall_s=r.get("wall_s"), xtb_barrier_g1_kcal=r.get("xtb_barrier_g1_kcal")))
    df = pd.DataFrame(rows)
    tmp = Path(str(out_csv) + ".tmp"); df.to_csv(tmp, index=False); os.replace(tmp, out_csv)
    n = len(df); okm = df.status == "ok"
    rep = dict(n=n, n_ok=int(okm.sum()), success_rate=float(okm.mean()),
               n_missing=int((df.status == "missing").sum()),
               fail_counts=df.loc[~okm, "status"].value_counts().to_dict(),
               all_reason_counts=pd.Series([x for s in df.fail_reasons.dropna() for x in str(s).split(";") if x])
               .value_counts().to_dict(),
               partition_mismatch_rate=float((df.partition_match == False).sum() / n),     # noqa: E712
               core_h_total=float(df.wall_s.sum() * ORCA_NPROCS / 3600))
    for c in ("rmsd_ts_heavy_A", "dd_form_short_A", "dd_form_long_A", "rmsd_rel1_heavy_A", "rmsd_rel2_heavy_A"):
        v = df.loc[okm, c].astype(float)
        a = v.abs() if c.startswith("dd") else v
        rep[c] = dict(median=float(v.median()), p90_abs=float(a.quantile(0.9)), median_abs=float(a.median()),
                      max_abs=float(a.max()))
    rep["stop_success_lt_90pct"] = rep["success_rate"] < 0.90
    rep["stop_partition_mismatch_gt_1pct"] = rep["partition_mismatch_rate"] > 0.01
    od_write_json(Path(str(out_csv).replace(".csv", ".json")), rep)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "smoke":
        smoke()
    elif cmd == "run":
        run_slice(int(sys.argv[2]), int(sys.argv[3]))
    elif cmd == "summary":
        summary(sys.argv[2])
    else:
        raise SystemExit(__doc__)
