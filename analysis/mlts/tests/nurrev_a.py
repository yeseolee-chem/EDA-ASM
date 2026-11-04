#!/usr/bin/env python3
"""tests/nurrev_a.py — LENS A independent review probe of inv_nurion.py (mlts-nur-2). Sandbox / compute node only.
READ ONLY on the uploaded tar, $MLT/dft/nurion, $LAD; writes only /gpfs/tmp_cpu2/yeseo1ee/mlts/tmp/nurrev/<tar>_a.json.
Independent: does not import inv_nurion / inv_dft / orca_io / mlt_common (own ORCA .out / .hess / .inp / extxyz parsers).

  nurrev_a.py <tar_stem> <J1,J2,...>

Per sampled job: stream the outer tar ('r|'), unpack archive.tgz (+ nested .tar) in memory, write ORCA files to node-local
$TMPDIR, parse every .out with a cycle-segment parser, classify level from the .inp (fallback: echo), and compare
step by step with the tar's nurion.extxyz.gz rows (struct_id / geometry key), the label TS Hessian npz and
hess_other.json; inventory of everything in the job that is not extracted."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import re
import shlex
import shutil
import sys
import tarfile
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

UP = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
NUR = Path("/gpfs/tmp_cpu2/yeseo1ee/mlts/dft/nurion")
LAD = Path("/gpfs/tmp_cpu2/yeseo1ee/d1_ladder")
D1J = Path("/gpfs/tmp_cpu2/yeseo1ee/d1_pair/d1_jobs")
SF = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-mlts/analysis/d1_pair/results/S_folds.csv")
RES = Path("/gpfs/tmp_cpu2/yeseo1ee/mlts/tmp/nurrev")
EH2EV = 27.211386245988
BOHR2A = 0.529177210903
GRAD2F = EH2EV / BOHR2A
AU2CM = 5140.48674
RE_GL = re.compile(r"^\s*\d+\s+([A-Z][a-z]?)\s*:\s+(\S+)\s+(\S+)\s+(\S+)\s*$")
RE_CO = re.compile(r"^\s*([A-Z][a-z]?)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$")
RE_KV = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)=("(?:[^"\\]|\\.)*"|\S+)')
TSK = ("ts_path", "ts_final", "tsguess_opt", "path", "neb", "fwd_bwd")
_G = {}


def sha(b):
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------- labels
def load_labels():
    L = json.loads((LAD / "labels_d1_cr2.json").read_text())
    import csv
    cf = {}
    with open(SF) as f:
        for r in csv.DictReader(f):
            cf.setdefault(r["core"], r["fold_T1"])
    T = {}
    for r in L:
        core = f"{r['dipole']}/{r['skeleton']}"
        T[r["job_id"]] = dict(rxn=int(r["rxn_id"]), pairs=[tuple(int(x) for x in t.split("-")) for t in
                                                            str(r["formed_pairs_ts"]).split()],
                              fold=str(cf.get(core, "-1")), dipole=r["dipole"])
    return T


def read_xyz_text(t):
    ln = t.splitlines()
    n = int(ln[0].split()[0])
    s, p = [], []
    for x in ln[2:2 + n]:
        q = x.split()
        s.append(q[0])
        p.append([float(q[1]), float(q[2]), float(q[3])])
    return s, np.array(p)


def kabsch_max(P, Q):
    P = np.asarray(P, float) - np.mean(P, 0)
    Q = np.asarray(Q, float) - np.mean(Q, 0)
    H = P.T @ Q
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return float(np.linalg.norm(P @ R.T - Q, axis=1).max())


def dform(pos, lab, pairs):
    return float(max(abs(np.linalg.norm(pos[a] - pos[b]) - np.linalg.norm(lab[a] - lab[b])) for a, b in pairs))


# ---------------------------------------------------------------- own ORCA parsers
def bang_tokens(text):
    toks = set()
    for ln in text.splitlines():
        s = ln.split("#")[0].strip()
        if s.startswith("!"):
            toks |= set(s[1:].upper().split())
    return toks


def level_of(inp_text):
    t = bang_tokens(inp_text)
    low = inp_text.lower()
    smd = "SMD(WATER)" in t or ("CPCM(WATER)" in t and re.search(r"%cpcm.*?smd\s+true.*?end", low, re.S) is not None
                                 and "water" in low)
    lab = {"B3LYP", "D3BJ", "DEF2-SVP", "RIJCOSX", "DEF2/J"} <= t and smd and not (t & {"DEF2-TZVP", "XTB2", "XTB"})
    grid = sorted(x for x in t if re.match(r"^(DEFGRID|GRID|FINALGRID)", x))
    if "OPTTS" in t:
        jt = "optts"
    elif t & {"OPT", "LOOSEOPT", "TIGHTOPT", "NORMALOPT", "VERYTIGHTOPT"}:
        jt = "const_opt" if re.search(r"(?is)%geom.*?constraints", inp_text) else "opt"
    elif "ENGRAD" in t:
        jt = "engrad"
    elif t & {"FREQ", "NUMFREQ"}:
        jt = "freq"
    else:
        jt = "sp"
    basis = sorted(x for x in t if x.startswith("DEF2-") or x in ("XTB2", "XTB"))
    return dict(label=bool(lab and not grid), jt=jt, basis=basis, smd=bool(smd), grid=grid,
                freq_in_bang=bool(t & {"FREQ", "NUMFREQ"}))


def echo_text(text):
    out = []
    for ln in text.splitlines():
        m = re.match(r"^\|\s*\d+>\s?(.*)$", ln)
        if m:
            out.append(m.group(1))
        elif out and "END OF INPUT" in ln:
            break
    return "\n".join(out)


def parse_out(text):
    """cycle-segment parser: segment k = lines between 'GEOMETRY OPTIMIZATION CYCLE' k and k+1 (EnGrad: whole file).
    In each segment: first coordinate block, then the first FINAL SINGLE POINT ENERGY after it, then the first
    CARTESIAN GRADIENT block after the energy; counts every gradient block per segment."""
    lines = text.splitlines()
    ver = re.search(r"Program Version\s+(\S+)", text[:200000])
    info = dict(version=ver.group(1) if ver else None, term="ORCA TERMINATED NORMALLY" in text[-4000:],
                conv="THE OPTIMIZATION HAS CONVERGED" in text,
                n_cyc=sum(1 for ln in lines if "GEOMETRY OPTIMIZATION CYCLE" in ln),
                n_grad_blocks=sum(1 for ln in lines if ln.strip() == "CARTESIAN GRADIENT"),
                is_orca=("O   R   C   A" in text[:100000]) or ver is not None)
    cyc = [i for i, ln in enumerate(lines) if "GEOMETRY OPTIMIZATION CYCLE" in ln]
    segs = [(cyc[k], cyc[k + 1] if k + 1 < len(cyc) else len(lines), k + 1) for k in range(len(cyc))] or \
        [(0, len(lines), 0)]
    steps, multi = [], 0
    for a, b, c in segs:
        i = a
        sym = pos = e = g = None
        ngb = 0
        while i < b:
            s = lines[i].strip()
            if s == "CARTESIAN COORDINATES (ANGSTROEM)" and pos is None:
                j = i + 2
                sym, P = [], []
                while j < b:
                    m = RE_CO.match(lines[j])
                    if not m:
                        break
                    sym.append(m.group(1))
                    P.append([float(m.group(2)), float(m.group(3)), float(m.group(4))])
                    j += 1
                pos = np.array(P)
                i = j
                continue
            if pos is not None and e is None and s.startswith("FINAL SINGLE POINT ENERGY"):
                e = float(s.split()[-1])
            if s == "CARTESIAN GRADIENT":
                ngb += 1
                if pos is not None and e is not None and g is None:
                    j = i + 1
                    G = []
                    while j < b and not RE_GL.match(lines[j]):
                        j += 1
                    while j < b:
                        m = RE_GL.match(lines[j])
                        if not m:
                            break
                        G.append([float(m.group(k).replace("D", "E")) for k in (2, 3, 4)])
                        j += 1
                    g = np.array(G)
            i += 1
        if ngb > 1:
            multi += 1
        if pos is not None and e is not None and g is not None and len(g) == len(sym):
            steps.append(dict(sym=sym, pos=pos, e=e, g=g, cycle=c))
    info["multi_grad_segments"] = multi
    info["n_steps"] = len(steps)
    return info, steps


def read_trj(text):
    ln = text.splitlines()
    fr, k = [], 0
    while k < len(ln) and ln[k].strip():
        n = int(ln[k].split()[0])
        m = re.search(r"E\s+(-?\d+\.\d+)", ln[k + 1])
        P = np.array([[float(x) for x in s.split()[1:4]] for s in ln[k + 2:k + 2 + n]])
        fr.append((float(m.group(1)) if m else None, P))
        k += 2 + n
    return fr


def read_engrad(text):
    v = [s.strip() for s in text.splitlines() if s.strip() and not s.strip().startswith("#")]
    n = int(v[0])
    e = float(v[1])
    g = np.array([float(x) for x in v[2:2 + 3 * n]]).reshape(n, 3)
    xyz = np.array([[float(x) for x in s.split()[1:4]] for s in v[2 + 3 * n:2 + 4 * n]])
    return e, g, xyz * BOHR2A


def read_hess(text):
    L = text.splitlines()
    i = next(k for k, s in enumerate(L) if s.strip() == "$hessian")
    n3 = int(L[i + 1].split()[0])
    H = np.full((n3, n3), np.nan)
    k = i + 2
    while k < len(L):
        hd = L[k].split()
        if not hd or L[k].strip().startswith("$"):
            break
        cols = [int(c) for c in hd]
        for r in range(n3):
            t = L[k + 1 + r].split()
            H[int(t[0]), cols] = [float(x) for x in t[1:]]
        k += 1 + n3
    j = next(k for k, s in enumerate(L) if s.strip() == "$atoms")
    na = int(L[j + 1].split()[0])
    sym, ms, P = [], [], []
    for s in L[j + 2:j + 2 + na]:
        t = s.split()
        sym.append(t[0])
        ms.append(float(t[1]))
        P.append([float(t[2]), float(t[3]), float(t[4])])
    return H, sym, np.array(ms), np.array(P) * BOHR2A


def rot_null(H, pos, masses=None):
    """|H v_rot| / |H| for rigid rotations built from positions (A -> bohr); small iff H and positions share a frame
    (at a stationary point)."""
    X = (np.asarray(pos) / BOHR2A)
    X = X - X.mean(0)
    r = []
    for k in range(3):
        e = np.zeros(3)
        e[k] = 1
        v = np.cross(e, X).ravel()
        v /= np.linalg.norm(v)
        r.append(np.linalg.norm(H @ v))
    return float(max(r) / np.linalg.norm(H, 2))


def freqs(H, masses, pos):
    m = np.repeat(masses, 3)
    Hm = 0.5 * (H + H.T) / np.sqrt(np.outer(m, m))
    X = np.asarray(pos) / BOHR2A
    com = (masses[:, None] * X).sum(0) / masses.sum()
    X = X - com
    B = []
    for k in range(3):
        t = np.zeros_like(X)
        t[:, k] = 1
        B.append((t * np.sqrt(masses)[:, None]).ravel())
        e = np.zeros(3)
        e[k] = 1
        B.append((np.cross(e, X) * np.sqrt(masses)[:, None]).ravel())
    Q, R = np.linalg.qr(np.array(B).T)
    keep = np.abs(np.diag(R)) > 1e-6 * np.abs(np.diag(R)).max()
    Q = Q[:, keep]
    P = np.eye(len(m)) - Q @ Q.T
    w = np.linalg.eigvalsh(P @ Hm @ P)
    w = np.delete(w, np.argsort(np.abs(w))[:Q.shape[1]])
    return np.sort(np.sign(w) * np.sqrt(np.abs(w)) * AU2CM)


# ---------------------------------------------------------------- extxyz (own reader)
def read_ext_gz(path):
    rows = []
    with gzip.open(path, "rt") as f:
        while True:
            ln = f.readline()
            if not ln:
                break
            n = int(ln.split()[0])
            hd = f.readline()
            kv = {}
            for m in RE_KV.finditer(hd):
                k, v = m.group(1), m.group(2)
                if v.startswith('"'):
                    v = v[1:-1].replace('\\"', '"').replace("\\\\", "\\")
                kv[k] = v
            sym, P, F = [], np.empty((n, 3)), np.empty((n, 3))
            for i in range(n):
                q = f.readline().split()
                sym.append(q[0])
                P[i] = [float(x) for x in q[1:4]]
                F[i] = [float(x) for x in q[4:7]]
            for t in ("rxn_ids", "folds", "dipoles"):
                v = kv.get(t, "")
                kv[t] = json.loads(v[6:]) if v.startswith("_JSON ") else ([x for x in v.split(",") if x] if v else [])
            rows.append(dict(kv=kv, sym=sym, pos=P, F=F))
    return rows


def gkey(sym, pos):
    h = hashlib.sha1((" ".join(sym)).encode())
    h.update((np.round(np.asarray(pos, float), 5) + 0.0).astype(np.float64).tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------- job tree
def expected_kind(nrel, jt, name):
    p = nrel.split("/")
    if p[0] == "sp":
        return None, "sp_stage"
    if "/conformers/" in f"/{nrel}":
        return None, "conformers"
    if p[0] == "ref":
        return ("ref_opt", "") if jt == "opt" else (None, f"ref_{jt}")
    if p[0] == "ts" and len(p) > 2:
        if p[1] == "single_points":
            return None, "ts_single_points"
        if p[1] == "thermal":
            return None, f"thermal_{jt}"
        if p[1] == "reactants_and_products":
            m = re.match(r"^([rp])\d+_", name)
            if jt == "opt" and m:
                return ("react_opt" if m.group(1) == "r" else "prod_opt"), ""
            return None, f"rp_{jt}"
        if p[1] == "transition_states":
            mid = p[2:-1]
            if "truncated" in mid:
                return "truncated", ""
            if "initial_path" in mid:
                return "path", ""
            if "neb" in mid:
                return "neb", ""
            if jt == "optts":
                return "ts", ""
            if re.search(r"_(forwards|backwards)_opt_orca\d*\.out$", name):
                return "fwd_bwd", ""
            if "_constrained_opt_orca" in name or jt == "const_opt":
                return "tsguess_opt", ""
            return None, f"tsdir_{jt}"
        return None, f"ts_other_{p[1]}_{jt}"
    return None, f"other_{jt}"


def walk(tf, prefix, files, inv, dups):
    for m in tf:
        nm = m.name[2:] if m.name.startswith("./") else m.name
        if m.isdir():
            continue
        if not m.isfile():
            inv.append([prefix + nm, m.size, "link" if (m.issym() or m.islnk()) else "other"])
            continue
        data = tf.extractfile(m).read()
        if nm.endswith(".tar"):
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as t2:
                walk(t2, prefix + nm + "::", files, inv, dups)
            continue
        inv.append([prefix + nm, m.size, "file"])
        if nm in files:
            dups.append([nm, files[nm][1], prefix])
        files[nm] = (data, prefix)


def norm_paths(files):
    roots = Counter(r.split("/")[1] for r in files if r.startswith("ts/") and r.count("/") >= 2)
    root = next(iter(roots)) if len(roots) == 1 else None
    out = {}
    for r in files:
        out[r] = "ts/" + r[len(f"ts/{root}/"):] if root and r.startswith(f"ts/{root}/") else r
    return out, sorted(roots)


def ftype(p):
    b = p.rsplit("/", 1)[-1]
    for suf in (".out", ".inp", "_trj.xyz", ".hess", ".engrad", ".xyz", ".json", ".log", ".chk", ".pdf", ".tsv",
                ".gbw", ".property.txt", ".cpcm", ".smd", ".cpcm_corr", ".bibtex", ".err", ".densities", ".tmp"):
        if b.endswith(suf):
            return suf
    if b.startswith("."):
        return "marker"
    if b.startswith("xcontrol"):
        return "xcontrol"
    return "other"


def review_job(J, pk, T, rows_by_sid, rows_by_key, rows_of_job, hidx, tmproot, tshort, rep_job):
    R = dict(job=J, problems=[], notes=[])
    lab = T.get(J)
    R["target"] = lab is not None
    files, inv, dups = {}, [], []
    for name, blob in pk["arch"]:
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:*") as tf:
            walk(tf, "", files, inv, dups)
    for rel, blob in pk["raw"]:
        if rel.endswith(".tar"):
            with tarfile.open(fileobj=io.BytesIO(blob), mode="r:") as t2:
                walk(t2, f"{rel}::", files, inv, dups)
    R["n_archive_files"] = len(files)
    R["dup_paths"] = dups[:5]
    nm, roots = norm_paths(files)
    R["ts_roots"] = roots
    d = Path(tempfile.mkdtemp(prefix=f"nurrev_{J}_", dir=tmproot))
    try:
        for r, (data, _) in files.items():
            if r.endswith((".out", ".inp", ".hess", "_trj.xyz", ".engrad", ".xyz")):
                p = d / nm[r]
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(data)
        loc = {nm[r]: d / nm[r] for r in files if r.endswith((".out", ".inp", ".hess", "_trj.xyz", ".engrad", ".xyz"))}
        # label TS
        if lab:
            ls, lp = read_xyz_text((LAD / "dft_final" / str(lab["rxn"]) / "ts.xyz").read_text())
            lab["sym"], lab["pos"] = ls, lp
            if "ts.xyz" in pk["top"]:
                s2, p2 = read_xyz_text(pk["top"]["ts.xyz"].decode())
                R["top_ts_eq_label_raw_A"] = float(np.abs(p2 - lp).max()) if s2 == ls else None
        # inventory + top-level vs d1_jobs copies
        R["inventory"] = dict(Counter(f"{nm[r].split('/')[0]}:{ftype(nm[r])}" for r in files))
        R["top_files"] = sorted(pk["top"]) + sorted(x[0] for x in pk["raw"])
        R["outer_other"] = pk["other"]
        if lab and (D1J / J).exists():
            cmp = {}
            for fn in ("ts.xyz", "ref_dipole.xyz", "ref_dipolarophile.xyz", "label.json"):
                if fn in pk["top"] and (D1J / J / fn).exists():
                    cmp[fn] = sha(pk["top"][fn]) == sha((D1J / J / fn).read_bytes())
            if "sp/eda.inp" in nm.values():
                r = next(r for r in files if nm[r] == "sp/eda.inp")
                cmp["sp/eda.inp"] = sha(files[r][0]) == sha((D1J / J / "sp" / "eda.inp").read_bytes()) \
                    if (D1J / J / "sp" / "eda.inp").exists() else None
            R["d1_jobs_copy_equal"] = cmp
        # every .out
        outs = []
        exp_rows = {}                                         # sid -> (kind, step, file)
        for nrel in sorted(x for x in loc if x.endswith(".out")):
            txt = loc[nrel].read_text(errors="replace")
            name = nrel.rsplit("/", 1)[-1]
            base = nrel[:-4]
            inp = loc.get(base + ".inp")
            src = "inp" if inp is not None else "echo"
            lv = level_of(inp.read_text(errors="replace") if inp is not None else echo_text(txt))
            lv_echo = level_of(echo_text(txt))
            xtb = bool(re.search(r"_xtb\d*\.out$", name))
            info, steps = parse_out(txt) if (lv["label"] or lv_echo["label"]) and not xtb else (
                dict(is_orca=("O   R   C   A" in txt[:100000]), term="ORCA TERMINATED NORMALLY" in txt[-4000:]), [])
            ek, why = expected_kind(nrel, lv["jt"], name)
            o = dict(path=nrel, src=src, label=lv["label"], label_echo=lv_echo["label"], jt=lv["jt"], basis=lv["basis"],
                     xtb=xtb, is_orca=info.get("is_orca"), ver=info.get("version"), term=info.get("term"),
                     conv=info.get("conv"), n_cyc=info.get("n_cyc"), n_steps=len(steps),
                     multi=info.get("multi_grad_segments"), ek=ek, why=why)
            if lv["label"] != lv_echo["label"] and not xtb:
                R["problems"].append(f"level inp != echo: {nrel}")
            want = (ek is not None and lv["label"] and not xtb and info.get("term") and steps and
                    (info["n_cyc"] == len(steps) if lv["jt"] in ("optts", "opt", "const_opt") else
                     (len(steps) == 1 and info["n_cyc"] == 0)) and str(info.get("version") or "").startswith("6.1.1"))
            o["want"] = bool(want)
            if ek is not None and lv["label"] and not xtb and not want:
                o["why_not"] = ("not_term" if not info.get("term") else
                                f"steps {len(steps)} vs cyc {info.get('n_cyc')}" if steps else "no_steps")
                o["lost_steps_if_valid"] = len(steps)
            # cross-checks vs trj / engrad / final xyz
            if steps:
                trj = loc.get(base + "_trj.xyz")
                if trj is not None:
                    fr = read_trj(trj.read_text())
                    o["trj_frames"] = len(fr)
                    n = min(len(fr), len(steps))
                    o["trj_dpos"] = float(max(np.abs(fr[i][1] - steps[i]["pos"]).max() for i in range(n))) if n else None
                    o["trj_de"] = float(max(abs(fr[i][0] - steps[i]["e"]) for i in range(n) if fr[i][0] is not None)) \
                        if n else None
                fx = loc.get(base + ".xyz")
                if fx is not None:
                    try:
                        s2, p2 = read_xyz_text(fx.read_text())
                        if s2 == steps[-1]["sym"]:
                            o["final_xyz_vs_last_A"] = float(np.abs(p2 - steps[-1]["pos"]).max())
                            if lab and s2 == lab["sym"]:
                                o["final_xyz_vs_label_raw_A"] = float(np.abs(p2 - lab["pos"]).max())
                                o["last_vs_label_raw_A"] = float(np.abs(steps[-1]["pos"] - lab["pos"]).max())
                    except Exception as ex:                    # noqa: BLE001
                        o["xyz_err"] = str(ex)[:60]
                eg = loc.get(base + ".engrad")
                if eg is not None:
                    try:
                        e, g, xyz = read_engrad(eg.read_text())
                        # engrad = last gradient evaluation
                        last = steps[-1]
                        o["engrad_vs_last"] = dict(de=abs(e - last["e"]), dg=float(np.abs(g - last["g"]).max()),
                                                   dx=float(np.abs(xyz - last["pos"]).max()))
                    except Exception as ex:                    # noqa: BLE001
                        o["engrad_err"] = str(ex)[:80]
            outs.append(o)
            if want:
                n = len(steps)
                for i, s in enumerate(steps):
                    k = ek
                    if ek == "ts":
                        k = "ts_final" if (info["conv"] and i == n - 1) else "ts_path"
                    sid = f"N:{J}:{base}:{s['cycle']}"
                    exp_rows[sid] = (k, s, nrel)
        R["outs"] = outs
        # compare expected rows with the extxyz
        pref = "nur_" if lab else "nur_x_"
        cmpc = Counter()
        maxd = dict(pos=0.0, e_ev=0.0, e_eh=0.0, F=0.0, fmax=0.0, dup_de_eh=0.0, dform=0.0)
        near = 0
        seen_sids = set()
        pf_n, pf_near = Counter(), Counter()
        for sid, (k, s, nrel) in exp_rows.items():
            pf_n[nrel] += 1
            if lab and k in TSK and s["sym"] == lab["sym"] and dform(s["pos"], lab["pos"], lab["pairs"]) <= 0.3:
                pf_near[nrel] += 1
            r = rows_by_sid.get(sid)
            dup = False
            if r is None:
                ks = rows_by_key.get(gkey(s["sym"], s["pos"]))
                if ks is None:
                    cmpc["MISSING"] += 1
                    R["problems"].append(f"missing row {sid}")
                    continue
                r, dup = ks, True
                cmpc["dup_elsewhere"] += 1
            seen_sids.add(r["kv"]["struct_id"])
            kv = r["kv"]
            if r["sym"] != s["sym"]:
                cmpc["symbol_order"] += 1
                R["problems"].append(f"symbols differ {sid}")
                continue
            maxd["pos"] = max(maxd["pos"], float(np.abs(r["pos"] - s["pos"]).max()))
            if dup:
                maxd["dup_de_eh"] = max(maxd["dup_de_eh"], abs(float(kv["e_eh"]) - s["e"]))
                if lab and str(lab["rxn"]) not in kv["rxn_ids"]:
                    R["problems"].append(f"dup row {kv['struct_id']} lacks own rxn {lab['rxn']}")
                continue
            if kv["kind"] != pref + k:
                cmpc["kind_mismatch"] += 1
                R["problems"].append(f"kind {sid}: row {kv['kind']} expected {pref + k}")
            maxd["e_ev"] = max(maxd["e_ev"], abs(float(kv["REF_energy"]) - s["e"] * EH2EV))
            maxd["e_eh"] = max(maxd["e_eh"], abs(float(kv["e_eh"]) - s["e"]))
            maxd["F"] = max(maxd["F"], float(np.abs(r["F"] - (-s["g"] * GRAD2F)).max()))
            maxd["fmax"] = max(maxd["fmax"], abs(float(kv["fmax"]) - float(np.abs(s["g"]).max())))
            if int(kv["cycle"]) != s["cycle"]:
                R["problems"].append(f"cycle {sid}")
            if lab:
                if str(lab["rxn"]) not in kv["rxn_ids"] or lab["fold"] not in kv["folds"] or lab["dipole"] not in kv["dipoles"]:
                    R["problems"].append(f"tags {sid}: {kv['rxn_ids'][:3]} {kv['folds']} {kv['dipoles']}")
                if k in ("ts_path", "ts_final", "tsguess_opt", "path", "neb", "fwd_bwd") and s["sym"] == lab["sym"]:
                    df = dform(s["pos"], lab["pos"], lab["pairs"])
                    if "dform_label_A" not in kv:
                        R["problems"].append(f"no dform {sid}")
                    else:
                        maxd["dform"] = max(maxd["dform"], abs(float(kv["dform_label_A"]) - df))
                    near += df <= 0.3
            if not lab and k in TSK and kv["rxn_ids"]:
                R["notes"].append(f"nontarget TS row with rxn_ids {sid} {kv['rxn_ids'][:3]}")
            cmpc["matched"] += 1
        # vs the report's per-file list (path, kind, n, n_near, trj, label_run)
        kept_rep = {x[0]: x for x in (rep_job or {}).get("kept", [])}
        diff = []
        for nrel in sorted(set(kept_rep) | set(pf_n)):
            a = kept_rep.get(nrel)
            if a is None or pf_n.get(nrel, 0) != a[2] or (lab and pf_near.get(nrel, 0) != a[3]):
                diff.append([nrel, a[2] if a else None, pf_n.get(nrel, 0), a[3] if a else None, pf_near.get(nrel, 0)])
        R["report_file_diff"] = diff
        if any(x[1] != x[2] for x in diff):
            R["problems"].append(f"report kept list n != mine: {[x for x in diff if x[1] != x[2]][:3]}")
        R["report_excluded"] = (rep_job or {}).get("excluded_label_level")
        R["expected_rows"] = len(exp_rows)
        R["cmp"] = dict(cmpc)
        R["maxdiff"] = maxd
        R["ts_near_mine"] = int(near)
        extra = [x["kv"]["struct_id"] for x in rows_of_job.get(J, []) if x["kv"]["struct_id"] not in seen_sids]
        R["rows_of_job"] = len(rows_of_job.get(J, []))
        R["extra_rows"] = extra[:10]
        R["n_extra_rows"] = len(extra)
        if extra:
            R["problems"].append(f"{len(extra)} rows of job not explained by my expected set")
        # label TS among rows
        if lab:
            best = (9e9, None)
            for x in rows_of_job.get(J, []):
                if x["sym"] == lab["sym"]:
                    dv = float(np.abs(x["pos"] - lab["pos"]).max())
                    if dv < best[0]:
                        best = (dv, x["kv"]["struct_id"], x["kv"]["kind"], x["kv"].get("label_ts"), x["kv"]["fmax"])
            R["label_ts_nearest_row"] = best
            R["label_ts_flag_rows"] = [x["kv"]["struct_id"] for x in rows_of_job.get(J, []) if x["kv"].get("label_ts")]
        # Hessians
        hs = []
        for nrel in sorted(x for x in loc if x.endswith(".hess")):
            try:
                H, hsym, hm, hp = read_hess(loc[nrel].read_text())
            except Exception as ex:                            # noqa: BLE001
                hs.append(dict(path=nrel, err=str(ex)[:80]))
                continue
            m = re.match(r"^(.*)\.(\d{3})\.hess$", nrel)
            ob = (m.group(1) if m else nrel[:-5]) + ".out"
            oo = next((o for o in outs if o["path"] == ob), None)
            h = dict(path=nrel, out=ob, out_label=oo["label"] if oo else None, out_jt=oo["jt"] if oo else None,
                     out_conv=oo["conv"] if oo else None, out_term=oo["term"] if oo else None, n=len(hsym),
                     nan=bool(np.isnan(H).any()))
            if lab and hsym == lab["sym"]:
                h["kabsch_label"] = kabsch_max(hp, lab["pos"])
                h["raw_label"] = float(np.abs(hp - lab["pos"]).max())
            h["_H"], h["_p"], h["_m"], h["_s"] = H, hp, hm, hsym
            hs.append(h)
        R["hess_files"] = [{k: v for k, v in h.items() if not k.startswith("_")} for h in hs]
        if lab:
            npz = NUR / "hess" / f"{lab['rxn']}.npz"
            cands = [h for h in hs if h.get("kabsch_label") is not None and h["kabsch_label"] <= 1e-4]
            R["label_hess_candidates"] = [[h["path"], h["kabsch_label"], h["out_conv"], h["out_term"]] for h in cands]
            if npz.exists():
                z = np.load(npz, allow_pickle=False)
                src = str(z["source"])
                h = next((h for h in hs if h["path"] == src), None)
                hl = dict(source=src, tar=str(z["tar"]), n_imag=int(z["n_imag"]), freq_low=z["freq_low_cm"].tolist())
                if h is None:
                    R["problems"].append(f"npz source {src} not in tree")
                else:
                    Hz = z["hessian"].astype(float)
                    hl["dH_max"] = float(np.abs(Hz - h["_H"]).max())
                    hl["dH_rel"] = float(np.abs(Hz - h["_H"]).max() / np.abs(h["_H"]).max())
                    hl["dpos"] = float(np.abs(z["positions_A"] - h["_p"]).max())
                    hl["sym_eq"] = list(z["symbols"]) == h["_s"]
                    hl["mass_eq"] = bool(np.allclose(z["masses"], h["_m"]))
                    hl["kabsch_label"] = h.get("kabsch_label")
                    hl["raw_label"] = h.get("raw_label")
                    hl["rot_null_hessframe"] = rot_null(h["_H"], h["_p"])
                    hl["rot_null_labelframe"] = rot_null(h["_H"], lab["pos"])
                    fr = freqs(h["_H"], h["_m"], h["_p"])
                    hl["my_freq_low"] = [round(float(x), 2) for x in fr[:3]]
                    hl["my_n_imag_lt50"] = int((fr < -50).sum())
                    hl["my_n_imag_lt0"] = int((fr < -10).sum())
                    # is this the final (non-indexed) .hess of a converged label-level OptTS?
                    hl["src_ok"] = (h["out_jt"] == "optts" and h["out_conv"] and h["out_label"] and
                                    not re.search(r"\.\d{3}\.hess$", src))
                    # alternatives with an even closer geometry
                    hl["alt_closer"] = [[c["path"], c["kabsch_label"]] for c in cands if c["path"] != src
                                        and c["kabsch_label"] < (h.get("kabsch_label") or 9) - 1e-7]
                    if hl["dH_rel"] > 1e-6 or hl["dpos"] > 1e-9 or not hl["sym_eq"]:
                        R["problems"].append(f"npz != tree hess {src}")
                    if hl["my_n_imag_lt50"] != hl["n_imag"]:
                        R["notes"].append(f"n_imag mine {hl['my_n_imag_lt50']} vs npz {hl['n_imag']}")
                R["label_hess"] = hl
            else:
                R["label_hess"] = None
                if cands:
                    R["problems"].append(f"label TS Hessian candidates exist but no npz: {[c['path'] for c in cands]}")
        # hess_other index coverage
        idx = {e["path"]: e for e in hidx.get(J, [])}
        cov = Counter()
        lost = []
        lab_src = (R.get("label_hess") or {}).get("source")
        kept_h = [h for h in hs if "_H" in h and (h["path"] in idx or h["path"] == lab_src)]
        for h in hs:
            if "_H" not in h:
                continue
            if not h["out_label"]:
                cov["not_label_level"] += 1
                continue
            if h["path"] == lab_src:
                cov["label_npz"] += 1
            elif h["path"] in idx:
                cov["in_hess_other"] += 1
            else:
                same = [k for k in kept_h if k["_s"] == h["_s"] and np.abs(k["_p"] - h["_p"]).max() < 1e-5
                        and np.abs(k["_H"] - h["_H"]).max() < 1e-7]
                if same:
                    cov["dropped_identical"] += 1
                else:
                    near_geo = [k for k in kept_h if k["_s"] == h["_s"] and np.abs(k["_p"] - h["_p"]).max() < 1e-5]
                    lost.append([h["path"], [k["path"] for k in near_geo][:2],
                                 float(min((np.abs(k["_H"] - h["_H"]).max() for k in near_geo), default=np.nan))])
                    cov["dropped_NOT_identical"] += 1
        R["hess_cov"] = dict(cov)
        R["hess_dropped_not_identical"] = lost[:6]
        # spot check one hess_other entry numerically
        worst = dict(n=0, dH_rel=0.0, dpos=0.0, sym_bad=0)
        Z = _G.get("hz")
        for h in hs:
            if "_H" in h and h["path"] in idx and Z is not None:
                e = idx[h["path"]]
                i = e["i"]
                n3 = 3 * len(e["symbols"])
                Hf = np.zeros((n3, n3))
                iu = np.triu_indices(n3)
                Hf[iu] = Z[f"H{i}"]
                Hf = Hf + np.triu(Hf, 1).T
                worst["n"] += 1
                worst["dH_rel"] = max(worst["dH_rel"], float(np.abs(Hf - 0.5 * (h["_H"] + h["_H"].T)).max()
                                                             / np.abs(h["_H"]).max()))
                worst["dpos"] = max(worst["dpos"], float(np.abs(Z[f"P{i}"] - h["_p"]).max()))
                worst["sym_bad"] += int(e["symbols"] != h["_s"])
                worst["asym_file"] = max(worst.get("asym_file", 0.0), float(np.abs(h["_H"] - h["_H"].T).max()))
        R["hess_other_numeric"] = worst
        if worst["dH_rel"] > 1e-6 or worst["dpos"] > 1e-9 or worst["sym_bad"]:
            R["problems"].append(f"hess_other numeric mismatch {worst}")
        return R
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main():
    t0 = time.time()
    stem, jobs = sys.argv[1], [x for x in sys.argv[2].split(",") if x]
    tshort = re.sub(r"^d1_\d{8}_", "", stem)
    want = set(jobs)
    T = load_labels()
    tmproot = tempfile.mkdtemp(prefix="nurrev_", dir=os.environ.get("TMPDIR", "/tmp"))
    pk = {j: dict(top={}, arch=[], raw=[], other=[], n_members=0) for j in want}
    t1 = time.time()
    nb = 0
    with open(UP / f"{stem}.tar", "rb") as fh, tarfile.open(fileobj=fh, mode="r|") as tf:
        for m in tf:
            nb += m.size
            name = m.name[2:] if m.name.startswith("./") else m.name
            p = name.split("/")
            j = None
            if len(p) >= 2 and p[0].startswith("jobs"):
                j = p[1]
            elif len(p) >= 2 and p[0] in ("logs", "attempts"):
                j = re.sub(r"\.log$", "", p[1])
            if j not in want or not m.isfile():
                continue
            pk[j]["n_members"] += 1
            if p[0] in ("logs", "attempts"):
                pk[j]["other"].append([name, m.size])
                continue
            rest = "/".join(p[2:])
            if rest == "archive.tgz":
                pk[j]["arch"].append((rest, tf.extractfile(m).read()))
            elif rest.endswith(".tar"):
                pk[j]["raw"].append((rest, tf.extractfile(m).read()))
            elif "/" not in rest and m.size < 4_000_000:
                pk[j]["top"][rest] = tf.extractfile(m).read()
            else:
                pk[j]["other"].append([name, m.size])
    t_stream = time.time() - t1
    # extxyz of this tar
    t2 = time.time()
    rows = read_ext_gz(NUR / stem / "nurion.extxyz.gz")
    rows_by_sid, rows_by_key, rows_of_job = {}, {}, defaultdict(list)
    gl = Counter()
    for r in rows:
        kv = r["kv"]
        rows_by_sid[kv["struct_id"]] = r
        rows_by_key.setdefault(gkey(r["sym"], r["pos"]), r)
        rows_of_job[kv.get("job")].append(r)
        # global checks on every row
        if abs(float(kv["REF_energy"]) - float(kv["e_eh"]) * EH2EV) > 1e-6:
            gl["energy_units"] += 1
        if abs(float(np.abs(r["F"]).max()) / GRAD2F - float(kv["fmax"])) > 1e-7:
            gl["fmax_vs_F"] += 1
        if not np.isfinite(r["F"]).all():
            gl["nonfinite"] += 1
        if kv.get("level") != "label":
            gl["level_not_label"] += 1
        gl["rows"] += 1
    t_ext = time.time() - t2
    hj = json.loads((NUR / stem / "hess_other.json").read_text())
    hidx = defaultdict(list)
    for e in hj["hessians"]:
        hidx[e["job"]].append(e)
    rep = json.loads((NUR / stem / "report.json").read_text())
    rj = {j["job"]: j for j in rep["jobs"]}
    _G["hz"] = np.load(NUR / stem / "hess_other.npz", allow_pickle=False)
    gl["sid_unique"] = int(len(rows_by_sid) == len(rows))
    gl["kinds_eq_report"] = int(dict(Counter(r["kv"]["kind"] for r in rows)) == rep["summary"]["kinds"])
    out = dict(tar=stem, jobs=jobs, t_stream_s=round(t_stream, 1), bytes_streamed=nb, t_ext_s=round(t_ext, 1),
               global_row_checks=dict(gl), per_job=[])
    for J in jobs:
        try:
            out["per_job"].append(review_job(J, pk[J], T, rows_by_sid, rows_by_key, rows_of_job, hidx, tmproot, tshort,
                                             rj.get(J)))
        except Exception as ex:                                # noqa: BLE001
            import traceback
            out["per_job"].append(dict(job=J, error=traceback.format_exc()[-800:]))
    # numeric spot check of hess_other storage for a few entries
    z = np.load(NUR / stem / "hess_other.npz", allow_pickle=False)
    sp = []
    for e in hj["hessians"][:3]:
        i = e["i"]
        n3 = 3 * len(e["symbols"])
        Ht = z[f"H{i}"]
        Hf = np.zeros((n3, n3), np.float32)
        iu = np.triu_indices(n3)
        Hf[iu] = Ht
        sp.append(dict(i=i, path=e["path"], n=len(Ht) == n3 * (n3 + 1) // 2, sym_ok=Hf[iu].shape == Ht.shape))
    out["hess_other_storage"] = sp
    shutil.rmtree(tmproot, ignore_errors=True)
    out["wall_s"] = round(time.time() - t0, 1)
    RES.mkdir(parents=True, exist_ok=True)
    tmp = RES / f".{stem}_a.json.tmp"
    tmp.write_text(json.dumps(out, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    tmp.replace(RES / f"{stem}_a.json")
    # compact stdout
    summ = dict(tar=stem, wall=out["wall_s"], t_stream=out["t_stream_s"], t_ext=out["t_ext_s"], glob=out["global_row_checks"])
    print(json.dumps(summ))
    for r in out["per_job"]:
        if "error" in r:
            print(r["job"], "ERROR", r["error"])
            continue
        print(json.dumps(dict(job=r["job"], tgt=r["target"], exp=r["expected_rows"], cmp=r["cmp"],
                              md={k: float(f"{v:.2e}") for k, v in r["maxdiff"].items()}, extra=r["n_extra_rows"],
                              lh=(r.get("label_hess") or {}).get("dH_rel"), prob=r["problems"][:4],
                              notes=r["notes"][:2], hcov=r["hess_cov"])))


if __name__ == "__main__":
    main()
