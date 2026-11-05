#!/usr/bin/env python3
"""trace B (read only): label-level TS inventory + FT-2 pilot stats. Writes only under $MLT/refcheck.
  python traceB.py stats   -> traceB_stats.txt   (config evidence, audit d3 gradients + Newton, G2 label-level TSs, rec stats)
  python traceB.py nurion  -> traceB_nurion.txt  (nurion nur_ts_final label_ts=1 gradients + label TS Hessians -> Newton)"""
import glob
import gzip
import io
import json
import os
import re
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, "/gpfs/home1/yeseo1ee/projects/eda-asm-prediction-mlts/analysis/mlts")
import mlt_common as M  # noqa: E402

OUT = Path("/gpfs/tmp_cpu2/yeseo1ee/mlts/refcheck")
FT2 = M.MLT / "ft2" / "FTpilot_AIMNet2_4"
BOHR = M.BOHR2A
NOT_ATT = ("no_start", "no_label", "not_covered")
LINES = []


def P(*a):
    s = " ".join(str(x) for x in a)
    LINES.append(s)
    print(s, flush=True)


def flush(name):
    (OUT / name).write_text("\n".join(LINES) + "\n")


def q(x, p):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    return float(np.percentile(x, p)) if len(x) else float("nan")


def med(x):
    return q(x, 50)


def dists(pos, pairs):
    p = np.asarray(pos, float)
    return [float(np.linalg.norm(p[i] - p[j])) for i, j in pairs]


def kabsch(A, B):
    A, B = np.asarray(A, float), np.asarray(B, float)
    ca, cb = A.mean(0), B.mean(0)
    U, S, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, ca, cb, float(np.abs((A - ca) @ R.T + cb - B).max())


def heavy_rmsd(sym, A, B):
    idx = [k for k, s in enumerate(sym) if s != "H"]
    a, b = np.asarray(A, float)[idx], np.asarray(B, float)[idx]
    R, ca, cb, _ = kabsch(a, b)
    return float(np.sqrt((((a - ca) @ R.T + cb - b) ** 2).sum(1).mean()))


def tr_basis(X):
    X = np.asarray(X, float)
    n = len(X)
    c = X.mean(0)
    V = []
    for k in range(3):
        v = np.zeros((n, 3)); v[:, k] = 1.0; V.append(v.ravel())
    for k in range(3):
        e = np.zeros(3); e[k] = 1.0
        V.append(np.cross(e, X - c).ravel())
    V = np.array(V).T
    U, S, _ = np.linalg.svd(V, full_matrices=False)
    return U[:, S > 1e-6 * S.max()]


def newton(H, g, X_bohr):
    """projected Newton step (bohr) to the stationary point of the quadratic model; also n_neg and TS-mode part."""
    V = tr_basis(X_bohr)
    n3 = H.shape[0]
    Pm = np.eye(n3) - V @ V.T
    Hp = Pm @ H @ Pm
    Hp = 0.5 * (Hp + Hp.T)
    gp = Pm @ g
    w, U = np.linalg.eigh(Hp)
    ov = ((V.T @ U) ** 2).sum(0)
    keep = (ov < 0.5) & (np.abs(w) > 1e-6)
    Ui, wi = U[:, keep], w[keep]
    c = Ui.T @ gp
    step = -Ui @ (c / wi)
    j = int(np.argmin(wi))
    step_ts = -Ui[:, j] * (c[j] / wi[j])
    return step, step_ts, int((wi < -1e-4).sum()), float(wi[j]), float(np.abs(gp).max())


def targets():
    T = M.targets()
    T["pairs"] = [M.formed_pairs(s) for s in T.formed_pairs_ts]
    return T


def load_recs(start):
    out = {}
    for p in glob.glob(str(FT2 / start / "recs" / "*.json")):
        try:
            r = json.loads(Path(p).read_text())
            out[int(r["rxn_id"])] = r
        except Exception:  # noqa: BLE001
            pass
    return out


def fnum(v):
    try:
        x = float(v)
        return x if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def truthy(v):
    return v is True or str(v).strip().lower() == "true"


def fail_class(r):
    if str(r.get("status")) in NOT_ATT:
        return None
    if not truthy(r.get("converged")):
        return "not_converged"
    ni = fnum(r.get("n_imag"))
    if ni is None or int(round(ni)) != 1:
        return "n_imag"
    dd = fnum(r.get("dd_max"))
    if not truthy(r.get("same_ts")) or dd is None or dd > 0.030:
        return "other_saddle"
    return None


def short_long(r):
    """(dd_short, dd_long) with bonds ordered by label distance."""
    dl, dd = r.get("d_label"), r.get("dd")
    if not dl or not dd or len(dl) != 2:
        return None, None
    o = np.argsort(dl)
    return float(dd[o[0]]), float(dd[o[1]])


def table(rs, tag):
    att = [r for r in rs if str(r.get("status")) not in NOT_ATT]
    n = len(att)
    if not n:
        P(f"  {tag}: n=0")
        return
    same = sum(truthy(r.get("same_ts")) for r in att) / n
    dd = [fnum(r.get("dd_max")) for r in att]
    rm = [fnum(r.get("rmsd_heavy")) for r in att]
    conv = [r for r in att if truthy(r.get("converged"))]
    sl = [short_long(r) for r in conv]
    ss = [a for a, b in sl if a is not None]
    ll = [b for a, b in sl if b is not None]
    npass = sum(fail_class(r) is None for r in att)
    P(f"  {tag}: n={n} same={same:.3f} pass={npass / n:.3f} |dd|med={med(dd):.4f} p90={q(dd, 90):.4f} "
      f"rmsd_med={med(rm):.4f} | conv={len(conv)} signed dd short med={med(ss):+.4f} (>0 {np.mean(np.array(ss) > 0):.2f}) "
      f"long med={med(ll):+.4f} (>0 {np.mean(np.array(ll) > 0):.2f})")


# ====================================================================== stats mode
def config_evidence():
    P("== 0. level / convergence evidence")
    cfg = M.D1_CONFIG
    P(f"D1 config: {cfg} exists={cfg.is_file()}")
    if cfg.is_file():
        pat = re.compile(r"optts|recalc|trust|maxiter|keyword|tight|opt_ts|same_ts|functional|basis|solv|dispers|engine|"
                         r"grid|conv|hmethod|rmsd|dform", re.I)
        k = 0
        for ln in cfg.read_text().splitlines():
            if pat.search(ln) and k < 40:
                P("  cfg| " + ln.rstrip()[:200]); k += 1
    P(f"LABEL_GRAD_LINE (training EnGrad) = {M.LABEL_GRAD_LINE!r}; solv block = {M.LABEL_SOLV_BLOCK!r}")
    # one G2 label-level OptTS output: input echo + tolerances
    outs = sorted(glob.glob(str(M.MLT / "dft" / "runs" / "G2" / "*" / "ts.out.gz")))
    P(f"G2 label-level OptTS outputs: {len(outs)}")
    if outs:
        txt = gzip.open(outs[0], "rt", errors="replace").read()
        P(f"  sample {outs[0]}")
        ins = txt.split("INPUT FILE", 1)[-1].split("****END OF INPUT****", 1)[0]
        for ln in ins.splitlines():
            if re.match(r"^\|\s*\d+>", ln) and not re.match(r"^\|\s*\d+>\s+[A-Z][a-z]?\s+-?\d", ln):
                P("  inp| " + ln.strip()[:160])
        i = txt.find("Convergence Tolerances")
        if i >= 0:
            for ln in txt[i:i + 900].splitlines()[:10]:
                P("  tol| " + ln.rstrip()[:120])
    aud = sorted(glob.glob(str(M.MLT / "audit" / "orca" / "d3_*.out.gz")))
    if aud:
        txt = gzip.open(aud[0], "rt", errors="replace").read()
        ins = txt.split("INPUT FILE", 1)[-1].split("****END OF INPUT****", 1)[0]
        P(f"  audit sample {aud[0]}")
        for ln in ins.splitlines():
            if re.match(r"^\|\s*\d+>", ln) and not re.match(r"^\|\s*\d+>\s+[A-Z][a-z]?\s+-?\d", ln):
                P("  aud| " + ln.strip()[:160])


def hess_npz(rid):
    p = M.MLT / "dft" / "nurion" / "hess" / f"{int(rid)}.npz"
    if not p.is_file():
        return None
    with np.load(p, allow_pickle=False) as z:
        return dict(H=np.asarray(z["hessian"], float), X=np.asarray(z["positions_A"], float),
                    sym=[str(s) for s in z["symbols"]], n_imag=int(z["n_imag"]),
                    freq=np.asarray(z["freq_low_cm"], float), src=str(z["source"]))


def audit_d3(T, recs_lab):
    P("== A. label-level gradient AT the label TS (audit d3 EnGrads, mlt_common.orca_grad_input = training level)")
    rows = []
    byid = T.set_index("rxn_id")
    for p in sorted(glob.glob(str(M.MLT / "audit" / "orca" / "d3_*.json"))):
        j = json.loads(Path(p).read_text())
        if not j.get("ok"):
            P(f"  {Path(p).name}: not ok {j.get('reason')}")
            continue
        rid = int(Path(p).stem.split("_")[1])
        t = byid.loc[rid]
        pairs = t.pairs
        sym, xl = M.read_xyz(M.label_ts(rid))
        xl = np.asarray(xl, float)
        g = np.asarray(j["grad_eh_bohr"], float).reshape(-1, 3)
        # gradient projected on forming-bond stretch (Eh/bohr per bohr of bond length)
        gf = []
        for a, b in pairs:
            u = (xl[a] - xl[b]) / np.linalg.norm(xl[a] - xl[b])
            gf.append(float(np.dot(g[a] - g[b], u)) / 2.0)   # dE/dr, symmetric stretch (Eh/bohr)
        row = dict(rid=rid, fold=int(t.fold_T1), dip=t.dipole, nat=int(t.n_atoms), gmax=float(np.abs(g).max()),
                   grms=float(np.sqrt((g ** 2).mean())), gf=gf, dl=dists(xl, pairs))
        h = hess_npz(rid)
        if h is not None and h["sym"] == list(sym):
            R, ca, cb, dev = kabsch(xl, h["X"])
            gh = g @ R.T
            st, stts, nneg, wmin, gpm = newton(h["H"], gh.ravel(), h["X"] / BOHR)
            xs = h["X"] + st.reshape(-1, 3) * BOHR
            xt = h["X"] + stts.reshape(-1, 3) * BOHR
            row.update(dd_newton=list(np.subtract(dists(xs, pairs), dists(h["X"], pairs))),
                       dd_newton_tsmode=list(np.subtract(dists(xt, pairs), dists(h["X"], pairs))),
                       step_max_A=float(np.abs(st.reshape(-1, 3) * BOHR).max()), nneg=nneg, n_imag=h["n_imag"],
                       imag=float(h["freq"][0]) if len(h["freq"]) else None, kdev=dev)
        r = recs_lab.get(rid)
        if r and r.get("dd"):
            row["dd_mlip"] = r["dd"]
        rows.append(row)
    P(f"  n={len(rows)} (20 planned; fold 4 among them: {sum(r['fold'] == 4 for r in rows)})")
    P("  rxn      fold dip               nat  gmax_Eh/b  grms     dE/dr_form(Eh/b)        n_imag imag_cm  dd_newton(A)         "
      "dd_newton_TSmode     step_max  dd_mlip(f4)")
    for r in rows:
        f2 = lambda v: "[" + ",".join(f"{x:+.4f}" for x in v) + "]" if v else "-"  # noqa: E731
        f5 = lambda v: "[" + ",".join(f"{x:+.5f}" for x in v) + "]" if v else "-"  # noqa: E731
        P(f"  {r['rid']} {r['fold']:>3d} {r['dip'][:16]:16s} {r['nat']:4d} {r['gmax']:.2e} {r['grms']:.2e} {f5(r['gf']):24s} "
          f"{r.get('n_imag', '-')!s:>4} {r.get('imag') or float('nan'):8.1f} {f2(r.get('dd_newton')):20s} "
          f"{f2(r.get('dd_newton_tsmode')):20s} {r.get('step_max_A', float('nan')):.4f}  {f2(r.get('dd_mlip'))}")
    P(f"  median gmax {med([r['gmax'] for r in rows]):.2e} Eh/bohr (ORCA NormalOpt TolMaxG 3e-4, TightOpt 1e-4); "
      f"max {max(r['gmax'] for r in rows):.2e}")
    nd = [max(abs(x) for x in r["dd_newton"]) for r in rows if r.get("dd_newton")]
    P(f"  Newton |dd| max-over-bonds: n={len(nd)} median {med(nd):.4f} p90 {q(nd, 90):.4f} max {max(nd) if nd else float('nan'):.4f} A")
    return rows


def g2_runs(T, recs_lab, recs_g2):
    P("== B. label-level OptTS from the G2 TS (gen_dft G-2, $MLT/dft/runs/G2; training kinds g2_path / ts_final_g2)")
    byid = T.set_index("rxn_id")
    units = sorted(glob.glob(str(M.MLT / "dft" / "runs" / "G2" / "*" / "unit.json")))
    st = Counter()
    rows = []
    for up in units:
        try:
            u = json.loads(Path(up).read_text())
        except Exception:  # noqa: BLE001
            st["unreadable"] += 1
            continue
        rid = int(u.get("rxn_id") or Path(up).parent.name)
        st[str(u.get("status"))] += 1
        if rid not in byid.index:
            continue
        t = byid.loc[rid]
        tsx = Path(up).parent / "ts.xyz"
        if u.get("status") != "ok" or not tsx.is_file():
            continue
        sym, xl = M.read_xyz(M.label_ts(rid))
        s2, x2 = M.read_xyz(tsx)
        if list(s2) != list(sym):
            st["order_mismatch"] += 1
            continue
        pairs = t.pairs
        dl = dists(xl, pairs)
        dd2 = list(np.subtract(dists(x2, pairs), dl))
        row = dict(rid=rid, fold=int(t.fold_T1), dip=t.dipole, nat=int(t.n_atoms), dl=dl, dd_g2lab=dd2,
                   rmsd=heavy_rmsd(sym, xl, x2), same=(u.get("converged_to_label")), lastg=u.get("last_max_abs_grad"))
        stp = Path(up).parent / "steps.json"
        if stp.is_file():
            S = json.loads(stp.read_text())
            if S.get("steps"):
                xlast = np.asarray(S["steps"][-1]["pos_A"], float)
                row["dd_lastgrad"] = list(np.subtract(dists(xlast, pairs), dl))
        for nm, R in (("mlip_lab", recs_lab), ("mlip_g2", recs_g2)):
            r = R.get(rid)
            if r and r.get("dd"):
                row[nm] = r["dd"]
        rows.append(row)
    P(f"  units {len(units)} status {dict(st)}")
    a = [max(abs(x) for x in r["dd_g2lab"]) for r in rows]
    P(f"  ok D1 units n={len(rows)}: |dd| (G2-start label-level TS - label TS) median {med(a):.4f} p90 {q(a, 90):.4f}; "
      f"heavy RMSD median {med([r['rmsd'] for r in rows]):.4f} p90 {q([r['rmsd'] for r in rows], 90):.4f}; "
      f"converged_to_label {Counter(str(r['same']) for r in rows)}")
    same_rows = [r for r in rows if r["rmsd"] <= 0.10 and max(abs(x) for x in r["dd_g2lab"]) <= 0.02 * 5]
    P(f"  among rmsd<=0.10 & |dd|<=0.10 (same-basin): n={len(same_rows)} |dd| median "
      f"{med([max(abs(x) for x in r['dd_g2lab']) for r in same_rows]):.4f} p90 "
      f"{q([max(abs(x) for x in r['dd_g2lab']) for r in same_rows], 90):.4f}")
    sl = [(r["dd_g2lab"][int(np.argmin(r["dl"]))], r["dd_g2lab"][int(np.argmax(r["dl"]))]) for r in same_rows]
    if sl:
        P(f"  same-basin signed dd short med {med([s for s, l in sl]):+.4f}  long med {med([l for s, l in sl]):+.4f}")
    f4 = [r for r in rows if r["fold"] == 4]
    P(f"  fold-4 (f4_test) reactions with a G2-start label-level TS: {len(f4)}; with MLIP label-start rec: "
      f"{sum('mlip_lab' in r for r in f4)}")
    P("  rxn      dip              nat  d_label(s,l)     dd_G2lab        dd_lastgrad      dd_MLIP_lab      dd_MLIP_g2   rmsdG2lab")

    def f2(v):
        return "[" + ",".join(f"{x:+.4f}" for x in v) + "]" if v else "-"

    def order(r, v):
        if not v:
            return None
        o = np.argsort(r["dl"])
        return [v[o[0]], v[o[1]]]
    for r in sorted(f4, key=lambda r: r["rid"])[:25]:
        dls = sorted(r["dl"])
        P(f"  {r['rid']} {r['dip'][:16]:16s} {r['nat']:3d}  [{dls[0]:.3f},{dls[1]:.3f}] {f2(order(r, r['dd_g2lab']))} "
          f"{f2(order(r, r.get('dd_lastgrad')))} {f2(order(r, r.get('mlip_lab')))} {f2(order(r, r.get('mlip_g2')))} {r['rmsd']:.3f}")
    both = [r for r in f4 if "mlip_lab" in r and r["rmsd"] <= 0.10]
    if both:
        a = [max(abs(x) for x in r["dd_g2lab"]) for r in both]
        b = [max(abs(x) for x in r["mlip_lab"]) for r in both]
        c = [max(abs(x) for x in np.subtract(r["mlip_lab"], r["dd_g2lab"])) for r in both]
        P(f"  f4 same-basin n={len(both)}: med |dd| G2lab-vs-label {med(a):.4f}; MLIP-vs-label {med(b):.4f}; "
          f"MLIP-vs-G2lab {med(c):.4f}")
    return rows


def rec_stats(T, R, start):
    P(f"== C. FT-2 records {FT2}/{start}/recs: n={len(R)}")
    rs = list(R.values())
    att = [r for r in rs if str(r.get("status")) not in NOT_ATT]
    P(f"  status {dict(Counter(str(r.get('status')) for r in rs))}")
    P(f"  converged {dict(Counter(str(r.get('converged')) for r in att))}  n_imag {dict(sorted(Counter(str(r.get('n_imag')) for r in att).items()))}")
    P(f"  fail_class {dict(Counter(str(fail_class(r)) for r in att))}  same_ts {dict(Counter(str(r.get('same_ts')) for r in att))}")
    P(f"  tight {dict(Counter(str(r.get('tight')) for r in rs))} recalc {dict(Counter(str(r.get('recalc')) for r in rs))}")
    table(rs, "ALL")
    dd = [fnum(r.get("dd_max")) for r in att if truthy(r.get("converged"))]
    P(f"  converged only |dd_max| median {med(dd):.4f} p90 {q(dd, 90):.4f} p10 {q(dd, 10):.4f}; MLIP gmax_final med "
      f"{med([fnum(r.get('gmax_final_eh_bohr')) for r in att]):.2e}")
    # same_ts failure split
    oth = [r for r in att if fail_class(r) == "other_saddle"]
    P(f"  other_saddle: rmsd>0.10 {sum((fnum(r.get('rmsd_heavy')) or 9) > 0.10 for r in oth)}, dform_sorted>0.02 "
      f"{sum((fnum(r.get('dform_sorted')) or 9) > 0.02 for r in oth)}, dd>0.03 {sum((fnum(r.get('dd_max')) or 9) > 0.03 for r in oth)}")
    conv = [r for r in att if truthy(r.get("converged"))]
    sl = [short_long(r) for r in conv]
    s = np.array([a for a, b in sl if a is not None])
    l_ = np.array([b for a, b in sl if b is not None])
    if len(s):
        P(f"  SIGN (converged, bond order by label distance): short bond dd med {np.median(s):+.4f} mean {s.mean():+.4f} "
          f"frac>0 {np.mean(s > 0):.2f} | long bond dd med {np.median(l_):+.4f} mean {l_.mean():+.4f} frac>0 {np.mean(l_ > 0):.2f}"
          f" | sum med {np.median(s + l_):+.4f} | long-short med {np.median(l_ - s):+.4f}")
        P(f"  quartiles short {np.percentile(s, [10, 25, 50, 75, 90]).round(4).tolist()} long "
          f"{np.percentile(l_, [10, 25, 50, 75, 90]).round(4).tolist()}")
    P("  by dipole:")
    for d in M.DIPOLES:
        table([r for r in rs if r.get("dipole") == d], f"{d[:16]:16s}")
    na = np.array([int(r.get("n_atoms") or 0) for r in rs])
    e1, e2 = np.percentile(na, [100 / 3, 200 / 3])
    P(f"  by n_atoms tertile (edges {e1:.1f}, {e2:.1f}):")
    table([r for r in rs if int(r.get("n_atoms") or 0) <= e1], f"T1 <= {e1:.0f}")
    table([r for r in rs if e1 < int(r.get("n_atoms") or 0) <= e2], f"T2 <= {e2:.0f}")
    table([r for r in rs if int(r.get("n_atoms") or 0) > e2], f"T3 > {e2:.0f}")
    if start == "g2":
        P("  by g1_class / start_same_ts:")
        for c in sorted(set(str(r.get("g1_class")) for r in rs)):
            table([r for r in rs if str(r.get("g1_class")) == c], f"g1={c}")
        for c in ("True", "False"):
            table([r for r in rs if str(r.get("start_same_ts")) == c], f"start_same={c}")


def cmd_stats():
    T = targets()
    f4 = set(T.rxn_id[T.fold_T1 == 4].astype(int))
    P(f"trace B stats; targets {len(T)}, f4_test {len(f4)}; MLT {M.MLT}")
    recs_lab, recs_g2 = load_recs("label"), load_recs("g2")
    P(f"records label {len(recs_lab)} g2 {len(recs_g2)}; in f4: {len(set(recs_lab) & f4)} / {len(set(recs_g2) & f4)}")
    hd = M.MLT / "dft" / "nurion" / "hess"
    hs = {int(Path(p).stem) for p in glob.glob(str(hd / "*.npz"))}
    P(f"label TS Hessians {hd}: {len(hs)} files; f4_test with one: {len(hs & f4)}")
    for fn in (config_evidence, lambda: audit_d3(T, recs_lab), lambda: g2_runs(T, recs_lab, recs_g2),
               lambda: rec_stats(T, recs_lab, "label"), lambda: rec_stats(T, recs_g2, "g2")):
        try:
            fn()
        except Exception:  # noqa: BLE001
            P("  ERROR " + traceback.format_exc()[-800:])
        flush("traceB_stats.txt")


# ====================================================================== nurion mode
RX_ID = re.compile(r"\d{7}")


def stream_ts_final(path, want):
    """yield (rxn_id, header dict-ish, symbols, pos, forces_eVA) for nur_ts_final label_ts=1 frames."""
    from ase.io.extxyz import key_val_str_to_dict
    with gzip.open(path, "rt") as f:
        while True:
            ln = f.readline()
            if not ln:
                return
            s = ln.strip()
            if not s:
                continue
            n = int(s.split()[0])
            hdr = f.readline()
            hit = ("nur_ts_final" in hdr) and re.search(r"\blabel_ts=1\b", hdr)
            if not hit:
                for _ in range(n):
                    f.readline()
                continue
            body = [f.readline() for _ in range(n)]
            kv = key_val_str_to_dict(hdr)
            if str(kv.get("kind")) != "nur_ts_final":
                continue
            ids = sorted(set(RX_ID.findall(str(kv.get("rxn_ids", "")))))
            if not ids:
                m = re.search(r"rxn_ids=(\S+)", hdr)
                ids = sorted(set(RX_ID.findall(m.group(1)))) if m else []
            if len(ids) != 1 or int(ids[0]) not in want:
                yield ("skip", len(ids), hdr[:600], None, None, None)
                continue
            props = str(kv["Properties"]).split(":")
            col, off = {}, 0
            for i in range(0, len(props), 3):
                nm, w = props[i], int(props[i + 2])
                col[nm] = (off, off + w)
                off += w
            toks = [b.split() for b in body]
            sym = [t[col["species"][0]] for t in toks]
            pos = np.array([[float(x) for x in t[col["pos"][0]:col["pos"][1]]] for t in toks])
            frc = np.array([[float(x) for x in t[col["REF_forces"][0]:col["REF_forces"][1]]] for t in toks])
            yield (int(ids[0]), kv, sym, pos, frc, hdr)


def cmd_nurion():
    T = targets()
    byid = T.set_index("rxn_id")
    f4 = set(T.rxn_id[T.fold_T1 == 4].astype(int))
    allid = set(T.rxn_id.astype(int))
    recs_lab, recs_g2 = load_recs("label"), load_recs("g2")
    P(f"trace B nurion; f4_test {len(f4)}")
    path = M.MLT / "dft" / "nurion.extxyz.gz"
    rows, nskip, nf = [], 0, Counter()
    for item in stream_ts_final(path, allid):
        if item[0] == "skip":
            nskip += 1
            nf[f"skip_nids_{item[1]}"] += 1
            if nskip <= 2:
                P("  skipped header sample: " + item[2].strip())
            continue
        rid, kv, sym, pos, frc, hdr = item
        nf["frames"] += 1
        g = -frc / M.EHBOHR2EVA   # Eh/bohr
        t = byid.loc[rid]
        pairs = t.pairs
        ls, lx = M.read_xyz(M.label_ts(rid))
        lx = np.asarray(lx, float)
        if list(ls) != list(sym):
            nf["order_mismatch"] += 1
            continue
        row = dict(rid=rid, fold=int(t.fold_T1), dip=t.dipole, nat=int(t.n_atoms), gmax_last=float(np.abs(g).max()),
                   fmax_info=fnum(kv.get("fmax")), dl=dists(lx, pairs),
                   dd_last=list(np.subtract(dists(pos, pairs), dists(lx, pairs))),
                   kabsch_last_label=kabsch(pos, lx)[3], n_cycles=kv.get("n_cycles"))
        h = hess_npz(rid) if (rid in f4 or nf["frames"] % 10 == 0) else None
        if h is not None and h["sym"] == list(sym):
            R, ca, cb, dev = kabsch(pos, h["X"])          # last-gradient frame -> Hessian (label TS) frame
            gh = g @ R.T
            xlast_h = (pos - ca) @ R.T + cb
            # quadratic model about the label TS: g(label) = g_last + H (x_label - x_last)
            glab = gh.ravel() + h["H"] @ ((h["X"] - xlast_h).ravel() / BOHR)
            st, stts, nneg, wmin, gpm = newton(h["H"], glab, h["X"] / BOHR)
            xs = h["X"] + st.reshape(-1, 3) * BOHR
            xt = h["X"] + stts.reshape(-1, 3) * BOHR
            dX = dists(h["X"], pairs)
            row.update(dd_newton=list(np.subtract(dists(xs, pairs), dX)),
                       dd_newton_ts=list(np.subtract(dists(xt, pairs), dX)), gmax_label_est=gpm,
                       step_max_A=float(np.abs(st.reshape(-1, 3) * BOHR).max()), n_imag=h["n_imag"],
                       imag=float(h["freq"][0]) if len(h["freq"]) else None, nneg=nneg)
        for nm, Rr in (("mlip_lab", recs_lab), ("mlip_g2", recs_g2)):
            r = Rr.get(rid)
            if r and r.get("dd"):
                row[nm] = r["dd"]
                row[nm + "_same"] = r.get("same_ts")
        rows.append(row)
    P(f"nur_ts_final label_ts=1 frames: {nf['frames']} (+{nskip} skipped multi/other ids); {dict(nf)}")
    P(f"  reactions covered {len(rows)}; f4_test covered {sum(r['fold'] == 4 for r in rows)}")
    if not rows:
        return
    gl = [r["gmax_last"] for r in rows]
    P(f"  last-gradient step |g|max (Eh/bohr): median {med(gl):.2e} p90 {q(gl, 90):.2e} max {max(gl):.2e}; frac<=3e-4 "
      f"{np.mean(np.array(gl) <= 3e-4):.3f} frac<=1e-4 {np.mean(np.array(gl) <= 1e-4):.3f}")
    dlast = [max(abs(x) for x in r["dd_last"]) for r in rows]
    P(f"  last-gradient geom vs label TS |dd| median {med(dlast):.4f} p90 {q(dlast, 90):.4f}; kabsch max-dev median "
      f"{med([r['kabsch_last_label'] for r in rows]):.4f}")
    nr = [r for r in rows if r.get("dd_newton")]
    nd = [max(abs(x) for x in r["dd_newton"]) for r in nr]
    P(f"  Newton (label-level Hessian at label TS) shift to the label-level stationary point |dd|: n={len(nr)} median "
      f"{med(nd):.4f} p90 {q(nd, 90):.4f} p99 {q(nd, 99):.4f} max {max(nd) if nd else float('nan'):.4f}; "
      f"est |g|max at label TS median {med([r['gmax_label_est'] for r in nr]):.2e}")
    P(f"  label-TS Hessian n_imag {dict(Counter(r['n_imag'] for r in nr))}; imag cm-1 median {med([r['imag'] for r in nr if r.get('imag') is not None]):.1f}")
    f4r = [r for r in nr if r["fold"] == 4 and r.get("mlip_lab")]
    P(f"== f4_test with Newton + MLIP label-start record: n={len(f4r)}")

    def ordv(r, v):
        o = np.argsort(r["dl"])
        return [v[o[0]], v[o[1]]]
    if f4r:
        ns = np.array([ordv(r, r["dd_newton"]) for r in f4r])
        ms = np.array([ordv(r, r["mlip_lab"]) for r in f4r])
        P(f"  signed (short,long) median: Newton {np.median(ns, 0).round(4).tolist()}  MLIP {np.median(ms, 0).round(4).tolist()}")
        P(f"  median |dd| max: Newton {np.median(np.abs(ns).max(1)):.4f}  MLIP {np.median(np.abs(ms).max(1)):.4f}  "
          f"MLIP-Newton {np.median(np.abs(ms - ns).max(1)):.4f}")
        for k, nm in ((0, "short"), (1, "long")):
            c = np.corrcoef(ns[:, k], ms[:, k])[0, 1]
            sg = np.mean(np.sign(ns[:, k]) == np.sign(ms[:, k]))
            P(f"  {nm}: corr(Newton, MLIP) {c:+.3f}; sign agreement {sg:.2f}; |Newton|>=0.01 frac {np.mean(np.abs(ns[:, k]) >= 0.01):.3f}")
        # would same_ts change if the reference were the Newton-corrected stationary point?
        corr = np.abs(ms - ns).max(1)
        P(f"  frac |dd|<=0.02 vs label: {np.mean(np.abs(ms).max(1) <= 0.02):.3f}; vs Newton-corrected label: {np.mean(corr <= 0.02):.3f}")
        rng = np.random.default_rng(20261105)
        pick = sorted(rng.choice(len(f4r), size=min(20, len(f4r)), replace=False))
        P("  rxn      dip              nat  d_label(s,l)    dd_newton(s,l)    dd_newton_TSmode  dd_last          dd_MLIP_lab      dd_MLIP_g2      gmax_last n_imag imag")

        def f2(v):
            return "[" + ",".join(f"{x:+.4f}" for x in v) + "]" if v is not None else "-"
        for i in pick:
            r = f4r[i]
            dls = sorted(r["dl"])
            P(f"  {r['rid']} {r['dip'][:16]:16s} {r['nat']:3d}  [{dls[0]:.3f},{dls[1]:.3f}] {f2(ordv(r, r['dd_newton']))} "
              f"{f2(ordv(r, r['dd_newton_ts']))} {f2(ordv(r, r['dd_last']))} {f2(ordv(r, r['mlip_lab']))} "
              f"{f2(ordv(r, r['mlip_g2']) if r.get('mlip_g2') else None)} {r['gmax_last']:.1e} {r['n_imag']} {r.get('imag') or float('nan'):.0f}")
    with open(OUT / "traceB_nurion_rows.json", "w") as f:
        json.dump(rows, f)


if __name__ == "__main__":
    mode = sys.argv[1]
    try:
        if mode == "stats":
            cmd_stats()
            flush("traceB_stats.txt")
        else:
            cmd_nurion()
            flush("traceB_nurion.txt")
    except Exception:  # noqa: BLE001
        P("FATAL " + traceback.format_exc()[-1500:])
        flush(f"traceB_{mode}.txt")
        raise
