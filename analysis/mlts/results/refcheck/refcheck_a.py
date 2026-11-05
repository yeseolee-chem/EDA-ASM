#!/usr/bin/env python3
"""trace A (reference provenance) summary job. Read only everywhere; writes only refcheck/summary_a*.txt."""
import gzip, io, json, os, re, sys, tarfile, time, glob
from pathlib import Path
from collections import Counter
import numpy as np

OUTD = Path("/gpfs/tmp_cpu2/yeseo1ee/mlts/refcheck")
MLT = Path("/gpfs/tmp_cpu2/yeseo1ee/mlts")
LAD = Path("/gpfs/tmp_cpu2/yeseo1ee/d1_ladder")
D1PAIR = Path("/gpfs/tmp_cpu2/yeseo1ee/d1_pair")
D1RUN = Path("/gpfs/tmp_cpu2/yeseo1ee/d1/d1_run")
D1HF = Path("/gpfs/tmp_cpu2/yeseo1ee/d1_hf")
UPLOAD = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction")
EHB2EVA = 27.211386245988 / 0.529177210903
L = []


def P(*a):
    L.append(" ".join(str(x) for x in a))


def flush(name):
    tmp = OUTD / f".{name}.tmp"
    tmp.write_text("\n".join(L) + "\n")
    os.replace(tmp, OUTD / name)


def rxyz(p):
    ln = Path(p).read_text().splitlines()
    n = int(ln[0].split()[0])
    s, x = [], []
    for t in ln[2:2 + n]:
        q = t.split()
        s.append(q[0]); x.append([float(v) for v in q[1:4]])
    return s, np.array(x)


def kabsch(Pp, Q):
    Pp = Pp - Pp.mean(0); Q = Q - Q.mean(0)
    U, _, Vt = np.linalg.svd(Pp.T @ Q)
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt)) or 1.0])
    X = Pp @ U @ D @ Vt - Q
    n = np.linalg.norm(X, axis=1)
    return float(np.sqrt((n ** 2).mean())), float(n.max())


def job_of(r):
    return f"T{int(r) - 1000000:05d}"


def q(a, p):
    a = np.asarray([v for v in a if v is not None and np.isfinite(v)], float)
    return float(np.quantile(a, p)) if len(a) else float("nan")


t0 = time.time()
# ------------------------------------------------------------ 1. D1 run config
P("== 1. D1 run config", D1RUN / "config_run.yaml")
cf = D1RUN / "config_run.yaml"
if cf.exists():
    for i, ln in enumerate(cf.read_text().splitlines(), 1):
        if re.search(r"(?i)engine|functional|dispersion|opt_basis|sp_basis|solvent|solvation|optts|g16_ts|hmethod_conf|same_ts|imag_tol|eda_level|frag_level", ln):
            P(f"  L{i}: {ln.rstrip()[:160]}")
else:
    P("  MISSING")

# ------------------------------------------------------------ 2. f4_test list
lst = MLT / "pipeline" / "lists" / "f4_test.csv"
ids = []
for ln in lst.read_text().splitlines()[1:]:
    t = ln.split(",")[0].strip()
    if t.isdigit():
        ids.append(int(t))
P("\n== 2. f4_test", lst, "n =", len(ids), "header:", lst.read_text().splitlines()[0][:120])

# ------------------------------------------------------------ 3. nurion per-tar reports: label run per job
rep_by_job = {}
for rp in sorted((MLT / "dft" / "nurion").glob("*/report.json")):
    try:
        R = json.loads(rp.read_text())
    except Exception as e:
        P("  bad report", rp, e); continue
    for j in R.get("jobs", []):
        if j.get("target"):
            rep_by_job[j["job"]] = dict(tar=R.get("tar"), ref_hess=j.get("ref_hess"), dev=j.get("ref_hess_best_dev_A"),
                                        kept=[k for k in (j.get("kept") or []) if len(k) > 5 and k[5]],
                                        markers=j.get("markers"))
P("\n== 3. nurion reports: target jobs", len(rep_by_job))
f4jobs = [job_of(r) for r in ids]
have = [j for j in f4jobs if j in rep_by_job]
P("  f4_test jobs in nurion reports:", len(have), "/", len(f4jobs),
  "with label-run hess:", sum(1 for j in have if rep_by_job[j]["ref_hess"]))
rh = Counter()
for j in have:
    p = rep_by_job[j]["ref_hess"] or ""
    rh[re.sub(r"TS_[^/]*", "TS_*", p)] += 1
P("  label-run file patterns (f4_test):", dict(rh.most_common(8)))
tars_present = {}
for j in have:
    t = rep_by_job[j]["tar"]
    if t not in tars_present:
        tars_present[t] = (UPLOAD / f"{t}.tar").exists()
P("  tars of f4 jobs present on disk:", sum(tars_present.values()), "/", len(tars_present))

# choose 3 example rxns: first / middle / last f4 rxn whose tar is present (else any)
cand = [r for r in ids if job_of(r) in rep_by_job and tars_present.get(rep_by_job[job_of(r)]["tar"])]
if len(cand) < 3:
    cand = ids
ex = [cand[0], cand[len(cand) // 2], cand[-1]]
P("  example rxns:", ex)

# ------------------------------------------------------------ 4. per-example provenance
P("\n== 4. examples")
for r in ex:
    T = job_of(r)
    P(f"-- rxn {r} job {T}")
    d = LAD / "dft_final" / str(r)
    P("  dft_final files:", sorted(os.listdir(d)) if d.exists() else "MISSING")
    rj = d / "result.json"
    if rj.exists():
        J = json.loads(rj.read_text())
        P("  result.json:", {k: J.get(k) for k in ("ts_header", "orca_bin", "e_ts_eh", "status", "n_imag", "imag_freqs_cm",
                                                    "hess_geom_max_dev_A", "formed_pairs_label", "d_form_dft")})
    try:
        s1, x1 = rxyz(d / "ts.xyz")
        s2, x2 = rxyz(D1PAIR / "d1_jobs" / T / "ts.xyz")
        P(f"  label ts.xyz vs d1_jobs/{T}/ts.xyz: same symbols {s1 == s2}, max raw dev {np.abs(x1 - x2).max():.2e} A, n={len(s1)}")
        P("  ts.xyz line 3:", (d / "ts.xyz").read_text().splitlines()[2])
    except Exception as e:
        P("  ts.xyz compare error", e)
    jd = D1PAIR / "d1_jobs" / T
    if jd.exists():
        P("  d1_jobs files:", sorted(os.listdir(jd))[:20])
        for fn in ("ts_result.json", "result.json"):
            if (jd / fn).exists():
                try:
                    J = json.loads((jd / fn).read_text())
                    P(f"  d1_jobs/{fn} keys:", list(J)[:30])
                    for k in ("hmethod_keywords", "keywords", "ts_keywords", "imag_freq", "imag_cm", "formed_pairs_ts", "engine", "status"):
                        if k in J:
                            P(f"    {k}: {str(J[k])[:600]}")
                except Exception as e:
                    P("  json err", e)
    hi = D1HF / "inputs" / T / "hf" / "eda.inp"
    if hi.exists():
        tx = hi.read_text().splitlines()
        P("  d1_hf eda.inp '!' lines:", [x for x in tx if x.strip().startswith("!")][:3])
    rr = rep_by_job.get(T)
    if rr:
        P("  nurion tar:", rr["tar"], "label-run hess:", rr["ref_hess"], "dev_A:", rr["dev"], "markers:", rr["markers"])
        P("  label-run kept files:", [k[:2] for k in rr["kept"]][:4])
    hz = MLT / "dft" / "nurion" / "hess" / f"{r}.npz"
    if hz.exists():
        with np.load(hz, allow_pickle=False) as z:
            pos = z["positions_A"]
            P("  hess npz keys:", list(z.keys()), "source:", str(z["source"]))
            for k in ("n_imag", "freq_low", "dev_ref_raw_A", "dev_ref_kabsch_A"):
                if k in z:
                    P(f"    {k}: {z[k]}")
            try:
                P("  label ts.xyz vs DFT-Hessian (label OptTS final) geometry kabsch rmsd/max:", kabsch(pos, x1))
            except Exception as e:
                P("   kabsch err", e)
flush("summary_a.txt")

# ------------------------------------------------------------ 5. stationarity of all f4_test label TSs from nurion ts_final records
P("\n== 5. label-level gradient at the label TS (nurion nur_ts_final, label_ts=1; = last step of the converged label OptTS)")
f4set = set(f4jobs)
fm, df, kinds_lab = {}, {}, Counter()
alljobs_fm = []
ext = MLT / "dft" / "nurion.extxyz.gz"
re_kv = {k: re.compile(r"(?:^|\s)" + k + r"=(\"[^\"]*\"|\S+)") for k in ("kind", "label_ts", "fmax", "job", "dform_label_A", "level", "source")}
nfr = 0
try:
    with gzip.open(ext, "rt") as f:
        while True:
            h = f.readline()
            if not h:
                break
            n = int(h.split()[0])
            c = f.readline()
            for _ in range(n):
                f.readline()
            nfr += 1
            if "label_ts=" not in c:
                continue
            g = {k: (m.group(1).strip('"') if (m := rx.search(c)) else None) for k, rx in re_kv.items()}
            if g["label_ts"] not in ("1", "T", "True"):
                continue
            kinds_lab[(g["kind"], g["level"], g["source"])] += 1
            fx = float(g["fmax"]) if g["fmax"] else float("nan")
            alljobs_fm.append(fx)
            if g["job"] in f4set:
                fm[g["job"]] = fx
                df[g["job"]] = float(g["dform_label_A"]) if g["dform_label_A"] else float("nan")
except Exception as e:
    P("  extxyz read error", type(e).__name__, e)
P(f"  frames read {nfr}; label_ts=1 frames {len(alljobs_fm)}; kinds {dict(kinds_lab)}")
v = list(fm.values())
P(f"  f4_test with label_ts frame: {len(v)} / {len(f4jobs)}")
P("  max|grad| Eh/bohr at label TS (f4_test): median %.2e p90 %.2e p99 %.2e max %.2e; n>3e-4 (ORCA NormalOpt TolMaxG) %d; n>1e-4 %d"
  % (q(v, .5), q(v, .9), q(v, .99), q(v, 1.0), sum(x > 3e-4 for x in v), sum(x > 1e-4 for x in v)))
P("  same in kcal/mol/A: median %.3f p90 %.3f max %.3f" % (q(v, .5) * 627.5095 / 0.529177, q(v, .9) * 627.5095 / 0.529177, q(v, 1) * 627.5095 / 0.529177))
P("  all targets: median %.2e p90 %.2e max %.2e" % (q(alljobs_fm, .5), q(alljobs_fm, .9), q(alljobs_fm, 1.0)))
dv = list(df.values())
P("  dform_label_A of those frames (should be ~0): median %.2e max %.2e" % (q(dv, .5), q(dv, 1.0)))
flush("summary_a.txt")

# ------------------------------------------------------------ 6. DFT Hessian at the label TS: n_imag for f4_test
P("\n== 6. label-level DFT Hessian at label TS ($MLT/dft/nurion/hess/<rxn>.npz)")
ni, dk = Counter(), []
for r in ids:
    hz = MLT / "dft" / "nurion" / "hess" / f"{r}.npz"
    if not hz.exists():
        ni["missing"] += 1; continue
    with np.load(hz, allow_pickle=False) as z:
        ni[int(z["n_imag"]) if "n_imag" in z else "?"] += 1
        if "dev_ref_kabsch_A" in z:
            dk.append(float(z["dev_ref_kabsch_A"]))
P("  n_imag counts:", dict(ni), "; hess-geom vs label TS kabsch max dev: max %.2e" % (max(dk) if dk else float("nan")))

# ------------------------------------------------------------ 7. MLIP FT-2 records vs label stationarity
P("\n== 7. FT-2 records FTpilot_AIMNet2_4 (label start)")
recs = {}
for p in glob.glob(str(MLT / "ft2" / "FTpilot_AIMNet2_4" / "label" / "recs" / "*.json")):
    try:
        R = json.loads(Path(p).read_text())
        recs[int(R["rxn_id"])] = R
    except Exception:
        pass
ok = [R for R in recs.values() if R.get("dd_max") is not None]
P(f"  recs {len(recs)}, with dd_max {len(ok)}; status {dict(Counter(R.get('status') for R in recs.values()))}")
if ok:
    dd = [R["dd_max"] for R in ok]
    P("  dd_max median %.4f p90 %.4f; same_ts frac %.3f; rmsd_heavy median %.4f" %
      (q(dd, .5), q(dd, .9), np.mean([bool(R.get("same_ts")) for R in ok]), q([R.get("rmsd_heavy") for R in ok], .5)))
    sg = Counter()
    for R in ok:
        for x in (R.get("dd") or []):
            sg["neg" if x < 0 else "pos"] += 1
    P("  sign of dd (all forming pairs):", dict(sg), " mean dd %.4f" % np.mean([x for R in ok for x in (R.get("dd") or [])]))
    pr = [(fm.get(job_of(R["rxn_id"])), R["dd_max"]) for R in ok if job_of(R["rxn_id"]) in fm]
    if len(pr) > 10:
        a = np.array(pr)
        from scipy.stats import spearmanr
        P("  spearman(label-TS DFT fmax, MLIP dd_max) = %.3f over %d" % (spearmanr(a[:, 0], a[:, 1])[0], len(a)))
        lo = a[a[:, 0] <= np.median(a[:, 0])]
        P("  dd_max median for label fmax <= median: %.4f ; > median: %.4f" % (np.median(lo[:, 1]), np.median(a[a[:, 0] > np.median(a[:, 0])][:, 1])))
    P("  gmax_final_eh_bohr (MLIP) median %.2e; converged frac %.3f" % (q([R.get("gmax_final_eh_bohr") for R in ok], .5),
                                                                       np.mean([bool(R.get("converged")) for R in ok])))
flush("summary_a.txt")

# ------------------------------------------------------------ 8. training inputs (G1, G2) headers
P("\n== 8. training input headers")
def first_files(root, pat, k=1, maxn=200000):
    out, n = [], 0
    for dp, dn, fn in os.walk(root):
        dn.sort(); fn.sort()
        for x in fn:
            n += 1
            if re.search(pat, x):
                out.append(Path(dp) / x)
                if len(out) >= k:
                    return out
        if n > maxn:
            break
    return out

def header(p, nmax=25):
    op = gzip.open if str(p).endswith(".gz") else open
    with op(p, "rt", errors="replace") as f:
        txt = f.read(400000)
    if p.name.endswith((".inp",)):
        lines = []
        for ln in txt.splitlines():
            if re.match(r"^\s*[A-Z][a-z]?\s+-?\d+\.\d+\s+-?\d+\.\d+\s+-?\d+\.\d+", ln):
                continue
            lines.append(ln)
        return lines[:nmax]
    m = re.findall(r"\|\s*\d+>\s?(.*)", txt)
    return [x for x in m if not re.match(r"^\s*[A-Z][a-z]?\s+-?\d+\.\d+", x)][:nmax]

for grp, pat in (("G1", r"\.(inp|out(\.gz)?)$"), ("G2", r"\.inp$"), ("D0G1", r"\.(inp|out(\.gz)?)$")):
    root = MLT / "dft" / "runs" / grp
    fs = first_files(root, pat, 1) if root.exists() else []
    P(f"  {grp}: {fs[0] if fs else 'none found under ' + str(root)}")
    if fs:
        for ln in header(fs[0]):
            P("    |", ln[:150])
flush("summary_a.txt")

# ------------------------------------------------------------ 9. D0 reference
P("\n== 9. D0 reference (d0_holdout_ref.json)")
dr = MLT / "dft" / "plan" / "d0_holdout_ref.json"
if dr.exists():
    J = json.loads(dr.read_text())
    u = (J.get("units") or [{}])[0]
    P("  n units", len(J.get("units") or []), "; unit0 reference:", str(u.get("reference"))[:300])
P(f"\nelapsed {time.time() - t0:.0f} s")
flush("summary_a.txt")

# ------------------------------------------------------------ 10. actual label OptTS input echo from the Nurion tar (examples)
L2 = L[:]
L.clear()
P("== 10. label-run ORCA OptTS input echo for the example rxns (streamed from the Nurion tar, read only)")
want = {job_of(r): r for r in ex if job_of(r) in rep_by_job}
bytar = {}
for T in want:
    bytar.setdefault(rep_by_job[T]["tar"], []).append(T)
found = {}

def scan_out(name, data, T):
    lbl = rep_by_job[T]["ref_hess"] or ""
    stem = lbl[:-len(".hess")].rsplit("/", 1)[-1] if lbl else None
    b = name.rsplit("/", 1)[-1]
    if not re.search(r"optts.*_orca\d*\.out(\.gz)?$", b):
        return
    if stem and not b.startswith(stem):
        return
    if b.endswith(".gz"):
        data = gzip.decompress(data)
    txt = data.decode(errors="replace")
    echo = re.findall(r"\|\s*\d+>\s?(.*)", txt)
    echo = [x for x in echo if not re.match(r"^\s*[A-Z][a-z]?\s+-?\d+\.\d+", x)]
    oth = []
    for ln in txt.splitlines():
        if re.search(r"(?i)(Program Version|TolE |TolRMSG|TolMaxG|TolRMSD|TolMaxD|Grid|OPTIMIZATION HAS CONVERGED|ORCA TERMINATED|SMD-CDS|Epsilon|Solvent:|Dispersion correction|Number of imaginary)", ln):
            if len(oth) < 40 and ln.strip() not in oth:
                oth.append(ln.strip())
    found.setdefault(T, []).append(dict(file=name, echo=echo[:30], other=oth, n_cycles=len(re.findall("GEOMETRY OPTIMIZATION CYCLE", txt))))

def walk(tf, T, depth=0):
    for m in tf:
        if not m.isfile():
            continue
        nm = m.name
        if nm.endswith((".tgz", ".tar.gz", ".tar")) and depth < 3:
            try:
                blob = tf.extractfile(m).read()
                with tarfile.open(fileobj=io.BytesIO(blob), mode="r:*") as t2:
                    walk(t2, T, depth + 1)
            except Exception as e:
                P("  nested err", nm, e)
        elif re.search(r"optts.*_orca\d*\.out(\.gz)?$", nm):
            scan_out(nm, tf.extractfile(m).read(), T)

RE_UNIT = re.compile(r"^(?:\./)?(?:.*/)?(jobs[A-Za-z0-9_]*)/([^/]+)/(.+)$")
for tar, Ts in bytar.items():
    tp = UPLOAD / f"{tar}.tar"
    P(f"  tar {tp} exists={tp.exists()} jobs {Ts}")
    if not tp.exists():
        continue
    seen, t1 = set(), time.time()
    try:
        with tarfile.open(tp, mode="r|*") as tf:
            for m in tf:
                mm = RE_UNIT.match(m.name)
                if not mm:
                    continue
                jb = mm.group(2)
                if seen >= set(Ts) and jb not in Ts:
                    break
                if jb not in Ts or not m.isfile():
                    continue
                seen.add(jb)
                rest = mm.group(3)
                if rest.endswith((".tgz", ".tar.gz", ".tar")):
                    blob = tf.extractfile(m).read()
                    try:
                        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:*") as t2:
                            walk(t2, jb)
                    except Exception as e:
                        P("  arc err", rest, e)
                elif re.search(r"optts.*_orca\d*\.out(\.gz)?$", rest):
                    scan_out(rest, tf.extractfile(m).read(), jb)
    except Exception as e:
        P("  tar err", type(e).__name__, e)
    P(f"  streamed {time.time() - t1:.0f} s")
for T, lst_ in found.items():
    P(f"-- {T} (rxn {want[T]}) label-run hess {rep_by_job[T]['ref_hess']}: {len(lst_)} matching OptTS outputs")
    for e in lst_[:1]:
        P("   file:", e["file"], "cycles:", e["n_cycles"])
        for ln in e["echo"]:
            P("    >", ln[:150])
        for ln in e["other"][:18]:
            P("    #", ln[:150])
L[:0] = L2
flush("summary_a.txt")
print("done")
