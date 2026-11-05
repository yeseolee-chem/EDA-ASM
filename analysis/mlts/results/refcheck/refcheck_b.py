#!/usr/bin/env python3
"""trace A part B: actual label-run ORCA OptTS inputs/outputs on GPFS for f4_test D1 jobs (UBAI-resumed trees under
/gpfs/tmp_cpu2/yeseo1ee/d1) + d1_jobs provenance files + G2/G1 training input echo. Read only; writes summary_b.txt."""
import gzip, json, os, re, time
from pathlib import Path

OUTD = Path("/gpfs/tmp_cpu2/yeseo1ee/mlts/refcheck")
MLT = Path("/gpfs/tmp_cpu2/yeseo1ee/mlts")
D1 = Path("/gpfs/tmp_cpu2/yeseo1ee/d1")
D1PAIR = Path("/gpfs/tmp_cpu2/yeseo1ee/d1_pair")
L = []


def P(*a):
    L.append(" ".join(str(x) for x in a))


def flush():
    t = OUTD / ".summary_b.tmp"
    t.write_text("\n".join(L) + "\n")
    os.replace(t, OUTD / "summary_b.txt")


def job_of(r):
    return f"T{int(r) - 1000000:05d}"


def rd(p):
    op = gzip.open if str(p).endswith(".gz") else open
    with op(p, "rt", errors="replace") as f:
        return f.read()


COORD = re.compile(r"^\s*[A-Z][a-z]?\s+-?\d+\.\d+\s+-?\d+\.\d+\s+-?\d+\.\d+")
KEYL = re.compile(r"(?i)(Program Version|TolE\b|TolRMSG|TolMAXG|TolRMSD|TolMAXD|Strict Convergence|Grid|IntAcc|"
                  r"OPTIMIZATION HAS CONVERGED|ORCA TERMINATED|SMD|Epsilon|Solvent|Dispersion|Number of imaginary|"
                  r"Exchange Functional|Correlation Functional|LDA part|Basis Dimension|RIJCOSX|Your calculation utilizes the basis)")


def show_out(p, tag):
    txt = rd(p)
    echo = [x for x in re.findall(r"\|\s*\d+>\s?(.*)", txt) if not COORD.match(x)]
    P(f"   [{tag}] {p}  cycles={len(re.findall('GEOMETRY OPTIMIZATION CYCLE', txt))}")
    for x in echo[:22]:
        P("     >", x[:150])
    seen = []
    for ln in txt.splitlines():
        s = ln.strip()
        if KEYL.search(s) and s not in seen:
            seen.append(s)
    for s in seen[:30]:
        P("     #", s[:150])


t0 = time.time()
ids = [int(l.split(",")[0]) for l in (MLT / "pipeline/lists/f4_test.csv").read_text().splitlines()[1:] if l[:1].isdigit()]
f4 = {job_of(r): r for r in ids}
P("== B1. D1 job trees on GPFS under", D1, "(UBAI port) for f4_test jobs")
hits = {}
nd = 0
for dp, dn, fn in os.walk(D1):
    nd += 1
    dpp = Path(dp)
    if dpp.name == "transition_states":
        for x in sorted(fn):
            if re.search(r"optts.*_orca\d*\.out(\.gz)?$", x):
                for part in dpp.parts:
                    m = re.fullmatch(r"(T\d{5})(?:_.*)?", part)
                    if m and m.group(1) in f4:
                        hits.setdefault(m.group(1), []).append(dpp / x)
                        break
        dn[:] = [d for d in dn if d not in ("conformers", "neb", "truncated", "initial_path")]
    dn[:] = [d for d in dn if d not in ("conformers", "sp", "single_points", "thermal", "reactants_and_products", "ref", "env", "venv", "home01")]
    if time.time() - t0 > 1800:
        P("  walk time limit hit"); break
P(f"  dirs walked {nd}; f4_test jobs with OptTS outputs on GPFS: {len(hits)}")
rep = {}
for rp in sorted((MLT / "dft" / "nurion").glob("*/report.json")):
    try:
        R = json.loads(rp.read_text())
        for j in R.get("jobs", []):
            if j.get("target"):
                rep[j["job"]] = j.get("ref_hess") or ""
    except Exception:
        pass
shown = 0
for T in sorted(hits):
    lab = rep.get(T, "")
    stem = lab.rsplit("/", 1)[-1][:-len(".hess")] if lab else None
    P(f"-- {T} rxn {f4[T]} label-run (nurion report): {lab}")
    pick = [p for p in hits[T] if stem and p.name.startswith(stem)] or hits[T][:1]
    show_out(pick[0], "label-run" if stem and pick[0].name.startswith(stem) else "first optts")
    inp = Path(re.sub(r"\.out(\.gz)?$", ".inp", str(pick[0])))
    if inp.exists():
        P("     inp non-coordinate lines:", [x for x in rd(inp).splitlines() if not COORD.match(x)][:20])
    shown += 1
    flush()
    if shown >= 3:
        break

P("\n== B2. d1_jobs provenance (label.json / source.txt / sp) for 3 f4_test jobs")
for r in (1000001, 1005855, 1011376):
    T = job_of(r)
    jd = D1PAIR / "d1_jobs" / T
    P(f"-- {T}")
    if (jd / "source.txt").exists():
        for x in rd(jd / "source.txt").splitlines()[:12]:
            P("   source.txt:", x[:200])
    if (jd / "label.json").exists():
        J = json.loads(rd(jd / "label.json"))
        P("   label.json keys:", list(J)[:40])
        for k, v in J.items():
            if re.search(r"(?i)key|level|method|header|engine|orca|ts_|imag|source|geom|optts|hmethod", k):
                P(f"     {k}: {str(v)[:400]}")
    if (jd / "sp").exists():
        P("   sp files:", sorted(os.listdir(jd / "sp"))[:12])
        for x in sorted((jd / "sp").glob("*.inp"))[:1]:
            P("   sp", x.name, "'!' lines:", [y for y in rd(x).splitlines() if y.strip().startswith("!")])
flush()

P("\n== B3. training data: one G2 label-level OptTS and one G1 EnGrad (echo from output)")
for grp, pat in (("G2", r"\.out(\.gz)?$"), ("G1", r"\.out(\.gz)?$")):
    root = MLT / "dft" / "runs" / grp
    got = None
    n = 0
    for dp, dn, fn in os.walk(root):
        dn.sort()
        for x in sorted(fn):
            n += 1
            if re.search(pat, x):
                got = Path(dp) / x
                break
        if got or n > 300000:
            break
    if got:
        show_out(got, grp)
    else:
        P(f"   {grp}: no .out under {root}")
    flush()
P(f"\nelapsed {time.time() - t0:.0f} s")
flush()
