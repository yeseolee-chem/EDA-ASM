#!/usr/bin/env python3
"""verify_data.py — DEPLOY-V2 (MLTS) audit before the launcher is unpaused (user request 2026-11-05: (1) data-source
consistency, (2) leakage by identity, (3) D3 consistency, + extra FT-1 / optimizer numbers). Spec
docs/specs/DEPLOY_V2_MLTS.md 5; DECISIONS.md / DEVIATIONS.md. Compute nodes only (sbatch jobs/mlt.slurm, MLTENV=mlts).
Every input is read only; writes go to $MLT/audit/ (parts, ORCA records, checkpoint snapshot, exported test model)
and results/AUDIT_report.json (report). gen_*.py, finetune.py, pipeline.py, mlt_common.py, orca_io.py, inv_nurion.py,
mlts_bridge.py, tests_c1.py are imported, never modified.

Sources of the fold0 training data (finetune.py inputs; FT-cap tiers = finetune.tier_of):
  S1  D1 G + Phase I: input G1 / G2 / G3, and I rows that are not d0_* kinds (G3L / Part B / V conformers)
  S2  Nurion TS-near: input NUR, tier 2 (nur_ts_final, or a nur_ TS kind with dform_label_A <= 0.3 A)
  S3  Nurion other: input NUR, tier 3
  S4  D0: input D0G1 / D0G2 / D0G3, and Phase I d0_* kinds (d1_validation runs)

  sample            fold0 TRAIN draws for check 1: per source 10 same-composition pairs of one reaction / molecule
                    (seeded; strata = the source's kinds, round-robin by count), + 20 label TSs for check 3 (seeded:
                    5 Si, 5 S, 2 F, 2 Cl, 2 Br, 4 C/H/N/O only) -> $MLT/audit/sample.json (kept if present)
  orca [--workers]  label-level EnGrad of every drawn structure exactly as gen_dft.run_struct (mlt_common.
                    orca_grad_input, serial ORCA, node-local tmp, gen_dft.engrad, up to 2 attempts) -> $MLT/audit/orca/
                    <tag>.json + .out.gz (existing records are skipped)
  consist           check 1 numbers vs the stored labels (REF_energy / e_eh, REF_forces) -> part_consist.json
  levels            check 1 table from input echoes / output headers (I_sources.csv, gen_dft run outputs) and the
                    inv_nurion acceptance rules + N_report exclusions -> part_levels.json
  d3                check 3: AIMNet2 DFTD3 of the trained FTpilot_AIMNet2_0 (B3LYP-D3(BJ) swapped in) vs ORCA
                    'Edisp' + 'DISPERSION GRADIENT' and simple-dftd3 (two-body), float32 / float64 -> part_d3.json
  leak [--workers]  check 2: reactant-pair identity of the fold-0 test reactions (S_folds fold_T1 == 0: a_rxn + b_rxn)
                    vs every fold0 train / valid structure -> part_leak.json (+ $MLT/audit/leak_remove.json if any
                    reaction-pair overlap); both criteria (stereo removed / stereo kept) are reported per set and, per
                    user 11-05 (1), per role (a = a_rxn, b = the reference b_rxn): by_role of the reaction-pair,
                    molecule and product overlaps (structures hit, distinct test reactions hit)
  ft1src            (GPU) FT-1 of FTpilot_AIMNet2_0 by source on fold0 valid / test (+ D0 holdout TS rows) -> part_ft1
  c1opt             (GPU) tests_c1 (ii) optimizer parity (ORCA ExtOpt OptTS vs Sella) with the fine-tuned AIMNet2
                    (exported to $MLT/audit/models when the pilot has not exported yet) -> part_c1.json
  report            merge the parts -> results/AUDIT_report.json
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import copy  # noqa: E402
import gzip  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import socket  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mlt_common as M  # noqa: E402
import orca_io as O  # noqa: E402

CODE = "mlts-audit-1"
AUD = M.MLT / "audit"
ORC = AUD / "orca"
CKPT = AUD / "ckpt"
SPLIT = M.MLT / "ft" / "data" / "fold0"
RUN = "FTpilot_AIMNet2_0"
RUN_DIR = M.MLT / "ft" / RUN
FINAL_MODEL = M.MLT / "models" / f"{RUN}.jpt"
TRAIN_JSON = M.RES / "ft" / f"{RUN}_train.json"
OUT = M.RES / "AUDIT_report.json"
LEAK_REMOVE = AUD / "leak_remove.json"
I_SOURCES = M.RES / "I_sources.csv"
N_REPORT = M.RES / "N_report.json"
C1_JSON = M.RES / "C1_tests.json"
D1_MANIFEST = M.D1RUN / "d1_manifest.csv"
LAD_PAIRS = M.LAD / "F_pairs_cr2.csv"
D0_SMILES = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/dipolar_cycloaddition/full_dataset.csv")
D0_HOLDOUT = M.MLT / "dft" / "plan" / "d0_holdout.csv"
EXISTING = M.MLT / "dft" / "existing.extxyz"
GEN_D0G1 = M.MLT / "dft" / "gen_D0G1.extxyz"
RUNS = M.MLT / "dft" / "runs"
SOURCES = ("S1", "S2", "S3", "S4")
SRC_DESC = {"S1": "D1 G + Phase I (G1, G2, G3, existing.extxyz non-d0 kinds: G3L / Part B / V conformers)",
            "S2": "Nurion TS-near (FT-cap tier 2: nur_ts_final or nur_ TS kind within 0.3 A of the label TS)",
            "S3": "Nurion other (FT-cap tier 3)",
            "S4": "D0 (D0G1 / D0G2 / D0G3 + Phase I d0_* kinds = d1_validation runs)"}
N_PAIR = 10
F_TOL = 0.2                      # kcal/mol/A, user rule: 1/3 of the FT-1 force criterion 0.6
FT1_F = 0.6
D3_TOL_E, D3_TOL_G = 1e-6, 1e-6  # Eh, Eh/bohr (user)
EV2KCAL = M.EH2KCAL / M.EH2EV
MOL_KINDS = ("react_opt", "react_ref", "react_disp", "d0_react_opt", "d0_react_ref", "d0_react_disp",
             "nur_react_opt", "nur_ref_opt", "nur_x_react_opt", "nur_x_ref_opt")
PROD_KINDS = ("d0_prod_opt", "nur_prod_opt", "nur_x_prod_opt")
MOL_ALL = MOL_KINDS + PROD_KINDS
JOB_TOKENS = {"ENGRAD", "OPT", "OPTTS", "LOOSEOPT", "TIGHTOPT", "VERYTIGHTOPT", "NORMALOPT", "FREQ", "NUMFREQ", "ANFREQ",
              "SP", "COPT", "ZOPT", "IRC", "NEB", "NEB-TS", "EXTOPT"}
LEVEL_FIELDS = ("version", "method", "x", "c", "lda", "hfx", "rijcosx", "basis", "auxj", "d3_params", "d3_version",
                "bj", "eps", "smd_block", "cds", "intacc", "angular", "radial")
CONV_FIELDS = ("scf_tole", "thresh", "tcut")
CMP_FIELDS = LEVEL_FIELDS + CONV_FIELDS
D3_B3LYP = dict(s6=1.0, s8=1.9889, s9=0.0, a1=0.3981, a2=4.4211, alp=14.0)
D3_CUT31 = dict(disp2=math.sqrt(9000.0), disp3=40.0, cn=40.0)     # inputs_mlts.D3_CUTOFF_BOHR (DFTD3 3.1 defaults)

RXS = {
    "version": re.compile(r"Program Version\s+(\S+)"),
    "x": re.compile(r"Exchange Functional\s+Exchange\s+\.+\s+(\S+)"),
    "c": re.compile(r"Correlation Functional\s+Correlation\s+\.+\s+(\S+)"),
    "lda": re.compile(r"LDA part of GGA corr\.\s+LDAOpt\s+\.+\s+(\S+)"),
    "hfx": re.compile(r"Fraction HF Exchange\s+ScalHFX\s+\.+\s+(\S+)"),
    "rijcosx": re.compile(r"RIJ-COSX \(HFX calculated with COS-X\)\)\.+\s+(\S+)"),
    "basis": re.compile(r"Your calculation utilizes the basis:\s*(\S+)"),
    "auxj": re.compile(r"Your calculation utilizes the auxiliary basis:\s*(\S+)"),
    "eps": re.compile(r"^\s*Epsilon\s+\.\.\.\s+(\S+)", re.M),
    "nproc": re.compile(r"Program running with\s+(\d+)\s+parallel MPI-processes"),
    "maxiter": re.compile(r"Maximum # iterations\s+MaxIter\s+\.+\s+(\d+)"),
    "thresh": re.compile(r"Integral Threshold\s+Thresh\s+\.+\s+(\S+)"),
    "tcut": re.compile(r"Primitive CutOff\s+TCut\s+\.+\s+(\S+)"),
    "radial": re.compile(r"Radial Grid Type\s+RadialGrid\s+\.\.\.\s+(.+?)\s*$", re.M),
}
RX_INTACC = re.compile(r"General Integration Accuracy\s+IntAcc\s+\.\.\.\s+(\S+)")
RX_ANG = re.compile(r"Angular Grid \(max\. ang\.\)\s+AngularGrid\s+\.\.\.\s+(\d+\s+\(Lebedev-\d+\))")
RX_D3P = {k: re.compile(rf"\b{k} scaling factor\s*:\s*(\S+)") for k in ("s6", "a1", "s8", "a2")}
RX_EDISP = re.compile(r"Edisp/kcal,au:\s+(\S+)\s+(\S+)")
RX_ECHO = re.compile(r"^\|\s*\d+>\s?(.*)$", re.M)


# ================================================================ small helpers
def _rj(p):
    try:
        return M.read_json(p) if Path(p).is_file() else None
    except Exception:                                                          # noqa: BLE001
        return None


def _part(name):
    return AUD / f"part_{name}.json"


def _write_part(name, obj):
    obj = dict(obj, part=name, code=CODE, written=M.now(), job=os.environ.get("SLURM_JOB_ID"),
               host=socket.gethostname())
    M.write_json(_part(name), obj)
    print(f"[audit] wrote {_part(name)}", flush=True)
    return obj


def _q(v):
    a = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if not len(a):
        return dict(n=0)
    return dict(n=int(len(a)), mean=float(a.mean()), median=float(np.median(a)), q90=float(np.quantile(a, 0.9)),
                max=float(a.max()))


def _wgz(t, text):
    with gzip.open(t, "wt") as f:
        f.write(text)


def write_gz(path, text):
    return M._atomic(path, lambda t: _wgz(t, text))


def _read_text(p):
    p = str(p)
    if p.endswith(".gz"):
        with gzip.open(p, "rt", errors="replace") as f:
            return f.read()
    return Path(p).read_text(errors="replace")


def source_of(inp, kind, tier):
    inp, kind = str(inp), str(kind)
    if inp == "NUR":
        return "S2" if int(tier) == 2 else "S3"
    if inp in ("D0G1", "D0G2", "D0G3") or (inp == "I" and kind.startswith("d0_")):
        return "S4"
    if inp in ("I", "G1", "G2", "G3"):
        return "S1"
    return "other"


def unit_of(sid, kind, inp, rxn_ids):
    """the reaction / molecule a structure belongs to (pairs for relative energies are drawn inside one unit and one
    composition)."""
    s = str(sid)
    p = s.split(":")
    if inp == "NUR":
        job = p[1] if len(p) > 1 else s
        return f"N:{job}:{p[2]}" if (kind in MOL_ALL and len(p) > 2) else f"N:{job}"
    if str(inp).startswith("D0G"):
        m = re.match(r"d0g\d_(d0m[0-9a-f]+)_", s)
        if m:
            return m.group(1)
        m = re.match(r"d0g\d_(\d+)_", s)
        return f"D0:{m.group(1)}" if m else s
    if inp == "I":
        if len(p) >= 3 and p[1] == "v_conf":
            return ":".join(p[:3])
        if len(p) >= 4 and p[1] == "d1val":
            return ":".join(p[:4]) if kind in MOL_ALL else ":".join(p[:3])
        return f"R:{rxn_ids}" if rxn_ids else ":".join(p[:3])
    m = re.match(r"(g3_m[0-9a-f]+)_", s)
    if m:
        return m.group(1)
    return f"R:{rxn_ids}" if rxn_ids else s


def load_set(name):
    """(atoms list, DataFrame: i, struct_id, kind, source, input, rxn_ids, e_eh, charge, n_atoms, formula, dform, tier,
    src) of fold0 <name>.xyz."""
    import pandas as pd
    import finetune as F
    t0 = time.time()
    al = M.read_extxyz(SPLIT / f"{name}.xyz")
    rows = []
    for i, a in enumerate(al):
        f = a.info
        d = f.get("dform_label_A")
        rows.append(dict(i=i, struct_id=str(f.get("struct_id")), kind=str(f.get("kind")), source=str(f.get("source")),
                         input=str(f.get("input")), rxn_ids=str(f.get("rxn_ids") or ""), e_eh=float(f["e_eh"]),
                         charge=int(f.get("total_charge", 0) or 0), n_atoms=len(a),
                         formula=a.get_chemical_formula(mode="hill"),
                         dform=float(d) if d is not None else float("nan")))
    D = pd.DataFrame(rows)
    D["tier"] = F.tier_of(D)
    D["src"] = [source_of(a, k, t) for a, k, t in zip(D["input"], D["kind"], D["tier"])]
    print(f"[audit] read {name}: {len(al)} structures in {time.time() - t0:.0f} s; sources {dict(Counter(D['src']))}",
          flush=True)
    return al, D


# ================================================================ ORCA output settings
def settings_from_text(txt):
    """level / grid / SCF settings of one ORCA output (input echo + printed settings)."""
    i_end = txt.find("****END OF INPUT****")
    echo = "\n".join(RX_ECHO.findall(txt[:i_end] if i_end > 0 else txt[:200000]))
    bang = " ".join(" ".join(ln.split("#")[0].split()) for ln in echo.splitlines() if ln.strip().startswith("!"))
    toks = bang.upper().replace("!", " ").split()

    def f1(rx):
        m = rx.search(txt)
        return m.group(1).strip() if m else None

    st = dict(bang=bang, method=" ".join(sorted(t for t in toks if t not in JOB_TOKENS)),
              job_tokens=" ".join(t for t in toks if t in JOB_TOKENS))
    for k in ("version", "x", "c", "lda", "hfx", "rijcosx", "basis", "auxj", "eps", "thresh", "tcut", "radial",
              "maxiter"):
        st[k] = f1(RXS[k])
    st["nproc"] = int(f1(RXS["nproc"]) or 1)
    st["intacc"] = RX_INTACC.findall(txt)[:4]
    st["angular"] = [" ".join(x.split()) for x in RX_ANG.findall(txt)[:4]]
    i = txt.find("Convergence Tolerance:")
    m = re.search(r"TolE\s+\.+\s+(\S+)", txt[i:i + 1500]) if i >= 0 else None
    st["scf_tole"] = m.group(1) if m else None
    st["d3_params"] = {k: f1(RX_D3P[k]) for k in ("s6", "a1", "s8", "a2")}
    m = re.search(r"(DFTD3 V\S+\s+Rev\s+\d+)", txt)
    st["d3_version"] = " ".join(m.group(1).split()) if m else None
    st["bj"] = "USING Becke-Johnson damping" in txt
    st["smd_block"] = bool(re.search(r"(?i)smd\s+true", echo) and re.search(r'(?i)smdsolvent\s+"water"', echo))
    st["cds"] = "SMD CDS free energy correction energy" in txt
    st["terminated"] = "ORCA TERMINATED NORMALLY" in txt[-5000:]
    return st


def cmp_fields(st):
    return {k: (st or {}).get(k) for k in CMP_FIELDS}


def disp_grad(txt, n):
    i = txt.find("DISPERSION GRADIENT")
    if i < 0:
        return None
    g = []
    for ln in txt[i:i + 200 * (n + 20)].splitlines()[1:]:
        m = O._RE_GL.match(ln)
        if m:
            g.append([float(m.group(j).replace("D", "E")) for j in (3, 4, 5)])
        elif g:
            break
    return g if len(g) == n else None


# ================================================================ sample
def draw(S, rng, n=N_PAIR):
    """n (row a, row b, stratum kind) pairs: round-robin over the source's kinds (most rows first); a group = one
    unit (reaction / molecule) and one composition with >= 2 rows; one pair per unit; partner of the same kind if
    the group has one."""
    S = S.sort_values("i").reset_index(drop=True)
    S["g"] = S["unit"] + "|" + S["formula"]
    E = S[S["g"].map(S["g"].value_counts()) >= 2]
    vc = E["kind"].value_counts()
    order = sorted(vc.index, key=lambda k: (-int(vc[k]), k))
    picks, used_g, used_u = [], set(), set()
    active = list(order)
    while len(picks) < n and active:
        for k in list(active):
            if len(picks) >= n:
                break
            Ck = E[(E["kind"] == k) & ~E["g"].isin(used_g) & ~E["unit"].isin(used_u)]
            if not len(Ck):
                active.remove(k)
                continue
            gs = sorted(Ck["g"].unique())
            g = gs[int(rng.integers(len(gs)))]
            Gk = Ck[Ck["g"] == g]
            r0 = Gk.iloc[int(rng.integers(len(Gk)))]
            rest = E[(E["g"] == g) & (E["i"] != r0["i"])]
            same = rest[rest["kind"] == k]
            pool = same if len(same) else rest
            r1 = pool.iloc[int(rng.integers(len(pool)))]
            picks.append((r0, r1, k))
            used_g.add(g)
            used_u.add(r0["unit"])
    return picks


def pick_d3():
    T = M.targets()
    pool = []
    for rx, fold in zip(T["rxn_id"].astype(int), T["fold_T1"].astype(int)):
        p = M.label_ts(rx)
        if p.is_file():
            s, pos = M.read_xyz(p)
            pool.append((int(rx), frozenset(s), s, pos, int(fold)))
    pool.sort(key=lambda x: x[0])
    rng = np.random.default_rng([M.SEED, 7100])
    cats = [("Si", 5, lambda e: "Si" in e), ("S", 5, lambda e: "S" in e), ("F", 2, lambda e: "F" in e),
            ("Cl", 2, lambda e: "Cl" in e), ("Br", 2, lambda e: "Br" in e),
            ("CHNO", 4, lambda e: e <= {"H", "C", "N", "O"})]
    used, out = set(), []
    for name, k, fn in cats:
        cand = [x for x in pool if fn(x[1]) and x[0] not in used]
        for ii in rng.permutation(len(cand))[:k]:
            rx, es, s, pos, fold = cand[int(ii)]
            used.add(rx)
            z = sum(O.SYM2Z[x] for x in s)
            out.append(dict(tag=f"d3_{rx}", src="D3", category=name, rxn_id=rx, fold_T1=fold, struct_id=f"label_ts:{rx}",
                            kind="label_ts", n_atoms=len(s), charge=0, parity_ok=bool(z % 2 == 0), symbols=list(s),
                            pos_A=np.asarray(pos, float).tolist(), elements=sorted(es)))
    return out


def cmd_sample(force=False):
    p = AUD / "sample.json"
    if p.is_file() and not force:
        print(f"[sample] {p} exists (kept: the ORCA records are keyed by its tags)", flush=True)
        return 0
    t0 = time.time()
    al, D = load_set("train")
    D["unit"] = [unit_of(s, k, a, r) for s, k, a, r in zip(D["struct_id"], D["kind"], D["input"], D["rxn_ids"])]
    jobs, summ = [], {}
    for si, src in enumerate(SOURCES):
        rng = np.random.default_rng([M.SEED, 7000, si])
        S = D[D["src"] == src]
        picks = draw(S, rng)
        rows = []
        for j, (r0, r1, k) in enumerate(picks):
            for ab, r in (("a", r0), ("b", r1)):
                a = al[int(r["i"])]
                z = np.asarray(a.numbers)
                ch = int(r["charge"])
                dv = float(r["dform"])
                rec = dict(tag=f"c_{src}_{j:02d}{ab}", src=src, pair=j, ab=ab, stratum=k, struct_id=str(r["struct_id"]),
                           kind=str(r["kind"]), source=str(r["source"]), input=str(r["input"]), unit=str(r["unit"]),
                           formula=str(r["formula"]), tier=int(r["tier"]), dform=dv if np.isfinite(dv) else None,
                           n_atoms=len(a), charge=ch, parity_ok=bool((int(z.sum()) - ch) % 2 == 0),
                           e_eh=float(r["e_eh"]), symbols=list(a.get_chemical_symbols()),
                           pos_A=np.asarray(a.positions, float).tolist(),
                           forces_eV_A=np.asarray(a.arrays["REF_forces"], float).tolist())
                jobs.append(rec)
                rows.append({x: rec[x] for x in ("tag", "struct_id", "kind", "source", "input", "unit", "formula",
                                                 "n_atoms", "charge", "tier", "dform")})
        summ[src] = dict(desc=SRC_DESC[src], n_train_rows=int(len(S)), kinds_train=dict(Counter(S["kind"])),
                         n_pairs=len(picks), strata=[k for _a, _b, k in picks], rows=rows)
        print(f"[sample] {src}: {len(S)} train rows, {len(picks)} pairs, strata {[k for _a, _b, k in picks]}",
              flush=True)
    d3 = pick_d3()
    jobs += d3
    M.write_json(p, dict(code=CODE, written=M.now(), split=str(SPLIT), n_train=int(len(D)),
                         src_counts=dict(Counter(D["src"])), sources=summ,
                         d3_pick=[{x: j[x] for x in ("tag", "rxn_id", "category", "fold_T1", "n_atoms", "elements")}
                                  for j in d3],
                         seeds=dict(pairs=[M.SEED, 7000, "source index"], d3=[M.SEED, 7100]), jobs=jobs,
                         seconds=round(time.time() - t0, 1)))
    print(f"[sample] {len(jobs)} ORCA jobs -> {p}", flush=True)
    return 0


# ================================================================ orca
def run_one(job, local, env, G):
    tag = job["tag"]
    rp = ORC / f"{tag}.json"
    if rp.is_file():
        return "skip"
    syms = list(job["symbols"])
    pos = np.asarray(job["pos_A"], float)
    ch = int(job["charge"])
    rec = dict(tag=tag, src=job.get("src"), struct_id=job.get("struct_id"), n_atoms=len(syms), charge=ch, mult=1,
               keywords=M.LABEL_GRAD_LINE.strip(), host=socket.gethostname(), slurm_job=os.environ.get("SLURM_JOB_ID"),
               runner="gen_dft.engrad (as gen_dft.run_struct: serial ORCA, mlt_common.orca_grad_input, <= 2 attempts)")
    if not job.get("parity_ok", True):
        rec.update(ok=False, level_bad=False, reason="odd electron count for multiplicity 1")
        M.write_json(rp, rec)
        return "fail"
    text = M.orca_grad_input(syms, pos, nprocs=1, maxcore=M.LABEL_MAXCORE_MB, charge=ch, mult=1, title=tag)
    work = local / tag
    atts, res, txt = [], None, ""
    try:
        for _ in (1, 2):
            res = G.engrad(work, text, syms, pos, env)
            atts.append(dict(res["att"], ok=res["ok"], reason=res["reason"]))
            if res["level_bad"] or res["ok"]:
                break
        if (work / "g.out").is_file():
            txt = (work / "g.out").read_text(errors="replace")
            write_gz(ORC / f"{tag}.out.gz", txt)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    rec.update(attempts=atts, ok=bool(res["ok"]), level_bad=bool(res["level_bad"]), reason=res["reason"],
               core_h=float(sum(a["core_h"] for a in atts)), settings=settings_from_text(txt) if txt else None)
    if res["ok"]:
        g = np.asarray(res["grad"], float)
        rec.update(energy_eh=float(res["energy_eh"]), grad_eh_bohr=g.tolist(), e_out_eh=res.get("e_out_eh"),
                   pos_dev_A=res.get("pos_dev_A"), scf_cycles=res.get("scf_cycles"))
    if txt:
        m = RX_EDISP.search(txt)
        rec.update(edisp_kcal=float(m.group(1)) if m else None, edisp_eh=float(m.group(2)) if m else None,
                   disp_grad_eh_bohr=disp_grad(txt, len(syms)))
    M.write_json(rp, rec)
    return "ok" if res["ok"] else "fail"


def cmd_orca(workers):
    import gen_dft as G
    S = M.read_json(AUD / "sample.json")
    jobs = sorted(S["jobs"], key=lambda j: (-int(j["n_atoms"]), j["tag"]))
    local = Path(os.environ.get("TMPDIR") or "/tmp") / f"mlt_audit_{os.environ.get('SLURM_JOB_ID', 'x')}_{os.getpid()}"
    local.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    stats = Counter()
    t0 = time.time()

    def safe(j):
        try:
            return run_one(j, local, env, G)
        except Exception as e:                                                 # noqa: BLE001
            print(f"[orca] {j['tag']}: {type(e).__name__}: {e}\n{traceback.format_exc(limit=4)}", flush=True)
            return "error"
    try:
        with ThreadPoolExecutor(max(1, min(int(workers), len(jobs)))) as ex:
            futs = {ex.submit(safe, j): j["tag"] for j in jobs}
            for f in as_completed(futs):
                how = f.result()
                stats[how] += 1
                print(f"[orca] {futs[f]} {how} ({sum(stats.values())}/{len(jobs)}, {time.time() - t0:.0f} s)",
                      flush=True)
    finally:
        shutil.rmtree(local, ignore_errors=True)
    print(f"[orca] done {dict(stats)} in {time.time() - t0:.0f} s", flush=True)
    return 0 if not stats.get("error") else 1


# ================================================================ consist (check 1 numbers)
def cmd_consist():
    S = M.read_json(AUD / "sample.json")
    out = {}
    for src in SOURCES:
        J = [j for j in S["jobs"] if j.get("src") == src]
        rows, pairs, fae, dE, core = [], defaultdict(dict), [], [], 0.0
        for j in J:
            r = _rj(ORC / f"{j['tag']}.json")
            row = dict(tag=j["tag"], struct_id=j["struct_id"], kind=j["kind"], n_atoms=j["n_atoms"], charge=j["charge"],
                       pair=j["pair"], ab=j["ab"], unit=j["unit"])
            if not r or not r.get("ok"):
                row.update(ok=False, reason=(r or {}).get("reason", "no record"))
                rows.append(row)
                continue
            core += float(r.get("core_h") or 0.0)
            fref = np.asarray(j["forces_eV_A"], float)
            fnew = -np.asarray(r["grad_eh_bohr"], float) * M.EHBOHR2EVA
            d = np.abs(fnew - fref) * EV2KCAL
            de = float(r["energy_eh"]) - float(j["e_eh"])
            row.update(ok=True, dE_eh=de, f_mae_kcal_A=float(d.mean()), f_max_kcal_A=float(d.max()),
                       f_ref_rms_kcal_A=float(np.sqrt((fref ** 2).mean()) * EV2KCAL), scf_cycles=r.get("scf_cycles"))
            fae.append(d.ravel())
            dE.append(abs(de))
            pairs[j["pair"]][j["ab"]] = (float(j["e_eh"]), float(r["energy_eh"]))
            rows.append(row)
        rel = [abs((p["a"][1] - p["b"][1]) - (p["a"][0] - p["b"][0])) * M.EH2KCAL
               for p in pairs.values() if "a" in p and "b" in p]
        fa = np.concatenate(fae) if fae else np.zeros(0)
        n_ok = sum(1 for x in rows if x.get("ok"))
        fm = float(fa.mean()) if len(fa) else None
        out[src] = dict(desc=SRC_DESC[src], n_drawn=len(J), n_ok=n_ok, n_pairs_complete=len(rel),
                        strata=S["sources"].get(src, {}).get("strata"),
                        abs_dE_eh=_q(dE), rel_pair_kcal=dict(_q(rel), mae=float(np.mean(rel)) if rel else None),
                        force_mae_kcal_A=fm, force_max_kcal_A=float(fa.max()) if len(fa) else None,
                        force_rule_kcal_A=F_TOL, pass_force=bool(fm is not None and fm <= F_TOL and n_ok == len(J)),
                        core_h=core, rows=rows)
        print(f"[consist] {src}: ok {n_ok}/{len(J)} | max |dE| {out[src]['abs_dE_eh'].get('max')} Eh | rel MAE "
              f"{out[src]['rel_pair_kcal'].get('mae')} kcal/mol | F MAE {fm} max {out[src]['force_max_kcal_A']} "
              f"kcal/mol/A", flush=True)
    sets = defaultdict(lambda: defaultdict(Counter))
    for j in S["jobs"]:
        r = _rj(ORC / f"{j['tag']}.json")
        st = (r or {}).get("settings")
        if st:
            for k, v in cmp_fields(st).items():
                sets[str(j.get("src"))][k][json.dumps(v, sort_keys=True)] += 1
            sets[str(j.get("src"))]["nproc"][str(st.get("nproc"))] += 1
    out_set = {s: {k: dict(c) for k, c in d.items()} for s, d in sets.items()}
    _write_part("consist", dict(sources=out, recompute_settings=out_set,
                                definitions=dict(abs_dE="|E_recomputed - e_eh stored| (Eh)",
                                                 rel_pair="|(E_a - E_b)_recomputed - (E_a - E_b)_stored| x 627.509 "
                                                          "(kcal/mol), a / b = same reaction or molecule, same "
                                                          "composition",
                                                 force="|F_recomputed - REF_forces| per Cartesian component, eV/A -> "
                                                       "kcal/mol/A (x 23.0605); MAE over all components of the "
                                                       "source's structures (= the FT-1 force-MAE definition)")))
    return 0


# ================================================================ levels (check 1 table)
def cmd_levels():
    import pandas as pd
    import yaml
    rng = np.random.default_rng([M.SEED, 7200])

    def pickn(xs, k):
        xs = sorted(xs)
        return [xs[int(i)] for i in rng.permutation(len(xs))[:k]]
    files = []
    for grp, src in (("G1", "S1"), ("G3", "S1"), ("D0G1", "S4"), ("D0G3", "S4")):
        d = RUNS / grp
        if d.is_dir():
            names = [e.name for e in os.scandir(d) if e.name.endswith(".out.gz")]
            files += [(src, f"gen_dft run_struct {grp} (EnGrad)", d / x) for x in pickn(names, 4)]
    for grp, src in (("G2", "S1"), ("D0G2", "S4")):
        d = RUNS / grp
        if d.is_dir():
            us = [e.name for e in os.scandir(d) if e.is_dir() and (d / e.name / "ts.out.gz").is_file()]
            files += [(src, f"gen_dft run_g2 {grp} (g3l OptTS)", d / u / "ts.out.gz") for u in pickn(us, 3)]
    I = pd.read_csv(I_SOURCES, keep_default_na=False, low_memory=False)
    kept = I[I["kept"].astype(str).str.lower() == "true"]
    phase_i = {}
    for (source, sub), g in kept.groupby(["source", "sub"]):
        src = "S4" if source == "d1val" else "S1"
        phase_i[f"{source}/{sub}"] = dict(src=src, n_files_kept=int(len(g)), bang=dict(Counter(g["bang"])),
                                          orca_version=dict(Counter(g["orca_version"].astype(str))),
                                          job_type=dict(Counter(g["job_type"])),
                                          level_reason=dict(Counter(g["level_reason"])))
        ps = [p for p in g["path"] if Path(p).is_file()]
        phase_i[f"{source}/{sub}"]["n_files_on_disk"] = len(ps)
        files += [(src, f"Phase I {source}/{sub}", Path(p)) for p in pickn(ps, 2)]
    parsed = []
    for src, grp, p in files:
        try:
            st = settings_from_text(_read_text(p))
            parsed.append(dict(src=src, group=grp, path=str(p), **st))
        except Exception as e:                                                 # noqa: BLE001
            parsed.append(dict(src=src, group=grp, path=str(p), error=f"{type(e).__name__}: {e}"))
    ok = [x for x in parsed if "error" not in x]
    ref = [x for x in ok if x["group"].startswith("gen_dft run_struct G1")]

    def vals(P):
        return {k: sorted({json.dumps(x.get(k), sort_keys=True) for x in P}) for k in CMP_FIELDS}
    ref_vals = vals(ref)

    def compare(P):
        v = vals(P)
        lev = {k: dict(source=v[k], reference=ref_vals[k]) for k in LEVEL_FIELDS if v[k] != ref_vals[k]}
        conv = {k: dict(source=v[k], reference=ref_vals[k]) for k in CONV_FIELDS if v[k] != ref_vals[k]}
        return v, lev, conv
    by_src, by_group = {}, {}
    for grp in sorted({x["group"] for x in ok}):
        P = [x for x in ok if x["group"] == grp]
        v, lev, conv = compare(P)
        by_group[grp] = dict(src=P[0]["src"], n_files=len(P), values=v, level_differs=lev, convergence_differs=conv,
                             nproc=sorted({x.get("nproc") for x in P}), job_tokens=sorted({x.get("job_tokens") for x in P}),
                             bang=sorted({x.get("bang") for x in P}))
    for src in ("S1", "S4"):
        P = [x for x in ok if x["src"] == src]
        v, lev, conv = compare(P)
        by_src[src] = dict(desc=SRC_DESC[src], n_files_parsed=len(P), values=v, level_differs=lev,
                           convergence_differs=conv, level_same=bool(P) and not lev,
                           evidence="input echo + printed SCF / grid / D3 / SMD settings of the parsed outputs "
                                    "(groups below), compared with the gen_dft run_struct G1 EnGrad outputs "
                                    "(= mlt_common.orca_grad_input, the D1 config grad keyword set)")
    NR = M.read_json(N_REPORT)
    excl = NR.get("exclusions", {})
    lev_excl = {k: v for k, v in excl.items() if re.search(r"orca_version|grid|other level|%method|no SMD", k)}
    proxy = [x for x in ok if x["group"].startswith("Phase I d1val")]
    pv, plev, pconv = compare(proxy)
    cfg = {}
    try:
        cfg = yaml.safe_load(M.D1_CONFIG.read_text()).get("ts_stage", {})
    except Exception as e:                                                     # noqa: BLE001
        cfg = dict(error=str(e))
    nur = dict(n_files_parsed=0,
               note="the Nurion tars were deleted after extraction (DECISIONS 'tar 삭제'); inv_nurion kept no input "
                    "echo per file, so the level is shown by (a) the acceptance rule applied to every kept file, (b) "
                    "the N_report exclusion counts, (c) the D1 run config (same method block as the d1_validation "
                    "runs, whose outputs are parsed as a proxy), (d) the recompute numbers of check 1",
               acceptance_rule="inv_nurion._parse_tree -> inv_dft.parse_file: orca_io.label_level(input echo) "
                               "(B3LYP + D3BJ + def2-SVP, no other basis / functional, SMD water via CPCM(Water) + "
                               "'smd true' + water or SMD(water)) AND inv_dft.same_grid (RIJCOSX + def2/J on the '!' "
                               "lines, no grid keyword, no %method grid) AND 'Program Version' startswith 6.1.1 AND "
                               "terminated normally AND parsed steps == optimisation cycles (== *_trj.xyz frames)",
               exclusions=excl, level_related_exclusions=lev_excl,
               d1_config_ts_stage=cfg, orca_build="config_run.yaml: 'UBAI install = Nurion's shared_openmpi418_avx2 "
                                                  "build, all 3,030 files sha256-equal'; nprocs 8 (MPI), maxcore 3000",
               d1val_proxy=dict(n_files=len(proxy), values=pv, level_differs=plev, convergence_differs=pconv),
               level_same=bool(not lev_excl and proxy and not plev))
    by_src["S2"] = dict(nur, desc=SRC_DESC["S2"])
    by_src["S3"] = dict(nur, desc=SRC_DESC["S3"])
    _write_part("levels", dict(by_source=by_src, by_group=by_group, phase_i_records=phase_i, reference_values=ref_vals,
                               level_fields=LEVEL_FIELDS, convergence_fields=CONV_FIELDS, parsed=parsed))
    return 0


# ================================================================ d3 (check 3)
def snapshot_ckpt():
    """copy of the pilot's best checkpoint (the pilot job keeps training; the copy fixes the weights used here)."""
    import torch
    dst = CKPT / "aim_best.pt"
    src = RUN_DIR / "aim_best.pt"
    if not dst.is_file():
        M._atomic(dst, lambda t: shutil.copy2(src, t))
    info = dict(source=str(src), snapshot=str(dst), sha256=M.sha256(dst),
                mtime=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(dst.stat().st_mtime)))
    try:
        ck = torch.load(str(RUN_DIR / "aim_last.pt"), map_location="cpu", weights_only=False)
        h = ck.get("hist") or []
        be = ck.get("best_epoch")
        best = next((x for x in h if x.get("epoch") == be), None)
        info.update(epochs_done_at_read=len(h), best_epoch_at_read=be,
                    best_valid_at_read=None if best is None else {k: best.get(k) for k in
                                                                 ("loss", "mae_e_kcal", "mae_f_kcal_A", "gpu_h")})
    except Exception as e:                                                     # noqa: BLE001
        info["last_read_error"] = f"{type(e).__name__}: {e}"
    return dst, info


def cmd_d3():
    import torch
    import finetune as F
    from aimnet import constants as AC
    from aimnet import nbops, ops
    from aimnet.modules import DFTD3
    from dftd3.interface import DispersionModel, RationalDampingParam
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    S = M.read_json(AUD / "sample.json")
    ck, ckinfo = snapshot_ckpt()
    C = F.CANDIDATES["AIMNet2"]
    sd = torch.load(str(ck), map_location="cpu", weights_only=False)
    model, lk = F.aim_build(C, sd, d3="label")
    model.eval()
    dm = model.outputs.dftd3
    par = {k: float(getattr(dm, k)) for k in ("s6", "s8", "a1", "a2")}
    dm64 = copy.deepcopy(dm).double()

    class NoClamp(DFTD3):
        """diagnostic: the same module without the CN <= cnmax clamp (standard D3 has no clamp)."""

        def _calc_c6ij(self, data):
            data = ops.lazy_calc_dij_lr(data)
            d_ij = data["d_ij_lr"] * AC.Bohr_inv
            numbers = data["numbers"]
            numbers_i, numbers_j = nbops.get_ij(numbers, data, suffix="_lr")
            rcov_i, rcov_j = nbops.get_ij(self.rcov[numbers], data, suffix="_lr")
            rcov_ij = rcov_i + rcov_j
            cn_ij = 1.0 / (1.0 + torch.exp(self.k1 * (rcov_ij / d_ij - 1.0)))
            cn_ij = nbops.mask_ij_(cn_ij, data, 0.0, suffix="_lr")
            cn = cn_ij.sum(-1).unsqueeze(-1).unsqueeze(-1)
            cn_i, cn_j = nbops.get_ij(cn, data, suffix="_lr")
            c6ab = self.c6ab[numbers_i, numbers_j]
            c6ref, cnref_i, cnref_j = torch.unbind(c6ab, dim=-1)
            c6ref = nbops.mask_ij_(c6ref, data, 0.0, suffix="_lr")
            l_ij = torch.exp(self.k3 * ((cn_i - cnref_i).pow(2) + (cn_j - cnref_j).pow(2)))
            w = l_ij.flatten(-2, -1).sum(-1)
            z = torch.einsum("...ij,...ij->...", c6ref, l_ij)
            _w = w < 1e-5
            z[_w] = 0.0
            return z / w.clamp(min=1e-5)
    nc64 = NoClamp(s8=par["s8"], a1=par["a1"], a2=par["a2"], s6=par["s6"]).double()
    buf_same = {b: bool(torch.equal(getattr(dm, b).detach().double().cpu(), getattr(nc64, b).detach().cpu()))
                for b in ("c6ab", "r4r2", "rcov", "cnmax")}

    def mod_eval(mod, Z, pos, dtype):
        c = torch.tensor(np.asarray(pos, float)[None], dtype=dtype, requires_grad=True)
        data = {"coord": c, "numbers": torch.tensor([list(Z)], dtype=torch.int64)}
        data["_nb_mode"] = torch.tensor(0)
        data = nbops.calc_masks(data)
        d_ij, _r = ops.calc_distances(data)
        data["d_ij"] = d_ij
        data = mod(data)
        e = data["energy"].sum()
        g = torch.autograd.grad(e, c)[0][0]
        return float(e.detach().double()) / AC.Hartree, g.detach().double().numpy() * AC.Bohr / AC.Hartree

    def in_model(Z, pos):
        cap = {}

        def pre(mod, args):
            d = args[0]
            cap["pre"] = d["energy"]
            d["energy"] = torch.zeros_like(d["energy"])

        def post(mod, args, out):
            cap["post"] = out["energy"]
        h1 = dm.register_forward_pre_hook(pre)
        h2 = dm.register_forward_hook(post)
        try:
            c = torch.tensor(np.asarray(pos, float)[None], dtype=torch.float32, requires_grad=True)
            model(dict(coord=c, numbers=torch.tensor([list(Z)], dtype=torch.int64), charge=torch.zeros(1)))
            e = cap["post"].sum()
            g = torch.autograd.grad(e, c)[0][0]
        finally:
            h1.remove()
            h2.remove()
        return float(e.detach().double()) / AC.Hartree, g.detach().double().numpy() * AC.Bohr / AC.Hartree

    def pkg(Z, pos, cutoff):
        m = DispersionModel(np.asarray(Z, dtype=int), np.asarray(pos, float) / M.BOHR2A)
        if cutoff:
            m.set_realspace_cutoff(**D3_CUT31)
        r = m.get_dispersion(RationalDampingParam(**D3_B3LYP), grad=True)
        return float(r["energy"]), np.asarray(r["gradient"], float)

    def cn_clamped(Z, pos):
        Zt = torch.tensor(list(Z), dtype=torch.int64)
        x = torch.tensor(np.asarray(pos, float), dtype=torch.float64) * AC.Bohr_inv
        d = torch.cdist(x, x)
        rc = nc64.rcov[Zt]
        cn_ij = 1.0 / (1.0 + torch.exp(nc64.k1 * ((rc[:, None] + rc[None, :]) / d - 1.0)))
        cn_ij.fill_diagonal_(0.0)
        cn = cn_ij.sum(-1)
        over = (cn - nc64.cnmax[Zt]).clamp(min=0.0)
        return int((over > 0).sum()), float(over.max())

    names = ("aim32", "aim64", "aim_inmodel32", "aim64_noclamp", "pkg_cut31", "pkg_default")
    rows = []
    for j in S["jobs"]:
        r = _rj(ORC / f"{j['tag']}.json") or {}
        Z = [O.SYM2Z[s] for s in j["symbols"]]
        pos = j["pos_A"]
        row = dict(tag=j["tag"], set="label_ts" if j["tag"].startswith("d3_") else "check1_sample",
                   category=j.get("category"), src=j.get("src"), n_atoms=len(Z),
                   elements=sorted(set(j["symbols"])))
        try:
            v = dict(aim32=mod_eval(dm, Z, pos, torch.float32), aim64=mod_eval(dm64, Z, pos, torch.float64),
                     aim_inmodel32=in_model(Z, pos), aim64_noclamp=mod_eval(nc64, Z, pos, torch.float64),
                     pkg_cut31=pkg(Z, pos, True), pkg_default=pkg(Z, pos, False))
        except Exception as e:                                                 # noqa: BLE001
            row["error"] = f"{type(e).__name__}: {e}"
            rows.append(row)
            continue
        orca = None
        if r.get("edisp_eh") is not None and r.get("disp_grad_eh_bohr"):
            orca = (float(r["edisp_eh"]), np.asarray(r["disp_grad_eh_bohr"], float))
        row["E_orca_eh"] = orca[0] if orca else None
        row["E_eh"] = {k: v[k][0] for k in names}
        for k in names:
            if orca:
                row[f"{k}_vs_orca"] = dict(dE=abs(v[k][0] - orca[0]), dG=float(np.abs(v[k][1] - orca[1]).max()))
            row[f"{k}_vs_pkg"] = dict(dE=abs(v[k][0] - v["pkg_cut31"][0]),
                                      dG=float(np.abs(v[k][1] - v["pkg_cut31"][1]).max()))
        row["n_atoms_cn_over_cnmax"], row["max_cn_over_cnmax"] = cn_clamped(Z, pos)
        rows.append(row)

    def summ(sel):
        out = {}
        for k in names:
            for ref in ("orca", "pkg"):
                key = f"{k}_vs_{ref}"
                xs = [x[key] for x in sel if key in x]
                if xs:
                    me, mg = max(x["dE"] for x in xs), max(x["dG"] for x in xs)
                    out[key] = dict(n=len(xs), max_dE_eh=me, max_dG_eh_bohr=mg,
                                    passed=bool(me <= D3_TOL_E and mg <= D3_TOL_G))
        out["n_with_clamped_atoms"] = sum(1 for x in sel if x.get("n_atoms_cn_over_cnmax", 0) > 0)
        out["n_errors"] = sum(1 for x in sel if "error" in x)
        return out
    lab = [x for x in rows if x["set"] == "label_ts"]
    res = dict(module="aimnet.modules.DFTD3 of the trained model (finetune.aim_build(.., d3='label') on the "
                      "FTpilot_AIMNet2_0 best checkpoint): model.outputs.dftd3",
               checkpoint=ckinfo, parameters=par, b3lyp_reference=dict(D3_B3LYP), buffers_equal_to_aimnet_constants=buf_same,
               three_body="none in aimnet DFTD3 ('BJ dumping, C6 and C8 terms, without 3-body term'); ORCA D3BJ prints "
                          "E6 + E8 only; simple-dftd3 with s9 = 0",
               precision=dict(trained_model="float32: AIMNet2Base._prepare_dtype casts coord / charge to the default "
                                            "dtype at import (float32); finetune.aim_predict and mlts_bridge.AIMNet2Dense "
                                            "pass float32 coordinates; training batches float32; the export doubles "
                                            "only the atomic shifts",
                              model_required_dtypes=[str(x) for x in getattr(model, "_required_keys_dtype", [])],
                              evaluated=["float32 module (as trained)", "float32 in the full model (forward hooks, "
                                         "dense all-pairs d_ij from the AEV)", "float64 copy of the module",
                                         "float64 without the CN clamp (diagnostic)"]),
               orca_reference="label-level EnGrad (verify_data orca): 'Edisp/kcal,au:' (Eh, 12 decimals) and the "
                              "'DISPERSION GRADIENT' block (Eh/bohr, 9 decimals); ORCA 6.1.1 DFTD3 V3.1 Rev 1",
               pkg_reference="simple-dftd3 python " + _pkg_version("dftd3") + ", RationalDampingParam(B3LYP: s6 1, s8 "
                             "1.9889, a1 0.3981, a2 4.4211, s9 0); pkg_cut31 = DFTD3 3.1 cutoffs (disp2 sqrt(9000), "
                             "cn 40 bohr; the H-0 setting), pkg_default = package defaults",
               criteria=dict(dE_eh=D3_TOL_E, dG_eh_bohr=D3_TOL_G),
               summary_label_ts=summ(lab), summary_all=summ(rows), rows=rows,
               unit_note="AIMNet2 energies eV -> Eh with aimnet.constants.Hartree (27.211386024); gradients "
                         "dE/dx (eV/A) x Bohr / Hartree")
    for k, v in res["summary_label_ts"].items():
        if isinstance(v, dict):
            print(f"[d3] label TS {k}: max dE {v['max_dE_eh']:.3e} Eh, max dG {v['max_dG_eh_bohr']:.3e} Eh/bohr, "
                  f"pass {v['passed']}", flush=True)
    _write_part("d3", res)
    return 0


def _pkg_version(name):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:                                                          # noqa: BLE001
        return "?"


# ================================================================ leak (check 2)
def _rd():
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    return Chem


def gkey_mol(m):
    """connectivity identity of one molecule: element graph with every H as a node, all bonds single, canonical
    SMILES of that graph (stereo, charges, bond orders and resonance form ignored)."""
    Chem = _rd()
    rw = Chem.RWMol()
    for a in m.GetAtoms():
        b = Chem.Atom(int(a.GetAtomicNum()))
        b.SetNoImplicit(True)
        rw.AddAtom(b)
    for bd in m.GetBonds():
        rw.AddBond(int(bd.GetBeginAtomIdx()), int(bd.GetEndAtomIdx()), Chem.BondType.SINGLE)
    g = rw.GetMol()
    g.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(g)
    return Chem.MolToSmiles(g, canonical=True)


def gkey_smiles(s):
    Chem = _rd()
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    return gkey_mol(Chem.AddHs(m))


def smi_canon(s):
    """map-free, stereo-free RDKit canonical SMILES of one molecule (geo1b canon_rxn per molecule)."""
    Chem = _rd()
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    for a in m.GetAtoms():
        a.SetAtomMapNum(0)
    m = Chem.RemoveHs(m)
    Chem.RemoveStereochemistry(m)
    return Chem.MolToSmiles(m)


def smi_iso(s):
    """map-free RDKit canonical SMILES of one molecule WITH stereo (RDKit default isomeric canonical SMILES)."""
    Chem = _rd()
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    for a in m.GetAtoms():
        a.SetAtomMapNum(0)
    return Chem.MolToSmiles(Chem.RemoveHs(m))


def inchi_of(s, stereo):
    """standard InChI of the map-free molecule, with (stereo=True) or without stereo layers."""
    Chem = _rd()
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    for a in m.GetAtoms():
        a.SetAtomMapNum(0)
    m = Chem.RemoveHs(m)
    if not stereo:
        Chem.RemoveStereochemistry(m)
    try:
        return Chem.MolToInchi(m) or None
    except Exception:                                                          # noqa: BLE001
        return None


def _xyz_block(syms, pos):
    return f"{len(syms)}\n\n" + "".join(f"{s} {x:.6f} {y:.6f} {z:.6f}\n" for s, (x, y, z) in zip(syms, pos))


def geom_frags(syms, pos):
    """sorted connectivity identities of the fragments of a geometry (rdDetermineBonds.DetermineConnectivity,
    covalent radii x 1.3)."""
    Chem = _rd()
    from rdkit.Chem import rdDetermineBonds
    m = Chem.MolFromXYZBlock(_xyz_block(syms, pos))
    if m is None:
        return None
    rdDetermineBonds.DetermineConnectivity(m)
    return sorted(gkey_mol(f) for f in Chem.GetMolFrags(m, asMols=True, sanitizeFrags=False))


def geom_frag_inchis(syms, pos):
    """[(InChI with stereo, InChI without stereo)] per fragment of a geometry: DetermineConnectivity, then
    DetermineBondOrders (charge 0, charged fragments allowed, stereo from the 3D coordinates); (None, None) where
    the bond orders cannot be assigned."""
    Chem = _rd()
    from rdkit.Chem import rdDetermineBonds
    m = Chem.MolFromXYZBlock(_xyz_block(syms, pos))
    if m is None:
        return None
    rdDetermineBonds.DetermineConnectivity(m)
    out = []
    for f in Chem.GetMolFrags(m, asMols=True, sanitizeFrags=False):
        try:
            g = Chem.Mol(f)
            rdDetermineBonds.DetermineBondOrders(g, charge=0, allowChargedFragments=True, embedChiral=True)
            Chem.SanitizeMol(g)
            Chem.AssignStereochemistryFrom3D(g)
            st = Chem.MolToInchi(g) or None
            h = Chem.Mol(g)
            Chem.RemoveStereochemistry(h)
            out.append((st, Chem.MolToInchi(h) or None))
        except Exception:                                                      # noqa: BLE001
            out.append((None, None))
    return sorted(out, key=lambda t: (t[1] or "", t[0] or ""))


def stereo_class(fi, want, TSx):
    """stereo-aware check of a connectivity hit: 'same_stereo' (the stereo InChI(s) equal a test reactant pair /
    reactant / product), 'stereo_differs' (only the stereo-free InChIs agree), 'inchi_mismatch' (bond orders from the
    geometry give another InChI), 'undetermined' (bond orders not assignable)."""
    if not fi or any(a is None or b is None for a, b in fi):
        return "undetermined", set()
    st = tuple(sorted(a for a, _b in fi))
    ns = tuple(sorted(b for _a, b in fi))
    if want == "pair":
        if len(fi) != 2:
            return "undetermined", set()
        if st in TSx["pair_inchi_iso"]:
            return "same_stereo", set(TSx["pair_inchi_iso"][st])
        return ("stereo_differs" if ns in TSx["pair_inchi_ns"] else "inchi_mismatch"), set()
    if len(fi) != 1:
        return "undetermined", set()
    ki, kn = ("reac_inchi_iso", "reac_inchi_ns") if want == "mol" else ("prod_inchi_iso", "prod_inchi_ns")
    if st[0] in TSx[ki]:
        return "same_stereo", set(TSx[ki][st[0]])
    return ("stereo_differs" if ns[0] in TSx[kn] else "inchi_mismatch"), set()


def _geom_chunk(items):
    out = []
    for syms, pos in items:
        try:
            out.append(geom_frags(syms, pos))
        except Exception:                                                      # noqa: BLE001
            out.append(None)
    return out


def geom_all(al, workers):
    items = [(list(a.get_chemical_symbols()), np.asarray(a.positions, float)) for a in al]
    ch = [items[k:k + 2000] for k in range(0, len(items), 2000)]
    out = []
    with ProcessPoolExecutor(max(1, int(workers))) as ex:
        for r in ex.map(_geom_chunk, ch):
            out += r
    return out


_ID = {}


def rxn_idents(smi):
    """identities of one reaction SMILES: pair_smi (sorted stereo-free canonical reactant SMILES), pair_inchi
    (gen_d0.pair_ident: resonance-invariant InChI pair, decision D0-ident), pair_graph / reac_graph / prod_graph
    (connectivity identities), prod_smi."""
    s = str(smi or "").strip()
    if not s or ">>" not in s:
        return None
    if s in _ID:
        return _ID[s]
    import gen_d0 as G0
    out = None
    try:
        reac, prod = s.split(">>")[0].split("."), s.split(">>")[1].split(".")
        cs = [smi_canon(x) for x in reac]
        gr = [gkey_smiles(x) for x in reac]
        gp = [gkey_smiles(x) for x in prod]
        ps = [smi_canon(x) for x in prod]
        iso = [smi_iso(x) for x in reac]
        ri, rn = [inchi_of(x, True) for x in reac], [inchi_of(x, False) for x in reac]
        pi, pn = [inchi_of(x, True) for x in prod], [inchi_of(x, False) for x in prod]
        out = dict(pair_smi=tuple(sorted(cs)) if None not in cs else None, pair_inchi=G0.pair_ident(s),
                   pair_graph=tuple(sorted(gr)) if None not in gr else None,
                   reac_graph=[x for x in gr if x], prod_graph=[x for x in gp if x],
                   prod_smi=".".join(sorted(x for x in ps if x)),
                   pair_smi_iso=tuple(sorted(iso)) if None not in iso else None,
                   pair_inchi_iso=tuple(sorted(ri)) if None not in ri else None,
                   pair_inchi_ns=tuple(sorted(rn)) if None not in rn else None,
                   reac_inchi_iso=[x for x in ri if x], reac_inchi_ns=[x for x in rn if x],
                   prod_inchi_iso=[x for x in pi if x], prod_inchi_ns=[x for x in pn if x])
    except Exception:                                                          # noqa: BLE001
        out = None
    _ID[s] = out
    return out


def d1design_smiles():
    out, metas = {}, []
    jd = M.D1VAL / "jobs"
    if jd.is_dir():
        for j in sorted(p for p in jd.iterdir() if p.is_dir()):
            for n in ("label.json", "ts_result.json"):
                q = _rj(j / n)
                if q:
                    metas.append(q)
                    break
    v7 = M.D1VAL / "v7"
    if v7.is_dir():
        for d in sorted(v7.glob("*")):
            q = _rj(M.D1VAL / "pilot_copy" / "jobs" / d.name / "label.json")
            if q:
                metas.append(q)
    for m in metas:
        if m.get("ts_id") and m.get("rxn_smiles"):
            out[str(m["ts_id"])] = m["rxn_smiles"]
    return out


def cmd_leak(workers):
    import pandas as pd
    import finetune as F
    t0 = time.time()
    T = M.targets()
    lab_smi = dict(zip(T["rxn_id"].astype(int), T["rxn_smiles"]))
    Sf = pd.read_csv(M.S_FOLDS)
    P0 = Sf[Sf["fold_T1"] == 0]
    A, B_ = set(P0["a_rxn"].astype(int)), set(P0["b_rxn"].astype(int))
    test = A | B_
    pm = P0[P0["in_P_main"].astype(str).str.lower() == "true"]
    test_pm = set(pm["a_rxn"].astype(int)) | set(pm["b_rxn"].astype(int))
    tgt0 = set(T.loc[T["fold_T1"] == 0, "rxn_id"].astype(int))
    LP = pd.read_csv(LAD_PAIRS, usecols=["pair_id", "a_rxn", "b_rxn"]).drop_duplicates("pair_id").set_index("pair_id")
    miss_lp = [p for p in P0["pair_id"] if p not in LP.index]
    mism_lp = [p for p, a, b in zip(P0["pair_id"], P0["a_rxn"], P0["b_rxn"])
               if p in LP.index and (int(LP.at[p, "a_rxn"]) != int(a) or int(LP.at[p, "b_rxn"]) != int(b))]
    man = pd.read_csv(D1_MANIFEST, usecols=["job_id", "rxn_smiles", "meta_rxn_id"], keep_default_na=False, dtype=str)
    smi_job = dict(zip(man["job_id"], man["rxn_smiles"]))
    smi_mrx = {int(r): s for r, s in zip(man["meta_rxn_id"], man["rxn_smiles"]) if str(r).strip().isdigit()}
    d0 = pd.read_csv(D0_SMILES, usecols=["rxn_id", "rxn_smiles"])
    smi_d0 = dict(zip(d0["rxn_id"].astype(int), d0["rxn_smiles"]))
    d1d = d1design_smiles()
    man_vs_lab = Counter()
    for r, s in lab_smi.items():
        idn_a, idn_b = rxn_idents(s), rxn_idents(smi_mrx.get(r, ""))
        man_vs_lab["same_pair_smi" if (idn_a and idn_b and idn_a["pair_smi"] == idn_b["pair_smi"]) else "differ"] += 1

    def d1_smi(r):
        return lab_smi.get(int(r)) or smi_mrx.get(int(r))

    PAIR_KEYS = ("pair_smi", "pair_inchi", "pair_graph", "pair_smi_iso", "pair_inchi_iso", "pair_inchi_ns")
    MOL_KEYS = ("reac_graph", "prod_graph", "reac_inchi_iso", "reac_inchi_ns", "prod_inchi_iso", "prod_inchi_ns")

    def test_sets(rxns):
        TS = {k: defaultdict(set) for k in PAIR_KEYS + MOL_KEYS}
        bad = []
        for r in sorted(rxns):
            idn = rxn_idents(d1_smi(r) or "")
            if not idn:
                bad.append(int(r))
                continue
            for k in PAIR_KEYS:
                if idn[k]:
                    TS[k][idn[k]].add(int(r))
            for k in MOL_KEYS:
                for x in idn[k]:
                    TS[k][x].add(int(r))
        return TS, bad
    TS, bad = test_sets(test)
    TS0, bad0 = test_sets(tgt0)
    xf, xf_iso, nx = defaultdict(set), defaultdict(set), Counter()
    for r, f in zip(T["rxn_id"].astype(int), T["fold_T1"].astype(int)):
        idn = rxn_idents(lab_smi.get(r) or "")
        if idn:
            xf[idn["pair_smi"]].add(int(f))
            xf_iso[idn["pair_smi_iso"]].add(int(f))
    for r, f in zip(T["rxn_id"].astype(int), T["fold_T1"].astype(int)):
        idn = rxn_idents(lab_smi.get(r) or "")
        if idn and len(xf[idn["pair_smi"]]) > 1:
            nx["targets_in_multi_fold_stereo_free_pairs"] += 1
            if 0 in xf[idn["pair_smi"]]:
                nx[f"fold{f}_targets_sharing_a_stereo_free_pair_with_fold0" if f != 0 else
                   "fold0_targets_with_a_stereo_free_pair_in_another_fold"] += 1
    cross_fold = dict(stereo_free_pairs_spanning_folds=sum(1 for v in xf.values() if len(v) > 1),
                      isomeric_pairs_spanning_folds=sum(1 for v in xf_iso.values() if len(v) > 1),
                      counts=dict(nx), examples=[dict(pair=list(k), folds=sorted(v)) for k, v in xf.items()
                                                 if len(v) > 1][:15],
                      note="D1 targets (all 9,266) whose stereo-free reactant pair occurs in more than one T1 fold "
                           "(e.g. (E)- vs (Z)-2-butene cores AppE / AppZ): S_folds assigns folds by core, so stereo "
                           "siblings can sit in different folds")
    geo_chk = Counter()
    for r in sorted(test):
        p = M.label_ts(r)
        if not p.is_file():
            geo_chk["no_label_ts"] += 1
            continue
        s, pos = M.read_xyz(p)
        try:
            fr = geom_frags(s, pos)
        except Exception:                                                      # noqa: BLE001
            fr = None
        idn = rxn_idents(d1_smi(r) or "")
        if fr is None or idn is None:
            geo_chk["error"] += 1
        elif len(fr) == 2:
            geo_chk["agree" if tuple(fr) == idn["pair_graph"] else "disagree_2frag"] += 1
        elif len(fr) == 1:
            geo_chk["one_fragment"] += 1
        else:
            geo_chk["other"] += 1
    print(f"[leak] test reactions {len(test)} (a {len(A)}, b {len(B_)}); fold-0 targets {len(tgt0)}; label-TS "
          f"geometry check {dict(geo_chk)}", flush=True)
    grp_I = {str(a.info.get("struct_id")): str(a.info.get("group", "")) for a in M.read_extxyz(EXISTING)}

    def row_ids(sid, inp, rxn_ids):
        ids = []
        if inp == "NUR":
            p = sid.split(":")
            mj = re.match(r"^(T\d{5})", p[1] if len(p) > 1 else "")
            if mj:
                ids.append(("JOB", mj.group(1)))
        for x in str(rxn_ids).split(","):
            if x.strip():
                ids.append(("D1", int(x)))
        if inp.startswith("D0G"):
            r = F.d0id_from_sid(sid)
            if r >= 0:
                ids.append(("D0", int(r)))
        if inp == "I":
            g = grp_I.get(sid, "")
            m = re.fullmatch(r"D0-(\d+)", g)
            if m:
                ids.append(("D0", int(m.group(1))))
            elif re.fullmatch(r"D1-\d+-r\d+", g):
                ids.append(("D1D", g))
        return ids

    def smi_of(t, v):
        if t == "JOB":
            return smi_job.get(v)
        if t == "D1":
            return d1_smi(v)
        if t == "D0":
            return smi_d0.get(int(v))
        return d1d.get(v)

    def pair_hit(TSx, idns, fr):
        ev, trx = [], set()
        for idn in idns:
            if idn["pair_smi"] and idn["pair_smi"] in TSx["pair_smi"]:
                ev.append("smiles_pair")
                trx |= TSx["pair_smi"][idn["pair_smi"]]
            if idn["pair_inchi"] and idn["pair_inchi"] in TSx["pair_inchi"]:
                ev.append("inchi_pair")
                trx |= TSx["pair_inchi"][idn["pair_inchi"]]
        if fr is not None and len(fr) == 2 and tuple(fr) in TSx["pair_graph"]:
            ev.append("geometry_pair")
            trx |= TSx["pair_graph"][tuple(fr)]
        return sorted(set(ev)), trx

    def pair_hit_iso(TSx, idns):
        ev, trx = [], set()
        for idn in idns:
            if idn.get("pair_smi_iso") and idn["pair_smi_iso"] in TSx["pair_smi_iso"]:
                ev.append("smiles_pair_iso")
                trx |= TSx["pair_smi_iso"][idn["pair_smi_iso"]]
            if idn.get("pair_inchi_iso") and idn["pair_inchi_iso"] in TSx["pair_inchi_iso"]:
                ev.append("inchi_pair_iso")
                trx |= TSx["pair_inchi_iso"][idn["pair_inchi_iso"]]
        return ev, trx

    def sclass(a, want):
        try:
            fi = geom_frag_inchis(list(a.get_chemical_symbols()), np.asarray(a.positions, float))
        except Exception:                                                      # noqa: BLE001
            fi = None
        return stereo_class(fi, want, TS)
    # per role of the fold-0 test pairs (user 11-05 (1)): a = a_rxn (the evaluated reaction), b = b_rxn (the pair's
    # reference reaction); both criteria: stereo removed (smiles_pair | inchi_pair | geometry_pair) and stereo kept
    # (smiles_pair_iso | inchi_pair_iso | geometry_pair with the same stereo InChIs); per set (train / valid)
    ROLES = (("a", A), ("b", B_))
    role_acc = {kind: {role: dict(n_free=Counter(), n_iso=Counter(), n_unres=Counter(), rx_free=defaultdict(set),
                                  rx_iso=defaultdict(set)) for role, _R in ROLES} for kind in ("pair", "mol", "prod")}

    def role_hit(kind, nm, trx, trx_i, unresolved=False):
        for role, R in ROLES:
            acc = role_acc[kind][role]
            x, y = set(trx) & R, set(trx_i) & R
            if x:
                acc["n_free"][nm] += 1
                acc["rx_free"][nm] |= x
                if unresolved and not y:
                    acc["n_unres"][nm] += 1
            if y:
                acc["n_iso"][nm] += 1
                acc["rx_iso"][nm] |= y

    def role_out(kind):
        out = {}
        for role, R in ROLES:
            acc = role_acc[kind][role]
            out[role] = dict(n_test_rxns=len(R), **{
                crit: {s: dict(n_structures=int(acc[nk][s]), n_test_rxns_hit=len(acc[rk][s])) for s in ("train", "valid")}
                for crit, nk, rk in (("stereo_removed", "n_free", "rx_free"), ("stereo_kept", "n_iso", "rx_iso"))})
            if kind == "pair":
                out[role]["stereo_unresolved_geometry_only"] = {s: int(acc["n_unres"][s]) for s in ("train", "valid")}
        return out
    stats = defaultdict(Counter)
    hits, mol_hits, prod_hits, kinds_hit = [], [], [], defaultdict(Counter)
    kinds_hit_iso = defaultdict(Counter)
    mol_unique = defaultdict(lambda: dict(n_rows=0, test_rxns=set(), kinds=Counter(), stereo=Counter()))
    id_src = Counter()
    for nm in ("train", "valid"):
        al, D = load_set(nm)
        t1 = time.time()
        frs = geom_all(al, workers)
        print(f"[leak] {nm}: geometry identities in {time.time() - t1:.0f} s", flush=True)
        for r, fr in zip(D.itertuples(index=False), frs):
            sid, kind, inp = str(r.struct_id), str(r.kind), str(r.input)
            cls = "mol" if kind in MOL_KINDS else ("prod" if kind in PROD_KINDS else "rxn")
            st = stats[f"{nm}|{r.src}|{cls}"]
            st["n"] += 1
            if fr is None:
                st["geom_error"] += 1
            if cls == "rxn":
                ids = row_ids(sid, inp, r.rxn_ids)
                idns = []
                for t, v in ids:
                    s = smi_of(t, v)
                    idn = rxn_idents(s) if s else None
                    if idn:
                        idns.append(idn)
                        id_src[f"{nm}|{r.src}|{t}"] += 1
                if not idns:
                    st["no_tag_identity"] += 1
                if fr is not None and len(fr) == 2:
                    st["geom_2frag"] += 1
                    if idns:
                        st["tag_geom_agree" if any(i["pair_graph"] == tuple(fr) for i in idns)
                           else "tag_geom_disagree"] += 1
                elif fr is not None and len(fr) == 1:
                    st["geom_1frag"] += 1
                    if fr[0] in TS["prod_graph"]:
                        sc, _sx = sclass(al[int(r.i)], "prod")
                        st["geom_product_hit"] += 1
                        st[f"geom_product_hit_{sc}"] += 1
                        role_hit("prod", nm, TS["prod_graph"][fr[0]], _sx)
                        prod_hits.append(dict(struct_id=sid, set=nm, src=r.src, kind=kind, stereo=sc,
                                              test_rxns=sorted(TS["prod_graph"][fr[0]])[:10]))
                elif fr is not None:
                    st["geom_multi_frag"] += 1
                ev, trx = pair_hit(TS, idns, fr)
                ev0, _trx0 = pair_hit(TS0, idns, fr)
                if ev0:
                    st["pair_hit_vs_fold0_targets"] += 1
                if ev:
                    ev_i, trx_i = pair_hit_iso(TS, idns)
                    gsc = None
                    if "geometry_pair" in ev:
                        gsc, sx = sclass(al[int(r.i)], "pair")
                        st[f"geometry_pair_{gsc}"] += 1
                        if gsc == "same_stereo":
                            ev_i.append("geometry_pair_stereo")
                            trx_i |= sx
                    ev_i = sorted(set(ev_i))
                    st["pair_hit_any"] += 1
                    for e in ev:
                        st[f"pair_hit_{e}"] += 1
                    if trx & test_pm:
                        st["pair_hit_P_main_rxn"] += 1
                    unres_g = (not ev_i and gsc in ("undetermined", "inchi_mismatch")
                               and not any(e in ev for e in ("smiles_pair", "inchi_pair")))
                    role_hit("pair", nm, trx, trx_i, unresolved=unres_g)
                    if ev_i:
                        st["pair_hit_isomeric"] += 1
                        kinds_hit_iso[f"{nm}|{r.src}"][kind] += 1
                    elif unres_g:
                        st["pair_hit_geometry_only_stereo_unresolved"] += 1
                    kinds_hit[f"{nm}|{r.src}"][kind] += 1
                    hits.append(dict(struct_id=sid, set=nm, src=r.src, kind=kind, input=inp, evidence=ev,
                                     evidence_isomeric=ev_i, geometry_stereo=gsc, n_test_rxns=len(trx),
                                     test_rxns=sorted(trx)[:20], test_rxns_isomeric=sorted(trx_i)[:20],
                                     tag_ids=[f"{t}:{v}" for t, v in ids][:5]))
            else:
                rx = {int(x) for x in str(r.rxn_ids).split(",") if x.strip()}
                if rx & test:
                    st["tag_union_has_test_rxn"] += 1
                if fr is not None and len(fr) == 1:
                    tgt = TS["reac_graph"] if cls == "mol" else TS["prod_graph"]
                    if fr[0] in tgt:
                        sc, sx = sclass(al[int(r.i)], cls)
                        st["geom_molecule_hit"] += 1
                        st[f"geom_molecule_hit_{sc}"] += 1
                        role_hit(cls, nm, tgt[fr[0]], sx)
                        rec = dict(struct_id=sid, set=nm, src=r.src, kind=kind, stereo=sc, n_test_rxns=len(tgt[fr[0]]),
                                   test_rxns=sorted(tgt[fr[0]])[:10], test_rxns_same_stereo=sorted(sx)[:10])
                        if cls == "mol":
                            mol_hits.append(rec)
                            u = mol_unique[fr[0]]
                            u["n_rows"] += 1
                            u["test_rxns"] |= tgt[fr[0]]
                            u["kinds"][f"{nm}|{r.src}|{kind}"] += 1
                            u["stereo"][sc] += 1
                        else:
                            prod_hits.append(rec)
                elif fr is not None:
                    st["geom_not_single"] += 1
        del al, D, frs
    tot = Counter()
    for k, c in stats.items():
        tot.update({f"{k.split('|')[0]}|{x}": v for x, v in c.items()
                    if x.startswith(("pair_hit", "geom_molecule_hit", "geom_product_hit", "tag_union",
                                     "geometry_pair_"))})
    mol_u = sorted(({"graph": g[:300], "n_rows": u["n_rows"], "n_test_rxns": len(u["test_rxns"]),
                     "test_rxns": sorted(u["test_rxns"])[:10], "kinds": dict(u["kinds"]), "stereo": dict(u["stereo"])}
                    for g, u in mol_unique.items()), key=lambda x: -x["n_rows"])
    n_pair = len(hits)
    hits_iso = [h for h in hits if h["evidence_isomeric"]]
    unres = [h for h in hits if not h["evidence_isomeric"] and h.get("geometry_stereo") in ("undetermined",
                                                                                          "inchi_mismatch")
             and not any(e in h["evidence"] for e in ("smiles_pair", "inchi_pair"))]
    by_set = {s: dict(stereo_free=sum(1 for h in hits if h["set"] == s),
                      isomeric=sum(1 for h in hits_iso if h["set"] == s),
                      geometry_only_stereo_unresolved=sum(1 for h in unres if h["set"] == s)) for s in ("train", "valid")}
    by_role = dict(pair=role_out("pair"), mol=role_out("mol"), prod=role_out("prod"))
    role_def = ("a = S_folds a_rxn of the fold-0 test pairs (the evaluated reaction), b = b_rxn (the pair's reference "
                "reaction; a reaction can be both); per set (train / valid) the number of structures whose identity hits "
                "a test reaction of that role and the number of distinct test reactions of that role hit. stereo_removed "
                "= smiles_pair (stereo-free canonical SMILES pair) | inchi_pair (gen_d0.pair_ident) | geometry_pair "
                "(connectivity of the two fragments); stereo_kept = smiles_pair_iso (isomeric canonical SMILES pair) | "
                "inchi_pair_iso (stereo InChI pair) | geometry_pair with the fragments' stereo InChIs equal to the test "
                "pair's. mol / prod: molecule / product graph hits (stereo_kept = same stereo InChI from the geometry)")
    rem = None
    if n_pair:
        M.write_json(LEAK_REMOVE, dict(
            code=CODE, written=M.now(), split="fold0", n=n_pair, n_isomeric=len(hits_iso), by_set=by_set,
            by_role=by_role["pair"], by_role_definition=role_def,
            remove=[h["struct_id"] for h in hits], remove_isomeric=[h["struct_id"] for h in hits_iso],
            review_geometry_stereo_unresolved=[h["struct_id"] for h in unres], detail=hits,
            rule="reaction-pair overlap with the fold-0 test reactions (a_rxn + b_rxn of S_folds fold_T1 == 0). "
                 "'remove' = stereo-free identity (the task rule): smiles_pair (stereo-free canonical reactant SMILES "
                 "pair), inchi_pair (gen_d0.pair_ident) or geometry_pair (connectivity of the structure's two "
                 "fragments). 'remove_isomeric' = the subset that is the same pair WITH stereo (isomeric canonical "
                 "SMILES / stereo InChI from the tags, or stereo InChIs of the geometry's fragments). 'set' says train "
                 "or valid (valid rows are not trained on; they select the checkpoint). The lead removes and "
                 "retrains."))
        rem = str(LEAK_REMOVE)
    res = dict(test_set=dict(definition="fold-0 test pairs = analysis/d1_pair/results/S_folds.csv fold_T1 == 0; test "
                                        "reactions = their a_rxn and b_rxn",
                             n_pairs=int(len(P0)), n_a=len(A), n_b=len(B_), n_union=len(test),
                             n_b_not_in_a=len(B_ - A), n_P_main_pairs=int(len(pm)), n_P_main_rxns=len(test_pm),
                             n_fold0_targets=len(tgt0), union_minus_fold0_targets=sorted(test - tgt0)[:50],
                             n_union_minus_fold0_targets=len(test - tgt0),
                             n_fold0_targets_minus_union=len(tgt0 - test),
                             ladder_pairs=dict(file=str(LAD_PAIRS), missing=len(miss_lp), ab_mismatch=len(mism_lp),
                                               examples=(miss_lp + mism_lp)[:10]),
                             n_without_identity=len(bad), without_identity=bad[:20],
                             n_identities=dict(pair_smi=len(TS["pair_smi"]), pair_inchi=len(TS["pair_inchi"]),
                                               pair_graph=len(TS["pair_graph"]), reactant_molecules=len(TS["reac_graph"]),
                                               products=len(TS["prod_graph"])),
                             manifest_vs_labels_rxn_smiles=dict(man_vs_lab)),
               identity=dict(smiles="RDKit canonical SMILES of each reactant of the label rxn_smiles, atom maps and "
                                    "stereo removed, sorted pair",
                             isomeric="the same WITH stereo: RDKit isomeric canonical SMILES (default) and standard "
                                      "InChI with stereo layers of each reactant; for geometry hits, the fragments' "
                                      "bond orders from rdDetermineBonds.DetermineBondOrders (charge 0, charged "
                                      "fragments allowed) and stereo from the 3D coordinates -> stereo InChI",
                             inchi="gen_d0.pair_ident (decision D0-ident: standard InChI of the map- and stereo-free "
                                   "molecule; resonance / charge placement invariant)",
                             geometry="independent of the tags: rdDetermineBonds.DetermineConnectivity of the stored "
                                      "geometry (covalent radii x 1.3) -> fragments -> element graph with explicit H, "
                                      "all bonds single; a 2-fragment structure is compared with the test reactant "
                                      "pairs, a 1-fragment structure with the test products (reaction rows) or the "
                                      "test reactants (molecule rows)",
                             row_tags="D1 rows: rxn_ids -> labels_d1_cr2 rxn_smiles (manifest for non-targets); D0 rows: "
                                      "d0_id (struct id d0g1_/d0g2_/d0g3_<rid>, Phase I group D0-<rid>) -> Coley "
                                      "full_dataset.csv rxn_smiles (labels_all.json has no SMILES); Nurion rows: job "
                                      "(struct id N:<job>:...) -> d1_manifest.csv rxn_smiles; Phase I d1_validation "
                                      "D1-design groups -> $D1VAL jobs label.json",
                             classes=dict(rxn="TS-derived / path kinds (pair identity)", mol=list(MOL_KINDS),
                                          prod=list(PROD_KINDS))),
               geometry_method_check_on_test_label_ts=dict(geo_chk),
               counts=dict(stats), totals=dict(tot), identity_sources=dict(id_src),
               by_role_definition=role_def,
               reaction_pair_overlap=dict(n_structures=n_pair, n_structures_isomeric=len(hits_iso), by_set=by_set,
                                          by_role=by_role["pair"],
                                          by_set_src_kind={k: dict(v) for k, v in kinds_hit.items()},
                                          by_set_src_kind_isomeric={k: dict(v) for k, v in kinds_hit_iso.items()},
                                          examples=hits[:60], examples_isomeric=hits_iso[:60],
                                          geometry_stereo_unresolved=unres[:60], leak_remove=rem),
               cross_fold_stereo_siblings=cross_fold,
               molecule_overlap=dict(n_structures=len(mol_hits), n_unique_molecules=len(mol_u), unique=mol_u[:100],
                                     by_role=by_role["mol"], examples=mol_hits[:50],
                                     note="molecule rows whose geometry graph equals a reactant of a fold-0 test "
                                          "reaction (reported separately, not part of the reaction-pair rule)"),
               product_overlap=dict(n_structures=len(prod_hits), by_role=by_role["prod"], examples=prod_hits[:50],
                                    note="product-graph matches (1-fragment reaction rows or product optimisations) "
                                         "with a fold-0 test reaction product (reported separately)"),
               seconds=round(time.time() - t0, 1))
    print(f"[leak] reaction-pair overlap {n_pair} structures (isomeric {len(hits_iso)}, by set {by_set}); molecule "
          f"overlap {len(mol_hits)}; product overlap {len(prod_hits)}; cross-fold {cross_fold['counts']}; totals "
          f"{dict(tot)}", flush=True)
    for kind in ("pair", "mol", "prod"):
        for role, v in by_role[kind].items():
            print(f"[leak] by role {kind}/{role} ({v['n_test_rxns']} test reactions): " + "; ".join(
                f"{crit} " + ", ".join(f"{s} {v[crit][s]['n_structures']} structures / {v[crit][s]['n_test_rxns_hit']} "
                                       f"reactions" for s in ("train", "valid"))
                for crit in ("stereo_removed", "stereo_kept")), flush=True)
    _write_part("leak", res)
    return 0


# ================================================================ FT-1 by source (GPU)
def load_aim(dev):
    """(model, offset(atoms list) -> eV array, info): the exported pilot model if training is done, else the best
    checkpoint (snapshot) built with finetune.aim_build + SAE + ATOM_REF offsets (= what export_aimnet adds)."""
    import torch
    import finetune as F
    tj = _rj(TRAIN_JSON)
    if tj and tj.get("status") == "done" and FINAL_MODEL.is_file():
        m = torch.jit.load(str(FINAL_MODEL), map_location=dev)
        return m, (lambda al: np.zeros(len(al))), dict(which="final export", model=str(FINAL_MODEL),
                                                       sha256=M.sha256(FINAL_MODEL), train=dict(
                                                           best_epoch=(tj.get("epochs") or {}).get("best_epoch"),
                                                           gpu_h=tj.get("gpu_h"), stop_reason=tj.get("stop_reason")))
    ck, info = snapshot_ckpt()
    C = F.CANDIDATES["AIMNet2"]
    sd = torch.load(str(ck), map_location="cpu", weights_only=False)
    model, lk = F.aim_build(C, sd, d3="label")
    model = model.to(dev).eval()
    sae = {int(z): float(v) for z, v in M.read_json(RUN_DIR / "sae.json")["sae_eV"].items()}

    def off(al):
        return np.array([sum(sae[int(z)] for z in a.numbers) + F.atom_ref_ev(a.numbers) for a in al])
    return model, off, dict(which="best checkpoint (pilot still training: no finished export)", checkpoint=info,
                            load=dict(missing=len(lk["missing"]), unexpected=len(lk["unexpected"])))


def ft1_block(F, Dfr, Fref, Fp, mask):
    idx = np.flatnonzero(mask)
    if not len(idx):
        return dict(n=0)
    D = Dfr.iloc[idx].reset_index(drop=True)
    out = F.ft1_metrics(D, [Fref[i] for i in idx], [Fp[i] for i in idx])
    ts = F.ts_near_mask(D).to_numpy()
    ae = np.abs(D["e_abs_pred"].to_numpy() - D["e_ref"].to_numpy()) * EV2KCAL
    out["abs_e_mae_kcal"] = dict(ts_near=float(ae[ts].mean()) if ts.any() else None, all=float(ae.mean()))
    out.pop("force_mae_by_kind_kcal_A", None)
    out.pop("n_by_kind", None)
    return out


def eval_set(F, model, offset, al, src, dev, rxn_override=None):
    t0 = time.time()
    Dfr = F.frame(al)
    if rxn_override is not None:
        Dfr["rxn"] = rxn_override
    e, Fp = F.aim_predict(model, al, dev, batch=64, forces=True)
    Dfr["e_abs_pred"] = np.asarray(e, float) + offset(al)
    Dfr["e_rel_pred"] = Dfr["e_abs_pred"]
    Fref = [np.asarray(a.arrays["REF_forces"], float) for a in al]
    out = dict(n=len(al), predict_s=round(time.time() - t0, 1),
               all=ft1_block(F, Dfr, Fref, Fp, np.ones(len(al), bool)))
    for s in SOURCES:
        out[s] = ft1_block(F, Dfr, Fref, Fp, np.asarray(src) == s)
    return out


def cmd_ft1src():
    import pandas as pd
    import torch
    import finetune as F
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, offset, minfo = load_aim(dev)
    res = dict(run=RUN, model=minfo, device=torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
               definitions=dict(ts_near=F.DECISIONS["FT-ts-near"], rel_e=F.DECISIONS["FT-relE"],
                                force_mae="mean |F_pred - F_ref| over Cartesian components, kcal/mol/A",
                                abs_e_mae="mean |E_pred - E_ref| per structure (absolute label-level energy), kcal/mol"),
               ft1_targets=F.FT1_TARGET, sets={})
    for nm in ("valid", "test"):
        al, D = load_set(nm)
        res["sets"][nm] = eval_set(F, model, offset, al, D["src"].to_numpy(), dev)
        b = res["sets"][nm]
        print(f"[ft1] {nm}: all TS-near F {b['all'].get('force_mae_kcal_A')} rel {b['all'].get('rel_e')}", flush=True)
        del al, D
    H = pd.read_csv(D0_HOLDOUT)
    hold = {int(x) for x in H["d0_id"]}
    al0 = [a for a in M.read_extxyz(GEN_D0G1)
           if int(a.info.get("d0_id", -1)) in hold and str(a.info.get("kind")) in ("d0_ts_near", "d0_ts_imag",
                                                                                     "d0_ts_label")]
    for a in al0:
        a.info["total_charge"] = int(a.info.get("charge", 0) or 0)
    if al0:
        res["sets"]["d0_holdout_ts"] = eval_set(F, model, offset, al0, np.array(["S4"] * len(al0)), dev,
                                                rxn_override=[(int(a.info["d0_id"]),) for a in al0])
        res["sets"]["d0_holdout_ts"]["note"] = ("gen_D0G1 TS-near rows of the D0 holdout reactions (d0_holdout.csv, eval "
                                                "+ sibling roles; never in any FT split, FT-d0hold); relative energies "
                                                "grouped by d0_id")
        res["sets"]["d0_holdout_ts"]["n_reactions"] = len({int(a.info["d0_id"]) for a in al0})
    else:
        res["sets"]["d0_holdout_ts"] = dict(n=0)
    res["note_S4"] = "fold0 valid / test contain no D0 rows (D0 = train only, DECISIONS FT-d0); S4 is measured on " \
                     "the D0 holdout TS rows instead"
    _write_part("ft1", res)
    return 0


# ================================================================ C-1 (ii) rerun (GPU)
def cmd_c1opt():
    import torch
    import finetune as F
    import mlts_bridge as B
    import tests_c1 as TC
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tj = _rj(TRAIN_JSON)
    if tj and tj.get("status") == "done" and FINAL_MODEL.is_file():
        dst, minfo = FINAL_MODEL, dict(which="final export", model=str(FINAL_MODEL))
    else:
        ck, ckinfo = snapshot_ckpt()
        dst = AUD / "models" / f"{RUN}_best.jpt"
        minfo = dict(which="best checkpoint exported by finetune.export_aimnet into $MLT/audit/models", model=str(dst),
                     checkpoint=ckinfo)
        if not dst.is_file():
            R = F.parse_run(RUN)
            meta = M.read_json(SPLIT / "meta.json")
            sae = {int(z): float(v) for z, v in M.read_json(RUN_DIR / "sae.json")["sae_eV"].items()}
            tr = M.read_extxyz(SPLIT / "valid.xyz", index=":8")
            minfo["export"] = F.export_aimnet(F.CANDIDATES["AIMNet2"], ck, dst, sae, R, "AIMNet2", meta, dev, tr)
    spec = B.resolve_spec(str(dst))
    picks = TC.pick()
    h = B.start_server(spec)
    t0 = time.time()
    try:
        d, _cache = TC.t_optimizer(spec, h.socket, picks)
        try:
            d["server"] = {k: v for k, v in (h.stats() or {}).items() if k != "spec"}
        except Exception:                                                      # noqa: BLE001
            pass
    finally:
        h.shutdown()
    zero = (((_rj(C1_JSON) or {}).get("models") or {}).get("AIMNet2") or {}).get("optimizer") or {}
    zrows = [dict(rxn_id=r.get("rxn_id"), pass_production=r.get("pass_production"), dform_max_A=r.get("dform_max_A"),
                  orca_converged=(r.get("orca") or {}).get("converged"),
                  sella_converged=(r.get("sella") or {}).get("converged")) for r in zero.get("rows", [])]
    _write_part("c1", dict(model=minfo, spec={k: v for k, v in spec.items() if k != "card"}, picks=picks,
                           optimizer=d, wall_s=time.time() - t0, device=B.device_info(),
                           zero_shot_reference=dict(file=str(C1_JSON), passed=zero.get("pass"),
                                                    pass_production=zero.get("pass_production"), rows=zrows),
                           note="tests_c1.t_optimizer called in-process (results/C1_tests.json is not modified): ORCA "
                                "`ExtOpt OptTS` (production: D1 block, NormalOpt) vs Sella (order 1, MLIP Hessian, fmax "
                                "1e-3 eV/A) from the 5 C1_pick label TSs; a miss is re-run tight; pass = all 5 within "
                                "0.005 A"))
    print(f"[c1] pass {d.get('pass')} production {d.get('pass_production')}", flush=True)
    return 0


# ================================================================ report
def optimizer_info():
    try:
        import mlts_bridge as B
        import ts_geom as TG
        return dict(engine="ORCA 6.1.1 `! ExtOpt OptTS` via mlts_bridge.optimize_ts (MLTS server on the job's GPU + "
                           "mlts_client as ProgExt); ts_geom.py run = FT-2 / X",
                    geom_block=dict(B.D1_OPTTS), start_hessian="MLIP Hessian written as .hess -> %geom InHess Read "
                    "(AIMNet2: " + str(B.AIMNet2Dense.hessian_method) + ")",
                    recalc=str(B.RECALC_DEFAULT) + " (new MLIP Hessian every 30 cycles = a new segment)",
                    convergence=f"ts_geom TIGHT_RUN = {TG.TIGHT_RUN!r} (TightOpt, decision TSG-tight)",
                    n_imag=f"MLIP Hessian modes below -{B.IMAG_TOL_CM} cm-1", sella="not used in FT-2; only the C-1 (ii) "
                    "reference optimiser (tests_c1.sella_ts)")
    except Exception as e:                                                     # noqa: BLE001
        return dict(error=f"{type(e).__name__}: {e}")


def cmd_report():
    parts = {n: _rj(_part(n)) for n in ("levels", "consist", "d3", "leak", "ft1", "c1")}
    errs = {p.stem: _rj(p) for p in AUD.glob("part_error_*.json")} if AUD.is_dir() else {}
    L, Cn = parts["levels"] or {}, parts["consist"] or {}
    verdict = {}
    for s in SOURCES:
        lv = (L.get("by_source") or {}).get(s, {})
        cs = (Cn.get("sources") or {}).get(s, {})
        same = lv.get("level_same")
        fm = cs.get("force_mae_kcal_A")
        ok = bool(same) and fm is not None and fm <= F_TOL and cs.get("n_ok") == cs.get("n_drawn")
        why = []
        if not same:
            why.append("level differs / not shown")
        if fm is None or fm > F_TOL:
            why.append(f"force MAE {fm} > {F_TOL}")
        if cs.get("n_ok") != cs.get("n_drawn"):
            why.append(f"recompute {cs.get('n_ok')}/{cs.get('n_drawn')}")
        verdict[s] = dict(desc=SRC_DESC[s], level_same=same, level_differs=lv.get("level_differs"),
                          convergence_differs=lv.get("convergence_differs"), n_recomputed=cs.get("n_ok"),
                          n_drawn=cs.get("n_drawn"), force_mae_kcal_A=fm, force_max_kcal_A=cs.get("force_max_kcal_A"),
                          rel_pair_mae_kcal=(cs.get("rel_pair_kcal") or {}).get("mae"),
                          rel_pair_max_kcal=(cs.get("rel_pair_kcal") or {}).get("max"),
                          abs_dE_max_eh=(cs.get("abs_dE_eh") or {}).get("max"),
                          abs_dE_median_eh=(cs.get("abs_dE_eh") or {}).get("median"),
                          decision="keep" if ok else "EXCLUDE from training (record in DEVIATIONS)", reasons=why)
    lk = parts["leak"] or {}
    d3 = parts["d3"] or {}
    zero = (((_rj(C1_JSON) or {}).get("models") or {}).get("AIMNet2") or {})
    J = dict(title="DEPLOY-V2 (MLTS) pre-unpause audit", code=CODE, written=M.now(),
             request="user 2026-11-05: (1) data-source consistency, (2) leakage by identity, (3) D3 consistency, extra "
                     "FT-1 by source + FT-2 optimizer",
             missing_parts=[n for n, v in parts.items() if v is None], part_errors=errs,
             check1=dict(rule=f"exclude a source whose level differs or whose recomputed force MAE > {F_TOL} kcal/mol/A",
                         verdict=verdict, levels=L, recompute=Cn),
             check2=dict(summary=dict(reaction_pair_overlap=(lk.get("reaction_pair_overlap") or {}).get("n_structures"),
                                      reaction_pair_overlap_isomeric=(lk.get("reaction_pair_overlap") or {}).get(
                                          "n_structures_isomeric"),
                                      by_set=(lk.get("reaction_pair_overlap") or {}).get("by_set"),
                                      by_role=(lk.get("reaction_pair_overlap") or {}).get("by_role"),
                                      by_role_definition=lk.get("by_role_definition"),
                                      molecule_overlap=(lk.get("molecule_overlap") or {}).get("n_structures"),
                                      product_overlap=(lk.get("product_overlap") or {}).get("n_structures"),
                                      leak_remove=(lk.get("reaction_pair_overlap") or {}).get("leak_remove")),
                         detail=lk),
             check3=dict(summary_label_ts=d3.get("summary_label_ts"), summary_all=d3.get("summary_all"),
                         precision=d3.get("precision"), parameters=d3.get("parameters"), detail=d3),
             extra=dict(ft1_by_source=parts["ft1"], optimizer=optimizer_info(),
                        c1_zero_shot=dict(passes=zero.get("passes"), pass_C1=zero.get("pass_C1")),
                        c1_rerun_finetuned=parts["c1"]))
    M.write_json(OUT, J)
    print(f"[report] {OUT}; missing {J['missing_parts']}; verdict "
          f"{ {s: v['decision'] for s, v in verdict.items()} }", flush=True)
    return 0


# ================================================================ main
def main(argv=None):
    ap = argparse.ArgumentParser(description="DEPLOY-V2 (MLTS) pre-unpause audit (see module docstring)")
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("sample")
    p.add_argument("--force", action="store_true")
    for n in ("consist", "levels", "d3", "ft1src", "c1opt", "report"):
        sp.add_parser(n)
    ncpu = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    for n in ("orca", "leak"):
        p = sp.add_parser(n)
        p.add_argument("--workers", type=int, default=ncpu)
    a = ap.parse_args(argv)
    fn = dict(sample=lambda: cmd_sample(a.force), orca=lambda: cmd_orca(a.workers), consist=cmd_consist,
              levels=cmd_levels, d3=cmd_d3, leak=lambda: cmd_leak(a.workers), ft1src=cmd_ft1src, c1opt=cmd_c1opt,
              report=cmd_report)[a.cmd]
    try:
        return fn() or 0
    except SystemExit:
        raise
    except Exception as e:                                                     # noqa: BLE001
        tb = traceback.format_exc()
        print(tb, flush=True)
        try:
            _write_part(f"error_{a.cmd}", dict(error=f"{type(e).__name__}: {e}", trace=tb[-4000:]))
        except Exception:                                                      # noqa: BLE001
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())
