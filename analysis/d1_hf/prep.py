#!/usr/bin/env python3
"""D1-HF Phase X (docs/specs/D1HF.md section 3) and the S / P selections (sections 5, 6). Run through sbatch only.

  python prep.py run [N]     # X-1..X-3 for every target: the 5 sp/*.inp read-only from the label archives (the same
                             #   sources as d1_pair/d1_extract_jobs.py) -> $HF/inputs/<T>/orig/, checks -> check.json,
                             #   HF inputs (spec 2) -> $HF/inputs/<T>/hf/. A job with check.json is skipped (idempotent).
  python prep.py report      # X-4: results/X_manifest.csv, results/X_report.json (exit 3 = STOP, spec 12 X)
  python prep.py smoke       # S selection -> results/S_select.csv; S-1 inputs $HF/inputs/<T>/orig1/ (orig, %pal 1)
  python prep.py pilot       # P selection -> results/P_sample.csv; G1 EDA inputs $HF/inputs/<T>/g1/eda.inp
  python prep.py g1frag      # G1 fragment inputs (additional instruction 2): $HF/inputs/<T>/g1/frag*.inp
  python prep.py main        # M lists (spec 7): results/M_batches/{dft,g1}_bNN.csv (<= 1,000 each), all G1 inputs

Sources (never modified): d1_pair/d1_jobs/<T>/source.txt names the bundle (nurion:<bundle>.tar = NURION_TARS/<bundle>.tar,
member jobs/<T>/archive.tgz) or the UBAI folder (ubai:UBAI_JOBS/<T>/archive.tgz); inside archive.tgz the members
sp/{eda,frag1_dist,frag2_dist,frag1_rel,frag2_rel}.inp (exact names; audit_view/ copies are ignored).
HF inputs: in the '!' line and the %eda FRAG1 / FRAG2 strings the tokens B3LYP -> HF and def2-TZVP -> 6-31G*, and the
eda.inp '%pal nprocs N end' -> 'nprocs 1'; nothing else changes (checked line by line).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tarfile
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hf_common as H  # noqa: E402

DP = H.DP
INP = {c: f"sp/{c}.inp" for c in H.CALCS}
EDA_BANG = ["B3LYP", "D3BJ", "def2-TZVP", "SMD(water)", "NoSym", "EDA", "TightSCF"]
FRAG_STR = ["B3LYP", "D3BJ", "def2-TZVP", "SMD(water)", "NoSym", "TightSCF"]
SA_BANG = ["B3LYP", "D3BJ", "def2-TZVP", "CPCM(water)", "NoSym", "TightSCF"]          # standalone fragment inputs
EDA_BLOCKS, SA_BLOCKS = ["eda", "maxcore", "pal"], ["cpcm", "maxcore"]
P_SEED = 20261021
S2_ELEMENTS = ("F", "Si", "S", "Cl", "Br")
COORD_TOL = 1e-6


def norm(n):
    return n[2:] if n.startswith("./") else n


def sp_from_tgz(fobj):
    """{calc: bytes} of the 5 sp/*.inp members of a (streamed) archive.tgz; stops once all 5 are read."""
    got = {}
    want = {v: k for k, v in INP.items()}
    with tarfile.open(fileobj=fobj, mode="r|*") as t:
        for m in t:
            n = norm(m.name)
            if m.isfile() and n in want:
                got[want[n]] = t.extractfile(m).read()
                if len(got) == len(INP):
                    break
    return got


def hf_text(text, kind):
    """spec 2 substitution (whole tokens only) -> (new text, changed line numbers)."""
    out, changed = [], []
    for i, ln in enumerate(text.splitlines(keepends=True)):
        s, new = ln.strip(), ln
        if s.startswith("!") or (kind == "eda" and s.split()[:1] in (["FRAG1"], ["FRAG2"])):
            for a, b in H.SUB:
                new = re.sub(rf"(?<![\w\-]){re.escape(a)}(?![\w\-*()])", b, new)
        elif kind == "eda" and s.lower().startswith("%pal"):
            new = re.sub(r"nprocs\s+\d+", "nprocs 1", ln)
        if new != ln:
            changed.append(i)
        out.append(new)
    return "".join(out), changed


def pal1_text(text):
    """S-1: the original input with only '%pal nprocs N end' -> 'nprocs 1' (serial runner, as for HF)."""
    return re.sub(r"(?m)^(\s*%pal\s+nprocs\s+)\d+", r"\g<1>1", text)


def same_atoms(a, b):
    """same length, same elements in order, coordinates within COORD_TOL (A)."""
    return len(a) == len(b) and all(x[0] == y[0] for x, y in zip(a, b)) and \
        (not a or float(np.abs(np.array([x[1:] for x in a]) - np.array([y[1:] for y in b])).max()) <= COORD_TOL)


def check_job(T, files, lab):
    """X-2 checks + X-3 HF inputs -> (record, {relpath: bytes})."""
    rec = dict(job_id=T, rxn_id=int(lab["rxn_id"]), dipole=lab["dipole"], skeleton=lab["skeleton"], reasons=[],
               stop=[], keywords={})
    miss = [c for c in H.CALCS if c not in files]
    if miss:
        rec["reasons"].append("missing:" + ",".join(miss))
        return rec, {}
    d1 = DP.D1_JOBS / T
    if files["eda"] != (d1 / "sp" / "eda.inp").read_bytes():
        rec["reasons"].append("eda_inp_differs_from_d1_jobs")
    try:
        P = {c: H.parse_inp(files[c].decode()) for c in H.CALCS}
    except (ValueError, UnicodeDecodeError, IndexError) as e:
        rec["reasons"].append(f"parse:{e}")
        return rec, {}
    e = P["eda"]
    tag = {k: [(a[0], a[2], a[3], a[4]) for a in e["atoms"] if a[1] == k] for k in (1, 2)}
    rec.update(n_atoms=len(e["atoms"]), n_f1=len(tag[1]), n_f2=len(tag[2]),
               elements=";".join(sorted({a[0] for a in e["atoms"]})))
    if any(a[1] not in (1, 2) for a in e["atoms"]):
        rec["reasons"].append("eda_untagged_atom")
    for k in (1, 2):
        if not same_atoms([(a[0], a[2], a[3], a[4]) for a in P[f"frag{k}_dist"]["atoms"]], tag[k]):
            rec["reasons"].append(f"frag{k}_dist_atoms_differ_from_eda")
    refs = {1: H.read_xyz(d1 / "ref_dipole.xyz"), 2: H.read_xyz(d1 / "ref_dipolarophile.xyz")}
    for k in (1, 2):
        if not same_atoms([(a[0], a[2], a[3], a[4]) for a in P[f"frag{k}_rel"]["atoms"]], refs[k]):
            rec["reasons"].append(f"frag{k}_rel_atoms_differ_from_ref")
    lj = json.loads((d1 / "label.json").read_text())
    if not (lj.get("role1") == "dipole" and lj.get("role2") == "dipolarophile"
            and sorted(a[0] for a in tag[1]) == sorted(a[0] for a in refs[1])
            and sorted(a[0] for a in tag[2]) == sorted(a[0] for a in refs[2])):
        rec["stop"].append("frag1_not_dipole")
    try:
        q = {k: (int(e["frag"][f"FRAG{k}_C"]), int(e["frag"][f"FRAG{k}_M"])) for k in (1, 2)}
    except (KeyError, ValueError):
        q = None
        rec["reasons"].append("eda_frag_charge_mult_missing")
    if q:
        for c in H.CALCS[:4]:
            if (P[c]["charge"], P[c]["mult"]) != q[int(c[4])]:
                rec["reasons"].append(f"{c}_charge_mult_differ_from_eda")
        if e["charge"] != q[1][0] + q[2][0]:
            rec["reasons"].append("eda_total_charge")
    for c in H.CALCS:
        if not H.smd_ok(P[c], "eda" if c == "eda" else "frag"):
            rec["reasons"].append(f"{c}_no_smd")
            rec["stop"].append("input_without_smd")
    # keyword inventory and the expected sets (anything else = something that might need changing -> STOP)
    for c in H.CALCS:
        p = P[c]
        kw = dict(bang=p["bang"], blocks=sorted(p["blocks"]), maxcore=p["blocks"].get("maxcore", [""])[0])
        if c == "eda":
            kw.update(FRAG1=p["frag"].get("FRAG1"), FRAG2=p["frag"].get("FRAG2"), pal=p["pal"])
            ok = (p["bang"] == EDA_BANG and sorted(p["blocks"]) == EDA_BLOCKS
                  and all(p["frag"].get(f, "").split() == FRAG_STR for f in ("FRAG1", "FRAG2"))
                  and set(p["frag"]) == {"FRAG1", "FRAG2", "FRAG1_C", "FRAG1_M", "FRAG2_C", "FRAG2_M"})
        else:
            kw.update(cpcm=p["blocks"].get("cpcm"))
            ok = p["bang"] == SA_BANG and sorted(p["blocks"]) == SA_BLOCKS
        ok &= kw["maxcore"].split() == ["%maxcore", "3500"]
        rec["keywords"][c] = kw
        if not ok:
            rec["stop"].append(f"unexpected_keywords:{c}")
    # X-3: HF inputs; the diff must be exactly the substituted lines (+ %pal)
    out = {f"orig/{c}.inp": files[c] for c in H.CALCS}
    for c in H.CALCS:
        kind = "eda" if c == "eda" else "frag"
        txt = files[c].decode()
        new, changed = hf_text(txt, kind)
        o, n = txt.splitlines(), new.splitlines()
        allowed = {i for i, ln in enumerate(o) if ln.strip().startswith("!")}
        if kind == "eda":
            allowed |= {i for i, ln in enumerate(o) if ln.strip().split()[:1] in (["FRAG1"], ["FRAG2"])
                        or ln.strip().lower().startswith("%pal")}
        pn = H.parse_inp(new)
        want_bang = [dict(H.SUB).get(t, t) for t in P[c]["bang"]]
        bad = (len(o) != len(n) or not set(changed) <= allowed or pn["bang"] != want_bang
               or pn["atoms"] != P[c]["atoms"] or (pn["charge"], pn["mult"]) != (P[c]["charge"], P[c]["mult"])
               or sum(ln.count("HF") for ln in n) - sum(ln.count("HF") for ln in o) != (3 if kind == "eda" else 1)
               or sum(ln.count("6-31G*") for ln in n) != (3 if kind == "eda" else 1)
               or any(w in new for w in ("B3LYP", "def2-TZVP")))
        if kind == "eda":
            bad |= (pn["pal"] != 1 or any(pn["frag"].get(f, "").split() != [dict(H.SUB).get(t, t) for t in FRAG_STR]
                                          for f in ("FRAG1", "FRAG2"))
                    or {k: v for k, v in pn["frag"].items() if k not in ("FRAG1", "FRAG2")}
                    != {k: v for k, v in P[c]["frag"].items() if k not in ("FRAG1", "FRAG2")})
        if bad:
            rec["reasons"].append(f"hf_diff_check:{c}")
        out[f"hf/{c}.inp"] = new.encode()
        rec[f"sha_orig_{c}"], rec[f"sha_hf_{c}"] = H.sha256_bytes(files[c]), H.sha256_bytes(new.encode())
    return rec, out


def finish(T, rec, out):
    """$HF/inputs/<T>/{orig,hf}/*.inp + check.json, written into <T>.tmp/ and renamed (atomic per job)."""
    final, tmp = H.INPUTS / T, H.INPUTS / f"{T}.tmp"
    H.assert_path(tmp)
    if tmp.exists():
        shutil.rmtree(tmp)
    for rel, data in out.items():
        (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp / rel).write_bytes(data)
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "check.json").write_text(json.dumps(rec, indent=1))
    if final.exists():
        shutil.rmtree(final)
    os.replace(tmp, final)


def done(T):
    return (H.INPUTS / T / "check.json").is_file()


def from_bundle(args):
    tar, jobs, labs = args
    want = {j for j in jobs if not done(j)}
    got = {j: None for j in want}
    if want:
        with tarfile.open(tar, mode="r:") as Tt:
            for m in Tt:
                p = norm(m.name).split("/")
                if m.isfile() and len(p) == 3 and p[0] == "jobs" and p[1] in got and p[2] == "archive.tgz":
                    got[p[1]] = sp_from_tgz(Tt.extractfile(m))
    for j, files in got.items():
        rec, out = check_job(j, files or {}, labs[j])
        finish(j, rec, out)
    return str(tar), len(jobs), len(want)


def from_ubai(args):
    jid, path, lab = args
    if done(jid):
        return jid, "skip"
    with open(path, "rb") as f:
        files = sp_from_tgz(f)
    rec, out = check_job(jid, files, lab)
    finish(jid, rec, out)
    return jid, "ok"


def cmd_run(nproc):
    tg = H.targets()
    labs = {d["job_id"]: d for d in tg}
    H.INPUTS.mkdir(parents=True, exist_ok=True)
    groups, ubai, bad = {}, [], []
    for T in labs:
        src = (DP.D1_JOBS / T / "source.txt").read_text().strip()
        kind, _, where = src.partition(":")
        if kind == "nurion":
            groups.setdefault(DP.NURION_TARS / where, []).append(T)
        elif kind == "ubai" and where.endswith("archive.tgz"):
            ubai.append((T, where, labs[T]))
        else:
            bad.append((T, src))
    if bad:
        H.stop(f"unknown sources: {bad[:5]}")
    print(f"[X] {len(labs)} targets: {sum(map(len, groups.values()))} in {len(groups)} bundles, {len(ubai)} UBAI",
          flush=True)
    with Pool(nproc) as pool:
        for tar, n, m in pool.imap_unordered(from_bundle, [(t, j, {x: labs[x] for x in j}) for t, j in groups.items()]):
            print(f"  {Path(tar).name}: {n} jobs, {m} processed now", flush=True)
        for jid, s in pool.imap_unordered(from_ubai, ubai):
            pass
    print(f"[X] inputs ready: {sum(done(T) for T in labs)} / {len(labs)}", flush=True)
    return 0


def cmd_report():
    tg = H.targets()
    rows, kw = [], {c: Counter() for c in H.CALCS}
    for d in tg:
        f = H.INPUTS / d["job_id"] / "check.json"
        r = json.loads(f.read_text()) if f.is_file() else dict(job_id=d["job_id"], rxn_id=d["rxn_id"],
                                                                 reasons=["not_prepared"], stop=[], keywords={})
        for c, k in r.get("keywords", {}).items():
            kw[c][json.dumps(k, sort_keys=True)] += 1
        rows.append(r)
    M = pd.DataFrame([{k: v for k, v in r.items() if k != "keywords"} for r in rows])
    M["x_fail"] = M.reasons.map(len) > 0
    M["reasons"] = M.reasons.map(";".join)
    M["stop"] = M.stop.map(";".join)
    H.write_atomic(H.RES / "X_manifest.csv", lambda t: M.to_csv(t, index=False))
    n, nf = len(M), int(M.x_fail.sum())
    stops = sorted({s for r in rows for s in r["stop"]})
    if nf / n > 0.005:
        stops.append(f"x_fail {nf}/{n} > 0.5 %")
    rep = dict(n_labels=9496, n_dpp_excluded=9496 - n, n_targets=n, n_x_fail=nf, x_fail_rate=nf / n,
               x_fail_reasons=Counter(x for r in rows for x in r["reasons"]).most_common(),
               stop=stops, keywords={c: [dict(n=v, **json.loads(k)) for k, v in kw[c].most_common()] for c in H.CALCS},
               elements=Counter(e for s in M.elements.dropna() for e in s.split(";")).most_common(),
               n_atoms=dict(min=int(M.n_atoms.min()), median=float(M.n_atoms.median()), max=int(M.n_atoms.max())),
               substitution=[list(s) for s in H.SUB], pal="eda.inp %pal nprocs N end -> nprocs 1",
               smd_rule="eda.inp: SMD(water) on the ! line and in both FRAG strings, no %cpcm; standalone fragment "
                        "inputs: CPCM(water) + %cpcm smd true / smdsolvent \"water\" (label convention) or SMD(water)",
               manifest_sha256=H.sha256(H.RES / "X_manifest.csv"))
    H.write_json(H.RES / "X_report.json", rep)
    print(json.dumps({k: v for k, v in rep.items() if k != "keywords"}, indent=1, default=str), flush=True)
    return 3 if stops else 0


def ok_manifest():
    M = pd.read_csv(H.RES / "X_manifest.csv", keep_default_na=False)
    return M[~M.x_fail.astype(bool)].copy()


def cmd_smoke():
    """S-1: the 2 smallest targets (n_atoms, then job_id); S-2: per element F, Si, S, Cl, Br the smallest target that
    contains it, + the largest target (distinct reactions)."""
    M = ok_manifest().sort_values(["n_atoms", "job_id"])
    rows = [dict(group="S-1", job_id=r.job_id, why="smallest") for r in M.head(2).itertuples()]
    taken = {r["job_id"] for r in rows}
    for el in S2_ELEMENTS:
        q = M[M.elements.str.split(";").map(lambda s, el=el: el in s) & ~M.job_id.isin(taken)]
        if not len(q):
            H.stop(f"S-2: no target contains {el}")
        rows.append(dict(group="S-2", job_id=q.iloc[0].job_id, why=f"smallest with {el}"))
        taken.add(q.iloc[0].job_id)
    q = M[~M.job_id.isin(taken)].sort_values(["n_atoms", "job_id"], ascending=[False, True])
    rows.append(dict(group="S-2", job_id=q.iloc[0].job_id, why="largest"))
    S = pd.DataFrame(rows).merge(M[["job_id", "rxn_id", "dipole", "skeleton", "n_atoms", "elements"]], on="job_id")
    for T in S[S.group == "S-1"].job_id:
        for c in H.CALCS:
            H.write_atomic(H.INPUTS / T / "orig1" / f"{c}.inp", pal1_text((H.INPUTS / T / "orig" / f"{c}.inp").read_text()))
    H.write_atomic(H.RES / "S_select.csv", lambda t: S.to_csv(t, index=False))
    for g, f in (("S-1", "S1_list.csv"), ("S-2", "S2_list.csv")):
        H.write_atomic(H.RES / f, lambda t, g=g: S[S.group == g][["job_id"]].to_csv(t, index=False))
    print(S.to_string(index=False), flush=True)
    return 0


def g1_input(T, rxn_id):
    """G1 EDA input = the HF eda.inp with the coordinates of g1_final/<rxn_id>/ts.xyz (same tags, charges, fragments);
    None if no G1 structure or its atom order / elements differ from the DFT TS."""
    g = H.G1_FINAL / str(rxn_id)
    if not ((g / ".done").is_file() and (g / "ts.xyz").is_file()):
        return None, "no_g1"
    G = H.read_xyz(g / "ts.xyz")
    hf = (H.INPUTS / T / "hf" / "eda.inp").read_text()
    p = H.parse_inp(hf)
    ts = H.read_xyz(DP.D1_JOBS / T / "ts.xyz")
    if [a[0] for a in G] != [a[0] for a in p["atoms"]] or [a[0] for a in G] != [a[0] for a in ts]:
        return None, "g1_order_or_elements_differ"
    lines, k = [], 0
    for ln in hf.splitlines(keepends=True):
        m = H.XYZ_LINE.match(ln)
        if m and k < len(G):
            el, x, y, z = G[k]
            lines.append(f"  {f'{el}({m.group(2)})':<6}{x:16.8f}{y:14.8f}{z:14.8f}\n")
            k += 1
        else:
            lines.append(ln)
    if k != len(G):
        return None, "g1_atom_count"
    return "".join(lines), "ok"


def cmd_pilot():
    """P sample (spec 6): candidates = X-ok targets - S-2 reactions; strata (dipole, skeleton) in sorted order, 2 per
    stratum (1 if the stratum has 1 row) with default_rng(20261021) without replacement, then filled from the remaining
    candidates with the same generator to exactly 100."""
    M = ok_manifest().sort_values("job_id")
    S = pd.read_csv(H.RES / "S_select.csv")
    C = M[~M.job_id.isin(S[S.group == "S-2"].job_id)]
    rng = np.random.default_rng(P_SEED)
    pick = []
    for (dp, sk), g in C.groupby(["dipole", "skeleton"], sort=True):
        ids = g.job_id.tolist()
        pick += [(ids[i], "stratum") for i in rng.choice(len(ids), size=min(2, len(ids)), replace=False)]
    rest = sorted(set(C.job_id) - {p for p, _ in pick})
    need = 100 - len(pick)
    if need > 0:
        pick += [(rest[i], "fill") for i in rng.choice(len(rest), size=need, replace=False)]
    elif need < 0:
        H.stop(f"{len(pick)} stratum picks > 100")
    P = pd.DataFrame(pick, columns=["job_id", "pick"]).merge(M[["job_id", "rxn_id", "dipole", "skeleton", "n_atoms",
                                                               "elements"]], on="job_id")
    st = []
    for r in P.itertuples():
        txt, s = g1_input(r.job_id, r.rxn_id)
        if txt is not None:
            H.write_atomic(H.INPUTS / r.job_id / "g1" / "eda.inp", txt)
        st.append(s)
    P["g1"] = st
    H.write_atomic(H.RES / "P_sample.csv", lambda t: P.to_csv(t, index=False))
    H.write_atomic(H.RES / "P_list.csv", lambda t: P[["job_id"]].to_csv(t, index=False))
    H.write_atomic(H.RES / "P_g1_list.csv", lambda t: P[P.g1 == "ok"][["job_id"]].to_csv(t, index=False))
    print(f"[P] {len(P)} sampled ({(P.pick == 'stratum').sum()} stratum, {(P.pick == 'fill').sum()} fill, "
          f"{C.groupby(['dipole', 'skeleton']).ngroups} strata); G1: {P.g1.value_counts().to_dict()}", flush=True)
    return 0


def batches(df, size=1000):
    """size-balanced batches of <= size reactions: sorted by n_atoms, dealt round-robin."""
    df = df.sort_values(["n_atoms", "job_id"]).reset_index(drop=True)
    nb = int(np.ceil(len(df) / size)) if len(df) else 0
    return [df.iloc[k::nb] for k in range(nb)]


def cmd_main():
    """Phase M (spec 7) lists: dft = every X-ok target without runs/dft/<T>/.done; g1 (user decision at P, taken as the
    recommended option): the G1 EDA inputs of every X-ok target with a G1 structure (g1_input, same rules as the pilot)
    and every such target without runs/g1/<T>/.done. Batches of <= 1,000 -> results/M_batches/{dft,g1}_bNN.csv."""
    M = ok_manifest()
    g1s = []
    for r in M.itertuples():
        f = H.INPUTS / r.job_id / "g1" / "eda.inp"
        if f.is_file():
            g1s.append("ok"); continue
        txt, s = g1_input(r.job_id, r.rxn_id)
        if txt is not None:
            H.write_atomic(f, txt)
        g1s.append(s)
    M["g1"] = g1s
    H.write_atomic(H.RES / "M_g1_inputs.csv", lambda t: M[["job_id", "rxn_id", "n_atoms", "g1"]].to_csv(t, index=False))
    out = H.RES / "M_batches"
    summ = {}
    for geom, cand in (("dft", M), ("g1", M[M.g1 == "ok"])):
        todo = cand[[not (H.RUNS / geom / T / ".done").is_file() for T in cand.job_id]]
        for k, b in enumerate(batches(todo)):
            H.write_atomic(out / f"{geom}_b{k:02d}.csv", lambda t, b=b: b[["job_id", "n_atoms"]].to_csv(t, index=False))
        summ[geom] = dict(candidates=len(cand), todo=len(todo), n_batches=int(np.ceil(len(todo) / 1000)))
    summ["g1_status"] = M.g1.value_counts().to_dict()
    H.write_json(H.RES / "M_plan.json", summ)
    print(json.dumps(summ, indent=1), flush=True)
    return 0


def replace_coords(text, atoms):
    """the input with its coordinate lines replaced, in order, by atoms [(element, x, y, z)] (tags kept)."""
    lines, k = [], 0
    for ln in text.splitlines(keepends=True):
        m = H.XYZ_LINE.match(ln)
        if m and k < len(atoms):
            el, x, y, z = atoms[k]
            lab = f"{el}({m.group(2)})" if m.group(2) else el
            lines.append(f"  {lab:<6}{x:16.8f}{y:14.8f}{z:14.8f}\n")
            k += 1
        else:
            lines.append(ln)
    if k != len(atoms):
        raise ValueError("atom count")
    return "".join(lines)


def g1_frag_inputs(T, rxn_id):
    """D1HF additional instruction 2 (2026-10-22): G1 fragment single points. frag<k>_dist = the G1 TS atoms of
    fragment k (eda.inp (k) tags, TS index order); frag<k>_rel = g1_final/<rxn_id>/rel<k>.xyz (the G1 reactants the
    D1PAIR features use; rel1 = dipole, rel2 = dipolarophile); method, charge, multiplicity and solvent = the DFT-TS HF
    inputs, only the coordinates change. Element order must equal the DFT-TS HF input -> ({name: text} or None, status)."""
    g = H.G1_FINAL / str(rxn_id)
    if not ((g / ".done").is_file() and all((g / f).is_file() for f in ("ts.xyz", "rel1.xyz", "rel2.xyz"))):
        return None, "no_g1"
    G = H.read_xyz(g / "ts.xyz")
    e = H.parse_inp((H.INPUTS / T / "hf" / "eda.inp").read_text())
    if [a[0] for a in G] != [a[0] for a in e["atoms"]]:
        return None, "g1_ts_order"
    tags = [a[1] for a in e["atoms"]]
    out = {}
    for k in (1, 2):
        for kind, atoms in (("dist", [G[i] for i in range(len(G)) if tags[i] == k]), ("rel", H.read_xyz(g / f"rel{k}.xyz"))):
            src = (H.INPUTS / T / "hf" / f"frag{k}_{kind}.inp").read_text()
            if [a[0] for a in H.parse_inp(src)["atoms"]] != [a[0] for a in atoms]:
                return None, f"frag{k}_{kind}_order"
            out[f"frag{k}_{kind}.inp"] = replace_coords(src, atoms)
    return out, "ok"


def cmd_g1frag():
    """G1 fragment inputs for every target with a G1 EDA input -> $HF/inputs/<T>/g1/frag*.inp, results/M_g1frag_inputs.csv."""
    M = ok_manifest().sort_values("job_id")
    rows = []
    for r in M.itertuples():
        if not (H.INPUTS / r.job_id / "g1" / "eda.inp").is_file():
            continue
        out, s = g1_frag_inputs(r.job_id, r.rxn_id)
        if out:
            for n, txt in out.items():
                H.write_atomic(H.INPUTS / r.job_id / "g1" / n, txt)
        rows.append(dict(job_id=r.job_id, rxn_id=r.rxn_id, status=s))
    D = pd.DataFrame(rows)
    H.write_atomic(H.RES / "M_g1frag_inputs.csv", lambda t: D.to_csv(t, index=False))
    print(f"[g1frag] {len(D)} G1 reactions: {D.status.value_counts().to_dict()}", flush=True)
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["run"]:
        sys.exit(cmd_run(int(a[1]) if len(a) > 1 else int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))))
    cmds = {"report": cmd_report, "smoke": cmd_smoke, "pilot": cmd_pilot, "main": cmd_main, "g1frag": cmd_g1frag}
    if a[:1] and a[0] in cmds:
        sys.exit(cmds[a[0]]())
    raise SystemExit(__doc__)
