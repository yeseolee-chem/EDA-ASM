#!/usr/bin/env python3
"""d1_phase_p.py — D1PAIR Phase P (§2): rows and pairs from the appendix scripts run unchanged, their gates, the
G1 target list, the P-4 stereo diagnostic and results/INPUTS.md.

  python d1_phase_p.py run       # appendix A -> B -> E (python -I, unchanged) + D self-test; gates P-1..P-3;
                                 # results/P_report.json, results/G_targets.csv (exit 3 = STOP)
  python d1_phase_p.py stereo    # gate P-4 on every reference and anchor pair -> results/P4_stereo.csv, P4_report.json
  python d1_phase_p.py inputs    # results/INPUTS.md: input sha256, <D1_JOBS>, the 8,659 job folders complete

Appendix outputs: rowattr.pkl (+ .all.pkl) under $D1PAIR_SCRATCH/P, pairs_d1.csv and pairs_d1_anchor.csv under
results/; an existing output is not rebuilt (the gates are always re-checked).
P-4 rule (§2-4): the new 5-ring of the DFT TS = dipole termini of the two formed_pairs_ts bonds (the (1)-tagged atom of
each pair), the dipole centre (the (1) heavy atom bonded to both termini) and the two dipolarophile atoms. For every
ring atom: substituent branches = connected components of the TS bond graph (fragmenter.bond_matrix, inter-fragment
contacts removed) minus the ring atoms, entered from that atom. The heaviest branch (summed atomic mass; a tie for
the heaviest, or a single branch, = no stereo at that atom -> not compared) is placed on one side of the ring mean
plane: sign of (x_first_atom - ring centroid) . n, n = Newell normal of the ring in its canonical order (centre,
terminus, dipolarophile atom, dipolarophile atom, terminus; the pair order is tried both ways for b). Ring atoms of
a and b correspond when their elements agree and their branch multisets (Weisfeiler-Lehman hash of each branch,
first atom marked) agree, except at one atom: the X site, compared by the branch that differs (X or Ph / H first
atom; skipped when that branch is not unique). A pair mismatches when no admissible orientation agrees in every sign.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import d1pair_common as C  # noqa: E402

P_DIR = C.SCRATCH / "P"
STDOUT = C.RES / "P_appendix_stdout.txt"


def sh(path):
    return C.sha256(path) if Path(path).is_file() else None


def run_appendix(name, args, log):
    t0 = time.time()
    r = subprocess.run([sys.executable, "-I", str(HERE / name), *map(str, args)], capture_output=True, text=True)
    err = ("#### stderr\n" + r.stderr) if r.stderr.strip() else ""
    log.append(f"$ python -I {name} {' '.join(map(str, args))}   (exit {r.returncode}, {time.time() - t0:.0f} s)\n"
               f"{r.stdout}{err}")
    if r.returncode != 0:
        raise SystemExit(f"STOP: {name} exit {r.returncode}: {r.stderr[-2000:]}")
    return r.stdout


def check_appendix_sha():
    bad = {k: sh(HERE / k) for k, v in C.APPENDIX_SHA.items() if sh(HERE / k) != v}
    if bad:
        print(f"STOP: appendix sha256 differs: {bad}", file=sys.stderr)
        sys.exit(3)


def hyg(df, d1="d1_kcal", d2="d2_kcal"):
    return C.hygiene_mask(df[d1], df[d2])


# ================================================================ run
def cmd_run():
    check_appendix_sha()
    C.load_d1_labels()
    P_DIR.mkdir(parents=True, exist_ok=True)
    log = []
    if not (C.ROWATTR.is_file() and Path(str(C.ROWATTR) + ".all.pkl").is_file()):
        run_appendix("build_rowattr.py", [C.D1_LABELS, C.SIGMA, C.ROWATTR], log)
    if not C.PAIRS.is_file():
        run_appendix("d1_pairs.py", [C.ROWATTR, C.SIGMA, C.PAIRS], log)
    if not C.ANCHOR.is_file():
        run_appendix("d1_anchor_pairs.py", [C.ROWATTR, C.SIGMA, C.PAIRS, C.ANCHOR], log)
    dtxt = run_appendix("hammett_eval.py", [C.PAIRS], log)
    if log:
        old = STDOUT.read_text() if STDOUT.is_file() else ""
        C.write_atomic(STDOUT, old + "\n".join(log))
    import importlib.metadata as md
    allr = pd.read_pickle(str(C.ROWATTR) + ".all.pkl")
    kept = pd.read_pickle(C.ROWATTR)
    P = pd.read_csv(C.PAIRS)
    A = pd.read_csv(C.ANCHOR)
    rep, fails = dict(versions={k: md.version(k) for k in ("pandas", "numpy", "rdkit")},
                      python=sys.version.split()[0]), []

    def gate(name, got, want):
        ok = got == want
        rep.setdefault("gates", {})[name] = dict(got=got, want=want, ok=bool(ok))
        if not ok:
            fails.append(name)

    # ---- P-1
    e1 = C.EXPECT_P1
    gate("P1_n_labels", len(allr), e1["n_labels"])
    gate("P1_excluded", len(allr) - len(kept), e1["excluded"])
    gate("P1_kept", len(kept), e1["kept"])
    gate("P1_pos", {k: int(v) for k, v in kept.pos.value_counts().items()}, e1["pos"])
    rep["P1_excluded_by"] = dict(dpp=int(allr.skeleton.isin(C.DPP).sum()),
                                 ref_excess_non_dpp=int((((allr.ex1 > 20) | (allr.ex2 > 20)) & ~allr.skeleton.isin(C.DPP)).sum()),
                                 flag_async=int(allr.fl_flag_async.sum()), flag_short_forming=int(allr.fl_flag_short_forming.sum()),
                                 flag_regio_ambiguous=int(allr.fl_flag_regio_ambiguous.sum()))
    # ---- P-2
    e2 = C.EXPECT_P2
    posb = dict(zip(kept.job_id, kept.pos))
    gate("P2_n_pairs", len(P), e2["n_pairs"])
    gate("P2_pos", {k: int(v) for k, v in P.pos.value_counts().items()}, e2["pos"])
    gate("P2_families", int(P.fam.nunique()), e2["families"])
    gate("P2_cores", int(P.core.nunique()), e2["cores"])
    refs = sorted(set(P.b_job))
    gate("P2_refs", len(refs), e2["refs"])
    gate("P2_refs_aryl", sum(posb[j] == "ref_aryl" for j in refs), e2["refs_aryl"])
    gate("P2_refs_H", sum(posb[j] == "ref_direct" for j in refs), e2["refs_H"])
    gate("P2_reactions", len(set(P.a_job) | set(P.b_job)), e2["reactions"])
    gate("P2_para_fam_12", int((P[(P.pos == "p") & P.sigma.notna()].groupby("fam").Xb.nunique() >= C.MIN_X_FAM - 1).sum()),
         e2["para_fam_12"])
    nf = P.groupby("b_job").fam.nunique()
    shared = nf[nf > 1].index
    gate("P2_shared_refs", int(len(shared)), e2["shared_refs"])
    gate("P2_shared_ref_pairs", int(P.b_job.isin(shared).sum()), e2["shared_ref_pairs"])
    gate("P2_ref_in_one_core", int((P.groupby("b_job").core.nunique() > 1).sum()), 0)
    lab = kept.set_index("job_id")
    okh = dict(zip(kept.job_id, hyg(kept)))
    ph = np.array([okh[a] and okh[b] for a, b in zip(P.a_job, P.b_job)])
    gate("P2_hygiene_dropped", int((~ph).sum()), e2["hygiene_dropped"])
    gate("P2_hygiene_dropped_refs", int(sum(not okh[j] for j in refs)), e2["hygiene_dropped_refs"])
    dsum = P[[f"dd_{t}" for t in C.CHANNELS9]].sum(axis=1) - P.dd_barrier
    rep["P2_identity_resid"] = dict(median_abs=float(dsum.abs().median()), max_abs=float(dsum.abs().max()))
    rep["P2_sha256"] = dict(file=sh(C.PAIRS), reference=e2["sha256"], identical=sh(C.PAIRS) == e2["sha256"],
                            list_sha256=C.list_sha(P[["pair_id", "a_job", "b_job"]].itertuples(index=False)))
    # ---- P-3
    e3 = C.EXPECT_P3
    gate("P3_n_pairs", len(A), e3["n_pairs"])
    gate("P3_pos", {k: int(v) for k, v in A.pos.value_counts().items()}, e3["pos"])
    gate("P3_families", int(A.fam.nunique()), e3["families"])
    arx = set(A.a_job) | set(A.b_job)
    gate("P3_reactions", len(arx), e3["reactions"])
    gate("P3_overlap_with_ref_pairs", len(arx & (set(P.a_job) | set(P.b_job))), 0)
    gate("P3_cores", int(A.core.nunique()), e3["cores"])
    gate("P3_cores_new", sorted(set(A.core) - set(P.core)), e3["cores_new"])
    ah = np.array([okh[a] and okh[b] for a, b in zip(A.a_job, A.b_job)])
    gate("P3_hygiene_dropped", int((~ah).sum()), e3["hygiene_dropped"])
    rep["P3_sha256"] = dict(file=sh(C.ANCHOR), reference=e3["sha256"], identical=sh(C.ANCHOR) == e3["sha256"],
                            list_sha256=C.list_sha(A[["pair_id", "a_job", "b_job"]].itertuples(index=False)))
    # ---- appendix D self-test (§5-2; repeated in Phase S)
    m = re.search(r"para families evaluated: (\d+)", dtxt)
    bent = eval(re.search(r"DFT bend calls per target: (\{.*\})", dtxt).group(1))     # noqa: S307 (own stdout)
    vv = eval(re.search(r"DFT V calls per target: (\{.*\})", dtxt).group(1))          # noqa: S307
    hid = int(re.search(r"barrier straight & >= 1 channel bent: (\d+)", dtxt).group(1))
    ed = C.EXPECT_D
    gate("D_n_fam", int(m.group(1)), ed["n_fam"])
    gate("D_bent", bent, ed["bent"])
    gate("D_V", vv, ed["V"])
    gate("D_hidden_bend", hid, ed["hidden_bend"])
    # ---- G1 targets (§7-1): references -> anchors -> rest (job_id)
    anchors = sorted(set(A.b_job))
    rest = sorted((set(P.a_job) | arx) - set(refs) - set(anchors))
    order = [(j, "reference") for j in refs] + [(j, "anchor") for j in anchors] + [(j, "other") for j in rest]
    gate("G_n_targets", len(order), C.N_G_TARGETS)
    # §7-1 says "anchor 75": appendix E picks one anchor per (aryl | direct, F) group, and a fam string with both
    # aryl and direct rows has two anchors -> more anchor rows than anchor families (reported, not a gate)
    na = A.groupby("fam").b_job.nunique()
    rep["G_anchor_rows"] = dict(n_anchor_rows=len(anchors), n_anchor_families=int(A.fam.nunique()),
                                fams_with_2_anchors=int((na > 1).sum()),
                                fams_with_2_anchors_pos=A[A.fam.isin(na[na > 1].index)].groupby("fam").pos.agg(
                                    lambda s: "/".join(sorted(set(s)))).value_counts().to_dict())
    tg = pd.DataFrame([dict(order=k, job_id=j, rxn_id=int(lab.loc[j, "rxn_id"]), role=r,
                            core=f"{lab.loc[j, 'dipole']}/{lab.loc[j, 'skeleton']}",
                            in_ref_pairs=j in set(P.a_job) | set(P.b_job), in_anchor_pairs=j in arx)
                       for k, (j, r) in enumerate(order)])
    C.write_atomic(C.G_TARGETS, lambda t: tg.to_csv(t, index=False))
    rep.update(fails=fails, stop=bool(fails), outputs=dict(rowattr=sh(C.ROWATTR), rowattr_all=sh(str(C.ROWATTR) + ".all.pkl"),
                                                         pairs=sh(C.PAIRS), anchor=sh(C.ANCHOR), g_targets=sh(C.G_TARGETS)))
    C.write_json_safe(C.RES / "P_report.json", rep)
    print(json.dumps(rep, indent=1, default=str))
    return 3 if fails else 0


# ================================================================ P-4 stereo
MASS = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "Si": 28.086, "S": 32.06, "Cl": 35.45,
        "Br": 79.904, "I": 126.904}
TAG_RE = re.compile(r"^\s*([A-Za-z]{1,2})\((1|2)\)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$")


def parse_eda_inp(path):
    """(syms, tags 1/2, xyz) of the '* xyz' block of an ORCA EDA input (atom order = TS order)."""
    syms, tags, xyz, on = [], [], [], False
    for l in Path(path).read_text().splitlines():
        s = l.strip()
        if s.startswith("* xyz"):
            on = True
            continue
        if on and s == "*":
            break
        if on:
            m = TAG_RE.match(l)
            if not m:
                raise ValueError(f"eda.inp atom line not understood: {l!r}")
            syms.append(m.group(1)); tags.append(int(m.group(2)))
            xyz.append([float(m.group(k)) for k in (3, 4, 5)])
    return syms, np.array(tags), np.array(xyz)


def ring_info(job, label):
    """ring + branch description of one DFT TS; raises ValueError when the ring cannot be built."""
    import fragmenter as fr
    import networkx as nx
    d = C.D1_JOBS / job
    syms, xyz = fr.read_xyz(d / "ts.xyz")
    s2, tags, x2 = parse_eda_inp(d / "sp" / "eda.inp")
    if s2 != list(syms) or np.abs(x2 - xyz).max() > C.G1_TOL_COORD_A:
        raise ValueError("eda.inp_vs_ts.xyz")
    # fragmenter.bond_matrix with its own radii (Cordero 2008) + Si 1.11 A (fragmenter.COV has no Si; P-4 only)
    rad = np.array([{**fr.COV, "Si": 1.11}[s] for s in syms])
    adj = fr.dist_matrix(xyz) < fr.FACTOR * (rad[:, None] + rad[None, :])
    np.fill_diagonal(adj, False)
    adj = adj & (tags[:, None] == tags[None, :])               # no inter-fragment bond at the TS
    pairs = [tuple(map(int, p.split("-"))) for p in str(label["formed_pairs_ts"]).split()]
    if len(pairs) != 2:
        raise ValueError("not_two_formed_pairs")
    term, dph = [], []
    for i, j in pairs:
        if tags[i] == tags[j]:
            raise ValueError("formed_pair_within_one_fragment")
        term.append(i if tags[i] == 1 else j); dph.append(j if tags[i] == 1 else i)
    heavy1 = [k for k in range(len(syms)) if tags[k] == 1 and syms[k] != "H" and k not in term]
    cen = [k for k in heavy1 if adj[k, term[0]] and adj[k, term[1]]]
    if len(cen) != 1:
        raise ValueError(f"dipole_centre_{len(cen)}")
    c = cen[0]
    ring = [c, term[0], dph[0], dph[1], term[1]]                # orientation '+': pair 0 first
    G = nx.Graph()
    G.add_nodes_from(range(len(syms)))
    G.add_edges_from((int(i), int(j)) for i, j in zip(*np.where(np.triu(adj))))
    rest = G.subgraph([k for k in range(len(syms)) if k not in ring]).copy()
    comp = {}
    for cc in nx.connected_components(rest):
        for k in cc:
            comp[k] = frozenset(cc)
    branches = []
    for r in ring:
        br = []
        for n in G.neighbors(r):
            if n in ring:
                continue
            cc = comp[n]
            sub = G.subgraph(cc).copy()
            for k in sub.nodes:
                sub.nodes[k]["lab"] = syms[k] + ("*" if k == n else "") + ("r" if any(adj[k, q] for q in ring) else "")
            h = nx.weisfeiler_lehman_graph_hash(sub, node_attr="lab", iterations=4)
            br.append(dict(first=int(n), hash=h, mass=float(sum(MASS[syms[k]] for k in cc))))
        branches.append(br)
    return dict(syms=syms, xyz=xyz, ring=ring, branches=branches, elems=[syms[k] for k in ring])


def oriented(info, flip):
    """ring order, branches and signed plane distances for orientation flip (pair order swapped)."""
    idx = [0, 4, 3, 2, 1] if flip else [0, 1, 2, 3, 4]
    ring = [info["ring"][i] for i in idx]
    br = [info["branches"][i] for i in idx]
    P = info["xyz"][ring]
    cen = P.mean(0)
    n = np.zeros(3)
    for k in range(5):
        n += np.cross(P[k] - cen, P[(k + 1) % 5] - cen)
    n /= np.linalg.norm(n)
    out = []
    for k, bl in enumerate(br):
        out.append([dict(b, d=float(np.dot(info["xyz"][b["first"]] - cen, n)),
                         d_local=float(np.dot(info["xyz"][b["first"]] - P[k], n))) for b in bl])
    return dict(elems=[info["syms"][r] for r in ring], br=out)


def heaviest(bl):
    if len(bl) < 2:
        return None
    s = sorted(bl, key=lambda b: -b["mass"])
    return None if abs(s[0]["mass"] - s[1]["mass"]) < 1e-6 else s[0]


def compare(ia, ib):
    """compare a (orientation '+') with b in both orientations (pair order swapped).
    literal rule (§2-4 text): every compared atom must have the same sign; the orientation with fewest mismatches.
    stereo rule (rule v2, reported next to it): only atoms with >= 2 exocyclic branches in both a and b (possible
    stereocentres; an sp2 ring atom with one substituent has no configuration), and the sign pattern compared up to one
    global mirror flip (enantiomeric TSs have identical energies and EDA terms in the achiral SMD solvent)."""
    oa = oriented(ia, False)
    res = []
    for flip in (False, True):
        ob = oriented(ib, flip)
        if ob["elems"] != oa["elems"]:
            continue
        diff = [k for k in range(5) if Counter(b["hash"] for b in oa["br"][k]) != Counter(b["hash"] for b in ob["br"][k])]
        comp = []
        for k in range(5):
            if k in diff:
                if len(diff) != 1:
                    continue
                ca, cb = Counter(b["hash"] for b in oa["br"][k]), Counter(b["hash"] for b in ob["br"][k])
                xa, xb = list((ca - cb).elements()), list((cb - ca).elements())
                if len(xa) != 1 or len(xb) != 1 or ca[xa[0]] != 1 or cb[xb[0]] != 1:
                    continue
                ba = next(b for b in oa["br"][k] if b["hash"] == xa[0])
                bb = next(b for b in ob["br"][k] if b["hash"] == xb[0])
                kind = "x_site"
            else:
                ba, bb = heaviest(oa["br"][k]), heaviest(ob["br"][k])
                if ba is None or bb is None or ba["hash"] != bb["hash"]:
                    if len(oa["br"][k]) == 1 and len(ob["br"][k]) == 1 and oa["br"][k][0]["hash"] == ob["br"][k][0]["hash"]:
                        ba, bb = oa["br"][k][0], ob["br"][k][0]       # single substituent: literal rule only
                    else:
                        continue
                kind = "heaviest"
            comp.append(dict(pos=k, kind=kind, d_a=round(ba["d"], 3), d_b=round(bb["d"], 3),
                             same=bool(np.sign(ba["d"]) == np.sign(bb["d"])),
                             same_local=bool(np.sign(ba["d_local"]) == np.sign(bb["d_local"])),
                             stereo=bool(len(oa["br"][k]) >= 2 and len(ob["br"][k]) >= 2)))
        lit = [c for c in comp if c["kind"] == "x_site" or len(oa["br"][c["pos"]]) >= 2]   # §2-4: heaviest of >= 2
        st = [c for c in comp if c["stereo"]]
        sa = np.sign([c["d_a"] for c in st]); sb = np.sign([c["d_b"] for c in st])
        v2 = ("no_relative_info" if len(st) < 2 else "match" if (sa == sb).all() else
              "mirror_match" if (sa == -sb).all() else "mismatch")
        res.append(dict(flip=flip, n_diff_pos=len(diff), x_site=diff[0] if len(diff) == 1 else None,
                        n_compared=len(lit), n_mismatch=sum(not c["same"] for c in lit),
                        n_mismatch_local=sum(not c["same_local"] for c in lit),
                        min_abs_d=float(min(min(abs(c["d_a"]), abs(c["d_b"])) for c in lit)) if lit else None,
                        n_stereo=len(st), v2=v2, comp=comp))
    if not res:
        return dict(status="ring_elements_differ", v2="ring_elements_differ")
    adm = [r for r in res if r["n_diff_pos"] <= 1] or res
    best = min(adm, key=lambda r: (r["n_mismatch"], r["n_diff_pos"], -r["n_compared"]))
    rank = {"match": 0, "mirror_match": 1, "no_relative_info": 2, "mismatch": 3}
    v2 = min((r["v2"] for r in adm), key=rank.get)
    return dict(status="ok" if best["n_diff_pos"] <= 1 else "x_site_ambiguous", **{k: v for k, v in best.items() if k != "v2"},
                v2=v2, n_orientations=len(adm))


def cmd_stereo():
    lab = {d["job_id"]: d for d in C.load_d1_labels()}
    st = pd.read_csv(C.D1_STATUS, dtype=str, keep_default_na=False).set_index("job_id")
    P = pd.read_csv(C.PAIRS).assign(ref_kind="reference")
    A = pd.read_csv(C.ANCHOR)
    allp = pd.concat([P[["pair_id", "ref_kind", "a_job", "b_job", "core", "fam", "pos"]],
                      A[["pair_id", "ref_kind", "a_job", "b_job", "core", "fam", "pos"]]], ignore_index=True)
    cache, rows = {}, []
    for r in allp.itertuples():
        for j in (r.a_job, r.b_job):
            if j not in cache:
                try:
                    cache[j] = ring_info(j, lab[j])
                except Exception as e:                              # noqa: BLE001
                    cache[j] = f"{type(e).__name__}:{e}"
        ia, ib = cache[r.a_job], cache[r.b_job]
        base = dict(pair_id=r.pair_id, ref_kind=r.ref_kind, a_job=r.a_job, b_job=r.b_job, core=r.core, fam=r.fam,
                    pos=r.pos, stereo_a=st.loc[r.a_job, "stereo_index"], stereo_b=st.loc[r.b_job, "stereo_index"])
        if isinstance(ia, str) or isinstance(ib, str):
            rows.append(dict(base, status="ring_fail", v2="ring_fail", detail=ia if isinstance(ia, str) else ib))
            continue
        c = compare(ia, ib)
        comp = c.pop("comp", [])
        rows.append(dict(base, **c, detail=json.dumps(comp)))
    df = pd.DataFrame(rows)
    df["stereo_index_differs"] = df.stereo_a != df.stereo_b
    C.write_atomic(C.RES / "P4_stereo.csv", lambda t: df.to_csv(t, index=False))
    mis = df[df.n_mismatch.fillna(0) > 0]
    rep = dict(n_pairs=len(df), by_kind=df.ref_kind.value_counts().to_dict(),
               status=df.status.value_counts().to_dict(),
               stereo_index_differs_ref=int(df[(df.ref_kind == "reference")].stereo_index_differs.sum()),
               stereo_index_differs_ref_expected=C.EXPECT_P4_STEREO_DIFF,
               stereo_index_differs_cores=sorted(df[(df.ref_kind == "reference") & df.stereo_index_differs].core.unique()),
               n_compared_positions=int(df.n_compared.fillna(0).sum()),
               pairs_with_0_compared=int((df.n_compared.fillna(0) == 0).sum()),
               n_mismatch_pairs=int(len(mis)), n_mismatch_pairs_ref=int((mis.ref_kind == "reference").sum()),
               n_mismatch_pairs_anchor=int((mis.ref_kind == "anchor").sum()),
               n_mismatch_pairs_stereo_differs=int(mis.stereo_index_differs.sum()),
               n_mismatch_local_rule_pairs=int((df.n_mismatch_local.fillna(0) > 0).sum()),
               mismatch_list=mis[["pair_id", "ref_kind", "a_job", "b_job", "fam", "stereo_a", "stereo_b", "n_mismatch",
                                  "min_abs_d"]].to_dict("records")[:200],
               ring_fail=df[df.status == "ring_fail"][["pair_id", "detail"]].to_dict("records")[:50],
               v2_rule="possible stereocentres only (>= 2 exocyclic branches in a and b), sign pattern up to a mirror flip",
               v2={k: {kk: int(vv) for kk, vv in g.v2.value_counts().items()} for k, g in df.groupby("ref_kind")},
               v2_mismatch_stereo_differs=int(((df.v2 == "mismatch") & df.stereo_index_differs).sum()),
               v2_mismatch_list=df[df.v2 == "mismatch"][["pair_id", "ref_kind", "a_job", "b_job", "fam", "stereo_a",
                                                         "stereo_b", "n_stereo", "detail"]].to_dict("records")[:300],
               stop_literal=bool(len(mis)), stop_v2=bool((df.v2 == "mismatch").sum()),
               user_decision=C.P4_DECISION)
    drop = df[df.v2 == "mismatch"][["pair_id", "ref_kind", "a_job", "b_job", "fam", "core", "pos"]].assign(
        reason="p4_v2_stereo_mismatch")
    C.write_atomic(C.P4_DROP, lambda t: drop.to_csv(t, index=False))
    rep.update(n_drop=len(drop), drop_sha256=C.sha256(C.P4_DROP), stop=False)
    C.write_json_safe(C.RES / "P4_report.json", rep)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("mismatch_list", "v2_mismatch_list")}, indent=1,
                     default=str))
    print("v2 mismatches (first 30):", json.dumps(rep["v2_mismatch_list"][:30], indent=0, default=str))
    print("mismatches (first 30):", json.dumps(rep["mismatch_list"][:30], indent=0, default=str))
    return 3 if rep["stop"] else 0


# ================================================================ INPUTS.md
def cmd_inputs():
    tg = pd.read_csv(C.G_TARGETS)
    miss = [(j, [f for f in C.JOB_FILES if not (C.D1_JOBS / j / f).is_file()]) for j in tg.job_id]
    miss = [m for m in miss if m[1]]
    chk = json.loads((C.RES / "EXTRACT_CHECK.json").read_text())
    pre6 = json.loads(C.D0_PREREG6.read_text())["inputs_sha256"]
    rows = [("D1 labels", C.D1_LABELS, C.D1_LABELS_SHA), ("D1 status", C.D1_STATUS, C.D1_STATUS_SHA),
            ("D0 labels (repo root)", C.D0_LABELS, pre6[str(C.D0_PREREG6.parents[3] / "labels_all.json")]),
            ("D0 features X182_all", R6X(), pre6[str(R6X())]),
            ("D0 targets S_targets", C.R6.S_TARGETS, pre6[str(C.R6.S_TARGETS)]),
            ("D0 rows S_splits_appendixA", C.R6.S_SPLITS_A,
             pre6[str(C.D0_PREREG6.parent / "S_splits_appendixA.csv")])]
    lines = ["# D1PAIR rev 9 — inputs (Phase 0, §1)", "",
             f"written {time.strftime('%Y-%m-%d %H:%M')} by `d1_phase_p.py inputs` (compute node)", "",
             "| input | path | sha256 | expected | ok |", "|---|---|---|---|---|"]
    bad = []
    for name, p, want in rows:
        got = sh(p)
        lines.append(f"| {name} | `{p}` | `{got}` | `{want}` | {'yes' if got == want else '**NO**'} |")
        if got != want:
            bad.append(name)
    n0 = len(json.loads(C.D0_LABELS.read_text()))
    lines += ["", f"- D0 `labels_all.json`: {n0} rows (the D1 interim file `a2df136b…` had 9,459 rows; this one is "
              f"{'not ' if not sh(C.D0_LABELS).startswith(C.D1_INTERIM_SHA_PREFIX) else ''}that file).",
              f"- D1 `labels_d1.json`: {C.N_D1_LABELS} rows, all status ok (checked by `d1pair_common.load_d1_labels`).",
              "", "## <D1_JOBS>", "",
              "No unpacked job folder with `T00001/ts.xyz` and `T00001/sp/eda.inp` exists on the server (the Nurion job "
              "folders are in the 23 bundle tars of `D1_build/nurion/`, `sp/eda.inp` inside each `jobs/<T>/archive.tgz`; "
              "58 rows remain as UBAI folders in `d1_scratch/jobs/`). Phase 0 STOP -> user decision 2026-10-19: extract "
              "read-only into scratch (`d1_extract_jobs.py`).", "",
              f"- `<D1_JOBS>` = `{C.D1_JOBS}`",
              f"- extracted ok rows: {chk['n_complete']} / {chk['n_ok_rows']}; sources {chk['sources']}",
              f"- ts.xyz / ref_dipole.xyz / ref_dipolarophile.xyz byte-identical to `{C.D1_GEOM}/<T>/`: "
              f"{chk['n_geom_identical']} / {chk['n_ok_rows']}",
              f"- label.json job_id / rxn_id = labels_d1.json: {chk['n_label_ids_match']}; file fields equal: "
              f"{chk['n_label_files_match']}",
              f"- `MANIFEST.tsv` sha256 `{chk['manifest_sha256']}`",
              f"- the {len(tg)} G1 / pair job folders with all of {', '.join(C.JOB_FILES)}: {len(tg) - len(miss)} "
              f"(missing: {miss[:10]})"]
    C.write_atomic(C.RES / "INPUTS.md", "\n".join(lines) + "\n")
    print("\n".join(lines))
    return 3 if bad or miss or chk["n_bad"] else 0


def R6X():
    return C.R6.X182_ALL


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    fn = {"run": cmd_run, "stereo": cmd_stereo, "inputs": cmd_inputs}.get(cmd)
    if fn is None:
        raise SystemExit(__doc__)
    sys.exit(fn())
