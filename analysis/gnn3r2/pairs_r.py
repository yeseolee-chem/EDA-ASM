#!/usr/bin/env python3
"""GNN3R Phase B (docs/specs/GNN3R.md 4): the pairs = the fixed data set.

    python analysis/gnn3r/pairs_r.py build     # 4-0 .. 4-6, gates B-0 .. B-7 -> results/ (exit 3 = STOP)
    python analysis/gnn3r/pairs_r.py rerun     # B-R: the same build in a new process -> $GR/B/rerun/, sha256 of the
                                               # three outputs compared; results/B_label_report.json only if all pass
    job: analysis/d1_pair/jobs/py.slurm (PYENV=reactot) 'pairs_r.py build ::: pairs_r.py rerun'

Families (4-1) come from d1_pairs.build / d1_anchor_pairs.build (imported, unchanged) on $GR/A/rowattr_r.pkl; only
their membership is used for anchor families. P-4 (4-3) = d1_phase_p.ring_info / compare (imported, unchanged); the
per-reaction ring descriptions are cached in $GR/B/p4_cache.pkl. Labels (4-4) = labels_d1_cr2.json, dd_t = y_t(a) - y_t(b).

B_row_status.csv status: the row is `in_dataset` when it is an end of a final pair; for every other row the first of
not_usable -> dedup_dropped -> ref_duplicated_family -> H_ambiguous -> no_family -> singleton -> p4_dropped that holds:
  not_usable             U_usable.csv usable False
  dedup_dropped          not in any family; an aryl / direct row duplicated within F + Xb + pos (keep=False)
  ref_duplicated_family  not in any family; its F has >= 2 ref_aryl rows (aryl row or the ref_aryl row itself)
  H_ambiguous            not in any family; a direct row with >= 2 H candidates after the d1_pairs narrowing
  no_family              not in any family otherwise
  singleton              in a family but in no candidate pair (< 2 usable members, or a usable reference without a
                         usable member)
  p4_dropped             every candidate pair of the row was dropped by P-4 (v2 mismatch)
(in_dataset is tested first so that a direct row that d1_pairs pairs although its F + Xb + pos repeats - d1_pairs does
not deduplicate direct rows - is counted in the data set; their number is reported as in_dataset_also_dup_key.)
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gr_common as C  # noqa: E402

C.import_d1pair()
import d1pair_common as DP  # noqa: E402
import d1_anchor_pairs  # noqa: E402
import d1_pairs  # noqa: E402
import d1_phase_p as PP  # noqa: E402

ARYL = ("p", "m", "o")
SUBST = ("p", "m", "o", "direct")
IDENTITY_TOL = 0.2                                       # B-5, kcal/mol
CR2_TOL = 1e-9                                           # B-4
COLUMNS = (["pair_id", "ref_kind", "a_job", "a_rxn", "b_job", "b_rxn", "core", "fam"] + C.F
           + ["pos", "Xb", "sigma", "anchor_Xb", "anchor_pos", "p4_v2"] + [f"dd_{t}" for t in DP.D1_TCOL])
STATUS_ORDER = ("not_usable", "dedup_dropped", "ref_duplicated_family", "H_ambiguous", "no_family", "singleton",
                "p4_dropped", "in_dataset")


def norm(r: pd.DataFrame) -> pd.DataFrame:
    """F columns as d1_pairs.build sees them (fillna('') then str)."""
    r = r.copy()
    for c in C.F:
        r[c] = r[c].fillna("").astype(str)
    return r


def fkey(row) -> tuple:
    return tuple(row[c] for c in C.F)


def d7_key(row, job: str) -> tuple:
    """D-7 anchor order: Xb rank (Me, CO2Me, tBu, CHO, Ts, NMe2, rest), position p < m < o < direct, Xb, job_id."""
    return (C.XB_RANK.get(str(row.Xb), C.XB_RANK_OTHER), C.POS_RANK[row.pos], str(row.Xb), job)


# ================================================================ 4-0 reproduction of the D1PAIR pairs
def check_b0(sg) -> dict:
    kept = pd.read_pickle(C.ROWATTR_KEPT)
    p0, a0 = d1_pairs.build(kept, sg), d1_anchor_pairs.build(kept, sg)
    out = {}
    for name, built, path in (("pairs_d1", p0, C.PAIRS_D1), ("pairs_d1_anchor", a0, C.ANCHOR_D1)):
        want = pd.read_csv(path, usecols=["pair_id", "a_job", "b_job"], dtype=str)
        got = built[["pair_id", "a_job", "b_job"]].astype(str)
        same = want.values.tolist() == got.values.tolist()
        out[name] = dict(n_built=int(len(got)), n_file=int(len(want)), identical_in_order=same)
    out["ok"] = all(v["identical_in_order"] for v in out.values())
    return out


# ================================================================ 4-1 families, 4-2 usable filter and anchor b
def families(r: pd.DataFrame, sg):
    pref, a0 = d1_pairs.build(r, sg), d1_anchor_pairs.build(r, sg)
    fams = []
    for b, g in pref.groupby("b_job", sort=True):
        fams.append(dict(kind="reference", ref=b, members=sorted(g.a_job), src=f"reference:{b}"))
    for fam, g in a0.groupby("fam", sort=True):
        fams.append(dict(kind="anchor", ref=None, members=sorted(set(g.a_job) | set(g.b_job)), src=f"A0:{fam}"))
    return pref, a0, fams


def candidate_pairs(fams, rows: pd.DataFrame, usable: set) -> tuple[pd.DataFrame, list]:
    """4-2: reference pairs (usable reference) and anchor pairs (anchor families, and reference families whose
    reference is not usable regrouped by F); returns the pairs and one record per family / group."""
    out, groups_log = [], []
    for f in fams:
        mem = [m for m in f["members"] if m in usable]
        if f["kind"] == "reference" and f["ref"] in usable:
            out += [dict(a_job=a, b_job=f["ref"], ref_kind="reference", src=f["src"], anchor_from="") for a in mem]
            groups_log.append(dict(src=f["src"], kind="reference", n_members=len(f["members"]), n_usable=len(mem),
                                   b=f["ref"], n_pairs=len(mem)))
            continue
        if f["kind"] == "reference":
            by_f = {}
            for m in mem:
                by_f.setdefault(fkey(rows.loc[m]), []).append(m)
            groups, how = [by_f[k] for k in sorted(by_f)], "reference_not_usable"
        else:
            groups, how = [mem], "anchor_family"
        for g in groups:
            b = min(g, key=lambda j: d7_key(rows.loc[j], j)) if len(g) >= 2 else None
            groups_log.append(dict(src=f["src"], kind=how, n_members=len(f["members"]), n_usable=len(g), b=b,
                                   n_pairs=max(len(g) - 1, 0)))
            if b is None:
                continue
            out += [dict(a_job=a, b_job=b, ref_kind="anchor", src=f["src"], anchor_from=how) for a in sorted(g) if a != b]
    return pd.DataFrame(out), groups_log


# ================================================================ 4-3 P-4
def p4(cand: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    jobs = sorted(set(cand.a_job) | set(cand.b_job))
    cache = pickle.loads(C.P4_CACHE.read_bytes()) if C.P4_CACHE.is_file() else {}
    todo = [j for j in jobs if j not in cache]
    if todo:
        lab = {d["job_id"]: d for d in DP.load_d1_labels()}            # sha256 / 9,496-row checks (exit 3)
        for j in todo:
            try:
                cache[j] = PP.ring_info(j, lab[j])
            except Exception as e:                                       # noqa: BLE001  (= d1_phase_p.cmd_stereo)
                cache[j] = f"{type(e).__name__}:{e}"
        C.write_atomic(C.P4_CACHE, pickle.dumps(cache, protocol=4))
    rows = []
    for a, b in zip(cand.a_job, cand.b_job):
        ia, ib = cache[a], cache[b]
        if isinstance(ia, str) or isinstance(ib, str):
            rows.append(dict(p4_status="ring_fail", p4_v2="ring_fail", p4_detail=ia if isinstance(ia, str) else ib))
            continue
        c = PP.compare(ia, ib)
        rows.append(dict(p4_status=c["status"], p4_v2=c["v2"], p4_detail=""))
    return pd.DataFrame(rows, index=cand.index), dict(n_jobs=len(jobs), n_computed_now=len(todo),
                                                      n_ring_fail_jobs=sum(isinstance(cache[j], str) for j in jobs))


# ================================================================ 4-4 / 4-5 labels, ids, columns
def assemble(P: pd.DataFrame, rows: pd.DataFrame, sg, cr2: dict) -> pd.DataFrame:
    A, B = rows.loc[P.a_job], rows.loc[P.b_job]
    anchor = (P.ref_kind == "anchor").to_numpy()
    out = pd.DataFrame(dict(ref_kind=P.ref_kind.to_numpy(), a_job=P.a_job.to_numpy(),
                            a_rxn=A.rxn_id.astype(int).to_numpy(), b_job=P.b_job.to_numpy(),
                            b_rxn=B.rxn_id.astype(int).to_numpy()))
    for c in C.F:
        out[c] = A[c].to_numpy()
    out["core"] = out.dipole + "/" + out.skeleton
    out["fam"] = out[C.F].agg("/".join, axis=1) + np.where(anchor, "/anchor", "")
    out["pos"] = A.pos.astype(str).to_numpy()
    out["Xb"] = A.Xb.astype(str).to_numpy()
    out["sigma"] = np.where(out.pos == "m", out.Xb.map(sg.sigma_m), out.Xb.map(sg.sigma_p))
    out["anchor_Xb"] = np.where(anchor, B.Xb.astype(str).to_numpy(), None)
    out["anchor_pos"] = np.where(anchor, B.pos.astype(str).to_numpy(), None)
    out["p4_v2"] = P.p4_v2.to_numpy()
    for t, col in DP.D1_TCOL.items():
        out[f"dd_{t}"] = np.array([float(cr2[a][col]) - float(cr2[b][col]) for a, b in zip(out.a_job, out.b_job)],
                                  dtype=np.float64)
    key = [(0 if k == "reference" else 1, c, f, C.POS_RANK[p], x, a) for k, c, f, p, x, a in
           zip(out.ref_kind, out.core, out.fam, out.pos, out.Xb, out.a_job)]
    out = out.iloc[sorted(range(len(out)), key=key.__getitem__)].reset_index(drop=True)
    ids, n = [], {"reference": 0, "anchor": 0}
    for k in out.ref_kind:
        ids.append(f"{'GRP' if k == 'reference' else 'GRA'}{n[k]:05d}")
        n[k] += 1
    out.insert(0, "pair_id", ids)
    return out[COLUMNS]


# ================================================================ 4-6 reaction and row tables
def reaction_table(P: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    ends = pd.concat([P[["a_job", "core", "fam"]].rename(columns={"a_job": "job_id"}).assign(role="a"),
                      P[["b_job", "core", "fam"]].rename(columns={"b_job": "job_id"}).assign(role="b")])
    g = ends.groupby("job_id", sort=True)
    t = pd.DataFrame(dict(core=g.core.agg(lambda s: ";".join(sorted(set(s)))),
                          role=g.role.agg(lambda s: ";".join(sorted(set(s)))), n_pairs=g.size(),
                          fam=g.fam.agg(lambda s: ";".join(sorted(set(s))))))
    t.insert(0, "rxn_id", rows.loc[t.index, "rxn_id"].astype(int).to_numpy())
    return t.reset_index()[["job_id", "rxn_id", "core", "role", "n_pairs", "fam"]]


def h_candidates(rn: pd.DataFrame) -> dict:
    """number of H candidates of every direct row after the d1_pairs.build narrowing (orient, cfg, c3)"""
    H = rn[rn.pos == "ref_direct"]
    out = {}
    for _, x in rn[rn.pos == "direct"].iterrows():
        c = H[(H.dipole == x.dipole) & (H.skeleton == x.skeleton) & (H.face == x.face)]
        for col in ("orient", "cfg", "c3"):
            if len(c) > 1 and (c[col] != "").all():
                c = c[c[col] == x[col]]
        out[x.job_id] = len(c)
    return out


def row_status(rn, usable, in_family, anchor_family_rows, cand, P, R) -> tuple[pd.DataFrame, dict]:
    sub = rn[rn.pos.isin(SUBST)]
    dup = set(sub.loc[sub.duplicated(C.F + ["Xb", "pos"], keep=False), "job_id"])
    refn = rn[rn.pos == "ref_aryl"].groupby(C.F).size()
    refdup = {j for j, k, p in zip(rn.job_id, map(tuple, rn[C.F].to_numpy()), rn.pos)
              if p in ARYL + ("ref_aryl",) and refn.get(k, 0) >= 2}
    nh = h_candidates(rn)
    in_cand = set(cand.a_job) | set(cand.b_job)
    in_data = set(P.a_job) | set(P.b_job)
    data_fam = dict(zip(R.job_id, R.fam))
    out = []
    for x in rn.itertuples(index=False):
        j = x.job_id
        if j not in usable:
            s = "not_usable"
        elif j in in_data:
            s = "in_dataset"
        elif j not in in_family:
            s = ("dedup_dropped" if j in dup else "ref_duplicated_family" if j in refdup else
                 "H_ambiguous" if x.pos == "direct" and nh.get(j, 0) >= 2 else "no_family")
        elif j not in in_cand:
            s = "singleton"
        else:
            s = "p4_dropped"
        fam = data_fam.get(j) or ("/".join(getattr(x, c) for c in C.F) + ("/anchor" if j in anchor_family_rows else ""))
        out.append(dict(job_id=j, rxn_id=int(x.rxn_id), usable=j in usable, status=s, fam=fam))
    T = pd.DataFrame(out)
    extra = dict(in_dataset_also_dup_key=int(len(in_data & dup)), in_dataset_also_ref_dup=int(len(in_data & refdup)),
                 h_candidates_ge2_direct_rows=int(sum(v >= 2 for v in nh.values())))
    return T, extra


def diff_vs_est(P: pd.DataFrame, R: pd.DataFrame, S: pd.DataFrame) -> pd.DataFrame:
    er = pd.read_csv(C.REF_REACTIONS, dtype=str, keep_default_na=False)
    ep = pd.read_csv(C.REF_PAIRS, dtype=str, keep_default_na=False)
    group, basis = dict(zip(er.job_id, er.est_group)), dict(zip(er.job_id, er.family_basis))
    status, sfam = dict(zip(S.job_id, S.status)), dict(zip(S.job_id, S.fam))
    srv = set(R.job_id)
    rows = []
    for j in sorted(set(er.job_id) ^ srv):
        rows.append(dict(kind="reaction", id=j, in_est=j in group, in_server=j in srv, est_group=group.get(j, ""),
                         server_status=status.get(j, "not_target"), server_fam=sfam.get(j, ""),
                         family_basis=basis.get(j, ""),
                         reason=("server " + status.get(j, "not_target")) if j in group else "not in the estimate list"))
    ek = {(a, b): (k, g) for a, b, k, g in zip(ep.a_job, ep.b_job, ep.ref_kind, ep.est_group)}
    sk = {(a, b): (k, f) for a, b, k, f in zip(P.a_job, P.b_job, P.ref_kind, P.fam)}
    est_b = {a: (b, k) for (a, b), (k, _) in ek.items()}
    srv_b = {a: (b, k) for (a, b), (k, _) in sk.items()}
    est_bs, srv_bs = {b for _, b in ek}, {b for _, b in sk}
    for key in sorted(set(ek) | set(sk)):
        a, b = key
        ie, isv = key in ek, key in sk
        if ie and isv:
            if ek[key][0] == sk[key][0]:
                continue
            reason = f"ref_kind estimate {ek[key][0]} vs server {sk[key][0]}"
            kind = "pair_ref_kind"
        elif ie:
            kind = "pair"
            reason = (f"server pairs a with b={srv_b[a][0]} ({srv_b[a][1]})" if a in srv_b else
                      "a is a server anchor b" if a in srv_bs else f"a server {status.get(a, 'not_target')}")
        else:
            kind = "pair"
            reason = (f"estimate pairs a with b={est_b[a][0]} ({est_b[a][1]})" if a in est_b else
                      "a is an estimate anchor b" if a in est_bs else
                      "a not in the estimate list" if a not in group else "a in the estimate list without a pair")
        rows.append(dict(kind=kind, id=f"{a}>{b}", in_est=ie, in_server=isv,
                         est_group=ek[key][1] if ie else group.get(a, ""),
                         server_status=sk[key][0] if isv else status.get(a, "not_target"),
                         server_fam=sk[key][1] if isv else sfam.get(a, ""), family_basis=basis.get(a, ""), reason=reason))
    cols = ["kind", "id", "in_est", "in_server", "est_group", "server_status", "server_fam", "family_basis", "reason"]
    return pd.DataFrame(rows, columns=cols)


def reason_class(s: str) -> str:
    """coarse reason of a B_diff_vs_est row (for the summary counts)"""
    for k in ("server pairs a with", "estimate pairs a with", "ref_kind estimate"):
        if s.startswith(k):
            return k
    return s


# ================================================================ checks
def check_b2(P, rows, usable) -> dict:
    A, B = rows.loc[P.a_job], rows.loc[P.b_job]
    same_f = (A[C.F].to_numpy() == B[C.F].to_numpy()).all(1)
    same_dsf = (A[["dipole", "skeleton", "face"]].to_numpy() == B[["dipole", "skeleton", "face"]].to_numpy()).all(1)
    core_a = (A.dipole + "/" + A.skeleton).to_numpy()
    core_b = (B.dipole + "/" + B.skeleton).to_numpy()
    aryl_ref = ((P.ref_kind == "reference") & P.pos.isin(ARYL)).to_numpy()
    direct_ref = ((P.ref_kind == "reference") & (P.pos == "direct")).to_numpy()
    anchor = (P.ref_kind == "anchor").to_numpy()
    r = dict(ends_usable=bool(P.a_job.isin(usable).all() and P.b_job.isin(usable).all()),
             a_ne_b=bool((P.a_job != P.b_job).all()), same_core=bool((core_a == core_b).all()),
             core_column_is_a_core=bool((P.core.to_numpy() == core_a).all()),
             a_job_unique=bool(P.a_job.is_unique), a_and_b_disjoint=not (set(P.a_job) & set(P.b_job)),
             aryl_reference_same_F=bool(same_f[aryl_ref].all()), anchor_same_F=bool(same_f[anchor].all()),
             direct_reference_same_dipole_skeleton_face=bool(same_dsf[direct_ref].all()),
             reference_b_pos=sorted(set(B.pos[(P.ref_kind == "reference").to_numpy()])),
             n_aryl_reference=int(aryl_ref.sum()), n_direct_reference=int(direct_ref.sum()), n_anchor=int(anchor.sum()))
    r["ok"] = all(v for k, v in r.items() if isinstance(v, bool))
    return r


def check_b3(cand) -> dict:
    st = pd.read_csv(C.P4_STEREO, dtype=str, keep_default_na=False)
    ref = {(a, b): v for a, b, v in zip(st.a_job, st.b_job, st.v2)}
    shared = [(a, b, v) for a, b, v in zip(cand.a_job, cand.b_job, cand.p4_v2) if (a, b) in ref]
    bad = [dict(a_job=a, b_job=b, v2=v, p4_stereo=ref[(a, b)]) for a, b, v in shared if v != ref[(a, b)]]
    return dict(n_shared=len(shared), n_differ=len(bad), differ=bad[:20], ok=not bad)


def check_b4(P) -> dict:
    F0 = pd.read_csv(C.F_PAIRS_CR2, float_precision="round_trip")
    m = P.merge(F0, on=["a_job", "b_job"], suffixes=("", "_cr2"))
    mx = {t: float(np.abs(m[f"dd_{t}"] - m[f"dd_{t}_cr2"]).max()) if len(m) else 0.0 for t in DP.D1_TCOL}
    return dict(n_shared=int(len(m)), max_abs=mx, tol=CR2_TOL, ok=bool(len(m)) and max(mx.values()) <= CR2_TOL)


def check_b5(P) -> dict:
    dd = P[[f"dd_{t}" for t in DP.D1_TCOL]].to_numpy(float)
    res = np.abs(P.dd_barrier.to_numpy(float) - P[[f"dd_{t}" for t in DP.CHANNELS9]].to_numpy(float).sum(1))
    return dict(all_finite=bool(np.isfinite(dd).all()), max_identity_residual=float(res.max()), tol=IDENTITY_TOL,
                n_over=int((res > IDENTITY_TOL).sum()), ok=bool(np.isfinite(dd).all() and res.max() <= IDENTITY_TOL))


# ================================================================ the build
def build(out_dir: Path, with_b0: bool) -> dict:
    t0 = time.time()
    sg = pd.read_csv(C.SIGMA).set_index("X")
    gates = {}
    if with_b0:
        gates["B-0"] = check_b0(sg)
    r = pd.read_pickle(C.ROWATTR_R)
    rn = norm(r)
    rows = rn.set_index("job_id")
    U = pd.read_csv(C.USABLE)
    usable = set(U.loc[U.usable.astype(bool), "job_id"])
    pref, a0, fams = families(r, sg)
    pref_rows = set(pref.a_job) | set(pref.b_job)
    a0_rows = set(a0.a_job) | set(a0.b_job)
    gates["B-1"] = dict(n_pref_rows=len(pref_rows), n_a0_rows=len(a0_rows), overlap=sorted(pref_rows & a0_rows)[:20],
                        ok=not (pref_rows & a0_rows))
    cand, groups_log = candidate_pairs(fams, rows, usable)
    p4_rows, p4_info = p4(cand)
    cand = pd.concat([cand, p4_rows], axis=1)
    gates["B-3"] = check_b3(cand)
    keep = cand[cand.p4_v2 != "mismatch"].reset_index(drop=True)
    dropped = cand[cand.p4_v2 == "mismatch"]
    cr2 = {d["job_id"]: d for d in json.loads(C.LABELS_CR2.read_text())}
    P = assemble(keep, rows, sg, cr2)
    R = reaction_table(P, rows)
    anchor_family_rows = a0_rows | {m for f in fams if f["kind"] == "reference" and f["ref"] not in usable
                                    for m in f["members"]}
    S, status_extra = row_status(rn, usable, pref_rows | a0_rows, anchor_family_rows, cand, P, R)
    D = diff_vs_est(P, R, S)
    gates["B-2"] = check_b2(P, rows, usable)
    gates["B-4"] = check_b4(P)
    gates["B-5"] = check_b5(P)
    n_r, n_p = int(len(R)), int(len(P))
    gates["B-6"] = dict(n_reactions=n_r, range_reactions=C.RANGE_REACTIONS, n_pairs=n_p, range_pairs=C.RANGE_PAIRS,
                        ok=C.RANGE_REACTIONS[0] <= n_r <= C.RANGE_REACTIONS[1]
                        and C.RANGE_PAIRS[0] <= n_p <= C.RANGE_PAIRS[1])
    sc = S.status.value_counts().to_dict()
    gates["B-7"] = dict(n_rows=int(len(S)), status=sc, n_in_dataset=sc.get("in_dataset", 0), n_dataset_reactions=n_r,
                        n_not_usable=sc.get("not_usable", 0), want_not_usable=C.N_NOT_USABLE,
                        ok=len(S) == C.N_TARGETS and sc.get("in_dataset", 0) == n_r
                        and sc.get("not_usable", 0) == C.N_NOT_USABLE and set(S.job_id) == set(rn.job_id))
    # outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    texts = {"F_pairs_r.csv": P.to_csv(index=False, float_format=None), "B_reactions.csv": R.to_csv(index=False),
             "B_row_status.csv": S.to_csv(index=False), "B_diff_vs_est.csv": D.to_csv(index=False)}
    for name, text in texts.items():
        if "\r" in text:
            raise RuntimeError(f"{name}: line ending is not \\n")
        C.write_atomic(out_dir / name, text)
    folds = pd.read_csv(C.S_FOLDS, usecols=["core", "fold_T1"]).drop_duplicates()
    fold_of = dict(zip(folds.core, folds.fold_T1))
    Pf = P.assign(fold_T1=P.core.map(fold_of).fillna(-99).astype(int))
    xt = lambda col: {str(k): {str(kk): int(vv) for kk, vv in v.items()}            # noqa: E731
                      for k, v in Pf.groupby(col).ref_kind.value_counts().unstack(fill_value=0).T.to_dict().items()}
    gl = pd.DataFrame(groups_log)
    dclass = D.assign(reason=D.reason.map(reason_class)).groupby(["kind", "in_est", "in_server", "reason"]).size()
    rep = dict(spec="docs/specs/GNN3R.md 4", written=time.strftime("%Y-%m-%d %H:%M:%S"),
               slurm_job=os.environ.get("SLURM_JOB_ID"), pid=os.getpid(), out_dir=str(out_dir),
               versions=dict(pandas=pd.__version__, numpy=np.__version__, python=sys.version.split()[0]),
               inputs={str(p): C.sha256(p) for p in (C.ROWATTR_R, C.USABLE, C.LABELS_CR2, C.SIGMA, C.P4_STEREO,
                                                    C.F_PAIRS_CR2, C.REF_REACTIONS, C.REF_PAIRS, C.S_FOLDS)},
               code={n: C.sha256(C.D1PAIR_CODE / n) for n in ("d1_pairs.py", "d1_anchor_pairs.py", "d1_phase_p.py")},
               counts=dict(pref_pairs=int(len(pref)), a0_pairs=int(len(a0)),
                           families=dict(reference=sum(f["kind"] == "reference" for f in fams),
                                         anchor=sum(f["kind"] == "anchor" for f in fams)),
                           groups=gl.groupby("kind").agg(n=("src", "size"), with_pairs=("n_pairs", lambda s: int((s > 0).sum())),
                                                         pairs=("n_pairs", "sum")).astype(int).to_dict("index"),
                           candidate_pairs=int(len(cand)), pairs=n_p, reactions=n_r,
                           distinct_fam=int(P.fam.nunique()), cores=int(P.core.nunique()),
                           reference_b=int(P.loc[P.ref_kind == "reference", "b_job"].nunique()),
                           anchor_b=int(P.loc[P.ref_kind == "anchor", "b_job"].nunique()),
                           by_ref_kind=P.ref_kind.value_counts().to_dict(), by_dipole=xt("dipole"), by_pos=xt("pos"),
                           by_fold_T1=xt("fold_T1"), cores_without_fold=sorted(set(P.core) - set(fold_of)),
                           anchor_from=cand.loc[cand.ref_kind == "anchor"].anchor_from.value_counts().to_dict(),
                           fam_strings_with_several_anchor_b=int(
                               (P[P.ref_kind == "anchor"].groupby("fam").b_job.nunique() > 1).sum())),
               row_status=dict(counts=sc, order=list(STATUS_ORDER), **status_extra,
                               usable_not_in_dataset=S[(S.usable) & (S.status != "in_dataset")].to_dict("records")),
               p4=dict(**p4_info, n_pairs_checked=int(len(cand)), v2=cand.p4_v2.value_counts().to_dict(),
                       status=cand.p4_status.value_counts().to_dict(), decision=DP.P4_DECISION,
                       n_dropped=int(len(dropped)),
                       dropped=dropped[["a_job", "b_job", "ref_kind", "src"]].to_dict("records"),
                       ring_fail=cand.loc[cand.p4_v2 == "ring_fail", ["a_job", "b_job", "p4_detail"]].to_dict("records")[:50]),
               diff_vs_est=dict(n_rows=int(len(D)), by_reason={" | ".join(map(str, k)): int(v) for k, v in dclass.items()}),
               gates=gates, outputs={str(out_dir / n): C.sha256(out_dir / n) for n in texts},
               wall_s=round(time.time() - t0, 1))
    rep["failed"] = [k for k, v in gates.items() if not v["ok"]]
    rep["all_pass_before_rerun"] = not rep["failed"]
    C.write_json(out_dir / "B_report.json", rep)
    return rep


def label_report_current() -> bool:
    if not (C.LABEL_REPORT.is_file() and C.PAIRS_R.is_file()):
        return False
    lr = C.read_json(C.LABEL_REPORT)
    return bool(lr.get("all_pass")) and lr.get("outputs", {}).get(str(C.PAIRS_R)) == C.sha256(C.PAIRS_R)


def summary(rep):
    return json.dumps(dict(failed=rep["failed"], counts={k: rep["counts"][k] for k in ("pairs", "reactions",
                                                                                        "by_ref_kind", "distinct_fam")},
                           row_status=rep["row_status"]["counts"], p4_v2=rep["p4"]["v2"],
                           **{k: v["ok"] for k, v in rep["gates"].items()}), indent=1, default=str)


def cmd_build() -> int:
    if label_report_current():
        print(f"Phase B done ({C.LABEL_REPORT}); not rebuilt")
        return 0
    rp = C.RES / "B_report.json"
    if rp.is_file():
        old = C.read_json(rp)
        if old.get("all_pass_before_rerun") and all(Path(p).is_file() and C.sha256(p) == s
                                                    for p, s in old.get("outputs", {}).items()):
            print(f"Phase B outputs present and passed ({rp}); not rebuilt")
            return 0
    rep = build(C.RES, with_b0=True)
    print(summary(rep))
    return 0 if rep["all_pass_before_rerun"] else 3


def cmd_rerun() -> int:
    if label_report_current():
        print(f"Phase B done ({C.LABEL_REPORT}); not rerun")
        return 0
    rp = C.RES / "B_report.json"
    main_rep = C.read_json(rp)
    if not main_rep.get("all_pass_before_rerun"):
        print(f"STOP: {rp} did not pass: {main_rep.get('failed')}")
        return 3
    again = build(C.B_RERUN, with_b0=False)
    same = {n: dict(results=C.sha256(C.RES / n), rerun=C.sha256(C.B_RERUN / n)) for n in C.B_OUTPUT_NAMES}
    for v in same.values():
        v["equal"] = v["results"] == v["rerun"]
    gate = dict(rerun_dir=str(C.B_RERUN), rerun_pid=again["pid"], build_pid=main_rep["pid"], files=same,
                rerun_gates_pass=again["all_pass_before_rerun"],
                ok=all(v["equal"] for v in same.values()) and again["all_pass_before_rerun"])
    main_rep["gates"]["B-R"] = gate
    main_rep["failed"] = [k for k, v in main_rep["gates"].items() if not v["ok"]]
    main_rep["all_pass"] = not main_rep["failed"]
    C.write_json(rp, main_rep)
    if main_rep["all_pass"]:
        C.write_json(C.LABEL_REPORT, {"all_pass": True, "outputs": {str(C.PAIRS_R): C.sha256(C.PAIRS_R)},
                                      "gates": {k: v["ok"] for k, v in main_rep["gates"].items()},
                                      "spec": "docs/specs/GNN3R.md 4-7", "b_report": str(rp),
                                      "written": time.strftime("%Y-%m-%d %H:%M:%S")})
    print(json.dumps(dict(all_pass=main_rep["all_pass"], failed=main_rep["failed"], B_R=gate), indent=1, default=str))
    return 0 if main_rep["all_pass"] else 3


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("build", "rerun"):
        raise SystemExit(__doc__)
    sys.exit(cmd_build() if sys.argv[1] == "build" else cmd_rerun())
