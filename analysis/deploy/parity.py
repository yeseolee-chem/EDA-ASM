#!/usr/bin/env python3
"""parity.py — DEPLOY-MODEL Phase Q (docs/specs/DEPLOY_MODEL.md 7): the LADDER g2 inputs (label-dependent path:
$G2/g2_final, $G2/X182_d1_g2.parquet, $G2/fu2/tokens) against the lf inputs (label-free path: $DEP/g2lf_final,
$DEP/X182_lf.parquet, $DEP/fu2_lf/tokens) on the same reactions, atoms aligned by the L-2 permutation.
Before training; not a verdict except G-A (pre-registered) and the interpretation rule (STOP on a suspected bug).

  python parity.py run       # -> results/Q_parity.csv (one row per $DEP/g2lf_final reaction), results/Q_parity.json

Job: analysis/d1_pair/jobs/g.slurm from analysis/d1_pair, --export=ALL,PYENV=venv5,D1PAIR_PHASE=F (fragmenter + Si
entries for the LADDER map-source reconstruction). Needs results/F_report.json (feat_lf.py merge); tokens are compared
where both token files exist (results/T_lf_check.json is recorded, not required).

Per reaction (one row of Q_parity.csv):
  permutation  p with lf_ts[p] == g2_ts (p[i_label] = i_lf): identical (element, coordinates) set, |dx| <= 1e-6 A per
               coordinate (= the L-2 rule), recomputed here; rel1 / rel2 likewise. If Phase L recorded one (any key
               containing 'perm' in g2lf_final/<rid>/result.json, or a 'perm' column of results/L_status.csv), it is
               compared (perm_L: same / inverse / differs / absent) and this one is used.
  L-3          lf A_idx and formed_pairs_ts mapped through p vs the label A_idx ($G2/G/input_meta_d1_g2.csv) and the
               LADDER formed_pairs_label.
  182          max |lf - g2| over the 182 columns, its column, the columns above 1e-6 (all, and outside B6).
  tokens       after the permutation: Z / bin / edges + etype exact, num per column, pos, glob (= the 182) max |diff|.
  redetection  lf xtb_status (results/F_report.json; partition:* / partition_mismatch_g1 = F-2 failure).
  map_source   LADDER partition source reconstructed without DFT: fragmenter.partition on the g2_final TS / references
               (label order) reproduces the label A_idx -> 'g1_redetect' (r6_partition.label_partition's first branch),
               else 'dft_ts' (LADDER borrowed the DFT-TS atom map) when LADDER has a feature row, else 'none'.
  category     both paths ok: equal (182 and tokens within 1e-6, discrete parts identical) | b6_only (only the 3 B6
               columns, spec 7 (b)) | ladder_dft_map (map_source dft_ts, spec 7 (a)) | perm_symmetric_map (see below) |
               unexplained (spec 7 (c) candidate).
  perm_symmetric_map (decision recorded here, standing instruction = recommended option): the TS atom order differs
               (p not the identity) and every 182 difference above 1e-6 lies in the reactant-side columns whose atoms
               come from fragmenter.partition's first heavy-atom map (mulliken_R_*, wbo_valence_R_*, dist_R_*; + B6),
               and within each such group the lf values are a permutation of the LADDER values (|sorted diff| <=
               1e-6): the VF2 map of a symmetric reactant picks the other equivalent atom when the TS atoms come in
               another order (seen on the fixtures: cyclooctyne, r2_dph [0, 1] vs [1, 0]; atoms.sym_remap keeps the
               reacting atoms fixed, so it cannot undo it). Token differences are allowed only in q_rel / v_rel (per
               fragment the lf values are a permutation of the LADDER ones) and in dq_dist / dv_dist alongside them
               (q_frag / v_frag, Z, bin, edges, pos must agree).
Summary (Q_parity.json): common reactions, how many agree within 1e-6, the columns that differ, redetection failures
and their dipole / skeleton distribution; G-A: redetection failures <= 2 % of the Phase L .done reactions (else STOP);
interpretation rule: any 'unexplained' reaction = STOP (bug suspected; reported with its columns).
Exit codes: 0, 2 (inputs missing), 3 (STOP: G-A or unexplained differences).
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
THIS = Path(__file__).resolve()
sys.path.insert(0, str(HERE))
import feat_lf as FL  # noqa: E402

D = FL.D
TOL = 1e-6
GA_MAX = 0.02                                          # spec 7 G-A
G2_TOK = D.G2 / "fu2" / "tokens"
G2_META = D.G2 / "G" / "input_meta_d1_g2.csv"
G2_RES = D.REPO / "analysis" / "d1_g2" / "results"
G2_STATUS = G2_RES / "G_status.csv"                   # dipole, skeleton, remapped (read only)
G2_TMAN = G2_RES / "fu2" / "T_manifest.csv"
G2_TREP = G2_RES / "fu2" / "T_report.json"
G2_MREP = D.G2 / "F" / "mergeroot_report.json"         # LADDER feature failure reasons (not_ok)
LF_TMAN = D.RES / "fu2_lf" / "T_manifest.csv"
LF_TREP = D.RES / "fu2_lf" / "T_report.json"
L_STATUS, L_REPORT = D.RES / "L_status.csv", D.RES / "L_report.json"
OUT_CSV, OUT_JSON = D.RES / "Q_parity.csv", D.RES / "Q_parity.json"
B6 = ("nu_imag", "mode_share", "mode_async")
R_GROUPS = (("mulliken_R_dip_0", "mulliken_R_dip_1", "mulliken_R_dip_2"),
            ("wbo_valence_R_dip_0", "wbo_valence_R_dip_1", "wbo_valence_R_dip_2"),
            ("mulliken_R_dph_0", "mulliken_R_dph_1"), ("wbo_valence_R_dph_0", "wbo_valence_R_dph_1"),
            ("dist_R_dip_ab", "dist_R_dip_bc"), ("dist_R_dip_ac",), ("dist_R_dph_ab",))
R_COLS = {c for g in R_GROUPS for c in g}
TOK_SYM_COLS = ("q_rel", "v_rel", "dq_dist", "dv_dist")


def read_xyz(p):
    import fragmenter as fr
    return fr.read_xyz(p)


def perm_of(lab, lf, tol=TOL):
    """p with lf_xyz[p] == lab_xyz (p[i_label] = i_lf), or (None, reason); max |coordinate diff|."""
    (sl, xl), (sf, xf) = lab, lf
    if len(sl) != len(sf):
        return None, f"n_atoms {len(sl)} vs {len(sf)}", np.inf
    p, worst = [], 0.0
    for i in range(len(sl)):
        d = np.abs(xf - xl[i]).max(axis=1)
        c = [j for j in np.flatnonzero(d <= tol) if sf[j] == sl[i]]
        if len(c) != 1:
            return None, f"atom {i}: {len(c)} candidates", float(d.min())
        p.append(int(c[0]))
        worst = max(worst, float(d[c[0]]))
    if sorted(p) != list(range(len(sf))):
        return None, "not a bijection", worst
    return p, "ok", worst


def _l_perm(res, lrow):
    """a permutation recorded by Phase L (result.json key or L_status.csv column containing 'perm'), or None."""
    cands = [v for k, v in (res or {}).items() if "perm" in k.lower()] + [v for k, v in (lrow or {}).items()
                                                                          if "perm" in str(k).lower()]
    for v in cands:
        if isinstance(v, str):
            v = [int(x) for x in re.findall(r"-?\d+", v)]
        if isinstance(v, (list, tuple)) and v and all(isinstance(x, (int, np.integer)) for x in v):
            return [int(x) for x in v]
    return None


def _cmp_tokens(tg, tl, p):
    """g2 token (label order) vs lf token (lf order) permuted into label order."""
    q = np.empty(len(p), dtype=np.int64)
    q[np.asarray(p)] = np.arange(len(p))                      # q[i_lf] = i_label
    P = np.asarray(p)
    out = dict(tok_Z_equal=bool(np.array_equal(tg["Z"], tl["Z"][P])),
               tok_bin_equal=bool(np.array_equal(tg["bin"], tl["bin"][P])))
    eg = {(int(a), int(b), int(t)) for (a, b), t in zip(tg["edges"], tg["etype"])}
    el = {(int(min(q[a], q[b])), int(max(q[a], q[b])), int(t)) for (a, b), t in zip(tl["edges"], tl["etype"])}
    out["tok_edges_equal"] = eg == el
    num = {}
    if tg["num"].shape == tl["num"].shape:
        dn = np.abs(tg["num"] - tl["num"][P])
        num = {k: float(dn[:, j].max()) for j, k in enumerate(NUM_COLS)}
    else:
        out["tok_num_shape"] = f"{tg['num'].shape} vs {tl['num'].shape}"
    out["tok_num_max"] = num
    out["tok_pos_maxdiff"] = float(np.abs(tg["pos"] - tl["pos"][P]).max())
    out["tok_glob_maxdiff"] = float(np.abs(tg["glob"] - tl["glob"]).max()) if tg["glob"].shape == tl["glob"].shape else np.inf
    # symmetric-map test of the reactant-map columns q_rel / v_rel (per fragment multiset equality; dq_dist / dv_dist =
    # q_frag - q_rel / v_frag - v_rel then follow, with q_frag / v_frag required equal)
    sym_ok = True
    if num:
        frag = tg["bin"][:, 0]
        for j, k in enumerate(NUM_COLS):
            if k not in ("q_rel", "v_rel") or num[k] <= TOL:
                continue
            for f in (0, 1):
                a, b = np.sort(tg["num"][frag == f, j]), np.sort(tl["num"][P][frag == f, j])
                sym_ok &= bool(np.abs(a - b).max() <= TOL)
    out["tok_sym_multiset_ok"] = bool(sym_ok)
    return out


NUM_COLS = None                                           # set from the g2 T_report in run(); workers get it via init


def _init(num_cols):
    global NUM_COLS
    NUM_COLS = list(num_cols)


def _is(v, want):
    """v is a (numpy) bool equal to want (NaN / None -> False)."""
    return isinstance(v, (bool, np.bool_)) and bool(v) == want


def _one(task):
    rid, lab_A, have_g2_tok, have_lf_tok, lrow = task
    gd, ld = D.G2_FINAL / str(rid), D.LF_FINAL / str(rid)
    mk = lambda d: ".done" if (d / ".done").is_file() else (sorted(p.name for p in d.glob(".fail_*")) or [None])[0]  # noqa: E731
    r = dict(rxn_id=rid, g2_marker=mk(gd) if gd.is_dir() else "missing", lf_marker=mk(ld) if ld.is_dir() else "missing")
    try:
        lres = json.loads((ld / "result.json").read_text()) if (ld / "result.json").is_file() else {}
        gres = json.loads((gd / "result.json").read_text()) if (gd / "result.json").is_file() else {}
        lab_fp = sorted(tuple(sorted(map(int, x))) for x in gres.get("formed_pairs_label") or [])
        r["job_id"] = lres.get("job_id") or gres.get("job_id")
        r["ladder_remapped"] = gres.get("remapped")
        if r["g2_marker"] == ".done":                          # LADDER map source (no DFT structure needed)
            import fragmenter as fr
            ts, r1, r2 = (read_xyz(gd / f) for f in ("ts.xyz", "rel1.xyz", "rel2.xyz"))
            P0 = fr.partition(ts, r1, r2)
            r["g2_redetect_status"] = P0.status
            r["g2_redetect_matches_label"] = bool(P0.status == "ok" and sorted(P0.A) == lab_A)
        if r["g2_marker"] != ".done" or r["lf_marker"] != ".done":
            return r
        pts = perm_of(read_xyz(gd / "ts.xyz"), read_xyz(ld / "ts.xyz"))
        p, why, worst = pts
        r.update(perm_status=why, perm_coord_maxdiff=worst)
        for k in (1, 2):
            pk, wk, _ = perm_of(read_xyz(gd / f"rel{k}.xyz"), read_xyz(ld / f"rel{k}.xyz"))
            r[f"perm_rel{k}"] = "identity" if pk == list(range(len(pk or []))) and pk else (wk if pk is None else "permuted")
        if p is None:
            return r
        r["perm_identity"] = p == list(range(len(p)))
        r["perm"] = " ".join(map(str, p)) if not r["perm_identity"] else ""
        pl = _l_perm(lres, lrow)
        if pl is None:
            r["perm_L"] = "absent"
        else:
            inv = [0] * len(p)
            for a, b in enumerate(p):
                inv[b] = a
            r["perm_L"] = "same" if pl == p else ("inverse" if pl == inv else "differs")
        li = (lrow or {}).get("l2_ts_identity")                # finalize_lf report: identity flag of its L-2 perm
        if isinstance(li, str):
            li = {"True": True, "False": False}.get(li)
        r["perm_L_identity_agrees"] = None if not isinstance(li, (bool, np.bool_)) else bool(li) == r["perm_identity"]
        q = {b: a for a, b in enumerate(p)}                    # lf index -> label index
        A_lf = sorted(q[int(i)] for i in lres.get("A_idx", []))
        fp_lf = sorted(tuple(sorted((q[int(a)], q[int(b)]))) for a, b in lres.get("formed_pairs_ts", []))
        r["L3_A_idx_equal"] = A_lf == lab_A
        r["L3_formed_equal"] = fp_lf == lab_fp
        il, ig = lres.get("imag_freqs_cm") or [], gres.get("imag_freqs_cm") or []
        r["L4_dnu_cm"] = float(il[0] - ig[0]) if il and ig else np.nan
        if have_g2_tok and have_lf_tok:
            with np.load(G2_TOK / f"{rid}.npz") as zg, np.load(D.TOK_LF / f"{rid}.npz") as zl:
                tg, tl = {k: zg[k] for k in zg.files}, {k: zl[k] for k in zl.files}
            t = _cmp_tokens(tg, tl, p)
            num = t.pop("tok_num_max")
            r.update(t)
            r["tok_num_maxdiff"] = max(num.values()) if num else np.inf
            r["tok_num_maxcol"] = max(num, key=num.get) if num else ""
            r["tok_num_cols_gt_tol"] = ";".join(k for k, v in num.items() if v > TOL)
    except Exception as e:                                     # noqa: BLE001 — a per-reaction comparison error, reported
        r["error"] = f"{type(e).__name__}: {e}"[:300]
    return r


def _txt(v):
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)


def category(r):
    """the interpretation category of one Q_parity row (a pandas Series / dict-like; see the module docstring)."""
    if not (_is(r.get("g2_ok"), True) and _is(r.get("lf_ok"), True)):
        return "not_both_ok"
    if _txt(r.get("error")):
        return "compare_error"
    x_cols = [c for c in _txt(r.get("x182_cols_gt_tol")).split(";") if c]
    t_cols = [c for c in _txt(r.get("tok_num_cols_gt_tol")).split(";") if c]
    pos = r.get("tok_pos_maxdiff")
    disc = _is(r.get("tok_Z_equal"), True) and _is(r.get("tok_bin_equal"), True) and \
        _is(r.get("tok_edges_equal"), True) and pos is not None and float(pos) <= TOL
    if not x_cols and not t_cols and disc:
        return "equal"
    if disc and not t_cols and set(x_cols) <= set(B6):
        return "b6_only"
    if r.get("ladder_map_source") == "dft_ts":
        return "ladder_dft_map"
    if (_is(r.get("perm_identity"), False) and disc and _is(r.get("x182_sym_ok"), True)
            and set(t_cols) <= set(TOK_SYM_COLS) and (not t_cols or _is(r.get("tok_sym_multiset_ok"), True))
            and ("dq_dist" not in t_cols or "q_rel" in t_cols) and ("dv_dist" not in t_cols or "v_rel" in t_cols)):
        return "perm_symmetric_map"
    return "unexplained"


def _sym_ok(xl, xg, diff_cols):
    """reactant-side 182 differences are permutations within their groups (perm_symmetric_map rule)."""
    if not set(diff_cols) <= (R_COLS | set(B6)):
        return False
    for g in R_GROUPS:
        if not set(g) & set(diff_cols):
            continue
        a, b = np.sort(xl[list(g)].astype(float)), np.sort(xg[list(g)].astype(float))
        if not np.abs(a - b).max() <= TOL:
            return False
    return True


def run():
    FL.parent_env()
    fr_ = FL.need_f_report()
    for p in (FL.X182_G2, G2_META, G2_STATUS, G2_TMAN, G2_TREP):
        if not Path(p).is_file():
            D.not_ready(f"{p} missing")
    num_cols = D.read_json(G2_TREP)["num_cols"]
    lab_A = {int(r.rxn_id): sorted(int(x) for x in str(r.A_idx).split()) for r in pd.read_csv(G2_META).itertuples()}
    st = pd.read_csv(G2_STATUS)
    info = st.set_index("rxn_id")[["job_id", "dipole", "skeleton", "remapped"]].to_dict("index")
    xg = pd.read_parquet(FL.X182_G2).set_index("rxn_id")
    xl = pd.read_parquet(D.X182_LF).set_index("rxn_id")
    cols = list(xg.columns)
    if list(xl.columns) != cols:
        D.stop("X182_lf columns != X182_d1_g2 columns (F-1)")
    tg = pd.read_csv(G2_TMAN)
    g2_bad = {int(b["rxn_id"]) for b in D.read_json(G2_TREP).get("bad") or []}
    g2_tok = set(int(r) for r in tg.rxn_id) - g2_bad
    lf_tok, lf_bad = set(), set()
    if LF_TMAN.is_file() and LF_TREP.is_file():
        lf_bad = {int(b["rxn_id"]) for b in D.read_json(LF_TREP).get("bad") or []}
        lf_tok = set(int(r) for r in pd.read_csv(LF_TMAN).rxn_id) - lf_bad
    lstat = {}
    if L_STATUS.is_file():
        L = pd.read_csv(L_STATUS)
        if "rxn_id" in L:
            lstat = {int(r["rxn_id"]): r for r in L.to_dict("records")}
    ids = sorted(int(p.name) for p in D.LF_FINAL.iterdir() if p.is_dir() and p.name.isdigit())
    g2_done = sorted(int(p.parent.name) for p in D.G2_FINAL.glob("*/.done") if p.parent.name.isdigit())
    tasks = [(rid, lab_A.get(rid), rid in g2_tok, rid in lf_tok, lstat.get(rid)) for rid in ids]
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(num_cols,)) as pool:
        rows = pool.map(_one, tasks, chunksize=20)
    R = pd.DataFrame(rows)
    # ---- 182 columns
    lf_why = {int(k): v for k, v in (fr_.get("not_ok") or {}).items()}
    g2_why = {int(k): v for k, v in (D.read_json(G2_MREP).get("not_ok") or {}).items()} if G2_MREP.is_file() else {}
    g1lf = pd.read_parquet(Path(fr_["outputs"]["g1"]), columns=["rxn_id", "xtb_status"]).set_index("rxn_id").xtb_status
    ext = []
    for r in R.itertuples():
        rid = r.rxn_id
        e = dict(dipole=info.get(rid, {}).get("dipole"), skeleton=info.get(rid, {}).get("skeleton"),
                 lf_xtb_status=g1lf.get(rid), g2_row=rid in xg.index, lf_row=rid in xl.index,
                 g2_tok=rid in g2_tok, lf_tok=rid in lf_tok)
        e["redetect_status"] = ("not_run" if rid not in g1lf.index else
                                ("fail:" + str(g1lf[rid]) if FL.is_redetect_fail(g1lf[rid]) else "ok"))
        e["g2_ok"], e["lf_ok"] = e["g2_row"] and e["g2_tok"], e["lf_row"] and e["lf_tok"]
        e["lf_reason"] = ("ok" if e["lf_ok"] else (f"L:{r.lf_marker}" if r.lf_marker != ".done" else
                          lf_why.get(rid, "no_token" if e["lf_row"] else "no_x182_row")))
        e["g2_reason"] = ("ok" if e["g2_ok"] else (f"G2:{r.g2_marker}" if r.g2_marker != ".done" else
                          g2_why.get(rid, "no_token" if e["g2_row"] else "no_x182_row")))
        gs = getattr(r, "g2_redetect_matches_label", None)
        e["ladder_map_source"] = ("none" if not e["g2_row"] else ("g1_redetect" if _is(gs, True) else "dft_ts"))
        if e["g2_row"] and e["lf_row"]:
            d = (xl.loc[rid, cols].astype(float) - xg.loc[rid, cols].astype(float)).abs()
            big = [c for c in cols if d[c] > TOL]
            e.update(x182_maxdiff=float(d.max()), x182_maxcol=str(d.idxmax()), x182_n_gt_tol=len(big),
                     x182_cols_gt_tol=";".join(big[:40]), x182_maxdiff_nonB6=float(d[[c for c in cols if c not in B6]].max()),
                     x182_n_nonzero=int((d > 0).sum()), x182_sym_ok=_sym_ok(xl.loc[rid], xg.loc[rid], big))
        ext.append(e)
    R = pd.concat([R, pd.DataFrame(ext)], axis=1)
    for c in ("tok_num_maxdiff", "tok_pos_maxdiff", "tok_glob_maxdiff", "x182_maxdiff", "perm_identity",
              "tok_Z_equal", "tok_bin_equal", "tok_edges_equal", "tok_sym_multiset_ok", "x182_sym_ok", "x182_cols_gt_tol",
              "tok_num_cols_gt_tol", "perm_L_identity_agrees"):
        if c not in R:
            R[c] = np.nan

    R["category"] = [category(r) for _, r in R.iterrows()]
    keep = ["rxn_id", "job_id", "dipole", "skeleton", "g2_marker", "lf_marker", "g2_ok", "lf_ok", "g2_reason",
            "lf_reason", "category", "x182_maxdiff", "x182_maxcol", "x182_n_gt_tol", "x182_cols_gt_tol",
            "x182_maxdiff_nonB6", "x182_n_nonzero", "tok_num_maxdiff", "tok_num_maxcol", "tok_num_cols_gt_tol",
            "tok_pos_maxdiff", "tok_glob_maxdiff", "tok_Z_equal", "tok_bin_equal", "tok_edges_equal",
            "redetect_status", "lf_xtb_status", "ladder_map_source", "g2_redetect_status", "ladder_remapped",
            "perm_status", "perm_identity", "perm_L", "perm_L_identity_agrees", "perm_coord_maxdiff", "perm_rel1", "perm_rel2", "L3_A_idx_equal",
            "L3_formed_equal", "L4_dnu_cm", "perm", "error"]
    for c in keep:
        if c not in R:
            R[c] = np.nan
    R = R[keep]
    D.write_csv(OUT_CSV, R)
    # ---- summary
    both = R[R.g2_ok.astype(bool) & R.lf_ok.astype(bool)]
    diffcols = Counter(c for s in both.x182_cols_gt_tol.fillna("") for c in str(s).split(";") if c)
    tokcols = Counter(c for s in both.tok_num_cols_gt_tol.fillna("") for c in str(s).split(";") if c)
    colmax = {}
    for rid in both.rxn_id:
        d = (xl.loc[rid, cols].astype(float) - xg.loc[rid, cols].astype(float)).abs()
        for c in d.index[d > TOL]:
            colmax[c] = max(colmax.get(c, 0.0), float(d[c]))
    n_ldone = int((R.lf_marker == ".done").sum())                # Phase L .done reactions (live scan)
    n_ldone_report = (D.read_json(L_REPORT).get("L1") or {}).get("n_done") if L_REPORT.is_file() else None
    f2 = fr_.get("F2") or {}
    red = R[R.redetect_status.astype(str).str.startswith("fail:")]
    ga_rate = (f2.get("n", len(red)) / n_ldone) if n_ldone else None
    unexpl = both[both.category == "unexplained"]
    cat = both.category.value_counts().to_dict()
    lf_only = R[R.lf_ok.astype(bool) & ~R.g2_ok.astype(bool)]
    g2_only = R[R.g2_ok.astype(bool) & ~R.lf_ok.astype(bool)]
    tchk = D.read_json(D.RES / "T_lf_check.json") if (D.RES / "T_lf_check.json").is_file() else None
    stop = []
    if ga_rate is None or ga_rate > GA_MAX:
        stop.append(f"G-A: redetection failures {f2.get('n')} / {n_ldone} Phase L .done = {ga_rate} > {GA_MAX}")
    if len(unexpl):
        stop.append(f"interpretation rule: {len(unexpl)} reactions with unexplained (non-B6, non-symmetric-map) "
                    f"differences: {unexpl.rxn_id.tolist()[:20]}")
    n_err = int(R.error.notna().sum())
    if n_err:
        stop.append(f"{n_err} reactions could not be compared (error column)")
    if n_ldone_report is not None and int(n_ldone_report) != n_ldone:
        stop.append(f"results/L_report.json L1.n_done {n_ldone_report} != live .done count {n_ldone} (rerun L report)")
    if (R.perm_L_identity_agrees == False).any():                                     # noqa: E712
        stop.append(f"L-2 identity flags of results/L_status.csv disagree with the recomputed permutation for "
                    f"{R.rxn_id[R.perm_L_identity_agrees == False].tolist()[:20]}")                  # noqa: E712
    rep = dict(spec="docs/specs/DEPLOY_MODEL.md 7 (Phase Q; STOP 16 Q)", written=FL.now(), tol=TOL,
               inputs=dict(x182_g2=str(FL.X182_G2), x182_g2_sha256=D.sha256(FL.X182_G2), x182_lf=str(D.X182_LF),
                           x182_lf_sha256=D.sha256(D.X182_LF), tokens_g2=str(G2_TOK), tokens_lf=str(D.TOK_LF),
                           g2_T_manifest_sha256=D.sha256(G2_TMAN),
                           lf_T_manifest_sha256=D.sha256(LF_TMAN) if LF_TMAN.is_file() else None,
                           F_report_sha256=D.sha256(FL.F_REPORT), L_status=str(L_STATUS) if L_STATUS.is_file() else None,
                           T_lf_check_stop=(tchk or {}).get("tokens", {}).get("stop") if tchk else None),
               counts=dict(lf_folders=len(ids), lf_done=int((R.lf_marker == ".done").sum()), g2_done=len(g2_done),
                           g2_done_without_lf_folder=len(set(g2_done) - set(ids)), g2_ok=int(R.g2_ok.astype(bool).sum()),
                           lf_ok=int(R.lf_ok.astype(bool).sum()), common_both_ok=len(both), lf_only_ok=len(lf_only),
                           g2_only_ok=len(g2_only)),
               lf_only_ok=dict(rxn=lf_only.rxn_id.tolist()[:200], g2_reasons=dict(Counter(lf_only.g2_reason).most_common(20))),
               g2_only_ok=dict(rxn=g2_only.rxn_id.tolist()[:200], lf_reasons=dict(Counter(g2_only.lf_reason).most_common(20))),
               agreement=dict(x182_equal=int((both.x182_n_gt_tol == 0).sum()),
                              x182_bit_equal=int((both.x182_n_nonzero == 0).sum()),
                              tokens_equal=int(((both.tok_num_maxdiff <= TOL) & (both.tok_pos_maxdiff <= TOL)
                                                & both.tok_Z_equal.astype(bool) & both.tok_bin_equal.astype(bool)
                                                & both.tok_edges_equal.astype(bool)).sum()),
                              all_equal=int((both.category == "equal").sum()),
                              x182_maxdiff=float(both.x182_maxdiff.max()) if len(both) else None,
                              x182_maxdiff_nonB6=float(both.x182_maxdiff_nonB6.max()) if len(both) else None,
                              tok_num_maxdiff=float(both.tok_num_maxdiff.max()) if len(both) else None),
               categories=cat,
               differing_columns=dict(x182_n_reactions=dict(diffcols.most_common()), x182_max_absdiff=colmax,
                                      token_num_n_reactions=dict(tokcols.most_common())),
               differing_reactions={str(r.rxn_id): dict(category=r.category, cols=r.x182_cols_gt_tol,
                                                        tok=r.tok_num_cols_gt_tol, perm_identity=r.perm_identity,
                                                        map_source=r.ladder_map_source)
                                    for r in both[both.category != "equal"].itertuples()},
               permutation=dict(identity=int((R.perm_identity == True).sum()), non_identity=int((R.perm_identity == False).sum()),  # noqa: E712
                                status=R.perm_status.value_counts().to_dict(), perm_L=R.perm_L.value_counts().to_dict(),
                                L_status_identity_agrees=R.perm_L_identity_agrees.astype(str).value_counts().to_dict(),
                                coord_maxdiff=float(R.perm_coord_maxdiff.max()) if R.perm_coord_maxdiff.notna().any() else None,
                                rel=dict(rel1=R.perm_rel1.value_counts().to_dict(), rel2=R.perm_rel2.value_counts().to_dict())),
               L3=dict(A_idx_differs=R.loc[R.L3_A_idx_equal == False, "rxn_id"].tolist(),  # noqa: E712
                       formed_differs=R.loc[R.L3_formed_equal == False, "rxn_id"].tolist()),  # noqa: E712
               L4_dnu_cm=dict(max_abs=float(R.L4_dnu_cm.abs().max()) if R.L4_dnu_cm.notna().any() else None,
                              n_gt_1=int((R.L4_dnu_cm.abs() > 1).sum())),
               map_source=R.ladder_map_source.value_counts().to_dict(),
               redetection=dict(rule=FL.REDETECT_FAIL, n=f2.get("n"), n_L_done=n_ldone,
                                n_L_done_L_report=n_ldone_report, rate=ga_rate,
                                statuses=f2.get("statuses"), rxn=red.rxn_id.tolist(),
                                by_dipole=red.dipole.value_counts().to_dict(), by_skeleton=red.skeleton.value_counts().to_dict(),
                                ladder_map_source=red.ladder_map_source.value_counts().to_dict()),
               G_A=dict(threshold=GA_MAX, rate=ga_rate, pass_=ga_rate is not None and ga_rate <= GA_MAX),
               interpretation=dict(rule="spec 7: (a) ladder_dft_map, (b) b6_only, (c) a bug if non-B6 columns differ at "
                                        "random; decision: perm_symmetric_map (TS atom order differs; reactant-side "
                                        "columns are a permutation within their groups) is explained, any other "
                                        "difference > 1e-6 = 'unexplained' = STOP",
                                   n_unexplained=len(unexpl),
                                   unexplained={str(r.rxn_id): r.x182_cols_gt_tol for r in unexpl.itertuples()}),
               n_compare_errors=n_err, stop=bool(stop), stop_reasons=stop)
    D.write_json(OUT_JSON, rep)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("differing_reactions",)}, indent=1, default=str))
    if stop:
        print(f"STOP: {stop}", file=sys.stderr)
    return 3 if stop else 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["run"]:
        sys.exit(run())
    print(__doc__)
    sys.exit(2)
