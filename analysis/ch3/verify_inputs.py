#!/usr/bin/env python3
"""verify_inputs.py — CH3-MODEL Phase V (docs/specs/CH3_MODEL.md 4): the regenerated lf inputs ($CH3/lf, reports in
results/lf_inputs) against the values recorded in the deploy commit's reports -> results/V_inputs.json.
CPU, env_d1fu2 (jobs/cpu_ch3.slurm CH3ENV=d1fu2). Exit 0 all pass, 2 not ready, 3 STOP.

  python verify_inputs.py

V-1 .done reactions in $CH3/lf/g2lf_final and labels_lf.json records = 8,616 (deploy L_report / M_report)
V-2 labels_lf.json sha256 = e7fdf9b5... (deploy F_report.labels_sha256)
V-3 input_meta_lf.csv sha256 = 104ab323... (deploy F_report.meta_sha256)
V-4 X182_lf.parquet sha256 = 8917a728... (deploy fu2_lf/T_report.x182_sha256), 8,504 rows (deploy F_report.n_feat)
V-5 token manifest sha256 = 72e0523c... (deploy T_report.manifest_sha256), 8,502 token reactions
V-6 DeployData('lf') usable pair list sha256 = deploy K_report.pair_list_sha
Content comparison, only where a sha256 differs (spec 4): the deploy scratch files were deleted (R2-DEL), so the
comparison uses what the deploy commit recorded: row lists (deploy L_status.csv .done reactions, F_report not_ok),
the per-reaction token file sha256 of the deploy T_manifest.csv (equal sha256 = token arrays identical, max |diff| 0),
and the 182 columns against the 182 global values stored in those identical token files (max |diff| <= 1e-6).
A difference beyond that, or one that cannot be compared, fails the check (STOP).
Bridge through the LADDER g2 inputs (added 11-10 after the first V run, job 1070701: only 3,900 token files were
byte-identical; deploy Q_parity itself found the deploy lf 182 columns bit-equal to g2 in only 4,145 reactions, i.e.
xtb values are reproducible to ~1e-6 but not bit for bit): deploy Q_parity.csv (committed) records, per reaction, the
max |deploy lf - g2| of the 182 columns and of the token arrays (num, pos, glob; Z / bin / edges equality) with the
lf -> label atom permutation. The g2 files are checked to be the ones Q_parity used (sha256 in Q_parity.json). For a
reaction whose new token is not byte-identical to deploy's:
  exact  the deploy record is 0 (deploy lf == g2): |new - deploy| = |new - g2|, computed here;
  bound  otherwise |new - deploy| <= |new - g2| + |deploy - g2| (triangle inequality per column);
  record category perm_symmetric_map (7 reactions, lf and g2 differ by a symmetric reactant map): the new |new - g2|
         must reproduce the recorded |deploy - g2| (same maximum column, |difference of the maxima| <= 1e-6);
  none   no g2 counterpart in the record (deploy lf_only_ok: g2 feature failure) -> listed, not comparable.
Content passes when every exact / bound value is <= 1e-6, every record case reproduces, and the not-comparable
reactions are only the deploy lf_only_ok ones (at most 3).
The gates of the regeneration reports themselves (L all_pass, M all_pass, F no stop, T gate / tokens no stop) are
recorded and required.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ch3_common as C3  # noqa: E402

WANT = dict(labels="e7fdf9b57371f7c8e2851cd7fef4ef131604b6257047bd5e75f55ee326b5f720",
            meta="104ab323de72fb7e4a999744e5957914e3d4dfedefbeddafa8c1bf4941451a09",
            x182="8917a728c231a48baff5bafe697dc88f745e9c0fc8d807690805c47dbf42bd18",
            manifest="72e0523c2ac14f2312431f86109191858de314eed875f74486d6eefeb5db2b92")
N_DONE, N_FEAT, N_TOK = 8616, 8504, 8502
TOL = 1e-6
NONE_MAX = 3                                        # deploy Q_parity lf_only_ok reactions (no g2 counterpart)


def _perm(q, n):
    """deploy Q_parity perm column: p[i_label] = i_lf ('' / NaN = identity)."""
    s = q.get("perm")
    if s is None or (isinstance(s, float) and np.isnan(s)) or not str(s).strip():
        return np.arange(n)
    return np.array([int(x) for x in str(s).split()])


def _tok_cmp(tg, tl, p, num_cols):
    """= analysis/deploy/parity.py _cmp_tokens (g2 token in label order vs lf token permuted into label order)."""
    q = np.empty(len(p), dtype=np.int64)
    q[p] = np.arange(len(p))
    disc = bool(np.array_equal(tg["Z"], tl["Z"][p]) and np.array_equal(tg["bin"], tl["bin"][p]))
    eg = {(int(a), int(b), int(t)) for (a, b), t in zip(tg["edges"], tg["etype"])}
    el = {(int(min(q[a], q[b])), int(max(q[a], q[b])), int(t)) for (a, b), t in zip(tl["edges"], tl["etype"])}
    disc = disc and eg == el
    dn = np.abs(tg["num"] - tl["num"][p]) if tg["num"].shape == tl["num"].shape else None
    num = float(dn.max()) if dn is not None and dn.size else np.inf
    numcol = num_cols[int(dn.max(0).argmax())] if dn is not None and dn.size else ""
    pos = float(np.abs(tg["pos"] - tl["pos"][p]).max())
    glob = float(np.abs(tg["glob"] - tl["glob"]).max()) if tg["glob"].shape == tl["glob"].shape else np.inf
    return dict(disc=disc, num=num, numcol=numcol, pos=pos, glob=glob, max=max(num, pos, glob))


def bridge(dep, X, cols, same_ids, tok_rows, tok_dir):
    """content comparison of the reactions whose new token file differs from deploy's, through the LADDER g2 inputs
    and the deploy Q_parity record (see the module docstring)."""
    Q = pd.read_csv(dep / "Q_parity.csv", dtype={"perm": str}, low_memory=False)
    Q = Q.assign(rxn_id=Q.rxn_id.astype(int)).set_index("rxn_id")
    qj = C3.read_json(dep / "Q_parity.json")
    g2x = C3.G2 / "X182_d1_g2.parquet"
    g2m = C3.REPO / "analysis" / "d1_g2" / "results" / "fu2" / "T_manifest.csv"
    g2tok = C3.G2 / "fu2" / "tokens"
    src = dict(x182_g2_sha256_equals_Q=C3.sha256(g2x) == (qj.get("inputs") or {}).get("x182_g2_sha256"),
               g2_T_manifest_sha256_equals_Q=C3.sha256(g2m) == (qj.get("inputs") or {}).get("g2_T_manifest_sha256"))
    if not all(src.values()):
        return dict(ok=False, sources=src, why="the g2 files differ from the ones deploy Q_parity compared")
    num_cols = list(C3.read_json(C3.REPO / "analysis" / "d1_g2" / "results" / "fu2" / "T_report.json")["num_cols"])
    GM = pd.read_csv(g2m)
    gsha = dict(zip(GM.rxn_id.astype(int), GM.sha256.astype(str)))
    G = pd.read_parquet(g2x)
    G = G.assign(rxn_id=G.rxn_id.astype(int)).set_index("rxn_id")
    Xi = X.assign(rxn_id=X.rxn_id.astype(int)).set_index("rxn_id")
    rows = []
    for r in sorted(set(Xi.index) - set(same_ids)):
        q = Q.loc[r] if r in Q.index else None
        cat = str(q["category"]) if q is not None else None
        rec = dict(rxn_id=int(r), category=cat)
        # 182 columns
        if q is None or r not in G.index or not np.isfinite(float(q["x182_maxdiff"])):
            rec.update(x_kind="none")
        else:
            dv = np.abs(Xi.loc[r, cols].to_numpy(float) - G.loc[r, cols].to_numpy(float))
            d_new, d_old = float(dv.max()), float(q["x182_maxdiff"])
            rec.update(x_new_g2=d_new, x_old_g2=d_old, x_new_g2_col=cols[int(dv.argmax())],
                       x_old_g2_col=str(q.get("x182_maxcol")))
            if cat == "perm_symmetric_map":
                rec.update(x_kind="record", x_ok=bool(abs(d_new - d_old) <= TOL and
                                                      (d_old <= TOL or rec["x_new_g2_col"] == rec["x_old_g2_col"])))
            elif d_old == 0.0:
                rec.update(x_kind="exact", x_value=d_new, x_ok=d_new <= TOL)
            else:
                rec.update(x_kind="bound", x_value=d_new + d_old, x_ok=d_new + d_old <= TOL)
        # token arrays
        if r in tok_rows:
            if q is None or r not in gsha or not np.isfinite(float(q.get("tok_num_maxdiff", np.nan))):
                rec.update(t_kind="none")
            else:
                gp = g2tok / f"{r}.npz"
                if C3.sha256(gp) != gsha[r]:
                    rec.update(t_kind="none", t_why="g2 token sha256 != g2 T_manifest")
                else:
                    with np.load(gp) as zg, np.load(tok_dir / f"{r}.npz") as zl:
                        tg, tl = {k: zg[k] for k in zg.files}, {k: zl[k] for k in zl.files}
                    p = _perm(q, len(tg["Z"]))
                    c = _tok_cmp(tg, tl, p, num_cols)
                    t_old = max(float(q["tok_num_maxdiff"]), float(q["tok_pos_maxdiff"]), float(q["tok_glob_maxdiff"]))
                    old_disc = all(str(q.get(k)) == "True" for k in ("tok_Z_equal", "tok_bin_equal", "tok_edges_equal"))
                    rec.update(t_new_g2=c["max"], t_old_g2=t_old, t_new_disc=c["disc"], t_old_disc=old_disc,
                               t_new_numcol=c["numcol"], t_old_numcol=str(q.get("tok_num_maxcol")))
                    if cat == "perm_symmetric_map":
                        rec.update(t_kind="record", t_ok=bool(c["disc"] == old_disc and abs(c["max"] - t_old) <= TOL))
                    elif t_old == 0.0 and old_disc:
                        rec.update(t_kind="exact", t_value=c["max"], t_ok=bool(c["disc"] and c["max"] <= TOL))
                    else:
                        rec.update(t_kind="bound", t_value=c["max"] + t_old,
                                   t_ok=bool(c["disc"] and old_disc and c["max"] + t_old <= TOL))
        rows.append(rec)
    B = pd.DataFrame(rows)
    C3.write_csv(C3.RES / "V_bridge.csv", B)
    lf_only = {int(x) for x in ((qj.get("lf_only_ok") or {}).get("rxn") or [])}
    none_x = sorted(int(x) for x in B.loc[B.get("x_kind") == "none", "rxn_id"]) if len(B) else []
    none_t = sorted(int(x) for x in B.loc[B.get("t_kind") == "none", "rxn_id"]) if len(B) and "t_kind" in B else []
    none = sorted(set(none_x) | set(none_t))

    def kinds(prefix):
        k = f"{prefix}_kind"
        if not len(B) or k not in B:
            return {}
        out = {}
        for kind, g in B.groupby(k):
            d = dict(n=int(len(g)))
            if f"{prefix}_value" in g and g[f"{prefix}_value"].notna().any():
                d["max_value"] = float(g[f"{prefix}_value"].max())
            if f"{prefix}_ok" in g and g[f"{prefix}_ok"].notna().any():
                d["n_fail"] = int((g[f"{prefix}_ok"] == False).sum())         # noqa: E712
            out[str(kind)] = d
        return out
    fails_x = B.loc[B.get("x_ok", pd.Series(dtype=object)) == False, "rxn_id"].tolist() if "x_ok" in B else []   # noqa: E712
    fails_t = B.loc[B.get("t_ok", pd.Series(dtype=object)) == False, "rxn_id"].tolist() if "t_ok" in B else []   # noqa: E712
    ok = bool(not fails_x and not fails_t and set(none) <= lf_only and len(none) <= NONE_MAX)
    return dict(ok=ok, sources=src, n_reactions=int(len(B)), x182=kinds("x"), tokens=kinds("t"),
                fails_x182=[int(x) for x in fails_x][:50], fails_tokens=[int(x) for x in fails_t][:50],
                not_comparable=none, not_comparable_allowed=sorted(lf_only), tol=TOL, table=str(C3.RES / "V_bridge.csv"))


def main():
    import data_ch3 as DC3                                                  # binds DEP_SCRATCH / DEP_RESULTS
    DL = DC3.DL
    dep = C3.DEPLOY_RES
    dL, dM, dF = (C3.read_json(dep / f) for f in ("L_report.json", "M_report.json", "F_report.json"))
    dT, dK = C3.read_json(dep / "fu2_lf" / "T_report.json"), C3.read_json(dep / "K_report.json")
    rec = dict(spec="docs/specs/CH3_MODEL.md 4", written=C3.now(), checks={}, recorded={})
    rec["recorded"] = dict(F_report_labels_sha256=dF.get("labels_sha256"), F_report_meta_sha256=dF.get("meta_sha256"),
                           T_report_x182_sha256=dT.get("x182_sha256"), T_report_manifest_sha256=dT.get("manifest_sha256"),
                           F_report_n_feat=dF.get("n_feat"), T_report_n_reactions=dT.get("n_reactions"),
                           L_report_done=(dL.get("lf_markers") or {}).get(".done"), M_report_n_records=dM.get("n_records"),
                           K_report_pair_list_sha=dK.get("pair_list_sha"), K_report_n_pairs_P_lf=dK.get("n_pairs_P_lf"))
    spec_vs_rec = dict(labels=dF.get("labels_sha256") == WANT["labels"], meta=dF.get("meta_sha256") == WANT["meta"],
                       x182=dT.get("x182_sha256") == WANT["x182"], manifest=dT.get("manifest_sha256") == WANT["manifest"])
    rec["spec_values_equal_deploy_reports"] = spec_vs_rec
    # regeneration gates
    R = C3.LF_RES
    need = [R / "L_report.json", R / "M_report.json", R / "F_report.json", R / "T_lf_check.json",
            R / "fu2_lf" / "T_report.json", R / "fu2_lf" / "T_manifest.csv"]
    miss = [str(p) for p in need if not p.is_file()]
    if miss:
        C3.not_ready(f"regeneration reports missing: {miss}")
    nL, nM, nF = (C3.read_json(R / f) for f in ("L_report.json", "M_report.json", "F_report.json"))
    tc = C3.read_json(R / "T_lf_check.json")
    gates = dict(L_all_pass=bool(nL.get("all_pass")), M_all_pass=bool(nM.get("all_pass")), F_no_stop=not nF.get("stop"),
                 T_gate_no_stop=not (tc.get("gate") or {}).get("stop", True),
                 T_tokens_no_stop=not (tc.get("tokens") or {}).get("stop", True))
    rec["regeneration_gates"] = gates
    # V-1
    root = C3.LF / "g2lf_final"
    done = sorted(int(p.parent.name) for p in root.glob("*/.done"))
    labels = json.loads((C3.LF / "labels_lf.json").read_text())
    lab_ids = sorted(int(r["rxn_id"]) for r in labels)
    Ls = pd.read_csv(dep / "L_status.csv")
    dep_done = sorted(int(r) for r in Ls.loc[Ls.lf_marker == ".done", "rxn_id"])
    rec["checks"]["V-1"] = dict(n_done=len(done), n_records=len(lab_ids), want=N_DONE,
                                done_equals_deploy_list=done == dep_done, records_equal_done=lab_ids == done,
                                ok=len(done) == N_DONE == len(lab_ids) and done == dep_done and lab_ids == done)
    # V-5 first (the token identity is the content reference of V-4)
    T = pd.read_csv(R / "fu2_lf" / "T_manifest.csv")
    dTm = pd.read_csv(dep / "fu2_lf" / "T_manifest.csv")
    s_man = C3.sha256(R / "fu2_lf" / "T_manifest.csv")
    same_rows = sorted(T.rxn_id.astype(int)) == sorted(dTm.rxn_id.astype(int))
    mm = T.merge(dTm[["rxn_id", "sha256"]], on="rxn_id", how="outer", suffixes=("", "_dep"))
    n_same_file = int((mm.sha256 == mm.sha256_dep).sum())
    tok_dir = C3.LF / "fu2_lf" / "tokens"
    file_now = {int(r): C3.sha256(tok_dir / f"{int(r)}.npz") for r in T.rxn_id}
    files_match_manifest = all(file_now[int(r)] == s for r, s in zip(T.rxn_id, T.sha256))
    v5_sha = s_man == WANT["manifest"]
    same = set(mm.loc[mm.sha256 == mm.sha256_dep, "rxn_id"].astype(int))
    # X182 + content comparison (identical token files, then the g2 bridge for the rest)
    X = pd.read_parquet(C3.LF / "X182_lf.parquet")
    s_x = C3.sha256(C3.LF / "X182_lf.parquet")
    not_ok = {int(k) for k in (dF.get("not_ok") or {})}
    dep_rows = sorted(set(dep_done) - not_ok)
    rows_eq = sorted(int(r) for r in X.rxn_id) == dep_rows
    cols = [c for c in X.columns if c != "rxn_id"]
    Xi = X.assign(rxn_id=X.rxn_id.astype(int)).set_index("rxn_id")
    dmax, n_cmp = 0.0, 0
    for r in sorted(same & set(Xi.index)):
        with np.load(tok_dir / f"{r}.npz") as t:
            g = np.asarray(t["glob"], float)
        dmax = max(dmax, float(np.nanmax(np.abs(Xi.loc[r, cols].to_numpy(float) - g))))
        n_cmp += 1
    need_bridge = not (v5_sha and s_x == WANT["x182"])
    br = bridge(dep, X, cols, same, {int(r) for r in T.rxn_id}, tok_dir) if need_bridge else None
    v5_content = bool(same_rows and files_match_manifest and (n_same_file == len(dTm) or (br or {}).get("ok")))
    rec["bridge_g2"] = br
    rec["checks"]["V-5"] = dict(sha256=s_man, want=WANT["manifest"], sha_equal=v5_sha, n_tokens=int(len(T)), want_n=N_TOK,
                                content=dict(rows_equal=same_rows, n_token_files_same_sha_as_deploy=n_same_file,
                                             n_deploy=int(len(dTm)), files_match_new_manifest=files_match_manifest,
                                             identical_files_max_abs_diff=0.0,
                                             bridge_tokens=(br or {}).get("tokens"), bridge_ok=(br or {}).get("ok")),
                                ok=len(T) == N_TOK and files_match_manifest and (v5_sha or v5_content))
    # V-2, V-3
    s_lab, s_meta = C3.sha256(C3.LF / "labels_lf.json"), C3.sha256(C3.LF / "input_meta_lf.csv")
    meta = pd.read_csv(C3.LF / "input_meta_lf.csv")
    meta_rows = sorted(int(r) for r in meta.rxn_id) == dep_done
    rec["checks"]["V-2"] = dict(sha256=s_lab, want=WANT["labels"], sha_equal=s_lab == WANT["labels"],
                                content=dict(rows_equal_deploy_done=lab_ids == dep_done),
                                ok=s_lab == WANT["labels"] or (lab_ids == dep_done and v5_content))
    rec["checks"]["V-3"] = dict(sha256=s_meta, want=WANT["meta"], sha_equal=s_meta == WANT["meta"],
                                content=dict(rows_equal_deploy_done=meta_rows),
                                ok=s_meta == WANT["meta"] or (meta_rows and v5_content))
    # V-4
    content = dict(rows_equal_deploy=rows_eq, n_deploy_rows=len(dep_rows), n_rows_compared_via_identical_tokens=n_cmp,
                   identical_tokens_max_abs_diff_182=dmax, n_rows_via_bridge=(br or {}).get("n_reactions"),
                   bridge_x182=(br or {}).get("x182"), bridge_ok=(br or {}).get("ok"))
    v4_content = bool(rows_eq and dmax <= TOL and (n_cmp == len(Xi) or (br or {}).get("ok")))
    rec["checks"]["V-4"] = dict(sha256=s_x, want=WANT["x182"], sha_equal=s_x == WANT["x182"], n_rows=int(len(X)),
                                want_n=N_FEAT, columns=len(cols), content=content,
                                ok=len(X) == N_FEAT and len(cols) == 182 and (s_x == WANT["x182"] or v4_content))
    # V-6
    D = DL.DeployData("lf")
    rec["checks"]["V-6"] = dict(pair_list_sha=D.pair_list_sha, want=dK.get("pair_list_sha"),
                                n_pairs_P_lf=len(D.pair_list), want_n=dK.get("n_pairs_P_lf"),
                                ok=D.pair_list_sha == dK.get("pair_list_sha") and len(D.pair_list) == dK.get("n_pairs_P_lf"))
    rec["inputs_sha256"] = dict(D.sha)
    bad = [k for k, v in rec["checks"].items() if not v["ok"]] + [k for k, v in gates.items() if not v]
    rec["all_pass"] = not bad
    rec["failed"] = bad
    rec["any_content_comparison_used"] = any(not v.get("sha_equal", True) for v in rec["checks"].values())
    C3.write_json(C3.RES / "V_inputs.json", rec)
    print(json.dumps(dict(all_pass=rec["all_pass"], failed=bad, gates=gates,
                          **{k: dict(ok=v["ok"], sha_equal=v.get("sha_equal")) for k, v in rec["checks"].items()}),
                     indent=1))
    return 0 if not bad else 3


if __name__ == "__main__":
    sys.exit(main())
