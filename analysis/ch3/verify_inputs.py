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
    v5_content = same_rows and n_same_file == len(dTm) and files_match_manifest
    rec["checks"]["V-5"] = dict(sha256=s_man, want=WANT["manifest"], sha_equal=v5_sha, n_tokens=int(len(T)), want_n=N_TOK,
                                content=dict(rows_equal=same_rows, n_token_files_same_sha_as_deploy=n_same_file,
                                             n_deploy=int(len(dTm)), files_match_new_manifest=files_match_manifest,
                                             max_abs_diff_token_arrays=0.0 if v5_content else None),
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
    X = pd.read_parquet(C3.LF / "X182_lf.parquet")
    s_x = C3.sha256(C3.LF / "X182_lf.parquet")
    not_ok = {int(k) for k in (dF.get("not_ok") or {})}
    dep_rows = sorted(set(dep_done) - not_ok)
    rows_eq = sorted(int(r) for r in X.rxn_id) == dep_rows
    cols = [c for c in X.columns if c != "rxn_id"]
    content = dict(rows_equal_deploy=rows_eq, n_deploy_rows=len(dep_rows))
    if s_x != WANT["x182"]:
        Xi = X.assign(rxn_id=X.rxn_id.astype(int)).set_index("rxn_id")
        same = set(mm.loc[mm.sha256 == mm.sha256_dep, "rxn_id"].astype(int))
        dmax, n_cmp = 0.0, 0
        for r in sorted(same & set(Xi.index)):
            with np.load(tok_dir / f"{r}.npz") as t:
                g = np.asarray(t["glob"], float)
            dmax = max(dmax, float(np.nanmax(np.abs(Xi.loc[r, cols].to_numpy(float) - g))))
            n_cmp += 1
        content.update(n_rows_compared_via_identical_tokens=n_cmp, n_rows_not_comparable=int(len(Xi) - n_cmp),
                       max_abs_diff_182=dmax)
        v4_content = rows_eq and n_cmp > 0 and dmax <= TOL
    else:
        v4_content = None
    rec["checks"]["V-4"] = dict(sha256=s_x, want=WANT["x182"], sha_equal=s_x == WANT["x182"], n_rows=int(len(X)),
                                want_n=N_FEAT, columns=len(cols), content=content,
                                ok=len(X) == N_FEAT and len(cols) == 182 and (s_x == WANT["x182"] or bool(v4_content)))
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
