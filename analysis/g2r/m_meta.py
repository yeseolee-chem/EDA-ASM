#!/usr/bin/env python3
"""D1-G2R Phase M (docs/specs/D1G2R.md 7) when repaired reactions reached L .done: deploy meta_lf.cmd_run with only its
L precondition replaced (recorded decision). D1 run env (imports analysis/deploy/meta_lf -> finalize_lf -> g2_autode).

    python analysis/g2r/m_meta.py          jobs/cpu_g2r.slurm with --export=ALL,DEP_SCRATCH=$G2R/lf,
                                           DEP_RESULTS=<wt>/analysis/g2r/results/lf,DEP_G2_JOBS=$G2R/jobs

Why: deploy `meta_lf.py run --jobs <ids>` requires results/lf/L_report.json to have exactly that subset (meta_lf.py:131),
and the deploy L report can only judge reuse reactions (repaired ones fail its L-1 / L-2 against $G2/g2_final, spec L-K),
so `--jobs reuse_ids.csv` would leave the repaired .done folders without records and feat_lf.run_slice would STOP
(labels must cover every .done folder, feat_lf.py:182-190). If no repaired reaction is .done, the unchanged
`meta_lf.py run --jobs $G2R/x/reuse_ids.csv` is used instead.
Precondition here: L_report.json complete + all_pass with subset = sorted reuse job_ids, results/L_check.json all_pass
(L-1, L-R; every lf_ids reaction has a marker), reuse .done set = L_status.csv .done set. Then meta_lf.build (unchanged)
on every .done folder of lf_ids, and meta_lf.cmd_run's checks / outputs / M_report.json (copied, meta_lf.py:144-199) with
subset = sorted lf_ids job_ids. Exit 0 / 2 not ready / 3 STOP.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "deploy"))
import meta_lf as ML  # noqa: E402

C, FL = ML.C, ML.FL
G2R_RES = HERE / "results"
G2R_X = Path("/gpfs/tmp_cpu2/yeseo1ee/g2r/x")


def ids(name):
    return pd.read_csv(G2R_X / name, dtype=str).job_id.tolist()


def main():
    reuse, rep_ok, lf = ids("reuse_ids.csv"), ids("repaired_ok_ids.csv"), ids("lf_ids.csv")
    if sorted(lf) != sorted(set(reuse) | set(rep_ok)):
        C.stop("lf_ids.csv != reuse_ids.csv + repaired_ok_ids.csv")
    lrep_p = C.RES / "L_report.json"
    lrep = C.read_json(lrep_p) if lrep_p.is_file() else None
    if not lrep or not lrep.get("complete") or not lrep.get("all_pass") or lrep.get("subset") != sorted(reuse):
        C.not_ready(f"{lrep_p}: missing, incomplete, not passed or subset != sorted reuse job_ids")
    lchk_p = G2R_RES / "L_check.json"
    lchk = C.read_json(lchk_p) if lchk_p.is_file() else None
    if not lchk or not lchk.get("all_pass"):
        C.not_ready(f"{lchk_p}: missing or not all_pass")
    tg = FL.load_targets(lf)
    rid_of = {t["job_id"]: int(t["rxn_id"]) for t in tg}
    tg_rid = set(rid_of.values())
    rids = sorted(int(p.parent.name) for p in C.LF_FINAL.glob("*/.done")
                  if p.parent.name.isdigit() and int(p.parent.name) in tg_rid)
    reuse_rid, rep_rid = {rid_of[j] for j in reuse}, {rid_of[j] for j in rep_ok}
    st = pd.read_csv(C.RES / "L_status.csv")
    st_done = sorted(int(r) for r in st.rxn_id[st.lf_marker == ".done"])
    if sorted(r for r in rids if r in reuse_rid) != st_done:
        C.not_ready(f"{C.LF_FINAL} reuse .done set != L_status.csv .done set: rerun report")
    rep_done = sorted(r for r in rids if r in rep_rid)
    man = FL.load_manifest()
    recs, meta, charges, prob, cross = ML.build(C.LF_FINAL, man, rids)
    # ---- meta_lf.cmd_run l.144-199, unchanged except subset / the extra fields
    Q = pd.DataFrame(charges)
    nz = Q[Q.nonzero] if len(Q) else Q
    key_missing = sorted({k for r in recs for k in ML.READ_KEYS if k not in r})
    fmt_bad = [r["rxn_id"] for r in recs if not ML.FP_RE.match(r["formed_pairs_ts"])]
    s_g2 = ML.schema_vs(ML.G2_LABELS, recs)
    s_d0 = ML.schema_vs(ML.D0_LABELS, recs)
    cmp = {}
    if isinstance(s_g2, tuple):
        s_g2, g2lab = s_g2
        L = {int(d["rxn_id"]): d for d in g2lab}
        both = [r for r in recs if r["rxn_id"] in L]
        cmp = dict(n_both=len(both), n_atoms_equal=sum(r["n_atoms"] == int(L[r["rxn_id"]]["n_atoms"]) for r in both),
                   n_f1_equal=sum(r["n_f1"] == int(L[r["rxn_id"]]["n_f1"]) for r in both),
                   formed_pairs_ts_equal=sum(r["formed_pairs_ts"] == L[r["rxn_id"]]["formed_pairs_ts"] for r in both),
                   alt_used_equal=sum(r["alt_used"] == int(L[r["rxn_id"]]["alt_used"]) for r in both),
                   note="label values (DFT TS, label atom order) vs lf (G2 TS, autodE order): information only; "
                        "formed_pairs_ts differs where G2 was re-mapped, alt_used is the G2 (xtb) reference choice")
        del g2lab, L
    if isinstance(s_d0, tuple):
        s_d0 = s_d0[0]
    rep = dict(spec="docs/specs/D1G2R.md 7 (deploy meta_lf.cmd_run via analysis/g2r/m_meta.py)", lf_root=str(C.LF_FINAL),
               subset=sorted(lf), l_report_subset="reuse_ids.csv (sorted)", n_reuse=len(reuse),
               n_repaired_ok=len(rep_ok), n_repaired_done=len(rep_done), repaired_done_rxn_ids=rep_done,
               n_done_folders=len(rids), n_records=len(recs), n_problems=len(prob), problems=prob[:50],
               n_charge_nonzero=len(nz), charge_nonzero=nz.to_dict("records")[:50] if len(nz) else [],
               charge_sources=["manifest charge", "RDKit formal charge of manifest dipole_smiles / dipolarophile_smiles",
                               "ts_result.json charge_dipole / charge_dipolarophile"],
               read_keys=list(ML.READ_KEYS), read_keys_missing=key_missing, dft_keys_nan=list(ML.DFT_KEYS),
               n_formed_pairs_ts_format_bad=len(fmt_bad), formed_pairs_ts_format_bad=fmt_bad[:20],
               n_pairs_not_across_partition=sum(not x for x in cross),
               schema_vs_labels_d1_g2=s_g2, schema_vs_labels_all=s_d0, values_vs_labels_d1_g2=cmp,
               l_report_sha256=C.sha256(lrep_p), l_check_sha256=C.sha256(lchk_p),
               manifest_sha256=C.sha256(C.D1_MANIFEST), meta_lf_sha256=C.sha256(Path(ML.__file__)),
               m_meta_sha256=C.sha256(Path(__file__)), written=time.strftime("%F %T"))
    stop = []
    if len(nz):
        stop.append(f"{len(nz)} reactions with a reactant charge != 0")
    if prob:
        stop.append(f"{len(prob)} .done folders whose record cannot be built consistently")
    if key_missing or fmt_bad:
        stop.append(f"record keys missing {key_missing} / formed_pairs_ts format bad {len(fmt_bad)}")
    rep["stop"] = stop
    rep["all_pass"] = not stop
    if not stop:
        recs.sort(key=lambda r: r["rxn_id"])
        C.write_json(C.LABELS_LF, recs)
        C.write_csv(C.META_LF, pd.DataFrame(meta, columns=["rxn_id", "A_idx"]).sort_values("rxn_id"))
        back = json.loads(C.LABELS_LF.read_text())
        rep["nan_roundtrip_ok"] = all(isinstance(b[k], float) and math.isnan(b[k]) for b in back[:50] for k in ML.DFT_KEYS)
        rep["outputs"] = {str(C.LABELS_LF): C.sha256(C.LABELS_LF), str(C.META_LF): C.sha256(C.META_LF)}
        if not rep["nan_roundtrip_ok"]:
            rep["stop"] = stop = ["DFT keys do not read back as NaN"]
            rep["all_pass"] = False
    C.write_json(C.RES / "M_report.json", rep)
    print(json.dumps({k: rep[k] for k in ("n_done_folders", "n_records", "n_repaired_done", "n_problems",
                                          "n_charge_nonzero", "read_keys_missing", "n_pairs_not_across_partition",
                                          "all_pass", "stop")}), flush=True)
    if stop:
        C.stop("; ".join(stop))
    return 0


if __name__ == "__main__":
    if sys.argv[1:]:
        raise SystemExit(__doc__)
    sys.exit(main())
