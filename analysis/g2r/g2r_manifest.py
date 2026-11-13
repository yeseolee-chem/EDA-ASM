#!/usr/bin/env python3
"""D1-G2R dataset manifest: one JSON describing the new g2 ($G2R) after Phase D (user request 2026-11-13).

    python analysis/g2r/g2r_manifest.py          env_d1fu2 (pandas); py_g2r.slurm

Writes $G2R/g2r.json and the identical analysis/g2r/results/g2r.json:
  header    roots, the environment variables that point deploy / d1_g2 code at the new g2, counts (targets, groups,
            usable, missing by stage), sha256 of the model inputs, label sources (not copied: join on rxn_id / job_id),
            the D-5 tolerance patch, the deleted old roots, the git commit of the results.
  reactions one record per target (9,252, X_targets.csv order): identity and label flags, g2 group / marker, usable /
            missing stage / reason (U_usable.csv), repair outcome, file paths in $G2R (job folder; for L .done the
            g2lf_final files and the token), and the g2 geometry metadata of usable reactions (labels_lf.json record
            fields, input_meta_lf.csv A_idx, lowest imaginary frequency and forming-bond share of the L result.json).
Every listed path is checked to exist; exit 3 if one is missing. Reads only; writes the two JSON files.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g2r_common as C  # noqa: E402

LF = C.G2R_LF
FINAL = LF / "g2lf_final"
TOK = LF / "fu2_lf" / "tokens"
GEO_KEYS = ("n_atoms", "n_f1", "n_f2", "formed_pairs_ts", "flag_async", "flag_foreign_bond", "alt_used", "charge1",
            "charge2", "role1", "role2")


def rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def b(v):
    return {"True": True, "False": False, "": None}.get(str(v), v)


def main() -> int:
    X = rows(C.RES / "X_targets.csv")
    U = {r["job_id"]: r for r in rows(C.RES / "U_usable.csv")}
    R = {r["job_id"]: r for r in rows(C.RES / "R_status.csv")}
    lab = {int(r["rxn_id"]): r for r in json.loads((LF / "labels_lf.json").read_text())}
    meta = {int(r["rxn_id"]): r["A_idx"] for r in rows(LF / "input_meta_lf.csv")}
    missing, recs = [], []

    def path(p: Path):
        if not p.exists():
            missing.append(str(p))
        return str(p)

    for x in X:
        rid, j = int(x["rxn_id"]), x["job_id"]
        u = U[j]
        rec = dict(rxn_id=rid, job_id=j, dipole=x["dipole"], skeleton=x["skeleton"], series=x["series"] or None,
                   attach=x["attach"] or None, flag_async_label=b(x["flag_async"]), ex_orig_gt20=b(x["ex_orig_gt20"]),
                   g2_group=x["group"], g2_marker=x["g2_marker"], usable=b(u["usable"]),
                   missing_stage=u["missing_stage"] or None, reason=u["reason"] or None,
                   repair=(dict(status=R[j]["new_status"], reason=R[j]["new_reason"] or None,
                                orig_reason=R[j]["orig_reason"]) if j in R else None),
                   job_dir=path(C.G2R_JOBS / j))
        d = FINAL / str(rid)
        if (d / ".done").is_file():
            res = C.read_json(d / "result.json")
            imag = [float(v) for v in (res.get("imag_freqs_cm") or [])]
            rec.update(lf_dir=path(d), ts_xyz=path(d / "ts.xyz"), rel1_xyz=path(d / "rel1.xyz"),
                       rel2_xyz=path(d / "rel2.xyz"), ts_hess=path(d / "ts.hess"),
                       imag1_cm=min(imag) if imag else None, n_imag=res.get("n_imag"),
                       imag_mode_forming_share=res.get("imag_mode_forming_share"))
        if rec["usable"]:
            g = lab[rid]
            rec.update(token=path(TOK / f"{rid}.npz"), A_idx=meta[rid],
                       **{("flag_async_g2" if k == "flag_async" else k): g[k] for k in GEO_KEYS})
        recs.append(rec)
    n_use = sum(r["usable"] for r in recs)
    by_stage, by_group = {}, {}
    for r in recs:
        by_group[r["g2_group"]] = by_group.get(r["g2_group"], 0) + 1
        if not r["usable"]:
            by_stage[r["missing_stage"]] = by_stage.get(r["missing_stage"], 0) + 1
    files = {"labels_lf.json": LF / "labels_lf.json", "input_meta_lf.csv": LF / "input_meta_lf.csv",
             "X182_lf.parquet": LF / "X182_lf.parquet",
             "xtb_features_ext_g1_lf.parquet": LF / "F" / "xtb" / "xtb_features_ext_g1_lf.parquet",
             "xtb_features_g1_lf.parquet": LF / "F" / "xtb" / "xtb_features_g1_lf.parquet",
             "T_manifest.csv": C.RES / "lf" / "fu2_lf" / "T_manifest.csv",
             "Q_manifest.csv": C.RES / "lf" / "fu2_lf" / "Q_manifest.csv",
             "C_manifest.csv.gz": C.G2R_X / "C_manifest.csv.gz"}
    head = subprocess.run(["git", "-C", str(C.REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    lab_src = {"labels_d1.json (D1 DFT labels, original references)": "/gpfs/tmp_cpu2/yeseo1ee/d1/d1_run/results/labels_d1.json",
               "labels_d1_cr2.parquet (corrected references, committed)": str(C.CR2_LABELS),
               "labels_d1_cr2.json (corrected references, LADDER ground truth)": "/gpfs/tmp_cpu2/yeseo1ee/d1_ladder/labels_d1_cr2.json"}
    header = dict(
        name="D1-G2R g2 (xTB-level autodE TS from SMILES, label-free model inputs)",
        spec="docs/specs/D1G2R.md", branch="g2r", results_commit=head,
        report="analysis/g2r/results/RUN_REPORT_G2R.md", written=time.strftime("%Y-%m-%d %H:%M:%S"),
        roots=dict(g2r=str(C.G2R), ts_sources=str(C.G2R_JOBS), model_inputs=str(LF), lf_final=str(FINAL),
                   tokens=str(TOK), results="analysis/g2r/results (lf/ = deploy reports)"),
        use_with=dict(DEP_G2_JOBS=str(C.G2R_JOBS), DEP_SCRATCH=str(LF), D1G2_SCRATCH=str(C.G2R),
                      DEP_RESULTS="<worktree>/analysis/g2r/results/lf",
                      note="deploy / d1_g2 defaults point at the deleted $G2 and fail; pass these"),
        counts=dict(targets=len(recs), groups=by_group, usable=n_use, missing_by_stage=by_stage,
                    repaired_ok=sum(1 for r in recs if r["repair"] and r["repair"]["status"] == "ok")),
        target_rule="D1 non-D'' labels 9,266 - flag_short_forming 9 - flag_regio_ambiguous 5 (D-1)",
        usable_rule="target and L .done and a finite X182_lf row and a token",
        files={k: dict(path=str(p), sha256=C.sha256(p)) for k, p in files.items()},
        labels=dict(included=False, join_keys=["rxn_id", "job_id"],
                    sources={k: dict(path=v, sha256=C.sha256(v) if Path(v).is_file() else None) for k, v in lab_src.items()},
                    note="labels_lf.json holds the g2 metadata with every DFT key NaN; label values come from the label "
                         "files above"),
        geometry_note="ts.xyz / rel1.xyz / rel2.xyz / A_idx / formed_pairs_ts / n_f1 are in the g2 (autodE) atom order, "
                      "not the DFT label order",
        tolerances=dict(GATE_E_EH=1e-4, TOL_DIP_AU=1e-2, how="analysis/g2r/site_g2r/sitecustomize.py with "
                        "G2R_TOL_PATCH=1 (jobs/g_g2r.slurm); every other tolerance unchanged (decision D-5)"),
        deleted_old=dict(paths=["/gpfs/tmp_cpu2/yeseo1ee/d1_g2", "/gpfs/tmp_cpu2/yeseo1ee/ch3/lf"],
                         record="analysis/g2r/results/D_report.json"),
        n_missing_paths=len(missing), missing_paths=missing[:20])
    out = dict(header=header, reactions=recs)
    text = json.dumps(out, indent=1, allow_nan=False) + "\n"
    C.write_atomic(C.G2R / "g2r.json", text)
    C.write_atomic(C.RES / "g2r.json", text)
    print(json.dumps(dict(targets=len(recs), usable=n_use, groups=by_group, missing_by_stage=by_stage,
                          missing_paths=len(missing), sha256=C.sha256(C.G2R / "g2r.json"),
                          bytes=os.path.getsize(C.G2R / "g2r.json")), indent=1))
    return 0 if not missing else 3


if __name__ == "__main__":
    sys.exit(main())
