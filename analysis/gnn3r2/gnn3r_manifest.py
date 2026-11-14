#!/usr/bin/env python3
"""GNN3R2 data set manifest (docs/specs/GNN3R2.md 10-2; copied from the GNN3R manifest): one JSON describing the fixed
GNN3R2 data set and model.

    python analysis/gnn3r2/gnn3r_manifest.py         env_d1fu2 (gnn3/jobs/run.slurm), CPU; after Phase W

Writes results/gnn3r2.json and the identical $GR2/gnn3r2.json (the form of analysis/gnn3r/results/gnn3r.json):
  header     spec, branch, results commit, written; roots ($GR, $G2R, cv, final); files (path + sha256 of F_pairs_r.csv,
             S_folds_r.csv, S_splits_r.json, the 5 split files, B_label_report.json, rowattr_r.pkl, U_usable.csv, X182,
             T_manifest, labels_d1_cr2.json); counts; rules (decisions 0-2); model (final setting, the .pt files);
             use_with (how to load the data set again without rebuilding B)
  reactions  one record per data-set reaction (results/B_reactions.csv order)
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

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gr_common as C  # noqa: E402

RULES = {
    "D-1 targets": "D1 non-D'' labels 9,266 - flag_short_forming 9 - flag_regio_ambiguous 5 = 9,252",
    "D-2 inputs": "g2r ($G2R/lf); usable = analysis/g2r/results/U_usable.csv usable True (8,617)",
    "D-3 labels": "cr2 ($LAD/labels_d1_cr2.json); dd_t = y_t(a) - y_t(b) for the 10 labels of d1pair_common.D1_TCOL",
    "D-4 row exclusion": "none by label value; no in_P_feat filter",
    "D-5 B' c3": "Bp need rows: C4's methyl is the reference substituent (rowattr_r.c3_orientation_bp)",
    "D-6 pair rules": "d1_pairs.build / d1_anchor_pairs.build (unchanged); F = (dipole, skeleton, series, cfg, orient, "
                      "face, c3)",
    "D-7 anchor b": "Xb Me < CO2Me < tBu < CHO < Ts < NMe2 < rest, then p < m < o < direct, Xb, job_id",
    "D-8 P-4": "pairs with v2 mismatch dropped (d1pair_common.P4_DECISION)",
    "D-9 folds": "T1 core -> fold of analysis/d1_pair/results/S_folds.csv; -1 always train; inner = first "
                 "GroupKFold(4) by core",
    "D-10 sets": "in_P_main = in_P_sens = reference pairs; in_T_main = every pair",
    "D-11 model": "GNN3 unchanged (gnn3/gnn3/constants.py): grid {4,6} layers x {64,128} width, 5 seeds, RTX 3090",
    "D-12 c3": "c3_core for every need row with a C3 substituent: X cut back to CH3, canonical atom order, 5 seeds "
               "(7, 11, 23, 42, 97) of EmbedMolecule + MMFF must agree; changes T10104, T10116 cis -> trans",
    "D-13 old GNN3R": "deleted after GNN3R2 passed and was pushed (Phase X)",
}


def rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    missing = []

    def entry(p: Path) -> dict:
        if not p.exists():
            missing.append(str(p))
            return dict(path=str(p), sha256=None)
        return dict(path=str(p), sha256=C.sha256(p) if p.is_file() else None)

    files = {"F_pairs_r.csv": C.PAIRS_R, "S_folds_r.csv": C.S_FOLDS_R, "S_splits_r.json": C.S_SPLITS_R,
             **{f"T1__f{k}.json": C.SPLIT_DIR / f"T1__f{k}.json" for k in range(5)},
             "B_label_report.json": C.LABEL_REPORT, "rowattr_r.pkl": C.ROWATTR_R, "U_usable.csv": C.USABLE,
             "X182_lf.parquet": C.X182, "T_manifest.csv": C.TOKEN_REPORTS / "T_manifest.csv",
             "labels_d1_cr2.json": C.LABELS_CR2}
    b = C.read_json(C.RES / "B_report.json")
    s = C.read_json(C.S_SPLITS_R)
    k = C.read_json(C.RES / "cv" / "K_check.json")
    w = C.read_json(C.RES / "final" / "W_check.json")
    nets = sorted(str(p) for p in C.FINAL.glob("gnn3_final__seed*.pt"))
    cv_nets = sorted(str(p) for p in C.CV.glob("gnn3_fold*__seed*.pt"))
    head = subprocess.run(["git", "-C", str(C.REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    g = C.read_json(C.RES / "G_report.json")
    header = dict(
        name="GNN3R2 data set (g2r inputs, rebuilt D1PAIR pairs, c3 by the CH3-core rule D-12, cr2 labels) and GNN3 model",
        spec="docs/specs/GNN3R2.md", branch="gnn3r2", results_commit=head,
        report="analysis/gnn3r2/results/RUN_REPORT_GNN3R2.md", written=time.strftime("%Y-%m-%d %H:%M:%S"),
        roots=dict(gnn3r2=str(C.GR), g2r=str(C.G2R), cv=str(C.CV), final=str(C.FINAL),
                   results="analysis/gnn3r2/results", gnn3r_reference="analysis/gnn3r2/ref_gnn3r"),
        changes_vs_gnn3r=dict(c3_changed={j: list(v) for j, v in C.EXPECT_C3_CHANGED.items()},
                              pairs_added=g["gates"]["G-2"]["added"], pairs_removed=g["gates"]["G-2"]["removed"],
                              rows_changed=g["gates"]["G-3"]["jobs"]),
        files={n: entry(p) for n, p in files.items()},
        counts=dict(targets=C.N_TARGETS, usable=C.N_USABLE, reactions=b["counts"]["reactions"],
                    pairs=b["counts"]["pairs"], by_ref_kind=b["counts"]["by_ref_kind"],
                    distinct_fam=b["counts"]["distinct_fam"], cores=b["counts"]["cores"],
                    by_fold_T1=b["counts"]["by_fold_T1"], row_status=b["row_status"]["counts"],
                    p4_dropped_pairs=b["p4"]["n_dropped"], test_sizes=s["test_sizes"]),
        rules=RULES,
        model=dict(name="GNN3", setting=w.get("setting"), final_networks=nets, cv_networks=cv_nets,
                   cv_chosen={str(f): v["chosen"] for f, v in k["gates"]["K-2"]["per_fold"].items()},
                   gpu=sorted({v.get("gpu") for v in k["gates"]["K-2"]["per_fold"].values()})),
        use_with=dict(python="analysis/gnn3r2/gr_common.dataset() (gnn3.data.PairDataset on these files)",
                      note="retrain from F_pairs_r.csv (sha256 in B_label_report.json); do not rebuild Phase B"),
        n_missing_paths=0, missing_paths=[])
    for p in nets + cv_nets:
        if not Path(p).is_file():
            missing.append(p)
    recs = [dict(job_id=r["job_id"], rxn_id=int(r["rxn_id"]), core=r["core"], role=r["role"],
                 n_pairs=int(r["n_pairs"]), fam=r["fam"]) for r in rows(C.RES / "B_reactions.csv")]
    header.update(n_missing_paths=len(missing), missing_paths=missing[:20])
    text = json.dumps(dict(header=header, reactions=recs), indent=1, allow_nan=False) + "\n"
    C.write_atomic(C.GR / "gnn3r2.json", text)
    C.write_atomic(C.RES / "gnn3r2.json", text)
    print(json.dumps(dict(reactions=len(recs), pairs=header["counts"]["pairs"], missing_paths=len(missing),
                          sha256=C.sha256(C.RES / "gnn3r2.json"), bytes=os.path.getsize(C.RES / "gnn3r2.json")), indent=1))
    return 0 if not missing else 3


if __name__ == "__main__":
    sys.exit(main())
