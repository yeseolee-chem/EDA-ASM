#!/usr/bin/env python3
"""d1c_prep.py — SPEC_D1_ALL Phase C, step 1: the D1 inputs of the unchanged rev 5 feature scripts.

  python d1c_prep.py LABELS GEOM_DIR MANIFEST [--scan ARCHIVE_SCAN_CSV]

  LABELS    D1_build/d1_run/results/labels_d1.json (all rows status ok, kind D1; the D0 label keys)
  GEOM_DIR  D1_build/d1_run/results/geometries/<job_id>/{ts,ref_dipole,ref_dipolarophile}.xyz
  MANIFEST  D1_build/d1_run/d1_manifest.csv (job_id -> mapped_smiles, the atom-mapped reaction SMILES)
  --scan    archive_scan.csv of d1_archive_scan.py (a_idx_tag = the atoms tagged (1) in sp/eda.inp, i.e. the EDA
            partition of the label); optional cross-check of the A_idx below

Writes (rev5_common D1 constants; read by g1_geom.py / xtb_slice.py / ext_features.py through phase_c_env.sh):
  labels_g1in.json   the LABELS records unchanged (= ESPLEY_LABELS); asserted: status ok, unique int rxn_id, no D0 id
  prof/<rxn_id>      symlink -> GEOM_DIR/<job_id> (= ESPLEY_PROF; label ts_file / rel1_file / rel2_file live there)
  input_meta.csv     rxn_id, job_id, A_idx (= ESPLEY_META): fragmenter.partition(ts, rel1, rel2) of the DFT
                     geometries (Si-added fragmenter, as the label), A = the rel1 (role1 = dipole) atoms, 0-based
  smiles.csv         rxn_id, rxn_smiles (= ESPLEY_SMILES; the manifest mapped_smiles)
  prep.json          input sha256, counts, partition failures, A_idx disagreements with --scan
A row whose DFT partition fails or whose |A| != n_f1 keeps no A_idx (it fails G1 as partition_mismatch and is
counted in gate C (ii)). Idempotent: a rerun with the same inputs rewrites the same files; a rerun with other
inputs is refused (exit 2) unless the D1 feature root holds no G1 result yet.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
XR = HERE.parent / "espley_xtb_repro"
sys.path.insert(0, str(XR))
import fragmenter as fr  # noqa: E402
import rev5_common as R5  # noqa: E402

D0_IDS_MAX = 999_999                     # D0 rxn_id < 1e6; D1 rxn_id = 1,000,000 + n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels", type=Path)
    ap.add_argument("geom", type=Path)
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--scan", type=Path)
    a = ap.parse_args()
    root = R5.D1_FEAT_ROOT
    root.mkdir(parents=True, exist_ok=True)
    fp = {"labels": str(a.labels.resolve()), "labels_sha256": R5.sha256(a.labels),
          "manifest": str(a.manifest.resolve()), "manifest_sha256": R5.sha256(a.manifest),
          "geometries": str(a.geom.resolve())}
    old = root / "prep.json"
    if old.is_file():
        prev = json.loads(old.read_text())
        same = all(prev.get("inputs", {}).get(k) == v for k, v in fp.items())
        if not same and any(R5.D1_G1_ROOT.glob("*/.done")):
            sys.exit(f"STOP: {old} was made from other inputs and G1 results exist under {R5.D1_G1_ROOT}: "
                     f"{prev.get('inputs')} vs {fp}. Move {root} away to start over.")

    labels = json.loads(a.labels.read_text())
    bad = [d.get("job_id") for d in labels if d.get("status") != "ok"]
    if bad:
        sys.exit(f"STOP: {len(bad)} label rows are not status ok: {bad[:10]}")
    rids = [int(d["rxn_id"]) for d in labels]
    if len(set(rids)) != len(rids) or min(rids) <= D0_IDS_MAX:
        sys.exit("STOP: rxn_id not unique or a D0-range id among the D1 labels")
    man = pd.read_csv(a.manifest, dtype=str).set_index("job_id")
    scan = None
    if a.scan is not None and a.scan.is_file():
        scan = pd.read_csv(a.scan, dtype=str).set_index("job_id")

    prof = root / "prof"
    prof.mkdir(exist_ok=True)
    meta, smi, part_fail, scan_diff, n_scan = [], [], [], [], 0
    for d in labels:
        rid, jid = int(d["rxn_id"]), d["job_id"]
        src = (a.geom / jid).resolve()
        for k in ("ts_file", "rel1_file", "rel2_file"):
            if not (src / d[k]).is_file():
                sys.exit(f"STOP: {src / d[k]} missing")
        link = prof / str(rid)
        if link.is_symlink() or link.exists():
            if Path(os.readlink(link)) != src:
                link.unlink()
        if not link.is_symlink():
            os.symlink(src, link)
        smi.append(dict(rxn_id=rid, job_id=jid, rxn_smiles=man.loc[jid, "mapped_smiles"]))
        ts, r1, r2 = (fr.read_xyz(src / d[k]) for k in ("ts_file", "rel1_file", "rel2_file"))
        P = fr.partition(ts, r1, r2)
        a_idx = sorted(int(i) for i in P.A) if P.status == "ok" else None
        if a_idx is not None and len(a_idx) != int(d["n_f1"]):
            part_fail.append(dict(job_id=jid, reason=f"|A| {len(a_idx)} != n_f1 {d['n_f1']}")); a_idx = None
        elif a_idx is None:
            part_fail.append(dict(job_id=jid, reason=f"partition {P.status}"))
        if scan is not None and jid in scan.index and isinstance(scan.loc[jid, "a_idx_tag"], str):
            n_scan += 1
            tag = sorted(int(x) for x in scan.loc[jid, "a_idx_tag"].split())
            if tag != a_idx:
                scan_diff.append(dict(job_id=jid, a_idx_partition=a_idx, a_idx_tag=tag))
        if a_idx is not None:
            meta.append(dict(rxn_id=rid, job_id=jid, A_idx=" ".join(map(str, a_idx))))

    R5.write_atomic(root / "labels_g1in.json", json.dumps(labels))
    R5.write_atomic(root / "input_meta.csv", lambda t: pd.DataFrame(meta).to_csv(t, index=False))
    R5.write_atomic(root / "smiles.csv", lambda t: pd.DataFrame(smi).to_csv(t, index=False))
    rec = dict(inputs=fp, n_labels=len(labels), n_meta=len(meta), n_partition_fail=len(part_fail),
               partition_fail=part_fail, scan=str(a.scan) if scan is not None else None, n_scan_checked=n_scan,
               n_scan_disagree=len(scan_diff), scan_disagree=scan_diff[:50])
    R5.write_json(old, rec)
    print(json.dumps({k: v for k, v in rec.items() if k not in ("partition_fail", "scan_disagree")}, indent=1))
    if scan_diff:
        print(f"WARNING: {len(scan_diff)} rows: DFT partition != EDA tag partition (first: {scan_diff[:3]})")


if __name__ == "__main__":
    main()
