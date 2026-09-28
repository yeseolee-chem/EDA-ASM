#!/usr/bin/env python3
"""Stage 3 for the SMD relabel: run stage3_parse.parse_one_rxn on SMD outputs.

The SMD rerun (2026-09) replaced eda.out / eda_frag1.out / eda_frag2.out; they
live in the relabel scratch inputs/. The standalone distorted-fragment SPEs
(frag{1,2}_dist.out) and the relaxed references (frag{1,2}_rel.out) are
unchanged and stay in label_true/work/inputs/. A per-rxn symlink view points
stage3's fixed file names at the right source, so the canonical parser runs
unchanged (same gates, same channel convention).

Outputs go to label_true/work_smd/ (labels.json, labels_all.json,
failures.json, metadata.json); the CPCM-era label_true/work/*.json are left
untouched. Every JSON is written atomically.
"""
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage3_parse as s3  # noqa: E402
from _common import BUNDLE_ROOT, load_config, log  # noqa: E402

SCRATCH = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_smd_relabel/inputs")
VIEW = Path("/gpfs/tmp_cpu2/yeseo1ee/eda_smd_relabel/stage3_view")
NEW_SPES = ("eda", "eda_frag1", "eda_frag2")


def build_view(rid_name, old_dir):
    """rxn view dir: SMD outputs from scratch, dist/rel outputs from label_true."""
    v = VIEW / rid_name
    v.mkdir(parents=True, exist_ok=True)
    for s in s3.SPES:
        src = (SCRATCH / rid_name if s in NEW_SPES else old_dir) / f"{s}.out"
        link = v / f"{s}.out"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(src)
    return v


def dump(obj, path):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2)
    os.replace(tmp, path)


def main():
    cfg = load_config()
    work = BUNDLE_ROOT / cfg["work_dir"]
    out = BUNDLE_ROOT / "work_smd"
    out.mkdir(exist_ok=True)
    meta = pd.read_csv(work / "input_meta.csv")
    log(f"smd stage3: parsing {len(meta)} rxns")

    labels, failures, all_parsed, stats = [], [], [], Counter()
    permanent_map = s3.load_permanent_ids(work)
    for _, row in meta.iterrows():
        rid = int(row["rxn_id"])
        name = f"rxn_{rid:04d}"
        old_dir = work / "inputs" / name
        if rid in permanent_map:
            failures.append({"rxn_id": rid, "reason": "permanent_stage2", "classes": permanent_map[rid]})
            stats["permanent_stage2"] += 1
            continue
        if not (SCRATCH / name).is_dir() or not old_dir.is_dir():
            failures.append({"rxn_id": rid, "reason": "missing_dir"}); stats["missing_dir"] += 1
            continue
        d = s3.parse_one_rxn(build_view(name, old_dir), row)
        stats[d["status"]] += 1
        if d["status"] in ("ok", "eda_sum_mismatch"):
            all_parsed.append(d)
        if d["status"] == "ok":
            labels.append(d)
        else:
            failures.append({"rxn_id": rid, "reason": d["status"],
                             "detail": d.get("_error") or d.get("_missing") or d.get("eda_sum_minus_bond_kcal")})

    for lst in (labels, failures, all_parsed):
        lst.sort(key=lambda x: x["rxn_id"])
    for k, v in stats.most_common():
        log(f"  {k}: {v}")
    dump(labels, out / "labels.json")
    dump(all_parsed, out / "labels_all.json")
    dump(failures, out / "failures.json")
    dump({
        "generated_utc": datetime.now(timezone.utc).isoformat(), "bundle_version": "2.1-smd",
        "n_target": len(meta), "n_labeled_ok": len(labels), "n_failures": len(failures),
        "n_labels_all": len(all_parsed), "status_counts": dict(stats),
        "channel_convention": "pauli_dft = Pauli Energy + Delta E^0(XC); all from Hartree column x 627.5094740631",
        "fragment_reference_convention": "option ① (2026-09-28): e_frag{1,2}_dist_eh = standalone frag{1,2}_dist.out (own basis, own cavity); d1/d2 = deformation energies vs frag{1,2}_rel.out; eint_spe = E(AB) - E(frag1_dist) - E(frag2_dist), so d1+d2+eint_spe == barrier exactly. ORCA Bond Energy / 6 channels refer to eda_frag{1,2}.out (ghost basis, AB cavity; kept as e_frag{1,2}_dist_ghost_eh); bsse_shift_kcal = eint_spe - e_bond, so d1+d2+e_bond+bsse_shift == barrier exactly and d1+d2+sum(6ch)+bsse_shift == barrier + eda_sum_minus_bond_kcal.",
        "solvation": "SMD(water) keyword in the eda.inp header and in the FRAG1/FRAG2 strings, so ORCA's fragment SCFs (eda_frag*.out) include SMD-CDS. frag*_rel.out / frag*_dist.out: CPCM(water) + %cpcm smd true smdsolvent \"water\" (same SMD model).",
        "sources": {"eda.out, eda_frag1.out, eda_frag2.out": str(SCRATCH),
                    "frag1_dist.out, frag2_dist.out, frag1_rel.out, frag2_rel.out": str(work / "inputs")},
        "config": cfg,
    }, out / "metadata.json")
    log(f"  work_smd/labels_all.json: {len(all_parsed)} parsed rxns (ok + eda_sum_mismatch)")
    log("smd stage3 complete")


if __name__ == "__main__":
    main()
