#!/usr/bin/env python3
"""D1-G2 Phase X (docs/specs/D1G2.md section 3): targets and their autodE inputs, checked against what the D1 run used.
Run through sbatch only (jobs/g2.slurm, G2ENV=d1: the D1 venv has rdkit and the D1 run code).

  python prep_x.py run     # results/X_targets.csv, results/X_report.json (exit 3 = STOP, spec 14 X)

Targets = labels_d1.json rows without the D'' skeletons (d1pair_common) = the D1HF DFT targets (X_manifest.csv; STOP if
the lists differ). Inputs per row = the D1 manifest (d1_run/d1_manifest.csv): rxn_smiles (the only thing autodE gets),
product_smiles (the requested regio / stereo isomer; = the product side of rxn_smiles), mapped_smiles, charge, mult.
Checked against the D1 job records: ts_result.json (rxn_smiles, ts_id, kind, ade_name, autode_version 1.4.5, dry_xtb
false) and stereo_result.json (requested = stereo_check.canonical(product_smiles)), read from the Nurion ledger summary
tars (d1_ledger/summary/<bundle>.small.tar, sha256 = bundles.tsv) or the UBAI job folders, and against labels_d1.json
(rxn_smiles, ts_id, charge1 + charge2).
"""
from __future__ import annotations

import json
import sys
import tarfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g2_common as G  # noqa: E402

sys.path.insert(0, str(G.D1_RUN))
import stereo_check as SC  # noqa: E402                (canonical: the D1 run's own product canonicalisation)


def records_from_tar(tar, want):
    """{job_id: {name: dict}} of ts_result.json / stereo_result.json for the wanted jobs of one summary tar."""
    out = {j: {} for j in want}
    with tarfile.open(tar, "r:") as T:
        for m in T:
            p = (m.name[2:] if m.name.startswith("./") else m.name).split("/")
            if m.isfile() and len(p) == 3 and p[0] == "jobs" and p[1] in out and p[2] in ("ts_result.json",
                                                                                         "stereo_result.json"):
                out[p[1]][p[2]] = json.loads(T.extractfile(m).read())
    return out


def cmd_run():
    tg = G.targets()
    hf = pd.read_csv(G.HF_RES / "X_manifest.csv", usecols=["job_id"])
    if sorted(hf.job_id) != [d["job_id"] for d in tg]:
        G.stop(f"targets ({len(tg)}) != D1HF DFT targets ({len(hf)})")
    if G.sha256(G.MANIFEST) != G.MANIFEST_SHA:
        G.stop(f"{G.MANIFEST} sha256 changed")
    M = pd.read_csv(G.MANIFEST, dtype=str, keep_default_na=False).set_index("job_id")
    st = pd.read_csv(G.DP.D1_STATUS, dtype=str, keep_default_na=False).set_index("job_id")
    bt = pd.read_csv(G.LEDGER / "bundles.tsv", sep="\t", dtype=str).set_index("bundle")
    ids = [d["job_id"] for d in tg]
    rec, bad, sha = {}, [], {}
    for b, g in st.loc[ids].groupby("harvested"):
        if b == "":
            for j in g.index:
                d = G.UBAI_JOBS / j
                rec[j] = {n: json.loads((d / n).read_text()) for n in ("ts_result.json", "stereo_result.json")
                          if (d / n).is_file()}
            continue
        tar = G.LEDGER / "summary" / f"{b}.small.tar"
        sha[tar.name] = G.sha256(tar)
        if sha[tar.name] != bt.loc[b, "summary_sha256"]:
            G.stop(f"{tar} sha256 != bundles.tsv summary_sha256")
        rec.update(records_from_tar(tar, set(g.index)))
    rows = []
    for d in tg:
        j, m, r = d["job_id"], M.loc[d["job_id"]], rec.get(d["job_id"], {})
        ts, so = r.get("ts_result.json", {}), r.get("stereo_result.json", {})
        chk = {"ts_result_present": bool(ts), "stereo_result_present": bool(so),
               "ts_rxn_smiles": ts.get("rxn_smiles") == m.rxn_smiles, "ts_ts_id": ts.get("ts_id") == m.ts_id,
               "ts_kind_D1": ts.get("kind") == "D1" == m.kind,
               "ts_ade_name": ts.get("ade_name", j) == j, "ts_autode_1.4.5": ts.get("autode_version") == "1.4.5",
               "ts_not_dry_xtb": ts.get("dry_xtb") is False,
               "stereo_requested": so.get("requested") == SC.canonical(m.product_smiles),
               "product_side": m.rxn_smiles.split(">>")[1] == m.product_smiles,
               "label_rxn_smiles": d.get("rxn_smiles") == m.rxn_smiles, "label_ts_id": d.get("ts_id") == m.ts_id,
               "charge": int(d["charge1"]) + int(d["charge2"]) == int(m.charge), "mult_1": m.mult == "1"}
        miss = [k for k, v in chk.items() if not v]
        if miss:
            bad.append(f"{j}: {miss}")
        rows.append(dict(job_id=j, rxn_id=int(d["rxn_id"]), dipole=d["dipole"], skeleton=d["skeleton"], ts_id=m.ts_id,
                         n_atoms=int(d["n_atoms"]), charge=int(m.charge), mult=int(m.mult), rxn_smiles=m.rxn_smiles,
                         product_smiles=m.product_smiles, mapped_smiles=m.mapped_smiles,
                         source=st.loc[j, "harvested"] or "ubai", checks_failed=";".join(miss)))
    X = pd.DataFrame(rows)
    G.write_atomic(G.RES / "X_targets.csv", lambda t: X.to_csv(t, index=False))
    rep = dict(n_targets=len(X), n_check_failed=len(bad), failed=bad[:30], same_as_d1hf=True,
               sources=X.source.map(lambda s: "ubai" if s == "ubai" else "nurion").value_counts().to_dict(),
               sha256={"d1_manifest.csv": G.MANIFEST_SHA, "labels_d1.json": G.DP.D1_LABELS_SHA,
                       "d1_status.csv": G.sha256(G.DP.D1_STATUS), "bundles.tsv": G.sha256(G.LEDGER / "bundles.tsv"),
                       **sha, "X_targets.csv": G.sha256(G.RES / "X_targets.csv")},
               note="autodE receives rxn_smiles only (run_ts.py: ade.Reaction(rxn_smiles, name=job_id, solvent_name, "
                    "temp)); the requested isomer is the product side; charge / multiplicity follow from SMILES")
    G.write_json(G.RES / "X_report.json", rep)
    print(json.dumps({k: v for k, v in rep.items() if k != "sha256"}, indent=1), flush=True)
    return 3 if bad else 0


if __name__ == "__main__":
    if sys.argv[1:2] == ["run"]:
        sys.exit(cmd_run())
    raise SystemExit(__doc__)
