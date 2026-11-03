#!/usr/bin/env python3
"""DEPLOY-MODEL Phase M (docs/specs/DEPLOY_MODEL.md 4): label-free metadata for the rev 4-6 feature code.
Run through sbatch only (jobs/cpu_d1.slurm, the D1 run env: rdkit for the reactant charges).

  python meta_lf.py run [--jobs J1,J2,..|ids.csv]   # $DEP/labels_lf.json, $DEP/input_meta_lf.csv, results/M_report.json
  exit 0 ok, 2 not ready (L_report.json missing / not passed / other subset, or L outputs changed since the report),
  3 STOP (a reactant charge != 0, or a record that does not reproduce its result.json / G2 work folder)

One record per `.done` reaction of $DEP/g2lf_final (spec 4-1): rxn_id (manifest meta_rxn_id = folder name), job_id
(extra, for traceability), role1/2 dipole/dipolarophile, charge1/2 = 0/0 (the manifest total charge, the RDKit formal
charges of the manifest dipole_smiles / dipolarophile_smiles and the ts_result.json charge_dipole / charge_dipolarophile
must all be 0, else STOP), n_atoms (ts.xyz), n_f1 = len(A_idx), n_f2, formed_pairs_ts "i-j k-l" (TS 0-based, i < j,
pairs sorted - the labels_all.json format), flag_async / flag_foreign_bond (ts_result.json checks: flag_async,
not no_foreign_bond), status "ok", alt_used (ref_result.json), and the 14 DFT value keys = NaN (float NaN, written as
the JSON token NaN: xtb_slice.process_one l.302-311 compares / subtracts them, None would raise).
ts_result.json / ref_result.json are read from result.json source_dir and must still have the sha256 recorded by
finalize_lf. input_meta_lf.csv: rxn_id, A_idx (space separated, = result.json A_idx).
Keys xtb_slice.process_one / ts_fragments read from a record (l.295-311, 247): rxn_id, charge1, charge2, role1, role2,
n_atoms, n_f1, n_f2, status, alt_used, flag_foreign_bond, flag_async, d1_kcal, d2_kcal, e_ab_eh, e_frag1_dist_eh,
e_frag2_dist_eh, e_frag1_rel_eh, e_frag2_rel_eh, e_bond_kcal, elst_dft, pauli_dft, oi_dft, disp_dft, cpcm_dft,
cds_dft, formed_pairs_ts (ext_features.inputs_fp l.905 / atoms.compute l.391: formed_pairs_ts, charge1, charge2;
ext_features.load_labels l.1137: status) - all present in every record (checked, STOP otherwise).
"""
from __future__ import annotations

import json
import math
import re
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dep_common as C  # noqa: E402
import finalize_lf as FL  # noqa: E402              (load_manifest / load_targets / marker / parse_jobs; D1 run env)

DFT_KEYS = ("d1_kcal", "d2_kcal", "e_bond_kcal", "e_ab_eh", "e_frag1_dist_eh", "e_frag2_dist_eh", "e_frag1_rel_eh",
            "e_frag2_rel_eh", "elst_dft", "pauli_dft", "oi_dft", "disp_dft", "cpcm_dft", "cds_dft")
READ_KEYS = ("rxn_id", "charge1", "charge2", "role1", "role2", "n_atoms", "n_f1", "n_f2", "status", "alt_used",
             "flag_foreign_bond", "flag_async", "formed_pairs_ts") + DFT_KEYS
FP_RE = re.compile(r"^\d+-\d+ \d+-\d+$")
G2_LABELS = C.G2 / "G" / "labels_d1_g2.json"
D0_LABELS = C.REPO / "labels_all.json"


def formal_charge(smi):
    from rdkit import Chem
    m = Chem.MolFromSmiles(str(smi))
    return None if m is None else int(Chem.GetFormalCharge(m))


def fp_string(pairs):
    return " ".join(f"{i}-{j}" for i, j in sorted(tuple(sorted(int(x) for x in p)) for p in pairs))


def make_record(rid, res, tsr, refr, n_atoms, job_id):
    """the spec 4-1 record from a .done result.json and its ts_result.json / ref_result.json."""
    A = [int(i) for i in res["A_idx"]]
    ck = tsr.get("checks") or {}
    rec = dict(rxn_id=int(rid), job_id=str(job_id), role1="dipole", role2="dipolarophile", charge1=0, charge2=0,
               n_atoms=int(n_atoms), n_f1=len(A), n_f2=int(n_atoms) - len(A), formed_pairs_ts=fp_string(res["formed_pairs_ts"]),
               flag_async=bool(ck["flag_async"]), flag_foreign_bond=not bool(ck["no_foreign_bond"]), status="ok",
               alt_used=int(refr["alt_used"]))
    rec.update({k: math.nan for k in DFT_KEYS})
    return rec


def build(lf_root, man, rids):
    """records, meta rows, charge rows, problems for the .done folders `rids` of lf_root."""
    recs, meta, charges, prob, cross = [], [], [], [], []
    for rid in rids:
        d = Path(lf_root) / str(rid)
        try:
            res = json.loads((d / "result.json").read_text())
            src = Path(res["source_dir"])
            sha = res["inputs_sha256"]
            for f in ("ts_result.json", "ref_result.json"):
                if C.sha256(src / f) != sha.get(f):
                    raise ValueError(f"{src / f} changed since finalize_lf (sha256)")
            tsr = json.loads((src / "ts_result.json").read_text())
            refr = json.loads((src / "ref_result.json").read_text())
            n = int((d / "ts.xyz").read_text().split()[0])
            job = str(res["job_id"])
            m = man.get(job)
            if m is None or int(m["meta_rxn_id"]) != int(rid):
                raise ValueError(f"manifest row for {job} missing or meta_rxn_id != {rid}")
            if res.get("status") != "ok" or int(res["rxn_id"]) != int(rid):
                raise ValueError(f"result.json status {res.get('status')} / rxn_id {res.get('rxn_id')}")
            if sorted(int(i) for i in tsr["A_idx"]) != [int(i) for i in res["A_idx"]] or \
                    fp_string(tsr["formed_pairs_ts"]) != fp_string(res["formed_pairs_ts"]):
                raise ValueError("result.json A_idx / formed_pairs_ts != ts_result.json")
            rec = make_record(rid, res, tsr, refr, n, job)
            if not (0 < rec["n_f1"] < n) or any(not (0 <= int(i) < n) for p in res["formed_pairs_ts"] for i in p):
                raise ValueError(f"n_f1 {rec['n_f1']} / pair indices out of range (n_atoms {n})")
        except Exception as e:                                        # noqa: BLE001
            prob.append(dict(rxn_id=int(rid), problem=f"{type(e).__name__}: {e}"))
            continue
        q = dict(rxn_id=int(rid), job_id=job, manifest_charge=int(m["charge"]),
                 dipole_smiles_charge=formal_charge(m["dipole_smiles"]),
                 dipolarophile_smiles_charge=formal_charge(m["dipolarophile_smiles"]),
                 ts_result_charge_dipole=tsr.get("charge_dipole"), ts_result_charge_dipolarophile=tsr.get("charge_dipolarophile"))
        q["nonzero"] = any(v is None or int(v) != 0 for k, v in q.items() if k not in ("rxn_id", "job_id"))
        charges.append(q)
        A = set(int(i) for i in res["A_idx"])
        cross.append(all((int(i) in A) != (int(j) in A) for i, j in res["formed_pairs_ts"]))
        recs.append(rec)
        meta.append(dict(rxn_id=int(rid), A_idx=" ".join(str(int(i)) for i in res["A_idx"])))
    return recs, meta, charges, prob, cross


def schema_vs(path, recs):
    """key differences against the first record of another label file (+ its formed_pairs_ts format)."""
    try:
        other = json.loads(Path(path).read_text())
    except (OSError, ValueError) as e:
        return dict(path=str(path), error=f"{type(e).__name__}: {e}")
    ko, kl = set(other[0]), set(recs[0]) if recs else set()
    fmt = [d.get("formed_pairs_ts") for d in other]
    return dict(path=str(path), sha256=C.sha256(path), n=len(other), lf_keys_not_in_it=sorted(kl - ko),
                its_keys_not_in_lf=sorted(ko - kl), read_keys_missing_in_it=sorted(set(READ_KEYS) - ko),
                n_formed_pairs_ts_format_ok=sum(bool(isinstance(s, str) and FP_RE.match(s)) for s in fmt)), other


def cmd_run(jobs=None):
    lrep_p = C.RES / "L_report.json"
    lrep = C.read_json(lrep_p) if lrep_p.is_file() else None
    want = sorted(jobs) if jobs else None
    if not lrep or not lrep.get("complete") or not lrep.get("all_pass") or lrep.get("subset") != want:
        C.not_ready(f"{lrep_p}: missing, incomplete, not passed or subset {None if not lrep else lrep.get('subset')} "
                    f"!= {want}")
    tg = FL.load_targets(jobs)
    tg_rid = {int(t["rxn_id"]) for t in tg}
    rids = sorted(int(p.parent.name) for p in C.LF_FINAL.glob("*/.done")
                  if p.parent.name.isdigit() and int(p.parent.name) in tg_rid)
    st = pd.read_csv(C.RES / "L_status.csv")
    st_done = sorted(int(r) for r in st.rxn_id[st.lf_marker == ".done"])
    if rids != st_done:
        C.not_ready(f"{C.LF_FINAL} .done set ({len(rids)}) != L_status.csv .done set ({len(st_done)}): rerun report")
    man = FL.load_manifest()
    recs, meta, charges, prob, cross = build(C.LF_FINAL, man, rids)
    Q = pd.DataFrame(charges)
    nz = Q[Q.nonzero] if len(Q) else Q
    key_missing = sorted({k for r in recs for k in READ_KEYS if k not in r})
    fmt_bad = [r["rxn_id"] for r in recs if not FP_RE.match(r["formed_pairs_ts"])]
    s_g2 = schema_vs(G2_LABELS, recs)
    s_d0 = schema_vs(D0_LABELS, recs)
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
    rep = dict(spec="docs/specs/DEPLOY_MODEL.md 4", lf_root=str(C.LF_FINAL), subset=want, n_done_folders=len(rids),
               n_records=len(recs), n_problems=len(prob), problems=prob[:50], n_charge_nonzero=len(nz),
               charge_nonzero=nz.to_dict("records")[:50] if len(nz) else [],
               charge_sources=["manifest charge", "RDKit formal charge of manifest dipole_smiles / dipolarophile_smiles",
                               "ts_result.json charge_dipole / charge_dipolarophile"],
               read_keys=list(READ_KEYS), read_keys_missing=key_missing, dft_keys_nan=list(DFT_KEYS),
               n_formed_pairs_ts_format_bad=len(fmt_bad), formed_pairs_ts_format_bad=fmt_bad[:20],
               n_pairs_not_across_partition=sum(not x for x in cross),
               schema_vs_labels_d1_g2=s_g2, schema_vs_labels_all=s_d0, values_vs_labels_d1_g2=cmp,
               l_report_sha256=C.sha256(lrep_p), manifest_sha256=C.sha256(C.D1_MANIFEST), written=time.strftime("%F %T"))
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
        back = json.loads(C.LABELS_LF.read_text())                     # NaN survives the round trip as float NaN
        rep["nan_roundtrip_ok"] = all(isinstance(b[k], float) and math.isnan(b[k]) for b in back[:50] for k in DFT_KEYS)
        rep["outputs"] = {str(C.LABELS_LF): C.sha256(C.LABELS_LF), str(C.META_LF): C.sha256(C.META_LF)}
        if not rep["nan_roundtrip_ok"]:
            rep["stop"] = stop = ["DFT keys do not read back as NaN"]
            rep["all_pass"] = False
    C.write_json(C.RES / "M_report.json", rep)
    print(json.dumps({k: rep[k] for k in ("n_done_folders", "n_records", "n_problems", "n_charge_nonzero",
                                          "read_keys_missing", "n_pairs_not_across_partition", "all_pass", "stop")}),
          flush=True)
    if stop:
        C.stop("; ".join(stop))
    return 0


if __name__ == "__main__":
    a, jobs = FL.parse_jobs(sys.argv[1:])
    if a == ["run"]:
        sys.exit(cmd_run(jobs))
    raise SystemExit(__doc__)
