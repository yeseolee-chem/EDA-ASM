#!/usr/bin/env python3
"""gate_audit_d0.py — V9 (VALIDATION_SPEC §6.8, A8): the pilot TS gate on every D0 record (no QM).

  python analysis/gate_audit_d0.py --task V9
For each labels_all record (5,260 ok + 5 excluded): Coley TS, plain reactants r*.xyz, atom-mapped SMILES,
and the imaginary mode from TS_imag_mode.xyz (autodE's animation: frames run TS - A*mode -> TS + A*mode
-> back, so mode ~ frame[k*] - frame[0] with k* the frame farthest from frame 0) go through
run_ts.analyse: partition, forming-bond audit, foreign bond, forming range, regio flag, mode share.
The imaginary-count gate is not applicable (no D0 frequencies): imag = None.
Writes <scratch>/v9/v9_gate_audit.csv and result.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

VAL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VAL.parent / "d1_autode_pilot"))
import pilot_common as pc  # noqa: E402

G = {}


def mode_from_animation(path: Path):
    lines = path.read_text().splitlines()
    n = int(lines[0].split()[0])
    frames = []
    for k in range(0, len(lines) - n - 1, n + 2):
        frames.append([[float(v) for v in l.split()[1:4]] for l in lines[k + 2:k + 2 + n]])
    X = np.array(frames)
    d = np.linalg.norm((X - X[0]).reshape(len(X), -1), axis=1)
    return X[int(np.argmax(d))] - X[0]


def init(cfg_path):
    cfg = pc.load_config(cfg_path)
    fr, _ = pc.import_repo_modules(cfg)
    import run_ts
    coley = pd.read_csv(cfg["d0_csv"])
    G.update(cfg=cfg, fr=fr, run_ts=run_ts, smi=dict(zip(coley.rxn_id.astype(int), coley.rxn_smiles)),
             tmp=Path(cfg["scratch"]) / "v9" / "tmp" / str(os.getpid()))
    G["tmp"].mkdir(parents=True, exist_ok=True)


def one(rec):
    rid, status = int(rec["rxn_id"]), rec["status"]
    cfg, rt = G["cfg"], G["run_ts"]
    pdir = Path(cfg["d0_profiles"]) / str(rid)
    row = dict(rxn_id=rid, status=status)
    try:
        ts = pc.read_xyz(sorted(p for p in pdir.glob("TS_*.xyz") if p.name != "TS_imag_mode.xyz")[0])
        reacs = [pc.read_xyz(p) for p in sorted(pdir.glob("r*.xyz")) if "_alt" not in p.name]
        mode = mode_from_animation(pdir / "TS_imag_mode.xyz")
        job = dict(job_id=f"D0-{rid:04d}", kind="D0_replay", mapped_smiles=G["smi"][rid])
        res = {}
        got = dict(ts=ts, reacs=reacs, prod=None, imag=None, mode=mode)
        try:
            rt.analyse(cfg, job, G["tmp"], res, got, G["fr"])
            row["hard_fail"] = ""
        except rt.StageFail as e:
            row["hard_fail"] = str(e)
        c = res.get("checks") or {}
        row.update(partition=(res.get("partition") or {}).get("status"), mode_share=res.get("imag_mode_forming_share"),
                   formed_d=res.get("formed_d"), n_foreign=len(res.get("foreign") or []),
                   flag_regio_ambiguous=c.get("flag_regio_ambiguous"), flag_async=c.get("flag_async"),
                   flag_short_forming=c.get("flag_short_forming"),
                   regio_margin=(res["regio_sum_crossed"] - res["regio_sum_formed"]) if "regio_sum_formed" in res else None)
    except Exception as e:                                            # noqa: BLE001
        row["hard_fail"] = f"error:{type(e).__name__}: {e}"
    return row


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--task", default="V9")
    a = ap.parse_args()
    cfg_path = os.environ.get("D1P_CONFIG")
    cfg = pc.load_config(cfg_path)
    out = Path(cfg["scratch"]) / "v9"; out.mkdir(parents=True, exist_ok=True)
    if (out / "result.json").is_file():
        print("V9: done"); return
    recs = [dict(rxn_id=r["rxn_id"], status=r["status"]) for r in json.load(open(pc.repo_path(cfg, cfg["labels_all"])))]
    with Pool(pc.n_cores(), initializer=init, initargs=(cfg_path,)) as pool:
        rows = pool.map(one, recs, chunksize=20)
    df = pd.DataFrame(rows).sort_values("rxn_id")
    df.to_csv(out / "v9_gate_audit.csv", index=False)
    ok = df[df.status == "ok"]
    bad_ok = ok[ok.hard_fail != ""]
    A8 = cfg["prereg"]["A8"]
    hf = dict(zip(df.rxn_id, df.hard_fail.astype(str)))
    # foreign-bond rxns must fail on no_foreign_bond; no-bond rxns (FIX F-A) on any hard gate
    fdet = {int(r): ("no_foreign_bond" in hf[int(r)]) if int(r) in hf else None for r in A8["foreign_ids"]}
    ndet = {int(r): (hf[int(r)] != "") if int(r) in hf else None for r in A8.get("no_bond_ids", [])}
    res = dict(n_records=len(df), n_ok=len(ok), n_ok_hard_fail=len(bad_ok),
               ok_hard_fail=bad_ok[["rxn_id", "hard_fail"]].to_dict("records"),
               foreign_detected=fdet, no_bond_detected=ndet,
               n_short_forming_flag_ok=int(ok.flag_short_forming.fillna(False).astype(bool).sum()) if "flag_short_forming" in ok else None,
               excluded=df[df.status != "ok"][["rxn_id", "status", "hard_fail"]].to_dict("records"),
               mode_share_min_ok=float(ok.mode_share.min()), mode_share_p001_ok=float(ok.mode_share.quantile(0.001)),
               n_regio_flag_ok=int((ok.flag_regio_ambiguous == True).sum()),         # noqa: E712
               n_async_flag_ok=int((ok.flag_async == True).sum()))                   # noqa: E712
    (out / "result.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k not in ("ok_hard_fail", "excluded")}, indent=1))


if __name__ == "__main__":
    main()
