#!/usr/bin/env python3
"""GNN3R Phase D (docs/specs/GNN3R.md 6): the GNN3 data built by gr_common.dataset() -> results/D_check.json.

    python analysis/gnn3r/check_data.py          env_d1fu2 (gnn3/jobs/run.slurm), CPU

D-1 usable reactions = the g2r usable set (8,617); D-2 every pair usable and in a fold; D-3 fold_split(k) drops
nothing and has the S_splits_r.json sizes; D-4 all_training_pairs() = every pair. Exit 0 pass, 3 = STOP.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gr_common as C  # noqa: E402


def input_sha256() -> dict:
    """every file D reads (PairDataset inputs + the files D-1 / D-4 read that ds.sha256 does not cover)"""
    files = [C.TOKEN_REPORTS / n for n in ("T_report.json", "T_manifest.csv", "Q_report.json")] + [
        C.X182, C.PAIRS_R, C.LABEL_REPORT, C.S_SPLITS_R, C.S_FOLDS_R, C.USABLE] + [
        C.SPLIT_DIR / f"T1__f{k}.json" for k in range(5)]
    return {str(p): C.sha256(p) for p in files}


def main() -> int:
    t0 = time.time()
    out = C.RES / "D_check.json"
    inputs = input_sha256()
    if out.is_file():
        old = C.read_json(out)
        if old.get("all_pass") and old.get("inputs") == inputs:
            print(f"Phase D done ({out}); not rebuilt")
            return 0
    ds = C.dataset()
    U = pd.read_csv(C.USABLE)
    want = {int(r) for r in U.loc[U.usable.astype(bool), "rxn_id"]}
    got = {int(r) for r in ds.reactions.reaction_ids[ds.usable_reaction]}
    gates = {"D-1": dict(n_usable_reaction=int(ds.usable_reaction.sum()), want=C.N_USABLE, n_token_reactions=
                         int(len(ds.reactions.reaction_ids)), same_set=got == want,
                         ok=int(ds.usable_reaction.sum()) == C.N_USABLE and got == want)}
    all_ids = sorted(ds.pairs.index.astype(str))
    gates["D-2"] = dict(n_pairs=int(ds.n_pairs), n_usable_pair=int(ds.usable_pair.sum()), n_pair_list=len(ds.pair_list),
                        ok=bool(ds.usable_pair.all()) and ds.pair_list == all_ids)
    rep_s = C.read_json(C.S_SPLITS_R)
    folds = {}
    for k in range(5):
        sp = ds.fold_split(k)
        unit = rep_s["units"][f"T1__f{k}.json"]
        sizes = dict(n_train=len(sp.train), n_test=len(sp.test), n_inner_train=len(sp.inner_train),
                     n_inner_val=len(sp.inner_val))
        no_drop = all(v == 0 for d in sp.dropped.values() for v in d.values())
        folds[k] = dict(sizes=sizes, dropped=sp.dropped, no_drop=no_drop,
                        sizes_equal=all(sizes[n] == unit[n] for n in sizes), pairs_sha256=sp.pairs_sha256)
    gates["D-3"] = dict(folds=folds, ok=all(f["no_drop"] and f["sizes_equal"] for f in folds.values()))
    try:
        allp = ds.all_training_pairs()
        gates["D-4"] = dict(n_train=int(len(allp.train)), n_pairs=int(ds.n_pairs), dropped=allp.dropped,
                            ok=len(allp.train) == ds.n_pairs)
    except Exception as e:                                                    # noqa: BLE001
        gates["D-4"] = dict(error=f"{type(e).__name__}: {e}", ok=False)
    bad = [g for g, v in gates.items() if not v["ok"]]
    rep = dict(spec="docs/specs/GNN3R.md 6", written=time.strftime("%Y-%m-%d %H:%M:%S"),
               slurm_job=os.environ.get("SLURM_JOB_ID"), wall_s=round(time.time() - t0, 1),
               dataset=dict(token_dir=str(C.TOKENS), report_dir=str(C.TOKEN_REPORTS), global_table=str(C.X182),
                            pair_labels=str(C.PAIRS_R), split_dir=str(C.SPLIT_DIR), split_report=str(C.S_SPLITS_R),
                            pair_attributes=str(C.S_FOLDS_R), label_report=str(C.LABEL_REPORT)),
               dataset_sha256=ds.sha256, inputs=inputs, gates=gates, all_pass=not bad, failed=bad)
    C.write_json(out, rep)
    print(json.dumps(dict(all_pass=not bad, failed=bad, **{g: v["ok"] for g, v in gates.items()},
                          sizes={k: f["sizes"] for k, f in folds.items()}), indent=1, default=str))
    return 0 if not bad else 3


if __name__ == "__main__":
    sys.exit(main())
