#!/usr/bin/env python3
"""SPEC17rev2 Step 2 — P atom-order verification + R complex construction.

Uses the canonical §4 algorithm from _frag_align.py (do not modify it).

GATE-2 targets (from spec §3):
  P -> TS identity dominant : 200/200 sample => require 100% here
  R build success           : ≥ 98%
  connectivity fail count   : 0   (build_reactant_complex enforces internally)
  inter_R < 1.0 Å count     : 0   (STOP if any)
  MAX_ISO(5000) hit rate    : ≤ 10%  (11/240 = 4.6% in spec sample)

Idempotent: re-writes reactant_complex/rxn_NNNN.npz + reactant_build.csv.
Single-process by design (~24 min for 5269 rxns per spec) so numbers stay
reproducible against the spec sample.
"""
from __future__ import annotations

import collections
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _frag_align import (  # noqa: E402
    MAX_ISO, build_graph, build_reactant_complex, rmsd, verify_product,
)

BASE = Path("/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/analysis/otfm_train")
PROF = BASE / "coley_profiles/full_dataset_profiles"
OUT = BASE / "data" / "reactant_complex"


def main() -> int:
    if not PROF.exists():
        print(f"[FATAL] profiles missing: {PROF}. Run Step 0.", file=sys.stderr)
        return 1
    OUT.mkdir(parents=True, exist_ok=True)

    ids = sorted(int(d.name) for d in PROF.iterdir()
                 if d.is_dir() and d.name.isdigit())
    print(f"scanning {len(ids)} reactions")

    # ---- 2a. P atom-order verification (full 5269) -------------------------
    t0 = time.time()
    pv_true = pv_false = pv_none = 0
    pv_fail_ids = []
    for rid in ids:
        v = verify_product(PROF, rid)
        if v is True:
            pv_true += 1
        elif v is False:
            pv_false += 1
            pv_fail_ids.append(rid)
        else:
            pv_none += 1
    print(f"[P->TS identity] pass {pv_true}, fail {pv_false}, "
          f"undecidable {pv_none}   ({time.time()-t0:.1f}s)")
    if pv_fail_ids:
        (BASE / "artifacts" / "p_order_failures.csv").write_text(
            "rxn_id\n" + "\n".join(str(r) for r in pv_fail_ids) + "\n"
        )

    # ---- 2b. R complex build -----------------------------------------------
    t0 = time.time()
    rows, fails = [], collections.Counter()
    n_cap = 0
    for i, rid in enumerate(ids):
        res, err = build_reactant_complex(PROF, rid)
        if err:
            fails[err] += 1
            rows.append(dict(rxn_id=rid, ok=False, reason=err))
            continue
        A, B = res["frag1"], res["frag2"]
        R, TS = res["R"], res["TS"]
        inter = float(np.linalg.norm(
            R[A][:, None, :] - R[B][None, :, :], axis=-1).min())
        ts_inter = float(np.linalg.norm(
            TS[A][:, None, :] - TS[B][None, :, :], axis=-1).min())
        cap = any(i2["hit_cap"] for i2 in res["info"])
        n_cap += int(cap)
        rows.append(dict(
            rxn_id=rid, ok=True, n_atoms=len(res["syms"]),
            rmsd_R_TS=rmsd(R, TS), shift=res["shift"],
            inter_R=inter, inter_TS=ts_inter,
            p_order_ok=res["p_order_ok"], hit_cap=cap,
            n_iso_max=max(x["n_iso"] for x in res["info"]),
            max_disp_heavy=max(x["max_disp_heavy"] for x in res["info"]),
            max_disp_h=max(x["max_disp_h"] for x in res["info"]),
        ))
        # atomic write via temp + rename
        tmp = OUT / f".rxn_{rid:04d}.npz.tmp"
        np.savez(tmp,
                 R=R, TS=TS, P=res["P"], syms=np.array(res["syms"]),
                 frag1=np.array(A), frag2=np.array(B))
        tmp.replace(OUT / f"rxn_{rid:04d}.npz")

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(ids)}  elapsed {elapsed:.0f}s "
                  f"({elapsed/(i+1):.3f}s/rxn)")

    D = pd.DataFrame(rows)
    D.to_csv(BASE / "artifacts" / "reactant_build.csv", index=False)
    ok = D[D.ok]
    print(f"\nR build success: {len(ok)} / {len(D)}   failures: {dict(fails)}")
    print(f"MAX_ISO({MAX_ISO}) hit: {n_cap} "
          f"({n_cap/max(1,len(ok)):.1%})")
    if len(ok):
        print(f"n_iso max: {ok.n_iso_max.max()}")
        print(f"\nseparation shift (Å): median {ok.shift.median():.2f}  "
              f"max {ok.shift.max():.2f}")
        print(f"fragment-fragment min distance (Å)")
        print(f"  R : median {ok.inter_R.median():.3f}  "
              f"min {ok.inter_R.min():.3f}  "
              f"<1.0Å {(ok.inter_R < 1.0).mean():.1%}")
        print(f"  TS: median {ok.inter_TS.median():.3f}  "
              f"min {ok.inter_TS.min():.3f}")
        print(f"R-TS RMSD(Å): median {ok.rmsd_R_TS.median():.3f}  "
              f"p95 {np.percentile(ok.rmsd_R_TS,95):.3f}")
        print(f"max displacement heavy (Å): median "
              f"{ok.max_disp_heavy.median():.3f}  max "
              f"{ok.max_disp_heavy.max():.3f}")
        print(f"max displacement H (Å):     median "
              f"{ok.max_disp_h.median():.3f}  max "
              f"{ok.max_disp_h.max():.3f}")

    # ---- gate evaluation ----------------------------------------------------
    n = len(D)
    p_ok = (pv_false == 0)
    r_ok = len(ok) / n >= 0.98 if n else False
    conn_ok = ("correspondence verify failed" not in fails)
    inter_ok = len(ok) > 0 and (ok.inter_R < 1.0).sum() == 0
    p_col_ok = len(ok) > 0 and bool(ok.p_order_ok.all())
    cap_ok = n_cap / max(1, len(ok)) <= 0.10

    passed = all([p_ok, r_ok, conn_ok, inter_ok, p_col_ok, cap_ok])
    (BASE / "artifacts" / "GATE2_STATUS.txt").write_text(
        ("PASS" if passed else "FAIL") + "\n"
        f"p_verify_pass={pv_true}\n"
        f"p_verify_fail={pv_false}\n"
        f"p_verify_undecidable={pv_none}\n"
        f"r_build_success={len(ok)}/{n}\n"
        f"connectivity_verify_fail={fails.get('correspondence verify failed', 0)}\n"
        f"inter_R_lt_1A_count={int((ok.inter_R < 1.0).sum()) if len(ok) else 0}\n"
        f"p_order_ok_all={p_col_ok}\n"
        f"max_iso_hit_rate={n_cap/max(1,len(ok)):.4f}\n"
    )
    if not passed:
        print("[GATE-2 FAIL] see artifacts/GATE2_STATUS.txt", file=sys.stderr)
        return 1
    print("=== GATE-2 PASS ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
