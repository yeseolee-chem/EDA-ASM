#!/usr/bin/env python3
"""c_deviation.py — the recorded C-3′ deviation (user 11-10 "추천 방식대로 진행") that lets PREREG / K / W proceed although
results/C_tests.json is all_pass false. Stdlib only. Exit 0 ok, 2 not ready, 3 STOP.

  python c_deviation.py write    -> results/C_deviation.json; proceed only when every condition below holds
  (check() is the gate used by prereg_ch3.py write and run_dev.py)

Rule (fixed before the diagnostics finished, reported to the user 13:1x and accepted):
  1. C_tests.json: every check passes except "C-3′ ..." (the deviation covers C-3′ only);
  2. D1 = deploy train_tf.py c3 rerun now (job 1070796, $CH3/diag/dep_res/C3_regress.json): ok true and the tuning
     validation values / best epochs equal the LADDER unit's (|delta val| <= 1e-12) -> the GPU training is deterministic
     today, so the C-3′ differences come from the inputs (regenerated, <= 8.6e-7 from deploy, V_inputs.json);
  3. D2 = seed-only replicate on the regenerated inputs (job 1070797, $CH3/diag/c3r_null.json): it fails at least one
     C-3′ criterion against deploy as well (other cell, |best epoch diff| > 2, or a target outside +-2 %) -> the C-3′
     tolerances are not reachable without bit-identical inputs.
The C_tests.json, V_inputs.json and the two diagnostic files are fixed by sha256 in C_deviation.json; check() exits 3
when any of them changed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ch3_common as C3  # noqa: E402

OUT = C3.RES / "C_deviation.json"
CT = C3.RES / "C_tests.json"
V = C3.RES / "V_inputs.json"
D1 = C3.CH3 / "diag" / "dep_res" / "C3_regress.json"
D2 = C3.CH3 / "diag" / "c3r_null.json"
C3R = C3.RES / "C3r_regress.json"
C3_KEY = "C-3′ 10-target regression vs deploy K fold 0"
USER = "11-10 13:1x: 추천 방식대로 진행 (C-3′ 실패를 입력 미세 차이 편차로 기록하고 PREREG → K → E → W → I 진행; 조건 D1 재현 · D2 도 ±2 % 이탈)"


def _files():
    return {str(p): C3.sha256(p) for p in (CT, V, D1, D2, C3R)}


def cmd_write():
    miss = [str(p) for p in (CT, V, D1, D2, C3R) if not p.is_file()]
    if miss:
        C3.not_ready(f"missing {miss} (diagnostics D1 / D2 not finished?)")
    ct, d1, d2, c3r = (C3.read_json(p) for p in (CT, D1, D2, C3R))
    checks = ct.get("checks") or {}
    failed = [k for k, v in checks.items() if not v]
    cond1 = failed == [C3_KEY] and bool(C3.read_json(V).get("all_pass"))
    dv = [abs(float(a["val"]) - float(b["val"])) for a, b in zip(d1.get("tune") or [], d1.get("tune_ladder") or [])]
    de = [int(a["best_epoch"]) - int(b["best_epoch"]) for a, b in zip(d1.get("tune") or [], d1.get("tune_ladder") or [])]
    cond2 = bool(d1.get("ok")) and len(dv) == 4 and max(dv) <= 1e-12 and not any(de)
    ch = d2.get("chosen") or {}
    cell_diff = json.dumps((ch.get("null") or {}).get("cell"), sort_keys=True) != \
        json.dumps((ch.get("deploy") or {}).get("cell"), sort_keys=True)
    ep_diff = abs(int((d2.get("best_epoch_diff") or {}).get("null_vs_deploy", 0)))
    n_in = int((d2.get("n_targets_within_2pct") or {}).get("null_vs_deploy", 10))
    cond3 = bool(cell_diff or ep_diff > 2 or n_in < 10)
    rel_null = (d2.get("rel") or {}).get("null_vs_deploy") or {}
    rel_c3r = c3r.get("para_mae_rel_diff") or {}
    rec = dict(spec="docs/specs/CH3_MODEL.md 6 / 15 (C-3′ STOP) — deviation", written=C3.now(), user_decision=USER,
               conditions={"1 only C-3′ failed, V all_pass": cond1,
                           "2 D1 deploy C-3 rerun reproduces LADDER exactly (deterministic GPU training)": cond2,
                           "3 D2 seed-only replicate also misses a C-3′ criterion vs deploy": cond3},
               evidence=dict(c_tests_failed=failed,
                             D1=dict(ok=d1.get("ok"), max_abs_dval_vs_ladder=max(dv) if dv else None,
                                     best_epoch_diffs=de, chosen=d1.get("chosen"), gpu=d1.get("gpu")),
                             D2=dict(chosen=ch, best_epoch_diff=d2.get("best_epoch_diff"),
                                     n_targets_within_2pct=d2.get("n_targets_within_2pct"),
                                     rel_null_vs_deploy=rel_null, rel_c3r_vs_deploy=rel_c3r,
                                     max_abs_rel_null_vs_deploy=max((abs(v) for v in rel_null.values()), default=None),
                                     max_abs_rel_c3r_vs_deploy=max((abs(v) for v in rel_c3r.values()), default=None)),
                             C3r=dict(chosen=c3r.get("chosen"), chosen_deploy=c3r.get("chosen_deploy"),
                                      best_epoch_diff=c3r.get("best_epoch_diff"))),
               files_sha256=_files())
    rec["proceed"] = bool(cond1 and cond2 and cond3)
    C3.write_json(OUT, rec)
    print(json.dumps(dict(proceed=rec["proceed"], conditions=rec["conditions"]), indent=1, ensure_ascii=False))
    if not rec["proceed"]:
        C3.stop("C-3′ deviation conditions not met: the STOP stands (report to the user)")
    return 0


def check():
    """gate: C_deviation.json proceed true and every file it fixed unchanged (exit 3 otherwise)."""
    if not OUT.is_file():
        C3.stop(f"{OUT} missing: C_tests.json is not all_pass and no recorded deviation")
    rec = C3.read_json(OUT)
    if not rec.get("proceed"):
        C3.stop(f"{OUT.name}: proceed false")
    now = _files()
    bad = [p for p, s in (rec.get("files_sha256") or {}).items() if now.get(p) != s]
    if bad:
        C3.stop(f"{OUT.name}: files changed since the deviation was recorded: {bad}")
    return dict(rec, c_deviation_sha256=C3.sha256(OUT))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a == ["write"]:
        sys.exit(cmd_write())
    if a == ["check"]:
        r = check()
        print(json.dumps(dict(ok=True, proceed=r["proceed"]), indent=1))
        sys.exit(0)
    print(__doc__)
    sys.exit(2)
