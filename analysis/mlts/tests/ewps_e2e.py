#!/usr/bin/env python3
"""ewps_e2e.py — test of predict_v2.e2e without autodE (EWPS implementer): the selfcheck root $MLT/p/selfcheck is
pre-filled with stand-in A-2 folders (inputs_mlts smoke_src: G2 xTB TSs), so run_one is skipped (folder exists); then
the real chain (inputs_mlts H commands through predict_v2 _im), committee spread with stand-in members
(MLTS_W_MEMBERS, zero-shot MLIPs) and the W smoke model card. mlts env, GPU sandbox. Stand-in, not MLTS results."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import mlt_common as M  # noqa: E402
import predict_v2 as PV  # noqa: E402

if "ewpstest" not in str(M.MLT):
    raise SystemExit("refusing: MLT_SCRATCH must be the ewpstest root")
card = M.read_json(M.MLT / "w" / "smoke" / "MODEL_CARD.json")
rep8 = {t: tuple(v) for t, v in card["REP8"].items()}
have = {}
for k, v in card["parts"].items():
    a, m = k.split("|")
    for t in (v.get("targets") or PV.TARGETS):
        if rep8.get(t) == (a, m):
            have[t] = (a, m)
pairs = [("TESTPAIR_0", 1000312, 1000848)]
news = []
rec = PV.e2e(pairs, news, "C1", dict(name="C1", refs="min"), have, "min", card, cores=2, workers=2)
M.write_json(PV.SC_DIR / "e2e_test.json", rec)
print(json.dumps({k: rec.get(k) for k in ("steps", "chain_stop", "delta_com", "terminal", "predicted_all_ok_ts",
                                          "predict_error", "pass_", "n_inputs_usable", "n_pairs_needing_prediction")},
                 indent=1, default=str))
