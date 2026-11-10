#!/usr/bin/env python3
"""final_m.py — CH3M 9: `train_m.py final <candidate>` behind the W gates the code review of 11-10 asked for (added after
the C jobs had started, so train_m.py itself — hashed into C_tests / unit fingerprints — is left unchanged):
  - results/EM_report.json was computed on the current results/KM_predictions.parquet (= deploy / ch3 W gates);
  - every K unit of the candidate is present with its 5 fold .pt files (exit 2 otherwise, instead of a crash in
    train_tf.choose_final on an empty set; the 25 .pt are what I-1 reads).
The EM_report sha256 is recorded in results/WM_gate.json next to the W outputs.

  python final_m.py <TF10|GNN10|GNN3>      GPU, env_d1fu2 (jobs/gpu_m.slurm)
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ch3m_common as C  # noqa: E402
import train_m as TM  # noqa: E402


def main(cand):
    if cand not in ("TF10", "GNN10", "GNN3"):
        C.stop(f"final {cand}?")
    er, kp = C.RES / "EM_report.json", C.RES / "KM_predictions.parquet"
    if not (er.is_file() and kp.is_file()):
        C.not_ready(f"{er} / {kp} missing")
    rep = C.read_json(er)
    if (rep.get("inputs_sha256") or {}).get("KM_predictions") != C.sha256(kp):
        C.stop("EM_report.json was computed on another KM_predictions.parquet")
    model, mode = (cand[:-2], 10) if cand.endswith("10") else (cand[:-1], 3)
    recs, missing = TM.load_units(model, mode)
    if missing:
        C.not_ready(f"K units missing: {missing}")
    lack = [f"{x}__s{j}.pt" for x in recs for j in range(len(TM.F.FINAL_SEEDS)) if not (C.UNITS / f"{x}__s{j}.pt").is_file()]
    if lack:
        C.stop(f"fold model files missing: {lack[:5]}")
    C.write_json(C.RES / "WM_gate.json", dict(candidate=cand, written=C.now(), em_report_sha256=C.sha256(er),
                                             km_predictions_sha256=C.sha256(kp), n_units=len(recs)))
    return TM.cmd_final(cand)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    sys.exit(main(sys.argv[1]))
