"""gc_deviation.py — the recorded G-C deviation (results/DECISIONS.md 'G-C 영가설 검사 설계' and 'G-C 편차').

The lf G-C failed (E_report.json G_C.pass false). A seed-only replicate of LADDER g2 . S . TF also failed G-C
(results/NULL_gc.json null_pass false), so by the pre-registered null rule the gate is underpowered and the pipeline
proceeds with the failure recorded as a deviation. `allowed()` returns the evidence when exactly that holds on the
current files, and stops (exit 3) otherwise. Used only by final_dev.py and w/prereg_w_dev.py (no DEPLOY-MODEL file is
modified)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dep_common as C  # noqa: E402

NULL_JSON = C.RES / "NULL_gc.json"


def allowed():
    er, kp = C.RES / "E_report.json", C.RES / "K_predictions.parquet"
    if not (er.is_file() and kp.is_file() and NULL_JSON.is_file()):
        C.not_ready(f"need {er.name}, {kp.name} and {NULL_JSON.name}")
    e, n = C.read_json(er), C.read_json(NULL_JSON)
    if (e.get("inputs_sha256") or {}).get("K_predictions") != C.sha256(kp):
        C.stop("E_report.json was computed on another K_predictions.parquet")
    gc = e.get("G_C") or {}
    if gc.get("pass") is True:
        return dict(deviation=False, reason="G-C passed")
    if not (n.get("null_pass") is False and n.get("proceed") is True):
        C.stop(f"G-C failed and the null check does not allow proceeding (null_pass {n.get('null_pass')}, "
               f"proceed {n.get('proceed')})")
    return dict(deviation=True, lf_failed=gc.get("failed"), null_failed=(n.get("null_gc") or {}).get("failed"),
                null_json_sha256=C.sha256(NULL_JSON), e_report_sha256=C.sha256(er),
                rule="DECISIONS 'G-C 영가설 검사 설계': null fails G-C -> gate underpowered -> proceed, deviation recorded")
