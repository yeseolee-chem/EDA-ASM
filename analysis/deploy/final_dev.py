#!/usr/bin/env python3
"""final_dev.py — Phase W (train_tf.py final) under the recorded G-C deviation (gc_deviation.allowed()).

  python final_dev.py      GPU, env_d1fu2 (jobs/gpu_fu2.slurm): train_tf.cmd_final with only train_tf.gate_e_report
                           replaced in this process by the same checks minus 'G-C passed', which is replaced by the
                           null-check evidence; writes results/W_deviation.json next to W_report.json.
train_tf.py itself is not modified (its sha256 stays the one the C tests recorded)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dep_common as C  # noqa: E402
import gc_deviation as GD  # noqa: E402
import train_tf as TT  # noqa: E402

_EV: dict = {}


def gate_e_report_dev():
    ev = GD.allowed()                                  # K_predictions sha check + G-C pass or recorded deviation
    _EV.update(ev)
    return C.read_json(C.RES / "E_report.json")


if __name__ == "__main__":
    TT.gate_e_report = gate_e_report_dev
    rc = TT.cmd_final()
    C.write_json(C.RES / "W_deviation.json", dict(phase="W", rc=rc, **_EV))
    sys.exit(rc)
