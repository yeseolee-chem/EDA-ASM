#!/usr/bin/env python3
"""tests/a_freq_bench.py — A2-freq measurement (GPU sandbox, MLTENV=mlts): per-Hessian cost of the three Freq routes on
label TS geometries, with one model (default the zero-shot MACE-OMOL-0 stand-in):
  analytic  MLIP autograd Hessian (mlts_bridge MACEWrap hessian='analytic', the A-1 / A2 default)
  fd        MLIP central differences, all 6N displaced graphs batched through the model (MACEWrap hessian='fd')
  numfreq   ORCA `! ExtOpt NumFreq` through the server (wall, client calls, GPU-s)
Output: <MLTS_A_BASE>/reports/A2_freq_bench.json. Run:
  MLTS_A_BASE=$MLT/tmp/atest SBX_GPU=1 SBX_C=4 SBX_MEM=16G bash jobs/sbx.sh mlts tests/a_freq_bench.py [rxn ...]"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
os.environ.setdefault("MLTS_A_BASE", "/gpfs/tmp_cpu2/yeseo1ee/mlts/tmp/atest")
import mlt_common as M  # noqa: E402
import mlts_bridge as B  # noqa: E402

BASE = Path(os.environ["MLTS_A_BASE"])
rxns = [int(x) for x in sys.argv[1:]] or [1000312, 1001912, 1003634]
model = os.environ.get("A_BENCH_MODEL", "MACE-OMOL-0")
an = B.resolve_spec(model)
fd = dict(an, hessian="fd", name=an["name"] + "_fd")
rows = []
h = B.start_server(an, sock_path=str(B.local_tmp() / "mlts_atest_bench" / "an.sock"),
                   log_path=str(BASE / "a" / "logs" / "server_bench.log")) \
    if (B.local_tmp() / "mlts_atest_bench").mkdir(parents=True, exist_ok=True) is None else None
try:
    for rid in rxns:
        s, p = M.read_xyz(M.label_ts(rid))
        p = np.asarray(p, float)
        r = dict(rxn_id=rid, n_atoms=len(s))
        for tag, spec in (("analytic", an), ("fd", fd)):
            if tag == "fd":
                B.evaluator(fd).batch = int(os.environ.get("A_FD_BATCH", "8"))  # 64 OOMs on the shared 24 GB card
            import torch
            torch.cuda.empty_cache()
            B.direct_ev(spec, s, p, need_hessian=True)                       # warm-up / compile
            ts = []
            for _ in range(3):
                t = time.perf_counter()
                H = B.hessian(spec, s, p)
                ts.append(time.perf_counter() - t)
            r[tag] = dict(s_median=float(np.median(ts)), s_all=[round(x, 3) for x in ts],
                          imag=B.hessian_summary(H, s, p))
            r[f"H_{tag}"] = H
        r["max_abs_dH_au"] = float(np.abs(r.pop("H_analytic") - r.pop("H_fd")).max())
        # server analytic (what the workers pay, incl. JSON transport)
        sv = []
        for _ in range(2):
            t = time.perf_counter()
            _, dt = B.server_hessian(h.socket, s, p)
            sv.append((time.perf_counter() - t, dt))
        r["server_analytic"] = dict(wall_s=sv[-1][0], eval_s=sv[-1][1], first_eval_s=sv[0][1])
        r["fd_batch"] = B.evaluator(fd).batch
        w = B.local_tmp() / "mlts_atest_bench" / f"nf_{rid}"
        shutil.rmtree(w, ignore_errors=True)
        o = B.run_orca_ext(s, p, "NumFreq", an, h.socket, w, base="nf")
        r["numfreq"] = dict(wall_s=o["wall_s"], ext_calls=o["ext"]["n"], gpu_s=o["ext"]["eval_s"],
                            terminated=o["terminated"], n_imag=o.get("n_imag"), imag=o.get("imag_cm", [])[:2])
        shutil.rmtree(w, ignore_errors=True)
        rows.append(r)
        print(json.dumps(r), flush=True)
finally:
    if h is not None:
        h.shutdown()
rep = dict(model=model, gpu=B.device_info(), rows=rows, written=M.now(),
           note="per Hessian; analytic/fd = in-process evaluator time (3 repeats, median); server_analytic = one "
                "Hessian through the server as the autodE workers get it; numfreq = ORCA ExtOpt NumFreq through the "
                "same server (client log: calls and GPU-s)")
M.write_json(BASE / "reports" / "A2_freq_bench.json", rep)
print(json.dumps({r["rxn_id"]: dict(n=r["n_atoms"], an=round(r["analytic"]["s_median"], 2), fd=round(r["fd"]["s_median"], 2),
                                    srv=round(r["server_analytic"]["eval_s"], 2), nf_wall=round(r["numfreq"]["wall_s"], 1),
                                    nf_gpu=round(r["numfreq"]["gpu_s"], 1), dH=r["max_abs_dH_au"]) for r in rows}),
      flush=True)
