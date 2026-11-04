#!/usr/bin/env python3
"""tests/dbg_import.py — step-by-step import / load probe of the Z candidates in the mlts env (faulthandler on), to
locate env problems (segfaults, version clashes). Usage (sandbox): bash jobs/sbx.sh mlts tests/dbg_import.py <what>
  what = aimnet | aimnet_cpu | mace | torchvision | numba"""
import faulthandler
import sys
import time

faulthandler.enable()
what = sys.argv[1] if len(sys.argv) > 1 else "aimnet"
t0 = time.time()


def step(msg):
    print(f"[{time.time() - t0:7.2f}s] {msg}", flush=True)


step(f"python {sys.version.split()[0]} what={what}")
import numpy as np  # noqa: E402
import torch  # noqa: E402
step(f"torch {torch.__version__} cuda {torch.cuda.is_available()} numpy {np.__version__}")
if what == "numba":
    import numba
    import numba.cuda
    step(f"numba {numba.__version__} cuda available {numba.cuda.is_available()}")
    try:
        numba.cuda.detect()
    except Exception as e:  # noqa: BLE001
        step(f"detect failed {e}")
if what == "torchvision":
    try:
        import torchvision
        step(f"torchvision {torchvision.__version__}")
    except Exception as e:  # noqa: BLE001
        step(f"torchvision import failed: {type(e).__name__}: {e}")
if what in ("aimnet", "aimnet_cpu"):
    import numba.cuda
    step(f"numba.cuda.is_available {numba.cuda.is_available()}")
    if what == "aimnet_cpu":
        dev = "cpu"
    else:
        dev = "cuda"
        numba.cuda.is_available = lambda: True
    step("import aimnet.calculators")
    from aimnet.calculators import AIMNet2Calculator
    step("construct AIMNet2Calculator")
    calc = AIMNet2Calculator("aimnet2") if dev == "cuda" else None
    if calc is None:
        import aimnet.calculators.calculator as AC
        step(f"calculator module {AC.__file__}")
        calc = AIMNet2Calculator("aimnet2")
    step(f"device {getattr(calc, 'device', '?')}")
    d = dict(coord=np.array([[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]], np.float32),
             numbers=np.array([8, 1, 1]), charge=np.array([0.0], np.float32), mult=np.array([1.0], np.float32))
    step("evaluate")
    out = calc(d, forces=True)
    step(f"E {out['energy']} F {out['forces']}")
    out = calc(d, forces=True, hessian=True)
    step(f"hessian shape {tuple(out['hessian'].shape)}")
if what == "aimnet_dense":
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
    import mlts_bridge as B
    ev = B.evaluator("AIMNet2")
    s = ["O", "H", "H"]
    p = np.array([[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]]) + 0.01
    e, f, h = ev.evaluate(s, p, 0, 1, True, True)
    step(f"E {e} F nan {np.isnan(f).sum()} H nan {np.isnan(h).sum()} of {h.size}")
    step(f"H row0 {h[0]}")
    import torch
    T = torch
    z = T.tensor([[8, 1, 1]], dtype=T.int64, device="cuda")
    c = T.tensor(p.reshape(1, -1, 3), dtype=T.float32, device="cuda", requires_grad=True)
    q = T.tensor([0.0], dtype=T.float32, device="cuda")

    def efn(x):
        return ev.model(dict(coord=x, numbers=z, charge=q))["energy"].sum()
    try:
        Hf = T.autograd.functional.hessian(efn, c)
        step(f"functional.hessian nan {T.isnan(Hf).sum().item()}")
    except Exception as e2:  # noqa: BLE001
        step(f"functional.hessian failed {e2}")
    # finite differences of forces
    hfd = np.zeros((9, 9))
    for k in range(9):
        a, cc = divmod(k, 3)
        pp, pm = p.copy(), p.copy()
        pp[a, cc] += 1e-3
        pm[a, cc] -= 1e-3
        hfd[:, k] = -(ev.evaluate(s, pp, 0, 1, True, False)[1] - ev.evaluate(s, pm, 0, 1, True, False)[1]).ravel() / 2e-3
    step(f"FD H row0 {hfd[0]}")
if what == "mace":
    try:
        from mace.calculators import mace_off
        step("mace import ok")
        c = mace_off(model="medium", device="cuda", default_dtype="float64")
        step(f"mace_off ok {c}")
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        step(f"mace failed {type(e).__name__}: {e}")
step("done")
