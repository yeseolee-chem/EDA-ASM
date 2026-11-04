#!/usr/bin/env python3
"""mlts_bridge.py — DEPLOY-V2 (MLTS) A-1: the ORCA <-> MLIP bridge (docs/specs/DEPLOY_V2_MLTS.md 6 A-1, C-1, C-4).

Design (orca-mlips 1.3.0 underneath, our own server loop and client):
  * model = a spec dict, e.g. {"backend": "mace", "model": "MACE-OMOL-0", "dtype": "float64"},
    {"backend": "mace", "weights": "$MLT/models/FT_0.model"}, {"backend": "aimnet2", "model": "aimnet2"},
    {"backend": "xtb"} (= plain ORCA `XTB2 ALPB(water)`, no ExtOpt: the G2 level baseline). resolve_spec() also takes
    a known name (MACE-OMOL-0, AIMNet2, MACE-OFF24, MACE-OFF23, xtb), 'FT_<k>', a .model path, a JSON string / file.
  * MACE models are loaded with orca_mlips_plugins.mlip_backends.MACEEvaluator (foundation names, URLs and local
    `--weights-file` checkpoints, i.e. what `orca-mlips-mace` loads; head / enable_cueq passed through). AIMNet2
    (released .jpt or a finetune.py export) runs in AIMNet2Dense, the numba-free batched layout of
    finetune.aim_predict (orca-mlips' aimnet2 backend needs numba CUDA, which these nodes do not provide).
    Fine-tuned models resolve from their finetune.py model card $MLT/models/<run>.json.
  * server: `python mlts_bridge.py serve` = one process per model spec on the GPU, one node-local Unix socket, the
    orca-mlips wire format (4-byte length + JSON; actions ping / stats / evaluate / shutdown). Our loop instead of
    orca-mlips' MLIPServer: listen backlog 512 (many ORCA processes), per-call GPU time returned and counted, the
    server exits when its parent (the GPU job's python) dies. Evaluations are serialised on the one GPU.
  * client: mlts_client.py (standard library only) installed next to the socket with the shebang
    '#!<mlts python> -IS'; ORCA runs it for every E/G (ProgExt), it writes <base>_EXT.engrad and logs its time.
  * Hessians: analytic from the MLIP (MACE autograd get_hessian / AIMNet2 3N backward passes) via the server,
    written as an ORCA .hess (write_orca_hess) for `InHess Read`; frequencies from the mass-weighted, translation/
    rotation-projected Hessian (freqs_cm). ORCA `ExtOpt NumFreq` (numerical, through the client) is the C-1
    cross-check only: ORCA spends ~0.47 s per displaced gradient there, so Recalc_Hess is done by restarting ORCA
    with a new MLIP Hessian (RECALC_DEFAULT 'analytic:30').
Units (C-4): MLIP eV, eV/A, eV/A^2  ->  ORCA Eh, Eh/bohr, Eh/bohr^2 (mlt_common EH2EV, BOHR2A).
Nothing here runs at import; python only on compute nodes (sbatch / jobs/sbx.sh)."""
from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import signal
import socket as _socket
import struct
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import mlt_common as C  # noqa: E402
import mlts_client as MC  # noqa: E402
import orca_io as OI  # noqa: E402

# ================================================================ constants and decisions
XTB_LEVEL = "XTB2 ALPB(water)"                          # G2 hmethod keyword set (analysis/d1_g2/g2_autode.py)
G2_OPTTS_BLOCK = "%geom\nCalc_Hess true\nRecalc_Hess 5\nMaxIter 200\nend\n"   # config_g2.yaml orca_optts_block
D1_OPTTS = dict(trust=-0.1, maxiter=100, recalc_every=30)                     # config_run.yaml (D1 label OptTS)
# decision A-1-recalc: D1's Recalc_Hess 30 is kept as "an exact Hessian every 30 cycles", but from the MLIP (analytic,
# ORCA restarted with InHess Read) instead of ORCA's numerical Hessian through ExtOpt, which costs ~0.47 s of ORCA
# time per displaced gradient (37.9 s for one 27-atom recalculation with AIMNet2, tests_c1 / C1_tests.json).
RECALC_DEFAULT = os.environ.get("MLTS_RECALC", "analytic:30")
IMAG_TOL_CM = 50.0      # n_imag counts modes below -50 cm-1 (D1 gate extra_imag_tol_cm = -50: smaller ones are noise)
EXT_MAXCORE_MB = 500
ORCA_TIMEOUT_S = int(os.environ.get("MLTS_ORCA_TIMEOUT_S", "7200"))
# foundation files: ~/.cache/<kind>/ (read only) or $MLT/cache/<kind>/ (the finetune.py fetch location; downloaded there)
FOUNDATION = {
    "MACE-OMOL-0": ("mace", "MACE-omol-0-extra-large-1024.model",          # = mace_omol('extra_large'), head omol
                    "https://github.com/ACEsuit/mace-foundations/releases/download/mace_omol_0/"
                    "MACE-omol-0-extra-large-1024.model"),
    "MACE-OFF24": ("mace", "MACE-OFF24_medium.model",
                   "https://github.com/ACEsuit/mace-off/raw/main/mace_off24/MACE-OFF24_medium.model"),
    "MACE-MH-1": ("mace", "mace-mh-1.model",
                  "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mh_1/mace-mh-1.model"),
    "aimnet2": ("aimnet", "aimnet2_wb97m_d3_0.jpt",
                "https://storage.googleapis.com/aimnetcentral/AIMNet2/aimnet2_wb97m_d3_0.jpt"),
}
_E = 1.0 / C.EH2EV
GRAD_PER_FORCE = -C.BOHR2A / C.EH2EV                   # F [eV/A] -> gradient [Eh/bohr]
HESS_AU_PER_EV = C.BOHR2A ** 2 / C.EH2EV               # [eV/A^2] -> [Eh/bohr^2]
_EH_J, _BOHR_M, _AMU_KG, _C_CM = 4.3597447222071e-18, 5.29177210903e-11, 1.66053906660e-27, 2.99792458e10
AU2CM = float(np.sqrt(_EH_J / (_BOHR_M ** 2 * _AMU_KG)) / (2 * np.pi * _C_CM))   # sqrt(Eh/(bohr^2 amu)) -> cm-1
MASS = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "Si": 28.085, "P": 30.974, "S": 32.06,
        "Cl": 35.45, "Br": 79.904, "I": 126.904, "B": 10.81}                     # ORCA .hess $atoms masses

KNOWN = {
    "MACE-OMOL-0": dict(backend="mace", model="MACE-OMOL-0", head="omol", dtype="float64"),
    "MACE-MH-1": dict(backend="mace", model="MACE-MH-1", head="omol", dtype="float64"),   # lead 11-04: added to Z
    "AIMNet2": dict(backend="aimnet2", model="aimnet2"),
    "MACE-OFF24": dict(backend="mace", model="MACE-OFF24", dtype="float64", no_si=True),
    "MACE-OFF23": dict(backend="mace", model="off:medium", dtype="float64", no_si=True),
    "xtb": dict(backend="xtb"),
}
ALIASES = {k.lower(): k for k in KNOWN}
ALIASES.update({"gfn2": "xtb", "gfn2-xtb": "xtb", "xtb2": "xtb", "gfn2-xtb-alpb-water": "xtb", "xtb2-alpb-water": "xtb",
                "mace-off24-medium": "MACE-OFF24", "mace-off23-medium": "MACE-OFF23", "aimnet-2": "AIMNet2",
                "mace-mh-1-omol": "MACE-MH-1", "mh-1": "MACE-MH-1"})
MODELS_DIR = C.MLT / "models"                          # fine-tuned models + model cards (finetune.py export)


# ================================================================ specs
def resolve_spec(s):
    """model spec dict (with 'name') from a dict / JSON string / .json file / known name / FT_<k> / .model path."""
    if isinstance(s, dict):
        d = dict(s)
    else:
        t = str(s).strip()
        if t.startswith("{"):
            d = json.loads(t)
        elif t.endswith(".json") and Path(t).is_file():
            d = json.loads(Path(t).read_text())
        elif t.lower() in ALIASES:
            d = dict(KNOWN[ALIASES[t.lower()]], name=ALIASES[t.lower()])
        elif (MODELS_DIR / f"{t}.json").is_file():                 # finetune.py model card
            d = _from_card(t, json.loads((MODELS_DIR / f"{t}.json").read_text()))
        elif re.fullmatch(r"FT_[A-Za-z0-9_.-]+", t):
            p = MODELS_DIR / f"{t}.jpt"
            d = (dict(backend="aimnet2", weights=str(p), name=t) if p.is_file() else
                 dict(backend="mace", weights=str(MODELS_DIR / f"{t}.model"), dtype="float64", name=t))
        elif t.endswith(".model"):
            d = dict(backend="mace", weights=str(Path(t).resolve()), dtype="float64", name=Path(t).stem)
        elif t.endswith(".jpt"):
            d = dict(backend="aimnet2", weights=str(Path(t).resolve()), name=Path(t).stem)
        else:
            raise ValueError(f"unknown model spec {t!r}")
    d.setdefault("backend", "mace")
    if d["backend"] not in ("mace", "aimnet2", "xtb"):
        raise ValueError(f"unknown backend {d['backend']!r}")
    if d["backend"] == "mace":
        d.setdefault("dtype", "float64")
    if "name" not in d:
        d["name"] = spec_name(d)
    return d


def spec_name(spec):
    if isinstance(spec, dict) and spec.get("name"):
        return str(spec["name"])
    d = spec if isinstance(spec, dict) else resolve_spec(spec)
    if d.get("backend") == "xtb":
        return "xtb"
    if d.get("weights"):
        return Path(d["weights"]).stem
    return str(d.get("model", d.get("backend")))


def is_xtb(spec):
    return resolve_spec(spec)["backend"] == "xtb"


def covers(spec, symbols):
    """False if the model has no parameters for one of the elements (MACE-OFF: no Si)."""
    d = resolve_spec(spec)
    return not (d.get("no_si") and "Si" in set(symbols))


def _from_card(name, card):
    """spec from a finetune.py model card ($MLT/models/<run>.json: kind mace|aimnet, file, dtype, heads)."""
    if card.get("kind") == "aimnet":
        return dict(backend="aimnet2", weights=card["file"], name=name, card=str(MODELS_DIR / f"{name}.json"))
    d = dict(backend="mace", weights=card["file"], dtype=card.get("dtype", "float64"), name=name,
             card=str(MODELS_DIR / f"{name}.json"))
    heads = list(card.get("heads") or [])
    if len(heads) > 1:
        d["head"] = card.get("head") or heads[-1]
    return d


# ================================================================ in-process models
# MACE: orca_mlips_plugins.mlip_backends.MACEEvaluator (foundation aliases, URLs, local checkpoints; MACECalculator
#   kwargs head / enable_cueq through calc_kwargs). AIMNet2: AIMNet2Dense below (lead 11-04: numba sees no CUDA on
#   these nodes, so aimnet's AIMNet2Calculator / orca-mlips' aimnet2 backend cannot build their neighbour list on the
#   GPU; the TorchScript model is called in the batched dense layout of finetune.aim_predict instead).
_EV = {}
_EV_MU = threading.Lock()


def _download(url, dest):
    dest = Path(dest)
    if dest.is_file() and dest.stat().st_size > 1000:
        return dest
    import urllib.request
    C.assert_write_ok(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.part{os.getpid()}")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    if tmp.stat().st_size < 1000:
        raise RuntimeError(f"download of {url} failed")
    os.replace(tmp, dest)
    return dest


def foundation_file(key):
    """local file of a foundation model (FOUNDATION): ~/.cache/<kind>, aimnet's assets dir, $MLT/cache/<kind>
    (downloaded there if absent, as finetune.fetch)."""
    kind, fname, url = FOUNDATION[key]
    alt = [Path.home() / ".cache" / kind / fname, C.MLT / "cache" / kind / fname]
    if kind == "aimnet":
        sp = importlib.util.find_spec("aimnet")
        if sp and sp.submodule_search_locations:
            alt.insert(0, Path(list(sp.submodule_search_locations)[0]) / "calculators" / "assets" / fname)
    for p in alt:
        if p.is_file() and p.stat().st_size > 1000:
            return p
    return _download(url, C.MLT / "cache" / kind / fname)


def _mace_model(spec):
    if spec.get("weights"):
        p = Path(os.path.expandvars(str(spec["weights"])))
        if not p.is_file():
            raise FileNotFoundError(f"MACE weights {p} missing")
        return str(p)
    m = str(spec.get("model", "MACE-OMOL-0"))
    return str(foundation_file(m)) if m in FOUNDATION else m


class AIMNet2Dense:
    """AIMNet2 TorchScript model (released aimnet2_wb97m_d3_0.jpt or a finetune.py export, absolute energies in eV)
    called as finetune.aim_predict does: coord (B, N, 3) float32, numbers (B, N), charge (B,) -> no neighbour list,
    no numba. Forces by autograd. The dense layout's second derivative is NaN (self pairs at zero distance,
    tests/dbg_import.py aimnet_dense), so the Hessian is a central difference of the dense forces with all 6N
    displaced geometries in batches on the GPU (HESS_FD_STEP_A). Same evaluate() interface as the orca-mlips
    evaluators (eV, eV/A, eV/A^2)."""
    HESS_FD_STEP_A = 2e-3
    HESS_BATCH = 192
    hessian_method = "central differences of dense AIMNet2 forces, h = 2e-3 A, batched"

    def __init__(self, path, device="auto"):
        import torch
        from ase.data import atomic_numbers
        self.torch = torch
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.path = str(path)
        self.model = torch.jit.load(self.path, map_location=self.device)
        self._Z = atomic_numbers

    def _ef(self, X, z, charge, forces=True):
        """X (B, N, 3) A -> energies (B,) eV float64 numpy, forces (B, N, 3) eV/A float64 numpy."""
        T = self.torch
        c = T.tensor(np.asarray(X, float), dtype=T.float32, device=self.device, requires_grad=forces)
        zz = T.tensor(np.repeat(np.asarray(z)[None], len(X), 0), dtype=T.int64, device=self.device)
        q = T.full((len(X),), float(charge), dtype=T.float32, device=self.device)
        e = self.model(dict(coord=c, numbers=zz, charge=q))["energy"]
        if not forces:
            return e.detach().double().cpu().numpy(), None
        g = T.autograd.grad(e.sum(), c)[0]
        return e.detach().double().cpu().numpy(), (-g).detach().double().cpu().numpy()

    def evaluate(self, symbols, coords_ang, charge, multiplicity, need_forces=True, need_hessian=False,
                 hessian_mode="Analytical", hessian_step=None):
        z = [self._Z[s] for s in symbols]
        x = np.asarray(coords_ang, float).reshape(-1, 3)
        e, f = self._ef(x[None], z, charge, forces=bool(need_forces or need_hessian))
        H = None
        if need_hessian:
            h = float(self.HESS_FD_STEP_A)
            n3 = x.size
            D = np.zeros((2 * n3, n3))
            D[np.arange(n3), np.arange(n3)] = h
            D[n3 + np.arange(n3), np.arange(n3)] = -h
            Xd = x.ravel()[None] + D
            F = np.concatenate([self._ef(Xd[k:k + self.HESS_BATCH].reshape(-1, len(z), 3), z, charge)[1]
                                for k in range(0, 2 * n3, self.HESS_BATCH)]).reshape(2 * n3, n3)
            H = -(F[:n3] - F[n3:]).T / (2 * h)                    # H[:, k] = -dF/dx_k
            H = 0.5 * (H + H.T)
        return float(e[0]), (None if f is None else f[0]), H


_NVRTC = {}


def _preload_nvrtc():
    """cuequivariance_ops (MACE enable_cueq) dlopens libnvrtc.so.12, which the pip wheel nvidia-cuda-nvrtc-cu12 puts
    outside the loader path ('Error while loading libcue_ops.so: libnvrtc.so.12', 11-04). Loading it once with
    RTLD_GLOBAL before mace is imported makes the soname resolvable in this process (no env change)."""
    if "done" in _NVRTC:
        return _NVRTC["done"]
    _NVRTC["done"] = None
    try:
        import ctypes
        sp = importlib.util.find_spec("nvidia.cuda_nvrtc")
        if sp and sp.submodule_search_locations:
            p = Path(list(sp.submodule_search_locations)[0]) / "lib" / "libnvrtc.so.12"
            if p.is_file():
                ctypes.CDLL(str(p), mode=ctypes.RTLD_GLOBAL)
                _NVRTC["done"] = str(p)
    except Exception:                                          # noqa: BLE001
        pass
    return _NVRTC["done"]


class MACEWrap:
    """orca-mlips MACEEvaluator for E / F; Hessian analytic (MACE autograd, spec hessian='analytic', default) or a
    central difference of MACE forces with all 6N displaced graphs batched through the model (spec hessian='fd',
    h = 1e-3 A; the batching of finetune.mace_predict)."""
    FD_STEP_A = 1e-3

    def __init__(self, ev, hessian="analytic", batch=64):
        self.ev, self.hess_mode, self.batch = ev, str(hessian or "analytic"), int(batch)
        self.hessian_method = ("MACE analytic (autograd)" if self.hess_mode == "analytic" else
                               f"central differences of MACE forces, h = {self.FD_STEP_A} A, batched")

    def _fd_hessian(self, symbols, x, charge, mult):
        import torch
        from ase import Atoms
        from mace import data as mdata
        from mace.tools import torch_geometric, torch_tools
        calc = self.ev._calc
        model = calc.models[0]
        n, n3, h = len(symbols), x.size, self.FD_STEP_A
        D = np.zeros((2 * n3, n3))
        D[np.arange(n3), np.arange(n3)] = h
        D[n3 + np.arange(n3), np.arange(n3)] = -h
        keyspec = mdata.KeySpecification(info_keys=calc.info_keys, arrays_keys=calc.arrays_keys)
        data = []
        with torch_tools.default_dtype(calc.default_dtype):
            for row in x.ravel()[None] + D:
                a = Atoms(symbols, positions=row.reshape(n, 3))
                a.info["charge"], a.info["spin"] = int(charge), int(mult)
                cfg = mdata.config_from_atoms(a, key_specification=keyspec, head_name=calc.head)
                data.append(mdata.AtomicData.from_config(cfg, z_table=calc.z_table, cutoff=calc.r_max,
                                                         heads=calc.available_heads))
        F, bs = [], self.batch
        k = 0
        while k < len(data):
            try:
                b = next(iter(torch_geometric.dataloader.DataLoader(dataset=data[k:k + bs], batch_size=bs,
                                                                     shuffle=False, drop_last=False))).to(calc.device)
                o = model(b.to_dict(), compute_force=True, training=False)
                F.append(o["forces"].detach().double().cpu().numpy().reshape(-1, n3))
                k += bs
            except RuntimeError as e:          # torch.cuda.OutOfMemoryError, or TorchScript's plain RuntimeError
                if not (isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()):
                    raise
                b = o = None                   # drop the failed batch before freeing the cache
                torch.cuda.empty_cache()
                if bs == 1:
                    raise
                bs = max(1, bs // 2)
        F = np.concatenate(F)
        H = -(F[:n3] - F[n3:]).T / (2 * h)
        return 0.5 * (H + H.T)

    def evaluate(self, symbols, coords_ang, charge, multiplicity, need_forces=True, need_hessian=False,
                 hessian_mode="Analytical", hessian_step=1e-3):
        if not need_hessian or self.hess_mode == "analytic":
            return self.ev.evaluate(symbols=symbols, coords_ang=coords_ang, charge=charge, multiplicity=multiplicity,
                                    need_forces=need_forces, need_hessian=need_hessian, hessian_mode="Analytical",
                                    hessian_step=hessian_step)
        e, f, _ = self.ev.evaluate(symbols=symbols, coords_ang=coords_ang, charge=charge, multiplicity=multiplicity,
                                   need_forces=True, need_hessian=False, hessian_mode="Analytical", hessian_step=1e-3)
        return e, f, self._fd_hessian(list(symbols), np.asarray(coords_ang, float).reshape(-1, 3), charge, multiplicity)


def _make_evaluator(spec):
    dev = spec.get("device", "auto")
    ck = dict(spec.get("calc_kwargs") or {})
    if spec["backend"] == "mace":
        _preload_nvrtc()
        from orca_mlips_plugins import mlip_backends as MB
        if spec.get("head"):
            ck["head"] = spec["head"]
        if spec.get("enable_cueq"):
            ck["enable_cueq"] = True
        ev = MB.MACEEvaluator(model=_mace_model(spec), device=dev, default_dtype=spec.get("dtype", "float64"),
                              calc_kwargs=ck)
        return MACEWrap(ev, hessian=spec.get("hessian", "analytic"))
    if spec["backend"] == "aimnet2":
        p = spec.get("weights") or foundation_file(str(spec.get("model", "aimnet2")).lower())
        return AIMNet2Dense(os.path.expandvars(str(p)), device=dev)
    raise ValueError(f"no in-process evaluator for backend {spec['backend']}")


def _key(spec):
    return json.dumps({k: v for k, v in sorted(spec.items()) if k != "name"}, sort_keys=True, default=str)


def evaluator(spec):
    spec = resolve_spec(spec)
    k = _key(spec)
    with _EV_MU:
        if k not in _EV:
            _EV[k] = _make_evaluator(spec)
        return _EV[k]


def device_info():
    try:
        import torch
        if torch.cuda.is_available():
            return dict(device="cuda", gpu=torch.cuda.get_device_name(0), torch=torch.__version__)
        return dict(device="cpu", torch=torch.__version__)
    except Exception as exc:                                   # noqa: BLE001
        return dict(device="?", error=str(exc)[:200])


def direct_ev(spec, symbols, pos_A, charge=0, mult=1, need_hessian=False):
    """in-process MLIP call: (E eV, F eV/A (N,3), H eV/A^2 (3N,3N) or None)."""
    ev = evaluator(spec)
    e, f, h = ev.evaluate(symbols=list(symbols), coords_ang=np.asarray(pos_A, float), charge=int(charge),
                          multiplicity=int(mult), need_forces=True, need_hessian=bool(need_hessian),
                          hessian_mode="Analytical", hessian_step=1e-3)
    return float(e), np.asarray(f, float).reshape(-1, 3), (None if h is None else np.asarray(h, float))


def direct_engrad(spec, symbols, pos_A, charge=0, mult=1):
    """(E Eh, gradient Eh/bohr (N,3)) in this process; xtb = ORCA XTB2 ALPB(water) EnGrad in node-local tmp."""
    spec = resolve_spec(spec)
    if spec["backend"] == "xtb":
        return xtb_engrad(symbols, pos_A, charge, mult)
    e, f, _ = direct_ev(spec, symbols, pos_A, charge, mult)
    return e * _E, f * GRAD_PER_FORCE


def hessian(spec, symbols, pos_A, socket=None, charge=0, mult=1):
    """Cartesian Hessian in Eh/bohr^2 (3N,3N, symmetrised): MLIP analytic (in-process, or through the server when
    socket is given); xtb = ORCA XTB2 ALPB(water) Freq (.hess)."""
    spec = resolve_spec(spec)
    if spec["backend"] == "xtb":
        return xtb_hessian(symbols, pos_A, charge, mult)
    if socket:
        return server_hessian(socket, symbols, pos_A, charge, mult)[0]
    _, _, h = direct_ev(spec, symbols, pos_A, charge, mult, need_hessian=True)
    h = h.reshape(3 * len(symbols), 3 * len(symbols)) * HESS_AU_PER_EV
    return 0.5 * (h + h.T)


# ================================================================ server (one per model spec, GPU job)
_HDR = struct.Struct("!I")


def _send(conn, obj):
    data = json.dumps(obj).encode()
    conn.sendall(_HDR.pack(len(data)) + data)


def _recv(conn):
    h = MC._recv_exact(conn, _HDR.size)
    if h is None:
        return None
    n = _HDR.unpack(h)[0]
    if n > 512 * 1024 * 1024:
        raise ValueError("message too large")
    b = MC._recv_exact(conn, n)
    return None if b is None else json.loads(b.decode())


def _pid_alive(pid):
    try:
        os.kill(int(pid), 0)
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def serve(spec, sock_path, parent_pid=None, idle_s=None):
    """server main loop (runs in its own process: `mlts_bridge.py serve`)."""
    spec = resolve_spec(spec)
    t0 = time.time()
    ev = evaluator(spec)
    w = (["O", "H", "H"], [[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]])
    ev.evaluate(symbols=w[0], coords_ang=np.array(w[1]), charge=0, multiplicity=1, need_forces=True,
                need_hessian=False, hessian_mode="Analytical", hessian_step=1e-3)          # warm-up (CUDA init)
    st = dict(spec=spec, pid=os.getpid(), host=_socket.gethostname(), started=C.now(), load_s=round(time.time() - t0, 2),
              n_eval=0, n_hess=0, n_err=0, eval_s=0.0, hess_s=0.0, evaluator=type(ev).__name__, **device_info())
    if os.path.exists(sock_path):
        if health(sock_path):
            raise SystemExit(f"another server answers at {sock_path}")
        os.unlink(sock_path)
    srv = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
    srv.bind(sock_path)
    os.chmod(sock_path, 0o600)
    srv.listen(512)
    srv.settimeout(1.0)
    run = {"on": True}

    def _sig(*_):
        run["on"] = False
    for sg in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sg, _sig)
    print(f"[mlts-server] ready {sock_path} {json.dumps({k: v for k, v in st.items() if k != 'spec'})}", flush=True)
    last = time.time()
    try:
        while run["on"]:
            if parent_pid and not _pid_alive(parent_pid):
                print("[mlts-server] parent gone, exiting", flush=True)
                break
            if idle_s and time.time() - last > float(idle_s):
                print("[mlts-server] idle timeout, exiting", flush=True)
                break
            try:
                conn, _ = srv.accept()
            except _socket.timeout:
                continue
            try:
                conn.settimeout(900)
                req = _recv(conn)
                if req is None:
                    continue
                last = time.time()
                a = req.get("action")
                if a == "ping":
                    _send(conn, {"status": "ok", "message": "pong"})
                elif a == "stats":
                    _send(conn, dict(st, status="ok", uptime_s=round(time.time() - t0, 1)))
                elif a == "shutdown":
                    _send(conn, {"status": "ok", "message": "shutting down"})
                    run["on"] = False
                elif a == "evaluate":
                    hes = bool(req.get("need_hessian", False))
                    t = time.perf_counter()
                    try:
                        e, f, h = ev.evaluate(symbols=req["symbols"], coords_ang=np.asarray(req["coords_ang"], float),
                                              charge=int(req.get("charge", 0)), multiplicity=int(req.get("multiplicity", 1)),
                                              need_forces=bool(req.get("need_forces", True)), need_hessian=hes,
                                              hessian_mode="Analytical", hessian_step=1e-3)
                        dt = time.perf_counter() - t
                        resp = {"status": "ok", "energy_ev": float(e),
                                "forces_ev_ang": None if f is None else np.asarray(f, float).tolist(),
                                "hessian_ev_ang2": None if h is None else np.asarray(h, float).reshape(
                                    3 * len(req["symbols"]), -1).tolist(), "t_eval_s": dt}
                        if hes:
                            st["n_hess"] += 1
                            st["hess_s"] += dt
                        else:
                            st["n_eval"] += 1
                            st["eval_s"] += dt
                    except Exception as exc:                   # noqa: BLE001
                        st["n_err"] += 1
                        traceback.print_exc()
                        resp = {"status": "error", "message": f"{type(exc).__name__}: {exc}"[:500]}
                    _send(conn, resp)
                else:
                    _send(conn, {"status": "error", "message": f"unknown action {a!r}"})
            except Exception:                                 # noqa: BLE001
                traceback.print_exc()
            finally:
                conn.close()
    finally:
        srv.close()
        try:
            os.unlink(sock_path)
        except OSError:
            pass
        print(f"[mlts-server] stopped n_eval={st['n_eval']} eval_s={st['eval_s']:.1f} n_hess={st['n_hess']} "
              f"hess_s={st['hess_s']:.1f}", flush=True)


def jobtag():
    j, a = os.environ.get("SLURM_JOB_ID"), os.environ.get("SLURM_ARRAY_TASK_ID")
    return f"{j}_{a}" if j and a else (j or f"pid{os.getpid()}")


def local_tmp():
    """node-local scratch ($TMPDIR, else /tmp)."""
    p = Path(os.environ.get("TMPDIR") or "/tmp")
    return p if p.is_dir() and os.access(p, os.W_OK) else Path("/tmp")


def default_socket(name):
    d = local_tmp() / f"mlts_{jobtag()}"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', name)[:40]}_{os.getpid()}.sock"
    if len(str(p)) > 100:                                      # AF_UNIX path limit (108)
        p = Path("/tmp") / f"mlts_{abs(hash(str(p))) % 10 ** 10}.sock"
    return str(p)


class ServerHandle:
    def __init__(self, spec, socket, proc, log):
        self.spec, self.socket, self.proc, self.log = spec, socket, proc, log

    def stats(self):
        return stats(self.socket)

    def shutdown(self):
        shutdown_server(self)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.shutdown()


def start_server(spec, sock_path=None, log_path=None, timeout_s=1200):
    """start `mlts_bridge.py serve` for this spec (GPU of this job), wait until it answers; None for xtb."""
    spec = resolve_spec(spec)
    if spec["backend"] == "xtb":
        return None
    name = spec_name(spec)
    sock_path = sock_path or default_socket(name)
    log_path = Path(log_path or C.MLT / "logs" / f"mlts_server_{jobtag()}_{re.sub(r'[^A-Za-z0-9_.-]', '_', name)}.log")
    C.assert_write_ok(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(HERE / "mlts_bridge.py"), "serve", "--spec", json.dumps(spec), "--socket", sock_path,
           "--parent-pid", str(os.getpid())]
    lf = open(log_path, "a")
    proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=str(HERE))
    lf.close()
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            tail = Path(log_path).read_text(errors="replace")[-3000:]
            raise RuntimeError(f"server for {name} exited (rc {proc.returncode}):\n{tail}")
        if os.path.exists(sock_path) and health(sock_path, timeout=2.0):
            return ServerHandle(spec, sock_path, proc, str(log_path))
        time.sleep(1.0)
    proc.terminate()
    raise RuntimeError(f"server for {name} not ready after {timeout_s} s (log {log_path})")


def health(sock_path, timeout=5.0):
    """the server's stats dict if it answers, else None."""
    try:
        r = MC.call(sock_path, {"action": "stats"}, timeout=timeout, tries=1)
        return r if r.get("status") == "ok" else None
    except Exception:                                          # noqa: BLE001
        return None


def stats(sock_path):
    return health(sock_path)


def shutdown_server(h, timeout_s=60):
    sock = h.socket if isinstance(h, ServerHandle) else str(h)
    try:
        MC.call(sock, {"action": "shutdown"}, timeout=10, tries=1)
    except Exception:                                          # noqa: BLE001
        pass
    if isinstance(h, ServerHandle) and h.proc is not None:
        try:
            h.proc.wait(timeout_s)
        except subprocess.TimeoutExpired:
            h.proc.kill()


def server_engrad(sock, symbols, pos_A, charge=0, mult=1):
    """(E Eh, gradient Eh/bohr (N,3), GPU seconds) through the server."""
    r = MC.call(sock, dict(action="evaluate", symbols=list(symbols), coords_ang=np.asarray(pos_A, float).tolist(),
                           charge=int(charge), multiplicity=int(mult), need_forces=True, need_hessian=False))
    if r.get("status") != "ok":
        raise RuntimeError(f"server error: {r.get('message')}")
    return r["energy_ev"] * _E, np.asarray(r["forces_ev_ang"], float) * GRAD_PER_FORCE, float(r.get("t_eval_s", 0))


def server_hessian(sock, symbols, pos_A, charge=0, mult=1):
    """(Hessian Eh/bohr^2 (3N,3N) symmetrised, GPU seconds) through the server."""
    r = MC.call(sock, dict(action="evaluate", symbols=list(symbols), coords_ang=np.asarray(pos_A, float).tolist(),
                           charge=int(charge), multiplicity=int(mult), need_forces=True, need_hessian=True))
    if r.get("status") != "ok":
        raise RuntimeError(f"server error: {r.get('message')}")
    n3 = 3 * len(symbols)
    h = np.asarray(r["hessian_ev_ang2"], float).reshape(n3, n3) * HESS_AU_PER_EV
    return 0.5 * (h + h.T), float(r.get("t_eval_s", 0))


# ================================================================ ORCA side
def client_exe(d=None):
    """install mlts_client.py as an executable with '#!<this python> -IS' in a node-local folder (idempotent)."""
    d = Path(d or local_tmp() / f"mlts_bin_{os.getuid()}")
    d.mkdir(parents=True, exist_ok=True)
    exe = d / "mlts_client"
    src = (HERE / "mlts_client.py").read_text()
    body = src.split("\n", 1)[1] if src.startswith("#!") else src
    she = f"#!{sys.executable} -IS\n"
    if len(she) > 120:                                         # kernel shebang limit: go through bash
        exe_py = d / "mlts_client.py"
        if not exe_py.is_file() or exe_py.read_text() != src:
            t = d / f".c.tmp{os.getpid()}_{threading.get_ident()}"
            t.write_text(src)
            os.replace(t, exe_py)
        txt = f'#!/bin/bash\nexec "{sys.executable}" -IS "{exe_py}" "$@"\n'
    else:
        txt = she + body
    if exe.is_file() and exe.read_text() == txt and os.access(exe, os.X_OK):
        return exe
    t = d / f".mlts_client.tmp{os.getpid()}_{threading.get_ident()}"
    t.write_text(txt)
    os.chmod(t, 0o755)
    os.replace(t, exe)
    return exe


def ext_method_block(spec, sock, log=None):
    """'%method ProgExt "<client>" Ext_Params "--socket <sock>" end' for `! ExtOpt` (MLIP specs only)."""
    if is_xtb(spec):
        raise ValueError("xtb runs plain ORCA XTB2 ALPB(water), no ExtOpt")
    exe = client_exe(Path(sock).parent)
    par = f"--socket {sock}" + (f" --log {log}" if log else "")
    return f'%method\n  ProgExt "{exe}"\n  Ext_Params "{par}"\nend\n'


def _orca_env():
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env["PATH"] = f"{C.ORCA_DIR}:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{C.ORCA_DIR}:{env.get('LD_LIBRARY_PATH', '')}"
    return env


def xyz_lines(symbols, pos_A):
    return "".join(f"{s:<3s}{x:16.10f}{y:16.10f}{z:16.10f}\n" for s, (x, y, z) in zip(symbols, np.asarray(pos_A, float)))


_RE_FSPE = re.compile(r"^FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", re.M)
_RE_INHESS = re.compile(r"InHessName:\s*(\S+)\s+The file <([^>]+)> is opened as a \.hess file")


def last_coords(out_text):
    k = out_text.rfind("CARTESIAN COORDINATES (ANGSTROEM)")
    if k < 0:
        return None
    lines = out_text[k:].splitlines()[2:]
    pos = []
    for ln in lines:
        t = ln.split()
        if len(t) != 4:
            break
        try:
            pos.append([float(t[1]), float(t[2]), float(t[3])])
        except ValueError:
            break
    return np.array(pos) if pos else None


def hess_freqs(path):
    """$vibrational_frequencies of an ORCA .hess (cm-1, ORCA order); [] if absent."""
    with OI._open(path) as f:
        lines = f.read().splitlines()
    i = next((k for k, ln in enumerate(lines) if ln.strip() == "$vibrational_frequencies"), None)
    if i is None:
        return []
    n = int(lines[i + 1].split()[0])
    return [float(lines[i + 2 + k].split()[1]) for k in range(n)]


def ext_log_stats(path):
    """client log (wall_ms eval_ms n_atoms grad per call) -> counts and medians."""
    p = Path(path)
    if not p.is_file():
        return dict(n=0, eval_s=0.0, wall_s=0.0)
    a = np.array([[float(x) for x in ln.split()[:2]] for ln in p.read_text().splitlines() if ln.strip()], float)
    if not len(a):
        return dict(n=0, eval_s=0.0, wall_s=0.0)
    return dict(n=int(len(a)), eval_s=float(a[:, 1].sum() / 1e3), wall_s=float(a[:, 0].sum() / 1e3),
                eval_ms_median=float(np.median(a[:, 1])), client_ms_median=float(np.median(a[:, 0])),
                client_overhead_ms_median=float(np.median(a[:, 0] - a[:, 1])))


def parse_run(work, base, symbols, pos_in):
    work = Path(work)
    outp = work / f"{base}.out"
    txt = outp.read_text(errors="replace") if outp.is_file() else ""
    e = _RE_FSPE.findall(txt)
    res = dict(terminated="ORCA TERMINATED NORMALLY" in txt[-4000:],
               converged="THE OPTIMIZATION HAS CONVERGED" in txt,
               max_iter_hit=bool(re.search(r"reached the maximum number of optimization cycles", txt, re.I)),
               n_cycles=txt.count("GEOMETRY OPTIMIZATION CYCLE"),
               n_energies=len(e), energy_eh=float(e[-1]) if e else None,
               inhess_read=[m.group(2) for m in _RE_INHESS.finditer(txt)][:4],
               errors=[ln.strip() for ln in txt.splitlines() if re.search(r"ERROR|aborting|Abort", ln)][:5])
    pos = None
    xyz = work / f"{base}.xyz"
    if res["n_cycles"] and xyz.is_file():
        s, p = C.read_xyz(xyz)
        if list(s) == list(symbols):
            pos = np.array(p, float)
    if pos is None and res["n_cycles"]:
        pos = last_coords(txt)
    res["pos_A"] = np.asarray(pos_in, float) if pos is None else pos
    eg = work / f"{base}.engrad"
    if eg.is_file():
        try:
            d = OI.read_engrad(eg)
            res["grad_eh_bohr"] = d["grad_eh_bohr"]
            res["engrad_energy_eh"] = d["energy_eh"]
            res["engrad_pos_A"] = d["pos_bohr"] * C.BOHR2A
        except Exception as exc:                               # noqa: BLE001
            res["engrad_error"] = str(exc)[:200]
    hs = work / f"{base}.hess"
    if hs.is_file():
        fr = hess_freqs(hs)
        res["freqs_cm"] = fr
        if fr:
            res["imag_cm"] = sorted(f for f in fr if f < -10.0)
            res["n_imag"] = int(sum(f < -IMAG_TOL_CM for f in fr))
    res["out_tail"] = "\n".join(txt.splitlines()[-25:])
    return res


def run_orca_ext(symbols, pos_A, runtype, spec, socket, workdir, geom_block="", inhess=None, nprocs=1, base="ext",
                 maxcore=None, timeout_s=None, extra="", charge=0, mult=1):
    """one ORCA run: `! ExtOpt <runtype>` through the server (MLIP specs) or `! XTB2 ALPB(water) <runtype>` (xtb).
    runtype: EnGrad | Opt | OptTS | NumFreq | 'OptTS NumFreq' ... ('Freq' becomes NumFreq for ExtOpt).
    Returns dict(rc, wall_s, terminated, converged, n_cycles, energy_eh, pos_A (final), grad_eh_bohr (if .engrad),
    freqs_cm / imag_cm / n_imag (if .hess), ext = client-call stats, inhess_read, out_tail)."""
    spec = resolve_spec(spec)
    xtb = spec["backend"] == "xtb"
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    rt = runtype.strip()
    if not xtb:
        rt = re.sub(r"(?<![A-Za-z])Freq\b", "NumFreq", rt)
    txt = (f"! {XTB_LEVEL} {rt}\n" if xtb else f"! ExtOpt {rt}\n")
    if not xtb:
        txt += ext_method_block(spec, socket)
    if inhess:
        src = Path(inhess)
        if src.resolve().parent != work.resolve():
            shutil.copy(src, work / src.name)
    txt += geom_block + extra
    txt += f"%maxcore {int(maxcore or (EXT_MAXCORE_MB if not xtb else 1000))}\n"
    if int(nprocs) > 1:
        txt += f"%pal nprocs {int(nprocs)}\nend\n"
    txt += f"*xyz {int(charge)} {int(mult)}\n" + xyz_lines(symbols, pos_A) + "*\n"
    (work / f"{base}.inp").write_text(txt)
    log = work / "mlts_ext.log"
    if log.exists():
        log.unlink()
    t0 = time.time()
    rc, timed_out = None, False
    with open(work / f"{base}.out", "w") as fo:
        try:
            rc = subprocess.run([str(C.ORCA), f"{base}.inp"], cwd=work, stdout=fo, stderr=subprocess.STDOUT,
                                env=_orca_env(), timeout=timeout_s or ORCA_TIMEOUT_S).returncode
        except subprocess.TimeoutExpired:
            timed_out = True
    res = parse_run(work, base, symbols, pos_A)
    res.update(rc=rc, timed_out=timed_out, wall_s=time.time() - t0, ext=ext_log_stats(log), runtype=rt,
               input_head=txt.split("*xyz")[0])
    return res


# ================================================================ Hessian files and frequencies
def masses(symbols):
    return np.array([MASS[s] for s in symbols], float)


def write_orca_hess(path, symbols, pos_A, H, energy_eh=0.0):
    """ORCA .hess (the layout ORCA 6.1.1 writes: $act_energy, $multiplicity, $hessian in 5-column blocks, $atoms with
    masses and bohr coordinates) for `InHess Read` / `InHessName`. H in Eh/bohr^2 (3N,3N)."""
    H = np.asarray(H, float)
    n3 = 3 * len(symbols)
    if H.shape != (n3, n3):
        raise ValueError(f"Hessian shape {H.shape} != ({n3}, {n3})")
    H = 0.5 * (H + H.T)
    xb = np.asarray(pos_A, float) / C.BOHR2A
    out = ["$orca_hessian_file", "", "$act_atom", "  0", "", "$act_coord", "  0", "", "$act_energy",
           f"  {float(energy_eh or 0.0):.6f}", "", "$multiplicity", "  1", "", "$hessian", str(n3)]
    for c0 in range(0, n3, 5):
        cols = range(c0, min(c0 + 5, n3))
        out.append("     " + "".join(f"{c:19d}" for c in cols))
        for r in range(n3):
            out.append(f"{r:5d}   " + "".join(f"{H[r, c]:19.10E}" for c in cols))
    out += ["", "#", "# The atoms: label  mass x y z (in bohrs)", "#", "$atoms", str(len(symbols))]
    for s, m, (x, y, z) in zip(symbols, masses(symbols), xb):
        out.append(f" {s:<3s}{m:12.5f}{x:20.12f}{y:20.12f}{z:20.12f}")
    out += ["", "$end", ""]
    p = Path(path)
    t = p.with_name(f".{p.name}.tmp{os.getpid()}_{threading.get_ident()}")
    t.write_text("\n".join(out))
    os.replace(t, p)
    return p


def freqs_cm(H, symbols, pos_A):
    """signed harmonic frequencies (cm-1, ascending; 3N - 6 or 3N - 5 values) of the mass-weighted Hessian with the
    translations / rotations projected out (as gen_dft.vib_modes). H in Eh/bohr^2."""
    m = masses(symbols)
    x = np.asarray(pos_A, float) / C.BOHR2A
    n = len(m)
    sm = np.sqrt(np.repeat(m, 3))
    Hm = 0.5 * (np.asarray(H, float) + np.asarray(H, float).T) / np.outer(sm, sm)
    r = x - (m[:, None] * x).sum(0) / m.sum()
    B = []
    for a in range(3):
        v = np.zeros((n, 3))
        v[:, a] = 1.0
        B.append((v * np.sqrt(m)[:, None]).ravel())
    for a in range(3):
        e = np.zeros(3)
        e[a] = 1.0
        B.append((np.cross(e, r) * np.sqrt(m)[:, None]).ravel())
    U, S, _ = np.linalg.svd(np.array(B).T, full_matrices=True)
    ntr = int((S > 1e-6 * S.max()).sum())
    Q = U[:, ntr:]
    lam = np.linalg.eigvalsh(Q.T @ Hm @ Q)
    return np.sort(np.sign(lam) * np.sqrt(np.abs(lam)) * AU2CM)


def imag_summary(nu):
    nu = np.sort(np.asarray(nu, float))
    return dict(n_imag=int((nu < -IMAG_TOL_CM).sum()), n_imag_10=int((nu < -10.0).sum()),
                imag_cm=[float(v) for v in nu if v < -10.0], freqs_low=[round(float(v), 2) for v in nu[:8]])


def hessian_summary(H, symbols, pos_A):
    """imag_summary of an MLIP Hessian; n_imag None (and the reason) if the Hessian is not finite."""
    if not np.isfinite(np.asarray(H, float)).all():
        return dict(n_imag=None, n_imag_10=None, imag_cm=[], freqs_low=[], hessian_error="non-finite Hessian")
    try:
        return imag_summary(freqs_cm(H, symbols, pos_A))
    except np.linalg.LinAlgError as exc:
        return dict(n_imag=None, n_imag_10=None, imag_cm=[], freqs_low=[], hessian_error=str(exc)[:100])


# ================================================================ geometry comparisons
_RMSD = {}


def rmsd_mod():
    """the D1 validation rmsd module (same_ts), loaded under a private name (no clash with a pip 'rmsd')."""
    if "m" not in _RMSD:
        sp = importlib.util.spec_from_file_location("d1_val_rmsd", str(C.RMSD_DIR / "rmsd.py"))
        mod = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(mod)
        _RMSD["m"] = mod
    return _RMSD["m"]


def same_ts(a, b, formed_a, formed_b, rmsd_A=C.SAME_TS["rmsd_A"], dform_A=C.SAME_TS["dform_A"]):
    """D1 rmsd.same_ts: (same?, mapped heavy-atom RMSD, max sorted forming-distance difference)."""
    s, r, d = rmsd_mod().same_ts((list(a[0]), np.asarray(a[1], float)), (list(b[0]), np.asarray(b[1], float)),
                                 formed_a, formed_b, rmsd_A=rmsd_A, dform_A=dform_A)
    return bool(s), float(r), float(d)


def dists(pos_A, pairs):
    p = np.asarray(pos_A, float)
    return [float(np.linalg.norm(p[i] - p[j])) for i, j in pairs]


def heavy_rmsd_direct(symbols, P, Q):
    """Kabsch heavy-atom RMSD with the atom order as given (no mirror, no remapping)."""
    idx = [k for k, s in enumerate(symbols) if s != "H"]
    a = np.asarray(P, float)[idx]
    b = np.asarray(Q, float)[idx]
    a = a - a.mean(0)
    b = b - b.mean(0)
    U, _, Vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return float(np.sqrt(((a @ R.T - b) ** 2).sum(1).mean()))


# ================================================================ TS optimisation
def optts_block(inhess=None, recalc_every=0, numhess=False, calc_hess=False, trust=D1_OPTTS["trust"],
                maxiter=D1_OPTTS["maxiter"]):
    ln = ["%geom"]
    if inhess:
        ln += ["InHess Read", f'InHessName "{inhess}"']
    if calc_hess:
        ln.append("Calc_Hess true")
    if recalc_every:
        ln.append(f"Recalc_Hess {int(recalc_every)}")
    if numhess:
        ln.append("NumHess true")
    ln += [f"Trust {trust}", f"MaxIter {int(maxiter)}", "end"]
    return "\n".join(ln) + "\n"


def modify_internal(pairs):
    """autodE's forming-bond blocks (0-based atom indices, as in the D1 OptTS inputs)."""
    return "".join(f"%geom\nmodify_internal\n{{ B {int(i)} {int(j)} A }} end\nend\n" for i, j in pairs)


def parse_recalc(s):
    """'numeric:30' (ORCA Recalc_Hess 30, numerical through ExtOpt) | 'analytic:30' (restart every 30 cycles with the
    MLIP Hessian) | 'none' (InHess only, the G3L / A'' block)."""
    s = str(s).strip().lower()
    if s in ("none", "0", ""):
        return "none", 0
    m, _, k = s.partition(":")
    if m not in ("numeric", "analytic"):
        raise ValueError(f"recalc {s!r}")
    return m, int(k or D1_OPTTS["recalc_every"])


def _optimize_ts_xtb(symbols, pos_A, pairs, work, timeout_s):
    t0 = time.time()
    r = run_orca_ext(symbols, pos_A, "OptTS Freq", "xtb", None, work, geom_block=G2_OPTTS_BLOCK + modify_internal(pairs),
                     base="ts", maxcore=1000, timeout_s=timeout_s)
    out = dict(pos_A=np.asarray(r["pos_A"], float), converged=bool(r["converged"]), n_cycles=r["n_cycles"],
               wall_s=time.time() - t0, n_ext_calls=0, gpu_s=0.0, e_final_eh=r.get("energy_eh"), rc=r["rc"],
               terminated=r["terminated"], max_iter_hit=r["max_iter_hit"], n_segments=1, recalc="G2 block",
               start_hess="Calc_Hess (xTB)", hessian_source="ORCA XTB2 ALPB(water) Freq (OptTS Freq)")
    fr = r.get("freqs_cm") or []
    vib = sorted(f for f in fr if abs(f) > 1e-8)
    out.update(imag_summary(vib) if vib else dict(n_imag=None, n_imag_10=None, imag_cm=[], freqs_low=[]))
    out["status"] = _status(r, out)
    if out["status"] != "ok":
        out["out_tail"] = r.get("out_tail")
    return out


def _status(r, out):
    if r.get("timed_out"):
        return "timeout"
    if not r.get("terminated") and not out.get("converged"):
        return "orca_error"
    return "ok" if out.get("converged") else "not_converged"


def optimize_ts(spec, symbols, pos_A, formed_pairs, workdir, socket, start_hess="analytic", recalc=None,
                maxiter=D1_OPTTS["maxiter"], trust=D1_OPTTS["trust"], tight=False, timeout_s=None, final_hessian=True):
    """ORCA `! ExtOpt OptTS` with the D1 OptTS block (Trust -0.1, MaxIter 100, forming-bond modify_internal) and the
    MLIP Hessian as the initial Hessian (InHess Read) instead of the numerical Calc_Hess. recalc: see parse_recalc
    (default RECALC_DEFAULT). xtb spec: ORCA `OptTS Freq XTB2 ALPB(water)` with the G2 block (= G2 level).
    n_imag from the MLIP Hessian at the final geometry (projected; modes below -IMAG_TOL_CM).
    Returns dict(pos_A, converged, n_cycles, wall_s, n_ext_calls, gpu_s, imag_cm, n_imag, status, ...)."""
    spec = resolve_spec(spec)
    pairs = [(int(a), int(b)) for a, b in formed_pairs]
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    if spec["backend"] == "xtb":
        return _optimize_ts_xtb(symbols, pos_A, pairs, work, timeout_s)
    if not socket:
        raise ValueError("optimize_ts with an MLIP needs a running server socket (start_server)")
    mode, every = parse_recalc(recalc or RECALC_DEFAULT)
    t0 = time.time()
    cur = np.asarray(pos_A, float)
    cycles, seg, gpu, n_calls, n_hess, orca_s, client_s = 0, 0, 0.0, 0, 0, 0.0, 0.0
    ext_med, inh_read, last = [], [], None
    while True:
        seg += 1
        inh, calc_hess, numhess = None, False, mode == "numeric"
        if start_hess == "analytic" or seg > 1:
            H, dt = server_hessian(socket, symbols, cur)
            gpu += dt
            n_hess += 1
            inh = f"start{seg}.hess"
            write_orca_hess(work / inh, symbols, cur, H)
        elif start_hess == "numeric":
            calc_hess, numhess = True, True
        left = maxiter - cycles
        smax = min(left, every) if mode == "analytic" else left
        gb = optts_block(inhess=inh, recalc_every=every if mode == "numeric" else 0, numhess=numhess,
                         calc_hess=calc_hess, trust=trust, maxiter=smax) + modify_internal(pairs)
        conv_kw = {True: " TightOpt", "tight": " TightOpt", "verytight": " VeryTightOpt"}.get(tight, "")
        r = run_orca_ext(symbols, cur, "OptTS" + conv_kw, spec, socket, work / f"seg{seg}",
                         geom_block=gb, inhess=(work / inh) if inh else None, base="ts", timeout_s=timeout_s)
        last = r
        cycles += r["n_cycles"]
        n_calls += r["ext"]["n"]
        gpu += r["ext"]["eval_s"]
        orca_s += r["wall_s"]
        client_s += r["ext"]["wall_s"]
        if r["ext"]["n"]:
            ext_med.append(r["ext"].get("client_overhead_ms_median"))
        inh_read += r["inhess_read"]
        cur = np.asarray(r["pos_A"], float)
        if r["converged"] or r["timed_out"] or mode != "analytic" or cycles >= maxiter or r["n_cycles"] < smax:
            break
    out = dict(pos_A=cur, converged=bool(last["converged"]), n_cycles=cycles, n_segments=seg, n_ext_calls=n_calls,
               rc=last["rc"], terminated=last["terminated"], max_iter_hit=bool(last["max_iter_hit"]) or cycles >= maxiter,
               recalc=f"{mode}:{every}" if mode != "none" else "none", start_hess=start_hess, tight=tight,
               inhess_read=inh_read[:4], e_orca_eh=last.get("energy_eh"),
               client_overhead_ms=float(np.median([x for x in ext_med if x is not None])) if ext_med else None,
               orca_wall_s=orca_s, ext_client_wall_s=client_s, hessian_source="MLIP analytic (server)",
               orca_side_ms_per_call=(1e3 * (orca_s - client_s) / n_calls) if n_calls else None)
    if final_hessian:
        H, dt = server_hessian(socket, symbols, cur)
        gpu += dt
        n_hess += 1
        out.update(hessian_summary(H, symbols, cur))
    e, g, dt = server_engrad(socket, symbols, cur)
    gpu += dt
    out.update(e_final_eh=e, gmax_final_eh_bohr=float(np.abs(g).max()), gpu_s=gpu, n_hess_calls=n_hess,
               wall_s=time.time() - t0)
    out["status"] = _status(last, out)
    if out["status"] != "ok":
        out["out_tail"] = last.get("out_tail")
    return out


# ================================================================ xtb helpers (plain ORCA, CPU)
def _tmpwork(tag):
    import tempfile
    d = local_tmp() / f"mlts_{jobtag()}"
    d.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f"{tag}_", dir=str(d)))


def xtb_engrad(symbols, pos_A, charge=0, mult=1):
    w = _tmpwork("xeg")
    try:
        r = run_orca_ext(symbols, pos_A, "EnGrad", "xtb", None, w, base="x", charge=charge, mult=mult, timeout_s=1800)
        if "grad_eh_bohr" not in r:
            raise RuntimeError(f"ORCA xtb EnGrad failed: {r.get('out_tail', '')[-800:]}")
        return float(r["engrad_energy_eh"]), np.asarray(r["grad_eh_bohr"], float)
    finally:
        shutil.rmtree(w, ignore_errors=True)


def xtb_hessian(symbols, pos_A, charge=0, mult=1):
    w = _tmpwork("xhs")
    try:
        r = run_orca_ext(symbols, pos_A, "Freq", "xtb", None, w, base="x", charge=charge, mult=mult, timeout_s=3600)
        if not (w / "x.hess").is_file():
            raise RuntimeError(f"ORCA xtb Freq failed: {r.get('out_tail', '')[-800:]}")
        H = OI.read_hess(w / "x.hess")["hessian"]
        return 0.5 * (H + H.T)
    finally:
        shutil.rmtree(w, ignore_errors=True)


# ================================================================ ASE calculator (Sella in C-1)
def ase_calculator(spec, socket=None):
    from ase.calculators.calculator import Calculator, all_changes

    class BridgeCalc(Calculator):
        implemented_properties = ["energy", "forces"]

        def __init__(self):
            super().__init__()
            self.n_calls = 0

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            sym = self.atoms.get_chemical_symbols()
            if socket:
                e, g, _ = server_engrad(socket, sym, self.atoms.positions)
            else:
                e, g = direct_engrad(spec, sym, self.atoms.positions)
            self.n_calls += 1
            self.results = dict(energy=e * C.EH2EV, forces=-np.asarray(g) * C.EHBOHR2EVA)
    return BridgeCalc()


# ================================================================ CLI
def selftest(spec_s, workroot=None):
    """quick check on water: in-process E/F/H, server round trip, ORCA ExtOpt EnGrad parity, client timing."""
    spec = resolve_spec(spec_s)
    rep = dict(spec=spec, host=_socket.gethostname(), **device_info())
    sym = ["O", "H", "H"]
    pos = np.array([[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]]) + 0.01
    t = time.time()
    e, g = direct_engrad(spec, sym, pos)
    rep["direct_first_s"] = time.time() - t
    rep["direct_E_eh"], rep["direct_gmax"] = e, float(np.abs(g).max())
    if spec["backend"] == "xtb":
        print(json.dumps(rep, indent=1, default=C._json_default))
        return 0
    H = hessian(spec, sym, pos)
    rep["freqs_cm"] = [round(float(v), 1) for v in freqs_cm(H, sym, pos)]
    h = start_server(spec)
    try:
        rep["server"] = {k: v for k, v in h.stats().items() if k != "spec"}
        es, gs, dt = server_engrad(h.socket, sym, pos)
        rep["server_vs_direct"] = dict(dE=abs(es - e), dG=float(np.abs(gs - g).max()), t_eval_s=dt)
        w = Path(workroot or _tmpwork("self"))
        r = run_orca_ext(sym, pos, "EnGrad", spec, h.socket, w / "eg", base="w")
        rep["orca"] = dict(rc=r["rc"], terminated=r["terminated"], wall_s=r["wall_s"], ext=r["ext"],
                           dE=abs(r["engrad_energy_eh"] - e) if "engrad_energy_eh" in r else None,
                           dG=float(np.abs(r["grad_eh_bohr"] - g).max()) if "grad_eh_bohr" in r else None,
                           tail=None if r["terminated"] else r["out_tail"])
        rep["client_exe"] = str(client_exe(Path(h.socket).parent))
    finally:
        h.shutdown()
    print(json.dumps(rep, indent=1, default=C._json_default))
    return 0


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--spec", required=True)
    s.add_argument("--socket", required=True)
    s.add_argument("--parent-pid", type=int, default=None)
    s.add_argument("--idle-s", type=float, default=None)
    t = sub.add_parser("selftest")
    t.add_argument("spec")
    a = ap.parse_args(argv)
    if a.cmd == "serve":
        serve(a.spec, a.socket, parent_pid=a.parent_pid, idle_s=a.idle_s)
        return 0
    if a.cmd == "selftest":
        return selftest(a.spec)
    return 2


if __name__ == "__main__":
    sys.exit(main())
