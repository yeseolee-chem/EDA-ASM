#!/usr/bin/env python3
"""env_check.py — DEPLOY-V2 start condition 0-7 (run by jobs/env_build.slurm inside the `mlts` env, on a GPU node).

Loads every Z candidate that the env can provide and evaluates E / F on a small Si-containing molecule
(trimethyl(vinyl)silane, all D1 elements except halogens / S appear in other tests) on the GPU, so Phase Z knows which
candidates exist. Writes results/0_env.json (atomic). Exit 0 = env usable (torch + CUDA + at least one MLIP), 3 = not.
Candidates that fail are recorded with the reason and later dropped from Z (spec 4)."""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"


def ver(p):
    try:
        return version(p)
    except PackageNotFoundError:
        return None


def mol():
    from ase import Atoms
    from rdkit import Chem
    from rdkit.Chem import AllChem
    m = Chem.AddHs(Chem.MolFromSmiles("C=C[Si](C)(C)C"))
    AllChem.EmbedMolecule(m, randomSeed=20261104)
    AllChem.MMFFOptimizeMolecule(m)
    pos = m.GetConformer().GetPositions()
    return Atoms([a.GetSymbol() for a in m.GetAtoms()], positions=pos)


def try_calc(name, make):
    import numpy as np
    rec = dict(name=name)
    t0 = time.time()
    try:
        calc = make()
        rec["load_s"] = round(time.time() - t0, 2)
        a = mol()
        a.calc = calc
        t1 = time.time()
        e = float(a.get_potential_energy())
        f = np.asarray(a.get_forces())
        t2 = time.time()
        for _ in range(5):
            a.positions = a.positions + 1e-4
            a.get_forces()
        rec.update(ok=bool(np.isfinite(e) and np.isfinite(f).all() and f.shape == (len(a), 3)),
                   energy_eV=e, fmax_eV_A=float(np.abs(f).max()), first_call_s=round(t2 - t1, 3),
                   per_call_s=round((time.time() - t2) / 5, 4), n_atoms=len(a))
    except Exception as exc:  # recorded, candidate dropped in Z
        rec.update(ok=False, error=f"{type(exc).__name__}: {exc}"[:600], trace=traceback.format_exc()[-1500:])
    print(json.dumps({k: v for k, v in rec.items() if k != "trace"}), flush=True)
    return rec


def main():
    import torch
    out = dict(python=sys.version.split()[0], host=os.uname().nodename,
               packages={p: ver(p) for p in ("torch", "mace-torch", "e3nn", "ase", "sella", "aimnet", "orca-mlips",
                                             "rdkit", "numpy", "scipy", "tblite", "torch-geometric", "jax")},
               cuda=torch.cuda.is_available(),
               gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    dev = "cuda" if out["cuda"] else "cpu"
    cands = []

    def mace_omol():
        from mace.calculators import mace_omol as f
        return f(model="extra_large", device=dev, default_dtype="float64")
    cands.append(("MACE-OMOL-0", mace_omol))

    def mace_off24():
        from mace.calculators import mace_off as f
        errs = []
        for m in ("medium-24", "medium_off24", "https://github.com/ACEsuit/mace-off/raw/main/mace_off24/MACE-OFF24_medium.model"):
            try:
                return f(model=m, device=dev, default_dtype="float64")
            except Exception as exc:
                errs.append(f"{m}: {type(exc).__name__}: {exc}"[:200])
        raise RuntimeError(" | ".join(errs))
    cands.append(("MACE-OFF24", mace_off24))

    def mace_off23():  # reference only (already cached); SPICE level, no Si
        from mace.calculators import mace_off as f
        return f(model="medium", device=dev, default_dtype="float64")
    cands.append(("MACE-OFF23-medium", mace_off23))

    def aimnet2():
        from aimnet.calculators import AIMNet2ASE
        return AIMNet2ASE("aimnet2")
    cands.append(("AIMNet2", aimnet2))

    def gfn2():
        from tblite.ase import TBLite
        return TBLite(method="GFN2-xTB", solvation=("alpb", "water"), verbosity=0)
    cands.append(("GFN2-xTB-ALPB-water", gfn2))

    out["candidates"] = [try_calc(n, f) for n, f in cands]
    imp = {}
    for mod in ("sella", "orca_mlips"):
        try:
            m = __import__(mod)
            imp[mod] = dict(ok=True, file=getattr(m, "__file__", None))
        except Exception as exc:
            imp[mod] = dict(ok=False, error=f"{type(exc).__name__}: {exc}"[:300])
    out["imports"] = imp
    mlip_ok = [c["name"] for c in out["candidates"] if c.get("ok") and not c["name"].startswith("GFN2")]
    out["mlip_ok"] = mlip_ok
    out["pass_0_7"] = bool(out["cuda"] and out["packages"]["torch"] and mlip_ok)
    out["time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    RES.mkdir(parents=True, exist_ok=True)
    tmp = RES / "0_env.json.tmp"
    tmp.write_text(json.dumps(out, indent=1))
    tmp.replace(RES / "0_env.json")
    print(json.dumps(dict(pass_0_7=out["pass_0_7"], mlip_ok=mlip_ok, imports=imp, gpu=out["gpu"]), indent=1))
    return 0 if out["pass_0_7"] else 3


if __name__ == "__main__":
    sys.exit(main())
