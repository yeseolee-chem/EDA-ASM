"""tests/aim_charge.py — sandbox only, READ ONLY: finetune.aim_predict passes atoms.info total_charge to AIMNet2
(review defect: charge was always 0). The smallest charged d0_frag of a D0 plan (argv[1] = plan_d0.parquet) is
evaluated with the released AIMNet2 at total_charge = its charge and = 0; the energies must differ. Prints JSON."""
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import finetune as F  # noqa: E402

F._guard_imports()
import torch  # noqa: E402
from ase import Atoms  # noqa: E402

P = pd.read_parquet(sys.argv[1], columns=["struct_id", "kind", "charge", "n_atoms", "symbols", "pos_A"])
x = P[(P.kind == "d0_frag") & (P.charge != 0)].sort_values(["n_atoms", "struct_id"]).iloc[0]
syms = str(x.symbols).split()
pos = np.asarray(x.pos_A, float).reshape(-1, 3)
al = []
for q in (int(x.charge), 0):
    a = Atoms(symbols=syms, positions=pos)
    a.info["total_charge"] = q
    al.append(a)
jpt = F.fetch("AIMNet2")
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.jit.load(str(jpt), map_location=dev)
E, Fr = F.aim_predict(model, al, dev, forces=True)
print(json.dumps(dict(struct_id=x.struct_id, n_atoms=int(x.n_atoms), charge=int(x.charge), jpt=str(jpt), device=dev,
                      e_charged_eV=float(E[0]), e_neutral_eV=float(E[1]), dE_eV=float(E[0] - E[1]),
                      max_dF_eV_A=float(np.abs(Fr[0] - Fr[1]).max()), differ=bool(abs(E[0] - E[1]) > 1e-3)), indent=1))
