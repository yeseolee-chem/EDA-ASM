#!/usr/bin/env python3
"""S0 check: the installed xtb prints the lines Coley's rotatability parser reads.
Runs make_reference.xtb_scan on a small azomethine ylide (C=N+ dihedral, expected hindered)."""
import sys, tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import numpy as np
import pilot_common as pc
from make_reference import xtb_scan

cfg = pc.load_config(sys.argv[1] if len(sys.argv) > 1 else None)
import autode as ade
m = ade.Molecule(smiles="[CH2-][N+](C)=CC", name="ay")          # atoms 0 C- 1 N+ 2 CH3 3 C= 4 CH3
syms = [a.label for a in m.atoms]; xyz = np.array([a.coord for a in m.atoms], dtype=float)
r = xtb_scan(cfg["xtb_bin"], Path(tempfile.mkdtemp()), syms, xyz, (0, 1, 3, 4), 0)
print(r)
ok = r["n_points"] >= 60 and r["rotatable"] is not None
print("XTB_SCAN_FORMAT", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
