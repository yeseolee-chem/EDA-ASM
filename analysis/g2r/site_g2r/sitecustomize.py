"""sitecustomize.py — D1-G2R F / T jobs only (analysis/g2r/jobs/g_g2r.slurm puts this folder first on PYTHONPATH).

docs/specs/D1G2R.md 8-1 (user decision D-5, 2026-11-13):
  1. analysis/d1_pair/site_si/sitecustomize.py runs unchanged (runpy; Python imports only the first sitecustomize on
     sys.path, so site_si would not run by itself): the Si entries of the rev 4-6 element tables.
  2. only when G2R_TOL_PATCH=1: rev5_common.GATE_E_EH 1e-6 -> 1e-4 Eh (B1 engine gate, read at call time as
     R5.GATE_E_EH, ext_features.py:637) and ext_features.TOL_DIP_AU 1e-3 -> 1e-2 a.u. (dipole-sum check, a module
     global read at call time, ext_features.py:401; also used by fu2/atoms.py:434). Every other tolerance is unchanged.
This runs at every interpreter start, so the feat_lf / tokens_lf children and their spawn workers get it too.
Python swallows exceptions raised here (site.execsitecustomize), so analysis/g2r/f_preflight.py asserts the values in
the job process and in a spawned interpreter before any feature / token computation.
"""
import os
import runpy
import sys

_here = os.path.dirname(os.path.abspath(__file__))
runpy.run_path(os.path.join(_here, "..", "..", "d1_pair", "site_si", "sitecustomize.py"), run_name="site_si")
if os.environ.get("G2R_TOL_PATCH") == "1" and os.environ.get("D1PAIR_ESP_DIR"):
    if os.environ["D1PAIR_ESP_DIR"] not in sys.path:
        sys.path.append(os.environ["D1PAIR_ESP_DIR"])
    import rev5_common as _r5
    import ext_features as _ef
    _r5.GATE_E_EH = 1e-4
    _ef.TOL_DIP_AU = 1e-2
