"""sitecustomize.py — D1PAIR G / F jobs only (jobs/g_env.sh puts this folder on PYTHONPATH).

User decision 2026-10-19: the rev 4-6 element tables have no Si, so every D1 SiMe3 row (442 of the 8,659 G1 targets)
would stop with a KeyError. Add the Si entries, nothing else: fragmenter.COV (Cordero et al. 2008 covalent radius
1.11 A), fragmenter.Z (14), rev5_common.BONDI (Bondi 1964 vdW radius 2.10 A), rev5_common.GFN2_VALENCE (4 valence
electrons). D0 contains no Si, so no D0 value can change (G-0 / F-0 check the D0 path). The rev 4-6 files are not
edited; this runs at interpreter start, so the spawned ext_features / multiprocessing workers get it too.
"""
import os
import sys

_esp = os.environ.get("D1PAIR_ESP_DIR")
if _esp:
    if _esp not in sys.path:
        sys.path.append(_esp)
    import fragmenter as _fr
    import rev5_common as _r5
    _fr.COV.setdefault("Si", 1.11)
    _fr.Z.setdefault("Si", 14)
    _r5.BONDI.setdefault("Si", 2.10)
    _r5.GFN2_VALENCE.setdefault("Si", 4)
    os.environ["D1PAIR_SI_PATCH"] = "1"
