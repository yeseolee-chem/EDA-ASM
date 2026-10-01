#!/usr/bin/env python3
"""r6_partition.py — rev 6 partition rule (docs/specs/REV6_FAIR_ESPLEY_COMPARISON.md R-1; results_rev6/R_plan.md §2).

Fragment A = the label A_idx (input_meta.csv: frag1 = dipole TS atom indices), B = the rest, for every gate and every
feature. The reactant <-> TS atom maps (map0 / map1, needed by fragmenter.audit_forming_bonds and by the reactant-atom
features) come from the re-detection on the given (G1) TS when it reproduces A_idx — the rev 4/5 path, so the rows
that passed rev 4 get identical maps — and otherwise from the re-detection on the DFT TS (label ts_file + rel files),
which has the same atom order. The re-detections are diagnostics only; a DFT-TS re-detection that differs from A_idx is
an R-5 STOP (the caller decides).

Used by r6_recover.py (gates) and, with ESPLEY_PARTITION=label, by xtb_slice.ts_fragments (features).
"""
from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
import fragmenter as fr  # noqa: E402

PROF = Path(os.environ.get("ESPLEY_PROF",
                           "/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/dipolar_cycloaddition/extracted/full_dataset_profiles"))


def dft_structures(label, prof=PROF):
    """(ts0, rel0_1, rel0_2) of the label: the Coley DFT TS and the label reference files, as (syms, xyz)."""
    pdir = Path(prof) / str(int(label["rxn_id"]))
    return (fr.read_xyz(pdir / label["ts_file"]), fr.read_xyz(pdir / label["rel1_file"]),
            fr.read_xyz(pdir / label["rel2_file"]))


def label_partition(label, ts, r0, r1, a_idx, prof=PROF):
    """(Partition or None, diag). ts / r0 / r1 = the structures the caller works on (G1 TS and references, atom order
    of the label files); a_idx = the label A_idx (sorted list). None = the DFT TS re-detection does not reproduce
    a_idx either (no map available; an R-5 STOP for the caller)."""
    a = sorted(int(i) for i in a_idx)
    n = len(ts[0])
    b = sorted(set(range(n)) - set(a))
    P = fr.partition(ts, r0, r1)
    diag = dict(redetect_status=P.status, redetect_A_match=bool(P.status == "ok" and sorted(P.A) == a),
                map_source=None, dft_status=None, dft_A_match=None)
    if diag["redetect_A_match"]:
        diag["map_source"] = "g1_redetect"
        return fr.Partition("ok", frozenset(a), frozenset(b), P.map0, P.map1, dict(P.diag)), diag
    ts0, q0, q1 = dft_structures(label, prof)
    if len(ts0[0]) != n or list(ts0[0]) != list(ts[0]):
        diag.update(dft_status="atom_order_differs", dft_A_match=False)
        return None, diag
    P0 = fr.partition(ts0, q0, q1)
    diag.update(dft_status=P0.status, dft_A_match=bool(P0.status == "ok" and sorted(P0.A) == a))
    if not diag["dft_A_match"]:
        return None, diag
    diag["map_source"] = "dft_ts"
    return fr.Partition("ok", frozenset(a), frozenset(b), P0.map0, P0.map1, dict(P0.diag)), diag


def dft_redetect_matches(label, a_idx, prof=PROF):
    """R-1 diagnostic on the DFT TS alone: (status, sorted(P0.A) == a_idx)."""
    ts0, q0, q1 = dft_structures(label, prof)
    P0 = fr.partition(ts0, q0, q1)
    return P0.status, bool(P0.status == "ok" and sorted(P0.A) == sorted(int(i) for i in a_idx))
