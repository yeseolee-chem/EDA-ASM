#!/usr/bin/env python3
"""rmsd.py — TS comparison (VALIDATION_SPEC §6.4): heavy-atom RMSD over graph isomorphisms and mirror
images (atom orders may differ: Coley vs autodE, repeat runs), and the "same TS" rule."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from networkx.algorithms.isomorphism import GraphMatcher

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "analysis" / "espley_xtb_repro"))
import fragmenter as fr  # noqa: E402


def kabsch_rmsd(P, Q):
    P = P - P.mean(0); Q = Q - Q.mean(0)
    U, S, Vt = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return float(np.sqrt(((P @ R.T - Q) ** 2).sum(1).mean()))


def mapped_heavy_rmsd(a, b, cap=5000):
    """a, b = (syms, xyz ndarray). Min heavy-atom RMSD over graph isomorphisms and mirror images."""
    a = (a[0], np.asarray(a[1], dtype=float)); b = (b[0], np.asarray(b[1], dtype=float))
    Ga = fr.heavy_graph(*a)[0]; Gb = fr.heavy_graph(*b)[0]
    best = np.inf
    gm = GraphMatcher(Ga, Gb, node_match=lambda x, y: x["lab"] == y["lab"])
    for k, m in enumerate(gm.isomorphisms_iter()):
        ia = list(m); ib = [m[i] for i in ia]
        for mirror in (1.0, -1.0):
            Q = b[1][ib] * np.array([mirror, 1.0, 1.0])
            best = min(best, kabsch_rmsd(a[1][ia], Q))
        if k + 1 >= cap:
            break
    return best


def aligned_max_dev(a_xyz, b_xyz):
    """Same atom order (e.g. two runs of one ORCA input): max per-atom deviation after Kabsch alignment."""
    P = np.asarray(a_xyz, float); Q = np.asarray(b_xyz, float)
    P = P - P.mean(0); Q = Q - Q.mean(0)
    U, _, Vt = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return float(np.linalg.norm(P @ R.T - Q, axis=1).max())


def same_ts(a, b, formed_a, formed_b, rmsd_A=0.10, dform_A=0.02):
    """(same?, mapped heavy RMSD, max |forming-distance difference|); forming distances compared sorted."""
    r = mapped_heavy_rmsd(a, b)
    dd = float(np.max(np.abs(np.sort(np.asarray(formed_a, float)) - np.sort(np.asarray(formed_b, float)))))
    return (r <= rmsd_A and dd <= dform_A), r, dd
