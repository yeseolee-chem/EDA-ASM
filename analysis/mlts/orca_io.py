"""orca_io.py — readers for ORCA 6.1.1 files used by DEPLOY-V2 (MLTS) Phases I and G.

  read_engrad(path)        .engrad of an EnGrad / ExtOpt step: energy (Eh), gradient (Eh/bohr), Z, coordinates (bohr)
  read_hess(path)          .hess: Hessian (Eh/bohr^2, 3N x 3N), symbols, masses, coordinates (bohr)
  read_out_steps(path)     .out / .out.gz of an Opt / OptTS / EnGrad run: one record per geometry that has
                           coordinates (A), FINAL SINGLE POINT ENERGY and a CARTESIAN GRADIENT block, in file order
  input_echo(path)         the echoed input lines of an .out ('|  n> ...'), joined
  label_level(text)        is an input (echo) at the D1 label level? (B3LYP D3BJ def2-SVP, CPCM(water) + smd / SMD(water))
All arrays are numpy float64. Gzip is detected by the '.gz' suffix."""
from __future__ import annotations

import gzip
import re
from pathlib import Path

import numpy as np

Z2SYM = {1: "H", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 14: "Si", 15: "P", 16: "S", 17: "Cl", 35: "Br", 53: "I"}
SYM2Z = {v: k for k, v in Z2SYM.items()}


def _open(path):
    p = str(path)
    return gzip.open(p, "rt", errors="replace") if p.endswith(".gz") else open(p, errors="replace")


def read_engrad(path):
    """ORCA .engrad: '#'-commented blocks: n_atoms, energy, 3N gradient values, then N lines 'Z x y z' (bohr)."""
    vals = []
    with _open(path) as f:
        for ln in f:
            s = ln.strip()
            if s and not s.startswith("#"):
                vals.append(s)
    n = int(vals[0])
    e = float(vals[1])
    g = np.array([float(x) for x in vals[2:2 + 3 * n]]).reshape(n, 3)
    zs, xyz = [], []
    for s in vals[2 + 3 * n:2 + 4 * n]:
        t = s.split()
        zs.append(int(t[0]))
        xyz.append([float(t[1]), float(t[2]), float(t[3])])
    return dict(n_atoms=n, energy_eh=e, grad_eh_bohr=g, Z=zs, pos_bohr=np.array(xyz))


def read_hess(path):
    """ORCA .hess: $hessian (3N, column blocks), $atoms (symbol mass x y z, bohr)."""
    with _open(path) as f:
        lines = f.read().splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.strip() == "$hessian")
    n3 = int(lines[i + 1].split()[0])
    H = np.zeros((n3, n3))
    k = i + 2
    while k < len(lines):
        head = lines[k].split()
        if not head or lines[k].lstrip().startswith("$"):
            break
        cols = [int(c) for c in head]
        for r in range(n3):
            t = lines[k + 1 + r].split()
            H[int(t[0]), cols] = [float(x) for x in t[1:1 + len(cols)]]
        k += 1 + n3
        if cols[-1] == n3 - 1:
            break
    j = next(k for k, ln in enumerate(lines) if ln.strip() == "$atoms")
    na = int(lines[j + 1].split()[0])
    sym, mass, xyz = [], [], []
    for ln in lines[j + 2:j + 2 + na]:
        t = ln.split()
        sym.append(t[0])
        mass.append(float(t[1]))
        xyz.append([float(t[2]), float(t[3]), float(t[4])])
    if n3 != 3 * na:
        raise ValueError(f"{path}: hessian dim {n3} != 3 x {na} atoms")
    return dict(hessian=H, symbols=sym, masses=np.array(mass), pos_bohr=np.array(xyz))


_RE_E = re.compile(r"^FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)")
_RE_GL = re.compile(r"^\s*(\d+)\s+([A-Za-z]{1,2})\s*:\s+(-?\d+\.\d+(?:[EeDd][-+]?\d+)?)\s+(-?\d+\.\d+(?:[EeDd][-+]?\d+)?)\s+"
                    r"(-?\d+\.\d+(?:[EeDd][-+]?\d+)?)")


def read_out_steps(path):
    """Scan an ORCA .out in order. The current geometry is the last 'CARTESIAN COORDINATES (ANGSTROEM)' block; a
    'FINAL SINGLE POINT ENERGY' line attaches to it; a 'CARTESIAN GRADIENT' block closes the step. Geometries without
    an energy and a gradient (e.g. the final re-print after convergence) are not returned.
    Returns (steps, info): steps = [dict(symbols, pos_A, energy_eh, grad_eh_bohr, cycle)], info = counts."""
    steps = []
    cur = None          # dict(symbols, pos_A, energy_eh)
    n_cycles = n_coord = n_energy = n_grad = 0
    cycle = 0
    with _open(path) as f:
        lines = f.read().splitlines()
    k = 0
    while k < len(lines):
        ln = lines[k]
        s = ln.strip()
        if "GEOMETRY OPTIMIZATION CYCLE" in s:
            n_cycles += 1
            cycle = n_cycles
        if s == "CARTESIAN COORDINATES (ANGSTROEM)":
            n_coord += 1
            k += 2                                    # dashes line
            sym, pos = [], []
            while k < len(lines) and lines[k].strip():
                t = lines[k].split()
                if len(t) < 4:
                    break
                sym.append(t[0])
                pos.append([float(t[1]), float(t[2]), float(t[3])])
                k += 1
            cur = dict(symbols=sym, pos_A=np.array(pos), energy_eh=None, cycle=cycle)
            continue
        m = _RE_E.match(s)
        if m:
            n_energy += 1
            if cur is not None:
                cur["energy_eh"] = float(m.group(1))
        if s == "CARTESIAN GRADIENT":
            n_grad += 1
            k += 1
            while k < len(lines) and not _RE_GL.match(lines[k]):
                k += 1
            g = []
            while k < len(lines):
                mm = _RE_GL.match(lines[k])
                if not mm:
                    break
                g.append([float(mm.group(j).replace("D", "E").replace("d", "e")) for j in (3, 4, 5)])
                k += 1
            if cur is not None and cur.get("energy_eh") is not None and len(g) == len(cur["symbols"]):
                steps.append(dict(symbols=list(cur["symbols"]), pos_A=cur["pos_A"].copy(), energy_eh=cur["energy_eh"],
                                  grad_eh_bohr=np.array(g), cycle=cur["cycle"]))
                cur = dict(cur, energy_eh=None)       # one gradient per (geometry, energy)
            continue
        k += 1
    info = dict(n_opt_cycles=n_cycles, n_coord_blocks=n_coord, n_energies=n_energy, n_gradients=n_grad,
                n_steps=len(steps), terminated_normally=any("ORCA TERMINATED NORMALLY" in x for x in lines[-30:]))
    return steps, info


def input_echo(path):
    out = []
    with _open(path) as f:
        for ln in f:
            m = re.match(r"^\|\s*\d+>\s?(.*)$", ln)
            if m:
                out.append(m.group(1))
            elif out and "****END OF INPUT****" in ln:
                break
    return "\n".join(out)


def label_level(text):
    """(ok, reason) for an ORCA input text: label level = B3LYP + D3BJ + def2-SVP (+ def2/J) and SMD water
    (either 'CPCM(Water)' + '%cpcm smd true' or 'SMD(water)'); no other basis / functional on the '!' lines."""
    bang = " ".join(ln.split("#")[0] for ln in text.splitlines() if ln.strip().startswith("!")).upper()
    low = text.lower()
    toks = set(re.split(r"\s+", bang.replace("!", " ").strip()))
    need = {"B3LYP", "D3BJ", "DEF2-SVP"}
    if not need <= toks:
        return False, f"missing {sorted(need - toks)}"
    bad = {"DEF2-TZVP", "DEF2-TZVPP", "6-31G*", "HF", "R2SCAN-3C", "XTB2", "PBE0", "WB97X-D3", "DEF2-SV(P)"} & toks
    if bad:
        return False, f"other level {sorted(bad)}"
    smd = ("SMD(WATER)" in toks) or ("CPCM(WATER)" in toks and re.search(r"smd\s+true", low) and "water" in low)
    if not smd:
        return False, "no SMD water"
    return True, "label level"
