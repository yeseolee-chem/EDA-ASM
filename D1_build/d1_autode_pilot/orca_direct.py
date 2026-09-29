#!/usr/bin/env python3
"""orca_direct.py — ORCA runs that do not go through autodE (validation, VALIDATION_SPEC §6.3 / §6.6),
and the parsers for their outputs.

Levels (§6.3)
  L0  the pilot TS level, character for character: the header autodE 1.4.5 wrote for the pilot OptTS
      (config val.pilot_optts_inp). autodE also writes one `%geom modify_internal { B i j A } end end`
      block per forming bond (0-based ORCA indices); those blocks are re-made for the TS at hand
      (val.l0_modify_internal) and are excluded from the header comparison.
  L1  L0 with B3LYP -> B3LYP/G (VWN-III, as G16). Gradients only (V2a).
  L2  G16-like: val.L2_keywords (`! OptTS Freq B3LYP/G NORI D3BJ def2-SVP DefGrid3 TightSCF CPCM(Water)`)
      + the L0 blocks.
  EnGrad variants replace `OptTS Freq` by `EnGrad` and drop every %geom block.
  Opt  autodE's own reactant Opt header from the pilot (val.pilot_opt_inp): reference re-optimisation
       (refopt) and the relaxed scan (V7).
The pilot header is compared with the pre-registered text (prereg.l0_expected_header) after masking
whitespace, %pal and %maxcore values and the modify_internal blocks; a difference is a STOP (§6.3).
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import numpy as np

MODIFY_RE = re.compile(r"%geom[ \t]*\nmodify_internal[ \t]*\n\{[^}]*\}[ \t]*end[ \t]*\nend[ \t]*\n")
CALC_HESS_GEOM_RE = re.compile(r"%geom[ \t]*\nCalc_Hess[^\n]*\n(?:[^\n]*\n)*?end[ \t]*\n")
FSPE_RE = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)")
RUNTIME_RE = re.compile(r"TOTAL RUN TIME:\s+(\d+) days (\d+) hours (\d+) minutes (\d+) seconds (\d+) msec")
VWN_V_RE = re.compile(r"VWN-?(5|V)\b", re.I)
VWN_III_RE = re.compile(r"VWN-?(3|III)\b", re.I)


class LevelMismatch(Exception):
    pass


def stop(msg: str):
    """A §6.3 STOP: under the validation worker pool (VAL_QUEUE set) also stop further claims."""
    q = os.environ.get("VAL_QUEUE")
    if q:
        Path(q).mkdir(parents=True, exist_ok=True)
        (Path(q) / "STOP").write_text(msg + "\n")
    raise LevelMismatch(msg)


# ---------------------------------------------------------------- headers
def input_header(path) -> str:
    """Everything before the coordinate block (first line starting with '*')."""
    out = []
    for line in Path(path).read_text().splitlines(keepends=True):
        if line.lstrip().startswith("*"):
            break
        out.append(line)
    return "".join(out)


def normalise_header(text: str) -> str:
    t = MODIFY_RE.sub("", text)
    t = re.sub(r"%pal\s+nprocs\s+\d+", "%pal nprocs N", t)
    t = re.sub(r"%\s*maxcore\s+\d+", "%maxcore N", t)
    return " ".join(t.split())


def modify_blocks(pairs) -> str:
    return "".join(f"%geom\nmodify_internal\n{{ B {int(i)} {int(j)} A }} end\nend\n" for i, j in sorted(pairs))


def _set_resources(header: str, n_cores: int, maxcore_mb: int) -> str:
    header = re.sub(r"%pal nprocs \d+", f"%pal nprocs {int(n_cores)}", header)
    return re.sub(r"(%\s*maxcore\s+)\d+", lambda m: f"{m.group(1)}{int(maxcore_mb)}", header)


def l0_header(cfg: dict, pairs=(), n_cores: int = 8) -> str:
    v = cfg["val"]
    base = input_header(v["pilot_optts_inp"])
    got, want = normalise_header(base), normalise_header(cfg["prereg"]["l0_expected_header"])
    if got != want:
        stop(f"L0 header of {v['pilot_optts_inp']} differs from the pre-registered text:\n  {got}\n  != {want}")
    h = MODIFY_RE.sub("", base)
    if v.get("l0_modify_internal", True) and pairs:
        k = h.index("%output")
        h = h[:k] + modify_blocks(pairs) + h[k:]
    return _set_resources(h, n_cores, cfg["maxcore_mb"])


def level_header(cfg: dict, level: str, pairs=(), n_cores: int = 8, engrad: bool = False) -> str:
    h = l0_header(cfg, () if engrad else pairs, n_cores)
    kw, body = h.split("\n", 1)
    if level == "L1":
        kw = re.sub(r"(?<=\s)B3LYP(?=\s)", "B3LYP/G", kw)
    elif level == "L2":
        kw = cfg["val"]["L2_keywords"].strip()
    elif level != "L0":
        raise ValueError(f"unknown level {level}")
    if engrad:
        kw = kw.replace("OptTS Freq", "EnGrad")
        body = MODIFY_RE.sub("", CALC_HESS_GEOM_RE.sub("", body))
    return kw + "\n" + body


def opt_header(cfg: dict, n_cores: int = 8, extra: str = "") -> str:
    """autodE's reactant Opt header from the pilot; `extra` (e.g. a scan %geom block) goes before %output."""
    h = input_header(cfg["val"]["pilot_opt_inp"])
    if not h.startswith("! Opt "):
        stop(f"{cfg['val']['pilot_opt_inp']} is not an autodE Opt input")
    h = _set_resources(h, n_cores, cfg["maxcore_mb"])
    if extra:
        k = h.index("%output")
        h = h[:k] + extra + h[k:]
    return h


def scan_block(i: int, j: int, k: int, l: int, start: float, end: float, n: int) -> str:
    return (f"%geom Scan\n  B {i} {j} = {start:.2f}, {end:.2f}, {n}\n"
            f"  B {k} {l} = {start:.2f}, {end:.2f}, {n}\nend\nend\n")


def write_inp(path, header: str, syms, xyz, charge: int = 0, mult: int = 1):
    body = "".join(f"{s:<3}{x:16.8f}{y:16.8f}{z:16.8f}\n" for s, (x, y, z) in zip(syms, np.asarray(xyz)))
    Path(path).write_text(f"{header}* xyz {int(charge)} {int(mult)}\n{body}*\n")


# ---------------------------------------------------------------- running
def orca_bin(cfg: dict | None = None) -> str:
    return os.environ.get("ORCA_BIN") or (cfg or {}).get("orca_bin")


def run(cfg: dict, inp) -> Path:
    """ORCA on `inp` in its own folder. Skipped when <base>.out already terminated normally (idempotent)."""
    inp = Path(inp)
    out = inp.with_suffix(".out")
    if out.is_file() and terminated(out.read_text(errors="replace")):
        return out
    with open(out, "w") as fo, open(inp.with_suffix(".err"), "w") as fe:
        subprocess.run([orca_bin(cfg), inp.name], cwd=inp.parent, stdout=fo, stderr=fe, check=False)
    return out


# ---------------------------------------------------------------- parsers
def terminated(text: str) -> bool:
    return "ORCA TERMINATED NORMALLY" in text


def opt_converged(text: str) -> bool:
    return "THE OPTIMIZATION HAS CONVERGED" in text


def last_fspe(text: str):
    m = FSPE_RE.findall(text)
    return float(m[-1]) if m else None


def run_hours(text: str):
    m = RUNTIME_RE.findall(text)
    if not m:
        return None
    d, h, mi, s, ms = (int(x) for x in m[-1])
    return d * 24 + h + mi / 60 + (s + ms / 1000) / 3600


def parse_hess(path):
    """(frequencies [cm-1], normal modes 3N x 3N, columns = modes) from an ORCA .hess file."""
    lines = Path(path).read_text().splitlines()
    sec = {l.strip(): i for i, l in enumerate(lines) if l.startswith("$")}
    i = sec["$vibrational_frequencies"]
    n = int(lines[i + 1].split()[0])
    freqs = np.array([float(lines[i + 2 + k].split()[1]) for k in range(n)])
    j = sec["$normal_modes"]
    nr, nc = (int(x) for x in lines[j + 1].split()[:2])
    modes = np.zeros((nr, nc))
    k, filled = j + 2, 0
    while filled < nc:
        cols = [int(c) for c in lines[k].split()]
        k += 1
        for r in range(nr):
            for c, v in zip(cols, lines[k + r].split()[1:]):
                modes[r, c] = float(v)
        k += nr
        filled += len(cols)
    return freqs, modes


def imag_and_mode(hess_path):
    """(imaginary frequencies ascending, (N,3) displacement of the most negative mode or None)."""
    freqs, modes = parse_hess(hess_path)
    imag = sorted(float(f) for f in freqs if f < 0)
    if not imag:
        return [], None
    return imag, modes[:, int(np.argmin(freqs))].reshape(-1, 3)


def parse_engrad(path):
    """(energy Eh, gradient (N,3) Eh/bohr) from an ORCA .engrad file."""
    vals = [l.split()[0] for l in Path(path).read_text().splitlines() if l.strip() and not l.lstrip().startswith("#")]
    n = int(vals[0])
    return float(vals[1]), np.array([float(v) for v in vals[2:2 + 3 * n]]).reshape(n, 3)


def level_checks(text: str) -> dict:
    """What the output says about the level (§6.3 'output checks'): the strings are kept verbatim."""
    first = lambda rx, flags=0: (re.findall(rx, text, flags) or [None])[0]      # noqa: E731
    kw = first(r"^\|\s*1>\s*(!.*?)\s*$", re.M)
    ri = sorted({l.strip() for l in text.splitlines()
                 if re.search(r"(RI-?J|RIJCOSX|RIJ-COSX|COSX|RI-approx)", l) and "..." in l})
    d3 = {k: float(v) for k, v in re.findall(r"^\s*(s6|a1|s8|a2) scaling factor\s*:\s*(-?\d+\.\d+)", text, re.M)}
    return dict(
        keyword_line=kw,
        vwn=first(r"LDA part of GGA corr\.\s+LDAOpt\s+\.+\s+(\S+)"),
        rijcosx=first(r"RIJ-COSX \(HFX calculated with COS-X\)\)\.+\s*(\S+)"),
        ri_lines=ri[:12],
        d3=d3,
        eps=first(r"^\s*Epsilon\s+\.\.\.\s*(\S+)", re.M),
        smd_cds="SMD CDS free energy correction energy" in text,
        angular_grid=(first(r"Angular Grid \(max\. ang\.\)\s+AngularGrid\s+\.\.\.\s+([^\n]+)") or "").strip() or None,
    )


def level_ok(level: str, chk: dict) -> dict:
    """§6.3 output checks as booleans (None when the string was not printed)."""
    vwn = chk.get("vwn") or ""
    return dict(
        vwn_ok=(bool(VWN_V_RE.search(vwn)) if level == "L0" else bool(VWN_III_RE.search(vwn))) if vwn else None,
        no_rijcosx=None if level != "L2" else (chk.get("rijcosx") in (None, "off")),
        d3_present=len(chk.get("d3") or {}) == 4,
        smd_ok=chk.get("smd_cds") and chk.get("eps") == "78.3550",
    )
