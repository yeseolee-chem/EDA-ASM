"""D1-HF common paths, constants and helpers (docs/specs/D1HF.md).

Writes only under $HF (= $D1HF_SCRATCH, default /gpfs/tmp_cpu2/yeseo1ee/d1_hf) and RES (= $D1HF_RES, default
analysis/d1_hf/results) (spec 11); every D1 / D1PAIR input is read only. The D1 label loader and the D'' list come from
analysis/d1_pair/d1pair_common.py (spec 1).
Optional overrides (D1-G2 Phase T-3 / H, docs/specs/D1G2.md 4, 9; unset = the D1HF paths): D1HF_RES (where the results
files are written), D1HF_TEMPLATES (read-only root of the DFT-TS HF inputs <T>/hf/*.inp, default $HF/inputs),
D1HF_G2_ROOT (the G2 structures <rxn_id>/{.done, ts.xyz, rel1.xyz, rel2.xyz}, default
/gpfs/tmp_cpu2/yeseo1ee/d1_g2/g2_final). X_MANIFEST is always analysis/d1_hf/results/X_manifest.csv (read only)."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RES = Path(os.environ.get("D1HF_RES", HERE / "results"))
X_MANIFEST = HERE / "results" / "X_manifest.csv"        # D1HF Phase X (read only; the same file whatever RES is)
HF = Path(os.environ.get("D1HF_SCRATCH", "/gpfs/tmp_cpu2/yeseo1ee/d1_hf"))
INPUTS, RUNS, LOGS = HF / "inputs", HF / "runs", HF / "logs"
TEMPLATES = Path(os.environ.get("D1HF_TEMPLATES", INPUTS))  # <T>/hf/*.inp = the DFT-TS HF inputs (read only)
LABELS_JSON = HF / "labels_d1_hf.json"
sys.path.insert(0, str(REPO / "analysis" / "d1_pair"))
import d1pair_common as DP  # noqa: E402              (load_d1_labels, DPP, D1_JOBS, NURION_TARS, UBAI_JOBS, ...)

ORCA_DIR = Path("/home1/yeseo1ee/orca_6_1_1_avx2")
ORCA_BIN = ORCA_DIR / "orca"
ORCA_VERSION, ORCA_GIT = "6.1.1", "487d211c"            # header of every label output (C0020 canary, G1 runs)
KCAL = 627.5094740631                                   # = label_true/scripts/_common.py
CALCS = ("frag1_dist", "frag2_dist", "frag1_rel", "frag2_rel", "eda")       # run order: 4 fragments, then the EDA
SUB = (("B3LYP", "HF"), ("def2-TZVP", "6-31G*"))        # spec 2: the only two words that change
EPS_WATER = 78.3550
G1_FINAL = DP.G1_ROOT_D1F                               # /gpfs/tmp_cpu2/yeseo1ee/d1_pair/g1_final/<rxn_id>/ts.xyz
G2_FINAL = Path(os.environ.get("D1HF_G2_ROOT", "/gpfs/tmp_cpu2/yeseo1ee/d1_g2/g2_final"))   # D1G2 2: G1-format folders
GEOM_ROOT = {"g1": G1_FINAL, "g2": G2_FINAL}            # structure root of each non-DFT geometry
# gates (spec 4). The spec table says "same as the label" but prints tighter g2 / g3 thresholds than the label used;
# decision = the label thresholds (D1_build config_run.yaml: 1e-6 / 0.11 / 0.11), the spec values are reported too
G1_TOL, G2_TOL, G3_TOL = 1e-6, 0.11, 0.11
G2_TOL_SPEC, G3_TOL_SPEC = 0.01, 1e-3
ENUC_TOL_EH = 1e-6                                       # = d1_run gates.py enuc_tol_eh


def stop(msg):
    print(f"STOP: {msg}", file=sys.stderr, flush=True)
    sys.exit(3)


def assert_path(p):
    """writes only under $HF or analysis/d1_hf/results (spec 11); exit 3 otherwise."""
    q = Path(os.path.abspath(p))
    for root in (HF, RES):
        try:
            q.relative_to(Path(os.path.abspath(root)))
            return q
        except ValueError:
            continue
    stop(f"write path {q} outside $HF / analysis/d1_hf/results")


def write_atomic(path, data):
    """str / bytes, or a callable taking the temporary path; written to <name>.tmp then renamed."""
    path = assert_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if callable(data):
        data(tmp)
    else:
        tmp.write_bytes(data.encode() if isinstance(data, str) else data)
    os.replace(tmp, path)
    return path


def write_json(path, obj):
    return write_atomic(path, json.dumps(obj, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def read_text(path):
    """text of an ORCA output, plain or .gz."""
    p = Path(path)
    if p.suffix == ".gz":
        with gzip.open(p, "rt", errors="replace") as f:
            return f.read()
    return p.read_text(errors="replace")


def targets():
    """spec 1: labels_d1.json rows (d1pair_common.load_d1_labels: sha256 / 9,496 / status ok checks) whose skeleton is not
    in d1pair_common.DPP, sorted by job_id."""
    lab = DP.load_d1_labels()
    return sorted((d for d in lab if d["skeleton"] not in DP.DPP), key=lambda d: d["job_id"])


# ---------------------------------------------------------------- ORCA input parsing
XYZ_LINE = re.compile(r"^\s*([A-Z][a-z]?)(?:\((\d)\))?\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$")


def parse_inp(text, atoms=True):
    """{bang: [tokens of the '!' lines], blocks: {name: [lines]}, frag: {FRAG1: str, ...}, charge, mult,
    atoms: [(element, tag or None, x, y, z)]}."""
    out = dict(bang=[], blocks={}, frag={}, charge=None, mult=None, atoms=[], pal=None)
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith("!"):
            out["bang"] += s[1:].split()
        elif s.startswith("%"):
            name = s[1:].split()[0].lower()
            if name == "maxcore" or s.lower().endswith(" end"):           # one-line blocks (%maxcore N, %pal nprocs N end)
                out["blocks"][name] = [s]
                if name == "pal":
                    out["pal"] = int(s.split()[2])
            else:
                body = [s]
                i += 1
                while i < len(lines) and lines[i].strip().lower() != "end":
                    body.append(lines[i].strip())
                    i += 1
                out["blocks"][name] = body
                if name == "eda":
                    for b in body[1:]:
                        k, _, v = b.partition(" ")
                        out["frag"][k] = v.strip().strip('"') if k in ("FRAG1", "FRAG2") else v.strip()
        elif atoms and s.startswith("*") and "xyz" in s:
            t = s.split()
            out["charge"], out["mult"] = int(t[2]), int(t[3])
            i += 1
            while i < len(lines) and lines[i].strip() != "*":
                m = XYZ_LINE.match(lines[i])
                if not m:
                    raise ValueError(f"bad coordinate line: {lines[i]!r}")
                out["atoms"].append((m.group(1), int(m.group(2)) if m.group(2) else None,
                                     float(m.group(3)), float(m.group(4)), float(m.group(5))))
                i += 1
        i += 1
    return out


def read_xyz(path):
    lines = Path(path).read_text().splitlines()
    n = int(lines[0].split()[0])
    at = []
    for ln in lines[2:2 + n]:
        t = ln.split()
        at.append((t[0], float(t[1]), float(t[2]), float(t[3])))
    return at


def smd_ok(p, kind):
    """SMD(water) reached the run: eda.inp has SMD(water) on the '!' line and in both FRAG strings and no %cpcm block;
    a standalone fragment input has SMD(water) on the '!' line, or CPCM(water) plus a %cpcm block with 'smd true' and
    smdsolvent "water" (the label convention, d1_run method_lines.py; CLAUDE.md: the same SMD model)."""
    if kind == "eda":
        return ("SMD(water)" in p["bang"] and all("SMD(water)" in p["frag"].get(f, "").split() for f in ("FRAG1", "FRAG2"))
                and "cpcm" not in p["blocks"])
    if "SMD(water)" in p["bang"]:
        return True
    blk = [b.lower().replace(" ", "") for b in p["blocks"].get("cpcm", [])]
    return "CPCM(water)" in p["bang"] and "smdtrue" in blk and 'smdsolvent"water"' in blk


# ---------------------------------------------------------------- ORCA output parsing (label_true/scripts/stage3_parse.py)
RE_FSPE = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)")
RE_EDA = re.compile(r"Energy Term\s+Hartree\s+Kcal/mol\s*\n-+\s*\n(.*?)\n\s*-+", re.S)
RE_ROW = re.compile(r"^\s+(.+?)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$", re.M)
RE_ENUC = re.compile(r"^Nuclear Repulsion\s*:\s*(-?\d+\.\d+)\s*Eh", re.M)          # full-precision line
RE_VER = re.compile(r"Program Version\s+(\S+)")
RE_GIT = re.compile(r"\(GIT:\s*\$([0-9a-f]+)\$\)")
RE_TIME = re.compile(r"TOTAL RUN TIME:\s*(\d+) days (\d+) hours (\d+) minutes (\d+) seconds (\d+) msec")
RE_EPS = re.compile(r"^\s*Epsilon\s*\.+\s*(-?\d+\.\d+)", re.M)
RE_SOLV = re.compile(r"^\s*Solvent:\s*\.*\s*(\S+)", re.M)


def terminated(text):
    return "ORCA TERMINATED NORMALLY" in text[-4096:]


def run_seconds(text):
    m = RE_TIME.findall(text)
    if not m:
        return None
    d, h, mi, s, ms = map(int, m[-1])
    return d * 86400 + h * 3600 + mi * 60 + s + ms / 1000


def eda_table(text):
    """[(row label, Hartree)] of the single EDA table (None if there is not exactly one)."""
    t = RE_EDA.findall(text)
    if len(t) != 1:
        return None
    return [(m.group(1).strip(), float(m.group(2))) for m in RE_ROW.finditer(t[0])]
RE_BASIS = re.compile(r"Your calculation utilizes the basis:\s*(\S+)")
RE_METHOD = re.compile(r"Hamiltonian:\s*\n\s*(?:Density Functional|Ab initio Hamiltonian)\s+Method\s+\.+\s+(\S+)")
