#!/usr/bin/env python3
"""bath_am1_start.py — REV4 §4-4: where did Espley's ds3 ([3+2]) AM1 TS optimisations start?

Espley 2024's Gaussian16 logs are in the Bath archive BATH-01480 v2 (data_archive_files.zip, ~4.3 GB; the landing
page of v1 BATH-01398 lists no file: "Access restrictions apply"). Nothing is downloaded whole: HTTP Range requests
read the zip tail (EOCD, ZIP64 locator + record), the central directory and, for the sampled logs only, the local
header + the raw deflate stream (zlib, CRC-checked). urllib keeps the Range header across redirects; a server that
answers a Range request with anything but 206 is refused before any body is read.

  1. every archive in ARCHIVES: central directory -> three_two_cycloaddition/optimisations/am1_ts/ts_<n>.out; the
     first archive (list order) holding them is the ds3 archive (the 2026-09-29 session saw 3,663 in BATH-01480 v2)
  2. ts_<n>.out -> Coley rxn id n. Checked before sampling; any failure prints diagnostics and stops (exit 3):
       names   every am1_ts file is ts_<int>.out, no duplicates
       E       Espley reaction_number = Coley rxn_id: their DFT interaction / summed distortion (manual_tt_solvent.pkl)
               vs labels_all.json, MAE < 1 kcal/mol at offset 0 and > 2x that at offsets ±1, ±2 (neighbouring
               Coley ids are often regioisomers of one reactant pair)
       P       their feature_extraction/_tt/mapping.pkl ('ts_<n>.out' -> {reactant_1, reactant_2: TS atom -> fragment
               atom, 1-based}) splits >= 90 % of the ML set, and no offset k != 0 matches rxn n+k better than offset 0
               does rxn n: fragment TS-atom sets = label A_idx (input_meta.csv) or its complement (counted when
               >= 90 %), or sorted fragment sizes = (n_f1, n_f2) (atom-order independent). atom_order_is_coley = the
               A_idx share is >= 90 % at offset 0 and larger than elsewhere (e.g. ts_9: TS atoms 1-4, 9-15 = A_idx
               0-3, 8-14 of rxn 9); a non-Coley order is recorded, not a stop (mapped RMSD and m4 do not need it)
     per sampled log (exit 3 after all 20): formula of the first and last geometry = Coley TS of rxn n and atom count
     = mapping.pkl for every log; m4 = Espley's reacting_distance_0/1_ts_ts (row n) among the inter-fragment heavy-atom
     distances of the final AM1 geometry (|dev| < M4_TOL_A) for >= M4_MIN_OK logs (the one check that tells rxn n
     from its regioisomer neighbours). Reported only: element order = Coley; m4 pairs = formed_pairs_ts
  3. candidates = sorted ML-set reaction_numbers (manual_tt_solvent.pkl, 3,510) with an am1_ts log; 20 drawn with
     numpy default_rng(20260930)
  4. start geometry = the Cartesian 'Symbolic Z-matrix' echo of the first job, else the first 'Input orientation'
     (then 'Standard orientation') block; final geometry = the last orientation block
  5. heavy-atom RMSD, rmsd.mapped_heavy_rmsd (D1 snapshot: graph isomorphisms + mirror images, Kabsch; inf = heavy
     graphs not isomorphic) and same-atom-order Kabsch, of the start and the final AM1 TS vs the Coley DFT TS
     (PROF/<rid>/<ts_file>); for context the start vs Coley's DFT OptTS input (first geometry of
     frequency_logs/<ts>_optts_g16.log = the TS guess Coley optimised), the TS_imag_mode.xyz frames, the product and
     the AM1 final geometry
  6. verdict: median mapped RMSD(start, DFT TS) < 0.01 A => AM1 started from the DFT TS => G1 is the input-matched
     arm; otherwise the nearest Coley structure is reported

  python bath_am1_start.py [run]    # -> results_rev4/espley_am1_start_rmsd.{csv,json}; skipped when both exist
  python bath_am1_start.py archives # step 1 only (prints the archive report)
Cache: $R4_SCRATCH/bath/<archive>/{cd.bin, cd_meta.json, members/ts_<n>.out} (reused; members CRC-checked).
Exit: 0 ok / skipped, 2 no readable archive holds ds3, 3 mapping not established, 4 no archive could be read (network /
HTTP: resubmit); for 2-4 the diagnostics are printed and written to $R4_SCRATCH/bath/mapping_diagnostics.json, and
results_rev4/ is not touched.
"""
from __future__ import annotations

import collections
import hashlib
import http.client
import json
import os
import pickle
import re
import ssl
import struct
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fragmenter as fr  # noqa: E402  (imported first: rmsd.py below reuses this module)

D1_SNAPSHOT = Path(os.environ.get("D1_SNAPSHOT", "/gpfs/tmp_cpu2/yeseo1ee/espley_rev4/d1_snapshot_d573111e"))
D1_COMMIT = "d573111eb899fd78c71e8ce4ac5c0f53553d987b"
sys.path.insert(0, str(D1_SNAPSHOT / "D1_build/validation/analysis"))
import rmsd as rm  # noqa: E402

LABELS = Path(os.environ.get("ESPLEY_LABELS") or HERE.parent.parent / "labels_all.json")
PROF = Path(os.environ.get("ESPLEY_PROF", "/gpfs/tmp_cpu2/yeseo1ee/eda_asm_raw/dipolar_cycloaddition/extracted/full_dataset_profiles"))
META = Path(os.environ.get("ESPLEY_META", "/gpfs/home1/yeseo1ee/projects/eda-asm-prediction/label_true/work/input_meta.csv"))
ESP = Path(os.environ.get("ESPLEY_REPO_DATA", "/gpfs/tmp_cpu2/yeseo1ee/espley_compare"))
ESP_DS = ESP / "feature_selection/_f_selection/tt/manual_tt_solvent.pkl"
ESP_MAP = ESP / "feature_extraction/_tt/mapping.pkl"
BATH = Path(os.environ.get("R4_SCRATCH", "/gpfs/tmp_cpu2/yeseo1ee/espley_rev4")) / "bath"
EARLIER_CD = Path(os.environ.get("BATH_EARLIER_CD", "/gpfs/tmp_cpu2/yeseo1ee/espley_archive/cd.bin"))  # 2026-09-29
RES = HERE / "results_rev4"
OUT_CSV, OUT_JSON = RES / "espley_am1_start_rmsd.csv", RES / "espley_am1_start_rmsd.json"

ARCHIVES = [("BATH-01480_v2", "https://researchdata.bath.ac.uk/1480/1/data_archive_files.zip"),
            ("BATH-01398_v1", "https://researchdata.bath.ac.uk/1398/1/data_archive_files.zip")]
if os.environ.get("BATH_ARCHIVES"):                  # "tag=url,tag=url"
    ARCHIVES = [tuple(a.split("=", 1)) for a in os.environ["BATH_ARCHIVES"].split(",")]
LANDING = {"BATH-01480_v2": "researchdata.bath.ac.uk/1480 (read 2026-09-30): version 2 of 31 Jan 2025, one file "
                            "data_archive_files.zip (4 GB), Gaussian16 A.03/C.01 outputs, CC-BY 4.0",
           "BATH-01398_v1": "researchdata.bath.ac.uk/1398 (read 2026-09-30): version 1 of 21 Oct 2024, 'Access "
                            "restrictions apply: Due to the duplication of a file, a new version was created'; no file "
                            "listed, the URL is the EPrints pattern of v2"}
SEED, N_PICK = 20260930, 20
VERDICT_A = 0.01
OFFSETS = (-2, -1, 0, 1, 2)
E_MAE_MAX, E_RATIO_MIN, P_SHARE_MIN = 1.0, 2.0, 0.90
M4_TOL_A, M4_MIN_OK = 0.005, 15
AM1_TS_RE = re.compile(r"(?:^|/)three_two_cycloaddition/optimisations/am1_ts/([^/]+)$")
AM1_NAME_RE = re.compile(r"ts_(\d+)\.out")
UA = "eda-asm-prediction rev4 bath_am1_start.py (HTTP Range reader)"
CHUNK = 4 << 20
try:
    import certifi
    SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CTX = ssl.create_default_context()


# ---------------------------------------------------------------- HTTP Range + zip
class RangeRefused(Exception):
    pass


def http_range(url, a, b, tries=5):
    """(bytes a..b inclusive, total size) by one Range request; a non-206 reply is refused unread."""
    err = None
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}", "User-Agent": UA})
            with urllib.request.urlopen(req, timeout=120, context=SSL_CTX) as r:
                if r.status != 206:
                    raise RangeRefused(f"HTTP {r.status} ({r.headers.get('Content-Type')}) to a Range request")
                cr = r.headers.get("Content-Range", "")
                data = r.read()
            m = re.match(r"bytes (\d+)-(\d+)/(\d+|\*)", cr)
            if not m or int(m.group(1)) != a or len(data) != b - a + 1:
                raise IOError(f"bad range reply: Content-Range '{cr}', {len(data)} bytes for {a}-{b}")
            return data, (int(m.group(3)) if m.group(3) != "*" else None)
        except RangeRefused:
            raise
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404, 410, 416):
                raise
            err = e
        except (OSError, http.client.HTTPException) as e:
            err = e
        print(f"  retry {t + 1}/{tries} bytes {a}-{b}: {type(err).__name__}: {err}", flush=True)
        time.sleep(15 * (t + 1))
    raise err


def fetch_big(url, a, n):
    return b"".join(http_range(url, s, min(s + CHUNK, a + n) - 1)[0] for s in range(a, a + n, CHUNK))


CD_FMT = struct.Struct("<4sHHHHHHIIIHHHHHII")         # central directory file header (46 bytes)
LOC_FMT = struct.Struct("<4sHHHHHIIIHH")              # local file header (30 bytes)


def zip64_extra(extra, usize, csize, lho):
    q = 0
    while q + 4 <= len(extra):
        hid, hsz = struct.unpack_from("<HH", extra, q)
        if hid == 1:
            k, out = q + 4, []
            for v in (usize, csize, lho):
                if v == 0xFFFFFFFF:
                    v = struct.unpack_from("<Q", extra, k)[0]; k += 8
                out.append(v)
            return tuple(out)
        q += 4 + hsz
    raise ValueError("ZIP64 sizes without a ZIP64 extra field")


def parse_cd(buf, n_expected=None):
    ents, p = [], 0
    while p + CD_FMT.size <= len(buf) and buf[p:p + 4] == b"PK\x01\x02":
        (_, _, _, flags, method, _, _, crc, csize, usize, nlen, elen, clen, _, _, _, lho) = CD_FMT.unpack_from(buf, p)
        q = p + CD_FMT.size
        name = buf[q:q + nlen].decode("utf-8" if flags & 0x800 else "cp437")
        if 0xFFFFFFFF in (csize, usize, lho):
            usize, csize, lho = zip64_extra(buf[q + nlen:q + nlen + elen], usize, csize, lho)
        ents.append(dict(name=name, flags=flags, method=method, crc=crc, csize=csize, usize=usize, lho=lho))
        p = q + nlen + elen + clen
    if n_expected is not None and len(ents) != n_expected:
        raise ValueError(f"central directory: parsed {len(ents)} entries, EOCD says {n_expected}")
    return ents


def read_cd(tag, url):
    """(entries, info) of one archive's central directory; cached as $R4_SCRATCH/bath/<tag>/cd.bin."""
    d = BATH / tag
    cd_f, meta_f = d / "cd.bin", d / "cd_meta.json"
    if cd_f.is_file() and meta_f.is_file():
        info = json.loads(meta_f.read_text())
        buf = cd_f.read_bytes()
        if info.get("url") == url and len(buf) == info["cd_size"]:
            return parse_cd(buf, info["n_entries_eocd"]), dict(info, cached=True)
    head, size = http_range(url, 0, 3)
    if head != b"PK\x03\x04" or size is None:
        raise RangeRefused(f"not a zip archive (first bytes {head!r}, size {size})")
    t0 = max(0, size - (22 + 0xFFFF + 20))          # EOCD + max comment + ZIP64 locator
    tail = http_range(url, t0, size - 1)[0]
    p = tail.rfind(b"PK\x05\x06")
    if p < 0:
        raise ValueError("no end-of-central-directory record")
    _, _, _, _, n_tot, cd_size, cd_off, _ = struct.unpack_from("<4sHHHHIIH", tail, p)
    zip64 = p >= 20 and tail[p - 20:p - 16] == b"PK\x06\x07"
    if zip64:
        off64 = struct.unpack_from("<4sIQI", tail, p - 20)[2]
        rec = tail[off64 - t0:off64 - t0 + 56] if off64 >= t0 else http_range(url, off64, off64 + 55)[0]
        if rec[:4] != b"PK\x06\x06":
            raise ValueError("bad ZIP64 end-of-central-directory record")
        n_tot, cd_size, cd_off = struct.unpack("<4sQHHIIQQQQ", rec[:56])[7:]
    elif n_tot == 0xFFFF or 0xFFFFFFFF in (cd_size, cd_off):
        raise ValueError("ZIP64 sentinel values without a ZIP64 locator")
    buf = fetch_big(url, cd_off, cd_size)
    ents = parse_cd(buf, n_tot)
    info = dict(url=url, size=size, zip64=zip64, cd_off=cd_off, cd_size=cd_size, n_entries_eocd=n_tot,
                cd_sha256=hashlib.sha256(buf).hexdigest(), fetched_utc=utc())
    d.mkdir(parents=True, exist_ok=True)
    write_atomic(cd_f, lambda f: f.write_bytes(buf))
    write_atomic(meta_f, lambda f: f.write_text(json.dumps(info, indent=1)))
    return ents, dict(info, cached=False)


def read_member(url, e, cache: Path):
    """(uncompressed bytes, from cache?) of one entry: local header + raw deflate by Range, CRC-checked, cached."""
    if cache.is_file():
        data = cache.read_bytes()
        if len(data) == e["usize"] and zlib.crc32(data) == e["crc"]:
            return data, True
    if e["flags"] & 1:
        raise ValueError(f"{e['name']}: encrypted")
    sig, *_, nlen, elen = LOC_FMT.unpack(http_range(url, e["lho"], e["lho"] + LOC_FMT.size - 1)[0])
    if sig != b"PK\x03\x04":
        raise ValueError(f"{e['name']}: no local header at offset {e['lho']}")
    a = e["lho"] + LOC_FMT.size + nlen + elen
    raw = http_range(url, a, a + e["csize"] - 1)[0] if e["csize"] else b""
    if e["method"] == 8:
        z = zlib.decompressobj(-15)
        data = z.decompress(raw) + z.flush()
    elif e["method"] == 0:
        data = raw
    else:
        raise ValueError(f"{e['name']}: compression method {e['method']} (only stored / deflate)")
    if len(data) != e["usize"] or zlib.crc32(data) != e["crc"]:
        raise ValueError(f"{e['name']}: size / CRC mismatch after inflate")
    cache.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(cache, lambda f: f.write_bytes(data))
    return data, False


def archive_report(tag, url):
    """(report, {n: entry} of the ds3 AM1 TS logs or None)."""
    rep = dict(tag=tag, url=url, landing=LANDING.get(tag))
    try:
        ents, info = read_cd(tag, url)
    except Exception as e:  # noqa: BLE001
        rep.update(status="unavailable", error=f"{type(e).__name__}: {e}", holds_ds3=False)
        return rep, None
    ds3, am1, odd = collections.Counter(), {}, []
    for e in ents:
        name = e["name"]
        if name.endswith("/"):
            continue
        k = name.find("three_two_cycloaddition/")
        if k >= 0:
            ds3["/".join(name[k:].split("/")[1:-1])] += 1
        m = AM1_TS_RE.search(name)
        if m:
            mm = AM1_NAME_RE.fullmatch(m.group(1))
            if mm is None or int(mm.group(1)) in am1:
                odd.append(name)
            else:
                am1[int(mm.group(1))] = e
    rep.update(status="ok", size=info["size"], zip64=info["zip64"], n_entries=len(ents), cd_off=info["cd_off"],
               cd_size=info["cd_size"], cd_sha256=info["cd_sha256"], cd_cached=info["cached"],
               ds3_files_by_folder=dict(sorted(ds3.items())), n_am1_ts=len(am1),
               am1_ts_id_range=[min(am1), max(am1)] if am1 else None, am1_ts_unparsed=odd, holds_ds3=len(am1) > 0)
    if EARLIER_CD.is_file() and EARLIER_CD.stat().st_size == info["cd_size"]:
        rep["earlier_session_cd_bin_identical"] = EARLIER_CD.read_bytes() == (BATH / tag / "cd.bin").read_bytes()
    return rep, am1


# ---------------------------------------------------------------- Gaussian16 logs
DASH = re.compile(r"^\s*-{5,}\s*$")
Z2SYM = {z: s for s, z in fr.Z.items()}


def g16_sym(tok):
    if tok.isdigit():
        return Z2SYM[int(tok)]
    s = re.match(r"[A-Za-z]{1,2}", tok).group(0).capitalize()
    return s if s in fr.COV else s[0]


def cart_line(t):
    """[x, y, z] of a Cartesian input line 'El x y z' or 'El flag x y z', else None (Z-matrix, variables)."""
    if len(t) not in (4, 5) or (len(t) == 5 and not t[1].lstrip("-").isdigit()):
        return None
    try:
        return [float(v.replace("D", "E")) for v in t[-3:]]
    except ValueError:
        return None


def orient_blocks(L, key):
    out = []
    for i, l in enumerate(L):
        if key in l:
            syms, xyz, k = [], [], i + 5
            while k < len(L) and not DASH.match(L[k]):
                t = L[k].split()
                syms.append(Z2SYM.get(int(t[1]), f"X{t[1]}")); xyz.append([float(v) for v in t[-3:]]); k += 1
            out.append((syms, np.array(xyz)))
    return out


def parse_g16(text):
    L = text.splitlines()
    g = dict(routes=[], title=None, charge=None, mult=None, start=None, start_source=None, final=None)
    for i, l in enumerate(L):
        if l.startswith(" #") and i and DASH.match(L[i - 1]):
            j = i
            while j < len(L) and not DASH.match(L[j]):
                j += 1
            g["routes"].append("".join(x[1:] for x in L[i:j]).strip())
    i = next((k for k, l in enumerate(L) if l.strip() == "Symbolic Z-matrix:"), None)
    if i is not None:
        if DASH.match(L[i - 1]):
            m = i - 2
            while m > 0 and not DASH.match(L[m]):
                m -= 1
            g["title"] = " ".join(x.strip() for x in L[m + 1:i - 1]).strip()
        k = i + 1
        while k < len(L) and L[k].strip().startswith("Charge"):
            c = re.search(r"Charge\s*=\s*(-?\d+)\s+Multiplicity\s*=\s*(\d+)", L[k])
            if c and g["charge"] is None:
                g["charge"], g["mult"] = int(c.group(1)), int(c.group(2))
            k += 1
        syms, xyz = [], []
        while k < len(L) and L[k].strip():
            t = L[k].split()
            v = cart_line(t)
            if v is None:
                syms = []
                break
            syms.append(g16_sym(t[0])); xyz.append(v); k += 1
        if syms:
            g["start"], g["start_source"] = (syms, np.array(xyz)), "symbolic_zmatrix"
    key = "Input orientation:" if "Input orientation:" in text else "Standard orientation:"
    blocks = orient_blocks(L, key)
    if g["start"] is None and blocks:
        g["start"], g["start_source"] = blocks[0], "first_" + key.split()[0].lower() + "_orientation"
    g["final"] = blocks[-1] if blocks else None
    v = re.search(r"Gaussian (\d+), Revision ([A-Z]\.\d+)", text)
    h = text.rfind("Harmonic frequencies")
    freqs = []
    for l in re.findall(r"Frequencies --\s+(.*)", text[h:] if h >= 0 else text):
        for x in l.split():
            try:
                freqs.append(float(x))
            except ValueError:
                pass
    st = re.search(r"Stoichiometry\s+(\S+)", text)
    g.update(gaussian=f"G{v.group(1)} {v.group(2)}" if v else None, n_orient=len(blocks),
             n_opt_steps=len(re.findall(r"Step number\s+\d+", text)),
             n_normal_term=text.count("Normal termination of Gaussian"), n_error_term=text.count("Error termination"),
             stationary_point=("Stationary point found" in text), stoichiometry=st.group(1) if st else None,
             geom_from_chk=("from the checkpoint file" in text
                            or any(re.search(r"geom\s*=\s*\(?\s*(all)?check", r, re.I) for r in g["routes"])),
             imag_freqs_cm=[f for f in freqs if f < 0] if freqs else None)
    return g


# ---------------------------------------------------------------- geometry helpers
def formula(syms):
    c = collections.Counter(syms)
    order = [s for s in ("C", "H") if s in c] + sorted(s for s in c if s not in ("C", "H"))
    return "".join(f"{s}{c[s] if c[s] > 1 else ''}" for s in order)


def mrmsd(a, b):
    """rmsd.mapped_heavy_rmsd; nan if a structure is missing or the formulas differ, inf if the heavy graphs differ."""
    if a is None or b is None or sorted(a[0]) != sorted(b[0]):
        return float("nan")
    return float(rm.mapped_heavy_rmsd(a, b))


def so_rmsd(a, b):
    """Heavy-atom Kabsch RMSD in the given atom order; nan unless the element sequences are identical."""
    if a is None or b is None or list(a[0]) != list(b[0]):
        return float("nan")
    h = [i for i, s in enumerate(a[0]) if s != "H"]
    return rm.kabsch_rmsd(np.asarray(a[1])[h], np.asarray(b[1])[h])


def best_rmsd(a, b):
    """min(mapped, same-order) heavy-atom RMSD; inf if neither applies."""
    return float(np.nanmin([mrmsd(a, b), so_rmsd(a, b), np.inf]))


def read_frames(path):
    L, out, k = Path(path).read_text().splitlines(), [], 0
    while k < len(L) and L[k].strip():
        n = int(L[k]); rows = [l.split() for l in L[k + 2:k + 2 + n]]
        out.append(([r[0] for r in rows], np.array([[float(v) for v in r[1:4]] for r in rows]))); k += 2 + n
    return out


def split_of(entry):
    """(frag1, frag2) 0-based TS atom sets of one mapping.pkl entry, or None."""
    if not isinstance(entry, dict) or not {"reactant_1", "reactant_2"} <= set(entry):
        return None
    r1, r2 = entry["reactant_1"], entry["reactant_2"]
    for side in ("keys", "values"):
        for base in (1, 0):
            s1 = {int(x) - base for x in getattr(r1, side)()}
            s2 = {int(x) - base for x in getattr(r2, side)()}
            if not s1 & s2 and s1 | s2 == set(range(len(s1) + len(s2))):
                return frozenset(s1), frozenset(s2)
    return None


def coley_ts_file(rid, lab):
    f = (lab.get(rid) or {}).get("ts_file")
    if not f:
        c = [p.name for p in (PROF / str(rid)).glob("TS_*.xyz") if p.name != "TS_imag_mode.xyz"]
        f = c[0] if len(c) == 1 else None
    return f


def coley_optts(pdir, ts_file):
    """(parsed log, member name) of Coley's DFT OptTS log for this TS, or (None, None)."""
    f = pdir / "frequency_logs.tar.gz"
    if not f.is_file():
        return None, None
    stem = Path(ts_file).stem
    with tarfile.open(f) as tf:
        logs = [m for m in tf.getmembers() if m.isfile() and m.name.endswith("_optts_g16.log")]
        pick = [m for m in logs if Path(m.name).name == f"{stem}_optts_g16.log"] or (logs if len(logs) == 1 else [])
        if not pick:
            return None, None
        text = tf.extractfile(pick[0]).read().decode("utf-8", errors="replace")
    return parse_g16(text), pick[0].name


# ---------------------------------------------------------------- mapping evidence
def energy_evidence(ids, esp, lab):
    ok = {i: r for i, r in lab.items() if r.get("status") == "ok"}
    ei, sd = esp["interaction_energies_dft"].to_numpy(float), esp["sum_distortion_energies_dft"].to_numpy(float)
    by = {}
    for k in OFFSETS:
        a = np.array([(abs(r["eint_spe_kcal"] + ei[j]), abs(r["d1_kcal"] + r["d2_kcal"] - sd[j]))
                      for j, n in enumerate(ids) if (r := ok.get(n + k)) is not None], float).reshape(-1, 2)
        by[k] = dict(n=len(a), mae_interaction=float(np.nanmean(a[:, 0])) if len(a) else float("nan"),
                     mae_sum_distortion=float(np.nanmean(a[:, 1])) if len(a) else float("nan"))
    c = ("mae_interaction", "mae_sum_distortion")
    passed = (all(by[0][x] < E_MAE_MAX for x in c)
              and all(by[k][x] > E_RATIO_MIN * by[0][x] for k in OFFSETS if k for x in c))
    return dict(passed=bool(passed), by_offset=by,
                rule=f"Espley reaction_number n vs our rxn_id n+k: MAE(eint_spe + interaction_energies_dft), "
                     f"MAE(d1 + d2 - sum_distortion_energies_dft); offset 0 < {E_MAE_MAX} kcal/mol and every "
                     f"other offset > {E_RATIO_MIN}x offset 0")


def partition_evidence(ids, lab, meta, mapping):
    splits = {i: split_of(mapping.get(f"ts_{i}.out")) for i in ids}
    by = {}
    for k in OFFSETS:
        n = agree = nz = zagree = 0
        for i, sp in splits.items():
            if sp is None:
                continue
            a, L = meta.get(i + k), lab.get(i + k) or {}
            if a is not None:
                n += 1; agree += a in sp
            if L.get("n_f1") is not None and L.get("n_f2") is not None:
                nz += 1; zagree += sorted(map(len, sp)) == sorted((L["n_f1"], L["n_f2"]))
        by[k] = dict(n=n, share_split_eq_A_idx=agree / n if n else float("nan"),
                     n_sizes=nz, share_sizes_eq_label=zagree / nz if nz else float("nan"))
    n_at = [(len(s[0]) + len(s[1]), (lab.get(i) or {}).get("n_atoms")) for i, s in splits.items() if s is not None]
    s, z = ({k: by[k][c] for k in OFFSETS} for c in ("share_split_eq_A_idx", "share_sizes_eq_label"))
    other = [k for k in OFFSETS if k]
    n_valid = sum(v is not None for v in splits.values())
    order_coley = s[0] >= P_SHARE_MIN and all(s[0] > s[k] for k in other)
    better_elsewhere = [k for k in other if (s[k] >= P_SHARE_MIN and s[k] > s[0]) or z[k] > z[0]]
    passed = n_valid >= P_SHARE_MIN * len(ids) and not better_elsewhere
    return dict(passed=bool(passed), atom_order_is_coley=bool(order_coley), offsets_better_than_0=better_elsewhere,
                by_offset=by, n_ml=len(ids), n_ml_with_mapping_key=sum(f"ts_{i}.out" in mapping for i in ids),
                n_ml_with_valid_split=n_valid,
                n_atoms_eq_label=sum(a == b for a, b in n_at if b is not None),
                n_atoms_compared=sum(b is not None for _, b in n_at),
                rule=f"mapping.pkl 'ts_<n>.out' split for >= {P_SHARE_MIN} of the ML set, and no offset k != 0 where "
                     f"rxn n+k matches better than rxn n: fragment TS-atom sets = A_idx (input_meta.csv; counted when "
                     f">= {P_SHARE_MIN}) or sorted fragment sizes = (n_f1, n_f2) (order-independent); "
                     f"atom_order_is_coley = A_idx share >= {P_SHARE_MIN} at offset 0 and larger than elsewhere")


# ---------------------------------------------------------------- one sampled log
def one(rid, e, url, tag, esp_row, lab, meta, mapping):
    base = e["name"].rsplit("/", 1)[-1]
    data, cached = read_member(url, e, BATH / tag / "members" / base)
    g = parse_g16(data.decode("utf-8", errors="replace"))
    pdir = PROF / str(rid)
    tsf = coley_ts_file(rid, lab)
    ts = fr.read_xyz(pdir / tsf)
    start, final = g["start"], g["final"]
    row = dict(rxn_id=rid, member=e["name"], member_crc32=f"{e['crc']:08x}", member_usize=e["usize"],
               member_csize=e["csize"], from_cache=cached, gaussian=g["gaussian"], routes=" | ".join(g["routes"]),
               title=g["title"], charge=g["charge"], mult=g["mult"], start_source=g["start_source"],
               geom_from_chk=g["geom_from_chk"], n_orient=g["n_orient"], n_opt_steps=g["n_opt_steps"],
               n_normal_term=g["n_normal_term"], n_error_term=g["n_error_term"],
               stationary_point=g["stationary_point"], stoichiometry=g["stoichiometry"],
               imag_freqs_cm=None if g["imag_freqs_cm"] is None else " ".join(f"{f:.1f}" for f in g["imag_freqs_cm"]),
               ts_file=tsf, n_atoms_coley=len(ts[0]), formula_coley=formula(ts[0]),
               n_atoms_am1=len(start[0]) if start else None, formula_am1_start=formula(start[0]) if start else None,
               formula_am1_final=formula(final[0]) if final else None,
               atom_order_same=bool(start is not None and list(start[0]) == list(ts[0])))

    # ---- RMSD vs the Coley DFT TS (headline), AM1 start vs final
    row.update(rmsd_start_dft_ts_A=mrmsd(start, ts), rmsd_final_dft_ts_A=mrmsd(final, ts),
               rmsd_start_final_A=mrmsd(start, final),
               rmsd_start_dft_ts_same_order_A=so_rmsd(start, ts), rmsd_final_dft_ts_same_order_A=so_rmsd(final, ts),
               rmsd_start_final_same_order_A=so_rmsd(start, final))

    # ---- other Coley structures
    og, oname = coley_optts(pdir, tsf)
    oin = og["start"] if og else None
    row.update(coley_optts_log=oname, rmsd_start_coley_optts_input_A=mrmsd(start, oin),
               rmsd_start_coley_optts_input_same_order_A=so_rmsd(start, oin),
               rmsd_coley_optts_input_dft_ts_A=mrmsd(oin, ts),
               rmsd_coley_optts_final_dft_ts_A=mrmsd(og["final"] if og else None, ts))
    fm = pdir / "TS_imag_mode.xyz"
    frames = read_frames(fm) if fm.is_file() else []
    fr_m = [best_rmsd(start, f) for f in frames]
    row.update(n_imag_mode_frames=len(frames), rmsd_start_imag_mode_min_A=min(fr_m) if fr_m else float("nan"),
               imag_mode_best_frame=int(np.argmin(fr_m)) if fr_m else None)
    prods = sorted(pdir.glob("p[0-9]*_*.xyz"))
    pr = [best_rmsd(start, fr.read_xyz(p)) for p in prods]
    row.update(product_files=" ".join(p.name for p in prods), rmsd_start_product_A=min(pr) if pr else float("nan"))

    # ---- forming bonds (label formed_pairs_ts; valid when the element order is Coley's)
    L = lab.get(rid) or {}
    pairs = [tuple(sorted(map(int, p.split("-")))) for p in str(L.get("formed_pairs_ts") or "").split()]
    D0 = fr.dist_matrix(ts[1])
    order = sorted(pairs, key=lambda p: D0[p])
    for nm, s in (("dft", ts), ("start", start), ("final", final), ("coley_optts_input", oin)):
        ok = len(order) == 2 and s is not None and list(s[0]) == list(ts[0])
        D = fr.dist_matrix(s[1]) if ok else None
        row[f"d_form_short_{nm}_A"] = float(D[order[0]]) if ok else float("nan")
        row[f"d_form_long_{nm}_A"] = float(D[order[1]]) if ok else float("nan")

    # ---- mapping checks: mapping.pkl split, Espley's AM1 reacting distances on the final geometry
    sp = split_of(mapping.get(base))
    a_lab = meta.get(rid)
    row.update(mapping_pkl_key=base in mapping, mapping_pkl_n_atoms=len(sp[0]) + len(sp[1]) if sp else None,
               mapping_split_eq_label_A=(a_lab in sp) if sp is not None and a_lab is not None else None)
    rd = [esp_row.get(c) for c in ("reacting_distance_0_ts_ts", "reacting_distance_1_ts_ts")]
    row.update(espley_reacting_d0_A=rd[0], espley_reacting_d1_A=rd[1], m4_max_dev_A=float("nan"), m4_ok=None,
               m4_pairs_eq_label_formed=None)
    if sp is not None and final is not None and len(final[0]) == row["mapping_pkl_n_atoms"] \
            and all(v is not None and np.isfinite(v) for v in rd):
        D = fr.dist_matrix(final[1])
        cross = [(min(i, j), max(i, j)) for i in sp[0] for j in sp[1] if final[0][i] != "H" and final[0][j] != "H"]
        hit = [min(cross, key=lambda p: abs(D[p] - d)) for d in rd]
        row["m4_max_dev_A"] = float(max(abs(D[p] - d) for p, d in zip(hit, rd)))
        row["m4_ok"] = row["m4_max_dev_A"] < M4_TOL_A
        if row["atom_order_same"] and len(order) == 2:
            row["m4_pairs_eq_label_formed"] = set(hit) == set(order)

    # ---- hard per-log mapping checks
    bad = []
    if start is None or final is None:
        bad.append("no_geometry_parsed")
    else:
        if row["formula_am1_start"] != row["formula_coley"]:
            bad.append("start_formula_ne_coley")
        if row["formula_am1_final"] != row["formula_coley"]:
            bad.append("final_formula_ne_coley")
    if sp is not None and row["mapping_pkl_n_atoms"] != row["n_atoms_am1"]:
        bad.append("mapping_pkl_n_atoms_ne_log")
    row["mapping_fail"] = ";".join(bad)

    # ---- nearest Coley structure to the start (min of mapped / same-order heavy RMSD)
    refs = dict(dft_ts=best_rmsd(start, ts), coley_optts_input=best_rmsd(start, oin),
                coley_imag_mode_frame=np.nanmin([row["rmsd_start_imag_mode_min_A"], np.inf]),
                coley_product=np.nanmin([row["rmsd_start_product_A"], np.inf]))
    fin = {k: float(v) for k, v in refs.items() if np.isfinite(v)}
    row["start_nearest_ref"] = min(fin, key=fin.get) if fin else None
    row["start_nearest_rmsd_A"] = fin[row["start_nearest_ref"]] if fin else float("nan")
    return row


# ---------------------------------------------------------------- report
def dist(v):
    v = np.asarray(v, float)
    w = v[~np.isnan(v)]
    f = w[np.isfinite(w)]
    return dict(n=len(v), n_nan=int(np.isnan(v).sum()), n_inf=int(np.isinf(w).sum()),
                median=float(np.median(w)) if len(w) else float("nan"),
                p90=float(np.quantile(w, 0.9)) if len(w) else float("nan"),
                min=float(w.min()) if len(w) else float("nan"), max=float(w.max()) if len(w) else float("nan"),
                n_lt_0p01=int((w < VERDICT_A).sum()), median_finite=float(np.median(f)) if len(f) else float("nan"))


def verdict(df, S):
    med = S["rmsd_start_dft_ts_A"]["median"]
    started = bool(np.isfinite(med) and med < VERDICT_A)
    near = df["start_nearest_ref"].fillna("none").value_counts().to_dict()
    if started:
        text = (f"AM1 TS optimisations started from the Coley DFT TS: median mapped heavy-atom RMSD(start, DFT TS) "
                f"{med:.4f} A < {VERDICT_A} A ({S['rmsd_start_dft_ts_A']['n_lt_0p01']}/{len(df)} logs < {VERDICT_A} A) "
                f"=> G1 (GFN2-xTB re-optimised from the DFT TS) is the input-matched arm to Espley's AM1 geometry.")
    else:
        other = {k: S[c]["median"] for k, c in (("coley_optts_input", "rmsd_start_coley_optts_input_A"),
                                               ("coley_imag_mode_frame", "rmsd_start_imag_mode_min_A"),
                                               ("coley_product", "rmsd_start_product_A"))}
        hit = [k for k, v in other.items() if np.isfinite(v) and v < VERDICT_A]
        text = (f"AM1 did NOT start from the Coley DFT TS: median mapped RMSD(start, DFT TS) {med:.3f} A "
                f">= {VERDICT_A} A. "
                + (f"The start matches Coley's {hit[0]} (median {other[hit[0]]:.4f} A). " if hit else
                   "The start matches none of Coley's files within 0.01 A. ")
                + f"Nearest Coley structure per log: {near}; median RMSD(start, AM1 final) "
                  f"{S['rmsd_start_final_A']['median']:.3f} A.")
    return dict(rule=f"median rmsd.mapped_heavy_rmsd(first AM1 input geometry, Coley DFT TS) < {VERDICT_A} A => AM1 "
                     f"started from the DFT TS => G1 is the input-matched arm; otherwise report the start structure",
                median_start_rmsd_A=med, started_from_dft_ts=started, g1_input_matched_arm=started,
                start_nearest_ref_counts=near, statement=text)


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set, frozenset)):
        return [clean(v) for v in (sorted(x) if isinstance(x, (set, frozenset)) else x)]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return None if np.isnan(x) else ("inf" if np.isinf(x) else x)
    return x


def write_atomic(path, write):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def stop(diag, msg, code=3):
    diag = clean(dict(diag, stop=msg))
    write_atomic(BATH / "mapping_diagnostics.json", lambda f: f.write_text(json.dumps(diag, indent=1)))
    print(json.dumps(diag, indent=1))
    print(f"STOP: {msg}\n(diagnostics: {BATH / 'mapping_diagnostics.json'})", flush=True)
    sys.exit(code)


# ---------------------------------------------------------------- driver
def archives():
    reps, ds3 = [], None
    for tag, url in ARCHIVES:
        print(f"[archive] {tag} {url}", flush=True)
        rep, am1 = archive_report(tag, url)
        reps.append(rep)
        print(json.dumps(clean({k: v for k, v in rep.items() if k != "am1_ts_unparsed"}), indent=1), flush=True)
        if am1 and ds3 is None:
            ds3 = (tag, url, am1, rep)
    return reps, ds3


def run():
    if OUT_CSV.is_file() and OUT_JSON.is_file():
        print(f"exists: {OUT_CSV} {OUT_JSON}")
        return
    reps, ds3 = archives()
    if ds3 is None:
        bad = {r["tag"]: r["error"] for r in reps if r["status"] != "ok"}
        if len(bad) == len(reps):
            stop(dict(archives=reps), f"no candidate archive could be read (network / HTTP; resubmit): {bad}", code=4)
        stop(dict(archives=reps), "no readable candidate archive holds three_two_cycloaddition/optimisations/am1_ts/"
             + (f"; unreadable: {bad}" if bad else ""), code=2)
    tag, url, am1, arep = ds3
    print(f"ds3 archive: {tag} ({len(am1)} AM1 TS logs)", flush=True)

    esp = pd.read_pickle(ESP_DS).reset_index(drop=True)
    ids = [int(i) for i in esp["reaction_number"].astype(int)]
    assert len(set(ids)) == len(ids), "duplicate reaction_number in Espley ds3"
    esp_by = esp.set_index(pd.Index(ids))
    lab = {int(r["rxn_id"]): r for r in json.load(open(LABELS))}
    m = pd.read_csv(META)
    meta = {int(r.rxn_id): frozenset(int(x) for x in str(r.A_idx).split()) for r in m.itertuples() if pd.notna(r.A_idx)}
    mapping = pickle.loads(ESP_MAP.read_bytes())

    ev = dict(names=dict(passed=not arep["am1_ts_unparsed"], unparsed=arep["am1_ts_unparsed"],
                         rule="every am1_ts file is ts_<int>.out, one per int"),
              energy=energy_evidence(ids, esp, lab), partition=partition_evidence(ids, lab, meta, mapping))
    print(json.dumps(clean(ev), indent=1), flush=True)
    if not all(v["passed"] for v in ev.values()):
        stop(dict(archive=arep, mapping_evidence=ev), "ts_<n>.out -> Coley rxn id not established (global checks)")

    cands = sorted(i for i in set(ids) if i in am1)
    no_log = sorted(set(ids) - set(am1))
    chosen = sorted(int(i) for i in np.random.default_rng(SEED).choice(cands, size=N_PICK, replace=False))
    sel = dict(seed=SEED, n_pick=N_PICK, n_ml_set=len(ids), n_ml_with_am1_log=len(cands), ml_without_am1_log=no_log,
               n_am1_logs_not_in_ml=len(set(am1) - set(ids)), chosen=chosen,
               rule=f"numpy.random.default_rng({SEED}).choice(sorted candidates, {N_PICK}, replace=False), sorted")
    print(f"candidates {len(cands)} (ML set {len(ids)}, without log {len(no_log)}); chosen {chosen}", flush=True)

    rows = []
    for k, rid in enumerate(chosen):
        r = one(rid, am1[rid], url, tag, esp_by.loc[rid], lab, meta, mapping)
        rows.append(r)
        print(f"[{k + 1}/{N_PICK}] rxn {rid}: start {r['start_source']} "
              f"rmsd(start,DFT TS) {r['rmsd_start_dft_ts_A']:.4f} rmsd(final,DFT TS) {r['rmsd_final_dft_ts_A']:.4f} "
              f"nearest {r['start_nearest_ref']} "
              f"m4 {r['m4_ok']} {r['mapping_fail'] or 'ok'}", flush=True)
    df = pd.DataFrame(rows).sort_values("rxn_id").reset_index(drop=True)
    ev["members"] = dict(n=len(df), n_mapping_fail=int((df.mapping_fail != "").sum()),
                         n_atom_order_same=int(df.atom_order_same.sum()),
                         n_m4_ok=int((df.m4_ok == True).sum()),                                  # noqa: E712
                         n_m4_pairs_eq_label_formed=int((df.m4_pairs_eq_label_formed == True).sum()),  # noqa: E712
                         n_mapping_split_eq_label_A=int((df.mapping_split_eq_label_A == True).sum()),  # noqa: E712
                         rule=f"hard: formula of start and final = Coley TS and atom count = mapping.pkl for every "
                              f"log; Espley's reacting distances (row n) on inter-fragment heavy pairs of the final "
                              f"geometry within {M4_TOL_A} A (m4) for >= {M4_MIN_OK}/{N_PICK} logs; reported: element "
                              f"order, m4 pairs = formed_pairs_ts")
    mem = ev["members"]
    mem["passed"] = not mem["n_mapping_fail"] and mem["n_m4_ok"] >= M4_MIN_OK
    if not mem["passed"]:
        print(df[["rxn_id", "mapping_fail", "n_atoms_am1", "mapping_pkl_n_atoms", "espley_reacting_d0_A",
                  "espley_reacting_d1_A", "m4_max_dev_A", "m4_ok"]].to_string(index=False), flush=True)
        stop(dict(archive=arep, mapping_evidence=ev, selection=sel, rows=df.to_dict("records")),
             f"sampled AM1 logs inconsistent with the Coley rxn of the same number: {mem['n_mapping_fail']} with "
             f"hard failures, m4 ok {mem['n_m4_ok']}/{len(df)} (>= {M4_MIN_OK} required)")

    cols = ["rmsd_start_dft_ts_A", "rmsd_final_dft_ts_A", "rmsd_start_final_A", "rmsd_start_dft_ts_same_order_A",
            "rmsd_final_dft_ts_same_order_A", "rmsd_start_final_same_order_A", "rmsd_start_coley_optts_input_A",
            "rmsd_start_coley_optts_input_same_order_A", "rmsd_coley_optts_input_dft_ts_A",
            "rmsd_coley_optts_final_dft_ts_A", "rmsd_start_imag_mode_min_A", "rmsd_start_product_A",
            "start_nearest_rmsd_A"]
    S = {c: dist(df[c]) for c in cols}
    dd = {f"{b}_{w}": dist(df[f"d_form_{b}_{w}_A"] - df[f"d_form_{b}_dft_A"])
          for b in ("short", "long") for w in ("start", "final", "coley_optts_input")}
    V = verdict(df, S)
    rep = dict(phase="REV4 4-4: start structure of Espley's ds3 AM1 TS optimisations",
               generated_utc=utc(), verdict=V, summary=S,
               d_form_minus_dft_A=dd, archives=reps, ds3_archive=dict(tag=tag, url=url, n_am1_ts=len(am1)),
               mapping_evidence=ev, selection=sel,
               gaussian=dict(routes=df.routes.value_counts().to_dict(),
                             start_source=df.start_source.value_counts().to_dict(),
                             versions=df.gaussian.value_counts().to_dict(),
                             n_stationary_point=int(df.stationary_point.sum()),
                             n_geom_from_chk=int(df.geom_from_chk.sum())),
               inputs=dict(labels=str(LABELS), labels_sha256=sha256(LABELS), prof=str(PROF), meta=str(META),
                           espley_ds=str(ESP_DS), espley_ds_sha256=sha256(ESP_DS), espley_mapping=str(ESP_MAP),
                           espley_mapping_sha256=sha256(ESP_MAP), d1_snapshot=str(D1_SNAPSHOT), d1_commit=D1_COMMIT,
                           cache=str(BATH / tag)),
               outputs=dict(csv=str(OUT_CSV), json=str(OUT_JSON)))
    write_atomic(OUT_CSV, lambda f: df.to_csv(f, index=False))
    write_atomic(OUT_JSON, lambda f: f.write_text(json.dumps(clean(rep), indent=1)))
    print(json.dumps(clean(dict(verdict=V, summary=S)), indent=1))
    print("wrote", OUT_CSV, OUT_JSON)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "run":
        run()
    elif cmd == "archives":
        archives()
    else:
        raise SystemExit(__doc__)
