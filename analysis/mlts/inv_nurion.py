#!/usr/bin/env python3
"""inv_nurion.py — DEPLOY-V2 (MLTS) Phase I extension: label-level DFT force data and TS Hessians in the Nurion D1 run
tars (user decision 'Nurion' 2026-11-04, DECISIONS.md; spec docs/specs/DEPLOY_V2_MLTS.md sections 2, 3, 5).
Compute nodes only (sbatch / jobs/sbx.sh). The uploaded tars are opened READ ONLY and streamed (tarfile 'r|*');
nothing is extracted onto GPFS: each job's archive.tgz (+ nested ts.tar / sp.tar) is unpacked in memory and only the
ORCA files are written to a node-local ${TMPDIR:-/tmp} dir (deleted after the job). Writes: $MLT/dft/nurion/ only
(or a test root under $MLT/tmp) and results/N_report.json (collect).

  python inv_nurion.py pending                       uploaded tars (registry = D1_build/nurion manifests + .sha256) and
                                                     whether each is processed (<root>/<tar>/.done), with sizes
  python inv_nurion.py verify_upload <tar> [--root DIR] [--recheck]
        size (index/bundles.tsv) + full sha256 (<tar>.sha256 == bundles.tsv) of the uploaded copy; exit 2 = not
        matching (still uploading / corrupt). A pass is cached in <root>/<tar>/upload_ok.json (size + mtime + sha).
  python inv_nurion.py scan <tar> [--jobs a,b] [--workers N] [--root DIR] [--force]
        verify_upload first (exit 2 if it fails), then stream the tar -> <root>/<tar>/{nurion.extxyz.gz,
        hess_other.npz, hess_other.json, report.json, .done} + <root>/hess/<rxn_id>.npz (label TS Hessian per D1
        target); the stream's own sha256 must match again. --jobs = test subset (only with a test --root).
        A finished tar (.done) is skipped; .done is written last (atomic), so a crash leaves nothing marked done.
  python inv_nurion.py verify <tar> [--root DIR]     read the outputs back (extxyz tags / units / ids, npz) vs report
  python inv_nurion.py collect [--root DIR] [--force] [--workers N]  every .done tar -> $MLT/dft/nurion.extxyz.gz
        (exact duplicate geometries across tars dropped, tags united; files Phase I already read were dropped at scan
        time (sha256); rows with a Phase I geometry from another run kept + counted: finetune._dedupe unites their
        tags) + results/N_report.json. Read a .gz with read_ext() (ase seeks per frame: quadratic on gzip).
  python inv_nurion.py release <tar> --dry           what deleting the uploaded copy would do + every precondition
        (.done, sha256 ok twice, report + extxyz read back, not yet deleted). Without --dry it needs --reviewed-by
        (deletion only after an independent review; DECISIONS 'tar 삭제'); it removes only that one uploaded .tar
        and appends it to $MLT/dft/nurion/deleted_tars.tsv.
  python inv_nurion.py status [--root DIR]

Level (as inv_dft.parse_file): orca_io.label_level on the input echo (B3LYP D3BJ def2-SVP, SMD water) + inv_dft.same_grid
(RIJCOSX def2/J, default grid) + ORCA 6.1.1; parsed (coords, E, CARTESIAN GRADIENT) steps == 'GEOMETRY OPTIMIZATION
CYCLE' count and == *_trj.xyz frames where present; terminated normally. Excluded with counted reasons: the SP stage
(sp/: def2-TZVP EDA single points), autodE single_points/ and ref/*_sp (def2-TZVP), xTB (*_xtb.out, ref/scan_*), Freq
outputs (no gradient; their .hess are read instead), abnormal / mismatched files, files already in Phase I
(results/I_sources.csv sha256).

Kinds (prefix nur_ for D1-target jobs, nur_x_ for every other job: D'' rows, failed rows, extras; FT TS-near lists
(finetune.TS_NEAR_IF_CLOSE) do not contain them, so FT-1 metrics are unchanged until finetune's owner adds them):
  ts_path / ts_final   transition_states/TS_*_optts*_orca.out (OptTS incl. _conf/_reopt/_dis/neb variants); the
                       last step of a converged OptTS is ts_final (info label_ts=1 for the run whose geometry is the
                       label TS)
  tsguess_opt          transition_states/*_constrained_opt_orca.out (template TS guess, constrained Opt)
  path                 transition_states/initial_path/path_opt<k>_orca.out (constrained LooseOpt along the forming
                       bonds)
  neb                  transition_states/neb/<k>/<k>_<i>_orca.out (EnGrad)
  fwd_bwd              transition_states/*_{forwards,backwards}_opt_orca.out (autodE TS check)
  truncated            transition_states/truncated/** (truncated system, other atom set: no TS-near check)
  react_opt / prod_opt reactants_and_products/r<k>_* / p<k>_* Opt;  ref_opt  ref/**/*_opt_orca.out (reference stage)
  hess_geom            gradient steps inside a Freq output (none expected)
TS-near (info dform_label_A, target jobs, kinds ts_path/ts_final/tsguess_opt/path/neb/fwd_bwd, same element order as
the label TS): max over the two formed pairs of |d - d_label| (internal distances: frame independent); <= 0.3 A.
Non-target jobs: dform_own_A vs their own converged TS (top-level ts.xyz, ts_result.json formed_pairs_ts).

Leakage tags (mlt_common.EXT_KEYS; inv_dft.Tagger): target jobs rxn_ids/folds/dipoles of the reaction; reactant
molecules (react_opt, ref_opt) the union over every D1 target using the molecule (refs_q.mirror_min SMILES = LADDER cr
key), + the job's own reaction; products (prod_opt) the job's tags + every target with the same product; exact
duplicate geometries (molecule opts repeat in every job) are kept once with the union of the tags; non-target jobs
(D'' rows, failed rows): rxn_ids = '', folds = the fold of the job's core (core_folds()['<dipole>/<skeleton>'],
none for D'' cores; + the folds of same-cell_id targets if they differ, counted), dipoles = own dipole;
extras: S0 -> rxn 1000010 (Phase I precedent), D0 controls -> d0_id + dipoles = D0 class (gen_dft d0_class.csv).

Hessians: label TS Hessian = the final .hess of a converged label-level OptTS whose geometry equals the label TS
($LAD/dft_final/<rxn>/ts.xyz) within 1e-4 A (max atom displacement after Kabsch alignment; the .hess geometry must
also equal the run's final <base>.xyz within 1e-4 A) -> <root>/hess/<rxn_id>.npz (float32 Hessian Eh/bohr^2, symbols,
positions A, masses, source, deviations, n_imag (< -50 cm^-1, unprojected), lowest frequencies). Every other
label-level Hessian of the tree (other converged TSs, Recalc_Hess .00k (k>=2), .001 = the OptTS start Hessian (dropped
when it has the TS-guess Freq geometry), TS-guess Freq, unconverged OptTS leftovers, thermal / ref molecules, truncated)
-> <root>/<tar>/hess_other.npz (float32 upper triangles) + hess_other.json (index with the job's tags)."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inv_dft as I  # noqa: E402
import mlt_common as C  # noqa: E402
import orca_io as O  # noqa: E402

CODE_VERSION = "mlts-nur-2"
UPLOAD = Path(os.environ.get("MLT_NUR_UPLOAD", "/gpfs/home1/yeseo1ee/projects/eda-asm-prediction"))
MANI = Path(os.environ.get("MLT_NUR_MANI", str(UPLOAD / "D1_build" / "nurion")))
D1_MANIFEST = C.D1RUN / "d1_manifest.csv"
OUT = C.MLT / "dft" / "nurion"
COLLECT_EXT = C.MLT / "dft" / "nurion.extxyz.gz"
RES_N = C.RES / "N_report.json"
PHASE_I_EXT = C.MLT / "dft" / "existing.extxyz"
D0_CLASS = C.MLT / "dft" / "plan" / "d0_class.csv"
EXTRA_DATA = ("d1_extra_02_jobs_unfinished", "d1_extra_03_queue_stale_pilot")   # extras that hold job trees
N_RESULT_TARS = 23
EXTRA_RXN = {"S0": 1000010}            # Phase I (DECISIONS Phase I (1)): S0 = the D1 reaction 1000010 re-computed
NEAR_A = 0.3
HESS_GEOM_TOL_A = 1e-4                 # .hess geometry == final / label TS geometry
LABEL_ORDER_TOL_A = 1e-6               # top-level ts.xyz == label TS (scout 2: all 9,208 targets)
IMAG_TOL_CM = -50.0                    # D1 extra_imag_tol
AU2CM = 5140.48674                     # sqrt(Eh / (bohr^2 amu)) / (2 pi c) in cm^-1
DEDUPE_DEC = 5                         # finetune.DEDUPE_DEC: positions rounded to 1e-5 A for the exact-duplicate key
TS_KINDS = ("ts_path", "ts_final", "tsguess_opt", "path", "neb", "fwd_bwd")
EXT_CHUNK = 2000                       # structures per gzip member (parallel extxyz.gz writes)
OPTTS_KINDS = ("ts_path", "ts_final")
MOL_KINDS = ("react_opt", "prod_opt", "ref_opt")
SMALL_MAX = 4_000_000                  # top-level job files read into memory (json, xyz, markers)
WANT_SUFFIX = (".out", ".out.gz", "_trj.xyz", ".hess", "_orca.xyz")
RE_UNIT = re.compile(r"^(?:\./)?(?:.*/)?(jobs[A-Za-z0-9_]*)/([^/]+)/(.+)$")
RE_RP = re.compile(r"^([rp])(\d+)_([A-Za-z0-9]+)_opt_orca\d*\.out$")    # autodE may append a digit (_orca0.out)
RE_XTB = re.compile(r"_xtb\d*\.out$")
RE_FWD_BWD = re.compile(r"_(forwards|backwards)_opt_orca\d*\.out$")
RE_CONSTR = re.compile(r"_constrained_opt_orca\d*\.out$")
RE_HESS_IDX = re.compile(r"^(.*)\.(\d{3})\.hess$")
RE_TJOB = re.compile(r"^(T\d{5})(?:_.*)?$")

_W = {}                                # worker globals (initializer)


# ---------------------------------------------------------------- registry of the Nurion bundles
def registry():
    """bundle stem -> dict(kind result|extra|other, file, sha256, bytes, n_rows). Result tars = *.manifest.tsv (23);
    extras = the other *.sha256 bundles (EXTRA_DATA hold job trees; the rest: code, logs, ledger, env, ORCA)."""
    reg = {}
    sizes = {}
    bt = MANI / "index" / "bundles.tsv"
    if bt.exists():
        lines = bt.read_text().splitlines()
        hdr = lines[0].split("\t")
        for ln in lines[1:]:
            r = dict(zip(hdr, ln.split("\t")))
            sizes[r["bundle"]] = dict(bytes=int(r["bytes"]), n_rows=int(r["n_rows"]), sha256_index=r.get("sha256"))
    for m in sorted(MANI.glob("d1_*.manifest.tsv")):
        stem = m.name[: -len(".manifest.tsv")]
        reg[stem] = dict(kind="result", file=f"{stem}.tar", manifest=str(m), **sizes.get(stem, {}))
    for s in sorted(MANI.glob("*.sha256")):
        t = s.read_text().split()
        if len(t) < 2:
            continue
        fn = t[1].lstrip("*")
        stem = re.sub(r"\.tar(\.gz|\.xz)?$", "", fn)
        e = reg.setdefault(stem, dict(kind="extra" if stem in EXTRA_DATA else "other", file=fn))
        e["sha256"] = t[0]
        e["file"] = fn
    return reg


def short(stem):
    """d1_20261010_g01_002 -> g01_002 (job names, info 'tar'); extras keep their name without 'd1_'."""
    m = re.fullmatch(r"d1_\d{8}_(g\d+_\d+)", stem)
    return m.group(1) if m else stem.replace("d1_", "")


def resolve(name, reg):
    n = Path(str(name)).name
    stem = re.sub(r"\.tar(\.gz|\.xz)?$", "", n)
    if stem not in reg:
        C.stop(f"unknown tar {name!r} (not in the {MANI} registry)")
    return stem, reg[stem]


def load_manifest(stem):
    out = {}
    f = MANI / f"{stem}.manifest.tsv"
    if f.exists():
        lines = f.read_text().splitlines()
        hdr = lines[0].split("\t")
        for ln in lines[1:]:
            r = dict(zip(hdr, ln.split("\t")))
            out[r["job_id"]] = r
    return out


def out_dirs(root):
    root = Path(root) if root else OUT
    C.assert_write_ok(root / "x")
    return root, root / "hess"


def verify_upload(stem, ent, root=None, recheck=False):
    """the uploaded copy vs D1_build/nurion: size == index/bundles.tsv bytes and full sha256 == <tar>.sha256 (== the
    bundles.tsv sha256). Returns a record with ok True/False + reason; a pass is cached in <root>/<tar>/upload_ok.json
    keyed by (size, mtime_ns, expected sha256), so a re-scan of an unchanged file does not hash it again."""
    root, _ = out_dirs(root)
    p = UPLOAD / ent["file"]
    rec = dict(tar=stem, file=str(p), expected_bytes=ent.get("bytes"), sha256_expected=ent.get("sha256"),
               sha256_index=ent.get("sha256_index"))
    if not p.exists():
        return dict(rec, ok=False, reason="not uploaded")
    st = p.stat()
    rec.update(bytes=st.st_size, mtime_ns=st.st_mtime_ns)
    if not ent.get("sha256"):
        return dict(rec, ok=False, reason="no <tar>.sha256 in the registry")
    if ent.get("sha256_index") and ent["sha256_index"] != ent["sha256"]:
        return dict(rec, ok=False, reason="<tar>.sha256 != index/bundles.tsv sha256")
    if ent.get("bytes") and st.st_size != ent["bytes"]:
        return dict(rec, ok=False, reason=f"size {st.st_size} != expected {ent['bytes']} (still uploading?)")
    cache = root / stem / "upload_ok.json"
    if cache.exists() and not recheck:
        c = C.read_json(cache)
        if c.get("ok") and c.get("bytes") == st.st_size and c.get("mtime_ns") == st.st_mtime_ns \
                and c.get("sha256") == ent["sha256"]:
            return dict(c, cached=True)
    t0 = time.time()
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    sha = h.hexdigest()
    st2 = p.stat()
    rec.update(sha256=sha, t_hash_s=round(time.time() - t0, 1), time=C.now(), host=os.uname().nodename,
               slurm_job=os.environ.get("SLURM_JOB_ID"))
    if (st2.st_size, st2.st_mtime_ns) != (st.st_size, st.st_mtime_ns):
        return dict(rec, ok=False, reason="file changed while hashing (still uploading?)")
    if sha != ent["sha256"]:
        return dict(rec, ok=False, reason="sha256 mismatch (corrupt or incomplete upload)")
    rec.update(ok=True, reason="")
    C.write_json(cache, rec)
    return dict(rec, cached=False)


def cmd_verify_upload(args):
    reg = registry()
    stem, ent = resolve(args.tar, reg)
    r = verify_upload(stem, ent, args.root, recheck=args.recheck)
    print(json.dumps(r, indent=1, default=C._json_default), flush=True)
    return 0 if r["ok"] else 2


class HashReader(io.RawIOBase):
    """file wrapper: sha256 + byte count of everything read (the stream tar reads every byte once)."""

    def __init__(self, f):
        self.f, self.h, self.n = f, hashlib.sha256(), 0

    def readable(self):
        return True

    def read(self, n=-1):
        b = self.f.read(n)
        self.h.update(b)
        self.n += len(b)
        return b

    def readinto(self, buf):
        b = self.read(len(buf))
        buf[: len(b)] = b
        return len(b)

    def drain(self):
        while self.read(1 << 22):
            pass


# ---------------------------------------------------------------- geometry helpers
def dists(pos, pairs):
    pos = np.asarray(pos, float)
    return np.array([np.linalg.norm(pos[a] - pos[b]) for a, b in pairs])


def kabsch(P, Q):
    """(rmsd, max atom displacement) of P onto Q after centring + optimal rotation (A)."""
    P = np.asarray(P, float) - np.mean(P, 0)
    Q = np.asarray(Q, float) - np.mean(Q, 0)
    U, _, Vt = np.linalg.svd(P.T @ Q)
    D = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(U @ Vt)) or 1.0)])
    X = P @ U @ D @ Vt - Q
    n = np.linalg.norm(X, axis=1)
    return float(np.sqrt((n ** 2).mean())), float(n.max())


def raw_dev(P, Q):
    return float(np.linalg.norm(np.asarray(P, float) - np.asarray(Q, float), axis=1).max())


def freqs_cm(H, masses, pos=None):
    """harmonic frequencies (cm^-1, imaginary < 0) of a Cartesian Hessian (Eh/bohr^2, masses amu). With positions the
    translations and rotations are projected out of the mass-weighted Hessian first (their 5-6 zero modes dropped)."""
    ms = np.asarray(masses, float)
    m = np.repeat(ms, 3)
    Hm = 0.5 * (H + H.T) / np.sqrt(np.outer(m, m))
    if pos is None:
        w = np.linalg.eigvalsh(Hm)
        return np.sign(w) * np.sqrt(np.abs(w)) * AU2CM
    X = np.asarray(pos, float)
    X = X - (ms[:, None] * X).sum(0) / ms.sum()
    sq = np.sqrt(ms)[:, None]
    vec = []
    for k in range(3):
        t = np.zeros_like(X)
        t[:, k] = 1.0
        vec.append((t * sq).ravel())
        e = np.zeros(3)
        e[k] = 1.0
        vec.append((np.cross(e, X) * sq).ravel())
    U, s, _ = np.linalg.svd(np.array(vec).T, full_matrices=False)
    D = U[:, s > 1e-6 * s.max()]
    P = np.eye(len(m)) - D @ D.T
    w = np.linalg.eigvalsh(P @ Hm @ P)
    w = np.delete(w, np.argsort(np.abs(w))[:D.shape[1]])
    return np.sign(w) * np.sqrt(np.abs(w)) * AU2CM


def read_xyz_text(txt):
    ln = txt.splitlines()
    n = int(ln[0].split()[0])
    sym, pos = [], []
    for s in ln[2:2 + n]:
        q = s.split()
        sym.append(q[0])
        pos.append([float(q[1]), float(q[2]), float(q[3])])
    return sym, np.array(pos)


def geom_key(symbols, pos):
    h = hashlib.sha1((" ".join(symbols) + "|").encode())
    h.update((np.round(np.asarray(pos, float), DEDUPE_DEC) + 0.0).tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------- per-job worker
def _winit(tmproot, i_shas):
    _W["tmp"] = tmproot
    _W["i_shas"] = i_shas


def _safe(name):
    return not (name.startswith("/") or ".." in name.split("/"))


def _wanted(name):
    b = name.rsplit("/", 1)[-1]
    if RE_XTB.search(b) or "/conformers/" in f"/{name}":
        return False
    return b.endswith(WANT_SUFFIX)


def classify(nrel):
    """directory class of a normalised path (relative to the job folder; ts/<autodE name>/ stripped to ts/)."""
    p = nrel.split("/")
    top = p[0]
    if top == "sp":
        return "sp_stage"
    if top == "ref":
        return "ref"
    if top == "ts" and len(p) > 2:
        sub = p[1]
        if sub == "transition_states":
            mid = p[2:-1]
            if "truncated" in mid:
                return "trunc"
            if "initial_path" in mid:
                return "path"
            if "neb" in mid:
                return "neb"
            return "tsdir"
        return {"single_points": "ts_sp", "thermal": "thermal", "reactants_and_products": "rp"}.get(sub, "ts_other")
    return "other"


def base_kind(cat, name, jt):
    if cat in ("trunc", "path", "neb"):
        return "truncated" if cat == "trunc" else cat
    if cat == "tsdir":
        if jt == "optts":
            return "ts"
        if RE_FWD_BWD.search(name):
            return "fwd_bwd"
        if RE_CONSTR.search(name) or jt == "const_opt":
            return "tsguess_opt"
        return f"other_{jt}"
    if cat == "rp":
        m = RE_RP.match(name)
        return ("react_opt" if m.group(1) == "r" else "prod_opt") if (m and jt == "opt") else f"other_{jt}"
    if cat == "ref":
        return "ref_opt" if jt == "opt" else f"other_{jt}"
    if jt == "freq":
        return "hess_geom"
    return f"other_{jt}"


def _norm_map(rels):
    """'ts/<root>/x' -> 'ts/x' when the ts/ tree has exactly one autodE reaction folder."""
    roots = Counter(r.split("/")[1] for r in rels if r.startswith("ts/") and r.count("/") >= 2)
    root = next(iter(roots)) if len(roots) == 1 else None
    out = {}
    for r in rels:
        if root and r.startswith(f"ts/{root}/"):
            out[r] = "ts/" + r[len(f"ts/{root}/"):]
        else:
            out[r] = r
    return out, sorted(roots)


def scan_unit(pack):
    """one job folder (archive bytes + top-level small files + raw tree files) -> parsed result (no tags)."""
    t0 = time.time()
    unit = pack["unit"]
    res = dict(unit=unit, job=pack["job"], counts=Counter(), errors=[], files=[], kept=[], hess=[], dup_paths=[],
               n_unpacked=0, bytes_unpacked=0)
    d = Path(tempfile.mkdtemp(prefix=f"{re.sub(r'[^A-Za-z0-9_.-]', '_', unit)}_", dir=_W["tmp"]))
    files = {}

    def put(rel, data, src):
        if not _safe(rel):
            res["counts"]["unsafe_path"] += 1
            return
        if rel in files:
            res["dup_paths"].append([rel, files[rel]["src"], src])
            if not src.endswith("ts.tar"):                       # a nested stage tar wins over a plain duplicate
                return
        p = d / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        files[rel] = dict(local=str(p), src=src, bytes=len(data))

    def walk(tf, src):
        for m in tf:
            name = m.name[2:] if m.name.startswith("./") else m.name
            if m.isdir():
                continue
            if m.issym() or m.islnk():
                res["counts"]["link_skipped"] += 1
                continue
            if not m.isfile():
                res["counts"]["other_member_type"] += 1
                continue
            res["n_unpacked"] += 1
            res["bytes_unpacked"] += m.size
            if name.endswith(".tar"):
                try:
                    with tarfile.open(fileobj=io.BytesIO(tf.extractfile(m).read()), mode="r:") as t2:
                        walk(t2, f"{src}::{name}")
                    res["counts"][f"nested:{name.rsplit('/', 1)[-1]}"] += 1
                except (tarfile.TarError, OSError, EOFError) as e:
                    res["errors"].append(f"nested {name}: {type(e).__name__}: {e}"[:200])
                continue
            if not _wanted(name):
                res["counts"]["xtb_out" if RE_XTB.search(name) else "not_needed"] += 1
                continue
            put(name, tf.extractfile(m).read(), src)

    try:
        for name, blob in pack["archives"]:
            res["arc_bytes"] = len(blob)
            res["md5"] = hashlib.md5(blob).hexdigest()
            try:
                with tarfile.open(fileobj=io.BytesIO(blob), mode="r:*") as tf:
                    walk(tf, name)
            except (tarfile.TarError, OSError, EOFError) as e:
                res["errors"].append(f"archive {name}: {type(e).__name__}: {e}"[:200])
        for rel, blob in pack["raw"]:
            if rel.endswith(".tar"):
                try:
                    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:") as t2:
                        walk(t2, rel)
                    res["counts"][f"nested:{rel.rsplit('/', 1)[-1]}"] += 1
                except (tarfile.TarError, OSError, EOFError) as e:
                    res["errors"].append(f"nested {rel}: {type(e).__name__}: {e}"[:200])
            else:
                res["n_unpacked"] += 1
                put(rel, blob, "raw")
        res["t_unpack_s"] = round(time.time() - t0, 3)
        nmap, roots = _norm_map(list(files))
        res["ts_roots"] = roots
        _parse_tree(pack, res, files, nmap)
    except Exception as e:                                       # noqa: BLE001
        res["errors"].append(f"worker: {type(e).__name__}: {e}"[:300])
    finally:
        shutil.rmtree(d, ignore_errors=True)
    res["t_total_s"] = round(time.time() - t0, 3)
    return res


def _parse_tree(pack, res, files, nmap):
    lab, own = pack.get("label"), pack.get("own")
    ref = lab or own                                            # TS of reference for dform / Hessian checks
    i_shas = _W["i_shas"]
    outs = {}
    t1 = time.time()
    for rel in sorted(files, key=lambda r: nmap[r]):
        if not rel.endswith((".out", ".out.gz")):
            continue
        nrel = nmap[rel]
        name = nrel.rsplit("/", 1)[-1]
        cat = classify(nrel)
        rec, steps = I.parse_file(dict(path=files[rel]["local"], source="nurion"))
        jt = rec.get("job_type")
        f = dict(path=nrel, cat=cat, jt=jt, level_ok=bool(rec.get("level_ok")), sha16=(rec.get("sha256") or "")[:16],
                 ver=rec.get("orca_version"), term=rec.get("terminated_normally"), conv=bool(rec.get("converged")),
                 n_cycles=rec.get("n_opt_cycles"), n_steps=rec.get("n_steps"), trj=rec.get("trj_frames"),
                 trj_ok=rec.get("trj_ok"), bytes=files[rel]["bytes"], src=files[rel]["src"])
        if str(rec.get("reason") or "").startswith("parse_error"):
            reason = rec["reason"]
        elif not rec.get("level_ok") and not rec.get("bang"):
            reason = "not_orca_or_no_echo"
        elif cat == "sp_stage":
            reason = "sp_stage" + ("" if not rec.get("level_ok") else "_label_level")
        elif not rec.get("kept"):
            reason = rec.get("reason") or "?"
        elif not str(rec.get("orca_version") or "").startswith("6.1.1"):
            reason = f"orca_version {rec.get('orca_version')}"
        elif rec.get("sha256") in i_shas:
            reason = "dup_phase_i"
        else:
            reason = ""
        if reason.startswith("not_label_level"):
            f["reason"] = "not_label_level: " + (rec.get("level_reason") or "")[:60]
        else:
            f["reason"] = reason
        outs[nrel] = (f, steps if not reason else [])
        if reason:
            f["kept"] = False
            res["files"].append(f)
            continue
        f["kept"] = True
        bk = base_kind(cat, name, jt)
        n = len(steps)
        kinds = [bk] * n
        if bk == "ts":
            kinds = ["ts_path"] * n
            if f["conv"]:
                kinds[-1] = "ts_final"
        sym = list(steps[0]["symbols"])
        pos = np.array([s["pos_A"] for s in steps])
        grad = np.array([s["grad_eh_bohr"] for s in steps])
        en = np.array([s["energy_eh"] for s in steps])
        cyc = [int(s["cycle"]) for s in steps]
        dform = None
        if bk in ("ts", "tsguess_opt", "path", "neb", "fwd_bwd") and ref is not None:
            if sym == ref["symbols"]:
                dl = dists(ref["pos"], ref["pairs"])
                dform = [float(np.abs(dists(p, ref["pairs"]) - dl).max()) for p in pos]
            else:
                res["counts"]["ts_kind_order_mismatch"] += 1
        if bk == "ts" and f["conv"] and ref is not None and sym == ref["symbols"]:
            fin = _final_geom(files, rel, pos[-1])
            f["final_vs_ref_kabsch_A"] = kabsch(fin, ref["pos"])[1]
        res["kept"].append(dict(f=f, base=bk, kinds=kinds, symbols=sym, pos=pos, grad=grad, energy=en, cycles=cyc,
                                dform=dform))
        res["files"].append(f)
    res["t_parse_s"] = round(time.time() - t1, 3)
    _hessians(pack, res, files, nmap, outs, ref)


def _final_geom(files, rel_out, fallback):
    """ORCA's final geometry of an optimisation: <base>.xyz (same frame as the .out), else the last gradient step."""
    base = re.sub(r"\.out(\.gz)?$", "", rel_out)
    x = files.get(base + ".xyz")
    if x:
        try:
            return read_xyz_text(Path(x["local"]).read_text())[1]
        except Exception:                                        # noqa: BLE001
            pass
    return fallback


def _hessians(pack, res, files, nmap, outs, ref):
    lab = pack.get("label")
    seen, guess_geo = {}, {}
    hrels = [r for r in files if r.endswith(".hess")]
    # non-indexed (final / guess / molecule) first, then the OptTS .00k copies, so the dedupe keeps the named file
    for rel in sorted(hrels, key=lambda r: (RE_HESS_IDX.match(nmap[r]) is not None, nmap[r])):
        nrel = nmap[rel]
        m = RE_HESS_IDX.match(nrel)
        idx = int(m.group(2)) if m else None
        stem = m.group(1) if m else nrel[: -len(".hess")]
        orec = outs.get(stem + ".out") or outs.get(stem + ".out.gz")
        h = dict(path=nrel, idx=idx, out=stem + ".out", cat=classify(nrel))
        if orec is None:
            h.update(kind="excluded", reason="no_matching_out")
            res["hess"].append(h)
            continue
        of = orec[0]
        h.update(out_jt=of["jt"], out_level_ok=of["level_ok"], out_conv=of["conv"], out_term=of["term"])
        if not of["level_ok"]:
            h.update(kind="excluded", reason="out_not_label_level")
            res["hess"].append(h)
            continue
        try:
            hh = O.read_hess(files[rel]["local"])
        except Exception as e:                                   # noqa: BLE001
            h.update(kind="excluded", reason=f"read_error {type(e).__name__}: {e}"[:120])
            res["hess"].append(h)
            continue
        H = hh["hessian"]
        sym = list(hh["symbols"])
        pos = hh["pos_bohr"] * C.BOHR2A
        h.update(n_atoms=len(sym), finite=bool(np.isfinite(H).all()), asym=float(np.abs(H - H.T).max()))
        if not h["finite"]:
            h.update(kind="excluded", reason="non_finite")
            res["hess"].append(h)
            continue
        try:
            fr = freqs_cm(H, hh["masses"], hh["pos_bohr"])
            h.update(n_imag=int((fr < IMAG_TOL_CM).sum()), freq_low=[round(float(x), 2) for x in fr[:3]])
        except Exception as e:                                   # noqa: BLE001
            h.update(n_imag=None, freq_err=f"{type(e).__name__}"[:40])
        gk = geom_key(sym, pos)
        key = gk + hashlib.sha1(np.round(H, 8).tobytes()).hexdigest()[:16]
        if key in seen:
            h.update(kind="excluded", reason=f"duplicate_of {seen[key]}")
            res["hess"].append(h)
            continue
        if idx == 1 and of["jt"] == "optts" and gk in guess_geo:  # .001 = the OptTS start Hessian = the guess Freq
            g = guess_geo[gk]
            h.update(kind="excluded", reason=f"initial_eq_guess {g[0]}", max_dH=float(np.abs(H - g[1]).max()))
            res["hess"].append(h)
            continue
        seen[key] = nrel
        cat = h["cat"]
        trunc = cat == "trunc"
        if of["jt"] == "freq" and cat not in ("thermal", "ref"):
            guess_geo[gk] = (nrel, H)
        if of["jt"] == "optts":
            if idx is not None:
                k = "ts_hess_initial" if idx == 1 else "ts_hess_recalc"
            else:
                rel_out = next((r for r in files if nmap[r] == stem + ".out"), None)
                fin = _final_geom(files, rel_out, None) if rel_out else None
                dfin = kabsch(pos, fin)[1] if fin is not None and len(fin) == len(pos) else None
                h["dev_final_A"] = dfin
                if not of["conv"]:
                    k = "ts_hess_unconv"                       # leftover of the last Recalc of an unconverged OptTS
                elif dfin is not None and dfin <= HESS_GEOM_TOL_A:
                    k = "ts_hess"
                else:
                    k = "ts_hess_geom_mismatch"
        elif of["jt"] == "freq":
            k = "mol_hess" if cat in ("thermal", "ref") else "ts_hess_guess"
        else:
            k = f"hess_{of['jt']}"
        if trunc:
            k = "trunc_" + k
        h["kind"] = k
        if ref is not None and sym == ref["symbols"] and not trunc:
            h["dev_ref_raw_A"] = raw_dev(pos, ref["pos"])
            h["dev_ref_kabsch_A"] = kabsch(pos, ref["pos"])[1]
            dl = dists(ref["pos"], ref["pairs"])
            h["dform_ref_A"] = float(np.abs(dists(pos, ref["pairs"]) - dl).max())
        h["_H"] = H.astype(np.float32)
        h["_pos"] = pos
        h["_sym"] = sym
        h["_masses"] = np.asarray(hh["masses"], float)
        res["hess"].append(h)
    # the label TS Hessian (targets) / the own TS Hessian (non-targets): final .hess of a converged OptTS == ref TS
    cand = [h for h in res["hess"] if h.get("kind") == "ts_hess" and h.get("dev_ref_kabsch_A") is not None]
    best = min(cand, key=lambda h: h["dev_ref_kabsch_A"]) if cand else None
    res["ref_hess"] = None
    if best is not None:
        res["ref_hess_best_dev_A"] = best["dev_ref_kabsch_A"]
        if best["dev_ref_kabsch_A"] <= HESS_GEOM_TOL_A:
            best["kind"] = "ts_hess_label" if lab is not None else "ts_hess_own"
            res["ref_hess"] = best["path"]
            run = best["out"]
            for kf in res["kept"]:
                if kf["f"]["path"] == run:
                    kf["f"]["label_run"] = True


# ---------------------------------------------------------------- main-process context: targets, tags, molecules
class Ctx:
    def __init__(self):
        import pandas as pd
        self.tg = I.Tagger()
        self.T = self.tg.T                                     # targets() indexed by rxn_id
        self.job_rxn = {j: int(r) for r, j in zip(self.T.index, self.T.job_id)}
        self.cf = C.core_folds()
        M = pd.read_csv(D1_MANIFEST, dtype=str).fillna("")
        self.d1m = {r["job_id"]: r for r in M.to_dict("records")}
        self.cell_rxns = defaultdict(set)
        for j, r in self.d1m.items():
            if j in self.job_rxn:
                self.cell_rxns[r["cell_id"]].add(self.job_rxn[j])
        self._prod = None
        self._form = {}
        self._mm = {}
        self.d0cls = {}
        if D0_CLASS.exists():
            D = pd.read_csv(D0_CLASS, usecols=["d0_id", "d0_class"])
            self.d0cls = dict(zip(D.d0_id.astype(int), D.d0_class.astype(str)))
        self.checks = Counter()

    def mm(self, smi):
        if smi not in self._mm:
            self._mm[smi] = self.tg.mm(smi)
        return self._mm[smi]

    def formula(self, smi):
        if smi not in self._form:
            try:
                from rdkit import Chem
                m = Chem.MolFromSmiles(smi)
                self._form[smi] = Counter(a.GetSymbol() for a in Chem.AddHs(m).GetAtoms()) if m else None
            except Exception:                                    # noqa: BLE001
                self._form[smi] = None
        return self._form[smi]

    def prod_rxns(self, smi):
        if self._prod is None:
            self._prod = defaultdict(set)
            for rid, rs in self.tg.rxn_smiles.items():
                if ">>" in rs:
                    self._prod[self.mm(rs.split(">>")[1])].add(int(rid))
        return self._prod.get(self.mm(smi), set())

    def tags(self, rxns, extra_folds=(), extra_dipoles=(), rule=""):
        rx = sorted({int(r) for r in rxns if int(r) in self.T.index})
        folds = {str(int(self.T.at[r, "fold_T1"])) for r in rx} | {str(x) for x in extra_folds}
        dips = {str(self.T.at[r, "dipole"]) for r in rx} | {str(x) for x in extra_dipoles if x}
        return dict(rxn_ids=C.tag_list(rx), folds=C.tag_list(folds), dipoles=C.tag_list(dips), tag_rule=rule)

    def job_info(self, unit, meta):
        """tags + identity of one job folder."""
        jb = unit.rsplit("/", 1)[-1]
        m = RE_TJOB.match(jb)
        jid = m.group(1) if m else jb
        row = self.d1m.get(jid, {})
        info = dict(job=jid, unit=unit, target=jid in self.job_rxn, rxn=self.job_rxn.get(jid), d1m=row,
                    dipole=row.get("dipole", ""), group=row.get("ts_id") or (meta.get("ts_result") or {}).get("ts_id")
                    or jid)
        if info["target"]:
            info["tags"] = self.tags([info["rxn"]], rule="target_rxn")
        elif jid in EXTRA_RXN:
            info["tags"] = self.tags([EXTRA_RXN[jid]], rule="extra_phase_i_map")
        elif row:
            # caller rule: rxn_ids '' (not a D1 target), dipoles = own dipole, folds = the core's T1 fold (blocks
            # core-level leakage into FT_k); same-cell targets' folds are added only if they differ (counted)
            core = f"{row.get('dipole')}/{row.get('skeleton')}"
            ef = {str(int(self.cf[core]))} if core in self.cf else set()
            cf = {str(int(self.T.at[r, "fold_T1"])) for r in self.cell_rxns.get(row.get("cell_id"), set())}
            if cf - ef:
                self.checks["nontarget_cell_folds_outside_core"] += 1
            self.checks["nontarget_core_fold" if ef else "nontarget_no_core_fold"] += 1
            info["tags"] = self.tags([], extra_folds=ef | cf, extra_dipoles=[row.get("dipole")], rule="nontarget_core")
        else:
            ts_id = str((meta.get("ts_result") or {}).get("ts_id") or "")
            mm_ = re.fullmatch(r"D0-(\d+)", ts_id)
            if mm_:
                d0 = int(mm_.group(1))
                info["d0_id"] = d0
                info["tags"] = self.tags([], extra_dipoles=[self.d0cls.get(d0, "")], rule="extra_d0")
            else:
                info["tags"] = self.tags([], rule="none_unmapped")
        return info

    def mol_smiles(self, ji, meta, nrel, symbols):
        """SMILES of the molecule of a react_opt / prod_opt / ref_opt file (rxn_smiles component, formula-checked)."""
        tr = meta.get("ts_result") or {}
        rs = tr.get("rxn_smiles") or ji["d1m"].get("rxn_smiles") or ""
        if ">>" not in rs:
            return None, "no_rxn_smiles"
        lhs, rhs = rs.split(">>")
        comps = lhs.split(".") if "ref/" not in nrel else []
        form = Counter(symbols)
        name = nrel.rsplit("/", 1)[-1]
        if nrel.startswith("ref/"):
            cand = [ji["d1m"].get("dipole_smiles")] if ji["d1m"].get("dipole_smiles") else []
            k = tr.get("reactant_index_dipole")
            if not cand and k is not None:
                cand = [lhs.split(".")[int(k)]]
            hit = [c for c in cand if self.formula(c) == form]
            return (hit[0], "ref_dipole") if hit else (None, "ref_formula_mismatch")
        m = RE_RP.match(name)
        if not m:
            return None, "name"
        pool = comps if m.group(1) == "r" else rhs.split(".")
        k = int(m.group(2))
        if k < len(pool) and self.formula(pool[k]) == form:
            return pool[k], "index"
        hit = [c for c in pool if self.formula(c) == form]
        if len(hit) == 1:
            return hit[0], "formula"
        return None, "ambiguous" if hit else "formula_mismatch"

    def mol_tags(self, ji, meta, base, nrel, symbols):
        smi, how = self.mol_smiles(ji, meta, nrel, symbols)
        self.checks[f"mol_map:{how}"] += 1
        if smi is None:
            t = dict(ji["tags"])
            t["tag_rule"] = "mol_fallback_job"
            return t, None
        key = self.mm(smi)
        if base == "prod_opt":
            jt = ji["tags"]
            rx = {int(x) for x in jt["rxn_ids"].split(",") if x} | self.prod_rxns(smi)
            t = self.tags(rx, extra_folds=[x for x in jt["folds"].split(",") if x],
                          extra_dipoles=[x for x in jt["dipoles"].split(",") if x], rule="prod_union")
            return t, key
        rx = set(self.tg.smi_rxns.get(key, set()))
        if ji["target"]:
            self.checks["mol_own_rxn_in_union" if ji["rxn"] in rx else "mol_own_rxn_not_in_union"] += 1
            rx.add(ji["rxn"])
        elif ji.get("tags", {}).get("tag_rule") == "extra_phase_i_map":
            rx.add(EXTRA_RXN[ji["job"]])
        is_dip = nrel.startswith("ref/") or (ji["d1m"].get("dipole_smiles") and self.mm(ji["d1m"]["dipole_smiles"]) == key)
        xd = [ji["dipole"]] if (is_dip and not ji["target"] and ji["dipole"]) else []
        if ji.get("d0_id") is not None:
            xd = [x for x in ji["tags"]["dipoles"].split(",") if x]
        return self.tags(rx, extra_dipoles=xd, rule="mol_union"), key


# ---------------------------------------------------------------- scan
def _small_meta(small):
    meta = dict(markers=sorted(n for n in small if n.startswith(".") and not n.startswith((".done_", ".packed",
                                                                                          ".progress"))),
                done=sorted(n for n in small if n.startswith(".done_")), top_files=len(small))
    for n in ("ts_result.json", "label.json"):
        if n in small:
            try:
                meta[n.split(".")[0]] = json.loads(small[n])
            except Exception:                                    # noqa: BLE001
                meta[n.split(".")[0] + "_error"] = True
    if "ts.xyz" in small:
        try:
            meta["ts_xyz"] = read_xyz_text(small["ts.xyz"].decode())
        except Exception:                                        # noqa: BLE001
            meta["ts_xyz_error"] = True
    return meta


def _ref_for(ctx, ji, meta):
    """reference TS for dform / Hessian: label TS (targets), own top-level ts.xyz + formed pairs (others)."""
    if ji["target"]:
        sym, pos = C.read_xyz(C.label_ts(ji["rxn"]))
        lab = dict(symbols=list(sym), pos=np.asarray(pos, float),
                   pairs=C.formed_pairs(ctx.T.at[ji["rxn"], "formed_pairs_ts"]))
        if meta.get("ts_xyz"):
            s2, p2 = meta["ts_xyz"]
            ok = s2 == lab["symbols"] and raw_dev(p2, lab["pos"]) <= LABEL_ORDER_TOL_A
            ctx.checks["ts_xyz_eq_label" if ok else "ts_xyz_ne_label"] += 1
        return lab, None
    tr = meta.get("ts_result") or {}
    if meta.get("ts_xyz") and tr.get("formed_pairs_ts"):
        s2, p2 = meta["ts_xyz"]
        try:
            pairs = [tuple(int(x) for x in p) for p in tr["formed_pairs_ts"]]
            return None, dict(symbols=list(s2), pos=np.asarray(p2, float), pairs=pairs)
        except Exception:                                        # noqa: BLE001
            return None, None
    return None, None


def cmd_scan(args):
    t0 = time.time()
    reg = registry()
    stem, ent = resolve(args.tar, reg)
    tar_path = UPLOAD / ent["file"]
    root, hdir = out_dirs(args.root)
    production = root.resolve() == OUT.resolve()
    jobs_filter = {x.strip() for x in args.jobs.split(",") if x.strip()} if args.jobs else None
    if jobs_filter and production:
        C.not_ready("--jobs (a test subset) needs a test --root (e.g. $MLT/tmp/nurtest/scan); production scans are "
                    "whole tars")
    od = root / stem
    done = od / ".done"
    if done.exists() and not args.force:
        print(f"[nur] {stem}: done ({done}) — skipped", flush=True)
        return 0
    t_up = time.time()
    up = verify_upload(stem, ent, args.root)
    print(f"[nur] {stem}: verify_upload ok={up['ok']} cached={up.get('cached')} ({time.time() - t_up:.1f} s) "
          f"{up.get('reason') or ''}", flush=True)
    if not up["ok"]:
        C.not_ready(f"verify_upload {stem}: {up['reason']}")
    size = up["bytes"]
    if done.exists():
        C.assert_write_ok(done)
        done.unlink()                                          # --force: nothing stays marked done while rewriting
    ctx = Ctx()
    mani = load_manifest(stem)
    i_shas = set()
    if I.RES_SRC.exists():
        import pandas as pd
        i_shas = set(pd.read_csv(I.RES_SRC, usecols=["sha256"]).sha256.dropna().astype(str))
    tmproot = tempfile.mkdtemp(prefix="nurscan_", dir=os.environ.get("TMPDIR", "/tmp"))
    nw = max(1, int(args.workers))
    pool = ProcessPoolExecutor(max_workers=nw, initializer=_winit, initargs=(tmproot, i_shas))
    results, inflight, units_seen, outer = [], set(), [], Counter()
    metas, infos = {}, {}
    cur = None
    stopped = None
    hr = None
    t_read = 0.0

    def flush(u):
        if u is None:
            return
        meta = _small_meta(u["small"])
        ji = ctx.job_info(u["unit"], meta)
        metas[u["unit"]], infos[u["unit"]] = meta, ji
        lab, own = _ref_for(ctx, ji, meta)
        mrow = mani.get(ji["job"]) or {}
        pack = dict(unit=u["unit"], job=ji["job"], archives=u["archives"], raw=u["raw"], label=lab, own=own)
        meta["manifest"] = {k: mrow.get(k) for k in ("class", "reason", "n_files", "bytes", "archive_md5")}
        meta["stale_archives"] = u["stale"]
        meta["top_bytes"] = u["top_bytes"]
        meta["n_top"] = u["n_top"]
        while len(inflight) >= 2 * nw:
            dn, rest = wait(inflight, return_when=FIRST_COMPLETED)
            inflight.clear()
            inflight.update(rest)
            results.extend(f.result() for f in dn)
        inflight.add(pool.submit(scan_unit, pack))

    try:
        with open(tar_path, "rb") as fh:
            hr = HashReader(fh)
            with tarfile.open(fileobj=hr, mode="r|*") as tf:
                for m in tf:
                    name = m.name[2:] if m.name.startswith("./") else m.name
                    mm = RE_UNIT.match(name)
                    if not mm:
                        outer[("dir:" if m.isdir() else "outside_jobs:") + name.split("/")[0]] += 1
                        continue
                    jd, jb, rest = mm.groups()
                    unit = jb if jd == "jobs" else f"{jd}/{jb}"
                    if cur is None or cur["unit"] != unit:
                        if cur is not None and (jobs_filter is None or cur["unit"].rsplit("/", 1)[-1] in jobs_filter):
                            flush(cur)
                        if jobs_filter is not None and units_seen and \
                                {x for x in jobs_filter} <= {u.rsplit("/", 1)[-1] for u in units_seen}:
                            stopped = f"all {len(jobs_filter)} requested jobs read"
                            cur = None
                            break
                        if unit in units_seen:
                            outer["noncontiguous_unit"] += 1
                        units_seen.append(unit)
                        cur = dict(unit=unit, small={}, archives=[], raw=[], stale=[], top_bytes=0, n_top=0)
                    if not m.isfile():
                        outer["non_file_member"] += 1
                        continue
                    if jobs_filter is not None and jb not in jobs_filter:
                        continue
                    depth1 = "/" not in rest
                    tr = time.time()
                    if depth1:
                        cur["top_bytes"] += m.size
                        cur["n_top"] += 1
                    if depth1 and re.fullmatch(r"archive.*\.tgz", rest):
                        if rest == "archive.tgz":
                            cur["archives"].append((rest, tf.extractfile(m).read()))
                        else:
                            cur["stale"].append([rest, m.size])
                    elif depth1 and rest.endswith(".tar"):
                        cur["raw"].append((rest, tf.extractfile(m).read()))
                    elif depth1 and m.size <= SMALL_MAX and not _wanted(rest):
                        cur["small"][rest] = tf.extractfile(m).read()
                    elif _wanted(rest) or rest.endswith(".tar"):
                        cur["raw"].append((rest, tf.extractfile(m).read()))
                    else:
                        outer["unit_member_not_needed"] += 1
                    t_read += time.time() - tr
                if cur is not None and (jobs_filter is None or cur["unit"].rsplit("/", 1)[-1] in jobs_filter):
                    flush(cur)
            if stopped is None:
                hr.drain()
        for f in inflight:
            results.append(f.result())
    except (tarfile.TarError, OSError, EOFError) as e:
        pool.shutdown(cancel_futures=True)
        shutil.rmtree(tmproot, ignore_errors=True)
        C.stop(f"{tar_path}: tar read error {type(e).__name__}: {e} (after {hr.n if hr else 0} bytes) — nothing marked "
               "done")
    finally:
        pool.shutdown()
        shutil.rmtree(tmproot, ignore_errors=True)
    t_stream = time.time() - t0
    sha = hr.h.hexdigest() if (hr is not None and stopped is None) else None
    sha_ok = (sha == ent.get("sha256")) if (sha and ent.get("sha256")) else None
    print(f"[nur] {stem}: {len(results)} jobs streamed in {t_stream:.1f} s (read {t_read:.1f} s), sha256 "
          f"{'ok' if sha_ok else sha_ok}", flush=True)
    if sha is not None and ent.get("sha256") and not sha_ok:
        C.not_ready(f"{tar_path}: streamed sha256 {sha} != {ent['sha256']} (file changed after verify_upload?) — "
                    "nothing marked done")
    rep, atoms, hess_label, hess_other = assemble(ctx, stem, results, metas, infos)
    rep["meta"] = dict(code=CODE_VERSION, time=C.now(), host=os.uname().nodename, slurm_job=os.environ.get("SLURM_JOB_ID"),
                       tar=str(tar_path), tar_bytes=size, sha256=sha, sha256_expected=ent.get("sha256"), sha256_ok=sha_ok,
                       bytes_read=hr.n if hr else None, workers=nw, jobs_filter=sorted(jobs_filter) if jobs_filter else None,
                       stopped=stopped, t_stream_s=round(t_stream, 1), t_read_s=round(t_read, 1),
                       outer_counts=dict(outer), n_manifest=len(mani), root=str(root),
                       upload={k: up.get(k) for k in ("cached", "t_hash_s", "sha256", "time", "bytes", "mtime_ns")})
    if jobs_filter:
        miss = sorted(jobs_filter - {r["job"] for r in results})
        rep["meta"]["jobs_not_found"] = miss
    else:
        rep["meta"]["manifest_jobs_missing"] = sorted(set(mani) - {i["job"] for i in infos.values()})[:50]
        rep["meta"]["jobs_not_in_manifest"] = sorted({i["job"] for i in infos.values()} - set(mani))[:50]
    t_w = time.time()
    written = write_outputs(od, hdir, stem, rep, atoms, hess_label, hess_other, workers=nw)
    rep["meta"]["t_write_s"] = round(time.time() - t_w, 1)
    rep["meta"]["wall_s"] = round(time.time() - t0, 1)
    rep["outputs"] = written
    C.write_json(od / "report.json", rep)
    complete = bool(jobs_filter is None and stopped is None and (sha_ok or not ent.get("sha256"))
                    and not rep["summary"]["n_worker_exceptions"])
    if complete:
        C.write_json(done, dict(code=CODE_VERSION, time=C.now(), tar=stem, sha256=sha, sha256_ok=sha_ok,
                                n_jobs=rep["summary"]["n_jobs"], n_structures=rep["summary"]["n_structures"],
                                n_hess_label=len(hess_label), outputs=written, slurm_job=os.environ.get("SLURM_JOB_ID")))
    s = rep["summary"]
    print(json.dumps(dict(tar=stem, done=complete, jobs=s["n_jobs"], structures=s["n_structures"], kinds=s["kinds"],
                          ts_near=s["ts_near"], hess=s["hess"], errors=s["n_job_errors"], wall_s=rep["meta"]["wall_s"],
                          outputs={k: v.get("bytes") for k, v in written.items() if isinstance(v, dict)}),
                     indent=1, default=C._json_default), flush=True)
    if not complete and jobs_filter is None:
        C.stop(f"{stem}: not marked done (sha {sha_ok}, worker exceptions {s['n_worker_exceptions']}, stopped {stopped})")
    return 0


def assemble(ctx, stem, results, metas, infos):
    """worker results -> report dict, Atoms list (deduplicated), label TS Hessians, other Hessians."""
    results = sorted(results, key=lambda r: r["unit"])
    atoms, ids, keys = [], set(), {}
    kinds, near, near_rx, excl, dup_by = Counter(), Counter(), defaultdict(set), Counter(), Counter()
    jobs_rep, hess_label, hess_other = [], [], []
    n_err = n_bug = 0
    de_dup = 0.0
    tshort = short(stem)
    for r in results:
        u = r["unit"]
        meta, ji = metas[u], infos[u]
        mrow = meta.get("manifest") or {}
        pref = "nur_" if ji["target"] else "nur_x_"
        jt = ji["tags"]
        jr = dict(unit=u, job=ji["job"], target=ji["target"], rxn=ji["rxn"], cls=mrow.get("class"), tag_rule=jt["tag_rule"],
                  markers=meta["markers"], done=meta["done"], md5_ok=(r.get("md5") == mrow.get("archive_md5"))
                  if mrow.get("archive_md5") else None, md5=r.get("md5"), arc_bytes=r.get("arc_bytes"),
                  n_unpacked=r["n_unpacked"], bytes_unpacked=r["bytes_unpacked"], ts_roots=r.get("ts_roots"),
                  stale_archives=meta.get("stale_archives"), errors=r["errors"], dup_paths=r["dup_paths"][:5],
                  counts=dict(r["counts"]), t_s=r.get("t_total_s"), t_unpack_s=r.get("t_unpack_s"),
                  t_parse_s=r.get("t_parse_s"), n_top_files=meta.get("n_top"), top_bytes=meta.get("top_bytes"),
                  manifest_bytes=mrow.get("bytes"),
                  manifest_n_files=mrow.get("n_files"), ref_hess=r.get("ref_hess"),
                  ref_hess_best_dev_A=r.get("ref_hess_best_dev_A"))
        if mrow.get("archive_md5") and not jr["md5_ok"]:
            r["errors"].append("archive md5 != manifest")
        if r["errors"]:
            n_err += 1
        if any(e.startswith("worker:") for e in r["errors"]):
            n_bug += 1
        fk = Counter()
        exc_files = []
        for f in r["files"]:
            if not f["kept"]:
                rs = f["reason"]
                excl[(f["cat"], rs.split(":")[0] if not rs.startswith("not_label_level") else rs)] += 1
                if (f["level_ok"] and f["cat"] != "sp_stage" and not (rs == "no_gradient_steps" and f["jt"] == "freq")) \
                        or (rs == "not_orca_or_no_echo" and f["cat"] not in ("ref",)) or rs.startswith("parse_error"):
                    exc_files.append([f["path"], rs])
        jr["excluded_label_level"] = exc_files
        jr["n_files_parsed"] = len(r["files"])
        kept_rep = []
        retry = "+".join(x.lstrip(".") for x in meta["markers"] if x.startswith((".retry", ".envfix")))
        for kf in r["kept"]:
            f = kf["f"]
            base = kf["base"]
            if base in MOL_KINDS:
                tags, mkey = ctx.mol_tags(ji, meta, base, f["path"], kf["symbols"])
            else:
                tags, mkey = jt, None
            stem_p = re.sub(r"\.out(\.gz)?$", "", f["path"])
            n_near_f = 0
            for i, kind in enumerate(kf["kinds"]):
                sid = f"N:{u}:{stem_p}:{kf['cycles'][i]}"
                if sid in ids:
                    C.stop(f"duplicate struct_id {sid}")
                ids.add(sid)
                k = pref + kind
                gk = geom_key(kf["symbols"], kf["pos"][i])
                if gk in keys:                                 # exact duplicate geometry: keep the first, unite tags
                    a0 = atoms[keys[gk]]
                    dup_by[f"{a0.info['kind']}<-{k}"] += 1
                    de_dup = max(de_dup, abs(a0.info["e_eh"] - float(kf["energy"][i])))
                    for t in C.EXT_TAGS:                       # finetune._dedupe rule
                        a0.info[t] = C.tag_list(set(x for x in str(a0.info[t]).split(",") if x) |
                                                set(x for x in str(tags[t]).split(",") if x))
                    continue
                info = dict(job=ji["job"], tar=tshort, group=ji["group"], cls=mrow.get("class"), cycle=kf["cycles"][i],
                            n_cycles=int(f.get("n_cycles") or 0), file_sha="h" + f["sha16"],   # 'h': stays a str in ase
                            fmax=float(np.abs(kf["grad"][i]).max()), tag_rule=tags["tag_rule"], retry=retry)
                if kf["dform"] is not None:
                    info["dform_label_A" if ji["target"] else "dform_own_A"] = kf["dform"][i]
                    if ji["target"] and kf["dform"][i] <= NEAR_A:
                        near[kind] += 1
                        near_rx[kind].add(ji["rxn"])
                        n_near_f += 1
                if kind == "ts_final" and f.get("label_run"):
                    info["label_ts"] = 1
                if base == "truncated":
                    info["jt"] = f["jt"]
                if mkey:
                    info["mol"] = "m" + hashlib.sha1(mkey.encode()).hexdigest()[:12]
                if ji.get("d0_id") is not None:
                    info["d0_id"] = ji["d0_id"]
                info = {kk: v for kk, v in info.items() if v is not None and v != ""}   # ase: '' breaks the reader
                a = C.ext_atoms(kf["symbols"], kf["pos"][i], float(kf["energy"][i]), kf["grad"][i], struct_id=sid,
                                kind=k, source="nurion", rxn_ids=tags["rxn_ids"], folds=tags["folds"],
                                dipoles=tags["dipoles"], level="label", **info)
                keys[gk] = len(atoms)
                atoms.append(a)
                kinds[k] += 1
                fk[k] += 1
            kept_rep.append([f["path"], pref + base, len(kf["kinds"]), n_near_f, f.get("trj"), bool(f.get("label_run"))])
        jr["kept"] = kept_rep
        jr["kinds"] = dict(fk)
        jr["n_structures"] = int(sum(fk.values()))
        # Hessians
        hj = Counter()
        for h in r["hess"]:
            k = h.get("kind", "excluded")
            if k == "excluded":
                hj[f"excluded:{h.get('reason', '?').split(' ')[0]}"] += 1
                continue
            hk = ("x_" if not ji["target"] else "") + k
            hj[hk] += 1
            rec = {kk: v for kk, v in h.items() if not kk.startswith("_")}
            rec.update(kind=hk, unit=u, job=ji["job"], struct_id=f"N:{u}:{h['path'][:-len('.hess')]}:hess",
                       rxn_ids=jt["rxn_ids"], folds=jt["folds"], dipoles=jt["dipoles"], tag_rule=jt["tag_rule"])
            if k in ("mol_hess",):
                rec["tag_note"] = "molecule Hessian: job tags (no molecule union)"
            arr = dict(H=h["_H"], pos=h["_pos"], sym=h["_sym"], masses=h["_masses"])
            if k == "ts_hess_label":
                rec.update(rxn_id=ji["rxn"], fold=int(ctx.T.at[ji["rxn"], "fold_T1"]), dipole=ctx.T.at[ji["rxn"], "dipole"])
                hess_label.append((rec, arr))
            else:
                hess_other.append((rec, arr))
        jr["hess"] = dict(hj)
        if ji["target"]:
            jr["label_ts_hess"] = bool(r.get("ref_hess"))
        jobs_rep.append(jr)
    # summary
    hsum = Counter()
    for j in jobs_rep:
        for k, v in j["hess"].items():
            hsum[k] += v
    tgt = [j for j in jobs_rep if j["target"]]
    lab_ok = [j for j in tgt if j.get("label_ts_hess")]
    hl_dev = [h[0]["dev_ref_raw_A"] for h in hess_label]
    hl_imag = Counter(h[0].get("n_imag") for h in hess_label)
    summ = dict(n_jobs=len(jobs_rep), n_target_jobs=len(tgt), n_nontarget_jobs=len(jobs_rep) - len(tgt),
                classes=dict(Counter(j["cls"] for j in jobs_rep)), tag_rules=dict(Counter(j["tag_rule"] for j in jobs_rep)),
                n_structures=len(atoms), kinds=dict(sorted(kinds.items())),
                n_structures_target=int(sum(v for k, v in kinds.items() if not k.startswith("nur_x_"))),
                ts_near=dict(threshold_A=NEAR_A, by_kind=dict(near), n=int(sum(near.values())),
                             n_reactions=len(set().union(*near_rx.values())) if near_rx else 0,
                             optts_n=int(sum(near[k] for k in OPTTS_KINDS)),
                             optts_reactions=len(set().union(*[near_rx[k] for k in OPTTS_KINDS if k in near_rx]))
                             if any(k in near_rx for k in OPTTS_KINDS) else 0),
                hess=dict(sorted(hsum.items())), label_ts_hess=dict(n=len(hess_label), target_jobs=len(tgt),
                                                                    target_jobs_ok_class=sum(1 for j in tgt if j["cls"] == "ok"),
                                                                    with_hess=len(lab_ok),
                                                                    max_dev_A=max(hl_dev) if hl_dev else None,
                                                                    n_imag=dict(hl_imag),
                                                                    ok_without=[j["job"] for j in tgt if j["cls"] == "ok"
                                                                                and not j.get("label_ts_hess")][:30]),
                excluded={f"{c}|{r}": n for (c, r), n in sorted(excl.items())},
                dedupe=dict(n_dropped=int(sum(dup_by.values())), pairs=dict(dup_by.most_common(20)), max_de_eh=de_dup),
                top_files_ne_manifest=[j["job"] for j in jobs_rep if j["manifest_n_files"] not in (None, "")
                                       and int(j["manifest_n_files"]) != j["n_top_files"]][:50],
                n_job_errors=n_err, n_worker_exceptions=n_bug,
                job_errors={j["job"]: j["errors"][:3] for j in jobs_rep if j["errors"]}, md5_mismatch=[j["job"] for j in jobs_rep if j["md5_ok"] is False],
                markers=dict(Counter(m for j in jobs_rep for m in j["markers"])),
                checks=dict(ctx.checks), step_check="inv_dft.parse_file: steps == optimisation cycles (EnGrad 1/0) and "
                                                    "== *_trj.xyz frames (+1 final frame of a converged run) for every kept file",
                t_job_s=dict(q50=float(np.median([j["t_s"] or 0 for j in jobs_rep])) if jobs_rep else None,
                             q90=float(np.quantile([j["t_s"] or 0 for j in jobs_rep], 0.9)) if jobs_rep else None,
                             max=max((j["t_s"] or 0 for j in jobs_rep), default=None),
                             sum=float(sum(j["t_s"] or 0 for j in jobs_rep))))
    rep = dict(tar=stem, summary=summ, jobs=jobs_rep,
               rules=dict(doc="inv_nurion.py module docstring (kinds, TS-near, tags, Hessians)",
                          dedupe=f"exact duplicate geometry (symbols + positions rounded to 1e-{DEDUPE_DEC} A, finetune "
                                 "rule) within the tar: the first in (job, path, cycle) order is kept, tags united",
                          done="written last, iff the whole tar was streamed, its sha256 equals the registry value and "
                               "no worker raised (per-job archive / md5 problems are listed in job_errors)"))
    return rep, atoms, hess_label, hess_other


def _gz_chunk(atoms_list):
    """mlt_common.write_extxyz text (tags as JSON lists) of some structures -> one gzip member (bytes)."""
    from ase.io import write
    out = []
    for a in atoms_list:
        b = a.copy()
        for k in C.EXT_TAGS:
            if k in b.info and not isinstance(b.info[k], list):
                b.info[k] = [x for x in str(b.info[k]).split(",") if x]
        out.append(b)
    buf = io.StringIO()
    if out:
        write(buf, out, format="extxyz")
    return gzip.compress(buf.getvalue().encode(), compresslevel=6)


def write_extxyz_gz(path, atoms_list, workers=1):
    """mlt_common.write_extxyz format into a multi-member gzip file (chunks of EXT_CHUNK structures formatted and
    compressed in parallel; gzip / ase / read_ext read the concatenated members as one stream)."""
    chunks = [atoms_list[i:i + EXT_CHUNK] for i in range(0, len(atoms_list), EXT_CHUNK)] or [[]]

    def w(t):
        with open(t, "wb") as fz:
            if workers > 1 and len(chunks) > 1:
                with ProcessPoolExecutor(max_workers=min(workers, len(chunks))) as ex:
                    for blob in ex.map(_gz_chunk, chunks):
                        fz.write(blob)
            else:
                for c in chunks:
                    fz.write(_gz_chunk(c))
    return C._atomic(path, w)


def read_ext(path):
    """extxyz or extxyz.gz -> mlt_common.read_extxyz. A .gz is first decompressed to a node-local temp file: ase seeks
    to every frame, which is quadratic on a gzip stream (759 structures: 8.8 s from .gz vs 0.5 s plain)."""
    p = Path(path)
    if not p.name.endswith(".gz"):
        return C.read_extxyz(p)
    fd, tmp = tempfile.mkstemp(suffix=".extxyz", prefix="nurread_", dir=os.environ.get("TMPDIR", "/tmp"))
    try:
        with os.fdopen(fd, "wb") as g, gzip.open(p, "rb") as f:
            shutil.copyfileobj(f, g, 1 << 22)
        if os.path.getsize(tmp) == 0:
            return []
        return C.read_extxyz(tmp)
    finally:
        os.unlink(tmp)


def _savez(path, arrs):
    def w(t):
        with open(t, "wb") as fz:
            np.savez_compressed(fz, **arrs)
    return C._atomic(path, w)


def write_outputs(od, hdir, stem, rep, atoms, hess_label, hess_other, workers=1):
    written = {}
    p = write_extxyz_gz(od / "nurion.extxyz.gz", atoms, workers=workers)
    written["extxyz"] = dict(path=str(p), bytes=p.stat().st_size, n=len(atoms))
    # label TS Hessians: one npz per reaction (never overwrite one written from another tar)
    nl, conflicts, hb = 0, [], 0
    for rec, a in hess_label:
        dst = hdir / f"{rec['rxn_id']}.npz"
        if dst.exists():
            try:
                with np.load(dst, allow_pickle=False) as z:
                    prev = str(z["tar"])
            except Exception:                                    # noqa: BLE001
                prev = "?"
            if prev != stem:
                conflicts.append([rec["rxn_id"], prev])
                continue
        _savez(dst, dict(symbols=np.array(a["sym"]), positions_A=a["pos"], hessian=a["H"], masses=a["masses"],
                         source=np.array(rec["path"]), tar=np.array(stem), job=np.array(rec["job"]),
                         rxn_id=np.array(rec["rxn_id"]), fold=np.array(rec["fold"]), dipole=np.array(rec["dipole"]),
                         max_dev_A=np.array(rec["dev_ref_raw_A"]), kabsch_max_dev_A=np.array(rec["dev_ref_kabsch_A"]),
                         n_imag=np.array(-1 if rec.get("n_imag") is None else rec["n_imag"]),
                         freq_low_cm=np.array(rec.get("freq_low") or [], float), units=np.array("Eh/bohr^2"),
                         code=np.array(CODE_VERSION)))
        nl += 1
        hb += dst.stat().st_size
        rec["file"] = str(dst)
    cset = {c[0] for c in conflicts}
    written["hess_label"] = dict(dir=str(hdir), n=nl, bytes=hb, conflicts=conflicts,
                                 rxn_ids=sorted(int(r["rxn_id"]) for r, _ in hess_label if r["rxn_id"] not in cset))
    # other Hessians: one npz per tar (float32 upper triangle) + json index
    arrs, index = {}, []
    for i, (rec, a) in enumerate(hess_other):
        n3 = a["H"].shape[0]
        iu = np.triu_indices(n3)
        arrs[f"H{i}"] = a["H"][iu]
        arrs[f"P{i}"] = a["pos"]
        arrs[f"M{i}"] = a["masses"]
        rec = dict(rec, i=i, symbols=a["sym"])
        index.append(rec)
    p = _savez(od / "hess_other.npz", arrs)
    written["hess_other"] = dict(path=str(p), bytes=p.stat().st_size, n=len(index),
                                 by_kind=dict(Counter(r["kind"] for r in index)))
    q = C.write_json(od / "hess_other.json", dict(tar=stem, code=CODE_VERSION, storage="H<i> = float32 upper triangle "
                                                  "(np.triu_indices(3N)) Eh/bohr^2, P<i> positions A, M<i> masses amu",
                                                  hessians=index))
    written["hess_other_index"] = dict(path=str(q), bytes=q.stat().st_size)
    rep["hess_label"] = [{k: v for k, v in r.items() if k not in ("symbols",)} for r, _ in hess_label]
    return written


# ---------------------------------------------------------------- verify / pending / status
def cmd_verify(args):
    reg = registry()
    stem, _ = resolve(args.tar, reg)
    root, hdir = out_dirs(args.root)
    od = root / stem
    if not (od / "report.json").exists():
        C.not_ready(f"{od}/report.json missing")
    rep = C.read_json(od / "report.json")
    al = read_ext(od / "nurion.extxyz.gz")
    T = C.targets().set_index("rxn_id")
    bad, kinds = Counter(), Counter()
    for a in al:
        kinds[a.info["kind"]] += 1
        rx = [x for x in a.info["rxn_ids"].split(",") if x]
        fo = {x for x in a.info["folds"].split(",") if x}
        dp = {x for x in a.info["dipoles"].split(",") if x}
        if any(int(x) not in T.index for x in rx):
            bad["rxn_ids_not_d1"] += 1
        want_f = {str(int(T.loc[int(x), "fold_T1"])) for x in rx}
        if not want_f <= fo:
            bad["folds_missing_target_fold"] += 1
        if not {T.loc[int(x), "dipole"] for x in rx} <= dp:
            bad["dipoles_missing_target_dipole"] += 1
        if a.info["kind"].startswith("nur_") and not a.info["kind"].startswith("nur_x_"):
            j = a.info["job"]
            if int(C.rxn_of(j)) not in {int(x) for x in rx}:
                bad["target_row_without_own_rxn"] += 1
        F = a.arrays["REF_forces"]
        if not (np.isfinite(F).all() and np.isfinite(a.info["REF_energy"])):
            bad["non_finite"] += 1
        if abs(np.abs(F).max() / C.EHBOHR2EVA - float(a.info["fmax"])) > 1e-8:
            bad["fmax_ne_forces"] += 1
        if abs(a.info["REF_energy"] / C.EH2EV - float(a.info["e_eh"])) > 1e-9:
            bad["energy_units"] += 1
    ids = [a.info["struct_id"] for a in al]
    hl = list(hdir.glob("*.npz"))
    hcheck = Counter()
    for r in rep.get("hess_label", []):
        p = Path(r.get("file", ""))
        if not p.exists():
            hcheck["missing"] += 1
            continue
        with np.load(p, allow_pickle=False) as z:
            H = z["hessian"]
            ok = (H.shape == (3 * len(z["symbols"]),) * 2 and np.isfinite(H).all()
                  and float(z["kabsch_max_dev_A"]) <= HESS_GEOM_TOL_A)
            hcheck[f"n_imag_{int(z['n_imag'])}"] += 1          # recorded, not a failure (< -50 cm^-1 rule)
            sym, pos = C.read_xyz(C.label_ts(int(z["rxn_id"])))
            ok = ok and list(z["symbols"]) == sym and kabsch(z["positions_A"], pos)[1] <= HESS_GEOM_TOL_A
            hcheck["raw_dev_le_1e-4"] += raw_dev(z["positions_A"], pos) <= HESS_GEOM_TOL_A
        hcheck["ok" if ok else "bad"] += 1
    with np.load(od / "hess_other.npz", allow_pickle=False) as z:
        n_other = len([k for k in z.files if k.startswith("H")])
    out = dict(tar=stem, n_read=len(al), n_report=rep["summary"]["n_structures"], unique_ids=len(set(ids)) == len(ids),
               kinds_match=dict(kinds) == rep["summary"]["kinds"], problems=dict(bad), hess_label=dict(hcheck),
               hess_label_files_in_dir=len(hl), hess_other=n_other, hess_other_report=rep["outputs"]["hess_other"]["n"],
               done=(od / ".done").exists())
    print(json.dumps(out, indent=1))
    good = (out["n_read"] == out["n_report"] and out["unique_ids"] and out["kinds_match"] and not bad
            and not hcheck.get("bad") and not hcheck.get("missing") and n_other == out["hess_other_report"])
    return 0 if good else 3


def _deleted_set():
    p = OUT / "deleted_tars.tsv"
    if not p.exists():
        return {}
    out = {}
    for ln in p.read_text().splitlines():
        t = ln.split("\t")
        if t and t[0] and not t[0].startswith("#"):
            out[re.sub(r"\.tar(\.gz|\.xz)?$", "", Path(t[0]).name)] = t[1:] if len(t) > 1 else []
    return out


def tar_states(root=None):
    reg = registry()
    root, _ = out_dirs(root)
    dele = _deleted_set()
    rows = []
    for stem, e in reg.items():
        p = UPLOAD / e["file"]
        up = p.exists()
        size = p.stat().st_size if up else None
        dn = root / stem / ".done"
        st = "done" if dn.exists() else ("uploading" if (up and e.get("bytes") and size != e["bytes"]) else
                                         ("pending" if up else ("deleted" if stem in dele else "not_uploaded")))
        d = C.read_json(dn) if dn.exists() else {}
        rows.append(dict(tar=stem, kind=e["kind"], file=e["file"], uploaded=up, bytes=size, expected_bytes=e.get("bytes"),
                         state=st, done_time=d.get("time"), n_structures=d.get("n_structures"),
                         n_hess_label=d.get("n_hess_label"), deleted=stem in dele))
    extra = sorted(x.name for x in UPLOAD.glob("d1_*.tar*") if re.sub(r"\.tar(\.gz|\.xz)?$", "", x.name) not in reg)
    return rows, extra


def cmd_pending(args):
    rows, extra = tar_states(args.root)
    print(f"{'tar':38s} {'kind':6s} {'state':12s} {'GB':>6s} {'structures':>10s}  done")
    for r in rows:
        if r["kind"] == "other" and not r["uploaded"]:
            continue
        gb = f"{r['bytes'] / 1e9:.2f}" if r["bytes"] else "-"
        print(f"{r['tar']:38s} {r['kind']:6s} {r['state']:12s} {gb:>6s} {str(r['n_structures'] or '-'):>10s}  "
              f"{r['done_time'] or ''}")
    todo = [r for r in rows if r["state"] == "pending" and r["kind"] in ("result", "extra")]
    print(f"\n{sum(r['state'] == 'done' for r in rows if r['kind'] in ('result', 'extra'))} done, {len(todo)} pending "
          f"(uploaded, not processed); unknown uploads: {extra or 'none'}")
    for r in todo:
        print(f"  sbatch -p cpu1,cpu2 -J nur_{short(r['tar'])} --time=48:00:00 -c 8 --mem=24G "
              f"--comment=mltp:nur:{r['tar']}:manual --export=ALL,MLTENV=reactot jobs/mlt.slurm inv_nurion.py scan "
              f"{r['tar']} --workers 8 ::: inv_nurion.py verify {r['tar']}")
    return 0


def cmd_status(args):
    rows, _ = tar_states(args.root)
    out = dict(tars={r["tar"]: dict(state=r["state"], structures=r["n_structures"], hess_label=r["n_hess_label"])
                     for r in rows if r["kind"] in ("result", "extra")})
    if RES_N.exists():
        n = C.read_json(RES_N)
        out["collect"] = dict(time=n.get("meta", {}).get("time"), complete=n.get("complete"),
                              tars_processed=n.get("tars_processed"), n_structures=n.get("n_structures"),
                              up_to_date=n.get("signature") == _signature(args.root))
    print(json.dumps(out, indent=1))
    return 0


# ---------------------------------------------------------------- collect
def _signature(root=None):
    root, _ = out_dirs(root)
    h = hashlib.sha256(CODE_VERSION.encode())
    for d in sorted(root.glob("*/.done")):
        h.update(d.parent.name.encode() + b"|" + d.read_bytes())
    return h.hexdigest()


def cmd_collect(args):
    t0 = time.time()
    root, hdir = out_dirs(args.root)
    production = root.resolve() == OUT.resolve()
    out_ext = COLLECT_EXT if production else root / "collect" / "nurion.extxyz.gz"
    out_rep = RES_N if production else root / "collect" / "N_report.json"
    sig = _signature(args.root)
    if out_rep.exists() and out_ext.exists() and not args.force and C.read_json(out_rep).get("signature") == sig:
        print(f"[nur] collect up to date ({out_rep})")
        return 0
    rows, _ = tar_states(args.root)
    done = sorted([r for r in rows if r["state"] == "done"], key=lambda x: x["tar"])
    exp = [r for r in rows if r["kind"] in ("result", "extra")]
    reps = {r["tar"]: C.read_json(root / r["tar"] / "report.json") for r in done}
    nw = max(1, min(int(args.workers), len(done) + 1))
    # pass 1 (parallel, one task per tar): geometry key, struct_id, kind, tags, own rxn, dform of every row.
    # Phase I geometry keys: files Phase I already read are excluded at scan time (full sha256 in
    # results/I_sources.csv, reason dup_phase_i); the same geometry from another run is kept and counted.
    with ProcessPoolExecutor(max_workers=nw) as ex:
        fpi = ex.submit(_col_keys_phase_i, str(PHASE_I_EXT)) if PHASE_I_EXT.exists() else None
        p1 = list(ex.map(_col_read, [str(root / r["tar"] / "nurion.extxyz.gz") for r in done]))
        pi_keys = fpi.result() if fpi is not None else set()
    t1 = time.time()
    # geometry key -> first struct_id (tar name order, then file order) + the union of the tags
    first, tagu, n_in = {}, {}, 0
    for rws in p1:
        for k, sid, _kd, tg, _own, _d in rws:
            n_in += 1
            if k not in first:
                first[k] = sid
                tagu[k] = [set(x) for x in tg]
            else:
                for s, x in zip(tagu[k], tg):
                    s.update(x)
    kinds, near, near_rx, drop, drop_kind, pi_same_geom = (Counter(), Counter(), defaultdict(set), Counter(), Counter(),
                                                           Counter())
    cover, cover_ts, per_tar = set(), set(), {}
    keeps = []
    for r, rws in zip(done, p1):
        keep = {}
        for k, sid, kd, _tg, own, dfo in rws:
            if first[k] != sid:
                drop["nurion_duplicate"] += 1
                drop_kind[kd] += 1
                continue
            if k in pi_keys:                                  # same geometry as a Phase I row (another run): kept,
                pi_same_geom[kd] += 1                         # finetune._dedupe unites the tags of exact duplicates
            keep[sid] = tuple(sorted(s) for s in tagu[k])     # JSON lists, as mlt_common.write_extxyz
            kinds[kd] += 1
            if own is not None:
                cover.add(own)
                base = kd[len("nur_"):]
                if base in TS_KINDS:
                    cover_ts.add(own)
                if dfo is not None and dfo <= NEAR_A:
                    near[base] += 1
                    near_rx[base].add(own)
        per_tar[r["tar"]] = dict(n_in=len(rws), n_kept=len(keep))
        keeps.append((str(root / r["tar"] / "nurion.extxyz.gz"), keep))
    n_out = int(sum(kinds.values()))
    t2 = time.time()
    # pass 2 (parallel): each tar's kept rows with the united tags -> gzip members, concatenated in tar order
    n_written = 0

    def w(tmp):
        nonlocal n_written
        with open(tmp, "wb") as fz, ProcessPoolExecutor(max_workers=nw) as ex:
            for blob, n in ex.map(_col_write, keeps):
                fz.write(blob)
                n_written += n
    if not args.dry:
        C._atomic(out_ext, w)
        if n_written != n_out:
            C.stop(f"collect wrote {n_written} structures, expected {n_out}")
    # Hessians
    hl, hother, excl = 0, Counter(), Counter()
    for t, rp in reps.items():
        hl += rp["outputs"]["hess_label"]["n"]
        for k, v in rp["outputs"]["hess_other"]["by_kind"].items():
            hother[k] += v
        for k, v in rp["summary"]["excluded"].items():
            excl[k] += v
    inv = C.read_json(I.RES_INV) if I.RES_INV.exists() else {}
    pi1 = (inv.get("I1") or {})
    nn = int(sum(near.values()))
    nr = len(set().union(*near_rx.values())) if near_rx else 0
    no = int(sum(near[k] for k in OPTTS_KINDS))
    nro = len(set().union(*[near_rx[k] for k in OPTTS_KINDS if k in near_rx])) if any(k in near_rx for k in OPTTS_KINDS) else 0
    uploaded = [r for r in exp if r["uploaded"] or r["state"] in ("done", "deleted")]
    res_t = [r for r in exp if r["kind"] == "result"]
    res_done = [r for r in res_t if r["state"] == "done"]
    res_up = [r for r in res_t if r["uploaded"] or r["state"] in ("done", "deleted")]
    rep = dict(spec="docs/specs/DEPLOY_V2_MLTS.md section 2 (Phase I extension: Nurion D1 run tars)",
               complete=len(res_done) == N_RESULT_TARS,
               complete_uploaded=bool(uploaded) and all(r["state"] == "done" for r in uploaded),
               complete_all=all(r["state"] == "done" for r in exp),
               tars_done=len(res_done), tars_uploaded=len(res_up), tars_expected=N_RESULT_TARS,
               tars_not_done=[r["tar"] for r in res_t if r["state"] != "done"],
               extras=dict(expected=len(EXTRA_DATA), done=sum(r["state"] == "done" for r in exp if r["kind"] == "extra")),
               tars_processed=len(done),
               tars=[dict(tar=r["tar"], state=r["state"], n_structures=r["n_structures"], **per_tar.get(r["tar"], {}))
                     for r in exp],
               n_structures_in=n_in, n_structures=n_out, dropped=dict(drop), dropped_by_kind=dict(drop_kind),
               phase_i_same_geometry_kept=dict(pi_same_geom),
               kinds=dict(sorted(kinds.items())), reactions_covered=len(cover), reactions_with_ts_kinds=len(cover_ts),
               ts_near=dict(threshold_A=NEAR_A, n=nn, n_reactions=nr, by_kind=dict(near), optts_n=no, optts_reactions=nro),
               I1_recount=dict(rule="TS-near <= 0.3 A to the label TS: >= 50,000 structures over >= 5,000 reactions",
                               nurion_all_ts_kinds=dict(n=nn, reactions=nr, pass_=nn >= 50000 and nr >= 5000),
                               nurion_optts_only=dict(n=no, reactions=nro, pass_=no >= 50000 and nro >= 5000),
                               phase_i=dict(n=pi1.get("n_structures"), reactions=pi1.get("n_reactions"))),
               hessians=dict(label_ts=hl, label_ts_files=len(list(hdir.glob("*.npz"))), other=dict(hother)),
               exclusions=dict(sorted(excl.items())), signature=sig, extxyz=str(out_ext),
               extxyz_bytes=out_ext.stat().st_size if (not args.dry and out_ext.exists()) else None,
               meta=dict(code=CODE_VERSION, time=C.now(), host=os.uname().nodename, slurm_job=os.environ.get("SLURM_JOB_ID"),
                         workers=nw, pass1_s=round(t1 - t0, 1), merge_s=round(t2 - t1, 1),
                         pass2_s=round(time.time() - t2, 1), wall_s=round(time.time() - t0, 1), root=str(root),
                         dry=bool(args.dry)))
    if not args.dry:
        C.write_json(out_rep, rep)
    print(json.dumps({k: rep[k] for k in ("complete", "complete_uploaded", "tars_done", "tars_uploaded", "tars_expected",
                                          "n_structures_in", "n_structures", "dropped", "phase_i_same_geometry_kept",
                                          "kinds", "reactions_covered", "ts_near", "I1_recount", "hessians",
                                          "extxyz_bytes", "meta")},
                     indent=1, default=C._json_default))
    return 0


def _col_read(path):
    """collect pass 1 (worker): per row (geometry key, struct_id, kind, tags, own rxn of target rows, dform_label_A)."""
    out = []
    for a in read_ext(path):
        kd = a.info["kind"]
        own = C.rxn_of(a.info["job"]) if (kd.startswith("nur_") and not kd.startswith("nur_x_")) else None
        dfo = a.info.get("dform_label_A")
        out.append((geom_key(a.get_chemical_symbols(), a.positions), a.info["struct_id"], kd,
                    tuple(tuple(x for x in str(a.info[t]).split(",") if x) for t in C.EXT_TAGS), own,
                    None if dfo is None else float(dfo)))
    return out


def _col_keys_phase_i(path):
    return {geom_key(a.get_chemical_symbols(), a.positions) for a in C.read_extxyz(path)}


def _col_write(item):
    """collect pass 2 (worker): one tar's rows kept by the global dedupe (struct_id -> united tags) -> gzip bytes."""
    path, keep = item
    rows = []
    for a in read_ext(path):
        t = keep.get(a.info["struct_id"])
        if t is None:
            continue
        for k, v in zip(C.EXT_TAGS, t):
            a.info[k] = list(v)
        rows.append(a)
    return b"".join(_gz_chunk(rows[i:i + EXT_CHUNK]) for i in range(0, len(rows), EXT_CHUNK)), len(rows)


# ---------------------------------------------------------------- release (deletion of one uploaded tar copy)
def release_checks(stem, ent, deep=True):
    """every precondition for deleting the uploaded copy of <stem>; returns (ok, checks dict)."""
    od = OUT / stem
    p = UPLOAD / ent["file"]
    ck = {}
    ck["registry_kind_result_or_extra"] = ent["kind"] in ("result", "extra")
    ck["uploaded_file_exists"] = p.exists()
    ck["not_in_deleted_tars_tsv"] = stem not in _deleted_set()
    dn = od / ".done"
    ck["done_marker"] = dn.exists()
    d = C.read_json(dn) if dn.exists() else {}
    up = C.read_json(od / "upload_ok.json") if (od / "upload_ok.json").exists() else {}
    ck["done_sha256_ok"] = bool(d.get("sha256_ok")) and d.get("sha256") == ent.get("sha256")
    ck["verify_upload_sha256_ok"] = bool(up.get("ok")) and up.get("sha256") == ent.get("sha256")
    if p.exists():
        st = p.stat()
        ck["file_unchanged_since_verify"] = (up.get("bytes"), up.get("mtime_ns")) == (st.st_size, st.st_mtime_ns)
        ck["size_eq_index"] = (not ent.get("bytes")) or st.st_size == ent["bytes"]
    rp = od / "report.json"
    ck["report_json"] = rp.exists()
    ext = od / "nurion.extxyz.gz"
    ck["extxyz_exists"] = ext.exists()
    if rp.exists() and ext.exists():
        rep = C.read_json(rp)
        ck["report_full_tar"] = (rep.get("meta") or {}).get("jobs_filter") is None and \
            (rep.get("meta") or {}).get("stopped") is None
        ck["extxyz_size_eq_done"] = ext.stat().st_size == ((d.get("outputs") or {}).get("extxyz") or {}).get("bytes")
        if deep:
            n = len(read_ext(ext))
            ck["extxyz_read_back_n_eq_report"] = n == rep["summary"]["n_structures"]
            ck["extxyz_n"] = n
    ok = all(v for k, v in ck.items() if isinstance(v, bool))
    return ok, ck


def cmd_release(args):
    reg = registry()
    stem, ent = resolve(args.tar, reg)
    if args.root and Path(args.root).resolve() != OUT.resolve():
        C.stop("release works on the production root only")
    ok, ck = release_checks(stem, ent, deep=not args.shallow)
    p = UPLOAD / ent["file"]
    plan = dict(tar=stem, file=str(p), bytes=p.stat().st_size if p.exists() else None, would_delete=ok,
                ledger=str(OUT / "deleted_tars.tsv"), checks=ck, dry=bool(args.dry))
    print(json.dumps(plan, indent=1, default=C._json_default), flush=True)
    if args.dry:
        return 0 if ok else 3
    if not ok:
        C.stop(f"release {stem}: preconditions failed — nothing deleted")
    if not args.reviewed_by:
        C.stop("release without --dry needs --reviewed-by '<independent review reference>' (DECISIONS 'Nurion 속도')")
    led = OUT / "deleted_tars.tsv"
    old = led.read_text() if led.exists() else "# tar\tsha256\tbytes\tdeleted_at\treviewed_by\tdone_time\thost\n"
    d = C.read_json(OUT / stem / ".done")
    line = "\t".join([ent["file"], ent["sha256"], str(plan["bytes"]), C.now(), str(args.reviewed_by).replace("\t", " "),
                      str(d.get("time")), os.uname().nodename])
    C.write_text(led, old + line + "\n")          # ledger first: a crash after this leaves a recorded, present file
    os.unlink(p)                                   # the one uploaded tar copy (outside $MLT: explicit, single file)
    print(f"[nur] released {p}", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("pending", "status"):
        s = sub.add_parser(n)
        s.add_argument("--root", default=None)
    s = sub.add_parser("scan")
    s.add_argument("tar")
    s.add_argument("--jobs", default=None, help="comma-separated job ids (test subset; needs a test --root)")
    s.add_argument("--workers", type=int, default=max(1, I.n_cpus()))
    s.add_argument("--root", default=None, help="output root (default $MLT/dft/nurion)")
    s.add_argument("--force", action="store_true")
    v = sub.add_parser("verify")
    v.add_argument("tar")
    v.add_argument("--root", default=None)
    c = sub.add_parser("collect")
    c.add_argument("--root", default=None)
    c.add_argument("--force", action="store_true")
    c.add_argument("--dry", action="store_true")
    c.add_argument("--workers", type=int, default=max(1, I.n_cpus()))
    u = sub.add_parser("verify_upload")
    u.add_argument("tar")
    u.add_argument("--root", default=None)
    u.add_argument("--recheck", action="store_true", help="hash again even if upload_ok.json matches")
    r = sub.add_parser("release")
    r.add_argument("tar")
    r.add_argument("--dry", action="store_true")
    r.add_argument("--shallow", action="store_true", help="skip reading the extxyz back")
    r.add_argument("--reviewed-by", default=None)
    r.add_argument("--root", default=None)
    args = ap.parse_args()
    return dict(pending=cmd_pending, scan=cmd_scan, verify=cmd_verify, collect=cmd_collect, status=cmd_status,
                verify_upload=cmd_verify_upload, release=cmd_release)[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
