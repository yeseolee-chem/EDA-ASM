"""mlt_common.py — shared paths, constants and helpers for DEPLOY-V2 (MLTS), docs/specs/DEPLOY_V2_MLTS.md.

Every path can be overridden by an environment variable (MLT_*), so tests and smoke runs can point at their own roots.
Writes are allowed only under $MLT and analysis/mlts/results (assert_write_ok); everything else is read only
(spec header: $G2, $LAD, $DEP, $G3, $D1PAIR_SCRATCH, $D1RUN and the existing analysis/ code are never modified).
No compute here; importing this module is cheap (no numpy/pandas at import time)."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent                     # analysis/mlts
ANALYSIS = HERE.parent
REPO = ANALYSIS.parent


def _p(env, default):
    return Path(os.environ.get(env, default))


# ---------------------------------------------------------------- roots
MLT = _p("MLT_SCRATCH", "/gpfs/tmp_cpu2/yeseo1ee/mlts")
RES = _p("MLT_RESULTS", str(HERE / "results"))
G2 = _p("MLT_G2", "/gpfs/tmp_cpu2/yeseo1ee/d1_g2")
LAD = _p("MLT_LAD", "/gpfs/tmp_cpu2/yeseo1ee/d1_ladder")
DEP = _p("MLT_DEP", "/gpfs/tmp_cpu2/yeseo1ee/deploy")
G3 = _p("MLT_G3", "/gpfs/tmp_cpu2/yeseo1ee/d1_g3")
D1PAIR = _p("MLT_D1PAIR", os.environ.get("D1PAIR_SCRATCH", "/gpfs/tmp_cpu2/yeseo1ee/d1_pair"))
D1RUN = _p("MLT_D1RUN", "/gpfs/tmp_cpu2/yeseo1ee/d1/d1_run")
D1HF = _p("MLT_D1HF", "/gpfs/tmp_cpu2/yeseo1ee/d1_hf")
D1VAL = _p("MLT_D1VAL", "/gpfs/tmp_cpu2/yeseo1ee/d1_validation")
D1_ENV = _p("MLT_D1_ENV", "/gpfs/tmp_cpu2/yeseo1ee/d1/home01/d1_env")
PILOT = D1_ENV / "EDA-ASM" / "D1_build" / "d1_autode_pilot"           # pilot_common, run_ts, make_reference, orca_direct
RMSD_DIR = D1_ENV / "EDA-ASM" / "D1_build" / "validation" / "analysis"  # rmsd.same_ts

# ---------------------------------------------------------------- programs / environments
ORCA_DIR = _p("MLT_ORCA_DIR", "/home1/yeseo1ee/orca_6_1_1_avx2")
ORCA = ORCA_DIR / "orca"
CONDA = Path("/home1/yeseo1ee/miniconda3")
PY_MLTS = CONDA / "envs" / "mlts" / "bin" / "python"                  # new env (0-7): MLIPs, torch, sella
PY_REACTOT = CONDA / "envs" / "reactot" / "bin" / "python"            # parsing / sampling (never modified)
PY_D1 = D1_ENV / "venv" / "bin" / "python"                            # D1 venv (python 3.9.5, autodE 1.4.5 editable)
XTB_D1 = D1_ENV / "xtb-6.7.1" / "bin" / "xtb"

# ---------------------------------------------------------------- inputs (read only)
LABELS = _p("MLT_LABELS", str(LAD / "labels_d1_cr2.json"))           # 9,266 D1 targets (LADDER cr2 labels)
S_FOLDS = _p("MLT_S_FOLDS", str(ANALYSIS / "d1_pair" / "results" / "S_folds.csv"))
DFT_FINAL = LAD / "dft_final"                                        # <rxn_id>/{ts.xyz, rel1.xyz, rel2.xyz, work/ts.hess}
D1_JOBS = D1PAIR / "d1_jobs"                                         # T#####/{ts.xyz, ref_dipole.xyz, ref_dipolarophile.xyz}
G2_JOBS = G2 / "jobs"                                                # T#####/{ts.xyz, ts_result.json, ...} (autodE order)
G2_FINAL = G2 / "g2_final"                                           # <rxn_id>/ts.xyz (label order)
D1_CONFIG = D1RUN / "config_run.yaml"
G2_CONFIG = ANALYSIS / "d1_g2" / "config_g2.yaml"
C_COMPARE = ANALYSIS / "d1_g2" / "results" / "C_compare.csv"          # same_ts_g1 per reaction (g1 shape class)

# ---------------------------------------------------------------- constants
SEED = 20261104
TARGETS = ("d1", "d2", "elst", "pauli", "oi", "disp", "cpcm", "cds", "c_ghost", "barrier")
PRIMARY = ("d1", "d2", "elst", "pauli", "oi", "disp", "cpcm", "cds", "barrier")
ELEMENTS = ("H", "C", "N", "O", "F", "Si", "S", "Cl", "Br")
DIPOLES = ("azide", "azomethine ylide", "carbonyl ylide", "nitrile oxide", "nitrone")
EH2EV = 27.211386245988
BOHR2A = 0.529177210903
EH2KCAL = 627.509474
EHBOHR2EVA = EH2EV / BOHR2A                                          # gradient Eh/bohr -> eV/A
SAME_TS = dict(rmsd_A=0.10, dform_A=0.02)                            # config_run.yaml run.same_ts

# label level (spec 3: the D1 run `grad` keyword set exactly as autodE writes it; pilot_common.configure_autode)
LABEL_GRAD_LINE = "! EnGrad B3LYP RIJCOSX D3BJ def2-SVP def2/J  CPCM(Water) "
LABEL_SOLV_BLOCK = '%cpcm\nsmd true\nSMDsolvent "water"\nend\n'
LABEL_SCF_BLOCK = "%scf \nmaxiter 250 \nend\n"
LABEL_MAXCORE_MB = 3000


def orca_grad_input(symbols, pos_A, nprocs=1, maxcore=LABEL_MAXCORE_MB, charge=0, mult=1, title=None):
    """label-level EnGrad input text (same keywords and blocks as autodE's D1 inputs, without the opt/print extras)."""
    lines = [LABEL_GRAD_LINE]
    if title:
        lines.insert(0, f"# {title}")
    txt = "\n".join(lines) + "\n" + LABEL_SOLV_BLOCK + LABEL_SCF_BLOCK + f"% maxcore\n{int(maxcore)}\n"
    if int(nprocs) > 1:
        txt += f"%pal nprocs {int(nprocs)}\nend\n"
    txt += f"*xyz {int(charge)} {int(mult)}\n"
    txt += "".join(f"{s:<3s}{x:16.10f}{y:16.10f}{z:16.10f}\n" for s, (x, y, z) in zip(symbols, pos_A))
    return txt + "*\n"


# ---------------------------------------------------------------- exits / guards / atomic writes
def stop(msg):
    print(f"[STOP] {msg}", file=sys.stderr, flush=True)
    raise SystemExit(3)


def not_ready(msg):
    print(f"[NOT READY] {msg}", file=sys.stderr, flush=True)
    raise SystemExit(2)


def assert_write_ok(path):
    p = Path(path).resolve()
    for root in (MLT, RES):
        try:
            p.relative_to(Path(root).resolve())
            return Path(path)
        except ValueError:
            pass
    raise SystemExit(f"[GUARD] refusing to write outside $MLT / results: {p}")


def _atomic(path, write):
    path = assert_write_ok(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    write(tmp)
    os.replace(tmp, path)
    return path


def write_text(path, text):
    return _atomic(path, lambda t: Path(t).write_text(text))


def write_json(path, obj):
    return _atomic(path, lambda t: Path(t).write_text(json.dumps(obj, indent=1, default=_json_default) + "\n"))


def write_parquet(path, df):
    return _atomic(path, lambda t: df.to_parquet(t, index=False))


def write_csv(path, df):
    return _atomic(path, lambda t: df.to_csv(t, index=False))


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
    except ImportError:
        pass
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def array_task(i):
    """'@I' -> $SLURM_ARRAY_TASK_ID, else int(i)."""
    if str(i) == "@I":
        v = os.environ.get("SLURM_ARRAY_TASK_ID")
        if v is None:
            raise SystemExit("@I given but SLURM_ARRAY_TASK_ID is not set")
        return int(v)
    return int(i)


# ---------------------------------------------------------------- structures
def read_xyz(path):
    """(symbols, positions[[x,y,z]] in A) of a plain xyz file."""
    lines = Path(path).read_text().splitlines()
    n = int(lines[0].split()[0])
    sym, pos = [], []
    for ln in lines[2:2 + n]:
        t = ln.split()
        sym.append(t[0])
        pos.append([float(t[1]), float(t[2]), float(t[3])])
    return sym, pos


def job_of(rxn_id):
    return f"T{int(rxn_id) - 1000000:05d}"


def rxn_of(job_id):
    return 1000000 + int(str(job_id).lstrip("T"))


def label_ts(rxn_id):
    return DFT_FINAL / str(int(rxn_id)) / "ts.xyz"


def label_xtb_hess(rxn_id):
    """ORCA XTB2 ALPB(water) Freq at the exact label TS (LADDER G0; spec G-1 fallback: no DFT TS Hessian survives)."""
    return DFT_FINAL / str(int(rxn_id)) / "work" / "ts.hess"


def label_refs(rxn_id):
    """label reactant structures (dipole, dipolarophile) as written by D1 (label atom order)."""
    j = D1_JOBS / job_of(rxn_id)
    return j / "ref_dipole.xyz", j / "ref_dipolarophile.xyz"


def g2_ts_autode(rxn_id):
    return G2_JOBS / job_of(rxn_id) / "ts.xyz"


def g2_ts_label_order(rxn_id):
    return G2_FINAL / str(int(rxn_id)) / "ts.xyz"


def formed_pairs(s):
    """'3-7 5-6' (0-based, label order) -> [(3, 7), (5, 6)]."""
    return [tuple(int(x) for x in t.split("-")) for t in str(s).split()]


# ---------------------------------------------------------------- training structures (extxyz, shared by I / G / FT)
# One record = one structure at the label level. Leakage tags (spec 5, C-2) are carried as comma-separated strings:
#   rxn_ids  D1 rxn_ids the structure belongs to ('' for non-D1 data, e.g. D0 validation runs)
#   folds    T1 folds of those reactions (FT_k drops the structure iff k in folds)
#   dipoles  dipole names of those reactions (FT_LODO_d drops it iff d in dipoles)
EXT_KEYS = ("struct_id", "kind", "source", "rxn_ids", "folds", "dipoles", "e_eh", "level")


def ext_atoms(symbols, pos_A, energy_eh, grad_eh_bohr, **info):
    """ase.Atoms with REF_energy (eV) / REF_forces (eV/A, = -gradient) and the EXT_KEYS info fields."""
    import numpy as np
    from ase import Atoms
    a = Atoms(symbols=list(symbols), positions=np.asarray(pos_A, float))
    a.info["REF_energy"] = float(energy_eh) * EH2EV
    a.arrays["REF_forces"] = -np.asarray(grad_eh_bohr, float) * EHBOHR2EVA
    a.info["e_eh"] = float(energy_eh)
    a.info["level"] = info.pop("level", "label")
    for k in ("struct_id", "kind", "source", "rxn_ids", "folds", "dipoles"):
        a.info[k] = str(info.pop(k, ""))
    a.info.update({k: v for k, v in info.items() if v is not None})
    a.info["config_type"] = a.info["kind"]
    return a


def write_extxyz(path, atoms_list):
    from ase.io import write
    return _atomic(path, lambda t: write(str(t), list(atoms_list), format="extxyz"))


def tag_list(xs):
    return ",".join(str(x) for x in sorted({str(x) for x in xs if str(x) != ""}))


# ---------------------------------------------------------------- targets and folds
def core_folds():
    """S_folds core ('<dipole>/<skeleton>') -> T1 fold (-1 = anchor-only core, always training). One fold per core."""
    import pandas as pd
    S = pd.read_csv(S_FOLDS, usecols=["core", "fold_T1"])
    g = S.groupby("core").fold_T1.nunique()
    if (g > 1).any():
        stop(f"cores with more than one T1 fold: {list(g[g > 1].index)}")
    return S.drop_duplicates("core").set_index("core").fold_T1.astype(int).to_dict()


def targets():
    """the 9,266 D1 targets with rxn_id, job_id, dipole, skeleton, fold_core, fold_T1 (via core), lodo (dipole index),
    formed_pairs_ts (label order, 0-based string), n_atoms, n_f1, n_f2."""
    import pandas as pd
    L = pd.DataFrame(read_json(LABELS))
    cols = ["rxn_id", "job_id", "dipole", "skeleton", "core", "formed_pairs_ts", "n_atoms", "n_f1", "n_f2", "rxn_smiles"]
    T = L[cols].copy()
    T["rxn_id"] = T.rxn_id.astype(int)
    T["fold_core"] = T.dipole + "/" + T.skeleton
    cf = core_folds()
    miss = sorted(set(T.fold_core) - set(cf))
    T["fold_T1"] = T.fold_core.map(cf)
    T["fold_T1"] = T.fold_T1.fillna(-1).astype(int)
    T.attrs["cores_without_fold"] = miss
    T["lodo"] = T.dipole.map({d: i for i, d in enumerate(DIPOLES)}).astype(int)
    return T.sort_values("rxn_id").reset_index(drop=True)
