#!/usr/bin/env python3
r"""finetune.py — DEPLOY-V2 (MLTS) Phase FT: fine-tune a foundation MLIP to the D1 label level
(B3LYP-D3BJ/def2-SVP, SMD water) energies + forces (docs/specs/DEPLOY_V2_MLTS.md 5; budget 0-5; leakage C-2 (8);
LODO (12); final committee (13)). Compute nodes only (sbatch / jobs/sbx.sh): python never runs on the login node.

Runs (data split in brackets; suffix _al = + G-4 active-learning data, also set by --with-al)
  FT_<k>            k = 0..4   [fold<k>]   drop every structure with k in folds(tag)          (FT-2 / A evaluation)
  FT_all_s<i>       i = 0..2   [all_s<i>]  all data; validation split and MACE seed differ by i (deployment committee)
  FT_LODO_<d>       d = 0..4   [lodo<d>]   drop every structure with DIPOLES[d] in dipoles(tag)   (Phase X)
  FTpilot_<cand>_<k>           [fold<k>]   Z pilot (spec 4: fold-0 fine-tune of the top-2 Z candidates)
  BENCH_<cand>[_<tag>]         [fold0]     timing / smoke runs (--subset), never used downstream
Candidates: MACE-OMOL-0 (default; mace-torch 0.3.16 ships only the extra-large 1024-channel OMOL model),
  MACE-MH-1-omol (smaller multi-head MACE, omol head; optional size alternative), AIMNet2 (aimnet2_wb97m_d3_0),
  MACE-OFF23-medium / MACE-OFF24-medium (no Si: refused, see CAND_NOTES).

CLI (submit from analysis/mlts; exit 0 = done / up to date, 2 = not ready / incomplete (resubmit), 3 = STOP)
  data <run ...> | --all  [--with-al] [--fmax-cap 10] [--valid-frac 0.10] [--force]
        MLTENV=mlts or reactot, CPU: $MLT/ft/data/<split>/{train,valid[,test]}.xyz + meta.json,
        results/ft/<run>_data.json (counts by kind / source, C-2 leakage check, input sha256)
  fetch <cand>                     download the foundation into $MLT/cache (reuses ~/.cache/mace read-only)
  train <run> <cand> [--with-al] [--epochs 200] [--budget-h 12] [--stop-h 24] [--batch-size B] [--lr X]
        [--ew X] [--fw X] [--patience 6] [--subset N] [--allow-partial] [--force]
        MLTENV=mlts, 1 GPU: $MLT/ft/<run>/ (checkpoints, logs, E0s.json, budget.json, epochs.jsonl),
        $MLT/models/<run>.{model|jpt} + <run>.json (model card), results/ft/<run>_train.json
  ft1 <run> [--with-al] [--dtype float32] [--foundation <cand>]
        MLTENV=mlts, 1 GPU: FT-1 (record only) on valid (+ held-out test side) -> results/ft/<run>_ft1.json
  status                           one line per run

sbatch (from analysis/mlts; login: one-shot submits only; `squeue -u $USER -h | wc -l` first; L=$MLT/logs):
  JD=$(sbatch --parsable -p cpu1,cpu2 -J ft_data --time=48:00:00 -c 4 --mem=48G --output=$L/%x_%j.out \
       --dependency=afterok:<G collect job> --kill-on-invalid-dep=yes --export=ALL,MLTENV=mlts \
       jobs/mlt.slurm finetune.py data FTpilot_MACE-OMOL-0_0 FTpilot_AIMNet2_0)
  sbatch -p gpu1,gpu3,gpu4,gpu5 -J ft_p_omol --time=48:00:00 --gres=gpu:1 -c 8 --mem=96G --output=$L/%x_%j.out \
       --dependency=afterok:$JD --kill-on-invalid-dep=yes --export=ALL,MLTENV=mlts \
       jobs/mlt.slurm finetune.py train FTpilot_MACE-OMOL-0_0 MACE-OMOL-0 ::: finetune.py ft1 FTpilot_MACE-OMOL-0_0
  sbatch -p gpu1,gpu2,gpu3,gpu4,gpu5,gpu6 -J ft_p_aim --time=48:00:00 --gres=gpu:1 -c 8 --mem=64G \
       --output=$L/%x_%j.out --dependency=afterok:$JD --kill-on-invalid-dep=yes --export=ALL,MLTENV=mlts \
       jobs/mlt.slurm finetune.py train FTpilot_AIMNet2_0 AIMNet2 ::: finetune.py ft1 FTpilot_AIMNet2_0
  A 48 h wall / node loss: resubmit the same command (budget.json keeps the GPU-h, MACE --restart_latest, AIMNet2
  aim_last.pt). Exit 3 = STOP (> 24 GPU-h or C-2 leakage).

Units: extxyz REF_energy eV (absolute label level), REF_forces eV/A; training energy key E_fit = REF_energy -
sum_i ATOM_REF_EH[Z_i] * EH2EV (fixed per-element constants, only to keep float32 targets small; forces unchanged);
exported models predict absolute label-level energies (E0 += ATOM_REF) — see the model card.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import urllib.request  # noqa: E402
from collections import Counter, deque  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mlt_common as M  # noqa: E402

CODE_VERSION = "mlts-ft-1"
T_START = time.time()

# ---------------------------------------------------------------- paths
DFT = M.MLT / "dft"
FT = M.MLT / "ft"
FT_DATA = FT / "data"
MODELS = M.MLT / "models"
CACHE = M.MLT / "cache"
RES_FT = M.RES / "ft"
INPUTS = (("I", DFT / "existing.extxyz"), ("G1", DFT / "gen_G1.extxyz"), ("G2", DFT / "gen_G2.extxyz"),
          ("G3", DFT / "gen_G3.extxyz"))
AL_INPUT = ("G4", DFT / "gen_G4.extxyz")
G_REPORT = M.RES / "G_report.json"
SITE = Path(M.CONDA) / "envs" / "mlts" / "lib" / "python3.10" / "site-packages"
HOME_MACE_CACHE = Path.home() / ".cache" / "mace"

# ---------------------------------------------------------------- constants
EV2KCAL = M.EH2KCAL / M.EH2EV                      # 23.0605 kcal/mol per eV
ATOM_REF_EH = {1: -0.5, 6: -37.8, 7: -54.5, 8: -75.0, 9: -99.7, 14: -289.3, 16: -398.0, 17: -460.1, 35: -2573.9}
DEDUPE_DEC = 5                                     # positions rounded to 1e-5 A for the exact-duplicate key
FMAX_CAP = 10.0                                    # eV/A, per-structure max atomic |F| (decision FT-fcap)
VALID_FRAC = 0.10
TS_NEAR_ALWAYS = ("ts_near", "ts_imag", "ts_label")
TS_NEAR_IF_CLOSE = ("ts_path", "ts_final", "g2_path", "ts_final_g2", "al_ts")
TS_NEAR_DFORM_A = 0.3
FT1_TARGET = dict(force_mae_kcal_A=0.6, rel_e_mae_kcal=0.3)
BUDGET_H, STOP_H = 12.0, 24.0
ALL_RUNS = [f"FT_{k}" for k in range(5)] + [f"FT_all_s{i}" for i in range(3)] + [f"FT_LODO_{d}" for d in range(5)]
B3LYP_D3BJ = dict(s6=1.0, s8=1.9889, a1=0.3981, a2=4.4211)   # Grimme D3(BJ) B3LYP parameters

CANDIDATES = {
    "MACE-OMOL-0": dict(kind="mace", file="MACE-omol-0-extra-large-1024.model", head=None, si=True,
                        url="https://github.com/ACEsuit/mace-foundations/releases/download/mace_omol_0/"
                            "MACE-omol-0-extra-large-1024.model",
                        level="OMol25 wB97M-V/def2-TZVPD"),
    "MACE-MH-1-omol": dict(kind="mace", file="mace-mh-1.model", head="omol", si=True,
                           url="https://github.com/ACEsuit/mace-foundations/releases/download/mace_mh_1/mace-mh-1.model",
                           level="MACE-MH-1 omol head (OMol25)"),
    "MACE-OFF23-medium": dict(kind="mace", file="MACE-OFF23_medium.model", head=None, si=False,
                              url="https://raw.githubusercontent.com/ACEsuit/mace-off/main/mace_off23/MACE-OFF23_medium.model",
                              level="SPICE wB97M-D3(BJ)/def2-TZVPPD"),
    "MACE-OFF24-medium": dict(kind="mace", file="MACE-OFF24_medium.model", head=None, si=False,
                              url="https://github.com/ACEsuit/mace-off/raw/main/mace_off24/MACE-OFF24_medium.model",
                              level="SPICE 2 wB97M-D3(BJ)/def2-TZVPPD"),
    "AIMNet2": dict(kind="aimnet", file="aimnet2_wb97m_d3_0.jpt", head=None, si=True,
                    url="https://storage.googleapis.com/aimnetcentral/AIMNet2/aimnet2_wb97m_d3_0.jpt",
                    yaml="aimnet2_dftd3_wb97m.yaml", level="wB97M-D3/def2-TZVP"),
}
ALIASES = {"mace_omol": "MACE-OMOL-0", "omol": "MACE-OMOL-0", "mace-omol-0": "MACE-OMOL-0", "aimnet2": "AIMNet2",
           "mh-1": "MACE-MH-1-omol", "mace-mh-1-omol": "MACE-MH-1-omol", "off23": "MACE-OFF23-medium",
           "off24": "MACE-OFF24-medium"}
CAND_NOTES = {"no_si": "foundation has no Si; 465 D1 reactions carry SiMe3 and mace load_foundations_elements needs every "
                       "training element in the foundation z-table -> not fine-tunable on the D1 data (spec 4: Si-free "
                       "reference only)"}

DECISIONS = {
    "FT-split": "split = drop (C-2 rule on the tags, OR'ed with the targets() fold / dipole of every rxn_id — both must "
                "agree, a disagreement is counted) -> validation = whole fold_core groups ('<dipole>/<skeleton>', "
                "mlt_common.targets) drawn in a seeded random order until >= 10 % of the remaining D1 reactions; a "
                "structure is validation iff it has D1 rxn_ids and all of them are validation reactions (molecules "
                "shared with training reactions and untagged D0 data -> train). Seeds: SEED+100+k (fold k), SEED+200+i "
                "(all_s<i>), SEED+300+d (lodo d). test = dropped structures whose rxn_ids are all held out.",
    "FT-dedupe": "exact duplicates: same element sequence and positions rounded to 1e-5 A; the first occurrence (input "
                 "order I, G1, G2, G3, then G4) is kept and gets the union of the leakage tags; |dE|, |dF| recorded.",
    "FT-fcap": "per-structure max atomic |F| > 10 eV/A (0.19 Eh/bohr; early OptTS / LooseOpt steps of Phase I) is "
               "dropped; the measured distribution is in the split meta (fmax_eV_A).",
    "FT-frame": "training energy E_fit = REF_energy - sum ATOM_REF_EH[Z] x EH2EV (fixed, rounded per-element constants): "
                "float32 total energies of ~2-7e4 eV are quantised at 2-8 meV (0.05-0.18 kcal/mol, Br worst); the shift "
                "keeps |E_fit| ~ 1e2 eV (spacing ~1e-5 eV). Forces untouched. Export adds ATOM_REF back to the E0s, so "
                "exported models predict absolute label-level energies (eV).",
    "FT-E0": "atomic reference energies by least squares: E0(Z) = E0_foundation(Z) + lstsq(counts, E_fit - "
             "E_foundation) on the training structures (<= 40,000, all structures with a rare element kept) — the "
             "formula of mace.data.estimate_e0s_from_foundation (MACE --E0s=estimated); MACE --E0s=average (verified "
             "in mace/data/utils.compute_average_E0s: lstsq of total energies on atom counts) would double count the "
             "foundation's interaction energy because load_foundations_elements copies the foundation scale/shift. "
             "AIMNet2: the same residual least squares with the pretrained atomic shifts zeroed (= aimnet SAE, mode "
             "linreg).",
    "FT-mace": "naive fine-tuning (mace_run_train --foundation_model <local file> --multiheads_finetuning False: no "
               "OMol25 replay data on disk; the deployment domain is D1 only). Loss 'weighted' energy_weight 1, "
               "forces_weight 1 (MACE README fine-tuning example), EMA 0.99, amsgrad, rms_forces_scaling; lr 1e-3 "
               "(README naive-FT example 1e-2 is for MP-small with batch 2; multihead FT forces 1e-4; 1e-3 for the "
               "1024-channel model), batch 8 (RTX 3090 bench: batch 8 = batch 16 structures/s, 9.6 vs 15.6 GB, 1.3x "
               "batch 4), ReduceLROnPlateau factor 0.8 patience 3; float32 training (FP64 "
               "is ~1/64 rate on A6000 / 3090 / A10), export float64. mace 0.3.16 patience does not stop a "
               "non-distributed run (tools/train.py exit_now is None) -> early stopping (6 epochs without a better "
               "validation loss) and the 12 GPU-h budget are enforced here: terminate, then finalise with "
               "--restart_latest --max_num_epochs <best epoch> (MACE loads the best checkpoint (saved with EMA "
               "weights) and writes the model). The final MACE train+valid error table (a full forward/force pass "
               "over the training set, ~20 min) is skipped; FT-1 computes the metrics.",
    "FT-aimnet": "AIMNet2 with the aimnet package parts (model yaml aimnet2_dftd3_wb97m + weights of the released "
                 "aimnet2_wb97m_d3_0.jpt, SizeGroupedDataset / Sampler, MTLoss energy_loss_fn + peratom_loss_fn, "
                 "Forces, pt2jpt export helpers) and a loop here (aimnet train needs wandb for metrics and has no time "
                 "budget). D3 head switched to B3LYP-D3(BJ) (s8 1.9889, a1 0.3981, a2 4.4211) = the label "
                 "dispersion, so the network only learns the electronic difference. Loss weights energy 1 : forces "
                 "0.2 (aimnet default_train.yaml, charges term dropped: no label charges), RAdam lr 1e-4 (default 4e-4 "
                 "is from scratch), grad clip 0.4, ReduceLROnPlateau 0.75 / 3.",
    "FT-budget": "GPU-h = wall-h of the train command (1 GPU), summed over attempts (budget.json, updated each "
                 "minute). An epoch is not started when it would end past 12 h (target, spec 0-5); > 24 h = STOP "
                 "(exit 3).",
    "FT-ts-near": "TS-near (FT-1) = kind ts_near / ts_imag / ts_label, or kind ts_path / ts_final / g2_path / "
                  "ts_final_g2 / al_ts with dform_label_A <= 0.3 A (max over the two forming bonds of |d - d_label|, "
                  "label TS + formed_pairs_ts; computed here when the record has none).",
    "FT-relE": "within-reaction relative energy: structures of one reaction (single rxn_id) and one composition, "
               "E - mean(E) of the group (groups of >= 2), MAE over structures; primary on TS-near structures, also "
               "on all single-reaction structures. MACE uses interaction_energy (no float32 E0 quantisation).",
}


# ================================================================ helpers
def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _jobtag():
    return dict(host=socket.gethostname(), slurm_job=os.environ.get("SLURM_JOB_ID"),
                task=os.environ.get("SLURM_ARRAY_TASK_ID"), time=_now(), code=CODE_VERSION)


def _q(v, qs=(0.0, 0.1, 0.5, 0.9, 0.99, 1.0)):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if not len(v):
        return dict(n=0)
    out = {f"q{int(round(q * 100))}": float(np.quantile(v, q)) for q in qs}
    out.update(n=int(len(v)), mean=float(v.mean()))
    return out


def _sha_obj(o):
    return hashlib.sha256(json.dumps(o, sort_keys=True, default=str).encode()).hexdigest()


def _rj(p):
    try:
        return M.read_json(p)
    except Exception:                                                     # noqa: BLE001
        return None


def atom_ref_ev(numbers):
    return float(sum(ATOM_REF_EH[int(z)] for z in numbers)) * M.EH2EV


def canon_cand(c):
    if c in CANDIDATES:
        return c
    c2 = ALIASES.get(str(c).lower())
    if c2 is None:
        raise SystemExit(f"unknown candidate {c}; known: {sorted(CANDIDATES)}")
    return c2


def parse_run(run, with_al=False):
    r = str(run)
    al = r.endswith("_al")
    base = r[:-3] if al else r
    al = al or bool(with_al)
    cand, bench = None, False
    if m := re.fullmatch(r"FT_([0-4])", base):
        mode, val = "fold", int(m.group(1))
    elif m := re.fullmatch(r"FT_all_s(\d+)", base):
        mode, val = "all", int(m.group(1))
    elif m := re.fullmatch(r"FT_LODO_([0-4])", base):
        mode, val = "lodo", int(m.group(1))
    elif m := re.fullmatch(r"FTpilot_(.+)_([0-4])", base):
        mode, val, cand = "fold", int(m.group(2)), canon_cand(m.group(1))
    elif m := re.fullmatch(r"BENCH_([^_]+)(?:_(.+))?", base):
        mode, val, cand, bench = "fold", 0, canon_cand(m.group(1)), True
    else:
        raise SystemExit(f"unknown run name {run} (FT_<k>, FT_all_s<i>, FT_LODO_<d>, FTpilot_<cand>_<k>, BENCH_<cand>)")
    split = {"fold": f"fold{val}", "all": f"all_s{val}", "lodo": f"lodo{val}"}[mode] + ("_al" if al else "")
    seed_off = {"fold": 100, "all": 200, "lodo": 300}[mode] + val
    return dict(run=base + ("_al" if al else ""), base=base, mode=mode, val=val, cand=cand, split=split, al=al,
                bench=bench, valid_seed=M.SEED + seed_off, train_seed=M.SEED + (val if mode == "all" else 0))


def _wtorch(obj, path):
    import torch
    path = M.assert_write_ok(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    torch.save(obj, tmp)
    os.replace(tmp, path)
    return path


def _mkdir(p):
    M.assert_write_ok(p)
    Path(p).mkdir(parents=True, exist_ok=True)
    return Path(p)


def _du_gb(p):
    tot = 0
    for root, _d, files in os.walk(p):
        for f in files:
            try:
                tot += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return tot / 1e9


def _guard_imports():
    """the mlts env is a clone of reactot: torchvision 0.17.1 (built for torch 2.2) breaks `import mace` under torch 2.8
    ('operator torchvision::nms does not exist'). Nothing here needs torchvision; when its minor version does not match
    torch (0.<minor+15>), a None entry makes `import torchvision` raise ImportError (process-local, env unchanged)."""
    os.environ.setdefault("XDG_CACHE_HOME", str(CACHE))
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    try:
        from importlib.metadata import version
        tv, tt = version("torchvision"), version("torch")
    except Exception:                                                     # noqa: BLE001
        return "torchvision absent"
    if int(tv.split(".")[1]) == int(tt.split(".")[1]) + 15:
        return "torchvision ok"
    blocked = ("torchvision", "torchaudio")
    for b in blocked:
        sys.modules.setdefault(b, None)                  # importlib.util.find_spec -> None, import -> ImportError
    try:                                                 # torchmetrics decides from the dist metadata (RequirementCache)
        from lightning_utilities.core import imports as LI
        orig = LI.RequirementCache._check_available
        if not getattr(orig, "_mlts_guard", False):
            def _check_available(self):
                if hasattr(self, "available"):
                    return
                name = str(self.requirement or self.module or "").strip()
                if re.match(r"(torchvision|torchaudio)\b", name):
                    self.available, self.message = False, f"{name} blocked by finetune._guard_imports"
                    return
                orig(self)
            _check_available._mlts_guard = True
            LI.RequirementCache._check_available = _check_available
    except ImportError:
        pass
    return f"blocked torchvision {tv} / torchaudio (torch {tt}; process-local)"


# ================================================================ data
def input_files(with_al):
    return list(INPUTS) + ([AL_INPUT] if with_al else [])


def inputs_state(with_al):
    st = {}
    for name, p in input_files(with_al):
        if p.is_file() and p.stat().st_size > 0:
            st[name] = dict(path=str(p), present=True, sha256=M.sha256(p), bytes=p.stat().st_size)
        else:
            st[name] = dict(path=str(p), present=False)
    g = _rj(G_REPORT) or {}
    complete = all(st[n]["present"] for n, _ in input_files(with_al)) and bool(g.get("complete"))
    return st, complete, dict(G_report=str(G_REPORT), G_report_complete=g.get("complete"))


def split_fingerprint(split, st, params):
    used = {n: v.get("sha256") for n, v in st.items() if v["present"] and (n != "G4" or split.endswith("_al"))}
    return _sha_obj(dict(split=split, inputs=used, params=params))


class Pool:
    """every label-level structure of the inputs after dedupe, level / fmax / electron-count filters (not yet split)."""

    def __init__(self, with_al, fmax_cap):
        import pandas as pd
        T = M.targets()
        self.T = T
        Ti = T.set_index("rxn_id")
        self.rfold = Ti.fold_T1.astype(int).to_dict()
        self.rdip = Ti.dipole.to_dict()
        self.rcore = Ti.fold_core.to_dict()
        self.rpairs = Ti.formed_pairs_ts.to_dict()
        self.atoms, rows = [], []
        self.read_info = {}
        for name, p in input_files(with_al):
            if not (p.is_file() and p.stat().st_size > 0):
                continue
            t0 = time.time()
            al = M.read_extxyz(p)
            self.read_info[name] = dict(n=len(al), read_s=round(time.time() - t0, 1))
            print(f"[data] read {name} {len(al)} structures ({time.time() - t0:.0f} s)", flush=True)
            for a in al:
                rows.append(self._row(a, name))
                rows[-1]["i"] = len(self.atoms)
                self.atoms.append(a)
        self.n_read = len(rows)
        self.stats = {}
        self._fill_dform(rows)
        rows = self._dedupe(rows)
        P = pd.DataFrame(rows)
        P = self._filters(P, fmax_cap)
        self._tag_check(P)
        self.P = P.reset_index(drop=True)

    @staticmethod
    def _tags(s):
        return [x for x in str(s or "").split(",") if x]

    def _row(self, a, inp):
        f = np.asarray(a.arrays["REF_forces"], float)
        fn = np.linalg.norm(f, axis=1)
        rx = tuple(sorted({int(x) for x in self._tags(a.info.get("rxn_ids"))}))
        e_eh = float(a.info.get("e_eh", float(a.info["REF_energy"]) / M.EH2EV))
        dform = a.info.get("dform_label_A")
        return dict(struct_id=str(a.info.get("struct_id")), kind=str(a.info.get("kind")), source=str(a.info.get("source")),
                    input=inp, level=str(a.info.get("level", "label")), n_atoms=len(a),
                    formula=a.get_chemical_formula(mode="hill"), rxn=rx,
                    folds=frozenset(self._tags(a.info.get("folds"))),
                    dipoles=frozenset(self._tags(a.info.get("dipoles"))),
                    fmax=float(fn.max()) if len(fn) else 0.0, e_eh=e_eh,
                    e_ref_check=float(a.info["REF_energy"]) - e_eh * M.EH2EV,
                    charge=int(a.info.get("charge", 0) or 0), nelec=int(np.asarray(a.numbers).sum()),
                    dform=float(dform) if dform is not None else float("nan"),
                    finite=bool(np.isfinite(f).all() and np.isfinite(e_eh)))

    def _fill_dform(self, rows):
        kinds = set(TS_NEAR_ALWAYS) | set(TS_NEAR_IF_CLOSE)
        need = [r for r in rows if r["kind"] in kinds and not np.isfinite(r["dform"]) and len(r["rxn"]) == 1]
        cache, n_ok, n_bad = {}, 0, 0
        for r in need:
            rid = r["rxn"][0]
            if rid not in self.rpairs:
                n_bad += 1
                continue
            if rid not in cache:
                sym, pos = M.read_xyz(M.label_ts(rid))
                prs = M.formed_pairs(self.rpairs[rid])
                pos = np.asarray(pos, float)
                cache[rid] = (sym, [float(np.linalg.norm(pos[i] - pos[j])) for i, j in prs], prs)
            sym, dl, prs = cache[rid]
            a = self.atoms[r["i"]]
            if list(a.get_chemical_symbols()) != list(sym):
                n_bad += 1
                continue
            d = [float(np.linalg.norm(a.positions[i] - a.positions[j])) for i, j in prs]
            r["dform"] = float(max(abs(x - y) for x, y in zip(d, dl)))
            n_ok += 1
        self.stats["dform_filled"] = dict(n_computed=n_ok, n_not_label_order_or_unknown=n_bad,
                                          by_kind=dict(Counter(r["kind"] for r in need)))

    def _dedupe(self, rows):
        def key(a):
            h = hashlib.sha1((" ".join(a.get_chemical_symbols()) + "|").encode())
            h.update((np.round(np.asarray(a.positions, float), DEDUPE_DEC) + 0.0).tobytes())
            return h.hexdigest()

        first, kept, pairs, de, dfm, ndrop = {}, [], Counter(), [], [], 0
        # AL data is deduplicated after the base inputs: a base split never depends on G4
        for r in [r for r in rows if r["input"] != "G4"] + [r for r in rows if r["input"] == "G4"]:
            k = key(self.atoms[r["i"]])
            if k not in first:
                first[k] = r
                kept.append(r)
                continue
            j = first[k]
            ndrop += 1
            pairs[f"{j['kind']}<-{r['kind']}"] += 1
            de.append(abs(j["e_eh"] - r["e_eh"]) * M.EH2KCAL)
            dfm.append(float(np.abs(self.atoms[j["i"]].arrays["REF_forces"]
                                    - self.atoms[r["i"]].arrays["REF_forces"]).max()))
            if r["input"] != "G4" or j["input"] == "G4":
                j["rxn"] = tuple(sorted(set(j["rxn"]) | set(r["rxn"])))
                j["folds"] = j["folds"] | r["folds"]
                j["dipoles"] = j["dipoles"] | r["dipoles"]
        kept.sort(key=lambda r: r["i"])
        self.stats["dedupe"] = dict(method=DECISIONS["FT-dedupe"], n_read=len(rows), n_dropped=ndrop,
                                    pairs=dict(pairs), de_kcal=_q(de), dfmax_eV_A=_q(dfm))
        return kept

    def _filters(self, P, fmax_cap):
        st = {}
        bad_level = P.level != "label"
        st["not_label_level"] = int(bad_level.sum())
        bad_fin = ~P.finite
        st["non_finite"] = int(bad_fin.sum())
        odd = ((P.nelec - P.charge) % 2) == 1
        st["odd_electron"] = int(odd.sum())
        fm = P.fmax.to_numpy()
        st["fmax_eV_A"] = _q(fm, (0.0, 0.5, 0.9, 0.99, 0.999, 1.0))
        st["fmax_above"] = {str(t): int((fm > t).sum()) for t in (1.0, 2.0, 5.0, 10.0, 20.0)}
        st["fmax_above_by_kind"] = {str(t): dict(Counter(P.kind[fm > t])) for t in (5.0, 10.0)}
        top = P.sort_values("fmax", ascending=False).head(20)
        st["fmax_top20"] = [dict(struct_id=r.struct_id, kind=r.kind, fmax_eV_A=round(r.fmax, 3)) for r in top.itertuples()]
        cap = P.fmax > fmax_cap if fmax_cap and fmax_cap > 0 else P.fmax < -1
        st["fmax_cap_eV_A"] = fmax_cap
        st["dropped_fmax_cap"] = int(cap.sum())
        st["dropped_fmax_cap_by_kind"] = dict(Counter(P.kind[cap]))
        st["fmax_eh_bohr_equiv_cap"] = (fmax_cap or 0) / M.EHBOHR2EVA
        e_fit = np.array([P.e_eh.iat[n] * M.EH2EV - atom_ref_ev(self.atoms[int(P.i.iat[n])].numbers)
                          for n in range(len(P))])
        P["e_fit"] = e_fit
        st["e_fit_eV"] = _q(e_fit, (0.0, 0.5, 1.0))
        st["e_ref_check_max_abs_eV"] = float(np.abs(P.e_ref_check).max()) if len(P) else 0.0
        if st["e_ref_check_max_abs_eV"] > 1e-4:
            M.stop(f"REF_energy != e_eh x EH2EV (max {st['e_ref_check_max_abs_eV']:.2e} eV)")
        if len(P) and np.abs(e_fit).max() > 5000:
            M.stop(f"|E_fit| up to {np.abs(e_fit).max():.0f} eV: ATOM_REF_EH constants wrong")
        drop = bad_level | bad_fin | odd | cap
        st["kept"] = int((~drop).sum())
        self.stats["filters"] = st
        return P[~drop]

    def _tag_check(self, P):
        nf, nd, unknown = 0, 0, 0
        for r in P.itertuples():
            if not r.rxn:
                if r.folds or r.dipoles:
                    nf += 1
                continue
            if any(x not in self.rfold for x in r.rxn):
                unknown += 1
                continue
            if {str(self.rfold[x]) for x in r.rxn} != set(r.folds):
                nf += 1
            if {self.rdip[x] for x in r.rxn} != set(r.dipoles):
                nd += 1
        self.stats["tag_check"] = dict(fold_tag_mismatch=nf, dipole_tag_mismatch=nd, unknown_rxn_ids=unknown,
                                       note="drop rule = tag rule OR targets() rule, so a mismatch cannot leak")


def build_split(pool, split_name, R, valid_frac, fp, st, complete, gmeta, params):
    P, T = pool.P, pool.T
    if not R["al"]:
        P = P[P.input != "G4"]
    mode, val = R["mode"], R["val"]
    if mode == "fold":
        held = set(T.rxn_id[T.fold_T1 == val].astype(int))
        tag_hit = P.folds.map(lambda s: str(val) in s).astype(bool)
    elif mode == "lodo":
        dn = M.DIPOLES[val]
        held = set(T.rxn_id[T.dipole == dn].astype(int))
        tag_hit = P.dipoles.map(lambda s: dn in s).astype(bool)
    else:
        held = set()
        tag_hit = P.folds.map(lambda s: False).astype(bool)
    rx_hit = P.rxn.map(lambda r: any(x in held for x in r)).astype(bool)
    drop = tag_hit | rx_hit
    keep, dropped = P[~drop], P[drop]
    test = dropped[dropped.rxn.map(lambda r: len(r) > 0 and all(x in held for x in r)).astype(bool)]
    # validation: whole cores, seeded order, until >= valid_frac of the remaining reactions
    Rr = sorted({x for r in keep.rxn for x in r if x in pool.rcore})
    ncore = Counter(pool.rcore[x] for x in Rr)
    cores = sorted(ncore)
    rng = np.random.default_rng(R["valid_seed"])
    target = int(math.ceil(valid_frac * len(Rr)))
    chosen, n = set(), 0
    for i in rng.permutation(len(cores)):
        if n >= target:
            break
        chosen.add(cores[i])
        n += ncore[cores[i]]
    V = {x for x in Rr if pool.rcore[x] in chosen}
    is_val = keep.rxn.map(lambda r: len(r) > 0 and all(x in V for x in r)).astype(bool)
    train, valid = keep[~is_val], keep[is_val]
    if not len(valid) or not len(train):
        M.not_ready(f"{split_name}: empty train ({len(train)}) or validation ({len(valid)}) set")
    # C-2 leakage check (independent of the drop code: targets() folds / dipoles of every used rxn_id)
    used = {x for r in keep.rxn for x in r}
    if mode == "fold":
        bad_t = {x for x in used if pool.rfold.get(x) == val}
        bad_tag = int(keep.folds.map(lambda s: str(val) in s).sum())
    elif mode == "lodo":
        bad_t = {x for x in used if pool.rdip.get(x) == M.DIPOLES[val]}
        bad_tag = int(keep.dipoles.map(lambda s: M.DIPOLES[val] in s).sum())
    else:
        bad_t, bad_tag = set(), 0
    leak = dict(rule="FT_k: drop iff k in folds(tag); FT_LODO_d: drop iff d in dipoles(tag); OR targets() of rxn_ids",
                n_heldout_rxns=len(held), n_used_rxns=len(used), used_cap_heldout=len(used & held),
                used_with_heldout_fold_or_dipole=len(bad_t), kept_with_heldout_tag=bad_tag,
                dropped_by_tag=int(tag_hit.sum()), dropped_by_targets=int(rx_hit.sum()),
                tag_vs_targets_disagree=int((tag_hit != rx_hit).sum()))
    leak["pass_"] = leak["used_cap_heldout"] == 0 and len(bad_t) == 0 and bad_tag == 0
    if not leak["pass_"]:
        M.stop(f"C-2 leakage in split {split_name}: {leak}")

    def counts(D):
        return dict(n=int(len(D)), by_kind=dict(Counter(D.kind)), by_source=dict(Counter(D.source)),
                    by_input=dict(Counter(D.input)), n_rxns=len({x for r in D.rxn for x in r}),
                    n_atoms=_q(D.n_atoms, (0.0, 0.5, 1.0)), n_ts_near=int(ts_near_mask(D).sum()))

    sd = _mkdir(FT_DATA / split_name)
    files = {}
    for nm, D in (("train", train), ("valid", valid), ("test", test)):
        if nm == "test" and mode == "all":
            continue
        t0 = time.time()
        M.write_extxyz(sd / f"{nm}.xyz", [export_atoms(pool.atoms[int(r.i)], r) for r in D.itertuples()])
        files[nm] = dict(path=str(sd / f"{nm}.xyz"), n=int(len(D)), bytes=(sd / f"{nm}.xyz").stat().st_size,
                         write_s=round(time.time() - t0, 1))
    meta = dict(split=split_name, mode=mode, val=val, al=R["al"], fingerprint=fp, inputs=st, inputs_complete=complete,
                g_report=gmeta, params=params, files=files, valid_seed=R["valid_seed"], valid_frac_target=valid_frac,
                valid=dict(n_cores=len(chosen), n_rxns=len(V), frac_rxns=len(V) / max(1, len(Rr)),
                           n_cores_total=len(cores), n_rxns_total=len(Rr)),
                counts=dict(train=counts(train), valid=counts(valid), test=counts(test)), leakage_C2=leak,
                pool=dict(read=pool.read_info, n_read=pool.n_read, **pool.stats), atom_ref_eh=ATOM_REF_EH,
                energy_keys=dict(REF_energy="eV absolute label level", E_fit="eV, REF_energy - sum ATOM_REF_EH x EH2EV "
                                 "(training key)", REF_forces="eV/A"),
                decisions={k: DECISIONS[k] for k in ("FT-split", "FT-dedupe", "FT-fcap", "FT-frame", "FT-ts-near")},
                written=_now(), **_jobtag())
    M.write_json(sd / "meta.json", meta)
    print(f"[data] {split_name}: train {len(train)} valid {len(valid)} test {len(test)} | valid rxns {len(V)}/"
          f"{len(Rr)} | leak pass {leak['pass_']}", flush=True)
    return meta


def export_atoms(a, r):
    from ase import Atoms
    b = Atoms(numbers=a.numbers, positions=a.positions)
    b.arrays["REF_forces"] = np.asarray(a.arrays["REF_forces"], float)
    b.info.update(struct_id=r.struct_id, kind=r.kind, source=r.source, rxn_ids=M.tag_list(r.rxn),
                  folds=M.tag_list(r.folds), dipoles=M.tag_list(r.dipoles), e_eh=float(r.e_eh),
                  REF_energy=float(r.e_eh) * M.EH2EV, E_fit=float(r.e_fit), total_charge=int(r.charge), total_spin=1,
                  config_type=r.kind, input=r.input)
    if np.isfinite(r.dform):
        b.info["dform_label_A"] = float(r.dform)
    return b


def ts_near_mask(D):
    kind = D.kind if hasattr(D, "kind") else D["kind"]
    dform = D.dform if hasattr(D, "dform") else D["dform"]
    return kind.isin(TS_NEAR_ALWAYS) | (kind.isin(TS_NEAR_IF_CLOSE) & (dform <= TS_NEAR_DFORM_A))


def cmd_data(runs, all_, with_al, fmax_cap, valid_frac, force):
    Rs = [parse_run(r, with_al) for r in runs] + ([parse_run(r, with_al) for r in ALL_RUNS] if all_ else [])
    if not Rs:
        raise SystemExit("data: give run names or --all")
    any_al = any(R["al"] for R in Rs)
    st, complete, gmeta = inputs_state(any_al)
    if any_al and not st["G4"]["present"]:
        M.not_ready(f"--with-al: {AL_INPUT[1]} missing")
    params = dict(code=CODE_VERSION, fmax_cap=fmax_cap, valid_frac=valid_frac, dedupe_decimals=DEDUPE_DEC,
                  atom_ref_eh=ATOM_REF_EH, ts_near=[TS_NEAR_ALWAYS, TS_NEAR_IF_CLOSE, TS_NEAR_DFORM_A])
    splits = {}
    for R in Rs:
        splits.setdefault(R["split"], R)
    todo = []
    for s, R in splits.items():
        fp = split_fingerprint(s, st, params)
        meta = _rj(FT_DATA / s / "meta.json")
        ok = meta and meta.get("fingerprint") == fp and all(Path(f["path"]).is_file() for f in meta["files"].values())
        if force or not ok:
            todo.append((s, R, fp))
    metas = {}
    if todo:
        pool = Pool(any_al, fmax_cap)
        for s, R, fp in todo:
            metas[s] = build_split(pool, s, R, valid_frac, fp, st, complete, gmeta, params)
    for R in Rs:
        meta = metas.get(R["split"]) or M.read_json(FT_DATA / R["split"] / "meta.json")
        out = dict(run=R["run"], split=R["split"], split_dir=str(FT_DATA / R["split"]), run_info=R,
                   fingerprint=meta["fingerprint"], inputs_complete=meta["inputs_complete"],
                   counts=meta["counts"], valid=meta["valid"], leakage_C2=meta["leakage_C2"],
                   inputs={k: dict(present=v["present"], sha256=v.get("sha256")) for k, v in meta["inputs"].items()},
                   filters=meta["pool"].get("filters"), dedupe=meta["pool"].get("dedupe"),
                   tag_check=meta["pool"].get("tag_check"), written=_now(), **_jobtag())
        M.write_json(RES_FT / f"{R['run']}_data.json", out)
    print(json.dumps(dict(inputs_complete=complete, runs={R["run"]: dict(split=R["split"], built=R["split"] in metas)
                                                           for R in Rs})), flush=True)
    return 0


# ================================================================ foundations
def fetch(cand):
    """local path of the foundation file: ~/.cache/mace (read only) or $MLT/cache/<kind>/ (downloaded here)."""
    C = CANDIDATES[cand]
    if C["kind"] == "mace":
        alt = [HOME_MACE_CACHE / C["file"], HOME_MACE_CACHE / "".join(c for c in C["file"] if c.isalnum() or c == "_")]
        dst = CACHE / "mace" / C["file"]
    else:
        alt = [SITE / "aimnet" / "calculators" / "assets" / C["file"]]
        dst = CACHE / "aimnet" / C["file"]
    for p in alt + [dst]:
        if p.is_file() and p.stat().st_size > 1000:
            return p
    _mkdir(dst.parent)
    tmp = dst.with_name(f".{dst.name}.part{os.getpid()}")
    t0 = time.time()
    print(f"[fetch] {C['url']} -> {dst}", flush=True)
    with urllib.request.urlopen(C["url"], timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    if tmp.stat().st_size < 1000:
        raise SystemExit(f"download of {C['url']} failed ({tmp.stat().st_size} bytes)")
    os.replace(tmp, dst)
    print(f"[fetch] {dst.stat().st_size / 1e6:.1f} MB in {time.time() - t0:.0f} s", flush=True)
    return dst


def cmd_fetch(cand):
    cand = canon_cand(cand)
    p = fetch(cand)
    info = dict(cand=cand, path=str(p), bytes=p.stat().st_size, sha256=M.sha256(p))
    if CANDIDATES[cand]["kind"] == "mace":
        _guard_imports()
        import torch
        m = torch.load(p, map_location="cpu", weights_only=False)
        info.update(model_class=type(m).__name__, heads=list(getattr(m, "heads", []) or []),
                    elements=[int(z) for z in m.atomic_numbers], r_max=float(m.r_max),
                    n_params=int(sum(x.numel() for x in m.parameters())), dtype=str(next(m.parameters()).dtype),
                    hidden_irreps=str(m.products[0].linear.irreps_out), n_interactions=int(m.num_interactions))
        miss = sorted(set(M.ELEMENTS) - {_sym(z) for z in info["elements"]})
        info["missing_d1_elements"] = miss
    print(json.dumps(info, indent=1), flush=True)
    M.write_json(RES_FT / f"foundation_{cand}.json", dict(info, written=_now(), **_jobtag()))
    return 0


def _sym(z):
    from ase.data import chemical_symbols
    return chemical_symbols[int(z)]


# ================================================================ MACE prediction (batched)
def mace_predict(model, atoms_list, device, batch_size=16, forces=True, head=None, chunk=2000):
    """per structure: energy (model dtype, eV), interaction_energy (eV), forces (eV/A) — as eval_configs.py."""
    import torch
    from mace import data as mdata
    from mace.tools import torch_geometric
    from mace.tools.utils import AtomicNumberTable
    heads = list(getattr(model, "heads", None) or ["Default"])
    head = head or heads[0]
    torch.set_default_dtype(next(model.parameters()).dtype)
    zt = AtomicNumberTable([int(z) for z in model.atomic_numbers])
    ks = mdata.KeySpecification(info_keys={"total_charge": "total_charge", "total_spin": "total_spin"}, arrays_keys={})
    rmax = float(model.r_max)
    E, I, F = [], [], []
    for p in model.parameters():
        p.requires_grad_(False)
    for s in range(0, len(atoms_list), chunk):
        cf = [mdata.config_from_atoms(a, key_specification=ks, head_name=head) for a in atoms_list[s:s + chunk]]
        ds = [mdata.AtomicData.from_config(c, z_table=zt, cutoff=rmax, heads=heads) for c in cf]
        dl = torch_geometric.dataloader.DataLoader(dataset=ds, batch_size=batch_size, shuffle=False, drop_last=False)
        for b in dl:
            b = b.to(device)
            if forces:
                o = model(b.to_dict(), compute_force=True, training=False)
                fb = o["forces"].detach().cpu().numpy()
                F.extend(np.split(fb, b.ptr[1:-1].cpu().numpy()))
            else:
                with torch.no_grad():
                    o = model(b.to_dict(), compute_force=False, training=False)
            E.append(o["energy"].detach().double().cpu().numpy())
            I.append(o["interaction_energy"].detach().double().cpu().numpy())
    return np.concatenate(E), np.concatenate(I), (F if forces else None)


def mace_head_e0(model, head=None):
    import torch
    ae = torch.atleast_2d(model.atomic_energies_fn.atomic_energies).double().cpu().numpy()
    heads = list(getattr(model, "heads", None) or ["Default"])
    hi = heads.index(head) if head in heads else 0
    return {int(z): float(ae[hi, j]) for j, z in enumerate(model.atomic_numbers)}


def lstsq_e0(atoms_list, resid, base=None):
    zs = sorted({int(z) for a in atoms_list for z in a.numbers})
    A = np.array([[int((a.numbers == z).sum()) for z in zs] for a in atoms_list], float)
    c, res, rank, _sv = np.linalg.lstsq(A, np.asarray(resid, float), rcond=None)
    fit = A @ c - resid
    out = {z: float(c[j]) + (base[z] if base else 0.0) for j, z in enumerate(zs)}
    return out, dict(n=len(atoms_list), rank=int(rank), n_elements=len(zs),
                     resid_after_fit_kcal=_q(np.abs(fit) * EV2KCAL, (0.5, 0.9, 1.0)),
                     resid_before_fit_kcal=_q(np.abs(np.asarray(resid) - np.mean(resid)) * EV2KCAL, (0.5, 0.9, 1.0)))


def e0_sample(atoms_list, nmax=40000, seed=M.SEED):
    if len(atoms_list) <= nmax:
        return list(range(len(atoms_list)))
    cnt = Counter(int(z) for a in atoms_list for z in set(a.numbers.tolist()))
    rare = {z for z, n in cnt.items() if n < 2000}
    keep = [i for i, a in enumerate(atoms_list) if rare & set(a.numbers.tolist())]
    rest = sorted(set(range(len(atoms_list))) - set(keep))
    rng = np.random.default_rng(seed)
    keep += list(rng.choice(rest, size=max(0, nmax - len(keep)), replace=False))
    return sorted(keep)


# ================================================================ budget / process control
class Budget:
    def __init__(self, rd, budget_h, stop_h):
        self.path = rd / "budget.json"
        self.d = _rj(self.path) or dict(attempts=[])
        self.prev_h = float(sum(a.get("wall_h", 0.0) for a in self.d["attempts"]))
        self.t0, self.last = time.time(), 0.0
        self.budget_h, self.stop_h = budget_h, stop_h
        self.d["attempts"].append(dict(start=_now(), wall_h=0.0, **_jobtag()))
        self.tick(True)

    def now_h(self):
        return self.prev_h + (time.time() - self.t0) / 3600.0

    def tick(self, force=False, note=None):
        if force or time.time() - self.last > 60:
            self.d["attempts"][-1]["wall_h"] = (time.time() - self.t0) / 3600.0
            if note:
                self.d["attempts"][-1]["note"] = note
            self.d["total_h"] = self.now_h()
            M.write_json(self.path, self.d)
            self.last = time.time()


class Proc:
    """subprocess with stdout teed to a log; '[FTEPOCH] {json}' lines collected."""

    def __init__(self, cmd, log, env):
        self.p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                  env=env, cwd=str(HERE))
        self.log, self.epochs, self.tail = log, [], deque(maxlen=80)
        self.th = threading.Thread(target=self._read, daemon=True)
        self.th.start()

    def _read(self):
        with open(M.assert_write_ok(self.log), "a") as f:
            for line in self.p.stdout:
                f.write(line)
                f.flush()
                self.tail.append(line.rstrip())
                if line.startswith("[FTEPOCH] "):
                    try:
                        rec = json.loads(line[10:])
                        rec["t_wall"] = time.time()
                        self.epochs.append(rec)
                    except ValueError:
                        pass

    def stop(self, grace=90):
        if self.p.poll() is None:
            self.p.terminate()
            try:
                self.p.wait(grace)
            except subprocess.TimeoutExpired:
                self.p.kill()
                self.p.wait()
        self.th.join(10)


def epoch_stats(hist):
    ep = [h for h in hist if h.get("epoch") is not None]
    if not ep:
        return dict(n=0)
    best = min(ep, key=lambda h: h["loss"])
    dts = [b["t_wall"] - a["t_wall"] for a, b in zip(hist, hist[1:]) if b.get("epoch") is not None
           and a.get("attempt") == b.get("attempt")]
    return dict(n=len(ep), last=ep[-1]["epoch"], best_epoch=best["epoch"], best=best,
                since_best=sum(1 for h in ep if h["epoch"] > best["epoch"]),
                epoch_s=_q(dts, (0.0, 0.5, 1.0)), epoch_s_median=float(np.median(dts)) if dts else None)


# ================================================================ train (MACE)
def mace_cmd_args(R, C, rd, fpath, train_file, valid_file, e0, o, max_epochs):
    e0s = "{" + ", ".join(f"{z}: {v!r}" for z, v in sorted(e0.items())) + "}"
    a = ["--name", R["run"], "--seed", str(R["train_seed"]), "--work_dir", str(rd), "--model_dir", str(rd / "mace_model"),
         "--checkpoints_dir", str(rd / "checkpoints"), "--results_dir", str(rd / "results"),
         "--log_dir", str(rd / "logs"), "--downloads_dir", str(rd / "downloads"),
         "--foundation_model", str(fpath), "--multiheads_finetuning", "False",
         "--train_file", str(train_file), "--valid_file", str(valid_file),
         "--energy_key", "E_fit", "--forces_key", "REF_forces", "--E0s", e0s,
         "--loss", "weighted", "--energy_weight", str(o.ew), "--forces_weight", str(o.fw),
         "--lr", str(o.lr), "--scaling", "rms_forces_scaling",
         "--batch_size", str(o.batch_size), "--valid_batch_size", str(2 * o.batch_size),
         "--max_num_epochs", str(int(max_epochs)), "--patience", "100000",
         "--ema", "--ema_decay", "0.99", "--amsgrad",
         "--scheduler", "ReduceLROnPlateau", "--lr_factor", "0.8", "--scheduler_patience", "3",
         "--default_dtype", "float32", "--device", "cuda", "--error_table", "PerAtomMAE", "--eval_interval", "1",
         "--restart_latest", "--save_cpu", "--clip_grad", "10", "--num_workers", str(o.num_workers),
         "--pin_memory", "True", "--plot", "False", "--enable_cueq", "False"]
    if C.get("head"):
        a += ["--foundation_head", C["head"]]
    return a


def ckpt_epochs(rd, tag):
    out = {}
    d = rd / "checkpoints"
    if d.is_dir():
        for p in d.iterdir():
            m = re.fullmatch(re.escape(tag) + r"_epoch-(\d+)\.pt", p.name)
            if m:
                out[int(m.group(1))] = p
    return out


def subset_files(R, rd, sd, n, seed):
    trn, val = rd / "sub_train.xyz", rd / "sub_valid.xyz"
    if not (trn.is_file() and val.is_file()):
        tr = M.read_extxyz(sd / "train.xyz")
        va = M.read_extxyz(sd / "valid.xyz")
        rng = np.random.default_rng(seed)
        it = sorted(rng.choice(len(tr), size=min(n, len(tr)), replace=False))
        iv = sorted(rng.choice(len(va), size=min(max(n // 5, 20), len(va)), replace=False))
        M.write_extxyz(trn, [tr[i] for i in it])
        M.write_extxyz(val, [va[i] for i in iv])
    return trn, val


def train_mace(R, cand, C, sd, meta, rd, o):
    import torch
    guard = _guard_imports()
    out_json = RES_FT / f"{R['run']}_train.json"
    model_out = model_path(R, C)
    bud = Budget(rd, o.budget_h, o.stop_h)
    if bud.prev_h >= o.stop_h:
        _write_train(out_json, dict(run=R["run"], cand=cand, status="STOP", gpu_h=bud.prev_h))
        M.stop(f"{R['run']}: {bud.prev_h:.1f} GPU-h already used (> {o.stop_h})")
    fpath = fetch(cand)
    if o.subset:
        train_file, valid_file = subset_files(R, rd, sd, o.subset, R["valid_seed"])
    else:
        train_file, valid_file = sd / "train.xyz", sd / "valid.xyz"
    dev = "cuda"
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    # ---- E0s (least squares on the foundation residual)
    e0p = rd / "E0s.json"
    e0j = _rj(e0p)
    key = dict(fp=meta["fingerprint"], cand=cand, subset=o.subset, code=CODE_VERSION)
    if not e0j or e0j.get("key") != key:
        t0 = time.time()
        found = torch.load(fpath, map_location="cpu", weights_only=False).float().to(dev)   # e3nn consts: .to()
        tr = M.read_extxyz(train_file)
        idx = e0_sample(tr)
        sub = [tr[i] for i in idx]
        e_pred, _i, _f = mace_predict(found, sub, dev, batch_size=max(8, 2 * o.batch_size), forces=False,
                                      head=C.get("head"))
        e_fit = np.array([a.info["E_fit"] for a in sub], float)
        base = mace_head_e0(found, C.get("head"))
        e0, fitst = lstsq_e0(sub, e_fit - e_pred, base=base)
        e0j = dict(key=key, E0s_eV=e0, foundation_E0s_eV={z: base[z] for z in e0}, fit=fitst, n_sample=len(sub),
                   seconds=round(time.time() - t0, 1), method=DECISIONS["FT-E0"], frame="E_fit (eV)")
        M.write_json(e0p, e0j)
        del found
        torch.cuda.empty_cache()
    e0 = {int(z): float(v) for z, v in e0j["E0s_eV"].items()}
    # ---- training phases
    tag = f"{R['run']}_run-{R['train_seed']}"
    hist_p = rd / "epochs.jsonl"
    hist = [json.loads(x) for x in hist_p.read_text().splitlines()] if hist_p.is_file() else []
    attempt = len(bud.d["attempts"])
    env = dict(os.environ, XDG_CACHE_HOME=str(CACHE), PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    mace_model = rd / "mace_model" / f"{R['run']}.model"
    reason, rc = None, None
    if not mace_model.is_file():
        args = mace_cmd_args(R, C, rd, fpath, train_file, valid_file, e0, o, o.epochs)
        cmd = [sys.executable, str(HERE / "finetune.py"), "_mace", *args]
        M.write_json(rd / "mace_cmd.json", dict(cmd=cmd, written=_now()))
        pr = Proc(cmd, rd / f"mace_attempt{attempt}.log", env)
        seen = 0
        while True:
            rc = pr.p.poll()
            bud.tick()
            new = pr.epochs[seen:]
            seen += len(new)
            for rec in new:
                rec["attempt"] = attempt
                hist.append(rec)
                with open(M.assert_write_ok(hist_p), "a") as f:
                    f.write(json.dumps(rec) + "\n")
                es = epoch_stats(hist)
                if rec.get("epoch") is not None:
                    print(f"[train] epoch {rec['epoch']} loss {rec['loss']:.5g} MAE_F {rec.get('mae_f', float('nan')) * 1e3:.2f} "
                          f"meV/A MAE_E {rec.get('mae_e_per_atom', float('nan')) * 1e3:.3f} meV/atom | best "
                          f"{es['best_epoch']} | {bud.now_h():.2f} GPU-h | mem {rec.get('max_mem_gb')} GB", flush=True)
                    ep_h = (es.get("epoch_s_median") or 0) / 3600.0
                    if es["since_best"] >= o.patience:
                        reason = f"early stop: {es['since_best']} epochs without improvement"
                    elif bud.now_h() + 1.3 * ep_h > o.budget_h:
                        reason = f"budget: next epoch would end past {o.budget_h} GPU-h"
            if rc is not None:
                break
            if reason is None and bud.now_h() > o.budget_h:
                reason = f"budget: {bud.now_h():.2f} > {o.budget_h} GPU-h mid-epoch"
            if bud.now_h() > o.stop_h:
                pr.stop()
                bud.tick(True, note="STOP > stop_h")
                _write_train(out_json, dict(run=R["run"], cand=cand, status="STOP", gpu_h=bud.now_h()))
                M.stop(f"{R['run']}: > {o.stop_h} GPU-h")
            if reason:
                print(f"[train] {reason} -> terminate + finalise", flush=True)
                pr.stop()
                rc = pr.p.returncode
                break
            time.sleep(10)
        bud.tick(True)
        if not mace_model.is_file():
            ck = ckpt_epochs(rd, tag)
            if not ck:
                _write_train(out_json, dict(run=R["run"], cand=cand, status="failed", rc=rc, reason=reason,
                                            tail=list(pr.tail)[-40:], gpu_h=bud.now_h()))
                print("\n".join(list(pr.tail)[-40:]), file=sys.stderr)
                raise SystemExit(1 if reason is None else 2)
            es = epoch_stats(hist)
            best = es.get("best_epoch")
            best = best if best in ck else max(ck)
            moved = []
            for ep, p in sorted(ck.items()):
                if ep > best:                                  # a newer non-best checkpoint (restart artefact)
                    dst = _mkdir(rd / "ckpt_superseded") / p.name
                    os.replace(p, dst)
                    moved.append(p.name)
            args = mace_cmd_args(R, C, rd, fpath, train_file, valid_file, e0, o, best)
            cmd = [sys.executable, str(HERE / "finetune.py"), "_mace", *args]
            print(f"[train] finalise from checkpoint epoch {best} (moved aside {moved})", flush=True)
            pf = Proc(cmd, rd / f"mace_attempt{attempt}_finalise.log", env)
            while pf.p.poll() is None:
                bud.tick()
                if bud.now_h() > o.stop_h:
                    pf.stop()
                    M.stop(f"{R['run']}: > {o.stop_h} GPU-h during finalise")
                time.sleep(10)
            pf.th.join(10)
            bud.tick(True)
            if not mace_model.is_file():
                _write_train(out_json, dict(run=R["run"], cand=cand, status="failed_finalise", tail=list(pf.tail)[-40:],
                                            gpu_h=bud.now_h()))
                print("\n".join(list(pf.tail)[-40:]), file=sys.stderr)
                raise SystemExit(1)
    # ---- export: float64, E0 += ATOM_REF (absolute label-level energies)
    exp = export_mace(mace_model, model_out, R, cand, meta, o)
    es = epoch_stats(hist)
    gpu_h = bud.now_h()
    res = dict(run=R["run"], cand=cand, kind="mace", status="done", model=str(model_out), model_card=str(model_out.with_suffix(".json")),
               foundation=str(fpath), split=R["split"], data_fingerprint=meta["fingerprint"],
               inputs_complete=meta["inputs_complete"], subset=o.subset,
               n_train=_count_xyz(train_file), n_valid=_count_xyz(valid_file), gpu=gpu, gpu_h=gpu_h,
               budget_h=o.budget_h, stop_h=o.stop_h, stop_reason=reason, epochs=es, E0s=e0j,
               hyper=dict(lr=o.lr, ew=o.ew, fw=o.fw, batch_size=o.batch_size, max_epochs=o.epochs, patience=o.patience,
                          ema=0.99, scheduler="ReduceLROnPlateau 0.8/3", dtype_train="float32", seed=R["train_seed"],
                          weights_round=o.weights_round),
               projection=_projection(es, train_file, valid_file, sd, o), export=exp, import_guard=guard,
               disk_gb=dict(run_dir=_du_gb(rd), model=model_out.stat().st_size / 1e9),
               decisions={k: DECISIONS[k] for k in ("FT-mace", "FT-E0", "FT-frame", "FT-budget")},
               written=_now(), **_jobtag())
    _write_train(out_json, res)
    print(json.dumps(dict(run=R["run"], status="done", gpu_h=round(gpu_h, 3), epochs=es.get("n"),
                          best=es.get("best", {}).get("loss"), model=str(model_out)), indent=1), flush=True)
    return 0


def _count_xyz(p):
    n = 0
    with open(p) as f:
        while True:
            line = f.readline()
            if not line:
                break
            k = int(line.split()[0])
            n += 1
            for _ in range(k + 1):
                f.readline()
    return n


def _projection(es, train_file, valid_file, sd, o):
    """epoch time measured on this run's data, scaled (linearly in structures) to the full split."""
    if not es.get("epoch_s_median"):
        return None
    n_tr, n_va = _count_xyz(train_file), _count_xyz(valid_file)
    full = M.read_json(sd / "meta.json")["counts"]
    f_tr, f_va = full["train"]["n"], full["valid"]["n"]
    scale = (f_tr + f_va / 3.0) / max(1.0, (n_tr + n_va / 3.0))
    ep_full = es["epoch_s_median"] * scale
    return dict(epoch_s_measured=es["epoch_s_median"], n_train_measured=n_tr, n_valid_measured=n_va,
                n_train_full=f_tr, n_valid_full=f_va, epoch_s_full=ep_full, epoch_h_full=ep_full / 3600,
                epochs_in_budget=int(o.budget_h * 3600 * 0.9 // ep_full) if ep_full > 0 else None,
                note="valid pass ~1/3 of a train step per structure; setup/E0/finalise excluded")


def export_mace(src, dst, R, cand, meta, o):
    import torch
    m = torch.load(src, map_location="cpu", weights_only=False)
    m = m.double()
    zs = [int(z) for z in m.atomic_numbers]
    add = torch.tensor([ATOM_REF_EH[z] * M.EH2EV for z in zs], dtype=torch.float64)
    with torch.no_grad():
        m.atomic_energies_fn.atomic_energies += add
    _wtorch(m, dst)
    card = dict(run=R["run"], cand=cand, kind="mace", file=str(dst), dtype="float64", heads=list(getattr(m, "heads", [])),
                elements=zs, r_max=float(m.r_max), energy="eV, absolute label level (B3LYP-D3BJ/def2-SVP SMD water)",
                forces="eV/A", charge_spin="total_charge 0, total_spin 1 (multiplicity) defaults",
                atom_ref_added_eh=ATOM_REF_EH, split=R["split"], data_fingerprint=meta["fingerprint"],
                usage="mace.calculators.MACECalculator(model_paths=<file>, device='cuda', default_dtype='float64')",
                written=_now())
    M.write_json(dst.with_suffix(".json"), card)
    return dict(file=str(dst), bytes=dst.stat().st_size, n_params=int(sum(p.numel() for p in m.parameters())))


def _write_train(path, d):
    M.write_json(path, d)


def cmd_mace_wrapper(argv):
    """child process: mace_run_train with an epoch hook ([FTEPOCH] json lines: loss, MAE, peak GPU memory)."""
    _guard_imports()
    import torch
    import mace.tools  # noqa: F401  (mace.tools.train is shadowed by the train() function in the package namespace)
    MT = sys.modules["mace.tools.train"]
    orig = MT.valid_err_log

    def hook(valid_loss, eval_metrics, logger, log_errors, epoch=None, valid_loader_name="Default"):
        orig(valid_loss, eval_metrics, logger, log_errors, epoch, valid_loader_name)
        rec = dict(epoch=epoch, head=valid_loader_name, loss=float(valid_loss))
        for k in ("mae_e_per_atom", "mae_f", "rmse_e_per_atom", "rmse_f", "mae_e"):
            v = eval_metrics.get(k)
            if v is not None:
                try:
                    rec[k] = float(v)
                except (TypeError, ValueError):
                    pass
        if torch.cuda.is_available():
            rec["max_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        print("[FTEPOCH] " + json.dumps(rec), flush=True)

    MT.valid_err_log = hook
    from mace.cli import run_train
    if os.environ.get("MLT_MACE_ERROR_TABLE", "0") != "1":     # final train+valid table = a full forward/force pass
        run_train.create_error_table = lambda **kw: "skipped by finetune.py (FT-1 computes the metrics)"
    sys.argv = ["mace_run_train", *argv]
    run_train.main()
    return 0


# ================================================================ train (AIMNet2)
def aim_cfg(C, d3="label"):
    import yaml
    cfg = yaml.safe_load((SITE / "aimnet" / "models" / C["yaml"]).read_text())
    if d3 == "label":
        cfg["kwargs"]["outputs"]["dftd3"]["kwargs"].update(B3LYP_D3BJ)
    return cfg


def aim_build(C, sd, d3="label", shift_double=False):
    from aimnet.config import build_module
    model = build_module(aim_cfg(C, d3))
    if shift_double:                                     # as the released jpt (pt2jpt.add_sae_to_shifts)
        model.outputs.atomic_shift.double()
    res = model.load_state_dict(sd, strict=False)
    return model, dict(missing=list(res.missing_keys), unexpected=list(res.unexpected_keys))


def aim_groups(atoms_list, idx=None):
    idx = list(range(len(atoms_list))) if idx is None else list(idx)
    by = {}
    for i in idx:
        by.setdefault(len(atoms_list[i]), []).append(i)
    return by


def aim_predict(model, atoms_list, device, batch=64, forces=True, idx=None):
    import torch
    n = len(atoms_list) if idx is None else len(idx)
    order = list(range(len(atoms_list))) if idx is None else list(idx)
    pos_of = {g: k for k, g in enumerate(order)}
    E = np.zeros(n)
    F = [None] * n
    for nat, ii in aim_groups(atoms_list, order).items():
        for s in range(0, len(ii), batch):
            jj = ii[s:s + batch]
            c = torch.tensor(np.stack([atoms_list[j].positions for j in jj]), dtype=torch.float32, device=device)
            z = torch.tensor(np.stack([atoms_list[j].numbers for j in jj]), dtype=torch.int64, device=device)
            q = torch.zeros(len(jj), dtype=torch.float32, device=device)
            if forces:
                c.requires_grad_(True)
                out = model(dict(coord=c, numbers=z, charge=q))
                e = out["energy"]
                g = torch.autograd.grad(e.sum(), c)[0]
                fb = (-g).detach().cpu().numpy()
            else:
                with torch.no_grad():
                    out = model(dict(coord=c, numbers=z, charge=q))
                e = out["energy"]
            ev = e.detach().double().cpu().numpy()
            for k, j in enumerate(jj):
                E[pos_of[j]] = ev[k]
                if forces:
                    F[pos_of[j]] = fb[k]
    return E, F


def train_aimnet(R, cand, C, sd, meta, rd, o):
    import torch
    from aimnet.data import SizeGroupedDataset, SizeGroupedSampler
    from aimnet.modules import Forces
    from aimnet.train.loss import MTLoss
    out_json = RES_FT / f"{R['run']}_train.json"
    model_out = model_path(R, C)
    bud = Budget(rd, o.budget_h, o.stop_h)
    if bud.prev_h >= o.stop_h:
        M.stop(f"{R['run']}: {bud.prev_h:.1f} GPU-h already used (> {o.stop_h})")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if dev == "cuda" else None
    jpt = fetch(cand)
    if o.subset:
        train_file, valid_file = subset_files(R, rd, sd, o.subset, R["valid_seed"])
    else:
        train_file, valid_file = sd / "train.xyz", sd / "valid.xyz"
    tr, va = M.read_extxyz(train_file), M.read_extxyz(valid_file)
    # ---- weights of the released model + parity (built module with the wB97M D3 head == jpt)
    jm = torch.jit.load(str(jpt), map_location=dev)
    sd0 = {k: v.detach().clone() for k, v in jm.state_dict().items()}
    chk = [tr[i] for i in range(min(8, len(tr)))]
    m_par, lk = aim_build(C, sd0, d3="foundation", shift_double=True)
    m_par = m_par.to(dev).eval()
    e_j, f_j = aim_predict(jm, chk, dev, forces=True)
    e_b, f_b = aim_predict(m_par, chk, dev, forces=True)
    parity = dict(max_abs_dE_eV=float(np.abs(e_j - e_b).max()),
                  max_abs_dF_eV_A=float(max(np.abs(a - b).max() for a, b in zip(f_j, f_b))), load=lk)
    real_missing = [k for k in lk["missing"] if not k.endswith(("c6ab", "r4r2", "rcov", "cnmax"))]
    if parity["max_abs_dE_eV"] > 1e-3 or parity["max_abs_dF_eV_A"] > 1e-3 or real_missing:
        _write_train(out_json, dict(run=R["run"], cand=cand, status="infeasible", parity=parity,
                                    reason="released AIMNet2 weights do not load into the aimnet yaml model"))
        raise SystemExit(f"AIMNet2 weight parity failed: {parity}")
    del m_par, jm
    # ---- training model: label D3 head, shifts zeroed, NaN (masked species) rows -> 0
    sd0["outputs.atomic_shift.shifts.weight"] = torch.zeros_like(sd0["outputs.atomic_shift.shifts.weight"])
    model, _lk = aim_build(C, sd0, d3="label")
    with torch.no_grad():
        model.afv.weight.nan_to_num_(0.0)
    model = model.to(dev)
    sae_p = rd / "sae.json"
    key = dict(fp=meta["fingerprint"], cand=cand, subset=o.subset, code=CODE_VERSION)
    saej = _rj(sae_p)
    if not saej or saej.get("key") != key:
        t0 = time.time()
        idx = e0_sample(tr)
        model.eval()
        e_pred, _ = aim_predict(model, tr, dev, forces=False, idx=idx)
        sub = [tr[i] for i in idx]
        sae, fitst = lstsq_e0(sub, np.array([a.info["E_fit"] for a in sub]) - e_pred)
        saej = dict(key=key, sae_eV=sae, fit=fitst, n_sample=len(sub), seconds=round(time.time() - t0, 1),
                    method=DECISIONS["FT-E0"], frame="E_fit (eV)")
        M.write_json(sae_p, saej)
    sae = {int(z): float(v) for z, v in saej["sae_eV"].items()}

    def dataset(al):
        d = {}
        for nat, ii in aim_groups(al).items():
            d[nat] = dict(coord=np.stack([al[i].positions for i in ii]).astype(np.float32),
                          numbers=np.stack([al[i].numbers for i in ii]).astype(np.int64),
                          charge=np.zeros(len(ii), np.float32),
                          energy=np.array([al[i].info["E_fit"] - sum(sae[int(z)] for z in al[i].numbers) for i in ii],
                                          np.float64).astype(np.float32),
                          forces=np.stack([al[i].arrays["REF_forces"] for i in ii]).astype(np.float32))
        return SizeGroupedDataset(d)

    ds_tr, ds_va = dataset(tr), dataset(va)
    bs = o.batch_size
    ds_tr.merge_groups(min_size=min(8 * bs, max(1, len(ds_tr) // 4)))     # aimnet get_loaders: 8 x batch
    torch.manual_seed(R["train_seed"])
    np.random.seed(R["train_seed"] % (2 ** 32))
    ld_tr = ds_tr.get_loader(SizeGroupedSampler(ds_tr, batch_size=bs, shuffle=True), ["coord", "numbers", "charge"],
                             ["energy", "forces"], num_workers=0, pin_memory=True)
    ld_va = ds_va.get_loader(SizeGroupedSampler(ds_va, batch_size=2 * bs, shuffle=False),
                             ["coord", "numbers", "charge"], ["energy", "forces"], num_workers=0, pin_memory=True)
    fmodel = Forces(model)
    shifts = [p for n, p in model.named_parameters() if n.endswith("atomic_shift.shifts.weight")]
    other = [p for n, p in model.named_parameters() if not n.endswith("atomic_shift.shifts.weight")]
    opt = torch.optim.RAdam([dict(params=other), dict(params=shifts, weight_decay=0.0)], lr=o.lr, weight_decay=1e-8)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.75, patience=3)
    loss_fn = MTLoss(components=dict(
        energy=dict(fn="aimnet.train.loss.energy_loss_fn", weight=o.ew),
        forces=dict(fn="aimnet.train.loss.peratom_loss_fn", weight=o.fw,
                    kwargs=dict(key_true="forces", key_pred="forces"))))
    last_p, best_p = rd / "aim_last.pt", rd / "aim_best.pt"
    hist, start, best_loss, best_epoch = [], 0, float("inf"), None
    if last_p.is_file():
        ck = torch.load(last_p, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sch.load_state_dict(ck["sch"])
        hist, start, best_loss, best_epoch = ck["hist"], ck["epoch"] + 1, ck["best_loss"], ck["best_epoch"]
    reason = None
    ep_times = [h["epoch_s"] for h in hist]
    for epoch in range(start, o.epochs):
        if ep_times and bud.now_h() + 1.3 * float(np.median(ep_times)) / 3600 > o.budget_h:
            reason = f"budget: next epoch would end past {o.budget_h} GPU-h"
            break
        t0 = time.time()
        model.train()
        tl, nb = 0.0, 0
        for x, y in ld_tr:
            x = {k: v.to(dev, non_blocking=True) for k, v in x.items()}
            y = {k: v.to(dev, non_blocking=True) for k, v in y.items()}
            opt.zero_grad()
            pred = fmodel(x)
            loss = loss_fn(pred, y)["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 0.4)
            opt.step()
            tl += float(loss.item())
            nb += 1
            bud.tick()
            if bud.now_h() > o.stop_h:
                M.stop(f"{R['run']}: > {o.stop_h} GPU-h")
        model.eval()
        vl, vb, ae, af, na = 0.0, 0, [], [], 0
        for x, y in ld_va:
            x = {k: v.to(dev) for k, v in x.items()}
            y = {k: v.to(dev) for k, v in y.items()}
            pred = fmodel(x)
            vl += float(loss_fn(pred, y)["loss"].item())
            vb += 1
            ae.append((pred["energy"] - y["energy"]).abs().detach().cpu().numpy())
            mask = (x["numbers"] > 0).unsqueeze(-1).expand_as(y["forces"])
            af.append((pred["forces"] - y["forces"]).abs()[mask].detach().cpu().numpy())
        vl /= max(1, vb)
        sch.step(vl)
        rec = dict(epoch=epoch, loss=vl, train_loss=tl / max(1, nb), mae_e_kcal=float(np.concatenate(ae).mean() * EV2KCAL),
                   mae_f_kcal_A=float(np.concatenate(af).mean() * EV2KCAL), lr=opt.param_groups[0]["lr"],
                   epoch_s=time.time() - t0, gpu_h=bud.now_h(),
                   max_mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2) if dev == "cuda" else None)
        hist.append(rec)
        ep_times.append(rec["epoch_s"])
        if vl < best_loss:
            best_loss, best_epoch = vl, epoch
            _wtorch(model.state_dict(), best_p)
        _wtorch(dict(model=model.state_dict(), opt=opt.state_dict(), sch=sch.state_dict(), hist=hist, epoch=epoch,
                     best_loss=best_loss, best_epoch=best_epoch), last_p)
        bud.tick(True)
        print(f"[train] epoch {epoch} val loss {vl:.5g} MAE_E {rec['mae_e_kcal']:.3f} kcal/mol MAE_F "
              f"{rec['mae_f_kcal_A']:.3f} kcal/mol/A | best {best_epoch} | {bud.now_h():.2f} GPU-h | {rec['epoch_s']:.0f} s",
              flush=True)
        if epoch - best_epoch >= o.patience:
            reason = f"early stop: {epoch - best_epoch} epochs without improvement"
            break
        if opt.param_groups[0]["lr"] < 1e-6:
            reason = "lr < 1e-6"
            break
    exp = export_aimnet(C, best_p, model_out, sae, R, cand, meta, dev, tr)
    bud.tick(True)
    es = dict(n=len(hist), best_epoch=best_epoch, best=next((h for h in hist if h["epoch"] == best_epoch), None),
              epoch_s_median=float(np.median(ep_times)) if ep_times else None, history=hist)
    res = dict(run=R["run"], cand=cand, kind="aimnet", status="done", model=str(model_out),
               model_card=str(model_out.with_suffix(".json")), foundation=str(jpt), split=R["split"],
               data_fingerprint=meta["fingerprint"], inputs_complete=meta["inputs_complete"], subset=o.subset,
               n_train=len(tr), n_valid=len(va), gpu=gpu, gpu_h=bud.now_h(), budget_h=o.budget_h, stop_h=o.stop_h,
               stop_reason=reason, epochs=es, sae=saej, parity_released_vs_built=parity,
               hyper=dict(lr=o.lr, ew=o.ew, fw=o.fw, batch_size=bs, max_epochs=o.epochs, patience=o.patience,
                          optimizer="RAdam wd 1e-8 (shifts 0)", scheduler="ReduceLROnPlateau 0.75/3", clip_value=0.4,
                          d3=B3LYP_D3BJ, seed=R["train_seed"], weights_round=o.weights_round),
               projection=_projection(es, train_file, valid_file, sd, o), export=exp,
               disk_gb=dict(run_dir=_du_gb(rd), model=model_out.stat().st_size / 1e9),
               decisions={k: DECISIONS[k] for k in ("FT-aimnet", "FT-E0", "FT-frame", "FT-budget")},
               written=_now(), **_jobtag())
    _write_train(out_json, res)
    print(json.dumps(dict(run=R["run"], status="done", gpu_h=round(bud.now_h(), 3), epochs=len(hist),
                          best=best_loss, model=str(model_out)), indent=1), flush=True)
    return 0


def export_aimnet(C, best_p, dst, sae, R, cand, meta, dev, tr):
    import torch
    from aimnet.train.pt2jpt import add_cutoff, mask_not_implemented_species, set_eval
    sdb = torch.load(best_p, map_location="cpu", weights_only=False)
    m, _ = aim_build(C, sdb, d3="label")
    m = set_eval(m)
    m = add_cutoff(m, cutoff_lr=float("inf"))
    m.outputs.atomic_shift.double()
    with torch.no_grad():
        for z, v in sae.items():
            m.outputs.atomic_shift.shifts.weight[int(z)] += v + ATOM_REF_EH[int(z)] * M.EH2EV
    species = sorted(sae)
    m = mask_not_implemented_species(m, species)
    m.register_buffer("impemented_species", torch.tensor(species, dtype=torch.int64))
    js = torch.jit.script(m)
    M.assert_write_ok(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.tmp{os.getpid()}")
    js.save(str(tmp))
    os.replace(tmp, dst)
    # check: exported (absolute) = training frame + sum(sae + ATOM_REF)
    chk = tr[:4]
    mt, _ = aim_build(C, sdb, d3="label")
    mt = mt.to(dev).eval()
    e_t, _ = aim_predict(mt, chk, dev, forces=False)
    e_x, _ = aim_predict(torch.jit.load(str(dst), map_location=dev), chk, dev, forces=False)
    off = np.array([sum(sae[int(z)] for z in a.numbers) + atom_ref_ev(a.numbers) for a in chk])
    dmax = float(np.abs(e_x - (e_t + off)).max())
    card = dict(run=R["run"], cand=cand, kind="aimnet", file=str(dst), format="TorchScript (aimnet jitcompile layout)",
                elements=species, d3=B3LYP_D3BJ, energy="eV, absolute label level (B3LYP-D3BJ/def2-SVP SMD water)",
                forces="eV/A (autograd)", split=R["split"], data_fingerprint=meta["fingerprint"],
                usage="aimnet.calculators.AIMNet2Calculator(<file>) / orca-mlips aimnet2 backend with a local path",
                export_check_max_abs_eV=dmax, written=_now())
    M.write_json(dst.with_suffix(".json"), card)
    if dmax > 1e-3:
        raise SystemExit(f"AIMNet2 export check failed: {dmax} eV")
    return dict(file=str(dst), bytes=dst.stat().st_size, check_max_abs_eV=dmax)


# ================================================================ train dispatcher
def model_path(R, C):
    """deployable runs -> $MLT/models/<run>.<ext>; BENCH runs stay in their run dir (never used downstream)."""
    ext = "model" if C["kind"] == "mace" else "jpt"
    return (FT / R["run"] / f"{R['run']}.{ext}") if R["bench"] else (MODELS / f"{R['run']}.{ext}")


def cmd_train(run, cand, o):
    R = parse_run(run, o.with_al)
    cand = canon_cand(cand)
    if R["cand"] and R["cand"] != cand:
        raise SystemExit(f"run {R['run']} is for candidate {R['cand']}, not {cand}")
    C = CANDIDATES[cand]
    if not C["si"]:
        raise SystemExit(f"{cand}: {CAND_NOTES['no_si']}")
    sd = FT_DATA / R["split"]
    meta = _rj(sd / "meta.json")
    if not meta or not (sd / "train.xyz").is_file():
        M.not_ready(f"split {R['split']} not built: run `finetune.py data {R['run']}` first")
    if not meta["inputs_complete"] and not (o.allow_partial or R["bench"]):
        M.not_ready(f"split {R['split']} was built from incomplete inputs (G collect not complete); rebuild with "
                    f"`data` after G collect, or pass --allow-partial")
    _guard_imports()
    if o.epochs is None:
        o.epochs = 200 if C["kind"] == "mace" else 1000
    if o.patience is None:
        o.patience = 6 if C["kind"] == "mace" else 25
    if o.batch_size is None:
        o.batch_size = 8 if C["kind"] == "mace" else 64
    if o.lr is None:
        o.lr = 1e-3 if C["kind"] == "mace" else 1e-4
    if o.ew is None:
        o.ew = 1.0
    if o.fw is None:
        o.fw = 1.0 if C["kind"] == "mace" else 0.2
    o.weights_round = 0 if (o.ew, o.fw) == ((1.0, 1.0) if C["kind"] == "mace" else (1.0, 0.2)) else 1
    rd = FT / R["run"]
    prev = _rj(RES_FT / f"{R['run']}_train.json")
    model_out = model_path(R, C)
    if prev and prev.get("status") == "done" and model_out.is_file() and not o.force:
        if prev.get("data_fingerprint") == meta["fingerprint"] and prev.get("cand") == cand:
            print(f"[train] {R['run']} done already ({model_out})", flush=True)
            return 0
    old = _rj(rd / "run_key.json")
    key = dict(cand=cand, fp=meta["fingerprint"], subset=o.subset, lr=o.lr, ew=o.ew, fw=o.fw, bs=o.batch_size)
    if rd.exists() and (old is not None) and (old != key or o.force):
        stale = rd.with_name(f"{rd.name}.stale_{time.strftime('%Y%m%d_%H%M%S')}")
        os.replace(rd, stale)
        print(f"[train] run key changed / --force: previous run dir moved to {stale} (not deleted)", flush=True)
    _mkdir(rd)
    M.write_json(rd / "run_key.json", key)
    if C["kind"] == "mace":
        return train_mace(R, cand, C, sd, meta, rd, o)
    return train_aimnet(R, cand, C, sd, meta, rd, o)


# ================================================================ FT-1
def ft1_metrics(D, Fref, Fpred):
    """D: DataFrame per structure (kind, dform, rxn, formula, n_atoms, e_ref, e_rel_pred); forces lists (eV/A)."""
    ts = ts_near_mask(D).to_numpy()
    fae = [np.abs(a - b).ravel() for a, b in zip(Fref, Fpred)]

    def fmae(mask):
        v = [fae[i] for i in np.where(mask)[0]]
        return float(np.concatenate(v).mean() * EV2KCAL) if v else None

    def rel(mask):
        S = D[mask & (D.rxn.map(len) == 1).to_numpy()].copy()
        if not len(S):
            return dict(n=0, n_groups=0, mae_kcal=None)
        S["g"] = S.rxn.map(lambda r: r[0]).astype(str) + "|" + S.formula
        S = S[S.groupby("g").g.transform("size") >= 2]
        if not len(S):
            return dict(n=0, n_groups=0, mae_kcal=None)
        dr = S.e_ref - S.groupby("g").e_ref.transform("mean")
        dp = S.e_rel_pred - S.groupby("g").e_rel_pred.transform("mean")
        err = np.abs(dr - dp) * EV2KCAL
        return dict(n=int(len(S)), n_groups=int(S.g.nunique()), mae_kcal=float(err.mean()),
                    q90_kcal=float(np.quantile(err, 0.9)), n_rxns=int(S.rxn.map(lambda r: r[0]).nunique()))

    allm = np.ones(len(D), bool)
    out = dict(n=int(len(D)), n_ts_near=int(ts.sum()),
               force_mae_kcal_A=dict(ts_near=fmae(ts), all=fmae(allm)),
               rel_e=dict(ts_near=rel(ts), all_single_rxn=rel(allm)),
               force_mae_by_kind_kcal_A={k: fmae((D.kind == k).to_numpy()) for k in sorted(D.kind.unique())},
               n_by_kind=dict(Counter(D.kind)))
    if "e_abs_pred" in D:
        out["abs_e_mae_kcal_per_atom"] = float((np.abs(D.e_abs_pred - D.e_ref) / D.n_atoms).mean() * EV2KCAL)
    fm, em = out["force_mae_kcal_A"]["ts_near"], out["rel_e"]["ts_near"]["mae_kcal"]
    out["targets"] = FT1_TARGET
    out["meets"] = dict(force=None if fm is None else fm <= FT1_TARGET["force_mae_kcal_A"],
                        rel_e=None if em is None else em <= FT1_TARGET["rel_e_mae_kcal"])
    if fm is not None and em is not None:
        rf, re_ = fm / FT1_TARGET["force_mae_kcal_A"], em / FT1_TARGET["rel_e_mae_kcal"]
        out["ratio_to_target"] = dict(force=rf, rel_e=re_)
        out["weights_hint"] = ("raise energy weight" if re_ > 1 and re_ > 1.5 * rf else
                               "raise forces weight" if rf > 1 and rf > 1.5 * re_ else "keep")
    return out


def frame(al):
    import pandas as pd
    rows = []
    for a in al:
        rx = tuple(sorted({int(x) for x in str(a.info.get("rxn_ids") or "").split(",") if x}))
        d = a.info.get("dform_label_A")
        rows.append(dict(struct_id=a.info.get("struct_id"), kind=a.info.get("kind"), rxn=rx,
                         formula=a.get_chemical_formula(mode="hill"), n_atoms=len(a),
                         e_ref=float(a.info["e_eh"]) * M.EH2EV, dform=float(d) if d is not None else float("nan")))
    return pd.DataFrame(rows)


def cmd_ft1(run, o):
    import torch
    _guard_imports()
    R = parse_run(run, o.with_al)
    sd = FT_DATA / R["split"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if o.foundation:
        cand = canon_cand(o.foundation)
        tj = None
        label = f"foundation_{cand}"
    else:
        tj = _rj(RES_FT / f"{R['run']}_train.json")
        if not tj or tj.get("status") != "done":
            M.not_ready(f"{R['run']}: no finished training (results/ft/{R['run']}_train.json)")
        cand, label = tj["cand"], "finetuned"
    C = CANDIDATES[cand]
    meta = M.read_json(sd / "meta.json")
    if tj and tj.get("data_fingerprint") != meta["fingerprint"]:
        print("[ft1] WARNING: split rebuilt after training (fingerprint differs)", flush=True)
    sets = [("valid", sd / "valid.xyz")] + ([("test", sd / "test.xyz")] if (sd / "test.xyz").is_file() else [])
    if tj and tj.get("subset"):
        sets = [("valid_subset", FT / R["run"] / "sub_valid.xyz")] + [s for s in sets if s[0] == "test"]
    if C["kind"] == "mace":
        mp = Path(tj["model"]) if tj else fetch(cand)
        model = torch.load(mp, map_location="cpu", weights_only=False)
        model = (model.float() if o.dtype == "float32" else model.double()).to(dev)
        head = None if tj else C.get("head")
    else:
        mp = Path(tj["model"]) if tj else fetch(cand)
        model = torch.jit.load(str(mp), map_location=dev)
        head = None
    out = dict(run=R["run"], cand=cand, model=str(mp), which=label, dtype=o.dtype if C["kind"] == "mace" else
               "float32 (shifts float64)", device=torch.cuda.get_device_name(0) if dev == "cuda" else "cpu", sets={},
               definitions=dict(ts_near=DECISIONS["FT-ts-near"], rel_e=DECISIONS["FT-relE"],
                                force_mae="mean |F_pred - F_ref| over Cartesian components, kcal mol-1 A-1"),
               note="FT-1 is recorded only (spec 5), the gate is FT-2 geometry")
    for nm, p in sets:
        al = M.read_extxyz(p)
        if o.max_n:
            al = al[: o.max_n]
        D = frame(al)
        t0 = time.time()
        if C["kind"] == "mace":
            e, ei, F = mace_predict(model, al, dev, batch_size=o.batch_size, forces=True, head=head)
            D["e_abs_pred"] = e
            D["e_rel_pred"] = ei
        else:
            e, F = aim_predict(model, al, dev, batch=max(16, o.batch_size * 4), forces=True)
            D["e_abs_pred"] = e
            D["e_rel_pred"] = e
        dt = time.time() - t0
        if tj is None:
            D = D.drop(columns=["e_abs_pred"])                 # foundation: different level, absolute E meaningless
        Fref = [np.asarray(a.arrays["REF_forces"], float) for a in al]
        met = ft1_metrics(D, Fref, F)
        met.update(path=str(p), predict_s=round(dt, 1), structures_per_s=round(len(al) / max(dt, 1e-9), 2))
        out["sets"][nm] = met
        D["f_mae_kcal_A"] = [float(np.abs(a - b).mean() * EV2KCAL) for a, b in zip(Fref, F)]
        D["rxn"] = D.rxn.map(M.tag_list)
        M.write_parquet(FT / R["run"] / f"ft1_{label}_{nm}.parquet", D)
        print(f"[ft1] {R['run']} {nm}: n {len(al)} | TS-near F MAE {met['force_mae_kcal_A']['ts_near']} kcal/mol/A | "
              f"rel E MAE {met['rel_e']['ts_near']['mae_kcal']} kcal/mol | {len(al) / max(dt, 1e-9):.1f} struct/s",
              flush=True)
    out.update(written=_now(), **_jobtag())
    name = f"{R['run']}_ft1.json" if tj else f"{R['run']}_ft1_{label}.json"
    M.write_json(RES_FT / name, out)
    return 0


# ================================================================ status
def cmd_status():
    runs = sorted({p.name.rsplit("_", 1)[0] for p in RES_FT.glob("*_data.json")} |
                  {p.name[: -len("_train.json")] for p in RES_FT.glob("*_train.json")})
    print(f"{'run':28s} {'split':10s} {'n_train':>8s} {'n_valid':>7s} {'complete':>8s} {'cand':16s} {'status':10s} "
          f"{'ep':>4s} {'GPU-h':>6s} {'F_TS':>6s} {'relE':>6s}")
    for r in runs:
        d = _rj(RES_FT / f"{r}_data.json") or {}
        t = _rj(RES_FT / f"{r}_train.json") or {}
        f = _rj(RES_FT / f"{r}_ft1.json") or {}
        v = (f.get("sets") or {}).get("valid") or (f.get("sets") or {}).get("valid_subset") or {}
        fts = (v.get("force_mae_kcal_A") or {}).get("ts_near")
        rel = ((v.get("rel_e") or {}).get("ts_near") or {}).get("mae_kcal")
        c = d.get("counts") or {}
        print(f"{r:28s} {d.get('split', ''):10s} {c.get('train', {}).get('n', ''):>8} {c.get('valid', {}).get('n', ''):>7} "
              f"{str(d.get('inputs_complete', '')):>8s} {t.get('cand', ''):16s} {t.get('status', ''):10s} "
              f"{(t.get('epochs') or {}).get('n', ''):>4} {('%.2f' % t['gpu_h']) if t.get('gpu_h') is not None else '':>6s} "
              f"{('%.3f' % fts) if fts is not None else '':>6s} {('%.3f' % rel) if rel is not None else '':>6s}")
    return 0


# ================================================================ main
def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "_mace":
        return cmd_mace_wrapper(argv[1:])
    ap = argparse.ArgumentParser(description="DEPLOY-V2 (MLTS) Phase FT")
    sub = ap.add_subparsers(dest="cmd")
    a = sub.add_parser("data")
    a.add_argument("runs", nargs="*")
    a.add_argument("--all", action="store_true")
    a.add_argument("--with-al", action="store_true")
    a.add_argument("--fmax-cap", type=float, default=FMAX_CAP)
    a.add_argument("--valid-frac", type=float, default=VALID_FRAC)
    a.add_argument("--force", action="store_true")
    a = sub.add_parser("fetch")
    a.add_argument("cand")
    a = sub.add_parser("train")
    a.add_argument("run")
    a.add_argument("cand")
    a.add_argument("--with-al", action="store_true")
    a.add_argument("--epochs", type=int, default=None, help="max epochs (MACE 200, AIMNet2 1000)")
    a.add_argument("--budget-h", type=float, default=BUDGET_H)
    a.add_argument("--stop-h", type=float, default=STOP_H)
    a.add_argument("--batch-size", type=int, default=None,
                   help="MACE 8 (OMOL-0 bench: same structures/s as 16, 9.6 vs 15.6 GB), AIMNet2 64")
    a.add_argument("--lr", type=float, default=None, help="MACE 1e-3, AIMNet2 1e-4")
    a.add_argument("--ew", type=float, default=None, help="energy weight (1)")
    a.add_argument("--fw", type=float, default=None, help="forces weight (MACE 1, AIMNet2 0.2)")
    a.add_argument("--patience", type=int, default=None, help="epochs without a better valid loss (MACE 6, AIMNet2 25)")
    a.add_argument("--subset", type=int, default=0)
    a.add_argument("--num-workers", dest="num_workers", type=int, default=0)
    a.add_argument("--allow-partial", action="store_true")
    a.add_argument("--force", action="store_true")
    a = sub.add_parser("ft1")
    a.add_argument("run")
    a.add_argument("--with-al", action="store_true")
    a.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    a.add_argument("--foundation", default=None)
    a.add_argument("--batch-size", type=int, default=16)
    a.add_argument("--max-n", dest="max_n", type=int, default=0)
    sub.add_parser("status")
    o = ap.parse_args(argv)
    if o.cmd == "data":
        return cmd_data(o.runs, o.all, o.with_al, o.fmax_cap, o.valid_frac, o.force)
    if o.cmd == "fetch":
        return cmd_fetch(o.cand)
    if o.cmd == "train":
        return cmd_train(o.run, o.cand, o)
    if o.cmd == "ft1":
        return cmd_ft1(o.run, o)
    if o.cmd == "status":
        return cmd_status()
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
