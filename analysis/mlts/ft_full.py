#!/usr/bin/env python3
r"""ft_full.py — DEPLOY-V2 (MLTS), step 1 of the user's consolidated instruction (11-05 20:5x): full-data AIMNet2
fine-tuning (FT-cap removed) from a compact binary store. Reuses finetune.py unchanged for everything that decides
WHICH structures exist and where they go (Pool: level / finite / odd-electron / fmax filters, exact-duplicate dedupe,
D0-holdout drop FT-d0hold, truncated-system drop FT-leak-trunc; build_split: FT-split / FT-nurion / FT-d1val membership,
FT-cap tier rule for the test cap, the C-2 leakage check) and for the AIMNet2 parts (aim_build, release parity,
aim_predict, lstsq_e0 E0 fit, export_aimnet, ft1_metrics). New here: the store, the split index files, the sharded
loader, the EMA trainer with the user's schedule and stop rules, the store-reading FT-1. Compute nodes only
(jobs/mlt.slurm); python never runs on the login node. The capped / pilot runs (FT_<k>, FT_all_s<i>, FT_LODO_<d>,
FTpilot_*, FTabl_*) and their files are untouched (finetune.py refuses *_full runs).

Runs (finetune.parse_run; membership and seeds = the capped run of the same split; '@I' = $SLURM_ARRAY_TASK_ID)
  FT_all_full (= all_s0 seeds), FT_all_full_s1, FT_all_full_s2   [full_all_s<i>]  all data, validation seed differs
  FT_<k>_full        k = 0..4   [full_fold<k>]  fold k held out (= FT_<k>): FT-2 / deploy-path TS of fold k
  FT_LODO_<d>_full   d = 0..4   [full_lodo<d>]  dipole d held out (= FT_LODO_<d>)

CLI (submit from analysis/mlts; exit 0 = done / up to date, 2 = not ready, or paused before the 48 h wall -> resubmit
the same command, 3 = STOP)
  data [<run> ...] | --all  [--no-g4] [--fmax-cap 10] [--valid-frac 0.10] [--cap-test 20000]
        [--geo-c2 stop|record|off] [--force]
        MLTENV=mlts, CPU: one finetune.Pool pass -> $MLT/ft/store/store_<fp16>/ (written once, shared by every run;
        meta.json, structs.npz, table.parquet, shards/sNNN/{Z,pos,frc}.npy) + per split
        $MLT/ft/full/<split>/<fp16>/{train,valid,test}.npy + meta.json and the pointer $MLT/ft/full/<split>/current.json;
        results/ft/<run>_data.json. Up-to-date splits are skipped (idempotent); nothing is deleted (a --force rebuild
        moves the old split dir aside).
  train <run> [--fw X] [--max-epochs 1000] [--cap-gpu-h 150] [--wall-h 47] [--allow-partial] [--force]
        MLTENV=mlts, 1 GPU: $MLT/ft/<run>/ (run_key.json, last.pt, best_ts_ema.pt, sae.json, budget.json, epochs.jsonl,
        stop.json, lock.json), $MLT/models/<run>.jpt + <run>.json (EMA weights picked by FTF-export: the best epoch on
        the selection metric = validation TS-near force MAE, user 11-06; mlts_bridge / ts_geom load it by name),
        results/ft/<run>_train.json (rewritten every epoch: status running, then done)
  ft1 <run> [--batch-size 64] [--force]
        MLTENV=mlts, 1 GPU: FT-1 (record only) on valid (+ test) read from the store -> results/ft/<run>_ft1.json,
        $MLT/ft/<run>/ft1_finetuned_<set>.parquet
  export <run> --tag <tag> [--force]
        MLTENV=mlts, GPU or CPU: interim model (user 11-06 step 2; training is not interrupted) from a COPY of
        $MLT/ft/<run>/last.pt (old or new checkpoint format; no lock, nothing written in the run dir), weights by
        FTF-export -> $MLT/models/<run>_<tag>.jpt + .json (card interim=true; mlts_bridge / ts_geom load
        '<run>_<tag>' by name), results/ft/<run>_<tag>_export.json; prints the TS-near metric history
  status                           one line per *_full run

sbatch (from analysis/mlts; login: one-shot sbatch only; L=/gpfs/tmp_cpu2/yeseo1ee/mlts/logs)
  sbatch -p cpu1,cpu2 -J ftf_data --time=48:00:00 -c 8 --mem=64G --output=$L/%x_%j.out --export=ALL,MLTENV=mlts \
         jobs/mlt.slurm ft_full.py data --all
  sbatch -p gpu1,gpu2,gpu3,gpu4,gpu5,gpu6 -J ftf_all --time=48:00:00 --gres=gpu:1 -c 8 --mem=32G \
         --output=$L/%x_%j.out --dependency=afterok:<data job> --kill-on-invalid-dep=yes --export=ALL,MLTENV=mlts \
         jobs/mlt.slurm ft_full.py train FT_all_full ::: ft_full.py ft1 FT_all_full
  sbatch -p gpu1,gpu2,gpu3,gpu4,gpu5,gpu6 -J ftf_k --array=0-4 --time=48:00:00 --gres=gpu:1 -c 8 --mem=32G \
         --output=$L/%x_%A_%a.out --dependency=afterok:<data job> --kill-on-invalid-dep=yes --export=ALL,MLTENV=mlts \
         jobs/mlt.slurm ft_full.py train FT_@I_full ::: ft_full.py ft1 FT_@I_full
  sbatch -p gpu1,gpu2,gpu3,gpu4,gpu5,gpu6 -J ftf_exp --time=48:00:00 --gres=gpu:1 -c 4 --mem=32G \
         --output=$L/%x_%j.out --export=ALL,MLTENV=mlts jobs/mlt.slurm ft_full.py export FT_0_full --tag mid
         (interim model FT_0_full_mid; the training of FT_0_full keeps running)
  A 48 h wall: the trainer pauses itself (exit 2, ft1 skipped) when the next epoch would end past --wall-h 47 h of this
  job; resubmit the same line (without the dependency): it resumes from last.pt (model + EMA + optimizer + scheduler +
  epoch / warmup state). A done run exits 0 at once and ft1 re-runs only when the training json is newer.
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
import signal  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mlt_common as M  # noqa: E402
import finetune as F  # noqa: E402

CODE_VERSION = "mlts-ftfull-1"
T0 = float(os.environ.get("MLT_JOB_T0") or time.time())     # job start (wall check)
STORE_ROOT = F.FT / "store"
SPLIT_ROOT = F.FT / "full"
SHARD_N = 65536                                     # structures per shard (~60 MB at ~34 atoms / structure)
PERM_SEED = M.SEED + 5000                           # store order = seeded permutation of the pool rows (shard = iid sample)
G4_REPORT = M.RES / "G4_report.json"                # g4_al collect --g4 (complete, n_selected, ...)
ABL_CHOICE = M.MLT / "pipeline" / "ft_abl_choice.json"   # lead-written FT-abl decision {"choice": tag, "fw": x}
PIPE_STATE = M.RES / "PIPELINE_STATE.json"          # params ft_abl_choice / ft_train_args / ft_abl_cand (decide_ftabl)
ABL_FW = {"baseline": 0.2, "fw1": 1.0, "fw5": 5.0}  # = pipeline.ABL_FW (pipeline.py is frozen; not imported)
ABL_CAND = "AIMNet2"
FIRST_RUNS = ["FT_all_full"] + [f"FT_{k}_full" for k in range(5)]
FILL_RUNS = ["FT_all_full_s1", "FT_all_full_s2"] + [f"FT_LODO_{d}_full" for d in range(5)]
FULL_RUNS = FIRST_RUNS + FILL_RUNS
LOCK_STALE_S = 900
H = dict(cand="AIMNet2", lr=1e-4, lr_warm0=1e-5, warmup_epochs=3, ema=0.99, plateau_factor=0.8, plateau_patience=10,
         plateau_threshold=0.0, lr_stop=1e-6, no_improve_stop=100, max_epochs=1000, cap_gpu_h=150.0, ew=1.0, fw_default=1.0, batch=64,
         valid_batch=128, clip_value=0.4, wall_h=47.0, e0_nmax=40000, grad_skip_consec_stop=10,
         grad_skip_epoch_stop=100)
# decision FTF-sel (user 11-06): ONE selection metric for best checkpoint, ReduceLROnPlateau and early stopping
SEL_KEY = "mae_f_ts_kcal_A"                         # epoch record key: validation TS-near force MAE, EMA weights
SEL_NAME = "val TS-near force MAE (EMA)"
PREV_SEL = "val force MAE over all validation structures (EMA) [mae_f_kcal_A]"   # the metric before 11-06
SEL_CODE = "ts-sel-1"                               # last.pt 'ckpt_format' of checkpoints with the TS-near fields
LEGACY_BEST = "best_ema_allF_legacy.pt"             # the old all-structure best_ema.pt, renamed at the migration

DECISIONS = {
    "FTF-scope": "user 11-05 20:5x step 1: the FT-cap train / valid caps are removed; every structure that passes the "
                 "finetune.Pool filters (level label, finite, even electron count, fmax <= 10 eV/A, exact-duplicate "
                 "dedupe FT-dedupe, D0 holdout FT-d0hold, truncated Nurion systems FT-leak-trunc) is used. The pilot / "
                 "capped models are not retrained; their split files and models are not touched.",
    "FTF-store": "the filtered pool is written once as a binary store (not extxyz text): rows in a seeded random order "
                 "(numpy default_rng(SEED + 5000).permutation; every shard is an iid sample of the pool), shards of "
                 "65,536 structures with Z int8, positions float32 (A), REF_forces float32 (eV/A) concatenated per atom; "
                 "structs.npz per structure: n_atoms, charge, e_fit / e_eh (float64), ts_near, tier, g4dup, shard, atom "
                 "offset, pool row; table.parquet: struct_id, kind, source, input, formula, rxn ids, folds, dipoles, D0 "
                 "class / id, job core, cap group, tier, dform_label_A, fmax, energies, ts_near. Store fingerprint = "
                 "input sha256s (incl. G4 when used) + pool parameters; a changed input makes a new store dir (old kept).",
    "FTF-split": "membership = finetune.build_split unchanged (drop by tags OR targets(); validation = whole fold_core "
                 "groups in a seeded order until >= 10 % of the remaining D1 reactions, + Nurion non-target rows of "
                 "those cores; test = dropped structures whose rxn_ids are all held out) with caps train 0 / valid 0 "
                 "(no cap) / test 20,000 (FT-cap tier rule and seed). Validation is NOT capped: the existing validation "
                 "logic has no size limit besides FT-cap (FT_all_s0 had 81,569 validation structures before the cap, "
                 "fold 4 41,739). Seeds = the capped runs' (valid SEED+100+k / +200+i / +300+d; cap SEED+1000+same), "
                 "so FT_<k>_full holds out exactly fold k as FT_<k>. Index files = sorted int32 store row ids.",
    "FTF-g4": "G-4 (kind al_ts, $MLT/dft/gen_G4.extxyz) is included iff the file exists and results/G4_report.json says "
              "complete at data-build time (finetune with_al rules: G4 read last, dedupe after the base inputs, "
              "hold_al, tier 1); otherwise 'G-4 not used' is recorded. A running training keeps its pinned split.",
    "FTF-c2geo": "besides finetune's C-2 (tags / targets / D0 holdout; STOP on failure), the geometric half of tests.py "
                 "C-2 (no train / valid structure with the element sequence of a held-out label TS within Kabsch RMSD "
                 "0.10 A, or with its formula and sorted-distance RMS <= 0.03 A; tests.kabsch_rmsd / dist_spectrum) runs "
                 "in the data job for every fold / lodo split, because tests.py c2 only checks fold<k> / lodo<d> cards "
                 "(a *_full card has split full_<...>). --geo-c2 stop (default): a hit = that split is not written, "
                 "its runs' data json say 'STOP geo-C2' and their trainers STOP (exit 3) at start; the data job exits "
                 "3 only when no requested split is clean (else 0 with a [STOP-PARTIAL] line), so afterok trainings "
                 "of clean splits still run. Kabsch runs only where the sorted-spectrum RMS <= 2 x 0.10 A (exact "
                 "bound, same hits). The mode is recorded in the meta, not in the split fingerprint.",
    "FTF-train": "AIMNet2 as finetune.train_aimnet: released aimnet2_wb97m_d3_0 weights (parity check vs the yaml "
                 "build), atomic shifts zeroed, NaN species rows of afv -> 0, D3 head = B3LYP-D3(BJ) (s8 1.9889, a1 "
                 "0.3981, a2 4.4211), molecular charge input, E0 (SAE) by least squares on <= 40,000 training structures "
                 "(all with a rare element), MTLoss energy_loss_fn (ew 1) + peratom_loss_fn forces (fw), RAdam wd 1e-8 "
                 "(shifts 0), batch 64, torch.nn.utils.clip_grad_value_ 0.4 (the existing value: element-wise value "
                 "clipping, not norm). Loader: per epoch the shards in a seeded order (rng [train_seed, epoch]); a "
                 "shard's training rows are gathered from the memory-mapped .npy files (next shard prefetched in a "
                 "thread), sorted by atom count after a seeded shuffle, cut into batches of <= 64 (padding to the batch "
                 "maximum, numbers 0 = padding as aimnet merge_groups), batch order shuffled. Validation rows are "
                 "gathered once (exact-size batches of 128, as before).",
    "FTF-ema": "weight EMA 0.99 updated after every optimizer step (all parameters incl. shifts; torch._foreach_lerp_); "
               "validation and the exported model use the EMA weights; best_ts_ema.pt = the EMA weights FTF-export "
               "picks (TS-near selection metric).",
    "FTF-sched": "selection metric (FTF-sel, user 11-06) = validation TS-near force MAE (kcal/mol/A, mean |dF| over "
                 "Cartesian components of real atoms of the TS-near validation structures) with EMA weights "
                 "(sel_metric(rec) = rec['mae_f_ts_kcal_A']); it drives ReduceLROnPlateau (factor 0.8, patience 10, "
                 "threshold 0 rel = strictly lower), best-checkpoint selection (strictly lower) and early stopping, so "
                 "all three share one improvement definition. Validation force MAE over all structures and energy MAE "
                 "(training frame, kcal/mol) are recorded only. Before 11-06 the metric was the all-structure force "
                 "MAE (runs migrated mid-training record switched_at_epoch and previous_metric). Warmup: epochs 0-2, lr "
                 "linear per optimizer step 1e-5 -> 1e-4; the plateau scheduler steps from epoch 3 on; right before "
                 "its first step scheduler.best = min(scheduler.best, best TS-near of epochs 0-2), so the scheduler's "
                 "best is the same value as the best-checkpoint / early-stop best.",
    "FTF-sel": "user 11-06: best checkpoint, ReduceLROnPlateau and early stop use the validation TS-near force MAE "
               "(EMA). Resume of a last.pt written by the old code (no TS fields): best_ts_epoch / best_ts_metric = "
               "first strict argmin / min of mae_f_ts_kcal_A over the history records (records without it skipped); "
               "early-stop counter = last epoch - best_ts_epoch; the current lr is kept; scheduler.best = "
               "best_ts_metric, num_bad_epochs = min(patience, number of epochs after best_ts_epoch at which the "
               "scheduler stepped (epoch >= 3)) (= min(since-best, patience) whenever best_ts_epoch >= 2), "
               "cooldown_counter 0, so with mode min / threshold 0 rel a stale run reduces the lr at its next "
               "non-improving epoch. Weights of the historical TS-near best exist only when best_ts_epoch is the epoch "
               "of the old best_ema (all-structure best) or the last epoch (EMA of last.pt); else ts_best_weights = "
               "'not retained (epoch N)'. From the switch on, best_ts_ema (EMA at the best TS-near metric seen) and "
               "best_ts_sw_ema (best TS-near among the epochs evaluated after the switch) are kept in last.pt; an "
               "epoch after the switch that beats best_ts_metric becomes the best (weights kept). The old "
               "best_ema.pt is renamed best_ema_allF_legacy.pt at the migration and is never exported.",
    "FTF-export": "export weights (final train export and the interim `export` command): the TS-near best if its "
                  "weights are kept; else the best TS-near since the switch; else (no epoch evaluated after the "
                  "switch) the saved candidate (old all-structure best_ema or last-epoch EMA) with the lowest recorded "
                  "TS-near metric. export_epoch / export_metric (TS-near) / export_rule go to the card and the train "
                  "json. The all-structure best is never exported.",
    "FTF-stop": "stop when lr < 1e-6 after a scheduler step, or 100 epochs without a better validation TS-near force "
                "MAE (FTF-sel), or 1,000 epochs, or the GPU-h cap; then export the FTF-export weights (stop.json keeps "
                "the reason so a resubmit goes straight to export).",
    "FTF-budget": "GPU-h = wall-h of the train command (finetune.Budget, budget.json, summed over attempts). For *_full "
                  "runs the 12 h target / 24 GPU-h STOP of FT-budget is replaced by a 150 GPU-h cap (user 11-05 "
                  "20:5x): no epoch is started when it would end past 150 GPU-h, and a cap reached mid-epoch drops that "
                  "epoch; both are a normal stop (export, exit 0), not exit 3.",
    "FTF-fw": "force weight: --fw if given; else the lead's $MLT/pipeline/ft_abl_choice.json ('fw', or 'choice' -> "
              "baseline 0.2 / fw1 1.0 / fw5 5.0); else PIPELINE_STATE.json params ft_abl_choice / ft_train_args (when "
              "ft_abl_cand is AIMNet2); else 1.0 (no FT-abl decision). The value is pinned in run_key.json at the "
              "first start; a later decision is recorded but does not change a started run (restart = --force).",
    "FTF-resume": "checkpoint every epoch (last.pt: model, EMA, optimizer, scheduler, epoch, optimizer step, TS-near "
                  "best metric / epoch + its EMA weights, best-since-switch metric / epoch + weights, selection record, "
                  "history; atomic; best_ts_ema.pt is rewritten from that copy after last.pt, and again right before "
                  "export, so weights and recorded epoch always agree). "
                  "lock.json is released on every exit (SIGTERM from scancel / the wall -> exit 143 through the "
                  "finally; a job killed without it leaves a lock that blocks a new job for 900 s after its last "
                  "heartbeat). A resubmit resumes; the run key (split fingerprint, store, lr, ew, "
                  "fw, batch, EMA) is pinned, a different key = STOP (--force moves the run dir aside, never deletes). "
                  "The trainer pauses (exit 2) when the next epoch would end past --wall-h of this job.",
    "FTF-selfcheck": "at every start: split meta (C-2 pass, inputs complete, store fingerprint), index overlap, the "
                     "first training shard (atom count, elements, finite arrays), one training batch forward + "
                     "backward (finite loss and gradients, no optimizer step) and one validation batch with the EMA "
                     "model (finite energies / forces); any failure = STOP (exit 3) before the E0 fit and training.",
}


def _jobtag():
    return dict(F._jobtag(), code_full=CODE_VERSION, code_sel=SEL_CODE)   # CODE_VERSION is in the data fingerprints


def _ranges(starts, lens):
    """concatenated aranges [starts[i], starts[i] + lens[i]) (vectorised)."""
    lens = np.asarray(lens, np.int64)
    tot = int(lens.sum())
    if tot == 0:
        return np.zeros(0, np.int64)
    excl = np.cumsum(lens) - lens
    return np.repeat(np.asarray(starts, np.int64) - excl, lens) + np.arange(tot, dtype=np.int64)


def _tests():
    """analysis/mlts/tests.py loaded by file path (the sibling directory tests/ must not shadow it); read only."""
    if "mlts_tests" not in sys.modules:
        import importlib.util
        spec = importlib.util.spec_from_file_location("mlts_tests", HERE / "tests.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["mlts_tests"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["mlts_tests"]


def full_run(run):
    r = str(run)
    if "@I" in r:
        r = r.replace("@I", str(M.array_task("@I")))
    R = F.parse_run(r)
    if not R.get("full"):
        raise SystemExit(f"{run}: not a *_full run (FT_<k>_full, FT_all_full[_s<i>], FT_LODO_<d>_full)")
    return R


# ================================================================ store
def store_ok(sdir, fp):
    m = F._rj(Path(sdir) / "meta.json")
    if not (m and m.get("fingerprint") == fp and m.get("complete")):
        return False
    sh = Path(sdir) / "shards"
    return ((Path(sdir) / "structs.npz").is_file() and (Path(sdir) / "table.parquet").is_file()
            and all((sh / f"s{s:03d}" / f"{k}.npy").is_file() for s in range(int(m["n_shards"]))
                    for k in ("Z", "pos", "frc")))


def order_sha(P, perm):
    h = hashlib.sha256()
    sid = P["struct_id"].astype(str).to_numpy()[perm]
    for a in range(0, len(sid), 100000):
        h.update(("\n".join(sid[a:a + 100000]) + "\n").encode())
    return h.hexdigest()


def write_store(pool, perm, sdir, fp, osha, info):
    """decision FTF-store: the whole filtered pool, written into a temp dir and renamed (atomic, idempotent)."""
    import pandas as pd
    t0 = time.time()
    P = pool.P
    n = len(perm)
    sdir = Path(sdir)
    if sdir.exists():                                   # incomplete leftover (never expected: the dir is renamed last)
        os.replace(sdir, sdir.with_name(f"{sdir.name}.bad_{time.strftime('%Y%m%d_%H%M%S')}"))
    tmp = sdir.with_name(f".{sdir.name}.tmp{os.getpid()}")
    F._mkdir(tmp / "shards")
    ii = P["i"].to_numpy()[perm]
    nat = P["n_atoms"].to_numpy()[perm].astype(np.int64)
    if not n or nat.min() < 1 or nat.max() > 32000:
        M.stop(f"store: {n} structures, atom counts {nat.min() if n else None}..{nat.max() if n else None}")
    shard = (np.arange(n) // SHARD_N).astype(np.int16)
    aoff = np.zeros(n, np.int64)
    shards, elems = [], set()
    n_sh = (n + SHARD_N - 1) // SHARD_N
    for s in range(n_sh):
        g0, g1 = s * SHARD_N, min(n, (s + 1) * SHARD_N)
        na = nat[g0:g1]
        off = np.concatenate([[0], np.cumsum(na)[:-1]]).astype(np.int64)
        aoff[g0:g1] = off
        A = int(na.sum())
        Z = np.empty(A, np.int8)
        X = np.empty((A, 3), np.float32)
        Fr = np.empty((A, 3), np.float32)
        for k in range(g1 - g0):
            a = pool.atoms[int(ii[g0 + k])]
            o, m = int(off[k]), int(na[k])
            if len(a) != m:
                M.stop(f"store: structure {P['struct_id'].iat[int(perm[g0 + k])]} has {len(a)} atoms, table {m}")
            Z[o:o + m] = a.numbers
            X[o:o + m] = a.positions
            Fr[o:o + m] = a.arrays["REF_forces"]
        if not (np.isfinite(X).all() and np.isfinite(Fr).all()):
            M.stop(f"store: shard {s} has non-finite positions / forces")
        elems |= {int(z) for z in np.unique(Z)}
        d = tmp / "shards" / f"s{s:03d}"
        d.mkdir(parents=True, exist_ok=True)
        nb = 0
        for nm, arr in (("Z", Z), ("pos", X), ("frc", Fr)):
            np.save(d / f"{nm}.npy", arr)
            nb += (d / f"{nm}.npy").stat().st_size
        shards.append(dict(shard=s, n=int(g1 - g0), n_atoms=A, bytes=nb))
        print(f"[store] shard {s + 1}/{n_sh}: {g1 - g0} structures, {A} atoms, {nb / 1e6:.1f} MB "
              f"({time.time() - t0:.0f} s)", flush=True)
    ts = F.ts_near_mask(P).to_numpy().astype(bool)[perm]
    np.savez(tmp / "structs.npz", nat=nat.astype(np.int16), charge=P["charge"].to_numpy()[perm].astype(np.int8),
             e_fit=P["e_fit"].to_numpy(np.float64)[perm], e_eh=P["e_eh"].to_numpy(np.float64)[perm], ts_near=ts,
             tier=P["tier"].to_numpy()[perm].astype(np.int8), g4dup=P["g4dup"].to_numpy(bool)[perm],
             shard=shard, aoff=aoff, pool_row=perm.astype(np.int64))
    Q = P.iloc[perm]
    tab = pd.DataFrame(dict(
        sid=np.arange(n, dtype=np.int64), pool_row=perm.astype(np.int64),
        struct_id=Q["struct_id"].astype(str).to_numpy(), kind=Q["kind"].astype(str).to_numpy(),
        source=Q["source"].astype(str).to_numpy(), input=Q["input"].astype(str).to_numpy(),
        formula=Q["formula"].astype(str).to_numpy(), n_atoms=Q["n_atoms"].to_numpy().astype(np.int16),
        rxn=[M.tag_list(x) for x in Q["rxn"]], folds=[M.tag_list(x) for x in Q["folds"]],
        dipoles=[M.tag_list(x) for x in Q["dipoles"]], d0cls=[M.tag_list(x) for x in Q["d0cls"]],
        d0id=Q["d0id"].to_numpy().astype(np.int64), jcore=Q["jcore"].astype(str).to_numpy(),
        grp=Q["grp"].astype(str).to_numpy(), tier=Q["tier"].to_numpy().astype(np.int8),
        g4dup=Q["g4dup"].to_numpy().astype(bool), hold_al=Q["hold_al"].to_numpy().astype(bool),
        dform=Q["dform"].to_numpy().astype(np.float64), fmax=Q["fmax"].to_numpy().astype(np.float64),
        charge=Q["charge"].to_numpy().astype(np.int8), e_eh=Q["e_eh"].to_numpy(np.float64),
        e_fit=Q["e_fit"].to_numpy(np.float64), ts_near=ts))
    M.write_parquet(tmp / "table.parquet", tab)
    del Q, tab
    nbytes = sum(x["bytes"] for x in shards) + (tmp / "structs.npz").stat().st_size + \
        (tmp / "table.parquet").stat().st_size
    meta = dict(fingerprint=fp, complete=True, n=int(n), n_shards=int(n_sh), shard_n=SHARD_N, perm_seed=PERM_SEED,
                order_sha256=osha, elements=sorted(elems), n_atoms_total=int(nat.sum()),
                n_atoms=F._q(nat, (0.0, 0.5, 1.0)), shards=shards, bytes=int(nbytes),
                bytes_per_structure=round(nbytes / max(1, n), 1),
                counts=dict(by_input=dict(Counter(P["input"])), by_kind=dict(Counter(P["kind"])),
                            by_tier={str(k): int(v) for k, v in sorted(Counter(P["tier"]).items())},
                            n_ts_near=int(ts.sum())),
                arrays=dict(Z="int8 [atoms]", pos="float32 [atoms, 3] A", frc="float32 [atoms, 3] eV/A (REF_forces)",
                            structs="nat int16, charge int8, e_fit / e_eh float64 (eV / Eh), ts_near bool, tier int8, "
                                    "g4dup bool, shard int16, aoff int64 (atom offset in the shard), pool_row int64",
                            table="table.parquet row = store row id (sid)"),
                energy_keys=dict(e_fit="eV, REF_energy - sum ATOM_REF_EH x EH2EV (training frame, finetune FT-frame)",
                                 e_eh="Eh, absolute label level"),
                pool=dict(read=pool.read_info, n_read=pool.n_read,
                          **{k: v for k, v in pool.stats.items() if k not in ("dedupe",)},
                          dedupe={k: v for k, v in (pool.stats.get("dedupe") or {}).items() if k != "method"}),
                write_s=round(time.time() - t0, 1), decisions={k: DECISIONS[k] for k in ("FTF-store", "FTF-g4")},
                written=F._now(), **info, **_jobtag())
    M.write_json(tmp / "meta.json", meta)
    os.replace(tmp, sdir)
    print(f"[store] {sdir}: {n} structures, {n_sh} shards, {nbytes / 1e9:.2f} GB "
          f"({meta['bytes_per_structure']} B / structure, {time.time() - t0:.0f} s, peak RSS {F._rss_gib()} GiB)",
          flush=True)
    return meta


class Store:
    """read side of the store: per-structure arrays in RAM (~45 B / structure), atom arrays memory-mapped per shard."""

    def __init__(self, sdir):
        self.dir = Path(sdir)
        self.meta = M.read_json(self.dir / "meta.json")
        S = np.load(self.dir / "structs.npz")
        self.nat = S["nat"].astype(np.int64)
        self.charge = S["charge"].astype(np.float32)
        self.e_fit = S["e_fit"].astype(np.float64)
        self.e_eh = S["e_eh"].astype(np.float64)
        self.ts_near = S["ts_near"].astype(bool)
        self.shard = S["shard"].astype(np.int64)
        self.aoff = S["aoff"].astype(np.int64)
        self.n = len(self.nat)
        self.n_shards = int(self.meta["n_shards"])
        self.shard_n = int(self.meta["shard_n"])
        self._mm = {}
        if self.n != int(self.meta["n"]):
            M.stop(f"{self.dir}: structs.npz has {self.n} rows, meta {self.meta['n']}")

    def mm(self, s):
        s = int(s)
        if s not in self._mm:
            d = self.dir / "shards" / f"s{s:03d}"
            self._mm[s] = tuple(np.load(d / f"{k}.npy", mmap_mode="r") for k in ("Z", "pos", "frc"))
        return self._mm[s]

    def per_struct(self, table):
        """per store row: sum over its atoms of table[Z] (one pass over every shard's Z)."""
        table = np.asarray(table, np.float64)
        out = np.zeros(self.n)
        for s in range(self.n_shards):
            g0, g1 = s * self.shard_n, min(self.n, (s + 1) * self.shard_n)
            Z = np.asarray(self.mm(s)[0]).astype(np.int64)
            out[g0:g1] = np.add.reduceat(table[Z], self.aoff[g0:g1])
        return out

    def atoms(self, rows):
        """(rows in gather order, ase.Atoms list with info total_charge / E_fit, arrays REF_forces) for small subsets."""
        from ase import Atoms
        ch = Chunk(self, rows)
        out = []
        for j in range(len(ch.rows)):
            s, n = int(ch.start[j]), int(ch.nat[j])
            a = Atoms(numbers=ch.Z[s:s + n].astype(np.int64), positions=ch.X[s:s + n].astype(np.float64))
            a.info["total_charge"] = float(self.charge[ch.rows[j]])
            a.info["E_fit"] = float(self.e_fit[ch.rows[j]])
            a.arrays["REF_forces"] = ch.Fr[s:s + n].astype(np.float64)
            out.append(a)
        return ch.rows, out


class Chunk:
    """the atoms of a set of store rows gathered into RAM (rows reordered by shard, stable); batches are built from it."""

    def __init__(self, store, rows):
        self.store = store
        rows = np.asarray(rows, np.int64)
        rows = rows[np.argsort(store.shard[rows], kind="stable")]
        self.rows = rows
        self.nat = store.nat[rows]
        self.start = (np.cumsum(self.nat) - self.nat).astype(np.int64)
        Zs, Xs, Fs = [], [], []
        sh = store.shard[rows]
        for s in np.unique(sh):
            r = rows[sh == s]
            Z, X, Fr = store.mm(int(s))
            ai = _ranges(store.aoff[r], store.nat[r])
            Zs.append(np.asarray(Z[ai]))
            Xs.append(np.asarray(X[ai]))
            Fs.append(np.asarray(Fr[ai]))
        e3 = np.zeros((0, 3), np.float32)
        self.Z = np.concatenate(Zs) if Zs else np.zeros(0, np.int8)
        self.X = np.concatenate(Xs) if Xs else e3
        self.Fr = np.concatenate(Fs) if Fs else e3

    def batch(self, sel, shift):
        """(x, y, store rows) numpy dicts of chunk positions sel, padded to the largest structure (numbers 0)."""
        sel = np.asarray(sel, np.int64)
        n = self.nat[sel]
        B, N = len(sel), int(n.max())
        bi = np.repeat(np.arange(B), n)
        ji = np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n)
        src = _ranges(self.start[sel], n)
        numbers = np.zeros((B, N), np.int64)
        numbers[bi, ji] = self.Z[src]
        coord = np.zeros((B, N, 3), np.float32)
        coord[bi, ji] = self.X[src]
        forces = np.zeros((B, N, 3), np.float32)
        forces[bi, ji] = self.Fr[src]
        r = self.rows[sel]
        x = dict(coord=coord, numbers=numbers, charge=self.store.charge[r].astype(np.float32))
        y = dict(energy=(self.store.e_fit[r] - shift[r]).astype(np.float32), forces=forces)
        return x, y, r


def train_batches(ch, bs, rng):
    m = len(ch.rows)
    if not m:
        return []
    p = rng.permutation(m)
    order = p[np.argsort(ch.nat[p], kind="stable")]
    bl = np.array_split(order, int(math.ceil(m / bs)))
    return [bl[i] for i in rng.permutation(len(bl))]


def eval_batches(ch, bs):
    """exact-size batches (no padding), as the SizeGroupedDataset validation loader / finetune.aim_predict."""
    if not len(ch.rows):
        return []
    order = np.argsort(ch.nat, kind="stable")
    cuts = np.flatnonzero(np.diff(ch.nat[order])) + 1
    out = []
    for g in np.split(order, cuts):
        out.extend(np.array_split(g, int(math.ceil(len(g) / bs))))
    return out


# ================================================================ data
def g4_state(no_g4):
    p = F.AL_INPUT[1]
    J = F._rj(G4_REPORT) or {}
    present = p.is_file() and p.stat().st_size > 0
    use = bool(present and J.get("complete") is True and not no_g4)
    if use:
        why = "G-4 used (kind al_ts)"
    elif no_g4:
        why = "G-4 not used (--no-g4)"
    elif not present:
        why = f"G-4 not used: {p} missing"
    else:
        why = f"G-4 not used: {G4_REPORT} not complete (complete={J.get('complete')})"
    return use, dict(used=use, why=why, extxyz=str(p), extxyz_present=present, report=str(G4_REPORT),
                     report_present=G4_REPORT.is_file(), report_complete=J.get("complete"),
                     n_selected=J.get("n_selected"), rule=DECISIONS["FTF-g4"])


def split_current(split):
    return F._rj(SPLIT_ROOT / split / "current.json")


def split_files_ok(sd):
    sd = Path(sd)
    return all((sd / f).is_file() for f in ("train.npy", "valid.npy", "test.npy", "meta.json"))


def geo_keys(pool):
    """per pool row: element-sequence key and formula key (tests.formula_key) for the geometric C-2."""
    TS = _tests()
    t0 = time.time()
    ii = pool.P["i"].to_numpy()
    seq, fk = [None] * len(ii), [None] * len(ii)
    for k, i in enumerate(ii):
        s = pool.atoms[int(i)].get_chemical_symbols()
        seq[k] = " ".join(s)
        fk[k] = TS.formula_key(s)
    print(f"[data] geo-C2 keys for {len(ii)} rows ({time.time() - t0:.0f} s)", flush=True)
    return seq, fk


def geo_c2(pool, R, rows, keys):
    """decision FTF-c2geo: the geometric C-2 of tests.py (_c2_one part c) on the train + valid pool rows."""
    TS = _tests()
    t0 = time.time()
    T = pool.T
    if R["mode"] == "fold":
        held = set(T.rxn_id[T.fold_T1 == R["val"]].astype(int))
    elif R["mode"] == "lodo":
        held = set(T.rxn_id[T.dipole == M.DIPOLES[R["val"]]].astype(int))
    else:
        return dict(applicable=False, pass_=True, note="all-data split: nothing is held out")
    # exact prefilter for the Kabsch half: for the same atom order, |d_ij(X) - d_ij(Y)| <= |a_i| + |a_j| (a = aligned
    # displacements) gives RMS over pairs <= 2 x RMSD, and sorting both spectra never increases their L2 distance, so
    # sorted-spectrum RMS > 2 x 0.10 A implies Kabsch RMSD > 0.10 A; Kabsch runs only below that bound (same hits).
    kb = 2.0 * float(TS.C2_RMSD_A) + 1e-9
    hts, hsp = {}, {}
    for r in sorted(held):
        p = M.label_ts(r)
        if p.is_file():
            s, x = M.read_xyz(p)
            x = np.asarray(x, float)
            sp = TS.dist_spectrum(x)
            hts.setdefault(" ".join(s), []).append((r, x, sp))
            hsp.setdefault(TS.formula_key(s), []).append((r, sp))
    hsp = {f: (np.array([r for r, _ in v]), np.stack([d for _, d in v])) for f, v in hsp.items()}
    hts = {s: ([r for r, _, _ in v], [x for _, x, _ in v], np.stack([d for _, _, d in v])) for s, v in hts.items()}
    seq, fk = keys
    ii = pool.P["i"].to_numpy()
    rxn = pool.P["rxn"].to_numpy()
    kind = pool.P["kind"].to_numpy()
    sid = pool.P["struct_id"].to_numpy()
    hits, n_k, n_s, n_kabsch = [], 0, 0, 0
    for k in rows:
        k = int(k)
        s, f = seq[k], fk[k]
        a_in, s_in = s in hts, f in hsp
        if not (a_in or s_in):
            continue
        X = np.asarray(pool.atoms[int(ii[k])].positions, float)
        spx = TS.dist_spectrum(X)
        hit = {}
        if a_in:
            n_k += 1
            rr, QQ, Hk = hts[s]
            vk = np.sqrt(((Hk - spx) ** 2).mean(axis=1))
            for j in np.flatnonzero(vk <= kb):
                n_kabsch += 1
                d = TS.kabsch_rmsd(X, QQ[j])
                if d <= TS.C2_RMSD_A:
                    hit[int(rr[j])] = dict(rmsd_A=round(d, 4))
        if s_in:
            n_s += 1
            rid, Hs = hsp[f]
            v = np.sqrt(((Hs - spx) ** 2).mean(axis=1))
            for j in np.flatnonzero(v <= TS.C2_SPEC_A):
                hit.setdefault(int(rid[j]), {})["spectrum_rms_A"] = round(float(v[j]), 4)
        for r, h in hit.items():
            hits.append(dict(struct_id=str(sid[k]), kind=str(kind[k]), held_rxn=r,
                             rxn_ids=sorted(int(x) for x in rxn[k])[:5], row=k, **h))
    return dict(applicable=True, rule=DECISIONS["FTF-c2geo"], rmsd_A=TS.C2_RMSD_A, spectrum_A=TS.C2_SPEC_A,
                hit_jobs=sorted({src_job(h["struct_id"]) for h in hits}),
                n_held_rxns=len(held), n_held_label_ts=sum(len(v[0]) for v in hts.values()), n_checked=int(len(rows)),
                n_kabsch_structures=n_k, n_kabsch_calls=n_kabsch,
                kabsch_prefilter=f"sorted-spectrum RMS <= {kb:.3f} A (= 2 x {TS.C2_RMSD_A} A, exact bound)",
                n_spectrum_structures=n_s, n_geom_hits=len(hits),
                n_geom_hits_untagged=sum(1 for h in hits if h["held_rxn"] not in h["rxn_ids"]), geom_hits=hits[:20],
                wall_s=round(time.time() - t0, 1), pass_=not hits)


def src_job(struct_id):
    """source job of a structure id: 'N:T01620:<file>:<frame>' -> 'N:T01620' (first two ':' fields)."""
    return ":".join(str(struct_id).split(":")[:2])


def geo_drop(pool, R, sets, geo, keys, max_rounds=3):
    """--geo-c2 drop (lead 11-05 22:1x; user rule 11-05 '누수가 0이 아니면 제거 후 재학습'): remove from train and valid
    EVERY structure of the source job of each geometric C-2 hit (whole Nurion job, not only the hit frames, so the
    neighbouring frames of the same trajectory go too), then re-check; up to max_rounds. Returns the final geo dict
    with a 'dropped' record; pass_ stays False if hits remain (the split is then not written, as with stop)."""
    jobs_all = pool.P["struct_id"].astype(str).map(src_job).to_numpy()
    rounds = []
    for rd in range(max_rounds):
        if geo.get("pass_") is not False:
            break
        hj = set(geo["hit_jobs"])
        drop = np.flatnonzero(np.isin(jobs_all, list(hj)))
        n0 = {p: len(sets[p]) for p in ("train", "valid")}
        for p in ("train", "valid"):
            sets[p] = sets[p][~sets[p].index.isin(drop)]
        rounds.append(dict(round=rd, hit_jobs=sorted(hj), n_hits=geo["n_geom_hits"],
                           n_dropped={p: n0[p] - len(sets[p]) for p in ("train", "valid")}))
        rows = np.concatenate([sets["train"].index.to_numpy(), sets["valid"].index.to_numpy()])
        geo = geo_c2(pool, R, rows, keys)
    geo["dropped"] = dict(rule="--geo-c2 drop: all train/valid structures of each hit's source job removed, re-checked",
                          rounds=rounds, n_train_after=len(sets["train"]), n_valid_after=len(sets["valid"]))
    return geo


def write_split(R, fp, sets, inv, bm, geo, sdir, store_fp, g4meta, geo_mode):
    base = SPLIT_ROOT / R["split"]
    sd = base / fp[:16]
    if sd.exists():                                      # --force rebuild: the old split is moved aside, not deleted
        os.replace(sd, base / f"{fp[:16]}.old_{time.strftime('%Y%m%d_%H%M%S')}")
    tmp = base / f".{fp[:16]}.tmp{os.getpid()}"
    F._mkdir(tmp)
    idx, files = {}, {}
    for nm in ("train", "valid", "test"):
        a = np.sort(inv[sets[nm].index.to_numpy()]).astype(np.int32)
        idx[nm] = a
        np.save(tmp / f"{nm}.npy", a)
        files[nm] = dict(path=str(sd / f"{nm}.npy"), n=int(len(a)), bytes=(tmp / f"{nm}.npy").stat().st_size)
    inter = {f"{a}&{b}": int(len(np.intersect1d(idx[a], idx[b]))) for a, b in
             (("train", "valid"), ("train", "test"), ("valid", "test"))}
    if any(inter.values()):
        M.stop(f"{R['split']}: index sets overlap {inter}")
    meta = dict(bm, files=files, fingerprint=fp, full=True, store_dir=str(sdir), store_fingerprint=store_fp, g4=g4meta,
                leakage_geo=geo, geo_c2_mode=geo_mode, index_overlap=inter, index_dtype="int32 store row ids, sorted",
                decisions_full={k: DECISIONS[k] for k in ("FTF-scope", "FTF-split", "FTF-g4", "FTF-c2geo")},
                code_full=CODE_VERSION, written=F._now())
    M.write_json(tmp / "meta.json", meta)
    os.replace(tmp, sd)
    M.write_json(base / "current.json", dict(fingerprint=fp, dir=str(sd), written=F._now()))
    return meta


def write_data_json(R, meta, build):
    pool = meta.get("pool") or {}
    out = dict(run=R["run"], split=R["split"], split_dir=str(SPLIT_ROOT / R["split"] / meta["fingerprint"][:16]),
               run_info=R, mode="full", fingerprint=meta["fingerprint"], store_dir=meta.get("store_dir"),
               store_fingerprint=meta.get("store_fingerprint"), g4=meta.get("g4"),
               inputs_complete=meta["inputs_complete"], counts=meta["counts"], counts_precap=meta.get("counts_precap"),
               cap={k: v for k, v in (meta.get("cap") or {}).items() if k != "rule"}, valid=meta["valid"],
               leakage_C2=meta["leakage_C2"], leakage_geo=meta.get("leakage_geo"), index_files=meta["files"],
               inputs={k: dict(present=v["present"], sha256=v.get("sha256")) for k, v in meta["inputs"].items()},
               filters=pool.get("filters"), dedupe=pool.get("dedupe"), tag_check=pool.get("tag_check"),
               dropped_leak_kinds=pool.get("dropped_leak_kinds"),
               d0_holdout={k: v for k, v in (pool.get("d0_holdout") or {}).items() if k != "rule"},
               d0_holdout_sha256=(pool.get("d0_holdout") or {}).get("sha256"), build=build,
               decisions={k: DECISIONS[k] for k in ("FTF-scope", "FTF-split", "FTF-g4", "FTF-c2geo")},
               written=F._now(), **_jobtag())
    M.write_json(F.RES_FT / f"{R['run']}_data.json", out)


def cmd_data(runs, o):
    t0 = time.time()
    Rs = [full_run(r) for r in (runs or [])] + ([full_run(r) for r in FULL_RUNS] if o.all else [])
    if not Rs:
        raise SystemExit("data: give run names or --all")
    if o.geo_c2 != "off":
        TS = _tests()                                    # fail fast: the geometric C-2 helpers exist
        for nm in ("formula_key", "dist_spectrum", "kabsch_rmsd", "C2_RMSD_A", "C2_SPEC_A"):
            getattr(TS, nm)
    g4, g4meta = g4_state(o.no_g4)
    print(f"[data] {g4meta['why']}", flush=True)
    st, complete, gmeta = F.inputs_state(g4)
    sp = dict(code=CODE_VERSION, finetune_code=F.CODE_VERSION, fmax_cap=o.fmax_cap, dedupe_decimals=F.DEDUPE_DEC,
              atom_ref_eh=F.ATOM_REF_EH, leak_drop_kinds=list(F.LEAK_DROP_KINDS),
              ts_near=[list(F.TS_NEAR_ALWAYS), list(F.TS_NEAR_IF_CLOSE), F.TS_NEAR_DFORM_A], g4_used=g4,
              shard_n=SHARD_N, perm_seed=PERM_SEED)
    used = {n: v.get("sha256") for n, v in st.items() if v["present"] and (n != "G4" or g4)}
    store_fp = F._sha_obj(dict(kind="ft_full_store", inputs=used, params=sp))
    sdir = STORE_ROOT / f"store_{store_fp[:16]}"
    # geo_c2 mode is NOT part of the split fingerprint (membership does not depend on it; it is recorded in the meta),
    # so a `--geo-c2 record` rerun after a geo STOP rebuilds only the blocked splits
    params = dict(code=CODE_VERSION, finetune_code=F.CODE_VERSION, valid_frac=o.valid_frac, g4_used=g4,
                  cap=dict(train=0, valid=0, test=int(o.cap_test), tier2_frac=F.CAP_TIER2_FRAC,
                           seed_base=M.SEED + F.CAP_SEED_OFF,
                           tiers="1 non-Nurion | 2 Nurion nur_ts_final or TS-near nur_ TS kind | 3 other Nurion",
                           sampler="round-robin over job ids (rr_pick); train / valid not capped (FTF-split)"))
    splits = {}
    for R in Rs:
        splits.setdefault(R["split"], dict(R, al=g4))
    todo = []
    for s, R in splits.items():
        fp = F._sha_obj(dict(split=s, store=store_fp, params=params, mode=R["mode"], val=R["val"],
                             valid_seed=R["valid_seed"], cap_seed=R["cap_seed"]))
        cur = split_current(s)
        ok = bool(cur and cur.get("fingerprint") == fp and split_files_ok(cur["dir"]))
        if ok and o.geo_c2 != "off" and R["mode"] != "all":     # a split built with --geo-c2 off is re-checked
            lg = (F._rj(Path(cur["dir"]) / "meta.json") or {}).get("leakage_geo") or {}
            ok = lg.get("pass_") is not None
        if o.force or not ok:
            todo.append((s, R, fp))
    need_store = not store_ok(sdir, store_fp)
    print(f"[data] store {sdir.name} {'to write' if need_store else 'up to date'}; splits to build "
          f"{[s for s, _, _ in todo]}", flush=True)
    built, fails, t_pool = {}, {}, None
    if todo or need_store:
        pool = F.Pool(g4, o.fmax_cap)
        F.nur_stream_check(pool, st)
        t_pool = round(time.time() - t0, 1)
        P = pool.P
        perm = np.random.default_rng(PERM_SEED).permutation(len(P))
        osha = order_sha(P, perm)
        if need_store:
            write_store(pool, perm, sdir, store_fp, osha,
                        dict(inputs=st, inputs_complete=complete, g_report=gmeta, params=sp, g4=g4meta))
        else:
            sm = M.read_json(sdir / "meta.json")
            if int(sm.get("n", -1)) != len(P) or sm.get("order_sha256") != osha:
                M.stop(f"{sdir}: same fingerprint but the pool differs ({sm.get('n')} vs {len(P)} rows, order sha "
                       f"{str(sm.get('order_sha256'))[:12]} vs {osha[:12]}) - Pool not deterministic?")
        inv = np.empty(len(P), np.int64)
        inv[perm] = np.arange(len(P), dtype=np.int64)
        keys = geo_keys(pool) if (o.geo_c2 != "off" and any(R["mode"] != "all" for _, R, _ in todo)) else None
        for s, R, fp in todo:
            ts = time.time()
            sets = {}
            bm = F.build_split(pool, s, R, o.valid_frac, fp, st, complete, gmeta, params, write=False, sets_out=sets)
            if not len(sets["train"]) or not len(sets["valid"]):
                M.not_ready(f"{s}: empty train ({len(sets['train'])}) or validation ({len(sets['valid'])}) set")
            if keys is not None and R["mode"] != "all":
                rows = np.concatenate([sets["train"].index.to_numpy(), sets["valid"].index.to_numpy()])
                geo = geo_c2(pool, R, rows, keys)
                if geo.get("pass_") is False and o.geo_c2 == "drop":
                    geo = geo_drop(pool, R, sets, geo, keys)
                    print(f"[data] {s}: geo-C2 drop {geo['dropped']['rounds']} -> hits {geo.get('n_geom_hits')}",
                          flush=True)
            else:
                geo = dict(applicable=R["mode"] != "all", pass_=None if R["mode"] != "all" else True,
                           note="--geo-c2 off" if o.geo_c2 == "off" else "all-data split: nothing is held out")
            print(f"[data] {s}: train {len(sets['train'])} valid {len(sets['valid'])} test {len(sets['test'])} | "
                  f"C-2 pass {bm['leakage_C2']['pass_']} | geo-C2 hits {geo.get('n_geom_hits')} "
                  f"({geo.get('wall_s')} s)", flush=True)
            if geo.get("pass_") is False and o.geo_c2 in ("stop", "drop"):
                fails[s] = dict(n_geom_hits=geo["n_geom_hits"], first=geo["geom_hits"][:3])
                for R2 in Rs:
                    if R2["split"] == s:
                        M.write_json(F.RES_FT / f"{R2['run']}_data.json",
                                     dict(run=R2["run"], split=s, mode="full", status="STOP geo-C2", leakage_geo=geo,
                                          leakage_C2=bm["leakage_C2"], written=F._now(), **_jobtag()))
                continue
            built[s] = write_split(R, fp, sets, inv, bm, geo, sdir, store_fp, g4meta, o.geo_c2)
            built[s]["_build_s"] = round(time.time() - ts, 1)
            del sets
    build = dict(wall_s=round(time.time() - t0, 1), pool_s=t_pool, peak_rss_gib=F._rss_gib(), store=str(sdir),
                 store_written_here=need_store, store_gb=round(_dir_gb(sdir), 3) if sdir.is_dir() else None)
    for R in Rs:
        if R["split"] in fails:
            continue
        cur = split_current(R["split"])
        meta = built.get(R["split"]) or (M.read_json(Path(cur["dir"]) / "meta.json") if cur else None)
        if meta is None:
            continue
        write_data_json(R, meta, dict(build, split_built_here=R["split"] in built))
    print(json.dumps(dict(store=str(sdir), g4=g4meta["why"], inputs_complete=complete, built=sorted(built),
                          geo_c2_stop=fails, wall_s=build["wall_s"], peak_rss_gib=build["peak_rss_gib"],
                          store_gb=build["store_gb"])), flush=True)
    if fails:
        msg = f"geometric C-2: held-out label TS geometry in train / valid of {sorted(fails)}: {fails}"
        ok_splits = sorted(s for s in splits if s not in fails and split_current(s))
        if not ok_splits:
            M.stop(msg)
        # some splits are clean: exit 0 so the afterok trainings of the clean splits run; every blocked run has a
        # results/ft/<run>_data.json with status "STOP geo-C2" and its trainer STOPs (exit 3) at start
        M.write_json(F.RES_FT / "ft_full_geo_c2_stop.json", dict(blocked=fails, clean=ok_splits, written=F._now(),
                                                                 **_jobtag()))
        print(f"[STOP-PARTIAL] {msg}; clean splits {ok_splits} written; the blocked runs' trainers STOP at start "
              f"(rerun `ft_full.py data <run> --geo-c2 record` if the hits are judged harmless)", file=sys.stderr,
              flush=True)
    return 0


def _dir_gb(p):
    return F._du_gb(p)


# ================================================================ train
def fw_decision():
    """decision FTF-fw: (fw, source) from the FT-abl decision, else 1.0."""
    j = F._rj(ABL_CHOICE)
    if j:
        if j.get("fw") is not None:
            return float(j["fw"]), dict(source=str(ABL_CHOICE), choice=j.get("choice"))
        if j.get("choice") in ABL_FW:
            return float(ABL_FW[j["choice"]]), dict(source=str(ABL_CHOICE), choice=j["choice"])
        M.stop(f"{ABL_CHOICE}: neither 'fw' nor a known 'choice' ({sorted(ABL_FW)}): {j}")
    st = F._rj(PIPE_STATE) or {}
    p = st.get("params") or {}
    ch = p.get("ft_abl_choice")
    if ch and (p.get("ft_abl_cand") or ABL_CAND) == ABL_CAND:
        a = [str(x) for x in (p.get("ft_train_args") or [])]
        fw = float(a[a.index("--fw") + 1]) if "--fw" in a else ABL_FW.get(ch)
        if fw is not None:
            return float(fw), dict(source=f"{PIPE_STATE} params", choice=ch, ft_train_args=a)
    return float(H["fw_default"]), dict(source="default 1.0 (no FT-abl decision found)")


def _lock_me():
    return os.environ.get("SLURM_JOB_ID") or f"pid{os.getpid()}"


def _unlock(rd):
    """release lock.json on every exit of cmd_train (normal, pause, STOP); only when this job still holds it."""
    p = rd / "lock.json"
    j = F._rj(p)
    if j and str(j.get("job")) == _lock_me() and not j.get("released"):
        M.write_json(p, dict(j, released=True, released_t=time.time(), released_time=F._now()))


def _lock(rd):
    p = rd / "lock.json"
    me = _lock_me()
    j = F._rj(p)
    if j and not j.get("released") and str(j.get("job")) != me and \
            time.time() - float(j.get("t") or 0) < LOCK_STALE_S:
        M.not_ready(f"{rd} is locked by job {j.get('job')} on {j.get('host')} (heartbeat "
                    f"{time.time() - float(j.get('t') or 0):.0f} s ago): another training of this run is live")

    def beat():
        M.write_json(p, dict(job=me, host=os.uname().nodename, t=time.time(), time=F._now()))
    beat()
    return beat


def e0_rows(store, rows, nmax, seed=M.SEED):
    """finetune.e0_sample on store rows: every structure with an element present in < 2000 training structures, the
    rest a seeded random fill up to nmax."""
    rows = np.asarray(rows, np.int64)
    if len(rows) <= nmax:
        return np.sort(rows)
    keep = np.zeros(len(rows), bool)
    cnt = {}
    for z in store.meta["elements"]:
        t = np.zeros(128)
        t[int(z)] = 1.0
        has = store.per_struct(t)[rows] > 0
        cnt[int(z)] = int(has.sum())
        if cnt[int(z)] < 2000:
            keep |= has
    rest = np.flatnonzero(~keep)
    rng = np.random.default_rng(seed)
    pick = rng.choice(rest, size=max(0, min(len(rest), nmax - int(keep.sum()))), replace=False)
    return np.sort(rows[np.concatenate([np.flatnonzero(keep), pick]).astype(np.int64)])


def _dev(d, dev):
    import torch
    return {k: torch.from_numpy(np.ascontiguousarray(v)).to(dev, non_blocking=True) for k, v in d.items()}


def evaluate(fm, ch, batches, shift, dev, loss_fn=None, tick=None):
    """force MAE (all / TS-near, kcal/mol/A) and energy MAE (kcal/mol, training frame) over the batches of a chunk;
    tick() (lock heartbeat + budget) is called every 50 batches."""
    m = len(ch.rows)
    fsum, fcnt, eabs, seen = np.zeros(m), np.zeros(m), np.zeros(m), np.zeros(m, bool)
    vl, nb = 0.0, 0
    for ib, sel in enumerate(batches):
        if tick is not None and ib and ib % 50 == 0:
            tick()
        x, y, _ = ch.batch(sel, shift)
        x, y = _dev(x, dev), _dev(y, dev)
        pred = fm(x)
        fp, ep = pred["forces"].detach(), pred["energy"].detach()
        if loss_fn is not None:
            vl += float(loss_fn(pred, y)["loss"].detach().item())
            nb += 1
        mask = (x["numbers"] > 0).unsqueeze(-1).to(fp.dtype)
        fsum[sel] = ((fp - y["forces"]).abs() * mask).sum(dim=(1, 2)).double().cpu().numpy()
        eabs[sel] = (ep.double() - y["energy"].double()).abs().cpu().numpy()
        fcnt[sel] = 3.0 * ch.nat[sel]
        seen[sel] = True
    ts = ch.store.ts_near[ch.rows] & seen

    def fmae(mk):
        c = fcnt[mk].sum()
        return float(fsum[mk].sum() / c * F.EV2KCAL) if c > 0 else None

    return dict(n=int(seen.sum()), n_ts_near=int(ts.sum()), f_all=fmae(seen), f_ts=fmae(ts),
                e=float(eabs[seen].mean() * F.EV2KCAL) if seen.any() else None, loss=vl / max(1, nb),
                finite=bool(np.isfinite(fsum[seen]).all() and np.isfinite(eabs[seen]).all()))


class EMA:
    """decision FTF-ema: ema <- 0.99 ema + 0.01 model after every optimizer step (all parameters)."""

    def __init__(self, model, ema_model, decay):
        self.w = 1.0 - float(decay)
        self.src = list(model.parameters())
        self.dst = list(ema_model.parameters())
        if len(self.src) != len(self.dst) or any(a.shape != b.shape for a, b in zip(self.src, self.dst)):
            M.stop("EMA: model and EMA model parameters differ")

    def update(self):
        import torch
        with torch.no_grad():
            try:
                torch._foreach_lerp_(self.dst, self.src, self.w)
            except (AttributeError, RuntimeError, TypeError):
                for d, s in zip(self.dst, self.src):
                    d.lerp_(s, self.w)


# ================================================================ selection metric (FTF-sel / FTF-export, user 11-06)
def _num(v):
    """finite float or None."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def sel_metric(rec):
    """THE selection metric of one epoch record (best checkpoint, ReduceLROnPlateau, early stop): validation TS-near
    force MAE with EMA weights = rec['mae_f_ts_kcal_A']; None when the record lacks it."""
    return _num((rec or {}).get(SEL_KEY))


def all_metric(rec):
    """validation force MAE over all validation structures (recorded only since 11-06)."""
    return _num((rec or {}).get("mae_f_kcal_A"))


def hist_best(hist, fn):
    """(epoch, value) of the first strict minimum of fn over the history records (strictly lower = the trainer's
    improvement rule); records without a value are skipped; (None, inf) when none has one."""
    be, bm = None, math.inf
    for h in hist or []:
        m = fn(h)
        if m is not None and h.get("epoch") is not None and m < bm:
            be, bm = int(h["epoch"]), m
    return be, bm


def ts_hist(hist):
    """{epoch: TS-near metric} of the history records that carry it."""
    return {int(h["epoch"]): sel_metric(h) for h in hist or []
            if h.get("epoch") is not None and sel_metric(h) is not None}


def _cpu_sd(sd):
    return None if sd is None else {k: t.detach().to("cpu", copy=True) for k, t in sd.items()}


def fresh_ts_state():
    """selection state of a run that starts with the new code (TS-near selection from epoch 0)."""
    return dict(fmt="fresh", bts_epoch=None, bts_metric=math.inf, bts_sd=None, bsw_epoch=None, bsw_metric=math.inf,
                bsw_sd=None, cands=[],
                sel=dict(selection_metric=SEL_NAME, metric_key=SEL_KEY, switched_at_epoch=0, previous_metric=None,
                         ts_best_weights="kept (TS-near selection from epoch 0)"))


def ts_state(ck):
    """selection state from a last.pt dict (decision FTF-sel). New format (ckpt_format SEL_CODE): its TS fields. Old
    format (written before 11-06): migrated from the history records; the old best_ema and the last-epoch EMA are kept
    as candidates (FTF-export third rule). Weights are CPU copies. Pure: reads ck only (train and export use it)."""
    last = int(ck["epoch"])
    if ck.get("ckpt_format") == SEL_CODE:
        bm, sm = _num(ck.get("best_ts_metric")), _num(ck.get("best_ts_sw_metric"))
        st = dict(fmt="new", bts_epoch=ck.get("best_ts_epoch"), bts_metric=math.inf if bm is None else bm,
                  bts_sd=_cpu_sd(ck.get("best_ts_ema")), bsw_epoch=ck.get("best_ts_sw_epoch"),
                  bsw_metric=math.inf if sm is None else sm, bsw_sd=None, cands=[],
                  sel=dict(ck.get("selection") or {}))
        if st["bts_sd"] is not None and st["bts_epoch"] is not None and st["bts_epoch"] == st["bsw_epoch"]:
            st["bsw_sd"] = st["bts_sd"]                  # same epoch: one CPU copy (saved once by torch.save)
        else:
            st["bsw_sd"] = _cpu_sd(ck.get("best_ts_sw_ema"))
        return st
    hist = ck.get("hist") or []
    warm, pat = int(H["warmup_epochs"]), int(H["plateau_patience"])
    be, bm = hist_best(hist, sel_metric)
    ob_e = ck.get("best_epoch")
    ob_e = None if ob_e is None else int(ob_e)
    ob_sd, last_sd = _cpu_sd(ck.get("best_ema")), _cpu_sd(ck.get("ema"))
    if be is None:
        sd, note = None, "not retained (no epoch record has the TS-near metric)"
    elif ob_e == be and ob_sd is not None:
        sd, note = ob_sd, f"retained (epoch {be} = the old all-structure best_ema)"
    elif be == last and last_sd is not None:
        sd, note = last_sd, f"retained (epoch {be} = the last epoch, EMA weights of last.pt)"
    else:
        sd, note = None, f"not retained (epoch {be})"
    # ReduceLROnPlateau(mode min, threshold 0 rel): is_better(a, best) = a < best; num_bad_epochs counts the scheduler
    # steps since the best (the trainer steps it at epochs >= warmup) and the lr drops when it exceeds patience
    n_bad = 0 if be is None else min(pat, sum(1 for h in hist if h.get("epoch") is not None
                                              and be < int(h["epoch"]) and int(h["epoch"]) >= warm))
    try:
        lr = float(ck["opt"]["param_groups"][0]["lr"])
    except (KeyError, IndexError, TypeError, ValueError):
        lr = None
    sel = dict(selection_metric=SEL_NAME, metric_key=SEL_KEY, switched_at_epoch=last + 1, previous_metric=PREV_SEL,
               previous_best=dict(epoch=ob_e, metric=_num(ck.get("best_metric"))), ts_best_weights=note,
               migration=dict(rule="FTF-sel (old last.pt without TS fields)", last_epoch=last, n_records=len(hist),
                              n_records_with_metric=len(ts_hist(hist)), best_ts_epoch=be, best_ts_metric=_num(bm),
                              since_best=None if be is None else last - be, sched_best=_num(bm),
                              sched_num_bad_epochs=int(n_bad), sched_cooldown_counter=0, lr_kept=lr,
                              migrated=F._now()))
    return dict(fmt="old", bts_epoch=be, bts_metric=bm, bts_sd=sd, bsw_epoch=None, bsw_metric=math.inf, bsw_sd=None,
                cands=[("old all-structure best_ema", ob_e, ob_sd), ("last-epoch EMA", last, last_sd)], sel=sel)


def pick_export(TS, hist):
    """decision FTF-export: (state_dict, epoch, TS-near metric, rule) or None (nothing exportable)."""
    if TS.get("bts_sd") is not None and TS.get("bts_epoch") is not None:
        return TS["bts_sd"], int(TS["bts_epoch"]), _num(TS["bts_metric"]), "TS-near best (EMA weights of that epoch)"
    if TS.get("bsw_sd") is not None and TS.get("bsw_epoch") is not None:
        return (TS["bsw_sd"], int(TS["bsw_epoch"]), _num(TS["bsw_metric"]),
                f"TS-near best since the switch at epoch {(TS.get('sel') or {}).get('switched_at_epoch')} (the "
                f"historical TS-near best epoch {TS.get('bts_epoch')} is not retained)")
    th = ts_hist(hist)
    c = [(th[int(e)], int(e), sd, lab) for lab, e, sd in TS.get("cands") or []
         if sd is not None and e is not None and int(e) in th]
    if c:
        m, e, sd, lab = min(c, key=lambda t: (t[0], t[1]))
        return sd, e, m, (f"no epoch evaluated after the switch: the saved candidate with the lowest recorded TS-near "
                          f"metric ({lab}, epoch {e})")
    return None


def ts_summary(hist, TS=None):
    """compact TS-near metric history (printed by export / at resume)."""
    th = ts_hist(hist)
    be, bm = hist_best(hist, sel_metric)
    ae, am = hist_best(hist, all_metric)
    last = hist[-1] if hist else {}
    pts = sorted(set([e for e in th if e % 10 == 0] + sorted(th)[-10:]))
    out = dict(n_epochs=len(hist), n_with_ts_metric=len(th), best_ts_epoch=be, best_ts_metric=_num(bm),
               best_all_epoch=ae, best_all_metric=_num(am), last_epoch=last.get("epoch"),
               last_ts_metric=sel_metric(last), last_lr_next=last.get("lr_next"),
               since_best_ts=(int(last["epoch"]) - be) if last and be is not None else None,
               curve={str(e): round(th[e], 5) for e in pts})
    if TS is not None:
        out.update(selection=TS.get("sel"), best_ts_since_switch=dict(epoch=TS.get("bsw_epoch"),
                                                                      metric=_num(TS.get("bsw_metric"))))
    return out


def cmd_train(run, o):
    """lock.json is released on every exit (normal return, pause exit 2, STOP exit 3, exceptions, SIGTERM)."""
    held = []
    try:                                                 # scancel / wall SIGTERM -> SystemExit(143): the finally below
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))   # releases lock.json (as mlts_autode.py)
    except ValueError:                                   # not the main thread: keep the default handler
        pass
    try:
        return _cmd_train(run, o, held)
    finally:
        for rel in held:
            try:
                rel()
            except Exception as e:                       # noqa: BLE001 - never mask the real exit
                print(f"[train] WARNING: lock release failed: {e!r}", file=sys.stderr, flush=True)


def _cmd_train(run, o, held):
    import torch
    guard = F._guard_imports()
    from aimnet.modules import Forces
    from aimnet.train.loss import MTLoss
    R = full_run(run)
    run = R["run"]
    cand = F.canon_cand(o.cand)
    C = F.CANDIDATES[cand]
    if C["kind"] != "aimnet":
        raise SystemExit(f"ft_full trains AIMNet2 only (base candidate {ABL_CAND}), not {cand}")
    rd = F.FT / run
    out_json = F.RES_FT / f"{run}_train.json"
    model_out = F.model_path(R, C)
    cap_h = float(o.cap_gpu_h)
    # ---- pinned run key / split (FTF-resume)
    key_p = rd / "run_key.json"
    old = F._rj(key_p)
    if old and o.force:
        stale = rd.with_name(f"{rd.name}.stale_{time.strftime('%Y%m%d_%H%M%S')}")
        os.replace(rd, stale)
        print(f"[train] --force: previous run dir moved to {stale} (not deleted)", flush=True)
        old = None
    cur = split_current(R["split"])
    if old:
        sdir, sfp = Path(old["split_dir"]), old["split_fp"]
        if not cur or cur.get("fingerprint") != sfp:
            print(f"[train] WARNING: split {R['split']} was rebuilt after this run started (current "
                  f"{str((cur or {}).get('fingerprint'))[:16]}, pinned {sfp[:16]}); continuing on the pinned split "
                  f"(--force restarts on the current one)", flush=True)
    else:
        dj = F._rj(F.RES_FT / f"{run}_data.json") or {}
        if str(dj.get("status") or "").startswith("STOP"):
            M.stop(f"{run}: the data job blocked split {R['split']} ({dj.get('status')}; "
                   f"{(dj.get('leakage_geo') or {}).get('n_geom_hits')} geometric C-2 hits, see "
                   f"results/ft/{run}_data.json); rerun `ft_full.py data {run} --geo-c2 record` if judged harmless")
        if not cur:
            M.not_ready(f"split {R['split']} not built: run `ft_full.py data {run}` first")
        sdir, sfp = Path(cur["dir"]), cur["fingerprint"]
    smeta = F._rj(sdir / "meta.json")
    if not smeta or smeta.get("fingerprint") != sfp or not split_files_ok(sdir):
        M.stop(f"{run}: split dir {sdir} missing / incomplete / fingerprint differs")
    prev = F._rj(out_json)
    # FTF-sel: a run finished by the pre-11-06 code (no selection_metric in its train json) is re-evaluated: resume
    # from last.pt, the old metric-dependent early stop is set aside, the TS-near rule decides, the export is redone
    old_sel_done = bool(prev and prev.get("status") == "done" and prev.get("selection_metric") != SEL_NAME)
    if prev and prev.get("status") == "done" and model_out.is_file() and prev.get("data_fingerprint") == sfp \
            and not o.force and not old_sel_done:
        print(f"[train] {run} done already ({model_out}; stop: {prev.get('stop_reason')})", flush=True)
        return 0
    if old_sel_done and not o.force:
        print(f"[train] {run} was finished by the old selection code (stop: {prev.get('stop_reason')}); re-evaluated "
              f"under FTF-sel ({SEL_NAME})", flush=True)
    if not smeta.get("inputs_complete") and not o.allow_partial:
        M.not_ready(f"split {R['split']} was built from incomplete inputs; rebuild with `ft_full.py data` after the "
                    f"collects, or pass --allow-partial")
    if not (smeta.get("leakage_C2") or {}).get("pass_"):
        M.stop(f"{run}: split meta leakage_C2 pass_ is not true")
    if (smeta.get("leakage_geo") or {}).get("pass_") is False:
        print(f"[train] WARNING: geometric C-2 hits recorded for {R['split']} (built with --geo-c2 record): "
              f"{(smeta.get('leakage_geo') or {}).get('n_geom_hits')}", flush=True)
    if o.fw is not None:
        fw, fw_src = float(o.fw), dict(source="--fw")
    elif old:
        cfw, csrc = fw_decision()
        fw, fw_src = float(old["fw"]), dict(old.get("fw_source") or {}, pinned=True,
                                            current_decision=dict(fw=cfw, **csrc))
        if cfw != fw:
            print(f"[train] WARNING: the current FT-abl decision gives fw {cfw} ({csrc}); this run is pinned to fw "
                  f"{fw} (FTF-fw; --force restarts)", flush=True)
    else:
        fw, fw_src = fw_decision()
    key = dict(run=run, cand=cand, split=R["split"], split_fp=sfp, split_dir=str(sdir),
               store_fp=smeta["store_fingerprint"], store_dir=smeta["store_dir"], lr=H["lr"], ew=H["ew"], fw=fw,
               bs=H["batch"], ema=H["ema"])
    if old:
        diff = {k: (old.get(k), v) for k, v in key.items() if old.get(k) != v}
        if diff:
            M.stop(f"{run}: run key differs from the pinned {key_p}: {diff}; pass --force to restart (old dir kept)")
    F._mkdir(rd)
    if not old:
        M.write_json(key_p, {**_jobtag(), **key, "fw_source": fw_src, "written": F._now()})   # _jobtag has code / code_full
    beat = _lock(rd)
    held.append(lambda: _unlock(rd))
    if not torch.cuda.is_available():
        M.stop(f"{run}: no CUDA device")
    dev = "cuda"
    gpu = torch.cuda.get_device_name(0)
    torch.cuda.reset_peak_memory_stats()
    # ---- store + indices
    store = Store(smeta["store_dir"])
    if store.meta.get("fingerprint") != smeta["store_fingerprint"]:
        M.stop(f"{run}: store {store.dir} fingerprint differs from the split meta")
    idx = {nm: np.load(sdir / f"{nm}.npy").astype(np.int64) for nm in ("train", "valid", "test")}
    tr, va = idx["train"], idx["valid"]
    ov = {f"{a}&{b}": int(len(np.intersect1d(idx[a], idx[b]))) for a, b in
          (("train", "valid"), ("train", "test"), ("valid", "test"))}
    if any(ov.values()) or not len(tr) or not len(va) or max(int(tr.max()), int(va.max())) >= store.n:
        M.stop(f"{run}: bad index sets (overlap {ov}, n_train {len(tr)}, n_valid {len(va)}, store n {store.n})")
    if not bool(store.ts_near[va].any()):                # FTF-sel: the selection metric needs TS-near validation rows
        M.stop(f"{run}: no TS-near structure in the validation set: the selection metric {SEL_NAME} is undefined")
    bs = int(H["batch"])
    tr_sh = store.shard[tr]
    by_shard = {int(s): tr[tr_sh == s] for s in np.unique(tr_sh)}
    nb_epoch = int(sum(math.ceil(len(v) / bs) for v in by_shard.values()))
    warm_steps = int(H["warmup_epochs"]) * nb_epoch
    bud = F.Budget(rd, cap_h, cap_h)
    jpt = F.fetch(cand)
    # ---- released weights + parity (as train_aimnet)
    _r8, chk = store.atoms(tr[:8])
    jm = torch.jit.load(str(jpt), map_location=dev)
    sd0 = {k: v.detach().clone() for k, v in jm.state_dict().items()}
    m_par, lk = F.aim_build(C, sd0, d3="foundation", shift_double=True)
    m_par = m_par.to(dev).eval()
    e_j, f_j = F.aim_predict(jm, chk, dev, forces=True)
    e_b, f_b = F.aim_predict(m_par, chk, dev, forces=True)
    parity = dict(max_abs_dE_eV=float(np.abs(e_j - e_b).max()),
                  max_abs_dF_eV_A=float(max(np.abs(a - b).max() for a, b in zip(f_j, f_b))), load=lk)
    real_missing = [k for k in lk["missing"] if not k.endswith(("c6ab", "r4r2", "rcov", "cnmax"))]
    if parity["max_abs_dE_eV"] > 1e-3 or parity["max_abs_dF_eV_A"] > 1e-3 or real_missing:
        M.stop(f"AIMNet2 weight parity failed: {parity}")
    del m_par, jm
    beat()
    sd0["outputs.atomic_shift.shifts.weight"] = torch.zeros_like(sd0["outputs.atomic_shift.shifts.weight"])

    def build():
        m, _ = F.aim_build(C, sd0, d3="label")
        with torch.no_grad():
            m.afv.weight.nan_to_num_(0.0)
        return m.to(dev)

    torch.manual_seed(R["train_seed"])
    model, ema_model = build(), build()
    ema_model.load_state_dict(model.state_dict())
    for p in ema_model.parameters():
        p.requires_grad_(False)
    fmodel, fema = Forces(model), Forces(ema_model)
    shifts = [p for n, p in model.named_parameters() if n.endswith("atomic_shift.shifts.weight")]
    other = [p for n, p in model.named_parameters() if not n.endswith("atomic_shift.shifts.weight")]
    opt = torch.optim.RAdam([dict(params=other), dict(params=shifts, weight_decay=0.0)], lr=H["lr"], weight_decay=1e-8)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=H["plateau_factor"],
                                                     patience=H["plateau_patience"], threshold=H["plateau_threshold"],
                                                     threshold_mode="rel")
    loss_fn = MTLoss(components=dict(
        energy=dict(fn="aimnet.train.loss.energy_loss_fn", weight=H["ew"]),
        forces=dict(fn="aimnet.train.loss.peratom_loss_fn", weight=fw, kwargs=dict(key_true="forces",
                                                                                    key_pred="forces"))))
    ema = EMA(model, ema_model, H["ema"])
    last_p, best_p, stop_p = rd / "last.pt", rd / "best_ts_ema.pt", rd / "stop.json"
    hist, start, gstep, sae_ck = [], 0, 0, None
    # FTF-sel selection state: bts_* = best TS-near metric / epoch / EMA weights (CPU; None = not retained), bsw_* = best
    # TS-near among the epochs evaluated after the switch, cands = old-format candidates, sel = the selection record
    TS = fresh_ts_state()
    if last_p.is_file():
        ck = torch.load(last_p, map_location=dev, weights_only=False)
        if (ck.get("key") or {}).get("split_fp") != sfp:
            M.stop(f"{last_p}: checkpoint of another split")
        model.load_state_dict(ck["model"])
        ema_model.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        sch.load_state_dict(ck["sch"])
        hist, start, gstep = ck["hist"], int(ck["epoch"]) + 1, int(ck["gstep"])
        sae_ck = ck.get("sae")
        TS = ts_state(ck)
        del ck
        if TS["fmt"] == "old":                           # FTF-sel migration of a run started by the old code
            mg = TS["sel"]["migration"]
            mg.update(sched_best_before=_num(sch.best), sched_num_bad_before=int(sch.num_bad_epochs),
                      sched_cooldown_before=int(sch.cooldown_counter))
            sch.best = float(TS["bts_metric"])           # inf when no record has the metric (is_better(a, inf) = True)
            sch.num_bad_epochs = int(mg["sched_num_bad_epochs"])
            sch.cooldown_counter = 0
            old_best = rd / "best_ema.pt"                # all-structure best: relabelled, never exported
            if old_best.is_file():
                try:
                    os.replace(old_best, rd / LEGACY_BEST)
                    TS["sel"]["legacy_best_ema"] = str(rd / LEGACY_BEST)
                except OSError as e:
                    print(f"[train] WARNING: could not rename {old_best} to {LEGACY_BEST}: {e!r}", flush=True)
            elif (rd / LEGACY_BEST).is_file():           # renamed by an earlier attempt that ended before an epoch
                TS["sel"]["legacy_best_ema"] = str(rd / LEGACY_BEST)
            print(f"[train] FTF-sel migration (selection metric -> {SEL_NAME}): best TS-near epoch {TS['bts_epoch']} "
                  f"({_num(TS['bts_metric'])} kcal/mol/A), weights {TS['sel']['ts_best_weights']}; since best "
                  f"{mg['since_best']}; lr kept {mg['lr_kept']}; scheduler best {mg['sched_best_before']} -> "
                  f"{_num(sch.best)}, num_bad {mg['sched_num_bad_before']} -> {sch.num_bad_epochs}, cooldown 0",
                  flush=True)
        print(f"[train] resumed from {last_p} ({TS['fmt']} format): epoch {start - 1}, step {gstep}, best TS-near "
              f"{TS['bts_epoch']} ({_num(TS['bts_metric'])} kcal/mol/A), best since switch {TS['bsw_epoch']}, "
              f"{bud.prev_h:.2f} GPU-h used", flush=True)
    # ---- validation chunk + self-check (FTF-selfcheck; before the E0 fit)
    t_sc = time.time()
    vch = Chunk(store, va)
    vb = eval_batches(vch, int(H["valid_batch"]))
    beat()
    s0 = min(by_shard)
    Z0, X0, F0 = store.mm(s0)
    shm = store.meta["shards"][s0]
    el0 = {int(z) for z in np.unique(np.asarray(Z0))}
    g0, g1 = s0 * store.shard_n, min(store.n, (s0 + 1) * store.shard_n)
    sc = dict(shard=s0, shard_atoms=int(len(Z0)), shard_atoms_meta=int(shm["n_atoms"]),
              shard_nat_sum=int(store.nat[g0:g1].sum()), elements=sorted(el0),
              finite_arrays=bool(np.isfinite(np.asarray(X0)).all() and np.isfinite(np.asarray(F0)).all()),
              index_overlap=ov)
    ch0 = Chunk(store, by_shard[s0])
    sel0 = train_batches(ch0, bs, np.random.default_rng([R["train_seed"], 999999]))[0]
    zero = np.zeros(store.n)
    model.train()
    fmodel.train()
    x, y, _ = ch0.batch(sel0, zero)
    x, y = _dev(x, dev), _dev(y, dev)
    opt.zero_grad(set_to_none=True)
    lsc = loss_fn(fmodel(x), y)["loss"]
    lsc.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    sc.update(train_batch=len(sel0), train_loss=float(lsc.detach().item()), n_grad_tensors=len(grads),
              grads_finite=bool(all(bool(torch.isfinite(g).all()) for g in grads)))
    opt.zero_grad(set_to_none=True)
    ema_model.eval()
    fema.eval()
    v0 = evaluate(fema, vch, vb[:1], zero, dev)
    sc.update(valid_batch=v0["n"], valid_finite=v0["finite"], valid_f_mae_kcal_A=v0["f_all"],
              seconds=round(time.time() - t_sc, 1))
    sc["pass_"] = bool(sc["shard_atoms"] == sc["shard_atoms_meta"] == sc["shard_nat_sum"] and sc["finite_arrays"]
                       and el0 <= set(int(z) for z in store.meta["elements"]) and math.isfinite(sc["train_loss"])
                       and sc["n_grad_tensors"] > 0 and sc["grads_finite"] and sc["valid_finite"])
    print(f"[train] self-check {'pass' if sc['pass_'] else 'FAIL'}: {sc}", flush=True)
    if not sc["pass_"]:
        M.stop(f"{run}: start self-check failed: {sc}")
    beat()
    # ---- E0 (SAE) fit, pinned (FT-E0)
    sae_p = rd / "sae.json"
    saej = F._rj(sae_p)
    skey = dict(split_fp=sfp, cand=cand, e0_nmax=H["e0_nmax"])
    if sae_ck is not None:
        saej = sae_ck
    elif not saej or saej.get("key") != skey:
        t1 = time.time()
        er = e0_rows(store, tr, int(H["e0_nmax"]))
        _rr, sub = store.atoms(er)
        model.eval()
        e_pred, _ = F.aim_predict(model, sub, dev, forces=False)
        sae, fitst = F.lstsq_e0(sub, np.array([a.info["E_fit"] for a in sub]) - e_pred)
        saej = dict(key=skey, sae_eV=sae, fit=fitst, n_sample=len(sub), seconds=round(time.time() - t1, 1),
                    method=F.DECISIONS["FT-E0"], frame="E_fit (eV)")
        M.write_json(sae_p, saej)
        del sub
    beat()
    sae = {int(z): float(v) for z, v in saej["sae_eV"].items()}
    miss = sorted(set(int(z) for z in store.meta["elements"]) - set(sae))
    if miss:
        print(f"[train] WARNING: elements {miss} have no SAE (absent from the E0 sample): shift 0", flush=True)
    tab = np.zeros(128)
    for z, v in sae.items():
        tab[z] = v
    shift = store.per_struct(tab)
    hyper = dict(lr=H["lr"], warmup=f"epochs 0-{H['warmup_epochs'] - 1} linear per step {H['lr_warm0']} -> {H['lr']}",
                 warm_steps=warm_steps, steps_per_epoch=nb_epoch, ema=H["ema"],
                 scheduler=f"ReduceLROnPlateau factor {H['plateau_factor']} patience {H['plateau_patience']} threshold "
                           f"{H['plateau_threshold']} rel (strictly lower = the best / early-stop rule) on {SEL_NAME}, "
                           f"from epoch {H['warmup_epochs']}",
                 selection_metric=SEL_NAME, selection_key=SEL_KEY,
                 switched_at_epoch=TS["sel"].get("switched_at_epoch"), previous_metric=TS["sel"].get("previous_metric"),
                 recorded_only="validation force MAE over all structures (mae_f_kcal_A), energy MAE (mae_e_kcal)",
                 grad_skip=f"non-finite gradient with finite loss: step skipped; STOP at "
                           f"{H['grad_skip_consec_stop']} in a row or {H['grad_skip_epoch_stop']} in one epoch",
                 stop=f"lr < {H['lr_stop']} | {H['no_improve_stop']} epochs without improvement of {SEL_NAME} | "
                      f"{o.max_epochs} epochs | {cap_h} GPU-h", ew=H["ew"], fw=fw, fw_source=fw_src, batch_size=bs,
                 valid_batch=H["valid_batch"], clip=f"clip_grad_value_ {H['clip_value']} (finetune.train_aimnet value)",
                 optimizer="RAdam wd 1e-8 (shifts 0)", d3=F.B3LYP_D3BJ, seed=R["train_seed"], wall_h=o.wall_h,
                 max_epochs=o.max_epochs, cap_gpu_h=cap_h)

    def record(status, stop_reason=None, **kw):
        ep_s = [h["epoch_s"] for h in hist]
        best_epoch = TS["bts_epoch"]
        brec = next((h for h in hist if h["epoch"] == best_epoch), None)
        ab_e, ab_m = hist_best(hist, all_metric)
        pk = pick_export(TS, hist)
        return dict(run=run, cand=cand, kind="aimnet", mode="full", status=status, model=str(model_out),
                    model_card=str(model_out.with_suffix(".json")), foundation=str(jpt), split=R["split"],
                    split_dir=str(sdir), data_fingerprint=sfp, store_dir=str(store.dir),
                    store_fingerprint=store.meta.get("fingerprint"), g4=(smeta.get("g4") or {}).get("why"),
                    inputs_complete=smeta.get("inputs_complete"), n_train=int(len(tr)), n_valid=int(len(va)),
                    n_test=int(len(idx["test"])), n_valid_ts_near=int(store.ts_near[va].sum()), gpu=gpu,
                    gpu_h=bud.now_h(), cap_gpu_h=cap_h, stop_reason=stop_reason,
                    selection_metric=SEL_NAME, switched_at_epoch=TS["sel"].get("switched_at_epoch"),
                    previous_metric=TS["sel"].get("previous_metric"),
                    epochs=dict(n=len(hist), best_epoch=best_epoch, best=brec,
                                best_val_ts_force_mae_kcal_A=_num(TS["bts_metric"]),
                                since_best=(hist[-1]["epoch"] - best_epoch) if hist and best_epoch is not None
                                else None,
                                best_ts_since_switch=dict(epoch=TS["bsw_epoch"], metric=_num(TS["bsw_metric"])),
                                best_all_structures=dict(epoch=ab_e, metric=_num(ab_m),
                                                         note="recorded only (no decision uses it since 11-06)"),
                                selection=TS["sel"],
                                export_pick=dict(epoch=pk[1], metric_ts=pk[2], rule=pk[3]) if pk else None,
                                lr_current=float(opt.param_groups[0]["lr"]),
                                epoch_s_median=float(np.median(ep_s)) if ep_s else None,
                                metric=f"{SEL_NAME} (kcal/mol/A) = best epoch / scheduler / early stop; all-structure "
                                       f"force MAE and energy MAE recorded only",
                                history=hist),
                    sae=saej, parity_released_vs_built=parity, self_check=sc, hyper=hyper, import_guard=guard,
                    disk_gb=dict(run_dir=F._du_gb(rd)),
                    decisions={k: DECISIONS[k] for k in DECISIONS if k not in ("FTF-store", "FTF-c2geo")},
                    written=F._now(), **_jobtag(), **kw)

    def save_ck(epoch):
        """last.pt in the new format (ckpt_format SEL_CODE; no all-structure best fields)."""
        F._wtorch(dict(model=model.state_dict(), ema=ema_model.state_dict(), opt=opt.state_dict(),
                       sch=sch.state_dict(), hist=hist, epoch=epoch, gstep=gstep, ckpt_format=SEL_CODE,
                       best_ts_metric=_num(TS["bts_metric"]), best_ts_epoch=TS["bts_epoch"], best_ts_ema=TS["bts_sd"],
                       best_ts_sw_metric=_num(TS["bsw_metric"]), best_ts_sw_epoch=TS["bsw_epoch"],
                       best_ts_sw_ema=TS["bsw_sd"], selection=TS["sel"], sae=saej, key=key, written=F._now()),
                  last_p)

    def write_best():
        """best_ts_ema.pt (plain EMA state_dict, as export_aimnet expects) = the FTF-export pick, from the in-memory
        copies that last.pt holds; returns the pick (None: nothing exportable)."""
        pk = pick_export(TS, hist)
        if pk is not None:
            F._wtorch(pk[0], best_p)
        return pk

    plist = list(model.parameters())

    def grads_finite():
        gl = [p.grad.detach() for p in plist if p.grad is not None]
        if not gl:
            return False
        try:
            nr = torch._foreach_norm(gl)
        except (AttributeError, RuntimeError, TypeError):
            nr = [torch.linalg.vector_norm(g) for g in gl]
        return bool(torch.isfinite(torch.stack(list(nr))).all())

    def tick_eval():
        bud.tick()
        beat()

    def job_h():
        return (time.time() - T0) / 3600.0

    def write_hist():
        M.write_text(rd / "epochs.jsonl", "".join(json.dumps(h) + "\n" for h in hist))

    stop = F._rj(stop_p)
    if stop and stop.get("selection_metric") != SEL_NAME and "without a better" in str(stop.get("reason") or ""):
        # FTF-sel: an early stop decided by the old all-structure metric is not a stop under the TS-near rule (the
        # lr / max-epoch / GPU-h stops do not depend on the metric and are kept); the TS-near rule below decides
        moved = rd / "stop_oldsel.json"
        try:
            os.replace(stop_p, moved)
            TS["sel"]["old_stop"] = dict(reason=stop.get("reason"), epoch=stop.get("epoch"), moved_to=str(moved))
            print(f"[train] FTF-sel: old early stop ({stop.get('reason')}) set aside -> {moved}", flush=True)
            stop = None
        except OSError as e:
            print(f"[train] WARNING: could not move the old stop.json aside ({e!r}); it is kept", flush=True)
    stop_reason = stop.get("reason") if stop else None
    if stop_reason is None and hist and TS["bts_epoch"] is not None and \
            int(hist[-1]["epoch"]) - int(TS["bts_epoch"]) >= int(H["no_improve_stop"]):
        stop_reason = (f"{int(hist[-1]['epoch']) - int(TS['bts_epoch'])} epochs without a better {SEL_NAME} "
                       f"(at resume, FTF-sel)")
    ex = ThreadPoolExecutor(max_workers=1)
    t_beat = time.time()
    epoch = start
    while stop_reason is None:
        ep_recent = [h["epoch_s"] for h in hist[-5:]]
        est_h = float(np.median(ep_recent)) / 3600.0 if ep_recent else 0.0
        if epoch >= int(o.max_epochs):
            stop_reason = f"max epochs {o.max_epochs}"
            break
        if bud.now_h() >= cap_h or (est_h and bud.now_h() + 1.1 * est_h > cap_h):
            stop_reason = f"GPU-h cap: the next epoch would end past {cap_h} GPU-h ({bud.now_h():.2f} used)"
            break
        if est_h and job_h() + 1.2 * est_h > float(o.wall_h):
            bud.tick(True, note="paused before the wall")
            F._write_train(out_json, record("running", note=f"paused at {job_h():.2f} h of the job before the "
                                                            f"{o.wall_h} h wall; resubmit the same command"))
            ex.shutdown(wait=False)
            M.not_ready(f"{run}: paused before the {o.wall_h} h wall after epoch {epoch - 1} - resubmit")
        t0 = time.time()
        rng = np.random.default_rng([R["train_seed"], epoch])
        order = [int(s) for s in rng.permutation(sorted(by_shard))]
        model.train()
        fmodel.train()
        tl, nb, abort, n_skip, consec = 0.0, 0, None, 0, 0
        fut = ex.submit(Chunk, store, by_shard[order[0]])
        for jsh, s in enumerate(order):
            ch = fut.result()
            if jsh + 1 < len(order):
                fut = ex.submit(Chunk, store, by_shard[order[jsh + 1]])
            for sel in train_batches(ch, bs, rng):
                if gstep < warm_steps:
                    lr = H["lr_warm0"] + (H["lr"] - H["lr_warm0"]) * min(1.0, gstep / max(1, warm_steps - 1))
                    for g in opt.param_groups:
                        g["lr"] = lr
                elif epoch == int(H["warmup_epochs"]) and nb == 0:
                    for g in opt.param_groups:
                        g["lr"] = H["lr"]
                x, y, _ = ch.batch(sel, shift)
                x, y = _dev(x, dev), _dev(y, dev)
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(fmodel(x), y)["loss"]
                loss.backward()
                lv = float(loss.detach().item())
                if not math.isfinite(lv):
                    M.stop(f"{run}: non-finite training loss at epoch {epoch}, step {gstep} (shard {s}); last.pt "
                           f"(epoch {epoch - 1}) is intact")
                if not grads_finite():                   # finite loss, inf / NaN gradient: skip this step (no
                    opt.zero_grad(set_to_none=True)      # optimizer / EMA update), STOP when it repeats
                    n_skip += 1
                    consec += 1
                    print(f"[train] WARNING: non-finite gradient at epoch {epoch}, step {gstep} (shard {s}, loss "
                          f"{lv:.4g}): step skipped ({n_skip} this epoch)", flush=True)
                    if consec >= H["grad_skip_consec_stop"] or n_skip >= H["grad_skip_epoch_stop"]:
                        M.stop(f"{run}: non-finite gradients ({consec} in a row, {n_skip} in epoch {epoch}); last.pt "
                               f"(epoch {epoch - 1}) is intact")
                    continue
                consec = 0
                torch.nn.utils.clip_grad_value_(plist, H["clip_value"])
                opt.step()
                ema.update()
                gstep += 1
                tl += lv
                nb += 1
                if time.time() - t_beat > 60:
                    bud.tick()
                    beat()
                    t_beat = time.time()
                if bud.now_h() > cap_h:
                    abort = "cap"
                    break
                if job_h() > float(o.wall_h) + 0.75:
                    abort = "wall"
                    break
            if abort:
                break
        if abort == "cap":
            stop_reason = f"GPU-h cap {cap_h} reached mid-epoch {epoch} (epoch dropped; best EMA kept)"
            break
        if abort == "wall":
            bud.tick(True, note="wall mid-epoch")
            F._write_train(out_json, record("running", note=f"epoch {epoch} cut at the wall; resubmit"))
            ex.shutdown(wait=False)
            M.not_ready(f"{run}: epoch {epoch} cut at {job_h():.2f} h of the job - resubmit (resumes after epoch "
                        f"{epoch - 1})")
        lr_used = float(opt.param_groups[0]["lr"])
        ema_model.eval()
        fema.eval()
        v = evaluate(fema, vch, vb, shift, dev, loss_fn, tick=tick_eval)
        vrec = dict(mae_f_kcal_A=v["f_all"], mae_f_ts_kcal_A=v["f_ts"], mae_e_kcal=v["e"])
        metric = sel_metric(vrec)                        # FTF-sel: TS-near force MAE (EMA); the others recorded only
        if not v["finite"] or metric is None or all_metric(vrec) is None:
            M.stop(f"{run}: non-finite validation at epoch {epoch}: {v}")
        bts_prev = float(TS["bts_metric"])               # the best before this epoch (scheduler alignment below)
        improved = metric < bts_prev
        sw_improved = metric < TS["bsw_metric"]          # improved implies sw_improved (bts_metric <= bsw_metric)
        if sw_improved:
            sd_now = {k: t.detach().to("cpu", copy=True) for k, t in ema_model.state_dict().items()}
            TS.update(bsw_metric=metric, bsw_epoch=epoch, bsw_sd=sd_now)
            if improved:                                 # the same CPU copy (saved once by torch.save)
                TS.update(bts_metric=metric, bts_epoch=epoch, bts_sd=sd_now)
                TS["sel"]["ts_best_weights"] = f"kept (epoch {epoch})"
        if epoch >= int(H["warmup_epochs"]):
            if epoch == int(H["warmup_epochs"]) and math.isfinite(bts_prev):
                # first scheduler step: its best = the best of the warmup epochs too, so the plateau rule and the
                # best / early-stop rule share one best value (a fresh scheduler has best = inf)
                sch.best = min(float(sch.best), bts_prev)
            sch.step(metric)
        lr_next = float(opt.param_groups[0]["lr"])
        rec = dict(epoch=epoch, loss=v["loss"], train_loss=tl / max(1, nb), **vrec, lr=lr_used, lr_next=lr_next,
                   improved=improved, improved_on=SEL_KEY,
                   n_batches=nb, n_skipped_steps=n_skip, gstep=gstep, epoch_s=time.time() - t0, gpu_h=bud.now_h(), job_h=job_h(),
                   max_mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2), gpu=gpu,
                   slurm_job=os.environ.get("SLURM_JOB_ID"))
        hist.append(rec)
        if TS["cands"]:                                  # an epoch after the switch exists: bsw_sd is the fallback;
            th0 = ts_hist(hist)                          # the dropped FTF-export third-rule candidates are recorded
            TS["sel"]["c3_dropped"] = [dict(label=lab, epoch=e, metric_ts=th0.get(int(e)) if e is not None else None)
                                       for lab, e, sd in TS["cands"] if sd is not None]
            TS["cands"] = []
        save_ck(epoch)                                   # last.pt first (holds the TS weights), then best_ts_ema.pt
        if sw_improved or not best_p.is_file():
            write_best()
        write_hist()
        bud.tick(True)
        beat()
        best_epoch = TS["bts_epoch"]
        print(f"[train] epoch {epoch} val F MAE TS-near {metric:.4f} (all {v['f_all']:.4f}) kcal/mol/A | E MAE "
              f"{v['e']:.3f} kcal/mol | lr {lr_used:.3g} -> {lr_next:.3g} | best TS {best_epoch} "
              f"({TS['bts_metric']:.4f}), since switch {TS['bsw_epoch']} | {bud.now_h():.2f} GPU-h | "
              f"{rec['epoch_s']:.0f} s", flush=True)
        if lr_next < H["lr_stop"]:
            stop_reason = f"lr {lr_next:.3g} < {H['lr_stop']}"
        elif epoch - best_epoch >= int(H["no_improve_stop"]):
            stop_reason = f"{epoch - best_epoch} epochs without a better {SEL_NAME}"
        F._write_train(out_json, record("running", stop_reason=stop_reason))
        epoch += 1
    ex.shutdown(wait=False)
    # ---- stop -> export the FTF-export weights (TS-near selection; FTF-stop)
    if not stop:
        M.write_json(stop_p, {**_jobtag(), "reason": stop_reason, "epoch": hist[-1]["epoch"] if hist else None,
                              "selection_metric": SEL_NAME, "best_ts_epoch": TS["bts_epoch"],
                              "gpu_h": bud.now_h(), "stopped": F._now()})   # _jobtag has time
    pk = write_best()                                    # best_ts_ema.pt = the weights of the recorded export epoch
    if pk is None:
        F._write_train(out_json, record("failed", stop_reason=stop_reason, note="no finished epoch / no exportable "
                                                                                "TS-near weights"))
        M.stop(f"{run}: stopped ({stop_reason}) without exportable EMA weights (FTF-export: {TS['sel']})")
    _sd, x_epoch, x_metric, x_rule = pk
    _r4, chk4 = store.atoms(tr[:4])
    exp = F.export_aimnet(C, best_p, model_out, sae, dict(R, run=run), cand, dict(fingerprint=sfp), dev, chk4)
    card_p = model_out.with_suffix(".json")
    card = F._rj(card_p) or {}
    card.update(name=run, mode="full", weights=f"EMA (decay {H['ema']}) of epoch {x_epoch} ({x_rule})",
                selection_metric=SEL_NAME, export_epoch=x_epoch, export_metric=x_metric, export_metric_key=SEL_KEY,
                export_rule=x_rule, best_ts_epoch=TS["bts_epoch"], best_val_ts_force_mae_kcal_A=_num(TS["bts_metric"]),
                switched_at_epoch=TS["sel"].get("switched_at_epoch"), previous_metric=TS["sel"].get("previous_metric"),
                ts_best_weights=TS["sel"].get("ts_best_weights"), stop_reason=stop_reason, fw=fw, split_dir=str(sdir),
                store_dir=str(store.dir), g4=(smeta.get("g4") or {}).get("why"), code_full=CODE_VERSION,
                code_sel=SEL_CODE)
    M.write_json(card_p, card)
    bud.tick(True)
    res = record("done", stop_reason=stop_reason, export=exp, export_epoch=x_epoch, export_metric=x_metric,
                 export_rule=x_rule)
    res["disk_gb"]["model"] = model_out.stat().st_size / 1e9
    F._write_train(out_json, res)
    print(json.dumps(dict(run=run, status="done", stop_reason=stop_reason, selection_metric=SEL_NAME,
                          export_epoch=x_epoch, export_metric=x_metric, export_rule=x_rule,
                          best_ts_epoch=TS["bts_epoch"], epochs=len(hist), gpu_h=round(bud.now_h(), 2),
                          model=str(model_out))), flush=True)
    return 0


# ================================================================ FT-1
def cmd_ft1(run, o):
    import pandas as pd
    import torch
    F._guard_imports()
    R = full_run(run)
    run = R["run"]
    tj = F._rj(F.RES_FT / f"{run}_train.json")
    if not tj or tj.get("status") != "done":
        M.not_ready(f"{run}: no finished training (results/ft/{run}_train.json)")
    out_p = F.RES_FT / f"{run}_ft1.json"
    pj = F._rj(out_p)
    if pj and pj.get("train_written") == tj.get("written") and not o.force:
        print(f"[ft1] {run} up to date ({out_p})", flush=True)
        return 0
    if not torch.cuda.is_available():
        M.stop(f"{run}: no CUDA device")
    dev = "cuda"
    sdir = Path(tj["split_dir"])
    store = Store(tj["store_dir"])
    model = torch.jit.load(str(tj["model"]), map_location=dev)
    tab = pd.read_parquet(store.dir / "table.parquet",
                          columns=["struct_id", "kind", "rxn", "formula", "n_atoms", "e_eh", "dform"])
    out = dict(run=run, cand=tj.get("cand"), model=tj["model"], which="finetuned", mode="full",
               dtype="float32 (shifts float64)", device=torch.cuda.get_device_name(0), sets={},
               split_dir=str(sdir), store_dir=str(store.dir), train_written=tj.get("written"),
               model_export=dict(selection_metric=tj.get("selection_metric"), epoch=tj.get("export_epoch"),
                                 metric_ts=tj.get("export_metric"), rule=tj.get("export_rule")),
               definitions=dict(ts_near=F.DECISIONS["FT-ts-near"], rel_e=F.DECISIONS["FT-relE"],
                                force_mae="mean |F_pred - F_ref| over Cartesian components, kcal mol-1 A-1"),
               note="FT-1 is recorded only (spec 5); FT-2 geometry is the report of step 2")
    zero = np.zeros(store.n)
    for nm in ("valid", "test"):
        rows = np.load(sdir / f"{nm}.npy").astype(np.int64)
        if not len(rows):
            continue
        ch = Chunk(store, rows)
        E = np.zeros(len(ch.rows))
        Fp = [None] * len(ch.rows)
        t0 = time.time()
        for sel in eval_batches(ch, int(o.batch_size)):
            x, _y, _r = ch.batch(sel, zero)
            c = torch.from_numpy(x["coord"]).to(dev).requires_grad_(True)
            z = torch.from_numpy(x["numbers"]).to(dev)
            q = torch.from_numpy(x["charge"]).to(dev)
            e = model(dict(coord=c, numbers=z, charge=q))["energy"]
            g = torch.autograd.grad(e.sum(), c)[0]
            ev = e.detach().double().cpu().numpy()
            fb = (-g).detach().cpu().numpy()
            for k, j in enumerate(sel):
                E[j] = ev[k]
                Fp[j] = fb[k, :int(ch.nat[j])].astype(np.float64)
        dt = time.time() - t0
        Fref = [ch.Fr[int(ch.start[j]):int(ch.start[j] + ch.nat[j])].astype(np.float64) for j in range(len(ch.rows))]
        T = tab.iloc[ch.rows].reset_index(drop=True)
        D = pd.DataFrame(dict(struct_id=T["struct_id"].to_numpy(), kind=T["kind"].to_numpy(),
                              rxn=[tuple(sorted(int(v) for v in str(s).split(",") if v)) for s in T["rxn"]],
                              formula=T["formula"].to_numpy(), n_atoms=T["n_atoms"].to_numpy().astype(int),
                              e_ref=T["e_eh"].to_numpy(np.float64) * M.EH2EV, dform=T["dform"].to_numpy(np.float64)))
        D["e_abs_pred"] = E
        D["e_rel_pred"] = E
        met = F.ft1_metrics(D, Fref, Fp)
        met.update(n_rows=int(len(rows)), predict_s=round(dt, 1), structures_per_s=round(len(rows) / max(dt, 1e-9), 2))
        out["sets"][nm] = met
        D["f_mae_kcal_A"] = [float(np.abs(a - b).mean() * F.EV2KCAL) for a, b in zip(Fref, Fp)]
        D["rxn"] = D.rxn.map(M.tag_list)
        M.write_parquet(F.FT / run / f"ft1_finetuned_{nm}.parquet", D)
        print(f"[ft1] {run} {nm}: n {len(rows)} | F MAE all {met['force_mae_kcal_A']['all']} TS-near "
              f"{met['force_mae_kcal_A']['ts_near']} kcal/mol/A | rel E MAE TS-near {met['rel_e']['ts_near']['mae_kcal']}"
              f" kcal/mol | {len(rows) / max(dt, 1e-9):.0f} struct/s", flush=True)
    out.update(written=F._now(), **_jobtag())
    M.write_json(out_p, out)
    return 0


# ================================================================ interim export (user 11-06 step 2)
EXPORT_TMP = F.FT / "_export_tmp"                   # temp copies (never inside a run dir); removed after the export
TAG_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9.-]*")


def cmd_export(run, o):
    """interim model of a running (or finished) *_full run without touching the training: a COPY of last.pt (old or
    new format) is read, the weights are picked by FTF-export, the model is built and checked exactly as the final
    export (finetune.export_aimnet: SAE + ATOM_REF shifts, B3LYP-D3(BJ) head) -> $MLT/models/<run>_<tag>.jpt + .json
    (card interim=true, kind aimnet: mlts_bridge.resolve_spec('<run>_<tag>') -> _from_card) and
    results/ft/<run>_<tag>_export.json. No lock is taken and nothing is written in $MLT/ft/<run>/."""
    import torch
    F._guard_imports()
    R = full_run(run)
    run = R["run"]
    tag = str(o.tag or "")
    if not TAG_RE.fullmatch(tag):
        raise SystemExit(f"--tag {tag!r}: letters, digits, '.' and '-' only (no '_'), not empty")
    name = f"{run}_{tag}"
    try:
        F.parse_run(name)
        clash = True
    except SystemExit:
        clash = False
    if clash or name in FULL_RUNS:
        raise SystemExit(f"{name} is a training run name; choose another --tag")
    rd = F.FT / run
    last_p = rd / "last.pt"
    if not last_p.is_file():
        M.not_ready(f"{run}: no checkpoint {last_p} yet")
    dst = F.MODELS / f"{name}.jpt"
    card_p = dst.with_suffix(".json")
    tmpd = EXPORT_TMP / f"{name}.{_lock_me()}.{os.getpid()}"
    F._mkdir(tmpd)
    try:
        cp = tmpd / "last.pt"
        shutil.copyfile(last_p, cp)                      # last.pt is replaced atomically (os.replace): one open = one version
        ck = torch.load(cp, map_location="cpu", weights_only=False)
        key = dict(ck.get("key") or {})
        hist = list(ck.get("hist") or [])
        last_epoch = int(ck["epoch"])
        saej = ck.get("sae") or F._rj(rd / "sae.json")
        TS = ts_state(ck)
        del ck
        summ = ts_summary(hist, TS)
        print(f"[export] {run}: last.pt ({TS['fmt']} format), last epoch {last_epoch}; TS-near metric history "
              f"{json.dumps(summ, default=str)}", flush=True)
        pk = pick_export(TS, hist)
        if pk is None:
            M.stop(f"{run}: no exportable EMA weights in {last_p} (FTF-export: {TS['sel']})")
        sd, x_epoch, x_metric, x_rule = pk
        old = F._rj(card_p)
        if dst.is_file() and old and not o.force:
            if old.get("source_run") == run and old.get("export_epoch") == x_epoch and old.get("export_rule") == x_rule:
                print(f"[export] {name} up to date ({dst}: epoch {x_epoch}, {x_rule})", flush=True)
                return 0
            raise SystemExit(f"{dst} exists (source {old.get('source_run')} epoch {old.get('export_epoch')}); the "
                             f"current pick is epoch {x_epoch}: pass --force to replace it or use another --tag")
        cand = F.canon_cand(key.get("cand") or H["cand"])
        C = F.CANDIDATES[cand]
        if C["kind"] != "aimnet":
            raise SystemExit(f"{run}: candidate {cand} is not AIMNet2")
        if not key.get("split_fp") or not key.get("split_dir") or not key.get("store_dir"):
            M.stop(f"{last_p}: checkpoint run key lacks split / store: {key}")
        if not saej or not saej.get("sae_eV"):
            M.stop(f"{run}: no SAE in {last_p} / sae.json")
        sae = {int(z): float(v) for z, v in saej["sae_eV"].items()}
        sdir = Path(key["split_dir"])
        store = Store(key["store_dir"])
        tr = np.load(sdir / "train.npy").astype(np.int64)
        _r4, chk4 = store.atoms(tr[:4])                  # the same check structures as the final export
        wp = tmpd / "weights.pt"
        F._wtorch(sd, wp)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        exp = F.export_aimnet(C, wp, dst, sae, dict(R, run=name), cand, dict(fingerprint=key["split_fp"]), dev, chk4)
        if R["mode"] == "fold":
            held = f"fold {R['val']} (T1)"
        elif R["mode"] == "lodo":
            held = f"dipole {M.DIPOLES[R['val']]}"
        else:
            held = "none (all data)"
        card = F._rj(card_p) or {}
        card.update(name=name, kind="aimnet", interim=True, mode="full_interim",
                    note="interim export (user 11-06 step 2): not the final model of the run; the training continues",
                    source_run=run, source_checkpoint=str(last_p), source_ckpt_format=TS["fmt"],
                    source_last_epoch=last_epoch, weights=f"EMA (decay {H['ema']}) of epoch {x_epoch} ({x_rule})",
                    selection_metric=SEL_NAME, export_epoch=x_epoch, export_metric=x_metric, export_metric_key=SEL_KEY,
                    export_rule=x_rule, best_ts_epoch=TS["bts_epoch"],
                    best_val_ts_force_mae_kcal_A=_num(TS["bts_metric"]),
                    ts_best_weights=(TS["sel"] or {}).get("ts_best_weights"),
                    switched_at_epoch=(TS["sel"] or {}).get("switched_at_epoch"),
                    previous_metric=(TS["sel"] or {}).get("previous_metric"), selection=TS["sel"], split_mode=R["mode"],
                    fold=R["val"] if R["mode"] == "fold" else None, held_out=held, split_dir=str(sdir),
                    store_dir=str(store.dir), fw=key.get("fw"), code_full=CODE_VERSION, code_sel=SEL_CODE)
        M.write_json(card_p, card)
        out = dict(name=name, model=str(dst), card=str(card_p), source_run=run, source_last_epoch=last_epoch,
                   source_ckpt_format=TS["fmt"], selection_metric=SEL_NAME, export_epoch=x_epoch,
                   export_metric=x_metric, export_rule=x_rule, split=R["split"], held_out=held, export=exp,
                   ts_history=summ, decisions={k: DECISIONS[k] for k in ("FTF-sel", "FTF-export")},
                   written=F._now(), **_jobtag())
        M.write_json(F.RES_FT / f"{name}_export.json", out)
        print(json.dumps(dict(name=name, model=str(dst), source_run=run, source_last_epoch=last_epoch,
                              export_epoch=x_epoch, export_metric=x_metric, export_rule=x_rule, split=R["split"],
                              check_max_abs_eV=exp.get("check_max_abs_eV"))), flush=True)
        return 0
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)


# ================================================================ status
def cmd_status():
    """one line per *_full run; best TS-near epoch / metric from the history (both json formats), export pick."""
    print(f"{'run':18s} {'split':16s} {'n_train':>8s} {'n_valid':>7s} {'n_test':>6s} {'G4':>3s} {'status':8s} "
          f"{'ep':>4s} {'bestTS':>6s} {'F_TS':>7s} {'F_all':>7s} {'since':>5s} {'lr':>8s} {'sw':>4s} {'exp':>4s} "
          f"{'GPU-h':>6s} stop")
    for r in FULL_RUNS:
        d = F._rj(F.RES_FT / f"{r}_data.json") or {}
        t = F._rj(F.RES_FT / f"{r}_train.json") or {}
        c = d.get("counts") or {}
        ep = t.get("epochs") or {}
        hist = ep.get("history") or []
        be, bm = hist_best(hist, sel_metric)
        b = next((h for h in hist if h.get("epoch") == be), {}) if be is not None else {}
        last = hist[-1] if hist else {}
        xp = ep.get("export_pick") or {}
        sw = t.get("switched_at_epoch")
        g4 = (d.get("g4") or {}).get("used")
        print(f"{r:18s} {d.get('split', ''):16s} {c.get('train', {}).get('n', ''):>8} {c.get('valid', {}).get('n', ''):>7} "
              f"{c.get('test', {}).get('n', ''):>6} {('y' if g4 else 'n') if d else '':>3s} {t.get('status', ''):8s} "
              f"{ep.get('n', ''):>4} {be if be is not None else '':>6} "
              f"{('%.4f' % bm) if _num(bm) is not None else '':>7s} "
              f"{('%.4f' % b['mae_f_kcal_A']) if _num(b.get('mae_f_kcal_A')) is not None else '':>7s} "
              f"{(int(last['epoch']) - be) if last and be is not None else '':>5} "
              f"{('%.2e' % last['lr_next']) if _num(last.get('lr_next')) is not None else '':>8s} "
              f"{sw if sw is not None else '':>4} {xp.get('epoch') if xp.get('epoch') is not None else '':>4} "
              f"{('%.2f' % t['gpu_h']) if t.get('gpu_h') is not None else '':>6s} {t.get('stop_reason') or ''}")
    print(f"bestTS / F_TS = first minimum of the selection metric ({SEL_NAME}) over the epoch history; F_all = "
          f"all-structure force MAE at that epoch (recorded only); since = epochs since bestTS; lr = lr after the last "
          f"epoch; sw = switched_at_epoch (FTF-sel migration); exp = epoch the FTF-export rule picks now")
    return 0


# ================================================================ main
def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(description="DEPLOY-V2 (MLTS) full-data fine-tuning (user 11-05 20:5x step 1)")
    sub = ap.add_subparsers(dest="cmd")
    a = sub.add_parser("data")
    a.add_argument("runs", nargs="*")
    a.add_argument("--all", action="store_true", help=f"every *_full run: {', '.join(FULL_RUNS)}")
    a.add_argument("--no-g4", dest="no_g4", action="store_true", help="never include G-4 even when complete")
    a.add_argument("--fmax-cap", dest="fmax_cap", type=float, default=F.FMAX_CAP)
    a.add_argument("--valid-frac", dest="valid_frac", type=float, default=F.VALID_FRAC)
    a.add_argument("--cap-test", dest="cap_test", type=int, default=F.CAP_TEST)
    a.add_argument("--geo-c2", dest="geo_c2", choices=["stop", "drop", "record", "off"], default="stop")
    a.add_argument("--force", action="store_true")
    a = sub.add_parser("train")
    a.add_argument("run")
    a.add_argument("--cand", default=H["cand"])
    a.add_argument("--fw", type=float, default=None, help="force weight (default: FT-abl decision, else 1.0)")
    a.add_argument("--max-epochs", dest="max_epochs", type=int, default=H["max_epochs"])
    a.add_argument("--cap-gpu-h", dest="cap_gpu_h", type=float, default=H["cap_gpu_h"])
    a.add_argument("--wall-h", dest="wall_h", type=float, default=H["wall_h"])
    a.add_argument("--allow-partial", dest="allow_partial", action="store_true")
    a.add_argument("--force", action="store_true")
    a = sub.add_parser("ft1")
    a.add_argument("run")
    a.add_argument("--batch-size", dest="batch_size", type=int, default=64)
    a.add_argument("--force", action="store_true")
    a = sub.add_parser("export", help="interim model <run>_<tag> from a copy of last.pt (FTF-export; no lock)")
    a.add_argument("run")
    a.add_argument("--tag", required=True)
    a.add_argument("--force", action="store_true", help="replace an existing <run>_<tag> model of another epoch")
    sub.add_parser("status")
    o = ap.parse_args(argv)
    if o.cmd == "data":
        return cmd_data(o.runs, o)
    if o.cmd == "train":
        return cmd_train(o.run, o)
    if o.cmd == "ft1":
        return cmd_ft1(o.run, o)
    if o.cmd == "export":
        return cmd_export(o.run, o)
    if o.cmd == "status":
        return cmd_status()
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
