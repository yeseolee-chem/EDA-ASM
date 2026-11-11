"""Training GNN3: the loss, one fit with early stopping, the cross-validation unit and the final model.

Rules (identical to the published runs, see constants.py):
    * standardise features with the training reactions, targets with the training pairs (``Standardizer``);
    * loss = mean |f - y| (standardised) + 0.1 * mean |barrier - (strain + inter + solv)| / s_barrier (kcal scale);
    * AdamW (lr 1e-3, weight decay 1e-4), cosine learning-rate decay over 300 epochs, dropout 0.1, 64 pairs per step;
    * every epoch the pairs are shuffled and each pair is randomly presented as (a, b) or (b, a) with the sign of its
      target flipped accordingly (order augmentation; the antisymmetric head makes both views consistent);
    * tuning: each grid cell is trained on the inner-training pairs with early stopping (patience 30) on the
      inner-validation pairs; the cell with the lowest validation metric (mean of the 4 standardised MAEs) wins;
    * final: the winning cell is retrained on all outer-training pairs for exactly its best epoch count, with 5 seeds;
      the test prediction is the mean of the 5 (their standard deviation = "seed spread").

This file re-implements fu2/train.py (fit, epoch_batches, orient, loss_fn, seed_all, gate_m1), deploy/train_tf.py
(leak_check, kcal, std_sha) and ch3m/train_m.py (run_unit, choose_final_grid, train_final) for the GNN in 3-channel
mode. The random-number calls (seed setting, shuffling, order flips, dropout) happen in the same order and number as in
the original, which is what makes a re-run bit-identical on the same GPU (equivalence check E-3).
"""
from __future__ import annotations

import copy
import json
import os
import socket
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import constants as K
from .data import PairDataset, Standardizer, default_device, list_sha256
from .model import GNNPairModel, build_model, make_reaction_batch, predict_standardized

# Deterministic cuDNN kernels (= fu2/train.py module settings): needed for run-to-run identical GPU results.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = default_device()
CODE_VERSION = "gnn3-clean-1"
PREDICTION_COLUMNS = ["pair_id", "ref_kind", "pos", "fold", "rung", "arm", "model", "target", "y", "yhat", "base", "beta",
                      "yhat_seed_sd"]                  # = the original K_predictions / KM_predictions columns


class TrainingError(RuntimeError):
    """A fit produced non-finite numbers, a check failed, or a cached result does not match the current settings."""


# =========================================================================== loss and batches
def seed_everything(seed: int) -> None:
    """Seed numpy's global generator and torch's CPU and CUDA generators (= fu2 train.seed_all)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def identity_penalty_loss(prediction_std: torch.Tensor, target_std: torch.Tensor, target_scale: torch.Tensor):
    """Training loss of one batch (= fu2 train.loss_fn in 3-channel mode).

    Parameters
    ----------
    prediction_std, target_std : tensor [n_pairs, 4]
        Model output and target in standardised units (target / target_scale).
    target_scale : tensor [4]
        Training-pair standard deviation of each target (kcal/mol).

    Returns
    -------
    (loss, l1, penalty) : scalar tensors
        l1 = mean absolute error (standardised); penalty = mean |barrier - (strain + inter + solv)| on the kcal scale
        divided by the barrier scale; loss = l1 + IDENTITY_PENALTY_WEIGHT * penalty.

    Notes
    -----
    The labels obey strain + inter + solv = barrier (up to < 0.1 kcal/mol), but the network predicts the barrier as its
    own output. The penalty keeps the four predictions physically consistent without forcing the barrier to be a sum.
    """
    l1 = (prediction_std - target_std).abs().mean()
    prediction_kcal = prediction_std * target_scale
    penalty = ((prediction_kcal[:, K.BARRIER_INDEX] - prediction_kcal[:, K.CHANNEL_INDICES].sum(1)).abs()
               / target_scale[K.BARRIER_INDEX]).mean()
    return l1 + K.IDENTITY_PENALTY_WEIGHT * penalty, l1, penalty


def swap_pair_order(rows_a: np.ndarray, rows_b: np.ndarray, swap: np.ndarray):
    """Present pairs with ``swap`` True as (b, a). Returns (rows_a', rows_b', sign) with target' = sign * target."""
    return np.where(swap, rows_b, rows_a), np.where(swap, rows_a, rows_b), np.where(swap, -1.0, 1.0)


def iterate_training_batches(rng: np.random.Generator, rows_a_all: np.ndarray, rows_b_all: np.ndarray,
                             targets_std: torch.Tensor, batch_size: int):
    """One epoch of training batches (= fu2 train.epoch_batches).

    The pairs are put in random order and each pair is flipped to (b, a) with probability 1/2, its target negated.

    Parameters
    ----------
    rng : numpy Generator (consumed: one permutation and one uniform draw per pair, in this order)
    rows_a_all, rows_b_all : array [n_train] int
    targets_std : tensor [n_train, 4]
    batch_size : int

    Yields
    ------
    (rows_a, rows_b, targets) for each batch; targets tensor [batch, 4].
    """
    n_pairs = len(rows_a_all)
    order, swap = rng.permutation(n_pairs), rng.random(n_pairs) < 0.5
    for start in range(0, n_pairs, batch_size):
        chosen = order[start:start + batch_size]
        rows_a, rows_b, sign = swap_pair_order(rows_a_all[chosen], rows_b_all[chosen], swap[chosen])
        targets = targets_std[torch.as_tensor(chosen, device=targets_std.device)] * torch.as_tensor(
            sign[:, None], dtype=targets_std.dtype, device=targets_std.device)
        yield rows_a, rows_b, targets


# =========================================================================== one fit
@dataclass
class FitResult:
    """Outcome of ``train_one_fit``."""

    model: GNNPairModel
    standardizer: Standardizer
    tensors: dict
    history: list
    best_epoch: int
    best_validation: float
    best_validation_per_target: list | None
    seconds: float

    @property
    def target_scale(self) -> np.ndarray:
        return self.standardizer.target_scale


def train_one_fit(dataset: PairDataset, cell: dict, train_pairs, validation_pairs=None, seed: int = K.BASE_SEED,
                  n_epochs: int = K.MAX_EPOCHS, patience: int = K.PATIENCE_EPOCHS, dropout: float = K.DROPOUT,
                  batch_size: int = K.BATCH_SIZE_PAIRS, log_every: int | None = None) -> FitResult:
    """Train one network on ``train_pairs`` (= fu2 train.fit).

    Parameters
    ----------
    dataset : PairDataset
    cell : dict(n_layers, width)
    train_pairs : array [n_train] int
        Pair rows used for the gradient steps AND for every standardisation statistic.
    validation_pairs : array [n_val] int, optional
        When given: evaluated after every epoch, early stopping with ``patience``, the weights of the best epoch are
        returned. When None: exactly ``n_epochs`` epochs (the final fits).
    seed : int
        Resets numpy / torch random generators; also seeds the shuffling generator.
    n_epochs, patience, dropout, batch_size, log_every : see constants.py

    Returns
    -------
    FitResult
        best_epoch = epoch of the lowest validation metric (1-based), or n_epochs without validation.

    Raises
    ------
    TrainingError
        A non-finite loss or validation value.

    Examples
    --------
    >>> fit = train_one_fit(dataset, dict(n_layers=4, width=64), split.inner_train, split.inner_val)  # doctest: +SKIP
    """
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    standardizer = Standardizer.from_training_pairs(dataset, train_pairs)
    tensors = standardizer.tensors(dataset.reactions)
    target_scale = torch.tensor(standardizer.target_scale, dtype=torch.float32, device=DEVICE)
    train_targets = torch.tensor(dataset.targets_kcal[train_pairs] / standardizer.target_scale, dtype=torch.float32,
                                 device=DEVICE)
    model = build_model(cell, dataset.reactions.token_width, dataset.reactions.n_global, len(K.TARGETS),
                        dropout).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=K.LEARNING_RATE, weight_decay=K.WEIGHT_DECAY)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=K.MAX_EPOCHS)
    rows_a_all, rows_b_all = dataset.rows_a[train_pairs], dataset.rows_b[train_pairs]
    history, best = [], dict(value=np.inf, epoch=0, state=None, per_target=None)
    start = time.time()
    for epoch in range(1, n_epochs + 1):
        model.train()
        losses = []
        for rows_a, rows_b, targets in iterate_training_batches(rng, rows_a_all, rows_b_all, train_targets, batch_size):
            batch, index_a, index_b = make_reaction_batch(tensors, rows_a, rows_b)
            loss, l1, penalty = identity_penalty_loss(model(batch, index_a, index_b), targets, target_scale)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append((float(loss), float(l1), float(penalty)))
        schedule.step()
        record = dict(epoch=epoch, loss=float(np.mean([x[0] for x in losses])), l1=float(np.mean([x[1] for x in losses])),
                      pen=float(np.mean([x[2] for x in losses])), t=time.time() - start)
        if not all(np.isfinite([value for triple in losses for value in triple])):
            raise TrainingError(f"training loss not finite at epoch {epoch} (cell {cell})")
        if validation_pairs is not None:
            prediction = predict_standardized(model, tensors, dataset.rows_a[validation_pairs],
                                              dataset.rows_b[validation_pairs])
            mae_per_target = np.abs(prediction - dataset.targets_kcal[validation_pairs] / standardizer.target_scale).mean(0)
            record["val"] = float(mae_per_target[K.VALIDATION_INDICES].mean())
            if not np.isfinite(record["val"]):
                raise TrainingError(f"validation metric not finite at epoch {epoch} (cell {cell})")
            if record["val"] < best["value"]:
                best = dict(value=record["val"], epoch=epoch, state=copy.deepcopy(model.state_dict()),
                            per_target=mae_per_target.tolist())
        history.append(record)
        if log_every and (epoch % log_every == 0 or epoch == 1):
            print(f"    ep {epoch} {record}", flush=True)
        if validation_pairs is not None and epoch - best["epoch"] >= patience:
            break
    if validation_pairs is not None:
        if best["state"] is None or best["epoch"] < 1:
            raise TrainingError(f"no finite best epoch (cell {cell})")
        model.load_state_dict(best["state"])
    return FitResult(model, standardizer, tensors, history, best["epoch"] if validation_pairs is not None else n_epochs,
                     best["value"], best["per_target"], time.time() - start)


def predictions_kcal(fit: FitResult, dataset: PairDataset, pair_rows) -> np.ndarray:
    """ΔΔ predictions in kcal/mol of one fit: standardised output x training target scale. Returns [n_pairs, 4]."""
    pair_rows = np.asarray(pair_rows, dtype=int)
    return predict_standardized(fit.model, fit.tensors, dataset.rows_a[pair_rows], dataset.rows_b[pair_rows]) \
        * fit.target_scale


# =========================================================================== checks run after every fit
def check_antisymmetry(fit: FitResult, dataset: PairDataset, pair_rows, seed: int,
                       n_pairs: int = K.ANTISYMMETRY_CHECK_PAIRS) -> dict:
    """Recompute f(a,b), f(b,a), f(a,a) in float64 on random pairs (= fu2 train.gate_m1).

    Returns
    -------
    dict(n, anti_max, self_max, ok)
        anti_max = max |f(a,b) + f(b,a)|, self_max = max |f(a,a)| in kcal/mol; ok when both <= 1e-4.

    Notes
    -----
    The head is antisymmetric by construction, so the check guards against wiring mistakes, not against learning.
    """
    rng = np.random.default_rng(seed)
    pair_rows = np.asarray(pair_rows)
    chosen = rng.choice(pair_rows, size=min(n_pairs, len(pair_rows)), replace=False)
    model64 = copy.deepcopy(fit.model).double().eval()
    tensors64 = fit.standardizer.tensors(dataset.reactions, dtype=torch.float64, device=DEVICE)
    rows_a, rows_b = dataset.rows_a[chosen], dataset.rows_b[chosen]
    f_ab, f_ba, f_aa = (predict_standardized(model64, tensors64, x, y, batch_size=K.ANTISYMMETRY_BATCH_PAIRS)
                        * fit.target_scale for x, y in ((rows_a, rows_b), (rows_b, rows_a), (rows_a, rows_a)))
    anti, self_value = float(np.abs(f_ab + f_ba).max()), float(np.abs(f_aa).max())
    return dict(n=len(chosen), anti_max=anti, self_max=self_value,
                ok=bool(anti <= K.ANTISYMMETRY_TOLERANCE_KCAL and self_value <= K.ANTISYMMETRY_TOLERANCE_KCAL))


def check_no_leak(fit: FitResult, dataset: PairDataset, fit_pairs, other_pair_sets, test_pairs) -> dict:
    """Leak check (= deploy train_tf.leak_check): the statistics came from exactly the fit's training pairs and their
    reactions; no validation / test pair and no test reaction is among them.

    Raises
    ------
    TrainingError
        On any leak.
    """
    standardizer = fit.standardizer
    used = set(np.asarray(standardizer.pair_rows).tolist())
    if used != set(np.asarray(fit_pairs).tolist()):
        raise TrainingError("leak check: scaling pairs differ from the fit's training pairs")
    for other in other_pair_sets:
        if used & set(np.asarray(other).tolist()):
            raise TrainingError("leak check: a validation / test pair is among the fit's scaling pairs")
    test_pairs = np.asarray(test_pairs, dtype=int)
    test_reactions = set(dataset.rows_a[test_pairs].tolist()) | set(dataset.rows_b[test_pairs].tolist()) \
        if len(test_pairs) else set()
    if test_reactions & set(np.asarray(standardizer.reaction_rows).tolist()):
        raise TrainingError("leak check: a test reaction is among the fit's scaling reactions")
    expected = set(dataset.rows_a[np.asarray(fit_pairs)].tolist()) | set(dataset.rows_b[np.asarray(fit_pairs)].tolist())
    if set(np.asarray(standardizer.reaction_rows).tolist()) != expected:
        raise TrainingError("leak check: scaling reactions differ from the reactions of the training pairs")
    return dict(n_fit_pairs=len(used), n_scale_rxns=int(len(standardizer.reaction_rows)), n_test_rxn_overlap=0, ok=True)


# =========================================================================== saving
def save_model_file(path: Path, fit: FitResult, cell: dict, dataset: PairDataset, extra: dict) -> None:
    """Write one trained network atomically (torch.save) with everything needed to use it later.

    File keys: state (weights, new names), cell, std (mu, sd, gmu, gsd, sy), layout (token layout), d_in, d_glob,
    TARGETS, M (10 -> 4 matrix rows), model "GNN", variant "A+G", code_version, plus ``extra``.
    """
    from .channels import channel_matrix
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(state=fit.model.state_dict(), cell=cell, std=fit.standardizer.as_saved(),
                   layout=dataset.reactions.layout.as_saved(), d_in=int(dataset.reactions.token_width),
                   d_glob=int(dataset.reactions.n_global), TARGETS=list(K.TARGETS), M=channel_matrix().tolist(),
                   model="GNN", variant=K.VARIANT, code_version=CODE_VERSION, **extra)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.close(handle)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(record, indent=1, default=_json_default) + "\n")
    os.replace(temporary, path)


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value))


def environment() -> dict:
    """Software and GPU versions (recorded in every result)."""
    return dict(torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", numpy=np.__version__,
                pandas=pd.__version__)


# =========================================================================== one cross-validation unit
@dataclass
class UnitSettings:
    """Everything that determines a cross-validation unit's result (stored with it; a mismatch refuses the cache)."""

    fold: int
    grid: list = field(default_factory=lambda: list(K.GRID))
    seeds: list = field(default_factory=lambda: list(K.FINAL_SEEDS))
    n_epochs: int = K.MAX_EPOCHS

    def as_dict(self, dataset: PairDataset, split) -> dict:
        return dict(code=CODE_VERSION, fold=self.fold, grid=self.grid, seeds=self.seeds, n_epochs=self.n_epochs,
                    split_pairs_sha256=split.pairs_sha256, inputs_sha256=dataset.sha256,
                    rules=dict(lam=K.IDENTITY_PENALTY_WEIGHT, lr=K.LEARNING_RATE, wd=K.WEIGHT_DECAY, dropout=K.DROPOUT,
                               batch=K.BATCH_SIZE_PAIRS, max_epochs=K.MAX_EPOCHS, patience=K.PATIENCE_EPOCHS,
                               tune_seed=K.BASE_SEED))


def run_cross_validation_unit(dataset: PairDataset, fold: int, out_dir: Path, settings: UnitSettings | None = None,
                              log_every: int | None = 50) -> dict:
    """Tune the grid on fold ``fold``'s inner split, train the final 5 seeds, predict its test pairs
    (= ch3m train_m.run_unit for the GNN in 3-channel mode).

    Writes ``gnn3_fold<k>.json`` (choices, validation curves, checks), ``gnn3_fold<k>.parquet`` (test predictions,
    columns ``PREDICTION_COLUMNS``) and ``gnn3_fold<k>__seed<j>.pt`` (the 5 final networks) into ``out_dir``.
    An existing json with the same settings is returned unchanged (resume after a time limit).

    Returns
    -------
    dict
        The unit record (also the json content): chosen cell / best epoch, tuning results, per-seed fits.

    Examples
    --------
    >>> record = run_cross_validation_unit(PairDataset(), fold=0, out_dir=Path("units"))  # doctest: +SKIP
    """
    settings = settings or UnitSettings(fold=fold)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"gnn3_fold{fold}"
    split = dataset.fold_split(fold)
    fingerprint = json.loads(json.dumps(settings.as_dict(dataset, split), default=_json_default))
    record_path = out_dir / f"{name}.json"
    if record_path.is_file():
        old = json.loads(record_path.read_text())
        if old.get("settings") != fingerprint:
            raise TrainingError(f"{record_path} was made with other settings / inputs")
        print(f"  {name}: cached", flush=True)
        return old
    unit_start = time.time()
    train, test, inner_train, inner_val = split.train, split.test, split.inner_train, split.inner_val
    tuning = []
    for cell_index, cell in enumerate(settings.grid):
        fit = train_one_fit(dataset, cell, inner_train, inner_val, seed=K.BASE_SEED, n_epochs=settings.n_epochs,
                            log_every=log_every)
        leak = check_no_leak(fit, dataset, inner_train, (inner_val, test), test)
        antisymmetry = check_antisymmetry(fit, dataset, test, seed=K.BASE_SEED + cell_index)
        validation_kcal = np.abs(predictions_kcal(fit, dataset, inner_val) - dataset.targets_kcal[inner_val]).mean(0)
        tuning.append(dict(cell=cell, grid_index=cell_index, best_epoch=fit.best_epoch, val=fit.best_validation,
                           val_per_target=fit.best_validation_per_target,
                           val_kcal_mae=dict(zip(K.TARGETS, validation_kcal.tolist())), epochs_run=len(fit.history),
                           seconds=fit.seconds, gate_M1=antisymmetry, sy=fit.target_scale.tolist(), leak=leak,
                           history=[dict(epoch=h["epoch"], loss=h["loss"], val=h.get("val")) for h in fit.history]))
        print(f"  {name} cell {cell}: val {fit.best_validation:.4f} at epoch {fit.best_epoch} ({fit.seconds:.0f} s) "
              f"gate {antisymmetry}", flush=True)
        if not antisymmetry["ok"]:
            raise TrainingError(f"antisymmetry check failed ({name}, cell {cell}): {antisymmetry}")
        del fit
    best = min(range(len(tuning)), key=lambda i: (tuning[i]["val"], i))        # ties: earlier grid cell
    cell, best_epoch = tuning[best]["cell"], tuning[best]["best_epoch"]
    seed_predictions, finals = [], []
    for seed in settings.seeds:
        fit = train_one_fit(dataset, cell, train, None, seed=seed, n_epochs=best_epoch, log_every=log_every)
        leak = check_no_leak(fit, dataset, train, (test,), test)
        antisymmetry = check_antisymmetry(fit, dataset, test, seed=seed)
        if not antisymmetry["ok"]:
            raise TrainingError(f"antisymmetry check failed ({name}, seed {seed}): {antisymmetry}")
        seed_predictions.append(predictions_kcal(fit, dataset, test))
        finals.append(dict(seed=seed, epochs=best_epoch, seconds=fit.seconds, gate_M1=antisymmetry,
                           sy=fit.target_scale.tolist(), leak=leak, std_sha256=fit.standardizer.sha256(),
                           history_tail=fit.history[-3:]))
        save_model_file(out_dir / f"{name}__seed{seed - K.BASE_SEED}.pt", fit, cell, dataset,
                        dict(fold=fold, seed=seed, best_epoch=best_epoch))
        del fit
    stacked = np.stack(seed_predictions)                                   # [seeds, n_test, 4] kcal/mol
    pair_ids = dataset.pairs.index.to_numpy()[test].astype(str)
    frames = [pd.DataFrame(dict(pair_id=pair_ids, ref_kind=dataset.pairs.ref_kind.to_numpy()[test],
                                pos=dataset.pairs.pos.to_numpy()[test], fold=int(fold), rung="lf3", arm="S", model="GNN",
                                target=target, y=dataset.targets_kcal[test, j], yhat=stacked[:, :, j].mean(0), base=0.0,
                                beta=0.0, yhat_seed_sd=stacked[:, :, j].std(0)))
              for j, target in enumerate(K.TARGETS)]
    predictions = pd.concat(frames, ignore_index=True)[PREDICTION_COLUMNS]
    parquet_path = out_dir / f"{name}.parquet"
    temporary = parquet_path.with_name(f".{parquet_path.name}.{os.getpid()}.tmp")
    predictions.to_parquet(temporary, index=False)
    os.replace(temporary, parquet_path)
    from .data import file_sha256
    record = dict(name=name, settings=fingerprint, tune=tuning,
                  chosen=dict(cell=cell, best_epoch=best_epoch, index=best, grid_index=tuning[best]["grid_index"]),
                  final=finals, val_kcal_mae=tuning[best]["val_kcal_mae"], n_train=int(len(train)), n_test=int(len(test)),
                  n_inner_train=int(len(inner_train)), n_inner_val=int(len(inner_val)), dropped=split.dropped,
                  parquet_sha256=file_sha256(parquet_path), environment=environment(),
                  wall_seconds=time.time() - unit_start, host=socket.gethostname(), slurm_job=os.environ.get("SLURM_JOB_ID"))
    write_json(record_path, record)
    return record


# =========================================================================== final model (one network per seed)
def choose_final_setting(unit_records: list[dict]) -> dict:
    """Pick the final cell and epoch count from the 5 cross-validation units (= ch3m train_m.choose_final_grid).

    Rule: the cell chosen by most folds (ties: lowest mean tuning validation value of that cell over all units, then grid
    order); epochs = median of the 5 best epochs (as int).

    Returns
    -------
    dict(cell, n_epochs, cell_counts, best_epochs, rule)
    """
    def key(cell):
        return json.dumps({k: int(v) for k, v in cell.items()}, sort_keys=True)
    counts = Counter(key(r["chosen"]["cell"]) for r in unit_records)
    top = max(counts.values())
    candidates = [k for k in counts if counts[k] == top]
    grid_position = {key(c): i for i, c in enumerate(K.GRID)}

    def mean_validation(k):
        values = [t["val"] for r in unit_records for t in r["tune"] if key(t["cell"]) == k]
        return float(np.mean(values)) if values else np.inf
    chosen = min(candidates, key=lambda k: (mean_validation(k), grid_position[k]))
    epochs = sorted(int(r["chosen"]["best_epoch"]) for r in unit_records)
    return dict(cell=json.loads(chosen), n_epochs=int(np.median(epochs)), cell_counts=dict(counts), best_epochs=epochs,
                tie_break=dict(candidates=candidates, mean_tuning_val={k: mean_validation(k) for k in candidates}),
                rule="mode of the chosen cells (ties: lowest mean tuning validation, then grid order); "
                     "n_epochs = median(best_epoch)")


def train_final_model(dataset: PairDataset, setting: dict, out_dir: Path, seeds=None, check_pairs: int = 1000) -> dict:
    """Train the deployable GNN3: ``setting`` (from ``choose_final_setting``) on every training pair, 5 seeds
    (= ch3m train_m.train_final). Writes ``gnn3_final__seed<j>.pt``, ``final_report.json`` and the predictions of
    ``check_pairs`` random training pairs (for checking a predictor later on the same GPU).

    Not run for the published results (out of scope of the clean-up); kept so that the procedure is documented in code.
    """
    seeds = list(K.FINAL_SEEDS) if seeds is None else list(seeds)
    out_dir = Path(out_dir)
    split = dataset.all_training_pairs()
    train = split.train
    epochs = int(setting["n_epochs"])
    rng = np.random.default_rng(K.BASE_SEED)
    check = np.sort(rng.choice(train, size=min(check_pairs, len(train)), replace=False))
    finals, check_predictions, first_std = [], [], None
    start = time.time()
    for seed in seeds:
        fit = train_one_fit(dataset, setting["cell"], train, None, seed=seed, n_epochs=epochs, log_every=10)
        leak = check_no_leak(fit, dataset, train, (), np.zeros(0, dtype=int))
        antisymmetry = check_antisymmetry(fit, dataset, train, seed=seed)
        if not antisymmetry["ok"]:
            raise TrainingError(f"antisymmetry check failed (final, seed {seed}): {antisymmetry}")
        if first_std is None:
            first_std = fit.standardizer.sha256()
        elif fit.standardizer.sha256() != first_std:
            raise TrainingError("final fits: standardisation differs between seeds")
        check_predictions.append(predictions_kcal(fit, dataset, check))
        finals.append(dict(seed=seed, epochs=epochs, seconds=fit.seconds, gate_M1=antisymmetry, leak=leak,
                           std_sha256=fit.standardizer.sha256()))
        save_model_file(out_dir / f"gnn3_final__seed{seed - K.BASE_SEED}.pt", fit, setting["cell"], dataset,
                        dict(fold="all", seed=seed, n_epochs=epochs))
        del fit
    stacked = np.stack(check_predictions)
    rows = [pd.DataFrame(dict(pair_id=dataset.pair_ids(check), target=t, yhat=stacked[:, :, j].mean(0),
                              yhat_sd=stacked[:, :, j].std(0))) for j, t in enumerate(K.TARGETS)]
    pd.concat(rows, ignore_index=True).to_parquet(out_dir / "final_check_predictions.parquet", index=False)
    report = dict(setting=setting, n_train_pairs=int(len(train)), train_pairs_sha256=list_sha256(dataset.pair_ids(train)),
                  finals=finals, environment=environment(), wall_seconds=time.time() - start)
    write_json(out_dir / "final_report.json", report)
    return report
