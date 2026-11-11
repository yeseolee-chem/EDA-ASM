"""Metrics reported for GNN3: pair MAE / RMSE / r / sign accuracy, ranking inside a family, cause agreement.

Definitions (identical to analysis/ch3/eval_ch3.py, with eval_geo2.boot_diff for confidence intervals):

pair sets
    P_main     reference pairs (b = the H reference) of the main evaluation set (``in_P_main``);
    P_sens_ref reference pairs of the sensitivity set (``in_P_sens``) - the set of the auxiliary tables;
    test_all   every test pair.
    "para" (pos == "p") is the headline position; o / m / direct are reported too.
confidence interval
    core bootstrap: the cores (dipole x skeleton) are resampled with replacement 2,000 times (seed 20261019); the
    interval is the 2.5 / 97.5 percentile of the resampled MAE (or MAE difference).
sign accuracy
    on pairs with |y| >= 1 kcal/mol: sign(yhat) == sign(y) (yhat = 0 counts as wrong).
ranking (barrier only)
    family = substituents sharing dipole, skeleton, series, ... and position; the H reference is added as a point with
    y = yhat = 0; families with >= 8 points. Pairwise order accuracy in |Δy| bins [0.41, 1), [1, 2), [2, inf); the share of
    families whose truly lowest barrier is predicted lowest (top 1) / among the 3 lowest (top 3); Spearman per family.
cause agreement (|y_barrier| >= 1)
    dominant channel = argmax over strain / inter / solv of value x sign(barrier); type: S (only strain pushes the
    barrier), I (only inter + solv), SI (both). Predicted labels use the model's own barrier sign. Single pairs and
    cell means (dipole x cfg x series x position x substituent, >= 4 pairs).

This module imports no torch: it runs in any numpy / pandas / scipy environment.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from . import constants as K
from .channels import labels_to_channels

CHANNELS = ["strain", "inter", "solv"]
TARGETS = list(K.TARGETS)
METRIC_SETS = ("P_main", "P_sens_ref", "test_all")
SET_DEFINITIONS = {"P_main": "ref_kind == reference & in_P_main", "P_sens_ref": "ref_kind == reference & in_P_sens",
                   "ref_all": "ref_kind == reference", "test_all": "every common test pair (reference + anchor)"}


# =========================================================================== inputs
def channel_truth(pair_labels: Path = K.PAIR_LABELS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The 4-target pair truth and the label-file attributes (= eval_ch3.labels).

    Returns
    -------
    (truth, attributes)
        truth: index pair_id, columns strain, inter, solv, barrier (kcal/mol) = dd_<10 labels> @ M;
        attributes: dipole, cfg ('-' when empty), series, pos_l, Xb, ref_kind_l, skeleton.
    """
    labels = pd.read_csv(pair_labels)
    labels["pair_id"] = labels.pair_id.astype(str)
    labels = labels.set_index("pair_id")
    truth = pd.DataFrame(labels_to_channels(labels[[f"dd_{t}" for t in K.LABEL_CHANNELS]].to_numpy(float)),
                         index=labels.index, columns=TARGETS)
    attributes = pd.DataFrame(dict(dipole=labels.dipole.astype(str),
                                   cfg=labels.cfg.fillna("").astype(str).replace("", "-"),
                                   series=labels.series.fillna("").astype(str), pos_l=labels.pos.fillna("").astype(str),
                                   Xb=labels.Xb.fillna("").astype(str), ref_kind_l=labels.ref_kind.fillna("").astype(str),
                                   skeleton=labels.skeleton.fillna("").astype(str)),
                              index=labels.index)
    return truth, attributes


def pair_sets(pair_attributes: Path = K.PAIR_ATTRIBUTES) -> pd.DataFrame:
    """Per-pair evaluation attributes from S_folds.csv (= eval_ch3.s_folds): core, fam, sigma, in_P_main, in_P_sens,
    ref_kind, pos; index pair_id."""
    table = pd.read_csv(pair_attributes)
    table["pair_id"] = table.pair_id.astype(str)
    table["pos"] = table.pos.fillna("").astype(str)
    table["ref_kind"] = table.ref_kind.fillna("").astype(str)
    return table[["pair_id", "core", "fam", "sigma", "in_P_main", "in_P_sens", "ref_kind", "pos"]].set_index("pair_id")


def assemble(predictions: dict[str, pd.DataFrame], truth: pd.DataFrame, attributes: pd.DataFrame,
             sets: pd.DataFrame) -> pd.DataFrame:
    """One long frame of every model's predictions with the truth and all attributes (= eval_ch3.assemble).

    Parameters
    ----------
    predictions : {model name: DataFrame with pair_id, fold, target, yhat, yhat_seed_sd}
    truth, attributes : from ``channel_truth``;  sets : from ``pair_sets``

    Returns
    -------
    DataFrame: model, pair_id, fold, target, y, yhat, yhat_seed_sd + core, fam, sigma, in_P_main, in_P_sens,
    ref_kind, pos, dipole, cfg, series, Xb, skeleton.

    Raises
    ------
    ValueError
        A pair without labels / attributes, or position / reference kind differing between the two attribute files.
    """
    frames = []
    for name, frame in predictions.items():
        frame = frame.copy()
        missing = sorted(set(frame.pair_id) - set(sets.index))
        if missing:
            raise ValueError(f"{name}: {len(missing)} pairs without evaluation attributes: {missing[:5]}")
        positions = truth.index.get_indexer(frame.pair_id)
        if (positions < 0).any():
            raise ValueError(f"{name}: pairs without labels")
        frame["y"] = truth.to_numpy()[positions, np.array([TARGETS.index(t) for t in frame.target])]
        frame["model"] = name
        frames.append(frame)
    long = pd.concat(frames, ignore_index=True).join(sets, on="pair_id").join(attributes, on="pair_id")
    if (long.pos != long.pos_l).any() or (long.ref_kind != long.ref_kind_l).any():
        raise ValueError("position / reference kind differ between S_folds.csv and the label file")
    long = long.drop(columns=["pos_l", "ref_kind_l"])
    long["in_P_main"] = long.in_P_main.astype(bool)
    long["in_P_sens"] = long.in_P_sens.astype(bool)
    return long


def set_mask(frame: pd.DataFrame, name: str) -> np.ndarray:
    """Boolean row mask of an evaluation set (see ``SET_DEFINITIONS``)."""
    reference = (frame.ref_kind == "reference").to_numpy()
    if name == "P_main":
        return reference & frame.in_P_main.to_numpy(bool)
    if name == "P_sens_ref":
        return reference & frame.in_P_sens.to_numpy(bool)
    if name == "ref_all":
        return reference
    if name == "test_all":
        return np.ones(len(frame), dtype=bool)
    raise ValueError(f"unknown set {name}")


def wide(long: pd.DataFrame, model: str, pair_ids=None) -> pd.DataFrame:
    """One row per pair: y_<target>, yhat_<target> + attributes (= eval_ch3.wide)."""
    frame = long[long.model == model]
    if pair_ids is not None:
        frame = frame[frame.pair_id.isin(pair_ids)]
    truth = frame.pivot(index="pair_id", columns="target", values="y")[TARGETS].add_prefix("y_")
    predicted = frame.pivot(index="pair_id", columns="target", values="yhat")[TARGETS].add_prefix("yhat_")
    columns = ["core", "fam", "sigma", "in_P_main", "in_P_sens", "ref_kind", "pos", "dipole", "cfg", "series", "Xb", "fold"]
    if "skeleton" in frame.columns:
        columns.append("skeleton")
    return pd.concat([truth, predicted], axis=1).join(frame.drop_duplicates("pair_id").set_index("pair_id")[columns])


# =========================================================================== statistics
def bootstrap_mae_difference(errors_x, errors_b, cores, n_resamples: int = K.BOOTSTRAP_RESAMPLES,
                             seed: int = K.BOOTSTRAP_SEED) -> tuple[float, float, float, float]:
    """MAE(x) - MAE(b) with a core bootstrap interval (= eval_geo2.boot_diff, copied verbatim in arithmetic).

    Parameters
    ----------
    errors_x, errors_b : array [n] (kcal/mol); pass zeros for errors_b to get the MAE of x and its interval
    cores : array [n] group label of each pair

    Returns
    -------
    (difference, lower 2.5 %, upper 97.5 %, two-sided p)
    """
    abs_x, abs_b = np.abs(errors_x), np.abs(errors_b)
    unique_cores, core_index = np.unique(cores, return_inverse=True)
    sum_x, sum_b, count = np.bincount(core_index, abs_x), np.bincount(core_index, abs_b), np.bincount(core_index)
    rng = np.random.default_rng(seed)
    n_cores = len(unique_cores)
    weights = np.stack([np.bincount(rng.integers(0, n_cores, n_cores), minlength=n_cores) for _ in range(n_resamples)])
    n_resampled = weights @ count
    resampled = (weights @ sum_x) / n_resampled - (weights @ sum_b) / n_resampled
    difference = abs_x.mean() - abs_b.mean()
    lower, upper = np.percentile(resampled, [2.5, 97.5])
    p_value = min(1.0, 2 * min((1 + (resampled <= 0).sum()) / (n_resamples + 1),
                               (1 + (resampled >= 0).sum()) / (n_resamples + 1)))
    return float(difference), float(lower), float(upper), float(p_value)


def pearson(a, b) -> float:
    """Pearson r; NaN for fewer than 3 points or a constant input."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def sign_accuracy(truth, predicted, threshold: float = K.SIGN_THRESHOLD_KCAL) -> tuple[float, int]:
    """Share of pairs with |truth| >= threshold whose predicted sign is right; and their count."""
    truth, predicted = np.asarray(truth, float), np.asarray(predicted, float)
    scored = np.abs(truth) >= threshold
    return (float((np.sign(predicted[scored]) == np.sign(truth[scored])).mean()) if scored.any() else np.nan), int(scored.sum())


# =========================================================================== tables
def error_table(long: pd.DataFrame, pair_ids, models, sets=METRIC_SETS) -> pd.DataFrame:
    """MAE, RMSE, Pearson r, sign accuracy per model x set x target x position (= eval_ch3.metrics).

    The P_main / para rows also carry the core-bootstrap 95 % interval of the MAE (mae_ci_lo, mae_ci_hi).
    """
    rows = []
    for model in models:
        model_rows = long[(long.model == model) & long.pair_id.isin(pair_ids)]
        for set_name in dict.fromkeys(sets):
            in_set = model_rows[set_mask(model_rows, set_name)]
            for target in TARGETS:
                frame = in_set[in_set.target == target]
                errors = (frame.yhat - frame.y).to_numpy(float)
                for position in K.POSITIONS:
                    chosen = np.ones(len(frame), dtype=bool) if position == "all" else (frame.pos == position).to_numpy()
                    if not chosen.any():
                        continue
                    accuracy, n_scored = sign_accuracy(frame.y.to_numpy()[chosen], frame.yhat.to_numpy()[chosen])
                    row = dict(model=model, eval_set=set_name, target=target, pos=position, n=int(chosen.sum()),
                               mae=float(np.abs(errors[chosen]).mean()), rmse=float(np.sqrt((errors[chosen] ** 2).mean())),
                               r=pearson(frame.y.to_numpy()[chosen], frame.yhat.to_numpy()[chosen]), sign_acc=accuracy,
                               n_sign=n_scored)
                    if position == "p" and set_name == "P_main":
                        cores = frame.core.to_numpy()[chosen]
                        _, lower, upper, _ = bootstrap_mae_difference(errors[chosen], np.zeros(int(chosen.sum())), cores)
                        row.update(n_cores=int(len(set(cores))), mae_ci_lo=lower, mae_ci_hi=upper)
                    rows.append(row)
    return pd.DataFrame(rows)


def _ranking_families(table: pd.DataFrame, scope: str):
    """Barrier points per family (family x position) with the H reference added as (0, 0); >= RANK_MIN_POINTS points."""
    families = []
    selected = table if scope == "all" else table[table.pos == "p"]
    for (family, position), group in selected.groupby(["fam", "pos"], sort=True):
        group = group.drop_duplicates("Xb")
        truth = np.r_[group.y_barrier.to_numpy(float), 0.0]
        predicted = np.r_[group.yhat_barrier.to_numpy(float), 0.0]
        if len(truth) >= K.RANK_MIN_POINTS:
            families.append((family, position, truth, predicted))
    return families


def ranking_table(long: pd.DataFrame, pair_ids, models, rank_set: str = "P_sens_ref") -> pd.DataFrame:
    """Within-family ranking of the barrier (= eval_ch3.rank_table); scopes para and all."""
    from scipy.stats import spearmanr
    rows = []
    for model in models:
        table = wide(long, model, pair_ids)
        table = table[set_mask(table, rank_set)]
        for scope in ("para", "all"):
            families = _ranking_families(table, scope)
            bins = {label: [0, 0] for _, _, label in K.RANK_BINS}
            top1, top3, expected1, expected3, rhos = [], [], [], [], []
            for _, _, truth, predicted in families:
                n_points = len(truth)
                for i, j in combinations(range(n_points), 2):
                    true_gap, predicted_gap = truth[i] - truth[j], predicted[i] - predicted[j]
                    for low, high, label in K.RANK_BINS:
                        if low <= abs(true_gap) < high:
                            bins[label][0] += int(np.sign(predicted_gap) == np.sign(true_gap))
                            bins[label][1] += 1
                lowest = int(np.argmin(truth))
                rank = 1 + int((predicted < predicted[lowest]).sum())
                top1.append(rank == 1)
                top3.append(rank <= 3)
                expected1.append(1.0 / n_points)
                expected3.append(min(3, n_points) / n_points)
                rhos.append(float(spearmanr(truth, predicted).correlation))
            row = dict(model=model, scope=scope, rank_set=rank_set, n_families=len(families),
                       n_points=int(sum(len(f[2]) for f in families)))
            for label, (correct, total) in bins.items():
                row[f"order_acc_{label}"] = correct / total if total else np.nan
                row[f"order_n_{label}"] = total
            rhos = np.asarray(rhos, float)
            row.update(lowest_top1=float(np.mean(top1)) if families else np.nan,
                       lowest_top1_random=float(np.mean(expected1)) if families else np.nan,
                       lowest_top3=float(np.mean(top3)) if families else np.nan,
                       lowest_top3_random=float(np.mean(expected3)) if families else np.nan,
                       spearman_median=float(np.nanmedian(rhos)) if len(rhos) else np.nan,
                       spearman_q25=float(np.nanpercentile(rhos, 25)) if len(rhos) else np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


def dominant_channel(values: np.ndarray, barrier_sign: np.ndarray) -> np.ndarray:
    """argmax over strain / inter / solv of value x barrier sign. values [n, 3], barrier_sign [n] -> names [n]."""
    return np.array(CHANNELS, dtype=object)[np.argmax(values * barrier_sign[:, None], axis=1)]


def cause_type(values: np.ndarray, barrier_sign: np.ndarray) -> np.ndarray:
    """S = strain pushes the barrier and inter + solv do not; I = the reverse; SI = both; none = neither."""
    strain_pushes = values[:, 0] * barrier_sign > 0
    interaction_pushes = (values[:, 1] + values[:, 2]) * barrier_sign > 0
    labels = np.full(len(barrier_sign), "none", dtype=object)
    labels[strain_pushes & ~interaction_pushes] = "S"
    labels[strain_pushes & interaction_pushes] = "SI"
    labels[~strain_pushes & interaction_pushes] = "I"
    return labels


def _agreement(truth: np.ndarray, predicted: np.ndarray, sign_mode: str = "own") -> dict:
    """Dominant-channel / type agreement on rows with |y_barrier| >= 1 (= eval_ch3.agree). Arrays [n, 4]."""
    scored = np.abs(truth[:, 3]) >= K.SIGN_THRESHOLD_KCAL
    truth, predicted = truth[scored], predicted[scored]
    true_sign = np.sign(truth[:, 3])
    predicted_sign = np.sign(predicted[:, 3]) if sign_mode == "own" else true_sign
    out = dict(n=int(scored.sum()))
    for key, label_of in (("dominant", dominant_channel), ("type", cause_type)):
        true_labels, predicted_labels = label_of(truth[:, :3], true_sign), label_of(predicted[:, :3], predicted_sign)
        shares = pd.Series(true_labels).value_counts(normalize=True)
        out[key] = float((true_labels == predicted_labels).mean()) if len(true_labels) else np.nan
        out[f"{key}_base"] = float(shares.iloc[0]) if len(shares) else np.nan
        out[f"{key}_base_class"] = str(shares.index[0]) if len(shares) else None
        out[f"{key}_true_dist"] = {str(k): float(v) for k, v in shares.items()}
    return out


def cause_agreement(long: pd.DataFrame, pair_ids, model: str, cause_set: str = "P_sens_ref",
                    sign_mode: str = "own") -> dict:
    """Cause agreement of one model: single pairs, cell means, and the cell-mean fit per target (= eval_ch3.cause)."""
    table = wide(long, model, pair_ids)
    table = table[set_mask(table, cause_set)]
    truth_columns, predicted_columns = [f"y_{t}" for t in TARGETS], [f"yhat_{t}" for t in TARGETS]
    single = _agreement(table[truth_columns].to_numpy(float), table[predicted_columns].to_numpy(float), sign_mode)
    groups = table.groupby(["dipole", "cfg", "series", "pos", "Xb"])
    size = groups.size()
    cell_means = groups[truth_columns + predicted_columns].mean()[size >= K.CELL_MIN_PAIRS]
    cell = _agreement(cell_means[truth_columns].to_numpy(float), cell_means[predicted_columns].to_numpy(float), sign_mode)
    cell["n_cells_all"] = int(len(cell_means))
    cell_fit = {t: dict(r=pearson(cell_means[f"y_{t}"], cell_means[f"yhat_{t}"]),
                        mae=float(np.abs(cell_means[f"yhat_{t}"] - cell_means[f"y_{t}"]).mean()) if len(cell_means)
                        else np.nan) for t in TARGETS}
    return dict(model=model, cause_set=cause_set, sign_mode=sign_mode, n_pairs=int(len(table)), single=single, cell=cell,
                cell_fit=cell_fit)


def cause_table(results: list[dict]) -> pd.DataFrame:
    """Rows of ``cause_agreement`` results (= eval_ch3.cause_rows)."""
    rows = []
    for result in results:
        for level in ("single", "cell"):
            part = result[level]
            for key in ("dominant", "type"):
                rows.append(dict(model=result["model"], level=level, measure=key, cause_set=result["cause_set"],
                                 sign_mode=result["sign_mode"], n=part["n"], agreement=part[key],
                                 baseline=part[f"{key}_base"], baseline_class=part[f"{key}_base_class"],
                                 true_dist=json.dumps(part[f"{key}_true_dist"])))
        for target, fit in result["cell_fit"].items():
            rows.append(dict(model=result["model"], level="cell_all", measure=f"cell_mean_{target}",
                             cause_set=result["cause_set"], sign_mode=result["sign_mode"], n=result["cell"]["n_cells_all"],
                             r=fit["r"], mae=fit["mae"]))
    return pd.DataFrame(rows)


def identity_residual(long: pd.DataFrame, pair_ids, model: str) -> dict:
    """|yhat_barrier - (yhat_strain + yhat_inter + yhat_solv)| mean / 95th percentile / max, and the label maximum."""
    table = wide(long, model, pair_ids)
    residual = table.yhat_barrier - (table.yhat_strain + table.yhat_inter + table.yhat_solv)
    label_residual = table.y_barrier - (table.y_strain + table.y_inter + table.y_solv)
    return dict(model=model, yhat_mean_abs=float(residual.abs().mean()), yhat_p95_abs=float(np.percentile(residual.abs(), 95)),
                yhat_max_abs=float(residual.abs().max()), y_max_abs=float(label_residual.abs().max()), n=int(len(table)))


def breakdown_table(long: pd.DataFrame, pair_ids, model: str, sets=("P_main", "test_all")) -> pd.DataFrame:
    """MAE and R² per substituent position (direct / o / m / p) and per skeleton, for the model card.

    R² = 1 - sum (yhat - y)^2 / sum (y - mean y)^2 within the group. New in the clean-up (no original counterpart).
    """
    rows = []
    frame = long[(long.model == model) & long.pair_id.isin(pair_ids)]
    for set_name in sets:
        in_set = frame[set_mask(frame, set_name)]
        for target in TARGETS:
            part = in_set[in_set.target == target]
            for group_by in ("pos", "skeleton"):
                for value, group in part.groupby(group_by, sort=True):
                    y, yhat = group.y.to_numpy(float), group.yhat.to_numpy(float)
                    total = float(((y - y.mean()) ** 2).sum())
                    rows.append(dict(model=model, eval_set=set_name, target=target, group_by=group_by, group=str(value),
                                     n=int(len(group)), mae=float(np.abs(yhat - y).mean()),
                                     r2=float(1.0 - ((yhat - y) ** 2).sum() / total) if total > 0 else np.nan,
                                     r=pearson(y, yhat)))
    return pd.DataFrame(rows)


def applicability_thresholds(predictions: pd.DataFrame) -> dict:
    """95th percentile (numpy linear) of the test-pair seed spread per target (= ch3m eval_m.ad / eval_ch3.AD_PCT)."""
    return {t: float(np.percentile(g.yhat_seed_sd.to_numpy(float), K.APPLICABILITY_PERCENTILE))
            for t, g in predictions.groupby("target")}
