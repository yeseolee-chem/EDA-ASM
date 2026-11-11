"""Predicting pair ΔΔ with saved GNN3 networks: seed mean, seed spread and the applicability flag.

A prediction is made by several networks of the same setting (the 5 seeds of a final model, or the 5 seeds of one
cross-validation fold). Each network carries its own training statistics (``std``) and standardises the inputs with
them; the ΔΔ is the mean over the networks and the "seed spread" is their standard deviation (ddof 0). A seed spread
above the 95th percentile seen on the cross-validation test pairs flags the pair as outside the applicability domain.

Re-implements deploy/predict_tf.py (load_models, std_object, InputSet, check_reactions, predict_all, run_predict
direct read-out) for the GNN in 3-channel mode. Model files of the original pipeline load unchanged (weight names are
translated by ``model.load_weights``).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import constants as K
from .data import ReactionTokens, Standardizer, TokenLayout, default_device, file_sha256
from .model import GNNPairModel, build_model, load_weights, predict_standardized

DEVICE = default_device()
#: Applicability thresholds of GNN3 (kcal/mol): 95th percentile of the cross-validation test-pair seed spread
#: (analysis/ch3m/results/EM_ad.json, key GNN3). A pair whose seed spread exceeds the threshold is flagged.
APPLICABILITY_THRESHOLDS_KCAL = {"strain": 0.9861354575972047, "inter": 0.5885272840111005,
                                 "solv": 0.5068039061957793, "barrier": 0.902111174521413}


@dataclass
class LoadedNetwork:
    """One saved network ready for prediction."""

    path: str
    sha256: str
    fold: int | str | None
    seed_index: int | None
    model: GNNPairModel
    standardizer: Standardizer
    layout: TokenLayout
    cell: dict


def load_network(path: Path, device=DEVICE) -> LoadedNetwork:
    """Load one model file (written by this package or by ch3m/train_m.py).

    Raises
    ------
    ValueError
        The file is not a 4-target GNN model.
    """
    payload = torch.load(path, map_location=device)
    if payload.get("model", "GNN") != "GNN" or list(payload["TARGETS"]) != list(K.TARGETS):
        raise ValueError(f"{path}: not a GNN3 model file (model {payload.get('model')}, targets {payload['TARGETS']})")
    model = build_model(payload["cell"], int(payload["d_in"]), int(payload["d_glob"]), len(K.TARGETS)).to(device)
    load_weights(model, payload["state"])
    model.eval()
    unit = payload.get("unit") or {}
    return LoadedNetwork(path=str(path), sha256=file_sha256(path), fold=payload.get("fold", unit.get("fold")),
                         seed_index=None, model=model, standardizer=Standardizer.from_saved(payload["std"]),
                         layout=TokenLayout.from_saved(payload["layout"]), cell=payload["cell"])


def load_networks(paths: list[Path]) -> list[LoadedNetwork]:
    """Load several model files; they must share one token layout (else ValueError)."""
    networks = [load_network(Path(p)) for p in paths]
    layouts = {json.dumps(n.layout.as_saved(), sort_keys=True) for n in networks}
    if len(layouts) > 1:
        raise ValueError("the model files use different token layouts")
    for index, network in enumerate(networks):
        network.seed_index = index
    return networks


def predict_with_networks(networks: list[LoadedNetwork], reactions: ReactionTokens, rows_a, rows_b) -> np.ndarray:
    """ΔΔ of every network for the pairs (rows_a[i], rows_b[i]).

    Parameters
    ----------
    networks : list of LoadedNetwork
    reactions : ReactionTokens holding every reaction of the pairs
    rows_a, rows_b : array [n_pairs] int (rows of ``reactions``)

    Returns
    -------
    numpy.ndarray [n_networks, n_pairs, 4] (kcal/mol)
    """
    out = []
    for network in networks:
        tensors = network.standardizer.tensors(reactions, device=DEVICE)       # each network's own statistics
        out.append(predict_standardized(network.model, tensors, rows_a, rows_b) * network.standardizer.target_scale)
        del tensors
    return np.stack(out) if out else np.zeros((0, len(rows_a), len(K.TARGETS)))


def summarize_networks(per_network: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation (ddof 0) over the networks: [n_networks, n_pairs, 4] -> two [n_pairs, 4]."""
    return per_network.mean(0), per_network.std(0)


def check_new_reactions(token_dir: Path, table_path: Path, reaction_ids, layout: TokenLayout) -> tuple[list, dict]:
    """Input checks before predicting new reactions (= deploy predict_tf.check_reactions).

    A reaction is accepted when its token file exists, has the model's token version, elements and column widths,
    finite values, exactly 2 forming bonds, and a finite 182-column table row equal to the token copy.

    Returns
    -------
    (accepted rxn ids, {rxn_id: reason} of the rejected ones)
    """
    table = pd.read_parquet(table_path)
    columns = [c for c in table.columns if c != "rxn_id"]
    if tuple(columns) != layout.global_columns:
        raise ValueError(f"{table_path}: the 182 column names / order differ from the model layout")
    table = table.assign(rxn_id=table.rxn_id.astype(int)).set_index("rxn_id")
    elements = set(layout.elements)
    rejected = {}
    for rid in sorted({int(r) for r in reaction_ids}):
        path = Path(token_dir) / f"{rid}.npz"
        if not path.is_file():
            rejected[rid] = "token file missing"
            continue
        token = np.load(path)
        version = str(token["token_version"][0]) if "token_version" in token.files else None
        n_atoms = len(token["Z"])
        reason = None
        if version != layout.token_version:
            reason = f"token_version {version} != {layout.token_version}"
        elif not {int(z) for z in token["Z"]} <= elements:
            reason = "elements outside the model elements"
        elif token["bin"].shape != (n_atoms, len(layout.binary_columns)) or \
                token["num"].shape != (n_atoms, len(layout.numeric_columns)):
            reason = "token widths differ from the model layout"
        elif not (np.isfinite(token["num"]).all() and np.isfinite(token["pos"]).all() and np.isfinite(token["glob"]).all()):
            reason = "non-finite token values"
        elif int((token["etype"] == 2).sum()) != 2:
            reason = "the TS does not have exactly 2 forming bonds"
        elif rid not in table.index:
            reason = "no 182-column table row"
        else:
            row = table.loc[rid, columns].to_numpy(dtype=float)
            if not np.isfinite(row).all():
                reason = "non-finite 182-column values"
            elif (np.abs(row - token["glob"]) > K.GLOBAL_TABLE_RELATIVE_TOLERANCE
                  * np.maximum(1.0, np.abs(token["glob"]))).any():
                reason = "token copy of the 182 features differs from the table"
        if reason:
            rejected[rid] = reason
    accepted = sorted({int(r) for r in reaction_ids} - set(rejected))
    return accepted, rejected


def predict_pairs(model_files: list[Path], token_dir: Path, table_path: Path, pairs: pd.DataFrame,
                  thresholds: dict | None = APPLICABILITY_THRESHOLDS_KCAL) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict the ΔΔ of pairs of (possibly new) reactions from their token files.

    Parameters
    ----------
    model_files : list of Path
        The networks to average (e.g. the 5 seeds of the final model).
    token_dir : Path
        ``<rxn_id>.npz`` atom tokens of every reaction (``scripts/prepare_inputs.py`` output).
    table_path : Path
        Parquet with rxn_id + the 182 global features.
    pairs : DataFrame with columns pair_id, a_rxn (substituted reaction), b_rxn (its H reference / anchor).

    Returns
    -------
    (predictions, excluded)
        predictions: pair_id, a_rxn, b_rxn, target, yhat (kcal/mol), yhat_sd (seed spread), out_of_range;
        excluded: pair_id, a_rxn, b_rxn, reason (a reaction failed the input checks).

    Examples
    --------
    >>> out, excluded = predict_pairs(sorted(Path("model").glob("gnn3_final__seed*.pt")), Path("tokens"),
    ...                               Path("X182.parquet"), pd.DataFrame(dict(pair_id=["p1"], a_rxn=[11], b_rxn=[10])))
    ... # doctest: +SKIP
    """
    networks = load_networks(model_files)
    layout = networks[0].layout
    pairs = pairs.assign(pair_id=pairs.pair_id.astype(str), a_rxn=pairs.a_rxn.astype(int), b_rxn=pairs.b_rxn.astype(int))
    accepted, rejected = check_new_reactions(token_dir, table_path, set(pairs.a_rxn) | set(pairs.b_rxn), layout)
    keep = pairs.a_rxn.isin(accepted) & pairs.b_rxn.isin(accepted)
    reasons = [f"a {a}: {rejected[a]}" if a in rejected else (f"b {b}: {rejected[b]}" if b in rejected else "")
               for a, b in zip(pairs.a_rxn, pairs.b_rxn)]
    excluded = pairs.loc[~keep, ["pair_id", "a_rxn", "b_rxn"]].assign(
        reason=np.array(reasons, dtype=object)[~keep.to_numpy()])
    kept = pairs[keep].reset_index(drop=True)
    if not len(kept):
        return pd.DataFrame(columns=["pair_id", "a_rxn", "b_rxn", "target", "yhat", "yhat_sd", "out_of_range"]), excluded
    reactions = ReactionTokens.from_files(Path(token_dir), sorted(set(kept.a_rxn) | set(kept.b_rxn)), layout)
    rows_a = np.array([reactions.row_of[int(r)] for r in kept.a_rxn])
    rows_b = np.array([reactions.row_of[int(r)] for r in kept.b_rxn])
    mean, spread = summarize_networks(predict_with_networks(networks, reactions, rows_a, rows_b))
    frames = []
    for j, target in enumerate(K.TARGETS):
        limit = None if not thresholds else thresholds.get(target)
        frames.append(pd.DataFrame(dict(pair_id=kept.pair_id, a_rxn=kept.a_rxn, b_rxn=kept.b_rxn, target=target,
                                        yhat=mean[:, j], yhat_sd=spread[:, j],
                                        out_of_range=None if limit is None else spread[:, j] > limit)))
    return pd.concat(frames, ignore_index=True), excluded
