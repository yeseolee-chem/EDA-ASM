"""The GNN3 network: a message-passing encoder per reaction and an antisymmetric head per pair.

A reaction enters as its transition-state (TS) atoms ("tokens") with 3-D positions and a bond graph whose edges are
covalent bonds or the two forming bonds of the cycloaddition. The encoder turns it into one vector r. A pair (a, b)
is scored by

    f(a, b) = 1/2 [ h([r_a, r_b]) - h([r_b, r_a]) ]

where h is an MLP. This makes f antisymmetric BY CONSTRUCTION: f(b, a) = -f(a, b) and f(a, a) = 0, exactly as a
difference ΔΔ = label(a) - label(b) must behave. Outputs are in standardised units (multiply by the training target
scale to get kcal/mol).

This file re-implements ``analysis/d1_pair/fu2/models.py`` (classes GNN, MPLayer, AttnPool and the helpers rbf,
pdist, mlp, masked_sum) with readable names. Sub-modules are created in the SAME order as the original, so the same
seed gives the same initial weights, and every forward operation is the same, so a saved original model gives the
same numbers (equivalence checks E-2, E-3). ``ORIGINAL_KEY_PREFIXES`` maps the original weight names to the new ones.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from . import constants as K


# --------------------------------------------------------------------------- small building blocks
def radial_basis(distances: torch.Tensor, n_centres: int, max_distance: float) -> torch.Tensor:
    """Gaussian radial-basis expansion of distances.

    Parameters
    ----------
    distances : tensor [...] (angstrom)
    n_centres : int
        Number of Gaussians, centres evenly spaced on [0, max_distance].
    max_distance : float (angstrom)

    Returns
    -------
    tensor [..., n_centres]
        exp(-gamma (d - mu_k)^2) with gamma = 1 / spacing^2.
    """
    centres = torch.linspace(0.0, max_distance, n_centres, device=distances.device, dtype=distances.dtype)
    gamma = 1.0 / (centres[1] - centres[0]) ** 2
    return torch.exp(-gamma * (distances.unsqueeze(-1) - centres) ** 2)


def pairwise_distances(positions_angstrom: torch.Tensor) -> torch.Tensor:
    """All inter-atomic distances of each reaction.

    Parameters
    ----------
    positions_angstrom : tensor [n_reactions, N, 3]

    Returns
    -------
    tensor [n_reactions, N, N] (angstrom), zero on the diagonal.

    Notes
    -----
    A tiny 1e-12 inside the square root keeps the gradient finite for coincident (padded) atoms.
    """
    difference = positions_angstrom.unsqueeze(2) - positions_angstrom.unsqueeze(1)
    off_diagonal = 1 - torch.eye(positions_angstrom.shape[1], device=positions_angstrom.device,
                                 dtype=positions_angstrom.dtype)
    return torch.sqrt((difference ** 2).sum(-1) + 1e-12) * off_diagonal


def make_mlp(n_in: int, n_hidden_units: int, n_out: int, dropout: float, n_hidden_layers: int = 1) -> nn.Sequential:
    """[Linear -> SiLU -> Dropout] x n_hidden_layers -> Linear (= models.mlp; Sequential keeps the original key indices)."""
    layers, width = [], n_in
    for _ in range(n_hidden_layers):
        layers += [nn.Linear(width, n_hidden_units), nn.SiLU(), nn.Dropout(dropout)]
        width = n_hidden_units
    return nn.Sequential(*layers, nn.Linear(width, n_out))


def masked_sum(values: torch.Tensor, atom_mask: torch.Tensor) -> torch.Tensor:
    """Sum over the real atoms. values [r, N, W], atom_mask [r, N] -> [r, W]."""
    return (values * atom_mask.unsqueeze(-1).to(values.dtype)).sum(1)


class AttentionPool(nn.Module):
    """Attention pooling over the real atoms of each reaction: softmax(w . h_i) weighted sum of h_i (= models.AttnPool)."""

    def __init__(self, width: int):
        super().__init__()
        self.score = nn.Linear(width, 1)

    def forward(self, values: torch.Tensor, atom_mask: torch.Tensor) -> torch.Tensor:
        """values [r, N, W], atom_mask [r, N] bool -> [r, W]."""
        scores = self.score(values).squeeze(-1).masked_fill(~atom_mask, K.ATTENTION_MASK_VALUE)
        weights = torch.softmax(scores, dim=1) * atom_mask.to(values.dtype)
        return (weights.unsqueeze(-1) * values).sum(1)


class MessagePassingLayer(nn.Module):
    """One message-passing step on the TS bond graph (= models.MPLayer).

    For atom i:  m_i = sum over bonded j of psi(W_h h_j + W_e e_ij),  h_i <- LayerNorm(h_i + U([h_i, m_i]))
    with the edge vector e_ij = embedding(bond type) + MLP(radial basis of the i-j distance).

    Notes
    -----
    psi's first linear map on [h_j, e_ij] is split into W_h h_j (per atom) + W_e e_ij (per edge) to keep memory at
    O(N^2 W). Only bonded pairs (edge type 1 covalent or 2 forming) pass messages; distance enters through e_ij.
    """

    def __init__(self, width: int, n_radial_basis: int, max_distance_angstrom: float, dropout: float):
        super().__init__()
        self.n_radial_basis, self.max_distance_angstrom = n_radial_basis, max_distance_angstrom
        # creation order = models.MPLayer (etype, edge, w_h, w_e, msg_out, upd, norm): same seed -> same initial weights
        self.edge_type_embedding = nn.Embedding(K.N_EDGE_TYPES, width)
        self.edge_distance_mlp = nn.Sequential(nn.Linear(n_radial_basis, width), nn.SiLU(), nn.Linear(width, width))
        self.node_to_message = nn.Linear(width, width)
        self.edge_to_message = nn.Linear(width, width, bias=False)
        self.message_output = nn.Sequential(nn.SiLU(), nn.Dropout(dropout), nn.Linear(width, width))
        self.node_update = make_mlp(2 * width, width, width, dropout)
        self.layer_norm = nn.LayerNorm(width)

    def forward(self, node_states: torch.Tensor, distances: torch.Tensor, edge_types: torch.Tensor,
                adjacency: torch.Tensor) -> torch.Tensor:
        """node_states [r, N, W], distances [r, N, N], edge_types [r, N, N] long, adjacency [r, N, N] bool -> [r, N, W]."""
        edges = self.edge_type_embedding(edge_types) + self.edge_distance_mlp(
            radial_basis(distances, self.n_radial_basis, self.max_distance_angstrom))        # [r, N(i), N(j), W]
        pre_message = self.node_to_message(node_states).unsqueeze(1) + self.edge_to_message(edges)
        messages = (self.message_output(pre_message) * adjacency.unsqueeze(-1).to(node_states.dtype)).sum(2)
        return self.layer_norm(node_states + self.node_update(torch.cat([node_states, messages], -1)))


class GNNPairModel(nn.Module):
    """GNN encoder (shared by both reactions of a pair) + antisymmetric pair head (= models.GNN, variant "A+G").

    Parameters
    ----------
    token_width : int
        Width of an atom token (29).
    n_outputs : int
        Number of predicted quantities (4: strain, inter, solv, barrier).
    width : int
        Hidden width W of the atom states (grid: 64 or 128).
    n_layers : int
        Number of message-passing layers (grid: 4 or 6).
    n_global : int
        Number of global features (182); 0 = atom tokens only (variant "A", not used by GNN3).
    """

    def __init__(self, token_width: int, n_outputs: int, width: int = 64, n_layers: int = 4, n_global: int = 0,
                 n_radial_basis: int = K.N_RADIAL_BASIS,
                 max_distance_angstrom: float = K.RADIAL_BASIS_MAX_DISTANCE_ANGSTROM, dropout: float = K.DROPOUT):
        super().__init__()
        # creation order = models.GNN (embed, layers, pool, gproj, head): same seed -> same initial weights
        self.atom_embedding = nn.Sequential(nn.Linear(token_width, width), nn.SiLU())
        self.message_layers = nn.ModuleList([MessagePassingLayer(width, n_radial_basis, max_distance_angstrom, dropout)
                                             for _ in range(n_layers)])
        self.attention_pool = AttentionPool(width)
        self.global_projection = nn.Linear(n_global, width) if n_global else None
        reaction_width = 2 * width + (width if n_global else 0)
        self.pair_head = make_mlp(2 * reaction_width, 2 * width, n_outputs, dropout, n_hidden_layers=2)

    def encode_reactions(self, batch: dict) -> torch.Tensor:
        """Reaction vectors r = [sum of atom states | attention pool | projected global features].

        Parameters
        ----------
        batch : dict
            atom_features [r, N, 29], positions_angstrom [r, N, 3], atom_mask [r, N] bool, edge_types [r, N, N] long,
            global_features [r, 182] (see ``make_reaction_batch``).

        Returns
        -------
        tensor [r, 3 * width]
        """
        atom_mask, edge_types = batch["atom_mask"], batch["edge_types"]
        distances = pairwise_distances(batch["positions_angstrom"])
        adjacency = (edge_types > 0) & atom_mask.unsqueeze(2) & atom_mask.unsqueeze(1)
        node_states = self.atom_embedding(batch["atom_features"])
        for layer in self.message_layers:
            node_states = layer(node_states, distances, edge_types, adjacency)
        parts = [masked_sum(node_states, atom_mask), self.attention_pool(node_states, atom_mask)]
        if self.global_projection is not None:
            parts.append(self.global_projection(batch["global_features"]))
        return torch.cat(parts, -1)

    def forward(self, batch: dict, index_a: torch.Tensor, index_b: torch.Tensor) -> torch.Tensor:
        """Pair predictions f(a, b) in standardised units.

        Parameters
        ----------
        batch : dict of the unique reactions of the pairs (``make_reaction_batch``)
        index_a, index_b : tensor [n_pairs] long
            Position of each pair's a and b reaction in the batch.

        Returns
        -------
        tensor [n_pairs, n_outputs]
        """
        reaction_vectors = self.encode_reactions(batch)
        vector_a, vector_b = reaction_vectors[index_a], reaction_vectors[index_b]
        # the head sees [a, b] and [b, a]; their half difference is antisymmetric whatever the head learns
        return 0.5 * (self.pair_head(torch.cat([vector_a, vector_b], -1))
                      - self.pair_head(torch.cat([vector_b, vector_a], -1)))


def build_model(cell: dict, token_width: int, n_global: int, n_outputs: int = len(K.TARGETS),
                dropout: float = K.DROPOUT) -> GNNPairModel:
    """The GNN3 network of one grid cell (= models.build('GNN', cell, d_in, n_out, d_glob, 'A+G', dropout)).

    Examples
    --------
    >>> model = build_model(dict(n_layers=4, width=64), token_width=29, n_global=182)
    """
    return GNNPairModel(token_width, n_outputs, width=int(cell["width"]), n_layers=int(cell["n_layers"]),
                        n_global=n_global, dropout=dropout)


# --------------------------------------------------------------------------- original weight names
#: Prefix map from the weight names of ``fu2/models.py`` (left) to this file (right). Applied in order to every key of
#: a saved original ``state_dict`` so that the published model files load unchanged.
ORIGINAL_KEY_PREFIXES: tuple[tuple[str, str], ...] = (
    ("embed.", "atom_embedding."),
    ("pool.w.", "attention_pool.score."),
    ("gproj.", "global_projection."),
    ("head.", "pair_head."),
    ("layers.", "message_layers."),
)
#: Inside each message layer (``message_layers.<i>.``).
ORIGINAL_LAYER_PARTS: tuple[tuple[str, str], ...] = (
    ("etype.", "edge_type_embedding."),
    ("edge.", "edge_distance_mlp."),
    ("w_h.", "node_to_message."),
    ("w_e.", "edge_to_message."),
    ("msg_out.", "message_output."),
    ("upd.", "node_update."),
    ("norm.", "layer_norm."),
)


def rename_original_key(key: str) -> str:
    """Translate one weight name of the original model into this file's name (unknown names raise KeyError).

    Examples
    --------
    >>> rename_original_key("layers.0.w_h.weight")
    'message_layers.0.node_to_message.weight'
    """
    for old, new in ORIGINAL_KEY_PREFIXES:
        if key.startswith(old):
            rest = key[len(old):]
            if old == "layers.":
                index, part = rest.split(".", 1)
                for old_part, new_part in ORIGINAL_LAYER_PARTS:
                    if part.startswith(old_part):
                        return f"{new}{index}.{new_part}{part[len(old_part):]}"
                raise KeyError(key)
            return new + rest
    raise KeyError(key)


def load_weights(model: GNNPairModel, state: dict) -> GNNPairModel:
    """Load a state_dict saved by this package OR by the original pipeline (detected by its key names)."""
    if any(k.startswith(("embed.", "layers.")) for k in state):
        state = {rename_original_key(k): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model


# --------------------------------------------------------------------------- batching and inference
def make_reaction_batch(tensors: dict, rows_a, rows_b) -> tuple[dict, torch.Tensor, torch.Tensor]:
    """Gather the unique reactions of a set of pairs into one padded batch.

    Parameters
    ----------
    tensors : dict
        Standardised tensors of ALL reactions (``Standardizer.tensors``).
    rows_a, rows_b : array or tensor [n_pairs] int
        Reaction rows of each pair's a and b.

    Returns
    -------
    (batch, index_a, index_b)
        batch: tensors of the unique reactions (sorted rows), trimmed to the largest reaction of the batch;
        index_a / index_b [n_pairs]: position of a / b in the batch.
    """
    device = tensors["atom_features"].device
    rows_a, rows_b = torch.as_tensor(rows_a, device=device), torch.as_tensor(rows_b, device=device)
    unique_rows, inverse = torch.unique(torch.cat([rows_a, rows_b]), return_inverse=True)
    max_atoms = int(tensors["n_atoms"][unique_rows].max())
    batch = dict(atom_features=tensors["atom_features"][unique_rows][:, :max_atoms],
                 positions_angstrom=tensors["positions_angstrom"][unique_rows][:, :max_atoms],
                 atom_mask=tensors["atom_mask"][unique_rows][:, :max_atoms],
                 edge_types=tensors["edge_types"][unique_rows][:, :max_atoms, :max_atoms].long(),
                 fragment_ids=tensors["fragment_ids"][unique_rows][:, :max_atoms].long(),
                 global_features=tensors["global_features"][unique_rows])
    return batch, inverse[:len(rows_a)], inverse[len(rows_a):]


@torch.no_grad()
def predict_standardized(model: GNNPairModel, tensors: dict, rows_a, rows_b,
                         batch_size: int = K.PREDICT_BATCH_PAIRS) -> np.ndarray:
    """Pair predictions in standardised units (eval mode, no dropout).

    Parameters
    ----------
    model : GNNPairModel
    tensors : dict (``Standardizer.tensors``)
    rows_a, rows_b : array [n_pairs] int
    batch_size : int

    Returns
    -------
    numpy.ndarray [n_pairs, n_outputs] float64
        Multiply by the training target scale (kcal/mol) to get ΔΔ in kcal/mol.
    """
    model.eval()
    chunks = []
    for start in range(0, len(rows_a), batch_size):
        batch, index_a, index_b = make_reaction_batch(tensors, rows_a[start:start + batch_size],
                                                      rows_b[start:start + batch_size])
        chunks.append(model(batch, index_a, index_b).double().cpu().numpy())
    return np.concatenate(chunks) if chunks else np.zeros((0, len(K.TARGETS)))
