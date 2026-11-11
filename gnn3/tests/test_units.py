"""Unit tests (E-6 of docs/specs/GNN3_CLEAN.md): CPU only, synthetic inputs, no data files.

    python gnn3/tests/test_units.py      built-in runner (env_d1fu2 has no pytest); writes gnn3/results/E_parts/E-6.json
    python -m pytest gnn3/tests/test_units.py      the same tests under pytest, where pytest is installed

Covered: antisymmetry f(a,b) = -f(b,a) and f(a,a) = 0, output shapes, the identity penalty, standardisation from the
training pairs only, the 10 -> 4 channel matrix, the original-weight key map (same seed -> same weights and outputs as
fu2/models.py), order augmentation and batching, the metric helpers.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO = PACKAGE_ROOT.parent
sys.path.insert(0, str(PACKAGE_ROOT))
from gnn3 import constants as K  # noqa: E402
from gnn3.channels import channel_matrix, labels_to_channels  # noqa: E402
from gnn3.data import Standardizer  # noqa: E402
from gnn3.metrics import cause_type, dominant_channel, sign_accuracy  # noqa: E402
from gnn3.model import (ORIGINAL_KEY_PREFIXES, build_model, load_weights, make_reaction_batch,  # noqa: E402
                        rename_original_key)
from gnn3.training import identity_penalty_loss, iterate_training_batches, swap_pair_order  # noqa: E402

CELL = dict(n_layers=4, width=64)


def synthetic_reactions(n_reactions=6, max_atoms=10, n_global=K.N_GLOBAL_FEATURES, seed=0, dtype=torch.float64):
    """Random padded reactions in the layout of ``Standardizer.tensors`` (sizes 4..max_atoms, a chain of bonds)."""
    rng = np.random.default_rng(seed)
    n_atoms = rng.integers(4, max_atoms + 1, n_reactions)
    n_atoms[0] = max_atoms
    features = np.zeros((n_reactions, max_atoms, K.ATOM_TOKEN_WIDTH))
    positions = np.zeros((n_reactions, max_atoms, 3))
    mask = np.zeros((n_reactions, max_atoms), dtype=bool)
    edges = np.zeros((n_reactions, max_atoms, max_atoms), dtype=np.int8)
    fragments = np.zeros((n_reactions, max_atoms), dtype=np.int8)
    for r, n in enumerate(n_atoms):
        features[r, :n] = rng.normal(size=(n, K.ATOM_TOKEN_WIDTH))
        positions[r, :n] = rng.normal(scale=1.5, size=(n, 3))
        mask[r, :n] = True
        fragments[r, n // 2:n] = 1
        for i in range(n - 1):
            edges[r, i, i + 1] = edges[r, i + 1, i] = 1
        edges[r, 0, n - 1] = edges[r, n - 1, 0] = 2                    # one "forming" bond
    return dict(atom_features=torch.tensor(features, dtype=dtype), positions_angstrom=torch.tensor(positions, dtype=dtype),
                atom_mask=torch.tensor(mask), edge_types=torch.tensor(edges), fragment_ids=torch.tensor(fragments),
                global_features=torch.tensor(rng.normal(size=(n_reactions, n_global)), dtype=dtype),
                n_atoms=torch.tensor(n_atoms))


def small_model(n_global=K.N_GLOBAL_FEATURES, seed=0):
    torch.manual_seed(seed)
    return build_model(CELL, K.ATOM_TOKEN_WIDTH, n_global).double().eval()


# --------------------------------------------------------------------------- model
def test_antisymmetry_and_self_difference_zero():
    model, tensors = small_model(), synthetic_reactions()
    rows_a, rows_b = np.array([0, 1, 2, 3, 4, 5, 0]), np.array([1, 2, 3, 4, 5, 0, 3])
    with torch.no_grad():
        f_ab = model(*make_reaction_batch(tensors, rows_a, rows_b))
        f_ba = model(*make_reaction_batch(tensors, rows_b, rows_a))
        f_aa = model(*make_reaction_batch(tensors, rows_a, rows_a))
    assert float((f_ab + f_ba).abs().max()) < 1e-12
    assert float(f_aa.abs().max()) == 0.0                              # h(x) - h(x) is exactly zero
    assert float(f_ab.abs().max()) > 0.0


def test_output_shapes():
    tensors = synthetic_reactions(n_reactions=5)
    model = small_model()
    batch, index_a, index_b = make_reaction_batch(tensors, np.array([0, 2, 4]), np.array([1, 1, 3]))
    with torch.no_grad():
        assert tuple(model.encode_reactions(batch).shape) == (5, 3 * CELL["width"])
        assert tuple(model(batch, index_a, index_b).shape) == (3, len(K.TARGETS))
    tokens_only = small_model(n_global=0)
    with torch.no_grad():
        assert tuple(tokens_only.encode_reactions(batch).shape) == (5, 2 * CELL["width"])


def test_reaction_batch_indices_and_trimming():
    tensors = synthetic_reactions(n_reactions=6, max_atoms=10)
    rows_a, rows_b = np.array([5, 3, 3]), np.array([1, 5, 4])
    batch, index_a, index_b = make_reaction_batch(tensors, rows_a, rows_b)
    unique = np.unique(np.r_[rows_a, rows_b])
    assert (unique[index_a.numpy()] == rows_a).all() and (unique[index_b.numpy()] == rows_b).all()
    n_max = int(tensors["n_atoms"][torch.as_tensor(unique)].max())
    assert batch["atom_features"].shape[1] == n_max and batch["edge_types"].shape[1:] == (n_max, n_max)
    assert batch["edge_types"].dtype == torch.long and batch["fragment_ids"].dtype == torch.long


def _original_models_module():
    sys.path.insert(0, str(REPO / "analysis" / "d1_pair" / "fu2"))
    import models as original                                          # fu2/models.py imports only torch
    return original


def test_original_weights_same_seed_same_numbers():
    """Same seed -> the original fu2 GNN and GNNPairModel get identical weights (after renaming) and outputs."""
    original = _original_models_module()
    for cell in K.GRID:
        torch.manual_seed(123)
        old = original.build("GNN", cell, K.ATOM_TOKEN_WIDTH, len(K.TARGETS), K.N_GLOBAL_FEATURES, "A+G", K.DROPOUT)
        torch.manual_seed(123)
        new = build_model(cell, K.ATOM_TOKEN_WIDTH, K.N_GLOBAL_FEATURES)
        old_state, new_state = old.state_dict(), new.state_dict()
        renamed = [rename_original_key(k) for k in old_state]
        assert renamed == list(new_state), "key order differs"
        for key, value in old_state.items():
            assert torch.equal(value, new_state[rename_original_key(key)]), key
        tensors = synthetic_reactions(dtype=torch.float32)
        rows_a, rows_b = np.array([0, 1, 2, 3]), np.array([4, 5, 5, 0])
        batch, index_a, index_b = make_reaction_batch(tensors, rows_a, rows_b)
        old_batch = dict(x=batch["atom_features"], pos=batch["positions_angstrom"], mask=batch["atom_mask"],
                         etype=batch["edge_types"], frag=batch["fragment_ids"], glob=batch["global_features"])
        old.eval(), new.eval()
        with torch.no_grad():
            assert torch.equal(old(old_batch, index_a, index_b), new(batch, index_a, index_b))
        reloaded = load_weights(build_model(cell, K.ATOM_TOKEN_WIDTH, K.N_GLOBAL_FEATURES).eval(), old_state)
        with torch.no_grad():
            assert torch.equal(reloaded(batch, index_a, index_b), new(batch, index_a, index_b))


def test_key_map_rejects_unknown_names():
    assert rename_original_key("layers.2.msg_out.2.weight") == "message_layers.2.message_output.2.weight"
    assert rename_original_key("pool.w.bias") == "attention_pool.score.bias"
    for bad in ("out.0.weight", "layers.0.unknown.weight"):
        try:
            rename_original_key(bad)
        except KeyError:
            continue
        raise AssertionError(f"{bad} was renamed")
    assert [old for old, _ in ORIGINAL_KEY_PREFIXES] == ["embed.", "pool.w.", "gproj.", "head.", "layers."]


# --------------------------------------------------------------------------- loss and batches
def test_identity_penalty():
    prediction = torch.tensor([[1.0, 2.0, 3.0, 4.0], [0.5, -1.0, 0.0, 1.0]])
    target = torch.zeros_like(prediction)
    scale = torch.tensor([2.0, 1.0, 1.0, 4.0])
    loss, l1, penalty = identity_penalty_loss(prediction, target, scale)
    kcal = prediction * scale                                          # barrier vs channel sum on the kcal scale
    expected_penalty = ((kcal[:, 3] - kcal[:, :3].sum(1)).abs() / scale[3]).mean()
    assert torch.isclose(l1, prediction.abs().mean())
    assert torch.isclose(penalty, expected_penalty)
    assert torch.isclose(loss, l1 + K.IDENTITY_PENALTY_WEIGHT * expected_penalty)
    consistent = torch.tensor([[1.0, 2.0, 3.0, 6.0]])
    assert float(identity_penalty_loss(consistent, consistent, torch.ones(4))[2]) == 0.0


def test_swap_and_epoch_batches():
    rows_a, rows_b = np.arange(10), np.arange(10, 20)
    swapped_a, swapped_b, sign = swap_pair_order(rows_a, rows_b, rows_a % 2 == 1)
    assert (swapped_a[1::2] == rows_b[1::2]).all() and (sign[1::2] == -1).all() and (sign[::2] == 1).all()
    targets = torch.arange(40, dtype=torch.float32).reshape(10, 4) + 1
    seen = []
    for batch_a, batch_b, batch_targets in iterate_training_batches(np.random.default_rng(0), rows_a, rows_b, targets, 4):
        for a, b, y in zip(batch_a, batch_b, batch_targets):
            pair = int(min(a, b))                                      # pair index = the a row
            assert abs(float(y[0])) == float(targets[pair, 0])
            assert (float(y[0]) < 0) == (a >= 10)                      # flipped pairs carry the negated target
            seen.append(pair)
    assert sorted(seen) == list(range(10))
    first = list(iterate_training_batches(np.random.default_rng(5), rows_a, rows_b, targets, 3))
    again = list(iterate_training_batches(np.random.default_rng(5), rows_a, rows_b, targets, 3))
    assert all((x[0] == y[0]).all() and torch.equal(x[2], y[2]) for x, y in zip(first, again))


# --------------------------------------------------------------------------- standardisation
def _toy_dataset(seed=0):
    rng = np.random.default_rng(seed)
    n_reactions, max_atoms = 8, 5
    n_atoms = rng.integers(2, max_atoms + 1, n_reactions)
    mask = np.arange(max_atoms)[None, :] < n_atoms[:, None]
    numeric = rng.normal(size=(n_reactions, max_atoms, len(K.NUMERIC_COLUMNS))) * mask[..., None]
    numeric[:, :, 3] = 7.0 * mask                                      # a constant column -> std set to 1
    reactions = SimpleNamespace(atom_mask=mask, numeric_features=numeric,
                                global_features=rng.normal(size=(n_reactions, 6)))
    return SimpleNamespace(reactions=reactions, rows_a=np.array([0, 1, 2, 3, 6]), rows_b=np.array([4, 4, 5, 5, 7]),
                           targets_kcal=rng.normal(loc=3.0, size=(5, 4)))


def test_standardisation_uses_training_pairs_only():
    dataset = _toy_dataset()
    train = np.array([0, 1, 2])                                        # reactions 0, 1, 2, 4, 5
    before = Standardizer.from_training_pairs(dataset, train)
    assert sorted(before.reaction_rows.tolist()) == [0, 1, 2, 4, 5]
    dataset.reactions.numeric_features[[3, 6, 7]] += 100.0             # change only reactions outside training
    dataset.reactions.global_features[[3, 6, 7]] -= 50.0
    dataset.targets_kcal[[3, 4]] *= 10.0
    after = Standardizer.from_training_pairs(dataset, train)
    assert before.sha256() == after.sha256()
    rows = before.reaction_rows
    atoms = dataset.reactions.numeric_features[rows][dataset.reactions.atom_mask[rows]]
    assert np.array_equal(before.numeric_mean, atoms.mean(0))
    assert before.numeric_std[3] == 1.0                                # constant column left unscaled
    assert np.array_equal(before.target_scale, dataset.targets_kcal[train].std(0))   # ddof 0


def test_saved_standardizer_roundtrip():
    dataset = _toy_dataset(1)
    s = Standardizer.from_training_pairs(dataset, np.array([0, 2, 4]))
    t = Standardizer.from_saved(s.as_saved())
    assert s.sha256() == t.sha256() and set(s.as_saved()) == {"mu", "sd", "gmu", "gsd", "sy"}


# --------------------------------------------------------------------------- channels and metrics
def test_channel_matrix():
    matrix = channel_matrix()
    assert matrix.shape == (10, 4) and set(np.unique(matrix)) == {0.0, 1.0}
    assert (matrix.sum(1) == 1).all()                                  # every label belongs to exactly one channel
    for j, target in enumerate(K.TARGETS):
        assert [K.LABEL_CHANNELS[i] for i in np.flatnonzero(matrix[:, j])] == list(K.CHANNEL_GROUPS[target])
    labels = np.random.default_rng(0).normal(size=(3, 10))
    channels = labels_to_channels(labels)
    column = {name: labels[:, i] for i, name in enumerate(K.LABEL_CHANNELS)}
    assert np.allclose(channels[:, 0], column["d1"] + column["d2"])
    assert np.allclose(channels[:, 1], column["elst"] + column["pauli"] + column["oi"] + column["disp"] + column["c_ghost"])
    assert np.allclose(channels[:, 2], column["cpcm"] + column["cds"])
    assert np.array_equal(channels[:, 3], column["barrier"])
    sys.path.insert(0, str(REPO / "analysis" / "ch3"))
    try:
        import ch3_common
    except Exception:                                                  # the original module is optional here
        return
    assert np.array_equal(matrix, np.array(ch3_common.M_ROWS, dtype=float))


def test_metric_helpers():
    accuracy, n = sign_accuracy([2.0, -3.0, 0.5, -1.0], [1.0, 1.0, -9.0, 0.0])
    assert n == 3 and abs(accuracy - 1 / 3) < 1e-12                   # yhat = 0 counts as wrong
    values = np.array([[2.0, 0.5, 0.1], [-1.0, 3.0, 0.0], [1.0, -2.0, -0.5]])
    signs = np.array([1.0, 1.0, -1.0])
    assert dominant_channel(values, signs).tolist() == ["strain", "inter", "inter"]
    assert cause_type(values, signs).tolist() == ["SI", "I", "I"]


# --------------------------------------------------------------------------- runner
def main() -> int:
    tests = [(name, f) for name, f in sorted(globals().items()) if name.startswith("test_") and callable(f)]
    results = {}
    for name, function in tests:
        try:
            function()
            results[name] = dict(ok=True)
        except Exception as error:                                     # report every failure, not only the first
            results[name] = dict(ok=False, error=f"{type(error).__name__}: {error}",
                                 traceback=traceback.format_exc(limit=4))
        print(f"{'PASS' if results[name]['ok'] else 'FAIL'}  {name}", flush=True)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from equivalence import write_part
    return write_part("E-6", dict(ok=all(r["ok"] for r in results.values()), n_tests=len(results),
                                  n_failed=sum(not r["ok"] for r in results.values()), tests=results, device="cpu",
                                  runner="built-in (env_d1fu2 has no pytest)", torch=torch.__version__))


if __name__ == "__main__":
    sys.exit(main())
