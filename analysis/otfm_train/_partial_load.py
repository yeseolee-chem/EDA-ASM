"""Load only the ckpt tensors whose shape matches the model.

Used because our node_nfs=11 (7 atom categories + 3 pos + 1 charge) differs from
the pretrained checkpoint's node_nfs=9 (5 categories). Embedding + output
projections resize; the message-passing core must load intact (GATE-6a).
"""
from __future__ import annotations

from typing import Iterable


def partial_load(model, ckpt_state_dict) -> tuple[list[str], list[tuple]]:
    own = model.state_dict()
    loaded, skipped = [], []
    for k, v in ckpt_state_dict.items():
        if k in own and hasattr(v, "shape") and own[k].shape == v.shape:
            own[k] = v
            loaded.append(k)
        else:
            shape = tuple(v.shape) if hasattr(v, "shape") else None
            own_shape = tuple(own[k].shape) if k in own else None
            skipped.append((k, shape, own_shape))
    model.load_state_dict(own)

    print(f"[partial_load] loaded {len(loaded)}   skipped {len(skipped)}")
    if skipped:
        print("[partial_load] skipped tensors (must be embed/output-layer only):")
        for name, ckpt_shape, model_shape in skipped:
            print(f"  {name}: ckpt{ckpt_shape} vs model{model_shape}")

    return loaded, skipped


def assert_gate_6a(skipped: Iterable[tuple], allow_substrings: tuple = (
    "embed", "one_hot", "encoder_out", "decoder_out",
    "atom_type", "node_type", "output", "readout",
)) -> None:
    bad = [name for name, *_ in skipped
           if not any(s in name.lower() for s in allow_substrings)]
    if bad:
        raise RuntimeError(
            "GATE-6a FAIL: skipped tensors escape embed/output layers:\n  "
            + "\n  ".join(bad)
        )
    print("[GATE-6a] all skipped tensors confined to embed/output layers.")
