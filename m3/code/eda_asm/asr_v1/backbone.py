"""Frozen NequIP feature extractor.

Loads the Halo8-pretrained NequIP checkpoint and exposes per-atom
invariant-scalar features (output of ``conv_to_output_hidden``, shape
``(n_atoms, feature_dim)``).

Implementation note: NequIP's top-level module wraps a ``GradientOutput``
that uses autograd to derive forces from energy. We therefore run the
forward pass with grads ENABLED and capture the per-atom features via a
forward hook on ``conv_to_output_hidden`` — then detach. This avoids the
``element 0 of tensors does not require grad`` error that occurs under
``torch.no_grad()``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import ase
import torch

from nequip.data import AtomicData, AtomicDataDict
from nequip.data.transforms import TypeMapper
from nequip.model import model_from_config
from nequip.utils import Config


class NequIPFeatureExtractor:
    """Frozen NequIP backbone returning per-atom invariant features.

    Parameters
    ----------
    config_path : path to the NequIP training-run ``config.yaml``.
    checkpoint_path : path to a NequIP state-dict checkpoint
        (``best_model.pth`` or ``ckptNN.pth``).
    device : torch device. Defaults to CUDA if available.
    """

    def __init__(
        self,
        config_path: str | Path,
        checkpoint_path: str | Path,
        device: Optional[str | torch.device] = None,
    ) -> None:
        self.cfg = Config.from_file(str(config_path))
        self.r_max = float(self.cfg["r_max"])
        self.chemical_symbols = list(self.cfg["chemical_symbols"])
        self._type_mapper = TypeMapper(chemical_symbols=self.chemical_symbols)

        model = model_from_config(self.cfg, initialize=False)
        state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        model.load_state_dict(state, strict=True)
        model.eval()

        # Resolve device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        model = model.to(self.device)
        self.model = model

        # Hook the layer that emits per-atom scalar hidden features
        # (output of conv_to_output_hidden; key = NODE_FEATURES_KEY).
        self._feat_layer = self.model.model.model.func.conv_to_output_hidden
        self._captured: dict[str, torch.Tensor] = {}

        def _hook(_module, _inp, out):
            # `out` is an AtomicDataDict-style dict
            self._captured["node_features"] = out[AtomicDataDict.NODE_FEATURES_KEY]

        self._hook_handle = self._feat_layer.register_forward_hook(_hook)

        # Probe the feature dim with H2 (needs ≥1 edge within r_max)
        probe_atoms = ase.Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
        with torch.enable_grad():
            self.extract(probe_atoms)
        self.feature_dim: int = int(self._captured["node_features"].shape[-1])

    def close(self) -> None:
        """Remove the forward hook (idempotent)."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def __del__(self) -> None:
        self.close()

    @torch.enable_grad()
    def extract(self, atoms: ase.Atoms) -> torch.Tensor:
        """Return per-atom invariant features for one structure.

        Returns
        -------
        Tensor of shape ``(n_atoms, feature_dim)`` on CPU, detached, float32.
        """
        ad = AtomicData.from_ase(atoms=atoms, r_max=self.r_max)
        ad = self._type_mapper(ad)
        data = AtomicData.to_AtomicDataDict(ad)
        data = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in data.items()}
        self._captured.clear()
        _ = self.model(data)
        feats = self._captured["node_features"].detach().to("cpu").float().contiguous()
        return feats
