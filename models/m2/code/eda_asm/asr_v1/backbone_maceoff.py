"""Frozen MACE-OFF feature extractor (drop-in for NequIPFeatureExtractor).

Wraps an ASE ``MACECalculator`` built from a pretrained MACE-OFF23 model and
exposes per-atom INVARIANT (l=0) features via ``MACECalculator.get_descriptors``,
concatenated across all interaction layers.

This is the controlled-comparison backbone for ``ASR_Backbone_Comparison_Spec
v1.0``: every downstream component (head, CV folds, ensemble, learning curve)
stays identical; only the cached feature tensor changes.

The interface mirrors ``NequIPFeatureExtractor``:

    fe = MACEOFFFeatureExtractor(model_size="medium", device="cuda")
    feats = fe.extract(atoms)            # (n_atoms, feature_dim) cpu float32
    fe.feature_dim                       # int, set after construction
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import ase
import numpy as np
import torch


# Sizes accepted by mace.calculators.mace_off (these map to download URLs).
_VALID_SIZES = ("small", "medium", "large")


class MACEOFFFeatureExtractor:
    """Frozen MACE-OFF backbone returning per-atom invariant features.

    Parameters
    ----------
    model_size : one of "small" | "medium" | "large", or a local path/URL
        accepted by ``mace.calculators.mace_off``.
    device : torch device string. Defaults to CUDA if available.
    default_dtype : "float32" or "float64". MACE-OFF ships as float64; we
        downcast to float32 for speed since downstream heads run in float32.
    num_layers : -1 (all layers concatenated; default) or a positive int
        for the first ``num_layers`` layers only. ``get_descriptors``
        with ``num_layers=-1`` returns ``(n_atoms, num_interactions * d_inv)``.
    """

    def __init__(
        self,
        model_size: Union[str, Path] = "medium",
        device: Optional[str] = None,
        default_dtype: str = "float32",
        num_layers: int = -1,
    ) -> None:
        # mace_off is imported lazily so the rest of the package keeps
        # importing without mace-torch installed.
        from mace.calculators import mace_off

        self.model_size = str(model_size)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = str(device)
        self.default_dtype = str(default_dtype)
        self.num_layers = int(num_layers)

        # MACE prints download/cache lines and a float-dtype warning here.
        self._calc = mace_off(
            model=self.model_size if self.model_size in _VALID_SIZES else self.model_size,
            device=self.device,
            default_dtype=self.default_dtype,
            return_raw_model=False,
        )

        # Probe with H2 to record feature_dim. Use a separation < the model's
        # r_max so at least one edge exists. MACE-OFF uses r_max ~5.0 Å.
        probe = ase.Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
        probe_feats = self.extract(probe)
        self.feature_dim: int = int(probe_feats.shape[-1])

    @torch.enable_grad()
    def extract(self, atoms: ase.Atoms) -> torch.Tensor:
        """Return per-atom invariant features for one structure.

        Grad is enabled because MACE's forward pass uses autograd to derive
        forces; running under ``torch.no_grad`` raises
        "element 0 of tensors does not require grad" the same way NequIP does.
        Returned tensor is detached.

        Returns
        -------
        Tensor of shape ``(n_atoms, feature_dim)`` on CPU, float32.
        """
        desc = self._calc.get_descriptors(
            atoms, invariants_only=True, num_layers=self.num_layers,
        )
        # MACECalculator.get_descriptors returns np.ndarray when num_models==1,
        # else a list of np.ndarray (we use single-model committees only).
        if isinstance(desc, list):
            # Average over committee members (matches NequIP single-checkpoint behaviour).
            desc = np.mean(np.stack(desc, axis=0), axis=0)
        feats = torch.from_numpy(desc).float().contiguous()
        return feats

    def close(self) -> None:
        """Compat with NequIPFeatureExtractor; nothing to release."""
        return None

    def __del__(self) -> None:
        self.close()
