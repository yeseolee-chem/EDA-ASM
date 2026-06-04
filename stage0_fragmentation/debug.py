"""Debug helpers (Appendix B of the spec).

These are not part of the run-time API but make test authoring and triage
much easier.
"""
from __future__ import annotations

import numpy as np
from rdkit import Chem

from .be_matrix import _ensure_kekulized
from .types import FragmentationResult


def print_be_matrix(B: np.ndarray, mol: Chem.Mol) -> None:
    """Print a BE matrix with element + index labels for readability."""
    n = B.shape[0]
    labels = [
        f"{a.GetSymbol()}{a.GetIdx()}"
        for a in mol.GetAtoms()
    ]
    header = "        " + " ".join(f"{lab:>5s}" for lab in labels)
    print(header)
    for i in range(n):
        row = " ".join(f"{int(B[i, j]):>5d}" for j in range(n))
        print(f"{labels[i]:>6s}: {row}")


def print_delta_be(B_R: np.ndarray, B_P: np.ndarray, mol: Chem.Mol) -> None:
    """Print only the nonzero entries of ΔBE."""
    delta = B_P - B_R
    n = delta.shape[0]
    labels = [
        f"{a.GetSymbol()}{a.GetIdx()}"
        for a in mol.GetAtoms()
    ]
    print("ΔBE nonzero entries (i, j, value):")
    for i in range(n):
        for j in range(n):
            if delta[i, j] != 0:
                tag = "diag" if i == j else ""
                print(f"  {labels[i]:>6s} {labels[j]:>6s} {int(delta[i, j]):+d}  {tag}")


def visualize_fragmentation(
    result: FragmentationResult,
    mol_R: Chem.Mol,
    save_path: str,
) -> None:
    """Write a PNG with the two fragments coloured separately.

    Requires RDKit's drawing dependencies. Silently no-ops if drawing fails
    so that headless test environments don't break."""
    try:
        from rdkit.Chem.Draw import rdMolDraw2D
    except ImportError:
        return
    mol = _ensure_kekulized(mol_R)
    drawer = rdMolDraw2D.MolDraw2DCairo(420, 420)
    palette = [(0.12, 0.47, 0.71), (1.00, 0.50, 0.05), (0.17, 0.63, 0.17)]
    highlight_atoms: list[int] = []
    atom_colors: dict[int, tuple[float, float, float]] = {}
    for idx, frag in enumerate(result.fragments):
        col = palette[idx % len(palette)]
        for a in frag:
            highlight_atoms.append(a)
            atom_colors[a] = col
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=atom_colors,
    )
    drawer.FinishDrawing()
    with open(save_path, "wb") as f:
        f.write(drawer.GetDrawingText())
