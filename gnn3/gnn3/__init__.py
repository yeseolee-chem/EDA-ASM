"""GNN3: a graph neural network that predicts how a substituent changes the activation barrier of a 1,3-dipolar
cycloaddition, split into three channels (strain, interaction, solvation) plus the barrier itself.

Modules, in dependency order (no cycles):
    constants  every number of the method (name, value, unit, source)
    channels   10 labels -> 3 channels + barrier
    data       atom tokens, global features, pair labels, folds, standardisation
    model      GNN encoder + antisymmetric pair head
    training   loss, one fit, cross-validation unit, final model
    predict    saved networks -> ΔΔ with seed mean / spread / applicability flag
    metrics    the evaluation tables (torch-free)
    inputs     thin wrappers around the input pipeline (SMILES -> TS -> tokens)

See README.md for the science and the commands.
"""
__all__ = ["constants", "channels", "data", "model", "training", "predict", "metrics"]
__version__ = "1.0.0"
