"""Thin wrappers around the existing code that turns a reaction into GNN3 inputs (no computation of their own).

Order of the stages (each one calls the original script unchanged; see ``pipeline.STAGES``):

    ts-search   SMILES -> transition state (TS): autodE 1.4.5 + xtb 6.7.1 (GFN2, ALPB water), templates from the D1
                run; random (conformer search), so a rerun can find another TS. analysis/deploy/g2_deploy.py
    finalize    ORCA 6.1.1 XTB2 ALPB(water) frequency job at the TS, imaginary mode, forming-bond share; the
                "label-free" result folder <root>/g2lf_final/<rxn_id>/. analysis/deploy/finalize_lf.py
    metadata    per-reaction record (charges, fragments, forming bonds) for the feature code. analysis/deploy/meta_lf.py
    features    xTB single points of the TS and of both fragments -> the 182 global features (X182_lf.parquet).
                analysis/deploy/feat_lf.py
    tokens      per-atom xTB values -> one 29-wide token per TS atom (<root>/fu2_lf/tokens/<rxn_id>.npz).
                analysis/deploy/tokens_lf.py

``pipeline.run_stage`` runs one stage for a list of D1 job ids into a fresh root; ``scripts/prepare_inputs.py`` is the
command line and ``jobs/prepare_inputs.sh`` the sbatch chain. The two interpreter environments (D1 run env for
ts-search / finalize / metadata, venv5 for features / tokens) are the ones the original jobs used.
"""
from .pipeline import STAGES, Stage, root_environment, run_stage, select_reactions  # noqa: F401
