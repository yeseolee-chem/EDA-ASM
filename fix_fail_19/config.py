"""Configuration constants for the 19-FAIL fix pipeline (per fix_fail_19_spec)."""

from __future__ import annotations

from dataclasses import dataclass


# ─── Exact reaction-id partition (4 + 7 + 8 = 19) ────────────────────────
GROUP_A_IDS: tuple[str, ...] = (
    # trajectory artifact — formed_bonds=0 or ΔE_rxn anomaly
    "Halogen_C4ClH5N2_rxn12962",
    "Halogen_BrC4H4NS_rxn10113",
    "Halogen_C4ClH5N2_rxn12941",
    "Halogen_C5FH5S_rxn16443",
)

GROUP_B_IDS: tuple[str, ...] = (
    # fragment spin reference mismatch — <S²> diagnosis + 4-coupling sweep
    "Halogen_BrC4H4NO_rxn10056",
    "Halogen_C4ClH4NS_rxn12917",
    "Halogen_C4ClH4NS_rxn12932",
    "Halogen_C4FH5N2O_rxn14222",
    "T1x_C7H12_rxn09748",
    "T1x_C5H6O_rxn06161",
    "T1x_C5H9NO_rxn08047",
)

GROUP_C_IDS: tuple[str, ...] = (
    # marginal residual 0.5–5 kcal/mol — accept + marginal_tag
    "Halogen_BrC5H5O_rxn11021",
    "Halogen_C4ClH5N2_rxn12960",
    "Halogen_C4FH4N_rxn13724",
    "T1x_C3H7NO_rxn01507",
    "T1x_C4H7NO_rxn03517",
    "T1x_C4H7NO_rxn03523",
    "T1x_C5H10O_rxn05246",
    "T1x_C5H6O_rxn06014",
)

assert len(GROUP_A_IDS) == 4
assert len(GROUP_B_IDS) == 7
assert len(GROUP_C_IDS) == 8
assert len(set(GROUP_A_IDS) | set(GROUP_B_IDS) | set(GROUP_C_IDS)) == 19, \
    "overlap or missing IDs in fix_fail_19 group lists"


@dataclass(frozen=True)
class Config:
    """All threshold + ADF settings for the fix_fail_19 pipeline."""

    # Shared ADF method
    n_workers: int = 2
    adf_functional: str = "BP86"
    adf_basis: str = "TZ2P"
    adf_dispersion: str = "D3BJ"
    adf_relativity: str = "ZORA_scalar"
    adf_integration: str = "Becke_Good"
    adf_scf_thr: float = 1e-6

    # Group A — endpoint re-optimization
    endpoint_grad_thr: float = 0.001         # Hartree/Å
    endpoint_max_step: int = 200
    endpoint_match_tol_kcal: float = 1.0     # |E_reopt - E_halo8| ≤ this → keep
    bond_cutoff_factor: float = 1.6          # ≤ this × Σ(covalent radii) → bonded

    # Group B — spin sweep
    s2_pure_singlet_max: float = 0.10        # <S²> below → closed shell
    s2_triplet_min: float = 1.90             # <S²> above → triplet
    s2_quintet_min: float = 5.90             # <S²> above → quintet
    coupling_candidates: tuple = (
        "closed_shell_singlet",
        "BS_singlet_yamaguchi",
        "triplet_ferro",
        "quintet_ferro",
    )
    spinsweep_winner_tol: float = 0.5        # > tau_warn but ≤ this → WARN

    # Group C — relaxed thresholds
    tau_pass: float = 0.1
    tau_warn: float = 0.5
    tau_fail_relaxed: float = 5.0            # Group C only; > this → FAIL
