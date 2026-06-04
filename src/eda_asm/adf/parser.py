"""Parse ADF EDA-NOCV text output to extract the 5-channel ASR vector.

We rely on the standard ETS bonding-energy decomposition section in the text
output of an `ams` SinglePoint with fragments + ETSNOCV:

    Pauli Repulsion
      ...
      Total Pauli Repulsion:               <Ha>   <eV>   <kcal/mol>   <kJ/mol>

    Steric Interaction
      Pauli Repulsion (Delta E^Pauli):     <Ha>   <eV>   <kcal/mol>   <kJ/mol>
      Electrostatic Interaction:           <Ha>   <eV>   <kcal/mol>   <kJ/mol>
      ...

    Orbital Interactions
      ...
      Total Orbital Interactions:          <Ha>   <eV>   <kcal/mol>   <kJ/mol>

    Total Bonding Energy:                  <Ha>   <eV>   <kcal/mol>   <kJ/mol>

For fragment SPs (no fragments block), `Total Bonding Energy` is the system's
energy relative to a built-in atomic baseline. Strain is the *difference* of
two such SPs at the same composition, so the baseline cancels.

All values returned in kcal/mol.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


# Regex to match a labelled value row with 4 numeric columns.
# Group "kcal" is the kcal/mol column (3rd numeric).
_VAL_ROW = re.compile(
    r"^\s*(?P<label>.+?):\s+(?P<ha>-?\d+\.\d+)\s+(?P<ev>-?\d+\.\d+)\s+"
    r"(?P<kcal>-?\d+\.\d+)\s+(?P<kj>-?\d+\.\d+)\s*$"
)


@dataclass
class EDAChannels:
    pauli_kcal: float
    elstat_kcal: float
    orb_kcal: float
    disp_kcal: float
    total_bond_kcal: float

    def to_dict(self) -> dict:
        return {
            "E_Pauli": self.pauli_kcal,
            "V_elst": self.elstat_kcal,
            "E_orb": self.orb_kcal,
            "E_disp": self.disp_kcal,
            "E_int_total": self.total_bond_kcal,
        }


@dataclass
class ASRVector:
    """The 5-channel label for one reaction (all kcal/mol)."""
    pauli_kcal: float       # ΔE_Pauli (EDA channel)
    elstat_kcal: float      # ΔV_elst (EDA channel)
    orb_kcal: float         # ΔE_orb (EDA channel)
    disp_kcal: float        # ΔE_disp (EDA channel)
    strain_kcal: float      # ΔE_strain = ΔE_prep_A + ΔE_prep_B

    @property
    def Ea_reconstructed_kcal(self) -> float:
        """ΔE‡ = ΔE_strain + ΔE_int = strain + (Pauli + elst + orb + disp)."""
        return (
            self.strain_kcal + self.pauli_kcal + self.elstat_kcal
            + self.orb_kcal + self.disp_kcal
        )

    def to_dict(self) -> dict:
        return {
            "E_Pauli": self.pauli_kcal,
            "V_elst": self.elstat_kcal,
            "E_orb": self.orb_kcal,
            "E_disp": self.disp_kcal,
            "E_strain": self.strain_kcal,
            "Ea_reconstructed": self.Ea_reconstructed_kcal,
        }


def _find_first_value(text: str, label_pattern: str) -> float | None:
    """Return the kcal/mol value of the first line whose label matches the pattern."""
    rex = re.compile(label_pattern, re.IGNORECASE)
    for line in text.splitlines():
        m = _VAL_ROW.match(line)
        if not m:
            continue
        if rex.search(m["label"].strip()):
            return float(m["kcal"])
    return None


def parse_eda_out(path: Path) -> EDAChannels:
    """Parse `eda_TS.out` for the four EDA channels + total bond energy."""
    text = Path(path).read_text(errors="replace")

    pauli = _find_first_value(text, r"^Total\s+Pauli\s+Repulsion$")
    elstat = _find_first_value(text, r"^Electrostatic\s+Interaction$")
    orb = _find_first_value(text, r"^Total\s+Orbital\s+Interactions$")
    total = _find_first_value(text, r"^Total\s+Bonding\s+Energy$")
    # Dispersion: try several known label variants
    disp = (
        _find_first_value(text, r"^Dispersion\s+Energy$")
        or _find_first_value(text, r"^Total\s+Dispersion\s+Energy$")
        or _find_first_value(text, r"^Dispersion$")
    )
    if disp is None:
        # If dispersion was requested but not separately tabulated, fall back to 0.
        disp = 0.0

    missing = [k for k, v in {"pauli": pauli, "elstat": elstat,
                              "orb": orb, "total": total}.items() if v is None]
    if missing:
        raise ValueError(f"missing EDA channel(s) {missing} in {path}")

    return EDAChannels(
        pauli_kcal=pauli,
        elstat_kcal=elstat,
        orb_kcal=orb,
        disp_kcal=disp,
        total_bond_kcal=total,
    )


def parse_fragment_total_bond(path: Path) -> float:
    """Parse a fragment SP `<jobname>.out` for its Total Bonding Energy (kcal/mol)."""
    text = Path(path).read_text(errors="replace")
    val = _find_first_value(text, r"^Total\s+Bonding\s+Energy$")
    if val is None:
        raise ValueError(f"Total Bonding Energy not found in {path}")
    return val


def parse_eda_run(run_dir: Path) -> ASRVector:
    """Parse all 5 sub-job outputs in `run_dir` and return the 5-channel ASR vector.

    Expects files (created by the run_eda.sh template):
        fragA_at_TS.out
        fragB_at_TS.out
        eda_TS.out
        fragA_relaxed.out
        fragB_relaxed.out
    """
    run_dir = Path(run_dir)
    eda = parse_eda_out(run_dir / "eda_TS.out")
    e_fA_TS = parse_fragment_total_bond(run_dir / "fragA_at_TS.out")
    e_fA_rel = parse_fragment_total_bond(run_dir / "fragA_relaxed.out")
    e_fB_TS = parse_fragment_total_bond(run_dir / "fragB_at_TS.out")
    e_fB_rel = parse_fragment_total_bond(run_dir / "fragB_relaxed.out")

    dE_prep_A = e_fA_TS - e_fA_rel
    dE_prep_B = e_fB_TS - e_fB_rel
    strain = dE_prep_A + dE_prep_B

    return ASRVector(
        pauli_kcal=eda.pauli_kcal,
        elstat_kcal=eda.elstat_kcal,
        orb_kcal=eda.orb_kcal,
        disp_kcal=eda.disp_kcal,
        strain_kcal=strain,
    )
